//! Drama 3 speech: text in, 44.1 kHz samples out. `Voice` opens the GGUF
//! once (backbone, tower, codec, and the encoder on first use); `Synthesizer`
//! splits the text into chunks (drama3_text.zig) and generates each in
//! Fish's loop (prompt with the references and the history, prefill, the
//! tower teacher-forced, then frame by frame), decoding as it goes. Beyond
//! Fish: a take that has not started speaking is held back and retried with
//! the next seed, and a long run of silent frames ends a chunk
//! (docs/models/drama3.md § Stalls, and what nuclis adds). The backbone, the
//! tower, and the codec decoder run on the CPU references or on Metal
//! (`Options.backend`).
const std = @import("std");
const gguf = @import("../formats/gguf.zig");
const weights = @import("../runtime/weights.zig");
const text_mod = @import("drama3_text.zig");
const sampler_mod = @import("drama3_sampler.zig");
const runtime = @import("drama3_runtime.zig");
const session = @import("../runtime/session.zig");
const metal_plan = @import("drama3_metal.zig");
const metal = @import("../backends/metal/root.zig");
const adapter = @import("drama3.zig");
const codec = @import("../audio/fish_codec.zig");
const codec_metal = @import("../audio/fish_codec_metal.zig");
const codec_encoder = @import("../audio/fish_codec_encoder.zig");

const Allocator = std.mem.Allocator;
pub const Codes = text_mod.Codes;
pub const Params = sampler_mod.Params;
pub const Repeat = sampler_mod.Repeat;
pub const Chunking = text_mod.Chunking;
pub const sample_rate = codec.sample_rate;
pub const samples_per_frame = codec.samples_per_frame;

/// Fish's per-chunk frame cap (about 27.9 s).
pub const chunk_frames = 600;
/// A chunk's budget also follows its length: this many frames, plus
/// `frames_per_char` for each code point (about twice a brisk speaking rate).
pub const budget_floor_frames = 64;
pub const frames_per_char = 3;
/// A take has spoken once its last `level_window` frames' RMS (the
/// codec's first frame left out) peaks at `spoken_rms` (−40 dBFS) and
/// `onset_ratio` times their quietest, so steady noise from the start is
/// not speech. After that a frame is silent under `silence_ratio` of the
/// take's loudest, clamped to 0.003–0.02, so a whisper is not
/// (docs/models/drama3.md § Stalls).
pub const level_window = 16;
pub const spoken_rms = 0.01;
pub const onset_ratio = 3.0;
pub const silence_ratio = 0.1;
pub const silence_rms_min = 0.003;
pub const silence_rms_max = 0.02;
/// The quietest level `onset_ratio` multiplies.
const level_floor = 0.002;
/// After speech starts, this many silent frames in a row (about 1.5 s) end
/// the chunk; the run is cut to `kept_silence` frames.
pub const silence_frames = 32;
pub const kept_silence = 6;
/// A take that has not spoken within this many frames, or ends first, is
/// retried with the next seed, at most `retries` times.
pub const start_frames = 32;
pub const retries = 2;
/// Frames decoded per codec push once a chunk is speaking.
pub const push_frames = 4;

/// Where the backbone, the tower, and the codec decoder run.
pub const Backend = enum { cpu, metal };
pub const Options = struct { backend: Backend = .cpu };

pub const Voice = struct {
    gpa: Allocator,
    io: std.Io,
    mapped: weights.Mapped,
    text: text_mod.Text,
    model: runtime.Model,
    backend: Backend,
    /// The device the synthesizers' plans run on, with `.metal`.
    gpu: ?metal.Backend = null,
    /// The codec decoder: the CPU reference, or (with `.metal`) its copy on
    /// the device, the CPU one released once copied.
    codec: ?codec.Runtime = null,
    decoder: ?codec_metal.Decoder = null,
    /// Built on the first `referenceCodes`.
    encoder: ?codec_encoder.Encoder = null,
    defaults: Params,
    repeat: Repeat,
    /// The state after each recent prompt prefix (the system turn and a
    /// voice's references), restored instead of prefilled again; at most
    /// `max_prefixes`, the least recently used dropped. The voice's
    /// synthesizers run one at a time.
    prefixes: std.ArrayList(Prefix) = .empty,
    prefix_clock: u64 = 0,

    /// `openWith` on the CPU.
    pub fn open(gpa: Allocator, io: std.Io, path: []const u8) !*Voice {
        return openWith(gpa, io, path, .{});
    }

    pub fn openWith(gpa: Allocator, io: std.Io, path: []const u8, options: Options) !*Voice {
        const self = try gpa.create(Voice);
        errdefer gpa.destroy(self);
        self.gpa = gpa;
        self.io = io;
        self.encoder = null;
        self.backend = options.backend;
        self.gpu = null;
        self.codec = null;
        self.decoder = null;
        self.prefixes = .empty;
        self.prefix_clock = 0;
        self.mapped = try weights.Mapped.open(gpa, io, path);
        errdefer self.mapped.deinit(io);
        const doc = &self.mapped.document;
        self.text = try text_mod.Text.init(gpa, doc.*, self.mapped.mapping.memory[0..@intCast(doc.directory_bytes)]);
        errdefer self.text.deinit();
        self.model = try runtime.Model.init(gpa, io, doc, self.mapped.view(), .{});
        errdefer self.model.deinit();
        self.codec = try codec.Runtime.init(gpa, io, doc.*, self.mapped.view());
        errdefer if (self.codec) |*c| c.deinit();
        if (options.backend == .metal) {
            var message: [512]u8 = @splat(0);
            self.gpu = try metal.Backend.init(gpa, &message);
        }
        errdefer if (self.gpu) |*g| g.deinit();
        if (self.gpu) |*g| {
            self.decoder = try codec_metal.Decoder.init(gpa, g, &self.codec.?);
            self.codec.?.deinit();
            self.codec = null;
        }
        errdefer if (self.decoder) |*d| d.deinit();
        self.defaults = .{
            .temperature = float(doc.*, "drama3.sampling.temperature") orelse 1.0,
            .top_p = float(doc.*, "drama3.sampling.top_p") orelse 0.9,
            .top_k = @intCast(@min(unsigned(doc.*, "drama3.sampling.top_k") orelse 30, 4097)),
        };
        self.repeat = .{
            .window = @intCast(@min(unsigned(doc.*, "drama3.repeat.window") orelse 16, 64)),
            .params = .{
                .temperature = float(doc.*, "drama3.repeat.temperature") orelse 1.0,
                .top_p = float(doc.*, "drama3.repeat.top_p") orelse 0.95,
                .top_k = @intCast(@min(unsigned(doc.*, "drama3.repeat.top_k") orelse 30, 4097)),
            },
        };
        return self;
    }

    pub fn close(self: *Voice) void {
        for (self.prefixes.items) |*p| p.deinit(self.gpa);
        self.prefixes.deinit(self.gpa);
        if (self.encoder) |*e| e.deinit();
        if (self.decoder) |*d| d.deinit();
        if (self.gpu) |*g| g.deinit();
        if (self.codec) |*c| c.deinit();
        self.model.deinit();
        self.text.deinit();
        self.mapped.deinit(self.io);
        self.gpa.destroy(self);
    }

    /// The state kept after `rows`, marked used; null when none is.
    fn findPrefix(self: *Voice, rows: []const text_mod.Row) ?*const Prefix {
        for (self.prefixes.items) |*p| if (std.mem.eql(u8, std.mem.sliceAsBytes(p.rows), std.mem.sliceAsBytes(rows))) {
            self.prefix_clock += 1;
            p.used = self.prefix_clock;
            return p;
        };
        return null;
    }

    /// Keeps `state` (its backbone after `rows`, its tower one row behind)
    /// and `last`, the backbone's hidden at the last of `rows`.
    fn keepPrefix(self: *Voice, rows: []const text_mod.Row, state: *const runtime.State, last: []const f32) !void {
        var kept: Prefix = .{ .rows = try self.gpa.dupe(text_mod.Row, rows), .backbone = undefined, .tower = undefined, .last = &.{}, .used = 0 };
        errdefer self.gpa.free(kept.rows);
        kept.backbone = try state.backbone.snapshot(self.gpa);
        errdefer kept.backbone.deinit();
        kept.tower = try state.tower.snapshot(self.gpa);
        errdefer kept.tower.deinit();
        kept.last = try self.gpa.dupe(f32, last);
        errdefer self.gpa.free(kept.last);
        self.prefix_clock += 1;
        kept.used = self.prefix_clock;
        if (self.prefixes.items.len == max_prefixes) {
            var oldest: usize = 0;
            for (self.prefixes.items, 0..) |p, i| if (p.used < self.prefixes.items[oldest].used) {
                oldest = i;
            };
            var dropped = self.prefixes.swapRemove(oldest);
            dropped.deinit(self.gpa);
        }
        try self.prefixes.append(self.gpa, kept);
    }

    /// The codes of a reference clip (44.1 kHz mono), owned by the caller.
    pub fn referenceCodes(self: *Voice, gpa: Allocator, samples: []const f32) ![]Codes {
        if (self.encoder == null) self.encoder = try codec_encoder.Encoder.init(self.gpa, self.io, self.mapped.document, self.mapped.view());
        const frames = try self.encoder.?.encode(gpa, samples);
        return frames;
    }
};

/// Prompt prefixes a voice keeps the state after (each about 90 MB with a
/// 12 s reference).
pub const max_prefixes = 4;

const Prefix = struct {
    rows: []text_mod.Row,
    backbone: session.Snapshot,
    tower: session.Snapshot,
    last: []f32,
    used: u64,

    fn deinit(self: *Prefix, gpa: Allocator) void {
        gpa.free(self.rows);
        self.backbone.deinit();
        self.tower.deinit();
        gpa.free(self.last);
    }
};

fn float(doc: gguf.Document, key: []const u8) ?f32 {
    return switch (doc.get(key) orelse return null) {
        .float => |f| @floatCast(f),
        else => null,
    };
}

fn unsigned(doc: gguf.Document, key: []const u8) ?u64 {
    return switch (doc.get(key) orelse return null) {
        .unsigned => |n| n,
        else => null,
    };
}

pub const Request = struct {
    text: []const u8,
    /// Each reference's transcript (speaker-tagged as Fish requires) and codes.
    references: []const text_mod.Reference = &.{},
    /// A sentence spoken before this text in the same voice (its text and
    /// codes), taken as the first chunk's history as a chunk of this text
    /// would be.
    context: ?text_mod.Turn = null,
    params: ?Params = null,
    seed: u64 = 0,
    chunking: Chunking = .{},
    /// Frames over the whole text (Fish's default 2,048).
    max_frames: usize = 2048,
};

/// Why a chunk ended.
pub const Stop = enum { eos, silence, length };

pub const Timing = struct {
    prefill_ns: u64 = 0,
    /// Takes whose prompt prefix was restored from the voice, and those
    /// that prefilled it (and kept it).
    prefix_hits: usize = 0,
    prefix_misses: usize = 0,
    prefix_rows: usize = 0,
    /// The frames' sampling, tower steps, and backbone decodes.
    frames_ns: u64 = 0,
    codec_ns: u64 = 0,
    prompt_rows: usize = 0,
};

/// Generates `request` chunk by chunk; `read` hands out the samples a few
/// frames at a time. Borrows the voice and the request's text and
/// references; must not move once `read` was called.
pub const Synthesizer = struct {
    gpa: Allocator,
    voice: *Voice,
    request: Request,
    params: Params,
    chunks: [][]u8,
    next_chunk: usize = 0,
    /// Every prompt's first rows (`Text.prefix`), prefilled apart so the
    /// voice can keep their state.
    prefix: []text_mod.Row,
    history: text_mod.History,
    engine: Engine,
    sampler: sampler_mod.Sampler,
    stream: Stream,
    /// The take being generated, if one is.
    take: ?Take = null,
    /// Frames one `read` generates at most before handing out what it has.
    frames_per_read: usize = 8,
    /// Samples ready to hand out, and how many of them were already read.
    ready: std.ArrayList(f32) = .empty,
    read_at: usize = 0,
    frames: usize = 0,
    retried: usize = 0,
    stops: [3]usize = @splat(0),
    timing: Timing = .{},

    pub fn init(gpa: Allocator, voice: *Voice, request: Request) !Synthesizer {
        try voice.text.validate(request.text);
        for (request.references) |r| try voice.text.validate(r.transcript);
        const chunks = try text_mod.split(gpa, request.text, request.chunking);
        errdefer {
            for (chunks) |c| gpa.free(c);
            gpa.free(chunks);
        }
        if (chunks.len == 0) return error.EmptyText;
        if (request.context) |c| try voice.text.validate(c.text);
        const prefix = try voice.text.prefix(gpa, request.references);
        errdefer gpa.free(prefix);
        var history: text_mod.History = .init(gpa, 1);
        errdefer history.deinit();
        if (request.context) |c| if (c.codes.len > 0) try history.push(c.text, c.codes);
        var engine = try Engine.init(gpa, voice);
        errdefer engine.deinit();
        var sampler = try sampler_mod.Sampler.init(gpa, request.seed, text_mod.codebook_size + 1);
        errdefer sampler.deinit(gpa);
        var stream = try Stream.init(gpa, voice);
        errdefer stream.deinit();
        return .{
            .gpa = gpa,
            .voice = voice,
            .request = request,
            .params = request.params orelse voice.defaults,
            .chunks = chunks,
            .prefix = prefix,
            .history = history,
            .engine = engine,
            .sampler = sampler,
            .stream = stream,
        };
    }

    pub fn deinit(self: *Synthesizer) void {
        for (self.chunks) |c| self.gpa.free(c);
        self.gpa.free(self.chunks);
        if (self.take) |*t| t.deinit(self.gpa);
        self.history.deinit();
        self.gpa.free(self.prefix);
        self.engine.deinit();
        self.sampler.deinit(self.gpa);
        self.stream.deinit();
        self.ready.deinit(self.gpa);
        self.* = undefined;
    }

    /// Up to `out.len` next samples; 0 once the text is spoken. A call
    /// generates at most `frames_per_read` frames, more only while a take
    /// has not started speaking (its silence is held back).
    pub fn read(self: *Synthesizer, out: []f32) !usize {
        while (self.read_at == self.ready.items.len) {
            self.ready.clearRetainingCapacity();
            self.read_at = 0;
            if (self.take == null) {
                if (self.next_chunk == self.chunks.len or self.frames >= self.request.max_frames) return 0;
                self.next_chunk += 1;
                try self.begin(0);
            }
            try self.advance();
        }
        const n = @min(out.len, self.ready.items.len - self.read_at);
        @memcpy(out[0..n], self.ready.items[self.read_at..][0..n]);
        self.read_at += n;
        return n;
    }

    /// The chunk the take speaks.
    /// The last sentence spoken (or the request's context when none was),
    /// borrowed until the synthesizer ends: the next request's `context`.
    pub fn lastTurn(self: *const Synthesizer) ?text_mod.Turn {
        const turns = self.history.turns.items;
        if (turns.len == 0) return null;
        const t = turns[turns.len - 1];
        return .{ .text = t.text, .codes = t.codes };
    }

    fn chunkText(self: *const Synthesizer) []const u8 {
        return self.chunks[self.next_chunk - 1];
    }

    /// A chunk's generation: its codes and decoded samples (those before
    /// `released` already handed to `ready`), and where the loop stands.
    const Take = struct {
        attempt: usize,
        budget: usize,
        codes: std.ArrayList(Codes) = .empty,
        samples: std.ArrayList(f32) = .empty,
        released: usize = 0,
        decoded: usize = 0,
        /// Silent frames since the last sounding one.
        silent: usize = 0,
        /// The loudest frame's RMS so far, and every decoded frame's.
        peak: f32 = 0,
        levels: std.ArrayList(f32) = .empty,
        spoke: bool = false,
        done: bool = false,
        stop: Stop = .length,
        /// The main tokens the repeat check looks back over.
        recent: std.ArrayList(u32) = .empty,
        input: runtime.Frame,
        logits: [text_mod.codebook_size + 1]f32 = undefined,

        fn deinit(self: *Take, gpa: Allocator) void {
            self.codes.deinit(gpa);
            self.samples.deinit(gpa);
            self.levels.deinit(gpa);
            self.recent.deinit(gpa);
        }
    };

    /// Starts a take of the current chunk from a fresh state: the longest
    /// history whose prompt fits, prefilled, the tower teacher-forced.
    fn begin(self: *Synthesizer, attempt: usize) !void {
        const gpa = self.gpa;
        const voice = self.voice;
        const io = voice.io;
        const chunk = self.chunkText();
        self.sampler.rng = .init(self.request.seed +% self.next_chunk *% 7919 +% attempt);
        self.engine.reset();
        self.stream.reset();

        var buffer: [2]text_mod.Turn = undefined;
        var candidate: usize = 0;
        const rows = while (self.history.candidate(candidate, &buffer)) |history| : (candidate += 1) {
            break voice.text.prompt(gpa, self.request.references, history, chunk) catch |err| switch (err) {
                error.PromptTooLong => continue,
                else => return err,
            };
        } else return error.PromptTooLong;
        defer gpa.free(rows);
        self.timing.prompt_rows = rows.len;

        var codepoints: usize = 0;
        var it: std.unicode.Utf8Iterator = .{ .bytes = chunk, .i = 0 };
        while (it.nextCodepoint()) |_| codepoints += 1;
        self.take = .{
            .attempt = attempt,
            .budget = @min(@min(chunk_frames, budget_floor_frames + frames_per_char * codepoints), self.request.max_frames -| self.frames),
            .input = undefined,
        };
        const take = &self.take.?;
        errdefer {
            take.deinit(gpa);
            self.take = null;
        }
        const frames = try gpa.alloc(runtime.Frame, rows.len);
        defer gpa.free(frames);
        for (rows, frames) |r, *f| f.* = toFrame(r.main, r.codes);
        const hidden = try gpa.alloc(f32, rows.len * adapter.hidden);
        defer gpa.free(hidden);
        const prefill_start = std.Io.Clock.awake.now(io);
        // The prefix and the rest are always prefilled apart, so a restored
        // prefix gives the same take bit for bit.
        const h = adapter.hidden;
        const n = self.prefix.len;
        std.debug.assert(n >= 2 and rows.len > n and std.mem.eql(u8, std.mem.sliceAsBytes(rows[0..n]), std.mem.sliceAsBytes(self.prefix)));
        self.timing.prefix_rows = n;
        if (voice.findPrefix(self.prefix)) |kept| {
            try self.engine.restore(kept);
            @memcpy(hidden[(n - 1) * h ..][0..h], kept.last);
            self.timing.prefix_hits += 1;
        } else {
            try self.engine.prefill(frames[0..n], hidden[0 .. n * h], null);
            try self.engine.towerPrefill(hidden[0 .. (n - 1) * h], frames[0..n]);
            try voice.keepPrefix(self.prefix, self.engine.state(), hidden[(n - 1) * h ..][0..h]);
            self.timing.prefix_misses += 1;
        }
        try self.engine.prefill(frames[n..], hidden[n * h ..], &take.logits);
        try self.engine.towerPrefill(hidden[(n - 1) * h .. (rows.len - 1) * h], frames[n - 1 ..]);
        self.timing.prefill_ns += elapsed(io, prefill_start);
        take.input = frames[rows.len - 1];
        for (rows[rows.len -| voice.repeat.window..]) |r| try take.recent.append(gpa, r.main);
    }

    /// Generates the take's next frames, decodes them, hands out what is
    /// settled, and ends the take when it is done (retrying one that never
    /// started speaking).
    fn advance(self: *Synthesizer) !void {
        const take = &self.take.?;
        var made: usize = 0;
        while (!take.done and (made < self.frames_per_read or !take.spoke)) : (made += 1) try self.frame(take);
        if (take.done and take.stop != .silence) _ = try self.decodePending(take);
        if (take.spoke or take.done) {
            // A trailing silent run past what a cut keeps may yet be cut.
            const held = if (take.done or take.silent <= kept_silence) 0 else (take.silent - kept_silence) * samples_per_frame;
            const end = take.samples.items.len - @min(held, take.samples.items.len);
            if (end > take.released) {
                if (take.spoke or take.attempt == retries) try self.ready.appendSlice(self.gpa, take.samples.items[take.released..end]);
                take.released = end;
            }
        }
        if (!take.done) return;
        var finished = take.*;
        self.take = null;
        defer finished.deinit(self.gpa);
        if (!finished.spoke and finished.attempt < retries) {
            self.retried += 1;
            return self.begin(finished.attempt + 1);
        }
        self.stops[@backingInt(finished.stop)] += 1;
        self.frames += finished.codes.items.len;
        if (finished.codes.items.len > 0) try self.history.push(self.chunkText(), finished.codes.items);
    }

    /// One frame: the main token (or the end), the tower's codes, the
    /// decode as speech is judged, and the backbone's step for the next.
    fn frame(self: *Synthesizer, take: *Take) !void {
        const voice = self.voice;
        const io = voice.io;
        if (take.codes.items.len >= take.budget) {
            take.done = true;
            return;
        }
        const frame_start = std.Io.Clock.awake.now(io);
        const column = try self.sampler.mainColumn(&take.logits, self.params, voice.repeat, take.recent.items, voice.text.semantic_start);
        if (column == 0) {
            take.stop = .eos;
            take.done = true;
            return;
        }
        const main = voice.text.semantic_start + column - 1;
        var pick_context: PickContext = .{ .sampler = &self.sampler, .params = self.params };
        const pick: runtime.Pick = .{ .context = &pick_context, .pick = PickContext.pick };
        const out = try self.engine.step(take.input, main, pick);
        try take.recent.append(self.gpa, main);
        var codes: Codes = undefined;
        for (&codes, out[1..]) |*c, v| c.* = @intCast(v);
        try take.codes.append(self.gpa, codes);
        take.input = out;
        self.timing.frames_ns += elapsed(io, frame_start);

        // One frame at a time until the take has spoken (so a silent start
        // can be judged), then in pushes.
        if (!take.spoke or take.codes.items.len - take.decoded >= push_frames) {
            if (try self.decodePending(take)) {
                take.stop = .silence;
                take.done = true;
                return;
            }
        }
        if (!take.spoke and take.codes.items.len >= start_frames) {
            take.done = true;
            return;
        }
        if (take.codes.items.len < take.budget) {
            const decode_start = std.Io.Clock.awake.now(io);
            try self.engine.decode(out, &take.logits);
            self.timing.frames_ns += elapsed(io, decode_start);
        }
    }

    /// Decodes the take's undecoded frames into its samples, tracking speech:
    /// returns true when a run of silence ends the take (the silence is cut).
    fn decodePending(self: *Synthesizer, take: *Take) !bool {
        const n = take.codes.items.len - take.decoded;
        if (n == 0) return false;
        const io = self.voice.io;
        const start = std.Io.Clock.awake.now(io);
        const at = take.samples.items.len;
        try take.samples.resize(self.gpa, at + n * samples_per_frame);
        try self.stream.push(take.codes.items[take.decoded..][0..n], take.samples.items[at..]);
        take.decoded += n;
        self.timing.codec_ns += elapsed(io, start);
        for (0..n) |i| {
            const samples = take.samples.items[at + i * samples_per_frame ..][0..samples_per_frame];
            var sum: f64 = 0;
            for (samples) |v| sum += @as(f64, v) * v;
            const rms: f32 = @floatCast(@sqrt(sum / samples_per_frame));
            try take.levels.append(self.gpa, rms);
            const levels = take.levels.items;
            if (!take.spoke) {
                const window = levels[@max(1, levels.len -| level_window)..];
                if (window.len >= 3) {
                    const range = std.mem.minMax(f32, window);
                    if (range[1] >= spoken_rms and range[1] >= onset_ratio * @max(range[0], level_floor)) {
                        take.spoke = true;
                        take.peak = range[1];
                    }
                }
                continue;
            }
            take.peak = @max(take.peak, rms);
            if (rms >= std.math.clamp(silence_ratio * take.peak, silence_rms_min, silence_rms_max)) {
                take.silent = 0;
            } else {
                take.silent += 1;
                if (take.silent == silence_frames) {
                    const end = at + (i + 1) * samples_per_frame;
                    const cut = (silence_frames - kept_silence) * samples_per_frame;
                    take.samples.shrinkRetainingCapacity(end - cut);
                    take.codes.shrinkRetainingCapacity(take.codes.items.len - (n - i - 1) - (silence_frames - kept_silence));
                    take.silent = kept_silence;
                    return true;
                }
            }
        }
        return false;
    }
};

/// The backbone and tower a synthesizer drives: the CPU reference with its
/// state, or the Metal plan, which owns its own.
const Engine = union(Backend) {
    cpu: struct { runtime: runtime.Runtime, state: runtime.State },
    metal: metal_plan.Plan,

    fn init(gpa: Allocator, voice: *Voice) !Engine {
        switch (voice.backend) {
            .cpu => {
                var rt = try runtime.Runtime.init(gpa, &voice.model);
                errdefer rt.deinit();
                return .{ .cpu = .{ .runtime = rt, .state = try runtime.State.init(gpa, runtime.max_rows) } };
            },
            .metal => return .{ .metal = try metal_plan.Plan.init(gpa, &voice.gpu.?, &voice.model, runtime.max_rows, metal_plan.default_chunk) },
        }
    }
    fn deinit(self: *Engine) void {
        switch (self.*) {
            .cpu => |*c| {
                c.runtime.deinit();
                c.state.deinit();
            },
            .metal => |*p| p.deinit(),
        }
    }
    fn reset(self: *Engine) void {
        switch (self.*) {
            .cpu => |*c| c.state.reset(),
            .metal => |*p| p.reset(),
        }
    }
    fn state(self: *Engine) *const runtime.State {
        return switch (self.*) {
            .cpu => |*c| &c.state,
            .metal => |*p| &p.state,
        };
    }
    /// Takes a kept prefix's state; the engine must be fresh (`reset`).
    fn restore(self: *Engine, kept: *const Prefix) !void {
        const s = switch (self.*) {
            .cpu => |*c| &c.state,
            .metal => |*p| &p.state,
        };
        try s.backbone.restore(&kept.backbone);
        try s.tower.restore(&kept.tower);
    }
    fn prefill(self: *Engine, frames: []const runtime.Frame, hidden: []f32, logits: ?[]f32) !void {
        switch (self.*) {
            .cpu => |*c| try c.runtime.prefill(&c.state, frames, hidden, logits, null),
            .metal => |*p| try p.prefill(frames, hidden, logits),
        }
    }
    fn towerPrefill(self: *Engine, hidden: []const f32, frames: []const runtime.Frame) !void {
        switch (self.*) {
            .cpu => |*c| try c.runtime.towerPrefill(&c.state, hidden, frames, null),
            .metal => |*p| try p.towerPrefill(hidden, frames),
        }
    }
    /// The tower at the backbone's last row, on its own hidden.
    fn step(self: *Engine, input: runtime.Frame, main: u32, pick: runtime.Pick) !runtime.Frame {
        return switch (self.*) {
            .cpu => |*c| c.runtime.step(&c.state, c.runtime.h, input, main, pick, null),
            .metal => |*p| p.step(p.h, input, main, pick),
        };
    }
    fn decode(self: *Engine, frame: runtime.Frame, logits: []f32) !void {
        switch (self.*) {
            .cpu => |*c| try c.runtime.decode(&c.state, frame, logits, null),
            .metal => |*p| try p.decode(frame, logits),
        }
    }
};

/// The codec stream a synthesizer decodes through, on the voice's backend.
const Stream = union(Backend) {
    cpu: codec.Stream,
    metal: codec_metal.Stream,

    fn init(gpa: Allocator, voice: *Voice) !Stream {
        return switch (voice.backend) {
            .cpu => .{ .cpu = try codec.Stream.init(gpa, &voice.codec.?) },
            .metal => .{ .metal = try codec_metal.Stream.init(&voice.decoder.?) },
        };
    }
    fn deinit(self: *Stream) void {
        switch (self.*) {
            inline else => |*s| s.deinit(),
        }
    }
    fn reset(self: *Stream) void {
        switch (self.*) {
            inline else => |*s| s.reset(),
        }
    }
    fn push(self: *Stream, codes: []const Codes, out: []f32) !void {
        switch (self.*) {
            inline else => |*s| try s.push(codes, out),
        }
    }
};

const PickContext = struct {
    sampler: *sampler_mod.Sampler,
    params: Params,

    fn pick(context: *anyopaque, codebook: usize, logits: []const f32) anyerror!u32 {
        _ = codebook;
        const self: *PickContext = @ptrCast(@alignCast(context));
        return self.sampler.draw(logits, self.params);
    }
};

fn toFrame(main: u32, codes: Codes) runtime.Frame {
    var f: runtime.Frame = undefined;
    f[0] = main;
    for (f[1..], codes) |*a, c| a.* = c;
    return f;
}

fn elapsed(io: std.Io, start: std.Io.Timestamp) u64 {
    return @intCast(start.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);
}

test {
    std.testing.refAllDecls(Voice);
    std.testing.refAllDecls(Synthesizer);
}
