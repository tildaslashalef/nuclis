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
/// A frame is voiced when its RMS reaches this (about −34 dBFS).
pub const voiced_rms = 0.02;
/// After speech starts, this many silent frames in a row (about 1.5 s) end
/// the chunk; the silence is cut to `kept_silence` frames.
pub const silence_frames = 32;
pub const kept_silence = 6;
/// A take that has not spoken within this many frames, or ends first, is
/// retried with the next seed, at most `retries` times.
pub const start_frames = 48;
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
        if (self.encoder) |*e| e.deinit();
        if (self.decoder) |*d| d.deinit();
        if (self.gpu) |*g| g.deinit();
        if (self.codec) |*c| c.deinit();
        self.model.deinit();
        self.text.deinit();
        self.mapped.deinit(self.io);
        self.gpa.destroy(self);
    }

    /// The codes of a reference clip (44.1 kHz mono), owned by the caller.
    pub fn referenceCodes(self: *Voice, gpa: Allocator, samples: []const f32) ![]Codes {
        if (self.encoder == null) self.encoder = try codec_encoder.Encoder.init(self.gpa, self.io, self.mapped.document, self.mapped.view());
        const frames = try self.encoder.?.encode(gpa, samples);
        return frames;
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
            .history = .init(gpa, 1),
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
        /// Unvoiced frames since the last voiced one.
        silent: usize = 0,
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
        try self.engine.prefill(frames, hidden, &take.logits);
        try self.engine.towerPrefill(hidden[0 .. (rows.len - 1) * adapter.hidden], frames);
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
            const voiced = @sqrt(sum / samples_per_frame) >= voiced_rms;
            if (voiced) {
                take.spoke = true;
                take.silent = 0;
            } else if (take.spoke) {
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
    fn prefill(self: *Engine, frames: []const runtime.Frame, hidden: []f32, logits: []f32) !void {
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
