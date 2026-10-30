//! Drama 3 speech: text in, 44.1 kHz samples out. `Voice` opens the GGUF
//! once (backbone, tower, codec, and the encoder on first use); `Synthesizer`
//! splits the text into chunks (drama3_text.zig) and generates each in
//! Fish's loop (prompt with the references and the history, prefill, the
//! tower teacher-forced, then frame by frame), decoding as it goes. Beyond
//! Fish: a take that has not started speaking is held back and retried with
//! the next seed, and a long run of silent frames ends a chunk
//! (docs/models/drama3.md § Stalls, and what nuclis adds).
const std = @import("std");
const gguf = @import("../formats/gguf.zig");
const weights = @import("../runtime/weights.zig");
const text_mod = @import("drama3_text.zig");
const sampler_mod = @import("drama3_sampler.zig");
const runtime = @import("drama3_runtime.zig");
const adapter = @import("drama3.zig");
const codec = @import("../audio/fish_codec.zig");
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

pub const Voice = struct {
    gpa: Allocator,
    io: std.Io,
    mapped: weights.Mapped,
    text: text_mod.Text,
    model: runtime.Model,
    codec: codec.Runtime,
    /// Built on the first `referenceCodes`.
    encoder: ?codec_encoder.Encoder = null,
    defaults: Params,
    repeat: Repeat,

    pub fn open(gpa: Allocator, io: std.Io, path: []const u8) !*Voice {
        const self = try gpa.create(Voice);
        errdefer gpa.destroy(self);
        self.gpa = gpa;
        self.io = io;
        self.encoder = null;
        self.mapped = try weights.Mapped.open(gpa, io, path);
        errdefer self.mapped.deinit(io);
        const doc = &self.mapped.document;
        self.text = try text_mod.Text.init(gpa, doc.*, self.mapped.mapping.memory[0..@intCast(doc.directory_bytes)]);
        errdefer self.text.deinit();
        self.model = try runtime.Model.init(gpa, io, doc, self.mapped.view(), .{});
        errdefer self.model.deinit();
        self.codec = try codec.Runtime.init(gpa, io, doc.*, self.mapped.view());
        errdefer self.codec.deinit();
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
        self.codec.deinit();
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
    frames_ns: u64 = 0,
    codec_ns: u64 = 0,
    prompt_rows: usize = 0,
};

/// Generates `request` chunk by chunk; `read` hands out the samples.
/// Borrows the voice and the request's text and references.
pub const Synthesizer = struct {
    gpa: Allocator,
    voice: *Voice,
    request: Request,
    params: Params,
    chunks: [][]u8,
    next_chunk: usize = 0,
    history: text_mod.History,
    runtime: runtime.Runtime,
    state: runtime.State,
    sampler: sampler_mod.Sampler,
    stream: codec.Stream,
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
        var rt = try runtime.Runtime.init(gpa, &voice.model);
        errdefer rt.deinit();
        var state = try runtime.State.init(gpa, runtime.max_rows);
        errdefer state.deinit();
        var sampler = try sampler_mod.Sampler.init(gpa, request.seed, text_mod.codebook_size + 1);
        errdefer sampler.deinit(gpa);
        return .{
            .gpa = gpa,
            .voice = voice,
            .request = request,
            .params = request.params orelse voice.defaults,
            .chunks = chunks,
            .history = .init(gpa, 1),
            .runtime = rt,
            .state = state,
            .sampler = sampler,
            .stream = try codec.Stream.init(gpa, &voice.codec),
        };
    }

    pub fn deinit(self: *Synthesizer) void {
        for (self.chunks) |c| self.gpa.free(c);
        self.gpa.free(self.chunks);
        self.history.deinit();
        self.runtime.deinit();
        self.state.deinit();
        self.sampler.deinit(self.gpa);
        self.stream.deinit();
        self.ready.deinit(self.gpa);
        self.* = undefined;
    }

    /// Up to `out.len` next samples; 0 once the text is spoken.
    pub fn read(self: *Synthesizer, out: []f32) !usize {
        while (self.read_at == self.ready.items.len) {
            self.ready.clearRetainingCapacity();
            self.read_at = 0;
            if (self.next_chunk == self.chunks.len or self.frames >= self.request.max_frames) return 0;
            try self.speakChunk();
        }
        const n = @min(out.len, self.ready.items.len - self.read_at);
        @memcpy(out[0..n], self.ready.items[self.read_at..][0..n]);
        self.read_at += n;
        return n;
    }

    /// Generates the next chunk into `ready`, retrying a take that never
    /// started speaking.
    fn speakChunk(self: *Synthesizer) !void {
        const chunk = self.chunks[self.next_chunk];
        self.next_chunk += 1;
        var attempt: usize = 0;
        while (true) : (attempt += 1) {
            var take = try self.generate(chunk, self.request.seed +% self.next_chunk *% 7919 +% attempt);
            defer take.deinit(self.gpa);
            if (!take.spoke and attempt < retries) {
                self.retried += 1;
                continue;
            }
            self.stops[@intFromEnum(take.stop)] += 1;
            self.frames += take.codes.items.len;
            try self.ready.appendSlice(self.gpa, take.samples.items);
            if (take.codes.items.len > 0) try self.history.push(chunk, take.codes.items);
            return;
        }
    }

    const Take = struct {
        codes: std.ArrayList(Codes) = .empty,
        samples: std.ArrayList(f32) = .empty,
        spoke: bool = false,
        stop: Stop = .eos,

        fn deinit(self: *Take, gpa: Allocator) void {
            self.codes.deinit(gpa);
            self.samples.deinit(gpa);
        }
    };

    /// One take of `chunk` from a fresh state at `seed`.
    fn generate(self: *Synthesizer, chunk: []const u8, seed: u64) !Take {
        const gpa = self.gpa;
        const voice = self.voice;
        const io = voice.io;
        self.sampler.rng = .init(seed);
        self.state.reset();
        self.stream.reset();

        // The longest history whose prompt fits.
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

        const frames = try gpa.alloc(runtime.Frame, rows.len + 1);
        defer gpa.free(frames);
        for (rows, frames[0..rows.len]) |r, *f| f.* = toFrame(r.main, r.codes);
        const hidden = try gpa.alloc(f32, rows.len * adapter.hidden);
        defer gpa.free(hidden);
        var logits: [text_mod.codebook_size + 1]f32 = undefined;
        const prefill_start = std.Io.Clock.awake.now(io);
        try self.runtime.prefill(&self.state, frames[0..rows.len], hidden, &logits, null);
        try self.runtime.towerPrefill(&self.state, hidden[0 .. (rows.len - 1) * adapter.hidden], frames[0..rows.len], null);
        self.timing.prefill_ns += elapsed(io, prefill_start);

        var recent: std.ArrayList(u32) = .empty;
        defer recent.deinit(gpa);
        for (rows[rows.len -| self.voice.repeat.window..]) |r| try recent.append(gpa, r.main);

        var take: Take = .{};
        errdefer take.deinit(gpa);
        var codepoints: usize = 0;
        var it: std.unicode.Utf8Iterator = .{ .bytes = chunk, .i = 0 };
        while (it.nextCodepoint()) |_| codepoints += 1;
        const budget = @min(@min(chunk_frames, budget_floor_frames + frames_per_char * codepoints), self.request.max_frames -| self.frames);
        var input = frames[rows.len - 1];
        var silent: usize = 0;
        var decoded: usize = 0;
        var pick_context: PickContext = .{ .sampler = &self.sampler, .params = self.params };
        const pick: runtime.Pick = .{ .context = &pick_context, .pick = PickContext.pick };
        take.stop = .length;
        while (take.codes.items.len < budget) {
            const frame_start = std.Io.Clock.awake.now(io);
            const column = try self.sampler.mainColumn(&logits, self.params, self.voice.repeat, recent.items, voice.text.semantic_start);
            if (column == 0) {
                take.stop = .eos;
                break;
            }
            const main = voice.text.semantic_start + column - 1;
            const frame = try self.runtime.step(&self.state, self.runtime.h, input, main, pick, null);
            try recent.append(gpa, main);
            var codes: Codes = undefined;
            for (&codes, frame[1..]) |*c, v| c.* = @intCast(v);
            try take.codes.append(gpa, codes);
            input = frame;
            self.timing.frames_ns += elapsed(io, frame_start);

            // Decode what is pending: one frame at a time until the take has
            // spoken (so a silent start can be judged), then in pushes.
            const pending = take.codes.items.len - decoded;
            if (!take.spoke or pending >= push_frames) {
                if (try self.decodePending(&take, &decoded, &silent)) {
                    take.stop = .silence;
                    break;
                }
            }
            if (!take.spoke and take.codes.items.len >= start_frames) break;
            try self.runtime.decode(&self.state, frame, &logits, null);
        }
        if (take.stop != .silence) _ = try self.decodePending(&take, &decoded, &silent);
        return take;
    }

    /// Decodes the take's undecoded frames into its samples, tracking speech:
    /// returns true when a run of silence ends the take (the silence is cut).
    fn decodePending(self: *Synthesizer, take: *Take, decoded: *usize, silent: *usize) !bool {
        const n = take.codes.items.len - decoded.*;
        if (n == 0) return false;
        const io = self.voice.io;
        const start = std.Io.Clock.awake.now(io);
        const at = take.samples.items.len;
        try take.samples.resize(self.gpa, at + n * samples_per_frame);
        try self.stream.push(take.codes.items[decoded.*..][0..n], take.samples.items[at..]);
        decoded.* += n;
        self.timing.codec_ns += elapsed(io, start);
        for (0..n) |i| {
            const frame = take.samples.items[at + i * samples_per_frame ..][0..samples_per_frame];
            var sum: f64 = 0;
            for (frame) |v| sum += @as(f64, v) * v;
            const voiced = @sqrt(sum / samples_per_frame) >= voiced_rms;
            if (voiced) {
                take.spoke = true;
                silent.* = 0;
            } else if (take.spoke) {
                silent.* += 1;
                if (silent.* == silence_frames) {
                    const end = at + (i + 1) * samples_per_frame;
                    const cut = (silence_frames - kept_silence) * samples_per_frame;
                    take.samples.shrinkRetainingCapacity(end - cut);
                    take.codes.shrinkRetainingCapacity(take.codes.items.len - (n - i - 1) - (silence_frames - kept_silence));
                    return true;
                }
            }
        }
        return false;
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
