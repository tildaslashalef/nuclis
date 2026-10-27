//! Explicit full-model check of Nemotron 3.5 ASR's front end and encoder
//! (CPU reference) against the trace pinned from the authors' code
//! (scripts/speech-traces/nemotron_trace.py, tests/fixtures/nemotron-asr/).
//! Per clip: the log-mel frames of its WAV; then at each pinned lookahead
//! the whole-clip pass and the authors' chunks fed as a stream, both from
//! the pinned features, stage by stage at the pinned rows; then the clip's
//! samples streamed in 80 ms pieces, which adds the front end's error.
//! Needs the pulled checkpoint directory; FIXTURES defaults to
//! `tests/fixtures/nemotron-asr`, read at run time. Results:
//! docs/models/nemotron-asr.md § Front end and encoder against the trace.
const std = @import("std");
const inference = @import("inference");
const safetensors = inference.safetensors;
const audio = inference.audio;
const mel = audio.mel;
const nemotron = audio.nemotron_encoder;

/// Bounds relative to magnitude: the largest difference over the largest
/// reference value, and the difference's RMS over the reference's. About
/// five times the first run's worst (2026-10-27): the authors' F32 STFT
/// against the reference's F64 one, and F32 summation order in the encoder.
const Bound = struct { max: f64, rms: f64 };
const features_bound: Bound = .{ .max = 1e-4, .rms = 5e-6 };
const encoder_bound: Bound = .{ .max = 1e-5, .rms = 2e-5 };
const samples_bound: Bound = .{ .max = 1e-5, .rms = 2e-5 };

const Array = struct { name: []const u8, file: []const u8, shape: []const usize, dtype: []const u8, sha256: []const u8, rows: ?[]const usize = null };
const Clip = struct { name: []const u8, wav: []const u8, wav_sha256: []const u8, samples: usize, frames: usize, rows: usize, prompt_id: usize, arrays: []const Array };
const Index = struct { clips: []const Clip };

const Stats = struct {
    max_abs: f64 = 0,
    max_ref: f64 = 0,
    diff2: f64 = 0,
    ref2: f64 = 0,

    fn add(self: *Stats, got: []const f32, want: []const f32) void {
        for (got, want) |g, w| {
            const d = @as(f64, g) - w;
            self.max_abs = @max(self.max_abs, @abs(d));
            self.max_ref = @max(self.max_ref, @abs(@as(f64, w)));
            self.diff2 += d * d;
            self.ref2 += @as(f64, w) * w;
        }
    }

    fn scaled(self: Stats) f64 {
        return self.max_abs / self.max_ref;
    }

    fn rms(self: Stats) f64 {
        return @sqrt(self.diff2 / self.ref2);
    }
};

const Check = struct {
    gpa: std.mem.Allocator,
    arena: std.mem.Allocator,
    io: std.Io,
    fixtures: []const u8,
    failed: bool = false,

    fn read(self: *Check, array: Array) ![]const f32 {
        const path = try std.fs.path.join(self.arena, &.{ self.fixtures, array.file });
        const bytes = try std.Io.Dir.cwd().readFileAlloc(self.io, path, self.arena, .limited(64 << 20));
        try expectDigest(bytes, array.sha256, path);
        if (!std.mem.eql(u8, array.dtype, "f32") or bytes.len % 4 != 0) return error.UnsupportedFixture;
        var elements: usize = 1;
        for (array.shape) |d| elements *= d;
        if (elements * 4 != bytes.len) return error.UnsupportedFixture;
        const values = try self.arena.alloc(f32, elements);
        for (values, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little));
        return values;
    }

    fn report(self: *Check, label: []const u8, stats: Stats, bound: Bound) void {
        const bad = stats.scaled() > bound.max or stats.rms() > bound.rms;
        if (bad) self.failed = true;
        std.debug.print("  {s:<26} max |Δ| {e:.2} of max |ref| {e:.2} ({e:.2}, bound {e:.0})  rel RMS {e:.2} (bound {e:.0}){s}\n", .{ label, stats.max_abs, stats.max_ref, stats.scaled(), bound.max, stats.rms(), bound.rms, if (bad) "  OVER" else "" });
    }
};

fn expectDigest(bytes: []const u8, hex: []const u8, path: []const u8) !void {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    const got = std.fmt.bytesToHex(digest, .lower);
    if (!std.mem.eql(u8, &got, hex)) {
        std.debug.print("{s}: SHA-256 {s}, the index pins {s}\n", .{ path, &got, hex });
        return error.FixtureDigest;
    }
}

fn find(clip: Clip, name: []const u8) !Array {
    for (clip.arrays) |a| if (std.mem.eql(u8, a.name, name)) return a;
    return error.MissingFixture;
}

/// Collects every stage's rows as the encoder shows them.
const Recorder = struct {
    hidden: usize,
    rows: usize,
    stages: std.StringHashMapUnmanaged([]f32) = .empty,
    arena: std.mem.Allocator,

    fn init(arena: std.mem.Allocator, hidden: usize, rows: usize) Recorder {
        return .{ .hidden = hidden, .rows = rows, .arena = arena };
    }

    fn observer(self: *Recorder) nemotron.Observer {
        return .{ .context = self, .record = record };
    }

    fn record(context: *anyopaque, stage: nemotron.Stage, first: usize, rows: []const f32) void {
        const self: *Recorder = @ptrCast(@alignCast(context));
        var buffer: [32]u8 = undefined;
        const name = switch (stage) {
            .subsampling => "subsampling",
            .block => |i| std.fmt.bufPrint(&buffer, "block.{d}", .{i}) catch unreachable,
            .prompt => "prompt",
        };
        const slot = self.stages.getOrPut(self.arena, name) catch @panic("out of memory");
        if (!slot.found_existing) {
            slot.key_ptr.* = self.arena.dupe(u8, name) catch @panic("out of memory");
            slot.value_ptr.* = self.arena.alloc(f32, self.rows * self.hidden) catch @panic("out of memory");
            @memset(slot.value_ptr.*, std.math.nan(f32));
        }
        const end = @min(self.rows, first + rows.len / self.hidden);
        if (first < end) @memcpy(slot.value_ptr.*[first * self.hidden .. end * self.hidden], rows[0 .. (end - first) * self.hidden]);
    }

    /// The pinned rows of `stage`.
    fn pick(self: *Recorder, stage: []const u8, rows: []const usize) ![]const f32 {
        const all = self.stages.get(stage) orelse return error.StageNotTraced;
        const out = try self.arena.alloc(f32, rows.len * self.hidden);
        for (rows, 0..) |r, i| @memcpy(out[i * self.hidden ..][0..self.hidden], all[r * self.hidden ..][0..self.hidden]);
        return out;
    }
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const args = try init.minimal.args.toSlice(arena);
    if (args.len < 2 or args.len > 3) return error.ExpectedModelDirectory;
    const directory = args[1];
    const fixtures = if (args.len == 3) args[2] else "tests/fixtures/nemotron-asr";

    const config_text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(arena, &.{ directory, "config.json" }), arena, .limited(1 << 20));
    const config = try nemotron.parseConfig(gpa, config_text);
    var checkpoint = try safetensors.Checkpoint.open(gpa, io, try std.fs.path.join(arena, &.{ directory, "model.safetensors" }), .{});
    defer checkpoint.deinit(io);
    var binding = try nemotron.bind(gpa, &checkpoint, config);
    defer binding.deinit();
    var encoder = try nemotron.Encoder.init(gpa, io, &binding.weights);
    defer encoder.deinit();
    const bank = encoder.bank orelse return error.NoFrontEnd;

    const index_text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(arena, &.{ fixtures, "index.json" }), arena, .limited(1 << 20));
    const index = try std.json.parseFromSliceLeaky(Index, arena, index_text, .{ .ignore_unknown_fields = true });
    var check: Check = .{ .gpa = gpa, .arena = arena, .io = io, .fixtures = fixtures };
    const h = config.hidden;

    for (index.clips) |clip| {
        std.debug.print("{s}: {d} samples, {d} mel frames, {d} rows, prompt {d}\n", .{ clip.name, clip.samples, clip.frames, clip.rows, clip.prompt_id });
        // The front end, from the WAV the trace read.
        const wav_path = try std.fs.path.join(arena, &.{ fixtures, clip.wav });
        const wav = try std.Io.Dir.cwd().readFileAlloc(io, wav_path, arena, .limited(16 << 20));
        try expectDigest(wav, clip.wav_sha256, wav_path);
        var pcm = try audio.audio.decodeWav(gpa, wav, std.math.maxInt(u32));
        defer pcm.deinit(gpa);
        if (pcm.samples.len != clip.samples or mel.Spec.nemotron.frameCount(pcm.samples.len) != clip.frames) return error.ClipMismatch;
        const want_features = try check.read(try find(clip, "features"));
        // The authors' last frame is the zeroed one past the mask.
        if (want_features.len != (clip.frames + 1) * mel.filters) return error.ClipMismatch;
        for (want_features[clip.frames * mel.filters ..]) |v| if (v != 0) return error.ClipMismatch;
        const pinned = want_features[0 .. clip.frames * mel.filters];
        const features = try arena.alloc(f32, pinned.len);
        try bank.features(pcm.samples, features);
        var feature_stats: Stats = .{};
        feature_stats.add(features, pinned);
        check.report("features", feature_stats, features_bound);

        const lookaheads = [_]usize{ 0, 3 };
        for (lookaheads) |r| {
            var whole = Recorder.init(arena, h, clip.rows);
            const start = std.Io.Clock.awake.now(io);
            const rows = try encoder.encode(gpa, pinned, r, clip.prompt_id, whole.observer());
            gpa.free(rows);
            const ms = start.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds();
            std.debug.print(" r = {d}: whole clip in {d} ms\n", .{ r, ms });
            if (r == lookaheads[0]) try compare(&check, &whole, clip, "subsampling", "subsampling", "subsampling", encoder_bound);
            var label: [64]u8 = undefined;
            var name: [64]u8 = undefined;
            for ([_][]const u8{ "block.0", "block.11", "block.23", "prompt" }) |stage| {
                try compare(&check, &whole, clip, try std.fmt.bufPrint(&name, "r{d}/whole/{s}", .{ r, stage }), stage, try std.fmt.bufPrint(&label, "whole {s}", .{stage}), encoder_bound);
            }

            // The authors' chunks of the pinned features, fed as a stream.
            var chunked = Recorder.init(arena, h, clip.rows);
            var stream = try nemotron.Stream.init(gpa, &encoder, r, clip.prompt_id);
            defer stream.deinit();
            stream.observer = chunked.observer();
            var out: std.ArrayList(f32) = .empty;
            defer out.deinit(gpa);
            var at: usize = 0;
            var size = 1 + 8 * r;
            while (at < clip.frames) : (size = 8 * (r + 1)) {
                const n = @min(size, clip.frames - at);
                try stream.pushFeatures(pinned[at * mel.filters ..][0 .. n * mel.filters], &out);
                at += n;
            }
            try stream.finish(&out);
            if (out.items.len != clip.rows * h) return error.RowCount;
            for ([_][]const u8{ "block.0", "block.23", "prompt" }) |stage| {
                try compare(&check, &chunked, clip, try std.fmt.bufPrint(&name, "r{d}/stream/{s}", .{ r, stage }), stage, try std.fmt.bufPrint(&label, "chunked {s}", .{stage}), encoder_bound);
            }

            // The clip's samples in 80 ms pieces: the front end's error included.
            var heard = Recorder.init(arena, h, clip.rows);
            var live = try nemotron.Stream.init(gpa, &encoder, r, clip.prompt_id);
            defer live.deinit();
            live.observer = heard.observer();
            var spoken: std.ArrayList(f32) = .empty;
            defer spoken.deinit(gpa);
            at = 0;
            while (at < pcm.samples.len) : (at += 1280) try live.push(pcm.samples[at..@min(pcm.samples.len, at + 1280)], &spoken);
            try live.finish(&spoken);
            if (spoken.items.len != clip.rows * h) return error.RowCount;
            for ([_][]const u8{ "block.23", "prompt" }) |stage| {
                try compare(&check, &heard, clip, try std.fmt.bufPrint(&name, "r{d}/stream/{s}", .{ r, stage }), stage, try std.fmt.bufPrint(&label, "samples {s}", .{stage}), samples_bound);
            }
        }
    }
    if (check.failed) return error.OutsideTolerance;
    std.debug.print("Nemotron ASR encoder check passed.\n", .{});
}

fn compare(check: *Check, recorder: *Recorder, clip: Clip, fixture: []const u8, stage: []const u8, label: []const u8, bound: Bound) !void {
    const array = try find(clip, fixture);
    const want = try check.read(array);
    const got = try recorder.pick(stage, array.rows orelse return error.UnsupportedFixture);
    var stats: Stats = .{};
    stats.add(got, want);
    check.report(label, stats, bound);
}
