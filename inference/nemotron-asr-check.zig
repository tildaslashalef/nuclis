//! Explicit full-model check of Nemotron 3.5 ASR's front end and encoder
//! (CPU reference) against the trace pinned from the authors' code
//! (scripts/speech-traces/nemotron_trace.py, tests/fixtures/nemotron-asr/).
//! Per clip: the log-mel frames of its WAV; then at each pinned lookahead
//! the whole-clip pass and the authors' chunks fed as a stream, both from
//! the pinned features, stage by stage at the pinned rows; then the clip's
//! samples streamed in 80 ms pieces, which adds the front end's error.
//! `--transducer` checks the rest instead: the projector rows, every greedy
//! step's row, argmax and top logits, and the final tokens and text, from
//! the pinned features as one pass and from the samples in 80 ms pieces.
//! Needs the pulled checkpoint directory; FIXTURES defaults to
//! `tests/fixtures/nemotron-asr`, read at run time. Results:
//! docs/models/nemotron-asr.md § Front end and encoder against the trace,
//! § Transducer against the trace, and word error rate.
const std = @import("std");
const inference = @import("inference");
const safetensors = inference.safetensors;
const audio = inference.audio;
const mel = audio.mel;
const nemotron = audio.nemotron_encoder;
const rnnt = audio.nemotron_rnnt;
const asr = audio.nemotron_asr;

/// Bounds relative to magnitude: the largest difference over the largest
/// reference value, and the difference's RMS over the reference's. About
/// five times the first run's worst (2026-10-27): the authors' F32 STFT
/// against the reference's F64 one, and F32 summation order in the encoder.
const Bound = struct { max: f64, rms: f64 };
const features_bound: Bound = .{ .max = 1e-4, .rms = 5e-6 };
const encoder_bound: Bound = .{ .max = 1e-5, .rms = 2e-5 };
const samples_bound: Bound = .{ .max = 1e-5, .rms = 2e-5 };

const Array = struct { name: []const u8, file: []const u8, shape: []const usize, dtype: []const u8, sha256: []const u8, rows: ?[]const usize = null };
const Transcript = struct { ids: []const u32, rows: []const u32, steps: usize, text: []const u8 };
const Clip = struct {
    name: []const u8,
    wav: []const u8,
    wav_sha256: []const u8,
    samples: usize,
    frames: usize,
    rows: usize,
    prompt_id: usize,
    arrays: []const Array,
    transcripts: ?std.json.ArrayHashMap(Transcript) = null,
};
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
        return self.readAs(f32, "f32", array);
    }

    /// An array of 4-byte little-endian `T` whose dtype is `dtype`.
    fn readAs(self: *Check, comptime T: type, dtype: []const u8, array: Array) ![]const T {
        const path = try std.fs.path.join(self.arena, &.{ self.fixtures, array.file });
        const bytes = try std.Io.Dir.cwd().readFileAlloc(self.io, path, self.arena, .limited(64 << 20));
        try expectDigest(bytes, array.sha256, path);
        if (!std.mem.eql(u8, array.dtype, dtype) or bytes.len % 4 != 0) return error.UnsupportedFixture;
        var elements: usize = 1;
        for (array.shape) |d| elements *= d;
        if (elements * 4 != bytes.len) return error.UnsupportedFixture;
        const values = try self.arena.alloc(T, elements);
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
    var positional: std.ArrayList([]const u8) = .empty;
    var transducer = false;
    for (args[1..]) |a| {
        if (std.mem.eql(u8, a, "--transducer")) transducer = true else try positional.append(arena, a);
    }
    if (positional.items.len < 1 or positional.items.len > 2) return error.ExpectedModelDirectory;
    const directory = positional.items[0];
    const fixtures = if (positional.items.len == 2) positional.items[1] else "tests/fixtures/nemotron-asr";
    if (transducer) return checkTransducer(gpa, io, arena, directory, fixtures);

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

// ---- --transducer: the projector, the greedy steps, the tokens and text ----

/// Relative bounds as above, about five times the first run's worst
/// (2026-10-27): projector 3.2e-6 and 2.4e-6, the logits at the trace's top
/// five ids 1.4e-5 and 6.8e-6 (the 80 ms pieces, the front end's error
/// included), the margins (top one minus top two) 1.7e-3 apart.
const projector_bound: Bound = .{ .max = 2e-5, .rms = 2e-5 };
const logit_bound: Bound = .{ .max = 1e-4, .rms = 5e-5 };
const margin_bound: f64 = 1e-2;
/// The language tags `tokenizer.json` marks special (nemotron-asr.md § Tokenizer).
const language_tags = 39;

/// Every greedy step as the decoder shows it: the row, the token, a copy of the logits.
const Steps = struct {
    arena: std.mem.Allocator,
    rows: std.ArrayList(usize) = .empty,
    tokens: std.ArrayList(u32) = .empty,
    logits: std.ArrayList([]const f32) = .empty,

    fn observer(self: *Steps) rnnt.Observer {
        return .{ .context = self, .record = record };
    }

    fn record(context: *anyopaque, step: rnnt.Step) void {
        const self: *Steps = @ptrCast(@alignCast(context));
        self.rows.append(self.arena, step.row) catch @panic("out of memory");
        self.tokens.append(self.arena, step.token) catch @panic("out of memory");
        self.logits.append(self.arena, self.arena.dupe(f32, step.logits) catch @panic("out of memory")) catch @panic("out of memory");
    }
};

fn checkTransducer(gpa: std.mem.Allocator, io: std.Io, arena: std.mem.Allocator, directory: []const u8, fixtures: []const u8) !void {
    const model = try asr.Model.open(gpa, io, directory);
    defer model.close();
    const weights = &model.transducer_binding.weights;
    const c = weights.config;
    var check: Check = .{ .gpa = gpa, .arena = arena, .io = io, .fixtures = fixtures };

    var tags: usize = 0;
    for (model.tags) |t| tags += @intFromBool(t != null);
    std.debug.print("piece table: {d} ids, {d} language tags, blank {d} special: {}\n", .{ model.table.count(), tags, c.blank, model.table.isSpecial(c.blank) });
    if (tags != language_tags or !model.table.isSpecial(c.blank)) return error.TokenizerMismatch;

    const index_text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(arena, &.{ fixtures, "index.json" }), arena, .limited(1 << 20));
    const index = try std.json.parseFromSliceLeaky(Index, arena, index_text, .{ .ignore_unknown_fields = true });
    const h = model.encoder.weights.config.hidden;

    for (index.clips, 0..) |clip, n| {
        std.debug.print("{s}: {d} rows, prompt {d}\n", .{ clip.name, clip.rows, clip.prompt_id });
        if (n == 0) {
            var decoder = try rnnt.Decoder.init(gpa, io, weights);
            defer decoder.deinit();
            var stats: Stats = .{};
            stats.add(decoder.g, try check.read(try find(clip, "decoder_start")));
            check.report("prediction of the start", stats, projector_bound);
        }
        const transcripts = clip.transcripts orelse return error.MissingFixture;
        const pinned = (try check.read(try find(clip, "features")))[0 .. clip.frames * mel.filters];
        const wav = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(arena, &.{ fixtures, clip.wav }), arena, .limited(16 << 20));
        try expectDigest(wav, clip.wav_sha256, clip.wav);
        var pcm = try audio.audio.decodeWav(gpa, wav, std.math.maxInt(u32));
        defer pcm.deinit(gpa);

        for ([_]usize{ 0, 3 }) |r| {
            var name: [64]u8 = undefined;
            // The whole clip from the pinned features, one push to the decoder.
            var steps: Steps = .{ .arena = arena };
            var start = std.Io.Clock.awake.now(io);
            const rows = try model.encoder.encode(gpa, pinned, r, clip.prompt_id, null);
            defer gpa.free(rows);
            var decoder = try rnnt.Decoder.init(gpa, io, weights);
            defer decoder.deinit();
            decoder.observer = steps.observer();
            var tokens: std.ArrayList(rnnt.Token) = .empty;
            defer tokens.deinit(gpa);
            try decoder.push(rows, &tokens);
            std.debug.print(" r = {d}, whole: {d} ms\n", .{ r, start.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds() });
            const whole = try std.fmt.bufPrint(&name, "r{d}/whole", .{r});
            try compareProjector(&check, weights, clip, whole, rows, h);
            try compareSteps(&check, clip, whole, &steps, tokens.items, model, transcripts.map.get(whole) orelse return error.MissingFixture);

            // The samples in 80 ms pieces through a transcriber.
            var heard = Recorder.init(arena, h, clip.rows);
            var live: Steps = .{ .arena = arena };
            var transcriber = try asr.Transcriber.init(gpa, model, r, clip.prompt_id);
            defer transcriber.deinit();
            transcriber.stream.observer = heard.observer();
            transcriber.decoder.observer = live.observer();
            start = std.Io.Clock.awake.now(io);
            var at: usize = 0;
            while (at < pcm.samples.len) : (at += 1280) _ = try transcriber.push(pcm.samples[at..@min(pcm.samples.len, at + 1280)]);
            _ = try transcriber.finish();
            std.debug.print(" r = {d}, 80 ms pieces: {d} ms\n", .{ r, start.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds() });
            const streamed = try std.fmt.bufPrint(&name, "r{d}/stream", .{r});
            const all = heard.stages.get("prompt") orelse return error.StageNotTraced;
            try compareProjector(&check, weights, clip, streamed, all, h);
            const want = transcripts.map.get(streamed) orelse return error.MissingFixture;
            try compareSteps(&check, clip, streamed, &live, transcriber.tokens.items, model, want);
            var text: std.Io.Writer.Allocating = .init(arena);
            try transcriber.writeText(&text.writer);
            if (!std.mem.eql(u8, text.written(), want.text)) {
                std.debug.print("  transcriber text differs: {s}\n", .{text.written()});
                check.failed = true;
            }
        }
    }
    if (check.failed) return error.OutsideTolerance;
    std.debug.print("Nemotron ASR transducer check passed.\n", .{});
}

/// `encoder_projector` of the encoder's `rows` at the pinned rows, against the trace.
fn compareProjector(check: *Check, weights: *const rnnt.Weights, clip: Clip, run: []const u8, rows: []const f32, h: usize) !void {
    const c = weights.config;
    var name: [64]u8 = undefined;
    const array = try find(clip, try std.fmt.bufPrint(&name, "{s}/projector", .{run}));
    const want = try check.read(array);
    const picked = array.rows orelse return error.UnsupportedFixture;
    const x = try check.arena.alloc(f32, picked.len * h);
    for (picked, 0..) |r, i| @memcpy(x[i * h ..][0..h], rows[r * h ..][0..h]);
    const got = try check.arena.alloc(f32, picked.len * c.hidden);
    try inference.cpu.dense.matmul(check.io, .{ .rows = picked.len, .inner = h, .outputs = c.hidden, .x = x, .w = weights.projector.weight, .bias = weights.projector.bias }, got);
    var stats: Stats = .{};
    stats.add(got, want);
    check.report(try std.fmt.bufPrint(&name, "{s} projector", .{run}), stats, projector_bound);
}

/// Step for step against the trace: the row, the argmax (exactly), the
/// logits at the trace's top five ids, the margins; then the tokens and text.
fn compareSteps(check: *Check, clip: Clip, run: []const u8, steps: *const Steps, tokens: []const rnnt.Token, model: *const asr.Model, want: Transcript) !void {
    var name: [64]u8 = undefined;
    const step_rows = try check.readAs(i32, "i32", try find(clip, try std.fmt.bufPrint(&name, "{s}/step_rows", .{run})));
    const top_ids = try check.readAs(i32, "i32", try find(clip, try std.fmt.bufPrint(&name, "{s}/top_ids", .{run})));
    const top_logits = try check.read(try find(clip, try std.fmt.bufPrint(&name, "{s}/top_logits", .{run})));
    const top = top_ids.len / step_rows.len;
    if (steps.rows.items.len != step_rows.len or step_rows.len != want.steps) {
        std.debug.print("  {s}: {d} steps, the trace {d}  OVER\n", .{ run, steps.rows.items.len, step_rows.len });
        check.failed = true;
        return;
    }
    var stats: Stats = .{};
    var mismatched: usize = 0;
    var ours_smallest: f64 = std.math.inf(f64);
    var theirs_smallest: f64 = std.math.inf(f64);
    var margin_gap: f64 = 0;
    for (0..step_rows.len) |s| {
        const logits = steps.logits.items[s];
        if (steps.rows.items[s] != step_rows[s] or steps.tokens.items[s] != top_ids[s * top]) mismatched += 1;
        for (0..top) |k| stats.add(&.{logits[@intCast(top_ids[s * top + k])]}, &.{top_logits[s * top + k]});
        // Our runner-up over the whole row, against theirs.
        const best = rnnt.argmax(logits);
        var second: f32 = -std.math.inf(f32);
        for (logits, 0..) |v, i| if (i != best) {
            second = @max(second, v);
        };
        const ours = @as(f64, logits[best]) - second;
        const theirs = @as(f64, top_logits[s * top]) - top_logits[s * top + 1];
        ours_smallest = @min(ours_smallest, ours);
        theirs_smallest = @min(theirs_smallest, theirs);
        margin_gap = @max(margin_gap, @abs(ours - theirs));
    }
    var label: [64]u8 = undefined;
    check.report(try std.fmt.bufPrint(&label, "{s} top-{d} logits", .{ run, top }), stats, logit_bound);
    // Margins agree with the trace's, and the smallest exceeds their
    // disagreement: no argmax matched by luck.
    const sane = margin_gap <= margin_bound and ours_smallest > margin_gap;
    std.debug.print("  {s}: {d} steps, {d} differ; smallest margin {d:.4} (trace {d:.4}), margins within {e:.2} (bound {e:.0}){s}\n", .{ run, step_rows.len, mismatched, ours_smallest, theirs_smallest, margin_gap, margin_bound, if (mismatched > 0 or !sane) "  OVER" else "" });
    if (mismatched > 0 or !sane) check.failed = true;

    var same = tokens.len == want.ids.len;
    if (same) for (tokens, want.ids, want.rows) |t, id, row| {
        if (t.id != id or t.row != row) same = false;
    };
    const ids = try check.arena.alloc(u32, tokens.len);
    for (tokens, ids) |t, *id| id.* = t.id;
    var text: std.Io.Writer.Allocating = .init(check.arena);
    try model.table.decode(ids, &text.writer);
    const text_same = std.mem.eql(u8, text.written(), want.text);
    std.debug.print("  {s}: {d} tokens{s}, text \"{s}\"{s}\n", .{ run, tokens.len, if (same) " as traced" else " DIFFER", text.written(), if (text_same) "" else "  DIFFERS" });
    if (!same or !text_same) check.failed = true;
}
