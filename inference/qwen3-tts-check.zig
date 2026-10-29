//! Explicit full-model check of Qwen3-TTS against the authors' bring-up
//! trace (`scripts/speech-traces/qwen3_tts_trace.py`, pinned under
//! `tests/fixtures/qwen3-tts/`). `codec`: each case's codes through the
//! codec decoder's `decode`, every stage the fixture holds compared with
//! its rows, then the samples; then the same frames pushed through a
//! `Stream` in pieces of 5, which must give `decode`'s samples bit for bit.
//! `talker`: each case of MODEL_DIR's size, replaying the trace's codes:
//! the token ids and the prompt's rows exactly, then the talker's input
//! rows, its layers and final norm, the group-0 logits, and the
//! predictor's logits against the trace. `talker-metal` replays the same
//! codes through the Metal plan (`--kv f32|f16|both`, the talker cache;
//! the F16 cache's bound is `--max-relative-rms-f16`), and
//! `codec-metal` decodes each case's codes on Metal against the CPU decode
//! and the trace, then in pieces of 1, 3, and 7 frames, which must equal the
//! Metal decode bit for bit. `clone` (a Base's MODEL_DIR): the reference
//! case's clip, read from its WAV at 24 kHz, through the speaker encoder
//! and the codec encoder, each stage the trace pins and the embedding
//! compared, and the codes exactly. The bounds are arguments, so the gate
//! manifest holds them.
//!
//!   qwen3-tts-check MODEL_DIR codec|codec-metal [--max-relative-rms X] [--max-absolute X] [--fixtures DIR]
//!   qwen3-tts-check MODEL_DIR talker|talker-metal [--max-relative-rms X] [--kv f32|f16|both] [--max-relative-rms-f16 X] [--fixtures DIR]
//!   qwen3-tts-check MODEL_DIR clone [--max-relative-rms X] [--fixtures DIR]
const std = @import("std");
const inference = @import("inference");
const codec = inference.audio.qwen3_codec;
const ecapa = inference.audio.ecapa;
const codec_encoder = inference.audio.qwen3_codec_encoder;
const safetensors = inference.safetensors;
const tts = inference.models.qwen3_tts;
const speech = inference.models.qwen3_tts_runtime;

const tts_metal = inference.models.qwen3_tts_metal;
const codec_metal = inference.audio.qwen3_codec_metal;
const metal = inference.metal;

const usage = "usage: qwen3-tts-check MODEL_DIR codec|talker|codec-metal|talker-metal|clone [--max-relative-rms X] [--max-absolute X] [--kv f32|f16|both] [--max-relative-rms-f16 X] [--fixtures DIR]\n";

const Stats = struct { max_abs: f64 = 0, max_ref: f64 = 0, diff2: f64 = 0, ref2: f64 = 0 };

fn compare(got: []const f32, want: []const f32) Stats {
    var s: Stats = .{};
    for (got, want) |g, w| {
        const d = @as(f64, g) - w;
        s.max_abs = @max(s.max_abs, @abs(d));
        s.max_ref = @max(s.max_ref, @abs(@as(f64, w)));
        s.diff2 += d * d;
        s.ref2 += @as(f64, w) * w;
    }
    return s;
}

/// Every stage's rows over the whole decode, appended push by push.
const Recorder = struct {
    gpa: std.mem.Allocator,
    rows: [@typeInfo(codec.Stage).@"enum".field_names.len]std.ArrayList(f32) = @splat(.empty),
    failed: bool = false,

    fn record(context: *anyopaque, stage: codec.Stage, rows: []const f32) void {
        const self: *Recorder = @ptrCast(@alignCast(context));
        self.rows[@backingInt(stage)].appendSlice(self.gpa, rows) catch {
            self.failed = true;
        };
    }

    fn deinit(self: *Recorder) void {
        for (&self.rows) |*r| r.deinit(self.gpa);
    }
};

/// Reads a fixture file, checking its SHA-256 against the index.
fn readFixture(arena: std.mem.Allocator, io: std.Io, dir: []const u8, entry: std.json.Value) ![]const u8 {
    const object = entry.object;
    const path = try std.fs.path.join(arena, &.{ dir, object.get("file").?.string });
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, arena, .limited(64 << 20));
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    const hex = std.fmt.bytesToHex(digest, .lower);
    if (!std.mem.eql(u8, &hex, object.get("sha256").?.string)) {
        std.debug.print("{s}: SHA-256 differs from the index\n", .{path});
        return error.FixtureMismatch;
    }
    return bytes;
}

fn floats(arena: std.mem.Allocator, bytes: []const u8) ![]f32 {
    const out = try arena.alloc(f32, bytes.len / 4);
    for (out, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little));
    return out;
}

const Bounds = struct { relative_rms: f64, absolute: f64 };

fn checkCodec(gpa: std.mem.Allocator, arena: std.mem.Allocator, io: std.Io, model: []const u8, fixtures: []const u8, bounds: Bounds) !bool {
    const index_path = try std.fs.path.join(arena, &.{ fixtures, "index.json" });
    const index_text = try std.Io.Dir.cwd().readFileAlloc(io, index_path, arena, .limited(1 << 20));
    const index = try std.json.parseFromSliceLeaky(std.json.Value, arena, index_text, .{});

    const codec_path = try std.fs.path.join(arena, &.{ model, "speech_tokenizer", "model.safetensors" });
    const load_start = std.Io.Clock.awake.now(io);
    var checkpoint = try safetensors.Checkpoint.open(gpa, io, codec_path, .{});
    defer checkpoint.deinit(io);
    var diagnostic: codec.Diagnostic = .{};
    var binding = codec.bind(gpa, &checkpoint, .{}, &diagnostic) catch |err| {
        std.debug.print("bind: {s} ({s})\n", .{ @errorName(err), diagnostic.name() });
        return err;
    };
    defer binding.deinit(gpa);
    var runtime = try codec.Runtime.init(gpa, io, &binding);
    defer runtime.deinit();
    std.debug.print("Codec decoder bound and loaded in {d} ms: {d} tensors, {d} MB.\n", .{ load_start.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds(), binding.tensors, binding.bytes / 1_000_000 });
    // The pinned codec's digest, from the pull's sidecar when it has one.
    const sidecar_path = try std.fs.path.join(arena, &.{ model, "speech_tokenizer", "model.safetensors.nuclis.json" });
    const sidecar: ?std.json.Value = if (std.Io.Dir.cwd().readFileAlloc(io, sidecar_path, arena, .limited(1 << 16))) |text|
        try std.json.parseFromSliceLeaky(std.json.Value, arena, text, .{})
    else |_|
        null;

    var passed = true;
    var cases = index.object.get("cases").?.object.iterator();
    while (cases.next()) |case| {
        const meta = case.value_ptr.object;
        const files = meta.get("files").?.object;
        if (files.get("codec/samples") == null) continue;
        if (sidecar) |s| if (!std.mem.eql(u8, s.object.get("sha256").?.string, meta.get("codec_sha256").?.string)) {
            std.debug.print("{s}: the codec file is not the one traced\n", .{case.key_ptr.*});
            return error.ArtifactMismatch;
        };
        const frames: usize = @intCast(meta.get("codec_frames").?.integer);
        const code_bytes = try readFixture(arena, io, fixtures, files.get("codes").?);
        if (code_bytes.len < frames * codec.groups * 2) return error.FixtureMismatch;
        const codes = try arena.alloc(codec.Frame, frames);
        for (codes, 0..) |*frame, t| for (frame, 0..) |*code, g| {
            code.* = std.mem.readInt(u16, code_bytes[(t * codec.groups + g) * 2 ..][0..2], .little);
        };
        const per = runtime.config.samplesPerFrame();
        const samples = try arena.alloc(f32, frames * per);
        var recorder: Recorder = .{ .gpa = gpa };
        defer recorder.deinit();
        const start = std.Io.Clock.awake.now(io);
        try runtime.decodeObserved(gpa, codes, samples, .{ .context = &recorder, .record = Recorder.record });
        const ms = start.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds();
        if (recorder.failed) return error.OutOfMemory;
        std.debug.print("{s}: {d} frames decoded in {d} ms ({d:.2}× real time)\n", .{ case.key_ptr.*, frames, ms, @as(f64, @floatFromInt(frames)) * 80 / @as(f64, @floatFromInt(@max(ms, 1))) });
        std.debug.print("  {s:<12} {s:>13} {s:>10} {s:>10} {s:>10}\n", .{ "stage", "rows", "max |Δ|", "max |ref|", "rel RMS" });
        // The fixture's stages in pipeline order, the samples last (one value a row).
        const labels = comptime blk: {
            var names: [@typeInfo(codec.Stage).@"enum".field_names.len + 1][]const u8 = undefined;
            for (std.enums.values(codec.Stage), 0..) |stage, i| names[i] = stage.label();
            names[names.len - 1] = "samples";
            break :blk names;
        };
        for (labels, 0..) |label, i| {
            const key = try std.fmt.allocPrint(arena, "codec/{s}", .{label});
            const entry = files.get(key) orelse continue;
            const is_samples = i == labels.len - 1;
            const got_all = if (is_samples) samples else recorder.rows[i].items;
            const want = try floats(arena, try readFixture(arena, io, fixtures, entry));
            const shape = entry.object.get("shape").?.array.items;
            const width: usize = if (is_samples) 1 else @intCast(shape[shape.len - 1].integer);
            const rows = entry.object.get("rows").?.array.items;
            const first: usize = @intCast(rows[0].integer);
            const last: usize = @intCast(rows[1].integer);
            if (got_all.len != last * width or want.len != (last - first) * width) {
                std.debug.print("  {s}: {d} values, the trace's {d} rows of {d}\n", .{ label, got_all.len, last, width });
                return error.FixtureMismatch;
            }
            const s = compare(got_all[first * width ..], want);
            const rel = @sqrt(s.diff2 / s.ref2);
            const over = rel > bounds.relative_rms or (is_samples and s.max_abs > bounds.absolute);
            if (over) passed = false;
            std.debug.print("  {s:<12} {d:>6}–{d:<6} {e:>10.2} {e:>10.2} {e:>10.2}{s}\n", .{ label, first, last, s.max_abs, s.max_ref, rel, if (over) "  OVER" else "" });
        }

        // The same frames through a stream, five at a time.
        var stream = try codec.Stream.init(gpa, &runtime);
        defer stream.deinit();
        const pieces = try arena.alloc(f32, samples.len);
        var at: usize = 0;
        while (at < frames) : (at += 5) {
            const n = @min(5, frames - at);
            try stream.push(codes[at..][0..n], pieces[at * per ..][0 .. n * per]);
        }
        const same = std.mem.eql(u8, std.mem.sliceAsBytes(samples), std.mem.sliceAsBytes(pieces));
        if (!same) passed = false;
        std.debug.print("  streamed in pieces of 5: {s}\n", .{if (same) "the decode's samples bit for bit" else "DIFFERS from the decode"});
    }
    std.debug.print("Bounds: relative RMS {e} per stage, samples max |Δ| {e}: {s}.\n", .{ bounds.relative_rms, bounds.absolute, if (passed) "passed" else "FAILED" });
    return passed;
}

fn readCodes(arena: std.mem.Allocator, io: std.Io, fixtures: []const u8, entry: std.json.Value, frames: usize) ![]codec.Frame {
    const code_bytes = try readFixture(arena, io, fixtures, entry);
    if (code_bytes.len < frames * codec.groups * 2) return error.FixtureMismatch;
    const codes = try arena.alloc(codec.Frame, frames);
    for (codes, 0..) |*frame, t| for (frame, 0..) |*code, g| {
        code.* = std.mem.readInt(u16, code_bytes[(t * codec.groups + g) * 2 ..][0..2], .little);
    };
    return codes;
}

/// Each case's traced codes (every frame the trace sampled) decoded on Metal
/// against the CPU decode of the same codes, and the trace's samples where it
/// pinned them; then the Metal stream in pieces of 1, 3, and 7 against its
/// own whole decode, bit for bit.
fn checkCodecMetal(gpa: std.mem.Allocator, arena: std.mem.Allocator, io: std.Io, model: []const u8, fixtures: []const u8, bounds: Bounds) !bool {
    const index_path = try std.fs.path.join(arena, &.{ fixtures, "index.json" });
    const index_text = try std.Io.Dir.cwd().readFileAlloc(io, index_path, arena, .limited(1 << 20));
    const index = try std.json.parseFromSliceLeaky(std.json.Value, arena, index_text, .{});
    const codec_path = try std.fs.path.join(arena, &.{ model, "speech_tokenizer", "model.safetensors" });
    var checkpoint = try safetensors.Checkpoint.open(gpa, io, codec_path, .{});
    defer checkpoint.deinit(io);
    var binding = try codec.bind(gpa, &checkpoint, .{}, null);
    defer binding.deinit(gpa);
    var runtime = try codec.Runtime.init(gpa, io, &binding);
    defer runtime.deinit();
    var message: [512]u8 = @splat(0);
    var gpu = metal.Backend.init(gpa, &message) catch |err| {
        std.debug.print("Metal: {s} {s}\n", .{ @errorName(err), std.mem.sliceTo(&message, 0) });
        return err;
    };
    defer gpu.deinit();
    var decoder = try codec_metal.Decoder.init(gpa, &gpu, &runtime);
    defer decoder.deinit();
    std.debug.print("Codec decoder on Metal: {d} MB of device buffers.\n", .{decoder.bytes / 1_000_000});
    const per = runtime.config.samplesPerFrame();

    var passed = true;
    var cases = index.object.get("cases").?.object.iterator();
    while (cases.next()) |case| {
        const meta = case.value_ptr.object;
        const files = meta.get("files").?.object;
        const sampled = meta.get("frames_generated") orelse continue;
        const frames: usize = @intCast(sampled.integer);
        const codes = try readCodes(arena, io, fixtures, files.get("codes") orelse continue, frames);
        const want = try arena.alloc(f32, frames * per);
        const cpu_start = std.Io.Clock.awake.now(io);
        try runtime.decode(gpa, codes, want);
        const cpu_ms = cpu_start.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds();
        var stream = try codec_metal.Stream.init(&decoder);
        defer stream.deinit();
        const whole = try arena.alloc(f32, frames * per);
        const start = std.Io.Clock.awake.now(io);
        try stream.push(codes, whole);
        const metal_us = start.durationTo(std.Io.Clock.awake.now(io)).toMicroseconds();
        const s = compare(whole, want);
        const rel = @sqrt(s.diff2 / s.ref2);
        const over = !(rel <= bounds.relative_rms and s.max_abs <= bounds.absolute);
        if (over) passed = false;
        std.debug.print("{s}: {d} frames, Metal {d:.1} ms ({d:.2} ms a frame; the CPU {d} ms)\n", .{ case.key_ptr.*, frames, @as(f64, @floatFromInt(metal_us)) / 1000, @as(f64, @floatFromInt(metal_us)) / 1000 / @as(f64, @floatFromInt(frames)), cpu_ms });
        std.debug.print("  {s:<26} max |Δ| {e:.2}  max |ref| {e:.2}  rel RMS {e:.2}{s}\n", .{ "samples vs the CPU decode", s.max_abs, s.max_ref, rel, if (over) "  OVER" else "" });
        if (files.get("codec/samples")) |entry| {
            const traced = try floats(arena, try readFixture(arena, io, fixtures, entry));
            const rows = entry.object.get("rows").?.array.items;
            const first: usize = @intCast(rows[0].integer);
            const last: usize = @intCast(rows[1].integer);
            if (last > whole.len or traced.len != last - first) return error.FixtureMismatch;
            const t = compare(whole[first..last], traced);
            const trel = @sqrt(t.diff2 / t.ref2);
            const tover = !(trel <= bounds.relative_rms and t.max_abs <= bounds.absolute);
            if (tover) passed = false;
            std.debug.print("  {s:<26} max |Δ| {e:.2}  max |ref| {e:.2}  rel RMS {e:.2}{s}\n", .{ "samples vs the trace", t.max_abs, t.max_ref, trel, if (tover) "  OVER" else "" });
        }
        for ([_]usize{ 1, 3, 7 }) |piece| {
            stream.reset();
            const pieces = try arena.alloc(f32, whole.len);
            var at: usize = 0;
            while (at < frames) : (at += piece) {
                const n = @min(piece, frames - at);
                try stream.push(codes[at..][0..n], pieces[at * per ..][0 .. n * per]);
            }
            const same = std.mem.eql(u8, std.mem.sliceAsBytes(whole), std.mem.sliceAsBytes(pieces));
            if (!same) passed = false;
            std.debug.print("  streamed in pieces of {d}: {s}\n", .{ piece, if (same) "the Metal decode bit for bit" else "DIFFERS from the Metal decode" });
        }
    }
    std.debug.print("Bounds: relative RMS {e}, max |Δ| {e}: {s}.\n", .{ bounds.relative_rms, bounds.absolute, if (passed) "passed" else "FAILED" });
    return passed;
}

/// Every talker layer's rows and the final norm's, appended call by call.
const Layers = struct {
    gpa: std.mem.Allocator,
    rows: []std.ArrayList(f32),
    norm: std.ArrayList(f32) = .empty,
    failed: bool = false,

    fn record(context: *anyopaque, layer: ?usize, rows: []const f32) void {
        const self: *Layers = @ptrCast(@alignCast(context));
        const list = if (layer) |i| &self.rows[i] else &self.norm;
        list.appendSlice(self.gpa, rows) catch {
            self.failed = true;
        };
    }
};

/// Forces the trace's codes on the predictor, keeping the logits it saw.
const Replay = struct {
    gpa: std.mem.Allocator,
    frame: codec.Frame = undefined,
    logits: ?*std.ArrayList(f32) = null,
    failed: bool = false,

    fn pick(context: *anyopaque, group: usize, logits: []const f32) anyerror!u16 {
        const self: *Replay = @ptrCast(@alignCast(context));
        if (self.logits) |list| try list.appendSlice(self.gpa, logits);
        return self.frame[group];
    }
};

fn integers(arena: std.mem.Allocator, value: ?std.json.Value) ![]u32 {
    const items = if (value) |v| v.array.items else return &.{};
    const out = try arena.alloc(u32, items.len);
    for (out, items) |*o, item| o.* = @intCast(item.integer);
    return out;
}

/// One stage's comparison, printed; whether it is within `bound`.
fn report(label: []const u8, first: usize, last: usize, got: []const f32, want: []const f32, bound: f64) bool {
    const s = compare(got, want);
    const rel = @sqrt(s.diff2 / s.ref2);
    const over = !(rel <= bound);
    std.debug.print("  {s:<18} {d:>4}–{d:<4} {e:>10.2} {e:>10.2} {e:>10.2}{s}\n", .{ label, first, last, s.max_abs, s.max_ref, rel, if (over) "  OVER" else "" });
    return !over;
}

/// Rows whose largest logit is the trace's, of `width`-wide rows.
fn sameArgmax(got: []const f32, want: []const f32, width: usize) usize {
    var same: usize = 0;
    for (0..want.len / width) |r| {
        const g = std.mem.indexOfMax(f32, got[r * width ..][0..width]);
        const w = std.mem.indexOfMax(f32, want[r * width ..][0..width]);
        same += @intFromBool(g == w);
    }
    return same;
}

/// The talker and predictor under test: the CPU reference or the Metal plan.
const Engine = union(enum) {
    cpu: struct { model: *const speech.Model, talker: *speech.Talker, predictor: *speech.Predictor },
    metal: *tts_metal.Plan,

    fn embed(self: Engine, gpa: std.mem.Allocator, rows: []const tts.Row, out: []f32) !void {
        switch (self) {
            .cpu => |e| try e.model.embed(gpa, rows, out),
            .metal => |plan| try plan.embed(rows, out),
        }
    }
    /// The prompt's rows `x` (consumed); the last post-norm row into `last` and its logits.
    fn prefill(self: Engine, x: []f32, n: usize, hidden: []f32, last: []f32, logits: []f32, observer: speech.Observer) !void {
        const w = last.len;
        switch (self) {
            .cpu => |e| {
                try e.talker.forward(x, n, hidden, observer);
                @memcpy(last, hidden[(n - 1) * w ..][0..w]);
                try e.talker.logits(last, logits);
            },
            .metal => |plan| {
                plan.observer = observer;
                try plan.prefill(x, n);
                @memcpy(last, plan.hiddenRow());
                @memcpy(logits, plan.talkerLogits());
            },
        }
    }
    fn frame(self: Engine, last: []const f32, c0: u16, pick: speech.Pick) !codec.Frame {
        return switch (self) {
            .cpu => |e| try e.predictor.frame(last, c0, pick),
            .metal => |plan| try plan.frame(c0, pick),
        };
    }
    fn nextRow(self: Engine, frame_codes: codec.Frame, pad: []const f32, row: []f32) !void {
        switch (self) {
            .cpu => |e| e.model.nextRow(frame_codes, pad, row),
            .metal => |plan| try plan.nextRow(frame_codes, pad, row),
        }
    }
    fn step(self: Engine, row: []f32, last: []f32, logits: []f32, observer: speech.Observer) !void {
        switch (self) {
            .cpu => |e| {
                try e.talker.forward(row, 1, last, observer);
                try e.talker.logits(last, logits);
            },
            .metal => |plan| {
                try plan.step(row);
                @memcpy(last, plan.hiddenRow());
                @memcpy(logits, plan.talkerLogits());
            },
        }
    }
};

fn checkTalker(gpa: std.mem.Allocator, arena: std.mem.Allocator, io: std.Io, model_dir: []const u8, fixtures: []const u8, bound: f64, on_metal: bool, kv: tts_metal.Precision) !bool {
    const index_path = try std.fs.path.join(arena, &.{ fixtures, "index.json" });
    const index_text = try std.Io.Dir.cwd().readFileAlloc(io, index_path, arena, .limited(1 << 20));
    const index = try std.json.parseFromSliceLeaky(std.json.Value, arena, index_text, .{});

    const read = struct {
        fn file(a: std.mem.Allocator, i: std.Io, dir: []const u8, name: []const u8) ![]u8 {
            const path = try std.fs.path.join(a, &.{ dir, name });
            return std.Io.Dir.cwd().readFileAlloc(i, path, a, .limited(1 << 20));
        }
    };
    var config = try tts.parseConfig(gpa, try read.file(arena, io, model_dir, "config.json"), try read.file(arena, io, model_dir, "generation_config.json"));
    defer config.deinit();
    var tokenizer = try tts.loadTokenizer(gpa, io, model_dir);
    defer tokenizer.deinit();
    const weights_path = try std.fs.path.join(arena, &.{ model_dir, "model.safetensors" });
    const load_start = std.Io.Clock.awake.now(io);
    var checkpoint = try safetensors.Checkpoint.open(gpa, io, weights_path, .{});
    defer checkpoint.deinit(io);
    var diagnostic: tts.Diagnostic = .{};
    var binding = tts.bind(gpa, &checkpoint, &config, &diagnostic) catch |err| {
        std.debug.print("bind: {s} ({s})\n", .{ @errorName(err), diagnostic.name() });
        return err;
    };
    defer binding.deinit(gpa);
    var model: ?speech.Model = null;
    defer if (model) |*m| m.deinit();
    var gpu: ?metal.Backend = null;
    defer if (gpu) |*g| g.deinit();
    var plan: ?tts_metal.Plan = null;
    defer if (plan) |*p| p.deinit();
    if (on_metal) {
        var message: [512]u8 = @splat(0);
        gpu = metal.Backend.init(gpa, &message) catch |err| {
            std.debug.print("Metal: {s} {s}\n", .{ @errorName(err), std.mem.sliceTo(&message, 0) });
            return err;
        };
        plan = try tts_metal.Plan.init(gpa, &gpu.?, &binding, &config, kv);
        std.debug.print("Talker and predictor ({s}) on Metal in {d} ms: {d} tensors, {d} MB as published (wrapped), {s} talker cache.\n", .{ config.size, load_start.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds(), binding.tensors, binding.bytes / 1_000_000, @tagName(kv) });
    } else {
        model = try speech.Model.init(gpa, io, &binding, &config);
        std.debug.print("Talker and predictor ({s}) bound and loaded in {d} ms: {d} tensors, {d} MB as published, {d} MB as F32.\n", .{ config.size, load_start.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds(), binding.tensors, binding.bytes / 1_000_000, model.?.bytes / 1_000_000 });
    }
    const sidecar: ?std.json.Value = if (read.file(arena, io, model_dir, "model.safetensors.nuclis.json")) |text|
        try std.json.parseFromSliceLeaky(std.json.Value, arena, text, .{})
    else |_|
        null;

    const w = config.talker.hidden;
    const dir_name = std.fs.path.basename(std.mem.trimEnd(u8, model_dir, "/"));
    var passed = true;
    var checked: usize = 0;
    var cases = index.object.get("cases").?.object.iterator();
    while (cases.next()) |case| {
        const meta = case.value_ptr.object;
        const files = meta.get("files").?.object;
        if (files.get("talker/inputs") == null) continue;
        if (!std.mem.eql(u8, std.fs.path.basename(meta.get("repo").?.string), dir_name)) continue;
        if (sidecar) |s| if (!std.mem.eql(u8, s.object.get("sha256").?.string, meta.get("model_sha256").?.string)) {
            std.debug.print("{s}: model.safetensors is not the one traced\n", .{case.key_ptr.*});
            return error.ArtifactMismatch;
        };
        checked += 1;
        std.debug.print("{s}:\n", .{case.key_ptr.*});

        // Token ids and the prompt's rows: exact.
        const instruct: ?[]const u8 = if (meta.get("instruct")) |v| v.string else null;
        var text = try tts.encodeText(gpa, &tokenizer, meta.get("text").?.string, instruct);
        defer text.deinit(gpa);
        const want_text = try integers(arena, meta.get("text_ids"));
        const want_instruct = try integers(arena, meta.get("instruct_ids"));
        const ids_equal = std.mem.eql(u32, text.all, want_text) and std.mem.eql(u32, text.instruct, want_instruct);
        std.debug.print("  token ids ({d} + {d}): {s}\n", .{ want_text.len, want_instruct.len, if (ids_equal) "equal" else "DIFFER" });
        if (!ids_equal) passed = false;
        var prompt = try tts.Prompt.build(gpa, &config, .{
            .role = text.role,
            .text = text.text,
            .instruct = text.instruct,
            .speaker = meta.get("speaker").?.string,
            .language = meta.get("language").?.string,
        });
        defer prompt.deinit(gpa);
        const want_rows = meta.get("prompt_rows").?.array.items;
        var rows_equal = want_rows.len == prompt.rows.len;
        if (rows_equal) for (prompt.rows, want_rows) |row, pair| {
            const t = pair.array.items[0].integer;
            const c = pair.array.items[1].integer;
            const got_t: i64 = if (row.text) |v| v else -1;
            const got_c: i64 = if (row.codec) |v| v else -1;
            rows_equal = rows_equal and t == got_t and c == got_c;
        };
        std.debug.print("  prompt rows ({d}): {s}\n", .{ want_rows.len, if (rows_equal) "equal" else "DIFFER" });
        if (!rows_equal) {
            passed = false;
            continue;
        }

        const frames: usize = @intCast(meta.get("talker_frames").?.integer);
        const predicted: usize = @intCast(meta.get("predictor_frames").?.integer);
        const code_bytes = try readFixture(arena, io, fixtures, files.get("codes").?);
        if (code_bytes.len < frames * codec.groups * 2) return error.FixtureMismatch;
        const codes = try arena.alloc(codec.Frame, frames);
        for (codes, 0..) |*frame, t| for (frame, 0..) |*code, g| {
            code.* = std.mem.readInt(u16, code_bytes[(t * codec.groups + g) * 2 ..][0..2], .little);
        };

        const length = prompt.rows.len;
        const inputs = try arena.alloc(f32, (length + frames) * w);
        const hidden = try arena.alloc(f32, length * w);
        const x = try arena.alloc(f32, length * w);
        const logits = try arena.alloc(f32, (frames + 1) * config.codec_vocab);
        var layers: Layers = .{ .gpa = gpa, .rows = try arena.alloc(std.ArrayList(f32), config.talker.layers) };
        @memset(layers.rows, .empty);
        defer {
            for (layers.rows) |*r| r.deinit(gpa);
            layers.norm.deinit(gpa);
        }
        const observer: speech.Observer = .{ .context = &layers, .record = Layers.record };
        var talker: ?speech.Talker = null;
        defer if (talker) |*t| t.deinit();
        var predictor: ?speech.Predictor = null;
        defer if (predictor) |*p| p.deinit();
        const engine: Engine = if (plan) |*p| .{ .metal = p } else blk: {
            talker = try speech.Talker.init(gpa, &model.?);
            predictor = try speech.Predictor.init(gpa, &model.?);
            break :blk .{ .cpu = .{ .model = &model.?, .talker = &talker.?, .predictor = &predictor.? } };
        };
        const pad = try arena.alloc(f32, w);
        try engine.embed(gpa, &.{.{ .text = config.text.tts_pad, .codec = null }}, pad);

        const prefill_start = std.Io.Clock.awake.now(io);
        try engine.embed(gpa, prompt.rows, inputs[0 .. length * w]);
        @memcpy(x, inputs[0 .. length * w]);
        const last = try arena.alloc(f32, w);
        try engine.prefill(x, length, hidden, last, logits[0..config.codec_vocab], observer);
        const prefill_ms = prefill_start.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds();

        var predictor_logits: std.ArrayList(f32) = .empty;
        defer predictor_logits.deinit(gpa);
        var replay: Replay = .{ .gpa = gpa };
        const frames_start = std.Io.Clock.awake.now(io);
        for (codes, 0..) |frame, f| {
            replay.frame = frame;
            replay.logits = if (f < predicted) &predictor_logits else null;
            const got = try engine.frame(last, frame[0], .{ .context = &replay, .pick = Replay.pick });
            std.debug.assert(std.mem.eql(u16, &got, &frame));
            const row = inputs[(length + f) * w ..][0..w];
            try engine.nextRow(frame, pad, row);
            const step = try arena.dupe(f32, row);
            try engine.step(step, last, logits[(f + 1) * config.codec_vocab ..][0..config.codec_vocab], observer);
        }
        const frame_ms = @as(f64, @floatFromInt(frames_start.durationTo(std.Io.Clock.awake.now(io)).toMicroseconds())) / 1000 / @as(f64, @floatFromInt(frames));
        if (layers.failed) return error.OutOfMemory;
        std.debug.print("  prefill of {d} rows {d} ms; {d:.1} ms a frame (predictor and talker step)\n", .{ length, prefill_ms, frame_ms });

        std.debug.print("  {s:<18} {s:>9} {s:>10} {s:>10} {s:>10}\n", .{ "stage", "rows", "max |Δ|", "max |ref|", "rel RMS" });
        var stages = files.iterator();
        while (stages.next()) |entry| {
            const key = entry.key_ptr.*;
            const got_all: []const f32 = if (std.mem.eql(u8, key, "talker/inputs"))
                inputs
            else if (std.mem.eql(u8, key, "talker/norm"))
                layers.norm.items
            else if (std.mem.startsWith(u8, key, "talker/layer."))
                layers.rows[try std.fmt.parseInt(usize, key["talker/layer.".len..], 10)].items
            else if (std.mem.eql(u8, key, "talker/logits"))
                logits
            else if (std.mem.eql(u8, key, "predictor/logits"))
                predictor_logits.items
            else
                continue;
            const want = try floats(arena, try readFixture(arena, io, fixtures, entry.value_ptr.*));
            const shape = entry.value_ptr.object.get("shape").?.array.items;
            const width: usize = @intCast(shape[1].integer);
            const first: usize = if (entry.value_ptr.object.get("rows")) |r| @intCast(r.array.items[0].integer) else 0;
            const last_row = first + want.len / width;
            if (got_all.len < last_row * width) {
                std.debug.print("  {s}: {d} values, the trace's rows end at {d} of {d}\n", .{ key, got_all.len, last_row, width });
                return error.FixtureMismatch;
            }
            const got = got_all[first * width .. last_row * width];
            if (!report(key, first, last_row, got, want, bound)) passed = false;
            if (std.mem.endsWith(u8, key, "logits"))
                std.debug.print("  {s:<18} argmax equal in {d} of {d} rows\n", .{ "", sameArgmax(got, want, width), want.len / width });
        }
    }
    if (checked == 0) {
        std.debug.print("No talker case in {s} for {s}.\n", .{ fixtures, dir_name });
        return false;
    }
    std.debug.print("Bound: relative RMS {e} per stage: {s}.\n", .{ bound, if (passed) "passed" else "FAILED" });
    return passed;
}

/// Every speaker-encoder stage's rows, by label.
const Stages = struct {
    arena: std.mem.Allocator,
    rows: std.StringArrayHashMapUnmanaged([]const f32) = .empty,
    failed: bool = false,

    fn record(context: *anyopaque, label: []const u8, values: []const f32) void {
        const self: *Stages = @ptrCast(@alignCast(context));
        const key = self.arena.dupe(u8, label) catch return self.fail();
        const copy = self.arena.dupe(f32, values) catch return self.fail();
        self.rows.put(self.arena, key, copy) catch return self.fail();
    }

    fn fail(self: *Stages) void {
        self.failed = true;
    }
};

/// The reference case's clip through both encoders, against the trace.
fn checkClone(gpa: std.mem.Allocator, arena: std.mem.Allocator, io: std.Io, model_dir: []const u8, fixtures: []const u8, bound: f64) !bool {
    const index_path = try std.fs.path.join(arena, &.{ fixtures, "index.json" });
    const index_text = try std.Io.Dir.cwd().readFileAlloc(io, index_path, arena, .limited(1 << 20));
    const index = try std.json.parseFromSliceLeaky(std.json.Value, arena, index_text, .{});
    const read = struct {
        fn file(a: std.mem.Allocator, i: std.Io, dir: []const u8, name: []const u8) ![]u8 {
            const path = try std.fs.path.join(a, &.{ dir, name });
            return std.Io.Dir.cwd().readFileAlloc(i, path, a, .limited(1 << 20));
        }
    };
    var config = try tts.parseConfig(gpa, try read.file(arena, io, model_dir, "config.json"), null);
    defer config.deinit();
    const encoder_config = config.speaker_encoder orelse {
        std.debug.print("{s}: not a Base (no speaker encoder)\n", .{model_dir});
        return error.NotABase;
    };
    const weights_path = try std.fs.path.join(arena, &.{ model_dir, "model.safetensors" });
    var checkpoint = try safetensors.Checkpoint.open(gpa, io, weights_path, .{});
    defer checkpoint.deinit(io);
    var diagnostic: tts.Diagnostic = .{};
    var binding = tts.bind(gpa, &checkpoint, &config, &diagnostic) catch |err| {
        std.debug.print("bind: {s} ({s})\n", .{ @errorName(err), diagnostic.name() });
        return err;
    };
    defer binding.deinit(gpa);
    var encoder = try ecapa.Encoder.init(gpa, &binding.speaker_encoder.?, encoder_config);
    defer encoder.deinit();
    const codec_path = try std.fs.path.join(arena, &.{ model_dir, "speech_tokenizer", "model.safetensors" });
    var codec_checkpoint = try safetensors.Checkpoint.open(gpa, io, codec_path, .{});
    defer codec_checkpoint.deinit(io);
    var codec_diagnostic: codec.Diagnostic = .{};
    var decoder_binding = codec.bind(gpa, &codec_checkpoint, .{}, &codec_diagnostic) catch |err| {
        std.debug.print("codec bind: {s} ({s})\n", .{ @errorName(err), codec_diagnostic.name() });
        return err;
    };
    defer decoder_binding.deinit(gpa);
    var encoder_binding = codec_encoder.bind(gpa, &codec_checkpoint, &decoder_binding, .{}, &codec_diagnostic) catch |err| {
        std.debug.print("codec encoder bind: {s} ({s})\n", .{ @errorName(err), codec_diagnostic.name() });
        return err;
    };
    defer encoder_binding.deinit(gpa);
    var codes_encoder = try codec_encoder.Encoder.init(gpa, io, &decoder_binding, &encoder_binding);
    defer codes_encoder.deinit();
    const sidecar: ?std.json.Value = if (read.file(arena, io, model_dir, "model.safetensors.nuclis.json")) |text|
        try std.json.parseFromSliceLeaky(std.json.Value, arena, text, .{})
    else |_|
        null;

    const dir_name = std.fs.path.basename(std.mem.trimEnd(u8, model_dir, "/"));
    var passed = true;
    var checked: usize = 0;
    var cases = index.object.get("cases").?.object.iterator();
    while (cases.next()) |case| {
        const meta = case.value_ptr.object;
        const files = meta.get("files").?.object;
        const reference = files.get("reference") orelse continue;
        if (!std.mem.eql(u8, std.fs.path.basename(meta.get("repo").?.string), dir_name)) continue;
        if (sidecar) |s| if (!std.mem.eql(u8, s.object.get("sha256").?.string, meta.get("model_sha256").?.string)) {
            std.debug.print("{s}: model.safetensors is not the one traced\n", .{case.key_ptr.*});
            return error.ArtifactMismatch;
        };
        checked += 1;
        var pcm = try inference.audio.audio.decodeWavAt(gpa, try readFixture(arena, io, fixtures, reference), std.math.maxInt(u32), ecapa.sample_rate);
        defer pcm.deinit(gpa);
        var stages: Stages = .{ .arena = arena };
        const embedding = try arena.alloc(f32, encoder_config.dim);
        const start = std.Io.Clock.awake.now(io);
        try encoder.embed(gpa, io, pcm.samples, embedding, .{ .context = &stages, .record = Stages.record });
        if (stages.failed) return error.OutOfMemory;
        const speaker_ms = start.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds();
        const codes = try arena.alloc(codec.Frame, codes_encoder.frameCount(pcm.samples.len));
        const codec_start = std.Io.Clock.awake.now(io);
        try codes_encoder.encode(gpa, pcm.samples, codes, .{ .context = &stages, .record = Stages.record });
        if (stages.failed) return error.OutOfMemory;
        std.debug.print("{s}: {d:.2} s of reference, speaker encoder in {d} ms, codec encoder in {d} ms\n", .{ case.key_ptr.*, @as(f64, @floatFromInt(pcm.samples.len)) / ecapa.sample_rate, speaker_ms, codec_start.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds() });
        std.debug.print("  {s:<12} {s:>13} {s:>10} {s:>10} {s:>10}\n", .{ "stage", "rows", "max |Δ|", "max |ref|", "rel RMS" });
        try stages.rows.put(arena, "embedding", embedding);
        for ([_][]const u8{ "mel", "blocks.0", "blocks.3", "mfa", "asp", "embedding", "seanet", "transformer", "downsample" }) |label| {
            const key = if (std.mem.eql(u8, label, "embedding")) label else try std.fmt.allocPrint(arena, "encoders/{s}", .{label});
            const entry = files.get(key) orelse continue;
            const want = try floats(arena, try readFixture(arena, io, fixtures, entry));
            const got_all = stages.rows.get(label) orelse {
                std.debug.print("  {s}: not computed\n", .{label});
                return error.FixtureMismatch;
            };
            // A stage pinned by rows keeps its last ones; the others are whole.
            var first: usize = 0;
            var last: usize = 1;
            var width = want.len;
            if (entry.object.get("rows")) |rows| {
                first = @intCast(rows.array.items[0].integer);
                last = @intCast(rows.array.items[1].integer);
                width = want.len / (last - first);
            }
            if (got_all.len != last * width) {
                std.debug.print("  {s}: {d} values, the trace's {d} rows of {d}\n", .{ label, got_all.len, last, width });
                return error.FixtureMismatch;
            }
            const s = compare(got_all[first * width ..], want);
            const rel = @sqrt(s.diff2 / s.ref2);
            const over = rel > bound;
            if (over) passed = false;
            std.debug.print("  {s:<12} {d:>6}–{d:<6} {e:>10.2} {e:>10.2} {e:>10.2}{s}\n", .{ label, first, last, s.max_abs, s.max_ref, rel, if (over) "  OVER" else "" });
        }
        // Codes are integers: equal, frame by frame and group by group.
        const want_frames: usize = @intCast(files.get("codes").?.object.get("shape").?.array.items[0].integer);
        if (want_frames != codes.len) {
            std.debug.print("  codes: {d} frames, the trace's {d}\n", .{ codes.len, want_frames });
            return error.FixtureMismatch;
        }
        const want = try readCodes(arena, io, fixtures, files.get("codes").?, codes.len);
        var differ: usize = 0;
        for (codes, want, 0..) |got, w, t| for (got, w, 0..) |a, b, g| if (a != b) {
            if (differ < 8) std.debug.print("    frame {d} group {d}: {d}, the trace's {d}\n", .{ t, g, a, b });
            differ += 1;
        };
        const total_codes = codes.len * codec.groups;
        if (differ > 0) passed = false;
        std.debug.print("  codes ({d} × {d}): {s}", .{ codes.len, codec.groups, if (differ == 0) "equal\n" else "" });
        if (differ > 0) std.debug.print("{d} of {d} DIFFER\n", .{ differ, total_codes });
    }
    if (checked == 0) {
        std.debug.print("No reference case in {s} for {s}.\n", .{ fixtures, dir_name });
        return false;
    }
    std.debug.print("Bound: relative RMS {e} per stage, codes equal: {s}.\n", .{ bound, if (passed) "passed" else "FAILED" });
    return passed;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    const Mode = enum { codec, talker, @"codec-metal", @"talker-metal", clone };
    const mode = std.meta.stringToEnum(Mode, if (args.len >= 3) args[2] else "") orelse {
        std.debug.print(usage, .{});
        return error.InvalidArguments;
    };
    var bounds: Bounds = .{ .relative_rms = 1e-5, .absolute = 1e-5 };
    var fixtures: []const u8 = "tests/fixtures/qwen3-tts";
    // Each Metal talker cache to check, with its bound (the F16 cache's from --max-relative-rms-f16).
    var f32_cache = true;
    var f16_cache = false;
    var f16_bound: f64 = 1e-3;
    var i: usize = 3;
    while (i + 1 < args.len) : (i += 2) {
        const value = args[i + 1];
        if (std.mem.eql(u8, args[i], "--max-relative-rms")) {
            bounds.relative_rms = try std.fmt.parseFloat(f64, value);
        } else if (std.mem.eql(u8, args[i], "--max-absolute")) {
            bounds.absolute = try std.fmt.parseFloat(f64, value);
        } else if (std.mem.eql(u8, args[i], "--fixtures")) {
            fixtures = value;
        } else if (std.mem.eql(u8, args[i], "--max-relative-rms-f16")) {
            f16_bound = try std.fmt.parseFloat(f64, value);
        } else if (std.mem.eql(u8, args[i], "--kv")) {
            f32_cache = std.mem.eql(u8, value, "f32") or std.mem.eql(u8, value, "both");
            f16_cache = std.mem.eql(u8, value, "f16") or std.mem.eql(u8, value, "both");
            if (!f32_cache and !f16_cache) return error.InvalidArguments;
        } else break;
    }
    if (i != args.len) {
        std.debug.print(usage, .{});
        return error.InvalidArguments;
    }
    const ok = switch (mode) {
        .codec => try checkCodec(gpa, arena, io, args[1], fixtures, bounds),
        .talker => try checkTalker(gpa, arena, io, args[1], fixtures, bounds.relative_rms, false, .f32),
        .@"codec-metal" => try checkCodecMetal(gpa, arena, io, args[1], fixtures, bounds),
        .@"talker-metal" => metal: {
            const exact = !f32_cache or try checkTalker(gpa, arena, io, args[1], fixtures, bounds.relative_rms, true, .f32);
            const half = !f16_cache or try checkTalker(gpa, arena, io, args[1], fixtures, f16_bound, true, .f16);
            break :metal exact and half;
        },
        .clone => try checkClone(gpa, arena, io, args[1], fixtures, bounds.relative_rms),
    };
    if (!ok) std.process.exit(1);
}
