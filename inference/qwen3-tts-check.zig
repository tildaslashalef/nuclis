//! Explicit full-model check of Qwen3-TTS against the authors' bring-up
//! trace (`scripts/speech-traces/qwen3_tts_trace.py`, pinned under
//! `tests/fixtures/qwen3-tts/`). `codec`: each case's codes through the
//! codec decoder's `decode`, every stage the fixture holds compared with
//! its rows, then the samples; then the same frames pushed through a
//! `Stream` in pieces of 5, which must give `decode`'s samples bit for bit.
//! The bounds are arguments, so the gate manifest holds them.
//!
//!   qwen3-tts-check MODEL_DIR codec [--max-relative-rms X] [--max-absolute X] [--fixtures DIR]
const std = @import("std");
const inference = @import("inference");
const codec = inference.audio.qwen3_codec;
const safetensors = inference.safetensors;

const usage = "usage: qwen3-tts-check MODEL_DIR codec [--max-relative-rms X] [--max-absolute X] [--fixtures DIR]\n";

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

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    if (args.len < 3 or !std.mem.eql(u8, args[2], "codec")) {
        std.debug.print(usage, .{});
        return error.InvalidArguments;
    }
    var bounds: Bounds = .{ .relative_rms = 1e-5, .absolute = 1e-5 };
    var fixtures: []const u8 = "tests/fixtures/qwen3-tts";
    var i: usize = 3;
    while (i + 1 < args.len) : (i += 2) {
        const value = args[i + 1];
        if (std.mem.eql(u8, args[i], "--max-relative-rms")) {
            bounds.relative_rms = try std.fmt.parseFloat(f64, value);
        } else if (std.mem.eql(u8, args[i], "--max-absolute")) {
            bounds.absolute = try std.fmt.parseFloat(f64, value);
        } else if (std.mem.eql(u8, args[i], "--fixtures")) {
            fixtures = value;
        } else break;
    }
    if (i != args.len) {
        std.debug.print(usage, .{});
        return error.InvalidArguments;
    }
    if (!try checkCodec(gpa, arena, io, args[1], fixtures, bounds)) std.process.exit(1);
}
