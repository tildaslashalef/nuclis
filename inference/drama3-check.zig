//! Explicit full-model check of Drama 3's backbone and tower on the CPU.
//! `smoke`: a short text prompt through the backbone and the tower's
//! teacher-forced prefill, the 4,097 logits checked finite, then a few
//! greedy frames (argmax main token and codes), each step timed.
//! `trace`: each case pinned by `scripts/speech-traces/drama3_trace.py`
//! under `tests/fixtures/drama3/` replayed from its frame rows: the
//! backbone's input rows, its layers, final norm, and logits, and the
//! tower's rows and head logits, against the trace (relative RMS per stage,
//! argmax agreement for logits). The bounds are arguments, so the gate
//! manifest holds them. `--value-order` and `--norm-offset` select the two
//! layout conventions (drama3_runtime.Options).
//!
//!   drama3-check MODEL.gguf smoke [--frames N] [--value-order grouped|tiled] [--norm-offset X]
//!   drama3-check MODEL.gguf trace [--max-relative-rms X] [--fixtures DIR] [--value-order ...] [--norm-offset X]
const std = @import("std");
const inference = @import("inference");
const drama3 = inference.models.drama3;
const runtime = inference.models.drama3_runtime;
const weights = inference.weights;

const usage = "usage: drama3-check MODEL.gguf smoke|trace [--frames N] [--max-relative-rms X] [--fixtures DIR] [--value-order grouped|tiled] [--norm-offset X]\n";

fn elapsedUs(io: std.Io, start: std.Io.Timestamp) f64 {
    return @floatFromInt(start.durationTo(std.Io.Clock.awake.now(io)).toMicroseconds());
}

fn argmax(values: []const f32) usize {
    return std.mem.indexOfMax(f32, values);
}

/// The argmax pick: a deterministic stand-in for the sampler.
const Greedy = struct {
    fn pick(context: *anyopaque, codebook: usize, logits: []const f32) anyerror!u32 {
        _ = context;
        _ = codebook;
        return @intCast(argmax(logits));
    }
};

fn textFrame(id: u32) runtime.Frame {
    var frame: runtime.Frame = @splat(0);
    frame[0] = id;
    return frame;
}

/// `reference-bracket-v1` without a reference, tokenized by hand from the
/// file's vocabulary (seams may differ from the real tokenizer's): the
/// smoke needs plausible rows, not the prompt builder's.
const smoke_prompt = [_]u32{
    248045, 8678, 198, 13761, 279, 3766, 1414, 310, 8540, 248046, 198, // system turn
    248045, 846, 198, 248083, 220, 9419, 1814, 13, 1204, 513, 488, 30, 248046, 198, // user turn
    248045, 74455, 198, 248081, // the open voice turn
};

fn smoke(gpa: std.mem.Allocator, io: std.Io, m: *const runtime.Model, frames: usize) !bool {
    const prompt_rows = smoke_prompt.len;
    var state = try runtime.State.init(gpa, prompt_rows + frames);
    defer state.deinit();
    var r = try runtime.Runtime.init(gpa, m);
    defer r.deinit();
    var rows: [smoke_prompt.len]runtime.Frame = undefined;
    for (&rows, smoke_prompt) |*row, id| row.* = textFrame(id);
    const hidden = try gpa.alloc(f32, prompt_rows * drama3.hidden);
    defer gpa.free(hidden);
    var logits: [drama3.output_rows]f32 = undefined;

    var start = std.Io.Clock.awake.now(io);
    try r.prefill(&state, &rows, hidden, &logits, null);
    const prefill_us = elapsedUs(io, start);
    start = std.Io.Clock.awake.now(io);
    try r.towerPrefill(&state, hidden[0 .. (prompt_rows - 1) * drama3.hidden], &rows, null);
    const tower_prefill_us = elapsedUs(io, start);
    var finite = true;
    for (logits) |x| finite = finite and std.math.isFinite(x);
    var low: f32 = std.math.inf(f32);
    var high: f32 = -std.math.inf(f32);
    for (logits) |x| {
        low = @min(low, x);
        high = @max(high, x);
    }
    std.debug.print("Backbone prefill: {d} rows in {d:.0} ms ({d:.2} rows/s); tower prefill of {d} rows in {d:.0} ms ({d:.1} ms a row).\n", .{ prompt_rows, prefill_us / 1000, @as(f64, @floatFromInt(prompt_rows)) * 1e6 / prefill_us, prompt_rows - 1, tower_prefill_us / 1000, tower_prefill_us / 1000 / @as(f64, @floatFromInt(prompt_rows - 1)) });
    std.debug.print("Logits of the last row: {s}, range [{d:.3}, {d:.3}], EOS {d:.3}, argmax row {d} (id {d}).\n", .{ if (finite) "finite" else "NOT FINITE", low, high, logits[0], argmax(&logits), drama3.outputToken(argmax(&logits)) });

    var input = rows[prompt_rows - 1];
    var greedy: u8 = 0;
    const pick: runtime.Pick = .{ .context = &greedy, .pick = Greedy.pick };
    var tower_us: f64 = 0;
    var decode_us: f64 = 0;
    var produced: usize = 0;
    var mains: [16]u32 = undefined;
    while (produced < frames) {
        const token = drama3.outputToken(argmax(&logits));
        start = std.Io.Clock.awake.now(io);
        const frame = try r.step(&state, r.h, input, token, pick, null);
        tower_us += elapsedUs(io, start);
        if (produced < mains.len) mains[produced] = token;
        produced += 1;
        if (token == drama3.eos_token or produced == frames) break;
        start = std.Io.Clock.awake.now(io);
        try r.decode(&state, frame, &logits, null);
        decode_us += elapsedUs(io, start);
        for (logits) |x| finite = finite and std.math.isFinite(x);
        if (produced == 1) std.debug.print("First frame: {any}\n", .{frame});
        input = frame;
    }
    const steps: f64 = @floatFromInt(produced);
    std.debug.print("Greedy frames: {d}; tower {d:.0} ms a frame, backbone decode {d:.0} ms a row; main ids {any}.\n", .{ produced, tower_us / 1000 / steps, decode_us / 1000 / @max(steps - 1, 1), mains[0..@min(produced, mains.len)] });
    std.debug.print("State: {d} MB for {d} rows. Logits {s}.\n", .{ state.bytes() / 1_000_000, prompt_rows + frames, if (finite) "finite throughout: passed" else "NOT FINITE: FAILED" });
    return finite;
}

// --- the trace ---------------------------------------------------------

const Stats = struct { max_abs: f64 = 0, max_ref: f64 = 0, diff2: f64 = 0, ref2: f64 = 0, dot: f64 = 0, got2: f64 = 0 };

fn compare(got: []const f32, want: []const f32) Stats {
    var s: Stats = .{};
    for (got, want) |g, w| {
        const d = @as(f64, g) - w;
        s.max_abs = @max(s.max_abs, @abs(d));
        s.max_ref = @max(s.max_ref, @abs(@as(f64, w)));
        s.diff2 += d * d;
        s.ref2 += @as(f64, w) * w;
        s.dot += @as(f64, g) * w;
        s.got2 += @as(f64, g) * g;
    }
    return s;
}

/// Reads a fixture file, checking its SHA-256 against the index.
fn readFixture(arena: std.mem.Allocator, io: std.Io, dir: []const u8, entry: std.json.Value) ![]const u8 {
    const object = entry.object;
    const path = try std.fs.path.join(arena, &.{ dir, object.get("file").?.string });
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, arena, .limited(256 << 20));
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    const hex = std.fmt.bytesToHex(digest, .lower);
    if (object.get("sha256")) |want| if (!std.mem.eql(u8, &hex, want.string)) {
        std.debug.print("{s}: SHA-256 differs from the index\n", .{path});
        return error.FixtureMismatch;
    };
    return bytes;
}

fn floats(arena: std.mem.Allocator, bytes: []const u8) ![]f32 {
    const out = try arena.alloc(f32, bytes.len / 4);
    for (out, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little));
    return out;
}

/// Every layer's rows in row order, appended call by call.
const Layers = struct {
    gpa: std.mem.Allocator,
    rows: []std.ArrayList(f32),
    failed: bool = false,

    fn record(context: *anyopaque, layer: usize, x: []const f32) anyerror!void {
        const self: *Layers = @ptrCast(@alignCast(context));
        try self.rows[layer].appendSlice(self.gpa, x);
    }
    fn observer(self: *Layers) runtime.Observer {
        return .{ .context = self, .layer = record };
    }
    fn deinit(self: *Layers) void {
        for (self.rows) |*r| r.deinit(self.gpa);
    }
};

/// Forces the trace's codes on the tower, keeping the head logits it saw.
const Replay = struct {
    gpa: std.mem.Allocator,
    frame: runtime.Frame = undefined,
    logits: std.ArrayList(f32) = .empty,

    fn pick(context: *anyopaque, codebook: usize, logits: []const f32) anyerror!u32 {
        const self: *Replay = @ptrCast(@alignCast(context));
        try self.logits.appendSlice(self.gpa, logits);
        return self.frame[1 + codebook];
    }
};

const Bounds = struct { relative_rms: f64 };

/// One stage's comparison over the trace's rows, printed; whether it passed.
fn stage(label: []const u8, got_all: []const f32, want: []const f32, width: usize, first: usize, bound: f64, logits: bool) bool {
    const count = want.len / width;
    if (want.len != count * width or got_all.len < (first + count) * width) {
        std.debug.print("  {s:<20} {d} values against the trace's {d} rows of {d} from {d}: MISMATCH\n", .{ label, got_all.len, count, width, first });
        return false;
    }
    const got = got_all[first * width ..][0 .. count * width];
    const s = compare(got, want);
    const rel = @sqrt(s.diff2 / s.ref2);
    const cosine = s.dot / @sqrt(s.got2 * s.ref2);
    var same: usize = 0;
    if (logits) for (0..count) |r| {
        same += @intFromBool(argmax(got[r * width ..][0..width]) == argmax(want[r * width ..][0..width]));
    };
    const ok = rel <= bound and (!logits or same == count);
    std.debug.print("  {s:<20} {d:>4}–{d:<4} {e:>10.2} {e:>10.2} {e:>10.2} {d:>10.7}", .{ label, first, first + count, s.max_abs, s.max_ref, rel, cosine });
    if (logits) std.debug.print("  argmax {d}/{d}", .{ same, count });
    std.debug.print("{s}\n", .{if (ok) "" else "  OVER"});
    return ok;
}

fn trace(gpa: std.mem.Allocator, arena: std.mem.Allocator, io: std.Io, m: *const runtime.Model, fixtures: []const u8, bounds: Bounds) !bool {
    const index_path = try std.fs.path.join(arena, &.{ fixtures, "index.json" });
    const index_text = std.Io.Dir.cwd().readFileAlloc(io, index_path, arena, .limited(4 << 20)) catch |err| {
        std.debug.print("{s}: {s} (the trace is not pinned yet)\n", .{ index_path, @errorName(err) });
        return err;
    };
    const index = try std.json.parseFromSliceLeaky(std.json.Value, arena, index_text, .{});
    var passed = true;
    var cases = index.object.get("cases").?.object.iterator();
    while (cases.next()) |case| {
        const meta = case.value_ptr.object;
        const files = meta.get("files").?.object;
        std.debug.print("{s}:\n", .{case.key_ptr.*});
        // Frame rows, int32 [rows][11]: the prompt's, then the traced frames.
        const frame_entry = files.get("frames").?;
        const frame_bytes = try readFixture(arena, io, fixtures, frame_entry);
        const width = 1 + drama3.codebooks;
        const total = frame_bytes.len / (4 * width);
        const frames = try arena.alloc(runtime.Frame, total);
        for (frames, 0..) |*frame, r| for (frame, 0..) |*v, c| {
            v.* = @intCast(std.mem.readInt(i32, frame_bytes[(r * width + c) * 4 ..][0..4], .little));
        };
        const prompt_rows: usize = @intCast(meta.get("prompt_rows").?.integer);
        if (prompt_rows == 0 or prompt_rows > total) return error.FixtureMismatch;
        const generated = total - prompt_rows;
        const rows = prompt_rows + generated;

        var state = try runtime.State.init(gpa, rows);
        defer state.deinit();
        var r = try runtime.Runtime.init(gpa, m);
        defer r.deinit();
        var backbone: Layers = .{ .gpa = gpa, .rows = try arena.alloc(std.ArrayList(f32), drama3.layer_count) };
        @memset(backbone.rows, .empty);
        defer backbone.deinit();
        var tower: Layers = .{ .gpa = gpa, .rows = try arena.alloc(std.ArrayList(f32), drama3.shared_layers + drama3.residual_layers) };
        @memset(tower.rows, .empty);
        defer tower.deinit();
        var replay: Replay = .{ .gpa = gpa };
        defer replay.logits.deinit(gpa);

        // Backbone inputs and the prompt; then each traced frame teacher-forced.
        const inputs = try arena.alloc(f32, rows * drama3.hidden);
        const scratch = try arena.alloc(f32, drama3.hidden);
        for (frames[0..rows], 0..) |frame, i| try m.embed(frame, inputs[i * drama3.hidden ..][0..drama3.hidden], scratch);
        const norm_rows = try arena.alloc(f32, rows * drama3.hidden);
        const logits = try arena.alloc(f32, rows * drama3.output_rows);
        var start = std.Io.Clock.awake.now(io);
        try r.prefill(&state, frames[0..prompt_rows], norm_rows[0 .. prompt_rows * drama3.hidden], null, backbone.observer());
        const prefill_us = elapsedUs(io, start);
        for (0..prompt_rows) |i| try r.mainLogits(norm_rows[i * drama3.hidden ..][0..drama3.hidden], logits[i * drama3.output_rows ..][0..drama3.output_rows]);
        start = std.Io.Clock.awake.now(io);
        if (prompt_rows > 1) try r.towerPrefill(&state, norm_rows[0 .. (prompt_rows - 1) * drama3.hidden], frames[0..prompt_rows], tower.observer());
        const tower_prefill_us = elapsedUs(io, start);
        const pick: runtime.Pick = .{ .context = &replay, .pick = Replay.pick };
        var tower_us: f64 = 0;
        var decode_us: f64 = 0;
        var codes_equal = true;
        for (0..generated) |f| {
            const row = prompt_rows - 1 + f;
            replay.frame = frames[row + 1];
            start = std.Io.Clock.awake.now(io);
            const frame = try r.step(&state, norm_rows[row * drama3.hidden ..][0..drama3.hidden], frames[row], frames[row + 1][0], pick, tower.observer());
            tower_us += elapsedUs(io, start);
            codes_equal = codes_equal and std.mem.eql(u32, &frame, &frames[row + 1]);
            if (row + 1 == rows) break;
            start = std.Io.Clock.awake.now(io);
            try r.decode(&state, frames[row + 1], null, backbone.observer());
            decode_us += elapsedUs(io, start);
            @memcpy(norm_rows[(row + 1) * drama3.hidden ..][0..drama3.hidden], r.h);
            try r.mainLogits(r.h, logits[(row + 1) * drama3.output_rows ..][0..drama3.output_rows]);
        }
        // The last traced frame's backbone row is not run: no row follows it.
        const backbone_rows = @min(rows, prompt_rows + generated -| 1 + 1);
        std.debug.print("  {d} prompt rows, {d} frames; prefill {d:.2} rows/s, tower prefill {d:.0} ms a row, tower {d:.0} ms a frame, decode {d:.0} ms a row\n", .{ prompt_rows, generated, @as(f64, @floatFromInt(prompt_rows)) * 1e6 / prefill_us, tower_prefill_us / 1000 / @as(f64, @floatFromInt(@max(prompt_rows - 1, 1))), tower_us / 1000 / @as(f64, @floatFromInt(@max(generated, 1))), decode_us / 1000 / @as(f64, @floatFromInt(@max(generated -| 1, 1))) });
        std.debug.print("  frames replayed through the tower: {s}\n", .{if (codes_equal) "equal" else "DIFFER"});
        if (!codes_equal) passed = false;
        std.debug.print("  {s:<20} {s:>9} {s:>10} {s:>10} {s:>10} {s:>10}\n", .{ "stage", "rows", "max |Δ|", "max |ref|", "rel RMS", "cosine" });

        // Each pinned stage: `rows` [first, last) of a [rows][width] array, or
        // `positions` naming the rows a sparse array holds.
        var stages = files.iterator();
        while (stages.next()) |entry| {
            const key = entry.key_ptr.*;
            if (std.mem.eql(u8, key, "frames")) continue;
            const object = entry.value_ptr.object;
            const shape = object.get("shape").?.array.items;
            const w: usize = @intCast(shape[shape.len - 1].integer);
            const want = try floats(arena, try readFixture(arena, io, fixtures, entry.value_ptr.*));
            const first: usize = if (object.get("rows")) |range| @intCast(range.array.items[0].integer) else 0;
            const got: ?[]const f32 = blk: {
                if (std.mem.eql(u8, key, "backbone/inputs")) break :blk inputs;
                if (std.mem.eql(u8, key, "backbone/norm")) break :blk norm_rows[0 .. backbone_rows * drama3.hidden];
                if (std.mem.startsWith(u8, key, "backbone/layer.")) {
                    const layer = try std.fmt.parseInt(usize, key["backbone/layer.".len..], 10);
                    break :blk backbone.rows[layer].items;
                }
                if (std.mem.startsWith(u8, key, "tower/layer.")) {
                    const layer = try std.fmt.parseInt(usize, key["tower/layer.".len..], 10);
                    break :blk tower.rows[layer].items;
                }
                if (std.mem.eql(u8, key, "tower/heads")) break :blk replay.logits.items;
                if (std.mem.eql(u8, key, "backbone/logits")) break :blk logits;
                break :blk null;
            };
            const values = got orelse {
                std.debug.print("  {s:<20} not compared (unknown stage)\n", .{key});
                continue;
            };
            const is_logits = std.mem.endsWith(u8, key, "logits") or std.mem.eql(u8, key, "tower/heads");
            if (object.get("positions")) |positions| {
                // Gather the named rows into the trace's order.
                const items = positions.array.items;
                const gathered = try arena.alloc(f32, items.len * w);
                for (items, 0..) |p, k| {
                    const at: usize = @intCast(p.integer);
                    if ((at + 1) * w > values.len) return error.FixtureMismatch;
                    @memcpy(gathered[k * w ..][0..w], values[at * w ..][0..w]);
                }
                if (!stage(key, gathered, want, w, 0, bounds.relative_rms, is_logits)) passed = false;
            } else if (!stage(key, values, want, w, first, bounds.relative_rms, is_logits)) passed = false;
        }
    }
    std.debug.print("Bounds: relative RMS {e} per stage, logits argmax on every row: {s}.\n", .{ bounds.relative_rms, if (passed) "passed" else "FAILED" });
    return passed;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    const Mode = enum { smoke, trace };
    const mode = std.meta.stringToEnum(Mode, if (args.len >= 3) args[2] else "") orelse {
        std.debug.print(usage, .{});
        return error.InvalidArguments;
    };
    var bounds: Bounds = .{ .relative_rms = 1e-3 };
    var fixtures: []const u8 = "tests/fixtures/drama3";
    var frames: usize = 4;
    var options: runtime.Options = .{};
    var i: usize = 3;
    while (i + 1 < args.len) : (i += 2) {
        const value = args[i + 1];
        if (std.mem.eql(u8, args[i], "--max-relative-rms")) {
            bounds.relative_rms = try std.fmt.parseFloat(f64, value);
        } else if (std.mem.eql(u8, args[i], "--fixtures")) {
            fixtures = value;
        } else if (std.mem.eql(u8, args[i], "--frames")) {
            frames = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, args[i], "--value-order")) {
            options.value_order = std.meta.stringToEnum(runtime.ValueOrder, value) orelse return error.InvalidArguments;
        } else if (std.mem.eql(u8, args[i], "--norm-offset")) {
            options.norm_offset = try std.fmt.parseFloat(f32, value);
        } else break;
    }
    if (i != args.len or frames == 0 or frames > drama3.max_frames) {
        std.debug.print(usage, .{});
        return error.InvalidArguments;
    }
    const load_start = std.Io.Clock.awake.now(io);
    var mapped = try weights.Mapped.open(gpa, io, args[1]);
    defer mapped.deinit(io);
    var m = try runtime.Model.init(gpa, io, &mapped.document, mapped.view(), options);
    defer m.deinit();
    std.debug.print("Drama 3 bound in {d:.0} ms: {d} tensors, {d} MB (and {d} codec tensors); value heads {s}, norm offset {d}.\n", .{ elapsedUs(io, load_start) / 1000, m.binding.tensors, m.binding.bytes / 1_000_000, m.binding.codec_tensors, @tagName(options.value_order), options.norm_offset });
    const ok = switch (mode) {
        .smoke => try smoke(gpa, io, &m, frames),
        .trace => try trace(gpa, arena, io, &m, fixtures, bounds),
    };
    if (!ok) std.process.exit(1);
}
