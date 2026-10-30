//! Explicit full-model check of Drama 3's backbone and tower, on the CPU and on Metal.
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
//!   drama3-check MODEL.gguf codec [--max-relative-rms X] [--fixtures DIR]
//!   drama3-check MODEL.gguf metal [--frames N] [--chunk N] [--max-relative-rms X] [--value-order ...]
//!   drama3-check MODEL.gguf speak --text T --out FILE.wav [--seed N] [--frames N] [--backend cpu|metal]
//! `metal`: the smoke prompt and N argmax frames recorded on the CPU, then
//! replayed through `drama3_metal.Plan` (prefill chunks of `--chunk` rows),
//! once with every kernel the generic decoder and once with the production
//! kernels, each within the bound (max relative RMS per stage); then timed.
//! `codec`: Fish's codec cases through the CPU encoder and decoder
//! (drama3_codec_check.zig).
//! `speak` runs `drama3_speech.Synthesizer` end to end (no reference voice)
//! and writes a 44.1 kHz 16-bit WAV.
const std = @import("std");
const inference = @import("inference");
const drama3 = inference.models.drama3;
const runtime = inference.models.drama3_runtime;
const weights = inference.weights;
const codec_check = @import("drama3_codec_check.zig");

const usage = "usage: drama3-check MODEL.gguf smoke|trace|codec|metal|speak [--text T] [--out FILE] [--seed N] [--frames N] [--chunk N] [--backend cpu|metal] [--max-relative-rms X] [--fixtures DIR] [--value-order grouped|tiled] [--norm-offset X]\n";

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

/// The rows of `values` (`[rows][width]`) a fixture entry holds, in its
/// order: its `rows` ranges ([start, end) pairs, or one pair), its
/// `positions`, or all of them; null when `values` lacks one.
fn select(arena: std.mem.Allocator, values: []const f32, width: usize, object: std.json.ObjectMap) !?[]const f32 {
    var picked: std.ArrayList(f32) = .empty;
    const total = values.len / width;
    if (object.get("positions")) |positions| {
        for (positions.array.items) |p| {
            const at: usize = @intCast(p.integer);
            if (at >= total) return null;
            try picked.appendSlice(arena, values[at * width ..][0..width]);
        }
        return picked.items;
    }
    const ranges = (object.get("rows") orelse return values).array.items;
    // One [start, end) pair, or a list of them.
    const single = ranges.len == 2 and ranges[0] == .integer;
    for (0..if (single) 1 else ranges.len) |k| {
        const pair = if (single) ranges else ranges[k].array.items;
        const first: usize = @intCast(pair[0].integer);
        const last: usize = @intCast(pair[1].integer);
        if (last > total or first > last) return null;
        try picked.appendSlice(arena, values[first * width .. last * width]);
    }
    return picked.items;
}

/// One stage's comparison, printed; whether it passed.
fn stage(label: []const u8, got: []const f32, want: []const f32, width: usize, bound: f64, logits: bool) bool {
    const count = want.len / width;
    if (want.len != count * width or got.len != want.len) {
        std.debug.print("  {s:<20} {d} values against the trace's {d}: MISMATCH\n", .{ label, got.len, want.len });
        return false;
    }
    const s = compare(got, want);
    const rel = @sqrt(s.diff2 / s.ref2);
    const cosine = s.dot / @sqrt(s.got2 * s.ref2);
    var same: usize = 0;
    if (logits) for (0..count) |r| {
        same += @intFromBool(argmax(got[r * width ..][0..width]) == argmax(want[r * width ..][0..width]));
    };
    const ok = rel <= bound and (!logits or same == count);
    std.debug.print("  {s:<20} {d:>6} {e:>10.2} {e:>10.2} {e:>10.2} {d:>10.7}", .{ label, count, s.max_abs, s.max_ref, rel, cosine });
    if (logits) std.debug.print("  argmax {d}/{d}", .{ same, count });
    std.debug.print("{s}\n", .{if (ok) "" else "  OVER"});
    if (!ok) {
        // The rows that carry the error, by their relative RMS.
        std.debug.print("    worst rows (index in the trace's order):", .{});
        var shown: [4]usize = @splat(count);
        for (&shown) |*slot| {
            var worst: f64 = -1;
            for (0..count) |i| {
                if (std.mem.indexOfScalar(usize, &shown, i) != null) continue;
                const t = compare(got[i * width ..][0..width], want[i * width ..][0..width]);
                const e = @sqrt(t.diff2 / t.ref2);
                if (e > worst) {
                    worst = e;
                    slot.* = i;
                }
            }
            if (slot.* < count) std.debug.print(" {d} ({e:.2})", .{ slot.*, worst });
        }
        std.debug.print("\n", .{});
    }
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
    var checked: usize = 0;
    var cases = index.object.get("cases").?.object.iterator();
    while (cases.next()) |case| {
        const meta = case.value_ptr.object;
        const files = meta.get("files").?.object;
        // Frame rows, int32 [rows][11]: the prompt's, then the traced frames.
        // Cases without them (the codec's) are another check's.
        const frame_entry = files.get("frames") orelse continue;
        checked += 1;
        std.debug.print("{s}:\n", .{case.key_ptr.*});
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
            start = std.Io.Clock.awake.now(io);
            try r.decode(&state, frames[row + 1], null, backbone.observer());
            decode_us += elapsedUs(io, start);
            @memcpy(norm_rows[(row + 1) * drama3.hidden ..][0..drama3.hidden], r.h);
            try r.mainLogits(r.h, logits[(row + 1) * drama3.output_rows ..][0..drama3.output_rows]);
        }
        std.debug.print("  {d} prompt rows, {d} frames; prefill {d:.2} rows/s, tower prefill {d:.0} ms a row, tower {d:.0} ms a frame, decode {d:.0} ms a row\n", .{ prompt_rows, generated, @as(f64, @floatFromInt(prompt_rows)) * 1e6 / prefill_us, tower_prefill_us / 1000 / @as(f64, @floatFromInt(@max(prompt_rows - 1, 1))), tower_us / 1000 / @as(f64, @floatFromInt(@max(generated, 1))), decode_us / 1000 / @as(f64, @floatFromInt(@max(generated -| 1, 1))) });
        std.debug.print("  frames replayed through the tower: {s}\n", .{if (codes_equal) "equal" else "DIFFER"});
        if (!codes_equal) passed = false;
        // Every tower row's head logits from its residual outputs; the rows
        // `step` ran must equal what it handed the pick.
        const tower_rows = tower.rows[drama3.shared_layers].items.len / drama3.hidden;
        const heads = try arena.alloc(f32, tower_rows * drama3.residual_layers * drama3.codebook_size);
        const residual = try arena.alloc(f32, tower_rows * drama3.residual_layers * drama3.hidden);
        for (0..tower_rows) |t| for (0..drama3.residual_layers) |j| {
            const h = tower.rows[drama3.shared_layers + j].items[t * drama3.hidden ..][0..drama3.hidden];
            @memcpy(residual[(t * drama3.residual_layers + j) * drama3.hidden ..][0..drama3.hidden], h);
            try r.headLogits(j, h, heads[(t * drama3.residual_layers + j) * drama3.codebook_size ..][0..drama3.codebook_size]);
        };
        const stepped = heads[(prompt_rows - 1) * drama3.residual_layers * drama3.codebook_size ..];
        const same_heads = std.mem.eql(f32, stepped, replay.logits.items);
        std.debug.print("  the step's head logits against the heads on its recorded rows: {s}\n", .{if (same_heads) "equal" else "DIFFER"});
        if (!same_heads) passed = false;
        std.debug.print("  {s:<20} {s:>6} {s:>10} {s:>10} {s:>10} {s:>10}\n", .{ "stage", "rows", "max |Δ|", "max |ref|", "rel RMS", "cosine" });

        // Each pinned stage, `[rows][...][width]`; `rows` index the first axis.
        var stages = files.iterator();
        while (stages.next()) |entry| {
            const key = entry.key_ptr.*;
            if (!std.mem.startsWith(u8, key, "backbone/") and !std.mem.startsWith(u8, key, "tower/")) continue;
            const object = entry.value_ptr.object;
            const shape = object.get("shape").?.array.items;
            const w: usize = @intCast(shape[shape.len - 1].integer);
            var row_width: usize = 1;
            for (shape[1..]) |d| row_width *= @intCast(d.integer);
            const values: []const f32 = blk: {
                if (std.mem.eql(u8, key, "backbone/inputs")) break :blk inputs;
                if (std.mem.eql(u8, key, "backbone/norm")) break :blk norm_rows;
                if (std.mem.eql(u8, key, "backbone/logits")) break :blk logits;
                if (std.mem.startsWith(u8, key, "backbone/layer.")) {
                    const layer = std.fmt.parseInt(usize, key["backbone/layer.".len..], 10) catch break :blk &.{};
                    if (layer < backbone.rows.len) break :blk backbone.rows[layer].items;
                }
                // The shared stack's output is shared layer 3's.
                if (std.mem.eql(u8, key, "tower/shared")) break :blk tower.rows[drama3.shared_layers - 1].items;
                if (std.mem.eql(u8, key, "tower/residual")) break :blk residual;
                if (std.mem.eql(u8, key, "tower/logits")) break :blk heads;
                break :blk &.{};
            };
            if (values.len == 0 or w == 0) {
                std.debug.print("  {s:<20} not compared (no such stage here)\n", .{key});
                passed = false;
                continue;
            }
            const want = try floats(arena, try readFixture(arena, io, fixtures, entry.value_ptr.*));
            const got = try select(arena, values, row_width, object) orelse {
                std.debug.print("  {s:<20} rows outside the {d} computed: MISMATCH\n", .{ key, values.len / row_width });
                passed = false;
                continue;
            };
            if (!stage(key, got, want, w, bounds.relative_rms, std.mem.endsWith(u8, key, "logits"))) passed = false;
        }
    }
    if (checked == 0) {
        std.debug.print("{s}: no case pins frame rows (the backbone's trace is not pinned yet)\n", .{index_path});
        return false;
    }
    std.debug.print("Bounds: relative RMS {e} per stage, logits argmax on every row: {s}.\n", .{ bounds.relative_rms, if (passed) "passed" else "FAILED" });
    return passed;
}

// --- Metal against the CPU ---------------------------------------------

/// Max relative RMS of one stage over the rows compared, and the rows whose
/// argmax agreed (logits only).
const Agreement = struct {
    name: []const u8,
    logits: bool,
    worst: f64 = 0,
    rows: usize = 0,
    same: usize = 0,

    fn add(self: *Agreement, got: []const f32, want: []const f32, width: usize) void {
        for (0..got.len / width) |r| {
            const s = compare(got[r * width ..][0..width], want[r * width ..][0..width]);
            self.worst = @max(self.worst, @sqrt(s.diff2 / s.ref2));
            self.rows += 1;
            if (self.logits and argmax(got[r * width ..][0..width]) == argmax(want[r * width ..][0..width])) self.same += 1;
        }
    }
};

/// The CPU's run, recorded for the Metal runs to replay: the prefill's
/// hidden rows and logits, then per frame the decode row's hidden and
/// logits (from the second frame on) and each codebook head's logits.
const Reference = struct {
    hidden: []f32,
    logits: [drama3.output_rows]f32 = undefined,
    frames: std.ArrayList(runtime.Frame) = .empty,
    decode_hidden: std.ArrayList(f32) = .empty,
    decode_logits: std.ArrayList(f32) = .empty,
    heads: std.ArrayList(f32) = .empty,
    gpa: std.mem.Allocator,

    /// The argmax pick, every codebook's logits kept.
    fn pick(context: *anyopaque, codebook: usize, logits: []const f32) anyerror!u32 {
        _ = codebook;
        const self: *Reference = @ptrCast(@alignCast(context));
        try self.heads.appendSlice(self.gpa, logits);
        return @intCast(argmax(logits));
    }
};

/// The Metal side of a tower step: compared against the CPU's logits of
/// the same codebook, then the CPU's code, so both sides run the same frames.
const Follower = struct {
    reference: *const Reference,
    frame: usize = 0,
    heads: *[drama3.residual_layers]Agreement,
    fn pick(context: *anyopaque, codebook: usize, logits: []const f32) anyerror!u32 {
        const self: *Follower = @ptrCast(@alignCast(context));
        const want = self.reference.heads.items[(self.frame * drama3.residual_layers + codebook - 1) * drama3.codebook_size ..][0..drama3.codebook_size];
        self.heads[codebook - 1].add(logits, want, drama3.codebook_size);
        return @intCast(argmax(want));
    }
};

/// The smoke prompt and `frames` frames on the CPU, recorded with argmax
/// picks (the main token from the CPU's logits).
fn reference(gpa: std.mem.Allocator, m: *const runtime.Model, frames: usize, rows: []const runtime.Frame) !Reference {
    var state = try runtime.State.init(gpa, rows.len + frames);
    defer state.deinit();
    var r = try runtime.Runtime.init(gpa, m);
    defer r.deinit();
    var out: Reference = .{ .gpa = gpa, .hidden = try gpa.alloc(f32, rows.len * drama3.hidden) };
    try r.prefill(&state, rows, out.hidden, &out.logits, null);
    try r.towerPrefill(&state, out.hidden[0 .. (rows.len - 1) * drama3.hidden], rows, null);
    var logits = out.logits;
    var input = rows[rows.len - 1];
    while (out.frames.items.len < frames) {
        const token = drama3.outputToken(argmax(&logits));
        const frame = try r.step(&state, r.h, input, token, .{ .context = &out, .pick = Reference.pick }, null);
        try out.frames.append(gpa, frame);
        if (token == drama3.eos_token or out.frames.items.len == frames) break;
        try r.decode(&state, frame, &logits, null);
        try out.decode_hidden.appendSlice(gpa, r.h);
        try out.decode_logits.appendSlice(gpa, &logits);
        input = frame;
    }
    return out;
}

/// The recorded run replayed on Metal (`generic`: every kernel the generic
/// decoder, `Backend.generic_only`): per-stage max relative RMS against
/// `bound`, and the plan's times.
fn metalRun(gpa: std.mem.Allocator, io: std.Io, gpu: *inference.metal.Backend, m: *const runtime.Model, ref: *const Reference, rows: []const runtime.Frame, chunk: usize, generic: bool, bound: f64) !bool {
    gpu.generic_only = generic;
    const frames = ref.frames.items.len;
    var plan = try inference.models.drama3_metal.Plan.init(gpa, gpu, m, rows.len + frames, chunk);
    defer plan.deinit();
    std.debug.print("Metal, {s} kernels, chunk {d} ({d} MB of device buffers and state):\n", .{ if (generic) "generic" else "production", chunk, plan.bytes() / 1_000_000 });
    var stages = [_]Agreement{
        .{ .name = "prefill hidden", .logits = false },
        .{ .name = "prefill logits", .logits = true },
        .{ .name = "decode hidden", .logits = false },
        .{ .name = "decode logits", .logits = true },
    };
    var heads: [drama3.residual_layers]Agreement = undefined;
    const head_names = [_][]const u8{ "codebook 1 logits", "codebook 2 logits", "codebook 3 logits", "codebook 4 logits", "codebook 5 logits", "codebook 6 logits", "codebook 7 logits", "codebook 8 logits", "codebook 9 logits" };
    for (&heads, head_names) |*a, name| a.* = .{ .name = name, .logits = true };

    const hidden = try gpa.alloc(f32, rows.len * drama3.hidden);
    defer gpa.free(hidden);
    var logits: [drama3.output_rows]f32 = undefined;
    try plan.prefill(rows, hidden, &logits);
    stages[0].add(hidden, ref.hidden, drama3.hidden);
    stages[1].add(&logits, &ref.logits, drama3.output_rows);
    try plan.towerPrefill(hidden[0 .. (rows.len - 1) * drama3.hidden], rows);
    var follower: Follower = .{ .reference = ref, .heads = &heads };
    var input = rows[rows.len - 1];
    var frames_equal = true;
    for (ref.frames.items, 0..) |want, f| {
        follower.frame = f;
        const frame = try plan.step(plan.h, input, want[0], .{ .context = &follower, .pick = Follower.pick });
        frames_equal = frames_equal and std.mem.eql(u32, &frame, &want);
        if (f + 1 == frames) break;
        try plan.decode(want, &logits);
        stages[2].add(plan.h, ref.decode_hidden.items[f * drama3.hidden ..][0..drama3.hidden], drama3.hidden);
        stages[3].add(&logits, ref.decode_logits.items[f * drama3.output_rows ..][0..drama3.output_rows], drama3.output_rows);
        input = want;
    }
    var passed = frames_equal;
    std.debug.print("  {s:<20} {s:>5} {s:>10} {s:>8}\n", .{ "stage", "rows", "rel RMS", "argmax" });
    for (&stages ++ heads) |a| {
        const ok = a.worst <= bound;
        passed = passed and ok;
        std.debug.print("  {s:<20} {d:>5} {e:>10.2}", .{ a.name, a.rows, a.worst });
        if (a.logits) std.debug.print(" {d:>4}/{d}", .{ a.same, a.rows });
        std.debug.print("{s}\n", .{if (ok) "" else "  OVER"});
    }
    std.debug.print("  frames {s}; bound {e}: {s}\n", .{ if (frames_equal) "equal" else "DIFFERENT", bound, if (passed) "passed" else "FAILED" });

    // Times, on the warmed plan: the same prompt and frames again.
    plan.reset();
    var start = std.Io.Clock.awake.now(io);
    try plan.prefill(rows, hidden, &logits);
    const prefill_us = elapsedUs(io, start);
    start = std.Io.Clock.awake.now(io);
    try plan.towerPrefill(hidden[0 .. (rows.len - 1) * drama3.hidden], rows);
    const tower_prefill_us = elapsedUs(io, start);
    input = rows[rows.len - 1];
    var step_us: f64 = 0;
    var decode_us: f64 = 0;
    for (ref.frames.items, 0..) |want, f| {
        follower.frame = f;
        start = std.Io.Clock.awake.now(io);
        _ = try plan.step(plan.h, input, want[0], .{ .context = &follower, .pick = Follower.pick });
        step_us += elapsedUs(io, start);
        if (f + 1 == frames) break;
        start = std.Io.Clock.awake.now(io);
        try plan.decode(want, &logits);
        decode_us += elapsedUs(io, start);
        input = want;
    }
    const steps: f64 = @floatFromInt(frames);
    std.debug.print("  times: prefill {d} rows {d:.1} ms ({d:.0} rows/s), tower prefill {d:.1} ms; tower {d:.1} ms a frame, backbone decode {d:.1} ms a row\n", .{ rows.len, prefill_us / 1000, @as(f64, @floatFromInt(rows.len)) * 1e6 / prefill_us, tower_prefill_us / 1000, step_us / 1000 / steps, decode_us / 1000 / @max(steps - 1, 1) });
    return passed;
}

/// The smoke prompt and `frames` frames on the CPU and on Metal from one
/// mapping, the CPU's argmax picks driving both: Metal with every kernel
/// the generic decoder, then with the production kernels, each within `bound`.
fn metalCheck(gpa: std.mem.Allocator, io: std.Io, m: *const runtime.Model, frames: usize, chunk: usize, bound: f64) !bool {
    var rows: [smoke_prompt.len]runtime.Frame = undefined;
    for (&rows, smoke_prompt) |*row, id| row.* = textFrame(id);
    var ref = try reference(gpa, m, frames, &rows);
    defer {
        gpa.free(ref.hidden);
        ref.frames.deinit(gpa);
        ref.decode_hidden.deinit(gpa);
        ref.decode_logits.deinit(gpa);
        ref.heads.deinit(gpa);
    }
    std.debug.print("CPU reference: {d} prompt rows, {d} frames.\n", .{ rows.len, ref.frames.items.len });
    var message: [512]u8 = @splat(0);
    var gpu = inference.metal.Backend.init(gpa, &message) catch |err| {
        std.debug.print("Metal: {s} {s}\n", .{ @errorName(err), std.mem.sliceTo(&message, 0) });
        return err;
    };
    defer gpu.deinit();
    const exact = try metalRun(gpa, io, &gpu, m, &ref, &rows, chunk, true, bound);
    const production = try metalRun(gpa, io, &gpu, m, &ref, &rows, chunk, false, bound);
    return exact and production;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    const Mode = enum { smoke, trace, codec, speak, metal };
    const mode = std.meta.stringToEnum(Mode, if (args.len >= 3) args[2] else "") orelse {
        std.debug.print(usage, .{});
        return error.InvalidArguments;
    };
    var bounds: Bounds = .{ .relative_rms = 1e-3 };
    var fixtures: []const u8 = "tests/fixtures/drama3";
    var frames: usize = 4;
    var options: runtime.Options = .{};
    var text: []const u8 = "Hello, this is a test.";
    var out_path: []const u8 = "drama3.wav";
    var seed: u64 = 0;
    var chunk: usize = 24;
    var on_metal = false;
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
        } else if (std.mem.eql(u8, args[i], "--text")) {
            text = value;
        } else if (std.mem.eql(u8, args[i], "--out")) {
            out_path = value;
        } else if (std.mem.eql(u8, args[i], "--seed")) {
            seed = try std.fmt.parseInt(u64, value, 10);
        } else if (std.mem.eql(u8, args[i], "--norm-offset")) {
            options.norm_offset = try std.fmt.parseFloat(f32, value);
        } else if (std.mem.eql(u8, args[i], "--chunk")) {
            chunk = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, args[i], "--backend")) {
            on_metal = std.mem.eql(u8, value, "metal");
            if (!on_metal and !std.mem.eql(u8, value, "cpu")) return error.InvalidArguments;
        } else break;
    }
    if (i != args.len or frames == 0 or frames > drama3.max_frames) {
        std.debug.print(usage, .{});
        return error.InvalidArguments;
    }
    if (mode == .speak) return speak(gpa, io, args[1], text, out_path, seed, frames, if (on_metal) .metal else .cpu);
    const load_start = std.Io.Clock.awake.now(io);
    var mapped = try weights.Mapped.open(gpa, io, args[1]);
    defer mapped.deinit(io);
    var m = try runtime.Model.init(gpa, io, &mapped.document, mapped.view(), options);
    defer m.deinit();
    std.debug.print("Drama 3 bound in {d:.0} ms: {d} tensors, {d} MB (and {d} codec tensors); value heads {s}, norm offset {d}.\n", .{ elapsedUs(io, load_start) / 1000, m.binding.tensors, m.binding.bytes / 1_000_000, m.binding.codec_tensors, @tagName(options.value_order), options.norm_offset });
    const ok = switch (mode) {
        .smoke => try smoke(gpa, io, &m, frames),
        .trace => try trace(gpa, arena, io, &m, fixtures, bounds),
        .codec => try codec_check.run(gpa, arena, io, &mapped, fixtures, bounds.relative_rms),
        .metal => try metalCheck(gpa, io, &m, frames, chunk, bounds.relative_rms),
        .speak => unreachable,
    };
    if (!ok) std.process.exit(1);
}

fn speak(gpa: std.mem.Allocator, io: std.Io, path: []const u8, text: []const u8, out_path: []const u8, seed: u64, frames: usize, backend: inference.models.drama3_speech.Backend) !void {
    const speech = inference.models.drama3_speech;
    const load_start = std.Io.Clock.awake.now(io);
    const voice = try speech.Voice.openWith(gpa, io, path, .{ .backend = backend });
    defer voice.close();
    std.debug.print("loaded in {d:.0} ms\n", .{elapsedUs(io, load_start) / 1000});
    var synth = try speech.Synthesizer.init(gpa, voice, .{ .text = text, .seed = seed, .max_frames = @max(frames, 64) });
    defer synth.deinit();
    var samples: std.ArrayList(i16) = .empty;
    defer samples.deinit(gpa);
    var buffer: [8192]f32 = undefined;
    const start = std.Io.Clock.awake.now(io);
    while (true) {
        const n = try synth.read(&buffer);
        if (n == 0) break;
        for (buffer[0..n]) |x| try samples.append(gpa, @intFromFloat(@round(std.math.clamp(x, -1, 1) * 32767)));
    }
    const file = try std.Io.Dir.cwd().createFile(io, out_path, .{});
    defer file.close(io);
    var header: [44]u8 = undefined;
    const data: u32 = @intCast(samples.items.len * 2);
    @memcpy(header[0..4], "RIFF");
    std.mem.writeInt(u32, header[4..8], 36 + data, .little);
    @memcpy(header[8..16], "WAVEfmt ");
    std.mem.writeInt(u32, header[16..20], 16, .little);
    std.mem.writeInt(u16, header[20..22], 1, .little);
    std.mem.writeInt(u16, header[22..24], 1, .little);
    std.mem.writeInt(u32, header[24..28], speech.sample_rate, .little);
    std.mem.writeInt(u32, header[28..32], speech.sample_rate * 2, .little);
    std.mem.writeInt(u16, header[32..34], 2, .little);
    std.mem.writeInt(u16, header[34..36], 16, .little);
    @memcpy(header[36..40], "data");
    std.mem.writeInt(u32, header[40..44], data, .little);
    try file.writePositionalAll(io, &header, 0);
    try file.writePositionalAll(io, std.mem.sliceAsBytes(samples.items), 44);
    const t = synth.timing;
    std.debug.print("wrote {s}: {d} frames, {d:.2} s of audio in {d:.1} s; chunks {d}, retried {d}, stops eos/silence/length {any}; prefill {d:.1} s ({d} rows), frames {d:.1} s, codec {d:.1} s\n", .{
        out_path,                                                        synth.frames,
        @as(f64, @floatFromInt(samples.items.len)) / speech.sample_rate, elapsedUs(io, start) / 1e6,
        synth.chunks.len,                                                synth.retried,
        synth.stops,                                                     @as(f64, @floatFromInt(t.prefill_ns)) / 1e9,
        t.prompt_rows,                                                   @as(f64, @floatFromInt(t.frames_ns)) / 1e9,
        @as(f64, @floatFromInt(t.codec_ns)) / 1e9,
    });
}
