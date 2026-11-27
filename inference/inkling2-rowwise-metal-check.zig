//! `make test-metal`'s cases for the row-wise operations Inkling 2 adds, each
//! against the CPU function it stands for at the model's geometry: the
//! short convolution (static and per-head dynamic taps, the history read
//! and then advanced), `l2Clip` with per-channel gains and the log position
//! scale, the depth mix, the message controller, SITU, the biased
//! sigmoid router, and the relative-position bias table.
const std = @import("std");
const inference = @import("inference");
const check_stats = @import("check_stats.zig");
const cpu = inference.cpu;
const Backend = inference.metal.Backend;
const Buffer = inference.metal.Buffer;

const embedding = 2304;
const kv_width = 1024;
const kv_heads = 8;
const heads = 24;
const head_size = 128;
const taps = 4;

fn upload(b: *Backend, values: []const f32) !Buffer {
    const buffer = try b.create(@max(4, values.len * 4));
    @memcpy(buffer.floats()[0..values.len], values);
    return buffer;
}

fn fill(random: std.Random, values: []f32, scale: f32) void {
    for (values) |*v| v.* = random.floatNorm(f32) * scale;
}

/// Largest |got − want| / max(1, |want|) after checking every element
/// against `bound`.
fn expectNear(name: []const u8, got: []const f32, want: []const f32, bound: f64) !f64 {
    var worst: check_stats.Worst = .{};
    for (got, want, 0..) |g, w, i| {
        const d = @abs(@as(f64, g) - w) / @max(1, @abs(@as(f64, w)));
        worst.observe(d);
        if (!(d <= bound)) {
            std.debug.print("{s}: element {d} got {d} want {d} (bound {e})\n", .{ name, i, g, w, bound });
            return error.MetalMismatch;
        }
    }
    return worst.shown();
}

/// The convolution over `rows` rows in one dispatch against `rows`
/// sequential CPU steps, the outputs and the advanced history.
fn checkShortConvolution(alloc: std.mem.Allocator, b: *Backend, random: std.Random, channels: usize, dynamic_heads: ?usize, rows: usize) !f64 {
    const stride = channels + 32;
    const history = try alloc.alloc(f32, channels * (taps - 1));
    defer alloc.free(history);
    fill(random, history, 1);
    const weights = try alloc.alloc(f32, channels * taps);
    defer alloc.free(weights);
    fill(random, weights, 0.5);
    const input = try alloc.alloc(f32, rows * stride);
    defer alloc.free(input);
    fill(random, input, 2);
    const dyn_stride = if (dynamic_heads) |h| h * taps + 3 else 0;
    const dynamic = try alloc.alloc(f32, @max(1, rows * dyn_stride));
    defer alloc.free(dynamic);
    fill(random, dynamic, 0.7);

    const history_buffer = try upload(b, history);
    defer b.release(history_buffer) catch {};
    const weights_buffer = try upload(b, weights);
    defer b.release(weights_buffer) catch {};
    const input_buffer = try upload(b, input);
    defer b.release(input_buffer) catch {};
    const dynamic_buffer = try upload(b, dynamic);
    defer b.release(dynamic_buffer) catch {};
    const out_stride = channels + 96;
    const out = try b.create(rows * out_stride * 4);
    defer b.release(out) catch {};
    @memset(out.floats(), std.math.nan(f32));
    const d: ?Backend.DynamicTaps = if (dynamic_heads) |h| .{ .taps = dynamic_buffer, .heads = h, .stride = dyn_stride } else null;
    try b.begin();
    try b.shortConvolution(history_buffer, input_buffer, weights_buffer, d, out, channels, taps, rows, stride, out_stride);
    try b.convolutionHistory(history_buffer, input_buffer, channels, taps, rows, stride);
    try b.commit();

    const scratch = try alloc.alloc(f64, channels);
    defer alloc.free(scratch);
    const want = try alloc.alloc(f32, channels);
    defer alloc.free(want);
    var worst: f64 = 0;
    for (0..rows) |r| {
        const dyn: ?cpu.recurrent.DynamicTaps = if (dynamic_heads) |h| .{ .taps = dynamic[r * dyn_stride ..][0 .. h * taps], .heads = h } else null;
        try cpu.recurrent.shortConvolution(input[r * stride ..][0..channels], weights, dyn, history, want, taps, scratch);
        worst = @max(worst, try expectNear("short convolution", out.floats()[r * out_stride ..][0..channels], want, 2e-6));
        // The output padding is untouched.
        if (r + 1 < rows and !std.math.isNan(out.floats()[r * out_stride + channels])) return error.MetalMismatch;
    }
    if (!std.mem.eql(f32, history, history_buffer.floats()[0..history.len])) {
        std.debug.print("short convolution: the advanced history differs from the CPU's\n", .{});
        return error.MetalMismatch;
    }
    return worst;
}

/// The clip on `rows` rows of `segments` heads of `width`, some segments
/// longer than the threshold, with gains and the log scale when asked.
fn checkClip(alloc: std.mem.Allocator, b: *Backend, random: std.Random, segments: usize, width: usize, threshold: f32, gained: bool, log_scale: ?Backend.LogScale) !f64 {
    const rows = 3;
    const row_stride = segments * width + 16;
    const data = try alloc.alloc(f32, rows * row_stride);
    defer alloc.free(data);
    for (data, 0..) |*x, i| x.* = random.floatNorm(f32) * @as(f32, if ((i / width) % 3 == 0) 0.2 else 6);
    const gains = try alloc.alloc(f32, segments * width);
    defer alloc.free(gains);
    for (gains) |*g| g.* = 0.5 + random.float(f32);
    const data_buffer = try upload(b, data);
    defer b.release(data_buffer) catch {};
    const gain_buffer = try upload(b, gains);
    defer b.release(gain_buffer) catch {};
    try b.begin();
    try b.l2ClipRows(data_buffer, if (gained) gain_buffer else null, threshold, log_scale, segments, width, rows, row_stride);
    try b.commit();
    const want = try alloc.alloc(f32, segments * width);
    defer alloc.free(want);
    var worst: f64 = 0;
    for (0..rows) |r| {
        for (0..segments) |h| {
            const x = data[r * row_stride + h * width ..][0..width];
            try cpu.l2Clip(x, if (gained) gains[h * width ..][0..width] else null, threshold, want[h * width ..][0..width]);
        }
        if (log_scale) |l| {
            const lambda = cpu.logPositionScale(l.position + r, l.floor, l.alpha);
            for (want) |*x| x.* *= lambda;
        }
        worst = @max(worst, try expectNear("l2 clip", data_buffer.floats()[r * row_stride ..][0 .. segments * width], want, 2e-6));
        // The padding after the heads is untouched.
        if (!std.mem.eql(f32, data[r * row_stride + segments * width ..][0..16], data_buffer.floats()[r * row_stride + segments * width ..][0..16])) return error.MetalMismatch;
    }
    return worst;
}

/// The mix for three tokens of `count` states with `queries` shared
/// queries, strided state and output rows.
fn checkMix(alloc: std.mem.Allocator, b: *Backend, random: std.Random, count: usize, queries: usize) !f64 {
    const tokens = 3;
    const state_stride = count * embedding + 64;
    const out_stride = queries * embedding + 32;
    const states = try alloc.alloc(f32, tokens * state_stride);
    defer alloc.free(states);
    // States of different scales: the mix reads each over its own RMS.
    for (states, 0..) |*x, i| x.* = random.floatNorm(f32) * @as(f32, @floatFromInt(1 + (i / embedding) % 5));
    const q = try alloc.alloc(f32, queries * embedding);
    defer alloc.free(q);
    fill(random, q, 1);
    const states_buffer = try upload(b, states);
    defer b.release(states_buffer) catch {};
    const q_buffer = try upload(b, q);
    defer b.release(q_buffer) catch {};
    const out = try b.create(tokens * out_stride * 4);
    defer b.release(out) catch {};
    try b.begin();
    try b.depthMix(states_buffer, q_buffer, out, embedding, count, queries, tokens, state_stride, out_stride, 1e-6);
    try b.commit();
    const want = try alloc.alloc(f32, queries * embedding);
    defer alloc.free(want);
    const scratch = try alloc.alloc(f64, 2 * count);
    defer alloc.free(scratch);
    var worst: f64 = 0;
    for (0..tokens) |t| {
        try cpu.depth.mix(states[t * state_stride ..][0 .. count * embedding], q, embedding, 1e-6, want, scratch);
        worst = @max(worst, try expectNear("depth mix", out.floats()[t * out_stride ..][0 .. queries * embedding], want, 2e-6));
    }
    return worst;
}

/// The controller as the plan runs it (join, the three products on
/// `matvec`, blend into `current` in place) against `cpu.depth.control`,
/// F32 matrices at the model's rank, two rows with a different stride for
/// every buffer.
fn checkControl(alloc: std.mem.Allocator, io: std.Io, b: *Backend, random: std.Random) !f64 {
    const rank = 64;
    const rows = 2;
    const current_stride = embedding + 32;
    const message_stride = embedding + 64;
    const joined_stride = 2 * embedding + 16;
    const gate_stride = embedding + 48;
    const down = try alloc.alloc(f32, rank * 2 * embedding);
    defer alloc.free(down);
    fill(random, down, 0.03);
    const gain = try alloc.alloc(f32, embedding * rank);
    defer alloc.free(gain);
    fill(random, gain, 0.3);
    const forget = try alloc.alloc(f32, embedding * rank);
    defer alloc.free(forget);
    fill(random, forget, 0.3);
    const current = try alloc.alloc(f32, rows * current_stride);
    defer alloc.free(current);
    fill(random, current, 3);
    const message = try alloc.alloc(f32, rows * message_stride);
    defer alloc.free(message);
    fill(random, message, 0.5);
    const matrix = struct {
        fn of(r: usize, columns: usize, values: []const f32) cpu.Matrix {
            return .{ .encoding = 0, .rows = r, .columns = columns, .bytes = std.mem.sliceAsBytes(values) };
        }
    }.of;
    const c: cpu.depth.Controller = .{ .down = matrix(rank, 2 * embedding, down), .gain = matrix(embedding, rank, gain), .forget = matrix(embedding, rank, forget) };
    const scratch = try alloc.alloc(f32, c.scratchLen());
    defer alloc.free(scratch);
    const want = try alloc.alloc(f32, rows * embedding);
    defer alloc.free(want);
    for (0..rows) |r| try cpu.depth.control(io, c, current[r * current_stride ..][0..embedding], message[r * message_stride ..][0..embedding], 1e-6, want[r * embedding ..][0..embedding], scratch);

    const down_buffer = try upload(b, down);
    defer b.release(down_buffer) catch {};
    const gain_buffer = try upload(b, gain);
    defer b.release(gain_buffer) catch {};
    const forget_buffer = try upload(b, forget);
    defer b.release(forget_buffer) catch {};
    const current_buffer = try upload(b, current);
    defer b.release(current_buffer) catch {};
    const message_buffer = try upload(b, message);
    defer b.release(message_buffer) catch {};
    const joined = try b.create(rows * joined_stride * 4);
    defer b.release(joined) catch {};
    const rank_buffer = try b.create(rows * rank * 4);
    defer b.release(rank_buffer) catch {};
    const gains = try b.create(rows * gate_stride * 4);
    defer b.release(gains) catch {};
    const forgets = try b.create(rows * gate_stride * 4);
    defer b.release(forgets) catch {};
    try b.begin();
    try b.controlJoin(current_buffer, message_buffer, joined, embedding, rows, current_stride, message_stride, joined_stride, 1e-6);
    for (0..rows) |r| {
        const rank_row = rank_buffer.slice(r * rank * 4, rank * 4);
        try b.matvec(down_buffer, c.down, joined.slice(r * joined_stride * 4, 2 * embedding * 4), rank_row);
        try b.matvec(gain_buffer, c.gain, rank_row, gains.slice(r * gate_stride * 4, embedding * 4));
        try b.matvec(forget_buffer, c.forget, rank_row, forgets.slice(r * gate_stride * 4, embedding * 4));
    }
    try b.controlBlend(current_buffer, message_buffer, gains, forgets, current_buffer, embedding, rows, current_stride, message_stride, gate_stride, current_stride);
    try b.commit();
    var worst: f64 = 0;
    for (0..rows) |r| {
        worst = @max(worst, try expectNear("loop controller", current_buffer.floats()[r * current_stride ..][0..embedding], want[r * embedding ..][0..embedding], 1e-5));
        // The padding between rows is untouched.
        if (!std.mem.eql(f32, current[r * current_stride + embedding ..][0..32], current_buffer.floats()[r * current_stride + embedding ..][0..32])) return error.MetalMismatch;
    }
    return worst;
}

/// SITU over eight slots of a fused gate-up row (the gate rows, then the up
/// rows), values far enough out to saturate both halves.
fn checkSitu(alloc: std.mem.Allocator, b: *Backend, random: std.Random) !f64 {
    const ff = 1152;
    const slots = 8;
    const gate_up = try alloc.alloc(f32, slots * 2 * ff);
    defer alloc.free(gate_up);
    for (gate_up, 0..) |*x, i| x.* = random.floatNorm(f32) * @as(f32, if (i % 7 == 0) 60 else 4);
    const buffer = try upload(b, gate_up);
    defer b.release(buffer) catch {};
    const out = try b.create(slots * ff * 4);
    defer b.release(out) catch {};
    try b.begin();
    try b.situMulRows(buffer, buffer.slice(ff * 4, buffer.len - ff * 4), out, ff, slots, 2 * ff, 2 * ff, ff, 4, 25, null);
    try b.commit();
    const want = try alloc.alloc(f32, slots * ff);
    defer alloc.free(want);
    for (0..slots) |s| for (0..ff) |i| {
        want[s * ff + i] = cpu.situ(gate_up[s * 2 * ff + i], gate_up[s * 2 * ff + ff + i], 4, 25);
    };
    // Two F32 tanh and an exp against F64: the GELU chain's bound.
    var worst = try expectNear("SITU", out.floats()[0 .. slots * ff], want, 2e-5);

    // Each slot's expert factor, as `cpu.experts.ffn` applies it before
    // the activation; slot 7 names an expert past the table, clamped.
    const experts = 16;
    var factors: [experts]f32 = undefined;
    for (&factors) |*f| f.* = 0.25 + 2 * random.float(f32);
    const ids = [slots]u32{ 3, 0, 15, 3, 9, 1, 12, 40 };
    const factors_buffer = try upload(b, &factors);
    defer b.release(factors_buffer) catch {};
    const ids_buffer = try b.create(slots * 4);
    defer b.release(ids_buffer) catch {};
    @memcpy(@as([*]u32, @ptrCast(@alignCast(ids_buffer.host)))[0..slots], &ids);
    try b.begin();
    try b.situMulRows(buffer, buffer.slice(ff * 4, buffer.len - ff * 4), out, ff, slots, 2 * ff, 2 * ff, ff, 4, 25, .{ .indices = ids_buffer, .scales = factors_buffer, .experts = experts });
    try b.commit();
    for (0..slots) |s| for (0..ff) |i| {
        const f = factors[@min(ids[s], experts - 1)];
        want[s * ff + i] = cpu.situ(gate_up[s * 2 * ff + i] * f, gate_up[s * 2 * ff + ff + i] * f, 4, 25);
    };
    worst = @max(worst, try expectNear("SITU, scaled", out.floats()[0 .. slots * ff], want, 2e-5));
    return worst;
}

/// The biased router over 128 experts, eight chosen, four rows: the
/// indices exactly, the weights within rounding.
fn checkRouteBiased(alloc: std.mem.Allocator, b: *Backend, random: std.Random) !f64 {
    const experts = 128;
    const k = 8;
    const rows = 4;
    const in_stride = experts + 8;
    const logits = try alloc.alloc(f32, rows * in_stride);
    defer alloc.free(logits);
    fill(random, logits, 2);
    const bias = try alloc.alloc(f32, experts);
    defer alloc.free(bias);
    fill(random, bias, 0.05);
    const logits_buffer = try upload(b, logits);
    defer b.release(logits_buffer) catch {};
    const bias_buffer = try upload(b, bias);
    defer b.release(bias_buffer) catch {};
    const indices = try b.create(rows * k * 4);
    defer b.release(indices) catch {};
    const weights = try b.create(rows * k * 4);
    defer b.release(weights) catch {};
    try b.begin();
    try b.routeBiased(logits_buffer, bias_buffer, experts, k, rows, in_stride, 8, indices, weights);
    try b.commit();
    const got_indices = @as([*]const u32, @ptrCast(@alignCast(indices.host)))[0 .. rows * k];
    // A row of NaN logits still writes every slot: the lowest indices, NaN weights.
    @memset(indices.host[0 .. rows * k * 4], 0xff);
    @memset(logits_buffer.floats()[0..experts], std.math.nan(f32));
    try b.begin();
    try b.routeBiased(logits_buffer, bias_buffer, experts, k, 1, in_stride, 8, indices, weights);
    try b.commit();
    for (got_indices[0..k], 0..) |index, slot| if (index != slot) {
        std.debug.print("biased route: a NaN row wrote slot {d} as {d}\n", .{ slot, index });
        return error.MetalMismatch;
    };
    @memcpy(logits_buffer.floats()[0..experts], logits[0..experts]);
    try b.begin();
    try b.routeBiased(logits_buffer, bias_buffer, experts, k, rows, in_stride, 8, indices, weights);
    try b.commit();
    var worst: f64 = 0;
    for (0..rows) |r| {
        var want_indices: [k]u32 = undefined;
        var want_weights: [k]f32 = undefined;
        try cpu.experts.routeBiased(logits[r * in_stride ..][0..experts], bias, 8, &want_indices, &want_weights);
        if (!std.mem.eql(u32, &want_indices, got_indices[r * k ..][0..k])) {
            std.debug.print("biased route: row {d} chose {any}, the CPU {any}\n", .{ r, got_indices[r * k ..][0..k], want_indices });
            return error.MetalMismatch;
        }
        worst = @max(worst, try expectNear("biased route weights", weights.floats()[r * k ..][0..k], &want_weights, 2e-6));
    }
    return worst;
}

/// The bias table of a chunk's `rows` heads (relative rows of 16) against
/// `cpu.matvec` of each row through the F32 `[1024][16]` bias tensor, then
/// the refusals: a non-F32 tensor and an output over the input.
fn checkRelativeBias(alloc: std.mem.Allocator, io: std.Io, b: *Backend, random: std.Random) !f64 {
    const length = 16;
    const extent = 1024;
    const rows = 3 * heads;
    const table = try alloc.alloc(f32, extent * length);
    defer alloc.free(table);
    fill(random, table, 0.5);
    const relative = try alloc.alloc(f32, rows * length);
    defer alloc.free(relative);
    fill(random, relative, 2);
    const matrix: cpu.Matrix = .{ .encoding = 0, .rows = extent, .columns = length, .bytes = std.mem.sliceAsBytes(table) };
    const weights = try upload(b, table);
    defer b.release(weights) catch {};
    const input = try upload(b, relative);
    defer b.release(input) catch {};
    const out = try b.create(rows * extent * 4);
    defer b.release(out) catch {};
    try b.begin();
    try b.relativeBias(input, weights, matrix, out, rows);
    try b.commit();
    const want = try alloc.alloc(f32, rows * extent);
    defer alloc.free(want);
    var scratch: [length]f32 = undefined;
    for (0..rows) |r| try cpu.matvec(io, matrix, relative[r * length ..][0..length], want[r * extent ..][0..extent], &scratch);
    const worst = try expectNear("relative bias", out.floats()[0 .. rows * extent], want, 2e-6);
    var half = matrix;
    half.encoding = 1;
    if (b.relativeBias(input, weights, half, out, rows)) |_| return error.ExpectedInvalidShape else |err| if (err != error.InvalidShape) return err;
    if (b.relativeBias(input, weights, matrix, input, 1)) |_| return error.ExpectedInvalidShape else |err| if (err != error.InvalidShape) return err;
    return worst;
}

/// The refusals: overlapping outputs (with the input or the history),
/// gains overlapping the clipped data, SITU's expert factors short,
/// misaligned, or overlapping its output, too many states, a nonpositive
/// threshold, and dynamic taps that do not divide the channels.
fn checkRefusals(b: *Backend) !void {
    const any = try b.create(1 << 20);
    defer b.release(any) catch {};
    const other = try b.create(1 << 20);
    defer b.release(other) catch {};
    try std.testing.expectError(error.InvalidShape, b.shortConvolution(other, any, other, null, any.slice(64, 4096), 64, taps, 2, 64, 64));
    try std.testing.expectError(error.InvalidShape, b.shortConvolution(other, any, other, .{ .taps = other, .heads = 3, .stride = 12 }, other.slice(8192, 4096), 64, taps, 1, 64, 64));
    try std.testing.expectError(error.InvalidShape, b.depthMix(any, other, other.slice(4096, 4096), 32, Backend.depth_max_states + 1, 1, 1, 33 * 32, 32, 1e-6));
    try std.testing.expectError(error.InvalidShape, b.depthMix(any, other, any, 32, 2, 1, 1, 64, 32, 1e-6));
    try std.testing.expectError(error.InvalidShape, b.l2ClipRows(any, null, 0, null, 2, 64, 1, 128));
    try std.testing.expectError(error.InvalidShape, b.l2ClipRows(any, any.slice(256, 512), 8, null, 2, 64, 1, 128));
    try std.testing.expectError(error.InvalidShape, b.shortConvolution(any.slice(0, 4096), other, other, null, any.slice(256, 4096), 64, taps, 1, 64, 64));
    try std.testing.expectError(error.InvalidShape, b.controlBlend(any, other, other, other, any.slice(4, 1024), 64, 1, 64, 64, 64, 64));
    try std.testing.expectError(error.InvalidShape, b.situMulRows(any, any, any, 64, 1, 64, 64, 64, 4, 25, null));
    const ids = any.slice(1024, 64);
    const sixteen = any.slice(2048, 64);
    try std.testing.expectError(error.InvalidShape, b.situMulRows(other, other, any.slice(0, 256), 64, 1, 64, 64, 64, 4, 25, .{ .indices = ids, .scales = sixteen, .experts = 0 }));
    try std.testing.expectError(error.InvalidShape, b.situMulRows(other, other, any.slice(0, 256), 64, 1, 64, 64, 64, 4, 25, .{ .indices = ids, .scales = sixteen, .experts = 17 }));
    try std.testing.expectError(error.InvalidShape, b.situMulRows(other, other, any.slice(0, 256), 64, 1, 64, 64, 64, 4, 25, .{ .indices = any.slice(1026, 4), .scales = sixteen, .experts = 16 }));
    try std.testing.expectError(error.InvalidShape, b.situMulRows(other, other, any.slice(0, 256), 64, 1, 64, 64, 64, 4, 25, .{ .indices = any.slice(128, 4), .scales = sixteen, .experts = 16 }));
    try std.testing.expectError(error.InvalidShape, b.routeBiased(any, other, 300, 8, 1, 300, 8, other, other));
}

pub fn check(alloc: std.mem.Allocator, io: std.Io, b: *Backend) !void {
    var prng = std.Random.DefaultPrng.init(0x1c2);
    const random = prng.random();
    var conv: f64 = 0;
    for ([_]struct { channels: usize, dynamic: ?usize }{ .{ .channels = kv_width, .dynamic = kv_heads }, .{ .channels = kv_width, .dynamic = null }, .{ .channels = embedding, .dynamic = null } }) |case| {
        for ([_]usize{ 1, 5 }) |rows| conv = @max(conv, try checkShortConvolution(alloc, b, random, case.channels, case.dynamic, rows));
    }
    var clip: f64 = 0;
    clip = @max(clip, try checkClip(alloc, b, random, heads, head_size, 22.627416610717773, true, null));
    clip = @max(clip, try checkClip(alloc, b, random, heads, head_size, 22.627416610717773, true, .{ .position = 20000, .floor = 8192, .alpha = 0.1 }));
    clip = @max(clip, try checkClip(alloc, b, random, heads, 16, 8, true, .{ .position = 8190, .floor = 8192, .alpha = 0.1 }));
    clip = @max(clip, try checkClip(alloc, b, random, kv_heads, head_size, 22.627416610717773, false, null));
    var mix: f64 = 0;
    for ([_][2]usize{ .{ 1, 1 }, .{ 2, 3 }, .{ 9, 2 }, .{ 9, 1 }, .{ Backend.depth_max_states, 2 } }) |shape| mix = @max(mix, try checkMix(alloc, b, random, shape[0], shape[1]));
    const control = try checkControl(alloc, io, b, random);
    const situ = try checkSitu(alloc, b, random);
    const route = try checkRouteBiased(alloc, b, random);
    const bias = try checkRelativeBias(alloc, io, b, random);
    try checkRefusals(b);
    std.debug.print("Inkling 2 row-wise operations vs the CPU, relative past 1: short convolution (static and dynamic taps, 1 and 5 rows) {e:.3}, the history advanced exactly; l2 clip (gains, log scale) {e:.3}; depth mix (1-32 states) {e:.3} (bounds 2e-6); controller {e:.3} (1e-5); SITU (with expert factors) {e:.3} (2e-5); biased route weights {e:.3}, its choices the CPU's, a NaN row's every slot written; the bias table of 72 heads {e:.3}\n", .{ conv, clip, mix, control, situ, route, bias });
}
