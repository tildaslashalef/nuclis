//! Attention over the depth: a token's earlier block states mixed by learned
//! queries (`mix`), and the gated message controller that blends a state
//! with one from another pass (`control`). Architecture-neutral references
//! with F64 intermediates and no allocation; which states and queries a
//! model passes, and when, belongs to its adapter.
const std = @import("std");
const cpu = @import("root.zig");

pub const Error = cpu.Error || error{NonFiniteInput};

/// `output[j] = Σ_s softmax_s(queries[j] · K_s / √width) · K_s` with
/// `K_s = rms(states[s])` (no gain, `epsilon` inside the root): each of the
/// `queries.len / width` query rows reads the state list as both keys and
/// values. `states` holds whole rows of `width`; `scratch` needs
/// `2 · states` F64 elements. `output` must not overlap the inputs. Errors
/// leave output and scratch untouched.
pub fn mix(states: []const f32, queries: []const f32, width: usize, epsilon: f32, output: []f32, scratch: []f64) Error!void {
    if (width == 0 or states.len == 0 or queries.len == 0 or states.len % width != 0 or queries.len % width != 0 or output.len != queries.len)
        return error.InvalidShape;
    if (!std.math.isFinite(epsilon) or epsilon <= 0) return error.InvalidEpsilon;
    const count = states.len / width;
    if (scratch.len < 2 * count) return error.ScratchTooSmall;
    for (states) |x| if (!std.math.isFinite(x)) return error.NonFiniteInput;
    for (queries) |x| if (!std.math.isFinite(x)) return error.NonFiniteInput;
    const norms = scratch[0..count];
    const scores = scratch[count..][0..count];
    const n: f64 = @floatFromInt(width);
    for (norms, 0..) |*norm, s| {
        var squares: f64 = 0;
        for (states[s * width ..][0..width]) |x| squares += @as(f64, x) * x;
        norm.* = @sqrt(squares / n + epsilon);
    }
    const scale = 1 / @sqrt(n);
    for (0..queries.len / width) |j| {
        const q = queries[j * width ..][0..width];
        var maximum: f64 = -std.math.inf(f64);
        for (scores, norms, 0..) |*score, norm, s| {
            var dot: f64 = 0;
            for (q, states[s * width ..][0..width]) |a, b| dot += @as(f64, a) * b;
            score.* = dot / norm * scale;
            maximum = @max(maximum, score.*);
        }
        var total: f64 = 0;
        for (scores) |*score| {
            score.* = @exp(score.* - maximum);
            total += score.*;
        }
        const out = output[j * width ..][0..width];
        for (out, 0..) |*o, c| {
            var sum: f64 = 0;
            for (scores, norms, 0..) |p, norm, s| sum += p * (@as(f64, states[s * width + c]) / norm);
            o.* = @floatCast(sum / total);
        }
    }
}

/// The three matrices of one controller: `down` maps `[rms(x); rms(m)]`
/// (`2 · width`) to a rank, `gain` and `forget` map the rank back to `width`.
pub const Controller = struct {
    down: cpu.Matrix,
    gain: cpu.Matrix,
    forget: cpu.Matrix,

    /// F32 scratch `control` needs: the concatenated input, the rank row,
    /// the two gate rows, and the widest decode row.
    pub fn scratchLen(self: Controller) usize {
        return self.down.columns + self.down.rows + self.gain.rows + self.forget.rows + @max(self.down.columns, self.gain.columns, self.forget.columns);
    }
};

/// `output = (1 − tanh(F h)) ⊙ current + tanh(G h) ⊙ message` with
/// `h = D [rms(current); rms(message)]`, each half normed on its own
/// (`epsilon` inside the root, no gain). `output` may exactly alias
/// `current`; it must not overlap `message` or `scratch`.
pub fn control(io: std.Io, c: Controller, current: []const f32, message: []const f32, epsilon: f32, output: []f32, scratch: []f32) Error!void {
    const width = current.len;
    if (width == 0 or message.len != width or output.len != width) return error.InvalidShape;
    if (c.down.columns != 2 * width or c.gain.columns != c.down.rows or c.forget.columns != c.down.rows or
        c.gain.rows != width or c.forget.rows != width) return error.InvalidShape;
    if (scratch.len < c.scratchLen()) return error.ScratchTooSmall;
    for (message) |x| if (!std.math.isFinite(x)) return error.NonFiniteInput;
    const joined = scratch[0 .. 2 * width];
    const rank = scratch[2 * width ..][0..c.down.rows];
    const gain = scratch[2 * width + c.down.rows ..][0..width];
    const forget = scratch[3 * width + c.down.rows ..][0..width];
    const decode = scratch[4 * width + c.down.rows ..];
    try cpu.rmsNorm(current, joined[0..width], epsilon);
    try cpu.rmsNorm(message, joined[width..], epsilon);
    try cpu.matvec(io, c.down, joined, rank, decode);
    try cpu.matvec(io, c.gain, rank, gain, decode);
    try cpu.matvec(io, c.forget, rank, forget, decode);
    for (output, current, message, gain, forget) |*o, x, m, g, f| {
        const kept = (1 - std.math.tanh(@as(f64, f))) * x;
        o.* = @floatCast(kept + std.math.tanh(@as(f64, g)) * m);
    }
}

fn matrix(rows: usize, columns: usize, values: []const f32) cpu.Matrix {
    return .{ .encoding = 0, .rows = rows, .columns = columns, .bytes = std.mem.sliceAsBytes(values) };
}

test "one state mixes to itself, normed, whatever the query" {
    var output: [2]f32 = undefined;
    var scratch: [2]f64 = undefined;
    try mix(&.{ 3, 4 }, &.{ -7, 100 }, 2, 1e-6, &output, &scratch);
    // rms(3, 4) = √12.5.
    const r: f64 = @sqrt(12.5 + 1e-6);
    try std.testing.expectApproxEqRel(@as(f32, @floatCast(3 / r)), output[0], 1e-7);
    try std.testing.expectApproxEqRel(@as(f32, @floatCast(4 / r)), output[1], 1e-7);
}

test "the mix softmaxes normed keys at 1/√width, several queries at once" {
    // States (2, 0) and (0, 3) norm to (√2, 0) and (0, √2); query (1, 0)
    // scores (√2·1/√2, 0) = (1, 0), query (0, 1) the reverse.
    var output: [4]f32 = undefined;
    var scratch: [4]f64 = undefined;
    try mix(&.{ 2, 0, 0, 3 }, &.{ 1, 0, 0, 1 }, 2, 1e-6, &output, &scratch);
    const e = std.math.e;
    const high: f32 = @floatCast(@sqrt(2.0) * e / (e + 1));
    const low: f32 = @floatCast(@sqrt(2.0) / (e + 1));
    try std.testing.expectApproxEqRel(high, output[0], 1e-6);
    try std.testing.expectApproxEqRel(low, output[1], 1e-6);
    try std.testing.expectApproxEqRel(low, output[2], 1e-6);
    try std.testing.expectApproxEqRel(high, output[3], 1e-6);
    // A state's scale does not change its key: (200, 0) mixes as (2, 0).
    var scaled: [4]f32 = undefined;
    try mix(&.{ 200, 0, 0, 3 }, &.{ 1, 0, 0, 1 }, 2, 1e-6, &scaled, &scratch);
    for (output, scaled) |a, b| try std.testing.expectApproxEqRel(a, b, 1e-6);
}

test "mix errors leave output and scratch untouched" {
    var output: [2]f32 = @splat(123);
    var scratch: [4]f64 = @splat(456);
    try std.testing.expectError(error.InvalidShape, mix(&.{ 1, 2, 3 }, &.{ 1, 0 }, 2, 1e-6, &output, &scratch));
    try std.testing.expectError(error.InvalidShape, mix(&.{ 1, 2 }, &.{ 1, 0, 0, 1 }, 2, 1e-6, &output, &scratch));
    try std.testing.expectError(error.ScratchTooSmall, mix(&.{ 1, 2, 3, 4 }, &.{ 1, 0 }, 2, 1e-6, &output, scratch[0..3]));
    try std.testing.expectError(error.InvalidEpsilon, mix(&.{ 1, 2 }, &.{ 1, 0 }, 2, 0, &output, &scratch));
    try std.testing.expectError(error.NonFiniteInput, mix(&.{ 1, std.math.nan(f32) }, &.{ 1, 0 }, 2, 1e-6, &output, &scratch));
    try std.testing.expectError(error.NonFiniteInput, mix(&.{ 1, 2 }, &.{ std.math.inf(f32), 0 }, 2, 1e-6, &output, &scratch));
    try std.testing.expectEqualSlices(f32, &.{ 123, 123 }, &output);
    try std.testing.expectEqualSlices(f64, &.{ 456, 456, 456, 456 }, &scratch);
}

test "the controller forgets and gains through tanh of a shared rank" {
    // Width 1, rank 1: rms(2) ≈ 1, rms(−3) ≈ −1, so h = 0.5·1 + 0.25·(−1) =
    // 0.25; forget tanh(2h), gain tanh(4h).
    const down = [_]f32{ 0.5, 0.25 };
    const gain = [_]f32{4};
    const forget = [_]f32{2};
    const c: Controller = .{ .down = matrix(1, 2, &down), .gain = matrix(1, 1, &gain), .forget = matrix(1, 1, &forget) };
    var scratch: [16]f32 = undefined;
    var output: [1]f32 = undefined;
    try control(std.testing.io, c, &.{2}, &.{-3}, 1e-6, &output, &scratch);
    const r2: f64 = 2.0 / @sqrt(4.0 + 1e-6);
    const r3: f64 = 3.0 / @sqrt(9.0 + 1e-6);
    const h = 0.5 * r2 - 0.25 * r3;
    const expected = (1 - std.math.tanh(2 * h)) * 2 + std.math.tanh(4 * h) * -3;
    try std.testing.expectApproxEqRel(@as(f32, @floatCast(expected)), output[0], 1e-6);
    // In place over the current state.
    var current = [_]f32{2};
    try control(std.testing.io, c, &current, &.{-3}, 1e-6, &current, &scratch);
    try std.testing.expectEqual(output[0], current[0]);
}

test "a zero controller keeps the state; errors are typed" {
    const zeros: [8]f32 = @splat(0);
    const c: Controller = .{ .down = matrix(2, 4, &zeros), .gain = matrix(2, 2, zeros[0..4]), .forget = matrix(2, 2, zeros[0..4]) };
    var scratch: [16]f32 = undefined;
    var output: [2]f32 = undefined;
    try control(std.testing.io, c, &.{ 1.5, -2 }, &.{ 9, 9 }, 1e-6, &output, &scratch);
    try std.testing.expectEqualSlices(f32, &.{ 1.5, -2 }, &output);
    try std.testing.expectError(error.InvalidShape, control(std.testing.io, c, &.{1}, &.{1}, 1e-6, output[0..1], &scratch));
    try std.testing.expectError(error.ScratchTooSmall, control(std.testing.io, c, &.{ 1, 2 }, &.{ 1, 2 }, 1e-6, &output, scratch[0..8]));
    try std.testing.expectError(error.NonFiniteInput, control(std.testing.io, c, &.{ 1, 2 }, &.{ 1, std.math.nan(f32) }, 1e-6, &output, &scratch));
}
