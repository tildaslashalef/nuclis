//! Mixture-of-experts references: routing (softmax or a biased sigmoid,
//! top-k, renormalized weights) and the gathered gated expert FFN (GELU or
//! SITU) over 3-D encoded tensors
//! whose experts are contiguous row-major matrices. F64 intermediates, no
//! allocation. The chain a model applies before the router logits (norms,
//! scales) and after the sum (post norms) belongs to its adapter.
const std = @import("std");
const cpu = @import("root.zig");
const vector = @import("vector.zig");

pub const Error = cpu.Error || error{ InvalidExpert, NonFiniteWeight };

/// Smallest F16 normal: the floor the reference clamps the selected weights'
/// sum to before dividing, so a token whose selected probabilities underflow
/// still gets finite weights.
pub const weight_sum_floor: f64 = 6.103515625e-5;

/// Borrowed 3-D encoded tensor `[experts][rows][columns]` (GGUF dimension 2
/// is the expert): `experts` contiguous `rows × columns` matrices of equal
/// byte length. Constructing it validates nothing; `expert` does.
pub const ExpertMatrix = struct {
    encoding: u32,
    experts: usize,
    rows: usize,
    columns: usize,
    bytes: []const u8,

    /// Byte length of one expert's matrix, or an error when the tensor
    /// does not divide into `experts` equal matrices of whole rows.
    pub fn expertBytes(self: ExpertMatrix) Error!usize {
        if (self.experts == 0 or self.rows == 0 or self.columns == 0) return error.InvalidShape;
        if (self.bytes.len % self.experts != 0) return error.InvalidByteLength;
        const per_expert = self.bytes.len / self.experts;
        if (per_expert % self.rows != 0) return error.InvalidByteLength;
        try @import("../../quant/decode.zig").validateRow(self.encoding, per_expert / self.rows, self.columns);
        return per_expert;
    }
    /// The `rows × columns` matrix of expert `index`.
    pub fn expert(self: ExpertMatrix, index: usize) Error!cpu.Matrix {
        const per_expert = try self.expertBytes();
        if (index >= self.experts) return error.InvalidExpert;
        return .{ .encoding = self.encoding, .rows = self.rows, .columns = self.columns, .bytes = self.bytes[index * per_expert ..][0..per_expert] };
    }
};

/// Selects `indices.len` experts for one token: the largest logits by
/// (value desc, index asc), their softmax probabilities renormalized to sum
/// one (the sum clamped below at `weight_sum_floor`). `indices` and
/// `weights` have the same length, at most `logits.len`. Logits must be
/// finite; errors leave the outputs untouched.
pub fn route(logits: []const f32, indices: []u32, weights: []f32) Error!void {
    const k = indices.len;
    if (k == 0 or k > logits.len or weights.len != k) return error.InvalidShape;
    var maximum: f64 = -std.math.inf(f64);
    for (logits) |x| {
        if (!std.math.isFinite(x)) return error.NonFiniteInput;
        maximum = @max(maximum, @as(f64, x));
    }
    var total: f64 = 0;
    for (logits) |x| total += @exp(@as(f64, x) - maximum);
    // Selection compares the logits themselves (softmax is monotone), so a
    // kernel that rounds the probabilities differently selects the same set.
    var taken: [max_experts]bool = @splat(false);
    if (logits.len > max_experts) return error.InvalidShape;
    var selected_sum: f64 = 0;
    for (0..k) |slot| {
        var best: usize = logits.len;
        for (logits, 0..) |x, i| {
            if (taken[i]) continue;
            if (best == logits.len or x > logits[best]) best = i;
        }
        taken[best] = true;
        indices[slot] = @intCast(best);
        const probability = @exp(@as(f64, logits[best]) - maximum) / total;
        weights[slot] = @floatCast(probability);
        selected_sum += probability;
    }
    const denominator = @max(selected_sum, weight_sum_floor);
    for (weights) |*w| w.* = @floatCast(@as(f64, w.*) / denominator);
}

/// The sigmoid router's floor under the chosen gates' sum.
pub const biased_weight_epsilon: f64 = 1e-20;

/// Selects `indices.len` experts for one token by `σ(logit) + bias[e]`
/// (value desc, index asc): the bias steers the choice only. A chosen
/// expert's weight is its gate `σ(logit)` over the chosen gates' sum plus
/// `biased_weight_epsilon`, times `scale`. Gates and selection values are
/// F32, so a tie at F32 precision goes to the lower index. Logits and bias
/// must be finite; errors leave the outputs untouched.
pub fn routeBiased(logits: []const f32, bias: []const f32, scale: f32, indices: []u32, weights: []f32) Error!void {
    const k = indices.len;
    if (k == 0 or k > logits.len or weights.len != k or bias.len != logits.len) return error.InvalidShape;
    if (logits.len > max_experts) return error.InvalidShape;
    if (!std.math.isFinite(scale)) return error.NonFiniteInput;
    for (logits, bias) |x, b| if (!std.math.isFinite(x) or !std.math.isFinite(b)) return error.NonFiniteInput;
    var taken: [max_experts]bool = @splat(false);
    var gates: [max_experts]f32 = undefined;
    var chosen_sum: f64 = 0;
    for (0..k) |slot| {
        var best: usize = logits.len;
        var best_value: f32 = undefined;
        for (logits, bias, 0..) |x, b, i| {
            if (taken[i]) continue;
            const value = vector.sigmoid(x) + b;
            if (best == logits.len or value > best_value) {
                best = i;
                best_value = value;
            }
        }
        taken[best] = true;
        indices[slot] = @intCast(best);
        gates[slot] = vector.sigmoid(logits[best]);
        chosen_sum += gates[slot];
    }
    const denominator = chosen_sum + biased_weight_epsilon;
    for (weights, gates[0..k]) |*w, g| w.* = @floatCast(@as(f64, g) / denominator * scale);
}

/// Experts a router may choose among; bounds the selection scratch.
pub const max_experts = 1024;

/// The gathered expert FFN of one token. `gate_up` holds, per expert, the
/// gate rows followed by the up rows (`2 · ff` rows); `down` maps `ff` back
/// to the model width. `gate_up_scale` and `down_scale`, when present, are
/// one factor per expert: the first multiplies both halves of the gate-up
/// projection before the activation, the second the down projection before
/// the routing weight.
pub const Ffn = struct {
    gate_up: ExpertMatrix,
    down: ExpertMatrix,
    gate_up_scale: ?[]const f32 = null,
    down_scale: ?[]const f32 = null,
    /// The hidden row's activation of the gate and up halves.
    activation: Activation = .gelu,

    /// F32 scratch `ffn` needs: the gate-up row, the hidden row, the down
    /// row, and the widest decode row.
    pub fn scratchLen(self: Ffn) usize {
        return self.gate_up.rows + self.down.columns + self.down.rows + @max(self.gate_up.columns, self.down.columns);
    }
};

/// `gelu(g) · u`, or `vector.situ(g, u, beta, linear_beta)`.
pub const Activation = union(enum) {
    gelu,
    situ: struct { beta: f32, linear_beta: f32 },
};

/// output = Σ_s weights[s] · scale[e_s] · down[e_s] · act(g, u) with
/// (g, u) the halves of gate_up_scale[e_s] · gate_up[e_s] · input, summed in
/// F64 over the slots.
/// `scratch` needs `spec.scratchLen()` floats and `accumulator` one F64
/// per output; all slices are borrowed and must not overlap the outputs.
/// Each expert's matvecs split their rows across `io` tasks as far as the
/// scratch past `scratchLen() - max(columns)` holds decode rows
/// (`cpu.matvecScratch`); the slots stay in order, so the F64 sum does too.
pub fn ffn(io: std.Io, spec: Ffn, input: []const f32, indices: []const u32, weights: []const f32, output: []f32, scratch: []f32, accumulator: []f64) Error!void {
    const gu = spec.gate_up;
    const down = spec.down;
    _ = try gu.expertBytes();
    _ = try down.expertBytes();
    if (gu.experts != down.experts or gu.rows != 2 * down.columns or gu.columns != input.len or down.rows != output.len) return error.InvalidShape;
    if (indices.len == 0 or indices.len != weights.len or accumulator.len < output.len) return error.InvalidShape;
    if (scratch.len < spec.scratchLen()) return error.ScratchTooSmall;
    if (spec.down_scale) |s| if (s.len != down.experts) return error.InvalidShape;
    if (spec.gate_up_scale) |s| if (s.len != gu.experts) return error.InvalidShape;
    for (indices) |e| if (e >= gu.experts) return error.InvalidExpert;
    for (weights) |w| if (!std.math.isFinite(w)) return error.NonFiniteWeight;
    const ff = down.columns;
    const gate_up_row = scratch[0..gu.rows];
    const hidden = scratch[gu.rows..][0..ff];
    const projected = scratch[gu.rows + ff ..][0..down.rows];
    const decode = scratch[gu.rows + ff + down.rows ..];
    const acc = accumulator[0..output.len];
    @memset(acc, 0);
    for (indices, weights) |e, w| {
        try cpu.matvec(io, try gu.expert(e), input, gate_up_row, decode);
        if (spec.gate_up_scale) |s| for (gate_up_row) |*x| {
            x.* *= s[e];
        };
        switch (spec.activation) {
            .gelu => for (hidden, gate_up_row[0..ff], gate_up_row[ff..]) |*h, g, u| {
                h.* = vector.gelu(g) * u;
            },
            .situ => |a| for (hidden, gate_up_row[0..ff], gate_up_row[ff..]) |*h, g, u| {
                h.* = vector.situ(g, u, a.beta, a.linear_beta);
            },
        }
        try cpu.matvec(io, try down.expert(e), hidden, projected, decode);
        const scale: f64 = if (spec.down_scale) |s| s[e] else 1;
        for (acc, projected) |*a, y| a.* += @as(f64, w) * (scale * @as(f64, y));
    }
    for (output, acc) |*o, a| o.* = @floatCast(a);
}

test "routing picks the largest logits, lowest index on ties, and renormalizes" {
    var indices: [3]u32 = undefined;
    var weights: [3]f32 = undefined;
    try route(&.{ 0.5, 2, -1, 2, 1.5, 2 }, &indices, &weights);
    try std.testing.expectEqualSlices(u32, &.{ 1, 3, 5 }, &indices);
    for (weights) |w| try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), w, 1e-7);
    try route(&.{ 0, 1, 2, 3 }, indices[0..2], weights[0..2]);
    try std.testing.expectEqualSlices(u32, &.{ 3, 2 }, indices[0..2]);
    // Renormalized softmax of the top two: e^3 / (e^3 + e^2) and e^2 / (e^3 + e^2).
    try std.testing.expectApproxEqAbs(@as(f32, 0.7310585786300049), weights[0], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.2689414213699951), weights[1], 1e-7);
    // The whole vector: weights are the plain softmax (sum already one).
    try route(&.{ 1, 1, 1 }, &indices, &weights);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2 }, &indices);
    for (weights) |w| try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), w, 1e-7);
}

test "routing clamps an underflowing selected sum and rejects bad shapes" {
    var indices: [1]u32 = undefined;
    var weights: [1]f32 = undefined;
    // The selected probability is exp(-800) ≈ 0 in F64: the floor keeps it finite.
    try route(&.{ 0, -800, 0 }, &indices, &weights);
    try std.testing.expectEqual(@as(u32, 0), indices[0]);
    try std.testing.expect(std.math.isFinite(weights[0]));
    var two: [2]u32 = undefined;
    var two_w: [2]f32 = undefined;
    try std.testing.expectError(error.InvalidShape, route(&.{1}, &two, &two_w));
    try std.testing.expectError(error.InvalidShape, route(&.{ 1, 2 }, &two, two_w[0..1]));
    try std.testing.expectError(error.InvalidShape, route(&.{ 1, 2 }, two[0..0], two_w[0..0]));
    try std.testing.expectError(error.NonFiniteInput, route(&.{ 1, std.math.nan(f32) }, &two, &two_w));
}

test "expert slices index the tensor by expert and reject partial tensors" {
    // Three F32 experts of 2 × 4.
    var bytes: [3 * 2 * 4 * 4]u8 = undefined;
    for (0..24) |i| std.mem.writeInt(u32, bytes[i * 4 ..][0..4], @bitCast(@as(f32, @floatFromInt(i))), .little);
    const tensor: ExpertMatrix = .{ .encoding = 0, .experts = 3, .rows = 2, .columns = 4, .bytes = &bytes };
    const second = try tensor.expert(1);
    try std.testing.expectEqual(@as(usize, 32), second.bytes.len);
    var output: [2]f32 = undefined;
    var scratch: [4]f32 = undefined;
    try cpu.matvec(std.testing.io, second, &.{ 1, 0, 0, 0 }, &output, &scratch);
    try std.testing.expectEqualSlices(f32, &.{ 8, 12 }, &output);
    try std.testing.expectError(error.InvalidExpert, tensor.expert(3));
    var bad = tensor;
    bad.bytes = bytes[0..95];
    try std.testing.expectError(error.InvalidByteLength, bad.expert(0));
    bad = tensor;
    bad.experts = 5;
    try std.testing.expectError(error.InvalidByteLength, bad.expert(0));
}

test "gathered FFN weights, scales, and sums the selected experts" {
    // Two experts, width 2, ff 1. Expert 0: gate [1, 0], up [0, 1], down
    // [1; 2]ᵀ (rows [1], [2]); expert 1: gate [0, 1], up [1, 0], down [3], [4].
    const gate_up_values = [_]f32{ 1, 0, 0, 1, 0, 1, 1, 0 };
    const down_values = [_]f32{ 1, 2, 3, 4 };
    var gate_up_bytes: [32]u8 = undefined;
    var down_bytes: [16]u8 = undefined;
    for (gate_up_values, 0..) |v, i| std.mem.writeInt(u32, gate_up_bytes[i * 4 ..][0..4], @bitCast(v), .little);
    for (down_values, 0..) |v, i| std.mem.writeInt(u32, down_bytes[i * 4 ..][0..4], @bitCast(v), .little);
    const spec: Ffn = .{
        .gate_up = .{ .encoding = 0, .experts = 2, .rows = 2, .columns = 2, .bytes = &gate_up_bytes },
        .down = .{ .encoding = 0, .experts = 2, .rows = 2, .columns = 1, .bytes = &down_bytes },
        .down_scale = &.{ 1, 0.5 },
    };
    var output: [2]f32 = undefined;
    var scratch: [16]f32 = undefined;
    var acc: [2]f64 = undefined;
    // input (2, 3): expert 0 hidden = gelu(2)·3, expert 1 hidden = gelu(3)·2.
    try ffn(std.testing.io, spec, &.{ 2, 3 }, &.{ 0, 1 }, &.{ 0.25, 0.75 }, &output, &scratch, &acc);
    const h0: f64 = @as(f64, vector.gelu(2)) * 3;
    const h1: f64 = @as(f64, vector.gelu(3)) * 2;
    try std.testing.expectApproxEqRel(@as(f32, @floatCast(0.25 * h0 * 1 + 0.75 * 0.5 * h1 * 3)), output[0], 1e-6);
    try std.testing.expectApproxEqRel(@as(f32, @floatCast(0.25 * h0 * 2 + 0.75 * 0.5 * h1 * 4)), output[1], 1e-6);
    try std.testing.expectError(error.InvalidExpert, ffn(std.testing.io, spec, &.{ 2, 3 }, &.{2}, &.{1}, &output, &scratch, &acc));
    try std.testing.expectError(error.ScratchTooSmall, ffn(std.testing.io, spec, &.{ 2, 3 }, &.{0}, &.{1}, &output, scratch[0..4], &acc));
    try std.testing.expectError(error.NonFiniteWeight, ffn(std.testing.io, spec, &.{ 2, 3 }, &.{0}, &.{std.math.inf(f32)}, &output, &scratch, &acc));
    var bad = spec;
    bad.down_scale = &.{1};
    try std.testing.expectError(error.InvalidShape, ffn(std.testing.io, bad, &.{ 2, 3 }, &.{0}, &.{1}, &output, &scratch, &acc));
    bad = spec;
    bad.gate_up_scale = &.{1};
    try std.testing.expectError(error.InvalidShape, ffn(std.testing.io, bad, &.{ 2, 3 }, &.{0}, &.{1}, &output, &scratch, &acc));

    // The gate-up scale enters before the activation: expert 1's hidden is
    // gelu(3 / 4) · (2 / 4), which the factor 1/16 on the weight misses.
    var scaled = spec;
    scaled.gate_up_scale = &.{ 1, 0.25 };
    try ffn(std.testing.io, scaled, &.{ 2, 3 }, &.{1}, &.{1}, &output, &scratch, &acc);
    const h_scaled: f64 = @as(f64, vector.gelu(0.75)) * 0.5;
    try std.testing.expectApproxEqRel(@as(f32, @floatCast(0.5 * h_scaled * 3)), output[0], 1e-6);
    try std.testing.expectApproxEqRel(@as(f32, @floatCast(0.5 * h_scaled * 4)), output[1], 1e-6);
    try std.testing.expect(@abs(output[0] - @as(f32, @floatCast(0.0625 * 0.5 * h1 * 3))) > 0.05);
}

test "the biased sigmoid router selects on gate plus bias and weights by gate" {
    // Gates σ(0, 1, −1, 2) = (0.5, 0.731, 0.269, 0.881); with bias (0.5, 0,
    // 0, −1) the selection values are (1.0, 0.731, 0.269, −0.119).
    var indices: [2]u32 = undefined;
    var weights: [2]f32 = undefined;
    try routeBiased(&.{ 0, 1, -1, 2 }, &.{ 0.5, 0, 0, -1 }, 8, &indices, &weights);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1 }, &indices);
    const g0: f64 = 0.5;
    const g1: f64 = vector.sigmoid(1);
    try std.testing.expectApproxEqRel(@as(f32, @floatCast(8 * g0 / (g0 + g1))), weights[0], 1e-6);
    try std.testing.expectApproxEqRel(@as(f32, @floatCast(8 * g1 / (g0 + g1))), weights[1], 1e-6);
    // Without the bias the gates themselves choose.
    try routeBiased(&.{ 0, 1, -1, 2 }, &.{ 0, 0, 0, 0 }, 1, &indices, &weights);
    try std.testing.expectEqualSlices(u32, &.{ 3, 1 }, &indices);
}

test "a near tie under F32 goes to the lower index; underflowing gates stay finite" {
    // 0.5 + 1e-9 rounds to 0.5 in F32: a tie, though F64 would pick index 1.
    var indices: [1]u32 = undefined;
    var weights: [1]f32 = undefined;
    try routeBiased(&.{ 0, 0 }, &.{ 0, 1e-9 }, 1, &indices, &weights);
    try std.testing.expectEqual(@as(u32, 0), indices[0]);
    // A bias one F32 step above wins.
    try routeBiased(&.{ 0, 0 }, &.{ 0, 6e-8 }, 1, &indices, &weights);
    try std.testing.expectEqual(@as(u32, 1), indices[0]);
    // Gates that underflow F32 sum to 0: the epsilon keeps weights at 0.
    var two: [2]u32 = undefined;
    var two_w: [2]f32 = undefined;
    try routeBiased(&.{ -200, -300, -250 }, &.{ 0, 0, 1 }, 8, &two, &two_w);
    try std.testing.expectEqualSlices(u32, &.{ 2, 0 }, &two);
    try std.testing.expectEqualSlices(f32, &.{ 0, 0 }, &two_w);
}

test "the biased router rejects bad shapes and non-finite input untouched" {
    var indices: [2]u32 = @splat(7);
    var weights: [2]f32 = @splat(9);
    try std.testing.expectError(error.InvalidShape, routeBiased(&.{ 1, 2 }, &.{1}, 1, &indices, &weights));
    try std.testing.expectError(error.InvalidShape, routeBiased(&.{1}, &.{1}, 1, &indices, &weights));
    try std.testing.expectError(error.NonFiniteInput, routeBiased(&.{ 1, std.math.nan(f32) }, &.{ 0, 0 }, 1, &indices, &weights));
    try std.testing.expectError(error.NonFiniteInput, routeBiased(&.{ 1, 2 }, &.{ 0, std.math.inf(f32) }, 1, &indices, &weights));
    try std.testing.expectError(error.NonFiniteInput, routeBiased(&.{ 1, 2 }, &.{ 0, 0 }, std.math.nan(f32), &indices, &weights));
    try std.testing.expectEqualSlices(u32, &.{ 7, 7 }, &indices);
    try std.testing.expectEqualSlices(f32, &.{ 9, 9 }, &weights);
}

test "the SITU expert FFN applies its activation after the gate-up scale" {
    // One expert, width 2, ff 1: gate [1, 0], up [0, 1], down [1], [2].
    const gate_up_values = [_]f32{ 1, 0, 0, 1 };
    const down_values = [_]f32{ 1, 2 };
    var gate_up_bytes: [16]u8 = undefined;
    var down_bytes: [8]u8 = undefined;
    for (gate_up_values, 0..) |v, i| std.mem.writeInt(u32, gate_up_bytes[i * 4 ..][0..4], @bitCast(v), .little);
    for (down_values, 0..) |v, i| std.mem.writeInt(u32, down_bytes[i * 4 ..][0..4], @bitCast(v), .little);
    const spec: Ffn = .{
        .gate_up = .{ .encoding = 0, .experts = 1, .rows = 2, .columns = 2, .bytes = &gate_up_bytes },
        .down = .{ .encoding = 0, .experts = 1, .rows = 2, .columns = 1, .bytes = &down_bytes },
        .gate_up_scale = &.{2},
        .down_scale = &.{0.5},
        .activation = .{ .situ = .{ .beta = 4, .linear_beta = 25 } },
    };
    var output: [2]f32 = undefined;
    var scratch: [16]f32 = undefined;
    var acc: [2]f64 = undefined;
    try ffn(std.testing.io, spec, &.{ 3, 5 }, &.{0}, &.{1.5}, &output, &scratch, &acc);
    // (g, u) = 2·(3, 5); hidden = situ(6, 10); y = 1.5 · 0.5 · (1, 2) · hidden.
    const h: f64 = vector.situ(6, 10, 4, 25);
    try std.testing.expectApproxEqRel(@as(f32, @floatCast(0.75 * h)), output[0], 1e-6);
    try std.testing.expectApproxEqRel(@as(f32, @floatCast(1.5 * h)), output[1], 1e-6);
}
