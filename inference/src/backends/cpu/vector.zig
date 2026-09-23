//! Architecture-neutral vector references with F64 intermediate arithmetic.
//! No allocation or I/O. Borrowed output may exactly alias input; partial
//! overlap is forbidden. Expected errors leave output untouched. Test constants
//! for transcendental formulas were checked with Python decimal at 80 digits.
const std = @import("std");

pub const Error = error{ InvalidShape, InvalidEpsilon, NonFiniteInput, EmptySupport };

/// x / max(sqrt(sum(x*x)), epsilon). Unlike RMSNorm, this uses a sum,
/// and epsilon clamps the norm after the square root. Zero vectors stay zero.
/// Borrowed slices may exactly alias; partial overlap is forbidden.
pub fn l2Norm(input: []const f32, output: []f32, epsilon: f32) Error!void {
    try shape(input, output);
    if (!std.math.isFinite(epsilon) or epsilon <= 0) return error.InvalidEpsilon;
    var squares: f64 = 0;
    for (input) |x| {
        if (!std.math.isFinite(x)) return error.NonFiniteInput;
        squares += @as(f64, x) * @as(f64, x);
    }
    const denominator = @max(@sqrt(squares), @as(f64, epsilon));
    for (input, output) |x, *y| y.* = @floatCast(@as(f64, x) / denominator);
}

test "L2 normalization clamps the norm and preserves finite extremes" {
    var values = [_]f32{ 3, 4 };
    try l2Norm(&values, &values, 1e-6);
    try std.testing.expectEqualSlices(f32, &.{ 0.6, 0.8 }, &values);
    try l2Norm(&.{ 3, 4 }, &values, 10);
    try std.testing.expectEqualSlices(f32, &.{ 0.3, 0.4 }, &values);
    try l2Norm(&.{ 0, 0 }, &values, 1e-6);
    try std.testing.expectEqualSlices(f32, &.{ 0, 0 }, &values);
    try l2Norm(&.{ std.math.floatMax(f32), 0 }, &values, 1e-6);
    try std.testing.expectEqualSlices(f32, &.{ 1, 0 }, &values);
    const tiny: f32 = @bitCast(@as(u32, 1));
    try l2Norm(&.{ tiny, 0 }, &values, tiny);
    try std.testing.expectEqualSlices(f32, &.{ 1, 0 }, &values);
}

test "L2 errors leave output unchanged" {
    var output = [_]f32{123} ** 2;
    try std.testing.expectError(error.InvalidShape, l2Norm(&.{}, output[0..0], 1));
    try std.testing.expectError(error.InvalidShape, l2Norm(&.{1}, &output, 1));
    for ([_]f32{ 0, -1, std.math.nan(f32), std.math.inf(f32) }) |epsilon|
        try std.testing.expectError(error.InvalidEpsilon, l2Norm(&.{ 1, 2 }, &output, epsilon));
    for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }) |bad|
        try std.testing.expectError(error.NonFiniteInput, l2Norm(&.{ 1, bad }, &output, 1));
    try std.testing.expectEqualSlices(f32, &.{ 123, 123 }, &output);
}

/// x / sqrt(mean(x*x) + epsilon), without learned weights or a bias.
/// Epsilon must be finite and positive; input must be finite and nonempty.
/// Model adapters own any subsequent learned scaling or gating convention.
pub fn rmsNorm(input: []const f32, output: []f32, epsilon: f32) Error!void {
    try shape(input, output);
    if (!std.math.isFinite(epsilon) or epsilon <= 0) return error.InvalidEpsilon;
    var squares: f64 = 0;
    for (input) |x| {
        if (!std.math.isFinite(x)) return error.NonFiniteInput;
        squares += @as(f64, x) * @as(f64, x);
    }
    // F64 can square even the largest finite F32 without overflowing.
    const denominator = @sqrt(squares / @as(f64, @floatFromInt(input.len)) + epsilon);
    for (input, output) |x, *y| y.* = @floatCast(@as(f64, x) / denominator);
}

/// exp(x-max(x)) / sum(exp(x-max(x))). Negative infinity denotes a masked
/// entry and yields zero. NaN/+infinity are rejected; all-masked is EmptySupport.
/// Reductions finish before writes, permitting exact in-place operation.
pub fn softmax(input: []const f32, output: []f32) Error!void {
    try shape(input, output);
    var maximum: f64 = -std.math.inf(f64);
    for (input) |x| {
        if (std.math.isNan(x) or std.math.isPositiveInf(x)) return error.NonFiniteInput;
        maximum = @max(maximum, @as(f64, x));
    }
    if (std.math.isNegativeInf(maximum)) return error.EmptySupport;
    var total: f64 = 0;
    for (input) |x| total += @exp(@as(f64, x) - maximum);
    // Recompute exponentials rather than allocating a full F64 temporary row.
    for (input, output) |x, *y| y.* = @floatCast(@exp(@as(f64, x) - maximum) / total);
}

/// Logistic gate, computed without exp(-x) overflow for large negative x.
/// +/-infinity map to 1/0; NaN propagates.
pub fn sigmoid(x: f32) f32 {
    return @floatCast(logistic(x));
}

/// SiLU(x) = x * sigmoid(x). Its limit at -infinity is negative zero;
/// +infinity remains +infinity and NaN propagates.
/// GELU in the tanh approximation, the form GGUF checkpoints declare as
/// `gelu_pytorch_tanh` (Gemma 4's `hidden_activation`):
/// 0.5·x·(1 + tanh(√(2/π)·x·(1 + 0.044715·x²))). Evaluated in F64 so the
/// F32 result is the correctly rounded value of the formula, which is what
/// a reference computing it in F32 agrees with to one ulp.
pub fn gelu(x: f32) f32 {
    const v: f64 = x;
    const inner = 0.7978845608028654 * v * (1.0 + 0.044715 * v * v);
    return @floatCast(0.5 * v * (1.0 + std.math.tanh(inner)));
}

test "tanh GELU matches the formula at pinned points and is odd-symmetric around its asymptotes" {
    // Values from the formula in double precision (independently evaluated).
    const cases = [_][2]f32{
        .{ 0, 0 },
        .{ 1, 0.8411919906082768 },
        .{ -1, -0.15880800939172324 },
        .{ 2, 1.954597694087775 },
        .{ -2, -0.04540230590592927 },
        .{ 0.5, 0.34571400982514394 },
        .{ 6, 6 },
        .{ -6, -0.0 },
    };
    for (cases) |c| try std.testing.expectApproxEqAbs(c[1], gelu(c[0]), 1e-6);
    try std.testing.expect(std.math.isNan(gelu(std.math.nan(f32))));
}

/// The quick GELU, x·σ(1.702x) (ggml's `gelu_quick`), in F64.
pub fn geluQuick(x: f32) f32 {
    const v: f64 = x;
    const z = 1.702 * v;
    const gate = if (z >= 0) 1 / (1 + @exp(-z)) else @exp(z) / (1 + @exp(z));
    return @floatCast(v * gate);
}

test "quick GELU is x times the logistic of 1.702x" {
    try std.testing.expectEqual(@as(f32, 0), geluQuick(0));
    try std.testing.expectApproxEqAbs(@as(f32, @floatCast(1.0 / (1.0 + @exp(-1.702)))), geluQuick(1), 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, -2.0 / (1.0 + @exp(3.404))), geluQuick(-2), 1e-7);
}

pub fn silu(x: f32) f32 {
    if (std.math.isNegativeInf(x)) return -0.0;
    return @floatCast(@as(f64, x) * logistic(x));
}

/// log(1+exp(x)), with log1p preserving the small negative-input tail.
/// -infinity maps to zero; +infinity remains +infinity; NaN propagates.
pub fn softplus(x: f32) f32 {
    if (std.math.isNan(x)) return x;
    const wide: f64 = x;
    return @floatCast(@max(wide, 0) + std.math.log1p(@exp(-@abs(wide))));
}

fn logistic(x: f32) f64 {
    const wide: f64 = x;
    if (wide >= 0) return 1 / (1 + @exp(-wide));
    const e = @exp(wide);
    return e / (1 + e);
}

fn shape(input: []const f32, output: []f32) Error!void {
    if (input.len == 0 or input.len != output.len) return error.InvalidShape;
}

test "RMS normalization uses mean and epsilon inside square root" {
    var output: [2]f32 = undefined;
    try rmsNorm(&.{ 3, 4 }, &output, 3.5);
    try std.testing.expectEqualSlices(f32, &.{ 0.75, 1 }, &output);
    var in_place = [_]f32{ -3, 4 };
    try rmsNorm(&in_place, &in_place, 3.5);
    try std.testing.expectEqualSlices(f32, &.{ -0.75, 1 }, &in_place);
    try rmsNorm(&.{ 0, 0 }, &output, 1e-6);
    try std.testing.expectEqualSlices(f32, &.{ 0, 0 }, &output);
    const big = std.math.floatMax(f32);
    try rmsNorm(&.{ big, -big }, &output, 1e-6);
    try std.testing.expectEqualSlices(f32, &.{ 1, -1 }, &output);
}

test "softmax known probabilities, translation invariance, and exact alias" {
    var output: [3]f32 = undefined;
    // Independently evaluated exp(0), exp(1), exp(2), normalized by their sum.
    const expected = [_]f32{ 0.09003057317038046, 0.24472847105479764, 0.6652409557748219 };
    try softmax(&.{ 0, 1, 2 }, &output);
    for (output, expected) |got, want| try std.testing.expectApproxEqAbs(want, got, 1e-7);
    var shifted = [_]f32{ 1000, 1001, 1002 };
    try softmax(&shifted, &shifted);
    try std.testing.expectEqualSlices(f32, &output, &shifted);
    try softmax(&.{ -1000, -1000, -1000 }, &output);
    for (output) |x| try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), x, 1e-7);
}

test "softmax masks and full F32 dynamic range" {
    var output: [3]f32 = undefined;
    const big = std.math.floatMax(f32);
    try softmax(&.{ -big, -std.math.inf(f32), big }, &output);
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 1 }, &output);
    try softmax(&.{ -std.math.inf(f32), 0, 0 }, &output);
    try std.testing.expectEqualSlices(f32, &.{ 0, 0.5, 0.5 }, &output);
}

test "vector errors do not partially write results" {
    var output = [_]f32{123} ** 2;
    try std.testing.expectError(error.InvalidShape, rmsNorm(&.{}, output[0..0], 1));
    try std.testing.expectError(error.InvalidShape, rmsNorm(&.{1}, &output, 1));
    try std.testing.expectError(error.InvalidShape, softmax(&.{}, output[0..0]));
    try std.testing.expectError(error.InvalidShape, softmax(&.{1}, &output));
    for ([_]f32{ 0, -1, std.math.nan(f32), std.math.inf(f32) }) |epsilon|
        try std.testing.expectError(error.InvalidEpsilon, rmsNorm(&.{ 1, 2 }, &output, epsilon));
    for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }) |bad|
        try std.testing.expectError(error.NonFiniteInput, rmsNorm(&.{ 1, bad }, &output, 1));
    for ([_]f32{ std.math.nan(f32), std.math.inf(f32) }) |bad|
        try std.testing.expectError(error.NonFiniteInput, softmax(&.{ 1, bad }, &output));
    try std.testing.expectError(error.EmptySupport, softmax(&.{ -std.math.inf(f32), -std.math.inf(f32) }, &output));
    try std.testing.expectEqualSlices(f32, &.{ 123, 123 }, &output);
}

test "activation values and stable extreme tails" {
    try std.testing.expectEqual(@as(f32, 0.5), sigmoid(0));
    try std.testing.expectApproxEqAbs(@as(f32, 0.7310585786300049), sigmoid(1), 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, -0.2689414213699951), silu(-1), 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.6931471805599453), softplus(0), 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 1.3132616875182228), softplus(1), 1e-7);
    try std.testing.expectApproxEqRel(@as(f32, 2.061153620314381e-9), softplus(-20), 1e-6);
    try std.testing.expectEqual(@as(f32, 0), sigmoid(-1000));
    try std.testing.expectEqual(@as(f32, 1), sigmoid(1000));
    try std.testing.expectEqual(@as(f32, 1000), silu(1000));
    try std.testing.expectEqual(@as(f32, 1000), softplus(1000));
    try std.testing.expectEqual(@as(f32, 0), softplus(-1000));
}

test "scalar activation nonfinite inputs follow explicit limits" {
    const inf = std.math.inf(f32);
    try std.testing.expectEqual(@as(f32, 0), sigmoid(-inf));
    try std.testing.expectEqual(@as(f32, 1), sigmoid(inf));
    try std.testing.expect(std.math.isPositiveInf(silu(inf)));
    try std.testing.expectEqual(@as(u32, 0x80000000), @as(u32, @bitCast(silu(-inf))));
    try std.testing.expectEqual(@as(f32, 0), softplus(-inf));
    try std.testing.expect(std.math.isPositiveInf(softplus(inf)));
    try std.testing.expect(std.math.isNan(sigmoid(std.math.nan(f32))));
    try std.testing.expect(std.math.isNan(silu(std.math.nan(f32))));
    try std.testing.expect(std.math.isNan(softplus(std.math.nan(f32))));
}
