//! Unscaled, split-half rotary position embedding for one attention head.
//! Architecture-neutral: the caller supplies the rotary width, frequency base,
//! and position. Model adapters own position streams and Q/K head traversal.
//! No allocation or I/O. Exact alias is supported; partial overlap is forbidden.
const std = @import("std");

pub const Options = struct {
    dimensions: usize,
    base: f32,
    position: i32,
    /// Per-pair frequency factors (`dimensions/2` of them): theta for pair i
    /// is divided by `factors[i]`. Null means all ones. Checkpoints use huge
    /// factors (1e30) to leave a pair unrotated; the angle then underflows
    /// toward zero and the pair copies through within one ulp (Gemma 4).
    factors: ?[]const f32 = null,
};
pub const Error = error{ InvalidShape, InvalidFrequencyBase, NonFiniteInput };

/// For pair i in the rotary prefix, theta = position * base^(-2*i/dimensions).
/// Rotate (input[i], input[i+dimensions/2]) by theta; copy the tail unchanged.
/// All arithmetic is F64 before rounding outputs to F32. Finite input can still
/// overflow F32 after rotation, producing infinity as in the matvec reference.
/// Expected errors are validated before any output writes. Base must be >= 1.
pub fn apply(input: []const f32, output: []f32, options: Options) Error!void {
    if (input.len == 0 or input.len != output.len or options.dimensions == 0 or
        options.dimensions % 2 != 0 or options.dimensions > input.len) return error.InvalidShape;
    if (!std.math.isFinite(options.base) or options.base < 1) return error.InvalidFrequencyBase;
    if (options.factors) |f| {
        if (f.len != options.dimensions / 2) return error.InvalidShape;
        for (f) |x| if (!std.math.isFinite(x) or x <= 0) return error.InvalidFrequencyBase;
    }
    for (input) |x| if (!std.math.isFinite(x)) return error.NonFiniteInput;
    if (options.position == 0) {
        // Preserve signed zero as well as values at the identity position.
        for (input, output) |x, *y| y.* = x;
        return;
    }
    const half = options.dimensions / 2;
    for (0..half) |i| {
        const exponent = -@as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(half));
        const factor: f64 = if (options.factors) |f| f[i] else 1;
        const theta = @as(f64, @floatFromInt(options.position)) * std.math.pow(f64, options.base, exponent) / factor;
        const cosine = @cos(theta);
        const sine = @sin(theta);
        // Read both values before writing either, so exact alias is safe.
        const a: f64 = input[i];
        const b: f64 = input[i + half];
        output[i] = @floatCast(a * cosine - b * sine);
        output[i + half] = @floatCast(a * sine + b * cosine);
    }
    for (input[options.dimensions..], output[options.dimensions..]) |x, *y| y.* = x;
}

test "RoPE split-half pairs, independent frequencies, and untouched tail" {
    var values = [_]f32{ 1, 2, 3, 4, 9, -0.0 };
    try apply(&values, &values, .{ .dimensions = 4, .base = 100, .position = 2 });
    const expected = [_]f32{ -3.144039116, 1.165455832, -0.339143083, 4.317604973 };
    for (values[0..4], expected) |got, want| try std.testing.expectApproxEqAbs(want, got, 5e-7);
    try std.testing.expectEqual(@as(f32, 9), values[4]);
    try std.testing.expectEqual(@as(u32, 0x80000000), @as(u32, @bitCast(values[5])));
    try apply(&values, &values, .{ .dimensions = 4, .base = 100, .position = -2 });
    for (values[0..4], [_]f32{ 1, 2, 3, 4 }) |got, want| try std.testing.expectApproxEqAbs(want, got, 5e-7);
}

test "RoPE frequency factors divide the angle; unit factors equal no factors; huge factors leave the pair in place" {
    const input = [_]f32{ 1, 2, 3, 4 };
    var plain: [4]f32 = undefined;
    var unit: [4]f32 = undefined;
    try apply(&input, &plain, .{ .dimensions = 4, .base = 100, .position = 3 });
    try apply(&input, &unit, .{ .dimensions = 4, .base = 100, .position = 3, .factors = &.{ 1, 1 } });
    try std.testing.expectEqualSlices(f32, &plain, &unit);
    // Factor 2 on pair 0 halves its angle: equal to position 1.5 for that pair.
    var halved: [4]f32 = undefined;
    try apply(&input, &halved, .{ .dimensions = 4, .base = 100, .position = 3, .factors = &.{ 2, 1 } });
    const theta = 1.5 * std.math.pow(f64, 100, 0);
    try std.testing.expectApproxEqAbs(@as(f32, @floatCast(1 * @cos(theta) - 3 * @sin(theta))), halved[0], 1e-6);
    try std.testing.expectEqual(plain[1], halved[1]);
    // A 1e30 factor: the pair is unchanged to F32 precision, the other rotates.
    var frozen: [4]f32 = undefined;
    try apply(&input, &frozen, .{ .dimensions = 4, .base = 100, .position = 32767, .factors = &.{ 1e30, 1 } });
    try std.testing.expectApproxEqAbs(@as(f32, 1), frozen[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3), frozen[2], 1e-6);
    try std.testing.expect(frozen[1] != input[1]);
    try std.testing.expectError(error.InvalidShape, apply(&input, &frozen, .{ .dimensions = 4, .base = 100, .position = 1, .factors = &.{1} }));
    try std.testing.expectError(error.InvalidFrequencyBase, apply(&input, &frozen, .{ .dimensions = 4, .base = 100, .position = 1, .factors = &.{ 0, 1 } }));
}

test "RoPE identity preserves bits and rotation preserves pair norm" {
    const input = [_]f32{ -0.0, 2, -3, 4 };
    var output: [4]f32 = undefined;
    try apply(&input, &output, .{ .dimensions = 4, .base = 1, .position = 0 });
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&input), std.mem.sliceAsBytes(&output));
    try apply(&input, &output, .{ .dimensions = 4, .base = 1e7, .position = 32767 });
    for (0..2) |i| {
        const before = input[i] * input[i] + input[i + 2] * input[i + 2];
        const after = output[i] * output[i] + output[i + 2] * output[i + 2];
        try std.testing.expectApproxEqRel(before, after, 2e-7);
    }
    const big = std.math.floatMax(f32);
    try apply(&.{ big, big }, output[0..2], .{ .dimensions = 2, .base = 1, .position = 1 });
    try std.testing.expect(std.math.isPositiveInf(output[1]));
}

test "RoPE rejects invalid shape, base, and nonfinite tail before writes" {
    var output = [_]f32{123} ** 4;
    try std.testing.expectError(error.InvalidShape, apply(&.{}, output[0..0], .{ .dimensions = 2, .base = 100, .position = 1 }));
    for ([_]usize{ 0, 1, 3, 6 }) |width|
        try std.testing.expectError(error.InvalidShape, apply(&.{ 1, 2, 3, 4 }, &output, .{ .dimensions = width, .base = 100, .position = 1 }));
    try std.testing.expectError(error.InvalidShape, apply(&.{1}, &output, .{ .dimensions = 2, .base = 100, .position = 1 }));
    for ([_]f32{ 0, -1, 0.5, std.math.inf(f32), std.math.nan(f32) }) |base|
        try std.testing.expectError(error.InvalidFrequencyBase, apply(&.{ 1, 2, 3, 4 }, &output, .{ .dimensions = 2, .base = base, .position = 1 }));
    for ([_]f32{ std.math.inf(f32), -std.math.inf(f32), std.math.nan(f32) }) |bad|
        try std.testing.expectError(error.NonFiniteInput, apply(&.{ 1, 2, 3, bad }, &output, .{ .dimensions = 2, .base = 100, .position = 1 }));
    try std.testing.expectEqualSlices(f32, &.{ 123, 123, 123, 123 }, &output);
}
