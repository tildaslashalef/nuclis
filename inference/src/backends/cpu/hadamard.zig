//! The activation-side half of a folded Hadamard rotation: a ±1 sign flip
//! and the normalized Sylvester Walsh-Hadamard transform applied in place
//! to each block of `block` consecutive elements (docs/reference/bonsai.md).
//! `H[r][c] = (-1)^popcount(r & c) / sqrt(block)` is symmetric and
//! orthonormal, so the same butterflies serve the inverse; only the order
//! of the sign flip differs. Block work happens in F64 scratch, so this is a
//! reference, not a kernel.
const std = @import("std");

pub const Error = error{ InvalidShape, InvalidBlock };

/// The largest block this reference transforms (the pinned contract's).
pub const max_block = 1024;

/// `x = H (signs ⊙ x)` per block: what a rotated projection reads.
pub fn forward(x: []f32, signs: []const f32, block: usize) Error!void {
    try validate(x, signs, block);
    for (x, signs) |*value, sign| value.* *= sign;
    transform(x, block);
}

/// `x = signs ⊙ (H x)` per block: what an embedding lookup stores rotated.
pub fn inverse(x: []f32, signs: []const f32, block: usize) Error!void {
    try validate(x, signs, block);
    transform(x, block);
    for (x, signs) |*value, sign| value.* *= sign;
}

fn validate(x: []const f32, signs: []const f32, block: usize) Error!void {
    if (block == 0 or block > max_block or !std.math.isPowerOfTwo(block)) return error.InvalidBlock;
    if (x.len == 0 or x.len % block != 0 or signs.len != x.len) return error.InvalidShape;
}

fn transform(x: []f32, block: usize) void {
    var scratch: [max_block]f64 = undefined;
    const scale = 1.0 / @sqrt(@as(f64, @floatFromInt(block)));
    var start: usize = 0;
    while (start < x.len) : (start += block) {
        const values = scratch[0..block];
        for (values, x[start..][0..block]) |*v, source| v.* = source;
        var half: usize = 1;
        while (half < block) : (half *= 2) {
            var i: usize = 0;
            while (i < block) : (i += 2 * half) {
                for (values[i..][0..half], values[i + half ..][0..half]) |*a, *b| {
                    const sum = a.* + b.*;
                    b.* = a.* - b.*;
                    a.* = sum;
                }
            }
        }
        for (x[start..][0..block], values) |*out, v| out.* = @floatCast(v * scale);
    }
}

fn direct(block: usize, x: []const f64, y: []f64) void {
    const scale = 1.0 / @sqrt(@as(f64, @floatFromInt(block)));
    for (y, 0..) |*out, r| {
        var sum: f64 = 0;
        for (x, 0..) |v, c| sum += if (@popCount(r & c) % 2 == 1) -v else v;
        out.* = sum * scale;
    }
}

test "the butterflies equal the parity-defined matrix and undo themselves" {
    var prng = std.Random.DefaultPrng.init(7);
    const random = prng.random();
    const alloc = std.testing.allocator;
    for ([_]usize{ 1, 2, 8, 1024 }) |block| {
        const width = block * 3;
        const x = try alloc.alloc(f32, width);
        defer alloc.free(x);
        const signs = try alloc.alloc(f32, width);
        defer alloc.free(signs);
        const expected = try alloc.alloc(f64, block);
        defer alloc.free(expected);
        const input = try alloc.alloc(f64, block);
        defer alloc.free(input);
        for (x) |*v| v.* = random.float(f32) * 4 - 2;
        for (signs) |*s| s.* = if (random.boolean()) 1 else -1;
        const original = try alloc.dupe(f32, x);
        defer alloc.free(original);
        try forward(x, signs, block);
        var start: usize = 0;
        while (start < width) : (start += block) {
            for (input, original[start..][0..block], signs[start..][0..block]) |*v, o, s| v.* = @as(f64, o) * s;
            direct(block, input, expected);
            for (x[start..][0..block], expected) |got, want| try std.testing.expectApproxEqAbs(want, @as(f64, got), 1e-5);
        }
        // `inverse` is (H S)^-1 = S H: it restores x from forward's output.
        try inverse(x, signs, block);
        for (x, original) |got, want| try std.testing.expectApproxEqAbs(want, got, 1e-5);
    }
}

test "a constant block and a delta block have known transforms" {
    var x = [_]f32{1} ** 8;
    const ones = [_]f32{1} ** 8;
    try forward(&x, &ones, 8);
    // The constant vector is H's first column: all energy in element 0.
    try std.testing.expectApproxEqAbs(@as(f32, @sqrt(8.0)), x[0], 1e-6);
    for (x[1..]) |v| try std.testing.expectApproxEqAbs(@as(f32, 0), v, 1e-6);
    var delta = [_]f32{ 0, 0, 0, 1, 0, 0, 0, 0 };
    try inverse(&delta, &ones, 8);
    // Row 3 of H: (-1)^popcount(3 & c) / sqrt(8).
    for (delta, 0..) |v, c| try std.testing.expectApproxEqAbs(@as(f32, if (@popCount(3 & c) % 2 == 1) -1 else 1) / @sqrt(8.0), v, 1e-6);
}

test "shape and block validation leaves the input untouched" {
    var x = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const signs = [_]f32{1} ** 6;
    try std.testing.expectError(error.InvalidShape, forward(&x, &signs, 4));
    try std.testing.expectError(error.InvalidShape, forward(x[0..4], &signs, 4));
    try std.testing.expectError(error.InvalidBlock, forward(&x, &signs, 3));
    try std.testing.expectError(error.InvalidBlock, forward(&x, &signs, 0));
    try std.testing.expectError(error.InvalidBlock, inverse(&x, &signs, 2048));
    try std.testing.expectError(error.InvalidShape, inverse(x[0..0], signs[0..0], 2));
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5, 6 }, &x);
}
