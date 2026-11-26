//! Single-position grouped-query attention reference. No allocation or I/O.
//! The caller owns projection, RoPE, KV storage, causal position, and output
//! gating. This function consumes borrowed contiguous F32 arrays only.
const std = @import("std");

pub const Input = struct {
    query_heads: usize,
    kv_heads: usize,
    key_width: usize,
    value_width: usize,
    tokens: usize,
    /// Only [0..visible_tokens] attends; the suffix is causally masked.
    visible_tokens: usize,
    scale: f32,
    /// [query_head][key_channel]
    queries: []const f32,
    /// [token][kv_head][key_channel]
    keys: []const f32,
    /// [token][kv_head][value_channel]
    values: []const f32,
    /// An additive score bias by distance, or none.
    bias: ?Bias = null,
    /// [query_head]: a logit that joins each head's softmax denominator as
    /// `exp(sink)` and carries no value, or none.
    sinks: ?[]const f32 = null,
};

/// `table[query_head][d]` is added to the scaled score of the key `d` rows
/// before the last visible one, which is the query's own row (`d = 0`);
/// keys `extent` or more rows back take no bias.
pub const Bias = struct { table: []const f32, extent: usize };
pub const Error = error{ InvalidShape, Overflow, EmptySupport, InvalidScale, ScratchTooSmall, NonFiniteInput };

/// Computes softmax(scale * Q.K + bias) V, with a sink's mass in each
/// softmax's denominator when given. Consecutive groups of query heads share a
/// KV head: kv_head = query_head / (query_heads/kv_heads). Output is flattened
/// [query_head][value_channel]. Scratch needs visible_tokens F64 elements; its
/// suffix is untouched. Neither writable buffer may overlap any other buffer.
///
/// Validate every input, including masked storage, before writes. Scores,
/// exponentials, and weighted sums stay F64 until final F32 output conversion.
/// No implicit scale, arbitrary mask, or sliding-window policy; a position
/// bias only as the explicit distance table.
pub fn apply(input: Input, output: []f32, scratch: []f64) Error!void {
    const x = input;
    if (x.query_heads == 0 or x.kv_heads == 0 or x.query_heads % x.kv_heads != 0 or
        x.key_width == 0 or x.value_width == 0 or x.tokens == 0 or x.visible_tokens > x.tokens)
        return error.InvalidShape;
    if (x.visible_tokens == 0) return error.EmptySupport;
    const queries = try std.math.mul(usize, x.query_heads, x.key_width);
    const key_row = try std.math.mul(usize, x.kv_heads, x.key_width);
    const value_row = try std.math.mul(usize, x.kv_heads, x.value_width);
    const keys = try std.math.mul(usize, x.tokens, key_row);
    const values = try std.math.mul(usize, x.tokens, value_row);
    const outputs = try std.math.mul(usize, x.query_heads, x.value_width);
    if (x.queries.len != queries or x.keys.len != keys or x.values.len != values or output.len != outputs)
        return error.InvalidShape;
    if (scratch.len < x.visible_tokens) return error.ScratchTooSmall;
    if (!std.math.isFinite(x.scale) or x.scale <= 0) return error.InvalidScale;
    for (x.queries) |v| if (!std.math.isFinite(v)) return error.NonFiniteInput;
    for (x.keys) |v| if (!std.math.isFinite(v)) return error.NonFiniteInput;
    for (x.values) |v| if (!std.math.isFinite(v)) return error.NonFiniteInput;
    if (x.bias) |bias| {
        if (bias.extent == 0 or bias.table.len != try std.math.mul(usize, x.query_heads, bias.extent)) return error.InvalidShape;
        for (bias.table) |v| if (!std.math.isFinite(v)) return error.NonFiniteInput;
    }
    if (x.sinks) |sinks| {
        if (sinks.len != x.query_heads) return error.InvalidShape;
        for (sinks) |v| if (!std.math.isFinite(v)) return error.NonFiniteInput;
    }

    const scores = scratch[0..x.visible_tokens];
    for (0..x.query_heads) |head| {
        const kv = head / (x.query_heads / x.kv_heads);
        const q = x.queries[head * x.key_width ..][0..x.key_width];
        var maximum: f64 = -std.math.inf(f64);
        for (scores, 0..) |*score, token| {
            const k = x.keys[token * key_row + kv * x.key_width ..][0..x.key_width];
            var dot: f64 = 0;
            for (q, k) |a, b| dot += @as(f64, a) * @as(f64, b);
            score.* = dot * @as(f64, x.scale);
            if (x.bias) |bias| {
                const distance = x.visible_tokens - 1 - token;
                if (distance < bias.extent) score.* += bias.table[head * bias.extent + distance];
            }
            maximum = @max(maximum, score.*);
        }
        const sink: ?f64 = if (x.sinks) |sinks| sinks[head] else null;
        if (sink) |t| maximum = @max(maximum, t);
        // Finite F32 inputs, F32 scale, and usize-bounded widths cannot overflow
        // F64 here. Subtracting the maximum leaves at least one exp(0) = 1.
        var total: f64 = if (sink) |t| @exp(t - maximum) else 0;
        for (scores) |*score| {
            score.* = @exp(score.* - maximum);
            total += score.*;
        }
        for (0..x.value_width) |channel| {
            var weighted: f64 = 0;
            for (scores, 0..) |weight, token|
                weighted += weight * @as(f64, x.values[token * value_row + kv * x.value_width + channel]);
            output[head * x.value_width + channel] = @floatCast(weighted / total);
        }
    }
}

fn simple() Input {
    return .{ .query_heads = 1, .kv_heads = 1, .key_width = 2, .value_width = 1, .tokens = 2, .visible_tokens = 2, .scale = 1, .queries = &.{ 1, 0 }, .keys = &.{ 1, 0, 0, 1 }, .values = &.{ 2, 4 } };
}

test "attention scaled probabilities and causal prefix" {
    var x = simple();
    var output: [1]f32 = undefined;
    var scratch: [3]f64 = @splat(123);
    try apply(x, &output, &scratch);
    try std.testing.expectApproxEqAbs(@as(f32, 2.5378828427399902), output[0], 3e-7);
    x.scale = 2;
    try apply(x, &output, &scratch);
    try std.testing.expectApproxEqAbs(@as(f32, 2.238405844044235), output[0], 3e-7);
    x.visible_tokens = 1;
    try apply(x, &output, &scratch);
    try std.testing.expectEqual(@as(f32, 2), output[0]);
    try std.testing.expectEqual(@as(f64, 123), scratch[2]);
}

test "a distance bias adds to the scaled score by rows back from the query" {
    var x = simple();
    var output: [1]f32 = undefined;
    var scratch: [2]f64 = undefined;
    // Scores (1, 0) at distances (1, 0); +5 at distance 0 only.
    x.bias = .{ .table = &.{5}, .extent = 1 };
    try apply(x, &output, &scratch);
    const e = std.math.e;
    try std.testing.expectApproxEqRel(@as(f32, @floatCast((2 * e + 4 * @exp(5.0)) / (e + @exp(5.0)))), output[0], 1e-6);
    // −1 at distance 1 levels the scores: the mean of the values.
    x.bias = .{ .table = &.{ 0, -1 }, .extent = 2 };
    try apply(x, &output, &scratch);
    try std.testing.expectEqual(@as(f32, 3), output[0]);
    // The bias adds after the scale: scale 2 gives (2, 0), then (1, 0).
    x.scale = 2;
    try apply(x, &output, &scratch);
    try std.testing.expectApproxEqRel(@as(f32, @floatCast((2 * e + 4) / (e + 1))), output[0], 1e-6);
}

test "a sink takes softmax mass and contributes no value" {
    var x = simple();
    var output: [1]f32 = undefined;
    var scratch: [2]f64 = undefined;
    // exp(ln 2) = 2 joins exp(1) + exp(0) in the denominator only.
    x.sinks = &.{@log(2.0)};
    try apply(x, &output, &scratch);
    const e = std.math.e;
    try std.testing.expectApproxEqRel(@as(f32, @floatCast((2 * e + 4) / (2 + e + 1))), output[0], 1e-6);
    // A sink far above every score leaves almost nothing: no overflow.
    x.sinks = &.{3e38};
    try apply(x, &output, &scratch);
    try std.testing.expectEqual(@as(f32, 0), output[0]);
    // Per head: head 1's sink does not touch head 0.
    const two: Input = .{ .query_heads = 2, .kv_heads = 1, .key_width = 1, .value_width = 1, .tokens = 1, .visible_tokens = 1, .scale = 1, .queries = &.{ 0, 0 }, .keys = &.{0}, .values = &.{6}, .sinks = &.{ -1e30, 0 } };
    var pair: [2]f32 = undefined;
    try apply(two, &pair, &scratch);
    try std.testing.expectEqualSlices(f32, &.{ 6, 3 }, &pair);
}

test "a zero bias and no sink reproduce the plain arithmetic bit for bit" {
    var prng: std.Random.DefaultPrng = .init(0x5eed);
    const random = prng.random();
    var queries: [8]f32 = undefined;
    var keys: [5 * 2 * 4]f32 = undefined;
    var values: [5 * 2 * 3]f32 = undefined;
    for (&queries) |*v| v.* = random.floatNorm(f32);
    for (&keys) |*v| v.* = random.floatNorm(f32);
    for (&values) |*v| v.* = random.floatNorm(f32);
    var x: Input = .{ .query_heads = 2, .kv_heads = 2, .key_width = 4, .value_width = 3, .tokens = 5, .visible_tokens = 4, .scale = 0.5, .queries = &queries, .keys = &keys, .values = &values };
    var plain: [6]f32 = undefined;
    var biased: [6]f32 = undefined;
    var scratch: [5]f64 = undefined;
    try apply(x, &plain, &scratch);
    x.bias = .{ .table = &@as([2 * 3]f32, @splat(0)), .extent = 3 };
    try apply(x, &biased, &scratch);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&plain), std.mem.sliceAsBytes(&biased));
}

test "attention maps consecutive query groups and token-major KV rows" {
    const x: Input = .{ .query_heads = 4, .kv_heads = 2, .key_width = 1, .value_width = 2, .tokens = 3, .visible_tokens = 2, .scale = 1, .queries = &.{ 0, 0, 0, 0 }, .keys = &.{ 1, 2, 3, 4, 5, 6 }, .values = &.{ 10, 20, 100, 200, 30, 40, 300, 400, -1000, -1000, -1000, -1000 } };
    var output: [8]f32 = undefined;
    var scratch: [2]f64 = undefined;
    try apply(x, &output, &scratch);
    try std.testing.expectEqualSlices(f32, &.{ 20, 30, 20, 30, 200, 300, 200, 300 }, &output);
}

test "attention handles scores beyond F32 without overflow" {
    const big = std.math.floatMax(f32);
    var x: Input = .{ .query_heads = 1, .kv_heads = 1, .key_width = 1, .value_width = 1, .tokens = 2, .visible_tokens = 2, .scale = big, .queries = &.{big}, .keys = &.{ big, -big }, .values = &.{ 7, -3 } };
    var output: [1]f32 = undefined;
    var scratch: [2]f64 = undefined;
    try apply(x, &output, &scratch);
    try std.testing.expectEqual(@as(f32, 7), output[0]);
    x.queries = &.{-big};
    try apply(x, &output, &scratch);
    try std.testing.expectEqual(@as(f32, -3), output[0]);
}

test "attention matches pinned CPU matrix-softmax-matrix graphs" {
    const Case = struct {
        query_heads: usize,
        kv_heads: usize,
        key_width: usize,
        value_width: usize,
        tokens: usize,
        visible_tokens: usize,
        scale: f32,
        queries: []const f32,
        keys: []const f32,
        values: []const f32,
        output: []const f32,
    };
    const Fixture = struct { revision: []const u8, cases: []const Case };
    const alloc = std.testing.allocator;
    const fixture_data = try std.json.parseFromSlice(Fixture, alloc, @embedFile("fixtures/attention.json"), .{});
    defer fixture_data.deinit();
    try std.testing.expectEqualStrings("7620399f58aebfd2196b74021f9581bcf7218cb9", fixture_data.value.revision);
    for (fixture_data.value.cases) |case| {
        const out = try alloc.alloc(f32, case.output.len);
        defer alloc.free(out);
        const scratch = try alloc.alloc(f64, case.visible_tokens);
        defer alloc.free(scratch);
        try apply(.{ .query_heads = case.query_heads, .kv_heads = case.kv_heads, .key_width = case.key_width, .value_width = case.value_width, .tokens = case.tokens, .visible_tokens = case.visible_tokens, .scale = case.scale, .queries = case.queries, .keys = case.keys, .values = case.values }, out, scratch);
        for (out, case.output) |got, want| try std.testing.expectApproxEqAbs(want, got, 2e-6);
    }
}

test "attention rejects malformed inputs before touching output or scratch" {
    var output = [_]f32{123};
    var scratch: [2]f64 = @splat(456);
    const Case = struct { input: Input, err: Error };
    var cases: std.ArrayList(Case) = .empty;
    defer cases.deinit(std.testing.allocator);
    var x = simple();
    x.visible_tokens = 0;
    try cases.append(std.testing.allocator, .{ .input = x, .err = error.EmptySupport });
    x = simple();
    x.visible_tokens = 3;
    try cases.append(std.testing.allocator, .{ .input = x, .err = error.InvalidShape });
    x = simple();
    x.query_heads = 3;
    x.kv_heads = 2;
    try cases.append(std.testing.allocator, .{ .input = x, .err = error.InvalidShape });
    x = simple();
    x.key_width = std.math.maxInt(usize);
    try cases.append(std.testing.allocator, .{ .input = x, .err = error.Overflow });
    x = simple();
    x.keys = &.{1};
    try cases.append(std.testing.allocator, .{ .input = x, .err = error.InvalidShape });
    x = simple();
    x.kv_heads = 0;
    try cases.append(std.testing.allocator, .{ .input = x, .err = error.InvalidShape });
    for ([_]f32{ 0, -1, std.math.inf(f32), std.math.nan(f32) }) |scale| {
        x = simple();
        x.scale = scale;
        try cases.append(std.testing.allocator, .{ .input = x, .err = error.InvalidScale });
    }
    x = simple();
    x.values = &.{ 2, std.math.nan(f32) };
    x.visible_tokens = 1;
    try cases.append(std.testing.allocator, .{ .input = x, .err = error.NonFiniteInput });
    x = simple();
    x.queries = &.{ std.math.inf(f32), 0 };
    try cases.append(std.testing.allocator, .{ .input = x, .err = error.NonFiniteInput });
    x = simple();
    x.keys = &.{ 1, 0, -std.math.inf(f32), 1 };
    x.visible_tokens = 1;
    try cases.append(std.testing.allocator, .{ .input = x, .err = error.NonFiniteInput });
    x = simple();
    x.bias = .{ .table = &.{ 1, 2 }, .extent = 1 };
    try cases.append(std.testing.allocator, .{ .input = x, .err = error.InvalidShape });
    x = simple();
    x.bias = .{ .table = &.{}, .extent = 0 };
    try cases.append(std.testing.allocator, .{ .input = x, .err = error.InvalidShape });
    x = simple();
    x.bias = .{ .table = &.{std.math.nan(f32)}, .extent = 1 };
    try cases.append(std.testing.allocator, .{ .input = x, .err = error.NonFiniteInput });
    x = simple();
    x.sinks = &.{ 0, 0 };
    try cases.append(std.testing.allocator, .{ .input = x, .err = error.InvalidShape });
    x = simple();
    x.sinks = &.{std.math.inf(f32)};
    try cases.append(std.testing.allocator, .{ .input = x, .err = error.NonFiniteInput });
    for (cases.items) |case| try std.testing.expectError(case.err, apply(case.input, &output, &scratch));
    try std.testing.expectError(error.ScratchTooSmall, apply(simple(), &output, scratch[0..1]));
    try std.testing.expectEqualSlices(f32, &.{123}, &output);
    try std.testing.expectEqualSlices(f64, &.{ 456, 456 }, &scratch);
}
