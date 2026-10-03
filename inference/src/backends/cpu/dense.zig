//! F32 kernels for running an encoder on the CPU: `Y = X·Wᵀ + b` over
//! row-major operands, LayerNorm, and bidirectional multi-head attention over
//! one sequence, full or windowed. Unlike the reference kernels beside them
//! these are for speed: sums accumulate in F32 SIMD lanes (so results match
//! an F32 reference to rounding, not bit for bit) and the matmul and attention
//! split their work across `Io` tasks. No allocation; buffers must not overlap.
const std = @import("std");

pub const Error = error{InvalidShape} || std.Io.Cancelable;

const lanes = 8;
const V = @Vector(lanes, f32);

fn load(values: []const f32, at: usize) V {
    return values[at..][0..lanes].*;
}

/// Tasks per call: enough to fill the cores, never more than there are work units.
fn taskCount(units: usize) usize {
    const cores = std.Thread.getCpuCount() catch 1;
    return @max(1, @min(cores, units));
}

pub const Matmul = struct {
    rows: usize,
    inner: usize,
    outputs: usize,
    /// [rows][inner]
    x: []const f32,
    /// [outputs][inner], a PyTorch `Linear` weight.
    w: []const f32,
    /// [outputs] or null.
    bias: ?[]const f32 = null,
};

/// y[i][j] = Σ_c x[i][c]·w[j][c] + bias[j], y being [rows][outputs]. Output
/// columns split across tasks, each a whole number of 4-wide tiles.
pub fn matmul(io: std.Io, m: Matmul, y: []f32) Error!void {
    if (m.rows == 0 or m.inner == 0 or m.outputs == 0 or m.x.len != m.rows * m.inner or
        m.w.len != m.outputs * m.inner or y.len != m.rows * m.outputs) return error.InvalidShape;
    if (m.bias) |b| if (b.len != m.outputs) return error.InvalidShape;
    const tiles = @divCeil(m.outputs, 4);
    const tasks = taskCount(tiles);
    var group: std.Io.Group = .init;
    for (0..tasks) |t| {
        const first = @min(m.outputs, tiles * t / tasks * 4);
        const last = @min(m.outputs, tiles * (t + 1) / tasks * 4);
        if (first < last) group.async(io, matmulColumns, .{ m, y, first, last });
    }
    try group.await(io);
}

fn matmulColumns(m: Matmul, y: []f32, first: usize, last: usize) void {
    var i: usize = 0;
    while (i < m.rows) : (i += 4) {
        var j = first;
        while (j < last) : (j += 4) {
            if (i + 4 <= m.rows and j + 4 <= last) {
                tile(4, 4, m, y, i, j);
            } else for (i..@min(i + 4, m.rows)) |r| for (j..@min(j + 4, last)) |q| tile(1, 1, m, y, r, q);
        }
    }
}

/// An R×Q block of outputs at (i, j): R rows of x against Q rows of w.
inline fn tile(comptime R: usize, comptime Q: usize, m: Matmul, y: []f32, i: usize, j: usize) void {
    var acc: [R][Q]V = @splat(@splat(@splat(0)));
    var c: usize = 0;
    while (c + lanes <= m.inner) : (c += lanes) {
        var xs: [R]V = undefined;
        inline for (0..R) |r| xs[r] = load(m.x, (i + r) * m.inner + c);
        inline for (0..Q) |q| {
            const wv = load(m.w, (j + q) * m.inner + c);
            inline for (0..R) |r| acc[r][q] += xs[r] * wv;
        }
    }
    inline for (0..R) |r| inline for (0..Q) |q| {
        var sum = @reduce(.Add, acc[r][q]);
        for (c..m.inner) |k| sum += m.x[(i + r) * m.inner + k] * m.w[(j + q) * m.inner + k];
        if (m.bias) |b| sum += b[j + q];
        y[(i + r) * m.outputs + j + q] = sum;
    };
}

/// LayerNorm of one row: (x − mean) / sqrt(var + eps) · weight (+ bias), the
/// biased variance, statistics in F64. `out` may alias `x`.
pub fn layerNorm(x: []const f32, weight: []const f32, bias: ?[]const f32, eps: f32, out: []f32) Error!void {
    if (x.len == 0 or weight.len != x.len or out.len != x.len) return error.InvalidShape;
    if (bias) |b| if (b.len != x.len) return error.InvalidShape;
    var sum: f64 = 0;
    for (x) |v| sum += v;
    const mean = sum / @as(f64, @floatFromInt(x.len));
    var squares: f64 = 0;
    for (x) |v| squares += (v - mean) * (v - mean);
    const inv = 1.0 / @sqrt(squares / @as(f64, @floatFromInt(x.len)) + eps);
    for (x, weight, out, 0..) |v, w, *o, k| {
        const normalized: f32 = @floatCast((v - mean) * inv);
        o.* = normalized * w + if (bias) |b| b[k] else 0;
    }
}

pub const Attention = struct {
    tokens: usize,
    heads: usize,
    head_dim: usize,
    /// Query i sees key j when |i − j| ≤ window; null sees every key.
    window: ?usize = null,
    scale: f32,
    /// Row `t` of q, k, and v starts at `t · stride`: [heads][head_dim] each,
    /// so they may be three column ranges of one fused projection.
    stride: usize,
    q: []const f32,
    k: []const f32,
    v: []const f32,
};

/// out[t] = [heads][head_dim] of softmax(scale · q·k) v over the visible keys;
/// out rows are `heads · head_dim` apart. Heads split across tasks; `scratch`
/// needs `heads · tokens` values.
pub fn attention(io: std.Io, a: Attention, out: []f32, scratch: []f32) Error!void {
    const width = a.heads * a.head_dim;
    if (a.tokens == 0 or a.heads == 0 or a.head_dim == 0 or a.head_dim % lanes != 0 or a.stride < width or
        out.len != a.tokens * width or scratch.len < a.heads * a.tokens) return error.InvalidShape;
    const span = (a.tokens - 1) * a.stride + width;
    if (a.q.len < span or a.k.len < span or a.v.len < span) return error.InvalidShape;
    const tasks = taskCount(a.heads);
    var group: std.Io.Group = .init;
    for (0..tasks) |t| {
        const first = a.heads * t / tasks;
        const last = a.heads * (t + 1) / tasks;
        if (first < last) group.async(io, attentionHeads, .{ a, out, scratch, first, last });
    }
    try group.await(io);
}

fn attentionHeads(a: Attention, out: []f32, scratch: []f32, first: usize, last: usize) void {
    const width = a.heads * a.head_dim;
    for (first..last) |h| {
        const scores = scratch[h * a.tokens ..][0..a.tokens];
        for (0..a.tokens) |i| {
            const lo = if (a.window) |w| i -| w else 0;
            const hi = if (a.window) |w| @min(a.tokens, i + w + 1) else a.tokens;
            const q = a.q[i * a.stride + h * a.head_dim ..][0..a.head_dim];
            var max: f32 = -std.math.inf(f32);
            for (lo..hi) |j| {
                const k = a.k[j * a.stride + h * a.head_dim ..][0..a.head_dim];
                var acc: V = @splat(0);
                var c: usize = 0;
                while (c < a.head_dim) : (c += lanes) acc += load(q, c) * load(k, c);
                scores[j] = @reduce(.Add, acc) * a.scale;
                max = @max(max, scores[j]);
            }
            var total: f32 = 0;
            for (scores[lo..hi]) |*s| {
                s.* = @exp(s.* - max);
                total += s.*;
            }
            const o = out[i * width + h * a.head_dim ..][0..a.head_dim];
            @memset(o, 0);
            for (lo..hi) |j| {
                const p = scores[j] / total;
                const v = a.v[j * a.stride + h * a.head_dim ..][0..a.head_dim];
                var c: usize = 0;
                while (c < a.head_dim) : (c += lanes) o[c..][0..lanes].* = load(o, c) + @as(V, @splat(p)) * load(v, c);
            }
        }
    }
}

fn naiveMatmul(m: Matmul, y: []f32) void {
    for (0..m.rows) |i| for (0..m.outputs) |j| {
        var sum: f64 = if (m.bias) |b| b[j] else 0;
        for (0..m.inner) |c| sum += @as(f64, m.x[i * m.inner + c]) * m.w[j * m.inner + c];
        y[i * m.outputs + j] = @floatCast(sum);
    };
}

fn fill(values: []f32, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    for (values) |*v| v.* = prng.random().float(f32) * 2 - 1;
}

test "matmul matches an F64 reference on ragged shapes, with and without bias" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    for ([_][3]usize{ .{ 1, 1, 1 }, .{ 5, 19, 7 }, .{ 8, 64, 12 }, .{ 13, 70, 33 } }) |shape| {
        const rows, const inner, const outputs = shape;
        const x = try gpa.alloc(f32, rows * inner);
        defer gpa.free(x);
        const w = try gpa.alloc(f32, outputs * inner);
        defer gpa.free(w);
        const b = try gpa.alloc(f32, outputs);
        defer gpa.free(b);
        const got = try gpa.alloc(f32, rows * outputs);
        defer gpa.free(got);
        const want = try gpa.alloc(f32, rows * outputs);
        defer gpa.free(want);
        fill(x, 1);
        fill(w, 2);
        fill(b, 3);
        for ([_]?[]const f32{ null, b }) |bias| {
            const m: Matmul = .{ .rows = rows, .inner = inner, .outputs = outputs, .x = x, .w = w, .bias = bias };
            try matmul(io, m, got);
            naiveMatmul(m, want);
            for (got, want) |g, e| try std.testing.expectApproxEqAbs(e, g, 1e-5);
        }
    }
    var y: [2]f32 = undefined;
    try std.testing.expectError(error.InvalidShape, matmul(io, .{ .rows = 1, .inner = 2, .outputs = 2, .x = &.{ 1, 2 }, .w = &.{ 1, 2, 3 } }, &y));
}

test "layer norm with and without bias, in place" {
    var x = [_]f32{ 1, 2, 3, 6 };
    try layerNorm(&x, &.{ 1, 1, 2, 1 }, &.{ 0, 0, 0, 1 }, 0, &x);
    // mean 3, variance 3.5
    const s: f32 = @sqrt(3.5);
    const want = [_]f32{ -2 / s, -1 / s, 0, 3 / s + 1 };
    for (x, want) |g, e| try std.testing.expectApproxEqAbs(e, g, 1e-6);
    var y: [4]f32 = undefined;
    try layerNorm(&.{ 2, 2, 2, 2 }, &.{ 1, 1, 1, 1 }, null, 1e-5, &y);
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 0, 0 }, &y);
}

test "windowed attention equals the single-query reference over each window" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const reference = @import("attention.zig");
    const tokens = 11;
    const heads = 2;
    const dim = 16;
    const width = heads * dim;
    const stride = 3 * width;
    const qkv = try gpa.alloc(f32, tokens * stride);
    defer gpa.free(qkv);
    fill(qkv, 4);
    var keys: [tokens * width]f32 = undefined;
    var values: [tokens * width]f32 = undefined;
    for (0..tokens) |t| {
        @memcpy(keys[t * width ..][0..width], qkv[t * stride + width ..][0..width]);
        @memcpy(values[t * width ..][0..width], qkv[t * stride + 2 * width ..][0..width]);
    }
    var out: [tokens * width]f32 = undefined;
    var scratch: [heads * tokens]f32 = undefined;
    var f64_scratch: [tokens]f64 = undefined;
    for ([_]?usize{ null, 0, 3 }) |window| {
        try attention(io, .{ .tokens = tokens, .heads = heads, .head_dim = dim, .window = window, .scale = 0.25, .stride = stride, .q = qkv, .k = qkv[width..], .v = qkv[2 * width ..] }, &out, &scratch);
        for (0..tokens) |i| {
            const lo: usize = if (window) |w| i -| w else 0;
            const hi: usize = if (window) |w| @min(tokens, i + w + 1) else tokens;
            var want: [width]f32 = undefined;
            try reference.apply(.{
                .query_heads = heads,
                .kv_heads = heads,
                .key_width = dim,
                .value_width = dim,
                .tokens = hi - lo,
                .visible_tokens = hi - lo,
                .scale = 0.25,
                .queries = qkv[i * stride ..][0..width],
                .keys = keys[lo * width .. hi * width],
                .values = values[lo * width .. hi * width],
            }, &want, &f64_scratch);
            for (out[i * width ..][0..width], want) |g, e| try std.testing.expectApproxEqAbs(e, g, 1e-5);
        }
    }
}
