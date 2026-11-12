//! The codecs' causal sliding-window attention on the CPU (qwen3_codec.zig,
//! fish_codec.zig, qwen3_codec_encoder.zig): scores and sums in F64, and
//! the per-layer cache a stream carries between pushes, the last
//! `window − 1` keys and values, all a later query can see. Rows are
//! `[rows][heads · head_dim]`; RoPE is applied by the caller.
const std = @import("std");

const Allocator = std.mem.Allocator;

pub const Shape = struct {
    heads: usize,
    head_dim: usize,
    /// Rows a query sees, its own included.
    window: usize,

    pub fn width(self: Shape) usize {
        return self.heads * self.head_dim;
    }
};

/// Sliding-window attention of `n` new queries: query t (row `past + t` of
/// `keys`) sees rows `past + t − window + 1 … past + t`. `scores` holds
/// `window` entries at least.
pub fn attention(shape: Shape, q: []const f32, keys: []const f32, values: []const f32, n: usize, past: usize, scores: []f64, out: []f32) void {
    const width = shape.width();
    const dim = shape.head_dim;
    const scale = 1 / @sqrt(@as(f64, @floatFromInt(dim)));
    for (0..n) |t| for (0..shape.heads) |head| {
        const i = past + t;
        const lowest = (i + 1) -| shape.window;
        const qi = q[t * width + head * dim ..][0..dim];
        var max: f64 = -std.math.inf(f64);
        for (lowest..i + 1, 0..) |j, s| {
            var dot: f64 = 0;
            for (qi, keys[j * width + head * dim ..][0..dim]) |a, b| dot += @as(f64, a) * b;
            scores[s] = dot * scale;
            max = @max(max, scores[s]);
        }
        const count = i + 1 - lowest;
        var total: f64 = 0;
        for (scores[0..count]) |*s| {
            s.* = @exp(s.* - max);
            total += s.*;
        }
        const o = out[t * width + head * dim ..][0..dim];
        for (o, 0..) |*value, d| {
            var sum: f64 = 0;
            for (lowest..i + 1, 0..) |j, s| sum += scores[s] * values[j * width + head * dim + d];
            value.* = @floatCast(sum / total);
        }
    };
}

/// One layer's keys (after RoPE) and values of the rows before the next
/// push: the last `window − 1` at most.
pub const Cache = struct {
    keys: []f32,
    values: []f32,
    count: usize,

    /// One cache per layer, empty, allocated from `a` (an arena: nothing
    /// is freed one by one).
    pub fn many(a: Allocator, shape: Shape, layers: usize) ![]Cache {
        const list = try a.alloc(Cache, layers);
        for (list) |*cache| cache.* = .{
            .keys = try a.alloc(f32, (shape.window - 1) * shape.width()),
            .values = try a.alloc(f32, (shape.window - 1) * shape.width()),
            .count = 0,
        };
        return list;
    }

    /// Attends `n` new rows (`q`, `k`, `v`) over the cached rows and their
    /// own into `out`, then keeps the last `window − 1` rows of both.
    pub fn attend(self: *Cache, shape: Shape, scratch: Scratch, q: []const f32, k: []const f32, v: []const f32, n: usize, out: []f32) void {
        const width = shape.width();
        const past = self.count;
        const total = past + n;
        const keys = scratch.keys[0 .. total * width];
        const values = scratch.values[0 .. total * width];
        @memcpy(keys[0 .. past * width], self.keys[0 .. past * width]);
        @memcpy(keys[past * width ..][0 .. n * width], k[0 .. n * width]);
        @memcpy(values[0 .. past * width], self.values[0 .. past * width]);
        @memcpy(values[past * width ..][0 .. n * width], v[0 .. n * width]);
        attention(shape, q, keys, values, n, past, scratch.scores, out);
        const kept = @min(shape.window - 1, total);
        @memcpy(self.keys[0 .. kept * width], keys[(total - kept) * width ..][0 .. kept * width]);
        @memcpy(self.values[0 .. kept * width], values[(total - kept) * width ..][0 .. kept * width]);
        self.count = kept;
    }
};

/// `Cache.attend`'s working rows for pushes of `n` rows at most, shared by
/// every layer.
pub const Scratch = struct {
    keys: []f32,
    values: []f32,
    scores: []f64,

    pub fn init(ws: Allocator, shape: Shape, n: usize) !Scratch {
        const rows = shape.window - 1 + n;
        return .{
            .keys = try ws.alloc(f32, rows * shape.width()),
            .values = try ws.alloc(f32, rows * shape.width()),
            .scores = try ws.alloc(f64, shape.window),
        };
    }
};

test "a cache fed one row at a time attends as one push of every row" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const shape: Shape = .{ .heads = 2, .head_dim = 3, .window = 4 };
    const rows = 9;
    const width = shape.width();
    const q = try a.alloc(f32, rows * width);
    const k = try a.alloc(f32, rows * width);
    const v = try a.alloc(f32, rows * width);
    for (q, k, v, 0..) |*x, *y, *z, i| {
        x.* = @sin(@as(f32, @floatFromInt(i)) * 0.7);
        y.* = @cos(@as(f32, @floatFromInt(i)) * 0.3);
        z.* = @as(f32, @floatFromInt(i % 7)) - 3;
    }
    const want = try a.alloc(f32, rows * width);
    attention(shape, q, k, v, rows, 0, try a.alloc(f64, shape.window), want);
    const caches = try Cache.many(a, shape, 1);
    const scratch: Scratch = try .init(a, shape, 1);
    const got = try a.alloc(f32, rows * width);
    for (0..rows) |t| caches[0].attend(shape, scratch, q[t * width ..][0..width], k[t * width ..][0..width], v[t * width ..][0..width], 1, got[t * width ..][0..width]);
    try std.testing.expectEqualSlices(f32, want, got);
    try std.testing.expectEqual(shape.window - 1, caches[0].count);
}
