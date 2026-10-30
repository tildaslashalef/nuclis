//! Fish's Drama 3 sampler (`tower.py` `sample_top_k_top_p`) and the repeat
//! window on the main token (RAS). Model-specific, so it lives beside the
//! adapter rather than in `sampling/`. The draws are Fish's distribution, not
//! its random stream. docs/models/drama3.md § Sampling.
const std = @import("std");

const Allocator = std.mem.Allocator;

pub const Params = struct {
    temperature: f32 = 1.0,
    top_p: f32 = 0.9,
    top_k: u32 = 30,
};

/// The second draw that replaces a repeated semantic main token.
pub const Repeat = struct {
    window: usize = 16,
    params: Params = .{ .temperature = 1.0, .top_p = 0.95, .top_k = 30 },
};

pub const min_temperature = 1e-5;

pub const Error = Allocator.Error || error{NonFiniteLogits};

pub const Sampler = struct {
    rng: std.Random.DefaultPrng,
    /// Column order scratch, `width` long; owned.
    order: []u32,
    weights: []f64,

    pub fn init(gpa: Allocator, seed: u64, width: usize) Allocator.Error!Sampler {
        const order = try gpa.alloc(u32, width);
        errdefer gpa.free(order);
        return .{ .rng = .init(seed), .order = order, .weights = try gpa.alloc(f64, width) };
    }

    pub fn deinit(self: *Sampler, gpa: Allocator) void {
        gpa.free(self.order);
        gpa.free(self.weights);
        self.* = undefined;
    }

    /// One column of `logits` (at most `width`). Columns ranked at or past
    /// `top_k`, and those where the cumulative probability exceeds `top_p`
    /// (the one crossing it included), are dropped, except the highest-ranked;
    /// the kept ones are drawn in proportion to their probability.
    pub fn draw(self: *Sampler, logits: []const f32, p: Params) Error!u32 {
        std.debug.assert(logits.len <= self.order.len and logits.len > 0);
        const inv: f64 = 1.0 / @max(@as(f64, p.temperature), min_temperature);
        var max: f64 = -std.math.inf(f64);
        for (logits) |l| {
            if (!std.math.isFinite(l)) return error.NonFiniteLogits;
            max = @max(max, @as(f64, l) * inv);
        }
        var total: f64 = 0;
        for (logits) |l| total += @exp(@as(f64, l) * inv - max);
        const k: usize = @min(@max(p.top_k, 1), logits.len);
        const ranked = self.top(logits, k);
        var cumulative: f64 = 0;
        var kept: usize = 0;
        for (ranked, 0..) |column, rank| {
            const probability = @exp(@as(f64, logits[column]) * inv - max) / total;
            cumulative += probability;
            if (rank > 0 and cumulative > p.top_p) break;
            self.weights[rank] = probability;
            kept = rank + 1;
        }
        // argmax of p / Exp(1): exponential-race sampling.
        var best: usize = 0;
        var best_score: f64 = -1;
        for (self.weights[0..kept], 0..) |w, rank| {
            const noise = -@log(1.0 - self.rng.random().float(f64));
            const score = w / noise;
            if (score > best_score) {
                best_score = score;
                best = rank;
            }
        }
        return ranked[best];
    }

    /// The `k` highest columns in descending order, lowest index first on ties.
    fn top(self: *Sampler, logits: []const f32, k: usize) []const u32 {
        const order = self.order[0..logits.len];
        for (order, 0..) |*o, i| o.* = @intCast(i);
        const Context = struct {
            logits: []const f32,
            fn greater(ctx: @This(), a: u32, b: u32) bool {
                return if (ctx.logits[a] != ctx.logits[b]) ctx.logits[a] > ctx.logits[b] else a < b;
            }
        };
        if (k * 8 < logits.len) {
            // Partial selection: keep the best k at the front by insertion.
            var n: usize = 0;
            for (0..logits.len) |column| {
                const c: u32 = @intCast(column);
                if (n == k and !Context.greater(.{ .logits = logits }, c, order[k - 1])) continue;
                var at = @min(n, k - 1);
                while (at > 0 and Context.greater(.{ .logits = logits }, c, order[at - 1])) : (at -= 1) order[at] = order[at - 1];
                order[at] = c;
                if (n < k) n += 1;
            }
            return order[0..k];
        }
        std.mem.sort(u32, order, Context{ .logits = logits }, Context.greater);
        return order[0..k];
    }

    /// The main token from the 4,097 head logits (column 0 is EOS, column
    /// c ≥ 1 is semantic code c − 1). A semantic first draw found among the
    /// last `repeat.window` main tokens of `recent` is replaced by a draw at
    /// `repeat.params`; EOS is never redrawn. Returns the column.
    pub fn mainColumn(self: *Sampler, logits: []const f32, p: Params, repeat: Repeat, recent: []const u32, semantic_start: u32) Error!u32 {
        const first = try self.draw(logits, p);
        if (first == 0) return first;
        const id = semantic_start + first - 1;
        const window = recent[recent.len -| repeat.window..];
        if (std.mem.indexOfScalar(u32, window, id) == null) return first;
        return self.draw(logits, repeat.params);
    }
};

const testing = std.testing;

test "top-k one and a tiny top-p are greedy" {
    var s = try Sampler.init(testing.allocator, 1, 8);
    defer s.deinit(testing.allocator);
    const logits = [_]f32{ 0.1, 2.0, 1.9, -3, 0, 0, 0, 0 };
    for (0..50) |_| {
        try testing.expectEqual(@as(u32, 1), try s.draw(&logits, .{ .top_k = 1, .top_p = 1 }));
        try testing.expectEqual(@as(u32, 1), try s.draw(&logits, .{ .top_k = 8, .top_p = 0.01 }));
    }
    const bad = [_]f32{ 0, std.math.nan(f32) };
    try testing.expectError(error.NonFiniteLogits, s.draw(&bad, .{}));
}

test "the column crossing top-p is dropped, and the kept ones are drawn in proportion" {
    var s = try Sampler.init(testing.allocator, 7, 3);
    defer s.deinit(testing.allocator);
    // Probabilities 0.5, 0.3, 0.2.
    const logits = [_]f32{ @log(0.5), @log(0.3), @log(0.2) };
    for (0..100) |_| try testing.expectEqual(@as(u32, 0), try s.draw(&logits, .{ .top_p = 0.7 }));
    var counts: [3]usize = @splat(0);
    const n = 40_000;
    for (0..n) |_| counts[try s.draw(&logits, .{ .top_p = 0.85 })] += 1;
    try testing.expectEqual(@as(usize, 0), counts[2]);
    const share = @as(f64, @floatFromInt(counts[0])) / n;
    try testing.expect(@abs(share - 0.625) < 0.01);
}

test "partial selection ranks like a full sort" {
    var s = try Sampler.init(testing.allocator, 3, 4097);
    defer s.deinit(testing.allocator);
    var logits: [4097]f32 = undefined;
    var rng: std.Random.DefaultPrng = .init(11);
    for (&logits) |*l| l.* = @floatFromInt(rng.random().intRangeAtMost(i32, -50, 50));
    const partial = try testing.allocator.dupe(u32, s.top(&logits, 30));
    defer testing.allocator.free(partial);
    const full = s.top(&logits, 4097);
    try testing.expectEqualSlices(u32, full[0..30], partial);
}

test "a repeated semantic main token is redrawn, EOS is not" {
    var s = try Sampler.init(testing.allocator, 5, 4);
    defer s.deinit(testing.allocator);
    // Greedy first draw is column 1; the redraw keeps columns 1 and 2 (the
    // third crosses top-p 0.95).
    const logits = [_]f32{ -20, 10, 9.9, 9.8 };
    const start: u32 = 100;
    var redrawn = false;
    for (0..200) |_| {
        const c = try s.mainColumn(&logits, .{ .top_k = 1 }, .{}, &.{ 7, start + 0 }, start);
        if (c == 2) redrawn = true;
        try testing.expect(c == 1 or c == 2);
    }
    try testing.expect(redrawn);
    // Outside the window: no redraw.
    var recent: [17]u32 = @splat(1);
    recent[0] = start;
    for (0..50) |_| try testing.expectEqual(@as(u32, 1), try s.mainColumn(&logits, .{ .top_k = 1 }, .{}, &recent, start));
    const eos = [_]f32{ 10, -20, -20, -20 };
    for (0..50) |_| try testing.expectEqual(@as(u32, 0), try s.mainColumn(&eos, .{ .top_k = 1 }, .{}, &.{248046}, start));
}
