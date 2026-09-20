//! Sampled speculative acceptance: the target `p` and draft `q` distributions
//! from `Sampler.distribution`, the `min(1, p/q)` accept test, and the
//! residual correction `normalize(max(0, p - q))`. The generation loop draws
//! the draft from `q`, accepts it against `p`, and on rejection draws the
//! correction from the residual; when every draft is accepted it draws the
//! bonus from the last target row. A draft absent from `p` is probability 0
//! and rejected. The module is pure but for `Sampler.rng`, so the seeded
//! tests hold it to the target distribution.
const std = @import("std");
const sampling = @import("root.zig");

pub const Candidate = sampling.Candidate;
pub const Sampler = sampling.Sampler;

/// The probability of `id` in a normalized distribution, or 0 when absent.
pub fn probability(candidates: []const Candidate, id: u32) f64 {
    for (candidates) |c| if (c.id == id) return c.weight;
    return 0;
}

/// Whether to accept `draft` under target `p` and draft `q`, with probability
/// `min(1, p(draft) / q(draft))`. A draft absent from `p` (probability 0) or
/// absent from `q` (impossible for a drawn draft) is rejected.
pub fn accept(sampler: *Sampler, p: []const Candidate, q: []const Candidate, draft: u32) bool {
    const qd = probability(q, draft);
    if (qd <= 0) return false;
    const pd = probability(p, draft);
    if (pd >= qd) return true;
    return sampler.rng.random().float(f64) < pd / qd;
}

/// The normalized correction distribution `max(0, p - q)`: every `p` candidate
/// whose `p` exceeds its `q` weight survives; the result sums to 1, or is
/// empty when `q` dominates `p` everywhere (the caller then draws from `p`).
/// `scratch` must hold at least `p.len` candidates and is borrowed.
pub fn residual(p: []const Candidate, q: []const Candidate, scratch: []Candidate) ![]Candidate {
    if (scratch.len < p.len) return error.InsufficientScratch;
    var count: usize = 0;
    for (p) |pc| {
        const weight = pc.weight - probability(q, pc.id);
        if (weight > 0) {
            scratch[count] = .{ .id = pc.id, .weight = weight };
            count += 1;
        }
    }
    if (count == 0) return scratch[0..0];
    var sum: f64 = 0;
    for (scratch[0..count]) |c| sum += c.weight;
    for (scratch[0..count]) |*c| c.weight /= sum;
    return scratch[0..count];
}

/// One seeded draw from a normalized, non-empty distribution.
pub fn draw(sampler: *Sampler, candidates: []const Candidate) u32 {
    std.debug.assert(candidates.len > 0);
    var r = sampler.rng.random().float(f64);
    for (candidates) |c| {
        if (r < c.weight) return c.id;
        r -= c.weight;
    }
    return candidates[candidates.len - 1].id;
}

/// The full speculative draw: sample a draft from `q`, accept it against `p`,
/// and otherwise fall back to the residual correction (or `p` when the
/// residual is empty). Exact for full `p`/`q`; its empirical counts are the
/// tests' oracle.
pub fn sample(sampler: *Sampler, p: []const Candidate, q: []const Candidate, scratch: []Candidate) !u32 {
    const draft = draw(sampler, q);
    if (accept(sampler, p, q, draft)) return draft;
    const correction = try residual(p, q, scratch);
    return draw(sampler, if (correction.len == 0) p else correction);
}

fn counts(id: u32, seen: []usize) void {
    seen[id] += 1;
}

test "speculative acceptance reproduces full target distributions" {
    const alloc = std.testing.allocator;
    var sampler = try Sampler.init(20260920, .{ .temperature = 1 });
    // Distinct supports so the union path and the zero-probability rule run.
    const p = [_]Candidate{ .{ .id = 0, .weight = 0.5 }, .{ .id = 1, .weight = 0.3 }, .{ .id = 2, .weight = 0.2 } };
    const q = [_]Candidate{ .{ .id = 0, .weight = 0.25 }, .{ .id = 1, .weight = 0.25 }, .{ .id = 3, .weight = 0.5 } };
    const scratch = try alloc.alloc(Candidate, p.len);
    defer alloc.free(scratch);
    const trials = 20000;
    var seen = [_]usize{0} ** 4;
    for (0..trials) |_| counts(try sample(&sampler, &p, &q, scratch), &seen);
    // The output distribution is exactly p: token 3, which `q` offers but `p`
    // does not, must never appear.
    const expected = [_]f64{ 0.5, 0.3, 0.2, 0 };
    for (expected, 0..) |prob, id| {
        const sigma = @sqrt(prob * (1 - prob) / trials);
        const observed = @as(f64, @floatFromInt(seen[id])) / trials;
        try std.testing.expect(@abs(observed - prob) <= 3 * sigma);
    }
    try std.testing.expectEqual(@as(usize, 0), seen[3]);
}

test "full acceptance and full rejection" {
    const alloc = std.testing.allocator;
    var sampler = try Sampler.init(7, .{ .temperature = 1 });
    const scratch = try alloc.alloc(Candidate, 2);
    defer alloc.free(scratch);
    // p == q: the draft is always accepted, so the draws are pure q.
    const same = [_]Candidate{ .{ .id = 0, .weight = 0.75 }, .{ .id = 1, .weight = 0.25 } };
    for (0..1000) |_| try std.testing.expect(try sample(&sampler, &same, &same, scratch) < 2);
    // q offers only a token absent from p: every draft is rejected and the
    // correction is p's only candidate.
    const only = [_]Candidate{.{ .id = 0, .weight = 1 }};
    const other = [_]Candidate{.{ .id = 1, .weight = 1 }};
    for (0..1000) |_| try std.testing.expectEqual(@as(u32, 0), try sample(&sampler, &only, &other, scratch));
    // A zero-probability draft is rejected by `accept`, never accepted.
    try std.testing.expect(!accept(&sampler, &only, &other, 1));
    try std.testing.expect(accept(&sampler, &same, &same, 1));
}

test "residual normalizes max(0, p - q) and empties when q dominates" {
    const p = [_]Candidate{ .{ .id = 0, .weight = 0.6 }, .{ .id = 1, .weight = 0.4 } };
    const q = [_]Candidate{ .{ .id = 0, .weight = 0.2 }, .{ .id = 1, .weight = 0.8 } };
    var scratch: [2]Candidate = undefined;
    const r = try residual(&p, &q, &scratch);
    try std.testing.expectEqual(@as(usize, 1), r.len);
    try std.testing.expectEqual(@as(u32, 0), r[0].id);
    try std.testing.expectApproxEqAbs(@as(f64, 1), r[0].weight, 1e-12);
    // q at least p everywhere: the residual is empty and the caller draws p.
    const dominated = [_]Candidate{ .{ .id = 0, .weight = 0.9 }, .{ .id = 1, .weight = 0.9 } };
    try std.testing.expectEqual(@as(usize, 0), (try residual(&p, &dominated, &scratch)).len);
    try std.testing.expectError(error.InsufficientScratch, residual(&p, &q, scratch[0..1]));
}

test "distribution is the shaped nucleus and greedy is its argmax" {
    const alloc = std.testing.allocator;
    const scratch = try alloc.alloc(Candidate, 5);
    defer alloc.free(scratch);
    const logits = [_]f32{ 3, 2, 1, 0, -1 };
    var greedy = try Sampler.init(0, .{});
    const picked = try greedy.distribution(&logits, scratch, null);
    try std.testing.expectEqual(@as(usize, 1), picked.len);
    try std.testing.expectEqual(@as(u32, 0), picked[0].id);
    try std.testing.expectEqual(@as(f64, 1), picked[0].weight);

    // top-k 2 then the nucleus; weights are probabilities summing to 1.
    var cut = try Sampler.init(0, .{ .temperature = 1, .top_k = 2 });
    const shaped = try cut.distribution(&logits, scratch, null);
    try std.testing.expectEqual(@as(usize, 2), shaped.len);
    var sum: f64 = 0;
    for (shaped) |c| sum += c.weight;
    try std.testing.expectApproxEqAbs(@as(f64, 1), sum, 1e-12);
    try std.testing.expect(shaped[0].weight > shaped[1].weight);
    // min_p drops the tail: only the top candidate survives at min_p 1.
    var narrow = try Sampler.init(0, .{ .temperature = 1, .min_p = 1 });
    const one = try narrow.distribution(&logits, scratch, null);
    try std.testing.expectEqual(@as(usize, 1), one.len);
    try std.testing.expectEqual(@as(u32, 0), one[0].id);
    // Penalties shape the distribution like `select`'s argmax.
    var history = try sampling.History.init(std.testing.allocator, 5);
    defer history.deinit();
    try history.observe(0);
    var penalized = try Sampler.init(0, .{ .temperature = 1, .presence_penalty = 4 });
    const shifted = try penalized.distribution(&logits, scratch, &history);
    try std.testing.expect(shifted[0].id != 0);
}
