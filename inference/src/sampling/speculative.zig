//! Sampled speculative acceptance. Each verify row's decision is the target's
//! own draw from its shaped distribution (`Sampler.distribution`): the draft
//! is accepted when the draw equals it and the draw is the correction when
//! not; after the last accepted row the same draw is the bonus token. Every
//! emitted token is therefore a draw from the target whatever policy proposed
//! the drafts (the drafters chain greedy candidates); the drafts decide only
//! how many of a batch's draws are used. The `min(1, p/q)` rejection rule with
//! a residual correction is exact only for drafts sampled from `q`, which
//! ours are not, so it is deliberately not used.
const std = @import("std");
const sampling = @import("root.zig");

pub const Candidate = sampling.Candidate;
pub const Sampler = sampling.Sampler;

/// One verify row's decision.
pub const Verdict = union(enum) { accepted, correction: u32 };

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

/// Draws the target's token for a row from `p` and compares it with `draft`.
pub fn decide(sampler: *Sampler, p: []const Candidate, draft: u32) Verdict {
    const token = draw(sampler, p);
    return if (token == draft) .accepted else .{ .correction = token };
}

/// The token a row emits under `decide`: the draft when accepted, else the
/// correction. The tests' oracle.
fn emitted(sampler: *Sampler, p: []const Candidate, draft: u32) u32 {
    return switch (decide(sampler, p, draft)) {
        .accepted => draft,
        .correction => |token| token,
    };
}

test "the emitted token is a target draw whatever the draft" {
    var sampler = try Sampler.init(20260920, .{ .temperature = 1 });
    const p = [_]Candidate{ .{ .id = 0, .weight = 0.5 }, .{ .id = 1, .weight = 0.3 }, .{ .id = 2, .weight = 0.2 } };
    const trials = 20000;
    // A draft the target never proposes (id 3) is never accepted and the
    // emitted counts are still p; a likely draft (id 0) is accepted at p(0).
    for ([_]u32{ 3, 0 }) |draft| {
        var seen = [_]usize{0} ** 4;
        var accepted: usize = 0;
        for (0..trials) |_| {
            if (decide(&sampler, &p, draft) == .accepted) accepted += 1;
            seen[emitted(&sampler, &p, draft)] += 1;
        }
        const expected = [_]f64{ 0.5, 0.3, 0.2, 0 };
        for (expected, 0..) |prob, id| {
            const sigma = @sqrt(prob * (1 - prob) / trials);
            const observed = @as(f64, @floatFromInt(seen[id])) / trials;
            try std.testing.expect(@abs(observed - prob) <= 3 * sigma);
        }
        try std.testing.expectEqual(@as(usize, 0), seen[3]);
        const rate = @as(f64, @floatFromInt(accepted)) / trials;
        const want: f64 = if (draft == 3) 0 else 0.5;
        try std.testing.expect(@abs(rate - want) <= 3 * @sqrt(0.25 / @as(f64, trials)));
    }
}

test "a point-mass distribution accepts exactly its token" {
    var sampler = try Sampler.init(7, .{ .temperature = 1 });
    const only = [_]Candidate{.{ .id = 4, .weight = 1 }};
    for (0..100) |_| {
        try std.testing.expect(decide(&sampler, &only, 4) == .accepted);
        try std.testing.expectEqual(@as(u32, 4), decide(&sampler, &only, 9).correction);
    }
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
