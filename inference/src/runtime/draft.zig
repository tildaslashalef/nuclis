//! The model-independent draft-source contract. A family adapter that can
//! propose tokens (an MTP head, a DFlash drafter) exposes a `Drafter`; the
//! generation loop asks it, and never reaches into a family's block. The
//! host pointer is supplied per call so the value carries no self-reference
//! and survives the engine being returned by value.
//!
//! `commit` advances the drafter's own state over tokens the main model has
//! committed, with the target hidden of each; `propose` chains candidates
//! from the state after the last committed token. Recovery is the session's
//! (the drafter's cache is one more layout in it), so the contract has only
//! `reset`, `carried` (what a snapshot must copy besides the session), and
//! `bytes` for the load plan.
const std = @import("std");

pub const Drafter = struct {
    /// The family executor the methods act on; the caller passes the live
    /// address, so nothing here outlives a move.
    host: *anyopaque,
    /// The target hidden width `commit` consumes and `verify` reports per row.
    hidden: usize,
    /// The family's block bound: `propose` never returns more than this and
    /// `verify` refuses a batch above `max_proposals + 1` rows. The engine
    /// caps a requested draft length by it (Muse's DFlash block proposes 15
    /// per forward, the MTP heads 7).
    max_proposals: usize,
    /// Greedy candidates from the state after the last committed token, using
    /// `token` (the last chosen token not yet fed) as the block's seed.
    /// Returns the count proposed, at most `out.len`, where `stop` ends the
    /// chain (`Stop.next`); an inactive `stop` proposes every position `out`
    /// has room for. Verification needs no draft distribution: acceptance
    /// draws from the target alone.
    propose_fn: *const fn (host: *anyopaque, token: u32, out: []u32, stop: Stop) anyerror!usize,
    /// Advances the drafter over `tokens`, whose target hidden rows are
    /// `h_rows` (`tokens.len * hidden` values), as committed by `recover`.
    commit_fn: *const fn (host: *anyopaque, tokens: []const u32, h_rows: []const f32) anyerror!void,
    reset_fn: *const fn (host: *anyopaque) void,
    /// The state the drafter carries between calls outside the session (the
    /// pending target hidden row of an MTP head), empty when none. Mutable:
    /// a snapshot copies it out and a restore writes it back.
    carried_fn: *const fn (host: *anyopaque) []f32,
    bytes_fn: *const fn (host: *anyopaque) usize,

    pub fn propose(self: Drafter, token: u32, out: []u32, stop: Stop) !usize {
        return self.propose_fn(self.host, token, out, stop);
    }
    pub fn commit(self: Drafter, tokens: []const u32, h_rows: []const f32) !void {
        return self.commit_fn(self.host, tokens, h_rows);
    }
    pub fn reset(self: Drafter) void {
        self.reset_fn(self.host);
    }
    pub fn carried(self: Drafter) []f32 {
        return self.carried_fn(self.host);
    }
    /// Bytes the drafter adds beyond the session (its workspace), for the
    /// memory record; the session reports its own cache region.
    pub fn bytes(self: Drafter) usize {
        return self.bytes_fn(self.host);
    }
};

/// Where a greedy chain stops, from each position's top-candidate
/// probability `p` (the full-vocabulary softmax at its choice): after a
/// position whose `p` is below `p_min`, that position still proposed; and
/// before a position that would take the product of the chain's `p` below
/// `chain_min`, the first position always proposed. 0 turns either off.
pub const Stop = struct {
    p_min: f32 = 0,
    chain_min: f32 = 0,

    pub fn valid(self: Stop) bool {
        return std.math.isFinite(self.p_min) and self.p_min >= 0 and self.p_min <= 1 and
            std.math.isFinite(self.chain_min) and self.chain_min >= 0 and self.chain_min <= 1;
    }

    /// Whether a position's `p` is needed at all; an inactive stop never cuts.
    pub fn active(self: Stop) bool {
        return self.p_min > 0 or self.chain_min > 0;
    }

    /// After choosing position `index` (0-based) with top probability `p`:
    /// null to go on, or the count to propose (`index + 1` when it is the
    /// chain's last, `index` when it is not proposed). `chain` carries the
    /// product, 1 before the first position.
    pub fn cut(self: Stop, chain: *f64, index: usize, p: f64) ?usize {
        chain.* *= p;
        if (index > 0 and chain.* < self.chain_min) return index;
        return if (p < self.p_min) index + 1 else null;
    }
};

test "a stop cuts after a low position and before a low product" {
    const testing = std.testing;
    const run = struct {
        fn f(stop: Stop, ps: []const f64) usize {
            var chain: f64 = 1;
            for (ps, 0..) |p, i| if (stop.cut(&chain, i, p)) |n| return n;
            return ps.len;
        }
    }.f;
    const ps = [_]f64{ 0.9, 0.8, 0.6, 0.95 };
    try testing.expectEqual(@as(usize, 4), run(.{}, &ps));
    try testing.expectEqual(@as(usize, 3), run(.{ .p_min = 0.7 }, &ps));
    // 0.9, 0.72, 0.432: the third position falls below 0.5.
    try testing.expectEqual(@as(usize, 2), run(.{ .chain_min = 0.5 }, &ps));
    try testing.expectEqual(@as(usize, 2), run(.{ .p_min = 0.85, .chain_min = 0.5 }, &ps));
    // The first position is proposed whatever its product.
    try testing.expectEqual(@as(usize, 1), run(.{ .chain_min = 0.99 }, &ps));
    try testing.expect(!(Stop{ .chain_min = 1.5 }).valid());
    try testing.expect(!(Stop{}).active());
}
