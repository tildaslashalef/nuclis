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
//! `reset` and `bytes` for the load plan.
const std = @import("std");

pub const Drafter = struct {
    /// The family executor the methods act on; the caller passes the live
    /// address, so nothing here outlives a move.
    host: *anyopaque,
    /// The target hidden width `commit` consumes and `verify` reports per row.
    hidden: usize,
    /// Greedy candidates from the state after the last committed token, using
    /// `token` (the last chosen token not yet fed) as the block's seed.
    /// `logits`, when given, holds `out.len * vocabulary` values, one row per
    /// proposed position (ENGN-12's sampled acceptance). Returns the count
    /// proposed, at most `out.len`.
    propose_fn: *const fn (host: *anyopaque, token: u32, out: []u32, logits: ?[]f32) anyerror!usize,
    /// Advances the drafter over `tokens`, whose target hidden rows are
    /// `h_rows` (`tokens.len * hidden` values), as committed by `recover`.
    commit_fn: *const fn (host: *anyopaque, tokens: []const u32, h_rows: []const f32) anyerror!void,
    reset_fn: *const fn (host: *anyopaque) void,
    bytes_fn: *const fn (host: *anyopaque) usize,

    pub fn propose(self: Drafter, token: u32, out: []u32, logits: ?[]f32) !usize {
        return self.propose_fn(self.host, token, out, logits);
    }
    pub fn commit(self: Drafter, tokens: []const u32, h_rows: []const f32) !void {
        return self.commit_fn(self.host, tokens, h_rows);
    }
    pub fn reset(self: Drafter) void {
        self.reset_fn(self.host);
    }
    /// Bytes the drafter adds beyond the session (its workspace), for the
    /// memory record; the session reports its own cache region.
    pub fn bytes(self: Drafter) usize {
        return self.bytes_fn(self.host);
    }
};
