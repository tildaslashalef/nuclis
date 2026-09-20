//! Seeded token selection, independent of model architecture and presentation.
//! The caller owns/reuses vocabulary-sized candidate storage. Equal logits use
//! ascending token IDs; validation finishes before RNG state advances.
//!
//! The selection chain, in the order the tests pin it: presence and
//! repetition penalties on the raw logits of tokens in the `History`
//! (conversation state, owned by the caller), then the argmax for greedy
//! decoding; otherwise the sort, temperature (in `exponentiate`), the top-k
//! cut, the `min_p` prefix filter, the top-p walk over the survivors, and one
//! seeded draw.
//!
//! Two entry points share one truncation-and-draw core (`finish`):
//! - `select` sorts the whole vocabulary on the CPU (the reference path).
//! - `selectFrom` consumes a `TopK` readback (the GPU's best `capacity` logits
//!   in the same order plus the softmax denominator) and either returns the
//!   token the reference path would return, bit for bit, or `null` when only
//!   the full vocabulary can decide. Exactness comes from doing every
//!   floating-point step that touches the chosen token on the CPU in the same
//!   order; the GPU only supplies the candidates and one sum whose error is
//!   bounded and guarded (see `total_band`). Penalties alter the sort itself,
//!   so a sampler with an active penalty is never GPU-eligible.
const std = @import("std");

/// Neutral defaults: greedy, no cut, no penalty. `Sampler.init` validates.
pub const Options = struct {
    temperature: f32 = 0,
    top_k: usize = 0,
    top_p: f32 = 1,
    /// Discards candidates whose probability is below `min_p · p_max`; 0
    /// disables the filter. Applied after temperature and top-k, before top-p.
    min_p: f32 = 0,
    /// Subtracted from the logit of every token already in the history.
    presence_penalty: f32 = 0,
    /// Divides positive (multiplies negative) logits of tokens in the history;
    /// 1 is neutral. Applied before `presence_penalty`.
    repetition_penalty: f32 = 1,

    /// Whether the history changes any logit, which forces the full-logit
    /// path: the GPU readback is taken after the projection, before the CPU
    /// could adjust the vector.
    pub fn penaltiesActive(self: Options) bool {
        return self.presence_penalty != 0 or self.repetition_penalty != 1;
    }

    /// Returns a copy with every non-null override applied. `Overrides`
    /// mirrors this struct with optional fields so a caller (a prompt profile's
    /// per-mode defaults plus command-line flags) can state only what it sets.
    /// The `inline for` walks the fields at compile time: one loop body per
    /// field, no runtime reflection.
    pub fn override(self: Options, overrides: Overrides) Options {
        var result = self;
        inline for (@typeInfo(Overrides).@"struct".fields) |field| {
            if (@field(overrides, field.name)) |value| @field(result, field.name) = value;
        }
        return result;
    }
};

pub const Overrides = struct {
    temperature: ?f32 = null,
    top_k: ?usize = null,
    top_p: ?f32 = null,
    min_p: ?f32 = null,
    presence_penalty: ?f32 = null,
    repetition_penalty: ?f32 = null,

    /// Returns `self` with every non-null option of `top` replacing it, so
    /// layers of overrides (a configuration file, then flags) compose before
    /// they are applied to a profile.
    pub fn merge(self: Overrides, top: Overrides) Overrides {
        var result = self;
        inline for (@typeInfo(Overrides).@"struct".fields) |field| {
            if (@field(top, field.name)) |value| @field(result, field.name) = value;
        }
        return result;
    }
};

test "overrides merge per option, the top layer winning" {
    const base: Overrides = .{ .temperature = 0.5, .top_k = 40 };
    const merged = base.merge(.{ .top_k = 10, .min_p = 0.1 });
    try std.testing.expectEqual(Overrides{ .temperature = 0.5, .top_k = 10, .min_p = 0.1 }, merged);
    try std.testing.expectEqual(base, base.merge(.{}));
    try std.testing.expectEqual(Options{ .temperature = 0.5, .top_k = 10, .min_p = 0.1 }, (Options{}).override(merged));
}

pub const Candidate = struct { id: u32, weight: f64 };

/// The set of token IDs a session has consumed or produced, for the penalties.
/// Conversation state, not policy: the caller creates one per session,
/// `observe`s every prompt and generated token (the engine loop does this),
/// and `reset`s it wherever the model session is reset. One bit per
/// vocabulary entry (31 KB for the pinned model); order and counts are not
/// kept because the penalties are defined on presence only.
pub const History = struct {
    alloc: std.mem.Allocator,
    seen: std.DynamicBitSetUnmanaged,

    pub fn init(alloc: std.mem.Allocator, vocabulary: usize) !History {
        if (vocabulary == 0 or vocabulary > std.math.maxInt(u32)) return error.InvalidLogits;
        return .{ .alloc = alloc, .seen = try std.DynamicBitSetUnmanaged.initEmpty(alloc, vocabulary) };
    }
    pub fn deinit(self: *History) void {
        self.seen.deinit(self.alloc);
        self.* = undefined;
    }
    pub fn observe(self: *History, id: u32) !void {
        if (id >= self.seen.bit_length) return error.InvalidToken;
        self.seen.set(id);
    }
    pub fn contains(self: *const History, id: u32) bool {
        return id < self.seen.bit_length and self.seen.isSet(id);
    }
    pub fn reset(self: *History) void {
        self.seen.unsetAll();
    }
    pub fn count(self: *const History) usize {
        return self.seen.count();
    }
};

/// The GPU partial top-k readback. `ids`/`values` are the `count` best
/// logits by (value descending, id ascending) — the reference sort order —
/// and `total` is Σ exp((l − max) / temperature) over *every* logit, an F64
/// sum of F32 partial sums (so its relative error is bounded, not zero).
/// `finite` reports whether every logit was finite; the reference path
/// rejects non-finite logits, so a readback with `finite == false` falls back.
pub const TopK = struct {
    pub const capacity = 256;
    /// The temperature the GPU used for `total`; must equal the sampler's.
    temperature: f32,
    count: usize = 0,
    ids: [capacity]u32 = undefined,
    values: [capacity]f32 = undefined,
    total: f64 = 0,
    finite: bool = false,
};

/// Relative uncertainty allowed on `TopK.total`. The nucleus decision
/// `retained >= top_p · total` is taken only when `retained` lies outside
/// `total · (1 ± total_band)`; inside the band the sampler asks for the full
/// vocabulary. `metal-check` asserts the GPU sum's error is ≤ 2e-6, so the
/// band holds with a 5× margin (see reference/generation.md).
pub const total_band: f64 = 1e-5;

pub const Sampler = struct {
    rng: std.Random.DefaultPrng,
    options: Options,
    pub fn init(seed: u64, options: Options) !Sampler {
        try validate(options);
        return .{ .rng = .init(seed), .options = options };
    }

    /// Replaces the options (a chat rebuilds them per turn from the current
    /// reasoning mode) without touching the RNG; validated like `init`.
    pub fn setOptions(self: *Sampler, options: Options) !void {
        try validate(options);
        self.options = options;
    }

    fn validate(o: Options) !void {
        if (!std.math.isFinite(o.temperature) or o.temperature < 0) return error.InvalidSamplingOptions;
        if (!std.math.isFinite(o.top_p) or o.top_p <= 0 or o.top_p > 1) return error.InvalidSamplingOptions;
        if (!std.math.isFinite(o.min_p) or o.min_p < 0 or o.min_p > 1) return error.InvalidSamplingOptions;
        if (!std.math.isFinite(o.presence_penalty)) return error.InvalidSamplingOptions;
        if (!std.math.isFinite(o.repetition_penalty) or o.repetition_penalty <= 0) return error.InvalidSamplingOptions;
    }

    /// Whether a `TopK` readback can decide tokens for these options: sampled
    /// (`temperature > 0`), no penalty (they change the sort), and either
    /// `1 ≤ top_k ≤ capacity` (the retained set is inside the readback, always
    /// exact) or `top_k = 0` with a nucleus (`top_p < 1`, guarded by `total`)
    /// or a `min_p` filter (its survivors are a prefix of the readback unless
    /// all of it survives, in which case the sampler defers). `top_k >
    /// capacity` and the full distribution (`top_k = 0, top_p = 1, min_p = 0`)
    /// need every logit and are not eligible.
    pub fn gpuEligible(self: *const Sampler) bool {
        const o = self.options;
        if (o.temperature == 0 or o.penaltiesActive()) return false;
        if (o.top_k >= 1 and o.top_k <= TopK.capacity) return true;
        return o.top_k == 0 and (o.top_p < 1 or o.min_p > 0);
    }

    /// Reference path: every logit, penalized against `history` when one is
    /// given, sorted on the CPU. Greedy decoding needs no `scratch`.
    pub fn select(self: *Sampler, logits: []const f32, scratch: []Candidate, history: ?*const History) !u32 {
        if (logits.len == 0 or logits.len > std.math.maxInt(u32)) return error.InvalidLogits;
        var best: usize = 0;
        var best_value: f32 = undefined;
        for (logits, 0..) |raw, i| {
            const value = self.penalize(raw, i, history);
            if (!std.math.isFinite(value)) return error.NonFiniteResult;
            if (i == 0 or value > best_value) {
                best = i;
                best_value = value;
            }
        }
        if (self.options.temperature == 0) return @intCast(best);
        if (scratch.len < logits.len) return error.InsufficientScratch;
        const candidates = scratch[0..logits.len];
        for (candidates, logits, 0..) |*c, raw, id| c.* = .{ .id = @intCast(id), .weight = self.penalize(raw, id, history) };
        std.mem.sort(Candidate, candidates, {}, lessThan);
        const k = if (self.options.top_k == 0) candidates.len else @min(self.options.top_k, candidates.len);
        const retained = self.exponentiate(candidates[0..k]);
        // The whole retained set is present, so the decision is exact and
        // `finish` cannot return null.
        return self.finish(retained.survivors, retained.sum, 0, true).?;
    }

    /// The shaped, normalized distribution `select` draws from: penalties,
    /// sort, top-k, temperature and `min_p`, then the top-p nucleus, with the
    /// retained candidates' weights normalized to probabilities summing to 1.
    /// Greedy (`temperature == 0`) returns the single argmax with weight 1, so
    /// the result is never empty. The slice borrows `scratch`; `scratch.len`
    /// must be at least `logits.len` for the sampled path and 1 for greedy.
    pub fn distribution(self: *const Sampler, logits: []const f32, scratch: []Candidate, history: ?*const History) ![]Candidate {
        if (logits.len == 0 or logits.len > std.math.maxInt(u32)) return error.InvalidLogits;
        var best: usize = 0;
        var best_value: f32 = undefined;
        for (logits, 0..) |raw, i| {
            const value = self.penalize(raw, i, history);
            if (!std.math.isFinite(value)) return error.NonFiniteResult;
            if (i == 0 or value > best_value) {
                best = i;
                best_value = value;
            }
        }
        if (self.options.temperature == 0) {
            if (scratch.len < 1) return error.InsufficientScratch;
            scratch[0] = .{ .id = @intCast(best), .weight = 1 };
            return scratch[0..1];
        }
        if (scratch.len < logits.len) return error.InsufficientScratch;
        const candidates = scratch[0..logits.len];
        for (candidates, logits, 0..) |*c, raw, id| c.* = .{ .id = @intCast(id), .weight = self.penalize(raw, id, history) };
        std.mem.sort(Candidate, candidates, {}, lessThan);
        const k = if (self.options.top_k == 0) candidates.len else @min(self.options.top_k, candidates.len);
        const retained = self.exponentiate(candidates[0..k]);
        // The nucleus walk of `finish` without the draw: stop after the first
        // candidate that reaches the top-p threshold, so the retained set is
        // the whole support `select` would draw from.
        const threshold = @as(f64, self.options.top_p) * retained.sum;
        var count: usize = 0;
        var accumulated: f64 = 0;
        while (count < retained.survivors.len) {
            accumulated += retained.survivors[count].weight;
            count += 1;
            if (accumulated >= threshold) break;
        }
        const kept = retained.survivors[0..count];
        for (kept) |*c| c.weight /= accumulated;
        return kept;
    }

    /// GPU path: returns the token `select` would return for the same logits
    /// and RNG state, or `null` when the readback cannot decide (the caller
    /// then reads the full logits and calls `select`; the RNG has not advanced).
    /// Active penalties always defer: the readback was taken before any
    /// history could be applied.
    pub fn selectFrom(self: *Sampler, top: *const TopK, scratch: []Candidate) !?u32 {
        if (top.count == 0 or top.count > TopK.capacity) return error.InvalidLogits;
        if (!top.finite) return null; // `select` reports NonFiniteResult
        if (self.options.penaltiesActive()) return null;
        if (self.options.temperature == 0) return top.ids[0];
        if (top.temperature != self.options.temperature) return error.InvalidSamplingOptions;
        if (scratch.len < top.count) return error.InsufficientScratch;
        const candidates = scratch[0..top.count];
        for (candidates, top.ids[0..top.count], top.values[0..top.count]) |*c, id, value| {
            if (!std.math.isFinite(value)) return error.NonFiniteResult;
            c.* = .{ .id = id, .weight = value };
        }
        // The kernels are tested to produce the reference order; a violation
        // here is a backend bug, not a sampling decision.
        for (candidates[1..], candidates[0 .. candidates.len - 1]) |next, prev| if (lessThan({}, next, prev)) return error.InvalidLogits;
        const o = self.options;
        if (o.top_k >= 1 and o.top_k <= TopK.capacity) {
            const k = @min(o.top_k, candidates.len);
            const retained = self.exponentiate(candidates[0..k]);
            return self.finish(retained.survivors, retained.sum, 0, true);
        }
        if (o.top_k != 0) return null;
        const retained = self.exponentiate(candidates);
        if (o.min_p > 0) {
            // The survivors are a prefix of the sorted vocabulary. When the
            // whole readback survives the prefix may continue past it, and
            // only the full path knows where it ends.
            if (retained.survivors.len == candidates.len) return null;
            return self.finish(retained.survivors, retained.sum, 0, true);
        }
        if (o.top_p >= 1) return null;
        return self.finish(candidates, top.total, total_band, false);
    }

    /// The logit after the history penalties: repetition first (`l / r` for
    /// positive, `l · r` for negative logits), then `l − presence`. Neutral
    /// options are exact identities in F32, so no branch is needed on them.
    fn penalize(self: *const Sampler, value: f32, id: usize, history: ?*const History) f32 {
        const h = history orelse return value;
        if (!h.contains(@intCast(id))) return value;
        const r = self.options.repetition_penalty;
        const scaled = if (value > 0) value / r else value * r;
        return scaled - self.options.presence_penalty;
    }

    fn lessThan(_: void, a: Candidate, b: Candidate) bool {
        return a.weight > b.weight or (a.weight == b.weight and a.id < b.id);
    }

    const Retained = struct { survivors: []Candidate, sum: f64 };

    /// Replaces raw logits with softmax numerators relative to the first
    /// (largest) candidate, then applies the `min_p` filter: a candidate's
    /// weight is exactly `p / p_max`, so the survivors are the leading
    /// candidates with `weight ≥ min_p` (the weights are non-increasing along
    /// the sort). Returns the survivors and their sum in order. Shared by
    /// both paths so the weights and sums are bit-identical for identical
    /// candidates.
    fn exponentiate(self: *const Sampler, candidates: []Candidate) Retained {
        const maximum = candidates[0].weight;
        const floor: f64 = self.options.min_p;
        var sum: f64 = 0;
        var kept: usize = 0;
        for (candidates, 0..) |*c, i| {
            c.weight = @exp((c.weight - maximum) / self.options.temperature);
            // The first weight is exp(0) = 1 ≥ min_p, so at least one survives;
            // `kept == i` keeps the survivors a prefix even if the weights
            // were ever not monotone.
            if (kept == i and c.weight >= floor) {
                sum += c.weight;
                kept += 1;
            }
        }
        return .{ .survivors = candidates[0..kept], .sum = sum };
    }

    /// Nucleus truncation over exponentiated `candidates` (the top-k set or
    /// the readback), then one seeded draw. `sum` is the denominator the
    /// reference path uses; when it carries uncertainty `band > 0`, a
    /// comparison that could go either way returns null before the RNG
    /// advances. `complete` says the candidates are the whole retained set;
    /// otherwise running past them (the nucleus is larger than the readback)
    /// also returns null.
    fn finish(self: *Sampler, candidates: []const Candidate, sum: f64, band: f64, complete: bool) ?u32 {
        const threshold = @as(f64, self.options.top_p) * sum;
        var count: usize = 0;
        var retained: f64 = 0;
        var decided = false;
        while (count < candidates.len) {
            retained += candidates[count].weight;
            count += 1;
            if (band == 0) {
                if (retained >= threshold) {
                    decided = true;
                    break;
                }
            } else {
                if (retained >= threshold * (1 + band)) {
                    decided = true;
                    break;
                }
                if (retained >= threshold * (1 - band)) return null;
            }
        }
        if (!decided and !complete) return null;
        var draw = self.rng.random().float(f64) * retained;
        for (candidates[0..count]) |c| {
            if (draw < c.weight) return c.id;
            draw -= c.weight;
        }
        return candidates[count - 1].id;
    }
};

test "greedy and seeded truncated distributions" {
    var sampler = try Sampler.init(9, .{});
    try std.testing.expectEqual(@as(u32, 1), try sampler.select(&.{ -1, 2, 2 }, &.{}, null));
    var scratch: [3]Candidate = undefined;
    sampler = try Sampler.init(9, .{ .temperature = 1, .top_k = 1 });
    for (0..10) |_| try std.testing.expectEqual(@as(u32, 1), try sampler.select(&.{ -1, 2, 2 }, &scratch, null));
    sampler = try Sampler.init(9, .{ .temperature = 1, .top_p = 0.1 });
    try std.testing.expectEqual(@as(u32, 0), try sampler.select(&.{ 0, 0, 0 }, &scratch, null));
    var a = try Sampler.init(42, .{ .temperature = 0.7 });
    var b = try Sampler.init(42, .{ .temperature = 0.7 });
    var seen: u8 = 0;
    for (0..100) |_| {
        const id = try a.select(&.{ 0, 0, 0 }, &scratch, null);
        try std.testing.expectEqual(id, try b.select(&.{ 0, 0, 0 }, &scratch, null));
        seen |= @as(u8, 1) << @intCast(id);
    }
    try std.testing.expectEqual(@as(u8, 7), seen);
    try std.testing.expectError(error.NonFiniteResult, a.select(&.{std.math.nan(f32)}, &scratch, null));
    try std.testing.expectError(error.InsufficientScratch, a.select(&.{1}, &.{}, null));
    try std.testing.expectError(error.InvalidSamplingOptions, Sampler.init(0, .{ .top_p = 0 }));
}

test "every option is validated before the RNG exists" {
    const inf = std.math.inf(f32);
    const nan = std.math.nan(f32);
    const bad = [_]Options{
        .{ .temperature = -0.1 },
        .{ .temperature = nan },
        .{ .top_p = 1.5 },
        .{ .top_p = nan },
        .{ .min_p = -0.01 },
        .{ .min_p = 1.01 },
        .{ .min_p = nan },
        .{ .presence_penalty = inf },
        .{ .presence_penalty = nan },
        .{ .repetition_penalty = 0 },
        .{ .repetition_penalty = -1 },
        .{ .repetition_penalty = inf },
        .{ .repetition_penalty = nan },
    };
    for (bad) |o| try std.testing.expectError(error.InvalidSamplingOptions, Sampler.init(0, o));
    const good = [_]Options{
        .{},
        .{ .min_p = 0 },
        .{ .min_p = 1 },
        .{ .presence_penalty = -2 },
        .{ .repetition_penalty = 0.5 },
        .{ .temperature = 1, .top_p = 0.8, .top_k = 20, .min_p = 0.05, .presence_penalty = 1.5, .repetition_penalty = 1.1 },
    };
    for (good) |o| _ = try Sampler.init(0, o);
    var s = try Sampler.init(3, .{ .temperature = 1 });
    try std.testing.expectError(error.InvalidSamplingOptions, s.setOptions(.{ .min_p = 2 }));
    try std.testing.expectEqual(@as(f32, 1), s.options.temperature); // rejected options leave the sampler unchanged
    try s.setOptions(.{ .temperature = 0.7, .presence_penalty = 1.5 });
    try std.testing.expect(s.options.penaltiesActive());
    try std.testing.expect(!(Options{ .presence_penalty = 0, .repetition_penalty = 1 }).penaltiesActive());
    const merged = (Options{ .temperature = 1, .top_p = 0.95, .top_k = 20 }).override(.{ .top_k = 40, .min_p = 0.1 });
    try std.testing.expectEqual(@as(f32, 1), merged.temperature);
    try std.testing.expectEqual(@as(usize, 40), merged.top_k);
    try std.testing.expectEqual(@as(f32, 0.1), merged.min_p);
    try std.testing.expectEqual(@as(f32, 0.95), merged.top_p);
}

test "history observes token ids within the vocabulary and resets" {
    var history = try History.init(std.testing.allocator, 10);
    defer history.deinit();
    try std.testing.expect(!history.contains(3));
    try history.observe(3);
    try history.observe(3);
    try history.observe(9);
    try std.testing.expect(history.contains(3) and history.contains(9) and !history.contains(0));
    try std.testing.expectEqual(@as(usize, 2), history.count());
    try std.testing.expectError(error.InvalidToken, history.observe(10));
    try std.testing.expect(!history.contains(10));
    history.reset();
    try std.testing.expectEqual(@as(usize, 0), history.count());
    try std.testing.expect(!history.contains(3));
    try std.testing.expectError(error.InvalidLogits, History.init(std.testing.allocator, 0));
}

test "penalties: hand-computed small vocabulary, neutral no-ops, greedy interaction" {
    const alloc = std.testing.allocator;
    var history = try History.init(alloc, 4);
    defer history.deinit();
    try history.observe(1);
    try history.observe(3);
    const logits = [_]f32{ 1.0, 2.0, 1.5, -1.0 };
    var scratch: [4]Candidate = undefined;

    // Greedy without history or with neutral penalties: the raw argmax (1).
    var greedy = try Sampler.init(0, .{});
    try std.testing.expectEqual(@as(u32, 1), try greedy.select(&logits, &.{}, null));
    try std.testing.expectEqual(@as(u32, 1), try greedy.select(&logits, &.{}, &history));

    // Presence 1.0 on {1, 3}: [1.0, 1.0, 1.5, -2.0] → argmax 2.
    var presence = try Sampler.init(0, .{ .presence_penalty = 1.0 });
    try std.testing.expectEqual(@as(u32, 2), try presence.select(&logits, &.{}, &history));
    // Presence 0.5 keeps token 1 on top: [1.0, 1.5, 1.5, -1.5] ties 1 and 2 → lowest id.
    var mild = try Sampler.init(0, .{ .presence_penalty = 0.5 });
    try std.testing.expectEqual(@as(u32, 1), try mild.select(&logits, &.{}, &history));
    // Negative presence rewards present tokens: with only token 3 in the
    // history it rises from -1 to 4 and wins.
    var only_three = try History.init(alloc, 4);
    defer only_three.deinit();
    try only_three.observe(3);
    var reward = try Sampler.init(0, .{ .presence_penalty = -5 });
    try std.testing.expectEqual(@as(u32, 3), try reward.select(&logits, &.{}, &only_three));

    // Repetition 2.0: positive logits divided (2.0 → 1.0), negative multiplied
    // (-1.0 → -2.0): [1.0, 1.0, 1.5, -2.0] → argmax 2.
    var repetition = try Sampler.init(0, .{ .repetition_penalty = 2.0 });
    try std.testing.expectEqual(@as(u32, 2), try repetition.select(&logits, &.{}, &history));
    // Order: repetition first, then presence. Repetition 4.0 and presence 1.0
    // on token 1 give 2/4 − 1 = −0.5 (presence first would give (2 − 1)/4 =
    // 0.25); token 3 gives −1·4 − 1 = −5. Checked on the penalized logits
    // themselves because the argmax (token 2) cannot tell the orders apart.
    var both = try Sampler.init(0, .{ .temperature = 1, .repetition_penalty = 4.0, .presence_penalty = 1.0 });
    var weights: [4]f32 = undefined;
    for (&weights, logits, 0..) |*w, raw, id| w.* = both.penalize(raw, id, &history);
    try std.testing.expectEqualSlices(f32, &.{ 1.0, -0.5, 1.5, -5.0 }, &weights);

    // Sampled with penalties: a token that is present and penalized to −∞-like
    // depth never appears; the seeded draws equal a run on pre-penalized logits.
    var penalized = try Sampler.init(7, .{ .temperature = 1, .presence_penalty = 30 });
    var reference = try Sampler.init(7, .{ .temperature = 1 });
    const manual = [_]f32{ 1.0, 2.0 - 30, 1.5, -1.0 - 30 };
    for (0..50) |_| {
        const got = try penalized.select(&logits, &scratch, &history);
        try std.testing.expectEqual(try reference.select(&manual, &scratch, null), got);
        try std.testing.expect(got == 0 or got == 2);
    }
    // No history means no penalty, whatever the options say.
    var lonely = try Sampler.init(0, .{ .presence_penalty = 1.0 });
    try std.testing.expectEqual(@as(u32, 1), try lonely.select(&logits, &.{}, null));
    // A penalty that overflows to infinity is reported, not sampled.
    var overflow = try Sampler.init(0, .{ .repetition_penalty = 1e-40 });
    var tiny = try History.init(alloc, 2);
    defer tiny.deinit();
    try tiny.observe(0);
    try std.testing.expectError(error.NonFiniteResult, overflow.select(&.{ 1e30, 0 }, &.{}, &tiny));
    // Penalties are never GPU-eligible and the readback path defers for them.
    try std.testing.expect(!presence.gpuEligible());
    var top: TopK = .{ .temperature = 1, .count = 1, .finite = true, .total = 1 };
    top.ids[0] = 1;
    top.values[0] = 2;
    try std.testing.expectEqual(@as(?u32, null), try presence.selectFrom(&top, &scratch));
}

test "min_p is a cutoff relative to the top candidate, applied after top-k and before top-p" {
    // Logits at temperature 1: probabilities ∝ e^3, e^2, e^1, e^0, e^-1 →
    // weights relative to the top: 1, e^-1 ≈ 0.368, e^-2 ≈ 0.135,
    // e^-3 ≈ 0.050, e^-4 ≈ 0.018.
    const logits = [_]f32{ 3, 2, 1, 0, -1 };
    var scratch: [5]Candidate = undefined;
    // min_p = 0 is a no-op: the draws equal a sampler without it.
    var with_zero = try Sampler.init(11, .{ .temperature = 1, .min_p = 0 });
    var without = try Sampler.init(11, .{ .temperature = 1 });
    for (0..40) |_| try std.testing.expectEqual(try without.select(&logits, &scratch, null), try with_zero.select(&logits, &scratch, null));
    // min_p = 0.1 keeps weights ≥ 0.1: tokens 0, 1, 2 only.
    var cut = try Sampler.init(11, .{ .temperature = 1, .min_p = 0.1 });
    var seen: u8 = 0;
    for (0..200) |_| seen |= @as(u8, 1) << @intCast(try cut.select(&logits, &scratch, null));
    try std.testing.expectEqual(@as(u8, 0b111), seen);
    // min_p = 1 keeps only the top candidate (weight exactly 1).
    var only_top = try Sampler.init(11, .{ .temperature = 1, .min_p = 1 });
    for (0..20) |_| try std.testing.expectEqual(@as(u32, 0), try only_top.select(&logits, &scratch, null));
    // The cutoff is relative to the top-1 probability, so temperature moves
    // it: at temperature 2 the weights are e^-0.5 ≈ 0.61, e^-1 ≈ 0.37,
    // e^-1.5 ≈ 0.22, e^-2 ≈ 0.14 and min_p 0.1 keeps all five.
    var warm = try Sampler.init(11, .{ .temperature = 2, .min_p = 0.1 });
    seen = 0;
    for (0..400) |_| seen |= @as(u8, 1) << @intCast(try warm.select(&logits, &scratch, null));
    try std.testing.expectEqual(@as(u8, 0b11111), seen);
    // Chain order. top-k 2 then min_p 0.5: survivors {0}; top-p 1 draws 0 always.
    var chained = try Sampler.init(11, .{ .temperature = 1, .top_k = 2, .min_p = 0.5 });
    for (0..20) |_| try std.testing.expectEqual(@as(u32, 0), try chained.select(&logits, &scratch, null));
    // top-p runs over the survivors, not the whole retained set. With min_p
    // 0.1 the survivors are tokens 0–2 (sum ≈ 1.503) and top_p 0.99 gives a
    // threshold ≈ 1.488 reached at token 2, so only tokens 0–2 are drawable.
    // Without min_p the sum over all five is ≈ 1.571, the threshold ≈ 1.555
    // needs token 4's 0.018, and every token is drawable.
    var narrow = try Sampler.init(5, .{ .temperature = 1, .min_p = 0.1, .top_p = 0.99 });
    var wide = try Sampler.init(5, .{ .temperature = 1, .top_p = 0.99 });
    var narrow_seen: u8 = 0;
    var wide_seen: u8 = 0;
    for (0..2000) |_| {
        narrow_seen |= @as(u8, 1) << @intCast(try narrow.select(&logits, &scratch, null));
        wide_seen |= @as(u8, 1) << @intCast(try wide.select(&logits, &scratch, null));
    }
    try std.testing.expectEqual(@as(u8, 0b00111), narrow_seen);
    try std.testing.expectEqual(@as(u8, 0b11111), wide_seen);
}

/// Builds the readback a correct GPU would produce: the reference sort's
/// first `capacity` entries and the F64 denominator, optionally perturbed by
/// a relative `error` to emulate the F32 reduction.
fn referenceTopK(logits: []const f32, scratch: []Candidate, temperature: f32, err: f64) TopK {
    for (scratch, logits, 0..) |*c, value, id| c.* = .{ .id = @intCast(id), .weight = value };
    std.mem.sort(Candidate, scratch, {}, Sampler.lessThan);
    var top: TopK = .{ .temperature = temperature, .count = @min(TopK.capacity, logits.len), .finite = true };
    for (scratch[0..top.count], 0..) |c, i| {
        top.ids[i] = c.id;
        top.values[i] = @floatCast(c.weight);
    }
    var total: f64 = 0;
    for (scratch) |c| total += @exp((c.weight - scratch[0].weight) / temperature);
    top.total = total * (1 + err);
    return top;
}

test "GPU top-k readback reproduces the reference sampler or defers" {
    // Vocabulary 1,024 with a 256-entry readback: the nucleus regularly
    // exceeds the readback at high temperature, exercising the deferral.
    // (`metal-check` runs the real kernels on the 248,320 vocabulary.)
    const alloc = std.testing.allocator;
    const vocabulary = 1024;
    const logits = try alloc.alloc(f32, vocabulary);
    defer alloc.free(logits);
    const scratch = try alloc.alloc(Candidate, vocabulary);
    defer alloc.free(scratch);
    const scratch_top = try alloc.alloc(Candidate, vocabulary);
    defer alloc.free(scratch_top);
    var prng = std.Random.DefaultPrng.init(2026);
    const random = prng.random();
    const top_ks = [_]usize{ 0, 1, 40, 256, 300 };
    const top_ps = [_]f32{ 0.5, 0.95, 1 };
    const min_ps = [_]f32{ 0, 0.05 };
    const temperatures = [_]f32{ 0.3, 0.7, 1.5 };
    const errors = [_]f64{ 0, 4e-6, -4e-6 };
    var fallbacks: usize = 0;
    var decided: usize = 0;
    var min_p_decided: usize = 0;
    var exact_path_fallbacks: usize = 0;
    for (0..20) |vector| {
        // Peakedness varies per vector so nucleus sizes range from one token to hundreds.
        const scale: f32 = 0.5 + 7.5 * @as(f32, @floatFromInt(vector % 8)) / 7;
        for (logits) |*l| l.* = random.floatNorm(f32) * scale;
        if (vector % 5 == 0) logits[7] = logits[3]; // a tie among the leaders
        for (top_ks) |top_k| for (top_ps) |top_p| for (min_ps) |min_p| for (temperatures) |temperature| for (errors) |err| {
            const top = referenceTopK(logits, scratch, temperature, err);
            for (0..3) |seed| {
                const options: Options = .{ .temperature = temperature, .top_k = top_k, .top_p = top_p, .min_p = min_p };
                var reference = try Sampler.init(seed, options);
                var gpu = try Sampler.init(seed, options);
                const expected = try reference.select(logits, scratch, null);
                const eligible = gpu.gpuEligible();
                const token = (try gpu.selectFrom(&top, scratch_top)) orelse blk: {
                    fallbacks += 1;
                    if (eligible and top_k >= 1) exact_path_fallbacks += 1;
                    break :blk try gpu.select(logits, scratch, null);
                };
                if (eligible and (try gpu.selectFrom(&top, scratch_top)) != null) {
                    decided += 1;
                    if (top_k == 0 and top_p == 1 and min_p > 0) min_p_decided += 1;
                }
                try std.testing.expectEqual(expected, token);
                // Ineligible options must always defer.
                if (!eligible) try std.testing.expectEqual(@as(?u32, null), try gpu.selectFrom(&top, scratch_top));
            }
        };
    }
    try std.testing.expect(decided > 0);
    try std.testing.expect(fallbacks > 0);
    // `top_k = 0, top_p = 1` is decidable only through the min_p prefix.
    try std.testing.expect(min_p_decided > 0);
    // 1 ≤ top_k ≤ capacity never defers: the retained set is in the readback.
    try std.testing.expectEqual(@as(usize, 0), exact_path_fallbacks);
}

test "readback validation and the uncertainty band" {
    var scratch: [TopK.capacity]Candidate = undefined;
    var sampler = try Sampler.init(1, .{ .temperature = 1, .top_p = 0.5 });
    var top: TopK = .{ .temperature = 1, .count = 2, .finite = true, .total = 2 };
    top.ids[0] = 5;
    top.values[0] = 0;
    top.ids[1] = 9;
    top.values[1] = 0;
    // retained after the first candidate is exactly top_p · total: inside the band → defer.
    try std.testing.expectEqual(@as(?u32, null), try sampler.selectFrom(&top, &scratch));
    top.total = 1.9; // threshold 0.95 < 1 − band: the first candidate decides
    try std.testing.expectEqual(@as(?u32, 5), try sampler.selectFrom(&top, &scratch));
    top.finite = false;
    try std.testing.expectEqual(@as(?u32, null), try sampler.selectFrom(&top, &scratch));
    top.finite = true;
    top.temperature = 2;
    try std.testing.expectError(error.InvalidSamplingOptions, sampler.selectFrom(&top, &scratch));
    top.temperature = 1;
    top.values[1] = 1; // out of reference order
    try std.testing.expectError(error.InvalidLogits, sampler.selectFrom(&top, &scratch));
    top.count = 0;
    try std.testing.expectError(error.InvalidLogits, sampler.selectFrom(&top, &scratch));
    var greedy = try Sampler.init(1, .{});
    top.count = 2;
    top.values[1] = 0;
    try std.testing.expectEqual(@as(?u32, 5), try greedy.selectFrom(&top, &scratch));
    try std.testing.expect(!greedy.gpuEligible());
    try std.testing.expect((try Sampler.init(0, .{ .temperature = 1, .top_k = 40 })).gpuEligible());
    try std.testing.expect(!(try Sampler.init(0, .{ .temperature = 1, .top_k = 300 })).gpuEligible());
    try std.testing.expect(!(try Sampler.init(0, .{ .temperature = 1 })).gpuEligible());
    // min_p alone makes the full distribution eligible; a penalty removes eligibility again.
    try std.testing.expect((try Sampler.init(0, .{ .temperature = 1, .min_p = 0.05 })).gpuEligible());
    try std.testing.expect(!(try Sampler.init(0, .{ .temperature = 1, .top_k = 20, .presence_penalty = 1.5 })).gpuEligible());
    try std.testing.expect(!(try Sampler.init(0, .{ .temperature = 1, .top_k = 20, .repetition_penalty = 1.1 })).gpuEligible());
    // The min_p prefix defers when the whole readback survives (all weights ≥ min_p).
    var prefix = try Sampler.init(1, .{ .temperature = 1, .min_p = 0.5 });
    try std.testing.expectEqual(@as(?u32, null), try prefix.selectFrom(&top, &scratch)); // both weights are 1
    top.values[1] = -10; // second weight e^-10 < 0.5: the prefix is {5}, decided exactly
    try std.testing.expectEqual(@as(?u32, 5), try prefix.selectFrom(&top, &scratch));
}
