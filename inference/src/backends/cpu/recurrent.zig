//! Single-step recurrent references. Callers own and initialize state (usually
//! zero at session start), sequence order, head mapping, and checkpoint copies.
//! No allocation or I/O, projections, activation, normalization, or implicit reset.
const std = @import("std");
pub const Error = error{ InvalidShape, Overflow, InvalidGate, NonFiniteInput, NonFiniteResult, ScratchTooSmall };

fn finite(values: []const f32) Error!void {
    for (values) |v| if (!std.math.isFinite(v)) return error.NonFiniteInput;
}
fn representable(value: f64) Error!void {
    if (!std.math.isFinite(value) or @abs(value) > std.math.floatMax(f32)) return error.NonFiniteResult;
}

/// Depthwise causal convolution. Weights: [channel][tap], oldest tap first,
/// current input last. History: [channel][kernel-1], oldest sample first.
/// All buffers must be disjoint. On success, append input to each history row;
/// on any error, history and output are unchanged. Kernel 1 needs empty history.
/// No bias or SiLU is included; adapters compose those explicitly.
pub fn convolution(input: []const f32, weights: []const f32, history: []f32, output: []f32, kernel: usize) Error!void {
    if (input.len == 0 or kernel == 0 or output.len != input.len) return error.InvalidShape;
    if (weights.len != try std.math.mul(usize, input.len, kernel) or
        history.len != try std.math.mul(usize, input.len, kernel - 1)) return error.InvalidShape;
    try finite(input);
    try finite(weights);
    try finite(history);
    // Two passes avoid scratch allocation while ensuring overflow is detected
    // before any output/state mutation. Each dot product accumulates in F64.
    for (0..2) |pass| {
        for (input, 0..) |x, c| {
            const w = weights[c * kernel ..][0..kernel];
            const h = history[c * (kernel - 1) ..][0 .. kernel - 1];
            var sum: f64 = 0;
            for (h, w[0 .. kernel - 1]) |a, b| sum += @as(f64, a) * b;
            sum += @as(f64, x) * w[kernel - 1];
            if (pass == 0) try representable(sum) else output[c] = @floatCast(sum);
        }
    }
    if (kernel > 1) for (input, 0..) |x, c| {
        const h = history[c * (kernel - 1) ..][0 .. kernel - 1];
        std.mem.copyForwards(f32, h[0 .. h.len - 1], h[1..]);
        h[h.len - 1] = x;
    };
}

pub const Delta = struct {
    query: []const f32,
    key: []const f32,
    value: []const f32,
    /// Finite nonpositive log decay; actual decay is exp(log_decay).
    log_decay: f32,
    /// Update strength, already passed through the model's sigmoid: [0,1].
    beta: f32,
    /// Explicit query scaling, normally 1/sqrt(key width).
    scale: f32,
};

/// One scalar-gated DeltaNet head, with state in [value_channel][key_channel]
/// order. Q/K normalization and gate preparation belong to the caller.
/// D = exp(log_decay)*state; correction[j] = beta*(value[j] - dot(D[j], key));
/// next[j,i] = D[j,i] + correction[j]*key[i]; output[j] = scale*dot(next[j], query).
/// State is F32 between steps; calculations within a step use F64.
///
/// next_state may exactly alias state; all other overlap is forbidden. Scratch
/// needs state.len + value.len F64 elements. Validation errors touch nothing;
/// numerical overflow may change scratch but never output or next_state.
pub fn delta(x: Delta, state: []const f32, next_state: []f32, output: []f32, scratch: []f64) Error!void {
    if (x.key.len == 0 or x.value.len == 0 or x.query.len != x.key.len) return error.InvalidShape;
    const count = try std.math.mul(usize, x.key.len, x.value.len);
    if (state.len != count or next_state.len != count or output.len != x.value.len) return error.InvalidShape;
    const needed = try std.math.add(usize, count, x.value.len);
    if (scratch.len < needed) return error.ScratchTooSmall;
    if (!std.math.isFinite(x.log_decay) or x.log_decay > 0 or !std.math.isFinite(x.beta) or
        x.beta < 0 or x.beta > 1 or !std.math.isFinite(x.scale) or x.scale <= 0) return error.InvalidGate;
    try finite(x.query);
    try finite(x.key);
    try finite(x.value);
    try finite(state);
    const decay = @exp(@as(f64, x.log_decay));
    for (x.value, 0..) |v, j| {
        const old = state[j * x.key.len ..][0..x.key.len];
        const next = scratch[j * x.key.len ..][0..x.key.len];
        var prediction: f64 = 0;
        for (old, x.key) |s, k| prediction += decay * @as(f64, s) * k;
        const correction = @as(f64, x.beta) * (@as(f64, v) - prediction);
        var result: f64 = 0;
        for (old, x.key, x.query, next) |s, k, q, *n| {
            n.* = decay * @as(f64, s) + correction * k;
            try representable(n.*);
            result += n.* * q;
        }
        scratch[count + j] = result * x.scale;
        try representable(scratch[count + j]);
    }
    for (next_state, scratch[0..count]) |*s, n| s.* = @floatCast(n);
    for (output, scratch[count..needed]) |*o, n| o.* = @floatCast(n);
}

/// A chunk of `tokens` consecutive DeltaNet steps for one head, given to
/// `deltaChunk` as `[token][channel]` rows. Same per-step semantics as
/// `Delta` (log decay, sigmoid-ed beta, explicit query scale per token).
pub const DeltaChunk = struct {
    tokens: usize,
    key_width: usize,
    value_width: usize,
    /// [token][key_width]
    queries: []const f32,
    /// [token][key_width]
    keys: []const f32,
    /// [token][value_width]
    values: []const f32,
    /// [token]; each finite and nonpositive
    log_decays: []const f32,
    /// [token]; each in [0, 1]
    betas: []const f32,
    scale: f32,
};

/// F64 scratch `deltaChunk` needs for `x`: 2C² + 3CV + VK + C elements.
pub fn deltaChunkScratch(x: DeltaChunk) Error!usize {
    const c = x.tokens;
    const cc = try std.math.mul(usize, c, c);
    const cv = try std.math.mul(usize, c, x.value_width);
    const vk = try std.math.mul(usize, x.value_width, x.key_width);
    var total = try std.math.add(usize, try std.math.mul(usize, cc, 2), try std.math.mul(usize, cv, 3));
    total = try std.math.add(usize, total, vk);
    return try std.math.add(usize, total, c);
}

/// The chunkwise form of `tokens` sequential `delta` steps: same
/// outputs and final state, computed without stepping the matrix per
/// token. Per token the sequential update is
///
///   S_t = a_t S_{t-1} + u_t k_tᵀ,   u_t = β_t (v_t − a_t S_{t-1} k_t),
///   o_t = scale · S_t q_t,
///
/// so with γ_t = Π_{r≤t} a_r and r(t,s) = γ_t/γ_s the state unrolls to
/// S_t = γ_t S_0 + Σ_{s≤t} r(t,s) u_s k_sᵀ, and substituting into u_t gives
/// the triangular system (I + A) U = B with
///
///   A_ts = β_t r(t,s) (k_s·k_t) for s < t (strictly lower),
///   B_t  = β_t (v_t − γ_t S_0 k_t),
///
/// solved by forward substitution (the WY form: each u_t depends on the
/// earlier u_s through inner products of keys, never on the matrix). Then
/// o_t = scale (γ_t S_0 q_t + Σ_{s≤t} r(t,s) (k_s·q_t) u_s) and the carry
/// S_C = γ_C S_0 + Σ_s r(C,s) u_s k_sᵀ. Decay ratios are exponentials of
/// differences of cumulative log decays, so long chunks never divide by an
/// underflowed product. Everything inside the chunk is F64; `state` and
/// `next_state` are F32 like the session, and `next_state` may exactly
/// alias `state`. `output` is [token][value_width]. Validation errors and
/// overflow leave `next_state` and `output` untouched. This is the CPU
/// oracle for the `nu_delta_chunk` kernel; the sequential `delta` remains
/// the oracle for this function (see the tests).
pub fn deltaChunk(x: DeltaChunk, state: []const f32, next_state: []f32, output: []f32, scratch: []f64) Error!void {
    const c = x.tokens;
    const kw = x.key_width;
    const vw = x.value_width;
    if (c == 0 or kw == 0 or vw == 0) return error.InvalidShape;
    const count = try std.math.mul(usize, kw, vw);
    if (x.queries.len != try std.math.mul(usize, c, kw) or x.keys.len != x.queries.len) return error.InvalidShape;
    if (x.values.len != try std.math.mul(usize, c, vw) or x.log_decays.len != c or x.betas.len != c) return error.InvalidShape;
    if (state.len != count or next_state.len != count or output.len != x.values.len) return error.InvalidShape;
    if (scratch.len < try deltaChunkScratch(x)) return error.ScratchTooSmall;
    if (!std.math.isFinite(x.scale) or x.scale <= 0) return error.InvalidGate;
    for (x.log_decays, x.betas) |l, b| if (!std.math.isFinite(l) or l > 0 or !std.math.isFinite(b) or b < 0 or b > 1) return error.InvalidGate;
    try finite(x.queries);
    try finite(x.keys);
    try finite(x.values);
    try finite(state);

    // Scratch layout.
    var at: usize = 0;
    const cum = scratch[at..][0..c]; // cumulative log decay L_t
    at += c;
    const a = scratch[at..][0 .. c * c]; // A_ts (strictly lower), row-major [t][s]
    at += c * c;
    const kq = scratch[at..][0 .. c * c]; // r(t,s)·(k_s·q_t) for s ≤ t
    at += c * c;
    const b = scratch[at..][0 .. c * vw]; // B_t, then u_t in place
    at += c * vw;
    const p = scratch[at..][0 .. c * vw]; // γ_t S_0 q_t
    at += c * vw;
    const o = scratch[at..][0 .. c * vw]; // outputs before the F32 cast
    at += c * vw;
    const next = scratch[at..][0..count]; // carry before the F32 cast

    var running: f64 = 0;
    for (cum, x.log_decays) |*l, d| {
        running += @as(f64, d);
        l.* = running;
    }
    // B_t and P_t need S_0 k_t and S_0 q_t per value row.
    for (0..c) |t| {
        const gamma = @exp(cum[t]);
        const k = x.keys[t * kw ..][0..kw];
        const q = x.queries[t * kw ..][0..kw];
        const v = x.values[t * vw ..][0..vw];
        for (0..vw) |j| {
            const row = state[j * kw ..][0..kw];
            var sk: f64 = 0;
            var sq: f64 = 0;
            for (row, k, q) |s0, ki, qi| {
                sk += @as(f64, s0) * ki;
                sq += @as(f64, s0) * qi;
            }
            b[t * vw + j] = @as(f64, x.betas[t]) * (@as(f64, v[j]) - gamma * sk);
            p[t * vw + j] = gamma * sq;
        }
    }
    // Key/key and key/query inner products with their decay ratios.
    for (0..c) |t| {
        const kt = x.keys[t * kw ..][0..kw];
        const qt = x.queries[t * kw ..][0..kw];
        for (0..t + 1) |sidx| {
            const ks = x.keys[sidx * kw ..][0..kw];
            var kk: f64 = 0;
            var kqd: f64 = 0;
            for (ks, kt, qt) |ksi, kti, qti| {
                kk += @as(f64, ksi) * kti;
                kqd += @as(f64, ksi) * qti;
            }
            const ratio = @exp(cum[t] - cum[sidx]);
            kq[t * c + sidx] = ratio * kqd;
            if (sidx < t) a[t * c + sidx] = @as(f64, x.betas[t]) * ratio * kk;
        }
    }
    // Forward substitution: u_t = B_t − Σ_{s<t} A_ts u_s, in place over b.
    for (0..c) |t| {
        for (0..t) |sidx| {
            const coeff = a[t * c + sidx];
            for (b[t * vw ..][0..vw], b[sidx * vw ..][0..vw]) |*ut, us| ut.* -= coeff * us;
        }
    }
    // Outputs: o_t = scale (P_t + Σ_{s≤t} kq[t][s] u_s).
    for (0..c) |t| {
        for (0..vw) |j| {
            var sum: f64 = p[t * vw + j];
            for (0..t + 1) |sidx| sum += kq[t * c + sidx] * b[sidx * vw + j];
            o[t * vw + j] = sum * x.scale;
            try representable(o[t * vw + j]);
        }
    }
    // Carry: S_C = γ_C S_0 + Σ_s r(C,s) u_s k_sᵀ.
    const last = c - 1;
    const gamma_c = @exp(cum[last]);
    for (0..vw) |j| {
        const row = state[j * kw ..][0..kw];
        for (next[j * kw ..][0..kw], row) |*n, s0| n.* = gamma_c * @as(f64, s0);
        for (0..c) |sidx| {
            const w = @exp(cum[last] - cum[sidx]) * b[sidx * vw + j];
            for (next[j * kw ..][0..kw], x.keys[sidx * kw ..][0..kw]) |*n, ki| n.* += w * ki;
        }
        for (next[j * kw ..][0..kw]) |n| try representable(n);
    }
    for (next_state, next) |*s, n| s.* = @floatCast(n);
    for (output, o) |*out, n| out.* = @floatCast(n);
}

test "causal convolution history order, independent channels, and kernel one" {
    var history = [_]f32{ 1, 2, 10, 20 };
    var out: [2]f32 = undefined;
    try convolution(&.{ 3, 30 }, &.{ 1, 2, 4, -1, 0, 2 }, &history, &out, 3);
    try std.testing.expectEqualSlices(f32, &.{ 17, 50 }, &out);
    try std.testing.expectEqualSlices(f32, &.{ 2, 3, 20, 30 }, &history);
    try convolution(&.{ 4, 40 }, &.{ 1, 2, 4, -1, 0, 2 }, &history, &out, 3);
    try std.testing.expectEqualSlices(f32, &.{ 24, 60 }, &out);
    try convolution(&.{ 3, 4 }, &.{ 2, -1 }, history[0..0], &out, 1);
    try std.testing.expectEqualSlices(f32, &.{ 6, -4 }, &out);
}

test "DeltaNet corrects decayed prediction and reads updated state" {
    var state = [_]f32{ 1, 2, 3, 4 };
    var out: [2]f32 = undefined;
    var scratch = [_]f64{999} ** 7;
    const x: Delta = .{ .query = &.{ 0, 1 }, .key = &.{ 1, 0 }, .value = &.{ 5, 7 }, .log_decay = 0, .beta = 0.5, .scale = 2 };
    try delta(x, &state, &state, &out, &scratch);
    try std.testing.expectEqualSlices(f32, &.{ 3, 2, 5, 4 }, &state);
    try std.testing.expectEqualSlices(f32, &.{ 4, 8 }, &out);
    try std.testing.expectEqual(@as(f64, 999), scratch[6]);
    var forget = x;
    forget.beta = 0;
    forget.log_decay = -1000;
    try delta(forget, &state, &state, &out, &scratch);
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 0, 0 }, &state);
}

test "restoring a recurrent checkpoint reproduces continuation" {
    var state = [_]f32{0} ** 4;
    var out: [2]f32 = undefined;
    var scratch: [6]f64 = undefined;
    const x: Delta = .{ .query = &.{ 0.5, -0.5 }, .key = &.{ 0.6, 0.8 }, .value = &.{ 2, -1 }, .log_decay = -0.2, .beta = 0.7, .scale = 1 };
    try delta(x, &state, &state, &out, &scratch);
    const checkpoint = state;
    try delta(x, &state, &state, &out, &scratch);
    const expected_state = state;
    const expected_out = out;
    state = checkpoint;
    try delta(x, &state, &state, &out, &scratch);
    try std.testing.expectEqualSlices(f32, &expected_state, &state);
    try std.testing.expectEqualSlices(f32, &expected_out, &out);
    try std.testing.expect(!std.mem.eql(f32, &checkpoint, &state));
}

test "recurrent steps match pinned CPU outputs and state" {
    const DeltaCase = struct { query: []const f32, key: []const f32, value: []const f32, log_decay: f32, beta: f32, scale: f32, state: []const f32, output: []const f32, next_state: []const f32 };
    const ConvCase = struct { input: []const f32, weights: []const f32, history: []const f32, kernel: usize, output: []const f32, next_history: []const f32 };
    const Fixture = struct { revision: []const u8, delta: []const DeltaCase, convolution: []const ConvCase };
    const alloc = std.testing.allocator;
    const fixture = try std.json.parseFromSlice(Fixture, alloc, @embedFile("fixtures/recurrent.json"), .{});
    defer fixture.deinit();
    try std.testing.expectEqualStrings("7620399f58aebfd2196b74021f9581bcf7218cb9", fixture.value.revision);
    // Carry OUR state through the entire sequence, not the reference's state
    // into every step: this catches accumulated orientation/update mistakes.
    const state = try alloc.dupe(f32, fixture.value.delta[0].state);
    defer alloc.free(state);
    const out = try alloc.alloc(f32, fixture.value.delta[0].value.len);
    defer alloc.free(out);
    const scratch = try alloc.alloc(f64, state.len + out.len);
    defer alloc.free(scratch);
    for (fixture.value.delta) |case| {
        try delta(.{ .query = case.query, .key = case.key, .value = case.value, .log_decay = case.log_decay, .beta = case.beta, .scale = case.scale }, state, state, out, scratch);
        for (out, case.output) |got, want| try std.testing.expectApproxEqAbs(want, got, 1e-6);
        for (state, case.next_state) |got, want| try std.testing.expectApproxEqAbs(want, got, 1e-6);
    }
    const history = try alloc.dupe(f32, fixture.value.convolution[0].history);
    defer alloc.free(history);
    for (fixture.value.convolution) |case| {
        try convolution(case.input, case.weights, history, out, case.kernel);
        try std.testing.expectEqualSlices(f32, case.output, out);
        try std.testing.expectEqualSlices(f32, case.next_history, history);
    }
}

test "DeltaNet supports rectangular state and beta one replaces a keyed column" {
    const old = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var next: [6]f32 = undefined;
    var out: [3]f32 = undefined;
    var scratch: [9]f64 = undefined;
    try delta(.{ .query = &.{ 1, 0 }, .key = &.{ 1, 0 }, .value = &.{ 7, 8, 9 }, .log_decay = 0, .beta = 1, .scale = 1 }, &old, &next, &out, &scratch);
    try std.testing.expectEqualSlices(f32, &.{ 7, 2, 8, 4, 9, 6 }, &next);
    try std.testing.expectEqualSlices(f32, &.{ 7, 8, 9 }, &out);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5, 6 }, &old);
}

test "recurrent failures preserve durable state and output" {
    var state = [_]f32{1};
    var out = [_]f32{123};
    var scratch: [2]f64 = undefined;
    try std.testing.expectError(error.InvalidShape, convolution(&.{1}, &.{1}, &state, &out, 1));
    try std.testing.expectError(error.NonFiniteInput, convolution(&.{std.math.nan(f32)}, &.{ 1, 1 }, &state, &out, 2));
    try std.testing.expectError(error.NonFiniteResult, convolution(&.{std.math.floatMax(f32)}, &.{ 1, 2 }, &state, &out, 2));
    var x: Delta = .{ .query = &.{1}, .key = &.{1}, .value = &.{1}, .log_decay = 0, .beta = 1, .scale = 1 };
    try std.testing.expectError(error.InvalidShape, delta(x, &state, &state, out[0..0], &scratch));
    try std.testing.expectError(error.ScratchTooSmall, delta(x, &state, &state, &out, scratch[0..1]));
    for ([_]f32{ 0.1, std.math.nan(f32), -std.math.inf(f32) }) |bad| {
        x.log_decay = bad;
        try std.testing.expectError(error.InvalidGate, delta(x, &state, &state, &out, &scratch));
    }
    x.log_decay = 0;
    for ([_]f32{ 0, -1, std.math.inf(f32) }) |bad| {
        x.scale = bad;
        try std.testing.expectError(error.InvalidGate, delta(x, &state, &state, &out, &scratch));
    }
    x.scale = 1;
    x.beta = -1;
    try std.testing.expectError(error.InvalidGate, delta(x, &state, &state, &out, &scratch));
    x.beta = 1;
    x.query = &.{std.math.inf(f32)};
    try std.testing.expectError(error.NonFiniteInput, delta(x, &state, &state, &out, &scratch));
    x.query = &.{std.math.floatMax(f32)};
    x.value = &.{2};
    try std.testing.expectError(error.NonFiniteResult, delta(x, &state, &state, &out, &scratch));
    try std.testing.expectEqualSlices(f32, &.{1}, &state);
    try std.testing.expectEqualSlices(f32, &.{123}, &out);
}

test "late recurrent overflow is transactional across channels" {
    var history = [_]f32{ 1, 2 };
    var out = [_]f32{ 123, 456 };
    try std.testing.expectError(error.NonFiniteResult, convolution(&.{ 1, std.math.floatMax(f32) }, &.{ 1, 1, 1, 2 }, &history, &out, 2));
    try std.testing.expectEqualSlices(f32, &.{ 1, 2 }, &history);
    var next = [_]f32{ 7, 8 };
    var scratch: [4]f64 = undefined;
    const x: Delta = .{ .query = &.{std.math.floatMax(f32)}, .key = &.{1}, .value = &.{ 0, 2 }, .log_decay = 0, .beta = 1, .scale = 1 };
    try std.testing.expectError(error.NonFiniteResult, delta(x, &history, &next, &out, &scratch));
    try std.testing.expectEqualSlices(f32, &.{ 7, 8 }, &next);
    try std.testing.expectEqualSlices(f32, &.{ 123, 456 }, &out);
}

/// Test oracle for `deltaChunk`: the sequential update with F64 state
/// carried between steps, so the chunkwise algebra can be checked to F64
/// precision (the production `delta` rounds state to F32 per step).
fn sequentialF64(x: DeltaChunk, state: []f64, output: []f64) void {
    const kw = x.key_width;
    const vw = x.value_width;
    for (0..x.tokens) |t| {
        const decay = @exp(@as(f64, x.log_decays[t]));
        const k = x.keys[t * kw ..][0..kw];
        const q = x.queries[t * kw ..][0..kw];
        for (0..vw) |j| {
            const row = state[j * kw ..][0..kw];
            var prediction: f64 = 0;
            for (row, k) |s, ki| prediction += decay * s * ki;
            const correction = @as(f64, x.betas[t]) * (@as(f64, x.values[t * vw + j]) - prediction);
            var result: f64 = 0;
            for (row, k, q) |*s, ki, qi| {
                s.* = decay * s.* + correction * ki;
                result += s.* * qi;
            }
            output[t * vw + j] = result * x.scale;
        }
    }
}

fn randomChunk(alloc: std.mem.Allocator, random: std.Random, tokens: usize, kw: usize, vw: usize) !DeltaChunk {
    const queries = try alloc.alloc(f32, tokens * kw);
    const keys = try alloc.alloc(f32, tokens * kw);
    const values = try alloc.alloc(f32, tokens * vw);
    const log_decays = try alloc.alloc(f32, tokens);
    const betas = try alloc.alloc(f32, tokens);
    for (queries) |*v| v.* = random.floatNorm(f32) * 0.3;
    for (keys) |*v| v.* = random.floatNorm(f32) * 0.3;
    for (values) |*v| v.* = random.floatNorm(f32);
    for (log_decays) |*v| v.* = -random.float(f32) * 0.5;
    for (betas) |*v| v.* = random.float(f32);
    return .{ .tokens = tokens, .key_width = kw, .value_width = vw, .queries = queries, .keys = keys, .values = values, .log_decays = log_decays, .betas = betas, .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(kw))) };
}
fn freeChunk(alloc: std.mem.Allocator, x: DeltaChunk) void {
    alloc.free(x.queries);
    alloc.free(x.keys);
    alloc.free(x.values);
    alloc.free(x.log_decays);
    alloc.free(x.betas);
}

test "chunkwise DeltaNet equals sequential F64 steps on a model-shaped chunk" {
    const alloc = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xe22);
    const random = prng.random();
    const x = try randomChunk(alloc, random, 64, 128, 128);
    defer freeChunk(alloc, x);
    const state = try alloc.alloc(f32, 128 * 128);
    defer alloc.free(state);
    for (state) |*v| v.* = random.floatNorm(f32) * 0.1;
    const state64 = try alloc.alloc(f64, state.len);
    defer alloc.free(state64);
    for (state64, state) |*d, s| d.* = s;
    const expected = try alloc.alloc(f64, 64 * 128);
    defer alloc.free(expected);
    sequentialF64(x, state64, expected);
    const next = try alloc.alloc(f32, state.len);
    defer alloc.free(next);
    const out = try alloc.alloc(f32, 64 * 128);
    defer alloc.free(out);
    const scratch = try alloc.alloc(f64, try deltaChunkScratch(x));
    defer alloc.free(scratch);
    try deltaChunk(x, state, next, out, scratch);
    // The chunk keeps F64 throughout, so the only difference from the F64
    // sequence is the final F32 cast: cast the expectation the same way and
    // require equality within 1e-9 (measured 2026-09-09: 0, bit-identical
    // after the cast).
    var worst: f64 = 0;
    for (out, expected) |got, want| worst = @max(worst, @abs(@as(f64, got) - @as(f64, @as(f32, @floatCast(want)))));
    for (next, state64) |got, want| worst = @max(worst, @abs(@as(f64, got) - @as(f64, @as(f32, @floatCast(want)))));
    try std.testing.expect(worst <= 1e-9);
}

test "chunkwise DeltaNet matches the production sequential steps and is split-invariant" {
    const alloc = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x22e);
    const random = prng.random();
    const x = try randomChunk(alloc, random, 64, 128, 128);
    defer freeChunk(alloc, x);
    const start = try alloc.alloc(f32, 128 * 128);
    defer alloc.free(start);
    for (start) |*v| v.* = random.floatNorm(f32) * 0.1;
    // Sequential `delta`, F32 state between steps.
    const seq_state = try alloc.dupe(f32, start);
    defer alloc.free(seq_state);
    const seq_out = try alloc.alloc(f32, 64 * 128);
    defer alloc.free(seq_out);
    const step_scratch = try alloc.alloc(f64, 128 * 128 + 128);
    defer alloc.free(step_scratch);
    for (0..64) |t| {
        try delta(.{ .query = x.queries[t * 128 ..][0..128], .key = x.keys[t * 128 ..][0..128], .value = x.values[t * 128 ..][0..128], .log_decay = x.log_decays[t], .beta = x.betas[t], .scale = x.scale }, seq_state, seq_state, seq_out[t * 128 ..][0..128], step_scratch);
    }
    // One chunk of 64.
    const scratch = try alloc.alloc(f64, try deltaChunkScratch(x));
    defer alloc.free(scratch);
    const one_state = try alloc.dupe(f32, start);
    defer alloc.free(one_state);
    const one_out = try alloc.alloc(f32, 64 * 128);
    defer alloc.free(one_out);
    try deltaChunk(x, one_state, one_state, one_out, scratch);
    // Two chunks of 40 + 24 with the F32 state carried between them.
    const two_state = try alloc.dupe(f32, start);
    defer alloc.free(two_state);
    const two_out = try alloc.alloc(f32, 64 * 128);
    defer alloc.free(two_out);
    const first: DeltaChunk = .{ .tokens = 40, .key_width = 128, .value_width = 128, .queries = x.queries[0 .. 40 * 128], .keys = x.keys[0 .. 40 * 128], .values = x.values[0 .. 40 * 128], .log_decays = x.log_decays[0..40], .betas = x.betas[0..40], .scale = x.scale };
    const second: DeltaChunk = .{ .tokens = 24, .key_width = 128, .value_width = 128, .queries = x.queries[40 * 128 ..], .keys = x.keys[40 * 128 ..], .values = x.values[40 * 128 ..], .log_decays = x.log_decays[40..], .betas = x.betas[40..], .scale = x.scale };
    try deltaChunk(first, two_state, two_state, two_out[0 .. 40 * 128], scratch);
    try deltaChunk(second, two_state, two_state, two_out[40 * 128 ..], scratch);
    var worst: f32 = 0;
    for (seq_out, one_out, two_out) |s, o, w| worst = @max(worst, @max(@abs(s - o), @abs(s - w)));
    for (seq_state, one_state, two_state) |s, o, w| worst = @max(worst, @max(@abs(s - o), @abs(s - w)));
    // The sequential path rounds its state to F32 sixty-four times; the
    // chunk does not. Measured 2026-09-09: 2.4e-7; bound 1e-4.
    try std.testing.expect(worst <= 1e-4);
}

test "chunkwise DeltaNet reproduces the pinned sequential fixture as one chunk" {
    const DeltaCase = struct { query: []const f32, key: []const f32, value: []const f32, log_decay: f32, beta: f32, scale: f32, state: []const f32, output: []const f32, next_state: []const f32 };
    const Fixture = struct { revision: []const u8, delta: []const DeltaCase };
    const alloc = std.testing.allocator;
    const fixture = try std.json.parseFromSlice(Fixture, alloc, @embedFile("fixtures/recurrent.json"), .{ .ignore_unknown_fields = true });
    defer fixture.deinit();
    const cases = fixture.value.delta;
    const kw = cases[0].key.len;
    const vw = cases[0].value.len;
    var queries: std.ArrayList(f32) = .empty;
    defer queries.deinit(alloc);
    var keys: std.ArrayList(f32) = .empty;
    defer keys.deinit(alloc);
    var values: std.ArrayList(f32) = .empty;
    defer values.deinit(alloc);
    var decays: std.ArrayList(f32) = .empty;
    defer decays.deinit(alloc);
    var betas: std.ArrayList(f32) = .empty;
    defer betas.deinit(alloc);
    for (cases) |case| {
        try queries.appendSlice(alloc, case.query);
        try keys.appendSlice(alloc, case.key);
        try values.appendSlice(alloc, case.value);
        try decays.append(alloc, case.log_decay);
        try betas.append(alloc, case.beta);
        try std.testing.expectEqual(cases[0].scale, case.scale);
    }
    const x: DeltaChunk = .{ .tokens = cases.len, .key_width = kw, .value_width = vw, .queries = queries.items, .keys = keys.items, .values = values.items, .log_decays = decays.items, .betas = betas.items, .scale = cases[0].scale };
    const state = try alloc.dupe(f32, cases[0].state);
    defer alloc.free(state);
    const out = try alloc.alloc(f32, cases.len * vw);
    defer alloc.free(out);
    const scratch = try alloc.alloc(f64, try deltaChunkScratch(x));
    defer alloc.free(scratch);
    try deltaChunk(x, state, state, out, scratch);
    for (cases, 0..) |case, t| for (out[t * vw ..][0..vw], case.output) |got, want| try std.testing.expectApproxEqAbs(want, got, 1e-6);
    for (state, cases[cases.len - 1].next_state) |got, want| try std.testing.expectApproxEqAbs(want, got, 1e-6);
}

test "chunkwise DeltaNet validates before writing and keeps a single token exact" {
    var state = [_]f32{ 1, 2, 3, 4 };
    var out = [_]f32{ 9, 9 };
    var scratch: [64]f64 = undefined;
    const good: DeltaChunk = .{ .tokens = 1, .key_width = 2, .value_width = 2, .queries = &.{ 0, 1 }, .keys = &.{ 1, 0 }, .values = &.{ 5, 7 }, .log_decays = &.{0}, .betas = &.{0.5}, .scale = 2 };
    var bad = good;
    bad.betas = &.{1.5};
    try std.testing.expectError(error.InvalidGate, deltaChunk(bad, &state, &state, &out, &scratch));
    bad = good;
    bad.log_decays = &.{0.1};
    try std.testing.expectError(error.InvalidGate, deltaChunk(bad, &state, &state, &out, &scratch));
    bad = good;
    bad.keys = &.{ 1, 0, 0 };
    try std.testing.expectError(error.InvalidShape, deltaChunk(bad, &state, &state, &out, &scratch));
    bad = good;
    bad.values = &.{ std.math.inf(f32), 7 };
    try std.testing.expectError(error.NonFiniteInput, deltaChunk(bad, &state, &state, &out, &scratch));
    try std.testing.expectError(error.ScratchTooSmall, deltaChunk(good, &state, &state, &out, scratch[0..3]));
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4 }, &state);
    try std.testing.expectEqualSlices(f32, &.{ 9, 9 }, &out);
    // The single-token case is the sequential step (see "DeltaNet corrects…").
    try deltaChunk(good, &state, &state, &out, &scratch);
    try std.testing.expectEqualSlices(f32, &.{ 3, 2, 5, 4 }, &state);
    try std.testing.expectEqualSlices(f32, &.{ 4, 8 }, &out);
}
