//! The DSpark drafter on the CPU: the numerical reference for Inkling 2
//! Mini's companion (`dspark.zig`). It owns its workspace, its decoded
//! constants, and the carried context feature `f_rnn`, but no cache: its
//! three attention caches are session layouts the target's runtime appends
//! after its own (`first_layer ..`), so recovery rewinds them with the
//! session. The target supplies its token embedding and output head per
//! call. The equations are in docs/engine/speculative-decoding.md
//! § Inkling 2's DSpark drafter.
const std = @import("std");
const DraftStop = @import("../runtime/draft.zig").Stop;
const dspark = @import("dspark.zig");
const weights = @import("../runtime/weights.zig");
const session = @import("../runtime/session.zig");
const cpu = @import("../backends/cpu/root.zig");
const Tensor = @import("../formats/gguf.zig").Tensor;

const E = dspark.embedding;
const rows = dspark.block_size;
const rank = dspark.markov_rank;
const attention_scale: f32 = 1.0 / @sqrt(@as(f32, dspark.head_size));

/// The target model's tensors the drafter shares: its embedding table and
/// output head, read through the target's mapping.
pub const Target = struct {
    view: weights.View,
    token_embedding: *const Tensor,
    output: *const Tensor,
};

/// One block's decoded constants.
const Constants = struct {
    attention_norm: []f32,
    query_norm: []f32,
    key_norm: []f32,
    ffn_norm: []f32,
    context_mix: []f32,
    context_norm: []f32,
    context_key_norm: []f32,
};

/// Values `carried` holds: `f_rnn` of the last committed row, then that
/// row's position plus one (0 when nothing is committed).
pub const carried_len = E + 1;

pub const Draft = struct {
    binding: dspark.Binding,
    io: std.Io,
    constants: [dspark.block_count]Constants,
    token_norm: []f32,
    mask: []f32,
    output_norm: []f32,
    joint_bias: []f32,
    w2_bias: []f32,
    markov_mix: []f32,
    markov_norm: []f32,
    /// The context feature of the last committed row and its tag
    /// (`carried_len`); what a snapshot copies besides the session.
    carried: []f32,
    /// One commit row's mixed feature, and its key and value.
    mixed_context: []f32,
    k: []f32,
    v: []f32,
    /// The block's rows: the residual stream, the normed sublayer input, the
    /// queries, and every row's final normed hidden.
    x_rows: []f32,
    normalized_rows: []f32,
    q_rows: []f32,
    hidden: []f32,
    mixed: []f32,
    sub: []f32,
    gate: []f32,
    up: []f32,
    /// The Markov chain's position workspace: the joint input and output,
    /// the recurrent state, the head's base logits over the padded rows,
    /// the vocabulary bias, and the capped logits of the last position.
    joint_in: []f32,
    joint_out: []f32,
    state: []f32,
    markov: []f32,
    base: []f32,
    bias: []f32,
    logits: []f32,
    row: []f32,
    attention_scratch: []f64,
    /// The anchor of the last block and the next chain position (1-based
    /// block row) `nextPosition` computes.
    anchor: u32 = 0,
    position: usize = 0,

    /// Allocates from `a` (an arena the caller owns) and decodes the small
    /// F32 weights through the companion's own mapping. `capacity` bounds
    /// the positions attention reads.
    pub fn init(a: std.mem.Allocator, io: std.Io, binding: dspark.Binding, capacity: usize) !Draft {
        var result: Draft = undefined;
        result.binding = binding;
        result.io = io;
        const view = binding.view;
        for (&result.constants, binding.layers) |*c, layer| c.* = .{
            .attention_norm = try view.vector(a, layer.attention_norm),
            .query_norm = try view.vector(a, layer.query_norm),
            .key_norm = try view.vector(a, layer.key_norm),
            .ffn_norm = try view.vector(a, layer.ffn_norm),
            .context_mix = try view.vector(a, layer.context_mix),
            .context_norm = try view.vector(a, layer.context_norm),
            .context_key_norm = try view.vector(a, layer.context_key_norm),
        };
        result.token_norm = try view.vector(a, binding.token_norm);
        result.mask = try view.vector(a, binding.mask);
        result.output_norm = try view.vector(a, binding.output_norm);
        result.joint_bias = try view.vector(a, binding.joint_bias);
        result.w2_bias = try view.vector(a, binding.markov_w2_bias);
        result.markov_mix = try view.vector(a, binding.markov_mix);
        result.markov_norm = try view.vector(a, binding.markov_norm);
        result.carried = try a.alloc(f32, carried_len);
        @memset(result.carried, 0);
        result.mixed_context = try a.alloc(f32, E);
        result.k = try a.alloc(f32, dspark.kv_width);
        result.v = try a.alloc(f32, dspark.kv_width);
        inline for (.{ "x_rows", "normalized_rows", "hidden" }) |field| @field(result, field) = try a.alloc(f32, rows * E);
        result.q_rows = try a.alloc(f32, rows * dspark.query_width);
        result.mixed = try a.alloc(f32, dspark.query_width);
        result.sub = try a.alloc(f32, E);
        result.gate = try a.alloc(f32, dspark.feed_forward);
        result.up = try a.alloc(f32, dspark.feed_forward);
        result.joint_in = try a.alloc(f32, dspark.joint_input);
        result.joint_out = try a.alloc(f32, dspark.joint_output);
        result.state = try a.alloc(f32, rank);
        result.markov = try a.alloc(f32, rank);
        result.base = try a.alloc(f32, dspark.padded_vocabulary);
        result.bias = try a.alloc(f32, dspark.vocabulary);
        result.logits = try a.alloc(f32, dspark.vocabulary);
        result.row = try a.alloc(f32, cpu.matvecScratch(@max(dspark.joint_input, dspark.feed_forward)));
        result.attention_scratch = try a.alloc(f64, capacity + rows);
        result.anchor = 0;
        result.position = 0;
        return result;
    }

    /// Workspace bytes beyond the session, for the memory record.
    pub fn bytes(self: *const Draft) usize {
        var n: usize = 0;
        inline for (.{ "carried", "mixed_context", "k", "v", "x_rows", "normalized_rows", "q_rows", "hidden", "mixed", "sub", "gate", "up", "joint_in", "joint_out", "state", "markov", "base", "bias", "logits", "row" }) |field| n += @field(self, field).len * @sizeOf(f32);
        return n + self.attention_scratch.len * @sizeOf(f64);
    }

    fn mm(self: *Draft, tensor: *const Tensor, input: []const f32, output: []f32) !void {
        try cpu.matvec(self.io, try self.binding.view.matrix(tensor), input, output, self.row);
    }

    fn norm(input: []const f32, output: []f32, weight: []const f32) !void {
        if (weight.len != output.len) return error.InvalidShape;
        try cpu.rmsNorm(input, output, dspark.rms_epsilon);
        for (output, weight) |*x, w| x.* *= w;
    }

    fn rope(head: []f32, position: usize) !void {
        try cpu.rope.apply(head, head, .{ .dimensions = dspark.head_size, .base = dspark.rope_base, .position = @intCast(position) });
    }

    /// Forgets the committed context feature: `propose` drafts nothing until
    /// the next commit.
    pub fn reset(self: *Draft) void {
        @memset(self.carried, 0);
    }

    /// Commits one target row (`dspark.hidden_width`: the four target rows)
    /// at `position`: each block's key and value, `k_ctx_norm(k_ctx(c))`
    /// with RoPE and `v_ctx(c)` for `c = ctx_norm(mix(row))`, and the
    /// carried `f_rnn = markov_norm(markov_mix(row))`.
    pub fn commitRow(self: *Draft, state: *session.Session, first_layer: usize, position: usize, features: []const f32) !void {
        if (features.len != dspark.hidden_width) return error.InvalidShape;
        for (features) |x| if (!std.math.isFinite(x)) return error.NonFiniteResult;
        for (self.binding.layers, self.constants, 0..) |layer, c, il| {
            dspark.mixTaps(features, c.context_mix, self.mixed_context);
            try norm(self.mixed_context, self.mixed_context, c.context_norm);
            try self.mm(layer.context_key, self.mixed_context, self.k);
            try self.mm(layer.context_value, self.mixed_context, self.v);
            for (0..dspark.kv_heads) |h| {
                const head = self.k[h * dspark.head_size ..][0..dspark.head_size];
                try norm(head, head, c.context_key_norm);
                try rope(head, position);
            }
            const cache = state.layers[first_layer + il].attention;
            @memcpy(cache.keys.floats(position, 1), self.k);
            @memcpy(cache.values.floats(position, 1), self.v);
        }
        const f_rnn = self.carried[0..E];
        dspark.mixTaps(features, self.markov_mix, f_rnn);
        try norm(f_rnn, f_rnn, self.markov_norm);
        self.carried[E] = @floatFromInt(position + 1);
    }

    /// Whether the carried feature is the row before `position`: the block
    /// anchored there can draft.
    pub fn ready(self: *const Draft, position: usize) bool {
        return position > 0 and self.carried[E] == @as(f32, @floatFromInt(position));
    }

    /// Runs the block `[anchor, mask × 7]` at positions `base … base + 7`:
    /// non-causal inside the block, each sliding block reading the committed
    /// rows within `window` of the anchor, the full block every committed
    /// row. Every row's final normed hidden lands in `hidden`; the block's
    /// key/value rows are written at their positions (stale once the caller
    /// commits, never read past the block). Then the chain restarts.
    pub fn block(self: *Draft, target: Target, state: *session.Session, first_layer: usize, anchor: u32, base: usize) !void {
        if (base + rows > state.capacity) return error.ContextFull;
        const last = base + rows - 1;
        const x0 = self.x_rows[0..E];
        try target.view.row(target.token_embedding, anchor, x0);
        try norm(x0, x0, self.token_norm);
        @memcpy(self.x_rows[E..][0 .. (rows - 1) * E], self.mask);
        for (self.binding.layers, self.constants, 0..) |layer, c, il| {
            const cache = state.layers[first_layer + il].attention;
            for (0..rows) |r| {
                const position = base + r;
                const x = self.normalized_rows[r * E ..][0..E];
                try norm(self.x_rows[r * E ..][0..E], x, c.attention_norm);
                const q = self.q_rows[r * dspark.query_width ..][0..dspark.query_width];
                try self.mm(layer.query, x, q);
                try self.mm(layer.key, x, self.k);
                try self.mm(layer.value, x, self.v);
                for (0..dspark.heads) |h| {
                    const head = q[h * dspark.head_size ..][0..dspark.head_size];
                    try norm(head, head, c.query_norm);
                    try rope(head, position);
                }
                for (0..dspark.kv_heads) |h| {
                    const head = self.k[h * dspark.head_size ..][0..dspark.head_size];
                    try norm(head, head, c.key_norm);
                    try rope(head, position);
                }
                @memcpy(cache.keys.floats(position, 1), self.k);
                @memcpy(cache.values.floats(position, 1), self.v);
            }
            // Every row shares the anchor's window: a sliding block reads
            // the anchor and the `window - 1` committed rows before it.
            const first = if (dspark.sliding[il] and base + 1 > dspark.window) base + 1 - dspark.window else 0;
            const visible = last - first + 1;
            for (0..rows) |r| {
                try cpu.attention.apply(.{
                    .query_heads = dspark.heads,
                    .kv_heads = dspark.kv_heads,
                    .key_width = dspark.head_size,
                    .value_width = dspark.head_size,
                    .tokens = visible,
                    .visible_tokens = visible,
                    .scale = attention_scale,
                    .queries = self.q_rows[r * dspark.query_width ..][0..dspark.query_width],
                    .keys = cache.keys.floats(first, visible),
                    .values = cache.values.floats(first, visible),
                }, self.mixed, self.attention_scratch);
                try self.mm(layer.output, self.mixed, self.sub);
                for (self.x_rows[r * E ..][0..E], self.sub) |*x, add| x.* += add;
            }
            for (0..rows) |r| {
                const xr = self.x_rows[r * E ..][0..E];
                try norm(xr, self.sub, c.ffn_norm);
                try self.mm(layer.ffn_gate, self.sub, self.gate);
                try self.mm(layer.ffn_up, self.sub, self.up);
                for (self.gate, self.up) |*g, u| g.* = cpu.silu(g.*) * u;
                try self.mm(layer.ffn_down, self.gate, self.sub);
                for (xr, self.sub) |*x, add| x.* += add;
            }
        }
        for (0..rows) |r| {
            const x = self.x_rows[r * E ..][0..E];
            for (x) |v| if (!std.math.isFinite(v)) return error.NonFiniteResult;
            try norm(x, self.hidden[r * E ..][0..E], self.output_norm);
        }
        self.anchor = anchor;
        self.restart();
    }

    /// The next chain position of the last block: its base logits (the
    /// target's head on the row's hidden, in `base`), the Markov bias from
    /// `previous` (the anchor, then the token taken at the position before),
    /// and the capped logits in `logits`. Returns the argmax, the lowest id
    /// on a tie. Call after `block`, at most `max_proposals` times.
    pub fn nextPosition(self: *Draft, target: Target, previous: u32) !u32 {
        try self.baseLogits(target, self.position, self.base);
        return self.chainStep(self.base[0..dspark.vocabulary], previous);
    }

    /// The target's head on block row `row`'s hidden: `padded_vocabulary`
    /// values into `out`, uncapped.
    pub fn baseLogits(self: *Draft, target: Target, row: usize, out: []f32) !void {
        if (row == 0 or row >= rows) return error.InvalidShape;
        try cpu.matvec(self.io, try target.view.matrix(target.output), self.hidden[row * E ..][0..E], out, self.row);
    }

    /// Restarts the last block's chain at its first draft position.
    pub fn restart(self: *Draft) void {
        self.position = 1;
        @memset(self.state, 0);
    }

    /// One chain position from its base logits (`vocabulary` values): the
    /// recurrence from `previous`, the bias, and the capped logits in
    /// `logits`. Returns their argmax, the lowest id on a tie.
    pub fn chainStep(self: *Draft, base: []const f32, previous: u32) !u32 {
        const i = self.position;
        if (i == 0 or i >= rows) return error.InvalidShape;
        if (base.len != dspark.vocabulary) return error.InvalidShape;
        if (previous >= dspark.vocabulary) return error.InvalidTokenId;
        @memcpy(self.joint_in[0..rank], self.state);
        try self.binding.view.row(self.binding.markov_w1, previous, self.joint_in[rank..][0..rank]);
        @memcpy(self.joint_in[2 * rank ..][0..E], self.hidden[i * E ..][0..E]);
        @memcpy(self.joint_in[2 * rank + E ..][0..E], self.carried[0..E]);
        try self.mm(self.binding.joint, self.joint_in, self.joint_out);
        for (self.joint_out, self.joint_bias) |*j, b| j.* += b;
        dspark.markovStep(self.joint_out, self.state, self.markov);
        try self.mm(self.binding.markov_w2, self.markov, self.bias);
        for (self.bias, self.w2_bias) |*x, b| x.* += b;
        dspark.capped(base, self.bias, self.logits);
        var best: usize = 0;
        for (self.logits, 0..) |v, t| {
            if (!std.math.isFinite(v)) return error.NonFiniteResult;
            if (v > self.logits[best]) best = t;
        }
        self.position = i + 1;
        return @intCast(best);
    }

    /// The greedy chain over block rows `1 .. 1 + out.len`; `stop` ends it
    /// on each choice's full-vocabulary probability.
    pub fn select(self: *Draft, target: Target, out: []u32, stop: DraftStop) !usize {
        var previous = self.anchor;
        var count: usize = 0;
        var chain: f64 = 1;
        while (count < out.len) {
            const chosen = try self.nextPosition(target, previous);
            out[count] = chosen;
            previous = chosen;
            count += 1;
            if (stop.active()) {
                if (stop.cut(&chain, count - 1, topProbability(self.logits))) |n| {
                    count = n;
                    break;
                }
            }
        }
        return count;
    }
};

/// The full-vocabulary softmax probability of the argmax.
pub fn topProbability(logits: []const f32) f64 {
    var maximum: f64 = -std.math.inf(f64);
    for (logits) |v| maximum = @max(maximum, v);
    var total: f64 = 0;
    for (logits) |v| total += @exp(@as(f64, v) - maximum);
    return 1.0 / total;
}

test "the top probability of a uniform row is one over its width" {
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), topProbability(&.{ 1, 1, 1, 1 }), 1e-12);
    try std.testing.expect(topProbability(&.{ 50, 0, 0 }) > 0.999);
}
