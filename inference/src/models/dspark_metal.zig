//! The DSpark drafter on Metal: Inkling 2 Mini's companion (`dspark.zig`)
//! as one command buffer per commit and one per proposal, what
//! `dspark_runtime.Draft` computes (that file is the reference). Its three
//! attention caches are session layouts after the target's, at the target's
//! cache precision, so the session's rewind and truncate cover them; the
//! carried context feature `f_rnn` and its position tag live in one shared
//! buffer that a snapshot copies. The Markov chain runs on the device over
//! the full vocabulary: each position's codebook row is gathered by the id
//! the previous position's argmax wrote, so the drafts need no host round
//! trip (docs/engine/speculative-decoding.md § Inkling 2's DSpark drafter).
const std = @import("std");
const DraftStop = @import("../runtime/draft.zig").Stop;
const dspark = @import("dspark.zig");
const session = @import("../runtime/session.zig");
const metal = @import("../backends/metal/root.zig");
const cpu = @import("../backends/cpu/root.zig");
const Tensor = @import("../formats/gguf.zig").Tensor;
const Buffer = metal.Buffer;

const E = dspark.embedding;
const qw = dspark.query_width;
const kw = dspark.kv_width;
const hd = dspark.head_size;
const ff = dspark.feed_forward;
const rank = dspark.markov_rank;
const rows = dspark.block_size;
const V = dspark.vocabulary;
const PV = dspark.padded_vocabulary;
const W = dspark.hidden_width;
const attention_scale: f32 = 1.0 / @sqrt(@as(f32, hd));
/// Commit rows one command buffer encodes.
const commit_rows = 256;

/// Values `carried` holds, as on the CPU: `f_rnn` of the last committed
/// row, then that row's position plus one (0 when nothing is committed).
pub const carried_len = E + 1;

/// A weight matrix ready for dispatch.
pub const Weight = struct { buffer: Buffer, matrix: cpu.Matrix };

/// What the target lends each call: its session and the shared buffer
/// wrapping it, its token embedding and head, and `logits`, scratch of
/// `matmulPadded(block_size)` padded-vocabulary rows that no other work
/// reads while a proposal runs.
pub const Target = struct {
    backend: *metal.Backend,
    state: *session.Session,
    state_buffer: Buffer,
    token_embedding: Weight,
    output: Weight,
    logits: Buffer,

    fn slice(self: Target, region: []const u8) Buffer {
        return self.state_buffer.slice(self.state.offsetOf(region), region.len);
    }
};

const Constants = struct {
    attention_norm: Buffer,
    query_norm: Buffer,
    key_norm: Buffer,
    ffn_norm: Buffer,
    context_mix: Buffer,
    context_norm: Buffer,
    context_key_norm: Buffer,
};

pub const Draft = struct {
    binding: dspark.Binding,
    /// The first of the drafter's three session layers.
    first_layer: usize,
    constants: [dspark.block_count]Constants,
    token_norm: Buffer,
    mask: Buffer, // (block_size − 1) × E
    output_norm: Buffer,
    joint_bias: Buffer,
    w2_bias: Buffer,
    markov_mix: Buffer,
    markov_norm: Buffer,
    rope: Buffer,
    carried: Buffer, // carried_len
    /// Commit workspace: the uploaded rows, their mixed feature, keys, values.
    stage: Buffer, // commit_rows × W
    context: Buffer, // padded(commit_rows) × E
    context_k: Buffer, // padded(commit_rows) × kw
    context_v: Buffer,
    /// The block's rows (padded for the tiles) and every row's final
    /// normed hidden in `hidden`.
    x: Buffer, // padded(block) × E
    normalized: Buffer,
    q: Buffer, // padded(block) × qw
    k: Buffer, // padded(block) × kw
    v: Buffer,
    mixed: Buffer, // padded(block) × qw
    projected: Buffer, // padded(block) × E
    gate: Buffer, // padded(block) × ff
    up: Buffer,
    hidden: Buffer, // padded(block) × E
    partials: Buffer,
    /// The block's queries and attention output with its rows folded
    /// into the query heads (`[KV head][row][group]`), so one decode
    /// pass reads each cache row once for all 8; `fold` is the
    /// permutation of 384-float head groups, its own inverse.
    folded_q: Buffer, // block × qw
    folded_out: Buffer, // block × qw
    fold: Buffer, // block × kv_heads u32
    /// The chain: `[state; w1[prev]; hidden_i; f_rnn]`, `joint_proj`'s
    /// output, the Markov feature, its vocabulary bias, and the chosen ids
    /// (`codes[0]` the anchor, `codes[i]` draft `i`).
    joint_in: Buffer, // joint_input
    joint_out: Buffer, // joint_output
    markov: Buffer, // rank
    bias: Buffer, // V
    codes: Buffer, // block_size u32
    argmax_values: Buffer,
    argmax_indices: Buffer,
    topk: [dspark.max_proposals]metal.Backend.TopKBuffers,

    /// Creates every buffer through `created` and decodes the small F32
    /// weights through the companion's mapping. `capacity` sizes the rope
    /// table; `first_layer` is where the drafter's layouts start.
    pub fn init(alloc: std.mem.Allocator, created: *metal.Created, binding: dspark.Binding, capacity: usize, first_layer: usize) !Draft {
        var self: Draft = undefined;
        self.binding = binding;
        self.first_layer = first_layer;
        const view = binding.view;
        const constant = struct {
            fn f(a: std.mem.Allocator, c: *metal.Created, v: @TypeOf(view), tensor: *const Tensor) !Buffer {
                const values = try v.vector(a, tensor);
                defer a.free(values);
                const buffer = try c.create(values.len * 4);
                @memcpy(buffer.floats()[0..values.len], values);
                return buffer;
            }
        }.f;
        for (&self.constants, binding.layers) |*c, layer| c.* = .{
            .attention_norm = try constant(alloc, created, view, layer.attention_norm),
            .query_norm = try constant(alloc, created, view, layer.query_norm),
            .key_norm = try constant(alloc, created, view, layer.key_norm),
            .ffn_norm = try constant(alloc, created, view, layer.ffn_norm),
            .context_mix = try constant(alloc, created, view, layer.context_mix),
            .context_norm = try constant(alloc, created, view, layer.context_norm),
            .context_key_norm = try constant(alloc, created, view, layer.context_key_norm),
        };
        self.token_norm = try constant(alloc, created, view, binding.token_norm);
        self.mask = try constant(alloc, created, view, binding.mask);
        self.output_norm = try constant(alloc, created, view, binding.output_norm);
        self.joint_bias = try constant(alloc, created, view, binding.joint_bias);
        self.w2_bias = try constant(alloc, created, view, binding.markov_w2_bias);
        self.markov_mix = try constant(alloc, created, view, binding.markov_mix);
        self.markov_norm = try constant(alloc, created, view, binding.markov_norm);
        self.rope = try created.create(capacity * (hd / 2) * 8);
        try metal.Backend.ropeTable(self.rope, capacity, hd, dspark.rope_base, null);
        self.carried = try created.create(carried_len * 4);
        @memset(self.carried.floats(), 0);
        const commit_pad = metal.Backend.matmulPadded(commit_rows);
        self.stage = try created.create(commit_rows * W * 4);
        self.context = try created.create(commit_pad * E * 4);
        self.context_k = try created.create(commit_pad * kw * 4);
        self.context_v = try created.create(commit_pad * kw * 4);
        const pad = metal.Backend.matmulPadded(rows);
        inline for (.{ "x", "normalized", "projected", "hidden" }) |f| @field(self, f) = try created.create(pad * E * 4);
        inline for (.{ "q", "mixed" }) |f| @field(self, f) = try created.create(pad * qw * 4);
        inline for (.{ "k", "v" }) |f| @field(self, f) = try created.create(pad * kw * 4);
        inline for (.{ "gate", "up" }) |f| @field(self, f) = try created.create(pad * ff * 4);
        self.partials = try created.create(metal.Backend.attentionDecodePartials(rows * dspark.heads, hd) * 4);
        self.folded_q = try created.create(rows * qw * 4);
        self.folded_out = try created.create(rows * qw * 4);
        self.fold = try created.create(rows * dspark.kv_heads * 4);
        for (@as([*]u32, @ptrCast(@alignCast(self.fold.host)))[0 .. rows * dspark.kv_heads], 0..) |*m, i| m.* = @intCast(i % rows * dspark.kv_heads + i / rows);
        self.joint_in = try created.create(dspark.joint_input * 4);
        self.joint_out = try created.create(dspark.joint_output * 4);
        self.markov = try created.create(rank * 4);
        self.bias = try created.create(V * 4);
        self.codes = try created.create(rows * 4);
        self.argmax_values = try created.create(metal.Backend.argmax_partials * 4);
        self.argmax_indices = try created.create(metal.Backend.argmax_partials * 4);
        for (&self.topk) |*scratch| scratch.* = try created.topk(1);
        return self;
    }

    /// Device bytes the drafter owns beyond the session.
    pub fn bytes(self: *const Draft) usize {
        var total: usize = 0;
        inline for (.{ "token_norm", "mask", "output_norm", "joint_bias", "w2_bias", "markov_mix", "markov_norm", "rope", "carried", "stage", "context", "context_k", "context_v", "x", "normalized", "q", "k", "v", "mixed", "projected", "gate", "up", "hidden", "partials", "folded_q", "folded_out", "fold", "joint_in", "joint_out", "markov", "bias", "codes", "argmax_values", "argmax_indices" }) |f| total += @field(self, f).len;
        for (self.constants) |c| inline for (.{ "attention_norm", "query_norm", "key_norm", "ffn_norm", "context_mix", "context_norm", "context_key_norm" }) |f| {
            total += @field(c, f).len;
        };
        for (self.topk) |t| total += t.partial_values.len + t.partial_indices.len + t.values.len + t.indices.len + t.sums.len + t.flags.len;
        return total;
    }

    /// The carried values `Model.snapshot` copies.
    pub fn carriedValues(self: *const Draft) []f32 {
        return self.carried.floats()[0..carried_len];
    }

    /// Forgets the committed context feature: `propose` drafts nothing until
    /// the next commit.
    pub fn reset(self: *Draft) void {
        @memset(self.carriedValues(), 0);
    }

    /// Whether the carried feature is the row before `position`.
    pub fn ready(self: *const Draft, position: usize) bool {
        return position > 0 and self.carried.floats()[E] == @as(f32, @floatFromInt(position));
    }

    /// A projection whose weight lives in the companion's file: the decode
    /// matvec for one row, the tiles otherwise (inputs and outputs padded).
    fn mm(self: *const Draft, b: *metal.Backend, tensor: *const Tensor, input: Buffer, in_stride: usize, output: Buffer, out_stride: usize, count: usize) !void {
        const matrix = try self.binding.view.matrix(tensor);
        const weights = try b.wrap(matrix.bytes);
        if (count == 1) return b.matvec(weights, matrix, input.slice(0, matrix.columns * 4), output.slice(0, matrix.rows * 4));
        try b.matmul(weights, matrix, input, in_stride, output, out_stride, count);
    }

    /// Commits the target rows of `tokens` (`W` floats each, row-major as
    /// `Drafter.commit` passes them), which end at the session position:
    /// each block's context keys and values at the rows' positions, and the
    /// last row's `f_rnn` carried with its tag.
    pub fn commit(self: *Draft, t: Target, tokens: []const u32, h_rows: []const f32) !void {
        if (h_rows.len != tokens.len * W) return error.InvalidShape;
        if (tokens.len == 0) return;
        if (t.state.position < tokens.len) return error.InvalidShape;
        for (h_rows) |x| if (!std.math.isFinite(x)) return error.NonFiniteResult;
        const start = t.state.position - tokens.len;
        var offset: usize = 0;
        while (offset < tokens.len) {
            const count = @min(commit_rows, tokens.len - offset);
            try self.commitRows(t, h_rows[offset * W ..][0 .. count * W], start + offset, offset + count == tokens.len);
            offset += count;
        }
        self.carried.floats()[E] = @floatFromInt(t.state.position);
    }

    fn commitRows(self: *Draft, t: Target, h_rows: []const f32, start: usize, last: bool) !void {
        const b = t.backend;
        const count = h_rows.len / W;
        // The previous command buffer has completed: the host may write.
        @memcpy(self.stage.floats()[0..h_rows.len], h_rows);
        try b.begin();
        errdefer b.discard();
        for (self.binding.layers, self.constants, 0..) |layer, c, il| {
            try b.dsparkMix(self.stage, c.context_mix, self.context, count, E, dspark.taps, W, E);
            try b.rmsNorm(self.context, c.context_norm, self.context, norm(count, E));
            try self.mm(b, layer.context_key, self.context, E, self.context_k, kw, count);
            try self.mm(b, layer.context_value, self.context, E, self.context_v, kw, count);
            try b.rmsNorm(self.context_k, c.context_key_norm, self.context_k, norm(count * dspark.kv_heads, hd));
            try b.ropeRows(self.context_k, self.rope, dspark.kv_heads, hd, hd, start, count, kw, .split_half);
            const cache = t.state.layers[self.first_layer + il].attention;
            try b.storeRows(&.{
                .{ .dst = t.slice(cache.keys.range(start, count)), .src = self.context_k, .count = count * kw },
                .{ .dst = t.slice(cache.values.range(start, count)), .src = self.context_v, .count = count * kw },
            }, cache.keys.precision);
        }
        if (last) {
            const f_rnn = self.carried.slice(0, E * 4);
            try b.dsparkMix(self.stage.slice((count - 1) * W * 4, W * 4), self.markov_mix, f_rnn, 1, E, dspark.taps, W, E);
            try b.rmsNorm(f_rnn, self.markov_norm, f_rnn, norm(1, E));
        }
        try b.commit();
    }

    /// One block from `anchor` at the session position, then the greedy
    /// Markov chain over `out.len` positions (at most `max_proposals`), all
    /// in one command buffer; `stop` then keeps a prefix of it from each
    /// position's full-vocabulary probability (`draft.Stop.next`). Returns
    /// the count; none when the row before the position was never committed
    /// or the block does not fit.
    pub fn propose(self: *Draft, t: Target, anchor: u32, out: []u32, stop: DraftStop) !usize {
        if (anchor >= V) return error.InvalidTokenId;
        if (!stop.valid()) return error.InvalidShape;
        const count: usize = @min(out.len, dspark.max_proposals);
        const base = t.state.position;
        if (count == 0 or !self.ready(base) or base + rows > t.state.capacity) return 0;
        try self.block(t, anchor, base, count, stop.active(), false);
        const ids = @as([*]const u32, @ptrCast(@alignCast(self.codes.host)))[0..rows];
        var chain: f64 = 1;
        for (0..count) |i| {
            const id = ids[1 + i];
            if (id >= V) return error.NonFiniteResult;
            out[i] = id;
            if (stop.active()) if (stop.cut(&chain, i, topProbability(self.topk[i]))) |n| return n;
        }
        return count;
    }

    /// `generation-check --draft-stats`: one block from `anchor` at the
    /// session position and `count` chain positions, greedy, or fed
    /// `forced` (position `i`'s previous token `forced[i - 1]`, the anchor
    /// before position 1). Returns the greedy drafts (empty when forced);
    /// `chainLogits` then reads each position's capped logits.
    pub fn statsChain(self: *Draft, t: Target, anchor: u32, count: usize, forced: ?[]const u32) ![]const u32 {
        if (anchor >= V) return error.InvalidTokenId;
        if (count == 0 or count > dspark.max_proposals) return error.InvalidShape;
        const base = t.state.position;
        if (!self.ready(base) or base + rows > t.state.capacity) return error.InvalidShape;
        const codes = @as([*]u32, @ptrCast(@alignCast(self.codes.host)))[0..rows];
        if (forced) |f| {
            if (f.len + 1 < count) return error.InvalidShape;
            for (f[0 .. count - 1], 1..) |id, i| {
                if (id >= V) return error.InvalidTokenId;
                codes[i] = id;
            }
        }
        try self.block(t, anchor, base, count, false, forced != null);
        if (forced != null) return &.{};
        for (codes[1 .. count + 1]) |id| if (id >= V) return error.NonFiniteResult;
        return codes[1 .. count + 1];
    }

    /// Chain position `i`'s capped logits (0-based) after `statsChain`.
    pub fn chainLogits(_: *const Draft, t: Target, i: usize) []const f32 {
        return t.logits.floats()[(i + 1) * PV ..][0..V];
    }

    /// Records and runs the block and `count` chain positions; `probability`
    /// also takes each position's partial softmax sums. `forced` keeps the
    /// host-written ids in `codes` as each position's previous token instead
    /// of the chain's own argmax.
    fn block(self: *Draft, t: Target, anchor: u32, base: usize, count: usize, probability: bool, forced: bool) !void {
        const b = t.backend;
        const last = base + rows - 1;
        // The chain's fixed inputs: a zero state, the carried f_rnn, the anchor.
        @memset(self.joint_in.floats()[0..rank], 0);
        @memcpy(self.joint_in.floats()[2 * rank + E ..][0..E], self.carried.floats()[0..E]);
        @as([*]u32, @ptrCast(@alignCast(self.codes.host)))[0] = anchor;
        try b.begin();
        errdefer b.discard();
        try b.embed(t.token_embedding.buffer, t.token_embedding.matrix, anchor, self.x.slice(0, E * 4));
        try b.rmsNorm(self.x, self.token_norm, self.x, norm(1, E));
        try b.copy(self.x.slice(E * 4, (rows - 1) * E * 4), self.mask, (rows - 1) * E);
        for (self.binding.layers, self.constants, 0..) |layer, c, il| {
            try b.rmsNorm(self.x, c.attention_norm, self.normalized, norm(rows, E));
            try self.mm(b, layer.query, self.normalized, E, self.q, qw, rows);
            try self.mm(b, layer.key, self.normalized, E, self.k, kw, rows);
            try self.mm(b, layer.value, self.normalized, E, self.v, kw, rows);
            try b.rmsNorm(self.q, c.query_norm, self.q, norm(rows * dspark.heads, hd));
            try b.ropeRows(self.q, self.rope, dspark.heads, hd, hd, base, rows, qw, .split_half);
            try b.rmsNorm(self.k, c.key_norm, self.k, norm(rows * dspark.kv_heads, hd));
            try b.ropeRows(self.k, self.rope, dspark.kv_heads, hd, hd, base, rows, kw, .split_half);
            const cache = t.state.layers[self.first_layer + il].attention;
            try b.storeRows(&.{
                .{ .dst = t.slice(cache.keys.range(base, rows)), .src = self.k, .count = rows * kw },
                .{ .dst = t.slice(cache.values.range(base, rows)), .src = self.v, .count = rows * kw },
            }, cache.keys.precision);
            // Non-causal inside the block, every row in the anchor's window:
            // a sliding block reads the anchor and the `window − 1`
            // committed rows before it.
            const first = if (dspark.sliding[il] and base + 1 > dspark.window) base + 1 - dspark.window else 0;
            const visible = last - first + 1;
            const keys = t.slice(cache.keys.range(first, visible));
            const values = t.slice(cache.values.range(first, visible));
            const group = dspark.heads / dspark.kv_heads * hd;
            const groups = rows * dspark.kv_heads;
            try b.gatherRows(self.folded_q, self.q, self.fold, group, groups, 1, rows * qw, rows * qw);
            try b.attentionDecode(keys, values, self.folded_q, self.partials, self.folded_out, .{
                .query_heads = rows * dspark.heads,
                .kv_heads = dspark.kv_heads,
                .key_width = hd,
                .value_width = hd,
                .visible = visible,
                .scale = attention_scale,
                .precision = cache.keys.precision,
            });
            try b.gatherRows(self.mixed, self.folded_out, self.fold, group, groups, 1, rows * qw, rows * qw);
            try self.mm(b, layer.output, self.mixed, qw, self.projected, E, rows);
            try b.add(self.x, self.projected, rows * E);
            try b.rmsNorm(self.x, c.ffn_norm, self.normalized, norm(rows, E));
            try self.mm(b, layer.ffn_gate, self.normalized, E, self.gate, ff, rows);
            try self.mm(b, layer.ffn_up, self.normalized, E, self.up, ff, rows);
            try b.siluMul(self.gate, self.up, rows * ff);
            try self.mm(b, layer.ffn_down, self.gate, ff, self.projected, E, rows);
            try b.add(self.x, self.projected, rows * E);
        }
        try b.rmsNorm(self.x, self.output_norm, self.hidden, norm(rows, E));
        // Base logits of rows 0 … count in one pass of the target's head
        // (row 0's are unused): the 8-row matmul costs about what one row's
        // matvec does, and the chain needs no other head read.
        const logits = t.logits;
        try b.matmul(t.output.buffer, t.output.matrix, self.hidden, E, logits, PV, count + 1);
        try b.addBiasRows(logits.slice(PV * 4, logits.len - PV * 4), self.w2_bias, V, count, PV);
        const w1 = try self.wrapped(b, self.binding.markov_w1);
        const joint = try self.wrapped(b, self.binding.joint);
        const w2 = try self.wrapped(b, self.binding.markov_w2);
        const state = self.joint_in.slice(0, rank * 4);
        for (1..count + 1) |i| {
            try b.copy(self.joint_in.slice(2 * rank * 4, E * 4), self.hidden.slice(i * E * 4, E * 4), E);
            try b.ttsEmbedRow(w1.buffer, V, rank, self.codes, i - 1, self.joint_in.slice(rank * 4, rank * 4), false);
            try b.matvec(joint.buffer, joint.matrix, self.joint_in, self.joint_out);
            try b.dsparkMarkov(self.joint_out, self.joint_bias, state, self.markov, rank);
            try b.matvec(w2.buffer, w2.matrix, self.markov, self.bias);
            const row = logits.slice(i * PV * 4, V * 4);
            try b.add(row, self.bias, V);
            try b.softcap(row, V, dspark.softcap);
            if (!forced) try b.argmax(row, V, self.argmax_values, self.argmax_indices, self.codes.slice(i * 4, 4));
            if (probability) try b.topk(row, V, 1, 1.0, self.topk[i - 1]);
        }
        try b.commit();
    }

    fn wrapped(self: *const Draft, b: *metal.Backend, tensor: *const Tensor) !Weight {
        const matrix = try self.binding.view.matrix(tensor);
        return .{ .buffer = try b.wrap(matrix.bytes), .matrix = matrix };
    }
};

/// The full-vocabulary softmax probability of a position's choice from its
/// partial sums: one over `Σ exp(l − max)`, or 1 when a partition was not
/// finite (the id check reports that).
fn topProbability(scratch: metal.Backend.TopKBuffers) f64 {
    var total: f64 = 0;
    for (scratch.sums.floats()[0..metal.Backend.topk_partials]) |partial| total += partial;
    for (@as([*]const u32, @ptrCast(@alignCast(scratch.flags.host)))[0..metal.Backend.topk_partials]) |flag| if (flag != 0) return 1;
    return if (total > 0) 1.0 / total else 1;
}

fn norm(count: usize, width: usize) metal.Backend.Norm {
    return .{ .rows = count, .width = width, .in_stride = width, .out_stride = width, .eps = dspark.rms_epsilon };
}
