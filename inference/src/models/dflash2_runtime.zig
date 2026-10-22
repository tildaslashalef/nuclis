//! The DFlash2 drafter on the CPU: the numerical reference for the Qwen3.8
//! companion (`dflash2.zig`). It owns its workspace and decoded constants but
//! no cache: its five attention caches are session layouts the target's
//! runtime appends after its own (`first_layer ..`), so recovery rewinds
//! them with the session. The target supplies its token embedding and output
//! head per call. The equations are in docs/engine/speculative-decoding.md
//! § The Qwen DFlash2 drafter.
const std = @import("std");
const dflash2 = @import("dflash2.zig");
const weights = @import("../runtime/weights.zig");
const session = @import("../runtime/session.zig");
const cpu = @import("../backends/cpu/root.zig");
const Tensor = @import("../formats/gguf.zig").Tensor;

const embedding = dflash2.embedding;
const block_size = dflash2.block_size;
const attention_scale: f32 = 1.0 / @sqrt(@as(f32, dflash2.head_size));

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
    attention_base: []f32,
    ffn_base: []f32,
};

pub const Draft = struct {
    binding: dflash2.Binding,
    io: std.Io,
    constants: [dflash2.block_count]Constants,
    encoder_norm: []f32,
    output_norm: []f32,
    /// One position's encoder output, the source of its injected keys/values.
    encoded: []f32,
    k: []f32,
    v: []f32,
    /// The block's rows: the residual stream, the normed sublayer input, its
    /// convolution taps, the convolved input, the sublayer output, and the
    /// convolved output. `block_size` rows each.
    x_rows: []f32,
    normalized_rows: []f32,
    taps_rows: []f32,
    prepared_rows: []f32,
    sub_rows: []f32,
    finished_rows: []f32,
    q_rows: []f32,
    mixed: []f32,
    gate: []f32,
    up: []f32,
    /// Every block row's final normed hidden.
    hidden: []f32,
    logits: []f32,
    /// The selector's per-position workspace.
    projected: []f32,
    predecessor: []f32,
    successors: []f32,
    unary: [dflash2.selector_top_k]f32,
    candidates: [dflash2.selector_top_k]u32,
    scores: [dflash2.selector_top_k]f32,
    row: []f32,
    attention_scratch: []f64,

    /// Allocates from `a` (an arena the caller owns) and decodes the small
    /// F32 weights through the companion's own mapping.
    pub fn init(a: std.mem.Allocator, io: std.Io, binding: dflash2.Binding) !Draft {
        var result: Draft = undefined;
        result.binding = binding;
        result.io = io;
        const view = binding.view;
        for (&result.constants, binding.layers) |*c, layer| c.* = .{
            .attention_norm = try view.vector(a, layer.attention_norm),
            .query_norm = try view.vector(a, layer.query_norm),
            .key_norm = try view.vector(a, layer.key_norm),
            .ffn_norm = try view.vector(a, layer.ffn_norm),
            .attention_base = try view.vector(a, layer.attention_conv.base),
            .ffn_base = try view.vector(a, layer.ffn_conv.base),
        };
        result.encoder_norm = try view.vector(a, binding.encoder_norm);
        result.output_norm = try view.vector(a, binding.output_norm);
        result.encoded = try a.alloc(f32, embedding);
        result.k = try a.alloc(f32, dflash2.kv_width);
        result.v = try a.alloc(f32, dflash2.kv_width);
        inline for (.{ "x_rows", "normalized_rows", "prepared_rows", "sub_rows", "finished_rows", "hidden" }) |field| @field(result, field) = try a.alloc(f32, block_size * embedding);
        result.taps_rows = try a.alloc(f32, block_size * dflash2.conv_projection);
        result.q_rows = try a.alloc(f32, block_size * dflash2.query_width);
        result.mixed = try a.alloc(f32, dflash2.query_width);
        result.gate = try a.alloc(f32, dflash2.feed_forward);
        result.up = try a.alloc(f32, dflash2.feed_forward);
        result.logits = try a.alloc(f32, dflash2.vocabulary);
        result.projected = try a.alloc(f32, dflash2.selector_rank);
        result.predecessor = try a.alloc(f32, dflash2.selector_rank);
        result.successors = try a.alloc(f32, dflash2.selector_top_k * dflash2.selector_rank);
        result.row = try a.alloc(f32, cpu.matvecScratch(dflash2.hidden_width));
        result.attention_scratch = try a.alloc(f64, dflash2.window + block_size);
        return result;
    }

    fn mm(self: *Draft, tensor: *const Tensor, input: []const f32, output: []f32) !void {
        try cpu.matvec(self.io, try self.binding.view.matrix(tensor), input, output, self.row);
    }

    fn norm(input: []const f32, output: []f32, weight: []const f32) !void {
        if (weight.len != output.len) return error.InvalidShape;
        try cpu.rmsNorm(input, output, dflash2.rms_epsilon);
        for (output, weight) |*x, w| x.* *= w;
    }

    fn rope(head: []f32, position: usize) !void {
        try cpu.rope.apply(head, head, .{ .dimensions = dflash2.head_size, .base = dflash2.rope_base, .position = @intCast(position) });
    }

    /// Five target residuals to one feature: `enc.output_norm(fc(row))`.
    pub fn encode(self: *Draft, residuals: []const f32) !void {
        if (residuals.len != dflash2.hidden_width) return error.InvalidShape;
        try self.mm(self.binding.fc, residuals, self.encoded);
        try norm(self.encoded, self.encoded, self.encoder_norm);
    }

    /// Injects the encoder output as every block's key and value at
    /// `position`: `k_norm(k(feature))` with RoPE, and `v(feature)`. The
    /// context rows take no convolution.
    pub fn inject(self: *Draft, state: *session.Session, first_layer: usize, position: usize) !void {
        for (self.binding.layers, self.constants, 0..) |layer, c, il| {
            try self.mm(layer.key, self.encoded, self.k);
            try self.mm(layer.value, self.encoded, self.v);
            for (0..dflash2.kv_heads) |h| {
                const head = self.k[h * dflash2.head_size ..][0..dflash2.head_size];
                try norm(head, head, c.key_norm);
                try rope(head, position);
            }
            const cache = state.layers[first_layer + il].attention;
            @memcpy(cache.keys.floats(position, 1), self.k);
            @memcpy(cache.values.floats(position, 1), self.v);
        }
    }

    /// Runs the noise block of `rows` rows from `token` at `base`: row 0 the
    /// anchor, the rest the mask token. Each layer runs across all rows,
    /// since the convolutions mix a row with the one before it. Every row's
    /// final normed hidden lands in `hidden`; the block's key/value rows are
    /// written at their positions (stale once the caller commits, as the
    /// reference crops them).
    pub fn block(self: *Draft, target: Target, state: *session.Session, first_layer: usize, token: u32, rows: usize, base: usize) !void {
        if (rows == 0 or rows > block_size) return error.InvalidShape;
        if (base + rows > state.capacity) return error.ContextFull;
        const last = base + rows - 1;
        for (0..rows) |r| {
            const input = if (r == 0) token else dflash2.mask_token;
            try target.view.row(target.token_embedding, input, self.x_rows[r * embedding ..][0..embedding]);
        }
        for (self.binding.layers, self.constants, 0..) |layer, c, il| {
            const cache = state.layers[first_layer + il].attention;
            // Attention: norm, prepare-convolve, project, rope, cache; then
            // every row attends and the output is finish-convolved.
            try self.prepare(layer.attention_conv, c.attention_base, c.attention_norm, rows);
            for (0..rows) |r| {
                const position = base + r;
                const x = self.prepared_rows[r * embedding ..][0..embedding];
                const q = self.q_rows[r * dflash2.query_width ..][0..dflash2.query_width];
                try self.mm(layer.query, x, q);
                try self.mm(layer.key, x, self.k);
                try self.mm(layer.value, x, self.v);
                for (0..dflash2.heads) |h| {
                    const head = q[h * dflash2.head_size ..][0..dflash2.head_size];
                    try norm(head, head, c.query_norm);
                    try rope(head, position);
                }
                for (0..dflash2.kv_heads) |h| {
                    const head = self.k[h * dflash2.head_size ..][0..dflash2.head_size];
                    try norm(head, head, c.key_norm);
                    try rope(head, position);
                }
                @memcpy(cache.keys.floats(position, 1), self.k);
                @memcpy(cache.values.floats(position, 1), self.v);
            }
            for (0..rows) |r| {
                const position = base + r;
                // Non-causal inside the block; the prefix back to the window.
                const first = if (position + 1 > dflash2.window) position + 1 - dflash2.window else 0;
                const visible = last - first + 1;
                try cpu.attention.apply(.{
                    .query_heads = dflash2.heads,
                    .kv_heads = dflash2.kv_heads,
                    .key_width = dflash2.head_size,
                    .value_width = dflash2.head_size,
                    .tokens = visible,
                    .visible_tokens = visible,
                    .scale = attention_scale,
                    .queries = self.q_rows[r * dflash2.query_width ..][0..dflash2.query_width],
                    .keys = cache.keys.floats(first, visible),
                    .values = cache.values.floats(first, visible),
                }, self.mixed, self.attention_scratch);
                try self.mm(layer.output, self.mixed, self.sub_rows[r * embedding ..][0..embedding]);
            }
            try self.finish(c.attention_base, rows);
            // The FFN, wrapped the same way.
            try self.prepare(layer.ffn_conv, c.ffn_base, c.ffn_norm, rows);
            for (0..rows) |r| {
                const x = self.prepared_rows[r * embedding ..][0..embedding];
                try self.mm(layer.ffn_gate, x, self.gate);
                try self.mm(layer.ffn_up, x, self.up);
                for (self.gate, self.up) |*g, u| g.* = cpu.silu(g.*) * u;
                try self.mm(layer.ffn_down, self.gate, self.sub_rows[r * embedding ..][0..embedding]);
            }
            try self.finish(c.ffn_base, rows);
        }
        for (0..rows) |r| {
            const x = self.x_rows[r * embedding ..][0..embedding];
            for (x) |v| if (!std.math.isFinite(v)) return error.NonFiniteResult;
            try norm(x, self.hidden[r * embedding ..][0..embedding], self.output_norm);
        }
    }

    /// A sublayer's input: `normalized = norm(x)`, its taps from the
    /// projection, and `prepared = convolve(half 0, normalized)`.
    fn prepare(self: *Draft, conv: dflash2.Conv, base: []const f32, weight: []const f32, rows: usize) !void {
        for (0..rows) |r| {
            const normalized = self.normalized_rows[r * embedding ..][0..embedding];
            try norm(self.x_rows[r * embedding ..][0..embedding], normalized, weight);
            try self.mm(conv.projection, normalized, self.taps_rows[r * dflash2.conv_projection ..][0..dflash2.conv_projection]);
        }
        dflash2.convolve(0, base, self.taps_rows, self.normalized_rows, self.prepared_rows, rows);
    }

    /// The sublayer's output `sub` convolved with half 1 of the same taps,
    /// added to the residual stream.
    fn finish(self: *Draft, base: []const f32, rows: usize) !void {
        dflash2.convolve(1, base, self.taps_rows, self.sub_rows, self.finished_rows, rows);
        for (self.x_rows[0 .. rows * embedding], self.finished_rows[0 .. rows * embedding]) |*x, contribution| x.* += contribution;
    }

    /// The selector over block rows `1 .. 1 + count`, chained from `anchor`:
    /// each position's top-16 logits from the target's head, re-scored with
    /// the bilinear term of the chosen predecessor, the greedy choice taken.
    /// `p_min > 0` stops after a position whose choice's full-vocabulary
    /// softmax probability is below it; the first position is always proposed.
    pub fn select(self: *Draft, target: Target, anchor: u32, out: []u32, p_min: f32) !usize {
        const head = try target.view.matrix(target.output);
        var previous = anchor;
        var count: usize = 0;
        while (count < out.len) {
            const row = self.hidden[(1 + count) * embedding ..][0..embedding];
            try cpu.matvec(self.io, head, row, self.logits, self.row);
            topK(self.logits, &self.unary, &self.candidates);
            try self.mm(self.binding.selector_hidden, row, self.projected);
            try self.binding.view.row(self.binding.selector_predecessor, previous, self.predecessor);
            for (self.candidates, 0..) |candidate, k| try self.binding.view.row(self.binding.selector_successor, candidate, self.successors[k * dflash2.selector_rank ..][0..dflash2.selector_rank]);
            dflash2.selectorScores(&self.unary, self.predecessor, self.projected, self.successors, &self.scores);
            const chosen = self.candidates[dflash2.selectorChoice(&self.scores)];
            out[count] = chosen;
            previous = chosen;
            count += 1;
            if (p_min > 0 and probability(self.logits, chosen) < p_min) break;
        }
        return count;
    }
};

/// The `k` largest values and their indices, unordered; ties keep the lower
/// index. `values` must hold at least `k` finite entries.
pub fn topK(values: []const f32, top: []f32, ids: []u32) void {
    const k = top.len;
    std.debug.assert(ids.len == k and values.len >= k);
    for (0..k) |i| {
        top[i] = values[i];
        ids[i] = @intCast(i);
    }
    var lowest = minIndex(top);
    for (values[k..], k..) |v, i| {
        if (v > top[lowest]) {
            top[lowest] = v;
            ids[lowest] = @intCast(i);
            lowest = minIndex(top);
        }
    }
}

fn minIndex(values: []const f32) usize {
    var lowest: usize = 0;
    for (values, 0..) |v, i| if (v < values[lowest]) {
        lowest = i;
    };
    return lowest;
}

/// The full-vocabulary softmax probability of `token`.
fn probability(logits: []const f32, token: u32) f64 {
    var maximum: f64 = -std.math.inf(f64);
    for (logits) |v| maximum = @max(maximum, v);
    var total: f64 = 0;
    for (logits) |v| total += @exp(@as(f64, v) - maximum);
    return @exp(@as(f64, logits[token]) - maximum) / total;
}

test "top-k keeps the largest values" {
    const values = [_]f32{ 1, 9, 3, 7, 5, 8 };
    var top: [3]f32 = undefined;
    var ids: [3]u32 = undefined;
    topK(&values, &top, &ids);
    var seen: [6]bool = @splat(false);
    for (ids) |id| seen[id] = true;
    try std.testing.expect(seen[1] and seen[3] and seen[5] and !seen[0]);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), probability(&[_]f32{ 0, 0, 0 }, 1), 1e-12);
}
