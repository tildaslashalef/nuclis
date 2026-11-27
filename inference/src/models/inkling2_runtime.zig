//! CPU execution of the pinned Inkling 2 Mini schedule: the numerical
//! reference a GPU plan is compared against, written from the forward pass
//! in docs/models/inkling2.md. Weight views and the binding borrow the
//! loaded model; this runtime owns the session and its workspace. A failed
//! step poisons the session.
//!
//! The session is hybrid: application `il` owns attention layer `il` (a
//! window on sliding applications) and recurrent layer `block_count + il`,
//! whose history is the four short convolutions' last rows (card § Session
//! state). The depth residual (`blocks`, the loop's prelude and bank) lives
//! in the workspace, rebuilt from the token on every step, so it is never
//! session state.
const std = @import("std");
const alloc_check = @import("../alloc_check.zig");
const model = @import("inkling2.zig");
const weights = @import("../runtime/weights.zig");
const session = @import("../runtime/session.zig");
const cpu = @import("../backends/cpu/root.zig");
const quant_encode = @import("../quant/encode.zig");
const quant_decode = @import("../quant/decode.zig");
const Tensor = @import("../formats/gguf.zig").Tensor;

pub const Observer = @import("../runtime/observer.zig").Observer;

const E = model.embedding;
const hd = model.head_size;
const taps = model.conv_taps;
/// Blocks a token's residual can hold at once: one opened every
/// `residual_span` stored blocks along the path, the prelude's included.
const max_blocks = model.stored_blocks / model.residual_span;
/// The bank: the previous pass's blocks past the prelude and its first core
/// block, then that pass's final `cur`.
const max_bank = model.core_blocks / model.residual_span;
/// Channels of one application's convolution history, in the order K, V,
/// attention output, FFN output.
pub const conv_channels = 2 * model.kv_width + 2 * E;
const k_offset = 0;
const v_offset = model.kv_width;
const attention_offset = 2 * model.kv_width;
const ffn_offset = 2 * model.kv_width + E;
/// The recurrent layout's matrix region: the session refuses an empty one,
/// so the convolution-only state carries one value, unread.
const unused_matrix = 1;

/// The session layouts: attention `[0, block_count)`, recurrent after.
pub fn layouts(precision: session.Precision, room: usize) [2 * model.block_count]session.Layout {
    var result: [2 * model.block_count]session.Layout = undefined;
    for (0..model.block_count) |il| {
        result[il] = .{ .attention = .{
            .key_row = model.kv_width,
            .value_row = model.kv_width,
            .precision = precision,
            .window = if (model.isGlobal(il)) null else model.window,
            .room = room,
        } };
        result[model.block_count + il] = .{ .recurrent = .{ .history = (taps - 1) * conv_channels, .matrix = unused_matrix } };
    }
    return result;
}

/// A stored block's small F32 tensors, decoded once at init.
const Constants = struct {
    /// Sliding: `[attn_norm]`; global: the q, k and v input norms.
    norms: [3][]f32,
    query_clip: []f32,
    relative_clip: []f32,
    key_conv: []f32,
    value_conv: []f32,
    attention_post_conv: []f32,
    ffn_post_conv: []f32,
    /// One row, or three on a global block; null on block 0.
    attention_query: ?[]f32,
    ffn_query: []f32,
    ffn_norm: []f32,
    router_bias: []f32,
    gate_up_scale: ?[]f32,
    down_scale: ?[]f32,
};

pub const Runtime = struct {
    io: std.Io,
    storage: std.heap.ArenaAllocator,
    state: session.Session,
    view: weights.View,
    binding: model.Binding,
    /// The precision each key and value row is rounded through as it
    /// enters the F32 cache: what a Metal cache of that precision stores.
    /// F32, no rounding, is the reference; the lossy caches' checks set
    /// the others (docs/models/inkling2.md § Metal).
    cache_rounding: session.Precision = .f32,
    /// A Q8_0 rounding then takes each row through F16 too: what a Metal
    /// chunk's attention reads of a Q8_0 cache, its rows expanded to F16.
    /// The tiny check's chunked comparison sets it.
    cache_half_reads: bool = false,
    constants: []Constants,
    token_norm: []f32,
    output_norm: []f32,
    output_query: []f32,
    loop_query: []f32,
    loop_kv_query: []f32,

    cur: []f32,
    /// The layer's input residual (`inpSA`), then the FFN input (`ffn_inp`).
    residual: []f32,
    /// `max_blocks` closed blocks, then a slot for `cur` while a mix reads it.
    states: []f32,
    prelude_cur: []f32,
    /// `max_bank` rows: the bank, then its `cur`.
    bank: []f32,
    /// attn, k, v, mlp.
    messages: []f32,
    /// Three mix outputs or attention inputs; `normed` their normed rows.
    inputs: []f32,
    normed: []f32,
    q: []f32,
    k: []f32,
    v: []f32,
    projected_k: []f32,
    projected_v: []f32,
    relative: []f32,
    gate: []f32,
    sinks: []f32,
    bias_table: []f32,
    dynamic_k: []f32,
    dynamic_v: []f32,
    mixed_out: []f32,
    projected: []f32,
    convolved: []f32,
    router_logits: []f32,
    indices: []u32,
    route_weights: []f32,
    expert_out: []f32,
    expert_scratch: []f32,
    accumulator: []f64,
    control_scratch: []f32,
    mix_scratch: []f64,
    conv_scratch: []f64,
    attention_scratch: []f64,
    padded_logits: []f32,
    row: []f32,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, view: weights.View, binding: model.Binding, capacity: usize, checkpoint: bool, draft: bool) !Runtime {
        // No drafter is bound to this family; a draft request runs without one.
        _ = draft;
        const all = layouts(.f32, 0);
        var state = try session.Session.init(gpa, &all, capacity, checkpoint);
        errdefer state.deinit();
        var storage: std.heap.ArenaAllocator = .init(gpa);
        errdefer storage.deinit();
        const a = storage.allocator();
        // Finish allocations before transferring the arena, whose internal
        // linked-list head can change on each allocation.
        var r: Runtime = undefined;
        inline for (.{ "cur", "residual", "prelude_cur", "projected", "convolved", "expert_out" }) |f| @field(r, f) = try a.alloc(f32, E);
        r.states = try a.alloc(f32, (max_blocks + 1) * E);
        r.bank = try a.alloc(f32, max_bank * E);
        r.messages = try a.alloc(f32, 4 * E);
        r.inputs = try a.alloc(f32, 3 * E);
        r.normed = try a.alloc(f32, 3 * E);
        inline for (.{ "q", "gate", "mixed_out" }) |f| @field(r, f) = try a.alloc(f32, model.query_width);
        inline for (.{ "k", "v", "projected_k", "projected_v" }) |f| @field(r, f) = try a.alloc(f32, model.kv_width);
        r.relative = try a.alloc(f32, model.heads * model.relative_length);
        r.sinks = try a.alloc(f32, model.heads);
        r.bias_table = try a.alloc(f32, model.heads * @max(model.relative_extent, model.window));
        r.dynamic_k = try a.alloc(f32, model.kv_heads * taps);
        r.dynamic_v = try a.alloc(f32, model.kv_heads * taps);
        r.router_logits = try a.alloc(f32, model.experts);
        r.indices = try a.alloc(u32, model.experts_used);
        r.route_weights = try a.alloc(f32, model.experts_used);
        const ff = model.expert_feed_forward;
        // `Ffn.scratchLen` for gate_up `[2·ff, E]` and down `[E, ff]`.
        r.expert_scratch = try a.alloc(f32, 2 * ff + ff + E + cpu.matvecScratch(@max(E, ff)));
        r.accumulator = try a.alloc(f64, E);
        // `Controller.scratchLen` for down `[rank, 2·E]`, gain and forget `[E, rank]`.
        r.control_scratch = try a.alloc(f32, 2 * E + model.gate_rank + E + E + 2 * E);
        r.mix_scratch = try a.alloc(f64, 2 * (max_blocks + 1));
        r.conv_scratch = try a.alloc(f64, model.kv_width);
        r.attention_scratch = try a.alloc(f64, capacity);
        r.padded_logits = try a.alloc(f32, model.padded_vocabulary);
        // The widest row a projection decodes: the controller's `2·E` input.
        r.row = try a.alloc(f32, cpu.matvecScratch(@max(2 * E, model.query_width)));
        r.token_norm = try view.vector(a, binding.token_embedding_norm);
        r.output_norm = try view.vector(a, binding.output_norm);
        r.output_query = try view.vector(a, binding.output_residual_query);
        r.loop_query = try view.vector(a, binding.loop_residual_query);
        r.loop_kv_query = try view.vector(a, binding.loop_kv_residual_query);
        r.constants = try a.alloc(Constants, model.stored_blocks);
        for (&binding.blocks, r.constants) |*block, *c| {
            const opt = struct {
                fn f(v: weights.View, al: std.mem.Allocator, t: ?*const Tensor) !?[]f32 {
                    return if (t) |x| try v.vector(al, x) else null;
                }
            }.f;
            c.* = .{
                .norms = switch (block.input) {
                    .sliding => |s| .{ try view.vector(a, s.norm), &.{}, &.{} },
                    .global => |g| .{ try view.vector(a, g.query_norm), try view.vector(a, g.key_norm), try view.vector(a, g.value_norm) },
                },
                .query_clip = try view.vector(a, block.query_clip),
                .relative_clip = try view.vector(a, block.relative_clip),
                .key_conv = try view.vector(a, block.key_conv),
                .value_conv = try view.vector(a, block.value_conv),
                .attention_post_conv = try view.vector(a, block.attention_post_conv),
                .ffn_post_conv = try view.vector(a, block.ffn_post_conv),
                .attention_query = try opt(view, a, block.attention_residual_query),
                .ffn_query = try view.vector(a, block.ffn_residual_query),
                .ffn_norm = try view.vector(a, block.ffn_norm),
                .router_bias = try view.vector(a, block.router_bias),
                .gate_up_scale = try opt(view, a, block.gate_up_scale),
                .down_scale = try opt(view, a, block.down_scale),
            };
        }
        r.storage = storage;
        r.cache_rounding = .f32;
        r.cache_half_reads = false;
        r.state = state;
        r.view = view;
        r.io = io;
        r.binding = binding;
        return r;
    }

    pub fn deinit(self: *Runtime) void {
        self.state.deinit();
        self.storage.deinit();
        self.* = undefined;
    }

    pub fn reset(self: *Runtime) void {
        self.state.reset();
    }

    fn mm(self: *Runtime, tensor: *const Tensor, input: []const f32, output: []f32) !void {
        try cpu.matvec(self.io, try self.view.matrix(tensor), input, output, self.row);
    }

    fn norm(input: []const f32, output: []f32, weight: []const f32) !void {
        if (weight.len != output.len) return error.InvalidShape;
        try cpu.rmsNorm(input, output, model.rms_epsilon);
        for (output, weight) |*x, w| x.* *= w;
    }

    fn rowOf(values: []f32, index: usize) []f32 {
        return values[index * E ..][0..E];
    }

    /// `mix(states[0..count] ∪ {cur}, queries)` into `output`.
    fn mixWithCur(self: *Runtime, count: usize, queries: []const f32, output: []f32) !void {
        @memcpy(rowOf(self.states, count), self.cur);
        try cpu.depth.mix(self.states[0 .. (count + 1) * E], queries, E, model.rms_epsilon, output, self.mix_scratch);
    }

    fn controller(self: *Runtime, loop: model.Loop, module: usize) !cpu.depth.Controller {
        return .{
            .down = try self.view.matrix(loop.down[module]),
            .gain = try self.view.matrix(loop.gain[module]),
            .forget = try self.view.matrix(loop.forget[module]),
        };
    }

    /// One application's convolution history for `count` channels from `first`.
    fn history(self: *Runtime, il: usize, first: usize, count: usize) []f32 {
        return self.state.layers[model.block_count + il].recurrent.history[first * (taps - 1) ..][0 .. count * (taps - 1)];
    }

    /// `logits` may be null for non-final prefill tokens, avoiding the head;
    /// otherwise it holds exactly the vocabulary. A successful step commits
    /// one token; an error after `begin` requires reset.
    pub fn step(self: *Runtime, token: u32, logits: ?[]f32, observer: ?Observer) !void {
        if (token >= model.vocabulary) return error.InvalidTokenId;
        if (logits) |out| if (out.len != model.vocabulary) return error.InvalidShape;
        try self.state.begin();
        errdefer self.state.fail();
        try self.view.row(self.binding.token_embedding, token, self.cur);
        try norm(self.cur, self.cur, self.token_norm);
        var count: usize = 0;
        var prelude_count: usize = 0;
        var bank_count: usize = 0;
        for (0..model.block_count) |il| {
            const ip = model.storedBlock(il);
            const block = &self.binding.blocks[ip];
            const c = &self.constants[ip];
            var looped = false;
            if (model.loops > 1 and ip >= model.core_start and ip < model.core_end) {
                const pass = (il - model.core_start) / model.core_blocks;
                if (ip == model.core_start) {
                    if (pass == 0) {
                        prelude_count = count;
                        @memcpy(self.prelude_cur, self.cur);
                    } else {
                        bank_count = count - prelude_count - 1;
                        @memcpy(self.bank[0 .. bank_count * E], self.states[(prelude_count + 1) * E ..][0 .. bank_count * E]);
                        @memcpy(rowOf(self.bank, bank_count), self.cur);
                        count = prelude_count;
                        @memcpy(self.cur, self.prelude_cur);
                    }
                }
                if (pass > 0) {
                    const at = 2 * (ip - model.core_start) * E;
                    const bank = self.bank[0 .. (bank_count + 1) * E];
                    try cpu.depth.mix(bank, self.loop_query[at..][0 .. 2 * E], E, model.rms_epsilon, self.inputs[0 .. 2 * E], self.mix_scratch);
                    @memcpy(rowOf(self.messages, 0), rowOf(self.inputs, 0));
                    @memcpy(rowOf(self.messages, 3), rowOf(self.inputs, 1));
                    try cpu.depth.mix(bank, self.loop_kv_query[at..][0 .. 2 * E], E, model.rms_epsilon, self.inputs[0 .. 2 * E], self.mix_scratch);
                    @memcpy(rowOf(self.messages, 1), rowOf(self.inputs, 0));
                    @memcpy(rowOf(self.messages, 2), rowOf(self.inputs, 1));
                    looped = true;
                }
            }
            @memcpy(self.residual, self.cur);
            try self.attention(il, block, c, count, looped);
            if (ip % model.residual_span == 0) {
                @memcpy(rowOf(self.states, count), self.residual);
                count += 1;
                @memcpy(self.cur, self.projected);
            } else {
                for (self.cur, self.residual, self.projected) |*x, base, add| x.* = base + add;
            }
            @memcpy(self.residual, self.cur);
            try self.feedForward(il, block, c, count, looped);
            for (self.cur, self.residual, self.projected) |*x, base, add| {
                x.* = base + add;
                if (!std.math.isFinite(x.*)) return error.NonFiniteResult;
            }
            if (observer) |o| {
                if (o.check) |check| try check(o.context);
                if (o.layer) |report| try report(o.context, il, self.cur);
            }
        }
        if (logits) |out| {
            const y = rowOf(self.inputs, 0);
            try self.mixWithCur(count, self.output_query, y);
            try norm(y, y, self.output_norm);
            try self.mm(self.binding.output, y, self.padded_logits);
            for (out, self.padded_logits[0..model.vocabulary]) |*x, z| {
                x.* = model.final_softcap * std.math.tanh(z / model.final_softcap);
                if (!std.math.isFinite(x.*)) return error.NonFiniteResult;
            }
        }
        try self.state.commit();
    }

    /// The attention of application `il` over `cur` and the `count` closed
    /// blocks, into `projected`; `looped` applies the pass's messages.
    fn attention(self: *Runtime, il: usize, block: *const model.Block, c: *const Constants, count: usize, looped: bool) !void {
        const global = block.kind() == .global;
        if (c.attention_query) |query| {
            const rows = query.len / E;
            try self.mixWithCur(count, query, self.inputs[0 .. rows * E]);
            if (rows == 1) for (1..3) |j| @memcpy(rowOf(self.inputs, j), rowOf(self.inputs, 0));
        } else for (0..3) |j| @memcpy(rowOf(self.inputs, j), self.cur);
        if (looped) {
            const loop = block.loop orelse return error.InvalidShape;
            for (0..3) |j| {
                const x = rowOf(self.inputs, j);
                try cpu.depth.control(self.io, try self.controller(loop, j), x, rowOf(self.messages, j), model.rms_epsilon, x, self.control_scratch);
            }
        }
        // Which normed row feeds q, k, and v.
        var source: [3]usize = .{ 0, 1, 2 };
        if (global) {
            for (0..3) |j| try norm(rowOf(self.inputs, j), rowOf(self.normed, j), c.norms[j]);
        } else if (looped) {
            for (0..3) |j| try norm(rowOf(self.inputs, j), rowOf(self.normed, j), c.norms[0]);
        } else {
            try norm(rowOf(self.inputs, 0), rowOf(self.normed, 0), c.norms[0]);
            source = .{ 0, 0, 0 };
        }
        const in_q = rowOf(self.normed, source[0]);
        const in_k = rowOf(self.normed, source[1]);
        const in_v = rowOf(self.normed, source[2]);
        try self.mm(block.query, in_q, self.q);
        try self.mm(block.key, in_k, self.projected_k);
        try self.mm(block.value, in_v, self.projected_v);
        try self.mm(block.relative, in_q, self.relative);
        try self.mm(block.gate, in_q, self.gate);
        try self.mm(block.sink_projection, in_q, self.sinks);
        var dynamic_k: ?cpu.recurrent.DynamicTaps = null;
        var dynamic_v: ?cpu.recurrent.DynamicTaps = null;
        switch (block.input) {
            .sliding => {},
            .global => |g| {
                try self.mm(g.key_conv_projection, in_k, self.dynamic_k);
                try self.mm(g.value_conv_projection, in_v, self.dynamic_v);
                dynamic_k = .{ .taps = self.dynamic_k, .heads = model.kv_heads };
                dynamic_v = .{ .taps = self.dynamic_v, .heads = model.kv_heads };
            },
        }
        const kw = model.kv_width;
        try cpu.recurrent.shortConvolution(self.projected_k, c.key_conv, dynamic_k, self.history(il, k_offset, kw), self.k, taps, self.conv_scratch);
        try cpu.recurrent.shortConvolution(self.projected_v, c.value_conv, dynamic_v, self.history(il, v_offset, kw), self.v, taps, self.conv_scratch);
        const position = self.state.position;
        const lambda: f32 = if (global) cpu.logPositionScale(position, model.log_scale_floor, model.log_scale_alpha) else 1;
        const rl = model.relative_length;
        // The bias table's columns: the window on a sliding application
        // (the binder holds the tensor to it), the extent on a global one.
        const extent: usize = if (global) model.relative_extent else model.window;
        const table = self.bias_table[0 .. model.heads * extent];
        for (0..model.heads) |h| {
            const head = self.q[h * hd ..][0..hd];
            try cpu.l2Clip(head, c.query_clip[h * hd ..][0..hd], model.qk_clip, head);
            const r = self.relative[h * rl ..][0..rl];
            try cpu.l2Clip(r, c.relative_clip[h * rl ..][0..rl], model.relative_clip, r);
            if (global) {
                for (head) |*x| x.* *= lambda;
                for (r) |*x| x.* *= lambda;
            }
            try self.mm(block.relative_bias, r, table[h * extent ..][0..extent]);
        }
        for (0..model.kv_heads) |h| {
            const key = self.k[h * hd ..][0..hd];
            try cpu.l2Clip(key, null, model.qk_clip, key);
            const value = self.v[h * hd ..][0..hd];
            try cpu.rmsNorm(value, value, model.rms_epsilon);
        }
        try roundRow(self.cache_rounding, self.cache_half_reads, self.k);
        try roundRow(self.cache_rounding, self.cache_half_reads, self.v);
        const cache = self.state.layers[il].attention;
        @memcpy(cache.keys.floats(position, 1), self.k);
        @memcpy(cache.values.floats(position, 1), self.v);
        // A sliding application sees the last `window` rows, its own included.
        const first = if (!global and position + 1 > model.window) position + 1 - model.window else 0;
        const visible = position + 1 - first;
        try cpu.attention.apply(.{
            .query_heads = model.heads,
            .kv_heads = model.kv_heads,
            .key_width = hd,
            .value_width = hd,
            .tokens = visible,
            .visible_tokens = visible,
            .scale = 1.0 / @sqrt(@as(f32, hd)),
            .queries = self.q,
            .keys = cache.keys.floats(first, visible),
            .values = cache.values.floats(first, visible),
            .bias = .{ .table = table, .extent = extent },
            .sinks = self.sinks,
        }, self.mixed_out, self.attention_scratch);
        for (self.mixed_out, self.gate) |*o, g| o.* *= cpu.sigmoid(g);
        try self.mm(block.output, self.mixed_out, self.convolved);
        try cpu.recurrent.shortConvolution(self.convolved, c.attention_post_conv, null, self.history(il, attention_offset, E), self.projected, taps, &.{});
    }

    /// `values` as a cache of `precision` holds them: F16 by round to
    /// nearest, Q8_0 through `quant_encode.q8_0Row` and back, then through
    /// F16 when `half_reads`.
    fn roundRow(precision: session.Precision, half_reads: bool, values: []f32) !void {
        switch (precision) {
            .f32 => {},
            .f16 => for (values) |*x| {
                x.* = @as(f16, @floatCast(x.*));
            },
            .q8_0 => {
                var bytes: [model.kv_width / quant_encode.q8_0_block * quant_encode.q8_0_block_bytes]u8 = undefined;
                try quant_encode.q8_0Row(values, &bytes);
                try quant_decode.row(8, &bytes, values);
                if (half_reads) for (values) |*x| {
                    x.* = @as(f16, @floatCast(x.*));
                };
            },
        }
    }

    /// The experts of application `il` over the mixed residual, into
    /// `projected`, ready to add to `residual`.
    fn feedForward(self: *Runtime, il: usize, block: *const model.Block, c: *const Constants, count: usize, looped: bool) !void {
        const f = rowOf(self.inputs, 0);
        try self.mixWithCur(count, c.ffn_query, f);
        if (looped) {
            const loop = block.loop orelse return error.InvalidShape;
            try cpu.depth.control(self.io, try self.controller(loop, 3), f, rowOf(self.messages, 3), model.rms_epsilon, f, self.control_scratch);
        }
        try norm(f, f, c.ffn_norm);
        try self.mm(block.router, f, self.router_logits);
        try cpu.experts.routeBiased(self.router_logits, c.router_bias, model.expert_weights_scale, self.indices, self.route_weights);
        const spec: cpu.experts.Ffn = .{
            .gate_up = try self.view.expertMatrix(block.gate_up_experts),
            .down = try self.view.expertMatrix(block.down_experts),
            .gate_up_scale = c.gate_up_scale,
            .down_scale = c.down_scale,
            .activation = .{ .situ = .{ .beta = model.situ_beta, .linear_beta = model.situ_linear_beta } },
        };
        try cpu.experts.ffn(self.io, spec, f, self.indices, self.route_weights, self.expert_out, self.expert_scratch, self.accumulator);
        try cpu.recurrent.shortConvolution(self.expert_out, c.ffn_post_conv, null, self.history(il, ffn_offset, E), self.projected, taps, &.{});
    }
};

test "the session layout: a window on sliding applications, a convolution history on each" {
    const all = layouts(.f32, 0);
    var sliding: usize = 0;
    for (all[0..model.block_count], 0..) |layout, il| {
        try std.testing.expectEqual(model.isGlobal(il), layout.attention.window == null);
        if (layout.attention.window != null) sliding += 1;
    }
    try std.testing.expectEqual(model.block_count - model.block_count / model.attention_period, sliding);
    for (all[model.block_count..]) |layout| try std.testing.expectEqual(3 * conv_channels, layout.recurrent.history);
    var state = try session.Session.init(std.testing.allocator, &all, 64, true);
    defer state.deinit();
    try std.testing.expect(state.hasRecurrent());
    try std.testing.expectError(error.RecurrentStateNotRewindable, state.rewindTo(0));
}

test "the session frees on every failed allocation" {
    try alloc_check.checkAll(std.testing.allocator, struct {
        fn f(gpa: std.mem.Allocator) !void {
            const all = layouts(.f32, 0);
            var state = try session.Session.init(gpa, &all, 4, true);
            state.deinit();
        }
    }.f, .{});
}
