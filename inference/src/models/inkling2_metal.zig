//! Inkling 2 on Metal: the stepped plan, one command buffer per token over
//! the 48 applications of the 24 stored blocks. It records, on the GPU,
//! what `inkling2_runtime.Runtime.step` computes (that file is the
//! reference and names each stage); only the logits, or the selection
//! over them, are read back.
//!
//! The session is the CPU runtime's layout (`inkling2_runtime.layouts`),
//! wrapped whole as one buffer: each application's KV rows and its
//! convolution history live there, so snapshots, rewinds, and resets are
//! the session's. The residual bookkeeping (which blocks are open, the
//! core's bank) is host integers, the same on every token. A prompt is
//! stepped token by token, and the observer's `experts` hook is not
//! called; docs/models/inkling2.md § Metal says what the plan covers.
const std = @import("std");
const model = @import("inkling2.zig");
const runtime = @import("inkling2_runtime.zig");
const weights = @import("../runtime/weights.zig");
const session = @import("../runtime/session.zig");
const sampling = @import("../sampling/root.zig");
const metal = @import("../backends/metal/root.zig");
const chat_plan = @import("chat_plan.zig");
const cpu = @import("../backends/cpu/root.zig");
const Tensor = @import("../formats/gguf.zig").Tensor;
const Buffer = metal.Buffer;

pub const Observer = @import("../runtime/observer.zig").Observer;

pub const vocabulary = model.vocabulary;

const E = model.embedding;
const hd = model.head_size;
const taps = model.conv_taps;
const ff = model.expert_feed_forward;
const k_used = model.experts_used;
const rl = model.relative_length;
/// Closed blocks a residual holds at once, and the core's bank
/// (`inkling2_runtime`'s `max_blocks` and `max_bank`).
const max_blocks = model.stored_blocks / model.residual_span;
const max_bank = model.core_blocks / model.residual_span;
/// The widest bias table: the window on a sliding application, the extent
/// on a global one.
const max_extent = @max(model.relative_extent, model.window);
/// Rows of `messages`: the attention input's, the experts', then K's and
/// V's, so each of the two bank mixes writes two adjacent rows.
const message_row = [4]usize{ 0, 2, 3, 1 };

/// A stored block's small F32 tensors, one GPU buffer each.
const Constants = struct {
    /// Sliding: `[attn_norm]`; global: the q, k, and v input norms.
    norms: [3]?Buffer,
    query_clip: Buffer,
    relative_clip: Buffer,
    key_conv: Buffer,
    value_conv: Buffer,
    attention_post_conv: Buffer,
    ffn_post_conv: Buffer,
    attention_query: ?Buffer,
    /// Rows of `attention_query`: 3 on a global block, 1 on a sliding one.
    attention_queries: usize,
    ffn_query: Buffer,
    ffn_norm: Buffer,
    router_bias: Buffer,
    gate_up_scale: ?Buffer,
    down_scale: ?Buffer,
};

const Weight = struct { buffer: Buffer, matrix: cpu.Matrix };
const ExpertWeight = struct { buffer: Buffer, tensor: cpu.ExpertMatrix };

pub const Plan = struct {
    alloc: std.mem.Allocator,
    backend: *metal.Backend,
    view: weights.View,
    binding: model.Binding,
    state: session.Session,
    state_buffer: Buffer,
    constants: []Constants,
    token_norm: Buffer,
    output_norm: Buffer,
    output_query: Buffer,
    loop_query: Buffer,
    loop_kv_query: Buffer,
    /// A row of ones: the per-head value norm has no weight.
    ones: Buffer, // hd

    // One token's activations (F32 counts in comments).
    cur: Buffer, // E
    projected: Buffer, // E
    convolved: Buffer, // E
    expert_out: Buffer, // E
    prelude_cur: Buffer, // E
    /// `max_blocks` closed blocks, then a row for `cur` while a mix reads it.
    states: Buffer, // (max_blocks + 1) × E
    bank: Buffer, // max_bank × E
    /// The pass's messages, in `message_row` order.
    messages: Buffer, // 4 × E
    inputs: Buffer, // 3 × E
    normed: Buffer, // 3 × E
    q: Buffer, // query_width
    gate: Buffer, // query_width
    mixed_out: Buffer, // query_width
    projected_k: Buffer, // kv_width
    projected_v: Buffer, // kv_width
    /// The current row's keys and values before they are packed, when the
    /// cache is not F32 (an F32 cache takes them in place).
    k: Buffer, // kv_width
    v: Buffer, // kv_width
    relative: Buffer, // heads × rl
    sinks: Buffer, // heads
    bias_table: Buffer, // heads × max_extent
    dynamic_k: Buffer, // kv_heads × taps
    dynamic_v: Buffer, // kv_heads × taps
    partials: Buffer,
    router_logits: Buffer, // experts
    indices: Buffer, // k_used u32
    route_weights: Buffer, // k_used
    gate_up: Buffer, // k_used × 2·ff
    hidden: Buffer, // k_used × ff
    down: Buffer, // k_used × E
    joined: Buffer, // 2 × E
    rank: Buffer, // gate_rank
    gains: Buffer, // E
    forgets: Buffer, // E
    /// The head's padded rows; the first `vocabulary`, soft-capped, are the logits.
    padded_logits: Buffer, // padded_vocabulary
    readout: chat_plan.Readout(vocabulary, 1),

    /// `chunk` is accepted for the engine's signature; this plan steps a
    /// prompt. `kv` is every application's cache precision. No drafter is
    /// bound to this family, so `draft` changes nothing.
    pub fn init(alloc: std.mem.Allocator, backend: *metal.Backend, view: weights.View, binding: model.Binding, capacity: usize, chunk: usize, kv: session.Precision, checkpoint: bool, draft: bool) !Plan {
        _ = draft;
        if (chunk == 0) return error.InvalidShape;
        const all = runtime.layouts(kv, 0);
        var state = try session.Session.init(alloc, &all, capacity, checkpoint);
        errdefer state.deinit();
        const constants = try alloc.alloc(Constants, model.stored_blocks);
        errdefer alloc.free(constants);
        var self: Plan = undefined;
        self.alloc = alloc;
        self.backend = backend;
        self.view = view;
        self.binding = binding;
        self.state = state;
        self.constants = constants;
        var created: metal.Created = .init(alloc, backend);
        defer created.deinit();
        errdefer created.releaseAll();
        self.state_buffer = try backend.wrap(state.memory);
        errdefer backend.unwrap(state.memory);
        for (&binding.blocks, constants) |*block, *c| {
            const query = block.attention_residual_query;
            c.* = .{
                .norms = switch (block.input) {
                    .sliding => |s| .{ try self.constant(&created, s.norm), null, null },
                    .global => |g| .{ try self.constant(&created, g.query_norm), try self.constant(&created, g.key_norm), try self.constant(&created, g.value_norm) },
                },
                .query_clip = try self.constant(&created, block.query_clip),
                .relative_clip = try self.constant(&created, block.relative_clip),
                .key_conv = try self.constant(&created, block.key_conv),
                .value_conv = try self.constant(&created, block.value_conv),
                .attention_post_conv = try self.constant(&created, block.attention_post_conv),
                .ffn_post_conv = try self.constant(&created, block.ffn_post_conv),
                .attention_query = if (query) |t| try self.constant(&created, t) else null,
                .attention_queries = if (query) |t| @intCast(t.elements / E) else 0,
                .ffn_query = try self.constant(&created, block.ffn_residual_query),
                .ffn_norm = try self.constant(&created, block.ffn_norm),
                .router_bias = try self.constant(&created, block.router_bias),
                .gate_up_scale = if (block.gate_up_scale) |t| try self.constant(&created, t) else null,
                .down_scale = if (block.down_scale) |t| try self.constant(&created, t) else null,
            };
            if (query != null and c.attention_queries * E != query.?.elements) return error.InvalidShape;
        }
        self.token_norm = try self.constant(&created, binding.token_embedding_norm);
        self.output_norm = try self.constant(&created, binding.output_norm);
        self.output_query = try self.constant(&created, binding.output_residual_query);
        self.loop_query = try self.constant(&created, binding.loop_residual_query);
        self.loop_kv_query = try self.constant(&created, binding.loop_kv_residual_query);
        self.ones = try created.create(hd * 4);
        @memset(self.ones.floats(), 1);
        inline for (.{ "cur", "projected", "convolved", "expert_out", "prelude_cur", "gains", "forgets" }) |f| @field(self, f) = try created.create(E * 4);
        self.states = try created.create((max_blocks + 1) * E * 4);
        self.bank = try created.create(max_bank * E * 4);
        self.messages = try created.create(4 * E * 4);
        self.inputs = try created.create(3 * E * 4);
        self.normed = try created.create(3 * E * 4);
        self.joined = try created.create(2 * E * 4);
        inline for (.{ "q", "gate", "mixed_out" }) |f| @field(self, f) = try created.create(model.query_width * 4);
        inline for (.{ "projected_k", "projected_v", "k", "v" }) |f| @field(self, f) = try created.create(model.kv_width * 4);
        self.relative = try created.create(model.heads * rl * 4);
        self.sinks = try created.create(model.heads * 4);
        self.bias_table = try created.create(model.heads * max_extent * 4);
        self.dynamic_k = try created.create(model.kv_heads * taps * 4);
        self.dynamic_v = try created.create(model.kv_heads * taps * 4);
        self.partials = try created.create(metal.Backend.attentionDecodePartials(model.heads, hd) * 4);
        self.router_logits = try created.create(model.experts * 4);
        self.indices = try created.create(k_used * 4);
        self.route_weights = try created.create(k_used * 4);
        self.gate_up = try created.create(k_used * 2 * ff * 4);
        self.hidden = try created.create(k_used * ff * 4);
        self.down = try created.create(k_used * E * 4);
        self.rank = try created.create(model.gate_rank * 4);
        self.padded_logits = try created.create(model.padded_vocabulary * 4);
        self.readout = try .init(&created, false);
        return self;
    }

    pub fn deinit(self: *Plan) void {
        // GPU buffers are released with the backend; the session and the
        // constants table are ours.
        self.alloc.free(self.constants);
        self.backend.unwrap(self.state.memory);
        self.state.deinit();
        self.* = undefined;
    }

    pub fn reset(self: *Plan) void {
        self.state.reset();
    }

    /// Small F32 tensor copied into its own GPU buffer once at init.
    fn constant(self: *Plan, created: *metal.Created, tensor: *const Tensor) !Buffer {
        const values = try self.view.vector(self.alloc, tensor);
        defer self.alloc.free(values);
        if (values.len == 0) return error.InvalidShape;
        const buffer = try created.create(values.len * 4);
        @memcpy(buffer.floats(), values);
        return buffer;
    }
    /// A matrix wrapped in place (the backend caches by address).
    fn weight(self: *Plan, tensor: *const Tensor) !Weight {
        const matrix = try self.view.matrix(tensor);
        return .{ .buffer = try self.backend.wrap(matrix.bytes), .matrix = matrix };
    }
    fn expertWeight(self: *Plan, tensor: *const Tensor) !ExpertWeight {
        const matrix = try self.view.expertMatrix(tensor);
        return .{ .buffer = try self.backend.wrap(matrix.bytes), .tensor = matrix };
    }
    fn mm(self: *Plan, tensor: *const Tensor, input: Buffer, output: Buffer) !void {
        const w = try self.weight(tensor);
        try self.backend.matvec(w.buffer, w.matrix, input, output);
    }
    /// The GPU range of one session region.
    fn stateSlice(self: *Plan, region: []const u8) Buffer {
        return self.state_buffer.slice(self.state.offsetOf(region), region.len);
    }
    fn row(buffer: Buffer, index: usize) Buffer {
        return buffer.slice(index * E * 4, E * 4);
    }
    fn rows(buffer: Buffer, first: usize, count: usize) Buffer {
        return buffer.slice(first * E * 4, count * E * 4);
    }
    /// Application `il`'s convolution history for `count` channels from `first`.
    fn history(self: *Plan, il: usize, first: usize, count: usize) Buffer {
        const h = self.state.layers[model.block_count + il].recurrent.history[first * (taps - 1) ..][0 .. count * (taps - 1)];
        return self.stateSlice(std.mem.sliceAsBytes(h));
    }

    /// `cpu.recurrent.shortConvolution` of one row, its history then
    /// advanced past it.
    fn convolve(self: *Plan, il: usize, first: usize, channels: usize, input: Buffer, w: Buffer, dynamic: ?metal.Backend.DynamicTaps, output: Buffer) !void {
        const h = self.history(il, first, channels);
        try self.backend.shortConvolution(h, input, w, dynamic, output, channels, taps, 1, channels, channels);
        try self.backend.convolutionHistory(h, input, channels, taps, 1, channels);
    }

    /// `mix(states[0..count] ∪ {cur}, queries)` into `output` (`queries` rows).
    fn mixWithCur(self: *Plan, count: usize, query: Buffer, queries: usize, output: Buffer) !void {
        try self.backend.copy(row(self.states, count), self.cur, E);
        try self.backend.depthMix(self.states, query, output, E, count + 1, queries, 1, (max_blocks + 1) * E, queries * E, model.rms_epsilon);
    }

    /// `cpu.depth.control` of `loop`'s `module` on `x` with its message, in place.
    fn control(self: *Plan, loop: model.Loop, module: usize, x: Buffer) !void {
        const b = self.backend;
        const message = row(self.messages, message_row[module]);
        try b.controlJoin(x, message, self.joined, E, 1, E, E, 2 * E, model.rms_epsilon);
        try self.mm(loop.down[module], self.joined, self.rank);
        try self.mm(loop.gain[module], self.rank, self.gains);
        try self.mm(loop.forget[module], self.rank, self.forgets);
        try b.controlBlend(x, message, self.gains, self.forgets, x, E, 1, E, E, E, E);
    }

    /// `relative_bias` (`[extent, rl]`) times every head's relative row
    /// into the bias table, at most `matvec_rows_max` heads a dispatch.
    fn biasTable(self: *Plan, tensor: *const Tensor, extent: usize) !void {
        const w = try self.weight(tensor);
        if (w.matrix.rows != extent or w.matrix.columns != rl) return error.InvalidShape;
        var h: usize = 0;
        while (h < model.heads) {
            const n: usize = @min(metal.Backend.matvec_rows_max, model.heads - h);
            const input = self.relative.slice(h * rl * 4, n * rl * 4);
            const output = self.bias_table.slice(h * extent * 4, n * extent * 4);
            if (n == 1) try self.backend.matvec(w.buffer, w.matrix, input, output) else try self.backend.matvecRows(w.buffer, w.matrix, input, rl, output, extent, n);
            h += n;
        }
    }

    /// Records and runs one token. `logits` receives the vocabulary scores,
    /// `greedy` the argmax, and `topk` the best `TopK.capacity` logits, each
    /// when non-null (the selections on the GPU). An observer's `check` runs
    /// between recorded applications; its `layer` callback commits after
    /// every application (traces only). A successful step commits one
    /// token; an error after `begin` requires reset.
    pub fn step(self: *Plan, token: u32, logits: ?[]f32, greedy: ?*u32, topk: ?*sampling.TopK, penalties: ?sampling.Penalties, observer: ?Observer) !void {
        if (token >= vocabulary) return error.InvalidTokenId;
        if (logits) |out| if (out.len != vocabulary) return error.InvalidShape;
        if (topk) |top| if (!std.math.isFinite(top.temperature) or top.temperature <= 0) return error.InvalidShape;
        if (penalties) |p| try self.readout.syncPenalties(p);
        try self.state.begin();
        errdefer self.state.fail();
        const b = self.backend;
        try b.begin();
        errdefer b.discard();
        const embedding = try self.weight(self.binding.token_embedding);
        try b.embed(embedding.buffer, embedding.matrix, token, self.cur);
        try b.rmsNorm(self.cur, self.token_norm, self.cur, .{ .rows = 1, .width = E, .in_stride = E, .out_stride = E });
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
                        try b.copy(self.prelude_cur, self.cur, E);
                    } else {
                        bank_count = count - prelude_count - 1;
                        if (bank_count > 0) try b.copy(rows(self.bank, 0, bank_count), rows(self.states, prelude_count + 1, bank_count), bank_count * E);
                        try b.copy(row(self.bank, bank_count), self.cur, E);
                        count = prelude_count;
                        try b.copy(self.cur, self.prelude_cur, E);
                    }
                }
                if (pass > 0) {
                    const at = 2 * (ip - model.core_start) * E * 4;
                    const bank = rows(self.bank, 0, bank_count + 1);
                    try b.depthMix(bank, self.loop_query.slice(at, 2 * E * 4), rows(self.messages, 0, 2), E, bank_count + 1, 2, 1, (bank_count + 1) * E, 2 * E, model.rms_epsilon);
                    try b.depthMix(bank, self.loop_kv_query.slice(at, 2 * E * 4), rows(self.messages, 2, 2), E, bank_count + 1, 2, 1, (bank_count + 1) * E, 2 * E, model.rms_epsilon);
                    looped = true;
                }
            }
            try self.attention(il, block, c, count, looped);
            if (ip % model.residual_span == 0) {
                try b.copy(row(self.states, count), self.cur, E);
                count += 1;
                try b.copy(self.cur, self.projected, E);
            } else try b.add(self.cur, self.projected, E);
            try self.feedForward(il, block, c, count, looped);
            try b.add(self.cur, self.projected, E);
            if (observer) |o| {
                if (o.check) |check| try check(o.context);
                if (o.layer) |report| {
                    try b.commit();
                    try finite(self.cur);
                    try report(o.context, il, self.cur.floats()[0..E]);
                    try b.begin();
                }
            }
        }
        const outputs = logits != null or greedy != null or topk != null;
        if (outputs) {
            const y = row(self.inputs, 0);
            try self.mixWithCur(count, self.output_query, 1, y);
            try b.rmsNorm(y, self.output_norm, y, .{ .rows = 1, .width = E, .in_stride = E, .out_stride = E });
            try self.mm(self.binding.output, y, self.padded_logits);
            try b.copy(self.readout.logits, self.padded_logits, vocabulary);
            try b.softcap(self.readout.logits, vocabulary, model.final_softcap);
            try self.readout.select(b, greedy != null, topk, penalties);
        }
        try b.commit();
        try finite(self.cur);
        try self.readout.readOutputs(logits, greedy, topk, penalties != null);
        try self.state.commit();
    }

    /// The CPU forward fails a step whose residual leaves the finite range;
    /// the plan checks the last one, which any non-finite value reaches.
    fn finite(buffer: Buffer) !void {
        for (buffer.floats()[0..E]) |x| if (!std.math.isFinite(x)) return error.NonFiniteResult;
    }

    /// The attention of application `il` over `cur` and the `count` closed
    /// blocks, into `projected`; `looped` applies the pass's messages.
    fn attention(self: *Plan, il: usize, block: *const model.Block, c: *const Constants, count: usize, looped: bool) !void {
        const b = self.backend;
        const global = block.kind() == .global;
        if (c.attention_query) |query| {
            try self.mixWithCur(count, query, c.attention_queries, rows(self.inputs, 0, c.attention_queries));
            if (c.attention_queries == 1) for (1..3) |j| try b.copy(row(self.inputs, j), row(self.inputs, 0), E);
        } else for (0..3) |j| try b.copy(row(self.inputs, j), self.cur, E);
        if (looped) {
            const loop = block.loop orelse return error.InvalidShape;
            for (0..3) |j| try self.control(loop, j, row(self.inputs, j));
        }
        const one: metal.Backend.Norm = .{ .rows = 1, .width = E, .in_stride = E, .out_stride = E };
        // Which normed row feeds q, k, and v.
        var source: [3]usize = .{ 0, 1, 2 };
        if (global) {
            for (0..3) |j| try b.rmsNorm(row(self.inputs, j), c.norms[j] orelse return error.InvalidShape, row(self.normed, j), one);
        } else if (looped) {
            try b.rmsNorm(self.inputs, c.norms[0].?, self.normed, .{ .rows = 3, .width = E, .in_stride = E, .out_stride = E });
        } else {
            try b.rmsNorm(row(self.inputs, 0), c.norms[0].?, row(self.normed, 0), one);
            source = .{ 0, 0, 0 };
        }
        const in_q = row(self.normed, source[0]);
        const in_k = row(self.normed, source[1]);
        const in_v = row(self.normed, source[2]);
        try self.mm(block.query, in_q, self.q);
        try self.mm(block.key, in_k, self.projected_k);
        try self.mm(block.value, in_v, self.projected_v);
        try self.mm(block.relative, in_q, self.relative);
        try self.mm(block.gate, in_q, self.gate);
        try self.mm(block.sink_projection, in_q, self.sinks);
        var dynamic_k: ?metal.Backend.DynamicTaps = null;
        var dynamic_v: ?metal.Backend.DynamicTaps = null;
        switch (block.input) {
            .sliding => {},
            .global => |g| {
                try self.mm(g.key_conv_projection, in_k, self.dynamic_k);
                try self.mm(g.value_conv_projection, in_v, self.dynamic_v);
                dynamic_k = .{ .taps = self.dynamic_k, .heads = model.kv_heads, .stride = model.kv_heads * taps };
                dynamic_v = .{ .taps = self.dynamic_v, .heads = model.kv_heads, .stride = model.kv_heads * taps };
            },
        }
        const kw = model.kv_width;
        const cache = self.state.layers[il].attention;
        const position = self.state.position;
        const precision = cache.keys.precision;
        const k_slot = self.stateSlice(cache.keys.range(position, 1));
        const v_slot = self.stateSlice(cache.values.range(position, 1));
        // An F32 cache takes the convolved rows directly; F16 and Q8_0 take
        // them through F32 rows and one pack dispatch.
        const k_row = if (precision == .f32) k_slot else self.k;
        const v_row = if (precision == .f32) v_slot else self.v;
        try self.convolve(il, 0, kw, self.projected_k, c.key_conv, dynamic_k, k_row);
        try self.convolve(il, kw, kw, self.projected_v, c.value_conv, dynamic_v, v_row);
        const log_scale: ?metal.Backend.LogScale = if (global) .{ .position = position, .floor = model.log_scale_floor, .alpha = model.log_scale_alpha } else null;
        try b.l2ClipRows(self.q, c.query_clip, model.qk_clip, log_scale, model.heads, hd, 1, model.query_width);
        try b.l2ClipRows(self.relative, c.relative_clip, model.relative_clip, log_scale, model.heads, rl, 1, model.heads * rl);
        // The bias table's columns: the window on a sliding application
        // (the binder holds the tensor to it), the extent on a global one.
        const extent: usize = if (global) model.relative_extent else model.window;
        try self.biasTable(block.relative_bias, extent);
        try b.l2ClipRows(k_row, null, model.qk_clip, null, model.kv_heads, hd, 1, kw);
        try b.rmsNorm(v_row, self.ones, v_row, .{ .rows = model.kv_heads, .width = hd, .in_stride = hd, .out_stride = hd });
        if (precision != .f32) try b.storeRows(&.{ .{ .dst = k_slot, .src = k_row, .count = kw }, .{ .dst = v_slot, .src = v_row, .count = kw } }, precision);
        // A sliding application sees the last `window` rows, its own included.
        const first = if (!global and position + 1 > model.window) position + 1 - model.window else 0;
        const visible = position + 1 - first;
        try b.attentionDecode(self.stateSlice(cache.keys.range(first, visible)), self.stateSlice(cache.values.range(first, visible)), self.q, self.partials, self.mixed_out, .{
            .query_heads = model.heads,
            .kv_heads = model.kv_heads,
            .key_width = hd,
            .value_width = hd,
            .visible = visible,
            .scale = 1.0 / @sqrt(@as(f32, hd)),
            .precision = precision,
            .bias = .{ .table = self.bias_table.slice(0, model.heads * extent * 4), .extent = extent },
            .sinks = self.sinks,
        });
        try b.sigmoidGate(self.mixed_out, self.gate, 1, model.query_width, model.query_width, 0);
        try self.mm(block.output, self.mixed_out, self.convolved);
        try self.convolve(il, 2 * kw, E, self.convolved, c.attention_post_conv, null, self.projected);
    }

    /// The experts of application `il` over the mixed residual, into
    /// `projected`, ready to add to `cur`.
    fn feedForward(self: *Plan, il: usize, block: *const model.Block, c: *const Constants, count: usize, looped: bool) !void {
        const b = self.backend;
        const f = row(self.inputs, 0);
        try self.mixWithCur(count, c.ffn_query, 1, f);
        if (looped) try self.control(block.loop orelse return error.InvalidShape, 3, f);
        try b.rmsNorm(f, c.ffn_norm, f, .{ .rows = 1, .width = E, .in_stride = E, .out_stride = E });
        try self.mm(block.router, f, self.router_logits);
        try b.routeBiased(self.router_logits, c.router_bias, model.experts, k_used, 1, model.experts, model.expert_weights_scale, self.indices, self.route_weights);
        const gate_up = try self.expertWeight(block.gate_up_experts);
        const down = try self.expertWeight(block.down_experts);
        if (gate_up.tensor.experts != model.experts or down.tensor.experts != model.experts or gate_up.tensor.rows != 2 * ff or down.tensor.columns != ff) return error.InvalidShape;
        try b.matvecExperts(gate_up.buffer, gate_up.tensor, self.indices, k_used, f, 0, self.gate_up, 2 * ff);
        const scaled: ?metal.Backend.RowScales = if (c.gate_up_scale) |s| .{ .indices = self.indices, .scales = s, .experts = model.experts } else null;
        try b.situMulRows(self.gate_up, self.gate_up.slice(ff * 4, self.gate_up.len - ff * 4), self.hidden, ff, k_used, 2 * ff, 2 * ff, ff, model.situ_beta, model.situ_linear_beta, scaled);
        try b.matvecExperts(down.buffer, down.tensor, self.indices, k_used, self.hidden, ff, self.down, E);
        try b.combineExperts(self.down, self.route_weights, self.indices, c.down_scale, self.expert_out, .{ .columns = E, .slots = k_used, .rows = 1, .experts = model.experts, .in_stride = E, .out_stride = E });
        try self.convolve(il, 2 * model.kv_width + E, E, self.expert_out, c.ffn_post_conv, null, self.projected);
    }

    /// Steps the prompt, one command buffer a token; only the last token
    /// runs the head. `hidden_rows` is refused: no drafter reads them.
    pub fn prefill(self: *Plan, tokens: []const u32, logits: ?[]f32, greedy: ?*u32, topk: ?*sampling.TopK, penalties: ?sampling.Penalties, hidden_rows: ?[]f32, observer: ?Observer) !void {
        if (tokens.len == 0) return error.InvalidShape;
        if (hidden_rows != null) return error.HiddenUnsupported;
        for (tokens) |t| if (t >= vocabulary) return error.InvalidTokenId;
        if (self.state.status != .ready) return error.SessionNotReady;
        // The whole prompt must fit before the first token; each token then
        // commits, so an error part way leaves those before it consumed.
        if (tokens.len > self.state.capacity - self.state.position) return error.ContextFull;
        var inner = observer;
        if (inner) |*o| o.progress = null;
        for (tokens, 0..) |token, i| {
            const last = i + 1 == tokens.len;
            try self.step(token, if (last) logits else null, if (last) greedy else null, if (last) topk else null, if (last) penalties else null, inner);
            if (observer) |o| if (o.progress) |call| try call(o.context, .{ .phase = .prefill, .position = i + 1, .target = tokens.len });
        }
    }

    pub fn readLogits(self: *Plan, out: []f32) !void {
        return self.readout.readLogits(out);
    }
};
