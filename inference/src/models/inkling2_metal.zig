//! Inkling 2 on Metal: one command buffer records the 48 applications of
//! the 24 stored blocks over a batch of rows, what
//! `inkling2_runtime.Runtime.step` computes per token (that file is the
//! reference and names each stage). A decode step is a batch of one row on
//! the decode kernels; a prompt chunk and a verify batch take the tiles,
//! the gathered experts, and the materialized attention.
//!
//! The session is the CPU runtime's layout (`inkling2_runtime.layouts`),
//! wrapped whole as one buffer: each application's KV rows and its
//! convolution history live there, so snapshots, rewinds, and resets are
//! the session's, and a chunk's convolutions read the history the last
//! chunk left. The residual bookkeeping (which blocks are open, the core's
//! bank) is host integers, the same for every row and every token.
//! docs/models/inkling2.md § Metal says what the plan covers.
const std = @import("std");
const model = @import("inkling2.zig");
const models = @import("root.zig");
const runtime = @import("inkling2_runtime.zig");
const weights = @import("../runtime/weights.zig");
const session = @import("../runtime/session.zig");
const sampling = @import("../sampling/root.zig");
const metal = @import("../backends/metal/root.zig");
const chat_plan = @import("chat_plan.zig");
const cpu = @import("../backends/cpu/root.zig");
const Tensor = @import("../formats/gguf.zig").Tensor;
const Buffer = metal.Buffer;
const observer_types = @import("../runtime/observer.zig");

pub const Observer = observer_types.Observer;

pub const vocabulary = model.vocabulary;
/// Rows a verify batch may hold: the attention's few-row pass.
pub const max_verify_rows = metal.Backend.verify_max_rows;
/// Batches of at most this many rows run their experts through the
/// gathered matvec, one slot at a time, instead of the gathered tiles
/// (docs/models/inkling2.md § Metal).
pub const matvec_expert_rows = 32;

const E = model.embedding;
const hd = model.head_size;
const taps = model.conv_taps;
const ff = model.expert_feed_forward;
const k_used = model.experts_used;
const rl = model.relative_length;
const qw = model.query_width;
const kw = model.kv_width;
/// Closed blocks a residual holds at once, and the core's bank
/// (`inkling2_runtime`'s `max_blocks` and `max_bank`).
const max_blocks = model.stored_blocks / model.residual_span;
const max_bank = model.core_blocks / model.residual_span;
/// Floats of one row's states: the closed blocks, then `cur` while a mix
/// reads it.
const state_stride = (max_blocks + 1) * E;
/// The widest bias table: the window on a sliding application, the extent
/// on a global one.
const max_extent = @max(model.relative_extent, model.window);
/// Rows of a row's `messages`: the attention input's, the experts', then
/// K's and V's, so each of the two bank mixes writes two adjacent rows.
const message_row = [4]usize{ 0, 2, 3, 1 };
const eps = model.rms_epsilon;

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
    /// One u32 zero: `gatherRows`'s map for a strided row copy.
    zero_map: Buffer,
    /// The prompt rows one command buffer records.
    chunk: usize,
    /// Rows every batch buffer holds: the chunk or a verify batch, padded
    /// to the matmul tile (`n` in the counts below).
    capacity_rows: usize,
    has_draft: bool,

    // A batch's activations, row-major (F32 counts in comments).
    cur: Buffer, // n × E
    projected: Buffer, // n × E
    convolved: Buffer, // n × E
    expert_out: Buffer, // n × E
    prelude_cur: Buffer, // n × E
    /// The experts' mixed input, and the head's.
    mixed: Buffer, // n × E
    states: Buffer, // n × state_stride
    bank: Buffer, // n × max_bank × E
    messages: Buffer, // n × 4 × E
    /// Each row's three attention inputs (q, k, v).
    inputs: Buffer, // n × 3 × E
    /// The normed inputs, one contiguous plane each, so the projections
    /// read rows of `E`.
    normed: Buffer, // 3 × n × E
    joined: Buffer, // n × 2E
    rank: Buffer, // n × gate_rank
    gains: Buffer, // n × E
    forgets: Buffer, // n × E
    q: Buffer, // n × query_width
    gate: Buffer, // n × query_width
    mixed_out: Buffer, // n × query_width
    projected_k: Buffer, // n × kv_width
    projected_v: Buffer, // n × kv_width
    /// The rows' keys and values before they are packed, when the cache is
    /// not F32 (an F32 cache takes them in place).
    k: Buffer, // n × kv_width
    v: Buffer, // n × kv_width
    relative: Buffer, // n × heads × rl
    sinks: Buffer, // n × heads
    bias_table: Buffer, // n × heads × max_extent
    dynamic_k: Buffer, // n × kv_heads × taps
    dynamic_v: Buffer, // n × kv_heads × taps
    partials: Buffer,
    /// The materialized attention's score strips; null when no batch is
    /// past the few-row pass.
    strips: ?Buffer,
    /// A Q8_0 cache's rows expanded to F16 for a chunk, keys then values,
    /// `capacity × kv_width` halves each; null otherwise.
    kv_stage: ?Buffer,
    router_logits: Buffer, // n × experts
    indices: Buffer, // n × k_used u32
    route_weights: Buffer, // n × k_used
    lists: Buffer,
    gate_up: Buffer, // n × k_used × 2·ff
    hidden: Buffer, // n × k_used × ff
    down: Buffer, // n × k_used × E
    /// The head's padded rows for one row; the first `vocabulary`,
    /// soft-capped, are the logits.
    padded_logits: Buffer, // padded_vocabulary
    /// The head's padded rows for a verify batch; a float when no drafter
    /// is bound.
    head_rows: Buffer, // matmulPadded(max_verify_rows) × padded_vocabulary
    readout: chat_plan.Readout(vocabulary, max_verify_rows),
    /// Every application's routing for the batch, while an observer asks.
    expert_tap: ?Buffer,
    tapping: bool,
    /// The projector's rows for the chunk being recorded, when it is an
    /// image span's; borrowed for that one chunk.
    image_rows: ?[]const f32,

    /// The prompt chunk the engine should ask for (docs/models/inkling2.md
    /// § Metal).
    pub fn preferredChunk(binding: model.Binding, default: usize) usize {
        _ = binding;
        _ = default;
        return 1024;
    }

    /// `chunk` bounds the prompt rows one command buffer records and sizes
    /// the batch buffers. `kv` is every application's cache precision.
    /// `draft` sizes the verify head; no drafter is bound to this family
    /// yet, so a plan without it refuses `verify`.
    pub fn init(alloc: std.mem.Allocator, backend: *metal.Backend, view: weights.View, binding: model.Binding, capacity: usize, chunk: usize, kv: session.Precision, checkpoint: bool, draft: bool) !Plan {
        if (chunk == 0 or chunk > metal.Backend.max_chunk_rows) return error.InvalidShape;
        const all = runtime.layouts(kv, chunk);
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
        self.chunk = chunk;
        self.has_draft = draft;
        self.expert_tap = null;
        self.tapping = false;
        self.image_rows = null;
        const rows: usize = if (draft) @max(chunk, max_verify_rows) else chunk;
        const n = metal.Backend.matmulPadded(rows);
        self.capacity_rows = n;
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
        self.loop_query = try created.create(model.loop_queries * E * 4);
        self.loop_kv_query = try created.create(model.loop_queries * E * 4);
        try model.loopQueries(view, &binding, self.loop_query.floats()[0 .. model.loop_queries * E], self.loop_kv_query.floats()[0 .. model.loop_queries * E]);
        self.ones = try created.create(hd * 4);
        @memset(self.ones.floats(), 1);
        self.zero_map = try created.create(4);
        @memset(self.zero_map.floats(), 0);
        inline for (.{ "cur", "projected", "convolved", "expert_out", "prelude_cur", "mixed", "gains", "forgets" }) |f| @field(self, f) = try created.create(n * E * 4);
        self.states = try created.create(n * state_stride * 4);
        self.bank = try created.create(n * max_bank * E * 4);
        self.messages = try created.create(n * 4 * E * 4);
        self.inputs = try created.create(n * 3 * E * 4);
        self.normed = try created.create(3 * n * E * 4);
        self.joined = try created.create(n * 2 * E * 4);
        self.rank = try created.create(n * model.gate_rank * 4);
        inline for (.{ "q", "gate", "mixed_out" }) |f| @field(self, f) = try created.create(n * qw * 4);
        inline for (.{ "projected_k", "projected_v", "k", "v" }) |f| @field(self, f) = try created.create(n * kw * 4);
        self.relative = try created.create(n * model.heads * rl * 4);
        self.sinks = try created.create(n * model.heads * 4);
        self.bias_table = try created.create(n * model.heads * max_extent * 4);
        self.dynamic_k = try created.create(n * model.kv_heads * taps * 4);
        self.dynamic_v = try created.create(n * model.kv_heads * taps * 4);
        self.partials = try created.create(metal.Backend.attentionVerifyPartials(metal.Backend.verify_max_rows, model.heads, hd) * 4);
        const materialized = chunk > metal.Backend.verify_max_rows;
        self.strips = if (materialized) try created.create(metal.Backend.attentionStripScratch(model.heads, chunk, capacity)) else null;
        self.kv_stage = if (materialized and kv == .q8_0) try created.create(2 * capacity * kw * 2) else null;
        self.router_logits = try created.create(n * model.experts * 4);
        self.indices = try created.create(n * k_used * 4);
        self.route_weights = try created.create(n * k_used * 4);
        self.lists = try created.create(metal.Backend.expertListsLayout(n * k_used, model.experts).words * 4);
        self.gate_up = try created.create(n * k_used * 2 * ff * 4);
        self.hidden = try created.create(n * k_used * ff * 4);
        self.down = try created.create(n * k_used * E * 4);
        self.padded_logits = try created.create(model.padded_vocabulary * 4);
        self.head_rows = try created.create(if (draft) metal.Backend.matmulPadded(max_verify_rows) * model.padded_vocabulary * 4 else 4);
        self.readout = try .init(&created, draft);
        return self;
    }

    pub fn deinit(self: *Plan) void {
        // GPU buffers are released with the backend, except the tap made
        // on demand; the session and the constants table are ours.
        if (self.expert_tap) |tap| self.backend.release(tap) catch {};
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
    /// A projection of `count` rows. One row takes the decode matvec; a
    /// shape the tiles refuse (the tiny model's) the multi-row matvec,
    /// eight rows a dispatch; every other batch the tiles.
    fn mmRows(self: *Plan, tensor: *const Tensor, input: Buffer, in_stride: usize, output: Buffer, out_stride: usize, count: usize) !void {
        const w = try self.weight(tensor);
        const m = w.matrix;
        const b = self.backend;
        if (count == 1) return b.matvec(w.buffer, m, input.slice(0, m.columns * 4), output.slice(0, m.rows * 4));
        if (m.rows % 8 == 0 and m.columns % 64 == 0) return b.matmul(w.buffer, m, input, in_stride, output, out_stride, count);
        var t: usize = 0;
        while (t < count) {
            const r: usize = @min(metal.Backend.matvec_rows_max, count - t);
            const in = input.slice(t * in_stride * 4, input.len - t * in_stride * 4);
            const out = output.slice(t * out_stride * 4, output.len - t * out_stride * 4);
            if (r == 1) try b.matvec(w.buffer, m, in.slice(0, m.columns * 4), out.slice(0, m.rows * 4)) else try b.matvecRows(w.buffer, m, in, in_stride, out, out_stride, r);
            t += r;
        }
    }
    /// The GPU range of one session region.
    fn stateSlice(self: *Plan, region: []const u8) Buffer {
        return self.state_buffer.slice(self.state.offsetOf(region), region.len);
    }
    /// `buffer` from float `offset` to its end.
    fn from(buffer: Buffer, offset: usize) Buffer {
        return buffer.slice(offset * 4, buffer.len - offset * 4);
    }
    /// Row `j` of each row's three attention inputs (stride `3E`).
    fn inputRow(self: *Plan, j: usize) Buffer {
        return from(self.inputs, j * E);
    }
    /// Plane `j` of the normed inputs (`n × E`, contiguous).
    fn plane(self: *Plan, j: usize) Buffer {
        return self.normed.slice(j * self.capacity_rows * E * 4, self.capacity_rows * E * 4);
    }
    /// `count` rows of `width` floats from `src` (rows `src_stride` apart)
    /// to `dst` (`dst_stride` apart); the two must not overlap.
    fn copyRows(self: *Plan, dst: Buffer, dst_stride: usize, src: Buffer, src_stride: usize, width: usize, count: usize) !void {
        if (count == 1 or (dst_stride == width and src_stride == width)) return self.backend.copy(dst, src, count * width);
        try self.backend.gatherRows(dst, src, self.zero_map, width, 1, count, src_stride, dst_stride);
    }
    /// Application `il`'s convolution history for `count` channels from `first`.
    fn history(self: *Plan, il: usize, first: usize, count: usize) Buffer {
        const h = self.state.layers[model.block_count + il].recurrent.history[first * (taps - 1) ..][0 .. count * (taps - 1)];
        return self.stateSlice(std.mem.sliceAsBytes(h));
    }

    /// `cpu.recurrent.shortConvolution` of `count` contiguous rows, the
    /// history read for the rows before the batch and then advanced past it.
    fn convolve(self: *Plan, il: usize, first: usize, channels: usize, in: Buffer, w: Buffer, dynamic: ?metal.Backend.DynamicTaps, output: Buffer, count: usize) !void {
        const h = self.history(il, first, channels);
        try self.backend.shortConvolution(h, in, w, dynamic, output, channels, taps, count, channels, channels);
        try self.backend.convolutionHistory(h, in, channels, taps, count, channels);
    }

    /// Puts each row's `cur` after its `closed` closed blocks, so a mix
    /// reads `states[0..closed] ∪ {cur}`.
    fn stageCur(self: *Plan, closed: usize, count: usize) !void {
        try self.copyRows(from(self.states, closed * E), state_stride, self.cur, E, E, count);
    }
    /// `mix` of each row's staged states by `queries` shared query rows
    /// into `output` (rows `out_stride` apart).
    fn mix(self: *Plan, closed: usize, query: Buffer, queries: usize, output: Buffer, out_stride: usize, count: usize) !void {
        try self.backend.depthMix(self.states, query, output, E, closed + 1, queries, count, state_stride, out_stride, eps);
    }

    /// `cpu.depth.control` of `loop`'s `module` on `count` rows of `x`
    /// (`stride` apart) with their messages, in place.
    fn control(self: *Plan, loop: model.Loop, module: usize, x: Buffer, stride: usize, count: usize) !void {
        const b = self.backend;
        const message = from(self.messages, message_row[module] * E);
        try b.controlJoin(x, message, self.joined, E, count, stride, 4 * E, 2 * E, eps);
        try self.mmRows(loop.down[module], self.joined, 2 * E, self.rank, model.gate_rank, count);
        try self.mmRows(loop.gain[module], self.rank, model.gate_rank, self.gains, E, count);
        try self.mmRows(loop.forget[module], self.rank, model.gate_rank, self.forgets, E, count);
        try b.controlBlend(x, message, self.gains, self.forgets, x, E, count, stride, 4 * E, E, stride);
    }

    /// Records the embedding and the 48 applications over `tokens`, one
    /// row each, leaving the residuals in `cur`; returns the closed blocks
    /// each row holds. `layered` commits and reports after every
    /// application (a one-row step only). The caller owns the command
    /// buffer and the state admission.
    fn record(self: *Plan, tokens: []const u32, observer: ?Observer, layered: bool) !usize {
        const b = self.backend;
        const count = tokens.len;
        if (self.image_rows) |rows| {
            // Image rows skip the token norm. The previous command buffer
            // has completed, so the host may write the batch rows.
            @memcpy(self.cur.floats()[0 .. count * E], rows[0 .. count * E]);
        } else {
            const embedding = try self.weight(self.binding.token_embedding);
            for (tokens, 0..) |token, t| try b.embed(embedding.buffer, embedding.matrix, token, self.cur.slice(t * E * 4, E * 4));
            try b.rmsNorm(self.cur, self.token_norm, self.cur, .{ .rows = count, .width = E, .in_stride = E, .out_stride = E });
        }
        var closed: usize = 0;
        var prelude_closed: usize = 0;
        var bank_closed: usize = 0;
        for (0..model.block_count) |il| {
            const ip = model.storedBlock(il);
            const block = &self.binding.blocks[ip];
            const c = &self.constants[ip];
            var looped = false;
            if (model.loops > 1 and ip >= model.core_start and ip < model.core_end) {
                const pass = (il - model.core_start) / model.core_blocks;
                if (ip == model.core_start) {
                    if (pass == 0) {
                        prelude_closed = closed;
                        try b.copy(self.prelude_cur, self.cur, count * E);
                    } else {
                        bank_closed = closed - prelude_closed - 1;
                        if (bank_closed > 0) try self.copyRows(self.bank, max_bank * E, from(self.states, (prelude_closed + 1) * E), state_stride, bank_closed * E, count);
                        try self.copyRows(from(self.bank, bank_closed * E), max_bank * E, self.cur, E, E, count);
                        closed = prelude_closed;
                        try b.copy(self.cur, self.prelude_cur, count * E);
                    }
                }
                if (pass > 0) {
                    const at = 2 * (ip - model.core_start) * E * 4;
                    try b.depthMix(self.bank, self.loop_query.slice(at, 2 * E * 4), self.messages, E, bank_closed + 1, 2, count, max_bank * E, 4 * E, eps);
                    try b.depthMix(self.bank, self.loop_kv_query.slice(at, 2 * E * 4), from(self.messages, 2 * E), E, bank_closed + 1, 2, count, max_bank * E, 4 * E, eps);
                    looped = true;
                }
            }
            try self.attention(il, block, c, closed, looped, count);
            if (ip % model.residual_span == 0) {
                try self.copyRows(from(self.states, closed * E), state_stride, self.cur, E, E, count);
                closed += 1;
                try b.copy(self.cur, self.projected, count * E);
            } else try b.add(self.cur, self.projected, count * E);
            try self.feedForward(il, block, c, closed, looped, count);
            try b.add(self.cur, self.projected, count * E);
            if (observer) |o| {
                if (o.check) |check| try check(o.context);
                if (layered) if (o.layer) |report| {
                    try b.commit();
                    try finite(self.cur, 1);
                    try report(o.context, il, self.cur.floats()[0..E]);
                    try b.begin();
                };
            }
        }
        return closed;
    }

    /// The head over row `row`: its mix, norm, padded logits, soft-cap,
    /// and the selections asked for.
    fn head(self: *Plan, row: usize, closed: usize, greedy: bool, topk: ?*sampling.TopK, penalties: ?sampling.Penalties) !void {
        const b = self.backend;
        const states = self.states.slice(row * state_stride * 4, state_stride * 4);
        try b.copy(states.slice(closed * E * 4, E * 4), self.cur.slice(row * E * 4, E * 4), E);
        const y = self.mixed.slice(0, E * 4);
        try b.depthMix(states, self.output_query, y, E, closed + 1, 1, 1, state_stride, E, eps);
        try b.rmsNorm(y, self.output_norm, y, .{ .rows = 1, .width = E, .in_stride = E, .out_stride = E });
        try self.mmRows(self.binding.output, y, E, self.padded_logits, model.padded_vocabulary, 1);
        try b.copy(self.readout.logits, self.padded_logits, vocabulary);
        try b.softcap(self.readout.logits, vocabulary, model.final_softcap);
        try self.readout.select(b, greedy, topk, penalties);
    }

    /// The head over every row of a verify batch, into `head_rows`
    /// (padded rows, unsoftcapped).
    fn headRows(self: *Plan, closed: usize, count: usize) !void {
        const b = self.backend;
        try self.stageCur(closed, count);
        try self.mix(closed, self.output_query, 1, self.mixed, E, count);
        try b.rmsNorm(self.mixed, self.output_norm, self.mixed, .{ .rows = count, .width = E, .in_stride = E, .out_stride = E });
        try self.mmRows(self.binding.output, self.mixed, E, self.head_rows, model.padded_vocabulary, count);
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
        try self.beginTap(observer);
        defer self.tapping = false;
        const b = self.backend;
        try b.begin();
        errdefer b.discard();
        const closed = try self.record(&.{token}, observer, true);
        if (logits != null or greedy != null or topk != null) try self.head(0, closed, greedy != null, topk, penalties);
        try b.commit();
        try finite(self.cur, 1);
        try self.reportTap(observer, .decode, 1);
        try self.readout.readOutputs(logits, greedy, topk, penalties != null);
        try self.state.commit();
    }

    /// The CPU forward fails a step whose residual leaves the finite range;
    /// the plan checks the last residual of every row, which any
    /// non-finite value reaches.
    fn finite(buffer: Buffer, count: usize) !void {
        for (buffer.floats()[0 .. count * E]) |x| if (!std.math.isFinite(x)) return error.NonFiniteResult;
    }

    /// The attention of application `il` over each row's `cur` and its
    /// `closed` closed blocks, into `projected`; `looped` applies the
    /// pass's messages.
    fn attention(self: *Plan, il: usize, block: *const model.Block, c: *const Constants, closed: usize, looped: bool, count: usize) !void {
        const b = self.backend;
        const global = block.kind() == .global;
        if (c.attention_query) |query| {
            try self.stageCur(closed, count);
            if (c.attention_queries == 3) {
                try self.mix(closed, query, 3, self.inputs, 3 * E, count);
            } else {
                // One shared query: the norms read every row's copy when
                // they differ or the controller changes them.
                const rows: usize = if (global or looped) 3 else 1;
                for (0..rows) |j| try self.mix(closed, query, 1, self.inputRow(j), 3 * E, count);
            }
        } else for (0..3) |j| try self.copyRows(self.inputRow(j), 3 * E, self.cur, E, E, count);
        if (looped) {
            const loop = block.loop orelse return error.InvalidShape;
            for (0..3) |j| try self.control(loop, j, self.inputRow(j), 3 * E, count);
        }
        const strided: metal.Backend.Norm = .{ .rows = count, .width = E, .in_stride = 3 * E, .out_stride = E };
        // Which normed plane feeds q, k, and v.
        var source: [3]usize = .{ 0, 1, 2 };
        if (global) {
            for (0..3) |j| try b.rmsNorm(self.inputRow(j), c.norms[j] orelse return error.InvalidShape, self.plane(j), strided);
        } else if (looped) {
            for (0..3) |j| try b.rmsNorm(self.inputRow(j), c.norms[0].?, self.plane(j), strided);
        } else {
            try b.rmsNorm(self.inputRow(0), c.norms[0].?, self.plane(0), strided);
            source = .{ 0, 0, 0 };
        }
        const in_q = self.plane(source[0]);
        const in_k = self.plane(source[1]);
        const in_v = self.plane(source[2]);
        const heads = model.heads;
        try self.mmRows(block.query, in_q, E, self.q, qw, count);
        try self.mmRows(block.key, in_k, E, self.projected_k, kw, count);
        try self.mmRows(block.value, in_v, E, self.projected_v, kw, count);
        try self.mmRows(block.relative, in_q, E, self.relative, heads * rl, count);
        try self.mmRows(block.gate, in_q, E, self.gate, qw, count);
        try self.mmRows(block.sink_projection, in_q, E, self.sinks, heads, count);
        var dynamic_k: ?metal.Backend.DynamicTaps = null;
        var dynamic_v: ?metal.Backend.DynamicTaps = null;
        switch (block.input) {
            .sliding => {},
            .global => |g| {
                const width = model.kv_heads * taps;
                try self.mmRows(g.key_conv_projection, in_k, E, self.dynamic_k, width, count);
                try self.mmRows(g.value_conv_projection, in_v, E, self.dynamic_v, width, count);
                dynamic_k = .{ .taps = self.dynamic_k, .heads = model.kv_heads, .stride = width };
                dynamic_v = .{ .taps = self.dynamic_v, .heads = model.kv_heads, .stride = width };
            },
        }
        const cache = self.state.layers[il].attention;
        const position = self.state.position;
        const precision = cache.keys.precision;
        const k_slot = self.stateSlice(cache.keys.range(position, count));
        const v_slot = self.stateSlice(cache.values.range(position, count));
        // An F32 cache takes the convolved rows directly; F16 and Q8_0 take
        // them through F32 rows and one pack dispatch.
        const k_rows = if (precision == .f32) k_slot else self.k;
        const v_rows = if (precision == .f32) v_slot else self.v;
        try self.convolve(il, 0, kw, self.projected_k, c.key_conv, dynamic_k, k_rows, count);
        try self.convolve(il, kw, kw, self.projected_v, c.value_conv, dynamic_v, v_rows, count);
        const log_scale: ?metal.Backend.LogScale = if (global) .{ .position = position, .floor = model.log_scale_floor, .alpha = model.log_scale_alpha } else null;
        try b.l2ClipRows(self.q, c.query_clip, model.qk_clip, log_scale, heads, hd, count, qw);
        try b.l2ClipRows(self.relative, c.relative_clip, model.relative_clip, log_scale, heads, rl, count, heads * rl);
        // The bias table's columns: the window on a sliding application
        // (the binder holds the tensor to it), the extent on a global one.
        const extent: usize = if (global) model.relative_extent else model.window;
        const bias = try self.weight(block.relative_bias);
        if (bias.matrix.rows != extent or bias.matrix.columns != rl) return error.InvalidShape;
        try b.relativeBias(self.relative, bias.buffer, bias.matrix, self.bias_table, count * heads);
        try b.l2ClipRows(k_rows, null, model.qk_clip, null, model.kv_heads, hd, count, kw);
        try b.rmsNorm(v_rows, self.ones, v_rows, .{ .rows = count * model.kv_heads, .width = hd, .in_stride = hd, .out_stride = hd });
        if (precision != .f32) try b.storeRows(&.{ .{ .dst = k_slot, .src = k_rows, .count = count * kw }, .{ .dst = v_slot, .src = v_rows, .count = count * kw } }, precision);
        const table: metal.Backend.Bias = .{ .table = self.bias_table.slice(0, count * heads * extent * 4), .extent = extent };
        const scale = 1.0 / @sqrt(@as(f32, hd));
        // A sliding application's row sees the last `window` rows, its own
        // included: the cache is sliced at the first row's earliest key and
        // the window mask hides the rest per row.
        const window: usize = if (global) 0 else model.window;
        const first = if (window != 0 and position + 1 > window) position + 1 - window else 0;
        const total = position + count - first;
        var keys = self.stateSlice(cache.keys.range(first, total));
        var values = self.stateSlice(cache.values.range(first, total));
        if (count == 1) {
            try b.attentionDecode(keys, values, self.q.slice(0, qw * 4), self.partials, self.mixed_out.slice(0, qw * 4), .{
                .query_heads = heads,
                .kv_heads = model.kv_heads,
                .key_width = hd,
                .value_width = hd,
                .visible = total,
                .scale = scale,
                .precision = precision,
                .bias = table,
                .sinks = self.sinks.slice(0, heads * 4),
            });
        } else {
            const few = count <= metal.Backend.verify_max_rows;
            var operands = precision;
            if (!few and precision == .q8_0) {
                // The materialized body reads half rows: a Q8_0 cache's are
                // expanded first.
                const stage = self.kv_stage orelse return error.InvalidShape;
                const half = stage.len / 2;
                const half_keys = stage.slice(0, total * kw * 2);
                const half_values = stage.slice(half, total * kw * 2);
                try b.unpackQ8(&.{ .{ .dst = half_keys, .src = keys, .count = total * kw }, .{ .dst = half_values, .src = values, .count = total * kw } });
                keys = half_keys;
                values = half_values;
                operands = .f16;
            }
            const shape: metal.Backend.AttentionChunkShape = .{ .query_heads = heads, .kv_heads = model.kv_heads, .key_width = hd, .value_width = hd, .position = position - first, .count = count, .q_stride = qw, .out_stride = qw, .scale = scale, .precision = operands, .window = window, .bias = table, .sinks = self.sinks };
            if (few) {
                try b.attentionVerify(keys, values, self.q, self.partials, self.mixed_out, shape);
            } else try b.attentionChunkMaterialized(keys, values, self.q, self.mixed_out, self.strips orelse return error.InvalidShape, shape);
        }
        try b.sigmoidGate(self.mixed_out, self.gate, count, qw, qw, 0);
        try self.mmRows(block.output, self.mixed_out, qw, self.convolved, E, count);
        try self.convolve(il, 2 * kw, E, self.convolved, c.attention_post_conv, null, self.projected, count);
    }

    /// The experts of application `il` over each row's mixed residual,
    /// into `projected`, ready to add to `cur`.
    fn feedForward(self: *Plan, il: usize, block: *const model.Block, c: *const Constants, closed: usize, looped: bool, count: usize) !void {
        const b = self.backend;
        const f = self.mixed;
        try self.stageCur(closed, count);
        try self.mix(closed, c.ffn_query, 1, f, E, count);
        if (looped) try self.control(block.loop orelse return error.InvalidShape, 3, f, E, count);
        try b.rmsNorm(f, c.ffn_norm, f, .{ .rows = count, .width = E, .in_stride = E, .out_stride = E });
        try self.mmRows(block.router, f, E, self.router_logits, model.experts, count);
        const indices = self.tapSlice(il, count) orelse self.indices;
        try b.routeBiased(self.router_logits, c.router_bias, model.experts, k_used, count, model.experts, model.expert_weights_scale, indices, self.route_weights);
        const gate_up = try self.expertWeight(block.gate_up_experts);
        const down = try self.expertWeight(block.down_experts);
        if (gate_up.tensor.experts != model.experts or down.tensor.experts != model.experts or gate_up.tensor.rows != 2 * ff or down.tensor.columns != ff) return error.InvalidShape;
        const slots = count * k_used;
        const tiles = count > matvec_expert_rows;
        if (count == 1) {
            try b.matvecExperts(gate_up.buffer, gate_up.tensor, indices, k_used, f, 0, self.gate_up, 2 * ff);
        } else if (!tiles) {
            // A few rows spread over nearly k experts each: the gathered
            // tiles would run one- or two-row lists, so each slot takes the
            // decode step's gathered matvec, its row shared by its k slots.
            try b.matvecExpertsGrouped(gate_up.buffer, gate_up.tensor, indices, slots, f, E, k_used, self.gate_up, 2 * ff);
        } else {
            try b.expertLists(indices, count, k_used, model.experts, self.lists);
            try b.matmulExperts(gate_up.buffer, gate_up.tensor, self.lists, count, k_used, f, E, k_used, self.gate_up, 2 * ff);
        }
        const scaled: ?metal.Backend.RowScales = if (c.gate_up_scale) |s| .{ .indices = indices, .scales = s, .experts = model.experts } else null;
        try b.situMulRows(self.gate_up, from(self.gate_up, ff), self.hidden, ff, slots, 2 * ff, 2 * ff, ff, model.situ_beta, model.situ_linear_beta, scaled);
        if (tiles) {
            try b.matmulExperts(down.buffer, down.tensor, self.lists, count, k_used, self.hidden, ff, 1, self.down, E);
        } else try b.matvecExperts(down.buffer, down.tensor, indices, slots, self.hidden, ff, self.down, E);
        try b.combineExperts(self.down, self.route_weights, indices, c.down_scale, self.expert_out, .{ .columns = E, .slots = k_used, .rows = count, .experts = model.experts, .in_stride = E, .out_stride = E });
        try self.convolve(il, 2 * kw + E, E, self.expert_out, c.ffn_post_conv, null, self.projected, count);
    }

    /// Application `il`'s slice of the routing tap for `count` rows, or
    /// null when not tapping.
    fn tapSlice(self: *Plan, il: usize, count: usize) ?Buffer {
        if (!self.tapping) return null;
        const tap = self.expert_tap orelse return null;
        return tap.slice(il * self.capacity_rows * k_used * 4, count * k_used * 4);
    }
    /// Starts tapping the routing when `observer` asks for it, creating the
    /// tap on first use.
    fn beginTap(self: *Plan, observer: ?Observer) !void {
        self.tapping = false;
        const o = observer orelse return;
        if (o.experts == null) return;
        if (self.expert_tap == null) self.expert_tap = try self.backend.create(model.block_count * self.capacity_rows * k_used * 4);
        self.tapping = true;
    }
    /// After the batch's commit: hands every application's routing to the
    /// observer and stops tapping.
    fn reportTap(self: *Plan, observer: ?Observer, batch: observer_types.Batch, count: usize) !void {
        if (!self.tapping) return;
        self.tapping = false;
        const report = observer.?.experts.?;
        const stride = self.capacity_rows * k_used;
        const all = @as([*]const u32, @ptrCast(@alignCast(self.expert_tap.?.host)))[0 .. model.block_count * stride];
        for (0..model.block_count) |il| {
            const ids = all[il * stride ..][0 .. count * k_used];
            for (ids) |id| if (id >= model.experts) return error.NonFiniteResult;
            try report(observer.?.context, .{ .batch = batch, .layer = il, .rows = count, .k = k_used, .experts = model.experts, .indices = ids });
        }
    }

    /// Consumes a prompt in chunks of at most `chunk` rows, one command
    /// buffer each; only the last row runs the head. The whole prompt is
    /// admitted before the first chunk, and each chunk commits, so an error
    /// part way leaves the chunks before it consumed. A per-layer observer
    /// is a per-token contract (`step`), refused here; `hidden_rows` is
    /// refused too: no drafter reads them.
    pub fn prefill(self: *Plan, tokens: []const u32, logits: ?[]f32, greedy: ?*u32, topk: ?*sampling.TopK, penalties: ?sampling.Penalties, hidden_rows: ?[]f32, observer: ?Observer) !void {
        if (tokens.len == 0) return error.InvalidShape;
        if (hidden_rows != null) return error.HiddenUnsupported;
        if (observer) |o| if (o.layer != null) return error.InvalidShape;
        for (tokens) |t| if (t >= vocabulary) return error.InvalidTokenId;
        if (logits) |out| if (out.len != vocabulary) return error.InvalidShape;
        if (topk) |top| if (!std.math.isFinite(top.temperature) or top.temperature <= 0) return error.InvalidShape;
        if (self.state.status != .ready) return error.SessionNotReady;
        if (tokens.len > self.state.capacity - self.state.position) return error.ContextFull;
        var offset: usize = 0;
        while (offset < tokens.len) {
            const count = @min(self.chunk, tokens.len - offset);
            const last = offset + count == tokens.len;
            try self.prefillChunk(tokens[offset..][0..count], if (last) logits else null, if (last) greedy else null, if (last) topk else null, if (last) penalties else null, observer);
            offset += count;
            if (observer) |o| if (o.progress) |call| try call(o.context, .{ .phase = .prefill, .position = offset, .target = tokens.len });
        }
    }

    /// Prefills a prompt with image spans: text runs and image spans are
    /// chunked separately (a span never shares a chunk with text), and a
    /// span's rows take the projector's rows (`features`, `Σ span.count ×
    /// E`) in place of the token embedding and its norm. Attention over
    /// them is causal, as for text.
    pub fn prefillVision(self: *Plan, tokens: []const u32, spans: []const model.VisionSpan, features: []const f32, logits: ?[]f32, greedy: ?*u32, topk: ?*sampling.TopK, penalties: ?sampling.Penalties, observer: ?Observer) !void {
        if (tokens.len == 0) return error.InvalidShape;
        try models.checkSpans(tokens.len, spans);
        var span_rows: usize = 0;
        for (spans) |span| span_rows += span.count;
        if (features.len < span_rows * E) return error.InvalidShape;
        if (observer) |o| if (o.layer != null) return error.InvalidShape;
        for (tokens) |t| if (t >= vocabulary) return error.InvalidTokenId;
        if (logits) |out| if (out.len != vocabulary) return error.InvalidShape;
        if (topk) |top| if (!std.math.isFinite(top.temperature) or top.temperature <= 0) return error.InvalidShape;
        if (self.state.status != .ready) return error.SessionNotReady;
        if (tokens.len > self.state.capacity - self.state.position) return error.ContextFull;
        var i: usize = 0;
        var si: usize = 0;
        var frow: usize = 0;
        defer self.image_rows = null;
        while (i < tokens.len) {
            const in_span = si < spans.len and spans[si].start == i;
            const end = if (in_span) i + spans[si].count else if (si < spans.len) spans[si].start else tokens.len;
            while (i < end) {
                const c = @min(self.chunk, end - i);
                const last = i + c == tokens.len;
                self.image_rows = if (in_span) features[frow * E ..][0 .. c * E] else null;
                if (in_span) frow += c;
                try self.prefillChunk(tokens[i..][0..c], if (last) logits else null, if (last) greedy else null, if (last) topk else null, if (last) penalties else null, observer);
                i += c;
                if (observer) |o| if (o.progress) |call| try call(o.context, .{ .phase = .prefill, .position = i, .target = tokens.len });
            }
            if (in_span) si += 1;
        }
    }

    fn prefillChunk(self: *Plan, tokens: []const u32, logits: ?[]f32, greedy: ?*u32, topk: ?*sampling.TopK, penalties: ?sampling.Penalties, observer: ?Observer) !void {
        const count = tokens.len;
        std.debug.assert(count >= 1 and count <= self.chunk);
        if (penalties) |p| try self.readout.syncPenalties(p);
        try self.state.beginChunk(count);
        errdefer self.state.fail();
        try self.beginTap(observer);
        defer self.tapping = false;
        const b = self.backend;
        try b.begin();
        errdefer b.discard();
        const closed = try self.record(tokens, observer, false);
        if (logits != null or greedy != null or topk != null) try self.head(count - 1, closed, greedy != null, topk, penalties);
        try b.commit();
        try finite(self.cur, count);
        try self.reportTap(observer, .prefill, count);
        try self.readout.readOutputs(logits, greedy, topk, penalties != null);
        try self.state.commitChunk(count);
    }

    /// Admits a verify batch of `tokens` (at most `max_verify_rows`) and
    /// records it. Refused without a drafter.
    fn beginVerify(self: *Plan, tokens: []const u32, h_rows: ?[]f32, observer: ?Observer) !void {
        if (!self.has_draft) return error.NoDraftBlock;
        if (h_rows != null) return error.HiddenUnsupported;
        if (observer) |o| if (o.layer != null) return error.InvalidShape;
        if (tokens.len == 0 or tokens.len > max_verify_rows) return error.InvalidShape;
        for (tokens) |t| if (t >= vocabulary) return error.InvalidTokenId;
        if (self.state.status != .ready) return error.SessionNotReady;
        if (tokens.len > self.state.capacity - self.state.position) return error.ContextFull;
    }

    /// Verifies a batch: records the applications once and runs the head
    /// over every row. Exactly one of `rows` (`tokens.len × vocabulary`)
    /// and `tops` (one `sampling.TopK` per row) is given; the soft-capped
    /// logits stay resident either way, so `readVerifyRow` serves a
    /// fallback. `h_rows` is refused: the family binds no drafter yet.
    pub fn verify(self: *Plan, tokens: []const u32, rows: ?[]f32, tops: ?[]sampling.TopK, h_rows: ?[]f32, observer: ?Observer) !void {
        try self.beginVerify(tokens, h_rows, observer);
        if ((rows == null) == (tops == null)) return error.InvalidShape;
        const count = tokens.len;
        if (rows) |r| if (r.len != count * vocabulary) return error.InvalidShape;
        if (tops) |t| {
            if (t.len != count) return error.InvalidShape;
            for (t) |top| if (!std.math.isFinite(top.temperature) or top.temperature <= 0) return error.InvalidShape;
        }
        try self.state.beginChunk(count);
        errdefer self.state.fail();
        try self.beginTap(observer);
        defer self.tapping = false;
        const b = self.backend;
        try b.begin();
        errdefer b.discard();
        const closed = try self.record(tokens, observer, false);
        try self.headRows(closed, count);
        const logits = self.readout.verify_logits;
        try self.copyRows(logits, vocabulary, self.head_rows, model.padded_vocabulary, vocabulary, count);
        // The rows are the distributions acceptance draws from, so they
        // carry the soft-cap `step` applies.
        try b.softcap(logits, count * vocabulary, model.final_softcap);
        if (tops) |out| {
            for (out, 0..) |top, i| try b.topk(logits.slice(i * vocabulary * 4, vocabulary * 4), vocabulary, sampling.TopK.capacity, top.temperature, self.readout.topk_rows[i]);
        }
        try b.commit();
        try finite(self.cur, count);
        try self.reportTap(observer, .verify, count);
        if (rows) |r| @memcpy(r, logits.floats()[0 .. count * vocabulary]);
        if (tops) |out| {
            for (out, 0..) |*top, i| try self.readout.readVerifyTopK(i, top);
        }
        try self.state.commitChunk(count);
    }

    /// `verify`'s greedy sibling: per-row argmax read back, no logit
    /// readback (the soft-cap is monotonic, so it is skipped).
    pub fn verifyGreedy(self: *Plan, tokens: []const u32, out: []u32, h_rows: ?[]f32, observer: ?Observer) !void {
        try self.beginVerify(tokens, h_rows, observer);
        if (out.len != tokens.len) return error.InvalidShape;
        const count = tokens.len;
        try self.state.beginChunk(count);
        errdefer self.state.fail();
        try self.beginTap(observer);
        defer self.tapping = false;
        const b = self.backend;
        try b.begin();
        errdefer b.discard();
        const closed = try self.record(tokens, observer, false);
        try self.headRows(closed, count);
        const r = self.readout;
        for (0..count) |i| try b.argmax(self.head_rows.slice(i * model.padded_vocabulary * 4, vocabulary * 4), vocabulary, r.argmax_values, r.argmax_indices, r.verify_argmax.slice(i * 4, 4));
        try b.commit();
        try finite(self.cur, count);
        try self.reportTap(observer, .verify, count);
        const ids = @as([*]const u32, @ptrCast(@alignCast(r.verify_argmax.host)))[0..count];
        for (ids, out) |id, *token| {
            if (id >= vocabulary) return error.NonFiniteResult;
            token.* = id;
        }
        try self.state.commitChunk(count);
    }

    /// Copies one row of the last `verify`'s logits out of the shared buffer:
    /// the fallback for a row whose readback could not decide.
    pub fn readVerifyRow(self: *Plan, row: usize, out: []f32) !void {
        return self.readout.readVerifyRow(row, out);
    }

    pub fn readLogits(self: *Plan, out: []f32) !void {
        return self.readout.readLogits(out);
    }
};
