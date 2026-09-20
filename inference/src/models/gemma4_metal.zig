//! GPU-resident execution of the pinned Gemma 4 text schedules, the dense
//! 12B and the 26B-A4B mixture of experts. One command buffer per token
//! (`step`) or per prompt chunk (`prefill`): the CPU supplies token ids and
//! reads back logits, a greedy token, or a partial top-k; every activation,
//! norm, gate, routing decision, and cache write stays on the GPU. The
//! schedule is `gemma4_runtime.zig`'s, which remains the CPU reference these
//! results are compared against (docs/reference/gemma4.md).
//!
//! What this plan asks of the backend beyond the Qwen plan, all of it
//! decided by the forward pass rather than by this file: the tanh-GELU gate
//! pair (`gelu_mul_pair`), the embedding scale and the per-layer output
//! scale as scalar epilogues (`scale`, `addScale`), the final logit soft-cap
//! (`softcap`), RoPE tables with the checkpoint's frequency factors, the
//! sliding window (a cache-row slice on decode, a `window` mask on prefill
//! chunks), the wide attention geometry of the global layers (16 query heads
//! of 512 channels over one or two KV heads: the decode kernel's wide
//! instantiation, the chunk kernel's value-column splits), and on the expert
//! configuration the gathered kernels (`route`, `matvecExperts`, and for
//! chunks `expertLists` + `matmulExperts`; docs/reference/metal-backend.md
//! § Gathered expert kernels). Every expert matrix is wrapped resident; a
//! token's dispatches read only its selected experts' bytes.
//!
//! Ownership: the plan borrows the mapped weights and the `Backend`; it owns
//! the session and the GPU buffers it creates. Session memory is wrapped once
//! as a shared buffer, so `reset()` is a CPU memset that is only valid after
//! `commit()` — which the synchronous backend guarantees. Every attention
//! cache is allocated for the full capacity, sliding layers included (the
//! reference does the same); a ring layout for the windowed layers is a
//! session-layout unit of its own (ENGN-19).
const std = @import("std");
const model = @import("gemma4.zig");
const weights = @import("../runtime/weights.zig");
const session = @import("../runtime/session.zig");
const sampling = @import("../sampling/root.zig");
const metal = @import("../backends/metal/root.zig");
const cpu = @import("../backends/cpu/root.zig");
const Tensor = @import("../formats/gguf.zig").Tensor;
const Buffer = metal.Buffer;

pub const Observer = @import("../runtime/observer.zig").Observer;

pub const vocabulary = model.vocabulary;
const heads = model.heads;
/// The widest per-layer geometry; sliding layers use a prefix of each buffer.
const q_width = model.Kind.global.queryWidth(); // 16 × 512
const kv_width = model.max_kv_width; // 8 × 256
const head_max = model.Kind.global.headSize();

/// The expert block's small F32 weights, one buffer each.
const ExpertConstants = struct {
    router_scale: Buffer,
    /// One factor per expert on its down projection (`combineExperts`).
    down_scale: Buffer,
    post_ffn_norm_1: Buffer,
    pre_ffn_norm_2: Buffer,
    post_ffn_norm_2: Buffer,
};

const LayerConstants = struct {
    attention_norm: Buffer,
    post_attention_norm: Buffer,
    ffn_norm: Buffer,
    post_ffn_norm: Buffer,
    query_norm: Buffer,
    key_norm: Buffer,
    /// The scalar multiplying the layer's whole new residual stream.
    output_scale: f32,
    experts: ?ExpertConstants,
};

/// GPU workspace of the expert block: one token's `k` slots for `step`, and
/// the chunk's `padded · k` slot rows (token `t`, slot `s` at row `t · k + s`,
/// the order `route` writes) for `prefill`.
const ExpertBuffers = struct {
    router_logits: Buffer, // experts
    indices: Buffer, // k u32
    route_weights: Buffer, // k
    gate_up: Buffer, // k × 2·ff (gate half, then up half, per slot)
    hidden: Buffer, // k × ff
    down: Buffer, // k × hidden
    out: Buffer, // hidden
    router_logits_c: Buffer, // padded × experts
    indices_c: Buffer, // padded·k u32
    route_weights_c: Buffer, // padded·k
    /// The slot rows grouped by expert (`expertListsLayout`), rebuilt per chunk.
    lists: Buffer,
    gate_up_c: Buffer, // padded·k × 2·ff
    hidden_c: Buffer, // padded·k × ff
    down_c: Buffer, // padded·k × hidden
    out_c: Buffer, // padded × hidden
};

/// A weight matrix ready for dispatch: its GPU range and CPU-side descriptor.
const Weight = struct { buffer: Buffer, matrix: cpu.Matrix };
/// A 3-D expert tensor ready for the gathered kernels.
const ExpertWeight = struct { buffer: Buffer, tensor: cpu.ExpertMatrix };

pub const Plan = struct {
    alloc: std.mem.Allocator,
    backend: *metal.Backend,
    view: weights.View,
    binding: model.Binding,
    state: session.Session,
    state_buffer: Buffer,
    constants: []LayerConstants,
    output_norm: Buffer,
    /// A row of ones: the per-head value norm has no weight.
    ones: Buffer, // head_max
    /// (cos, sin) tables: base 1e4 over 256 channels for sliding layers, base
    /// 1e6 over 512 channels with the checkpoint's factors for global layers.
    rope_sliding: Buffer,
    rope_global: Buffer,
    // Activations (F32 counts in comments).
    x: Buffer, // hidden
    normalized: Buffer, // hidden
    projected: Buffer, // hidden
    gate: Buffer, // ffn
    up: Buffer, // ffn
    q: Buffer, // q_width
    /// F32 key/value rows of the current token when the cache is F16:
    /// projected, normalized, and rotated here, then packed into the slot.
    /// With an F32 cache the projections write the slot directly.
    k: Buffer, // kv_width
    v: Buffer, // kv_width
    mixed_out: Buffer, // q_width
    /// Flash-decoding partials for the wide geometry: `[16][splits][2 + 512]`.
    partials: Buffer,
    logits: Buffer, // vocabulary
    argmax_values: Buffer,
    argmax_indices: Buffer,
    argmax_result: Buffer,
    topk: metal.Backend.TopKBuffers,
    /// The history bit set as device words for `nu_penalize`, uploaded when
    /// the history's revision changed; `penalty_revision` is the last upload.
    penalty_history: Buffer,
    penalty_revision: u64,
    /// Chunked prefill: up to `chunk` prompt tokens per command buffer
    /// through the batched matmul. The `_c` buffers hold `padded` token rows
    /// ([token][feature]); `step` keeps its own single-token buffers above.
    chunk: usize,
    padded: usize,
    x_c: Buffer, // padded × hidden
    normalized_c: Buffer,
    projected_c: Buffer,
    gate_c: Buffer, // padded × ffn
    up_c: Buffer,
    q_c: Buffer, // padded × q_width
    k_c: Buffer, // padded × kv_width
    v_c: Buffer,
    /// Half copy of `q_c` for the F16 chunk attention (its operands share one type).
    q_c_h: Buffer, // padded × q_width halves
    mixed_out_c: Buffer, // padded × q_width
    /// Present on the expert configuration only.
    experts: ?ExpertBuffers,

    /// The prompt chunk the engine should ask for: `default` on the dense
    /// configuration; 512 on the expert one, where a chunk's slot rows fill
    /// the gathered 32-row tiles better (measured 10–14 % faster prefill
    /// than 256, and 1,024 buys 6 % more only on long prompts for twice the
    /// workspace; gemma4.md § 26B-A4B).
    pub fn preferredChunk(binding: model.Binding, default: usize) usize {
        return if (binding.config.experts != null) 512 else default;
    }

    /// `chunk` bounds the tokens one `prefill` command buffer processes (and
    /// sizes its activation buffers: about 0.3 MB per token, plus 0.16 MB per
    /// token of expert slot rows on the 26B-A4B). `kv` is the attention
    /// cache precision of every layer.
    pub fn init(alloc: std.mem.Allocator, backend: *metal.Backend, view: weights.View, binding: model.Binding, capacity: usize, chunk: usize, kv: session.Precision, checkpoint: bool, draft: bool) !Plan {
        _ = draft;
        if (chunk == 0 or chunk > 4096) return error.InvalidShape;
        const config = binding.config;
        const hidden = config.embedding;
        const ffn = config.feed_forward;
        var layouts: [model.max_layers]session.Layout = undefined;
        for (binding.active(), layouts[0..config.layer_count]) |layer, *layout| {
            const width = layer.kvWidth();
            layout.* = .{ .attention = .{ .key_row = width, .value_row = width, .precision = kv } };
        }
        var state = try session.Session.init(alloc, layouts[0..config.layer_count], capacity, checkpoint, 0);
        errdefer state.deinit();
        const constants = try alloc.alloc(LayerConstants, config.layer_count);
        errdefer alloc.free(constants);
        var self: Plan = undefined;
        self.alloc = alloc;
        self.backend = backend;
        self.view = view;
        self.binding = binding;
        self.state = state;
        self.constants = constants;
        self.state_buffer = try backend.wrap(state.memory);
        for (binding.active(), constants) |layer, *c| {
            c.* = .{
                .attention_norm = try self.constant(layer.attention_norm),
                .post_attention_norm = try self.constant(layer.post_attention_norm),
                .ffn_norm = try self.constant(layer.ffn_norm),
                .post_ffn_norm = try self.constant(layer.post_ffn_norm),
                .query_norm = try self.constant(layer.query_norm),
                .key_norm = try self.constant(layer.key_norm),
                .output_scale = try view.scalar(layer.output_scale, 0),
                .experts = if (layer.experts) |e| .{
                    .router_scale = try self.constant(e.router_scale),
                    .down_scale = try self.constant(e.down_scale),
                    .post_ffn_norm_1 = try self.constant(e.post_ffn_norm_1),
                    .pre_ffn_norm_2 = try self.constant(e.pre_ffn_norm_2),
                    .post_ffn_norm_2 = try self.constant(e.post_ffn_norm_2),
                } else null,
            };
            if (!std.math.isFinite(c.output_scale)) return error.InvalidShape;
        }
        self.output_norm = try self.constant(binding.output_norm);
        self.ones = try backend.create(head_max * 4);
        @memset(self.ones.floats(), 1);
        const sliding_dims = model.Kind.sliding.headSize();
        const global_dims = model.Kind.global.headSize();
        self.rope_sliding = try backend.create(capacity * (sliding_dims / 2) * 8);
        try metal.Backend.ropeTable(self.rope_sliding, capacity, sliding_dims, model.Kind.sliding.ropeBase(), null);
        const factors = try view.vector(alloc, binding.rope_factors);
        defer alloc.free(factors);
        self.rope_global = try backend.create(capacity * (global_dims / 2) * 8);
        try metal.Backend.ropeTable(self.rope_global, capacity, global_dims, model.Kind.global.ropeBase(), factors);
        self.x = try backend.create(hidden * 4);
        self.normalized = try backend.create(hidden * 4);
        self.projected = try backend.create(hidden * 4);
        self.gate = try backend.create(ffn * 4);
        self.up = try backend.create(ffn * 4);
        self.q = try backend.create(q_width * 4);
        self.k = try backend.create(kv_width * 4);
        self.v = try backend.create(kv_width * 4);
        self.mixed_out = try backend.create(q_width * 4);
        self.partials = try backend.create(metal.Backend.attentionDecodePartials(heads, head_max) * 4);
        self.logits = try backend.create(vocabulary * 4);
        self.argmax_values = try backend.create(metal.Backend.argmax_partials * 4);
        self.argmax_indices = try backend.create(metal.Backend.argmax_partials * 4);
        self.argmax_result = try backend.create(4);
        self.topk = try backend.topkBuffers(sampling.TopK.capacity);
        self.penalty_history = try backend.create((vocabulary + 31) / 32 * 4);
        self.penalty_revision = std.math.maxInt(u64);
        self.chunk = chunk;
        self.padded = metal.Backend.matmulPadded(chunk);
        const n = self.padded;
        self.x_c = try backend.create(n * hidden * 4);
        self.normalized_c = try backend.create(n * hidden * 4);
        self.projected_c = try backend.create(n * hidden * 4);
        self.gate_c = try backend.create(n * ffn * 4);
        self.up_c = try backend.create(n * ffn * 4);
        self.q_c = try backend.create(n * q_width * 4);
        self.k_c = try backend.create(n * kv_width * 4);
        self.v_c = try backend.create(n * kv_width * 4);
        self.q_c_h = try backend.create(n * q_width * 2);
        self.mixed_out_c = try backend.create(n * q_width * 4);
        self.experts = if (config.experts) |spec| try self.expertBuffers(spec) else null;
        return self;
    }
    fn expertBuffers(self: *Plan, spec: model.Experts) !ExpertBuffers {
        const b = self.backend;
        const hidden = self.binding.config.embedding;
        const k = spec.used;
        const ff = spec.feed_forward;
        const n = self.padded * k;
        return .{
            .router_logits = try b.create(spec.count * 4),
            .indices = try b.create(k * 4),
            .route_weights = try b.create(k * 4),
            .gate_up = try b.create(k * 2 * ff * 4),
            .hidden = try b.create(k * ff * 4),
            .down = try b.create(k * hidden * 4),
            .out = try b.create(hidden * 4),
            .router_logits_c = try b.create(self.padded * spec.count * 4),
            .indices_c = try b.create(n * 4),
            .route_weights_c = try b.create(n * 4),
            .lists = try b.create(metal.Backend.expertListsLayout(n, spec.count).words * 4),
            .gate_up_c = try b.create(n * 2 * ff * 4),
            .hidden_c = try b.create(n * ff * 4),
            .down_c = try b.create(n * hidden * 4),
            .out_c = try b.create(self.padded * hidden * 4),
        };
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
    fn constant(self: *Plan, tensor: *const Tensor) !Buffer {
        const values = try self.view.vector(self.alloc, tensor);
        defer self.alloc.free(values);
        if (values.len == 0) return error.InvalidShape;
        const buffer = try self.backend.create(values.len * 4);
        @memcpy(buffer.floats(), values);
        return buffer;
    }
    /// Tensors are wrapped one at a time (cached by address), as in the Qwen plan.
    fn weight(self: *Plan, tensor: *const Tensor) !Weight {
        const matrix = try self.view.matrix(tensor);
        return .{ .buffer = try self.backend.wrap(matrix.bytes), .matrix = matrix };
    }
    /// The whole 3-D tensor is wrapped (no copy); the kernels read the
    /// selected experts' ranges of it.
    fn expertWeight(self: *Plan, tensor: *const Tensor) !ExpertWeight {
        const matrix = try self.view.expertMatrix(tensor);
        return .{ .buffer = try self.backend.wrap(matrix.bytes), .tensor = matrix };
    }
    /// The GPU range of one session region (a byte range of the wrapped block).
    fn stateSlice(self: *Plan, region: []const u8) Buffer {
        return self.state_buffer.slice(self.state.offsetOf(region), region.len);
    }
    fn mm(self: *Plan, tensor: *const Tensor, input: Buffer, output: Buffer) !void {
        const w = try self.weight(tensor);
        try self.backend.matvec(w.buffer, w.matrix, input, output);
    }
    /// Batched projection over `count` token rows.
    fn mmRows(self: *Plan, tensor: *const Tensor, input: Buffer, in_stride: usize, output: Buffer, out_stride: usize, count: usize) !void {
        const w = try self.weight(tensor);
        try self.backend.matmul(w.buffer, w.matrix, input, in_stride, output, out_stride, count);
    }
    /// The rope table and rotary width of a layer kind (the whole head).
    fn ropeOf(self: *Plan, kind: model.Kind) struct { table: Buffer, dims: usize } {
        return .{ .table = if (kind == .global) self.rope_global else self.rope_sliding, .dims = kind.headSize() };
    }
    /// First visible cache row for the token at `position`: sliding layers
    /// see the last `window` positions including their own (the reference
    /// reads them as a contiguous suffix of the cache rows).
    fn firstVisible(kind: model.Kind, position: usize) usize {
        return if (kind == .sliding and position + 1 > model.window) position + 1 - model.window else 0;
    }

    /// Merge only shapes the backend can encode as whole threadgroups. A
    /// rejected merge records no work; the standalone path remains the fallback.
    fn projections(self: *Plan, tensors: []const *const Tensor, outputs: []const Buffer, mode: metal.Backend.SegmentMode) !void {
        std.debug.assert(tensors.len == outputs.len and tensors.len <= 4);
        var segments: [4]metal.Backend.Segment = undefined;
        for (tensors, outputs, 0..) |tensor, output, i| {
            const w = try self.weight(tensor);
            segments[i] = .{ .weights = w.buffer, .matrix = w.matrix, .output = if (mode != .plain) outputs[0] else output };
        }
        self.backend.matvecSegments(segments[0..tensors.len], self.normalized, mode) catch |err| switch (err) {
            error.InvalidShape => {
                for (segments[0..tensors.len], outputs) |s, output| try self.backend.matvec(s.weights, s.matrix, self.normalized, output);
                switch (mode) {
                    .plain => {},
                    .silu_mul_pair => try self.backend.siluMul(outputs[0], outputs[1], segments[0].matrix.rows),
                    .gelu_mul_pair => try self.backend.geluMul(outputs[0], outputs[1], segments[0].matrix.rows),
                }
            },
            else => return err,
        };
    }

    /// Records and runs one token. `logits` receives the vocabulary scores when
    /// non-null; `greedy` receives the argmax when non-null (computed on the
    /// GPU, no logit readback); `topk` receives the best `TopK.capacity`
    /// logits and the softmax denominator for `topk.temperature`. An
    /// observer's `check` runs between recorded layers at no GPU cost; its
    /// `layer` callback forces a commit after every layer (48 command buffers
    /// per token) and is for numerical traces only.
    pub fn step(self: *Plan, token: u32, logits: ?[]f32, greedy: ?*u32, topk: ?*sampling.TopK, penalties: ?sampling.Penalties, observer: ?Observer) !void {
        if (token >= vocabulary) return error.InvalidTokenId;
        if (logits) |out| if (out.len != vocabulary) return error.InvalidShape;
        if (topk) |top| if (!std.math.isFinite(top.temperature) or top.temperature <= 0) return error.InvalidShape;
        if (penalties) |p| try self.syncPenalties(p);
        try self.state.begin();
        errdefer self.state.fail();
        const b = self.backend;
        try b.begin();
        errdefer if (b.recording) b.commit() catch {};
        const embedding = try self.weight(self.binding.token_embedding);
        const hidden = self.binding.config.embedding;
        try b.embed(embedding.buffer, embedding.matrix, token, self.x);
        try b.scale(self.x, hidden, self.binding.config.embeddingScale());
        const norm: metal.Backend.Norm = .{ .rows = 1, .width = hidden, .in_stride = hidden, .out_stride = hidden };
        for (self.binding.active(), self.constants, 0..) |layer, c, il| {
            try b.rmsNorm(self.x, c.attention_norm, self.normalized, norm);
            try self.attention(layer, c, il);
            try b.rmsNorm(self.projected, c.post_attention_norm, self.projected, norm);
            try b.add(self.x, self.projected, hidden);
            try self.feedForward(layer, c);
            try b.addScale(self.x, self.projected, hidden, c.output_scale);
            if (observer) |o| {
                if (o.check) |check| try check(o.context);
                if (o.layer) |report| {
                    try b.commit();
                    for (self.x.floats()) |v| if (!std.math.isFinite(v)) return error.NonFiniteResult;
                    try report(o.context, il, self.x.floats());
                    try b.begin();
                }
            }
        }
        if (logits != null or greedy != null or topk != null) {
            try b.rmsNorm(self.x, self.output_norm, self.normalized, norm);
            try self.recordOutputs(greedy, topk, penalties);
        }
        try b.commit();
        try self.readOutputs(logits, greedy, topk, penalties != null);
        try self.state.commit();
    }

    /// Uploads the history words to the device when the set changed since the
    /// last upload; valid before the command buffer's first dispatch.
    fn syncPenalties(self: *Plan, p: sampling.Penalties) !void {
        if (p.history.revision == self.penalty_revision) return;
        const words = (vocabulary + 31) / 32;
        const out = @as([*]u32, @ptrCast(@alignCast(self.penalty_history.host)))[0..words];
        if (p.history.writeWords(out) != words) return error.InvalidShape;
        self.penalty_revision = p.history.revision;
    }

    /// The feed-forward block over the residual `x` (the attention output
    /// already added) into `projected`, post-normed and ready for the
    /// scaled add: the CPU reference's `feedForward`. On an expert layer the
    /// dense FFN is the shared branch, and the router input is `rms(x)`
    /// weighted by `router_scale` then scaled by 1/sqrt(width); the
    /// reference applies the two factors in the other order, one F32
    /// rounding apart.
    fn feedForward(self: *Plan, layer: model.Layer, c: LayerConstants) !void {
        const b = self.backend;
        const hidden = self.binding.config.embedding;
        const norm: metal.Backend.Norm = .{ .rows = 1, .width = hidden, .in_stride = hidden, .out_stride = hidden };
        try b.rmsNorm(self.x, c.ffn_norm, self.normalized, norm);
        try self.projections(&.{ layer.ffn_gate, layer.ffn_up }, &.{ self.gate, self.up }, .gelu_mul_pair);
        try self.mm(layer.ffn_down, self.gate, self.projected);
        if (layer.experts) |experts| {
            const ec = c.experts orelse return error.InvalidShape;
            const e = self.experts orelse return error.InvalidShape;
            const spec = self.binding.config.experts orelse return error.InvalidShape;
            const k = spec.used;
            const ff = spec.feed_forward;
            try b.rmsNorm(self.projected, ec.post_ffn_norm_1, self.projected, norm);
            try b.rmsNorm(self.x, ec.router_scale, self.normalized, norm);
            try b.scale(self.normalized, hidden, 1.0 / self.binding.config.embeddingScale());
            try self.mm(experts.router, self.normalized, e.router_logits);
            try b.route(e.router_logits, spec.count, k, 1, spec.count, e.indices, e.route_weights);
            try b.rmsNorm(self.x, ec.pre_ffn_norm_2, self.normalized, norm);
            const gate_up = try self.expertWeight(experts.gate_up);
            const down = try self.expertWeight(experts.down);
            try b.matvecExperts(gate_up.buffer, gate_up.tensor, e.indices, k, self.normalized, 0, e.gate_up, 2 * ff);
            try b.geluMulRows(e.gate_up, e.gate_up.slice(ff * 4, e.gate_up.len - ff * 4), e.hidden, ff, k, 2 * ff, 2 * ff, ff);
            try b.matvecExperts(down.buffer, down.tensor, e.indices, k, e.hidden, ff, e.down, hidden);
            try b.combineExperts(e.down, e.route_weights, e.indices, ec.down_scale, e.out, .{ .columns = hidden, .slots = k, .rows = 1, .experts = spec.count, .in_stride = hidden, .out_stride = hidden });
            try b.rmsNorm(e.out, ec.post_ffn_norm_2, e.out, norm);
            try b.add(self.projected, e.out, hidden);
        }
        try b.rmsNorm(self.projected, c.post_ffn_norm, self.projected, norm);
    }

    /// `feedForward` over the chunk's `count` rows of `x_c` into
    /// `projected_c`. The expert branch routes every row, groups the
    /// `count · k` slot rows by expert, and runs the gathered tiles once per
    /// group instead of the decode matvec once per slot.
    fn feedForwardChunk(self: *Plan, layer: model.Layer, c: LayerConstants, count: usize) !void {
        const b = self.backend;
        const hidden = self.binding.config.embedding;
        const ffn = self.binding.config.feed_forward;
        const norm: metal.Backend.Norm = .{ .rows = count, .width = hidden, .in_stride = hidden, .out_stride = hidden };
        try b.rmsNorm(self.x_c, c.ffn_norm, self.normalized_c, norm);
        try self.mmRows(layer.ffn_gate, self.normalized_c, hidden, self.gate_c, ffn, count);
        try self.mmRows(layer.ffn_up, self.normalized_c, hidden, self.up_c, ffn, count);
        try b.geluMul(self.gate_c, self.up_c, count * ffn);
        try self.mmRows(layer.ffn_down, self.gate_c, ffn, self.projected_c, hidden, count);
        if (layer.experts) |experts| {
            const ec = c.experts orelse return error.InvalidShape;
            const e = self.experts orelse return error.InvalidShape;
            const spec = self.binding.config.experts orelse return error.InvalidShape;
            const k = spec.used;
            const ff = spec.feed_forward;
            const n = count * k;
            try b.rmsNorm(self.projected_c, ec.post_ffn_norm_1, self.projected_c, norm);
            try b.rmsNorm(self.x_c, ec.router_scale, self.normalized_c, norm);
            try b.scale(self.normalized_c, count * hidden, 1.0 / self.binding.config.embeddingScale());
            try self.mmRows(experts.router, self.normalized_c, hidden, e.router_logits_c, spec.count, count);
            try b.route(e.router_logits_c, spec.count, k, count, spec.count, e.indices_c, e.route_weights_c);
            try b.rmsNorm(self.x_c, ec.pre_ffn_norm_2, self.normalized_c, norm);
            try b.expertLists(e.indices_c, count, k, spec.count, e.lists);
            const gate_up = try self.expertWeight(experts.gate_up);
            const down = try self.expertWeight(experts.down);
            try b.matmulExperts(gate_up.buffer, gate_up.tensor, e.lists, count, k, self.normalized_c, hidden, k, e.gate_up_c, 2 * ff);
            try b.geluMulRows(e.gate_up_c, e.gate_up_c.slice(ff * 4, e.gate_up_c.len - ff * 4), e.hidden_c, ff, n, 2 * ff, 2 * ff, ff);
            try b.matmulExperts(down.buffer, down.tensor, e.lists, count, k, e.hidden_c, ff, 1, e.down_c, hidden);
            try b.combineExperts(e.down_c, e.route_weights_c, e.indices_c, ec.down_scale, e.out_c, .{ .columns = hidden, .slots = k, .rows = count, .experts = spec.count, .in_stride = hidden, .out_stride = hidden });
            try b.rmsNorm(e.out_c, ec.post_ffn_norm_2, e.out_c, norm);
            try b.add(self.projected_c, e.out_c, count * hidden);
        }
        try b.rmsNorm(self.projected_c, c.post_ffn_norm, self.projected_c, norm);
    }

    /// Records the tied output head for the normalized last-token row in
    /// `self.normalized`, the soft-cap, and whichever readbacks were requested.
    /// `penalties` applies `nu_penalize` after the soft-cap — the vector the
    /// CPU sampler penalizes — before either selection.
    fn recordOutputs(self: *Plan, greedy: ?*u32, topk: ?*sampling.TopK, penalties: ?sampling.Penalties) !void {
        const b = self.backend;
        try self.mm(self.binding.token_embedding, self.normalized, self.logits);
        try b.softcap(self.logits, vocabulary, model.final_softcap);
        // In place, so only a selection readback may request it: a raw
        // `logits` readback must stay unpenalized for a CPU sampler.
        if (penalties) |p| if (greedy != null or topk != null) try b.penalize(self.logits, vocabulary, self.penalty_history, p.repetition, p.presence);
        if (greedy != null) try b.argmax(self.logits, vocabulary, self.argmax_values, self.argmax_indices, self.argmax_result);
        if (topk) |top| try b.topk(self.logits, vocabulary, sampling.TopK.capacity, top.temperature, self.topk);
    }
    /// Copies the requested readbacks out after `commit`; `penalized` marks a
    /// `topk` readback taken after the device penalties.
    fn readOutputs(self: *Plan, logits: ?[]f32, greedy: ?*u32, topk: ?*sampling.TopK, penalized: bool) !void {
        if (logits) |out| try self.readLogits(out);
        if (greedy) |out| {
            const id = @as(*const u32, @ptrCast(@alignCast(self.argmax_result.host))).*;
            if (id >= vocabulary) return error.NonFiniteResult;
            out.* = id;
        }
        if (topk) |top| {
            top.penalized = penalized;
            const ids = @as([*]const u32, @ptrCast(@alignCast(self.topk.indices.host)))[0..sampling.TopK.capacity];
            for (ids) |id| if (id >= vocabulary) return error.NonFiniteResult;
            @memcpy(top.ids[0..], ids);
            @memcpy(top.values[0..], self.topk.values.floats()[0..sampling.TopK.capacity]);
            top.count = sampling.TopK.capacity;
            // Partition order is fixed, so this F64 sum is deterministic run to run.
            var total: f64 = 0;
            for (self.topk.sums.floats()[0..metal.Backend.topk_partials]) |partial| total += partial;
            top.total = total;
            top.finite = true;
            for (@as([*]const u32, @ptrCast(@alignCast(self.topk.flags.host)))[0..metal.Backend.topk_partials]) |flag| if (flag != 0) {
                top.finite = false;
            };
        }
    }

    /// Copies the last step's logits out of the shared buffer. Valid after a
    /// `step` that computed them (any non-null `logits`/`greedy`/`topk`) and
    /// until the next step; the sampler's fallback path uses it. When a
    /// selection was requested with penalties, the vector is the penalized
    /// one (the raw vector is not kept), so the sampler must not penalize it
    /// again.
    pub fn readLogits(self: *Plan, out: []f32) !void {
        if (out.len != vocabulary) return error.InvalidShape;
        @memcpy(out, self.logits.floats());
        for (out) |v| if (!std.math.isFinite(v)) return error.NonFiniteResult;
    }

    fn attention(self: *Plan, layer: model.Layer, c: LayerConstants, il: usize) !void {
        const b = self.backend;
        const kind = layer.kind;
        const hd = layer.headSize();
        const kv_heads = layer.kv_heads;
        const qw = layer.queryWidth();
        const kvw = layer.kvWidth();
        const cache = self.state.layers[il].attention;
        const position = self.state.position;
        const precision = cache.keys.precision;
        const k_slot = self.stateSlice(cache.keys.range(position, 1));
        const v_slot = self.stateSlice(cache.values.range(position, 1));
        // An F32 cache takes the projections directly; an F16 cache takes them
        // through F32 scratch rows and one pack dispatch.
        const k_row = if (precision == .f32) k_slot else self.k.slice(0, kvw * 4);
        const v_row = if (precision == .f32) v_slot else self.v.slice(0, kvw * 4);
        const q = self.q.slice(0, qw * 4);
        if (layer.value) |value| {
            try self.projections(&.{ layer.query, layer.key, value }, &.{ q, k_row, v_row }, .plain);
        } else {
            // Global layers have no value projection: V is the raw key
            // projection, taken before the key norm and RoPE.
            try self.projections(&.{ layer.query, layer.key }, &.{ q, k_row }, .plain);
            try b.copy(v_row, k_row, kvw);
        }
        const rope = self.ropeOf(kind);
        try b.rmsNorm(q, c.query_norm, q, .{ .rows = heads, .width = hd, .in_stride = hd, .out_stride = hd });
        try b.rope(q, rope.table, heads, hd, rope.dims, position, .split_half);
        try b.rmsNorm(k_row, c.key_norm, k_row, .{ .rows = kv_heads, .width = hd, .in_stride = hd, .out_stride = hd });
        try b.rope(k_row, rope.table, kv_heads, hd, rope.dims, position, .split_half);
        try b.rmsNorm(v_row, self.ones, v_row, .{ .rows = kv_heads, .width = hd, .in_stride = hd, .out_stride = hd });
        if (precision == .f16) try b.packHalf(&.{ .{ .dst = k_slot, .src = k_row, .count = kvw }, .{ .dst = v_slot, .src = v_row, .count = kvw } });
        const first = firstVisible(kind, position);
        const visible = position + 1 - first;
        const out = self.mixed_out.slice(0, qw * 4);
        try b.attentionDecode(self.stateSlice(cache.keys.range(first, visible)), self.stateSlice(cache.values.range(first, visible)), q, self.partials, out, .{ .query_heads = heads, .kv_heads = kv_heads, .key_width = hd, .value_width = hd, .visible = visible, .scale = 1.0, .precision = precision });
        try self.mm(layer.output, out, self.projected);
    }

    /// Chunked prefill: consumes `tokens` in chunks of at most `chunk`,
    /// one command buffer per chunk, with the projections and feed-forward
    /// batched through the matmul kernel and attention as one causal tiled
    /// dispatch per layer, with the window mask on sliding layers).
    /// Readbacks refer to the last token, as in `step`. The
    /// observer's `check` runs between layers of every chunk; its `layer`
    /// callback is per token by contract and is not supported here
    /// (`error.InvalidShape`): trace through `step`. Arithmetic order differs
    /// from `step` (tile accumulation), so results agree within a tolerance,
    /// not bit for bit; `generation-check --metal` measures it.
    pub fn prefill(self: *Plan, tokens: []const u32, logits: ?[]f32, greedy: ?*u32, topk: ?*sampling.TopK, penalties: ?sampling.Penalties, hidden_rows: ?[]f32, observer: ?Observer) !void {
        if (hidden_rows != null) return error.HiddenUnsupported;
        if (tokens.len == 0) return error.InvalidShape;
        if (observer) |o| if (o.layer != null) return error.InvalidShape;
        for (tokens) |t| if (t >= vocabulary) return error.InvalidTokenId;
        if (logits) |out| if (out.len != vocabulary) return error.InvalidShape;
        if (topk) |top| if (!std.math.isFinite(top.temperature) or top.temperature <= 0) return error.InvalidShape;
        // The whole prompt must fit: a prompt is never half-consumed.
        if (self.state.status != .ready) return error.SessionNotReady;
        if (tokens.len > self.state.capacity - self.state.position) return error.ContextFull;
        var offset: usize = 0;
        while (offset < tokens.len) {
            const count = @min(self.chunk, tokens.len - offset);
            const last = offset + count == tokens.len;
            try self.prefillChunk(tokens[offset..][0..count], if (last) logits else null, if (last) greedy else null, if (last) topk else null, if (last) penalties else null, observer);
            offset += count;
            // The whole prompt is one `step` for the loop's hooks, so this is
            // the only place a caller can learn how far a long prefill has got.
            if (observer) |o| if (o.progress) |call| try call(o.context, .{ .phase = .prefill, .position = offset, .target = tokens.len });
        }
    }

    fn prefillChunk(self: *Plan, tokens: []const u32, logits: ?[]f32, greedy: ?*u32, topk: ?*sampling.TopK, penalties: ?sampling.Penalties, observer: ?Observer) !void {
        const count = tokens.len;
        std.debug.assert(count >= 1 and count <= self.chunk);
        if (penalties) |p| try self.syncPenalties(p);
        try self.state.beginChunk(count);
        errdefer self.state.fail();
        const b = self.backend;
        try b.begin();
        errdefer if (b.recording) b.commit() catch {};
        const embedding = try self.weight(self.binding.token_embedding);
        const hidden = self.binding.config.embedding;
        for (tokens, 0..) |token, t| try b.embed(embedding.buffer, embedding.matrix, token, self.x_c.slice(t * hidden * 4, hidden * 4));
        try b.scale(self.x_c, count * hidden, self.binding.config.embeddingScale());
        const norm: metal.Backend.Norm = .{ .rows = count, .width = hidden, .in_stride = hidden, .out_stride = hidden };
        for (self.binding.active(), self.constants, 0..) |layer, c, il| {
            try b.rmsNorm(self.x_c, c.attention_norm, self.normalized_c, norm);
            try self.attentionChunk(layer, c, il, count);
            try b.rmsNorm(self.projected_c, c.post_attention_norm, self.projected_c, norm);
            try b.add(self.x_c, self.projected_c, count * hidden);
            try self.feedForwardChunk(layer, c, count);
            try b.addScale(self.x_c, self.projected_c, count * hidden, c.output_scale);
            if (observer) |o| if (o.check) |check| try check(o.context);
        }
        if (logits != null or greedy != null or topk != null) {
            const last_row = self.x_c.slice((count - 1) * hidden * 4, hidden * 4);
            try b.rmsNorm(last_row, self.output_norm, self.normalized, .{ .rows = 1, .width = hidden, .in_stride = hidden, .out_stride = hidden });
            try self.recordOutputs(greedy, topk, penalties);
        }
        try b.commit();
        try self.readOutputs(logits, greedy, topk, penalties != null);
        try self.state.commitChunk(count);
    }

    fn attentionChunk(self: *Plan, layer: model.Layer, c: LayerConstants, il: usize, count: usize) !void {
        const b = self.backend;
        const kind = layer.kind;
        const hd = layer.headSize();
        const kv_heads = layer.kv_heads;
        const qw = layer.queryWidth();
        const kvw = layer.kvWidth();
        const hidden = self.binding.config.embedding;
        const cache = self.state.layers[il].attention;
        const position = self.state.position;
        try self.mmRows(layer.query, self.normalized_c, hidden, self.q_c, qw, count);
        try self.mmRows(layer.key, self.normalized_c, hidden, self.k_c, kvw, count);
        if (layer.value) |value| try self.mmRows(value, self.normalized_c, hidden, self.v_c, kvw, count) else try b.copy(self.v_c, self.k_c, count * kvw);
        const rope = self.ropeOf(kind);
        try b.rmsNorm(self.q_c, c.query_norm, self.q_c, .{ .rows = count * heads, .width = hd, .in_stride = hd, .out_stride = hd });
        try b.ropeRows(self.q_c, rope.table, heads, hd, rope.dims, position, count, qw, .split_half);
        try b.rmsNorm(self.k_c, c.key_norm, self.k_c, .{ .rows = count * kv_heads, .width = hd, .in_stride = hd, .out_stride = hd });
        try b.ropeRows(self.k_c, rope.table, kv_heads, hd, rope.dims, position, count, kvw, .split_half);
        try b.rmsNorm(self.v_c, self.ones, self.v_c, .{ .rows = count * kv_heads, .width = hd, .in_stride = hd, .out_stride = hd });
        // The chunk's keys and values are contiguous in the cache; padding rows never leave the chunk buffers.
        const precision = cache.keys.precision;
        const k_rows = self.stateSlice(cache.keys.range(position, count));
        const v_rows = self.stateSlice(cache.values.range(position, count));
        var queries = self.q_c;
        switch (precision) {
            .f32 => {
                try b.copy(k_rows, self.k_c, count * kvw);
                try b.copy(v_rows, self.v_c, count * kvw);
            },
            .f16 => {
                try b.packHalf(&.{ .{ .dst = k_rows, .src = self.k_c, .count = count * kvw }, .{ .dst = v_rows, .src = self.v_c, .count = count * kvw } });
                try b.packHalf(&.{.{ .dst = self.q_c_h, .src = self.q_c, .count = metal.Backend.attentionChunkRows(count) * qw }});
                queries = self.q_c_h;
            },
        }
        // One causal tiled dispatch over the whole chunk. The cache is
        // sliced at the earliest key the chunk's first row can see; the
        // window mask hides the rest per row on sliding layers.
        const first = firstVisible(kind, position);
        const total = position + count - first;
        try b.attentionChunk(self.stateSlice(cache.keys.range(first, total)), self.stateSlice(cache.values.range(first, total)), queries, self.mixed_out_c, .{ .query_heads = heads, .kv_heads = kv_heads, .key_width = hd, .value_width = hd, .position = position - first, .count = count, .q_stride = qw, .out_stride = qw, .scale = 1.0, .precision = precision, .window = if (kind == .sliding) model.window else 0 });
        try self.mmRows(layer.output, self.mixed_out_c, qw, self.projected_c, hidden, count);
    }
};

test "the visible window of a token is a suffix of the cache rows" {
    try std.testing.expectEqual(@as(usize, 0), Plan.firstVisible(.sliding, 0));
    try std.testing.expectEqual(@as(usize, 0), Plan.firstVisible(.sliding, model.window - 1));
    try std.testing.expectEqual(@as(usize, 1), Plan.firstVisible(.sliding, model.window));
    try std.testing.expectEqual(@as(usize, 5000 + 1 - model.window), Plan.firstVisible(.sliding, 5000));
    try std.testing.expectEqual(@as(usize, 0), Plan.firstVisible(.global, 5000));
}
