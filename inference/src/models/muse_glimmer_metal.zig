//! GPU-resident execution of the pinned Muse Glimmer 30B schedule. One
//! command buffer per token (`step`) or per prompt chunk (`prefill`): the
//! CPU supplies token ids and reads back logits, a greedy token, or a partial
//! top-k; every activation, norm, gate, and cache write stays on the GPU.
//! The schedule is `muse_glimmer_runtime.zig`'s, which remains the CPU
//! reference these results are compared against (docs/reference/muse-glimmer.md).
//!
//! What this plan asks of the backend beyond the Gemma plan, all of it
//! decided by the forward pass: RoPE with adjacent pairing over the whole
//! 128-wide head on sliding layers only (global layers carry no position
//! encoding), the weightless embedding norm (a ones vector as the weight),
//! the post norms at their own epsilon, the sigmoid attention gate as an
//! epilogue on the attention output (`sigmoidGate`, the Qwen3.5 path), the
//! untied head with the logit scale before the soft-cap, and the 2048
//! window as a cache-row slice on decode and a `window` mask on prefill.
//!
//! Ownership: the plan borrows the mapped weights and the `Backend`; it owns
//! the session and the GPU buffers it creates. Session memory is wrapped once
//! as a shared buffer, so `reset()` is a CPU memset that is only valid after
//! `commit()` — which the synchronous backend guarantees. Every attention
//! cache is allocated for the full capacity, sliding layers included (the
//! reference does the same); a ring layout for the windowed layers is a
//! session-layout unit of its own (roadmap).
const std = @import("std");
const model = @import("muse_glimmer.zig");
const weights = @import("../runtime/weights.zig");
const session = @import("../runtime/session.zig");
const sampling = @import("../sampling/root.zig");
const metal = @import("../backends/metal/root.zig");
const cpu = @import("../backends/cpu/root.zig");
const Tensor = @import("../formats/gguf.zig").Tensor;
const Buffer = metal.Buffer;

pub const Observer = @import("../runtime/observer.zig").Observer;

pub const vocabulary = model.vocabulary;
const hidden = model.embedding;
const ffn = model.feed_forward;
const heads = model.heads;
const kv_heads = model.kv_heads;
const hd = model.head_size;
const q_width = model.query_width;
const kv_width = model.kv_width;

const LayerConstants = struct {
    attention_norm: Buffer,
    post_attention_norm: Buffer,
    ffn_norm: Buffer,
    post_ffn_norm: Buffer,
    query_norm: Buffer,
    key_norm: Buffer,
};

/// A weight matrix ready for dispatch: its GPU range and CPU-side descriptor.
const Weight = struct { buffer: Buffer, matrix: cpu.Matrix };

pub const Plan = struct {
    alloc: std.mem.Allocator,
    backend: *metal.Backend,
    view: weights.View,
    binding: model.Binding,
    state: session.Session,
    state_buffer: Buffer,
    constants: []LayerConstants,
    output_norm: Buffer,
    /// A row of ones: the embedding norm has no weight.
    ones: Buffer, // hidden
    /// (cos, sin) table: base 5e5 over the 128-wide head, sliding layers only.
    rope_table: Buffer,
    // Activations (F32 counts in comments).
    x: Buffer, // hidden
    normalized: Buffer, // hidden
    projected: Buffer, // hidden
    gate: Buffer, // ffn
    up: Buffer, // ffn
    /// The query and the attention gate's pre-activation share one buffer
    /// (`q` then `attention_gate`), and the scratch key and value rows
    /// another, so the four attention projections fit the segment kernel's
    /// seven bindings (input, four weights, two outputs) and merge.
    qg: Buffer, // 2 × q_width
    q: Buffer, // q_width, a slice of qg
    /// One value per query channel, a slice of qg.
    attention_gate: Buffer, // q_width
    /// F32 key/value rows of the current token when the cache is F16:
    /// projected, normalized, and rotated here, then packed into the slot.
    /// With an F32 cache the projections write the slot directly.
    kv: Buffer, // 2 × kv_width
    k: Buffer, // kv_width, a slice of kv
    v: Buffer, // kv_width, a slice of kv
    mixed_out: Buffer, // q_width
    /// Flash-decoding partials: `[32][splits][2 + 128]`.
    partials: Buffer,
    logits: Buffer, // vocabulary
    argmax_values: Buffer,
    argmax_indices: Buffer,
    argmax_result: Buffer,
    topk: metal.Backend.TopKBuffers,
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
    attention_gate_c: Buffer, // padded × q_width
    /// Half copy of `q_c` for the F16 chunk attention (its operands share one type).
    q_c_h: Buffer, // padded × q_width halves
    mixed_out_c: Buffer, // padded × q_width

    /// `chunk` bounds the tokens one `prefill` command buffer processes (and
    /// sizes its activation buffers: about 0.4 MB per token). `kv` is the
    /// attention cache precision of every layer.
    pub fn init(alloc: std.mem.Allocator, backend: *metal.Backend, view: weights.View, binding: model.Binding, capacity: usize, chunk: usize, kv: session.Precision, checkpoint: bool, draft: bool) !Plan {
        _ = draft;
        if (chunk == 0 or chunk > 4096) return error.InvalidShape;
        var layouts: [model.layer_count]session.Layout = undefined;
        for (&layouts) |*layout| layout.* = .{ .attention = .{ .key_row = kv_width, .value_row = kv_width, .precision = kv } };
        var state = try session.Session.init(alloc, &layouts, capacity, checkpoint);
        errdefer state.deinit();
        const constants = try alloc.alloc(LayerConstants, model.layer_count);
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
            };
        }
        self.output_norm = try self.constant(binding.output_norm);
        self.ones = try backend.create(hidden * 4);
        @memset(self.ones.floats(), 1);
        self.rope_table = try backend.create(capacity * (hd / 2) * 8);
        try metal.Backend.ropeTable(self.rope_table, capacity, hd, model.rope_base, null);
        self.x = try backend.create(hidden * 4);
        self.normalized = try backend.create(hidden * 4);
        self.projected = try backend.create(hidden * 4);
        self.gate = try backend.create(ffn * 4);
        self.up = try backend.create(ffn * 4);
        self.qg = try backend.create(2 * q_width * 4);
        self.q = self.qg.slice(0, q_width * 4);
        self.attention_gate = self.qg.slice(q_width * 4, q_width * 4);
        self.kv = try backend.create(2 * kv_width * 4);
        self.k = self.kv.slice(0, kv_width * 4);
        self.v = self.kv.slice(kv_width * 4, kv_width * 4);
        self.mixed_out = try backend.create(q_width * 4);
        self.partials = try backend.create(metal.Backend.attentionDecodePartials(heads, hd) * 4);
        self.logits = try backend.create(vocabulary * 4);
        self.argmax_values = try backend.create(metal.Backend.argmax_partials * 4);
        self.argmax_indices = try backend.create(metal.Backend.argmax_partials * 4);
        self.argmax_result = try backend.create(4);
        self.topk = try backend.topkBuffers(sampling.TopK.capacity);
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
        self.attention_gate_c = try backend.create(n * q_width * 4);
        self.q_c_h = try backend.create(n * q_width * 2);
        self.mixed_out_c = try backend.create(n * q_width * 4);
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
    fn constant(self: *Plan, tensor: *const Tensor) !Buffer {
        const values = try self.view.vector(self.alloc, tensor);
        defer self.alloc.free(values);
        if (values.len == 0) return error.InvalidShape;
        const buffer = try self.backend.create(values.len * 4);
        @memcpy(buffer.floats(), values);
        return buffer;
    }
    /// Tensors are wrapped one at a time (cached by address), as in the other plans.
    fn weight(self: *Plan, tensor: *const Tensor) !Weight {
        const matrix = try self.view.matrix(tensor);
        return .{ .buffer = try self.backend.wrap(matrix.bytes), .matrix = matrix };
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
    /// A norm spec over `rows` rows of the residual width at `eps`.
    fn normOf(rows: usize, eps: f32) metal.Backend.Norm {
        return .{ .rows = rows, .width = hidden, .in_stride = hidden, .out_stride = hidden, .eps = eps };
    }
    /// The per-head norm spec over `rows` heads.
    fn headNorm(rows: usize) metal.Backend.Norm {
        return .{ .rows = rows, .width = hd, .in_stride = hd, .out_stride = hd, .eps = model.rms_epsilon };
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
    /// `layer` callback forces a commit after every layer (52 command buffers
    /// per token) and is for numerical traces only.
    pub fn step(self: *Plan, token: u32, logits: ?[]f32, greedy: ?*u32, topk: ?*sampling.TopK, observer: ?Observer) !void {
        if (token >= vocabulary) return error.InvalidTokenId;
        if (logits) |out| if (out.len != vocabulary) return error.InvalidShape;
        if (topk) |top| if (!std.math.isFinite(top.temperature) or top.temperature <= 0) return error.InvalidShape;
        try self.state.begin();
        errdefer self.state.fail();
        const b = self.backend;
        try b.begin();
        errdefer if (b.recording) b.commit() catch {};
        const embedding = try self.weight(self.binding.token_embedding);
        try b.embed(embedding.buffer, embedding.matrix, token, self.x);
        try b.rmsNorm(self.x, self.ones, self.x, normOf(1, model.rms_epsilon));
        for (self.binding.active(), self.constants, 0..) |layer, c, il| {
            try b.rmsNorm(self.x, c.attention_norm, self.normalized, normOf(1, model.rms_epsilon));
            try self.attention(layer, c, il);
            try b.rmsNorm(self.projected, c.post_attention_norm, self.projected, normOf(1, model.post_norm_epsilon));
            try b.add(self.x, self.projected, hidden);
            try self.feedForward(layer, c);
            try b.add(self.x, self.projected, hidden);
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
            try b.rmsNorm(self.x, self.output_norm, self.normalized, normOf(1, model.rms_epsilon));
            try self.recordOutputs(greedy, topk);
        }
        try b.commit();
        try self.readOutputs(logits, greedy, topk);
        try self.state.commit();
    }

    /// The feed-forward block over the residual `x` (the attention output
    /// already added) into `projected`, post-normed and ready to add: the
    /// CPU reference's `feedForward`.
    fn feedForward(self: *Plan, layer: model.Layer, c: LayerConstants) !void {
        const b = self.backend;
        try b.rmsNorm(self.x, c.ffn_norm, self.normalized, normOf(1, model.rms_epsilon));
        try self.projections(&.{ layer.ffn_gate, layer.ffn_up }, &.{ self.gate, self.up }, .silu_mul_pair);
        try self.mm(layer.ffn_down, self.gate, self.projected);
        try b.rmsNorm(self.projected, c.post_ffn_norm, self.projected, normOf(1, model.post_norm_epsilon));
    }

    /// `feedForward` over the chunk's `count` rows of `x_c` into `projected_c`.
    fn feedForwardChunk(self: *Plan, layer: model.Layer, c: LayerConstants, count: usize) !void {
        const b = self.backend;
        try b.rmsNorm(self.x_c, c.ffn_norm, self.normalized_c, normOf(count, model.rms_epsilon));
        try self.mmRows(layer.ffn_gate, self.normalized_c, hidden, self.gate_c, ffn, count);
        try self.mmRows(layer.ffn_up, self.normalized_c, hidden, self.up_c, ffn, count);
        try b.siluMul(self.gate_c, self.up_c, count * ffn);
        try self.mmRows(layer.ffn_down, self.gate_c, ffn, self.projected_c, hidden, count);
        try b.rmsNorm(self.projected_c, c.post_ffn_norm, self.projected_c, normOf(count, model.post_norm_epsilon));
    }

    /// Records the untied head for the normalized last-token row in
    /// `self.normalized`, the logit scale, the soft-cap, and whichever
    /// readbacks were requested. The reference folds the scale into the
    /// tanh argument; here it is one rounding before it.
    fn recordOutputs(self: *Plan, greedy: ?*u32, topk: ?*sampling.TopK) !void {
        const b = self.backend;
        try self.mm(self.binding.output, self.normalized, self.logits);
        try b.scale(self.logits, vocabulary, model.logit_scale);
        try b.softcap(self.logits, vocabulary, model.final_softcap);
        if (greedy != null) try b.argmax(self.logits, vocabulary, self.argmax_values, self.argmax_indices, self.argmax_result);
        if (topk) |top| try b.topk(self.logits, vocabulary, sampling.TopK.capacity, top.temperature, self.topk);
    }
    /// Copies the requested readbacks out after `commit`.
    fn readOutputs(self: *Plan, logits: ?[]f32, greedy: ?*u32, topk: ?*sampling.TopK) !void {
        if (logits) |out| try self.readLogits(out);
        if (greedy) |out| {
            const id = @as(*const u32, @ptrCast(@alignCast(self.argmax_result.host))).*;
            if (id >= vocabulary) return error.NonFiniteResult;
            out.* = id;
        }
        if (topk) |top| {
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
    /// until the next step; the sampler's fallback path uses it.
    pub fn readLogits(self: *Plan, out: []f32) !void {
        if (out.len != vocabulary) return error.InvalidShape;
        @memcpy(out, self.logits.floats());
        for (out) |v| if (!std.math.isFinite(v)) return error.NonFiniteResult;
    }

    fn attention(self: *Plan, layer: model.Layer, c: LayerConstants, il: usize) !void {
        const b = self.backend;
        const cache = self.state.layers[il].attention;
        const position = self.state.position;
        const precision = cache.keys.precision;
        const k_slot = self.stateSlice(cache.keys.range(position, 1));
        const v_slot = self.stateSlice(cache.values.range(position, 1));
        // An F32 cache takes the projections directly; an F16 cache takes them
        // through F32 scratch rows and one pack dispatch.
        const k_row = if (precision == .f32) k_slot else self.k;
        const v_row = if (precision == .f32) v_slot else self.v;
        try self.projections(&.{ layer.query, layer.key, layer.value, layer.gate }, &.{ self.q, k_row, v_row, self.attention_gate }, .plain);
        try b.rmsNorm(self.q, c.query_norm, self.q, headNorm(heads));
        try b.rmsNorm(k_row, c.key_norm, k_row, headNorm(kv_heads));
        if (layer.kind == .sliding) {
            try b.rope(self.q, self.rope_table, heads, hd, hd, position, .adjacent);
            try b.rope(k_row, self.rope_table, kv_heads, hd, hd, position, .adjacent);
        }
        if (precision == .f16) try b.packHalf(&.{ .{ .dst = k_slot, .src = k_row, .count = kv_width }, .{ .dst = v_slot, .src = v_row, .count = kv_width } });
        const first = firstVisible(layer.kind, position);
        const visible = position + 1 - first;
        try b.attentionDecode(self.stateSlice(cache.keys.range(first, visible)), self.stateSlice(cache.values.range(first, visible)), self.q, self.partials, self.mixed_out, .{ .query_heads = heads, .kv_heads = kv_heads, .key_width = hd, .value_width = hd, .visible = visible, .scale = model.attention_scale, .precision = precision });
        try b.sigmoidGate(self.mixed_out, self.attention_gate, heads, hd, hd, 0);
        try self.mm(layer.output, self.mixed_out, self.projected);
    }

    /// Chunked prefill: consumes `tokens` in chunks of at most `chunk`,
    /// one command buffer per chunk, with the projections and feed-forward
    /// batched through the matmul kernel and attention as one causal tiled
    /// dispatch per layer (the window mask on sliding layers). Readbacks
    /// refer to the last token, as in `step`. The observer's `check` runs
    /// between layers of every chunk; its `layer` callback is per token by
    /// contract and is not supported here (`error.InvalidShape`): trace
    /// through `step`. Arithmetic order differs from `step` (tile
    /// accumulation), so results agree within a tolerance, not bit for bit;
    /// `generation-check --metal` measures it.
    pub fn prefill(self: *Plan, tokens: []const u32, logits: ?[]f32, greedy: ?*u32, topk: ?*sampling.TopK, observer: ?Observer) !void {
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
            try self.prefillChunk(tokens[offset..][0..count], if (last) logits else null, if (last) greedy else null, if (last) topk else null, observer);
            offset += count;
            // The whole prompt is one `step` for the loop's hooks, so this is
            // the only place a caller can learn how far a long prefill has got.
            if (observer) |o| if (o.progress) |call| try call(o.context, .{ .phase = .prefill, .position = offset, .target = tokens.len });
        }
    }

    fn prefillChunk(self: *Plan, tokens: []const u32, logits: ?[]f32, greedy: ?*u32, topk: ?*sampling.TopK, observer: ?Observer) !void {
        const count = tokens.len;
        std.debug.assert(count >= 1 and count <= self.chunk);
        try self.state.beginChunk(count);
        errdefer self.state.fail();
        const b = self.backend;
        try b.begin();
        errdefer if (b.recording) b.commit() catch {};
        const embedding = try self.weight(self.binding.token_embedding);
        for (tokens, 0..) |token, t| try b.embed(embedding.buffer, embedding.matrix, token, self.x_c.slice(t * hidden * 4, hidden * 4));
        try b.rmsNorm(self.x_c, self.ones, self.x_c, normOf(count, model.rms_epsilon));
        for (self.binding.active(), self.constants, 0..) |layer, c, il| {
            try b.rmsNorm(self.x_c, c.attention_norm, self.normalized_c, normOf(count, model.rms_epsilon));
            try self.attentionChunk(layer, c, il, count);
            try b.rmsNorm(self.projected_c, c.post_attention_norm, self.projected_c, normOf(count, model.post_norm_epsilon));
            try b.add(self.x_c, self.projected_c, count * hidden);
            try self.feedForwardChunk(layer, c, count);
            try b.add(self.x_c, self.projected_c, count * hidden);
            if (observer) |o| if (o.check) |check| try check(o.context);
        }
        if (logits != null or greedy != null or topk != null) {
            const last_row = self.x_c.slice((count - 1) * hidden * 4, hidden * 4);
            try b.rmsNorm(last_row, self.output_norm, self.normalized, normOf(1, model.rms_epsilon));
            try self.recordOutputs(greedy, topk);
        }
        try b.commit();
        try self.readOutputs(logits, greedy, topk);
        try self.state.commitChunk(count);
    }

    fn attentionChunk(self: *Plan, layer: model.Layer, c: LayerConstants, il: usize, count: usize) !void {
        const b = self.backend;
        const cache = self.state.layers[il].attention;
        const position = self.state.position;
        try self.mmRows(layer.query, self.normalized_c, hidden, self.q_c, q_width, count);
        try self.mmRows(layer.key, self.normalized_c, hidden, self.k_c, kv_width, count);
        try self.mmRows(layer.value, self.normalized_c, hidden, self.v_c, kv_width, count);
        try self.mmRows(layer.gate, self.normalized_c, hidden, self.attention_gate_c, q_width, count);
        try b.rmsNorm(self.q_c, c.query_norm, self.q_c, headNorm(count * heads));
        try b.rmsNorm(self.k_c, c.key_norm, self.k_c, headNorm(count * kv_heads));
        if (layer.kind == .sliding) {
            try b.ropeRows(self.q_c, self.rope_table, heads, hd, hd, position, count, q_width, .adjacent);
            try b.ropeRows(self.k_c, self.rope_table, kv_heads, hd, hd, position, count, kv_width, .adjacent);
        }
        // The chunk's keys and values are contiguous in the cache; padding rows never leave the chunk buffers.
        const precision = cache.keys.precision;
        const k_rows = self.stateSlice(cache.keys.range(position, count));
        const v_rows = self.stateSlice(cache.values.range(position, count));
        var queries = self.q_c;
        switch (precision) {
            .f32 => {
                try b.copy(k_rows, self.k_c, count * kv_width);
                try b.copy(v_rows, self.v_c, count * kv_width);
            },
            .f16 => {
                try b.packHalf(&.{ .{ .dst = k_rows, .src = self.k_c, .count = count * kv_width }, .{ .dst = v_rows, .src = self.v_c, .count = count * kv_width } });
                try b.packHalf(&.{.{ .dst = self.q_c_h, .src = self.q_c, .count = metal.Backend.attentionChunkRows(count) * q_width }});
                queries = self.q_c_h;
            },
        }
        // One causal tiled dispatch over the whole chunk. The cache is
        // sliced at the earliest key the chunk's first row can see; the
        // window mask hides the rest per row on sliding layers.
        const first = firstVisible(layer.kind, position);
        const total = position + count - first;
        try b.attentionChunk(self.stateSlice(cache.keys.range(first, total)), self.stateSlice(cache.values.range(first, total)), queries, self.mixed_out_c, .{ .query_heads = heads, .kv_heads = kv_heads, .key_width = hd, .value_width = hd, .position = position - first, .count = count, .q_stride = q_width, .out_stride = q_width, .scale = model.attention_scale, .precision = precision, .window = if (layer.kind == .sliding) model.window else 0 });
        // Rows are contiguous at `q_width`, so the chunk's gates are one
        // long run of heads.
        try b.sigmoidGate(self.mixed_out_c, self.attention_gate_c, count * heads, hd, hd, 0);
        try self.mmRows(layer.output, self.mixed_out_c, q_width, self.projected_c, hidden, count);
    }
};

test "the visible window of a token is a suffix of the cache rows" {
    try std.testing.expectEqual(@as(usize, 0), Plan.firstVisible(.sliding, 0));
    try std.testing.expectEqual(@as(usize, 0), Plan.firstVisible(.sliding, model.window - 1));
    try std.testing.expectEqual(@as(usize, 1), Plan.firstVisible(.sliding, model.window));
    try std.testing.expectEqual(@as(usize, 5000 + 1 - model.window), Plan.firstVisible(.sliding, 5000));
    try std.testing.expectEqual(@as(usize, 0), Plan.firstVisible(.global, 5000));
}
