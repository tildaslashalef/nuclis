//! EmbeddingGemma 2 on Metal: embeddinggemma_runtime.zig's forward over
//! packed batches. Inputs sit back to back, each starting on an 8-row
//! boundary; the rows that pad one to the boundary are a zero-input segment
//! of their own. Attention reads each row's `[begin, end)` from a bounds
//! buffer and RoPE a per-row table at the row's position in its input. With
//! every SIMD group inside one input, an input's vector does not depend on
//! what it is packed with, bit for bit.
//!
//! Matrices are the mapping's Q8_0, BF16 or F32 bytes wrapped in place and
//! read by the generic F32 matmul tile (`matmulTile`; no half operands, the
//! card's f16 warning); residual, activations and attention are F32. Token
//! rows are decoded and pooling done on the host with the CPU's own
//! functions. The plan owns its `Backend`; the mapping and binding must
//! outlive it. Contract: docs/models/embeddinggemma.md.
const std = @import("std");
const metal = @import("../backends/metal/root.zig");
const cpu = @import("../backends/cpu/root.zig");
const weights = @import("../runtime/weights.zig");
const gguf = @import("../formats/gguf.zig");
const model = @import("embeddinggemma.zig");
const runtime = @import("embeddinggemma_runtime.zig");

const Backend = metal.Backend;
const Buffer = metal.Buffer;

/// Rows one batch holds, padding included: one input of `max_tokens`.
/// Activations cost about 34 KB a row.
pub const max_rows = model.max_tokens;
/// Rows an input of `tokens` rows takes in a batch.
pub fn batchRows(tokens: usize) usize {
    return std.mem.alignForward(usize, tokens, segment_alignment);
}
/// The SIMD-group height of the attention kernel.
const segment_alignment = 8;

const d = model.config.embedding;
const w = model.config.per_layer_input;
const ff = model.config.feed_forward;
const heads = model.config.heads;
const out_width = model.config.embedding_out;
const eps = model.rms_epsilon;

const Weight = struct { buffer: Buffer, matrix: cpu.Matrix };
const Layer = struct {
    kind: model.Kind,
    attention_norm: Buffer,
    post_attention_norm: Buffer,
    ffn_norm: Buffer,
    post_ffn_norm: Buffer,
    per_layer_post_norm: Buffer,
    query_norm: Buffer,
    key_norm: Buffer,
    output_scale: f32,
    query: Weight,
    key: Weight,
    value: Weight,
    output: Weight,
    ffn_gate: Weight,
    ffn_up: Weight,
    ffn_down: Weight,
    per_layer_gate: Weight,
    per_layer_projection: Weight,
    /// This layer's rows of the model's `per_layer_model_proj`.
    per_layer_input: Weight,
};

pub const Plan = struct {
    gpa: std.mem.Allocator,
    backend: Backend,
    view: weights.View,
    binding: *const model.Binding,
    layers: [model.layer_count]Layer,
    per_layer_norm: Buffer,
    output_norm: Buffer,
    output: Weight,
    /// The weight of V's unweighted norm.
    ones: Buffer,
    // Activations, `Backend.matmulPadded(max_rows)` rows each.
    x: Buffer,
    x0: Buffer,
    h: Buffer,
    /// Queries, then the feed-forward gate.
    q: Buffer,
    k: Buffer,
    v: Buffer,
    /// Attention's output, then the feed-forward up projection.
    attended: Buffer,
    ple: Buffer,
    pe: Buffer,
    projected: Buffer,
    /// Per row: `[begin, end)` of its segment, as u32 pairs.
    bounds: Buffer,
    /// Per row, (cos, sin) pairs at its position: global and sliding bases.
    rope_global: Buffer,
    rope_sliding: Buffer,
    /// The same pairs for positions 0..max_rows, copied per segment.
    table_global: Buffer,
    table_sliding: Buffer,

    /// Builds the plan: a new Metal backend, the matrices wrapped, the norms
    /// and tables made. `view`'s mapping and `binding` must outlive it.
    pub fn create(gpa: std.mem.Allocator, view: weights.View, binding: *const model.Binding) !*Plan {
        const self = try gpa.create(Plan);
        errdefer gpa.destroy(self);
        var diagnostic: [512]u8 = undefined;
        self.backend = try Backend.init(gpa, &diagnostic);
        errdefer self.backend.deinit();
        self.gpa = gpa;
        self.view = view;
        self.binding = binding;
        const ple_all = try self.matrix(binding.per_layer_projection);
        if (ple_all.matrix.rows != model.layer_count * w or ple_all.matrix.columns != d) return error.InvalidShape;
        const ple_stride = ple_all.matrix.bytes.len / ple_all.matrix.rows;
        for (&self.layers, binding.layers, 0..) |*l, layer, il| l.* = .{
            .kind = layer.kind,
            .attention_norm = try self.vector(layer.attention_norm),
            .post_attention_norm = try self.vector(layer.post_attention_norm),
            .ffn_norm = try self.vector(layer.ffn_norm),
            .post_ffn_norm = try self.vector(layer.post_ffn_norm),
            .per_layer_post_norm = try self.vector(layer.per_layer_post_norm),
            .query_norm = try self.vector(layer.query_norm),
            .key_norm = try self.vector(layer.key_norm),
            .output_scale = try view.scalar(layer.output_scale, 0),
            .query = try self.matrix(layer.query),
            .key = try self.matrix(layer.key),
            .value = try self.matrix(layer.value),
            .output = try self.matrix(layer.output),
            .ffn_gate = try self.matrix(layer.ffn_gate),
            .ffn_up = try self.matrix(layer.ffn_up),
            .ffn_down = try self.matrix(layer.ffn_down),
            .per_layer_gate = try self.matrix(layer.per_layer_gate),
            .per_layer_projection = try self.matrix(layer.per_layer_projection),
            .per_layer_input = .{
                .buffer = ple_all.buffer.slice(il * w * ple_stride, w * ple_stride),
                .matrix = .{ .encoding = ple_all.matrix.encoding, .rows = w, .columns = d, .bytes = ple_all.matrix.bytes[il * w * ple_stride ..][0 .. w * ple_stride] },
            },
        };
        self.per_layer_norm = try self.vector(binding.per_layer_norm);
        self.output_norm = try self.vector(binding.output_norm);
        self.output = try self.matrix(binding.output);

        const global = model.Kind.global.headSize();
        const sliding = model.Kind.sliding.headSize();
        self.ones = try self.created(global);
        @memset(self.ones.floats()[0..global], 1);
        const rows = Backend.matmulPadded(max_rows);
        const q_max = heads * global;
        self.x = try self.created(rows * d);
        self.x0 = try self.created(rows * d);
        self.h = try self.created(rows * d);
        self.q = try self.created(rows * @max(q_max, ff));
        self.k = try self.created(rows * d);
        self.v = try self.created(rows * d);
        self.attended = try self.created(rows * @max(q_max, ff));
        self.ple = try self.created(rows * w);
        self.pe = try self.created(rows * w);
        self.projected = try self.created(rows * out_width);
        self.bounds = try self.created(max_rows * 2);
        self.rope_global = try self.created(max_rows * global);
        self.rope_sliding = try self.created(max_rows * sliding);
        self.table_global = try self.created(max_rows * global);
        self.table_sliding = try self.created(max_rows * sliding);
        try Backend.ropeTable(self.table_global, max_rows, global, model.Kind.global.ropeBase(), null);
        try Backend.ropeTable(self.table_sliding, max_rows, sliding, model.Kind.sliding.ropeBase(), null);
        return self;
    }

    pub fn destroy(self: *Plan) void {
        const gpa = self.gpa;
        // Device buffers belong to the backend and go with it.
        self.backend.deinit();
        gpa.destroy(self);
    }

    /// A zeroed F32 buffer of `floats` values, the backend's.
    fn created(self: *Plan, floats: usize) !Buffer {
        const buffer = try self.backend.create(floats * 4);
        @memset(buffer.floats()[0..floats], 0);
        return buffer;
    }
    fn vector(self: *Plan, tensor: *const gguf.Tensor) !Buffer {
        const values = try self.view.vector(self.gpa, tensor);
        defer self.gpa.free(values);
        const buffer = try self.backend.create(values.len * 4);
        @memcpy(buffer.floats()[0..values.len], values);
        return buffer;
    }
    fn matrix(self: *Plan, tensor: *const gguf.Tensor) !Weight {
        const m = try self.view.matrix(tensor);
        if (!model.executableEncoding(m.encoding)) return error.UnsupportedEncoding;
        if (m.rows % 8 != 0 or m.columns % 64 != 0) return error.UnsupportedConfig;
        return .{ .buffer = try self.backend.wrap(m.bytes), .matrix = m };
    }

    /// The unit vector of each input into `vectors[i]` (`embedding_out`
    /// values): one command buffer for the batch. With an `observer` (one
    /// input only), one per stage so the host can read each stage's rows.
    pub fn run(self: *Plan, inputs: []const runtime.Input, vectors: []const []f32, observer: ?runtime.Observer) !void {
        if (inputs.len == 0 or inputs.len != vectors.len) return error.InvalidShape;
        if (observer != null and inputs.len != 1) return error.InvalidShape;
        var n: usize = 0;
        for (inputs, vectors) |input, vec| {
            if (input.tokens.len == 0 or input.tokens.len > model.max_tokens or vec.len != out_width) return error.InvalidShape;
            n += batchRows(input.tokens.len);
        }
        if (n > max_rows) return error.SequenceTooLong;

        // Host inputs: x0, each row's segment bounds, and its rotary rows.
        const x0 = self.x0.floats();
        var begin: usize = 0;
        for (inputs) |input| {
            const len = input.tokens.len;
            try runtime.inputRows(self.view, self.binding, input, x0[begin * d ..][0 .. len * d]);
            self.segment(begin, len);
            const pad = batchRows(len) - len;
            if (pad != 0) {
                @memset(x0[(begin + len) * d ..][0 .. pad * d], 0);
                self.segment(begin + len, pad);
            }
            begin += len + pad;
        }

        const b = &self.backend;
        const norm: Backend.Norm = .{ .rows = n, .width = d, .in_stride = d, .out_stride = d, .eps = eps };
        try b.begin();
        errdefer b.discard();
        try b.copy(self.x, self.x0, n * d);
        try self.stage(observer, .input, self.x0, d, inputs);
        for (self.layers, 0..) |l, il| {
            const hd = l.kind.headSize();
            const kvh = l.kind.kvHeads();
            const qw = heads * hd;
            const kvw = kvh * hd;
            const rope = if (l.kind == .global) self.rope_global else self.rope_sliding;
            // Attention.
            try b.rmsNorm(self.x, l.attention_norm, self.h, norm);
            try b.matmulTile(l.query.buffer, l.query.matrix, self.h, d, self.q, qw, n);
            try b.matmulTile(l.key.buffer, l.key.matrix, self.h, d, self.k, kvw, n);
            try b.matmulTile(l.value.buffer, l.value.matrix, self.h, d, self.v, kvw, n);
            try b.rmsNorm(self.q, l.query_norm, self.q, .{ .rows = n * heads, .width = hd, .in_stride = hd, .out_stride = hd, .eps = eps });
            try b.rmsNorm(self.k, l.key_norm, self.k, .{ .rows = n * kvh, .width = hd, .in_stride = hd, .out_stride = hd, .eps = eps });
            try b.rmsNorm(self.v, self.ones, self.v, .{ .rows = n * kvh, .width = hd, .in_stride = hd, .out_stride = hd, .eps = eps });
            try b.ropeRows(self.q, rope, heads, hd, hd, 0, n, qw, .split_half);
            try b.ropeRows(self.k, rope, kvh, hd, hd, 0, n, kvw, .split_half);
            try b.attentionSegmentsGrouped(self.q, self.k, self.v, self.attended, self.bounds, .{
                .query_heads = heads,
                .kv_heads = kvh,
                .width = hd,
                .rows = n,
                .q_stride = qw,
                .kv_stride = kvw,
                .out_stride = qw,
                .scale = 1,
                .window = if (l.kind == .sliding) model.config.half_window else null,
            });
            try b.matmulTile(l.output.buffer, l.output.matrix, self.attended, qw, self.h, d, n);
            try b.rmsNormAdd(self.x, self.h, l.post_attention_norm, 1, norm);
            // Feed-forward: the gate in `q`, the up projection in `attended`.
            try b.rmsNorm(self.x, l.ffn_norm, self.h, norm);
            try b.matmulTile(l.ffn_gate.buffer, l.ffn_gate.matrix, self.h, d, self.q, ff, n);
            try b.matmulTile(l.ffn_up.buffer, l.ffn_up.matrix, self.h, d, self.attended, ff, n);
            try b.geluMul(self.q, self.attended, n * ff);
            try b.matmulTile(l.ffn_down.buffer, l.ffn_down.matrix, self.q, ff, self.h, d, n);
            try b.rmsNormAdd(self.x, self.h, l.post_ffn_norm, 1, norm);
            // Per-layer input: ple = rmsnorm(P[l] · x0 / √d) · per_layer_norm.
            try b.matmulTile(l.per_layer_input.buffer, l.per_layer_input.matrix, self.x0, d, self.ple, w, n);
            try b.scale(self.ple, n * w, 1 / @sqrt(@as(f32, d)));
            try b.rmsNorm(self.ple, self.per_layer_norm, self.ple, .{ .rows = n, .width = w, .in_stride = w, .out_stride = w, .eps = eps });
            try self.stage(observer, .{ .per_layer = il }, self.ple, w, inputs);
            try b.matmulTile(l.per_layer_gate.buffer, l.per_layer_gate.matrix, self.x, d, self.pe, w, n);
            try b.geluMul(self.pe, self.ple, n * w);
            try b.matmulTile(l.per_layer_projection.buffer, l.per_layer_projection.matrix, self.pe, w, self.h, d, n);
            try b.rmsNormAdd(self.x, self.h, l.per_layer_post_norm, l.output_scale, norm);
            try self.stage(observer, .{ .layer = il }, self.x, d, inputs);
        }
        try b.rmsNorm(self.x, self.output_norm, self.h, norm);
        try self.stage(observer, .final_norm, self.h, d, inputs);
        try b.matmulTile(self.output.buffer, self.output.matrix, self.h, d, self.projected, out_width, n);
        try b.commit();
        if (observer) |o| o.record(o.context, .projected, self.projected.floats()[0 .. inputs[0].tokens.len * out_width]);

        const projected = self.projected.floats();
        begin = 0;
        for (inputs, vectors) |input, vec| {
            try runtime.pool(projected[begin * out_width ..], input.tokens.len, vec);
            begin += batchRows(input.tokens.len);
        }
    }

    /// Rows `[begin, begin + len)` form one segment: their bounds and their
    /// rotary rows at positions 0..len.
    fn segment(self: *Plan, begin: usize, len: usize) void {
        const pairs: []u32 = @as([*]u32, @ptrCast(@alignCast(self.bounds.host)))[2 * begin .. 2 * (begin + len)];
        for (0..len) |r| {
            pairs[2 * r] = @intCast(begin);
            pairs[2 * r + 1] = @intCast(begin + len);
        }
        const global = model.Kind.global.headSize();
        const sliding = model.Kind.sliding.headSize();
        @memcpy(self.rope_global.floats()[begin * global ..][0 .. len * global], self.table_global.floats()[0 .. len * global]);
        @memcpy(self.rope_sliding.floats()[begin * sliding ..][0 .. len * sliding], self.table_sliding.floats()[0 .. len * sliding]);
    }

    /// With an observer: finishes the stage's command buffer, shows the host
    /// the input's rows of `buffer`, and starts the next.
    fn stage(self: *Plan, observer: ?runtime.Observer, at: runtime.Stage, buffer: Buffer, width: usize, inputs: []const runtime.Input) !void {
        const o = observer orelse return;
        try self.backend.commit();
        o.record(o.context, at, buffer.floats()[0 .. inputs[0].tokens.len * width]);
        try self.backend.begin();
    }
};
