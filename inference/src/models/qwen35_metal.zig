//! GPU-resident execution of the pinned text-only Qwen schedule. One command
//! buffer per token: the CPU supplies a token ID and reads back logits (or a
//! greedy token); every activation, norm, gate, cache write, and recurrent
//! update stays on the GPU. The schedule is the same as `qwen35_runtime.zig`,
//! which remains the CPU reference these results are compared against. On a
//! rotated (Bonsai) file the embedding row is un-rotated after the gather and
//! every activation a rotated weight reads is transformed in place first
//! (`ssm_alpha` / `ssm_beta` read the residual before its transform).
//!
//! Ownership: the plan borrows the mapped weights and the `Backend`; it owns
//! the session and the GPU buffers it creates. Session memory is wrapped once
//! as a shared buffer, so `reset()` is a CPU memset that is only valid after
//! `commit()` — which the synchronous backend guarantees.
const std = @import("std");
const model = @import("qwen35.zig");
const weights = @import("../runtime/weights.zig");
const session = @import("../runtime/session.zig");
const sampling = @import("../sampling/root.zig");
const metal = @import("../backends/metal/root.zig");
const Observer = @import("qwen35_runtime.zig").Observer;
const Tensor = @import("../formats/gguf.zig").Tensor;
const Buffer = metal.Buffer;

pub const vocabulary = 248320;
const hidden = 5120;
const ffn = 17408;

const LayerConstants = struct {
    attention_norm: Buffer,
    post_attention_norm: Buffer,
    mixer: union(enum) {
        full_attention: struct { query_norm: Buffer, key_norm: Buffer },
        delta_net: struct { convolution: Buffer, a: Buffer, time_bias: Buffer, norm: Buffer },
    },
};

/// A weight matrix ready for dispatch: its GPU range and CPU-side descriptor.
const Weight = struct { buffer: Buffer, matrix: @import("../backends/cpu/root.zig").Matrix };

/// The rotated file's transform operands, uploaded once at init: the sign
/// vector per width (`model.Rotation.widths` order) and, when the fold used
/// the grouped value-head order, the gather map with the regathered scratch
/// `ssm_out` reads (one decode row, `padded` prefill rows).
const Rotation = struct {
    signs: [3]Buffer,
    value_map: ?Buffer,
    regrouped: Buffer,
    regrouped_c: Buffer,

    fn signsFor(self: Rotation, width: usize) !Buffer {
        for (model.Rotation.widths, self.signs) |w, signs| if (w == width) return signs;
        return error.InvalidShape;
    }
};

pub const Plan = struct {
    alloc: std.mem.Allocator,
    backend: *metal.Backend,
    view: weights.View,
    binding: model.Binding,
    state: session.Session,
    state_buffer: Buffer,
    rotation: ?Rotation,
    constants: []LayerConstants,
    output_norm: Buffer,
    rope_table: Buffer,
    // Activations (F32 counts in comments).
    x: Buffer, // hidden
    normalized: Buffer, // hidden
    projected: Buffer, // hidden
    gate: Buffer, // ffn
    up: Buffer, // ffn
    qg: Buffer, // 24 * 512
    q: Buffer, // 24 * 256
    /// F32 key/value rows of the current token when the cache is F16:
    /// projected, normalized, and rotated here, then packed into the slot.
    /// With an F32 cache the projections write the slot directly.
    k: Buffer, // 1024
    v: Buffer, // 1024
    mixed: Buffer, // 10240
    convolved: Buffer, // 10240
    z: Buffer, // 6144
    alpha: Buffer, // 48
    beta: Buffer, // 48
    mixed_out: Buffer, // 6144
    /// Flash-decoding partials: `[24][splits][2 + 256]`, 1.6 MB, independent of the capacity.
    partials: Buffer,
    logits: Buffer, // vocabulary
    argmax_values: Buffer,
    argmax_indices: Buffer,
    argmax_result: Buffer,
    /// Partial top-k scratch for sampled decoding; `sampling.TopK.capacity` entries.
    topk: metal.Backend.TopKBuffers,
    /// Chunked prefill: up to `chunk` prompt tokens per command buffer
    /// through the batched matmul. The `_c` buffers hold `padded` token rows
    /// ([token][feature]); `step` keeps its own single-token buffers above, so
    /// the two paths never alias.
    chunk: usize,
    padded: usize,
    x_c: Buffer, // padded × hidden
    normalized_c: Buffer,
    projected_c: Buffer,
    gate_c: Buffer, // padded × ffn
    up_c: Buffer,
    qg_c: Buffer, // padded × 24 × 512
    q_c: Buffer, // padded × 24 × 256
    k_c: Buffer, // padded × 1024
    v_c: Buffer,
    /// Half copy of `q_c` for the F16 chunk attention (its operands share one type).
    q_c_h: Buffer, // padded × 24 × 256 halves
    mixed_c: Buffer, // padded × 10240
    convolved_c: Buffer,
    z_c: Buffer, // padded × 6144
    alpha_c: Buffer, // padded × 48
    beta_c: Buffer,
    mixed_out_c: Buffer, // padded × 6144

    /// `chunk` bounds the tokens one `prefill` command buffer processes (and
    /// sizes its activation buffers: about 0.4 MB per token). `kv` is the
    /// attention cache precision: `f16` halves cache memory and the
    /// bytes attention reads per token; the recurrent state stays F32.
    pub fn init(alloc: std.mem.Allocator, backend: *metal.Backend, view: weights.View, binding: model.Binding, capacity: usize, chunk: usize, kv: session.Precision) !Plan {
        if (chunk == 0 or chunk > 4096) return error.InvalidShape;
        var layouts: [64]session.Layout = undefined;
        for (binding.layers, &layouts) |layer, *layout| layout.* = switch (layer.mixer) {
            .full_attention => .{ .attention = .{ .key_row = 1024, .value_row = 1024, .precision = kv } },
            .delta_net => .{ .recurrent = .{ .history = 10240 * 3, .matrix = 48 * 128 * 128 } },
        };
        var state = try session.Session.init(alloc, &layouts, capacity);
        errdefer state.deinit();
        const constants = try alloc.alloc(LayerConstants, binding.layers.len);
        errdefer alloc.free(constants);
        var self: Plan = undefined;
        self.alloc = alloc;
        self.backend = backend;
        self.view = view;
        self.binding = binding;
        self.state = state;
        self.constants = constants;
        self.state_buffer = try backend.wrap(state.memory);
        for (binding.layers, constants) |layer, *c| {
            c.* = .{
                .attention_norm = try self.constant(layer.attention_norm),
                .post_attention_norm = try self.constant(layer.post_attention_norm),
                .mixer = switch (layer.mixer) {
                    .full_attention => |attn| .{ .full_attention = .{ .query_norm = try self.constant(attn.query_norm), .key_norm = try self.constant(attn.key_norm) } },
                    .delta_net => |linear| .{ .delta_net = .{ .convolution = try self.constant(linear.convolution), .a = try self.constant(linear.a), .time_bias = try self.constant(linear.time_bias), .norm = try self.constant(linear.norm) } },
                },
            };
        }
        self.output_norm = try self.constant(binding.output_norm);
        self.rope_table = try backend.create(capacity * 32 * 8);
        try metal.Backend.ropeTable(self.rope_table, capacity, 64, 1e7, null);
        self.x = try backend.create(hidden * 4);
        self.normalized = try backend.create(hidden * 4);
        self.projected = try backend.create(hidden * 4);
        self.gate = try backend.create(ffn * 4);
        self.up = try backend.create(ffn * 4);
        self.qg = try backend.create(24 * 512 * 4);
        self.q = try backend.create(24 * 256 * 4);
        self.k = try backend.create(1024 * 4);
        self.v = try backend.create(1024 * 4);
        // One shared projection allocation lets four weights and all DeltaNet
        // outputs fit the merged kernel's seven data bindings. The named slices
        // preserve the downstream operation interfaces and borrowed lifetimes.
        const delta_projections = try backend.create((10240 + 6144 + 48 + 48) * 4);
        self.mixed = delta_projections.slice(0, 10240 * 4);
        self.convolved = try backend.create(10240 * 4);
        self.z = delta_projections.slice(10240 * 4, 6144 * 4);
        self.alpha = delta_projections.slice((10240 + 6144 + 48) * 4, 48 * 4);
        self.beta = delta_projections.slice((10240 + 6144) * 4, 48 * 4);
        self.mixed_out = try backend.create(6144 * 4);
        self.partials = try backend.create(metal.Backend.attentionDecodePartials(24, 256) * 4);
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
        self.qg_c = try backend.create(n * 24 * 512 * 4);
        self.q_c = try backend.create(n * 24 * 256 * 4);
        self.k_c = try backend.create(n * 1024 * 4);
        self.v_c = try backend.create(n * 1024 * 4);
        self.q_c_h = try backend.create(n * 24 * 256 * 2);
        self.mixed_c = try backend.create(n * 10240 * 4);
        self.convolved_c = try backend.create(n * 10240 * 4);
        self.z_c = try backend.create(n * 6144 * 4);
        self.alpha_c = try backend.create(n * 48 * 4);
        self.beta_c = try backend.create(n * 48 * 4);
        self.mixed_out_c = try backend.create(n * 6144 * 4);
        self.rotation = null;
        if (binding.rotation) |rotation| {
            var signs: [3]Buffer = undefined;
            for (&signs, rotation.signs) |*buffer, source| {
                if (source.values.len == 0) return error.InvalidShape;
                buffer.* = try backend.create(source.values.len * 4);
                for (buffer.floats(), source.values) |*out, value| out.* = @floatFromInt(value.signed);
            }
            var value_map: ?Buffer = null;
            if (rotation.value_grouped) {
                // Grouped head `nk * 3 + rep` takes the mixer's tiled head `rep * 16 + nk`.
                const map = try backend.create(48 * 4);
                for (@as([*]u32, @ptrCast(@alignCast(map.host)))[0..48], 0..) |*m, g| m.* = @intCast((g % 3) * 16 + g / 3);
                value_map = map;
            }
            self.rotation = .{ .signs = signs, .value_map = value_map, .regrouped = try backend.create(6144 * 4), .regrouped_c = try backend.create(n * 6144 * 4) };
        }
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
    /// Tensors are wrapped one at a time (cached by address). Wrapping the whole
    /// 16 GB mapping as one buffer was measured to make GPU time unstable by
    /// 3-4x between runs, presumably from per-command-buffer residency work over
    /// the entire range; per-tensor buffers are steady.
    fn weight(self: *Plan, tensor: *const Tensor) !Weight {
        const matrix = try self.view.matrix(tensor);
        return .{ .buffer = try self.backend.wrap(matrix.bytes), .matrix = matrix };
    }
    /// The GPU range of one session region (a byte range of the wrapped block).
    fn stateSlice(self: *Plan, region: []const u8) Buffer {
        return self.state_buffer.slice(self.state.offsetOf(region), region.len);
    }
    fn stateFloats(self: *Plan, values: []const f32) Buffer {
        return self.stateSlice(std.mem.sliceAsBytes(values));
    }
    fn mm(self: *Plan, tensor: *const Tensor, input: Buffer, output: Buffer) !void {
        const w = try self.weight(tensor);
        try self.backend.matvec(w.buffer, w.matrix, input, output);
    }
    /// On a rotated file, transforms `rows` rows of `width` (packed at that
    /// stride) in place into the basis the rotated weights were folded for;
    /// no dispatch on a plain file.
    fn rotate(self: *Plan, data: Buffer, width: usize, rows: usize) !void {
        const rotation = self.rotation orelse return;
        try self.backend.hadamard(data, try rotation.signsFor(width), width, rows, width, false);
    }
    /// The activation `ssm_out` reads: the mixer output as is on a plain file;
    /// on a rotated one, transformed in place, or first regathered into the
    /// scratch of its path (`chunk`: the prefill rows) from the tiled to the
    /// grouped head order when the fold used it (`qwen35_runtime.rotateGrouped`).
    fn rotateGrouped(self: *Plan, input: Buffer, rows: usize, chunk: bool) !Buffer {
        const rotation = self.rotation orelse return input;
        const map = rotation.value_map orelse {
            try self.rotate(input, 6144, rows);
            return input;
        };
        const scratch = if (chunk) rotation.regrouped_c else rotation.regrouped;
        try self.backend.gatherRows(scratch, input, map, 128, 48, rows, 6144, 6144);
        try self.rotate(scratch, 6144, rows);
        return scratch;
    }

    /// Merge only shapes the backend can encode as whole threadgroups. A
    /// rejected merge records no work; the standalone path remains the fallback.
    fn projections(self: *Plan, tensors: []const *const Tensor, outputs: []const Buffer, mode: metal.Backend.SegmentMode) !void {
        std.debug.assert(tensors.len == outputs.len and tensors.len <= 4);
        var segments: [4]metal.Backend.Segment = undefined;
        for (tensors, outputs, 0..) |tensor, output, i| {
            const w = try self.weight(tensor);
            segments[i] = .{ .weights = w.buffer, .matrix = w.matrix, .output = if (mode == .silu_mul_pair) outputs[0] else output };
        }
        self.backend.matvecSegments(segments[0..tensors.len], self.normalized, mode) catch |err| switch (err) {
            error.InvalidShape => {
                for (segments[0..tensors.len], outputs) |s, output| try self.backend.matvec(s.weights, s.matrix, self.normalized, output);
                if (mode == .silu_mul_pair) try self.backend.siluMul(outputs[0], outputs[1], segments[0].matrix.rows);
            },
            else => return err,
        };
    }

    /// Records and runs one token. `logits` receives the vocabulary scores when
    /// non-null; `greedy` receives the argmax when non-null (computed on the
    /// GPU, no logit readback); `topk` receives the best `TopK.capacity`
    /// logits and the softmax denominator for `topk.temperature` (2 KB read
    /// back instead of 1 MB; `readLogits` still works afterwards when the
    /// sampler must fall back). An observer's `check` runs between recorded
    /// layers at no GPU cost; its `layer` callback forces a commit after every
    /// layer (64 command buffers per token) and is for numerical traces only.
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
        if (self.rotation) |rotation| try b.hadamard(self.x, try rotation.signsFor(hidden), hidden, 1, hidden, true);
        for (self.binding.layers, self.constants, 0..) |layer, c, il| {
            try b.rmsNorm(self.x, c.attention_norm, self.normalized, .{ .rows = 1, .width = hidden, .in_stride = hidden, .out_stride = hidden });
            switch (layer.mixer) {
                .full_attention => |attn| try self.fullAttention(attn, c.mixer.full_attention, il),
                .delta_net => |linear| try self.linearAttention(linear, c.mixer.delta_net, il),
            }
            try b.add(self.x, self.projected, hidden);
            try b.rmsNorm(self.x, c.post_attention_norm, self.normalized, .{ .rows = 1, .width = hidden, .in_stride = hidden, .out_stride = hidden });
            try self.rotate(self.normalized, hidden, 1);
            try self.projections(&.{ layer.ffn_gate, layer.ffn_up }, &.{ self.gate, self.up }, .silu_mul_pair);
            try self.rotate(self.gate, ffn, 1);
            try self.mm(layer.ffn_down, self.gate, self.projected);
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
            try b.rmsNorm(self.x, self.output_norm, self.normalized, .{ .rows = 1, .width = hidden, .in_stride = hidden, .out_stride = hidden });
            try self.recordOutputs(greedy, topk);
        }
        try b.commit();
        try self.readOutputs(logits, greedy, topk);
        try self.state.commit();
    }

    /// Records the output head for the normalized last-token row in
    /// `self.normalized` and whichever readbacks were requested.
    fn recordOutputs(self: *Plan, greedy: ?*u32, topk: ?*sampling.TopK) !void {
        const b = self.backend;
        try self.rotate(self.normalized, hidden, 1);
        try self.mm(self.binding.output, self.normalized, self.logits);
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

    /// Chunked prefill: consumes `tokens` in chunks of at most `chunk`,
    /// one command buffer per chunk, with the projections and feed-forward
    /// batched through the matmul kernel, attention as one causal tiled
    /// dispatch per layer, and DeltaNet as one chunkwise dispatch per
    /// layer; nothing steps per token inside a chunk.
    /// Readbacks (`logits`, `greedy`, `topk`) refer to the last token, as in
    /// `step`. The observer's `check` runs between layers of every chunk;
    /// its `layer` callback is per token by contract and is not supported
    /// here (`error.InvalidShape`): trace through `step`. Arithmetic order
    /// differs from `step` (tile accumulation), so results agree within a
    /// tolerance, not bit for bit; `generation-check --metal` measures it.
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
        if (self.rotation) |rotation| try b.hadamard(self.x_c, try rotation.signsFor(hidden), hidden, count, hidden, true);
        const norm: metal.Backend.Norm = .{ .rows = count, .width = hidden, .in_stride = hidden, .out_stride = hidden };
        for (self.binding.layers, self.constants, 0..) |layer, c, il| {
            try b.rmsNorm(self.x_c, c.attention_norm, self.normalized_c, norm);
            switch (layer.mixer) {
                .full_attention => |attn| try self.attentionChunk(attn, c.mixer.full_attention, il, count),
                .delta_net => |linear| try self.deltaChunk(linear, c.mixer.delta_net, il, count),
            }
            try b.add(self.x_c, self.projected_c, count * hidden);
            try b.rmsNorm(self.x_c, c.post_attention_norm, self.normalized_c, norm);
            try self.rotate(self.normalized_c, hidden, count);
            try self.mmRows(layer.ffn_gate, self.normalized_c, hidden, self.gate_c, ffn, count);
            try self.mmRows(layer.ffn_up, self.normalized_c, hidden, self.up_c, ffn, count);
            try b.siluMul(self.gate_c, self.up_c, count * ffn);
            try self.rotate(self.gate_c, ffn, count);
            try self.mmRows(layer.ffn_down, self.gate_c, ffn, self.projected_c, hidden, count);
            try b.add(self.x_c, self.projected_c, count * hidden);
            if (observer) |o| if (o.check) |check| try check(o.context);
        }
        if (logits != null or greedy != null or topk != null) {
            const last_row = self.x_c.slice((count - 1) * hidden * 4, hidden * 4);
            try b.rmsNorm(last_row, self.output_norm, self.normalized, .{ .rows = 1, .width = hidden, .in_stride = hidden, .out_stride = hidden });
            try self.recordOutputs(greedy, topk);
        }
        try b.commit();
        try self.readOutputs(logits, greedy, topk);
        try self.state.commitChunk(count);
    }

    /// Batched projection over `count` token rows.
    fn mmRows(self: *Plan, tensor: *const Tensor, input: Buffer, in_stride: usize, output: Buffer, out_stride: usize, count: usize) !void {
        const w = try self.weight(tensor);
        try self.backend.matmul(w.buffer, w.matrix, input, in_stride, output, out_stride, count);
    }

    fn attentionChunk(self: *Plan, attn: model.FullAttention, c: anytype, il: usize, count: usize) !void {
        const b = self.backend;
        const cache = self.state.layers[il].attention;
        const position = self.state.position;
        try self.rotate(self.normalized_c, hidden, count);
        try self.mmRows(attn.query_and_gate, self.normalized_c, hidden, self.qg_c, 24 * 512, count);
        try self.mmRows(attn.key, self.normalized_c, hidden, self.k_c, 1024, count);
        try self.mmRows(attn.value, self.normalized_c, hidden, self.v_c, 1024, count);
        try b.rmsNorm(self.qg_c, c.query_norm, self.q_c, .{ .rows = count * 24, .width = 256, .in_stride = 512, .out_stride = 256 });
        try b.ropeRows(self.q_c, self.rope_table, 24, 256, 64, position, count, 24 * 256);
        try b.rmsNorm(self.k_c, c.key_norm, self.k_c, .{ .rows = count * 4, .width = 256, .in_stride = 256, .out_stride = 256 });
        try b.ropeRows(self.k_c, self.rope_table, 4, 256, 64, position, count, 1024);
        // The chunk's keys and values are contiguous in the cache; padding rows never leave the chunk buffers.
        const precision = cache.keys.precision;
        const k_rows = self.stateSlice(cache.keys.range(position, count));
        const v_rows = self.stateSlice(cache.values.range(position, count));
        var queries = self.q_c;
        switch (precision) {
            .f32 => {
                try b.copy(k_rows, self.k_c, count * 1024);
                try b.copy(v_rows, self.v_c, count * 1024);
            },
            .f16 => {
                // The F16 chunk kernel multiplies half operands: the cache rows and
                // a half copy of the rotated queries (padded rows included, as the
                // kernel computes on them).
                try b.packHalf(&.{ .{ .dst = k_rows, .src = self.k_c, .count = count * 1024 }, .{ .dst = v_rows, .src = self.v_c, .count = count * 1024 } });
                try b.packHalf(&.{.{ .dst = self.q_c_h, .src = self.q_c, .count = metal.Backend.attentionChunkRows(count) * 24 * 256 }});
                queries = self.q_c_h;
            },
        }
        // One causal tiled dispatch over the whole chunk: row t attends to
        // cache[0 .. position + t]; padded rows of q_c/mixed_out_c absorb the tail.
        const total = position + count;
        try b.attentionChunk(self.stateSlice(cache.keys.range(0, total)), self.stateSlice(cache.values.range(0, total)), queries, self.mixed_out_c, .{ .query_heads = 24, .kv_heads = 4, .key_width = 256, .value_width = 256, .position = position, .count = count, .q_stride = 24 * 256, .out_stride = 6144, .scale = 1.0 / 16.0, .precision = precision });
        try b.sigmoidGate(self.mixed_out_c, self.qg_c, count * 24, 256, 512, 256);
        try self.rotate(self.mixed_out_c, 6144, count);
        try self.mmRows(attn.output, self.mixed_out_c, 6144, self.projected_c, hidden, count);
    }

    fn deltaChunk(self: *Plan, linear: model.DeltaNet, c: anytype, il: usize, count: usize) !void {
        const b = self.backend;
        const state = self.state.layers[il].recurrent;
        // The gating projections read the residual before its transform.
        try self.mmRows(linear.beta, self.normalized_c, hidden, self.beta_c, 48, count);
        try self.mmRows(linear.alpha, self.normalized_c, hidden, self.alpha_c, 48, count);
        try self.rotate(self.normalized_c, hidden, count);
        try self.mmRows(linear.qkv, self.normalized_c, hidden, self.mixed_c, 10240, count);
        try self.mmRows(linear.gate, self.normalized_c, hidden, self.z_c, 6144, count);
        const history = self.stateFloats(state.history);
        try b.convolutionRows(history, self.mixed_c, c.convolution, self.convolved_c, 10240, 4, count, 10240);
        try b.convolutionHistory(history, self.mixed_c, 10240, 4, count, 10240);
        try b.silu(self.convolved_c, count * 10240);
        try b.l2NormRows(self.convolved_c, count, 32, 128, 128, 10240, 1e-6);
        try b.deltaGatesRows(self.alpha_c, self.beta_c, c.a, c.time_bias, 48, count);
        // One chunkwise dispatch for the whole chunk: the WY form over
        // 32-token sub-chunks with the state carried in place.
        try b.deltaChunk(self.stateFloats(state.matrix), self.convolved_c, self.alpha_c, self.beta_c, self.mixed_out_c, .{ .qheads = 16, .vheads = 48, .keys = 128, .values = 128, .count = count, .in_stride = 10240, .gate_stride = 48, .out_stride = 6144, .scale = 1.0 / @sqrt(@as(f32, 128)) });
        try b.rmsNorm(self.mixed_out_c, c.norm, self.mixed_out_c, .{ .rows = count * 48, .width = 128, .in_stride = 128, .out_stride = 128, .silu_multiplier = .{ .buffer = self.z_c, .stride = 128 } });
        const out_input = try self.rotateGrouped(self.mixed_out_c, count, true);
        try self.mmRows(linear.output, out_input, 6144, self.projected_c, hidden, count);
    }

    /// Copies the last step's logits out of the shared buffer. Valid after a
    /// `step` that computed them (any non-null `logits`/`greedy`/`topk`) and
    /// until the next step; the sampler's fallback path uses it.
    pub fn readLogits(self: *Plan, out: []f32) !void {
        if (out.len != vocabulary) return error.InvalidShape;
        @memcpy(out, self.logits.floats());
        for (out) |v| if (!std.math.isFinite(v)) return error.NonFiniteResult;
    }

    fn fullAttention(self: *Plan, attn: model.FullAttention, c: anytype, il: usize) !void {
        const b = self.backend;
        const cache = self.state.layers[il].attention;
        const position = self.state.position;
        const k_slot = self.stateSlice(cache.keys.range(position, 1));
        const v_slot = self.stateSlice(cache.values.range(position, 1));
        const precision = cache.keys.precision;
        // An F32 cache takes the projections directly; an F16 cache takes them
        // through F32 scratch rows and one pack dispatch.
        const k_row = if (precision == .f32) k_slot else self.k;
        const v_row = if (precision == .f32) v_slot else self.v;
        try self.rotate(self.normalized, hidden, 1);
        try self.projections(&.{ attn.query_and_gate, attn.key, attn.value }, &.{ self.qg, k_row, v_row }, .plain);
        // Each projected head stores query then gate; the gate is applied after attention.
        try b.rmsNorm(self.qg, c.query_norm, self.q, .{ .rows = 24, .width = 256, .in_stride = 512, .out_stride = 256 });
        try b.rope(self.q, self.rope_table, 24, 256, 64, position);
        try b.rmsNorm(k_row, c.key_norm, k_row, .{ .rows = 4, .width = 256, .in_stride = 256, .out_stride = 256 });
        try b.rope(k_row, self.rope_table, 4, 256, 64, position);
        if (precision == .f16) try b.packHalf(&.{ .{ .dst = k_slot, .src = k_row, .count = 1024 }, .{ .dst = v_slot, .src = v_row, .count = 1024 } });
        const visible = position + 1;
        try b.attentionDecode(self.stateSlice(cache.keys.range(0, visible)), self.stateSlice(cache.values.range(0, visible)), self.q, self.partials, self.mixed_out, .{ .query_heads = 24, .kv_heads = 4, .key_width = 256, .value_width = 256, .visible = visible, .scale = 1.0 / 16.0, .precision = precision });
        try b.sigmoidGate(self.mixed_out, self.qg, 24, 256, 512, 256);
        try self.rotate(self.mixed_out, 6144, 1);
        try self.mm(attn.output, self.mixed_out, self.projected);
    }

    fn linearAttention(self: *Plan, linear: model.DeltaNet, c: anytype, il: usize) !void {
        const b = self.backend;
        const state = self.state.layers[il].recurrent;
        if (self.rotation == null) {
            try self.projections(&.{ linear.qkv, linear.gate, linear.beta, linear.alpha }, &.{ self.mixed, self.z, self.beta, self.alpha }, .plain);
        } else {
            // The gating projections read the residual before its transform,
            // so the merge splits in two around it.
            try self.projections(&.{ linear.beta, linear.alpha }, &.{ self.beta, self.alpha }, .plain);
            try self.rotate(self.normalized, hidden, 1);
            try self.projections(&.{ linear.qkv, linear.gate }, &.{ self.mixed, self.z }, .plain);
        }
        try b.convolution(self.stateFloats(state.history), self.mixed, c.convolution, self.convolved, 10240, 4);
        try b.silu(self.convolved, 10240);
        // Convolution output is [Q:16x128 | K:16x128 | V:48x128]; Q and K are L2-normalized.
        try b.l2Norm(self.convolved, 32, 128, 128, 1e-6);
        try b.deltaGates(self.alpha, self.beta, c.a, c.time_bias, 48);
        try b.delta(self.stateFloats(state.matrix), self.convolved, self.alpha, self.beta, self.mixed_out, .{ .qheads = 16, .vheads = 48, .keys = 128, .values = 128, .scale = 1.0 / @sqrt(@as(f32, 128)) });
        try b.rmsNorm(self.mixed_out, c.norm, self.mixed_out, .{ .rows = 48, .width = 128, .in_stride = 128, .out_stride = 128, .silu_multiplier = .{ .buffer = self.z, .stride = 128 } });
        const out_input = try self.rotateGrouped(self.mixed_out, 1, false);
        try self.mm(linear.output, out_input, self.projected);
    }
};
