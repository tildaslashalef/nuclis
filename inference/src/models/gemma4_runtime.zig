//! CPU execution of the pinned Gemma 4 12B text schedule: the
//! numerical reference `gemma4_metal.zig` is compared against, written from
//! the forward pass recorded in docs/reference/gemma4.md. Immutable weight
//! views and the binding borrow the loaded model; this runtime owns the
//! session (one attention cache per layer, F32) and its workspace. A failed
//! step poisons the session. Text only: no vision, no MTP head.
//!
//! What differs from the Qwen runtime, operation by operation:
//! - the embedding row is scaled by sqrt(3840) after decoding;
//! - every layer has four residual-stream norms (pre/post attention,
//!   pre/post FFN) whose weights are stored raw (`rms(x) * w`);
//! - queries and keys are RMS-normed per head with a weight, values RMS-normed
//!   per head *without* one; on global layers the value projection is the
//!   key projection before its norm;
//! - RoPE is split-half over the whole head, base 1e4 on sliding layers and
//!   1e6 with the checkpoint's frequency factors on global layers;
//! - attention scores are unscaled (`scale = 1`), and sliding layers see only
//!   the last 1024 positions, which the reference reads by slicing the
//!   cache rows rather than masking;
//! - the FFN gate is tanh-GELU, and each layer's new residual is multiplied
//!   by its scalar output scale;
//! - logits come from the embedding matrix and are soft-capped at 30.
const std = @import("std");
const model = @import("gemma4.zig");
const weights = @import("../runtime/weights.zig");
const session = @import("../runtime/session.zig");
const cpu = @import("../backends/cpu/root.zig");
const Tensor = @import("../formats/gguf.zig").Tensor;

pub const Observer = @import("../runtime/observer.zig").Observer;

const embedding = model.embedding;
const embedding_scale: f32 = @sqrt(@as(f32, embedding));

/// Small F32 weights decoded once per layer at init.
const LayerConstants = struct {
    attention_norm: []f32,
    post_attention_norm: []f32,
    ffn_norm: []f32,
    post_ffn_norm: []f32,
    query_norm: []f32,
    key_norm: []f32,
    output_scale: f32,
};

pub const Runtime = struct {
    storage: std.heap.ArenaAllocator,
    state: session.Session,
    view: weights.View,
    binding: model.Binding,
    constants: []LayerConstants,
    output_norm: []f32,
    rope_factors: []f32,
    x: []f32,
    normalized: []f32,
    projected: []f32,
    gate: []f32,
    up: []f32,
    row: []f32,
    q: []f32,
    k: []f32,
    v: []f32,
    mixed_out: []f32,
    attention_scratch: []f64,

    pub fn init(gpa: std.mem.Allocator, view: weights.View, binding: model.Binding, capacity: usize) !Runtime {
        var layouts: [model.layer_count]session.Layout = undefined;
        for (binding.layers, &layouts) |layer, *layout| {
            const width = layer.kind.kvWidth();
            layout.* = .{ .attention = .{ .key_row = width, .value_row = width } };
        }
        var state = try session.Session.init(gpa, &layouts, capacity);
        errdefer state.deinit();
        var storage: std.heap.ArenaAllocator = .init(gpa);
        errdefer storage.deinit();
        const a = storage.allocator();
        // Finish allocations before transferring the arena, whose internal
        // linked-list head can change on each allocation.
        var result: Runtime = undefined;
        inline for (.{ "x", "normalized", "projected" }) |field| @field(result, field) = try a.alloc(f32, embedding);
        inline for (.{ "gate", "up", "row" }) |field| @field(result, field) = try a.alloc(f32, model.feed_forward);
        // Sized for the wider (global) geometry; sliding layers use a prefix.
        result.q = try a.alloc(f32, model.Kind.global.queryWidth());
        result.mixed_out = try a.alloc(f32, model.Kind.global.queryWidth());
        result.k = try a.alloc(f32, model.Kind.sliding.kvWidth());
        result.v = try a.alloc(f32, model.Kind.sliding.kvWidth());
        result.attention_scratch = try a.alloc(f64, capacity);
        result.output_norm = try view.vector(a, binding.output_norm);
        result.rope_factors = try view.vector(a, binding.rope_factors);
        result.constants = try a.alloc(LayerConstants, binding.layers.len);
        for (binding.layers, result.constants) |layer, *constants| {
            constants.* = .{
                .attention_norm = try view.vector(a, layer.attention_norm),
                .post_attention_norm = try view.vector(a, layer.post_attention_norm),
                .ffn_norm = try view.vector(a, layer.ffn_norm),
                .post_ffn_norm = try view.vector(a, layer.post_ffn_norm),
                .query_norm = try view.vector(a, layer.query_norm),
                .key_norm = try view.vector(a, layer.key_norm),
                .output_scale = try view.scalar(layer.output_scale, 0),
            };
        }
        result.storage = storage;
        result.state = state;
        result.view = view;
        result.binding = binding;
        return result;
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
        try cpu.matvec(try self.view.matrix(tensor), input, output, self.row);
    }

    /// `rms(input) * weight`, the stored-raw weight convention of this checkpoint.
    fn norm(input: []const f32, output: []f32, weight: []const f32) !void {
        if (weight.len != output.len) return error.InvalidShape;
        try cpu.rmsNorm(input, output, model.rms_epsilon);
        for (output, weight) |*x, w| x.* *= w;
    }

    /// logits may be null for non-final prefill tokens, avoiding the LM head.
    /// It must otherwise hold exactly the vocabulary size. A successful step
    /// commits one token; a callback/error after begin requires reset.
    pub fn step(self: *Runtime, token: u32, logits: ?[]f32, observer: ?Observer) !void {
        if (token >= model.vocabulary) return error.InvalidTokenId;
        if (logits) |out| if (out.len != model.vocabulary) return error.InvalidShape;
        try self.state.begin();
        errdefer self.state.fail();
        try self.view.row(self.binding.token_embedding, token, self.x);
        for (self.x) |*x| x.* *= embedding_scale;
        for (self.binding.layers, self.constants, 0..) |layer, constants, il| {
            try norm(self.x, self.normalized, constants.attention_norm);
            try self.attention(layer, constants, il);
            try norm(self.projected, self.projected, constants.post_attention_norm);
            for (self.x, self.projected) |*x, contribution| x.* += contribution;
            try norm(self.x, self.normalized, constants.ffn_norm);
            try self.mm(layer.ffn_gate, self.normalized, self.gate);
            try self.mm(layer.ffn_up, self.normalized, self.up);
            for (self.gate, self.up) |*g, u| g.* = cpu.gelu(g.*) * u;
            try self.mm(layer.ffn_down, self.gate, self.projected);
            try norm(self.projected, self.projected, constants.post_ffn_norm);
            for (self.x, self.projected) |*x, contribution| {
                x.* = (x.* + contribution) * constants.output_scale;
                if (!std.math.isFinite(x.*)) return error.NonFiniteResult;
            }
            if (observer) |o| {
                if (o.check) |check| try check(o.context);
                if (o.layer) |report| try report(o.context, il, self.x);
            }
        }
        if (logits) |out| {
            try norm(self.x, self.normalized, self.output_norm);
            try self.mm(self.binding.token_embedding, self.normalized, out);
            for (out) |*x| {
                x.* = model.final_softcap * std.math.tanh(x.* / model.final_softcap);
                if (!std.math.isFinite(x.*)) return error.NonFiniteResult;
            }
        }
        try self.state.commit();
    }

    fn attention(self: *Runtime, layer: model.Layer, constants: LayerConstants, il: usize) !void {
        const kind = layer.kind;
        const hd = kind.headSize();
        const kv_heads = kind.kvHeads();
        const q = self.q[0..kind.queryWidth()];
        const k = self.k[0..kind.kvWidth()];
        const v = self.v[0..kind.kvWidth()];
        const position = self.state.position;
        const rope: cpu.rope.Options = .{
            .dimensions = hd,
            .base = kind.ropeBase(),
            .position = @intCast(position),
            .factors = if (kind == .global) self.rope_factors else null,
        };
        try self.mm(layer.query, self.normalized, q);
        try self.mm(layer.key, self.normalized, k);
        // Global layers have no value projection: V is the raw key projection,
        // taken before the key norm and RoPE.
        if (layer.value) |value| try self.mm(value, self.normalized, v) else @memcpy(v, k);
        for (0..model.heads) |h| {
            const head = q[h * hd ..][0..hd];
            try norm(head, head, constants.query_norm);
            try cpu.rope.apply(head, head, rope);
        }
        for (0..kv_heads) |h| {
            const key = k[h * hd ..][0..hd];
            try norm(key, key, constants.key_norm);
            try cpu.rope.apply(key, key, rope);
            const value = v[h * hd ..][0..hd];
            try cpu.rmsNorm(value, value, model.rms_epsilon);
        }
        const cache = self.state.layers[il].attention;
        // The reference cache is F32 by decision: the views assert it.
        @memcpy(cache.keys.floats(position, 1), k);
        @memcpy(cache.values.floats(position, 1), v);
        // Sliding layers attend to the last `window` positions including the
        // current one; the visible rows are a contiguous suffix of the cache.
        const first = if (kind == .sliding and position + 1 > model.window) position + 1 - model.window else 0;
        const visible = position + 1 - first;
        const out = self.mixed_out[0..kind.queryWidth()];
        try cpu.attention.apply(.{
            .query_heads = model.heads,
            .kv_heads = kv_heads,
            .key_width = hd,
            .value_width = hd,
            .tokens = visible,
            .visible_tokens = visible,
            .scale = 1.0,
            .queries = q,
            .keys = cache.keys.floats(first, visible),
            .values = cache.values.floats(first, visible),
        }, out, self.attention_scratch);
        try self.mm(layer.output, out, self.projected);
    }
};

test "runtime workspace cleanup and invalid steps preserve session admission" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, struct {
        fn check(alloc: std.mem.Allocator) !void {
            // The binding only supplies layer kinds and small-weight tensors
            // during preparation. Empty tensors make every weight read fail
            // after admission, which is what the invalid-step checks need.
            const tensor: Tensor = .{ .name = "empty", .dimensions = &.{0}, .encoding_id = 0, .offset = 0, .elements = 0, .bytes = 0 };
            // The per-layer output scale is read at init, so it needs one
            // real F32: the four-byte "file" below holds it.
            const scalar: Tensor = .{ .name = "scale", .dimensions = &.{1}, .encoding_id = 0, .offset = 0, .elements = 1, .bytes = 4 };
            const file = [_]u8{ 0, 0, 0, 0 };
            var layers: [model.layer_count]model.Layer = undefined;
            for (&layers, 0..) |*layer, i| layer.* = .{ .kind = model.kindOf(i), .attention_norm = &tensor, .post_attention_norm = &tensor, .ffn_norm = &tensor, .post_ffn_norm = &tensor, .query = &tensor, .key = &tensor, .value = if (model.kindOf(i) == .sliding) &tensor else null, .output = &tensor, .query_norm = &tensor, .key_norm = &tensor, .ffn_gate = &tensor, .ffn_up = &tensor, .ffn_down = &tensor, .output_scale = &scalar };
            const binding: model.Binding = .{ .token_embedding = &tensor, .output_norm = &tensor, .rope_factors = &tensor, .layers = layers, .summary = .{ .profile = "test", .decoder_layers = 48, .layer_kinds = &.{}, .text_tensors = 0, .auxiliary_tensors = 0, .text_tensor_bytes = 0, .auxiliary_tensor_bytes = 0 } };
            var runtime = try Runtime.init(alloc, .{ .file = &file, .data_offset = 0 }, binding, 1);
            defer runtime.deinit();
            try std.testing.expectError(error.InvalidTokenId, runtime.step(model.vocabulary, null, null));
            try std.testing.expectEqual(.ready, runtime.state.status);
            try std.testing.expectError(error.InvalidShape, runtime.step(0, &.{}, null));
            try std.testing.expectEqual(.ready, runtime.state.status);
            try std.testing.expect(runtime.step(0, null, null) != error.InvalidTokenId);
            try std.testing.expectEqual(.failed, runtime.state.status);
            runtime.reset();
            try std.testing.expectEqual(.ready, runtime.state.status);
        }
    }.check, .{});
}

test "session layouts follow the layer kinds" {
    // Not runnable without weights; the layout is checked through the
    // session's byte size: 40 sliding layers of 2 × 2048 F32 rows and 8
    // global layers of 2 × 512 rows per position.
    const per_position = 40 * 2 * 2048 * 4 + 8 * 2 * 512 * 4;
    var layouts: [model.layer_count]session.Layout = undefined;
    for (&layouts, 0..) |*layout, i| {
        const width = model.kindOf(i).kvWidth();
        layout.* = .{ .attention = .{ .key_row = width, .value_row = width } };
    }
    var state = try session.Session.init(std.testing.allocator, &layouts, 4);
    defer state.deinit();
    try std.testing.expect(state.bytes() >= per_position * 4);
    try std.testing.expect(state.bytes() < per_position * 4 + 48 * 2 * 16);
}
