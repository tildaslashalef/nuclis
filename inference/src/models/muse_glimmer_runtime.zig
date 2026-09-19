//! CPU execution of the pinned Muse Glimmer 30B schedule: the numerical
//! reference `muse_glimmer_metal.zig` is compared against, written from the
//! forward pass recorded in docs/reference/muse-glimmer.md. Immutable weight
//! views and the binding borrow the loaded model; this runtime owns the
//! session (one attention cache per layer, F32) and its workspace. A failed
//! step poisons the session. Text only: no vision, no drafter.
//!
//! What differs from the Gemma runtime, operation by operation:
//! - the embedding row is RMS-normed without a weight, not scaled;
//! - the post-attention and post-FFN norms use their own epsilon (1e-8),
//!   the pre norms and the query/key norms the file's (1e-5);
//! - every layer has a value projection and the same geometry (32 query
//!   heads of 128 over 2 KV heads); RoPE pairs adjacent dimensions over the
//!   whole head at base 5e5 on sliding layers, and global layers have no
//!   position encoding at all;
//! - scores are scaled by 1/sqrt(128); sliding layers see the last 2048
//!   positions as a cache-row slice;
//! - the attention output is multiplied by a sigmoid gate projected from
//!   the same normed input before the output projection;
//! - the FFN gate is SiLU, there is no per-layer output scale, and logits
//!   come from the untied head, scaled by `logit_scale`, soft-capped at 20.
const std = @import("std");
const model = @import("muse_glimmer.zig");
const weights = @import("../runtime/weights.zig");
const session = @import("../runtime/session.zig");
const cpu = @import("../backends/cpu/root.zig");
const Tensor = @import("../formats/gguf.zig").Tensor;

pub const Observer = @import("../runtime/observer.zig").Observer;

/// Small F32 weights decoded once per layer at init.
const LayerConstants = struct {
    attention_norm: []f32,
    post_attention_norm: []f32,
    ffn_norm: []f32,
    post_ffn_norm: []f32,
    query_norm: []f32,
    key_norm: []f32,
};

pub const Runtime = struct {
    storage: std.heap.ArenaAllocator,
    state: session.Session,
    view: weights.View,
    binding: model.Binding,
    constants: []LayerConstants,
    output_norm: []f32,
    x: []f32,
    normalized: []f32,
    projected: []f32,
    gate: []f32,
    up: []f32,
    row: []f32,
    q: []f32,
    k: []f32,
    v: []f32,
    attention_gate: []f32,
    mixed_out: []f32,
    attention_scratch: []f64,

    pub fn init(gpa: std.mem.Allocator, view: weights.View, binding: model.Binding, capacity: usize, checkpoint: bool, draft: bool) !Runtime {
        _ = draft;
        var layouts: [model.layer_count]session.Layout = undefined;
        for (&layouts) |*layout| layout.* = .{ .attention = .{ .key_row = model.kv_width, .value_row = model.kv_width } };
        var state = try session.Session.init(gpa, &layouts, capacity, checkpoint);
        errdefer state.deinit();
        var storage: std.heap.ArenaAllocator = .init(gpa);
        errdefer storage.deinit();
        const a = storage.allocator();
        // Finish allocations before transferring the arena, whose internal
        // linked-list head can change on each allocation.
        var result: Runtime = undefined;
        inline for (.{ "x", "normalized", "projected" }) |field| @field(result, field) = try a.alloc(f32, model.embedding);
        inline for (.{ "gate", "up" }) |field| @field(result, field) = try a.alloc(f32, model.feed_forward);
        inline for (.{ "q", "attention_gate", "mixed_out" }) |field| @field(result, field) = try a.alloc(f32, model.query_width);
        inline for (.{ "k", "v" }) |field| @field(result, field) = try a.alloc(f32, model.kv_width);
        // The matvec decode row holds the widest row any projection decodes.
        result.row = try a.alloc(f32, @max(model.feed_forward, model.embedding));
        result.attention_scratch = try a.alloc(f64, capacity);
        result.output_norm = try view.vector(a, binding.output_norm);
        result.constants = try a.alloc(LayerConstants, model.layer_count);
        for (binding.active(), result.constants) |layer, *constants| {
            constants.* = .{
                .attention_norm = try view.vector(a, layer.attention_norm),
                .post_attention_norm = try view.vector(a, layer.post_attention_norm),
                .ffn_norm = try view.vector(a, layer.ffn_norm),
                .post_ffn_norm = try view.vector(a, layer.post_ffn_norm),
                .query_norm = try view.vector(a, layer.query_norm),
                .key_norm = try view.vector(a, layer.key_norm),
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

    /// `rms(input) * weight` at the given epsilon; the weights are stored
    /// with their `+1` folded in.
    fn norm(input: []const f32, output: []f32, weight: []const f32, epsilon: f32) !void {
        if (weight.len != output.len) return error.InvalidShape;
        try cpu.rmsNorm(input, output, epsilon);
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
        try cpu.rmsNorm(self.x, self.x, model.rms_epsilon);
        for (self.binding.active(), self.constants, 0..) |layer, constants, il| {
            try norm(self.x, self.normalized, constants.attention_norm, model.rms_epsilon);
            try self.attention(layer, constants, il);
            try norm(self.projected, self.projected, constants.post_attention_norm, model.post_norm_epsilon);
            for (self.x, self.projected) |*x, contribution| x.* += contribution;
            try self.feedForward(layer, constants);
            for (self.x, self.projected) |*x, contribution| {
                x.* += contribution;
                if (!std.math.isFinite(x.*)) return error.NonFiniteResult;
            }
            if (observer) |o| {
                if (o.check) |check| try check(o.context);
                if (o.layer) |report| try report(o.context, il, self.x);
            }
        }
        if (logits) |out| {
            try norm(self.x, self.normalized, self.output_norm, model.rms_epsilon);
            try self.mm(self.binding.output, self.normalized, out);
            for (out) |*x| {
                x.* = model.final_softcap * std.math.tanh(x.* * model.logit_scale / model.final_softcap);
                if (!std.math.isFinite(x.*)) return error.NonFiniteResult;
            }
        }
        try self.state.commit();
    }

    /// The feed-forward block over the residual `x` into `projected`,
    /// post-normed and ready to add.
    fn feedForward(self: *Runtime, layer: model.Layer, constants: LayerConstants) !void {
        try norm(self.x, self.normalized, constants.ffn_norm, model.rms_epsilon);
        try self.mm(layer.ffn_gate, self.normalized, self.gate);
        try self.mm(layer.ffn_up, self.normalized, self.up);
        for (self.gate, self.up) |*g, u| g.* = cpu.silu(g.*) * u;
        try self.mm(layer.ffn_down, self.gate, self.projected);
        try norm(self.projected, self.projected, constants.post_ffn_norm, model.post_norm_epsilon);
    }

    fn attention(self: *Runtime, layer: model.Layer, constants: LayerConstants, il: usize) !void {
        const hd = model.head_size;
        const position = self.state.position;
        const rope: cpu.rope.Options = .{ .dimensions = hd, .base = model.rope_base, .position = @intCast(position), .pairing = .adjacent };
        try self.mm(layer.query, self.normalized, self.q);
        try self.mm(layer.key, self.normalized, self.k);
        try self.mm(layer.value, self.normalized, self.v);
        try self.mm(layer.gate, self.normalized, self.attention_gate);
        for (0..model.heads) |h| {
            const head = self.q[h * hd ..][0..hd];
            try norm(head, head, constants.query_norm, model.rms_epsilon);
            if (layer.kind == .sliding) try cpu.rope.apply(head, head, rope);
        }
        for (0..model.kv_heads) |h| {
            const key = self.k[h * hd ..][0..hd];
            try norm(key, key, constants.key_norm, model.rms_epsilon);
            if (layer.kind == .sliding) try cpu.rope.apply(key, key, rope);
        }
        const cache = self.state.layers[il].attention;
        // The reference cache is F32 by decision: the views assert it.
        @memcpy(cache.keys.floats(position, 1), self.k);
        @memcpy(cache.values.floats(position, 1), self.v);
        // Sliding layers attend to the last `window` positions including the
        // current one; the visible rows are a contiguous suffix of the cache.
        const first = if (layer.kind == .sliding and position + 1 > model.window) position + 1 - model.window else 0;
        const visible = position + 1 - first;
        try cpu.attention.apply(.{
            .query_heads = model.heads,
            .kv_heads = model.kv_heads,
            .key_width = hd,
            .value_width = hd,
            .tokens = visible,
            .visible_tokens = visible,
            .scale = model.attention_scale,
            .queries = self.q,
            .keys = cache.keys.floats(first, visible),
            .values = cache.values.floats(first, visible),
        }, self.mixed_out, self.attention_scratch);
        for (self.mixed_out, self.attention_gate) |*o, g| o.* *= cpu.sigmoid(g);
        try self.mm(layer.output, self.mixed_out, self.projected);
    }
};

/// A binding over empty tensors for the workspace tests: every weight read
/// fails after admission, which is what the invalid-step checks need.
fn emptyBinding(tensor: *const Tensor) model.Binding {
    var binding: model.Binding = undefined;
    binding.token_embedding = tensor;
    binding.output = tensor;
    binding.output_norm = tensor;
    for (&binding.layers, 0..) |*layer, i| {
        layer.* = .{ .kind = model.kindOf(i), .attention_norm = tensor, .post_attention_norm = tensor, .ffn_norm = tensor, .post_ffn_norm = tensor, .query = tensor, .key = tensor, .value = tensor, .gate = tensor, .output = tensor, .query_norm = tensor, .key_norm = tensor, .ffn_gate = tensor, .ffn_up = tensor, .ffn_down = tensor };
    }
    binding.summary = .{ .profile = model.profile, .decoder_layers = model.layer_count, .layer_kinds = &.{}, .text_tensors = 0, .auxiliary_tensors = 0, .text_tensor_bytes = 0, .auxiliary_tensor_bytes = 0 };
    return binding;
}

test "runtime workspace cleanup and invalid steps preserve session admission" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, struct {
        fn check(alloc: std.mem.Allocator) !void {
            const tensor: Tensor = .{ .name = "empty", .dimensions = &.{0}, .encoding_id = 0, .offset = 0, .elements = 0, .bytes = 0 };
            const file = [_]u8{ 0, 0, 0, 0 };
            var runtime = try Runtime.init(alloc, .{ .file = &file, .data_offset = 0 }, emptyBinding(&tensor), 1, false, false);
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

test "the session layout is 52 attention layers of two 256-float rows per position" {
    var layouts: [model.layer_count]session.Layout = undefined;
    for (&layouts) |*layout| layout.* = .{ .attention = .{ .key_row = model.kv_width, .value_row = model.kv_width } };
    var state = try session.Session.init(std.testing.allocator, &layouts, 4, false);
    defer state.deinit();
    const per_position = model.layer_count * 2 * model.kv_width * 4;
    try std.testing.expect(state.bytes() >= per_position * 4);
    try std.testing.expect(state.bytes() < per_position * 4 + model.layer_count * 2 * 16);
}
