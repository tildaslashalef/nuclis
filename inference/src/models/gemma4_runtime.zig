//! CPU execution of the pinned Gemma 4 text schedules (the dense 12B and
//! the 26B-A4B mixture of experts): the numerical reference
//! `gemma4_metal.zig` is compared against, written from the forward pass
//! recorded in docs/reference/gemma4.md. Immutable weight views and the
//! binding borrow the loaded model; this runtime owns the session (one
//! attention cache per layer, F32) and its workspace. A failed step poisons
//! the session. Text only: no vision, no MTP head.
//!
//! What differs from the Qwen runtime, operation by operation:
//! - the embedding row is scaled by sqrt(width) after decoding;
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
//! - on an expert layer the dense FFN is a shared branch: its output and the
//!   routed experts' sum each get their own post norm before they are added
//!   and the ordinary post-FFN norm applies to the sum (`feedForward`);
//! - logits come from the embedding matrix and are soft-capped at 30.
const std = @import("std");
const model = @import("gemma4.zig");
const weights = @import("../runtime/weights.zig");
const session = @import("../runtime/session.zig");
const cpu = @import("../backends/cpu/root.zig");
const Tensor = @import("../formats/gguf.zig").Tensor;

pub const Observer = @import("../runtime/observer.zig").Observer;

/// Small F32 weights of the expert block decoded once per layer at init.
const ExpertConstants = struct {
    router_scale: []f32,
    down_scale: []f32,
    post_ffn_norm_1: []f32,
    pre_ffn_norm_2: []f32,
    post_ffn_norm_2: []f32,
};

/// Small F32 weights decoded once per layer at init.
const LayerConstants = struct {
    attention_norm: []f32,
    post_attention_norm: []f32,
    ffn_norm: []f32,
    post_ffn_norm: []f32,
    query_norm: []f32,
    key_norm: []f32,
    output_scale: f32,
    experts: ?ExpertConstants,
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
    // The expert block's workspace; empty on the dense configuration.
    router_logits: []f32,
    indices: []u32,
    route_weights: []f32,
    expert_out: []f32,
    expert_scratch: []f32,
    accumulator: []f64,

    pub fn init(gpa: std.mem.Allocator, view: weights.View, binding: model.Binding, capacity: usize, checkpoint: bool, draft: bool) !Runtime {
        _ = draft;
        const config = binding.config;
        var layouts: [model.max_layers]session.Layout = undefined;
        for (binding.active(), layouts[0..config.layer_count]) |layer, *layout| {
            const width = layer.kvWidth();
            layout.* = .{ .attention = .{ .key_row = width, .value_row = width } };
        }
        var state = try session.Session.init(gpa, layouts[0..config.layer_count], capacity, checkpoint);
        errdefer state.deinit();
        var storage: std.heap.ArenaAllocator = .init(gpa);
        errdefer storage.deinit();
        const a = storage.allocator();
        // Finish allocations before transferring the arena, whose internal
        // linked-list head can change on each allocation.
        var result: Runtime = undefined;
        inline for (.{ "x", "normalized", "projected", "expert_out" }) |field| @field(result, field) = try a.alloc(f32, config.embedding);
        inline for (.{ "gate", "up" }) |field| @field(result, field) = try a.alloc(f32, config.feed_forward);
        // Sized for the wider (global) geometry; sliding layers use a prefix.
        result.q = try a.alloc(f32, model.Kind.global.queryWidth());
        result.mixed_out = try a.alloc(f32, model.Kind.global.queryWidth());
        result.k = try a.alloc(f32, model.max_kv_width);
        result.v = try a.alloc(f32, model.max_kv_width);
        result.attention_scratch = try a.alloc(f64, capacity);
        const experts = config.experts orelse model.Experts{ .count = 0, .used = 0, .feed_forward = 0 };
        result.router_logits = try a.alloc(f32, experts.count);
        result.indices = try a.alloc(u32, experts.used);
        result.route_weights = try a.alloc(f32, experts.used);
        // The matvec decode row must hold the widest row any projection
        // decodes: the global attention output's 16 × 512 columns, the FFN
        // down projection's, or the embedding width (the expert router).
        result.row = try a.alloc(f32, @max(@max(config.feed_forward, config.embedding), model.Kind.global.queryWidth()));
        result.expert_scratch = try a.alloc(f32, 2 * experts.feed_forward + experts.feed_forward + config.embedding + @max(config.embedding, experts.feed_forward));
        result.accumulator = try a.alloc(f64, config.embedding);
        result.output_norm = try view.vector(a, binding.output_norm);
        result.rope_factors = try view.vector(a, binding.rope_factors);
        result.constants = try a.alloc(LayerConstants, config.layer_count);
        for (binding.active(), result.constants) |layer, *constants| {
            constants.* = .{
                .attention_norm = try view.vector(a, layer.attention_norm),
                .post_attention_norm = try view.vector(a, layer.post_attention_norm),
                .ffn_norm = try view.vector(a, layer.ffn_norm),
                .post_ffn_norm = try view.vector(a, layer.post_ffn_norm),
                .query_norm = try view.vector(a, layer.query_norm),
                .key_norm = try view.vector(a, layer.key_norm),
                .output_scale = try view.scalar(layer.output_scale, 0),
                .experts = if (layer.experts) |e| .{
                    .router_scale = try view.vector(a, e.router_scale),
                    .down_scale = try view.vector(a, e.down_scale),
                    .post_ffn_norm_1 = try view.vector(a, e.post_ffn_norm_1),
                    .pre_ffn_norm_2 = try view.vector(a, e.pre_ffn_norm_2),
                    .post_ffn_norm_2 = try view.vector(a, e.post_ffn_norm_2),
                } else null,
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
        const embedding_scale = self.binding.config.embeddingScale();
        for (self.x) |*x| x.* *= embedding_scale;
        for (self.binding.active(), self.constants, 0..) |layer, constants, il| {
            try norm(self.x, self.normalized, constants.attention_norm);
            try self.attention(layer, constants, il);
            try norm(self.projected, self.projected, constants.post_attention_norm);
            for (self.x, self.projected) |*x, contribution| x.* += contribution;
            try self.feedForward(layer, constants);
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

    /// The feed-forward block over the residual `x` (the attention output
    /// already added) into `projected`, post-normed and ready to add. On
    /// an expert layer the dense FFN is the shared branch: `post_ffn_norm_1`
    /// on its output, the router over `rms(x) / sqrt(width) ⊙ router_scale`
    /// (the pre-norm residual, not a normed copy), the gathered experts over
    /// `pre_ffn_norm_2(x)`, `post_ffn_norm_2` on their weighted sum, then
    /// the two branches add and the ordinary post norm applies to the sum.
    fn feedForward(self: *Runtime, layer: model.Layer, constants: LayerConstants) !void {
        try norm(self.x, self.normalized, constants.ffn_norm);
        try self.mm(layer.ffn_gate, self.normalized, self.gate);
        try self.mm(layer.ffn_up, self.normalized, self.up);
        for (self.gate, self.up) |*g, u| g.* = cpu.gelu(g.*) * u;
        try self.mm(layer.ffn_down, self.gate, self.projected);
        if (layer.experts) |experts| {
            const ec = constants.experts orelse return error.InvalidShape;
            try norm(self.projected, self.projected, ec.post_ffn_norm_1);
            try cpu.rmsNorm(self.x, self.normalized, model.rms_epsilon);
            // The reference scales by 1/sqrt(width) first, then by the vector.
            const inverse_root: f32 = 1.0 / self.binding.config.embeddingScale();
            for (self.normalized, ec.router_scale) |*t, s| t.* = (t.* * inverse_root) * s;
            try self.mm(experts.router, self.normalized, self.router_logits);
            try cpu.experts.route(self.router_logits, self.indices, self.route_weights);
            try norm(self.x, self.normalized, ec.pre_ffn_norm_2);
            const spec: cpu.experts.Ffn = .{
                .gate_up = try self.view.expertMatrix(experts.gate_up),
                .down = try self.view.expertMatrix(experts.down),
                .down_scale = ec.down_scale,
            };
            try cpu.experts.ffn(spec, self.normalized, self.indices, self.route_weights, self.expert_out, self.expert_scratch, self.accumulator);
            try norm(self.expert_out, self.expert_out, ec.post_ffn_norm_2);
            for (self.projected, self.expert_out) |*p, e| p.* += e;
        }
        try norm(self.projected, self.projected, constants.post_ffn_norm);
    }

    fn attention(self: *Runtime, layer: model.Layer, constants: LayerConstants, il: usize) !void {
        const kind = layer.kind;
        const hd = layer.headSize();
        const kv_heads = layer.kv_heads;
        const q = self.q[0..layer.queryWidth()];
        const k = self.k[0..layer.kvWidth()];
        const v = self.v[0..layer.kvWidth()];
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
        const out = self.mixed_out[0..layer.queryWidth()];
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

/// A binding over empty tensors for the workspace tests: every weight read
/// fails after admission, which is what the invalid-step checks need. The
/// per-layer output scale is read at init, so it points at `scale`.
fn emptyBinding(config: *const model.Config, tensor: *const Tensor, scalar: *const Tensor) model.Binding {
    var binding: model.Binding = undefined;
    binding.config = config;
    binding.token_embedding = tensor;
    binding.output_norm = tensor;
    binding.rope_factors = tensor;
    for (binding.layers[0..config.layer_count], 0..) |*layer, i| {
        const kind = model.kindOf(i);
        layer.* = .{ .kind = kind, .kv_heads = model.kvHeadsOf(config, kind), .attention_norm = tensor, .post_attention_norm = tensor, .ffn_norm = tensor, .post_ffn_norm = tensor, .query = tensor, .key = tensor, .value = if (kind == .sliding) tensor else null, .output = tensor, .query_norm = tensor, .key_norm = tensor, .ffn_gate = tensor, .ffn_up = tensor, .ffn_down = tensor, .output_scale = scalar, .experts = if (config.experts != null) .{ .router = tensor, .router_scale = tensor, .gate_up = tensor, .down = tensor, .down_scale = tensor, .post_ffn_norm_1 = tensor, .pre_ffn_norm_2 = tensor, .post_ffn_norm_2 = tensor } else null };
    }
    binding.summary = .{ .profile = config.profile, .decoder_layers = @intCast(config.layer_count), .layer_kinds = &.{}, .text_tensors = 0, .auxiliary_tensors = 0, .text_tensor_bytes = 0, .auxiliary_tensor_bytes = 0 };
    return binding;
}

test "runtime workspace cleanup and invalid steps preserve session admission" {
    inline for (.{ &model.config_12b, &model.config_26b_a4b }) |config| {
        try std.testing.checkAllAllocationFailures(std.testing.allocator, struct {
            fn check(alloc: std.mem.Allocator, cfg: *const model.Config) !void {
                const tensor: Tensor = .{ .name = "empty", .dimensions = &.{0}, .encoding_id = 0, .offset = 0, .elements = 0, .bytes = 0 };
                const scalar: Tensor = .{ .name = "scale", .dimensions = &.{1}, .encoding_id = 0, .offset = 0, .elements = 1, .bytes = 4 };
                const file = [_]u8{ 0, 0, 0, 0 };
                var runtime = try Runtime.init(alloc, .{ .file = &file, .data_offset = 0 }, emptyBinding(cfg, &tensor, &scalar), 1, false, false);
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
        }.check, .{config});
    }
}

test "session layouts follow the layer kinds and the configuration's KV heads" {
    // Not runnable without weights; the layout is checked through the
    // session's byte size. 12B: 40 sliding layers of 2 × 2048 F32 rows and
    // 8 global layers of 2 × 512 per position; 26B-A4B: 25 × 2 × 2048 and
    // 5 global layers of 2 × 1024 (two KV heads of 512).
    const cases = [_]struct { config: *const model.Config, per_position: usize }{
        .{ .config = &model.config_12b, .per_position = 40 * 2 * 2048 * 4 + 8 * 2 * 512 * 4 },
        .{ .config = &model.config_26b_a4b, .per_position = 25 * 2 * 2048 * 4 + 5 * 2 * 1024 * 4 },
    };
    for (cases) |case| {
        var layouts: [model.max_layers]session.Layout = undefined;
        for (layouts[0..case.config.layer_count], 0..) |*layout, i| {
            const width = model.kvHeadsOf(case.config, model.kindOf(i)) * model.kindOf(i).headSize();
            layout.* = .{ .attention = .{ .key_row = width, .value_row = width } };
        }
        var state = try session.Session.init(std.testing.allocator, layouts[0..case.config.layer_count], 4, false);
        defer state.deinit();
        try std.testing.expect(state.bytes() >= case.per_position * 4);
        try std.testing.expect(state.bytes() < case.per_position * 4 + case.config.layer_count * 2 * 16);
    }
}
