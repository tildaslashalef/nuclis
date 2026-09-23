//! CPU execution of the pinned Gemma 4 text schedules (the dense 12B and
//! the 26B-A4B mixture of experts): the numerical reference
//! `gemma4_metal.zig` is compared against, written from the forward pass
//! recorded in docs/reference/gemma4.md. Immutable weight views and the
//! binding borrow the loaded model; this runtime owns the session (one
//! attention cache per layer, F32) and its workspace. A failed step poisons
//! the session. With a `gemma4-assistant` companion
//! bound the runtime also runs the draft head (`Head`), which reads the
//! target's caches and owns no attention state of its own.
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
//! - logits come from the embedding matrix and are soft-capped at 30;
//! - the draft head's four blocks read the target's layer-46 (sliding) and
//!   layer-47 (global) caches, write none, and chain only its `h_next`;
//!   `commit` is a copy of the last target hidden into `pending_h`.
const std = @import("std");
const model = @import("gemma4.zig");
const assistant = @import("gemma4_assistant.zig");
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

/// One draft block's decoded constants.
const HeadConstants = struct {
    attention_norm: []f32,
    post_attention_norm: []f32,
    ffn_norm: []f32,
    post_ffn_norm: []f32,
    query_norm: []f32,
    output_scale: f32,
};

/// The `gemma4-assistant` head's workspace and decoded constants. It owns no
/// attention cache: every block reads the target's cache rows, and the only
/// state is `pending_h`, the target hidden of the last committed token.
const Head = struct {
    binding: assistant.Binding,
    constants: [assistant.block_count]HeadConstants,
    output_norm: []f32,
    rope_factors: []f32,
    /// The target embedding row, scaled by sqrt(target width).
    row: []f32,
    x: []f32,
    normalized: []f32,
    concat: []f32,
    projected: []f32,
    q: []f32,
    mixed: []f32,
    gate: []f32,
    up: []f32,
    logits: []f32,
    h_next: []f32,
    chain: []f32,
    pending_h: []f32,

    fn init(alloc: std.mem.Allocator, binding: assistant.Binding) !Head {
        const head_width = assistant.embedding;
        const out = binding.config.embedding_out;
        var result: Head = undefined;
        result.binding = binding;
        for (&result.constants, binding.layers) |*c, layer| {
            c.* = .{
                .attention_norm = try binding.view.vector(alloc, layer.attention_norm),
                .post_attention_norm = try binding.view.vector(alloc, layer.post_attention_norm),
                .ffn_norm = try binding.view.vector(alloc, layer.ffn_norm),
                .post_ffn_norm = try binding.view.vector(alloc, layer.post_ffn_norm),
                .query_norm = try binding.view.vector(alloc, layer.query_norm),
                .output_scale = try binding.view.scalar(layer.output_scale, 0),
            };
            if (!std.math.isFinite(c.output_scale)) return error.InvalidShape;
        }
        result.output_norm = try binding.view.vector(alloc, binding.output_norm);
        result.rope_factors = try binding.view.vector(alloc, binding.rope_factors);
        result.row = try alloc.alloc(f32, out);
        inline for (.{ "x", "normalized", "projected" }) |field| @field(result, field) = try alloc.alloc(f32, head_width);
        inline for (.{ "q", "mixed" }) |field| @field(result, field) = try alloc.alloc(f32, assistant.Kind.global.queryWidth());
        inline for (.{ "gate", "up" }) |field| @field(result, field) = try alloc.alloc(f32, assistant.feed_forward);
        result.concat = try alloc.alloc(f32, 2 * out);
        result.logits = try alloc.alloc(f32, assistant.vocabulary);
        result.h_next = try alloc.alloc(f32, out);
        result.chain = try alloc.alloc(f32, out);
        result.pending_h = try alloc.alloc(f32, out);
        @memset(result.pending_h, 0);
        return result;
    }
    fn bytes(self: *const Head) usize {
        var total: usize = 0;
        inline for (@typeInfo(Head).@"struct".fields) |field| {
            if (field.type == []f32) total += @field(self, field.name).len * @sizeOf(f32);
        }
        // The head's binding borrows weights; only the decoded constants and
        // the workspace above are this runtime's own.
        return total + self.constants.len * (@sizeOf(HeadConstants) + 6 * assistant.embedding * @sizeOf(f32));
    }
};

pub const Runtime = struct {
    /// For an image span's per-call rows; the workspace lives in `storage`.
    gpa: std.mem.Allocator,
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
    /// The companion draft head, present only when one was bound and asked for.
    draft: ?Head,
    has_draft: bool,

    pub fn init(gpa: std.mem.Allocator, view: weights.View, binding: model.Binding, capacity: usize, checkpoint: bool, draft: bool) !Runtime {
        const config = binding.config;
        var layouts: [model.max_layers]session.Layout = undefined;
        for (binding.active(), layouts[0..config.layer_count]) |layer, *layout| {
            const width = layer.kvWidth();
            layout.* = .{ .attention = .{ .key_row = width, .value_row = width } };
        }
        var state = try session.Session.init(gpa, layouts[0..config.layer_count], capacity, checkpoint, 0);
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
        result.gpa = gpa;
        result.storage = storage;
        result.state = state;
        result.view = view;
        result.binding = binding;
        result.has_draft = draft and binding.draft != null;
        result.draft = if (result.has_draft) try Head.init(a, binding.draft.?) else null;
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
        // The post-`output_norm` hidden is kept on every step: it is the
        // drafter's `commit` input (and `verify`'s per-row hidden), so it must
        // not depend on whether the caller wanted logits.
        try norm(self.x, self.normalized, self.output_norm);
        if (logits) |out| {
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
        const position = self.state.position;
        const q = self.q[0..layer.queryWidth()];
        try self.project(layer, constants, il, self.normalized, position, q);
        try self.attend(layer, il, q, firstVisible(layer.kind, position), position + 1);
    }

    /// Sliding layers attend to the last `window` positions including the
    /// current one; the visible rows are a contiguous suffix of the cache.
    fn firstVisible(kind: model.Kind, position: usize) usize {
        return if (kind == .sliding and position + 1 > model.window) position + 1 - model.window else 0;
    }

    /// One row's projections at cache row `position`: its query (normed and
    /// rotated) into `q`, its key and value normed, rotated, and written to
    /// the layer's cache row.
    fn project(self: *Runtime, layer: model.Layer, constants: LayerConstants, il: usize, input: []const f32, position: usize, q: []f32) !void {
        const kind = layer.kind;
        const hd = layer.headSize();
        const kv_heads = layer.kv_heads;
        const k = self.k[0..layer.kvWidth()];
        const v = self.v[0..layer.kvWidth()];
        const rope: cpu.rope.Options = .{
            .dimensions = hd,
            .base = kind.ropeBase(),
            .position = @intCast(position),
            .factors = if (kind == .global) self.rope_factors else null,
        };
        try self.mm(layer.query, input, q);
        try self.mm(layer.key, input, k);
        // Global layers have no value projection: V is the raw key projection,
        // taken before the key norm and RoPE.
        if (layer.value) |value| try self.mm(value, input, v) else @memcpy(v, k);
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
    }

    /// Attention of `q` over the cache rows `[first, last)`, then the output
    /// projection into `projected`.
    fn attend(self: *Runtime, layer: model.Layer, il: usize, q: []const f32, first: usize, last: usize) !void {
        const hd = layer.headSize();
        const visible = last - first;
        const cache = self.state.layers[il].attention;
        const out = self.mixed_out[0..layer.queryWidth()];
        try cpu.attention.apply(.{
            .query_heads = model.heads,
            .kv_heads = layer.kv_heads,
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

    /// Prefills a prompt with image spans: text tokens step as usual; a
    /// span's rows are the projector's feature rows (`features`,
    /// `Σ span.count × embedding`, unscaled) and run as one batched pass
    /// (`prefillSpan`). `logits`, when given, receives the last row's.
    pub fn prefillVision(self: *Runtime, tokens: []const u32, spans: []const model.VisionSpan, features: []const f32, logits: ?[]f32, observer: ?Observer) !void {
        const width = self.binding.config.embedding;
        if (tokens.len == 0) return error.InvalidShape;
        if (tokens.len > self.state.capacity - self.state.position) return error.ContextFull;
        var i: usize = 0;
        var si: usize = 0;
        var frow: usize = 0;
        while (i < tokens.len) {
            if (si < spans.len and spans[si].start == i) {
                const sp = spans[si];
                if (sp.count == 0 or frow + sp.count > features.len / width) return error.InvalidShape;
                const last = i + sp.count == tokens.len;
                try self.prefillSpan(features[frow * width ..][0 .. sp.count * width], if (last) logits else null, observer);
                i += sp.count;
                frow += sp.count;
                si += 1;
            } else {
                try self.step(tokens[i], if (i + 1 == tokens.len) logits else null, observer);
                i += 1;
            }
        }
    }

    /// One image span of `rows.len / embedding` rows at the current
    /// position, layer by layer: every row projects and writes its cache
    /// row, then each row attends over `[first, span end)` on a sliding
    /// layer (the span is bidirectional there, the reference's
    /// `LLAMA_NON_CAUSAL_TYPE_SWA_ONLY`) and over `[0, row]` on a global one.
    /// Cache row, rotary position, and visible bound are the row's position.
    fn prefillSpan(self: *Runtime, rows: []const f32, logits: ?[]f32, observer: ?Observer) !void {
        const width = self.binding.config.embedding;
        const count = rows.len / width;
        if (count == 0 or rows.len != count * width) return error.InvalidShape;
        if (logits) |out| if (out.len != model.vocabulary) return error.InvalidShape;
        const q_stride = model.Kind.global.queryWidth();
        const xs = try self.gpa.dupe(f32, rows);
        defer self.gpa.free(xs);
        const qs = try self.gpa.alloc(f32, count * q_stride);
        defer self.gpa.free(qs);
        try self.state.beginChunk(count);
        errdefer self.state.fail();
        const start = self.state.position;
        for (self.binding.active(), self.constants, 0..) |layer, constants, il| {
            const qw = layer.queryWidth();
            for (0..count) |r| {
                try norm(xs[r * width ..][0..width], self.normalized, constants.attention_norm);
                try self.project(layer, constants, il, self.normalized, start + r, qs[r * q_stride ..][0..qw]);
            }
            for (0..count) |r| {
                const position = start + r;
                const last = if (layer.kind == .sliding) start + count else position + 1;
                try self.attend(layer, il, qs[r * q_stride ..][0..qw], firstVisible(layer.kind, position), last);
                const x = xs[r * width ..][0..width];
                @memcpy(self.x, x);
                try norm(self.projected, self.projected, constants.post_attention_norm);
                for (self.x, self.projected) |*v, contribution| v.* += contribution;
                try self.feedForward(layer, constants);
                for (self.x, self.projected) |*v, contribution| {
                    v.* = (v.* + contribution) * constants.output_scale;
                    if (!std.math.isFinite(v.*)) return error.NonFiniteResult;
                }
                @memcpy(x, self.x);
            }
            if (observer) |o| if (o.check) |check| try check(o.context);
        }
        // `x` holds the last row: its post-`output_norm` hidden, as `step` keeps.
        try norm(self.x, self.normalized, self.output_norm);
        if (logits) |out| {
            try self.mm(self.binding.token_embedding, self.normalized, out);
            for (out) |*v| {
                v.* = model.final_softcap * std.math.tanh(v.* / model.final_softcap);
                if (!std.math.isFinite(v.*)) return error.NonFiniteResult;
            }
        }
        try self.state.commitChunk(count);
    }

    /// Reference prefill: one step per token; `hidden`, when given
    /// (`tokens.len × embedding`), receives every row's post-`output_norm`
    /// hidden — the rows the drafter's `commit` consumes.
    pub fn prefill(self: *Runtime, tokens: []const u32, logits: ?[]f32, hidden: ?[]f32, observer: ?Observer) !void {
        const width = self.binding.config.embedding;
        if (hidden) |h| if (h.len != tokens.len * width) return error.InvalidShape;
        if (tokens.len > self.state.capacity - self.state.position) return error.ContextFull;
        for (tokens, 0..) |token, i| {
            try self.step(token, if (i + 1 == tokens.len) logits else null, observer);
            if (hidden) |h| @memcpy(h[i * width ..][0..width], self.normalized);
        }
    }

    // --- the draft head (MODL-19) --------------------------------------

    /// The target cache layer a head block reads: the target's last sliding
    /// layer for a sliding block, its last (global) layer otherwise.
    fn sourceLayer(self: *const Runtime, kind: assistant.Kind) usize {
        const n = self.binding.config.layer_count;
        return if (kind == .sliding) n - 2 else n - 1;
    }

    /// One proposed position: pair `token` with `head.pending_h`, project, run
    /// the four blocks over the target's caches at `position`, then leave the
    /// classifier logits (when asked) and the head's `h_next` in its
    /// workspace. `position` is the proposal position — the position of the
    /// token being proposed, which is not in the target cache — and the
    /// reference uses it for every chained block.
    fn headForward(self: *Runtime, h_prev: []const f32, token: u32, position: usize, logits: ?[]f32) !void {
        const head = &(self.draft orelse return error.NoDraftBlock);
        const out = head.binding.config.embedding_out;
        if (token >= model.vocabulary or position == 0 or position > self.state.capacity) return error.InvalidShape;
        if (logits) |values| if (values.len != model.vocabulary) return error.InvalidShape;
        // [sqrt(w)·embed_target(x_p); h_{p-1}] with no norm on either half.
        try self.view.row(self.binding.token_embedding, token, head.row);
        const scale: f32 = @sqrt(@as(f32, @floatFromInt(out)));
        for (head.row) |*v| v.* *= scale;
        @memcpy(head.concat[0..out], head.row);
        @memcpy(head.concat[out..], h_prev);
        try self.headMm(head, head.binding.pre_projection, head.concat, head.x);
        for (head.binding.layers, head.constants) |layer, constants| {
            try norm(head.x, head.normalized, constants.attention_norm);
            try self.headAttention(head, layer, constants, position);
            try norm(head.projected, head.projected, constants.post_attention_norm);
            for (head.x, head.projected) |*x, contribution| x.* += contribution;
            try norm(head.x, head.normalized, constants.ffn_norm);
            try self.headMm(head, layer.ffn_gate, head.normalized, head.gate);
            try self.headMm(head, layer.ffn_up, head.normalized, head.up);
            for (head.gate, head.up) |*g, u| g.* = cpu.gelu(g.*) * u;
            try self.headMm(head, layer.ffn_down, head.gate, head.projected);
            try norm(head.projected, head.projected, constants.post_ffn_norm);
            for (head.x, head.projected) |*x, contribution| {
                x.* = (x.* + contribution) * constants.output_scale;
                if (!std.math.isFinite(x.*)) return error.NonFiniteResult;
            }
        }
        try norm(head.x, head.normalized, head.output_norm);
        try self.headMm(head, head.binding.post_projection, head.normalized, head.h_next);
        if (logits) |values| {
            try self.headMm(head, head.binding.classifier, head.normalized, values);
            for (values) |v| if (!std.math.isFinite(v)) return error.NonFiniteResult;
        }
    }

    /// A head projection: the head's own file is the weight source, unlike
    /// the target's (`self.mm`).
    fn headMm(self: *Runtime, head: *Head, tensor: *const Tensor, input: []const f32, output: []f32) !void {
        try cpu.matvec(try head.binding.view.matrix(tensor), input, output, self.row);
    }

    fn headAttention(self: *Runtime, head: *Head, layer: assistant.Layer, constants: HeadConstants, position: usize) !void {
        const kind = layer.kind;
        const hd = layer.kind.headSize();
        const q = head.q[0..layer.queryWidth()];
        const rope: cpu.rope.Options = .{
            .dimensions = hd,
            .base = kind.ropeBase(),
            .position = @intCast(position),
            .factors = if (kind == .global) head.rope_factors else null,
        };
        try self.headMm(head, layer.query, head.normalized, q);
        for (0..assistant.heads) |h| {
            const channel = q[h * hd ..][0..hd];
            try norm(channel, channel, constants.query_norm);
            try cpu.rope.apply(channel, channel, rope);
        }
        // The target's rows at the proposal position: a sliding block sees
        // the last `window − 1` rows (the reference's mask is
        // `query − key >= window`, and the proposal's own row does not exist);
        // a global block sees every committed row.
        const cache = self.state.layers[self.sourceLayer(kind)].attention;
        const first = if (kind == .sliding and position >= model.window - 1) position - (model.window - 1) else 0;
        const visible = position - first;
        if (visible == 0) return error.EmptySupport;
        const out = head.mixed[0..layer.queryWidth()];
        try cpu.attention.apply(.{
            .query_heads = assistant.heads,
            .kv_heads = layer.kv_heads,
            .key_width = hd,
            .value_width = hd,
            .tokens = visible,
            .visible_tokens = visible,
            .scale = 1.0,
            .queries = q,
            .keys = cache.keys.floats(first, visible),
            .values = cache.values.floats(first, visible),
        }, out, self.attention_scratch);
        try self.headMm(head, layer.output, out, head.projected);
    }

    /// One head row for a trace: the caller supplies the previous position's
    /// target hidden, exactly as `propose` would, and receives the head's
    /// `h_next` and greedy token for the position. The target session must
    /// already hold rows `0 .. position − 1`.
    pub fn draftForwardTrace(self: *Runtime, h_prev: []const f32, token: u32, position: usize, h_out: []f32, greedy: ?*u32, logits: ?[]f32) !void {
        const head = &(self.draft orelse return error.NoDraftBlock);
        const out = head.h_next.len;
        if (h_prev.len != out or h_out.len != out) return error.InvalidShape;
        try self.headForward(h_prev, token, position, head.logits);
        @memcpy(h_out, head.h_next);
        if (greedy) |value| value.* = argmax(head.logits);
        if (logits) |values| @memcpy(values, head.logits);
    }

    /// Greedy candidates from the state after the last committed token, each
    /// block step at the target's position and chained through the head's own
    /// `h_next`; `out.len` bounds the count. `p_min > 0` stops after a
    /// position whose top candidate's softmax probability is below it.
    pub fn propose(self: *Runtime, token: u32, out: []u32, p_min: f32) !usize {
        if (!self.has_draft) return error.NoDraftBlock;
        if (!std.math.isFinite(p_min) or p_min < 0 or p_min > 1) return error.InvalidShape;
        const head = &self.draft.?;
        const position = self.state.position;
        var h_prev: []const f32 = head.pending_h;
        var next = token;
        var count: usize = 0;
        while (count < out.len) : (count += 1) {
            try self.headForward(h_prev, next, position, head.logits);
            out[count] = argmax(head.logits);
            if (p_min > 0) {
                var maximum: f64 = -std.math.inf(f64);
                for (head.logits) |v| maximum = @max(maximum, v);
                var total: f64 = 0;
                for (head.logits) |v| total += @exp(@as(f64, v) - maximum);
                if (total > 0 and 1.0 / total < p_min) {
                    count += 1;
                    break;
                }
            }
            next = out[count];
            h_prev = head.h_next;
        }
        return count;
    }

    /// Advances the head over tokens the main model committed: its only state
    /// is the target hidden of the last committed token, which the next
    /// `propose` pairs with the next seed. `h_rows` is
    /// `tokens.len × embedding_out`.
    pub fn commit(self: *Runtime, tokens: []const u32, h_rows: []const f32) !void {
        const head = &(self.draft orelse return error.NoDraftBlock);
        const width = head.pending_h.len;
        if (h_rows.len != tokens.len * width) return error.InvalidShape;
        if (tokens.len == 0) return;
        @memcpy(head.pending_h, h_rows[h_rows.len - width ..][0..width]);
    }

    /// Reference verify: one step per token, keeping every row's logits and,
    /// when asked, the post-`output_norm` hidden the drafter's `commit` reads.
    pub fn verify(self: *Runtime, tokens: []const u32, rows: []f32, h_rows: ?[]f32, observer: ?Observer) !void {
        const width = self.binding.config.embedding;
        if (rows.len != tokens.len * model.vocabulary) return error.InvalidShape;
        if (h_rows) |h| if (h.len != tokens.len * width) return error.InvalidShape;
        for (tokens, 0..) |token, i| {
            try self.step(token, rows[i * model.vocabulary ..][0..model.vocabulary], observer);
            if (h_rows) |h| @memcpy(h[i * width ..][0..width], self.normalized);
        }
    }

    /// `verify`'s greedy sibling: the same steps with a per-row argmax instead
    /// of full logits, reusing the head's logits scratch (a drafter is loaded
    /// whenever the loop speculates).
    pub fn verifyGreedy(self: *Runtime, tokens: []const u32, out: []u32, h_rows: ?[]f32, observer: ?Observer) !void {
        const head = &(self.draft orelse return error.NoDraftBlock);
        const width = self.binding.config.embedding;
        if (out.len != tokens.len) return error.InvalidShape;
        if (h_rows) |h| if (h.len != tokens.len * width) return error.InvalidShape;
        for (tokens, 0..) |token, i| {
            try self.step(token, head.logits, observer);
            out[i] = argmax(head.logits);
            if (h_rows) |h| @memcpy(h[i * width ..][0..width], self.normalized);
        }
    }

    /// The contract value the engine holds, or null when no head is loaded.
    pub fn drafter(self: *Runtime) ?@import("../runtime/draft.zig").Drafter {
        if (!self.has_draft) return null;
        const head = &self.draft.?;
        return .{ .host = self, .hidden = head.binding.config.embedding_out, .max_proposals = model.max_draft_proposals, .propose_fn = proposeFn, .commit_fn = commitFn, .reset_fn = resetDraftFn, .bytes_fn = draftBytes };
    }
    fn proposeFn(host: *anyopaque, token: u32, out: []u32, p_min: f32) anyerror!usize {
        const self: *Runtime = @ptrCast(@alignCast(host));
        return self.propose(token, out, p_min);
    }
    fn commitFn(host: *anyopaque, tokens: []const u32, h_rows: []const f32) anyerror!void {
        const self: *Runtime = @ptrCast(@alignCast(host));
        return self.commit(tokens, h_rows);
    }
    fn resetDraftFn(host: *anyopaque) void {
        const self: *Runtime = @ptrCast(@alignCast(host));
        if (self.draft) |*head| @memset(head.pending_h, 0);
    }
    fn draftBytes(host: *anyopaque) usize {
        const self: *Runtime = @ptrCast(@alignCast(host));
        if (self.draft) |*head| return head.bytes();
        return 0;
    }

    fn argmax(values: []const f32) u32 {
        var best: usize = 0;
        for (values, 0..) |v, i| {
            if (v > values[best]) best = i;
        }
        return @intCast(best);
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
        var state = try session.Session.init(std.testing.allocator, layouts[0..case.config.layer_count], 4, false, 0);
        defer state.deinit();
        try std.testing.expect(state.bytes() >= case.per_position * 4);
        try std.testing.expect(state.bytes() < case.per_position * 4 + case.config.layer_count * 2 * 16);
    }
}
