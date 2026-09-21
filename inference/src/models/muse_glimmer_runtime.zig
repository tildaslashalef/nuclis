//! CPU execution of the pinned Muse Glimmer 30B schedule: the numerical
//! reference `muse_glimmer_metal.zig` is compared against, written from the
//! forward pass recorded in docs/reference/muse-glimmer.md. Immutable weight
//! views and the binding borrow the loaded model; this runtime owns the
//! session (one attention cache per layer, F32) and its workspace. A failed
//! step poisons the session. Text only: no vision.
//!
//! When the DFlash companion is bound the runtime also runs the drafter
//! (`Draft`): the language model keeps the input residual of the five target
//! layers `dflash.target_layers` for every row it consumes (`prefill`,
//! `verify`, `verifyGreedy` with `hidden`), and the drafter turns five
//! residuals into one cached feature (`commit`) from which a 16-row mask
//! block proposes up to 15 drafts in one forward (`propose`). Its five
//! attention caches are session layouts after the language model's, so the
//! session's position rewind covers them.
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
const dflash = @import("dflash.zig");
const weights = @import("../runtime/weights.zig");
const session = @import("../runtime/session.zig");
const draft_contract = @import("../runtime/draft.zig");
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

/// One DFlash block's decoded constants.
const DraftConstants = struct {
    attention_norm: []f32,
    query_norm: []f32,
    key_norm: []f32,
    ffn_norm: []f32,
};

/// The DFlash companion's workspace and decoded constants. It owns no session
/// memory: the drafter's five attention caches are session layouts
/// `model.layer_count .. model.layer_count + dflash.block_count`.
const Draft = struct {
    binding: dflash.Binding,
    constants: [dflash.block_count]DraftConstants,
    encoder_norm: []f32,
    output_norm: []f32,
    /// The encoder output, also the residual stream through the blocks.
    encoded: []f32,
    normalized: []f32,
    /// Every noise row's final normed hidden, `block_size` rows.
    hidden: []f32,
    /// The rows' residual streams while the block runs, `block_size` rows.
    x_rows: []f32,
    /// The rows' projected queries, held across the two passes of a layer.
    q_rows: []f32,
    q: []f32,
    k: []f32,
    v: []f32,
    mixed: []f32,
    gate: []f32,
    up: []f32,
    /// The widest matvec scratch: the encoder's `hidden_width` input.
    row: []f32,
    logits: []f32,
};

/// The target layer whose input residual is kept at capture slot `index`,
/// or null for every other layer.
fn targetSlot(index: usize) ?usize {
    inline for (dflash.target_layers, 0..) |layer, slot| if (index == layer) return slot;
    return null;
}

pub const Runtime = struct {
    storage: std.heap.ArenaAllocator,
    state: session.Session,
    view: weights.View,
    binding: model.Binding,
    constants: []LayerConstants,
    output_norm: []f32,
    draft: ?Draft,
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
        const has_draft = draft and binding.draft != null;
        var layouts: [model.layer_count + dflash.block_count]session.Layout = undefined;
        for (layouts[0..model.layer_count]) |*layout| layout.* = .{ .attention = .{ .key_row = model.kv_width, .value_row = model.kv_width } };
        if (has_draft) {
            for (layouts[model.layer_count..]) |*layout| layout.* = .{ .attention = .{ .key_row = dflash.kv_width, .value_row = dflash.kv_width } };
        }
        var state = try session.Session.init(gpa, layouts[0 .. model.layer_count + @as(usize, if (has_draft) dflash.block_count else 0)], capacity, checkpoint, 0);
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
        if (has_draft) result.draft = try draftWorkspace(a, binding.draft.?) else result.draft = null;
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

    /// The drafter's buffers and decoded norms, bound to the companion's own
    /// view (never the target's).
    fn draftWorkspace(a: std.mem.Allocator, binding: dflash.Binding) !Draft {
        var result: Draft = undefined;
        result.binding = binding;
        result.encoded = try a.alloc(f32, dflash.embedding);
        result.normalized = try a.alloc(f32, dflash.embedding);
        result.hidden = try a.alloc(f32, dflash.block_size * dflash.embedding);
        result.x_rows = try a.alloc(f32, dflash.block_size * dflash.embedding);
        result.q_rows = try a.alloc(f32, dflash.block_size * dflash.query_width);
        result.q = try a.alloc(f32, dflash.query_width);
        result.k = try a.alloc(f32, dflash.kv_width);
        result.v = try a.alloc(f32, dflash.kv_width);
        result.mixed = try a.alloc(f32, dflash.query_width);
        result.gate = try a.alloc(f32, dflash.feed_forward);
        result.up = try a.alloc(f32, dflash.feed_forward);
        result.row = try a.alloc(f32, dflash.hidden_width);
        result.logits = try a.alloc(f32, model.vocabulary);
        result.encoder_norm = try binding.view.vector(a, binding.encoder_norm);
        result.output_norm = try binding.view.vector(a, binding.output_norm);
        for (binding.layers, &result.constants) |layer, *constants| {
            constants.* = .{
                .attention_norm = try binding.view.vector(a, layer.attention_norm),
                .query_norm = try binding.view.vector(a, layer.query_norm),
                .key_norm = try binding.view.vector(a, layer.key_norm),
                .ffn_norm = try binding.view.vector(a, layer.ffn_norm),
            };
        }
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
        return self.forward(token, logits, null, observer);
    }

    /// One token through the language model. `capture`, when given
    /// (`dflash.hidden_width` values), receives the five target layers' input
    /// residuals in `dflash.target_layers` order — the rows the DFlash
    /// drafter's `commit` consumes.
    fn forward(self: *Runtime, token: u32, logits: ?[]f32, capture: ?[]f32, observer: ?Observer) !void {
        if (token >= model.vocabulary) return error.InvalidTokenId;
        if (logits) |out| if (out.len != model.vocabulary) return error.InvalidShape;
        if (capture) |c| if (c.len != dflash.hidden_width) return error.InvalidShape;
        try self.state.begin();
        errdefer self.state.fail();
        try self.view.row(self.binding.token_embedding, token, self.x);
        try cpu.rmsNorm(self.x, self.x, model.rms_epsilon);
        for (self.binding.active(), self.constants, 0..) |layer, constants, il| {
            // The layer's input residual, before any of the layer's own
            // writes; the reference exposes it as `t_layer_inp`.
            if (capture) |c| if (targetSlot(il)) |slot| @memcpy(c[slot * model.embedding ..][0..model.embedding], self.x);
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

    /// Reference prefill: one step per token; `hidden`, when given
    /// (`tokens.len × dflash.hidden_width`), receives every row's five layer
    /// input residuals in `dflash.target_layers` order — the rows the
    /// drafter's `commit` consumes. Without a bound drafter `hidden` is
    /// `HiddenUnsupported`.
    pub fn prefill(self: *Runtime, tokens: []const u32, logits: ?[]f32, hidden: ?[]f32, observer: ?Observer) !void {
        if (hidden != null and self.draft == null) return error.HiddenUnsupported;
        if (hidden) |h| if (h.len != tokens.len * dflash.hidden_width) return error.InvalidShape;
        if (tokens.len > self.state.capacity - self.state.position) return error.ContextFull;
        for (tokens, 0..) |token, i| {
            const capture: ?[]f32 = if (hidden) |h| h[i * dflash.hidden_width ..][0..dflash.hidden_width] else null;
            try self.forward(token, if (i + 1 == tokens.len) logits else null, capture, observer);
        }
    }

    /// Reference verify: one step per token, keeping every row's logits and,
    /// when asked, the five layer input residuals per row.
    pub fn verify(self: *Runtime, tokens: []const u32, rows: []f32, h_rows: ?[]f32, observer: ?Observer) !void {
        if (rows.len != tokens.len * model.vocabulary) return error.InvalidShape;
        if (h_rows) |h| if (h.len != tokens.len * dflash.hidden_width) return error.InvalidShape;
        if (h_rows != null and self.draft == null) return error.HiddenUnsupported;
        for (tokens, 0..) |token, i| {
            const capture: ?[]f32 = if (h_rows) |h| h[i * dflash.hidden_width ..][0..dflash.hidden_width] else null;
            try self.forward(token, rows[i * model.vocabulary ..][0..model.vocabulary], capture, observer);
        }
    }

    /// Reference verify, greedy: the same steps with a per-row argmax instead
    /// of full logits. Reuses the drafter's logits scratch, which exists
    /// exactly when a drafter does.
    pub fn verifyGreedy(self: *Runtime, tokens: []const u32, out: []u32, h_rows: ?[]f32, observer: ?Observer) !void {
        const d = &(self.draft orelse return error.NoDraftBlock);
        if (out.len != tokens.len) return error.InvalidShape;
        if (h_rows) |h| if (h.len != tokens.len * dflash.hidden_width) return error.InvalidShape;
        for (tokens, 0..) |token, i| {
            const capture: ?[]f32 = if (h_rows) |h| h[i * dflash.hidden_width ..][0..dflash.hidden_width] else null;
            try self.forward(token, null, capture, observer);
            try norm(self.x, self.normalized, self.output_norm, model.rms_epsilon);
            try self.mm(self.binding.output, self.normalized, d.logits);
            out[i] = argmax(d.logits);
        }
    }

    fn argmax(values: []const f32) u32 {
        var best: usize = 0;
        for (values, 0..) |v, i| {
            if (v > values[best]) best = i;
        }
        return @intCast(best);
    }

    // --- the DFlash drafter (MODL-20) ----------------------------------

    fn draftMm(d: *Draft, tensor: *const Tensor, input: []const f32, output: []f32) !void {
        try cpu.matvec(try d.binding.view.matrix(tensor), input, output, d.row);
    }

    /// Five target residuals to one feature: `enc.output_norm(fc(row))`.
    fn draftEncode(self: *Runtime, residuals: []const f32, out: []f32) !void {
        const d = &(self.draft orelse return error.NoDraftBlock);
        if (residuals.len != dflash.hidden_width or out.len != dflash.embedding) return error.InvalidShape;
        try draftMm(d, d.binding.fc, residuals, out);
        try cpu.rmsNorm(out, out, dflash.rms_epsilon);
        for (out, d.encoder_norm) |*v, w| v.* *= w;
    }

    /// Injects the encoder output of one position as the blocks' keys and
    /// values: `k_norm(rk(feature))` and `rv(feature)`, RoPE at the position.
    fn draftInject(self: *Runtime, position: usize) !void {
        const d = &(self.draft orelse return error.NoDraftBlock);
        const rope: cpu.rope.Options = .{ .dimensions = dflash.head_size, .base = dflash.rope_base, .position = @intCast(position) };
        for (d.binding.layers, 0..) |layer, il| {
            try draftMm(d, layer.key, d.encoded, d.k);
            try draftMm(d, layer.value, d.encoded, d.v);
            for (0..dflash.kv_heads) |h| {
                const head = d.k[h * dflash.head_size ..][0..dflash.head_size];
                try norm(head, head, d.constants[il].key_norm, dflash.rms_epsilon);
                try cpu.rope.apply(head, head, rope);
            }
            const cache = self.state.layers[model.layer_count + il].attention;
            @memcpy(cache.keys.floats(position, 1), d.k);
            @memcpy(cache.values.floats(position, 1), d.v);
        }
    }

    /// Runs the noise block `rows` rows from `token` at the session position:
    /// row 0 is the anchor, the rest the mask token. Every row's final normed
    /// hidden lands in `d.hidden`; the blocks' key/value rows are written at
    /// their positions, and attention is the pinned configuration's —
    /// non-causal inside the block, each row seeing the injected prefix back
    /// to its sliding window.
    ///
    /// A layer runs in two passes: every row's keys and values enter the
    /// cache first (the reference's one ubatch copies all rows before
    /// attention), then every row attends. The rows' residual streams and
    /// projected queries are held in `x_rows`/`q_rows` across the passes.
    fn draftBlock(self: *Runtime, token: u32, rows: usize) !void {
        const d = &(self.draft orelse return error.NoDraftBlock);
        if (rows == 0 or rows > dflash.block_size) return error.InvalidShape;
        const base = self.state.position;
        if (base + rows > self.state.capacity) return error.ContextFull;
        const last = base + rows - 1;
        // Embed the noise block: the anchor token then the mask rows.
        for (0..rows) |r| {
            const input = if (r == 0) token else dflash.mask_token;
            try self.view.row(self.binding.token_embedding, input, d.x_rows[r * dflash.embedding ..][0..dflash.embedding]);
        }
        for (d.binding.layers, 0..) |layer, il| {
            const cache = self.state.layers[model.layer_count + il].attention;
            const constants = d.constants[il];
            for (0..rows) |r| {
                const position = base + r;
                const x = d.x_rows[r * dflash.embedding ..][0..dflash.embedding];
                try norm(x, d.normalized, constants.attention_norm, dflash.rms_epsilon);
                const q = d.q_rows[r * dflash.query_width ..][0..dflash.query_width];
                try draftMm(d, layer.query, d.normalized, q);
                try draftMm(d, layer.key, d.normalized, d.k);
                try draftMm(d, layer.value, d.normalized, d.v);
                const rope: cpu.rope.Options = .{ .dimensions = dflash.head_size, .base = dflash.rope_base, .position = @intCast(position) };
                for (0..dflash.heads) |h| {
                    const head = q[h * dflash.head_size ..][0..dflash.head_size];
                    try norm(head, head, constants.query_norm, dflash.rms_epsilon);
                    try cpu.rope.apply(head, head, rope);
                }
                for (0..dflash.kv_heads) |h| {
                    const head = d.k[h * dflash.head_size ..][0..dflash.head_size];
                    try norm(head, head, constants.key_norm, dflash.rms_epsilon);
                    try cpu.rope.apply(head, head, rope);
                }
                @memcpy(cache.keys.floats(position, 1), d.k);
                @memcpy(cache.values.floats(position, 1), d.v);
            }
            for (0..rows) |r| {
                const position = base + r;
                const x = d.x_rows[r * dflash.embedding ..][0..dflash.embedding];
                const q = d.q_rows[r * dflash.query_width ..][0..dflash.query_width];
                // Rows inside the block see each other regardless of order;
                // the visible prefix is the row's own sliding window. Rows
                // past the block are stale proposal rows and never read.
                const first = if (position + 1 > dflash.window) position + 1 - dflash.window else 0;
                const visible = last - first + 1;
                try cpu.attention.apply(.{
                    .query_heads = dflash.heads,
                    .kv_heads = dflash.kv_heads,
                    .key_width = dflash.head_size,
                    .value_width = dflash.head_size,
                    .tokens = visible,
                    .visible_tokens = visible,
                    .scale = model.attention_scale,
                    .queries = q,
                    .keys = cache.keys.floats(first, visible),
                    .values = cache.values.floats(first, visible),
                }, d.mixed, self.attention_scratch);
                try draftMm(d, layer.output, d.mixed, d.encoded);
                for (x, d.encoded) |*value, contribution| value.* += contribution;
                try norm(x, d.normalized, constants.ffn_norm, dflash.rms_epsilon);
                try draftMm(d, layer.ffn_gate, d.normalized, d.gate);
                try draftMm(d, layer.ffn_up, d.normalized, d.up);
                for (d.gate, d.up) |*g, u| g.* = cpu.silu(g.*) * u;
                try draftMm(d, layer.ffn_down, d.gate, d.encoded);
                for (x, d.encoded) |*value, contribution| {
                    value.* += contribution;
                    if (!std.math.isFinite(value.*)) return error.NonFiniteResult;
                }
            }
        }
        for (0..rows) |r| {
            const x = d.x_rows[r * dflash.embedding ..][0..dflash.embedding];
            try norm(x, d.hidden[r * dflash.embedding ..][0..dflash.embedding], d.output_norm, dflash.rms_epsilon);
        }
    }

    /// Advances the drafter over tokens the main model committed: each token's
    /// five residuals (`tokens.len × dflash.hidden_width`) encode to one
    /// feature, and its keys and values enter the drafter's cache at the
    /// token's position. The accepted prefix ends at `state.position`; the
    /// caller has already `recover`ed the session. The token ids themselves
    /// are the contract's, not the encoder's input: the reference's embd batch
    /// carries none.
    pub fn commit(self: *Runtime, tokens: []const u32, h_rows: []const f32) !void {
        if (self.draft == null) return error.NoDraftBlock;
        if (h_rows.len != tokens.len * dflash.hidden_width) return error.InvalidShape;
        if (tokens.len == 0) return;
        if (self.state.position < tokens.len) return error.InvalidShape;
        const start = self.state.position - tokens.len;
        for (0..tokens.len) |i| {
            try self.draftEncode(h_rows[i * dflash.hidden_width ..][0..dflash.hidden_width], self.draft.?.encoded);
            try self.draftInject(start + i);
        }
    }

    /// Greedy candidates from the state after the last committed token: one
    /// noise block proposes every requested position in a single forward.
    /// `out.len` bounds the count (at most `block_size - 1`). `p_min > 0`
    /// stops after a position whose top candidate's softmax probability (the
    /// exact host maximum) is below it; the first position is always proposed.
    pub fn propose(self: *Runtime, token: u32, out: []u32, p_min: f32) !usize {
        if (self.draft == null) return error.NoDraftBlock;
        if (!std.math.isFinite(p_min) or p_min < 0 or p_min > 1) return error.InvalidShape;
        if (out.len == 0) return 0;
        const count_max = @min(out.len, dflash.block_size - 1);
        try self.draftBlock(token, 1 + count_max);
        const d = &self.draft.?;
        var count: usize = 0;
        while (count < count_max) : (count += 1) {
            const row = d.hidden[(1 + count) * dflash.embedding ..][0..dflash.embedding];
            try self.mm(self.binding.output, row, d.logits);
            out[count] = argmax(d.logits);
            if (p_min > 0) {
                var maximum: f64 = -std.math.inf(f64);
                for (d.logits) |v| maximum = @max(maximum, v);
                var total: f64 = 0;
                for (d.logits) |v| total += @exp(@as(f64, v) - maximum);
                if (total > 0 and 1.0 / total < p_min) {
                    count += 1;
                    break;
                }
            }
        }
        return count;
    }

    /// The encoder for one position's pinned residual rows, for `--draft-trace`:
    /// `inp` is `dflash.hidden_width` values, `out` the feature the blocks see.
    pub fn draftEncodeTrace(self: *Runtime, inp: []const f32, out: []f32) !void {
        return self.draftEncode(inp, out);
    }

    /// One pinned noise block for `--draft-trace`: runs `rows` rows from
    /// `token` at the session position, copying every row's final normed
    /// hidden into `hidden` (`rows × dflash.embedding`) and every row's greedy
    /// token into `greedy` (`rows`).
    pub fn draftBlockTrace(self: *Runtime, token: u32, rows: usize, hidden: []f32, greedy: []u32, observer: ?Observer) !void {
        _ = observer;
        if (self.draft == null) return error.NoDraftBlock;
        if (hidden.len != rows * dflash.embedding or greedy.len != rows) return error.InvalidShape;
        try self.draftBlock(token, rows);
        const d = &self.draft.?;
        @memcpy(hidden, d.hidden[0 .. rows * dflash.embedding]);
        for (0..rows) |r| {
            try self.mm(self.binding.output, hidden[r * dflash.embedding ..][0..dflash.embedding], d.logits);
            greedy[r] = argmax(d.logits);
        }
    }

    /// The contract value the engine holds, or null when no companion bound.
    pub fn drafter(self: *Runtime) ?draft_contract.Drafter {
        if (self.draft == null) return null;
        return .{ .host = self, .hidden = dflash.hidden_width, .propose_fn = proposeFn, .commit_fn = commitFn, .reset_fn = resetDraftFn, .bytes_fn = draftBytes };
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
        // The drafter's state is its session cache, which `Session.reset`
        // clears; nothing else is carried between calls.
        _ = host;
    }
    fn draftBytes(host: *anyopaque) usize {
        const self: *Runtime = @ptrCast(@alignCast(host));
        const d = &(self.draft orelse return 0);
        return (d.encoded.len + d.normalized.len + d.hidden.len + d.x_rows.len + d.q_rows.len + d.q.len + d.k.len + d.v.len + d.mixed.len + d.gate.len + d.up.len + d.row.len + d.logits.len) * @sizeOf(f32);
    }
};

/// A binding over empty tensors for the workspace tests: every weight read
/// fails after admission, which is what the invalid-step checks need.
fn emptyBinding(tensor: *const Tensor) model.Binding {
    var binding: model.Binding = undefined;
    binding.draft = null;
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
    var state = try session.Session.init(std.testing.allocator, &layouts, 4, false, 0);
    defer state.deinit();
    const per_position = model.layer_count * 2 * model.kv_width * 4;
    try std.testing.expect(state.bytes() >= per_position * 4);
    try std.testing.expect(state.bytes() < per_position * 4 + model.layer_count * 2 * 16);
}

test "a bound drafter adds five layouts after the language model's" {
    const tensor: Tensor = .{ .name = "empty", .dimensions = &.{0}, .encoding_id = 0, .offset = 0, .elements = 0, .bytes = 0 };
    const file = [_]u8{0};
    var binding = emptyBinding(&tensor);
    // A binding with a companion but no readable weights: the workspace
    // allocates, the layout count is what this checks.
    var companion: dflash.Binding = undefined;
    companion.view = .{ .file = &file, .data_offset = 0 };
    companion.fc = &tensor;
    companion.encoder_norm = &tensor;
    companion.output_norm = &tensor;
    for (&companion.layers) |*layer| layer.* = .{ .attention_norm = &tensor, .query = &tensor, .query_norm = &tensor, .key = &tensor, .key_norm = &tensor, .value = &tensor, .output = &tensor, .ffn_norm = &tensor, .ffn_gate = &tensor, .ffn_up = &tensor, .ffn_down = &tensor };
    binding.draft = companion;
    var runtime = try Runtime.init(std.testing.allocator, .{ .file = &file, .data_offset = 0 }, binding, 4, true, true);
    defer runtime.deinit();
    try std.testing.expectEqual(model.layer_count + dflash.block_count, runtime.state.layers.len);
    try std.testing.expect(runtime.drafter() != null);
    try std.testing.expectEqual(@as(usize, dflash.hidden_width), runtime.drafter().?.hidden);
    // The workspace the memory record cites.
    try std.testing.expectEqual(@as(usize, 2_309_376), runtime.drafter().?.bytes());
    // A drafter request without a bound companion keeps the language model
    // layout and reports no drafter.
    var plain = try Runtime.init(std.testing.allocator, .{ .file = &file, .data_offset = 0 }, emptyBinding(&tensor), 4, true, true);
    defer plain.deinit();
    try std.testing.expectEqual(model.layer_count, plain.state.layers.len);
    try std.testing.expect(plain.drafter() == null);
}

test "targetSlot finds exactly the five residual layers" {
    var found: [dflash.block_count]bool = @splat(false);
    for (0..model.layer_count) |il| {
        if (targetSlot(il)) |slot| {
            try std.testing.expectEqual(dflash.target_layers[slot], il);
            found[slot] = true;
        }
    }
    for (found) |seen| try std.testing.expect(seen);
}
