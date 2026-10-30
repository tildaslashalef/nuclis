//! CPU reference of Drama 3's backbone and codebook tower. `Model` holds the
//! binding and the small F32 constants (immutable, shared); `State` holds a
//! chunk's caches (the backbone's 10 KV caches and 30 DeltaNet states, the
//! tower's 13 KV caches); `Runtime` is the workspace that runs one on the
//! other. Matrices are read from the mapping through `cpu.matvec` (F64
//! sums); the residual streams stay F32. The sampler is the caller's: the
//! tower asks a `Pick` for each code. A failed row poisons its session until
//! `State.reset`. The equations: docs/models/drama3.md.
const std = @import("std");
const alloc_check = @import("../alloc_check.zig");
const model = @import("drama3.zig");
const gguf = @import("../formats/gguf.zig");
const weights = @import("../runtime/weights.zig");
const session = @import("../runtime/session.zig");
const cpu = @import("../backends/cpu/root.zig");
const Tensor = gguf.Tensor;

pub const Observer = @import("../runtime/observer.zig").Observer;

/// One row of the prompt or one generated frame: [main token, code 0, …, code 9].
/// A text row holds its token id and zero codes.
pub const Frame = [1 + model.codebooks]u32;

/// The most rows a chunk holds: the longest prompt plus the most frames.
pub const max_rows = model.max_prompt_rows + model.max_frames;

/// Which key head a DeltaNet value head reads. Fish repeats each key head
/// over consecutive value heads (`grouped`: h / 2); llama.cpp's Qwen3.5
/// converter reorders the value heads so the key heads tile (`tiled`: h % 16).
pub const ValueOrder = enum { grouped, tiled };

/// The two layout conventions the traces settle (docs/models/drama3.md).
pub const Options = struct {
    value_order: ValueOrder = .grouped,
    /// Added to every zero-centred norm weight; the GGUF stores 1 + w, so 0.
    norm_offset: f32 = 0,
};

const AttentionConstants = struct { query_norm: []f32, key_norm: []f32 };
const DeltaConstants = struct { convolution: []f32, a: []f32, time_bias: []f32, norm: []f32 };
const LayerConstants = struct {
    attention_norm: []f32,
    post_attention_norm: []f32,
    mixer: union(enum) { full_attention: AttentionConstants, delta_net: DeltaConstants },
};
const TowerConstants = struct {
    enorm: []f32,
    hnorm: []f32,
    final_norm: []f32,
    attention_norm: []f32,
    ffn_norm: []f32,
    attention: AttentionConstants,
};

/// The bound weights and their decoded F32 constants. Borrows the Document
/// and the mapping `view` reads; both must outlive the model.
pub const Model = struct {
    io: std.Io,
    view: weights.View,
    binding: model.Binding,
    options: Options,
    storage: std.heap.ArenaAllocator,
    output_norm: []f32,
    layers: [model.layer_count]LayerConstants,
    shared: [model.shared_layers]TowerConstants,
    residual: [model.residual_layers]TowerConstants,

    /// Binds `doc`, checks its descriptor arrays against `view`'s file, and
    /// decodes the norms and DeltaNet parameters (about 4 MB).
    pub fn init(gpa: std.mem.Allocator, io: std.Io, doc: *const gguf.Document, view: weights.View, options: Options) !Model {
        const binding = try model.bind(gpa, doc);
        try model.verifyArrays(doc, view.file);
        var storage: std.heap.ArenaAllocator = .init(gpa);
        errdefer storage.deinit();
        const a = storage.allocator();
        var self: Model = .{ .io = io, .view = view, .binding = binding, .options = options, .storage = undefined, .output_norm = undefined, .layers = undefined, .shared = undefined, .residual = undefined };
        self.output_norm = try self.centred(a, binding.output_norm);
        for (&self.layers, binding.layers) |*c, layer| c.* = .{
            .attention_norm = try self.centred(a, layer.attention_norm),
            .post_attention_norm = try self.centred(a, layer.post_attention_norm),
            .mixer = switch (layer.mixer) {
                .full_attention => |attn| .{ .full_attention = try self.attentionConstants(a, attn) },
                .delta_net => |d| blk: {
                    const a_values = try view.vector(a, d.a_log);
                    for (a_values) |*x| x.* = -@exp(x.*);
                    break :blk .{ .delta_net = .{ .convolution = try view.vector(a, d.convolution), .a = a_values, .time_bias = try view.vector(a, d.time_bias), .norm = try view.vector(a, d.norm) } };
                },
            },
        };
        for (&self.shared, binding.shared) |*c, layer| c.* = try self.towerConstants(a, layer);
        for (&self.residual, binding.residual) |*c, layer| c.* = try self.towerConstants(a, layer);
        self.storage = storage;
        return self;
    }

    pub fn deinit(self: *Model) void {
        self.storage.deinit();
        self.* = undefined;
    }

    fn centred(self: *const Model, a: std.mem.Allocator, tensor: *const Tensor) ![]f32 {
        const out = try self.view.vector(a, tensor);
        for (out) |*x| x.* += self.options.norm_offset;
        return out;
    }
    fn attentionConstants(self: *const Model, a: std.mem.Allocator, attn: model.FullAttention) !AttentionConstants {
        return .{ .query_norm = try self.centred(a, attn.query_norm), .key_norm = try self.centred(a, attn.key_norm) };
    }
    fn towerConstants(self: *const Model, a: std.mem.Allocator, layer: model.TowerLayer) !TowerConstants {
        return .{
            .enorm = try self.centred(a, layer.enorm),
            .hnorm = try self.centred(a, layer.hnorm),
            .final_norm = try self.centred(a, layer.final_norm),
            .attention_norm = try self.centred(a, layer.attention_norm),
            .ffn_norm = try self.centred(a, layer.ffn_norm),
            .attention = try self.attentionConstants(a, layer.attention),
        };
    }

    /// The backbone's input row for `frame` (`embed_backbone`): the token's
    /// embedding, or for a semantic row (embedding + codebook 1's row) / √2.
    pub fn embed(self: *const Model, frame: Frame, out: []f32, scratch: []f32) !void {
        try validFrame(frame);
        if (out.len != model.hidden or scratch.len != model.hidden) return error.InvalidShape;
        try self.view.row(self.binding.token_embedding, frame[0], out);
        if (!model.isSemantic(frame[0])) return;
        try self.view.row(self.binding.codebook_embedding, frame[2], scratch);
        const scale: f32 = 1.0 / @sqrt(@as(f32, 2));
        for (out, scratch) |*x, e| x.* = (x.* + e) * scale;
    }
};

/// Refuses an id past the vocabulary and a code past its codebook.
pub fn validFrame(frame: Frame) !void {
    if (frame[0] >= model.vocabulary) return error.InvalidTokenId;
    for (frame[1..]) |code| if (code >= model.codebook_size) return error.InvalidCode;
}

/// One chunk's mutable state, sized for `rows` backbone rows. Fresh per chunk:
/// the recurrent states cannot be rewound, so nothing here rewinds.
pub const State = struct {
    backbone: session.Session,
    tower: session.Session,

    pub fn init(gpa: std.mem.Allocator, rows: usize) !State {
        if (rows == 0 or rows > max_rows) return error.InvalidShape;
        var layouts: [model.layer_count]session.Layout = undefined;
        for (&layouts, 0..) |*layout, i| layout.* = if (model.isFullAttention(i))
            .{ .attention = .{ .key_row = model.kv_width, .value_row = model.kv_width } }
        else
            .{ .recurrent = .{ .history = model.delta_history, .matrix = model.delta_matrix } };
        var backbone = try session.Session.init(gpa, &layouts, rows, false);
        errdefer backbone.deinit();
        const tower_layouts: [model.shared_layers + model.residual_layers]session.Layout = @splat(.{ .attention = .{ .key_row = model.kv_width, .value_row = model.kv_width } });
        return .{ .backbone = backbone, .tower = try session.Session.init(gpa, &tower_layouts, rows, false) };
    }

    pub fn deinit(self: *State) void {
        self.backbone.deinit();
        self.tower.deinit();
        self.* = undefined;
    }

    pub fn reset(self: *State) void {
        self.backbone.reset();
        self.tower.reset();
    }

    pub fn bytes(self: *const State) usize {
        return self.backbone.bytes() + self.tower.bytes();
    }
};

/// Chooses codebook `codebook`'s code (1…9) from its 4,096 logits, borrowed
/// for the call: the sampler, or a trace's code.
pub const Pick = struct {
    context: *anyopaque,
    pick: *const fn (context: *anyopaque, codebook: usize, logits: []const f32) anyerror!u32,
};

/// The workspace. Borrows `model`, which must outlive it; one runtime runs
/// one state at a time.
pub const Runtime = struct {
    model: *const Model,
    storage: std.heap.ArenaAllocator,
    /// The last backbone row's hidden after `output_norm`: the tower's input.
    h: []f32,
    x: []f32,
    normalized: []f32,
    projected: []f32,
    embedding: []f32,
    q: []f32,
    gate: []f32,
    k: []f32,
    v: []f32,
    attended: []f32,
    mixed: []f32,
    convolved: []f32,
    z: []f32,
    alpha: []f32,
    beta: []f32,
    router: []f32,
    indices: []u32,
    route_weights: []f32,
    expert_gate: []f32,
    expert_up: []f32,
    expert_out: []f32,
    shared_scale: []f32,
    accumulator: []f64,
    // The tower's rows.
    tower_h: []f32,
    shared_input: []f32,
    concat: []f32,
    tower_x: []f32,
    tower_gate: []f32,
    tower_up: []f32,
    head_logits: []f32,
    row: []f32,
    attention_scratch: []f64,
    delta_scratch: []f64,

    pub fn init(gpa: std.mem.Allocator, m: *const Model) !Runtime {
        var storage: std.heap.ArenaAllocator = .init(gpa);
        errdefer storage.deinit();
        const a = storage.allocator();
        var self: Runtime = undefined;
        self.model = m;
        inline for (.{ "h", "x", "normalized", "projected", "embedding", "expert_out", "tower_h", "shared_input", "tower_x" }) |field|
            @field(self, field) = try a.alloc(f32, model.hidden);
        inline for (.{ "q", "gate", "attended" }) |field| @field(self, field) = try a.alloc(f32, model.query_width);
        inline for (.{ "k", "v" }) |field| @field(self, field) = try a.alloc(f32, model.kv_width);
        inline for (.{ "mixed", "convolved" }) |field| @field(self, field) = try a.alloc(f32, model.qkv_width);
        self.z = try a.alloc(f32, model.inner_width);
        inline for (.{ "alpha", "beta" }) |field| @field(self, field) = try a.alloc(f32, model.value_heads);
        self.router = try a.alloc(f32, model.experts);
        self.indices = try a.alloc(u32, model.experts_used);
        self.route_weights = try a.alloc(f32, model.experts_used);
        inline for (.{ "expert_gate", "expert_up" }) |field| @field(self, field) = try a.alloc(f32, model.expert_ffn);
        self.shared_scale = try a.alloc(f32, 1);
        self.accumulator = try a.alloc(f64, model.hidden);
        self.concat = try a.alloc(f32, 2 * model.hidden);
        inline for (.{ "tower_gate", "tower_up" }) |field| @field(self, field) = try a.alloc(f32, model.tower_ffn);
        self.head_logits = try a.alloc(f32, model.codebook_size);
        self.row = try a.alloc(f32, cpu.matvecScratch(model.tower_ffn));
        self.attention_scratch = try a.alloc(f64, max_rows);
        self.delta_scratch = try a.alloc(f64, model.state_width * (model.state_width + 1));
        self.storage = storage;
        return self;
    }

    pub fn deinit(self: *Runtime) void {
        self.storage.deinit();
        self.* = undefined;
    }

    fn mm(self: *Runtime, tensor: *const Tensor, input: []const f32, output: []f32) !void {
        try cpu.matvec(self.model.io, try self.model.view.matrix(tensor), input, output, self.row);
    }
    fn norm(input: []const f32, output: []f32, weight: []const f32) !void {
        if (weight.len != output.len) return error.InvalidShape;
        try cpu.rmsNorm(input, output, model.norm_eps);
        for (output, weight) |*x, w| x.* *= w;
    }

    // --- the backbone ---------------------------------------------------

    /// Runs `frames` at the next backbone rows. `hidden_rows`, when given
    /// (`frames.len × hidden`), receives every row's post-norm hidden (the
    /// tower prefill's input); `logits` (4,097) the last row's. The last
    /// row's hidden stays in `h`.
    pub fn prefill(self: *Runtime, state: *State, frames: []const Frame, hidden_rows: ?[]f32, logits: ?[]f32, observer: ?Observer) !void {
        if (frames.len == 0) return error.InvalidShape;
        if (hidden_rows) |rows| if (rows.len != frames.len * model.hidden) return error.InvalidShape;
        if (logits) |out| if (out.len != model.output_rows) return error.InvalidShape;
        if (frames.len > state.backbone.capacity - state.backbone.position) return error.ContextFull;
        for (frames) |frame| try validFrame(frame);
        for (frames, 0..) |frame, i| {
            try self.backboneRow(&state.backbone, frame, observer);
            if (hidden_rows) |rows| @memcpy(rows[i * model.hidden ..][0..model.hidden], self.h);
        }
        if (logits) |out| try self.mainLogits(self.h, out);
    }

    /// One generated frame at the next backbone row; `prefill` of one.
    pub fn decode(self: *Runtime, state: *State, frame: Frame, logits: ?[]f32, observer: ?Observer) !void {
        return self.prefill(state, &.{frame}, null, logits, observer);
    }

    /// The 4,097 main-token logits of a post-norm hidden `h` (rows map to ids
    /// by `drama3.outputToken`).
    pub fn mainLogits(self: *Runtime, h: []const f32, out: []f32) !void {
        if (h.len != model.hidden or out.len != model.output_rows) return error.InvalidShape;
        try self.mm(self.model.binding.output, h, out);
        for (out) |x| if (!std.math.isFinite(x)) return error.NonFiniteResult;
    }

    fn backboneRow(self: *Runtime, s: *session.Session, frame: Frame, observer: ?Observer) !void {
        const m = self.model;
        try s.begin();
        errdefer s.fail();
        try m.embed(frame, self.x, self.embedding);
        for (m.binding.layers, m.layers, 0..) |layer, constants, il| {
            if (observer) |o| if (o.check) |check| try check(o.context);
            try norm(self.x, self.normalized, constants.attention_norm);
            switch (layer.mixer) {
                .full_attention => |attn| {
                    const cache = s.layers[il].attention;
                    try self.attention(attn, constants.mixer.full_attention, cache.keys, cache.values, s.position, .split_half);
                },
                .delta_net => |d| try self.deltaNet(d, constants.mixer.delta_net, s.layers[il].recurrent),
            }
            for (self.x, self.projected) |*x, y| x.* += y;
            try norm(self.x, self.normalized, constants.post_attention_norm);
            try self.moe(layer.moe);
            for (self.x, self.projected) |*x, y| {
                x.* += y;
                if (!std.math.isFinite(x.*)) return error.NonFiniteResult;
            }
            if (observer) |o| if (o.layer) |report| try report(o.context, il, self.x);
        }
        try norm(self.x, self.h, m.output_norm);
        try s.commit();
    }

    /// Gated full attention over `self.normalized` at `position`, into
    /// `self.projected`: the backbone's (NeoX RoPE) and the tower's
    /// (interleaved RoPE) differ only in `pairing`.
    fn attention(self: *Runtime, attn: model.FullAttention, constants: AttentionConstants, keys: session.Rows, values: session.Rows, position: usize, pairing: cpu.rope.Pairing) !void {
        try self.mm(attn.query, self.normalized, self.q);
        try self.mm(attn.gate, self.normalized, self.gate);
        try self.mm(attn.key, self.normalized, self.k);
        try self.mm(attn.value, self.normalized, self.v);
        const rope: cpu.rope.Options = .{ .dimensions = model.rotary_dims, .base = model.rope_base, .position = @intCast(position), .pairing = pairing };
        for (0..model.heads) |h| {
            const head = self.q[h * model.head_width ..][0..model.head_width];
            try norm(head, head, constants.query_norm);
            try cpu.rope.apply(head, head, rope);
        }
        for (0..model.kv_heads) |h| {
            const head = self.k[h * model.head_width ..][0..model.head_width];
            try norm(head, head, constants.key_norm);
            try cpu.rope.apply(head, head, rope);
        }
        @memcpy(keys.floats(position, 1), self.k);
        @memcpy(values.floats(position, 1), self.v);
        try cpu.attention.apply(.{
            .query_heads = model.heads,
            .kv_heads = model.kv_heads,
            .key_width = model.head_width,
            .value_width = model.head_width,
            .tokens = position + 1,
            .visible_tokens = position + 1,
            .scale = 1.0 / @sqrt(@as(f32, model.head_width)),
            .queries = self.q,
            .keys = keys.floats(0, position + 1),
            .values = values.floats(0, position + 1),
        }, self.attended, self.attention_scratch[0 .. position + 1]);
        for (self.attended, self.gate) |*x, g| x.* *= cpu.sigmoid(g);
        try self.mm(attn.output, self.attended, self.projected);
    }

    /// The gated delta rule over `self.normalized`, into `self.projected`.
    fn deltaNet(self: *Runtime, d: model.DeltaNet, constants: DeltaConstants, state: anytype) !void {
        try self.mm(d.qkv, self.normalized, self.mixed);
        try self.mm(d.z, self.normalized, self.z);
        try self.mm(d.beta, self.normalized, self.beta);
        try self.mm(d.alpha, self.normalized, self.alpha);
        try cpu.recurrent.convolution(self.mixed, constants.convolution, state.history, self.convolved, model.conv_kernel);
        for (self.convolved) |*x| x.* = cpu.silu(x.*);
        // [Q: 16 × 128 | K: 16 × 128 | V: 32 × 128]
        for (0..2 * model.key_heads) |h| {
            const head = self.convolved[h * model.state_width ..][0..model.state_width];
            try cpu.l2Norm(head, head, model.norm_eps);
        }
        for (0..model.value_heads) |h| {
            self.alpha[h] = constants.a[h] * cpu.softplus(self.alpha[h] + constants.time_bias[h]);
            self.beta[h] = cpu.sigmoid(self.beta[h]);
        }
        const out = self.attended[0..model.inner_width];
        for (0..model.value_heads) |h| {
            const kh = switch (self.model.options.value_order) {
                .grouped => h / (model.value_heads / model.key_heads),
                .tiled => h % model.key_heads,
            };
            const matrix = state.matrix[h * model.state_width * model.state_width ..][0 .. model.state_width * model.state_width];
            const head = out[h * model.state_width ..][0..model.state_width];
            try cpu.recurrent.delta(.{
                .query = self.convolved[kh * model.state_width ..][0..model.state_width],
                .key = self.convolved[model.key_width + kh * model.state_width ..][0..model.state_width],
                .value = self.convolved[2 * model.key_width + h * model.state_width ..][0..model.state_width],
                .log_decay = self.alpha[h],
                .beta = self.beta[h],
                .scale = 1.0 / @sqrt(@as(f32, model.state_width)),
            }, matrix, matrix, head, self.delta_scratch);
            // GatedRMSNorm: weight ⊙ rmsnorm(o) ⊙ silu(z), per value head.
            try norm(head, head, constants.norm);
            for (head, self.z[h * model.state_width ..][0..model.state_width]) |*x, z| x.* *= cpu.silu(z);
        }
        try self.mm(d.output, out, self.projected);
    }

    /// The routed experts and the gated shared expert over
    /// `self.normalized`, into `self.projected`.
    fn moe(self: *Runtime, e: model.Moe) !void {
        const view = self.model.view;
        try self.mm(e.router, self.normalized, self.router);
        try cpu.experts.route(self.router, self.indices, self.route_weights);
        const gate = try view.expertMatrix(e.gate);
        const up = try view.expertMatrix(e.up);
        const down = try view.expertMatrix(e.down);
        @memset(self.accumulator, 0);
        for (self.indices, self.route_weights) |expert, w| {
            try cpu.matvec(self.model.io, try gate.expert(expert), self.normalized, self.expert_gate, self.row);
            try cpu.matvec(self.model.io, try up.expert(expert), self.normalized, self.expert_up, self.row);
            for (self.expert_gate, self.expert_up) |*g, u| g.* = cpu.silu(g.*) * u;
            try cpu.matvec(self.model.io, try down.expert(expert), self.expert_gate, self.expert_out, self.row);
            for (self.accumulator, self.expert_out) |*acc, y| acc.* += @as(f64, w) * y;
        }
        try self.mm(e.shared_gate, self.normalized, self.expert_gate);
        try self.mm(e.shared_up, self.normalized, self.expert_up);
        for (self.expert_gate, self.expert_up) |*g, u| g.* = cpu.silu(g.*) * u;
        try self.mm(e.shared_down, self.expert_gate, self.expert_out);
        try self.mm(e.shared_router, self.normalized, self.shared_scale);
        const scale: f64 = cpu.sigmoid(self.shared_scale[0]);
        for (self.projected, self.accumulator, self.expert_out) |*y, acc, s| y.* = @floatCast(acc + scale * s);
    }

    // --- the tower ------------------------------------------------------

    /// Teacher-forces the tower over the next `hidden_rows.len / hidden`
    /// rows (`prefill`): row i's hidden is labelled with `frames[i + 1]`'s
    /// codes and enriched by `frames[i]`'s, so `frames` holds one more frame
    /// than there are rows. A prompt's last row is left to `step`.
    pub fn towerPrefill(self: *Runtime, state: *State, hidden_rows: []const f32, frames: []const Frame, observer: ?Observer) !void {
        const rows = hidden_rows.len / model.hidden;
        if (rows == 0 or hidden_rows.len != rows * model.hidden or frames.len != rows + 1) return error.InvalidShape;
        if (state.tower.position + rows > state.backbone.position) return error.InvalidShape;
        for (frames) |frame| try validFrame(frame);
        for (0..rows) |i| {
            const next = frames[i + 1];
            @memcpy(self.tower_h, hidden_rows[i * model.hidden ..][0..model.hidden]);
            _ = try self.towerRow(&state.tower, frames[i], next, null, observer);
        }
    }

    /// The tower at the backbone's last row, whose hidden is `h` and input
    /// frame `input`: the main token `main` (sampled by the caller from the
    /// row's logits) gives code 0, then `pick` chooses codes 1–9. Returns
    /// the new frame, the input of the next backbone row.
    pub fn step(self: *Runtime, state: *State, h: []const f32, input: Frame, main: u32, pick: Pick, observer: ?Observer) !Frame {
        if (h.len != model.hidden) return error.InvalidShape;
        if (state.tower.position + 1 != state.backbone.position) return error.InvalidShape;
        try validFrame(input);
        if (main >= model.vocabulary) return error.InvalidTokenId;
        var frame: Frame = @splat(0);
        frame[0] = main;
        // Fish clamps code 0 into the codebook; a non-semantic main embeds nothing.
        if (model.isSemantic(main)) frame[1] = main - model.semantic_start;
        @memcpy(self.tower_h, h);
        return self.towerRow(&state.tower, input, frame, pick, observer);
    }

    /// One tower row at the session's position from `self.tower_h`: the
    /// shared stack on `label`'s code 0 (layer 0 also sees `input`'s codes
    /// 2–9), then residual layer j on code j. With `pick`, code j + 1 is
    /// chosen from head j's logits; without, `label`'s codes are forced.
    /// Returns `label` with the codes chosen.
    fn towerRow(self: *Runtime, s: *session.Session, input: Frame, label: Frame, pick: ?Pick, observer: ?Observer) !Frame {
        const m = self.model;
        try s.begin();
        errdefer s.fail();
        const semantic = model.isSemantic(label[0]);
        @memset(self.shared_input, 0);
        if (model.isSemantic(input[0])) for (2..model.codebooks) |k| {
            try m.view.row(m.binding.codebook_embedding, (k - 1) * model.codebook_size + input[1 + k], self.embedding);
            for (self.shared_input, self.embedding) |*x, e| x.* += e;
        };
        var codes = label;
        for (m.binding.shared, m.shared, 0..) |layer, constants, i| {
            try self.towerLayer(s, i, layer, constants, codes[1], semantic, if (i == 0) self.shared_input else null);
            if (observer) |o| if (o.layer) |report| try report(o.context, i, self.tower_h);
        }
        for (m.binding.residual, m.residual, m.binding.heads, 0..) |layer, constants, head, j| {
            const il = model.shared_layers + j;
            try self.towerLayer(s, il, layer, constants, codes[1 + j], semantic, null);
            if (observer) |o| if (o.layer) |report| try report(o.context, il, self.tower_h);
            if (pick) |p| {
                try self.mm(head, self.tower_h, self.head_logits);
                for (self.head_logits) |x| if (!std.math.isFinite(x)) return error.NonFiniteResult;
                const code = try p.pick(p.context, j + 1, self.head_logits);
                if (code >= model.codebook_size) return error.InvalidCode;
                codes[2 + j] = code;
            }
        }
        try s.commit();
        return codes;
    }

    /// `TowerLayer.forward` on `self.tower_h` in place, at cache layer `il`.
    fn towerLayer(self: *Runtime, s: *session.Session, il: usize, layer: model.TowerLayer, constants: TowerConstants, code: u32, semantic: bool, shared_input: ?[]const f32) !void {
        const m = self.model;
        if (semantic) try m.view.row(layer.embedding, @min(code, model.codebook_size - 1), self.embedding) else @memset(self.embedding, 0);
        if (shared_input) |extra| for (self.embedding, extra) |*e, x| {
            e.* += x;
        };
        try norm(self.embedding, self.concat[0..model.hidden], constants.enorm);
        try norm(self.tower_h, self.concat[model.hidden..], constants.hnorm);
        try self.mm(layer.eh_proj, self.concat, self.tower_x);
        try norm(self.tower_x, self.normalized, constants.attention_norm);
        const cache = s.layers[il].attention;
        try self.attention(layer.attention, constants.attention, cache.keys, cache.values, s.position, .adjacent);
        for (self.tower_x, self.projected) |*x, y| x.* += y;
        try norm(self.tower_x, self.normalized, constants.ffn_norm);
        try self.mm(layer.ffn_gate, self.normalized, self.tower_gate);
        try self.mm(layer.ffn_up, self.normalized, self.tower_up);
        for (self.tower_gate, self.tower_up) |*g, u| g.* = cpu.silu(g.*) * u;
        try self.mm(layer.ffn_down, self.tower_gate, self.projected);
        for (self.tower_x, self.projected) |*x, y| {
            x.* += y;
            if (!std.math.isFinite(x.*)) return error.NonFiniteResult;
        }
        try norm(self.tower_x, self.tower_h, constants.final_norm);
    }
};

test "frames are validated: ids within the vocabulary, codes within a codebook" {
    var frame: Frame = @splat(0);
    frame[0] = model.semantic_start + 7;
    frame[1] = 7;
    try validFrame(frame);
    frame[5] = model.codebook_size;
    try std.testing.expectError(error.InvalidCode, validFrame(frame));
    frame[5] = 0;
    frame[0] = model.vocabulary;
    try std.testing.expectError(error.InvalidTokenId, validFrame(frame));
}

test "state sizing and cleanup on every allocation failure" {
    try alloc_check.checkAll(std.testing.allocator, struct {
        fn check(alloc: std.mem.Allocator) !void {
            var state = try State.init(alloc, 4);
            defer state.deinit();
            try std.testing.expectEqual(@as(usize, model.layer_count), state.backbone.layers.len);
            try std.testing.expectEqual(@as(usize, 13), state.tower.layers.len);
            try std.testing.expect(state.backbone.hasRecurrent() and !state.tower.hasRecurrent());
        }
    }.check, .{});
    try std.testing.expectError(error.InvalidShape, State.init(std.testing.allocator, max_rows + 1));
    try std.testing.expectError(error.InvalidShape, State.init(std.testing.allocator, 0));
}
