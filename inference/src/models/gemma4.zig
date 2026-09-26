//! Gemma 4 text adapter: metadata validation and named weight binding for
//! three pinned checkpoints of one architecture, the dense 12B, the 26B-A4B
//! mixture of experts, and the E4B. The facts this file encodes were read
//! from the artifacts and the reference and are recorded, with their
//! provenance, in docs/reference/gemma4.md; nothing here is generic Gemma
//! knowledge. Like the Qwen adapter it is deliberately narrow: a file whose
//! `gemma4.*` keys differ from a pinned configuration is rejected with
//! `UnsupportedConfiguration` rather than run with guessed semantics.
//!
//! Shared by all: decoder layers in a period-6 pattern, five sliding-window
//! layers (head size 256) then one global layer (full causal attention, head
//! size 512), four RMS norms per layer (pre/post attention, pre/post FFN),
//! per-head query and key norms, a tanh-GELU gated FFN, a scalar output
//! scale per layer, tied embeddings, and final logit soft-capping at 30.
//! Per `Config`: widths, query and KV heads, the window, whether global
//! layers have their own value projection, the expert block (26B-A4B), and
//! the E4B's per-layer embeddings and shared KV layers (`kvSource`). The
//! runtimes (`gemma4_runtime.zig`, `gemma4_metal.zig`) execute exactly this
//! binding.
const std = @import("std");
const gguf = @import("../formats/gguf.zig");
const models = @import("root.zig");
const weights = @import("../runtime/weights.zig");
const gemma4_assistant = @import("gemma4_assistant.zig");
const Tensor = gguf.Tensor;

pub const architecture = "gemma4";

/// The assistant draft heads' proposal bound: a head forward chains one
/// draft per position and the Metal plan's verify tile holds 8 rows (the
/// seed plus 7). Declared by the adapter so both executors cap a requested
/// draft length the same way.
pub const max_draft_proposals = 7;

/// The most query heads of any pinned configuration; buffers size by it.
pub const max_heads = 16;
pub const vocabulary = 262144;
/// One image span in a prompt (the shared vision contract's `Span`), used by
/// the CPU runtime and the Metal plan.
pub const VisionSpan = @import("../vision/root.zig").Span;
pub const rms_epsilon: f32 = 1e-6;
pub const final_softcap: f32 = 30.0;
/// The largest layer count of any pinned configuration; bindings carry
/// this many layer slots and `Binding.active` narrows them.
pub const max_layers = 48;
/// The widest key/value row of any layer: the 12B's sliding geometry (8
/// heads of 256); a global layer has one or two heads of 512.
pub const max_kv_width = 8 * 256;

/// The expert block of a configuration whose layers have one.
pub const Experts = struct {
    /// Experts per layer and how many each token uses.
    count: usize,
    used: usize,
    /// Hidden width of one expert's FFN.
    feed_forward: usize,
};

/// One pinned checkpoint's dimensions; everything a runtime sizes from.
pub const Config = struct {
    /// The adapter's name for the configuration (`Summary.profile`).
    profile: []const u8,
    layer_count: usize,
    embedding: usize,
    /// The dense FFN's hidden width (the shared expert on the 26B-A4B).
    feed_forward: usize,
    /// Query heads of every layer.
    heads: usize,
    /// Positions a sliding layer sees, its own included.
    window: usize,
    context_length: usize,
    sliding_kv_heads: usize,
    global_kv_heads: usize,
    /// Whether global layers have their own value projection; without one,
    /// V is the key projection taken before the key norm and RoPE.
    global_value: bool,
    experts: ?Experts,
    /// Width of each layer's slice of the per-layer embedding; 0 when the
    /// checkpoint has none.
    per_layer_input: usize = 0,
    /// Layers `[0, kv_layers)` own a cache; the rest read one (`kvSource`).
    kv_layers: usize,
    /// Whether an image span's rows see each other on sliding layers (the
    /// reference's non-causal image decode); the E4B's span stays causal.
    bidirectional_images: bool = true,
    layer_kinds: []const models.LayerKind,

    /// The embedding row is multiplied by sqrt(width) after decoding.
    pub fn embeddingScale(self: *const Config) f32 {
        return @sqrt(@as(f32, @floatFromInt(self.embedding)));
    }

    /// The cache layer `il` attends over: its own, or on a shared layer the
    /// last cache-owning layer of its kind (the reference's reuse rule:
    /// `kv_layers − 2` for sliding, `kv_layers − 1` for global).
    pub fn kvSource(self: *const Config, il: usize) usize {
        if (il < self.kv_layers) return il;
        return self.kv_layers - @as(usize, if (kindOf(il) == .sliding) 2 else 1);
    }

    /// The first cache row visible to the token at `position` on a layer
    /// of `kind`: sliding layers read a contiguous suffix of the cache.
    pub fn firstVisible(self: *const Config, kind: Kind, position: usize) usize {
        return if (kind == .sliding and position + 1 > self.window) position + 1 - self.window else 0;
    }
};

pub const config_12b: Config = .{
    .profile = "gemma4_12b",
    .layer_count = 48,
    .embedding = 3840,
    .feed_forward = 15360,
    .heads = 16,
    .window = 1024,
    .context_length = 262144,
    .sliding_kv_heads = 8,
    .global_kv_heads = 1,
    .global_value = false,
    .experts = null,
    .kv_layers = 48,
    .layer_kinds = &.{ .{ .kind = "sliding_attention", .count = 40 }, .{ .kind = "global_attention", .count = 8 } },
};
pub const config_26b_a4b: Config = .{
    .profile = "gemma4_26b_a4b",
    .layer_count = 30,
    .embedding = 2816,
    .feed_forward = 2112,
    .heads = 16,
    .window = 1024,
    .context_length = 262144,
    .sliding_kv_heads = 8,
    .global_kv_heads = 2,
    .global_value = false,
    .experts = .{ .count = 128, .used = 8, .feed_forward = 704 },
    .kv_layers = 30,
    .layer_kinds = &.{ .{ .kind = "sliding_attention", .count = 25 }, .{ .kind = "global_attention", .count = 5 } },
};
/// The on-device E4B: 8 query heads, a 512-position window, a value
/// projection on global layers, 256-wide per-layer embeddings, and the
/// last 18 layers reading the caches of layers 22 (sliding) and 23 (global).
pub const config_e4b: Config = .{
    .profile = "gemma4_e4b",
    .layer_count = 42,
    .embedding = 2560,
    .feed_forward = 10240,
    .heads = 8,
    .window = 512,
    .context_length = 131072,
    .sliding_kv_heads = 2,
    .global_kv_heads = 2,
    .global_value = true,
    .experts = null,
    .per_layer_input = 256,
    .kv_layers = 24,
    .bidirectional_images = false,
    .layer_kinds = &.{ .{ .kind = "sliding_attention", .count = 35 }, .{ .kind = "global_attention", .count = 7 } },
};
/// The pinned configurations, selected by `gemma4.block_count`.
pub const configs = [_]*const Config{ &config_12b, &config_26b_a4b, &config_e4b };

/// The two layer kinds and the attention geometry they share across
/// configurations; the KV head count is the layer's (`Layer.kv_heads`).
pub const Kind = enum {
    sliding,
    global,

    pub fn headSize(self: Kind) usize {
        return switch (self) {
            .sliding => 256,
            .global => 512,
        };
    }
    pub fn ropeBase(self: Kind) f32 {
        return switch (self) {
            .sliding => 10_000,
            .global => 1_000_000,
        };
    }
    /// The widest query projection of this kind (`max_heads` heads); a
    /// layer's own is `Layer.queryWidth`.
    pub fn queryWidth(self: Kind) usize {
        return max_heads * self.headSize();
    }
};

/// Layer `index` is global when `index % 6 == 5` (layers 5, 11, …), the
/// pattern both per-layer metadata arrays declare; `validateMetadata`
/// checks the arrays against this function, so the runtimes can use it.
pub fn kindOf(index: usize) Kind {
    return if (index % 6 == 5) .global else .sliding;
}
/// KV heads of a layer kind under a configuration.
pub fn kvHeadsOf(config: *const Config, kind: Kind) usize {
    return switch (kind) {
        .sliding => config.sliding_kv_heads,
        .global => config.global_kv_heads,
    };
}

/// Whether a weight matrix of this encoding can be bound: the storage
/// layouts the shared CPU reference and Metal kernels execute. Q4_0 (id 2),
/// the QAT checkpoints' only weight encoding, completes the set.
pub fn executableEncoding(encoding_id: u32) bool {
    return switch (encoding_id) {
        0, 2, 8, 11, 12, 13, 14, 20, 21, 23 => true,
        else => false,
    };
}

pub const Summary = models.Summary;

/// The expert block of one layer (26B-A4B). The 3-D tensors are
/// `[experts][rows][columns]`, experts contiguous (`cpu.ExpertMatrix`).
pub const ExpertLayer = struct {
    /// Router logits: an F32 `[count][embedding]` matrix over the RMS-normed
    /// attention output scaled by 1/sqrt(embedding) and `router_scale`.
    router: *const Tensor,
    router_scale: *const Tensor,
    /// Fused gate rows then up rows per expert, `[count][2·ff][embedding]`.
    gate_up: *const Tensor,
    /// `[count][embedding][ff]`, with one F32 scale per expert on its output.
    down: *const Tensor,
    down_scale: *const Tensor,
    /// The shared FFN's post norm and the expert branch's pre and post norms.
    post_ffn_norm_1: *const Tensor,
    pre_ffn_norm_2: *const Tensor,
    post_ffn_norm_2: *const Tensor,
};

/// A layer's per-layer-embedding block (E4B): `inp_gate` `[embedding → w]`,
/// `proj` `[w → embedding]`, and the norm on its output.
pub const PerLayerBlock = struct {
    gate: *const Tensor,
    projection: *const Tensor,
    post_norm: *const Tensor,
};

pub const Layer = struct {
    kind: Kind,
    heads: usize,
    kv_heads: usize,
    attention_norm: *const Tensor,
    post_attention_norm: *const Tensor,
    ffn_norm: *const Tensor,
    post_ffn_norm: *const Tensor,
    query: *const Tensor,
    /// Absent on a layer that reads another layer's cache (`Config.kvSource`).
    key: ?*const Tensor,
    /// Absent on shared layers and, without `Config.global_value`, on global
    /// layers, whose values are then the key projection.
    value: ?*const Tensor,
    output: *const Tensor,
    query_norm: *const Tensor,
    key_norm: ?*const Tensor,
    ffn_gate: *const Tensor,
    ffn_up: *const Tensor,
    ffn_down: *const Tensor,
    /// One F32 scalar multiplying the layer's residual output.
    output_scale: *const Tensor,
    /// Present on every layer of an expert configuration.
    experts: ?ExpertLayer,
    /// Present on every layer of a configuration with per-layer embeddings.
    per_layer: ?PerLayerBlock,

    pub fn headSize(self: Layer) usize {
        return self.kind.headSize();
    }
    pub fn queryWidth(self: Layer) usize {
        return self.heads * self.kind.headSize();
    }
    /// Key (and value) projection width: `kv_heads` of `headSize`.
    pub fn kvWidth(self: Layer) usize {
        return self.kv_heads * self.kind.headSize();
    }
};

/// All tensor pointers borrow doc.tensors. No weights are read or allocated;
/// keep the source Document alive for the entire binding lifetime.
pub const Binding = struct {
    config: *const Config,
    /// The draft head when a companion was loaded; null for the main file.
    draft: ?gemma4_assistant.Binding = null,
    /// Also the output projection (tied): logits = token_embedding · y.
    token_embedding: *const Tensor,
    output_norm: *const Tensor,
    /// 256 per-pair RoPE frequency factors for the global layers.
    rope_factors: *const Tensor,
    /// The per-layer embedding inputs (E4B): a `[layers · w]` row per token,
    /// the projection of the scaled embedding onto it, and the norm of each
    /// `w`-wide slice of that projection.
    per_layer: ?struct {
        token_embedding: *const Tensor,
        model_projection: *const Tensor,
        projection_norm: *const Tensor,
    },
    /// `config.layer_count` leading entries are bound; see `active`.
    layers: [max_layers]Layer,
    summary: Summary,

    /// The bound layers, in order.
    pub fn active(self: *const Binding) []const Layer {
        return self.layers[0..self.config.layer_count];
    }
};

pub const Error = models.BindError;
/// `bindDraft` adds the companion's mismatch, the typed error `Engine.open`
/// surfaces.
pub const DraftError = models.BindError || error{ DraftSourceMismatch, InvalidShape, TensorOutOfBounds };

/// The registry entry (`models.table`).
pub const family = struct {
    pub const architecture = @import("gemma4.zig").architecture;
    pub const executableEncoding = @import("gemma4.zig").executableEncoding;
    pub const Binding = @import("gemma4.zig").Binding;
    pub const bind = @import("gemma4.zig").bind;
    pub const Runtime = @import("gemma4_runtime.zig").Runtime;
    pub const Plan = @import("gemma4_metal.zig").Plan;
    /// The draft source is the `gemma4-assistant` companion file.
    pub const draft_architecture = gemma4_assistant.architecture;
    pub const bindDraft = @import("gemma4.zig").bindDraft;
};

/// The companion head for a bound target: the head's architecture and
/// target-width checks already ran in `Engine.open`; this binds the file and
/// refuses a head whose width is not the target's.
/// The companion head for a bound target: `view` is the companion file's byte
/// source, `target_view` the target's. A head of another width is refused, and
/// so is one whose global-layer RoPE factors differ from the target's — the
/// Metal plan reuses the target's rope tables.
pub fn bindDraft(alloc: std.mem.Allocator, doc: *const gguf.Document, view: weights.View, target_view: weights.View, target: *const Binding) DraftError!gemma4_assistant.Binding {
    var head = gemma4_assistant.bind(alloc, doc) catch |err| switch (err) {
        error.UnsupportedArchitecture, error.UnsupportedConfiguration, error.MissingMetadata, error.InvalidMetadata => return error.DraftSourceMismatch,
        else => return err,
    };
    if (head.config.embedding_out != target.config.embedding) return error.DraftSourceMismatch;
    // The head attends over the target's cache rows with its own KV heads and mask.
    if (head.config.sliding_kv_heads != target.config.sliding_kv_heads or
        head.config.global_kv_heads != target.config.global_kv_heads or
        head.config.window != target.config.window) return error.DraftSourceMismatch;
    const head_factors = view.vector(alloc, head.rope_factors) catch return error.DraftSourceMismatch;
    defer alloc.free(head_factors);
    const target_factors = target_view.vector(alloc, target.rope_factors) catch return error.DraftSourceMismatch;
    defer alloc.free(target_factors);
    if (!std.mem.eql(f32, head_factors, target_factors)) return error.DraftSourceMismatch;
    head.view = view;
    return head;
}

const IntegerSetting = struct { key: []const u8, value: u64 };
/// Keys with one value across the pinned configurations.
const shared_integer_settings = [_]IntegerSetting{
    .{ .key = "gemma4.attention.key_length", .value = 512 },
    .{ .key = "gemma4.attention.value_length", .value = 512 },
    .{ .key = "gemma4.attention.key_length_swa", .value = 256 },
    .{ .key = "gemma4.attention.value_length_swa", .value = 256 },
    .{ .key = "gemma4.rope.dimension_count", .value = 512 },
    .{ .key = "gemma4.rope.dimension_count_swa", .value = 256 },
};
/// Keys whose value is the configuration's (checked in `validateMetadata`).
const per_config_keys = [_][]const u8{
    "gemma4.context_length",
    "gemma4.attention.head_count",
    "gemma4.attention.sliding_window",
    "gemma4.attention.shared_kv_layers",
    "gemma4.embedding_length_per_layer_input",
};
const config_integer_keys = [_][]const u8{ "gemma4.block_count", "gemma4.embedding_length", "gemma4.feed_forward_length" };
const expert_integer_keys = [_][]const u8{ "gemma4.expert_count", "gemma4.expert_used_count", "gemma4.expert_feed_forward_length" };
const FloatSetting = struct { key: []const u8, value: f64 };
const float_settings = [_]FloatSetting{
    .{ .key = "gemma4.rope.freq_base", .value = 1_000_000 },
    .{ .key = "gemma4.rope.freq_base_swa", .value = 10_000 },
    .{ .key = "gemma4.attention.layer_norm_rms_epsilon", .value = @as(f32, 1e-6) },
    .{ .key = "gemma4.final_logit_softcapping", .value = 30 },
};
const array_settings = [_][]const u8{ "gemma4.attention.head_count_kv", "gemma4.attention.sliding_window_pattern" };

/// Integer metadata regardless of the writer's chosen width or signedness.
fn integer(doc: *const gguf.Document, key: []const u8) Error!u64 {
    return switch (doc.get(key) orelse return error.MissingMetadata) {
        .unsigned => |n| n,
        .signed => |n| if (n < 0) error.InvalidMetadata else @intCast(n),
        else => error.InvalidMetadata,
    };
}
fn expectInteger(doc: *const gguf.Document, key: []const u8, value: u64) Error!void {
    if (try integer(doc, key) != value) return error.UnsupportedConfiguration;
}

/// The pinned configuration a directory declares, by its layer count.
fn configFor(doc: *const gguf.Document) Error!*const Config {
    const blocks = try integer(doc, "gemma4.block_count");
    for (configs) |config| if (config.layer_count == blocks) return config;
    return error.UnsupportedConfiguration;
}

fn validateMetadata(doc: *const gguf.Document) Error!*const Config {
    const arch = doc.string("general.architecture") orelse return error.MissingMetadata;
    if (!std.mem.eql(u8, arch, architecture)) return error.UnsupportedArchitecture;
    const config = try configFor(doc);
    for (shared_integer_settings) |setting| try expectInteger(doc, setting.key, setting.value);
    try expectInteger(doc, per_config_keys[0], config.context_length);
    try expectInteger(doc, per_config_keys[1], config.heads);
    try expectInteger(doc, per_config_keys[2], config.window);
    try expectInteger(doc, per_config_keys[3], config.layer_count - config.kv_layers);
    try expectInteger(doc, per_config_keys[4], config.per_layer_input);
    try expectInteger(doc, config_integer_keys[1], config.embedding);
    try expectInteger(doc, config_integer_keys[2], config.feed_forward);
    if (config.experts) |experts| {
        try expectInteger(doc, expert_integer_keys[0], experts.count);
        try expectInteger(doc, expert_integer_keys[1], experts.used);
        try expectInteger(doc, expert_integer_keys[2], experts.feed_forward);
    }
    for (float_settings) |setting| {
        switch (doc.get(setting.key) orelse return error.MissingMetadata) {
            .float => |value| if (value != setting.value) return error.UnsupportedConfiguration,
            else => return error.InvalidMetadata,
        }
    }
    // The per-layer arrays must declare the period-6 pattern `kindOf`
    // hard-codes, with the configuration's KV heads. A writer may store the
    // KV heads as one scalar when every layer has the same count.
    const swa = try retained(doc, array_settings[1], config.layer_count);
    const kv: ?[]const gguf.Value = switch (doc.get(array_settings[0]) orelse return error.MissingMetadata) {
        .array => try retained(doc, array_settings[0], config.layer_count),
        else => null,
    };
    const uniform: ?u64 = if (kv == null) try integer(doc, array_settings[0]) else null;
    for (swa, 0..) |sliding, index| {
        const kind = kindOf(index);
        const n: u64 = if (kv) |values| switch (values[index]) {
            .signed => |v| if (v < 0) return error.InvalidMetadata else @intCast(v),
            .unsigned => |v| v,
            else => return error.InvalidMetadata,
        } else uniform.?;
        if (n != kvHeadsOf(config, kind)) return error.UnsupportedConfiguration;
        const is_sliding = switch (sliding) {
            .boolean => |b| b,
            else => return error.InvalidMetadata,
        };
        if (is_sliding != (kind == .sliding)) return error.UnsupportedConfiguration;
    }
    const tokens = switch (doc.get("tokenizer.ggml.tokens") orelse return error.MissingMetadata) {
        .array => |a| a,
        else => return error.InvalidMetadata,
    };
    if (tokens.element_type != .string or tokens.count != vocabulary) return error.UnsupportedConfiguration;
    // Any other gemma4.* key could change the equations while every checked
    // dimension still matches; refuse rather than guess.
    for (doc.metadata) |entry| {
        if (std.mem.startsWith(u8, entry.key, "gemma4.") and !knownArchitectureKey(config, entry.key))
            return error.UnsupportedConfiguration;
    }
    return config;
}

fn retained(doc: *const gguf.Document, key: []const u8, count: usize) Error![]const gguf.Value {
    const array = switch (doc.get(key) orelse return error.MissingMetadata) {
        .array => |a| a,
        else => return error.InvalidMetadata,
    };
    if (array.count != count) return error.UnsupportedConfiguration;
    const values = array.values orelse return error.InvalidMetadata;
    if (values.len != count) return error.InvalidMetadata;
    return values;
}

fn knownArchitectureKey(config: *const Config, key: []const u8) bool {
    for (shared_integer_settings) |setting| if (std.mem.eql(u8, key, setting.key)) return true;
    for (per_config_keys) |known| if (std.mem.eql(u8, key, known)) return true;
    for (config_integer_keys) |known| if (std.mem.eql(u8, key, known)) return true;
    if (config.experts != null) for (expert_integer_keys) |known| if (std.mem.eql(u8, key, known)) return true;
    for (float_settings) |setting| if (std.mem.eql(u8, key, setting.key)) return true;
    for (array_settings) |known| if (std.mem.eql(u8, key, known)) return true;
    return false;
}

const Storage = enum { f32, matrix };
const Binder = struct {
    remaining: std.StringHashMap(*const Tensor),
    config: *const Config,
    tensors: u32 = 0,
    bytes: u64 = 0,

    fn take(self: *Binder, name: []const u8, dimensions: []const u64, storage: Storage) Error!*const Tensor {
        const entry = self.remaining.fetchRemove(name) orelse return error.MissingTensor;
        const tensor = entry.value;
        if (!std.mem.eql(u64, tensor.dimensions, dimensions)) return error.InvalidTensorShape;
        const supported = switch (storage) {
            .f32 => tensor.encoding_id == 0,
            .matrix => executableEncoding(tensor.encoding_id),
        };
        if (!supported) return error.UnsupportedTensorEncoding;
        self.tensors += 1;
        self.bytes = try std.math.add(u64, self.bytes, tensor.bytes);
        return tensor;
    }
    fn weight(self: *Binder, index: usize, suffix: []const u8, dimensions: []const u64, storage: Storage) Error!*const Tensor {
        var name: [128]u8 = undefined;
        const key = std.fmt.bufPrint(&name, "blk.{d}.{s}", .{ index, suffix }) catch unreachable;
        return self.take(key, dimensions, storage);
    }
    fn expertLayer(self: *Binder, index: usize, experts: Experts) Error!ExpertLayer {
        const d = self.config.embedding;
        const ff = experts.feed_forward;
        const n = experts.count;
        return .{
            .router = try self.weight(index, "ffn_gate_inp.weight", &.{ d, n }, .f32),
            .router_scale = try self.weight(index, "ffn_gate_inp.scale", &.{d}, .f32),
            .gate_up = try self.weight(index, "ffn_gate_up_exps.weight", &.{ d, 2 * ff, n }, .matrix),
            .down = try self.weight(index, "ffn_down_exps.weight", &.{ ff, d, n }, .matrix),
            .down_scale = try self.weight(index, "ffn_down_exps.scale", &.{n}, .f32),
            .post_ffn_norm_1 = try self.weight(index, "post_ffw_norm_1.weight", &.{d}, .f32),
            .pre_ffn_norm_2 = try self.weight(index, "pre_ffw_norm_2.weight", &.{d}, .f32),
            .post_ffn_norm_2 = try self.weight(index, "post_ffw_norm_2.weight", &.{d}, .f32),
        };
    }
    fn layer(self: *Binder, index: usize) Error!Layer {
        const config = self.config;
        const kind = kindOf(index);
        const kv_heads = kvHeadsOf(config, kind);
        const hd = kind.headSize();
        const q = config.heads * hd;
        const kv = kv_heads * hd;
        const d = config.embedding;
        const ff = config.feed_forward;
        const owns_kv = index < config.kv_layers;
        const has_value = owns_kv and (kind == .sliding or config.global_value);
        const w = config.per_layer_input;
        return .{
            .kind = kind,
            .heads = config.heads,
            .kv_heads = kv_heads,
            .attention_norm = try self.weight(index, "attn_norm.weight", &.{d}, .f32),
            .post_attention_norm = try self.weight(index, "post_attention_norm.weight", &.{d}, .f32),
            .ffn_norm = try self.weight(index, "ffn_norm.weight", &.{d}, .f32),
            .post_ffn_norm = try self.weight(index, "post_ffw_norm.weight", &.{d}, .f32),
            .query = try self.weight(index, "attn_q.weight", &.{ d, q }, .matrix),
            .key = if (owns_kv) try self.weight(index, "attn_k.weight", &.{ d, kv }, .matrix) else null,
            .value = if (has_value) try self.weight(index, "attn_v.weight", &.{ d, kv }, .matrix) else null,
            .output = try self.weight(index, "attn_output.weight", &.{ q, d }, .matrix),
            .query_norm = try self.weight(index, "attn_q_norm.weight", &.{hd}, .f32),
            .key_norm = if (owns_kv) try self.weight(index, "attn_k_norm.weight", &.{hd}, .f32) else null,
            .ffn_gate = try self.weight(index, "ffn_gate.weight", &.{ d, ff }, .matrix),
            .ffn_up = try self.weight(index, "ffn_up.weight", &.{ d, ff }, .matrix),
            .ffn_down = try self.weight(index, "ffn_down.weight", &.{ ff, d }, .matrix),
            .output_scale = try self.weight(index, "layer_output_scale.weight", &.{1}, .f32),
            .experts = if (config.experts) |experts| try self.expertLayer(index, experts) else null,
            .per_layer = if (w != 0) .{
                .gate = try self.weight(index, "inp_gate.weight", &.{ d, w }, .matrix),
                .projection = try self.weight(index, "proj.weight", &.{ w, d }, .matrix),
                .post_norm = try self.weight(index, "post_norm.weight", &.{d}, .f32),
            } else null,
        };
    }
};

/// doc must come from successful GGUF parsing. Allocations are temporary
/// lookup storage, freed before return; only the returned tensor references
/// borrow doc. Every tensor in the file must be claimed: an extra one is
/// `UnexpectedTensor` (a file with a separate `output.weight` is a different
/// checkpoint, not this one).
pub fn bind(alloc: std.mem.Allocator, doc: *const gguf.Document) Error!Binding {
    const config = try validateMetadata(doc);
    var binder: Binder = .{ .remaining = .init(alloc), .config = config };
    defer binder.remaining.deinit();
    for (doc.tensors) |*tensor| {
        const previous = try binder.remaining.fetchPut(tensor.name, tensor);
        if (previous != null) return error.DuplicateTensor;
    }
    var result: Binding = undefined;
    result.config = config;
    result.draft = null;
    result.token_embedding = try binder.take("token_embd.weight", &.{ config.embedding, vocabulary }, .matrix);
    result.output_norm = try binder.take("output_norm.weight", &.{config.embedding}, .f32);
    result.rope_factors = try binder.take("rope_freqs.weight", &.{256}, .f32);
    const w = config.per_layer_input;
    result.per_layer = if (w != 0) .{
        .token_embedding = try binder.take("per_layer_token_embd.weight", &.{ w * config.layer_count, vocabulary }, .matrix),
        .model_projection = try binder.take("per_layer_model_proj.weight", &.{ config.embedding, w * config.layer_count }, .matrix),
        .projection_norm = try binder.take("per_layer_proj_norm.weight", &.{w}, .f32),
    } else null;
    for (result.layers[0..config.layer_count], 0..) |*layer, index| layer.* = try binder.layer(index);
    if (binder.remaining.count() != 0) return error.UnexpectedTensor;
    result.summary = .{
        .profile = config.profile,
        .decoder_layers = @intCast(config.layer_count),
        .layer_kinds = config.layer_kinds,
        .auxiliary_prediction_layers = 0,
        .text_tensors = binder.tensors,
        .auxiliary_tensors = 0,
        .text_tensor_bytes = binder.bytes,
        .auxiliary_tensor_bytes = 0,
    };
    return result;
}

/// The pinned 12B K-quant artifact's directory inventory
/// (`fixtures/gemma4-12b.json`, docs/reference/gemma4.md) hydrated to a
/// Document with no weights.
pub fn inventoryDocument(gpa: std.mem.Allocator) !gguf.Document {
    return @import("inventory.zig").document(gpa, @embedFile("fixtures/gemma4-12b.json"));
}
/// The pinned 26B-A4B QAT artifact's inventory (`fixtures/gemma4-26b-a4b.json`).
pub fn inventoryDocument26bA4b(gpa: std.mem.Allocator) !gguf.Document {
    return @import("inventory.zig").document(gpa, @embedFile("fixtures/gemma4-26b-a4b.json"));
}
/// The pinned E4B QAT artifact's inventory (`fixtures/gemma4-e4b.json`).
pub fn inventoryDocumentE4b(gpa: std.mem.Allocator) !gguf.Document {
    return @import("inventory.zig").document(gpa, @embedFile("fixtures/gemma4-e4b.json"));
}

test "the pinned 12B inventory binds: 667 tensors, the period-6 pattern, no value on global layers" {
    var doc = try inventoryDocument(std.testing.allocator);
    defer doc.deinit();
    const binding = try bind(std.testing.allocator, &doc);
    try std.testing.expectEqual(&config_12b, binding.config);
    try std.testing.expectEqual(@as(u32, 667), binding.summary.text_tensors);
    try std.testing.expectEqual(@as(u64, 7_350_597_824), binding.summary.text_tensor_bytes);
    try std.testing.expectEqualStrings("gemma4_12b", binding.summary.profile);
    try std.testing.expectEqual(@as(usize, 48), binding.active().len);
    var globals: usize = 0;
    for (binding.active(), 0..) |layer, i| {
        try std.testing.expectEqual(kindOf(i), layer.kind);
        try std.testing.expect(layer.experts == null);
        if (layer.kind == .global) {
            globals += 1;
            try std.testing.expect(layer.value == null);
            try std.testing.expectEqual(@as(usize, 1), layer.kv_heads);
            try std.testing.expectEqual(@as(u64, 8192), layer.query.dimensions[1]);
            try std.testing.expectEqual(@as(u64, 512), layer.key_norm.?.dimensions[0]);
        } else {
            try std.testing.expect(layer.value != null);
            try std.testing.expectEqual(@as(u64, 2048), layer.value.?.dimensions[1]);
        }
    }
    try std.testing.expectEqual(@as(usize, 8), globals);
    try std.testing.expectEqualStrings("token_embd.weight", binding.token_embedding.name);
}

test "the pinned 26B-A4B inventory binds: 658 tensors, two KV heads on global layers, an expert block per layer" {
    var doc = try inventoryDocument26bA4b(std.testing.allocator);
    defer doc.deinit();
    const binding = try bind(std.testing.allocator, &doc);
    try std.testing.expectEqual(&config_26b_a4b, binding.config);
    try std.testing.expectEqual(@as(u32, 658), binding.summary.text_tensors);
    try std.testing.expectEqual(@as(u64, 14_233_222_264), binding.summary.text_tensor_bytes);
    try std.testing.expectEqualStrings("gemma4_26b_a4b", binding.summary.profile);
    try std.testing.expectEqual(@as(u32, 30), binding.summary.decoder_layers);
    try std.testing.expectEqual(@as(usize, 30), binding.active().len);
    var globals: usize = 0;
    for (binding.active(), 0..) |layer, i| {
        try std.testing.expectEqual(kindOf(i), layer.kind);
        const experts = layer.experts orelse return error.TestUnexpectedResult;
        try std.testing.expectEqualSlices(u64, &.{ 2816, 1408, 128 }, experts.gate_up.dimensions);
        try std.testing.expectEqualSlices(u64, &.{ 704, 2816, 128 }, experts.down.dimensions);
        try std.testing.expectEqual(@as(u32, 2), experts.down.encoding_id);
        try std.testing.expectEqual(@as(u32, 0), experts.router.encoding_id);
        if (layer.kind == .global) {
            globals += 1;
            try std.testing.expect(layer.value == null);
            try std.testing.expectEqual(@as(usize, 2), layer.kv_heads);
            try std.testing.expectEqual(@as(usize, 1024), layer.kvWidth());
            try std.testing.expectEqual(@as(u64, 1024), layer.key.?.dimensions[1]);
        } else {
            try std.testing.expectEqual(@as(usize, 2048), layer.kvWidth());
        }
    }
    try std.testing.expectEqual(@as(usize, 5), globals);
    try std.testing.expectEqual(@as(u64, 262144), binding.token_embedding.dimensions[1]);
}

test "the pinned E4B inventory binds: 666 tensors, per-layer embeddings, 24 cache-owning layers" {
    var doc = try inventoryDocumentE4b(std.testing.allocator);
    defer doc.deinit();
    const binding = try bind(std.testing.allocator, &doc);
    try std.testing.expectEqual(&config_e4b, binding.config);
    try std.testing.expectEqual(@as(u32, 666), binding.summary.text_tensors);
    try std.testing.expectEqualStrings("gemma4_e4b", binding.summary.profile);
    const ple = binding.per_layer orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualSlices(u64, &.{ 10752, 262144 }, ple.token_embedding.dimensions);
    try std.testing.expectEqualSlices(u64, &.{ 2560, 10752 }, ple.model_projection.dimensions);
    for (binding.active(), 0..) |layer, i| {
        try std.testing.expectEqual(kindOf(i), layer.kind);
        try std.testing.expectEqual(@as(usize, 2), layer.kv_heads);
        try std.testing.expectEqual(layer.kind.headSize() * 8, layer.queryWidth());
        try std.testing.expect(layer.per_layer != null);
        const owns = i < 24;
        try std.testing.expectEqual(owns, layer.key != null);
        try std.testing.expectEqual(owns, layer.key_norm != null);
        // Global layers have their own value projection on this checkpoint.
        try std.testing.expectEqual(owns, layer.value != null);
    }
    try std.testing.expectEqual(@as(usize, 22), config_e4b.kvSource(24));
    try std.testing.expectEqual(@as(usize, 23), config_e4b.kvSource(29));
    try std.testing.expectEqual(@as(usize, 22), config_e4b.kvSource(40));
    try std.testing.expectEqual(@as(usize, 23), config_e4b.kvSource(41));
    try std.testing.expectEqual(@as(usize, 23), config_e4b.kvSource(23));
    try std.testing.expectEqual(@as(usize, 47), config_12b.kvSource(47));
    try std.testing.expectEqual(@as(usize, 1), config_e4b.firstVisible(.sliding, 512));
    try std.testing.expectEqual(@as(usize, 513), config_12b.firstVisible(.sliding, 1536));
}

test "every configuration's shared layers read a cache-owning layer of their own kind" {
    for (configs) |config| {
        for (0..config.layer_count) |il| {
            const source = config.kvSource(il);
            try std.testing.expect(source < config.kv_layers);
            try std.testing.expectEqual(kindOf(il), kindOf(source));
        }
    }
}

const Mutation = union(enum) {
    drop_tensor: []const u8,
    reshape: struct { name: []const u8, dimensions: []const u64 },
    encode: struct { name: []const u8, id: u32 },
    add_tensor: []const u8,
    set_unsigned: struct { key: []const u8, value: u64 },
    flip_pattern: usize,
    drop_key: []const u8,
    add_key: []const u8,
};

fn mutated(base: *const Config, mutation: Mutation) !gguf.Document {
    var doc = try if (base == &config_12b) inventoryDocument(std.testing.allocator) else if (base == &config_e4b) inventoryDocumentE4b(std.testing.allocator) else inventoryDocument26bA4b(std.testing.allocator);
    errdefer doc.deinit();
    const alloc = doc.storage.allocator();
    // The Document's slices are const views; mutate owned copies in its arena.
    const tensors = try alloc.dupe(Tensor, doc.tensors);
    doc.tensors = tensors;
    const metadata = try alloc.dupe(gguf.Metadata, doc.metadata);
    doc.metadata = metadata;
    switch (mutation) {
        .drop_tensor => |name| {
            var kept: std.ArrayList(Tensor) = .empty;
            for (doc.tensors) |t| if (!std.mem.eql(u8, t.name, name)) try kept.append(alloc, t);
            doc.tensors = try kept.toOwnedSlice(alloc);
        },
        .reshape => |r| for (tensors) |*t| if (std.mem.eql(u8, t.name, r.name)) {
            const dims = try alloc.dupe(u64, r.dimensions);
            t.dimensions = dims;
        },
        .encode => |e| for (tensors) |*t| if (std.mem.eql(u8, t.name, e.name)) {
            t.encoding_id = e.id;
        },
        .add_tensor => |name| {
            const extra: Tensor = .{ .name = name, .dimensions = try alloc.dupe(u64, &.{base.embedding}), .encoding_id = 0, .offset = 0, .elements = base.embedding, .bytes = base.embedding * 4 };
            const more = try alloc.alloc(Tensor, doc.tensors.len + 1);
            @memcpy(more[0..doc.tensors.len], doc.tensors);
            more[doc.tensors.len] = extra;
            doc.tensors = more;
        },
        .set_unsigned => |s| for (metadata) |*m| if (std.mem.eql(u8, m.key, s.key)) {
            m.value = .{ .unsigned = s.value };
        },
        .flip_pattern => |index| for (metadata) |*m| if (std.mem.eql(u8, m.key, "gemma4.attention.sliding_window_pattern")) {
            const values = try alloc.dupe(gguf.Value, m.value.array.values.?);
            values[index] = .{ .boolean = !values[index].boolean };
            m.value.array.values = values;
        },
        .drop_key => |key| {
            var kept: std.ArrayList(gguf.Metadata) = .empty;
            for (doc.metadata) |m| if (!std.mem.eql(u8, m.key, key)) try kept.append(alloc, m);
            doc.metadata = try kept.toOwnedSlice(alloc);
        },
        .add_key => |key| {
            const more = try alloc.alloc(gguf.Metadata, doc.metadata.len + 1);
            @memcpy(more[0..doc.metadata.len], doc.metadata);
            more[doc.metadata.len] = .{ .key = key, .kind = .uint32, .value = .{ .unsigned = 1 } };
            doc.metadata = more;
        },
    }
    return doc;
}

test "deviations from the pinned configurations are typed rejections" {
    const cases = [_]struct { base: *const Config = &config_12b, mutation: Mutation, expected: anyerror }{
        .{ .mutation = .{ .drop_tensor = "blk.3.attn_v.weight" }, .expected = error.MissingTensor },
        .{ .mutation = .{ .drop_tensor = "rope_freqs.weight" }, .expected = error.MissingTensor },
        .{ .mutation = .{ .add_tensor = "output.weight" }, .expected = error.UnexpectedTensor },
        .{ .mutation = .{ .add_tensor = "blk.5.attn_v.weight" }, .expected = error.UnexpectedTensor },
        .{ .mutation = .{ .reshape = .{ .name = "blk.5.attn_q.weight", .dimensions = &.{ 3840, 4096 } } }, .expected = error.InvalidTensorShape },
        .{ .mutation = .{ .reshape = .{ .name = "blk.0.layer_output_scale.weight", .dimensions = &.{2} } }, .expected = error.InvalidTensorShape },
        .{ .mutation = .{ .encode = .{ .name = "blk.0.ffn_up.weight", .id = 30 } }, .expected = error.UnsupportedTensorEncoding }, // BF16: stored, never executed
        .{ .mutation = .{ .encode = .{ .name = "blk.0.attn_norm.weight", .id = 1 } }, .expected = error.UnsupportedTensorEncoding },
        .{ .mutation = .{ .set_unsigned = .{ .key = "gemma4.attention.sliding_window", .value = 512 } }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .set_unsigned = .{ .key = "gemma4.block_count", .value = 31 } }, .expected = error.UnsupportedConfiguration },
        // A 12B directory claiming the 26B-A4B's layer count fails its widths, not its arrays.
        .{ .mutation = .{ .set_unsigned = .{ .key = "gemma4.block_count", .value = 30 } }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .flip_pattern = 5 }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .flip_pattern = 0 }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .drop_key = "gemma4.final_logit_softcapping" }, .expected = error.MissingMetadata },
        .{ .mutation = .{ .drop_key = "gemma4.attention.head_count_kv" }, .expected = error.MissingMetadata },
        .{ .mutation = .{ .add_key = "gemma4.attention.logit_softcapping" }, .expected = error.UnsupportedConfiguration },
        // Expert keys are known only to the expert configuration.
        .{ .mutation = .{ .add_key = "gemma4.expert_count" }, .expected = error.UnsupportedConfiguration },
        .{ .base = &config_26b_a4b, .mutation = .{ .drop_tensor = "blk.0.ffn_down_exps.scale" }, .expected = error.MissingTensor },
        .{ .base = &config_26b_a4b, .mutation = .{ .drop_tensor = "blk.7.pre_ffw_norm_2.weight" }, .expected = error.MissingTensor },
        .{ .base = &config_26b_a4b, .mutation = .{ .reshape = .{ .name = "blk.0.ffn_gate_up_exps.weight", .dimensions = &.{ 2816, 1408, 127 } } }, .expected = error.InvalidTensorShape },
        .{ .base = &config_26b_a4b, .mutation = .{ .reshape = .{ .name = "blk.5.attn_k.weight", .dimensions = &.{ 2816, 512 } } }, .expected = error.InvalidTensorShape },
        .{ .base = &config_26b_a4b, .mutation = .{ .encode = .{ .name = "blk.0.ffn_gate_inp.weight", .id = 2 } }, .expected = error.UnsupportedTensorEncoding },
        .{ .base = &config_26b_a4b, .mutation = .{ .set_unsigned = .{ .key = "gemma4.expert_used_count", .value = 9 } }, .expected = error.UnsupportedConfiguration },
        .{ .base = &config_26b_a4b, .mutation = .{ .drop_key = "gemma4.expert_feed_forward_length" }, .expected = error.MissingMetadata },
        .{ .base = &config_26b_a4b, .mutation = .{ .flip_pattern = 11 }, .expected = error.UnsupportedConfiguration },
        .{ .base = &config_e4b, .mutation = .{ .drop_tensor = "blk.23.attn_v.weight" }, .expected = error.MissingTensor },
        .{ .base = &config_e4b, .mutation = .{ .add_tensor = "blk.24.attn_k_norm.weight" }, .expected = error.UnexpectedTensor },
        .{ .base = &config_e4b, .mutation = .{ .drop_tensor = "per_layer_proj_norm.weight" }, .expected = error.MissingTensor },
        .{ .base = &config_e4b, .mutation = .{ .drop_tensor = "blk.41.inp_gate.weight" }, .expected = error.MissingTensor },
        .{ .base = &config_e4b, .mutation = .{ .set_unsigned = .{ .key = "gemma4.attention.shared_kv_layers", .value = 20 } }, .expected = error.UnsupportedConfiguration },
        .{ .base = &config_e4b, .mutation = .{ .set_unsigned = .{ .key = "gemma4.embedding_length_per_layer_input", .value = 128 } }, .expected = error.UnsupportedConfiguration },
        .{ .base = &config_e4b, .mutation = .{ .set_unsigned = .{ .key = "gemma4.attention.head_count_kv", .value = 4 } }, .expected = error.UnsupportedConfiguration },
        .{ .base = &config_e4b, .mutation = .{ .set_unsigned = .{ .key = "gemma4.attention.sliding_window", .value = 1024 } }, .expected = error.UnsupportedConfiguration },
    };
    for (cases) |case| {
        var doc = try mutated(case.base, case.mutation);
        defer doc.deinit();
        try std.testing.expectError(case.expected, bind(std.testing.allocator, &doc));
    }
}

test "binding survives allocation failure without leaks" {
    inline for (.{ inventoryDocument, inventoryDocument26bA4b, inventoryDocumentE4b }) |open| {
        var doc = try open(std.testing.allocator);
        defer doc.deinit();
        try std.testing.checkAllAllocationFailures(std.testing.allocator, struct {
            fn check(alloc: std.mem.Allocator, d: *const gguf.Document) !void {
                _ = try bind(alloc, d);
            }
        }.check, .{&doc});
    }
}

test "kindOf and the geometry constants" {
    try std.testing.expectEqual(Kind.global, kindOf(5));
    try std.testing.expectEqual(Kind.global, kindOf(47));
    try std.testing.expectEqual(Kind.sliding, kindOf(0));
    try std.testing.expectEqual(Kind.sliding, kindOf(46));
    try std.testing.expectEqual(@as(usize, 4096), Kind.sliding.queryWidth());
    try std.testing.expectEqual(@as(usize, 8192), Kind.global.queryWidth());
    try std.testing.expectEqual(@as(usize, 8), kvHeadsOf(&config_12b, .sliding));
    try std.testing.expectEqual(@as(usize, 1), kvHeadsOf(&config_12b, .global));
    try std.testing.expectEqual(@as(usize, 2), kvHeadsOf(&config_26b_a4b, .global));
    try std.testing.expectEqual(@as(f32, @sqrt(3840.0)), config_12b.embeddingScale());
    try std.testing.expect(executableEncoding(12) and executableEncoding(2) and !executableEncoding(30));
}
