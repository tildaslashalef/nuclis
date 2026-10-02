//! Structural adapter for the hybrid DeltaNet GGUF family (`qwen35`):
//! Qwen3.8-27B and the Qwen3.5-9B shape (clef-flash's backbone), whose
//! dimensions the file declares (`Shape`); everything else is pinned.
//! Bind once at model load: later numerical code uses named tensor references,
//! never repeated GGUF string lookups. Bindings borrow the Document's storage.
//! Validation checks the declared profile and storage, not tensor values or
//! numerical execution. See docs/reference/qwen-validation.md for the pinned references.
//! The same adapter binds Bonsai 2 27B, the ternary re-encoding whose weights
//! sit in a Hadamard-rotated basis (`Rotation`, docs/reference/bonsai.md),
//! with or without the auxiliary prediction block.
const std = @import("std");
const gguf = @import("../formats/gguf.zig");
const models = @import("root.zig");
const Tensor = gguf.Tensor;

pub const FullAttention = struct {
    query_and_gate: *const Tensor,
    key: *const Tensor,
    value: *const Tensor,
    output: *const Tensor,
    query_norm: *const Tensor,
    key_norm: *const Tensor,
};

pub const DeltaNet = struct {
    qkv: *const Tensor,
    gate: *const Tensor,
    convolution: *const Tensor,
    time_bias: *const Tensor,
    a: *const Tensor,
    beta: *const Tensor,
    alpha: *const Tensor,
    norm: *const Tensor,
    output: *const Tensor,
};

pub const Layer = struct {
    attention_norm: *const Tensor,
    post_attention_norm: *const Tensor,
    ffn_gate: *const Tensor,
    ffn_up: *const Tensor,
    ffn_down: *const Tensor,
    mixer: union(enum) { full_attention: FullAttention, delta_net: DeltaNet },
};

/// The `blk.64` prediction head: the four `nextn` tensors plus one dense
/// full-attention decoder layer of the main shape. Bound only when the file
/// declares the 65th block; the text schedule never references it. The head
/// shares `token_embd` and `output` with the text binding
/// (docs/reference/speculative-decoding.md § The Qwen3.8 draft head).
pub const DraftBlock = struct {
    eh_proj: *const Tensor,
    enorm: *const Tensor,
    hnorm: *const Tensor,
    shared_head_norm: *const Tensor,
    layer: Layer,
};

/// The `general.architecture` id this adapter binds.
pub const architecture = "qwen35";

/// One image span in a prompt (the shared vision contract's `Span`), used by
/// the CPU runtime and the Metal plan.
pub const VisionSpan = @import("../vision/root.zig").Span;

/// The embedded prediction block's proposal bound: the block forward chains
/// one draft per position and the Metal plan's verify tile holds 8 rows (the
/// seed plus 7). Declared by the adapter so both executors cap a requested
/// draft length the same way.
pub const max_draft_proposals = 7;

/// Whether a weight matrix of this encoding can be bound: the storage
/// layouts the CPU reference and the Metal kernels execute. Layout support
/// in `formats/gguf` (which also stores Q4_0) is not this claim. BF16 and the
/// ternary PQ2_0 / PTQ1_0 are the Bonsai file's encodings.
pub fn executableEncoding(encoding_id: u32) bool {
    return switch (encoding_id) {
        0, 8, 11, 12, 13, 14, 20, 21, 23, 30, 142, 143 => true,
        else => false,
    };
}

/// The activation-side transform folded into a rotated file's weights
/// (`prism.hadamard.*`): before every projection except `ssm_alpha` and
/// `ssm_beta`, the input is multiplied elementwise by its width's sign
/// vector and then by the normalized Sylvester Walsh-Hadamard matrix per
/// `block` elements; the embedding row gets the transform and then the
/// signs after lookup, since the table stores rotated rows. The slices
/// borrow the Document's metadata.
pub const Rotation = struct {
    block: u32,
    /// One vector of `.signed` ±1 per projection input width, in `widths` order.
    signs: [3]Signs,
    /// `ssm_out` sees its input in group order (head dimension fastest, then
    /// the 16 key groups, then the 3 value heads of a group) before the signs
    /// and the transform; the tiled order is what the mixer produces.
    value_grouped: bool,

    pub const Signs = struct { width: u32, values: []const gguf.Value };
    pub const widths = [3]u32{ 5120, 6144, 17408 };
    pub const block_size = 1024;

    pub fn signsFor(self: Rotation, width: u32) ?[]const gguf.Value {
        for (self.signs) |signs| if (signs.width == width) return signs.values;
        return null;
    }
};

/// The dimensions a file declares: the six keys the 27B and 9B releases
/// differ in. The head widths, the 4 KV heads, the 16 DeltaNet key heads,
/// and the every-4th full-attention schedule stay pinned.
pub const Shape = struct {
    hidden: usize,
    ffn: usize,
    /// Full-attention query heads.
    heads: usize,
    /// Text layers, without the auxiliary prediction block.
    layers: usize,
    /// DeltaNet value heads (`ssm.time_step_rank`).
    value_heads: usize,

    pub const max_layers = 64;
    pub const kv_heads = 4;
    pub const head_width = 256;
    pub const kv_width = kv_heads * head_width;
    pub const key_heads = 16;
    pub const state_width = 128;
    pub const conv_kernel = 4;

    pub fn queryWidth(s: Shape) usize {
        return s.heads * head_width;
    }
    /// `attn_q` emits each head's query then its output gate.
    pub fn queryGateWidth(s: Shape) usize {
        return 2 * s.queryWidth();
    }
    /// The DeltaNet value width (`ssm.inner_size`).
    pub fn innerWidth(s: Shape) usize {
        return s.value_heads * state_width;
    }
    /// `attn_qkv`'s output: [Q | K: 16 key heads each | V: value heads].
    pub fn qkvWidth(s: Shape) usize {
        return 2 * key_heads * state_width + s.innerWidth();
    }
    /// One DeltaNet layer's recurrent state: the convolution history and the
    /// per-value-head matrices.
    pub fn deltaHistory(s: Shape) usize {
        return s.qkvWidth() * (conv_kernel - 1);
    }
    pub fn deltaMatrix(s: Shape) usize {
        return s.value_heads * state_width * state_width;
    }
    pub fn eql(a: Shape, b: Shape) bool {
        return std.meta.eql(a, b);
    }
};

/// Qwen3.8-27B (and Bonsai 2 27B): the shape the rotation contract is pinned to.
pub const qwen38_shape: Shape = .{ .hidden = 5120, .ffn = 17408, .heads = 24, .layers = 64, .value_heads = 48 };
/// Qwen3.5-9B, clef-flash's backbone.
pub const qwen35_9b_shape: Shape = .{ .hidden = 4096, .ffn = 12288, .heads = 16, .layers = 32, .value_heads = 32 };

/// What a binding reports (`models.Summary`): the layer composition (a
/// quarter full attention, the rest DeltaNet) plus, when the file carries
/// it, one auxiliary prediction layer excluded from the text schedule.
pub const Summary = models.Summary;
/// Layer kinds per accepted layer count (4 to 64 in steps of 4), so a summary
/// borrows static storage.
const layer_kind_table = blk: {
    var table: [Shape.max_layers / 4][2]models.LayerKind = undefined;
    for (&table, 1..) |*kinds, n| kinds.* = .{
        .{ .kind = "full_attention", .count = n },
        .{ .kind = "delta_net", .count = 3 * n },
    };
    break :blk table;
};

/// All tensor pointers borrow doc.tensors. No weights are read or allocated;
/// keep the source Document alive for the entire binding lifetime.
pub const Binding = struct {
    token_embedding: *const Tensor,
    output_norm: *const Tensor,
    output: *const Tensor,
    /// The first `shape.layers` entries are bound; read them through `layers()`.
    layer_slots: [Shape.max_layers]Layer,
    shape: Shape,
    summary: Summary,
    /// Present on a rotated (Bonsai) file; the runtimes must apply it.
    rotation: ?Rotation = null,
    /// The embedded prediction head, present on the Qwen3.8 release (65
    /// blocks) and absent from the Bonsai re-encoding.
    draft: ?DraftBlock = null,

    /// The text layers, in schedule order.
    pub fn layers(self: *const Binding) []const Layer {
        return self.layer_slots[0..self.shape.layers];
    }
};

/// The registry's shared binding error set; this adapter returns all of it.
pub const Error = models.BindError;

/// The registry entry (`models.table`): the names every family exposes,
/// bound to this adapter's modules. The runtime and plan modules import
/// this file for the binding, and this file names them here; Zig resolves
/// the cycle lazily.
pub const family = struct {
    pub const architecture = @import("qwen35.zig").architecture;
    pub const executableEncoding = @import("qwen35.zig").executableEncoding;
    pub const Binding = @import("qwen35.zig").Binding;
    pub const bind = @import("qwen35.zig").bind;
    pub const Runtime = @import("qwen35_runtime.zig").Runtime;
    pub const Plan = @import("qwen35_metal.zig").Plan;
    /// The artifact's own file carries the prediction block.
    pub const embedded_draft = true;
};

// The profile is deliberately narrow. Reject unimplemented variants
// instead of accepting a familiar architecture name with different semantics.
const IntegerSetting = struct { key: []const u8, value: u64 };
const integer_settings = [_]IntegerSetting{
    .{ .key = "qwen35.context_length", .value = 262144 },
    .{ .key = "qwen35.attention.head_count_kv", .value = 4 },
    .{ .key = "qwen35.attention.key_length", .value = 256 },
    .{ .key = "qwen35.attention.value_length", .value = 256 },
    .{ .key = "qwen35.ssm.conv_kernel", .value = 4 },
    .{ .key = "qwen35.ssm.state_size", .value = 128 },
    .{ .key = "qwen35.ssm.group_count", .value = 16 },
    .{ .key = "qwen35.full_attention_interval", .value = 4 },
    .{ .key = "qwen35.rope.dimension_count", .value = 64 },
    .{ .key = "general.quantization_version", .value = 2 },
};

/// The shape the file declares, bounded to what the runtimes size and the
/// kernels tile.
const shape_keys = [_][]const u8{
    "qwen35.embedding_length",     "qwen35.feed_forward_length",        "qwen35.attention.head_count",
    "qwen35.ssm.time_step_rank",   "qwen35.ssm.inner_size",             "qwen35.block_count",
    "qwen35.nextn_predict_layers", "qwen35.attention.recurrent_layers",
};

/// The text layers, plus the auxiliary prediction block when the file
/// declares it: the Qwen3.8 release counts it as one more block with one
/// `nextn` layer, the Bonsai re-encoding and the 9B drop it (no key).
fn validateMetadata(doc: *const gguf.Document) Error!struct { shape: Shape, auxiliary: bool } {
    const arch = switch (doc.get("general.architecture") orelse return error.MissingMetadata) {
        .string => |value| value,
        else => return error.InvalidMetadata,
    };
    if (!std.mem.eql(u8, arch, architecture)) return error.UnsupportedArchitecture;
    for (integer_settings) |setting| {
        const value = doc.get(setting.key) orelse return error.MissingMetadata;
        switch (value) {
            .unsigned => |n| if (n != setting.value) return error.UnsupportedConfiguration,
            else => return error.InvalidMetadata,
        }
    }
    const blocks = try unsignedValue(doc, "qwen35.block_count");
    const nextn = if (doc.get("qwen35.nextn_predict_layers")) |_| try unsignedValue(doc, "qwen35.nextn_predict_layers") else 0;
    if (nextn > 1 or blocks <= nextn) return error.UnsupportedConfiguration;
    const shape = try readShape(doc, blocks - nextn);
    // Newer converters spell the schedule out; it must be the pinned one.
    if (doc.get("qwen35.attention.recurrent_layers") != null) {
        const flags = try arrayValues(doc, "qwen35.attention.recurrent_layers", .boolean, shape.layers);
        for (flags, 0..) |flag, i| switch (flag) {
            .boolean => |recurrent| if (recurrent != ((i + 1) % 4 != 0)) return error.UnsupportedConfiguration,
            else => return error.InvalidMetadata,
        };
    }
    try expectFloat(doc, "qwen35.rope.freq_base", 10_000_000);
    try expectFloat(doc, "qwen35.attention.layer_norm_rms_epsilon", @as(f32, 1e-6));
    const sections = switch (doc.get("qwen35.rope.dimension_sections") orelse return error.MissingMetadata) {
        .array => |a| a,
        else => return error.InvalidMetadata,
    };
    if (sections.element_type != .int32 or sections.count != 4) return error.UnsupportedConfiguration;
    const values = sections.values orelse return error.InvalidMetadata;
    if (values.len != 4) return error.InvalidMetadata;
    for (values, [_]i64{ 11, 11, 10, 0 }) |value, expected| {
        switch (value) {
            .signed => |n| if (n != expected) return error.UnsupportedConfiguration,
            else => return error.InvalidMetadata,
        }
    }
    const tokens = switch (doc.get("tokenizer.ggml.tokens") orelse return error.MissingMetadata) {
        .array => |a| a,
        else => return error.InvalidMetadata,
    };
    if (tokens.element_type != .string or tokens.count != 248320) return error.UnsupportedConfiguration;

    // These keys can override the equations/schedule even when the scalar
    // dimensions still match. Supporting them requires a separate profile.
    for (doc.metadata) |entry| {
        if (std.mem.startsWith(u8, entry.key, "qwen35.") and !knownArchitectureKey(entry.key))
            return error.UnsupportedConfiguration;
    }
    return .{ .shape = shape, .auxiliary = nextn == 1 };
}

/// Each dimension a multiple of the 256-element quantization block, the
/// heads whole groups of the pinned KV and key heads, the layers whole
/// periods of the every-4th schedule.
fn readShape(doc: *const gguf.Document, layers: u64) Error!Shape {
    const hidden = try unsignedValue(doc, "qwen35.embedding_length");
    const ffn = try unsignedValue(doc, "qwen35.feed_forward_length");
    const heads = try unsignedValue(doc, "qwen35.attention.head_count");
    const value_heads = try unsignedValue(doc, "qwen35.ssm.time_step_rank");
    const inner = try unsignedValue(doc, "qwen35.ssm.inner_size");
    if (hidden == 0 or hidden > 16384 or hidden % 256 != 0) return error.UnsupportedConfiguration;
    if (ffn == 0 or ffn > 65536 or ffn % 256 != 0) return error.UnsupportedConfiguration;
    if (heads == 0 or heads > 64 or heads % Shape.kv_heads != 0) return error.UnsupportedConfiguration;
    if (value_heads == 0 or value_heads > 64 or value_heads % Shape.key_heads != 0) return error.UnsupportedConfiguration;
    if (inner != value_heads * Shape.state_width) return error.UnsupportedConfiguration;
    if (layers == 0 or layers > Shape.max_layers or layers % 4 != 0) return error.UnsupportedConfiguration;
    return .{ .hidden = @intCast(hidden), .ffn = @intCast(ffn), .heads = @intCast(heads), .layers = @intCast(layers), .value_heads = @intCast(value_heads) };
}

fn knownArchitectureKey(key: []const u8) bool {
    for (integer_settings) |setting| if (std.mem.eql(u8, key, setting.key)) return true;
    for (shape_keys) |known| if (std.mem.eql(u8, key, known)) return true;
    for ([_][]const u8{
        "qwen35.rope.freq_base", "qwen35.attention.layer_norm_rms_epsilon", "qwen35.rope.dimension_sections",
    }) |known| if (std.mem.eql(u8, key, known)) return true;
    return false;
}

fn unsignedValue(doc: *const gguf.Document, key: []const u8) Error!u64 {
    return switch (doc.get(key) orelse return error.MissingMetadata) {
        .unsigned => |value| value,
        else => error.InvalidMetadata,
    };
}

fn expectFloat(doc: *const gguf.Document, key: []const u8, expected: f64) Error!void {
    switch (doc.get(key) orelse return error.MissingMetadata) {
        .float => |value| if (value != expected) return error.UnsupportedConfiguration,
        else => return error.InvalidMetadata,
    }
}

fn expectString(doc: *const gguf.Document, key: []const u8, expected: []const u8) Error!void {
    switch (doc.get(key) orelse return error.MissingMetadata) {
        .string => |value| if (!std.mem.eql(u8, value, expected)) return error.UnsupportedConfiguration,
        else => return error.InvalidMetadata,
    }
}

/// A retained array of exactly `count` elements of `kind`.
fn arrayValues(doc: *const gguf.Document, key: []const u8, kind: gguf.Type, count: usize) Error![]const gguf.Value {
    const array = switch (doc.get(key) orelse return error.MissingMetadata) {
        .array => |a| a,
        else => return error.InvalidMetadata,
    };
    if (array.element_type != kind or array.count != count) return error.UnsupportedConfiguration;
    const values = array.values orelse return error.InvalidMetadata;
    if (values.len != count) return error.InvalidMetadata;
    return values;
}

const rotation_keys = [_][]const u8{
    "prism.hadamard.version",       "prism.hadamard.block_size",   "prism.hadamard.transform",
    "prism.hadamard.axis",          "prism.hadamard.sign_mode",    "prism.hadamard.sign_widths",
    "prism.hadamard.sign_values",   "prism.hadamard.weight_names", "prism.hadamard.inverse_weight_names",
    "prism.hadamard.gdn_v_grouped",
};

// The rotated set is pinned, not read: every matrix except the embedding
// (inverse after lookup) and the DeltaNet `ssm_alpha` / `ssm_beta`
// projections, one entry each. 48 * 6 + 16 * 7 + 1 = 401 names.
const delta_rotated = [_][]const u8{ "attn_qkv", "attn_gate", "ssm_out", "ffn_gate", "ffn_up", "ffn_down" };
const attention_rotated = [_][]const u8{ "attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down" };
const rotated_names = 48 * delta_rotated.len + 16 * attention_rotated.len + 1;

/// The rotation contract as the fork's loader reads it, pinned to the one
/// configuration the runtimes implement; anything else is refused rather
/// than run in the wrong basis. Null when the file carries no rotation.
fn validateRotation(doc: *const gguf.Document) Error!?Rotation {
    const version = doc.get("prism.hadamard.version") orelse {
        for (doc.metadata) |entry| if (std.mem.startsWith(u8, entry.key, "prism.hadamard.")) return error.UnsupportedConfiguration;
        return null;
    };
    switch (version) {
        .unsigned => |n| if (n != 1) return error.UnsupportedConfiguration,
        else => return error.InvalidMetadata,
    }
    for (doc.metadata) |entry| {
        if (!std.mem.startsWith(u8, entry.key, "prism.hadamard.")) continue;
        var known = false;
        for (rotation_keys) |key| known = known or std.mem.eql(u8, entry.key, key);
        if (!known) return error.UnsupportedConfiguration;
    }
    if (try unsignedValue(doc, "prism.hadamard.block_size") != Rotation.block_size) return error.UnsupportedConfiguration;
    try expectString(doc, "prism.hadamard.transform", "normalized-sylvester-walsh-hadamard");
    try expectString(doc, "prism.hadamard.axis", "input-last-dimension");
    try expectString(doc, "prism.hadamard.sign_mode", "explicit");
    const value_grouped = switch (doc.get("prism.hadamard.gdn_v_grouped") orelse return error.MissingMetadata) {
        .boolean => |b| b,
        else => return error.InvalidMetadata,
    };

    const widths = try arrayValues(doc, "prism.hadamard.sign_widths", .int32, Rotation.widths.len);
    var total: usize = 0;
    for (widths, Rotation.widths) |value, expected| {
        switch (value) {
            .signed => |n| if (n != expected) return error.UnsupportedConfiguration,
            else => return error.InvalidMetadata,
        }
        total += expected;
    }
    const values = try arrayValues(doc, "prism.hadamard.sign_values", .int32, total);
    for (values) |value| switch (value) {
        .signed => |n| if (n != 1 and n != -1) return error.InvalidMetadata,
        else => return error.InvalidMetadata,
    };
    var signs: [3]Rotation.Signs = undefined;
    var offset: usize = 0;
    for (&signs, Rotation.widths) |*entry, width| {
        entry.* = .{ .width = width, .values = values[offset..][0..width] };
        offset += width;
    }

    // Each listed name claims one slot of the pinned set; a name outside the
    // set or claimed twice is refused, so a full slot table means equality.
    const names = try arrayValues(doc, "prism.hadamard.weight_names", .string, rotated_names);
    var slots = [_][attention_rotated.len]bool{[_]bool{false} ** attention_rotated.len} ** 64;
    var head = false;
    for (names) |value| {
        const name = switch (value) {
            .string => |text| text,
            else => return error.InvalidMetadata,
        };
        if (std.mem.eql(u8, name, "output.weight")) {
            if (head) return error.UnsupportedConfiguration;
            head = true;
            continue;
        }
        const slot = rotatedSlot(name) orelse return error.UnsupportedConfiguration;
        if (slots[slot.layer][slot.kind]) return error.UnsupportedConfiguration;
        slots[slot.layer][slot.kind] = true;
    }
    const inverse = try arrayValues(doc, "prism.hadamard.inverse_weight_names", .string, 1);
    switch (inverse[0]) {
        .string => |name| if (!std.mem.eql(u8, name, "token_embd.weight")) return error.UnsupportedConfiguration,
        else => return error.InvalidMetadata,
    }
    return .{ .block = Rotation.block_size, .signs = signs, .value_grouped = value_grouped };
}

/// `blk.N.KIND.weight` for a rotated kind of layer N's mixer.
fn rotatedSlot(name: []const u8) ?struct { layer: usize, kind: usize } {
    if (!std.mem.startsWith(u8, name, "blk.")) return null;
    const rest = name[4..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    const layer = std.fmt.parseInt(usize, rest[0..dot], 10) catch return null;
    if (layer >= 64) return null;
    const kinds: []const []const u8 = if ((layer + 1) % 4 == 0) &attention_rotated else &delta_rotated;
    for (kinds, 0..) |kind, index| {
        if (rest.len == dot + 1 + kind.len + ".weight".len and
            std.mem.eql(u8, rest[dot + 1 ..][0..kind.len], kind) and
            std.mem.endsWith(u8, rest, ".weight")) return .{ .layer = layer, .kind = index };
    }
    return null;
}

const Storage = enum { f32, matrix };

const Binder = struct {
    remaining: std.StringHashMap(*const Tensor),
    shape: Shape,
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

    fn layer(self: *Binder, index: usize, full_attention: bool) Error!Layer {
        const s = self.shape;
        return .{
            .attention_norm = try self.weight(index, "attn_norm.weight", &.{s.hidden}, .f32),
            .post_attention_norm = try self.weight(index, "post_attention_norm.weight", &.{s.hidden}, .f32),
            .ffn_gate = try self.weight(index, "ffn_gate.weight", &.{ s.hidden, s.ffn }, .matrix),
            .ffn_up = try self.weight(index, "ffn_up.weight", &.{ s.hidden, s.ffn }, .matrix),
            .ffn_down = try self.weight(index, "ffn_down.weight", &.{ s.ffn, s.hidden }, .matrix),
            .mixer = if (full_attention) .{
                .full_attention = .{
                    // Q contains both the query and the elementwise output gate.
                    .query_and_gate = try self.weight(index, "attn_q.weight", &.{ s.hidden, s.queryGateWidth() }, .matrix),
                    .key = try self.weight(index, "attn_k.weight", &.{ s.hidden, Shape.kv_width }, .matrix),
                    .value = try self.weight(index, "attn_v.weight", &.{ s.hidden, Shape.kv_width }, .matrix),
                    .output = try self.weight(index, "attn_output.weight", &.{ s.queryWidth(), s.hidden }, .matrix),
                    .query_norm = try self.weight(index, "attn_q_norm.weight", &.{Shape.head_width}, .f32),
                    .key_norm = try self.weight(index, "attn_k_norm.weight", &.{Shape.head_width}, .f32),
                },
            } else .{
                .delta_net = .{
                    .qkv = try self.weight(index, "attn_qkv.weight", &.{ s.hidden, s.qkvWidth() }, .matrix),
                    .gate = try self.weight(index, "attn_gate.weight", &.{ s.hidden, s.innerWidth() }, .matrix),
                    .convolution = try self.weight(index, "ssm_conv1d.weight", &.{ Shape.conv_kernel, s.qkvWidth() }, .f32),
                    .time_bias = try self.weight(index, "ssm_dt.bias", &.{s.value_heads}, .f32),
                    .a = try self.weight(index, "ssm_a", &.{s.value_heads}, .f32),
                    .beta = try self.weight(index, "ssm_beta.weight", &.{ s.hidden, s.value_heads }, .matrix),
                    .alpha = try self.weight(index, "ssm_alpha.weight", &.{ s.hidden, s.value_heads }, .matrix),
                    .norm = try self.weight(index, "ssm_norm.weight", &.{Shape.state_width}, .f32),
                    .output = try self.weight(index, "ssm_out.weight", &.{ s.innerWidth(), s.hidden }, .matrix),
                },
            },
        };
    }
};

/// doc must come from successful GGUF parsing. Allocations are temporary lookup
/// storage, freed before return; only the returned tensor references borrow doc.
pub fn bind(alloc: std.mem.Allocator, doc: *const gguf.Document) Error!Binding {
    const declared = try validateMetadata(doc);
    const shape = declared.shape;
    const auxiliary = declared.auxiliary;
    const rotation = try validateRotation(doc);
    // The rotation contract (sign widths, the rotated names) is pinned to the 27B.
    if (rotation != null and !shape.eql(qwen38_shape)) return error.UnsupportedConfiguration;
    var binder: Binder = .{ .remaining = .init(alloc), .shape = shape };
    defer binder.remaining.deinit();
    for (doc.tensors) |*tensor| {
        const slot = try binder.remaining.getOrPut(tensor.name);
        if (slot.found_existing) return error.DuplicateTensor;
        slot.value_ptr.* = tensor;
    }
    var result: Binding = undefined;
    result.shape = shape;
    result.token_embedding = try binder.take("token_embd.weight", &.{ shape.hidden, 248320 }, .matrix);
    result.output_norm = try binder.take("output_norm.weight", &.{shape.hidden}, .f32);
    result.output = try binder.take("output.weight", &.{ shape.hidden, 248320 }, .matrix);
    for (result.layer_slots[0..shape.layers], 0..) |*layer, i| layer.* = try binder.layer(i, (i + 1) % 4 == 0);
    // Unbound slots are never read (`layers()` stops at the shape's count).
    for (result.layer_slots[shape.layers..]) |*layer| layer.* = undefined;
    const text_tensors = binder.tensors;
    const text_bytes = binder.bytes;

    // The auxiliary block is bound separately, never in `layers`, so the text
    // schedule keeps its length. It always uses full attention, regardless of
    // the main schedule's modulo.
    result.draft = null;
    if (auxiliary) {
        const at = shape.layers;
        result.draft = .{
            .eh_proj = try binder.weight(at, "nextn.eh_proj.weight", &.{ 2 * shape.hidden, shape.hidden }, .matrix),
            .enorm = try binder.weight(at, "nextn.enorm.weight", &.{shape.hidden}, .f32),
            .hnorm = try binder.weight(at, "nextn.hnorm.weight", &.{shape.hidden}, .f32),
            .shared_head_norm = try binder.weight(at, "nextn.shared_head_norm.weight", &.{shape.hidden}, .f32),
            .layer = try binder.layer(at, true),
        };
    }
    if (binder.remaining.count() != 0) return error.UnexpectedTensor;
    result.summary = .{
        .profile = if (shape.eql(qwen38_shape)) "qwen35_27b" else if (shape.eql(qwen35_9b_shape)) "qwen35_9b" else "qwen35",
        .decoder_layers = @intCast(shape.layers),
        .layer_kinds = &layer_kind_table[shape.layers / 4 - 1],
        .auxiliary_prediction_layers = if (auxiliary) 1 else 0,
        .rotated_basis = if (rotation != null) "normalized-sylvester-walsh-hadamard, block 1024, explicit signs" else null,
        .text_tensors = text_tensors,
        .auxiliary_tensors = binder.tensors - text_tensors,
        .text_tensor_bytes = text_bytes,
        .auxiliary_tensor_bytes = binder.bytes - text_bytes,
    };
    result.rotation = rotation;
    return result;
}

// This fixture is an independently captured directory inventory of the pinned
// GGUF, not a list generated from Binder's expected shapes. It holds no weights.
// Hydration creates owned descriptors to test the public adapter without a model
// download; container parsing has separate wire-format tests.
fn fixture() !gguf.Document {
    return inventoryDocument(std.testing.allocator);
}

/// The pinned artifact's directory inventory (`fixtures/qwen35-27b.json`)
/// hydrated into a `Document` with sequential aligned offsets and no
/// weights; large arrays keep their counts without values. Test support
/// for this module and for `nuclis model inspect`'s verdict tests, which
/// serialize it back into GGUF bytes; never referenced by the binary.
pub fn inventoryDocument(gpa: std.mem.Allocator) !gguf.Document {
    return @import("inventory.zig").document(gpa, @embedFile("fixtures/qwen35-27b.json"));
}

/// The Bonsai 2 27B PQ2_0 file's inventory (`fixtures/bonsai-2-27b.json`),
/// rotation arrays included.
pub fn bonsaiInventoryDocument(gpa: std.mem.Allocator) !gguf.Document {
    return @import("inventory.zig").document(gpa, @embedFile("fixtures/bonsai-2-27b.json"));
}

// Fixture allocations are mutable; the public Document exposes const slices.
fn metadataEntry(doc: *gguf.Document, key: []const u8) *gguf.Metadata {
    for (@constCast(doc.metadata)) |*entry| if (std.mem.eql(u8, entry.key, key)) return entry;
    unreachable;
}

fn tensorEntry(doc: *gguf.Document, name: []const u8) *Tensor {
    for (@constCast(doc.tensors)) |*tensor| if (std.mem.eql(u8, tensor.name, name)) return tensor;
    unreachable;
}

test "bind real descriptor inventory with auxiliary weights outside the text layers" {
    var doc = try fixture();
    defer doc.deinit();
    const model = try bind(std.testing.allocator, &doc);
    try std.testing.expectEqual(@as(u32, 851), model.summary.text_tensors);
    try std.testing.expectEqual(@as(u32, 15), model.summary.auxiliary_tensors);
    try std.testing.expectEqual(@as(u64, 16102434816), model.summary.text_tensor_bytes);
    try std.testing.expectEqual(@as(u64, 351008768), model.summary.auxiliary_tensor_bytes);
    try std.testing.expect(model.token_embedding == tensorEntry(&doc, "token_embd.weight"));
    var full: u32 = 0;
    var recurrent: u32 = 0;
    for (model.layers(), 0..) |layer, i| {
        switch (layer.mixer) {
            .full_attention => |attention| {
                full += 1;
                try std.testing.expectEqual(@as(usize, 3), i % 4);
                try std.testing.expectEqual(@as(u64, 12288), attention.query_and_gate.dimensions[1]);
            },
            .delta_net => |delta| {
                recurrent += 1;
                try std.testing.expect(i % 4 != 3);
                try std.testing.expectEqual(@as(u64, 10240), delta.convolution.dimensions[1]);
            },
        }
    }
    try std.testing.expectEqual(@as(u32, 16), full);
    try std.testing.expectEqual(@as(u32, 48), recurrent);
}

test "reject changed architecture dimensions, auxiliary count, and schedule" {
    var doc = try fixture();
    defer doc.deinit();
    for ([_][]const u8{
        "qwen35.block_count",          "qwen35.nextn_predict_layers", "qwen35.full_attention_interval",
        "qwen35.attention.key_length", "qwen35.ssm.state_size",
    }) |key| {
        const entry = metadataEntry(&doc, key);
        const saved = entry.value;
        entry.value = .{ .unsigned = 0 };
        try std.testing.expectError(error.UnsupportedConfiguration, bind(std.testing.allocator, &doc));
        entry.value = saved;
    }
    const arch = metadataEntry(&doc, "general.architecture");
    arch.value = .{ .string = "llama" };
    try std.testing.expectError(error.UnsupportedArchitecture, bind(std.testing.allocator, &doc));
}

test "reject malformed metadata, altered RoPE, and semantic override keys" {
    var doc = try fixture();
    defer doc.deinit();
    const context = metadataEntry(&doc, "qwen35.context_length");
    const saved = context.*;
    context.value = .{ .signed = 262144 };
    try std.testing.expectError(error.InvalidMetadata, bind(std.testing.allocator, &doc));
    context.key = "missing";
    try std.testing.expectError(error.MissingMetadata, bind(std.testing.allocator, &doc));
    context.* = saved;
    const sections = metadataEntry(&doc, "qwen35.rope.dimension_sections");
    const values = @constCast(sections.value.array.values.?);
    values[0] = .{ .signed = 12 };
    try std.testing.expectError(error.UnsupportedConfiguration, bind(std.testing.allocator, &doc));
    values[0] = .{ .signed = 11 };
    const old = doc.metadata;
    const extended = try doc.storage.allocator().alloc(gguf.Metadata, old.len + 1);
    @memcpy(extended[0..old.len], old);
    extended[old.len] = .{ .key = "qwen35.rope.scaling.factor", .kind = .float32, .value = .{ .float = 2 } };
    doc.metadata = extended;
    try std.testing.expectError(error.UnsupportedConfiguration, bind(std.testing.allocator, &doc));
}

test "reject missing, mis-shaped, mistyped, and unexpected tensors" {
    var doc = try fixture();
    defer doc.deinit();
    const query = tensorEntry(&doc, "blk.3.attn_q.weight");
    const saved = query.*;
    query.name = "blk.3.typo.weight";
    try std.testing.expectError(error.MissingTensor, bind(std.testing.allocator, &doc));
    query.* = saved;
    query.dimensions = &.{ 5120, 6144 }; // Query-only shape loses the gate.
    try std.testing.expectError(error.InvalidTensorShape, bind(std.testing.allocator, &doc));
    query.* = saved;
    query.encoding_id = 1; // F16 container layout exists, but this profile excludes it.
    try std.testing.expectError(error.UnsupportedTensorEncoding, bind(std.testing.allocator, &doc));
    query.* = saved;
    const norm = tensorEntry(&doc, "blk.0.attn_norm.weight");
    norm.encoding_id = 8;
    try std.testing.expectError(error.UnsupportedTensorEncoding, bind(std.testing.allocator, &doc));
    norm.encoding_id = 0;
    const old = doc.tensors;
    const extended = try doc.storage.allocator().alloc(Tensor, old.len + 1);
    @memcpy(extended[0..old.len], old);
    extended[old.len] = saved;
    extended[old.len].name = "unexpected.weight";
    doc.tensors = extended;
    try std.testing.expectError(error.UnexpectedTensor, bind(std.testing.allocator, &doc));
    extended[old.len].name = saved.name;
    try std.testing.expectError(error.DuplicateTensor, bind(std.testing.allocator, &doc));
}

test "auxiliary prediction tensors are validated even though they are not executed" {
    var doc = try fixture();
    defer doc.deinit();
    const aux = tensorEntry(&doc, "blk.64.nextn.eh_proj.weight");
    aux.dimensions = &.{ 5120, 5120 };
    try std.testing.expectError(error.InvalidTensorShape, bind(std.testing.allocator, &doc));
    aux.name = "missing.auxiliary";
    try std.testing.expectError(error.MissingTensor, bind(std.testing.allocator, &doc));
}

test "the plain Qwen file carries no rotation and one auxiliary layer" {
    var doc = try fixture();
    defer doc.deinit();
    const model = try bind(std.testing.allocator, &doc);
    try std.testing.expect(model.rotation == null);
    try std.testing.expectEqual(@as(u32, 1), model.summary.auxiliary_prediction_layers);
}

test "the prediction head binds its fifteen tensors with the main shapes" {
    var doc = try fixture();
    defer doc.deinit();
    const model = try bind(std.testing.allocator, &doc);
    const draft = model.draft.?;
    try std.testing.expect(draft.eh_proj == tensorEntry(&doc, "blk.64.nextn.eh_proj.weight"));
    try std.testing.expect(draft.enorm == tensorEntry(&doc, "blk.64.nextn.enorm.weight"));
    try std.testing.expect(draft.hnorm == tensorEntry(&doc, "blk.64.nextn.hnorm.weight"));
    try std.testing.expect(draft.shared_head_norm == tensorEntry(&doc, "blk.64.nextn.shared_head_norm.weight"));
    try std.testing.expectEqualSlices(u64, &.{ 10240, 5120 }, draft.eh_proj.dimensions);
    const attention = draft.layer.mixer.full_attention;
    try std.testing.expectEqualSlices(u64, &.{ 5120, 12288 }, attention.query_and_gate.dimensions);
    try std.testing.expectEqualSlices(u64, &.{ 5120, 17408 }, draft.layer.ffn_gate.dimensions);
    // The block's key and value are Q8_0 on the pinned file, an encoding the
    // profile already executes.
    try std.testing.expectEqual(@as(u32, 8), attention.key.encoding_id);
    try std.testing.expectEqual(@as(u32, 8), attention.value.encoding_id);
    try std.testing.expectEqual(@as(u64, 10240), draft.eh_proj.dimensions[0]);
}

test "bind the Bonsai inventory: 64 blocks, ternary and BF16 matrices, the pinned rotation" {
    var doc = try bonsaiInventoryDocument(std.testing.allocator);
    defer doc.deinit();
    const model = try bind(std.testing.allocator, &doc);
    try std.testing.expectEqual(@as(u32, 851), model.summary.text_tensors);
    try std.testing.expectEqual(@as(u32, 0), model.summary.auxiliary_tensors);
    try std.testing.expectEqual(@as(u32, 0), model.summary.auxiliary_prediction_layers);
    try std.testing.expectEqual(@as(u64, 7_195_047_936), model.summary.text_tensor_bytes);
    try std.testing.expect(model.draft == null);
    try std.testing.expectEqual(@as(u32, 142), model.output.encoding_id);
    try std.testing.expectEqual(@as(u32, 30), model.layers()[0].mixer.delta_net.alpha.encoding_id);
    try std.testing.expect(model.summary.rotated_basis != null);
    const rotation = model.rotation.?;
    try std.testing.expectEqual(@as(u32, 1024), rotation.block);
    try std.testing.expect(rotation.value_grouped);
    try std.testing.expectEqual(@as(usize, 17408), rotation.signsFor(17408).?.len);
    try std.testing.expect(rotation.signsFor(10240) == null);
    var negative: usize = 0;
    for (rotation.signs) |signs| for (signs.values) |value| {
        if (value.signed == -1) negative += 1;
    };
    try std.testing.expectEqual(@as(usize, 14504), negative);
}

test "reject rotations outside the pinned contract" {
    var doc = try bonsaiInventoryDocument(std.testing.allocator);
    defer doc.deinit();
    const version = metadataEntry(&doc, "prism.hadamard.version");
    version.value = .{ .unsigned = 2 };
    try std.testing.expectError(error.UnsupportedConfiguration, bind(std.testing.allocator, &doc));
    version.value = .{ .unsigned = 1 };
    // Rotation keys without the version key are a rotation this adapter cannot read.
    version.key = "general.rotation";
    try std.testing.expectError(error.UnsupportedConfiguration, bind(std.testing.allocator, &doc));
    version.key = "prism.hadamard.version";
    const mode = metadataEntry(&doc, "prism.hadamard.sign_mode");
    mode.value = .{ .string = "identity" };
    try std.testing.expectError(error.UnsupportedConfiguration, bind(std.testing.allocator, &doc));
    mode.value = .{ .string = "explicit" };
    const signs = @constCast(metadataEntry(&doc, "prism.hadamard.sign_values").value.array.values.?);
    signs[6000] = .{ .signed = 0 };
    try std.testing.expectError(error.InvalidMetadata, bind(std.testing.allocator, &doc));
    signs[6000] = .{ .signed = 1 };
    const names = @constCast(metadataEntry(&doc, "prism.hadamard.weight_names").value.array.values.?);
    const saved = names[1];
    names[1] = .{ .string = "blk.0.ssm_alpha.weight" }; // never rotated
    try std.testing.expectError(error.UnsupportedConfiguration, bind(std.testing.allocator, &doc));
    names[1] = names[2]; // a slot claimed twice, another left empty
    try std.testing.expectError(error.UnsupportedConfiguration, bind(std.testing.allocator, &doc));
    names[1] = saved;
    metadataEntry(&doc, "prism.hadamard.axis").key = "prism.hadamard.axes";
    try std.testing.expectError(error.UnsupportedConfiguration, bind(std.testing.allocator, &doc));
    metadataEntry(&doc, "prism.hadamard.axes").key = "prism.hadamard.axis";
    _ = try bind(std.testing.allocator, &doc);
    // The block count and the auxiliary declaration must agree.
    metadataEntry(&doc, "qwen35.block_count").value = .{ .unsigned = 65 };
    try std.testing.expectError(error.UnsupportedConfiguration, bind(std.testing.allocator, &doc));
}

test "bind clef-flash's 9B inventory: the shape from the file, no prediction block" {
    var doc = try @import("inventory.zig").document(std.testing.allocator, @embedFile("fixtures/clef-flash-q6k.json"));
    defer doc.deinit();
    const model = try bind(std.testing.allocator, &doc);
    try std.testing.expect(model.shape.eql(qwen35_9b_shape));
    try std.testing.expectEqualStrings("qwen35_9b", model.summary.profile);
    try std.testing.expectEqual(@as(u32, 32), model.summary.decoder_layers);
    try std.testing.expectEqual(@as(u32, 8), model.summary.layer_kinds[0].count);
    try std.testing.expectEqual(@as(u32, 24), model.summary.layer_kinds[1].count);
    try std.testing.expect(model.draft == null and model.rotation == null);
    try std.testing.expectEqual(@as(usize, 32), model.layers().len);
    try std.testing.expectEqualSlices(u64, &.{ 4096, 8192 }, model.layers()[0].mixer.delta_net.qkv.dimensions);
    try std.testing.expectEqualSlices(u64, &.{ 4096, 8192 }, model.layers()[3].mixer.full_attention.query_and_gate.dimensions);
    try std.testing.expectEqual(@as(usize, 8192 * 3), model.shape.deltaHistory());
    // The spelled-out schedule must be the every-4th one.
    const flags = @constCast(metadataEntry(&doc, "qwen35.attention.recurrent_layers").value.array.values.?);
    flags[3] = .{ .boolean = true };
    try std.testing.expectError(error.UnsupportedConfiguration, bind(std.testing.allocator, &doc));
}

test "reject shapes outside the accepted bounds" {
    var doc = try fixture();
    defer doc.deinit();
    for ([_]struct { key: []const u8, value: u64, err: anyerror }{
        .{ .key = "qwen35.attention.head_count", .value = 22, .err = error.UnsupportedConfiguration },
        .{ .key = "qwen35.ssm.time_step_rank", .value = 40, .err = error.UnsupportedConfiguration },
        .{ .key = "qwen35.ssm.inner_size", .value = 4096, .err = error.UnsupportedConfiguration },
        .{ .key = "qwen35.embedding_length", .value = 5000, .err = error.UnsupportedConfiguration },
        .{ .key = "qwen35.block_count", .value = 66, .err = error.UnsupportedConfiguration },
        // A valid shape the tensors do not have.
        .{ .key = "qwen35.embedding_length", .value = 4096, .err = error.InvalidTensorShape },
    }) |case| {
        const entry = metadataEntry(&doc, case.key);
        const saved = entry.value;
        entry.value = .{ .unsigned = case.value };
        try std.testing.expectError(case.err, bind(std.testing.allocator, &doc));
        entry.value = saved;
    }
}

fn bindWithAllocator(alloc: std.mem.Allocator, doc: *const gguf.Document) !void {
    _ = try bind(alloc, doc);
}

test "binding lookup allocations are freed on every allocation failure" {
    var doc = try fixture();
    defer doc.deinit();
    try std.testing.checkAllAllocationFailures(std.testing.allocator, bindWithAllocator, .{&doc});
}
