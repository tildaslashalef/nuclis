//! EmbeddingGemma 2 text adapter: metadata validation and named weight
//! binding for the pinned `gemma-embedding2` checkpoint, a 24-block Gemma 4–
//! style encoder with bidirectional attention whose output is one pooled,
//! unit vector (docs/models/embeddinggemma.md). Narrow like `gemma4.zig`: a
//! `gemma-embedding2.*` key or value other than the pinned file's is
//! `UnsupportedConfiguration`. Not a registry family: it decodes nothing.
//!
//! Matrices bind only as F32, BF16 or Q8_0. The card warns that
//! activations overflow f16, and every kernel that runs these three keeps
//! activations in F32. `embeddinggemma_runtime.zig` executes this binding.
const std = @import("std");
const gguf = @import("../formats/gguf.zig");
const models = @import("root.zig");
const Tensor = gguf.Tensor;

pub const architecture = "gemma-embedding2";
pub const vocabulary = 262144;
pub const rms_epsilon: f32 = 1e-6;
/// The input limit every modality shares (Google's card); the header's
/// `context_length` is the base model's, not this one's.
pub const max_tokens = 8192;
pub const layer_count = 24;

/// The pinned checkpoint's shape; `validateMetadata` checks every key against it.
pub const Config = struct {
    embedding: usize = 512,
    embedding_out: usize = 768,
    feed_forward: usize = 2048,
    heads: usize = 4,
    per_layer_input: usize = 512,
    /// Key `j` is visible from query `i` on a sliding layer when
    /// |i − j| ≤ this: half the header's symmetric `sliding_window`.
    half_window: usize = 512,
};
pub const config: Config = .{};

pub const Kind = enum {
    sliding,
    global,

    pub fn headSize(self: Kind) usize {
        return switch (self) {
            .sliding => 256,
            .global => 512,
        };
    }
    pub fn kvHeads(self: Kind) usize {
        return switch (self) {
            .sliding => 2,
            .global => 1,
        };
    }
    pub fn ropeBase(self: Kind) f32 {
        return switch (self) {
            .sliding => 10_000,
            .global => 1_000_000,
        };
    }
};

/// F32, Q8_0 and BF16: the encodings whose kernels keep activations in F32.
pub fn executableEncoding(encoding_id: u32) bool {
    return switch (encoding_id) {
        0, 8, 30 => true,
        else => false,
    };
}

pub const Layer = struct {
    kind: Kind,
    attention_norm: *const Tensor,
    post_attention_norm: *const Tensor,
    ffn_norm: *const Tensor,
    post_ffn_norm: *const Tensor,
    query: *const Tensor,
    key: *const Tensor,
    value: *const Tensor,
    output: *const Tensor,
    query_norm: *const Tensor,
    key_norm: *const Tensor,
    ffn_gate: *const Tensor,
    ffn_up: *const Tensor,
    ffn_down: *const Tensor,
    /// The per-layer input gate [d → per_layer_input] and its projection back.
    per_layer_gate: *const Tensor,
    per_layer_projection: *const Tensor,
    per_layer_post_norm: *const Tensor,
    /// One F32 scalar multiplying the layer's residual output.
    output_scale: *const Tensor,

    pub fn queryWidth(self: Layer) usize {
        return config.heads * self.kind.headSize();
    }
    pub fn kvWidth(self: Layer) usize {
        return self.kind.kvHeads() * self.kind.headSize();
    }
};

/// All tensor pointers borrow doc.tensors; keep the Document alive for the
/// binding's lifetime. No weights are read.
pub const Binding = struct {
    token_embedding: *const Tensor,
    /// [d, layers · per_layer_input]: rows `l · w …` project x0 onto layer l's input.
    per_layer_projection: *const Tensor,
    per_layer_norm: *const Tensor,
    output_norm: *const Tensor,
    /// [d → embedding_out], no bias.
    output: *const Tensor,
    layers: [layer_count]Layer,
    text_tensors: u32,
    text_tensor_bytes: u64,
};

pub const Error = models.BindError;

const keys = struct {
    const prefix = architecture ++ ".";
    const block_count = prefix ++ "block_count";
    const integers = [_]struct { []const u8, u64 }{
        .{ prefix ++ "block_count", layer_count },
        .{ prefix ++ "embedding_length", 512 },
        .{ prefix ++ "embedding_length_out", 768 },
        .{ prefix ++ "feed_forward_length", 2048 },
        .{ prefix ++ "attention.head_count", 4 },
        .{ prefix ++ "attention.key_length", 512 },
        .{ prefix ++ "attention.value_length", 512 },
        .{ prefix ++ "attention.key_length_swa", 256 },
        .{ prefix ++ "attention.value_length_swa", 256 },
        .{ prefix ++ "attention.sliding_window", 1024 },
        .{ prefix ++ "attention.shared_kv_layers", 0 },
        .{ prefix ++ "embedding_length_per_layer_input", 512 },
        .{ prefix ++ "rope.dimension_count", 512 },
        .{ prefix ++ "rope.dimension_count_swa", 256 },
        .{ prefix ++ "pooling_type", 1 },
    };
    const floats = [_]struct { []const u8, f32 }{
        .{ prefix ++ "rope.freq_base", 1_000_000 },
        .{ prefix ++ "rope.freq_base_swa", 10_000 },
        .{ prefix ++ "attention.layer_norm_rms_epsilon", rms_epsilon },
    };
    const causal = prefix ++ "attention.causal";
    const kv_heads = prefix ++ "attention.head_count_kv";
    const pattern = prefix ++ "attention.sliding_window_pattern";
    /// Carried by the file but not read: the base model's position limit.
    const context_length = prefix ++ "context_length";

    fn known(key: []const u8) bool {
        for (integers) |entry| if (std.mem.eql(u8, key, entry[0])) return true;
        for (floats) |entry| if (std.mem.eql(u8, key, entry[0])) return true;
        for ([_][]const u8{ causal, kv_heads, pattern, context_length }) |k| if (std.mem.eql(u8, key, k)) return true;
        return false;
    }
};

fn integer(doc: *const gguf.Document, key: []const u8) Error!u64 {
    return switch (doc.get(key) orelse return error.MissingMetadata) {
        .unsigned => |n| n,
        .signed => |n| if (n < 0) error.InvalidMetadata else @intCast(n),
        else => error.InvalidMetadata,
    };
}

fn retained(doc: *const gguf.Document, key: []const u8) Error![]const gguf.Value {
    const array = switch (doc.get(key) orelse return error.MissingMetadata) {
        .array => |a| a,
        else => return error.InvalidMetadata,
    };
    if (array.count != layer_count) return error.UnsupportedConfiguration;
    const values = array.values orelse return error.InvalidMetadata;
    if (values.len != layer_count) return error.InvalidMetadata;
    return values;
}

/// The layer kinds the file declares, each checked against its KV heads.
fn validateMetadata(doc: *const gguf.Document) Error![layer_count]Kind {
    const arch = doc.string("general.architecture") orelse return error.MissingMetadata;
    if (!std.mem.eql(u8, arch, architecture)) return error.UnsupportedArchitecture;
    for (keys.integers) |entry| if (try integer(doc, entry[0]) != entry[1]) return error.UnsupportedConfiguration;
    for (keys.floats) |entry| switch (doc.get(entry[0]) orelse return error.MissingMetadata) {
        .float => |value| if (@as(f32, @floatCast(value)) != entry[1]) return error.UnsupportedConfiguration,
        else => return error.InvalidMetadata,
    };
    // An encoder only: a causal file would be another model.
    switch (doc.get(keys.causal) orelse return error.MissingMetadata) {
        .boolean => |causal| if (causal) return error.UnsupportedConfiguration,
        else => return error.InvalidMetadata,
    }
    var kinds: [layer_count]Kind = undefined;
    for (try retained(doc, keys.pattern), try retained(doc, keys.kv_heads), &kinds) |sliding, heads, *kind| {
        kind.* = switch (sliding) {
            .boolean => |b| if (b) .sliding else .global,
            else => return error.InvalidMetadata,
        };
        const n: u64 = switch (heads) {
            .signed => |v| if (v < 0) return error.InvalidMetadata else @intCast(v),
            .unsigned => |v| v,
            else => return error.InvalidMetadata,
        };
        if (n != kind.kvHeads()) return error.UnsupportedConfiguration;
    }
    const tokens = switch (doc.get("tokenizer.ggml.tokens") orelse return error.MissingMetadata) {
        .array => |a| a,
        else => return error.InvalidMetadata,
    };
    if (tokens.element_type != .string or tokens.count != vocabulary) return error.UnsupportedConfiguration;
    for (doc.metadata) |entry| {
        if (std.mem.startsWith(u8, entry.key, keys.prefix) and !keys.known(entry.key)) return error.UnsupportedConfiguration;
    }
    return kinds;
}

const Storage = enum { f32, matrix };
const Binder = struct {
    remaining: std.StringHashMap(*const Tensor),
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
        var name: [64]u8 = undefined;
        const key = std.fmt.bufPrint(&name, "blk.{d}.{s}", .{ index, suffix }) catch unreachable;
        return self.take(key, dimensions, storage);
    }
    fn layer(self: *Binder, index: usize, kind: Kind) Error!Layer {
        const d = config.embedding;
        const ff = config.feed_forward;
        const w = config.per_layer_input;
        const hd = kind.headSize();
        const q = config.heads * hd;
        const kv = kind.kvHeads() * hd;
        return .{
            .kind = kind,
            .attention_norm = try self.weight(index, "attn_norm.weight", &.{d}, .f32),
            .post_attention_norm = try self.weight(index, "post_attention_norm.weight", &.{d}, .f32),
            .ffn_norm = try self.weight(index, "ffn_norm.weight", &.{d}, .f32),
            .post_ffn_norm = try self.weight(index, "post_ffw_norm.weight", &.{d}, .f32),
            .query = try self.weight(index, "attn_q.weight", &.{ d, q }, .matrix),
            .key = try self.weight(index, "attn_k.weight", &.{ d, kv }, .matrix),
            .value = try self.weight(index, "attn_v.weight", &.{ d, kv }, .matrix),
            .output = try self.weight(index, "attn_output.weight", &.{ q, d }, .matrix),
            .query_norm = try self.weight(index, "attn_q_norm.weight", &.{hd}, .f32),
            .key_norm = try self.weight(index, "attn_k_norm.weight", &.{hd}, .f32),
            .ffn_gate = try self.weight(index, "ffn_gate.weight", &.{ d, ff }, .matrix),
            .ffn_up = try self.weight(index, "ffn_up.weight", &.{ d, ff }, .matrix),
            .ffn_down = try self.weight(index, "ffn_down.weight", &.{ ff, d }, .matrix),
            .per_layer_gate = try self.weight(index, "inp_gate.weight", &.{ d, w }, .matrix),
            .per_layer_projection = try self.weight(index, "proj.weight", &.{ w, d }, .matrix),
            .per_layer_post_norm = try self.weight(index, "post_norm.weight", &.{d}, .f32),
            .output_scale = try self.weight(index, "layer_output_scale.weight", &.{1}, .f32),
        };
    }
};

/// doc must come from successful GGUF parsing. Allocations are temporary
/// lookup storage, freed before return. Every tensor in the file must be
/// claimed; an extra one is `UnexpectedTensor`.
pub fn bind(alloc: std.mem.Allocator, doc: *const gguf.Document) Error!Binding {
    const kinds = try validateMetadata(doc);
    var binder: Binder = .{ .remaining = .init(alloc) };
    defer binder.remaining.deinit();
    for (doc.tensors) |*tensor| {
        if (try binder.remaining.fetchPut(tensor.name, tensor) != null) return error.DuplicateTensor;
    }
    const d = config.embedding;
    var result: Binding = undefined;
    result.token_embedding = try binder.take("token_embd.weight", &.{ d, vocabulary }, .matrix);
    result.per_layer_projection = try binder.take("per_layer_model_proj.weight", &.{ d, layer_count * config.per_layer_input }, .matrix);
    result.per_layer_norm = try binder.take("per_layer_proj_norm.weight", &.{config.per_layer_input}, .f32);
    result.output_norm = try binder.take("output_norm.weight", &.{d}, .f32);
    result.output = try binder.take("output.weight", &.{ d, config.embedding_out }, .matrix);
    for (&result.layers, kinds, 0..) |*layer, kind, index| layer.* = try binder.layer(index, kind);
    if (binder.remaining.count() != 0) return error.UnexpectedTensor;
    result.text_tensors = binder.tensors;
    result.text_tensor_bytes = binder.bytes;
    return result;
}

/// The pinned Q8_0 artifact's directory inventory
/// (`fixtures/embeddinggemma-2-q8_0.json`) hydrated to a Document with no weights.
pub fn inventoryDocument(gpa: std.mem.Allocator) !gguf.Document {
    return @import("inventory.zig").document(gpa, @embedFile("fixtures/embeddinggemma-2-q8_0.json"));
}

test "the pinned Q8_0 inventory binds: 413 tensors, global layers 5, 11, 17, 23" {
    var doc = try inventoryDocument(std.testing.allocator);
    defer doc.deinit();
    const binding = try bind(std.testing.allocator, &doc);
    try std.testing.expectEqual(@as(u32, 413), binding.text_tensors);
    for (binding.layers, 0..) |layer, i| {
        try std.testing.expectEqual(if (i % 6 == 5) Kind.global else Kind.sliding, layer.kind);
        try std.testing.expectEqual(@as(u32, 8), layer.query.encoding_id);
    }
    try std.testing.expectEqual(@as(usize, 2048), binding.layers[5].queryWidth());
    try std.testing.expectEqual(@as(usize, 512), binding.layers[5].kvWidth());
    try std.testing.expectEqual(@as(usize, 512), binding.layers[0].kvWidth());
    // BF16, the one matrix the Q8_0 file keeps unquantized.
    try std.testing.expectEqual(@as(u32, 30), binding.per_layer_projection.encoding_id);
}

fn metadataEntry(doc: *gguf.Document, key: []const u8) *gguf.Metadata {
    for (@constCast(doc.metadata)) |*entry| if (std.mem.eql(u8, entry.key, key)) return entry;
    unreachable;
}

test "a causal file, a K-quant matrix, an unknown key, and another architecture are refused" {
    const gpa = std.testing.allocator;
    {
        var doc = try inventoryDocument(gpa);
        defer doc.deinit();
        metadataEntry(&doc, keys.causal).value = .{ .boolean = true };
        try std.testing.expectError(error.UnsupportedConfiguration, bind(gpa, &doc));
    }
    {
        var doc = try inventoryDocument(gpa);
        defer doc.deinit();
        for (@constCast(doc.tensors)) |*t| if (std.mem.eql(u8, t.name, "blk.3.ffn_up.weight")) {
            t.encoding_id = 12; // Q4_K, whose prefill tiles hold activations as half
        };
        try std.testing.expectError(error.UnsupportedTensorEncoding, bind(gpa, &doc));
    }
    {
        var doc = try inventoryDocument(gpa);
        defer doc.deinit();
        metadataEntry(&doc, keys.context_length).key = keys.prefix ++ "final_logit_softcapping";
        try std.testing.expectError(error.UnsupportedConfiguration, bind(gpa, &doc));
    }
    {
        var doc = try inventoryDocument(gpa);
        defer doc.deinit();
        metadataEntry(&doc, "general.architecture").value = .{ .string = "gemma4" };
        try std.testing.expectError(error.UnsupportedArchitecture, bind(gpa, &doc));
    }
}
