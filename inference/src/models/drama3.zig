//! Structural adapter for Fish Audio Drama 3 (`drama3`, speech.cpp's GGUF
//! layout): the Qwen3.5-MoE hybrid backbone and the codebook tower. Every
//! dimension is pinned; `bind` checks the metadata against them and binds
//! each tensor by name, borrowing the Document. The codec's `codec.*`
//! tensors are counted, not bound (its own binder reads them). Arrays the
//! parser leaves as descriptors (`layer_types`, `output.token_ids`) are
//! checked against the file's bytes by `verifyArrays`.
//! Not a registry family: it decodes frames, not text. The facts:
//! docs/models/drama3.md.
const std = @import("std");
const alloc_check = @import("../alloc_check.zig");
const gguf = @import("../formats/gguf.zig");
const models = @import("root.zig");
const Tensor = gguf.Tensor;

pub const architecture = "drama3";

pub const hidden = 2048;
pub const layer_count = 40;
pub const vocabulary = 252288;
/// Full attention (backbone and tower): 16 query heads over 2 KV heads of 256.
pub const heads = 16;
pub const kv_heads = 2;
pub const head_width = 256;
pub const query_width = heads * head_width;
pub const kv_width = kv_heads * head_width;
/// Partial RoPE: the first 64 of each head's 256 dims, base 1e7.
pub const rotary_dims = 64;
pub const rope_base: f32 = 1e7;
/// DeltaNet: 16 key heads and 32 value heads of 128.
pub const key_heads = 16;
pub const value_heads = 32;
pub const state_width = 128;
pub const key_width = key_heads * state_width;
pub const inner_width = value_heads * state_width;
pub const qkv_width = 2 * key_width + inner_width;
pub const conv_kernel = 4;
pub const delta_history = qkv_width * (conv_kernel - 1);
pub const delta_matrix = value_heads * state_width * state_width;
/// MoE: 256 experts, 8 routed per row, SwiGLU of 512; a shared expert of 512.
pub const experts = 256;
pub const experts_used = 8;
pub const expert_ffn = 512;
pub const norm_eps: f32 = 1e-6;
/// The tower: 4 shared and 9 residual layers, SwiGLU of 6144, 10 codebooks of 4096.
pub const tower_ffn = 6144;
pub const codebooks = 10;
pub const codebook_size = 4096;
pub const shared_layers = 4;
pub const residual_layers = codebooks - 1;
/// Tokens: the end of speech, the semantic band, and pad.
pub const eos_token = 248046;
pub const semantic_start = 248099;
pub const semantic_end = 252194;
pub const pad_token = 248044;
/// `output`'s rows: EOS, then the semantic band in order (`drama3.output.token_ids`).
pub const output_rows = 1 + semantic_end - semantic_start + 1;
pub const max_prompt_rows = 8192;
pub const max_frames = 2048;

/// The full-attention layer at `i`: every fourth, from 3.
pub fn isFullAttention(i: usize) bool {
    return i % 4 == 3;
}

/// Whether `token` is a semantic code's id.
pub fn isSemantic(token: u32) bool {
    return token >= semantic_start and token <= semantic_end;
}

/// The token id an `output` row produces.
pub fn outputToken(row: usize) u32 {
    return if (row == 0) eos_token else @intCast(semantic_start + row - 1);
}

pub const FullAttention = struct {
    query: *const Tensor,
    gate: *const Tensor,
    key: *const Tensor,
    value: *const Tensor,
    output: *const Tensor,
    query_norm: *const Tensor,
    key_norm: *const Tensor,
};

pub const DeltaNet = struct {
    qkv: *const Tensor,
    z: *const Tensor,
    /// `ssm_in_b`, β before its sigmoid.
    beta: *const Tensor,
    /// `ssm_in_a`, the decay input before `dt_bias` and softplus.
    alpha: *const Tensor,
    convolution: *const Tensor,
    time_bias: *const Tensor,
    a_log: *const Tensor,
    norm: *const Tensor,
    output: *const Tensor,
};

pub const Moe = struct {
    router: *const Tensor,
    gate: *const Tensor,
    up: *const Tensor,
    down: *const Tensor,
    shared_gate: *const Tensor,
    shared_up: *const Tensor,
    shared_down: *const Tensor,
    /// `ffn_gate_inp_shexp` [2048, 1]: the shared expert's sigmoid gate.
    shared_router: *const Tensor,
};

pub const Layer = struct {
    attention_norm: *const Tensor,
    post_attention_norm: *const Tensor,
    moe: Moe,
    mixer: union(enum) { full_attention: FullAttention, delta_net: DeltaNet },
};

/// One `TowerLayer`: a code embedding mixed into the hidden, one pre-norm block.
pub const TowerLayer = struct {
    embedding: *const Tensor,
    enorm: *const Tensor,
    hnorm: *const Tensor,
    eh_proj: *const Tensor,
    final_norm: *const Tensor,
    attention_norm: *const Tensor,
    ffn_norm: *const Tensor,
    attention: FullAttention,
    ffn_gate: *const Tensor,
    ffn_up: *const Tensor,
    ffn_down: *const Tensor,
};

/// All tensor pointers borrow the Document; keep it alive for the binding.
pub const Binding = struct {
    token_embedding: *const Tensor,
    output_norm: *const Tensor,
    /// [2048, 4097]: rows map to ids by `outputToken`.
    output: *const Tensor,
    layers: [layer_count]Layer,
    /// Codebooks 1–9: codebook k, code c at row (k − 1)·4096 + c.
    codebook_embedding: *const Tensor,
    shared: [shared_layers]TowerLayer,
    residual: [residual_layers]TowerLayer,
    /// Head j predicts code j + 1.
    heads: [residual_layers]*const Tensor,
    /// Tensors and bytes bound here, and the `codec.*` tensors left to the codec.
    tensors: u32,
    bytes: u64,
    codec_tensors: u32,
};

pub const Error = models.BindError;

const IntegerSetting = struct { key: []const u8, value: u64 };
const integer_settings = [_]IntegerSetting{
    .{ .key = "drama3.text.hidden_size", .value = hidden },
    .{ .key = "drama3.text.num_hidden_layers", .value = layer_count },
    .{ .key = "drama3.text.num_attention_heads", .value = heads },
    .{ .key = "drama3.text.num_key_value_heads", .value = kv_heads },
    .{ .key = "drama3.text.head_dim", .value = head_width },
    .{ .key = "drama3.text.linear_num_key_heads", .value = key_heads },
    .{ .key = "drama3.text.linear_num_value_heads", .value = value_heads },
    .{ .key = "drama3.text.linear_key_head_dim", .value = state_width },
    .{ .key = "drama3.text.linear_value_head_dim", .value = state_width },
    .{ .key = "drama3.text.linear_conv_kernel_dim", .value = conv_kernel },
    .{ .key = "drama3.text.num_experts", .value = experts },
    .{ .key = "drama3.text.num_experts_per_tok", .value = experts_used },
    .{ .key = "drama3.text.moe_intermediate_size", .value = expert_ffn },
    .{ .key = "drama3.text.shared_expert_intermediate_size", .value = expert_ffn },
    .{ .key = "drama3.text.vocab_size", .value = vocabulary },
    .{ .key = "drama3.text.max_position_embeddings", .value = 262144 },
    .{ .key = "drama3.text.rotary_dim", .value = rotary_dims },
    .{ .key = "drama3.audio_decoder.dim", .value = hidden },
    .{ .key = "drama3.audio_decoder.n_head", .value = heads },
    .{ .key = "drama3.audio_decoder.n_local_heads", .value = kv_heads },
    .{ .key = "drama3.audio_decoder.head_dim", .value = head_width },
    .{ .key = "drama3.audio_decoder.intermediate_size", .value = tower_ffn },
    .{ .key = "drama3.audio_decoder.num_codebooks", .value = codebooks },
    .{ .key = "drama3.audio_decoder.vocab_size", .value = codebook_size },
    .{ .key = "drama3.audio_decoder.shared_enrichment_layers", .value = shared_layers },
    .{ .key = "drama3.audio_decoder.vq_backbone_codebooks", .value = 1 },
    .{ .key = "drama3.audio_decoder.rotary_dim", .value = rotary_dims },
    .{ .key = "drama3.eos_token_id", .value = eos_token },
    .{ .key = "drama3.semantic_start_token_id", .value = semantic_start },
    .{ .key = "drama3.semantic_end_token_id", .value = semantic_end },
    .{ .key = "drama3.pad_token_id", .value = pad_token },
    .{ .key = "general.quantization_version", .value = 2 },
};
const float_settings = [_]struct { key: []const u8, value: f64 }{
    .{ .key = "drama3.text.rms_norm_eps", .value = norm_eps },
    .{ .key = "drama3.text.rope_theta", .value = rope_base },
    .{ .key = "drama3.audio_decoder.norm_eps", .value = norm_eps },
    .{ .key = "drama3.audio_decoder.rope_base", .value = rope_base },
};
/// Keys under the model's prefixes this adapter reads; any other key there
/// could change the equations and is refused.
const array_keys = [_][]const u8{ "drama3.text.layer_types", "drama3.output.token_ids" };
const boolean_keys = [_][]const u8{"drama3.text.norm_topk_prob"};
/// The tokenizer the prompt builder constructs; only its presence is checked here.
const tokenizer_keys = [_][]const u8{ "drama3.tokenizer.tokens", "drama3.tokenizer.merges", "drama3.tokenizer.added_ids", "drama3.tokenizer.pre_tokenizer" };

fn validateMetadata(doc: *const gguf.Document) Error!void {
    const arch = doc.string("general.architecture") orelse return error.MissingMetadata;
    if (!std.mem.eql(u8, arch, architecture)) return error.UnsupportedArchitecture;
    for (integer_settings) |setting| {
        if (try integer(doc, setting.key) != setting.value) return error.UnsupportedConfiguration;
    }
    for (float_settings) |setting| switch (doc.get(setting.key) orelse return error.MissingMetadata) {
        .float => |value| if (value != @as(f64, @as(f32, @floatCast(setting.value)))) return error.UnsupportedConfiguration,
        else => return error.InvalidMetadata,
    };
    switch (doc.get("drama3.text.norm_topk_prob") orelse return error.MissingMetadata) {
        .boolean => |b| if (!b) return error.UnsupportedConfiguration,
        else => return error.InvalidMetadata,
    }
    try arrayShape(doc, "drama3.text.layer_types", .string, layer_count);
    try arrayShape(doc, "drama3.output.token_ids", .int32, output_rows);
    for (tokenizer_keys) |key| if (doc.get(key) == null) return error.MissingMetadata;
    for (doc.metadata) |entry| {
        const governed = std.mem.startsWith(u8, entry.key, "drama3.text.") or
            std.mem.startsWith(u8, entry.key, "drama3.audio_decoder.") or
            std.mem.startsWith(u8, entry.key, "drama3.output.");
        if (governed and !knownKey(entry.key)) return error.UnsupportedConfiguration;
    }
}

fn knownKey(key: []const u8) bool {
    for (integer_settings) |s| if (std.mem.eql(u8, key, s.key)) return true;
    for (float_settings) |s| if (std.mem.eql(u8, key, s.key)) return true;
    for (array_keys ++ boolean_keys) |k| if (std.mem.eql(u8, key, k)) return true;
    return false;
}

/// An integer key of either signedness, as converters differ.
fn integer(doc: *const gguf.Document, key: []const u8) Error!u64 {
    return switch (doc.get(key) orelse return error.MissingMetadata) {
        .unsigned => |n| n,
        .signed => |n| std.math.cast(u64, n) orelse error.InvalidMetadata,
        else => error.InvalidMetadata,
    };
}

fn arrayShape(doc: *const gguf.Document, key: []const u8, kind: gguf.Type, count: u64) Error!void {
    switch (doc.get(key) orelse return error.MissingMetadata) {
        .array => |a| if (a.element_type != kind or a.count != count) return error.UnsupportedConfiguration,
        else => return error.InvalidMetadata,
    }
}

/// Checks the two descriptor-only arrays against the file: `layer_types` is
/// the every-4th schedule and `output.token_ids` is EOS then the band.
/// `directory` is the file from offset 0 (a mapping covers it).
pub fn verifyArrays(doc: *const gguf.Document, directory: []const u8) (Error || gguf.Error)!void {
    const types = doc.get("drama3.text.layer_types") orelse return error.MissingMetadata;
    if (types != .array) return error.InvalidMetadata;
    var reader = try gguf.ArrayReader.init(types.array, directory, .{ .directory_bytes = directory.len });
    var i: usize = 0;
    while (try reader.nextString()) |kind| : (i += 1) {
        const want: []const u8 = if (isFullAttention(i)) "full_attention" else "linear_attention";
        if (!std.mem.eql(u8, kind, want)) return error.UnsupportedConfiguration;
    }
    if (i != layer_count) return error.UnsupportedConfiguration;
    const ids = doc.get("drama3.output.token_ids") orelse return error.MissingMetadata;
    if (ids != .array) return error.InvalidMetadata;
    reader = try gguf.ArrayReader.init(ids.array, directory, .{ .directory_bytes = directory.len });
    var row: usize = 0;
    while (try reader.nextInt32()) |id| : (row += 1) {
        if (row >= output_rows or id != outputToken(row)) return error.UnsupportedConfiguration;
    }
    if (row != output_rows) return error.UnsupportedConfiguration;
}

const Storage = enum { f32, matrix };

/// The matrix encodings the CPU reference executes for this file: F32, F16, Q4_K.
pub fn executableEncoding(encoding_id: u32) bool {
    return switch (encoding_id) {
        0, 1, 12 => true,
        else => false,
    };
}

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

    fn named(self: *Binder, comptime format: []const u8, args: anytype, dimensions: []const u64, storage: Storage) Error!*const Tensor {
        var buffer: [96]u8 = undefined;
        const name = std.fmt.bufPrint(&buffer, format, args) catch unreachable;
        return self.take(name, dimensions, storage);
    }

    fn attention(self: *Binder, comptime prefix: []const u8, index: usize) Error!FullAttention {
        return .{
            .query = try self.named(prefix ++ ".{d}.attn_q", .{index}, &.{ hidden, query_width }, .matrix),
            .gate = try self.named(prefix ++ ".{d}.attn_gate", .{index}, &.{ hidden, query_width }, .matrix),
            .key = try self.named(prefix ++ ".{d}.attn_k", .{index}, &.{ hidden, kv_width }, .matrix),
            .value = try self.named(prefix ++ ".{d}.attn_v", .{index}, &.{ hidden, kv_width }, .matrix),
            .output = try self.named(prefix ++ ".{d}.attn_output", .{index}, &.{ query_width, hidden }, .matrix),
            .query_norm = try self.named(prefix ++ ".{d}.attn_q_norm", .{index}, &.{head_width}, .f32),
            .key_norm = try self.named(prefix ++ ".{d}.attn_k_norm", .{index}, &.{head_width}, .f32),
        };
    }

    fn layer(self: *Binder, i: usize) Error!Layer {
        return .{
            .attention_norm = try self.named("blk.{d}.attn_norm", .{i}, &.{hidden}, .f32),
            .post_attention_norm = try self.named("blk.{d}.post_attn_norm", .{i}, &.{hidden}, .f32),
            .moe = .{
                .router = try self.named("blk.{d}.ffn_gate_inp", .{i}, &.{ hidden, experts }, .f32),
                .gate = try self.named("blk.{d}.ffn_gate_exps", .{i}, &.{ hidden, expert_ffn, experts }, .matrix),
                .up = try self.named("blk.{d}.ffn_up_exps", .{i}, &.{ hidden, expert_ffn, experts }, .matrix),
                .down = try self.named("blk.{d}.ffn_down_exps", .{i}, &.{ expert_ffn, hidden, experts }, .matrix),
                .shared_gate = try self.named("blk.{d}.ffn_gate_shexp", .{i}, &.{ hidden, expert_ffn }, .matrix),
                .shared_up = try self.named("blk.{d}.ffn_up_shexp", .{i}, &.{ hidden, expert_ffn }, .matrix),
                .shared_down = try self.named("blk.{d}.ffn_down_shexp", .{i}, &.{ expert_ffn, hidden }, .matrix),
                .shared_router = try self.named("blk.{d}.ffn_gate_inp_shexp", .{i}, &.{ hidden, 1 }, .f32),
            },
            .mixer = if (isFullAttention(i)) .{ .full_attention = try self.attention("blk", i) } else .{ .delta_net = .{
                .qkv = try self.named("blk.{d}.ssm_in_qkv", .{i}, &.{ hidden, qkv_width }, .matrix),
                .z = try self.named("blk.{d}.ssm_in_z", .{i}, &.{ hidden, inner_width }, .matrix),
                .beta = try self.named("blk.{d}.ssm_in_b", .{i}, &.{ hidden, value_heads }, .f32),
                .alpha = try self.named("blk.{d}.ssm_in_a", .{i}, &.{ hidden, value_heads }, .f32),
                .convolution = try self.named("blk.{d}.ssm_conv1d", .{i}, &.{ conv_kernel, qkv_width }, .f32),
                .time_bias = try self.named("blk.{d}.ssm_dt_bias", .{i}, &.{value_heads}, .f32),
                .a_log = try self.named("blk.{d}.ssm_a_log", .{i}, &.{value_heads}, .f32),
                .norm = try self.named("blk.{d}.ssm_norm", .{i}, &.{state_width}, .f32),
                .output = try self.named("blk.{d}.ssm_out", .{i}, &.{ inner_width, hidden }, .matrix),
            } },
        };
    }

    fn towerLayer(self: *Binder, comptime prefix: []const u8, i: usize) Error!TowerLayer {
        return .{
            .embedding = try self.named(prefix ++ ".{d}.embd", .{i}, &.{ hidden, codebook_size }, .matrix),
            .enorm = try self.named(prefix ++ ".{d}.enorm", .{i}, &.{hidden}, .f32),
            .hnorm = try self.named(prefix ++ ".{d}.hnorm", .{i}, &.{hidden}, .f32),
            .eh_proj = try self.named(prefix ++ ".{d}.eh_proj", .{i}, &.{ 2 * hidden, hidden }, .matrix),
            .final_norm = try self.named(prefix ++ ".{d}.final_norm", .{i}, &.{hidden}, .f32),
            .attention_norm = try self.named(prefix ++ ".{d}.attn_norm", .{i}, &.{hidden}, .f32),
            .ffn_norm = try self.named(prefix ++ ".{d}.ffn_norm", .{i}, &.{hidden}, .f32),
            .attention = try self.attention(prefix, i),
            .ffn_gate = try self.named(prefix ++ ".{d}.ffn_gate", .{i}, &.{ hidden, tower_ffn }, .matrix),
            .ffn_up = try self.named(prefix ++ ".{d}.ffn_up", .{i}, &.{ hidden, tower_ffn }, .matrix),
            .ffn_down = try self.named(prefix ++ ".{d}.ffn_down", .{i}, &.{ tower_ffn, hidden }, .matrix),
        };
    }
};

/// `doc` must come from successful GGUF parsing. The lookup table is freed
/// before return; only the returned tensor references borrow `doc`.
pub fn bind(alloc: std.mem.Allocator, doc: *const gguf.Document) Error!Binding {
    try validateMetadata(doc);
    var binder: Binder = .{ .remaining = .init(alloc) };
    defer binder.remaining.deinit();
    var codec_tensors: u32 = 0;
    for (doc.tensors) |*tensor| {
        const slot = try binder.remaining.getOrPut(tensor.name);
        if (slot.found_existing) return error.DuplicateTensor;
        slot.value_ptr.* = tensor;
        if (std.mem.startsWith(u8, tensor.name, "codec.")) codec_tensors += 1;
    }
    var result: Binding = undefined;
    result.token_embedding = try binder.take("token_embd", &.{ hidden, vocabulary }, .matrix);
    result.output_norm = try binder.take("output_norm", &.{hidden}, .f32);
    result.output = try binder.take("output", &.{ hidden, output_rows }, .matrix);
    for (&result.layers, 0..) |*layer, i| layer.* = try binder.layer(i);
    result.codebook_embedding = try binder.take("ad.codebook_embd", &.{ hidden, residual_layers * codebook_size }, .matrix);
    for (&result.shared, 0..) |*layer, i| layer.* = try binder.towerLayer("ad.shared", i);
    for (&result.residual, 0..) |*layer, i| layer.* = try binder.towerLayer("ad.layer", i);
    for (&result.heads, 0..) |*head, i| head.* = try binder.named("ad.head.{d}", .{i}, &.{ hidden, codebook_size }, .matrix);
    if (binder.remaining.count() != codec_tensors) return error.UnexpectedTensor;
    result.tensors = binder.tensors;
    result.bytes = binder.bytes;
    result.codec_tensors = codec_tensors;
    return result;
}

/// The pinned artifact's directory inventory (`fixtures/drama3-q4_k.json`)
/// hydrated with no weights; test support only.
pub fn inventoryDocument(gpa: std.mem.Allocator) !gguf.Document {
    return @import("inventory.zig").document(gpa, @embedFile("fixtures/drama3-q4_k.json"));
}

fn metadataEntry(doc: *gguf.Document, key: []const u8) *gguf.Metadata {
    for (@constCast(doc.metadata)) |*entry| if (std.mem.eql(u8, entry.key, key)) return entry;
    unreachable;
}

fn tensorEntry(doc: *gguf.Document, name: []const u8) *Tensor {
    for (@constCast(doc.tensors)) |*tensor| if (std.mem.eql(u8, tensor.name, name)) return tensor;
    unreachable;
}

test "the pinned inventory binds: 40 layers, the tower, the codec left over" {
    var doc = try inventoryDocument(std.testing.allocator);
    defer doc.deinit();
    const b = try bind(std.testing.allocator, &doc);
    // 3 + 30·(2 + 8 + 9) + 10·(2 + 8 + 7) + 1 + 13·17 + 9.
    try std.testing.expectEqual(@as(u32, 974), b.tensors);
    try std.testing.expectEqual(@as(u32, 557), b.codec_tensors);
    try std.testing.expectEqual(@as(usize, 1531), doc.tensors.len);
    try std.testing.expect(b.token_embedding == tensorEntry(&doc, "token_embd"));
    for (b.layers, 0..) |layer, i| switch (layer.mixer) {
        .full_attention => |a| {
            try std.testing.expect(isFullAttention(i));
            try std.testing.expectEqual(@as(u64, 4096), a.gate.dimensions[1]);
        },
        .delta_net => |d| {
            try std.testing.expect(!isFullAttention(i));
            try std.testing.expectEqual(@as(u64, 8192), d.convolution.dimensions[1]);
        },
    };
    try std.testing.expect(b.layers[39].moe.down == tensorEntry(&doc, "blk.39.ffn_down_exps"));
    try std.testing.expect(b.residual[8].attention.key_norm == tensorEntry(&doc, "ad.layer.8.attn_k_norm"));
    try std.testing.expect(b.heads[8] == tensorEntry(&doc, "ad.head.8"));
    try std.testing.expectEqual(@as(u32, 252194), outputToken(output_rows - 1));
    try std.testing.expectEqual(@as(u32, eos_token), outputToken(0));
}

test "reject changed geometry, eps, routing, and unknown model keys" {
    var doc = try inventoryDocument(std.testing.allocator);
    defer doc.deinit();
    for ([_][]const u8{ "drama3.text.num_experts", "drama3.audio_decoder.num_codebooks", "drama3.semantic_start_token_id", "drama3.text.rotary_dim" }) |key| {
        const entry = metadataEntry(&doc, key);
        const saved = entry.value;
        entry.value = .{ .unsigned = 7 };
        try std.testing.expectError(error.UnsupportedConfiguration, bind(std.testing.allocator, &doc));
        entry.value = .{ .string = "7" };
        try std.testing.expectError(error.InvalidMetadata, bind(std.testing.allocator, &doc));
        entry.value = saved;
    }
    const eps = metadataEntry(&doc, "drama3.audio_decoder.norm_eps");
    eps.value = .{ .float = 1e-5 };
    try std.testing.expectError(error.UnsupportedConfiguration, bind(std.testing.allocator, &doc));
    eps.value = .{ .float = @as(f32, 1e-6) };
    const topk = metadataEntry(&doc, "drama3.text.norm_topk_prob");
    topk.value = .{ .boolean = false };
    try std.testing.expectError(error.UnsupportedConfiguration, bind(std.testing.allocator, &doc));
    topk.value = .{ .boolean = true };
    const ids = metadataEntry(&doc, "drama3.output.token_ids");
    ids.value.array.count = 4096;
    try std.testing.expectError(error.UnsupportedConfiguration, bind(std.testing.allocator, &doc));
    ids.value.array.count = output_rows;
    const tokens = metadataEntry(&doc, "drama3.tokenizer.tokens");
    tokens.key = "drama3.tokenizer.vocabulary";
    try std.testing.expectError(error.MissingMetadata, bind(std.testing.allocator, &doc));
    tokens.key = "drama3.tokenizer.tokens";
    const arch = metadataEntry(&doc, "general.architecture");
    arch.value = .{ .string = "qwen35" };
    try std.testing.expectError(error.UnsupportedArchitecture, bind(std.testing.allocator, &doc));
    arch.value = .{ .string = architecture };
    const old = doc.metadata;
    const extended = try doc.storage.allocator().alloc(gguf.Metadata, old.len + 1);
    @memcpy(extended[0..old.len], old);
    extended[old.len] = .{ .key = "drama3.text.attn_output_gate", .kind = .boolean, .value = .{ .boolean = false } };
    doc.metadata = extended;
    try std.testing.expectError(error.UnsupportedConfiguration, bind(std.testing.allocator, &doc));
}

test "reject missing, mis-shaped, mistyped, and unexpected tensors" {
    var doc = try inventoryDocument(std.testing.allocator);
    defer doc.deinit();
    const gate = tensorEntry(&doc, "blk.3.attn_gate");
    const saved = gate.*;
    gate.name = "blk.3.attn_gates";
    try std.testing.expectError(error.MissingTensor, bind(std.testing.allocator, &doc));
    gate.* = saved;
    gate.dimensions = &.{ 2048, 8192 }; // Fish's packed query and gate.
    try std.testing.expectError(error.InvalidTensorShape, bind(std.testing.allocator, &doc));
    gate.* = saved;
    gate.encoding_id = 2; // Q4_0 is a layout, not this file's executed set.
    try std.testing.expectError(error.UnsupportedTensorEncoding, bind(std.testing.allocator, &doc));
    gate.* = saved;
    const norm = tensorEntry(&doc, "ad.shared.0.enorm");
    norm.encoding_id = 1;
    try std.testing.expectError(error.UnsupportedTensorEncoding, bind(std.testing.allocator, &doc));
    norm.encoding_id = 0;
    const head = tensorEntry(&doc, "ad.head.8");
    head.dimensions = &.{ 2048, 1024 };
    try std.testing.expectError(error.InvalidTensorShape, bind(std.testing.allocator, &doc));
    head.dimensions = &.{ 2048, 4096 };
    const old = doc.tensors;
    const extended = try doc.storage.allocator().alloc(Tensor, old.len + 1);
    @memcpy(extended[0..old.len], old);
    extended[old.len] = saved;
    extended[old.len].name = "ad.head.9";
    doc.tensors = extended;
    try std.testing.expectError(error.UnexpectedTensor, bind(std.testing.allocator, &doc));
    extended[old.len].name = "blk.3.attn_gate";
    try std.testing.expectError(error.DuplicateTensor, bind(std.testing.allocator, &doc));
}

fn bindWithAllocator(alloc: std.mem.Allocator, doc: *const gguf.Document) !void {
    _ = try bind(alloc, doc);
}

test "binding lookup allocations are freed on every allocation failure" {
    var doc = try inventoryDocument(std.testing.allocator);
    defer doc.deinit();
    try alloc_check.checkAll(std.testing.allocator, bindWithAllocator, .{&doc});
}

test "the descriptor arrays are checked against the file's bytes" {
    // A directory holding the two arrays as the parser would leave them.
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(std.testing.allocator);
    const a = std.testing.allocator;
    const types_offset = bytes.items.len;
    for (0..layer_count) |i| {
        const kind: []const u8 = if (isFullAttention(i)) "full_attention" else "linear_attention";
        var len: [8]u8 = undefined;
        std.mem.writeInt(u64, &len, kind.len, .little);
        try bytes.appendSlice(a, &len);
        try bytes.appendSlice(a, kind);
    }
    const ids_offset = bytes.items.len;
    for (0..output_rows) |row| {
        var id: [4]u8 = undefined;
        std.mem.writeInt(i32, &id, @intCast(outputToken(row)), .little);
        try bytes.appendSlice(a, &id);
    }
    var metadata = [_]gguf.Metadata{
        .{ .key = "drama3.text.layer_types", .kind = .array, .value = .{ .array = .{ .element_type = .string, .count = layer_count, .file_offset = types_offset } } },
        .{ .key = "drama3.output.token_ids", .kind = .array, .value = .{ .array = .{ .element_type = .int32, .count = output_rows, .file_offset = ids_offset } } },
    };
    const doc: gguf.Document = .{ .storage = .init(a), .version = 3, .file_bytes = 0, .directory_bytes = 0, .data_offset = 0, .alignment = 32, .metadata = &metadata, .tensors = &.{} };
    try verifyArrays(&doc, bytes.items);
    // The last semantic id out of place.
    bytes.items[bytes.items.len - 1] ^= 1;
    try std.testing.expectError(error.UnsupportedConfiguration, verifyArrays(&doc, bytes.items));
    bytes.items[bytes.items.len - 1] ^= 1;
    // Layer 3 declared linear.
    bytes.items[types_offset + 3 * 24 + 8] = 'l';
    try std.testing.expectError(error.UnsupportedConfiguration, verifyArrays(&doc, bytes.items));
}
