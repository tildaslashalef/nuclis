//! Structural adapter for OmniVoice / VoiceTut-TTS in `audio.cpp`'s single-GGUF
//! packed container (`general.architecture = "audiocpp"`,
//! `audiocpp.model_spec.family = "omnivoice"`).
//!
//! Binds the 313 generator tensors (`weights/*`), validates the container
//! metadata and embedded sidecar descriptors, and verifies the 527 codec
//! tensors (`audio_tokenizer_weights/*`, aliased as `_audiocpp.N` when > 64 B)
//! left to the Higgs Audio V2 codec binder. Not a chat registry family: it
//! generates 8-codebook audio frames via masked diffusion.
//! Facts: docs/models/omnivoice.md, docs/benchmarks/record/omnivoice.md.
const std = @import("std");
const alloc_check = @import("../alloc_check.zig");
const gguf = @import("../formats/gguf.zig");
const models = @import("root.zig");
const Tensor = gguf.Tensor;
const Allocator = std.mem.Allocator;

pub const family_name = "omnivoice";
pub const container_architecture = "audiocpp";

pub const num_audio_codebooks = 8;
pub const audio_vocab_size = 1025;
pub const audio_mask_id = 1024;
pub const audio_table_rows = num_audio_codebooks * audio_vocab_size;
pub const audio_codebook_weights = [num_audio_codebooks]u32{ 8, 8, 6, 6, 4, 4, 2, 2 };
pub const expected_codebook_offsets = [num_audio_codebooks]i64{ 0, 1025, 2050, 3075, 4100, 5125, 6150, 7175 };

pub const hidden_size = 1024;
pub const intermediate_size = 3072;
pub const layer_count = 28;
pub const num_attention_heads = 16;
pub const num_key_value_heads = 8;
pub const head_dim = 128;
pub const query_width = num_attention_heads * head_dim;
pub const kv_width = num_key_value_heads * head_dim;
pub const rms_norm_eps: f32 = 1e-6;
pub const rope_theta: f32 = 1_000_000.0;
pub const vocab_size = 151676;

pub const generator_tensor_count = 5 + layer_count * 11; // 313
pub const codec_tensor_count = 527;
pub const total_tensor_count = generator_tensor_count + codec_tensor_count; // 840

pub const Layer = struct {
    input_layernorm: *const Tensor,
    post_attention_layernorm: *const Tensor,
    q_proj: *const Tensor,
    k_proj: *const Tensor,
    v_proj: *const Tensor,
    o_proj: *const Tensor,
    q_norm: *const Tensor,
    k_norm: *const Tensor,
    gate_proj: *const Tensor,
    up_proj: *const Tensor,
    down_proj: *const Tensor,
};

/// All tensor pointers borrow the Document; keep it alive for the binding.
pub const Binding = struct {
    audio_embeddings: *const Tensor,
    audio_heads: *const Tensor,
    codebook_layer_offsets: *const Tensor,
    token_embedding: *const Tensor,
    output_norm: *const Tensor,
    layers: [layer_count]Layer,
    tensors: u32,
    bytes: u64,
    codec_tensors: u32,
};

/// Borrowed slices of the embedded JSON sidecars inside a GGUF directory image.
pub const EmbeddedFiles = struct {
    config_json: []const u8,
    tokenizer_config_json: []const u8,
    tokenizer_json: []const u8,
    audio_tokenizer_config_json: []const u8,
    audio_tokenizer_preprocessor_json: []const u8,

    /// Extracts the required embedded sidecars without copying from `directory`
    /// (starting at file offset zero).
    pub fn extract(doc: *const gguf.Document, directory: []const u8) (gguf.Error || Error)!EmbeddedFiles {
        const names_arr = try requireArray(doc, "audiocpp.embedded_files.names", .string);
        const offsets_arr = try requireArray(doc, "audiocpp.embedded_files.offsets", .uint64);
        const data_arr = try requireArray(doc, "audiocpp.embedded_files.data", .uint8);
        if (names_arr.count == 0 or names_arr.count > 64 or offsets_arr.count != names_arr.count + 1)
            return error.UnsupportedConfiguration;

        const blob = try gguf.ArrayReader.byteSlice(data_arr, directory, .{});
        var name_reader = try gguf.ArrayReader.init(names_arr, directory, .{});
        var offset_reader = try gguf.ArrayReader.init(offsets_arr, directory, .{});

        var prev = (try offset_reader.nextUint64()) orelse return error.UnsupportedConfiguration;
        if (prev != 0) return error.UnsupportedConfiguration;

        var config_json: ?[]const u8 = null;
        var tokenizer_config_json: ?[]const u8 = null;
        var tokenizer_json: ?[]const u8 = null;
        var audio_tokenizer_config_json: ?[]const u8 = null;
        var audio_tokenizer_preprocessor_json: ?[]const u8 = null;

        for (0..@intCast(names_arr.count)) |_| {
            const name = (try name_reader.nextString()) orelse return error.UnsupportedConfiguration;
            const end = (try offset_reader.nextUint64()) orelse return error.UnsupportedConfiguration;
            if (end < prev or end > blob.len) return error.UnsupportedConfiguration;
            const slice = blob[@intCast(prev)..@intCast(end)];
            prev = end;

            if (std.mem.eql(u8, name, "config.json")) {
                if (config_json != null) return error.UnsupportedConfiguration;
                config_json = slice;
            } else if (std.mem.eql(u8, name, "tokenizer_config.json")) {
                if (tokenizer_config_json != null) return error.UnsupportedConfiguration;
                tokenizer_config_json = slice;
            } else if (std.mem.eql(u8, name, "tokenizer.json")) {
                if (tokenizer_json != null) return error.UnsupportedConfiguration;
                tokenizer_json = slice;
            } else if (std.mem.eql(u8, name, "audio_tokenizer/config.json")) {
                if (audio_tokenizer_config_json != null) return error.UnsupportedConfiguration;
                audio_tokenizer_config_json = slice;
            } else if (std.mem.eql(u8, name, "audio_tokenizer/preprocessor_config.json")) {
                if (audio_tokenizer_preprocessor_json != null) return error.UnsupportedConfiguration;
                audio_tokenizer_preprocessor_json = slice;
            }
        }
        if (prev != blob.len) return error.UnsupportedConfiguration;

        return .{
            .config_json = config_json orelse return error.MissingMetadata,
            .tokenizer_config_json = tokenizer_config_json orelse return error.MissingMetadata,
            .tokenizer_json = tokenizer_json orelse return error.MissingMetadata,
            .audio_tokenizer_config_json = audio_tokenizer_config_json orelse return error.MissingMetadata,
            .audio_tokenizer_preprocessor_json = audio_tokenizer_preprocessor_json orelse return error.MissingMetadata,
        };
    }
};

pub const Error = models.BindError;

/// Matrix / embedding encodings accepted by the OmniVoice generator.
pub fn executableWeightEncoding(id: u32) bool {
    return switch (id) {
        0, 1, 8, 30 => true, // F32, F16, Q8_0, BF16
        else => false,
    };
}

/// Validates container metadata and binds all 313 generator tensors, counting
/// the 527 codec tensors left to the audio tokenizer binder.
pub fn bind(gpa: Allocator, doc: *const gguf.Document) Error!Binding {
    if (!std.mem.eql(u8, doc.string("general.architecture") orelse return error.MissingMetadata, container_architecture))
        return error.UnsupportedArchitecture;
    if (!std.mem.eql(u8, doc.string("audiocpp.model_spec.family") orelse return error.MissingMetadata, family_name))
        return error.UnsupportedArchitecture;
    const spec_version = switch (doc.get("audiocpp.model_spec.version") orelse return error.MissingMetadata) {
        .unsigned => |n| n,
        else => return error.InvalidMetadata,
    };
    if (spec_version != 1) return error.UnsupportedConfiguration;

    const names_arr = try requireArray(doc, "audiocpp.embedded_files.names", .string);
    const offsets_arr = try requireArray(doc, "audiocpp.embedded_files.offsets", .uint64);
    const data_arr = try requireArray(doc, "audiocpp.embedded_files.data", .uint8);
    if (names_arr.count < 5 or offsets_arr.count != names_arr.count + 1 or data_arr.count == 0)
        return error.UnsupportedConfiguration;
    if (offsets_arr.values) |vals| {
        if (vals.len != offsets_arr.count) return error.UnsupportedConfiguration;
        var prev: u64 = 0;
        for (vals, 0..) |v, i| {
            const off = switch (v) {
                .unsigned => |n| n,
                else => return error.InvalidMetadata,
            };
            if (i == 0 and off != 0) return error.UnsupportedConfiguration;
            if (off < prev or off > data_arr.count) return error.UnsupportedConfiguration;
            prev = off;
        }
        if (prev != data_arr.count) return error.UnsupportedConfiguration;
    }

    const tensor_names = try requireArray(doc, "audiocpp.tensor_names", .string);
    if (tensor_names.count != doc.tensors.len) return error.UnsupportedConfiguration;

    var lookup: std.StringHashMapUnmanaged(*const Tensor) = .empty;
    defer lookup.deinit(gpa);
    try lookup.ensureTotalCapacity(gpa, @intCast(doc.tensors.len));
    for (doc.tensors) |*t| {
        const gop = lookup.getOrPutAssumeCapacity(t.name);
        if (gop.found_existing) return error.DuplicateTensor;
        gop.value_ptr.* = t;
    }

    var consumed: std.StringHashMapUnmanaged(void) = .empty;
    defer consumed.deinit(gpa);
    try consumed.ensureTotalCapacity(gpa, generator_tensor_count);

    var total_bytes: u64 = 0;
    const audio_embeddings = try takeWeight(&lookup, &consumed, "weights/audio_embeddings.weight", &.{ hidden_size, audio_table_rows }, &total_bytes);
    const audio_heads = try takeWeight(&lookup, &consumed, "weights/audio_heads.weight", &.{ hidden_size, audio_table_rows }, &total_bytes);
    const codebook_layer_offsets = try takeExact(&lookup, &consumed, "weights/codebook_layer_offsets", &.{num_audio_codebooks}, 27, &total_bytes);
    const token_embedding = try takeWeight(&lookup, &consumed, "weights/llm.embed_tokens.weight", &.{ hidden_size, vocab_size }, &total_bytes);
    const output_norm = try takeExact(&lookup, &consumed, "weights/llm.norm.weight", &.{hidden_size}, 0, &total_bytes);

    var layers: [layer_count]Layer = undefined;
    var name_buf: [96]u8 = undefined;
    for (&layers, 0..) |*layer, i| {
        layer.* = .{
            .input_layernorm = try takeExact(&lookup, &consumed, layerName(&name_buf, i, "input_layernorm.weight"), &.{hidden_size}, 0, &total_bytes),
            .post_attention_layernorm = try takeExact(&lookup, &consumed, layerName(&name_buf, i, "post_attention_layernorm.weight"), &.{hidden_size}, 0, &total_bytes),
            .q_proj = try takeWeight(&lookup, &consumed, layerName(&name_buf, i, "self_attn.q_proj.weight"), &.{ hidden_size, query_width }, &total_bytes),
            .k_proj = try takeWeight(&lookup, &consumed, layerName(&name_buf, i, "self_attn.k_proj.weight"), &.{ hidden_size, kv_width }, &total_bytes),
            .v_proj = try takeWeight(&lookup, &consumed, layerName(&name_buf, i, "self_attn.v_proj.weight"), &.{ hidden_size, kv_width }, &total_bytes),
            .o_proj = try takeWeight(&lookup, &consumed, layerName(&name_buf, i, "self_attn.o_proj.weight"), &.{ query_width, hidden_size }, &total_bytes),
            .q_norm = try takeExact(&lookup, &consumed, layerName(&name_buf, i, "self_attn.q_norm.weight"), &.{head_dim}, 0, &total_bytes),
            .k_norm = try takeExact(&lookup, &consumed, layerName(&name_buf, i, "self_attn.k_norm.weight"), &.{head_dim}, 0, &total_bytes),
            .gate_proj = try takeWeight(&lookup, &consumed, layerName(&name_buf, i, "mlp.gate_proj.weight"), &.{ hidden_size, intermediate_size }, &total_bytes),
            .up_proj = try takeWeight(&lookup, &consumed, layerName(&name_buf, i, "mlp.up_proj.weight"), &.{ hidden_size, intermediate_size }, &total_bytes),
            .down_proj = try takeWeight(&lookup, &consumed, layerName(&name_buf, i, "mlp.down_proj.weight"), &.{ intermediate_size, hidden_size }, &total_bytes),
        };
    }

    var codec_count: u32 = 0;
    for (doc.tensors) |*t| {
        if (consumed.contains(t.name)) continue;
        if (std.mem.startsWith(u8, t.name, "audio_tokenizer_weights/") or std.mem.startsWith(u8, t.name, "_audiocpp.")) {
            codec_count += 1;
        } else {
            return error.UnexpectedTensor;
        }
    }
    if (codec_count != codec_tensor_count) {
        return if (codec_count < codec_tensor_count) error.MissingTensor else error.UnexpectedTensor;
    }

    return .{
        .audio_embeddings = audio_embeddings,
        .audio_heads = audio_heads,
        .codebook_layer_offsets = codebook_layer_offsets,
        .token_embedding = token_embedding,
        .output_norm = output_norm,
        .layers = layers,
        .tensors = generator_tensor_count,
        .bytes = total_bytes,
        .codec_tensors = codec_count,
    };
}

/// Verifies `audiocpp.tensor_names` against the GGUF tensor table: every
/// `_audiocpp.N` alias must resolve to a unique `audio_tokenizer_weights/*`
/// name, every non-aliased tensor must keep its name, and the resolved set
/// must partition into 313 `weights/*` and 527 `audio_tokenizer_weights/*`.
pub fn verifyTensorNames(gpa: Allocator, doc: *const gguf.Document, directory: []const u8) (gguf.Error || Error)!void {
    const names_arr = try requireArray(doc, "audiocpp.tensor_names", .string);
    if (names_arr.count != doc.tensors.len) return error.UnsupportedConfiguration;
    var reader = try gguf.ArrayReader.init(names_arr, directory, .{});

    var seen: std.StringHashMapUnmanaged(void) = .empty;
    defer seen.deinit(gpa);
    try seen.ensureTotalCapacity(gpa, @intCast(doc.tensors.len));

    var weight_count: u32 = 0;
    var codec_count: u32 = 0;
    for (doc.tensors) |t| {
        const native = (try reader.nextString()) orelse return error.UnsupportedConfiguration;
        if (native.len == 0) return error.UnsupportedConfiguration;
        if ((seen.getOrPutAssumeCapacity(native)).found_existing) return error.DuplicateTensor;
        if (!std.mem.startsWith(u8, t.name, "_audiocpp.") and !std.mem.eql(u8, t.name, native))
            return error.UnsupportedConfiguration;
        if (std.mem.startsWith(u8, native, "weights/")) {
            weight_count += 1;
        } else if (std.mem.startsWith(u8, native, "audio_tokenizer_weights/")) {
            codec_count += 1;
        } else {
            return error.UnexpectedTensor;
        }
    }
    if (weight_count != generator_tensor_count or codec_count != codec_tensor_count)
        return error.UnsupportedConfiguration;
}

/// Validates embedded `config.json` against OmniVoice's pinned geometry.
pub fn validateConfigJson(gpa: Allocator, config_json: []const u8) Error!void {
    const RopeParams = struct {
        rope_theta: f64 = 0,
    };
    const LlmConfig = struct {
        model_type: []const u8 = "",
        num_hidden_layers: u32 = 0,
        hidden_size: u32 = 0,
        intermediate_size: u32 = 0,
        num_attention_heads: u32 = 0,
        num_key_value_heads: u32 = 0,
        head_dim: u32 = 0,
        hidden_act: []const u8 = "",
        attention_bias: bool = true,
        rms_norm_eps: f64 = 0,
        rope_theta: ?f64 = null,
        rope_parameters: ?RopeParams = null,
        vocab_size: u32 = 0,
    };
    const RootConfig = struct {
        model_type: []const u8 = "",
        num_audio_codebook: u32 = 0,
        audio_vocab_size: u32 = 0,
        audio_mask_id: u32 = 0,
        audio_codebook_weights: []const u32 = &.{},
        llm_config: ?LlmConfig = null,
    };
    const parsed = std.json.parseFromSlice(RootConfig, gpa, config_json, .{
        .ignore_unknown_fields = true,
    }) catch |err| return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.UnsupportedConfiguration,
    };
    defer parsed.deinit();
    const c = parsed.value;
    if (!std.mem.eql(u8, c.model_type, "omnivoice")) return error.UnsupportedConfiguration;
    if (c.num_audio_codebook != num_audio_codebooks or c.audio_vocab_size != audio_vocab_size or c.audio_mask_id != audio_mask_id)
        return error.UnsupportedConfiguration;
    if (!std.mem.eql(u32, c.audio_codebook_weights, &audio_codebook_weights))
        return error.UnsupportedConfiguration;
    const llm = c.llm_config orelse return error.UnsupportedConfiguration;
    if (!std.mem.eql(u8, llm.model_type, "qwen3")) return error.UnsupportedConfiguration;
    if (llm.num_hidden_layers != layer_count or llm.hidden_size != hidden_size or llm.intermediate_size != intermediate_size or
        llm.num_attention_heads != num_attention_heads or llm.num_key_value_heads != num_key_value_heads or
        llm.head_dim != head_dim or llm.vocab_size != vocab_size)
        return error.UnsupportedConfiguration;
    if (!std.mem.eql(u8, llm.hidden_act, "silu") or llm.attention_bias)
        return error.UnsupportedConfiguration;
    if (@abs(llm.rms_norm_eps - @as(f64, rms_norm_eps)) > 1e-9)
        return error.UnsupportedConfiguration;
    const theta = llm.rope_theta orelse if (llm.rope_parameters) |rp| rp.rope_theta else return error.UnsupportedConfiguration;
    if (@abs(theta - @as(f64, rope_theta)) > 1.0)
        return error.UnsupportedConfiguration;
}

/// Verifies the mapped `weights/codebook_layer_offsets` I64 tensor equals
/// `[0, 1025, 2050, 3075, 4100, 5125, 6150, 7175]`.
pub fn verifyCodebookOffsets(binding: *const Binding, weights_data: []const u8) error{UnsupportedConfiguration}!void {
    const t = binding.codebook_layer_offsets;
    if (t.offset > weights_data.len or t.bytes != num_audio_codebooks * 8 or t.bytes > weights_data.len - t.offset)
        return error.UnsupportedConfiguration;
    const raw = weights_data[@intCast(t.offset)..][0 .. num_audio_codebooks * 8];
    for (expected_codebook_offsets, 0..) |expected, i| {
        const actual = std.mem.readInt(i64, raw[i * 8 ..][0..8], .little);
        if (actual != expected) return error.UnsupportedConfiguration;
    }
}

fn requireArray(doc: *const gguf.Document, key: []const u8, element_type: gguf.Type) Error!gguf.Array {
    const val = doc.get(key) orelse return error.MissingMetadata;
    const arr = switch (val) {
        .array => |a| a,
        else => return error.InvalidMetadata,
    };
    if (arr.element_type != element_type) return error.InvalidMetadata;
    return arr;
}

fn layerName(buf: *[96]u8, layer: usize, suffix: []const u8) []const u8 {
    return std.fmt.bufPrint(buf, "weights/llm.layers.{d}.{s}", .{ layer, suffix }) catch unreachable;
}

fn takeWeight(
    lookup: *const std.StringHashMapUnmanaged(*const Tensor),
    consumed: *std.StringHashMapUnmanaged(void),
    name: []const u8,
    shape: []const u64,
    total_bytes: *u64,
) Error!*const Tensor {
    const t = lookup.get(name) orelse return error.MissingTensor;
    if (!std.mem.eql(u64, t.dimensions, shape)) return error.InvalidTensorShape;
    if (!executableWeightEncoding(t.encoding_id)) return error.UnsupportedTensorEncoding;
    consumed.putAssumeCapacity(t.name, {});
    total_bytes.* += t.bytes;
    return t;
}

fn takeExact(
    lookup: *const std.StringHashMapUnmanaged(*const Tensor),
    consumed: *std.StringHashMapUnmanaged(void),
    name: []const u8,
    shape: []const u64,
    encoding_id: u32,
    total_bytes: *u64,
) Error!*const Tensor {
    const t = lookup.get(name) orelse return error.MissingTensor;
    if (!std.mem.eql(u64, t.dimensions, shape)) return error.InvalidTensorShape;
    if (t.encoding_id != encoding_id) return error.UnsupportedTensorEncoding;
    consumed.putAssumeCapacity(t.name, {});
    total_bytes.* += t.bytes;
    return t;
}

// ---- tests -----------------------------------------------------------------

pub fn inventoryDocument(gpa: Allocator) !gguf.Document {
    return models.inventory.document(gpa, @embedFile("fixtures/omnivoice-voicetut-q8_0.json"));
}

fn metadataEntry(doc: *gguf.Document, key: []const u8) *gguf.Metadata {
    const mutable = @constCast(doc.metadata);
    for (mutable) |*entry| if (std.mem.eql(u8, entry.key, key)) return entry;
    unreachable;
}

fn tensorEntry(doc: *gguf.Document, name: []const u8) *Tensor {
    const mutable = @constCast(doc.tensors);
    for (mutable) |*t| if (std.mem.eql(u8, t.name, name)) return t;
    unreachable;
}

test "bind VoiceTut-TTS inventory fixture" {
    var doc = try inventoryDocument(std.testing.allocator);
    defer doc.deinit();
    const b = try bind(std.testing.allocator, &doc);
    try std.testing.expectEqual(@as(u32, generator_tensor_count), b.tensors);
    try std.testing.expectEqual(@as(u32, codec_tensor_count), b.codec_tensors);
    try std.testing.expectEqual(@as(u32, 27), b.codebook_layer_offsets.encoding_id);
    try std.testing.expectEqual(@as(u64, 64), b.codebook_layer_offsets.bytes);
    try std.testing.expectEqual(@as(u32, 8), b.layers[0].q_proj.encoding_id);
    try std.testing.expectEqual(@as(u32, 0), b.layers[27].q_norm.encoding_id);
}

test "reject invalid container metadata, shapes, encodings, and extra tensors" {
    var doc = try inventoryDocument(std.testing.allocator);
    defer doc.deinit();

    // Architecture mismatch.
    const arch = metadataEntry(&doc, "general.architecture");
    arch.value = .{ .string = "qwen3" };
    try std.testing.expectError(error.UnsupportedArchitecture, bind(std.testing.allocator, &doc));
    arch.value = .{ .string = container_architecture };

    // Family mismatch.
    const fam = metadataEntry(&doc, "audiocpp.model_spec.family");
    fam.value = .{ .string = "other" };
    try std.testing.expectError(error.UnsupportedArchitecture, bind(std.testing.allocator, &doc));
    fam.value = .{ .string = family_name };

    // Spec version mismatch.
    const ver = metadataEntry(&doc, "audiocpp.model_spec.version");
    ver.value = .{ .unsigned = 2 };
    try std.testing.expectError(error.UnsupportedConfiguration, bind(std.testing.allocator, &doc));
    ver.value = .{ .unsigned = 1 };

    // Wrong tensor_names count.
    const tnames = metadataEntry(&doc, "audiocpp.tensor_names");
    tnames.value.array.count = total_tensor_count - 1;
    try std.testing.expectError(error.UnsupportedConfiguration, bind(std.testing.allocator, &doc));
    tnames.value.array.count = total_tensor_count;

    // Missing tensor.
    const q0 = tensorEntry(&doc, "weights/llm.layers.0.self_attn.q_proj.weight");
    const saved_q0 = q0.*;
    q0.name = "weights/llm.layers.0.self_attn.q_proj.wrong";
    try std.testing.expectError(error.MissingTensor, bind(std.testing.allocator, &doc));
    q0.* = saved_q0;

    // Wrong shape.
    q0.dimensions = &.{ hidden_size, hidden_size };
    try std.testing.expectError(error.InvalidTensorShape, bind(std.testing.allocator, &doc));
    q0.* = saved_q0;

    // Unsupported weight encoding (Q4_0).
    q0.encoding_id = 2;
    try std.testing.expectError(error.UnsupportedTensorEncoding, bind(std.testing.allocator, &doc));
    q0.* = saved_q0;

    // Wrong encoding on I64 offsets tensor.
    const offs = tensorEntry(&doc, "weights/codebook_layer_offsets");
    offs.encoding_id = 0;
    try std.testing.expectError(error.UnsupportedTensorEncoding, bind(std.testing.allocator, &doc));
    offs.encoding_id = 27;

    // Wrong encoding on norm tensor.
    const norm = tensorEntry(&doc, "weights/llm.norm.weight");
    norm.encoding_id = 1;
    try std.testing.expectError(error.UnsupportedTensorEncoding, bind(std.testing.allocator, &doc));
    norm.encoding_id = 0;

    // Unexpected extra tensor and duplicate tensor.
    const old = doc.tensors;
    const extended = try doc.storage.allocator().alloc(Tensor, old.len + 1);
    @memcpy(extended[0..old.len], old);
    extended[old.len] = saved_q0;
    extended[old.len].name = "weights/extra.weight";
    doc.tensors = extended;
    tnames.value.array.count = extended.len;
    try std.testing.expectError(error.UnexpectedTensor, bind(std.testing.allocator, &doc));
    extended[old.len].name = saved_q0.name;
    try std.testing.expectError(error.DuplicateTensor, bind(std.testing.allocator, &doc));
}

test "validate config.json and codebook_layer_offsets values" {
    const valid_cfg =
        \\{"model_type":"omnivoice","num_audio_codebook":8,"audio_vocab_size":1025,"audio_mask_id":1024,
        \\ "audio_codebook_weights":[8,8,6,6,4,4,2,2],
        \\ "llm_config":{"model_type":"qwen3","num_hidden_layers":28,"hidden_size":1024,"intermediate_size":3072,
        \\  "num_attention_heads":16,"num_key_value_heads":8,"head_dim":128,"hidden_act":"silu",
        \\  "attention_bias":false,"rms_norm_eps":1e-6,"rope_parameters":{"rope_theta":1000000},"vocab_size":151676}}
    ;
    try validateConfigJson(std.testing.allocator, valid_cfg);

    const bad_cfg = try std.mem.replaceOwned(u8, std.testing.allocator, valid_cfg, "\"num_hidden_layers\":28", "\"num_hidden_layers\":24");
    defer std.testing.allocator.free(bad_cfg);
    try std.testing.expectError(error.UnsupportedConfiguration, validateConfigJson(std.testing.allocator, bad_cfg));

    var doc = try inventoryDocument(std.testing.allocator);
    defer doc.deinit();
    var b = try bind(std.testing.allocator, &doc);
    var offs_tensor = b.codebook_layer_offsets.*;
    offs_tensor.offset = 0;
    b.codebook_layer_offsets = &offs_tensor;
    var raw: [64]u8 = undefined;
    for (expected_codebook_offsets, 0..) |v, i| std.mem.writeInt(i64, raw[i * 8 ..][0..8], v, .little);
    try verifyCodebookOffsets(&b, &raw);
    std.mem.writeInt(i64, raw[8..16], 1024, .little);
    try std.testing.expectError(error.UnsupportedConfiguration, verifyCodebookOffsets(&b, &raw));
}

fn bindWithAllocator(alloc: Allocator, doc: *const gguf.Document) !void {
    _ = try bind(alloc, doc);
}

test "binding lookup allocations are freed on every allocation failure" {
    var doc = try inventoryDocument(std.testing.allocator);
    defer doc.deinit();
    try alloc_check.checkAll(std.testing.allocator, bindWithAllocator, .{&doc});
}
