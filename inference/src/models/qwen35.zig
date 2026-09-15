//! Structural adapter for the initial Qwen3.8-27B GGUF profile (`qwen35`).
//! Bind once at model load: later numerical code uses named tensor references,
//! never repeated GGUF string lookups. Bindings borrow the Document's storage.
//! Validation checks the declared profile and storage, not tensor values or
//! numerical execution. See docs/qwen-validation.md for the pinned references.
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

/// The `general.architecture` id this adapter binds.
pub const architecture = "qwen35";

/// Whether a weight matrix of this encoding can be bound: the storage
/// layouts the CPU reference and the Metal kernels execute. Layout support
/// in `formats/gguf` (which also stores Q4_0 and BF16) is not this claim.
pub fn executableEncoding(encoding_id: u32) bool {
    return switch (encoding_id) {
        0, 8, 11, 12, 13, 14, 20, 21, 23 => true,
        else => false,
    };
}

/// What a binding reports (`models.Summary`): the pinned configuration's
/// composition, 16 full-attention and 48 DeltaNet layers plus one
/// auxiliary prediction layer excluded from the text schedule.
pub const Summary = models.Summary;
const layer_kinds = [_]models.LayerKind{
    .{ .kind = "full_attention", .count = 16 },
    .{ .kind = "delta_net", .count = 48 },
};

/// All tensor pointers borrow doc.tensors. No weights are read or allocated;
/// keep the source Document alive for the entire binding lifetime.
pub const Binding = struct {
    token_embedding: *const Tensor,
    output_norm: *const Tensor,
    output: *const Tensor,
    layers: [64]Layer,
    summary: Summary,
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
};

// This first profile is deliberately narrow. Reject unimplemented variants
// instead of accepting a familiar architecture name with different semantics.
const IntegerSetting = struct { key: []const u8, value: u64 };
const integer_settings = [_]IntegerSetting{
    .{ .key = "qwen35.block_count", .value = 65 },
    .{ .key = "qwen35.context_length", .value = 262144 },
    .{ .key = "qwen35.embedding_length", .value = 5120 },
    .{ .key = "qwen35.feed_forward_length", .value = 17408 },
    .{ .key = "qwen35.attention.head_count", .value = 24 },
    .{ .key = "qwen35.attention.head_count_kv", .value = 4 },
    .{ .key = "qwen35.attention.key_length", .value = 256 },
    .{ .key = "qwen35.attention.value_length", .value = 256 },
    .{ .key = "qwen35.nextn_predict_layers", .value = 1 },
    .{ .key = "qwen35.ssm.conv_kernel", .value = 4 },
    .{ .key = "qwen35.ssm.state_size", .value = 128 },
    .{ .key = "qwen35.ssm.group_count", .value = 16 },
    .{ .key = "qwen35.ssm.time_step_rank", .value = 48 },
    .{ .key = "qwen35.ssm.inner_size", .value = 6144 },
    .{ .key = "qwen35.full_attention_interval", .value = 4 },
    .{ .key = "qwen35.rope.dimension_count", .value = 64 },
    .{ .key = "general.quantization_version", .value = 2 },
};

fn validateMetadata(doc: *const gguf.Document) Error!void {
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
}

fn knownArchitectureKey(key: []const u8) bool {
    for (integer_settings) |setting| if (std.mem.eql(u8, key, setting.key)) return true;
    for ([_][]const u8{
        "qwen35.rope.freq_base", "qwen35.attention.layer_norm_rms_epsilon", "qwen35.rope.dimension_sections",
    }) |known| if (std.mem.eql(u8, key, known)) return true;
    return false;
}

fn expectFloat(doc: *const gguf.Document, key: []const u8, expected: f64) Error!void {
    switch (doc.get(key) orelse return error.MissingMetadata) {
        .float => |value| if (value != expected) return error.UnsupportedConfiguration,
        else => return error.InvalidMetadata,
    }
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
        var name: [128]u8 = undefined;
        const key = std.fmt.bufPrint(&name, "blk.{d}.{s}", .{ index, suffix }) catch unreachable;
        return self.take(key, dimensions, storage);
    }

    fn layer(self: *Binder, index: usize, full_attention: bool) Error!Layer {
        return .{
            .attention_norm = try self.weight(index, "attn_norm.weight", &.{5120}, .f32),
            .post_attention_norm = try self.weight(index, "post_attention_norm.weight", &.{5120}, .f32),
            .ffn_gate = try self.weight(index, "ffn_gate.weight", &.{ 5120, 17408 }, .matrix),
            .ffn_up = try self.weight(index, "ffn_up.weight", &.{ 5120, 17408 }, .matrix),
            .ffn_down = try self.weight(index, "ffn_down.weight", &.{ 17408, 5120 }, .matrix),
            .mixer = if (full_attention) .{
                .full_attention = .{
                    // Q contains both the query and the elementwise output gate.
                    .query_and_gate = try self.weight(index, "attn_q.weight", &.{ 5120, 12288 }, .matrix),
                    .key = try self.weight(index, "attn_k.weight", &.{ 5120, 1024 }, .matrix),
                    .value = try self.weight(index, "attn_v.weight", &.{ 5120, 1024 }, .matrix),
                    .output = try self.weight(index, "attn_output.weight", &.{ 6144, 5120 }, .matrix),
                    .query_norm = try self.weight(index, "attn_q_norm.weight", &.{256}, .f32),
                    .key_norm = try self.weight(index, "attn_k_norm.weight", &.{256}, .f32),
                },
            } else .{
                .delta_net = .{
                    // 2 * (16 key heads * 128) + (48 value heads * 128).
                    .qkv = try self.weight(index, "attn_qkv.weight", &.{ 5120, 10240 }, .matrix),
                    .gate = try self.weight(index, "attn_gate.weight", &.{ 5120, 6144 }, .matrix),
                    .convolution = try self.weight(index, "ssm_conv1d.weight", &.{ 4, 10240 }, .f32),
                    .time_bias = try self.weight(index, "ssm_dt.bias", &.{48}, .f32),
                    .a = try self.weight(index, "ssm_a", &.{48}, .f32),
                    .beta = try self.weight(index, "ssm_beta.weight", &.{ 5120, 48 }, .matrix),
                    .alpha = try self.weight(index, "ssm_alpha.weight", &.{ 5120, 48 }, .matrix),
                    .norm = try self.weight(index, "ssm_norm.weight", &.{128}, .f32),
                    .output = try self.weight(index, "ssm_out.weight", &.{ 6144, 5120 }, .matrix),
                },
            },
        };
    }
};

/// doc must come from successful GGUF parsing. Allocations are temporary lookup
/// storage, freed before return; only the returned tensor references borrow doc.
pub fn bind(alloc: std.mem.Allocator, doc: *const gguf.Document) Error!Binding {
    try validateMetadata(doc);
    var binder: Binder = .{ .remaining = .init(alloc) };
    defer binder.remaining.deinit();
    for (doc.tensors) |*tensor| {
        const slot = try binder.remaining.getOrPut(tensor.name);
        if (slot.found_existing) return error.DuplicateTensor;
        slot.value_ptr.* = tensor;
    }
    var result: Binding = undefined;
    result.token_embedding = try binder.take("token_embd.weight", &.{ 5120, 248320 }, .matrix);
    result.output_norm = try binder.take("output_norm.weight", &.{5120}, .f32);
    result.output = try binder.take("output.weight", &.{ 5120, 248320 }, .matrix);
    for (&result.layers, 0..) |*layer, i| layer.* = try binder.layer(i, (i + 1) % 4 == 0);
    const text_tensors = binder.tensors;
    const text_bytes = binder.bytes;

    // The auxiliary block is validated but excluded from the text binding.
    // It always uses full attention, regardless of the main schedule's modulo.
    _ = try binder.layer(64, true);
    _ = try binder.weight(64, "nextn.eh_proj.weight", &.{ 10240, 5120 }, .matrix);
    _ = try binder.weight(64, "nextn.enorm.weight", &.{5120}, .f32);
    _ = try binder.weight(64, "nextn.hnorm.weight", &.{5120}, .f32);
    _ = try binder.weight(64, "nextn.shared_head_norm.weight", &.{5120}, .f32);
    if (binder.remaining.count() != 0) return error.UnexpectedTensor;
    result.summary = .{
        .profile = "qwen35_27b",
        .decoder_layers = 64,
        .layer_kinds = &layer_kinds,
        .auxiliary_prediction_layers = 1,
        .text_tensors = text_tensors,
        .auxiliary_tensors = binder.tensors - text_tensors,
        .text_tensor_bytes = text_bytes,
        .auxiliary_tensor_bytes = binder.bytes - text_bytes,
    };
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
    for (model.layers, 0..) |layer, i| {
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

fn bindWithAllocator(alloc: std.mem.Allocator, doc: *const gguf.Document) !void {
    _ = try bind(alloc, doc);
}

test "binding lookup allocations are freed on every allocation failure" {
    var doc = try fixture();
    defer doc.deinit();
    try std.testing.checkAllAllocationFailures(std.testing.allocator, bindWithAllocator, .{&doc});
}
