//! Gemma 4 12B text adapter: metadata validation and named weight
//! binding for the pinned dense checkpoint. The facts this file encodes were
//! read from the artifact and the reference and are recorded, with their
//! provenance, in docs/reference/gemma4.md; nothing here is generic Gemma
//! knowledge. Like the Qwen adapter it is deliberately narrow: a file whose
//! `gemma4.*` keys differ from the pinned configuration is rejected with
//! `UnsupportedConfiguration` rather than run with guessed semantics.
//!
//! The architecture: 48 decoder layers in a period-6 pattern, five
//! sliding-window layers (window 1024, head size 256, 8 KV heads) then one
//! global layer (full causal attention, head size 512, one KV head, no
//! value projection: V is the key projection). Four RMS norms per layer
//! (pre/post attention, pre/post FFN), per-head query and key norms, a
//! tanh-GELU gated FFN, a scalar output scale per layer, tied embeddings,
//! and final logit soft-capping at 30. The runtimes (`gemma4_runtime.zig`,
//! `gemma4_metal.zig`) execute exactly this binding.
const std = @import("std");
const gguf = @import("../formats/gguf.zig");
const models = @import("root.zig");
const Tensor = gguf.Tensor;

pub const architecture = "gemma4";

pub const embedding = 3840;
pub const feed_forward = 15360;
pub const heads = 16;
pub const layer_count = 48;
pub const vocabulary = 262144;
pub const window = 1024;
pub const rms_epsilon: f32 = 1e-6;
pub const final_softcap: f32 = 30.0;

/// The two layer kinds and their attention geometry.
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
            .sliding => 8,
            .global => 1,
        };
    }
    pub fn ropeBase(self: Kind) f32 {
        return switch (self) {
            .sliding => 10_000,
            .global => 1_000_000,
        };
    }
    /// Query projection width: 16 heads of `headSize`.
    pub fn queryWidth(self: Kind) usize {
        return heads * self.headSize();
    }
    /// Key (and value) projection width: `kvHeads` of `headSize`.
    pub fn kvWidth(self: Kind) usize {
        return self.kvHeads() * self.headSize();
    }
};

/// Layer `index` is global when `index % 6 == 5` (layers 5, 11, …, 47), the
/// pattern both 48-element metadata arrays declare; `validateMetadata`
/// checks the arrays against this function, so the runtimes can use it.
pub fn kindOf(index: usize) Kind {
    return if (index % 6 == 5) .global else .sliding;
}

/// Whether a weight matrix of this encoding can be bound: the storage
/// layouts the shared CPU reference and Metal kernels execute. Q4_0 (id 2),
/// the QAT checkpoint's only weight encoding, completes the set.
pub fn executableEncoding(encoding_id: u32) bool {
    return switch (encoding_id) {
        0, 2, 8, 11, 12, 13, 14, 20, 21, 23 => true,
        else => false,
    };
}

pub const Summary = models.Summary;
const layer_kinds = [_]models.LayerKind{
    .{ .kind = "sliding_attention", .count = 40 },
    .{ .kind = "global_attention", .count = 8 },
};

pub const Layer = struct {
    kind: Kind,
    attention_norm: *const Tensor,
    post_attention_norm: *const Tensor,
    ffn_norm: *const Tensor,
    post_ffn_norm: *const Tensor,
    query: *const Tensor,
    key: *const Tensor,
    /// Absent on global layers, whose values are the key projection.
    value: ?*const Tensor,
    output: *const Tensor,
    query_norm: *const Tensor,
    key_norm: *const Tensor,
    ffn_gate: *const Tensor,
    ffn_up: *const Tensor,
    ffn_down: *const Tensor,
    /// One F32 scalar multiplying the layer's residual output.
    output_scale: *const Tensor,
};

/// All tensor pointers borrow doc.tensors. No weights are read or allocated;
/// keep the source Document alive for the entire binding lifetime.
pub const Binding = struct {
    /// Also the output projection (tied): logits = token_embedding · y.
    token_embedding: *const Tensor,
    output_norm: *const Tensor,
    /// 256 per-pair RoPE frequency factors for the global layers.
    rope_factors: *const Tensor,
    layers: [layer_count]Layer,
    summary: Summary,
};

pub const Error = models.BindError;

/// The registry entry (`models.table`).
pub const family = struct {
    pub const architecture = @import("gemma4.zig").architecture;
    pub const executableEncoding = @import("gemma4.zig").executableEncoding;
    pub const Binding = @import("gemma4.zig").Binding;
    pub const bind = @import("gemma4.zig").bind;
    pub const Runtime = @import("gemma4_runtime.zig").Runtime;
    pub const Plan = @import("gemma4_metal.zig").Plan;
};

const IntegerSetting = struct { key: []const u8, value: u64 };
const integer_settings = [_]IntegerSetting{
    .{ .key = "gemma4.block_count", .value = layer_count },
    .{ .key = "gemma4.context_length", .value = 262144 },
    .{ .key = "gemma4.embedding_length", .value = embedding },
    .{ .key = "gemma4.feed_forward_length", .value = feed_forward },
    .{ .key = "gemma4.attention.head_count", .value = heads },
    .{ .key = "gemma4.attention.sliding_window", .value = window },
    .{ .key = "gemma4.attention.key_length", .value = 512 },
    .{ .key = "gemma4.attention.value_length", .value = 512 },
    .{ .key = "gemma4.attention.key_length_swa", .value = 256 },
    .{ .key = "gemma4.attention.value_length_swa", .value = 256 },
    .{ .key = "gemma4.rope.dimension_count", .value = 512 },
    .{ .key = "gemma4.rope.dimension_count_swa", .value = 256 },
    .{ .key = "gemma4.attention.shared_kv_layers", .value = 0 },
    .{ .key = "gemma4.embedding_length_per_layer_input", .value = 0 },
};
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

fn validateMetadata(doc: *const gguf.Document) Error!void {
    const arch = doc.string("general.architecture") orelse return error.MissingMetadata;
    if (!std.mem.eql(u8, arch, architecture)) return error.UnsupportedArchitecture;
    for (integer_settings) |setting| {
        if (try integer(doc, setting.key) != setting.value) return error.UnsupportedConfiguration;
    }
    for (float_settings) |setting| {
        switch (doc.get(setting.key) orelse return error.MissingMetadata) {
            .float => |value| if (value != setting.value) return error.UnsupportedConfiguration,
            else => return error.InvalidMetadata,
        }
    }
    // The two per-layer arrays must both declare the period-6 pattern
    // `kindOf` hard-codes; the parser retains their 48 values.
    const kv = try retained(doc, array_settings[0]);
    const swa = try retained(doc, array_settings[1]);
    for (kv, swa, 0..) |kv_heads, sliding, index| {
        const kind = kindOf(index);
        const n: u64 = switch (kv_heads) {
            .signed => |v| if (v < 0) return error.InvalidMetadata else @intCast(v),
            .unsigned => |v| v,
            else => return error.InvalidMetadata,
        };
        if (n != kind.kvHeads()) return error.UnsupportedConfiguration;
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
        if (std.mem.startsWith(u8, entry.key, "gemma4.") and !knownArchitectureKey(entry.key))
            return error.UnsupportedConfiguration;
    }
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

fn knownArchitectureKey(key: []const u8) bool {
    for (integer_settings) |setting| if (std.mem.eql(u8, key, setting.key)) return true;
    for (float_settings) |setting| if (std.mem.eql(u8, key, setting.key)) return true;
    for (array_settings) |known| if (std.mem.eql(u8, key, known)) return true;
    return false;
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
    fn layer(self: *Binder, index: usize) Error!Layer {
        const kind = kindOf(index);
        const hd = kind.headSize();
        const q = kind.queryWidth();
        const kv = kind.kvWidth();
        return .{
            .kind = kind,
            .attention_norm = try self.weight(index, "attn_norm.weight", &.{embedding}, .f32),
            .post_attention_norm = try self.weight(index, "post_attention_norm.weight", &.{embedding}, .f32),
            .ffn_norm = try self.weight(index, "ffn_norm.weight", &.{embedding}, .f32),
            .post_ffn_norm = try self.weight(index, "post_ffw_norm.weight", &.{embedding}, .f32),
            .query = try self.weight(index, "attn_q.weight", &.{ embedding, q }, .matrix),
            .key = try self.weight(index, "attn_k.weight", &.{ embedding, kv }, .matrix),
            .value = if (kind == .sliding) try self.weight(index, "attn_v.weight", &.{ embedding, kv }, .matrix) else null,
            .output = try self.weight(index, "attn_output.weight", &.{ q, embedding }, .matrix),
            .query_norm = try self.weight(index, "attn_q_norm.weight", &.{hd}, .f32),
            .key_norm = try self.weight(index, "attn_k_norm.weight", &.{hd}, .f32),
            .ffn_gate = try self.weight(index, "ffn_gate.weight", &.{ embedding, feed_forward }, .matrix),
            .ffn_up = try self.weight(index, "ffn_up.weight", &.{ embedding, feed_forward }, .matrix),
            .ffn_down = try self.weight(index, "ffn_down.weight", &.{ feed_forward, embedding }, .matrix),
            .output_scale = try self.weight(index, "layer_output_scale.weight", &.{1}, .f32),
        };
    }
};

/// doc must come from successful GGUF parsing. Allocations are temporary
/// lookup storage, freed before return; only the returned tensor references
/// borrow doc. Every tensor in the file must be claimed: an extra one is
/// `UnexpectedTensor` (a file with a separate `output.weight` is a different
/// checkpoint, not this one).
pub fn bind(alloc: std.mem.Allocator, doc: *const gguf.Document) Error!Binding {
    try validateMetadata(doc);
    var binder: Binder = .{ .remaining = .init(alloc) };
    defer binder.remaining.deinit();
    for (doc.tensors) |*tensor| {
        const previous = try binder.remaining.fetchPut(tensor.name, tensor);
        if (previous != null) return error.DuplicateTensor;
    }
    var result: Binding = undefined;
    result.token_embedding = try binder.take("token_embd.weight", &.{ embedding, vocabulary }, .matrix);
    result.output_norm = try binder.take("output_norm.weight", &.{embedding}, .f32);
    result.rope_factors = try binder.take("rope_freqs.weight", &.{256}, .f32);
    for (&result.layers, 0..) |*layer, index| layer.* = try binder.layer(index);
    if (binder.remaining.count() != 0) return error.UnexpectedTensor;
    result.summary = .{
        .profile = "gemma4_12b",
        .decoder_layers = layer_count,
        .layer_kinds = &layer_kinds,
        .auxiliary_prediction_layers = 0,
        .text_tensors = binder.tensors,
        .auxiliary_tensors = 0,
        .text_tensor_bytes = binder.bytes,
        .auxiliary_tensor_bytes = 0,
    };
    return result;
}

/// The pinned K-quant artifact's directory inventory
/// (`fixtures/gemma4-12b.json`, docs/reference/gemma4.md) hydrated to a
/// Document with no weights.
pub fn inventoryDocument(gpa: std.mem.Allocator) !gguf.Document {
    return @import("inventory.zig").document(gpa, @embedFile("fixtures/gemma4-12b.json"));
}

test "the pinned inventory binds: 667 tensors, the period-6 pattern, no value on global layers" {
    var doc = try inventoryDocument(std.testing.allocator);
    defer doc.deinit();
    const binding = try bind(std.testing.allocator, &doc);
    try std.testing.expectEqual(@as(u32, 667), binding.summary.text_tensors);
    try std.testing.expectEqual(@as(u64, 7_350_597_824), binding.summary.text_tensor_bytes);
    try std.testing.expectEqualStrings("gemma4_12b", binding.summary.profile);
    var globals: usize = 0;
    for (binding.layers, 0..) |layer, i| {
        try std.testing.expectEqual(kindOf(i), layer.kind);
        if (layer.kind == .global) {
            globals += 1;
            try std.testing.expect(layer.value == null);
            try std.testing.expectEqual(@as(u64, 8192), layer.query.dimensions[1]);
            try std.testing.expectEqual(@as(u64, 512), layer.key_norm.dimensions[0]);
        } else {
            try std.testing.expect(layer.value != null);
            try std.testing.expectEqual(@as(u64, 2048), layer.value.?.dimensions[1]);
        }
    }
    try std.testing.expectEqual(@as(usize, 8), globals);
    try std.testing.expectEqualStrings("token_embd.weight", binding.token_embedding.name);
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

fn mutated(mutation: Mutation) !gguf.Document {
    var doc = try inventoryDocument(std.testing.allocator);
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
            const extra: Tensor = .{ .name = name, .dimensions = try alloc.dupe(u64, &.{embedding}), .encoding_id = 0, .offset = 0, .elements = embedding, .bytes = embedding * 4 };
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

test "deviations from the pinned configuration are typed rejections" {
    const cases = [_]struct { mutation: Mutation, expected: anyerror }{
        .{ .mutation = .{ .drop_tensor = "blk.3.attn_v.weight" }, .expected = error.MissingTensor },
        .{ .mutation = .{ .drop_tensor = "rope_freqs.weight" }, .expected = error.MissingTensor },
        .{ .mutation = .{ .add_tensor = "output.weight" }, .expected = error.UnexpectedTensor },
        .{ .mutation = .{ .add_tensor = "blk.5.attn_v.weight" }, .expected = error.UnexpectedTensor },
        .{ .mutation = .{ .reshape = .{ .name = "blk.5.attn_q.weight", .dimensions = &.{ 3840, 4096 } } }, .expected = error.InvalidTensorShape },
        .{ .mutation = .{ .reshape = .{ .name = "blk.0.layer_output_scale.weight", .dimensions = &.{2} } }, .expected = error.InvalidTensorShape },
        .{ .mutation = .{ .encode = .{ .name = "blk.0.ffn_up.weight", .id = 30 } }, .expected = error.UnsupportedTensorEncoding }, // BF16: stored, never executed
        .{ .mutation = .{ .encode = .{ .name = "blk.0.attn_norm.weight", .id = 1 } }, .expected = error.UnsupportedTensorEncoding },
        .{ .mutation = .{ .set_unsigned = .{ .key = "gemma4.attention.sliding_window", .value = 512 } }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .set_unsigned = .{ .key = "gemma4.block_count", .value = 30 } }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .flip_pattern = 5 }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .flip_pattern = 0 }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .drop_key = "gemma4.final_logit_softcapping" }, .expected = error.MissingMetadata },
        .{ .mutation = .{ .drop_key = "gemma4.attention.head_count_kv" }, .expected = error.MissingMetadata },
        .{ .mutation = .{ .add_key = "gemma4.attention.logit_softcapping" }, .expected = error.UnsupportedConfiguration },
    };
    for (cases) |case| {
        var doc = try mutated(case.mutation);
        defer doc.deinit();
        try std.testing.expectError(case.expected, bind(std.testing.allocator, &doc));
    }
}

test "binding survives allocation failure without leaks" {
    var doc = try inventoryDocument(std.testing.allocator);
    defer doc.deinit();
    try std.testing.checkAllAllocationFailures(std.testing.allocator, struct {
        fn check(alloc: std.mem.Allocator, d: *const gguf.Document) !void {
            _ = try bind(alloc, d);
        }
    }.check, .{&doc});
}

test "kindOf and the geometry constants" {
    try std.testing.expectEqual(Kind.global, kindOf(5));
    try std.testing.expectEqual(Kind.global, kindOf(47));
    try std.testing.expectEqual(Kind.sliding, kindOf(0));
    try std.testing.expectEqual(Kind.sliding, kindOf(46));
    try std.testing.expectEqual(@as(usize, 4096), Kind.sliding.queryWidth());
    try std.testing.expectEqual(@as(usize, 2048), Kind.sliding.kvWidth());
    try std.testing.expectEqual(@as(usize, 8192), Kind.global.queryWidth());
    try std.testing.expectEqual(@as(usize, 512), Kind.global.kvWidth());
    try std.testing.expect(executableEncoding(12) and executableEncoding(2) and !executableEncoding(30));
}
