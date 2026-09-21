//! DFlash drafter adapter: the metrics of the companion GGUF
//! (`general.architecture = dflash`) that Muse Glimmer loads as its draft
//! source. A 5-block, 2.6B model at the target's width that shares the
//! target's token embedding and output head: its cache is filled from the
//! target's layer-2/14/26/38/50 *input* residuals through `fc`, and a
//! 16-row mask block proposes up to 15 drafts in one forward. The facts and
//! their provenance are in docs/reference/speculative-decoding.md § The Muse
//! Glimmer DFlash drafter; the reference's mask token, non-causal block
//! attention, and anchor-first sampling are part of the pinned
//! configuration, so a file that changes them is refused rather than run.
//! The binder is deliberately narrow like the other companions.
const std = @import("std");
const gguf = @import("../formats/gguf.zig");
const models = @import("root.zig");
const weights = @import("../runtime/weights.zig");
const Tensor = gguf.Tensor;

pub const architecture = "dflash";
pub const block_count = 5;
pub const embedding = 6656;
pub const feed_forward = 19968;
pub const heads = 32;
pub const kv_heads = 8;
pub const head_size = 128;
pub const query_width = heads * head_size;
pub const kv_width = kv_heads * head_size;
pub const vocabulary = 202048;
/// The draft attention window; the reference pattern makes every block sliding.
pub const window = 2048;
/// The trained noise block: an anchor row plus up to `block_size - 1` masks.
pub const block_size = 16;
pub const rope_base: f32 = 500_000;
pub const rms_epsilon: f32 = 1e-5;
/// The tokenizer's reserved mask token, the block's non-anchor rows.
pub const mask_token: u32 = 201818;
/// The target layers whose input residuals `fc` concatenates, in order.
pub const target_layers = [_]usize{ 2, 14, 26, 38, 50 };
/// One commit row: the concatenated residuals `fc` consumes.
pub const hidden_width = block_count * embedding;
/// The adapter's name for this configuration (`Summary.profile`).
pub const profile = "muse_glimmer_dflash";

/// The whole file is K-quant or F32 norms (Q4_K/Q6_K matrices); the same set
/// the Muse language model executes.
pub fn executableEncoding(encoding_id: u32) bool {
    return switch (encoding_id) {
        0, 2, 8, 11, 12, 13, 14, 20, 21, 23 => true,
        else => false,
    };
}

pub const Error = models.BindError;
pub const Summary = models.Summary;

pub const Layer = struct {
    attention_norm: *const Tensor,
    query: *const Tensor,
    query_norm: *const Tensor,
    key: *const Tensor,
    key_norm: *const Tensor,
    value: *const Tensor,
    output: *const Tensor,
    ffn_norm: *const Tensor,
    ffn_gate: *const Tensor,
    ffn_up: *const Tensor,
    ffn_down: *const Tensor,
};

/// All tensor pointers borrow doc.tensors, and `view` is the byte source they
/// were bound from (the companion file's, not the target's). No weights are
/// read or allocated; keep the source Document and mapping alive for the
/// entire binding lifetime.
pub const Binding = struct {
    view: weights.View,
    /// Concatenated target residuals [5 × width] to one feature per position.
    fc: *const Tensor,
    /// The encoder's post-`fc` RMS norm.
    encoder_norm: *const Tensor,
    output_norm: *const Tensor,
    layers: [block_count]Layer,
    summary: Summary,
};

const IntegerSetting = struct { key: []const u8, value: u64 };
const integer_settings = [_]IntegerSetting{
    .{ .key = "dflash.block_count", .value = block_count },
    .{ .key = "dflash.context_length", .value = 131072 },
    .{ .key = "dflash.embedding_length", .value = embedding },
    .{ .key = "dflash.feed_forward_length", .value = feed_forward },
    .{ .key = "dflash.attention.head_count", .value = heads },
    .{ .key = "dflash.attention.head_count_kv", .value = kv_heads },
    .{ .key = "dflash.attention.key_length", .value = head_size },
    .{ .key = "dflash.attention.value_length", .value = head_size },
    .{ .key = "dflash.attention.sliding_window", .value = window },
    .{ .key = "dflash.block_size", .value = block_size },
};
const FloatSetting = struct { key: []const u8, value: f64 };
const float_settings = [_]FloatSetting{
    .{ .key = "dflash.rope.freq_base", .value = rope_base },
    .{ .key = "dflash.attention.layer_norm_rms_epsilon", .value = rms_epsilon },
};
const array_settings = [_][]const u8{ "dflash.target_layers", "dflash.attention.sliding_window_pattern" };

/// Integer metadata regardless of the writer's chosen width or signedness.
fn integer(doc: *const gguf.Document, key: []const u8) Error!u64 {
    return switch (doc.get(key) orelse return error.MissingMetadata) {
        .unsigned => |n| n,
        .signed => |n| if (n < 0) return error.InvalidMetadata else @intCast(n),
        else => return error.InvalidMetadata,
    };
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

fn validateMetadata(doc: *const gguf.Document) Error!void {
    const arch = doc.string("general.architecture") orelse return error.MissingMetadata;
    if (!std.mem.eql(u8, arch, architecture)) return error.UnsupportedArchitecture;
    for (integer_settings) |setting| if (try integer(doc, setting.key) != setting.value) return error.UnsupportedConfiguration;
    for (float_settings) |setting| {
        switch (doc.get(setting.key) orelse return error.MissingMetadata) {
            .float => |value| if (value != setting.value) return error.UnsupportedConfiguration,
            else => return error.InvalidMetadata,
        }
    }
    // The residual concat's source layers and the all-sliding pattern are the
    // schedule, not a preference.
    const layers = try retained(doc, array_settings[0], block_count);
    for (layers, target_layers) |value, expected| {
        const n: u64 = switch (value) {
            .signed => |v| if (v < 0) return error.InvalidMetadata else @intCast(v),
            .unsigned => |v| v,
            else => return error.InvalidMetadata,
        };
        if (n != expected) return error.UnsupportedConfiguration;
    }
    const pattern = try retained(doc, array_settings[1], block_count);
    for (pattern) |value| {
        const sliding = switch (value) {
            .boolean => |b| b,
            else => return error.InvalidMetadata,
        };
        if (!sliding) return error.UnsupportedConfiguration;
    }
    // The mask block only denoises from the tokenizer's reserved mask row.
    if (try integer(doc, "tokenizer.ggml.mask_token_id") != mask_token) return error.UnsupportedConfiguration;
    const tokens = switch (doc.get("tokenizer.ggml.tokens") orelse return error.MissingMetadata) {
        .array => |a| a,
        else => return error.InvalidMetadata,
    };
    if (tokens.element_type != .string or tokens.count != vocabulary) return error.UnsupportedConfiguration;
    // Keys the pinned configuration does not carry could change the equations
    // while every checked value still matches: `sample_from_anchor`, a causal
    // block mask, a rope dimension count, the selector/conv keys of the
    // newer DFlash2 exports. Refuse rather than guess.
    for (doc.metadata) |entry| {
        if (std.mem.startsWith(u8, entry.key, "dflash.") and !knownArchitectureKey(entry.key))
            return error.UnsupportedConfiguration;
    }
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
        const d = embedding;
        const q = query_width;
        const kv = kv_width;
        return .{
            .attention_norm = try self.weight(index, "attn_norm.weight", &.{d}, .f32),
            .query = try self.weight(index, "attn_q.weight", &.{ d, q }, .matrix),
            .query_norm = try self.weight(index, "attn_q_norm.weight", &.{head_size}, .f32),
            .key = try self.weight(index, "attn_k.weight", &.{ d, kv }, .matrix),
            .key_norm = try self.weight(index, "attn_k_norm.weight", &.{head_size}, .f32),
            .value = try self.weight(index, "attn_v.weight", &.{ d, kv }, .matrix),
            .output = try self.weight(index, "attn_output.weight", &.{ q, d }, .matrix),
            .ffn_norm = try self.weight(index, "ffn_norm.weight", &.{d}, .f32),
            .ffn_gate = try self.weight(index, "ffn_gate.weight", &.{ d, feed_forward }, .matrix),
            .ffn_up = try self.weight(index, "ffn_up.weight", &.{ d, feed_forward }, .matrix),
            .ffn_down = try self.weight(index, "ffn_down.weight", &.{ feed_forward, d }, .matrix),
        };
    }
};

/// doc must come from successful GGUF parsing. Allocations are temporary
/// lookup storage, freed before return; only the returned tensor references
/// borrow doc. Every tensor in the file must be claimed.
pub fn bind(alloc: std.mem.Allocator, doc: *const gguf.Document) Error!Binding {
    try validateMetadata(doc);
    var binder: Binder = .{ .remaining = .init(alloc) };
    defer binder.remaining.deinit();
    for (doc.tensors) |*tensor| {
        const previous = try binder.remaining.fetchPut(tensor.name, tensor);
        if (previous != null) return error.DuplicateTensor;
    }
    var result: Binding = undefined;
    result.fc = try binder.take("fc.weight", &.{ hidden_width, embedding }, .matrix);
    result.encoder_norm = try binder.take("enc.output_norm.weight", &.{embedding}, .f32);
    result.output_norm = try binder.take("output_norm.weight", &.{embedding}, .f32);
    for (&result.layers, 0..) |*layer, index| layer.* = try binder.layer(index);
    if (binder.remaining.count() != 0) return error.UnexpectedTensor;
    result.summary = .{
        .profile = profile,
        .decoder_layers = block_count,
        .layer_kinds = &.{.{ .kind = "sliding_attention", .count = block_count }},
        .auxiliary_prediction_layers = block_count,
        .text_tensors = binder.tensors,
        .auxiliary_tensors = 0,
        .text_tensor_bytes = binder.bytes,
        .auxiliary_tensor_bytes = 0,
    };
    return result;
}

pub fn inventoryDocument(gpa: std.mem.Allocator) !gguf.Document {
    return @import("inventory.zig").document(gpa, @embedFile("fixtures/dflash-kquant.json"));
}

test "the pinned DFlash drafter binds: 58 tensors, five sliding blocks, the five residuals" {
    var doc = try inventoryDocument(std.testing.allocator);
    defer doc.deinit();
    const binding = try bind(std.testing.allocator, &doc);
    try std.testing.expectEqual(@as(u32, 58), binding.summary.text_tensors);
    try std.testing.expectEqualStrings("muse_glimmer_dflash", binding.summary.profile);
    try std.testing.expectEqual(@as(u64, 33280), binding.fc.dimensions[0]);
    try std.testing.expectEqual(@as(u64, 6656), binding.fc.dimensions[1]);
    try std.testing.expectEqual(@as(u64, 4096), binding.layers[0].query.dimensions[1]);
    try std.testing.expectEqual(@as(u64, 1024), binding.layers[4].key.dimensions[1]);
    try std.testing.expectEqual(@as(u32, 12), binding.fc.encoding_id);
    try std.testing.expectEqual(@as(u32, 14), binding.layers[0].value.encoding_id);
    try std.testing.expectEqual(@as(u32, 0), binding.encoder_norm.encoding_id);
}

test "deviations from the pinned drafter are typed rejections" {
    const Mutation = union(enum) {
        drop_tensor: []const u8,
        add_tensor: []const u8,
        reshape: struct { name: []const u8, dimensions: []const u64 },
        set_unsigned: struct { key: []const u8, value: u64 },
        set_float: struct { key: []const u8, value: f64 },
        set_array_int: struct { key: []const u8, index: usize, value: i64 },
        set_array_bool: struct { key: []const u8, index: usize, value: bool },
        drop_key: []const u8,
        add_key: []const u8,
    };
    const cases = [_]struct { mutation: Mutation, expected: anyerror }{
        .{ .mutation = .{ .drop_tensor = "fc.weight" }, .expected = error.MissingTensor },
        .{ .mutation = .{ .drop_tensor = "blk.3.attn_q_norm.weight" }, .expected = error.MissingTensor },
        .{ .mutation = .{ .add_tensor = "blk.5.attn_q.weight" }, .expected = error.UnexpectedTensor },
        .{ .mutation = .{ .reshape = .{ .name = "blk.0.attn_v.weight", .dimensions = &.{ 6656, 2048 } } }, .expected = error.InvalidTensorShape },
        .{ .mutation = .{ .set_unsigned = .{ .key = "dflash.block_size", .value = 8 } }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .set_unsigned = .{ .key = "dflash.attention.sliding_window", .value = 1024 } }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .set_array_int = .{ .key = "dflash.target_layers", .index = 0, .value = 3 } }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .set_array_bool = .{ .key = "dflash.attention.sliding_window_pattern", .index = 4, .value = false } }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .set_unsigned = .{ .key = "tokenizer.ggml.mask_token_id", .value = 201819 } }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .drop_key = "tokenizer.ggml.mask_token_id" }, .expected = error.MissingMetadata },
        .{ .mutation = .{ .set_float = .{ .key = "dflash.rope.freq_base", .value = 10000 } }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .drop_key = "dflash.block_size" }, .expected = error.MissingMetadata },
        // The unpinned variants: non-causal default is the shipped one, and a
        // selector or sample-from-anchor key would change the driver's shape.
        .{ .mutation = .{ .add_key = "dflash.attention.causal" }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .add_key = "dflash.sample_from_anchor" }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .add_key = "dflash.selector_rank" }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .add_key = "dflash.rope.dimension_count" }, .expected = error.UnsupportedConfiguration },
    };
    for (cases) |case| {
        var doc = try inventoryDocument(std.testing.allocator);
        defer doc.deinit();
        const alloc = doc.storage.allocator();
        const tensors = try alloc.dupe(Tensor, doc.tensors);
        doc.tensors = tensors;
        const metadata = try alloc.dupe(gguf.Metadata, doc.metadata);
        doc.metadata = metadata;
        switch (case.mutation) {
            .drop_tensor => |name| {
                var kept: std.ArrayList(Tensor) = .empty;
                for (doc.tensors) |t| if (!std.mem.eql(u8, t.name, name)) try kept.append(alloc, t);
                doc.tensors = try kept.toOwnedSlice(alloc);
            },
            .add_tensor => |name| {
                const extra: Tensor = .{ .name = name, .dimensions = try alloc.dupe(u64, &.{ embedding, embedding }), .encoding_id = 0, .offset = 0, .elements = embedding * embedding, .bytes = embedding * embedding * 4 };
                const more = try alloc.alloc(Tensor, doc.tensors.len + 1);
                @memcpy(more[0..doc.tensors.len], doc.tensors);
                more[doc.tensors.len] = extra;
                doc.tensors = more;
            },
            .reshape => |r| for (tensors) |*t| if (std.mem.eql(u8, t.name, r.name)) {
                t.dimensions = try alloc.dupe(u64, r.dimensions);
            },
            .set_unsigned => |s| for (metadata) |*m| if (std.mem.eql(u8, m.key, s.key)) {
                m.value = .{ .unsigned = s.value };
            },
            .set_float => |s| for (metadata) |*m| if (std.mem.eql(u8, m.key, s.key)) {
                m.value = .{ .float = s.value };
            },
            .set_array_int => |s| for (metadata) |*m| if (std.mem.eql(u8, m.key, s.key)) {
                const values = try alloc.dupe(gguf.Value, m.value.array.values.?);
                values[s.index] = .{ .signed = s.value };
                m.value.array.values = values;
            },
            .set_array_bool => |s| for (metadata) |*m| if (std.mem.eql(u8, m.key, s.key)) {
                const values = try alloc.dupe(gguf.Value, m.value.array.values.?);
                values[s.index] = .{ .boolean = s.value };
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

test "the geometry constants" {
    try std.testing.expectEqual(@as(usize, 4096), query_width);
    try std.testing.expectEqual(@as(usize, 1024), kv_width);
    try std.testing.expectEqual(@as(usize, 33280), hidden_width);
    try std.testing.expect(executableEncoding(12) and executableEncoding(14) and !executableEncoding(30));
}
