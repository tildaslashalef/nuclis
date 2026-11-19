//! DFlash drafter for a Gemma 4 E4B target: the companion GGUF
//! (`general.architecture = dflash`) NVIDIA ships beside Nemotron 3.5 Bolt
//! E4B. Five Qwen3-style blocks at the target's width that share the
//! target's token embedding (scaled by `embedding_scale`, as Gemma scales
//! its own) and tied output head: the cache is filled from the target's
//! layer-6/12/18/24/36 *input* residuals through `fc`, and an 8-row mask
//! block proposes up to 7 drafts in one non-causal forward. The equations
//! are Muse's drafter (docs/engine/speculative-decoding.md § Muse Glimmer's
//! DFlash drafter) at this file's constants; the binder pins them and
//! refuses any other `dflash.*` key.
const std = @import("std");
const gguf = @import("../formats/gguf.zig");
const models = @import("root.zig");
const weights = @import("../runtime/weights.zig");
const Tensor = gguf.Tensor;

pub const architecture = "dflash";
pub const block_count = 5;
pub const embedding = 2560;
pub const feed_forward = 5120;
pub const heads = 32;
pub const kv_heads = 8;
pub const head_size = 128;
pub const query_width = heads * head_size;
pub const kv_width = kv_heads * head_size;
/// Every block is sliding at this window.
pub const window = 1024;
/// The trained noise block: an anchor row plus up to `block_size - 1` masks.
pub const block_size = 8;
pub const rope_base: f32 = 1_000_000;
pub const rms_epsilon: f32 = 1e-6;
pub const attention_scale: f32 = 1.0 / @sqrt(@as(f32, head_size));
/// `sqrt(embedding)`: the block rows' embedding scale, the target's own.
pub const embedding_scale: f32 = 50.59644317626953;
/// Gemma's reserved mask token, the block's non-anchor rows.
pub const mask_token: u32 = 4;
/// The target layers whose input residuals `fc` concatenates, in order.
pub const target_layers = [_]usize{ 6, 12, 18, 24, 36 };
/// One commit row: the concatenated residuals `fc` consumes.
pub const hidden_width = block_count * embedding;
pub const profile = "gemma4_dflash";

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

/// Tensor pointers borrow doc.tensors; `view` is the companion's mapping.
pub const Binding = struct {
    view: weights.View,
    fc: *const Tensor,
    encoder_norm: *const Tensor,
    output_norm: *const Tensor,
    layers: [block_count]Layer,
    summary: Summary,
};

fn executableEncoding(id: u32) bool {
    return switch (id) {
        0, 2, 8, 12, 13, 14 => true,
        else => false,
    };
}

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
    .{ .key = "dflash.embedding_scale", .value = embedding_scale },
};
const other_keys = [_][]const u8{ "dflash.target_layers", "dflash.attention.sliding_window_pattern", "dflash.attention.causal" };

fn integer(value: gguf.Value) Error!u64 {
    return switch (value) {
        .unsigned => |n| n,
        .signed => |n| if (n < 0) error.InvalidMetadata else @intCast(n),
        else => error.InvalidMetadata,
    };
}

fn array(doc: *const gguf.Document, key: []const u8) Error![]const gguf.Value {
    const a = switch (doc.get(key) orelse return error.MissingMetadata) {
        .array => |a| a,
        else => return error.InvalidMetadata,
    };
    if (a.count != block_count) return error.UnsupportedConfiguration;
    const values = a.values orelse return error.InvalidMetadata;
    if (values.len != block_count) return error.InvalidMetadata;
    return values;
}

/// Whether `doc` is this drafter (architecture `dflash` with Gemma's width).
pub fn matches(doc: *const gguf.Document) bool {
    const arch = doc.string("general.architecture") orelse return false;
    if (!std.mem.eql(u8, arch, architecture)) return false;
    const width = integer(doc.get("dflash.embedding_length") orelse return false) catch return false;
    return width == embedding;
}

fn validateMetadata(doc: *const gguf.Document) Error!void {
    if (!matches(doc)) return error.UnsupportedArchitecture;
    for (integer_settings) |s| if (try integer(doc.get(s.key) orelse return error.MissingMetadata) != s.value) return error.UnsupportedConfiguration;
    for (float_settings) |s| switch (doc.get(s.key) orelse return error.MissingMetadata) {
        // The file stores F32; compare at that precision.
        .float => |v| if (@as(f32, @floatCast(v)) != @as(f32, @floatCast(s.value))) return error.UnsupportedConfiguration,
        else => return error.InvalidMetadata,
    };
    for (try array(doc, other_keys[0]), target_layers) |v, expected| if (try integer(v) != expected) return error.UnsupportedConfiguration;
    for (try array(doc, other_keys[1])) |v| switch (v) {
        .boolean => |sliding| if (!sliding) return error.UnsupportedConfiguration,
        else => return error.InvalidMetadata,
    };
    switch (doc.get(other_keys[2]) orelse return error.MissingMetadata) {
        .boolean => |causal| if (causal) return error.UnsupportedConfiguration,
        else => return error.InvalidMetadata,
    }
    if (try integer(doc.get("tokenizer.ggml.mask_token_id") orelse return error.MissingMetadata) != mask_token) return error.UnsupportedConfiguration;
    for (doc.metadata) |entry| {
        if (!std.mem.startsWith(u8, entry.key, "dflash.")) continue;
        if (!known(entry.key)) return error.UnsupportedConfiguration;
    }
}

fn known(key: []const u8) bool {
    for (integer_settings) |s| if (std.mem.eql(u8, key, s.key)) return true;
    for (float_settings) |s| if (std.mem.eql(u8, key, s.key)) return true;
    for (other_keys) |k| if (std.mem.eql(u8, key, k)) return true;
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
        const ok = switch (storage) {
            .f32 => tensor.encoding_id == 0,
            .matrix => executableEncoding(tensor.encoding_id),
        };
        if (!ok) return error.UnsupportedTensorEncoding;
        self.tensors += 1;
        self.bytes = try std.math.add(u64, self.bytes, tensor.bytes);
        return tensor;
    }
    fn weight(self: *Binder, index: usize, suffix: []const u8, dimensions: []const u64, storage: Storage) Error!*const Tensor {
        var name: [128]u8 = undefined;
        const key = std.fmt.bufPrint(&name, "blk.{d}.{s}", .{ index, suffix }) catch unreachable;
        return self.take(key, dimensions, storage);
    }
    fn layer(self: *Binder, i: usize) Error!Layer {
        const d = embedding;
        return .{
            .attention_norm = try self.weight(i, "attn_norm.weight", &.{d}, .f32),
            .query = try self.weight(i, "attn_q.weight", &.{ d, query_width }, .matrix),
            .query_norm = try self.weight(i, "attn_q_norm.weight", &.{head_size}, .f32),
            .key = try self.weight(i, "attn_k.weight", &.{ d, kv_width }, .matrix),
            .key_norm = try self.weight(i, "attn_k_norm.weight", &.{head_size}, .f32),
            .value = try self.weight(i, "attn_v.weight", &.{ d, kv_width }, .matrix),
            .output = try self.weight(i, "attn_output.weight", &.{ query_width, d }, .matrix),
            .ffn_norm = try self.weight(i, "ffn_norm.weight", &.{d}, .f32),
            .ffn_gate = try self.weight(i, "ffn_gate.weight", &.{ d, feed_forward }, .matrix),
            .ffn_up = try self.weight(i, "ffn_up.weight", &.{ d, feed_forward }, .matrix),
            .ffn_down = try self.weight(i, "ffn_down.weight", &.{ feed_forward, d }, .matrix),
        };
    }
};

/// Binds every tensor of `doc` or fails; allocations are temporary.
pub fn bind(alloc: std.mem.Allocator, doc: *const gguf.Document, view: weights.View) Error!Binding {
    try validateMetadata(doc);
    var binder: Binder = .{ .remaining = .init(alloc) };
    defer binder.remaining.deinit();
    for (doc.tensors) |*tensor| {
        if (try binder.remaining.fetchPut(tensor.name, tensor) != null) return error.DuplicateTensor;
    }
    var result: Binding = undefined;
    result.view = view;
    result.fc = try binder.take("fc.weight", &.{ hidden_width, embedding }, .matrix);
    result.encoder_norm = try binder.take("enc.output_norm.weight", &.{embedding}, .f32);
    result.output_norm = try binder.take("output_norm.weight", &.{embedding}, .f32);
    for (&result.layers, 0..) |*layer, i| layer.* = try binder.layer(i);
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
