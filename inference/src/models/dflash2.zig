//! DFlash2 drafter adapter: the companion GGUF (`general.architecture =
//! dflash` with the DFlash2 keys) that Qwen3.8-27B loads as a draft source.
//! A 5-block drafter at the target's width sharing the target's token
//! embedding and output head. Its cache is filled from the target's
//! layer-6/20/34/48/62 input residuals through `fc`; an 8-row mask block
//! proposes up to 7 drafts in one forward. Over Muse's DFlash (`dflash.zig`)
//! it adds a two-tap dynamic convolution around each block's attention and
//! FFN, and a candidate selector that re-ranks each position's top 16
//! logits by the chosen predecessor. The equations and their provenance are
//! in docs/engine/speculative-decoding.md § The Qwen DFlash2 drafter. The
//! configuration is pinned to the Qwen3.8 file: a file that changes it is
//! refused rather than run.
const std = @import("std");
const alloc_check = @import("../alloc_check.zig");
const gguf = @import("../formats/gguf.zig");
const models = @import("root.zig");
const weights = @import("../runtime/weights.zig");
const Tensor = gguf.Tensor;

pub const architecture = "dflash";
pub const block_count = 5;
pub const embedding = 5120;
pub const feed_forward = 17408;
pub const heads = 32;
pub const kv_heads = 8;
pub const head_size = 128;
pub const query_width = heads * head_size;
pub const kv_width = kv_heads * head_size;
pub const vocabulary = 248320;
/// The draft attention window, both directions; every block is sliding.
pub const window = 2048;
/// The trained noise block: an anchor row plus up to `block_size - 1` masks.
pub const block_size = 8;
pub const rope_base: f32 = 10_000_000;
pub const rms_epsilon: f32 = 1e-6;
/// The tokenizer's reserved mask token, the block's non-anchor rows.
pub const mask_token: u32 = 248070;
/// The target layers whose input residuals `fc` concatenates, in order.
pub const target_layers = [_]usize{ 6, 20, 34, 48, 62 };
/// One commit row: the concatenated residuals `fc` consumes.
pub const hidden_width = block_count * embedding;
/// The convolution's taps (the row and the one before it) and channel group.
pub const conv_taps = 2;
pub const conv_group = 16;
pub const conv_groups = embedding / conv_group;
/// `*_conv_proj` rows: `[half][tap][group]`, half 0 before the sublayer and
/// half 1 after it.
pub const conv_projection = 2 * conv_taps * conv_groups;
/// The selector's candidates per position and its bilinear rank.
pub const selector_top_k = 16;
pub const selector_rank = 256;
/// The adapter's name for this configuration (`Summary.profile`).
pub const profile = "qwen38_dflash2";

/// The published files' encodings: F32 norms and convolution bases, and
/// Q8_0, BF16, or K-quant matrices.
pub fn executableEncoding(encoding_id: u32) bool {
    return switch (encoding_id) {
        0, 8, 12, 13, 14, 30 => true,
        else => false,
    };
}

pub const Error = models.BindError;
pub const Summary = models.Summary;

/// One dynamic convolution: `base` is `[half][tap][embedding]` F32, and
/// `projection` maps the sublayer's normed input to `conv_projection` taps.
pub const Conv = struct {
    base: *const Tensor,
    projection: *const Tensor,
};

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
    attention_conv: Conv,
    ffn_conv: Conv,
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
    /// The selector: `hidden` projects a block row to the rank, and the two
    /// codebooks hold one rank-wide row per token.
    selector_hidden: *const Tensor,
    selector_predecessor: *const Tensor,
    selector_successor: *const Tensor,
    layers: [block_count]Layer,
    summary: Summary,
};

const IntegerSetting = struct { key: []const u8, value: u64 };
const integer_settings = [_]IntegerSetting{
    .{ .key = "dflash.block_count", .value = block_count },
    .{ .key = "dflash.context_length", .value = 262144 },
    .{ .key = "dflash.embedding_length", .value = embedding },
    .{ .key = "dflash.feed_forward_length", .value = feed_forward },
    .{ .key = "dflash.attention.head_count", .value = heads },
    .{ .key = "dflash.attention.head_count_kv", .value = kv_heads },
    .{ .key = "dflash.attention.key_length", .value = head_size },
    .{ .key = "dflash.attention.value_length", .value = head_size },
    .{ .key = "dflash.attention.sliding_window", .value = window },
    .{ .key = "dflash.block_size", .value = block_size },
    .{ .key = "dflash.conv_kernel_size", .value = conv_taps },
    .{ .key = "dflash.conv_group_size", .value = conv_group },
    .{ .key = "dflash.selector_rank", .value = selector_rank },
    .{ .key = "dflash.selector_top_k", .value = selector_top_k },
};
const FloatSetting = struct { key: []const u8, value: f64 };
const float_settings = [_]FloatSetting{
    .{ .key = "dflash.rope.freq_base", .value = rope_base },
    .{ .key = "dflash.attention.layer_norm_rms_epsilon", .value = rms_epsilon },
};
const array_settings = [_][]const u8{ "dflash.target_layers", "dflash.attention.sliding_window_pattern", "dflash.rope.dimension_sections" };
const bool_settings = [_][]const u8{"dflash.attention.causal"};

/// Integer metadata regardless of the writer's chosen width or signedness.
fn integer(doc: *const gguf.Document, key: []const u8) Error!u64 {
    return switch (doc.get(key) orelse return error.MissingMetadata) {
        .unsigned => |n| n,
        .signed => |n| if (n < 0) return error.InvalidMetadata else @intCast(n),
        else => return error.InvalidMetadata,
    };
}

fn integerValue(value: gguf.Value) Error!u64 {
    return switch (value) {
        .signed => |v| if (v < 0) return error.InvalidMetadata else @intCast(v),
        .unsigned => |v| v,
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

/// Whether `doc` carries the DFlash2 keys: the selector distinguishes it
/// from Muse's DFlash under the same architecture id.
pub fn isDflash2(doc: *const gguf.Document) bool {
    return doc.get("dflash.selector_rank") != null;
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
    // Non-causal inside the block: the reference's default and the shipped one.
    switch (doc.get(bool_settings[0]) orelse return error.MissingMetadata) {
        .boolean => |causal| if (causal) return error.UnsupportedConfiguration,
        else => return error.InvalidMetadata,
    }
    // The residual concat's source layers and the all-sliding pattern are the
    // schedule, not a preference.
    for (try retained(doc, array_settings[0], block_count), target_layers) |value, expected| {
        if (try integerValue(value) != expected) return error.UnsupportedConfiguration;
    }
    for (try retained(doc, array_settings[1], block_count)) |value| {
        const sliding = switch (value) {
            .boolean => |b| b,
            else => return error.InvalidMetadata,
        };
        if (!sliding) return error.UnsupportedConfiguration;
    }
    // RoPE over the whole 128-wide head: 64 pairs in the first section.
    for (try retained(doc, array_settings[2], 4), [_]u64{ head_size / 2, 0, 0, 0 }) |value, expected| {
        if (try integerValue(value) != expected) return error.UnsupportedConfiguration;
    }
    if (try integer(doc, "tokenizer.ggml.mask_token_id") != mask_token) return error.UnsupportedConfiguration;
    const tokens = switch (doc.get("tokenizer.ggml.tokens") orelse return error.MissingMetadata) {
        .array => |a| a,
        else => return error.InvalidMetadata,
    };
    if (tokens.element_type != .string or tokens.count != vocabulary) return error.UnsupportedConfiguration;
    // A key the pinned configuration does not carry could change the
    // equations while every checked value still matches. Refuse it.
    for (doc.metadata) |entry| {
        if (std.mem.startsWith(u8, entry.key, "dflash.") and !knownArchitectureKey(entry.key))
            return error.UnsupportedConfiguration;
    }
}

fn knownArchitectureKey(key: []const u8) bool {
    for (integer_settings) |setting| if (std.mem.eql(u8, key, setting.key)) return true;
    for (float_settings) |setting| if (std.mem.eql(u8, key, setting.key)) return true;
    for (array_settings) |known| if (std.mem.eql(u8, key, known)) return true;
    for (bool_settings) |known| if (std.mem.eql(u8, key, known)) return true;
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
    fn conv(self: *Binder, index: usize, comptime prefix: []const u8) Error!Conv {
        return .{
            .base = try self.weight(index, prefix ++ "_conv_base", &.{ embedding, conv_taps, 2 }, .f32),
            .projection = try self.weight(index, prefix ++ "_conv_proj.weight", &.{ embedding, conv_projection }, .matrix),
        };
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
            .attention_conv = try self.conv(index, "attn"),
            .ffn_conv = try self.conv(index, "ffn"),
        };
    }
};

/// doc must come from successful GGUF parsing. Allocations are temporary
/// lookup storage, freed before return; only the returned tensor references
/// borrow doc. Every tensor in the file must be claimed. `view` is the
/// companion's mapping, recorded on the binding.
pub fn bind(alloc: std.mem.Allocator, doc: *const gguf.Document, view: weights.View) Error!Binding {
    try validateMetadata(doc);
    var binder: Binder = .{ .remaining = .init(alloc) };
    defer binder.remaining.deinit();
    for (doc.tensors) |*tensor| {
        const previous = try binder.remaining.fetchPut(tensor.name, tensor);
        if (previous != null) return error.DuplicateTensor;
    }
    var result: Binding = undefined;
    result.view = view;
    result.fc = try binder.take("fc.weight", &.{ hidden_width, embedding }, .matrix);
    result.encoder_norm = try binder.take("enc.output_norm.weight", &.{embedding}, .f32);
    result.output_norm = try binder.take("output_norm.weight", &.{embedding}, .f32);
    result.selector_hidden = try binder.take("selector_hidden.weight", &.{ embedding, selector_rank }, .matrix);
    result.selector_predecessor = try binder.take("selector_predecessor.weight", &.{ selector_rank, vocabulary }, .matrix);
    result.selector_successor = try binder.take("selector_successor.weight", &.{ selector_rank, vocabulary }, .matrix);
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

/// The two-tap dynamic convolution over a block's rows, in place of nothing:
/// `out[t] = Σ_tap (base[half][tap] + taps[t][half][tap][group]) ⊙ in[t − tap]`,
/// with `in[−1] = 0` (the block's first row has no predecessor). `in` and
/// `out` hold `rows × embedding` values and must not alias; `taps` holds each
/// row's `conv_projection` values and `base` the layer's `[2][2][embedding]`.
pub fn convolve(half: usize, base: []const f32, taps: []const f32, in: []const f32, out: []f32, rows: usize) void {
    std.debug.assert(half < 2 and base.len == 2 * conv_taps * embedding);
    std.debug.assert(in.len >= rows * embedding and out.len >= rows * embedding and taps.len >= rows * conv_projection);
    for (0..rows) |t| {
        const o = out[t * embedding ..][0..embedding];
        @memset(o, 0);
        for (0..conv_taps) |tap| {
            if (tap > t) continue;
            const x = in[(t - tap) * embedding ..][0..embedding];
            const b = base[(half * conv_taps + tap) * embedding ..][0..embedding];
            const dynamic = taps[t * conv_projection + (half * conv_taps + tap) * conv_groups ..][0..conv_groups];
            for (o, x, b, 0..) |*value, input, weight, c| value.* += (weight + dynamic[c / conv_group]) * input;
        }
    }
}

/// The selector's score of each of a position's candidates:
/// `unary[k] + Σ_r predecessor[r] · hidden[r] · successors[k][r]`.
pub fn selectorScores(unary: []const f32, predecessor: []const f32, hidden: []const f32, successors: []const f32, out: []f32) void {
    std.debug.assert(predecessor.len == selector_rank and hidden.len == selector_rank);
    std.debug.assert(successors.len == unary.len * selector_rank and out.len == unary.len);
    for (out, unary, 0..) |*score, base, k| {
        const successor = successors[k * selector_rank ..][0..selector_rank];
        var sum: f64 = 0;
        for (predecessor, hidden, successor) |p, h, s| sum += @as(f64, p) * h * s;
        score.* = base + @as(f32, @floatCast(sum));
    }
}

/// The candidate the greedy selector takes: the highest score, the lowest
/// index on a tie.
pub fn selectorChoice(scores: []const f32) usize {
    var best: usize = 0;
    for (scores, 0..) |score, k| if (score > scores[best]) {
        best = k;
    };
    return best;
}

pub fn inventoryDocument(gpa: std.mem.Allocator) !gguf.Document {
    return @import("inventory.zig").document(gpa, @embedFile("fixtures/dflash2-qwen38-q8_0.json"));
}

const test_view: weights.View = .{ .file = &.{}, .data_offset = 0 };

test "the pinned DFlash2 drafter binds: 81 tensors, five sliding blocks with their convolutions, the selector" {
    var doc = try inventoryDocument(std.testing.allocator);
    defer doc.deinit();
    try std.testing.expect(isDflash2(&doc));
    const binding = try bind(std.testing.allocator, &doc, test_view);
    try std.testing.expectEqual(@as(u32, 81), binding.summary.text_tensors);
    try std.testing.expectEqualStrings("qwen38_dflash2", binding.summary.profile);
    try std.testing.expectEqual(@as(u64, 25600), binding.fc.dimensions[0]);
    try std.testing.expectEqual(@as(u64, 4096), binding.layers[0].query.dimensions[1]);
    try std.testing.expectEqual(@as(u64, 1280), binding.layers[4].ffn_conv.projection.dimensions[1]);
    try std.testing.expectEqual(@as(u64, 248320), binding.selector_successor.dimensions[1]);
    try std.testing.expectEqual(@as(u32, 8), binding.fc.encoding_id);
    try std.testing.expectEqual(@as(u32, 0), binding.layers[2].attention_conv.base.encoding_id);
}

test "Muse's DFlash file is not a DFlash2 file" {
    var doc = try @import("dflash.zig").inventoryDocument(std.testing.allocator);
    defer doc.deinit();
    try std.testing.expect(!isDflash2(&doc));
    try std.testing.expectError(error.UnsupportedConfiguration, bind(std.testing.allocator, &doc, test_view));
}

test "deviations from the pinned DFlash2 drafter are typed rejections" {
    const Mutation = union(enum) {
        drop_tensor: []const u8,
        set_unsigned: struct { key: []const u8, value: u64 },
        set_bool: struct { key: []const u8, value: bool },
        add_key: []const u8,
    };
    const cases = [_]struct { mutation: Mutation, expected: anyerror }{
        .{ .mutation = .{ .drop_tensor = "selector_successor.weight" }, .expected = error.MissingTensor },
        .{ .mutation = .{ .drop_tensor = "blk.4.ffn_conv_base" }, .expected = error.MissingTensor },
        .{ .mutation = .{ .set_unsigned = .{ .key = "dflash.block_size", .value = 16 } }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .set_unsigned = .{ .key = "dflash.selector_top_k", .value = 8 } }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .set_unsigned = .{ .key = "dflash.conv_kernel_size", .value = 3 } }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .set_bool = .{ .key = "dflash.attention.causal", .value = true } }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .add_key = "dflash.sample_from_anchor" }, .expected = error.UnsupportedConfiguration },
    };
    for (cases) |case| {
        var doc = try inventoryDocument(std.testing.allocator);
        defer doc.deinit();
        const alloc = doc.storage.allocator();
        const metadata = try alloc.dupe(gguf.Metadata, doc.metadata);
        doc.metadata = metadata;
        switch (case.mutation) {
            .drop_tensor => |name| {
                var kept: std.ArrayList(Tensor) = .empty;
                for (doc.tensors) |t| if (!std.mem.eql(u8, t.name, name)) try kept.append(alloc, t);
                doc.tensors = try kept.toOwnedSlice(alloc);
            },
            .set_unsigned => |s| for (metadata) |*m| if (std.mem.eql(u8, m.key, s.key)) {
                m.value = .{ .unsigned = s.value };
            },
            .set_bool => |s| for (metadata) |*m| if (std.mem.eql(u8, m.key, s.key)) {
                m.value = .{ .boolean = s.value };
            },
            .add_key => |key| {
                const more = try alloc.alloc(gguf.Metadata, doc.metadata.len + 1);
                @memcpy(more[0..doc.metadata.len], doc.metadata);
                more[doc.metadata.len] = .{ .key = key, .kind = .uint32, .value = .{ .unsigned = 1 } };
                doc.metadata = more;
            },
        }
        try std.testing.expectError(case.expected, bind(std.testing.allocator, &doc, test_view));
    }
}

test "binding survives allocation failure without leaks" {
    var doc = try inventoryDocument(std.testing.allocator);
    defer doc.deinit();
    try alloc_check.checkAll(std.testing.allocator, struct {
        fn check(alloc: std.mem.Allocator, d: *const gguf.Document) !void {
            _ = try bind(alloc, d, test_view);
        }
    }.check, .{&doc});
}

test "the convolution: the static and dynamic taps, the first row without a predecessor" {
    const rows = 2;
    var base: [2 * conv_taps * embedding]f32 = undefined;
    // half 1: tap 0 weight 2, tap 1 weight 3 (half 0 is ignored here).
    for (0..embedding) |c| {
        base[(0 * conv_taps + 0) * embedding + c] = 100;
        base[(0 * conv_taps + 1) * embedding + c] = 100;
        base[(1 * conv_taps + 0) * embedding + c] = 2;
        base[(1 * conv_taps + 1) * embedding + c] = 3;
    }
    var taps: [rows * conv_projection]f32 = @splat(0);
    // Row 1, half 1, tap 1, group 1 (channels 16..31) gets a dynamic 0.5.
    taps[1 * conv_projection + (1 * conv_taps + 1) * conv_groups + 1] = 0.5;
    var in: [rows * embedding]f32 = undefined;
    for (0..embedding) |c| {
        in[c] = 1;
        in[embedding + c] = 10;
    }
    var out: [rows * embedding]f32 = undefined;
    convolve(1, &base, &taps, &in, &out, rows);
    // Row 0: tap 0 only, 2 · 1.
    try std.testing.expectEqual(@as(f32, 2), out[0]);
    // Row 1: 2 · 10 + 3 · 1 = 23; in group 1, 2 · 10 + 3.5 · 1 = 23.5.
    try std.testing.expectEqual(@as(f32, 23), out[embedding + 0]);
    try std.testing.expectEqual(@as(f32, 23.5), out[embedding + 16]);
    try std.testing.expectEqual(@as(f32, 23), out[embedding + 32]);
}

test "the selector: the bilinear term can overturn the top logit" {
    var predecessor: [selector_rank]f32 = @splat(0);
    var hidden: [selector_rank]f32 = @splat(0);
    predecessor[3] = 2;
    hidden[3] = 1.5;
    const unary = [_]f32{ 5, 4.5 };
    var successors: [2 * selector_rank]f32 = @splat(0);
    successors[selector_rank + 3] = 1; // candidate 1: +3 from the bilinear term
    var scores: [2]f32 = undefined;
    selectorScores(&unary, &predecessor, &hidden, &successors, &scores);
    try std.testing.expectEqual(@as(f32, 5), scores[0]);
    try std.testing.expectEqual(@as(f32, 7.5), scores[1]);
    try std.testing.expectEqual(@as(usize, 1), selectorChoice(&scores));
    try std.testing.expectEqual(@as(usize, 0), selectorChoice(&[_]f32{ 1, 1 }));
}
