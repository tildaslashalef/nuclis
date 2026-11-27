//! DSpark drafter adapter: Inkling 2 Mini's companion GGUF
//! (`general.architecture = dflash` with a Markov head), bound onto the
//! target by `inkling2.attachDraft`. Three blocks at the target's width
//! share the target's token embedding and output head; each block keeps its
//! own context keys and values, mixed from the target's residuals at
//! applications 5, 23 and 41 and its final normed hidden. An 8-row block
//! `[anchor, mask × 7]` gives seven positions, and a gated recurrent Markov
//! head adds a full-vocabulary bias per position, chained from the token
//! chosen before it. The equations are in docs/engine/speculative-decoding.md
//! § Inkling 2's DSpark drafter. The configuration is pinned to the one
//! file: a file that changes it is refused rather than run. It is told from
//! DFlash and DFlash2 files, which share its architecture id, by content.
const std = @import("std");
const alloc_check = @import("../alloc_check.zig");
const gguf = @import("../formats/gguf.zig");
const models = @import("root.zig");
const weights = @import("../runtime/weights.zig");
const Tensor = gguf.Tensor;

pub const architecture = "dflash";
pub const block_count = 3;
pub const embedding = 2304;
pub const feed_forward = 9216;
pub const heads = 24;
pub const kv_heads = 8;
pub const head_size = 128;
pub const query_width = heads * head_size;
pub const kv_width = kv_heads * head_size;
/// The target's tokens; the head's padded rows past them are dropped.
pub const vocabulary = 200058;
pub const padded_vocabulary = 201024;
/// The sliding blocks' window, counted from the block's anchor: the anchor
/// sees itself and the 1,024 committed rows before it.
pub const window = 1025;
/// Which blocks slide; the middle one attends to every committed row.
pub const sliding = [block_count]bool{ true, false, true };
/// The trained block: the anchor row and `block_size - 1` mask rows.
pub const block_size = 8;
pub const max_proposals = block_size - 1;
pub const rope_base: f32 = 1_000_000;
pub const rms_epsilon: f32 = 1e-6;
/// The soft cap of each position's biased logits.
pub const softcap: f32 = 30;
/// The target applications whose input residuals feed the context, and the
/// target's final normed hidden (`application = 48`, the head's input).
pub const target_layers = [_]usize{ 5, 23, 41, 48 };
pub const taps = target_layers.len;
/// One commit row: the four target rows, in `target_layers` order.
pub const hidden_width = taps * embedding;
/// The Markov head's rank: its state, its token codebook rows, its bias rank.
pub const markov_rank = 256;
/// `joint_proj`'s input `[state; w1[prev]; hidden; f_rnn]` and output
/// `[gate; candidate; markov]`.
pub const joint_input = 2 * markov_rank + 2 * embedding;
pub const joint_output = 3 * markov_rank;
/// The adapter's name for this configuration (`Summary.profile`).
pub const profile = "inkling2_dspark";

/// The file's encodings: F32 norms, mixes and biases, Q8_0 projections, and
/// BF16 Markov matrices.
pub fn executableEncoding(encoding_id: u32) bool {
    return switch (encoding_id) {
        0, 8, 30 => true,
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
    /// The context path: the four-tap mix, its norm, and the projections
    /// with the key norm.
    context_mix: *const Tensor,
    context_norm: *const Tensor,
    context_key: *const Tensor,
    context_key_norm: *const Tensor,
    context_value: *const Tensor,
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
    /// The anchor row's norm over the target's token embedding.
    token_norm: *const Tensor,
    /// `[embedding, 7]`: mask row `k` is block row `k + 1`'s input.
    mask: *const Tensor,
    output_norm: *const Tensor,
    /// The Markov head: the token codebook `w1`, the recurrence
    /// `joint_proj`, the vocabulary bias `w2` with its own bias, and the
    /// context feature `f_rnn` (its four-tap mix and norm).
    markov_w1: *const Tensor,
    joint: *const Tensor,
    joint_bias: *const Tensor,
    markov_w2: *const Tensor,
    markov_w2_bias: *const Tensor,
    markov_mix: *const Tensor,
    markov_norm: *const Tensor,
    layers: [block_count]Layer,
    summary: Summary,
};

const IntegerSetting = struct { key: []const u8, value: u64 };
const integer_settings = [_]IntegerSetting{
    .{ .key = "dflash.block_count", .value = block_count },
    .{ .key = "dflash.context_length", .value = 1048576 },
    .{ .key = "dflash.embedding_length", .value = embedding },
    .{ .key = "dflash.feed_forward_length", .value = feed_forward },
    .{ .key = "dflash.attention.head_count", .value = heads },
    .{ .key = "dflash.attention.head_count_kv", .value = kv_heads },
    .{ .key = "dflash.attention.key_length", .value = head_size },
    .{ .key = "dflash.attention.value_length", .value = head_size },
    .{ .key = "dflash.attention.sliding_window", .value = window },
    .{ .key = "dflash.rope.dimension_count", .value = head_size },
    .{ .key = "dflash.block_size", .value = block_size },
};
const FloatSetting = struct { key: []const u8, value: f32 };
const float_settings = [_]FloatSetting{
    .{ .key = "dflash.rope.freq_base", .value = rope_base },
    .{ .key = "dflash.attention.layer_norm_rms_epsilon", .value = rms_epsilon },
    .{ .key = "dflash.final_logit_softcapping", .value = softcap },
};
/// The three switches and their only accepted values: the block's anchor is
/// not a draft position, there is no confidence head, and the Markov head
/// reads the context feature.
const BoolSetting = struct { key: []const u8, value: bool };
const bool_settings = [_]BoolSetting{
    .{ .key = "dflash.sample_from_anchor", .value = false },
    .{ .key = "dflash.has_confidence_head", .value = false },
    .{ .key = "dflash.has_markov_context", .value = true },
};
const array_settings = [_][]const u8{ "dflash.target_layers", "dflash.attention.sliding_window_pattern" };

/// Integer metadata regardless of the writer's chosen width or signedness.
fn integer(doc: *const gguf.Document, key: []const u8) Error!u64 {
    return switch (doc.get(key) orelse return error.MissingMetadata) {
        .unsigned => |n| n,
        .signed => |n| if (n < 0) error.InvalidMetadata else @intCast(n),
        else => error.InvalidMetadata,
    };
}

fn integerValue(value: gguf.Value) Error!u64 {
    return switch (value) {
        .signed => |v| if (v < 0) error.InvalidMetadata else @intCast(v),
        .unsigned => |v| v,
        else => error.InvalidMetadata,
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

/// Whether `doc` is a DSpark drafter: a `dflash` file with the Markov head's
/// codebook and the block's mask rows, which neither DFlash nor DFlash2
/// carries.
pub fn isDspark(doc: *const gguf.Document) bool {
    const arch = doc.string("general.architecture") orelse return false;
    if (!std.mem.eql(u8, arch, architecture)) return false;
    var codebook = false;
    var mask = false;
    for (doc.tensors) |tensor| {
        if (std.mem.eql(u8, tensor.name, "markov_w1.weight")) codebook = true;
        if (std.mem.eql(u8, tensor.name, "mask_embd.weight")) mask = true;
    }
    return codebook and mask;
}

/// Whether the target layers fit a target of `applications` applications:
/// strictly increasing, each an application's input below `applications`,
/// and the last the target's final hidden (`applications` itself).
pub fn fitsTarget(applications: usize) bool {
    for (target_layers[0 .. taps - 1], target_layers[1..]) |a, b| if (a >= b) return false;
    return target_layers[taps - 2] < applications and target_layers[taps - 1] == applications;
}

fn validateMetadata(doc: *const gguf.Document) Error!void {
    const arch = doc.string("general.architecture") orelse return error.MissingMetadata;
    if (!std.mem.eql(u8, arch, architecture)) return error.UnsupportedArchitecture;
    if (!isDspark(doc)) return error.UnsupportedConfiguration;
    for (integer_settings) |setting| if (try integer(doc, setting.key) != setting.value) return error.UnsupportedConfiguration;
    for (float_settings) |setting| {
        switch (doc.get(setting.key) orelse return error.MissingMetadata) {
            .float => |value| if (value != setting.value) return error.UnsupportedConfiguration,
            else => return error.InvalidMetadata,
        }
    }
    for (bool_settings) |setting| {
        switch (doc.get(setting.key) orelse return error.MissingMetadata) {
            .boolean => |value| if (value != setting.value) return error.UnsupportedConfiguration,
            else => return error.InvalidMetadata,
        }
    }
    // The context taps and the window pattern are the schedule, not a
    // preference.
    for (try retained(doc, array_settings[0], taps), target_layers) |value, expected| {
        if (try integerValue(value) != expected) return error.UnsupportedConfiguration;
    }
    for (try retained(doc, array_settings[1], block_count), sliding) |value, expected| {
        const windowed = switch (value) {
            .boolean => |b| b,
            else => return error.InvalidMetadata,
        };
        if (windowed != expected) return error.UnsupportedConfiguration;
    }
    const tokens = switch (doc.get("tokenizer.ggml.tokens") orelse return error.MissingMetadata) {
        .array => |a| a,
        else => return error.InvalidMetadata,
    };
    if (tokens.element_type != .string or tokens.count != vocabulary) return error.UnsupportedConfiguration;
    // A key the pinned configuration does not carry (causal attention, a
    // selector, a rope scaling) could change the equations while every
    // checked value still matches. Refuse it.
    for (doc.metadata) |entry| {
        if (std.mem.startsWith(u8, entry.key, "dflash.") and !knownArchitectureKey(entry.key))
            return error.UnsupportedConfiguration;
    }
}

fn knownArchitectureKey(key: []const u8) bool {
    for (integer_settings) |setting| if (std.mem.eql(u8, key, setting.key)) return true;
    for (float_settings) |setting| if (std.mem.eql(u8, key, setting.key)) return true;
    for (bool_settings) |setting| if (std.mem.eql(u8, key, setting.key)) return true;
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
        var name: [64]u8 = undefined;
        const key = std.fmt.bufPrint(&name, "blk.{d}.{s}", .{ index, suffix }) catch unreachable;
        return self.take(key, dimensions, storage);
    }
    fn layer(self: *Binder, index: usize) Error!Layer {
        const d = embedding;
        return .{
            .attention_norm = try self.weight(index, "attn_norm.weight", &.{d}, .f32),
            .query = try self.weight(index, "attn_q.weight", &.{ d, query_width }, .matrix),
            .query_norm = try self.weight(index, "attn_q_norm.weight", &.{head_size}, .f32),
            .key = try self.weight(index, "attn_k.weight", &.{ d, kv_width }, .matrix),
            .key_norm = try self.weight(index, "attn_k_norm.weight", &.{head_size}, .f32),
            .value = try self.weight(index, "attn_v.weight", &.{ d, kv_width }, .matrix),
            .output = try self.weight(index, "attn_output.weight", &.{ query_width, d }, .matrix),
            .context_mix = try self.weight(index, "ctx_mix.weight", &.{taps}, .f32),
            .context_norm = try self.weight(index, "ctx_norm.weight", &.{d}, .f32),
            .context_key = try self.weight(index, "attn_k_ctx.weight", &.{ d, kv_width }, .matrix),
            .context_key_norm = try self.weight(index, "attn_k_ctx_norm.weight", &.{head_size}, .f32),
            .context_value = try self.weight(index, "attn_v_ctx.weight", &.{ d, kv_width }, .matrix),
            .ffn_norm = try self.weight(index, "ffn_norm.weight", &.{d}, .f32),
            .ffn_gate = try self.weight(index, "ffn_gate.weight", &.{ d, feed_forward }, .matrix),
            .ffn_up = try self.weight(index, "ffn_up.weight", &.{ d, feed_forward }, .matrix),
            .ffn_down = try self.weight(index, "ffn_down.weight", &.{ feed_forward, d }, .matrix),
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
    result.token_norm = try binder.take("token_embd_norm.weight", &.{embedding}, .f32);
    result.mask = try binder.take("mask_embd.weight", &.{ embedding, block_size - 1 }, .f32);
    result.output_norm = try binder.take("output_norm.weight", &.{embedding}, .f32);
    result.markov_w1 = try binder.take("markov_w1.weight", &.{ markov_rank, vocabulary }, .matrix);
    result.joint = try binder.take("joint_proj.weight", &.{ joint_input, joint_output }, .matrix);
    result.joint_bias = try binder.take("joint_proj.bias", &.{joint_output}, .f32);
    result.markov_w2 = try binder.take("markov_w2.weight", &.{ markov_rank, vocabulary }, .matrix);
    result.markov_w2_bias = try binder.take("markov_w2.bias", &.{vocabulary}, .f32);
    result.markov_mix = try binder.take("markov_ctx_mix.weight", &.{taps}, .f32);
    result.markov_norm = try binder.take("markov_ctx_norm.weight", &.{embedding}, .f32);
    for (&result.layers, 0..) |*layer, index| layer.* = try binder.layer(index);
    if (binder.remaining.count() != 0) return error.UnexpectedTensor;
    result.summary = .{
        .profile = profile,
        .decoder_layers = block_count,
        .layer_kinds = &.{
            .{ .kind = "sliding_attention", .count = 2 },
            .{ .kind = "full_attention", .count = 1 },
        },
        .auxiliary_prediction_layers = block_count,
        .text_tensors = binder.tensors,
        .auxiliary_tensors = 0,
        .text_tensor_bytes = binder.bytes,
        .auxiliary_tensor_bytes = 0,
    };
    return result;
}

/// `out = Σ_i mix[i] · features[i]` over the four commit-row taps, each
/// `out.len` wide, accumulated in F64 in tap order.
pub fn mixTaps(features: []const f32, mix: []const f32, out: []f32) void {
    std.debug.assert(mix.len == taps and features.len == taps * out.len);
    for (out, 0..) |*o, e| {
        var sum: f64 = 0;
        for (mix, 0..) |m, i| sum += @as(f64, m) * features[i * out.len + e];
        o.* = @floatCast(sum);
    }
}

/// One step of the Markov head's recurrence from `joint` (`joint_proj`'s
/// output, bias added): `gate = σ(joint[0])`, `candidate = tanh(joint[1])`,
/// `state ← candidate + gate · (state − candidate)`, and `markov =
/// tanh(joint[2])`, each chunk `markov_rank` wide.
pub fn markovStep(joint: []const f32, state: []f32, markov: []f32) void {
    std.debug.assert(joint.len == joint_output and state.len == markov_rank and markov.len == markov_rank);
    for (state, markov, 0..) |*s, *m, r| {
        const gate = 1.0 / (1.0 + @exp(-joint[r]));
        const candidate = std.math.tanh(joint[markov_rank + r]);
        s.* = candidate + gate * (s.* - candidate);
        m.* = std.math.tanh(joint[2 * markov_rank + r]);
    }
}

/// A position's biased, capped logits: `cap · tanh((base + bias) / cap)`.
pub fn capped(base: []const f32, bias: []const f32, out: []f32) void {
    std.debug.assert(base.len == out.len and bias.len == out.len);
    for (out, base, bias) |*o, b, x| o.* = softcap * std.math.tanh((b + x) / softcap);
}

pub fn inventoryDocument(gpa: std.mem.Allocator) !gguf.Document {
    return @import("inventory.zig").document(gpa, @embedFile("fixtures/dspark-inkling2-q8_0.json"));
}

const test_view: weights.View = .{ .file = &.{}, .data_offset = 0 };

test "the pinned DSpark drafter binds: 58 tensors, three blocks with their context path, the Markov head" {
    var doc = try inventoryDocument(std.testing.allocator);
    defer doc.deinit();
    try std.testing.expect(isDspark(&doc));
    const binding = try bind(std.testing.allocator, &doc, test_view);
    try std.testing.expectEqual(@as(u32, 58), binding.summary.text_tensors);
    try std.testing.expectEqualStrings("inkling2_dspark", binding.summary.profile);
    try std.testing.expectEqual(@as(u64, 5120), binding.joint.dimensions[0]);
    try std.testing.expectEqual(@as(u64, 768), binding.joint.dimensions[1]);
    try std.testing.expectEqual(@as(u32, 30), binding.markov_w2.encoding_id);
    try std.testing.expectEqual(@as(u32, 8), binding.layers[1].context_key.encoding_id);
    try std.testing.expectEqual(@as(u64, 7), binding.mask.dimensions[1]);
    try std.testing.expect(fitsTarget(48) and !fitsTarget(47) and !fitsTarget(49));
}

test "DFlash and DFlash2 files are not DSpark files, and refuse one" {
    var muse = try @import("dflash.zig").inventoryDocument(std.testing.allocator);
    defer muse.deinit();
    try std.testing.expect(!isDspark(&muse));
    try std.testing.expectError(error.UnsupportedConfiguration, bind(std.testing.allocator, &muse, test_view));
    var qwen = try @import("dflash2.zig").inventoryDocument(std.testing.allocator);
    defer qwen.deinit();
    try std.testing.expect(!isDspark(&qwen));
    try std.testing.expectError(error.UnsupportedConfiguration, bind(std.testing.allocator, &qwen, test_view));
    var doc = try inventoryDocument(std.testing.allocator);
    defer doc.deinit();
    try std.testing.expect(!@import("dflash2.zig").isDflash2(&doc));
    try std.testing.expectError(error.UnsupportedConfiguration, @import("dflash2.zig").bind(std.testing.allocator, &doc, test_view));
    try std.testing.expectError(error.UnsupportedConfiguration, @import("dflash.zig").bind(std.testing.allocator, &doc, test_view));
}

test "deviations from the pinned DSpark drafter are typed rejections" {
    const Mutation = union(enum) {
        drop_tensor: []const u8,
        set_unsigned: struct { key: []const u8, value: u64 },
        set_bool: struct { key: []const u8, value: bool },
        set_float: struct { key: []const u8, value: f64 },
        add_key: []const u8,
        retarget: [taps]u64,
        reshape: struct { name: []const u8, dimensions: []const u64 },
    };
    const cases = [_]struct { mutation: Mutation, expected: anyerror }{
        .{ .mutation = .{ .drop_tensor = "markov_w2.bias" }, .expected = error.MissingTensor },
        .{ .mutation = .{ .drop_tensor = "blk.2.attn_v_ctx.weight" }, .expected = error.MissingTensor },
        .{ .mutation = .{ .drop_tensor = "markov_w1.weight" }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .set_unsigned = .{ .key = "dflash.block_size", .value = 16 } }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .set_unsigned = .{ .key = "dflash.attention.sliding_window", .value = 1024 } }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .set_bool = .{ .key = "dflash.sample_from_anchor", .value = true } }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .set_bool = .{ .key = "dflash.has_markov_context", .value = false } }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .set_float = .{ .key = "dflash.final_logit_softcapping", .value = 0 } }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .add_key = "dflash.attention.causal" }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .retarget = .{ 5, 23, 41, 47 } }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .reshape = .{ .name = "joint_proj.weight", .dimensions = &.{ 2816, 768 } } }, .expected = error.InvalidTensorShape },
        .{ .mutation = .{ .reshape = .{ .name = "mask_embd.weight", .dimensions = &.{ 2304, 15 } } }, .expected = error.InvalidTensorShape },
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
            .set_float => |s| for (metadata) |*m| if (std.mem.eql(u8, m.key, s.key)) {
                m.value = .{ .float = s.value };
            },
            .add_key => |key| {
                const more = try alloc.alloc(gguf.Metadata, doc.metadata.len + 1);
                @memcpy(more[0..doc.metadata.len], doc.metadata);
                more[doc.metadata.len] = .{ .key = key, .kind = .boolean, .value = .{ .boolean = false } };
                doc.metadata = more;
            },
            .retarget => |layers| for (metadata) |*m| if (std.mem.eql(u8, m.key, "dflash.target_layers")) {
                var array = m.value.array;
                const values = try alloc.alloc(gguf.Value, taps);
                for (values, layers) |*v, l| v.* = .{ .signed = @intCast(l) };
                array.values = values;
                m.value = .{ .array = array };
            },
            .reshape => |r| {
                const tensors = try alloc.dupe(Tensor, doc.tensors);
                for (tensors) |*t| if (std.mem.eql(u8, t.name, r.name)) {
                    t.dimensions = r.dimensions;
                };
                doc.tensors = tensors;
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

test "the tap mix weighs each target row in order" {
    const features = [_]f32{ 1, 2, 10, 20, 100, 200, 1000, 2000 };
    var out: [2]f32 = undefined;
    mixTaps(&features, &.{ 1, 0.5, -0.1, 0.01 }, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 1 + 5 - 10 + 10), out[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2 + 10 - 20 + 20), out[1], 1e-5);
}

test "the Markov step: a closed gate takes the candidate, an open one keeps the state" {
    var joint: [joint_output]f32 = @splat(0);
    var state: [markov_rank]f32 = @splat(0.5);
    var markov: [markov_rank]f32 = undefined;
    joint[0] = -100; // gate 0: state becomes the candidate tanh(1)
    joint[markov_rank] = 1;
    joint[1] = 100; // gate 1: state 1 stays 0.5
    joint[markov_rank + 1] = -1;
    joint[2 * markov_rank] = 0.25;
    markovStep(&joint, &state, &markov);
    try std.testing.expectApproxEqAbs(std.math.tanh(@as(f32, 1)), state[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), state[1], 1e-6);
    // Gate 0.5 elsewhere: the mean of the state and a zero candidate.
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), state[2], 1e-6);
    try std.testing.expectApproxEqAbs(std.math.tanh(@as(f32, 0.25)), markov[0], 1e-6);
    try std.testing.expectEqual(@as(f32, 0), markov[1]);
    var out: [2]f32 = undefined;
    capped(&.{ 10, 1000 }, &.{ 5, 0 }, &out);
    try std.testing.expectApproxEqAbs(30 * std.math.tanh(@as(f32, 0.5)), out[0], 1e-4);
    try std.testing.expect(out[1] <= 30 and out[1] > 29.99);
}
