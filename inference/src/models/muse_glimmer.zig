//! Muse Glimmer 30B text adapter: metadata validation and named weight
//! binding for one pinned checkpoint. The facts this file encodes were read
//! from the artifact and the reference and are recorded, with their
//! provenance, in docs/reference/muse-glimmer.md; nothing here is generic
//! knowledge of the family. Deliberately narrow, like the other adapters:
//! a file whose `muse-glimmer.*` keys differ from the pinned configuration
//! is rejected with `UnsupportedConfiguration` rather than run with guessed
//! semantics.
//!
//! A dense decoder in a period-4 pattern: three sliding-window layers
//! (window 2048, RoPE) then one global layer (full causal attention, no
//! position encoding); 32 query heads of 128 over 2 KV heads on every
//! layer, per-head query and key norms, an attention output gate, four
//! residual-stream norms per layer with two epsilons, a SiLU-gated FFN, an
//! untied output head, a logit scale, and a tanh soft-cap at 20. The
//! runtimes (`muse_glimmer_runtime.zig`, `muse_glimmer_metal.zig`) execute
//! exactly this binding.
const std = @import("std");
const gguf = @import("../formats/gguf.zig");
const models = @import("root.zig");
const weights = @import("../runtime/weights.zig");
const dflash = @import("dflash.zig");
const Tensor = gguf.Tensor;

pub const architecture = "muse-glimmer";

/// One image span in a prompt (the shared vision contract's `Span`), used by
/// the CPU runtime and the Metal plan.
pub const VisionSpan = @import("../vision/root.zig").Span;

/// The DFlash block's proposal bound: one 16-row forward proposes the anchor
/// row plus up to 15 drafts, so a verify batch holds at most 16 rows.
pub const max_draft_proposals = dflash.block_size - 1;

pub const layer_count = 52;
pub const embedding = 6656;
pub const feed_forward = 19968;
pub const heads = 32;
pub const kv_heads = 2;
pub const head_size = 128;
pub const query_width = heads * head_size;
pub const kv_width = kv_heads * head_size;
pub const vocabulary = 202048;
pub const window = 2048;
pub const rope_base: f32 = 500_000;
/// The pre norms' and the query/key norms' epsilon (the file's key).
pub const rms_epsilon: f32 = 1e-5;
/// The post-attention and post-FFN norms' epsilon, hard-coded in the
/// reference graph rather than read from the file.
pub const post_norm_epsilon: f32 = 1e-8;
pub const logit_scale: f32 = 0.1961161345243454;
pub const final_softcap: f32 = 20.0;
/// Scores are `q · k / sqrt(head_size)`.
pub const attention_scale: f32 = 1.0 / @sqrt(@as(f32, head_size));
pub const profile = "muse_glimmer_30b";
pub const layer_kinds = [_]models.LayerKind{ .{ .kind = "sliding_attention", .count = 39 }, .{ .kind = "global_attention", .count = 13 } };

/// The two layer kinds; the attention geometry is the same on both, only
/// the visible window and the position encoding differ.
pub const Kind = enum { sliding, global };

/// Layer `index` is global when `index % 4 == 3` (layers 3, 7, …, 51): the
/// reference's `set_swa_pattern(4)` over the file's `sliding_window_pattern`.
pub fn kindOf(index: usize) Kind {
    return if (index % 4 == 3) .global else .sliding;
}

/// Whether a weight matrix of this encoding can be bound: the storage
/// layouts the shared CPU reference and Metal kernels execute (the file
/// uses Q4_K and Q5_K).
pub fn executableEncoding(encoding_id: u32) bool {
    return switch (encoding_id) {
        0, 2, 8, 11, 12, 13, 14, 20, 21, 23 => true,
        else => false,
    };
}

pub const Summary = models.Summary;

pub const Layer = struct {
    kind: Kind,
    attention_norm: *const Tensor,
    post_attention_norm: *const Tensor,
    ffn_norm: *const Tensor,
    post_ffn_norm: *const Tensor,
    query: *const Tensor,
    key: *const Tensor,
    value: *const Tensor,
    /// The attention output gate: `sigmoid(gate · a)` multiplies the
    /// attention output before the output projection.
    gate: *const Tensor,
    output: *const Tensor,
    query_norm: *const Tensor,
    key_norm: *const Tensor,
    ffn_gate: *const Tensor,
    ffn_up: *const Tensor,
    ffn_down: *const Tensor,
};

/// All tensor pointers borrow doc.tensors. No weights are read or allocated;
/// keep the source Document alive for the entire binding lifetime.
pub const Binding = struct {
    /// The DFlash draft companion when one was loaded; null for the main file.
    draft: ?dflash.Binding = null,
    token_embedding: *const Tensor,
    /// The untied output head: logits = output · y.
    output: *const Tensor,
    output_norm: *const Tensor,
    layers: [layer_count]Layer,
    summary: Summary,

    pub fn active(self: *const Binding) []const Layer {
        return &self.layers;
    }
};

pub const Error = models.BindError;
/// `bindDraft` adds the companion's mismatch, the typed error `Engine.open`
/// surfaces.
pub const DraftError = models.BindError || error{ DraftSourceMismatch, InvalidShape, TensorOutOfBounds };

/// The registry entry (`models.table`).
pub const family = struct {
    pub const architecture = @import("muse_glimmer.zig").architecture;
    pub const executableEncoding = @import("muse_glimmer.zig").executableEncoding;
    pub const Binding = @import("muse_glimmer.zig").Binding;
    pub const bind = @import("muse_glimmer.zig").bind;
    pub const Runtime = @import("muse_glimmer_runtime.zig").Runtime;
    pub const Plan = @import("muse_glimmer_metal.zig").Plan;
    /// The draft source is the `dflash` companion file.
    pub const draft_architecture = dflash.architecture;
    pub const bindDraft = @import("muse_glimmer.zig").bindDraft;
};

/// The DFlash companion for a bound target: the companion's architecture,
/// target width, and vocabulary checks already ran in `Engine.open`; this
/// binds the file and refuses one whose pinned configuration the adapter's
/// equations cannot execute. The companion shares no tensor with the target
/// (its own token embedding and head are the target's, read through the
/// engine's binding), so `view` is the only mapping it needs.
pub fn bindDraft(alloc: std.mem.Allocator, doc: *const gguf.Document, view: weights.View, target_view: weights.View, target: *const Binding) DraftError!dflash.Binding {
    _ = target_view;
    _ = target;
    var drafter = dflash.bind(alloc, doc) catch |err| switch (err) {
        error.UnsupportedArchitecture, error.UnsupportedConfiguration, error.MissingMetadata, error.InvalidMetadata => return error.DraftSourceMismatch,
        else => return err,
    };
    // Both widths are the adapter's pinned constants; the check keeps the
    // failure typed if either file ever stops being the pinned one.
    if (dflash.embedding != embedding) return error.DraftSourceMismatch;
    drafter.view = view;
    return drafter;
}

const IntegerSetting = struct { key: []const u8, value: u64 };
const integer_settings = [_]IntegerSetting{
    .{ .key = "muse-glimmer.block_count", .value = layer_count },
    .{ .key = "muse-glimmer.context_length", .value = 131072 },
    .{ .key = "muse-glimmer.embedding_length", .value = embedding },
    .{ .key = "muse-glimmer.feed_forward_length", .value = feed_forward },
    .{ .key = "muse-glimmer.attention.head_count", .value = heads },
    .{ .key = "muse-glimmer.attention.head_count_kv", .value = kv_heads },
    .{ .key = "muse-glimmer.attention.key_length", .value = head_size },
    .{ .key = "muse-glimmer.attention.value_length", .value = head_size },
    .{ .key = "muse-glimmer.attention.sliding_window", .value = window },
    .{ .key = "muse-glimmer.attention.sliding_window_pattern", .value = 4 },
};
const FloatSetting = struct { key: []const u8, value: f64 };
const float_settings = [_]FloatSetting{
    .{ .key = "muse-glimmer.rope.freq_base", .value = rope_base },
    .{ .key = "muse-glimmer.attention.layer_norm_rms_epsilon", .value = rms_epsilon },
    .{ .key = "muse-glimmer.final_logit_softcapping", .value = final_softcap },
    .{ .key = "muse-glimmer.logit_scale", .value = logit_scale },
};

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
    const tokens = switch (doc.get("tokenizer.ggml.tokens") orelse return error.MissingMetadata) {
        .array => |a| a,
        else => return error.InvalidMetadata,
    };
    if (tokens.element_type != .string or tokens.count != vocabulary) return error.UnsupportedConfiguration;
    // Any other muse-glimmer.* key (a rope dimension count, a scaling
    // block, per-layer arrays) could change the equations while every
    // checked value still matches; refuse rather than guess.
    for (doc.metadata) |entry| {
        if (std.mem.startsWith(u8, entry.key, "muse-glimmer.") and !knownArchitectureKey(entry.key))
            return error.UnsupportedConfiguration;
    }
}

fn knownArchitectureKey(key: []const u8) bool {
    for (integer_settings) |setting| if (std.mem.eql(u8, key, setting.key)) return true;
    for (float_settings) |setting| if (std.mem.eql(u8, key, setting.key)) return true;
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
        const ff = feed_forward;
        return .{
            .kind = kindOf(index),
            .attention_norm = try self.weight(index, "attn_norm.weight", &.{d}, .f32),
            .post_attention_norm = try self.weight(index, "post_attention_norm.weight", &.{d}, .f32),
            .ffn_norm = try self.weight(index, "ffn_norm.weight", &.{d}, .f32),
            .post_ffn_norm = try self.weight(index, "post_ffw_norm.weight", &.{d}, .f32),
            .query = try self.weight(index, "attn_q.weight", &.{ d, query_width }, .matrix),
            .key = try self.weight(index, "attn_k.weight", &.{ d, kv_width }, .matrix),
            .value = try self.weight(index, "attn_v.weight", &.{ d, kv_width }, .matrix),
            .gate = try self.weight(index, "attn_gate.weight", &.{ d, query_width }, .matrix),
            .output = try self.weight(index, "attn_output.weight", &.{ query_width, d }, .matrix),
            .query_norm = try self.weight(index, "attn_q_norm.weight", &.{head_size}, .f32),
            .key_norm = try self.weight(index, "attn_k_norm.weight", &.{head_size}, .f32),
            .ffn_gate = try self.weight(index, "ffn_gate.weight", &.{ d, ff }, .matrix),
            .ffn_up = try self.weight(index, "ffn_up.weight", &.{ d, ff }, .matrix),
            .ffn_down = try self.weight(index, "ffn_down.weight", &.{ ff, d }, .matrix),
        };
    }
};

/// doc must come from successful GGUF parsing. Allocations are temporary
/// lookup storage, freed before return; only the returned tensor references
/// borrow doc. Every tensor in the file must be claimed: an extra one is
/// `UnexpectedTensor` (a different checkpoint, not this one).
pub fn bind(alloc: std.mem.Allocator, doc: *const gguf.Document) Error!Binding {
    try validateMetadata(doc);
    var binder: Binder = .{ .remaining = .init(alloc) };
    defer binder.remaining.deinit();
    for (doc.tensors) |*tensor| {
        const previous = try binder.remaining.fetchPut(tensor.name, tensor);
        if (previous != null) return error.DuplicateTensor;
    }
    var result: Binding = undefined;
    result.draft = null;
    result.token_embedding = try binder.take("token_embd.weight", &.{ embedding, vocabulary }, .matrix);
    result.output = try binder.take("output.weight", &.{ embedding, vocabulary }, .matrix);
    result.output_norm = try binder.take("output_norm.weight", &.{embedding}, .f32);
    for (&result.layers, 0..) |*layer, index| layer.* = try binder.layer(index);
    if (binder.remaining.count() != 0) return error.UnexpectedTensor;
    result.summary = .{
        .profile = profile,
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

/// The pinned artifact's directory inventory (`fixtures/muse-glimmer-30b.json`,
/// docs/reference/muse-glimmer.md) hydrated to a Document with no weights.
pub fn inventoryDocument(gpa: std.mem.Allocator) !gguf.Document {
    return @import("inventory.zig").document(gpa, @embedFile("fixtures/muse-glimmer-30b.json"));
}

test "the pinned inventory binds: 731 tensors, the period-4 pattern, the untied head" {
    var doc = try inventoryDocument(std.testing.allocator);
    defer doc.deinit();
    const binding = try bind(std.testing.allocator, &doc);
    try std.testing.expectEqual(@as(u32, 731), binding.summary.text_tensors);
    try std.testing.expectEqual(@as(u64, 15_865_108_480), binding.summary.text_tensor_bytes);
    try std.testing.expectEqualStrings("muse_glimmer_30b", binding.summary.profile);
    try std.testing.expectEqual(@as(u32, 52), binding.summary.decoder_layers);
    try std.testing.expectEqualStrings("output.weight", binding.output.name);
    try std.testing.expectEqual(@as(u32, 13), binding.output.encoding_id);
    var globals: usize = 0;
    for (binding.active(), 0..) |layer, i| {
        try std.testing.expectEqual(kindOf(i), layer.kind);
        if (layer.kind == .global) globals += 1;
        try std.testing.expectEqual(@as(u64, 4096), layer.gate.dimensions[1]);
        try std.testing.expectEqual(@as(u64, 256), layer.value.dimensions[1]);
        try std.testing.expectEqual(@as(u32, if (i >= 45) 13 else 12), layer.output.encoding_id);
    }
    try std.testing.expectEqual(@as(usize, 13), globals);
    try std.testing.expectEqual(Kind.global, binding.layers[51].kind);
}

const Mutation = union(enum) {
    drop_tensor: []const u8,
    reshape: struct { name: []const u8, dimensions: []const u64 },
    encode: struct { name: []const u8, id: u32 },
    add_tensor: []const u8,
    set_unsigned: struct { key: []const u8, value: u64 },
    set_float: struct { key: []const u8, value: f64 },
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
        .set_float => |s| for (metadata) |*m| if (std.mem.eql(u8, m.key, s.key)) {
            m.value = .{ .float = s.value };
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
        .{ .mutation = .{ .drop_tensor = "blk.3.attn_gate.weight" }, .expected = error.MissingTensor },
        .{ .mutation = .{ .drop_tensor = "output.weight" }, .expected = error.MissingTensor },
        .{ .mutation = .{ .add_tensor = "blk.0.layer_output_scale.weight" }, .expected = error.UnexpectedTensor },
        .{ .mutation = .{ .add_tensor = "rope_freqs.weight" }, .expected = error.UnexpectedTensor },
        .{ .mutation = .{ .reshape = .{ .name = "blk.7.attn_k.weight", .dimensions = &.{ 6656, 512 } } }, .expected = error.InvalidTensorShape },
        .{ .mutation = .{ .reshape = .{ .name = "blk.0.attn_q_norm.weight", .dimensions = &.{4096} } }, .expected = error.InvalidTensorShape },
        .{ .mutation = .{ .encode = .{ .name = "blk.0.ffn_up.weight", .id = 30 } }, .expected = error.UnsupportedTensorEncoding }, // BF16: stored, never executed
        .{ .mutation = .{ .encode = .{ .name = "blk.0.post_ffw_norm.weight", .id = 1 } }, .expected = error.UnsupportedTensorEncoding },
        .{ .mutation = .{ .set_unsigned = .{ .key = "muse-glimmer.attention.sliding_window", .value = 1024 } }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .set_unsigned = .{ .key = "muse-glimmer.block_count", .value = 51 } }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .set_unsigned = .{ .key = "muse-glimmer.attention.sliding_window_pattern", .value = 6 } }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .set_float = .{ .key = "muse-glimmer.logit_scale", .value = 1 } }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .set_float = .{ .key = "muse-glimmer.rope.freq_base", .value = 10000 } }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .drop_key = "muse-glimmer.final_logit_softcapping" }, .expected = error.MissingMetadata },
        .{ .mutation = .{ .drop_key = "muse-glimmer.attention.head_count_kv" }, .expected = error.MissingMetadata },
        .{ .mutation = .{ .add_key = "muse-glimmer.rope.dimension_count" }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .add_key = "muse-glimmer.attention.logit_softcapping" }, .expected = error.UnsupportedConfiguration },
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
    try std.testing.expectEqual(Kind.global, kindOf(3));
    try std.testing.expectEqual(Kind.global, kindOf(51));
    try std.testing.expectEqual(Kind.sliding, kindOf(0));
    try std.testing.expectEqual(Kind.sliding, kindOf(50));
    try std.testing.expectEqual(@as(usize, 4096), query_width);
    try std.testing.expectEqual(@as(usize, 256), kv_width);
    try std.testing.expectApproxEqAbs(@as(f32, 0.08838834764831845), attention_scale, 1e-9);
    try std.testing.expect(executableEncoding(12) and executableEncoding(13) and !executableEncoding(30));
}

test "the pinned DFlash companion binds into the target binding through bindDraft" {
    var doc = try inventoryDocument(std.testing.allocator);
    defer doc.deinit();
    var target = try bind(std.testing.allocator, &doc);
    var companion = try dflash.inventoryDocument(std.testing.allocator);
    defer companion.deinit();
    const view: weights.View = .{ .file = &.{}, .data_offset = 0 };
    target.draft = try bindDraft(std.testing.allocator, &companion, view, view, &target);
    try std.testing.expectEqual(@as(u32, 58), target.draft.?.summary.text_tensors);
    try std.testing.expectEqual(@as(usize, 33280), dflash.hidden_width);
    try std.testing.expectEqualStrings("dflash", family.draft_architecture);
}
