//! Inkling 2 Mini text adapter: metadata validation and named weight binding
//! for one pinned checkpoint (docs/models/inkling2.md). A file whose
//! `inkling2.*` keys differ from the pinned configuration is rejected with
//! `UnsupportedConfiguration` rather than run with guessed semantics.
//!
//! A looped decoder: 24 stored blocks apply 48 times, the core (stored 6–17)
//! three times over, so `storedBlock` maps an application to the block whose
//! weights it reads and `isGlobal` gives its attention kind (global every
//! sixth application, sliding otherwise). Experts are a 128-way MoE; the
//! core carries the loop's message controllers. `tiny_models` is the
//! test-only build (docs/models/README.md § Tiny models): it shrinks the
//! depth, widths, windows, experts, and vocabulary and keeps the schedule's
//! shape. `Runtime` is the CPU reference (`inkling2_runtime.zig`), `Plan` the
//! Metal plan (`inkling2_metal.zig`).
const std = @import("std");
const alloc_check = @import("../alloc_check.zig");
const gguf = @import("../formats/gguf.zig");
const models = @import("root.zig");
const session = @import("../runtime/session.zig");
const weights = @import("../runtime/weights.zig");
const tiny_models = @import("build_options").tiny_models;
const Tensor = gguf.Tensor;

pub const architecture = "inkling2";
pub const profile = "inkling2_mini";

/// Applications of a block: `inkling2.block_count`.
pub const block_count = if (tiny_models) 24 else 48;
/// Blocks whose weights the file stores: `blk.0` … `blk.{stored_blocks - 1}`.
pub const stored_blocks = if (tiny_models) 18 else 24;
pub const core_start = 6;
pub const core_end = if (tiny_models) 12 else 18;
pub const loops = if (tiny_models) 2 else 3;
pub const core_blocks = core_end - core_start;
/// `attn_res.block_size`: the blocks one attention residual spans.
pub const residual_span = 6;
/// The attention pattern's period; the last application of each is global.
pub const attention_period = 6;
pub const VisionSpan = @import("../vision/root.zig").Span;
pub const embedding = if (tiny_models) 256 else 2304;
pub const heads = if (tiny_models) 4 else 24;
pub const kv_heads = if (tiny_models) 2 else 8;
pub const head_size = 128;
pub const query_width = heads * head_size;
pub const kv_width = kv_heads * head_size;
pub const experts = if (tiny_models) 16 else 128;
pub const experts_used = if (tiny_models) 4 else 8;
pub const expert_feed_forward = if (tiny_models) 64 else 1152;
/// `tokenizer.ggml.tokens`, equal to `vocab_size_unpadded`.
pub const vocabulary = if (tiny_models) 1000 else 200058;
/// `vocab_size`: the embedding's and the head's rows; the rows past
/// `vocabulary` are padding no token reaches.
pub const padded_vocabulary = if (tiny_models) 1024 else 201024;
pub const context_length = if (tiny_models) 4096 else 1048576;
pub const window = if (tiny_models) 8 else 1024;
pub const relative_length = 16;
/// The global layers' relative-bias extent; a sliding layer's is `window`.
pub const relative_extent = if (tiny_models) 16 else 1024;
pub const log_scale_floor = if (tiny_models) 4 else 8192;
/// Short-convolution taps (`shortconv.l_cache`).
pub const conv_taps = 4;
pub const gate_rank = if (tiny_models) 16 else 64;
/// The loop residual queries' width: two per core block.
pub const loop_queries = 2 * core_blocks;
pub const rms_epsilon: f32 = 1e-6;
pub const expert_weights_scale: f32 = 8;
pub const situ_beta: f32 = 4;
pub const situ_linear_beta: f32 = 25;
pub const final_softcap: f32 = 30;
pub const qk_clip: f32 = 22.627416610717773;
pub const relative_clip: f32 = 8;
pub const log_scale_alpha: f32 = 0.1;

comptime {
    // The schedule: before the core, the core `loops` times, after it.
    std.debug.assert(block_count == core_start + loops * core_blocks + (stored_blocks - core_end));
    std.debug.assert(core_start % residual_span == 0 and core_end % residual_span == 0);
    // A stored block's attention kind is then the same on every pass.
    std.debug.assert(core_blocks % attention_period == 0);
}

pub const Kind = enum { sliding, global };

pub const layer_kinds = [_]models.LayerKind{
    .{ .kind = "sliding_attention", .count = block_count - block_count / attention_period },
    .{ .kind = "global_attention", .count = block_count / attention_period },
};

/// The stored block that application `application` (0 ≤ it < block_count) reads.
pub fn storedBlock(application: usize) usize {
    std.debug.assert(application < block_count);
    if (application < core_start) return application;
    const looped = application - core_start;
    if (looped < loops * core_blocks) return core_start + looped % core_blocks;
    return application - (loops - 1) * core_blocks;
}

/// Whether application `application` attends globally (no window).
pub fn isGlobal(application: usize) bool {
    return application % attention_period == attention_period - 1;
}

/// The attention kind of a stored block, the same on every application of it.
pub fn kindOf(stored: usize) Kind {
    return if (isGlobal(stored)) .global else .sliding;
}

/// Whether a weight matrix of this encoding can be bound: F32, Q8_0, BF16,
/// and NVFP4, the encodings the pinned file stores.
pub fn executableEncoding(encoding_id: u32) bool {
    return switch (encoding_id) {
        0, 8, 30, 40 => true,
        else => false,
    };
}

pub const Summary = models.Summary;

/// The attention input differs by kind: one norm on a sliding block; on a
/// global one, a norm per projection and input-dependent K and V conv taps.
pub const Input = union(Kind) {
    sliding: struct { norm: *const Tensor },
    global: struct {
        query_norm: *const Tensor,
        key_norm: *const Tensor,
        value_norm: *const Tensor,
        key_conv_projection: *const Tensor,
        value_conv_projection: *const Tensor,
    },
};

/// A core block's message controllers, in the order attention, key, value,
/// MLP: `down` is `[2·embedding, gate_rank]`, `gain` and `forget`
/// `[gate_rank, embedding]`.
pub const Loop = struct {
    down: [4]*const Tensor,
    gain: [4]*const Tensor,
    forget: [4]*const Tensor,
};

pub const Block = struct {
    input: Input,
    query: *const Tensor,
    key: *const Tensor,
    value: *const Tensor,
    gate: *const Tensor,
    output: *const Tensor,
    query_clip: *const Tensor,
    sink_projection: *const Tensor,
    relative: *const Tensor,
    relative_bias: *const Tensor,
    relative_clip: *const Tensor,
    key_conv: *const Tensor,
    value_conv: *const Tensor,
    attention_post_conv: *const Tensor,
    ffn_post_conv: *const Tensor,
    /// Null on block 0; `[embedding, 3]` on a global block, `[embedding]` on
    /// a sliding one.
    attention_residual_query: ?*const Tensor,
    ffn_residual_query: *const Tensor,
    ffn_norm: *const Tensor,
    router: *const Tensor,
    router_bias: *const Tensor,
    gate_up_experts: *const Tensor,
    down_experts: *const Tensor,
    /// Per-expert F32 multipliers of the two expert matrices; both or
    /// neither, present on the file's NVFP4 blocks.
    gate_up_scale: ?*const Tensor,
    down_scale: ?*const Tensor,
    /// Present exactly on the core's stored blocks.
    loop: ?Loop,

    pub fn kind(self: *const Block) Kind {
        return self.input;
    }
};

/// All tensor pointers borrow doc.tensors. No weights are read or allocated;
/// keep the source Document alive for the entire binding lifetime.
pub const Binding = struct {
    token_embedding: *const Tensor,
    token_embedding_norm: *const Tensor,
    output: *const Tensor,
    output_norm: *const Tensor,
    output_residual_query: *const Tensor,
    loop_residual_query: *const Tensor,
    loop_kv_residual_query: *const Tensor,
    blocks: [stored_blocks]Block,
    summary: Summary,
};

pub const Error = models.BindError;

/// The registry entry (`models.table`).
pub const family = struct {
    pub const architecture = @import("inkling2.zig").architecture;
    pub const executableEncoding = @import("inkling2.zig").executableEncoding;
    pub const Binding = @import("inkling2.zig").Binding;
    pub const bind = @import("inkling2.zig").bind;
    pub const Runtime = @import("inkling2.zig").Runtime;
    pub const Plan = @import("inkling2.zig").Plan;
};

pub const Runtime = @import("inkling2_runtime.zig").Runtime;

pub const Plan = @import("inkling2_metal.zig").Plan;

const IntegerSetting = struct { key: []const u8, value: u64 };
const integer_settings = [_]IntegerSetting{
    .{ .key = "inkling2.block_count", .value = block_count },
    .{ .key = "inkling2.context_length", .value = context_length },
    .{ .key = "inkling2.embedding_length", .value = embedding },
    .{ .key = "inkling2.feed_forward_length", .value = expert_feed_forward },
    .{ .key = "inkling2.expert_count", .value = experts },
    .{ .key = "inkling2.expert_used_count", .value = experts_used },
    .{ .key = "inkling2.expert_feed_forward_length", .value = expert_feed_forward },
    .{ .key = "inkling2.attention.key_length", .value = head_size },
    .{ .key = "inkling2.attention.value_length", .value = head_size },
    .{ .key = "inkling2.attention.key_length_swa", .value = head_size },
    .{ .key = "inkling2.attention.value_length_swa", .value = head_size },
    .{ .key = "inkling2.attention.sliding_window", .value = window },
    .{ .key = "inkling2.attention.relative_length", .value = relative_length },
    .{ .key = "inkling2.attention.relative_extent", .value = relative_extent },
    .{ .key = "inkling2.attention.log_scale_floor", .value = log_scale_floor },
    .{ .key = "inkling2.rope.dimension_count", .value = 0 },
    .{ .key = "inkling2.num_loops", .value = loops },
    .{ .key = "inkling2.loop.core_start", .value = core_start },
    .{ .key = "inkling2.loop.core_end", .value = core_end },
    .{ .key = "inkling2.loop.gate_rank", .value = gate_rank },
    .{ .key = "inkling2.attn_res.block_size", .value = residual_span },
    .{ .key = "inkling2.shortconv.l_cache", .value = conv_taps },
    .{ .key = "inkling2.vocab_size", .value = padded_vocabulary },
    .{ .key = "inkling2.vocab_size_unpadded", .value = vocabulary },
};
const FloatSetting = struct { key: []const u8, value: f32 };
const float_settings = [_]FloatSetting{
    .{ .key = "inkling2.attention.layer_norm_rms_epsilon", .value = rms_epsilon },
    .{ .key = "inkling2.expert_weights_scale", .value = expert_weights_scale },
    .{ .key = "inkling2.activation.situ_beta", .value = situ_beta },
    .{ .key = "inkling2.activation.situ_linear_beta", .value = situ_linear_beta },
    .{ .key = "inkling2.final_logit_softcapping", .value = final_softcap },
    .{ .key = "inkling2.attention.qk_clip", .value = qk_clip },
    .{ .key = "inkling2.attention.relative_clip", .value = relative_clip },
    .{ .key = "inkling2.attention.log_scale_alpha", .value = log_scale_alpha },
};
/// The per-application arrays, `block_count` long.
const array_keys = [_][]const u8{
    "inkling2.attention.head_count",
    "inkling2.attention.head_count_kv",
    "inkling2.attention.sliding_window_pattern",
};

/// Integer metadata regardless of the writer's chosen width or signedness.
fn integer(doc: *const gguf.Document, key: []const u8) Error!u64 {
    return switch (doc.get(key) orelse return error.MissingMetadata) {
        .unsigned => |n| n,
        .signed => |n| if (n < 0) error.InvalidMetadata else @intCast(n),
        else => error.InvalidMetadata,
    };
}

/// A per-application array's retained values, `block_count` of them.
fn perApplication(doc: *const gguf.Document, key: []const u8) Error![]const gguf.Value {
    const array = switch (doc.get(key) orelse return error.MissingMetadata) {
        .array => |a| a,
        else => return error.InvalidMetadata,
    };
    if (array.count != block_count) return error.UnsupportedConfiguration;
    return array.values orelse error.InvalidMetadata;
}

fn expectCounts(doc: *const gguf.Document, key: []const u8, expected: u64) Error!void {
    for (try perApplication(doc, key)) |value| {
        const n: u64 = switch (value) {
            .unsigned => |n| n,
            .signed => |n| if (n < 0) return error.InvalidMetadata else @intCast(n),
            else => return error.InvalidMetadata,
        };
        if (n != expected) return error.UnsupportedConfiguration;
    }
}

fn validateMetadata(doc: *const gguf.Document) Error!void {
    const arch = doc.string("general.architecture") orelse return error.MissingMetadata;
    if (!std.mem.eql(u8, arch, architecture)) return error.UnsupportedArchitecture;
    if (try integer(doc, "general.quantization_version") != 2) return error.UnsupportedConfiguration;
    for (integer_settings) |setting| {
        if (try integer(doc, setting.key) != setting.value) return error.UnsupportedConfiguration;
    }
    for (float_settings) |setting| {
        switch (doc.get(setting.key) orelse return error.MissingMetadata) {
            .float => |value| if (value != setting.value) return error.UnsupportedConfiguration,
            else => return error.InvalidMetadata,
        }
    }
    try expectCounts(doc, array_keys[0], heads);
    try expectCounts(doc, array_keys[1], kv_heads);
    // The pattern marks a sliding application true.
    for (try perApplication(doc, array_keys[2]), 0..) |value, application| switch (value) {
        .boolean => |sliding| if (sliding == isGlobal(application)) return error.UnsupportedConfiguration,
        else => return error.InvalidMetadata,
    };
    // The tokenizer covers exactly the unpadded rows (`vocab_size_unpadded`,
    // checked above); `bind` holds the head to `vocab_size`.
    const tokens = switch (doc.get("tokenizer.ggml.tokens") orelse return error.MissingMetadata) {
        .array => |a| a,
        else => return error.InvalidMetadata,
    };
    if (tokens.element_type != .string or tokens.count != vocabulary) return error.UnsupportedConfiguration;
    for (doc.metadata) |entry| {
        if (std.mem.startsWith(u8, entry.key, "inkling2.") and !knownArchitectureKey(entry.key))
            return error.UnsupportedConfiguration;
    }
}

fn knownArchitectureKey(key: []const u8) bool {
    for (integer_settings) |setting| if (std.mem.eql(u8, key, setting.key)) return true;
    for (float_settings) |setting| if (std.mem.eql(u8, key, setting.key)) return true;
    for (array_keys) |known| if (std.mem.eql(u8, key, known)) return true;
    return false;
}

const Storage = enum { f32, matrix };
const Binder = struct {
    remaining: std.StringHashMap(*const Tensor),
    tensors: u32 = 0,
    bytes: u64 = 0,

    fn take(self: *Binder, name: []const u8, dimensions: []const u64, storage: Storage) Error!*const Tensor {
        return try self.optional(name, dimensions, storage) orelse error.MissingTensor;
    }
    fn optional(self: *Binder, name: []const u8, dimensions: []const u64, storage: Storage) Error!?*const Tensor {
        const entry = self.remaining.fetchRemove(name) orelse return null;
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
    fn label(buffer: []u8, index: usize, suffix: []const u8) []const u8 {
        return std.fmt.bufPrint(buffer, "blk.{d}.{s}", .{ index, suffix }) catch unreachable;
    }
    fn weight(self: *Binder, index: usize, suffix: []const u8, dimensions: []const u64, storage: Storage) Error!*const Tensor {
        var name: [64]u8 = undefined;
        return self.take(label(&name, index, suffix), dimensions, storage);
    }
    fn maybe(self: *Binder, index: usize, suffix: []const u8, dimensions: []const u64, storage: Storage) Error!?*const Tensor {
        var name: [64]u8 = undefined;
        return self.optional(label(&name, index, suffix), dimensions, storage);
    }
    fn block(self: *Binder, index: usize) Error!Block {
        const d = embedding;
        const kind = kindOf(index);
        const input: Input = switch (kind) {
            .sliding => .{ .sliding = .{ .norm = try self.weight(index, "attn_norm.weight", &.{d}, .f32) } },
            .global => .{ .global = .{
                .query_norm = try self.weight(index, "attn_q_inp_norm.weight", &.{d}, .f32),
                .key_norm = try self.weight(index, "attn_k_inp_norm.weight", &.{d}, .f32),
                .value_norm = try self.weight(index, "attn_v_inp_norm.weight", &.{d}, .f32),
                .key_conv_projection = try self.weight(index, "attn_k_conv_proj.weight", &.{ d, conv_taps * kv_heads }, .matrix),
                .value_conv_projection = try self.weight(index, "attn_v_conv_proj.weight", &.{ d, conv_taps * kv_heads }, .matrix),
            } },
        };
        const extent: u64 = if (kind == .global) relative_extent else window;
        const residual_shape: []const u64 = if (kind == .global) &.{ d, 3 } else &.{d};
        const gate_up_scale = try self.maybe(index, "ffn_gate_up_exps.scale", &.{experts}, .f32);
        const down_scale = try self.maybe(index, "ffn_down_exps.scale", &.{experts}, .f32);
        if ((gate_up_scale == null) != (down_scale == null)) return error.MissingTensor;
        return .{
            .input = input,
            .query = try self.weight(index, "attn_q.weight", &.{ d, query_width }, .matrix),
            .key = try self.weight(index, "attn_k.weight", &.{ d, kv_width }, .matrix),
            .value = try self.weight(index, "attn_v.weight", &.{ d, kv_width }, .matrix),
            .gate = try self.weight(index, "attn_gate.weight", &.{ d, query_width }, .matrix),
            .output = try self.weight(index, "attn_output.weight", &.{ query_width, d }, .matrix),
            .query_clip = try self.weight(index, "attn_q_clip.weight", &.{ head_size, heads }, .f32),
            .sink_projection = try self.weight(index, "attn_sink_proj.weight", &.{ d, heads }, .matrix),
            .relative = try self.weight(index, "attn_rel.weight", &.{ d, relative_length * heads }, .matrix),
            .relative_bias = try self.weight(index, "attn_rel_bias.weight", &.{ relative_length, extent }, .f32),
            .relative_clip = try self.weight(index, "attn_rel_clip.weight", &.{ relative_length, heads }, .f32),
            .key_conv = try self.weight(index, "attn_k_conv.weight", &.{ conv_taps, kv_width }, .f32),
            .value_conv = try self.weight(index, "attn_v_conv.weight", &.{ conv_taps, kv_width }, .f32),
            .attention_post_conv = try self.weight(index, "attn_post_conv.weight", &.{ conv_taps, d }, .f32),
            .ffn_post_conv = try self.weight(index, "ffn_post_conv.weight", &.{ conv_taps, d }, .f32),
            .attention_residual_query = if (index == 0) null else try self.weight(index, "attn_res_query.weight", residual_shape, .f32),
            .ffn_residual_query = try self.weight(index, "ffn_res_query.weight", &.{d}, .f32),
            .ffn_norm = try self.weight(index, "ffn_norm.weight", &.{d}, .f32),
            .router = try self.weight(index, "ffn_gate_inp.weight", &.{ d, experts }, .f32),
            .router_bias = try self.weight(index, "exp_probs_b.bias", &.{experts}, .f32),
            .gate_up_experts = try self.weight(index, "ffn_gate_up_exps.weight", &.{ d, 2 * expert_feed_forward, experts }, .matrix),
            .down_experts = try self.weight(index, "ffn_down_exps.weight", &.{ expert_feed_forward, d, experts }, .matrix),
            .gate_up_scale = gate_up_scale,
            .down_scale = down_scale,
            .loop = if (index >= core_start and index < core_end) try self.loop(index) else null,
        };
    }
    fn loop(self: *Binder, index: usize) Error!Loop {
        const parts = [_][]const u8{ "attn", "k", "v", "mlp" };
        var result: Loop = undefined;
        inline for (parts, 0..) |part, m| {
            result.down[m] = try self.weight(index, "loop_" ++ part ++ "_down.weight", &.{ 2 * embedding, gate_rank }, .f32);
            result.gain[m] = try self.weight(index, "loop_" ++ part ++ "_gain.weight", &.{ gate_rank, embedding }, .f32);
            result.forget[m] = try self.weight(index, "loop_" ++ part ++ "_forget.weight", &.{ gate_rank, embedding }, .f32);
        }
        return result;
    }
};

/// doc must come from successful GGUF parsing. Allocations are temporary
/// lookup storage, freed before return; only the returned tensor references
/// borrow doc. Every tensor in the file must be claimed: an extra one (a
/// loop tensor outside the core, a residual query on block 0) is
/// `UnexpectedTensor`.
pub fn bind(alloc: std.mem.Allocator, doc: *const gguf.Document) Error!Binding {
    try validateMetadata(doc);
    var binder: Binder = .{ .remaining = .init(alloc) };
    defer binder.remaining.deinit();
    for (doc.tensors) |*tensor| {
        const previous = try binder.remaining.fetchPut(tensor.name, tensor);
        if (previous != null) return error.DuplicateTensor;
    }
    var result: Binding = undefined;
    result.token_embedding = try binder.take("token_embd.weight", &.{ embedding, padded_vocabulary }, .matrix);
    result.token_embedding_norm = try binder.take("token_embd_norm.weight", &.{embedding}, .f32);
    result.output = try binder.take("output.weight", &.{ embedding, padded_vocabulary }, .matrix);
    result.output_norm = try binder.take("output_norm.weight", &.{embedding}, .f32);
    result.output_residual_query = try binder.take("output_res_query.weight", &.{embedding}, .f32);
    result.loop_residual_query = try binder.take("loop_res_query.weight", &.{ embedding, loop_queries }, .f32);
    result.loop_kv_residual_query = try binder.take("loop_kv_res_query.weight", &.{ embedding, loop_queries }, .f32);
    for (&result.blocks, 0..) |*block, index| block.* = try binder.block(index);
    if (binder.remaining.count() != 0) return error.UnexpectedTensor;
    result.summary = .{
        .profile = profile,
        .decoder_layers = block_count,
        .layer_kinds = &layer_kinds,
        .text_tensors = binder.tensors,
        .auxiliary_tensors = 0,
        .text_tensor_bytes = binder.bytes,
        .auxiliary_tensor_bytes = 0,
        .inference_available = true,
        .metal_available = false,
    };
    return result;
}

/// The pinned artifact's directory inventory (`fixtures/inkling2-q4_k_xl.json`,
/// docs/models/inkling2.md) hydrated to a Document with no weights.
pub fn inventoryDocument(gpa: std.mem.Allocator) !gguf.Document {
    return @import("inventory.zig").document(gpa, @embedFile("fixtures/inkling2-q4_k_xl.json"));
}

test "the schedule: 48 applications over 24 stored blocks, global every sixth" {
    for (0..6) |a| try std.testing.expectEqual(a, storedBlock(a));
    for (0..3) |pass| for (6..18) |s| try std.testing.expectEqual(s, storedBlock(s + 12 * pass));
    for (42..48) |a| try std.testing.expectEqual(a - 24, storedBlock(a));
    var globals: usize = 0;
    for (0..block_count) |application| {
        if (isGlobal(application)) globals += 1;
        try std.testing.expectEqual(isGlobal(application), kindOf(storedBlock(application)) == .global);
    }
    try std.testing.expectEqual(@as(usize, 8), globals);
    try std.testing.expect(isGlobal(5) and isGlobal(47) and !isGlobal(0) and !isGlobal(42));
}

test "the pinned inventory binds: 734 tensors, the looped core, the padded head" {
    var doc = try inventoryDocument(std.testing.allocator);
    defer doc.deinit();
    const binding = try bind(std.testing.allocator, &doc);
    try std.testing.expectEqual(@as(u32, 734), binding.summary.text_tensors);
    try std.testing.expectEqual(@as(u32, 48), binding.summary.decoder_layers);
    try std.testing.expect(binding.summary.inference_available and !binding.summary.metal_available);
    try std.testing.expectEqual(@as(u64, 201024), binding.output.dimensions[1]);
    for (binding.blocks, 0..) |block, i| {
        try std.testing.expectEqual(kindOf(i), block.kind());
        try std.testing.expectEqual(i >= 6 and i < 18, block.loop != null);
        try std.testing.expectEqual(i == 0, block.attention_residual_query == null);
        // NVFP4 experts with their scales on 0–19, Q8_0 without on 20–23.
        try std.testing.expectEqual(i < 20, block.gate_up_scale != null and block.down_scale != null);
        try std.testing.expectEqual(@as(u32, if (i < 20) 40 else 8), block.gate_up_experts.encoding_id);
        try std.testing.expectEqual(@as(u32, 30), block.sink_projection.encoding_id);
        if (block.attention_residual_query) |q| try std.testing.expectEqual(@as(usize, if (kindOf(i) == .global) 2 else 1), q.dimensions.len);
    }
    try std.testing.expectEqual(Kind.global, binding.blocks[23].kind());
    try std.testing.expectEqualStrings("blk.17.loop_mlp_forget.weight", binding.blocks[17].loop.?.forget[3].name);
}

const Mutation = union(enum) {
    drop_tensor: []const u8,
    reshape: struct { name: []const u8, dimensions: []const u64 },
    encode: struct { name: []const u8, id: u32 },
    add_tensor: []const u8,
    set_unsigned: struct { key: []const u8, value: u64 },
    set_float: struct { key: []const u8, value: f64 },
    set_element: struct { key: []const u8, index: usize, value: gguf.Value },
    set_count: struct { key: []const u8, count: u64 },
    drop_key: []const u8,
    add_key: []const u8,
};

fn mutated(mutations: []const Mutation) !gguf.Document {
    var doc = try inventoryDocument(std.testing.allocator);
    errdefer doc.deinit();
    const alloc = doc.storage.allocator();
    // The Document's slices are const views; mutate owned copies in its arena.
    for (mutations) |mutation| {
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
                t.dimensions = try alloc.dupe(u64, r.dimensions);
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
            .set_element => |s| for (metadata) |*m| if (std.mem.eql(u8, m.key, s.key)) {
                const values = try alloc.dupe(gguf.Value, m.value.array.values.?);
                values[s.index] = s.value;
                m.value.array.values = values;
            },
            .set_count => |s| for (metadata) |*m| if (std.mem.eql(u8, m.key, s.key)) {
                m.value.array.count = s.count;
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
    }
    return doc;
}

test "deviations from the pinned configuration are typed rejections" {
    const cases = [_]struct { mutation: Mutation, expected: anyerror }{
        // The schedule and its arrays.
        .{ .mutation = .{ .set_unsigned = .{ .key = "inkling2.block_count", .value = 47 } }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .set_unsigned = .{ .key = "inkling2.num_loops", .value = 2 } }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .set_unsigned = .{ .key = "inkling2.loop.core_start", .value = 3 } }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .set_unsigned = .{ .key = "inkling2.loop.core_end", .value = 12 } }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .set_unsigned = .{ .key = "inkling2.attn_res.block_size", .value = 4 } }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .set_element = .{ .key = "inkling2.attention.head_count", .index = 30, .value = .{ .signed = 16 } } }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .set_element = .{ .key = "inkling2.attention.head_count_kv", .index = 0, .value = .{ .signed = 4 } } }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .set_element = .{ .key = "inkling2.attention.head_count_kv", .index = 0, .value = .{ .float = 8 } } }, .expected = error.InvalidMetadata },
        .{ .mutation = .{ .set_element = .{ .key = "inkling2.attention.sliding_window_pattern", .index = 5, .value = .{ .boolean = true } } }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .set_element = .{ .key = "inkling2.attention.sliding_window_pattern", .index = 0, .value = .{ .boolean = false } } }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .set_count = .{ .key = "inkling2.attention.sliding_window_pattern", .count = 24 } }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .drop_key = "inkling2.attention.head_count" }, .expected = error.MissingMetadata },
        // Geometry, windows, experts.
        .{ .mutation = .{ .set_unsigned = .{ .key = "inkling2.attention.sliding_window", .value = 512 } }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .set_unsigned = .{ .key = "inkling2.attention.relative_extent", .value = 2048 } }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .set_unsigned = .{ .key = "inkling2.attention.log_scale_floor", .value = 4096 } }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .set_unsigned = .{ .key = "inkling2.expert_used_count", .value = 4 } }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .set_unsigned = .{ .key = "inkling2.rope.dimension_count", .value = 64 } }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .set_float = .{ .key = "inkling2.attention.qk_clip", .value = 16 } }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .set_float = .{ .key = "inkling2.attention.log_scale_alpha", .value = 0.2 } }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .set_float = .{ .key = "inkling2.final_logit_softcapping", .value = 20 } }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .set_unsigned = .{ .key = "inkling2.activation.situ_beta", .value = 4 } }, .expected = error.InvalidMetadata },
        .{ .mutation = .{ .drop_key = "inkling2.attention.layer_norm_rms_epsilon" }, .expected = error.MissingMetadata },
        .{ .mutation = .{ .add_key = "inkling2.rope.freq_base" }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .set_unsigned = .{ .key = "general.quantization_version", .value = 1 } }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .drop_key = "general.architecture" }, .expected = error.MissingMetadata },
        // The vocabulary rule: tokens = unpadded ≤ padded = the head's rows.
        .{ .mutation = .{ .set_unsigned = .{ .key = "inkling2.vocab_size_unpadded", .value = 200000 } }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .set_count = .{ .key = "tokenizer.ggml.tokens", .count = 201024 } }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .set_unsigned = .{ .key = "inkling2.vocab_size", .value = 200058 } }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .reshape = .{ .name = "output.weight", .dimensions = &.{ 2304, 200058 } } }, .expected = error.InvalidTensorShape },
        // Tensors: the core's controllers, the kinds' inputs, the residual queries.
        .{ .mutation = .{ .drop_tensor = "blk.6.loop_k_gain.weight" }, .expected = error.MissingTensor },
        .{ .mutation = .{ .add_tensor = "blk.18.loop_k_gain.weight" }, .expected = error.UnexpectedTensor },
        .{ .mutation = .{ .add_tensor = "blk.0.attn_res_query.weight" }, .expected = error.UnexpectedTensor },
        .{ .mutation = .{ .add_tensor = "blk.5.attn_norm.weight" }, .expected = error.UnexpectedTensor },
        .{ .mutation = .{ .drop_tensor = "blk.11.attn_v_inp_norm.weight" }, .expected = error.MissingTensor },
        .{ .mutation = .{ .drop_tensor = "blk.4.attn_norm.weight" }, .expected = error.MissingTensor },
        .{ .mutation = .{ .reshape = .{ .name = "blk.5.attn_res_query.weight", .dimensions = &.{2304} } }, .expected = error.InvalidTensorShape },
        .{ .mutation = .{ .reshape = .{ .name = "blk.4.attn_res_query.weight", .dimensions = &.{ 2304, 3 } } }, .expected = error.InvalidTensorShape },
        .{ .mutation = .{ .reshape = .{ .name = "loop_res_query.weight", .dimensions = &.{ 2304, 12 } } }, .expected = error.InvalidTensorShape },
        .{ .mutation = .{ .reshape = .{ .name = "blk.5.attn_rel_bias.weight", .dimensions = &.{ 16, 512 } } }, .expected = error.InvalidTensorShape },
        .{ .mutation = .{ .drop_tensor = "blk.3.ffn_down_exps.scale" }, .expected = error.MissingTensor },
        .{ .mutation = .{ .add_tensor = "rope_freqs.weight" }, .expected = error.UnexpectedTensor },
        // Encodings.
        .{ .mutation = .{ .encode = .{ .name = "blk.0.ffn_gate_up_exps.weight", .id = 12 } }, .expected = error.UnsupportedTensorEncoding },
        .{ .mutation = .{ .encode = .{ .name = "blk.0.attn_q_clip.weight", .id = 1 } }, .expected = error.UnsupportedTensorEncoding },
        .{ .mutation = .{ .encode = .{ .name = "blk.3.ffn_down_exps.scale", .id = 30 } }, .expected = error.UnsupportedTensorEncoding },
    };
    for (cases) |case| {
        errdefer std.debug.print("mutation {any}\n", .{case.mutation});
        var doc = try mutated(&.{case.mutation});
        defer doc.deinit();
        try std.testing.expectError(case.expected, bind(std.testing.allocator, &doc));
    }
}

test "a block without expert scales binds without them" {
    var doc = try mutated(&.{ .{ .drop_tensor = "blk.3.ffn_gate_up_exps.scale" }, .{ .drop_tensor = "blk.3.ffn_down_exps.scale" } });
    defer doc.deinit();
    const binding = try bind(std.testing.allocator, &doc);
    try std.testing.expect(binding.blocks[3].gate_up_scale == null and binding.blocks[3].down_scale == null);
    try std.testing.expect(binding.blocks[2].gate_up_scale != null);
    try std.testing.expectEqual(@as(u32, 732), binding.summary.text_tensors);
}

test "binding survives allocation failure without leaks" {
    var doc = try inventoryDocument(std.testing.allocator);
    defer doc.deinit();
    try alloc_check.checkAll(std.testing.allocator, struct {
        fn check(alloc: std.mem.Allocator, d: *const gguf.Document) !void {
            _ = try bind(alloc, d);
        }
    }.check, .{&doc});
}
