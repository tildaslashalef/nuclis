//! Gemma 4 assistant head adapter: the metrics of the companion GGUF
//! (`general.architecture = gemma4-assistant`) that Gemma 4 targets load as
//! their draft source. Four trained heads, one per proposed position, that
//! read the target's sliding (layer `n_layer − 2`) and global (layer
//! `n_layer − 1`) key/value caches and write neither; the facts and their
//! provenance are in docs/engine/speculative-decoding.md § The Gemma 4
//! assistant heads. Three pinned heads, keyed by their target's width (the
//! 12B's 3840, the 26B-A4B's 2816, the E4B's 2560); the E4B's is also
//! narrower inside (`Config`). The binder is deliberately narrow: a file
//! whose keys differ from the pinned configuration is rejected with
//! `UnsupportedConfiguration` rather than run with guessed semantics.
const std = @import("std");
const gguf = @import("../formats/gguf.zig");
const models = @import("root.zig");
const weights = @import("../runtime/weights.zig");
const Tensor = gguf.Tensor;

pub const architecture = "gemma4-assistant";
pub const block_count = 4;
/// The widest head of any pinned configuration; workspaces size by these.
pub const max_heads = 16;
pub const max_embedding = 1024;
pub const max_feed_forward = 8192;
pub const vocabulary = 262144;
pub const rms_epsilon: f32 = 1e-6;

/// The per-block geometry of one pinned head width.
pub const Config = struct {
    /// The adapter's name for the configuration (`Summary.profile`).
    profile: []const u8,
    /// The target's hidden width (`embedding_length_out`).
    embedding_out: usize,
    /// The head's own residual width and FFN width.
    embedding: usize = 1024,
    feed_forward: usize = 8192,
    heads: usize = 16,
    /// Positions a sliding block's mask admits (the target's window).
    window: usize = 1024,
    /// The target's KV heads per layer kind, mirrored by the head's blocks.
    sliding_kv_heads: usize = 8,
    global_kv_heads: usize,
    context_length: usize,
};
pub const config_12b: Config = .{ .profile = "gemma4_assistant_12b", .embedding_out = 3840, .global_kv_heads = 1, .context_length = 262144 };
pub const config_26b_a4b: Config = .{ .profile = "gemma4_assistant_26b_a4b", .embedding_out = 2816, .global_kv_heads = 2, .context_length = 131072 };
pub const config_e4b: Config = .{ .profile = "gemma4_assistant_e4b", .embedding_out = 2560, .embedding = 256, .feed_forward = 2048, .heads = 4, .window = 512, .sliding_kv_heads = 2, .global_kv_heads = 2, .context_length = 131072 };
pub const configs = [_]*const Config{ &config_12b, &config_26b_a4b, &config_e4b };

/// Layers 0–2 are sliding, layer 3 global, the pattern `[1,1,1,0]`.
pub const Kind = enum {
    sliding,
    global,

    pub fn headSize(self: Kind) usize {
        return switch (self) {
            .sliding => 256,
            .global => 512,
        };
    }
    pub fn ropeBase(self: Kind) f32 {
        return switch (self) {
            .sliding => 10_000,
            .global => 1_000_000,
        };
    }
    /// The widest query projection of this kind (`max_heads` heads).
    pub fn queryWidth(self: Kind) usize {
        return max_heads * self.headSize();
    }
};
pub fn kindOf(index: usize) Kind {
    return if (index == block_count - 1) .global else .sliding;
}
pub fn kvHeadsOf(config: *const Config, kind: Kind) usize {
    return switch (kind) {
        .sliding => config.sliding_kv_heads,
        .global => config.global_kv_heads,
    };
}
/// The same storage layouts the main Gemma 4 adapter executes.
pub const executableEncoding = @import("gemma4.zig").executableEncoding;

pub const Error = models.BindError;

pub const Summary = models.Summary;

pub const Layer = struct {
    kind: Kind,
    heads: usize,
    kv_heads: usize,
    attention_norm: *const Tensor,
    post_attention_norm: *const Tensor,
    ffn_norm: *const Tensor,
    post_ffn_norm: *const Tensor,
    query: *const Tensor,
    query_norm: *const Tensor,
    output: *const Tensor,
    ffn_gate: *const Tensor,
    ffn_up: *const Tensor,
    ffn_down: *const Tensor,
    /// One F32 scalar multiplying the block's whole output.
    output_scale: *const Tensor,

    pub fn headSize(self: Layer) usize {
        return self.kind.headSize();
    }
    pub fn queryWidth(self: Layer) usize {
        return self.heads * self.kind.headSize();
    }
    pub fn kvWidth(self: Layer) usize {
        return self.kv_heads * self.kind.headSize();
    }
};

/// All tensor pointers borrow doc.tensors, and `view` is the byte source
/// they were bound from (the companion file's, not the target's). No weights
/// are read or allocated; keep the source Document and mapping alive for the
/// entire binding lifetime.
pub const Binding = struct {
    config: *const Config,
    view: weights.View,
    /// The head's own tied classifier over the vocabulary (`token_embd`);
    /// the *input* embedding is the target's.
    classifier: *const Tensor,
    output_norm: *const Tensor,
    pre_projection: *const Tensor,
    post_projection: *const Tensor,
    /// 256 per-pair RoPE frequency factors for the global block.
    rope_factors: *const Tensor,
    layers: [block_count]Layer,
    summary: Summary,
};

const IntegerSetting = struct { key: []const u8, value: u64 };
const shared_integer_settings = [_]IntegerSetting{
    .{ .key = "gemma4-assistant.block_count", .value = block_count },
    .{ .key = "gemma4-assistant.attention.key_length", .value = 512 },
    .{ .key = "gemma4-assistant.attention.value_length", .value = 512 },
    .{ .key = "gemma4-assistant.attention.key_length_swa", .value = 256 },
    .{ .key = "gemma4-assistant.attention.value_length_swa", .value = 256 },
    .{ .key = "gemma4-assistant.rope.dimension_count", .value = 512 },
    .{ .key = "gemma4-assistant.rope.dimension_count_swa", .value = 256 },
    .{ .key = "gemma4-assistant.attention.shared_kv_layers", .value = block_count },
    .{ .key = "gemma4-assistant.nextn_predict_layers", .value = block_count },
    .{ .key = "gemma4-assistant.embedding_length_per_layer_input", .value = 0 },
};
const FloatSetting = struct { key: []const u8, value: f64 };
const float_settings = [_]FloatSetting{
    .{ .key = "gemma4-assistant.rope.freq_base", .value = 1_000_000 },
    .{ .key = "gemma4-assistant.rope.freq_base_swa", .value = 10_000 },
    .{ .key = "gemma4-assistant.attention.layer_norm_rms_epsilon", .value = @as(f32, 1e-6) },
};
const array_settings = [_][]const u8{ "gemma4-assistant.attention.head_count_kv", "gemma4-assistant.attention.sliding_window_pattern" };
/// Keys whose value is the configuration's.
const per_config_keys = [_][]const u8{
    "gemma4-assistant.embedding_length",
    "gemma4-assistant.feed_forward_length",
    "gemma4-assistant.attention.head_count",
    "gemma4-assistant.attention.sliding_window",
};

/// Integer metadata regardless of the writer's chosen width or signedness.
fn integer(doc: *const gguf.Document, key: []const u8) Error!u64 {
    return switch (doc.get(key) orelse return error.MissingMetadata) {
        .unsigned => |n| n,
        .signed => |n| if (n < 0) error.InvalidMetadata else @intCast(n),
        else => error.InvalidMetadata,
    };
}
fn expectInteger(doc: *const gguf.Document, key: []const u8, value: u64) Error!void {
    if (try integer(doc, key) != value) return error.UnsupportedConfiguration;
}
fn configFor(doc: *const gguf.Document) Error!*const Config {
    const out = try integer(doc, "gemma4-assistant.embedding_length_out");
    for (configs) |config| if (config.embedding_out == out) {
        try expectInteger(doc, "gemma4-assistant.context_length", config.context_length);
        return config;
    };
    return error.UnsupportedConfiguration;
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

fn validateMetadata(doc: *const gguf.Document) Error!*const Config {
    const arch = doc.string("general.architecture") orelse return error.MissingMetadata;
    if (!std.mem.eql(u8, arch, architecture)) return error.UnsupportedArchitecture;
    const config = try configFor(doc);
    for (shared_integer_settings) |setting| try expectInteger(doc, setting.key, setting.value);
    try expectInteger(doc, per_config_keys[0], config.embedding);
    try expectInteger(doc, per_config_keys[1], config.feed_forward);
    try expectInteger(doc, per_config_keys[2], config.heads);
    try expectInteger(doc, per_config_keys[3], config.window);
    for (float_settings) |setting| {
        switch (doc.get(setting.key) orelse return error.MissingMetadata) {
            .float => |value| if (value != setting.value) return error.UnsupportedConfiguration,
            else => return error.InvalidMetadata,
        }
    }
    // The KV heads are the configuration's (an array, or one scalar when
    // every block has the same count), and the sliding pattern must be the
    // block_count-long `[1,1,1,0]`.
    const swa = try retained(doc, array_settings[1], block_count);
    const kv: ?[]const gguf.Value = switch (doc.get(array_settings[0]) orelse return error.MissingMetadata) {
        .array => try retained(doc, array_settings[0], block_count),
        else => null,
    };
    const uniform: ?u64 = if (kv == null) try integer(doc, array_settings[0]) else null;
    for (swa, 0..) |sliding, index| {
        const kind = kindOf(index);
        const n: u64 = if (kv) |values| switch (values[index]) {
            .signed => |v| if (v < 0) return error.InvalidMetadata else @intCast(v),
            .unsigned => |v| v,
            else => return error.InvalidMetadata,
        } else uniform.?;
        if (n != kvHeadsOf(config, kind)) return error.UnsupportedConfiguration;
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
    for (doc.metadata) |entry| {
        if (std.mem.startsWith(u8, entry.key, "gemma4-assistant.") and !knownArchitectureKey(entry.key))
            return error.UnsupportedConfiguration;
    }
    return config;
}

fn knownArchitectureKey(key: []const u8) bool {
    for (shared_integer_settings) |setting| if (std.mem.eql(u8, key, setting.key)) return true;
    for (per_config_keys) |known| if (std.mem.eql(u8, key, known)) return true;
    for (float_settings) |setting| if (std.mem.eql(u8, key, setting.key)) return true;
    for (array_settings) |known| if (std.mem.eql(u8, key, known)) return true;
    if (std.mem.eql(u8, key, "gemma4-assistant.embedding_length_out")) return true;
    if (std.mem.eql(u8, key, "gemma4-assistant.context_length")) return true;
    return false;
}

const Storage = enum { f32, matrix };
const Binder = struct {
    remaining: std.StringHashMap(*const Tensor),
    config: *const Config,
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
    fn layer(self: *Binder, index: usize) Error!Layer {
        const kind = kindOf(index);
        const hd = kind.headSize();
        const q = self.config.heads * hd;
        const embedding = self.config.embedding;
        const feed_forward = self.config.feed_forward;
        return .{
            .kind = kind,
            .heads = self.config.heads,
            .kv_heads = kvHeadsOf(self.config, kind),
            .attention_norm = try self.weight(index, "attn_norm.weight", &.{embedding}, .f32),
            .post_attention_norm = try self.weight(index, "post_attention_norm.weight", &.{embedding}, .f32),
            .ffn_norm = try self.weight(index, "ffn_norm.weight", &.{embedding}, .f32),
            .post_ffn_norm = try self.weight(index, "post_ffw_norm.weight", &.{embedding}, .f32),
            .query = try self.weight(index, "attn_q.weight", &.{ embedding, q }, .matrix),
            .query_norm = try self.weight(index, "attn_q_norm.weight", &.{hd}, .f32),
            .output = try self.weight(index, "attn_output.weight", &.{ q, embedding }, .matrix),
            .ffn_gate = try self.weight(index, "ffn_gate.weight", &.{ embedding, feed_forward }, .matrix),
            .ffn_up = try self.weight(index, "ffn_up.weight", &.{ embedding, feed_forward }, .matrix),
            .ffn_down = try self.weight(index, "ffn_down.weight", &.{ feed_forward, embedding }, .matrix),
            .output_scale = try self.weight(index, "layer_output_scale.weight", &.{1}, .f32),
        };
    }
    fn weight(self: *Binder, index: usize, suffix: []const u8, dimensions: []const u64, storage: Storage) Error!*const Tensor {
        var name: [128]u8 = undefined;
        const key = std.fmt.bufPrint(&name, "blk.{d}.{s}", .{ index, suffix }) catch unreachable;
        return self.take(key, dimensions, storage);
    }
};

/// doc must come from successful GGUF parsing. Allocations are temporary
/// lookup storage, freed before return; only the returned tensor references
/// borrow doc. Every tensor in the file must be claimed.
pub fn bind(alloc: std.mem.Allocator, doc: *const gguf.Document) Error!Binding {
    const config = try validateMetadata(doc);
    var binder: Binder = .{ .remaining = .init(alloc), .config = config };
    defer binder.remaining.deinit();
    for (doc.tensors) |*tensor| {
        const previous = try binder.remaining.fetchPut(tensor.name, tensor);
        if (previous != null) return error.DuplicateTensor;
    }
    var result: Binding = undefined;
    result.config = config;
    result.classifier = try binder.take("token_embd.weight", &.{ config.embedding, vocabulary }, .matrix);
    result.output_norm = try binder.take("output_norm.weight", &.{config.embedding}, .f32);
    result.pre_projection = try binder.take("nextn.pre_projection.weight", &.{ 2 * config.embedding_out, config.embedding }, .matrix);
    result.post_projection = try binder.take("nextn.post_projection.weight", &.{ config.embedding, config.embedding_out }, .matrix);
    result.rope_factors = try binder.take("rope_freqs.weight", &.{256}, .f32);
    for (&result.layers, 0..) |*layer, index| layer.* = try binder.layer(index);
    if (binder.remaining.count() != 0) return error.UnexpectedTensor;
    result.summary = .{
        .profile = config.profile,
        .decoder_layers = block_count,
        .layer_kinds = &.{
            .{ .kind = "sliding_attention", .count = 3 },
            .{ .kind = "global_attention", .count = 1 },
        },
        .auxiliary_prediction_layers = block_count,
        .text_tensors = binder.tensors,
        .auxiliary_tensors = 0,
        .text_tensor_bytes = binder.bytes,
        .auxiliary_tensor_bytes = 0,
    };
    return result;
}

test "the pinned 12B head binds: 49 tensors, three sliding blocks then the global one" {
    var doc = try @import("inventory.zig").document(std.testing.allocator, @embedFile("fixtures/gemma4-head-12b.json"));
    defer doc.deinit();
    const binding = try bind(std.testing.allocator, &doc);
    try std.testing.expectEqual(&config_12b, binding.config);
    try std.testing.expectEqual(@as(u32, 49), binding.summary.text_tensors);
    try std.testing.expectEqual(@as(u64, 237_922_320), binding.summary.text_tensor_bytes);
    for (&binding.layers, 0..) |layer, i| {
        try std.testing.expectEqual(kindOf(i), layer.kind);
        try std.testing.expectEqual(kvHeadsOf(&config_12b, layer.kind), layer.kv_heads);
    }
    try std.testing.expectEqual(@as(usize, 4096), binding.layers[0].queryWidth());
    try std.testing.expectEqual(@as(usize, 8192), binding.layers[3].queryWidth());
    try std.testing.expectEqual(@as(u64, 262144), binding.classifier.dimensions[1]);
    try std.testing.expectEqual(@as(u64, 3840), binding.post_projection.dimensions[1]);
}

test "the pinned E4B head binds: 256 wide, four query heads, two KV heads on every block" {
    var doc = try @import("inventory.zig").document(std.testing.allocator, @embedFile("fixtures/gemma4-head-e4b.json"));
    defer doc.deinit();
    const binding = try bind(std.testing.allocator, &doc);
    try std.testing.expectEqual(&config_e4b, binding.config);
    try std.testing.expectEqual(@as(u32, 49), binding.summary.text_tensors);
    for (&binding.layers, 0..) |layer, i| {
        try std.testing.expectEqual(kindOf(i), layer.kind);
        try std.testing.expectEqual(@as(usize, 2), layer.kv_heads);
    }
    try std.testing.expectEqual(@as(usize, 1024), binding.layers[0].queryWidth());
    try std.testing.expectEqual(@as(usize, 2048), binding.layers[3].queryWidth());
    try std.testing.expectEqualSlices(u64, &.{ 5120, 256 }, binding.pre_projection.dimensions);
}

test "deviations from the pinned head are typed rejections" {
    const Mutation = union(enum) {
        drop_tensor: []const u8,
        add_tensor: []const u8,
        set_out: u64,
        set_kv: u64,
        drop_key: []const u8,
    };
    const cases = [_]struct { mutation: Mutation, expected: anyerror }{
        .{ .mutation = .{ .drop_tensor = "blk.2.attn_q_norm.weight" }, .expected = error.MissingTensor },
        .{ .mutation = .{ .drop_tensor = "nextn.pre_projection.weight" }, .expected = error.MissingTensor },
        .{ .mutation = .{ .add_tensor = "blk.4.attn_q.weight" }, .expected = error.UnexpectedTensor },
        .{ .mutation = .{ .set_out = 5120 }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .set_kv = 8 }, .expected = error.UnsupportedConfiguration },
        .{ .mutation = .{ .drop_key = "gemma4-assistant.attention.shared_kv_layers" }, .expected = error.MissingMetadata },
        .{ .mutation = .{ .drop_key = "gemma4-assistant.nextn_predict_layers" }, .expected = error.MissingMetadata },
    };
    for (cases) |case| {
        var doc = try @import("inventory.zig").document(std.testing.allocator, @embedFile("fixtures/gemma4-head-12b.json"));
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
                const extra: Tensor = .{ .name = name, .dimensions = try alloc.dupe(u64, &.{ 1024, 1024 }), .encoding_id = 0, .offset = 0, .elements = 1024 * 1024, .bytes = 1024 * 1024 * 4 };
                const more = try alloc.alloc(Tensor, doc.tensors.len + 1);
                @memcpy(more[0..doc.tensors.len], doc.tensors);
                more[doc.tensors.len] = extra;
                doc.tensors = more;
            },
            .set_out => |value| for (metadata) |*m| if (std.mem.eql(u8, m.key, "gemma4-assistant.embedding_length_out")) {
                m.value = .{ .unsigned = value };
            },
            .set_kv => |value| for (metadata) |*m| if (std.mem.eql(u8, m.key, "gemma4-assistant.attention.head_count_kv")) {
                const values = try alloc.dupe(gguf.Value, m.value.array.values.?);
                values[3] = .{ .signed = @intCast(value) };
                m.value.array.values = values;
            },
            .drop_key => |key| {
                var kept: std.ArrayList(gguf.Metadata) = .empty;
                for (doc.metadata) |m| if (!std.mem.eql(u8, m.key, key)) try kept.append(alloc, m);
                doc.metadata = try kept.toOwnedSlice(alloc);
            },
        }
        try std.testing.expectError(case.expected, bind(std.testing.allocator, &doc));
    }
}
