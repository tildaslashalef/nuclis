//! Generated random-weight models for the language families. Test support:
//! only `inference/tiny-check.zig` reaches this, through the module instance
//! built with `tiny_models`, whose adapters pin the small shapes these files
//! declare. A file goes through the real parser, validators, tokenizer, and
//! executors (docs/models/README.md § Tiny models). It proves the load and
//! forward paths, and Metal against the CPU reference; never accuracy.
//!
//! A new variant is one row of `variants`, plus an `emit` for a new family:
//! it declares metadata and tensors on a `Builder`, weights follow from the
//! seed and the tensor's name, so a file is a pure function of (variant,
//! matrix encoding, seed).
const std = @import("std");
const gguf = @import("../formats/gguf.zig");
const gguf_writer = @import("../formats/gguf_writer.zig");
const sample = @import("../quant/sample.zig");
const layouts = @import("../tensor/encoding.zig");
const qwen35 = @import("qwen35.zig");
const muse = @import("muse_glimmer.zig");
const gemma4 = @import("gemma4.zig");
const Allocator = std.mem.Allocator;

pub const Error = gguf_writer.Error || sample.Error;

/// How a tensor's bytes follow from the seed.
const Weight = union(enum) {
    /// Zero-mean noise of this RMS, in the tensor's encoding.
    noise: f32,
    /// F32 around 1: a norm's weight.
    gain,
    /// F32 uniform over the range.
    uniform: struct { low: f32, high: f32 },
    /// F32 values given by the caller, little endian.
    values: []const f32,
};

pub const Builder = struct {
    /// Owns everything the builder holds; freed with the image.
    arena: Allocator,
    /// The file's matrix encoding: what `matrix` stores.
    encoding: u32,
    /// When not empty, `matrix` walks these in turn instead: a real file's mix.
    cycle: []const u32 = &.{},
    cycled: usize = 0,
    metadata: std.ArrayList(gguf_writer.Entry) = .empty,
    tensors: std.ArrayList(gguf_writer.Tensor) = .empty,
    weights: std.ArrayList(Weight) = .empty,

    pub fn set(self: *Builder, key: []const u8, value: gguf_writer.Value) Allocator.Error!void {
        try self.metadata.append(self.arena, .{ .key = key, .value = value });
    }

    pub fn setf(self: *Builder, comptime fmt: []const u8, args: anytype, value: gguf_writer.Value) Allocator.Error!void {
        try self.set(try std.fmt.allocPrint(self.arena, fmt, args), value);
    }

    fn tensor(self: *Builder, name: []const u8, dimensions: []const u64, encoding: u32, weight: Weight) Allocator.Error!void {
        try self.tensors.append(self.arena, .{ .name = name, .dimensions = try self.arena.dupe(u64, dimensions), .encoding = encoding });
        try self.weights.append(self.arena, weight);
    }

    /// A weight matrix in the file's encoding. Its input width sets the noise,
    /// so a product roughly preserves the size of the activation.
    pub fn matrix(self: *Builder, name: []const u8, dimensions: []const u64) Allocator.Error!void {
        if (self.cycle.len == 0) return self.encoded(name, dimensions, self.encoding);
        const index = self.cycled % self.cycle.len;
        self.cycled += 1;
        try self.encoded(name, dimensions, self.cycle[index]);
    }

    /// A matrix in a named encoding (a real file's mix).
    pub fn encoded(self: *Builder, name: []const u8, dimensions: []const u64, encoding: u32) Allocator.Error!void {
        const width: f32 = @floatFromInt(dimensions[0]);
        try self.tensor(name, dimensions, encoding, .{ .noise = 1 / @sqrt(width) });
    }

    /// An F32 tensor around 1.
    pub fn gain(self: *Builder, name: []const u8, dimensions: []const u64) Allocator.Error!void {
        try self.tensor(name, dimensions, 0, .gain);
    }

    /// An F32 tensor of zero-mean noise.
    pub fn noise(self: *Builder, name: []const u8, dimensions: []const u64, rms: f32) Allocator.Error!void {
        try self.tensor(name, dimensions, 0, .{ .noise = rms });
    }

    /// An F32 tensor uniform over `[low, high)`.
    pub fn uniform(self: *Builder, name: []const u8, dimensions: []const u64, low: f32, high: f32) Allocator.Error!void {
        try self.tensor(name, dimensions, 0, .{ .uniform = .{ .low = low, .high = high } });
    }

    /// An F32 tensor of given values; borrowed, keep them in `arena`.
    pub fn values(self: *Builder, name: []const u8, dimensions: []const u64, data: []const f32) Allocator.Error!void {
        try self.tensor(name, dimensions, 0, .{ .values = data });
    }

    fn label(self: *Builder, comptime fmt: []const u8, args: anytype) Allocator.Error![]const u8 {
        return std.fmt.allocPrint(self.arena, fmt, args);
    }
};

pub const Variant = struct {
    name: []const u8,
    emit: *const fn (*Builder, seed: u64) Allocator.Error!void,
    /// The matrix encodings the Metal plan is checked on, one file each. The
    /// CPU test runs the first; the check runs them all.
    encodings: []const u32,
    /// A mix: one file whose matrices walk these encodings, so a fused pair
    /// meets two layouts. Checked on Metal only; `encodings` then names it.
    cycle: []const u32 = &.{},
};

/// The shapes that mirror a released file: qwen35's are the smallest the
/// adapter accepts from the file; the others are the pinned tiny constants.
pub const variants = [_]Variant{
    .{ .name = "qwen35", .emit = emitQwen35, .encodings = &.{ 8, 0, 11, 12, 13, 14, 20, 21, 23, 30, 142, 143 } },
    // Saluki's IQ2 mix: one file per encoding the mix uses.
    .{ .name = "saluki", .emit = emitQwen35, .encodings = &.{ 18, 10, 16, 17, 19, 22, 29 } },
    // Bonsai: the rotated basis at the shape the rotation contract pins, its
    // two ternary files with the BF16 `ssm_alpha` / `ssm_beta` pair.
    .{ .name = "bonsai", .emit = emitBonsai, .encodings = &.{ 142, 143 } },
    .{ .name = "muse-glimmer", .emit = emitMuse, .encodings = &.{ 12, 0, 2, 8, 11, 13, 14, 20, 21, 23 } },
    .{ .name = "gemma4-dense", .emit = emitGemmaDense, .encodings = &.{ 12, 0, 2, 6, 7, 8, 11, 13, 14, 20, 21, 23 } },
    .{ .name = "gemma4-experts", .emit = emitGemmaExperts, .encodings = &.{ 2, 0, 6, 7, 8, 11, 12, 13, 14, 20, 21, 23 } },
    .{ .name = "gemma4-e4b", .emit = emitGemmaE4b, .encodings = &.{ 2, 0, 6, 7, 8, 11, 12, 13, 14, 20, 21, 23 } },
    .{ .name = "saluki-mix", .emit = emitQwen35, .encodings = &.{16}, .cycle = &.{ 16, 10, 17, 18, 19, 22, 29 } },
    .{ .name = "muse-mix", .emit = emitMuse, .encodings = &.{12}, .cycle = &.{ 12, 13 } },
    .{ .name = "gemma4-mix", .emit = emitGemmaDense, .encodings = &.{12}, .cycle = &.{ 12, 13, 2, 8 } },
};

pub fn find(name: []const u8) ?*const Variant {
    for (&variants) |*variant| if (std.mem.eql(u8, variant.name, name)) return variant;
    return null;
}

/// The file for `variant` in `encoding`. Caller owns the bytes. Two calls
/// with the same arguments return the same bytes.
pub fn generate(gpa: Allocator, variant: *const Variant, encoding: u32, seed: u64) Error![]u8 {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var builder: Builder = .{ .arena = arena.allocator(), .encoding = encoding, .cycle = variant.cycle };
    try variant.emit(&builder, seed);
    const context: FillContext = .{ .builder = &builder, .seed = seed };
    return gguf_writer.image(gpa, builder.metadata.items, builder.tensors.items, context, fill);
}

/// `variant`'s directory parsed, weights never generated: what the validators
/// read of a file.
fn directoryDocument(gpa: Allocator, variant: *const Variant, encoding: u32) !gguf.Document {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var builder: Builder = .{ .arena = arena.allocator(), .encoding = encoding };
    try variant.emit(&builder, 0);
    const head = try gguf_writer.directory(gpa, builder.metadata.items, builder.tensors.items);
    defer gpa.free(head.bytes);
    var reader: std.Io.Reader = .fixed(head.bytes);
    return gguf.parse(gpa, &reader, head.file_bytes, .{});
}

const FillContext = struct { builder: *const Builder, seed: u64 };

fn fill(context: FillContext, index: usize, bytes: []u8) void {
    const tensor = context.builder.tensors.items[index];
    var prng: std.Random.DefaultPrng = .init(std.hash.Wyhash.hash(context.seed, tensor.name));
    const random = prng.random();
    switch (context.builder.weights.items[index]) {
        // The shapes are ours and the encodings the table's: a refusal here
        // is a bad table row, not an input to report.
        .noise => |rms| sample.row(tensor.encoding, random, rms, bytes) catch unreachable,
        .gain => for (0..bytes.len / 4) |i| {
            const value = 1 + 0.1 * random.floatNorm(f32);
            std.mem.writeInt(u32, bytes[i * 4 ..][0..4], @bitCast(value), .little);
        },
        .uniform => |range| for (0..bytes.len / 4) |i| {
            const value = range.low + (range.high - range.low) * random.float(f32);
            std.mem.writeInt(u32, bytes[i * 4 ..][0..4], @bitCast(value), .little);
        },
        .values => |data| for (data, 0..) |value, i| {
            std.mem.writeInt(u32, bytes[i * 4 ..][0..4], @bitCast(value), .little);
        },
    }
}

// --- tokenizers -------------------------------------------------------

/// Ranks in file order. `3 4` and `o W` come first so a pair the splitter
/// must keep apart would merge if it did not: `gpt4o` cuts digit runs every
/// three and a word at a new capital.
const byte_level_merges = [_][2][]const u8{
    .{ "3", "4" },           .{ "o", "W" },       .{ "H", "e" },        .{ "He", "l" },        .{ "Hel", "l" },
    .{ "Hell", "o" },        .{ "\u{120}", "w" }, .{ "\u{120}w", "o" }, .{ "\u{120}wo", "r" }, .{ "\u{120}wor", "l" },
    .{ "\u{120}worl", "d" }, .{ "1", "2" },       .{ "12", "3" },       .{ "'", "s" },
};

const spm_merges = [_][2][]const u8{
    .{ "H", "e" },         .{ "He", "l" },         .{ "Hel", "l" },         .{ "Hell", "o" },         .{ "\u{2581}", "w" },
    .{ "\u{2581}w", "o" }, .{ "\u{2581}wo", "r" }, .{ "\u{2581}wor", "l" }, .{ "\u{2581}worl", "d" },
};

pub const end_marker = "<|endoftext|>";

/// GPT-2's byte alphabet: printable bytes stand for themselves, the rest
/// take code points from U+0100 up, in byte order.
fn byteLetter(byte: u8) u21 {
    const printable = (byte >= '!' and byte <= '~') or (byte >= 0xa1 and byte <= 0xac) or byte >= 0xae;
    if (printable) return byte;
    var skipped: u21 = 0;
    for (0..byte) |b| {
        if (!((b >= '!' and b <= '~') or (b >= 0xa1 and b <= 0xac) or b >= 0xae)) skipped += 1;
    }
    return 256 + skipped;
}

fn utf8(arena: Allocator, code_point: u21) Allocator.Error![]const u8 {
    var buffer: [4]u8 = undefined;
    const len = std.unicode.utf8Encode(code_point, &buffer) catch unreachable;
    return arena.dupe(u8, buffer[0..len]);
}

fn mergeStrings(arena: Allocator, merges: []const [2][]const u8) Allocator.Error![]const []const u8 {
    const out = try arena.alloc([]const u8, merges.len);
    for (out, merges) |*text, pair| text.* = try std.fmt.allocPrint(arena, "{s} {s}", .{ pair[0], pair[1] });
    return out;
}

fn setTokens(b: *Builder, tokens: *std.ArrayList([]const u8), kinds: *std.ArrayList(i32), vocabulary: usize) Allocator.Error!void {
    // Up to the pinned count: reserved slots, as a real vocabulary has.
    while (tokens.items.len < vocabulary) {
        try tokens.append(b.arena, try b.label("<unused{d}>", .{tokens.items.len}));
        try kinds.append(b.arena, 5);
    }
    try b.set("tokenizer.ggml.tokens", .{ .strings = tokens.items });
    try b.set("tokenizer.ggml.token_type", .{ .int32s = kinds.items });
}

/// `tokenizer.ggml.model = gpt2`: every byte, the merges' products, the end
/// marker, then reserved slots.
fn byteLevelTokenizer(b: *Builder, pre: []const u8, vocabulary: usize) Allocator.Error!void {
    var tokens: std.ArrayList([]const u8) = .empty;
    var kinds: std.ArrayList(i32) = .empty;
    for (0..256) |byte| {
        try tokens.append(b.arena, try utf8(b.arena, byteLetter(@intCast(byte))));
        try kinds.append(b.arena, 1);
    }
    for (byte_level_merges) |pair| {
        try tokens.append(b.arena, try b.label("{s}{s}", .{ pair[0], pair[1] }));
        try kinds.append(b.arena, 1);
    }
    const end: u32 = @intCast(tokens.items.len);
    try tokens.append(b.arena, end_marker);
    try kinds.append(b.arena, 3);
    try b.set("tokenizer.ggml.model", .{ .string = "gpt2" });
    try b.set("tokenizer.ggml.pre", .{ .string = pre });
    try b.set("tokenizer.ggml.merges", .{ .strings = try mergeStrings(b.arena, &byte_level_merges) });
    try b.set("tokenizer.ggml.eos_token_id", .{ .uint32 = end });
    try setTokens(b, &tokens, &kinds, vocabulary);
}

/// `tokenizer.ggml.model = gemma4`: the control tokens, `<0xNN>` bytes, the
/// printable ASCII symbols and U+2581, the merges' products, reserved slots.
fn spmTokenizer(b: *Builder, vocabulary: usize) Allocator.Error!void {
    var tokens: std.ArrayList([]const u8) = .empty;
    var kinds: std.ArrayList(i32) = .empty;
    for ([_][]const u8{ "<pad>", "<eos>", "<bos>", "<unk>" }, [_]i32{ 3, 3, 3, 2 }) |text, kind| {
        try tokens.append(b.arena, text);
        try kinds.append(b.arena, kind);
    }
    for (0..256) |byte| {
        try tokens.append(b.arena, try b.label("<0x{X:0>2}>", .{byte}));
        try kinds.append(b.arena, 6);
    }
    for ('!'..'~' + 1) |symbol| {
        try tokens.append(b.arena, try utf8(b.arena, @intCast(symbol)));
        try kinds.append(b.arena, 1);
    }
    try tokens.append(b.arena, "\u{2581}");
    try kinds.append(b.arena, 1);
    for (spm_merges) |pair| {
        try tokens.append(b.arena, try b.label("{s}{s}", .{ pair[0], pair[1] }));
        try kinds.append(b.arena, 1);
    }
    try b.set("tokenizer.ggml.model", .{ .string = "gemma4" });
    try b.set("tokenizer.ggml.merges", .{ .strings = try mergeStrings(b.arena, &spm_merges) });
    try b.set("tokenizer.ggml.bos_token_id", .{ .uint32 = 2 });
    try b.set("tokenizer.ggml.eos_token_id", .{ .uint32 = 1 });
    try b.set("tokenizer.ggml.padding_token_id", .{ .uint32 = 0 });
    try setTokens(b, &tokens, &kinds, vocabulary);
}

// --- qwen35: Qwen3.5/3.8, Saluki, Bonsai --------------------------------

/// The smallest shape `qwen35.bind` reads from a file.
const qwen35_shape: qwen35.Shape = .{ .hidden = 256, .ffn = 256, .heads = 4, .layers = 4, .value_heads = 16 };

/// `alpha_beta`: the encoding of the DeltaNet gate pair, the file's own or
/// the mix's (`null` is the file's).
fn qwen35Tensors(b: *Builder, shape: qwen35.Shape, alpha_beta: ?u32) Allocator.Error!void {
    const S = qwen35.Shape;
    const d: u64 = shape.hidden;
    const ff: u64 = shape.ffn;
    try b.matrix("token_embd.weight", &.{ d, qwen35.vocabulary });
    try b.gain("output_norm.weight", &.{d});
    try b.matrix("output.weight", &.{ d, qwen35.vocabulary });
    for (0..shape.layers) |i| {
        try b.gain(try b.label("blk.{d}.attn_norm.weight", .{i}), &.{d});
        try b.gain(try b.label("blk.{d}.post_attention_norm.weight", .{i}), &.{d});
        try b.matrix(try b.label("blk.{d}.ffn_gate.weight", .{i}), &.{ d, ff });
        try b.matrix(try b.label("blk.{d}.ffn_up.weight", .{i}), &.{ d, ff });
        try b.matrix(try b.label("blk.{d}.ffn_down.weight", .{i}), &.{ ff, d });
        if ((i + 1) % 4 == 0) {
            try b.matrix(try b.label("blk.{d}.attn_q.weight", .{i}), &.{ d, shape.queryGateWidth() });
            try b.matrix(try b.label("blk.{d}.attn_k.weight", .{i}), &.{ d, S.kv_width });
            try b.matrix(try b.label("blk.{d}.attn_v.weight", .{i}), &.{ d, S.kv_width });
            try b.matrix(try b.label("blk.{d}.attn_output.weight", .{i}), &.{ shape.queryWidth(), d });
            try b.gain(try b.label("blk.{d}.attn_q_norm.weight", .{i}), &.{S.head_width});
            try b.gain(try b.label("blk.{d}.attn_k_norm.weight", .{i}), &.{S.head_width});
        } else {
            try b.matrix(try b.label("blk.{d}.attn_qkv.weight", .{i}), &.{ d, shape.qkvWidth() });
            try b.matrix(try b.label("blk.{d}.attn_gate.weight", .{i}), &.{ d, shape.innerWidth() });
            try b.noise(try b.label("blk.{d}.ssm_conv1d.weight", .{i}), &.{ S.conv_kernel, shape.qkvWidth() }, 0.25);
            try b.noise(try b.label("blk.{d}.ssm_dt.bias", .{i}), &.{shape.value_heads}, 0.5);
            // A log-decay's factor: the recurrence refuses a growing state.
            try b.uniform(try b.label("blk.{d}.ssm_a", .{i}), &.{shape.value_heads}, -1, -0.1);
            try b.encoded(try b.label("blk.{d}.ssm_beta.weight", .{i}), &.{ d, shape.value_heads }, alpha_beta orelse b.encoding);
            try b.encoded(try b.label("blk.{d}.ssm_alpha.weight", .{i}), &.{ d, shape.value_heads }, alpha_beta orelse b.encoding);
            try b.gain(try b.label("blk.{d}.ssm_norm.weight", .{i}), &.{S.state_width});
            try b.matrix(try b.label("blk.{d}.ssm_out.weight", .{i}), &.{ shape.innerWidth(), d });
        }
    }
}

fn qwen35Metadata(b: *Builder, shape: qwen35.Shape) Allocator.Error!void {
    const S = qwen35.Shape;
    try b.set("general.architecture", .{ .string = qwen35.architecture });
    try b.set("general.name", .{ .string = "tiny qwen35" });
    try b.set("general.quantization_version", .{ .uint32 = 2 });
    for ([_]struct { []const u8, usize }{
        .{ "context_length", 262144 },
        .{ "block_count", shape.layers },
        .{ "embedding_length", shape.hidden },
        .{ "feed_forward_length", shape.ffn },
        .{ "attention.head_count", shape.heads },
        .{ "attention.head_count_kv", S.kv_heads },
        .{ "attention.key_length", S.head_width },
        .{ "attention.value_length", S.head_width },
        .{ "ssm.conv_kernel", S.conv_kernel },
        .{ "ssm.state_size", S.state_width },
        .{ "ssm.group_count", S.key_heads },
        .{ "ssm.time_step_rank", shape.value_heads },
        .{ "ssm.inner_size", shape.innerWidth() },
        .{ "full_attention_interval", 4 },
        .{ "rope.dimension_count", 64 },
    }) |entry| try b.setf("qwen35.{s}", .{entry[0]}, .{ .uint32 = @intCast(entry[1]) });
    try b.set("qwen35.rope.freq_base", .{ .float32 = 10_000_000 });
    try b.set("qwen35.attention.layer_norm_rms_epsilon", .{ .float32 = 1e-6 });
    try b.set("qwen35.rope.dimension_sections", .{ .int32s = &.{ 11, 11, 10, 0 } });
    const recurrent = try b.arena.alloc(bool, shape.layers);
    for (recurrent, 0..) |*flag, i| flag.* = (i + 1) % 4 != 0;
    try b.set("qwen35.attention.recurrent_layers", .{ .booleans = recurrent });
    try byteLevelTokenizer(b, "qwen35", qwen35.vocabulary);
}

fn emitQwen35(b: *Builder, _: u64) Allocator.Error!void {
    try qwen35Metadata(b, qwen35_shape);
    try qwen35Tensors(b, qwen35_shape, null);
}

fn emitBonsai(b: *Builder, seed: u64) Allocator.Error!void {
    const shape = qwen35.qwen38_shape;
    try qwen35Metadata(b, shape);
    try qwen35Tensors(b, shape, 30);
    const R = qwen35.Rotation;
    try b.set("prism.hadamard.version", .{ .uint32 = 1 });
    try b.set("prism.hadamard.block_size", .{ .uint32 = R.block_size });
    try b.set("prism.hadamard.transform", .{ .string = "normalized-sylvester-walsh-hadamard" });
    try b.set("prism.hadamard.axis", .{ .string = "input-last-dimension" });
    try b.set("prism.hadamard.sign_mode", .{ .string = "explicit" });
    try b.set("prism.hadamard.gdn_v_grouped", .{ .boolean = true });
    var widths: [R.widths.len]i32 = undefined;
    var total: usize = 0;
    for (&widths, R.widths) |*width, value| {
        width.* = @intCast(value);
        total += value;
    }
    try b.set("prism.hadamard.sign_widths", .{ .int32s = try b.arena.dupe(i32, &widths) });
    const signs = try b.arena.alloc(i32, total);
    var prng: std.Random.DefaultPrng = .init(seed);
    for (signs) |*sign| sign.* = if (prng.random().boolean()) 1 else -1;
    try b.set("prism.hadamard.sign_values", .{ .int32s = signs });
    var names: std.ArrayList([]const u8) = .empty;
    for (0..shape.layers) |i| {
        const kinds: []const []const u8 = if ((i + 1) % 4 == 0)
            &.{ "attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down" }
        else
            &.{ "attn_qkv", "attn_gate", "ssm_out", "ffn_gate", "ffn_up", "ffn_down" };
        for (kinds) |kind| try names.append(b.arena, try b.label("blk.{d}.{s}.weight", .{ i, kind }));
    }
    try names.append(b.arena, "output.weight");
    try b.set("prism.hadamard.weight_names", .{ .strings = names.items });
    try b.set("prism.hadamard.inverse_weight_names", .{ .strings = &.{"token_embd.weight"} });
}

// --- muse-glimmer -------------------------------------------------------

fn emitMuse(b: *Builder, _: u64) Allocator.Error!void {
    try b.set("general.architecture", .{ .string = muse.architecture });
    try b.set("general.quantization_version", .{ .uint32 = 2 });
    try b.set("general.name", .{ .string = "tiny muse-glimmer" });
    for ([_]struct { []const u8, u32 }{
        .{ "block_count", muse.layer_count },
        .{ "context_length", 131072 },
        .{ "embedding_length", muse.embedding },
        .{ "feed_forward_length", muse.feed_forward },
        .{ "attention.head_count", muse.heads },
        .{ "attention.head_count_kv", muse.kv_heads },
        .{ "attention.key_length", muse.head_size },
        .{ "attention.value_length", muse.head_size },
        .{ "attention.sliding_window", muse.window },
        .{ "attention.sliding_window_pattern", 4 },
    }) |entry| try b.setf("muse-glimmer.{s}", .{entry[0]}, .{ .uint32 = entry[1] });
    try b.set("muse-glimmer.rope.freq_base", .{ .float32 = muse.rope_base });
    try b.set("muse-glimmer.attention.layer_norm_rms_epsilon", .{ .float32 = muse.rms_epsilon });
    try b.set("muse-glimmer.final_logit_softcapping", .{ .float32 = muse.final_softcap });
    try b.set("muse-glimmer.logit_scale", .{ .float32 = muse.logit_scale });
    try byteLevelTokenizer(b, "llama4", muse.vocabulary);

    const d: u64 = muse.embedding;
    const ff: u64 = muse.feed_forward;
    try b.matrix("token_embd.weight", &.{ d, muse.vocabulary });
    try b.matrix("output.weight", &.{ d, muse.vocabulary });
    try b.gain("output_norm.weight", &.{d});
    for (0..muse.layer_count) |i| {
        for ([_][]const u8{ "attn_norm", "post_attention_norm", "ffn_norm", "post_ffw_norm" }) |kind|
            try b.gain(try b.label("blk.{d}.{s}.weight", .{ i, kind }), &.{d});
        try b.matrix(try b.label("blk.{d}.attn_q.weight", .{i}), &.{ d, muse.query_width });
        try b.matrix(try b.label("blk.{d}.attn_k.weight", .{i}), &.{ d, muse.kv_width });
        try b.matrix(try b.label("blk.{d}.attn_v.weight", .{i}), &.{ d, muse.kv_width });
        try b.matrix(try b.label("blk.{d}.attn_gate.weight", .{i}), &.{ d, muse.query_width });
        try b.matrix(try b.label("blk.{d}.attn_output.weight", .{i}), &.{ muse.query_width, d });
        try b.gain(try b.label("blk.{d}.attn_q_norm.weight", .{i}), &.{muse.head_size});
        try b.gain(try b.label("blk.{d}.attn_k_norm.weight", .{i}), &.{muse.head_size});
        try b.matrix(try b.label("blk.{d}.ffn_gate.weight", .{i}), &.{ d, ff });
        try b.matrix(try b.label("blk.{d}.ffn_up.weight", .{i}), &.{ d, ff });
        try b.matrix(try b.label("blk.{d}.ffn_down.weight", .{i}), &.{ ff, d });
    }
}

// --- gemma4 -------------------------------------------------------------

/// `per_layer_model_proj` is BF16 on a K-quant file, as a real one keeps it.
fn emitGemma(b: *Builder, config: *const gemma4.Config) Allocator.Error!void {
    try b.set("general.architecture", .{ .string = gemma4.architecture });
    try b.set("general.quantization_version", .{ .uint32 = 2 });
    try b.set("general.name", .{ .string = config.profile });
    for ([_]struct { []const u8, usize }{
        .{ "attention.key_length", 512 },
        .{ "attention.value_length", 512 },
        .{ "attention.key_length_swa", 256 },
        .{ "attention.value_length_swa", 256 },
        .{ "rope.dimension_count", 512 },
        .{ "rope.dimension_count_swa", 256 },
        .{ "context_length", config.context_length },
        .{ "attention.head_count", config.heads },
        .{ "attention.sliding_window", config.window },
        .{ "attention.shared_kv_layers", config.layer_count - config.kv_layers },
        .{ "embedding_length_per_layer_input", config.per_layer_input },
        .{ "block_count", config.layer_count },
        .{ "embedding_length", config.embedding },
        .{ "feed_forward_length", config.feed_forward },
    }) |entry| try b.setf("gemma4.{s}", .{entry[0]}, .{ .uint32 = @intCast(entry[1]) });
    try b.set("gemma4.rope.freq_base", .{ .float32 = 1_000_000 });
    try b.set("gemma4.rope.freq_base_swa", .{ .float32 = 10_000 });
    try b.set("gemma4.attention.layer_norm_rms_epsilon", .{ .float32 = gemma4.rms_epsilon });
    try b.set("gemma4.final_logit_softcapping", .{ .float32 = gemma4.final_softcap });
    const kv_heads = try b.arena.alloc(i32, config.layer_count);
    const sliding = try b.arena.alloc(bool, config.layer_count);
    for (kv_heads, sliding, 0..) |*heads, *flag, i| {
        const kind = gemma4.kindOf(i);
        heads.* = @intCast(gemma4.kvHeadsOf(config, kind));
        flag.* = kind == .sliding;
    }
    try b.set("gemma4.attention.head_count_kv", .{ .int32s = kv_heads });
    try b.set("gemma4.attention.sliding_window_pattern", .{ .booleans = sliding });
    if (config.experts) |experts| {
        try b.set("gemma4.expert_count", .{ .uint32 = @intCast(experts.count) });
        try b.set("gemma4.expert_used_count", .{ .uint32 = @intCast(experts.used) });
        try b.set("gemma4.expert_feed_forward_length", .{ .uint32 = @intCast(experts.feed_forward) });
    }
    try spmTokenizer(b, gemma4.vocabulary);

    const d: u64 = config.embedding;
    const ff: u64 = config.feed_forward;
    try b.matrix("token_embd.weight", &.{ d, gemma4.vocabulary });
    try b.gain("output_norm.weight", &.{d});
    // The global layers' per-pair frequency factors, as a converter derives them.
    const factors = try b.arena.alloc(f32, 256);
    for (factors, 0..) |*factor, i| factor.* = std.math.pow(f32, 1_000_000, -@as(f32, @floatFromInt(i)) / 256);
    try b.values("rope_freqs.weight", &.{256}, factors);
    const w: u64 = config.per_layer_input;
    const layers: u64 = config.layer_count;
    if (w != 0) {
        try b.matrix("per_layer_token_embd.weight", &.{ w * layers, gemma4.vocabulary });
        try b.encoded("per_layer_model_proj.weight", &.{ d, w * layers }, if (b.encoding == 12) 30 else b.encoding);
        try b.gain("per_layer_proj_norm.weight", &.{w});
    }
    for (0..config.layer_count) |i| {
        const kind = gemma4.kindOf(i);
        const hd: u64 = kind.headSize();
        const q = config.heads * hd;
        const kv = gemma4.kvHeadsOf(config, kind) * hd;
        const owns_kv = i < config.kv_layers;
        for ([_][]const u8{ "attn_norm", "post_attention_norm", "ffn_norm", "post_ffw_norm" }) |name|
            try b.gain(try b.label("blk.{d}.{s}.weight", .{ i, name }), &.{d});
        try b.matrix(try b.label("blk.{d}.attn_q.weight", .{i}), &.{ d, q });
        if (owns_kv) {
            try b.matrix(try b.label("blk.{d}.attn_k.weight", .{i}), &.{ d, kv });
            try b.gain(try b.label("blk.{d}.attn_k_norm.weight", .{i}), &.{hd});
            if (kind == .sliding or config.global_value)
                try b.matrix(try b.label("blk.{d}.attn_v.weight", .{i}), &.{ d, kv });
        }
        try b.matrix(try b.label("blk.{d}.attn_output.weight", .{i}), &.{ q, d });
        // Scores are unscaled: at unit gain the softmax is one-hot and rounding
        // is amplified a thousandfold. This gain keeps scores near 1.
        const score_gain = 1 / @sqrt(@as(f32, @floatFromInt(hd)));
        try b.uniform(try b.label("blk.{d}.attn_q_norm.weight", .{i}), &.{hd}, 0.8 * score_gain, 1.2 * score_gain);
        try b.matrix(try b.label("blk.{d}.ffn_gate.weight", .{i}), &.{ d, ff });
        try b.matrix(try b.label("blk.{d}.ffn_up.weight", .{i}), &.{ d, ff });
        try b.matrix(try b.label("blk.{d}.ffn_down.weight", .{i}), &.{ ff, d });
        try b.gain(try b.label("blk.{d}.layer_output_scale.weight", .{i}), &.{1});
        if (config.experts) |experts| {
            const n: u64 = experts.count;
            const eff: u64 = experts.feed_forward;
            // Spreads the router's logits: rounding across a close top-k swaps an expert.
            try b.noise(try b.label("blk.{d}.ffn_gate_inp.weight", .{i}), &.{ d, n }, 1);
            try b.gain(try b.label("blk.{d}.ffn_gate_inp.scale", .{i}), &.{d});
            try b.matrix(try b.label("blk.{d}.ffn_gate_up_exps.weight", .{i}), &.{ d, 2 * eff, n });
            // A row too short for a 256-value block: a K-quant file stores it in 32s.
            const whole = eff % layouts.layout(b.encoding).?.elements_per_block == 0;
            try b.encoded(try b.label("blk.{d}.ffn_down_exps.weight", .{i}), &.{ eff, d, n }, if (whole) b.encoding else 6);
            try b.gain(try b.label("blk.{d}.ffn_down_exps.scale", .{i}), &.{n});
            try b.gain(try b.label("blk.{d}.post_ffw_norm_1.weight", .{i}), &.{d});
            try b.gain(try b.label("blk.{d}.pre_ffw_norm_2.weight", .{i}), &.{d});
            try b.gain(try b.label("blk.{d}.post_ffw_norm_2.weight", .{i}), &.{d});
        }
        if (w != 0) {
            try b.matrix(try b.label("blk.{d}.inp_gate.weight", .{i}), &.{ d, w });
            try b.matrix(try b.label("blk.{d}.proj.weight", .{i}), &.{ w, d });
            try b.gain(try b.label("blk.{d}.post_norm.weight", .{i}), &.{d});
        }
    }
}

fn emitGemmaDense(b: *Builder, _: u64) Allocator.Error!void {
    return emitGemma(b, &gemma4.config_tiny_dense);
}
fn emitGemmaExperts(b: *Builder, _: u64) Allocator.Error!void {
    return emitGemma(b, &gemma4.config_tiny_experts);
}
fn emitGemmaE4b(b: *Builder, _: u64) Allocator.Error!void {
    return emitGemma(b, &gemma4.config_tiny_e4b);
}

// These run in the ordinary test binary, against the constants the
// application links.
test "the tiny Qwen shape passes the production validators" {
    var doc = try directoryDocument(std.testing.allocator, find("qwen35").?, 8);
    defer doc.deinit();
    const binding = try qwen35.bind(std.testing.allocator, &doc);
    try std.testing.expectEqualStrings("qwen35", binding.summary.profile);
    try std.testing.expectEqual(@as(u32, 4), binding.summary.decoder_layers);
}

test "production refuses a shape only the tiny build pins" {
    var doc = try directoryDocument(std.testing.allocator, find("gemma4-dense").?, 12);
    defer doc.deinit();
    try std.testing.expectError(error.UnsupportedConfiguration, gemma4.bind(std.testing.allocator, &doc));
}
