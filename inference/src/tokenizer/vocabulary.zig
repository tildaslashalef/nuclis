//! Owned BPE vocabulary data loaded from a parsed GGUF directory: the GPT-2
//! family (`tokenizer.ggml.model = "gpt2"`, byte-level alphabet, a `pre`
//! key naming the splitter) and Gemma 4's SPM-style BPE (`"gemma4"`, raw
//! UTF-8 with U+2581 for spaces, byte tokens `<0xNN>`; `pre` is implied).
//! This module knows storage and ranks, not Unicode splitting or model layers.
//! IDs are array indices; merge ranks are file order (lower ranks merge first).
//! An arena owns strings, entries, and lookup tables, so the input Document and
//! directory image can be released as soon as load returns.
const std = @import("std");
const gguf = @import("../formats/gguf.zig");

pub const TokenType = enum(i32) { normal = 1, unknown = 2, control = 3, user_defined = 4, unused = 5, byte = 6 };
pub const Token = struct { text: []const u8, kind: TokenType };
pub const Limits = struct {
    tokens: u32 = 1_000_000,
    merges: u32 = 1_000_000,
    /// Total copied token/merge string bytes. Tables and entry arrays have
    /// separate count bounds; this is not a total allocator budget.
    text_bytes: usize = 64 * 1024 * 1024,
    gguf: gguf.Limits = .{},
};
pub const Error = gguf.Error || error{
    MissingMetadata,
    InvalidTokenizerMetadata,
    UnsupportedTokenizer,
    InvalidToken,
    DuplicateToken,
    InvalidMerge,
    DuplicateMerge,
    InvalidSpecialToken,
};

pub const Vocabulary = struct {
    storage: std.heap.ArenaAllocator,
    tokens: []const Token,
    pre: []const u8,
    bos: ?u32,
    eos: ?u32,
    padding: ?u32,
    // Keys borrow arena-owned strings. Never mutate these tables after loading.
    token_ids: std.StringHashMapUnmanaged(u32),
    merge_ranks: std.StringHashMapUnmanaged(u32),

    /// Do not copy this owning value and deinitialize both copies.
    pub fn deinit(self: *Vocabulary) void {
        self.storage.deinit();
        self.* = undefined;
    }

    pub fn tokenId(self: *const Vocabulary, encoded_text: []const u8) ?u32 {
        return self.token_ids.get(encoded_text);
    }

    /// Key is two byte-alphabet strings separated by one ASCII space, exactly
    /// as stored in GGUF. This performs no byte mapping or tokenization.
    pub fn mergeRank(self: *const Vocabulary, encoded_pair: []const u8) ?u32 {
        return self.merge_ranks.get(encoded_pair);
    }
};

/// `directory` must be the same file's bytes [0..doc.directory_bytes]. The
/// caller owns file I/O and limits that read; this pure loader reads no weights.
/// Only gpt2 and gemma4 storage are accepted. `pre` is retained, not
/// interpreted or selected automatically (for gemma4 it is the model name:
/// the file carries no `pre` key): successful loading does not establish
/// tokenizer support; the encoder decides that.
pub fn load(gpa: std.mem.Allocator, doc: gguf.Document, directory: []const u8, limits: Limits) Error!Vocabulary {
    if (directory.len != doc.directory_bytes) return error.InvalidTokenizerMetadata;
    const model = try string(doc, "tokenizer.ggml.model");
    const pre = if (std.mem.eql(u8, model, "gpt2"))
        try string(doc, "tokenizer.ggml.pre")
    else if (std.mem.eql(u8, model, "gemma4"))
        "gemma4"
    else
        return error.UnsupportedTokenizer;
    if (pre.len == 0 or pre.len > 128 or !std.unicode.utf8ValidateSlice(pre)) return error.InvalidTokenizerMetadata;
    const words = try array(doc, "tokenizer.ggml.tokens", .string);
    const types = try array(doc, "tokenizer.ggml.token_type", .int32);
    const merges = try array(doc, "tokenizer.ggml.merges", .string);
    if (words.count == 0 or types.count != words.count) return error.InvalidTokenizerMetadata;
    if (words.count > limits.tokens or merges.count > limits.merges) return error.LimitExceeded;
    var word_reader = try gguf.ArrayReader.init(words, directory, limits.gguf);
    var type_reader = try gguf.ArrayReader.init(types, directory, limits.gguf);
    var merge_reader = try gguf.ArrayReader.init(merges, directory, limits.gguf);
    const bos = try special(doc, "tokenizer.ggml.bos_token_id", words.count);
    const eos = try special(doc, "tokenizer.ggml.eos_token_id", words.count);
    const padding = try special(doc, "tokenizer.ggml.padding_token_id", words.count);

    var arena: std.heap.ArenaAllocator = .init(gpa);
    errdefer arena.deinit();
    const alloc = arena.allocator();
    const tokens = try alloc.alloc(Token, @intCast(words.count));
    var ids: std.StringHashMapUnmanaged(u32) = .empty;
    var ranks: std.StringHashMapUnmanaged(u32) = .empty;
    try ids.ensureTotalCapacity(alloc, @intCast(words.count));
    try ranks.ensureTotalCapacity(alloc, @intCast(merges.count));
    var remaining = limits.text_bytes;
    for (tokens, 0..) |*token, id| {
        const text = (try word_reader.nextString()).?;
        if (text.len == 0 or !std.unicode.utf8ValidateSlice(text)) return error.InvalidToken;
        const kind = std.enums.fromInt(TokenType, (try type_reader.nextInt32()).?) orelse return error.InvalidToken;
        const owned = try copyText(alloc, text, &remaining);
        const entry = try ids.getOrPut(alloc, owned);
        if (entry.found_existing) return error.DuplicateToken;
        entry.value_ptr.* = @intCast(id);
        token.* = .{ .text = owned, .kind = kind };
    }
    // The reference re-types these spellings as user-defined in every
    // vocabulary, so they match in plain text even with parse_special off.
    for ([_][]const u8{ "<|start|>", "<|message|>", "<|channel|>", "<|constrain|>" }) |text| {
        if (ids.get(text)) |id| tokens[id].kind = .user_defined;
    }
    var rank: u32 = 0;
    while (try merge_reader.nextString()) |text| : (rank += 1) {
        const space = std.mem.indexOfScalar(u8, text, ' ') orelse return error.InvalidMerge;
        if (space == 0 or space + 1 == text.len or
            std.mem.indexOfScalar(u8, text[space + 1 ..], ' ') != null or
            !std.unicode.utf8ValidateSlice(text)) return error.InvalidMerge;
        const owned = try copyText(alloc, text, &remaining);
        const entry = try ranks.getOrPut(alloc, owned);
        if (entry.found_existing) return error.DuplicateMerge;
        entry.value_ptr.* = rank;
    }
    const owned_pre = try alloc.dupe(u8, pre);
    return .{ .storage = arena, .tokens = tokens, .pre = owned_pre, .bos = bos, .eos = eos, .padding = padding, .token_ids = ids, .merge_ranks = ranks };
}

fn copyText(alloc: std.mem.Allocator, text: []const u8, remaining: *usize) Error![]const u8 {
    if (text.len > remaining.*) return error.LimitExceeded;
    remaining.* -= text.len;
    return alloc.dupe(u8, text);
}

fn string(doc: gguf.Document, key: []const u8) Error![]const u8 {
    return switch (doc.get(key) orelse return error.MissingMetadata) {
        .string => |s| s,
        else => error.InvalidTokenizerMetadata,
    };
}

fn array(doc: gguf.Document, key: []const u8, kind: gguf.Type) Error!gguf.Array {
    const value = switch (doc.get(key) orelse return error.MissingMetadata) {
        .array => |a| a,
        else => return error.InvalidTokenizerMetadata,
    };
    if (value.element_type != kind) return error.InvalidTokenizerMetadata;
    return value;
}

fn special(doc: gguf.Document, key: []const u8, count: u64) Error!?u32 {
    const id = switch (doc.get(key) orelse return null) {
        .unsigned => |n| n,
        else => return error.InvalidSpecialToken,
    };
    if (id >= count) return error.InvalidSpecialToken;
    return @intCast(id);
}

const Fixture = struct {
    words: []const []const u8 = &.{ "a", "b", "ab", "<end>" },
    kinds: []const i32 = &.{ 1, 1, 1, 3 },
    merges: []const []const u8 = &.{"a b"},
    model: []const u8 = "gpt2",
    eos: u32 = 3,
};

fn writeString(w: *std.Io.Writer, text: []const u8) !void {
    try w.writeInt(u64, text.len, .little);
    try w.writeAll(text);
}

fn fixture(options: Fixture) !std.Io.Writer.Allocating {
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    errdefer out.deinit();
    const w = &out.writer;
    try w.writeAll("GGUF");
    try w.writeInt(u32, 3, .little);
    try w.writeInt(u64, 0, .little);
    try w.writeInt(u64, 6, .little);
    for ([_][]const u8{ "tokenizer.ggml.model", "tokenizer.ggml.pre" }, [_][]const u8{ options.model, "qwen35" }) |key, value| {
        try writeString(w, key);
        try w.writeInt(u32, 8, .little);
        try writeString(w, value);
    }
    for ([_][]const u8{ "tokenizer.ggml.tokens", "tokenizer.ggml.merges" }, [_][]const []const u8{ options.words, options.merges }) |key, values| {
        try writeString(w, key);
        try w.writeInt(u32, 9, .little);
        try w.writeInt(u32, 8, .little);
        try w.writeInt(u64, values.len, .little);
        for (values) |value| try writeString(w, value);
    }
    try writeString(w, "tokenizer.ggml.token_type");
    try w.writeInt(u32, 9, .little);
    try w.writeInt(u32, 5, .little);
    try w.writeInt(u64, options.kinds.len, .little);
    for (options.kinds) |kind| try w.writeInt(i32, kind, .little);
    try writeString(w, "tokenizer.ggml.eos_token_id");
    try w.writeInt(u32, 4, .little);
    try w.writeInt(u32, options.eos, .little);
    try w.splatByteAll(0, (32 - out.written().len % 32) % 32);
    return out;
}

fn loadFixture(alloc: std.mem.Allocator, bytes: []const u8, limits: Limits) !Vocabulary {
    var reader: std.Io.Reader = .fixed(bytes);
    var doc = try gguf.parse(std.testing.allocator, &reader, bytes.len, .{});
    defer doc.deinit();
    return load(alloc, doc, bytes[0..@intCast(doc.directory_bytes)], limits);
}

test "vocabulary owns text and retains IDs, types, and ordered merge ranks" {
    var bytes = try fixture(.{ .merges = &.{ "a b", "ab a" } });
    var vocab = try loadFixture(std.testing.allocator, bytes.written(), .{});
    defer vocab.deinit();
    // Neither the parsed document nor source image must outlive the vocabulary.
    bytes.deinit();
    try std.testing.expectEqual(@as(?u32, 2), vocab.tokenId("ab"));
    try std.testing.expectEqual(@as(?u32, null), vocab.tokenId("absent"));
    try std.testing.expectEqual(@as(?u32, 0), vocab.mergeRank("a b"));
    try std.testing.expectEqual(@as(?u32, 1), vocab.mergeRank("ab a"));
    try std.testing.expectEqual(@as(?u32, null), vocab.mergeRank("b a"));
    try std.testing.expectEqual(TokenType.control, vocab.tokens[3].kind);
    try std.testing.expectEqualStrings("qwen35", vocab.pre);
    try std.testing.expectEqual(@as(?u32, 3), vocab.eos);
    try std.testing.expectEqual(@as(?u32, null), vocab.bos);
}

test "vocabulary re-types the reference's always-rendered markers as user-defined" {
    var bytes = try fixture(.{ .words = &.{ "a", "<|start|>", "<|message|>", "<end>" }, .kinds = &.{ 1, 3, 3, 3 } });
    defer bytes.deinit();
    var vocab = try loadFixture(std.testing.allocator, bytes.written(), .{});
    defer vocab.deinit();
    try std.testing.expectEqual(TokenType.user_defined, vocab.tokens[1].kind);
    try std.testing.expectEqual(TokenType.user_defined, vocab.tokens[2].kind);
    try std.testing.expectEqual(TokenType.control, vocab.tokens[3].kind);
}

test "vocabulary rejects unsupported storage and malformed entries" {
    const cases = .{
        .{ Fixture{ .model = "llama" }, error.UnsupportedTokenizer },
        .{ Fixture{ .words = &.{ "a", "a", "ab", "<end>" } }, error.DuplicateToken },
        .{ Fixture{ .words = &.{ "a", "\xff", "ab", "<end>" } }, error.InvalidToken },
        .{ Fixture{ .kinds = &.{ 1, 1, 1 } }, error.InvalidTokenizerMetadata },
        .{ Fixture{ .kinds = &.{ 1, -1, 1, 3 } }, error.InvalidToken },
        .{ Fixture{ .eos = 4 }, error.InvalidSpecialToken },
        .{ Fixture{ .merges = &.{"a "} }, error.InvalidMerge },
        .{ Fixture{ .merges = &.{"a b c"} }, error.InvalidMerge },
        .{ Fixture{ .merges = &.{ "a b", "a b" } }, error.DuplicateMerge },
    };
    inline for (cases) |case| {
        var bytes = try fixture(case[0]);
        defer bytes.deinit();
        try std.testing.expectError(case[1], loadFixture(std.testing.allocator, bytes.written(), .{}));
    }
}

test "vocabulary count and aggregate string limits" {
    var bytes = try fixture(.{});
    defer bytes.deinit();
    for ([_]Limits{ .{ .tokens = 3 }, .{ .merges = 0 }, .{ .text_bytes = 11 }, .{ .gguf = .{ .string_bytes = 4 } } }) |limits| {
        try std.testing.expectError(error.LimitExceeded, loadFixture(std.testing.allocator, bytes.written(), limits));
    }
    var vocab = try loadFixture(std.testing.allocator, bytes.written(), .{ .tokens = 4, .merges = 1, .text_bytes = 12 });
    defer vocab.deinit();
}

test "vocabulary rejects missing metadata, wrong types, and mismatched directory images" {
    var bytes = try fixture(.{});
    defer bytes.deinit();
    var reader: std.Io.Reader = .fixed(bytes.written());
    var doc = try gguf.parse(std.testing.allocator, &reader, bytes.written().len, .{});
    defer doc.deinit();
    const directory = bytes.written()[0..@intCast(doc.directory_bytes)];
    try std.testing.expectError(error.InvalidTokenizerMetadata, load(std.testing.allocator, doc, directory[0 .. directory.len - 1], .{}));
    const metadata = @constCast(doc.metadata);
    const original = metadata[2]; // tokens array, owned by the parsed fixture
    metadata[2].key = "absent";
    try std.testing.expectError(error.MissingMetadata, load(std.testing.allocator, doc, directory, .{}));
    metadata[2] = original;
    metadata[2].value.array.element_type = .float32;
    try std.testing.expectError(error.InvalidTokenizerMetadata, load(std.testing.allocator, doc, directory, .{}));
    metadata[2] = original;
    metadata[2].value = .{ .unsigned = 4 };
    try std.testing.expectError(error.InvalidTokenizerMetadata, load(std.testing.allocator, doc, directory, .{}));
}

fn allocationCheck(alloc: std.mem.Allocator, bytes: []const u8) !void {
    var vocab = try loadFixture(alloc, bytes, .{});
    defer vocab.deinit();
}

test "vocabulary releases every partially initialized allocation" {
    var bytes = try fixture(.{});
    defer bytes.deinit();
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationCheck, .{bytes.written()});
}
