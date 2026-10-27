//! A `Qwen2Tokenizer` stored the slow way, `vocab.json` + `merges.txt` with
//! the added tokens in `tokenizer_config.json`'s `added_tokens_decoder`
//! (Qwen3-TTS ships no `tokenizer.json`), loaded into the shared Vocabulary
//! and encoded by hf_json.zig's path: added tokens matched first, NFC, the
//! case-seam split (`fix_mistral_regex`; cased.zig), byte-level BPE.
//! docs/engine/tokenizer.md § Vocabulary and merges files.
const std = @import("std");
const vocabulary = @import("vocabulary.zig");
const hf_json = @import("hf_json.zig");

pub const Limits = hf_json.Limits;
pub const Error = hf_json.Error || error{ InvalidVocabJson, InvalidTokenizerConfig };

const Added = struct {
    content: []const u8,
    single_word: bool = false,
    lstrip: bool = false,
    rstrip: bool = false,
    normalized: bool = true,
    special: bool = false,
};
const Config = struct {
    tokenizer_class: []const u8,
    add_prefix_space: bool = false,
    added_tokens_decoder: std.json.ArrayHashMap(Added) = .{},
};

/// Builds a tokenizer from the three files' bytes, which may be freed once
/// this returns. Ids are dense from zero across the vocabulary and the added
/// tokens; added tokens that are `special` are `.control`, the others
/// `.user_defined`. Rejects another tokenizer class, a prefix space, and
/// the `single_word`/`rstrip` flags by name.
pub fn load(gpa: std.mem.Allocator, vocab_json: []const u8, merges_txt: []const u8, config_json: []const u8, limits: Limits) Error!hf_json.Tokenizer {
    var scratch: std.heap.ArenaAllocator = .init(gpa);
    defer scratch.deinit();
    const s = scratch.allocator();
    const words = std.json.parseFromSliceLeaky(std.json.ArrayHashMap(u32), s, vocab_json, .{
        .duplicate_field_behavior = .@"error",
        .max_value_len = vocab_json.len,
    }) catch |err| return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.InvalidVocabJson,
    };
    const config = std.json.parseFromSliceLeaky(Config, s, config_json, .{
        .ignore_unknown_fields = true,
        .max_value_len = config_json.len,
    }) catch |err| return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.InvalidTokenizerConfig,
    };
    if (!std.mem.eql(u8, config.tokenizer_class, "Qwen2Tokenizer") and !std.mem.eql(u8, config.tokenizer_class, "Qwen2TokenizerFast"))
        return error.UnsupportedModel;
    if (config.add_prefix_space) return error.UnsupportedPreTokenizer;
    const added = config.added_tokens_decoder.map;
    if (words.map.count() == 0 or added.count() > limits.added) return error.LimitExceeded;

    var count: usize = 0;
    for (words.map.values()) |id| count = @max(count, @as(usize, id) + 1);
    for (added.keys()) |key| {
        const id = std.fmt.parseInt(u32, key, 10) catch return error.InvalidTokenizerConfig;
        count = @max(count, @as(usize, id) + 1);
    }
    if (count > limits.tokens) return error.LimitExceeded;

    var arena: std.heap.ArenaAllocator = .init(gpa);
    errdefer arena.deinit();
    const alloc = arena.allocator();
    var remaining = limits.text_bytes;
    const tokens = try alloc.alloc(vocabulary.Token, count);
    const filled = try s.alloc(bool, count);
    @memset(filled, false);
    var ids: std.StringHashMapUnmanaged(u32) = .empty;
    try ids.ensureTotalCapacity(alloc, @intCast(count));
    var it = words.map.iterator();
    while (it.next()) |entry| {
        const id = entry.value_ptr.*;
        const text = entry.key_ptr.*;
        if (filled[id] or text.len == 0 or !std.unicode.utf8ValidateSlice(text)) return error.InvalidToken;
        const owned = try copyText(alloc, text, &remaining);
        const slot = ids.getOrPutAssumeCapacity(owned);
        if (slot.found_existing) return error.DuplicateToken;
        slot.value_ptr.* = id;
        tokens[id] = .{ .text = owned, .kind = .normal };
        filled[id] = true;
    }

    var lstrip = try std.bit_set.Dynamic.initEmpty(alloc, count);
    var raw: std.ArrayList(u32) = .empty;
    var normalized: std.ArrayList(u32) = .empty;
    var added_it = added.iterator();
    while (added_it.next()) |entry| {
        const id = std.fmt.parseInt(u32, entry.key_ptr.*, 10) catch unreachable;
        const a = entry.value_ptr.*;
        if (a.single_word or a.rstrip) return error.UnsupportedAddedToken;
        if (a.content.len == 0 or a.content.len > limits.added_bytes or !std.unicode.utf8ValidateSlice(a.content) or filled[id])
            return error.InvalidAddedToken;
        const owned = try copyText(alloc, a.content, &remaining);
        const slot = ids.getOrPutAssumeCapacity(owned);
        if (slot.found_existing) return error.DuplicateToken;
        slot.value_ptr.* = id;
        tokens[id] = .{ .text = owned, .kind = if (a.special) .control else .user_defined };
        filled[id] = true;
        if (a.lstrip) lstrip.set(id);
        try (if (a.normalized) &normalized else &raw).append(alloc, id);
    }
    for (filled) |f| if (!f) return error.InvalidToken;
    for ([_]*std.ArrayList(u32){ &raw, &normalized }) |list| std.mem.sort(u32, list.items, tokens, struct {
        fn longer(t: []const vocabulary.Token, a: u32, b: u32) bool {
            return if (t[a].text.len != t[b].text.len) t[a].text.len > t[b].text.len else a < b;
        }
    }.longer);

    var ranks: std.StringHashMapUnmanaged(u32) = .empty;
    var lines = std.mem.splitScalar(u8, merges_txt, '\n');
    var rank: u32 = 0;
    var first = true;
    while (lines.next()) |line_raw| {
        const line = std.mem.trimEnd(u8, line_raw, "\r");
        defer first = false;
        if (line.len == 0) continue;
        if (first and std.mem.startsWith(u8, line, "#version")) continue;
        if (rank == limits.merges) return error.LimitExceeded;
        const space = std.mem.indexOfScalar(u8, line, ' ') orelse return error.InvalidMerge;
        if (space == 0 or space + 1 == line.len or std.mem.indexOfScalarPos(u8, line, space + 1, ' ') != null or
            !std.unicode.utf8ValidateSlice(line)) return error.InvalidMerge;
        const key = try copyText(alloc, line, &remaining);
        const slot = try ranks.getOrPut(alloc, key);
        if (slot.found_existing) return error.DuplicateMerge;
        slot.value_ptr.* = rank;
        rank += 1;
    }

    return .{
        .vocab = .{ .storage = arena, .tokens = tokens, .pre = "gpt2", .bos = null, .eos = null, .padding = null, .token_ids = ids, .merge_ranks = ranks },
        .shape = .cased,
        .raw = raw.items,
        .normalized = normalized.items,
        .lstrip = lstrip,
        .nfc = true,
    };
}

fn copyText(alloc: std.mem.Allocator, text: []const u8, remaining: *usize) Error![]const u8 {
    if (text.len > remaining.*) return error.LimitExceeded;
    remaining.* -= text.len;
    return alloc.dupe(u8, text);
}

// The GPT-2 alphabet spells space `Ġ` and newline `Ċ`; é (0xC3 0xA9) is `Ã©`.
const test_vocab =
    \\{"a":0,"b":1,"Ġ":2,"Ġb":3,"bb":4,"Ã":5,"©":6,"Ã©":7,"Ċ":8,"'":9,"t":10,"'t":11,"M":12,"c":13,"D":14,"Mc":15}
;
const test_merges = "#version: 0.2\nĠ b\nb b\nÃ ©\n' t\nM c\n";
const test_config =
    \\{"tokenizer_class":"Qwen2Tokenizer","add_prefix_space":false,"added_tokens_decoder":{
    \\ "16":{"content":"<|im_start|>","lstrip":false,"normalized":false,"rstrip":false,"single_word":false,"special":true},
    \\ "17":{"content":"<think>","lstrip":false,"normalized":false,"rstrip":false,"single_word":false,"special":false}}}
;

test "vocab.json, merges.txt, and added tokens encode as the pinned processor does" {
    const gpa = std.testing.allocator;
    var t = try load(gpa, test_vocab, test_merges, test_config, .{});
    defer t.deinit();
    try std.testing.expectEqual(hf_json.Shape.cased, t.shape);
    try std.testing.expectEqual(@as(usize, 18), t.vocab.tokens.len);
    try std.testing.expectEqual(vocabulary.TokenType.control, t.vocab.tokens[16].kind);
    try std.testing.expectEqual(vocabulary.TokenType.user_defined, t.vocab.tokens[17].kind);
    try std.testing.expectEqual(@as(?u32, 0), t.vocab.mergeRank("Ġ b"));
    // Added tokens split first, special or not; the rest is NFC (e + U+0301
    // is é), the cased pieces `a`, ` bb`, `é`, `'t`, ` Mc`, `D`, then BPE.
    const ids = try t.encode(gpa, "<|im_start|>a bb<think>e\u{301}'t McD", .{});
    defer gpa.free(ids);
    try std.testing.expectEqualSlices(u32, &.{ 16, 0, 3, 1, 17, 7, 11, 2, 15, 14 }, ids);
}

test "refusals name the unsupported flag or class" {
    const gpa = std.testing.allocator;
    try std.testing.expectError(error.UnsupportedModel, load(gpa, test_vocab, test_merges, "{\"tokenizer_class\":\"LlamaTokenizer\"}", .{}));
    try std.testing.expectError(error.UnsupportedPreTokenizer, load(gpa, test_vocab, test_merges, "{\"tokenizer_class\":\"Qwen2Tokenizer\",\"add_prefix_space\":true}", .{}));
    try std.testing.expectError(error.UnsupportedAddedToken, load(gpa, test_vocab, test_merges,
        \\{"tokenizer_class":"Qwen2Tokenizer","added_tokens_decoder":{"16":{"content":"<x>","rstrip":true}}}
    , .{}));
    try std.testing.expectError(error.InvalidToken, load(gpa, test_vocab, test_merges,
        \\{"tokenizer_class":"Qwen2Tokenizer","added_tokens_decoder":{"20":{"content":"<x>"}}}
    , .{}));
    try std.testing.expectError(error.InvalidMerge, load(gpa, test_vocab, "a  b\n", "{\"tokenizer_class\":\"Qwen2Tokenizer\"}", .{}));
    try std.testing.expectError(error.InvalidVocabJson, load(gpa, "[1]", test_merges, "{\"tokenizer_class\":\"Qwen2Tokenizer\"}", .{}));
}
