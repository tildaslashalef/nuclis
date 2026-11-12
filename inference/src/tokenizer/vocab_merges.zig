//! A `Qwen2Tokenizer` stored the slow way, `vocab.json` + `merges.txt` with
//! the added tokens in `tokenizer_config.json`'s `added_tokens_decoder`
//! (Qwen3-TTS ships no `tokenizer.json`), loaded into the shared Vocabulary
//! and encoded by hf_json.zig's path: added tokens matched first, NFC, the
//! case-seam split (`fix_mistral_regex`; cased.zig), byte-level BPE.
//! docs/engine/tokenizer.md § Vocabulary and merges files.
const std = @import("std");
const vocabulary = @import("vocabulary.zig");
const hf_json = @import("hf_json.zig");
const table = @import("table.zig");

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
    if (words.map.count() == 0) return error.InvalidVocabJson;
    if (added.count() > limits.added) return error.LimitExceeded;

    const list = try s.alloc(table.Added, added.count());
    for (list, added.keys(), added.values()) |*out, key, a| {
        const id = std.fmt.parseInt(u32, key, 10) catch return error.InvalidTokenizerConfig;
        if (a.single_word or a.rstrip) return error.UnsupportedAddedToken;
        out.* = .{ .id = id, .content = a.content, .special = a.special, .lstrip = a.lstrip, .normalized = a.normalized };
    }

    var arena: std.heap.ArenaAllocator = .init(gpa);
    errdefer arena.deinit();
    const alloc = arena.allocator();
    var remaining = limits.text_bytes;
    const t = try table.build(alloc, s, words.map, list, .{ .tokens = limits.tokens, .added_bytes = limits.added_bytes }, &remaining);

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
        const key = try table.copyText(alloc, line, &remaining);
        const slot = try ranks.getOrPut(alloc, key);
        if (slot.found_existing) return error.DuplicateMerge;
        slot.value_ptr.* = rank;
        rank += 1;
    }

    return .{
        .vocab = .{ .storage = arena, .tokens = t.tokens, .pre = "gpt2", .bos = null, .eos = null, .padding = null, .token_ids = t.ids, .merge_ranks = ranks },
        .shape = .cased,
        .raw = t.raw,
        .normalized = t.normalized,
        .lstrip = t.lstrip,
        .nfc = true,
    };
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

test "an added token over a vocabulary entry that spells it exactly re-types it" {
    var t = try load(std.testing.allocator, test_vocab, test_merges,
        \\{"tokenizer_class":"Qwen2Tokenizer","added_tokens_decoder":{"0":{"content":"a","special":true}}}
    , .{});
    defer t.deinit();
    try std.testing.expectEqual(vocabulary.TokenType.control, t.vocab.tokens[0].kind);
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
    try std.testing.expectError(error.InvalidVocabJson, load(gpa, "{}", test_merges, "{\"tokenizer_class\":\"Qwen2Tokenizer\"}", .{}));
    try std.testing.expectError(error.InvalidAddedToken, load(gpa, test_vocab, test_merges,
        \\{"tokenizer_class":"Qwen2Tokenizer","added_tokens_decoder":{"0":{"content":"b"}}}
    , .{}));
    try std.testing.expectError(error.InvalidVocabJson, load(gpa, "[1]", test_merges, "{\"tokenizer_class\":\"Qwen2Tokenizer\"}", .{}));
}
