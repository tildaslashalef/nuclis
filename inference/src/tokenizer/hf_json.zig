//! A Hugging Face `tokenizer.json` with a byte-level BPE model, loaded into
//! the shared Vocabulary and encoded the way that library encodes without
//! special-token insertion: added tokens split the text in two passes, each
//! the leftmost-longest match; first those with `normalized: false` on the
//! raw text, then, after NFC, those with `normalized: true`; what remains is
//! split by the GPT-2 pattern (gpt2.zig) and merged by bpe.zig. Other
//! models, normalizers, pre-tokenizers, decoders, and added-token flags are
//! rejected by name. The post-processor, truncation, and padding are
//! call-time policy and are ignored: encode never adds tokens.
const std = @import("std");
const vocabulary = @import("vocabulary.zig");
const bpe = @import("bpe.zig");
const encoder = @import("encode.zig");
const gpt2 = @import("gpt2.zig");
pub const nfc = @import("nfc.zig");
const pre = @import("pre.zig");

pub const Limits = struct {
    tokens: u32 = 1_000_000,
    merges: u32 = 1_000_000,
    added: u32 = 4096,
    /// Longest added token; markers and whitespace runs are short.
    added_bytes: usize = 256,
    /// Total copied token and merge string bytes.
    text_bytes: usize = 64 * 1024 * 1024,
};

pub const Error = std.mem.Allocator.Error || error{
    InvalidTokenizerJson,
    UnsupportedModel,
    UnsupportedNormalizer,
    UnsupportedPreTokenizer,
    UnsupportedDecoder,
    UnsupportedAddedToken,
    InvalidAddedToken,
    InvalidToken,
    DuplicateToken,
    InvalidMerge,
    DuplicateMerge,
    LimitExceeded,
};

pub const EncodeError = encoder.Error || nfc.Error;

pub const Tokenizer = struct {
    /// `pre` is "gpt2": normal tokens decode through the byte alphabet, added
    /// tokens (control or user-defined) are their raw text.
    vocab: vocabulary.Vocabulary,
    /// Added-token ids of each pass, longest text first; arena-owned.
    raw: []const u32,
    normalized: []const u32,
    lstrip: std.DynamicBitSetUnmanaged,
    nfc: bool,

    pub fn deinit(self: *Tokenizer) void {
        self.vocab.deinit();
        self.* = undefined;
    }

    /// Returns owned ids, freed with `alloc`; `limits.special_work` bounds the
    /// added-token matching. Never adds special tokens.
    pub fn encode(self: *const Tokenizer, alloc: std.mem.Allocator, text: []const u8, limits: encoder.Limits) EncodeError![]u32 {
        if (text.len > limits.input_bytes) return error.LimitExceeded;
        if (!std.unicode.utf8ValidateSlice(text)) return error.InvalidUtf8;
        var state: Encoding = .{ .tokenizer = self, .alloc = alloc, .limits = limits, .match_work = limits.special_work, .bpe_work = limits.bpe_work };
        errdefer state.output.deinit(alloc);
        try state.split(text, self.raw, .raw);
        return state.output.toOwnedSlice(alloc);
    }
};

const Pass = enum { raw, normalized };

const Encoding = struct {
    tokenizer: *const Tokenizer,
    alloc: std.mem.Allocator,
    limits: encoder.Limits,
    output: std.ArrayList(u32) = .empty,
    match_work: usize,
    bpe_work: usize,

    /// Emits `text`'s added-token matches from `ids` and hands each gap to
    /// the next stage.
    fn split(self: *Encoding, text: []const u8, ids: []const u32, pass: Pass) EncodeError!void {
        var done: usize = 0;
        while (try self.find(text, done, ids)) |match| {
            var start = match.start;
            if (self.tokenizer.lstrip.isSet(match.id)) {
                while (start > done) {
                    const back = previous(text, start);
                    if (!pre.charAt(text, back).space()) break;
                    start = back;
                }
            }
            try self.gap(text[done..start], pass);
            try self.emit(match.id);
            done = match.end;
        }
        try self.gap(text[done..], pass);
    }

    fn gap(self: *Encoding, text: []const u8, pass: Pass) EncodeError!void {
        if (text.len == 0) return;
        switch (pass) {
            .raw => {
                const normalized = if (self.tokenizer.nfc) try nfc.normalize(self.alloc, text) else null;
                defer if (normalized) |n| self.alloc.free(n);
                try self.split(normalized orelse text, self.tokenizer.normalized, .normalized);
            },
            .normalized => {
                var it = try gpt2.Iterator.init(text);
                while (it.next()) |piece| {
                    var piece_limits = self.limits.piece;
                    piece_limits.output_tokens = @min(piece_limits.output_tokens, self.limits.output_tokens - self.output.items.len);
                    const ids = try bpe.encodePieceBudget(self.alloc, &self.tokenizer.vocab, piece, piece_limits, &self.bpe_work);
                    defer self.alloc.free(ids);
                    try self.output.appendSlice(self.alloc, ids);
                }
            },
        }
    }

    fn emit(self: *Encoding, id: u32) EncodeError!void {
        if (self.output.items.len == self.limits.output_tokens) return error.LimitExceeded;
        try self.output.append(self.alloc, id);
    }

    const Match = struct { start: usize, end: usize, id: u32 };

    /// The leftmost match at or after `from`, the longest there (`ids` is
    /// sorted longest first). Each comparison is charged to the budget.
    fn find(self: *Encoding, text: []const u8, from: usize, ids: []const u32) EncodeError!?Match {
        if (ids.len == 0) return null;
        const tokens = self.tokenizer.vocab.tokens;
        for (from..text.len) |at| {
            for (ids) |id| {
                const t = tokens[id].text;
                if (t[0] != text[at]) continue;
                try spend(&self.match_work, t.len);
                if (std.mem.startsWith(u8, text[at..], t)) return .{ .start = at, .end = at + t.len, .id = id };
            }
            try spend(&self.match_work, 1);
        }
        return null;
    }
};

fn previous(text: []const u8, at: usize) usize {
    var back = at - 1;
    while (back > 0 and text[back] & 0xC0 == 0x80) back -= 1;
    return back;
}

fn spend(work: *usize, amount: usize) error{WorkLimitExceeded}!void {
    if (amount > work.*) return error.WorkLimitExceeded;
    work.* -= amount;
}

const AddedJson = struct {
    id: u32,
    content: []const u8,
    single_word: bool = false,
    lstrip: bool = false,
    rstrip: bool = false,
    normalized: bool = true,
    special: bool = false,
};

const File = struct {
    added_tokens: []const AddedJson = &.{},
    normalizer: ?std.json.Value = null,
    pre_tokenizer: ?std.json.Value = null,
    decoder: ?std.json.Value = null,
    model: struct {
        type: []const u8,
        dropout: ?f64 = null,
        unk_token: ?[]const u8 = null,
        continuing_subword_prefix: ?[]const u8 = null,
        end_of_word_suffix: ?[]const u8 = null,
        byte_fallback: bool = false,
        ignore_merges: bool = false,
        vocab: std.json.ArrayHashMap(u32),
        merges: []const std.json.Value,
    },
};

/// Parses a whole `tokenizer.json` image; the caller owns reading and bounding
/// the file. `bytes` may be freed once this returns.
pub fn parse(gpa: std.mem.Allocator, bytes: []const u8, limits: Limits) Error!Tokenizer {
    var scratch: std.heap.ArenaAllocator = .init(gpa);
    defer scratch.deinit();
    const file = std.json.parseFromSliceLeaky(File, scratch.allocator(), bytes, .{
        .ignore_unknown_fields = true,
        .duplicate_field_behavior = .@"error",
        .max_value_len = bytes.len,
    }) catch |err| return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.InvalidTokenizerJson,
    };
    const model = file.model;
    if (!std.mem.eql(u8, model.type, "BPE") or model.dropout != null or model.unk_token != null or
        nonEmpty(model.continuing_subword_prefix) or nonEmpty(model.end_of_word_suffix) or
        model.byte_fallback or model.ignore_merges) return error.UnsupportedModel;
    const use_nfc = if (file.normalizer) |n| blk: {
        if (!isType(n, "NFC")) return error.UnsupportedNormalizer;
        break :blk true;
    } else false;
    const splitter = file.pre_tokenizer orelse return error.UnsupportedPreTokenizer;
    if (!isType(splitter, "ByteLevel") or flag(splitter, "add_prefix_space", false) or !flag(splitter, "use_regex", true))
        return error.UnsupportedPreTokenizer;
    if (file.decoder) |d| if (!isType(d, "ByteLevel")) return error.UnsupportedDecoder;

    if (model.vocab.map.count() == 0 or model.merges.len > limits.merges or file.added_tokens.len > limits.added)
        return error.LimitExceeded;
    // Ids are the model's and the added tokens' together, dense from zero.
    var count: usize = 0;
    for (model.vocab.map.values()) |id| count = @max(count, @as(usize, id) + 1);
    for (file.added_tokens) |a| count = @max(count, @as(usize, a.id) + 1);
    if (count > limits.tokens) return error.LimitExceeded;

    var arena: std.heap.ArenaAllocator = .init(gpa);
    errdefer arena.deinit();
    const alloc = arena.allocator();
    var remaining = limits.text_bytes;
    const tokens = try alloc.alloc(vocabulary.Token, count);
    const filled = try alloc.alloc(bool, count);
    @memset(filled, false);
    var ids: std.StringHashMapUnmanaged(u32) = .empty;
    try ids.ensureTotalCapacity(alloc, @intCast(count));
    var it = model.vocab.map.iterator();
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

    var lstrip = try std.DynamicBitSetUnmanaged.initEmpty(alloc, count);
    var raw: std.ArrayList(u32) = .empty;
    var normalized: std.ArrayList(u32) = .empty;
    for (file.added_tokens) |a| {
        if (a.single_word or a.rstrip) return error.UnsupportedAddedToken;
        if (a.content.len == 0 or a.content.len > limits.added_bytes or !std.unicode.utf8ValidateSlice(a.content))
            return error.InvalidAddedToken;
        const kind: vocabulary.TokenType = if (a.special) .control else .user_defined;
        if (filled[a.id]) {
            // An added token over a vocabulary entry must spell it exactly.
            if (!std.mem.eql(u8, tokens[a.id].text, a.content) or tokens[a.id].kind != .normal) return error.InvalidAddedToken;
            tokens[a.id].kind = kind;
        } else {
            const owned = try copyText(alloc, a.content, &remaining);
            const entry = ids.getOrPutAssumeCapacity(owned);
            if (entry.found_existing) return error.DuplicateToken;
            entry.value_ptr.* = a.id;
            tokens[a.id] = .{ .text = owned, .kind = kind };
            filled[a.id] = true;
        }
        if (a.lstrip) lstrip.set(a.id);
        try (if (a.normalized) &normalized else &raw).append(alloc, a.id);
    }
    for (filled) |f| if (!f) return error.InvalidToken;
    for ([_]*std.ArrayList(u32){ &raw, &normalized }) |list| std.mem.sort(u32, list.items, tokens, struct {
        fn longer(t: []const vocabulary.Token, a: u32, b: u32) bool {
            return if (t[a].text.len != t[b].text.len) t[a].text.len > t[b].text.len else a < b;
        }
    }.longer);

    var ranks: std.StringHashMapUnmanaged(u32) = .empty;
    try ranks.ensureTotalCapacity(alloc, @intCast(model.merges.len));
    for (model.merges, 0..) |merge, rank| {
        const key = switch (merge) {
            .string => |s| blk: {
                const space = std.mem.indexOfScalar(u8, s, ' ') orelse return error.InvalidMerge;
                if (space == 0 or space + 1 == s.len or std.mem.indexOfScalarPos(u8, s, space + 1, ' ') != null)
                    return error.InvalidMerge;
                break :blk try copyText(alloc, s, &remaining);
            },
            .array => |pair| blk: {
                if (pair.items.len != 2) return error.InvalidMerge;
                var parts: [2][]const u8 = undefined;
                for (&parts, pair.items) |*part, item| {
                    part.* = switch (item) {
                        .string => |s| s,
                        else => return error.InvalidMerge,
                    };
                    if (part.len == 0 or std.mem.indexOfScalar(u8, part.*, ' ') != null) return error.InvalidMerge;
                }
                if (parts[0].len + parts[1].len + 1 > remaining) return error.LimitExceeded;
                remaining -= parts[0].len + parts[1].len + 1;
                break :blk try std.mem.concat(alloc, u8, &.{ parts[0], " ", parts[1] });
            },
            else => return error.InvalidMerge,
        };
        if (!std.unicode.utf8ValidateSlice(key)) return error.InvalidMerge;
        const entry = ranks.getOrPutAssumeCapacity(key);
        if (entry.found_existing) return error.DuplicateMerge;
        entry.value_ptr.* = @intCast(rank);
    }

    return .{
        .vocab = .{ .storage = arena, .tokens = tokens, .pre = "gpt2", .bos = null, .eos = null, .padding = null, .token_ids = ids, .merge_ranks = ranks },
        .raw = raw.items,
        .normalized = normalized.items,
        .lstrip = lstrip,
        .nfc = use_nfc,
    };
}

fn nonEmpty(text: ?[]const u8) bool {
    return if (text) |t| t.len != 0 else false;
}

fn isType(value: std.json.Value, name: []const u8) bool {
    const object = switch (value) {
        .object => |o| o,
        else => return false,
    };
    const kind = object.get("type") orelse return false;
    return kind == .string and std.mem.eql(u8, kind.string, name);
}

/// A boolean field of a JSON object, `default` when absent or not boolean.
fn flag(value: std.json.Value, name: []const u8, default: bool) bool {
    const field = value.object.get(name) orelse return default;
    return if (field == .bool) field.bool else default;
}

fn copyText(alloc: std.mem.Allocator, text: []const u8, remaining: *usize) Error![]const u8 {
    if (text.len > remaining.*) return error.LimitExceeded;
    remaining.* -= text.len;
    return alloc.dupe(u8, text);
}

// A byte-level vocabulary small enough to read: the GPT-2 alphabet spells
// space as `Ġ`, and bytes 0xC3 0xA9 (é) and 0xCC 0xB8 (U+0338) as `Ã©` and `Ì¸`.
const fixture =
    \\{"version":"1.0","truncation":null,"padding":null,
    \\ "added_tokens":[
    \\  {"id":9,"content":"  ","single_word":false,"lstrip":false,"rstrip":false,"normalized":true,"special":false},
    \\  {"id":10,"content":"   ","single_word":false,"lstrip":false,"rstrip":false,"normalized":true,"special":false},
    \\  {"id":11,"content":"[MASK]","single_word":false,"lstrip":true,"rstrip":false,"normalized":false,"special":true},
    \\  {"id":12,"content":"<x>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
    \\  {"id":0,"content":"a","single_word":false,"lstrip":false,"rstrip":false,"normalized":true,"special":false}],
    \\ "normalizer":{"type":"NFC"},
    \\ "pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true},
    \\ "post_processor":null,
    \\ "decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},
    \\ "model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,
    \\  "end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
    \\  "vocab":{"a":0,"b":1,"Ġ":2,"Ġb":3,"bb":4,"Ã":5,"©":6,"Ã©":7,"Ì¸":8,"Ì":13,"¸":14},
    \\  "merges":[["Ġ","b"],"b b",["Ã","©"],["Ì","¸"]]}}
;

fn fixtureWith(gpa: std.mem.Allocator, from: []const u8, to: []const u8) !Tokenizer {
    const bytes = try std.mem.replaceOwned(u8, gpa, fixture, from, to);
    defer gpa.free(bytes);
    return parse(gpa, bytes, .{});
}

fn expectIds(t: *const Tokenizer, text: []const u8, expected: []const u32) !void {
    const gpa = std.testing.allocator;
    const got = try t.encode(gpa, text, .{});
    defer gpa.free(got);
    try std.testing.expectEqualSlices(u32, expected, got);
}

test "added-token passes, lstrip, NFC between them, and byte-level BPE" {
    const gpa = std.testing.allocator;
    var t = try parse(gpa, fixture, .{});
    defer t.deinit();
    try std.testing.expectEqual(@as(usize, 15), t.vocab.tokens.len);
    try std.testing.expectEqual(vocabulary.TokenType.control, t.vocab.tokens[11].kind);
    try std.testing.expectEqual(vocabulary.TokenType.user_defined, t.vocab.tokens[0].kind);
    try expectIds(&t, "", &.{});
    try expectIds(&t, "bb b", &.{ 4, 3 });
    // Whitespace runs are added tokens: leftmost-longest, before splitting.
    try expectIds(&t, "b     b", &.{ 1, 10, 9, 1 });
    // `[MASK]` is matched first, on the raw text, and swallows the spaces before it.
    try expectIds(&t, "b   [MASK]b [MASK]", &.{ 1, 11, 1, 11 });
    // NFC composes é; `<x>` is matched before NFC, so its `>` keeps U+0338.
    try expectIds(&t, "e\u{301}", &.{7});
    try expectIds(&t, "<x>\u{338}", &.{ 12, 8 });
    const decoded = try bpe.decode(gpa, &t.vocab, &.{ 1, 10, 11, 3, 7 }, true, .{});
    defer gpa.free(decoded);
    try std.testing.expectEqualStrings("b   [MASK] b\u{e9}", decoded);
    try std.testing.expectError(error.InvalidUtf8, t.encode(gpa, "\xff", .{}));
    try std.testing.expectError(error.LimitExceeded, t.encode(gpa, "bbbb", .{ .output_tokens = 1 }));
    try std.testing.expectError(error.WorkLimitExceeded, t.encode(gpa, "b b b b", .{ .special_work = 3 }));
}

test "unsupported components and inconsistent tables are rejected by name" {
    const gpa = std.testing.allocator;
    const cases = .{
        .{ "\"type\":\"BPE\"", "\"type\":\"WordPiece\"", error.UnsupportedModel },
        .{ "\"byte_fallback\":false", "\"byte_fallback\":true", error.UnsupportedModel },
        .{ "{\"type\":\"NFC\"}", "{\"type\":\"NFKC\"}", error.UnsupportedNormalizer },
        .{ "\"add_prefix_space\":false", "\"add_prefix_space\":true", error.UnsupportedPreTokenizer },
        .{ "{\"type\":\"ByteLevel\",\"add_prefix_space\":true", "{\"type\":\"WordPiece\",\"add_prefix_space\":true", error.UnsupportedDecoder },
        .{ "\"lstrip\":true,\"rstrip\":false", "\"lstrip\":true,\"rstrip\":true", error.UnsupportedAddedToken },
        .{ "{\"id\":0,\"content\":\"a\"", "{\"id\":0,\"content\":\"b\"", error.InvalidAddedToken },
        .{ "\"id\":12,", "\"id\":20,", error.InvalidToken },
        .{ "\"Ì\":13", "\"Ì\":1", error.InvalidToken },
        .{ "\"b b\"", "\"Ġ b\"", error.DuplicateMerge },
        .{ "\"b b\"", "\"bb\"", error.InvalidMerge },
        .{ "\"model\":{", "\"model\":[", error.InvalidTokenizerJson },
    };
    inline for (cases) |case| {
        try std.testing.expectError(case[2], fixtureWith(gpa, case[0], case[1]));
    }
    try std.testing.expectError(error.LimitExceeded, parse(gpa, fixture, .{ .tokens = 10 }));
}

test "parse and encode release everything on allocation failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, struct {
        fn run(gpa: std.mem.Allocator) !void {
            var t = try parse(gpa, fixture, .{});
            defer t.deinit();
            const ids = try t.encode(gpa, "b   [MASK]e\u{301} bb", .{});
            gpa.free(ids);
        }
    }.run, .{});
}
