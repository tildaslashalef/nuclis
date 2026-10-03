//! A Hugging Face `tokenizer.json` with a BPE model in one of two shapes,
//! loaded into the shared Vocabulary and encoded the way that library encodes
//! without special-token insertion. Added tokens split the text in two
//! passes, each the leftmost-longest match: first those with `normalized:
//! false` on the raw text, then, after normalization, those with `normalized:
//! true`. What remains is split and merged by the shape:
//! - byte-level (ModernBERT): NFC or no normalizer, the GPT-2 pattern
//!   (gpt2.zig), byte-alphabet merges;
//! - Metaspace (mmBERT, a Gemma vocabulary): spaces become U+2581, one is
//!   prepended to each gap, each U+2581 starts a piece, and pieces merge as
//!   SentencePiece BPE with byte fallback (`bpe.encodeSpmBudget`).
//! Other models, normalizers, pre-tokenizers, decoders, and added-token flags
//! are rejected by name. The post-processor, truncation, and padding are
//! call-time policy and are ignored: encode never adds tokens.
const std = @import("std");
const alloc_check = @import("../alloc_check.zig");
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

/// The accepted shapes, named by their pre-tokenizer.
pub const Shape = enum { byte_level, metaspace };

pub const Tokenizer = struct {
    /// `pre` is "gpt2" (byte-level: normal tokens decode through the byte
    /// alphabet) or "metaspace" (`<0xNN>` tokens are `.byte`, U+2581 decodes
    /// to a space); added tokens (control or user-defined) are their raw text.
    vocab: vocabulary.Vocabulary,
    shape: Shape,
    /// Added-token ids of each pass, longest text first; arena-owned.
    raw: []const u32,
    normalized: []const u32,
    lstrip: std.bit_set.Dynamic,
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
            .normalized => switch (self.tokenizer.shape) {
                .byte_level => {
                    var it = try gpt2.Iterator.init(text);
                    while (it.next()) |text_piece| try self.piece(text_piece);
                },
                .metaspace => try self.metaspace(text),
            },
        }
    }

    /// Spaces become U+2581, one is prepended unless the text starts with
    /// one, and each U+2581 starts a new piece.
    fn metaspace(self: *Encoding, text: []const u8) EncodeError!void {
        var spaces: usize = 0;
        for (text) |byte| spaces += @intFromBool(byte == ' ');
        const prepend = text[0] != ' ' and !std.mem.startsWith(u8, text, escaped_space);
        const escaped = try self.alloc.alloc(u8, text.len + spaces * 2 + @as(usize, if (prepend) 3 else 0));
        defer self.alloc.free(escaped);
        var n: usize = 0;
        if (prepend) {
            @memcpy(escaped[0..3], escaped_space);
            n = 3;
        }
        for (text) |byte| if (byte == ' ') {
            @memcpy(escaped[n..][0..3], escaped_space);
            n += 3;
        } else {
            escaped[n] = byte;
            n += 1;
        };
        // U+2581's lead byte never occurs inside another character, so a
        // byte search finds only whole ones.
        var start: usize = 0;
        while (start < escaped.len) {
            const end = std.mem.indexOfPos(u8, escaped, start + 1, escaped_space) orelse escaped.len;
            try self.piece(escaped[start..end]);
            start = end;
        }
    }

    fn piece(self: *Encoding, text: []const u8) EncodeError!void {
        var piece_limits = self.limits.piece;
        piece_limits.output_tokens = @min(piece_limits.output_tokens, self.limits.output_tokens - self.output.items.len);
        const vocab = &self.tokenizer.vocab;
        const ids = switch (self.tokenizer.shape) {
            .byte_level => try bpe.encodePieceBudget(self.alloc, vocab, text, piece_limits, &self.bpe_work),
            .metaspace => try bpe.encodeSpmBudget(self.alloc, vocab, text, piece_limits, &self.bpe_work),
        };
        defer self.alloc.free(ids);
        try self.output.appendSlice(self.alloc, ids);
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
    const splitter = file.pre_tokenizer orelse return error.UnsupportedPreTokenizer;
    const shape: Shape = if (isType(splitter, "ByteLevel")) .byte_level else if (isType(splitter, "Metaspace")) .metaspace else return error.UnsupportedPreTokenizer;
    if (!std.mem.eql(u8, model.type, "BPE") or model.dropout != null or
        nonEmpty(model.continuing_subword_prefix) or nonEmpty(model.end_of_word_suffix) or model.ignore_merges)
        return error.UnsupportedModel;
    var use_nfc = false;
    switch (shape) {
        .byte_level => {
            if (model.unk_token != null or model.byte_fallback) return error.UnsupportedModel;
            if (file.normalizer) |n| {
                if (!isType(n, "NFC")) return error.UnsupportedNormalizer;
                use_nfc = true;
            }
            if (flag(splitter, "add_prefix_space", false) or !flag(splitter, "use_regex", true))
                return error.UnsupportedPreTokenizer;
            if (file.decoder) |d| if (!isType(d, "ByteLevel")) return error.UnsupportedDecoder;
        },
        .metaspace => {
            // `unk_token` is never produced: the byte tokens cover every input (below).
            if (!model.byte_fallback) return error.UnsupportedModel;
            if (!isReplace(file.normalizer orelse return error.UnsupportedNormalizer, " ", escaped_space))
                return error.UnsupportedNormalizer;
            if (!isString(splitter, "replacement", escaped_space) or !isString(splitter, "prepend_scheme", "always") or !flag(splitter, "split", false))
                return error.UnsupportedPreTokenizer;
            if (file.decoder) |d| if (!isMetaspaceDecoder(d)) return error.UnsupportedDecoder;
        },
    }

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

    var lstrip = try std.bit_set.Dynamic.initEmpty(alloc, count);
    var raw: std.ArrayList(u32) = .empty;
    var normalized: std.ArrayList(u32) = .empty;
    for (file.added_tokens) |a| {
        if (a.single_word or a.rstrip or (a.normalized and shape == .metaspace)) return error.UnsupportedAddedToken;
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
    if (shape == .metaspace) try markByteTokens(tokens, &ids);
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
        .vocab = .{ .storage = arena, .tokens = tokens, .pre = if (shape == .metaspace) "metaspace" else "gpt2", .bos = null, .eos = null, .padding = null, .token_ids = ids, .merge_ranks = ranks },
        .shape = shape,
        .raw = raw.items,
        .normalized = normalized.items,
        .lstrip = lstrip,
        .nfc = use_nfc,
    };
}

const escaped_space = "\u{2581}";

/// Marks the `<0xNN>` tokens `.byte`. Byte fallback must never miss: a byte
/// without its token is accepted only when no valid UTF-8 input can reach
/// it (never a UTF-8 byte, or ASCII spelled by a token of its own).
fn markByteTokens(tokens: []vocabulary.Token, ids: *const std.StringHashMapUnmanaged(u32)) Error!void {
    const hex = "0123456789ABCDEF";
    for (0..256) |b| {
        const name = [_]u8{ '<', '0', 'x', hex[b >> 4], hex[b & 15], '>' };
        if (ids.get(&name)) |id| {
            if (tokens[id].kind != .normal) return error.InvalidToken;
            tokens[id].kind = .byte;
        } else if (b >= 0x80) {
            if (b != 0xC0 and b != 0xC1 and b < 0xF5) return error.UnsupportedModel;
        } else if (ids.get(&[_]u8{@intCast(b)}) == null) return error.UnsupportedModel;
    }
}

/// `{"type":"Replace","pattern":{"String":from},"content":to}`.
fn isReplace(value: std.json.Value, from: []const u8, to: []const u8) bool {
    if (!isType(value, "Replace") or !isString(value, "content", to)) return false;
    const pattern = value.object.get("pattern") orelse return false;
    return pattern == .object and pattern.object.count() == 1 and isString(pattern, "String", from);
}

/// U+2581 back to spaces, then byte tokens to bytes, then one string.
fn isMetaspaceDecoder(value: std.json.Value) bool {
    if (!isType(value, "Sequence")) return false;
    const steps = value.object.get("decoders") orelse return false;
    if (steps != .array or steps.array.items.len != 3) return false;
    const items = steps.array.items;
    return isReplace(items[0], escaped_space, " ") and isType(items[1], "ByteFallback") and isType(items[2], "Fuse");
}

fn isString(value: std.json.Value, name: []const u8, expected: []const u8) bool {
    const field = value.object.get(name) orelse return false;
    return field == .string and std.mem.eql(u8, field.string, expected);
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
    try alloc_check.checkAll(std.testing.allocator, struct {
        fn run(gpa: std.mem.Allocator) !void {
            var t = try parse(gpa, fixture, .{});
            defer t.deinit();
            const ids = try t.encode(gpa, "b   [MASK]e\u{301} bb", .{});
            gpa.free(ids);
        }
    }.run, .{});
}

/// A Metaspace tokenizer: the 256 byte tokens are ids 0..255, `replace`
/// edits the JSON, and `drop` renames one byte token (`""` for none).
fn metaspaceFixture(gpa: std.mem.Allocator, replace: [2][]const u8, drop: []const u8) !Tokenizer {
    var json: std.Io.Writer.Allocating = .init(gpa);
    defer json.deinit();
    const w = &json.writer;
    try w.writeAll(
        \\{"added_tokens":[
        \\  {"id":256,"content":"<unk>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
        \\  {"id":257,"content":"<mask>","single_word":false,"lstrip":true,"rstrip":false,"normalized":false,"special":true},
        \\  {"id":264,"content":"\n\n","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":false}],
        \\ "normalizer":{"type":"Replace","pattern":{"String":" "},"content":"▁"},
        \\ "pre_tokenizer":{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true},
        \\ "decoder":{"type":"Sequence","decoders":[{"type":"Replace","pattern":{"String":"▁"},"content":" "},{"type":"ByteFallback"},{"type":"Fuse"}]},
        \\ "model":{"type":"BPE","dropout":null,"unk_token":"<unk>","fuse_unk":true,"byte_fallback":true,"vocab":{
    );
    for (0..256) |b| {
        var name: [6]u8 = undefined;
        _ = try std.fmt.bufPrint(&name, "<0x{X:0>2}>", .{b});
        // A dropped byte token keeps its id under another name: ids stay dense.
        if (std.mem.eql(u8, &name, drop)) try w.print("\"r{d}\":{d},", .{ b, b }) else try w.print("\"{s}\":{d},", .{ name, b });
    }
    try w.writeAll(
        \\"<unk>":256,"<mask>":257,"▁":258,"a":259,"b":260,"▁a":261,"ab":262,"▁ab":263,"\n\n":264},
        \\ "merges":[["▁","a"],["a","b"],["▁a","b"]]}}
    );
    if (replace[0].len == 0) return parse(gpa, json.written(), .{});
    const bytes = try std.mem.replaceOwned(u8, gpa, json.written(), replace[0], replace[1]);
    defer gpa.free(bytes);
    return parse(gpa, bytes, .{});
}

test "Metaspace: a U+2581 prepended per gap, one piece per U+2581, byte fallback" {
    const gpa = std.testing.allocator;
    var t = try metaspaceFixture(gpa, .{ "", "" }, "");
    defer t.deinit();
    try std.testing.expectEqual(Shape.metaspace, t.shape);
    try std.testing.expectEqual(vocabulary.TokenType.byte, t.vocab.tokens[0xA9].kind);
    try expectIds(&t, "", &.{});
    try expectIds(&t, "ab", &.{263});
    try expectIds(&t, " ab", &.{263});
    try expectIds(&t, "a  b", &.{ 261, 258, 258, 260 });
    // `<mask>` swallows the spaces before it; the gap after it gets its own U+2581.
    try expectIds(&t, "a <mask>b", &.{ 261, 257, 258, 260 });
    // Characters the vocabulary lacks fall back to their bytes, unmerged.
    try expectIds(&t, "\u{e9}", &.{ 258, 0xC3, 0xA9 });
    try expectIds(&t, "x\n\nab", &.{ 258, 'x', 264, 263 });
    const decoded = try bpe.decode(gpa, &t.vocab, &.{ 261, 257, 258, 0xC3, 0xA9, 264 }, true, .{});
    defer gpa.free(decoded);
    try std.testing.expectEqualStrings(" a<mask> \u{e9}\n\n", decoded);
}

test "Metaspace: other normalizers, schemes, and uncovered bytes are rejected" {
    const gpa = std.testing.allocator;
    const none: [2][]const u8 = .{ "", "" };
    try std.testing.expectError(error.UnsupportedNormalizer, metaspaceFixture(gpa, .{ "\"content\":\"▁\"}", "\"content\":\"_\"}" }, ""));
    try std.testing.expectError(error.UnsupportedPreTokenizer, metaspaceFixture(gpa, .{ "\"always\"", "\"first\"" }, ""));
    try std.testing.expectError(error.UnsupportedDecoder, metaspaceFixture(gpa, .{ "{\"type\":\"Fuse\"}", "{\"type\":\"Strip\"}" }, ""));
    try std.testing.expectError(error.UnsupportedModel, metaspaceFixture(gpa, .{ "\"byte_fallback\":true", "\"byte_fallback\":false" }, ""));
    try std.testing.expectError(error.UnsupportedAddedToken, metaspaceFixture(gpa, .{ "\"normalized\":false,\"special\":false", "\"normalized\":true,\"special\":false" }, ""));
    // A UTF-8 byte, or ASCII without a token of its own, must have its byte token.
    try std.testing.expectError(error.UnsupportedModel, metaspaceFixture(gpa, none, "<0x80>"));
    try std.testing.expectError(error.UnsupportedModel, metaspaceFixture(gpa, none, "<0x78>"));
    // Neither can reach byte fallback: 0xC0 is never UTF-8, `a` is a token.
    inline for (.{ "<0xC0>", "<0x61>" }) |drop| {
        var t = try metaspaceFixture(gpa, none, drop);
        t.deinit();
    }
}
