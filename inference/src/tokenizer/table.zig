//! The token table both Hugging Face sources share (hf_json.zig's
//! `tokenizer.json`, vocab_merges.zig's `vocab.json` with
//! `added_tokens_decoder`): a model vocabulary and its added tokens, by id,
//! dense from zero. An added token may fill a gap or sit over a vocabulary
//! entry that spells it exactly, which it re-types; any other overlap is
//! `InvalidAddedToken`. Each caller checks its format's flags first.
//! docs/engine/tokenizer.md § Hugging Face `tokenizer.json`.
const std = @import("std");
const vocabulary = @import("vocabulary.zig");

pub const Error = std.mem.Allocator.Error || error{ InvalidToken, DuplicateToken, InvalidAddedToken, LimitExceeded };

pub const Limits = struct {
    tokens: usize,
    /// Longest added token.
    added_bytes: usize,
};

/// One added token, flags already checked against its format.
pub const Added = struct {
    id: u32,
    content: []const u8,
    special: bool,
    lstrip: bool,
    normalized: bool,
};

/// Arena-owned by the `alloc` passed to `build`.
pub const Table = struct {
    tokens: []vocabulary.Token,
    ids: std.StringHashMapUnmanaged(u32),
    /// Added-token ids of each pass, longest text first, ties by id.
    raw: []const u32,
    normalized: []const u32,
    lstrip: std.bit_set.Dynamic,
};

/// Copies `text` into `alloc`, charged to `remaining`.
pub fn copyText(alloc: std.mem.Allocator, text: []const u8, remaining: *usize) error{ OutOfMemory, LimitExceeded }![]const u8 {
    if (text.len > remaining.*) return error.LimitExceeded;
    remaining.* -= text.len;
    return alloc.dupe(u8, text);
}

/// Builds the table into `alloc` (an arena the caller frees on error);
/// `scratch` holds what does not outlive the call. Every id below the
/// largest must be filled. Added tokens become `.control` when `special`,
/// else `.user_defined`.
pub fn build(
    alloc: std.mem.Allocator,
    scratch: std.mem.Allocator,
    vocab: std.StringArrayHashMapUnmanaged(u32),
    added: []const Added,
    limits: Limits,
    remaining: *usize,
) Error!Table {
    var count: usize = 0;
    for (vocab.values()) |id| count = @max(count, @as(usize, id) + 1);
    for (added) |a| count = @max(count, @as(usize, a.id) + 1);
    if (count > limits.tokens) return error.LimitExceeded;

    const tokens = try alloc.alloc(vocabulary.Token, count);
    const filled = try scratch.alloc(bool, count);
    @memset(filled, false);
    var ids: std.StringHashMapUnmanaged(u32) = .empty;
    try ids.ensureTotalCapacity(alloc, @intCast(count));
    var it = vocab.iterator();
    while (it.next()) |entry| {
        const id = entry.value_ptr.*;
        const text = entry.key_ptr.*;
        if (filled[id] or text.len == 0 or !std.unicode.utf8ValidateSlice(text)) return error.InvalidToken;
        const owned = try copyText(alloc, text, remaining);
        const slot = ids.getOrPutAssumeCapacity(owned);
        if (slot.found_existing) return error.DuplicateToken;
        slot.value_ptr.* = id;
        tokens[id] = .{ .text = owned, .kind = .normal };
        filled[id] = true;
    }

    var lstrip = try std.bit_set.Dynamic.initEmpty(alloc, count);
    var raw: std.ArrayList(u32) = .empty;
    var normalized: std.ArrayList(u32) = .empty;
    for (added) |a| {
        if (a.content.len == 0 or a.content.len > limits.added_bytes or !std.unicode.utf8ValidateSlice(a.content))
            return error.InvalidAddedToken;
        const kind: vocabulary.TokenType = if (a.special) .control else .user_defined;
        if (filled[a.id]) {
            if (!std.mem.eql(u8, tokens[a.id].text, a.content) or tokens[a.id].kind != .normal) return error.InvalidAddedToken;
            tokens[a.id].kind = kind;
        } else {
            const owned = try copyText(alloc, a.content, remaining);
            const slot = ids.getOrPutAssumeCapacity(owned);
            if (slot.found_existing) return error.DuplicateToken;
            slot.value_ptr.* = a.id;
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
    return .{ .tokens = tokens, .ids = ids, .raw = raw.items, .normalized = normalized.items, .lstrip = lstrip };
}

const test_limits: Limits = .{ .tokens = 16, .added_bytes = 8 };

fn testVocab(alloc: std.mem.Allocator, words: []const []const u8) !std.StringArrayHashMapUnmanaged(u32) {
    var map: std.StringArrayHashMapUnmanaged(u32) = .empty;
    for (words, 0..) |w, id| try map.put(alloc, w, @intCast(id));
    return map;
}

test "added tokens fill gaps, re-type an exact overlay, and sort longest first by pass" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const vocab = try testVocab(a, &.{ "a", "b", "<s>" });
    var remaining: usize = 64;
    const t = try build(a, a, vocab, &.{
        .{ .id = 2, .content = "<s>", .special = true, .lstrip = false, .normalized = false },
        .{ .id = 3, .content = "<mask>", .special = false, .lstrip = true, .normalized = false },
        .{ .id = 4, .content = "  ", .special = false, .lstrip = false, .normalized = true },
    }, test_limits, &remaining);
    try std.testing.expectEqual(@as(usize, 5), t.tokens.len);
    try std.testing.expectEqual(vocabulary.TokenType.control, t.tokens[2].kind);
    try std.testing.expectEqual(vocabulary.TokenType.user_defined, t.tokens[3].kind);
    try std.testing.expectEqual(@as(?u32, 3), t.ids.get("<mask>"));
    try std.testing.expectEqualSlices(u32, &.{ 3, 2 }, t.raw);
    try std.testing.expectEqualSlices(u32, &.{4}, t.normalized);
    try std.testing.expect(t.lstrip.isSet(3) and !t.lstrip.isSet(2));
    // The overlay copies nothing: a, b, <s>, <mask>, and the two spaces.
    try std.testing.expectEqual(@as(usize, 64 - 13), remaining);
}

test "an overlay that spells another entry, a second overlay, gaps, and limits are refused" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const vocab = try testVocab(a, &.{ "a", "b" });
    const cases = [_]struct { added: []const Added, err: Error }{
        .{ .added = &.{.{ .id = 0, .content = "b", .special = true, .lstrip = false, .normalized = false }}, .err = error.InvalidAddedToken },
        .{ .added = &.{
            .{ .id = 0, .content = "a", .special = true, .lstrip = false, .normalized = false },
            .{ .id = 0, .content = "a", .special = false, .lstrip = false, .normalized = false },
        }, .err = error.InvalidAddedToken },
        .{ .added = &.{.{ .id = 2, .content = "b", .special = true, .lstrip = false, .normalized = false }}, .err = error.DuplicateToken },
        .{ .added = &.{.{ .id = 3, .content = "<x>", .special = true, .lstrip = false, .normalized = false }}, .err = error.InvalidToken },
        .{ .added = &.{.{ .id = 2, .content = "<toolong>", .special = true, .lstrip = false, .normalized = false }}, .err = error.InvalidAddedToken },
        .{ .added = &.{.{ .id = 16, .content = "<x>", .special = true, .lstrip = false, .normalized = false }}, .err = error.LimitExceeded },
    };
    for (cases) |c| {
        var remaining: usize = 64;
        try std.testing.expectError(c.err, build(a, a, vocab, c.added, test_limits, &remaining));
    }
    var remaining: usize = 1;
    try std.testing.expectError(error.LimitExceeded, build(a, a, vocab, &.{}, test_limits, &remaining));
}
