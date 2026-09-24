//! GPT-2 byte-alphabet BPE reference for ONE already separated piece.
//! The caller owns pre-tokenization and special-token recognition. Feeding a
//! whole prompt here can merge across forbidden boundaries and is not correct
//! text tokenization. No BOS/EOS or other special tokens are inserted.
//!
//! Merges join the adjacent pair with the lowest rank, the leftmost on ties:
//! a short piece rescans its pairs, a long one (a Gemma line) keeps them in a
//! queue so it costs O(n log n), and a test holds the two to the same order.
//! A byte-work budget bounds the pair lookups.
const std = @import("std");
const Vocabulary = @import("vocabulary.zig").Vocabulary;

pub const Limits = struct {
    /// One piece may be a whole input (a Gemma line has no inner splits);
    /// the queued merge keeps its cost linear in the piece, so the bound is
    /// the encoder's input bound.
    piece_bytes: usize = 1024 * 1024,
    output_tokens: usize = 1024 * 1024,
    /// Sum of encoded pair lengths (including separator) looked up.
    work_bytes: usize = 64 * 1024 * 1024,
};
pub const DecodeLimits = struct {
    input_tokens: usize = 1024 * 1024,
    output_bytes: usize = 4 * 1024 * 1024,
};
pub const Error = std.mem.Allocator.Error || error{
    LimitExceeded,
    WorkLimitExceeded,
    MissingToken,
    InvalidTokenId,
    UnsupportedTokenType,
    InvalidByteEncoding,
};

// The byte alphabet preserves these visible Latin-1 scalars and maps all other
// bytes, in byte order, to U+0100 onward. Each byte becomes one scalar, which
// occupies one or two UTF-8 bytes. This is reversible even for invalid raw UTF-8.
// The mapping is the GPT-2 byte-level alphabet the vocabulary is stored in
// (THIRD_PARTY_NOTICES.md). The merge scan below is independently implemented.
fn visible(byte: u21) bool {
    return (byte >= 33 and byte <= 126) or (byte >= 161 and byte <= 172) or (byte >= 174 and byte <= 255);
}
const alphabet: [256]u21 = blk: {
    var result: [256]u21 = undefined;
    var extra: u21 = 256;
    for (&result, 0..) |*cp, byte| {
        if (visible(@intCast(byte))) cp.* = @intCast(byte) else {
            cp.* = extra;
            extra += 1;
        }
    }
    break :blk result;
};
const inverse: [324]?u8 = blk: {
    var result = [_]?u8{null} ** 324;
    for (alphabet, 0..) |cp, byte| result[cp] = @intCast(byte);
    break :blk result;
};

const Span = struct { start: usize, end: usize, next: usize };

/// Returns owned token IDs; free with `alloc`. Inputs are borrowed for this call.
/// An empty piece yields no IDs. Arbitrary byte sequences are accepted. Final
/// symbols must be normal vocabulary tokens; missing entries fail explicitly.
pub fn encodePiece(alloc: std.mem.Allocator, vocab: *const Vocabulary, bytes: []const u8, limits: Limits) Error![]u32 {
    var remaining = limits.work_bytes;
    return encodePieceBudget(alloc, vocab, bytes, limits, &remaining);
}

/// As encodePiece, but charges an aggregate caller-owned budget across pieces.
/// Failed calls also consume work already performed. The per-piece limit still
/// applies; callers must not reset this budget between pieces of one request.
pub fn encodePieceBudget(alloc: std.mem.Allocator, vocab: *const Vocabulary, bytes: []const u8, limits: Limits, remaining: *usize) Error![]u32 {
    const allowance = @min(remaining.*, limits.work_bytes);
    var work = allowance;
    defer remaining.* -= allowance - work;
    if (bytes.len > limits.piece_bytes or bytes.len > (std.math.maxInt(usize) - 1) / 2)
        return error.LimitExceeded;
    if (bytes.len == 0) return alloc.alloc(u32, 0);
    const encoded = try alloc.alloc(u8, bytes.len * 2);
    defer alloc.free(encoded);
    const pair = try alloc.alloc(u8, bytes.len * 2 + 1);
    defer alloc.free(pair);
    const spans = try alloc.alloc(Span, bytes.len);
    defer alloc.free(spans);
    var used: usize = 0;
    for (bytes, spans, 0..) |byte, *span, i| {
        const len = std.unicode.utf8Encode(alphabet[byte], encoded[used..]) catch unreachable;
        span.* = .{ .start = used, .end = used + len, .next = i + 1 };
        used += len;
    }
    const count = try mergeSpans(alloc, vocab, encoded, spans, pair, &work);
    if (count > limits.output_tokens) return error.LimitExceeded;
    const result = try alloc.alloc(u32, count);
    errdefer alloc.free(result);
    var current: usize = 0;
    for (result) |*id| {
        id.* = vocab.tokenId(encoded[spans[current].start..spans[current].end]) orelse return error.MissingToken;
        if (id.* >= vocab.tokens.len) return error.InvalidTokenId;
        if (vocab.tokens[id.*].kind != .normal) return error.UnsupportedTokenType;
        current = spans[current].next;
    }
    return result;
}

/// One queued merge: the adjacent pair whose left symbol is `left`, valid
/// while both symbols still end where they did when it was queued.
const Candidate = struct { rank: u32, left: usize, right: usize, left_end: usize, right_end: usize };

/// Lowest rank first; among equal ranks the leftmost pair (symbol indices
/// follow text order, and a merge keeps the left index).
fn candidateOrder(_: void, a: Candidate, b: Candidate) std.math.Order {
    if (a.rank != b.rank) return std.math.order(a.rank, b.rank);
    return std.math.order(a.left, b.left);
}
const Queue = std.PriorityQueue(Candidate, void, candidateOrder);
/// `next` of a symbol merged into its left neighbour.
const merged = std.math.maxInt(usize);

/// Symbols up to which a piece rescans its pairs: a word's few merges cost
/// fewer lookups rescanned (n²/2) than queued (3n plus the heap).
const scan_symbols = 32;

/// The rank-ordered merge shared by both alphabets: repeatedly joins the
/// adjacent span pair with the lowest merge rank (leftmost on ties) until no
/// pair is a merge, charging `work` per pair looked up. `spans` is the
/// symbol list (`next` of the last symbol is `spans.len`); returns the
/// symbol count.
fn mergeSpans(alloc: std.mem.Allocator, vocab: *const Vocabulary, encoded: []const u8, spans: []Span, pair: []u8, work: *usize) Error!usize {
    if (spans.len <= scan_symbols) return mergeScan(vocab, encoded, spans, pair, work);
    return mergeQueued(alloc, vocab, encoded, spans, pair, work);
}

/// The merge order by definition: rescan every adjacent pair and join the
/// lowest-ranked, the leftmost on ties. O(n²) lookups.
fn mergeScan(vocab: *const Vocabulary, encoded: []const u8, spans: []Span, pair: []u8, work: *usize) Error!usize {
    var count = spans.len;
    while (count > 1) {
        var best: ?usize = null;
        var best_rank: u32 = std.math.maxInt(u32);
        var left: usize = 0;
        while (spans[left].next < spans.len) {
            const right = spans[left].next;
            if (try lookup(vocab, encoded, spans, pair, work, left)) |rank| {
                // Strict comparison keeps the leftmost occurrence for ties.
                if (best == null or rank < best_rank) {
                    best = left;
                    best_rank = rank;
                }
            }
            left = right;
        }
        const chosen = best orelse break;
        const right = spans[chosen].next;
        spans[chosen].end = spans[right].end;
        spans[chosen].next = spans[right].next;
        count -= 1;
    }
    return count;
}

/// The same order kept by a queue: each pair is looked up once when it forms
/// (at most 3n lookups), and a queued pair whose symbols have since merged is
/// skipped.
fn mergeQueued(alloc: std.mem.Allocator, vocab: *const Vocabulary, encoded: []const u8, spans: []Span, pair: []u8, work: *usize) Error!usize {
    if (spans.len < 2) return spans.len;
    const prev = try alloc.alloc(usize, spans.len);
    defer alloc.free(prev);
    for (prev, 0..) |*p, i| p.* = if (i == 0) merged else i - 1;
    var queue: Queue = .initContext({});
    defer queue.deinit(alloc);
    for (0..spans.len - 1) |left| try queuePair(alloc, &queue, vocab, encoded, spans, pair, work, left);
    var count = spans.len;
    while (queue.pop()) |c| {
        if (spans[c.left].next != c.right or spans[c.left].end != c.left_end or spans[c.right].end != c.right_end) continue;
        const after = spans[c.right].next;
        spans[c.left].end = spans[c.right].end;
        spans[c.left].next = after;
        spans[c.right].next = merged;
        count -= 1;
        if (after < spans.len) {
            prev[after] = c.left;
            try queuePair(alloc, &queue, vocab, encoded, spans, pair, work, c.left);
        }
        if (prev[c.left] != merged) try queuePair(alloc, &queue, vocab, encoded, spans, pair, work, prev[c.left]);
    }
    return count;
}

/// Queues the pair starting at symbol `left` when it is a merge.
fn queuePair(alloc: std.mem.Allocator, queue: *Queue, vocab: *const Vocabulary, encoded: []const u8, spans: []const Span, pair: []u8, work: *usize, left: usize) Error!void {
    const rank = try lookup(vocab, encoded, spans, pair, work, left) orelse return;
    const right = spans[left].next;
    try queue.push(alloc, .{ .rank = rank, .left = left, .right = right, .left_end = spans[left].end, .right_end = spans[right].end });
}

/// The merge rank of the pair starting at symbol `left`, charging its bytes.
fn lookup(vocab: *const Vocabulary, encoded: []const u8, spans: []const Span, pair: []u8, work: *usize, left: usize) Error!?u32 {
    const right = spans[left].next;
    const a = encoded[spans[left].start..spans[left].end];
    const b = encoded[spans[right].start..spans[right].end];
    const size = a.len + 1 + b.len;
    if (size > work.*) return error.WorkLimitExceeded;
    work.* -= size;
    @memcpy(pair[0..a.len], a);
    pair[a.len] = ' ';
    @memcpy(pair[a.len + 1 ..][0..b.len], b);
    return vocab.mergeRank(pair[0..size]);
}

/// SPM-style BPE over one piece (Gemma 4; docs/reference/gemma4.md
/// § Tokenizer): `text` is already in the vocabulary's alphabet (valid UTF-8
/// with spaces escaped to U+2581 by the caller), symbols are code points,
/// the merges are the same rank scan, and a merged symbol with no token
/// falls back to its bytes as `<0xNN>` byte tokens. Budgets as
/// `encodePieceBudget`.
pub fn encodeSpmBudget(alloc: std.mem.Allocator, vocab: *const Vocabulary, text: []const u8, limits: Limits, remaining: *usize) Error![]u32 {
    const allowance = @min(remaining.*, limits.work_bytes);
    var work = allowance;
    defer remaining.* -= allowance - work;
    if (text.len > limits.piece_bytes or text.len > (std.math.maxInt(usize) - 1) / 2) return error.LimitExceeded;
    if (text.len == 0) return alloc.alloc(u32, 0);
    const pair = try alloc.alloc(u8, text.len * 2 + 1);
    defer alloc.free(pair);
    const storage = try alloc.alloc(Span, text.len);
    defer alloc.free(storage);
    var symbols: usize = 0;
    var offset: usize = 0;
    while (offset < text.len) {
        const len = std.unicode.utf8ByteSequenceLength(text[offset]) catch return error.InvalidByteEncoding;
        if (offset + len > text.len) return error.InvalidByteEncoding;
        storage[symbols] = .{ .start = offset, .end = offset + len, .next = symbols + 1 };
        symbols += 1;
        offset += len;
    }
    const spans = storage[0..symbols];
    const count = try mergeSpans(alloc, vocab, text, spans, pair, &work);
    var result: std.ArrayList(u32) = .empty;
    errdefer result.deinit(alloc);
    var current: usize = 0;
    for (0..count) |_| {
        const symbol = text[spans[current].start..spans[current].end];
        if (vocab.tokenId(symbol)) |id| {
            if (vocab.tokens[id].kind != .normal and vocab.tokens[id].kind != .byte) return error.UnsupportedTokenType;
            if (result.items.len == limits.output_tokens) return error.LimitExceeded;
            try result.append(alloc, id);
        } else for (symbol) |byte| {
            if (result.items.len == limits.output_tokens) return error.LimitExceeded;
            try result.append(alloc, byteToken(vocab, byte) orelse return error.MissingToken);
        }
        current = spans[current].next;
    }
    return result.toOwnedSlice(alloc);
}

/// The `<0xNN>` byte token of an SPM-style vocabulary (uppercase hex), or null.
pub fn byteToken(vocab: *const Vocabulary, byte: u8) ?u32 {
    const hex = "0123456789ABCDEF";
    const name = [_]u8{ '<', '0', 'x', hex[byte >> 4], hex[byte & 15], '>' };
    const id = vocab.tokenId(&name) orelse return null;
    return if (vocab.tokens[id].kind == .byte) id else null;
}

/// The byte an SPM `<0xNN>` byte token stands for.
fn byteValue(text: []const u8) Error!u8 {
    if (text.len != 6 or !std.mem.startsWith(u8, text, "<0x") or text[5] != '>') return error.InvalidByteEncoding;
    return std.fmt.parseInt(u8, text[3..5], 16) catch error.InvalidByteEncoding;
}

const escaped_space = "\xE2\x96\x81";

/// SPM normal-token text with U+2581 unescaped to spaces; returns the byte
/// count, writing when `output` is given.
fn decodeSpmNormal(text: []const u8, output: ?[]u8) usize {
    var count: usize = 0;
    var i: usize = 0;
    while (i < text.len) {
        if (text.len - i >= 3 and std.mem.eql(u8, text[i..][0..3], escaped_space)) {
            if (output) |out| out[count] = ' ';
            i += 3;
        } else {
            if (output) |out| out[count] = text[i];
            i += 1;
        }
        count += 1;
    }
    return count;
}

/// Returns owned raw bytes, which may end inside a UTF-8 character. A streaming
/// renderer must retain incomplete characters across calls. `special` controls
/// literal emission of control/user-defined tokens; false omits them. No BOS/EOS
/// removal or cleanup heuristics are applied. Other token kinds fail explicitly.
pub fn decode(alloc: std.mem.Allocator, vocab: *const Vocabulary, ids: []const u32, special: bool, limits: DecodeLimits) Error![]u8 {
    if (ids.len > limits.input_tokens) return error.LimitExceeded;
    // SPM-style vocabularies (Gemma 4): normal tokens unescape U+2581, byte
    // tokens are one byte each, and user-defined tokens (the visible chat
    // markers) are always rendered, as the pinned reference renders them.
    const spm = std.mem.eql(u8, vocab.pre, "gemma4");
    var size: usize = 0;
    for (ids) |id| {
        if (id >= vocab.tokens.len) return error.InvalidTokenId;
        const token = vocab.tokens[id];
        const len = switch (token.kind) {
            .normal => if (spm) decodeSpmNormal(token.text, null) else try decodeNormal(token.text, null),
            .byte => if (spm) blk: {
                _ = try byteValue(token.text);
                break :blk 1;
            } else return error.UnsupportedTokenType,
            .user_defined => if (special or spm) token.text.len else 0,
            .control => if (special) token.text.len else 0,
            else => return error.UnsupportedTokenType,
        };
        if (len > limits.output_bytes - size) return error.LimitExceeded;
        size += len;
    }
    const output = try alloc.alloc(u8, size);
    errdefer alloc.free(output);
    var offset: usize = 0;
    for (ids) |id| {
        const token = vocab.tokens[id];
        switch (token.kind) {
            .normal => offset += if (spm) decodeSpmNormal(token.text, output[offset..]) else try decodeNormal(token.text, output[offset..]),
            .byte => {
                output[offset] = try byteValue(token.text);
                offset += 1;
            },
            .user_defined, .control => if (special or (spm and token.kind == .user_defined)) {
                @memcpy(output[offset..][0..token.text.len], token.text);
                offset += token.text.len;
            },
            else => unreachable,
        }
    }
    return output;
}

// Synthetic SPM-style vocabulary: a handful of U+2581-escaped words, the
// merges that build them, three byte tokens, newline tokens, and one marker
// of each special kind; token ids are positions in this list.
fn spmFixture() !Vocabulary {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    errdefer arena.deinit();
    const alloc = arena.allocator();
    const T = @import("vocabulary.zig").Token;
    const words = [_]T{
        .{ .text = "<bos>", .kind = .control },
        .{ .text = "<|turn>", .kind = .control },
        .{ .text = "<|channel>", .kind = .user_defined },
        .{ .text = "▁", .kind = .normal },
        .{ .text = "h", .kind = .normal },
        .{ .text = "e", .kind = .normal },
        .{ .text = "l", .kind = .normal },
        .{ .text = "o", .kind = .normal },
        .{ .text = "▁h", .kind = .normal },
        .{ .text = "▁he", .kind = .normal },
        .{ .text = "▁hel", .kind = .normal },
        .{ .text = "▁hello", .kind = .normal },
        .{ .text = "lo", .kind = .normal },
        .{ .text = "\n", .kind = .normal },
        .{ .text = "\n\n", .kind = .normal },
        .{ .text = "<0xE2>", .kind = .byte },
        .{ .text = "<0x82>", .kind = .byte },
        .{ .text = "<0xAC>", .kind = .byte },
        .{ .text = "▁▁", .kind = .normal },
    };
    const merges = [_][]const u8{ "▁ h", "▁h e", "l o", "▁he l", "▁hel lo", "\n \n", "▁ ▁" };
    var tokens: std.ArrayList(T) = .empty;
    var ids: std.StringHashMapUnmanaged(u32) = .empty;
    var ranks: std.StringHashMapUnmanaged(u32) = .empty;
    for (words, 0..) |w, id| {
        try tokens.append(alloc, w);
        try ids.put(alloc, w.text, @intCast(id));
    }
    for (merges, 0..) |m, rank| try ranks.put(alloc, m, @intCast(rank));
    return .{ .storage = arena, .tokens = try tokens.toOwnedSlice(alloc), .token_ids = ids, .merge_ranks = ranks, .pre = "gemma4", .bos = 0, .eos = null, .padding = null };
}

test "SPM merges over code points, byte fallback, and unescaping decode" {
    var vocab = try spmFixture();
    defer vocab.deinit();
    var budget: usize = 1 << 20;
    // "▁hello" merges ▁+h, ▁h+e, l+o, ▁he+l, ▁hel+lo in rank order.
    const hello = try encodeSpmBudget(std.testing.allocator, &vocab, "▁hello", .{}, &budget);
    defer std.testing.allocator.free(hello);
    try std.testing.expectEqualSlices(u32, &.{11}, hello);
    // "▁hello€": the euro sign has no token and falls back to three byte tokens.
    const euro = try encodeSpmBudget(std.testing.allocator, &vocab, "▁hello€", .{}, &budget);
    defer std.testing.allocator.free(euro);
    try std.testing.expectEqualSlices(u32, &.{ 11, 15, 16, 17 }, euro);
    // A symbol with no token and no byte tokens is a typed failure.
    try std.testing.expectError(error.MissingToken, encodeSpmBudget(std.testing.allocator, &vocab, "x", .{}, &budget));
    // Decoding unescapes U+2581, rebuilds bytes, renders user-defined markers
    // always and control markers only when asked.
    const text = try decode(std.testing.allocator, &vocab, &.{ 0, 11, 15, 16, 17, 2, 13 }, false, .{});
    defer std.testing.allocator.free(text);
    try std.testing.expectEqualStrings(" hello€<|channel>\n", text);
    const with_special = try decode(std.testing.allocator, &vocab, &.{ 0, 18 }, true, .{});
    defer std.testing.allocator.free(with_special);
    try std.testing.expectEqualStrings("<bos>  ", with_special);
    try std.testing.expectEqual(@as(?u32, 15), byteToken(&vocab, 0xE2));
    try std.testing.expect(byteToken(&vocab, 0x00) == null);
}

fn decodeNormal(text: []const u8, output: ?[]u8) Error!usize {
    const view = std.unicode.Utf8View.init(text) catch return error.InvalidByteEncoding;
    var it = view.iterator();
    var count: usize = 0;
    while (it.nextCodepoint()) |cp| {
        if (cp >= inverse.len) return error.InvalidByteEncoding;
        const byte = inverse[cp] orelse return error.InvalidByteEncoding;
        if (output) |out| out[count] = byte;
        count += 1;
    }
    return count;
}

// Synthetic vocabulary: byte token IDs equal their raw byte values. Additional
// tokens and ordered merge rules exercise the algorithm without model weights.
test "the queued merge picks exactly the scan's pairs" {
    // Merges over a three-letter alphabet with ties, chains, and pairs whose
    // halves are themselves merges, so orders interact on long runs.
    var vocab = try fixture(&.{ "a b", "b a", "ab a", "a a", "ab ab", "c a", "b c", "abab c", "aa b", "ca ab", "ba ba", "c c" }, &.{});
    defer vocab.deinit();
    var prng: std.Random.DefaultPrng = .init(0x5eed);
    const random = prng.random();
    var text: [300]u8 = undefined;
    var fast: [300]Span = undefined;
    var slow: [300]Span = undefined;
    var pair: [601]u8 = undefined;
    for (0..400) |round| {
        const len = 1 + random.uintLessThan(usize, text.len);
        for (text[0..len]) |*c| c.* = "abc"[random.uintLessThan(usize, if (round % 4 == 0) 2 else 3)];
        for (fast[0..len], slow[0..len], 0..) |*f, *w, i| {
            f.* = .{ .start = i, .end = i + 1, .next = i + 1 };
            w.* = f.*;
        }
        var work: usize = std.math.maxInt(usize);
        const got = try mergeQueued(std.testing.allocator, &vocab, text[0..len], fast[0..len], &pair, &work);
        const want = try mergeScan(&vocab, text[0..len], slow[0..len], &pair, &work);
        try std.testing.expectEqual(want, got);
        var f: usize = 0;
        var w: usize = 0;
        for (0..want) |_| {
            try std.testing.expectEqual(slow[w].end, fast[f].end);
            f = fast[f].next;
            w = slow[w].next;
        }
    }
}

test "the queued merge's work grows linearly with the piece" {
    // A 40,000-symbol run the scan would examine ~10^9 pairs for: the queue
    // looks each pair up once when it forms, within the chat's default budget.
    var vocab = try fixture(&.{ "a b", "ab ab", "abab abab" }, &.{});
    defer vocab.deinit();
    const text = try std.testing.allocator.alloc(u8, 40_000);
    defer std.testing.allocator.free(text);
    for (text, 0..) |*c, i| c.* = "ab"[i % 2];
    const ids = try encodePiece(std.testing.allocator, &vocab, text, .{ .output_tokens = text.len, .work_bytes = 64 * text.len });
    defer std.testing.allocator.free(ids);
    try std.testing.expectEqual(@as(usize, 5_000), ids.len);
}

fn fixture(merges: []const []const u8, extras: []const []const u8) !Vocabulary {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    errdefer arena.deinit();
    const alloc = arena.allocator();
    var tokens: std.ArrayList(@import("vocabulary.zig").Token) = .empty;
    var ids: std.StringHashMapUnmanaged(u32) = .empty;
    var ranks: std.StringHashMapUnmanaged(u32) = .empty;
    for (alphabet, 0..) |cp, id| {
        var buffer: [4]u8 = undefined;
        const len = try std.unicode.utf8Encode(cp, &buffer);
        const text = try alloc.dupe(u8, buffer[0..len]);
        try tokens.append(alloc, .{ .text = text, .kind = .normal });
        try ids.put(alloc, text, @intCast(id));
    }
    for (merges, 0..) |pair, rank| {
        try ranks.put(alloc, pair, @intCast(rank));
        const space = std.mem.indexOfScalar(u8, pair, ' ').?;
        const text = try std.mem.concat(alloc, u8, &.{ pair[0..space], pair[space + 1 ..] });
        if (!ids.contains(text)) {
            try ids.put(alloc, text, @intCast(tokens.items.len));
            try tokens.append(alloc, .{ .text = text, .kind = .normal });
        }
    }
    for (extras) |text| {
        try ids.put(alloc, text, @intCast(tokens.items.len));
        try tokens.append(alloc, .{ .text = text, .kind = .normal });
    }
    const owned = try tokens.toOwnedSlice(alloc);
    return .{ .storage = arena, .tokens = owned, .token_ids = ids, .merge_ranks = ranks, .pre = "synthetic", .bos = null, .eos = null, .padding = null };
}

test "byte alphabet covers all 256 bytes and preserves partial UTF-8" {
    var vocab = try fixture(&.{}, &.{});
    defer vocab.deinit();
    try std.testing.expectEqual(@as(u21, 'Ġ'), alphabet[' ']);
    try std.testing.expectEqual(@as(u21, 'Ċ'), alphabet['\n']);
    try std.testing.expectEqual(@as(u21, 'Ā'), alphabet[0]);
    var raw: [256]u8 = undefined;
    for (&raw, 0..) |*byte, i| byte.* = @intCast(i);
    const ids = try encodePiece(std.testing.allocator, &vocab, &raw, .{});
    defer std.testing.allocator.free(ids);
    for (ids, 0..) |id, i| try std.testing.expectEqual(i, id);
    const output = try decode(std.testing.allocator, &vocab, ids, true, .{});
    defer std.testing.allocator.free(output);
    try std.testing.expectEqualSlices(u8, &raw, output);
    const partial = try decode(std.testing.allocator, &vocab, &.{0xf0}, true, .{});
    defer std.testing.allocator.free(partial);
    try std.testing.expectEqualSlices(u8, "\xf0", partial);
}

test "BPE uses ranks rather than longest match and keeps leftmost ties" {
    var ranked = try fixture(&.{ "b c", "a b" }, &.{"abc"});
    defer ranked.deinit();
    const ids = try encodePiece(std.testing.allocator, &ranked, "abc", .{});
    defer std.testing.allocator.free(ids);
    try std.testing.expectEqualSlices(u32, &.{ 'a', ranked.tokenId("bc").? }, ids);
    var tied = try fixture(&.{"a a"}, &.{});
    defer tied.deinit();
    const ties = try encodePiece(std.testing.allocator, &tied, "aaa", .{});
    defer std.testing.allocator.free(ties);
    try std.testing.expectEqualSlices(u32, &.{ tied.tokenId("aa").?, 'a' }, ties);
}

test "BPE revisits both neighbors and merges byte-alphabet whitespace" {
    var vocab = try fixture(&.{ "b c", "a bc", "abc d", "Ġ a" }, &.{});
    defer vocab.deinit();
    for ([_][]const u8{ "abcd", " a" }, [_][]const u8{ "abcd", "Ġa" }) |raw, encoded| {
        const ids = try encodePiece(std.testing.allocator, &vocab, raw, .{});
        defer std.testing.allocator.free(ids);
        try std.testing.expectEqualSlices(u32, &.{vocab.tokenId(encoded).?}, ids);
        const output = try decode(std.testing.allocator, &vocab, ids, true, .{});
        defer std.testing.allocator.free(output);
        try std.testing.expectEqualStrings(raw, output);
    }
}

test "BPE enforces input, output, and exact work limits" {
    var vocab = try fixture(&.{"a b"}, &.{});
    defer vocab.deinit();
    const empty = try encodePiece(std.testing.allocator, &vocab, "", .{ .piece_bytes = 0, .output_tokens = 0, .work_bytes = 0 });
    defer std.testing.allocator.free(empty);
    try std.testing.expectEqual(@as(usize, 0), empty.len);
    const exact = try encodePiece(std.testing.allocator, &vocab, "ab", .{ .piece_bytes = 2, .output_tokens = 1, .work_bytes = 3 });
    defer std.testing.allocator.free(exact);
    try std.testing.expectEqualSlices(u32, &.{vocab.tokenId("ab").?}, exact);
    try std.testing.expectError(error.LimitExceeded, encodePiece(std.testing.allocator, &vocab, "ab", .{ .piece_bytes = 1 }));
    try std.testing.expectError(error.LimitExceeded, encodePiece(std.testing.allocator, &vocab, "ac", .{ .output_tokens = 1 }));
    try std.testing.expectError(error.WorkLimitExceeded, encodePiece(std.testing.allocator, &vocab, "ab", .{ .work_bytes = 2 }));
    _ = vocab.token_ids.remove("ab");
    try std.testing.expectError(error.MissingToken, encodePiece(std.testing.allocator, &vocab, "ab", .{}));
}

test "decoding validates IDs, byte alphabet, kinds, and limits" {
    var vocab = try fixture(&.{}, &.{ "<end>", "λ", "\xff" });
    defer vocab.deinit();
    @constCast(vocab.tokens)[256].kind = .control;
    const shown = try decode(std.testing.allocator, &vocab, &.{ 'a', 256 }, true, .{ .input_tokens = 2, .output_bytes = 6 });
    defer std.testing.allocator.free(shown);
    try std.testing.expectEqualStrings("a<end>", shown);
    const hidden = try decode(std.testing.allocator, &vocab, &.{ 'a', 256 }, false, .{ .output_bytes = 1 });
    defer std.testing.allocator.free(hidden);
    try std.testing.expectEqualStrings("a", hidden);
    try std.testing.expectError(error.LimitExceeded, decode(std.testing.allocator, &vocab, &.{ 'a', 256 }, true, .{ .output_bytes = 5 }));
    try std.testing.expectError(error.LimitExceeded, decode(std.testing.allocator, &vocab, &.{ 'a', 'b' }, true, .{ .input_tokens = 1 }));
    try std.testing.expectError(error.InvalidTokenId, decode(std.testing.allocator, &vocab, &.{259}, true, .{}));
    for ([_]u32{ 257, 258 }) |id| try std.testing.expectError(error.InvalidByteEncoding, decode(std.testing.allocator, &vocab, &.{id}, true, .{}));
    @constCast(vocab.tokens)['a'].kind = .unused;
    try std.testing.expectError(error.UnsupportedTokenType, decode(std.testing.allocator, &vocab, &.{'a'}, true, .{}));
    try std.testing.expectError(error.UnsupportedTokenType, encodePiece(std.testing.allocator, &vocab, "a", .{}));
}

fn allocationCheck(alloc: std.mem.Allocator, vocab: *const Vocabulary) !void {
    const ids = try encodePiece(alloc, vocab, " ab", .{});
    defer alloc.free(ids);
    const raw = try decode(alloc, vocab, ids, true, .{});
    defer alloc.free(raw);
    try std.testing.expectEqualStrings(" ab", raw);
}

test "BPE and decoding release partial allocations" {
    var vocab = try fixture(&.{"a b"}, &.{});
    defer vocab.deinit();
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationCheck, .{&vocab});
}
