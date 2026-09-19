//! Native text encoding, composing special-token partitioning, the splitter
//! the vocabulary's `pre` label selects, and BPE. The immutable Vocabulary
//! must outlive Encoder. No model-layer schedule, conversation format, or
//! implicit BOS/EOS policy.
const std = @import("std");
const vocabulary = @import("vocabulary.zig");
const bpe = @import("bpe.zig");
pub const pre = @import("pre.zig");
pub const gpt4o = @import("gpt4o.zig");
pub const Limits = struct {
    input_bytes: usize = 1024 * 1024,
    output_tokens: usize = 1024 * 1024,
    /// Conservative byte-comparison budget for special-token partitioning.
    special_work: usize = 1024 * 1024 * 1024,
    bpe_work: usize = 64 * 1024 * 1024,
    piece: bpe.Limits = .{},
};
pub const Error = bpe.Error || error{ UnsupportedPreTokenizer, InvalidUtf8 };

/// The splitters, keyed by the `pre` labels the encoder accepts: byte-level
/// `qwen35` and `llama4` (gpt4o.zig), and Gemma's SPM-style `gemma4`.
pub const Splitter = enum { qwen35, gpt4o, spm };

pub const Encoder = struct {
    vocab: *const vocabulary.Vocabulary,
    special_ids: []u32,
    allocator: std.mem.Allocator,
    splitter: Splitter,

    /// Owns only the special-ID index. At most 4,096 markers, each at most 256
    /// bytes, are accepted. These are operational bounds, not model semantics.
    pub fn init(alloc: std.mem.Allocator, vocab: *const vocabulary.Vocabulary) Error!Encoder {
        const splitter: Splitter = if (std.mem.eql(u8, vocab.pre, "qwen35"))
            .qwen35
        else if (std.mem.eql(u8, vocab.pre, "llama4"))
            .gpt4o
        else if (std.mem.eql(u8, vocab.pre, "gemma4"))
            .spm
        else
            return error.UnsupportedPreTokenizer;
        var ids: std.ArrayList(u32) = .empty;
        errdefer ids.deinit(alloc);
        for (vocab.tokens, 0..) |token, id| {
            switch (token.kind) {
                .control, .user_defined, .unknown => {
                    if (ids.items.len == 4096 or token.text.len == 0 or token.text.len > 256) return error.LimitExceeded;
                    try ids.append(alloc, @intCast(id));
                },
                else => {},
            }
        }
        std.mem.sort(u32, ids.items, vocab, struct {
            fn less(v: *const vocabulary.Vocabulary, a: u32, b: u32) bool {
                const x = v.tokens[a].text.len;
                const y = v.tokens[b].text.len;
                return if (x != y) x > y else a < b;
            }
        }.less);
        return .{ .vocab = vocab, .special_ids = try ids.toOwnedSlice(alloc), .allocator = alloc, .splitter = splitter };
    }

    /// Byte-level pieces: `Splitter` is a pre.zig-style iterator over the fragment.
    fn encodePieces(self: *const Encoder, comptime Iterator: type, alloc: std.mem.Allocator, fragment: []const u8, output: *std.ArrayList(u32), limits: Limits, work: *usize) Error!void {
        var it = try Iterator.init(fragment);
        while (it.next()) |piece| {
            var piece_limits = limits.piece;
            piece_limits.output_tokens = @min(piece_limits.output_tokens, limits.output_tokens - output.items.len);
            const ids = try bpe.encodePieceBudget(alloc, self.vocab, piece, piece_limits, work);
            defer alloc.free(ids);
            try output.appendSlice(alloc, ids);
        }
    }

    /// Gemma 4's pre-tokenizer (`[^\n]+|[\n]+`; docs/reference/gemma4.md
    /// § Tokenizer): a run of newlines is a piece of its own, emitted whole
    /// when the run is a token; everything else is one piece with spaces
    /// escaped to U+2581 before the merges. No other splitting.
    fn encodeSpm(self: *const Encoder, alloc: std.mem.Allocator, fragment: []const u8, output: *std.ArrayList(u32), limits: Limits, work: *usize) Error!void {
        var pos: usize = 0;
        while (pos < fragment.len) {
            const start = pos;
            const newline = fragment[pos] == '\n';
            while (pos < fragment.len and (fragment[pos] == '\n') == newline) : (pos += 1) {}
            const piece = fragment[start..pos];
            if (newline) if (self.vocab.tokenId(piece)) |id| {
                if (output.items.len == limits.output_tokens) return error.LimitExceeded;
                try output.append(alloc, id);
                continue;
            };
            var piece_limits = limits.piece;
            piece_limits.output_tokens = @min(piece_limits.output_tokens, limits.output_tokens - output.items.len);
            const ids = if (newline) try bpe.encodeSpmBudget(alloc, self.vocab, piece, piece_limits, work) else blk: {
                var spaces: usize = 0;
                for (piece) |byte| spaces += @intFromBool(byte == ' ');
                if (spaces > (piece_limits.piece_bytes -| piece.len) / 2) return error.LimitExceeded;
                const escaped = try alloc.alloc(u8, piece.len + spaces * 2);
                defer alloc.free(escaped);
                var n: usize = 0;
                for (piece) |byte| if (byte == ' ') {
                    @memcpy(escaped[n..][0..3], "\xE2\x96\x81");
                    n += 3;
                } else {
                    escaped[n] = byte;
                    n += 1;
                };
                break :blk try bpe.encodeSpmBudget(alloc, self.vocab, escaped, piece_limits, work);
            };
            defer alloc.free(ids);
            try output.appendSlice(alloc, ids);
        }
    }

    pub fn deinit(self: *Encoder) void {
        self.allocator.free(self.special_ids);
        self.* = undefined;
    }

    /// Returns owned IDs, freed with alloc. `parse_special=false` treats control
    /// and unknown spellings as ordinary text; user-defined tokens always match,
    /// following the pinned reference. Never automatically adds BOS/EOS.
    pub fn encode(self: *const Encoder, alloc: std.mem.Allocator, text: []const u8, parse_special: bool, limits: Limits) Error![]u32 {
        if (text.len > limits.input_bytes) return error.LimitExceeded;
        if (!std.unicode.utf8ValidateSlice(text)) return error.InvalidUtf8;
        const marks = try alloc.alloc(?u32, text.len);
        defer alloc.free(marks);
        @memset(marks, null);
        var special_work = limits.special_work;
        // Longest marker types partition all still-raw fragments first, matching
        // reference precedence even when a shorter marker starts earlier. Each
        // marker's pass is a vectorized search for its first byte (charged per
        // 16-byte block) plus one comparison per candidate: a vocabulary with
        // thousands of markers (Muse Glimmer's 2,048 reserved tokens) would
        // otherwise spend its whole budget on a few kilobytes of text.
        for (self.special_ids) |id| {
            const token = self.vocab.tokens[id];
            if (!parse_special and token.kind != .user_defined) continue;
            const size = token.text.len;
            var pos: usize = 0;
            while (std.mem.indexOfScalarPos(u8, text, pos, token.text[0])) |at| {
                try spend(&special_work, (at - pos + 15) / 16);
                pos = at + 1;
                if (size > text.len - at) break;
                try spend(&special_work, size);
                if (!std.mem.eql(u8, text[at..][0..size], token.text)) continue;
                if (try self.covered(marks, at, size, &special_work)) continue;
                marks[at] = id;
                pos = at + size;
            }
            try spend(&special_work, (text.len -| pos + 15) / 16);
        }
        var output: std.ArrayList(u32) = .empty;
        errdefer output.deinit(alloc);
        var work = limits.bpe_work;
        var pos: usize = 0;
        while (pos < text.len) {
            if (marks[pos]) |id| {
                if (output.items.len == limits.output_tokens) return error.LimitExceeded;
                try output.append(alloc, id);
                pos += self.vocab.tokens[id].text.len;
            } else {
                const start = pos;
                while (pos < text.len and marks[pos] == null) : (pos += 1) {}
                switch (self.splitter) {
                    .spm => try self.encodeSpm(alloc, text[start..pos], &output, limits, &work),
                    .qwen35 => try self.encodePieces(pre.Iterator, alloc, text[start..pos], &output, limits, &work),
                    .gpt4o => try self.encodePieces(gpt4o.Iterator, alloc, text[start..pos], &output, limits, &work),
                }
            }
        }
        return output.toOwnedSlice(alloc);
    }

    /// Whether a marker placed earlier (a longer type, or this type further
    /// left) starts inside `[at, at + size)` or extends into it from before.
    fn covered(self: *const Encoder, marks: []const ?u32, at: usize, size: usize, work: *usize) Error!bool {
        try spend(work, size);
        for (marks[at..][0..size]) |mark| if (mark != null) return true;
        // Markers are at most 256 bytes, so only that many earlier starts matter.
        const back = @min(at, 256);
        try spend(work, back);
        for (1..back + 1) |k| if (marks[at - k]) |earlier| {
            if (self.vocab.tokens[earlier].text.len > k) return true;
        };
        return false;
    }
};

fn spend(work: *usize, amount: usize) Error!void {
    if (amount > work.*) return error.WorkLimitExceeded;
    work.* -= amount;
}

test {
    _ = pre;
    _ = gpt4o;
}

fn fixture() !vocabulary.Vocabulary {
    const tokens = &[_]vocabulary.Token{
        .{ .text = "a", .kind = .normal },    .{ .text = "b", .kind = .normal },
        .{ .text = "ab", .kind = .normal },   .{ .text = "1", .kind = .normal },
        .{ .text = "2", .kind = .normal },    .{ .text = "12", .kind = .normal },
        .{ .text = "<", .kind = .normal },    .{ .text = ">", .kind = .normal },
        .{ .text = "x", .kind = .normal },    .{ .text = "u", .kind = .normal },
        .{ .text = "<x>", .kind = .control }, .{ .text = "<u>", .kind = .user_defined },
        // The longer marker starts later: longest-type precedence must win.
        .{ .text = "ax", .kind = .control },  .{ .text = "xu>", .kind = .control },
    };
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    errdefer arena.deinit();
    const alloc = arena.allocator();
    var ids: std.StringHashMapUnmanaged(u32) = .empty;
    var ranks: std.StringHashMapUnmanaged(u32) = .empty;
    for (tokens, 0..) |token, id| try ids.put(alloc, token.text, @intCast(id));
    try ranks.put(alloc, "a b", 0);
    try ranks.put(alloc, "1 2", 1);
    return .{ .storage = arena, .tokens = tokens, .pre = "qwen35", .bos = null, .eos = null, .padding = null, .token_ids = ids, .merge_ranks = ranks };
}

test "full encoding partitions special markers before Unicode and BPE" {
    var vocab = try fixture();
    defer vocab.deinit();
    var encoder = try Encoder.init(std.testing.allocator, &vocab);
    defer encoder.deinit();
    const cases = .{
        .{ "a<x>b", true, &[_]u32{ 0, 10, 1 } },
        .{ "a<x>b", false, &[_]u32{ 0, 6, 8, 7, 1 } },
        .{ "<u>", false, &[_]u32{11} },
        .{ "axu>", true, &[_]u32{ 0, 13 } },
        .{ "<x><x>", true, &[_]u32{ 10, 10 } },
        .{ "ab12ab", false, &[_]u32{ 2, 3, 4, 2 } },
    };
    inline for (cases) |case| {
        const ids = try encoder.encode(std.testing.allocator, case[0], case[1], .{});
        defer std.testing.allocator.free(ids);
        try std.testing.expectEqualSlices(u32, case[2], ids);
    }
}

test "full encoding bounds input, output, and aggregate work across pieces" {
    var vocab = try fixture();
    defer vocab.deinit();
    var encoder = try Encoder.init(std.testing.allocator, &vocab);
    defer encoder.deinit();
    const ids = try encoder.encode(std.testing.allocator, "ab1ab", true, .{ .input_bytes = 5, .output_tokens = 3, .bpe_work = 6 });
    defer std.testing.allocator.free(ids);
    try std.testing.expectEqualSlices(u32, &.{ 2, 3, 2 }, ids);
    for ([_]Limits{ .{ .input_bytes = 4 }, .{ .output_tokens = 2 }, .{ .piece = .{ .piece_bytes = 1 } } }) |limits|
        try std.testing.expectError(error.LimitExceeded, encoder.encode(std.testing.allocator, "ab1ab", true, limits));
    try std.testing.expectError(error.WorkLimitExceeded, encoder.encode(std.testing.allocator, "ab1ab", true, .{ .bpe_work = 5 }));
    try std.testing.expectError(error.WorkLimitExceeded, encoder.encode(std.testing.allocator, "a<x>", true, .{ .special_work = 0 }));
    try std.testing.expectError(error.InvalidUtf8, encoder.encode(std.testing.allocator, "\xff", true, .{}));
    const empty = try encoder.encode(std.testing.allocator, "", true, .{ .input_bytes = 0, .output_tokens = 0, .special_work = 0, .bpe_work = 0 });
    defer std.testing.allocator.free(empty);
    try std.testing.expectEqual(@as(usize, 0), empty.len);
    vocab.pre = "gpt2";
    try std.testing.expectError(error.UnsupportedPreTokenizer, Encoder.init(std.testing.allocator, &vocab));
}

test "a marker starting inside an earlier placement is skipped, and thousands of markers stay within budget" {
    var vocab = try fixture();
    defer vocab.deinit();
    var encoder = try Encoder.init(std.testing.allocator, &vocab);
    defer encoder.deinit();
    // `xu>` (longest) takes bytes 1..4 of `axu>`; `ax` would start at 0 and
    // reach into it, and `<u>`-like candidates inside it are covered too.
    const ids = try encoder.encode(std.testing.allocator, "axu>axu>", true, .{});
    defer std.testing.allocator.free(ids);
    try std.testing.expectEqualSlices(u32, &.{ 0, 13, 0, 13 }, ids);
    // 2,048 reserved markers over 64 KiB of text: the per-marker pass is a
    // first-byte search, so the default budget holds with room to spare.
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tokens: std.ArrayList(vocabulary.Token) = .empty;
    try tokens.appendSlice(a, &.{ .{ .text = "a", .kind = .normal }, .{ .text = "<", .kind = .normal }, .{ .text = "|", .kind = .normal }, .{ .text = ">", .kind = .normal }, .{ .text = "r", .kind = .normal } });
    for (0..2048) |i| try tokens.append(a, .{ .text = try std.fmt.allocPrint(a, "<|reserved_special_token_{d}|>", .{i}), .kind = .control });
    var token_ids: std.StringHashMapUnmanaged(u32) = .empty;
    for (tokens.items, 0..) |token, id| try token_ids.put(a, token.text, @intCast(id));
    var many: vocabulary.Vocabulary = .{ .storage = arena, .tokens = tokens.items, .pre = "llama4", .bos = null, .eos = null, .padding = null, .token_ids = token_ids, .merge_ranks = .empty };
    var wide = try Encoder.init(std.testing.allocator, &many);
    defer wide.deinit();
    const text = try std.testing.allocator.alloc(u8, 64 * 1024);
    defer std.testing.allocator.free(text);
    @memset(text, 'a');
    @memcpy(text[100..][0.."<|reserved_special_token_7|>".len], "<|reserved_special_token_7|>");
    const out = try wide.encode(std.testing.allocator, text, true, .{ .special_work = 64 * 1024 * 1024 });
    defer std.testing.allocator.free(out);
    try std.testing.expectEqual(@as(usize, 64 * 1024 - "<|reserved_special_token_7|>".len + 1), out.len);
    try std.testing.expectEqual(@as(u32, 5 + 7), out[100]);
}

test "the pre label selects the splitter: llama4 keeps digit runs together" {
    var vocab = try fixture();
    defer vocab.deinit();
    vocab.pre = "llama4";
    var encoder = try Encoder.init(std.testing.allocator, &vocab);
    defer encoder.deinit();
    const ids = try encoder.encode(std.testing.allocator, "ab12ab<x>", true, .{});
    defer std.testing.allocator.free(ids);
    try std.testing.expectEqualSlices(u32, &.{ 2, 5, 2, 10 }, ids);
}

fn allocationCheck(alloc: std.mem.Allocator, vocab: *const vocabulary.Vocabulary) !void {
    var encoder = try Encoder.init(alloc, vocab);
    defer encoder.deinit();
    const ids = try encoder.encode(alloc, "ab<x>12<u>", true, .{});
    defer alloc.free(ids);
    try std.testing.expectEqualSlices(u32, &.{ 2, 10, 3, 4, 11 }, ids);
}

test "encoder initialization and encoding clean up all allocation failures" {
    var vocab = try fixture();
    defer vocab.deinit();
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationCheck, .{&vocab});
}
