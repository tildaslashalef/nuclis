//! A decode-only piece table from a Hugging Face `tokenizer.json` whose
//! decoder is `Metaspace`: ids to text, never text to ids, so the model's
//! normalizer and merges are not read (a speech recognizer's output side;
//! hf_json.zig is the encoder and refuses normalizers such as `Precompiled`).
//! Decoding follows the library's `Metaspace` decoder: special added tokens
//! are skipped, U+2581 becomes a space, and the first piece loses one
//! leading space unless the scheme is `never`. docs/engine/tokenizer.md.
const std = @import("std");

pub const Limits = struct {
    tokens: u32 = 1_000_000,
    /// Total copied piece bytes.
    text_bytes: usize = 64 * 1024 * 1024,
};

pub const Error = std.mem.Allocator.Error || error{
    InvalidTokenizerJson,
    UnsupportedModel,
    UnsupportedDecoder,
    InvalidToken,
    DuplicateToken,
    LimitExceeded,
};

pub const Table = struct {
    arena: std.heap.ArenaAllocator,
    /// By id; null where no piece has the id. Arena-owned.
    pieces: []const ?[]const u8,
    /// By id: an added token marked `special`.
    special: []const bool,
    /// The decoder's replacement, usually U+2581.
    replacement: []const u8,
    /// The first decoded piece loses one leading replacement.
    strip_first: bool,

    pub fn deinit(self: *Table) void {
        self.arena.deinit();
        self.* = undefined;
    }

    pub fn count(self: *const Table) usize {
        return self.pieces.len;
    }

    pub fn piece(self: *const Table, id: u32) ?[]const u8 {
        return if (id < self.pieces.len) self.pieces[id] else null;
    }

    pub fn isSpecial(self: *const Table, id: u32) bool {
        return id < self.special.len and self.special[id];
    }

    /// Writes the text of `ids`: special and unknown ids skipped.
    pub fn decode(self: *const Table, ids: []const u32, out: *std.Io.Writer) std.Io.Writer.Error!void {
        var started = false;
        for (ids) |id| try self.decodeNext(id, &started, out);
    }

    /// One id of a decoding in progress; `started` (false at the start)
    /// records whether a piece has been written yet.
    pub fn decodeNext(self: *const Table, id: u32, started: *bool, out: *std.Io.Writer) std.Io.Writer.Error!void {
        if (self.isSpecial(id)) return;
        var text = self.piece(id) orelse return;
        if (!started.* and self.strip_first and std.mem.startsWith(u8, text, self.replacement)) text = text[self.replacement.len..];
        started.* = true;
        try self.writePiece(text, out);
    }

    /// One piece's text with the replacement as spaces, nothing stripped.
    pub fn writePiece(self: *const Table, text: []const u8, out: *std.Io.Writer) std.Io.Writer.Error!void {
        var rest = text;
        while (std.mem.indexOf(u8, rest, self.replacement)) |at| {
            try out.writeAll(rest[0..at]);
            try out.writeByte(' ');
            rest = rest[at + self.replacement.len ..];
        }
        try out.writeAll(rest);
    }
};

const AddedJson = struct { id: u32, content: []const u8, special: bool = false };

const File = struct {
    added_tokens: []const AddedJson = &.{},
    decoder: ?std.json.Value = null,
    model: struct {
        type: []const u8,
        vocab: std.json.ArrayHashMap(u32),
    },
};

/// Parses a whole `tokenizer.json` image: a BPE model's vocabulary and the
/// added tokens, under a `Metaspace` decoder. `bytes` may be freed after.
pub fn parse(gpa: std.mem.Allocator, bytes: []const u8, limits: Limits) Error!Table {
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
    if (!std.mem.eql(u8, file.model.type, "BPE")) return error.UnsupportedModel;
    const decoder = file.decoder orelse return error.UnsupportedDecoder;
    if (decoder != .object) return error.UnsupportedDecoder;
    const kind = decoder.object.get("type") orelse return error.UnsupportedDecoder;
    const replacement = decoder.object.get("replacement") orelse return error.UnsupportedDecoder;
    if (kind != .string or !std.mem.eql(u8, kind.string, "Metaspace") or replacement != .string or replacement.string.len == 0)
        return error.UnsupportedDecoder;
    const scheme: []const u8 = if (decoder.object.get("prepend_scheme")) |s| (if (s == .string) s.string else return error.UnsupportedDecoder) else "always";
    if (!std.mem.eql(u8, scheme, "always") and !std.mem.eql(u8, scheme, "first") and !std.mem.eql(u8, scheme, "never"))
        return error.UnsupportedDecoder;

    var total: usize = 0;
    for (file.model.vocab.map.values()) |id| total = @max(total, @as(usize, id) + 1);
    for (file.added_tokens) |a| total = @max(total, @as(usize, a.id) + 1);
    if (total == 0 or total > limits.tokens) return error.LimitExceeded;

    var table: Table = .{ .arena = .init(gpa), .pieces = &.{}, .special = &.{}, .replacement = &.{}, .strip_first = !std.mem.eql(u8, scheme, "never") };
    errdefer table.arena.deinit();
    const alloc = table.arena.allocator();
    var remaining = limits.text_bytes;
    const pieces = try alloc.alloc(?[]const u8, total);
    @memset(pieces, null);
    const special = try alloc.alloc(bool, total);
    @memset(special, false);
    var it = file.model.vocab.map.iterator();
    while (it.next()) |entry| try put(alloc, pieces, entry.value_ptr.*, entry.key_ptr.*, &remaining);
    for (file.added_tokens) |a| {
        // An added token may repeat a vocabulary piece under its own id only.
        if (pieces[a.id]) |existing| {
            if (!std.mem.eql(u8, existing, a.content)) return error.DuplicateToken;
        } else try put(alloc, pieces, a.id, a.content, &remaining);
        special[a.id] = a.special;
    }
    table.pieces = pieces;
    table.special = special;
    table.replacement = try alloc.dupe(u8, replacement.string);
    return table;
}

fn put(alloc: std.mem.Allocator, pieces: []?[]const u8, id: u32, text: []const u8, remaining: *usize) Error!void {
    if (pieces[id] != null) return error.DuplicateToken;
    if (text.len == 0 or !std.unicode.utf8ValidateSlice(text)) return error.InvalidToken;
    if (text.len > remaining.*) return error.LimitExceeded;
    remaining.* -= text.len;
    pieces[id] = try alloc.dupe(u8, text);
}

const sample =
    \\{"added_tokens":[{"id":0,"content":"<unk>","special":true},{"id":1,"content":"<de-DE>","special":true},
    \\ {"id":6,"content":"<pad>","special":true}],
    \\ "normalizer":{"type":"Precompiled","precompiled_charsmap":"AA"},
    \\ "decoder":{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true},
    \\ "model":{"type":"BPE","vocab":{"<unk>":0,"<de-DE>":1,"▁":2,"▁Hal":3,"lo":4,".":5},"merges":[]}}
;

test "Metaspace decoding: specials skipped, U+2581 a space, the first one dropped" {
    const gpa = std.testing.allocator;
    var table = try parse(gpa, sample, .{});
    defer table.deinit();
    try std.testing.expectEqual(@as(usize, 7), table.count());
    try std.testing.expect(table.isSpecial(1) and table.isSpecial(6) and !table.isSpecial(3));
    try std.testing.expectEqualStrings("<de-DE>", table.piece(1).?);
    var buffer: [64]u8 = undefined;
    var out: std.Io.Writer = .fixed(&buffer);
    try table.decode(&.{ 0, 3, 4, 3, 4, 5, 1, 6, 99 }, &out);
    try std.testing.expectEqualStrings("Hallo Hallo.", out.buffered());
    // The first piece decoded, not the first id, loses its space.
    out = .fixed(&buffer);
    try table.decode(&.{ 1, 2, 3, 4 }, &out);
    try std.testing.expectEqualStrings(" Hallo", out.buffered());
}

test "other models and decoders, and clashing pieces, are refused" {
    const gpa = std.testing.allocator;
    const cases = [_]struct { []const u8, []const u8, anyerror }{
        .{ "\"type\":\"BPE\"", "\"type\":\"Unigram\"", error.UnsupportedModel },
        .{ "\"type\":\"Metaspace\"", "\"type\":\"ByteLevel\"", error.UnsupportedDecoder },
        .{ "\"prepend_scheme\":\"always\"", "\"prepend_scheme\":\"sometimes\"", error.UnsupportedDecoder },
        .{ "\"content\":\"<de-DE>\"", "\"content\":\"<fr-FR>\"", error.DuplicateToken },
        .{ "\"lo\":4", "\"lo\":3", error.DuplicateToken },
        .{ "\"model\"", "\"modell\"", error.InvalidTokenizerJson },
    };
    for (cases) |case| {
        const bad = try std.mem.replaceOwned(u8, gpa, sample, case[0], case[1]);
        defer gpa.free(bad);
        try std.testing.expectError(case[2], parse(gpa, bad, .{}));
    }
    try std.testing.expectError(error.LimitExceeded, parse(gpa, sample, .{ .tokens = 4 }));
}
