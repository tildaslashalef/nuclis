//! Display width and fitting to a column, in grapheme clusters
//! (`tui/graphemes.zig`), so a cut never splits an emoji or a combining
//! mark. Strings from files, the Hub, or a model pass through `clean`
//! before they reach a terminal.
const std = @import("std");
const graphemes = @import("../tui/graphemes.zig");

pub const Align = enum { left, right };

/// Cells `s` occupies; `s` holds no escapes or newlines.
pub fn width(s: []const u8) usize {
    return graphemes.width(s);
}

/// Writes `s` into exactly `cols` cells: cut on a cluster boundary with
/// `ellipsis` when too wide, padded with spaces otherwise. Without `pad`,
/// no trailing spaces (the last column of a row).
pub fn writeFit(w: *std.Io.Writer, s: []const u8, cols: usize, alignment: Align, ellipsis: []const u8, pad: bool) std.Io.Writer.Error!void {
    const sw = width(s);
    if (sw <= cols) {
        const fill = cols - sw;
        if (alignment == .right) try w.splatByteAll(' ', fill);
        try w.writeAll(s);
        if (alignment == .left and pad) try w.splatByteAll(' ', fill);
        return;
    }
    const ew = width(ellipsis);
    if (cols <= ew) {
        _ = try writePrefix(w, s, cols);
        return;
    }
    const used = try writePrefix(w, s, cols - ew);
    try w.writeAll(ellipsis);
    if (pad) try w.splatByteAll(' ', cols - ew - used);
}

/// The longest prefix of at most `cols` cells; returns its width.
fn writePrefix(w: *std.Io.Writer, s: []const u8, cols: usize) std.Io.Writer.Error!usize {
    var used: usize = 0;
    var end: usize = 0;
    while (end < s.len) {
        const next = graphemes.clusterNext(s, end);
        const cw = graphemes.clusterWidth(s[end..next]);
        if (used + cw > cols) break;
        used += cw;
        end = next;
    }
    try w.writeAll(s[0..end]);
    return used;
}

/// `s` without terminal controls: C0 (a newline and a tab become a space),
/// DEL, C1 raw or UTF-8 encoded, and the bidirectional overrides become
/// `?`. Returns `s` itself when it is already clean; else allocated in
/// `arena`.
pub fn clean(arena: std.mem.Allocator, s: []const u8) std.mem.Allocator.Error![]const u8 {
    for (s) |c| {
        if (c < 0x20 or c >= 0x7f) break;
    } else return s;
    var out: std.ArrayList(u8) = try .initCapacity(arena, s.len);
    var i: usize = 0;
    while (i < s.len) {
        const len = std.unicode.utf8ByteSequenceLength(s[i]) catch 1;
        const cp: u21 = if (i + len <= s.len) std.unicode.utf8Decode(s[i .. i + len]) catch 0xfffd else 0xfffd;
        const step = if (cp == 0xfffd and i + len > s.len) 1 else len;
        if (cp == '\n' or cp == '\t' or cp == '\r') {
            try out.append(arena, ' ');
        } else if (cp < 0x20 or cp == 0x7f or (cp >= 0x80 and cp <= 0x9f) or (cp >= 0x202a and cp <= 0x202e) or (cp >= 0x2066 and cp <= 0x2069) or cp == 0xfffd) {
            try out.append(arena, '?');
        } else try out.appendSlice(arena, s[i .. i + step]);
        i += step;
    }
    return out.items;
}

const testing = std.testing;

test "writeFit pads, aligns, and cuts on a cluster boundary" {
    var buf: std.Io.Writer.Allocating = .init(testing.allocator);
    defer buf.deinit();
    const w = &buf.writer;
    try writeFit(w, "qwen", 6, .left, "…", true);
    try w.writeByte('|');
    try writeFit(w, "42", 5, .right, "…", true);
    try w.writeByte('|');
    try writeFit(w, "gemma-4-e4b-qat", 8, .left, "…", true);
    try w.writeByte('|');
    try writeFit(w, "日本語テキスト", 6, .left, "…", true);
    try w.writeByte('|');
    try writeFit(w, "abcdef", 4, .left, "...", false);
    try w.writeByte('|');
    try writeFit(w, "abc", 6, .left, "…", false);
    try w.writeByte('|');
    try testing.expectEqualStrings("qwen  |   42|gemma-4…|日本… |a...|abc|", buf.written());
}

test "clean replaces controls and keeps clean text as it is" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    try testing.expectEqualStrings("a?[31mred", try clean(a, "a\x1b[31mred"));
    try testing.expectEqualStrings("csi?2J", try clean(a, "csi\xc2\x9b2J"));
    try testing.expectEqualStrings("bidi?x", try clean(a, "bidi\u{202e}x"));
    try testing.expectEqualStrings("two lines", try clean(a, "two\nlines"));
    try testing.expectEqualStrings("Zürich ✓", try clean(a, "Zürich ✓"));
    const s: []const u8 = "no-alloc";
    try testing.expectEqual(s.ptr, (try clean(a, s)).ptr);
}
