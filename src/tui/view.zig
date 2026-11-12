//! Pure presentation: folding never mutates the conversation used by inference.
//! ANSI from model/user text is stripped before rendering into the terminal.
//! Width and line breaking work in grapheme clusters (`graphemes.zig`), so an
//! emoji ZWJ sequence or a combining mark is one unit, never split.
const std = @import("std");
const graphemes = @import("graphemes.zig");

/// Display width of a single line in terminal cells (no newlines expected).
pub fn width(text: []const u8) usize {
    return graphemes.width(text);
}

/// How a row that does not fit is broken.
pub const Wrap = enum {
    /// Exactly at the column. What a status bar or a hint wants: the result
    /// is truncated to its first row anyway, so every cell should be used.
    character,
    /// At the last space that fits, the way the editor wraps.
    /// A word longer than the row still breaks mid-word, because the
    /// alternative is a row that cannot be drawn. This is what the
    /// transcript uses, so a prompt and its echo break at the same places.
    word,
};

/// Where a word-wrapped row ends, given the offset just after the last space
/// on it: before that space, so a trailing blank is never drawn at the edge.
fn rowEnd(text: []const u8, break_at: ?usize, offset: usize) usize {
    const at = break_at orelse return offset;
    return if (at > 0 and text[at - 1] == ' ') at - 1 else at;
}

/// Creates explicit lines before screen placement. Rows are slices of `text`,
/// so only the array of slices is allocated.
pub fn lines(alloc: std.mem.Allocator, text: []const u8, columns: usize, wrap: Wrap) ![][]const u8 {
    var result: std.ArrayList([]const u8) = .empty;
    errdefer result.deinit(alloc);
    var start: usize = 0;
    var offset: usize = 0;
    var used: usize = 0;
    // Byte offset just after the last space seen on this row, and the width
    // of everything after it: where a word wrap breaks and how much of the
    // row the carried word takes on the next one.
    var break_at: ?usize = null;
    var since_break: usize = 0;
    while (offset < text.len) {
        const end = graphemes.clusterNext(text, offset);
        const cluster = text[offset..end];
        const w = graphemes.clusterWidth(cluster);
        if (cluster.len == 1 and cluster[0] == '\n') {
            try result.append(alloc, text[start..if (used > columns) offset - 1 else offset]);
            start = end;
            used = 0;
            break_at = null;
            since_break = 0;
            offset = end;
            continue;
        }
        // One space that lands past the edge is not drawn there and does not
        // break the row: it becomes the next break point, so a word ending
        // exactly at the last column keeps its row. A second breaks it, so a
        // run of spaces never hangs over the edge.
        const is_space = cluster.len == 1 and cluster[0] == ' ';
        // A cluster wider than the row starts it anyway: an empty row first
        // would never end.
        if (used + w > columns and used > 0 and !(wrap == .word and is_space and used <= columns)) {
            const wrapped = wrap == .word;
            try result.append(alloc, text[start..if (wrapped) rowEnd(text, break_at, offset) else offset]);
            start = if (wrapped) break_at orelse offset else offset;
            used = if (wrapped and break_at != null) since_break else 0;
            break_at = null;
            since_break = 0;
        }
        used += w;
        since_break += w;
        offset = end;
        if (is_space) {
            break_at = offset;
            since_break = 0;
        }
    }
    // A row can end in the one space it hid past the edge; it is not drawn.
    try result.append(alloc, text[start..if (used > columns) text.len - 1 else text.len]);
    return result.toOwnedSlice(alloc);
}
/// The longest prefix of `text` that fits in `columns` cells, cut on a cluster
/// boundary. Used for single-line bars that must not wrap.
pub fn fit(alloc: std.mem.Allocator, text: []const u8, columns: usize) ![]const u8 {
    const rows = try lines(alloc, text, columns, .character);
    defer alloc.free(rows);
    return rows[0];
}
/// Wraps pre-styled text (containing SGR escape sequences) to `columns`
/// display cells. Escape sequences are copied verbatim and occupy no width;
/// attributes persist across the introduced line breaks, so no resetting is
/// needed at boundaries. Returns owned lines.
pub fn wrapStyled(alloc: std.mem.Allocator, text: []const u8, columns: usize, wrap: Wrap) ![][]const u8 {
    var rows: std.ArrayList([]const u8) = .empty;
    errdefer {
        for (rows.items) |row| alloc.free(row);
        rows.deinit(alloc);
    }
    var line: std.ArrayList(u8) = .empty;
    defer line.deinit(alloc);
    var used: usize = 0;
    var offset: usize = 0;
    // The word-wrap bookkeeping, in bytes of the row being built rather than
    // of the source: `break_bytes` is the index in `line` just past the last
    // space, so a break splits the buffer there and carries the tail over.
    var break_bytes: ?usize = null;
    var since_break: usize = 0;
    // Where in `line` the one space hidden past the edge sits, if one does.
    var hidden: ?usize = null;
    while (offset < text.len) {
        if (escapeSpan(text, offset)) |end| {
            try line.appendSlice(alloc, text[offset..end]);
            offset = end;
            continue;
        }
        const end = graphemes.clusterNext(text, offset);
        const cluster = text[offset..end];
        const w = graphemes.clusterWidth(cluster);
        if (cluster.len == 1 and cluster[0] == '\n') {
            try rows.append(alloc, try dupeRow(alloc, line.items, hidden));
            line.clearRetainingCapacity();
            hidden = null;
            used = 0;
            break_bytes = null;
            since_break = 0;
            offset = end;
            continue;
        }
        const is_space = cluster.len == 1 and cluster[0] == ' ';
        if (used + w > columns and used > 0 and !(wrap == .word and is_space and used <= columns)) {
            if (wrap == .word and break_bytes != null) {
                // Cut before the space the row broke on and carry the rest of
                // the word — escapes included — to the next row.
                const at = break_bytes.?;
                const cut = if (at > 0 and line.items[at - 1] == ' ') at - 1 else at;
                try rows.append(alloc, try alloc.dupe(u8, line.items[0..cut]));
                const carried = line.items.len - at;
                std.mem.copyForwards(u8, line.items[0..carried], line.items[at..]);
                line.shrinkRetainingCapacity(carried);
                used = since_break;
            } else {
                try rows.append(alloc, try alloc.dupe(u8, line.items));
                line.clearRetainingCapacity();
                used = 0;
            }
            break_bytes = null;
            since_break = 0;
            hidden = null;
        }
        if (is_space and used + w > columns) hidden = line.items.len;
        try line.appendSlice(alloc, cluster);
        used += w;
        since_break += w;
        offset = end;
        if (is_space) {
            break_bytes = line.items.len;
            since_break = 0;
        }
    }
    try rows.append(alloc, try dupeRow(alloc, line.items, hidden));
    return rows.toOwnedSlice(alloc);
}

/// A styled row without the one space it hid past the edge at `hidden`,
/// escapes after it kept. Caller owns the copy.
fn dupeRow(alloc: std.mem.Allocator, row: []const u8, hidden: ?usize) ![]const u8 {
    const at = hidden orelse return alloc.dupe(u8, row);
    return std.mem.concat(alloc, u8, &.{ row[0..at], row[at + 1 ..] });
}

/// The end offset of an ANSI escape sequence that starts at `offset` (an ESC),
/// or null when no escape starts there. CSI runs to its final byte; OSC runs
/// to BEL or ST. Unterminated sequences consume the rest of the text so a cut
/// never drops bytes. Both occupy zero cells.
fn escapeSpan(text: []const u8, offset: usize) ?usize {
    if (text[offset] != 0x1b) return null;
    var end = offset + 1;
    if (end >= text.len) return end;
    switch (text[end]) {
        '[' => {
            end += 1;
            while (end < text.len and text[end] >= 0x20 and text[end] <= 0x3f) end += 1;
            if (end < text.len) end += 1;
        },
        ']' => {
            end += 1;
            while (end < text.len) {
                if (text[end] == 0x07) {
                    end += 1;
                    break;
                }
                if (text[end] == 0x1b and end + 1 < text.len and text[end + 1] == '\\') {
                    end += 2;
                    break;
                }
                end += 1;
            }
        },
        else => end += 1,
    }
    return end;
}

/// Display width of a pre-styled line: escape sequences occupy no cells.
pub fn styledWidth(text: []const u8) usize {
    var total: usize = 0;
    var offset: usize = 0;
    while (offset < text.len) {
        if (escapeSpan(text, offset)) |end| {
            offset = end;
            continue;
        }
        const end = graphemes.clusterNext(text, offset);
        total += graphemes.clusterWidth(text[offset..end]);
        offset = end;
    }
    return total;
}

/// Writes `text` without control characters: C0 but newline, DEL, and the
/// C1 controls (`U+0080`–`U+009F`, which a terminal may read as CSI or ST).
/// A tab is dropped here; text that keeps its tabs goes through
/// `expandTabs` before it is wrapped.
pub fn safe(out: *std.Io.Writer, text: []const u8) !void {
    var i: usize = 0;
    while (i < text.len) : (i += 1) {
        const b = text[i];
        if (isC1(text[i..])) {
            i += 1;
        } else if ((b >= 32 and b != 127) or b == '\n') try out.writeByte(b);
    }
}

/// Whether `text` starts with a UTF-8 encoded C1 control.
pub fn isC1(text: []const u8) bool {
    return text.len >= 2 and text[0] == 0xc2 and text[1] >= 0x80 and text[1] <= 0x9f;
}

/// Tab stops are every `tab_width` cells.
pub const tab_width = 4;

/// `text` with each tab replaced by spaces to the next tab stop of its line,
/// counted in cells; `text` itself when it holds no tab. Otherwise allocated
/// in `a`, which should be an arena: the caller cannot tell which it got.
pub fn expandTabs(a: std.mem.Allocator, text: []const u8) ![]const u8 {
    if (std.mem.indexOfScalar(u8, text, '\t') == null) return text;
    var out: std.ArrayList(u8) = .empty;
    var line_start: usize = 0;
    for (text) |b| {
        if (b == '\t') {
            const column = width(out.items[line_start..]);
            try out.appendNTimes(a, ' ', tab_width - column % tab_width);
            continue;
        }
        try out.append(a, b);
        if (b == '\n') line_start = out.items.len;
    }
    return out.items;
}
test "a cluster wider than the row takes a row of its own, with no empty one" {
    for ([_]Wrap{ .character, .word }) |mode| {
        const rows = try lines(std.testing.allocator, "中a", 1, mode);
        defer std.testing.allocator.free(rows);
        try std.testing.expectEqual(@as(usize, 2), rows.len);
        try std.testing.expectEqualStrings("中", rows[0]);
        try std.testing.expectEqualStrings("a", rows[1]);
    }
}

test "C1 controls are dropped and tabs reach the next stop" {
    var buffer: [64]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buffer);
    try safe(&writer, "a\u{9b}31mb\u{9c}c\u{a0}d");
    try std.testing.expectEqualStrings("a31mbc\u{a0}d", writer.buffered());
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expectEqualStrings("a   b\n    c\nab中    d", try expandTabs(arena.allocator(), "a\tb\n\tc\nab中\td"));
}

test "presentation strips escapes and wraps text" {
    var buffer: [100]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buffer);
    try safe(&writer, "hello\x1b[2J\rworld");
    try std.testing.expectEqualStrings("hello[2Jworld", writer.buffered());
    const wrapped = try lines(std.testing.allocator, "ab中c\nd", 4, .character);
    defer std.testing.allocator.free(wrapped);
    try std.testing.expectEqual(@as(usize, 3), wrapped.len);
    try std.testing.expectEqualStrings("ab中", wrapped[0]);
    try std.testing.expectEqualStrings("a │", try fit(std.testing.allocator, "a │ b", 3));
    try std.testing.expectEqualStrings("a │ b", try fit(std.testing.allocator, "a │ b", 40));
    try std.testing.expectEqual(@as(usize, 4), width("ab中"));
    try std.testing.expectEqual(@as(usize, 1), width("e\u{301}"));
    const styled = try wrapStyled(std.testing.allocator, "\x1b[1mab中cd\x1b[0m", 4, .character);
    defer std.testing.allocator.free(styled);
    defer for (styled) |s| std.testing.allocator.free(s);
    try std.testing.expectEqual(@as(usize, 2), styled.len);
    try std.testing.expectEqualStrings("\x1b[1mab中", styled[0]);
    try std.testing.expectEqualStrings("cd\x1b[0m", styled[1]);
}

test "word wrapping breaks at the last space that fits, in plain and styled text" {
    const a = std.testing.allocator;
    // The rule the editor already used; the transcript joins it
    // here, so a prompt and the answer under it break the same way.
    const rows = try lines(a, "the quick brown fox jumps", 12, .word);
    defer a.free(rows);
    try std.testing.expectEqual(@as(usize, 3), rows.len);
    try std.testing.expectEqualStrings("the quick", rows[0]);
    try std.testing.expectEqualStrings("brown fox", rows[1]);
    try std.testing.expectEqualStrings("jumps", rows[2]);
    // A word longer than the row still breaks: a row that cannot be drawn is
    // not an option.
    const long = try lines(a, "supercalifragilistic", 8, .word);
    defer a.free(long);
    try std.testing.expectEqual(@as(usize, 3), long.len);
    try std.testing.expectEqualStrings("supercal", long[0]);
    // Explicit newlines still end a row without a wrap, and an empty line survives.
    const hard = try lines(a, "a\n\nbb cc", 8, .word);
    defer a.free(hard);
    try std.testing.expectEqual(@as(usize, 3), hard.len);
    try std.testing.expectEqualStrings("", hard[1]);
    try std.testing.expectEqualStrings("bb cc", hard[2]);

    // The styled form keeps the escapes with the text they opened, wherever
    // the break lands: the bold span survives the carried word.
    const styled = try wrapStyled(a, "keep \x1b[1mtogether\x1b[0m now", 10, .word);
    defer {
        for (styled) |row| a.free(row);
        a.free(styled);
    }
    try std.testing.expectEqual(@as(usize, 3), styled.len);
    try std.testing.expectEqualStrings("keep", styled[0]);
    try std.testing.expectEqualStrings("\x1b[1mtogether\x1b[0m", styled[1]);
    try std.testing.expectEqualStrings("now", styled[2]);
    try std.testing.expectEqual(@as(usize, 4), styledWidth(styled[0]));

    // Character wrapping is still available where every cell counts.
    const tight = try wrapStyled(a, "keep together", 10, .character);
    defer {
        for (tight) |row| a.free(row);
        a.free(tight);
    }
    try std.testing.expectEqualStrings("keep toget", tight[0]);
}

test "spaces past the edge break the row instead of hanging over it" {
    const a = std.testing.allocator;
    for ([_]struct { text: []const u8, columns: usize }{
        .{ .text = "m   x", .columns = 1 },
        .{ .text = "ab    \x1b[0m", .columns = 2 },
        .{ .text = "a    \nb", .columns = 3 },
        .{ .text = "ab ", .columns = 2 },
        .{ .text = "say ok" ++ &@as([40]u8, @splat(' ')) ++ "then   stop", .columns = 26 },
    }) |case| {
        errdefer std.debug.print("case: \"{s}\" at {d}\n", .{ case.text, case.columns });
        const rows = try wrapStyled(a, case.text, case.columns, .word);
        defer {
            for (rows) |row| a.free(row);
            a.free(rows);
        }
        for (rows) |row| try std.testing.expect(styledWidth(row) <= case.columns);
        const plain = try lines(a, case.text, case.columns, .word);
        defer a.free(plain);
        for (plain) |row| try std.testing.expect(styledWidth(row) <= case.columns);
    }
    // A word that ends at the last column keeps its row, the space after it hidden.
    const exact = try lines(a, "abc def", 3, .word);
    defer a.free(exact);
    try std.testing.expectEqual(@as(usize, 2), exact.len);
    try std.testing.expectEqualStrings("abc", exact[0]);
    try std.testing.expectEqualStrings("def", exact[1]);
}

test "an OSC sequence is zero width and survives a wrap" {
    const a = std.testing.allocator;
    const link = "\x1b]8;;http://x\x1b\\label\x1b]8;;\x1b\\";
    try std.testing.expectEqual(@as(usize, 5), styledWidth(link));
    const rows = try wrapStyled(a, link, 3, .character);
    defer {
        for (rows) |row| a.free(row);
        a.free(rows);
    }
    try std.testing.expectEqual(@as(usize, 2), rows.len);
    try std.testing.expect(std.mem.indexOf(u8, rows[0], "lab") != null);
    try std.testing.expect(std.mem.indexOf(u8, rows[1], "el\x1b]8;;\x1b\\") != null);
}
