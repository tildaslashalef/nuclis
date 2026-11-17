//! Small markdown renderer for assistant turns. Produces pre-styled,
//! pre-wrapped lines: SGR spans come from the theme, decorative characters
//! come from its glyph set (Unicode or ASCII), and layout goes through
//! `view.wrapStyled` (which does not count escape sequences as width).
//! Input is untrusted: control bytes are stripped before styling, so
//! model-supplied ANSI can never reach stdout, and unclosed markers degrade
//! to literal text.
//!
//! Supported: `#`..`######` headings, `**bold**`, `*italic*`, `***both***`,
//! `~~strike~~`, `` `code` ``, `[text](url)` links (an http(s) URL becomes an
//! OSC 8 hyperlink, anything else keeps the styled label), `>` quotes with
//! nesting, `-`/`*`/`+` unordered lists with nesting, `1.` ordered lists
//! from any number, `- [ ]`/`- [x]` task lists, fenced code blocks with a
//! language label, `---` rules, and pipe tables with per-column alignment.
//! A soft line break inside a paragraph stays a line break. No row is ever
//! wider than the width asked for, whatever the input.
//!
//! Streaming. A turn is rendered while it arrives. `split` finds the
//! prefix that can no longer change, `renderFrom` renders it, and the
//! `Carry` it leaves (an open fence and its highlighter state) lets the
//! next prefix start where that one stopped, so a long answer reaches the
//! scrollback line by line with the rows one render of the whole would
//! give. The rules are in docs/app/agent-surface.md § Surface.
//!
//! A code block carries **no box glyphs**: its rows are the code on the
//! block background, filled to the edge by erase-to-end rather than spaces,
//! and the language is a caption row above them, so selecting a block in the
//! terminal copies the code and nothing else, and a reflow wraps nothing.
const std = @import("std");
const repeat = @import("../text.zig").repeat;
const builtin = @import("builtin");
const theme_mod = @import("theme.zig");
const view = @import("view.zig");
const highlight = @import("highlight.zig");

/// A partial document: the part that can no longer change, and the rest.
pub const Split = struct { closed: []const u8, open: []const u8 };

/// What a rendered prefix leaves for the text after it: the fence it ended
/// inside, if any, with that fence's language and highlighter state. A
/// prefix `split` closed ends nowhere else that carries state.
pub const Carry = struct {
    fence: ?Fence = null,

    pub const Fence = struct { lang: highlight.Language, hl: highlight.State = .{} };
};

/// Splits a streaming turn at the last point after which nothing before it
/// can render differently. The rules are the renderer's own:
///
/// - a blank line, heading, rule, quote, or list item is stable at its
///   newline, and so is a paragraph line, unless an inline span opened in
///   the paragraph is still waiting for its closer (`spanOpen`);
/// - every line inside a fence is stable at its newline; the opener waits
///   for the first code line, so its caption goes out with it;
/// - table rows stay open until a line that is not a row ends the table,
///   because column widths need every row.
///
/// `carry` is what the text before `text` left (`renderFrom`). The last
/// line is never closed: the newline that would end it has not arrived.
pub fn split(text: []const u8, carry: Carry) Split {
    var in_fence = carry.fence != null;
    var closed: usize = 0;
    // Where the paragraph lines not yet closed begin.
    var para: ?usize = null;
    var offset: usize = 0;
    while (std.mem.indexOfScalarPos(u8, text, offset, '\n')) |newline| {
        const line = std.mem.trim(u8, text[offset..newline], " \t\r");
        const next = newline + 1;
        if (in_fence) {
            if (std.mem.startsWith(u8, line, "```")) in_fence = false;
            closed = next;
        } else if (std.mem.startsWith(u8, line, "```")) {
            in_fence = true;
            para = null;
        } else if (line.len == 0 or !isParagraph(line)) {
            closed = next;
            para = null;
        } else if (line[0] == '|') {
            para = null;
        } else {
            const start = para orelse offset;
            if (paragraphStable(text[start..newline])) {
                closed = next;
                para = null;
            } else para = start;
        }
        offset = next;
    }
    return .{ .closed = text[0..closed], .open = text[closed..] };
}

/// Whether the paragraph lines in `lines` (whole source lines, no trailing
/// newline) render the same whatever lines follow them. The body is built
/// as `render` builds it, each line trimmed and joined by line breaks; one
/// longer than the scratch buffer is held, which is the safe answer.
fn paragraphStable(lines: []const u8) bool {
    var buf: [4096]u8 = undefined;
    var len: usize = 0;
    var iter = std.mem.splitScalar(u8, lines, '\n');
    while (iter.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \t\r");
        if (len + line.len + 1 > buf.len) return false;
        if (len > 0) {
            buf[len] = '\n';
            len += 1;
        }
        @memcpy(buf[len..][0..line.len], line);
        len += line.len;
    }
    return !spanOpen(buf[0..len]);
}

/// Whether text appended after `text` could change how `inlineSpan` renders
/// it: a closer search ran off the end, or a marker sits last, where only
/// the next byte decides what it is. This is `inlineSpan`'s top level with
/// the output left out; a span it closes is fixed whatever follows, so only
/// the top level can stay open.
fn spanOpen(text: []const u8) bool {
    var brackets: Brackets = .{};
    var i: usize = 0;
    while (i < text.len) {
        const b = text[i];
        if (b == '\\') {
            if (i + 1 >= text.len) return true;
            i += 2;
            continue;
        }
        if (b == '`') {
            const j = std.mem.indexOfScalarPos(u8, text, i + 1, '`') orelse return true;
            i = j + 1;
            continue;
        }
        if (b == '[') {
            if (linkEnd(text, i, &brackets)) |end| {
                i = end.consumed;
                continue;
            }
            // No `]` yet, or `](` with no `)` yet: a later line may finish it.
            if (brackets.close >= text.len) return true;
            if (brackets.close + 1 < text.len and text[brackets.close + 1] == '(') return true;
        }
        if (i + 2 < text.len and (b == '*' or b == '_') and text[i + 1] == b and text[i + 2] == b) {
            const close = i + 3;
            const j = findSeq(text, close, text[i .. i + 3]) orelse return true;
            if (j > close) {
                i = j + 3;
                continue;
            }
        }
        if (i + 1 < text.len and text[i + 1] == b and (b == '*' or b == '_' or b == '~')) {
            const j = findSeq(text, i + 2, text[i .. i + 2]) orelse return true;
            i = j + 2;
            continue;
        }
        if (b == '*' or b == '_') {
            if (i + 1 >= text.len) return true;
            if (text[i + 1] != ' ') {
                const j = findSingle(text, i + 1, b) orelse return true;
                if (j > i + 1) {
                    i = j + 1;
                    continue;
                }
            }
        }
        if (view.isC1(text[i..])) {
            i += 2;
            continue;
        }
        i += 1;
    }
    return false;
}

/// Test instrumentation: how many times `render` ran, so a streaming test
/// can prove the closed part of a turn is rendered once and the open tail
/// never through here.
pub var render_calls: usize = 0;

/// Renders a whole document to styled rows, each at most `columns` wide.
pub fn render(alloc: std.mem.Allocator, text: []const u8, columns: usize, th: theme_mod.Theme) ![][]const u8 {
    var carry: Carry = .{};
    return renderFrom(alloc, text, columns, th, &carry);
}

/// Renders `text` as the continuation of a document whose earlier part left
/// `carry`, and updates `carry` for the text after it. For a `text` that
/// `split` closed, the rows are the ones a render of the whole document
/// gives for those lines.
pub fn renderFrom(alloc: std.mem.Allocator, text: []const u8, columns: usize, th: theme_mod.Theme, carry: *Carry) ![][]const u8 {
    if (builtin.is_test) render_calls += 1;
    var rows: std.ArrayList([]const u8) = .empty;
    errdefer rows.deinit(alloc);
    var para: std.ArrayList(u8) = .empty;
    defer para.deinit(alloc);
    var table: std.ArrayList([]const u8) = .empty;
    defer table.deinit(alloc);
    if (text.len == 0) return rows.toOwnedSlice(alloc);
    var in_code = carry.fence != null;
    var hl_state = if (carry.fence) |f| f.hl else highlight.State{};
    var hl_lang = if (carry.fence) |f| f.lang else highlight.Language.generic;
    // One trailing newline is the end of the last block, not an empty one
    // after it. Dropping it is also what makes a streamed prefix render as a
    // prefix of the finished turn (`split`).
    var iter = std.mem.splitScalar(u8, if (text[text.len - 1] == '\n') text[0 .. text.len - 1] else text, '\n');
    while (iter.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, " \t\r");
        if (std.mem.startsWith(u8, line, "```")) {
            try flushPara(alloc, &para, &rows, columns, th);
            try flushTable(alloc, &table, &rows, columns, th);
            if (!in_code) {
                const info = std.mem.trim(u8, line[3..], " \t\r");
                hl_lang = highlight.detect(info);
                hl_state = .{};
                // The language is a caption over the block's background, not
                // a border: the code rows below it stay plain text.
                if (info.len > 0) try rows.append(alloc, try captionRow(alloc, info, columns, th));
            }
            in_code = !in_code;
            continue;
        }
        if (in_code) {
            var code_text: std.ArrayList(u8) = .empty;
            defer code_text.deinit(alloc);
            // Trailing blanks are invisible on the band and would only be
            // copied with a selection.
            try putSanitized(alloc, &code_text, std.mem.trimEnd(u8, raw_line, " \t\r"));
            const styled_code = try highlight.line(alloc, code_text.items, hl_lang, &hl_state, th);
            defer alloc.free(styled_code);
            var styled: std.ArrayList(u8) = .empty;
            defer styled.deinit(alloc);
            try styled.appendSlice(alloc, th.paint(.code_block));
            try styled.appendSlice(alloc, styled_code);
            try appendCode(alloc, &rows, styled.items, columns, th);
            continue;
        }
        if (line.len == 0) {
            try flushPara(alloc, &para, &rows, columns, th);
            try flushTable(alloc, &table, &rows, columns, th);
            try rows.append(alloc, "");
            continue;
        }
        if (line[0] == '|') {
            try flushPara(alloc, &para, &rows, columns, th);
            try table.append(alloc, line);
            continue;
        }
        try flushTable(alloc, &table, &rows, columns, th);
        // Every block but a paragraph line ends the paragraph before it, so
        // its rows come first.
        if (!isParagraph(line)) try flushPara(alloc, &para, &rows, columns, th);
        if (headingBody(line)) |body| {
            var styled: std.ArrayList(u8) = .empty;
            defer styled.deinit(alloc);
            try styled.appendSlice(alloc, th.paint(.heading));
            try inlineSpan(alloc, &styled, body, th, 0);
            try styled.appendSlice(alloc, theme_mod.reset);
            try appendWrapped(alloc, &rows, styled.items, columns);
        } else if (isRule(line)) {
            try rows.append(alloc, try ruleRow(alloc, @min(columns, 40), th));
        } else if (quoteBody(line)) |quote| {
            var styled: std.ArrayList(u8) = .empty;
            defer styled.deinit(alloc);
            try styled.appendSlice(alloc, th.paint(.quote));
            for (0..quote.level) |_| {
                try styled.appendSlice(alloc, th.glyphs().quote);
                try styled.appendSlice(alloc, " ");
            }
            try inlineSpan(alloc, &styled, quote.body, th, 0);
            try styled.appendSlice(alloc, theme_mod.reset);
            try appendWrapped(alloc, &rows, styled.items, columns);
        } else if (listMarker(line, th.glyphs())) |marker| {
            try listItem(alloc, &rows, marker, indentLevel(raw_line), columns, th);
        } else {
            // A soft line break stays a line break: a model that writes one
            // item per line without markers is read that way.
            try para.appendSlice(alloc, line);
            try para.append(alloc, '\n');
        }
    }
    try flushPara(alloc, &para, &rows, columns, th);
    try flushTable(alloc, &table, &rows, columns, th);
    carry.* = if (in_code) .{ .fence = .{ .lang = hl_lang, .hl = hl_state } } else .{};
    return rows.toOwnedSlice(alloc);
}

/// Whether a non-blank line outside a fence or table is paragraph text.
fn isParagraph(line: []const u8) bool {
    return headingBody(line) == null and !isRule(line) and quoteBody(line) == null and
        listMarker(line, theme_mod.unicode_glyphs) == null;
}

fn flushPara(alloc: std.mem.Allocator, para: *std.ArrayList(u8), rows: *std.ArrayList([]const u8), columns: usize, th: theme_mod.Theme) !void {
    const body = std.mem.trimEnd(u8, para.items, " \n");
    if (body.len == 0) return;
    var styled: std.ArrayList(u8) = .empty;
    defer styled.deinit(alloc);
    try inlineSpan(alloc, &styled, body, th, 0);
    try appendWrapped(alloc, rows, styled.items, columns);
    para.clearRetainingCapacity();
}

fn appendWrapped(alloc: std.mem.Allocator, rows: *std.ArrayList([]const u8), styled: []const u8, columns: usize) !void {
    const wrapped = try view.wrapStyled(alloc, styled, @max(columns, 1), .word);
    defer alloc.free(wrapped);
    try rows.appendSlice(alloc, wrapped);
}

/// Erase in line, cursor to end: it paints the cells with the background in
/// force, so a code row's band reaches the edge without a space in the row,
/// and reflow and selection see only the code.
const erase_to_end = "\x1b[K";

/// Appends one code line's rows: `styled` (the block background opened, not
/// closed) wrapped to `columns`. Each row stands alone: it reopens the SGR
/// state the row before it left, fills its band with `erase_to_end`, and ends
/// in a reset, so no background reaches a line clear, an inserted line, or a
/// scroll that follows it.
fn appendCode(alloc: std.mem.Allocator, rows: *std.ArrayList([]const u8), styled: []const u8, columns: usize, th: theme_mod.Theme) !void {
    const base = th.paint(.code_block);
    const wrapped = try view.wrapStyled(alloc, styled, @max(columns, 1), .word);
    defer {
        for (wrapped) |fragment| alloc.free(fragment);
        alloc.free(wrapped);
    }
    var state: std.ArrayList(u8) = .empty;
    defer state.deinit(alloc);
    for (wrapped) |fragment| {
        var row: std.ArrayList(u8) = .empty;
        errdefer row.deinit(alloc);
        try row.appendSlice(alloc, state.items);
        try row.appendSlice(alloc, fragment);
        try closeBand(alloc, &row, view.styledWidth(fragment), columns, base);
        const owned = try row.toOwnedSlice(alloc);
        errdefer alloc.free(owned);
        try rows.append(alloc, owned);
        try carrySgr(alloc, &state, fragment, base);
    }
}

/// Ends a row painted on the band `base`: erase-to-end when a band is painted
/// and the row stops short of the edge, then a reset. A full row takes no
/// erase, since the cursor waits on its last cell and the erase would clear it.
fn closeBand(alloc: std.mem.Allocator, row: *std.ArrayList(u8), used: usize, columns: usize, base: []const u8) !void {
    if (base.len > 0 and used < @max(columns, 1)) try row.appendSlice(alloc, erase_to_end);
    try row.appendSlice(alloc, theme_mod.reset);
}

/// Folds the SGR sequences of `fragment` into `state`, the sequences that
/// reproduce the attributes in force at its end. A reset empties it; the
/// baseline `base` restarts it, which is exact because the highlighter
/// clears every attribute a token sets before it re-stamps the baseline. That
/// keeps `state` to a baseline and one token's style, however long the line.
fn carrySgr(alloc: std.mem.Allocator, state: *std.ArrayList(u8), fragment: []const u8, base: []const u8) !void {
    var i: usize = 0;
    while (i < fragment.len) {
        const end = view.escapeSpan(fragment, i) orelse {
            i += 1;
            continue;
        };
        const seq = fragment[i..end];
        i = end;
        if (seq.len < 3 or seq[1] != '[' or seq[seq.len - 1] != 'm') continue;
        if (std.mem.eql(u8, seq, theme_mod.reset) or std.mem.eql(u8, seq, "\x1b[m")) {
            state.clearRetainingCapacity();
        } else if (base.len > 0 and std.mem.eql(u8, seq, base)) {
            state.clearRetainingCapacity();
            try state.appendSlice(alloc, base);
        } else try state.appendSlice(alloc, seq);
    }
}

/// The caption above a fenced block: the language name over the block's own
/// background, dimmed, the strip filled to the edge like a code row.
fn captionRow(alloc: std.mem.Allocator, info: []const u8, columns: usize, th: theme_mod.Theme) ![]const u8 {
    var styled: std.ArrayList(u8) = .empty;
    errdefer styled.deinit(alloc);
    try styled.appendSlice(alloc, th.paint(.code_block));
    try styled.appendSlice(alloc, th.paint(.comment));
    try styled.append(alloc, ' ');
    try putSanitized(alloc, &styled, info);
    try closeBand(alloc, &styled, view.width(info) + 1, columns, th.paint(.code_block));
    return styled.toOwnedSlice(alloc);
}

/// Nesting depth of a list item, from its leading whitespace: two spaces per
/// level (a tab counts as four), capped so a deeply indented paste cannot
/// push the text off the row. Four-space indentation therefore reads as two
/// levels, which is a difference of indent only — the glyphs cycle.
fn indentLevel(raw_line: []const u8) usize {
    var cells_w: usize = 0;
    for (raw_line) |c| {
        if (c == ' ') cells_w += 1 else if (c == '\t') cells_w += 4 else break;
    }
    return @min(cells_w / 2, 3);
}

/// Builds one styled row: `open` prefix, inline-rendered `body`, reset.
fn spanRow(alloc: std.mem.Allocator, th: theme_mod.Theme, open: []const u8, body: []const u8) ![]const u8 {
    var styled: std.ArrayList(u8) = .empty;
    errdefer styled.deinit(alloc);
    try styled.appendSlice(alloc, open);
    try inlineSpan(alloc, &styled, body, th, 0);
    try styled.appendSlice(alloc, theme_mod.reset);
    return styled.toOwnedSlice(alloc);
}

/// `ch` repeated `n` times (byte-wise; `ch` is one UTF-8 glyph).
fn fill(alloc: std.mem.Allocator, ch: []const u8, n: usize) ![]const u8 {
    const bytes = try alloc.alloc(u8, ch.len * n);
    for (0..n) |i| @memcpy(bytes[ch.len * i ..][0..ch.len], ch);
    return bytes;
}

fn ruleRow(alloc: std.mem.Allocator, cells_w: usize, th: theme_mod.Theme) ![]const u8 {
    var styled: std.ArrayList(u8) = .empty;
    errdefer styled.deinit(alloc);
    try styled.appendSlice(alloc, th.paint(.dim));
    const glyph = try fill(alloc, th.glyphs().rule, cells_w);
    defer alloc.free(glyph);
    try styled.appendSlice(alloc, glyph);
    try styled.appendSlice(alloc, theme_mod.reset);
    return styled.toOwnedSlice(alloc);
}

/// The first line of `text` that holds words, as plain text with its block
/// markers (heading, quote, list item, task box) and inline markers dropped
/// and its control bytes stripped; blank lines, rules, and fence lines are
/// skipped. Null when no line holds any. Owned by the caller.
pub fn firstPlainLine(alloc: std.mem.Allocator, text: []const u8) !?[]const u8 {
    const plain: theme_mod.Theme = .{ .kind = .plain };
    var rows = std.mem.splitScalar(u8, text, '\n');
    var styled: std.ArrayList(u8) = .empty;
    defer styled.deinit(alloc);
    while (rows.next()) |row| {
        var line = std.mem.trim(u8, row, " \t\r");
        if (line.len == 0 or isRule(line) or std.mem.startsWith(u8, line, "```")) continue;
        if (headingBody(line)) |body| line = body;
        if (quoteBody(line)) |quote| line = quote.body;
        if (listMarker(line, plain.glyphs())) |marker| line = std.mem.trim(u8, marker.body, " ");
        styled.clearRetainingCapacity();
        try inlineSpan(alloc, &styled, line, plain, 0);
        // The plain theme still paints with SGR and links with OSC 8.
        var kept: usize = 0;
        var i: usize = 0;
        while (i < styled.items.len) {
            if (view.escapeSpan(styled.items, i)) |end| {
                i = end;
                continue;
            }
            styled.items[kept] = styled.items[i];
            kept += 1;
            i += 1;
        }
        const words = std.mem.trim(u8, styled.items[0..kept], " ");
        if (words.len > 0) return try alloc.dupe(u8, words);
    }
    return null;
}

/// `#`..`######` followed by a space; returns the heading text.
fn headingBody(line: []const u8) ?[]const u8 {
    var level: usize = 0;
    while (level < line.len and line[level] == '#') level += 1;
    if (level == 0 or level > 6) return null;
    if (level >= line.len or line[level] != ' ') return null;
    return std.mem.trim(u8, line[level + 1 ..], " ");
}

fn isRule(line: []const u8) bool {
    if (line.len < 3) return false;
    for (line) |c| if (c != '-' and c != '_' and c != ' ') return false;
    return true;
}

const Quote = struct { level: usize, body: []const u8 };

/// A `>` quote line: every leading `>` (spaces between them allowed) is one
/// level of nesting, drawn as one bar each, capped so a pasted mail thread
/// cannot push the text off the row.
fn quoteBody(line: []const u8) ?Quote {
    if (line.len < 2 or line[0] != '>') return null;
    var level: usize = 0;
    var rest = line;
    while (rest.len > 0 and rest[0] == '>') {
        level += 1;
        rest = std.mem.trimStart(u8, rest[1..], " ");
    }
    return .{ .level = @min(level, 4), .body = rest };
}

const Marker = struct {
    /// The marker glyph without its trailing space ("•", "☑", "1.").
    label: []const u8,
    /// Item text after the marker.
    body: []const u8,
};

/// A list item's marker, or null when the line is not one. The bullet glyph
/// comes from the theme (Unicode or ASCII) and, for unordered items, from the
/// nesting level — the level itself is read from the raw line's indentation.
fn listMarker(line: []const u8, gl: theme_mod.Glyphs) ?Marker {
    if (line.len >= 2 and (line[0] == '-' or line[0] == '*' or line[0] == '+') and line[1] == ' ') {
        const body = std.mem.trimStart(u8, line[2..], " ");
        if (body.len >= 4 and body[0] == '[' and body[2] == ']' and body[3] == ' ' and (body[1] == ' ' or body[1] == 'x' or body[1] == 'X')) {
            return .{ .label = if (body[1] == ' ') gl.task_todo else gl.task_done, .body = body[4..] };
        }
        return .{ .label = gl.bullet, .body = body };
    }
    var digits: usize = 0;
    while (digits < line.len and std.ascii.isDigit(line[digits])) digits += 1;
    if (digits > 0 and digits + 1 < line.len and line[digits] == '.' and line[digits + 1] == ' ') {
        return .{ .label = line[0 .. digits + 1], .body = line[digits + 2 ..] };
    }
    return null;
}

/// The bullet for a nesting level; ordered items and task boxes keep theirs.
fn levelBullet(label: []const u8, level: usize, gl: theme_mod.Glyphs) []const u8 {
    if (!std.mem.eql(u8, label, gl.bullet)) return label;
    return switch (level % 3) {
        0 => gl.bullet,
        1 => gl.bullet2,
        else => gl.bullet3,
    };
}

/// One list item: two spaces of indent per nesting level, the styled marker,
/// then the text with a hanging indent under it. The indent is measured in
/// display cells, because the ASCII task boxes (`[x]`) are wider than the
/// Unicode ones.
fn listItem(alloc: std.mem.Allocator, rows: *std.ArrayList([]const u8), marker: Marker, level: usize, columns: usize, th: theme_mod.Theme) !void {
    const label = levelBullet(marker.label, level, th.glyphs());
    const lead = 2 * level;
    const indent = @min(lead + view.width(label) + 1, columns -| 1);
    const body = try spanRow(alloc, th, "", marker.body);
    defer alloc.free(body);
    const wrapped = try view.wrapStyled(alloc, body, @max(columns -| indent, 1), .word);
    defer {
        for (wrapped) |w| alloc.free(w);
        alloc.free(wrapped);
    }
    var first = true;
    for (wrapped) |l| {
        var line: std.ArrayList(u8) = .empty;
        errdefer line.deinit(alloc);
        const spaces = try fill(alloc, " ", if (first) lead else indent);
        defer alloc.free(spaces);
        try line.appendSlice(alloc, spaces);
        if (first) {
            first = false;
            try line.appendSlice(alloc, th.paint(.bullet));
            try line.appendSlice(alloc, label);
            try line.appendSlice(alloc, theme_mod.reset);
            try line.append(alloc, ' ');
        }
        try line.appendSlice(alloc, l);
        try rows.append(alloc, try line.toOwnedSlice(alloc));
    }
}

// ----- pipe tables -----

fn flushTable(alloc: std.mem.Allocator, table: *std.ArrayList([]const u8), rows: *std.ArrayList([]const u8), columns: usize, th: theme_mod.Theme) !void {
    defer table.clearRetainingCapacity();
    if (table.items.len < 2 or !isDivider(table.items[1])) {
        // Not a table: render the pipe lines as ordinary paragraphs.
        for (table.items) |l| {
            try paraLine(alloc, rows, l, columns, th);
        }
        return;
    }
    var grid: std.ArrayList([]const []const u8) = .empty;
    defer {
        for (grid.items) |row_cells| alloc.free(row_cells);
        grid.deinit(alloc);
    }
    // Every row is cut or padded to the header's cell count here, so no
    // later index needs a guard.
    const header_cells = try splitCells(alloc, table.items[0]);
    const ncols = header_cells.len;
    {
        errdefer alloc.free(header_cells);
        try grid.ensureUnusedCapacity(alloc, table.items.len - 1);
    }
    grid.appendAssumeCapacity(header_cells);
    for (table.items[2..]) |l| {
        const given = try splitCells(alloc, l);
        defer alloc.free(given);
        const cells = try alloc.alloc([]const u8, ncols);
        const kept = @min(given.len, ncols);
        @memcpy(cells[0..kept], given[0..kept]);
        @memset(cells[kept..], "");
        grid.appendAssumeCapacity(cells);
    }
    // The divider row carries the alignment of every column (`:---`, `---:`,
    // `:---:`); a column it does not mention is left-aligned.
    const alignments = try splitCells(alloc, table.items[1]);
    defer alloc.free(alignments);
    var widths = try alloc.alloc(usize, ncols);
    defer alloc.free(widths);
    @memset(widths, 1);
    for (grid.items) |row_cells| {
        for (row_cells, 0..) |cell, i| widths[i] = @max(widths[i], view.width(cell));
    }
    // Shrink columns when the table cannot fit; cells wrap inside their
    // column instead of widening the table past the terminal.
    const budget = columns -| (ncols + 1);
    var total: usize = 0;
    for (widths) |w| total += w;
    if (total > budget) {
        const share = @max(budget / ncols, 1);
        for (widths) |*w| w.* = @min(w.*, share);
    }
    for (grid.items, 0..) |row_cells, row_index| {
        const header = row_index == 0;
        var bands: usize = 1;
        var wrapped_cells: std.ArrayList([][]const u8) = .empty;
        defer {
            for (wrapped_cells.items) |cell_lines| {
                for (cell_lines) |l| alloc.free(l);
                alloc.free(cell_lines);
            }
            wrapped_cells.deinit(alloc);
        }
        for (row_cells, 0..) |cell, i| {
            const w = widths[i];
            const styled_cell = if (header)
                try spanRow(alloc, th, th.paint(.heading), cell)
            else
                try spanRow(alloc, th, "", cell);
            defer alloc.free(styled_cell);
            const lines = try view.wrapStyled(alloc, styled_cell, w, .word);
            try wrapped_cells.append(alloc, lines);
            bands = @max(bands, lines.len);
        }
        for (0..bands) |band| {
            var line: std.ArrayList(u8) = .empty;
            defer line.deinit(alloc);
            for (wrapped_cells.items, 0..) |cell_lines, i| {
                const text = if (band < cell_lines.len) cell_lines[band] else "";
                if (i > 0) try line.appendSlice(alloc, th.glyphs().table_bar);
                try line.appendSlice(alloc, " ");
                // Alignment is the column's, not the cell's: the padding is
                // split between the two sides according to the divider row.
                const room = widths[i] -| view.styledWidth(text);
                const before = switch (if (i < alignments.len) alignOf(alignments[i]) else .left) {
                    .left => 0,
                    .right => room,
                    .center => room / 2,
                };
                try padCells(alloc, &line, before);
                try line.appendSlice(alloc, text);
                try padCells(alloc, &line, room - before);
                try line.append(alloc, ' ');
            }
            try tableRow(alloc, rows, line.items, columns);
        }
        if (header) {
            var divider: std.ArrayList(u8) = .empty;
            defer divider.deinit(alloc);
            for (widths, 0..) |w, i| {
                if (i > 0) try divider.appendSlice(alloc, th.glyphs().table_joint);
                const seg = try fill(alloc, th.glyphs().rule, w + 2);
                defer alloc.free(seg);
                try divider.appendSlice(alloc, seg);
            }
            try tableRow(alloc, rows, divider.items, columns);
        }
    }
}

/// One table row, cut at the terminal's edge. A table with more columns than
/// the width has cells cannot be narrowed further; its rows break at the
/// edge so nothing is lost and no row is ever wider than the screen.
fn tableRow(alloc: std.mem.Allocator, rows: *std.ArrayList([]const u8), line: []const u8, columns: usize) !void {
    const wrapped = try view.wrapStyled(alloc, line, @max(columns, 1), .character);
    defer alloc.free(wrapped);
    try rows.appendSlice(alloc, wrapped);
}

/// A paragraph-shaped fallback for pipe lines that do not form a table.
fn paraLine(alloc: std.mem.Allocator, rows: *std.ArrayList([]const u8), text: []const u8, columns: usize, th: theme_mod.Theme) !void {
    var para: std.ArrayList(u8) = .empty;
    defer para.deinit(alloc);
    try para.appendSlice(alloc, text);
    try para.append(alloc, ' ');
    try flushPara(alloc, &para, rows, columns, th);
}

/// Column alignment, read from one cell of the divider row.
const Alignment = enum { left, center, right };

fn alignOf(cell: []const u8) Alignment {
    const c = std.mem.trim(u8, cell, " ");
    if (c.len == 0) return .left;
    const left = c[0] == ':';
    const right = c[c.len - 1] == ':';
    if (left and right) return .center;
    return if (right) .right else .left;
}

fn padCells(alloc: std.mem.Allocator, line: *std.ArrayList(u8), n: usize) !void {
    const spaces = try fill(alloc, " ", n);
    defer alloc.free(spaces);
    try line.appendSlice(alloc, spaces);
}

fn isDivider(line: []const u8) bool {
    var cells: usize = 0;
    var iter = std.mem.splitScalar(u8, std.mem.trim(u8, line, "| "), '|');
    while (iter.next()) |cell| {
        const c = std.mem.trim(u8, cell, " ");
        if (c.len == 0) return false;
        for (c) |ch| if (ch != '-' and ch != ':') return false;
        cells += 1;
    }
    return cells > 0;
}

fn splitCells(alloc: std.mem.Allocator, line: []const u8) ![][]const u8 {
    var cells: std.ArrayList([]const u8) = .empty;
    errdefer cells.deinit(alloc);
    var inner = line;
    if (inner.len > 0 and inner[0] == '|') inner = inner[1..];
    if (inner.len > 0 and inner[inner.len - 1] == '|') inner = inner[0 .. inner.len - 1];
    var iter = std.mem.splitScalar(u8, inner, '|');
    while (iter.next()) |cell| try cells.append(alloc, std.mem.trim(u8, cell, " "));
    return cells.toOwnedSlice(alloc);
}

// ----- inline spans -----

/// Renders inline markdown into `out` (unclosed markers stay literal).
/// `depth` bounds nesting; control bytes are stripped from all literal text.
fn inlineSpan(alloc: std.mem.Allocator, out: *std.ArrayList(u8), text: []const u8, th: theme_mod.Theme, depth: usize) !void {
    var brackets: Brackets = .{};
    var i: usize = 0;
    while (i < text.len) {
        const b = text[i];
        if (b == '\\' and i + 1 < text.len) {
            try putSanitizedByte(alloc, out, text[i + 1]);
            i += 2;
            continue;
        }
        if (b == '`') {
            if (std.mem.indexOfScalarPos(u8, text, i + 1, '`')) |j| {
                try out.appendSlice(alloc, th.paint(.code));
                try putSanitized(alloc, out, text[i + 1 .. j]);
                try out.appendSlice(alloc, theme_mod.reset);
                i = j + 1;
                continue;
            }
        }
        if (b == '[') {
            if (linkEnd(text, i, &brackets)) |end| {
                // An http(s) target becomes a real hyperlink (OSC 8); any other
                // target keeps the label styled but is not made clickable, so a
                // model cannot smuggle an arbitrary scheme into the terminal.
                const target = text[end.target_start..end.target_end];
                const hyperlink = isHyperlink(target);
                if (hyperlink) {
                    try out.appendSlice(alloc, "\x1b]8;;");
                    try out.appendSlice(alloc, target);
                    try out.appendSlice(alloc, "\x1b\\");
                }
                try out.appendSlice(alloc, th.paint(.link));
                try putSanitized(alloc, out, text[i + 1 .. end.label_close]);
                try out.appendSlice(alloc, theme_mod.reset);
                if (hyperlink) try out.appendSlice(alloc, "\x1b]8;;\x1b\\");
                i = end.consumed;
                continue;
            }
        }
        if (depth < 3 and i + 2 < text.len and (b == '*' or b == '_') and text[i + 1] == b and text[i + 2] == b) {
            // ***bold italic*** / ___bold italic___: the triple before the
            // double, or the double would swallow one star of each end.
            const close = i + 3;
            if (findSeq(text, close, text[i .. i + 3])) |j| {
                if (j > close) {
                    try out.appendSlice(alloc, th.paint(.bold));
                    try out.appendSlice(alloc, th.paint(.italic));
                    try inlineSpan(alloc, out, text[close..j], th, depth + 1);
                    try out.appendSlice(alloc, theme_mod.reset);
                    i = j + 3;
                    continue;
                }
            }
        }
        if (depth < 3 and i + 1 < text.len and text[i + 1] == b and (b == '*' or b == '_' or b == '~')) {
            // **bold** / __bold__ / ~~strike~~ (double markers first).
            const close = i + 2;
            if (findSeq(text, close, text[i .. i + 2])) |j| {
                const style: theme_mod.Style = if (b == '~') .dim else .bold;
                try out.appendSlice(alloc, th.paint(style));
                if (b == '~') try out.appendSlice(alloc, "\x1b[9m");
                try inlineSpan(alloc, out, text[close..j], th, depth + 1);
                try out.appendSlice(alloc, theme_mod.reset);
                i = j + 2;
                continue;
            }
        }
        if (depth < 3 and (b == '*' or b == '_') and i + 1 < text.len and text[i + 1] != ' ') {
            // *italic* / _italic_: a single marker with a single closer and
            // non-empty content (so a lone `**` pair stays literal).
            if (findSingle(text, i + 1, b)) |j| {
                if (j > i + 1) {
                    try out.appendSlice(alloc, th.paint(.italic));
                    try inlineSpan(alloc, out, text[i + 1 .. j], th, depth + 1);
                    try out.appendSlice(alloc, theme_mod.reset);
                    i = j + 1;
                    continue;
                }
            }
        }
        if (view.isC1(text[i..])) {
            i += 2;
            continue;
        }
        try putSanitizedByte(alloc, out, b);
        i += 1;
    }
}

const LinkEnd = struct { label_close: usize, target_start: usize, target_end: usize, consumed: usize };

/// `[label](target)` starting at `open`. The label is rendered; the target is
/// kept so an http(s) one can become an OSC 8 hyperlink.
/// The `]` and `)` the last `linkEnd` found. Opens are tried left to right,
/// so the first `]` after one is still the first after the next while that
/// one comes before it: a run of `[` scans the text once, not once each.
const Brackets = struct {
    scanned: bool = false,
    /// The first `]` after the open the scan started from; `text.len` for none.
    close: usize = 0,
    /// The first `)` after `close`'s `(`, once looked for.
    paren: ?usize = null,
};

fn linkEnd(text: []const u8, open: usize, cache: *Brackets) ?LinkEnd {
    if (!cache.scanned or cache.close < open + 1) {
        cache.close = std.mem.indexOfScalarPos(u8, text, open + 1, ']') orelse text.len;
        cache.paren = null;
        cache.scanned = true;
    }
    const label_close = cache.close;
    if (label_close + 1 >= text.len or text[label_close + 1] != '(') return null;
    if (cache.paren == null) cache.paren = std.mem.indexOfScalarPos(u8, text, label_close + 2, ')') orelse text.len;
    const target_close = cache.paren.?; // checked: set on the line above
    if (target_close >= text.len) return null;
    return .{ .label_close = label_close, .target_start = label_close + 2, .target_end = target_close, .consumed = target_close + 1 };
}

/// Whether a link target is safe to emit as an OSC 8 hyperlink: an http(s)
/// URL, bounded, with no control bytes that could terminate the sequence early
/// or inject another escape. Everything else renders as styled label text.
fn isHyperlink(target: []const u8) bool {
    if (target.len == 0 or target.len > 2048) return false;
    if (!std.ascii.startsWithIgnoreCase(target, "http://") and
        !std.ascii.startsWithIgnoreCase(target, "https://")) return false;
    for (target, 0..) |c, i| {
        if (c < 0x20 or c == 0x7f or view.isC1(target[i..])) return false;
    }
    return true;
}

fn findSeq(text: []const u8, from: usize, seq: []const u8) ?usize {
    return std.mem.indexOfPos(u8, text, from, seq);
}

/// Next lone occurrence of `c` (not doubled) at or after `from`.
fn findSingle(text: []const u8, from: usize, c: u8) ?usize {
    var i = from;
    while (i < text.len) : (i += 1) {
        if (text[i] == c and (i + 1 >= text.len or text[i + 1] != c)) return i;
    }
    return null;
}

fn putSanitized(alloc: std.mem.Allocator, out: *std.ArrayList(u8), text: []const u8) !void {
    var i: usize = 0;
    while (i < text.len) : (i += 1) {
        if (view.isC1(text[i..])) {
            i += 1;
            continue;
        }
        try putSanitizedByte(alloc, out, text[i]);
    }
}

/// Every control byte but the line break is dropped: the break is the one
/// the paragraph kept on purpose, and the wrapper turns it into a row.
fn putSanitizedByte(alloc: std.mem.Allocator, out: *std.ArrayList(u8), b: u8) !void {
    if (b == '\t') {
        // To the next multiple of four cells of the row so far.
        const row_start = if (std.mem.lastIndexOfScalar(u8, out.items, '\n')) |n| n + 1 else 0;
        const column = view.styledWidth(out.items[row_start..]);
        return out.appendNTimes(alloc, ' ', view.tab_width - column % view.tab_width);
    }
    if ((b >= 32 and b != 127) or b == '\n') try out.append(alloc, b);
}

test "inline styles, links, and sanitization" {
    const th = theme_mod.Theme{ .kind = .plain };
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(std.testing.allocator);
    try inlineSpan(std.testing.allocator, &out, "**b** *i* ~~s~~ `c` [t](http://x) \x1b[31mred\x1b[0m", th, 0);
    const s = out.items;
    try std.testing.expect(std.mem.indexOf(u8, s, "\x1b[1mb\x1b[0m") != null);
    try std.testing.expect(std.mem.indexOf(u8, s, "\x1b[3mi\x1b[0m") != null);
    try std.testing.expect(std.mem.indexOf(u8, s, "~~") == null);
    try std.testing.expect(std.mem.indexOf(u8, s, "\x1b[4mt\x1b[0m") != null);
    // An http(s) target wraps the styled label in OSC 8.
    try std.testing.expect(std.mem.indexOf(u8, s, "\x1b]8;;http://x\x1b\\") != null);
    try std.testing.expect(std.mem.indexOf(u8, s, "\x1b]8;;\x1b\\") != null);
    var escapes: usize = 0;
    for (s) |c| {
        if (c == 0x1b) escapes += 1;
    }
    try std.testing.expect(escapes >= 6);
    // Unclosed markers stay literal.
    var out2: std.ArrayList(u8) = .empty;
    defer out2.deinit(std.testing.allocator);
    try inlineSpan(std.testing.allocator, &out2, "a ** b", th, 0);
    try std.testing.expect(std.mem.indexOf(u8, out2.items, "**") != null);

    // A non-http target keeps the styled label but is not clickable, and a
    // control byte in the target is refused rather than emitted.
    for ([_][]const u8{ "see [here](ftp://x)", "[x](http://a\x07b)" }) |source| {
        var out3: std.ArrayList(u8) = .empty;
        defer out3.deinit(std.testing.allocator);
        try inlineSpan(std.testing.allocator, &out3, source, th, 0);
        try std.testing.expect(std.mem.indexOf(u8, out3.items, "\x1b]8") == null);
        try std.testing.expect(std.mem.indexOf(u8, out3.items, "\x1b[4m") != null);
    }
}

/// Every row of a render, joined with newlines and stripped of its escape
/// sequences: the tests below assert on layout and glyphs, while the styles
/// themselves are pinned in `theme.zig`. Caller frees.
fn joined(alloc: std.mem.Allocator, src: []const u8, columns: usize, th: theme_mod.Theme) ![]u8 {
    const rows = try render(alloc, src, columns, th);
    defer {
        for (rows) |r| alloc.free(r);
        alloc.free(rows);
    }
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    for (rows) |r| {
        try stripEscapes(alloc, &out, r);
        try out.append(alloc, '\n');
    }
    return out.toOwnedSlice(alloc);
}

/// Copies `row` without its SGR sequences.
fn stripEscapes(alloc: std.mem.Allocator, out: *std.ArrayList(u8), row: []const u8) !void {
    var i: usize = 0;
    while (i < row.len) {
        if (row[i] == 0x1b) {
            i += 1;
            if (i < row.len and row[i] == '[') {
                i += 1;
                while (i < row.len and row[i] >= 0x20 and row[i] <= 0x3f) i += 1;
            }
            if (i < row.len) i += 1;
            continue;
        }
        try out.append(alloc, row[i]);
        i += 1;
    }
}

test "blocks: headings, lists, tasks, quotes, code, tables" {
    const alloc = std.testing.allocator;
    const th = theme_mod.Theme{ .kind = .plain };
    const src =
        "## Title\n" ++
        "- plain\n" ++
        "- [x] done\n" ++
        "- [ ] todo\n" ++
        "1. first\n" ++
        "> quoted\n" ++
        "```zig\n" ++
        "x = y;\n" ++
        "```\n" ++
        "| a | bb |\n" ++
        "| --- | --- |\n" ++
        "| 1 | long cell |\n";
    const s = try joined(alloc, src, 60, th);
    defer alloc.free(s);
    try std.testing.expect(std.mem.indexOf(u8, s, "Title") != null);
    try std.testing.expect(std.mem.indexOf(u8, s, "##") == null);
    try std.testing.expect(std.mem.indexOf(u8, s, "☑") != null);
    try std.testing.expect(std.mem.indexOf(u8, s, "☐") != null);
    try std.testing.expect(std.mem.indexOf(u8, s, "•") != null);
    try std.testing.expect(std.mem.indexOf(u8, s, "1. ") != null);
    try std.testing.expect(std.mem.indexOf(u8, s, "first") != null);
    try std.testing.expect(std.mem.indexOf(u8, s, "▌ quoted") != null);
    try std.testing.expect(std.mem.indexOf(u8, s, "x = y;") != null);
    try std.testing.expect(std.mem.indexOf(u8, s, "---") == null);
    try std.testing.expect(std.mem.indexOf(u8, s, "┼") != null);
    try std.testing.expect(std.mem.indexOf(u8, s, "│") != null);
    // The fence's language is a caption row above the code, not a border.
    try std.testing.expect(std.mem.indexOf(u8, s, " zig") != null);
    try std.testing.expect(std.mem.indexOf(u8, s, "```") == null);
}

test "a code row is the code alone: the band is erase-to-end, each row closes its own style" {
    const alloc = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(alloc);
    defer arena.deinit();
    const a = arena.allocator();
    const th = theme_mod.Theme{ .kind = .truecolor };
    const base = th.paint(.code_block);
    const long = "values = [first_value, second_value, third_value, fourth_value]";
    const rows = try render(a, "```python\ndef mean(values):   \n\n    return sum(values)\n" ++ long ++ "\n```\n", 24, th);
    // Caption, three lines, and the long line in four rows.
    try std.testing.expectEqual(@as(usize, 8), rows.len);
    for (rows, 0..) |row, i| {
        errdefer std.debug.print("row {d}: {any}\n", .{ i, row });
        // Every row opens the band and closes everything it set, so no
        // background reaches the line clear, the inserted line, or the
        // scroll that follows it on the screen.
        try std.testing.expect(std.mem.startsWith(u8, row, base));
        try std.testing.expect(std.mem.endsWith(u8, row, theme_mod.reset));
        var plain: std.ArrayList(u8) = .empty;
        try stripEscapes(a, &plain, row);
        try std.testing.expect(plain.items.len == 0 or plain.items[plain.items.len - 1] != ' ');
        // A row short of the edge fills it; a full row does not, since the
        // erase would clear the cell the cursor waits on.
        const filled = std.mem.endsWith(u8, row, erase_to_end ++ theme_mod.reset);
        try std.testing.expectEqual(view.styledWidth(row) < 24, filled);
    }
    var first: std.ArrayList(u8) = .empty;
    try stripEscapes(a, &first, rows[1]);
    try std.testing.expectEqualStrings("def mean(values):", first.items);
    // The empty line is a band with nothing on it.
    try std.testing.expectEqualStrings(try std.mem.concat(a, u8, &.{ base, erase_to_end, theme_mod.reset }), rows[2]);
    // A row cut inside a token reopens the token's style on the band.
    const cut = try render(a, "```python\nx = \"aaaa bbbb cccc\"\n```\n", 10, th);
    try std.testing.expectEqual(@as(usize, 3), cut.len);
    try std.testing.expect(std.mem.startsWith(u8, cut[2], try std.mem.concat(a, u8, &.{ base, th.paint(.string), "bbbb" })));
    // Plain output paints no band, so it has nothing to erase.
    const bare = try render(a, "```\nx\n```\n", 24, .{ .kind = .plain });
    try std.testing.expectEqualStrings("x" ++ theme_mod.reset, bare[0]);
}

test "nested list items indent by level and cycle their bullet" {
    const alloc = std.testing.allocator;
    const th = theme_mod.Theme{ .kind = .plain };
    const s = try joined(alloc,
        \\- top
        \\  - second
        \\    - third
        \\  - [ ] task
        \\  3. ordered
    , 40, th);
    defer alloc.free(s);
    try std.testing.expect(std.mem.indexOf(u8, s, "• top") != null);
    try std.testing.expect(std.mem.indexOf(u8, s, "  ◦ second") != null);
    try std.testing.expect(std.mem.indexOf(u8, s, "    ▪ third") != null);
    // A task box and an ordered number keep their own marker at any level.
    try std.testing.expect(std.mem.indexOf(u8, s, "  ☐ task") != null);
    try std.testing.expect(std.mem.indexOf(u8, s, "  3. ordered") != null);
}

test "a wrapped list item hangs under its text, not under its bullet" {
    const alloc = std.testing.allocator;
    const th = theme_mod.Theme{ .kind = .plain };
    const s = try joined(alloc, "  - alpha beta gamma delta", 14, th);
    defer alloc.free(s);
    var rows = std.mem.splitScalar(u8, s, '\n');
    // Two cells of nesting plus the glyph and its space: the text has ten of
    // the fourteen columns, and the continuation rows line up under it.
    try std.testing.expectEqualStrings("  ◦ alpha beta", rows.next().?);
    try std.testing.expectEqualStrings("    gamma", rows.next().?);
    try std.testing.expectEqualStrings("    delta", rows.next().?);
}

test "table columns follow the alignment of the divider row" {
    const alloc = std.testing.allocator;
    const th = theme_mod.Theme{ .kind = .plain };
    const s = try joined(alloc,
        \\| left | mid | right |
        \\| :--- | :---: | ---: |
        \\| a | b | c |
    , 60, th);
    defer alloc.free(s);
    // Column widths are 4, 3, 5; the single letters sit at the three sides.
    try std.testing.expect(std.mem.indexOf(u8, s, " a    │  b  │     c ") != null);
}

test "a tab reaches the next stop in code and prose; C1 controls are dropped" {
    const alloc = std.testing.allocator;
    const th = theme_mod.Theme{ .kind = .plain };
    const s = try joined(alloc, "```\nif x:\n\treturn 1\n```\n\na\tb \u{9b}31m", 40, th);
    defer alloc.free(s);
    try std.testing.expect(std.mem.indexOf(u8, s, "    return 1") != null);
    try std.testing.expect(std.mem.indexOf(u8, s, "a   b") != null);
    try std.testing.expect(std.mem.indexOf(u8, s, "\u{9b}") == null);
}

test "a row wider or narrower than the header is cut or padded to it" {
    const alloc = std.testing.allocator;
    const th = theme_mod.Theme{ .kind = .plain };
    const s = try joined(alloc,
        \\| a |
        \\| --- |
        \\| 1 | 2 | 3 |
        \\| x | y |
    , 40, th);
    defer alloc.free(s);
    try std.testing.expect(std.mem.indexOf(u8, s, "2") == null);
    try std.testing.expect(std.mem.indexOf(u8, s, "y") == null);
    const narrow = try joined(alloc,
        \\| a | b | c |
        \\| --- | --- | --- |
        \\| 1 |
    , 40, th);
    defer alloc.free(narrow);
    try std.testing.expect(std.mem.indexOf(u8, narrow, " 1 │   │   ") != null);
}

test "the ASCII glyph set replaces every decoration" {
    const alloc = std.testing.allocator;
    const th = theme_mod.Theme{ .kind = .plain, .glyph_set = .ascii };
    const s = try joined(alloc,
        \\- [x] done
        \\- item
        \\> quoted
        \\
        \\---
        \\
        \\| a | b |
        \\| --- | --- |
        \\| 1 | 2 |
    , 40, th);
    defer alloc.free(s);
    for ([_][]const u8{ "☑", "•", "▌", "─", "│", "┼" }) |glyph| {
        try std.testing.expect(std.mem.indexOf(u8, s, glyph) == null);
    }
    try std.testing.expect(std.mem.indexOf(u8, s, "[x] done") != null);
    try std.testing.expect(std.mem.indexOf(u8, s, "* item") != null);
    try std.testing.expect(std.mem.indexOf(u8, s, "| quoted") != null);
    try std.testing.expect(std.mem.indexOf(u8, s, "---+---") != null);
}

test "the streaming split closes what can no longer change" {
    // A paragraph line is stable at its newline; the line still being
    // written never is.
    const partial = split("done.\nnext li", .{});
    try std.testing.expectEqualStrings("done.\n", partial.closed);
    try std.testing.expectEqualStrings("next li", partial.open);
    const first = split("one line so far", .{});
    try std.testing.expectEqualStrings("", first.closed);
    try std.testing.expectEqualStrings("one line so far", first.open);
    // An inline span still open holds its paragraph until it closes.
    try std.testing.expectEqualStrings("", split("a **bold\nstill\n", .{}).closed);
    try std.testing.expectEqualStrings("a **bold\nstill** here\n", split("a **bold\nstill** here\nx", .{}).closed);
    try std.testing.expectEqualStrings("", split("see `the\n", .{}).closed);
    try std.testing.expectEqualStrings("", split("a [link\n", .{}).closed);
    // A blank line ends a held paragraph whatever is open in it.
    try std.testing.expectEqualStrings("a *b\n\n", split("a *b\n\nc", .{}).closed);
    // A fence: the opener waits for its first line, then each line closes.
    try std.testing.expectEqualStrings("intro\n", split("intro\n```zig\n", .{}).closed);
    try std.testing.expectEqualStrings("intro\n```zig\nconst x = 1;\n", split("intro\n```zig\nconst x = 1;\nconst", .{}).closed);
    // Inside a fence a carried state continues it: a heading-like line is code.
    const inside: Carry = .{ .fence = .{ .lang = .zig } };
    try std.testing.expectEqualStrings("# not a heading\n", split("# not a heading\n| x", inside).closed);
    try std.testing.expectEqualStrings("```\nafter\n", split("```\nafter\n| x", inside).closed);
    // Single-line blocks close at their newline; a table keeps accumulating
    // rows until a line that is not one ends it.
    try std.testing.expectEqualStrings("# Title\n", split("# Title\nnext", .{}).closed);
    try std.testing.expectEqualStrings("- one\n", split("- one\nnext", .{}).closed);
    try std.testing.expectEqualStrings("", split("| a |\n| - |\n", .{}).closed);
    try std.testing.expectEqualStrings("| a |\n| - |\nafter\n", split("| a |\n| - |\nafter\n", .{}).closed);
    // The closed prefix plus the open tail is always the whole text.
    const text = "a\n\n- b\n| c |\n";
    const both = split(text, .{});
    try std.testing.expectEqual(text.len, both.closed.len + both.open.len);
}

/// Streams `document` through `split` and `renderFrom` one byte at a time,
/// as the transcript does, and checks that the rows written equal one
/// render of the whole and that the open part never exceeds `max_open`
/// lines. Returns how many renders the stream took.
fn expectStreamsAsWhole(document: []const u8, columns: usize, th: theme_mod.Theme, max_open: usize) !usize {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const whole = try render(alloc, document, columns, th);
    var streamed: std.ArrayList([]const u8) = .empty;
    var carry: Carry = .{};
    var flushed: usize = 0;
    var renders: usize = 0;
    for (1..document.len + 1) |n| {
        const part = split(document[flushed..n], carry);
        if (part.closed.len > 0) {
            try streamed.appendSlice(alloc, try renderFrom(alloc, part.closed, columns, th, &carry));
            flushed += part.closed.len;
            renders += 1;
        }
        // The open part restarts at the same point from the start, too.
        try std.testing.expectEqual(flushed, split(document[0..n], .{}).closed.len);
        try std.testing.expect(std.mem.count(u8, document[flushed..n], "\n") < max_open);
    }
    try streamed.appendSlice(alloc, try renderFrom(alloc, document[flushed..], columns, th, &carry));
    try std.testing.expectEqual(whole.len, streamed.items.len);
    for (whole, streamed.items) |w, r| try std.testing.expectEqualStrings(w, r);
    return renders;
}

test "a streamed document renders the same as the finished one, line by line" {
    const th = theme_mod.Theme{ .kind = .truecolor };
    for (fixtures) |document| _ = try expectStreamsAsWhole(document, 32, th, 64);
    // Plain lines, a list, quotes, and a fenced block with a block comment
    // whose highlighting carries across lines.
    _ = try expectStreamsAsWhole("one\ntwo **b** three\n- item\n  continued\n> q\ntext\n# Head\n```c\nint x; /* a\nb */ int y;\n```\nend\n", 40, th, 2);
    // A paragraph followed by a heading keeps its order in one render too.
    const ordered = try joined(std.testing.allocator, "text\n# Head\nmore\n", 40, .{ .kind = .plain });
    defer std.testing.allocator.free(ordered);
    try std.testing.expectEqualStrings("text\nHead\nmore\n", ordered);
}

test "a paragraph line closed by spanOpen renders the same with any line after it" {
    // A small alphabet of every marker the inline renderer reads, over
    // random two-line paragraphs: when the first line is not open, the
    // inline render of both is the render of each, joined by a break.
    const alloc = std.testing.allocator;
    const th = theme_mod.Theme{ .kind = .plain };
    const alphabet = "a *_`[]()\\~";
    var prng = std.Random.DefaultPrng.init(0x5eed);
    const random = prng.random();
    var buf: [24]u8 = undefined;
    var stable: usize = 0;
    for (0..20_000) |_| {
        const len_a = random.intRangeAtMost(usize, 1, 10);
        const len_b = random.intRangeAtMost(usize, 1, 10);
        for (buf[0 .. len_a + len_b + 1], 0..) |*c, k| c.* = if (k == len_a) '\n' else alphabet[random.uintLessThan(usize, alphabet.len)];
        const both = buf[0 .. len_a + len_b + 1];
        const first = both[0..len_a];
        const second = both[len_a + 1 ..];
        if (spanOpen(first)) continue;
        stable += 1;
        var whole: std.ArrayList(u8) = .empty;
        defer whole.deinit(alloc);
        try inlineSpan(alloc, &whole, both, th, 0);
        var parts: std.ArrayList(u8) = .empty;
        defer parts.deinit(alloc);
        try inlineSpan(alloc, &parts, first, th, 0);
        try parts.append(alloc, '\n');
        try inlineSpan(alloc, &parts, second, th, 0);
        try std.testing.expectEqualStrings(parts.items, whole.items);
    }
    // The alphabet is dense in markers; a fair share still closes.
    try std.testing.expect(stable > 2_000);
}

test "inline code in headings and items, bold italic, nested quotes, numbered starts, empty cells, soft breaks" {
    const alloc = std.testing.allocator;
    const th = theme_mod.Theme{ .kind = .plain };
    const s = try joined(alloc,
        \\## Use `zig build`
        \\- run `make check` first
        \\- ***all three***
        \\> > deep
        \\> > > deeper
        \\3. third
        \\4. fourth
        \\
        \\| a |  | c |
        \\| --- | --- | --- |
        \\| 1 |  | 3 |
        \\
        \\one
        \\two
        \\three
    , 40, th);
    defer alloc.free(s);
    const expected =
        "Use zig build\n" ++
        "• run make check first\n" ++
        "• all three\n" ++
        "▌ ▌ deep\n" ++
        "▌ ▌ ▌ deeper\n" ++
        "3. third\n" ++
        "4. fourth\n" ++
        "\n" ++
        " a │   │ c \n" ++
        "───┼───┼───\n" ++
        " 1 │   │ 3 \n" ++
        "\n" ++
        "one\n" ++
        "two\n" ++
        "three\n";
    try std.testing.expectEqualStrings(expected, s);
    // The triple marker is bold and italic at once, closed as one span.
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(alloc);
    try inlineSpan(alloc, &out, "***all three*** and *one*", th, 0);
    try std.testing.expect(std.mem.indexOf(u8, out.items, "\x1b[1m\x1b[3mall three\x1b[0m and \x1b[3mone\x1b[0m") != null);
    // Inline code keeps its style inside a heading.
    const rows = try render(alloc, "# Use `zig build`", 40, th);
    defer {
        for (rows) |r| alloc.free(r);
        alloc.free(rows);
    }
    try std.testing.expect(std.mem.indexOf(u8, rows[0], "\x1b[4mzig build\x1b[0m") != null);
}

/// The documents every pathological golden and the prefix fuzz are run on.
const pathological = struct {
    const nested_lists = blk: {
        @setEvalBranchQuota(10_000); // one branch per repeated space
        var text: []const u8 = "";
        for (0..64) |level| {
            text = text ++ repeat(" ", 2 * level) ++ "- x\n";
        }
        break :blk text;
    };
    const wide_table = blk: {
        var head: []const u8 = "|";
        var divider: []const u8 = "|";
        var body: []const u8 = "|";
        for (0..40) |_| {
            head = head ++ " c |";
            divider = divider ++ " --- |";
            body = body ++ " v |";
        }
        break :blk head ++ "\n" ++ divider ++ "\n" ++ body ++ "\n";
    };
    const unclosed_fence = "```zig\nconst x = 1;\n";
    const only_hash = "#\n# \n##\n";
    const escaped_link = "[x](http://a\x1bb) [y](http://ok)\n";
    const mixed_crlf = "a\r\nb\nc\r\n\r\n- d\r\n";
};

test "pathological input: a long word, deep nesting, an open fence, a wide table, bare hashes, an escaped link, mixed CRLF" {
    const alloc = std.testing.allocator;
    const th = theme_mod.Theme{ .kind = .plain };
    // A 10,000-character word breaks at the edge, 250 full rows, nothing lost.
    const long_word = try alloc.alloc(u8, 10_000);
    defer alloc.free(long_word);
    @memset(long_word, 'a');
    const word_rows = try render(alloc, long_word, 40, th);
    defer {
        for (word_rows) |r| alloc.free(r);
        alloc.free(word_rows);
    }
    try std.testing.expectEqual(@as(usize, 250), word_rows.len);
    for (word_rows) |r| try std.testing.expectEqual(@as(usize, 40), view.styledWidth(r));

    // 64 nesting levels: the indent stops at level three, the glyph cycles.
    const lists = try joined(alloc, pathological.nested_lists, 40, th);
    defer alloc.free(lists);
    var list_rows = std.mem.splitScalar(u8, std.mem.trimEnd(u8, lists, "\n"), '\n');
    var count: usize = 0;
    var last: []const u8 = "";
    while (list_rows.next()) |r| : (count += 1) last = r;
    try std.testing.expectEqual(@as(usize, 64), count);
    try std.testing.expectEqualStrings("      • x", last);

    // An unclosed fence at the end of the stream is code to the end.
    const fence = try joined(alloc, pathological.unclosed_fence, 40, th);
    defer alloc.free(fence);
    try std.testing.expect(std.mem.startsWith(u8, fence, " zig"));
    try std.testing.expect(std.mem.indexOf(u8, fence, "const x = 1;") != null);

    // Forty columns in sixty cells: the rows break at the edge, none wider.
    const table = try render(alloc, pathological.wide_table, 60, th);
    defer {
        for (table) |r| alloc.free(r);
        alloc.free(table);
    }
    try std.testing.expect(table.len >= 3 and table.len <= 9);
    for (table) |r| try std.testing.expect(view.styledWidth(r) <= 60);

    // A heading of only `#` is text, not an empty heading.
    const hashes = try joined(alloc, pathological.only_hash, 40, th);
    defer alloc.free(hashes);
    try std.testing.expectEqualStrings("#\n#\n##\n", hashes);

    // An escape in a link target never becomes a hyperlink; a clean one does.
    const links = try render(alloc, pathological.escaped_link, 60, th);
    defer {
        for (links) |r| alloc.free(r);
        alloc.free(links);
    }
    try std.testing.expect(std.mem.indexOf(u8, links[0], "\x1b]8;;http://a") == null);
    try std.testing.expect(std.mem.indexOf(u8, links[0], "\x1b]8;;http://ok\x1b\\") != null);
    try std.testing.expect(std.mem.indexOf(u8, links[0], "\x1bb") == null);

    // CRLF and LF lines mix: no `\r` survives, and the soft breaks hold.
    const crlf = try joined(alloc, pathological.mixed_crlf, 40, th);
    defer alloc.free(crlf);
    try std.testing.expectEqualStrings("a\nb\nc\n\n• d\n", crlf);
}

/// The fixture documents: every golden's source and the pathological set.
const fixtures = [_][]const u8{
    "# Heading\n\nA paragraph.\n\n- one\n- two\n",
    "## Title\n- plain\n- [x] done\n- [ ] todo\n1. first\n> quoted\n```zig\nx = y;\n```\n| a | bb |\n| --- | --- |\n| 1 | long cell |\n",
    "## Use `zig build`\n- run `make check` first\n- ***all three***\n> > deep\n3. third\n\n| a |  | c |\n| --- | --- | --- |\n| 1 |  | 3 |\n\none\ntwo\n",
    "**b** *i* ~~s~~ `c` [t](http://x) \x1b[31mred\x1b[0m a ** b\n\n---\n\n\t- tabbed\n",
    pathological.nested_lists,
    pathological.wide_table,
    pathological.unclosed_fence,
    pathological.only_hash,
    pathological.escaped_link,
    pathological.mixed_crlf,
};

/// Fails on a control byte outside an escape sequence: the one thing the
/// renderer must never let a model write to the terminal.
fn expectNoControlBytes(row: []const u8) !void {
    var i: usize = 0;
    while (i < row.len) {
        const b = row[i];
        if (b == 0x1b) {
            i += 1;
            if (i < row.len and row[i] == '[') {
                i += 1;
                while (i < row.len and row[i] >= 0x20 and row[i] <= 0x3f) i += 1;
                i += 1;
            } else if (i < row.len and row[i] == ']') {
                while (i < row.len and row[i] != 0x07 and !(row[i] == 0x1b and i + 1 < row.len and row[i + 1] == '\\')) i += 1;
                i += if (i < row.len and row[i] == 0x07) 1 else 2;
            } else i += 1;
            continue;
        }
        try std.testing.expect(b >= 0x20 and b != 0x7f);
        i += 1;
    }
}

test "every prefix of every fixture renders: no error, no control byte, bounded rows, no row past the edge" {
    // Thousands of renders: an arena, reset per prefix, keeps the run in
    // seconds; the goldens above hold the leak checks.
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const th = theme_mod.Theme{ .kind = .truecolor };
    const columns: usize = 32;
    for (fixtures) |document| {
        for (0..document.len + 1) |n| {
            _ = arena.reset(.retain_capacity);
            const alloc = arena.allocator();
            const prefix = document[0..n];
            const parts = split(prefix, .{});
            try std.testing.expectEqual(n, parts.closed.len + parts.open.len);
            for ([_][]const u8{ prefix, parts.closed }) |text| {
                const rows = try render(alloc, text, columns, th);
                try std.testing.expect(rows.len <= text.len + 2);
                for (rows) |r| {
                    try expectNoControlBytes(r);
                    try std.testing.expect(view.styledWidth(r) <= columns);
                }
            }
        }
    }
}

test "fuzz markdown: any text renders at any width with no control byte and no row past the edge" {
    const fuzz = @import("inference").fuzz;
    const Case = struct {
        /// The first byte picks the width (1 to 64), the colour level, and
        /// the glyph set; the rest is the document.
        fn one(_: void, smith: *fuzz.Smith) !void {
            var buf: [8192]u8 = undefined;
            const input = fuzz.bytes(smith, &buf);
            if (input.len == 0) return;
            const columns: usize = (input[0] & 0x3f) + 1;
            const kinds = std.enums.values(theme_mod.Kind);
            const th: theme_mod.Theme = .{ .kind = kinds[(input[0] >> 6) % kinds.len], .glyph_set = if (input[0] & 0x80 != 0) .ascii else .unicode };
            const text = input[1..];
            const parts = split(text, .{});
            try std.testing.expectEqual(text.len, parts.closed.len + parts.open.len);
            var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
            defer arena.deinit();
            const rows = try render(arena.allocator(), text, columns, th);
            for (rows) |r| {
                try expectNoControlBytes(r);
                try std.testing.expect(view.styledWidth(r) <= columns);
            }
        }
    };
    try std.testing.fuzz({}, Case.one, .{ .corpus = fuzz.corpus(&(.{
        "\x00| a | b |\n|:-|-:|\n| 1 | 2 | 3 |\n| x |\n",
        "\x05> > > - [ ] **_deep_** `code`\n>>>>>>>> 1. x\n```zig\nopen fence",
        "\x80# h\x1b[31m\t\xc2\x9b\x07\r\n[link](http://x\x00)\\*\n\n---\n",
        "\x3f" ++ fixtures[0],
    })) });
}
