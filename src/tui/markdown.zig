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
//! Streaming. A turn is rendered while it arrives, and the last
//! block of a partial document is still being written: a paragraph may gain
//! words, a fenced block its closing fence, a table another row. `split`
//! draws that line — everything before it can no longer change and is
//! rendered; the rest is shown as raw text until it closes. The agent
//! repaints with both on every token, so styling appears block by block
//! instead of at the end of the turn.
//!
//! A code block carries **no box glyphs**: its rows are the code padded to
//! the block background, and the language is a caption row above them, so
//! selecting a block in the terminal copies the code and nothing else.
const std = @import("std");
const builtin = @import("builtin");
const theme_mod = @import("theme.zig");
const view = @import("view.zig");
const highlight = @import("highlight.zig");

/// A partial document: the blocks that can no longer change, and the one
/// still being written.
pub const Split = struct { closed: []const u8, open: []const u8 };

/// Splits a streaming turn at the last block boundary. The rules are the
/// renderer's own: a blank line ends a block; a heading, rule, quote, list
/// item, or closing fence ends one at its newline; a paragraph or a table
/// keeps its block open until one of those arrives, and everything from a
/// fence opener on stays open until the fence closes. The last line is never
/// closed, because the newline that would end it has not arrived.
pub fn split(text: []const u8) Split {
    var in_fence = false;
    var closed: usize = 0;
    var offset: usize = 0;
    while (std.mem.indexOfScalarPos(u8, text, offset, '\n')) |newline| {
        const line = std.mem.trim(u8, text[offset..newline], " \t\r");
        const next = newline + 1;
        if (in_fence) {
            if (std.mem.startsWith(u8, line, "```")) {
                in_fence = false;
                closed = next;
            }
        } else if (std.mem.startsWith(u8, line, "```")) {
            in_fence = true; // the block runs until the closing fence
        } else if (line.len == 0 or !continues(line)) {
            closed = next;
        }
        offset = next;
    }
    return .{ .closed = text[0..closed], .open = text[closed..] };
}

/// Whether a line leaves its block open for the next one: paragraph text
/// accumulates until a blank line, and table rows accumulate into a grid.
fn continues(line: []const u8) bool {
    if (line[0] == '|') return true;
    if (headingBody(line) != null or isRule(line) or quoteBody(line) != null) return false;
    return listMarker(line, theme_mod.unicode_glyphs) == null;
}

/// Test instrumentation: how many times `render` ran, so a streaming test
/// can prove the closed part of a turn is rendered once and the open tail
/// never through here.
pub var render_calls: usize = 0;

pub fn render(alloc: std.mem.Allocator, text: []const u8, columns: usize, th: theme_mod.Theme) ![][]const u8 {
    if (builtin.is_test) render_calls += 1;
    var rows: std.ArrayList([]const u8) = .empty;
    errdefer rows.deinit(alloc);
    var para: std.ArrayList(u8) = .empty;
    defer para.deinit(alloc);
    var table: std.ArrayList([]const u8) = .empty;
    defer table.deinit(alloc);
    if (text.len == 0) return rows.toOwnedSlice(alloc);
    var in_code = false;
    var hl_state = highlight.State{};
    var hl_lang = highlight.Language.generic;
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
            try putSanitized(alloc, &code_text, raw_line);
            // Pad to the full width so the code-block background spans it;
            // wrapStyled persists the background across soft wraps.
            const styled_code = try highlight.line(alloc, code_text.items, hl_lang, &hl_state, th);
            defer alloc.free(styled_code);
            const pad = (columns -| 1) -| view.width(code_text.items);
            const filler = try fill(alloc, " ", pad);
            defer alloc.free(filler);
            var styled: std.ArrayList(u8) = .empty;
            defer styled.deinit(alloc);
            try styled.appendSlice(alloc, th.paint(.code_block));
            try styled.appendSlice(alloc, styled_code);
            try styled.appendSlice(alloc, filler);
            try styled.appendSlice(alloc, theme_mod.reset);
            try appendWrapped(alloc, &rows, styled.items, columns);
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
            try flushPara(alloc, &para, &rows, columns, th);
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
    return rows.toOwnedSlice(alloc);
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

/// The caption above a fenced block: the language name over the block's own
/// background, dimmed, padded so the strip spans the block's width.
fn captionRow(alloc: std.mem.Allocator, info: []const u8, columns: usize, th: theme_mod.Theme) ![]const u8 {
    var styled: std.ArrayList(u8) = .empty;
    errdefer styled.deinit(alloc);
    try styled.appendSlice(alloc, th.paint(.code_block));
    try styled.appendSlice(alloc, th.paint(.comment));
    try styled.append(alloc, ' ');
    try putSanitized(alloc, &styled, info);
    const pad = (columns -| 1) -| (view.width(info) + 1);
    const filler = try fill(alloc, " ", pad);
    defer alloc.free(filler);
    try styled.appendSlice(alloc, filler);
    try styled.appendSlice(alloc, theme_mod.reset);
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
    for (table.items, 0..) |l, index| {
        if (index == 1) continue; // divider row
        try grid.append(alloc, try splitCells(alloc, l));
    }
    if (grid.items.len == 0) return;
    const ncols = grid.items[0].len;
    // The divider row carries the alignment of every column (`:---`, `---:`,
    // `:---:`); a column it does not mention is left-aligned.
    const alignments = try splitCells(alloc, table.items[1]);
    defer alloc.free(alignments);
    var widths = try alloc.alloc(usize, ncols);
    defer alloc.free(widths);
    @memset(widths, 1);
    for (grid.items) |row_cells| {
        for (row_cells, 0..) |cell, i| {
            if (i < ncols) widths[i] = @max(widths[i], view.width(cell));
        }
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
            const w = if (i < ncols) widths[i] else widths[ncols - 1];
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
            if (linkEnd(text, i)) |end| {
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
        try putSanitizedByte(alloc, out, b);
        i += 1;
    }
}

const LinkEnd = struct { label_close: usize, target_start: usize, target_end: usize, consumed: usize };

/// `[label](target)` starting at `open`. The label is rendered; the target is
/// kept so an http(s) one can become an OSC 8 hyperlink.
fn linkEnd(text: []const u8, open: usize) ?LinkEnd {
    const label_close = std.mem.indexOfScalarPos(u8, text, open + 1, ']') orelse return null;
    if (label_close + 1 >= text.len or text[label_close + 1] != '(') return null;
    const target_close = std.mem.indexOfScalarPos(u8, text, label_close + 2, ')') orelse return null;
    return .{ .label_close = label_close, .target_start = label_close + 2, .target_end = target_close, .consumed = target_close + 1 };
}

/// Whether a link target is safe to emit as an OSC 8 hyperlink: an http(s)
/// URL, bounded, with no control bytes that could terminate the sequence early
/// or inject another escape. Everything else renders as styled label text.
fn isHyperlink(target: []const u8) bool {
    if (target.len == 0 or target.len > 2048) return false;
    if (!std.ascii.startsWithIgnoreCase(target, "http://") and
        !std.ascii.startsWithIgnoreCase(target, "https://")) return false;
    for (target) |c| {
        if (c < 0x20 or c == 0x7f) return false;
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
    for (text) |b| try putSanitizedByte(alloc, out, b);
}

/// Every control byte but the line break is dropped: the break is the one
/// the paragraph kept on purpose, and the wrapper turns it into a row.
fn putSanitizedByte(alloc: std.mem.Allocator, out: *std.ArrayList(u8), b: u8) !void {
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

test "the streaming split closes blocks that can no longer change" {
    // A paragraph stays open until a blank line ends it.
    const partial = split("done.\n\nstill writ");
    try std.testing.expectEqualStrings("done.\n\n", partial.closed);
    try std.testing.expectEqualStrings("still writ", partial.open);
    // Nothing at all is closed before the first boundary.
    const first = split("one line so far");
    try std.testing.expectEqualStrings("", first.closed);
    try std.testing.expectEqualStrings("one line so far", first.open);
    // A fence holds its block open until the closing fence arrives...
    const fence = split("intro\n\n```zig\nconst x = 1;\n");
    try std.testing.expectEqualStrings("intro\n\n", fence.closed);
    const finished = split("intro\n\n```zig\nconst x = 1;\n```\nafter");
    try std.testing.expectEqualStrings("intro\n\n```zig\nconst x = 1;\n```\n", finished.closed);
    try std.testing.expectEqualStrings("after", finished.open);
    // ...while single-line blocks close at their newline, and a table keeps
    // accumulating rows.
    try std.testing.expectEqualStrings("# Title\n", split("# Title\nnext").closed);
    try std.testing.expectEqualStrings("- one\n", split("- one\nnext").closed);
    try std.testing.expectEqualStrings("", split("| a |\n| - |\n").closed);
    // The closed prefix plus the open tail is always the whole text.
    const text = "a\n\n- b\n| c |\n";
    const both = split(text);
    try std.testing.expectEqual(text.len, both.closed.len + both.open.len);
}

test "a streamed document renders the same as the finished one, block by block" {
    const alloc = std.testing.allocator;
    const th = theme_mod.Theme{ .kind = .plain };
    const document = "# Heading\n\nA paragraph.\n\n- one\n- two\n";
    const whole = try joined(alloc, document, 40, th);
    defer alloc.free(whole);
    // Feeding the document one byte at a time, the closed prefix only ever
    // grows and never renders differently from the same prefix of the whole.
    var seen: usize = 0;
    for (1..document.len + 1) |n| {
        const part = split(document[0..n]);
        try std.testing.expect(part.closed.len >= seen);
        seen = part.closed.len;
        const rendered = try joined(alloc, part.closed, 40, th);
        defer alloc.free(rendered);
        try std.testing.expect(std.mem.startsWith(u8, whole, rendered));
    }
    try std.testing.expectEqual(document.len, seen);
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
        var text: []const u8 = "";
        for (0..64) |level| {
            text = text ++ " " ** (2 * level) ++ "- x\n";
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
            const parts = split(prefix);
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
