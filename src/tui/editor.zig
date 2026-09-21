//! The prompt editor: a flat UTF-8 buffer, a cursor, paste chips, and prompt
//! history — everything needed to compose a turn, and nothing about models,
//! tokens, or files. Rows wrap at the last space that fits; a scrolled view
//! keeps the cursor's row and marks hidden ones `… N lines above`.
//!
//! A paste longer than a few lines is not inserted as text: the bytes go into
//! the buffer and a `Chip` records the range, rendering as
//! `[pasted 96 lines, 6.1 KB]`. The chip is a view over the buffer, never a
//! copy — what is sent is the text itself. Cursor motion steps over a chip,
//! Backspace at its edge deletes it whole, Ctrl-E expands it, and any edit
//! inside it expands it too. Up/Down move within a multi-line input and reach
//! history only from the first or last row.
//!
//! The editor owns its buffer, chips, draft, and history through one
//! allocator; `layout` borrows a short-lived allocator for the rows it
//! returns, which die with the frame. `handleKey` returns an `Action` rather
//! than mutating shared UI state, so a test drives it with keys and asserts
//! on `layout`. See docs/agent-spec.md § Editor for the design.
const std = @import("std");
const theme = @import("theme.zig");
const screen = @import("screen.zig");
const keys = @import("keys.zig");
const view = @import("view.zig");
const graphemes = @import("graphemes.zig");

const Allocator = std.mem.Allocator;

/// The editor refuses input past this; a prompt is a prompt, not a file.
pub const max_input = 128 * 1024;
/// Submitted prompts kept for recall.
pub const max_history = 200;
/// A paste at or above either bound becomes a chip instead of visible text.
pub const chip_min_lines = 4;
pub const chip_min_bytes = 400;

/// A pasted range of the buffer, shown as one token. `start`/`end` are byte
/// offsets into `buffer` and move with every edit before them.
pub const Chip = struct {
    start: usize,
    end: usize,
    /// Line count of the pasted text, for the label (counted once, at paste
    /// time: the bytes never change while the chip exists).
    lines: usize,

    pub fn contains(self: Chip, offset: usize) bool {
        return offset > self.start and offset < self.end;
    }
};

/// What a key did. `ignored` means the editor does not own that key and the
/// caller should decide (the agent's Ctrl-T, Ctrl-W, Tab, quit keys).
pub const Action = enum { none, ignored, submit };

pub const Layout = struct {
    rows: []const screen.Row,
    /// Index into `rows` of the row holding the insertion point.
    cursor_row: usize,
    /// Display column of the insertion point, 0-based inside the text area.
    cursor_col: usize,
};

pub const Options = struct {
    /// Display cells available for text, excluding the two-cell prompt.
    width: usize,
    /// The idle height of the box, padded with empty rows.
    min_rows: usize,
    max_rows: usize,
    theme: theme.Theme,
};

/// How the box is framed. `width` is the frame's outer width; the text area
/// inside is `width - frame_cells`, which is what `layout` must have been
/// given.
pub const Frame = struct {
    width: usize,
    theme: theme.Theme,
    /// The frame's colour: the effort ramp, chosen by the caller.
    style: theme.Style = .frame_off,
    /// Drawn in the top edge after the corner while a turn runs; empty when
    /// idle.
    spinner: []const u8 = "",
};

/// Cells the frame and the prompt take from a row: the two edges, a space
/// inside each, and the two-cell prompt.
pub const frame_cells: usize = 6;
/// Cells before the first text cell of a framed row: edge, space, prompt.
pub const frame_lead: usize = 4;

/// The framed box: the top edge, every layout row between two edges, and the
/// bottom edge. Rows are raw and carry the editor background inside the
/// frame; the caller places the cursor `frame_lead` cells in on row
/// `1 + cursor_row`.
pub fn frame(a: Allocator, layout: Layout, options: Frame) ![]const screen.Row {
    const th = options.theme;
    const gl = th.glyphs();
    var rows: std.ArrayList(screen.Row) = .empty;
    const inner = options.width -| 2;
    var top: std.Io.Writer.Allocating = .init(a);
    try top.writer.print("{s}{s}", .{ th.paint(options.style), gl.box_tl });
    var drawn: usize = 0;
    if (options.spinner.len > 0 and inner >= 3) {
        try top.writer.print("{s}{s}{s}", .{ gl.box_h, options.spinner, gl.box_h });
        drawn = 3;
    }
    for (drawn..inner) |_| try top.writer.writeAll(gl.box_h);
    try top.writer.print("{s}{s}", .{ gl.box_tr, theme.reset });
    try rows.append(a, .{ .text = top.written(), .raw = true });
    for (layout.rows) |row| {
        var w: std.Io.Writer.Allocating = .init(a);
        try w.writer.print("{s}{s}{s}{s} {s}{s}{s}{s} {s}{s}{s}", .{
            th.paint(options.style), gl.box_v,             theme.reset,
            th.paint(.editor_bg),    row.prefix,           row.text,
            theme.reset,             th.paint(.editor_bg), theme.reset,
            th.paint(options.style), gl.box_v,
        });
        try w.writer.writeAll(theme.reset);
        try rows.append(a, .{ .text = w.written(), .raw = true });
    }
    var bottom: std.Io.Writer.Allocating = .init(a);
    try bottom.writer.print("{s}{s}", .{ th.paint(options.style), gl.box_bl });
    for (0..inner) |_| try bottom.writer.writeAll(gl.box_h);
    try bottom.writer.print("{s}{s}", .{ gl.box_br, theme.reset });
    try rows.append(a, .{ .text = bottom.written(), .raw = true });
    return rows.items;
}

/// A row of the wrapped buffer, as byte offsets. `end` excludes the newline
/// or the space a wrap broke on, so rendering a span never shows it.
const Span = struct { start: usize, end: usize };

pub const Editor = struct {
    alloc: Allocator,
    buffer: std.ArrayList(u8) = .empty,
    /// Insertion point, always on a UTF-8 code point boundary and never
    /// strictly inside a chip.
    cursor: usize = 0,
    chips: std.ArrayList(Chip) = .empty,
    /// Bytes of a bracketed paste in progress; committed by `endPaste`.
    pending: std.ArrayList(u8) = .empty,
    pasting: bool = false,
    /// Submitted prompts, oldest first; `index` is null when not browsing
    /// and `draft` holds the input the first recall displaced.
    history: std.ArrayList([]const u8) = .empty,
    index: ?usize = null,
    draft: std.ArrayList(u8) = .empty,
    /// First visible row of the wrapped buffer.
    scroll: usize = 0,
    /// Text width of the last `layout`, so vertical motion knows where the
    /// rows break. Zero before the first layout: Up/Down are history then.
    last_width: usize = 0,

    pub fn deinit(self: *Editor) void {
        self.buffer.deinit(self.alloc);
        self.chips.deinit(self.alloc);
        self.pending.deinit(self.alloc);
        for (self.history.items) |item| self.alloc.free(item);
        self.history.deinit(self.alloc);
        self.draft.deinit(self.alloc);
    }

    /// The text a turn would send: the buffer as typed, chips expanded,
    /// because a chip was only ever a way to look at it.
    pub fn text(self: *const Editor) []const u8 {
        return self.buffer.items;
    }

    pub fn isEmpty(self: *const Editor) bool {
        return self.buffer.items.len == 0;
    }

    pub fn clear(self: *Editor) void {
        self.buffer.clearRetainingCapacity();
        self.chips.clearRetainingCapacity();
        self.cursor = 0;
        self.index = null;
        self.scroll = 0;
    }

    // ----- keys -----

    pub fn handleKey(self: *Editor, key: keys.Key) !Action {
        // A bracketed paste is content: nothing inside it is a command, and
        // the bytes accumulate until the end marker so the whole paste can
        // be weighed at once.
        switch (key) {
            .paste_begin => {
                self.pasting = true;
                self.pending.clearRetainingCapacity();
                return .none;
            },
            .paste_end => {
                try self.endPaste();
                return .none;
            },
            else => {},
        }
        if (self.pasting) {
            switch (key) {
                .text => |t| try self.pending.appendSlice(self.alloc, t),
                .enter, .newline => try self.pending.append(self.alloc, '\n'),
                // No tab width here; four spaces keep the cursor honest.
                .tab => try self.pending.appendSlice(self.alloc, "    "),
                else => {},
            }
            return .none;
        }
        switch (key) {
            .text => |t| try self.insert(t),
            .enter => return .submit,
            .newline => try self.insert("\n"),
            .backspace => try self.backspace(),
            .delete => try self.deleteForward(),
            .left => self.cursor = self.prev(self.cursor),
            .right => self.cursor = self.next(self.cursor),
            .home => self.cursor = 0,
            .end => self.cursor = self.buffer.items.len,
            .up => try self.vertical(.up),
            .down => try self.vertical(.down),
            .ctrl => |c| switch (c) {
                'j' => try self.insert("\n"),
                'e' => self.expandChip(),
                else => return .ignored,
            },
            else => return .ignored,
        }
        return .none;
    }

    // ----- editing -----

    /// Inserts typed text at the cursor. Input past the limit is refused
    /// whole rather than truncated: half a paste is worse than none.
    pub fn insert(self: *Editor, t: []const u8) !void {
        if (t.len == 0 or self.buffer.items.len + t.len > max_input) return;
        self.expandAt(self.cursor);
        try self.buffer.insertSlice(self.alloc, self.cursor, t);
        self.shift(self.cursor, @intCast(t.len));
        self.cursor += t.len;
        self.index = null;
    }

    /// Commits a bracketed paste: a long one becomes a chip over the bytes,
    /// a short one is ordinary typed text.
    fn endPaste(self: *Editor) !void {
        self.pasting = false;
        const pasted = self.pending.items;
        if (pasted.len == 0 or self.buffer.items.len + pasted.len > max_input) {
            self.pending.clearRetainingCapacity();
            return;
        }
        const line_count = std.mem.count(u8, pasted, "\n") + 1;
        const chip = line_count >= chip_min_lines or pasted.len >= chip_min_bytes;
        const start = self.cursor;
        self.expandAt(start);
        try self.buffer.insertSlice(self.alloc, start, pasted);
        self.shift(start, @intCast(pasted.len));
        if (chip) {
            try self.chips.append(self.alloc, .{ .start = start, .end = start + pasted.len, .lines = line_count });
            std.mem.sort(Chip, self.chips.items, {}, lessThan);
        }
        self.cursor = start + pasted.len;
        self.index = null;
        self.pending.clearRetainingCapacity();
    }

    fn lessThan(_: void, a: Chip, b: Chip) bool {
        return a.start < b.start;
    }

    fn backspace(self: *Editor) !void {
        if (self.chipEndingAt(self.cursor)) |i| return self.removeChip(i);
        const start = self.prev(self.cursor);
        if (start == self.cursor) return;
        try self.remove(start, self.cursor);
        self.cursor = start;
    }

    fn deleteForward(self: *Editor) !void {
        if (self.chipStartingAt(self.cursor)) |i| return self.removeChip(i);
        const end = self.next(self.cursor);
        if (end == self.cursor) return;
        try self.remove(self.cursor, end);
    }

    fn removeChip(self: *Editor, i: usize) !void {
        const chip = self.chips.items[i];
        _ = self.chips.orderedRemove(i);
        try self.remove(chip.start, chip.end);
        self.cursor = chip.start;
    }

    /// Deletes `[from, to)` and moves every chip that outlives it.
    fn remove(self: *Editor, from: usize, to: usize) !void {
        try self.buffer.replaceRange(self.alloc, from, to - from, "");
        const len: isize = @intCast(to - from);
        var i: usize = 0;
        while (i < self.chips.items.len) {
            const chip = &self.chips.items[i];
            // A chip the deletion touched at all is no longer what was
            // pasted: it becomes ordinary text.
            if (chip.start < to and from < chip.end) {
                _ = self.chips.orderedRemove(i);
                continue;
            }
            if (chip.start >= to) {
                chip.start = @intCast(@as(isize, @intCast(chip.start)) - len);
                chip.end = @intCast(@as(isize, @intCast(chip.end)) - len);
            }
            i += 1;
        }
        self.index = null;
    }

    fn shift(self: *Editor, at: usize, len: isize) void {
        for (self.chips.items) |*chip| {
            if (chip.start >= at) {
                chip.start = @intCast(@as(isize, @intCast(chip.start)) + len);
                chip.end = @intCast(@as(isize, @intCast(chip.end)) + len);
            }
        }
    }

    /// Ctrl-E on a chip: drop the range so the pasted text is editable.
    fn expandChip(self: *Editor) void {
        if (self.chipStartingAt(self.cursor) orelse self.chipEndingAt(self.cursor)) |i| {
            _ = self.chips.orderedRemove(i);
        }
    }

    /// Any edit strictly inside a chip expands it first.
    fn expandAt(self: *Editor, offset: usize) void {
        for (self.chips.items, 0..) |chip, i| {
            if (chip.contains(offset)) {
                _ = self.chips.orderedRemove(i);
                return;
            }
        }
    }

    fn chipStartingAt(self: *const Editor, offset: usize) ?usize {
        for (self.chips.items, 0..) |chip, i| if (chip.start == offset) return i;
        return null;
    }
    fn chipEndingAt(self: *const Editor, offset: usize) ?usize {
        for (self.chips.items, 0..) |chip, i| if (chip.end == offset) return i;
        return null;
    }
    fn chipAt(self: *const Editor, offset: usize) ?Chip {
        for (self.chips.items) |chip| if (chip.start == offset) return chip;
        return null;
    }

    /// Byte offset of the grapheme cluster before `from`, stepping over a chip.
    fn prev(self: *const Editor, from: usize) usize {
        if (self.chipEndingAt(from)) |i| return self.chips.items[i].start;
        return graphemes.clusterPrev(self.buffer.items, from);
    }
    /// Byte offset of the grapheme cluster after `from`, stepping over a chip.
    fn next(self: *const Editor, from: usize) usize {
        if (self.chipStartingAt(from)) |i| return self.chips.items[i].end;
        return graphemes.clusterNext(self.buffer.items, from);
    }

    // ----- completion -----

    /// The word the cursor is at the end of: everything back to the last
    /// space or newline. Empty when the cursor follows one of those, or when
    /// it sits inside a paste chip, where a completion would be meaningless.
    pub fn wordBefore(self: *const Editor) []const u8 {
        for (self.chips.items) |chip| {
            if (chip.contains(self.cursor) or chip.end == self.cursor) return "";
        }
        return self.buffer.items[self.wordStart()..self.cursor];
    }

    fn wordStart(self: *const Editor) usize {
        var start = self.cursor;
        while (start > 0) {
            const c = self.buffer.items[start - 1];
            if (c == ' ' or c == '\t' or c == '\n') break;
            start -= 1;
        }
        return start;
    }

    /// Whether the word under the cursor is the first thing in the buffer,
    /// which is the only place a slash command can be.
    pub fn atFirstWord(self: *const Editor) bool {
        return self.wordStart() == 0;
    }

    /// Replaces that word with `replacement` (accepting a completion). The
    /// cursor lands after it, as if the text had been typed.
    pub fn replaceWord(self: *Editor, replacement: []const u8) !void {
        const start = self.wordStart();
        if (start < self.cursor) {
            try self.remove(start, self.cursor);
            self.cursor = start;
        }
        try self.insert(replacement);
    }

    // ----- history -----

    /// Archives a submitted prompt. Duplicates of the newest entry are not
    /// repeated, so Up is not full of the same line.
    pub fn remember(self: *Editor, t: []const u8) !void {
        if (t.len == 0) return;
        if (self.history.items.len > 0 and std.mem.eql(u8, self.history.items[self.history.items.len - 1], t)) {
            self.index = null;
            return;
        }
        const copy = try self.alloc.dupe(u8, t);
        errdefer self.alloc.free(copy);
        try self.history.append(self.alloc, copy);
        self.index = null;
        if (self.history.items.len > max_history) self.alloc.free(self.history.orderedRemove(0));
    }

    /// Seeds the history from storage (oldest first), as `src/agent/` loads
    /// it at startup. Takes ownership of nothing: every entry is copied.
    pub fn seedHistory(self: *Editor, items: []const []const u8) !void {
        for (items) |item| try self.remember(item);
    }

    pub fn historyItems(self: *const Editor) []const []const u8 {
        return self.history.items;
    }

    /// Up from the first row and Down from the last reach the history;
    /// anywhere else they move the cursor by one display row.
    fn vertical(self: *Editor, direction: enum { up, down }) !void {
        if (self.last_width > 0) {
            var arena = std.heap.ArenaAllocator.init(self.alloc);
            defer arena.deinit();
            const spans = try self.wrap(arena.allocator(), self.last_width);
            const row = self.rowOf(spans, self.cursor);
            const target: ?usize = switch (direction) {
                .up => if (row > 0) row - 1 else null,
                .down => if (row + 1 < spans.len) row + 1 else null,
            };
            if (target) |t| {
                const column = self.columnIn(spans[row], self.cursor);
                self.cursor = self.offsetIn(spans[t], column);
                return;
            }
        }
        try self.recall(if (direction == .up) .older else .newer);
    }

    /// Recalls the previous or next submitted prompt. The input is saved as
    /// a draft before the first recall and restored past the newest.
    fn recall(self: *Editor, which: enum { older, newer }) !void {
        if (self.history.items.len == 0) return;
        const idx = self.index orelse blk: {
            if (which == .newer) return;
            self.draft.clearRetainingCapacity();
            try self.draft.appendSlice(self.alloc, self.buffer.items);
            break :blk self.history.items.len;
        };
        const target: ?usize = switch (which) {
            .older => if (idx == 0) null else idx - 1,
            .newer => if (idx + 1 < self.history.items.len) idx + 1 else null,
        };
        if (which == .older and target == null) return; // already at the oldest
        self.index = target;
        const t = if (target) |i| self.history.items[i] else self.draft.items;
        self.buffer.clearRetainingCapacity();
        self.chips.clearRetainingCapacity();
        try self.buffer.appendSlice(self.alloc, t);
        self.cursor = self.buffer.items.len;
        self.scroll = 0;
    }

    // ----- layout -----

    /// Wraps the buffer into rows, breaking at the last space that fits and
    /// treating a chip as one unbreakable token. Returns byte spans, so the
    /// cost of laying out a large buffer is an array of offsets.
    fn wrap(self: *const Editor, a: Allocator, width: usize) ![]const Span {
        var spans: std.ArrayList(Span) = .empty;
        const items = self.buffer.items;
        const columns = @max(width, 1);
        var start: usize = 0;
        var offset: usize = 0;
        var used: usize = 0;
        // Byte offset just after the last space seen on this row, and the
        // width of everything after it: where a word wrap breaks.
        var break_at: ?usize = null;
        var since_break: usize = 0;
        while (offset < items.len) {
            if (self.chipAt(offset)) |chip| {
                const w = chipWidth(chip);
                if (used + w > columns and used > 0) {
                    try spans.append(a, .{ .start = start, .end = breakEnd(items, break_at, offset) });
                    start = break_at orelse offset;
                    used = if (break_at != null) since_break else 0;
                    break_at = null;
                    since_break = 0;
                }
                used += w;
                since_break += w;
                offset = chip.end;
                continue;
            }
            const end = graphemes.clusterNext(items, offset);
            const cluster = items[offset..end];
            const w = graphemes.clusterWidth(cluster);
            if (cluster.len == 1 and cluster[0] == '\n') {
                try spans.append(a, .{ .start = start, .end = offset });
                offset = end;
                start = offset;
                used = 0;
                break_at = null;
                since_break = 0;
                continue;
            }
            const is_space = cluster.len == 1 and cluster[0] == ' ';
            // A space past the edge is never drawn there: it becomes the next
            // break point instead of breaking the row, so a word that ends
            // exactly at the last column keeps it (same rule as `view.lines`).
            if (used + w > columns and used > 0 and !is_space) {
                try spans.append(a, .{ .start = start, .end = breakEnd(items, break_at, offset) });
                start = break_at orelse offset;
                used = if (break_at != null) since_break else 0;
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
        try spans.append(a, .{ .start = start, .end = items.len });
        return spans.toOwnedSlice(a);
    }

    /// Where a row ends: before the space a word wrap broke on, so the
    /// trailing space is not drawn at the edge of the box.
    fn breakEnd(items: []const u8, break_at: ?usize, offset: usize) usize {
        const at = break_at orelse return offset;
        return if (at > 0 and items[at - 1] == ' ') at - 1 else at;
    }

    fn chipWidth(chip: Chip) usize {
        var buffer: [48]u8 = undefined;
        return view.width(chipLabel(&buffer, chip));
    }

    fn chipLabel(buffer: []u8, chip: Chip) []const u8 {
        const bytes = chip.end - chip.start;
        if (bytes < 1024) return std.fmt.bufPrint(buffer, "[pasted {d} lines, {d} B]", .{ chip.lines, bytes }) catch "[pasted]";
        const kb = @as(f64, @floatFromInt(bytes)) / 1024.0;
        return std.fmt.bufPrint(buffer, "[pasted {d} lines, {d:.1} KB]", .{ chip.lines, kb }) catch "[pasted]";
    }

    fn rowOf(self: *const Editor, spans: []const Span, offset: usize) usize {
        _ = self;
        for (spans, 0..) |span, i| {
            if (offset <= span.end) return i;
            // A wrap break swallows the space between the rows: an offset
            // inside that gap belongs to the row that follows.
            if (i + 1 < spans.len and offset < spans[i + 1].start) return i + 1;
        }
        return spans.len -| 1;
    }

    /// Display column of `offset` inside a row.
    fn columnIn(self: *const Editor, span: Span, offset: usize) usize {
        var column: usize = 0;
        var at = span.start;
        const items = self.buffer.items;
        while (at < @min(offset, span.end)) {
            if (self.chipAt(at)) |chip| {
                column += chipWidth(chip);
                at = chip.end;
                continue;
            }
            const end = graphemes.clusterNext(items, at);
            column += graphemes.clusterWidth(items[at..end]);
            at = end;
        }
        return column;
    }

    /// The offset in a row nearest to a display column: vertical motion
    /// keeps the column it started from, clamped to the row's end.
    fn offsetIn(self: *const Editor, span: Span, column: usize) usize {
        var used: usize = 0;
        var at = span.start;
        const items = self.buffer.items;
        while (at < span.end) {
            if (self.chipAt(at)) |chip| {
                if (used + chipWidth(chip) > column) return at;
                used += chipWidth(chip);
                at = chip.end;
                continue;
            }
            const end = graphemes.clusterNext(items, at);
            const w = graphemes.clusterWidth(items[at..end]);
            if (used + w > column) return at;
            used += w;
            at = end;
        }
        return span.end;
    }

    /// The rows to paint, the cursor's place among them, and the scroll
    /// indicator when the buffer is taller than the box. Rows are padded to
    /// the box width and carry the editor background, so the caller appends
    /// them to the live region unchanged.
    pub fn layout(self: *Editor, a: Allocator, options: Options) !Layout {
        self.last_width = options.width;
        const spans = try self.wrap(a, options.width);
        const cursor_row = self.rowOf(spans, self.cursor);
        const capacity = @max(options.max_rows, 1);
        // Keep the cursor's row visible. A scrolled view spends one row on
        // the indicator, so the content it can show is one less than the
        // box — which is exactly the row the cursor would otherwise fall
        // off the bottom of.
        if (spans.len <= capacity or (cursor_row < capacity and self.scroll == 0)) {
            self.scroll = 0;
        } else {
            const content = capacity - 1;
            if (self.scroll == 0) self.scroll = cursor_row + 1 - content;
            if (cursor_row < self.scroll) self.scroll = cursor_row;
            if (cursor_row >= self.scroll + content) self.scroll = cursor_row + 1 - content;
            if (self.scroll + content > spans.len) self.scroll = spans.len - content;
        }
        // A scrolled view spends its first row on the indicator.
        const hidden = self.scroll;
        const content_rows = if (hidden > 0) capacity - 1 else capacity;
        const shown = @min(spans.len - hidden, content_rows);

        var rows: std.ArrayList(screen.Row) = .empty;
        if (hidden > 0) {
            const label = try std.fmt.allocPrint(a, "{s} {d} lines above", .{ options.theme.glyphs().ellipsis, hidden });
            try rows.append(a, try self.styledRow(a, label, options, .dim, "  "));
        }
        var i: usize = 0;
        while (i < shown) : (i += 1) {
            const span = spans[hidden + i];
            const rendered = try self.renderSpan(a, span, options.theme);
            const prefix: []const u8 = if (rows.items.len == 0) "> " else "  ";
            try rows.append(a, try self.styledRow(a, rendered, options, null, prefix));
        }
        while (rows.items.len < options.min_rows) {
            try rows.append(a, try self.styledRow(a, "", options, null, "  "));
        }
        return .{
            .rows = rows.items,
            .cursor_row = (cursor_row - hidden) + @intFromBool(hidden > 0),
            .cursor_col = self.columnIn(spans[cursor_row], self.cursor),
        };
    }

    /// One box row: pre-styled text padded to the box width, so the editor
    /// background spans it exactly.
    fn styledRow(self: *const Editor, a: Allocator, t: []const u8, options: Options, style: ?theme.Style, prefix: []const u8) !screen.Row {
        _ = self;
        var w: std.Io.Writer.Allocating = .init(a);
        if (style) |s| try w.writer.writeAll(options.theme.paint(s));
        try w.writer.writeAll(t);
        if (style != null) try w.writer.writeAll(theme.reset);
        const pad = options.width -| view.styledWidth(w.written());
        try w.writer.splatByteAll(' ', pad);
        return .{ .text = w.written(), .raw = true, .editor = true, .prefix = prefix };
    }

    /// Renders one row: the buffer's bytes with control characters removed,
    /// and every chip replaced by its label in the chip style.
    fn renderSpan(self: *const Editor, a: Allocator, span: Span, th: theme.Theme) ![]const u8 {
        var w: std.Io.Writer.Allocating = .init(a);
        var at = span.start;
        var plain = span.start;
        while (at < span.end) {
            if (self.chipAt(at)) |chip| {
                try view.safe(&w.writer, self.buffer.items[plain..at]);
                var buffer: [48]u8 = undefined;
                try w.writer.writeAll(th.paint(.chip));
                try w.writer.writeAll(chipLabel(&buffer, chip));
                try w.writer.writeAll(theme.reset);
                at = @min(chip.end, span.end);
                plain = at;
                continue;
            }
            at += @min(std.unicode.utf8ByteSequenceLength(self.buffer.items[at]) catch 1, span.end - at);
        }
        try view.safe(&w.writer, self.buffer.items[plain..span.end]);
        return w.written();
    }
};

// ----- tests -----

const testing = std.testing;

fn editor() Editor {
    return .{ .alloc = testing.allocator };
}

fn typeText(e: *Editor, t: []const u8) !void {
    try e.insert(t);
}

fn paste(e: *Editor, t: []const u8) !void {
    _ = try e.handleKey(.paste_begin);
    _ = try e.handleKey(.{ .text = t });
    _ = try e.handleKey(.paste_end);
}

fn rowTexts(layout: Layout, a: Allocator) ![][]const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    for (layout.rows) |row| try out.append(a, row.text);
    return out.toOwnedSlice(a);
}

test "rows break at the last space that fits, not mid-word" {
    var e = editor();
    defer e.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try typeText(&e, "the quick brown fox jumps");
    const spans = try e.wrap(a, 12);
    try testing.expectEqual(@as(usize, 3), spans.len);
    try testing.expectEqualStrings("the quick", e.buffer.items[spans[0].start..spans[0].end]);
    try testing.expectEqualStrings("brown fox", e.buffer.items[spans[1].start..spans[1].end]);
    try testing.expectEqualStrings("jumps", e.buffer.items[spans[2].start..spans[2].end]);
}

test "a word longer than the row still breaks, because it must" {
    var e = editor();
    defer e.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try typeText(&e, "supercalifragilistic");
    const spans = try e.wrap(a, 8);
    try testing.expectEqual(@as(usize, 3), spans.len);
    try testing.expectEqualStrings("supercal", e.buffer.items[spans[0].start..spans[0].end]);
}

test "explicit newlines end a row without a wrap" {
    var e = editor();
    defer e.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try typeText(&e, "a\n\nb");
    const spans = try e.wrap(a, 20);
    try testing.expectEqual(@as(usize, 3), spans.len);
    try testing.expectEqualStrings("", e.buffer.items[spans[1].start..spans[1].end]);
}

test "a long paste becomes one chip over the buffer, and the buffer is the text" {
    var e = editor();
    defer e.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try typeText(&e, "look: ");
    const pasted = "line\n" ** 96;
    try paste(&e, pasted);
    try testing.expectEqual(@as(usize, 1), e.chips.items.len);
    // The text sent is the paste itself, chip or no chip.
    try testing.expectEqualStrings("look: " ++ pasted, e.text());
    const l = try e.layout(a, .{ .width = 40, .min_rows = 3, .max_rows = 6, .theme = .{ .kind = .plain } });
    // One row: the typed prefix, then the chip's label in the chip style.
    try testing.expect(std.mem.startsWith(u8, l.rows[0].text, "look: "));
    try testing.expect(std.mem.indexOf(u8, l.rows[0].text, "[pasted 97 lines, 480 B]") != null);
    try testing.expectEqual(@as(usize, 0), l.cursor_row);
    // The chip is one token wide, so the whole paste occupies one row.
    try testing.expectEqual(@as(usize, 30), l.cursor_col);
}

test "a short paste is ordinary text" {
    var e = editor();
    defer e.deinit();
    try paste(&e, "two\nlines");
    try testing.expectEqual(@as(usize, 0), e.chips.items.len);
    try testing.expectEqualStrings("two\nlines", e.text());
}

test "the cursor steps over a chip, and Backspace at its edge deletes it whole" {
    var e = editor();
    defer e.deinit();
    try paste(&e, "x\n" ** 8);
    try typeText(&e, "!");
    const end = e.cursor;
    // Left from after the "!" lands after the chip, then before it.
    _ = try e.handleKey(.left);
    try testing.expectEqual(end - 1, e.cursor);
    _ = try e.handleKey(.left);
    try testing.expectEqual(@as(usize, 0), e.cursor);
    // Right steps over it in one move.
    _ = try e.handleKey(.right);
    try testing.expectEqual(@as(usize, 16), e.cursor);
    // Backspace at the chip's end removes the whole paste.
    _ = try e.handleKey(.backspace);
    try testing.expectEqualStrings("!", e.text());
    try testing.expectEqual(@as(usize, 0), e.chips.items.len);
}

test "the cursor steps and deletes a whole grapheme cluster" {
    var e = editor();
    defer e.deinit();
    try typeText(&e, "a");
    try typeText(&e, "👩‍🚀");
    // Left crosses the whole ZWJ sequence in one move; Right returns.
    _ = try e.handleKey(.left);
    try testing.expectEqual(@as(usize, 1), e.cursor);
    _ = try e.handleKey(.right);
    try testing.expectEqual(@as(usize, 12), e.cursor);
    // Backspace removes the cluster, not one code point.
    _ = try e.handleKey(.backspace);
    try testing.expectEqualStrings("a", e.text());

    // A base plus a combining mark is one cluster too.
    try typeText(&e, "e\u{301}");
    _ = try e.handleKey(.backspace);
    try testing.expectEqualStrings("a", e.text());
}

test "a cluster wraps as one unit and is measured in cells" {
    var e = editor();
    defer e.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try typeText(&e, "ab👩‍🚀cd");
    const spans = try e.wrap(a, 4); // "ab" + the 2-cell emoji fill the row
    try testing.expectEqual(@as(usize, 2), spans.len);
    try testing.expectEqualStrings("ab👩‍🚀", e.buffer.items[spans[0].start..spans[0].end]);
    try testing.expectEqualStrings("cd", e.buffer.items[spans[1].start..spans[1].end]);
}

test "Ctrl-E expands a chip so the text can be edited in place" {
    var e = editor();
    defer e.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try paste(&e, "keep\n" ** 5);
    e.cursor = 0;
    _ = try e.handleKey(.{ .ctrl = 'e' });
    try testing.expectEqual(@as(usize, 0), e.chips.items.len);
    const l = try e.layout(arena.allocator(), .{ .width = 40, .min_rows = 3, .max_rows = 8, .theme = .{ .kind = .plain } });
    try testing.expect(std.mem.indexOf(u8, l.rows[0].text, "keep") != null);
}

test "an edit inside a chip expands it rather than corrupting the range" {
    var e = editor();
    defer e.deinit();
    try paste(&e, "a\n" ** 8);
    e.cursor = 5; // strictly inside
    try typeText(&e, "Z");
    try testing.expectEqual(@as(usize, 0), e.chips.items.len);
    try testing.expectEqual(@as(usize, 17), e.text().len);
}

test "chips move with edits before them" {
    var e = editor();
    defer e.deinit();
    try paste(&e, "p\n" ** 6);
    const before = e.chips.items[0];
    e.cursor = 0;
    try typeText(&e, "hi ");
    try testing.expectEqual(before.start + 3, e.chips.items[0].start);
    try testing.expectEqual(before.end + 3, e.chips.items[0].end);
    // Deleting before the chip moves it back; Right would have stepped
    // over the whole chip, so walk left into the typed text instead.
    _ = try e.handleKey(.left);
    _ = try e.handleKey(.backspace);
    try testing.expectEqual(before.start + 2, e.chips.items[0].start);
    try testing.expectEqual(before.end + 2, e.chips.items[0].end);
}

test "a scrolled editor shows how many rows are hidden and keeps the cursor visible" {
    var e = editor();
    defer e.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try typeText(&e, "r1\nr2\nr3\nr4\nr5\nr6\nr7\nr8");
    const l = try e.layout(a, .{ .width = 20, .min_rows = 3, .max_rows = 4, .theme = .{ .kind = .plain } });
    try testing.expectEqual(@as(usize, 4), l.rows.len);
    // The indicator row carries the dim style, so it is not a bare prefix.
    try testing.expect(std.mem.indexOf(u8, l.rows[0].text, "… 5 lines above") != null);
    try testing.expect(std.mem.startsWith(u8, l.rows[3].text, "r8"));
    // The cursor is on the last row, which is visible.
    try testing.expectEqual(@as(usize, 3), l.cursor_row);
    try testing.expectEqual(@as(usize, 2), l.cursor_col);

    // Home scrolls back to the top and the indicator goes.
    _ = try e.handleKey(.home);
    const top = try e.layout(a, .{ .width = 20, .min_rows = 3, .max_rows = 4, .theme = .{ .kind = .plain } });
    try testing.expect(std.mem.startsWith(u8, top.rows[0].text, "r1"));
    try testing.expectEqual(@as(usize, 0), top.cursor_row);
}

test "the box is padded to its idle height and to the box width" {
    var e = editor();
    defer e.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const l = try e.layout(arena.allocator(), .{ .width = 10, .min_rows = 3, .max_rows = 6, .theme = .{ .kind = .plain } });
    try testing.expectEqual(@as(usize, 3), l.rows.len);
    try testing.expectEqualStrings("> ", l.rows[0].prefix);
    try testing.expectEqualStrings("  ", l.rows[1].prefix);
    for (l.rows) |row| {
        try testing.expect(row.editor);
        try testing.expectEqual(@as(usize, 10), view.styledWidth(row.text));
    }
}

test "Up and Down move inside the input and reach the history only at the edges" {
    var e = editor();
    defer e.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try e.remember("older");
    try typeText(&e, "one\ntwo\nthree");
    _ = try e.layout(a, .{ .width = 20, .min_rows = 3, .max_rows = 6, .theme = .{ .kind = .plain } });
    // From the last row, Up walks the input.
    _ = try e.handleKey(.up);
    try testing.expectEqual(@as(usize, 7), e.cursor); // end of "two"
    _ = try e.handleKey(.up);
    try testing.expectEqual(@as(usize, 3), e.cursor); // end of "one"
    // Another Up is on the first row: the history takes over.
    _ = try e.handleKey(.up);
    try testing.expectEqualStrings("older", e.text());
    // Down past the newest restores the draft.
    _ = try e.handleKey(.down);
    try testing.expectEqualStrings("one\ntwo\nthree", e.text());
}

test "history keeps its bound and does not repeat the newest entry" {
    var e = editor();
    defer e.deinit();
    try e.remember("a");
    try e.remember("a");
    try testing.expectEqual(@as(usize, 1), e.historyItems().len);
    var i: usize = 0;
    var buffer: [8]u8 = undefined;
    while (i < max_history + 10) : (i += 1) {
        try e.remember(try std.fmt.bufPrint(&buffer, "p{d}", .{i}));
    }
    try testing.expectEqual(max_history, e.historyItems().len);
    try testing.expectEqualStrings("p209", e.historyItems()[max_history - 1]);
}

test "input past the limit is refused whole" {
    var e = editor();
    defer e.deinit();
    const big = try testing.allocator.alloc(u8, max_input - 4);
    defer testing.allocator.free(big);
    @memset(big, 'x');
    try e.insert(big);
    try testing.expectEqual(max_input - 4, e.text().len);
    try e.insert("12345"); // would exceed: nothing is inserted
    try testing.expectEqual(max_input - 4, e.text().len);
    try e.insert("1234");
    try testing.expectEqual(max_input, e.text().len);
}

test "the word before the cursor is what a completion replaces" {
    var e = editor();
    defer e.deinit();
    try typeText(&e, "/ct");
    try testing.expectEqualStrings("/ct", e.wordBefore());
    try testing.expect(e.atFirstWord());
    try e.replaceWord("/ctx ");
    try testing.expectEqualStrings("/ctx ", e.text());
    // After a space the word is empty again, and no longer the first.
    try testing.expectEqualStrings("", e.wordBefore());
    try testing.expect(!e.atFirstWord());
    try typeText(&e, "look at @src/ma");
    try testing.expectEqualStrings("@src/ma", e.wordBefore());
    try e.replaceWord("@src/main.zig");
    try testing.expectEqualStrings("/ctx look at @src/main.zig", e.text());
    // Inside or at the end of a paste chip there is nothing to complete.
    var pasted = editor();
    defer pasted.deinit();
    try paste(&pasted, "x\n" ** 8);
    try testing.expectEqualStrings("", pasted.wordBefore());
}

test "keys the editor does not own are reported back to the caller" {
    var e = editor();
    defer e.deinit();
    try testing.expectEqual(Action.ignored, try e.handleKey(.{ .ctrl = 't' }));
    try testing.expectEqual(Action.ignored, try e.handleKey(.tab));
    try testing.expectEqual(Action.submit, try e.handleKey(.enter));
    try testing.expectEqual(Action.none, try e.handleKey(.{ .ctrl = 'j' }));
    try testing.expectEqualStrings("\n", e.text());
}

test "a pasted Enter is text, and a pasted control key is not a command" {
    var e = editor();
    defer e.deinit();
    _ = try e.handleKey(.paste_begin);
    _ = try e.handleKey(.{ .text = "a" });
    _ = try e.handleKey(.enter);
    _ = try e.handleKey(.{ .ctrl = 'c' });
    _ = try e.handleKey(.tab);
    _ = try e.handleKey(.{ .text = "b" });
    _ = try e.handleKey(.paste_end);
    try testing.expectEqualStrings("a\n    b", e.text());
}

test "control bytes in the buffer never reach the row" {
    var e = editor();
    defer e.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try typeText(&e, "a\x07b");
    const l = try e.layout(arena.allocator(), .{ .width = 20, .min_rows = 1, .max_rows = 4, .theme = .{ .kind = .plain } });
    try testing.expect(std.mem.indexOfScalar(u8, l.rows[0].text, 0x07) == null);
}

test "a 5 KB bracketed paste arrives in read-sized chunks and lays out as one chip" {
    // The shape the terminal actually delivers: CSI 200~, the bytes split
    // across 4 KiB reads, CSI 201~ — decoded by `keys.next` exactly as the
    // agent's poll loop does it, so a regression in either half shows here.
    var e = editor();
    defer e.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var stream: std.ArrayList(u8) = .empty;
    defer stream.deinit(testing.allocator);
    try stream.appendSlice(testing.allocator, "\x1b[200~");
    for (0..80) |_| try stream.appendSlice(testing.allocator, "The quick brown fox jumps over the lazy dog near the riverbank. ");
    try stream.appendSlice(testing.allocator, "\x1b[201~");

    var offset: usize = 0;
    while (offset < stream.items.len) {
        const chunk = stream.items[offset..@min(offset + 4096, stream.items.len)];
        var i: usize = 0;
        while (i < chunk.len) {
            const decoded = keys.next(chunk[i..]) orelse break;
            i += decoded.consumed;
            _ = try e.handleKey(decoded.key);
        }
        // A split sequence would stall the loop; the chunk size is chosen so
        // this stays a whole-chunk decode, as the agent's `pending` handles
        // the general case.
        try testing.expectEqual(chunk.len, i);
        offset += chunk.len;
    }
    try testing.expectEqual(@as(usize, 1), e.chips.items.len);
    try testing.expectEqual(@as(usize, 5120), e.text().len);
    const l = try e.layout(a, .{ .width = 97, .min_rows = 3, .max_rows = 6, .theme = .{ .kind = .truecolor } });
    try testing.expectEqual(@as(usize, 3), l.rows.len);
    try testing.expect(std.mem.indexOf(u8, l.rows[0].text, "[pasted 1 lines, 5.0 KB]") != null);
    // The whole paste is one token, so the cursor sits just past the chip.
    try testing.expectEqual(@as(usize, 0), l.cursor_row);
    try testing.expectEqual(@as(usize, 24), l.cursor_col);
}

/// A row's text without its escapes: these goldens pin the cells, the
/// styles are pinned in theme.zig.
fn stripped(a: Allocator, text: []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < text.len) : (i += 1) {
        if (text[i] == 0x1b) {
            i += 1;
            while (i < text.len and text[i] != 'm') i += 1;
            continue;
        }
        try out.append(a, text[i]);
    }
    return out.items;
}

test "the frame wraps the box: edges around every row, the spinner in the top edge while busy" {
    var e = editor();
    defer e.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try typeText(&e, "hello\nworld");
    const plain: theme.Theme = .{ .kind = .plain };
    const l = try e.layout(a, .{ .width = 30 - frame_cells, .min_rows = 3, .max_rows = 6, .theme = plain });
    const idle = try frame(a, l, .{ .width = 30, .theme = plain });
    try testing.expectEqual(@as(usize, 5), idle.len);
    for (idle) |row| {
        try testing.expect(row.raw);
        try testing.expectEqual(@as(usize, 30), view.styledWidth(row.text));
    }
    try testing.expectEqualStrings("╭" ++ ("─" ** 28) ++ "╮", try stripped(a, idle[0].text));
    try testing.expectEqualStrings("│ > hello" ++ (" " ** 19) ++ " │", try stripped(a, idle[1].text));
    try testing.expectEqualStrings("│   world" ++ (" " ** 19) ++ " │", try stripped(a, idle[2].text));
    try testing.expectEqualStrings("│  " ++ (" " ** 25) ++ " │", try stripped(a, idle[3].text));
    try testing.expectEqualStrings("╰" ++ ("─" ** 28) ++ "╯", try stripped(a, idle[4].text));
    // The cursor sits after "world": row 1 of the layout, column 5.
    try testing.expectEqual(@as(usize, 1), l.cursor_row);
    try testing.expectEqual(@as(usize, 5), l.cursor_col);

    const busy = try frame(a, l, .{ .width = 30, .theme = plain, .spinner = "⠋", .style = .frame_high });
    try testing.expectEqualStrings("╭─⠋" ++ ("─" ** 26) ++ "╮", try stripped(a, busy[0].text));
    try testing.expectEqual(@as(usize, 30), view.styledWidth(busy[0].text));

    const ascii = try frame(a, l, .{ .width = 30, .theme = .{ .kind = .plain, .glyph_set = .ascii }, .spinner = "|" });
    try testing.expectEqualStrings("+-|" ++ ("-" ** 26) ++ "+", try stripped(a, ascii[0].text));
    try testing.expectEqualStrings("| > hello" ++ (" " ** 19) ++ " |", try stripped(a, ascii[1].text));

    // A coloured theme paints the frame and the box background; the width holds.
    const styled = try frame(a, l, .{ .width = 30, .theme = .{ .kind = .truecolor }, .style = .frame_low });
    try testing.expect(std.mem.indexOf(u8, styled[1].text, "\x1b[") != null);
    try testing.expectEqual(@as(usize, 30), view.styledWidth(styled[1].text));
}

test "a scrolled box keeps its indicator row inside the frame" {
    var e = editor();
    defer e.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try typeText(&e, "1\n2\n3\n4\n5\n6");
    const plain: theme.Theme = .{ .kind = .plain };
    const l = try e.layout(a, .{ .width = 24, .min_rows = 3, .max_rows = 3, .theme = plain });
    const boxed = try frame(a, l, .{ .width = 30, .theme = plain });
    try testing.expectEqual(@as(usize, 5), boxed.len);
    try testing.expect(std.mem.startsWith(u8, try stripped(a, boxed[1].text), "│   … 4 lines above"));
    try testing.expectEqualStrings("│   6" ++ (" " ** 23) ++ " │", try stripped(a, boxed[3].text));
}
