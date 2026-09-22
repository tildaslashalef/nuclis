//! The live region: how rows reach the terminal, and where the cursor ends
//! up afterwards. Everything else in `src/tui/` produces rows; this file is
//! the only place that knows an escape sequence for *movement*.
//!
//! nuclis does not use the alternate screen: completed turns are written once
//! into the terminal's scrollback and never repainted, so scrolling, search,
//! selection, and copy are the terminal's job and the transcript survives
//! exit. A bounded **live region** at the bottom holds the active turn, the
//! editor, and the status bar — the only thing repainted.
//!
//! Three operations, and the invariants that make them safe:
//!
//! - `paint(rows, cursor)` rewrites the region in place, wrapped in
//!   synchronized output (`CSI ? 2026 h/l`) so a frame never tears.
//! - `insertAbove(rows)` puts completed lines above the region. With `DECSTBM`
//!   the scrolling region is narrowed to the rows above the live region, so
//!   inserting scrolls only those rows and never erases the region; the top
//!   margin stays at row 1 so scrolled-off lines reach the scrollback. Without
//!   the capability it falls back to the cursor-up rewrite.
//! - `replaceAbove(rows, replacing)` rewrites the last `replacing` rows above
//!   the region (a fold toggle or a resize) and always takes the rewrite path,
//!   since the new block may have a different height.
//!
//! Geometry. `region_top` is the absolute row of the region's first line and
//! `region_rows` its height; every operation updates them, which is what lets
//! `insertAbove` name an absolute bottom margin without asking the terminal
//! where the cursor is. The region's **bottom** is the anchor: a region that
//! shrinks moves its top down and the rows it released stay blank above it
//! as `slack`, which the next growth takes back and the next insertion fills
//! before it scrolls, so the transcript never keeps a gap. The one input the
//! model cannot survive is a terminal reflowing rows under it, which is what
//! `resized` reports.
//!
//! `Screen` writes to a plain `std.Io.Writer` and never allocates, so its
//! tests capture the exact escape stream with no TTY in sight.
const std = @import("std");
const theme = @import("theme.zig");
const view = @import("view.zig");

pub const Size = @import("terminal.zig").Size;

/// One paintable row of the live region or of the transcript. `text` is
/// plain (written through `view.safe`) unless `raw` is set, in which case it
/// already carries its own SGR spans and sanitization (markdown output).
pub const Row = struct {
    text: []const u8,
    style: ?theme.Style = null,
    raw: bool = false,
    /// Painted with the status bar background.
    bar: bool = false,
    /// Painted with the editor background (highlighted input box).
    editor: bool = false,
    /// Written before the text (the editor prompt `"> "`).
    prefix: []const u8 = "",
};

/// Where the cursor is parked after a paint: a row of the region and an
/// absolute, 1-based column. `null` leaves it hidden at the end of the
/// region, which is what a frame painted while the model generates wants.
pub const Cursor = struct { row: usize, column: usize };

/// Escape-sequence capabilities. Both default to on: synchronized output is
/// ignored by terminals that do not implement it, and a scrolling region is
/// VT100. `NUCLIS_NO_SCROLL_REGION=1` forces the rewrite fallback for a
/// terminal that mishandles `DECSTBM`, and a dumb or unset `TERM` turns both
/// off, since neither can be relied on there.
pub const Caps = struct {
    synchronized: bool = true,
    scroll_region: bool = true,

    pub fn detect(environ: *const std.process.Environ.Map) Caps {
        const term = environ.get("TERM") orelse "";
        if (term.len == 0 or std.mem.eql(u8, term, "dumb")) return .{ .synchronized = false, .scroll_region = false };
        const forced_off = if (environ.get("NUCLIS_NO_SCROLL_REGION")) |v| !std.mem.eql(u8, v, "0") else false;
        return .{ .synchronized = true, .scroll_region = !forced_off };
    }
};

pub const Screen = struct {
    out: *std.Io.Writer,
    th: theme.Theme,
    caps: Caps = .{},
    size: Size = .{},
    /// Absolute row of the live region's first line; 0 when none is painted.
    region_top: usize = 0,
    region_rows: usize = 0,
    /// Blank rows directly above the region, released by a shrink and not
    /// yet reused. The transcript ends `slack + 1` rows above `region_top`.
    slack: usize = 0,
    /// Cursor offset inside the region after the last paint, and the
    /// absolute column it was placed at.
    cursor_row: usize = 0,
    cursor_col: usize = 1,
    /// Absolute row of the cursor, tracked across every operation. Scrolling
    /// clamps it at the last row, exactly as the terminal does.
    at: usize = 1,

    /// Clears the visible screen — the scrollback is untouched — and puts the
    /// cursor at the top. Used at startup, before the header.
    pub fn clear(self: *Screen) !void {
        try self.out.writeAll("\x1b[H\x1b[2J");
        self.region_top = 0;
        self.region_rows = 0;
        self.slack = 0;
        self.at = 1;
    }

    /// Walks the cursor down so that a region of `rows` lines painted next
    /// will end on the last row of the screen. Called once at startup: with
    /// the region anchored at the bottom, everything above it is transcript,
    /// and `insertAbove` can scroll all of it. The blank rows walked over are
    /// slack: insertions fill them and a growing region takes them back
    /// before anything above scrolls off the top.
    pub fn anchor(self: *Screen, rows: usize) !void {
        const target = self.size.rows -| (rows -| 1);
        if (self.at < target) self.slack = target - self.at;
        while (self.at < target) try self.newline();
        try self.out.flush();
    }

    /// Rewrites the live region in place: the one frame the agent paints on
    /// every tick. `rows` is the whole region, top to bottom.
    pub fn paint(self: *Screen, rows: []const Row, cursor: ?Cursor) !void {
        return self.paintFrom(rows, cursor, 0);
    }

    /// `paint` for a caller that kept the previous frame: the first
    /// `unchanged` rows are identical to what the region already shows and
    /// are skipped when the region's geometry is the same, so a frame that
    /// only advanced the bar rewrites one row. A frame with nothing changed
    /// and the cursor in place writes nothing at all. Any geometry change
    /// (height, slack, a region rebuilt from nothing) rewrites every row.
    pub fn paintFrom(self: *Screen, rows: []const Row, cursor: ?Cursor, unchanged: usize) !void {
        const out = self.out;
        const same_geometry = self.region_rows == rows.len and unchanged > 0;
        if (same_geometry and unchanged >= rows.len) {
            const target = cursor orelse Cursor{ .row = rows.len -| 1, .column = 1 };
            if (@min(target.row, rows.len -| 1) == self.cursor_row and target.column == self.cursor_col) return;
        }
        try self.beginFrame();
        try out.writeAll("\x1b[?25l");
        if (self.region_rows > 0 and self.cursor_row > 0) {
            try out.print("\x1b[{d}A\r", .{self.cursor_row});
            self.at -|= self.cursor_row;
        } else try out.writeAll("\r");
        // The region's *bottom* is the anchor, not its top. A shorter region
        // (the turn ends, a tool call settles) would otherwise leave blank
        // rows under the editor, so the rows it no longer needs are erased
        // and released above it, where `insertAbove` fills them before it
        // scrolls. A taller one grows back into that slack first and only
        // then pushes the transcript up, into the scrollback. Only the
        // shrinking case erases, so the common frame still overwrites in
        // place and a terminal without synchronized output does not flicker.
        const released = self.region_rows -| rows.len;
        if (released > 0) {
            try out.writeAll("\x1b[0J");
            for (0..released) |_| try self.newline();
            self.slack += released;
        } else {
            // Growth comes out of the slack first: a region growing back
            // keeps its bottom and moves its top up into the blank rows, and
            // the first frame after `anchor`, when it is taller than the
            // anchor walked for, does the same with the rows that would run
            // past the bottom. Neither scrolls the transcript.
            const growth = if (self.region_rows > 0) rows.len - self.region_rows else (self.at + rows.len -| 1) -| self.size.rows;
            const reclaimed = @min(growth, self.slack);
            if (reclaimed > 0) {
                try out.print("\x1b[{d}A", .{reclaimed});
                self.at -= reclaimed;
                self.slack -= reclaimed;
            }
        }
        const top = self.at;
        // Rows the region already shows are stepped over, never rewritten;
        // moving down inside the region cannot scroll.
        const skip = if (same_geometry) @min(unchanged, rows.len -| 1) else 0;
        if (skip > 0) {
            try out.print("\x1b[{d}B", .{skip});
            self.at += skip;
        }
        for (rows[skip..], skip..) |row, i| {
            if (i >= unchanged or !same_geometry) try self.writeRow(row);
            if (i + 1 < rows.len) try self.newline();
        }
        // A taller previous region leaves rows below the last one written.
        try out.writeAll("\x1b[0J");
        // Writing may have scrolled: the region ends at `self.at`, so its top
        // is that many rows back, never the row the writes started on.
        self.region_rows = rows.len;
        self.region_top = self.at + 1 -| rows.len;
        std.debug.assert(self.region_top <= top or rows.len == 0);
        if (cursor) |c| {
            const row = @min(c.row, rows.len -| 1);
            const up = (rows.len -| 1) - row;
            if (up > 0) try out.print("\x1b[{d}A", .{up});
            try out.print("\x1b[{d}G\x1b[?25h", .{c.column});
            self.cursor_row = row;
            self.cursor_col = c.column;
            self.at -|= up;
        } else {
            self.cursor_row = rows.len -| 1;
            self.cursor_col = 1;
        }
        try self.endFrame();
        try out.flush();
    }

    /// Writes completed rows above the live region. They belong to the
    /// scrollback from then on and are never repainted.
    pub fn insertAbove(self: *Screen, rows: []const Row) !void {
        if (rows.len == 0) return;
        // A scrolling region needs at least one row above the live one, and
        // a region to protect in the first place.
        if (!self.caps.scroll_region or self.region_rows == 0 or self.region_top <= 1) {
            return self.rewriteAbove(rows, 0);
        }
        const out = self.out;
        const bottom = self.region_top - 1;
        try self.beginFrame();
        try out.writeAll("\x1b[?25l");
        // Rows released by a shrink sit blank between the transcript and the
        // region: they are filled first, so the transcript stays contiguous
        // and nothing scrolls until the slack is gone.
        const fill = @min(self.slack, rows.len);
        for (rows[0..fill], 0..) |row, i| {
            try out.print("\x1b[{d};1H", .{self.region_top - self.slack + i});
            try self.writeRow(row);
        }
        self.slack -= fill;
        if (fill < rows.len) {
            // Top margin at row 1: the condition for scrolled-off rows to
            // reach the scrollback. The live region sits below `bottom` and
            // is not touched by the scrolling below.
            try out.print("\x1b[1;{d}r\x1b[{d};1H", .{ bottom, bottom });
            for (rows[fill..]) |row| {
                // One line feed on the bottom margin scrolls the area above
                // the region up by one and frees this row for the next line.
                try out.writeAll("\n");
                try self.writeRow(row);
            }
            try out.writeAll("\x1b[r");
        }
        // The region did not move; put the cursor back where the last paint
        // left it so the next one can rewrite in place.
        try out.print("\x1b[{d};{d}H", .{ self.region_top + self.cursor_row, self.cursor_col });
        self.at = self.region_top + self.cursor_row;
        try self.endFrame();
        try out.flush();
    }

    /// Replaces the last `replacing` rows above the region with `rows` (a
    /// fold toggle, a resize replay). The region is erased and rebuilt by
    /// the next `paint`, since the new block may have a different height.
    pub fn replaceAbove(self: *Screen, rows: []const Row, replacing: usize) !void {
        return self.rewriteAbove(rows, replacing);
    }

    /// Re-reads the terminal size. Returns true when it changed, which is
    /// the agent's signal to rebuild its rows at the new width: after a
    /// reflow the recorded geometry describes rows the terminal has already
    /// moved, so only relative motion (the rewrite path) is trustworthy.
    pub fn resized(self: *Screen, size: Size) bool {
        if (size.rows == self.size.rows and size.columns == self.size.columns) return false;
        self.size = size;
        return true;
    }

    /// Leaves the terminal holding the transcript and nothing else: the live
    /// region is erased, the scrolling region reset, the cursor shown.
    pub fn finish(self: *Screen) !void {
        const out = self.out;
        // The slack goes with the region: the transcript ends where it ends.
        const up = @min(self.cursor_row + self.slack, self.at -| 1);
        if (up > 0) try out.print("\x1b[{d}A", .{up});
        try out.writeAll("\r\x1b[0J\x1b[r\x1b[?25h");
        self.at -= up;
        self.region_rows = 0;
        self.region_top = 0;
        self.slack = 0;
        self.cursor_row = 0;
        try out.flush();
    }

    // ----- internals -----

    /// Writes rows starting at the top of the live region (or at the cursor
    /// when no region is pending), then clears whatever remains below: the
    /// next paint starts right after the written rows. The walk crosses the
    /// slack above the region as well, so the rows land right after the
    /// transcript; `extra` moves the start further up, over rows that are
    /// being replaced.
    /// The cursor-up rewrite: over the region, the slack, and the `extra`
    /// rows being replaced; the new rows, then the slack again so the region
    /// keeps its place. Rows the block grew by come out of the slack (the
    /// region scrolls only past it); rows it shrank by join it, so a fold
    /// never lifts the editor off the bottom of the screen.
    fn rewriteAbove(self: *Screen, rows: []const Row, extra: usize) !void {
        const out = self.out;
        try self.beginFrame();
        const up = @min(self.cursor_row + self.slack + extra, self.at -| 1);
        if (up > 0) try out.print("\x1b[{d}A", .{up});
        try out.writeAll("\r");
        self.at -= up;
        for (rows) |row| {
            try self.writeRow(row);
            try self.newline();
        }
        self.slack = (self.slack + (extra -| rows.len)) -| (rows.len -| extra);
        for (0..self.slack) |_| try out.writeAll("\x1b[2K\r\n");
        self.at = @min(self.at + self.slack, self.size.rows);
        try out.writeAll("\x1b[0J");
        self.region_rows = 0;
        self.region_top = 0;
        self.cursor_row = 0;
        self.cursor_col = 1;
        try self.endFrame();
        try out.flush();
    }

    /// Writes one row: clear the line, prefix, optional background and
    /// style, then the text (sanitized unless it is pre-styled output).
    fn writeRow(self: *Screen, row: Row) !void {
        const out = self.out;
        // The cursor may sit anywhere after the previous frame, so every row
        // starts by clearing its line and returning to column 1.
        try out.writeAll("\x1b[2K\r");
        if (row.editor or row.bar) {
            // The background spans the prefix as well, so box rows read as
            // one highlighted area.
            try out.writeAll(self.th.paint(if (row.bar) .status_bg else .editor_bg));
        }
        try out.writeAll(row.prefix);
        if (row.style) |s| try out.writeAll(self.th.paint(s));
        if (row.raw) try out.writeAll(row.text) else try view.safe(out, row.text);
        if (row.style != null or row.bar or row.editor) try out.writeAll(theme.reset);
    }

    /// A line feed at the last row scrolls instead of moving down.
    fn newline(self: *Screen) !void {
        try self.out.writeAll("\r\n");
        if (self.at < self.size.rows) self.at += 1;
    }

    fn beginFrame(self: *Screen) !void {
        if (self.caps.synchronized) try self.out.writeAll("\x1b[?2026h");
    }
    fn endFrame(self: *Screen) !void {
        if (self.caps.synchronized) try self.out.writeAll("\x1b[?2026l");
    }
};

// ----- tests: the escape stream is the contract -----

const testing = std.testing;

fn testScreen(out: *std.Io.Writer, caps: Caps) Screen {
    return .{ .out = out, .th = .{ .kind = .plain }, .caps = caps, .size = .{ .rows = 10, .columns = 40 } };
}

test "a repaint is one synchronized frame that rewrites the region in place" {
    var buffer: std.Io.Writer.Allocating = .init(testing.allocator);
    defer buffer.deinit();
    var screen = testScreen(&buffer.writer, .{});
    const rows = [_]Row{ .{ .text = "one" }, .{ .text = "two" } };
    try screen.paint(&rows, .{ .row = 1, .column = 3 });
    try testing.expectEqualStrings(
        "\x1b[?2026h\x1b[?25l\r\x1b[2K\rone\r\n\x1b[2K\rtwo\x1b[0J\x1b[3G\x1b[?25h\x1b[?2026l",
        buffer.written(),
    );
    // Two rows written from row 1: the region is rows 1..2 and the cursor
    // sits on the second of them.
    try testing.expectEqual(@as(usize, 1), screen.region_top);
    try testing.expectEqual(@as(usize, 2), screen.region_rows);
    try testing.expectEqual(@as(usize, 1), screen.cursor_row);

    // The next paint returns to the top of the region first.
    buffer.clearRetainingCapacity();
    try screen.paint(&rows, .{ .row = 0, .column = 1 });
    try testing.expect(std.mem.startsWith(u8, buffer.written(), "\x1b[?2026h\x1b[?25l\x1b[1A\r"));
    try testing.expectEqual(@as(usize, 0), screen.cursor_row);
    try testing.expectEqual(@as(usize, 1), screen.region_top);
}

test "a frame rewrites only the rows after the unchanged prefix, and nothing when nothing changed" {
    var buffer: std.Io.Writer.Allocating = .init(testing.allocator);
    defer buffer.deinit();
    var screen = testScreen(&buffer.writer, .{});
    const rows = [_]Row{ .{ .text = "one" }, .{ .text = "two" }, .{ .text = "bar" } };
    try screen.paintFrom(&rows, .{ .row = 1, .column = 3 }, 0);
    try testing.expectEqual(@as(usize, 1), screen.region_top);
    try testing.expectEqual(@as(usize, 2), screen.at);

    // Only the bar changed: up to the top, two rows down, one row written.
    buffer.clearRetainingCapacity();
    const bar = [_]Row{ .{ .text = "one" }, .{ .text = "two" }, .{ .text = "BAR" } };
    try screen.paintFrom(&bar, .{ .row = 1, .column = 3 }, 2);
    try testing.expectEqualStrings(
        "\x1b[?2026h\x1b[?25l\x1b[1A\r\x1b[2B\x1b[2K\rBAR\x1b[0J\x1b[1A\x1b[3G\x1b[?25h\x1b[?2026l",
        buffer.written(),
    );
    try testing.expectEqual(@as(usize, 1), screen.region_top);
    try testing.expectEqual(@as(usize, 1), screen.cursor_row);

    // Nothing changed and the cursor is in place: no bytes.
    buffer.clearRetainingCapacity();
    try screen.paintFrom(&bar, .{ .row = 1, .column = 3 }, 3);
    try testing.expectEqualStrings("", buffer.written());

    // Nothing changed but the cursor moved: a reposition without a rewrite.
    try screen.paintFrom(&bar, .{ .row = 0, .column = 2 }, 3);
    try testing.expectEqualStrings("\x1b[?2026h\x1b[?25l\x1b[1A\r\x1b[2B\x1b[0J\x1b[2A\x1b[2G\x1b[?25h\x1b[?2026l", buffer.written());
    try testing.expectEqual(@as(usize, 0), screen.cursor_row);

    // A height change ignores the prefix and rewrites every row.
    buffer.clearRetainingCapacity();
    const taller = [_]Row{ .{ .text = "one" }, .{ .text = "two" }, .{ .text = "x" }, .{ .text = "BAR" } };
    try screen.paintFrom(&taller, .{ .row = 3, .column = 1 }, 2);
    try testing.expect(std.mem.indexOf(u8, buffer.written(), "\x1b[2K\rone\r\n\x1b[2K\rtwo") != null);
    try testing.expectEqual(@as(usize, 4), screen.region_rows);
}

test "the startup anchor records the blank rows it walks as slack, so an insertion fills them" {
    var buffer: std.Io.Writer.Allocating = .init(testing.allocator);
    defer buffer.deinit();
    var screen = testScreen(&buffer.writer, .{});
    // A three-row header on a 10-row screen, then a two-row region at the
    // bottom: rows 4..8 are blank.
    try screen.insertAbove(&.{ .{ .text = "h1" }, .{ .text = "h2" }, .{ .text = "h3" } });
    try testing.expectEqual(@as(usize, 4), screen.at);
    try screen.anchor(2);
    try testing.expectEqual(@as(usize, 9), screen.at);
    try testing.expectEqual(@as(usize, 5), screen.slack);
    try screen.paint(&.{ .{ .text = "a" }, .{ .text = "b" } }, .{ .row = 0, .column = 1 });
    try testing.expectEqual(@as(usize, 9), screen.region_top);
    try testing.expectEqual(@as(usize, 5), screen.slack);
    // The notice lands on row 4, the first blank one; nothing scrolls.
    buffer.clearRetainingCapacity();
    try screen.insertAbove(&.{.{ .text = "notice" }});
    try testing.expect(std.mem.indexOf(u8, buffer.written(), "\x1b[4;1H") != null);
    try testing.expect(std.mem.indexOf(u8, buffer.written(), "\x1b[1;8r") == null);
    try testing.expectEqual(@as(usize, 4), screen.slack);

    // A first frame taller than the anchor walked for takes the extra rows
    // from the slack rather than scrolling: anchored for 2, painted with 4.
    var fresh = testScreen(&buffer.writer, .{});
    try fresh.insertAbove(&.{ .{ .text = "h1" }, .{ .text = "h2" }, .{ .text = "h3" } });
    try fresh.anchor(2);
    buffer.clearRetainingCapacity();
    try fresh.paint(&.{ .{ .text = "a" }, .{ .text = "b" }, .{ .text = "c" }, .{ .text = "d" } }, .{ .row = 0, .column = 1 });
    try testing.expect(std.mem.startsWith(u8, buffer.written(), "\x1b[?2026h\x1b[?25l\r\x1b[2A"));
    try testing.expectEqual(@as(usize, 7), fresh.region_top);
    try testing.expectEqual(@as(usize, 4), fresh.region_rows);
    try testing.expectEqual(@as(usize, 3), fresh.slack);
}

test "a fold toggle rewrites the turn and keeps the region at the bottom, the slack between them" {
    var buffer: std.Io.Writer.Allocating = .init(testing.allocator);
    defer buffer.deinit();
    var screen = testScreen(&buffer.writer, .{});
    // A two-row turn, the anchor, a two-row region at rows 9..10: five blank
    // rows of slack between them.
    try screen.insertAbove(&.{ .{ .text = "prompt" }, .{ .text = "folded" } });
    try screen.anchor(2);
    try screen.paint(&.{ .{ .text = "a" }, .{ .text = "b" } }, .{ .row = 0, .column = 1 });
    try testing.expectEqual(@as(usize, 9), screen.region_top);
    try testing.expectEqual(@as(usize, 6), screen.slack);
    // Unfolding replaces two rows with four: the slack absorbs the growth and
    // the next paint lands on the same rows.
    buffer.clearRetainingCapacity();
    try screen.replaceAbove(&.{ .{ .text = "prompt" }, .{ .text = "open" }, .{ .text = "t1" }, .{ .text = "t2" } }, 2);
    try testing.expectEqual(@as(usize, 4), screen.slack);
    try screen.paint(&.{ .{ .text = "a" }, .{ .text = "b" } }, .{ .row = 0, .column = 1 });
    try testing.expectEqual(@as(usize, 9), screen.region_top);
    // Folding back gives the rows to the slack; the region stays put.
    try screen.replaceAbove(&.{ .{ .text = "prompt" }, .{ .text = "folded" } }, 4);
    try testing.expectEqual(@as(usize, 6), screen.slack);
    try screen.paint(&.{ .{ .text = "a" }, .{ .text = "b" } }, .{ .row = 0, .column = 1 });
    try testing.expectEqual(@as(usize, 9), screen.region_top);
    // A turn taller than the slack pushes the region down the normal way.
    try screen.replaceAbove(&.{ .{ .text = "1" }, .{ .text = "2" }, .{ .text = "3" }, .{ .text = "4" }, .{ .text = "5" }, .{ .text = "6" }, .{ .text = "7" }, .{ .text = "8" }, .{ .text = "9" } }, 2);
    try testing.expectEqual(@as(usize, 0), screen.slack);
    try screen.paint(&.{ .{ .text = "a" }, .{ .text = "b" } }, .{ .row = 0, .column = 1 });
    try testing.expectEqual(@as(usize, 9), screen.region_top);
}

test "a shorter region keeps its bottom and releases the rows above it" {
    var buffer: std.Io.Writer.Allocating = .init(testing.allocator);
    defer buffer.deinit();
    var screen = testScreen(&buffer.writer, .{});
    screen.at = 6;
    // A four-row region ending on the last row of a 10-row screen.
    try screen.paint(&.{ .{ .text = "a" }, .{ .text = "b" }, .{ .text = "c" }, .{ .text = "d" } }, .{ .row = 3, .column = 1 });
    try testing.expectEqual(@as(usize, 6), screen.region_top);
    try testing.expectEqual(@as(usize, 9), screen.at);
    buffer.clearRetainingCapacity();

    // Two rows now: the bottom stays at 9, the top moves down to 8.
    try screen.paint(&.{ .{ .text = "x" }, .{ .text = "y" } }, .{ .row = 1, .column = 1 });
    try testing.expect(std.mem.startsWith(u8, buffer.written(), "\x1b[?2026h\x1b[?25l\x1b[3A\r\x1b[0J\r\n\r\n"));
    try testing.expectEqual(@as(usize, 8), screen.region_top);
    try testing.expectEqual(@as(usize, 2), screen.region_rows);
    try testing.expectEqual(@as(usize, 9), screen.at);
    try testing.expectEqual(@as(usize, 2), screen.slack);

    // Growing takes the released rows back first: the region is written
    // from row 6 again and nothing scrolls. Nothing is erased first: a
    // growing frame overwrites in place.
    buffer.clearRetainingCapacity();
    try screen.paint(&.{ .{ .text = "a" }, .{ .text = "b" }, .{ .text = "c" }, .{ .text = "d" } }, .{ .row = 3, .column = 1 });
    try testing.expect(std.mem.startsWith(u8, buffer.written(), "\x1b[?2026h\x1b[?25l\x1b[1A\r\x1b[2A\x1b[2K\ra"));
    try testing.expectEqual(@as(usize, 9), screen.at);
    try testing.expectEqual(@as(usize, 6), screen.region_top);
    try testing.expectEqual(@as(usize, 0), screen.slack);
    try testing.expect(std.mem.indexOf(u8, buffer.written(), "\x1b[0J\r\n") == null);

    // Past the slack, growth runs past the last row and the screen scrolls,
    // so the region ends anchored at the bottom.
    buffer.clearRetainingCapacity();
    try screen.paint(&.{ .{ .text = "a" }, .{ .text = "b" }, .{ .text = "c" }, .{ .text = "d" }, .{ .text = "e" } }, .{ .row = 4, .column = 1 });
    try testing.expectEqual(@as(usize, 10), screen.at);
    try testing.expectEqual(@as(usize, 6), screen.region_top);
}

test "insertion fills the rows a shrink released before it scrolls" {
    var buffer: std.Io.Writer.Allocating = .init(testing.allocator);
    defer buffer.deinit();
    var screen = testScreen(&buffer.writer, .{});
    screen.at = 6;
    try screen.paint(&.{ .{ .text = "a" }, .{ .text = "b" }, .{ .text = "c" }, .{ .text = "d" } }, .{ .row = 3, .column = 1 });
    // The region shrinks to rows 8..9; rows 6 and 7 are blank slack.
    try screen.paint(&.{ .{ .text = "x" }, .{ .text = "y" } }, .{ .row = 1, .column = 2 });
    try testing.expectEqual(@as(usize, 2), screen.slack);
    buffer.clearRetainingCapacity();

    // One row lands on row 6, in place: no scrolling region, no line feed,
    // and the cursor goes back into the region.
    try screen.insertAbove(&.{.{ .text = "one" }});
    try testing.expectEqualStrings(
        "\x1b[?2026h\x1b[?25l\x1b[6;1H\x1b[2K\rone\x1b[9;2H\x1b[?2026l",
        buffer.written(),
    );
    try testing.expectEqual(@as(usize, 1), screen.slack);
    try testing.expectEqual(@as(usize, 8), screen.region_top);
    buffer.clearRetainingCapacity();

    // Two more: the first fills row 7, the second scrolls rows 1..7 as
    // every insertion did before the slack existed.
    try screen.insertAbove(&.{ .{ .text = "two" }, .{ .text = "three" } });
    try testing.expectEqualStrings(
        "\x1b[?2026h\x1b[?25l\x1b[7;1H\x1b[2K\rtwo\x1b[1;7r\x1b[7;1H\n\x1b[2K\rthree\x1b[r\x1b[9;2H\x1b[?2026l",
        buffer.written(),
    );
    try testing.expectEqual(@as(usize, 0), screen.slack);
    try testing.expectEqual(@as(usize, 8), screen.region_top);
}

test "finish and the rewrite fallback walk over the slack as well" {
    var buffer: std.Io.Writer.Allocating = .init(testing.allocator);
    defer buffer.deinit();
    var screen = testScreen(&buffer.writer, .{ .scroll_region = false });
    screen.at = 6;
    try screen.paint(&.{ .{ .text = "a" }, .{ .text = "b" }, .{ .text = "c" }, .{ .text = "d" } }, .{ .row = 3, .column = 1 });
    try screen.paint(&.{ .{ .text = "x" }, .{ .text = "y" } }, .{ .row = 1, .column = 1 });
    buffer.clearRetainingCapacity();
    // One region row above the cursor plus two rows of slack: the inserted
    // row lands on row 6, right after the transcript, and the remaining
    // slack row is walked again so the region keeps its place.
    try screen.insertAbove(&.{.{ .text = "turn" }});
    try testing.expect(std.mem.startsWith(u8, buffer.written(), "\x1b[?2026h\x1b[3A\r\x1b[2K\rturn\r\n\x1b[2K\r\n\x1b[0J"));
    try testing.expectEqual(@as(usize, 1), screen.slack);
    try testing.expectEqual(@as(usize, 8), screen.at);

    // The same walk on exit leaves the cursor right after the transcript.
    try screen.paint(&.{ .{ .text = "a" }, .{ .text = "b" }, .{ .text = "c" } }, .{ .row = 2, .column = 1 });
    try screen.paint(&.{.{ .text = "x" }}, .{ .row = 0, .column = 1 });
    try testing.expectEqual(@as(usize, 3), screen.slack);
    buffer.clearRetainingCapacity();
    try screen.finish();
    try testing.expectEqualStrings("\x1b[3A\r\x1b[0J\x1b[r\x1b[?25h", buffer.written());
    try testing.expectEqual(@as(usize, 7), screen.at);
}

test "a frame painted while busy leaves the cursor hidden at the last row" {
    var buffer: std.Io.Writer.Allocating = .init(testing.allocator);
    defer buffer.deinit();
    var screen = testScreen(&buffer.writer, .{});
    try screen.paint(&.{ .{ .text = "a" }, .{ .text = "b" } }, null);
    try testing.expect(std.mem.indexOf(u8, buffer.written(), "\x1b[?25h") == null);
    try testing.expectEqual(@as(usize, 1), screen.cursor_row);
}

test "insertion above the region scrolls only the rows above it" {
    var buffer: std.Io.Writer.Allocating = .init(testing.allocator);
    defer buffer.deinit();
    var screen = testScreen(&buffer.writer, .{});
    // Paint a two-row region at the bottom of a 10-row screen.
    screen.at = 9;
    try screen.paint(&.{ .{ .text = "editor" }, .{ .text = "bar" } }, .{ .row = 0, .column = 5 });
    try testing.expectEqual(@as(usize, 9), screen.region_top);
    buffer.clearRetainingCapacity();

    try screen.insertAbove(&.{.{ .text = "turn" }});
    try testing.expectEqualStrings(
        // Margins 1..8 (everything above the region), cursor to the bottom
        // margin, one line feed to scroll, the row, margins reset, cursor
        // back where the paint left it.
        "\x1b[?2026h\x1b[?25l\x1b[1;8r\x1b[8;1H\n\x1b[2K\rturn\x1b[r\x1b[9;5H\x1b[?2026l",
        buffer.written(),
    );
    // The region did not move, so the next paint still rewrites in place.
    try testing.expectEqual(@as(usize, 9), screen.region_top);
    try testing.expectEqual(@as(usize, 2), screen.region_rows);
    try testing.expectEqual(@as(usize, 0), screen.cursor_row);
}

test "without the capability insertion falls back to the cursor-up rewrite" {
    var buffer: std.Io.Writer.Allocating = .init(testing.allocator);
    defer buffer.deinit();
    var screen = testScreen(&buffer.writer, .{ .scroll_region = false });
    screen.at = 9;
    try screen.paint(&.{ .{ .text = "editor" }, .{ .text = "bar" } }, .{ .row = 1, .column = 1 });
    buffer.clearRetainingCapacity();

    try screen.insertAbove(&.{.{ .text = "turn" }});
    try testing.expectEqualStrings(
        "\x1b[?2026h\x1b[1A\r\x1b[2K\rturn\r\n\x1b[0J\x1b[?2026l",
        buffer.written(),
    );
    // No region is pending: the next paint rebuilds it from the cursor.
    try testing.expectEqual(@as(usize, 0), screen.region_rows);
    try testing.expectEqual(@as(usize, 0), screen.cursor_row);
}

test "a region with nothing above it cannot scroll and rewrites instead" {
    var buffer: std.Io.Writer.Allocating = .init(testing.allocator);
    defer buffer.deinit();
    var screen = testScreen(&buffer.writer, .{});
    try screen.paint(&.{.{ .text = "only" }}, .{ .row = 0, .column = 1 });
    try testing.expectEqual(@as(usize, 1), screen.region_top);
    buffer.clearRetainingCapacity();
    try screen.insertAbove(&.{.{ .text = "x" }});
    try testing.expect(std.mem.indexOf(u8, buffer.written(), "\x1b[1;") == null);
}

test "replacing rows above the region walks over them and erases the rest" {
    var buffer: std.Io.Writer.Allocating = .init(testing.allocator);
    defer buffer.deinit();
    var screen = testScreen(&buffer.writer, .{});
    screen.at = 9;
    try screen.paint(&.{ .{ .text = "editor" }, .{ .text = "bar" } }, .{ .row = 1, .column = 1 });
    buffer.clearRetainingCapacity();
    // One row of region above the cursor plus the three rows being replaced;
    // the row the block shrank by becomes slack and is walked blank.
    try screen.replaceAbove(&.{ .{ .text = "new" }, .{ .text = "rows" } }, 3);
    try testing.expectEqualStrings(
        "\x1b[?2026h\x1b[4A\r\x1b[2K\rnew\r\n\x1b[2K\rrows\r\n\x1b[2K\r\n\x1b[0J\x1b[?2026l",
        buffer.written(),
    );
    try testing.expectEqual(@as(usize, 1), screen.slack);
}

test "a paint never walks above the first row" {
    var buffer: std.Io.Writer.Allocating = .init(testing.allocator);
    defer buffer.deinit();
    var screen = testScreen(&buffer.writer, .{});
    screen.at = 2;
    screen.cursor_row = 1;
    screen.region_rows = 2;
    screen.region_top = 1;
    try screen.replaceAbove(&.{.{ .text = "x" }}, 10);
    // Bounded by the cursor's own row, not by the requested count.
    try testing.expect(std.mem.indexOf(u8, buffer.written(), "\x1b[1A\r") != null);
}

test "finish erases the region and restores the terminal's scrolling region" {
    var buffer: std.Io.Writer.Allocating = .init(testing.allocator);
    defer buffer.deinit();
    var screen = testScreen(&buffer.writer, .{});
    screen.at = 9;
    try screen.paint(&.{ .{ .text = "editor" }, .{ .text = "bar" } }, .{ .row = 0, .column = 3 });
    buffer.clearRetainingCapacity();
    try screen.finish();
    try testing.expectEqualStrings("\r\x1b[0J\x1b[r\x1b[?25h", buffer.written());
    try testing.expectEqual(@as(usize, 0), screen.region_rows);
}

test "rows carry their own background, style, and sanitization" {
    var buffer: std.Io.Writer.Allocating = .init(testing.allocator);
    defer buffer.deinit();
    var screen: Screen = .{ .out = &buffer.writer, .th = .{ .kind = .truecolor }, .size = .{ .rows = 10, .columns = 40 } };
    try screen.paint(&.{
        .{ .text = "> hi\x07", .editor = true, .prefix = "> " },
        .{ .text = "raw\x1b[0m", .raw = true, .bar = true },
    }, null);
    const written = buffer.written();
    // The editor background opens before the prefix and closes after the row.
    try testing.expect(std.mem.indexOf(u8, written, "\x1b[48;2;50;48;47m> ") != null);
    // A control byte in plain text never reaches the terminal…
    try testing.expect(std.mem.indexOf(u8, written, "\x07") == null);
    // …while a raw row is written verbatim.
    try testing.expect(std.mem.indexOf(u8, written, "raw\x1b[0m") != null);
}

test "capabilities come from the environment, with an escape hatch" {
    const a = testing.allocator;
    var map = std.process.Environ.Map.init(a);
    defer map.deinit();
    try map.put("TERM", "xterm-256color");
    try testing.expectEqual(Caps{ .synchronized = true, .scroll_region = true }, Caps.detect(&map));
    try map.put("NUCLIS_NO_SCROLL_REGION", "1");
    try testing.expectEqual(Caps{ .synchronized = true, .scroll_region = false }, Caps.detect(&map));
    try map.put("NUCLIS_NO_SCROLL_REGION", "0");
    try testing.expect(Caps.detect(&map).scroll_region);
    try map.put("TERM", "dumb");
    try testing.expectEqual(Caps{ .synchronized = false, .scroll_region = false }, Caps.detect(&map));
}

test "a resize is reported once per change" {
    var buffer: std.Io.Writer.Allocating = .init(testing.allocator);
    defer buffer.deinit();
    var screen = testScreen(&buffer.writer, .{});
    try testing.expect(!screen.resized(.{ .rows = 10, .columns = 40 }));
    try testing.expect(screen.resized(.{ .rows = 10, .columns = 60 }));
    try testing.expect(!screen.resized(.{ .rows = 10, .columns = 60 }));
    try testing.expectEqual(@as(usize, 60), screen.size.columns);
}
