//! The live region: how rows reach the terminal, and where the cursor ends
//! up afterwards. Everything else in `src/tui/` produces rows; this file is
//! the only place that knows an escape sequence for *movement*.
//!
//! nuclis does not use the alternate screen: completed turns are written once
//! into the terminal's scrollback and never repainted, so scrolling, search,
//! selection, and copy are the terminal's job and the transcript survives
//! exit. A bounded **live region** under the transcript holds the active
//! turn, the editor, and the status bar — the only thing repainted.
//!
//! Three operations, and the invariants that make them safe:
//!
//! - `paint(rows, cursor)` rewrites the region in place, wrapped in
//!   synchronized output (`CSI ? 2026 h/l`) so a frame never tears.
//!   `paintWith(above, rows, …)` also puts the rows that closed since the
//!   last frame above it, in the same frame and after the paint, so rows
//!   that leave the region for the transcript move the region by nothing.
//! - `insertAbove(rows)` puts completed lines above the region. It first
//!   opens blank rows under the region by inserting lines at its top
//!   (`IL`), which moves the region down; once the region reaches the last
//!   row, the scrolling region (`DECSTBM`) is narrowed to the rows above it,
//!   so inserting scrolls only those and never erases the region, and the
//!   top margin stays at row 1 so scrolled-off lines reach the scrollback.
//!   Without the capability it falls back to the cursor-up rewrite.
//! - `replaceAbove(rows, replacing)` rewrites the last `replacing` rows above
//!   the region (a fold toggle or a resize) and always takes the rewrite path,
//!   since the new block may have a different height.
//!
//! Geometry. `region_top` is the absolute row of the region's first line and
//! `region_rows` its height; every operation updates them, which is what lets
//! `insertAbove` name absolute rows without asking the terminal where the
//! cursor is. The region **follows the content**: its top is the row after
//! the transcript, a shorter region keeps its top and leaves blank rows under
//! the status bar (`slack`), and a taller one takes those rows back before
//! the transcript scrolls into the scrollback. Once the region ends on the
//! last row, growth and insertion scroll the transcript. A resize keeps the
//! region's top where the terminal reports the cursor (`resized`). What the
//! model cannot survive is the terminal reflowing rows under it, which is why
//! the caller replays the last turn after a resize.
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

    /// Paints nothing but the line it clears.
    pub fn isBlank(row: Row) bool {
        return row.text.len == 0 and row.prefix.len == 0 and !row.bar and !row.editor;
    }
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
    /// Cursor offset inside the region after the last paint, and the
    /// absolute column it was placed at.
    cursor_row: usize = 0,
    cursor_col: usize = 1,
    /// Whether the last paint showed the cursor, so an insertion that hides
    /// it while it moves shows it again.
    cursor_shown: bool = false,
    /// Absolute row of the cursor, tracked across every operation. Scrolling
    /// clamps it at the last row, exactly as the terminal does.
    at: usize = 1,
    /// Whether the last row written above the region paints nothing, so the
    /// caller can leave out a spacer of its own.
    above_blank: bool = false,

    /// Clears the visible screen — the scrollback is untouched — and puts the
    /// cursor at the top. Used at startup, before the header.
    pub fn clear(self: *Screen) !void {
        try self.out.writeAll("\x1b[H\x1b[2J");
        self.region_top = 0;
        self.region_rows = 0;
        self.at = 1;
        self.above_blank = false;
    }

    /// Blank rows under the painted region, down to the last row: what a
    /// growing region or an insertion takes before anything scrolls.
    pub fn slack(self: *const Screen) usize {
        if (self.region_rows == 0) return 0;
        return self.size.rows -| (self.region_top + self.region_rows - 1);
    }

    /// Rewrites the live region in place: the one frame the agent paints on
    /// every tick. `rows` is the whole region, top to bottom.
    pub fn paint(self: *Screen, rows: []const Row, cursor: ?Cursor) !void {
        return self.paintWith(&.{}, rows, cursor, 0);
    }

    /// `paint` for a caller that kept the previous frame: the first
    /// `unchanged` rows are identical to what the region already shows and
    /// are skipped when the region's geometry is the same, so a frame that
    /// only advanced the bar rewrites one row. A frame with nothing changed
    /// and the cursor in place writes nothing at all. Any geometry change
    /// (height, a region rebuilt from nothing) rewrites every row.
    pub fn paintFrom(self: *Screen, rows: []const Row, cursor: ?Cursor, unchanged: usize) !void {
        return self.paintWith(&.{}, rows, cursor, unchanged);
    }

    /// `paintFrom`, then `above` inserted above the region, in one frame.
    /// The order matters: rows that closed in the region since the last
    /// frame shrink it first and fill the rows that released, so a frame
    /// that only moves rows into the transcript leaves the region where it
    /// was and scrolls nothing.
    pub fn paintWith(self: *Screen, above: []const Row, rows: []const Row, cursor: ?Cursor, unchanged: usize) !void {
        if (above.len == 0 and self.region_rows == rows.len and unchanged >= rows.len and rows.len > 0) {
            const target = cursor orelse Cursor{ .row = rows.len - 1, .column = 1 };
            if (@min(target.row, rows.len - 1) == self.cursor_row and target.column == self.cursor_col) return;
        }
        try self.beginFrame();
        if (above.len > 0 and !self.canScroll(rows.len)) {
            // The rewrite puts `above` where the region was and leaves the
            // region to be rebuilt under it from nothing.
            try self.rewriteRows(above, 0);
            try self.writeRegion(rows, cursor, 0);
        } else {
            try self.writeRegion(rows, cursor, unchanged);
            if (above.len > 0) try self.insertRows(above);
        }
        try self.endFrame();
        try self.out.flush();
    }

    /// Writes completed rows above the live region. They belong to the
    /// scrollback from then on and are never repainted.
    pub fn insertAbove(self: *Screen, rows: []const Row) !void {
        if (rows.len == 0) return;
        try self.beginFrame();
        if (self.canScroll(self.region_rows)) try self.insertRows(rows) else try self.rewriteRows(rows, 0);
        try self.endFrame();
        try self.out.flush();
    }

    /// Replaces the last `replacing` rows above the region with `rows` (a
    /// fold toggle, a resize replay). The region is erased and rebuilt by
    /// the next `paint`, since the new block may have a different height.
    pub fn replaceAbove(self: *Screen, rows: []const Row, replacing: usize) !void {
        try self.beginFrame();
        try self.rewriteRows(rows, replacing);
        try self.endFrame();
        try self.out.flush();
    }

    /// Takes the terminal's new size. Returns true when it changed, which is
    /// the agent's signal to rebuild its rows at the new width: after a
    /// reflow the recorded geometry describes rows the terminal has already
    /// moved, so only relative motion (the rewrite path) is trustworthy.
    /// A painted region is erased where it stood and the next paint rebuilds
    /// it from the same top. `cursor` is the terminal's own answer to where
    /// the cursor is now (a resize moves it: a shorter terminal trims its
    /// top to keep the cursor's row, a taller one may pull lines back from
    /// the scrollback); without one, the cursor is taken to be where it was,
    /// clamped to the last row.
    pub fn resized(self: *Screen, size: Size, cursor: ?usize) !bool {
        if (size.rows == self.size.rows and size.columns == self.size.columns) return false;
        self.size = size;
        if (self.region_rows == 0) return true;
        const out = self.out;
        try self.beginFrame();
        self.at = @max(1, @min(cursor orelse self.at, size.rows));
        const up = @min(self.cursor_row, self.at - 1);
        if (up > 0) try out.print("\x1b[{d}A", .{up});
        self.at -= up;
        try out.writeAll("\r\x1b[0J");
        self.forgetRegion();
        try self.endFrame();
        try out.flush();
        return true;
    }

    /// Leaves the terminal holding the transcript and nothing else: the live
    /// region is erased, the scrolling region reset, the cursor shown.
    pub fn finish(self: *Screen) !void {
        const out = self.out;
        // `DECSTBM` homes the cursor, so the margins are reset inside a
        // cursor save/restore, before the relative walk below.
        try out.writeAll("\x1b7\x1b[r\x1b8");
        const up = @min(self.cursor_row, self.at -| 1);
        if (up > 0) try out.print("\x1b[{d}A", .{up});
        try out.writeAll("\r\x1b[0J\x1b[?25h");
        self.at -= up;
        self.forgetRegion();
        try out.flush();
    }

    // ----- internals -----

    /// Whether rows can go above a region of `rows` lines by moving and
    /// scrolling rather than by the rewrite: it needs the capability, a
    /// painted region to protect, and, once the slack is used up, at least
    /// one row above the region for the scrolling region to hold.
    fn canScroll(self: *const Screen, rows: usize) bool {
        return self.caps.scroll_region and self.region_rows > 0 and rows < self.size.rows;
    }

    /// Writes the region from its top: every row on a geometry change,
    /// otherwise only those after the `unchanged` prefix. Shrinking keeps
    /// the top and erases what is under the last row; growing writes past
    /// the old bottom into the slack, and past the last row the terminal
    /// scrolls the transcript up into the scrollback.
    fn writeRegion(self: *Screen, rows: []const Row, cursor: ?Cursor, unchanged: usize) !void {
        const out = self.out;
        const same_geometry = self.region_rows == rows.len and unchanged > 0;
        try out.writeAll("\x1b[?25l");
        if (self.region_rows > 0 and self.cursor_row > 0) {
            try out.print("\x1b[{d}A\r", .{self.cursor_row});
            self.at -|= self.cursor_row;
        } else try out.writeAll("\r");
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
        // A taller previous region leaves rows below the last one written:
        // erased, they are the slack.
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
            self.cursor_shown = true;
            self.at -|= up;
        } else {
            self.cursor_row = rows.len -| 1;
            self.cursor_col = 1;
            self.cursor_shown = false;
        }
    }

    /// Puts `rows` above a painted region, which `canScroll` has allowed.
    /// The slack is used first: lines inserted at the region's top push it
    /// down into the blank rows under it, and the rows land in the gap, so
    /// the transcript stays contiguous and nothing scrolls. The rest scroll
    /// the rows above the region, which by then ends on the last row.
    fn insertRows(self: *Screen, rows: []const Row) !void {
        const out = self.out;
        self.noteAbove(rows);
        try out.writeAll("\x1b[?25l");
        const fill = @min(self.slack(), rows.len);
        if (fill > 0) {
            // `IL` shifts the region and everything under it down; what falls
            // off the bottom is blank slack.
            try out.print("\x1b[{d};1H\x1b[{d}L", .{ self.region_top, fill });
            for (rows[0..fill], 0..) |row, i| {
                if (i > 0) try out.print("\x1b[{d};1H", .{self.region_top + i});
                try self.writeRow(row);
            }
            self.region_top += fill;
        }
        if (fill < rows.len) {
            // Top margin at row 1: the condition for scrolled-off rows to
            // reach the scrollback. The live region sits below `bottom` and
            // is not touched by the scrolling below.
            const bottom = self.region_top - 1;
            try out.print("\x1b[1;{d}r\x1b[{d};1H", .{ bottom, bottom });
            for (rows[fill..]) |row| {
                // One line feed on the bottom margin scrolls the area above
                // the region up by one and frees this row for the next line.
                try out.writeAll("\n");
                try self.writeRow(row);
            }
            try out.writeAll("\x1b[r");
        }
        // Put the cursor back where the last paint left it, relative to the
        // region, so the next paint can rewrite in place.
        try out.print("\x1b[{d};{d}H", .{ self.region_top + self.cursor_row, self.cursor_col });
        if (self.cursor_shown) try out.writeAll("\x1b[?25h");
        self.at = self.region_top + self.cursor_row;
    }

    /// The cursor-up rewrite: walks up over the region above the cursor and
    /// the `extra` rows being replaced (from the cursor when no region is
    /// pending), writes `rows`, and clears what remains below. The region is
    /// left for the next paint to rebuild right under the rows, so a block
    /// that shrank leaves no gap and one that grew scrolls the screen.
    fn rewriteRows(self: *Screen, rows: []const Row, extra: usize) !void {
        const out = self.out;
        self.noteAbove(rows);
        const up = @min(self.cursor_row + extra, self.at -| 1);
        if (up > 0) try out.print("\x1b[{d}A", .{up});
        try out.writeAll("\r");
        self.at -= up;
        for (rows) |row| {
            try self.writeRow(row);
            try self.newline();
        }
        try out.writeAll("\x1b[0J");
        self.forgetRegion();
    }

    fn noteAbove(self: *Screen, rows: []const Row) void {
        if (rows.len > 0) self.above_blank = rows[rows.len - 1].isBlank();
    }

    /// No region is painted: the next paint builds one from the cursor.
    fn forgetRegion(self: *Screen) void {
        self.region_rows = 0;
        self.region_top = 0;
        self.cursor_row = 0;
        self.cursor_col = 1;
        self.cursor_shown = false;
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

test "at startup the region follows the header, and an insertion fills the rows under it" {
    var buffer: std.Io.Writer.Allocating = .init(testing.allocator);
    defer buffer.deinit();
    var screen = testScreen(&buffer.writer, .{});
    // A three-row header on a 10-row screen, then a two-row region right
    // under it: rows 6..10 are blank slack.
    try screen.insertAbove(&.{ .{ .text = "h1" }, .{ .text = "h2" }, .{ .text = "h3" } });
    try testing.expectEqual(@as(usize, 4), screen.at);
    try screen.paint(&.{ .{ .text = "a" }, .{ .text = "b" } }, .{ .row = 0, .column = 1 });
    try testing.expectEqual(@as(usize, 4), screen.region_top);
    try testing.expectEqual(@as(usize, 5), screen.slack());
    // The notice lands on row 4 and pushes the region down one row; no
    // scrolling region is set.
    buffer.clearRetainingCapacity();
    try screen.insertAbove(&.{.{ .text = "notice" }});
    try testing.expectEqualStrings(
        "\x1b[?2026h\x1b[?25l\x1b[4;1H\x1b[1L\x1b[2K\rnotice\x1b[5;1H\x1b[?25h\x1b[?2026l",
        buffer.written(),
    );
    try testing.expectEqual(@as(usize, 5), screen.region_top);
    try testing.expectEqual(@as(usize, 4), screen.slack());
    // A taller region grows down into the slack: same top, nothing scrolls.
    buffer.clearRetainingCapacity();
    try screen.paint(&.{ .{ .text = "a" }, .{ .text = "b" }, .{ .text = "c" }, .{ .text = "d" } }, .{ .row = 0, .column = 1 });
    try testing.expect(std.mem.startsWith(u8, buffer.written(), "\x1b[?2026h\x1b[?25l\r\x1b[2K\ra"));
    try testing.expectEqual(@as(usize, 5), screen.region_top);
    try testing.expectEqual(@as(usize, 2), screen.slack());
}

test "the screen remembers whether the transcript ends with a blank row" {
    var buffer: std.Io.Writer.Allocating = .init(testing.allocator);
    defer buffer.deinit();
    var screen = testScreen(&buffer.writer, .{});
    try screen.insertAbove(&.{ .{ .text = "answer" }, .{ .text = "" } });
    try testing.expect(screen.above_blank);
    try screen.paint(&.{.{ .text = "editor", .editor = true }}, .{ .row = 0, .column = 1 });
    try screen.paintWith(&.{.{ .text = "notice" }}, &.{.{ .text = "editor", .editor = true }}, .{ .row = 0, .column = 1 }, 0);
    try testing.expect(!screen.above_blank);
    try screen.replaceAbove(&.{ .{ .text = "turn" }, .{ .text = "" } }, 1);
    try testing.expect(screen.above_blank);
    try testing.expect(!Row.isBlank(.{ .text = "", .bar = true }));
}

test "a fold toggle rewrites the turn and the region follows it" {
    var buffer: std.Io.Writer.Allocating = .init(testing.allocator);
    defer buffer.deinit();
    var screen = testScreen(&buffer.writer, .{});
    // A two-row turn, then a two-row region right under it at rows 3..4.
    try screen.insertAbove(&.{ .{ .text = "prompt" }, .{ .text = "folded" } });
    try screen.paint(&.{ .{ .text = "a" }, .{ .text = "b" } }, .{ .row = 0, .column = 1 });
    try testing.expectEqual(@as(usize, 3), screen.region_top);
    // Unfolding replaces two rows with four: the region moves down by two.
    try screen.replaceAbove(&.{ .{ .text = "prompt" }, .{ .text = "open" }, .{ .text = "t1" }, .{ .text = "t2" } }, 2);
    try screen.paint(&.{ .{ .text = "a" }, .{ .text = "b" } }, .{ .row = 0, .column = 1 });
    try testing.expectEqual(@as(usize, 5), screen.region_top);
    // Folding back moves it up again, leaving no gap above it.
    buffer.clearRetainingCapacity();
    try screen.replaceAbove(&.{ .{ .text = "prompt" }, .{ .text = "folded" } }, 4);
    try testing.expectEqualStrings(
        "\x1b[?2026h\x1b[4A\r\x1b[2K\rprompt\r\n\x1b[2K\rfolded\r\n\x1b[0J\x1b[?2026l",
        buffer.written(),
    );
    try screen.paint(&.{ .{ .text = "a" }, .{ .text = "b" } }, .{ .row = 0, .column = 1 });
    try testing.expectEqual(@as(usize, 3), screen.region_top);
    // A turn taller than the screen scrolls, and the region ends on the
    // last row.
    try screen.replaceAbove(&.{ .{ .text = "1" }, .{ .text = "2" }, .{ .text = "3" }, .{ .text = "4" }, .{ .text = "5" }, .{ .text = "6" }, .{ .text = "7" }, .{ .text = "8" }, .{ .text = "9" } }, 2);
    try screen.paint(&.{ .{ .text = "a" }, .{ .text = "b" } }, .{ .row = 0, .column = 1 });
    try testing.expectEqual(@as(usize, 9), screen.region_top);
    try testing.expectEqual(@as(usize, 0), screen.slack());
}

test "a shorter region keeps its top, and growth takes the rows back before the screen scrolls" {
    var buffer: std.Io.Writer.Allocating = .init(testing.allocator);
    defer buffer.deinit();
    var screen = testScreen(&buffer.writer, .{});
    screen.at = 6;
    // A four-row region at rows 6..9 of a 10-row screen.
    try screen.paint(&.{ .{ .text = "a" }, .{ .text = "b" }, .{ .text = "c" }, .{ .text = "d" } }, .{ .row = 3, .column = 1 });
    try testing.expectEqual(@as(usize, 6), screen.region_top);
    try testing.expectEqual(@as(usize, 1), screen.slack());
    buffer.clearRetainingCapacity();

    // Two rows now: the top stays at 6 and rows 8..10 are blank under it.
    try screen.paint(&.{ .{ .text = "x" }, .{ .text = "y" } }, .{ .row = 1, .column = 1 });
    try testing.expectEqualStrings(
        "\x1b[?2026h\x1b[?25l\x1b[3A\r\x1b[2K\rx\r\n\x1b[2K\ry\x1b[0J\x1b[1G\x1b[?25h\x1b[?2026l",
        buffer.written(),
    );
    try testing.expectEqual(@as(usize, 6), screen.region_top);
    try testing.expectEqual(@as(usize, 7), screen.at);
    try testing.expectEqual(@as(usize, 3), screen.slack());

    // Growing back writes down into those rows from the same top.
    buffer.clearRetainingCapacity();
    try screen.paint(&.{ .{ .text = "a" }, .{ .text = "b" }, .{ .text = "c" }, .{ .text = "d" } }, .{ .row = 3, .column = 1 });
    try testing.expect(std.mem.startsWith(u8, buffer.written(), "\x1b[?2026h\x1b[?25l\x1b[1A\r\x1b[2K\ra"));
    try testing.expectEqual(@as(usize, 6), screen.region_top);
    try testing.expectEqual(@as(usize, 1), screen.slack());

    // Past the slack, growth runs past the last row and the screen scrolls
    // the transcript up: the region ends on the last row.
    const six = [_]Row{ .{ .text = "a" }, .{ .text = "b" }, .{ .text = "c" }, .{ .text = "d" }, .{ .text = "e" }, .{ .text = "f" } };
    try screen.paint(&six, .{ .row = 5, .column = 1 });
    try testing.expectEqual(@as(usize, 10), screen.at);
    try testing.expectEqual(@as(usize, 5), screen.region_top);
    try testing.expectEqual(@as(usize, 0), screen.slack());

    // On a full screen a shrink still keeps the top; the rows go under the
    // bar, and growing again takes them back without scrolling.
    try screen.paint(&.{ .{ .text = "x" }, .{ .text = "y" } }, .{ .row = 1, .column = 1 });
    try testing.expectEqual(@as(usize, 5), screen.region_top);
    try testing.expectEqual(@as(usize, 4), screen.slack());
    try screen.paint(&six, .{ .row = 5, .column = 1 });
    try testing.expectEqual(@as(usize, 5), screen.region_top);
    try testing.expectEqual(@as(usize, 0), screen.slack());
}

test "rows that close in the region fill the rows they release, in one frame" {
    var buffer: std.Io.Writer.Allocating = .init(testing.allocator);
    defer buffer.deinit();
    var screen = testScreen(&buffer.writer, .{});
    // A three-row region on the last rows, 8..10.
    screen.at = 8;
    try screen.paint(&.{ .{ .text = "line" }, .{ .text = "editor" }, .{ .text = "bar" } }, .{ .row = 1, .column = 1 });
    try testing.expectEqual(@as(usize, 8), screen.region_top);
    buffer.clearRetainingCapacity();
    // The line closes: the region shrinks to two rows at 8..9, then the line
    // is inserted at row 8 and the region moves back to 9..10. Nothing
    // scrolls and the bar does not move.
    try screen.paintWith(&.{.{ .text = "line" }}, &.{ .{ .text = "editor" }, .{ .text = "bar" } }, .{ .row = 0, .column = 3 }, 0);
    try testing.expectEqualStrings(
        "\x1b[?2026h\x1b[?25l\x1b[1A\r\x1b[2K\reditor\r\n\x1b[2K\rbar\x1b[0J\x1b[1A\x1b[3G\x1b[?25h" ++
            "\x1b[?25l\x1b[8;1H\x1b[1L\x1b[2K\rline\x1b[9;3H\x1b[?25h\x1b[?2026l",
        buffer.written(),
    );
    try testing.expectEqual(@as(usize, 9), screen.region_top);
    try testing.expectEqual(@as(usize, 0), screen.slack());

    // Without the capability the same frame writes the line where the
    // region was and rebuilds the region under it.
    var fallback = testScreen(&buffer.writer, .{ .scroll_region = false });
    fallback.at = 8;
    try fallback.paint(&.{ .{ .text = "line" }, .{ .text = "editor" }, .{ .text = "bar" } }, .{ .row = 1, .column = 1 });
    buffer.clearRetainingCapacity();
    try fallback.paintWith(&.{.{ .text = "line" }}, &.{ .{ .text = "editor" }, .{ .text = "bar" } }, .{ .row = 0, .column = 3 }, 0);
    try testing.expectEqualStrings(
        "\x1b[?2026h\x1b[1A\r\x1b[2K\rline\r\n\x1b[0J" ++
            "\x1b[?25l\r\x1b[2K\reditor\r\n\x1b[2K\rbar\x1b[0J\x1b[1A\x1b[3G\x1b[?25h\x1b[?2026l",
        buffer.written(),
    );
    try testing.expectEqual(@as(usize, 9), fallback.region_top);
}

test "insertion fills the rows a shrink released before it scrolls" {
    var buffer: std.Io.Writer.Allocating = .init(testing.allocator);
    defer buffer.deinit();
    var screen = testScreen(&buffer.writer, .{});
    screen.at = 6;
    try screen.paint(&.{ .{ .text = "a" }, .{ .text = "b" }, .{ .text = "c" }, .{ .text = "d" } }, .{ .row = 3, .column = 1 });
    // The region shrinks to rows 6..7; rows 8..10 are blank slack.
    try screen.paint(&.{ .{ .text = "x" }, .{ .text = "y" } }, .{ .row = 1, .column = 2 });
    try testing.expectEqual(@as(usize, 3), screen.slack());
    buffer.clearRetainingCapacity();

    // One row lands on row 6 and the region moves to 7..8: no scrolling
    // region, and the cursor goes back into the region.
    try screen.insertAbove(&.{.{ .text = "one" }});
    try testing.expectEqualStrings(
        "\x1b[?2026h\x1b[?25l\x1b[6;1H\x1b[1L\x1b[2K\rone\x1b[8;2H\x1b[?25h\x1b[?2026l",
        buffer.written(),
    );
    try testing.expectEqual(@as(usize, 2), screen.slack());
    try testing.expectEqual(@as(usize, 7), screen.region_top);
    buffer.clearRetainingCapacity();

    // Three more: two fill rows 7 and 8 and bring the region to the last
    // row, the third scrolls rows 1..8.
    try screen.insertAbove(&.{ .{ .text = "two" }, .{ .text = "three" }, .{ .text = "four" } });
    try testing.expectEqualStrings(
        "\x1b[?2026h\x1b[?25l\x1b[7;1H\x1b[2L\x1b[2K\rtwo\x1b[8;1H\x1b[2K\rthree" ++
            "\x1b[1;8r\x1b[8;1H\n\x1b[2K\rfour\x1b[r\x1b[10;2H\x1b[?25h\x1b[?2026l",
        buffer.written(),
    );
    try testing.expectEqual(@as(usize, 0), screen.slack());
    try testing.expectEqual(@as(usize, 9), screen.region_top);
}

/// Fails when a line clear, an inserted line, or a line feed meets a painted
/// background, or an erase-to-end meets none: with background-colour erase,
/// those would band a row that is not code, or leave code unbanded.
fn expectBandOnlyOnErase(stream: []const u8) !void {
    var band = false;
    var i: usize = 0;
    while (i < stream.len) {
        if (stream[i] == '\n') try testing.expect(!band);
        const end = view.escapeSpan(stream, i) orelse {
            i += 1;
            continue;
        };
        const seq = stream[i..end];
        i = end;
        if (seq.len < 3 or seq[1] != '[') continue;
        switch (seq[seq.len - 1]) {
            'm' => {
                if (std.mem.indexOf(u8, seq, "48;") != null) band = true;
                if (std.mem.eql(u8, seq, theme.reset)) band = false;
            },
            'L' => try testing.expect(!band),
            'K' => try testing.expectEqual(!std.mem.eql(u8, seq, "\x1b[2K"), band),
            else => {},
        }
    }
}

test "a code row's band is painted by its own erase, never by an insert, a scroll, or a clear" {
    var buffer: std.Io.Writer.Allocating = .init(testing.allocator);
    defer buffer.deinit();
    const code = "\x1b[48;5;237mx = 1\x1b[K\x1b[0m";
    const rows = [_]Row{ .{ .text = code, .raw = true }, .{ .text = code, .raw = true }, .{ .text = code, .raw = true } };
    for ([_]Caps{ .{}, .{ .scroll_region = false } }) |caps| {
        var screen = testScreen(&buffer.writer, caps);
        // Two rows of slack under a two-row region: the first two rows are
        // inserted lines, the third scrolls the rows above the region.
        screen.at = 6;
        try screen.paint(&.{ .{ .text = "a" }, .{ .text = "b" }, .{ .text = "c" }, .{ .text = "d" } }, .{ .row = 3, .column = 1 });
        try screen.paint(&.{ .{ .text = "x" }, .{ .text = "y" } }, .{ .row = 1, .column = 1 });
        buffer.clearRetainingCapacity();
        try screen.insertAbove(&rows);
        try screen.paintWith(&rows, &.{ .{ .text = "x" }, .{ .text = "y" } }, .{ .row = 1, .column = 1 }, 0);
        try testing.expect(std.mem.indexOf(u8, buffer.written(), "x = 1\x1b[K") != null);
        try expectBandOnlyOnErase(buffer.written());
    }
}

test "the rewrite fallback and finish leave no gap after a shrink" {
    var buffer: std.Io.Writer.Allocating = .init(testing.allocator);
    defer buffer.deinit();
    var screen = testScreen(&buffer.writer, .{ .scroll_region = false });
    screen.at = 6;
    try screen.paint(&.{ .{ .text = "a" }, .{ .text = "b" }, .{ .text = "c" }, .{ .text = "d" } }, .{ .row = 3, .column = 1 });
    try screen.paint(&.{ .{ .text = "x" }, .{ .text = "y" } }, .{ .row = 1, .column = 1 });
    buffer.clearRetainingCapacity();
    // One region row above the cursor: the inserted row lands on row 6,
    // where the region began, and the region is rebuilt under it.
    try screen.insertAbove(&.{.{ .text = "turn" }});
    try testing.expectEqualStrings("\x1b[?2026h\x1b[1A\r\x1b[2K\rturn\r\n\x1b[0J\x1b[?2026l", buffer.written());
    try testing.expectEqual(@as(usize, 7), screen.at);
    try screen.paint(&.{ .{ .text = "x" }, .{ .text = "y" } }, .{ .row = 1, .column = 1 });
    try testing.expectEqual(@as(usize, 7), screen.region_top);

    // Exit walks up over the region only: the transcript ends on row 6.
    buffer.clearRetainingCapacity();
    try screen.finish();
    try testing.expectEqualStrings("\x1b7\x1b[r\x1b8\x1b[1A\r\x1b[0J\x1b[?25h", buffer.written());
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
        "\x1b[?2026h\x1b[?25l\x1b[1;8r\x1b[8;1H\n\x1b[2K\rturn\x1b[r\x1b[9;5H\x1b[?25h\x1b[?2026l",
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

test "a region that fills the screen cannot scroll and rewrites instead" {
    var buffer: std.Io.Writer.Allocating = .init(testing.allocator);
    defer buffer.deinit();
    var screen = testScreen(&buffer.writer, .{});
    // A one-row region on row 1 still has slack under it: an insertion
    // pushes it down with no scrolling region.
    try screen.paint(&.{.{ .text = "only" }}, .{ .row = 0, .column = 1 });
    try testing.expectEqual(@as(usize, 1), screen.region_top);
    buffer.clearRetainingCapacity();
    try screen.insertAbove(&.{.{ .text = "x" }});
    try testing.expect(std.mem.indexOf(u8, buffer.written(), "\x1b[1;1H\x1b[1L") != null);
    try testing.expectEqual(@as(usize, 2), screen.region_top);
    // Ten rows on a 10-row screen leave no row above for the margins.
    var full = testScreen(&buffer.writer, .{});
    const ten: [10]Row = @splat(.{ .text = "r" });
    try full.paint(&ten, .{ .row = 9, .column = 1 });
    buffer.clearRetainingCapacity();
    try full.insertAbove(&.{.{ .text = "x" }});
    try testing.expectEqualStrings("\x1b[?2026h\x1b[9A\r\x1b[2K\rx\r\n\x1b[0J\x1b[?2026l", buffer.written());
    try testing.expectEqual(@as(usize, 0), full.region_rows);
}

test "replacing rows above the region walks over them and erases the rest" {
    var buffer: std.Io.Writer.Allocating = .init(testing.allocator);
    defer buffer.deinit();
    var screen = testScreen(&buffer.writer, .{});
    screen.at = 9;
    try screen.paint(&.{ .{ .text = "editor" }, .{ .text = "bar" } }, .{ .row = 1, .column = 1 });
    buffer.clearRetainingCapacity();
    // One row of region above the cursor plus the three rows being replaced;
    // the region is rebuilt right under the shorter block.
    try screen.replaceAbove(&.{ .{ .text = "new" }, .{ .text = "rows" } }, 3);
    try testing.expectEqualStrings(
        "\x1b[?2026h\x1b[4A\r\x1b[2K\rnew\r\n\x1b[2K\rrows\r\n\x1b[0J\x1b[?2026l",
        buffer.written(),
    );
    try testing.expectEqual(@as(usize, 8), screen.at);
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
    try testing.expectEqualStrings("\x1b7\x1b[r\x1b8\r\x1b[0J\x1b[?25h", buffer.written());
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
    try testing.expect(!try screen.resized(.{ .rows = 10, .columns = 40 }, null));
    try testing.expect(try screen.resized(.{ .rows = 10, .columns = 60 }, null));
    try testing.expect(!try screen.resized(.{ .rows = 10, .columns = 60 }, null));
    try testing.expectEqual(@as(usize, 60), screen.size.columns);
    // Nothing painted yet: nothing to move.
    try testing.expectEqualStrings("", buffer.written());
}

test "a resize keeps the region's top where the terminal reports the cursor" {
    var buffer: std.Io.Writer.Allocating = .init(testing.allocator);
    defer buffer.deinit();
    var screen = testScreen(&buffer.writer, .{});
    // A three-row header, then a four-row region that shrank to two: rows
    // 4..5, with rows 6..10 blank under it.
    try screen.insertAbove(&.{ .{ .text = "h1" }, .{ .text = "h2" }, .{ .text = "h3" } });
    try screen.paint(&.{ .{ .text = "a" }, .{ .text = "b" }, .{ .text = "c" }, .{ .text = "d" } }, .{ .row = 3, .column = 3 });
    const rows = [_]Row{ .{ .text = "editor" }, .{ .text = "bar" } };
    try screen.paint(&rows, .{ .row = 1, .column = 3 });
    try testing.expectEqual(@as(usize, 4), screen.region_top);
    try testing.expectEqual(@as(usize, 5), screen.slack());
    buffer.clearRetainingCapacity();
    // Four rows appear at the bottom; the terminal keeps its top and the
    // cursor on row 5. The region is erased from its top and nothing is
    // walked: the new rows are slack.
    try testing.expect(try screen.resized(.{ .rows = 14, .columns = 40 }, 5));
    try testing.expectEqualStrings("\x1b[?2026h\x1b[1A\r\x1b[0J\x1b[?2026l", buffer.written());
    try testing.expectEqual(@as(usize, 0), screen.region_rows);
    try testing.expectEqual(@as(usize, 4), screen.at);
    try screen.paint(&rows, .{ .row = 1, .column = 3 });
    try testing.expectEqual(@as(usize, 4), screen.region_top);
    try testing.expectEqual(@as(usize, 9), screen.slack());
    // Down to 4 rows: the terminal trimmed its top to keep the cursor's row
    // and reports it on row 3, so the region now starts on row 2.
    buffer.clearRetainingCapacity();
    try testing.expect(try screen.resized(.{ .rows = 4, .columns = 40 }, 3));
    try testing.expectEqualStrings("\x1b[?2026h\x1b[1A\r\x1b[0J\x1b[?2026l", buffer.written());
    try screen.paint(&rows, .{ .row = 1, .column = 3 });
    try testing.expectEqual(@as(usize, 2), screen.region_top);
    try testing.expectEqual(@as(usize, 1), screen.slack());
    // Without a report the cursor is taken to be where it was.
    try testing.expect(try screen.resized(.{ .rows = 4, .columns = 60 }, null));
    try screen.paint(&rows, .{ .row = 1, .column = 3 });
    try testing.expectEqual(@as(usize, 2), screen.region_top);
}
