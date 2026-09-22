//! An inline list with a selection: the one interactive component that is
//! neither the editor nor the transcript.
//!
//! It renders *inside the live region* — nuclis has no alternate screen and no
//! overlay windows (docs/spec.md § The agent, Surface), so a picker is simply a
//! few more rows above the prompt. One component serves every use: the slash
//! command and `@` path completion of phase 1 session 2, the `/resume` session
//! picker of phase 2, and — if a permission step is ever wanted — an approval
//! prompt. That is why it exists before any of them does.
//!
//! Ownership: `set` copies the labels, so a caller may rebuild the list from a
//! frame arena on every keystroke without worrying about what the component
//! still points at. `layout` allocates its rows from the caller's short-lived
//! allocator, like every other view in this module.
const std = @import("std");
const screen = @import("screen.zig");
const theme = @import("theme.zig");
const view = @import("view.zig");

const Allocator = std.mem.Allocator;
pub const Row = screen.Row;

/// One offered choice: what to show, and an optional dim hint after it.
pub const Item = struct { label: []const u8, detail: []const u8 = "" };

pub const Options = struct {
    width: usize,
    max_rows: usize,
    th: theme.Theme,
};

pub const Choice = struct {
    alloc: Allocator,
    items: std.ArrayList(Item) = .empty,
    /// Index of the highlighted row; always in range while the list is not
    /// empty.
    selected: usize = 0,
    /// First visible row, moved only as far as keeping the selection visible
    /// requires.
    scroll: usize = 0,

    pub fn deinit(self: *Choice) void {
        self.clear();
        self.items.deinit(self.alloc);
    }

    fn clear(self: *Choice) void {
        for (self.items.items) |item| {
            self.alloc.free(item.label);
            self.alloc.free(item.detail);
        }
        self.items.clearRetainingCapacity();
    }

    /// Replaces the list, keeping the selection on the same row when it still
    /// exists (a completion list narrowing as the user types should not jump
    /// back to the top).
    pub fn set(self: *Choice, items: []const Item) !void {
        const previous = self.selected;
        self.clear();
        for (items) |item| {
            const label = try self.alloc.dupe(u8, item.label);
            errdefer self.alloc.free(label);
            const detail = try self.alloc.dupe(u8, item.detail);
            errdefer self.alloc.free(detail);
            try self.items.append(self.alloc, .{ .label = label, .detail = detail });
        }
        self.selected = @min(previous, self.items.items.len -| 1);
        self.scroll = 0;
    }

    pub fn isEmpty(self: *const Choice) bool {
        return self.items.items.len == 0;
    }

    /// The selected item, or null when the list is empty.
    pub fn current(self: *const Choice) ?Item {
        if (self.items.items.len == 0) return null;
        return self.items.items[self.selected];
    }

    /// Moves the selection, wrapping at both ends: a list short enough to be
    /// seen at once is faster to cycle than to walk.
    pub fn move(self: *Choice, direction: enum { next, previous }) void {
        const count = self.items.items.len;
        if (count == 0) return;
        self.selected = switch (direction) {
            .next => (self.selected + 1) % count,
            .previous => (self.selected + count - 1) % count,
        };
    }

    /// The rows to paint, scrolled so the selection is visible.
    pub fn layout(self: *Choice, a: Allocator, options: Options) ![]const Row {
        var rows: std.ArrayList(Row) = .empty;
        const count = self.items.items.len;
        if (count == 0) return rows.items;
        const capacity = @max(options.max_rows, 1);
        const shown = @min(count, capacity);
        if (self.selected < self.scroll) self.scroll = self.selected;
        if (self.selected >= self.scroll + shown) self.scroll = self.selected + 1 - shown;
        if (self.scroll + shown > count) self.scroll = count - shown;
        const th = options.th;
        for (self.items.items[self.scroll..][0..shown], self.scroll..) |item, index| {
            const selected = index == self.selected;
            var w: std.Io.Writer.Allocating = .init(a);
            if (selected) try w.writer.writeAll(th.paint(.choice_selected));
            try w.writer.print("{s} ", .{if (selected) th.glyphs().fold_closed else " "});
            try view.safe(&w.writer, item.label);
            if (item.detail.len > 0) {
                // The hint is dim next to a plain row and part of the
                // selection's own span when the row is selected, so a
                // highlighted row keeps one background across its width.
                if (!selected) try w.writer.writeAll(th.paint(.dim));
                try w.writer.writeByte(' ');
                try view.safe(&w.writer, item.detail);
                if (!selected) try w.writer.writeAll(theme.reset);
            }
            // Pad the selected row so its background spans the list's width.
            if (selected) {
                const pad = options.width -| view.styledWidth(w.written());
                try w.writer.splatByteAll(' ', pad);
                try w.writer.writeAll(theme.reset);
            }
            const fitted = try view.wrapStyled(a, w.written(), options.width, .character);
            try rows.append(a, .{ .text = fitted[0], .raw = true });
        }
        return rows.items;
    }
};

// ----- tests -----

const testing = std.testing;

fn choice() Choice {
    return .{ .alloc = testing.allocator };
}

test "an empty list draws nothing and has nothing to offer" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var c = choice();
    defer c.deinit();
    try testing.expect(c.isEmpty());
    try testing.expect(c.current() == null);
    c.move(.next); // must not trap on an empty list
    try testing.expectEqual(@as(usize, 0), (try c.layout(arena.allocator(), .{ .width = 20, .max_rows = 4, .th = .{ .kind = .plain } })).len);
}

test "the selection wraps, and the marker follows it" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var c = choice();
    defer c.deinit();
    try c.set(&.{
        .{ .label = "/new", .detail = "new session" },
        .{ .label = "/ctx", .detail = "context window" },
        .{ .label = "/help" },
    });
    try testing.expectEqualStrings("/new", c.current().?.label);
    const rows = try c.layout(a, .{ .width = 30, .max_rows = 4, .th = .{ .kind = .plain } });
    try testing.expectEqual(@as(usize, 3), rows.len);
    try testing.expect(std.mem.indexOf(u8, rows[0].text, "▸ /new") != null);
    try testing.expect(std.mem.indexOf(u8, rows[1].text, "  /ctx") != null);
    try testing.expect(std.mem.indexOf(u8, rows[1].text, "context window") != null);
    // Every row is exactly as wide as the list when it is the selected one.
    try testing.expectEqual(@as(usize, 30), view.styledWidth(rows[0].text));

    c.move(.previous);
    try testing.expectEqualStrings("/help", c.current().?.label);
    c.move(.next);
    try testing.expectEqualStrings("/new", c.current().?.label);
}

test "a long list scrolls only as far as the selection needs" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var c = choice();
    defer c.deinit();
    var items: [8]Item = undefined;
    var labels: [8][4]u8 = undefined;
    for (0..8) |i| {
        labels[i] = .{ 'p', '0' + @as(u8, @intCast(i)), 0, 0 };
        items[i] = .{ .label = labels[i][0..2] };
    }
    try c.set(&items);
    const options: Options = .{ .width = 20, .max_rows = 3, .th = .{ .kind = .plain } };
    const top = try c.layout(a, options);
    try testing.expectEqual(@as(usize, 3), top.len);
    try testing.expect(std.mem.indexOf(u8, top[0].text, "p0") != null);
    // Walking past the bottom scrolls by one, not to the end.
    for (0..3) |_| c.move(.next);
    const scrolled = try c.layout(a, options);
    try testing.expect(std.mem.indexOf(u8, scrolled[0].text, "p1") != null);
    try testing.expect(std.mem.indexOf(u8, scrolled[2].text, "▸ p3") != null);
    // Wrapping to the last item brings the end of the list into view.
    c.selected = 0;
    c.move(.previous);
    const end = try c.layout(a, options);
    try testing.expect(std.mem.indexOf(u8, end[2].text, "▸ p7") != null);
}

test "rebuilding the list keeps the selection where it still exists" {
    var c = choice();
    defer c.deinit();
    try c.set(&.{ .{ .label = "alpha" }, .{ .label = "beta" }, .{ .label = "gamma" } });
    c.move(.next);
    c.move(.next);
    try testing.expectEqualStrings("gamma", c.current().?.label);
    // A narrower list clamps rather than losing the selection entirely.
    try c.set(&.{ .{ .label = "alpha" }, .{ .label = "beta" } });
    try testing.expectEqualStrings("beta", c.current().?.label);
    try c.set(&.{});
    try testing.expect(c.current() == null);
}

test "control bytes in a label never reach the row" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var c = choice();
    defer c.deinit();
    try c.set(&.{.{ .label = "we\x07ird", .detail = "de\x1b[31mtail" }});
    const rows = try c.layout(arena.allocator(), .{ .width = 40, .max_rows = 4, .th = .{ .kind = .plain } });
    try testing.expect(std.mem.indexOfScalar(u8, rows[0].text, 0x07) == null);
    // The escape is neutralised, not the text: what was pasted stays
    // readable, but it can never colour the terminal.
    try testing.expect(std.mem.indexOf(u8, rows[0].text, "\x1b[31m") == null);
    try testing.expect(std.mem.indexOf(u8, rows[0].text, "de[31mtail") != null);
}
