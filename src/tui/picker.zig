//! The chooser every command that picks something opens (docs/spec.md
//! § 7.2, the agent's commands): a title, a search box filtered with
//! `fuzzy.score`, the list with the current item marked, optional option
//! rows adjusted with ←/→, and a footer naming the keys. A destructive
//! action asks inside it (`confirm`) rather than in a second picker.
//!
//! The picker owns the keyboard while open and reports what a key meant
//! (`Action`); what choosing *does* is the caller's. Rows render inside the
//! live region like every other overlay. `set`, `setOptions`, `setNote`, and
//! `confirm` copy their strings, so callers may pass frame-arena slices.
const std = @import("std");
const screen = @import("screen.zig");
const theme = @import("theme.zig");
const view = @import("view.zig");
const keys = @import("keys.zig");
const fuzzy = @import("fuzzy.zig");

const Allocator = std.mem.Allocator;
pub const Row = screen.Row;

pub const Item = struct {
    label: []const u8,
    /// A dim hint after the label.
    detail: []const u8 = "",
    /// What the search box matches; the label when empty.
    search: []const u8 = "",
    /// Marked as the one in use (the running model, this session).
    current: bool = false,
};

/// A setting chosen with the item: a name and the values ←/→ step through.
pub const Option = struct {
    name: []const u8,
    values: []const []const u8,
    index: usize = 0,
};

/// What a key meant. `moved` and `adjusted` let the caller refresh what
/// depends on the selection; `delete` asks the caller to `confirm` (only
/// when `deletable`).
pub const Action = enum { none, choose, close, moved, adjusted, delete, confirmed, declined };

pub const Layout = struct {
    width: usize,
    /// Rows the list may take; the title, search, options, and footer are
    /// extra.
    max_rows: usize,
    th: theme.Theme,
};

pub const Picker = struct {
    alloc: Allocator,
    title: []u8 = &.{},
    query: std.ArrayList(u8) = .empty,
    items: std.ArrayList(Item) = .empty,
    /// Indices into `items` that match the query, best first.
    shown: std.ArrayList(usize) = .empty,
    /// Index into `shown`; in range while `shown` is not empty.
    selected: usize = 0,
    scroll: usize = 0,
    options: std.ArrayList(Option) = .empty,
    /// The option row ←/→ adjust, or null while the list has the focus.
    focus: ?usize = null,
    note: []u8 = &.{},
    /// The question a destructive action waits on; only `y` confirms.
    question: ?[]u8 = null,
    /// Ctrl-D reports `delete` (the footer says so).
    deletable: bool = false,

    pub fn deinit(self: *Picker) void {
        self.clear();
        self.query.deinit(self.alloc);
        self.items.deinit(self.alloc);
        self.shown.deinit(self.alloc);
        self.options.deinit(self.alloc);
    }

    /// Empties the picker; it is then closed (`isOpen` is false).
    pub fn clear(self: *Picker) void {
        self.alloc.free(self.title);
        self.title = &.{};
        for (self.items.items) |item| freeItem(self.alloc, item);
        self.items.clearRetainingCapacity();
        self.shown.clearRetainingCapacity();
        self.query.clearRetainingCapacity();
        self.clearOptions();
        self.alloc.free(self.note);
        self.note = &.{};
        self.dismiss();
        self.selected = 0;
        self.scroll = 0;
        self.focus = null;
        self.deletable = false;
    }

    pub fn isOpen(self: *const Picker) bool {
        return self.title.len > 0;
    }

    /// Opens with `items`, the current one selected, and `query` typed
    /// into the search box already (a command argument that named no single
    /// item).
    pub fn open(self: *Picker, title: []const u8, items: []const Item, query: []const u8) !void {
        self.clear();
        errdefer self.clear();
        self.title = try self.alloc.dupe(u8, if (title.len == 0) " " else title);
        try self.items.ensureTotalCapacity(self.alloc, items.len);
        for (items) |item| self.items.appendAssumeCapacity(try dupeItem(self.alloc, item));
        try self.query.appendSlice(self.alloc, query);
        try self.filter();
        for (self.shown.items, 0..) |index, at| if (self.items.items[index].current) {
            self.selected = at;
            break;
        };
    }

    /// The selected item's index in the list `open` was given, or null when
    /// nothing matches the query.
    pub fn chosen(self: *const Picker) ?usize {
        if (self.shown.items.len == 0) return null;
        return self.shown.items[self.selected];
    }

    /// Selects item `index` (as `open` was given) when the query shows it.
    pub fn select(self: *Picker, index: usize) bool {
        for (self.shown.items, 0..) |shown, at| if (shown == index) {
            self.selected = at;
            self.focus = null;
            return true;
        };
        return false;
    }

    /// The items the query leaves, for a caller that resolves a typed
    /// argument without showing the picker.
    pub fn matches(self: *const Picker) []const usize {
        return self.shown.items;
    }

    pub fn setOptions(self: *Picker, options: []const Option) !void {
        self.clearOptions();
        for (options) |option| {
            const values = try self.alloc.alloc([]const u8, option.values.len);
            var filled: usize = 0;
            errdefer {
                for (values[0..filled]) |v| self.alloc.free(v);
                self.alloc.free(values);
            }
            for (option.values, values) |v, *out| {
                out.* = try self.alloc.dupe(u8, v);
                filled += 1;
            }
            const name = try self.alloc.dupe(u8, option.name);
            errdefer self.alloc.free(name);
            try self.options.append(self.alloc, .{ .name = name, .values = values, .index = @min(option.index, values.len -| 1) });
        }
        if (self.focus) |f| if (f >= self.options.items.len) {
            self.focus = null;
        };
    }

    /// The value index of option `n`.
    pub fn optionIndex(self: *const Picker, n: usize) usize {
        return self.options.items[n].index;
    }

    /// One dim line above the footer (what choosing will cost); empty for none.
    pub fn setNote(self: *Picker, text: []const u8) !void {
        const copy = try self.alloc.dupe(u8, text);
        self.alloc.free(self.note);
        self.note = copy;
    }

    /// Asks `question`; the next key answers it.
    pub fn confirm(self: *Picker, question: []const u8) !void {
        const copy = try self.alloc.dupe(u8, question);
        self.dismiss();
        self.question = copy;
    }

    fn dismiss(self: *Picker) void {
        if (self.question) |q| self.alloc.free(q);
        self.question = null;
    }

    /// Routes one key while the picker is open.
    pub fn handle(self: *Picker, key: keys.Key) !Action {
        if (self.question != null) {
            const yes = switch (key) {
                .text => |t| std.ascii.eqlIgnoreCase(t, "y"),
                else => false,
            };
            self.dismiss();
            return if (yes) .confirmed else .declined;
        }
        switch (key) {
            .up => return self.step(.previous),
            .down => return self.step(.next),
            .left, .right => {
                const f = self.focus orelse return .none;
                const option = &self.options.items[f];
                const before = option.index;
                option.index = if (key == .left) option.index -| 1 else @min(option.index + 1, option.values.len -| 1);
                return if (option.index == before) .none else .adjusted;
            },
            .enter, .tab => return if (self.chosen() != null) .choose else .none,
            .escape => return .close,
            .ctrl => |c| return switch (c) {
                'c' => .close,
                'd' => if (self.deletable and self.chosen() != null) .delete else .none,
                else => .none,
            },
            .text => |t| {
                try self.query.appendSlice(self.alloc, t);
                return self.refilter();
            },
            .backspace => {
                if (self.query.items.len == 0) return .none;
                var end = self.query.items.len - 1;
                while (end > 0 and self.query.items[end] & 0xC0 == 0x80) end -= 1;
                self.query.shrinkRetainingCapacity(end);
                return self.refilter();
            },
            else => return .none,
        }
    }

    /// One focus ring: the list's rows, then the option rows.
    fn step(self: *Picker, direction: enum { next, previous }) Action {
        const listed = self.shown.items.len;
        const ring = listed + self.options.items.len;
        if (ring == 0) return .none;
        const at = if (self.focus) |f| listed + f else self.selected;
        const to = switch (direction) {
            .next => (at + 1) % ring,
            .previous => (at + ring - 1) % ring,
        };
        if (to < listed) {
            self.focus = null;
            self.selected = to;
            return .moved;
        }
        self.focus = to - listed;
        return .none;
    }

    fn refilter(self: *Picker) !Action {
        const previous = self.chosen();
        try self.filter();
        self.selected = 0;
        if (previous) |p| for (self.shown.items, 0..) |index, at| if (index == p) {
            self.selected = at;
            break;
        };
        self.focus = null;
        return .moved;
    }

    fn filter(self: *Picker) !void {
        const Hit = struct { index: usize, score: i32 };
        var hits: std.ArrayList(Hit) = .empty;
        defer hits.deinit(self.alloc);
        for (self.items.items, 0..) |item, index| {
            const haystack = if (item.search.len > 0) item.search else item.label;
            const s = fuzzy.score(self.query.items, haystack) orelse continue;
            try hits.append(self.alloc, .{ .index = index, .score = s });
        }
        // Stable: equal scores keep the caller's order (newest first, the
        // registry's order), which an empty query leaves untouched.
        std.mem.sort(Hit, hits.items, {}, struct {
            fn better(_: void, a: Hit, b: Hit) bool {
                return if (a.score != b.score) a.score > b.score else a.index < b.index;
            }
        }.better);
        self.shown.clearRetainingCapacity();
        for (hits.items) |hit| try self.shown.append(self.alloc, hit.index);
        self.scroll = 0;
    }

    fn clearOptions(self: *Picker) void {
        for (self.options.items) |option| {
            for (option.values) |v| self.alloc.free(v);
            self.alloc.free(option.values);
            self.alloc.free(option.name);
        }
        self.options.clearRetainingCapacity();
    }

    /// The rows to paint: title, search, list, options, note, footer.
    pub fn layout(self: *Picker, a: Allocator, options: Layout) ![]const Row {
        var rows: std.ArrayList(Row) = .empty;
        if (!self.isOpen()) return rows.items;
        const th = options.th;
        const ascii = th.glyph_set == .ascii;
        const g = th.glyphs();
        const w = options.width;

        try rows.append(a, try fit(a, w, try a.print("{s}{s}{s}{s}  {d} of {d}{s}", .{ th.paint(.heading), self.title, theme.reset, th.paint(.dim), self.shown.items.len, self.items.items.len, theme.reset })));
        const typed = try sanitized(a, self.query.items);
        try rows.append(a, try fit(a, w, if (self.query.items.len == 0)
            try a.print("{s}{s} type to filter{s}", .{ th.paint(.dim), if (ascii) ">" else "›", theme.reset })
        else
            try a.print("{s} {s}{s}", .{ if (ascii) ">" else "›", typed, if (ascii) "_" else "▏" })));

        // The list, scrolled so the selection is visible.
        const count = self.shown.items.len;
        if (count == 0) try rows.append(a, try fit(a, w, try a.print("  {s}no match{s}", .{ th.paint(.dim), theme.reset })));
        const capacity = @max(options.max_rows, 1);
        const visible = @min(count, capacity);
        if (self.selected < self.scroll) self.scroll = self.selected;
        if (self.selected >= self.scroll + visible) self.scroll = self.selected + 1 - visible;
        if (self.scroll + visible > count) self.scroll = count - visible;
        var label_width: usize = 0;
        for (self.shown.items[self.scroll..][0..visible]) |index| label_width = @max(label_width, view.width(self.items.items[index].label));
        label_width = @min(label_width, w / 2);
        for (self.shown.items[self.scroll..][0..visible], self.scroll..) |index, at| {
            const item = self.items.items[index];
            const lit = at == self.selected and self.focus == null;
            var line: std.Io.Writer.Allocating = .init(a);
            if (lit) try line.writer.writeAll(th.paint(.choice_selected));
            try line.writer.print("{s} {s} ", .{ if (at == self.selected) g.fold_closed else " ", if (item.current) g.done else " " });
            try view.safe(&line.writer, item.label);
            if (item.detail.len > 0) {
                try line.writer.splatByteAll(' ', (label_width -| view.width(item.label)) + 2);
                if (!lit) try line.writer.writeAll(th.paint(.dim));
                try view.safe(&line.writer, item.detail);
                if (!lit) try line.writer.writeAll(theme.reset);
            }
            if (lit) {
                try line.writer.splatByteAll(' ', w -| view.styledWidth(line.written()));
                try line.writer.writeAll(theme.reset);
            }
            try rows.append(a, try fit(a, w, line.written()));
        }
        if (count > visible) try rows.append(a, try fit(a, w, try a.print("    {s}{s} {d} more{s}", .{ th.paint(.dim), g.ellipsis, count - visible, theme.reset })));

        var name_width: usize = 0;
        for (self.options.items) |option| name_width = @max(name_width, option.name.len);
        for (self.options.items, 0..) |option, n| {
            const lit = self.focus == n;
            var line: std.Io.Writer.Allocating = .init(a);
            if (lit) try line.writer.writeAll(th.paint(.choice_selected));
            try line.writer.print("{s} {s}", .{ if (lit) g.fold_closed else " ", option.name });
            try line.writer.splatByteAll(' ', name_width - option.name.len + 2);
            const value = if (option.values.len > 0) option.values[option.index] else "";
            const before = option.index > 0;
            const after = option.index + 1 < option.values.len;
            try line.writer.print("{s} {s} {s}", .{ if (!before) " " else if (ascii) "<" else "‹", value, if (!after) " " else if (ascii) ">" else "›" });
            if (lit) {
                try line.writer.splatByteAll(' ', w -| view.styledWidth(line.written()));
                try line.writer.writeAll(theme.reset);
            }
            try rows.append(a, try fit(a, w, line.written()));
        }
        if (self.note.len > 0) try rows.append(a, try fit(a, w, try a.print("  {s}{s}{s}", .{ th.paint(.dim), try sanitized(a, self.note), theme.reset })));
        if (self.question) |q| {
            try rows.append(a, try fit(a, w, try a.print("  {s}{s} y / n{s}", .{ th.paint(.warning), try sanitized(a, q), theme.reset })));
        } else {
            var footer: std.Io.Writer.Allocating = .init(a);
            const dot = if (ascii) " - " else " · ";
            try footer.writer.print("  {s}{s} move", .{ th.paint(.dim), if (ascii) "Up/Down" else "↑↓" });
            if (self.options.items.len > 0) try footer.writer.print("{s}{s} adjust", .{ dot, if (ascii) "Left/Right" else "←→" });
            try footer.writer.print("{s}Enter choose", .{dot});
            if (self.deletable) try footer.writer.print("{s}Ctrl-D delete", .{dot});
            try footer.writer.print("{s}Esc back{s}", .{ dot, theme.reset });
            try rows.append(a, try fit(a, w, footer.written()));
        }
        return rows.items;
    }
};

fn fit(a: Allocator, width: usize, text: []const u8) !Row {
    const wrapped = try view.wrapStyled(a, text, @max(width, 1), .character);
    return .{ .text = wrapped[0], .raw = true };
}

fn sanitized(a: Allocator, text: []const u8) ![]const u8 {
    var w: std.Io.Writer.Allocating = .init(a);
    try view.safe(&w.writer, text);
    return w.written();
}

fn dupeItem(alloc: Allocator, item: Item) !Item {
    const label = try alloc.dupe(u8, item.label);
    errdefer alloc.free(label);
    const detail = try alloc.dupe(u8, item.detail);
    errdefer alloc.free(detail);
    const search = try alloc.dupe(u8, item.search);
    return .{ .label = label, .detail = detail, .search = search, .current = item.current };
}

fn freeItem(alloc: Allocator, item: Item) void {
    alloc.free(item.label);
    alloc.free(item.detail);
    alloc.free(item.search);
}

// ----- tests -----

const testing = std.testing;

fn joined(a: Allocator, rows: []const Row) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (rows) |row| {
        try out.appendSlice(a, row.text);
        try out.append(a, '\n');
    }
    return out.items;
}

const models = [_]Item{
    .{ .label = "qwen3.8-27b", .detail = "qwen35 · Q4_K_M", .current = true },
    .{ .label = "gemma-4-12b-qat", .detail = "gemma4 · Q4_0" },
    .{ .label = "bonsai", .detail = "qwen35 · Q2" },
};

test "a picker opens on the current item and filters as the query is typed" {
    var p: Picker = .{ .alloc = testing.allocator };
    defer p.deinit();
    try testing.expect(!p.isOpen());
    try p.open("Model", &models, "");
    try testing.expect(p.isOpen());
    try testing.expectEqual(@as(?usize, 0), p.chosen());
    try testing.expectEqual(Action.moved, try p.handle(.down));
    try testing.expectEqual(@as(?usize, 1), p.chosen());
    // Typing narrows the list and keeps the selection when it survives.
    try testing.expectEqual(Action.moved, try p.handle(.{ .text = "gem" }));
    try testing.expectEqual(@as(usize, 1), p.matches().len);
    try testing.expectEqual(@as(?usize, 1), p.chosen());
    try testing.expectEqual(Action.choose, try p.handle(.enter));
    // Nothing matches: nothing to choose, and Backspace widens it again.
    _ = try p.handle(.{ .text = "zz" });
    try testing.expect(p.chosen() == null);
    try testing.expectEqual(Action.none, try p.handle(.enter));
    _ = try p.handle(.backspace);
    _ = try p.handle(.backspace);
    try testing.expectEqual(@as(?usize, 1), p.chosen());
    // Only an item the query shows can be selected.
    try testing.expect(!p.select(2));
    try testing.expect(p.select(1));
    try testing.expectEqual(@as(?usize, 1), p.chosen());
    try testing.expectEqual(Action.close, try p.handle(.escape));
    p.clear();
    try testing.expect(!p.isOpen());
}

test "a query given at open filters before the first frame" {
    var p: Picker = .{ .alloc = testing.allocator };
    defer p.deinit();
    try p.open("Model", &models, "bon");
    try testing.expectEqual(@as(usize, 1), p.matches().len);
    try testing.expectEqual(@as(?usize, 2), p.chosen());
    // Backspace drops a whole UTF-8 character, never half of one.
    _ = try p.handle(.{ .text = "é" });
    _ = try p.handle(.backspace);
    try testing.expectEqualStrings("bon", p.query.items);
}

test "option rows join the focus ring and adjust with left and right, clamped" {
    var p: Picker = .{ .alloc = testing.allocator };
    defer p.deinit();
    try p.open("Model", &models, "");
    try p.setOptions(&.{
        .{ .name = "effort", .values = &.{ "off", "low", "medium" }, .index = 1 },
        .{ .name = "context", .values = &.{ "8K", "16K" }, .index = 1 },
    });
    // On the list, left and right do nothing.
    try testing.expectEqual(Action.none, try p.handle(.right));
    // Up from the first item wraps to the last option row.
    try testing.expectEqual(Action.none, try p.handle(.up));
    try testing.expectEqual(@as(?usize, 1), p.focus);
    try testing.expectEqual(Action.none, try p.handle(.right)); // already at the end
    try testing.expectEqual(Action.adjusted, try p.handle(.left));
    try testing.expectEqual(@as(usize, 0), p.optionIndex(1));
    _ = try p.handle(.up);
    try testing.expectEqual(Action.adjusted, try p.handle(.right));
    try testing.expectEqual(@as(usize, 2), p.optionIndex(0));
    // The list keeps its selection while an option has the focus.
    try testing.expectEqual(@as(?usize, 0), p.chosen());
    try testing.expectEqual(Action.choose, try p.handle(.enter));
    // Down past the last option row returns to the list's first row.
    _ = try p.handle(.down);
    _ = try p.handle(.down);
    try testing.expect(p.focus == null);
}

test "a destructive action asks inside the picker, and only y confirms" {
    var p: Picker = .{ .alloc = testing.allocator };
    defer p.deinit();
    try p.open("Resume", &models, "");
    try testing.expectEqual(Action.none, try p.handle(.{ .ctrl = 'd' })); // not deletable
    p.deletable = true;
    try testing.expectEqual(Action.delete, try p.handle(.{ .ctrl = 'd' }));
    try p.confirm("delete qwen?");
    try testing.expectEqual(Action.declined, try p.handle(.enter));
    try p.confirm("delete qwen?");
    try testing.expectEqual(Action.declined, try p.handle(.escape));
    try p.confirm("delete qwen?");
    try testing.expectEqual(Action.confirmed, try p.handle(.{ .text = "y" }));
    try testing.expect(p.question == null);
}

test "the rows name the title, the count, the current item, the options, and the keys" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var p: Picker = .{ .alloc = testing.allocator };
    defer p.deinit();
    try p.open("Model", &models, "");
    try p.setOptions(&.{.{ .name = "effort", .values = &.{ "off", "low" }, .index = 1 }});
    try p.setNote("a full prefill");
    const th: theme.Theme = .{ .kind = .plain };
    const rows = try p.layout(a, .{ .width = 60, .max_rows = 2, .th = th });
    const text = try joined(a, rows);
    try testing.expect(std.mem.indexOf(u8, text, "Model") != null);
    try testing.expect(std.mem.indexOf(u8, text, "3 of 3") != null);
    try testing.expect(std.mem.indexOf(u8, text, "▸ ● qwen3.8-27b") != null);
    try testing.expect(std.mem.indexOf(u8, text, "1 more") != null); // two rows of three
    try testing.expect(std.mem.indexOf(u8, text, "effort  ‹ low") != null);
    try testing.expect(std.mem.indexOf(u8, text, "a full prefill") != null);
    try testing.expect(std.mem.indexOf(u8, text, "←→ adjust") != null);
    try testing.expect(std.mem.indexOf(u8, text, "Esc back") != null);
    for (rows) |row| try testing.expect(view.styledWidth(row.text) <= 60);
    // A question replaces the footer.
    p.deletable = true;
    try p.confirm("delete it?");
    const asking = try joined(a, try p.layout(a, .{ .width = 60, .max_rows = 4, .th = th }));
    try testing.expect(std.mem.indexOf(u8, asking, "delete it? y / n") != null);
    try testing.expect(std.mem.indexOf(u8, asking, "Esc back") == null);
}

test "control bytes in a label or the query never reach a row" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var p: Picker = .{ .alloc = testing.allocator };
    defer p.deinit();
    try p.open("T", &.{.{ .label = "we\x07ird", .detail = "\x1b[31mred" }}, "");
    _ = try p.handle(.{ .text = "\x1b" });
    const text = try joined(a, try p.layout(a, .{ .width = 40, .max_rows = 4, .th = .{ .kind = .plain } }));
    try testing.expect(std.mem.indexOfScalar(u8, text, 0x07) == null);
    try testing.expect(std.mem.indexOf(u8, text, "\x1b[31m") == null);
}
