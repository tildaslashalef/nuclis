//! Width-aware tables: natural column widths; on a narrow terminal the
//! optional columns go first (highest `drop` first), then truncatable
//! columns shrink, widest first, never below `min` or the title.
//! Identifier columns (`truncate = false`) are never cut. No width (piped
//! output) means no limit. Cells must already be clean (`text.clean`).
const std = @import("std");
const text = @import("text.zig");
const style = @import("style.zig");
const Ui = @import("Ui.zig");

pub const Column = struct {
    /// Upper case by convention (`NAME`, `SIZE`).
    title: []const u8,
    alignment: text.Align = .left,
    /// 0: always shown; higher numbers are dropped first.
    drop: u8 = 0,
    truncate: bool = true,
    min: usize = 4,
};

pub const Cell = struct {
    text: []const u8,
    role: style.Role = .text,
};

const gutter = 2;

/// Rows are slices of `columns.len` cells; an empty cell or `--` reads
/// muted. A column with no text in any row is left out, and a row ends
/// at its last cell with text.
pub fn render(w: *std.Io.Writer, ui: Ui, arena: std.mem.Allocator, indent: usize, columns: []const Column, rows: []const []const Cell) !void {
    const n = columns.len;
    const widths = try arena.alloc(usize, n);
    const shown = try arena.alloc(bool, n);
    for (columns, 0..) |c, i| {
        widths[i] = text.width(c.title);
        shown[i] = rows.len == 0;
    }
    for (rows) |row| {
        for (row[0..n], 0..) |cell, i| {
            widths[i] = @max(widths[i], text.width(cell.text));
            if (cell.text.len > 0) shown[i] = true;
        }
    }
    if (ui.width) |full| {
        const limit = full -| indent;
        while (total(widths, shown) > limit) {
            var victim: ?usize = null;
            for (columns, 0..) |c, i| {
                if (!shown[i] or c.drop == 0) continue;
                if (victim == null or c.drop >= columns[victim.?].drop) victim = i;
            }
            shown[victim orelse break] = false;
        }
        while (total(widths, shown) > limit) {
            var victim: ?usize = null;
            for (columns, 0..) |c, i| {
                if (!shown[i] or !c.truncate or widths[i] <= floor(c)) continue;
                if (victim == null or widths[i] > widths[victim.?]) victim = i;
            }
            const v = victim orelse break;
            widths[v] -= @min(widths[v] - floor(columns[v]), total(widths, shown) - limit);
        }
    }
    const last = lastShown(shown);
    try w.splatByteAll(' ', indent);
    for (columns, 0..) |c, i| {
        if (!shown[i]) continue;
        try w.writeAll(ui.on(.title));
        try text.writeFit(w, c.title, widths[i], c.alignment, ui.glyphs.ellipsis, i != last);
        try w.writeAll(ui.off(.title));
        if (i != last) try w.splatByteAll(' ', gutter);
    }
    try w.writeByte('\n');
    for (rows) |row| {
        try w.splatByteAll(' ', indent);
        var row_last: usize = 0;
        for (row[0..n], 0..) |cell, i| {
            if (shown[i] and cell.text.len > 0) row_last = i;
        }
        for (columns, 0..) |c, i| {
            if (!shown[i]) continue;
            if (i > row_last) break;
            const cell = row[i];
            const role: style.Role = if (cell.text.len == 0 or std.mem.eql(u8, cell.text, "--")) .muted else cell.role;
            try w.writeAll(ui.on(role));
            try text.writeFit(w, cell.text, widths[i], c.alignment, ui.glyphs.ellipsis, i != row_last);
            try w.writeAll(ui.off(role));
            if (i != row_last) try w.splatByteAll(' ', gutter);
        }
        try w.writeByte('\n');
    }
}

fn floor(c: Column) usize {
    return @max(c.min, text.width(c.title));
}

fn total(widths: []const usize, shown: []const bool) usize {
    var sum: usize = 0;
    var count: usize = 0;
    for (widths, shown) |wd, s| {
        if (!s) continue;
        sum += wd;
        count += 1;
    }
    return sum + gutter * (if (count > 0) count - 1 else 0);
}

fn lastShown(shown: []const bool) usize {
    var last: usize = 0;
    for (shown, 0..) |s, i| {
        if (s) last = i;
    }
    return last;
}

const testing = std.testing;

const test_columns = [_]Column{
    .{ .title = "NAME", .truncate = false },
    .{ .title = "QUANT" },
    .{ .title = "SIZE", .alignment = .right, .truncate = false },
    .{ .title = "STATUS", .drop = 1 },
};

const test_rows = [_][]const Cell{
    &.{ .{ .text = "qwen3.8-27b" }, .{ .text = "UD-Q4_K_M" }, .{ .text = "16.56 GB" }, .{ .text = "✓ installed", .role = .ok } },
    &.{ .{ .text = "laya" }, .{ .text = "F16" }, .{ .text = "842.6 MB" }, .{ .text = "--" } },
};

test "a table aligns, and narrows by dropping optional columns first" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    var buf: std.Io.Writer.Allocating = .init(testing.allocator);
    defer buf.deinit();
    try render(&buf.writer, .plain, arena_state.allocator(), 0, &test_columns, &test_rows);
    try testing.expectEqualStrings(
        \\NAME         QUANT          SIZE  STATUS
        \\qwen3.8-27b  UD-Q4_K_M  16.56 GB  ✓ installed
        \\laya         F16        842.6 MB  --
        \\
    , buf.written());
    buf.clearRetainingCapacity();
    // Forty cells: STATUS, the optional column, goes.
    try render(&buf.writer, .{ .width = 40 }, arena_state.allocator(), 0, &test_columns, &test_rows);
    try testing.expectEqualStrings(
        \\NAME         QUANT          SIZE
        \\qwen3.8-27b  UD-Q4_K_M  16.56 GB
        \\laya         F16        842.6 MB
        \\
    , buf.written());
    buf.clearRetainingCapacity();
    // Thirty, indented two: QUANT shrinks; the name and size are never cut.
    try render(&buf.writer, .{ .width = 30 }, arena_state.allocator(), 2, &test_columns, &test_rows);
    try testing.expectEqualStrings(
        \\  NAME         QUANT      SIZE
        \\  qwen3.8-27b  UD-Q…  16.56 GB
        \\  laya         F16    842.6 MB
        \\
    , buf.written());
}
