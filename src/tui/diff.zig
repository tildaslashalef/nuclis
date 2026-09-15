//! A pure line diff from two byte strings, and nothing else.
//!
//! One function, `compute`, turns an old and a new content into **structured
//! rows** — old/new line number, kind, text, and an optional changed span —
//! and, from the same edit script, the compact unified text. The transcript
//! renders the rows (width-adaptively); the agent hands the unified text to
//! the model and the session file. Neither form is a description of the other,
//! which is the point: the two can never disagree about what changed.
//!
//! This module has no theme, no terminal, and no I/O: it allocates owned
//! bytes and is tested against fixtures (insertion, deletion, replacement,
//! context, CRLF, a missing trailing newline). `tui.transcript` adds the
//! colour; `src/agent/` decides where the contents come from.
//!
//! Algorithm. Lines are split on `\n` only, so a CRLF file keeps its `\r` and
//! diffs cleanly against itself. Common leading and trailing lines are
//! stripped (the usual case: a small local edit), and the remaining middle is
//! diffed with an LCS table bounded by `max_cells`; a middle too large for the
//! table degrades to "all removed, all added" rather than spending unbounded
//! time. Rows are materialised only for the changed hunks plus `context_lines`
//! of unchanged context, so a one-line edit in a large file stays small.
//!
//! Zig note. `Row.text` is owned by whoever creates it: `compute` allocates
//! copies, and `cloneRows` copies a borrowed event's rows into a block. That
//! keeps the event seam's borrow-for-the-call rule intact (see `event.zig`)
//! without this module knowing it exists.
const std = @import("std");

const Allocator = std.mem.Allocator;

/// What a row represents.
pub const Kind = enum { context, add, remove };

/// A byte range inside a row's text: the part that differs within a paired
/// removal and addition. Absent when the whole line is new or gone.
pub const Span = struct { start: usize, len: usize };

/// One line of the diff. The text never includes the line's `\n` (or `\r`).
pub const Row = struct {
    /// 1-based line in the old content; null for a pure addition.
    old_line: ?usize = null,
    /// 1-based line in the new content; null for a pure removal.
    new_line: ?usize = null,
    kind: Kind,
    text: []const u8,
    /// Set on a paired removal/addition: the bytes that changed.
    change: ?Span = null,
};

/// A computed diff: the rows to render, and the unified text to hand to the
/// model and the session file. Both are owned.
pub const Diff = struct {
    rows: []Row,
    unified: []u8,

    pub fn deinit(self: *Diff, alloc: Allocator) void {
        freeRows(alloc, self.rows);
        alloc.free(self.unified);
        self.* = undefined;
    }
};

/// Copies borrowed rows (an event's payload) into owned rows (a transcript
/// block). The text of every row is duplicated.
pub fn cloneRows(alloc: Allocator, rows: []const Row) ![]Row {
    const owned = try alloc.alloc(Row, rows.len);
    var done: usize = 0;
    errdefer {
        for (owned[0..done]) |row| alloc.free(row.text);
        alloc.free(owned);
    }
    for (rows, 0..) |row, i| {
        owned[i] = row;
        owned[i].text = try alloc.dupe(u8, row.text);
        done = i + 1;
    }
    return owned;
}

/// Frees the text of every row and the row array itself.
pub fn freeRows(alloc: Allocator, rows: []Row) void {
    for (rows) |row| alloc.free(row.text);
    alloc.free(rows);
}

/// Unchanged lines kept on each side of a hunk.
pub const context_lines: usize = 3;
/// Largest LCS middle (cells) before degrading to a whole-middle replacement.
/// Bounds the diff of two large, wholly different files. At 8 bytes a cell
/// this is 8 MiB, paid only for a genuinely large edit.
const max_cells: usize = 1_000_000;

/// Diffs `old` against `new`. The result owns its bytes; free it with
/// `Diff.deinit`.
pub fn compute(alloc: Allocator, old: []const u8, new: []const u8) Allocator.Error!Diff {
    const old_lines = try splitLines(alloc, old);
    defer alloc.free(old_lines.items);
    const new_lines = try splitLines(alloc, new);
    defer alloc.free(new_lines.items);

    const ops = try buildOps(alloc, old_lines.items, new_lines.items);
    defer alloc.free(ops);

    var ranges: std.ArrayList([2]usize) = .empty;
    defer ranges.deinit(alloc);
    var i: usize = 0;
    while (i < ops.len) {
        if (ops[i].kind == .context) {
            i += 1;
            continue;
        }
        const start = if (i >= context_lines) i - context_lines else 0;
        const end = @min(ops.len, i + 1 + context_lines);
        if (ranges.items.len > 0) {
            const last = &ranges.items[ranges.items.len - 1];
            if (last[1] >= start) {
                last[1] = @max(last[1], end);
                i = end;
                continue;
            }
        }
        try ranges.append(alloc, .{ start, end });
        i = end;
    }

    var rows: std.ArrayList(Row) = .empty;
    errdefer {
        freeRows(alloc, rows.items);
        rows.deinit(alloc);
    }
    var unified: std.ArrayList(u8) = .empty;
    errdefer unified.deinit(alloc);

    for (ranges.items) |range| {
        const start = range[0];
        const end = range[1];
        var old_start: usize = 0;
        var new_start: usize = 0;
        var old_count: usize = 0;
        var new_count: usize = 0;
        for (ops[start..end]) |op| {
            if (op.oi) |x| {
                if (old_count == 0) old_start = x + 1;
                old_count += 1;
            }
            if (op.ni) |x| {
                if (new_count == 0) new_start = x + 1;
                new_count += 1;
            }
        }
        var header: [64]u8 = undefined;
        try unified.appendSlice(alloc, std.fmt.bufPrint(&header, "@@ -{d},{d} +{d},{d} @@\n", .{ old_start, old_count, new_start, new_count }) catch unreachable);
        for (ops[start..end]) |op| {
            const text = switch (op.kind) {
                .context, .remove => old_lines.items[op.oi.?],
                .add => new_lines.items[op.ni.?],
            };
            try rows.append(alloc, .{
                .old_line = if (op.oi) |x| x + 1 else null,
                .new_line = if (op.ni) |x| x + 1 else null,
                .kind = op.kind,
                .text = try alloc.dupe(u8, text),
            });
            const sign: u8 = switch (op.kind) {
                .context => ' ',
                .remove => '-',
                .add => '+',
            };
            try unified.append(alloc, sign);
            try unified.appendSlice(alloc, text);
            try unified.append(alloc, '\n');
            const last_old = if (op.oi) |x| x + 1 == old_lines.items.len and !old_lines.trailing_newline else false;
            const last_new = if (op.ni) |x| x + 1 == new_lines.items.len and !new_lines.trailing_newline else false;
            if (last_old or last_new) try unified.appendSlice(alloc, "\\ No newline at end of file\n");
        }
    }

    pairChanges(rows.items);
    return .{ .rows = try rows.toOwnedSlice(alloc), .unified = try unified.toOwnedSlice(alloc) };
}

const Op = struct {
    kind: Kind,
    oi: ?usize = null,
    ni: ?usize = null,
};

/// The edit script for the whole file: common prefix and suffix as context,
/// the middle from the LCS (or the bounded fallback).
fn buildOps(alloc: Allocator, old: []const []const u8, new: []const []const u8) Allocator.Error![]Op {
    var ops: std.ArrayList(Op) = .empty;
    errdefer ops.deinit(alloc);

    var lo: usize = 0;
    while (lo < old.len and lo < new.len and std.mem.eql(u8, old[lo], new[lo])) : (lo += 1) {
        try ops.append(alloc, .{ .kind = .context, .oi = lo, .ni = lo });
    }
    var ho = old.len;
    var hn = new.len;
    while (ho > lo and hn > lo and std.mem.eql(u8, old[ho - 1], new[hn - 1])) {
        ho -= 1;
        hn -= 1;
    }
    try appendMiddle(alloc, &ops, old[lo..ho], new[lo..hn], lo);
    var k: usize = 0;
    while (k < old.len - ho) : (k += 1) {
        try ops.append(alloc, .{ .kind = .context, .oi = ho + k, .ni = hn + k });
    }
    return ops.toOwnedSlice(alloc);
}

fn appendMiddle(alloc: Allocator, ops: *std.ArrayList(Op), old: []const []const u8, new: []const []const u8, base: usize) Allocator.Error!void {
    const m = old.len;
    const n = new.len;
    if (m == 0 or n == 0 or (m + 1) * (n + 1) > max_cells) {
        // A bounded fallback: everything old is removed, everything new added.
        for (0..m) |i| try ops.append(alloc, .{ .kind = .remove, .oi = base + i });
        for (0..n) |j| try ops.append(alloc, .{ .kind = .add, .ni = base + j });
        return;
    }
    const cols = n + 1;
    const table = try alloc.alloc(u32, (m + 1) * cols);
    defer alloc.free(table);
    @memset(table, 0);
    var i: usize = 1;
    while (i <= m) : (i += 1) {
        var j: usize = 1;
        while (j <= n) : (j += 1) {
            table[i * cols + j] = if (std.mem.eql(u8, old[i - 1], new[j - 1]))
                table[(i - 1) * cols + (j - 1)] + 1
            else
                @max(table[(i - 1) * cols + j], table[i * cols + (j - 1)]);
        }
    }
    // Walk back to an ops list in reverse, then reverse it into place.
    const start = ops.items.len;
    var a = m;
    var b = n;
    while (a > 0 or b > 0) {
        if (a > 0 and b > 0 and std.mem.eql(u8, old[a - 1], new[b - 1])) {
            try ops.append(alloc, .{ .kind = .context, .oi = base + a - 1, .ni = base + b - 1 });
            a -= 1;
            b -= 1;
        } else if (b > 0 and (a == 0 or table[a * cols + (b - 1)] >= table[(a - 1) * cols + b])) {
            try ops.append(alloc, .{ .kind = .add, .ni = base + b - 1 });
            b -= 1;
        } else {
            try ops.append(alloc, .{ .kind = .remove, .oi = base + a - 1 });
            a -= 1;
        }
    }
    std.mem.reverse(Op, ops.items[start..]);
    normalizeRuns(ops, start);
}

/// Within each run of changes, puts removals before additions: the unified
/// and side-by-side forms both read top-down as "gone, then arrived".
fn normalizeRuns(ops: *std.ArrayList(Op), start: usize) void {
    var i = start;
    while (i < ops.items.len) {
        if (ops.items[i].kind == .context) {
            i += 1;
            continue;
        }
        var end = i;
        while (end < ops.items.len and ops.items[end].kind != .context) end += 1;
        var j = i;
        var write = i;
        // Removals first, in order…
        while (j < end) : (j += 1) {
            if (ops.items[j].kind == .remove) {
                ops.items[write] = ops.items[j];
                write += 1;
            }
        }
        // …then additions, in order.
        j = i;
        while (j < end) : (j += 1) {
            if (ops.items[j].kind == .add) {
                ops.items[write] = ops.items[j];
                write += 1;
            }
        }
        i = end;
    }
}

/// Pairs consecutive removals with additions and marks the bytes that differ.
fn pairChanges(rows: []Row) void {
    var i: usize = 0;
    while (i < rows.len) {
        if (rows[i].kind != .remove) {
            i += 1;
            continue;
        }
        var removes_end = i;
        while (removes_end < rows.len and rows[removes_end].kind == .remove) removes_end += 1;
        var adds_end = removes_end;
        while (adds_end < rows.len and rows[adds_end].kind == .add) adds_end += 1;
        const pairs = @min(removes_end - i, adds_end - removes_end);
        for (0..pairs) |k| {
            const changed = changedSpan(rows[i + k].text, rows[removes_end + k].text);
            rows[i + k].change = changed.old;
            rows[removes_end + k].change = changed.new;
        }
        i = adds_end;
    }
}

fn changedSpan(old: []const u8, new: []const u8) struct { old: ?Span, new: ?Span } {
    var prefix: usize = 0;
    while (prefix < old.len and prefix < new.len and old[prefix] == new[prefix]) prefix += 1;
    var suffix: usize = 0;
    while (suffix < old.len - prefix and suffix < new.len - prefix and old[old.len - 1 - suffix] == new[new.len - 1 - suffix]) suffix += 1;
    const old_len = old.len - prefix - suffix;
    const new_len = new.len - prefix - suffix;
    return .{
        .old = if (old_len > 0) .{ .start = prefix, .len = old_len } else null,
        .new = if (new_len > 0) .{ .start = prefix, .len = new_len } else null,
    };
}

const Lines = struct { items: []const []const u8, trailing_newline: bool };

/// Splits on `\n` only, keeping any `\r`. A trailing newline yields no empty
/// final line; it is recorded as `trailing_newline` instead.
fn splitLines(alloc: Allocator, content: []const u8) Allocator.Error!Lines {
    if (content.len == 0) return .{ .items = try alloc.alloc([]const u8, 0), .trailing_newline = false };
    var list: std.ArrayList([]const u8) = .empty;
    errdefer list.deinit(alloc);
    var it = std.mem.splitScalar(u8, content, '\n');
    while (it.next()) |line| try list.append(alloc, line);
    var trailing = false;
    if (content[content.len - 1] == '\n') {
        trailing = true;
        _ = list.pop();
    }
    return .{ .items = try list.toOwnedSlice(alloc), .trailing_newline = trailing };
}

// ----- tests -----

const testing = std.testing;

fn expectKind(diff: Diff, index: usize, kind: Kind, text: []const u8) !void {
    try testing.expect(index < diff.rows.len);
    try testing.expectEqual(kind, diff.rows[index].kind);
    try testing.expectEqualStrings(text, diff.rows[index].text);
}

test "no change is no rows and no unified text" {
    const d = try compute(testing.allocator, "a\nb\n", "a\nb\n");
    defer {
        var mut = d;
        mut.deinit(testing.allocator);
    }
    try testing.expectEqual(@as(usize, 0), d.rows.len);
    try testing.expectEqualStrings("", d.unified);
}

test "a replacement pairs the old and new line and marks the bytes that changed" {
    const d = try compute(testing.allocator, "keep\nold value\nkeep\n", "keep\nnew value\nkeep\n");
    defer {
        var mut = d;
        mut.deinit(testing.allocator);
    }
    try testing.expectEqual(@as(usize, 4), d.rows.len);
    try expectKind(d, 0, .context, "keep");
    try expectKind(d, 1, .remove, "old value");
    try expectKind(d, 2, .add, "new value");
    try testing.expectEqual(@as(?usize, 2), d.rows[1].old_line);
    try testing.expect(d.rows[1].new_line == null);
    try testing.expect(d.rows[2].old_line == null);
    try testing.expectEqual(@as(?usize, 2), d.rows[2].new_line);
    // "old value" vs "new value": three bytes differ at offset 0.
    try testing.expectEqual(Span{ .start = 0, .len = 3 }, d.rows[1].change.?);
    try testing.expectEqual(Span{ .start = 0, .len = 3 }, d.rows[2].change.?);
    try testing.expect(std.mem.indexOf(u8, d.unified, "@@ -1,3 +1,3 @@") != null);
    try testing.expect(std.mem.indexOf(u8, d.unified, "-old value\n+new value\n") != null);
}

test "an insertion is an addition with context on both sides" {
    const d = try compute(testing.allocator, "one\ntwo\n", "one\nand a half\ntwo\n");
    defer {
        var mut = d;
        mut.deinit(testing.allocator);
    }
    try testing.expectEqual(@as(usize, 3), d.rows.len);
    try expectKind(d, 0, .context, "one");
    try expectKind(d, 1, .add, "and a half");
    try testing.expectEqual(@as(?usize, 2), d.rows[1].new_line);
    try expectKind(d, 2, .context, "two");
    try testing.expect(std.mem.indexOf(u8, d.unified, "+and a half\n") != null);
}

test "a deletion is a removal with context on both sides" {
    const d = try compute(testing.allocator, "one\ntwo\nthree\n", "one\nthree\n");
    defer {
        var mut = d;
        mut.deinit(testing.allocator);
    }
    try testing.expectEqual(@as(usize, 3), d.rows.len);
    try expectKind(d, 1, .remove, "two");
    try testing.expectEqual(@as(?usize, 2), d.rows[1].old_line);
    try testing.expect(d.rows[1].new_line == null);
    try testing.expect(std.mem.indexOf(u8, d.unified, "-two\n") != null);
}

test "context is elided beyond the hunk, so a large unchanged file stays small" {
    var old: std.Io.Writer.Allocating = .init(testing.allocator);
    defer old.deinit();
    var new: std.Io.Writer.Allocating = .init(testing.allocator);
    defer new.deinit();
    for (0..100) |i| {
        try old.writer.print("line {d}\n", .{i});
        if (i == 50) {
            try new.writer.writeAll("changed\n");
        } else {
            try new.writer.print("line {d}\n", .{i});
        }
    }
    const d = try compute(testing.allocator, old.written(), new.written());
    defer {
        var mut = d;
        mut.deinit(testing.allocator);
    }
    // Three context lines on each side of the one changed line.
    try testing.expectEqual(@as(usize, 7), d.rows.len);
    try testing.expectEqual(Kind.remove, d.rows[3].kind);
    try testing.expectEqual(Kind.add, d.rows[4].kind);
    try testing.expect(std.mem.indexOf(u8, d.unified, "@@ -48,6 +48,6 @@") != null);
}

test "a CRLF file diffs cleanly against itself" {
    const d = try compute(testing.allocator, "a\r\nb\r\n", "a\r\nb\r\n");
    defer {
        var mut = d;
        mut.deinit(testing.allocator);
    }
    try testing.expectEqual(@as(usize, 0), d.rows.len);
}

test "a file without a trailing newline is marked in the unified text" {
    const d = try compute(testing.allocator, "one\ntwo", "one\n2");
    defer {
        var mut = d;
        mut.deinit(testing.allocator);
    }
    try testing.expectEqual(@as(usize, 3), d.rows.len);
    try testing.expect(std.mem.indexOf(u8, d.unified, "-two\n\\ No newline at end of file\n") != null);
    try testing.expect(std.mem.indexOf(u8, d.unified, "+2\n\\ No newline at end of file\n") != null);
}

test "a wholly different large middle degrades without unbounded work" {
    // 2000 lines each, no common prefix or suffix: over the cell cap.
    var old: std.Io.Writer.Allocating = .init(testing.allocator);
    defer old.deinit();
    var new: std.Io.Writer.Allocating = .init(testing.allocator);
    defer new.deinit();
    for (0..2000) |i| {
        try old.writer.print("old {d}\n", .{i});
        try new.writer.print("new {d}\n", .{i});
    }
    const d = try compute(testing.allocator, old.written(), new.written());
    defer {
        var mut = d;
        mut.deinit(testing.allocator);
    }
    // Context lines around the first change pull in the single hunk; the
    // whole thing is still reported as removed then added.
    try testing.expect(d.rows.len > 0);
    try testing.expect(std.mem.indexOf(u8, d.unified, "-old 0\n") != null);
    try testing.expect(std.mem.indexOf(u8, d.unified, "+new 0\n") != null);
}

test "cloneRows owns its text and freeRows releases it" {
    const d = try compute(testing.allocator, "a\n", "b\n");
    defer {
        var mut = d;
        mut.deinit(testing.allocator);
    }
    const owned = try cloneRows(testing.allocator, d.rows);
    try testing.expectEqualStrings("a", owned[0].text);
    freeRows(testing.allocator, owned);
}
