//! `glob` — workspace-relative file paths matching one pattern.
//!
//! Supports `*` (any run within a segment), `?` (one character), `[...]`
//! classes, and `**` (zero or more path segments). Hidden entries are skipped
//! unless the pattern itself names one (starts with `.` or contains `/.`).
//! At most `max_results` paths, sorted, with no symlink following — the host
//! constants keep a deep tree from consuming the turn.
const std = @import("std");
const root = @import("root.zig");

pub const tool: root.Tool = .{
    .name = "glob",
    .description = "List workspace-relative file paths matching one glob pattern (`*`, `?`, `[...]`, `**`). Hidden entries are excluded unless the pattern names them.",
    .parameters = "{\"type\":\"object\",\"properties\":{\"pattern\":{\"type\":\"string\"}},\"required\":[\"pattern\"]}",
    .label = "Listing",
    .display = "Glob",
    .subject = "pattern",
    .run = run,
};

const max_results: usize = 200;
/// Paths collected before sorting: the shown 200 are the first sorted ones
/// of these, and a walk that reaches this many stops.
const max_collected: usize = 10_000;

const Args = struct { pattern: []const u8 };

fn run(workspace: root.Workspace, alloc: std.mem.Allocator, arguments: []const u8) std.mem.Allocator.Error!root.Result {
    const parsed = std.json.parseFromSlice(Args, alloc, arguments, .{ .ignore_unknown_fields = true }) catch {
        return root.fail(alloc, "glob: arguments must be a JSON object with a \"pattern\" string", .{});
    };
    defer parsed.deinit();
    const pattern = parsed.value.pattern;
    if (pattern.len == 0) return root.fail(alloc, "glob: pattern is empty", .{});

    var found: std.ArrayList([]u8) = .empty;
    defer {
        for (found.items) |path| alloc.free(path);
        found.deinit(alloc);
    }
    var walker: Walker = .{ .alloc = alloc, .io = workspace.io, .pattern = pattern, .found = &found, .allow_hidden = namesHidden(pattern) };
    try walker.walk(workspace.dir, "");
    std.mem.sort([]u8, found.items, {}, lessThan);

    const shown = found.items[0..@min(found.items.len, max_results)];
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    for (shown, 0..) |path, i| {
        if (i != 0) try out.append(alloc, '\n');
        try out.appendSlice(alloc, path);
    }
    const truncated = found.items.len > max_results;
    const summary = try summarize(alloc, found.items.len, walker.capped, walker.unreadable);
    errdefer alloc.free(summary);
    // The model reads the summary only when the list alone would mislead it.
    if (truncated or walker.unreadable > 0) try out.print(alloc, "{s}[{s}]", .{ if (out.items.len > 0) "\n" else "", summary });
    return .{ .text = try out.toOwnedSlice(alloc), .truncated = truncated or walker.capped, .summary = summary };
}

/// The detail row: how many paths, whether the bound cut the list, and the
/// directories that could not be read.
fn summarize(alloc: std.mem.Allocator, count: usize, capped: bool, unreadable: usize) std.mem.Allocator.Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    switch (count) {
        0 => try out.appendSlice(alloc, "no files"),
        1 => try out.appendSlice(alloc, "1 file"),
        else => try out.print(alloc, "{s}{d} files", .{ if (capped) "over " else "", count }),
    }
    if (count > max_results) try out.print(alloc, " · first {d} shown; narrow the pattern", .{max_results});
    if (unreadable > 0) try out.print(alloc, " · {d} director{s} could not be read", .{ unreadable, if (unreadable == 1) "y" else "ies" });
    return out.toOwnedSlice(alloc);
}

fn lessThan(_: void, a: []u8, b: []u8) bool {
    return std.mem.lessThan(u8, a, b);
}

/// Whether the pattern asks for hidden entries: a leading `.` or a `/.`
/// segment start.
fn namesHidden(pattern: []const u8) bool {
    return std.mem.startsWith(u8, pattern, ".") or std.mem.indexOf(u8, pattern, "/.") != null;
}

const Walker = struct {
    alloc: std.mem.Allocator,
    io: std.Io,
    pattern: []const u8,
    found: *std.ArrayList([]u8),
    allow_hidden: bool,
    collect_limit: usize = max_collected,
    /// The walk stopped at `collect_limit`.
    capped: bool = false,
    unreadable: usize = 0,

    fn walk(self: *Walker, dir: std.Io.Dir, prefix: []const u8) std.mem.Allocator.Error!void {
        var iterable = dir.openDir(self.io, if (prefix.len == 0) "." else prefix, .{ .iterate = true }) catch {
            self.unreadable += 1;
            return;
        };
        defer iterable.close(self.io);
        var it = iterable.iterate();
        while (true) {
            const entry = (it.next(self.io) catch {
                self.unreadable += 1;
                return;
            }) orelse return;
            if (self.found.items.len >= self.collect_limit) {
                self.capped = true;
                return;
            }
            if (entry.name.len == 0) continue;
            if (entry.name[0] == '.' and !self.allow_hidden) continue;
            const path = try std.fs.path.join(self.alloc, &.{ prefix, entry.name });
            defer self.alloc.free(path);
            switch (entry.kind) {
                .directory => try self.walk(dir, path),
                .file => if (matchPattern(self.pattern, path)) {
                    const owned = try self.alloc.dupe(u8, path);
                    errdefer self.alloc.free(owned);
                    try self.found.append(self.alloc, owned);
                },
                else => {},
            }
        }
    }
};

// ----- pattern matching -----

/// Whether a workspace-relative `path` matches `pattern` (the `glob` syntax:
/// `*` and `?` within a segment, `[...]`, `**` across segments).
pub fn matchPattern(pattern: []const u8, path: []const u8) bool {
    return matchFrom(pattern, 0, path, 0);
}

fn segmentEnd(text: []const u8, start: usize) usize {
    return std.mem.indexOfScalarPos(u8, text, start, '/') orelse text.len;
}

fn afterSegment(text: []const u8, end: usize) usize {
    return if (end < text.len) end + 1 else end;
}

fn matchFrom(pattern: []const u8, pi: usize, path: []const u8, si: usize) bool {
    if (pi >= pattern.len) return si >= path.len;
    const pe = segmentEnd(pattern, pi);
    if (std.mem.eql(u8, pattern[pi..pe], "**")) {
        const next = afterSegment(pattern, pe);
        var k = si;
        while (true) {
            if (matchFrom(pattern, next, path, k)) return true;
            if (k >= path.len) return false;
            k = afterSegment(path, segmentEnd(path, k));
        }
    }
    if (si >= path.len) return false;
    const se = segmentEnd(path, si);
    if (!matchSegment(pattern[pi..pe], path[si..se])) return false;
    return matchFrom(pattern, afterSegment(pattern, pe), path, afterSegment(path, se));
}

/// One path segment: `*`, `?`, and `[...]` within it.
fn matchSegment(pattern: []const u8, name: []const u8) bool {
    var pi: usize = 0;
    var ni: usize = 0;
    var star: ?usize = null;
    var star_name: usize = 0;
    while (ni < name.len) {
        if (pi < pattern.len and pattern[pi] == '*') {
            star = pi;
            star_name = ni;
            pi += 1;
            continue;
        }
        if (pi < pattern.len and matchOne(pattern, &pi, name[ni])) {
            ni += 1;
            continue;
        }
        if (star) |at| {
            pi = at + 1;
            star_name += 1;
            ni = star_name;
            continue;
        }
        return false;
    }
    while (pi < pattern.len and pattern[pi] == '*') pi += 1;
    return pi == pattern.len;
}

/// Whether `pattern[pi]` matches `ch`, advancing `pi` past one pattern unit.
fn matchOne(pattern: []const u8, pi: *usize, ch: u8) bool {
    const c = pattern[pi.*];
    if (c == '?') {
        pi.* += 1;
        return true;
    }
    if (c != '[') {
        pi.* += 1;
        return c == ch;
    }
    const close = std.mem.indexOfScalarPos(u8, pattern, pi.* + 1, ']') orelse {
        pi.* += 1;
        return c == ch;
    };
    var i = pi.* + 1;
    var negate = false;
    if (i < close and (pattern[i] == '!' or pattern[i] == '^')) {
        negate = true;
        i += 1;
    }
    var matched = false;
    while (i < close) {
        if (i + 2 < close and pattern[i + 1] == '-') {
            if (ch >= pattern[i] and ch <= pattern[i + 2]) matched = true;
            i += 3;
        } else {
            if (ch == pattern[i]) matched = true;
            i += 1;
        }
    }
    pi.* = close + 1;
    return matched != negate;
}

// ----- tests -----

const testing = std.testing;

test "segment matching covers stars, questions, and classes" {
    try testing.expect(matchPattern("*.zig", "main.zig"));
    try testing.expect(!matchPattern("*.zig", "src/main.zig"));
    try testing.expect(matchPattern("src/*.zig", "src/main.zig"));
    try testing.expect(matchPattern("src/**/*.zig", "src/a/b/main.zig"));
    try testing.expect(matchPattern("src/**/*.zig", "src/main.zig"));
    try testing.expect(matchPattern("a?c", "abc"));
    try testing.expect(!matchPattern("a?c", "ac"));
    try testing.expect(matchPattern("[a-c]x", "bx"));
    try testing.expect(!matchPattern("[!a-c]x", "bx"));
}

test "glob lists matching files, sorted and bounded" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const root_path = try tmp.dir.realPathFileAlloc(testing.io, ".", testing.allocator);
    defer testing.allocator.free(root_path);
    const ws: root.Workspace = .{ .io = testing.io, .dir = tmp.dir, .root = root_path };
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "b.zig", .data = "" });
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "a.zig", .data = "" });
    try tmp.dir.writeFile(testing.io, .{ .sub_path = ".hidden.zig", .data = "" });
    try tmp.dir.createDir(testing.io, "sub", .default_dir);
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "sub/c.zig", .data = "" });
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "sub/notes.txt", .data = "" });

    var result = try run(ws, testing.allocator, "{\"pattern\":\"**/*.zig\"}");
    defer result.deinit(testing.allocator);
    try testing.expectEqualStrings("a.zig\nb.zig\nsub/c.zig", result.text);
    try testing.expect(!result.truncated);

    var hidden = try run(ws, testing.allocator, "{\"pattern\":\".*.zig\"}");
    defer hidden.deinit(testing.allocator);
    try testing.expectEqualStrings(".hidden.zig", hidden.text);

    var bad = try run(ws, testing.allocator, "{}");
    defer bad.deinit(testing.allocator);
    try testing.expect(bad.is_error);
}

test "glob shows the first 200 sorted, and says when more matched" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const root_path = try tmp.dir.realPathFileAlloc(testing.io, ".", testing.allocator);
    defer testing.allocator.free(root_path);
    const ws: root.Workspace = .{ .io = testing.io, .dir = tmp.dir, .root = root_path };
    var name: [16]u8 = undefined;
    // Exactly the bound is not truncated.
    for (0..max_results) |i| try tmp.dir.writeFile(testing.io, .{ .sub_path = try std.fmt.bufPrint(&name, "f{d:0>4}.txt", .{i}), .data = "" });
    var exact = try run(ws, testing.allocator, "{\"pattern\":\"*.txt\"}");
    defer exact.deinit(testing.allocator);
    try testing.expect(!exact.truncated);
    try testing.expectEqual(@as(usize, max_results), std.mem.count(u8, exact.text, "\n") + 1);
    // One more: the first 200 in sorted order, marked.
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "a.txt", .data = "" });
    var over = try run(ws, testing.allocator, "{\"pattern\":\"*.txt\"}");
    defer over.deinit(testing.allocator);
    try testing.expect(over.truncated);
    try testing.expect(std.mem.startsWith(u8, over.text, "a.txt\nf0000.txt\n"));
    try testing.expect(std.mem.indexOf(u8, over.text, "f0199.txt") == null);
    try testing.expect(std.mem.endsWith(u8, over.text, "[201 files · first 200 shown; narrow the pattern]"));
}
