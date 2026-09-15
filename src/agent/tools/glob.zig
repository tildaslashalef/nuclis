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
    .verb = "Glob",
    .subject = "pattern",
    .params = &.{"pattern"},
    .run = run,
};

const max_results: usize = 200;

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
    walker.walk(workspace.dir, "");
    std.mem.sort([]u8, found.items, {}, lessThan);

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    for (found.items, 0..) |path, i| {
        if (i != 0) try out.append(alloc, '\n');
        try out.appendSlice(alloc, path);
    }
    return .{ .text = try out.toOwnedSlice(alloc), .truncated = found.items.len == max_results };
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

    fn walk(self: *Walker, dir: std.Io.Dir, prefix: []const u8) void {
        var iterable = dir.openDir(self.io, if (prefix.len == 0) "." else prefix, .{ .iterate = true }) catch return;
        defer iterable.close(self.io);
        var it = iterable.iterate();
        while (it.next(self.io) catch return) |entry| {
            if (self.found.items.len >= max_results) return;
            if (entry.name.len == 0) continue;
            if (entry.name[0] == '.' and !self.allow_hidden) continue;
            const path = std.fs.path.join(self.alloc, &.{ prefix, entry.name }) catch return;
            defer self.alloc.free(path);
            switch (entry.kind) {
                .directory => self.walk(dir, path),
                .file => if (matchPattern(self.pattern, path)) {
                    const owned = self.alloc.dupe(u8, path) catch return;
                    self.found.append(self.alloc, owned) catch {
                        self.alloc.free(owned);
                        return;
                    };
                },
                else => {},
            }
        }
    }
};

// ----- pattern matching -----

fn matchPattern(pattern: []const u8, path: []const u8) bool {
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

    const result = try run(ws, testing.allocator, "{\"pattern\":\"**/*.zig\"}");
    defer testing.allocator.free(result.text);
    try testing.expectEqualStrings("a.zig\nb.zig\nsub/c.zig", result.text);
    try testing.expect(!result.truncated);

    const hidden = try run(ws, testing.allocator, "{\"pattern\":\".*.zig\"}");
    defer testing.allocator.free(hidden.text);
    try testing.expectEqualStrings(".hidden.zig", hidden.text);

    const bad = try run(ws, testing.allocator, "{}");
    defer testing.allocator.free(bad.text);
    try testing.expect(bad.is_error);
}
