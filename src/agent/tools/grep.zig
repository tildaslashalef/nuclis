//! `grep` — literal, case-sensitive search across the workspace.
//!
//! Native Zig, bounded. At most `max_matches` matching lines and `max_bytes`
//! of rendered output per call; a file larger than `max_file_bytes` is
//! searched up to that point and the whole result is marked truncated. Hidden
//! entries and a small set of generated trees (`node_modules`, `target`, …)
//! are skipped; symlinks are not followed, so a link cannot walk the search
//! out of the workspace. Each match is one line of `path:line: text`, paths
//! workspace-relative.
//!
//! Deliberately literal, not a regex engine and not `rg`: the agent's search
//! contract is one predictable rule, and tools that need more (regex,
//! `.gitignore` semantics, the whole tree) are one `bash` call away.
const std = @import("std");
const root = @import("root.zig");

pub const tool: root.Tool = .{
    .name = "grep",
    .description = "Search workspace files for a literal, case-sensitive string. Returns matching lines as `path:line: text`; bounded and sorted.",
    .parameters = "{\"type\":\"object\",\"properties\":{\"pattern\":{\"type\":\"string\"}},\"required\":[\"pattern\"]}",
    .label = "Searching",
    .subject = "pattern",
    .run = run,
};

const max_matches: usize = 200;
const max_bytes: usize = 1024 * 1024;
const max_file_bytes: usize = 4 * 1024 * 1024;
const max_line: usize = 500;
/// Binary sniffing window; a NUL here marks a file as not text.
const sniff_bytes: usize = 8000;
/// Generated trees that make a whole-workspace literal scan slow. Skipped by
/// name; hidden entries are skipped regardless.
const skip_dirs = [_][]const u8{ "node_modules", "target", "zig-out", "vendor", "dist", "__pycache__" };

const Args = struct { pattern: []const u8 };

fn run(workspace: root.Workspace, alloc: std.mem.Allocator, arguments: []const u8) std.mem.Allocator.Error!root.Result {
    const parsed = std.json.parseFromSlice(Args, alloc, arguments, .{ .ignore_unknown_fields = true }) catch {
        return root.fail(alloc, "grep: arguments must be a JSON object with a \"pattern\" string", .{});
    };
    defer parsed.deinit();
    const pattern = parsed.value.pattern;
    if (pattern.len == 0) return root.fail(alloc, "grep: pattern is empty", .{});
    if (!std.unicode.utf8ValidateSlice(pattern)) return root.fail(alloc, "grep: pattern is not valid UTF-8", .{});

    var walker: Walker = .{ .alloc = alloc, .io = workspace.io, .dir = workspace.dir, .pattern = pattern };
    defer walker.deinit();
    try walker.walk("");
    std.mem.sort(Match, walker.matches.items, {}, lessThan);

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    for (walker.matches.items, 0..) |match, i| {
        if (i != 0) try out.append(alloc, '\n');
        try out.appendSlice(alloc, match.path);
        try out.append(alloc, ':');
        var number: [20]u8 = undefined;
        try out.appendSlice(alloc, std.fmt.bufPrint(&number, "{d}: ", .{match.line}) catch unreachable);
        try out.appendSlice(alloc, match.text);
    }
    const summary = try summarize(alloc, walker.matches.items, walker.truncated);
    errdefer alloc.free(summary);
    return .{ .text = try out.toOwnedSlice(alloc), .truncated = walker.truncated, .summary = summary };
}

/// The detail row: matches and the files they fall in. `matches` is sorted
/// by path, so a file boundary is a path change.
fn summarize(alloc: std.mem.Allocator, matches: []const Match, truncated: bool) std.mem.Allocator.Error![]u8 {
    var files: usize = 0;
    for (matches, 0..) |match, i| {
        if (i == 0 or !std.mem.eql(u8, matches[i - 1].path, match.path)) files += 1;
    }
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    if (matches.len == 0) {
        try out.appendSlice(alloc, "no matches");
    } else {
        try out.print(alloc, "{d} match{s} in {d} file{s}", .{ matches.len, plural(matches.len, "es"), files, plural(files, "s") });
    }
    if (truncated) try out.appendSlice(alloc, " · truncated");
    return out.toOwnedSlice(alloc);
}

fn plural(count: usize, suffix: []const u8) []const u8 {
    return if (count == 1) "" else suffix;
}

const Match = struct {
    path: []u8,
    line: usize,
    text: []u8,
};

fn lessThan(_: void, a: Match, b: Match) bool {
    return switch (std.mem.order(u8, a.path, b.path)) {
        .lt => true,
        .gt => false,
        .eq => a.line < b.line,
    };
}

const Walker = struct {
    alloc: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    pattern: []const u8,
    matches: std.ArrayList(Match) = .empty,
    bytes: usize = 0,
    truncated: bool = false,

    fn deinit(self: *Walker) void {
        for (self.matches.items) |match| {
            self.alloc.free(match.path);
            self.alloc.free(match.text);
        }
        self.matches.deinit(self.alloc);
    }

    fn walk(self: *Walker, prefix: []const u8) std.mem.Allocator.Error!void {
        if (self.truncated) return;
        var iterable = self.dir.openDir(self.io, if (prefix.len == 0) "." else prefix, .{ .iterate = true }) catch return;
        defer iterable.close(self.io);
        var iterator = iterable.iterate();
        while (iterator.next(self.io) catch return) |entry| {
            if (self.truncated) return;
            if (entry.name.len == 0 or entry.name[0] == '.') continue;
            const path = try std.fs.path.join(self.alloc, &.{ prefix, entry.name });
            defer self.alloc.free(path);
            switch (entry.kind) {
                .directory => if (!skipped(entry.name)) try self.walk(path),
                .file => try self.search(path),
                else => {}, // symlinks are not followed
            }
        }
    }

    fn search(self: *Walker, path: []const u8) std.mem.Allocator.Error!void {
        // One byte past the bound distinguishes "at the limit" from "larger".
        const bytes = self.dir.readFileAlloc(self.io, path, self.alloc, .limited(max_file_bytes + 1)) catch return;
        defer self.alloc.free(bytes);
        const content = if (bytes.len > max_file_bytes) blk: {
            self.truncated = true;
            break :blk bytes[0..max_file_bytes];
        } else bytes;
        if (looksBinary(content)) return;
        var line_number: usize = 1;
        var lines = std.mem.splitScalar(u8, content, '\n');
        while (lines.next()) |line| : (line_number += 1) {
            if (std.mem.indexOf(u8, line, self.pattern) == null) continue;
            try self.add(path, line_number, line);
            if (self.truncated) return;
        }
    }

    fn add(self: *Walker, path: []const u8, line_number: usize, line: []const u8) std.mem.Allocator.Error!void {
        if (self.matches.items.len >= max_matches) {
            self.truncated = true;
            return;
        }
        var text = if (line.len > max_line) line[0..max_line] else line;
        // The cut may have split a UTF-8 scalar; back off to the boundary.
        while (text.len > 0 and !std.unicode.utf8ValidateSlice(text)) {
            text = text[0 .. text.len - 1];
        }
        if (self.bytes + path.len + text.len + 24 > max_bytes) {
            self.truncated = true;
            return;
        }
        self.bytes += path.len + text.len + 24;
        const owned_path = try self.alloc.dupe(u8, path);
        errdefer self.alloc.free(owned_path);
        const owned_text = try self.alloc.dupe(u8, text);
        errdefer self.alloc.free(owned_text);
        try self.matches.append(self.alloc, .{ .path = owned_path, .line = line_number, .text = owned_text });
    }
};

fn skipped(name: []const u8) bool {
    for (skip_dirs) |dir| if (std.mem.eql(u8, name, dir)) return true;
    return false;
}

fn looksBinary(content: []const u8) bool {
    return std.mem.indexOfScalar(u8, content[0..@min(content.len, sniff_bytes)], 0) != null;
}

// ----- tests -----

const testing = std.testing;

const Fixture = struct {
    tmp: testing.TmpDir,
    ws: root.Workspace,
    root_path: [:0]u8,

    fn init(alloc: std.mem.Allocator) !Fixture {
        var tmp = testing.tmpDir(.{});
        const root_path = try tmp.dir.realPathFileAlloc(testing.io, ".", alloc);
        return .{ .tmp = tmp, .ws = .{ .io = testing.io, .dir = tmp.dir, .root = root_path }, .root_path = root_path };
    }
    fn deinit(self: *Fixture, alloc: std.mem.Allocator) void {
        self.tmp.cleanup();
        alloc.free(self.root_path);
    }
};

test "grep finds literal matches as path:line: text, sorted, and skips hidden and generated trees" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    const dir = fixture.tmp.dir;
    try dir.writeFile(testing.io, .{ .sub_path = "b.txt", .data = "beta\nneedle two\n" });
    try dir.writeFile(testing.io, .{ .sub_path = "a.txt", .data = "needle one\nplain\n" });
    try dir.writeFile(testing.io, .{ .sub_path = ".hidden.txt", .data = "needle hidden\n" });
    try dir.createDir(testing.io, "node_modules", .default_dir);
    try dir.writeFile(testing.io, .{ .sub_path = "node_modules/dep.txt", .data = "needle dependency\n" });
    try dir.createDir(testing.io, "sub", .default_dir);
    try dir.writeFile(testing.io, .{ .sub_path = "sub/c.txt", .data = "line\nneedle three\n" });
    try dir.writeFile(testing.io, .{ .sub_path = "binary.bin", .data = "needle\x00\x01" });

    var result = try run(fixture.ws, alloc, "{\"pattern\":\"needle\"}");
    defer result.deinit(alloc);
    try testing.expectEqualStrings("a.txt:1: needle one\nb.txt:2: needle two\nsub/c.txt:2: needle three", result.text);
    try testing.expect(!result.truncated);
    try testing.expect(!result.is_error);
    try testing.expectEqualStrings("3 matches in 3 files", result.summary.?);

    var none = try run(fixture.ws, alloc, "{\"pattern\":\"absent\"}");
    defer none.deinit(alloc);
    try testing.expectEqualStrings("no matches", none.summary.?);
}

test "grep reports bad arguments and an empty pattern as results" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    var bad = try run(fixture.ws, alloc, "not json");
    defer bad.deinit(alloc);
    try testing.expect(bad.is_error);
    var empty = try run(fixture.ws, alloc, "{\"pattern\":\"\"}");
    defer empty.deinit(alloc);
    try testing.expect(empty.is_error);
}

test "grep marks the line and byte bounds as truncated" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    // 250 matching lines is more than the 200-match bound.
    var content: std.Io.Writer.Allocating = .init(alloc);
    defer content.deinit();
    for (0..250) |i| try content.writer.print("needle {d}\n", .{i});
    try fixture.tmp.dir.writeFile(testing.io, .{ .sub_path = "many.txt", .data = content.written() });

    var result = try run(fixture.ws, alloc, "{\"pattern\":\"needle\"}");
    defer result.deinit(alloc);
    try testing.expect(result.truncated);
    var count: usize = 0;
    var lines = std.mem.splitScalar(u8, result.text, '\n');
    while (lines.next()) |line| : (count += 1) try testing.expect(std.mem.indexOf(u8, line, "needle") != null);
    try testing.expectEqual(max_matches, count);
    try testing.expectEqualStrings("200 matches in 1 file · truncated", result.summary.?);
}
