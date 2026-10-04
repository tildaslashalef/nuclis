//! `grep` — literal, case-sensitive search across the workspace.
//!
//! Native Zig, bounded. Results are grouped by file (the path once, then
//! `line: text`, context lines as `line- text`), at most `max_shown` matches
//! with the total beyond them stated; a file larger than `max_file_bytes` is
//! searched up to that point. Hidden entries, a small set of generated trees
//! (`node_modules`, `target`, …), and what each directory's `.gitignore`
//! excludes are skipped; symlinks are not followed, so a link cannot walk the
//! search out of the workspace. Directories are walked in name order, so the
//! shown matches are the first ones in path order.
//!
//! Deliberately literal, not a regex engine and not `rg`: the agent's search
//! contract is one predictable rule, and a regex is one `bash` call away.
const std = @import("std");
const root = @import("root.zig");
const glob = @import("glob.zig");
const ignore = @import("ignore.zig");

pub const tool: root.Tool = .{
    .name = "grep",
    .description = "Search workspace files for a literal, case-sensitive string. `path` narrows it to a directory, a file, or a glob (`src/**/*.py`); `context` adds up to 5 lines around each match. Output is grouped by file: the path, then `line: text` (context `line- text`); at most 50 matches, the total stated. Files .gitignore excludes are skipped.",
    .parameters = "{\"type\":\"object\",\"properties\":{\"pattern\":{\"type\":\"string\"},\"path\":{\"type\":\"string\"},\"context\":{\"type\":\"integer\"}},\"required\":[\"pattern\"]}",
    .label = "Searching",
    .display = "Grep",
    .subject = "pattern",
    .run = run,
};

const max_shown: usize = 50;
const max_context: usize = 5;
const max_file_bytes: usize = 4 * 1024 * 1024;
const max_line: usize = 500;
/// Binary sniffing window; a NUL here marks a file as not text.
const sniff_bytes: usize = 8000;
/// Generated trees that make a whole-workspace literal scan slow. Skipped by
/// name; hidden entries are skipped regardless.
const skip_dirs = [_][]const u8{ "node_modules", "target", "zig-out", "vendor", "dist", "__pycache__" };

const Args = struct {
    pattern: []const u8,
    path: ?[]const u8 = null,
    context: ?i64 = null,
};

fn run(workspace: root.Workspace, alloc: std.mem.Allocator, arguments: []const u8) std.mem.Allocator.Error!root.Result {
    const parsed = std.json.parseFromSlice(Args, alloc, arguments, .{ .ignore_unknown_fields = true }) catch {
        return root.fail(alloc, "grep: arguments must be a JSON object with a \"pattern\" string", .{});
    };
    defer parsed.deinit();
    const pattern = parsed.value.pattern;
    if (pattern.len == 0) return root.fail(alloc, "grep: pattern is empty", .{});
    if (!std.unicode.utf8ValidateSlice(pattern)) return root.fail(alloc, "grep: pattern is not valid UTF-8", .{});
    if (std.mem.indexOfScalar(u8, pattern, '\n') != null) return root.fail(alloc, "grep: pattern spans lines; search for one line of it", .{});
    const context: usize = @intCast(std.math.clamp(parsed.value.context orelse 0, 0, max_context));

    var walker: Walker = .{ .alloc = alloc, .io = workspace.io, .dir = workspace.dir, .pattern = pattern, .context = context, .rules = .{ .alloc = alloc } };
    defer walker.deinit();

    const scope = std.mem.trim(u8, parsed.value.path orelse "", " ");
    if (scope.len == 0 or std.mem.eql(u8, scope, ".")) {
        try walker.walk("");
    } else if (std.mem.indexOfAny(u8, scope, "*?[") != null) {
        // A glob: walk from its literal leading directories, keep what matches.
        const filter = if (std.mem.startsWith(u8, scope, "./")) scope[2..] else scope;
        walker.filter = filter;
        const meta = std.mem.indexOfAny(u8, filter, "*?[").?;
        const start = if (std.mem.lastIndexOfScalar(u8, filter[0..meta], '/')) |slash| filter[0..slash] else "";
        try walker.enter(start);
    } else {
        const abs = workspace.resolve(alloc, scope) catch |err| switch (err) {
            error.OutsideWorkspace => return root.fail(alloc, "grep: {s} is outside the workspace", .{scope}),
            error.OutOfMemory => return error.OutOfMemory,
            else => return root.fail(alloc, "grep: {s}: {s}", .{ scope, @errorName(err) }),
        };
        defer alloc.free(abs);
        const rel = if (abs.len == workspace.root.len) "" else abs[workspace.root.len + 1 ..];
        const stat = workspace.dir.statFile(workspace.io, abs, .{}) catch |err|
            return root.fail(alloc, "grep: {s}: {s}", .{ scope, @errorName(err) });
        // A file named outright is searched even when an ignore rule covers it.
        if (stat.kind == .directory) try walker.enter(rel) else try walker.search(rel);
    }

    var text = walker.out;
    walker.out = .empty;
    errdefer text.deinit(alloc);
    var note: std.ArrayList(u8) = .empty;
    defer note.deinit(alloc);
    if (walker.total == 0) {
        try note.appendSlice(alloc, "no matches");
    } else {
        try note.print(alloc, "{d} match{s} in {d} file{s}", .{ walker.total, plural(walker.total, "es"), walker.files, plural(walker.files, "s") });
        if (walker.shown < walker.total) try note.print(alloc, " · first {d} shown; narrow the pattern or the path for the rest", .{walker.shown});
    }
    if (walker.cut_files > 0) try note.print(alloc, " · {d} file{s} over {d} MiB searched in part", .{ walker.cut_files, plural(walker.cut_files, "s"), max_file_bytes / (1024 * 1024) });
    // The model reads a note only when the list alone would mislead it.
    if (walker.total == 0 or walker.shown < walker.total or walker.cut_files > 0) {
        if (text.items.len > 0) try text.append(alloc, '\n');
        try text.print(alloc, "[{s}]", .{note.items});
    }
    const summary = try alloc.dupe(u8, note.items);
    errdefer alloc.free(summary);
    return .{ .text = try text.toOwnedSlice(alloc), .truncated = walker.shown < walker.total or walker.cut_files > 0, .summary = summary };
}

fn plural(count: usize, suffix: []const u8) []const u8 {
    return if (count == 1) "" else suffix;
}

const Entry = struct { name: []u8, kind: std.Io.File.Kind };

fn entryLess(_: void, a: Entry, b: Entry) bool {
    return std.mem.lessThan(u8, a.name, b.name);
}

const Walker = struct {
    alloc: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    pattern: []const u8,
    context: usize,
    rules: ignore.Rules,
    /// A glob a file's workspace-relative path must match; null for all.
    filter: ?[]const u8 = null,
    /// The rendered groups of the shown matches.
    out: std.ArrayList(u8) = .empty,
    shown: usize = 0,
    total: usize = 0,
    files: usize = 0,
    cut_files: usize = 0,

    fn deinit(self: *Walker) void {
        self.rules.deinit();
        self.out.deinit(self.alloc);
    }

    /// Walks `prefix` after loading the `.gitignore` of every directory above
    /// it, so a narrowed search skips what a whole one would.
    fn enter(self: *Walker, prefix: []const u8) std.mem.Allocator.Error!void {
        var pushed: usize = 0;
        defer for (0..pushed) |_| self.rules.pop();
        if (try self.rules.push(self.io, self.dir, "")) pushed += 1;
        var at: usize = 0;
        while (std.mem.indexOfScalarPos(u8, prefix, at, '/')) |slash| : (at = slash + 1) {
            if (try self.rules.push(self.io, self.dir, prefix[0..slash])) pushed += 1;
        }
        if (prefix.len == 0) return self.walkEntries("");
        return self.walk(prefix);
    }

    fn walk(self: *Walker, prefix: []const u8) std.mem.Allocator.Error!void {
        const pushed = try self.rules.push(self.io, self.dir, prefix);
        defer if (pushed) self.rules.pop();
        try self.walkEntries(prefix);
    }

    fn walkEntries(self: *Walker, prefix: []const u8) std.mem.Allocator.Error!void {
        var iterable = self.dir.openDir(self.io, if (prefix.len == 0) "." else prefix, .{ .iterate = true }) catch return;
        defer iterable.close(self.io);
        var entries: std.ArrayList(Entry) = .empty;
        defer {
            for (entries.items) |entry| self.alloc.free(entry.name);
            entries.deinit(self.alloc);
        }
        var iterator = iterable.iterate();
        while (iterator.next(self.io) catch null) |entry| {
            if (entry.name.len == 0 or entry.name[0] == '.') continue;
            if (entry.kind != .directory and entry.kind != .file) continue; // symlinks are not followed
            const name = try self.alloc.dupe(u8, entry.name);
            errdefer self.alloc.free(name);
            try entries.append(self.alloc, .{ .name = name, .kind = entry.kind });
        }
        std.mem.sort(Entry, entries.items, {}, entryLess);
        for (entries.items) |entry| {
            const path = try std.fs.path.join(self.alloc, &.{ prefix, entry.name });
            defer self.alloc.free(path);
            const is_dir = entry.kind == .directory;
            if (self.rules.ignored(path, is_dir)) continue;
            if (is_dir) {
                if (!skipped(entry.name)) try self.walk(path);
            } else if (self.filter == null or glob.matchPattern(self.filter.?, path)) {
                try self.search(path);
            }
        }
    }

    /// Counts a file's matches and renders the ones still within `max_shown`
    /// with their context, merged where they overlap.
    fn search(self: *Walker, path: []const u8) std.mem.Allocator.Error!void {
        // One byte past the bound distinguishes "at the limit" from "larger".
        const bytes = self.dir.readFileAlloc(self.io, path, self.alloc, .limited(max_file_bytes + 1)) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return,
        };
        defer self.alloc.free(bytes);
        const content = if (bytes.len > max_file_bytes) bytes[0..max_file_bytes] else bytes;
        if (looksBinary(content)) return;
        if (bytes.len > max_file_bytes) self.cut_files += 1;

        var lines: std.ArrayList([]const u8) = .empty;
        defer lines.deinit(self.alloc);
        var matches: std.ArrayList(usize) = .empty;
        defer matches.deinit(self.alloc);
        var it = std.mem.splitScalar(u8, content, '\n');
        while (it.next()) |line| {
            if (std.mem.indexOf(u8, line, self.pattern) != null) try matches.append(self.alloc, lines.items.len);
            try lines.append(self.alloc, line);
        }
        if (matches.items.len == 0) return;
        self.total += matches.items.len;
        self.files += 1;
        const take = @min(matches.items.len, max_shown - self.shown);
        if (take == 0) return;
        self.shown += take;

        if (self.out.items.len > 0) try self.out.appendSlice(self.alloc, "\n\n");
        try self.out.appendSlice(self.alloc, path);
        var next: usize = 0; // the first line not yet rendered
        for (matches.items[0..take], 0..) |hit, i| {
            const from = @max(hit -| self.context, next);
            const to = @min(lines.items.len, hit + self.context + 1);
            if (i > 0 and from > next) try self.out.appendSlice(self.alloc, "\n--");
            for (from..to) |index| {
                const is_match = index == hit or std.mem.indexOf(u8, lines.items[index], self.pattern) != null;
                try self.renderLine(index + 1, is_match, lines.items[index]);
            }
            next = to;
        }
    }

    fn renderLine(self: *Walker, number: usize, is_match: bool, line: []const u8) std.mem.Allocator.Error!void {
        var text = std.mem.trimEnd(u8, line, "\r");
        if (text.len > max_line) {
            text = text[0..max_line];
            // The cut may have split a UTF-8 scalar; back off to the boundary.
            while (text.len > 0 and !std.unicode.utf8ValidateSlice(text)) text = text[0 .. text.len - 1];
        }
        try self.out.print(self.alloc, "\n{d}{s} {s}", .{ number, if (is_match) ":" else "-", text });
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

test "grep groups literal matches by file in path order and skips hidden, generated, and ignored trees" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    const dir = fixture.tmp.dir;
    try dir.writeFile(testing.io, .{ .sub_path = "b.txt", .data = "beta\nneedle two\nneedle again\n" });
    try dir.writeFile(testing.io, .{ .sub_path = "a.txt", .data = "needle one\nplain\n" });
    try dir.writeFile(testing.io, .{ .sub_path = ".hidden.txt", .data = "needle hidden\n" });
    try dir.writeFile(testing.io, .{ .sub_path = ".gitignore", .data = "out/\n*.gen\n" });
    try dir.writeFile(testing.io, .{ .sub_path = "x.gen", .data = "needle generated\n" });
    try dir.createDirPath(testing.io, "out");
    try dir.writeFile(testing.io, .{ .sub_path = "out/o.txt", .data = "needle built\n" });
    try dir.createDirPath(testing.io, "node_modules");
    try dir.writeFile(testing.io, .{ .sub_path = "node_modules/dep.txt", .data = "needle dependency\n" });
    try dir.createDirPath(testing.io, "sub");
    try dir.writeFile(testing.io, .{ .sub_path = "sub/c.txt", .data = "line\nneedle three\n" });
    try dir.writeFile(testing.io, .{ .sub_path = "binary.bin", .data = "needle\x00\x01" });

    var result = try run(fixture.ws, alloc, "{\"pattern\":\"needle\"}");
    defer result.deinit(alloc);
    try testing.expectEqualStrings("a.txt\n1: needle one\n\nb.txt\n2: needle two\n3: needle again\n\nsub/c.txt\n2: needle three", result.text);
    try testing.expect(!result.truncated);
    try testing.expect(!result.is_error);
    try testing.expectEqualStrings("4 matches in 3 files", result.summary.?);

    // Nothing found is said, not left blank.
    var none = try run(fixture.ws, alloc, "{\"pattern\":\"absent\"}");
    defer none.deinit(alloc);
    try testing.expectEqualStrings("[no matches]", none.text);
    try testing.expectEqualStrings("no matches", none.summary.?);
}

test "grep narrows to a directory, a file, or a glob, and adds merged context" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    const dir = fixture.tmp.dir;
    try dir.createDirPath(testing.io, "src/deep");
    try dir.writeFile(testing.io, .{ .sub_path = "src/m.py", .data = "a\nb\nhit 1\nc\nhit 2\nd\ne\nf\ng\nhit 3\n" });
    try dir.writeFile(testing.io, .{ .sub_path = "src/deep/n.py", .data = "hit deep\n" });
    try dir.writeFile(testing.io, .{ .sub_path = "src/notes.txt", .data = "hit text\n" });
    try dir.writeFile(testing.io, .{ .sub_path = "top.py", .data = "hit top\n" });

    var under = try run(fixture.ws, alloc, "{\"pattern\":\"hit\",\"path\":\"src/deep\"}");
    defer under.deinit(alloc);
    try testing.expectEqualStrings("src/deep/n.py\n1: hit deep", under.text);

    var file = try run(fixture.ws, alloc, "{\"pattern\":\"hit\",\"path\":\"./top.py\"}");
    defer file.deinit(alloc);
    try testing.expectEqualStrings("top.py\n1: hit top", file.text);

    var pattern = try run(fixture.ws, alloc, "{\"pattern\":\"hit deep\",\"path\":\"src/**/*.py\"}");
    defer pattern.deinit(alloc);
    try testing.expectEqualStrings("src/deep/n.py\n1: hit deep", pattern.text);
    var py_only = try run(fixture.ws, alloc, "{\"pattern\":\"hit\",\"path\":\"**/*.txt\"}");
    defer py_only.deinit(alloc);
    try testing.expectEqualStrings("src/notes.txt\n1: hit text", py_only.text);

    // Context 1: the first two hits share their lines, the third stands apart.
    var around = try run(fixture.ws, alloc, "{\"pattern\":\"hit\",\"path\":\"src/m.py\",\"context\":1}");
    defer around.deinit(alloc);
    try testing.expectEqualStrings("src/m.py\n2- b\n3: hit 1\n4- c\n5: hit 2\n6- d\n--\n9- g\n10: hit 3\n11- ", around.text);

    var outside = try run(fixture.ws, alloc, "{\"pattern\":\"hit\",\"path\":\"..\"}");
    defer outside.deinit(alloc);
    try testing.expect(outside.is_error);
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
    var missing = try run(fixture.ws, alloc, "{\"pattern\":\"x\",\"path\":\"nope\"}");
    defer missing.deinit(alloc);
    try testing.expect(missing.is_error);
}

test "grep shows the first matches and states the total" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    var content: std.Io.Writer.Allocating = .init(alloc);
    defer content.deinit();
    for (0..250) |i| try content.writer.print("needle {d}\n", .{i});
    try fixture.tmp.dir.writeFile(testing.io, .{ .sub_path = "many.txt", .data = content.written() });
    try fixture.tmp.dir.writeFile(testing.io, .{ .sub_path = "more.txt", .data = "needle late\n" });

    var result = try run(fixture.ws, alloc, "{\"pattern\":\"needle\"}");
    defer result.deinit(alloc);
    try testing.expect(result.truncated);
    try testing.expectEqual(max_shown, std.mem.count(u8, result.text, ": needle"));
    try testing.expect(std.mem.indexOf(u8, result.text, "more.txt") == null);
    try testing.expect(std.mem.endsWith(u8, result.text, "\n[251 matches in 2 files · first 50 shown; narrow the pattern or the path for the rest]"));
}
