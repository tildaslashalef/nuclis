//! `write_file` — create, replace, or append to a UTF-8 text file in the
//! workspace.
//!
//! The resulting file is bounded at 1 MiB, a host constant; `append` lets a
//! large file go in parts, so one failed call does not lose all of it. The
//! write is a temp-file replacement (`Workspace.replaceFile`), so a crash
//! leaves either the old file or the new one. The target is resolved through
//! `Workspace.resolveTarget`: a symlink pointing outside the workspace is
//! refused, missing parent directories are created only once every check has
//! passed, and a file that is too large or not valid UTF-8 is refused rather
//! than blindly overwritten — the model is told why, as a result it can read.
//!
//! The old and new contents become a structured diff (`tui.diff`) attached to
//! the result, which the loop turns into the `diff` event and, for a
//! replacement, the unified text the model and session record; a new file's
//! or an append's diff would only repeat the content the model just wrote.
const std = @import("std");
const root = @import("root.zig");
const tui = @import("../../tui/root.zig");

pub const tool: root.Tool = .{
    .name = "write_file",
    .description = "Create or replace a UTF-8 text file in the workspace, giving the whole content at once; `append: true` adds to the end, to write a large file in parts. The write is atomic.",
    .parameters = "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},\"content\":{\"type\":\"string\"},\"append\":{\"type\":\"boolean\"}},\"required\":[\"path\",\"content\"]}",
    .label = "Writing",
    .display = "Write",
    .subject = "path",
    .run = run,
};

const max_bytes: usize = 1024 * 1024;

const Args = struct { path: []const u8, content: []const u8, append: bool = false };

fn run(workspace: root.Workspace, alloc: std.mem.Allocator, arguments: []const u8) std.mem.Allocator.Error!root.Result {
    const parsed = std.json.parseFromSlice(Args, alloc, arguments, .{ .ignore_unknown_fields = true }) catch {
        return root.fail(alloc, "write_file: arguments must be a JSON object with \"path\" and \"content\" strings and an optional boolean \"append\"", .{});
    };
    defer parsed.deinit();
    const args = parsed.value;
    if (args.path.len == 0) return root.fail(alloc, "write_file: path is empty", .{});
    if (args.content.len > max_bytes) {
        return root.fail(alloc, "write_file: content is too large ({d} bytes, max {d})", .{ args.content.len, max_bytes });
    }

    const target = workspace.resolveTarget(alloc, args.path) catch |err| switch (err) {
        error.OutsideWorkspace => return root.fail(alloc, "write_file: {s} is outside the workspace", .{args.path}),
        else => return root.fail(alloc, "write_file: {s}: {s}", .{ args.path, @errorName(err) }),
    };
    defer alloc.free(target.abs);
    if (args.append and !target.exists) {
        return root.fail(alloc, "write_file: {s} does not exist; append adds to an existing file, so create it first without append", .{args.path});
    }

    // The old content is needed for the diff, so it is read first and the
    // replacement is refused when it cannot be represented as text.
    var old_bytes: ?[]u8 = null;
    defer if (old_bytes) |bytes| alloc.free(bytes);
    if (target.exists) {
        const stat = workspace.dir.statFile(workspace.io, target.abs, .{}) catch |err|
            return root.fail(alloc, "write_file: {s}: {s}", .{ args.path, @errorName(err) });
        if (stat.kind == .directory) return root.fail(alloc, "write_file: {s} is a directory", .{args.path});
        const verb = if (args.append) "append to" else "replace";
        const bytes = workspace.dir.readFileAlloc(workspace.io, target.abs, alloc, .limited(max_bytes + 1)) catch |err|
            return root.fail(alloc, "write_file: {s}: {s}", .{ args.path, @errorName(err) });
        if (bytes.len > max_bytes) {
            alloc.free(bytes);
            return root.fail(alloc, "write_file: {s} is larger than {d} bytes; refusing to {s} it", .{ args.path, max_bytes, verb });
        }
        if (!std.unicode.utf8ValidateSlice(bytes)) {
            alloc.free(bytes);
            return root.fail(alloc, "write_file: {s} is not valid UTF-8; refusing to {s} it", .{ args.path, verb });
        }
        old_bytes = bytes;
    }
    const old = old_bytes orelse "";

    if (!args.append) {
        return commit(workspace, alloc, args.path, target, old, args.content, if (target.exists) .replace else .create);
    }
    if (old.len + args.content.len > max_bytes) {
        return root.fail(alloc, "write_file: appending {d} bytes to {s} ({d} bytes) would pass the {d}-byte bound; nothing was written", .{ args.content.len, args.path, old.len, max_bytes });
    }
    const joined = try std.mem.concat(alloc, u8, &.{ old, args.content });
    defer alloc.free(joined);
    return commit(workspace, alloc, args.path, target, old, joined, .append);
}

const Mode = enum { create, replace, append };

/// Computes the diff, creates any missing parent directories, writes the
/// file, and packages the result. Split out so the failure paths above can
/// return early without touching the file system.
fn commit(workspace: root.Workspace, alloc: std.mem.Allocator, display: []const u8, target: root.ResolvedTarget, old: []const u8, content: []const u8, mode: Mode) std.mem.Allocator.Error!root.Result {
    const path_owned = try alloc.dupe(u8, display);
    var change: root.Change = .{ .path = path_owned, .diff = undefined, .echo = mode == .replace };
    change.diff = tui.diff.compute(alloc, old, content) catch |err| {
        alloc.free(path_owned);
        return err;
    };
    var keep = false;
    defer if (!keep) change.deinit(alloc);

    if (target.missing_parents.len != 0) {
        std.Io.Dir.cwd().createDirPath(workspace.io, target.parent()) catch |err| {
            return root.fail(alloc, "write_file: {s}: could not create its directory: {s}", .{ display, @errorName(err) });
        };
    }
    workspace.replaceFile(alloc, target.abs, content) catch |err| {
        removeCreated(workspace, target);
        return root.fail(alloc, "write_file: {s}: {s}", .{ display, @errorName(err) });
    };

    const lines = countLines(content);
    const added = if (mode == .append) countLines(content[old.len..]) else lines;
    const plural = if (added == 1) "" else "s";
    const text = switch (mode) {
        .create => if (target.missing_parents.len != 0)
            try alloc.print("created {s} ({d} line{s}; created director{s} {s}/)", .{ display, lines, plural, if (std.mem.indexOfScalar(u8, target.missing_parents, '/') == null) "y" else "ies", createdPath(workspace, target) })
        else
            try alloc.print("created {s} ({d} line{s})", .{ display, lines, plural }),
        .replace => try alloc.print("replaced {s} ({d} line{s})", .{ display, lines, plural }),
        .append => try alloc.print("appended {d} line{s} to {s} ({d} line{s} now)", .{ added, plural, display, lines, if (lines == 1) "" else "s" }),
    };
    errdefer alloc.free(text);
    const summary = switch (mode) {
        .create => try alloc.print("Wrote {d} line{s} to {s}", .{ lines, plural, display }),
        .replace => blk: {
            const counts = try root.changeSummary(alloc, change.diff.rows);
            defer alloc.free(counts);
            break :blk try alloc.print("Replaced {s}: {s} lines", .{ display, counts });
        },
        .append => try alloc.print("Appended {d} line{s} to {s}", .{ added, plural, display }),
    };
    keep = true;
    return .{ .text = text, .change = change, .summary = summary };
}

/// The deepest directory `commit` created, relative to the workspace root.
fn createdPath(workspace: root.Workspace, target: root.ResolvedTarget) []const u8 {
    const end = @intFromPtr(target.missing_parents.ptr) - @intFromPtr(target.abs.ptr) + target.missing_parents.len;
    const created = target.abs[0..end];
    if (created.len > workspace.root.len and std.mem.startsWith(u8, created, workspace.root) and created[workspace.root.len] == '/') return created[workspace.root.len + 1 ..];
    return target.missing_parents;
}

/// Removes the directories `commit` created for a write that then failed,
/// innermost first, so a refused write leaves nothing behind. Best effort:
/// a directory something else filled since is kept.
fn removeCreated(workspace: root.Workspace, target: root.ResolvedTarget) void {
    if (target.missing_parents.len == 0) return;
    const start = @intFromPtr(target.missing_parents.ptr) - @intFromPtr(target.abs.ptr);
    var path: []const u8 = target.abs[0 .. start + target.missing_parents.len];
    while (path.len > start) {
        std.Io.Dir.cwd().deleteDir(workspace.io, path) catch return;
        path = std.fs.path.dirname(path) orelse return;
    }
}

fn countLines(content: []const u8) usize {
    return std.mem.count(u8, content, "\n") + @intFromBool(content.len > 0 and content[content.len - 1] != '\n');
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

test "write_file creates a file and returns its diff" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    var result = try run(fixture.ws, alloc, "{\"path\":\"note.txt\",\"content\":\"hello\\nworld\\n\"}");
    defer result.deinit(alloc);
    try testing.expect(!result.is_error);
    try testing.expect(std.mem.indexOf(u8, result.text, "created note.txt") != null);
    try testing.expect(result.change != null);
    try testing.expect(std.mem.indexOf(u8, result.change.?.diff.unified, "+hello") != null);
    try testing.expectEqualStrings("Wrote 2 lines to note.txt", result.summary.?);
    const contents = try fixture.tmp.dir.readFileAlloc(testing.io, "note.txt", alloc, .limited(1024));
    defer alloc.free(contents);
    try testing.expectEqualStrings("hello\nworld\n", contents);
}

test "write_file replaces an existing file and diffs the change" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    try fixture.tmp.dir.writeFile(testing.io, .{ .sub_path = "note.txt", .data = "old\n" });
    var result = try run(fixture.ws, alloc, "{\"path\":\"note.txt\",\"content\":\"new\\n\"}");
    defer result.deinit(alloc);
    try testing.expect(!result.is_error);
    try testing.expect(std.mem.indexOf(u8, result.text, "replaced note.txt") != null);
    try testing.expect(std.mem.indexOf(u8, result.change.?.diff.unified, "-old") != null);
    try testing.expect(std.mem.indexOf(u8, result.change.?.diff.unified, "+new") != null);
    try testing.expectEqualStrings("Replaced note.txt: +1 −1 lines", result.summary.?);
    const contents = try fixture.tmp.dir.readFileAlloc(testing.io, "note.txt", alloc, .limited(1024));
    defer alloc.free(contents);
    try testing.expectEqualStrings("new\n", contents);
}

test "write_file refuses a symlink pointing outside the workspace" {
    const alloc = testing.allocator;
    var inside = testing.tmpDir(.{});
    defer inside.cleanup();
    var outside = testing.tmpDir(.{});
    defer outside.cleanup();
    const root_path = try inside.dir.realPathFileAlloc(testing.io, ".", alloc);
    defer alloc.free(root_path);
    const outside_path = try outside.dir.realPathFileAlloc(testing.io, ".", alloc);
    defer alloc.free(outside_path);
    try outside.dir.writeFile(testing.io, .{ .sub_path = "secret.txt", .data = "do not touch" });
    const link_target = try std.fs.path.join(alloc, &.{ outside_path, "secret.txt" });
    defer alloc.free(link_target);
    try inside.dir.symLink(testing.io, link_target, "link", .{});
    const ws: root.Workspace = .{ .io = testing.io, .dir = inside.dir, .root = root_path };

    var result = try run(ws, alloc, "{\"path\":\"link\",\"content\":\"overwritten\"}");
    defer result.deinit(alloc);
    try testing.expect(result.is_error);
    try testing.expect(std.mem.indexOf(u8, result.text, "outside the workspace") != null);
    const secret = try outside.dir.readFileAlloc(testing.io, "secret.txt", alloc, .limited(1024));
    defer alloc.free(secret);
    try testing.expectEqualStrings("do not touch", secret);
}

test "write_file reports bad arguments and an oversized content as results" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    var bad = try run(fixture.ws, alloc, "not json");
    defer bad.deinit(alloc);
    try testing.expect(bad.is_error);
    // The size check is by content length, so build a JSON string just over
    // the bound without a real megabyte of source.
    var big: std.Io.Writer.Allocating = .init(alloc);
    defer big.deinit();
    try big.writer.writeAll("{\"path\":\"x\",\"content\":\"");
    for (0..max_bytes + 1) |_| try big.writer.writeByte('a');
    try big.writer.writeAll("\"}");
    var oversized = try run(fixture.ws, alloc, big.written());
    defer oversized.deinit(alloc);
    try testing.expect(oversized.is_error);
}

test "write_file creates the missing parent directories of a nested new path" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    var result = try run(fixture.ws, alloc, "{\"path\":\"src/handlers/todos.ts\",\"content\":\"export {};\\n\"}");
    defer result.deinit(alloc);
    try testing.expect(!result.is_error);
    try testing.expectEqualStrings("created src/handlers/todos.ts (1 line; created directories src/handlers/)", result.text);
    const contents = try fixture.tmp.dir.readFileAlloc(testing.io, "src/handlers/todos.ts", alloc, .limited(1024));
    defer alloc.free(contents);
    try testing.expectEqualStrings("export {};\n", contents);
}

test "write_file refuses an escape through a missing parent and creates nothing" {
    const alloc = testing.allocator;
    var inside = testing.tmpDir(.{});
    defer inside.cleanup();
    var outside = testing.tmpDir(.{ .iterate = true });
    defer outside.cleanup();
    const root_path = try inside.dir.realPathFileAlloc(testing.io, ".", alloc);
    defer alloc.free(root_path);
    const outside_path = try outside.dir.realPathFileAlloc(testing.io, ".", alloc);
    defer alloc.free(outside_path);
    try inside.dir.symLink(testing.io, outside_path, "link", .{});
    const ws: root.Workspace = .{ .io = testing.io, .dir = inside.dir, .root = root_path };

    var through_link = try run(ws, alloc, "{\"path\":\"link/new/x.txt\",\"content\":\"x\"}");
    defer through_link.deinit(alloc);
    try testing.expect(through_link.is_error);
    try testing.expect(std.mem.indexOf(u8, through_link.text, "outside the workspace") != null);
    var dot_dot = try run(ws, alloc, "{\"path\":\"a/../../x\",\"content\":\"x\"}");
    defer dot_dot.deinit(alloc);
    try testing.expect(dot_dot.is_error);

    // A dangling link planted earlier reads as a missing name; creating
    // through it must still refuse rather than follow it.
    const planted = try std.fs.path.join(alloc, &.{ outside_path, "newdir" });
    defer alloc.free(planted);
    try inside.dir.symLink(testing.io, planted, "d", .{});
    for ([_][]const u8{ "{\"path\":\"d/x.txt\",\"content\":\"x\"}", "{\"path\":\"d/sub/x.txt\",\"content\":\"x\"}" }) |arguments| {
        var dangling = try run(ws, alloc, arguments);
        defer dangling.deinit(alloc);
        try testing.expect(dangling.is_error);
    }

    var it = outside.dir.iterate();
    try testing.expect(try it.next(testing.io) == null);
    try testing.expectError(error.FileNotFound, inside.dir.statFile(testing.io, "a", .{}));
}

test "write_file appends to an existing file and diffs only the added lines" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    try fixture.tmp.dir.writeFile(testing.io, .{ .sub_path = "big.ts", .data = "one\ntwo\n" });
    var result = try run(fixture.ws, alloc, "{\"path\":\"big.ts\",\"content\":\"three\\nfour\\n\",\"append\":true}");
    defer result.deinit(alloc);
    try testing.expect(!result.is_error);
    try testing.expectEqualStrings("appended 2 lines to big.ts (4 lines now)", result.text);
    try testing.expectEqualStrings("Appended 2 lines to big.ts", result.summary.?);
    const unified = result.change.?.diff.unified;
    try testing.expect(std.mem.indexOf(u8, unified, "+three") != null);
    try testing.expect(std.mem.indexOf(u8, unified, "+four") != null);
    try testing.expect(std.mem.indexOf(u8, unified, "\n-") == null);
    try testing.expect(!result.change.?.echo);
    const contents = try fixture.tmp.dir.readFileAlloc(testing.io, "big.ts", alloc, .limited(1024));
    defer alloc.free(contents);
    try testing.expectEqualStrings("one\ntwo\nthree\nfour\n", contents);
}

test "write_file append refuses a missing file and a result past the bound, writing nothing" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);

    var missing = try run(fixture.ws, alloc, "{\"path\":\"src/none.ts\",\"content\":\"x\",\"append\":true}");
    defer missing.deinit(alloc);
    try testing.expect(missing.is_error);
    try testing.expect(std.mem.indexOf(u8, missing.text, "does not exist") != null);
    try testing.expectError(error.FileNotFound, fixture.tmp.dir.statFile(testing.io, "src", .{}));

    const half = max_bytes / 2 + 1;
    const existing = try alloc.alloc(u8, half);
    defer alloc.free(existing);
    @memset(existing, 'a');
    try fixture.tmp.dir.writeFile(testing.io, .{ .sub_path = "part.txt", .data = existing });
    var args: std.Io.Writer.Allocating = .init(alloc);
    defer args.deinit();
    try args.writer.writeAll("{\"path\":\"part.txt\",\"append\":true,\"content\":\"");
    for (0..half) |_| try args.writer.writeByte('b');
    try args.writer.writeAll("\"}");
    var over = try run(fixture.ws, alloc, args.written());
    defer over.deinit(alloc);
    try testing.expect(over.is_error);
    try testing.expect(std.mem.indexOf(u8, over.text, "nothing was written") != null);
    const kept = try fixture.tmp.dir.readFileAlloc(testing.io, "part.txt", alloc, .limited(max_bytes + 1));
    defer alloc.free(kept);
    try testing.expectEqualSlices(u8, existing, kept);
}
