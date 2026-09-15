//! `write_file` — create or replace a UTF-8 text file in the workspace.
//!
//! The whole content is given at once (there is no append mode); at 1 MiB, a
//! host constant, it is bounded like every other tool. The write is a
//! temp-file replacement (`Workspace.replaceFile`), so a crash leaves either
//! the old file or the new one. The target is resolved through
//! `Workspace.resolveTarget`: a symlink pointing outside the workspace is
//! refused, and a file that is too large or not valid UTF-8 is refused rather
//! than blindly overwritten — the model is told why, as a result it can read.
//!
//! The old and new contents become a structured diff (`tui.diff`) attached to
//! the result, which the loop turns into the `diff` event and the unified text
//! the model and session record.
const std = @import("std");
const root = @import("root.zig");
const tui = @import("../../tui/root.zig");

pub const tool: root.Tool = .{
    .name = "write_file",
    .description = "Create or replace a UTF-8 text file in the workspace, giving the whole content at once. The write is atomic.",
    .parameters = "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},\"content\":{\"type\":\"string\"}},\"required\":[\"path\",\"content\"]}",
    .verb = "Write",
    .subject = "path",
    .params = &.{ "path", "content" },
    .run = run,
};

const max_bytes: usize = 1024 * 1024;

const Args = struct { path: []const u8, content: []const u8 };

fn run(workspace: root.Workspace, alloc: std.mem.Allocator, arguments: []const u8) std.mem.Allocator.Error!root.Result {
    const parsed = std.json.parseFromSlice(Args, alloc, arguments, .{ .ignore_unknown_fields = true }) catch {
        return root.fail(alloc, "write_file: arguments must be a JSON object with \"path\" and \"content\" strings", .{});
    };
    defer parsed.deinit();
    if (parsed.value.path.len == 0) return root.fail(alloc, "write_file: path is empty", .{});
    if (parsed.value.content.len > max_bytes) {
        return root.fail(alloc, "write_file: content is too large ({d} bytes, max {d})", .{ parsed.value.content.len, max_bytes });
    }

    const target = workspace.resolveTarget(alloc, parsed.value.path) catch |err| switch (err) {
        error.OutsideWorkspace => return root.fail(alloc, "write_file: {s} is outside the workspace", .{parsed.value.path}),
        else => return root.fail(alloc, "write_file: {s}: {s}", .{ parsed.value.path, @errorName(err) }),
    };
    defer alloc.free(target.abs);

    // The old content is needed for the diff, so it is read first and the
    // replacement is refused when it cannot be represented as text.
    var old_bytes: ?[]u8 = null;
    defer if (old_bytes) |bytes| alloc.free(bytes);
    if (target.exists) {
        const stat = workspace.dir.statFile(workspace.io, target.abs, .{}) catch |err|
            return root.fail(alloc, "write_file: {s}: {s}", .{ parsed.value.path, @errorName(err) });
        if (stat.kind == .directory) return root.fail(alloc, "write_file: {s} is a directory", .{parsed.value.path});
        const bytes = workspace.dir.readFileAlloc(workspace.io, target.abs, alloc, .limited(max_bytes + 1)) catch |err|
            return root.fail(alloc, "write_file: {s}: {s}", .{ parsed.value.path, @errorName(err) });
        if (bytes.len > max_bytes) {
            alloc.free(bytes);
            return root.fail(alloc, "write_file: {s} is larger than {d} bytes; refusing to replace it", .{ parsed.value.path, max_bytes });
        }
        if (!std.unicode.utf8ValidateSlice(bytes)) {
            alloc.free(bytes);
            return root.fail(alloc, "write_file: {s} is not valid UTF-8; refusing to replace it", .{parsed.value.path});
        }
        old_bytes = bytes;
    }

    return commit(workspace, alloc, parsed.value.path, target.abs, old_bytes orelse "", parsed.value.content, target.exists);
}

/// Computes the diff, writes the file, and packages the result. Split out so
/// the failure paths above can return early without touching the file.
fn commit(workspace: root.Workspace, alloc: std.mem.Allocator, display: []const u8, target: []const u8, old: []const u8, content: []const u8, existed: bool) std.mem.Allocator.Error!root.Result {
    const path_owned = try alloc.dupe(u8, display);
    var change: root.Change = .{ .path = path_owned, .diff = undefined };
    change.diff = tui.diff.compute(alloc, old, content) catch |err| {
        alloc.free(path_owned);
        return err;
    };
    var keep = false;
    defer if (!keep) change.deinit(alloc);

    workspace.replaceFile(alloc, target, content) catch |err| {
        return root.fail(alloc, "write_file: {s}: {s}", .{ display, @errorName(err) });
    };

    const text = try std.fmt.allocPrint(alloc, "{s} {s} ({d} bytes)", .{ if (existed) "replaced" else "created", display, content.len });
    keep = true;
    return .{ .text = text, .change = change };
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
