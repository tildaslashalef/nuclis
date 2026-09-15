//! `edit_file` — replace one exact, unique byte sequence in a text file.
//!
//! The contract is deliberately strict: `old_string` must appear exactly once
//! and must not be empty. Zero or multiple matches fail **without writing**,
//! because a blind replacement is how an edit lands in the wrong place; the
//! model is told the count so it can add context and try again. `new_string`
//! may be empty, which deletes the matched text.
//!
//! Arguments are validated before anything is touched: the target must
//! resolve inside the workspace, exist as a UTF-8 text file within the 1 MiB
//! bound, and the match must be unique. The write itself is the same atomic
//! temp-file replacement `write_file` uses, and the old/new difference becomes
//! the structured diff on the result.
const std = @import("std");
const root = @import("root.zig");
const tui = @import("../../tui/root.zig");

pub const tool: root.Tool = .{
    .name = "edit_file",
    .description = "Replace one exact, unique text sequence in a workspace file. Fails without writing on zero or multiple matches; `new_string` may be empty to delete.",
    .parameters = "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},\"old_string\":{\"type\":\"string\"},\"new_string\":{\"type\":\"string\"}},\"required\":[\"path\",\"old_string\",\"new_string\"]}",
    .label = "Editing",
    .subject = "path",
    .run = run,
};

const max_bytes: usize = 1024 * 1024;

const Args = struct { path: []const u8, old_string: []const u8, new_string: []const u8 };

fn run(workspace: root.Workspace, alloc: std.mem.Allocator, arguments: []const u8) std.mem.Allocator.Error!root.Result {
    const parsed = std.json.parseFromSlice(Args, alloc, arguments, .{ .ignore_unknown_fields = true }) catch {
        return root.fail(alloc, "edit_file: arguments must be a JSON object with \"path\", \"old_string\", and \"new_string\" strings", .{});
    };
    defer parsed.deinit();
    const path = parsed.value.path;
    const old_string = parsed.value.old_string;
    const new_string = parsed.value.new_string;
    if (path.len == 0) return root.fail(alloc, "edit_file: path is empty", .{});
    if (old_string.len == 0) return root.fail(alloc, "edit_file: old_string is empty", .{});
    if (std.mem.eql(u8, old_string, new_string)) return root.fail(alloc, "edit_file: old_string and new_string are identical", .{});

    const target = workspace.resolveTarget(alloc, path) catch |err| switch (err) {
        error.OutsideWorkspace => return root.fail(alloc, "edit_file: {s} is outside the workspace", .{path}),
        else => return root.fail(alloc, "edit_file: {s}: {s}", .{ path, @errorName(err) }),
    };
    defer alloc.free(target.abs);
    if (!target.exists) return root.fail(alloc, "edit_file: {s} does not exist", .{path});

    const stat = workspace.dir.statFile(workspace.io, target.abs, .{}) catch |err|
        return root.fail(alloc, "edit_file: {s}: {s}", .{ path, @errorName(err) });
    if (stat.kind == .directory) return root.fail(alloc, "edit_file: {s} is a directory", .{path});

    const content = workspace.dir.readFileAlloc(workspace.io, target.abs, alloc, .limited(max_bytes + 1)) catch |err|
        return root.fail(alloc, "edit_file: {s}: {s}", .{ path, @errorName(err) });
    defer alloc.free(content);
    if (content.len > max_bytes) return root.fail(alloc, "edit_file: {s} is larger than {d} bytes", .{ path, max_bytes });
    if (!std.unicode.utf8ValidateSlice(content)) return root.fail(alloc, "edit_file: {s} is not valid UTF-8", .{path});

    // Count non-overlapping matches; only exactly one is editable.
    var count: usize = 0;
    var at: usize = 0;
    var search: usize = 0;
    while (std.mem.indexOfPos(u8, content, search, old_string)) |found| {
        if (count == 0) at = found;
        count += 1;
        search = found + old_string.len;
    }
    if (count == 0) return root.fail(alloc, "edit_file: old_string was not found in {s}", .{path});
    if (count > 1) return root.fail(alloc, "edit_file: old_string appears {d} times in {s}; it must be unique (add surrounding context)", .{ count, path });

    var updated: std.ArrayList(u8) = .empty;
    defer updated.deinit(alloc);
    try updated.appendSlice(alloc, content[0..at]);
    try updated.appendSlice(alloc, new_string);
    try updated.appendSlice(alloc, content[at + old_string.len ..]);

    const path_owned = try alloc.dupe(u8, path);
    var change: root.Change = .{ .path = path_owned, .diff = undefined };
    change.diff = tui.diff.compute(alloc, content, updated.items) catch |err| {
        alloc.free(path_owned);
        return err;
    };
    var keep = false;
    defer if (!keep) change.deinit(alloc);

    workspace.replaceFile(alloc, target.abs, updated.items) catch |err|
        return root.fail(alloc, "edit_file: {s}: {s}", .{ path, @errorName(err) });

    const text = try std.fmt.allocPrint(alloc, "edited {s}: 1 replacement", .{path});
    errdefer alloc.free(text);
    const summary = try root.changeSummary(alloc, change.diff.rows);
    keep = true;
    return .{ .text = text, .change = change, .summary = summary };
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

test "edit_file replaces one unique match and diffs the change" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    try fixture.tmp.dir.writeFile(testing.io, .{ .sub_path = "a.txt", .data = "one\ntwo\nthree\n" });
    var result = try run(fixture.ws, alloc, "{\"path\":\"a.txt\",\"old_string\":\"two\",\"new_string\":\"TWO\"}");
    defer result.deinit(alloc);
    try testing.expect(!result.is_error);
    try testing.expect(std.mem.indexOf(u8, result.text, "1 replacement") != null);
    try testing.expect(std.mem.indexOf(u8, result.change.?.diff.unified, "-two") != null);
    try testing.expect(std.mem.indexOf(u8, result.change.?.diff.unified, "+TWO") != null);
    try testing.expectEqualStrings("+1 −1", result.summary.?);
    const contents = try fixture.tmp.dir.readFileAlloc(testing.io, "a.txt", alloc, .limited(1024));
    defer alloc.free(contents);
    try testing.expectEqualStrings("one\nTWO\nthree\n", contents);
}

test "edit_file with two matches changes nothing and returns the count" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    try fixture.tmp.dir.writeFile(testing.io, .{ .sub_path = "a.txt", .data = "x\nx\n" });
    var result = try run(fixture.ws, alloc, "{\"path\":\"a.txt\",\"old_string\":\"x\",\"new_string\":\"y\"}");
    defer result.deinit(alloc);
    try testing.expect(result.is_error);
    try testing.expect(std.mem.indexOf(u8, result.text, "2 times") != null);
    try testing.expect(result.change == null);
    const contents = try fixture.tmp.dir.readFileAlloc(testing.io, "a.txt", alloc, .limited(1024));
    defer alloc.free(contents);
    try testing.expectEqualStrings("x\nx\n", contents);
}

test "edit_file rejects a missing match, an empty old_string, and a no-op" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    try fixture.tmp.dir.writeFile(testing.io, .{ .sub_path = "a.txt", .data = "hello\n" });

    var missing = try run(fixture.ws, alloc, "{\"path\":\"a.txt\",\"old_string\":\"nope\",\"new_string\":\"y\"}");
    defer missing.deinit(alloc);
    try testing.expect(missing.is_error);
    try testing.expect(std.mem.indexOf(u8, missing.text, "not found") != null);

    var empty = try run(fixture.ws, alloc, "{\"path\":\"a.txt\",\"old_string\":\"\",\"new_string\":\"y\"}");
    defer empty.deinit(alloc);
    try testing.expect(empty.is_error);

    var noop = try run(fixture.ws, alloc, "{\"path\":\"a.txt\",\"old_string\":\"hello\",\"new_string\":\"hello\"}");
    defer noop.deinit(alloc);
    try testing.expect(noop.is_error);

    const contents = try fixture.tmp.dir.readFileAlloc(testing.io, "a.txt", alloc, .limited(1024));
    defer alloc.free(contents);
    try testing.expectEqualStrings("hello\n", contents);
}
