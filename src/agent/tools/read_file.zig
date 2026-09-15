//! `read_file` — a bounded, line-addressed region of a UTF-8 text file.
//!
//! Host constants, never model-supplied: at most `max_lines` lines and
//! `max_bytes` bytes per call. Exceeding either is a result with `truncated`
//! set (and the text stops at the bound), not an abort. A file that is not
//! valid UTF-8 is a typed error result: the transcript can render it, the
//! model should not pretend it read it.
const std = @import("std");
const root = @import("root.zig");

pub const tool: root.Tool = .{
    .name = "read_file",
    .description = "Read a bounded region of a UTF-8 text file in the workspace. `offset` is a 1-based line, `count` the number of lines.",
    .parameters = "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},\"offset\":{\"type\":\"integer\"},\"count\":{\"type\":\"integer\"}},\"required\":[\"path\"]}",
    .label = "Reading",
    .subject = "path",
    .run = run,
};

const max_lines: usize = 2000;
const max_bytes: usize = 1024 * 1024;

const Args = struct {
    path: []const u8,
    offset: ?usize = null,
    count: ?usize = null,
};

fn run(workspace: root.Workspace, alloc: std.mem.Allocator, arguments: []const u8) std.mem.Allocator.Error!root.Result {
    const parsed = std.json.parseFromSlice(Args, alloc, arguments, .{ .ignore_unknown_fields = true }) catch {
        return root.fail(alloc, "read_file: arguments must be a JSON object with a \"path\" string", .{});
    };
    defer parsed.deinit();
    const offset = parsed.value.offset orelse 1;
    const count = @min(parsed.value.count orelse max_lines, max_lines);
    if (offset == 0) return root.fail(alloc, "read_file: offset is 1-based", .{});

    const abs = workspace.resolve(alloc, parsed.value.path) catch |err| switch (err) {
        error.OutsideWorkspace => return root.fail(alloc, "read_file: {s} is outside the workspace", .{parsed.value.path}),
        else => return root.fail(alloc, "read_file: {s}: {s}", .{ parsed.value.path, @errorName(err) }),
    };
    defer alloc.free(abs);

    // One byte past the bound distinguishes "exactly at the limit" from
    // "larger", without reading the whole file.
    const bytes = workspace.dir.readFileAlloc(workspace.io, abs, alloc, .limited(max_bytes + 1)) catch |err| {
        return root.fail(alloc, "read_file: {s}: {s}", .{ parsed.value.path, @errorName(err) });
    };
    defer alloc.free(bytes);
    const byte_truncated = bytes.len > max_bytes;
    const content = if (byte_truncated) bytes[0..max_bytes] else bytes;
    if (!std.unicode.utf8ValidateSlice(content)) {
        return root.fail(alloc, "read_file: {s} is not valid UTF-8 text", .{parsed.value.path});
    }

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    var line: usize = 1;
    var taken: usize = 0;
    var line_truncated = false;
    var it = std.mem.splitScalar(u8, content, '\n');
    while (it.next()) |text| {
        if (line < offset) {
            line += 1;
            continue;
        }
        if (taken == count) {
            line_truncated = true;
            break;
        }
        if (taken != 0) try out.append(alloc, '\n');
        try out.appendSlice(alloc, text);
        taken += 1;
        line += 1;
    }
    const total = std.mem.count(u8, content, "\n") + 1;
    const summary = try summarize(alloc, offset, taken, total, line_truncated, byte_truncated);
    errdefer alloc.free(summary);
    return .{ .text = try out.toOwnedSlice(alloc), .truncated = byte_truncated or line_truncated, .summary = summary };
}

/// The detail row: the range shown, the file's length, and, when the read
/// stopped early, the offset that continues it.
fn summarize(alloc: std.mem.Allocator, offset: usize, taken: usize, total: usize, line_truncated: bool, byte_truncated: bool) std.mem.Allocator.Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    if (taken == 0) {
        try out.print(alloc, "no lines at offset {d}; the file has {d}", .{ offset, total });
    } else {
        try out.print(alloc, "lines {d} to {d} of {d}", .{ offset, offset + taken - 1, total });
    }
    if (line_truncated) try out.print(alloc, " · truncated, continue with offset={d}", .{offset + taken});
    if (byte_truncated) try out.print(alloc, " · first {d} bytes only", .{max_bytes});
    return out.toOwnedSlice(alloc);
}

// ----- tests -----

const testing = std.testing;

fn testWorkspace() struct { tmp: testing.TmpDir, ws: root.Workspace, root_path: [:0]u8 } {
    var tmp = testing.tmpDir(.{});
    const root_path = tmp.dir.realPathFileAlloc(testing.io, ".", testing.allocator) catch @panic("root");
    return .{ .tmp = tmp, .ws = .{ .io = testing.io, .dir = tmp.dir, .root = root_path }, .root_path = root_path };
}

test "read_file returns the addressed lines and marks truncation" {
    var w = testWorkspace();
    defer w.tmp.cleanup();
    defer testing.allocator.free(w.root_path);
    try w.tmp.dir.writeFile(testing.io, .{ .sub_path = "a.txt", .data = "one\ntwo\nthree\nfour" });

    var all = try run(w.ws, testing.allocator, "{\"path\":\"a.txt\"}");
    defer all.deinit(testing.allocator);
    try testing.expectEqualStrings("one\ntwo\nthree\nfour", all.text);
    try testing.expect(!all.truncated);
    try testing.expect(!all.is_error);
    try testing.expectEqualStrings("lines 1 to 4 of 4", all.summary.?);

    var middle = try run(w.ws, testing.allocator, "{\"path\":\"a.txt\",\"offset\":2,\"count\":2}");
    defer middle.deinit(testing.allocator);
    try testing.expectEqualStrings("two\nthree", middle.text);
    try testing.expect(middle.truncated); // "four" was not shown
    try testing.expectEqualStrings("lines 2 to 3 of 4 · truncated, continue with offset=4", middle.summary.?);

    var past = try run(w.ws, testing.allocator, "{\"path\":\"a.txt\",\"offset\":9}");
    defer past.deinit(testing.allocator);
    try testing.expectEqualStrings("", past.text);
    try testing.expectEqualStrings("no lines at offset 9; the file has 4", past.summary.?);

    try testing.expectError(error.OutsideWorkspace, @as(root.Workspace, w.ws).resolve(testing.allocator, "/"));
}

test "read_file reports bad arguments, missing files, and non-UTF-8 as results" {
    var w = testWorkspace();
    defer w.tmp.cleanup();
    defer testing.allocator.free(w.root_path);
    try w.tmp.dir.writeFile(testing.io, .{ .sub_path = "bin", .data = "\xff\xfe" });

    var bad = try run(w.ws, testing.allocator, "not json");
    defer bad.deinit(testing.allocator);
    try testing.expect(bad.is_error);
    try testing.expect(bad.summary == null);

    var missing = try run(w.ws, testing.allocator, "{\"path\":\"nope.txt\"}");
    defer missing.deinit(testing.allocator);
    try testing.expect(missing.is_error);

    var binary = try run(w.ws, testing.allocator, "{\"path\":\"bin\"}");
    defer binary.deinit(testing.allocator);
    try testing.expect(binary.is_error);
    try testing.expect(std.mem.indexOf(u8, binary.text, "UTF-8") != null);

    var zero = try run(w.ws, testing.allocator, "{\"path\":\"bin\",\"offset\":0}");
    defer zero.deinit(testing.allocator);
    try testing.expect(zero.is_error);
}
