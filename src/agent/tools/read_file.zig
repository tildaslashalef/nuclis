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
    .description = "Read a bounded region of a UTF-8 text file in the workspace. `offset` is a 1-based line, `count` the number of lines (default 200, at most 2000); the result says where to continue.",
    .parameters = "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},\"offset\":{\"type\":\"integer\"},\"count\":{\"type\":\"integer\"}},\"required\":[\"path\"]}",
    .label = "Reading",
    .display = "Read",
    .subject = "path",
    .run = run,
};

const max_lines: usize = 2000;
/// A page the model asks past, rather than a whole file it did not need.
const default_lines: usize = 200;
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
    const count = @min(parsed.value.count orelse default_lines, max_lines);
    if (offset == 0) return root.fail(alloc, "read_file: offset is 1-based", .{});

    const abs = workspace.resolve(alloc, parsed.value.path) catch |err| switch (err) {
        error.OutsideWorkspace => return root.fail(alloc, "read_file: {s} is outside the workspace", .{parsed.value.path}),
        else => return root.fail(alloc, "read_file: {s}: {s}", .{ parsed.value.path, @errorName(err) }),
    };
    defer alloc.free(abs);

    // The first `max_bytes` of a larger file are served, never a refusal:
    // the size comes from `stat`, so the read itself stays within the bound.
    const file = workspace.dir.openFile(workspace.io, abs, .{}) catch |err| {
        return root.fail(alloc, "read_file: {s}: {s}", .{ parsed.value.path, @errorName(err) });
    };
    defer file.close(workspace.io);
    const stat = file.stat(workspace.io) catch |err| {
        return root.fail(alloc, "read_file: {s}: {s}", .{ parsed.value.path, @errorName(err) });
    };
    if (stat.kind == .directory) return root.fail(alloc, "read_file: {s} is a directory", .{parsed.value.path});
    const size: usize = @intCast(stat.size);
    const byte_truncated = size > max_bytes;
    const bytes = try alloc.alloc(u8, @min(size, max_bytes));
    defer alloc.free(bytes);
    var buffer: [16 * 1024]u8 = undefined;
    var reader = file.reader(workspace.io, &buffer);
    const read_len = reader.interface.readSliceShort(bytes) catch |err| {
        return root.fail(alloc, "read_file: {s}: {s}", .{ parsed.value.path, @errorName(err) });
    };
    const content = bytes[0..read_len];
    if (!std.unicode.utf8ValidateSlice(content)) {
        return root.fail(alloc, "read_file: {s} is not valid UTF-8 text", .{parsed.value.path});
    }

    // A trailing newline ends the last line; it is not an empty line after it.
    const body = if (content.len > 0 and content[content.len - 1] == '\n') content[0 .. content.len - 1] else content;
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    var line: usize = 1;
    var taken: usize = 0;
    var line_truncated = false;
    var it = std.mem.splitScalar(u8, body, '\n');
    while (it.next()) |text| {
        if (content.len == 0) break;
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
    const total = if (content.len == 0) 0 else std.mem.count(u8, body, "\n") + 1;
    const summary = try summarize(alloc, offset, taken, total, line_truncated, if (byte_truncated) size else null);
    errdefer alloc.free(summary);
    return .{ .text = try out.toOwnedSlice(alloc), .truncated = byte_truncated or line_truncated, .summary = summary, .lines = .{ .first = offset, .total = total } };
}

/// The detail row: the range shown, the file's length, and, when the read
/// stopped early, the offset that continues it.
fn summarize(alloc: std.mem.Allocator, offset: usize, taken: usize, total: usize, line_truncated: bool, file_size: ?usize) std.mem.Allocator.Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    if (taken == 0) {
        try out.print(alloc, "no lines at offset {d}; the file has {d}", .{ offset, total });
    } else {
        try out.print(alloc, "lines {d} to {d} of {d}", .{ offset, offset + taken - 1, total });
    }
    if (line_truncated) try out.print(alloc, " · truncated, continue with offset={d}", .{offset + taken});
    if (file_size) |size| try out.print(alloc, " · first {d} of {d} bytes only; use bash (head, wc, grep) for the rest", .{ max_bytes, size });
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
    try w.tmp.dir.writeFile(testing.io, .{ .sub_path = "b.txt", .data = "one\ntwo\n" });
    try w.tmp.dir.writeFile(testing.io, .{ .sub_path = "empty.txt", .data = "" });

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

    // A trailing newline is the end of the last line, not a line of its own:
    // an exact read of the whole file is complete, not truncated.
    var exact = try run(w.ws, testing.allocator, "{\"path\":\"b.txt\",\"count\":2}");
    defer exact.deinit(testing.allocator);
    try testing.expectEqualStrings("one\ntwo", exact.text);
    try testing.expect(!exact.truncated);
    try testing.expectEqualStrings("lines 1 to 2 of 2", exact.summary.?);

    var empty = try run(w.ws, testing.allocator, "{\"path\":\"empty.txt\"}");
    defer empty.deinit(testing.allocator);
    try testing.expectEqualStrings("", empty.text);
    try testing.expectEqualStrings("no lines at offset 1; the file has 0", empty.summary.?);

    try testing.expectError(error.OutsideWorkspace, @as(root.Workspace, w.ws).resolve(testing.allocator, "/"));
}

test "a file over the byte bound serves its first MiB with the size in the summary" {
    const alloc = testing.allocator;
    var f = testWorkspace();
    defer alloc.free(f.root_path);
    defer f.tmp.cleanup();
    // One byte over the bound: 32-byte lines, the last one cut by the bound.
    const line = "0123456789abcdef0123456789abcde\n";
    var data: std.ArrayList(u8) = .empty;
    defer data.deinit(alloc);
    while (data.items.len < max_bytes + 1) try data.appendSlice(alloc, line);
    try f.tmp.dir.writeFile(testing.io, .{ .sub_path = "big.txt", .data = data.items });

    var result = try tool.run(f.ws, alloc, "{\"path\":\"big.txt\",\"count\":2}");
    defer result.deinit(alloc);
    try testing.expect(!result.is_error);
    try testing.expect(result.truncated);
    try testing.expect(std.mem.startsWith(u8, result.text, line[0 .. line.len - 1]));
    const expected = try std.fmt.allocPrint(alloc, "first {d} of {d} bytes only", .{ max_bytes, data.items.len });
    defer alloc.free(expected);
    try testing.expect(std.mem.indexOf(u8, result.summary.?, expected) != null);
    try testing.expect(std.mem.indexOf(u8, result.summary.?, "use bash") != null);
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
