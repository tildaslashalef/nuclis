//! Prompt history on disk: `~/.nuclis/agent/history.jsonl`, one JSON object
//! per line, oldest first. `src/tui/editor.zig` owns the in-memory list and
//! the recall keys; this file is the only thing that touches the filesystem,
//! because the terminal surface does no I/O.
//!
//! The rules, in the same spirit as the session files (agent-spec § Sessions
//! and storage):
//!
//! - **Append-only.** A submitted prompt is one `write` at the end of the
//!   file. Nothing rewrites or compacts it, so a crash can only ever cost
//!   the line being written.
//! - **A truncated last line is dropped on load**, and so is any line that
//!   does not parse: history is a convenience, and refusing to start the
//!   agent because one line is malformed would be the wrong trade. (A
//!   *session* file is different — there a bad line in the middle is a typed
//!   error, because it is the record of a conversation.)
//! - **Only the tail is read.** The file grows for the life of the
//!   installation; `load` reads at most `max_read` bytes from the end and
//!   discards the first partial line, so startup cost is bounded no matter
//!   how long the file gets.
//! - **A very long prompt is not persisted.** A 100 KB paste is not
//!   something anyone recalls with Up, and writing it would make every later
//!   startup read it. It stays in the session's in-memory history.
//!
//! Ownership: `load` returns strings allocated from the caller's allocator
//! in a list the caller frees; `Loaded.deinit` does both.
const std = @import("std");

const Allocator = std.mem.Allocator;

/// Bytes read from the end of the file at startup.
pub const max_read = 256 * 1024;
/// Prompts longer than this stay in memory only.
pub const max_persisted = 8 * 1024;
/// Entries kept from the tail; the editor's own bound is the same order.
pub const max_entries = 200;

pub const Loaded = struct {
    alloc: Allocator,
    items: []const []const u8,

    pub fn deinit(self: Loaded) void {
        for (self.items) |item| self.alloc.free(item);
        self.alloc.free(self.items);
    }
};

/// Reads the tail of `path` (relative to `dir`, or absolute with
/// `Dir.cwd()`). A missing file, an unreadable one, or a file of nothing but
/// unparsable lines all yield an empty history: the agent starts either way.
pub fn load(alloc: Allocator, io: std.Io, dir: std.Io.Dir, path: []const u8) !Loaded {
    var items: std.ArrayList([]const u8) = .empty;
    errdefer {
        for (items.items) |item| alloc.free(item);
        items.deinit(alloc);
    }
    const bytes = readTail(alloc, io, dir, path) catch |err| switch (err) {
        error.FileNotFound => return .{ .alloc = alloc, .items = try items.toOwnedSlice(alloc) },
        else => return err,
    };
    defer alloc.free(bytes);
    var lines = std.mem.splitScalar(u8, bytes, '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        const parsed = std.json.parseFromSlice(std.json.Value, alloc, line, .{}) catch continue;
        defer parsed.deinit();
        const object = switch (parsed.value) {
            .object => |object| object,
            else => continue,
        };
        const value = object.get("text") orelse continue;
        const t = switch (value) {
            .string => |s| s,
            else => continue,
        };
        if (t.len == 0) continue;
        try items.append(alloc, try alloc.dupe(u8, t));
    }
    // Keep the newest entries; the editor recalls backwards from the end.
    if (items.items.len > max_entries) {
        const drop = items.items.len - max_entries;
        for (items.items[0..drop]) |item| alloc.free(item);
        std.mem.copyForwards([]const u8, items.items[0..max_entries], items.items[drop..]);
        items.shrinkRetainingCapacity(max_entries);
    }
    return .{ .alloc = alloc, .items = try items.toOwnedSlice(alloc) };
}

/// The last `max_read` bytes of the file, starting after the first newline
/// so a partial line at the cut is never parsed as a whole one.
fn readTail(alloc: Allocator, io: std.Io, dir: std.Io.Dir, path: []const u8) ![]u8 {
    const bytes = dir.readFileAlloc(io, path, alloc, .limited(max_read)) catch |err| switch (err) {
        error.StreamTooLong => return readCut(alloc, io, dir, path),
        else => return err,
    };
    return bytes;
}

fn readCut(alloc: Allocator, io: std.Io, dir: std.Io.Dir, path: []const u8) ![]u8 {
    const file = try dir.openFile(io, path, .{});
    defer file.close(io);
    const size = (try file.stat(io)).size;
    const from = size - @min(size, max_read);
    var reader_buffer: [4096]u8 = undefined;
    var reader = file.readerStreaming(io, &reader_buffer);
    try reader.seekTo(from);
    const bytes = try reader.interface.allocRemaining(alloc, .limited(max_read));
    errdefer alloc.free(bytes);
    if (from == 0) return bytes;
    // Drop everything up to and including the first newline: the cut landed
    // in the middle of a line.
    const cut = std.mem.indexOfScalar(u8, bytes, '\n') orelse bytes.len;
    const kept = try alloc.dupe(u8, bytes[@min(cut + 1, bytes.len)..]);
    alloc.free(bytes);
    return kept;
}

/// Appends one submitted prompt, creating `~/.nuclis/agent/` on the first
/// write. A prompt over `max_persisted` is skipped, and so is an empty one.
pub fn append(io: std.Io, dir: std.Io.Dir, path: []const u8, text: []const u8, now: []const u8) !void {
    if (text.len == 0 or text.len > max_persisted) return;
    if (std.fs.path.dirname(path)) |parent| try dir.createDirPath(io, parent);
    const file = try dir.createFile(io, path, .{ .truncate = false });
    defer file.close(io);
    const end = (try file.stat(io)).size;
    var buffer: [4096]u8 = undefined;
    var writer = file.writerStreaming(io, &buffer);
    try writer.seekTo(end);
    try writeEntry(&writer.interface, text, now);
    try writer.interface.flush();
}

/// One line of the file. Split out so the format is tested without a
/// filesystem.
pub fn writeEntry(out: *std.Io.Writer, text: []const u8, now: []const u8) !void {
    var s: std.json.Stringify = .{ .writer = out };
    try s.beginObject();
    try s.objectField("text");
    try s.write(text);
    if (now.len > 0) {
        try s.objectField("time");
        try s.write(now);
    }
    try s.endObject();
    try out.writeByte('\n');
}

// ----- tests -----

const testing = std.testing;

test "an entry is one JSON object per line" {
    var buffer: std.Io.Writer.Allocating = .init(testing.allocator);
    defer buffer.deinit();
    try writeEntry(&buffer.writer, "hello \"there\"\nsecond line", "2026-09-12T10:00:00Z");
    try testing.expectEqualStrings(
        "{\"text\":\"hello \\\"there\\\"\\nsecond line\",\"time\":\"2026-09-12T10:00:00Z\"}\n",
        buffer.written(),
    );
}

test "a missing file is an empty history, not an error" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const loaded = try load(testing.allocator, testing.io, tmp.dir, "history.jsonl");
    defer loaded.deinit();
    try testing.expectEqual(@as(usize, 0), loaded.items.len);
}

test "prompts round trip through the file, oldest first" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try append(testing.io, tmp.dir, "agent/history.jsonl", "first", "t1");
    try append(testing.io, tmp.dir, "agent/history.jsonl", "second\nline", "t2");
    // Over the bound: kept in memory by the caller, never written.
    const huge = try testing.allocator.alloc(u8, max_persisted + 1);
    defer testing.allocator.free(huge);
    @memset(huge, 'x');
    try append(testing.io, tmp.dir, "agent/history.jsonl", huge, "t3");
    try append(testing.io, tmp.dir, "agent/history.jsonl", "", "t4");

    const loaded = try load(testing.allocator, testing.io, tmp.dir, "agent/history.jsonl");
    defer loaded.deinit();
    try testing.expectEqual(@as(usize, 2), loaded.items.len);
    try testing.expectEqualStrings("first", loaded.items[0]);
    try testing.expectEqualStrings("second\nline", loaded.items[1]);
}

test "a truncated or malformed line is dropped, the rest still loads" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(testing.io, .{
        .sub_path = "history.jsonl",
        .data =
        \\{"text":"good"}
        \\not json at all
        \\{"text":42}
        \\{"other":"key"}
        \\{"text":"also good"}
        \\{"text":"trunc
        ,
    });
    const loaded = try load(testing.allocator, testing.io, tmp.dir, "history.jsonl");
    defer loaded.deinit();
    try testing.expectEqual(@as(usize, 2), loaded.items.len);
    try testing.expectEqualStrings("good", loaded.items[0]);
    try testing.expectEqualStrings("also good", loaded.items[1]);
}

test "only the newest entries are kept" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buffer: [16]u8 = undefined;
    for (0..max_entries + 5) |i| {
        try append(testing.io, tmp.dir, "history.jsonl", try std.fmt.bufPrint(&buffer, "p{d}", .{i}), "");
    }
    const loaded = try load(testing.allocator, testing.io, tmp.dir, "history.jsonl");
    defer loaded.deinit();
    try testing.expectEqual(max_entries, loaded.items.len);
    try testing.expectEqualStrings("p5", loaded.items[0]);
    try testing.expectEqualStrings("p204", loaded.items[max_entries - 1]);
}
