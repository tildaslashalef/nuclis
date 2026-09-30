//! Saved prefixes for `bench --prefix-cache`: a prefilled model's snapshot
//! on disk, keyed by the model files, the prefix tokens, and the session
//! layout, so a long-context measurement restores its prompt in seconds
//! instead of prefilling it for minutes. A timing aid only: a numerical
//! change to prefill leaves a saved prefix a valid state to time from, and a
//! layout change misses by digest. Files are private and never committed.
//!
//! File: one `Header`, the snapshot's session bytes, then its carried floats
//! (native byte order; the files never leave the machine that wrote them).
const std = @import("std");
const inference = @import("inference");

const Snapshot = inference.session.Snapshot;

pub const Key = struct {
    /// `modelDigest` of the target and draft files.
    model: u64,
    /// `tokenDigest` of the prefix.
    tokens: u64,
    /// The session's `layout_digest` (layouts, precision, and capacity).
    layout: u64,
};

const magic = "NUCLSNAP".*;
const version: u32 = 1;

const Header = extern struct {
    magic: [8]u8 = magic,
    version: u32 = version,
    reserved: u32 = 0,
    model: u64,
    tokens: u64,
    layout: u64,
    position: u64,
    capacity: u64,
    memory_bytes: u64,
    carried_floats: u64,
    /// Wyhash of the session bytes then the carried bytes.
    content: u64,
};

/// Bytes of each model file hashed with its size: the GGUF header and
/// metadata, which name the tensors and their encodings. The weights
/// themselves are not read (16 GB per run would defeat the cache); the
/// catalogue pins the files by SHA-256 and they are never edited in place.
const head_bytes = 1024 * 1024;

/// Digest of the target file and, when a drafter is loaded, the entry's
/// draft companion: each file's size and first MiB. A companion that does
/// not exist (a family that loaded its embedded block) hashes as missing.
pub fn modelDigest(io: std.Io, model_path: []const u8, draft_path: ?[]const u8) !u64 {
    var hasher = std.hash.Wyhash.init(0);
    var head: [4096]u8 = undefined;
    for ([_]?[]const u8{ model_path, draft_path }, 0..) |maybe, i| {
        const path = maybe orelse {
            hasher.update("none");
            continue;
        };
        const file = std.Io.Dir.cwd().openFile(io, path, .{}) catch |err| switch (err) {
            error.FileNotFound => if (i == 1) {
                hasher.update("missing");
                continue;
            } else return err,
            else => return err,
        };
        defer file.close(io);
        const size = (try file.stat(io)).size;
        hasher.update(std.mem.asBytes(&size));
        var offset: u64 = 0;
        while (offset < @min(size, head_bytes)) {
            const got = try file.readPositionalAll(io, &head, offset);
            if (got == 0) break;
            hasher.update(head[0..got]);
            offset += got;
        }
    }
    return hasher.final();
}

pub fn tokenDigest(tokens: []const u32) u64 {
    return std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(tokens));
}

/// `<model>-<tokens>-<layout>.snap`, 16 hex digits each.
pub fn fileName(buffer: *[64]u8, key: Key) []const u8 {
    return std.fmt.bufPrint(buffer, "{x:0>16}-{x:0>16}-{x:0>16}.snap", .{ key.model, key.tokens, key.layout }) catch unreachable;
}

fn contentDigest(snap: *const Snapshot) u64 {
    var hasher = std.hash.Wyhash.init(0);
    hasher.update(snap.memory);
    hasher.update(std.mem.sliceAsBytes(snap.carried));
    return hasher.final();
}

/// Writes `snap` under `dir` by its key, through a temporary file renamed
/// into place, so a reader never sees a partial file. A snapshot with
/// position spans (an image prompt) is refused: the bench never makes one.
pub fn save(io: std.Io, dir: std.Io.Dir, key: Key, snap: *const Snapshot) !void {
    if (snap.span_count != 0) return error.PrefixHasSpans;
    if (snap.layout_digest != key.layout) return error.PrefixMismatch;
    const header: Header = .{
        .model = key.model,
        .tokens = key.tokens,
        .layout = key.layout,
        .position = snap.position,
        .capacity = snap.capacity,
        .memory_bytes = snap.memory.len,
        .carried_floats = snap.carried.len,
        .content = contentDigest(snap),
    };
    var name: [64]u8 = undefined;
    var atomic = try dir.createFileAtomic(io, fileName(&name, key), .{ .replace = true });
    defer atomic.deinit(io);
    try atomic.file.writeStreamingAll(io, std.mem.asBytes(&header));
    try atomic.file.writeStreamingAll(io, snap.memory);
    try atomic.file.writeStreamingAll(io, std.mem.sliceAsBytes(snap.carried));
    try atomic.replace(io);
}

/// The saved prefix for `key`, or null when none exists. A file whose header
/// does not match the key or whose size does not match the header is
/// `PrefixMismatch`; one whose content digest differs is `PrefixCorrupt`.
/// The caller owns the snapshot (allocated from `gpa`) and restores it into
/// a model whose layout digest is `key.layout`.
pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, key: Key) !?Snapshot {
    var name: [64]u8 = undefined;
    const file = dir.openFile(io, fileName(&name, key), .{}) catch |err| switch (err) {
        error.FileNotFound => return null,
        else => return err,
    };
    defer file.close(io);
    const size = (try file.stat(io)).size;
    var header: Header = undefined;
    if (size < @sizeOf(Header) or try file.readPositionalAll(io, std.mem.asBytes(&header), 0) != @sizeOf(Header)) return error.PrefixMismatch;
    if (!std.mem.eql(u8, &header.magic, &magic) or header.version != version) return error.PrefixMismatch;
    if (header.model != key.model or header.tokens != key.tokens or header.layout != key.layout) return error.PrefixMismatch;
    const carried_bytes = std.math.mul(u64, header.carried_floats, @sizeOf(f32)) catch return error.PrefixMismatch;
    const payload = std.math.add(u64, header.memory_bytes, carried_bytes) catch return error.PrefixMismatch;
    if (size - @sizeOf(Header) != payload or header.position > header.capacity) return error.PrefixMismatch;
    const memory = try gpa.alloc(u8, @intCast(header.memory_bytes));
    errdefer gpa.free(memory);
    const carried = try gpa.alloc(f32, @intCast(header.carried_floats));
    errdefer gpa.free(carried);
    if (try file.readPositionalAll(io, memory, @sizeOf(Header)) != memory.len) return error.PrefixMismatch;
    if (try file.readPositionalAll(io, std.mem.sliceAsBytes(carried), @sizeOf(Header) + header.memory_bytes) != carried_bytes) return error.PrefixMismatch;
    const snap: Snapshot = .{
        .gpa = gpa,
        .memory = memory,
        .position = @intCast(header.position),
        .capacity = @intCast(header.capacity),
        .layout_digest = header.layout,
        .carried = carried,
    };
    if (contentDigest(&snap) != header.content) return error.PrefixCorrupt;
    return snap;
}

test "a saved prefix round-trips into a fresh session" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const Session = inference.session.Session;
    const layouts = [_]inference.session.Layout{ .{ .attention = .{ .key_row = 2, .value_row = 2, .precision = .f16 } }, .{ .recurrent = .{ .history = 2, .matrix = 2 } } };
    var s = try Session.init(a, &layouts, 8, false, 0);
    defer s.deinit();
    try s.beginChunk(3);
    try s.commitChunk(3);
    @memset(s.layers[0].attention.keys.range(0, 3), 0x5a);
    s.layers[1].recurrent.matrix[1] = 7;
    var snap = try s.snapshot(a);
    defer snap.deinit();
    snap.carried = try a.dupe(f32, &.{ 1.5, -2 });
    const key: Key = .{ .model = 1, .tokens = tokenDigest(&.{ 3, 4, 5 }), .layout = s.layout_digest };
    try std.testing.expect(try load(a, io, tmp.dir, key) == null);
    try save(io, tmp.dir, key, &snap);
    var back = (try load(a, io, tmp.dir, key)).?;
    defer back.deinit();
    try std.testing.expectEqualSlices(u8, snap.memory, back.memory);
    try std.testing.expectEqualSlices(f32, snap.carried, back.carried);
    var fresh = try Session.init(a, &layouts, 8, false, 0);
    defer fresh.deinit();
    try fresh.restore(&back);
    try std.testing.expectEqual(@as(usize, 3), fresh.position);
    try std.testing.expectEqual(@as(f32, 7), fresh.layers[1].recurrent.matrix[1]);
    // Another prefix is another file; another layout is refused on save.
    try std.testing.expect(try load(a, io, tmp.dir, .{ .model = 1, .tokens = tokenDigest(&.{ 3, 4 }), .layout = key.layout }) == null);
    try std.testing.expectError(error.PrefixMismatch, save(io, tmp.dir, .{ .model = 1, .tokens = key.tokens, .layout = key.layout + 1 }, &snap));
}

test "a saved prefix whose size or content changed is refused" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const memory = try a.dupe(u8, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    var snap: Snapshot = .{ .gpa = a, .memory = memory, .position = 1, .capacity = 4, .layout_digest = 9 };
    defer snap.deinit();
    const key: Key = .{ .model = 2, .tokens = 3, .layout = 9 };
    try save(io, tmp.dir, key, &snap);
    var name: [64]u8 = undefined;
    const path = fileName(&name, key);
    // One flipped payload byte: same size, different digest.
    {
        const file = try tmp.dir.openFile(io, path, .{ .mode = .read_write });
        defer file.close(io);
        try file.writePositionalAll(io, &.{0xff}, @sizeOf(Header) + 2);
    }
    try std.testing.expectError(error.PrefixCorrupt, load(a, io, tmp.dir, key));
    // One byte more than the header declares.
    {
        const file = try tmp.dir.openFile(io, path, .{ .mode = .read_write });
        defer file.close(io);
        try file.writePositionalAll(io, &.{0}, @sizeOf(Header) + memory.len);
    }
    try std.testing.expectError(error.PrefixMismatch, load(a, io, tmp.dir, key));
}
