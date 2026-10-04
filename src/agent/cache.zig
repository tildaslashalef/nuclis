//! The agent's token cache: model states saved at turn boundaries, so a
//! conversation that begins with a saved prefix restores it instead of
//! prefilling it again (docs/reference/session.md § The agent's token cache).
//!
//! Recurrent layers cannot be rewound by truncating the KV cache, so a state
//! resumes only where a snapshot was taken: the cache is a set of snapshots
//! and a longest-prefix lookup. Entries match on the rendered text they
//! consumed, not on tokens, because a turn's generated tokens need not be the
//! canonical encoding of their text. Every boundary sits before a control
//! token, which BPE never merges across, so the remainder encodes alike.
//!
//! `Memory` lives in the process, least recently used out first under a byte
//! budget. `Disk` keeps primed prefixes across processes through
//! `inference.prefix_cache`, keyed by the model files, the executor, and the
//! build; a hit refreshes a file's modification time, the eviction order.
//! Neither touches the engine: the caller snapshots and restores.
const std = @import("std");
const inference = @import("inference");
const build_options = @import("build_options");
const paths = @import("../paths.zig");

const Allocator = std.mem.Allocator;
const Snapshot = inference.session.Snapshot;
const prefix_cache = inference.prefix_cache;

/// One saved state. Owns its text, history, and snapshot.
pub const Entry = struct {
    /// The rendered text the state consumed.
    text: []u8,
    /// Tokens the state holds (its position).
    tokens: usize,
    /// The penalty history at that point; null when the caller keeps none.
    history: ?std.bit_set.Dynamic = null,
    snapshot: Snapshot,
    /// `Memory.clock` at the last insert or hit.
    used: u64 = 0,

    pub fn bytes(self: *const Entry) usize {
        return self.snapshot.memory.len + self.snapshot.carried.len * @sizeOf(f32) + self.text.len;
    }

    pub fn deinit(self: *Entry, alloc: Allocator) void {
        alloc.free(self.text);
        if (self.history) |*h| h.deinit(alloc);
        self.snapshot.deinit();
        self.* = undefined;
    }
};

/// The in-process tier. Pointers it returns stay valid until the next
/// `insert`, `remove`, or `clear`.
pub const Memory = struct {
    alloc: Allocator,
    /// The most bytes the entries may hold (`cache.memory_bytes`); 0 keeps
    /// nothing. A Qwen state is about 150 MB plus 64 KiB per token.
    budget: u64,
    entries: std.ArrayList(Entry) = .empty,
    clock: u64 = 0,

    pub fn deinit(self: *Memory) void {
        self.clear();
        self.entries.deinit(self.alloc);
    }

    pub fn clear(self: *Memory) void {
        for (self.entries.items) |*e| e.deinit(self.alloc);
        self.entries.clearRetainingCapacity();
    }

    pub fn bytes(self: *const Memory) u64 {
        var total: u64 = 0;
        for (self.entries.items) |*e| total += e.bytes();
        return total;
    }

    /// The entry whose text is `text`, marked used.
    pub fn exact(self: *Memory, text: []const u8) ?*Entry {
        for (self.entries.items) |*e| if (std.mem.eql(u8, e.text, text)) return self.touch(e);
        return null;
    }

    /// The longest entry whose text is a proper prefix of `full`, marked
    /// used. Proper, so at least one token is left to prefill: the next
    /// sample needs its logits.
    pub fn longest(self: *Memory, full: []const u8) ?*Entry {
        var best: ?*Entry = null;
        for (self.entries.items) |*e| {
            if (e.text.len >= full.len or !std.mem.startsWith(u8, full, e.text)) continue;
            if (best == null or e.text.len > best.?.text.len) best = e;
        }
        return if (best) |e| self.touch(e) else null;
    }

    /// Takes `entry`: replaces one with the same text, then evicts the least
    /// recently used until the budget holds it. An entry larger than the
    /// budget, or one the list cannot grow for, is freed instead; the
    /// result says which.
    pub fn insert(self: *Memory, entry: Entry) bool {
        var owned = entry;
        if (owned.bytes() > self.budget) {
            owned.deinit(self.alloc);
            return false;
        }
        if (self.exact(owned.text)) |old| self.remove(old);
        while (self.bytes() + owned.bytes() > self.budget) self.remove(self.oldest().?);
        self.entries.append(self.alloc, owned) catch {
            owned.deinit(self.alloc);
            return false;
        };
        _ = self.touch(&self.entries.items[self.entries.items.len - 1]);
        return true;
    }

    pub fn remove(self: *Memory, entry: *Entry) void {
        const index = (@intFromPtr(entry) - @intFromPtr(self.entries.items.ptr)) / @sizeOf(Entry);
        entry.deinit(self.alloc);
        _ = self.entries.swapRemove(index);
    }

    fn oldest(self: *Memory) ?*Entry {
        var found: ?*Entry = null;
        for (self.entries.items) |*e| if (found == null or e.used < found.?.used) {
            found = e;
        };
        return found;
    }

    fn touch(self: *Memory, entry: *Entry) *Entry {
        self.clock += 1;
        entry.used = self.clock;
        return entry;
    }
};

/// The model half of a disk key: the model files (`prefix_cache.modelDigest`),
/// the executor, and the build, so a state is restored only into the code
/// that computed it.
pub fn modelKey(files: u64, backend: []const u8, build: []const u8) u64 {
    var hasher = std.hash.Wyhash.init(files);
    hasher.update(backend);
    hasher.update(&.{0});
    hasher.update(build);
    return hasher.final();
}

/// The across-process tier: `<root>/cache/prefix/`, one `prefix_cache` file
/// per state. Owns the open directory.
pub const Disk = struct {
    io: std.Io,
    dir: std.Io.Dir,
    /// `modelKey` of the loaded model.
    model: u64,
    /// The most bytes the files may hold; 0 is a disabled tier, never opened.
    budget: u64,

    /// Opens, creating it when absent, the directory at `path`.
    pub fn open(io: std.Io, path: []const u8, model: u64, budget: u64) !Disk {
        const dir = try std.Io.Dir.cwd().createDirPathOpen(io, path, .{ .open_options = .{ .iterate = true } });
        return .{ .io = io, .dir = dir, .model = model, .budget = budget };
    }

    pub fn close(self: *Disk) void {
        self.dir.close(self.io);
        self.* = undefined;
    }

    pub fn key(self: *const Disk, tokens: []const u32, layout: u64) prefix_cache.Key {
        return .{ .model = self.model, .tokens = prefix_cache.tokenDigest(tokens), .layout = layout };
    }

    /// The state saved under `key`, or null. A file that does not match its
    /// name or whose content changed is deleted and reads as a miss; a hit
    /// becomes the newest file for eviction.
    pub fn load(self: *Disk, gpa: Allocator, k: prefix_cache.Key) !?Snapshot {
        var name: [64]u8 = undefined;
        const file = prefix_cache.fileName(&name, k);
        const snap = prefix_cache.load(gpa, self.io, self.dir, k) catch |err| switch (err) {
            error.PrefixMismatch, error.PrefixCorrupt => {
                self.dir.deleteFile(self.io, file) catch {};
                return null;
            },
            else => return err,
        };
        if (snap != null) self.dir.setTimestampsNow(self.io, file, .{}) catch {};
        return snap;
    }

    /// Writes `snap` under `key`, then evicts the oldest files beyond the
    /// budget. `alloc` holds the directory listing only.
    pub fn save(self: *Disk, alloc: Allocator, k: prefix_cache.Key, snap: *const Snapshot) !void {
        try prefix_cache.save(self.io, self.dir, k, snap);
        try self.evict(alloc);
    }

    /// Deletes the least recently used files until the rest fit the budget.
    pub fn evict(self: *Disk, alloc: Allocator) !void {
        const files = try self.list(alloc);
        defer freeList(alloc, files);
        std.mem.sort(File, files, {}, File.olderFirst);
        var total: u64 = 0;
        for (files) |f| total += f.size;
        for (files) |f| {
            if (total <= self.budget) break;
            self.dir.deleteFile(self.io, f.name) catch continue;
            total -= f.size;
        }
    }

    pub const File = struct {
        name: []u8,
        size: u64,
        mtime: std.Io.Timestamp,

        fn olderFirst(_: void, a: File, b: File) bool {
            return a.mtime.nanoseconds < b.mtime.nanoseconds;
        }
    };

    /// The saved states, in directory order. Free with `freeList`.
    pub fn list(self: *Disk, alloc: Allocator) ![]File {
        var files: std.ArrayList(File) = .empty;
        errdefer {
            for (files.items) |f| alloc.free(f.name);
            files.deinit(alloc);
        }
        var it = self.dir.iterate();
        while (try it.next(self.io)) |entry| {
            if (entry.kind != .file or !std.mem.endsWith(u8, entry.name, ".snap")) continue;
            const stat = self.dir.statFile(self.io, entry.name, .{}) catch continue;
            const name = try alloc.dupe(u8, entry.name);
            files.append(alloc, .{ .name = name, .size = stat.size, .mtime = stat.mtime }) catch |err| {
                alloc.free(name);
                return err;
            };
        }
        return files.toOwnedSlice(alloc);
    }

    pub fn freeList(alloc: Allocator, files: []File) void {
        for (files) |f| alloc.free(f.name);
        alloc.free(files);
    }
};

/// This build's identity: the version and commit, plus the executable's size
/// and modification time when the tree differed from the commit, since two
/// such builds share one. Null when a modified build cannot tell itself
/// apart, which disables the disk tier.
pub fn buildId(io: std.Io, buffer: []u8) ?[]const u8 {
    const base = std.fmt.bufPrint(buffer, "{s}+{s}", .{ build_options.version, build_options.revision }) catch return null;
    if (!build_options.dirty and !std.mem.eql(u8, build_options.revision, "unknown")) return base;
    const exe = std.process.openExecutable(io, .{}) catch return null;
    defer exe.close(io);
    const stat = exe.stat(io) catch return null;
    const suffix = std.fmt.bufPrint(buffer[base.len..], "-dirty.{d}.{d}", .{ stat.size, stat.mtime.nanoseconds }) catch return null;
    return buffer[0 .. base.len + suffix.len];
}

/// The disk tier for the model at `model_path` (and `draft_path`, when a
/// drafter is loaded) on `backend`, or null: no user root, a zero budget, an
/// unidentifiable build, or a directory that cannot be opened. A cache that
/// cannot open costs prefill time, never the session.
pub fn openDisk(alloc: Allocator, io: std.Io, root_dir: ?[]const u8, budget: u64, model_path: []const u8, draft_path: ?[]const u8, backend: []const u8) ?Disk {
    const root = root_dir orelse return null;
    if (budget == 0) return null;
    var buffer: [256]u8 = undefined;
    const build = buildId(io, &buffer) orelse return null;
    const files = prefix_cache.modelDigest(io, model_path, draft_path) catch return null;
    const path = paths.prefixCachePath(alloc, root) catch return null;
    defer alloc.free(path);
    return Disk.open(io, path, modelKey(files, backend, build), budget) catch null;
}

// ----- tests -----

const testing = std.testing;

/// A model-free snapshot of `n` bytes, each `fill`.
fn fakeSnapshot(n: usize, fill: u8) !Snapshot {
    const memory = try testing.allocator.alloc(u8, n);
    @memset(memory, fill);
    return .{ .gpa = testing.allocator, .memory = memory, .position = n, .capacity = n, .layout_digest = 7 };
}

fn fakeEntry(text: []const u8, n: usize) !Entry {
    const owned = try testing.allocator.dupe(u8, text);
    errdefer testing.allocator.free(owned);
    return .{ .text = owned, .tokens = n, .snapshot = try fakeSnapshot(n, @truncate(text.len)) };
}

test "the longest proper prefix wins, and an exact render is not a hit" {
    var m: Memory = .{ .alloc = testing.allocator, .budget = 1000 };
    defer m.deinit();
    try testing.expect(m.insert(try fakeEntry("system", 10)));
    try testing.expect(m.insert(try fakeEntry("system user hi answer", 20)));
    try testing.expect(m.insert(try fakeEntry("system user other", 20)));
    try testing.expectEqualStrings("system user hi answer", m.longest("system user hi answer<end> user more").?.text);
    try testing.expectEqualStrings("system", m.longest("system user third").?.text);
    // Nothing left to prefill is no hit; a different system block misses.
    try testing.expectEqualStrings("system", m.longest("system user hi answer").?.text);
    try testing.expect(m.longest("other system") == null);
    try testing.expectEqualStrings("system", m.exact("system").?.text);
}

test "the budget evicts the least recently used and refuses an entry larger than itself" {
    var m: Memory = .{ .alloc = testing.allocator, .budget = 100 };
    defer m.deinit();
    try testing.expect(m.insert(try fakeEntry("a", 40)));
    try testing.expect(m.insert(try fakeEntry("b", 40)));
    // A hit on `a` makes `b` the oldest: the third entry evicts `b`.
    _ = m.exact("a").?;
    try testing.expect(m.insert(try fakeEntry("c", 40)));
    try testing.expect(m.exact("b") == null);
    try testing.expect(m.exact("a") != null and m.exact("c") != null);
    try testing.expect(m.bytes() <= 100);
    try testing.expect(!m.insert(try fakeEntry("d", 200)));
    // The same text replaces its entry instead of holding two.
    try testing.expect(m.insert(try fakeEntry("a", 10)));
    try testing.expectEqual(@as(usize, 2), m.entries.items.len);
    try testing.expectEqual(@as(usize, 10), m.exact("a").?.tokens);
    var off: Memory = .{ .alloc = testing.allocator, .budget = 0 };
    defer off.deinit();
    try testing.expect(!off.insert(try fakeEntry("a", 1)));
}

test "a disk entry round-trips, misses on another build, and refuses a corrupt file" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const path = try tmp.dir.realPathFileAlloc(io, ".", testing.allocator);
    defer testing.allocator.free(path);
    var disk = try Disk.open(io, path, modelKey(1, "metal", "0.5.0-dev+abc"), 1 << 20);
    defer disk.close();
    var snap = try fakeSnapshot(32, 3);
    defer snap.deinit();
    const k = disk.key(&.{ 1, 2, 3 }, snap.layout_digest);
    try disk.save(testing.allocator, k, &snap);
    var back = (try disk.load(testing.allocator, k)).?;
    defer back.deinit();
    try testing.expectEqualSlices(u8, snap.memory, back.memory);
    // Another build or executor is another key: a miss, never a restore.
    var other = k;
    other.model = modelKey(1, "metal", "0.5.0-dev+abd");
    try testing.expect(try disk.load(testing.allocator, other) == null);
    try testing.expect(modelKey(1, "cpu", "0.5.0-dev+abc") != k.model);
    // One flipped byte: refused, and the file is gone.
    var name: [64]u8 = undefined;
    const file = prefix_cache.fileName(&name, k);
    {
        const f = try disk.dir.openFile(io, file, .{ .mode = .read_write });
        defer f.close(io);
        const size = (try f.stat(io)).size;
        try f.writePositionalAll(io, &.{0xff}, size - 1);
    }
    try testing.expect(try disk.load(testing.allocator, k) == null);
    try testing.expectError(error.FileNotFound, disk.dir.statFile(io, file, .{}));
}

test "the disk budget deletes the least recently used files first" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const path = try tmp.dir.realPathFileAlloc(io, ".", testing.allocator);
    defer testing.allocator.free(path);
    var disk = try Disk.open(io, path, 5, 1 << 20);
    defer disk.close();
    var snap = try fakeSnapshot(1000, 1);
    defer snap.deinit();
    var keys: [3]prefix_cache.Key = undefined;
    for (&keys, 0..) |*k, i| {
        k.* = disk.key(&.{@intCast(i)}, snap.layout_digest);
        try disk.save(testing.allocator, k.*, &snap);
        // Explicit times: a fast file system may stamp all three alike.
        var name: [64]u8 = undefined;
        try disk.dir.setTimestamps(io, prefix_cache.fileName(&name, k.*), .{ .modify_timestamp = .{ .new = .{ .nanoseconds = @as(i96, @intCast(i + 1)) * std.time.ns_per_s } } });
    }
    // Room for two: the oldest (the first) goes.
    const files = try disk.list(testing.allocator);
    const one = files[0].size;
    Disk.freeList(testing.allocator, files);
    disk.budget = 2 * one;
    try disk.evict(testing.allocator);
    try testing.expect(try disk.load(testing.allocator, keys[0]) == null);
    var kept = (try disk.load(testing.allocator, keys[1])).?;
    kept.deinit();
    const left = try disk.list(testing.allocator);
    defer Disk.freeList(testing.allocator, left);
    try testing.expectEqual(@as(usize, 2), left.len);
}
