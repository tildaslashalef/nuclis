//! Open decision models, keyed by checkpoint directory, at most `capacity`;
//! opening one more closes the least recently used. Every model counts
//! against the server's memory budget (`memory.zig`), which may close one
//! to make room for any other model. Only the GPU worker opens, uses, and
//! closes a decider (`acquire`), so a model is opened once however many
//! first requests race; other threads only read which models are open,
//! under the lock.
const std = @import("std");
const inference = @import("inference");
const memory = @import("../memory.zig");

const Decider = inference.decide.Decider;

pub const capacity = 2;

/// What a decision model holds open: the files of its checkpoint directory
/// (not its subdirectories) and the backbone and projector it names.
pub fn footprint(io: std.Io, location: inference.decide.Location) u64 {
    var total: u64 = 0;
    if (std.Io.Dir.cwd().openDir(io, location.directory, .{ .iterate = true })) |opened| {
        var dir = opened;
        defer dir.close(io);
        var it = dir.iterate();
        while (it.next(io) catch null) |entry| {
            if (entry.kind != .file) continue;
            const stat = dir.statFile(io, entry.name, .{}) catch continue;
            total += stat.size;
        }
    } else |_| {}
    for ([_]?[]const u8{ location.backbone, location.mmproj }) |file| if (file) |path| {
        const stat = std.Io.Dir.cwd().statFile(io, path, .{}) catch continue;
        total += stat.size;
    };
    return total;
}

pub const Pool = struct {
    gpa: std.mem.Allocator,
    backend: inference.decide.Backend,
    /// The server's; null in tests that need no limit.
    budget: ?*memory.Budget = null,
    mutex: std.Io.Mutex = .init,
    slots: [capacity]?Slot = @splat(null),
    tick: u64 = 0,

    const Slot = struct {
        /// Owned.
        directory: []u8,
        /// The name the opening request used; owned.
        name: []u8,
        decider: Decider,
        used: u64,
    };

    pub const Acquired = struct {
        decider: *const Decider,
        /// Time spent opening it for this call; zero when it was open.
        load_ns: u64,
    };

    pub fn init(gpa: std.mem.Allocator, backend: inference.decide.Backend) Pool {
        return .{ .gpa = gpa, .backend = backend };
    }

    /// GPU worker only, after the worker stopped.
    pub fn deinit(self: *Pool, io: std.Io) void {
        for (0..capacity) |i| self.closeSlot(io, i);
    }

    /// GPU worker only: the decider for `directory`, opened (closing the
    /// least recently used when full) if it is not open.
    pub fn acquire(self: *Pool, io: std.Io, location: inference.decide.Location, name: []const u8) !Acquired {
        const directory = location.directory;
        self.tick += 1;
        for (&self.slots) |*slot| if (slot.*) |*s| if (std.mem.eql(u8, s.directory, directory)) {
            s.used = self.tick;
            self.touch(io, directory);
            return .{ .decider = &s.decider, .load_ns = 0 };
        };
        const started = std.Io.Clock.awake.now(io);
        // Asked before anything closes: a decision retried while a pinned
        // generation holds the memory must not close a model each time.
        const bytes = footprint(io, location);
        if (self.budget) |b| try b.canFit(io, bytes);
        // The slot's model closes only once the new one has opened (a
        // failed open keeps it); the room it will leave counts already.
        const index = self.victim();
        if (self.budget) |b| {
            const leaving: ?[]const u8 = if (self.slots[index]) |s| s.directory else null;
            const freed = if (leaving) |key| b.bytesOf(io, key) else 0;
            try b.reserve(io, bytes -| freed, leaving);
        }
        var decider = try Decider.open(self.gpa, io, location, self.backend);
        errdefer decider.deinit(io);
        const owned_directory = try self.gpa.dupe(u8, directory);
        errdefer self.gpa.free(owned_directory);
        const owned_name = try self.gpa.dupe(u8, name);
        errdefer self.gpa.free(owned_name);
        self.closeSlot(io, index);
        if (self.budget) |b| try b.add(io, .decision, directory, name, bytes, .{ .context = self, .close = closeKey });
        self.mutex.lockUncancelable(io);
        self.slots[index] = .{ .directory = owned_directory, .name = owned_name, .decider = decider, .used = self.tick };
        self.mutex.unlock(io);
        const load_ns: u64 = @intCast(@max(0, started.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds()));
        return .{ .decider = &self.slots[index].?.decider, .load_ns = load_ns };
    }

    /// GPU worker only: closes slot `index`, if open, and leaves the budget.
    fn closeSlot(self: *Pool, io: std.Io, index: usize) void {
        self.mutex.lockUncancelable(io);
        const slot = self.slots[index];
        self.slots[index] = null;
        self.mutex.unlock(io);
        var s = slot orelse return;
        if (self.budget) |b| b.remove(io, s.directory);
        s.decider.deinit(io);
        self.gpa.free(s.directory);
        self.gpa.free(s.name);
    }

    /// The budget's closer: the model open from directory `key`.
    fn closeKey(context: *anyopaque, io: std.Io, key: []const u8) void {
        const self: *Pool = @ptrCast(@alignCast(context));
        for (self.slots, 0..) |slot, i| if (slot) |s| if (std.mem.eql(u8, s.directory, key)) return self.closeSlot(io, i);
    }

    /// An empty slot, else the least recently used.
    fn victim(self: *const Pool) usize {
        var best: usize = 0;
        for (self.slots, 0..) |slot, i| {
            const s = slot orelse return i;
            if (s.used < self.slots[best].?.used) best = i;
        }
        return best;
    }

    /// GPU worker only: marks the model as used for the budget's order.
    fn touch(self: *Pool, io: std.Io, directory: []const u8) void {
        if (self.budget) |b| b.touch(io, directory);
    }

    /// Any thread: whether `directory` is open.
    pub fn isOpen(self: *Pool, io: std.Io, directory: []const u8) bool {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        for (self.slots) |slot| if (slot) |s| if (std.mem.eql(u8, s.directory, directory)) return true;
        return false;
    }

    /// Any thread: the names the open models were opened by, in `arena`.
    pub fn openNames(self: *Pool, io: std.Io, arena: std.mem.Allocator) ![]const []const u8 {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        var names: std.ArrayList([]const u8) = .empty;
        for (self.slots) |slot| if (slot) |s| try names.append(arena, try arena.dupe(u8, s.name));
        return names.items;
    }
};

test "the pool opens once, keeps two, and closes the least recently used" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    for ([_][]const u8{ "a", "b", "c" }) |name| {
        try tmp.dir.createDirPath(io, name);
        var sub = try tmp.dir.openDir(io, name, .{});
        defer sub.close(io);
        try @import("../../decision/tiny.zig").write(gpa, io, sub);
    }
    const root = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(root);
    var paths: [3][]u8 = undefined;
    for (&paths, [_][]const u8{ "a", "b", "c" }) |*p, name| p.* = try std.fs.path.join(gpa, &.{ root, name });
    defer for (paths) |p| gpa.free(p);

    var pool: Pool = .init(gpa, .cpu);
    defer pool.deinit(io);
    try std.testing.expect((try pool.acquire(io, .{ .directory = paths[0] }, "a")).load_ns > 0);
    try std.testing.expectEqual(@as(u64, 0), (try pool.acquire(io, .{ .directory = paths[0] }, "a")).load_ns);
    _ = try pool.acquire(io, .{ .directory = paths[1] }, "b");
    _ = try pool.acquire(io, .{ .directory = paths[0] }, "a");
    _ = try pool.acquire(io, .{ .directory = paths[2] }, "c");
    try std.testing.expect(pool.isOpen(io, paths[0]) and !pool.isOpen(io, paths[1]) and pool.isOpen(io, paths[2]));
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    try std.testing.expectEqual(@as(usize, 2), (try pool.openNames(io, arena_state.allocator())).len);
    try std.testing.expectError(error.FileNotFound, pool.acquire(io, .{ .directory = root }, "root"));
    try std.testing.expect(pool.isOpen(io, paths[0]) and pool.isOpen(io, paths[2]));
}

test "the budget closes a model to make room and refuses while a pinned one holds it" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    for ([_][]const u8{ "a", "b" }) |name| {
        try tmp.dir.createDirPath(io, name);
        var sub = try tmp.dir.openDir(io, name, .{});
        defer sub.close(io);
        try @import("../../decision/tiny.zig").write(gpa, io, sub);
    }
    const root = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(root);
    const a = try std.fs.path.join(gpa, &.{ root, "a" });
    defer gpa.free(a);
    const b = try std.fs.path.join(gpa, &.{ root, "b" });
    defer gpa.free(b);
    const size = footprint(io, .{ .directory = a });
    try std.testing.expect(size > 0);

    // Room for one model and a half: the second open closes the first.
    var budget: memory.Budget = .init(gpa, size + size / 2);
    defer budget.deinit();
    var pool: Pool = .init(gpa, .cpu);
    pool.budget = &budget;
    defer pool.deinit(io);
    _ = try pool.acquire(io, .{ .directory = a }, "a");
    _ = try pool.acquire(io, .{ .directory = b }, "b");
    try std.testing.expect(!pool.isOpen(io, a) and pool.isOpen(io, b));
    try std.testing.expectEqual(size, budget.resident(io));

    // A pinned model (a running generation) holds what `a` needs.
    const Nothing = struct {
        fn close(_: *anyopaque, _: std.Io, _: []const u8) void {}
    };
    var unused: u8 = 0;
    try budget.add(io, .language, "gen", "gen", size / 2 + 1, .{ .context = &unused, .close = Nothing.close });
    budget.pin(io, "gen", true);
    try std.testing.expectError(error.Pinned, pool.acquire(io, .{ .directory = a }, "a"));
    try std.testing.expect(pool.isOpen(io, b));
    budget.remove(io, "gen");
}
