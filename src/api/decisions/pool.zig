//! Open decision models, keyed by checkpoint directory, at most `capacity`;
//! opening one more closes the least recently used. Only the GPU worker
//! opens, uses, and closes a decider (`acquire`), so a model is opened once
//! however many first requests race; other threads only read which models
//! are open, under the lock.
const std = @import("std");
const inference = @import("inference");

const Decider = inference.decide.Decider;

pub const capacity = 2;

pub const Pool = struct {
    gpa: std.mem.Allocator,
    backend: inference.decide.Backend,
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
        for (&self.slots) |*slot| if (slot.*) |*s| {
            s.decider.deinit(io);
            self.gpa.free(s.directory);
            self.gpa.free(s.name);
            slot.* = null;
        };
    }

    /// GPU worker only: the decider for `directory`, opened (closing the
    /// least recently used when full) if it is not open.
    pub fn acquire(self: *Pool, io: std.Io, directory: []const u8, name: []const u8) !Acquired {
        self.tick += 1;
        for (&self.slots) |*slot| if (slot.*) |*s| if (std.mem.eql(u8, s.directory, directory)) {
            s.used = self.tick;
            return .{ .decider = &s.decider, .load_ns = 0 };
        };
        const started = std.Io.Clock.awake.now(io);
        var decider = try Decider.open(self.gpa, io, directory, self.backend);
        errdefer decider.deinit(io);
        const owned_directory = try self.gpa.dupe(u8, directory);
        errdefer self.gpa.free(owned_directory);
        const owned_name = try self.gpa.dupe(u8, name);
        errdefer self.gpa.free(owned_name);
        const index = self.victim();
        self.mutex.lockUncancelable(io);
        const evicted = self.slots[index];
        self.slots[index] = .{ .directory = owned_directory, .name = owned_name, .decider = decider, .used = self.tick };
        self.mutex.unlock(io);
        if (evicted) |e| {
            var old = e;
            old.decider.deinit(io);
            self.gpa.free(old.directory);
            self.gpa.free(old.name);
        }
        const load_ns: u64 = @intCast(@max(0, started.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds()));
        return .{ .decider = &self.slots[index].?.decider, .load_ns = load_ns };
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
    try std.testing.expect((try pool.acquire(io, paths[0], "a")).load_ns > 0);
    try std.testing.expectEqual(@as(u64, 0), (try pool.acquire(io, paths[0], "a")).load_ns);
    _ = try pool.acquire(io, paths[1], "b");
    _ = try pool.acquire(io, paths[0], "a");
    _ = try pool.acquire(io, paths[2], "c");
    try std.testing.expect(pool.isOpen(io, paths[0]) and !pool.isOpen(io, paths[1]) and pool.isOpen(io, paths[2]));
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    try std.testing.expectEqual(@as(usize, 2), (try pool.openNames(io, arena_state.allocator())).len);
    try std.testing.expectError(error.FileNotFound, pool.acquire(io, root, "root"));
    try std.testing.expect(pool.isOpen(io, paths[0]) and pool.isOpen(io, paths[2]));
}
