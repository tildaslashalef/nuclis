//! One open model of a kind the server keeps a single one of (a voice, a
//! transcriber), keyed by its checkpoint directory; opening another closes
//! it. It counts its files against the server's memory budget
//! (`memory.zig`) and is pinned there while a request holds it across GPU
//! items (`hold`). Only the GPU worker opens, uses, and closes the model;
//! other tasks only read which one is open.
const std = @import("std");
const memory = @import("memory.zig");

/// `Model` is the opened thing; `open` and `close` are its constructor and
/// destructor on the worker.
pub fn Slot(comptime Model: type, comptime kind: memory.Kind) type {
    return struct {
        const Self = @This();

        pub const Open = *const fn (gpa: std.mem.Allocator, io: std.Io, directory: []const u8, options: Options) anyerror!*Model;
        pub const Close = *const fn (model: *Model, io: std.Io) void;
        pub const Options = struct { metal: bool };

        gpa: std.mem.Allocator,
        open_fn: Open,
        close_fn: Close,
        options: Options,
        /// The server's; null in tests that need no limit.
        budget: ?*memory.Budget = null,
        /// Guards `current`'s identity for readers on other tasks.
        mutex: std.Io.Mutex = .init,
        current: ?Current = null,
        /// Requests holding the model across items: the budget may not close
        /// it, and no other may replace it, while any does.
        held: u32 = 0,

        const Current = struct {
            /// Owned: the checkpoint directory and the name it was opened by.
            directory: []u8,
            name: []u8,
            model: *Model,
            bytes: u64,
        };

        pub const Acquired = struct {
            model: *Model,
            /// Time spent opening it for this call; zero when it was open.
            load_ns: u64,
        };

        pub fn init(gpa: std.mem.Allocator, open_fn: Open, close_fn: Close, options: Options) Self {
            return .{ .gpa = gpa, .open_fn = open_fn, .close_fn = close_fn, .options = options };
        }

        /// GPU worker only, after the worker stopped.
        pub fn deinit(self: *Self, io: std.Io) void {
            self.close(io);
        }

        /// GPU worker only: the model of `directory`, opened (closing the
        /// open one) when it is not open. `bytes` is what it counts.
        pub fn acquire(self: *Self, io: std.Io, directory: []const u8, name: []const u8, bytes: u64) !Acquired {
            if (self.current) |c| if (std.mem.eql(u8, c.directory, directory)) {
                if (self.budget) |b| b.touch(io, directory);
                return .{ .model = c.model, .load_ns = 0 };
            };
            if (self.held > 0) return error.Pinned;
            const started = std.Io.Clock.awake.now(io);
            if (self.budget) |b| {
                // Asked before anything closes, as the other pools do.
                try b.canFit(io, bytes);
                const leaving: ?[]const u8 = if (self.current) |c| c.directory else null;
                try b.reserve(io, bytes -| if (leaving) |key| b.bytesOf(io, key) else 0, leaving);
            }
            const owned_directory = try self.gpa.dupe(u8, directory);
            errdefer self.gpa.free(owned_directory);
            const owned_name = try self.gpa.dupe(u8, name);
            errdefer self.gpa.free(owned_name);
            self.close(io);
            const model = try self.open_fn(self.gpa, io, directory, self.options);
            errdefer self.close_fn(model, io);
            if (self.budget) |b| try b.add(io, kind, directory, name, bytes, .{ .context = self, .close = closeKey });
            self.mutex.lockUncancelable(io);
            self.current = .{ .directory = owned_directory, .name = owned_name, .model = model, .bytes = bytes };
            self.mutex.unlock(io);
            return .{ .model = model, .load_ns = @intCast(@max(0, started.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) };
        }

        /// GPU worker only: one more request uses the open model across
        /// items (true), or one fewer (false); the budget never closes it
        /// while any does. Each `true` is paired with a `false`.
        pub fn hold(self: *Self, io: std.Io, held: bool) void {
            const c = self.current orelse return;
            if (held) self.held += 1 else self.held -= 1;
            if (self.held == @intFromBool(held)) if (self.budget) |b| b.pin(io, c.directory, held);
        }

        /// GPU worker only: closes the open model, if any, and leaves the budget.
        pub fn close(self: *Self, io: std.Io) void {
            self.mutex.lockUncancelable(io);
            const current = self.current;
            self.current = null;
            self.mutex.unlock(io);
            const c = current orelse return;
            std.debug.assert(self.held == 0);
            if (self.budget) |b| b.remove(io, c.directory);
            self.close_fn(c.model, io);
            self.gpa.free(c.directory);
            self.gpa.free(c.name);
        }

        /// The budget's closer.
        fn closeKey(context: *anyopaque, io: std.Io, key: []const u8) void {
            const self: *Self = @ptrCast(@alignCast(context));
            if (self.current) |c| if (std.mem.eql(u8, c.directory, key)) self.close(io);
        }

        /// Any task: whether `directory` is open.
        pub fn isOpen(self: *Self, io: std.Io, directory: []const u8) bool {
            self.mutex.lockUncancelable(io);
            defer self.mutex.unlock(io);
            return if (self.current) |c| std.mem.eql(u8, c.directory, directory) else false;
        }

        /// Any task: the name the open model was opened by, in `arena`.
        pub fn openName(self: *Self, io: std.Io, arena: std.mem.Allocator) !?[]const u8 {
            self.mutex.lockUncancelable(io);
            defer self.mutex.unlock(io);
            return if (self.current) |c| try arena.dupe(u8, c.name) else null;
        }
    };
}

// ----- tests: a model that is a counter -----

const Counter = struct {
    var opened: usize = 0;
    var closed: usize = 0;
    value: u32,

    fn open(gpa: std.mem.Allocator, io: std.Io, directory: []const u8, options: Slot(Counter, .speech).Options) anyerror!*Counter {
        _ = io;
        _ = options;
        if (std.mem.eql(u8, directory, "/broken")) return error.InvalidCheckpoint;
        const c = try gpa.create(Counter);
        c.* = .{ .value = @intCast(directory.len) };
        opened += 1;
        return c;
    }

    fn close(c: *Counter, io: std.Io) void {
        _ = io;
        closed += 1;
        std.testing.allocator.destroy(c);
    }
};

test "one model at a time, counted in the budget, held models never closed" {
    const io = std.testing.io;
    var budget: memory.Budget = .init(std.testing.allocator, 100);
    defer budget.deinit();
    var slot: Slot(Counter, .speech) = .init(std.testing.allocator, Counter.open, Counter.close, .{ .metal = false });
    slot.budget = &budget;
    defer slot.deinit(io);
    Counter.opened = 0;
    Counter.closed = 0;
    const a = try slot.acquire(io, "/m/a", "a", 60);
    try std.testing.expect(slot.isOpen(io, "/m/a"));
    try std.testing.expectEqual(a.model, (try slot.acquire(io, "/m/a", "a", 60)).model);
    try std.testing.expectEqual(@as(usize, 1), Counter.opened);
    // Another directory replaces it, within the budget.
    _ = try slot.acquire(io, "/m/bb", "bb", 70);
    try std.testing.expectEqual(@as(usize, 1), Counter.closed);
    try std.testing.expectEqual(@as(u64, 70), budget.resident(io));
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    try std.testing.expectEqualStrings("bb", (try slot.openName(io, arena_state.allocator())).?);
    // Held, another model's room cannot come from it, nor may another replace it.
    slot.hold(io, true);
    slot.hold(io, true);
    try std.testing.expectError(error.Pinned, budget.reserve(io, 40, null));
    try std.testing.expectError(error.Pinned, slot.acquire(io, "/m/a", "a", 60));
    slot.hold(io, false);
    try std.testing.expectError(error.Pinned, budget.reserve(io, 40, null));
    slot.hold(io, false);
    try budget.reserve(io, 40, null);
    try std.testing.expect(!slot.isOpen(io, "/m/bb"));
    try std.testing.expectEqual(@as(u64, 0), budget.resident(io));
    // A model that fails to open leaves nothing behind; too large is refused first.
    try std.testing.expect(std.meta.isError(slot.acquire(io, "/broken", "x", 10)));
    try std.testing.expectError(error.ModelTooLarge, slot.acquire(io, "/m/c", "c", 101));
    try std.testing.expectEqual(@as(u64, 0), budget.resident(io));
    try std.testing.expectEqual(@as(?[]const u8, null), try slot.openName(io, arena_state.allocator()));
}
