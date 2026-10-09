//! The open embedding model, at most one, keyed by its GGUF file; opening
//! another closes it. It counts its file against the server's memory budget
//! (`memory.zig`): the weights are mapped and wrapped in place, and the
//! projector is not loaded. Only the GPU worker opens, uses, and closes the
//! embedder (`acquire`); other threads only read which model is open.
const std = @import("std");
const inference = @import("inference");
const memory = @import("../memory.zig");

const embed = inference.embed;

pub const Pool = struct {
    gpa: std.mem.Allocator,
    backend: embed.Backend,
    /// The server's; null in tests that need no limit.
    budget: ?*memory.Budget = null,
    mutex: std.Io.Mutex = .init,
    slot: ?Slot = null,

    const Slot = struct {
        /// Owned: the file, and the name the opening request used.
        path: []u8,
        name: []u8,
        embedder: *embed.Embedder,
    };

    pub const Acquired = struct {
        embedder: *embed.Embedder,
        /// Time spent opening it for this call; zero when it was open.
        load_ns: u64,
    };

    pub fn init(gpa: std.mem.Allocator, backend: embed.Backend) Pool {
        return .{ .gpa = gpa, .backend = backend };
    }

    /// GPU worker only, after the worker stopped.
    pub fn deinit(self: *Pool, io: std.Io) void {
        self.close(io);
    }

    /// GPU worker only: the embedder of `path`, opened (closing the open one)
    /// if it is not open.
    pub fn acquire(self: *Pool, io: std.Io, path: []const u8, name: []const u8) !Acquired {
        if (self.slot) |s| if (std.mem.eql(u8, s.path, path)) {
            if (self.budget) |b| b.touch(io, path);
            return .{ .embedder = s.embedder, .load_ns = 0 };
        };
        const started = std.Io.Clock.awake.now(io);
        const bytes = (try std.Io.Dir.cwd().statFile(io, path, .{})).size;
        if (self.budget) |b| {
            // Asked before anything closes, as the decision pool does.
            try b.canFit(io, bytes);
            const leaving: ?[]const u8 = if (self.slot) |s| s.path else null;
            try b.reserve(io, bytes -| if (leaving) |key| b.bytesOf(io, key) else 0, leaving);
        }
        const embedder = try embed.Embedder.open(self.gpa, io, path, self.backend);
        errdefer embedder.deinit();
        const owned_path = try self.gpa.dupe(u8, path);
        errdefer self.gpa.free(owned_path);
        const owned_name = try self.gpa.dupe(u8, name);
        errdefer self.gpa.free(owned_name);
        self.close(io);
        if (self.budget) |b| try b.add(io, .embedding, path, name, bytes, .{ .context = self, .close = closeKey });
        self.mutex.lockUncancelable(io);
        self.slot = .{ .path = owned_path, .name = owned_name, .embedder = embedder };
        self.mutex.unlock(io);
        const load_ns: u64 = @intCast(@max(0, started.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds()));
        return .{ .embedder = embedder, .load_ns = load_ns };
    }

    /// GPU worker only: closes the open model, if any, and leaves the budget.
    fn close(self: *Pool, io: std.Io) void {
        self.mutex.lockUncancelable(io);
        const slot = self.slot;
        self.slot = null;
        self.mutex.unlock(io);
        const s = slot orelse return;
        if (self.budget) |b| b.remove(io, s.path);
        s.embedder.deinit();
        self.gpa.free(s.path);
        self.gpa.free(s.name);
    }

    /// The budget's closer.
    fn closeKey(context: *anyopaque, io: std.Io, key: []const u8) void {
        const self: *Pool = @ptrCast(@alignCast(context));
        if (self.slot) |s| if (std.mem.eql(u8, s.path, key)) self.close(io);
    }

    /// Any thread: whether `path` is open.
    pub fn isOpen(self: *Pool, io: std.Io, path: []const u8) bool {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        return if (self.slot) |s| std.mem.eql(u8, s.path, path) else false;
    }

    /// Any thread: the name the open model was opened by, in `arena`.
    pub fn openName(self: *Pool, io: std.Io, arena: std.mem.Allocator) !?[]const u8 {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        return if (self.slot) |s| try arena.dupe(u8, s.name) else null;
    }
};

test "a file that is no embedding model leaves the pool empty and the budget untouched" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "x.gguf", .data = "not a gguf" });
    const path = try tmp.dir.realPathFileAlloc(io, "x.gguf", gpa);
    defer gpa.free(path);
    var budget: memory.Budget = .init(gpa, 1 << 20);
    defer budget.deinit();
    var pool: Pool = .init(gpa, .cpu);
    pool.budget = &budget;
    defer pool.deinit(io);
    try std.testing.expect(std.meta.isError(pool.acquire(io, path, "x")));
    try std.testing.expect(!pool.isOpen(io, path));
    try std.testing.expectEqual(@as(u64, 0), budget.resident(io));
    try std.testing.expectError(error.ModelTooLarge, blk: {
        var small: memory.Budget = .init(gpa, 4);
        defer small.deinit();
        pool.budget = &small;
        defer pool.budget = &budget;
        break :blk pool.acquire(io, path, "x");
    });
}
