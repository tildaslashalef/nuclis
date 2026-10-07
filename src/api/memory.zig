//! One memory budget over every model `nuclis serve` keeps open, decision
//! and language alike. Before a model opens, `reserve` closes the least
//! recently used others (through their own `close`) until it fits; a model
//! in use by a running request is pinned and never closed. Every change
//! happens on the GPU worker, which owns the models; `snapshot` may be
//! called from any task. docs/guide/api.md § Running the server.
const std = @import("std");

pub const Kind = enum { decision, language };

/// Closes the model registered under `key`; it calls `Budget.remove`.
pub const Closer = struct {
    context: *anyopaque,
    close: *const fn (context: *anyopaque, io: std.Io, key: []const u8) void,
};

pub const Error = error{
    /// The model alone is larger than the limit.
    ModelTooLarge,
    /// Only pinned models stand in the way: it fits once they are released.
    Pinned,
};

pub const Budget = struct {
    gpa: std.mem.Allocator,
    limit: u64,
    /// Held for every change and for `snapshot`; never while a model closes.
    mutex: std.Io.Mutex = .init,
    entries: std.ArrayList(Entry) = .empty,
    tick: u64 = 0,

    const Entry = struct {
        kind: Kind,
        /// Owned: the owner's key (a decision checkpoint's directory, a
        /// language model's name) and the name a person reads.
        key: []u8,
        name: []u8,
        bytes: u64,
        used: u64,
        pinned: bool = false,
        closer: Closer,
    };

    pub fn init(gpa: std.mem.Allocator, limit: u64) Budget {
        return .{ .gpa = gpa, .limit = limit };
    }

    /// After every model closed.
    pub fn deinit(self: *Budget) void {
        for (self.entries.items) |e| self.free(e);
        self.entries.deinit(self.gpa);
    }

    /// Makes room for `bytes` more: closes the least recently used unpinned
    /// models other than `keep` until they fit.
    pub fn reserve(self: *Budget, io: std.Io, bytes: u64, keep: ?[]const u8) Error!void {
        if (bytes > self.limit) return error.ModelTooLarge;
        while (true) {
            self.mutex.lockUncancelable(io);
            if (self.residentLocked() + bytes <= self.limit) {
                self.mutex.unlock(io);
                return;
            }
            const victim = self.victimLocked(keep) orelse {
                self.mutex.unlock(io);
                return error.Pinned;
            };
            const closer = victim.closer;
            const key = victim.key;
            self.mutex.unlock(io);
            closer.close(closer.context, io, key);
            // A closer that did not remove its entry must not loop forever.
            self.remove(io, key);
        }
    }

    /// Whether `reserve(bytes)` can succeed: everything unpinned may be
    /// closed. No side effects, so a caller can ask before closing its own.
    pub fn canFit(self: *Budget, io: std.Io, bytes: u64) Error!void {
        if (bytes > self.limit) return error.ModelTooLarge;
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        var pinned: u64 = 0;
        for (self.entries.items) |e| if (e.pinned) {
            pinned += e.bytes;
        };
        if (pinned + bytes > self.limit) return error.Pinned;
    }

    /// Registers a model that has opened, `bytes` resident.
    pub fn add(self: *Budget, io: std.Io, kind: Kind, key: []const u8, name: []const u8, bytes: u64, closer: Closer) !void {
        const owned_key = try self.gpa.dupe(u8, key);
        errdefer self.gpa.free(owned_key);
        const owned_name = try self.gpa.dupe(u8, name);
        errdefer self.gpa.free(owned_name);
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.tick += 1;
        try self.entries.append(self.gpa, .{ .kind = kind, .key = owned_key, .name = owned_name, .bytes = bytes, .used = self.tick, .closer = closer });
    }

    /// `key`'s model now holds `bytes` (its real size once open, or a
    /// projector loaded later); closes others as `reserve` does when that
    /// no longer fits.
    pub fn resize(self: *Budget, io: std.Io, key: []const u8, bytes: u64) Error!void {
        if (bytes > self.limit) return error.ModelTooLarge;
        {
            self.mutex.lockUncancelable(io);
            defer self.mutex.unlock(io);
            const e = self.findLocked(key) orelse return;
            e.bytes = bytes;
        }
        return self.reserve(io, 0, key);
    }

    /// Unregisters `key`'s model; nothing when it is not registered.
    pub fn remove(self: *Budget, io: std.Io, key: []const u8) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        for (self.entries.items, 0..) |e, i| if (std.mem.eql(u8, e.key, key)) {
            self.free(self.entries.swapRemove(i));
            return;
        };
    }

    /// Marks `key`'s model as just used.
    pub fn touch(self: *Budget, io: std.Io, key: []const u8) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.tick += 1;
        if (self.findLocked(key)) |e| e.used = self.tick;
    }

    /// A pinned model is in use and is never closed to make room.
    pub fn pin(self: *Budget, io: std.Io, key: []const u8, pinned: bool) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        if (self.findLocked(key)) |e| e.pinned = pinned;
    }

    pub const Model = struct { kind: Kind, name: []const u8, bytes: u64 };

    /// The resident models, most recently used first, in `arena`; any task.
    pub fn snapshot(self: *Budget, io: std.Io, arena: std.mem.Allocator) ![]Model {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        const out = try arena.alloc(Model, self.entries.items.len);
        const order = try arena.alloc(usize, out.len);
        for (order, 0..) |*o, i| o.* = i;
        std.mem.sort(usize, order, self, struct {
            fn newer(b: *Budget, x: usize, y: usize) bool {
                return b.entries.items[x].used > b.entries.items[y].used;
            }
        }.newer);
        for (order, out) |i, *m| {
            const e = self.entries.items[i];
            m.* = .{ .kind = e.kind, .name = try arena.dupe(u8, e.name), .bytes = e.bytes };
        }
        return out;
    }

    /// Bytes `key`'s model holds; 0 when it is not registered.
    pub fn bytesOf(self: *Budget, io: std.Io, key: []const u8) u64 {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        return if (self.findLocked(key)) |e| e.bytes else 0;
    }

    /// Bytes held by every registered model; any task.
    pub fn resident(self: *Budget, io: std.Io) u64 {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        return self.residentLocked();
    }

    fn residentLocked(self: *const Budget) u64 {
        var total: u64 = 0;
        for (self.entries.items) |e| total += e.bytes;
        return total;
    }

    fn findLocked(self: *Budget, key: []const u8) ?*Entry {
        for (self.entries.items) |*e| if (std.mem.eql(u8, e.key, key)) return e;
        return null;
    }

    fn victimLocked(self: *Budget, keep: ?[]const u8) ?*Entry {
        var best: ?*Entry = null;
        for (self.entries.items) |*e| {
            if (e.pinned) continue;
            if (keep) |k| if (std.mem.eql(u8, e.key, k)) continue;
            if (best == null or e.used < best.?.used) best = e;
        }
        return best;
    }

    fn free(self: *Budget, e: Entry) void {
        self.gpa.free(e.key);
        self.gpa.free(e.name);
    }
};

/// The default limit: physical memory less 16 GiB for the system and other
/// applications, at least 4 GiB.
pub fn defaultLimit() u64 {
    const physical = physicalMemory() orelse return 16 << 30;
    const reserve_for_system: u64 = 16 << 30;
    return @max(physical -| reserve_for_system, 4 << 30);
}

fn physicalMemory() ?u64 {
    var size: u64 = 0;
    var len: usize = @sizeOf(u64);
    if (std.c.sysctlbyname("hw.memsize", &size, &len, null, 0) != 0) return null;
    return size;
}

// ----- tests: models that record their closing -----

const Fake = struct {
    budget: *Budget,
    closed: std.ArrayList([]const u8) = .empty,

    fn closer(self: *Fake) Closer {
        return .{ .context = self, .close = close };
    }

    fn close(context: *anyopaque, io: std.Io, key: []const u8) void {
        const self: *Fake = @ptrCast(@alignCast(context));
        self.closed.append(std.testing.allocator, std.testing.allocator.dupe(u8, key) catch unreachable) catch unreachable;
        self.budget.remove(io, key);
    }

    fn deinit(self: *Fake) void {
        for (self.closed.items) |k| std.testing.allocator.free(k);
        self.closed.deinit(std.testing.allocator);
    }
};

test "room is made by closing the least recently used, across kinds" {
    const io = std.testing.io;
    var budget: Budget = .init(std.testing.allocator, 100);
    defer budget.deinit();
    var fake: Fake = .{ .budget = &budget };
    defer fake.deinit();
    try budget.add(io, .decision, "laya", "laya", 10, fake.closer());
    try budget.add(io, .language, "qwen", "qwen", 60, fake.closer());
    try budget.add(io, .decision, "clef", "clef", 20, fake.closer());
    budget.touch(io, "laya");
    // 90 resident; 30 more must close qwen (oldest use), not laya or clef.
    try budget.reserve(io, 30, null);
    try std.testing.expectEqual(@as(usize, 1), fake.closed.items.len);
    try std.testing.expectEqualStrings("qwen", fake.closed.items[0]);
    try std.testing.expectEqual(@as(u64, 30), budget.resident(io));
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const models = try budget.snapshot(io, arena_state.allocator());
    try std.testing.expectEqualStrings("laya", models[0].name);
}

test "a pinned model is never closed; too large is refused at once" {
    const io = std.testing.io;
    var budget: Budget = .init(std.testing.allocator, 100);
    defer budget.deinit();
    var fake: Fake = .{ .budget = &budget };
    defer fake.deinit();
    try budget.add(io, .language, "qwen", "qwen", 70, fake.closer());
    try budget.add(io, .decision, "laya", "laya", 10, fake.closer());
    budget.pin(io, "qwen", true);
    try std.testing.expectError(error.ModelTooLarge, budget.reserve(io, 101, null));
    // 25 more: laya goes, qwen stays, and 70 + 25 fits.
    try budget.reserve(io, 25, null);
    try std.testing.expectEqualStrings("laya", fake.closed.items[0]);
    try std.testing.expectError(error.Pinned, budget.canFit(io, 40));
    try std.testing.expectError(error.Pinned, budget.reserve(io, 40, null));
    budget.pin(io, "qwen", false);
    try budget.reserve(io, 40, null);
    try std.testing.expectEqualStrings("qwen", fake.closed.items[1]);
    try std.testing.expectEqual(@as(u64, 0), budget.resident(io));
}

test "a model that grows once open makes room around itself" {
    const io = std.testing.io;
    var budget: Budget = .init(std.testing.allocator, 100);
    defer budget.deinit();
    var fake: Fake = .{ .budget = &budget };
    defer fake.deinit();
    try budget.add(io, .decision, "laya", "laya", 30, fake.closer());
    try budget.add(io, .language, "qwen", "qwen", 50, fake.closer());
    try budget.resize(io, "qwen", 80);
    try std.testing.expectEqualStrings("laya", fake.closed.items[0]);
    try std.testing.expectEqual(@as(u64, 80), budget.resident(io));
    try std.testing.expectError(error.ModelTooLarge, budget.resize(io, "qwen", 120));
    try std.testing.expect(defaultLimit() >= 4 << 30);
}
