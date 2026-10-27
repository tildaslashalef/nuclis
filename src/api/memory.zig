//! One memory budget over every model `nuclis serve` keeps open, whatever
//! its kind. Before a model opens, `reserve` closes the least recently used
//! others (through their own `close`) until it fits. Two kinds of model are
//! never closed: one in use by a running request (`pin`, for its duration)
//! and one the server keeps for its whole life (`keep`, `--pin` by name);
//! a model that cannot fit beside the kept ones is too large. Every change
//! happens on the GPU worker, which owns the models; `snapshot` may be
//! called from any task. docs/guide/api.md § Memory.
const std = @import("std");
const builtin = @import("builtin");

pub const Kind = enum { decision, language, embedding, speech, transcription };

/// Closes the model registered under `key`; it calls `Budget.remove`.
pub const Closer = struct {
    context: *anyopaque,
    close: *const fn (context: *anyopaque, io: std.Io, key: []const u8) void,
};

pub const Error = error{
    /// The model alone is larger than the limit, or than what the kept
    /// models leave of it.
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
    /// Names of the models never closed to make room (`keep`); owned.
    kept: []const []const u8 = &.{},

    const Entry = struct {
        kind: Kind,
        /// Owned: the owner's key (a decision checkpoint's directory, a
        /// language model's name, an embedding model's file) and the name a person reads.
        key: []u8,
        name: []u8,
        bytes: u64,
        used: u64,
        /// In use by a running request.
        pinned: bool = false,
        /// Named by `keep`: never closed by the budget.
        kept: bool = false,
        closer: Closer,
    };

    pub fn init(gpa: std.mem.Allocator, limit: u64) Budget {
        return .{ .gpa = gpa, .limit = limit };
    }

    /// After every model closed.
    pub fn deinit(self: *Budget) void {
        for (self.entries.items) |e| self.free(e);
        self.entries.deinit(self.gpa);
        for (self.kept) |n| self.gpa.free(n);
        self.gpa.free(self.kept);
    }

    /// The models named `names` are never closed to make room, once open
    /// under that name; replaces any earlier list. Before any model opens.
    pub fn keep(self: *Budget, names: []const []const u8) std.mem.Allocator.Error!void {
        const owned = try self.gpa.alloc([]const u8, names.len);
        var done: usize = 0;
        errdefer {
            for (owned[0..done]) |n| self.gpa.free(n);
            self.gpa.free(owned);
        }
        for (names, owned) |n, *o| {
            o.* = try self.gpa.dupe(u8, n);
            done += 1;
        }
        for (self.kept) |n| self.gpa.free(n);
        self.gpa.free(self.kept);
        self.kept = owned;
    }

    /// Makes room for `bytes` more: closes the least recently used models,
    /// neither in use nor kept, other than `keep_key`, until they fit.
    pub fn reserve(self: *Budget, io: std.Io, bytes: u64, keep_key: ?[]const u8) Error!void {
        if (bytes > self.limit) return error.ModelTooLarge;
        while (true) {
            self.mutex.lockUncancelable(io);
            if (self.residentLocked() + bytes <= self.limit) {
                self.mutex.unlock(io);
                return;
            }
            if (self.keptLocked() + bytes > self.limit) {
                self.mutex.unlock(io);
                return error.ModelTooLarge;
            }
            const victim = self.victimLocked(keep_key) orelse {
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

    /// Whether `reserve(bytes)` can succeed: everything neither in use nor
    /// kept may be closed. No side effects, so a caller can ask before
    /// closing its own.
    pub fn canFit(self: *Budget, io: std.Io, bytes: u64) Error!void {
        if (bytes > self.limit) return error.ModelTooLarge;
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        const kept = self.keptLocked();
        if (kept + bytes > self.limit) return error.ModelTooLarge;
        var pinned: u64 = 0;
        for (self.entries.items) |e| if (e.pinned and !e.kept) {
            pinned += e.bytes;
        };
        if (kept + pinned + bytes > self.limit) return error.Pinned;
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
        const kept = for (self.kept) |n| {
            if (std.mem.eql(u8, n, name)) break true;
        } else false;
        try self.entries.append(self.gpa, .{ .kind = kind, .key = owned_key, .name = owned_name, .bytes = bytes, .used = self.tick, .kept = kept, .closer = closer });
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

    pub const Model = struct { kind: Kind, name: []const u8, bytes: u64, pinned: bool = false };

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
            m.* = .{ .kind = e.kind, .name = try arena.dupe(u8, e.name), .bytes = e.bytes, .pinned = e.kept };
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

    fn keptLocked(self: *const Budget) u64 {
        var total: u64 = 0;
        for (self.entries.items) |e| if (e.kept) {
            total += e.bytes;
        };
        return total;
    }

    fn findLocked(self: *Budget, key: []const u8) ?*Entry {
        for (self.entries.items) |*e| if (std.mem.eql(u8, e.key, key)) return e;
        return null;
    }

    fn victimLocked(self: *Budget, keep_key: ?[]const u8) ?*Entry {
        var best: ?*Entry = null;
        for (self.entries.items) |*e| {
            if (e.pinned or e.kept) continue;
            if (keep_key) |k| if (std.mem.eql(u8, e.key, k)) continue;
            if (best == null or e.used < best.?.used) best = e;
        }
        return best;
    }

    fn free(self: *Budget, e: Entry) void {
        self.gpa.free(e.key);
        self.gpa.free(e.name);
    }
};

/// The default limit: physical memory less a reserve for the system and
/// other applications, a third of it up to 16 GiB (32 GiB on a 48 GiB
/// machine, 2.7 GiB on a 4 GiB one). 16 GiB when the size is unknown.
pub fn defaultLimit(io: std.Io) u64 {
    return limitFor(physicalMemory(io) orelse return 16 << 30);
}

fn limitFor(physical: u64) u64 {
    return physical - @min(physical / 3, 16 << 30);
}

fn physicalMemory(io: std.Io) ?u64 {
    switch (builtin.os.tag) {
        .macos => {
            var size: u64 = 0;
            var len: usize = @sizeOf(u64);
            if (std.c.sysctlbyname("hw.memsize", &size, &len, null, 0) != 0) return null;
            return size;
        },
        .linux => {
            var info: std.os.linux.Sysinfo = undefined;
            if (std.os.linux.errno(std.os.linux.sysinfo(&info)) != .SUCCESS) return null;
            const total = @as(u64, info.totalram) * @max(info.mem_unit, 1);
            return @min(total, cgroupLimit(io) orelse total);
        },
        else => return null,
    }
}

/// A cgroup v2 memory limit (a container's), when one is set.
fn cgroupLimit(io: std.Io) ?u64 {
    var buffer: [32]u8 = undefined;
    const text = std.Io.Dir.cwd().readFile(io, "/sys/fs/cgroup/memory.max", &buffer) catch return null;
    return std.fmt.parseUnsigned(u64, std.mem.trim(u8, text, " \n"), 10) catch null;
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
}

test "kept models are never closed; what cannot fit beside them is too large" {
    const io = std.testing.io;
    var budget: Budget = .init(std.testing.allocator, 100);
    defer budget.deinit();
    var fake: Fake = .{ .budget = &budget };
    defer fake.deinit();
    try budget.keep(&.{ "e4b", "voice" });
    try budget.add(io, .language, "e4b", "e4b", 40, fake.closer());
    try budget.add(io, .speech, "/m/voice", "voice", 30, fake.closer());
    try budget.add(io, .decision, "laya", "laya", 20, fake.closer());
    try budget.add(io, .transcription, "/m/ears", "ears", 10, fake.closer());
    budget.touch(io, "e4b");
    budget.touch(io, "/m/voice");
    // 100 resident; 15 more closes the least recently used unkept model
    // (laya) though both kept ones are older by use than nothing else.
    try budget.reserve(io, 15, null);
    try std.testing.expectEqualStrings("laya", fake.closed.items[0]);
    // 31 more than the kept 70 leave: too large, at once, nothing closed.
    try std.testing.expectError(error.ModelTooLarge, budget.canFit(io, 31));
    try std.testing.expectError(error.ModelTooLarge, budget.reserve(io, 31, null));
    try std.testing.expectEqual(@as(usize, 1), fake.closed.items.len);
    // 30 fits once the unkept transcription model closes.
    try budget.reserve(io, 30, null);
    try std.testing.expectEqualStrings("/m/ears", fake.closed.items[1]);
    try std.testing.expectEqual(@as(u64, 70), budget.resident(io));
    // A kept model in use is still only kept: in use adds nothing to wait for.
    budget.pin(io, "e4b", true);
    try budget.canFit(io, 30);
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const models = try budget.snapshot(io, arena_state.allocator());
    for (models) |m| try std.testing.expect(m.pinned);
    // A kept model that grows past what the limit holds is too large.
    try std.testing.expectError(error.ModelTooLarge, budget.resize(io, "/m/voice", 61));
}

test "the default limit leaves a third, at most 16 GiB" {
    try std.testing.expectEqual(@as(u64, 32 << 30), limitFor(48 << 30));
    try std.testing.expectEqual(@as(u64, 112 << 30), limitFor(128 << 30));
    try std.testing.expectEqual(@as(u64, (4 << 30) - (4 << 30) / 3), limitFor(4 << 30));
    try std.testing.expect(defaultLimit(std.testing.io) > 0);
}
