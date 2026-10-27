//! Media encoded ahead of the turn that sends it: an attachment dropped into
//! the editor is read and run through its encoder on a background task while
//! the user types, and the turn takes the result by the file's digest
//! instead of encoding at Enter. The editor's chip shows the entry's state.
//!
//! One task at a time, on the model's own backend: the model's buffers are
//! never used by two threads at once because the driver calls `settle`
//! before anything else touches the engine (a turn, a command, a model
//! switch, quitting), and starts the next task (`pump`) only from its idle
//! loop. The backend's GPU-wait tick fires on the task's thread, so the
//! driver's tick must ignore calls off its own thread (it reads keys and
//! draws). A removed attachment's task is not interrupted; it finishes (a few
//! seconds at most) and its result is dropped. A failure is the entry's
//! state, which the chip shows; the turn then encodes again and reports it.
const std = @import("std");
const inference = @import("inference");
const completer = @import("../completer.zig");

const Allocator = std.mem.Allocator;
const Io = std.Io;
const Kind = completer.Media.Kind;
const Prepared = inference.engine.PreparedImage;

pub const State = enum { queued, encoding, ready, failed };

/// One attachment. The task writes `digest`, `bytes`, `prepared`, and
/// `failure`, then sets `done`; the driver reads them only after seeing it.
pub const Entry = struct {
    kind: Kind,
    /// The attachment's path as the editor holds it: the chip's key. Owned.
    key: []u8,
    /// The file the task reads (`~` expanded). Owned.
    path: []u8,
    state: State = .queued,
    digest: [32]u8 = undefined,
    bytes: u64 = 0,
    /// Owned while `state` is `ready`.
    prepared: Prepared = undefined,
    /// Why a failed entry failed.
    failure: ?anyerror = null,
    /// No attachment refers to it any more: freed when its task ends.
    orphan: bool = false,
    /// Marked by the driver's sweep (`keep`).
    kept: bool = true,
    done: std.atomic.Value(bool) = .init(false),

    /// The token count a ready entry's span takes.
    pub fn tokens(self: *const Entry) usize {
        return @as(usize, self.prepared.width_tokens) * self.prepared.height_tokens;
    }
};

pub const Ahead = struct {
    alloc: Allocator,
    io: Io,
    /// Heap entries, so a running task's pointer stays valid as the list grows.
    entries: std.ArrayList(*Entry) = .empty,
    running: ?Running = null,
    /// Set when the Io offers no concurrency: nothing is encoded ahead, and
    /// every attachment is encoded by its turn as before.
    unavailable: bool = false,

    const Running = struct { entry: *Entry, future: Io.Future(void) };

    pub fn init(alloc: Allocator, io: Io) Ahead {
        return .{ .alloc = alloc, .io = io };
    }

    /// Waits for the running task, then frees everything.
    pub fn deinit(self: *Ahead) void {
        self.reset();
        self.entries.deinit(self.alloc);
    }

    /// Waits for the running task and forgets every entry (a model switch:
    /// another encoder's rows mean nothing to the new model).
    pub fn reset(self: *Ahead) void {
        self.settle();
        for (self.entries.items) |entry| self.free(entry);
        self.entries.clearRetainingCapacity();
    }

    /// The entry for an attachment's path, if one exists.
    pub fn find(self: *const Ahead, key: []const u8) ?*Entry {
        for (self.entries.items) |entry| if (!entry.orphan and std.mem.eql(u8, entry.key, key)) return entry;
        return null;
    }

    /// The entry for `key`, queued when new; `path` is the file to read.
    pub fn want(self: *Ahead, kind: Kind, key: []const u8, path: []const u8) !*Entry {
        if (self.find(key)) |entry| {
            entry.kept = true;
            return entry;
        }
        const entry = try self.alloc.create(Entry);
        errdefer self.alloc.destroy(entry);
        const owned_key = try self.alloc.dupe(u8, key);
        errdefer self.alloc.free(owned_key);
        const owned_path = try self.alloc.dupe(u8, path);
        errdefer self.alloc.free(owned_path);
        entry.* = .{ .kind = kind, .key = owned_key, .path = owned_path };
        try self.entries.append(self.alloc, entry);
        return entry;
    }

    /// Starts a sweep: entries `want` does not touch before `endSweep` are
    /// dropped (an attachment removed from the editor).
    pub fn beginSweep(self: *Ahead) void {
        for (self.entries.items) |entry| entry.kept = false;
    }
    pub fn endSweep(self: *Ahead) void {
        var i: usize = 0;
        while (i < self.entries.items.len) {
            const entry = self.entries.items[i];
            if (entry.kept or entry.orphan) {
                i += 1;
                continue;
            }
            if (self.running != null and self.running.?.entry == entry) {
                entry.orphan = true;
                i += 1;
                continue;
            }
            self.free(entry);
            _ = self.entries.swapRemove(i);
        }
    }

    /// Driver idle loop: collects a finished task, then starts the next
    /// queued entry when none runs. `model` must be the engine's current one.
    pub fn pump(self: *Ahead, model: completer.Model) void {
        if (self.running) |r| {
            if (!r.entry.done.load(.acquire)) return;
            self.reap();
        }
        if (self.unavailable) return;
        for (self.entries.items) |entry| {
            if (entry.state != .queued or entry.orphan) continue;
            entry.state = .encoding;
            entry.done.store(false, .release);
            const future = self.io.concurrent(work, .{ entry, model, self.alloc, self.io }) catch {
                entry.state = .queued;
                self.unavailable = true;
                return;
            };
            self.running = .{ .entry = entry, .future = future };
            return;
        }
    }

    /// Whether a task is in flight (the driver keeps redrawing while one is).
    pub fn busy(self: *const Ahead) bool {
        return self.running != null;
    }

    /// Waits for the running task, if any, and collects it. Call before
    /// anything else uses the engine.
    pub fn settle(self: *Ahead) void {
        if (self.running == null) return;
        self.reap();
    }

    /// Takes a ready entry's rows whose digest is `digest` (the caller owns
    /// the features); null when none is ready, so the caller encodes.
    pub fn take(self: *Ahead, kind: Kind, digest: [32]u8) ?Prepared {
        for (self.entries.items, 0..) |entry, i| {
            if (entry.state != .ready or entry.kind != kind or !std.mem.eql(u8, &entry.digest, &digest)) continue;
            if (self.running != null and self.running.?.entry == entry) continue;
            const prepared = entry.prepared;
            entry.state = .queued; // features moved out
            self.free(entry);
            _ = self.entries.swapRemove(i);
            return prepared;
        }
        return null;
    }

    fn reap(self: *Ahead) void {
        var r = self.running.?;
        self.running = null;
        r.future.await(self.io);
        const entry = r.entry;
        entry.state = if (entry.failure == null) .ready else .failed;
        if (entry.orphan) {
            for (self.entries.items, 0..) |e, i| if (e == entry) {
                _ = self.entries.swapRemove(i);
                break;
            };
            self.free(entry);
        }
    }

    fn free(self: *Ahead, entry: *Entry) void {
        if (entry.state == .ready) self.alloc.free(entry.prepared.features);
        self.alloc.free(entry.key);
        self.alloc.free(entry.path);
        self.alloc.destroy(entry);
    }

    /// The task: read, digest, encode. Touches only `entry`'s result fields.
    fn work(entry: *Entry, model: completer.Model, alloc: Allocator, io: Io) void {
        defer entry.done.store(true, .release);
        entry.failure = null;
        encode(entry, model, alloc, io) catch |err| {
            entry.failure = err;
        };
    }

    fn encode(entry: *Entry, model: completer.Model, alloc: Allocator, io: Io) !void {
        const limit: usize = switch (entry.kind) {
            .image => inference.vision.image.max_bytes,
            .audio => inference.audio.audio.max_bytes,
        };
        const bytes = try Io.Dir.cwd().readFileAlloc(io, entry.path, alloc, .limited(limit));
        defer alloc.free(bytes);
        entry.digest = completer.digestOf(bytes);
        entry.bytes = bytes.len;
        entry.prepared = switch (entry.kind) {
            .image => try model.encode_image(model.context, alloc, bytes),
            .audio => try model.encode_audio(model.context, alloc, bytes),
        };
    }
};

/// Room for any `note`.
pub const note_bytes = 48;

/// The chip's word for an entry's state: `queued`, `encoding…`, the token
/// count, or why it failed.
pub fn note(entry: *const Entry, buffer: *[note_bytes]u8) []const u8 {
    return switch (entry.state) {
        .queued => "queued",
        .encoding => "encoding…",
        .ready => std.fmt.bufPrint(buffer, "{d} tokens", .{entry.tokens()}) catch "ready",
        .failed => std.fmt.bufPrint(buffer, "failed: {s}", .{@errorName(entry.failure orelse error.Unknown)}) catch "failed",
    };
}

const testing = std.testing;

/// A model whose encoder returns a 2×3 grid of the file's first byte.
const Fake = struct {
    calls: std.atomic.Value(usize) = .init(0),
    fn model(self: *Fake) completer.Model {
        return .{ .context = self, .run = undefined, .count = undefined, .encode_image = encodeImage };
    }
    fn encodeImage(context: *anyopaque, alloc: Allocator, bytes: []const u8) anyerror!Prepared {
        const self: *Fake = @ptrCast(@alignCast(context));
        _ = self.calls.fetchAdd(1, .monotonic);
        if (bytes.len == 0) return error.EmptyImage;
        const features = try alloc.alloc(f32, 6);
        @memset(features, @floatFromInt(bytes[0]));
        return .{ .width_tokens = 2, .height_tokens = 3, .features = features, .width = 4, .height = 4 };
    }
};

fn writeFile(dir: std.Io.Dir, name: []const u8, bytes: []const u8) !void {
    try dir.writeFile(testing.io, .{ .sub_path = name, .data = bytes });
}

test "an attachment is encoded in the background and taken by its digest" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeFile(tmp.dir, "a.png", "A-bytes");
    try writeFile(tmp.dir, "empty.png", "");
    const a_path = try tmp.dir.realPathFileAlloc(testing.io, "a.png", testing.allocator);
    defer testing.allocator.free(a_path);
    const empty_path = try tmp.dir.realPathFileAlloc(testing.io, "empty.png", testing.allocator);
    defer testing.allocator.free(empty_path);

    var fake: Fake = .{};
    var ahead: Ahead = .init(testing.allocator, testing.io);
    defer ahead.deinit();
    const a = try ahead.want(.image, "~/a.png", a_path);
    const e = try ahead.want(.image, "empty.png", empty_path);
    try testing.expectEqual(State.queued, a.state);
    try testing.expect(try ahead.want(.image, "~/a.png", a_path) == a);
    // One task at a time: the first runs, the second waits its turn.
    ahead.pump(fake.model());
    if (ahead.unavailable) return error.SkipZigTest;
    try testing.expectEqual(State.encoding, a.state);
    try testing.expectEqual(State.queued, e.state);
    ahead.settle();
    try testing.expectEqual(State.ready, a.state);
    try testing.expectEqual(@as(usize, 6), a.tokens());
    ahead.pump(fake.model());
    ahead.settle();
    try testing.expectEqual(State.failed, e.state);
    try testing.expectEqual(error.EmptyImage, e.failure.?);
    // The turn takes the rows by digest; a different file's digest misses.
    try testing.expect(ahead.take(.image, completer.digestOf("other")) == null);
    const prepared = ahead.take(.image, completer.digestOf("A-bytes")).?;
    defer testing.allocator.free(prepared.features);
    try testing.expectEqual(@as(f32, 'A'), prepared.features[0]);
    try testing.expect(ahead.find("~/a.png") == null);
    try testing.expectEqual(@as(usize, 2), fake.calls.load(.monotonic));
}

test "a removed attachment's task finishes and its result is dropped" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeFile(tmp.dir, "b.png", "B");
    const b_path = try tmp.dir.realPathFileAlloc(testing.io, "b.png", testing.allocator);
    defer testing.allocator.free(b_path);
    var fake: Fake = .{};
    var ahead: Ahead = .init(testing.allocator, testing.io);
    defer ahead.deinit();
    _ = try ahead.want(.image, "b.png", b_path);
    _ = try ahead.want(.image, "gone.png", "/nonexistent/gone.png");
    ahead.pump(fake.model());
    if (ahead.unavailable) return error.SkipZigTest;
    // Both chips are deleted: the running one becomes an orphan, the queued
    // one goes at once.
    ahead.beginSweep();
    ahead.endSweep();
    try testing.expect(ahead.find("b.png") == null);
    try testing.expect(ahead.find("gone.png") == null);
    ahead.settle();
    try testing.expectEqual(@as(usize, 0), ahead.entries.items.len);
    // A file that cannot be read fails as a state, not an error.
    const missing = try ahead.want(.image, "gone.png", "/nonexistent/gone.png");
    ahead.pump(fake.model());
    ahead.settle();
    try testing.expectEqual(State.failed, missing.state);
    try testing.expectEqual(@as(usize, 1), fake.calls.load(.monotonic));
}
