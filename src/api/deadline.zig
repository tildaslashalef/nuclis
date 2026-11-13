//! Connection deadlines. A `Guard` wraps one connection's socket reader and
//! writer: every read and write the connection blocks in stamps a deadline,
//! and a `Watch` task shuts down the socket of any connection past its own,
//! which wakes the blocked call with an end of stream or a failed write. The
//! durations are `http.Limits` (docs/guide/api.md § Conventions).
//!
//! A `Guard` is pinned: its reader's and writer's vtables find it by
//! `@fieldParentPtr`, so it is initialised in place and never moved.
const std = @import("std");

const never = std.math.maxInt(i64);

/// What a read may wait for: the whole head, one more piece of a body, or
/// nothing (an upgraded session reads at its own pace).
pub const Phase = enum { head, body, session };

pub const Guard = struct {
    io: std.Io,
    stream: std.Io.net.Stream,
    reader: std.Io.net.Stream.Reader,
    writer: std.Io.net.Stream.Writer,
    net_reader: *const std.Io.Reader.VTable,
    net_writer: *const std.Io.Writer.VTable,
    /// Awake-clock nanoseconds after which the watch shuts the socket down;
    /// `never` while the connection waits on nothing outside.
    at: std.atomic.Value(i64) = .init(never),
    /// Set by the watch when it shut the socket down.
    expired: std.atomic.Value(bool) = .init(false),
    phase: Phase = .head,
    head_until: i64 = never,
    head_ns: i64,
    body_idle_ns: i64,
    write_ns: i64,

    pub const Durations = struct { head_ns: i64, body_idle_ns: i64, write_ns: i64 };

    pub fn init(self: *Guard, io: std.Io, stream: std.Io.net.Stream, read_buffer: []u8, write_buffer: []u8, durations: Durations) void {
        self.* = .{
            .io = io,
            .stream = stream,
            .reader = stream.reader(io, read_buffer),
            .writer = stream.writer(io, write_buffer),
            .net_reader = undefined,
            .net_writer = undefined,
            .head_ns = durations.head_ns,
            .body_idle_ns = durations.body_idle_ns,
            .write_ns = durations.write_ns,
        };
        self.net_reader = self.reader.interface.vtable;
        self.net_writer = self.writer.interface.vtable;
        self.reader.interface.vtable = &reader_vtable;
        self.writer.interface.vtable = &writer_vtable;
    }

    /// Starts `phase`; a head's deadline runs from now for the whole head.
    pub fn enter(self: *Guard, phase: Phase) void {
        self.phase = phase;
        if (phase == .head) self.head_until = now(self.io) +| self.head_ns;
    }

    /// Shuts the socket down when its deadline has passed; called by the watch.
    pub fn check(self: *Guard, at_now: i64) void {
        if (self.at.load(.acquire) >= at_now) return;
        self.at.store(never, .release);
        self.expired.store(true, .release);
        self.stream.shutdown(self.io, .both) catch {};
    }

    fn arm(self: *Guard, until: i64) void {
        self.at.store(until, .release);
    }

    fn disarm(self: *Guard) void {
        self.at.store(never, .release);
    }

    fn readDeadline(self: *Guard) i64 {
        return switch (self.phase) {
            .head => self.head_until,
            .body => now(self.io) +| self.body_idle_ns,
            .session => never,
        };
    }

    fn ofReader(r: *std.Io.Reader) *Guard {
        const net: *std.Io.net.Stream.Reader = @alignCast(@fieldParentPtr("interface", r));
        return @fieldParentPtr("reader", net);
    }

    fn ofWriter(w: *std.Io.Writer) *Guard {
        const net: *std.Io.net.Stream.Writer = @alignCast(@fieldParentPtr("interface", w));
        return @fieldParentPtr("writer", net);
    }

    const reader_vtable: std.Io.Reader.VTable = .{ .stream = readStream, .readVec = readVec };
    const writer_vtable: std.Io.Writer.VTable = .{ .drain = drain, .sendFile = sendFile };

    fn readStream(r: *std.Io.Reader, w: *std.Io.Writer, limit: std.Io.Limit) std.Io.Reader.StreamError!usize {
        const g = ofReader(r);
        g.arm(g.readDeadline());
        defer g.disarm();
        return g.net_reader.stream(r, w, limit);
    }

    fn readVec(r: *std.Io.Reader, data: [][]u8) std.Io.Reader.Error!usize {
        const g = ofReader(r);
        g.arm(g.readDeadline());
        defer g.disarm();
        return g.net_reader.readVec(r, data);
    }

    fn drain(w: *std.Io.Writer, data: []const []const u8, splat: usize) std.Io.Writer.Error!usize {
        const g = ofWriter(w);
        g.arm(now(g.io) +| g.write_ns);
        defer g.disarm();
        return g.net_writer.drain(w, data, splat);
    }

    fn sendFile(w: *std.Io.Writer, file_reader: *std.Io.File.Reader, limit: std.Io.Limit) std.Io.Writer.FileError!usize {
        const g = ofWriter(w);
        g.arm(now(g.io) +| g.write_ns);
        defer g.disarm();
        return g.net_writer.sendFile(w, file_reader, limit);
    }
};

/// The guards of the connections open now; one task checks them all.
pub fn Watch(comptime capacity: usize) type {
    return struct {
        const Self = @This();
        mutex: std.Io.Mutex = .init,
        slots: [capacity]?*Guard = @splat(null),

        /// Fails when every slot is taken; the caller then serves unguarded.
        pub fn add(self: *Self, io: std.Io, guard: *Guard) bool {
            self.mutex.lockUncancelable(io);
            defer self.mutex.unlock(io);
            for (&self.slots) |*s| if (s.* == null) {
                s.* = guard;
                return true;
            };
            return false;
        }

        pub fn remove(self: *Self, io: std.Io, guard: *Guard) void {
            self.mutex.lockUncancelable(io);
            defer self.mutex.unlock(io);
            for (&self.slots) |*s| if (s.* == guard) {
                s.* = null;
            };
        }

        pub fn checkAll(self: *Self, io: std.Io) void {
            const at_now = now(io);
            self.mutex.lockUncancelable(io);
            defer self.mutex.unlock(io);
            for (self.slots) |s| if (s) |g| g.check(at_now);
        }

        /// The watch's task: checks every `period` until canceled.
        pub fn run(self: *Self, io: std.Io, period: std.Io.Duration) void {
            while (true) {
                io.sleep(period, .awake) catch return;
                self.checkAll(io);
            }
        }
    };
}

fn now(io: std.Io) i64 {
    return @intCast(std.Io.Clock.awake.now(io).nanoseconds); // checked: the awake clock, below 292 years of uptime
}

// ----- tests -----

const testing = std.testing;

/// A socket pair as streams; `[0]` is the server's end.
pub fn testPair() ![2]std.Io.net.Stream {
    var fds: [2]std.c.fd_t = undefined;
    if (std.c.socketpair(std.c.AF.UNIX, std.c.SOCK.STREAM, 0, &fds) != 0) return error.SocketPair;
    return .{
        .{ .socket = .{ .handle = fds[0], .address = .{ .ip4 = .loopback(0) } } },
        .{ .socket = .{ .handle = fds[1], .address = .{ .ip4 = .loopback(0) } } },
    };
}

/// Checks `watch` until `done` is set, for at most `bound_ms`.
pub fn watchUntil(io: std.Io, watch: anytype, done: *const std.atomic.Value(bool), bound_ms: i64) bool {
    const until = std.Io.Clock.awake.now(io).addDuration(.fromMilliseconds(bound_ms));
    while (!done.load(.acquire)) {
        if (std.Io.Clock.awake.now(io).compare(.gt, until)) return false;
        watch.checkAll(io);
        io.sleep(.fromMilliseconds(5), .awake) catch return false;
    }
    return true;
}

const short: Guard.Durations = .{ .head_ns = 100 * std.time.ns_per_ms, .body_idle_ns = 100 * std.time.ns_per_ms, .write_ns = 100 * std.time.ns_per_ms };

fn writeForever(guard: *Guard, done: *std.atomic.Value(bool), failed: *bool) void {
    defer done.store(true, .release);
    const chunk: [64 * 1024]u8 = @splat('x');
    for (0..1024) |_| guard.writer.interface.writeAll(&chunk) catch {
        failed.* = true;
        return;
    };
}

test "a write to a client that never reads ends at the write deadline" {
    const io = testing.io;
    const pair = try testPair();
    defer pair[0].close(io);
    defer pair[1].close(io);
    var read_buffer: [64]u8 = undefined;
    var write_buffer: [1024]u8 = undefined;
    var guard: Guard = undefined;
    guard.init(io, pair[0], &read_buffer, &write_buffer, short);
    var watch: Watch(4) = .{};
    try testing.expect(watch.add(io, &guard));
    var done: std.atomic.Value(bool) = .init(false);
    var failed = false;
    var task = try io.concurrent(writeForever, .{ &guard, &done, &failed });
    const ended = watchUntil(io, &watch, &done, 5000);
    // A writer still blocked after the bound is freed by closing its peer.
    if (!ended) pair[1].shutdown(io, .both) catch {};
    task.await(io);
    try testing.expect(ended);
    try testing.expect(failed);
    try testing.expect(guard.expired.load(.acquire));
}

test "a guard is armed only while a call waits outside" {
    const io = testing.io;
    const pair = try testPair();
    defer pair[0].close(io);
    defer pair[1].close(io);
    var read_buffer: [64]u8 = undefined;
    var write_buffer: [64]u8 = undefined;
    var guard: Guard = undefined;
    guard.init(io, pair[0], &read_buffer, &write_buffer, short);
    try guard.writer.interface.writeAll("hi");
    try guard.writer.interface.flush();
    try testing.expectEqual(never, guard.at.load(.acquire));
    var got: [2]u8 = undefined;
    try testing.expectEqual(@as(isize, 2), std.c.read(pair[1].socket.handle, &got, 2));
    try testing.expectEqual(@as(isize, 2), std.c.write(pair[1].socket.handle, "ok", 2));
    guard.enter(.session);
    try testing.expectEqualStrings("ok", try guard.reader.interface.take(2));
    try testing.expectEqual(never, guard.at.load(.acquire));
    // Past its deadline nothing waits, so nothing is shut down.
    guard.check(std.math.maxInt(i64) - 1);
    try testing.expect(!guard.expired.load(.acquire));
}
