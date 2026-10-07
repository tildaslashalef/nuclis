//! A bounded byte queue from one producer (the GPU worker writing a streamed
//! response) to one consumer (the connection task sending it), so the worker
//! never writes to a socket. The producer waits while the queue is full and
//! learns when the consumer has gone; the consumer waits with a deadline, so
//! it can send keepalives while nothing is produced. The buffer is allocated
//! once, at the capacity given.
const std = @import("std");

pub const Pipe = struct {
    mutex: std.Io.Mutex = .init,
    /// Signalled on every append, take, close, and abandon; both sides wait on it.
    changed: std.Io.Condition = .init,
    /// Owned; `len` bytes of it are queued.
    buffer: []u8,
    len: usize = 0,
    /// The producer is done: the consumer drains what is queued, then stops.
    closed: bool = false,
    /// The consumer is gone: writes are refused and dropped.
    abandoned: bool = false,

    pub fn init(gpa: std.mem.Allocator, capacity: usize) !Pipe {
        std.debug.assert(capacity > 0);
        return .{ .buffer = try gpa.alloc(u8, capacity) };
    }

    pub fn deinit(self: *Pipe, gpa: std.mem.Allocator) void {
        gpa.free(self.buffer);
        self.* = undefined;
    }

    /// Queues `bytes`, in pieces when they exceed the free space, waiting
    /// while the queue is full. `Abandoned` once the consumer has gone; what
    /// was not yet queued is dropped.
    pub fn write(self: *Pipe, io: std.Io, bytes: []const u8) error{Abandoned}!void {
        std.debug.assert(!self.closed);
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        var rest = bytes;
        while (rest.len > 0) {
            if (self.abandoned) return error.Abandoned;
            const room = self.buffer.len - self.len;
            if (room == 0) {
                self.changed.waitUncancelable(io, &self.mutex);
                continue;
            }
            const n = @min(room, rest.len);
            @memcpy(self.buffer[self.len..][0..n], rest[0..n]);
            self.len += n;
            rest = rest[n..];
            self.changed.broadcast(io);
        }
        if (self.abandoned) return error.Abandoned;
    }

    /// The producer is done; the consumer drains the queue and then reads `closed`.
    pub fn close(self: *Pipe, io: std.Io) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.closed = true;
        self.changed.broadcast(io);
    }

    /// The consumer is gone (the client closed, a write failed): the
    /// producer's next write, or the one waiting, returns `Abandoned`.
    pub fn abandon(self: *Pipe, io: std.Io) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.abandoned = true;
        self.changed.broadcast(io);
    }

    pub const Taken = union(enum) {
        /// Queued bytes, moved into the caller's buffer.
        data: []u8,
        /// Nothing arrived before the deadline.
        timeout,
        /// The producer closed and everything queued was taken.
        closed,
    };

    /// Waits until bytes are queued, the producer closes, or `deadline`
    /// passes, then moves up to `out.len` queued bytes into `out`.
    pub fn take(self: *Pipe, io: std.Io, out: []u8, deadline: std.Io.Clock.Timestamp) error{Canceled}!Taken {
        std.debug.assert(out.len > 0);
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        while (self.len == 0) {
            if (self.closed) return .closed;
            self.changed.waitTimeout(io, &self.mutex, .{ .deadline = deadline }) catch |err| switch (err) {
                error.Timeout => if (self.len == 0) return if (self.closed) .closed else .timeout,
                error.Canceled => return error.Canceled,
            };
        }
        const n = @min(out.len, self.len);
        @memcpy(out[0..n], self.buffer[0..n]);
        std.mem.copyForwards(u8, self.buffer[0 .. self.len - n], self.buffer[n..self.len]);
        self.len -= n;
        self.changed.broadcast(io);
        return .{ .data = out[0..n] };
    }
};

// ----- tests: a producer task and the test as the consumer -----

fn deadlineIn(io: std.Io, ms: i64) std.Io.Clock.Timestamp {
    return std.Io.Clock.Timestamp.fromNow(io, .{ .raw = .fromMilliseconds(ms), .clock = .awake });
}

const Producer = struct {
    fn run(pipe: *Pipe, io: std.Io, pieces: []const []const u8) error{Abandoned}!void {
        for (pieces) |piece| try pipe.write(io, piece);
        pipe.close(io);
    }
};

test "bytes arrive in order through a queue smaller than they are, then closed" {
    const io = std.testing.io;
    var pipe: Pipe = try .init(std.testing.allocator, 4);
    defer pipe.deinit(std.testing.allocator);
    const pieces = [_][]const u8{ "data: one\n\n", "data: two\n\n", "", "data: three\n\n" };
    var producer = try io.concurrent(Producer.run, .{ &pipe, io, &pieces });
    var received: std.ArrayList(u8) = .empty;
    defer received.deinit(std.testing.allocator);
    var out: [3]u8 = undefined;
    while (true) switch (try pipe.take(io, &out, deadlineIn(io, 5_000))) {
        .data => |bytes| try received.appendSlice(std.testing.allocator, bytes),
        .timeout => return error.TestUnexpectedResult,
        .closed => break,
    };
    try producer.await(io);
    try std.testing.expectEqualStrings("data: one\n\ndata: two\n\ndata: three\n\n", received.items);
}

test "nothing produced before the deadline is a timeout, not the end" {
    const io = std.testing.io;
    var pipe: Pipe = try .init(std.testing.allocator, 16);
    defer pipe.deinit(std.testing.allocator);
    var out: [16]u8 = undefined;
    try std.testing.expectEqual(Pipe.Taken.timeout, try pipe.take(io, &out, deadlineIn(io, 10)));
    try pipe.write(io, "x");
    pipe.close(io);
    try std.testing.expectEqualStrings("x", (try pipe.take(io, &out, deadlineIn(io, 10))).data);
    try std.testing.expectEqual(Pipe.Taken.closed, try pipe.take(io, &out, deadlineIn(io, 10)));
}

test "a producer waiting on a full queue learns the consumer is gone" {
    const io = std.testing.io;
    var pipe: Pipe = try .init(std.testing.allocator, 4);
    defer pipe.deinit(std.testing.allocator);
    const pieces = [_][]const u8{"0123456789"};
    var producer = try io.concurrent(Producer.run, .{ &pipe, io, &pieces });
    var out: [2]u8 = undefined;
    try std.testing.expectEqualStrings("01", (try pipe.take(io, &out, deadlineIn(io, 5_000))).data);
    pipe.abandon(io);
    try std.testing.expectError(error.Abandoned, producer.await(io));
    try std.testing.expectError(error.Abandoned, pipe.write(io, "late"));
}
