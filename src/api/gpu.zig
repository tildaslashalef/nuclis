//! The one executor that owns the GPU: services submit items, one worker
//! runs them one at a time in arrival order, so two models never run at
//! once and a model's buffers are only touched by the worker. The queue is
//! bounded (a full queue is `Busy`). An item lives in its submitter's frame:
//! `wait` returns only once the worker can no longer touch it, so a waiter
//! that times out unlinks an item that has not started and otherwise waits
//! for it to finish.
const std = @import("std");

pub const Item = struct {
    /// Runs on the worker; the item's owner reads the outcome after `wait`.
    run: *const fn (item: *Item, io: std.Io) void,
    next: ?*Item = null,
    state: State = .idle,
    done: std.Io.Event = .unset,

    pub const State = enum { idle, queued, running, finished, abandoned };
};

pub const Stats = struct { queued: usize, running: bool, completed: u64 };

pub const Executor = struct {
    mutex: std.Io.Mutex = .init,
    ready: std.Io.Condition = .init,
    head: ?*Item = null,
    tail: ?*Item = null,
    queued: usize = 0,
    max_queued: usize,
    running: bool = false,
    stopping: bool = false,
    completed: u64 = 0,

    pub fn init(max_queued: usize) Executor {
        return .{ .max_queued = max_queued };
    }

    /// Queues `item` behind everything waiting. `Busy` when the queue is
    /// full, `Stopped` once `stop` was called.
    pub fn submit(self: *Executor, io: std.Io, item: *Item) error{ Busy, Stopped }!void {
        std.debug.assert(item.state == .idle);
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        if (self.stopping) return error.Stopped;
        if (self.queued >= self.max_queued) return error.Busy;
        item.next = null;
        item.state = .queued;
        item.done = .unset;
        if (self.tail) |t| t.next = item else self.head = item;
        self.tail = item;
        self.queued += 1;
        self.ready.signal(io);
    }

    /// Waits for `item` up to `deadline`. `Timeout` only when the item never
    /// started (it is unlinked); an item that started is waited for whatever
    /// the deadline, and cancelation is deferred the same way.
    pub fn wait(self: *Executor, io: std.Io, item: *Item, deadline: std.Io.Clock.Timestamp) error{Timeout}!void {
        while (true) {
            item.done.waitTimeout(io, .{ .deadline = deadline }) catch {
                // A spurious wake before the deadline waits again.
                if (std.Io.Clock.Timestamp.now(io, deadline.clock).compare(.lt, deadline) and !item.done.isSet()) continue;
            };
            if (item.done.isSet()) return;
            self.mutex.lockUncancelable(io);
            if (item.state == .queued) {
                self.unlink(item);
                item.state = .abandoned;
                self.mutex.unlock(io);
                return error.Timeout;
            }
            self.mutex.unlock(io);
            item.done.waitUncancelable(io);
            return;
        }
    }

    fn unlink(self: *Executor, item: *Item) void {
        var previous: ?*Item = null;
        var cursor = self.head;
        while (cursor) |c| : ({
            previous = c;
            cursor = c.next;
        }) {
            if (c != item) continue;
            if (previous) |p| p.next = c.next else self.head = c.next;
            if (self.tail == c) self.tail = previous;
            self.queued -= 1;
            return;
        }
        unreachable;
    }

    /// The worker: runs items until `stop`, then returns with the queue
    /// drained (items still queued run first).
    pub fn run(self: *Executor, io: std.Io) void {
        while (true) {
            self.mutex.lockUncancelable(io);
            while (self.head == null and !self.stopping) self.ready.waitUncancelable(io, &self.mutex);
            const item = self.head orelse {
                self.mutex.unlock(io);
                return;
            };
            self.head = item.next;
            if (self.head == null) self.tail = null;
            self.queued -= 1;
            item.state = .running;
            self.running = true;
            self.mutex.unlock(io);

            item.run(item, io);

            self.mutex.lockUncancelable(io);
            item.state = .finished;
            self.running = false;
            self.completed += 1;
            self.mutex.unlock(io);
            item.done.set(io);
        }
    }

    pub fn stop(self: *Executor, io: std.Io) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.stopping = true;
        self.ready.broadcast(io);
    }

    pub fn stats(self: *Executor, io: std.Io) Stats {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        return .{ .queued = self.queued, .running = self.running, .completed = self.completed };
    }
};

const Counting = struct {
    item: Item = .{ .run = runOne },
    order: *std.ArrayList(u32),
    id: u32,
    gate: ?*std.Io.Event = null,

    fn runOne(item: *Item, io: std.Io) void {
        const self: *Counting = @fieldParentPtr("item", item);
        if (self.gate) |g| g.waitUncancelable(io);
        self.order.appendAssumeCapacity(self.id);
    }
};

fn deadlineIn(io: std.Io, ms: i64) std.Io.Clock.Timestamp {
    return std.Io.Clock.Timestamp.fromNow(io, .{ .raw = .fromMilliseconds(ms), .clock = .awake });
}

test "items run one at a time in arrival order; a full queue is busy" {
    const io = std.testing.io;
    var executor: Executor = .init(2);
    var order: std.ArrayList(u32) = try .initCapacity(std.testing.allocator, 8);
    defer order.deinit(std.testing.allocator);
    var gate: std.Io.Event = .unset;
    var first: Counting = .{ .order = &order, .id = 1, .gate = &gate };
    var second: Counting = .{ .order = &order, .id = 2 };
    var third: Counting = .{ .order = &order, .id = 3 };
    try executor.submit(io, &first.item);
    try executor.submit(io, &second.item);
    try std.testing.expectError(error.Busy, executor.submit(io, &third.item));
    var worker = try io.concurrent(Executor.run, .{ &executor, io });
    gate.set(io);
    try executor.wait(io, &first.item, deadlineIn(io, 5_000));
    try executor.wait(io, &second.item, deadlineIn(io, 5_000));
    try executor.submit(io, &third.item);
    try executor.wait(io, &third.item, deadlineIn(io, 5_000));
    executor.stop(io);
    worker.await(io);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3 }, order.items);
    try std.testing.expectEqual(@as(u64, 3), executor.stats(io).completed);
    var late: Counting = .{ .order = &order, .id = 4 };
    try std.testing.expectError(error.Stopped, executor.submit(io, &late.item));
}

test "a waiter that times out unlinks its queued item; a running one is waited for" {
    const io = std.testing.io;
    var executor: Executor = .init(4);
    var order: std.ArrayList(u32) = try .initCapacity(std.testing.allocator, 8);
    defer order.deinit(std.testing.allocator);
    var gate: std.Io.Event = .unset;
    var blocking: Counting = .{ .order = &order, .id = 1, .gate = &gate };
    var waiting: Counting = .{ .order = &order, .id = 2 };
    var after: Counting = .{ .order = &order, .id = 3 };
    try executor.submit(io, &blocking.item);
    try executor.submit(io, &waiting.item);
    try executor.submit(io, &after.item);
    var worker = try io.concurrent(Executor.run, .{ &executor, io });
    // The worker holds item 1 behind the gate; item 2 never starts.
    try std.testing.expectError(error.Timeout, executor.wait(io, &waiting.item, deadlineIn(io, 20)));
    try std.testing.expectEqual(Item.State.abandoned, waiting.item.state);
    try std.testing.expectEqual(@as(usize, 1), executor.stats(io).queued);
    gate.set(io);
    try executor.wait(io, &blocking.item, deadlineIn(io, 5_000));
    try executor.wait(io, &after.item, deadlineIn(io, 5_000));
    executor.stop(io);
    worker.await(io);
    try std.testing.expectEqualSlices(u32, &.{ 1, 3 }, order.items);
}
