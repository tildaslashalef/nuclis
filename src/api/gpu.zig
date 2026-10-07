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
    /// `.again` queues it once more at the tail instead of finishing it.
    run: *const fn (item: *Item, io: std.Io) After,
    next: ?*Item = null,
    state: State = .idle,
    done: std.Io.Event = .unset,
    /// Submitted again while it ran: it runs once more.
    rerun: bool = false,
    /// Short enough to run between the steps of another item (a decision
    /// pass inside a generation), through `runShort`.
    short: bool = false,

    pub const State = enum { idle, queued, running, finished, abandoned };
    pub const After = enum { done, again };
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
    /// An item submitted while it runs is not linked twice: it runs once
    /// more when it finishes (a drain that found nothing, racing a new job).
    pub fn submit(self: *Executor, io: std.Io, item: *Item) error{ Busy, Stopped }!void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        std.debug.assert(item.state != .queued);
        if (self.stopping) return error.Stopped;
        if (item.state == .running) {
            item.rerun = true;
            return;
        }
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

    /// Takes back `item` if it has not started: true when it was queued and
    /// is now unlinked (the worker will never touch it), false when it runs
    /// or has finished.
    pub fn withdraw(self: *Executor, io: std.Io, item: *Item) bool {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        if (item.state != .queued) return false;
        self.unlink(item);
        item.state = .abandoned;
        return true;
    }

    /// Whether `item` has finished, waiting for it at most until `until`.
    /// Unlike `wait`, never blocks past `until`: the caller polls, and
    /// checks its client between polls.
    pub fn poll(self: *Executor, io: std.Io, item: *Item, until: std.Io.Clock.Timestamp) bool {
        _ = self;
        item.done.waitTimeout(io, .{ .deadline = until }) catch {};
        return item.done.isSet();
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

            const after = item.run(item, io);
            self.mutex.lockUncancelable(io);
            self.running = false;
            self.mutex.unlock(io);
            self.settle(io, item, after);
        }
    }

    /// On the worker, from inside a running item: runs each short item
    /// queued at this moment, once, in arrival order. A long item calls it
    /// between its steps so short ones need not wait for it to end.
    pub fn runShort(self: *Executor, io: std.Io) void {
        var taken: [16]*Item = undefined;
        var count: usize = 0;
        self.mutex.lockUncancelable(io);
        var cursor = self.head;
        while (cursor) |c| {
            cursor = c.next;
            if (!c.short or count == taken.len) continue;
            self.unlink(c);
            c.state = .running;
            taken[count] = c;
            count += 1;
        }
        self.mutex.unlock(io);
        for (taken[0..count]) |item| self.settle(io, item, item.run(item, io));
    }

    /// After `item` ran: queued again at the tail, or finished and its
    /// waiter woken.
    fn settle(self: *Executor, io: std.Io, item: *Item, after: Item.After) void {
        self.mutex.lockUncancelable(io);
        self.completed += 1;
        if (after == .again or item.rerun) {
            // Requeued even when full or stopping: it was admitted once.
            item.rerun = false;
            item.next = null;
            item.state = .queued;
            if (self.tail) |t| t.next = item else self.head = item;
            self.tail = item;
            self.queued += 1;
            self.ready.signal(io);
            self.mutex.unlock(io);
            return;
        }
        item.state = .finished;
        self.mutex.unlock(io);
        item.done.set(io);
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
    repeats: u32 = 0,

    fn runOne(item: *Item, io: std.Io) Item.After {
        const self: *Counting = @fieldParentPtr("item", item);
        if (self.gate) |g| g.waitUncancelable(io);
        self.order.appendAssumeCapacity(self.id);
        if (self.repeats > 0) {
            self.repeats -= 1;
            return .again;
        }
        return .done;
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

test "an item that runs again goes behind what queued meanwhile" {
    const io = std.testing.io;
    var executor: Executor = .init(4);
    var order: std.ArrayList(u32) = try .initCapacity(std.testing.allocator, 8);
    defer order.deinit(std.testing.allocator);
    var twice: Counting = .{ .order = &order, .id = 1, .repeats = 1 };
    var other: Counting = .{ .order = &order, .id = 2 };
    try executor.submit(io, &twice.item);
    try executor.submit(io, &other.item);
    var worker = try io.concurrent(Executor.run, .{ &executor, io });
    try executor.wait(io, &twice.item, deadlineIn(io, 5_000));
    try executor.wait(io, &other.item, deadlineIn(io, 5_000));
    executor.stop(io);
    worker.await(io);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 1 }, order.items);
}

test "a queued item can be withdrawn, a running one cannot; poll returns at its deadline" {
    const io = std.testing.io;
    var executor: Executor = .init(4);
    var order: std.ArrayList(u32) = try .initCapacity(std.testing.allocator, 8);
    defer order.deinit(std.testing.allocator);
    var gate: std.Io.Event = .unset;
    var first: Counting = .{ .order = &order, .id = 1, .gate = &gate };
    var second: Counting = .{ .order = &order, .id = 2 };
    try executor.submit(io, &first.item);
    try executor.submit(io, &second.item);
    var worker = try io.concurrent(Executor.run, .{ &executor, io });
    // The first is running (held at its gate) once the queue holds only the second.
    while (executor.stats(io).queued != 1) try io.sleep(.fromMilliseconds(1), .awake);
    try std.testing.expect(!executor.poll(io, &first.item, deadlineIn(io, 20)));
    try std.testing.expect(!executor.withdraw(io, &first.item));
    try std.testing.expect(executor.withdraw(io, &second.item));
    gate.set(io);
    try std.testing.expect(executor.poll(io, &first.item, deadlineIn(io, 5_000)));
    executor.stop(io);
    worker.await(io);
    try std.testing.expectEqualSlices(u32, &.{1}, order.items);
}

const Long = struct {
    item: Item = .{ .run = runLong },
    executor: *Executor,
    order: *std.ArrayList(u32),
    started: std.Io.Event = .unset,
    gate: std.Io.Event = .unset,

    fn runLong(item: *Item, io: std.Io) Item.After {
        const self: *Long = @fieldParentPtr("item", item);
        self.started.set(io);
        self.gate.waitUncancelable(io);
        self.order.appendAssumeCapacity(100);
        // A step boundary: the short items queued meanwhile run here.
        self.executor.runShort(io);
        self.order.appendAssumeCapacity(101);
        return .done;
    }
};

test "short items run between a long item's steps; others wait for it" {
    const io = std.testing.io;
    var executor: Executor = .init(8);
    var order: std.ArrayList(u32) = try .initCapacity(std.testing.allocator, 16);
    defer order.deinit(std.testing.allocator);
    var long: Long = .{ .executor = &executor, .order = &order };
    var short: Counting = .{ .order = &order, .id = 1, .repeats = 1 };
    short.item.short = true;
    var other: Counting = .{ .order = &order, .id = 2 };
    try executor.submit(io, &long.item);
    var worker = try io.concurrent(Executor.run, .{ &executor, io });
    long.started.waitUncancelable(io);
    try executor.submit(io, &other.item);
    try executor.submit(io, &short.item);
    long.gate.set(io);
    try executor.wait(io, &long.item, deadlineIn(io, 5_000));
    try executor.wait(io, &other.item, deadlineIn(io, 5_000));
    try executor.wait(io, &short.item, deadlineIn(io, 5_000));
    executor.stop(io);
    worker.await(io);
    // The short item ran inside the long one; its `.again` went to the
    // tail, behind the item that was not short.
    try std.testing.expectEqualSlices(u32, &.{ 100, 1, 101, 2, 1 }, order.items);
}
