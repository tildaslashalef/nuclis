//! The queue the batching services (decisions, embeddings) share. Jobs wait
//! in arrival order; one short executor item drains them: it takes the
//! oldest job's model and every waiting job of that model, in order, hands
//! them to the service's pass, and puts what the pass left unfinished back
//! in front, in its order, ahead of later arrivals. A pass may answer any
//! prefix of what it was given and a job may span passes. A waiter that
//! times out takes back only a job no pass has started; a started job is
//! waited for whatever the deadline. No waiting window: an idle GPU starts a
//! lone job at once; jobs of another model wait at most one pass.
const std = @import("std");
const errors = @import("errors.zig");
const gpu = @import("gpu.zig");

/// Jobs one pass may take.
pub const max_jobs = 64;

pub const State = enum { idle, pending, taken, finished, abandoned };
pub const Stats = struct { waiting: usize, passes: u64, jobs: u64 };

/// A queue of `Job`s for the models of one `Pool`. `Job` has the fields
/// `next: ?*Job`, `state: State`, `done: std.Io.Event`, `started: bool` (a
/// pass did part of its work), and `outcome`, a union with a
/// `failed: errors.ApiError` case. `hooks` declares `same(a, b: *const Job)
/// bool` (one model) and `run(*Batcher(Job, Pool, hooks), std.Io, []const
/// *Job) void`, one pass that `finish`es each job it answers or fails and
/// leaves the others alone; it must not touch a job after finishing it (its
/// waiter may have returned and freed it).
pub fn Batcher(comptime Job: type, comptime Pool: type, comptime hooks: type) type {
    return struct {
        const Self = @This();
        pub const Outcome = @FieldType(Job, "outcome");

        executor: *gpu.Executor,
        pool: *Pool,
        mutex: std.Io.Mutex = .init,
        head: ?*Job = null,
        tail: ?*Job = null,
        waiting: usize = 0,
        max_waiting: usize,
        /// The drain item is queued or running.
        scheduled: bool = false,
        /// Short: a pass may run between a generation's steps.
        item: gpu.Item = .{ .run = drain, .short = true },
        /// The pass's lists; reset per pass.
        scratch: std.heap.ArenaAllocator,
        /// Worker-only: the running pass's jobs, null once finished, so
        /// nothing reads a job its waiter may have freed.
        taken: [max_jobs]?*Job = undefined,
        taken_count: usize = 0,
        passes: u64 = 0,
        jobs: u64 = 0,

        pub fn init(gpa: std.mem.Allocator, executor: *gpu.Executor, pool: *Pool, max_waiting: usize) Self {
            return .{ .executor = executor, .pool = pool, .max_waiting = max_waiting, .scratch = .init(gpa) };
        }

        /// After the executor's worker stopped.
        pub fn deinit(self: *Self) void {
            self.scratch.deinit();
        }

        /// Queues `job` and makes sure a drain is scheduled. `Busy` when
        /// `max_waiting` jobs wait already.
        pub fn submit(self: *Self, io: std.Io, job: *Job) error{ Busy, Stopped }!void {
            self.mutex.lockUncancelable(io);
            defer self.mutex.unlock(io);
            std.debug.assert(job.state == .idle); // checked: a job is submitted once, by its handler
            if (self.waiting >= self.max_waiting) return error.Busy;
            self.link(job);
            if (!self.scheduled) {
                self.executor.submit(io, &self.item) catch |err| {
                    self.unlink(job);
                    job.state = .idle;
                    return err;
                };
                self.scheduled = true;
            }
        }

        /// Waits for `job` up to `deadline`; `Timeout` only when no pass has
        /// started it (it is unlinked). A taken or started job is waited for
        /// whatever the deadline.
        pub fn wait(self: *Self, io: std.Io, job: *Job, deadline: std.Io.Clock.Timestamp) error{Timeout}!void {
            while (true) {
                job.done.waitTimeout(io, .{ .deadline = deadline }) catch {
                    if (std.Io.Clock.Timestamp.now(io, deadline.clock).compare(.lt, deadline) and !job.done.isSet()) continue;
                };
                if (job.done.isSet()) return;
                self.mutex.lockUncancelable(io);
                if (job.state == .pending and !job.started) {
                    self.unlink(job);
                    job.state = .abandoned;
                    self.mutex.unlock(io);
                    return error.Timeout;
                }
                self.mutex.unlock(io);
                job.done.waitUncancelable(io);
                return;
            }
        }

        pub fn stats(self: *Self, io: std.Io) Stats {
            self.mutex.lockUncancelable(io);
            defer self.mutex.unlock(io);
            return .{ .waiting = self.waiting, .passes = self.passes, .jobs = self.jobs };
        }

        /// From a pass, before it finishes the jobs it answered: one pass
        /// ran, answering `answered` jobs.
        pub fn record(self: *Self, io: std.Io, answered: usize) void {
            self.mutex.lockUncancelable(io);
            defer self.mutex.unlock(io);
            self.passes += 1;
            self.jobs += answered;
        }

        /// From a pass: sets `job`'s outcome and wakes its waiter, after
        /// which the pass must not touch it.
        pub fn finish(self: *Self, io: std.Io, job: *Job, outcome: Outcome) void {
            for (self.taken[0..self.taken_count]) |*t| {
                if (t.* == job) t.* = null;
            }
            job.outcome = outcome;
            self.mutex.lockUncancelable(io);
            job.state = .finished;
            self.mutex.unlock(io);
            job.done.set(io);
        }

        /// From a pass that cannot go on (out of memory): fails every job it
        /// has not finished, so none is lost or left waiting.
        pub fn failAll(self: *Self, io: std.Io) void {
            for (self.taken[0..self.taken_count]) |t| {
                if (t) |job| self.finish(io, job, .{ .failed = .init(.internal_server_error, "internal", "out of memory") });
            }
        }

        fn link(self: *Self, job: *Job) void {
            job.next = null;
            job.state = .pending;
            job.done = .unset;
            if (self.tail) |t| t.next = job else self.head = job;
            self.tail = job;
            self.waiting += 1;
        }

        fn unlink(self: *Self, job: *Job) void {
            var previous: ?*Job = null;
            var cursor = self.head;
            while (cursor) |c| : ({
                previous = c;
                cursor = c.next;
            }) {
                if (c != job) continue;
                if (previous) |p| p.next = c.next else self.head = c.next;
                if (self.tail == c) self.tail = previous;
                self.waiting -= 1;
                return;
            }
            unreachable; // checked: `unlink` is called under the mutex for a job that is still pending, so in the list
        }

        /// The executor item: one pass, then again while jobs wait.
        fn drain(item: *gpu.Item, io: std.Io) gpu.Item.After {
            const self: *Self = @fieldParentPtr("item", item);
            var given: [max_jobs]*Job = undefined;
            var count: usize = 0;
            self.mutex.lockUncancelable(io);
            const first = self.head orelse {
                self.scheduled = false;
                self.mutex.unlock(io);
                return .done;
            };
            var cursor: ?*Job = first;
            while (cursor) |c| {
                cursor = c.next;
                if (count == max_jobs) break;
                if (!hooks.same(c, first)) continue;
                self.unlink(c);
                c.state = .taken;
                given[count] = c;
                self.taken[count] = c;
                count += 1;
            }
            self.taken_count = count;
            self.mutex.unlock(io);

            _ = self.scratch.reset(.retain_capacity);
            hooks.run(self, io, given[0..count]);

            self.mutex.lockUncancelable(io);
            defer self.mutex.unlock(io);
            // The jobs the pass did not finish go back in front, in their order.
            var i = count;
            while (i > 0) {
                i -= 1;
                const job = self.taken[i] orelse continue;
                job.state = .pending;
                job.next = self.head;
                self.head = job;
                if (self.tail == null) self.tail = job;
                self.waiting += 1;
            }
            self.taken_count = 0;
            if (self.head != null) return .again;
            self.scheduled = false;
            return .done;
        }
    };
}

/// A model-free job: `work` passes of one unit each, on the model `model`.
const TestJob = struct {
    model: u8,
    id: u8,
    work: u8 = 1,
    next: ?*TestJob = null,
    state: State = .idle,
    done: std.Io.Event = .unset,
    started: bool = false,
    outcome: union(enum) { pending, done: u8, failed: errors.ApiError } = .pending,
};

/// Each pass serves at most `per_pass` jobs one unit each and logs their ids.
const TestPool = struct {
    per_pass: usize = max_jobs,
    out_of_memory: bool = false,
    log: std.ArrayList(u8) = .empty,
};

const TestBatcher = Batcher(TestJob, TestPool, struct {
    pub fn same(a: *const TestJob, b: *const TestJob) bool {
        return a.model == b.model;
    }

    pub fn run(self: *TestBatcher, io: std.Io, taken: []const *TestJob) void {
        const scratch = self.scratch.allocator();
        if (self.pool.out_of_memory) return self.failAll(io);
        var answered: std.ArrayList(*TestJob) = .empty;
        for (taken[0..@min(taken.len, self.pool.per_pass)]) |job| {
            self.pool.log.appendAssumeCapacity(job.id);
            job.started = true;
            job.work -= 1;
            if (job.work == 0) answered.append(scratch, job) catch return self.failAll(io);
        }
        self.record(io, answered.items.len);
        for (answered.items) |job| self.finish(io, job, .{ .done = job.id });
    }
});

/// Holds the worker until `gate` is set; `running` says it got there.
const Gate = struct {
    item: gpu.Item = .{ .run = hold },
    running: std.Io.Event = .unset,
    gate: std.Io.Event = .unset,

    fn hold(item: *gpu.Item, io: std.Io) gpu.Item.After {
        const self: *Gate = @fieldParentPtr("item", item);
        self.running.set(io);
        self.gate.waitUncancelable(io);
        return .done;
    }
};

fn deadlineIn(io: std.Io, ms: i64) std.Io.Clock.Timestamp {
    return std.Io.Clock.Timestamp.fromNow(io, .{ .raw = .fromMilliseconds(ms), .clock = .awake });
}

test "a pass takes the oldest job's model and every job of it, in order; the rest wait one pass" {
    const io = std.testing.io;
    var executor: gpu.Executor = .init(8);
    var pool: TestPool = .{ .log = try .initCapacity(std.testing.allocator, 16) };
    defer pool.log.deinit(std.testing.allocator);
    var batcher: TestBatcher = .init(std.testing.allocator, &executor, &pool, 8);
    defer batcher.deinit();
    var jobs = [_]TestJob{ .{ .model = 'a', .id = 1 }, .{ .model = 'b', .id = 2 }, .{ .model = 'a', .id = 3 } };
    for (&jobs) |*j| try batcher.submit(io, j);
    var worker = try io.concurrent(gpu.Executor.run, .{ &executor, io });
    for (&jobs) |*j| try batcher.wait(io, j, deadlineIn(io, 5_000));
    executor.stop(io);
    worker.await(io);
    try std.testing.expectEqualSlices(u8, &.{ 1, 3, 2 }, pool.log.items);
    for (jobs) |j| try std.testing.expectEqual(j.id, j.outcome.done);
    try std.testing.expectEqual(Stats{ .waiting = 0, .passes = 2, .jobs = 3 }, batcher.stats(io));
}

test "what a pass leaves goes back in front, ahead of later jobs, in its order" {
    const io = std.testing.io;
    var executor: gpu.Executor = .init(8);
    var pool: TestPool = .{ .per_pass = 2, .log = try .initCapacity(std.testing.allocator, 16) };
    defer pool.log.deinit(std.testing.allocator);
    var batcher: TestBatcher = .init(std.testing.allocator, &executor, &pool, 8);
    defer batcher.deinit();
    // Job 1 needs two passes; job 3 does not fit the first.
    var jobs = [_]TestJob{ .{ .model = 'a', .id = 1, .work = 2 }, .{ .model = 'a', .id = 2 }, .{ .model = 'a', .id = 3 } };
    for (&jobs) |*j| try batcher.submit(io, j);
    var worker = try io.concurrent(gpu.Executor.run, .{ &executor, io });
    for (&jobs) |*j| try batcher.wait(io, j, deadlineIn(io, 5_000));
    executor.stop(io);
    worker.await(io);
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 1, 3 }, pool.log.items);
    try std.testing.expectEqual(Stats{ .waiting = 0, .passes = 2, .jobs = 3 }, batcher.stats(io));
}

test "a waiter times out only a job no pass started; a started one is waited for" {
    const io = std.testing.io;
    var executor: gpu.Executor = .init(8);
    var pool: TestPool = .{ .log = try .initCapacity(std.testing.allocator, 16) };
    defer pool.log.deinit(std.testing.allocator);
    var batcher: TestBatcher = .init(std.testing.allocator, &executor, &pool, 8);
    defer batcher.deinit();
    var started: TestJob = .{ .model = 'a', .id = 1, .work = 2 };
    var unstarted: TestJob = .{ .model = 'b', .id = 2 };
    try batcher.submit(io, &started);
    try batcher.submit(io, &unstarted);
    // The drain runs one pass, then queues again behind the gate.
    var gate: Gate = .{};
    try executor.submit(io, &gate.item);
    var worker = try io.concurrent(gpu.Executor.run, .{ &executor, io });
    gate.running.waitUncancelable(io);
    try std.testing.expectError(error.Timeout, batcher.wait(io, &unstarted, deadlineIn(io, 10)));
    try std.testing.expectEqual(State.abandoned, unstarted.state);
    try std.testing.expectEqual(@as(usize, 1), batcher.stats(io).waiting);
    try std.testing.expectEqual(State.pending, started.state);
    var opener = try io.concurrent(openLater, .{ io, &gate.gate });
    try batcher.wait(io, &started, deadlineIn(io, 0));
    opener.await(io);
    executor.stop(io);
    worker.await(io);
    try std.testing.expectEqual(@as(u8, 1), started.outcome.done);
    try std.testing.expectEqualSlices(u8, &.{ 1, 1 }, pool.log.items);
}

fn openLater(io: std.Io, gate: *std.Io.Event) void {
    io.sleep(.fromMilliseconds(20), .awake) catch {};
    gate.set(io);
}

test "a pass out of memory fails every job it has not finished; none is requeued" {
    const io = std.testing.io;
    var executor: gpu.Executor = .init(8);
    var pool: TestPool = .{ .out_of_memory = true, .log = try .initCapacity(std.testing.allocator, 16) };
    defer pool.log.deinit(std.testing.allocator);
    var batcher: TestBatcher = .init(std.testing.allocator, &executor, &pool, 8);
    defer batcher.deinit();
    var jobs = [_]TestJob{ .{ .model = 'a', .id = 1 }, .{ .model = 'a', .id = 2 } };
    for (&jobs) |*j| try batcher.submit(io, j);
    var worker = try io.concurrent(gpu.Executor.run, .{ &executor, io });
    for (&jobs) |*j| try batcher.wait(io, j, deadlineIn(io, 5_000));
    executor.stop(io);
    worker.await(io);
    for (jobs) |j| try std.testing.expectEqualStrings("out of memory", j.outcome.failed.message);
    try std.testing.expectEqual(Stats{ .waiting = 0, .passes = 0, .jobs = 0 }, batcher.stats(io));
}

test "a full queue is busy; a stopped executor leaves the job unsubmitted" {
    const io = std.testing.io;
    var pool: TestPool = .{};
    var executor: gpu.Executor = .init(8);
    var batcher: TestBatcher = .init(std.testing.allocator, &executor, &pool, 1);
    defer batcher.deinit();
    var first: TestJob = .{ .model = 'a', .id = 1 };
    var second: TestJob = .{ .model = 'a', .id = 2 };
    try batcher.submit(io, &first);
    try std.testing.expectError(error.Busy, batcher.submit(io, &second));
    try std.testing.expectEqual(State.idle, second.state);
    // No worker runs: the job is still pending at its deadline.
    try std.testing.expectError(error.Timeout, batcher.wait(io, &first, deadlineIn(io, 0)));
    try std.testing.expectEqual(@as(usize, 0), batcher.stats(io).waiting);

    var stopped: gpu.Executor = .init(8);
    stopped.stop(io);
    var refused: TestBatcher = .init(std.testing.allocator, &stopped, &pool, 1);
    defer refused.deinit();
    try std.testing.expectError(error.Stopped, refused.submit(io, &second));
    try std.testing.expectEqual(State.idle, second.state);
    try std.testing.expectEqual(@as(usize, 0), refused.stats(io).waiting);
}
// 5
