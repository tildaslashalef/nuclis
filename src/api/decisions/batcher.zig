//! Decision requests that arrive together share one GPU pass. Jobs wait
//! here in arrival order; one executor item drains them: when the GPU frees
//! it takes the oldest job's model and every job waiting for that model, in
//! order, while their sequences fit one pass (`decide.batch_rows`; a job is
//! never split, the first always goes), answers them with one
//! `decideJobs`, and puts the rest back in front. No waiting window: an idle
//! GPU starts a lone job at once. Jobs of another model wait at most one
//! batch. Packed sequences do not affect each other, so batching changes
//! timing, never answers.
const std = @import("std");
const inference = @import("inference");
const errors = @import("../errors.zig");
const gpu = @import("../gpu.zig");
const pool_mod = @import("pool.zig");

const decide = inference.decide;
const ApiError = errors.ApiError;

/// Jobs one batch may take (the most a pass can hold anyway with any real
/// sequence length).
pub const max_jobs = 64;

pub const Job = struct {
    /// The request's arena: the job's sequences and results live here, and
    /// its connection leaves it alone until `wait` returns.
    arena: std.mem.Allocator,
    directory: []const u8,
    /// The name the request used (the pool remembers the opener's).
    name: []const u8,
    states: []const decide.State,
    questions: []const inference.profiles.laya.Question,
    options: decide.Options = .{},

    next: ?*Job = null,
    state: State = .idle,
    done: std.Io.Event = .unset,
    /// Built once, on the first batch that reaches it.
    prepared: ?decide.Job = null,
    prepare_ns: u64 = 0,
    outcome: Outcome = .pending,

    pub const State = enum { idle, pending, taken, finished, abandoned };
    pub const Outcome = union(enum) { pending, done: Done, failed: ApiError };
    pub const Done = struct {
        results: []const decide.StateResult,
        /// Opening the model, when this batch opened it.
        load_ns: u64,
        /// This job's tokenizing, the batch's encode.
        timings: decide.Timings,
        /// The pass it shared: jobs and rows.
        batch_jobs: usize,
        batch_rows: usize,
    };
};

pub const Stats = struct { waiting: usize, batches: u64, jobs: u64 };

pub const Batcher = struct {
    executor: *gpu.Executor,
    pool: *pool_mod.Pool,
    mutex: std.Io.Mutex = .init,
    head: ?*Job = null,
    tail: ?*Job = null,
    waiting: usize = 0,
    max_waiting: usize,
    /// The drain item is queued or running.
    scheduled: bool = false,
    item: gpu.Item = .{ .run = drain },
    /// The worker's lists for one batch; reset per batch.
    scratch: std.heap.ArenaAllocator,
    max_rows: usize = decide.batch_rows,
    batches: u64 = 0,
    jobs: u64 = 0,

    pub fn init(gpa: std.mem.Allocator, executor: *gpu.Executor, pool: *pool_mod.Pool, max_waiting: usize) Batcher {
        return .{ .executor = executor, .pool = pool, .max_waiting = max_waiting, .scratch = .init(gpa) };
    }

    /// After the executor's worker stopped.
    pub fn deinit(self: *Batcher) void {
        self.scratch.deinit();
    }

    /// Queues `job` and makes sure a drain is scheduled. `Busy` when
    /// `max_waiting` jobs wait already.
    pub fn submit(self: *Batcher, io: std.Io, job: *Job) error{ Busy, Stopped }!void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        std.debug.assert(job.state == .idle);
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

    /// Waits for `job` up to `deadline`; `Timeout` only when no batch took
    /// it (it is unlinked). A taken job is waited for whatever the deadline.
    pub fn wait(self: *Batcher, io: std.Io, job: *Job, deadline: std.Io.Clock.Timestamp) error{Timeout}!void {
        while (true) {
            job.done.waitTimeout(io, .{ .deadline = deadline }) catch {
                if (std.Io.Clock.Timestamp.now(io, deadline.clock).compare(.lt, deadline) and !job.done.isSet()) continue;
            };
            if (job.done.isSet()) return;
            self.mutex.lockUncancelable(io);
            if (job.state == .pending) {
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

    pub fn stats(self: *Batcher, io: std.Io) Stats {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        return .{ .waiting = self.waiting, .batches = self.batches, .jobs = self.jobs };
    }

    fn link(self: *Batcher, job: *Job) void {
        job.next = null;
        job.state = .pending;
        job.done = .unset;
        if (self.tail) |t| t.next = job else self.head = job;
        self.tail = job;
        self.waiting += 1;
    }

    fn unlink(self: *Batcher, job: *Job) void {
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
        unreachable;
    }

    /// The executor item: one batch, then again while jobs wait.
    fn drain(item: *gpu.Item, io: std.Io) gpu.Item.After {
        const self: *Batcher = @fieldParentPtr("item", item);
        var taken: [max_jobs]*Job = undefined;
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
            if (!std.mem.eql(u8, c.directory, first.directory)) continue;
            self.unlink(c);
            c.state = .taken;
            taken[count] = c;
            count += 1;
        }
        self.mutex.unlock(io);

        const rest = self.answer(io, taken[0..count]);

        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        // The jobs that did not fit go back in front, in their order.
        var i = rest.len;
        while (i > 0) {
            i -= 1;
            const job = rest[i];
            job.state = .pending;
            job.next = self.head;
            self.head = job;
            if (self.tail == null) self.tail = job;
            self.waiting += 1;
        }
        if (self.head != null) return .again;
        self.scheduled = false;
        return .done;
    }

    /// Answers the longest prefix of `taken` that fits one pass (at least
    /// one job) and returns the rest.
    fn answer(self: *Batcher, io: std.Io, taken: []const *Job) []const *Job {
        const first = taken[0];
        const acquired = self.pool.acquire(io, first.directory, first.name) catch |err| {
            for (taken) |job| self.finish(io, job, .{ .failed = openFailure(job.arena, job.name, err) });
            return &.{};
        };
        _ = self.scratch.reset(.retain_capacity);
        var batch: [max_jobs]*decide.Job = undefined;
        var owners: [max_jobs]*Job = undefined;
        var count: usize = 0;
        var rows: usize = 0;
        var rest: []const *Job = &.{};
        for (taken, 0..) |job, i| {
            if (job.prepared == null) {
                var clock = std.Io.Clock.awake.now(io);
                job.prepared = acquired.decider.prepare(job.arena, job.states, job.questions, job.options) catch |err| {
                    self.finish(io, job, .{ .failed = failure(job.arena, job.name, err) });
                    continue;
                };
                job.prepare_ns = lap(io, &clock);
            }
            const job_rows = job.prepared.?.rows;
            if (count > 0 and rows + job_rows > self.max_rows) {
                rest = taken[i..];
                break;
            }
            batch[count] = &job.prepared.?;
            owners[count] = job;
            count += 1;
            rows += job_rows;
        }
        if (count == 0) return rest;
        var timings: decide.Timings = .{};
        acquired.decider.decideJobs(self.scratch.allocator(), io, batch[0..count], &timings) catch |err| {
            for (owners[0..count]) |job| self.finish(io, job, .{ .failed = failure(job.arena, job.name, err) });
            return rest;
        };
        self.mutex.lockUncancelable(io);
        self.batches += 1;
        self.jobs += count;
        self.mutex.unlock(io);
        for (owners[0..count]) |job| self.finish(io, job, .{ .done = .{
            .results = job.prepared.?.results,
            .load_ns = acquired.load_ns,
            .timings = .{ .tokenize_ns = job.prepare_ns, .encode_ns = timings.encode_ns },
            .batch_jobs = count,
            .batch_rows = rows,
        } });
        return rest;
    }

    fn finish(self: *Batcher, io: std.Io, job: *Job, outcome: Job.Outcome) void {
        job.outcome = outcome;
        self.mutex.lockUncancelable(io);
        job.state = .finished;
        self.mutex.unlock(io);
        job.done.set(io);
    }
};

fn lap(io: std.Io, clock: *std.Io.Timestamp) u64 {
    const now = std.Io.Clock.awake.now(io);
    const elapsed: u64 = @intCast(@max(0, clock.durationTo(now).toNanoseconds()));
    clock.* = now;
    return elapsed;
}

/// A job's failure on the worker, as the client sees it.
pub fn failure(arena: std.mem.Allocator, name: []const u8, err: anyerror) ApiError {
    return switch (err) {
        error.OptionsExceedBudget => .init(.unprocessable_entity, "options_exceed_budget", "a question's options do not fit in the model's budget; shorten them or ask fewer"),
        error.OutOfMemory => .init(.internal_server_error, "internal", "out of memory"),
        error.InvalidUtf8, error.LimitExceeded, error.WorkLimitExceeded => .init(.unprocessable_entity, "invalid_request", std.fmt.allocPrint(arena, "a text could not be tokenized ({s})", .{@errorName(err)}) catch "a text could not be tokenized"),
        else => .init(.internal_server_error, "internal", std.fmt.allocPrint(arena, "{s}: {s}", .{ name, @errorName(err) }) catch @errorName(err)),
    };
}

pub fn openFailure(arena: std.mem.Allocator, name: []const u8, err: anyerror) ApiError {
    if (err == error.MetalNotEnabled) return .init(.internal_server_error, "model_failed", "this build has no Metal backend; serve with --backend cpu");
    return .init(.internal_server_error, "model_failed", std.fmt.allocPrint(arena, "{s}: not a Laya checkpoint directory nuclis can run ({s})", .{ name, @errorName(err) }) catch "the model failed to open");
}
