//! Embedding requests share GPU passes. Jobs wait here in arrival order;
//! one executor item runs one pass at a time: it opens the oldest job's
//! model, tokenizes each job the first time a pass reaches it, then packs
//! inputs of that model's jobs, in order, while their rows fit `max_rows`
//! (the first input always goes, so a longer one runs alone). A job whose
//! inputs do not all fit continues in the next pass, ahead of later jobs.
//! One pass per run keeps the item short: a generation's steps and other
//! models' jobs interleave with a large request. Packed inputs do not affect
//! each other, so batching changes timing, never vectors.
const std = @import("std");
const inference = @import("inference");
const errors = @import("../errors.zig");
const gpu = @import("../gpu.zig");
const pool_mod = @import("pool.zig");

const embed = inference.embed;
const ApiError = errors.ApiError;

/// Jobs one pass may take.
pub const max_jobs = 64;
/// Rows one pass holds: short enough to run between a generation's steps.
pub const pass_rows = 2048;

pub const Job = struct {
    /// The request's arena: tokens and vectors live here, and its connection
    /// leaves it alone until `wait` returns.
    arena: std.mem.Allocator,
    path: []const u8,
    /// The model's projector, when pulled.
    mmproj: ?[]const u8 = null,
    /// The name the request used (the pool remembers the opener's).
    name: []const u8,
    inputs: []const embed.Input,
    truncate: bool,
    image_tokens: u32 = embed.default_image_tokens,

    next: ?*Job = null,
    state: State = .idle,
    done: std.Io.Event = .unset,
    /// A pass embedded some of its inputs: it is waited for whatever the
    /// deadline.
    started: bool = false,
    /// Built once, on the first pass that reaches it.
    prepared: ?[]embed.Prepared = null,
    /// `embed.dimensions` values per input, filled pass by pass.
    vectors: []const []f32 = &.{},
    /// Inputs embedded so far.
    cursor: usize = 0,
    timings: Timings = .{},
    outcome: Outcome = .pending,

    pub const State = enum { idle, pending, taken, finished, abandoned };
    pub const Outcome = union(enum) { pending, done, failed: ApiError };
    pub const Timings = struct {
        /// Opening the model, when one of its passes opened it.
        load_ns: u64 = 0,
        tokenize_ns: u64 = 0,
        /// The passes it was in, whole.
        embed_ns: u64 = 0,
        passes: usize = 0,
        /// The most jobs a pass it was in held.
        shared: usize = 0,
    };
};

pub const Stats = struct { waiting: usize, passes: u64, jobs: u64 };

pub const Batcher = struct {
    executor: *gpu.Executor,
    pool: *pool_mod.Pool,
    mutex: std.Io.Mutex = .init,
    head: ?*Job = null,
    tail: ?*Job = null,
    waiting: usize = 0,
    max_waiting: usize,
    scheduled: bool = false,
    item: gpu.Item = .{ .run = drain, .short = true },
    /// The worker's lists for one pass; reset per pass.
    scratch: std.heap.ArenaAllocator,
    max_rows: usize = pass_rows,
    passes: u64 = 0,
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

    /// Waits for `job` up to `deadline`; `Timeout` only when no pass has
    /// started it (it is unlinked). A started job is waited for whatever
    /// the deadline.
    pub fn wait(self: *Batcher, io: std.Io, job: *Job, deadline: std.Io.Clock.Timestamp) error{Timeout}!void {
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

    pub fn stats(self: *Batcher, io: std.Io) Stats {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        return .{ .waiting = self.waiting, .passes = self.passes, .jobs = self.jobs };
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

    /// The executor item: one pass, then again while jobs wait.
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
            if (!std.mem.eql(u8, c.path, first.path)) continue;
            self.unlink(c);
            c.state = .taken;
            taken[count] = c;
            count += 1;
        }
        self.mutex.unlock(io);

        const rest = self.pass(io, taken[0..count]);

        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        // Unfinished jobs go back in front, in their order.
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

    /// Runs one pass over `taken` (one model's jobs, in order) and returns
    /// the jobs it did not finish, in `scratch`.
    fn pass(self: *Batcher, io: std.Io, taken: []const *Job) []const *Job {
        _ = self.scratch.reset(.retain_capacity);
        const scratch = self.scratch.allocator();
        const first = taken[0];
        const acquired = self.pool.acquire(io, first.path, first.mmproj, first.name) catch |err| {
            // A generation holds the memory this model needs: wait for it.
            if (err == error.Pinned) return taken;
            for (taken) |job| self.finish(io, job, .{ .failed = openFailure(job.arena, job.name, err) });
            return &.{};
        };
        var rows: std.ArrayList(embed.Prepared) = .empty;
        var vectors: std.ArrayList([]f32) = .empty;
        var members: std.ArrayList(Member) = .empty;
        var used: usize = 0;
        var rest: std.ArrayList(*Job) = .empty;
        for (taken, 0..) |job, i| {
            if (job.prepared == null) {
                var clock = std.Io.Clock.awake.now(io);
                prepare(acquired.embedder, job) catch |err| {
                    // An input over the limit set its own outcome, with the count.
                    self.finish(io, job, if (job.outcome == .failed) job.outcome else .{ .failed = failure(job.arena, job.name, err) });
                    continue;
                };
                job.timings.tokenize_ns = lap(io, &clock);
            }
            const prepared = job.prepared.?;
            // A first image built the vision encoder: the budget counts it.
            self.pool.account(io) catch {};
            const n = fit(prepared, job.cursor, used, self.max_rows);
            if (n == 0) {
                rest.appendSlice(scratch, taken[i..]) catch {};
                break;
            }
            for (prepared[job.cursor..][0..n]) |p| used += batchRows(p);
            rows.appendSlice(scratch, prepared[job.cursor..][0..n]) catch return self.failAll(io, taken);
            vectors.appendSlice(scratch, job.vectors[job.cursor..][0..n]) catch return self.failAll(io, taken);
            members.append(scratch, .{ .job = job, .count = n }) catch return self.failAll(io, taken);
            if (acquired.load_ns > 0) job.timings.load_ns = acquired.load_ns;
            if (job.cursor + n < prepared.len) {
                // The pass is full: this job continues next time, first.
                rest.appendSlice(scratch, taken[i..]) catch {};
                break;
            }
        }
        if (members.items.len == 0) return rest.items;
        var clock = std.Io.Clock.awake.now(io);
        acquired.embedder.embedBatch(rows.items, vectors.items) catch |err| {
            for (members.items) |m| self.finish(io, m.job, .{ .failed = failure(m.job.arena, m.job.name, err) });
            // A failed member that was also left over must not be requeued.
            var kept: std.ArrayList(*Job) = .empty;
            for (rest.items) |job| if (job.state != .finished) kept.append(scratch, job) catch {};
            return kept.items;
        };
        const embed_ns = lap(io, &clock);
        self.mutex.lockUncancelable(io);
        self.passes += 1;
        self.mutex.unlock(io);
        for (members.items) |m| {
            const job = m.job;
            job.started = true;
            job.cursor += m.count;
            job.timings.embed_ns += embed_ns;
            job.timings.passes += 1;
            job.timings.shared = @max(job.timings.shared, members.items.len);
            if (job.cursor == job.inputs.len) {
                self.mutex.lockUncancelable(io);
                self.jobs += 1;
                self.mutex.unlock(io);
                self.finish(io, job, .done);
            }
        }
        return rest.items;
    }

    const Member = struct { job: *Job, count: usize };

    fn failAll(self: *Batcher, io: std.Io, taken: []const *Job) []const *Job {
        for (taken) |job| if (job.state != .finished) self.finish(io, job, .{ .failed = .init(.internal_server_error, "internal", "out of memory") });
        return &.{};
    }

    fn finish(self: *Batcher, io: std.Io, job: *Job, outcome: Job.Outcome) void {
        job.outcome = outcome;
        self.mutex.lockUncancelable(io);
        job.state = .finished;
        self.mutex.unlock(io);
        job.done.set(io);
    }
};

/// Tokenizes every input of `job` (in its arena), encodes its images, and
/// gives it room for its vectors. An input over the limit fails the job
/// unless it truncates; an unreadable image fails it, naming the input.
fn prepare(embedder: *embed.Embedder, job: *Job) !void {
    const prepared = try job.arena.alloc(embed.Prepared, job.inputs.len);
    for (job.inputs, prepared, 0..) |input, *p, i| {
        // Prepared with truncation always, so a refusal can give the count.
        p.* = embedder.prepare(job.arena, input, .{ .truncate = true, .image_tokens = job.image_tokens }) catch |err| {
            switch (err) {
                error.UnsupportedImageFormat, error.MalformedImage, error.ImageTooLarge => {
                    var e: ApiError = .init(.bad_request, "invalid_image", job.arena.print("input[{d}]: {s}", .{ i, if (err == error.ImageTooLarge) "the image is too large to decode" else "not an image nuclis can read (PNG, JPEG, HEIC, WebP, TIFF, GIF, BMP)" }) catch "an image could not be read");
                    e.param = "input";
                    job.outcome = .{ .failed = e };
                },
                else => {},
            }
            return err;
        };
        if (p.truncated() and !job.truncate) {
            job.outcome = .{ .failed = tooLong(job.arena, i, p.length) };
            return error.InputTooLong;
        }
    }
    const vectors = try job.arena.alloc([]f32, job.inputs.len);
    for (vectors) |*v| v.* = try job.arena.alloc(f32, embed.dimensions);
    job.vectors = vectors;
    job.prepared = prepared;
}

fn batchRows(p: embed.Prepared) usize {
    return embed.batchRows(p.tokens.len);
}

/// How many inputs of `prepared`, from `start`, join a pass that holds
/// `used` rows: as many as fit in `max_rows`, and at least one in an empty
/// pass.
pub fn fit(prepared: []const embed.Prepared, start: usize, used: usize, max_rows: usize) usize {
    var rows = used;
    var n: usize = 0;
    for (prepared[start..]) |p| {
        const r = batchRows(p);
        if (rows + r > max_rows and !(rows == 0 and n == 0)) break;
        rows += r;
        n += 1;
    }
    return n;
}

fn lap(io: std.Io, clock: *std.Io.Timestamp) u64 {
    const now = std.Io.Clock.awake.now(io);
    const elapsed: u64 = @intCast(@max(0, clock.durationTo(now).toNanoseconds()));
    clock.* = now;
    return elapsed;
}

fn tooLong(arena: std.mem.Allocator, index: usize, tokens: usize) ApiError {
    var err: ApiError = .init(.bad_request, "input_too_long", arena.print("input[{d}] is {d} tokens; the model reads at most {d} (\"truncate\": true cuts the end)", .{ index, tokens, embed.max_tokens }) catch "an input is over the model's token limit");
    err.param = "input";
    return err;
}

/// A job's failure on the worker, as the client sees it.
pub fn failure(arena: std.mem.Allocator, name: []const u8, err: anyerror) ApiError {
    return switch (err) {
        error.OutOfMemory => .init(.internal_server_error, "internal", "out of memory"),
        error.InvalidUtf8 => .init(.bad_request, "invalid_request", "an input is not valid UTF-8"),
        else => .init(.internal_server_error, "internal", arena.print("{s}: {s}", .{ name, @errorName(err) }) catch @errorName(err)),
    };
}

pub fn openFailure(arena: std.mem.Allocator, name: []const u8, err: anyerror) ApiError {
    if (err == error.ModelTooLarge) return .init(.bad_request, "model_too_large", arena.print("{s} needs more memory than the server's budget allows (serve.memory_bytes, --memory)", .{name}) catch "the model needs more memory than the budget allows");
    if (err == error.MetalNotEnabled) return .init(.internal_server_error, "model_failed", "this build has no Metal backend; serve with --backend cpu");
    return .init(.internal_server_error, "model_failed", arena.print("{s}: not an embedding model nuclis can run ({s})", .{ name, @errorName(err) }) catch "the model failed to open");
}

test "a pass takes inputs while their rows fit, and a lone long input alone" {
    var t: [5][]u32 = undefined;
    var storage: [5][1100]u32 = undefined;
    const lengths = [_]usize{ 1000, 1000, 50, 1100, 9 };
    var prepared: [5]embed.Prepared = undefined;
    for (&prepared, &t, &storage, lengths) |*p, *tokens, *s, n| {
        tokens.* = s[0..n];
        p.* = .{ .tokens = tokens.*, .length = n };
    }
    // 1000 + 1000 rows fit 2048; the 50-row one (56 aligned) does not.
    try std.testing.expectEqual(@as(usize, 2), fit(&prepared, 0, 0, 2048));
    try std.testing.expectEqual(@as(usize, 3), fit(&prepared, 2, 0, 2048));
    try std.testing.expectEqual(@as(usize, 0), fit(&prepared, 2, 2000, 2048));
    // An empty pass takes one input however long; a full one, exactly.
    try std.testing.expectEqual(@as(usize, 1), fit(&prepared, 3, 0, 1000));
    try std.testing.expectEqual(@as(usize, 1), fit(&prepared, 4, 2032, 2048));
}
