//! Decision requests that arrive together share one GPU pass, through the
//! shared queue (`../batch.zig`): a pass takes the jobs of the oldest job's
//! model, in order, while their sequences fit one pass (`decide.batch_rows`;
//! a job is never split, the first always goes), answers them with one
//! `decideJobs`, and leaves the rest for the next. A decision job is
//! answered in the one pass it joins, so it is never `started` while it
//! waits. Packed sequences do not affect each other, so batching changes
//! timing, never answers.
const std = @import("std");
const inference = @import("inference");
const batch = @import("../batch.zig");
const errors = @import("../errors.zig");
const pool_mod = @import("pool.zig");
const timing = @import("../../timing.zig");

const decide = inference.decide;
const ApiError = errors.ApiError;

pub const Job = struct {
    /// The request's arena: the job's sequences and results live here, and
    /// its connection leaves it alone until `wait` returns.
    arena: std.mem.Allocator,
    directory: []const u8,
    /// clef's backbone and projector, when the catalogue names them.
    backbone: ?[]const u8 = null,
    mmproj: ?[]const u8 = null,
    /// The name the request used (the pool remembers the opener's).
    name: []const u8,
    states: []const decide.State,
    questions: []const inference.profiles.laya.Question,
    options: decide.Options = .{},

    next: ?*Job = null,
    state: batch.State = .idle,
    done: std.Io.Event = .unset,
    started: bool = false,
    /// Built once, on the first pass that reaches it.
    prepared: ?decide.Job = null,
    prepare_ns: u64 = 0,
    outcome: Outcome = .pending,

    pub const Outcome = union(enum) { pending, done: Done, failed: ApiError };
    pub const Done = struct {
        results: []const decide.StateResult,
        /// Opening the model, when this pass opened it.
        load_ns: u64,
        /// This job's tokenizing, the pass's encode.
        timings: decide.Timings,
        /// The jobs of the pass it shared.
        batch_jobs: usize,
    };
};

pub const Batcher = batch.Batcher(Job, pool_mod.Pool, struct {
    pub fn same(a: *const Job, b: *const Job) bool {
        return std.mem.eql(u8, a.directory, b.directory);
    }

    pub const run = answer;
});

/// Answers the longest prefix of `taken` that fits one pass (at least one
/// job); the rest stay for the next.
fn answer(self: *Batcher, io: std.Io, taken: []const *Job) void {
    const first = taken[0];
    const acquired = self.pool.acquire(io, .{ .directory = first.directory, .backbone = first.backbone, .mmproj = first.mmproj }, first.name) catch |err| {
        // A generation holds the memory this model needs: the jobs wait for
        // it to end instead of failing.
        if (err == error.Pinned) return;
        for (taken) |job| self.finish(io, job, .{ .failed = openFailure(job.arena, job.name, err) });
        return;
    };
    var jobs: [batch.max_jobs]*decide.Job = undefined;
    var owners: [batch.max_jobs]*Job = undefined;
    var count: usize = 0;
    var rows: usize = 0;
    for (taken) |job| {
        if (job.prepared == null) {
            var clock = std.Io.Clock.awake.now(io);
            job.prepared = acquired.decider.prepare(job.arena, job.states, job.questions, job.options) catch |err| {
                self.finish(io, job, .{ .failed = failure(job.arena, job.name, err) });
                continue;
            };
            job.prepare_ns = timing.lap(io, &clock);
        }
        const job_rows = job.prepared.?.rows; // checked: prepared above, or the loop continued
        if (count > 0 and rows + job_rows > @min(decide.batch_rows, acquired.decider.batchRows())) break;
        jobs[count] = &job.prepared.?; // checked: prepared above, or the loop continued
        owners[count] = job;
        count += 1;
        rows += job_rows;
    }
    if (count == 0) return;
    var timings: decide.Timings = .{};
    acquired.decider.decideJobs(self.scratch.allocator(), io, jobs[0..count], &timings) catch |err| {
        for (owners[0..count]) |job| self.finish(io, job, .{ .failed = failure(job.arena, job.name, err) });
        return;
    };
    self.record(io, count);
    for (owners[0..count]) |job| self.finish(io, job, .{
        .done = .{
            .results = job.prepared.?.results, // checked: every owner was prepared
            .load_ns = acquired.load_ns,
            .timings = .{ .tokenize_ns = job.prepare_ns, .encode_ns = timings.encode_ns },
            .batch_jobs = count,
        },
    });
}

/// A job's failure on the worker, as the client sees it.
pub fn failure(arena: std.mem.Allocator, name: []const u8, err: anyerror) ApiError {
    return switch (err) {
        error.OptionsExceedBudget => .init(.unprocessable_entity, "options_exceed_budget", "a question's options do not fit in the model's budget; shorten them or ask fewer"),
        error.SchemaExceedsBudget => .init(.unprocessable_entity, "options_exceed_budget", "the questions and their options do not fit in the model's 16,384 tokens; ask fewer"),
        error.ImagesUnsupported => .init(.unprocessable_entity, "images_unsupported", arena.print("{s} reads no images; clef-flash does", .{name}) catch "this model reads no images"),
        error.NoVision => .init(.unprocessable_entity, "images_unsupported", arena.print("{s} has no projector pulled (`nuclis model pull {s} --with mmproj`)", .{ name, name }) catch "no projector pulled"),
        error.UnsupportedImageFormat, error.MalformedImage, error.ImageTooLarge, error.TooManyImages => .init(.unprocessable_entity, "invalid_image", arena.print("an image could not be read ({s})", .{@errorName(err)}) catch "an image could not be read"),
        error.OutOfMemory => .init(.internal_server_error, "internal", "out of memory"),
        error.InvalidUtf8, error.LimitExceeded, error.WorkLimitExceeded => .init(.unprocessable_entity, "invalid_request", arena.print("a text could not be tokenized ({s})", .{@errorName(err)}) catch "a text could not be tokenized"),
        else => .init(.internal_server_error, "internal", arena.print("{s}: {s}", .{ name, @errorName(err) }) catch @errorName(err)),
    };
}

pub fn openFailure(arena: std.mem.Allocator, name: []const u8, err: anyerror) ApiError {
    if (err == error.ModelTooLarge) return .init(.unprocessable_entity, "model_too_large", arena.print("{s} needs more memory than the server's budget allows (serve.memory_bytes, --memory)", .{name}) catch "the model needs more memory than the budget allows");
    if (err == error.MetalNotEnabled) return .init(.internal_server_error, "model_failed", "this build has no Metal backend; serve with --backend cpu");
    return .init(.internal_server_error, "model_failed", arena.print("{s}: not a decision checkpoint nuclis can run ({s})", .{ name, @errorName(err) }) catch "the model failed to open");
}
