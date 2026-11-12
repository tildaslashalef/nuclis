//! Embedding requests share GPU passes, through the shared queue
//! (`../batch.zig`): a pass opens the oldest job's model, tokenizes each job
//! the first time a pass reaches it, then packs inputs of that model's jobs,
//! in order, while their rows fit `pass_rows` (the first input always goes,
//! so a longer one runs alone). A job whose inputs do not all fit continues
//! in the next pass, ahead of later jobs, and is `started` from then on. One
//! pass per run keeps the item short: a generation's steps and other models'
//! jobs interleave with a large request. Packed inputs do not affect each
//! other, so batching changes timing, never vectors.
const std = @import("std");
const inference = @import("inference");
const batch = @import("../batch.zig");
const errors = @import("../errors.zig");
const pool_mod = @import("pool.zig");
const timing = @import("../../timing.zig");

const embed = inference.embed;
const ApiError = errors.ApiError;

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
    state: batch.State = .idle,
    done: std.Io.Event = .unset,
    /// A pass embedded some of its inputs: it is waited for whatever the
    /// deadline.
    started: bool = false,
    /// Allocated on the first pass that reaches the job, filled to `ready`.
    prepared: ?[]embed.Prepared = null,
    /// `embed.dimensions` values per input, filled pass by pass.
    vectors: []const []f32 = &.{},
    /// Inputs embedded so far.
    cursor: usize = 0,
    /// Inputs tokenized so far (`prepare`), from 0; at least `cursor`.
    ready: usize = 0,
    timings: Timings = .{},
    outcome: Outcome = .pending,

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

pub const Batcher = batch.Batcher(Job, pool_mod.Pool, struct {
    pub fn same(a: *const Job, b: *const Job) bool {
        return std.mem.eql(u8, a.path, b.path);
    }

    pub const run = pass;
});

const Member = struct { job: *Job, count: usize };

/// Runs one pass over `taken` (one model's jobs, in order); the jobs it does
/// not finish stay for the next.
fn pass(self: *Batcher, io: std.Io, taken: []const *Job) void {
    const scratch = self.scratch.allocator();
    const first = taken[0];
    const acquired = self.pool.acquire(io, first.path, first.mmproj, first.name) catch |err| {
        // A generation holds the memory this model needs: wait for it.
        if (err == error.Pinned) return;
        for (taken) |job| self.finish(io, job, .{ .failed = openFailure(job.arena, job.name, err) });
        return;
    };
    var rows: std.ArrayList(embed.Prepared) = .empty;
    var vectors: std.ArrayList([]f32) = .empty;
    var members: std.ArrayList(Member) = .empty;
    var used: usize = 0;
    for (taken) |job| {
        {
            var clock = std.Io.Clock.awake.now(io);
            prepare(acquired.embedder, job, used, pass_rows) catch |err| {
                // An input over the limit set its own outcome, with the count.
                self.finish(io, job, if (job.outcome == .failed) job.outcome else .{ .failed = failure(job.arena, job.name, err) });
                continue;
            };
            job.timings.tokenize_ns += timing.lap(io, &clock);
        }
        const prepared = job.prepared.?[0..job.ready]; // checked: `prepare` above succeeded, which sets it
        // A first image built the vision encoder: the budget counts it.
        self.pool.account(io) catch |err| {
            if (err == error.Pinned) break;
            // The pool closed the model: this job fails, the rest reopen it next pass.
            self.finish(io, job, .{ .failed = openFailure(job.arena, job.name, err) });
            return;
        };
        const n = fit(prepared, job.cursor, used, pass_rows);
        if (n == 0) break;
        for (prepared[job.cursor..][0..n]) |p| used += batchRows(p);
        rows.appendSlice(scratch, prepared[job.cursor..][0..n]) catch return self.failAll(io);
        vectors.appendSlice(scratch, job.vectors[job.cursor..][0..n]) catch return self.failAll(io);
        members.append(scratch, .{ .job = job, .count = n }) catch return self.failAll(io);
        if (acquired.load_ns > 0) job.timings.load_ns = acquired.load_ns;
        // The pass is full: this job continues next time, first.
        if (job.cursor + n < job.inputs.len) break;
    }
    if (members.items.len == 0) return;
    var clock = std.Io.Clock.awake.now(io);
    acquired.embedder.embedBatch(rows.items, vectors.items) catch |err| {
        for (members.items) |m| self.finish(io, m.job, .{ .failed = failure(m.job.arena, m.job.name, err) });
        return;
    };
    const embed_ns = timing.lap(io, &clock);
    var answered: usize = 0;
    for (members.items) |m| {
        const job = m.job;
        job.started = true;
        job.cursor += m.count;
        job.timings.embed_ns += embed_ns;
        job.timings.passes += 1;
        job.timings.shared = @max(job.timings.shared, members.items.len);
        if (job.cursor == job.inputs.len) answered += 1;
    }
    self.record(io, answered);
    for (members.items) |m| if (m.job.cursor == m.job.inputs.len) self.finish(io, m.job, .done);
}

/// Tokenizes the inputs of `job` a pass holding `used` rows can still take
/// (at least one), in its arena, encoding their images and clips; the first
/// call gives it room for every vector. Tokenizing per pass keeps one GPU
/// item short however many inputs a request carries. An input over the
/// limit fails the job with the count in its outcome.
fn prepare(embedder: *embed.Embedder, job: *Job, used: usize, max_rows: usize) !void {
    if (job.prepared == null) {
        const vectors = try job.arena.alloc([]f32, job.inputs.len);
        for (vectors) |*v| v.* = try job.arena.alloc(f32, embed.dimensions);
        job.vectors = vectors;
        job.prepared = try job.arena.alloc(embed.Prepared, job.inputs.len);
    }
    const prepared = job.prepared.?; // checked: set above when null
    var rows = used;
    for (prepared[job.cursor..job.ready]) |p| rows += batchRows(p);
    while (job.ready < job.inputs.len and (rows < max_rows or job.ready == job.cursor)) : (job.ready += 1) {
        const i = job.ready;
        const p = &prepared[i];
        // Prepared with truncation always, so a refusal can give the count.
        p.* = embedder.prepare(job.arena, job.inputs[i], .{ .truncate = true, .image_tokens = job.image_tokens }) catch |err| {
            const refused: ?ApiError = switch (err) {
                error.UnsupportedImageFormat, error.MalformedImage => .init(.bad_request, "invalid_image", job.arena.print("input[{d}]: not an image nuclis can read ({s})", .{ i, inference.vision.image.readable }) catch "an image could not be read"),
                error.ImageTooLarge => .init(.bad_request, "invalid_image", job.arena.print("input[{d}]: the image is too large to decode", .{i}) catch "an image is too large"),
                error.UnsupportedAudioFormat => .init(.bad_request, "invalid_audio", job.arena.print("input[{d}]: not audio nuclis can read ({s})", .{ i, inference.audio.audio.readable }) catch "a clip could not be read"),
                error.AudioTooShort => .init(.bad_request, "invalid_audio", job.arena.print("input[{d}]: the clip is shorter than 10 ms", .{i}) catch "a clip is too short"),
                else => null,
            };
            if (refused) |r| {
                var e = r;
                e.param = "input";
                job.outcome = .{ .failed = e };
            }
            return err;
        };
        if (p.truncated() and !job.truncate) {
            job.outcome = .{ .failed = if (p.clipped) clipTooLong(job.arena, i) else tooLong(job.arena, i, p.length) };
            return error.InputTooLong;
        }
        rows += batchRows(p.*);
    }
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

fn clipTooLong(arena: std.mem.Allocator, index: usize) ApiError {
    var err: ApiError = .init(.bad_request, "input_too_long", arena.print("input[{d}]: a clip is at most {d} s (Google's processor's bound); \"truncate\": true keeps its first {d} s", .{ index, embed.max_audio_seconds, embed.max_audio_seconds }) catch "a clip is too long");
    err.param = "input";
    return err;
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
