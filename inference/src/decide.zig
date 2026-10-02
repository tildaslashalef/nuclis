//! The decision path beside `Engine`: a Laya checkpoint answering typed
//! questions about states. `Decider.open` loads the tokenizer, the agent
//! config, and the model from one directory; `decide` asks every question of
//! every state, each pair one sequence (the encoder is bidirectional, so
//! nothing is shared between them), and returns calibrated answers. Every
//! sequence of a call goes to the model at once, so the Metal plan can pack
//! them; `prepare` and `decideJobs` do the same for several calls (a server
//! batching its requests). Limits are host constants, never the caller's.
//! Contract: docs/reference/laya.md.
const std = @import("std");
const hf = @import("tokenizer/hf_json.zig");
const profile = @import("profiles/laya.zig");
const laya = @import("models/laya.zig");

pub const max_states = 64;
pub const max_questions = 32;
/// TypeSafe's limit for a choice; a model whose budget cannot hold that
/// many refuses the question itself (`OptionsExceedBudget`).
pub const max_options = 255;
pub const max_state_bytes = 1024 * 1024;
const max_tokenizer_bytes = 64 * 1024 * 1024;
const max_config_bytes = 1024 * 1024;

pub const Backend = laya.Backend;
/// Metal when the build has it.
pub const default_backend: Backend = if (@import("backends/metal/root.zig").enabled) .metal else .cpu;

pub const State = struct {
    /// As the model reads it: an object or list state already rendered as
    /// JSON (`profile.pythonJson`).
    text: []const u8,
    /// Which end survives a cut: a list state (a conversation) keeps its tail.
    truncate: profile.Truncate = .tail,
};

pub const Options = struct {
    /// Softmax of the raw logits (temperature 1).
    uncalibrated: bool = false,
};

pub const Answer = struct {
    calibrated: profile.Calibrated,
    logits: []const f32,
    /// `"<type>:<2|3-5|6-10|11+>"`, the temperature's bucket.
    bucket: []const u8,
    sequence: profile.Sequence,
};

pub const StateResult = struct {
    /// One per question, in order.
    answers: []const Answer,
    /// The state's length in tokens.
    state_tokens: usize,
    /// Whether any question's sequence cut the state.
    truncated: bool,
    /// Every sequence's length summed (the package's `usage.input_tokens`).
    input_tokens: usize,
};

pub const Timings = struct {
    tokenize_ns: u64 = 0,
    encode_ns: u64 = 0,
};

pub const Decider = struct {
    gpa: std.mem.Allocator,
    storage: std.heap.ArenaAllocator,
    tokenizer: hf.Tokenizer,
    config: profile.AgentConfig,
    specials: profile.Specials,
    model: laya.Laya,

    /// Opens a Laya checkpoint directory (`tokenizer/tokenizer.json`,
    /// `tokenizer/tokenizer_config.json`, `rl_agent_config.json`,
    /// `encoder/config.json`, `model.safetensors`).
    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, backend: Backend) !Decider {
        var storage: std.heap.ArenaAllocator = .init(gpa);
        errdefer storage.deinit();
        const arena = storage.allocator();
        const tokenizer_bytes = try readFile(gpa, io, dir, &.{ "tokenizer", "tokenizer.json" }, max_tokenizer_bytes);
        defer gpa.free(tokenizer_bytes);
        var tokenizer = try hf.parse(gpa, tokenizer_bytes, .{});
        errdefer tokenizer.deinit();
        const config_bytes = try readFile(gpa, io, dir, &.{"rl_agent_config.json"}, max_config_bytes);
        defer gpa.free(config_bytes);
        const config = try profile.parseAgentConfig(arena, config_bytes);
        const tokenizer_config = try readFile(gpa, io, dir, &.{ "tokenizer", "tokenizer_config.json" }, max_config_bytes);
        defer gpa.free(tokenizer_config);
        const specials = try profile.Specials.find(gpa, &tokenizer, tokenizer_config);
        var model = try laya.Laya.open(gpa, io, dir, backend);
        errdefer model.deinit(io);
        return .{ .gpa = gpa, .storage = storage, .tokenizer = tokenizer, .config = config, .specials = specials, .model = model };
    }

    pub fn deinit(self: *Decider, io: std.Io) void {
        self.model.deinit(io);
        self.tokenizer.deinit();
        self.storage.deinit();
        self.* = undefined;
    }

    /// Answers every question about every state; `arena` owns the results.
    pub fn decide(self: *const Decider, arena: std.mem.Allocator, io: std.Io, states: []const State, questions: []const profile.Question, options: Options, timings: *Timings) ![]StateResult {
        var clock = std.Io.Clock.awake.now(io);
        var job = try self.prepare(arena, states, questions, options);
        timings.tokenize_ns += lap(io, &clock);
        try self.decideJobs(arena, io, &.{&job}, timings);
        return job.results;
    }

    /// One call's sequences, tokenized and assembled in `arena`, which also
    /// owns its results once `decideJobs` ran. Fails alone: a job that
    /// cannot be built never reaches the model.
    pub fn prepare(self: *const Decider, arena: std.mem.Allocator, states: []const State, questions: []const profile.Question, options: Options) !Job {
        if (states.len == 0 or questions.len == 0) return error.NothingToDecide;
        if (states.len > max_states) return error.TooManyStates;
        if (questions.len > max_questions) return error.TooManyQuestions;
        for (questions) |q| if (q.texts.len > max_options) return error.TooManyOptions;
        for (states) |s| if (s.text.len > max_state_bytes) return error.StateTooLarge;

        const prepared = try arena.alloc(profile.Prepared, questions.len);
        for (prepared, questions) |*p, q| p.* = try profile.prepare(arena, &self.tokenizer, self.specials, q);
        const results = try arena.alloc(StateResult, states.len);
        const count = states.len * questions.len;
        var job: Job = .{
            .arena = arena,
            .questions = questions,
            .options = options,
            .results = results,
            .sequences = try arena.alloc(laya.Sequence, count),
            .outs = try arena.alloc([]f32, count),
            .built = try arena.alloc(profile.Sequence, count),
            .answers = try arena.alloc(Answer, count),
            .rows = 0,
        };
        for (results, states, 0..) |*result, state, si| {
            const state_ids = try self.tokenizer.encode(arena, try self.specials.unmask(arena, state.text), .{});
            result.* = .{ .answers = job.answers[si * questions.len ..][0..questions.len], .state_tokens = state_ids.len, .truncated = false, .input_tokens = 0 };
            for (questions, prepared, 0..) |q, p, qi| {
                const k = si * questions.len + qi;
                job.built[k] = try profile.assemble(arena, self.specials, p, state_ids, state.truncate, self.config.budget);
                job.sequences[k] = .{ .ids = job.built[k].ids, .markers = job.built[k].markers, .kind = q.kind };
                job.outs[k] = try arena.alloc(f32, job.built[k].markers.len);
                job.rows += job.built[k].ids.len;
            }
        }
        return job;
    }

    /// Sends every sequence of every job to the model in one batch (the
    /// Metal plan packs them), then calibrates each job's answers in its own
    /// arena. A packed sequence's logits do not depend on its neighbours, so
    /// batching changes timing, never answers. `scratch` holds the batch's
    /// lists only.
    pub fn decideJobs(self: *const Decider, scratch: std.mem.Allocator, io: std.Io, jobs: []const *Job, timings: *Timings) !void {
        var clock = std.Io.Clock.awake.now(io);
        var total: usize = 0;
        for (jobs) |job| total += job.sequences.len;
        const sequences = try scratch.alloc(laya.Sequence, total);
        const outs = try scratch.alloc([]f32, total);
        var at: usize = 0;
        for (jobs) |job| {
            @memcpy(sequences[at..][0..job.sequences.len], job.sequences);
            @memcpy(outs[at..][0..job.outs.len], job.outs);
            at += job.sequences.len;
        }
        try self.model.logitsBatch(io, scratch, sequences, outs);
        timings.encode_ns += lap(io, &clock);
        for (jobs) |job| try self.answer(job);
    }

    fn answer(self: *const Decider, job: *Job) !void {
        const questions = job.questions;
        for (job.results, 0..) |*result, si| {
            for (questions, 0..) |q, qi| {
                const k = si * questions.len + qi;
                const logits = job.outs[k];
                const temperature = if (job.options.uncalibrated) 1 else self.config.temperatureFor(q.kind, logits.len);
                var buffer: [16]u8 = undefined;
                job.answers[k] = .{
                    .calibrated = try profile.calibrate(job.arena, q.kind, logits, temperature),
                    .logits = logits,
                    .bucket = try job.arena.dupe(u8, profile.bucket(&buffer, q.kind, logits.len)),
                    .sequence = job.built[k],
                };
                result.truncated = result.truncated or job.built[k].truncated;
                result.input_tokens += job.built[k].ids.len;
            }
        }
    }
};

/// A prepared call (`Decider.prepare`): its sequences, where their logits
/// land, and its results once `decideJobs` ran. Everything lives in `arena`.
pub const Job = struct {
    arena: std.mem.Allocator,
    questions: []const profile.Question,
    options: Options,
    /// One per state, in order; filled by `decideJobs`.
    results: []StateResult,
    sequences: []laya.Sequence,
    outs: [][]f32,
    built: []profile.Sequence,
    answers: []Answer,
    /// Tokens over every sequence: the rows the job takes in a batch.
    rows: usize,
};

/// The most rows one Metal pass packs; a batch of jobs is cut to fit it.
pub const batch_rows = @import("models/laya_metal.zig").max_rows;

fn lap(io: std.Io, clock: *std.Io.Timestamp) u64 {
    const now = std.Io.Clock.awake.now(io);
    const elapsed: u64 = @intCast(@max(0, clock.durationTo(now).toNanoseconds()));
    clock.* = now;
    return elapsed;
}

fn readFile(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, parts: []const []const u8, limit: usize) ![]u8 {
    var path_parts: [4][]const u8 = undefined;
    path_parts[0] = dir;
    @memcpy(path_parts[1..][0..parts.len], parts);
    const path = try std.fs.path.join(gpa, path_parts[0 .. 1 + parts.len]);
    defer gpa.free(path);
    return std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(limit));
}
