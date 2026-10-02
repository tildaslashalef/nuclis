//! The decision path beside `Engine`: a decision checkpoint answering typed
//! questions about states. Two families sit behind one `Decider`: Laya (a
//! bidirectional encoder, one sequence per question and state, nothing
//! shared, so the Metal plan packs every sequence of a batch) and clef-flash
//! (a Qwen backbone and a joint schema head, one sequence per state holding
//! every question). `open` picks the family from the directory's files;
//! `parseQuestion` validates a question by the family's rules. `prepare` and
//! `decideJobs` answer several calls at once (a server batching its
//! requests). Limits are host constants, never the caller's. Contracts:
//! docs/reference/laya.md, docs/reference/clef.md.
const std = @import("std");
const hf = @import("tokenizer/hf_json.zig");
const profile = @import("profiles/laya.zig");
const clef_profile = @import("profiles/clef.zig");
const decision = @import("profiles/decision.zig");
const laya = @import("models/laya.zig");
const clef = @import("models/clef.zig");

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

pub const Question = decision.Question;
pub const Kind = decision.Kind;

pub const Family = enum {
    laya,
    clef,

    /// The family a checkpoint directory holds: clef's head file, else Laya.
    pub fn of(io: std.Io, directory: []const u8) Family {
        var buffer: [std.fs.max_path_bytes]u8 = undefined;
        const path = std.fmt.bufPrint(&buffer, "{s}/{s}", .{ directory, clef.head_file }) catch return .laya;
        std.Io.Dir.cwd().access(io, path, .{}) catch return .laya;
        return .clef;
    }
};

/// Validates a question definition by `family`'s rules; `arena` owns it.
pub fn parseQuestion(family: Family, arena: std.mem.Allocator, id: []const u8, value: std.json.Value, diag: *decision.Diagnostic) decision.QuestionError!Question {
    return switch (family) {
        .laya => profile.parseQuestion(arena, id, value, diag),
        .clef => clef_profile.parseQuestion(arena, id, value, diag),
    };
}

/// Where a checkpoint is: its directory, and for clef the backbone GGUF
/// (null: the directory's own `*.gguf` that is not a projector) and the
/// projector (null: requests with images are refused).
pub const Location = struct {
    directory: []const u8,
    backbone: ?[]const u8 = null,
    mmproj: ?[]const u8 = null,
};

/// Images per state, and each image's encoded bytes at most.
pub const max_images = 8;
pub const max_image_bytes = 16 * 1024 * 1024;

pub const State = struct {
    /// As Laya reads it: an object or list state already rendered as JSON
    /// (`profile.pythonJson`).
    text: []const u8,
    /// Which end survives a cut (Laya): a list state (a conversation) keeps
    /// its tail. clef keeps a state's head whatever its shape.
    truncate: profile.Truncate = .tail,
    /// The request's value when it was JSON rather than text; clef renders
    /// it in its own form (`clef_profile.render`).
    json: ?std.json.Value = null,
    /// Encoded images (PNG, JPEG, …) read with the state, in order; clef
    /// only.
    images: []const []const u8 = &.{},
};

pub const Options = struct {
    /// Softmax of the raw logits (temperature 1). clef is never calibrated.
    uncalibrated: bool = false,
};

pub const Answer = struct {
    calibrated: profile.Calibrated,
    /// In the question's key order.
    logits: []const f32,
    /// `"<type>:<2|3-5|6-10|11+>"`, Laya's temperature bucket; empty for clef.
    bucket: []const u8,
    /// Laya's sequence for this question; null for clef (one per state).
    sequence: ?profile.Sequence,
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
    family: union(Family) { laya: Laya, clef: Clef },

    pub fn open(gpa: std.mem.Allocator, io: std.Io, location: Location, backend: Backend) !Decider {
        return .{ .gpa = gpa, .family = switch (Family.of(io, location.directory)) {
            .laya => .{ .laya = try Laya.open(gpa, io, location.directory, backend) },
            .clef => .{ .clef = try Clef.open(gpa, io, location, backend) },
        } };
    }

    pub fn deinit(self: *Decider, io: std.Io) void {
        switch (self.family) {
            inline else => |*f| f.deinit(io),
        }
        self.* = undefined;
    }

    /// The token budget of one sequence (Laya's `max_len`, clef's
    /// `max_length`), for messages.
    pub fn sequenceBudget(self: *const Decider) usize {
        return switch (self.family) {
            .laya => |*l| l.config.budget.max_len,
            .clef => clef_profile.max_length,
        };
    }

    /// The most rows one pass takes; a batch of jobs is cut to fit it.
    pub fn batchRows(self: *const Decider) usize {
        return switch (self.family) {
            .laya => batch_rows,
            .clef => clef_profile.max_length,
        };
    }

    /// Answers every question about every state; `arena` owns the results.
    pub fn decide(self: *const Decider, arena: std.mem.Allocator, io: std.Io, states: []const State, questions: []const Question, options: Options, timings: *Timings) ![]StateResult {
        var clock = std.Io.Clock.awake.now(io);
        var job = try self.prepare(arena, states, questions, options);
        timings.tokenize_ns += lap(io, &clock);
        try self.decideJobs(arena, io, &.{&job}, timings);
        return job.results;
    }

    /// One call's sequences, tokenized and assembled in `arena`, which also
    /// owns its results once `decideJobs` ran. Fails alone: a job that
    /// cannot be built never reaches the model.
    pub fn prepare(self: *const Decider, arena: std.mem.Allocator, states: []const State, questions: []const Question, options: Options) !Job {
        if (states.len == 0 or questions.len == 0) return error.NothingToDecide;
        if (states.len > max_states) return error.TooManyStates;
        if (questions.len > max_questions) return error.TooManyQuestions;
        for (questions) |q| if (q.texts.len > max_options) return error.TooManyOptions;
        for (states) |s| {
            if (s.text.len > max_state_bytes) return error.StateTooLarge;
            if (s.images.len > max_images) return error.TooManyImages;
            for (s.images) |image| if (image.len > max_image_bytes) return error.ImageTooLarge;
            if (s.images.len > 0 and self.family == .laya) return error.ImagesUnsupported;
        }
        const results = try arena.alloc(StateResult, states.len);
        const answers = try arena.alloc(Answer, states.len * questions.len);
        for (results, 0..) |*r, si| r.* = .{ .answers = answers[si * questions.len ..][0..questions.len], .state_tokens = 0, .truncated = false, .input_tokens = 0 };
        var job: Job = .{ .arena = arena, .questions = questions, .options = options, .results = results, .answers = answers, .rows = 0, .family = undefined };
        switch (self.family) {
            .laya => |*l| job.family = .{ .laya = try l.prepare(&job, states) },
            .clef => |*c| job.family = .{ .clef = try c.prepare(&job, states) },
        }
        return job;
    }

    /// Runs every sequence of every job, then answers each job in its own
    /// arena. A sequence's logits do not depend on its neighbours, so
    /// batching changes timing, never answers. `scratch` holds the batch's
    /// lists only.
    pub fn decideJobs(self: *const Decider, scratch: std.mem.Allocator, io: std.Io, jobs: []const *Job, timings: *Timings) !void {
        var clock = std.Io.Clock.awake.now(io);
        switch (self.family) {
            .laya => |*l| try l.run(scratch, io, jobs),
            .clef => |*c| for (jobs) |job| try c.run(io, job),
        }
        timings.encode_ns += lap(io, &clock);
    }
};

/// A prepared call (`Decider.prepare`): its sequences, where their logits
/// land, and its results once `decideJobs` ran. Everything lives in `arena`.
pub const Job = struct {
    arena: std.mem.Allocator,
    questions: []const Question,
    options: Options,
    /// One per state, in order; filled by `decideJobs`.
    results: []StateResult,
    answers: []Answer,
    /// Tokens over every sequence: the rows the job takes in a batch.
    rows: usize,
    family: union(Family) { laya: LayaJob, clef: ClefJob },
};

/// The most rows one Laya Metal pass packs.
pub const batch_rows = @import("models/laya_metal.zig").max_rows;

const Laya = struct {
    storage: std.heap.ArenaAllocator,
    tokenizer: hf.Tokenizer,
    config: profile.AgentConfig,
    specials: profile.Specials,
    model: laya.Laya,

    /// `tokenizer/tokenizer.json`, `tokenizer/tokenizer_config.json`,
    /// `rl_agent_config.json`, `encoder/config.json`, `model.safetensors`.
    fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, backend: Backend) !Laya {
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
        return .{ .storage = storage, .tokenizer = tokenizer, .config = config, .specials = specials, .model = model };
    }

    fn deinit(self: *Laya, io: std.Io) void {
        self.model.deinit(io);
        self.tokenizer.deinit();
        self.storage.deinit();
    }

    fn prepare(self: *const Laya, job: *Job, states: []const State) !LayaJob {
        const arena = job.arena;
        const questions = job.questions;
        const prepared = try arena.alloc(profile.Prepared, questions.len);
        for (prepared, questions) |*p, q| p.* = try profile.prepare(arena, &self.tokenizer, self.specials, q);
        const count = states.len * questions.len;
        const out: LayaJob = .{
            .sequences = try arena.alloc(laya.Sequence, count),
            .outs = try arena.alloc([]f32, count),
            .built = try arena.alloc(profile.Sequence, count),
        };
        for (job.results, states, 0..) |*result, state, si| {
            const state_ids = try self.tokenizer.encode(arena, try self.specials.unmask(arena, state.text), .{});
            result.state_tokens = state_ids.len;
            for (questions, prepared, 0..) |q, p, qi| {
                const k = si * questions.len + qi;
                out.built[k] = try profile.assemble(arena, self.specials, p, state_ids, state.truncate, self.config.budget);
                out.sequences[k] = .{ .ids = out.built[k].ids, .markers = out.built[k].markers, .kind = q.kind };
                out.outs[k] = try arena.alloc(f32, out.built[k].markers.len);
                job.rows += out.built[k].ids.len;
            }
        }
        return out;
    }

    /// Every sequence of every job in one model batch (the Metal plan packs
    /// them), then each job calibrated.
    fn run(self: *const Laya, scratch: std.mem.Allocator, io: std.Io, jobs: []const *Job) !void {
        var total: usize = 0;
        for (jobs) |job| total += job.family.laya.sequences.len;
        const sequences = try scratch.alloc(laya.Sequence, total);
        const outs = try scratch.alloc([]f32, total);
        var at: usize = 0;
        for (jobs) |job| {
            const l = job.family.laya;
            @memcpy(sequences[at..][0..l.sequences.len], l.sequences);
            @memcpy(outs[at..][0..l.outs.len], l.outs);
            at += l.sequences.len;
        }
        try self.model.logitsBatch(io, scratch, sequences, outs);
        for (jobs) |job| try self.answer(job);
    }

    fn answer(self: *const Laya, job: *Job) !void {
        const questions = job.questions;
        const l = job.family.laya;
        for (job.results, 0..) |*result, si| {
            for (questions, 0..) |q, qi| {
                const k = si * questions.len + qi;
                const logits = l.outs[k];
                const temperature = if (job.options.uncalibrated) 1 else self.config.temperatureFor(q.kind, logits.len);
                var buffer: [16]u8 = undefined;
                job.answers[k] = .{
                    .calibrated = try profile.calibrate(job.arena, q.kind, logits, temperature),
                    .logits = logits,
                    .bucket = try job.arena.dupe(u8, profile.bucket(&buffer, q.kind, logits.len)),
                    .sequence = l.built[k],
                };
                result.truncated = result.truncated or l.built[k].truncated;
                result.input_tokens += l.built[k].ids.len;
            }
        }
    }
};

const LayaJob = struct {
    sequences: []laya.Sequence,
    outs: [][]f32,
    built: []profile.Sequence,
};

const Clef = struct {
    model: clef.Model,

    fn open(gpa: std.mem.Allocator, io: std.Io, location: Location, backend: Backend) !Clef {
        return .{ .model = try clef.Model.open(gpa, io, location.directory, location.backbone, location.mmproj, backend) };
    }

    fn deinit(self: *Clef, io: std.Io) void {
        self.model.deinit(io);
    }

    fn prepare(self: *const Clef, job: *Job, states: []const State) !ClefJob {
        const arena = job.arena;
        const ids = try arena.alloc([]const u8, job.questions.len);
        for (ids, job.questions) |*id, q| id.* = q.id;
        const schema = try clef_profile.schema(arena, self.model.encoder(), ids, job.questions);
        const sequences = try arena.alloc(clef_profile.Sequence, states.len);
        const images = try arena.alloc([]clef.Image, states.len);
        for (sequences, images, states, job.results) |*s, *state_images, state, *result| {
            // Decoded here, projected on the model's thread in `run`.
            state_images.* = try arena.alloc(clef.Image, state.images.len);
            const counts = try arena.alloc(usize, state.images.len);
            for (state_images.*, counts, state.images) |*img, *count, bytes| {
                const pixels = try @import("vision/image.zig").decode(arena, bytes);
                img.* = .{ .pixels = pixels, .grid = try self.model.imageGrid(.{ .width = pixels.width, .height = pixels.height }) };
                count.* = img.grid.tokens();
            }
            const media = try clef_profile.mediaIds(arena, self.model.encoder(), counts);
            const text = if (state.json) |v| try clef_profile.render(arena, v) else state.text;
            const state_ids = try clef_profile.tokens(arena, self.model.encoder(), text);
            s.* = try clef_profile.assemble(arena, self.model.frame, schema, media, state_ids, null);
            result.state_tokens = state_ids.len;
            result.truncated = s.truncated;
            result.input_tokens = s.ids.len;
            job.rows += s.ids.len;
        }
        return .{ .sequences = sequences, .images = images };
    }

    /// One state at a time: the backbone's rows, then the head; each
    /// question's logits back in its key order.
    fn run(self: *const Clef, io: std.Io, job: *Job) !void {
        const questions = job.questions;
        for (job.family.clef.sequences, 0..) |sequence, si| {
            const logits = try job.arena.alloc([]f32, questions.len);
            for (logits, questions) |*l, q| l.* = try job.arena.alloc(f32, q.keys.len);
            try self.model.logits(io, sequence, job.family.clef.images[si], logits);
            for (questions, logits, 0..) |q, model_logits, qi| {
                const ordered = try job.arena.alloc(f32, model_logits.len);
                for (model_logits, 0..) |z, i| ordered[if (q.model_order.len > 0) q.model_order[i] else i] = z;
                var calibrated = try profile.calibrate(job.arena, q.kind, ordered, 1);
                // clef's `systemone` reports the top probability as the
                // confidence, not Laya's normalized entropy.
                calibrated.confidence = calibrated.answer_confidence;
                job.answers[si * questions.len + qi] = .{
                    .calibrated = calibrated,
                    .logits = ordered,
                    .bucket = "",
                    .sequence = null,
                };
            }
        }
    }
};

const ClefJob = struct {
    sequences: []clef_profile.Sequence,
    /// Per state, its decoded images.
    images: [][]clef.Image,
};

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
