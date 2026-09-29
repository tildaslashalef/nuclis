//! The decision path beside `Engine`: a Laya checkpoint answering typed
//! questions about states. `Decider.open` loads the tokenizer, the agent
//! config, and the model from one directory; `decide` asks every question of
//! every state, each pair one sequence (the encoder is bidirectional, so
//! nothing is shared between them), and returns calibrated answers. Limits
//! are host constants, never the caller's. CPU only until the Metal plan
//! lands. Contract: docs/reference/laya.md.
const std = @import("std");
const hf = @import("tokenizer/hf_json.zig");
const profile = @import("profiles/laya.zig");
const laya = @import("models/laya.zig");

pub const max_states = 64;
pub const max_questions = 32;
pub const max_options = 64;
pub const max_state_bytes = 1024 * 1024;
const max_tokenizer_bytes = 64 * 1024 * 1024;
const max_config_bytes = 1024 * 1024;

pub const Backend = enum { cpu };

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
    /// `rl_agent_config.json`, `encoder/config.json`, `model.safetensors`).
    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, backend: Backend) !Decider {
        _ = backend;
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
        const specials = try profile.Specials.find(&tokenizer);
        var model = try laya.Laya.open(gpa, io, dir);
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
        if (states.len == 0 or questions.len == 0) return error.NothingToDecide;
        if (states.len > max_states) return error.TooManyStates;
        if (questions.len > max_questions) return error.TooManyQuestions;
        for (questions) |q| if (q.texts.len > max_options) return error.TooManyOptions;
        for (states) |s| if (s.text.len > max_state_bytes) return error.StateTooLarge;

        var clock = std.Io.Clock.awake.now(io);
        const prepared = try arena.alloc(profile.Prepared, questions.len);
        for (prepared, questions) |*p, q| p.* = try profile.prepare(arena, &self.tokenizer, self.specials, q);
        const results = try arena.alloc(StateResult, states.len);
        for (results, states) |*result, state| {
            const state_ids = try self.tokenizer.encode(arena, try profile.unmask(arena, state.text), .{});
            timings.tokenize_ns += lap(io, &clock);
            const answers = try arena.alloc(Answer, questions.len);
            result.* = .{ .answers = answers, .state_tokens = state_ids.len, .truncated = false, .input_tokens = 0 };
            for (answers, questions, prepared) |*answer, q, p| {
                const sequence = try profile.assemble(arena, self.specials, p, state_ids, state.truncate, self.config.budget);
                const logits = try arena.alloc(f32, sequence.markers.len);
                timings.tokenize_ns += lap(io, &clock);
                try self.model.logits(io, arena, sequence.ids, sequence.markers, q.kind, logits, null);
                timings.encode_ns += lap(io, &clock);
                const temperature = if (options.uncalibrated) 1 else self.config.temperatureFor(q.kind, logits.len);
                var buffer: [16]u8 = undefined;
                answer.* = .{
                    .calibrated = try profile.calibrate(arena, q.kind, logits, temperature),
                    .logits = logits,
                    .bucket = try arena.dupe(u8, profile.bucket(&buffer, q.kind, logits.len)),
                    .sequence = sequence,
                };
                result.truncated = result.truncated or sequence.truncated;
                result.input_tokens += sequence.ids.len;
            }
        }
        return results;
    }
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
