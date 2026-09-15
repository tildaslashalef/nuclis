//! The engine seam: what every consumer of the library (the `generate` and
//! `bench` commands, the chat playground and its agent, a future server)
//! composes the same way. `Engine` owns the mapped artifact, vocabulary,
//! encoder, and the executing `Model`; `runLoop` is the prefill/decode loop
//! over it. Model semantics stay in the adapters and backends; presentation,
//! files, terminals, and signals stay with the caller, which reaches in only
//! through `Hooks` and the layer `Observer`.
//!
//! The adapter is chosen once, in `open`, from `general.architecture`
//! through the registry (`models.select`); the prompt profile is
//! chosen from the template digest (`profiles.forDocument`). Per-token work
//! dispatches on an enum tag, never on a string.
//!
//! Ownership: `Engine.open` returns a value that owns everything it holds
//! until `deinit`; `prompt`/`render`/`encode` return caller-owned slices.
//! `runLoop` borrows the engine, the sampler, and caller-provided scratch
//! (`logits`, `candidates`, `generated`) and never allocates. `complete` adds
//! per-piece decoding scratch, borrowed events, and final metrics for agents.
const std = @import("std");
const inference = @import("root.zig");
const models = inference.models;
const profiles = inference.profiles;
pub const Observer = inference.observer.Observer;

pub const Backend = enum { cpu, metal };
/// Attention cache precision. A GPU option: the CPU reference runtime
/// keeps F32 whatever is requested, and `Engine.kv_precision` reports what
/// the session actually uses.
pub const KvPrecision = inference.session.Precision;

pub const StopReason = enum {
    eos,
    token_budget,
    context_limit,
    cancelled,
    failure,
};

/// One family's executors: the CPU reference runtime or the GPU-resident
/// plan. Both own a `session.Session` and expose the same step/reset
/// contract; the engine's `Model` wraps one `Executor` per registered family.
pub fn Executor(comptime Family: type) type {
    return union(enum) {
        const Self = @This();
        cpu: Family.Runtime,
        metal: struct { backend: *inference.metal.Backend, plan: Family.Plan },

        /// One token step. `greedy` selects the argmax on the GPU and `topk`
        /// reads back the partial top-k for sampling when available; the CPU
        /// runtime ignores both and callers sample from `logits`.
        pub fn step(self: *Self, token: u32, logits: ?[]f32, greedy: ?*u32, topk: ?*inference.sampling.TopK, observer: ?Observer) !void {
            switch (self.*) {
                .cpu => |*runtime| try runtime.step(token, logits, observer),
                .metal => |*m| try m.plan.step(token, logits, greedy, topk, observer),
            }
        }
        /// Consumes a prompt. The GPU plan batches it in chunks unless the
        /// observer wants per-layer activations, which are per token by contract;
        /// the CPU reference steps token by token. Readbacks refer to the last token.
        pub fn prefill(self: *Self, tokens: []const u32, logits: ?[]f32, greedy: ?*u32, topk: ?*inference.sampling.TopK, observer: ?Observer) !void {
            if (tokens.len == 0) return error.InvalidShape;
            switch (self.*) {
                .cpu => |*runtime| for (tokens, 0..) |token, i| try runtime.step(token, if (i + 1 == tokens.len) logits else null, observer),
                .metal => |*m| {
                    if (observer != null and observer.?.layer != null) {
                        for (tokens, 0..) |token, i| {
                            const last = i + 1 == tokens.len;
                            try m.plan.step(token, if (last) logits else null, if (last) greedy else null, if (last) topk else null, observer);
                        }
                    } else try m.plan.prefill(tokens, logits, greedy, topk, observer);
                },
            }
        }
        pub fn readLogits(self: *Self, out: []f32) !void {
            switch (self.*) {
                .cpu => return error.LogitsNotRetained,
                .metal => |*m| try m.plan.readLogits(out),
            }
        }
        pub fn reset(self: *Self) void {
            switch (self.*) {
                .cpu => |*runtime| runtime.reset(),
                .metal => |*m| m.plan.reset(),
            }
        }
        pub fn session(self: *const Self) *const inference.session.Session {
            return switch (self.*) {
                .cpu => |*runtime| &runtime.state,
                .metal => |*m| &m.plan.state,
            };
        }
        pub fn sessionMut(self: *Self) *inference.session.Session {
            return switch (self.*) {
                .cpu => |*runtime| &runtime.state,
                .metal => |*m| &m.plan.state,
            };
        }
        pub fn gpu(self: *const Self) ?*inference.metal.Backend {
            return switch (self.*) {
                .cpu => null,
                .metal => |*m| m.backend,
            };
        }
        pub fn deinit(self: *Self, alloc: std.mem.Allocator) void {
            switch (self.*) {
                .cpu => |*runtime| runtime.deinit(),
                .metal => |*m| {
                    // The plan's session memory is wrapped by the backend: release
                    // the plan (session) only after GPU work is complete, which the
                    // synchronous backend guarantees, then the backend itself.
                    m.plan.deinit();
                    m.backend.deinit();
                    alloc.destroy(m.backend);
                },
            }
        }
    };
}

/// The executor union over the registry's table, tagged by `models.Adapter`:
/// one field per family, holding that family's `Executor`. Built with
/// `@Union` from the same table as the tag, so a family cannot be registered
/// without an executor or the other way round.
pub const Executors = blk: {
    const families = models.table;
    var names: [families.len][]const u8 = undefined;
    var types: [families.len]type = undefined;
    for (families, 0..) |Family, i| {
        names[i] = Family.architecture;
        types[i] = Executor(Family);
    }
    break :blk @Union(.auto, models.Adapter, &names, &types, &@splat(.{}));
};

/// The executing model: one family's executor behind the adapter tag. Every
/// method is an `inline else` switch, so the per-token path is a jump on
/// the tag with no string comparison (the spec's loading-time dispatch rule).
pub const Model = struct {
    exec: Executors,

    pub fn adapter(self: *const Model) models.Adapter {
        return self.exec;
    }
    pub fn step(self: *Model, token: u32, logits: ?[]f32, greedy: ?*u32, topk: ?*inference.sampling.TopK, observer: ?Observer) !void {
        switch (self.exec) {
            inline else => |*e| try e.step(token, logits, greedy, topk, observer),
        }
    }
    pub fn prefill(self: *Model, tokens: []const u32, logits: ?[]f32, greedy: ?*u32, topk: ?*inference.sampling.TopK, observer: ?Observer) !void {
        switch (self.exec) {
            inline else => |*e| try e.prefill(tokens, logits, greedy, topk, observer),
        }
    }
    /// Whether `prefill` processes the prompt as chunks rather than per-token
    /// steps for this observer (the per-token hooks then run once per prompt).
    pub fn chunkedPrefill(self: *const Model, observer: ?Observer) bool {
        return self.gpu() != null and (observer == null or observer.?.layer == null);
    }
    /// The last step's full logits, for the sampler's fallback from `topk`.
    /// Only the GPU plan keeps them on the device; the CPU runtime returns
    /// them from `step` directly.
    pub fn readLogits(self: *Model, out: []f32) !void {
        switch (self.exec) {
            inline else => |*e| try e.readLogits(out),
        }
    }
    pub fn reset(self: *Model) void {
        switch (self.exec) {
            inline else => |*e| e.reset(),
        }
    }
    pub fn session(self: *const Model) *const inference.session.Session {
        return switch (self.exec) {
            inline else => |*e| e.session(),
        };
    }
    fn sessionMut(self: *Model) *inference.session.Session {
        return switch (self.exec) {
            inline else => |*e| e.sessionMut(),
        };
    }
    /// The Metal backend when the model executes on it, for profiling and
    /// GPU time; null on the CPU reference.
    pub fn gpu(self: *const Model) ?*inference.metal.Backend {
        return switch (self.exec) {
            inline else => |*e| e.gpu(),
        };
    }
    /// A caller-owned checkpoint of the committed state. Valid between
    /// steps: the synchronous GPU backend has waited for every command
    /// buffer, so the session memory is quiescent. A step that failed since
    /// the last reset is refused (`SessionNotReady`).
    pub fn snapshot(self: *const Model, gpa: std.mem.Allocator) !inference.session.Snapshot {
        return self.session().snapshot(gpa);
    }
    /// Returns the model to a snapshot taken from a session of the same
    /// capacity and layout; the next step continues from its position. A
    /// mismatch is `SnapshotMismatch` and leaves the session untouched.
    pub fn restore(self: *Model, snap: *const inference.session.Snapshot) !void {
        try self.sessionMut().restore(snap);
    }
    pub fn supportsGpuArgmax(self: *const Model) bool {
        return self.gpu() != null;
    }
    pub fn supportsGpuTopK(self: *const Model) bool {
        return self.gpu() != null;
    }
    fn deinit(self: *Model, alloc: std.mem.Allocator) void {
        switch (self.exec) {
            inline else => |*e| e.deinit(alloc),
        }
    }
};

/// Prompt tokens per prefill command buffer on the GPU plan. Bounds the
/// chunk activation buffers (~0.4 MB per token) and the work between
/// cancellation checks; the plan clamps it to the session capacity.
pub const prefill_chunk = 256;

/// Builds one family's executor for the backend. Heap-allocates the Metal
/// backend so the plan's pointer stays valid when the Engine value is
/// returned by value; its diagnostic text is logged on failure.
fn openExecutor(comptime Family: type, alloc: std.mem.Allocator, view: inference.weights.View, binding: Family.Binding, backend: Backend, capacity: usize, kv: KvPrecision) !Executor(Family) {
    return switch (backend) {
        .cpu => .{ .cpu = try Family.Runtime.init(alloc, view, binding, capacity) },
        .metal => blk: {
            const gpu = try alloc.create(inference.metal.Backend);
            errdefer alloc.destroy(gpu);
            var diagnostic: [8192]u8 = @splat(0);
            gpu.* = inference.metal.Backend.init(alloc, &diagnostic) catch |err| {
                // A build without the backend fails before it can say
                // anything; logging an empty line would only add noise above
                // the error the caller reports.
                const reason = std.mem.sliceTo(&diagnostic, 0);
                if (reason.len > 0) std.log.err("{s}", .{reason});
                return err;
            };
            errdefer gpu.deinit();
            const plan = try Family.Plan.init(alloc, gpu, view, binding, capacity, @min(prefill_chunk, capacity), kv);
            break :blk .{ .metal = .{ .backend = gpu, .plan = plan } };
        },
    };
}

/// A profile lists at most this many stop tokens (both today list two).
pub const max_stop_tokens = 4;

pub const Engine = struct {
    alloc: std.mem.Allocator,
    io: std.Io,
    backend: Backend,
    mapped: inference.weights.Mapped,
    vocab: inference.vocabulary.Vocabulary,
    encoder: inference.tokenizer.Encoder,
    model: Model,
    /// The profile pinned to the artifact's chat template, or null when no
    /// profile implements that exact template (`render` then refuses; raw
    /// prompts still run).
    profile: ?profiles.Profile,
    /// The ids that end generation: the profile's stop tokens resolved in
    /// this vocabulary, or the file's `eos_token_id` alone without a profile.
    stop_ids: [max_stop_tokens]u32,
    stop_count: usize,
    /// The attention cache precision the session was built with: the
    /// request on the GPU plan, always `f32` on the CPU reference.
    kv_precision: KvPrecision,
    /// `general.name` from the artifact metadata, borrowed from the mapping.
    name: []const u8,
    /// Wall time spent in `open`, including directory parsing and mapping.
    load: std.Io.Duration,

    /// Opens the artifact, selects its adapter and binds it, loads the
    /// vocabulary, and prepares a session of `capacity` tokens with an
    /// attention cache of `kv` precision (GPU only; the CPU reference stays
    /// F32). An architecture without an adapter is `UnknownArchitecture`;
    /// `models.known` names the ones the tree has.
    pub fn open(alloc: std.mem.Allocator, io: std.Io, model_path: []const u8, backend: Backend, capacity: usize, kv: KvPrecision) !Engine {
        const started = std.Io.Clock.awake.now(io);
        if (capacity == 0 or capacity > 32768) return error.InvalidGenerationBudget;
        var mapped = try inference.weights.Mapped.open(alloc, io, model_path);
        errdefer mapped.deinit(io);
        const adapter = try models.select(mapped.document.string("general.architecture") orelse "");
        var vocab = try inference.vocabulary.load(alloc, mapped.document, mapped.mapping.memory[0..@intCast(mapped.document.directory_bytes)], .{});
        errdefer vocab.deinit();
        const profile = profiles.forDocument(mapped.document);
        var stop_ids: [max_stop_tokens]u32 = undefined;
        var stop_count: usize = 0;
        if (profile) |p| {
            // A profile whose stop token is missing from the vocabulary is
            // a mismatch between template and tokenizer: refuse, never guess.
            for (p.stopTokens()) |text| {
                stop_ids[stop_count] = vocab.tokenId(text) orelse return error.MissingStopToken;
                stop_count += 1;
            }
        } else if (vocab.eos) |eos| {
            stop_ids[0] = eos;
            stop_count = 1;
        }
        var model: Model = switch (adapter) {
            inline else => |a| blk: {
                const Family = models.registry.family(a);
                const binding = try Family.bind(alloc, &mapped.document);
                break :blk .{ .exec = @unionInit(Executors, @tagName(a), try openExecutor(Family, alloc, mapped.view(), binding, backend, capacity, kv)) };
            },
        };
        errdefer model.deinit(alloc);
        var encoder = try inference.tokenizer.Encoder.init(alloc, &vocab);
        errdefer encoder.deinit();
        return .{
            .alloc = alloc,
            .io = io,
            .backend = backend,
            .mapped = mapped,
            .vocab = vocab,
            .encoder = encoder,
            .model = model,
            .profile = profile,
            .stop_ids = stop_ids,
            .stop_count = stop_count,
            .kv_precision = if (backend == .cpu) .f32 else kv,
            .name = mapped.document.string("general.name") orelse "unnamed model",
            .load = started.durationTo(std.Io.Clock.awake.now(io)),
        };
    }

    pub fn deinit(self: *Engine) void {
        self.model.deinit(self.alloc);
        self.encoder.deinit();
        self.vocab.deinit();
        self.mapped.deinit(self.io);
        self.* = undefined;
    }

    /// Renders one user turn with the artifact's profile, or returns the raw
    /// text unchanged. Caller owns the result.
    pub fn prompt(self: *const Engine, user: []const u8, raw: bool, effort: profiles.Effort) ![]u8 {
        if (raw) return self.alloc.dupe(u8, user);
        return self.render(&.{.{ .role = .user, .content = user }}, &.{}, effort);
    }

    /// Renders a completion-ready conversation (a user message or a completed
    /// tool-result group last) with the artifact's profile and optional tool
    /// definitions. Caller owns the result. A profile that cannot render the
    /// tools yet returns `error.ToolsUnsupported`.
    pub fn render(self: *const Engine, messages: []const profiles.Message, tools: []const profiles.ToolDefinition, effort: profiles.Effort) ![]u8 {
        const profile = self.profile orelse return error.UnsupportedPromptTemplate;
        return profile.render(self.alloc, messages, tools, effort, .{});
    }

    /// Encodes text with special-token markers recognized. Caller owns the IDs.
    pub fn encode(self: *Engine, text: []const u8) ![]u32 {
        const tokens = try self.encoder.encode(self.alloc, text, true, .{});
        if (tokens.len == 0) {
            self.alloc.free(tokens);
            return error.EmptyPrompt;
        }
        return tokens;
    }

    /// Whether `token` ends the turn: one of the profile's stop tokens (for
    /// Qwen3.8 `<|im_end|>` and `<|endoftext|>`, for Gemma 4 `<turn|>` and
    /// `<eos>`), or the file's EOS when no profile matched.
    pub fn isStop(self: *const Engine, token: u32) bool {
        return std.mem.indexOfScalar(u32, self.stop_ids[0..self.stop_count], token) != null;
    }

    /// The reasoning markers of the artifact's profile, for splitting
    /// generated text; null without a profile.
    pub fn reasoning(self: *const Engine) ?profiles.Reasoning {
        return if (self.profile) |p| p.reasoning() else null;
    }

    pub fn gpuSeconds(self: *const Engine) ?f64 {
        return if (self.model.gpu()) |gpu| gpu.gpuSeconds() else null;
    }
};

/// Timings for one prefill+decode pass. Durations are wall clock from the
/// injected Io; GPU busy time comes from Metal timestamps when available.
pub const Timing = struct {
    prompt_tokens: usize = 0,
    generated_tokens: usize = 0,
    prefill: std.Io.Duration = .zero,
    /// Time to first token: from the first prompt step to the first sampled
    /// token, so it includes prefill.
    first_token: std.Io.Duration = .zero,
    /// From the first sampled token to the last: covers `generated_tokens - 1`
    /// model steps plus sampling.
    decode: std.Io.Duration = .zero,
    gpu_seconds: ?f64 = null,
    /// Present when the GPU partial top-k path sampled this run: how many
    /// tokens needed the full-logit fallback (see reference/generation.md).
    topk_fallbacks: ?usize = null,
};

pub const Outcome = struct {
    stop: StopReason,
    timing: Timing,
};

/// Callbacks observed by the loop. `prefill` runs once with the final prompt
/// logits before decoding; `token` runs for each generated token, including
/// the stop token, before the next step.
pub const Hooks = struct {
    context: *anyopaque,
    prefill: ?*const fn (*anyopaque, []const f32) anyerror!void = null,
    token: ?*const fn (*anyopaque, u32) anyerror!void = null,
    /// Runs after every completed model step (prompt and generated tokens)
    /// with the session position; interactive callers poll input and redraw here.
    step: ?*const fn (*anyopaque, usize) anyerror!void = null,
    /// Runs before every model step with the session position the step will
    /// occupy; trace writers name their per-layer files by it.
    before_step: ?*const fn (*anyopaque, usize) anyerror!void = null,
};

/// Caller-owned sampling scratch and request mode. The effort must match
/// the prompt's render: it determines whether the reasoning channel is open.
pub const CompletionBuffers = struct {
    logits: []f32,
    candidates: []inference.sampling.Candidate,
    generated: []u32,
    effort: profiles.Effort,
};

/// Completes a rendered prompt as semantic events. `sink.send(Event)` is a
/// synchronous, fallible call; borrowed text expires when it returns. The
/// observer retains cancellation/progress without introducing logit readback.
/// A successful call emits exactly one final stop, including cancellation;
/// an inference or sink error propagates without a fabricated successful stop.
/// `runLoop` remains the raw-token primitive used by generate and benchmarks.
pub fn complete(
    eng: *Engine,
    tokens: []const u32,
    limit: usize,
    sampler: *inference.sampling.Sampler,
    history: ?*inference.sampling.History,
    buffers: CompletionBuffers,
    observer: ?Observer,
    sink: anytype,
) !Outcome {
    if (limit > buffers.generated.len) return error.InvalidGenerationBudget;
    const profile = eng.profile orelse return error.UnsupportedPromptTemplate;
    const Bridge = struct {
        eng: *Engine,
        decoder: profiles.stream.Decoder,
        sink: @TypeOf(sink),

        fn token(context: *anyopaque, id: u32) !void {
            const self: *@This() = @ptrCast(@alignCast(context));
            if (self.eng.isStop(id)) return;
            const piece = try inference.bpe.decode(self.eng.alloc, &self.eng.vocab, &.{id}, true, .{});
            defer self.eng.alloc.free(piece);
            try self.decoder.feed(id, piece, self.sink);
        }
    };
    var bridge: Bridge = .{ .eng = eng, .decoder = try profile.decoder(eng.alloc, &eng.vocab, buffers.effort), .sink = sink };
    defer bridge.decoder.deinit();
    const outcome = try runLoop(eng, tokens, limit, sampler, history, buffers.logits, buffers.candidates, buffers.generated, observer, .{ .context = &bridge, .token = Bridge.token });
    try bridge.decoder.end(outcome, sink);
    return outcome;
}

/// Shared prefill/decode loop. Prompt token IDs must fit the session; the
/// loop stops at EOS, the token budget, the context limit, or cancellation.
/// A cancelled step poisons the session; the loop resets it so the engine is
/// immediately reusable, and reports `cancelled` with the tokens completed.
///
/// `history` mirrors the session for the sampler's penalties: the loop
/// observes every prompt token once the prefill completes and every
/// generated token as it is chosen, and resets it whenever it resets the
/// session. Required when a penalty is active; a caller that never uses
/// penalties may pass null.
pub fn runLoop(
    eng: *Engine,
    tokens: []const u32,
    limit: usize,
    sampler: *inference.sampling.Sampler,
    history: ?*inference.sampling.History,
    logits: []f32,
    candidates: []inference.sampling.Candidate,
    generated: []u32,
    observer: ?Observer,
    hooks: ?Hooks,
) !Outcome {
    const io = eng.io;
    if (sampler.options.penaltiesActive() and history == null) return error.HistoryRequired;
    var timing: Timing = .{ .prompt_tokens = tokens.len };
    const gpu_before = eng.gpuSeconds();
    const prefill_start = std.Io.Clock.awake.now(io);
    // Greedy decoding on the GPU selects the token without reading back
    // logits, and eligible sampling reads back only the partial top-k;
    // callers that need logits still receive them. Penalties change the
    // argmax and the sort on the CPU, so both GPU paths are off while one
    // is active (see reference/generation.md).
    const gpu_greedy = sampler.options.temperature == 0 and !sampler.options.penaltiesActive() and eng.model.supportsGpuArgmax() and !hooks_need_logits(hooks);
    const gpu_topk = !gpu_greedy and sampler.gpuEligible() and eng.model.supportsGpuTopK() and !hooks_need_logits(hooks);
    if (gpu_topk) timing.topk_fallbacks = 0;
    var chosen: u32 = 0;
    var top: inference.sampling.TopK = .{ .temperature = sampler.options.temperature };
    const want_logits = !gpu_greedy and !gpu_topk;
    if (eng.model.chunkedPrefill(observer)) {
        // One prefill call for the whole prompt; the per-token hooks fire once.
        if (hooks) |h| if (h.before_step) |call| try call(h.context, eng.model.session().position);
        eng.model.prefill(tokens, if (want_logits) logits else null, if (gpu_greedy) &chosen else null, if (gpu_topk) &top else null, observer) catch |err| switch (err) {
            error.Cancelled => {
                resetAll(eng, history);
                timing.prefill = prefill_start.durationTo(std.Io.Clock.awake.now(io));
                return .{ .stop = .cancelled, .timing = timing };
            },
            else => return err,
        };
        if (hooks) |h| if (h.step) |call| try call(h.context, eng.model.session().position);
    } else for (tokens, 0..) |token, i| {
        if (hooks) |h| if (h.before_step) |call| try call(h.context, eng.model.session().position);
        const last = i + 1 == tokens.len;
        eng.model.step(token, if (last and want_logits) logits else null, if (last and gpu_greedy) &chosen else null, if (last and gpu_topk) &top else null, observer) catch |err| switch (err) {
            error.Cancelled => {
                resetAll(eng, history);
                timing.prefill = prefill_start.durationTo(std.Io.Clock.awake.now(io));
                return .{ .stop = .cancelled, .timing = timing };
            },
            else => return err,
        };
        if (hooks) |h| if (h.step) |call| try call(h.context, eng.model.session().position);
        // Token by token here (the CPU reference, or a GPU run with a layer
        // observer); the chunked path reports from inside the plan.
        if (observer) |o| if (o.progress) |call| try call(o.context, .{ .phase = .prefill, .position = i + 1, .target = tokens.len });
    }
    if (history) |h| for (tokens) |token| try h.observe(token);
    timing.prefill = prefill_start.durationTo(std.Io.Clock.awake.now(io));
    if (hooks) |h| if (h.prefill) |call| try call(h.context, logits);
    var decode_start = std.Io.Clock.awake.now(io);
    var count: usize = 0;
    var stop: StopReason = .token_budget;
    while (count < limit) {
        const token = if (gpu_greedy) chosen else if (gpu_topk) (try sampler.selectFrom(&top, candidates)) orelse blk: {
            // The readback could not decide exactly (nucleus beyond the
            // readback, a borderline denominator, or a non-finite logit):
            // read the full logits and take the reference path.
            timing.topk_fallbacks.? += 1;
            try eng.model.readLogits(logits);
            break :blk try sampler.select(logits, candidates, history);
        } else try sampler.select(logits, candidates, history);
        generated[count] = token;
        count += 1;
        if (history) |h| try h.observe(token);
        if (count == 1) {
            decode_start = std.Io.Clock.awake.now(io);
            timing.first_token = prefill_start.durationTo(decode_start);
        }
        if (hooks) |h| if (h.token) |call| try call(h.context, token);
        if (observer) |o| if (o.progress) |call| try call(o.context, .{ .phase = .decode, .position = count, .target = limit });
        if (eng.isStop(token)) {
            stop = .eos;
            break;
        }
        if (count == limit) break;
        if (eng.model.session().position >= eng.model.session().capacity) {
            stop = .context_limit;
            break;
        }
        if (hooks) |h| if (h.before_step) |call| try call(h.context, eng.model.session().position);
        eng.model.step(token, if (want_logits) logits else null, if (gpu_greedy) &chosen else null, if (gpu_topk) &top else null, observer) catch |err| switch (err) {
            error.Cancelled => {
                resetAll(eng, history);
                stop = .cancelled;
                break;
            },
            else => return err,
        };
        if (hooks) |h| if (h.step) |call| try call(h.context, eng.model.session().position);
    }
    timing.decode = decode_start.durationTo(std.Io.Clock.awake.now(io));
    timing.generated_tokens = count;
    if (eng.gpuSeconds()) |after| timing.gpu_seconds = after - (gpu_before orelse 0);
    return .{ .stop = stop, .timing = timing };
}

/// The session and its token history are reset together: the history is
/// meaningful only as a mirror of what the session has consumed.
fn resetAll(eng: *Engine, history: ?*inference.sampling.History) void {
    eng.model.reset();
    if (history) |h| h.reset();
}

fn hooks_need_logits(hooks: ?Hooks) bool {
    return if (hooks) |h| h.prefill != null else false;
}
