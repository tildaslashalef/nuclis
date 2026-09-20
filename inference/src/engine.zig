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

/// What draft source to load with the artifact. A load-time decision: the
/// drafter's weights and the checkpoints its recovery fills belong to the
/// memory plan (docs/spec.md § Speculative decoding).
pub const DraftRequest = union(enum) {
    none,
    /// The artifact's own embedded prediction block (the Qwen3.8 release);
    /// a family without one is `DraftSourceMissing`.
    embedded,
    /// The embedded block when the family has one, otherwise no drafter:
    /// `bench` measures what is available rather than failing.
    optional_embedded,
    /// A separate companion file, opened by MODL-19/20.
    file: []const u8,
};
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
        /// The whole batch must fit: it is admitted before the first write, so a
        /// refused batch leaves the position unchanged on both executors.
        ///
        /// `hidden`, when given (`tokens.len × hidden`), receives every row's
        /// post-`output_norm` hidden on both executors; a family with no such
        /// hidden refuses it (`error.HiddenUnsupported`).
        pub fn prefill(self: *Self, tokens: []const u32, logits: ?[]f32, greedy: ?*u32, topk: ?*inference.sampling.TopK, hidden: ?[]f32, observer: ?Observer) !void {
            if (tokens.len == 0) return error.InvalidShape;
            switch (self.*) {
                .cpu => |*runtime| {
                    if (tokens.len > runtime.state.capacity - runtime.state.position) return error.ContextFull;
                    if (hidden) |out| {
                        if (comptime @hasField(Family.Runtime, "h")) {
                            const width = runtime.h.len;
                            if (out.len != tokens.len * width) return error.InvalidShape;
                            for (tokens, 0..) |token, i| {
                                try runtime.step(token, if (i + 1 == tokens.len) logits else null, observer);
                                @memcpy(out[i * width ..][0..width], runtime.h);
                            }
                        } else return error.HiddenUnsupported;
                    } else for (tokens, 0..) |token, i| try runtime.step(token, if (i + 1 == tokens.len) logits else null, observer);
                },
                .metal => |*m| {
                    if (observer != null and observer.?.layer != null) {
                        if (hidden != null) return error.HiddenUnsupported;
                        for (tokens, 0..) |token, i| {
                            const last = i + 1 == tokens.len;
                            try m.plan.step(token, if (last) logits else null, if (last) greedy else null, if (last) topk else null, observer);
                        }
                    } else try m.plan.prefill(tokens, logits, greedy, topk, hidden, observer);
                },
            }
        }
        /// Logits for every row of a verify batch: `rows.len == tokens.len ×
        /// vocabulary`. The family's `verify` runs a chunk and the output head
        /// over all rows when it has one; otherwise the CPU runtime steps token
        /// by token. `h_rows`, when given, receives the post-`output_norm`
        /// hidden per row (`tokens.len × Family` hidden), which the drafter's
        /// `commit` consumes; a family without a specialized verify refuses it.
        pub fn verify(self: *Self, tokens: []const u32, vocabulary: usize, rows: []f32, h_rows: ?[]f32, observer: ?Observer) !void {
            switch (self.*) {
                .cpu => |*runtime| if (comptime @hasDecl(Family.Runtime, "verify")) {
                    try runtime.verify(tokens, rows, h_rows, observer);
                } else {
                    if (h_rows != null) return error.HiddenUnsupported;
                    for (tokens, 0..) |token, i| try runtime.step(token, rows[i * vocabulary ..][0..vocabulary], observer);
                },
                .metal => |*m| if (comptime @hasDecl(Family.Plan, "verify")) {
                    try m.plan.verify(tokens, rows, h_rows, observer);
                } else {
                    if (h_rows != null) return error.HiddenUnsupported;
                    for (tokens, 0..) |token, i| try m.plan.step(token, rows[i * vocabulary ..][0..vocabulary], null, null, observer);
                },
            }
        }
        /// `verify`'s greedy sibling: per-row argmax only, no logit readback
        /// where the family implements it.
        pub fn verifyGreedy(self: *Self, tokens: []const u32, vocabulary: usize, out: []u32, h_rows: ?[]f32, observer: ?Observer) !void {
            _ = vocabulary;
            switch (self.*) {
                .cpu => |*runtime| if (comptime @hasDecl(Family.Runtime, "verifyGreedy")) {
                    try runtime.verifyGreedy(tokens, out, h_rows, observer);
                } else return error.UnsupportedVerify,
                .metal => |*m| if (comptime @hasDecl(Family.Plan, "verifyGreedy")) {
                    try m.plan.verifyGreedy(tokens, out, h_rows, observer);
                } else {
                    if (h_rows != null) return error.HiddenUnsupported;
                    for (tokens, 0..) |token, i| try m.plan.step(token, null, &out[i], null, observer);
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
        /// Whether any layer's state is recurrent, which `recover` must replay
        /// rather than rewind by position.
        pub fn hasRecurrentState(self: *const Self) bool {
            return self.session().hasRecurrent();
        }
        /// The adapter's drafter when one is loaded and the backend runs it;
        /// null when the family or backend has none. The contract value's host
        /// is this live executor, so it never outlives a move.
        pub fn drafter(self: *Self) ?inference.draft.Drafter {
            return switch (self.*) {
                .cpu => |*r| if (comptime @hasDecl(@TypeOf(r.*), "drafter")) r.drafter() else null,
                .metal => |*m| if (comptime @hasDecl(@TypeOf(m.plan), "drafter")) m.plan.drafter() else null,
            };
        }
        pub fn checkpoint(self: *Self) !void {
            return self.sessionMut().checkpoint();
        }
        pub fn rewind(self: *Self) !void {
            return self.sessionMut().rewind();
        }
        pub fn truncate(self: *Self, position: usize) !void {
            return self.sessionMut().truncate(position);
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
    pub fn prefill(self: *Model, tokens: []const u32, logits: ?[]f32, greedy: ?*u32, topk: ?*inference.sampling.TopK, hidden: ?[]f32, observer: ?Observer) !void {
        switch (self.exec) {
            inline else => |*e| try e.prefill(tokens, logits, greedy, topk, hidden, observer),
        }
    }
    pub fn verify(self: *Model, tokens: []const u32, vocabulary: usize, rows: []f32, h_rows: ?[]f32, observer: ?Observer) !void {
        switch (self.exec) {
            inline else => |*e| try e.verify(tokens, vocabulary, rows, h_rows, observer),
        }
    }
    pub fn verifyGreedy(self: *Model, tokens: []const u32, vocabulary: usize, out: []u32, h_rows: ?[]f32, observer: ?Observer) !void {
        switch (self.exec) {
            inline else => |*e| try e.verifyGreedy(tokens, vocabulary, out, h_rows, observer),
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
    /// Records the committed recurrent state so a verify batch can be undone;
    /// see `recover`. Valid between steps and on both executors, as `snapshot`
    /// is. Nothing reads the region until `rewind`.
    pub fn checkpoint(self: *Model) !void {
        switch (self.exec) {
            inline else => |*e| try e.checkpoint(),
        }
    }
    pub fn rewind(self: *Model) !void {
        switch (self.exec) {
            inline else => |*e| try e.rewind(),
        }
    }
    pub fn truncate(self: *Model, position: usize) !void {
        switch (self.exec) {
            inline else => |*e| try e.truncate(position),
        }
    }
    /// Whether the model's state cannot be rewound by position alone.
    pub fn hasRecurrentState(self: *const Model) bool {
        return self.session().hasRecurrent();
    }
    /// The loaded drafter, or null. `propose`/`commitDraft` are the loop's
    /// only calls: propose chained candidates, commit advances the drafter
    /// over the accepted prefix after `recover`. The drafter's cache is one
    /// more layout in the session, so `checkpoint`/`rewind` cover it.
    pub fn drafter(self: *Model) ?inference.draft.Drafter {
        switch (self.exec) {
            inline else => |*e| return e.drafter(),
        }
    }
    pub fn propose(self: *Model, token: u32, out: []u32) !usize {
        const d = self.drafter() orelse return error.NoDrafter;
        return d.propose(token, out);
    }
    pub fn commitDraft(self: *Model, tokens: []const u32, h_rows: []const f32) !void {
        const d = self.drafter() orelse return error.NoDrafter;
        try d.commit(tokens, h_rows);
    }
    /// How one `recover` call spent its time: the checkpoint copy and the
    /// forward over the accepted prefix.
    pub const Recovery = struct {
        rewind: std.Io.Duration = .zero,
        replay: std.Io.Duration = .zero,
    };
    /// Returns the session to the state after the accepted prefix of a
    /// speculative verify batch. `accepted` is the tokens the main model
    /// committed, starting with the token fed before the batch; the position
    /// ends at the checkpoint plus `accepted.len`. Recurrent state is replayed
    /// (rewind then a forward), attention alone is truncated: rows past the
    /// position are ignored by contract. Returns the copy and replay times.
    /// A missing checkpoint is `NoCheckpoint`; a batch larger than the context
    /// is `ContextFull` for recurrent state (nothing is written past the
    /// capacity).
    pub fn recover(self: *Model, io: std.Io, accepted: []const u32) !Recovery {
        const at = self.session().checkpoint_position orelse return error.NoCheckpoint;
        var stats: Recovery = .{};
        if (self.hasRecurrentState()) {
            // The verify batch fed exactly the accepted prefix when every
            // draft was accepted (accepted is the whole batch): the recurrent
            // state is already a function of those tokens, so nothing is
            // rewound or replayed.
            if (accepted.len == self.session().position - at) return stats;
            // A batch that kept row checkpoints restores the accepted row
            // directly: one copy, no replay. The CPU reference and a batch
            // past the region replay below.
            if (accepted.len > 0 and self.session().row_checkpoints > 0 and accepted.len <= self.session().row_checkpoint_rows) {
                const restore_start = std.Io.Clock.awake.now(io);
                try self.sessionMut().restoreRow(accepted.len - 1);
                stats.rewind = restore_start.durationTo(std.Io.Clock.awake.now(io));
                return stats;
            }
            const rewind_start = std.Io.Clock.awake.now(io);
            try self.rewind();
            stats.rewind = rewind_start.durationTo(std.Io.Clock.awake.now(io));
            if (accepted.len > 0) {
                const replay_start = std.Io.Clock.awake.now(io);
                // A single-token replay takes the per-token matvec path; a
                // chunked prefill for one row reaches the small-batch tile,
                // which measured 2.3x slower on the prose replay.
                if (accepted.len == 1) {
                    try self.step(accepted[0], null, null, null, null);
                } else {
                    try self.prefill(accepted, null, null, null, null, null);
                }
                stats.replay = replay_start.durationTo(std.Io.Clock.awake.now(io));
            }
        } else {
            try self.truncate(at + accepted.len);
        }
        return stats;
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

/// The host bound on a draft block: KERN-11's 8-row token tile less the seed
/// row. A requested length above it is refused; the switch and the length are
/// the only speculative knobs exposed (docs/spec.md § Speculative decoding).
pub const max_draft_length = 7;

/// Runtime speculative settings, resolved by the caller from the
/// configuration file and flags. `enabled` alone does nothing without a
/// loaded drafter.
pub const Speculative = struct {
    enabled: bool = false,
    draft_length: usize = 0,
};

/// Engine-owned scratch for the speculative step, sized once when a drafter is
/// loaded: the proposed drafts, the target rows and hidden of a verify batch,
/// and the candidate buffer `distribution` writes. `runLoop` borrows it; it is
/// freed with the engine.
const SpeculativeScratch = struct {
    drafts: []u32,
    rows: []f32,
    hidden: []f32,
    tokens: []u32,
    choices: []u32,
    p: []inference.sampling.Candidate,

    fn init(alloc: std.mem.Allocator, vocabulary: usize, hidden_width: usize) !SpeculativeScratch {
        const rows = max_draft_length + 1;
        const drafts = try alloc.alloc(u32, max_draft_length);
        errdefer alloc.free(drafts);
        const logits = try alloc.alloc(f32, rows * vocabulary);
        errdefer alloc.free(logits);
        // The prompt commit harvests a full prefill chunk's hidden rows at
        // once; the verify batch uses only the first `rows` of it.
        const hidden = try alloc.alloc(f32, prefill_chunk * hidden_width);
        errdefer alloc.free(hidden);
        const tokens = try alloc.alloc(u32, rows);
        errdefer alloc.free(tokens);
        const choices = try alloc.alloc(u32, rows);
        errdefer alloc.free(choices);
        const p = try alloc.alloc(inference.sampling.Candidate, vocabulary);
        errdefer alloc.free(p);
        return .{ .drafts = drafts, .rows = logits, .hidden = hidden, .tokens = tokens, .choices = choices, .p = p };
    }
    fn deinit(self: *SpeculativeScratch, alloc: std.mem.Allocator) void {
        alloc.free(self.drafts);
        alloc.free(self.rows);
        alloc.free(self.hidden);
        alloc.free(self.tokens);
        alloc.free(self.choices);
        alloc.free(self.p);
        self.* = undefined;
    }
};

/// Prompt tokens per prefill command buffer on the GPU plan. Bounds the
/// chunk activation buffers (~0.4 MB per token) and the work between
/// cancellation checks; the plan clamps it to the session capacity. A
/// family whose plan declares `preferredChunk` chooses per binding (the
/// expert configuration fills its gathered tiles better with longer chunks).
pub const prefill_chunk = 256;
fn chunkFor(comptime Family: type, binding: Family.Binding) usize {
    return if (@hasDecl(Family.Plan, "preferredChunk")) Family.Plan.preferredChunk(binding, prefill_chunk) else prefill_chunk;
}

/// Builds one family's executor for the backend. Heap-allocates the Metal
/// backend so the plan's pointer stays valid when the Engine value is
/// returned by value; its diagnostic text is logged on failure.
fn openExecutor(comptime Family: type, alloc: std.mem.Allocator, view: inference.weights.View, binding: Family.Binding, backend: Backend, capacity: usize, kv: KvPrecision, draft: DraftRequest) !Executor(Family) {
    // A checkpoint region is sized only when a caller needs to undo a verify
    // batch, which is exactly when a drafter is loaded.
    const want_draft = switch (draft) {
        .none => false,
        .embedded, .optional_embedded => true,
        .file => return error.DraftSourceUnsupported,
    };
    return switch (backend) {
        .cpu => .{ .cpu = try Family.Runtime.init(alloc, view, binding, capacity, want_draft, want_draft) },
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
            const plan = try Family.Plan.init(alloc, gpu, view, binding, capacity, @min(chunkFor(Family, binding), capacity), kv, want_draft, want_draft);
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
    /// prompts still run). A caller may force one instead (`open`'s
    /// `forced`), which `profile_forced` records.
    profile: ?profiles.Profile,
    /// The profile was the caller's choice, not the template digest's: the
    /// file's own template is not the pinned one, so what the profile renders
    /// is the pinned protocol, not necessarily what this file's template says.
    profile_forced: bool,
    /// The ids that end generation: the profile's stop tokens resolved in
    /// this vocabulary, or the file's `eos_token_id` alone without a profile.
    stop_ids: [max_stop_tokens]u32,
    stop_count: usize,
    /// The attention cache precision the session was built with: the
    /// request on the GPU plan, always `f32` on the CPU reference.
    kv_precision: KvPrecision,
    /// Speculative-step scratch, allocated exactly when a drafter is loaded.
    spec: ?SpeculativeScratch,
    /// `general.name` from the artifact metadata, borrowed from the mapping.
    name: []const u8,
    /// Wall time spent in `open`, including directory parsing and mapping.
    load: std.Io.Duration,

    /// Opens the artifact, selects its adapter and binds it, loads the
    /// vocabulary, and prepares a session of `capacity` tokens with an
    /// attention cache of `kv` precision (GPU only; the CPU reference stays
    /// F32). An architecture without an adapter is `UnknownArchitecture`;
    /// `models.known` names the ones the tree has. `forced` selects the
    /// prompt profile regardless of the file's template digest.
    pub fn open(alloc: std.mem.Allocator, io: std.Io, model_path: []const u8, backend: Backend, capacity: usize, kv: KvPrecision, forced: ?profiles.Profile, draft: DraftRequest) !Engine {
        const started = std.Io.Clock.awake.now(io);
        if (capacity == 0 or capacity > 32768) return error.InvalidGenerationBudget;
        var mapped = try inference.weights.Mapped.open(alloc, io, model_path);
        errdefer mapped.deinit(io);
        const adapter = try models.select(mapped.document.string("general.architecture") orelse "");
        var vocab = try inference.vocabulary.load(alloc, mapped.document, mapped.mapping.memory[0..@intCast(mapped.document.directory_bytes)], .{});
        errdefer vocab.deinit();
        const detected = profiles.forDocument(mapped.document);
        const profile = forced orelse detected;
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
                break :blk .{ .exec = @unionInit(Executors, @tagName(a), try openExecutor(Family, alloc, mapped.view(), binding, backend, capacity, kv, draft)) };
            },
        };
        errdefer model.deinit(alloc);
        // A required embedded block that the family does not bind is a typed
        // load error, never a silent non-speculative run.
        if (std.meta.activeTag(draft) == .embedded and model.drafter() == null) return error.DraftSourceMissing;
        // The verify scratch is part of the load plan: it exists exactly when a
        // drafter was requested, so a plain run pays nothing.
        var spec: ?SpeculativeScratch = null;
        errdefer if (spec) |*s| s.deinit(alloc);
        if (model.drafter()) |drafter| spec = try SpeculativeScratch.init(alloc, vocab.tokens.len, drafter.hidden);
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
            .profile_forced = forced != null and forced != detected,
            .stop_ids = stop_ids,
            .stop_count = stop_count,
            .kv_precision = if (backend == .cpu) .f32 else kv,
            .spec = spec,
            .name = mapped.document.string("general.name") orelse "unnamed model",
            .load = started.durationTo(std.Io.Clock.awake.now(io)),
        };
    }

    pub fn deinit(self: *Engine) void {
        self.model.deinit(self.alloc);
        if (self.spec) |*s| s.deinit(self.alloc);
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

    /// The system block every rendering of these leading messages and tools
    /// starts with (`profiles.Profile.prefix`). Caller owns the result.
    pub fn prefix(self: *const Engine, messages: []const profiles.Message, tools: []const profiles.ToolDefinition, effort: profiles.Effort) ![]u8 {
        const profile = self.profile orelse return error.UnsupportedPromptTemplate;
        return profile.prefix(self.alloc, messages, tools, effort, .{});
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

/// One accepted length's recovery calls in a `Timing`: the accepted prefix
/// length is 1 (the seed alone) through `max_draft_length + 1` (every draft).
/// Full acceptance is counted too, with a zero replay.
pub const RecoverCall = struct {
    calls: usize = 0,
    rewind: std.Io.Duration = .zero,
    replay: std.Io.Duration = .zero,
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
    /// Speculative decoding: verify batches run, drafts accepted and
    /// proposed across them, and time spent in the model's `propose`,
    /// `verify`, the acceptance decision on the host (`accept`: the shaped
    /// distributions and draws of the sampled path), `recover`, and the
    /// drafter `commit` of the accepted prefix.
    speculative_steps: usize = 0,
    accepted_drafts: usize = 0,
    proposed_drafts: usize = 0,
    propose: std.Io.Duration = .zero,
    verify: std.Io.Duration = .zero,
    accept: std.Io.Duration = .zero,
    /// Every `recover` call, including the final one to the emitted count when
    /// a batch crosses the token budget; `recover_rewind` and `recover_replay`
    /// split it and `recover_by_length` attributes it.
    recover: std.Io.Duration = .zero,
    recover_rewind: std.Io.Duration = .zero,
    recover_replay: std.Io.Duration = .zero,
    recover_by_length: [max_draft_length + 2]RecoverCall = @splat(.{}),
    /// The verify batch's checkpoint copy, timed in `speculativeBatch`; no
    /// other field covers it.
    checkpoint: std.Io.Duration = .zero,
    commit: std.Io.Duration = .zero,
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
    settings: Speculative,
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
    const outcome = try runLoop(eng, tokens, limit, sampler, history, settings, buffers.logits, buffers.candidates, buffers.generated, observer, .{ .context = &bridge, .token = Bridge.token });
    try bridge.decoder.end(outcome, sink);
    return outcome;
}

/// Commits a prompt to the loaded drafter: the target consumes it in
/// `prefill_chunk` chunks and the block's cache is filled from the target
/// hidden of every committed position, so the drafter can `propose` from
/// after the last token. `logits`, when given, receives the last chunk's
/// last-token logits. Requires a loaded drafter and its scratch.
pub fn commitPrompt(eng: *Engine, tokens: []const u32, logits: ?[]f32, observer: ?Observer) !void {
    const drafter = eng.model.drafter() orelse return error.NoDrafter;
    const s = &(eng.spec orelse return error.NoSpeculativeScratch);
    // `prefill` reports progress per internal chunk; strip it here so the
    // caller sees one cumulative position for the whole prompt.
    const inner: ?Observer = if (observer) |o| .{ .context = o.context, .check = o.check, .layer = o.layer } else null;
    var offset: usize = 0;
    while (offset < tokens.len) {
        const count = @min(tokens.len - offset, prefill_chunk);
        const chunk = tokens[offset..][0..count];
        const hidden = s.hidden[0 .. count * drafter.hidden];
        try eng.model.prefill(chunk, if (offset + count == tokens.len) logits else null, null, null, hidden, inner);
        try eng.model.commitDraft(chunk, hidden);
        offset += count;
        if (observer) |o| if (o.progress) |call| try call(o.context, .{ .phase = .prefill, .position = offset, .target = tokens.len });
    }
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
    settings: Speculative,
    logits: []f32,
    candidates: []inference.sampling.Candidate,
    generated: []u32,
    observer: ?Observer,
    hooks: ?Hooks,
) !Outcome {
    const io = eng.io;
    if (sampler.options.penaltiesActive() and history == null) return error.HistoryRequired;
    if (settings.draft_length > max_draft_length) return error.InvalidDraftLength;
    const drafter = eng.model.drafter();
    // Speculation batches the step, so the per-token GPU greedy/top-k
    // shortcuts are off while it runs and the loop materializes every seed's
    // logits. A per-layer observer is a per-token contract and disables it.
    const can_speculate = settings.enabled and drafter != null and (observer == null or observer.?.layer == null);
    // Greedy acceptance compares the target's argmax without penalties, so a
    // penalized greedy run takes the sampled path (its point masses).
    const greedy_verify = sampler.options.temperature == 0 and !sampler.options.penaltiesActive();
    var timing: Timing = .{ .prompt_tokens = tokens.len };
    const gpu_before = eng.gpuSeconds();
    const prefill_start = std.Io.Clock.awake.now(io);
    // Greedy decoding on the GPU selects the token without reading back
    // logits, and eligible sampling reads back only the partial top-k;
    // callers that need logits still receive them. Penalties change the
    // argmax and the sort on the CPU, so both GPU paths are off while one
    // is active (see reference/generation.md).
    const gpu_greedy = !can_speculate and sampler.options.temperature == 0 and !sampler.options.penaltiesActive() and eng.model.supportsGpuArgmax() and !hooks_need_logits(hooks);
    const gpu_topk = !can_speculate and !gpu_greedy and sampler.gpuEligible() and eng.model.supportsGpuTopK() and !hooks_need_logits(hooks);
    if (gpu_topk) timing.topk_fallbacks = 0;
    var chosen: u32 = 0;
    var top: inference.sampling.TopK = .{ .temperature = sampler.options.temperature };
    const want_logits = !gpu_greedy and !gpu_topk;
    const vocabulary = logits.len;
    const spec: ?*SpeculativeScratch = if (can_speculate) &(eng.spec orelse return error.NoSpeculativeScratch) else null;
    if (spec != null) {
        // Speculation commits the prompt to the drafter too: the block's cache
        // is filled from the target hidden of every committed position, not
        // just the accepted drafts. The last chunk's logits seed decoding.
        if (hooks) |h| if (h.before_step) |call| try call(h.context, eng.model.session().position);
        commitPrompt(eng, tokens, if (want_logits) logits else null, observer) catch |err| switch (err) {
            error.Cancelled => {
                resetAll(eng, history);
                timing.prefill = prefill_start.durationTo(std.Io.Clock.awake.now(io));
                return .{ .stop = .cancelled, .timing = timing };
            },
            else => return err,
        };
        if (hooks) |h| if (h.step) |call| try call(h.context, eng.model.session().position);
    } else if (eng.model.chunkedPrefill(observer)) {
        // One prefill call for the whole prompt; the per-token hooks fire once.
        if (hooks) |h| if (h.before_step) |call| try call(h.context, eng.model.session().position);
        eng.model.prefill(tokens, if (want_logits) logits else null, if (gpu_greedy) &chosen else null, if (gpu_topk) &top else null, null, observer) catch |err| switch (err) {
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
    // A batch's correction: already chosen and emitted, it seeds the next
    // batch without being sampled or emitted again.
    var seed: ?u32 = null;
    while (count < limit) {
        var token: u32 = undefined;
        if (seed) |carried| {
            token = carried;
            seed = null;
        } else {
            token = if (gpu_greedy) chosen else if (gpu_topk) (try sampler.selectFrom(&top, candidates)) orelse blk: {
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
        }
        const position = eng.model.session().position;
        if (spec) |s| if (position + 1 < eng.model.session().capacity) {
            const room = eng.model.session().capacity - position - 1;
            const k = @min(settings.draft_length, @min(room, limit - count));
            if (hooks) |h| if (h.before_step) |call| try call(h.context, position);
            const result = speculativeBatch(eng, sampler, history, observer, s, token, k, greedy_verify, vocabulary, drafter.?) catch |err| switch (err) {
                // A cancelled batch poisons the session exactly as a cancelled
                // step does; the loop resets it and reports cancellation.
                error.Cancelled => {
                    resetAll(eng, history);
                    stop = .cancelled;
                    break;
                },
                else => return err,
            };
            if (hooks) |h| if (h.step) |call| try call(h.context, eng.model.session().position);
            timing.speculative_steps += 1;
            timing.accepted_drafts += result.accepted;
            timing.proposed_drafts += result.proposed;
            timing.propose = .{ .nanoseconds = timing.propose.nanoseconds + result.propose.nanoseconds };
            timing.verify = .{ .nanoseconds = timing.verify.nanoseconds + result.verify.nanoseconds };
            timing.accept = .{ .nanoseconds = timing.accept.nanoseconds + result.accept.nanoseconds };
            timing.checkpoint = .{ .nanoseconds = timing.checkpoint.nanoseconds + result.checkpoint.nanoseconds };
            recordRecovery(&timing, result.accepted + 1, result.recover, result.recover_split);
            timing.commit = .{ .nanoseconds = timing.commit.nanoseconds + result.commit.nanoseconds };
            // The accepted drafts and the correction go through the ordinary
            // per-token checks in order; the correction is the next batch's
            // seed, so it is not sampled again.
            @memcpy(s.choices[0..result.accepted], s.drafts[0..result.accepted]);
            s.choices[result.accepted] = result.correction;
            const extras = s.choices[0 .. result.accepted + 1];
            var kept: usize = extras.len;
            var broke = false;
            for (extras, 0..) |extra, i| {
                generated[count] = extra;
                count += 1;
                if (history) |h| try h.observe(extra);
                if (hooks) |h| if (h.token) |call| try call(h.context, extra);
                if (observer) |o| if (o.progress) |call| try call(o.context, .{ .phase = .decode, .position = count, .target = limit });
                if (eng.isStop(extra)) {
                    stop = .eos;
                    kept = i;
                    broke = true;
                    break;
                }
                if (count == limit) {
                    stop = .token_budget;
                    kept = i;
                    broke = true;
                    break;
                }
            }
            if (broke) {
                // Discard the accepted drafts past the stop/budget: recover to
                // the tokens actually emitted.
                if (kept < result.accepted) {
                    const recover_start = std.Io.Clock.awake.now(eng.io);
                    const split = try eng.model.recover(eng.io, s.tokens[0 .. 1 + kept]);
                    recordRecovery(&timing, kept + 1, recover_start.durationTo(std.Io.Clock.awake.now(eng.io)), split);
                }
                break;
            }
            if (eng.model.session().position >= eng.model.session().capacity) {
                stop = .context_limit;
                break;
            }
            seed = result.correction;
            continue;
        };
        if (hooks) |h| if (h.before_step) |call| try call(h.context, position);
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

/// The accepted length, the model's next token, and the timings of one verify
/// batch: `recover_split` carries the copy/replay split of `recover`.
const BatchResult = struct { accepted: usize, correction: u32, proposed: usize, propose: std.Io.Duration, verify: std.Io.Duration, accept: std.Io.Duration, recover: std.Io.Duration, recover_split: Model.Recovery, checkpoint: std.Io.Duration, commit: std.Io.Duration };

/// Proposes `k` drafts from `seed_token`, checkpoints, verifies `[seed] ++
/// drafts` on the main model, accepts the longest prefix (greedy: the row's
/// argmax equals the draft; sampled: the row's own draw equals the draft),
/// recovers the session to the accepted prefix, and advances the drafter over
/// it. The caller emits the accepted drafts and the correction. A `k` of 0
/// still runs the single-token batch so the drafter's cache stays aligned
/// with the committed token.
fn speculativeBatch(
    eng: *Engine,
    sampler: *inference.sampling.Sampler,
    history: ?*inference.sampling.History,
    observer: ?Observer,
    s: *SpeculativeScratch,
    seed_token: u32,
    k: usize,
    greedy: bool,
    vocabulary: usize,
    drafter: inference.draft.Drafter,
) !BatchResult {
    const propose_start = std.Io.Clock.awake.now(eng.io);
    const n = try eng.model.propose(seed_token, s.drafts[0..k]);
    const propose = propose_start.durationTo(std.Io.Clock.awake.now(eng.io));
    const checkpoint_start = std.Io.Clock.awake.now(eng.io);
    try eng.model.checkpoint();
    const checkpoint = checkpoint_start.durationTo(std.Io.Clock.awake.now(eng.io));
    s.tokens[0] = seed_token;
    @memcpy(s.tokens[1 .. 1 + n], s.drafts[0..n]);
    const batch = s.tokens[0 .. 1 + n];
    const hidden = s.hidden[0 .. (1 + n) * drafter.hidden];
    var result: BatchResult = .{ .accepted = 0, .correction = 0, .proposed = n, .propose = propose, .verify = .zero, .accept = .zero, .recover = .zero, .recover_split = .{}, .checkpoint = checkpoint, .commit = .zero };
    const verify_start = std.Io.Clock.awake.now(eng.io);
    if (greedy) {
        try eng.model.verifyGreedy(batch, vocabulary, s.choices[0 .. 1 + n], hidden, observer);
        const accept_start = std.Io.Clock.awake.now(eng.io);
        result.verify = verify_start.durationTo(accept_start);
        while (result.accepted < n and s.choices[result.accepted] == s.drafts[result.accepted]) result.accepted += 1;
        result.correction = s.choices[result.accepted];
        result.accept = accept_start.durationTo(std.Io.Clock.awake.now(eng.io));
    } else {
        try eng.model.verify(batch, vocabulary, s.rows[0 .. (1 + n) * vocabulary], hidden, observer);
        const accept_start = std.Io.Clock.awake.now(eng.io);
        result.verify = verify_start.durationTo(accept_start);
        // Row `i`'s shaped distribution sees the accepted drafts before it
        // through the history; its own draw is the draft when they agree and
        // the correction otherwise, so every emitted token is a target draw.
        var rejected = false;
        while (result.accepted < n) {
            const i = result.accepted;
            const p = try sampler.distribution(s.rows[i * vocabulary ..][0..vocabulary], s.p, history);
            switch (inference.speculative.decide(sampler, p, s.drafts[i])) {
                .accepted => {
                    if (history) |h| try h.observe(s.drafts[i]);
                    result.accepted += 1;
                },
                .correction => |token| {
                    result.correction = token;
                    rejected = true;
                    break;
                },
            }
        }
        if (!rejected) {
            const p = try sampler.distribution(s.rows[n * vocabulary ..][0..vocabulary], s.p, history);
            result.correction = inference.speculative.draw(sampler, p);
        }
        result.accept = accept_start.durationTo(std.Io.Clock.awake.now(eng.io));
    }
    const recover_start = std.Io.Clock.awake.now(eng.io);
    result.recover_split = try eng.model.recover(eng.io, batch[0 .. 1 + result.accepted]);
    result.recover = recover_start.durationTo(std.Io.Clock.awake.now(eng.io));
    const commit_start = std.Io.Clock.awake.now(eng.io);
    try eng.model.commitDraft(batch[0 .. 1 + result.accepted], hidden[0 .. (1 + result.accepted) * drafter.hidden]);
    result.commit = commit_start.durationTo(std.Io.Clock.awake.now(eng.io));
    return result;
}

/// Attributes one `recover` call to the run's timing: the aggregate, the
/// copy/replay split, and the accepted length's cell.
fn recordRecovery(timing: *Timing, accepted_len: usize, total: std.Io.Duration, split: Model.Recovery) void {
    std.debug.assert(accepted_len > 0 and accepted_len < timing.recover_by_length.len);
    timing.recover = .{ .nanoseconds = timing.recover.nanoseconds + total.nanoseconds };
    timing.recover_rewind = .{ .nanoseconds = timing.recover_rewind.nanoseconds + split.rewind.nanoseconds };
    timing.recover_replay = .{ .nanoseconds = timing.recover_replay.nanoseconds + split.replay.nanoseconds };
    const cell = &timing.recover_by_length[accepted_len];
    cell.calls += 1;
    cell.rewind = .{ .nanoseconds = cell.rewind.nanoseconds + split.rewind.nanoseconds };
    cell.replay = .{ .nanoseconds = cell.replay.nanoseconds + split.replay.nanoseconds };
}

/// The session and its token history are reset together: the history is
/// meaningful only as a mirror of what the session has consumed.
fn resetAll(eng: *Engine, history: ?*inference.sampling.History) void {
    eng.model.reset();
    if (eng.model.drafter()) |d| d.reset();
    if (history) |h| h.reset();
}

fn hooks_need_logits(hooks: ?Hooks) bool {
    return if (hooks) |h| h.prefill != null else false;
}
