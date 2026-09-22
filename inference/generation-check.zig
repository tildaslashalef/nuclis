//! Opt-in full-model reset/isolation/failure check. On the CPU this is
//! intentionally slow reference work; `--metal` runs the same protocol on the
//! GPU-resident plan, where results must also be bit-identical run to run.
//! The family is selected from the file's `general.architecture`: the
//! protocol is the same for every registered adapter, only the pinned token
//! ids and the recorded tolerances differ.
const std = @import("std");
const inference = @import("inference");
const Observer = inference.observer.Observer;

/// What the protocol needs from a family: its adapter types, two pinned
/// tokens (a word, then a comma: the second exercises a nonzero position),
/// and the tolerances of the chunked-prefill and F16-cache comparisons
/// against the stepped F32 logits, recorded per family (metal-backend.md).
const Bounds = struct {
    /// Chunked prefill (half matmul tiles) vs per-token F32 steps.
    chunk_max_abs: f64,
    chunk_rel_rms: f64,
    /// F16 cache vs the F32 cache, stepped and chunked.
    half_max_abs: f64,
    half_rel_rms: f64,
};
/// The prediction block's pinned trace: the tokens of `Hello,` and the
/// reference's greedy draft per position, against the embedded hidden rows.
const Draft = struct {
    tokens: [2]u32,
    greedy: [2]u32,
};
const Spec = struct {
    Family: type,
    vocabulary: usize,
    tokens: [2]u32,
    bounds: Bounds,
    /// The bounds of a mixture-of-experts configuration of the family, when
    /// it has one: discrete routing amplifies the tiles' rounding.
    expert_bounds: ?Bounds = null,
    /// The prediction block's trace to check, when the family has one.
    draft: ?Draft = null,
};
const qwen35_spec: Spec = .{ .Family = inference.models.qwen35.family, .vocabulary = inference.models.qwen35_metal.vocabulary, .tokens = .{ 9419, 11 }, .bounds = .{ .chunk_max_abs = 2e-2, .chunk_rel_rms = 1e-3, .half_max_abs = 2e-2, .half_rel_rms = 1e-3 }, .draft = .{ .tokens = .{ 9419, 11 }, .greedy = .{ 9419, 271 } } };
// Gemma's F16 tolerance is the model's own sensitivity to rounding keys
// (unscaled attention scores; gemma4.md), not the kernels': the same
// kernels are within 2e-4 of the CPU over the rounded operands (test-metal).
// The chunk tolerance covers both pinned 12B files' half-operand tile
// rounding: the K-quant file at 8.6e-2 / 2.5e-3, the QAT file at
// 4.4e-1 / 1.4e-2. On the 26B-A4B the same rounding moves router logits
// past near-ties and swaps whole experts for a token, so its chunked
// logits sit at 1.1e1 / 3.5e-1 (gemma4.md § 26B-A4B). The F32-tile
// comparison below, at its own unchanged bound, is what proves the schedule.
const gemma4_spec: Spec = .{
    .Family = inference.models.gemma4.family,
    .vocabulary = inference.models.gemma4.vocabulary,
    .tokens = .{ 9259, 236764 },
    .bounds = .{ .chunk_max_abs = 6e-1, .chunk_rel_rms = 2e-2, .half_max_abs = 2.0, .half_rel_rms = 6e-2 },
    .expert_bounds = .{ .chunk_max_abs = 2e1, .chunk_rel_rms = 5e-1, .half_max_abs = 2e1, .half_rel_rms = 5e-1 },
};
// Muse Glimmer: `Hello` then `,` (the pinned trace's tokens). The half
// tiles' rounding measured 1.4e-2 / 4.5e-3 on the 70-token prefill and
// the F16 cache 5.0e-3 / 6.0e-4 (muse-glimmer.md § Metal plan); the
// F32-tile comparison at its unchanged bound proves the schedule.
const muse_glimmer_spec: Spec = .{
    .Family = inference.models.muse_glimmer.family,
    .vocabulary = inference.models.muse_glimmer.vocabulary,
    .tokens = .{ 19873, 24 },
    .bounds = .{ .chunk_max_abs = 5e-2, .chunk_rel_rms = 1e-2, .half_max_abs = 2e-2, .half_rel_rms = 2e-3 },
};
/// The bounds of the bound configuration: the expert ones when the family
/// declares them and the file is an expert configuration.
fn boundsOf(comptime spec: Spec, binding: spec.Family.Binding) Bounds {
    if (spec.expert_bounds) |expert| {
        if (binding.config.experts != null) return expert;
    }
    return spec.bounds;
}

/// Trace-style cancellation: after layer 3's values are available (the GPU
/// plan has committed that layer).
fn cancel(_: *anyopaque, layer: usize, _: []const f32) !void {
    if (layer == 3) return error.Cancelled;
}
/// Production-style cancellation through `check`: the fourth boundary, while
/// the GPU plan is still recording, so partially recorded work is what runs.
fn cancelCheck(context: *anyopaque) !void {
    const boundaries: *u8 = @ptrCast(context);
    boundaries.* += 1;
    if (boundaries.* == 4) return error.Cancelled;
}

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var use_metal = false;
    var draft_stats = false;
    var speculative_check = false;
    var draft_trace: ?[]const u8 = null;
    var draft_model: ?[]const u8 = null;
    var vision_check: ?[]const u8 = null;
    var path: ?[]const u8 = null;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--metal")) {
            use_metal = true;
        } else if (std.mem.eql(u8, arg, "--draft-stats")) {
            draft_stats = true;
        } else if (std.mem.eql(u8, arg, "--speculative-check")) {
            speculative_check = true;
        } else if (std.mem.eql(u8, arg, "--draft-trace")) {
            i += 1;
            if (i >= args.len) return error.ExpectedTraceDirectory;
            draft_trace = args[i];
        } else if (std.mem.eql(u8, arg, "--draft-model")) {
            i += 1;
            if (i >= args.len) return error.ExpectedDraftModel;
            draft_model = args[i];
        } else if (std.mem.eql(u8, arg, "--vision-check")) {
            i += 1;
            if (i >= args.len) return error.ExpectedProjectorPath;
            vision_check = args[i];
        } else if (path == null) {
            path = arg;
        } else return error.UnknownOption;
    }
    const model_path = path orelse return error.ExpectedModelPath;
    var mapped = try inference.weights.Mapped.open(alloc, init.io, model_path);
    defer mapped.deinit(init.io);
    const architecture = mapped.document.string("general.architecture") orelse return error.MissingMetadata;
    switch (try inference.models.select(architecture)) {
        .qwen35 => if (vision_check) |projector|
            try visionCheck(alloc, init.io, model_path, projector, use_metal)
        else if (draft_trace) |dir|
            try draftTrace(qwen35_spec, alloc, init.io, &mapped, use_metal, dir)
        else if (draft_stats)
            try draftStats(qwen35_spec, alloc, init.io, &mapped, use_metal)
        else if (speculative_check)
            try speculativeCheck(qwen35_spec, alloc, init.io, &mapped, model_path, use_metal)
        else
            try run(qwen35_spec, alloc, init.io, &mapped, use_metal),
        .gemma4 => if (draft_trace != null)
            try gemmaDraftTrace(alloc, init.io, model_path, draft_model orelse return error.ExpectedDraftModel, use_metal)
        else if (draft_stats or speculative_check) return error.DraftStatsUnsupported else try run(gemma4_spec, alloc, init.io, &mapped, use_metal),
        .@"muse-glimmer" => if (draft_trace) |dir|
            try museDraftTrace(alloc, init.io, model_path, draft_model orelse return error.ExpectedDraftModel, use_metal, dir)
        else if (draft_stats or speculative_check) return error.DraftStatsUnsupported else try run(muse_glimmer_spec, alloc, init.io, &mapped, use_metal),
    }
}

fn run(comptime spec: Spec, alloc: std.mem.Allocator, io: std.Io, mapped: *inference.weights.Mapped, use_metal: bool) !void {
    const Family = spec.Family;
    const Runtime = Family.Runtime;
    const Plan = Family.Plan;
    // Either executor behind one step/reset surface for this check.
    const Model = union(enum) {
        cpu: Runtime,
        metal: Plan,
        fn step(self: *@This(), token: u32, logits: ?[]f32, observer: ?Observer) !void {
            switch (self.*) {
                .cpu => |*r| try r.step(token, logits, observer),
                .metal => |*p| try p.step(token, logits, null, null, null, observer),
            }
        }
        /// The whole batch is admitted before the first write, as the plans
        /// do; the CPU reference steps token by token.
        fn prefill(self: *@This(), tokens: []const u32, logits: ?[]f32, observer: ?Observer) !void {
            switch (self.*) {
                .cpu => |*r| {
                    if (tokens.len > r.state.capacity - r.state.position) return error.ContextFull;
                    for (tokens, 0..) |token, i| try r.step(token, if (i + 1 == tokens.len) logits else null, observer);
                },
                .metal => |*p| try p.prefill(tokens, logits, null, null, null, null, observer),
            }
        }
        /// Consumes one batch. The Metal plan with a drafter runs its greedy
        /// verify path, which also fills the row checkpoints; everything else
        /// steps token by token or prefills.
        fn batch(self: *@This(), tokens: []const u32) !void {
            switch (self.*) {
                .cpu => |*r| for (tokens) |token| try r.step(token, null, null),
                .metal => |*p| {
                    if (@hasField(@TypeOf(p.*), "has_draft")) {
                        if (p.has_draft and p.state.row_checkpoints > 0) {
                            var choices: [16]u32 = undefined;
                            try p.verifyGreedy(tokens, choices[0..tokens.len], null, null);
                            return;
                        }
                    }
                    try self.prefill(tokens, null, null);
                },
            }
        }
        fn reset(self: *@This()) void {
            switch (self.*) {
                .cpu => |*r| r.reset(),
                .metal => |*p| p.reset(),
            }
        }
        fn deinit(self: *@This()) void {
            switch (self.*) {
                .cpu => |*r| r.deinit(),
                .metal => |*p| p.deinit(),
            }
        }
        fn state(self: *@This()) *inference.session.Session {
            return switch (self.*) {
                .cpu => |*r| &r.state,
                .metal => |*p| &p.state,
            };
        }
        fn checkpoint(self: *@This()) !void {
            return self.state().checkpoint();
        }
        fn rewind(self: *@This()) !void {
            return self.state().rewind();
        }
        fn truncate(self: *@This(), position: usize) !void {
            return self.state().truncate(position);
        }
        /// The accepted-prefix operation, mirroring `engine.Model.recover`:
        /// a batch that kept row checkpoints restores the accepted row,
        /// recurrent state without them is replayed, attention alone is
        /// truncated.
        fn recover(self: *@This(), accepted: []const u32) !void {
            const at = self.state().checkpoint_position orelse return error.NoCheckpoint;
            if (self.state().hasRecurrent()) {
                // The batch already fed the accepted prefix when every draft
                // was accepted; nothing is rewound or replayed then.
                if (accepted.len == self.state().position - at) return;
                if (accepted.len > 0 and self.state().row_checkpoints > 0 and accepted.len <= self.state().row_checkpoint_rows) {
                    try self.state().restoreRow(accepted.len - 1);
                    return;
                }
                try self.rewind();
                if (accepted.len > 0) try self.prefill(accepted, null, null);
            } else try self.truncate(at + accepted.len);
        }
        fn sequence(self: *@This(), logits: []f32) !void {
            try self.step(spec.tokens[0], null, null);
            try self.step(spec.tokens[1], logits, null); // exercises a nonzero position
        }
    };
    const binding = try Family.bind(alloc, &mapped.document);
    var backend: ?inference.metal.Backend = null;
    defer if (backend) |*b| b.deinit();
    if (use_metal) {
        var diagnostic: [8192]u8 = @splat(0);
        backend = inference.metal.Backend.init(alloc, &diagnostic) catch |err| {
            std.debug.print("{s}\n", .{std.mem.sliceTo(&diagnostic, 0)});
            return err;
        };
    }
    // A checkpoint region so the recovery check can take and undo a verify
    // batch; 16 positions hold the header, an 8-row batch, and its correction.
    // The Metal plan also takes a drafter when the family has one, so the
    // recovery check can run verify batches and restore their row checkpoints.
    // The family may embed a draft block that a particular file drops
    // (Bonsai 2's re-encoding of Qwen3.8 has none), so the draft checks
    // follow the file, not the family.
    const has_draft = if (comptime spec.draft != null) binding.draft != null else false;
    const first_draft = use_metal and has_draft;
    var first: Model = if (backend) |*b| .{ .metal = try Plan.init(alloc, b, mapped.view(), binding, 16, 16, .f32, true, first_draft) } else .{ .cpu = try Runtime.init(alloc, mapped.view(), binding, 16, true, false) };
    defer first.deinit();
    var second: Model = if (backend) |*b| .{ .metal = try Plan.init(alloc, b, mapped.view(), binding, 16, 16, .f32, true, false) } else .{ .cpu = try Runtime.init(alloc, mapped.view(), binding, 16, true, false) };
    defer second.deinit();
    const expected = try alloc.alloc(f32, spec.vocabulary);
    defer alloc.free(expected);
    const actual = try alloc.alloc(f32, spec.vocabulary);
    defer alloc.free(actual);
    try first.sequence(expected);
    // A separate session must produce identical results while the first remains
    // populated, detecting accidental shared mutable storage.
    try second.sequence(actual);
    if (!std.mem.eql(f32, expected, actual)) return error.SessionIsolationMismatch;
    // A loaded drafter must not perturb decode while it is never asked to
    // propose: the block's cache and workspace are idle, so the same two
    // tokens decode identically (both executors).
    if (comptime spec.draft != null) if (has_draft) {
        var with_draft: Model = if (backend) |*b|
            .{ .metal = try Plan.init(alloc, b, mapped.view(), binding, 16, 16, .f32, true, true) }
        else
            .{ .cpu = try Runtime.init(alloc, mapped.view(), binding, 16, true, true) };
        defer with_draft.deinit();
        try with_draft.sequence(actual);
        if (!std.mem.eql(f32, expected, actual)) return error.DraftLoadedDecodeMismatch;
    };
    var context: u8 = 0;
    // Both cancellation paths must poison the session and recover on reset.
    for ([_]Observer{
        .{ .context = &context, .layer = cancel },
        .{ .context = &context, .check = cancelCheck },
    }) |observer| {
        first.reset();
        context = 0;
        if (first.step(spec.tokens[0], null, observer)) |_| {
            return error.ExpectedCancellation;
        } else |err| if (err != error.Cancelled) return err;
        if (first.step(spec.tokens[0], null, null)) |_| {
            return error.ExpectedPoisonedSession;
        } else |err| if (err != error.SessionNotReady) return err;
        first.reset();
        try first.sequence(actual);
        if (!std.mem.eql(f32, expected, actual)) return error.ResetMismatch;
    }
    std.debug.print("Generation check passed ({s}, {s}): two-token logits identical across independent sessions and reset after partial-step cancellation through both the layer and the check callbacks.\n", .{ Family.architecture, if (use_metal) "metal" else "cpu" });
    var other: Model = if (backend) |*b| .{ .metal = try Plan.init(alloc, b, mapped.view(), binding, 8, 4, .f32, true, false) } else .{ .cpu = try Runtime.init(alloc, mapped.view(), binding, 8, true, false) };
    defer other.deinit();
    try checkSnapshot(spec, Model, alloc, &first, &second, &other, expected, actual);
    if (comptime spec.draft != null) if (has_draft) {
        try checkDraft(spec, alloc, mapped.view(), binding, spec.draft.?, if (backend) |*b| b else null);
        try draftRecoveryCheck(spec, alloc, if (backend) |*b| b else null, mapped.view(), binding, spec.draft.?);
        if (backend) |*b| try draftBatchCommitCheck(spec, alloc, b, mapped.view(), binding);
    } else std.debug.print("Draft block checks skipped: the file carries no embedded block.\n", .{});
    try recoveryCheck(spec, Model, alloc, io, &first, &second, &other, 4, use_metal);
    if (use_metal) try recoveryCheck(spec, Model, alloc, io, &first, &second, &other, 8, true);
    if (backend) |*b| try checkChunkedPrefill(spec, alloc, b, mapped.view(), binding, expected, actual);
}

/// Step token 1, snapshot, step token 2 → logits A; restore, step
/// token 2 → logits B; A == B bit for bit. A third step after the restore
/// equals the same step of a never-snapshotted run. A session of another
/// capacity refuses the snapshot untouched, and a poisoned session refuses
/// to snapshot until reset.
fn checkSnapshot(comptime spec: Spec, comptime Model: type, alloc: std.mem.Allocator, first: *Model, second: *Model, other: *Model, a: []f32, b: []f32) !void {
    const word = spec.tokens[0];
    const comma = spec.tokens[1];
    const c = try alloc.alloc(f32, spec.vocabulary);
    defer alloc.free(c);
    const d = try alloc.alloc(f32, spec.vocabulary);
    defer alloc.free(d);
    first.reset();
    second.reset();
    try first.step(word, null, null);
    var snap = try first.state().snapshot(alloc);
    defer snap.deinit();
    if (snap.position != 1) return error.SnapshotPositionMismatch;
    try first.step(comma, a, null);
    try first.state().restore(&snap);
    if (first.state().position != 1) return error.RestorePositionMismatch;
    try first.step(comma, b, null);
    if (!std.mem.eql(f32, a, b)) return error.SnapshotRestoreMismatch;
    try first.step(word, c, null);
    try second.step(word, null, null);
    try second.step(comma, null, null);
    try second.step(word, d, null);
    if (!std.mem.eql(f32, c, d) or first.state().position != 3 or second.state().position != 3) return error.SnapshotContinuationMismatch;
    // Another capacity: typed refusal, session untouched.
    if (other.state().restore(&snap)) |_| return error.ExpectedSnapshotMismatch else |err| if (err != error.SnapshotMismatch) return err;
    if (other.state().position != 0) return error.RestoreTouchedSession;
    // A poisoned session has no committed state to keep.
    first.reset();
    var context: u8 = 0;
    if (first.step(word, null, .{ .context = &context, .check = cancelCheck })) |_| return error.ExpectedCancellation else |err| if (err != error.Cancelled) return err;
    if (first.state().snapshot(alloc)) |_| return error.ExpectedPoisonedSession else |err| if (err != error.SessionNotReady) return err;
    first.reset();
    std.debug.print("Snapshot check passed: restore reproduces the next step's logits bit for bit ({d} bytes at position 1), the continuation matches an unsnapshotted run, other capacities and poisoned sessions are refused.\n", .{snap.bytes()});
}

/// The accepted-prefix recovery contract. Step a header token, checkpoint,
/// then consume a `rows`-token batch. For every accepted length `a` in
/// `[0, rows]`, undo the batch, replay the accepted prefix, and step the next
/// token; the logits must match a second executor that consumed the same
/// tokens one by one from scratch. Replay is the same sequential arithmetic on
/// the CPU (bit-identical); on Metal the batch is chunked, so the family's
/// chunk-versus-step tolerance applies and the difference is printed. Then the
/// refusals: rewind without a checkpoint, truncate on a recurrent layout, a
/// cancelled batch poisoning the session, and an over-capacity batch refused
/// before any write.
fn recoveryCheck(comptime spec: Spec, comptime Model: type, alloc: std.mem.Allocator, io: std.Io, first: *Model, second: *Model, other: *Model, rows: usize, use_metal: bool) !void {
    const count = rows + 1;
    const capacity = first.state().capacity;
    const tokens = try alloc.alloc(u32, @max(count + 1, capacity + 1));
    defer alloc.free(tokens);
    var seed: u32 = 0x9e3779b9;
    for (tokens) |*t| {
        seed = seed *% 1664525 +% 1013904223;
        t.* = seed % @as(u32, @intCast(spec.vocabulary));
    }
    const a = try alloc.alloc(f32, spec.vocabulary);
    defer alloc.free(a);
    const b = try alloc.alloc(f32, spec.vocabulary);
    defer alloc.free(b);

    first.reset();
    second.reset();
    try first.step(tokens[0], null, null);
    const checkpoint_start = std.Io.Clock.awake.now(io);
    try first.checkpoint();
    const checkpoint_time = checkpoint_start.durationTo(std.Io.Clock.awake.now(io));
    try first.batch(tokens[1 .. 1 + rows]);
    for (0..rows + 1) |accepted| {
        try first.recover(tokens[1 .. 1 + accepted]);
        try first.step(tokens[1 + accepted], a, null);
        second.reset();
        for (tokens[0 .. 1 + accepted]) |t| try second.step(t, null, null);
        try second.step(tokens[1 + accepted], b, null);
        if (use_metal) {
            try compareRecovery(rows, b, a, spec.bounds.chunk_max_abs, spec.bounds.chunk_rel_rms);
        } else if (!std.mem.eql(f32, a, b)) return error.RecoveryMismatch;
    }
    if (first.state().row_checkpoints > 0) {
        // Slots cover every accepted prefix, not the post-batch state.
        if (first.state().restoreRow(rows + 1)) |_| return error.ExpectedRowNotCheckpointed else |err| if (err != error.RowNotCheckpointed) return err;
    }
    const rewind_start = std.Io.Clock.awake.now(io);
    try first.rewind();
    const rewind_time = rewind_start.durationTo(std.Io.Clock.awake.now(io));
    std.debug.print("Recovery check passed ({d} rows{s}): checkpoint {d:.3} ms, rewind {d:.3} ms, region {d} bytes, rows {d}x{d} bytes.\n", .{
        rows, if (first.state().row_checkpoints > 0) ", row slots" else "", checkpoint_time.toMilliseconds(), rewind_time.toMilliseconds(), first.state().checkpoint_region.len, first.state().row_checkpoints, first.state().row_slot_bytes,
    });

    // Rewind without a live checkpoint; a row restore without a batch is
    // refused the same way.
    first.reset();
    if (first.rewind()) |_| return error.ExpectedNoCheckpoint else |err| if (err != error.NoCheckpoint) return err;
    if (first.state().row_checkpoints > 0) {
        try first.step(tokens[0], null, null);
        try first.checkpoint();
        if (first.state().restoreRow(0)) |_| return error.ExpectedRowNotCheckpointed else |err| if (err != error.RowNotCheckpointed) return err;
    }

    // Position truncate: a recurrent layout refuses, an attention-only one accepts.
    first.reset();
    try first.step(tokens[0], null, null);
    try first.checkpoint();
    try first.step(tokens[1], null, null);
    if (first.state().hasRecurrent()) {
        if (first.truncate(1)) |_| return error.ExpectedRecurrentRefusal else |err| if (err != error.RecurrentStateNotRewindable) return err;
    } else {
        try first.truncate(1);
        if (first.state().position != 1) return error.TruncatePositionMismatch;
    }

    // A cancelled batch poisons the session exactly as a cancelled step does.
    first.reset();
    try first.step(tokens[0], null, null);
    try first.checkpoint();
    var context: u8 = 0;
    if (first.prefill(tokens[1 .. 1 + rows], null, .{ .context = &context, .check = cancelCheck })) |_| return error.ExpectedCancellation else |err| if (err != error.Cancelled) return err;
    if (first.checkpoint()) |_| return error.ExpectedPoisonedSession else |err| if (err != error.SessionNotReady) return err;
    if (first.rewind()) |_| return error.ExpectedPoisonedSession else |err| if (err != error.SessionNotReady) return err;
    first.reset();
    if (first.rewind()) |_| return error.ExpectedNoCheckpoint else |err| if (err != error.NoCheckpoint) return err;

    // A batch past the capacity is refused before the first write.
    other.reset();
    try other.step(tokens[0], null, null);
    const before = other.state().position;
    if (other.prefill(tokens[0..other.state().capacity], null, null)) |_| return error.ExpectedContextFull else |err| if (err != error.ContextFull) return err;
    if (other.state().position != before) return error.OverflowTouchedSession;
}

/// The prediction block against the reference's pinned trace, on the CPU
/// reference and, when a backend is given, on the Metal plan: at position 0
/// pair the first token with a zero hidden, at position 1 pair the second with
/// the first position's target hidden. Both the block's `h_nextn` and its
/// greedy token must match. The trace was captured from the pinned reference
/// with an F32 cache; the tolerances are the bring-up block tolerances
/// (docs/reference/speculative-decoding.md).
fn checkDraft(comptime spec: Spec, alloc: std.mem.Allocator, view: inference.weights.View, binding: spec.Family.Binding, draft: Draft, backend: ?*inference.metal.Backend) !void {
    const Runtime = spec.Family.Runtime;
    const hidden = 5120;
    const p0_h = @embedFile("src/models/fixtures/qwen35-mtp/p0-h.f32");
    const p1_h = @embedFile("src/models/fixtures/qwen35-mtp/p1-h.f32");
    const p1_hprev = @embedFile("src/models/fixtures/qwen35-mtp/p1-hprev.f32");
    const logits = try alloc.alloc(f32, spec.vocabulary);
    defer alloc.free(logits);
    const zeros = try alloc.alloc(f32, hidden);
    defer alloc.free(zeros);
    @memset(zeros, 0);
    const h_prev = try alloc.alloc(f32, hidden);
    defer alloc.free(h_prev);
    for (h_prev, 0..) |*v, i| v.* = std.mem.bytesToValue(f32, p1_hprev[i * 4 ..][0..4]);

    var runtime = try Runtime.init(alloc, view, binding, 4, false, true);
    defer runtime.deinit();
    try runtime.draftForward(zeros, draft.tokens[0], 0, logits);
    try compareDraft("cpu position 0", p0_h, runtime.draft_h, 2e-2, 1e-3);
    if (argmax(logits) != draft.greedy[0]) return error.DraftGreedyMismatch;
    try runtime.draftForward(h_prev, draft.tokens[1], 1, logits);
    try compareDraft("cpu position 1", p1_h, runtime.draft_h, 2e-2, 1e-3);
    if (argmax(logits) != draft.greedy[1]) return error.DraftGreedyMismatch;

    if (backend) |b| {
        const Plan = spec.Family.Plan;
        var plan = try Plan.init(alloc, b, view, binding, 4, 4, .f32, false, true);
        defer plan.deinit();
        try plan.draftForwardHost(zeros, draft.tokens[0], 0, null, null, logits);
        try compareDraft("metal f32 position 0", p0_h, plan.draft_h.floats()[0..hidden], 2e-2, 1e-3);
        if (argmax(logits) != draft.greedy[0]) return error.DraftGreedyMismatch;
        try plan.draftForwardHost(h_prev, draft.tokens[1], 1, null, null, logits);
        try compareDraft("metal f32 position 1", p1_h, plan.draft_h.floats()[0..hidden], 2e-2, 1e-3);
        if (argmax(logits) != draft.greedy[1]) return error.DraftGreedyMismatch;

        // The F16 cache's rounding of the block's keys and values at the
        // family's recorded tolerance, the same rows.
        const bounds = spec.bounds;
        var half = try Plan.init(alloc, b, view, binding, 4, 4, .f16, false, true);
        defer half.deinit();
        try half.draftForwardHost(zeros, draft.tokens[0], 0, null, null, logits);
        try compareDraft("metal f16 position 0", p0_h, half.draft_h.floats()[0..hidden], bounds.half_max_abs, bounds.half_rel_rms);
        if (argmax(logits) != draft.greedy[0]) return error.DraftGreedyMismatch;
        try half.draftForwardHost(h_prev, draft.tokens[1], 1, null, null, logits);
        try compareDraft("metal f16 position 1", p1_h, half.draft_h.floats()[0..hidden], bounds.half_max_abs, bounds.half_rel_rms);
        if (argmax(logits) != draft.greedy[1]) return error.DraftGreedyMismatch;

        // `prefill` yields the post-`output_norm` hidden of every row (the
        // drafter's commit input); it must reproduce `verify`'s rows, which
        // are already checked.
        const rows = draft.tokens.len;
        const prefill_hidden = try alloc.alloc(f32, rows * hidden);
        defer alloc.free(prefill_hidden);
        const verify_hidden = try alloc.alloc(f32, rows * hidden);
        defer alloc.free(verify_hidden);
        const verify_rows = try alloc.alloc(f32, rows * spec.vocabulary);
        defer alloc.free(verify_rows);
        var chunked = try Plan.init(alloc, b, view, binding, 4, 4, .f32, false, true);
        defer chunked.deinit();
        try chunked.prefill(&draft.tokens, null, null, null, null, prefill_hidden, null);
        var verified = try Plan.init(alloc, b, view, binding, 4, 4, .f32, false, true);
        defer verified.deinit();
        try verified.verify(&draft.tokens, verify_rows, null, verify_hidden, null);
        var hidden_max_abs: f64 = 0;
        for (verify_hidden, prefill_hidden) |e, a| hidden_max_abs = @max(hidden_max_abs, @abs(@as(f64, e) - a));
        std.debug.print("Prefill hidden check passed: {d} rows match verify (max abs {e:.3}).\n", .{ rows, hidden_max_abs });
        if (hidden_max_abs > 1e-3) return error.PrefillHiddenMismatch;
    }
    std.debug.print("Draft block check passed ({s}): positions 0 and 1 match the pinned trace; greedy tokens {d} and {d}.\n", .{ if (backend != null) "cpu and metal" else "cpu", draft.greedy[0], draft.greedy[1] });
}

/// The block's cache rides the session's recovery: `reset` clears it and a
/// checkpoint/rewind leaves it rewritable, so the same row reproduces its
/// hidden. `propose` is deterministic across two independently reset runners.
/// Runs on the CPU reference or the Metal plan (whichever backend is given;
/// null selects the CPU reference).
fn draftRecoveryCheck(comptime spec: Spec, alloc: std.mem.Allocator, backend: ?*inference.metal.Backend, view: inference.weights.View, binding: spec.Family.Binding, draft: Draft) !void {
    const Family = spec.Family;
    const hidden = 5120;
    const p1_hprev = @embedFile("src/models/fixtures/qwen35-mtp/p1-hprev.f32");
    const zeros = try alloc.alloc(f32, hidden);
    defer alloc.free(zeros);
    @memset(zeros, 0);
    const h_prev = try alloc.alloc(f32, hidden);
    defer alloc.free(h_prev);
    for (h_prev, 0..) |*v, i| v.* = std.mem.bytesToValue(f32, p1_hprev[i * 4 ..][0..4]);
    const before = try alloc.alloc(f32, hidden);
    defer alloc.free(before);

    var first: DraftRunner(spec) = if (backend) |b|
        .{ .metal = try Family.Plan.init(alloc, b, view, binding, 8, 8, .f32, true, true) }
    else
        .{ .cpu = try Family.Runtime.init(alloc, view, binding, 8, true, true) };
    defer first.deinit();
    var second: DraftRunner(spec) = if (backend) |b|
        .{ .metal = try Family.Plan.init(alloc, b, view, binding, 8, 8, .f32, true, true) }
    else
        .{ .cpu = try Family.Runtime.init(alloc, view, binding, 8, true, true) };
    defer second.deinit();

    // Reset clears both the block's cache and its pending target hidden.
    try first.forward(zeros, draft.tokens[0], 0, null, null);
    @memcpy(before, first.blockHidden());
    first.reset();
    try first.forward(zeros, draft.tokens[0], 0, null, null);
    if (!std.mem.eql(f32, before, first.blockHidden())) return error.DraftResetMismatch;

    // A checkpoint/rewind leaves the block's row rewritable to the same bytes.
    first.reset();
    try first.forward(zeros, draft.tokens[0], 0, null, null);
    try first.checkpoint();
    try first.forward(h_prev, draft.tokens[1], 1, null, null);
    @memcpy(before, first.blockHidden());
    try first.rewind();
    try first.forward(h_prev, draft.tokens[1], 1, null, null);
    if (!std.mem.eql(f32, before, first.blockHidden())) return error.DraftRewindMismatch;

    // Two reset runners propose the same greedy chain.
    var one: [4]u32 = undefined;
    var two: [4]u32 = undefined;
    first.reset();
    second.reset();
    _ = try first.propose(draft.tokens[0], &one, 0);
    _ = try second.propose(draft.tokens[0], &two, 0);
    if (!std.mem.eql(u32, &one, &two)) return error.DraftProposeMismatch;
    std.debug.print("Draft recovery check passed ({s}): reset and rewind reproduce the block's hidden; propose is deterministic.\n", .{if (backend != null) "metal" else "cpu"});
}

/// `Plan.commit` over a 16-token prefix in one batched forward against
/// committing the same rows one at a time (`draftForwardHost`): the proposed
/// chains must match and the block logits agree within the block's bound
/// (2e-2 max abs / 1e-3 relative RMS). The two plans consume the same prefix,
/// so the batched `prefill` also yields the target hidden the serial path is
/// driven with.
fn draftBatchCommitCheck(comptime spec: Spec, alloc: std.mem.Allocator, b: *inference.metal.Backend, view: inference.weights.View, binding: spec.Family.Binding) !void {
    const Plan = spec.Family.Plan;
    const hidden = 5120;
    const count = 16;
    const draft_n = 4;
    const tokens = try alloc.alloc(u32, count);
    defer alloc.free(tokens);
    var seed: u32 = 0xc0ffee;
    for (tokens) |*t| {
        seed = seed *% 1664525 +% 1013904223;
        t.* = seed % 150000;
    }
    const hidden_rows = try alloc.alloc(f32, count * hidden);
    defer alloc.free(hidden_rows);

    var batched = try Plan.init(alloc, b, view, binding, 2 * count, count, .f32, false, true);
    defer batched.deinit();
    var serial = try Plan.init(alloc, b, view, binding, 2 * count, count, .f32, false, true);
    defer serial.deinit();
    try batched.prefill(tokens, null, null, null, null, hidden_rows, null);
    try serial.prefill(tokens, null, null, null, null, null, null);
    try batched.commit(tokens, hidden_rows);
    for (tokens, 0..) |token, i| {
        const h_prev: []const f32 = if (i == 0) serial.draft_pending_h.floats()[0..hidden] else hidden_rows[(i - 1) * hidden ..][0..hidden];
        try serial.draftForwardHost(h_prev, token, i, null, null, null);
    }
    @memcpy(serial.draft_pending_h.floats()[0..hidden], hidden_rows[hidden_rows.len - hidden ..][0..hidden]);

    const seed_token = tokens[count - 1];
    var a: [draft_n]u32 = undefined;
    var c: [draft_n]u32 = undefined;
    const na = try batched.propose(seed_token, &a, 0);
    const nc = try serial.propose(seed_token, &c, 0);
    if (na != nc or !std.mem.eql(u32, a[0..na], c[0..nc])) return error.BatchedCommitDraftMismatch;

    const logits_a = try alloc.alloc(f32, spec.vocabulary);
    defer alloc.free(logits_a);
    const logits_b = try alloc.alloc(f32, spec.vocabulary);
    defer alloc.free(logits_b);
    try batched.draftForwardHost(batched.draft_pending_h.floats()[0..hidden], seed_token, count, null, null, logits_a);
    try serial.draftForwardHost(serial.draft_pending_h.floats()[0..hidden], seed_token, count, null, null, logits_b);
    try compareDraft("batched vs serial commit", std.mem.sliceAsBytes(logits_b), logits_a, 2e-2, 1e-3);
    std.debug.print("Batched commit check passed: {d} tokens batched match the serial commit ({d} drafts identical).\n", .{ count, na });
}

/// Greedy speculation against ordinary greedy decoding. Two seeded sessions
/// consume the pinned prompt; the first decodes token by token, the second
/// proposes a draft block, verifies it greedily, recovers the accepted prefix,
/// and continues from the correction. On the CPU reference the outputs must be
/// token for token identical (the replay is the same sequential arithmetic);
/// on Metal the verify batch uses the chunk tiles, so the first divergence is
/// recorded by position rather than failed.
fn speculativeCheck(comptime spec: Spec, alloc: std.mem.Allocator, io: std.Io, mapped: *inference.weights.Mapped, model_path: []const u8, use_metal: bool) !void {
    const Family = spec.Family;
    const draft = spec.draft.?;
    const hidden = 5120;
    const block_drafts = 4;
    const generated = 12;
    const capacity = spec.tokens.len + generated + block_drafts + 2;
    var backend: ?inference.metal.Backend = null;
    defer if (backend) |*b| b.deinit();
    if (use_metal) {
        var diagnostic: [8192]u8 = @splat(0);
        backend = inference.metal.Backend.init(alloc, &diagnostic) catch |err| {
            std.debug.print("{s}\n", .{std.mem.sliceTo(&diagnostic, 0)});
            return err;
        };
    }
    const binding = try Family.bind(alloc, &mapped.document);
    const view = mapped.view();
    const Runner = DraftRunner(spec);
    var sequential: Runner = if (backend) |*b|
        .{ .metal = try Family.Plan.init(alloc, b, view, binding, capacity, 8, .f32, true, false) }
    else
        .{ .cpu = try Family.Runtime.init(alloc, view, binding, capacity, true, false) };
    defer sequential.deinit();
    var speculative: Runner = if (backend) |*b|
        .{ .metal = try Family.Plan.init(alloc, b, view, binding, capacity, 8, .f32, true, true) }
    else
        .{ .cpu = try Family.Runtime.init(alloc, view, binding, capacity, true, true) };
    defer speculative.deinit();
    const logits = try alloc.alloc(f32, spec.vocabulary);
    defer alloc.free(logits);
    const prompt = [_]u32{ draft.tokens[0], draft.tokens[1] };
    const prompt_hidden = try alloc.alloc(f32, prompt.len * hidden);
    defer alloc.free(prompt_hidden);
    const base = try alloc.alloc(u32, generated);
    defer alloc.free(base);
    const result = try alloc.alloc(u32, generated);
    defer alloc.free(result);
    const drafts = try alloc.alloc(u32, block_drafts);
    defer alloc.free(drafts);
    const batch = try alloc.alloc(u32, block_drafts + 1);
    defer alloc.free(batch);
    const choices = try alloc.alloc(u32, block_drafts + 1);
    defer alloc.free(choices);
    const h_rows = try alloc.alloc(f32, (block_drafts + 1) * hidden);
    defer alloc.free(h_rows);

    // The token-by-token baseline.
    try sequential.prefill(&prompt, logits);
    var base_count: usize = 0;
    var next = argmax(logits);
    while (base_count < generated) : (base_count += 1) {
        base[base_count] = next;
        if (base_count + 1 == generated) break;
        try sequential.step(next, logits);
        next = argmax(logits);
    }

    // The speculative run: commit the prompt, then batch-verify each position.
    for (prompt, 0..) |token, i| {
        try speculative.step(token, logits);
        @memcpy(prompt_hidden[i * hidden ..][0..hidden], speculative.lastHidden());
    }
    try speculative.commit(&prompt, prompt_hidden);
    var count: usize = 0;
    var seed = argmax(logits);
    var accepted_total: usize = 0;
    var proposed_total: usize = 0;
    // The correction is already emitted when a batch ends; the next iteration
    // uses it as the seed without emitting it again.
    var carried = false;
    while (count < generated) {
        if (!carried) {
            result[count] = seed;
            count += 1;
            if (count == generated) break;
        }
        carried = false;
        const room = speculative.state().capacity - speculative.state().position - 1;
        const k = @min(block_drafts, room);
        const n = try speculative.propose(seed, drafts[0..k], 0);
        try speculative.checkpoint();
        batch[0] = seed;
        @memcpy(batch[1 .. 1 + n], drafts[0..n]);
        try speculative.verifyGreedy(batch[0 .. 1 + n], choices[0 .. 1 + n], h_rows[0 .. (1 + n) * hidden]);
        var accepted: usize = 0;
        while (accepted < n and choices[accepted] == drafts[accepted]) accepted += 1;
        accepted_total += accepted;
        proposed_total += n;
        const correction = choices[accepted];
        try speculative.recover(batch[0 .. 1 + accepted]);
        try speculative.commit(batch[0 .. 1 + accepted], h_rows[0 .. (1 + accepted) * hidden]);
        for (drafts[0..accepted]) |token| {
            if (count == generated) break;
            result[count] = token;
            count += 1;
        }
        if (count == generated) break;
        result[count] = correction;
        count += 1;
        seed = correction;
        carried = true;
    }

    var divergence: ?usize = null;
    for (base, result, 0..) |expected, actual, i| {
        if (expected != actual) {
            divergence = i;
            break;
        }
    }
    if (divergence) |at| {
        std.debug.print("Speculative greedy diverges from ordinary greedy at position {d} ({s}): sequential {d}, speculative {d}.\n", .{ at, if (backend != null) "metal chunk-versus-step" else "cpu", base[at], result[at] });
        if (backend == null) return error.SpeculativeMismatch;
    } else {
        std.debug.print("Speculative greedy passed ({s}): {d} tokens identical to ordinary greedy decoding; accepted {d}/{d} drafts.\n", .{ if (backend != null) "metal" else "cpu", generated, accepted_total, proposed_total });
    }
    // The same comparison through the engine's loop, which owns the seed,
    // batch, recover, and emission control flow the primitives above do not.
    // Backend-independent, so the Metal run covers it and the slow CPU oracle
    // stays on the primitives.
    if (use_metal) try speculativeLoop(alloc, io, model_path, use_metal);
    if (backend) |*b| try penaltyCheck(spec, alloc, b, view, binding);
    if (backend) |*b| try verifyTopKCheck(spec, alloc, b, view, binding);
}

/// The verify batch's per-row top-k readback against the same batch's full
/// rows: two identically configured plans consume the same four tokens, one
/// verifies into full rows and the other into the device readback. For every
/// row and a fixed seed — with and without the history penalties — the
/// readback path must select the token the rows path's `select` selects; a
/// row the readback cannot decide must fall back to the resident logits,
/// which must be bit-identical to the rows plan's, and `select` on those must
/// agree. Exercises the whole ENGN-15 seam on real logits.
fn verifyTopKCheck(comptime spec: Spec, alloc: std.mem.Allocator, b: *inference.metal.Backend, view: inference.weights.View, binding: spec.Family.Binding) !void {
    const Plan = spec.Family.Plan;
    const draft = spec.draft.?;
    const count = 4;
    const capacity = 64;
    var rows_plan = try Plan.init(alloc, b, view, binding, capacity, 8, .f32, false, true);
    defer rows_plan.deinit();
    var topk_plan = try Plan.init(alloc, b, view, binding, capacity, 8, .f32, false, true);
    defer topk_plan.deinit();
    const batch = [count]u32{ draft.tokens[0], draft.tokens[1], 9419, 271 };
    const rows = try alloc.alloc(f32, count * spec.vocabulary);
    defer alloc.free(rows);
    const resident = try alloc.alloc(f32, spec.vocabulary);
    defer alloc.free(resident);
    const candidates = try alloc.alloc(inference.sampling.Candidate, spec.vocabulary);
    defer alloc.free(candidates);
    var tops: [count]inference.sampling.TopK = undefined;
    const options_sets = [_]inference.sampling.Options{
        .{ .temperature = 0.7, .top_k = 20, .top_p = 0.8 },
        .{ .temperature = 0.7, .top_k = 20, .top_p = 0.8, .presence_penalty = 1.5, .repetition_penalty = 1.1 },
    };
    try rows_plan.verify(&batch, rows, null, null, null);
    for (&tops) |*top| top.* = .{ .temperature = 0.7 };
    try topk_plan.verify(&batch, null, &tops, null, null);
    var decided: usize = 0;
    var fallbacks: usize = 0;
    for (options_sets, 0..) |options, set| {
        var history = try inference.sampling.History.init(alloc, spec.vocabulary);
        defer history.deinit();
        try history.observe(0);
        try history.observe(1);
        try history.observe(draft.tokens[0]);
        for (0..count) |i| {
            const row = rows[i * spec.vocabulary ..][0..spec.vocabulary];
            var topk_sampler = try inference.sampling.Sampler.init(0x51 + set * count + i, options);
            var rows_sampler = try inference.sampling.Sampler.init(0x51 + set * count + i, options);
            const expected = try rows_sampler.select(row, candidates, &history);
            if (try topk_sampler.selectFromHistory(&tops[i], candidates, &history)) |token| {
                decided += 1;
                if (token != expected) return error.VerifyTopKMismatch;
            } else {
                fallbacks += 1;
                try topk_plan.readVerifyRow(i, resident);
                if (!std.mem.eql(f32, resident, row)) return error.VerifyRowMismatch;
                if (try topk_sampler.select(resident, candidates, &history) != expected) return error.VerifyTopKFallbackMismatch;
            }
        }
    }
    std.debug.print("Verify top-k check (metal): {d}/{d} rows decided by the readback, {d} fell back to the resident logits, all equal to the rows path.\n", .{ decided, decided + fallbacks, fallbacks });
}

/// The device history penalties against the sampler on the same logits: two
/// plans consume the same tokens, one steps with the partial top-k readback
/// taken after `nu_penalize`, the other reads full logits back and lets the
/// sampler apply the penalties itself. With one seed each, every selected
/// token must be identical — sampled (the top-k path) and greedy (the argmax
/// path), both with penalties active. Each plan's arithmetic is identical, so
/// the logits the two sides decide on are the same vector.
fn penaltyCheck(comptime spec: Spec, alloc: std.mem.Allocator, b: *inference.metal.Backend, view: inference.weights.View, binding: spec.Family.Binding) !void {
    const Plan = spec.Family.Plan;
    const draft = spec.draft.?;
    const capacity = 64;
    const steps = 8;
    var topk_plan = try Plan.init(alloc, b, view, binding, capacity, 8, .f32, false, false);
    defer topk_plan.deinit();
    var full_plan = try Plan.init(alloc, b, view, binding, capacity, 8, .f32, false, false);
    defer full_plan.deinit();
    const logits = try alloc.alloc(f32, spec.vocabulary);
    defer alloc.free(logits);
    const candidates = try alloc.alloc(inference.sampling.Candidate, spec.vocabulary);
    defer alloc.free(candidates);
    var history = try inference.sampling.History.init(alloc, spec.vocabulary);
    defer history.deinit();
    // A history with both ends of the vocabulary and the pinned tokens, so
    // the penalty reaches the ids the head proposes and the id-ordered ties.
    try history.observe(0);
    try history.observe(1);
    try history.observe(spec.vocabulary - 1);
    try history.observe(draft.tokens[0]);
    try history.observe(draft.tokens[1]);

    const options: inference.sampling.Options = .{ .temperature = 0.7, .top_k = 20, .top_p = 0.8, .presence_penalty = 1.5, .repetition_penalty = 1.1 };
    const penalties: inference.sampling.Penalties = .{ .history = &history, .repetition = options.repetition_penalty, .presence = options.presence_penalty };
    try topk_plan.step(draft.tokens[0], null, null, null, null, null);
    try full_plan.step(draft.tokens[0], null, null, null, null, null);
    try topk_plan.step(draft.tokens[1], null, null, null, null, null);
    try full_plan.step(draft.tokens[1], logits, null, null, null, null);

    var top: inference.sampling.TopK = .{ .temperature = options.temperature };
    var topk_sampler = try inference.sampling.Sampler.init(0x9e37, options);
    var full_sampler = try inference.sampling.Sampler.init(0x9e37, options);
    // The seeds come from their own sampler so the two compared streams both
    // start fresh at the first loop draw.
    var seed_sampler = try inference.sampling.Sampler.init(0x9e37, options);
    var token = try seed_sampler.select(logits, candidates, &history);
    for (0..steps) |_| {
        try topk_plan.step(token, null, null, &top, penalties, null);
        const from_topk = (try topk_sampler.selectFrom(&top, candidates)) orelse return error.PenaltyTopKFallback;
        try full_plan.step(token, logits, null, null, null, null);
        const from_full = try full_sampler.select(logits, candidates, &history);
        if (from_topk != from_full) return error.PenaltySampledMismatch;
        try history.observe(token);
        token = from_topk;
    }
    std.debug.print("Penalty check (metal, sampled): {d} tokens identical between the device top-k readback and the full-logit sampler.\n", .{steps});

    // The greedy sibling: the device argmax after the same penalties versus
    // the sampler's penalized argmax, stepped independently from scratch.
    topk_plan.reset();
    full_plan.reset();
    history.reset();
    try history.observe(0);
    try history.observe(1);
    try history.observe(spec.vocabulary - 1);
    try history.observe(draft.tokens[0]);
    try history.observe(draft.tokens[1]);
    var chosen: u32 = undefined;
    try topk_plan.step(draft.tokens[0], null, null, null, null, null);
    try full_plan.step(draft.tokens[0], null, null, null, null, null);
    try topk_plan.step(draft.tokens[1], null, null, null, null, null);
    try full_plan.step(draft.tokens[1], logits, null, null, null, null);
    var greedy_sampler = try inference.sampling.Sampler.init(0, .{ .presence_penalty = options.presence_penalty, .repetition_penalty = options.repetition_penalty });
    var expected = try greedy_sampler.select(logits, candidates, &history);
    for (0..steps) |_| {
        try history.observe(expected);
        try topk_plan.step(expected, null, &chosen, null, penalties, null);
        try full_plan.step(expected, logits, null, null, null, null);
        expected = try greedy_sampler.select(logits, candidates, &history);
        if (chosen != expected) return error.PenaltyGreedyMismatch;
    }
    std.debug.print("Penalty check (metal, greedy): {d} tokens identical between the device argmax and the penalized sampler.\n", .{steps + 1});
}

/// Runs `inference.engine.runLoop` twice on one loaded engine, ordinary greedy
/// then speculative greedy, and requires identical token streams. This is the
/// loop-level check: propose, checkpoint, verify, recover, commit, emission.
fn speculativeLoop(alloc: std.mem.Allocator, io: std.Io, model_path: []const u8, use_metal: bool) !void {
    const engine = inference.engine;
    const generated = 12;
    const max_limit = 24;
    const capacity = 64;
    const backend: engine.Backend = if (use_metal) .metal else .cpu;
    var eng = try engine.Engine.open(alloc, io, model_path, backend, capacity, .f32, null, .embedded);
    defer eng.deinit();
    const prompt = try eng.encode("Hello,");
    defer alloc.free(prompt);
    const vocab = eng.vocab.tokens.len;
    const logits = try alloc.alloc(f32, vocab);
    defer alloc.free(logits);
    const a = try alloc.alloc(u32, max_limit);
    defer alloc.free(a);
    const b = try alloc.alloc(u32, max_limit);
    defer alloc.free(b);
    var sampler = try inference.sampling.Sampler.init(0, .{});
    const no_candidates: []inference.sampling.Candidate = &.{};

    resetForRun(&eng);
    const first = try engine.runLoop(&eng, prompt, generated, &sampler, null, .{}, null, logits, no_candidates, a, null, null);
    resetForRun(&eng);
    const second = try engine.runLoop(&eng, prompt, generated, &sampler, null, .{ .enabled = true, .draft_length = 4 }, null, logits, no_candidates, b, null, null);
    if (first.timing.generated_tokens != second.timing.generated_tokens) return error.SpeculativeLoopLengthMismatch;
    if (!std.mem.eql(u32, a[0..first.timing.generated_tokens], b[0..second.timing.generated_tokens])) return error.SpeculativeLoopMismatch;
    // Partial acceptance: the run rejected at least one draft, so the
    // correction path ran.
    if (second.timing.accepted_drafts >= second.timing.proposed_drafts) return error.SpeculativeNoPartialAcceptance;
    std.debug.print("Speculative loop passed ({s}): {d} tokens identical to ordinary greedy through engine.runLoop; accepted {d}/{d} drafts.\n", .{ if (use_metal) "metal" else "cpu", first.timing.generated_tokens, second.timing.accepted_drafts, second.timing.proposed_drafts });

    const spec: engine.Speculative = .{ .enabled = true, .draft_length = 4 };

    // Budget inside a batch: a limit below what the first batch would emit.
    resetForRun(&eng);
    const short = try engine.runLoop(&eng, prompt, 3, &sampler, null, spec, null, logits, no_candidates, a, null, null);
    if (short.stop != .token_budget or short.timing.generated_tokens != 3) return error.SpeculativeBudgetMismatch;
    std.debug.print("Speculative budget passed: stopped at {d} tokens inside a batch.\n", .{short.timing.generated_tokens});

    // EOS inside a batch: the rendered turn ends with the profile's stop
    // token, which speculation must emit and then stop on.
    const rendered = try eng.prompt("Hello,", false, .off);
    defer alloc.free(rendered);
    const eos_tokens = try eng.encode(rendered);
    defer alloc.free(eos_tokens);
    resetForRun(&eng);
    const eos = try engine.runLoop(&eng, eos_tokens, 24, &sampler, null, spec, null, logits, no_candidates, a, null, null);
    if (eos.stop != .eos) return error.SpeculativeEosMismatch;
    std.debug.print("Speculative EOS passed: stopped after {d} tokens.\n", .{eos.timing.generated_tokens});

    // Cancellation mid-batch: the observer's check fires during the first
    // decode verify (the prompt commit leaves the position at its end), and
    // the loop resets the poisoned session and reports cancellation.
    var canceller = CancelAt{ .eng = &eng, .at = prompt.len };
    resetForRun(&eng);
    const cancelled = try engine.runLoop(&eng, prompt, 24, &sampler, null, spec, null, logits, no_candidates, a, .{ .context = &canceller, .check = CancelAt.check }, null);
    if (cancelled.stop != .cancelled) return error.SpeculativeCancellationMismatch;
    if (eng.model.session().position != 0) return error.SpeculativeCancellationNotReset;
    std.debug.print("Speculative cancellation passed: a cancelled verify reset the session.\n", .{});

    // Context limit at a batch: a session with room for the prompt and two
    // tokens cannot hold a verify batch plus its correction.
    var small = try engine.Engine.open(alloc, io, model_path, backend, prompt.len + 2, .f32, null, .embedded);
    defer small.deinit();
    const full = try engine.runLoop(&small, prompt, 24, &sampler, null, spec, null, logits, no_candidates, a, null, null);
    if (full.stop != .context_limit) return error.SpeculativeContextMismatch;
    std.debug.print("Speculative context passed: stopped at the {d}-token context.\n", .{prompt.len + 2});
}

/// Cancels a turn once the session position reaches `at`, which during a
/// verify batch is the batch's start: the prompt commit stays below it.
const CancelAt = struct {
    eng: *inference.engine.Engine,
    at: usize,
    fn check(context: *anyopaque) !void {
        const self: *CancelAt = @ptrCast(@alignCast(context));
        if (self.eng.model.session().position >= self.at) return error.Cancelled;
    }
};

fn resetForRun(eng: *inference.engine.Engine) void {
    eng.model.reset();
    if (eng.model.drafter()) |drafter| drafter.reset();
}

/// A per-depth acceptance statistic for the embedded prediction head. It
/// decodes a fixed coding prompt greedily, records the target hidden of every
/// token, and at each step proposes `max_drafts` chained candidates from the
/// state after the committed prefix, then counts whether draft `i` equals the
/// token the target chose `i` steps later. The table bounds the verification
/// theme cares about; a low rate is a measurement, not a failure. Draft
/// latency per position and the block's own bytes are reported alongside.
const max_drafts = 4;
const draft_generated = 32;
const draft_prompts = [_][]const u8{
    "Write a Zig function that reverses a string.",
    "def fibonacci(n):",
};

/// The block's top-candidate probability at temperature 1: the exact F64
/// softmax maximum over the row. The Metal policy reads the same quantity
/// from one top-1 pass (`1 / Σ exp(l − max)`).
fn draftPmax(logits: []const f32) f64 {
    var maximum: f64 = -std.math.inf(f64);
    for (logits) |v| maximum = @max(maximum, v);
    var total: f64 = 0;
    for (logits) |v| total += @exp(@as(f64, v) - maximum);
    return if (total > 0) 1.0 / total else 1.0;
}

/// The proposal policy's probability bins: `≥ 0.9`, `0.7–0.9`, `0.5–0.7`,
/// `< 0.5`.
fn pmaxBin(p: f64) u8 {
    if (p >= 0.9) return 0;
    if (p >= 0.7) return 1;
    if (p >= 0.5) return 2;
    return 3;
}

/// Either executor behind one step/hidden/propose surface for the draft
/// statistics and traces.
fn DraftRunner(comptime spec: Spec) type {
    const Family = spec.Family;
    const hidden = 5120;
    return union(enum) {
        cpu: Family.Runtime,
        metal: Family.Plan,
        fn step(self: *@This(), token: u32, logits: []f32) !void {
            switch (self.*) {
                .cpu => |*r| try r.step(token, logits, null),
                .metal => |*p| try p.step(token, logits, null, null, null, null),
            }
        }
        /// The post-`output_norm` hidden of the last step: the block's `h`.
        fn lastHidden(self: *@This()) []const f32 {
            return switch (self.*) {
                .cpu => |*r| r.h,
                .metal => |*p| p.normalized.floats()[0..hidden],
            };
        }
        /// Runs one block row with a host `h_prev`, the block's hidden
        /// landing in `blockHidden`. `pmax`, when given, receives the row's
        /// top-candidate probability (the device's top-1 readback on Metal,
        /// the host softmax on the CPU).
        fn forward(self: *@This(), h_prev: []const f32, token: u32, position: usize, logits: ?[]f32, pmax: ?*f32) !void {
            switch (self.*) {
                .cpu => |*r| {
                    try r.draftForward(h_prev, token, position, logits);
                    if (pmax) |out| out.* = @floatCast(draftPmax(logits.?));
                },
                .metal => |*p| try p.draftForwardHost(h_prev, token, position, null, pmax, logits),
            }
        }
        fn blockHidden(self: *@This()) []const f32 {
            return switch (self.*) {
                .cpu => |*r| r.draft_h,
                .metal => |*p| p.draft_h.floats()[0..hidden],
            };
        }
        /// The target hidden of the last committed token: the first proposed
        /// position's `h_prev`.
        fn pendingHidden(self: *@This()) []const f32 {
            return switch (self.*) {
                .cpu => |*r| r.draft_pending_h,
                .metal => |*p| p.draft_pending_h.floats()[0..hidden],
            };
        }
        fn propose(self: *@This(), token: u32, out: []u32, p_min: f32) !usize {
            return switch (self.*) {
                .cpu => |*r| r.propose(token, out, p_min),
                .metal => |*p| p.propose(token, out, p_min),
            };
        }
        fn commit(self: *@This(), tokens: []const u32, h_rows: []const f32) !void {
            return switch (self.*) {
                .cpu => |*r| r.commit(tokens, h_rows),
                .metal => |*p| p.commit(tokens, h_rows),
            };
        }
        /// One prefill over `tokens`, keeping only the last row's logits; the
        /// batch is admitted before the first write on both executors.
        fn prefill(self: *@This(), tokens: []const u32, logits: ?[]f32) !void {
            switch (self.*) {
                .cpu => |*r| {
                    if (tokens.len > r.state.capacity - r.state.position) return error.ContextFull;
                    for (tokens, 0..) |token, i| try r.step(token, if (i + 1 == tokens.len) logits else null, null);
                },
                .metal => |*p| try p.prefill(tokens, logits, null, null, null, null, null),
            }
        }
        fn verify(self: *@This(), tokens: []const u32, rows: []f32, h_rows: ?[]f32) !void {
            switch (self.*) {
                .cpu => |*r| try r.verify(tokens, rows, h_rows, null),
                .metal => |*p| try p.verify(tokens, rows, null, h_rows, null),
            }
        }
        fn verifyGreedy(self: *@This(), tokens: []const u32, out: []u32, h_rows: ?[]f32) !void {
            switch (self.*) {
                .cpu => |*r| try r.verifyGreedy(tokens, out, h_rows, null),
                .metal => |*p| try p.verifyGreedy(tokens, out, h_rows, null),
            }
        }
        fn state(self: *@This()) *inference.session.Session {
            return switch (self.*) {
                .cpu => |*r| &r.state,
                .metal => |*p| &p.state,
            };
        }
        fn truncate(self: *@This(), position: usize) !void {
            return self.state().truncate(position);
        }
        /// The accepted-prefix operation, mirroring `engine.Model.recover`.
        fn recover(self: *@This(), accepted: []const u32) !void {
            const at = self.state().checkpoint_position orelse return error.NoCheckpoint;
            if (self.state().hasRecurrent()) {
                // The batch already fed the accepted prefix when every draft
                // was accepted; nothing is rewound or replayed then.
                if (accepted.len == self.state().position - at) return;
                try self.rewind();
                if (accepted.len > 0) try self.prefill(accepted, null);
            } else try self.truncate(at + accepted.len);
        }
        fn bytes(self: *@This()) usize {
            return switch (self.*) {
                .cpu => |*r| r.drafter().?.bytes(),
                .metal => |*p| p.drafter().?.bytes(),
            };
        }
        fn reset(self: *@This()) void {
            switch (self.*) {
                .cpu => |*r| r.reset(),
                .metal => |*p| p.reset(),
            }
        }
        fn checkpoint(self: *@This()) !void {
            return switch (self.*) {
                .cpu => |*r| r.state.checkpoint(),
                .metal => |*p| p.state.checkpoint(),
            };
        }
        fn rewind(self: *@This()) !void {
            return switch (self.*) {
                .cpu => |*r| r.state.rewind(),
                .metal => |*p| p.state.rewind(),
            };
        }
        fn deinit(self: *@This()) void {
            switch (self.*) {
                .cpu => |*r| r.deinit(),
                .metal => |*p| p.deinit(),
            }
        }
    };
}

fn draftStats(comptime spec: Spec, alloc: std.mem.Allocator, io: std.Io, mapped: *inference.weights.Mapped, use_metal: bool) !void {
    const Family = spec.Family;
    const hidden = 5120;
    var backend: ?inference.metal.Backend = null;
    defer if (backend) |*b| b.deinit();
    if (use_metal) {
        var diagnostic: [8192]u8 = @splat(0);
        backend = inference.metal.Backend.init(alloc, &diagnostic) catch |err| {
            std.debug.print("{s}\n", .{std.mem.sliceTo(&diagnostic, 0)});
            return err;
        };
    }
    const binding = try Family.bind(alloc, &mapped.document);
    var vocab = try inference.vocabulary.load(alloc, mapped.document, mapped.mapping.memory[0..@intCast(mapped.document.directory_bytes)], .{});
    defer vocab.deinit();
    var encoder = try inference.tokenizer.Encoder.init(alloc, &vocab);
    defer encoder.deinit();

    const Runner = DraftRunner(spec);

    for (draft_prompts) |text| {
        const prompt = try encoder.encode(alloc, text, true, .{});
        defer alloc.free(prompt);
        const capacity = prompt.len + draft_generated + max_drafts + 1;
        var runner: Runner = if (backend) |*b|
            .{ .metal = try Family.Plan.init(alloc, b, mapped.view(), binding, capacity, @min(capacity, 256), .f32, false, true) }
        else
            .{ .cpu = try Family.Runtime.init(alloc, mapped.view(), binding, capacity, false, true) };
        defer runner.deinit();
        const logits = try alloc.alloc(f32, spec.vocabulary);
        defer alloc.free(logits);
        const sequence = try alloc.alloc(u32, capacity);
        defer alloc.free(sequence);
        const hidden_rows = try alloc.alloc(f32, capacity * hidden);
        defer alloc.free(hidden_rows);
        const seeds = try alloc.alloc(usize, draft_generated);
        defer alloc.free(seeds);
        const drafts = try alloc.alloc(u32, draft_generated * max_drafts);
        defer alloc.free(drafts);
        const counts = try alloc.alloc(usize, draft_generated);
        defer alloc.free(counts);
        @memset(counts, 0);
        // The proposal policy's bins: each proposed position's probability
        // bin, kept per draft so the acceptance loop can count both.
        const bins = 4;
        var bin_total = [_]usize{0} ** bins;
        var bin_accepted = [_]usize{0} ** bins;
        const bin_of_draft = try alloc.alloc(u8, draft_generated * max_drafts);
        defer alloc.free(bin_of_draft);
        @memset(bin_of_draft, 0);
        var pmax_worst: f64 = 0;

        // The prompt, then the greedy continuation; every step's target
        // hidden is kept for the drafter's `commit`.
        var len: usize = 0;
        for (prompt) |token| {
            try runner.step(token, logits);
            sequence[len] = token;
            @memcpy(hidden_rows[len * hidden ..][0..hidden], runner.lastHidden());
            len += 1;
        }
        try runner.commit(sequence[0..len], hidden_rows[0 .. len * hidden]);
        var next = argmax(logits);
        var propose_ns: i64 = 0;
        var proposed: usize = 0;
        var j: usize = 0;
        while (j < draft_generated) : (j += 1) {
            const start = std.Io.Clock.awake.now(io);
            // Propose position by position so each draft's p_max is known:
            // the device's top-1 probability on Metal, the host softmax on
            // the CPU; the proposed chain is the same greedy one `propose`
            // builds.
            var h_prev: []const f32 = runner.pendingHidden();
            var next_draft = next;
            var k: usize = 0;
            while (k < max_drafts) : (k += 1) {
                var device_pmax: f32 = 0;
                try runner.forward(h_prev, next_draft, len + k, logits, &device_pmax);
                const pmax = draftPmax(logits);
                pmax_worst = @max(pmax_worst, @abs(@as(f64, device_pmax) - pmax));
                const draft = argmax(logits);
                drafts[j * max_drafts + k] = draft;
                bin_of_draft[j * max_drafts + k] = pmaxBin(pmax);
                next_draft = draft;
                h_prev = runner.blockHidden();
            }
            propose_ns += @intCast(start.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds());
            proposed += k;
            counts[j] = k;
            seeds[j] = len; // the seed token is appended next; its index is `len`.
            try runner.step(next, logits);
            sequence[len] = next;
            @memcpy(hidden_rows[len * hidden ..][0..hidden], runner.lastHidden());
            len += 1;
            next = argmax(logits);
            try runner.commit(sequence[len - 1 .. len], hidden_rows[(len - 1) * hidden ..][0..hidden]);
        }

        var accepted = [_]usize{0} ** max_drafts;
        var total = [_]usize{0} ** max_drafts;
        for (0..draft_generated) |step| {
            for (0..counts[step]) |i| {
                const at = seeds[step] + 1 + i;
                if (at >= len) break;
                total[i] += 1;
                const bin = bin_of_draft[step * max_drafts + i];
                bin_total[bin] += 1;
                if (drafts[step * max_drafts + i] == sequence[at]) {
                    accepted[i] += 1;
                    bin_accepted[bin] += 1;
                }
            }
        }
        std.debug.print("Draft acceptance ({s}, \"{s}\", {d} prompt + {d} greedy):\n", .{ if (use_metal) "metal" else "cpu", text, prompt.len, draft_generated });
        for (accepted, total, 0..) |a, t, depth| {
            const rate: f64 = if (t == 0) 0 else @as(f64, @floatFromInt(a)) / @as(f64, @floatFromInt(t));
            std.debug.print("  depth {d}: {d}/{d} = {d:.1} %\n", .{ depth, a, t, rate * 100 });
        }
        const bin_names = [_][]const u8{ ">= 0.9", "0.7-0.9", "0.5-0.7", "< 0.5" };
        std.debug.print("  p_max bins (draft vs the target's next token):\n", .{});
        for (bin_names, bin_total, bin_accepted) |bin_name, t, a| {
            const rate: f64 = if (t == 0) 0 else @as(f64, @floatFromInt(a)) / @as(f64, @floatFromInt(t));
            std.debug.print("    {s:<8} {d}/{d} = {d:.1} %\n", .{ bin_name, a, t, rate * 100 });
        }
        const millis = @as(f64, @floatFromInt(propose_ns)) / std.time.ns_per_ms;
        std.debug.print("  drafts {d}, propose {d:.3} ms total, {d:.3} ms/position; device vs host p_max worst |diff| {e:.2}; block workspace {d} bytes.\n", .{ proposed, millis, millis / @as(f64, @floatFromInt(@max(proposed, 1))), pmax_worst, runner.bytes() });
    }
}

/// Writes the native prediction-block trace for the pinned `Hello,` tokens
/// (`p0-h.f32`, `p1-h.f32`, `p1-hprev.f32`, `greedy.txt`) so
/// `compare-generation.py --draft` can compare it against the reference's
/// captured rows. The target is stepped live for each position's hidden, so
/// this is an independent capture, not a replay of the pinned `hprev`.
fn draftTrace(comptime spec: Spec, alloc: std.mem.Allocator, io: std.Io, mapped: *inference.weights.Mapped, use_metal: bool, directory: []const u8) !void {
    const Family = spec.Family;
    const hidden = 5120;
    const tokens = spec.draft.?.tokens;
    var backend: ?inference.metal.Backend = null;
    defer if (backend) |*b| b.deinit();
    if (use_metal) {
        var diagnostic: [8192]u8 = @splat(0);
        backend = inference.metal.Backend.init(alloc, &diagnostic) catch |err| {
            std.debug.print("{s}\n", .{std.mem.sliceTo(&diagnostic, 0)});
            return err;
        };
    }
    const binding = try Family.bind(alloc, &mapped.document);
    const capacity = tokens.len + max_drafts + 2;
    var runner: DraftRunner(spec) = if (backend) |*b|
        .{ .metal = try Family.Plan.init(alloc, b, mapped.view(), binding, capacity, 8, .f32, false, true) }
    else
        .{ .cpu = try Family.Runtime.init(alloc, mapped.view(), binding, capacity, false, true) };
    defer runner.deinit();
    const logits = try alloc.alloc(f32, spec.vocabulary);
    defer alloc.free(logits);
    const zeros = try alloc.alloc(f32, hidden);
    defer alloc.free(zeros);
    @memset(zeros, 0);
    const h_prev = try alloc.alloc(f32, hidden);
    defer alloc.free(h_prev);
    var greedy: [2]u32 = undefined;

    try runner.step(tokens[0], logits);
    @memcpy(h_prev, runner.lastHidden());
    try runner.forward(zeros, tokens[0], 0, logits, null);
    try writeFloats(io, directory, "p0-h.f32", runner.blockHidden());
    greedy[0] = argmax(logits);

    try runner.step(tokens[1], logits);
    try runner.forward(h_prev, tokens[1], 1, logits, null);
    try writeFloats(io, directory, "p1-h.f32", runner.blockHidden());
    try writeFloats(io, directory, "p1-hprev.f32", h_prev);
    greedy[1] = argmax(logits);

    var path: [4096]u8 = undefined;
    const gpath = try std.fmt.bufPrint(&path, "{s}/greedy.txt", .{directory});
    const file = try std.Io.Dir.cwd().createFile(io, gpath, .{});
    defer file.close(io);
    var buffer: [128]u8 = undefined;
    var writer = file.writer(io, &buffer);
    try writer.interface.print("{d}\n{d}\n", .{ greedy[0], greedy[1] });
    try writer.interface.flush();
    std.debug.print("Draft trace ({s}) written to {s}: greedy {d}, {d}.\n", .{ if (use_metal) "metal" else "cpu", directory, greedy[0], greedy[1] });
}

fn writeFloats(io: std.Io, directory: []const u8, name: []const u8, values: []const f32) !void {
    var path: [4096]u8 = undefined;
    const full = try std.fmt.bufPrint(&path, "{s}/{s}", .{ directory, name });
    const file = try std.Io.Dir.cwd().createFile(io, full, .{});
    defer file.close(io);
    var buffer: [4096]u8 = undefined;
    var writer = file.writer(io, &buffer);
    for (values) |value| try writer.interface.writeInt(u32, @bitCast(value), .little);
    try writer.interface.flush();
}

fn compareDraft(label: []const u8, expected: []const u8, actual: []const f32, max_abs_bound: f64, rel_rms_bound: f64) !void {
    var max_abs: f64 = 0;
    var sum_sq: f64 = 0;
    var ref_sq: f64 = 0;
    for (actual, 0..) |a, i| {
        const e: f32 = std.mem.bytesToValue(f32, expected[i * 4 ..][0..4]);
        const d = @abs(@as(f64, e) - a);
        max_abs = @max(max_abs, d);
        sum_sq += d * d;
        ref_sq += @as(f64, e) * e;
    }
    const rel_rms = @sqrt(sum_sq / ref_sq);
    std.debug.print("Draft block ({s}): max abs {e:.3}, relative RMS {e:.3} (bounds {e:.0} / {e:.0})\n", .{ label, max_abs, rel_rms, max_abs_bound, rel_rms_bound });
    if (!(max_abs <= max_abs_bound) or !(rel_rms <= rel_rms_bound)) return error.DraftBlockMismatch;
}

/// The Gemma 4 assistant head against the pinned trace: the target consumes
/// `Hello, world` (9259, 236764, 1902) one token at a time, and before each
/// token the head runs at that token's proposal position with the previous
/// position's target hidden (the pinned `hprev` rows). Both the head's
/// `h_next` and its greedy token must match, and a `propose` seeded from the
/// committed prefix must return the pinned draft.
fn gemmaDraftTrace(alloc: std.mem.Allocator, io: std.Io, model_path: []const u8, draft_path: []const u8, use_metal: bool) !void {
    const Family = inference.models.gemma4.family;
    var mapped = try inference.weights.Mapped.open(alloc, io, model_path);
    defer mapped.deinit(io);
    var draft_mapped = try inference.weights.Mapped.open(alloc, io, draft_path);
    defer draft_mapped.deinit(io);
    var binding = try Family.bind(alloc, &mapped.document);
    binding.draft = try Family.bindDraft(alloc, &draft_mapped.document, draft_mapped.view(), mapped.view(), &binding);
    const tokens = [3]u32{ 9259, 236764, 1902 };
    const greedy = [2]u32{ 2613, 236764 };
    const hprev_files = [2][]const u8{ @embedFile("src/models/fixtures/gemma4-mtp/token-1-mtp-hprev.f32"), @embedFile("src/models/fixtures/gemma4-mtp/token-2-mtp-hprev.f32") };
    const h_files = [2][]const u8{ @embedFile("src/models/fixtures/gemma4-mtp/token-1-mtp-h.f32"), @embedFile("src/models/fixtures/gemma4-mtp/token-2-mtp-h.f32") };
    const hidden = 3840;
    const h_prev = try alloc.alloc(f32, hidden);
    defer alloc.free(h_prev);
    const h_out = try alloc.alloc(f32, hidden);
    defer alloc.free(h_out);
    var runtime = try Family.Runtime.init(alloc, mapped.view(), binding, 32, false, true);
    defer runtime.deinit();
    // The plan must go before the backend it borrows (LIFO defers: register
    // the backend's first).
    var gpu: ?*inference.metal.Backend = null;
    defer if (gpu) |b| {
        b.deinit();
        alloc.destroy(b);
    };
    var plan: ?Family.Plan = null;
    defer if (plan) |*p| p.deinit();
    if (use_metal) {
        var diagnostic: [8192]u8 = @splat(0);
        gpu = try alloc.create(inference.metal.Backend);
        gpu.?.* = inference.metal.Backend.init(alloc, &diagnostic) catch |err| {
            std.debug.print("{s}\n", .{std.mem.sliceTo(&diagnostic, 0)});
            return err;
        };
        plan = try Family.Plan.init(alloc, gpu.?, mapped.view(), binding, 32, 32, .f32, false, true);
    }
    for (0..2) |row| {
        for (h_prev, 0..) |*v, j| v.* = std.mem.bytesToValue(f32, hprev_files[row][j * 4 ..][0..4]);
        // The row's proposal position is `row + 1`, so the target consumes
        // tokens `0 .. row` first: its cache must hold those rows, exactly as
        // it does in the driver when `draft()` runs.
        try runtime.step(tokens[row], null, null);
        var token_greedy: u32 = 0;
        try runtime.draftForwardTrace(h_prev, tokens[row + 1], row + 1, h_out, &token_greedy, null);
        try compareDraft(if (row == 0) "gemma cpu position 1" else "gemma cpu position 2", h_files[row], h_out, 1e-2, 1e-4);
        if (token_greedy != greedy[row]) return error.DraftGreedyMismatch;
        if (plan) |*p| {
            try p.step(tokens[row], null, null, null, null, null);
            var metal_greedy: u32 = 0;
            try p.draftForwardTrace(h_prev, tokens[row + 1], row + 1, h_out, &metal_greedy, null);
            try compareDraft(if (row == 0) "gemma metal position 1" else "gemma metal position 2", h_files[row], h_out, 1e-2, 1e-4);
            if (metal_greedy != greedy[row]) return error.DraftGreedyMismatch;
        }
    }
    // `propose` from the committed prefix equals the pinned draft at row 2:
    // the prompt's hidden rows come from `prefill`, as the loop's prompt
    // commit supplies them.
    var proposed: [4]u32 = undefined;
    const prompt_hidden = try alloc.alloc(f32, 2 * hidden);
    defer alloc.free(prompt_hidden);
    var fresh = try Family.Runtime.init(alloc, mapped.view(), binding, 32, false, true);
    defer fresh.deinit();
    try fresh.prefill(tokens[0..2], null, prompt_hidden, null);
    try fresh.commit(tokens[0..2], prompt_hidden);
    const count = try fresh.propose(tokens[2], &proposed, 0);
    if (count < 1 or proposed[0] != greedy[1]) return error.DraftGreedyMismatch;
    std.debug.print("Gemma assistant draft check passed ({s}): both rows match the pinned trace; propose returns {d}.\n", .{ if (use_metal) "cpu and metal" else "cpu", proposed[0] });
}

fn argmax(values: []const f32) u32 {
    var best: usize = 0;
    for (values, 0..) |v, i| {
        if (v > values[best]) best = i;
    }
    return @intCast(best);
}

/// The Muse Glimmer DFlash drafter against the pinned trace: the target
/// consumes `The capital of France is` (954, 7963, 323, 11698, 373) one token
/// at a time, the drafter's encoder injects every position's five target
/// layer-input residuals, and at each of positions 1 and 2 a 16-row noise
/// block is decoded. The five residual rows, the encoder output of positions
/// 0 and 1, the first four rows of each block's final hidden, and the greedy
/// draft of every row must match the reference (`--dflash-draft` in
/// scripts/reference-generation.cpp).
fn museDraftTrace(alloc: std.mem.Allocator, io: std.Io, model_path: []const u8, draft_path: []const u8, use_metal: bool, directory: []const u8) !void {
    const Family = inference.models.muse_glimmer.family;
    const D = inference.models.dflash;
    var mapped = try inference.weights.Mapped.open(alloc, io, model_path);
    defer mapped.deinit(io);
    var draft_mapped = try inference.weights.Mapped.open(alloc, io, draft_path);
    defer draft_mapped.deinit(io);
    var binding = try Family.bind(alloc, &mapped.document);
    binding.draft = try Family.bindDraft(alloc, &draft_mapped.document, draft_mapped.view(), mapped.view(), &binding);
    // The reference's proposal never removed its block rows from the cache
    // between the two positions under test, so a capacity past them keeps
    // the same visible set.
    const capacity = 32;
    var runtime = try Family.Runtime.init(alloc, mapped.view(), binding, capacity, false, true);
    defer runtime.deinit();
    // The Metal plan runs the same pinned rows with F32 caches, the
    // precision the reference trace was captured with.
    var gpu: ?*inference.metal.Backend = null;
    defer if (gpu) |b| {
        b.deinit();
        alloc.destroy(b);
    };
    var plan: ?Family.Plan = null;
    defer if (plan) |*p| p.deinit();
    if (use_metal) {
        var diagnostic: [8192]u8 = @splat(0);
        gpu = try alloc.create(inference.metal.Backend);
        gpu.?.* = inference.metal.Backend.init(alloc, &diagnostic) catch |err| {
            std.debug.print("{s}\n", .{std.mem.sliceTo(&diagnostic, 0)});
            return err;
        };
        plan = try Family.Plan.init(alloc, gpu.?, mapped.view(), binding, capacity, 32, .f32, false, true);
    }
    const tokens = [5]u32{ 954, 7963, 323, 11698, 373 };
    const hidden = try alloc.alloc(f32, D.hidden_width);
    defer alloc.free(hidden);
    const encoder = try alloc.alloc(f32, D.embedding);
    defer alloc.free(encoder);
    const pinned_row = try alloc.alloc(f32, D.hidden_width);
    defer alloc.free(pinned_row);
    const block = try alloc.alloc(f32, D.block_size * D.embedding);
    defer alloc.free(block);
    var greedy: [D.block_size]u32 = undefined;
    var label_buffer: [96]u8 = undefined;
    // The Metal capture runs the generic F32 tile, whose accumulation order
    // differs from the reference decode; its relative RMS still pins the
    // slots (measured ~1.4e-6 against the CPU's ~4.9e-7), and the relaxed
    // max-abs bound covers the residual stream's magnitudes of hundreds.
    const residual_max_abs: f64 = if (use_metal) 5e-3 else 1e-3;
    const residual_rel_rms: f64 = 1e-5;
    // The encoder's `fc` runs the shipped half-operand tile over residual
    // rows of magnitudes in the hundreds, so its F16 input rounding sits an
    // order above the CPU reference (measured 1.4e-4 relative RMS on Metal
    // against 2.6e-7 on the CPU); the bound still catches a wiring error,
    // and the greedy rows pin the behaviour.
    const encoder_rel_rms: f64 = if (use_metal) 2e-3 else 1e-4;

    const residual_files = [2][D.block_count][]const u8{
        .{
            @embedFile("src/models/fixtures/muse-dflash/token-0-inp-0.f32"),
            @embedFile("src/models/fixtures/muse-dflash/token-0-inp-1.f32"),
            @embedFile("src/models/fixtures/muse-dflash/token-0-inp-2.f32"),
            @embedFile("src/models/fixtures/muse-dflash/token-0-inp-3.f32"),
            @embedFile("src/models/fixtures/muse-dflash/token-0-inp-4.f32"),
        },
        .{
            @embedFile("src/models/fixtures/muse-dflash/token-1-inp-0.f32"),
            @embedFile("src/models/fixtures/muse-dflash/token-1-inp-1.f32"),
            @embedFile("src/models/fixtures/muse-dflash/token-1-inp-2.f32"),
            @embedFile("src/models/fixtures/muse-dflash/token-1-inp-3.f32"),
            @embedFile("src/models/fixtures/muse-dflash/token-1-inp-4.f32"),
        },
    };
    const encoder_files = [2][]const u8{
        @embedFile("src/models/fixtures/muse-dflash/token-0-encoder.f32"),
        @embedFile("src/models/fixtures/muse-dflash/token-1-encoder.f32"),
    };
    const block_files = [2][]const u8{
        @embedFile("src/models/fixtures/muse-dflash/token-1-block-h.f32"),
        @embedFile("src/models/fixtures/muse-dflash/token-2-block-h.f32"),
    };
    const pinned_greedy = @embedFile("src/models/fixtures/muse-dflash/dflash-greedy.txt");

    // Position 0: the target's residual capture, then the encoder over the
    // same pinned rows (so a capture error is not charged to the encoder).
    // The CPU reference always runs; `--metal` runs the plan's protocol over
    // the same pinned rows, and the trace directory receives the plan's rows.
    try runtime.prefill(tokens[0..1], null, hidden, null);
    try compareResiduals("cpu residuals position 0", residual_files[0], hidden, residual_max_abs, residual_rel_rms);
    if (plan) |*p| {
        try museStepCapture(p, gpu.?, tokens[0..1], hidden);
        try compareResiduals("metal residuals position 0", residual_files[0], hidden, residual_max_abs, residual_rel_rms);
    }
    try assemblePinned(residual_files[0], pinned_row);
    try runtime.draftEncodeTrace(pinned_row, encoder);
    try compareDraft("cpu encoder position 0", encoder_files[0], encoder, 1e-2, encoder_rel_rms);
    if (plan) |*p| {
        try p.draftEncodeTrace(pinned_row, encoder);
        try compareDraft("metal encoder position 0", encoder_files[0], encoder, 1e-2, encoder_rel_rms);
    }
    try writeFloats(io, directory, "token-0-encoder.f32", encoder);
    try runtime.commit(tokens[0..1], hidden);
    if (plan) |*p| try p.commit(tokens[0..1], hidden);

    for (0..2) |row| {
        const label_position = row + 1;
        const block_label: []const u8 = if (row == 0) "block position 1" else "block position 2";
        const greedy_label: []const u8 = if (row == 0) "greedy position 1" else "greedy position 2";
        try runtime.draftBlockTrace(tokens[label_position], D.block_size, block, &greedy, null);
        // The block's residual stream magnifies the reference backends' own
        // spread (its CPU quantizes activations, its Metal does not; the two
        // differ by 2-5 % per layer here). The native rows sit 4.4e-3 from
        // the pinned Metal ones, inside that spread; the 30 greedy rows pin
        // the block's behaviour exactly.
        try compareDraft(
            try std.fmt.bufPrint(&label_buffer, "cpu {s}", .{block_label}),
            block_files[row],
            block[0 .. 4 * D.embedding],
            2e-2,
            1e-2,
        );
        try compareGreedy(
            try std.fmt.bufPrint(&label_buffer, "cpu {s}", .{greedy_label}),
            pinned_greedy,
            row,
            greedy[1..D.block_size],
        );
        if (plan) |*p| {
            try p.draftBlockTrace(tokens[label_position], D.block_size, block, &greedy, null);
            {
                var name: [64]u8 = undefined;
                const text = try std.fmt.bufPrint(&name, "token-{d}-block-h.f32", .{label_position});
                try writeFloats(io, directory, text, block[0 .. 4 * D.embedding]);
            }
            try compareDraft(
                try std.fmt.bufPrint(&label_buffer, "metal {s}", .{block_label}),
                block_files[row],
                block[0 .. 4 * D.embedding],
                2e-2,
                1e-2,
            );
            try compareGreedy(
                try std.fmt.bufPrint(&label_buffer, "metal {s}", .{greedy_label}),
                pinned_greedy,
                row,
                greedy[1..D.block_size],
            );
        } else {
            var name: [64]u8 = undefined;
            const text = try std.fmt.bufPrint(&name, "token-{d}-block-h.f32", .{label_position});
            try writeFloats(io, directory, text, block[0 .. 4 * D.embedding]);
        }
        if (row == 0) {
            // The next position: capture token 1's residuals and inject them,
            // then the second proposal reads both injected rows.
            try runtime.prefill(tokens[1..2], null, hidden, null);
            try compareResiduals("cpu residuals position 1", residual_files[1], hidden, residual_max_abs, residual_rel_rms);
            if (plan) |*p| {
                try museStepCapture(p, gpu.?, tokens[1..2], hidden);
                try compareResiduals("metal residuals position 1", residual_files[1], hidden, residual_max_abs, residual_rel_rms);
            }
            try writeFloats(io, directory, "token-1-residuals.f32", hidden);
            try assemblePinned(residual_files[1], pinned_row);
            try runtime.draftEncodeTrace(pinned_row, encoder);
            try compareDraft("cpu encoder position 1", encoder_files[1], encoder, 1e-2, encoder_rel_rms);
            if (plan) |*p| {
                try p.draftEncodeTrace(pinned_row, encoder);
                try compareDraft("metal encoder position 1", encoder_files[1], encoder, 1e-2, encoder_rel_rms);
            }
            try runtime.commit(tokens[1..2], hidden);
            if (plan) |*p| try p.commit(tokens[1..2], hidden);
        }
    }
    std.debug.print("Muse DFlash draft check passed ({s}): residuals, encoder, two blocks, and {d} greedy rows match the pinned trace.\n", .{ if (use_metal) "cpu and metal" else "cpu", 2 * (D.block_size - 1) });
}

/// The pinned residual rows are the reference's per-token target decode; the
/// chunked path's half-operand tiles round their inputs to F16 and would mask
/// the drafter's own numerics, so the trace's capture runs the same schedule
/// and capture path through the generic F32 tile.
fn museStepCapture(plan: anytype, gpu: *inference.metal.Backend, tokens: []const u32, hidden: []f32) !void {
    gpu.generic_only = true;
    defer gpu.generic_only = false;
    try plan.prefill(tokens, null, null, null, null, hidden, null);
}

/// Concatenates the five pinned residual rows into one `commit` row.
fn assemblePinned(files: [inference.models.dflash.block_count][]const u8, out: []f32) !void {
    const width = inference.models.dflash.embedding;
    if (out.len != files.len * width) return error.InvalidShape;
    for (files, 0..) |file, slot| {
        for (out[slot * width ..][0..width], 0..) |*value, i| {
            value.* = std.mem.bytesToValue(f32, file[i * 4 ..][0..4]);
        }
    }
}

/// The captured residual row is `block_count` slots of `embedding` values in
/// `target_layers` order; each is pinned as its own file.
fn compareResiduals(label: []const u8, files: [inference.models.dflash.block_count][]const u8, row: []const f32, max_abs: f64, rel_rms: f64) !void {
    const width = inference.models.dflash.embedding;
    for (files, 0..) |file, slot| {
        var name: [96]u8 = undefined;
        const text = try std.fmt.bufPrint(&name, "{s} slot {d}", .{ label, slot });
        try compareDraft(text, file, row[slot * width ..][0..width], max_abs, rel_rms);
    }
}

/// `rows` greedy draft tokens starting at proposal `proposal` of the pinned
/// file (one line per block row, `block_size - 1` rows per proposal).
fn compareGreedy(label: []const u8, pinned: []const u8, proposal: usize, actual: []const u32) !void {
    const per_proposal = inference.models.dflash.block_size - 1;
    var lines = std.mem.tokenizeScalar(u8, pinned, '\n');
    var index: usize = 0;
    while (lines.next()) |line| : (index += 1) {
        if (index < proposal * per_proposal or index >= (proposal + 1) * per_proposal) continue;
        const expected = try std.fmt.parseInt(u32, std.mem.trim(u8, line, " \r"), 10);
        const row = index - proposal * per_proposal;
        if (actual[row] != expected) {
            std.debug.print("{s}: row {d} greedy {d}, expected {d}\n", .{ label, row, actual[row], expected });
            return error.DraftGreedyMismatch;
        }
    }
    std.debug.print("{s}: {d} greedy rows match.\n", .{ label, per_proposal });
}

/// A 70-token prompt through per-token steps and through `prefill` with
/// 32-token chunks (two full chunks and a partial one, crossing chunk
/// boundaries inside attention and any recurrent state) must agree on the
/// final logits within the recorded tolerance, and a chunk that would
/// overflow the context must be refused whole. The tile matmul accumulates
/// in a different order than the matvec and rounds its operands to
/// half, so bit-identity is not expected; the observed maximum is printed
/// for the record. Chunks of 48 and 64 repeat the comparison through the
/// 64×64 tiles (a partial and a full token tile); chunks of at most 32
/// tokens take the 32×32 tiles.
fn checkChunkedPrefill(comptime spec: Spec, alloc: std.mem.Allocator, b: *inference.metal.Backend, view: inference.weights.View, binding: spec.Family.Binding, expected: []f32, actual: []f32) !void {
    const Plan = spec.Family.Plan;
    var tokens: [70]u32 = undefined;
    var seed: u32 = 12345;
    for (&tokens) |*t| {
        seed = seed *% 1664525 +% 1013904223;
        t.* = seed % 150000;
    }
    const bounds = boundsOf(spec, binding);
    var stepped = try Plan.init(alloc, b, view, binding, 128, 32, .f32, false, false);
    defer stepped.deinit();
    for (tokens, 0..) |t, i| try stepped.step(t, if (i + 1 == tokens.len) expected else null, null, null, null, null);
    for ([_]usize{ 64, 48 }) |chunk| {
        var big = try Plan.init(alloc, b, view, binding, 128, chunk, .f32, false, false);
        defer big.deinit();
        try big.prefill(&tokens, actual, null, null, null, null, null);
        try compareChunked("chunk", chunk, expected, actual, bounds.chunk_max_abs, bounds.chunk_rel_rms);
    }
    var chunked = try Plan.init(alloc, b, view, binding, 128, 32, .f32, false, false);
    defer chunked.deinit();
    try chunked.prefill(&tokens, actual, null, null, null, null, null);
    if (chunked.state.position != tokens.len or stepped.state.position != tokens.len) return error.PositionMismatch;
    try compareChunked("chunk", 32, expected, actual, bounds.chunk_max_abs, bounds.chunk_rel_rms);
    // The same comparison through the generic F32 tiles and matvecs
    // (`generic_only`): what remains is summation order alone, so this
    // separates the half-operand rounding of the specialized tiles (the
    // family's `chunk_*` tolerance) from any error in the chunk schedule.
    {
        const generic_expected = try alloc.alloc(f32, spec.vocabulary);
        defer alloc.free(generic_expected);
        b.generic_only = true;
        defer b.generic_only = false;
        var generic_stepped = try Plan.init(alloc, b, view, binding, 128, 32, .f32, false, false);
        defer generic_stepped.deinit();
        for (tokens, 0..) |t, i| try generic_stepped.step(t, if (i + 1 == tokens.len) generic_expected else null, null, null, null, null);
        var generic_chunked = try Plan.init(alloc, b, view, binding, 128, 32, .f32, false, false);
        defer generic_chunked.deinit();
        try generic_chunked.prefill(&tokens, actual, null, null, null, null, null);
        try compareChunked("F32 tiles, chunk", 32, generic_expected, actual, 5e-3, 2e-4);
    }
    // The same 70 tokens through an F16 cache, stepped and chunked,
    // against the F32 stepped logits (the session holds half the bytes).
    var half_stepped = try Plan.init(alloc, b, view, binding, 128, 32, .f16, false, false);
    defer half_stepped.deinit();
    if (half_stepped.state.bytes() >= stepped.state.bytes()) return error.HalfCacheNotSmaller;
    for (tokens, 0..) |t, i| try half_stepped.step(t, if (i + 1 == tokens.len) actual else null, null, null, null, null);
    try compareChunked("F16 KV stepped", 1, expected, actual, bounds.half_max_abs, bounds.half_rel_rms);
    var half_chunked = try Plan.init(alloc, b, view, binding, 128, 32, .f16, false, false);
    defer half_chunked.deinit();
    try half_chunked.prefill(&tokens, actual, null, null, null, null, null);
    try compareChunked("F16 KV chunk", 32, expected, actual, bounds.half_max_abs, bounds.half_rel_rms);
    // 60 more tokens do not fit the remaining 58 positions: refused before any work.
    if (chunked.prefill(tokens[0..60], null, null, null, null, null, null)) |_| return error.ExpectedContextFull else |err| if (err != error.ContextFull) return err;
    if (chunked.state.position != tokens.len) return error.PositionMismatch;
    try chunked.prefill(tokens[0..58], null, null, null, null, null, null);
    if (chunked.state.position != 128) return error.PositionMismatch;
    // A per-layer observer is a per-token contract: prefill refuses it.
    var context: u8 = 0;
    if (stepped.prefill(tokens[0..1], null, null, null, null, null, .{ .context = &context, .layer = cancel })) |_| return error.ExpectedInvalidShape else |err| if (err != error.InvalidShape) return err;
}

/// The distance between two logit rows: max abs, relative RMS, and each
/// argmax. Shared by the chunked-prefill and recovery comparisons.
const Difference = struct {
    max_abs: f64,
    rel_rms: f64,
    arg_expected: usize,
    arg_actual: usize,

    fn of(expected: []const f32, actual: []const f32) Difference {
        var max_abs: f64 = 0;
        var sum_sq: f64 = 0;
        var ref_sq: f64 = 0;
        for (expected, actual) |e, a| {
            const d = @abs(@as(f64, e) - a);
            max_abs = @max(max_abs, d);
            sum_sq += d * d;
            ref_sq += @as(f64, e) * e;
        }
        var arg_expected: usize = 0;
        var arg_actual: usize = 0;
        for (expected, 0..) |v, i| if (v > expected[arg_expected]) {
            arg_expected = i;
        };
        for (actual, 0..) |v, i| if (v > actual[arg_actual]) {
            arg_actual = i;
        };
        return .{ .max_abs = max_abs, .rel_rms = @sqrt(sum_sq / ref_sq), .arg_expected = arg_expected, .arg_actual = arg_actual };
    }
    fn within(self: Difference, max_abs_bound: f64, rel_rms_bound: f64) bool {
        return self.max_abs <= max_abs_bound and self.rel_rms <= rel_rms_bound and self.arg_expected == self.arg_actual;
    }
};

/// Final logits of a chunked prefill against the stepped ones: max abs,
/// relative RMS, and the greedy choice, within the family's recorded bound.
fn compareChunked(label: []const u8, chunk: usize, expected: []const f32, actual: []const f32, max_abs_bound: f64, rel_rms_bound: f64) !void {
    const d = Difference.of(expected, actual);
    std.debug.print("Prefill (70 tokens, {s} {d}) vs per-token F32 steps: max abs {e:.3}, relative RMS {e:.3}, argmax {d}/{d} (bounds {e:.0} / {e:.0})\n", .{ label, chunk, d.max_abs, d.rel_rms, d.arg_expected, d.arg_actual, max_abs_bound, rel_rms_bound });
    if (!d.within(max_abs_bound, rel_rms_bound)) return error.ChunkedPrefillMismatch;
}

/// A recovered verify batch's correction logits against the sequential run's:
/// the same metrics, recorded at the family's chunk-versus-step bound.
fn compareRecovery(rows: usize, expected: []const f32, actual: []const f32, max_abs_bound: f64, rel_rms_bound: f64) !void {
    const d = Difference.of(expected, actual);
    std.debug.print("Recovery (batch {d} rows) vs sequential F32 steps: max abs {e:.3}, relative RMS {e:.3}, argmax {d}/{d} (bounds {e:.0} / {e:.0})\n", .{ rows, d.max_abs, d.rel_rms, d.arg_expected, d.arg_actual, max_abs_bound, rel_rms_bound });
    if (!d.within(max_abs_bound, rel_rms_bound)) return error.RecoveryMismatch;
}

/// The Qwen3-VL projector against the pinned oracle: the synthetic fixture
/// image's feature rows (the reference's Metal projector, llama.cpp
/// `7620399f5`, 2026-09-22) on the executor under test, then the first
/// greedy tokens of `describe this image` through the language model with
/// the rows substituted for the image span, against the pinned eight.
fn visionCheck(alloc: std.mem.Allocator, io: std.Io, model_path: []const u8, projector_path: []const u8, use_metal: bool) !void {
    const vision = inference.vision;
    const qwen3vl = vision.qwen3vl;
    var projector = try inference.weights.Mapped.open(alloc, io, projector_path);
    defer projector.deinit(io);
    const binding = try qwen3vl.bind(alloc, &projector.document);
    var source = try vision.image.decodePpm(alloc, @embedFile("src/vision/fixtures/synthetic-96x64.ppm"));
    defer source.deinit(alloc);
    const grid = qwen3vl.gridFor(.{ .width = source.width, .height = source.height });
    const target: vision.preprocess.Size = .{ .width = grid.width_patches * qwen3vl.patch, .height = grid.height_patches * qwen3vl.patch };
    const resized = try vision.preprocess.resizeLetterbox(alloc, source, target);
    defer alloc.free(resized);
    var patches = try vision.preprocess.patches(alloc, resized, target, .{ .patch = qwen3vl.patch, .merge = qwen3vl.merge, .mean = binding.mean, .std = binding.std });
    defer patches.deinit(alloc);
    const expected = @embedFile("src/vision/fixtures/qwen3vl-synthetic/features.f32");
    const rows = grid.tokens();
    if (expected.len != rows * qwen3vl.output_width * 4) return error.FixtureMismatch;
    const features = try alloc.alloc(f32, rows * qwen3vl.output_width);
    defer alloc.free(features);
    const started = std.Io.Clock.awake.now(io);
    if (use_metal) {
        var diagnostic: [8192]u8 = @splat(0);
        var backend = inference.metal.Backend.init(alloc, &diagnostic) catch |err| {
            std.debug.print("{s}\n", .{std.mem.sliceTo(&diagnostic, 0)});
            return err;
        };
        defer backend.deinit();
        var plan = try qwen3vl.Plan.init(alloc, &backend, projector.view(), &binding);
        defer plan.deinit();
        try plan.encode(patches, features);
    } else {
        var runtime = try qwen3vl.Runtime.init(alloc, projector.view(), &binding);
        defer runtime.deinit();
        try runtime.encode(patches, features);
    }
    const seconds = @as(f64, @floatFromInt(started.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / std.time.ns_per_s;
    std.debug.print("Projector ({s}): {d} patches -> {d} rows in {d:.3} s.\n", .{ if (use_metal) "metal" else "cpu", grid.patches(), rows, seconds });
    // The bounds: the reference's own CPU and Metal projectors differ by
    // 0.36 max abs / 4.6e-3 relative RMS on this fixture (vision.md).
    try compareRows("projector rows", expected, features, 0.5, 1e-2);

    // The language-model path, isolated from the projector's tolerance: the
    // pinned feature rows are prefilled into the image span and the first
    // eight greedy tokens must equal the oracle's (`greedy.txt`).
    const pinned = try alloc.alloc(f32, rows * qwen3vl.output_width);
    defer alloc.free(pinned);
    for (pinned, 0..) |*v, i| v.* = std.mem.bytesToValue(f32, expected[i * 4 ..][0..4]);
    var eng = try inference.engine.Engine.open(alloc, io, model_path, if (use_metal) .metal else .cpu, 64, .f32, null, .none);
    defer eng.deinit();
    const refs = [_]inference.profiles.ImageRef{.{ .width_tokens = grid.widthTokens(), .height_tokens = grid.heightTokens() }};
    const prompt = try eng.render(&.{.{ .role = .user, .content = "describe this image", .images = &refs }}, &.{}, .off);
    defer alloc.free(prompt);
    const tokens = try eng.encode(prompt);
    defer alloc.free(tokens);
    const spans = try eng.locateImageSpans(tokens, &.{.{ .width_tokens = grid.widthTokens(), .height_tokens = grid.heightTokens(), .features = pinned }});
    defer alloc.free(spans);
    const logits = try alloc.alloc(f32, eng.vocab.tokens.len);
    defer alloc.free(logits);
    const generated = try alloc.alloc(u32, 8);
    defer alloc.free(generated);
    var sampler = try inference.sampling.Sampler.init(0, .{ .temperature = 0, .top_p = 1, .top_k = 0, .min_p = 0, .presence_penalty = 0, .repetition_penalty = 1 });
    const outcome = try inference.engine.runLoop(&eng, tokens, 8, &sampler, null, .{}, .{ .spans = spans, .features = pinned }, logits, &.{}, generated, null, null);
    const greedy_text = @embedFile("src/vision/fixtures/qwen3vl-synthetic/greedy.txt");
    var it = std.mem.tokenizeScalar(u8, greedy_text, '\n');
    var mismatch = false;
    std.debug.print("Vision greedy ({s}):", .{if (use_metal) "metal" else "cpu"});
    for (generated[0..outcome.timing.generated_tokens]) |got| {
        const want = std.fmt.parseInt(u32, it.next() orelse break, 10) catch break;
        std.debug.print(" {d}{s}", .{ got, if (got == want) "" else "!" });
        if (got != want) mismatch = true;
    }
    std.debug.print("\n", .{});
    if (mismatch) return error.VisionGreedyMismatch;
}

fn compareRows(label: []const u8, expected: []const u8, actual: []const f32, max_abs_bound: f64, rel_rms_bound: f64) !void {
    var max_abs: f64 = 0;
    var sum_sq: f64 = 0;
    var ref_sq: f64 = 0;
    for (actual, 0..) |a, i| {
        const e: f32 = std.mem.bytesToValue(f32, expected[i * 4 ..][0..4]);
        const d = @abs(@as(f64, e) - a);
        max_abs = @max(max_abs, d);
        sum_sq += d * d;
        ref_sq += @as(f64, e) * e;
    }
    const rel_rms = @sqrt(sum_sq / ref_sq);
    std.debug.print("{s}: max abs {e:.3}, relative RMS {e:.3} (bounds {e:.0} / {e:.0})\n", .{ label, max_abs, rel_rms, max_abs_bound, rel_rms_bound });
    if (!(max_abs <= max_abs_bound) or !(rel_rms <= rel_rms_bound)) return error.VisionMismatch;
}
