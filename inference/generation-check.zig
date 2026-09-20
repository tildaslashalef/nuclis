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
        } else if (path == null) {
            path = arg;
        } else return error.UnknownOption;
    }
    const model_path = path orelse return error.ExpectedModelPath;
    var mapped = try inference.weights.Mapped.open(alloc, init.io, model_path);
    defer mapped.deinit(init.io);
    const architecture = mapped.document.string("general.architecture") orelse return error.MissingMetadata;
    switch (try inference.models.select(architecture)) {
        .qwen35 => if (draft_trace) |dir|
            try draftTrace(qwen35_spec, alloc, init.io, &mapped, use_metal, dir)
        else if (draft_stats)
            try draftStats(qwen35_spec, alloc, init.io, &mapped, use_metal)
        else if (speculative_check)
            try speculativeCheck(qwen35_spec, alloc, init.io, &mapped, model_path, use_metal)
        else
            try run(qwen35_spec, alloc, init.io, &mapped, use_metal),
        .gemma4 => if (draft_stats or speculative_check or draft_trace != null) return error.DraftStatsUnsupported else try run(gemma4_spec, alloc, init.io, &mapped, use_metal),
        .@"muse-glimmer" => if (draft_stats or speculative_check or draft_trace != null) return error.DraftStatsUnsupported else try run(muse_glimmer_spec, alloc, init.io, &mapped, use_metal),
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
    const first_draft = use_metal and spec.draft != null;
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
    if (comptime spec.draft != null) {
        var with_draft: Model = if (backend) |*b|
            .{ .metal = try Plan.init(alloc, b, mapped.view(), binding, 16, 16, .f32, true, true) }
        else
            .{ .cpu = try Runtime.init(alloc, mapped.view(), binding, 16, true, true) };
        defer with_draft.deinit();
        try with_draft.sequence(actual);
        if (!std.mem.eql(f32, expected, actual)) return error.DraftLoadedDecodeMismatch;
    }
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
    if (comptime spec.draft != null) {
        try checkDraft(spec, alloc, mapped.view(), binding, spec.draft.?, if (backend) |*b| b else null);
        try draftRecoveryCheck(spec, alloc, if (backend) |*b| b else null, mapped.view(), binding, spec.draft.?);
        if (backend) |*b| try draftBatchCommitCheck(spec, alloc, b, mapped.view(), binding);
    }
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
        try plan.draftForwardHost(zeros, draft.tokens[0], 0, null, logits);
        try compareDraft("metal f32 position 0", p0_h, plan.draft_h.floats()[0..hidden], 2e-2, 1e-3);
        if (argmax(logits) != draft.greedy[0]) return error.DraftGreedyMismatch;
        try plan.draftForwardHost(h_prev, draft.tokens[1], 1, null, logits);
        try compareDraft("metal f32 position 1", p1_h, plan.draft_h.floats()[0..hidden], 2e-2, 1e-3);
        if (argmax(logits) != draft.greedy[1]) return error.DraftGreedyMismatch;

        // The F16 cache's rounding of the block's keys and values at the
        // family's recorded tolerance, the same rows.
        const bounds = spec.bounds;
        var half = try Plan.init(alloc, b, view, binding, 4, 4, .f16, false, true);
        defer half.deinit();
        try half.draftForwardHost(zeros, draft.tokens[0], 0, null, logits);
        try compareDraft("metal f16 position 0", p0_h, half.draft_h.floats()[0..hidden], bounds.half_max_abs, bounds.half_rel_rms);
        if (argmax(logits) != draft.greedy[0]) return error.DraftGreedyMismatch;
        try half.draftForwardHost(h_prev, draft.tokens[1], 1, null, logits);
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
        try verified.verify(&draft.tokens, verify_rows, verify_hidden, null);
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
    try first.forward(zeros, draft.tokens[0], 0, null);
    @memcpy(before, first.blockHidden());
    first.reset();
    try first.forward(zeros, draft.tokens[0], 0, null);
    if (!std.mem.eql(f32, before, first.blockHidden())) return error.DraftResetMismatch;

    // A checkpoint/rewind leaves the block's row rewritable to the same bytes.
    first.reset();
    try first.forward(zeros, draft.tokens[0], 0, null);
    try first.checkpoint();
    try first.forward(h_prev, draft.tokens[1], 1, null);
    @memcpy(before, first.blockHidden());
    try first.rewind();
    try first.forward(h_prev, draft.tokens[1], 1, null);
    if (!std.mem.eql(f32, before, first.blockHidden())) return error.DraftRewindMismatch;

    // Two reset runners propose the same greedy chain.
    var one: [4]u32 = undefined;
    var two: [4]u32 = undefined;
    first.reset();
    second.reset();
    _ = try first.propose(draft.tokens[0], &one);
    _ = try second.propose(draft.tokens[0], &two);
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
        try serial.draftForwardHost(h_prev, token, i, null, null);
    }
    @memcpy(serial.draft_pending_h.floats()[0..hidden], hidden_rows[hidden_rows.len - hidden ..][0..hidden]);

    const seed_token = tokens[count - 1];
    var a: [draft_n]u32 = undefined;
    var c: [draft_n]u32 = undefined;
    const na = try batched.propose(seed_token, &a);
    const nc = try serial.propose(seed_token, &c);
    if (na != nc or !std.mem.eql(u32, a[0..na], c[0..nc])) return error.BatchedCommitDraftMismatch;

    const logits_a = try alloc.alloc(f32, spec.vocabulary);
    defer alloc.free(logits_a);
    const logits_b = try alloc.alloc(f32, spec.vocabulary);
    defer alloc.free(logits_b);
    try batched.draftForwardHost(batched.draft_pending_h.floats()[0..hidden], seed_token, count, null, logits_a);
    try serial.draftForwardHost(serial.draft_pending_h.floats()[0..hidden], seed_token, count, null, logits_b);
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
        const n = try speculative.propose(seed, drafts[0..k]);
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
    const first = try engine.runLoop(&eng, prompt, generated, &sampler, null, .{}, logits, no_candidates, a, null, null);
    resetForRun(&eng);
    const second = try engine.runLoop(&eng, prompt, generated, &sampler, null, .{ .enabled = true, .draft_length = 4 }, logits, no_candidates, b, null, null);
    if (first.timing.generated_tokens != second.timing.generated_tokens) return error.SpeculativeLoopLengthMismatch;
    if (!std.mem.eql(u32, a[0..first.timing.generated_tokens], b[0..second.timing.generated_tokens])) return error.SpeculativeLoopMismatch;
    // Partial acceptance: the run rejected at least one draft, so the
    // correction path ran.
    if (second.timing.accepted_drafts >= second.timing.proposed_drafts) return error.SpeculativeNoPartialAcceptance;
    std.debug.print("Speculative loop passed ({s}): {d} tokens identical to ordinary greedy through engine.runLoop; accepted {d}/{d} drafts.\n", .{ if (use_metal) "metal" else "cpu", first.timing.generated_tokens, second.timing.accepted_drafts, second.timing.proposed_drafts });

    const spec: engine.Speculative = .{ .enabled = true, .draft_length = 4 };

    // Budget inside a batch: a limit below what the first batch would emit.
    resetForRun(&eng);
    const short = try engine.runLoop(&eng, prompt, 3, &sampler, null, spec, logits, no_candidates, a, null, null);
    if (short.stop != .token_budget or short.timing.generated_tokens != 3) return error.SpeculativeBudgetMismatch;
    std.debug.print("Speculative budget passed: stopped at {d} tokens inside a batch.\n", .{short.timing.generated_tokens});

    // EOS inside a batch: the rendered turn ends with the profile's stop
    // token, which speculation must emit and then stop on.
    const rendered = try eng.prompt("Hello,", false, .off);
    defer alloc.free(rendered);
    const eos_tokens = try eng.encode(rendered);
    defer alloc.free(eos_tokens);
    resetForRun(&eng);
    const eos = try engine.runLoop(&eng, eos_tokens, 24, &sampler, null, spec, logits, no_candidates, a, null, null);
    if (eos.stop != .eos) return error.SpeculativeEosMismatch;
    std.debug.print("Speculative EOS passed: stopped after {d} tokens.\n", .{eos.timing.generated_tokens});

    // Cancellation mid-batch: the observer's check fires during the first
    // decode verify (the prompt commit leaves the position at its end), and
    // the loop resets the poisoned session and reports cancellation.
    var canceller = CancelAt{ .eng = &eng, .at = prompt.len };
    resetForRun(&eng);
    const cancelled = try engine.runLoop(&eng, prompt, 24, &sampler, null, spec, logits, no_candidates, a, .{ .context = &canceller, .check = CancelAt.check }, null);
    if (cancelled.stop != .cancelled) return error.SpeculativeCancellationMismatch;
    if (eng.model.session().position != 0) return error.SpeculativeCancellationNotReset;
    std.debug.print("Speculative cancellation passed: a cancelled verify reset the session.\n", .{});

    // Context limit at a batch: a session with room for the prompt and two
    // tokens cannot hold a verify batch plus its correction.
    var small = try engine.Engine.open(alloc, io, model_path, backend, prompt.len + 2, .f32, null, .embedded);
    defer small.deinit();
    const full = try engine.runLoop(&small, prompt, 24, &sampler, null, spec, logits, no_candidates, a, null, null);
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
        /// landing in `blockHidden`.
        fn forward(self: *@This(), h_prev: []const f32, token: u32, position: usize, logits: ?[]f32) !void {
            switch (self.*) {
                .cpu => |*r| try r.draftForward(h_prev, token, position, logits),
                .metal => |*p| try p.draftForwardHost(h_prev, token, position, null, logits),
            }
        }
        fn blockHidden(self: *@This()) []const f32 {
            return switch (self.*) {
                .cpu => |*r| r.draft_h,
                .metal => |*p| p.draft_h.floats()[0..hidden],
            };
        }
        fn propose(self: *@This(), token: u32, out: []u32) !usize {
            return switch (self.*) {
                .cpu => |*r| r.propose(token, out),
                .metal => |*p| p.propose(token, out),
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
                .metal => |*p| try p.verify(tokens, rows, h_rows, null),
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
            const k = try runner.propose(next, drafts[j * max_drafts ..][0..max_drafts]);
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
                if (drafts[step * max_drafts + i] == sequence[at]) accepted[i] += 1;
            }
        }
        std.debug.print("Draft acceptance ({s}, \"{s}\", {d} prompt + {d} greedy):\n", .{ if (use_metal) "metal" else "cpu", text, prompt.len, draft_generated });
        for (accepted, total, 0..) |a, t, depth| {
            const rate: f64 = if (t == 0) 0 else @as(f64, @floatFromInt(a)) / @as(f64, @floatFromInt(t));
            std.debug.print("  depth {d}: {d}/{d} = {d:.1} %\n", .{ depth, a, t, rate * 100 });
        }
        const millis = @as(f64, @floatFromInt(propose_ns)) / std.time.ns_per_ms;
        std.debug.print("  drafts {d}, propose {d:.3} ms total, {d:.3} ms/position; block workspace {d} bytes.\n", .{ proposed, millis, millis / @as(f64, @floatFromInt(@max(proposed, 1))), runner.bytes() });
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
    try runner.forward(zeros, tokens[0], 0, logits);
    try writeFloats(io, directory, "p0-h.f32", runner.blockHidden());
    greedy[0] = argmax(logits);

    try runner.step(tokens[1], logits);
    try runner.forward(h_prev, tokens[1], 1, logits);
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

fn argmax(values: []const f32) u32 {
    var best: usize = 0;
    for (values, 0..) |v, i| {
        if (v > values[best]) best = i;
    }
    return @intCast(best);
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
