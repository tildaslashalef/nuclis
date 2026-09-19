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
const Spec = struct {
    Family: type,
    vocabulary: usize,
    tokens: [2]u32,
    bounds: Bounds,
    /// The bounds of a mixture-of-experts configuration of the family, when
    /// it has one: discrete routing amplifies the tiles' rounding.
    expert_bounds: ?Bounds = null,
};
const qwen35_spec: Spec = .{ .Family = inference.models.qwen35.family, .vocabulary = inference.models.qwen35_metal.vocabulary, .tokens = .{ 9419, 11 }, .bounds = .{ .chunk_max_abs = 2e-2, .chunk_rel_rms = 1e-3, .half_max_abs = 2e-2, .half_rel_rms = 1e-3 } };
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
    if (layer == 3) return error.Canceled;
}
/// Production-style cancellation through `check`: the fourth boundary, while
/// the GPU plan is still recording, so partially recorded work is what runs.
fn cancelCheck(context: *anyopaque) !void {
    const boundaries: *u8 = @ptrCast(context);
    boundaries.* += 1;
    if (boundaries.* == 4) return error.Canceled;
}

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2 or args.len > 3) return error.ExpectedModelPath;
    const use_metal = args.len == 3;
    if (use_metal and !std.mem.eql(u8, args[2], "--metal")) return error.UnknownOption;
    var mapped = try inference.weights.Mapped.open(alloc, init.io, args[1]);
    defer mapped.deinit(init.io);
    const architecture = mapped.document.string("general.architecture") orelse return error.MissingMetadata;
    switch (try inference.models.select(architecture)) {
        .qwen35 => try run(qwen35_spec, alloc, init.io, &mapped, use_metal),
        .gemma4 => try run(gemma4_spec, alloc, init.io, &mapped, use_metal),
        .@"muse-glimmer" => try run(muse_glimmer_spec, alloc, init.io, &mapped, use_metal),
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
                .metal => |*p| try p.step(token, logits, null, null, observer),
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
                .metal => |*p| try p.prefill(tokens, logits, null, null, observer),
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
        /// recurrent state is replayed, attention alone is truncated.
        fn recover(self: *@This(), accepted: []const u32) !void {
            const at = self.state().checkpoint_position orelse return error.NoCheckpoint;
            if (self.state().hasRecurrent()) {
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
    var first: Model = if (backend) |*b| .{ .metal = try Plan.init(alloc, b, mapped.view(), binding, 16, 16, .f32, true) } else .{ .cpu = try Runtime.init(alloc, mapped.view(), binding, 16, true) };
    defer first.deinit();
    var second: Model = if (backend) |*b| .{ .metal = try Plan.init(alloc, b, mapped.view(), binding, 16, 16, .f32, true) } else .{ .cpu = try Runtime.init(alloc, mapped.view(), binding, 16, true) };
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
        } else |err| if (err != error.Canceled) return err;
        if (first.step(spec.tokens[0], null, null)) |_| {
            return error.ExpectedPoisonedSession;
        } else |err| if (err != error.SessionNotReady) return err;
        first.reset();
        try first.sequence(actual);
        if (!std.mem.eql(f32, expected, actual)) return error.ResetMismatch;
    }
    std.debug.print("Generation check passed ({s}, {s}): two-token logits identical across independent sessions and reset after partial-step cancellation through both the layer and the check callbacks.\n", .{ Family.architecture, if (use_metal) "metal" else "cpu" });
    var other: Model = if (backend) |*b| .{ .metal = try Plan.init(alloc, b, mapped.view(), binding, 8, 4, .f32, true) } else .{ .cpu = try Runtime.init(alloc, mapped.view(), binding, 8, true) };
    defer other.deinit();
    try checkSnapshot(spec, Model, alloc, &first, &second, &other, expected, actual);
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
    if (first.step(word, null, .{ .context = &context, .check = cancelCheck })) |_| return error.ExpectedCancellation else |err| if (err != error.Canceled) return err;
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
    try first.prefill(tokens[1 .. 1 + rows], null, null);
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
    const rewind_start = std.Io.Clock.awake.now(io);
    try first.rewind();
    const rewind_time = rewind_start.durationTo(std.Io.Clock.awake.now(io));
    std.debug.print("Recovery check passed ({d} rows): checkpoint {d:.3} ms, rewind {d:.3} ms, region {d} bytes.\n", .{
        rows, checkpoint_time.toMilliseconds(), rewind_time.toMilliseconds(), first.state().checkpoint_region.len,
    });

    // Rewind without a live checkpoint.
    first.reset();
    if (first.rewind()) |_| return error.ExpectedNoCheckpoint else |err| if (err != error.NoCheckpoint) return err;

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
    if (first.prefill(tokens[1 .. 1 + rows], null, .{ .context = &context, .check = cancelCheck })) |_| return error.ExpectedCancellation else |err| if (err != error.Canceled) return err;
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
    var stepped = try Plan.init(alloc, b, view, binding, 128, 32, .f32, false);
    defer stepped.deinit();
    for (tokens, 0..) |t, i| try stepped.step(t, if (i + 1 == tokens.len) expected else null, null, null, null);
    for ([_]usize{ 64, 48 }) |chunk| {
        var big = try Plan.init(alloc, b, view, binding, 128, chunk, .f32, false);
        defer big.deinit();
        try big.prefill(&tokens, actual, null, null, null);
        try compareChunked("chunk", chunk, expected, actual, bounds.chunk_max_abs, bounds.chunk_rel_rms);
    }
    var chunked = try Plan.init(alloc, b, view, binding, 128, 32, .f32, false);
    defer chunked.deinit();
    try chunked.prefill(&tokens, actual, null, null, null);
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
        var generic_stepped = try Plan.init(alloc, b, view, binding, 128, 32, .f32, false);
        defer generic_stepped.deinit();
        for (tokens, 0..) |t, i| try generic_stepped.step(t, if (i + 1 == tokens.len) generic_expected else null, null, null, null);
        var generic_chunked = try Plan.init(alloc, b, view, binding, 128, 32, .f32, false);
        defer generic_chunked.deinit();
        try generic_chunked.prefill(&tokens, actual, null, null, null);
        try compareChunked("F32 tiles, chunk", 32, generic_expected, actual, 5e-3, 2e-4);
    }
    // The same 70 tokens through an F16 cache, stepped and chunked,
    // against the F32 stepped logits (the session holds half the bytes).
    var half_stepped = try Plan.init(alloc, b, view, binding, 128, 32, .f16, false);
    defer half_stepped.deinit();
    if (half_stepped.state.bytes() >= stepped.state.bytes()) return error.HalfCacheNotSmaller;
    for (tokens, 0..) |t, i| try half_stepped.step(t, if (i + 1 == tokens.len) actual else null, null, null, null);
    try compareChunked("F16 KV stepped", 1, expected, actual, bounds.half_max_abs, bounds.half_rel_rms);
    var half_chunked = try Plan.init(alloc, b, view, binding, 128, 32, .f16, false);
    defer half_chunked.deinit();
    try half_chunked.prefill(&tokens, actual, null, null, null);
    try compareChunked("F16 KV chunk", 32, expected, actual, bounds.half_max_abs, bounds.half_rel_rms);
    // 60 more tokens do not fit the remaining 58 positions: refused before any work.
    if (chunked.prefill(tokens[0..60], null, null, null, null)) |_| return error.ExpectedContextFull else |err| if (err != error.ContextFull) return err;
    if (chunked.state.position != tokens.len) return error.PositionMismatch;
    try chunked.prefill(tokens[0..58], null, null, null, null);
    if (chunked.state.position != 128) return error.PositionMismatch;
    // A per-layer observer is a per-token contract: prefill refuses it.
    var context: u8 = 0;
    if (stepped.prefill(tokens[0..1], null, null, null, .{ .context = &context, .layer = cancel })) |_| return error.ExpectedInvalidShape else |err| if (err != error.InvalidShape) return err;
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
