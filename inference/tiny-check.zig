//! Tiny-model checks: generated random-weight models through the real load
//! and forward paths, no download (docs/models/README.md § Tiny models).
//! Built against the `tiny_models` instance of the inference module.
//!
//! As a test binary (part of `zig build test`): each family loads on the CPU,
//! with no GPU, and its forward is finite and deterministic; a family with
//! no executor loads and is refused at the executor; a hybrid family's
//! session splits, restores, rewinds, and resets bit for bit.
//! As an executable (`zig build test-tiny -Dmetal=true -- [variant] [seed]`):
//! every variant in every encoding its Metal plan executes, against the CPU
//! reference, with a negative control per variant.
const std = @import("std");
const inference = @import("inference");
const check_stats = @import("check_stats.zig");
const tiny = inference.models.tiny;
const Engine = inference.engine.Engine;
const Precision = inference.session.Precision;
const Allocator = std.mem.Allocator;

/// Long enough to pass the tiny window of 8, under a single chunk.
const prompt = "Hello world, it's tiny. 12345 helloWorld and more of it, again";
const capacity = 64;
const default_seed = 0x7a11;

/// Writes the bytes into `dir`; the path the engine opens is `prefix` joined
/// with the name. Caller owns the path.
fn writeFile(gpa: Allocator, io: std.Io, dir: std.Io.Dir, prefix: []const u8, name: []const u8, bytes: []const u8) ![]u8 {
    try dir.writeFile(io, .{ .sub_path = name, .data = bytes });
    return std.fs.path.join(gpa, &.{ prefix, name });
}

fn promptTokens(gpa: Allocator, engine: *const Engine) ![]u32 {
    const ids = try engine.encoder.encode(gpa, prompt, false, .{});
    errdefer gpa.free(ids);
    if (ids.len < 12 or ids.len > capacity) return error.InvalidShape;
    return ids;
}

fn expectFinite(values: []const f32) !void {
    for (values) |x| if (!std.math.isFinite(x)) return error.NonFiniteResult;
}

fn same(a: []const f32, b: []const f32) bool {
    return std.mem.eql(u8, std.mem.sliceAsBytes(a), std.mem.sliceAsBytes(b));
}

/// Loads a file through `Engine.open`, then asserts that two rows of forward
/// are finite, differ, and repeat bit for bit after a reset. Two rows, because
/// `zig build test` runs the reference unoptimized: the second row reads an
/// attention cache and a recurrent state; the check goes past the window.
fn cpuRun(gpa: Allocator, io: std.Io, path: []const u8) !void {
    var engine = try Engine.open(gpa, io, path, .cpu, capacity, .f32, null, .none);
    defer engine.deinit();
    const tokens = try promptTokens(gpa, &engine);
    defer gpa.free(tokens);
    const logits = try gpa.alloc(f32, engine.vocab.tokens.len);
    defer gpa.free(logits);
    const next = try gpa.alloc(f32, logits.len);
    defer gpa.free(next);
    try engine.model.step(tokens[0], logits, null, null, null, null);
    try expectFinite(logits);
    try engine.model.step(tokens[1], next, null, null, null, null);
    try expectFinite(next);
    try std.testing.expect(!same(logits, next));
    engine.model.reset();
    try engine.model.step(tokens[0], next, null, null, null, null);
    if (!same(logits, next)) return error.NondeterministicForward;
}

test {
    _ = check_stats;
}

test "each family loads on the CPU, and its forward is finite and deterministic" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const prefix = try std.fs.path.join(gpa, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    defer gpa.free(prefix);
    for (&tiny.variants) |*variant| {
        if (variant.cycle.len != 0 or variant.forward != .both) continue;
        errdefer std.debug.print("tiny variant {s}\n", .{variant.name});
        const bytes = try tiny.generate(gpa, variant, variant.encodings[0], default_seed);
        defer gpa.free(bytes);
        const path = try writeFile(gpa, io, tmp.dir, prefix, variant.name, bytes);
        defer gpa.free(path);
        try cpuRun(gpa, io, path);
    }
}

/// Parses, binds, and builds the vocabulary through `Engine.open`; the
/// executor's refusal is the only failure allowed.
fn loadRun(gpa: Allocator, io: std.Io, path: []const u8, backend: inference.engine.Backend) !void {
    // By name: which refusals a registry can return depends on the
    // families it holds. Only Metal may refuse for want of a plan.
    var engine = Engine.open(gpa, io, path, backend, capacity, .f32, null, .none) catch |err| {
        const name = @errorName(err);
        if (std.mem.eql(u8, name, "InferenceUnavailable")) return;
        if (backend == .metal and std.mem.eql(u8, name, "MetalPlanUnavailable")) return;
        return err;
    };
    engine.deinit();
    return error.UnexpectedExecutor;
}

test "a family without a Metal plan loads, and runs on the CPU when it has a forward, in each encoding" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const prefix = try std.fs.path.join(gpa, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    defer gpa.free(prefix);
    for (&tiny.variants) |*variant| {
        if (variant.forward == .both) continue;
        for (variant.encodings) |encoding| {
            errdefer std.debug.print("tiny variant {s}, encoding {d}\n", .{ variant.name, encoding });
            const bytes = try tiny.generate(gpa, variant, encoding, default_seed);
            defer gpa.free(bytes);
            const path = try writeFile(gpa, io, tmp.dir, prefix, variant.name, bytes);
            defer gpa.free(path);
            switch (variant.forward) {
                .none => try loadRun(gpa, io, path, .cpu),
                .cpu => try cpuRun(gpa, io, path),
                .both => unreachable,
            }
        }
    }
}

/// Inkling 2's hybrid session through its CPU forward over `tokens`, each
/// case bit for bit against one uninterrupted run: the head skipped on the
/// row before `split`, a snapshot at `split` restored, a checkpoint there
/// rewound and replayed, and a reset. Every step runs the whole looped
/// schedule, so each crosses the core's pass boundaries. The runtime is
/// opened directly, with the checkpoint region the engine sizes only for a
/// verify.
fn hybridRun(gpa: Allocator, io: std.Io, path: []const u8, tokens: []const u32, split: usize) !void {
    const family = inference.models.inkling2;
    var mapped = try inference.weights.Mapped.open(gpa, io, path);
    defer mapped.deinit(io);
    const binding = try family.bind(gpa, &mapped.document);
    var runtime = try family.Runtime.init(gpa, io, mapped.view(), binding, capacity, true, false);
    defer runtime.deinit();
    const state = &runtime.state;
    const run = struct {
        fn f(r: *family.Runtime, ids: []const u32, logits: ?[]f32) !void {
            for (ids, 0..) |id, i| try r.step(id, if (i + 1 == ids.len) logits else null, null);
        }
    }.f;
    const whole = try gpa.alloc(f32, family.vocabulary);
    defer gpa.free(whole);
    const again = try gpa.alloc(f32, family.vocabulary);
    defer gpa.free(again);
    const half = split;
    try run(&runtime, tokens, whole);
    try expectFinite(whole);

    runtime.reset();
    try run(&runtime, tokens[0..half], null);
    try run(&runtime, tokens[half..], again);
    if (!same(whole, again)) return error.SkippedHeadDiffers;

    runtime.reset();
    try run(&runtime, tokens[0..half], null);
    var snap = try state.snapshot(gpa);
    defer snap.deinit();
    try run(&runtime, tokens[half..], again);
    try state.restore(&snap);
    // The window restores above row 0, so the replay reads past `base`.
    for (state.layers[0..inference.models.inkling2.block_count]) |layer| {
        if (layer.attention.keys.window != null) try std.testing.expect(layer.attention.keys.base > 0);
    }
    @memset(again, 0);
    try run(&runtime, tokens[half..], again);
    if (!same(whole, again)) return error.RestoreDiffers;

    runtime.reset();
    try run(&runtime, tokens[0..half], null);
    try state.checkpoint();
    try run(&runtime, tokens[half..], again);
    // A position alone cannot move the convolution history back.
    try std.testing.expectError(error.RecurrentStateNotRewindable, state.truncate(half));
    try std.testing.expectError(error.RecurrentStateNotRewindable, state.rewindTo(half));
    try state.rewind();
    try std.testing.expectEqual(half, state.position);
    @memset(again, 0);
    try run(&runtime, tokens[half..], again);
    if (!same(whole, again)) return error.ReplayDiffers;

    runtime.reset();
    try run(&runtime, tokens, again);
    if (!same(whole, again)) return error.ResetDiffers;
}

test "Inkling 2's hybrid CPU session splits, restores, rewinds, and resets exactly" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const prefix = try std.fs.path.join(gpa, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    defer gpa.free(prefix);
    const variant = for (&tiny.variants) |*v| {
        if (std.mem.eql(u8, v.name, "inkling2")) break v;
    } else return error.UnknownVariant;
    const bytes = try tiny.generate(gpa, variant, variant.encodings[0], default_seed);
    defer gpa.free(bytes);
    const path = try writeFile(gpa, io, tmp.dir, prefix, variant.name, bytes);
    defer gpa.free(path);
    // Past the tiny window of 8 and relative extent of 16; the split past
    // the window and its margin, so a restore moves the window's base.
    var tokens: [32]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast((i * 97 + 3) % inference.models.inkling2.vocabulary);
    try hybridRun(gpa, io, path, &tokens, 26);
}

test "Inkling 2's CPU runtime frees on every failed allocation" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const prefix = try std.fs.path.join(gpa, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    defer gpa.free(prefix);
    const family = inference.models.inkling2;
    const variant = for (&tiny.variants) |*v| {
        if (std.mem.eql(u8, v.name, "inkling2")) break v;
    } else return error.UnknownVariant;
    const bytes = try tiny.generate(gpa, variant, variant.encodings[0], default_seed);
    defer gpa.free(bytes);
    const path = try writeFile(gpa, io, tmp.dir, prefix, variant.name, bytes);
    defer gpa.free(path);
    var mapped = try inference.weights.Mapped.open(gpa, io, path);
    defer mapped.deinit(io);
    const binding = try family.bind(gpa, &mapped.document);
    try inference.alloc_check.checkAll(gpa, struct {
        fn f(a: Allocator, m: *const inference.weights.Mapped, b: family.Binding) !void {
            var runtime = try family.Runtime.init(a, std.testing.io, m.view(), b, 16, true, false);
            runtime.deinit();
        }
    }.f, .{ &mapped, binding });
}

test "a tiny file is a function of its seed" {
    const gpa = std.testing.allocator;
    const variant = tiny.find("saluki").?;
    const a = try tiny.generate(gpa, variant, 18, default_seed);
    defer gpa.free(a);
    const b = try tiny.generate(gpa, variant, 18, default_seed);
    defer gpa.free(b);
    const c = try tiny.generate(gpa, variant, 18, default_seed + 1);
    defer gpa.free(c);
    try std.testing.expectEqualSlices(u8, a, b);
    try std.testing.expect(!std.mem.eql(u8, a, c));
}

/// A generated file's vocabulary and encoder, the path `Engine.open` takes.
const Tokenizer = struct {
    vocabulary: inference.vocabulary.Vocabulary,
    encoder: inference.tokenizer.Encoder,

    fn init(self: *Tokenizer, gpa: Allocator, bytes: []const u8) !void {
        var reader: std.Io.Reader = .fixed(bytes);
        var doc = try inference.gguf.parse(gpa, &reader, bytes.len, .{});
        defer doc.deinit();
        self.vocabulary = try inference.vocabulary.load(gpa, doc, bytes[0..@intCast(doc.directory_bytes)], .{});
        errdefer self.vocabulary.deinit();
        self.encoder = try inference.tokenizer.Encoder.init(gpa, &self.vocabulary);
    }
    fn deinit(self: *Tokenizer) void {
        self.encoder.deinit();
        self.vocabulary.deinit();
    }
    fn expectRoundTrip(self: *const Tokenizer, gpa: Allocator, text: []const u8) !void {
        const ids = try self.encoder.encode(gpa, text, false, .{});
        defer gpa.free(ids);
        const decoded = try inference.bpe.decode(gpa, &self.vocabulary, ids, true, .{});
        defer gpa.free(decoded);
        try std.testing.expectEqualStrings(text, decoded);
    }
    fn id(self: *const Tokenizer, text: []const u8) u32 {
        return self.vocabulary.tokenId(text).?;
    }
};

test "each tokenizer path decodes what it encoded" {
    const gpa = std.testing.allocator;
    for ([_][]const u8{ "qwen35", "muse-glimmer", "gemma4-dense", "inkling2" }) |name| {
        const variant = tiny.find(name).?;
        const bytes = try tiny.generate(gpa, variant, variant.encodings[0], default_seed);
        defer gpa.free(bytes);
        var tokenizer: Tokenizer = undefined;
        try tokenizer.init(gpa, bytes);
        defer tokenizer.deinit();
        for ([_][]const u8{ prompt, "double  spaces\n\n\ttabs\r\n", "h\u{e9}llo \u{1f600} \u{4e16}\u{754c}" }) |text|
            try tokenizer.expectRoundTrip(gpa, text);
    }
}

test "the GPT-4o splitter keeps its cuts through the loaded vocabulary" {
    const gpa = std.testing.allocator;
    const variant = tiny.find("muse-glimmer").?;
    const bytes = try tiny.generate(gpa, variant, variant.encodings[0], default_seed);
    defer gpa.free(bytes);
    var tokenizer: Tokenizer = undefined;
    try tokenizer.init(gpa, bytes);
    defer tokenizer.deinit();
    try std.testing.expectEqualStrings("llama4", tokenizer.vocabulary.pre);
    const t = &tokenizer;
    const cases = [_]struct { text: []const u8, ids: []const u32 }{
        // Merges inside one piece reach the whole word, across the prefix space.
        .{ .text = "Hello world", .ids = &.{ t.id("Hello"), t.id("\u{120}world") } },
        // Digits cut every three: `3 4` ranks first and must not fire.
        .{ .text = "12345", .ids = &.{ t.id("123"), t.id("4"), t.id("5") } },
        // A capital opens a new word: `o W` ranks second and must not fire.
        .{ .text = "oW", .ids = &.{ t.id("o"), t.id("W") } },
        // The contraction stays with its word.
        .{ .text = "it's", .ids = &.{ t.id("i"), t.id("t"), t.id("'s") } },
    };
    for (cases) |case| {
        const ids = try t.encoder.encode(gpa, case.text, false, .{});
        defer gpa.free(ids);
        try std.testing.expectEqualSlices(u32, case.ids, ids);
    }
    const marked = try t.encoder.encode(gpa, "o" ++ tiny.end_marker, true, .{});
    defer gpa.free(marked);
    try std.testing.expectEqualSlices(u32, &.{ t.id("o"), t.vocabulary.eos.? }, marked);
}

/// How far Metal may sit from the CPU reference.
const Deviation = struct {
    max_abs: f64 = 0,
    rel_rms: f64 = 0,

    fn of(actual: []const f32, reference: []const f32) Deviation {
        var largest: check_stats.Worst = .{};
        var error_energy: f64 = 0;
        var energy: f64 = 0;
        for (actual, reference) |a, r| {
            const difference = @as(f64, a) - r;
            largest.observe(difference);
            error_energy += difference * difference;
            energy += @as(f64, r) * r;
        }
        return .{ .max_abs = largest.shown(), .rel_rms = @sqrt(error_energy / @max(energy, 1e-30)) };
    }
    /// The worse of each field; NaN stays NaN.
    fn merge(a: Deviation, b: Deviation) Deviation {
        var max_abs: check_stats.Worst = .{};
        var rel_rms: check_stats.Worst = .{};
        for ([_]Deviation{ a, b }) |d| {
            max_abs.observe(d.max_abs);
            rel_rms.observe(d.rel_rms);
        }
        return .{ .max_abs = max_abs.shown(), .rel_rms = rel_rms.shown() };
    }
    fn within(self: Deviation, bound: Deviation) bool {
        return self.max_abs <= bound.max_abs and self.rel_rms <= bound.rel_rms;
    }
};

/// Bounds per family: about three times the worst of three seeds, every
/// encoding (docs/models/README.md § Tiny models has the measurements). The
/// stepped row is the schedule: F32 activations on both sides, so what is
/// left is rounding. The chunked row adds the half-operand tiles', and the
/// F16 and Q8_0 rows their caches'.
const Bounds = struct {
    step: Deviation,
    chunk: Deviation,
    half: Deviation,
    q8: Deviation,
    /// The chunked prefill runs on the generic F32 tiles: where the router's
    /// top-k turns the half tiles' rounding into swaps the size of a
    /// corruption, the row then holds the chunk schedule alone.
    generic_chunk: bool = false,
};

const qwen_bounds: Bounds = .{
    .step = .{ .max_abs = 6e-5, .rel_rms = 1.5e-5 },
    .chunk = .{ .max_abs = 4e-2, .rel_rms = 1e-2 },
    .half = .{ .max_abs = 3e-2, .rel_rms = 1e-2 },
    .q8 = .{ .max_abs = 4e-2, .rel_rms = 1e-2 },
};
const muse_bounds: Bounds = .{
    .step = .{ .max_abs = 5e-6, .rel_rms = 1e-5 },
    .chunk = .{ .max_abs = 3e-3, .rel_rms = 5e-3 },
    .half = .{ .max_abs = 2e-3, .rel_rms = 3e-3 },
    .q8 = .{ .max_abs = 3e-2, .rel_rms = 5e-2 },
};
const gemma_bounds: Bounds = .{
    .step = .{ .max_abs = 1.5e-4, .rel_rms = 3e-5 },
    .chunk = .{ .max_abs = 2e-2, .rel_rms = 6e-3 },
    .half = .{ .max_abs = 3e-2, .rel_rms = 6e-3 },
    .q8 = .{ .max_abs = 1.5e-1, .rel_rms = 5e-2 },
};
/// The router's top-k is discontinuous: rounding that swaps an expert moves
/// the logits by far more than the rounding, so these bound one swap.
const gemma_experts_bounds: Bounds = .{
    .step = .{ .max_abs = 4e-4, .rel_rms = 1.2e-4 },
    .chunk = .{ .max_abs = 2e0, .rel_rms = 5e-1 },
    .half = .{ .max_abs = 2e0, .rel_rms = 5e-1 },
    .q8 = .{ .max_abs = 2e0, .rel_rms = 5e-1 },
};

/// The stepped plan is the schedule's rounding, and so is the chunked one
/// on the generic tiles; the caches are held to a reference rounded the
/// same way (a Q8_0 chunk's through its F16 expansion too), and the
/// rounding still crosses a code or the router's top-k boundary on some
/// seeds, so they bound one swap.
const inkling_bounds: Bounds = .{
    .step = .{ .max_abs = 3e-5, .rel_rms = 7e-6 },
    .chunk = .{ .max_abs = 3e-5, .rel_rms = 7e-6 },
    .generic_chunk = true,
    .half = .{ .max_abs = 2e0, .rel_rms = 5e-1 },
    .q8 = .{ .max_abs = 2e0, .rel_rms = 5e-1 },
};

fn boundsOf(variant: *const tiny.Variant) *const Bounds {
    if (std.mem.eql(u8, variant.name, "inkling2")) return &inkling_bounds;
    if (std.mem.eql(u8, variant.name, "gemma4-experts")) return &gemma_experts_bounds;
    if (std.mem.startsWith(u8, variant.name, "gemma4")) return &gemma_bounds;
    if (std.mem.startsWith(u8, variant.name, "muse")) return &muse_bounds;
    return &qwen_bounds;
}

/// The reference's logits at every prompt position, row after row.
const Reference = struct {
    tokens: []u32,
    rows: []f32,
    vocabulary: usize,

    /// `rounding` other than F32 rounds the cache rows as a lossy cache
    /// does, in a family whose runtime can (`cache_rounding`); null when
    /// it cannot. `half_reads` adds the F16 expansion a chunk's attention
    /// reads a Q8_0 cache through.
    fn init(gpa: Allocator, io: std.Io, path: []const u8, rounding: Precision, half_reads: bool) !?Reference {
        var engine = try Engine.open(gpa, io, path, .cpu, capacity, .f32, null, .none);
        defer engine.deinit();
        if (rounding != .f32) switch (engine.model.exec) {
            inline else => |*e| {
                if (!@hasField(@TypeOf(e.cpu), "cache_rounding")) return null;
                e.cpu.cache_rounding = rounding;
                e.cpu.cache_half_reads = half_reads;
            },
        };
        const tokens = try promptTokens(gpa, &engine);
        errdefer gpa.free(tokens);
        const vocabulary = engine.vocab.tokens.len;
        const rows = try gpa.alloc(f32, tokens.len * vocabulary);
        errdefer gpa.free(rows);
        for (tokens, 0..) |token, i| try engine.model.step(token, rows[i * vocabulary ..][0..vocabulary], null, null, null, null);
        try expectFinite(rows);
        return .{ .tokens = tokens, .rows = rows, .vocabulary = vocabulary };
    }
    fn deinit(self: Reference, gpa: Allocator) void {
        gpa.free(self.tokens);
        gpa.free(self.rows);
    }
    fn row(self: Reference, index: usize) []const f32 {
        return self.rows[index * self.vocabulary ..][0..self.vocabulary];
    }
};

/// Counts a prefill's progress reports. Everything that shares the GPU with a
/// generation runs on them, so every family's every prefill path must report.
const Beats = struct {
    reached: usize = 0,

    fn observer(self: *Beats) inference.observer.Observer {
        return .{ .context = self, .progress = progress };
    }

    fn progress(context: *anyopaque, p: inference.observer.Progress) anyerror!void {
        const self: *Beats = @ptrCast(@alignCast(context));
        self.reached = p.position;
    }

    fn expectReached(self: *const Beats, tokens: usize) !void {
        if (self.reached != tokens) return error.PrefillUnreported;
    }
};

const Run = struct { stepped: Deviation = .{}, chunked: Deviation = .{}, greedy_agrees: bool = true };

/// Where a generic-tile chunked run splits its prompt: 10 rows from 0, then
/// 5 (the verify attention) and 17 (the materialized one) whose windows
/// start past position 0.
const split_ends = [_]usize{ 10, 15, 32 };

/// The prompt on Metal, stepped and then chunked, against the reference. A
/// GPU argmax that disagrees with the reference's fails the row.
fn metalRun(gpa: Allocator, io: std.Io, path: []const u8, reference: Reference, chunk_reference: ?Reference, kv: Precision, generic_chunk: bool) !Run {
    var engine = try Engine.open(gpa, io, path, .metal, capacity, kv, null, .none);
    defer engine.deinit();
    const logits = try gpa.alloc(f32, reference.vocabulary);
    defer gpa.free(logits);
    var run: Run = .{};
    for (reference.tokens, 0..) |token, i| {
        var greedy: u32 = undefined;
        try engine.model.step(token, logits, &greedy, null, null, null);
        try expectFinite(logits);
        run.stepped = run.stepped.merge(.of(logits, reference.row(i)));
        run.greedy_agrees = run.greedy_agrees and greedy == std.mem.indexOfMax(f32, reference.row(i));
    }
    engine.model.reset();
    const gpu = engine.model.gpu().?;
    gpu.generic_only = generic_chunk;
    defer gpu.generic_only = false;
    var beats: Beats = .{};
    if (generic_chunk) {
        if (reference.tokens.len <= split_ends[split_ends.len - 1]) return error.PromptTooShort;
        var start: usize = 0;
        for (split_ends ++ [_]usize{reference.tokens.len}) |end| {
            try engine.model.prefill(reference.tokens[start..end], if (end == reference.tokens.len) logits else null, null, null, null, null, beats.observer());
            start = end;
        }
    } else try engine.model.prefill(reference.tokens, logits, null, null, null, null, beats.observer());
    try expectFinite(logits);
    run.chunked = .of(logits, (chunk_reference orelse reference).row(reference.tokens.len - 1));
    // Each call reports its own rows.
    try beats.expectReached(if (generic_chunk) reference.tokens.len - split_ends[split_ends.len - 1] else reference.tokens.len);
    engine.model.reset();
    beats = .{};
    engine.model.prefillVision(reference.tokens, &.{}, &.{}, logits, null, null, null, beats.observer()) catch |err| switch (err) {
        error.VisionUnsupported => return run,
        else => return err,
    };
    try beats.expectReached(reference.tokens.len);
    return run;
}

/// A run of 64 bytes in the middle of `blk.0.ffn_up.weight` changed (of
/// `blk.0.attn_q.weight` in a family without one), a block or so of one
/// early matrix: what a decoder wrong in a corner would do. The stepped
/// bound must catch it.
fn corruptBlock(gpa: Allocator, bytes: []u8) !void {
    var reader: std.Io.Reader = .fixed(bytes);
    var doc = try inference.gguf.parse(gpa, &reader, bytes.len, .{});
    defer doc.deinit();
    for ([_][]const u8{ "blk.0.ffn_up.weight", "blk.0.attn_q.weight" }) |name| for (doc.tensors) |tensor| {
        if (!std.mem.eql(u8, tensor.name, name)) continue;
        const middle: usize = @intCast(doc.data_offset + tensor.offset + tensor.bytes / 2);
        for (bytes[middle..][0..64]) |*byte| byte.* ^= 0x10;
        return;
    };
    return error.MissingTensor;
}

fn checkVariant(gpa: Allocator, io: std.Io, out: *std.Io.Writer, dir: std.Io.Dir, prefix: []const u8, variant: *const tiny.Variant, seed: u64) !bool {
    const bounds = boundsOf(variant);
    var passed = true;
    for (variant.encodings, 0..) |encoding, index| {
        const encoding_name = if (variant.cycle.len != 0) "mix" else inference.encoding.layout(encoding).?.name;
        const bytes = try tiny.generate(gpa, variant, encoding, seed);
        defer gpa.free(bytes);
        const path = try writeFile(gpa, io, dir, prefix, "model.gguf", bytes);
        defer gpa.free(path);
        const reference = (try Reference.init(gpa, io, path, .f32, false)).?;
        defer reference.deinit(gpa);
        const run = metalRun(gpa, io, path, reference, null, .f32, bounds.generic_chunk) catch |err| {
            try out.print("{s:<16} {s:>8} {t}\n", .{ variant.name, encoding_name, err });
            try out.flush();
            passed = false;
            continue;
        };
        const ok = run.stepped.within(bounds.step) and run.chunked.within(bounds.chunk) and run.greedy_agrees;
        passed = passed and ok;
        try out.print("{s:<16} {s:>8} {e:>9.2} {e:>9.2} {e:>9.2} {e:>9.2} {s}\n", .{ variant.name, encoding_name, run.stepped.max_abs, run.stepped.rel_rms, run.chunked.max_abs, run.chunked.rel_rms, if (ok) "ok" else "OUT OF BOUNDS" });
        if (index == 0) {
            // The two lossy caches, once per variant.
            for ([_]Precision{ .f16, .q8_0 }, [_]Deviation{ bounds.half, bounds.q8 }) |kv, bound| {
                // Against a reference whose cache rounds the same way, where
                // the family can: Inkling 2's clipped keys amplify a cache's
                // rounding past anything a bound against F32 could hold.
                const rounded = try Reference.init(gpa, io, path, kv, false);
                defer if (rounded) |r| r.deinit(gpa);
                // A chunk past the few-row pass reads a Q8_0 cache through
                // an F16 expansion; decode reads the codes.
                const expanded = kv == .q8_0 and rounded != null and reference.tokens.len > inference.metal.Backend.verify_max_rows;
                const half_reads = if (expanded) try Reference.init(gpa, io, path, kv, true) else null;
                defer if (half_reads) |r| r.deinit(gpa);
                const cached = try metalRun(gpa, io, path, rounded orelse reference, half_reads, kv, bounds.generic_chunk);
                const worst = cached.stepped.merge(cached.chunked);
                const fine = worst.within(bound);
                passed = passed and fine;
                try out.print("{s:<16} {s:>8} {e:>9.2} {e:>9.2} {s:>19} {s}\n", .{ variant.name, @tagName(kv), worst.max_abs, worst.rel_rms, "", if (fine) "ok" else "OUT OF BOUNDS" });
            }
            try corruptBlock(gpa, bytes);
            const mutated = try writeFile(gpa, io, dir, prefix, "mutated.gguf", bytes);
            defer gpa.free(mutated);
            const control = try metalRun(gpa, io, mutated, reference, null, .f32, bounds.generic_chunk);
            const seen = !control.stepped.within(bounds.step);
            passed = passed and seen;
            try out.print("{s:<16} {s:>8} {e:>9.2} {e:>9.2} {s:>19} {s}\n", .{ variant.name, "control", control.stepped.max_abs, control.stepped.rel_rms, "", if (seen) "seen" else "MISSED" });
        }
        try out.flush();
    }
    return passed;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const only: ?[]const u8 = if (args.len > 1) args[1] else null;
    const seed: u64 = if (args.len > 2) try std.fmt.parseInt(u64, args[2], 0) else default_seed;
    const cwd = std.Io.Dir.cwd();
    const prefix = ".zig-cache/tiny";
    var dir = try cwd.createDirPathOpen(io, prefix, .{});
    defer dir.close(io);
    defer cwd.deleteTree(io, prefix) catch {};
    var buffer: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(io, &buffer);
    const out = &stdout.interface;
    const known = for (&tiny.variants) |*variant| {
        if (only == null or std.mem.eql(u8, only.?, "all") or std.mem.eql(u8, only.?, variant.name)) break true;
    } else false;
    if (!known) {
        try out.print("no variant named {s}; the variants are all", .{only.?});
        for (&tiny.variants) |*variant| try out.print(", {s}", .{variant.name});
        try out.print("\n", .{});
        try out.flush();
        return error.UnknownVariant;
    }
    try out.print("{s:<16} {s:>8} {s:>9} {s:>9} {s:>9} {s:>9}\n", .{ "variant", "encoding", "step abs", "step rms", "chunk abs", "chunk rms" });
    var passed = true;
    var checked: usize = 0;
    for (&tiny.variants) |*variant| {
        if (only) |name| if (!std.mem.eql(u8, name, "all") and !std.mem.eql(u8, name, variant.name)) continue;
        checked += 1;
        if (variant.forward != .both) {
            // Nothing runs on Metal: the file loads up to the executor.
            const bytes = try tiny.generate(gpa, variant, variant.encodings[0], seed);
            defer gpa.free(bytes);
            const path = try writeFile(gpa, io, dir, prefix, "model.gguf", bytes);
            defer gpa.free(path);
            const loaded = if (loadRun(gpa, io, path, .metal)) "loads, no Metal plan" else |err| @errorName(err);
            passed = passed and std.mem.eql(u8, loaded, "loads, no Metal plan");
            try out.print("{s:<16} {s:>8} {s}\n", .{ variant.name, "", loaded });
            try out.flush();
            continue;
        }
        passed = try checkVariant(gpa, io, out, dir, prefix, variant, seed) and passed;
    }
    try out.print("{d} variants checked\n", .{checked});
    try out.flush();
    if (!passed) return error.TinyCheckFailed;
}
