//! Repeated prefill/decode measurements on one loaded engine. Every sample is
//! reported; means are computed over the measured (non-warmup) samples only.
//! Rates that cannot be measured are absent, never fabricated.
const std = @import("std");
const inference = @import("inference");
const engine = @import("engine.zig");
const generate = @import("generate.zig");
const expert_counts = @import("expert_counts.zig");
const config = @import("config.zig");
const kit = @import("ui.zig");
const interrupt = @import("interrupt.zig");
const prefix_cache = inference.prefix_cache;

/// What only this command reads from the command line. The model, backend,
/// and context come resolved from the configuration (`config.Resolved`);
/// the output budget (32 unless flagged) and the sampling flags never come
/// from the file, so a run is comparable from its command line alone.
pub const Options = struct {
    prompt: ?[]const u8 = null,
    prompt_file: ?[]const u8 = null,
    /// A JSON array of token IDs fed as the prompt without tokenization, so a
    /// run can reproduce a reference measurement's exact input.
    prompt_tokens: ?[]const u8 = null,
    raw: bool = false,
    repeat: ?usize = null,
    warmup: ?usize = null,
    seed: ?u64 = null,
    /// Per-kernel GPU timing (Metal only). Changes how work is recorded, so
    /// the whole-run rates of a profiled run are not comparable with unprofiled ones.
    profile: bool = false,
    /// Keep each run's prompt in the profile instead of pausing it out: with
    /// a short `--max-tokens`, prefill's per-kernel time.
    profile_prefill: bool = false,
    /// Run the norm pairs the fused kernels replace instead of the fused
    /// ones: the interleaved control that measures the fusions.
    unfused_norms: bool = false,
    /// Tally the experts every batch routes to (`expert_counts`) and report
    /// them; a mixture of experts on Metal only.
    expert_counts: bool = false,
    /// A `.gputrace` path: after the runs, one more decode step is recorded
    /// into it for Xcode's Metal debugger. Metal only.
    capture: ?[]const u8 = null,
    /// A directory of saved prefixes (`inference.prefix_cache`): the prompt less
    /// its last token is prefilled once, saved, and restored before every
    /// later run, which then feeds only the last token (prefill rates are
    /// omitted).
    prefix_cache: ?[]const u8 = null,
    /// Instead of decoding, time `--max-tokens` verify batches of this many
    /// rows per run from the prompt's depth (`inference.engine.forcedBatch`).
    verify_rows: ?usize = null,
    /// Drafts each verify batch accepts, below `verify_rows`; default 0.
    accept: ?usize = null,
    /// The drafter's proposal threshold and chain-product floor for the
    /// speculative runs, in place of the draft file's verdict (else
    /// `inference.engine.draft_stop`): tuning knobs, not user settings.
    draft_p_min: ?f32 = null,
    draft_chain_min: ?f32 = null,
    /// Print every Metal pipeline's compiled limits (`kernelStats`) instead
    /// of benchmarking; no model is loaded.
    kernel_stats: bool = false,
};

/// Compiles the Metal kernels and prints each pipeline's limits: a thread
/// limit under 1024 is the compiler saying the kernel's registers cap its
/// occupancy, the cheapest register-pressure signal a kernel change has.
pub fn kernelStats(alloc: std.mem.Allocator, json: bool, writer: *std.Io.Writer, ui: kit.Ui, diag: *config.Diagnostic) !void {
    const Backend = inference.metal.Backend;
    var diagnostic: [512]u8 = @splat(0);
    var backend = Backend.init(alloc, &diagnostic) catch |err| {
        // Widened: a Metal build's `init` cannot return MetalNotEnabled.
        if (@as(anyerror, err) == error.MetalNotEnabled)
            diag.set("this binary was built without the Metal backend: rebuild with `zig build -Dmetal=true` (or `make metal`)", .{})
        else
            diag.set("{s}", .{std.mem.sliceTo(&diagnostic, 0)});
        return err;
    };
    defer backend.deinit();
    var stats: [Backend.pipeline_count]inference.metal.PipelineStats = undefined;
    var limited: usize = 0;
    for (&stats, 0..) |*s, i| {
        s.* = backend.pipelineStats(i);
        limited += @intFromBool(s.max_threads < 1024);
    }
    if (json) {
        try std.json.Stringify.value(.{ .pipelines = &stats }, .{ .whitespace = .indent_2 }, writer);
        return writer.writeByte('\n');
    }
    var arena_state = std.heap.ArenaAllocator.init(alloc);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    try kit.blocks.header(writer, ui, "Metal pipelines", try std.fmt.allocPrint(arena, "{d} kernels · {d} below 1024 threads per threadgroup", .{ stats.len, limited }));
    const Cell = kit.table.Cell;
    const rows = try arena.alloc([]const Cell, stats.len);
    for (stats, rows) |s, *row| row.* = try arena.dupe(Cell, &.{
        .{ .text = s.name, .role = .id },
        .{ .text = try std.fmt.allocPrint(arena, "{d}", .{s.max_threads}), .role = if (s.max_threads < 1024) .warn else .text },
        .{ .text = try std.fmt.allocPrint(arena, "{d}", .{s.execution_width}) },
        .{ .text = try std.fmt.allocPrint(arena, "{d}", .{s.threadgroup_memory}) },
    });
    try kit.table.render(writer, ui, arena, 0, &.{
        .{ .title = "KERNEL" },                          .{ .title = "MAX THREADS", .alignment = .right },
        .{ .title = "SIMD WIDTH", .alignment = .right }, .{ .title = "THREADGROUP BYTES", .alignment = .right },
    }, rows);
}

/// The dispatches one Qwen token records (~1,240), with margin; the plan is not asked
/// because the bench does not know the model, only the backend.
const profile_dispatch_capacity = 4096;

/// One row of the per-kernel table: totals divided by the number of profiled
/// steps (command buffers), so the numbers are "per token".
pub const KernelTime = struct {
    kernel: []const u8,
    family: Family,
    encoding: ?[]const u8,
    rows: u32,
    columns: u32,
    dispatches_per_step: f64,
    milliseconds_per_step: f64,
    /// Fraction of all attributed kernel time.
    share: f64,
    /// Weight bytes read per second of kernel time; absent for kernels without weight traffic.
    gigabytes_per_second: ?f64,
};

/// What a kernel's time is spent on, for the per-step split that ranks the
/// decode-speed levers: weight matrices, the output head (a matrix kernel
/// over the vocabulary), attention, the DeltaNet recurrence, norms,
/// token selection, and the rest (residual adds, activations, copies).
pub const Family = enum {
    matrix,
    head,
    attention,
    deltanet,
    norms,
    selection,
    other,

    pub fn of(kernel: []const u8, rows: u32, vocabulary: usize) Family {
        const starts = std.mem.startsWith;
        const matrix = starts(u8, kernel, "matvec") or starts(u8, kernel, "matmul") or std.mem.endsWith(u8, kernel, "reduce_splits");
        if (matrix) return if (rows == vocabulary) .head else .matrix;
        if (starts(u8, kernel, "attention") or starts(u8, kernel, "chunk_") or starts(u8, kernel, "rope") or std.mem.eql(u8, kernel, "pack_half") or std.mem.eql(u8, kernel, "sigmoid_gate")) return .attention;
        if (starts(u8, kernel, "delta") or starts(u8, kernel, "convolution") or std.mem.eql(u8, kernel, "l2norm")) return .deltanet;
        if (std.mem.indexOf(u8, kernel, "norm") != null) return .norms;
        if (starts(u8, kernel, "argmax") or starts(u8, kernel, "topk") or starts(u8, kernel, "expsum") or std.mem.eql(u8, kernel, "penalize")) return .selection;
        return .other;
    }
};

pub const FamilyTime = struct {
    family: Family,
    milliseconds_per_step: f64,
    share: f64,
};

pub const ProfileReport = struct {
    steps: u64,
    dispatches_per_step: f64,
    /// Sum of timed kernel durations per step.
    attributed_milliseconds_per_step: f64,
    /// Whole command buffer GPU time per step in profile mode; the excess over
    /// `attributed` is the cost of one encoder boundary per dispatch.
    gpu_milliseconds_per_step: f64,
    unsampled_dispatches: u64,
    kernels: []const KernelTime,
    /// The kernels summed by family, slowest first; empty families omitted.
    families: []const FamilyTime,
};

/// One accepted prefix length's recovery calls in a sample. `accepted` is the
/// tokens replayed (1 = the seed alone); full acceptance replays nothing and
/// contributes calls with zero milliseconds.
pub const RecoverCell = struct {
    accepted: usize,
    calls: usize,
    rewind_milliseconds: f64,
    replay_milliseconds: f64,
};

pub const Sample = struct {
    warmup: bool,
    prompt_tokens: usize,
    generated_tokens: usize,
    stop_reason: []const u8,
    prefill_milliseconds: f64,
    first_token_milliseconds: f64,
    decode_milliseconds: f64,
    prefill_tokens_per_second: ?f64,
    decode_tokens_per_second: ?f64,
    gpu_busy_milliseconds: ?f64,
    /// Present on the GPU partial top-k sampling path: tokens that needed the full-logit fallback.
    topk_fallbacks: ?usize = null,
    /// This sample ran with the speculative switch on (the pair's second run).
    speculative: bool = false,
    /// A sampled pair: whether the "on" run generated the "off" run's
    /// tokens (both draw from the pair's seed, so they part only where a
    /// verify row's rounding moves a draw).
    same_tokens: ?bool = null,
    draft_length: ?usize = null,
    /// Verify batches run, accepted drafts per batch, and the run's time in
    /// the drafter's `propose`, the model's `verify`, the host acceptance
    /// decision, `recover`, and the drafter's `commit` of the accepted prefix
    /// (divide by the batches for the per-batch cost); speculative samples only.
    speculative_steps: ?usize = null,
    accepted_per_step: ?f64 = null,
    /// Drafts proposed per verify batch; with `accepted_per_step`, drafts per
    /// accepted token is `proposed / accepted` (the proposal policy's lever).
    proposed_per_step: ?f64 = null,
    propose_milliseconds: ?f64 = null,
    verify_milliseconds: ?f64 = null,
    accept_milliseconds: ?f64 = null,
    recover_milliseconds: ?f64 = null,
    /// The verify batch's checkpoint copy, and recovery split into the copy
    /// (`rewind`) and the forward over the accepted prefix (`replay`); these
    /// sum to `recover_milliseconds` less clock overhead.
    checkpoint_milliseconds: ?f64 = null,
    recover_rewind_milliseconds: ?f64 = null,
    recover_replay_milliseconds: ?f64 = null,
    /// Recovery calls per accepted length, index 1..=max_draft_length+1; the
    /// aggregate is `recover_milliseconds` above.
    recover_by_length: ?[inference.engine.max_draft_length + 2]RecoverCell = null,
    commit_milliseconds: ?f64 = null,
};

/// One run of `--verify-rows`: its batches and the mean milliseconds per
/// batch of each component `forcedBatch` times. `total` is the batch cost C
/// (the components plus the host bookkeeping between them).
pub const VerifySample = struct {
    warmup: bool,
    batches: usize,
    propose: f64,
    checkpoint: f64,
    verify: f64,
    recover: f64,
    commit: f64,
    total: f64,
};

pub const VerifyReport = struct {
    rows: usize,
    accepted: usize,
    /// Session position at the first batch of every run.
    depth: usize,
    /// Mean over the measured runs of their per-batch means; null when no
    /// run was measured.
    mean: ?VerifySample,
    samples: []const VerifySample,
};

/// The saved prefix a run restored or wrote: its file, its length in tokens,
/// and whether it came from the file (`restored`) or was prefilled now.
pub const PrefixReport = struct {
    path: ?[]const u8,
    tokens: usize,
    restored: bool,
    /// Loading the file, or prefilling and saving it.
    milliseconds: f64,
};

pub const Report = struct {
    /// Pre-1.0: the shape evolves with the tree, so no bump for the
    /// speculative sample fields and means added alongside the rest.
    schema_version: u32 = 1,
    backend: []const u8,
    model_path: []const u8,
    /// The configuration file the run read, or the built-in defaults.
    config: []const u8 = "built-in defaults",
    /// `text` (tokenized here, raw or rendered) or `tokens` (`--prompt-tokens`).
    prompt_source: []const u8 = "text",
    build_mode: []const u8 = @tagName(@import("builtin").mode),
    context: usize,
    /// Attention cache precision the session used (`f32` on the CPU reference).
    kv_precision: []const u8 = "f32",
    /// Session block (KV cache and recurrent state), page-padded.
    session_bytes: usize = 0,
    max_tokens: usize,
    sampling: []const u8 = "greedy",
    sampling_options: inference.sampling.Options = .{},
    seed: u64 = 0,
    warmup_runs: usize,
    measured_runs: usize,
    load_milliseconds: f64,
    samples: []const Sample,
    mean_prefill_tokens_per_second: ?f64,
    mean_decode_tokens_per_second: ?f64,
    mean_first_token_milliseconds: ?f64,
    /// Speculative decoding, when a drafter was loaded and both ways measured:
    /// the draft length, the decode rate with the switch on, the accepted
    /// drafts per verify batch, and the decode speedup over the baseline.
    speculative_draft_length: ?usize = null,
    speculative_p_min: ?f32 = null,
    speculative_chain_min: ?f32 = null,
    /// The depth cap the "on" runs drafted under; a prompt at or past it ran
    /// plainly, so its "on" rate is the plain one.
    speculative_max_depth: ?usize = null,
    mean_speculative_decode_tokens_per_second: ?f64 = null,
    mean_accepted_per_step: ?f64 = null,
    mean_proposed_per_step: ?f64 = null,
    decode_speedup: ?f64 = null,
    prefix: ?PrefixReport = null,
    verify: ?VerifyReport = null,
    profile: ?ProfileReport = null,
    /// `--expert-counts`: the routing tallied over every run, warm-ups and
    /// the prefix's prefill included.
    experts: ?expert_counts.Summary = null,

    pub fn render(self: Report, arena: std.mem.Allocator, out: *std.Io.Writer, json: bool, ui: kit.Ui) !void {
        if (json) {
            try std.json.Stringify.value(self, .{ .whitespace = .indent_2, .emit_null_optional_fields = false }, out);
            return out.writeByte('\n');
        }
        const v = kit.values;
        const model = try kit.text.clean(arena, self.model_path);
        try kit.blocks.header(out, ui, std.fs.path.basename(model), try std.fmt.allocPrint(arena, "{s} ({s})", .{ self.backend, self.build_mode }));
        var fields: std.ArrayList(kit.blocks.Field) = .empty;
        try fields.append(arena, .{ .key = "Model", .value = model, .role = .id });
        try fields.append(arena, .{ .key = "Config", .value = try kit.text.clean(arena, self.config), .role = .id });
        try fields.append(arena, .{ .key = "Context", .value = try std.fmt.allocPrint(arena, "{f} · KV {s} · session {f}", .{ v.count(self.context, "token"), self.kv_precision, v.size(self.session_bytes) }) });
        try fields.append(arena, .{ .key = "Output", .value = try std.fmt.allocPrint(arena, "budget of {f} · {s} · prompt from {s}", .{ v.count(self.max_tokens, "token"), self.sampling, self.prompt_source }) });
        try fields.append(arena, .{ .key = "Load", .value = try std.fmt.allocPrint(arena, "{f}", .{v.duration(self.load_milliseconds / 1000)}) });
        if (self.prefix) |p| {
            const where = if (p.path) |path| try std.fmt.allocPrint(arena, " · {s}", .{try kit.text.clean(arena, path)}) else "";
            try fields.append(arena, .{ .key = "Prefix", .value = try std.fmt.allocPrint(arena, "{f} {s} in {f}{s}", .{ v.count(p.tokens, "token"), if (p.restored) "restored" else "prefilled", v.duration(p.milliseconds / 1000), where }) });
        }
        if (self.profile != null) try fields.append(arena, .{ .key = "Profiled", .value = "one encoder per dispatch, so these rates are not comparable with unprofiled runs", .role = .warn });
        try kit.blocks.detail(out, ui, &.{.{ .fields = fields.items }});
        try out.writeByte('\n');
        if (self.verify) |verify| {
            try renderVerify(verify, arena, out, ui);
            if (self.profile) |p| try renderProfile(p, arena, out, ui);
            return;
        }
        try self.renderSamples(arena, out, ui);
        try out.writeByte('\n');
        var means: std.ArrayList(kit.blocks.Field) = .empty;
        try means.append(arena, .{ .key = "Prefill", .value = try rate(arena, self.mean_prefill_tokens_per_second) });
        try means.append(arena, .{ .key = "Decode", .value = try rate(arena, self.mean_decode_tokens_per_second) });
        try means.append(arena, .{ .key = "First token", .value = if (self.mean_first_token_milliseconds) |ms| try std.fmt.allocPrint(arena, "{d:.1} ms", .{ms}) else v.missing });
        if (self.speculative_draft_length) |drafts| {
            try means.append(arena, .{ .key = "Speculative", .value = try std.fmt.allocPrint(arena, "{s} decode at draft length {d}", .{ try rate(arena, self.mean_speculative_decode_tokens_per_second), drafts }) });
            try means.append(arena, .{ .key = "Speedup", .value = if (self.decode_speedup) |x| try std.fmt.allocPrint(arena, "{d:.2}x over the baseline", .{x}) else v.missing, .role = .emphasis });
            try means.append(arena, .{ .key = "Accepted", .value = if (self.mean_accepted_per_step) |a|
                (if (self.mean_proposed_per_step) |p| try std.fmt.allocPrint(arena, "{d:.2} of {d:.2} drafts per step", .{ a, p }) else try std.fmt.allocPrint(arena, "{d:.2} drafts per step", .{a}))
            else
                v.missing });
        }
        try kit.blocks.detail(out, ui, &.{.{ .title = try std.fmt.allocPrint(arena, "Mean of {f}", .{v.count(self.measured_runs, "measured run")}), .fields = means.items }});
        if (self.profile) |p| try renderProfile(p, arena, out, ui);
        if (self.experts) |e| {
            try out.writeByte('\n');
            try e.render(out);
        }
    }

    /// One row per run; a column no run measured (GPU time on the CPU,
    /// fallbacks off the top-k path, the switch without a drafter) is left out.
    fn renderSamples(self: Report, arena: std.mem.Allocator, out: *std.Io.Writer, ui: kit.Ui) !void {
        const Cell = kit.table.Cell;
        var any_gpu = false;
        var any_fallbacks = false;
        for (self.samples) |s| {
            any_gpu = any_gpu or s.gpu_busy_milliseconds != null;
            any_fallbacks = any_fallbacks or s.topk_fallbacks != null;
        }
        const missing: Cell = .{ .text = kit.values.missing, .role = .muted };
        const rows = try arena.alloc([]const Cell, self.samples.len);
        // A speculative run is a pair (off, then on) under one number.
        const per_run: usize = if (self.speculative_draft_length != null) 2 else 1;
        var measured: usize = 0;
        for (self.samples, rows) |s, *row| {
            if (!s.warmup) measured += 1;
            row.* = try arena.dupe(Cell, &.{
                if (s.warmup) .{ .text = "warm-up", .role = .muted } else .{ .text = try std.fmt.allocPrint(arena, "{d}", .{(measured - 1) / per_run + 1}) },
                .{ .text = try std.fmt.allocPrint(arena, "{d}", .{s.prompt_tokens}) },
                .{ .text = try std.fmt.allocPrint(arena, "{d}", .{s.generated_tokens}) },
                .{ .text = if (self.speculative_draft_length == null) "" else if (s.speculative) "on" else "off" },
                .{ .text = s.stop_reason, .role = if (std.mem.eql(u8, s.stop_reason, "cancelled")) .warn else .text },
                .{ .text = try std.fmt.allocPrint(arena, "{d:.1}", .{s.prefill_milliseconds}) },
                if (s.prefill_tokens_per_second) |r| .{ .text = try std.fmt.allocPrint(arena, "{d:.2}", .{r}), .role = .emphasis } else missing,
                .{ .text = try std.fmt.allocPrint(arena, "{d:.1}", .{s.first_token_milliseconds}) },
                .{ .text = try std.fmt.allocPrint(arena, "{d:.1}", .{s.decode_milliseconds}) },
                if (s.decode_tokens_per_second) |r| .{ .text = try std.fmt.allocPrint(arena, "{d:.2}", .{r}), .role = .emphasis } else missing,
                if (!any_gpu) .{ .text = "" } else if (s.gpu_busy_milliseconds) |ms| .{ .text = try std.fmt.allocPrint(arena, "{d:.1}", .{ms}) } else missing,
                if (!any_fallbacks) .{ .text = "" } else if (s.topk_fallbacks) |n| .{ .text = try std.fmt.allocPrint(arena, "{d}", .{n}) } else missing,
            });
        }
        try kit.table.render(out, ui, arena, 0, &.{
            .{ .title = "RUN" },                                    .{ .title = "PROMPT", .alignment = .right },
            .{ .title = "GEN", .alignment = .right },               .{ .title = "SPEC" },
            .{ .title = "STOP" },                                   .{ .title = "PREFILL MS", .alignment = .right },
            .{ .title = "PP TOK/S", .alignment = .right },          .{ .title = "FIRST MS", .alignment = .right },
            .{ .title = "DECODE MS", .alignment = .right },         .{ .title = "TG TOK/S", .alignment = .right },
            .{ .title = "GPU MS", .alignment = .right, .drop = 2 }, .{ .title = "FALLBACKS", .alignment = .right, .drop = 1 },
        }, rows);
    }

    fn rate(arena: std.mem.Allocator, value: ?f64) ![]const u8 {
        return if (value) |r| std.fmt.allocPrint(arena, "{d:.2} tok/s", .{r}) else kit.values.missing;
    }

    fn renderVerify(verify: VerifyReport, arena: std.mem.Allocator, out: *std.Io.Writer, ui: kit.Ui) !void {
        const Cell = kit.table.Cell;
        const rows = try arena.alloc([]const Cell, verify.samples.len);
        var measured: usize = 0;
        for (verify.samples, rows) |s, *row| {
            if (!s.warmup) measured += 1;
            row.* = try arena.dupe(Cell, &.{
                if (s.warmup) .{ .text = "warm-up", .role = .muted } else .{ .text = try std.fmt.allocPrint(arena, "{d}", .{measured}) },
                .{ .text = try std.fmt.allocPrint(arena, "{d}", .{s.batches}) },
                .{ .text = try std.fmt.allocPrint(arena, "{d:.2}", .{s.propose}) },
                .{ .text = try std.fmt.allocPrint(arena, "{d:.2}", .{s.checkpoint}) },
                .{ .text = try std.fmt.allocPrint(arena, "{d:.2}", .{s.verify}) },
                .{ .text = try std.fmt.allocPrint(arena, "{d:.2}", .{s.recover}) },
                .{ .text = try std.fmt.allocPrint(arena, "{d:.2}", .{s.commit}) },
                .{ .text = try std.fmt.allocPrint(arena, "{d:.2}", .{s.total}), .role = .emphasis },
            });
        }
        try kit.table.render(out, ui, arena, 0, &.{
            .{ .title = "RUN" },                          .{ .title = "BATCHES", .alignment = .right },
            .{ .title = "PROPOSE", .alignment = .right }, .{ .title = "CHECKPOINT", .alignment = .right },
            .{ .title = "VERIFY", .alignment = .right },  .{ .title = "RECOVER", .alignment = .right },
            .{ .title = "COMMIT", .alignment = .right },  .{ .title = "C MS/BATCH", .alignment = .right },
        }, rows);
        try out.writeByte('\n');
        try kit.blocks.detail(out, ui, &.{.{ .title = "Verify batch", .fields = &.{
            .{ .key = "Rows", .value = try std.fmt.allocPrint(arena, "{d} · {d} accepted · from depth {d}", .{ verify.rows, verify.accepted, verify.depth }) },
            .{ .key = "Cost C", .value = if (verify.mean) |m| try std.fmt.allocPrint(arena, "{d:.2} ms per batch", .{m.total}) else "no measured run", .role = .emphasis },
            .{ .key = "Of which", .value = if (verify.mean) |m| try std.fmt.allocPrint(arena, "propose {d:.2} · checkpoint {d:.2} · verify {d:.2} · recover {d:.2} · commit {d:.2}", .{ m.propose, m.checkpoint, m.verify, m.recover, m.commit }) else kit.values.missing },
        } }});
    }

    fn renderProfile(p: ProfileReport, arena: std.mem.Allocator, out: *std.Io.Writer, ui: kit.Ui) !void {
        const Cell = kit.table.Cell;
        try out.writeByte('\n');
        try kit.blocks.detail(out, ui, &.{.{ .title = try std.fmt.allocPrint(arena, "GPU time per kernel over {f}", .{kit.values.count(p.steps, "measured step")}), .fields = &.{
            .{ .key = "Dispatches", .value = try std.fmt.allocPrint(arena, "{d:.0} per step · {d} unsampled", .{ p.dispatches_per_step, p.unsampled_dispatches }) },
            .{ .key = "Attributed", .value = try std.fmt.allocPrint(arena, "{d:.1} ms per step of {d:.1} ms command-buffer time", .{ p.attributed_milliseconds_per_step, p.gpu_milliseconds_per_step }) },
        } }});
        try out.writeByte('\n');
        const kernels = try arena.alloc([]const Cell, p.kernels.len);
        for (p.kernels, kernels) |k, *row| row.* = try arena.dupe(Cell, &.{
            .{ .text = k.kernel, .role = .id },
            if (k.encoding) |e| .{ .text = e } else .{ .text = kit.values.missing, .role = .muted },
            .{ .text = try std.fmt.allocPrint(arena, "{d}", .{k.rows}) },
            .{ .text = try std.fmt.allocPrint(arena, "{d}", .{k.columns}) },
            .{ .text = try std.fmt.allocPrint(arena, "{d:.1}", .{k.dispatches_per_step}) },
            .{ .text = try std.fmt.allocPrint(arena, "{d:.3}", .{k.milliseconds_per_step}), .role = .emphasis },
            .{ .text = try std.fmt.allocPrint(arena, "{d:.1}%", .{k.share * 100}) },
            if (k.gigabytes_per_second) |gbps| .{ .text = try std.fmt.allocPrint(arena, "{d:.1}", .{gbps}) } else .{ .text = kit.values.missing, .role = .muted },
        });
        try kit.table.render(out, ui, arena, 0, &.{
            .{ .title = "KERNEL" },                      .{ .title = "ENCODING" },
            .{ .title = "ROWS", .alignment = .right },   .{ .title = "COLS", .alignment = .right },
            .{ .title = "N/STEP", .alignment = .right }, .{ .title = "MS/STEP", .alignment = .right },
            .{ .title = "SHARE", .alignment = .right },  .{ .title = "GB/S", .alignment = .right },
        }, kernels);
        try out.writeByte('\n');
        const families = try arena.alloc([]const Cell, p.families.len);
        for (p.families, families) |f, *row| row.* = try arena.dupe(Cell, &.{
            .{ .text = @tagName(f.family) },
            .{ .text = try std.fmt.allocPrint(arena, "{d:.3}", .{f.milliseconds_per_step}), .role = .emphasis },
            .{ .text = try std.fmt.allocPrint(arena, "{d:.1}%", .{f.share * 100}) },
        });
        try kit.table.render(out, ui, arena, 0, &.{
            .{ .title = "FAMILY" }, .{ .title = "MS/STEP", .alignment = .right }, .{ .title = "SHARE", .alignment = .right },
        }, families);
    }
};

/// Pure aggregation of a backend profile into per-step rows sorted by time,
/// so the table is testable without a GPU. `vocabulary` tells the output
/// head's matrix from the others. Caller frees `kernels` and `families`.
pub fn profileReport(alloc: std.mem.Allocator, profile: *const inference.metal.Profile, vocabulary: usize) !ProfileReport {
    const kernels = try alloc.alloc(KernelTime, profile.totals.count());
    errdefer alloc.free(kernels);
    const steps = profile.command_buffers;
    const per_step: f64 = if (steps == 0) 0 else 1.0 / @as(f64, @floatFromInt(steps));
    var attributed: f64 = 0;
    var dispatches: u64 = profile.unsampled;
    var it = profile.totals.iterator();
    while (it.next()) |entry| {
        attributed += entry.value_ptr.seconds;
        dispatches += entry.value_ptr.dispatches;
    }
    it = profile.totals.iterator();
    var i: usize = 0;
    while (it.next()) |entry| : (i += 1) {
        const key = entry.key_ptr.*;
        const total = entry.value_ptr.*;
        kernels[i] = .{
            .kernel = @tagName(key.kernel),
            .family = Family.of(@tagName(key.kernel), key.rows, vocabulary),
            .encoding = if (key.encoding) |id| (if (inference.encoding.layout(id)) |layout| layout.name else "?") else null,
            .rows = key.rows,
            .columns = key.columns,
            .dispatches_per_step = @as(f64, @floatFromInt(total.dispatches)) * per_step,
            .milliseconds_per_step = total.seconds * 1000 * per_step,
            .share = if (attributed > 0) total.seconds / attributed else 0,
            .gigabytes_per_second = if (total.bytes > 0 and total.seconds > 0) @as(f64, @floatFromInt(total.bytes)) / total.seconds / 1e9 else null,
        };
    }
    std.mem.sort(KernelTime, kernels, {}, struct {
        fn slower(_: void, a: KernelTime, b: KernelTime) bool {
            return a.milliseconds_per_step > b.milliseconds_per_step;
        }
    }.slower);
    var sums: [@typeInfo(Family).@"enum".field_names.len]f64 = @splat(0);
    for (kernels) |k| sums[@backingInt(k.family)] += k.milliseconds_per_step;
    var present: usize = 0;
    for (sums) |ms| present += @intFromBool(ms > 0);
    const families = try alloc.alloc(FamilyTime, present);
    errdefer alloc.free(families);
    const total_ms = attributed * 1000 * per_step;
    var f: usize = 0;
    for (sums, 0..) |ms, tag| if (ms > 0) {
        families[f] = .{ .family = @fromBackingInt(@intCast(tag)), .milliseconds_per_step = ms, .share = if (total_ms > 0) ms / total_ms else 0 };
        f += 1;
    };
    std.mem.sort(FamilyTime, families, {}, struct {
        fn slower(_: void, a: FamilyTime, b: FamilyTime) bool {
            return a.milliseconds_per_step > b.milliseconds_per_step;
        }
    }.slower);
    return .{
        .steps = steps,
        .dispatches_per_step = @as(f64, @floatFromInt(dispatches)) * per_step,
        .attributed_milliseconds_per_step = attributed * 1000 * per_step,
        .gpu_milliseconds_per_step = profile.gpu_seconds * 1000 * per_step,
        .unsampled_dispatches = profile.unsampled,
        .kernels = kernels,
        .families = families,
    };
}

fn perSecond(count: usize, duration: std.Io.Duration) ?f64 {
    if (count == 0 or duration.nanoseconds <= 0) return null;
    return @as(f64, @floatFromInt(count)) / (@as(f64, @floatFromInt(duration.nanoseconds)) / std.time.ns_per_s);
}

/// The run's recovery calls as one cell per accepted length, index 0 included
/// but always empty (no recovery replays zero tokens).
fn recoverByLength(t: inference.engine.Timing) [inference.engine.max_draft_length + 2]RecoverCell {
    var cells: [inference.engine.max_draft_length + 2]RecoverCell = undefined;
    for (&cells, t.recover_by_length, 0..) |*cell, call, accepted| {
        cell.* = .{
            .accepted = accepted,
            .calls = call.calls,
            .rewind_milliseconds = engine.milliseconds(call.rewind),
            .replay_milliseconds = engine.milliseconds(call.replay),
        };
    }
    return cells;
}

/// Pure aggregation over samples so the statistics are testable without a
/// model. The baseline means (switch off) and the speculative ones are kept
/// apart: mixing them would hide the very comparison the pair exists for.
pub const Summary = struct {
    prefill: ?f64,
    decode: ?f64,
    first: ?f64,
    measured: usize,
    speculative_decode: ?f64,
    accepted_per_step: ?f64,
    proposed_per_step: ?f64,
    speculative_measured: usize,
};

pub fn summarize(samples: []const Sample) Summary {
    var prefill_sum: f64 = 0;
    var prefill_n: usize = 0;
    var decode_sum: f64 = 0;
    var decode_n: usize = 0;
    var first_sum: f64 = 0;
    var measured: usize = 0;
    var spec_decode_sum: f64 = 0;
    var spec_decode_n: usize = 0;
    var accepted_sum: f64 = 0;
    var accepted_n: usize = 0;
    var proposed_sum: f64 = 0;
    var proposed_n: usize = 0;
    var speculative_measured: usize = 0;
    for (samples) |s| {
        if (s.warmup or std.mem.eql(u8, s.stop_reason, "cancelled")) continue;
        if (s.speculative) {
            speculative_measured += 1;
            if (s.decode_tokens_per_second) |v| {
                spec_decode_sum += v;
                spec_decode_n += 1;
            }
            if (s.accepted_per_step) |v| {
                accepted_sum += v;
                accepted_n += 1;
            }
            if (s.proposed_per_step) |v| {
                proposed_sum += v;
                proposed_n += 1;
            }
            continue;
        }
        measured += 1;
        first_sum += s.first_token_milliseconds;
        if (s.prefill_tokens_per_second) |v| {
            prefill_sum += v;
            prefill_n += 1;
        }
        if (s.decode_tokens_per_second) |v| {
            decode_sum += v;
            decode_n += 1;
        }
    }
    return .{
        .prefill = if (prefill_n > 0) prefill_sum / @as(f64, @floatFromInt(prefill_n)) else null,
        .decode = if (decode_n > 0) decode_sum / @as(f64, @floatFromInt(decode_n)) else null,
        .first = if (measured > 0) first_sum / @as(f64, @floatFromInt(measured)) else null,
        .measured = measured,
        .speculative_decode = if (spec_decode_n > 0) spec_decode_sum / @as(f64, @floatFromInt(spec_decode_n)) else null,
        .accepted_per_step = if (accepted_n > 0) accepted_sum / @as(f64, @floatFromInt(accepted_n)) else null,
        .proposed_per_step = if (proposed_n > 0) proposed_sum / @as(f64, @floatFromInt(proposed_n)) else null,
        .speculative_measured = speculative_measured,
    };
}

pub fn run(alloc: std.mem.Allocator, io: std.Io, model_path: []const u8, settings: config.Resolved, options: Options, json: bool, writer: *std.Io.Writer, ui: kit.Ui, step: *kit.Step) !void {
    var arena_state = std.heap.ArenaAllocator.init(alloc);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const user: ?[]u8 = if (options.prompt_tokens == null) try engine.readPrompt(alloc, io, options.prompt, options.prompt_file) else null;
    defer if (user) |text| alloc.free(text);
    const limit = settings.max_tokens;
    const capacity = settings.ctx_size;
    const repeat = options.repeat orelse 3;
    const warmup = options.warmup orelse 1;
    if (limit == 0 or limit > config.max_output_tokens or capacity == 0 or capacity > config.max_context or repeat == 0 or repeat > 100 or warmup > 100) return error.InvalidGenerationBudget;
    // Bench measures what the model offers: with the switch off the drafter
    // is not loaded at all (its weights and scratch are part of the load
    // plan), which is the true baseline; `--speculative on` loads the
    // family's source and each measured run is done both ways on that one
    // loaded model, so the pair's off sample is the loaded-but-off case.
    const draft_path = try engine.draftPath(alloc, model_path, try settings.draftFile());
    defer if (draft_path) |path| alloc.free(path);
    const draft: inference.engine.DraftRequest = if (settings.speculative) .{ .preferred = draft_path } else .none;
    var eng = try engine.Engine.open(alloc, io, model_path, settings.backend, capacity, settings.kv_precision, settings.forced_profile, draft);
    defer eng.deinit();
    const speculate = eng.model.drafter() != null;
    var stop = settings.draft_stop;
    if (options.draft_p_min) |p| stop.p_min = p;
    if (options.draft_chain_min) |c| stop.chain_min = c;
    const gpu: ?*inference.metal.Backend = eng.model.gpu();
    if (gpu) |backend| backend.fused_norms = !options.unfused_norms;
    if (options.profile) {
        const backend = gpu orelse return error.ProfileRequiresMetal;
        var diagnostic: [1024]u8 = @splat(0);
        backend.enableProfiling(profile_dispatch_capacity, &diagnostic) catch |err| {
            std.log.err("{s}", .{std.mem.sliceTo(&diagnostic, 0)});
            return err;
        };
    }
    // The capture runs after minutes of measurement, so its refusals are
    // found before them.
    const capture: ?[:0]u8 = if (options.capture) |path| blk: {
        if (gpu == null) return error.CaptureRequiresMetal;
        if (std.Io.Dir.cwd().access(io, path, .{})) |_| return error.CaptureExists else |_| {}
        break :blk try alloc.dupeSentinel(u8, path, 0);
    } else null;
    defer if (capture) |path| alloc.free(path);
    interrupt.install();
    const tokens = if (options.prompt_tokens) |path| try engine.readPromptTokens(alloc, io, path, eng.vocab.tokens.len) else blk: {
        const prompt = try eng.prompt(user.?, options.raw, .off);
        defer alloc.free(prompt);
        break :blk try eng.encode(prompt);
    };
    defer alloc.free(tokens);
    // The timed decode seeds from the last prompt token.
    if (tokens.len == 0) return error.EmptyPrompt;
    if (tokens.len > capacity or limit > capacity - tokens.len) return error.ContextFull;
    const logits = try alloc.alloc(f32, eng.vocab.tokens.len);
    defer alloc.free(logits);
    const generated = try alloc.alloc(u32, limit);
    defer alloc.free(generated);
    // Neutral options plus the flags: greedy unless the command line says
    // otherwise, never a model profile and never the file's overrides.
    var sampler = try inference.sampling.Sampler.init(options.seed orelse 0, (inference.sampling.Options{}).override(settings.sampling));
    const candidates = try alloc.alloc(inference.sampling.Candidate, if (sampler.options.temperature == 0) 0 else eng.vocab.tokens.len);
    defer alloc.free(candidates);
    var history = try inference.sampling.History.init(alloc, eng.vocab.tokens.len);
    defer history.deinit();
    var sampling_label: [192]u8 = undefined;
    const o = sampler.options;
    const sampling: []const u8 = if (o.temperature == 0 and !o.penaltiesActive()) "greedy" else try std.fmt.bufPrint(&sampling_label, "sampled: temperature {d}, top-k {d}, top-p {d}, min-p {d}, presence {d}, repetition {d}, seed {d}", .{ o.temperature, o.top_k, o.top_p, o.min_p, o.presence_penalty, o.repetition_penalty, options.seed orelse 0 });
    const accept = options.accept orelse 0;
    if (options.accept != null and options.verify_rows == null) return error.AcceptNeedsVerifyRows;
    if (options.verify_rows) |rows| {
        const d = eng.model.drafter() orelse return error.VerifyRowsNeedDrafter;
        if (rows - 1 > d.max_proposals or accept >= rows) return error.InvalidDraftLength;
        // Every run's last batch, and the capture's, still needs its rows' room.
        if (tokens.len - 1 + (limit - 1) * (accept + 1) + rows > capacity) return error.ContextFull;
    }
    var trace: generate.Trace = .{ .io = io, .directory = null, .started = std.Io.Clock.awake.now(io) };
    const counts: ?*expert_counts.Counts = if (options.expert_counts) try alloc.create(expert_counts.Counts) else null;
    defer if (counts) |c| alloc.destroy(c);
    if (counts) |c| {
        c.* = .{};
        trace.experts = c;
    }

    // The saved prefix is the prompt less its last token, so each run still
    // feeds one prompt token and samples its first token from real logits.
    // A verify run always starts from one, kept in memory without a directory.
    const prefix_tokens = tokens[0 .. tokens.len - 1];
    var prefix: ?inference.session.Snapshot = null;
    defer if (prefix) |*p| p.deinit();
    var prefix_report: ?PrefixReport = null;
    var prefix_path: [4096]u8 = undefined;
    if (options.prefix_cache != null or options.verify_rows != null) {
        const started = std.Io.Clock.awake.now(io);
        const dir: ?std.Io.Dir = if (options.prefix_cache) |path| try std.Io.Dir.cwd().createDirPathOpen(io, path, .{}) else null;
        defer if (dir) |d| d.close(io);
        var key: prefix_cache.Key = undefined;
        var path: ?[]const u8 = null;
        if (dir) |d| {
            key = .{ .model = try prefix_cache.modelDigest(io, model_path, if (speculate) draft_path else null), .tokens = prefix_cache.tokenDigest(prefix_tokens), .layout = eng.model.session().layout_digest };
            var name: [64]u8 = undefined;
            path = try std.fmt.bufPrint(&prefix_path, "{s}/{s}", .{ options.prefix_cache.?, prefix_cache.fileName(&name, key) });
            prefix = prefix_cache.load(alloc, io, d, key) catch |err| switch (err) {
                error.PrefixMismatch, error.PrefixCorrupt => blk: {
                    std.log.warn("saved prefix {s} refused ({s}); prefilling instead", .{ path.?, @errorName(err) });
                    break :blk null;
                },
                else => return err,
            };
        }
        const restored = prefix != null;
        if (!restored) {
            // Committed to the drafter too when one is loaded, as the
            // speculative loop commits a prompt, so its cache rows exist.
            eng.model.reset();
            if (prefix_tokens.len > 0) {
                if (speculate) {
                    try inference.engine.commitPrompt(&eng, prefix_tokens, null, trace.observer());
                } else try eng.model.prefill(prefix_tokens, null, null, null, null, null, trace.observer());
            }
            prefix = try eng.model.snapshot(alloc);
            // Minutes of prefill are not thrown away for a failed save: the
            // runs go on from the snapshot in memory.
            if (dir) |d| prefix_cache.save(io, d, key, &prefix.?) catch |err| {
                std.log.warn("could not save the prefix to {s}: {s}", .{ path.?, @errorName(err) });
            };
        }
        prefix_report = .{ .path = path, .tokens = prefix_tokens.len, .restored = restored, .milliseconds = engine.milliseconds(started.durationTo(std.Io.Clock.awake.now(io))) };
        step.update("prefix: {d} tokens {s}", .{ prefix_tokens.len, if (restored) "restored" else "prefilled" });
    }

    if (options.verify_rows) |rows| return runVerify(alloc, &eng, tokens, &prefix.?, rows, accept, limit, warmup, repeat, capture, json, writer, ui, step, .{
        .backend = @tagName(settings.backend),
        .model_path = model_path,
        .config = settings.config_file orelse "built-in defaults",
        .prompt_source = if (options.prompt_tokens != null) "tokens" else "text",
        .context = capacity,
        .kv_precision = @tagName(eng.kv_precision),
        .session_bytes = eng.model.session().bytes(),
        .max_tokens = limit,
        .sampling = sampling,
        .sampling_options = sampler.options,
        .seed = options.seed orelse 0,
        .warmup_runs = warmup,
        .measured_runs = 0,
        .load_milliseconds = engine.milliseconds(eng.load),
        .samples = &.{},
        .mean_prefill_tokens_per_second = null,
        .mean_decode_tokens_per_second = null,
        .mean_first_token_milliseconds = null,
        .prefix = prefix_report,
    });

    const per_iteration: usize = if (speculate) 2 else 1;
    const samples = try alloc.alloc(Sample, (warmup + repeat) * per_iteration);
    defer alloc.free(samples);
    // A sampled pair draws both runs from one seed, `seed + i`, so the two
    // sample the same continuation and the pair's ratio is the drafter's,
    // not two texts'.
    const paired_seed = speculate and sampler.options.temperature > 0;
    const off_tokens = try alloc.alloc(u32, if (paired_seed) limit else 0);
    defer alloc.free(off_tokens);
    var off_count: usize = 0;
    var completed: usize = 0;
    var cancelled = false;
    for (0..warmup + repeat) |i| {
        // Warm-up dispatches are timed too but must not enter the profile.
        if (i == warmup) if (gpu) |backend| if (backend.profile) |*p| p.clear();
        for ([_]bool{ false, true }) |on| {
            if (on and !speculate) continue;
            // Every run starts from an empty session and an empty token
            // history, or from the saved prefix and the history it implies.
            eng.model.reset();
            history.reset();
            if (prefix) |*p| {
                try eng.model.restore(p);
                for (prefix_tokens) |token| try history.observe(token);
            }
            const fed = if (prefix != null) tokens[prefix_tokens.len..] else tokens;
            if (paired_seed) sampler.rng = .init((options.seed orelse 0) +% i);
            const spec: inference.engine.Speculative = if (on) .{ .enabled = true, .draft_length = settings.draft_length, .stop = stop, .max_position = settings.draft_max_depth } else .{};
            // Decode covers the steps after the first sampled token; the first
            // token's latency (prefill included) is reported separately.
            // The profile covers decode steps only: a run's prefill is
            // paused out of it and the loop's first step hook resumes it.
            if (step.live()) step.update("{s} {d} of {d}{s}", .{ if (i < warmup) "warm-up" else "run", if (i < warmup) i + 1 else i - warmup + 1, if (i < warmup) warmup else repeat, if (on) ", speculative" else "" });
            var gate: ProfileGate = .{ .profile = if (gpu) |backend| (if (backend.profile) |*p| p else null) else null };
            if (gate.profile) |p| p.paused = !options.profile_prefill;
            const outcome = try generate.runLoop(&eng, fed, limit, &sampler, &history, spec, null, logits, candidates, generated, &trace, .{ .context = &gate, .step = ProfileGate.step });
            if (gate.profile) |p| p.paused = false;
            const t = outcome.timing;
            const decode_steps = if (t.generated_tokens > 1) t.generated_tokens - 1 else 0;
            if (paired_seed and !on) {
                off_count = t.generated_tokens;
                @memcpy(off_tokens[0..off_count], generated[0..off_count]);
            }
            samples[completed] = .{
                .warmup = i < warmup,
                .prompt_tokens = tokens.len,
                .generated_tokens = t.generated_tokens,
                .stop_reason = @tagName(outcome.stop),
                .prefill_milliseconds = engine.milliseconds(t.prefill),
                .first_token_milliseconds = engine.milliseconds(t.first_token),
                .decode_milliseconds = engine.milliseconds(t.decode),
                .prefill_tokens_per_second = if (prefix != null) null else perSecond(t.prompt_tokens, t.prefill),
                .decode_tokens_per_second = perSecond(decode_steps, t.decode),
                .gpu_busy_milliseconds = if (t.gpu_seconds) |sec| sec * 1000 else null,
                .topk_fallbacks = t.topk_fallbacks,
                .speculative = on,
                .same_tokens = if (paired_seed and on) std.mem.eql(u32, off_tokens[0..off_count], generated[0..t.generated_tokens]) else null,
                .draft_length = if (on) settings.draft_length else null,
                .speculative_steps = if (on) t.speculative_steps else null,
                .accepted_per_step = if (on and t.speculative_steps > 0) @as(f64, @floatFromInt(t.accepted_drafts)) / @as(f64, @floatFromInt(t.speculative_steps)) else null,
                .proposed_per_step = if (on and t.speculative_steps > 0) @as(f64, @floatFromInt(t.proposed_drafts)) / @as(f64, @floatFromInt(t.speculative_steps)) else null,
                .propose_milliseconds = if (on) engine.milliseconds(t.propose) else null,
                .verify_milliseconds = if (on) engine.milliseconds(t.verify) else null,
                .accept_milliseconds = if (on) engine.milliseconds(t.accept) else null,
                .recover_milliseconds = if (on) engine.milliseconds(t.recover) else null,
                .checkpoint_milliseconds = if (on) engine.milliseconds(t.checkpoint) else null,
                .recover_rewind_milliseconds = if (on) engine.milliseconds(t.recover_rewind) else null,
                .recover_replay_milliseconds = if (on) engine.milliseconds(t.recover_replay) else null,
                .recover_by_length = if (on) recoverByLength(t) else null,
                .commit_milliseconds = if (on) engine.milliseconds(t.commit) else null,
            };
            if (!step.live()) step.update("{s} {d} of {d}{s}: {d} prompt, {d} generated, {s}", .{ if (i < warmup) "warm-up" else "run", if (i < warmup) i + 1 else i - warmup + 1, if (i < warmup) warmup else repeat, if (on) ", speculative" else "", tokens.len, t.generated_tokens, @tagName(outcome.stop) });
            completed += 1;
            // A cancelled run is reported as its own sample and ends the
            // benchmark; partial runs must not be averaged as if they had
            // finished.
            if (outcome.stop == .cancelled) {
                cancelled = true;
                break;
            }
        }
        if (cancelled) break;
    }
    const measured_samples = samples[0..completed];
    const stats = summarize(measured_samples);
    var profile: ?ProfileReport = null;
    defer if (profile) |p| {
        alloc.free(p.kernels);
        alloc.free(p.families);
    };
    if (gpu) |backend| if (backend.profile) |*p| {
        profile = try profileReport(alloc, p, eng.vocab.tokens.len);
    };
    if (capture) |path| if (!cancelled) {
        // One decode step after the last run: the loop never feeds its last
        // sampled token, so the budget's room still holds this one.
        var diagnostic: [1024]u8 = @splat(0);
        gpu.?.captureNext(path, &diagnostic) catch |err| {
            std.log.err("{s}", .{std.mem.sliceTo(&diagnostic, 0)});
            return err;
        };
        var next: u32 = 0;
        try eng.model.step(generated[0], null, &next, null, null, null);
        step.done(.ok, "captured one decode step", try std.fmt.allocPrint(arena, "position {d} · {s}", .{ eng.model.session().position, path }));
    };
    const report: Report = .{
        .backend = @tagName(settings.backend),
        .model_path = model_path,
        .config = settings.config_file orelse "built-in defaults",
        .prompt_source = if (options.prompt_tokens != null) "tokens" else "text",
        .context = capacity,
        .kv_precision = @tagName(eng.kv_precision),
        .session_bytes = eng.model.session().bytes(),
        .max_tokens = limit,
        .sampling = sampling,
        .sampling_options = sampler.options,
        .seed = options.seed orelse 0,
        .warmup_runs = warmup,
        .measured_runs = stats.measured,
        .load_milliseconds = engine.milliseconds(eng.load),
        .samples = measured_samples,
        .mean_prefill_tokens_per_second = stats.prefill,
        .mean_decode_tokens_per_second = stats.decode,
        .mean_first_token_milliseconds = stats.first,
        .speculative_draft_length = if (speculate) settings.draft_length else null,
        .speculative_p_min = if (speculate) stop.p_min else null,
        .speculative_chain_min = if (speculate) stop.chain_min else null,
        .speculative_max_depth = if (speculate) settings.draft_max_depth else null,
        .mean_speculative_decode_tokens_per_second = stats.speculative_decode,
        .mean_accepted_per_step = stats.accepted_per_step,
        .mean_proposed_per_step = stats.proposed_per_step,
        .decode_speedup = if (stats.decode != null and stats.speculative_decode != null) stats.speculative_decode.? / stats.decode.? else null,
        .prefix = prefix_report,
        .profile = profile,
        .experts = if (counts) |c| c.summary() else null,
    };
    step.clear();
    try report.render(arena, writer, json, ui);
}

/// Resumes a paused profile at the first step after a run's prefill; a
/// `step` hook alone leaves the decode path as it is.
const ProfileGate = struct {
    profile: ?*inference.metal.Profile,
    fn step(context: *anyopaque, _: usize) anyerror!void {
        const self: *ProfileGate = @ptrCast(@alignCast(context));
        if (self.profile) |p| p.paused = false;
    }
};

/// The `--verify-rows` runs: each restores `prefix` and times `limit`
/// forced batches of `rows` rows (the prompt's last token as the first seed,
/// the prompt's first tokens as the fixed drafts), then renders `base` with
/// the verify report and the profile. A capture records one more batch's
/// verify command buffer from the prefix.
fn runVerify(alloc: std.mem.Allocator, eng: *engine.Engine, tokens: []const u32, prefix: *const inference.session.Snapshot, rows: usize, accept: usize, limit: usize, warmup: usize, repeat: usize, capture: ?[:0]const u8, json: bool, writer: *std.Io.Writer, ui: kit.Ui, step: *kit.Step, base: Report) !void {
    var arena_state = std.heap.ArenaAllocator.init(alloc);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const samples = try alloc.alloc(VerifySample, warmup + repeat);
    defer alloc.free(samples);
    var drafts: [inference.engine.max_draft_length]u32 = undefined;
    for (drafts[0 .. rows - 1], 0..) |*d, j| d.* = tokens[j % tokens.len];
    const temperature = base.sampling_options.temperature;
    const gpu = eng.model.gpu();
    var completed: usize = 0;
    var cancelled = false;
    for (0..warmup + repeat) |i| {
        if (i == warmup) if (gpu) |backend| if (backend.profile) |*p| p.clear();
        eng.model.reset();
        try eng.model.restore(prefix);
        var seed = tokens[tokens.len - 1];
        var sums: [6]f64 = @splat(0);
        var batches: usize = 0;
        while (batches < limit) : (batches += 1) {
            interrupt.check() catch {
                cancelled = true;
                break;
            };
            const cost = try inference.engine.forcedBatch(eng, seed, drafts[0 .. rows - 1], accept, temperature, null);
            seed = cost.next;
            for (&sums, [_]std.Io.Duration{ cost.propose, cost.checkpoint, cost.verify, cost.recover, cost.commit, cost.total }) |*sum, d| sum.* += engine.milliseconds(d);
        }
        const n: f64 = @floatFromInt(@max(batches, 1));
        samples[completed] = .{ .warmup = i < warmup, .batches = batches, .propose = sums[0] / n, .checkpoint = sums[1] / n, .verify = sums[2] / n, .recover = sums[3] / n, .commit = sums[4] / n, .total = sums[5] / n };
        completed += 1;
        step.update("{s} {d} of {d}: {d} batches of {d} rows, {d} accepted, {d:.2} ms each", .{ if (i < warmup) "warm-up" else "run", if (i < warmup) i + 1 else i - warmup + 1, if (i < warmup) warmup else repeat, batches, rows, accept, sums[5] / n });
        if (cancelled) break;
    }
    // A cancelled run is shown but never averaged.
    var mean: VerifySample = .{ .warmup = false, .batches = 0, .propose = 0, .checkpoint = 0, .verify = 0, .recover = 0, .commit = 0, .total = 0 };
    var measured: usize = 0;
    for (samples[0..completed], 0..) |s, i| {
        if (s.warmup or (cancelled and i + 1 == completed)) continue;
        measured += 1;
        mean.batches += s.batches;
        inline for (.{ "propose", "checkpoint", "verify", "recover", "commit", "total" }) |field| @field(mean, field) += @field(s, field);
    }
    if (measured > 0) inline for (.{ "propose", "checkpoint", "verify", "recover", "commit", "total" }) |field| {
        @field(mean, field) /= @floatFromInt(measured);
    };
    var profile: ?ProfileReport = null;
    defer if (profile) |p| {
        alloc.free(p.kernels);
        alloc.free(p.families);
    };
    if (gpu) |backend| if (backend.profile) |*p| {
        profile = try profileReport(alloc, p, eng.vocab.tokens.len);
    };
    if (capture) |path| if (!cancelled) {
        eng.model.reset();
        try eng.model.restore(prefix);
        var diagnostic: [1024]u8 = @splat(0);
        _ = inference.engine.forcedBatch(eng, tokens[tokens.len - 1], drafts[0 .. rows - 1], accept, temperature, .{ .path = path, .diagnostic = &diagnostic }) catch |err| {
            std.log.err("{s}", .{std.mem.sliceTo(&diagnostic, 0)});
            return err;
        };
        step.done(.ok, "captured one verify batch", try std.fmt.allocPrint(arena, "{d} rows · position {d} · {s}", .{ rows, prefix.position, path }));
    };
    var report = base;
    report.measured_runs = measured;
    report.verify = .{ .rows = rows, .accepted = accept, .depth = prefix.position, .mean = if (measured > 0) mean else null, .samples = samples[0..completed] };
    report.profile = profile;
    step.clear();
    try report.render(arena, writer, json, ui);
}

test "summary excludes warmups and omits rates that were not measured" {
    const samples = [_]Sample{
        .{ .warmup = true, .prompt_tokens = 4, .generated_tokens = 2, .stop_reason = "token_budget", .prefill_milliseconds = 1000, .first_token_milliseconds = 100, .decode_milliseconds = 200, .prefill_tokens_per_second = 4, .decode_tokens_per_second = 10, .gpu_busy_milliseconds = null },
        .{ .warmup = false, .prompt_tokens = 4, .generated_tokens = 2, .stop_reason = "token_budget", .prefill_milliseconds = 500, .first_token_milliseconds = 50, .decode_milliseconds = 150, .prefill_tokens_per_second = 8, .decode_tokens_per_second = 10, .gpu_busy_milliseconds = null },
        .{ .warmup = false, .prompt_tokens = 4, .generated_tokens = 1, .stop_reason = "eos", .prefill_milliseconds = 500, .first_token_milliseconds = 70, .decode_milliseconds = 70, .prefill_tokens_per_second = 8, .decode_tokens_per_second = null, .gpu_busy_milliseconds = null },
    };
    const stats = summarize(&samples);
    try std.testing.expectEqual(@as(usize, 2), stats.measured);
    try std.testing.expectEqual(@as(f64, 8), stats.prefill.?);
    try std.testing.expectEqual(@as(f64, 10), stats.decode.?);
    try std.testing.expectEqual(@as(f64, 60), stats.first.?);
    try std.testing.expect(summarize(samples[0..1]).decode == null);
    try std.testing.expect(perSecond(0, .{ .nanoseconds = 5 }) == null);
    try std.testing.expectEqual(@as(f64, 2), perSecond(2, .{ .nanoseconds = std.time.ns_per_s }).?);
}

test "profile report divides totals per step, sorts by time, and omits rates without bytes" {
    const alloc = std.testing.allocator;
    var profile: inference.metal.Profile = .{ .command_buffers = 4, .unsampled = 2, .gpu_seconds = 0.4 };
    defer profile.deinit(alloc);
    // 8 dispatches, 0.2 s, 4 GB read: 2/step, 50 ms/step, 20 GB/s.
    try profile.totals.put(alloc, .{ .kernel = .matvec_q4_k, .encoding = 12, .rows = 17408, .columns = 5120 }, .{ .dispatches = 8, .seconds = 0.2, .bytes = 4_000_000_000 });
    try profile.totals.put(alloc, .{ .kernel = .add, .encoding = null, .rows = 0, .columns = 0 }, .{ .dispatches = 40, .seconds = 0.1, .bytes = 0 });
    const report = try profileReport(alloc, &profile, 248320);
    defer alloc.free(report.kernels);
    defer alloc.free(report.families);
    try std.testing.expectEqual(@as(u64, 4), report.steps);
    try std.testing.expectApproxEqAbs(@as(f64, 12.5), report.dispatches_per_step, 1e-12); // (8 + 40 + 2 unsampled) / 4
    try std.testing.expectApproxEqAbs(@as(f64, 75), report.attributed_milliseconds_per_step, 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 100), report.gpu_milliseconds_per_step, 1e-9);
    try std.testing.expectEqual(@as(u64, 2), report.unsampled_dispatches);
    try std.testing.expectEqual(@as(usize, 2), report.kernels.len);
    const first = report.kernels[0];
    try std.testing.expectEqualStrings("matvec_q4_k", first.kernel);
    try std.testing.expectEqualStrings("Q4_K", first.encoding.?);
    try std.testing.expectEqual(@as(u32, 17408), first.rows);
    try std.testing.expectApproxEqAbs(@as(f64, 2), first.dispatches_per_step, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 50), first.milliseconds_per_step, 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), first.share, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 20), first.gigabytes_per_second.?, 1e-9);
    try std.testing.expect(report.kernels[1].encoding == null);
    try std.testing.expect(report.kernels[1].gigabytes_per_second == null);
    // By family: the matrix row then `add` under other, shares of attributed time.
    try std.testing.expectEqual(@as(usize, 2), report.families.len);
    try std.testing.expectEqual(Family.matrix, report.families[0].family);
    try std.testing.expectApproxEqAbs(@as(f64, 50), report.families[0].milliseconds_per_step, 1e-9);
    try std.testing.expectEqual(Family.other, report.families[1].family);
    try std.testing.expectEqual(Family.head, Family.of("matvec_q6_k", 248320, 248320));
    try std.testing.expectEqual(Family.attention, Family.of("attention_decode_wh", 0, 248320));
    try std.testing.expectEqual(Family.attention, Family.of("chunk_scores_h", 0, 248320));
    try std.testing.expectEqual(Family.deltanet, Family.of("delta_chunk", 0, 248320));
    try std.testing.expectEqual(Family.norms, Family.of("add_rmsnorm", 0, 248320));
    try std.testing.expectEqual(Family.selection, Family.of("argmax_partial", 0, 248320));
    // Rendering both ways includes the table, and an empty profile has no steps.
    const wrapped: Report = .{ .backend = "metal", .model_path = "m.gguf", .context = 64, .max_tokens = 2, .warmup_runs = 0, .measured_runs = 1, .load_milliseconds = 1, .samples = &.{}, .mean_prefill_tokens_per_second = null, .mean_decode_tokens_per_second = null, .mean_first_token_milliseconds = null, .profile = report };
    var out: std.Io.Writer.Allocating = .init(alloc);
    defer out.deinit();
    var arena_state = std.heap.ArenaAllocator.init(alloc);
    defer arena_state.deinit();
    try wrapped.render(arena_state.allocator(), &out.writer, false, .plain);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "KERNEL       ENCODING   ROWS  COLS  N/STEP  MS/STEP  SHARE  GB/S\nmatvec_q4_k  Q4_K      17408  5120     2.0   50.000  66.7%  20.0\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "Profiled  one encoder per dispatch") != null);
    out.clearRetainingCapacity();
    try wrapped.render(arena_state.allocator(), &out.writer, true, .plain);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "\"gigabytes_per_second\": 20") != null);
    var empty: inference.metal.Profile = .{};
    const none = try profileReport(alloc, &empty, 248320);
    defer alloc.free(none.kernels);
    defer alloc.free(none.families);
    try std.testing.expectEqual(@as(u64, 0), none.steps);
    try std.testing.expectEqual(@as(f64, 0), none.attributed_milliseconds_per_step);
}

test "report renders text and JSON from the same samples" {
    const samples = [_]Sample{
        .{ .warmup = false, .prompt_tokens = 4, .generated_tokens = 2, .stop_reason = "token_budget", .prefill_milliseconds = 500, .first_token_milliseconds = 50, .decode_milliseconds = 150, .prefill_tokens_per_second = 8, .decode_tokens_per_second = null, .gpu_busy_milliseconds = null },
    };
    const report: Report = .{ .backend = "cpu", .build_mode = "ReleaseFast", .model_path = "m.gguf", .context = 64, .max_tokens = 2, .warmup_runs = 0, .measured_runs = 1, .load_milliseconds = 12.5, .samples = &samples, .mean_prefill_tokens_per_second = 8, .mean_decode_tokens_per_second = null, .mean_first_token_milliseconds = 50 };
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    try report.render(arena, &out.writer, true, .plain);
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, out.written(), .{});
    defer parsed.deinit();
    try std.testing.expect(parsed.value.object.get("mean_decode_tokens_per_second") == null);
    // 8.0 serializes as the integer literal 8.
    try std.testing.expectEqual(@as(i64, 8), parsed.value.object.get("mean_prefill_tokens_per_second").?.integer);
    out.clearRetainingCapacity();
    try report.render(arena, &out.writer, false, .plain);
    try std.testing.expectEqualStrings(
        \\◆ m.gguf · cpu (ReleaseFast)
        \\
        \\Model    m.gguf
        \\Config   built-in defaults
        \\Context  64 tokens · KV f32 · session 0 B
        \\Output   budget of 2 tokens · greedy · prompt from text
        \\Load     12.5 ms
        \\
        \\RUN  PROMPT  GEN  STOP          PREFILL MS  PP TOK/S  FIRST MS  DECODE MS  TG TOK/S
        \\1         4    2  token_budget       500.0      8.00      50.0      150.0        --
        \\
        \\Mean of 1 measured run
        \\  Prefill      8.00 tok/s
        \\  Decode       --
        \\  First token  50.0 ms
        \\
    , out.written());
}

test {
    _ = prefix_cache;
}
