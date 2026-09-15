//! Repeated prefill/decode measurements on one loaded engine. Every sample is
//! reported; means are computed over the measured (non-warmup) samples only.
//! Rates that cannot be measured are absent, never fabricated.
const std = @import("std");
const inference = @import("inference");
const engine = @import("engine.zig");
const generate = @import("generate.zig");
const config = @import("config.zig");
const style = @import("tui/style.zig");
const interrupt = @import("interrupt.zig");

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
};

/// Dispatches a Qwen token records (~1,240) with margin; the plan is not asked
/// because the bench does not know the model, only the backend.
const profile_dispatch_capacity = 4096;

/// One row of the per-kernel table: totals divided by the number of profiled
/// steps (command buffers), so the numbers are "per token".
pub const KernelTime = struct {
    kernel: []const u8,
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
};

pub const Report = struct {
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
    profile: ?ProfileReport = null,

    pub fn render(self: Report, out: *std.Io.Writer, json: bool, sty: style.Style) !void {
        if (json) {
            try std.json.Stringify.value(self, .{ .whitespace = .indent_2, .emit_null_optional_fields = false }, out);
            return out.writeByte('\n');
        }
        const label = sty.on(.label);
        const off = sty.off();
        try out.print("{s}Backend:{s} {s} ({s})\n{s}Model:{s} {s}{s}{s}\n{s}Config:{s} {s}{s}{s}\n", .{ label, off, self.backend, self.build_mode, label, off, sty.on(.code), self.model_path, off, label, off, sty.on(.code), self.config, off });
        try out.print("{s}Context:{s} {d} tokens, KV {s} (session {d:.1} MiB), output budget: {d}, {s}, prompt from {s}{s}\n{s}Load:{s} {d:.1} ms\n\n", .{ label, off, self.context, self.kv_precision, @as(f64, @floatFromInt(self.session_bytes)) / (1024 * 1024), self.max_tokens, self.sampling, self.prompt_source, if (self.profile != null) ", profiled (one encoder per dispatch; rates not comparable)" else "", label, off, self.load_milliseconds });
        try out.print("{s}run    prompt  gen  stop           prefill ms  pp tok/s  first ms   decode ms  tg tok/s   gpu ms  fallbacks{s}\n", .{ sty.on(.header), off });
        for (self.samples, 0..) |s, i| {
            try out.print("{s}{d: <4} {d: >7} {d: >4}  {s: <13} {d: >11.1} ", .{ if (s.warmup) "w" else " ", i, s.prompt_tokens, s.generated_tokens, s.stop_reason, s.prefill_milliseconds });
            try rate(out, s.prefill_tokens_per_second);
            try out.print(" {d: >9.1} {d: >11.1} ", .{ s.first_token_milliseconds, s.decode_milliseconds });
            try rate(out, s.decode_tokens_per_second);
            if (s.gpu_busy_milliseconds) |gpu| try out.print(" {d: >8.1}", .{gpu}) else try out.writeAll("        —");
            if (s.topk_fallbacks) |n| try out.print(" {d: >10}", .{n}) else try out.writeAll("          —");
            try out.writeByte('\n');
        }
        try out.print("\n{s}Measured mean over {d} runs:{s} prefill {s}", .{ sty.on(.bold), self.measured_runs, off, sty.on(.number) });
        try rate(out, self.mean_prefill_tokens_per_second);
        try out.print("{s} tok/s, decode {s}", .{ off, sty.on(.number) });
        try rate(out, self.mean_decode_tokens_per_second);
        try out.print("{s} tok/s, first token ", .{off});
        if (self.mean_first_token_milliseconds) |ms| try out.print("{d:.1} ms\n", .{ms}) else try out.writeAll("—\n");
        if (self.profile) |p| try renderProfile(p, out, sty);
    }
    fn rate(out: *std.Io.Writer, value: ?f64) !void {
        if (value) |v| try out.print("{d: >9.2}", .{v}) else try out.writeAll("        —");
    }
    fn renderProfile(p: ProfileReport, out: *std.Io.Writer, sty: style.Style) !void {
        try out.print("\n{s}Per-kernel GPU time over {d} measured steps:{s} {d:.0} dispatches/step, {d:.1} ms/step attributed of {d:.1} ms/step command-buffer time, {d} unsampled dispatches\n", .{ sty.on(.bold), p.steps, sty.off(), p.dispatches_per_step, p.attributed_milliseconds_per_step, p.gpu_milliseconds_per_step, p.unsampled_dispatches });
        try out.print("{s}kernel             encoding    rows    cols   n/step   ms/step  share    GB/s{s}\n", .{ sty.on(.header), sty.off() });
        for (p.kernels) |k| {
            try out.print("{s: <18} {s: <8} {d: >7} {d: >7} {d: >8.1} {d: >9.3} {d: >5.1}%", .{ k.kernel, k.encoding orelse "—", k.rows, k.columns, k.dispatches_per_step, k.milliseconds_per_step, k.share * 100 });
            if (k.gigabytes_per_second) |gbps| try out.print(" {d: >7.1}\n", .{gbps}) else try out.writeAll("       —\n");
        }
    }
};

/// Pure aggregation of a backend profile into per-step rows sorted by time,
/// so the table is testable without a GPU. Caller frees `kernels`.
pub fn profileReport(alloc: std.mem.Allocator, profile: *const inference.metal.Profile) !ProfileReport {
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
    return .{
        .steps = steps,
        .dispatches_per_step = @as(f64, @floatFromInt(dispatches)) * per_step,
        .attributed_milliseconds_per_step = attributed * 1000 * per_step,
        .gpu_milliseconds_per_step = profile.gpu_seconds * 1000 * per_step,
        .unsampled_dispatches = profile.unsampled,
        .kernels = kernels,
    };
}

fn perSecond(count: usize, duration: std.Io.Duration) ?f64 {
    if (count == 0 or duration.nanoseconds <= 0) return null;
    return @as(f64, @floatFromInt(count)) / (@as(f64, @floatFromInt(duration.nanoseconds)) / std.time.ns_per_s);
}

/// Pure aggregation over samples so the statistics are testable without a model.
pub fn summarize(samples: []const Sample) struct { prefill: ?f64, decode: ?f64, first: ?f64, measured: usize } {
    var prefill_sum: f64 = 0;
    var prefill_n: usize = 0;
    var decode_sum: f64 = 0;
    var decode_n: usize = 0;
    var first_sum: f64 = 0;
    var measured: usize = 0;
    for (samples) |s| {
        if (s.warmup or std.mem.eql(u8, s.stop_reason, "cancelled")) continue;
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
    };
}

pub fn run(alloc: std.mem.Allocator, io: std.Io, model_path: []const u8, settings: config.Resolved, options: Options, json: bool, writer: *std.Io.Writer, sty: style.Style) !void {
    const user: ?[]u8 = if (options.prompt_tokens == null) try engine.readPrompt(alloc, io, options.prompt, options.prompt_file) else null;
    defer if (user) |text| alloc.free(text);
    const limit = settings.max_tokens;
    const capacity = settings.ctx_size;
    const repeat = options.repeat orelse 3;
    const warmup = options.warmup orelse 1;
    if (limit == 0 or limit > config.max_output_tokens or capacity == 0 or capacity > config.max_context or repeat == 0 or repeat > 100 or warmup > 100) return error.InvalidGenerationBudget;
    var eng = try engine.Engine.open(alloc, io, model_path, settings.backend, capacity, settings.kv_precision);
    defer eng.deinit();
    const gpu: ?*inference.metal.Backend = eng.model.gpu();
    if (options.profile) {
        const backend = gpu orelse return error.ProfileRequiresMetal;
        var diagnostic: [1024]u8 = @splat(0);
        backend.enableProfiling(profile_dispatch_capacity, &diagnostic) catch |err| {
            std.log.err("{s}", .{std.mem.sliceTo(&diagnostic, 0)});
            return err;
        };
    }
    interrupt.install();
    const tokens = if (options.prompt_tokens) |path| try engine.readPromptTokens(alloc, io, path, eng.vocab.tokens.len) else blk: {
        const prompt = try eng.prompt(user.?, options.raw, .off);
        defer alloc.free(prompt);
        break :blk try eng.encode(prompt);
    };
    defer alloc.free(tokens);
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
    const samples = try alloc.alloc(Sample, warmup + repeat);
    defer alloc.free(samples);
    var trace: generate.Trace = .{ .io = io, .directory = null, .started = std.Io.Clock.awake.now(io) };
    var completed: usize = 0;
    for (samples, 0..) |*sample, i| {
        // Every run starts from an empty session and an empty token history.
        eng.model.reset();
        history.reset();
        // Warm-up dispatches are timed too but must not enter the profile.
        if (i == warmup) if (gpu) |backend| if (backend.profile) |*p| p.clear();
        // Decode covers the steps after the first sampled token; the first
        // token's latency (prefill included) is reported separately.
        const outcome = try generate.runLoop(&eng, tokens, limit, &sampler, &history, logits, candidates, generated, &trace, null);
        const t = outcome.timing;
        const decode_steps = if (t.generated_tokens > 1) t.generated_tokens - 1 else 0;
        sample.* = .{
            .warmup = i < warmup,
            .prompt_tokens = t.prompt_tokens,
            .generated_tokens = t.generated_tokens,
            .stop_reason = @tagName(outcome.stop),
            .prefill_milliseconds = engine.milliseconds(t.prefill),
            .first_token_milliseconds = engine.milliseconds(t.first_token),
            .decode_milliseconds = engine.milliseconds(t.decode),
            .prefill_tokens_per_second = perSecond(t.prompt_tokens, t.prefill),
            .decode_tokens_per_second = perSecond(decode_steps, t.decode),
            .gpu_busy_milliseconds = if (t.gpu_seconds) |s| s * 1000 else null,
            .topk_fallbacks = t.topk_fallbacks,
        };
        if (!json) {
            try writer.print("{s} run {d}: {d} prompt, {d} generated, {s}\n", .{ if (i < warmup) "warmup" else "measured", i, t.prompt_tokens, t.generated_tokens, @tagName(outcome.stop) });
            try writer.flush();
        }
        completed += 1;
        // A cancelled run is reported as its own sample and ends the benchmark;
        // partial runs must not be averaged as if they had finished.
        if (outcome.stop == .cancelled) break;
    }
    const measured_samples = samples[0..completed];
    const stats = summarize(measured_samples);
    var profile: ?ProfileReport = null;
    defer if (profile) |p| alloc.free(p.kernels);
    if (gpu) |backend| if (backend.profile) |*p| {
        profile = try profileReport(alloc, p);
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
        .profile = profile,
    };
    if (!json) try writer.writeByte('\n');
    try report.render(writer, json, sty);
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
    const report = try profileReport(alloc, &profile);
    defer alloc.free(report.kernels);
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
    // Rendering both ways includes the table, and an empty profile has no steps.
    const wrapped: Report = .{ .backend = "metal", .model_path = "m.gguf", .context = 64, .max_tokens = 2, .warmup_runs = 0, .measured_runs = 1, .load_milliseconds = 1, .samples = &.{}, .mean_prefill_tokens_per_second = null, .mean_decode_tokens_per_second = null, .mean_first_token_milliseconds = null, .profile = report };
    var out: std.Io.Writer.Allocating = .init(alloc);
    defer out.deinit();
    try wrapped.render(&out.writer, false, .none);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "matvec_q4_k        Q4_K       17408    5120      2.0    50.000  66.7%    20.0") != null);
    out.clearRetainingCapacity();
    try wrapped.render(&out.writer, true, .none);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "\"gigabytes_per_second\": 20") != null);
    var empty: inference.metal.Profile = .{};
    const none = try profileReport(alloc, &empty);
    defer alloc.free(none.kernels);
    try std.testing.expectEqual(@as(u64, 0), none.steps);
    try std.testing.expectEqual(@as(f64, 0), none.attributed_milliseconds_per_step);
}

test "report renders text and JSON from the same samples" {
    const samples = [_]Sample{
        .{ .warmup = false, .prompt_tokens = 4, .generated_tokens = 2, .stop_reason = "token_budget", .prefill_milliseconds = 500, .first_token_milliseconds = 50, .decode_milliseconds = 150, .prefill_tokens_per_second = 8, .decode_tokens_per_second = null, .gpu_busy_milliseconds = null },
    };
    const report: Report = .{ .backend = "cpu", .model_path = "m.gguf", .context = 64, .max_tokens = 2, .warmup_runs = 0, .measured_runs = 1, .load_milliseconds = 12.5, .samples = &samples, .mean_prefill_tokens_per_second = 8, .mean_decode_tokens_per_second = null, .mean_first_token_milliseconds = 50 };
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    try report.render(&out.writer, true, .none);
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, out.written(), .{});
    defer parsed.deinit();
    try std.testing.expect(parsed.value.object.get("mean_decode_tokens_per_second") == null);
    // 8.0 serializes as the integer literal 8.
    try std.testing.expectEqual(@as(i64, 8), parsed.value.object.get("mean_prefill_tokens_per_second").?.integer);
    out.clearRetainingCapacity();
    try report.render(&out.writer, false, .none);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "decode         — tok/s") != null);
}
