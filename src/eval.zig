//! `nuclis eval`: teacher-forced perplexity over a text file. The text is
//! tokenized raw, cut into windows of `ctx` tokens, and each window runs from
//! an empty session with its first token replaced by the model's BOS (when
//! it has one); the negative log-likelihood of every actual next token in the
//! window's second half is summed in F64. This is the reference's
//! `llama-perplexity` method, so `--reference` compares against its run on
//! the same file (docs/reference/eval.md).
const std = @import("std");
const inference = @import("inference");
const engine = @import("engine.zig");
const config = @import("config.zig");
const interrupt = @import("interrupt.zig");
const style = @import("tui/style.zig");

/// What only this command reads from the command line; the model, backend,
/// cache precision, and forced profile arrive resolved (`config.Resolved`).
pub const Options = struct {
    file: ?[]const u8 = null,
    /// Tokens per window; the reference's when one is given, else `default_ctx`.
    ctx: ?usize = null,
    /// Windows evaluated from the start of the text; the reference's when one
    /// is given, else every whole window.
    chunks: ?usize = null,
    reference: ?[]const u8 = null,
};

pub const default_ctx = 512;
/// The largest text read; a corpus is never truncated, a larger one refused.
pub const max_text_bytes = 16 * 1024 * 1024;
/// The bound on |ppl − reference| / reference, unless the reference states
/// its own.
pub const tolerance = 0.005;

/// A pinned reference run, as `scripts/reference-perplexity.py` writes it.
pub const Reference = struct {
    tool: []const u8,
    revision: []const u8,
    text_sha256: []const u8,
    ctx: usize,
    chunks: usize,
    ppl: f64,
    ppl_error: f64,
    /// A wider bound for a model whose reference disagrees with itself by
    /// more than the default (the mixture-of-experts file's routing ties).
    tolerance: ?f64 = null,
};

pub const Comparison = struct {
    tool: []const u8,
    revision: []const u8,
    ppl: f64,
    ppl_error: f64,
    /// (ours − reference) / reference.
    relative_difference: f64,
    tolerance: f64,
    passed: bool,
};

pub const Report = struct {
    schema_version: u32 = 1,
    model: []const u8,
    model_path: []const u8,
    backend: []const u8,
    kv_precision: []const u8,
    text_bytes: usize,
    text_sha256: []const u8,
    text_tokens: usize,
    ctx: usize,
    chunks: usize,
    /// The id written over each window's first token; absent when none.
    bos: ?u32,
    scored_tokens: usize,
    nll: f64,
    ppl: f64,
    /// The perplexity's standard error, from the NLL's sample variance.
    ppl_error: f64,
    /// The running perplexity after each window, as the reference prints it.
    chunk_ppl: []const f64,
    eval_milliseconds: f64,
    /// Tokens fed (scored or not) per second of evaluation.
    tokens_per_second: f64,
    reference: ?Comparison = null,

    pub fn render(self: Report, out: *std.Io.Writer, json: bool, sty: style.Style) !void {
        if (json) {
            try std.json.Stringify.value(self, .{ .whitespace = .indent_2, .emit_null_optional_fields = false }, out);
            return out.writeByte('\n');
        }
        const label = sty.on(.label);
        const number = sty.on(.number);
        const off = sty.off();
        try out.print("{s}Perplexity:{s} {s}{d:.4}{s} ± {d:.5} (nll {d:.6}), {d} tokens scored in {d:.1} s, {d:.1} tok/s fed\n", .{ label, off, number, self.ppl, off, self.ppl_error, self.nll, self.scored_tokens, self.eval_milliseconds / 1000, self.tokens_per_second });
        if (self.reference) |r| {
            const verdict = if (r.passed) sty.on(.success) else sty.on(.error_text);
            try out.print("{s}Reference:{s} {s}{d:.4}{s} ± {d:.5} ({s} {s}), difference {d:.3} % (bound {d:.1} %): {s}{s}{s}\n", .{ label, off, number, r.ppl, off, r.ppl_error, r.tool, r.revision[0..@min(r.revision.len, 9)], r.relative_difference * 100, r.tolerance * 100, verdict, if (r.passed) "pass" else "FAIL", off });
        }
    }
};

/// −log softmax(`logits`)[`target`], accumulated in F64: the maximum is
/// subtracted in F32 and the exponentials summed in F64, as the reference does.
pub fn rowNll(logits: []const f32, target: u32) f64 {
    var max = logits[0];
    for (logits[1..]) |v| max = @max(max, v);
    var sum: f64 = 0;
    for (logits) |v| sum += @exp(v - max);
    return -(@as(f64, logits[target] - max) - @log(sum));
}

/// The running NLL sums and the perplexity with its standard error.
pub const Accumulator = struct {
    nll: f64 = 0,
    nll2: f64 = 0,
    count: usize = 0,

    pub fn add(self: *Accumulator, v: f64) void {
        self.nll += v;
        self.nll2 += v * v;
        self.count += 1;
    }
    pub fn mean(self: Accumulator) f64 {
        return self.nll / @as(f64, @floatFromInt(self.count));
    }
    pub fn ppl(self: Accumulator) f64 {
        return @exp(self.mean());
    }
    /// The perplexity's standard error: the NLL's sample standard error
    /// scaled by the perplexity (first-order), 0 below two samples.
    pub fn pplError(self: Accumulator) f64 {
        if (self.count < 2) return 0;
        const n: f64 = @floatFromInt(self.count);
        const m = self.mean();
        const variance = self.nll2 / n - m * m;
        if (variance <= 0) return 0;
        return @sqrt(variance / (n - 1)) * self.ppl();
    }
};

/// Row `first` of a window is the first whose next token is scored.
pub fn firstScored(ctx: usize) usize {
    return ctx / 2;
}

/// The windows a text of `tokens` yields at `ctx`, capped by `chunks`; an
/// error when not one whole window fits or more are asked for than exist.
pub fn windowCount(tokens: usize, ctx: usize, chunks: ?usize) !usize {
    const whole = tokens / ctx;
    if (whole == 0) return error.TextTooShort;
    const wanted = chunks orelse return whole;
    if (wanted == 0 or wanted > whole) return error.TextTooShort;
    return wanted;
}

/// Copies window `index` of `tokens` into `out` (`ctx` ids) and writes `bos`
/// over its first token, which is context only and never scored.
pub fn window(tokens: []const u32, index: usize, out: []u32, bos: ?u32) void {
    @memcpy(out, tokens[index * out.len ..][0..out.len]);
    if (bos) |id| out[0] = id;
}

/// Cancellation only: Ctrl-C and Io cancellation at every layer boundary. A
/// run's length is bounded by its window count, not a clock.
const Watch = struct {
    io: std.Io,
    fn check(context: *anyopaque) !void {
        const self: *Watch = @ptrCast(@alignCast(context));
        try self.io.checkCancel();
        try interrupt.check();
    }
    fn observer(self: *Watch) inference.engine.Observer {
        return .{ .context = self, .check = check };
    }
};

pub fn run(alloc: std.mem.Allocator, io: std.Io, model_path: []const u8, settings: config.Resolved, options: Options, json: bool, out: *std.Io.Writer, sty: style.Style, diag: *config.Diagnostic) !void {
    const path = options.file orelse return error.MissingTextFile;
    var parsed_reference: ?std.json.Parsed(Reference) = null;
    defer if (parsed_reference) |*p| p.deinit();
    if (options.reference) |reference_path| {
        const bytes = std.Io.Dir.cwd().readFileAlloc(io, reference_path, alloc, .limited(64 * 1024)) catch |err| {
            diag.set("could not read the reference {s}", .{reference_path});
            return err;
        };
        defer alloc.free(bytes);
        parsed_reference = std.json.parseFromSlice(Reference, alloc, bytes, .{ .ignore_unknown_fields = true, .allocate = .alloc_always }) catch |err| {
            diag.set("{s} is not a reference-perplexity.py report", .{reference_path});
            return err;
        };
    }
    const reference: ?Reference = if (parsed_reference) |p| p.value else null;
    const ctx = options.ctx orelse if (reference) |r| r.ctx else default_ctx;
    if (ctx < 2 or ctx > config.max_context) return error.InvalidGenerationBudget;
    if (reference) |r| if (r.ctx != ctx or (options.chunks != null and options.chunks.? != r.chunks)) {
        diag.set("the reference ran {d} windows of {d} tokens; drop --ctx-size and --chunks to use its", .{ r.chunks, r.ctx });
        return error.ReferenceMismatch;
    };
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, alloc, .limited(max_text_bytes)) catch |err| {
        if (err == error.StreamTooLong) diag.set("{s} is larger than {d} bytes", .{ path, max_text_bytes });
        return err;
    };
    defer alloc.free(text);
    var digest: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(text, &digest, .{});
    const text_sha256 = std.fmt.bytesToHex(digest, .lower);
    if (reference) |r| if (!std.mem.eql(u8, r.text_sha256, &text_sha256)) {
        diag.set("{s} (sha256 {s}) is not the text the reference ran (sha256 {s})", .{ path, text_sha256[0..16], r.text_sha256[0..@min(r.text_sha256.len, 16)] });
        return error.ReferenceMismatch;
    };

    var eng = try engine.Engine.open(alloc, io, model_path, settings.backend, ctx, settings.kv_precision, settings.forced_profile, .none);
    defer eng.deinit();
    interrupt.install();
    // The reference tokenizes without parsing special markers. The merge
    // work is linear in the text, so its bound scales with it (the text is
    // itself bounded).
    const encoded = try eng.encoder.encode(alloc, text, false, .{ .input_bytes = text.len, .output_tokens = text.len + 1, .bpe_work = @max(64 * 1024 * 1024, 64 * text.len) });
    defer alloc.free(encoded);
    const bos = eng.textBos();
    // The text starts with BOS as a whole, and each window again over its
    // first token: window `i` is ids `[i·ctx, (i+1)·ctx)` of this array.
    const tokens = try alloc.alloc(u32, encoded.len + @intFromBool(bos != null));
    defer alloc.free(tokens);
    if (bos) |id| tokens[0] = id;
    @memcpy(tokens[tokens.len - encoded.len ..], encoded);
    const chunks = windowCount(tokens.len, ctx, options.chunks orelse if (reference) |r| r.chunks else null) catch |err| {
        diag.set("{s} is {d} tokens: {d} windows of {d} need {d}", .{ path, tokens.len, options.chunks orelse 1, ctx, (options.chunks orelse 1) * ctx });
        return err;
    };
    const vocabulary = eng.vocab.tokens.len;
    const first = firstScored(ctx);
    // The unscored half is a plain prefill; the scored half is read back in
    // pieces of one prefill chunk, bounding the host rows (~250 MB at 248K).
    const piece = @min(ctx - first, inference.engine.prefill_chunk);
    const rows = try alloc.alloc(f32, piece * vocabulary);
    defer alloc.free(rows);
    const ids = try alloc.alloc(u32, ctx);
    defer alloc.free(ids);
    const chunk_ppl = try alloc.alloc(f64, chunks);
    defer alloc.free(chunk_ppl);

    if (!json) {
        try out.print("{s}Model:{s} {s} {s}({s}, {s}, kv {s}){s}\n", .{ sty.on(.label), sty.off(), eng.name, sty.on(.dim), model_path, @tagName(settings.backend), @tagName(eng.kv_precision), sty.off() });
        try out.print("{s}Text:{s} {s}, {d} bytes, {d} tokens, sha256 {s}\n", .{ sty.on(.label), sty.off(), path, text.len, tokens.len, text_sha256[0..16] });
        try out.print("{s}Windows:{s} {d} × {d} tokens, the last {d} of each scored, BOS {s}\n", .{ sty.on(.label), sty.off(), chunks, ctx, ctx - first - 1, if (bos != null) "written over the first" else "none" });
        try out.flush();
    }
    var watch: Watch = .{ .io = io };
    const observer = watch.observer();
    var sums: Accumulator = .{};
    const started = std.Io.Clock.awake.now(io);
    for (0..chunks) |c| {
        // A cancelled run ends the running line, so the error starts its own.
        errdefer if (!json and c > 0) {
            out.writeByte('\n') catch {};
            out.flush() catch {};
        };
        window(tokens, c, ids, bos);
        eng.model.reset();
        if (first > 0) try eng.model.prefill(ids[0..first], null, null, null, null, null, observer);
        var at = first;
        while (at < ctx) {
            const count = @min(piece, ctx - at);
            try eng.model.prefillRows(ids[at..][0..count], vocabulary, rows[0 .. count * vocabulary], observer);
            // Row `at + j` predicts token `at + j + 1`; the window's last row
            // has no next token.
            for (0..count) |j| {
                if (at + j + 1 >= ctx) break;
                sums.add(rowNll(rows[j * vocabulary ..][0..vocabulary], ids[at + j + 1]));
            }
            at += count;
        }
        chunk_ppl[c] = sums.ppl();
        if (!json) {
            try out.print("[{d}]{d:.4}{s}", .{ c + 1, chunk_ppl[c], if (c + 1 == chunks) "\n" else "," });
            try out.flush();
        }
    }
    const elapsed = started.durationTo(std.Io.Clock.awake.now(io));
    const seconds = @as(f64, @floatFromInt(elapsed.toNanoseconds())) / 1e9;
    var report: Report = .{
        .model = eng.name,
        .model_path = model_path,
        .backend = @tagName(settings.backend),
        .kv_precision = @tagName(eng.kv_precision),
        .text_bytes = text.len,
        .text_sha256 = &text_sha256,
        .text_tokens = tokens.len,
        .ctx = ctx,
        .chunks = chunks,
        .bos = bos,
        .scored_tokens = sums.count,
        .nll = sums.mean(),
        .ppl = sums.ppl(),
        .ppl_error = sums.pplError(),
        .chunk_ppl = chunk_ppl,
        .eval_milliseconds = seconds * 1000,
        .tokens_per_second = @as(f64, @floatFromInt(chunks * ctx)) / seconds,
    };
    if (reference) |r| {
        const difference = (report.ppl - r.ppl) / r.ppl;
        const bound = r.tolerance orelse tolerance;
        report.reference = .{ .tool = r.tool, .revision = r.revision, .ppl = r.ppl, .ppl_error = r.ppl_error, .relative_difference = difference, .tolerance = bound, .passed = @abs(difference) <= bound };
    }
    try report.render(out, json, sty);
    if (report.reference) |r| if (!r.passed) {
        diag.set("perplexity {d:.4} is {d:.3} % from the reference's {d:.4} (bound {d:.1} %)", .{ report.ppl, r.relative_difference * 100, r.ppl, r.tolerance * 100 });
        return error.ReferenceMismatch;
    };
}

test "a row's NLL is the negative log-softmax of its target" {
    // Uniform over four: −log(1/4).
    try std.testing.expectApproxEqAbs(@log(4.0), rowNll(&.{ 1, 1, 1, 1 }, 2), 1e-12);
    // Shift invariance: adding a constant to every logit changes nothing.
    const a = rowNll(&.{ 0.5, -2, 3, 1 }, 2);
    const b = rowNll(&.{ 100.5, 98, 103, 101 }, 2);
    try std.testing.expectApproxEqAbs(a, b, 1e-6);
    // Against a direct evaluation.
    const direct = -(3.0 - @log(@exp(0.5) + @exp(-2.0) + @exp(3.0) + @exp(1.0)));
    try std.testing.expectApproxEqAbs(direct, a, 1e-6);
    // A large logit gap does not overflow.
    try std.testing.expect(std.math.isFinite(rowNll(&.{ 1e4, 0, -1e4 }, 2)));
}

test "the accumulator's perplexity and error follow the reference's formulas" {
    var sums: Accumulator = .{};
    for ([_]f64{ 1, 2, 3, 4 }) |v| sums.add(v);
    try std.testing.expectEqual(@as(usize, 4), sums.count);
    try std.testing.expectApproxEqAbs(@exp(2.5), sums.ppl(), 1e-12);
    // Population variance 1.25 over n − 1 = 3, times the perplexity.
    try std.testing.expectApproxEqAbs(@sqrt(1.25 / 3.0) * @exp(2.5), sums.pplError(), 1e-9);
    var one: Accumulator = .{};
    one.add(0.7);
    try std.testing.expectEqual(@as(f64, 0), one.pplError());
}

test "windows cut whole ctx-token spans with BOS over the first token" {
    try std.testing.expectEqual(@as(usize, 3), try windowCount(1600, 512, null));
    try std.testing.expectEqual(@as(usize, 2), try windowCount(1600, 512, 2));
    try std.testing.expectError(error.TextTooShort, windowCount(511, 512, null));
    try std.testing.expectError(error.TextTooShort, windowCount(1600, 512, 4));
    try std.testing.expectError(error.TextTooShort, windowCount(1600, 512, 0));
    try std.testing.expectEqual(@as(usize, 256), firstScored(512));
    const tokens = [_]u32{ 10, 11, 12, 13, 14, 15, 16 };
    var out: [3]u32 = undefined;
    window(&tokens, 1, &out, 2);
    try std.testing.expectEqualSlices(u32, &.{ 2, 14, 15 }, &out);
    window(&tokens, 0, &out, null);
    try std.testing.expectEqualSlices(u32, &.{ 10, 11, 12 }, &out);
}
