//! Explicit full-model check of OmniVoice / VoiceTut-TTS (`audio.cpp` GGUF).
//!
//!   omnivoice-check MODEL.gguf vocab
//!   omnivoice-check MODEL.gguf codec|codec-encode|codec-decode [--fixtures DIR] [--max-relative-rms X]
//!   omnivoice-check MODEL.gguf trace [--fixtures DIR] [--max-relative-rms X] [--max-abs X]
//!   omnivoice-check MODEL.gguf speak --text T --out FILE.wav [--backend cpu|metal] [--seed N] [--steps N] [--duration D] [--speed S] [--lang L] [--instruct I]
const std = @import("std");
const inference = @import("inference");
const check_stats = @import("check_stats.zig");
const omnivoice = inference.models.omnivoice;
const text_mod = inference.models.omnivoice_text;
const runtime = inference.models.omnivoice_runtime;
const speech = inference.models.omnivoice_speech;
const hf_json = inference.hf_tokenizer;
const weights = inference.weights;
const hc = inference.audio.higgs_codec;
const hce = inference.audio.higgs_codec_encoder;
const audio_mod = inference.audio.audio;

const usage = "usage: omnivoice-check MODEL.gguf vocab|codec|codec-encode|codec-decode|trace|speak [options]\n";

const Vector = struct {
    name: []const u8,
    text: []const u8,
    expected: []const u32,
};

const standalone_vectors = [_]Vector{
    .{
        .name = "english",
        .text = "The quick brown fox jumps over the lazy dog.",
        .expected = &.{ 785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562, 13 },
    },
    .{
        .name = "egyptian_arabic",
        .text = "إزيك يا باشا؟ عامل إيه النهارده؟",
        .expected = &.{ 91962, 125089, 31073, 129720, 82611, 129637, 128332, 43982, 124466, 85153, 126236, 124080, 16157, 35038, 91335, 128332 },
    },
    .{
        .name = "code_switching",
        .text = "عندي meeting بكرة الصبح، فهحاول أخلص الشغل بدري النهارده.",
        .expected = &.{ 127206, 79820, 6438, 27846, 126176, 124265, 124449, 68785, 45577, 16157, 131725, 63415, 124429, 41593, 124209, 125718, 138829, 123832, 124080, 16157, 35038, 91335, 13 },
    },
    .{
        .name = "arabic_diacritics",
        .text = "مَرْحَبًا بِكُمْ",
        .expected = &.{ 10176, 27910, 11071, 59397, 29825, 27910, 21360, 127837, 27846, 52704, 31073, 64604, 10176, 59397 },
    },
    .{
        .name = "punct_ws",
        .text = "Hello,  world!\n123 — test.",
        .expected = &.{ 9707, 11, 220, 1879, 4894, 16, 17, 18, 1959, 1273, 13 },
    },
};

const PromptVector = struct {
    name: []const u8,
    request: text_mod.Request,
    expected: []const u32,
};

const prompt_vectors = [_]PromptVector{
    .{
        .name = "auto_prompt",
        .request = .{ .text = "عندي meeting بكرة الصبح، فهحاول أخلص الشغل بدري النهارده." },
        .expected = &.{ 151670, 277, 89, 151671, 151672, 4064, 151673, 151674, 127206, 79820, 6438, 27846, 126176, 124265, 124449, 68785, 45577, 16157, 131725, 63415, 124429, 41593, 124209, 125718, 138829, 123832, 124080, 16157, 35038, 91335, 13, 151675 },
    },
    .{
        .name = "design_prompt",
        .request = .{ .text = "Hello world!", .language = "en", .instruct = "female, young adult, british accent" },
        .expected = &.{ 151670, 268, 151671, 151672, 42634, 11, 3908, 6683, 11, 93927, 29100, 151673, 151674, 9707, 1879, 0, 151675 },
    },
    .{
        .name = "clone_prompt",
        .request = .{ .text = "إزيك النهارده؟", .ref_text = "أهلاً بيك.", .ref_frames = 50, .denoise = true },
        .expected = &.{ 151669, 151670, 277, 89, 151671, 151672, 4064, 151673, 151674, 69682, 16157, 80970, 124376, 27846, 126300, 13, 85153, 125089, 31073, 124080, 16157, 35038, 91335, 128332, 151675 },
    },
};

const uncond_expected = [_]u32{ 151670, 4064, 151671, 151672, 4064, 151673, 151674, 151675 };

fn checkIds(name: []const u8, got: []const u32, expected: []const u32) bool {
    if (std.mem.eql(u32, got, expected)) return true;
    std.debug.print("FAIL {s}: got {any}, expected {any}\n", .{ name, got, expected });
    return false;
}

fn readF32File(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) ![]f32 {
    const path = try std.fs.path.join(gpa, &.{ dir, name });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(64 << 20));
    defer gpa.free(bytes);
    if (bytes.len % 4 != 0) return error.InvalidFixture;
    const out = try gpa.alloc(f32, bytes.len / 4);
    for (out, 0..) |*dst, i| dst.* = @bitCast(std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little));
    return out;
}

fn readCodesFile(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) ![]hc.Frame {
    const path = try std.fs.path.join(gpa, &.{ dir, name });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(4 << 20));
    defer gpa.free(bytes);
    if (bytes.len % (hc.codebooks * 2) != 0) return error.InvalidFixture;
    const n = bytes.len / (hc.codebooks * 2);
    const out = try gpa.alloc(hc.Frame, n);
    for (out, 0..) |*frame, t| {
        for (0..hc.codebooks) |c| {
            const off = (t * hc.codebooks + c) * 2;
            frame[c] = std.mem.readInt(u16, bytes[off..][0..2], .little);
        }
    }
    return out;
}

fn relRms(got: []const f32, want: []const f32) struct { rel: f64, max_abs: f64 } {
    if (got.len != want.len or got.len == 0) return .{ .rel = std.math.nan(f64), .max_abs = std.math.nan(f64) };
    var diff_sq: f64 = 0;
    var ref_sq: f64 = 0;
    var max_abs: f64 = 0;
    for (got, want) |g, w| {
        if (!std.math.isFinite(g) or !std.math.isFinite(w)) return .{ .rel = std.math.nan(f64), .max_abs = std.math.nan(f64) };
        const d = @as(f64, g) - @as(f64, w);
        diff_sq += d * d;
        ref_sq += @as(f64, w) * @as(f64, w);
        max_abs = @max(max_abs, @abs(d));
    }
    const rel = if (ref_sq == 0) @sqrt(diff_sq) else @sqrt(diff_sq / ref_sq);
    return .{ .rel = rel, .max_abs = max_abs };
}

fn runVocab(gpa: std.mem.Allocator, mapped: *const weights.Mapped) !bool {
    const directory = mapped.mapping.memory[0..@intCast(mapped.document.directory_bytes)];
    const data = mapped.mapping.memory[@intCast(mapped.document.data_offset)..];

    const binding = try omnivoice.bind(gpa, &mapped.document);
    try omnivoice.verifyTensorNames(gpa, &mapped.document, directory);
    try omnivoice.verifyCodebookOffsets(&binding, data);
    const embedded = try omnivoice.EmbeddedFiles.extract(&mapped.document, directory);
    try omnivoice.validateConfigJson(gpa, embedded.config_json);

    var tok = try hf_json.parse(gpa, embedded.tokenizer_json, .{});
    defer tok.deinit();
    try text_mod.verifyControlTokens(&tok);

    var ok = true;
    for (standalone_vectors) |v| {
        const ids = try tok.encode(gpa, v.text, .{});
        defer gpa.free(ids);
        ok = checkIds(v.name, ids, v.expected) and ok;
    }
    for (prompt_vectors) |p| {
        const ids = try text_mod.encodePrompt(gpa, &tok, p.request, .{});
        defer gpa.free(ids);
        ok = checkIds(p.name, ids, p.expected) and ok;
    }
    const uncond = try text_mod.encodeUnconditionalPrompt(gpa, &tok, .{});
    defer gpa.free(uncond);
    ok = checkIds("uncond_prompt", uncond, &uncond_expected) and ok;

    if (ok) std.debug.print(
        "OmniVoice binding ({d} generator + {d} codec tensors) and embedded Qwen3 tokenizer match all {d} standalone and {d} prompt vectors.\n",
        .{ binding.tensors, binding.codec_tensors, standalone_vectors.len, prompt_vectors.len + 1 },
    );
    return ok;
}

const EncRec = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    dir: []const u8,
    bound: f64,
    worst: check_stats.Worst = .{},
    ok: bool = true,

    fn record(ctx: *anyopaque, stage: hce.Stage, rows: []const f32) void {
        const self: *EncRec = @ptrCast(@alignCast(ctx));
        const fname: []const u8 = switch (stage) {
            .resampled_16k => "resampled_16k.f32",
            .hubert => "enc_hubert.f32",
            .semantic => "enc_semantic.f32",
            .acoustic => "enc_acoustic.f32",
            .fused => "enc_fused.f32",
        };
        const want = readF32File(self.gpa, self.io, self.dir, fname) catch {
            self.ok = false;
            return;
        };
        defer self.gpa.free(want);
        const s = relRms(rows, want);
        self.worst.observe(s.rel);
        const pass = !check_stats.exceeds(s.rel, self.bound);
        if (!pass) self.ok = false;
        std.debug.print("  enc {s:<14} rel_rms={e:.3} max_abs={e:.3} {s}\n", .{ @tagName(stage), s.rel, s.max_abs, if (pass) "ok" else "FAIL" });
    }
};

const DecRec = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    dir: []const u8,
    bound: f64,
    worst: check_stats.Worst = .{},
    ok: bool = true,

    fn record(ctx: *anyopaque, stage: hc.Stage, rows: []const f32) void {
        const self: *DecRec = @ptrCast(@alignCast(ctx));
        const fname: []const u8 = switch (stage) {
            .quantized => "dec_quantized.f32",
            .fc2 => "dec_fc2.f32",
            .conv1 => "dec_conv1.f32",
            .block_0 => "dec_block0_edges.f32",
            .block_1 => "dec_block1_edges.f32",
            .block_2 => "dec_block2_edges.f32",
            .block_3 => "dec_block3_edges.f32",
            .block_4 => "dec_block4_edges.f32",
            .samples => "dec_samples.f32",
        };
        const want = readF32File(self.gpa, self.io, self.dir, fname) catch {
            self.ok = false;
            return;
        };
        defer self.gpa.free(want);
        const s = switch (stage) {
            .block_0, .block_1, .block_2, .block_3, .block_4 => blk: {
                const ch: usize = switch (stage) {
                    .block_0 => 512,
                    .block_1 => 256,
                    .block_2 => 128,
                    .block_3 => 64,
                    .block_4 => 32,
                    else => unreachable,
                };
                const edge = self.gpa.alloc(f32, 32 * ch) catch {
                    self.ok = false;
                    return;
                };
                defer self.gpa.free(edge);
                @memcpy(edge[0 .. 16 * ch], rows[0 .. 16 * ch]);
                @memcpy(edge[16 * ch ..], rows[rows.len - 16 * ch ..]);
                break :blk relRms(edge, want);
            },
            else => relRms(rows, want),
        };
        self.worst.observe(s.rel);
        const pass = !check_stats.exceeds(s.rel, self.bound);
        if (!pass) self.ok = false;
        std.debug.print("  dec {s:<14} rel_rms={e:.3} max_abs={e:.3} {s}\n", .{ @tagName(stage), s.rel, s.max_abs, if (pass) "ok" else "FAIL" });
    }
};

fn runCodecEncode(gpa: std.mem.Allocator, io: std.Io, mapped: *const weights.Mapped, fixtures_dir: []const u8, bound: f64) !bool {
    const directory = mapped.mapping.memory[0..@intCast(mapped.document.directory_bytes)];
    var enc = try hce.Encoder.init(gpa, io, mapped.document, directory, mapped.view());
    defer enc.deinit();

    const wav_bytes = try std.Io.Dir.cwd().readFileAlloc(io, "tests/fixtures/qwen3-tts/0.6b-clone-reference/reference.wav", gpa, .limited(16 << 20));
    defer gpa.free(wav_bytes);
    var pcm = try audio_mod.decodeWavAt(gpa, wav_bytes, 48_000, hc.sample_rate);
    defer pcm.deinit(gpa);

    var rec: EncRec = .{ .gpa = gpa, .io = io, .dir = fixtures_dir, .bound = bound };
    const got_codes = try enc.encodeObserved(gpa, pcm.samples, .{ .context = &rec, .record = EncRec.record });
    defer gpa.free(got_codes);

    const want_codes = try readCodesFile(gpa, io, fixtures_dir, "codes.u16");
    defer gpa.free(want_codes);
    if (got_codes.len != want_codes.len) {
        std.debug.print("FAIL codec-encode: got {d} frames, want {d}\n", .{ got_codes.len, want_codes.len });
        return false;
    }
    var equal: usize = 0;
    const total = got_codes.len * hc.codebooks;
    for (got_codes, want_codes) |gf, wf| {
        for (gf, wf) |g, w| equal += @intFromBool(g == w);
    }
    const ok = rec.ok and rec.worst.within(bound) and equal == total;
    std.debug.print("Higgs codec CPU encode: {d}/{d} codes exact, worst stage rel_rms={e:.3} (bound {e:.2}).\n", .{ equal, total, rec.worst.shown(), bound });
    return ok;
}

fn runCodecDecode(gpa: std.mem.Allocator, io: std.Io, mapped: *const weights.Mapped, fixtures_dir: []const u8, bound: f64) !bool {
    const directory = mapped.mapping.memory[0..@intCast(mapped.document.directory_bytes)];
    var dec = try hc.Runtime.init(gpa, io, mapped.document, directory, mapped.view());
    defer dec.deinit();

    const want_codes = try readCodesFile(gpa, io, fixtures_dir, "codes.u16");
    defer gpa.free(want_codes);

    var rec: DecRec = .{ .gpa = gpa, .io = io, .dir = fixtures_dir, .bound = bound };
    const samples = try dec.decodeObserved(gpa, want_codes, .{ .context = &rec, .record = DecRec.record });
    defer gpa.free(samples);

    const ok = rec.ok and rec.worst.within(bound);
    std.debug.print("Higgs codec CPU decode: {d} samples, worst stage rel_rms={e:.3} (bound {e:.2}).\n", .{ samples.len, rec.worst.shown(), bound });
    return ok;
}

const BackboneRec = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    dir: []const u8,
    prefix: []const u8,
    rel_bound: f64,
    abs_bound: f64,
    worst_rel: check_stats.Worst = .{},
    worst_abs: check_stats.Worst = .{},
    ok: bool = true,

    fn record(ctx: *anyopaque, stage: runtime.Stage, layer_index: ?usize, rows: []const f32) void {
        const self: *BackboneRec = @ptrCast(@alignCast(ctx));
        var name_buf: [64]u8 = undefined;
        const fname: []const u8 = switch (stage) {
            .embed => std.fmt.bufPrint(&name_buf, "{s}_embed.f32", .{self.prefix}) catch unreachable,
            .norm => std.fmt.bufPrint(&name_buf, "{s}_norm.f32", .{self.prefix}) catch unreachable,
            .logits => std.fmt.bufPrint(&name_buf, "{s}_logits.f32", .{self.prefix}) catch unreachable,
            .layer => blk: {
                const li = layer_index orelse return;
                if (li != 0 and li != 13 and li != 27) return;
                break :blk std.fmt.bufPrint(&name_buf, "{s}_layer{d}.f32", .{ self.prefix, li }) catch unreachable;
            },
        };
        const want = readF32File(self.gpa, self.io, self.dir, fname) catch {
            self.ok = false;
            return;
        };
        defer self.gpa.free(want);
        const s = relRms(rows, want);
        self.worst_rel.observe(s.rel);
        const check_abs = stage != .layer;
        if (check_abs) self.worst_abs.observe(s.max_abs);
        var argmax_ok = true;
        if (stage == .logits) {
            const slices = rows.len / runtime.audio_vocab;
            for (0..slices) |k| {
                const g_sl = rows[k * runtime.audio_vocab ..][0..runtime.audio_vocab];
                const w_sl = want[k * runtime.audio_vocab ..][0..runtime.audio_vocab];
                if (std.mem.indexOfMax(f32, g_sl) != std.mem.indexOfMax(f32, w_sl)) argmax_ok = false;
            }
        }
        const pass = !check_stats.exceeds(s.rel, self.rel_bound) and (!check_abs or !check_stats.exceeds(s.max_abs, self.abs_bound)) and argmax_ok;
        if (!pass) self.ok = false;
        std.debug.print("  {s:<20} rel_rms={e:.3} max_abs={e:.3}{s} {s}\n", .{ fname, s.rel, s.max_abs, if (stage == .logits) (if (argmax_ok) " argmax=ok" else " argmax=FAIL") else "", if (pass) "ok" else "FAIL" });
    }
};

fn runTrace(gpa: std.mem.Allocator, io: std.Io, mapped: *const weights.Mapped, fixtures_dir: []const u8, rel_bound: f64, abs_bound: f64) !bool {
    var m = try runtime.Model.init(gpa, io, &mapped.document, mapped.view());
    defer m.deinit();

    // Exact prompt_ids + 16 ref frames from codec fixture + 12 masked target frames.
    const prompt_ids = [_]u32{ 151669, 151670, 277, 89, 151671, 151672, 4064, 151673, 151674, 69682, 16157, 80970, 124376, 27846, 126300, 13, 85153, 125089, 31073, 124080, 16157, 35038, 91335, 128332, 151675 };
    const all_codes = try readCodesFile(gpa, io, "tests/fixtures/omnivoice/codec", "codes.u16");
    defer gpa.free(all_codes);
    const ref_frames: usize = 16;
    const target_frames: usize = 12;
    const masked: hc.Frame = @splat(runtime.audio_mask_id);

    const cond_rows = try gpa.alloc(runtime.Row, prompt_ids.len + ref_frames + target_frames);
    defer gpa.free(cond_rows);
    for (prompt_ids, 0..) |id, i| cond_rows[i] = .{ .text = id };
    for (0..ref_frames) |i| cond_rows[prompt_ids.len + i] = .{ .audio = all_codes[i] };
    for (0..target_frames) |i| cond_rows[prompt_ids.len + ref_frames + i] = .{ .audio = masked };

    const uncond_rows = try gpa.alloc(runtime.Row, target_frames);
    defer gpa.free(uncond_rows);
    for (uncond_rows) |*r| r.* = .{ .audio = masked };

    const logits = try gpa.alloc(f32, target_frames * runtime.audio_table_rows);
    defer gpa.free(logits);

    var cond_rec: BackboneRec = .{ .gpa = gpa, .io = io, .dir = fixtures_dir, .prefix = "cond", .rel_bound = rel_bound, .abs_bound = abs_bound };
    try m.forward(gpa, cond_rows, target_frames, logits, .{ .context = &cond_rec, .record = BackboneRec.record });

    var uncond_rec: BackboneRec = .{ .gpa = gpa, .io = io, .dir = fixtures_dir, .prefix = "uncond", .rel_bound = rel_bound, .abs_bound = abs_bound };
    try m.forward(gpa, uncond_rows, target_frames, logits, .{ .context = &uncond_rec, .record = BackboneRec.record });

    const worst_rel = @max(cond_rec.worst_rel.shown(), uncond_rec.worst_rel.shown());
    const worst_abs = @max(cond_rec.worst_abs.shown(), uncond_rec.worst_abs.shown());
    const ok = cond_rec.ok and uncond_rec.ok and cond_rec.worst_rel.within(rel_bound) and uncond_rec.worst_rel.within(rel_bound) and cond_rec.worst_abs.within(abs_bound) and uncond_rec.worst_abs.within(abs_bound);
    std.debug.print("OmniVoice CPU backbone trace: worst rel_rms={e:.3} (bound {e:.2}), worst max_abs={e:.3} (bound {e:.2}).\n", .{ worst_rel, rel_bound, worst_abs, abs_bound });
    return ok;
}

fn writeWav16(io: std.Io, path: []const u8, samples: []const f32, rate: u32) !void {
    const file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    const data_bytes: u32 = @intCast(samples.len * 2);
    var header: [44]u8 = undefined;
    @memcpy(header[0..4], "RIFF");
    std.mem.writeInt(u32, header[4..8], 36 + data_bytes, .little);
    @memcpy(header[8..12], "WAVE");
    @memcpy(header[12..16], "fmt ");
    std.mem.writeInt(u32, header[16..20], 16, .little);
    std.mem.writeInt(u16, header[20..22], 1, .little); // PCM
    std.mem.writeInt(u16, header[22..24], 1, .little); // mono
    std.mem.writeInt(u32, header[24..28], rate, .little);
    std.mem.writeInt(u32, header[28..32], rate * 2, .little);
    std.mem.writeInt(u16, header[32..34], 2, .little);
    std.mem.writeInt(u16, header[34..36], 16, .little);
    @memcpy(header[36..40], "data");
    std.mem.writeInt(u32, header[40..44], data_bytes, .little);

    var buf: [8192]u8 = undefined;
    var w = file.writer(io, &buf);
    try w.interface.writeAll(&header);
    for (samples) |s| {
        const clamped = std.math.clamp(s, -1.0, 1.0);
        const q: i16 = @intFromFloat(@round(clamped * 32767.0));
        try w.interface.writeInt(i16, q, .little);
    }
    try w.interface.flush();
}

fn runSpeak(
    gpa: std.mem.Allocator,
    io: std.Io,
    model_path: []const u8,
    text: []const u8,
    out_path: []const u8,
    backend: speech.Backend,
    lang: []const u8,
    instruct: ?[]const u8,
    duration_sec: ?f32,
    speed: f32,
    steps: u32,
    seed: u64,
) !bool {
    const start = std.Io.Clock.awake.now(io);
    const voice = try speech.Voice.openWith(gpa, io, model_path, .{ .backend = backend });
    defer voice.close();
    const open_ms = @as(f64, @floatFromInt(start.durationTo(std.Io.Clock.awake.now(io)).toMicroseconds())) / 1000.0;

    const synth_start = std.Io.Clock.awake.now(io);
    var synth = try voice.synthesize(gpa, .{
        .text = text,
        .language = lang,
        .instruct = instruct,
        .duration_sec = duration_sec,
        .speed = speed,
        .sampling = .{ .num_step = steps, .seed = seed },
    });
    defer synth.deinit();

    var all: std.ArrayList(f32) = .empty;
    defer all.deinit(gpa);
    var chunk: [4096]f32 = undefined;
    while (true) {
        const n = try synth.read(&chunk);
        if (n == 0) break;
        try all.appendSlice(gpa, chunk[0..n]);
    }
    const synth_ms = @as(f64, @floatFromInt(synth_start.durationTo(std.Io.Clock.awake.now(io)).toMicroseconds())) / 1000.0;

    if (out_path.len > 0) try writeWav16(io, out_path, all.items, speech.sample_rate);
    var sum_sq: f64 = 0;
    var finite = true;
    for (all.items) |s| {
        if (!std.math.isFinite(s)) finite = false;
        sum_sq += @as(f64, s) * @as(f64, s);
    }
    const rms = if (all.items.len > 0) @sqrt(sum_sq / @as(f64, @floatFromInt(all.items.len))) else 0.0;
    const audio_sec = @as(f64, @floatFromInt(all.items.len)) / @as(f64, @floatFromInt(speech.sample_rate));
    std.debug.print(
        "OmniVoice speak ({s}): {d} frames, {d} samples ({d:.2} s audio), open {d:.0} ms, synth {d:.0} ms (RTF {d:.2}x), rms={d:.4}, finite={}.\n",
        .{ @tagName(backend), synth.frames(), all.items.len, audio_sec, open_ms, synth_ms, if (audio_sec > 0) (synth_ms / 1000.0) / audio_sec else 0.0, rms, finite },
    );
    return finite and all.items.len > 0 and rms > 1e-4;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) {
        std.debug.print(usage, .{});
        std.process.exit(2);
    }
    const model_path = args[1];
    const mode = args[2];

    var fixtures_dir: ?[]const u8 = null;
    var rel_bound: f64 = 1e-4;
    var abs_bound: f64 = 2e-3;
    var text: []const u8 = "إزيك يا باشا؟ عامل إيه النهارده؟";
    var out_path: []const u8 = "";
    var backend: speech.Backend = .cpu;
    var lang: []const u8 = text_mod.default_language;
    var instruct: ?[]const u8 = null;
    var duration_sec: ?f32 = null;
    var speed: f32 = 1.0;
    var steps: u32 = 32;
    var seed: u64 = 0;

    var i: usize = 3;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--fixtures") and i + 1 < args.len) {
            i += 1;
            fixtures_dir = args[i];
        } else if (std.mem.eql(u8, args[i], "--max-relative-rms") and i + 1 < args.len) {
            i += 1;
            rel_bound = try std.fmt.parseFloat(f64, args[i]);
        } else if (std.mem.eql(u8, args[i], "--max-abs") and i + 1 < args.len) {
            i += 1;
            abs_bound = try std.fmt.parseFloat(f64, args[i]);
        } else if (std.mem.eql(u8, args[i], "--text") and i + 1 < args.len) {
            i += 1;
            text = args[i];
        } else if (std.mem.eql(u8, args[i], "--out") and i + 1 < args.len) {
            i += 1;
            out_path = args[i];
        } else if (std.mem.eql(u8, args[i], "--backend") and i + 1 < args.len) {
            i += 1;
            backend = if (std.mem.eql(u8, args[i], "metal")) .metal else .cpu;
        } else if (std.mem.eql(u8, args[i], "--lang") and i + 1 < args.len) {
            i += 1;
            lang = args[i];
        } else if (std.mem.eql(u8, args[i], "--instruct") and i + 1 < args.len) {
            i += 1;
            instruct = args[i];
        } else if (std.mem.eql(u8, args[i], "--duration") and i + 1 < args.len) {
            i += 1;
            duration_sec = try std.fmt.parseFloat(f32, args[i]);
        } else if (std.mem.eql(u8, args[i], "--speed") and i + 1 < args.len) {
            i += 1;
            speed = try std.fmt.parseFloat(f32, args[i]);
        } else if (std.mem.eql(u8, args[i], "--steps") and i + 1 < args.len) {
            i += 1;
            steps = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--seed") and i + 1 < args.len) {
            i += 1;
            seed = try std.fmt.parseInt(u64, args[i], 10);
        } else {
            std.debug.print(usage, .{});
            std.process.exit(2);
        }
    }

    if (std.mem.eql(u8, mode, "speak")) {
        const ok = try runSpeak(gpa, io, model_path, text, out_path, backend, lang, instruct, duration_sec, speed, steps, seed);
        if (!ok) std.process.exit(1);
        return;
    }

    var mapped = try weights.Mapped.open(gpa, io, model_path);
    defer mapped.deinit(io);

    const ok = if (std.mem.eql(u8, mode, "vocab"))
        try runVocab(gpa, &mapped)
    else if (std.mem.eql(u8, mode, "codec-encode"))
        try runCodecEncode(gpa, io, &mapped, fixtures_dir orelse "tests/fixtures/omnivoice/codec", rel_bound)
    else if (std.mem.eql(u8, mode, "codec-decode"))
        try runCodecDecode(gpa, io, &mapped, fixtures_dir orelse "tests/fixtures/omnivoice/codec", rel_bound)
    else if (std.mem.eql(u8, mode, "codec")) blk: {
        const d = fixtures_dir orelse "tests/fixtures/omnivoice/codec";
        const enc_ok = try runCodecEncode(gpa, io, &mapped, d, rel_bound);
        const dec_ok = try runCodecDecode(gpa, io, &mapped, d, rel_bound);
        break :blk enc_ok and dec_ok;
    } else if (std.mem.eql(u8, mode, "trace"))
        try runTrace(gpa, io, &mapped, fixtures_dir orelse "tests/fixtures/omnivoice/backbone", rel_bound, abs_bound)
    else {
        std.debug.print(usage, .{});
        std.process.exit(2);
    };

    if (!ok) std.process.exit(1);
}
