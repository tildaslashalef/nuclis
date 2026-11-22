//! Explicit full-model check of OmniVoice / VoiceTut-TTS (`audio.cpp` GGUF).
//!
//!   omnivoice-check MODEL.gguf vocab
//!
//! `vocab`: opens the GGUF, validates the structural binding, `audiocpp.tensor_names`,
//! embedded `config.json`, and `weights/codebook_layer_offsets`, parses the embedded
//! Qwen3 BPE `tokenizer.json`, checks all seven control tokens, and compares
//! standalone strings and full prompts against pinned reference token sequences.
const std = @import("std");
const inference = @import("inference");
const omnivoice = inference.models.omnivoice;
const text_mod = inference.models.omnivoice_text;
const hf_json = inference.hf_tokenizer;
const weights = inference.weights;

const usage = "usage: omnivoice-check MODEL.gguf vocab\n";

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
    if (!std.mem.eql(u8, mode, "vocab")) {
        std.debug.print(usage, .{});
        std.process.exit(2);
    }

    var mapped = try weights.Mapped.open(gpa, io, model_path);
    defer mapped.deinit(io);

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

    if (!ok) std.process.exit(1);
    std.debug.print(
        "OmniVoice binding ({d} generator + {d} codec tensors) and embedded Qwen3 tokenizer match all {d} standalone and {d} prompt vectors.\n",
        .{ binding.tensors, binding.codec_tensors, standalone_vectors.len, prompt_vectors.len + 1 },
    );
}
