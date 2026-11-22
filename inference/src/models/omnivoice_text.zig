//! OmniVoice / VoiceTut-TTS prompt builder, instruct-tag validator, duration
//! estimator, and long-text sentence chunker.
//!
//! Pure logic over the embedded Qwen3 BPE tokenizer (`hf_json.Tokenizer`).
//! Keeps user text verbatim (no implicit text normalizer). Supports three
//! voice modes: auto-voice, voice design (strict fixed attribute tags, mutually
//! exclusive with a reference clip), and zero-shot voice cloning.
const std = @import("std");
const hf_json = @import("../tokenizer/hf_json.zig");
const encode = @import("../tokenizer/encode.zig");
const Allocator = std.mem.Allocator;

/// OmniVoice special control token ids in `voicetut-tts-q8_0.gguf`.
pub const denoise_token_id: u32 = 151669;
pub const lang_start_token_id: u32 = 151670;
pub const lang_end_token_id: u32 = 151671;
pub const instruct_start_token_id: u32 = 151672;
pub const instruct_end_token_id: u32 = 151673;
pub const text_start_token_id: u32 = 151674;
pub const text_end_token_id: u32 = 151675;

pub const default_language = "arz";

pub const sample_rate_hz: u32 = 24_000;
pub const samples_per_frame: u32 = 960;
pub const frame_rate_hz: u32 = 25;
pub const min_frames: usize = 10; // 0.4 s
pub const max_frames: usize = 1500; // 60 s host ceiling
pub const default_chunk_bytes: usize = 240;

pub const reserved_tokens = [_][]const u8{
    "<|denoise|>",
    "<|lang_start|>",
    "<|lang_end|>",
    "<|instruct_start|>",
    "<|instruct_end|>",
    "<|text_start|>",
    "<|text_end|>",
};

pub const InstructCategory = enum {
    gender,
    age,
    pitch,
    style,
    accent,
};

const TagEntry = struct {
    tag: []const u8,
    category: InstructCategory,
};

/// OmniVoice's fixed voice-design attribute tags (English and Chinese).
pub const instruct_tags = [_]TagEntry{
    // Gender
    .{ .tag = "male", .category = .gender },
    .{ .tag = "female", .category = .gender },
    .{ .tag = "男", .category = .gender },
    .{ .tag = "女", .category = .gender },
    // Age
    .{ .tag = "child", .category = .age },
    .{ .tag = "teenager", .category = .age },
    .{ .tag = "young adult", .category = .age },
    .{ .tag = "middle-aged", .category = .age },
    .{ .tag = "elderly", .category = .age },
    .{ .tag = "儿童", .category = .age },
    .{ .tag = "少年", .category = .age },
    .{ .tag = "青年", .category = .age },
    .{ .tag = "中年", .category = .age },
    .{ .tag = "老年", .category = .age },
    // Pitch
    .{ .tag = "very low pitch", .category = .pitch },
    .{ .tag = "low pitch", .category = .pitch },
    .{ .tag = "moderate pitch", .category = .pitch },
    .{ .tag = "high pitch", .category = .pitch },
    .{ .tag = "very high pitch", .category = .pitch },
    .{ .tag = "极低音调", .category = .pitch },
    .{ .tag = "低音调", .category = .pitch },
    .{ .tag = "中音调", .category = .pitch },
    .{ .tag = "高音调", .category = .pitch },
    .{ .tag = "极高音调", .category = .pitch },
    // Style
    .{ .tag = "whisper", .category = .style },
    .{ .tag = "耳语", .category = .style },
    // English accents
    .{ .tag = "american accent", .category = .accent },
    .{ .tag = "british accent", .category = .accent },
    .{ .tag = "australian accent", .category = .accent },
    .{ .tag = "chinese accent", .category = .accent },
    .{ .tag = "canadian accent", .category = .accent },
    .{ .tag = "indian accent", .category = .accent },
    .{ .tag = "korean accent", .category = .accent },
    .{ .tag = "portuguese accent", .category = .accent },
    .{ .tag = "russian accent", .category = .accent },
    .{ .tag = "japanese accent", .category = .accent },
    // Chinese dialects
    .{ .tag = "河南话", .category = .accent },
    .{ .tag = "陕西话", .category = .accent },
    .{ .tag = "四川话", .category = .accent },
    .{ .tag = "贵州话", .category = .accent },
    .{ .tag = "云南话", .category = .accent },
    .{ .tag = "桂林话", .category = .accent },
    .{ .tag = "济南话", .category = .accent },
    .{ .tag = "石家庄话", .category = .accent },
    .{ .tag = "甘肃话", .category = .accent },
    .{ .tag = "宁夏话", .category = .accent },
    .{ .tag = "青岛话", .category = .accent },
    .{ .tag = "东北话", .category = .accent },
};

pub const InputError = error{
    EmptyText,
    ReservedText,
    InvalidLanguage,
    UnknownInstructTag,
    DuplicateInstructCategory,
    ConflictingVoiceMode,
    InvalidReference,
    InvalidDuration,
    InvalidSpeed,
};

pub const Error = Allocator.Error || hf_json.EncodeError || InputError;

pub const Request = struct {
    text: []const u8,
    /// Language id/name (`"arz"` by default for VoiceTut-TTS, or `"None"`).
    language: []const u8 = default_language,
    /// Voice-design attribute tags, comma-separated; mutually exclusive with a reference clip.
    instruct: ?[]const u8 = null,
    /// Reference transcript when cloning from a saved voice.
    ref_text: ?[]const u8 = null,
    /// Reference audio frame count (25 fps) when cloning.
    ref_frames: usize = 0,
    /// Prepend `<|denoise|>` when cloning from a noisy reference clip.
    denoise: bool = false,
    /// Explicit target duration in seconds (overrides `speed` and auto-estimation).
    duration_sec: ?f32 = null,
    /// Speaking rate multiplier (> 0, default 1.0).
    speed: f32 = 1.0,
};

/// Returns the category of an OmniVoice attribute tag, or null if unknown.
pub fn lookupInstructTag(tag: []const u8) ?InstructCategory {
    for (instruct_tags) |entry| {
        if (std.mem.eql(u8, entry.tag, tag)) return entry.category;
    }
    return null;
}

/// Validates a comma-separated list of OmniVoice voice-design tags.
/// Rejects empty lists, unknown tags, and duplicate tags in the same category.
pub fn validateInstruct(instruct: []const u8) InputError!void {
    const trimmed = std.mem.trim(u8, instruct, " \t\r\n");
    if (trimmed.len == 0) return error.UnknownInstructTag;
    var seen: std.EnumSet(InstructCategory) = .empty;
    var it = std.mem.splitScalar(u8, trimmed, ',');
    while (it.next()) |raw_part| {
        const part = std.mem.trim(u8, raw_part, " \t\r\n");
        if (part.len == 0) return error.UnknownInstructTag;
        const cat = lookupInstructTag(part) orelse return error.UnknownInstructTag;
        if (seen.contains(cat)) return error.DuplicateInstructCategory;
        seen.insert(cat);
    }
}

/// Validates a language tag: non-empty, <= 64 bytes, UTF-8, no control tokens or newlines.
pub fn validateLanguage(language: []const u8) InputError!void {
    const trimmed = std.mem.trim(u8, language, " \t\r\n");
    if (trimmed.len == 0 or trimmed.len > 64 or !std.unicode.utf8ValidateSlice(trimmed))
        return error.InvalidLanguage;
    if (containsReserved(trimmed)) return error.ReservedText;
    for (trimmed) |b| if (b < 0x20 or b == 0x7f) return error.InvalidLanguage;
}

/// Validates a speech request across all three modes (auto-voice, voice design, zero-shot clone).
pub fn validateRequest(req: Request) InputError!void {
    if (std.mem.trim(u8, req.text, " \t\r\n").len == 0) return error.EmptyText;
    if (containsReserved(req.text)) return error.ReservedText;
    try validateLanguage(req.language);

    const has_ref = req.ref_frames > 0 or req.ref_text != null;
    if (req.instruct) |inst| {
        if (has_ref) return error.ConflictingVoiceMode;
        try validateInstruct(inst);
    }
    if (has_ref) {
        if (req.ref_frames == 0) return error.InvalidReference;
        const rt = req.ref_text orelse return error.InvalidReference;
        if (std.mem.trim(u8, rt, " \t\r\n").len == 0) return error.InvalidReference;
        if (containsReserved(rt)) return error.ReservedText;
    } else if (req.denoise) {
        return error.ConflictingVoiceMode;
    }
    if (req.duration_sec) |d| {
        if (!std.math.isFinite(d) or d <= 0.0) return error.InvalidDuration;
    }
    if (!std.math.isFinite(req.speed) or req.speed <= 0.0) return error.InvalidSpeed;
}

/// Formats the OmniVoice conditional prompt string (caller owns returned slice).
/// Text is kept verbatim (no implicit normalization).
pub fn formatPrompt(alloc: Allocator, req: Request) (Allocator.Error || InputError)![]u8 {
    try validateRequest(req);
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);

    if (req.denoise) try out.appendSlice(alloc, "<|denoise|>");
    try out.appendSlice(alloc, "<|lang_start|>");
    try out.appendSlice(alloc, std.mem.trim(u8, req.language, " \t\r\n"));
    try out.appendSlice(alloc, "<|lang_end|><|instruct_start|>");
    if (req.instruct) |inst| {
        var it = std.mem.splitScalar(u8, std.mem.trim(u8, inst, " \t\r\n"), ',');
        var first = true;
        while (it.next()) |raw_part| {
            if (!first) try out.appendSlice(alloc, ", ");
            try out.appendSlice(alloc, std.mem.trim(u8, raw_part, " \t\r\n"));
            first = false;
        }
    } else {
        try out.appendSlice(alloc, "None");
    }
    try out.appendSlice(alloc, "<|instruct_end|><|text_start|>");
    if (req.ref_text) |rt| {
        try out.appendSlice(alloc, rt);
        try out.append(alloc, ' ');
    }
    try out.appendSlice(alloc, req.text);
    try out.appendSlice(alloc, "<|text_end|>");
    return out.toOwnedSlice(alloc);
}

/// Formats the unconditional CFG prompt string.
pub const unconditional_prompt = "<|lang_start|>None<|lang_end|><|instruct_start|>None<|instruct_end|><|text_start|><|text_end|>";

/// Encodes the conditional OmniVoice prompt into token IDs (caller owns returned slice).
pub fn encodePrompt(alloc: Allocator, tokenizer: *const hf_json.Tokenizer, req: Request, limits: encode.Limits) Error![]u32 {
    const prompt_text = try formatPrompt(alloc, req);
    defer alloc.free(prompt_text);
    return tokenizer.encode(alloc, prompt_text, limits);
}

/// Encodes the unconditional CFG prompt into token IDs (caller owns returned slice).
pub fn encodeUnconditionalPrompt(alloc: Allocator, tokenizer: *const hf_json.Tokenizer, limits: encode.Limits) hf_json.EncodeError![]u32 {
    return tokenizer.encode(alloc, unconditional_prompt, limits);
}

/// Verifies that the tokenizer contains all seven OmniVoice control tokens at
/// their expected IDs.
pub fn verifyControlTokens(tokenizer: *const hf_json.Tokenizer) error{UnsupportedConfiguration}!void {
    const expected = [_]struct { text: []const u8, id: u32 }{
        .{ .text = "<|denoise|>", .id = denoise_token_id },
        .{ .text = "<|lang_start|>", .id = lang_start_token_id },
        .{ .text = "<|lang_end|>", .id = lang_end_token_id },
        .{ .text = "<|instruct_start|>", .id = instruct_start_token_id },
        .{ .text = "<|instruct_end|>", .id = instruct_end_token_id },
        .{ .text = "<|text_start|>", .id = text_start_token_id },
        .{ .text = "<|text_end|>", .id = text_end_token_id },
    };
    for (expected) |e| {
        if (tokenizer.vocab.tokenId(e.text) != e.id) return error.UnsupportedConfiguration;
    }
}

/// Estimates target audio frame count (at 25 fps, bounded to `[min_frames, max_frames]`).
/// Explicit `duration_sec` overrides `speed` and automatic estimation.
pub fn estimateFrames(req: Request) InputError!usize {
    try validateRequest(req);
    if (req.duration_sec) |d| {
        const raw = @round(@as(f64, d) * @as(f64, frame_rate_hz));
        return clampFrames(raw);
    }
    const target_units = speechUnits(req.text);
    const base_frames: f64 = if (req.ref_frames > 0 and req.ref_text != null) blk: {
        const ref_units = @max(1.0, speechUnits(req.ref_text.?));
        const rate = @as(f64, @floatFromInt(req.ref_frames)) / ref_units;
        break :blk target_units * rate;
    } else blk: {
        // ~2.2 frames per weighted speech unit (~11 units/s at 25 fps) plus 12-frame onset/tail cushion.
        break :blk target_units * 2.2 + 12.0;
    };
    return clampFrames(@round(base_frames / @as(f64, req.speed)));
}

fn clampFrames(raw: f64) usize {
    if (!std.math.isFinite(raw) or raw <= @as(f64, @floatFromInt(min_frames))) return min_frames;
    if (raw >= @as(f64, @floatFromInt(max_frames))) return max_frames;
    return @intFromFloat(raw);
}

/// Counts script-aware speaking units for duration estimation (Arabic/Latin
/// characters vs CJK ideographs vs pauses).
fn speechUnits(text: []const u8) f64 {
    var units: f64 = 0.0;
    var i: usize = 0;
    while (i < text.len) {
        const len = std.unicode.utf8ByteSequenceLength(text[i]) catch {
            i += 1;
            continue;
        };
        if (i + len > text.len) break;
        const cp = std.unicode.utf8Decode(text[i..][0..len]) catch {
            i += 1;
            continue;
        };
        i += len;
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or (cp >= 0x3400 and cp <= 0x4DBF) or (cp >= 0x3040 and cp <= 0x30FF)) {
            units += 2.0; // CJK syllable
        } else if (cp >= 0x064B and cp <= 0x065F) {
            // Arabic combining diacritics add no extra duration.
        } else if (cp == '.' or cp == '!' or cp == '?' or cp == 0x061F or cp == 0x061B or cp == '\n') {
            units += 2.5; // sentence pause
        } else if (cp == ',' or cp == 0x060C or cp == ';' or cp == ':') {
            units += 1.2; // clause pause
        } else if (cp == ' ' or cp == '\t' or cp == '\r') {
            units += 0.25;
        } else {
            units += 1.0;
        }
    }
    return @max(1.0, units);
}

/// Splits long text into verbatim chunks of roughly `max_bytes` UTF-8 bytes,
/// preferring sentence ends (`.!?;…`, newlines, Arabic `؟؛۔`), then clause marks
/// (`,،:：`), then spaces, and never splitting inside `<|...|>` or `[...]` tags.
/// Returned slices borrow `text`; caller frees the outer slice with `alloc`.
pub fn chunkText(alloc: Allocator, text: []const u8, max_bytes: usize) Allocator.Error![]const []const u8 {
    const limit = @max(16, max_bytes);
    var chunks: std.ArrayList([]const u8) = .empty;
    errdefer chunks.deinit(alloc);

    var pos: usize = 0;
    while (pos < text.len) {
        while (pos < text.len and isAsciiSpace(text[pos])) pos += 1;
        if (pos >= text.len) break;

        const remaining = text[pos..];
        if (remaining.len <= limit) {
            const trimmed = std.mem.trimEnd(u8, remaining, " \t\r\n");
            if (trimmed.len > 0) try chunks.append(alloc, trimmed);
            break;
        }

        const split_len = findSplit(remaining, limit);
        const piece = std.mem.trimEnd(u8, remaining[0..split_len], " \t\r\n");
        if (piece.len > 0) try chunks.append(alloc, piece);
        pos += split_len;
    }
    return chunks.toOwnedSlice(alloc);
}

fn findSplit(text: []const u8, limit: usize) usize {
    var best_sentence: usize = 0;
    var best_clause: usize = 0;
    var best_space: usize = 0;
    var last_outside_tag: usize = 0;

    var i: usize = 0;
    var in_bracket: bool = false;
    var in_angle: bool = false;
    const cap = @min(text.len, limit);

    while (i < cap) {
        if (!in_angle and std.mem.startsWith(u8, text[i..], "<|")) {
            in_angle = true;
            i += 2;
            continue;
        }
        if (in_angle) {
            if (std.mem.startsWith(u8, text[i..], "|>")) {
                in_angle = false;
                i += 2;
                if (i <= cap) last_outside_tag = i;
                continue;
            }
            i += 1;
            continue;
        }
        const len = std.unicode.utf8ByteSequenceLength(text[i]) catch 1;
        const step = if (i + len <= text.len) len else 1;
        const end = i + step;
        const cp: u21 = if (step == len) (std.unicode.utf8Decode(text[i..end]) catch text[i]) else text[i];

        if (cp == '[') {
            in_bracket = true;
        } else if (cp == ']') {
            in_bracket = false;
            if (end <= cap) last_outside_tag = end;
        } else if (!in_bracket and end <= cap) {
            last_outside_tag = end;
            if (isSentenceEnd(cp, text, i, end)) {
                best_sentence = end;
            } else if (isClauseMark(cp)) {
                best_clause = end;
            } else if (cp == ' ' or cp == '\t') {
                best_space = end;
            }
        }
        i = end;
    }

    if (best_sentence > 0) return best_sentence;
    if (best_clause > 0) return best_clause;
    if (best_space > 0) return best_space;
    if (last_outside_tag > 0) return last_outside_tag;

    // Fallback when a single tag exceeds `limit`: advance past its closing delimiter.
    if (in_angle) {
        if (std.mem.indexOfPos(u8, text, i, "|>")) |close| return close + 2;
    }
    if (in_bracket) {
        if (std.mem.indexOfScalarPos(u8, text, i, ']')) |close| return close + 1;
    }
    return cap;
}

fn isSentenceEnd(cp: u21, text: []const u8, i: usize, end: usize) bool {
    if (cp == '.') {
        // Do not split inside decimal numbers like 3.14.
        const prev_digit = i > 0 and std.ascii.isDigit(text[i - 1]);
        const next_digit = end < text.len and std.ascii.isDigit(text[end]);
        if (prev_digit and next_digit) return false;
        return true;
    }
    return cp == '!' or cp == '?' or cp == ';' or cp == '\n' or
        cp == 0x2026 or // …
        cp == 0x061F or // Arabic ؟
        cp == 0x061B or // Arabic ؛
        cp == 0x06D4; // Arabic full stop ۔
}

fn isClauseMark(cp: u21) bool {
    return cp == ',' or cp == ':' or cp == 0x060C or cp == 0xFF0C or cp == 0xFF1A;
}

fn isAsciiSpace(b: u8) bool {
    return b == ' ' or b == '\t' or b == '\r' or b == '\n';
}

fn containsReserved(text: []const u8) bool {
    for (reserved_tokens) |tok| {
        if (std.mem.indexOf(u8, text, tok) != null) return true;
    }
    return false;
}

// ---- tests -----------------------------------------------------------------

test "instruct tag validation accepts valid combinations and rejects unknown or duplicate categories" {
    try validateInstruct("female, young adult, moderate pitch, british accent");
    try validateInstruct("male, whisper");
    try validateInstruct("女, 青年, 中音调, 四川话");
    try std.testing.expectError(error.UnknownInstructTag, validateInstruct(""));
    try std.testing.expectError(error.UnknownInstructTag, validateInstruct("female, "));
    try std.testing.expectError(error.UnknownInstructTag, validateInstruct("robotic voice"));
    try std.testing.expectError(error.DuplicateInstructCategory, validateInstruct("male, female"));
    try std.testing.expectError(error.DuplicateInstructCategory, validateInstruct("low pitch, high pitch"));
    try std.testing.expectError(error.DuplicateInstructCategory, validateInstruct("american accent, british accent"));
}

test "prompt formatting across auto-voice, voice design, and zero-shot cloning" {
    const gpa = std.testing.allocator;

    // Auto-voice (default Egyptian Arabic).
    const auto_p = try formatPrompt(gpa, .{ .text = "عندي meeting بكرة الصبح." });
    defer gpa.free(auto_p);
    try std.testing.expectEqualStrings(
        "<|lang_start|>arz<|lang_end|><|instruct_start|>None<|instruct_end|><|text_start|>عندي meeting بكرة الصبح.<|text_end|>",
        auto_p,
    );

    // Voice design.
    const design_p = try formatPrompt(gpa, .{
        .text = "Hello world!",
        .language = "en",
        .instruct = "female,low pitch, british accent",
    });
    defer gpa.free(design_p);
    try std.testing.expectEqualStrings(
        "<|lang_start|>en<|lang_end|><|instruct_start|>female, low pitch, british accent<|instruct_end|><|text_start|>Hello world!<|text_end|>",
        design_p,
    );

    // Zero-shot voice cloning with denoise.
    const clone_p = try formatPrompt(gpa, .{
        .text = "إزيك النهارده؟",
        .ref_text = "أهلاً بيك.",
        .ref_frames = 50,
        .denoise = true,
    });
    defer gpa.free(clone_p);
    try std.testing.expectEqualStrings(
        "<|denoise|><|lang_start|>arz<|lang_end|><|instruct_start|>None<|instruct_end|><|text_start|>أهلاً بيك. إزيك النهارده؟<|text_end|>",
        clone_p,
    );

    // Rejections.
    try std.testing.expectError(error.EmptyText, formatPrompt(gpa, .{ .text = "   " }));
    try std.testing.expectError(error.ReservedText, formatPrompt(gpa, .{ .text = "hi <|text_end|>" }));
    try std.testing.expectError(error.ConflictingVoiceMode, formatPrompt(gpa, .{ .text = "hi", .instruct = "male", .ref_text = "ref", .ref_frames = 25 }));
    try std.testing.expectError(error.ConflictingVoiceMode, formatPrompt(gpa, .{ .text = "hi", .denoise = true }));
    try std.testing.expectError(error.InvalidReference, formatPrompt(gpa, .{ .text = "hi", .ref_frames = 25 }));
    try std.testing.expectError(error.InvalidLanguage, formatPrompt(gpa, .{ .text = "hi", .language = "" }));
}

test "frame estimation respects duration override, speed, reference rate, and bounds" {
    try std.testing.expectEqual(@as(usize, 100), try estimateFrames(.{ .text = "hello", .duration_sec = 4.0 }));
    try std.testing.expectEqual(@as(usize, min_frames), try estimateFrames(.{ .text = "hello", .duration_sec = 0.05 }));
    try std.testing.expectEqual(@as(usize, max_frames), try estimateFrames(.{ .text = "hello", .duration_sec = 120.0 }));
    try std.testing.expectError(error.InvalidDuration, estimateFrames(.{ .text = "hello", .duration_sec = 0.0 }));
    try std.testing.expectError(error.InvalidSpeed, estimateFrames(.{ .text = "hello", .speed = -1.0 }));

    const normal = try estimateFrames(.{ .text = "عندي meeting بكرة الصبح، فهحاول أخلص الشغل بدري النهارده." });
    const fast = try estimateFrames(.{ .text = "عندي meeting بكرة الصبح، فهحاول أخلص الشغل بدري النهارده.", .speed = 2.0 });
    try std.testing.expect(normal > fast and fast >= min_frames);

    const cloned = try estimateFrames(.{ .text = "hello world", .ref_text = "hello world", .ref_frames = 80 });
    try std.testing.expectEqual(@as(usize, 80), cloned);
}

test "chunkText splits Arabic and English sentences without breaking tags" {
    const gpa = std.testing.allocator;
    const text = "إزيك يا باشا؟ عامل إيه النهارده؟ عندي meeting بكرة الصبح! [laughter] We finish at 3.14 pm.";
    const chunks = try chunkText(gpa, text, 60);
    defer gpa.free(chunks);
    try std.testing.expect(chunks.len >= 3);
    // Verify no tag was split across chunks.
    var saw_tag = false;
    for (chunks) |c| {
        if (std.mem.indexOf(u8, c, "[laughter]") != null) saw_tag = true;
        try std.testing.expect((std.mem.indexOfScalar(u8, c, '[') == null) == (std.mem.indexOfScalar(u8, c, ']') == null));
    }
    try std.testing.expect(saw_tag);
}
