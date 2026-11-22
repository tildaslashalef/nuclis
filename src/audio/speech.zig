//! The speech models behind one face for `nuclis speak` and the speech
//! service: Qwen3-TTS (a checkpoint directory), Drama 3 (one GGUF), and
//! OmniVoice / VoiceTut-TTS (one `audio.cpp` GGUF). `Voice` is the open model,
//! `Synthesizer` one utterance; what only one engine has (Qwen3-TTS's speakers
//! and instructions, Drama 3's references, OmniVoice's voice design and
//! zero-shot cloning) the callers reach through the union.
const std = @import("std");
const inference = @import("inference");
const audio_catalog = @import("catalog.zig");

const qwen = inference.models.qwen3_tts_runtime;
const drama = inference.models.drama3_speech;
const omni = inference.models.omnivoice_speech;
pub const Backend = qwen.Backend;

pub const Engine = enum { qwen3_tts, drama3, omnivoice };

/// Qwen3-TTS's codec: 24 kHz, 1,920 samples a frame (12.5 Hz).
const qwen_rate = 24000;
const qwen_per_frame = 1920;

/// Determines the speech engine for `path` by inspecting the GGUF header when
/// `path` is a `.gguf` file (`general.architecture` and
/// `audiocpp.model_spec.family`); falls back to `.drama3` for unreadable GGUFs.
pub fn engineOf(gpa: std.mem.Allocator, io: std.Io, path: []const u8) Engine {
    if (!audio_catalog.isFile(path)) return .qwen3_tts;
    var doc = inference.gguf.open(gpa, io, path, .{}) catch return .drama3;
    defer doc.deinit();
    if (audio_catalog.isOmniVoice(&doc)) return .omnivoice;
    return .drama3;
}

/// An open speech model, heap-allocated (`open`, `close`).
pub const Voice = struct {
    gpa: std.mem.Allocator,
    model: union(Engine) {
        qwen3_tts: *qwen.Voice,
        drama3: *drama.Voice,
        omnivoice: *omni.Voice,
    },

    /// `diag` names a Qwen3-TTS tensor the binding refused.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, path: []const u8, diag: ?*inference.models.qwen3_tts.Diagnostic, backend: Backend) !*Voice {
        const self = try gpa.create(Voice);
        errdefer gpa.destroy(self);
        self.gpa = gpa;
        self.model = switch (engineOf(gpa, io, path)) {
            .qwen3_tts => .{ .qwen3_tts = try qwen.Voice.open(gpa, io, path, diag, .{ .backend = backend }) },
            .drama3 => .{ .drama3 = try drama.Voice.openWith(gpa, io, path, .{ .backend = switch (backend) {
                .cpu => .cpu,
                .metal => .metal,
            } }) },
            .omnivoice => .{ .omnivoice = try omni.Voice.openWith(gpa, io, path, .{ .backend = switch (backend) {
                .cpu => .cpu,
                .metal => .metal,
            } }) },
        };
        return self;
    }

    pub fn close(self: *Voice, io: std.Io) void {
        switch (self.model) {
            .qwen3_tts => |v| v.close(io),
            .drama3 => |v| v.close(),
            .omnivoice => |v| v.close(),
        }
        self.gpa.destroy(self);
    }

    pub fn sampleRate(self: *const Voice) u32 {
        return switch (self.model) {
            .qwen3_tts => qwen_rate,
            .drama3 => drama.sample_rate,
            .omnivoice => omni.sample_rate,
        };
    }

    pub fn samplesPerFrame(self: *const Voice) usize {
        return switch (self.model) {
            .qwen3_tts => qwen_per_frame,
            .drama3 => drama.samples_per_frame,
            .omnivoice => omni.samples_per_frame,
        };
    }
};

pub const Finish = enum { eos, length };

/// One utterance of any speech engine; must not move once read.
pub const Synthesizer = union(Engine) {
    qwen3_tts: qwen.Synthesizer,
    drama3: drama.Synthesizer,
    omnivoice: omni.Synthesizer,

    pub fn deinit(self: *Synthesizer) void {
        switch (self.*) {
            inline else => |*s| s.deinit(),
        }
    }

    /// Up to `out.len` next samples; 0 once the utterance is over.
    pub fn read(self: *Synthesizer, out: []f32) !usize {
        return switch (self.*) {
            inline else => |*s| s.read(out),
        };
    }

    /// Frames a `read` makes at most (after the first, for Qwen3-TTS).
    pub fn setChunk(self: *Synthesizer, count: usize) void {
        switch (self.*) {
            .qwen3_tts => |*s| s.chunk = count,
            .drama3 => |*s| s.frames_per_read = count,
            .omnivoice => {},
        }
    }

    pub fn frames(self: *const Synthesizer) usize {
        return switch (self.*) {
            .qwen3_tts => |*s| s.frames,
            .drama3 => |*s| s.frames,
            .omnivoice => |*s| s.frames(),
        };
    }

    /// Why it ended: `length` when a budget cut it (Drama 3: any chunk).
    pub fn finish(self: *const Synthesizer) Finish {
        return switch (self.*) {
            .qwen3_tts => |*s| if (s.finish == .length) .length else .eos,
            .drama3 => |*s| if (s.stops[@backingInt(drama.Stop.length)] > 0) .length else .eos,
            .omnivoice => .eos,
        };
    }
};

/// Allowed OmniVoice voice-design attribute categories, for error messages.
pub const omnivoice_instruct_help = "comma-separated tags, at most one per category: gender (male, female), age (child, teenager, young adult, middle-aged, elderly), pitch (very low/low/moderate/high/very high pitch), style (whisper), accent (american/british/... accent)";

/// What is wrong with a Drama 3 input `err` refuses, for a message; null
/// for any other error.
pub fn inputProblem(err: anyerror) ?[]const u8 {
    return switch (err) {
        error.ReservedText => "it holds a token the prompt reserves (<|im_start|>, <|im_end|>, …)",
        error.InvalidSpeakerTag => "a speaker tag is <|speaker:0|> to <|speaker:15|>",
        error.UnknownSpecialToken => "it holds a <|…|> token other than a speaker tag",
        error.UnbalancedCue => "a [cue] has an unmatched bracket",
        error.NestedCue => "a [cue] holds another",
        error.UnclosedTag => "a paired tag (<whisper>…</whisper>) is not closed",
        error.PromptTooLong => "a sentence is too long for one prompt; split it",
        else => null,
    };
}

// ---- tests -----------------------------------------------------------------

fn writeTestGguf(io: std.Io, dir: std.Io.Dir, name: []const u8, arch: []const u8, family: ?[]const u8) !void {
    const gpa = std.testing.allocator;
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(gpa);
    const kv_count: u64 = if (family != null) 2 else 1;
    try buf.appendSlice(gpa, "GGUF");
    var tmp4: [4]u8 = undefined;
    var tmp8: [8]u8 = undefined;
    std.mem.writeInt(u32, &tmp4, 3, .little);
    try buf.appendSlice(gpa, &tmp4);
    std.mem.writeInt(u64, &tmp8, 0, .little); // tensor count
    try buf.appendSlice(gpa, &tmp8);
    std.mem.writeInt(u64, &tmp8, kv_count, .little);
    try buf.appendSlice(gpa, &tmp8);

    const writeKv = struct {
        fn put(out: *std.ArrayList(u8), a: std.mem.Allocator, k: []const u8, v: []const u8) !void {
            var b8: [8]u8 = undefined;
            var b4: [4]u8 = undefined;
            std.mem.writeInt(u64, &b8, k.len, .little);
            try out.appendSlice(a, &b8);
            try out.appendSlice(a, k);
            std.mem.writeInt(u32, &b4, 8, .little); // GGUF string
            try out.appendSlice(a, &b4);
            std.mem.writeInt(u64, &b8, v.len, .little);
            try out.appendSlice(a, &b8);
            try out.appendSlice(a, v);
        }
    }.put;
    try writeKv(&buf, gpa, "general.architecture", arch);
    if (family) |f| try writeKv(&buf, gpa, "audiocpp.model_spec.family", f);
    const pad = (32 - buf.items.len % 32) % 32;
    try buf.appendNTimes(gpa, 0, pad);
    try dir.writeFile(io, .{ .sub_path = name, .data = buf.items });
}

test "engineOf dispatches Qwen3-TTS directories, Drama 3 GGUFs, and OmniVoice GGUFs" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(dir);

    try std.testing.expectEqual(Engine.qwen3_tts, engineOf(gpa, io, dir));

    try writeTestGguf(io, tmp.dir, "drama.gguf", "drama3", null);
    const drama_path = try std.fs.path.join(gpa, &.{ dir, "drama.gguf" });
    defer gpa.free(drama_path);
    try std.testing.expectEqual(Engine.drama3, engineOf(gpa, io, drama_path));

    try writeTestGguf(io, tmp.dir, "omni.gguf", "audiocpp", "omnivoice");
    const omni_path = try std.fs.path.join(gpa, &.{ dir, "omni.gguf" });
    defer gpa.free(omni_path);
    try std.testing.expectEqual(Engine.omnivoice, engineOf(gpa, io, omni_path));

    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const described = (try audio_catalog.describeSpeech(arena_state.allocator(), io, omni_path)).?;
    try std.testing.expectEqual(audio_catalog.Mode.clone, described.mode);
    try std.testing.expect(described.instructions);
    try std.testing.expectEqual(@as(u32, 24000), described.sample_rate);
    try std.testing.expectEqualStrings("omnivoice", described.architecture);
    try std.testing.expectEqualStrings("Q8_0", described.quantization);
    try std.testing.expectEqual(@as(usize, 1), described.languages.len);
    try std.testing.expectEqualStrings("arz", described.languages[0]);
}
