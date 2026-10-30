//! The speech models behind one face for `nuclis speak` and the speech
//! service: Qwen3-TTS (a checkpoint directory) and Drama 3 (one GGUF,
//! `audio_catalog.isFile`). `Voice` is the open model, `Synthesizer` one
//! utterance; what only one engine has (Qwen3-TTS's speakers and
//! instructions, Drama 3's references) the callers reach through the union.
const std = @import("std");
const inference = @import("inference");
const audio_catalog = @import("catalog.zig");

const qwen = inference.models.qwen3_tts_runtime;
const drama = inference.models.drama3_speech;
pub const Backend = qwen.Backend;

pub const Engine = enum { qwen3_tts, drama3 };

/// Qwen3-TTS's codec: 24 kHz, 1,920 samples a frame (12.5 Hz).
const qwen_rate = 24000;
const qwen_per_frame = 1920;

pub fn engineOf(path: []const u8) Engine {
    return if (audio_catalog.isFile(path)) .drama3 else .qwen3_tts;
}

/// An open speech model, heap-allocated (`open`, `close`).
pub const Voice = struct {
    gpa: std.mem.Allocator,
    model: union(Engine) {
        qwen3_tts: *qwen.Voice,
        drama3: *drama.Voice,
    },

    /// `diag` names a Qwen3-TTS tensor the binding refused.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, path: []const u8, diag: ?*inference.models.qwen3_tts.Diagnostic, backend: Backend) !*Voice {
        const self = try gpa.create(Voice);
        errdefer gpa.destroy(self);
        self.gpa = gpa;
        self.model = switch (engineOf(path)) {
            .qwen3_tts => .{ .qwen3_tts = try qwen.Voice.open(gpa, io, path, diag, .{ .backend = backend }) },
            .drama3 => .{ .drama3 = try drama.Voice.open(gpa, io, path) },
        };
        return self;
    }

    pub fn close(self: *Voice, io: std.Io) void {
        switch (self.model) {
            .qwen3_tts => |v| v.close(io),
            .drama3 => |v| v.close(),
        }
        self.gpa.destroy(self);
    }

    pub fn sampleRate(self: *const Voice) u32 {
        return switch (self.model) {
            .qwen3_tts => qwen_rate,
            .drama3 => drama.sample_rate,
        };
    }

    pub fn samplesPerFrame(self: *const Voice) usize {
        return switch (self.model) {
            .qwen3_tts => qwen_per_frame,
            .drama3 => drama.samples_per_frame,
        };
    }
};

pub const Finish = enum { eos, length };

/// One utterance of either engine; must not move once read.
pub const Synthesizer = union(Engine) {
    qwen3_tts: qwen.Synthesizer,
    drama3: drama.Synthesizer,

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
        }
    }

    pub fn frames(self: *const Synthesizer) usize {
        return switch (self.*) {
            inline else => |*s| s.frames,
        };
    }

    /// Why it ended: `length` when a budget cut it (Drama 3: any chunk).
    pub fn finish(self: *const Synthesizer) Finish {
        return switch (self.*) {
            .qwen3_tts => |*s| if (s.finish == .length) .length else .eos,
            .drama3 => |*s| if (s.stops[@backingInt(drama.Stop.length)] > 0) .length else .eos,
        };
    }
};

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
