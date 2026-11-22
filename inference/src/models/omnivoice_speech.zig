//! OmniVoice / VoiceTut-TTS high-level speech synthesizer: text in, 24 kHz mono
//! F32 PCM samples out, mirroring `drama3_speech.zig`.
//!
//! `Voice` opens `voicetut-tts-q8_0.gguf` once (embedded tokenizer, backbone,
//! Higgs Audio V2 codec decoder, and lazy codec encoder on first reference
//! clip). `Synthesizer` chunks long text via `omnivoice_text.chunkText`,
//! generates each chunk through iterative masked diffusion (using chunk 0 as
//! reference for chunks 1+ when in auto-voice / voice-design mode to keep the
//! speaker consistent), decodes to 24 kHz PCM, post-processes (silence trim,
//! peak/RMS scaling, edge fade), and streams samples through `read([]f32)`.
const std = @import("std");
const weights = @import("../runtime/weights.zig");
const hf_json = @import("../tokenizer/hf_json.zig");
const model = @import("omnivoice.zig");
const text_mod = @import("omnivoice_text.zig");
const runtime = @import("omnivoice_runtime.zig");
const hc = @import("../audio/higgs_codec.zig");
const hce = @import("../audio/higgs_codec_encoder.zig");

const Allocator = std.mem.Allocator;

pub const sample_rate = hc.sample_rate;
pub const samples_per_frame = hc.samples_per_frame;
pub const Frame = hc.Frame;
pub const SamplingConfig = runtime.SamplingConfig;

pub const Backend = enum { cpu, metal };
pub const Options = struct { backend: Backend = .cpu };

pub const Request = struct {
    text: []const u8,
    language: []const u8 = text_mod.default_language,
    instruct: ?[]const u8 = null,
    ref_text: ?[]const u8 = null,
    ref_codes: ?[]const Frame = null,
    ref_rms: ?f32 = null,
    denoise: bool = false,
    duration_sec: ?f32 = null,
    speed: f32 = 1.0,
    sampling: SamplingConfig = .{},
    postprocess: bool = true,
    chunk_bytes: usize = text_mod.default_chunk_bytes,
    cancel: ?*const std.atomic.Value(bool) = null,
};

pub const Voice = struct {
    gpa: Allocator,
    io: std.Io,
    mapped: weights.Mapped,
    tokenizer: hf_json.Tokenizer,
    model: runtime.Model,
    backend: Backend,
    codec: ?hc.Runtime = null,
    encoder: ?hce.Encoder = null,

    pub fn open(gpa: Allocator, io: std.Io, path: []const u8) !*Voice {
        return openWith(gpa, io, path, .{});
    }

    pub fn openWith(gpa: Allocator, io: std.Io, path: []const u8, options: Options) !*Voice {
        if (options.backend == .metal) return error.UnsupportedBackend;
        const self = try gpa.create(Voice);
        errdefer gpa.destroy(self);
        self.gpa = gpa;
        self.io = io;
        self.backend = options.backend;
        self.codec = null;
        self.encoder = null;

        self.mapped = try weights.Mapped.open(gpa, io, path);
        errdefer self.mapped.deinit(io);
        const doc = &self.mapped.document;
        const directory = self.mapped.mapping.memory[0..@intCast(doc.directory_bytes)];

        try model.verifyTensorNames(gpa, doc, directory);
        const embedded = try model.EmbeddedFiles.extract(doc, directory);
        try model.validateConfigJson(gpa, embedded.config_json);

        self.tokenizer = try hf_json.parse(gpa, embedded.tokenizer_json, .{});
        errdefer self.tokenizer.deinit();
        try text_mod.verifyControlTokens(&self.tokenizer);

        self.model = try runtime.Model.init(gpa, io, doc, self.mapped.view());
        errdefer self.model.deinit();

        self.codec = try hc.Runtime.init(gpa, io, doc.*, directory, self.mapped.view());
        errdefer if (self.codec) |*c| c.deinit();

        return self;
    }

    pub fn close(self: *Voice) void {
        if (self.encoder) |*e| e.deinit();
        if (self.codec) |*c| c.deinit();
        self.model.deinit();
        self.tokenizer.deinit();
        self.mapped.deinit(self.io);
        self.gpa.destroy(self);
    }

    /// Encodes a 24 kHz mono reference clip into `[T][8]` Higgs Audio V2 codes
    /// and computes its RMS (normalizing quiet clips to 0.1 RMS before encoding,
    /// matching `OmniVoice.create_voice_clone_prompt`).
    pub fn encodeReference(self: *Voice, gpa: Allocator, samples: []const f32) !struct { codes: []Frame, rms: f32 } {
        if (samples.len == 0 or samples.len > hce.max_samples) return error.InvalidLength;
        if (self.encoder == null) {
            const directory = self.mapped.mapping.memory[0..@intCast(self.mapped.document.directory_bytes)];
            self.encoder = try hce.Encoder.init(self.gpa, self.io, self.mapped.document, directory, self.mapped.view());
        }
        var sum_sq: f64 = 0;
        for (samples) |s| sum_sq += @as(f64, s) * @as(f64, s);
        const rms: f32 = @floatCast(@sqrt(sum_sq / @as(f64, @floatFromInt(samples.len))));
        const normalized = if (rms > 0.0 and rms < 0.1) blk: {
            const buf = try gpa.alloc(f32, samples.len);
            const gain = 0.1 / rms;
            for (buf, samples) |*d, s| d.* = s * gain;
            break :blk buf;
        } else null;
        defer if (normalized) |n| gpa.free(n);
        const codes = try self.encoder.?.encode(gpa, normalized orelse samples);
        return .{ .codes = codes, .rms = rms };
    }

    pub fn synthesize(self: *Voice, gpa: Allocator, req: Request) !Synthesizer {
        return Synthesizer.init(gpa, self, req);
    }
};

/// Chunked pseudo-streaming synthesizer: generates and decodes one text chunk
/// at a time on demand inside `read([]f32)`.
pub const Synthesizer = struct {
    gpa: Allocator,
    voice: *Voice,
    req: Request,
    chunks: []const []const u8,
    chunk_index: usize = 0,
    total_frames: usize = 0,
    /// Anchor codes from chunk 0 when auto-voice / voice-design spans multiple chunks.
    anchor_codes: ?[]Frame = null,
    /// Buffered decoded samples of the current chunk.
    pcm: ?[]f32 = null,
    pcm_pos: usize = 0,

    pub fn init(gpa: Allocator, voice: *Voice, req: Request) !Synthesizer {
        try text_mod.validateRequest(.{
            .text = req.text,
            .language = req.language,
            .instruct = req.instruct,
            .ref_text = req.ref_text,
            .ref_frames = if (req.ref_codes) |rc| rc.len else 0,
            .denoise = req.denoise,
            .duration_sec = req.duration_sec,
            .speed = req.speed,
        });
        const chunks = if (req.duration_sec != null) blk: {
            const single = try gpa.alloc([]const u8, 1);
            single[0] = req.text;
            break :blk single;
        } else try text_mod.chunkText(gpa, req.text, req.chunk_bytes);

        return .{
            .gpa = gpa,
            .voice = voice,
            .req = req,
            .chunks = chunks,
        };
    }

    pub fn deinit(self: *Synthesizer) void {
        if (self.pcm) |p| self.gpa.free(p);
        if (self.anchor_codes) |a| self.gpa.free(a);
        self.gpa.free(self.chunks);
        self.* = undefined;
    }

    pub fn frames(self: *const Synthesizer) usize {
        return self.total_frames;
    }

    pub fn finish(self: *Synthesizer) void {
        self.chunk_index = self.chunks.len;
        if (self.pcm) |p| {
            self.gpa.free(p);
            self.pcm = null;
        }
        self.pcm_pos = 0;
    }

    /// Reads up to `out.len` mono 24 kHz F32 samples; returns `0` at end of stream.
    pub fn read(self: *Synthesizer, out: []f32) !usize {
        if (out.len == 0) return 0;
        var written: usize = 0;
        while (written < out.len) {
            if (self.pcm) |buf| {
                const avail = buf.len - self.pcm_pos;
                if (avail > 0) {
                    const take = @min(avail, out.len - written);
                    @memcpy(out[written..][0..take], buf[self.pcm_pos..][0..take]);
                    self.pcm_pos += take;
                    written += take;
                    continue;
                }
                self.gpa.free(buf);
                self.pcm = null;
                self.pcm_pos = 0;
            }
            if (self.chunk_index >= self.chunks.len) break;
            try self.produceNextChunk();
        }
        return written;
    }

    fn produceNextChunk(self: *Synthesizer) !void {
        const ci = self.chunk_index;
        self.chunk_index += 1;
        const chunk_text = self.chunks[ci];

        // Use explicit reference if provided; otherwise reuse chunk 0 as anchor for chunks 1+.
        const use_anchor = self.req.ref_codes == null and ci > 0 and self.anchor_codes != null;
        const ref_codes: ?[]const Frame = if (self.req.ref_codes) |rc| rc else if (use_anchor) self.anchor_codes else null;
        const ref_text: ?[]const u8 = if (self.req.ref_codes != null) self.req.ref_text else if (use_anchor) self.chunks[0] else null;
        const instruct: ?[]const u8 = if (ref_codes == null) self.req.instruct else null;
        const denoise = self.req.denoise and ref_codes != null;

        const text_req: text_mod.Request = .{
            .text = chunk_text,
            .language = self.req.language,
            .instruct = instruct,
            .ref_text = ref_text,
            .ref_frames = if (ref_codes) |rc| rc.len else 0,
            .denoise = denoise,
            .duration_sec = if (ci == 0) self.req.duration_sec else null,
            .speed = self.req.speed,
        };

        const target_frames = try text_mod.estimateFrames(text_req);
        const prompt_ids = try text_mod.encodePrompt(self.gpa, &self.voice.tokenizer, text_req, .{});
        defer self.gpa.free(prompt_ids);

        var step_cfg = self.req.sampling;
        step_cfg.seed +%= ci;
        const codes = try self.voice.model.generate(
            self.gpa,
            prompt_ids,
            ref_codes,
            target_frames,
            step_cfg,
            self.req.cancel,
        );
        errdefer self.gpa.free(codes);

        self.total_frames += codes.len;
        const raw_pcm = try self.voice.codec.?.decode(self.gpa, codes);
        errdefer self.gpa.free(raw_pcm);

        if (ci == 0 and self.req.ref_codes == null and self.chunks.len > 1) {
            self.anchor_codes = codes;
        } else {
            self.gpa.free(codes);
        }

        if (self.req.postprocess) {
            const processed = try postprocessAudio(self.gpa, raw_pcm, self.req.ref_rms);
            self.gpa.free(raw_pcm);
            self.pcm = processed;
        } else {
            self.pcm = raw_pcm;
        }
        self.pcm_pos = 0;
    }
};

/// Trims leading/trailing silence, applies OmniVoice's peak/RMS scaling, and
/// adds a 20 ms linear fade-in/out to avoid boundary clicks.
pub fn postprocessAudio(alloc: Allocator, samples: []const f32, ref_rms: ?f32) Allocator.Error![]f32 {
    if (samples.len == 0) return alloc.alloc(f32, 0);

    // Trim leading/trailing silence below -45 dBFS (keep 100 ms cushion on each side).
    const thresh: f32 = 0.0056;
    const cushion: usize = sample_rate / 10; // 2400 samples = 100 ms
    var start: usize = 0;
    while (start < samples.len and @abs(samples[start]) < thresh) start += 1;
    var end: usize = samples.len;
    while (end > start and @abs(samples[end - 1]) < thresh) end -= 1;

    const slice = if (start >= end) samples else samples[start -| cushion..@min(samples.len, end + cushion)];
    const out = try alloc.dupe(f32, slice);

    // Volume normalization matching OmniVoice._post_process_audio.
    if (ref_rms) |r| {
        if (r > 0.0 and r < 0.1) {
            const scale = r / 0.1;
            for (out) |*s| s.* *= scale;
        }
    } else {
        var peak: f32 = 0;
        for (out) |s| peak = @max(peak, @abs(s));
        if (peak > 1e-6) {
            const scale = 0.5 / peak;
            for (out) |*s| s.* *= scale;
        }
    }

    // 20 ms fade-in / fade-out.
    const fade = @min(out.len / 2, sample_rate / 50);
    for (0..fade) |i| {
        const gain = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(fade));
        out[i] *= gain;
        out[out.len - 1 - i] *= gain;
    }
    return out;
}

// ---- tests -----------------------------------------------------------------

test "postprocessAudio trims silence, normalizes peak, and fades edges" {
    const gpa = std.testing.allocator;
    const n = 24_000; // 1 s
    const buf = try gpa.alloc(f32, n);
    defer gpa.free(buf);
    @memset(buf, 0);
    for (10_000..14_000) |i| buf[i] = 0.25;

    const out = try postprocessAudio(gpa, buf, null);
    defer gpa.free(out);
    try std.testing.expect(out.len < n);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[0], 1e-6);
    var peak: f32 = 0;
    for (out) |s| peak = @max(peak, @abs(s));
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), peak, 1e-4);
}
