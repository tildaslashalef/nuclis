//! Qwen3-TTS CustomVoice and Base, 0.6B and 1.7B: `Config` from the
//! checkpoint's `config.json` and `generation_config.json`, `bind` over the
//! 402 (0.6B) or 404 (1.7B: the predictor's input projection) tensors of
//! `model.safetensors` by name and shape, plus a Base's 76 speaker-encoder
//! tensors, the text into token ids (`encodeText`), and `Prompt.build`, the
//! talker's input rows as (text id, codec id) pairs. The forward is
//! qwen3_tts_runtime.zig; the codec decoder audio/qwen3_codec.zig. Contract:
//! docs/models/qwen3-tts.md § Forward pass and § Voice cloning (the Base).
const std = @import("std");
const safetensors = @import("../formats/safetensors.zig");
const hf_json = @import("../tokenizer/hf_json.zig");
const vocab_merges = @import("../tokenizer/vocab_merges.zig");

const Allocator = std.mem.Allocator;
const Ref = safetensors.Ref;

/// Codes per frame: the talker's group 0 and the predictor's 1–15.
pub const groups = 16;
/// Codes 0…2047 are audio; the rest of the talker's vocabulary are markers.
pub const codebook_size = 2048;

pub const Error = Allocator.Error || error{
    InvalidConfig,
    UnsupportedModelType,
    UnsupportedConfiguration,
    MissingTensor,
    TensorShapeMismatch,
    UnsupportedDtype,
    UnexpectedTensor,
};

pub const PromptError = Allocator.Error || error{ UnknownSpeaker, UnknownLanguage, InstructionUnsupported, EmptyText, VoiceRequired, CloningUnsupported, InvalidVoice };

/// One decoder stack's geometry (the talker's or the predictor's).
pub const Stack = struct {
    layers: usize,
    hidden: usize,
    intermediate: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rms_eps: f32,
    rope_theta: f32,
};

pub const Speaker = struct { name: []const u8, id: u32, dialect: ?[]const u8 };
pub const Language = struct { name: []const u8, id: u32 };

/// Sampling as `generation_config.json` sets it (docs/models/qwen3-tts.md
/// § Configuration); the defaults are `_merge_generate_kwargs`'s. A
/// temperature of 0 is greedy (`do_sample` false).
pub const Generation = struct {
    temperature: f32 = 0.9,
    top_k: usize = 50,
    top_p: f32 = 1.0,
    repetition_penalty: f32 = 1.05,
    predictor_temperature: f32 = 0.9,
    predictor_top_k: usize = 50,
    predictor_top_p: f32 = 1.0,
    /// Group-0 samples at most: the last one sampled at the limit is dropped.
    max_new_tokens: usize = 2048,
    /// The end code stays blocked until this many frames were sampled.
    min_new_tokens: usize = 2,
};

/// Text-vocabulary ids of the rendered prompt.
pub const TextIds = struct { im_start: u32, im_end: u32, assistant: u32, tts_pad: u32, tts_bos: u32, tts_eos: u32 };
/// Codec-vocabulary marker ids.
pub const CodecIds = struct { pad: u32, bos: u32, eos: u32, think: u32, nothink: u32, think_bos: u32, think_eos: u32 };

/// A Base's speaker encoder (ECAPA-TDNN), `speaker_encoder_config` under
/// the package's defaults. Block 0 is a TDNN, blocks 1…n−2 SE-Res2Net, the
/// last entry `mfa`'s (docs/models/qwen3-tts.md § The speaker encoder).
pub const SpeakerEncoder = struct {
    mel: usize,
    /// The embedding's width: the talker's, as it is a prompt row.
    dim: usize,
    channels: []const usize,
    kernels: []const usize,
    dilations: []const usize,
    attention: usize,
    scale: usize,
    se: usize,
    sample_rate: u32,
};

/// Owns the speaker and language names (`deinit`).
pub const Config = struct {
    storage: std.heap.ArenaAllocator,
    talker: Stack,
    predictor: Stack,
    /// The talker's vocabulary: 2048 codes, then the markers.
    codec_vocab: usize,
    text_vocab: usize,
    text_hidden: usize,
    /// `tts_model_size`; the authors drop the instruction on `0b6`.
    size: []const u8,
    text: TextIds,
    codec: CodecIds,
    speakers: []const Speaker,
    languages: []const Language,
    generation: Generation,
    /// A Base's (`tts_model_type` `base`); null on CustomVoice.
    speaker_encoder: ?SpeakerEncoder,

    pub fn deinit(self: *Config) void {
        self.storage.deinit();
        self.* = undefined;
    }

    /// Whether the model performs an instruction (the 1.7B; `generate_custom_voice`).
    pub fn takesInstruction(self: *const Config) bool {
        return !std.mem.eql(u8, self.size, "0b6");
    }

    /// Whether the model clones a voice from a reference clip (a Base).
    pub fn clones(self: *const Config) bool {
        return self.speaker_encoder != null;
    }

    /// Whether the predictor's inputs pass through `small_to_mtp_projection`.
    pub fn projects(self: *const Config) bool {
        return self.predictor.hidden != self.talker.hidden;
    }

    pub fn speaker(self: *const Config, name: []const u8) ?Speaker {
        for (self.speakers) |s| if (std.ascii.eqlIgnoreCase(s.name, name)) return s;
        return null;
    }

    pub fn language(self: *const Config, name: []const u8) ?Language {
        for (self.languages) |l| if (std.ascii.eqlIgnoreCase(l.name, name)) return l;
        return null;
    }
};

const StackJson = struct {
    num_hidden_layers: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    hidden_act: []const u8,
    attention_bias: bool = false,
    use_sliding_window: bool = false,
    vocab_size: usize,
    num_code_groups: usize,

    fn stack(self: StackJson) Error!Stack {
        if (!std.mem.eql(u8, self.hidden_act, "silu") or self.attention_bias or self.use_sliding_window or
            self.num_code_groups != groups) return error.UnsupportedConfiguration;
        if (self.num_hidden_layers == 0 or self.hidden_size == 0 or self.intermediate_size == 0 or self.head_dim == 0 or
            self.head_dim % 2 != 0 or self.num_key_value_heads == 0 or self.num_attention_heads % self.num_key_value_heads != 0 or
            self.num_hidden_layers > 256 or self.hidden_size > 1 << 16 or self.intermediate_size > 1 << 18 or self.head_dim > 1024 or
            self.num_attention_heads > 256 or !(self.rms_norm_eps > 0) or !(self.rope_theta >= 1)) return error.InvalidConfig;
        return .{
            .layers = self.num_hidden_layers,
            .hidden = self.hidden_size,
            .intermediate = self.intermediate_size,
            .heads = self.num_attention_heads,
            .kv_heads = self.num_key_value_heads,
            .head_dim = self.head_dim,
            .rms_eps = self.rms_norm_eps,
            .rope_theta = self.rope_theta,
        };
    }
};

const SpeakerEncoderJson = struct {
    mel_dim: usize = 128,
    enc_dim: usize = 1024,
    enc_channels: []const usize = &.{ 512, 512, 512, 512, 1536 },
    enc_kernel_sizes: []const usize = &.{ 5, 3, 3, 3, 1 },
    enc_dilations: []const usize = &.{ 1, 2, 3, 4, 1 },
    enc_attention_channels: usize = 128,
    enc_res2net_scale: usize = 8,
    enc_se_channels: usize = 128,
    sample_rate: u32 = 24000,

    fn encoder(self: SpeakerEncoderJson, talker_hidden: usize) Error!SpeakerEncoder {
        // The front end is fixed by `extract_speaker_embedding`: 128 mel bins of 24 kHz.
        if (self.mel_dim != 128 or self.sample_rate != 24000) return error.UnsupportedConfiguration;
        const c = self.enc_channels;
        const n = c.len;
        if (self.enc_dim != talker_hidden or n < 3 or n > 16 or self.enc_kernel_sizes.len != n or self.enc_dilations.len != n or
            self.enc_res2net_scale < 2 or self.enc_res2net_scale > 64 or self.enc_attention_channels == 0 or
            self.enc_attention_channels > 1 << 14 or self.enc_se_channels == 0 or self.enc_se_channels > 1 << 14)
            return error.InvalidConfig;
        for (c, self.enc_kernel_sizes, self.enc_dilations) |width, k, d| {
            if (width == 0 or width > 1 << 14 or k == 0 or k > 15 or d == 0 or d > 16) return error.InvalidConfig;
            // `padding="same"` is symmetric only for an even d·(k − 1).
            if (d * (k - 1) % 2 != 0) return error.UnsupportedConfiguration;
        }
        // Each SE-Res2Net block adds its input and splits into `scale` parts;
        // `mfa` reads the blocks' outputs concatenated.
        var concatenated: usize = 0;
        for (1..n - 1) |i| {
            if (c[i] != c[i - 1] or c[i] % self.enc_res2net_scale != 0) return error.InvalidConfig;
            concatenated += c[i];
        }
        if (c[n - 1] != concatenated) return error.InvalidConfig;
        return .{
            .mel = self.mel_dim,
            .dim = self.enc_dim,
            .channels = c,
            .kernels = self.enc_kernel_sizes,
            .dilations = self.enc_dilations,
            .attention = self.enc_attention_channels,
            .scale = self.enc_res2net_scale,
            .se = self.enc_se_channels,
            .sample_rate = self.sample_rate,
        };
    }
};

const ConfigJson = struct {
    model_type: []const u8,
    tts_model_type: []const u8,
    speaker_encoder_config: ?SpeakerEncoderJson = null,
    tts_model_size: []const u8,
    im_start_token_id: u32,
    im_end_token_id: u32,
    assistant_token_id: u32,
    tts_pad_token_id: u32,
    tts_bos_token_id: u32,
    tts_eos_token_id: u32,
    talker_config: struct {
        num_hidden_layers: usize,
        hidden_size: usize,
        intermediate_size: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
        rms_norm_eps: f32,
        rope_theta: f32,
        hidden_act: []const u8,
        attention_bias: bool = false,
        use_sliding_window: bool = false,
        vocab_size: usize,
        num_code_groups: usize,
        text_vocab_size: usize,
        text_hidden_size: usize,
        codec_pad_id: u32,
        codec_bos_id: u32,
        codec_eos_token_id: u32,
        codec_think_id: u32,
        codec_nothink_id: u32,
        codec_think_bos_id: u32,
        codec_think_eos_id: u32,
        codec_language_id: std.json.ArrayHashMap(u32),
        spk_id: std.json.ArrayHashMap(u32),
        spk_is_dialect: std.json.ArrayHashMap(std.json.Value),
        code_predictor_config: StackJson,
    },
};

const GenerationJson = struct {
    temperature: ?f32 = null,
    top_k: ?usize = null,
    top_p: ?f32 = null,
    repetition_penalty: ?f32 = null,
    subtalker_temperature: ?f32 = null,
    subtalker_top_k: ?usize = null,
    subtalker_top_p: ?f32 = null,
    max_new_tokens: ?usize = null,
    do_sample: bool = true,
    subtalker_dosample: bool = true,
};

/// Parses `config.json` and, when present, `generation_config.json`. Any
/// model but `qwen3_tts` CustomVoice or Base is `UnsupportedModelType`. The
/// bytes may be freed once this returns.
pub fn parseConfig(gpa: Allocator, config_json: []const u8, generation_json: ?[]const u8) Error!Config {
    var storage: std.heap.ArenaAllocator = .init(gpa);
    errdefer storage.deinit();
    const a = storage.allocator();
    // Names outlive the bytes: every string is copied into the arena.
    const options: std.json.ParseOptions = .{ .ignore_unknown_fields = true, .max_value_len = config_json.len, .allocate = .alloc_always };
    const j = std.json.parseFromSliceLeaky(ConfigJson, a, config_json, options) catch |err| return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.InvalidConfig,
    };
    const base = std.mem.eql(u8, j.tts_model_type, "base");
    if (!std.mem.eql(u8, j.model_type, "qwen3_tts") or !(base or std.mem.eql(u8, j.tts_model_type, "custom_voice")))
        return error.UnsupportedModelType;
    const t = j.talker_config;
    const talker = try (StackJson{
        .num_hidden_layers = t.num_hidden_layers,
        .hidden_size = t.hidden_size,
        .intermediate_size = t.intermediate_size,
        .num_attention_heads = t.num_attention_heads,
        .num_key_value_heads = t.num_key_value_heads,
        .head_dim = t.head_dim,
        .rms_norm_eps = t.rms_norm_eps,
        .rope_theta = t.rope_theta,
        .hidden_act = t.hidden_act,
        .attention_bias = t.attention_bias,
        .use_sliding_window = t.use_sliding_window,
        .vocab_size = t.vocab_size,
        .num_code_groups = t.num_code_groups,
    }).stack();
    const predictor = try t.code_predictor_config.stack();
    // The predictor's heads cover exactly the codebook; the talker's
    // vocabulary is the codebook then its markers.
    if (t.code_predictor_config.vocab_size != codebook_size or t.vocab_size <= codebook_size or t.vocab_size > 1 << 16 or
        t.text_vocab_size == 0 or t.text_vocab_size > 1 << 24 or t.text_hidden_size == 0 or t.text_hidden_size > 1 << 16)
        return error.InvalidConfig;
    const codec: CodecIds = .{
        .pad = t.codec_pad_id,
        .bos = t.codec_bos_id,
        .eos = t.codec_eos_token_id,
        .think = t.codec_think_id,
        .nothink = t.codec_nothink_id,
        .think_bos = t.codec_think_bos_id,
        .think_eos = t.codec_think_eos_id,
    };
    inline for (@typeInfo(CodecIds).@"struct".field_names) |name| {
        const id = @field(codec, name);
        if (id < codebook_size or id >= t.vocab_size) return error.InvalidConfig;
    }
    const text: TextIds = .{
        .im_start = j.im_start_token_id,
        .im_end = j.im_end_token_id,
        .assistant = j.assistant_token_id,
        .tts_pad = j.tts_pad_token_id,
        .tts_bos = j.tts_bos_token_id,
        .tts_eos = j.tts_eos_token_id,
    };
    inline for (@typeInfo(TextIds).@"struct".field_names) |name| {
        if (@field(text, name) >= t.text_vocab_size) return error.InvalidConfig;
    }
    const languages = try a.alloc(Language, t.codec_language_id.map.count());
    for (languages, t.codec_language_id.map.keys(), t.codec_language_id.map.values()) |*l, name, id| {
        if (id < codebook_size or id >= t.vocab_size) return error.InvalidConfig;
        l.* = .{ .name = name, .id = id };
    }
    const speakers = try a.alloc(Speaker, t.spk_id.map.count());
    for (speakers, t.spk_id.map.keys(), t.spk_id.map.values()) |*s, name, id| {
        if (id < codebook_size or id >= t.vocab_size) return error.InvalidConfig;
        const dialect: ?[]const u8 = switch (t.spk_is_dialect.map.get(name) orelse std.json.Value{ .bool = false }) {
            .bool => |b| if (b) return error.InvalidConfig else null,
            .string => |d| d,
            else => return error.InvalidConfig,
        };
        if (dialect) |d| {
            var known = false;
            for (languages) |l| known = known or std.mem.eql(u8, l.name, d);
            if (!known) return error.InvalidConfig;
        }
        s.* = .{ .name = name, .id = id, .dialect = dialect };
    }
    var generation: Generation = .{};
    if (generation_json) |bytes| {
        const g = std.json.parseFromSliceLeaky(GenerationJson, a, bytes, .{ .ignore_unknown_fields = true }) catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            else => error.InvalidConfig,
        };
        if (g.temperature) |v| generation.temperature = v;
        if (g.top_k) |v| generation.top_k = v;
        if (g.top_p) |v| generation.top_p = v;
        if (g.repetition_penalty) |v| generation.repetition_penalty = v;
        if (g.subtalker_temperature) |v| generation.predictor_temperature = v;
        if (g.subtalker_top_k) |v| generation.predictor_top_k = v;
        if (g.subtalker_top_p) |v| generation.predictor_top_p = v;
        if (g.max_new_tokens) |v| generation.max_new_tokens = v;
        // Greedy is temperature 0; the repetition penalty and the end/marker
        // processors still apply, as in transformers' greedy search.
        if (!g.do_sample) generation.temperature = 0;
        if (!g.subtalker_dosample) generation.predictor_temperature = 0;
    }
    if (generation.max_new_tokens < generation.min_new_tokens) return error.InvalidConfig;
    // The package builds a speaker encoder for a Base only, from defaults when the key is absent.
    const speaker_encoder: ?SpeakerEncoder = if (base) try (j.speaker_encoder_config orelse SpeakerEncoderJson{}).encoder(talker.hidden) else null;
    return .{
        .storage = storage,
        .talker = talker,
        .predictor = predictor,
        .codec_vocab = t.vocab_size,
        .text_vocab = t.text_vocab_size,
        .text_hidden = t.text_hidden_size,
        .size = j.tts_model_size,
        .text = text,
        .codec = codec,
        .speakers = speakers,
        .languages = languages,
        .generation = generation,
        .speaker_encoder = speaker_encoder,
    };
}

// ---- binding ---------------------------------------------------------------

/// Names the tensor behind a `bind` error.
pub const Diagnostic = struct {
    buffer: [160]u8 = undefined,
    len: usize = 0,

    pub fn name(self: *const Diagnostic) []const u8 {
        return self.buffer[0..self.len];
    }
    pub fn set(self: ?*Diagnostic, text: []const u8) void {
        const d = self orelse return;
        d.len = @min(text.len, d.buffer.len);
        @memcpy(d.buffer[0..d.len], text[0..d.len]);
    }
};

pub const Layer = struct {
    input_norm: Ref,
    post_norm: Ref,
    q: Ref,
    k: Ref,
    v: Ref,
    o: Ref,
    q_norm: Ref,
    k_norm: Ref,
    gate: Ref,
    up: Ref,
    down: Ref,
};
pub const Linear = struct { weight: Ref, bias: ?Ref };
/// A convolution, weight [out, in, kernel] and bias [out].
pub const Conv = struct { weight: Ref, bias: Ref };
/// One SE-Res2Net block; `res2net[j]` maps part j + 1 (`deinit` frees the slice).
pub const SeRes2Net = struct { tdnn1: Conv, res2net: []Conv, tdnn2: Conv, se1: Conv, se2: Conv };
/// A Base's speaker encoder; `blocks` are blocks 1…n−2.
pub const SpeakerBinding = struct { first: Conv, blocks: []SeRes2Net, mfa: Conv, asp_tdnn: Conv, asp_conv: Conv, fc: Conv };

/// The tensors, borrowing the checkpoint; the layer slices are the allocator's (`deinit`).
pub const Binding = struct {
    text_embedding: Ref,
    text_fc1: Linear,
    text_fc2: Linear,
    codec_embedding: Ref,
    talker: []Layer,
    norm: Ref,
    codec_head: Ref,
    /// `small_to_mtp_projection`, the 1.7B's only.
    projection: ?Linear,
    predictor: []Layer,
    /// Group g's input table is `embeddings[g − 1]`, its head `heads[g − 1]`, g = 1…15.
    embeddings: [groups - 1]Ref,
    heads: [groups - 1]Ref,
    predictor_norm: Ref,
    /// A Base's; null on CustomVoice.
    speaker_encoder: ?SpeakerBinding,
    tensors: u32,
    bytes: u64,

    pub fn deinit(self: *Binding, gpa: Allocator) void {
        gpa.free(self.talker);
        gpa.free(self.predictor);
        if (self.speaker_encoder) |e| freeBlocks(gpa, e.blocks);
        self.* = undefined;
    }
};

/// A tensor `bind` asks for, for building test files.
const Spec = struct { name: []const u8, shape: []const u64 };

/// Looks tensors up by name and shape, or (with `record`) lists them instead.
const Binder = struct {
    gpa: Allocator,
    checkpoint: ?*const safetensors.Checkpoint,
    bound: std.StringHashMapUnmanaged(void) = .empty,
    record: ?*std.ArrayList(Spec) = null,
    diagnostic: ?*Diagnostic,
    tensors: u32 = 0,
    bytes: u64 = 0,
    placeholder: safetensors.Tensor = .{ .name = "", .dtype = .bf16, .shape = &.{}, .offset = 0, .bytes = 0, .elements = 0 },

    fn deinit(self: *Binder) void {
        self.bound.deinit(self.gpa);
    }

    fn get(self: *Binder, comptime format: []const u8, args: anytype, shape: []const u64) Error!Ref {
        return self.at("talker." ++ format, args, shape);
    }

    /// A speaker-encoder convolution: weight [out, in, kernel], bias [out].
    fn conv(self: *Binder, comptime format: []const u8, args: anytype, out: u64, in: u64, kernel: u64) Error!Conv {
        return .{
            .weight = try self.at("speaker_encoder." ++ format ++ ".weight", args, &.{ out, in, kernel }),
            .bias = try self.at("speaker_encoder." ++ format ++ ".bias", args, &.{out}),
        };
    }

    fn at(self: *Binder, comptime format: []const u8, args: anytype, shape: []const u64) Error!Ref {
        var buffer: [160]u8 = undefined;
        const name = std.fmt.bufPrint(&buffer, format, args) catch unreachable;
        if (self.record) |list| {
            try list.ensureUnusedCapacity(self.gpa, 1);
            const owned = try self.gpa.dupe(u8, name);
            errdefer self.gpa.free(owned);
            list.appendAssumeCapacity(.{ .name = owned, .shape = try self.gpa.dupe(u64, shape) });
            return .{ .tensor = &self.placeholder, .bytes = &.{} };
        }
        const ref = self.checkpoint.?.get(name) orelse {
            Diagnostic.set(self.diagnostic, name);
            return error.MissingTensor;
        };
        if (!std.mem.eql(u64, ref.tensor.shape, shape)) {
            Diagnostic.set(self.diagnostic, name);
            return error.TensorShapeMismatch;
        }
        switch (ref.tensor.dtype) {
            .f32, .f16, .bf16 => {},
            else => {
                Diagnostic.set(self.diagnostic, name);
                return error.UnsupportedDtype;
            },
        }
        try self.bound.put(self.gpa, ref.tensor.name, {});
        self.tensors += 1;
        self.bytes += ref.tensor.bytes;
        return ref;
    }

    fn layer(self: *Binder, comptime prefix: []const u8, i: usize, s: Stack) Error!Layer {
        const q: u64 = s.heads * s.head_dim;
        const kv: u64 = s.kv_heads * s.head_dim;
        const h: u64 = s.hidden;
        const f: u64 = s.intermediate;
        return .{
            .input_norm = try self.get(prefix ++ ".layers.{d}.input_layernorm.weight", .{i}, &.{h}),
            .post_norm = try self.get(prefix ++ ".layers.{d}.post_attention_layernorm.weight", .{i}, &.{h}),
            .q = try self.get(prefix ++ ".layers.{d}.self_attn.q_proj.weight", .{i}, &.{ q, h }),
            .k = try self.get(prefix ++ ".layers.{d}.self_attn.k_proj.weight", .{i}, &.{ kv, h }),
            .v = try self.get(prefix ++ ".layers.{d}.self_attn.v_proj.weight", .{i}, &.{ kv, h }),
            .o = try self.get(prefix ++ ".layers.{d}.self_attn.o_proj.weight", .{i}, &.{ h, q }),
            .q_norm = try self.get(prefix ++ ".layers.{d}.self_attn.q_norm.weight", .{i}, &.{s.head_dim}),
            .k_norm = try self.get(prefix ++ ".layers.{d}.self_attn.k_norm.weight", .{i}, &.{s.head_dim}),
            .gate = try self.get(prefix ++ ".layers.{d}.mlp.gate_proj.weight", .{i}, &.{ f, h }),
            .up = try self.get(prefix ++ ".layers.{d}.mlp.up_proj.weight", .{i}, &.{ f, h }),
            .down = try self.get(prefix ++ ".layers.{d}.mlp.down_proj.weight", .{i}, &.{ h, f }),
        };
    }
};

/// Binds `checkpoint` (`model.safetensors`) against `config`. A missing,
/// misshapen, non-float, or unknown tensor is refused and named in
/// `diagnostic`. The refs borrow `checkpoint`.
pub fn bind(gpa: Allocator, checkpoint: *const safetensors.Checkpoint, config: *const Config, diagnostic: ?*Diagnostic) Error!Binding {
    var binder: Binder = .{ .gpa = gpa, .checkpoint = checkpoint, .diagnostic = diagnostic };
    defer binder.deinit();
    var binding = try bindWith(&binder, config);
    errdefer binding.deinit(gpa);
    for (checkpoint.shards) |shard| for (shard.document.tensors) |t| {
        if (!binder.bound.contains(t.name)) {
            Diagnostic.set(diagnostic, t.name);
            return error.UnexpectedTensor;
        }
    };
    return binding;
}

fn bindWith(b: *Binder, c: *const Config) Error!Binding {
    var r: Binding = undefined;
    const w: u64 = c.talker.hidden;
    const p: u64 = c.predictor.hidden;
    const th: u64 = c.text_hidden;
    const cv: u64 = c.codec_vocab;
    r.text_embedding = try b.get("model.text_embedding.weight", .{}, &.{ c.text_vocab, th });
    r.text_fc1 = .{ .weight = try b.get("text_projection.linear_fc1.weight", .{}, &.{ th, th }), .bias = try b.get("text_projection.linear_fc1.bias", .{}, &.{th}) };
    r.text_fc2 = .{ .weight = try b.get("text_projection.linear_fc2.weight", .{}, &.{ w, th }), .bias = try b.get("text_projection.linear_fc2.bias", .{}, &.{w}) };
    r.codec_embedding = try b.get("model.codec_embedding.weight", .{}, &.{ cv, w });
    r.talker = try b.gpa.alloc(Layer, c.talker.layers);
    errdefer b.gpa.free(r.talker);
    for (r.talker, 0..) |*l, i| l.* = try b.layer("model", i, c.talker);
    r.norm = try b.get("model.norm.weight", .{}, &.{w});
    r.codec_head = try b.get("codec_head.weight", .{}, &.{ cv, w });
    r.projection = if (c.projects()) .{
        .weight = try b.get("code_predictor.small_to_mtp_projection.weight", .{}, &.{ p, w }),
        .bias = try b.get("code_predictor.small_to_mtp_projection.bias", .{}, &.{p}),
    } else null;
    r.predictor = try b.gpa.alloc(Layer, c.predictor.layers);
    errdefer b.gpa.free(r.predictor);
    for (r.predictor, 0..) |*l, i| l.* = try b.layer("code_predictor.model", i, c.predictor);
    for (&r.embeddings, &r.heads, 0..) |*e, *h, g| {
        e.* = try b.get("code_predictor.model.codec_embedding.{d}.weight", .{g}, &.{ codebook_size, w });
        h.* = try b.get("code_predictor.lm_head.{d}.weight", .{g}, &.{ codebook_size, p });
    }
    r.predictor_norm = try b.get("code_predictor.model.norm.weight", .{}, &.{p});
    r.speaker_encoder = if (c.speaker_encoder) |e| try bindSpeaker(b, e) else null;
    r.tensors = b.tensors;
    r.bytes = b.bytes;
    return r;
}

fn bindSpeaker(b: *Binder, e: SpeakerEncoder) Error!SpeakerBinding {
    const c = e.channels;
    const n = c.len;
    var r: SpeakerBinding = undefined;
    r.first = try b.conv("blocks.0.conv", .{}, c[0], e.mel, e.kernels[0]);
    r.blocks = try b.gpa.alloc(SeRes2Net, n - 2);
    var done: usize = 0;
    errdefer {
        for (r.blocks[0..done]) |block| b.gpa.free(block.res2net);
        b.gpa.free(r.blocks);
    }
    for (r.blocks, 1..) |*block, i| {
        const part = c[i] / e.scale;
        block.tdnn1 = try b.conv("blocks.{d}.tdnn1.conv", .{i}, c[i], c[i - 1], 1);
        block.res2net = try b.gpa.alloc(Conv, e.scale - 1);
        errdefer b.gpa.free(block.res2net);
        for (block.res2net, 0..) |*conv, j| conv.* = try b.conv("blocks.{d}.res2net_block.blocks.{d}.conv", .{ i, j }, part, part, e.kernels[i]);
        block.tdnn2 = try b.conv("blocks.{d}.tdnn2.conv", .{i}, c[i], c[i], 1);
        block.se1 = try b.conv("blocks.{d}.se_block.conv1", .{i}, e.se, c[i], 1);
        block.se2 = try b.conv("blocks.{d}.se_block.conv2", .{i}, c[i], e.se, 1);
        done += 1;
    }
    const last = c[n - 1];
    r.mfa = try b.conv("mfa.conv", .{}, last, last, e.kernels[n - 1]);
    r.asp_tdnn = try b.conv("asp.tdnn.conv", .{}, e.attention, 3 * last, 1);
    r.asp_conv = try b.conv("asp.conv", .{}, last, e.attention, 1);
    r.fc = try b.conv("fc", .{}, e.dim, 2 * last, 1);
    return r;
}

/// Frees `blocks` and each block's Res2Net slice.
fn freeBlocks(gpa: Allocator, blocks: []SeRes2Net) void {
    for (blocks) |block| gpa.free(block.res2net);
    gpa.free(blocks);
}

// ---- text and prompt ----------------------------------------------------------

/// Reads `vocab.json`, `merges.txt`, and `tokenizer_config.json` from `dir`.
pub fn loadTokenizer(gpa: Allocator, io: std.Io, dir: []const u8) !hf_json.Tokenizer {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var bytes: [3][]const u8 = undefined;
    for (&bytes, [_][]const u8{ "vocab.json", "merges.txt", "tokenizer_config.json" }) |*b, name| {
        const path = try std.fs.path.join(a, &.{ dir, name });
        b.* = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20));
    }
    return vocab_merges.load(gpa, bytes[0], bytes[1], bytes[2], .{});
}

/// Token ids of one request, sliced as the authors slice them (owned by the
/// caller's allocator: `deinit`).
pub const Text = struct {
    /// `<|im_start|>assistant\n`, three ids by position.
    role: [3]u32,
    text: []u32,
    /// `<|im_start|>user\n{instruct}<|im_end|>\n`, or empty.
    instruct: []u32,
    /// The rendered assistant text's ids, `role` and `text` slicing it.
    all: []u32,

    pub fn deinit(self: *Text, gpa: Allocator) void {
        gpa.free(self.all);
        gpa.free(self.instruct);
        self.* = undefined;
    }
};

/// Renders and tokenizes `text` and `instruct` as `Qwen3TTSModel` does
/// (`_build_assistant_text`, `_build_instruct_text`): ids 0–2 are the role,
/// the last 5 the closing `<|im_end|>\n<|im_start|>assistant\n`, the rest the
/// text, taken by position even when a text that begins with a newline makes
/// the role's third id another (docs/models/qwen3-tts.md § Tokenizer).
pub fn encodeText(gpa: Allocator, tokenizer: *const hf_json.Tokenizer, text: []const u8, instruct: ?[]const u8) !Text {
    const rendered = try std.fmt.allocPrint(gpa, "<|im_start|>assistant\n{s}<|im_end|>\n<|im_start|>assistant\n", .{text});
    defer gpa.free(rendered);
    const all = try tokenizer.encode(gpa, rendered, .{});
    errdefer gpa.free(all);
    if (all.len <= 8) return error.EmptyText;
    var instruct_ids: []u32 = &.{};
    if (instruct) |words| if (words.len > 0) {
        const block = try std.fmt.allocPrint(gpa, "<|im_start|>user\n{s}<|im_end|>\n", .{words});
        defer gpa.free(block);
        instruct_ids = try tokenizer.encode(gpa, block, .{});
    };
    return .{ .role = all[0..3].*, .text = all[3 .. all.len - 5], .instruct = instruct_ids, .all = all };
}

/// A cloned voice's reference transcript as the prompt takes it:
/// `_build_ref_text`'s rendering, its ids sliced `[3:-2]` (owned).
pub fn encodeReference(gpa: Allocator, tokenizer: *const hf_json.Tokenizer, text: []const u8) ![]u32 {
    const rendered = try std.fmt.allocPrint(gpa, "<|im_start|>assistant\n{s}<|im_end|>\n", .{text});
    defer gpa.free(rendered);
    const all = try tokenizer.encode(gpa, rendered, .{});
    defer gpa.free(all);
    if (all.len <= 5) return error.EmptyText;
    return gpa.dupe(u32, all[3 .. all.len - 2]);
}

/// One talker input row: the sum of a text-track and a codec-track term,
/// either of which may be absent, and on a Base a voice term.
pub const Row = struct {
    text: ?u32,
    codec: ?u32,
    voice: VoiceTerm = .none,

    /// A Base's codec-track terms that are no code of the table.
    pub const VoiceTerm = union(enum) {
        none,
        /// The speaker embedding (`Clone.embedding`).
        speaker,
        /// Reference frame k's 16 groups summed, as a generated frame's next row.
        frame: u32,
    };
};

/// A saved voice as a prompt reads it (borrowed): the speaker embedding at
/// the talker's width and, for ICL, the reference frames.
pub const Clone = struct {
    embedding: []const f32,
    frames: []const [groups]u16 = &.{},
};

pub const Request = struct {
    /// The rendered text's role ids and text ids (`encodeText`).
    role: [3]u32,
    text: []const u32,
    /// Instruction ids; non-empty only for a model that `takesInstruction`.
    instruct: []const u32 = &.{},
    /// A `Config.speakers` name; null leaves the speaker row out.
    speaker: ?[]const u8,
    /// A `Config.languages` name or `auto`.
    language: []const u8 = "auto",
};

pub const CloneRequest = struct {
    role: [3]u32,
    text: []const u32,
    /// The reference transcript's ids (`encodeReference`) for ICL; null for
    /// the speaker embedding alone (x-vector only).
    reference: ?[]const u32,
    language: []const u8 = "auto",
    /// Its frames are read only with `reference`.
    clone: Clone,
};

/// A prompt's rows and, during decoding, its text track: step k adds
/// `trailing[k]`, then `tts_pad` once they run out. `build` gives the
/// non-streaming prompt (docs/models/qwen3-tts.md § (a)), whose trailing
/// text is empty; `buildClone` a Base's streaming one (§ The prompt), whose
/// rows borrow its `clone`.
pub const Prompt = struct {
    rows: []Row,
    trailing: []u32 = &.{},
    clone: ?Clone = null,

    pub fn deinit(self: *Prompt, gpa: Allocator) void {
        gpa.free(self.rows);
        gpa.free(self.trailing);
        self.* = undefined;
    }

    /// The reference frames to decode, and discard, before the generated ones.
    pub fn primer(self: *const Prompt) []const [groups]u16 {
        return if (self.clone) |c| c.frames else &.{};
    }

    /// Instruction rows, role rows, the codec prefix, every text row over
    /// `codec_pad`, the end of text, then `tts_pad` over `codec_bos`. A Base
    /// has no speakers: `VoiceRequired` (`buildClone`).
    pub fn build(gpa: Allocator, config: *const Config, request: Request) PromptError!Prompt {
        if (config.clones()) return error.VoiceRequired;
        if (request.text.len == 0) return error.EmptyText;
        if (request.instruct.len > 0 and !config.takesInstruction()) return error.InstructionUnsupported;
        const auto = std.ascii.eqlIgnoreCase(request.language, "auto");
        var language: ?u32 = if (auto) null else (config.language(request.language) orelse return error.UnknownLanguage).id;
        var speaker: ?u32 = null;
        if (request.speaker) |name| {
            const s = config.speaker(name) orelse return error.UnknownSpeaker;
            speaker = s.id;
            // A dialect speaker speaks its dialect in Chinese or `auto`.
            if (s.dialect) |d| if (auto or std.ascii.eqlIgnoreCase(request.language, "chinese")) {
                language = config.language(d).?.id;
            };
        }
        const c = config.codec;
        var prefix_buffer: [6]u32 = undefined;
        var prefix: std.ArrayList(u32) = .initBuffer(&prefix_buffer);
        if (language) |id| prefix.appendSliceAssumeCapacity(&.{ c.think, c.think_bos, id, c.think_eos }) else prefix.appendSliceAssumeCapacity(&.{ c.nothink, c.think_bos, c.think_eos });
        if (speaker) |id| prefix.appendAssumeCapacity(id);
        prefix.appendAssumeCapacity(c.pad);

        const total = request.instruct.len + 3 + prefix.items.len + request.text.len + 2;
        const rows = try gpa.alloc(Row, total);
        var n: usize = 0;
        for (request.instruct, 0..) |id, i| rows[i] = .{ .text = id, .codec = null };
        n += request.instruct.len;
        for (request.role, 0..) |id, i| rows[n + i] = .{ .text = id, .codec = null };
        n += 3;
        // The text track is tts_pad until the last prefix row, which carries tts_bos.
        for (prefix.items, 0..) |id, i| rows[n + i] = .{ .text = if (i + 1 == prefix.items.len) config.text.tts_bos else config.text.tts_pad, .codec = id };
        n += prefix.items.len;
        for (request.text, 0..) |id, i| rows[n + i] = .{ .text = id, .codec = c.pad };
        n += request.text.len;
        rows[n] = .{ .text = config.text.tts_eos, .codec = c.pad };
        rows[n + 1] = .{ .text = config.text.tts_pad, .codec = c.bos };
        return .{ .rows = rows };
    }

    /// A Base's prompt for `request` (`generate_voice_clone`, streaming
    /// text): role rows; the codec prefix, the speaker embedding, and
    /// `codec_pad` over `tts_pad` … `tts_bos`; then x-vector only, the first
    /// text id over `codec_bos`, the rest of the text and `tts_eos`
    /// trailing; or ICL, the text track `reference ++ text ++ tts_eos`
    /// summed with `codec_bos` and the reference frames, the longer track's
    /// excess trailing (§ The prompt).
    pub fn buildClone(gpa: Allocator, config: *const Config, request: CloneRequest) PromptError!Prompt {
        if (!config.clones()) return error.CloningUnsupported;
        if (request.text.len == 0) return error.EmptyText;
        if (request.clone.embedding.len != config.talker.hidden) return error.InvalidVoice;
        if (request.reference) |r| if (r.len == 0 or request.clone.frames.len == 0) return error.InvalidVoice;
        for (request.clone.frames) |frame| for (frame) |code| if (code >= codebook_size) return error.InvalidVoice;
        const c = config.codec;
        const t = config.text;
        const auto = std.ascii.eqlIgnoreCase(request.language, "auto");
        var prefix_buffer: [4]u32 = undefined;
        var prefix: std.ArrayList(u32) = .initBuffer(&prefix_buffer);
        if (auto) prefix.appendSliceAssumeCapacity(&.{ c.nothink, c.think_bos, c.think_eos }) else {
            const language = config.language(request.language) orelse return error.UnknownLanguage;
            prefix.appendSliceAssumeCapacity(&.{ c.think, c.think_bos, language.id, c.think_eos });
        }
        // The text track's ids beyond the role and prefix, and the codec track's rows.
        var track: std.ArrayList(u32) = .empty;
        defer track.deinit(gpa);
        var trailing: std.ArrayList(u32) = .empty;
        errdefer trailing.deinit(gpa);
        var tail: usize = 1;
        if (request.reference) |reference| {
            try track.appendSlice(gpa, reference);
            try track.appendSlice(gpa, request.text);
            try track.append(gpa, t.tts_eos);
            tail = request.clone.frames.len + 1;
            if (track.items.len > tail) {
                try trailing.appendSlice(gpa, track.items[tail..]);
                track.shrinkRetainingCapacity(tail);
            } else {
                try trailing.append(gpa, t.tts_pad);
                try track.appendNTimes(gpa, t.tts_pad, tail - track.items.len);
            }
        } else {
            try track.append(gpa, request.text[0]);
            try trailing.appendSlice(gpa, request.text[1..]);
            try trailing.append(gpa, t.tts_eos);
        }
        const head = 3 + prefix.items.len + 2;
        const rows = try gpa.alloc(Row, head + tail);
        errdefer gpa.free(rows);
        for (request.role, 0..) |id, i| rows[i] = .{ .text = id, .codec = null };
        var n: usize = 3;
        for (prefix.items) |id| {
            rows[n] = .{ .text = t.tts_pad, .codec = id };
            n += 1;
        }
        rows[n] = .{ .text = t.tts_pad, .codec = null, .voice = .speaker };
        rows[n + 1] = .{ .text = t.tts_bos, .codec = c.pad };
        n += 2;
        for (track.items, 0..) |id, i| rows[n + i] = if (i == 0)
            .{ .text = id, .codec = c.bos }
        else
            .{ .text = id, .codec = null, .voice = .{ .frame = @intCast(i - 1) } };
        // x-vector only reads no frames, so none are decoded before its own.
        const clone: Clone = if (request.reference == null) .{ .embedding = request.clone.embedding } else request.clone;
        return .{ .rows = rows, .trailing = try trailing.toOwnedSlice(gpa), .clone = clone };
    }
};

// ---- tests -----------------------------------------------------------------

/// A small `config.json` of the published shape with the stacks scaled
/// down, for tests here and in the executable. A `base` has no speakers and
/// a speaker encoder of the published structure (five entries, scale 8).
pub fn testConfigJson(gpa: Allocator, size: []const u8, talker_hidden: usize, kind: []const u8) ![]u8 {
    const base = std.mem.eql(u8, kind, "base");
    const speakers =
        if (base) "\"spk_id\":{},\"spk_is_dialect\":{}" else "\"spk_id\":{\"ryan\":3061,\"eric\":2875},\"spk_is_dialect\":{\"ryan\":false,\"eric\":\"sichuan_dialect\"}";
    var encoder_buffer: [320]u8 = undefined;
    const encoder = if (base) try std.fmt.bufPrint(&encoder_buffer,
        \\,"speaker_encoder_config":{{"enc_dim":{d},"enc_channels":[16,16,16,16,48],"enc_kernel_sizes":[5,3,3,3,1],
        \\ "enc_dilations":[1,2,3,4,1],"enc_attention_channels":4,"enc_res2net_scale":8,"enc_se_channels":4,"sample_rate":24000}}
    , .{talker_hidden}) else "";
    return std.fmt.allocPrint(gpa,
        \\{{"model_type":"qwen3_tts","tts_model_type":"{s}","tts_model_size":"{s}"{s},
        \\ "im_start_token_id":20,"im_end_token_id":21,"assistant_token_id":7,"tts_pad_token_id":22,"tts_bos_token_id":23,"tts_eos_token_id":24,
        \\ "talker_config":{{"num_hidden_layers":2,"hidden_size":{d},"intermediate_size":12,"num_attention_heads":4,"num_key_value_heads":2,
        \\  "head_dim":4,"rms_norm_eps":1e-6,"rope_theta":1000000,"hidden_act":"silu","attention_bias":false,"use_sliding_window":false,
        \\  "vocab_size":3072,"num_code_groups":16,"text_vocab_size":32,"text_hidden_size":6,
        \\  "codec_pad_id":2148,"codec_bos_id":2149,"codec_eos_token_id":2150,"codec_think_id":2154,"codec_nothink_id":2155,
        \\  "codec_think_bos_id":2156,"codec_think_eos_id":2157,
        \\  "codec_language_id":{{"english":2050,"chinese":2055,"sichuan_dialect":2062}},
        \\  {s},
        \\  "code_predictor_config":{{"num_hidden_layers":1,"hidden_size":8,"intermediate_size":12,"num_attention_heads":4,
        \\   "num_key_value_heads":2,"head_dim":4,"rms_norm_eps":1e-6,"rope_theta":1000000,"hidden_act":"silu","vocab_size":2048,"num_code_groups":16}}}}}}
    , .{ kind, size, encoder, talker_hidden, speakers });
}

fn testConfig(gpa: Allocator, size: []const u8, talker_hidden: usize) !Config {
    return testConfigOf(gpa, size, talker_hidden, "custom_voice");
}

fn testConfigOf(gpa: Allocator, size: []const u8, talker_hidden: usize, kind: []const u8) !Config {
    const json = try testConfigJson(gpa, size, talker_hidden, kind);
    defer gpa.free(json);
    return parseConfig(gpa, json, "{\"max_new_tokens\":8192,\"top_k\":40}");
}

test "the config of either size, its generation settings, and its refusals" {
    const gpa = std.testing.allocator;
    var small = try testConfig(gpa, "0b6", 8);
    defer small.deinit();
    try std.testing.expect(!small.takesInstruction() and !small.projects());
    try std.testing.expectEqual(@as(usize, 8192), small.generation.max_new_tokens);
    try std.testing.expectEqual(@as(usize, 40), small.generation.top_k);
    try std.testing.expectEqual(@as(f32, 1.05), small.generation.repetition_penalty);
    try std.testing.expectEqualStrings("sichuan_dialect", small.speaker("Eric").?.dialect.?);
    var large = try testConfig(gpa, "1b7", 16);
    defer large.deinit();
    try std.testing.expect(large.takesInstruction() and large.projects() and !large.clones());

    const design = try testConfigJson(gpa, "1b7", 16, "voice_design");
    defer gpa.free(design);
    try std.testing.expectError(error.UnsupportedModelType, parseConfig(gpa, design, null));
    try std.testing.expectError(error.InvalidConfig, parseConfig(gpa, "{}", null));
    const json = try testConfigJson(gpa, "0b6", 8, "custom_voice");
    defer gpa.free(json);
    // `do_sample` false is greedy for that network alone; its other settings still apply.
    var greedy = try parseConfig(gpa, json, "{\"do_sample\":false,\"max_new_tokens\":256,\"temperature\":0.7}");
    defer greedy.deinit();
    try std.testing.expectEqual(@as(f32, 0), greedy.generation.temperature);
    try std.testing.expectEqual(@as(f32, 0.9), greedy.generation.predictor_temperature);
    try std.testing.expectEqual(@as(f32, 1.05), greedy.generation.repetition_penalty);
    try std.testing.expectEqual(@as(usize, 256), greedy.generation.max_new_tokens);
    var both = try parseConfig(gpa, json, "{\"do_sample\":false,\"subtalker_dosample\":false}");
    defer both.deinit();
    try std.testing.expectEqual(@as(f32, 0), both.generation.predictor_temperature);
}

test "a Base's speaker encoder: its config, the package's defaults, and its refusals" {
    const gpa = std.testing.allocator;
    var base = try testConfigOf(gpa, "1b7", 16, "base");
    defer base.deinit();
    try std.testing.expect(base.clones() and base.speakers.len == 0);
    const e = base.speaker_encoder.?;
    try std.testing.expectEqual(@as(usize, 16), e.dim);
    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 3, 4, 1 }, e.dilations);
    try std.testing.expectEqual(@as(usize, 8), e.scale);

    const json = try testConfigJson(gpa, "0b6", 8, "base");
    defer gpa.free(json);
    // Absent, the key takes the package's defaults: the published network, at width 1024.
    const start = std.mem.indexOf(u8, json, ",\"speaker_encoder_config\"").?;
    const end = std.mem.indexOfPos(u8, json, start, "}").? + 1;
    const bare = try std.mem.concat(gpa, u8, &.{ json[0..start], json[end..] });
    defer gpa.free(bare);
    try std.testing.expectError(error.InvalidConfig, parseConfig(gpa, bare, null));
    const wide = try std.mem.replaceOwned(u8, gpa, bare, "\"hidden_size\":8,\"intermediate_size\":12,\"num_attention_heads\":4,\"num_key_value_heads\":2,\n  \"head_dim\":4", "\"hidden_size\":1024,\"intermediate_size\":12,\"num_attention_heads\":4,\"num_key_value_heads\":2,\n  \"head_dim\":4");
    defer gpa.free(wide);
    var defaults = try parseConfig(gpa, wide, null);
    defer defaults.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 512, 512, 512, 512, 1536 }, defaults.speaker_encoder.?.channels);
    try std.testing.expectEqual(@as(usize, 128), defaults.speaker_encoder.?.attention);

    const Case = struct { from: []const u8, to: []const u8, err: Error };
    const cases = [_]Case{
        .{ .from = "\"enc_dim\":8", .to = "\"enc_dim\":16", .err = error.InvalidConfig },
        .{ .from = "\"sample_rate\":24000", .to = "\"sample_rate\":16000", .err = error.UnsupportedConfiguration },
        .{ .from = "[16,16,16,16,48]", .to = "[16,16,16,16,32]", .err = error.InvalidConfig },
        .{ .from = "[16,16,16,16,48]", .to = "[16,24,24,24,72]", .err = error.InvalidConfig },
        .{ .from = "[5,3,3,3,1]", .to = "[4,3,3,3,1]", .err = error.UnsupportedConfiguration },
        .{ .from = "[1,2,3,4,1]", .to = "[1,2,3,4]", .err = error.InvalidConfig },
    };
    for (cases) |case| {
        const changed = try std.mem.replaceOwned(u8, gpa, json, case.from, case.to);
        defer gpa.free(changed);
        try std.testing.expect(!std.mem.eql(u8, changed, json));
        try std.testing.expectError(case.err, parseConfig(gpa, changed, null));
    }
}

test "a Base's clone prompt: x-vector and ICL rows and trailing text, and the refusals" {
    const gpa = std.testing.allocator;
    var base = try testConfigOf(gpa, "0b6", 8, "base");
    defer base.deinit();
    const embedding: [8]f32 = @splat(0.5);
    const frames: [6][groups]u16 = @splat(@splat(7));
    const role = [3]u32{ 20, 7, 1 };
    const n: ?u32 = null;
    const head = [_]Row{
        .{ .text = 20, .codec = n },    .{ .text = 7, .codec = n },                     .{ .text = 1, .codec = n },
        .{ .text = 22, .codec = 2154 }, .{ .text = 22, .codec = 2156 },                 .{ .text = 22, .codec = 2050 },
        .{ .text = 22, .codec = 2157 }, .{ .text = 22, .codec = n, .voice = .speaker }, .{ .text = 23, .codec = 2148 },
    };

    // x-vector only: the first text id over codec_bos, the rest trailing.
    var xvector = try Prompt.buildClone(gpa, &base, .{ .role = role, .text = &.{ 30, 31, 32 }, .reference = null, .language = "english", .clone = .{ .embedding = &embedding, .frames = &frames } });
    defer xvector.deinit(gpa);
    try std.testing.expectEqualSlices(Row, &head ++ [_]Row{.{ .text = 30, .codec = 2149 }}, xvector.rows);
    try std.testing.expectEqualSlices(u32, &.{ 31, 32, 24 }, xvector.trailing);
    try std.testing.expectEqual(@as(usize, 0), xvector.primer().len);

    // ICL, the text track longer than the reference: its excess trails.
    var long = try Prompt.buildClone(gpa, &base, .{ .role = role, .text = &.{ 30, 31 }, .reference = &.{ 40, 41 }, .language = "english", .clone = .{ .embedding = &embedding, .frames = frames[0..3] } });
    defer long.deinit(gpa);
    try std.testing.expectEqualSlices(Row, &head ++ [_]Row{
        .{ .text = 40, .codec = 2149 },                        .{ .text = 41, .codec = n, .voice = .{ .frame = 0 } },
        .{ .text = 30, .codec = n, .voice = .{ .frame = 1 } }, .{ .text = 31, .codec = n, .voice = .{ .frame = 2 } },
    }, long.rows);
    try std.testing.expectEqualSlices(u32, &.{24}, long.trailing);
    try std.testing.expectEqual(@as(usize, 3), long.primer().len);

    // ICL, the reference longer: tts_pad fills the text track and trails.
    var short = try Prompt.buildClone(gpa, &base, .{ .role = role, .text = &.{ 30, 31 }, .reference = &.{ 40, 41 }, .clone = .{ .embedding = &embedding, .frames = &frames } });
    defer short.deinit(gpa);
    try std.testing.expectEqual(@as(?u32, 2155), short.rows[3].codec);
    const track = short.rows[short.rows.len - 7 ..];
    for (track, [_]u32{ 40, 41, 30, 31, 24, 22, 22 }, 0..) |row, id, i| {
        try std.testing.expectEqual(@as(?u32, id), row.text);
        if (i > 0) try std.testing.expectEqual(Row.VoiceTerm{ .frame = @intCast(i - 1) }, row.voice);
    }
    try std.testing.expectEqualSlices(u32, &.{22}, short.trailing);

    const voice: Clone = .{ .embedding = &embedding, .frames = &frames };
    try std.testing.expectError(error.VoiceRequired, Prompt.build(gpa, &base, .{ .role = role, .text = &.{30}, .speaker = null }));
    try std.testing.expectError(error.InvalidVoice, Prompt.buildClone(gpa, &base, .{ .role = role, .text = &.{30}, .reference = null, .clone = .{ .embedding = embedding[0..7] } }));
    try std.testing.expectError(error.InvalidVoice, Prompt.buildClone(gpa, &base, .{ .role = role, .text = &.{30}, .reference = &.{40}, .clone = .{ .embedding = &embedding } }));
    const wild = [1][groups]u16{@splat(codebook_size)};
    try std.testing.expectError(error.InvalidVoice, Prompt.buildClone(gpa, &base, .{ .role = role, .text = &.{30}, .reference = &.{40}, .clone = .{ .embedding = &embedding, .frames = &wild } }));
    try std.testing.expectError(error.UnknownLanguage, Prompt.buildClone(gpa, &base, .{ .role = role, .text = &.{30}, .reference = null, .language = "klingon", .clone = voice }));
    try std.testing.expectError(error.EmptyText, Prompt.buildClone(gpa, &base, .{ .role = role, .text = &.{}, .reference = null, .clone = voice }));
    var custom = try testConfig(gpa, "0b6", 8);
    defer custom.deinit();
    try std.testing.expectError(error.CloningUnsupported, Prompt.buildClone(gpa, &custom, .{ .role = role, .text = &.{30}, .reference = null, .clone = voice }));
}

test "the prompt's rows for both sizes, with and without an instruction" {
    const gpa = std.testing.allocator;
    var small = try testConfig(gpa, "0b6", 8);
    defer small.deinit();
    const role = [3]u32{ 20, 7, 1 };
    const text = [_]u32{ 30, 31 };
    var p = try Prompt.build(gpa, &small, .{ .role = role, .text = &text, .speaker = "ryan", .language = "english" });
    defer p.deinit(gpa);
    const n: ?u32 = null;
    const want = [_]Row{
        .{ .text = 20, .codec = n },    .{ .text = 7, .codec = n },     .{ .text = 1, .codec = n },
        .{ .text = 22, .codec = 2154 }, .{ .text = 22, .codec = 2156 }, .{ .text = 22, .codec = 2050 },
        .{ .text = 22, .codec = 2157 }, .{ .text = 22, .codec = 3061 }, .{ .text = 23, .codec = 2148 },
        .{ .text = 30, .codec = 2148 }, .{ .text = 31, .codec = 2148 }, .{ .text = 24, .codec = 2148 },
        .{ .text = 22, .codec = 2149 },
    };
    try std.testing.expectEqualSlices(Row, &want, p.rows);

    // `auto`: nothink, no language row; a dialect speaker in `auto` takes its dialect.
    var auto = try Prompt.build(gpa, &small, .{ .role = role, .text = &text, .speaker = "ryan" });
    defer auto.deinit(gpa);
    try std.testing.expectEqual(want.len - 1, auto.rows.len);
    try std.testing.expectEqual(@as(?u32, 2155), auto.rows[3].codec);
    var dialect = try Prompt.build(gpa, &small, .{ .role = role, .text = &text, .speaker = "eric", .language = "chinese" });
    defer dialect.deinit(gpa);
    try std.testing.expectEqual(@as(?u32, 2062), dialect.rows[5].codec);

    const instruct = [_]u32{ 20, 9, 1, 10, 21, 1 };
    try std.testing.expectError(error.InstructionUnsupported, Prompt.build(gpa, &small, .{ .role = role, .text = &text, .instruct = &instruct, .speaker = "ryan" }));
    try std.testing.expectError(error.UnknownSpeaker, Prompt.build(gpa, &small, .{ .role = role, .text = &text, .speaker = "nobody" }));
    try std.testing.expectError(error.UnknownLanguage, Prompt.build(gpa, &small, .{ .role = role, .text = &text, .speaker = "ryan", .language = "klingon" }));
    try std.testing.expectError(error.EmptyText, Prompt.build(gpa, &small, .{ .role = role, .text = &.{}, .speaker = "ryan" }));

    var large = try testConfig(gpa, "1b7", 16);
    defer large.deinit();
    var with = try Prompt.build(gpa, &large, .{ .role = role, .text = &text, .instruct = &instruct, .speaker = "ryan", .language = "english" });
    defer with.deinit(gpa);
    try std.testing.expectEqual(instruct.len + want.len, with.rows.len);
    for (instruct, with.rows[0..instruct.len]) |id, row| try std.testing.expectEqual(Row{ .text = id, .codec = null }, row);
    try std.testing.expectEqualSlices(Row, &want, with.rows[instruct.len..]);
}

/// The names and shapes `bind` asks for; freed with `freeSpecs`.
fn specs(gpa: Allocator, config: *const Config) !std.ArrayList(Spec) {
    var list: std.ArrayList(Spec) = .empty;
    errdefer freeSpecs(gpa, &list);
    var binder: Binder = .{ .gpa = gpa, .checkpoint = null, .record = &list, .diagnostic = null };
    defer binder.deinit();
    var binding = try bindWith(&binder, config);
    binding.deinit(gpa);
    return list;
}

fn freeSpecs(gpa: Allocator, list: *std.ArrayList(Spec)) void {
    for (list.items) |s| {
        gpa.free(s.name);
        gpa.free(s.shape);
    }
    list.deinit(gpa);
}

pub const TestChange = struct { skip: ?[]const u8 = null, reshape: ?Spec = null, extra: ?Spec = null };

/// Writes `model.safetensors` into `dir`: every tensor `bind` asks for under
/// `config`, BF16 values in [−0.5, 0.5) (norm weights near 1), then `change`.
pub fn writeTestCheckpoint(gpa: Allocator, io: std.Io, dir: std.Io.Dir, config: *const Config, change: TestChange) !void {
    var list = try specs(gpa, config);
    defer freeSpecs(gpa, &list);
    if (change.extra) |e| try list.append(gpa, .{ .name = try gpa.dupe(u8, e.name), .shape = try gpa.dupe(u64, e.shape) });
    var header: std.Io.Writer.Allocating = .init(gpa);
    defer header.deinit();
    var data: std.ArrayList(u8) = .empty;
    defer data.deinit(gpa);
    var prng = std.Random.DefaultPrng.init(17);
    try header.writer.writeByte('{');
    var first = true;
    for (list.items) |s| {
        if (change.skip) |name| if (std.mem.eql(u8, s.name, name)) continue;
        var shape = s.shape;
        if (change.reshape) |r| if (std.mem.eql(u8, s.name, r.name)) {
            shape = r.shape;
        };
        var elements: u64 = 1;
        for (shape) |x| elements *= x;
        const start = data.items.len;
        const norm = std.mem.endsWith(u8, s.name, "norm.weight");
        for (0..elements) |_| {
            const value: f32 = prng.random().float(f32) - 0.5 + if (norm) @as(f32, 1) else 0;
            const bits: u32 = @bitCast(value);
            try data.appendSlice(gpa, std.mem.asBytes(&@as(u16, @truncate(bits >> 16))));
        }
        if (!first) try header.writer.writeByte(',');
        first = false;
        try header.writer.print("\"{s}\":{{\"dtype\":\"BF16\",\"shape\":[", .{s.name});
        for (shape, 0..) |x, k| try header.writer.print("{s}{d}", .{ if (k == 0) "" else ",", x });
        try header.writer.print("],\"data_offsets\":[{d},{d}]}}", .{ start, data.items.len });
    }
    try header.writer.writeByte('}');
    const file = try dir.createFile(io, "model.safetensors", .{});
    defer file.close(io);
    var length: [8]u8 = undefined;
    std.mem.writeInt(u64, &length, header.written().len, .little);
    try file.writeStreamingAll(io, &length);
    try file.writeStreamingAll(io, header.written());
    try file.writeStreamingAll(io, data.items);
}

/// A test checkpoint in a temporary directory, open.
pub const TestFile = struct {
    tmp: std.testing.TmpDir,
    checkpoint: safetensors.Checkpoint,

    pub fn open(io: std.Io, config: *const Config, change: TestChange) !TestFile {
        const gpa = std.testing.allocator;
        var tmp = std.testing.tmpDir(.{});
        errdefer tmp.cleanup();
        try writeTestCheckpoint(gpa, io, tmp.dir, config, change);
        const path = try tmp.dir.realPathFileAlloc(io, ".", gpa);
        defer gpa.free(path);
        return .{ .tmp = tmp, .checkpoint = try safetensors.Checkpoint.open(gpa, io, path, .{}) };
    }

    pub fn close(self: *TestFile, io: std.Io) void {
        self.checkpoint.deinit(io);
        self.tmp.cleanup();
    }
};

/// The published checkpoints' tensor counts, from the binder alone.
fn publishedCount(gpa: Allocator, size: []const u8, talker_hidden: usize, kind: []const u8) !usize {
    var config = try testConfigOf(gpa, size, talker_hidden, kind);
    defer config.deinit();
    config.talker.layers = 28;
    config.predictor.layers = 5;
    var list = try specs(gpa, &config);
    defer freeSpecs(gpa, &list);
    return list.items.len;
}

test "402 tensors on the 0.6B, 404 on the 1.7B, 76 more on a Base; bind refuses others by name" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    try std.testing.expectEqual(@as(usize, 402), try publishedCount(gpa, "0b6", 8, "custom_voice"));
    try std.testing.expectEqual(@as(usize, 404), try publishedCount(gpa, "1b7", 16, "custom_voice"));
    try std.testing.expectEqual(@as(usize, 478), try publishedCount(gpa, "0b6", 8, "base"));
    try std.testing.expectEqual(@as(usize, 480), try publishedCount(gpa, "1b7", 16, "base"));

    var config = try testConfig(gpa, "1b7", 16);
    defer config.deinit();
    {
        var file = try TestFile.open(io, &config, .{});
        defer file.close(io);
        var binding = try bind(gpa, &file.checkpoint, &config, null);
        defer binding.deinit(gpa);
        try std.testing.expectEqual(@as(u32, @intCast(file.checkpoint.tensorCount())), binding.tensors);
        try std.testing.expect(binding.projection != null);
    }
    const Case = struct { change: TestChange, err: Error, name: []const u8 };
    const cases = [_]Case{
        .{ .change = .{ .skip = "talker.code_predictor.small_to_mtp_projection.bias" }, .err = error.MissingTensor, .name = "talker.code_predictor.small_to_mtp_projection.bias" },
        .{ .change = .{ .reshape = .{ .name = "talker.model.layers.1.self_attn.k_proj.weight", .shape = &.{ 16, 16 } } }, .err = error.TensorShapeMismatch, .name = "talker.model.layers.1.self_attn.k_proj.weight" },
        .{ .change = .{ .extra = .{ .name = "talker.code_predictor.model.layers.1.mlp.up_proj.weight", .shape = &.{ 12, 8 } } }, .err = error.UnexpectedTensor, .name = "talker.code_predictor.model.layers.1.mlp.up_proj.weight" },
        .{ .change = .{ .extra = .{ .name = "speaker_encoder.fc.weight", .shape = &.{4} } }, .err = error.UnexpectedTensor, .name = "speaker_encoder.fc.weight" },
    };
    for (cases) |case| {
        var file = try TestFile.open(io, &config, case.change);
        defer file.close(io);
        var diagnostic: Diagnostic = .{};
        try std.testing.expectError(case.err, bind(gpa, &file.checkpoint, &config, &diagnostic));
        try std.testing.expectEqualStrings(case.name, diagnostic.name());
    }

    var base = try testConfigOf(gpa, "0b6", 8, "base");
    defer base.deinit();
    {
        var file = try TestFile.open(io, &base, .{});
        defer file.close(io);
        var binding = try bind(gpa, &file.checkpoint, &base, null);
        defer binding.deinit(gpa);
        try std.testing.expectEqual(@as(u32, @intCast(file.checkpoint.tensorCount())), binding.tensors);
        const e = binding.speaker_encoder.?;
        try std.testing.expectEqual(@as(usize, 3), e.blocks.len);
        try std.testing.expectEqual(@as(usize, 7), e.blocks[2].res2net.len);
        try std.testing.expectEqualSlices(u64, &.{ 8, 96, 1 }, e.fc.weight.tensor.shape);
    }
    // Every failure point unwinds the partial blocks (the testing allocator checks).
    const base_cases = [_]Case{
        .{ .change = .{ .skip = "speaker_encoder.fc.bias" }, .err = error.MissingTensor, .name = "speaker_encoder.fc.bias" },
        .{ .change = .{ .skip = "speaker_encoder.blocks.2.res2net_block.blocks.4.conv.weight" }, .err = error.MissingTensor, .name = "speaker_encoder.blocks.2.res2net_block.blocks.4.conv.weight" },
        .{ .change = .{ .reshape = .{ .name = "speaker_encoder.blocks.1.se_block.conv2.weight", .shape = &.{ 16, 4 } } }, .err = error.TensorShapeMismatch, .name = "speaker_encoder.blocks.1.se_block.conv2.weight" },
        .{ .change = .{ .extra = .{ .name = "speaker_encoder.blocks.4.conv.weight", .shape = &.{4} } }, .err = error.UnexpectedTensor, .name = "speaker_encoder.blocks.4.conv.weight" },
    };
    for (base_cases) |case| {
        var file = try TestFile.open(io, &base, case.change);
        defer file.close(io);
        var diagnostic: Diagnostic = .{};
        try std.testing.expectError(case.err, bind(gpa, &file.checkpoint, &base, &diagnostic));
        try std.testing.expectEqualStrings(case.name, diagnostic.name());
    }
}
