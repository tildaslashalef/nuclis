//! The index: every artifact nuclis can pull and verify by name.
//!
//! One table, `artifacts`, of one type. *Installed* is what sidecars say is
//! under `<root>/models`; *known* is this table, consulted to pull, to
//! verify, and to supply defaults (docs/models/catalogue.md). A file is
//! *runnable* when its architecture has an adapter and *supported* only when
//! it is a `File` here with a pinned commit and SHA-256. A `legacy` artifact
//! still pulls, verifies, and runs, but `config init` never writes it, the
//! index view hides it without `--all`, and it has no gates.
//!
//! Entries are named after the model, not the quantization, so a re-pin
//! edits a row instead of renaming it in every configuration file. Facts are
//! copied from the Hub as `nuclis model inspect` reads them; status reads
//! sidecars only, never hashes a file, so a listing is instant.
const std = @import("std");
const inference = @import("inference");
const model = @import("model.zig");
const config = @import("config.zig");
const Allocator = std.mem.Allocator;

pub const Role = model.Role;

/// Which prompt profile (chat template, sampling defaults) a checkpoint uses.
/// It is the checkpoint's, not the user's: forced at open, and named by
/// `config show`.
pub const Profile = inference.profiles.Profile;

/// What runs an artifact: the text engine, `nuclis decide`, `nuclis embed`,
/// `nuclis speak`, or `nuclis transcribe`. Registry files store the tag, so
/// `generation` keeps its name; `model ls` prints it as "chat".
pub const ModelKind = enum { generation, decision, embedding, speech, transcription };

/// Whether the project still looks after an artifact.
pub const Lifecycle = enum { current, legacy };

/// How a speech model picks its voice (`config.json`'s `tts_model_type`).
pub const SpeechMode = enum {
    /// CustomVoice: one of its speakers.
    speakers,
    /// Base: a saved voice, cloned.
    clone,
    /// VoiceDesign: the instruction describes it.
    design,
};

/// A draft source's measured verdict: on or off by default, and the draft
/// length the record chose.
pub const Verdict = struct { speculative: bool, draft_length: usize };

/// One pinned file of an artifact. `path` is where it lands under the
/// artifact's repository directory and, without `remote`, its Hub name too.
pub const File = struct {
    role: Role,
    /// Names one of several files of a role (`mtp`, `dflash2`); null alone.
    variant: ?[]const u8 = null,
    path: []const u8,
    /// Set when the file comes from another repository (`revision` with it).
    repo: ?[]const u8 = null,
    revision: ?[]const u8 = null,
    /// The Hub name in `repo` when it lands at `path` beside the parent
    /// instead of under its own repository directory (the DFlash2 drafter).
    remote: ?[]const u8 = null,
    size: u64,
    sha256: []const u8,
    /// What it is loaded for, as `nuclis model ls` says it.
    used_for: []const u8 = "",
    /// Plain git files pulled beside this file (configuration, tokenizer):
    /// the Hub gives their blob ids, not SHA-256, so they are names here.
    support: []const []const u8 = &.{},
    /// A draft file's measured verdict.
    verdict: ?Verdict = null,
};

/// The options a model runs with when nothing overrides them. Each
/// non-trivial value names the record that chose it.
pub const Defaults = struct {
    /// The prompt profile forced at open (lets an entry pin a protocol onto a
    /// file whose own template digest is not pinned).
    profile: ?Profile = null,
    /// The draft variant used when several exist.
    draft: ?[]const u8 = null,
    /// The agent's effort and per-step reasoning cap; null leaves the
    /// global `agent.*` value.
    think: ?inference.profiles.Effort = null,
    thinking_budget: ?usize = null,
};

/// The per-kind facts that are real.
pub const Traits = struct {
    /// `<architecture>.block_count` where one architecture spans checkpoints
    /// with different verdicts (Gemma 4): `discover` matches on it.
    block_count: ?u64 = null,
    /// Its projector carries an audio encoder: the model hears clips.
    hears: bool = false,
    /// A speech model's voice selection, speakers, and whether it performs
    /// a delivery instruction.
    mode: SpeechMode = .speakers,
    speakers: []const []const u8 = &.{},
    instructions: bool = false,
};

pub const Artifact = struct {
    name: []const u8,
    /// The checkpoint as a person names it ("Qwen3.8 27B").
    title: []const u8,
    kind: ModelKind,
    status: Lifecycle = .current,
    repo: []const u8,
    /// The 40-character commit every digest below was read at.
    revision: []const u8,
    /// `files[0]` is the main file.
    files: []const File,
    /// `general.architecture` of the main file (or its family's name).
    architecture: []const u8,
    quantization: []const u8,
    defaults: Defaults = .{},
    traits: Traits = .{},

    pub fn main(self: *const Artifact) *const File {
        return &self.files[0];
    }

    /// The first file of `role`; for `draft`, the default variant.
    pub fn companion(self: *const Artifact, role: Role) ?*const File {
        if (role == .draft) return self.draft(null);
        for (self.files) |*f| if (f.role == role) return f;
        return null;
    }

    /// The draft file of `variant`, or the default draft when null.
    pub fn draft(self: *const Artifact, variant: ?[]const u8) ?*const File {
        const wanted: ?[]const u8 = if (variant) |v| v else self.defaults.draft;
        var first: ?*const File = null;
        for (self.files) |*f| if (f.role == .draft) {
            if (first == null) first = f;
            if (wanted) |w| if (f.variant) |v| if (std.mem.eql(u8, v, w)) return f;
        };
        return if (variant != null) null else first;
    }

    /// The default draft's verdict: off when there is no drafter.
    pub fn verdict(self: *const Artifact) Verdict {
        return if (self.draft(null)) |d| d.verdict orelse .{ .speculative = false, .draft_length = 4 } else .{ .speculative = false, .draft_length = 4 };
    }

    pub fn origin(self: *const Artifact, f: *const File) []const u8 {
        return f.repo orelse self.repo;
    }

    pub fn originRevision(self: *const Artifact, f: *const File) []const u8 {
        return f.revision orelse self.revision;
    }

    pub fn hubName(f: *const File) []const u8 {
        return f.remote orelse f.path;
    }

    /// The repository directory a file lands in: its origin's, unless it
    /// names `remote` and lands beside the parent.
    pub fn directoryOf(self: *const Artifact, f: *const File) []const u8 {
        return if (f.remote != null) self.repo else f.repo orelse self.repo;
    }

    /// The bytes a plain pull fetches: the main file and those pulled with
    /// it (codec, backbone), without optional companions.
    pub fn totalSize(self: *const Artifact) u64 {
        var total: u64 = 0;
        for (self.files) |f| if (pulledWith(f.role)) {
            total += f.size;
        };
        return total;
    }

    /// The artifact's draft variants, in table order.
    pub fn draftCount(self: *const Artifact) usize {
        var n: usize = 0;
        for (self.files) |f| n += @intFromBool(f.role == .draft);
        return n;
    }
};

/// Roles a pull always takes with the main file.
pub fn pulledWith(role: Role) bool {
    return switch (role) {
        .main, .codec, .backbone => true,
        else => false,
    };
}

const qwen3_tts_speakers = [_][]const u8{ "aiden", "dylan", "eric", "ono_anna", "ryan", "serena", "sohee", "uncle_fu", "vivian" };
const qwen3_tts_support = [_][]const u8{ "config.json", "generation_config.json", "merges.txt", "preprocessor_config.json", "tokenizer_config.json", "vocab.json" };
const qwen3_tts_codec_support = [_][]const u8{ "speech_tokenizer/config.json", "speech_tokenizer/configuration.json", "speech_tokenizer/preprocessor_config.json" };
const qwen3_tts_codec_sha256 = "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258";
const qwen3_tts_codec_size = 682_293_092;

fn qwen3TtsCodec() File {
    return .{ .role = .codec, .path = "speech_tokenizer/model.safetensors", .size = qwen3_tts_codec_size, .sha256 = qwen3_tts_codec_sha256, .used_for = "the speech codec (pulled with it)", .support = &qwen3_tts_codec_support };
}

pub const artifacts = [_]Artifact{
    .{
        .name = "qwen3.8-27b",
        .title = "Qwen3.8 27B",
        .kind = .generation,
        .repo = "unsloth/Qwen3.8-27B-GGUF",
        .revision = "ce331efc48d5722b4b8041f72793e530eae18d9b",
        .architecture = "qwen35",
        .quantization = "UD-Q4_K_M",
        .defaults = .{ .profile = .qwen38, .draft = "mtp", .think = .low, .thinking_budget = 1024 },
        .files = &.{
            .{ .role = .main, .path = "Qwen3.8-27B-UD-Q4_K_M.gguf", .size = 16_564_259_744, .sha256 = "1bce504dfecd84153666928a41d8a66706f0158d3a317bc5196b72819d7dc529" },
            .{ .role = .mmproj, .path = "mmproj-BF16.gguf", .size = 931_146_432, .sha256 = "83ee4f4f205fa514161778c41df1ea14144faa0f713510893b63c2395f5c2d53", .used_for = "image input" },
            // Prose 1.25-1.57x from 512 to 32K, code 1.92-1.95x at draft 7 on the
            // first pin; 1.42-1.61x and code 1.65x on this file
            // (docs/benchmarks/history.md § The re-made Qwen3.8 file (2026-11-12)).
            .{ .role = .draft, .variant = "mtp", .path = "MTP/mtp-Qwen3.8-27B-Q4_0.gguf", .size = 1_369_590_656, .sha256 = "50d9ce5a6da381bbcfb31061cf73df94a90e6faf8efeddee379a9cb8f1501c6e", .used_for = "speculative decoding", .verdict = .{ .speculative = true, .draft_length = 7 } },
            // z-lab's block drafter, placed beside the model it drafts for:
            // prose 1.366x at draft 7 and p_min 0.3, 1.012x MTP's on the first
            // pin; 1.53-1.63x and code 1.69x on this file at p_min 0.7
            // (docs/benchmarks/history.md § The re-made Qwen3.8 file (2026-11-12)).
            .{ .role = .draft, .variant = "dflash2", .path = "MTP/Qwen3.8-27B-DFlash2-Q8_0.gguf", .repo = "z-lab/Qwen3.8-27B-DFlash2-GGUF", .revision = "2d9571f8ce46e151f61c6499c99dee6079e1d610", .remote = "Qwen3.8-27B-DFlash2-Q8_0.gguf", .size = 2_056_414_816, .sha256 = "c18e800daedc59ca68fd13b6a856d795746af6d399a9279ac6a277d1d422f87e", .used_for = "speculative decoding (a DFlash2 drafter)", .verdict = .{ .speculative = true, .draft_length = 7 } },
        },
    },
    // Google's quantization-aware-trained 12B: every matrix Q4_0, the
    // encoding it was trained for. Legacy.
    .{
        .name = "gemma-4-12b-qat",
        .title = "Gemma 4 12B QAT",
        .kind = .generation,
        .status = .legacy,
        .repo = "unsloth/gemma-4-12B-it-qat-GGUF",
        .revision = "980b060c40a8539ac159e0501a3e0f66a6365af3",
        .architecture = "gemma4",
        // The Hub name says UD-Q4_K_XL; the header says every matrix is Q4_0.
        .quantization = "Q4_0 (QAT)",
        .traits = .{ .block_count = inference.models.gemma4.config_12b.layer_count },
        .defaults = .{ .profile = .gemma4, .think = .low, .thinking_budget = 1024 },
        .files = &.{
            .{ .role = .main, .path = "gemma-4-12B-it-qat-UD-Q4_K_XL.gguf", .size = 6_716_356_800, .sha256 = "90fd44e29e0d7cffeb0fd00dc73cfdab9ed0b0e95306ecf7821ea634c940c370" },
            .{ .role = .mmproj, .path = "mmproj-BF16.gguf", .size = 175_115_840, .sha256 = "dcb8103adad042b1bf99df767aaf34eb37c5a73a4a2f0417e4d7ba557e91664f", .used_for = "image input" },
            // Prose 1.34x geometric mean, 0.98x at 32K at draft 5
            // (docs/benchmarks/history.md § The re-priced speculative verdicts (2026-10-01)).
            .{ .role = .draft, .path = "mtp-gemma-4-12B-it.gguf", .size = 253_708_800, .sha256 = "fcb35dea42c71333db904cee11baac525c9ef872818ee3753f6cb156f3c6f4f6", .used_for = "speculative decoding", .verdict = .{ .speculative = true, .draft_length = 5 } },
        },
    },
    // The mixture-of-experts sibling, QAT file only: the K-quant release
    // stores expert down-projections as Q5_1, which no kernel executes.
    .{
        .name = "gemma-4-26b-a4b",
        .title = "Gemma 4 26B-A4B",
        .kind = .generation,
        .repo = "unsloth/gemma-4-26B-A4B-it-qat-GGUF",
        .revision = "7b92b5b28818151e8669af2e45e88d6086f490dd",
        .architecture = "gemma4",
        .quantization = "Q4_0 (QAT)",
        .traits = .{ .block_count = inference.models.gemma4.config_26b_a4b.layer_count },
        .defaults = .{ .profile = .gemma4, .think = .low, .thinking_budget = 1024 },
        .files = &.{
            .{ .role = .main, .path = "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf", .size = 14_249_047_104, .sha256 = "a7c5bc715f5ff8e99a3e8901ce7d2b42b402c669bf24f7c5250747633d0f5891" },
            .{ .role = .mmproj, .path = "mmproj-BF16.gguf", .size = 1_194_828_256, .sha256 = "7b06953ccdbe8cf363f47841a7afaacd2b1c2ff9a8d6b426fdec7521a6878744", .used_for = "image input" },
            // 0.63-0.76x on prose at drafts 2-7: rows route to their own
            // experts; off (docs/benchmarks/history.md § The re-priced speculative verdicts (2026-10-01)).
            .{ .role = .draft, .path = "MTP/mtp-gemma-4-26B-A4B-it-Q4_0.gguf", .size = 251_939_328, .sha256 = "7272d97595f0d4c74bd7b623492b7dbdaafd8b7c72f329a8270ba4eca68f768a", .used_for = "speculative decoding", .verdict = .{ .speculative = false, .draft_length = 4 } },
        },
    },
    // The on-device E4B, QAT file only: per-layer embeddings, shared KV
    // layers, its own template revision, and an audio encoder.
    .{
        .name = "gemma-4-e4b-qat",
        .title = "Gemma 4 E4B QAT",
        .kind = .generation,
        .repo = "unsloth/gemma-4-E4B-it-qat-GGUF",
        .revision = "8c5a9e4fd5482e2be20fe0bf013b4c262a8f4265",
        .architecture = "gemma4",
        .quantization = "Q4_0 (QAT)",
        .traits = .{ .block_count = inference.models.gemma4.config_e4b.layer_count, .hears = true },
        .defaults = .{ .profile = .gemma4_e, .think = .low, .thinking_budget = 1024 },
        .files = &.{
            .{ .role = .main, .path = "gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf", .size = 4_215_695_776, .sha256 = "df0fd4ee07072c607c29a0a1cb4f98918426cca12f45a2776bdd6ee6d09a4de3" },
            .{ .role = .mmproj, .path = "mmproj-BF16.gguf", .size = 991_552_320, .sha256 = "7c9bafa27f82d658eda805c1d82ef62bb0368e1ff75f64f77de58ad318beaaf9", .used_for = "image and audio input" },
            // Prose 1.47x geometric mean, 0.99x at 32K at draft 6
            // (docs/benchmarks/history.md § The re-priced speculative verdicts (2026-10-01)).
            .{ .role = .draft, .path = "MTP/mtp-gemma-4-E4B-it-Q4_0.gguf", .size = 59_678_016, .sha256 = "423074e537504b4f9ec5eafed5c639fac82c96631626efccacdd3c4039b20605", .used_for = "speculative decoding", .verdict = .{ .speculative = true, .draft_length = 6 } },
        },
    },
    // Meta's dense agentic model with its DFlash drafter. Legacy: no gate
    // checks it.
    .{
        .name = "muse-glimmer-30b",
        .title = "Muse Glimmer 30B",
        .kind = .generation,
        .status = .legacy,
        .repo = "unsloth/Muse-Glimmer-30B-GGUF",
        .revision = "faa5b025c584459c13febfa5c59883516710ae39",
        .architecture = "muse-glimmer",
        .quantization = "UD-Q4_K_XL",
        .defaults = .{ .profile = .muse_glimmer, .think = .low, .thinking_budget = 1024 },
        .files = &.{
            .{ .role = .main, .path = "Muse-Glimmer-30B-UD-Q4_K_XL.gguf", .size = 15_878_222_368, .sha256 = "82bece304887a313ece08400bc030f6066c7bff5b906b0cd40308ec8a409fd38" },
            .{ .role = .mmproj, .path = "mmproj-kquant.gguf", .size = 1_400_328_928, .sha256 = "f48b452316f9b213758e8659444029b961a24a07f99a1abb2a9f88b06f7c00c6", .used_for = "image input" },
            // Prose 1.27x geometric mean, 1.10x at 32K at draft 6
            // (docs/benchmarks/history.md § The re-priced speculative verdicts (2026-10-01)).
            .{ .role = .draft, .path = "dflash-kquant.gguf", .size = 1_631_205_312, .sha256 = "27d9a805fa29b943cfb6ad4843367cd4eaaaf06bd452d8cc3e00a2cd18a677bc", .used_for = "speculative decoding (a DFlash drafter)", .verdict = .{ .speculative = true, .draft_length = 6 } },
        },
    },
    // Laya's English root checkpoint, a ModernBERT-large encoder with a
    // typed-decision head.
    .{
        .name = "laya",
        .title = "Laya",
        .kind = .decision,
        .repo = "convaiinnovations/laya",
        .revision = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
        .architecture = "modernbert",
        .quantization = "F16",
        .files = &.{
            .{ .role = .main, .path = "model.safetensors", .size = 842_609_210, .sha256 = "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c", .support = &.{ "encoder/config.json", "rl_agent_config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json" } },
        },
    },
    // The multilingual checkpoint beside it, an mmBERT-base encoder with the
    // same head; the caller picks it, nothing routes by language.
    .{
        .name = "laya-multilingual",
        .title = "Laya multilingual",
        .kind = .decision,
        .repo = "convaiinnovations/laya",
        .revision = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
        .architecture = "modernbert",
        .quantization = "F16",
        .files = &.{
            .{ .role = .main, .path = "multilingual/model.safetensors", .size = 643_835_514, .sha256 = "9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204", .support = &.{ "multilingual/encoder/config.json", "multilingual/rl_agent_config.json", "multilingual/tokenizer/tokenizer.json", "multilingual/tokenizer/tokenizer_config.json" } },
        },
    },
    // Cloudflare's clef-flash: the joint schema head and support files, the
    // Qwen3.5-9B backbone as bartowski's Q6_K GGUF, and its BF16 projector.
    .{
        .name = "clef-flash",
        .title = "Cloudflare clef-flash",
        .kind = .decision,
        .repo = "Cloudflare/clef-flash",
        .revision = "17f0b0ad64efb65d273590632833508766b2aae6",
        .architecture = "qwen35",
        .quantization = "Q6_K",
        .files = &.{
            .{ .role = .main, .path = "joint_head.safetensors", .size = 243_538_016, .sha256 = "19cdcec8c81dc9212be320fff47462ab342fbc1278be4368fb3da71241cf5ba0", .support = &.{ "config.json", "joint_head_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "processor_config.json", "generation_config.json" } },
            .{ .role = .backbone, .path = "Cloudflare_clef-flash-Q6_K.gguf", .repo = "bartowski/Cloudflare_clef-flash-GGUF", .revision = "d7f376ea88c05e7bb1014dd5351a93df9dd8029e", .size = 7_793_714_496, .sha256 = "b80f7cfb803de25aade15853bab331e9f030a9ee64dee7bc91ed8c2b07312423", .used_for = "the backbone (pulled with it)" },
            .{ .role = .mmproj, .path = "mmproj-Cloudflare_clef-flash-bf16.gguf", .repo = "bartowski/Cloudflare_clef-flash-GGUF", .revision = "d7f376ea88c05e7bb1014dd5351a93df9dd8029e", .size = 921_704_928, .sha256 = "3c45b34aee6f353a0d41d6b96ba712a498a17a82f0a6152bf68a5021f9652c0f", .used_for = "image input" },
        },
    },
    // Google's EmbeddingGemma 2 as unsloth's Q8_0 conversion with the BF16
    // projector (docs/models/embeddinggemma.md § Files and encodings).
    .{
        .name = "embeddinggemma-2",
        .title = "EmbeddingGemma 2",
        .kind = .embedding,
        .repo = "unsloth/embeddinggemma-2-GGUF",
        .revision = "031f0d4b35536f69ab3509d4893c923264fcf253",
        .architecture = inference.embed.architecture,
        .quantization = "Q8_0",
        .files = &.{
            .{ .role = .main, .path = "embeddinggemma-2-Q8_0.gguf", .size = 309_855_520, .sha256 = "6f1bd4ac6c5df7444f9cca7ca36cafe6cfa34cd6f49fefb1e0b4be8143aed8bc" },
            .{ .role = .mmproj, .path = "mmproj-BF16.gguf", .size = 982_074_880, .sha256 = "995aaa56e88b9b631f651861d659b728a625ccc02a238be37cff56cf11dc0032", .used_for = "image and audio input" },
        },
    },
    // Qwen3-TTS 12Hz, as published (BF16 talker, F32 codec shared by all
    // five): docs/models/qwen3-tts.md. Only the 1.7B CustomVoice and the
    // VoiceDesign perform an instruction.
    .{
        .name = "qwen3-tts-1.7b-customvoice",
        .title = "Qwen3-TTS 1.7B CustomVoice",
        .kind = .speech,
        .repo = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        .revision = "0c0e3051f131929182e2c023b9537f8b1c68adfe",
        .architecture = "qwen3_tts",
        .quantization = "BF16",
        .traits = .{ .speakers = &qwen3_tts_speakers, .instructions = true },
        .files = &.{ .{ .role = .main, .path = "model.safetensors", .size = 3_833_402_552, .sha256 = "38b1d5971bdbd982b561cccec982669a53b0537c3cf5e9bd4778ed07bb2f5137", .support = &qwen3_tts_support }, qwen3TtsCodec() },
    },
    .{
        .name = "qwen3-tts-0.6b-customvoice",
        .title = "Qwen3-TTS 0.6B CustomVoice",
        .kind = .speech,
        .repo = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        .revision = "85e237c12c027371202489a0ec509ded67b5e4b5",
        .architecture = "qwen3_tts",
        .quantization = "BF16",
        .traits = .{ .speakers = &qwen3_tts_speakers },
        .files = &.{ .{ .role = .main, .path = "model.safetensors", .size = 1_811_626_576, .sha256 = "bc3c7e785eb961179c25450d1acff03f839e0002f2f3a5aeb67b5735c0fa2adb", .support = &qwen3_tts_support }, qwen3TtsCodec() },
    },
    // The Base: no speakers, a voice cloned from a saved clip.
    .{
        .name = "qwen3-tts-1.7b-base",
        .title = "Qwen3-TTS 1.7B Base",
        .kind = .speech,
        .repo = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        .revision = "fd4b254389122332181a7c3db7f27e918eec64e3",
        .architecture = "qwen3_tts",
        .quantization = "BF16",
        .traits = .{ .mode = .clone },
        .files = &.{ .{ .role = .main, .path = "model.safetensors", .size = 3_857_413_744, .sha256 = "38fc7fc51c5e776e840414b6fd443962e9411b9654888fd7913e4da643cb857c", .support = &qwen3_tts_support }, qwen3TtsCodec() },
    },
    .{
        .name = "qwen3-tts-0.6b-base",
        .title = "Qwen3-TTS 0.6B Base",
        .kind = .speech,
        .repo = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        .revision = "5d83992436eae1d760afd27aff78a71d676296fc",
        .architecture = "qwen3_tts",
        .quantization = "BF16",
        .traits = .{ .mode = .clone },
        .files = &.{ .{ .role = .main, .path = "model.safetensors", .size = 1_829_344_272, .sha256 = "180b3b10eb1c9f1b4db7806d5475bae3071c0243c299d49926bab1da3b6946f6", .support = &qwen3_tts_support }, qwen3TtsCodec() },
    },
    // VoiceDesign: the 1.7B CustomVoice's networks without speakers, the
    // instruction describing the voice.
    .{
        .name = "qwen3-tts-1.7b-voicedesign",
        .title = "Qwen3-TTS 1.7B VoiceDesign",
        .kind = .speech,
        .repo = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        .revision = "5ecdb67327fd37bb2e042aab12ff7391903235d3",
        .architecture = "qwen3_tts",
        .quantization = "BF16",
        .traits = .{ .mode = .design, .instructions = true },
        .files = &.{ .{ .role = .main, .path = "model.safetensors", .size = 3_833_402_552, .sha256 = "391e8db219f292c515297cdceeb43e4eae67cdde35fa57e79a6a8a532fca0522", .support = &qwen3_tts_support }, qwen3TtsCodec() },
    },
    // Fish Audio's Drama 3 (35B-A3B) as one Q4_K GGUF holding backbone,
    // codebook tower, codec, and tokenizer (docs/models/drama3.md).
    .{
        .name = "drama-3",
        .title = "Fish Audio Drama 3",
        .kind = .speech,
        .repo = "sakasegawa/drama-3-GGUF",
        .revision = "9a01354d04ab6eb521bc9cdfc56ab6824c474af2",
        .architecture = "drama3",
        .quantization = "Q4_K",
        .traits = .{ .mode = .clone },
        .files = &.{.{ .role = .main, .path = "Drama-35B-A3B-v3-Q4_K.gguf", .size = 20_787_264_352, .sha256 = "2a57b14c9b208a2ff833ab7305b920ca8f154ad807004a52d96a86a90a55523e" }},
    },
    // NVIDIA's Nemotron 3.5 ASR streaming 0.6B, F32 as published
    // (docs/models/nemotron-asr.md).
    .{
        .name = "nemotron-asr-0.6b",
        .title = "Nemotron 3.5 ASR Streaming 0.6B",
        .kind = .transcription,
        .repo = "nvidia/nemotron-3.5-asr-streaming-0.6b",
        .revision = "ea30d66debe3740a08b573244286791d423d6b3e",
        .architecture = "nemotron3_5_asr",
        .quantization = "F32",
        .files = &.{.{ .role = .main, .path = "model.safetensors", .size = 2_552_062_944, .sha256 = "9eebdd6590289cb3030f310858f3df93256600a800a3e8200c5993d5f967e174", .support = &.{ "config.json", "generation_config.json", "processor_config.json", "tokenizer.json", "tokenizer_config.json" } }},
    },
};

fn isHex(text: []const u8, len: usize) bool {
    if (text.len != len) return false;
    for (text) |c| if (!std.ascii.isDigit(c) and !(c >= 'a' and c <= 'f')) return false;
    return true;
}

comptime {
    @setEvalBranchQuota(200_000);
    for (&artifacts, 0..) |*a, i| {
        for (artifacts[i + 1 ..]) |other| if (std.mem.eql(u8, a.name, other.name)) @compileError("duplicate artifact name " ++ a.name);
        if (std.mem.indexOfScalar(u8, a.name, '/') != null or std.mem.endsWith(u8, a.name, ".gguf")) @compileError(a.name ++ ": a name never holds / or ends in .gguf");
        if (!isHex(a.revision, 40)) @compileError(a.name ++ ": revision is not 40 hex");
        if (a.files.len == 0 or a.files[0].role != .main) @compileError(a.name ++ ": files[0] must be the main file");
        if (a.kind == .generation and a.defaults.profile == null) @compileError(a.name ++ ": a chat artifact pins its profile");
        const drafts = a.draftCount();
        for (a.files, 0..) |f, j| {
            if (!isHex(f.sha256, 64)) @compileError(a.name ++ ": " ++ f.path ++ " digest is not 64 hex");
            if (j > 0 and f.role == .main) @compileError(a.name ++ ": a second main file");
            if (f.repo != null and (f.revision == null or !isHex(f.revision.?, 40))) @compileError(a.name ++ ": " ++ f.path ++ " names a repository without its 40-hex commit");
            if (f.remote != null and f.repo == null) @compileError(a.name ++ ": " ++ f.path ++ " names a remote without its repository");
            if (f.role == .draft and f.verdict == null) @compileError(a.name ++ ": " ++ f.path ++ " is a draft without its verdict");
            if (f.role == .draft and drafts > 1 and f.variant == null) @compileError(a.name ++ ": several drafts, each needs a variant");
            // At most one file per role, except drafts told apart by variant.
            for (a.files[j + 1 ..]) |g| if (g.role == f.role and f.role != .draft) @compileError(a.name ++ ": two files of one role");
            for (a.files[j + 1 ..]) |g| if (g.role == .draft and f.role == .draft and std.mem.eql(u8, g.variant orelse "", f.variant orelse "")) @compileError(a.name ++ ": two drafts of one variant");
        }
        if (a.defaults.draft) |v| if (a.draft(v) == null) @compileError(a.name ++ ": defaults.draft names no draft variant");
        if (drafts > 1 and a.defaults.draft == null) @compileError(a.name ++ ": several drafts need defaults.draft");
    }
}

/// Exact name match; names never contain `/` or end in `.gguf`, so a path
/// is never mistaken for one.
pub fn find(name: []const u8) ?*const Artifact {
    for (&artifacts) |*a| if (std.mem.eql(u8, a.name, name)) return a;
    return null;
}

/// `find`, of one kind only.
pub fn findKind(name: []const u8, kind: ModelKind) ?*const Artifact {
    const a = find(name) orelse return null;
    return if (a.kind == kind) a else null;
}

/// The embedding artifact whose main file has this digest, so a vector's
/// space names the checkpoint however the caller reached its file.
pub fn embeddingByDigest(sha256: []const u8) ?*const Artifact {
    for (&artifacts) |*a| if (a.kind == .embedding and std.mem.eql(u8, a.main().sha256, sha256)) return a;
    return null;
}

/// The first artifact of `kind`, for messages that name one.
pub fn firstOf(kind: ModelKind) *const Artifact {
    for (&artifacts) |*a| if (a.kind == kind and a.status == .current) return a;
    unreachable;
}

/// The audio catalogue's names before 0.11.0 and the entries they became.
const renamed_audio = [_][2][]const u8{
    .{ "qwen3-tts-1.7b", "qwen3-tts-1.7b-customvoice" },
    .{ "qwen3-tts-0.6b", "qwen3-tts-0.6b-customvoice" },
};

/// The artifact an old audio name became, for a message that refuses it;
/// no name resolves through it.
pub fn renamedAudio(name: []const u8) ?[]const u8 {
    for (renamed_audio) |pair| if (std.mem.eql(u8, pair[0], name)) return pair[1];
    return null;
}

/// What runs `name`: its registry entry's kind, else the index's; anything
/// else (a path, an unknown name) is a text model's.
pub fn kindOf(registry: config.Models, name: []const u8) ModelKind {
    if (registry.find(name)) |e| return e.kind orelse .generation;
    if (find(name)) |a| return a.kind;
    return .generation;
}

/// The command that runs a model of `kind`, for refusals that name it.
pub fn command(kind: ModelKind) []const u8 {
    return switch (kind) {
        .generation => "nuclis generate",
        .decision => "nuclis decide",
        .embedding => "nuclis embed",
        .speech => "nuclis speak",
        .transcription => "nuclis transcribe",
    };
}

/// The kind as a sentence names it ("a speech model").
pub fn kindName(kind: ModelKind) []const u8 {
    return switch (kind) {
        .generation => "language",
        .decision => "decision",
        .embedding => "embedding",
        .speech => "speech",
        .transcription => "transcription",
    };
}

/// The kind as `model ls` groups it.
pub fn groupName(kind: ModelKind) []const u8 {
    return if (kind == .generation) "chat" else @tagName(kind);
}

/// Where a model name that is not an index name points: `path` itself
/// (borrowed) when absolute or `./`- or `../`-prefixed, else `<root>/models/<path>`
/// in `arena`. A name joined under `models/` may not climb out of it with a
/// `..` segment (`PathEscapesModels`, said in `diag`).
pub fn underModels(arena: std.mem.Allocator, root: ?[]const u8, path: []const u8, diag: *config.Diagnostic) error{ MissingHome, PathEscapesModels, OutOfMemory }![]const u8 {
    if (std.fs.path.isAbsolute(path) or std.mem.startsWith(u8, path, "./") or std.mem.startsWith(u8, path, "../")) return path;
    var segments = std.mem.tokenizeAny(u8, path, "/\\");
    while (segments.next()) |segment| if (std.mem.eql(u8, segment, "..")) {
        diag.set("{s}: a name under the models directory may not leave it (give ./ or an absolute path)", .{path});
        return error.PathEscapesModels;
    };
    const dir = root orelse return error.MissingHome;
    return std.fs.path.join(arena, &.{ dir, "models", path });
}

test underModels {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var diag: config.Diagnostic = .{};
    try std.testing.expectEqualStrings("/abs/m", try underModels(arena, "/r", "/abs/m", &diag));
    try std.testing.expectEqualStrings("../m", try underModels(arena, "/r", "../m", &diag));
    try std.testing.expectEqualStrings("/r/models/org/m", try underModels(arena, "/r", "org/m", &diag));
    try std.testing.expectError(error.PathEscapesModels, underModels(arena, "/r", "org/../../etc", &diag));
    try std.testing.expectError(error.PathEscapesModels, underModels(arena, "/r", "..", &diag));
    try std.testing.expectError(error.MissingHome, underModels(arena, null, "m", &diag));
}

/// `<models>/<directory>/<path>`, where `model pull` writes `f`.
pub fn localPath(alloc: Allocator, models_dir: []const u8, a: *const Artifact, f: *const File) ![]u8 {
    return std.fs.path.join(alloc, &.{ models_dir, a.directoryOf(f), f.path });
}

/// The directory a checkpoint opens from: the main file's directory, or the
/// file itself for a single-file GGUF of a non-chat kind (Drama 3).
pub fn checkpointDirectory(alloc: Allocator, models_dir: []const u8, a: *const Artifact) ![]u8 {
    const weights = try localPath(alloc, models_dir, a, a.main());
    if (std.mem.endsWith(u8, a.main().path, ".gguf")) return weights;
    defer alloc.free(weights);
    return alloc.dupe(u8, std.fs.path.dirname(weights).?);
}

/// The widest artifact name, so every listing lines its columns up however
/// the index grows.
pub const name_width = blk: {
    var widest: usize = 0;
    for (&artifacts) |a| widest = @max(widest, a.name.len);
    break :blk widest + 1;
};

/// Pads `text` to `name_width` in `buffer` (which must be at least that
/// wide), so padding happens before any escape sequence.
pub fn padName(buffer: []u8, text: []const u8) []const u8 {
    const written = @min(text.len, buffer.len);
    @memcpy(buffer[0..written], text[0..written]);
    const total = @min(@max(written, name_width), buffer.len);
    @memset(buffer[written..total], ' ');
    return buffer[0..total];
}

/// The first current chat model that hears audio, for a refusal that names
/// where to go instead.
pub fn hearing() ?*const Artifact {
    for (&artifacts) |*a| if (a.kind == .generation and a.status == .current and a.traits.hears) return a;
    return null;
}

/// A file the index pins, by its origin repository and Hub name; `model
/// inspect` compares the Hub's digest against `file.sha256`.
pub const FileMatch = struct { artifact: *const Artifact, file: *const File };

pub fn findFile(repo: []const u8, name: []const u8) ?FileMatch {
    for (&artifacts) |*a| for (a.files) |*f| {
        if (std.mem.eql(u8, a.origin(f), repo) and std.mem.eql(u8, Artifact.hubName(f), name)) return .{ .artifact = a, .file = f };
    };
    return null;
}

/// The pinned file with this size and digest, for `model adopt`.
pub fn findDigest(sha256: []const u8, size: u64) ?FileMatch {
    for (&artifacts) |*a| for (a.files) |*f| {
        if (f.size == size and std.mem.eql(u8, f.sha256, sha256)) return .{ .artifact = a, .file = f };
    };
    return null;
}

/// The index as data, for the readers that import none of nuclis: the gate
/// runner and the bench scripts read `src/catalog.json`, which `make index`
/// writes with this and a test holds equal to the table.
pub fn writeJson(out: *std.Io.Writer) std.Io.Writer.Error!void {
    const JsonFile = struct { role: Role, variant: ?[]const u8, local: []const u8, size: u64, sha256: []const u8, verdict: ?Verdict };
    const JsonArtifact = struct {
        name: []const u8,
        title: []const u8,
        kind: ModelKind,
        status: Lifecycle,
        repo: []const u8,
        revision: []const u8,
        architecture: []const u8,
        quantization: []const u8,
        defaults: struct { profile: ?Profile, draft: ?[]const u8 },
        files: []const JsonFile,
    };
    var w: std.json.Stringify = .{ .writer = out, .options = .{ .whitespace = .indent_1 } };
    try w.beginObject();
    try w.objectField("$comment");
    try w.write("Generated from src/catalog.zig by `make index`; do not edit. docs/models/catalogue.md § The catalogue.");
    try w.objectField("schema_version");
    try w.write(1);
    try w.objectField("artifacts");
    try w.beginArray();
    for (&artifacts) |*a| {
        var files: [8]JsonFile = undefined;
        var local: [8][256]u8 = undefined;
        for (a.files, 0..) |*f, i| files[i] = .{
            .role = f.role,
            .variant = f.variant,
            .local = std.fmt.bufPrint(&local[i], "{s}/{s}", .{ a.directoryOf(f), f.path }) catch unreachable, // comptime-checked lengths
            .size = f.size,
            .sha256 = f.sha256,
            .verdict = f.verdict,
        };
        try w.write(JsonArtifact{
            .name = a.name,
            .title = a.title,
            .kind = a.kind,
            .status = a.status,
            .repo = a.repo,
            .revision = a.revision,
            .architecture = a.architecture,
            .quantization = a.quantization,
            .defaults = .{ .profile = a.defaults.profile, .draft = a.defaults.draft },
            .files = files[0..a.files.len],
        });
    }
    try w.endArray();
    try w.endObject();
    try out.writeByte('\n');
}

comptime {
    for (artifacts) |a| {
        if (a.files.len > 8) @compileError(a.name ++ ": writeJson holds at most 8 files");
        for (a.files) |f| if (a.directoryOf(&f).len + 1 + f.path.len > 256) @compileError(a.name ++ ": a local path over 256 bytes");
    }
}

test "src/catalog.json is the table (make index rewrites it)" {
    var buffer: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer buffer.deinit();
    try writeJson(&buffer.writer);
    std.testing.expectEqualStrings(@embedFile("catalog.json"), buffer.written()) catch |err| {
        std.debug.print("src/catalog.json is stale: run `make index`\n", .{});
        return err;
    };
}

/// From the sidecar alone: `present` when it records the pinned digest and
/// size, `mismatch` when it records something else, `unverified` when the
/// file is there without one (copied in by hand; `model adopt` or a `pull`
/// verifies it), `absent` otherwise.
pub const Status = enum { present, absent, mismatch, unverified };

pub fn status(alloc: Allocator, io: std.Io, path: []const u8, sha256: []const u8, size: u64) !Status {
    std.Io.Dir.cwd().access(io, path, .{}) catch |err| switch (err) {
        error.FileNotFound => return .absent,
        else => return err,
    };
    // The sidecar's strings live for this call only.
    var arena_state = std.heap.ArenaAllocator.init(alloc);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const sidecar = model.readSidecar(arena, io, .cwd(), try model.sidecarPath(arena, path)) catch |err| switch (err) {
        error.InvalidSidecar => return .mismatch,
        else => return err,
    } orelse return .unverified;
    return if (std.mem.eql(u8, sidecar.sha256, sha256) and sidecar.size == size) .present else .mismatch;
}

test "the index: kinds, lifecycle, and the per-kind facts the old tables held" {
    const qwen = find("qwen3.8-27b").?;
    try std.testing.expectEqualStrings("unsloth/Qwen3.8-27B-GGUF", qwen.repo);
    try std.testing.expectEqual(Profile.qwen38, qwen.defaults.profile.?);
    try std.testing.expectEqual(Profile.gemma4_e, find("gemma-4-e4b-qat").?.defaults.profile.?);
    try std.testing.expectEqual(Lifecycle.legacy, find("gemma-4-12b-qat").?.status);
    try std.testing.expectEqual(Lifecycle.legacy, find("muse-glimmer-30b").?.status);
    for ([_][]const u8{ "qwen3.8-27b", "gemma-4-26b-a4b", "gemma-4-e4b-qat", "clef-flash", "laya" }) |name| try std.testing.expectEqual(Lifecycle.current, find(name).?.status);
    // Files outside the index run through discovery, not by name.
    try std.testing.expect(find("gemma-4-12b") == null and find("qwen") == null and find("qwen/Qwen3.8-27B-UD-Q4_K_M.gguf") == null);
    // The measured verdicts: every drafter pays but the mixture-of-experts head.
    try std.testing.expect(qwen.verdict().speculative and qwen.verdict().draft_length == 7);
    try std.testing.expectEqual(@as(usize, 5), find("gemma-4-12b-qat").?.verdict().draft_length);
    try std.testing.expectEqual(@as(usize, 6), find("gemma-4-e4b-qat").?.verdict().draft_length);
    try std.testing.expect(!find("gemma-4-26b-a4b").?.verdict().speculative);
    for (&artifacts) |*a| for (a.files) |f| if (f.verdict) |v| if (v.speculative) try std.testing.expect(v.draft_length >= 1 and v.draft_length <= inference.engine.max_draft_length);
    try std.testing.expectEqualStrings("gemma-4-e4b-qat", hearing().?.name);
    for (&artifacts) |*a| if (a.traits.hears) try std.testing.expect(a.companion(.mmproj) != null);
    // Kinds keep apart: a name is one kind's only.
    try std.testing.expect(findKind("laya", .decision) != null and findKind("laya", .generation) == null);
    try std.testing.expectEqual(ModelKind.transcription, kindOf(.{}, "nemotron-asr-0.6b"));
    try std.testing.expectEqual(ModelKind.speech, kindOf(.{}, "drama-3"));
    try std.testing.expectEqual(ModelKind.generation, kindOf(.{}, "me/finetune"));
    try std.testing.expectEqualStrings("gemma-embedding2", find("embeddinggemma-2").?.architecture);
    try std.testing.expect(embeddingByDigest(find("embeddinggemma-2").?.main().sha256) == find("embeddinggemma-2"));
    // A speech model brings its codec from its own repository; all five
    // Qwen3-TTS checkpoints share one.
    for (&artifacts) |*a| if (a.kind == .speech) {
        const gguf = std.mem.endsWith(u8, a.main().path, ".gguf");
        try std.testing.expectEqual(!gguf, a.companion(.codec) != null);
        try std.testing.expectEqual(std.mem.endsWith(u8, a.name, "-customvoice"), a.traits.speakers.len > 0);
        try std.testing.expectEqual(a.traits.speakers.len > 0, a.traits.mode == .speakers);
    };
    try std.testing.expect(find("qwen3-tts-1.7b-customvoice").?.traits.instructions and !find("qwen3-tts-0.6b-customvoice").?.traits.instructions);
    try std.testing.expectEqual(@as(u64, 3_833_402_552 + 682_293_092), find("qwen3-tts-1.7b-customvoice").?.totalSize());
    try std.testing.expectEqual(@as(u64, 243_538_016 + 7_793_714_496), find("clef-flash").?.totalSize());
    try std.testing.expect(find("qwen3-tts-1.7b") == null);
    try std.testing.expectEqualStrings("qwen3-tts-0.6b-customvoice", renamedAudio("qwen3-tts-0.6b").?);
}

test "drafts: variants, the default, and a file placed beside its parent" {
    const qwen = find("qwen3.8-27b").?;
    try std.testing.expectEqual(@as(usize, 2), qwen.draftCount());
    try std.testing.expectEqualStrings("mtp", qwen.draft(null).?.variant.?);
    try std.testing.expectEqualStrings("mtp", qwen.companion(.draft).?.variant.?);
    const flash = qwen.draft("dflash2").?;
    try std.testing.expect(qwen.draft("eagle") == null);
    try std.testing.expectEqualStrings("z-lab/Qwen3.8-27B-DFlash2-GGUF", qwen.origin(flash));
    try std.testing.expectEqualStrings("2d9571f8ce46e151f61c6499c99dee6079e1d610", qwen.originRevision(flash));
    try std.testing.expectEqualStrings("Qwen3.8-27B-DFlash2-Q8_0.gguf", Artifact.hubName(flash));
    const path = try localPath(std.testing.allocator, "/m", qwen, flash);
    defer std.testing.allocator.free(path);
    try std.testing.expectEqualStrings("/m/unsloth/Qwen3.8-27B-GGUF/MTP/Qwen3.8-27B-DFlash2-Q8_0.gguf", path);
    // A cross-repository file without `remote` keeps its own repository's layout.
    const clef = find("clef-flash").?;
    const backbone = try localPath(std.testing.allocator, "/m", clef, clef.companion(.backbone).?);
    defer std.testing.allocator.free(backbone);
    try std.testing.expectEqualStrings("/m/bartowski/Cloudflare_clef-flash-GGUF/Cloudflare_clef-flash-Q6_K.gguf", backbone);
    // A single drafter needs no variant and is the default.
    try std.testing.expect(find("gemma-4-e4b-qat").?.draft(null).?.variant == null);
    try std.testing.expect(find("laya").?.draft(null) == null);
    // Files are found by origin and Hub name, and by digest.
    try std.testing.expectEqual(flash, findFile("z-lab/Qwen3.8-27B-DFlash2-GGUF", "Qwen3.8-27B-DFlash2-Q8_0.gguf").?.file);
    try std.testing.expect(findFile("unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-DFlash2-Q8_0.gguf") == null);
    try std.testing.expectEqual(Role.main, findFile("unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-Q4_K_M.gguf").?.file.role);
    try std.testing.expectEqualStrings("image input", findFile("unsloth/Qwen3.8-27B-GGUF", "mmproj-BF16.gguf").?.file.used_for);
    try std.testing.expect(findFile("other/repo", "Qwen3.8-27B-UD-Q4_K_M.gguf") == null);
    try std.testing.expectEqual(flash, findDigest(flash.sha256, flash.size).?.file);
    try std.testing.expect(findDigest(flash.sha256, flash.size + 1) == null);
}

test "checkpoint directories: the weights' directory, or a single GGUF" {
    const alloc = std.testing.allocator;
    const multilingual = try checkpointDirectory(alloc, "/m", find("laya-multilingual").?);
    defer alloc.free(multilingual);
    try std.testing.expectEqualStrings("/m/convaiinnovations/laya/multilingual", multilingual);
    const asr = try checkpointDirectory(alloc, "/m", find("nemotron-asr-0.6b").?);
    defer alloc.free(asr);
    try std.testing.expectEqualStrings("/m/nvidia/nemotron-3.5-asr-streaming-0.6b", asr);
    const drama = try checkpointDirectory(alloc, "/m", find("drama-3").?);
    defer alloc.free(drama);
    try std.testing.expectEqualStrings("/m/sakasegawa/drama-3-GGUF/Drama-35B-A3B-v3-Q4_K.gguf", drama);
    // Support files sit in the weights' directory or below it.
    for (&artifacts) |*a| for (a.files) |f| {
        const prefix = if (std.fs.path.dirnamePosix(f.path)) |d| d.len + 1 else 0;
        for (f.support) |name| try std.testing.expect(prefix == 0 or std.mem.startsWith(u8, name, f.path[0..prefix]));
    };
}

test "status derives from the sidecar without hashing" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", alloc);
    defer alloc.free(root);
    const a = find("qwen3.8-27b").?;
    const main_file = a.main();
    const path = try localPath(alloc, root, a, main_file);
    defer alloc.free(path);
    try std.testing.expectEqual(Status.absent, try status(alloc, io, path, main_file.sha256, main_file.size));
    try tmp.dir.createDirPath(io, "unsloth/Qwen3.8-27B-GGUF");
    try tmp.dir.writeFile(io, .{ .sub_path = "unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf", .data = "GGUF" });
    try std.testing.expectEqual(Status.unverified, try status(alloc, io, path, main_file.sha256, main_file.size));
    var sidecar: model.Sidecar = .{ .repo = a.repo, .file = main_file.path, .revision = a.revision, .sha256 = main_file.sha256, .size = main_file.size, .role = .main, .downloaded_at = "2026-09-11T00:00:00Z", .nuclis_version = model.version };
    const sidecar_path = try model.sidecarPath(alloc, path);
    defer alloc.free(sidecar_path);
    try model.writeSidecar(alloc, io, .cwd(), sidecar_path, sidecar);
    try std.testing.expectEqual(Status.present, try status(alloc, io, path, main_file.sha256, main_file.size));
    sidecar.size = 1;
    try model.writeSidecar(alloc, io, .cwd(), sidecar_path, sidecar);
    try std.testing.expectEqual(Status.mismatch, try status(alloc, io, path, main_file.sha256, main_file.size));
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = sidecar_path, .data = "{" });
    try std.testing.expectEqual(Status.mismatch, try status(alloc, io, path, main_file.sha256, main_file.size));
}
