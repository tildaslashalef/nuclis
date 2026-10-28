//! The built-in catalogue of supported artifacts.
//!
//! "Supported" has one source: this table. A file under `models/` is
//! *runnable* when its architecture id has an adapter, *supported* only when
//! it is here with a pinned commit and SHA-256; `generate` runs either, and
//! `model ls` and the docs speak only of the catalogue. Entries are named
//! after the model, not the quantization (`qwen3.8-27b`), so a later
//! quantization change edits the entry rather than renaming it in every
//! configuration file. Companions (vision projector, MTP draft head) carry
//! the unit that will consume them; nothing loads them today, but a pulled
//! model directory is complete and verified before those units exist.
//!
//! Facts here are copied from the Hub as `nuclis model pull` resolved them
//! (docs/models/catalogue.md § Pinned commits and digests); the digest
//! of the main file is also the spec's. Status derivation reads sidecars
//! only, never hashes a file, so a listing is instant.
const std = @import("std");
const inference = @import("inference");
const model = @import("model.zig");
const config = @import("config.zig");
const Allocator = std.mem.Allocator;

pub const Role = model.Role;

/// Which prompt profile (chat template, sampling defaults) an entry's
/// checkpoint uses: the profile registry's enum. The profile is the
/// checkpoint's, not the user's: `config show` names it and fills unset
/// sampling keys from it.
pub const Profile = inference.profiles.Profile;

pub const Companion = struct {
    role: Role,
    file: []const u8,
    size: u64,
    sha256: []const u8,
    /// What it is loaded for, as `nuclis model ls` says it.
    used_for: []const u8,
};

pub const Entry = struct {
    name: []const u8,
    /// The checkpoint as a person names it ("Qwen3.8 27B"), for clients
    /// that list models.
    title: []const u8,
    repo: []const u8,
    file: []const u8,
    /// The 40-character commit every digest below was read at.
    revision: []const u8,
    sha256: []const u8,
    size: u64,
    quantization: []const u8,
    /// `general.architecture` of the main file.
    architecture: []const u8,
    /// `<architecture>.block_count` where one architecture spans checkpoints
    /// with different verdicts (Gemma 4's 12B, 26B-A4B, E4B): `discover`
    /// takes a file's verdict from the entry its layer count matches.
    block_count: ?u64 = null,
    /// The checkpoint's prompt profile, forced at open (which lets an entry
    /// pin a protocol onto a file whose own template digest is not).
    profile: Profile,
    companions: []const Companion,
    /// The measured speculative verdict: whether the switch is on
    /// by default for this entry and the draft length the record chose. The
    /// entry's value wins over the global default (a user's `--speculative`
    /// still overrides it), because the verdict is a property of the family's
    /// drafter, not of the user's preference.
    speculative: bool,
    draft_length: usize,
    /// The agent's effort and per-step reasoning cap for this checkpoint,
    /// written into its registry entry by `config init` so they can be tuned
    /// per model; null leaves the global `agent.*` value. Starting values,
    /// not yet measured per family.
    think: ?inference.profiles.Effort = null,
    thinking_budget: ?usize = null,
    /// Its `mmproj` companion carries an audio encoder: the model hears
    /// clips (`input_audio`, `generate --audio`, a clip dropped in the agent).
    hears: bool = false,

    pub fn companion(self: *const Entry, role: Role) ?*const Companion {
        for (self.companions) |*c| if (c.role == role) return c;
        return null;
    }
};

pub const entries = [_]Entry{
    .{
        .name = "qwen3.8-27b",
        .title = "Qwen3.8 27B",
        .repo = "unsloth/Qwen3.8-27B-GGUF",
        .file = "Qwen3.8-27B-UD-Q4_K_M.gguf",
        .revision = "4ca720788d1e01f1bff70c033e0d0028fd02e502",
        .sha256 = "322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482",
        .size = 16_464_440_224,
        .quantization = "UD-Q4_K_M",
        .architecture = "qwen35",
        .profile = .qwen38,
        // The embedded block re-measured at draft 7: prose 1.25-1.57x from 512 to 32K, code 1.92-1.95x; on.
        .speculative = true,
        .draft_length = 7,
        .think = .low,
        .thinking_budget = 1024,
        .companions = &.{
            .{ .role = .mmproj, .file = "mmproj-BF16.gguf", .size = 931_146_432, .sha256 = "83ee4f4f205fa514161778c41df1ea14144faa0f713510893b63c2395f5c2d53", .used_for = "image input" },
            .{ .role = .mtp, .file = "MTP/mtp-Qwen3.8-27B-Q4_0.gguf", .size = 1_369_590_656, .sha256 = "50d9ce5a6da381bbcfb31061cf73df94a90e6faf8efeddee379a9cb8f1501c6e", .used_for = "speculative decoding" },
        },
    },
    // Gemma 4 12B as Google's quantization-aware-trained checkpoint, whose
    // every weight matrix is Q4_0 — the encoding it was *trained* for, which
    // is why it decodes faster than the K-quant release at the same size.
    // The K-quant file (`unsloth/gemma-4-12b-it-GGUF`, the adapter's bring-up
    // file) left the catalogue on 2026-09-26 and runs through discovery.
    .{
        .name = "gemma-4-12b-qat",
        .title = "Gemma 4 12B QAT",
        .repo = "unsloth/gemma-4-12B-it-qat-GGUF",
        .file = "gemma-4-12B-it-qat-UD-Q4_K_XL.gguf",
        .revision = "980b060c40a8539ac159e0501a3e0f66a6365af3",
        .sha256 = "90fd44e29e0d7cffeb0fd00dc73cfdab9ed0b0e95306ecf7821ea634c940c370",
        .size = 6_716_356_800,
        // The Hub name says UD-Q4_K_XL; the file's own header says
        // "smart Q4_0, QAT-lossless" and every matrix is Q4_0.
        .quantization = "Q4_0 (QAT)",
        .architecture = "gemma4",
        .block_count = inference.models.gemma4.config_12b.layer_count,
        .profile = .gemma4,
        // The assistant heads at draft 5: prose 1.34x geometric mean, 0.98x at 32K; on.
        .speculative = true,
        .draft_length = 5,
        .think = .low,
        .thinking_budget = 1024,
        .companions = &.{
            .{ .role = .mmproj, .file = "mmproj-BF16.gguf", .size = 175_115_840, .sha256 = "dcb8103adad042b1bf99df767aaf34eb37c5a73a4a2f0417e4d7ba557e91664f", .used_for = "image input" },
            .{ .role = .mtp, .file = "mtp-gemma-4-12B-it.gguf", .size = 253_708_800, .sha256 = "fcb35dea42c71333db904cee11baac525c9ef872818ee3753f6cb156f3c6f4f6", .used_for = "speculative decoding" },
        },
    },
    // The mixture-of-experts sibling, QAT file only: the K-quant release
    // stores expert down-projections as Q5_1, which no kernel executes.
    // Pinned 2026-09-17 by the pull; runnable once the adapter binds the
    // expert tensors, supported with its acceptance record.
    .{
        .name = "gemma-4-26b-a4b",
        .title = "Gemma 4 26B-A4B",
        .repo = "unsloth/gemma-4-26B-A4B-it-qat-GGUF",
        .file = "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
        .revision = "7b92b5b28818151e8669af2e45e88d6086f490dd",
        .sha256 = "a7c5bc715f5ff8e99a3e8901ce7d2b42b402c669bf24f7c5250747633d0f5891",
        .size = 14_249_047_104,
        .quantization = "Q4_0 (QAT)",
        .architecture = "gemma4",
        .block_count = inference.models.gemma4.config_26b_a4b.layer_count,
        .profile = .gemma4,
        // The 26B-A4B head loses 0.63-0.76x on prose at drafts 2-7 (rows route to their own experts); off.
        .speculative = false,
        .draft_length = 4,
        .think = .low,
        .thinking_budget = 1024,
        .companions = &.{
            .{ .role = .mmproj, .file = "mmproj-BF16.gguf", .size = 1_194_828_256, .sha256 = "7b06953ccdbe8cf363f47841a7afaacd2b1c2ff9a8d6b426fdec7521a6878744", .used_for = "image input" },
            .{ .role = .mtp, .file = "MTP/mtp-gemma-4-26B-A4B-it-Q4_0.gguf", .size = 251_939_328, .sha256 = "7272d97595f0d4c74bd7b623492b7dbdaafd8b7c72f329a8270ba4eca68f768a", .used_for = "speculative decoding" },
        },
    },
    // The on-device E4B, QAT file only: per-layer embeddings and shared
    // KV layers in the adapter, its own template revision (`gemma4_e`, no
    // empty thought channel with thinking off), and a causal image span.
    // Pinned 2026-09-26 by the pull.
    .{
        .name = "gemma-4-e4b-qat",
        .title = "Gemma 4 E4B QAT",
        .repo = "unsloth/gemma-4-E4B-it-qat-GGUF",
        .file = "gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf",
        .revision = "8c5a9e4fd5482e2be20fe0bf013b4c262a8f4265",
        .sha256 = "df0fd4ee07072c607c29a0a1cb4f98918426cca12f45a2776bdd6ee6d09a4de3",
        .size = 4_215_695_776,
        .quantization = "Q4_0 (QAT)",
        .architecture = "gemma4",
        .block_count = inference.models.gemma4.config_e4b.layer_count,
        .profile = .gemma4_e,
        // The E4B head at draft 6: prose 1.47x geometric mean, 0.99x at 32K; on.
        .speculative = true,
        .draft_length = 6,
        .think = .low,
        .thinking_budget = 1024,
        .hears = true,
        .companions = &.{
            .{ .role = .mmproj, .file = "mmproj-BF16.gguf", .size = 991_552_320, .sha256 = "7c9bafa27f82d658eda805c1d82ef62bb0368e1ff75f64f77de58ad318beaaf9", .used_for = "image and audio input" },
            .{ .role = .mtp, .file = "MTP/mtp-gemma-4-E4B-it-Q4_0.gguf", .size = 59_678_016, .sha256 = "423074e537504b4f9ec5eafed5c639fac82c96631626efccacdd3c4039b20605", .used_for = "speculative decoding" },
        },
    },
    // Meta's dense agentic model. Its draft companion is a DFlash drafter,
    // not an MTP head; it takes the `mtp` role because that role means
    // "the draft source the speculative-decoding unit loads", whatever
    // its mechanism (decided 2026-09-17). The adapter binds it and the
    // profile renders its prompts.
    .{
        .name = "muse-glimmer-30b",
        .title = "Muse Glimmer 30B",
        .repo = "unsloth/Muse-Glimmer-30B-GGUF",
        .file = "Muse-Glimmer-30B-UD-Q4_K_XL.gguf",
        .revision = "faa5b025c584459c13febfa5c59883516710ae39",
        .sha256 = "82bece304887a313ece08400bc030f6066c7bff5b906b0cd40308ec8a409fd38",
        .size = 15_878_222_368,
        .quantization = "UD-Q4_K_XL",
        .architecture = "muse-glimmer",
        .profile = .muse_glimmer,
        // The DFlash drafter at draft 6: prose 1.27x geometric mean, 1.10x at 32K; past 6 the block costs double; on.
        .speculative = true,
        .draft_length = 6,
        .think = .low,
        .thinking_budget = 1024,
        .companions = &.{
            .{ .role = .mmproj, .file = "mmproj-kquant.gguf", .size = 1_400_328_928, .sha256 = "f48b452316f9b213758e8659444029b961a24a07f99a1abb2a9f88b06f7c00c6", .used_for = "image input" },
            .{ .role = .mtp, .file = "dflash-kquant.gguf", .size = 1_631_205_312, .sha256 = "27d9a805fa29b943cfb6ad4843367cd4eaaaf06bd452d8cc3e00a2cd18a677bc", .used_for = "speculative decoding (a DFlash drafter)" },
        },
    },
};

/// What runs an artifact: the text engine, `nuclis decide`, `nuclis embed`,
/// `nuclis speak`, or `nuclis transcribe`.
pub const ModelKind = enum { generation, decision, embedding, speech, transcription };

/// A decision checkpoint (Laya, clef): safetensors weights and the support
/// files beside them, one directory `nuclis decide` opens, and for clef a
/// backbone GGUF from another repository. Separate from `entries`
/// so nothing that resolves a text model can pick one.
pub const DecisionEntry = struct {
    name: []const u8,
    repo: []const u8,
    /// The weights' Hub name; its directory is the checkpoint's.
    file: []const u8,
    revision: []const u8,
    sha256: []const u8,
    size: u64,
    architecture: []const u8,
    quantization: []const u8,
    /// Hub names the checkpoint needs beside the weights (plain git files:
    /// the Hub gives their blob ids, not SHA-256).
    support: []const []const u8,
    /// clef's backbone GGUF, in another repository, pulled with the head.
    backbone: ?Artifact = null,
    /// The backbone's vision projector (`--with mmproj`).
    mmproj: ?Artifact = null,

    /// The head (or weights) and the backbone, without the projector.
    pub fn totalSize(self: *const DecisionEntry) u64 {
        return self.size + if (self.backbone) |b| b.size else 0;
    }
};

/// A pinned file of another repository a decision checkpoint runs with.
pub const Artifact = struct {
    repo: []const u8,
    file: []const u8,
    revision: []const u8,
    sha256: []const u8,
    size: u64,
};

pub const decision_entries = [_]DecisionEntry{
    // Laya's English root checkpoint, a ModernBERT-large encoder with a
    // typed-decision head; pinned 2026-09-27 by the pull.
    .{
        .name = "laya",
        .repo = "convaiinnovations/laya",
        .file = "model.safetensors",
        .revision = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
        .sha256 = "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c",
        .size = 842_609_210,
        .architecture = "modernbert",
        .quantization = "F16",
        .support = &.{ "encoder/config.json", "rl_agent_config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json" },
    },
    // The multilingual checkpoint beside it, an mmBERT-base encoder with the
    // same head; no routing by language, the caller picks it.
    .{
        .name = "laya-multilingual",
        .repo = "convaiinnovations/laya",
        .file = "multilingual/model.safetensors",
        .revision = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
        .sha256 = "9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204",
        .size = 643_835_514,
        .architecture = "modernbert",
        .quantization = "F16",
        .support = &.{ "multilingual/encoder/config.json", "multilingual/rl_agent_config.json", "multilingual/tokenizer/tokenizer.json", "multilingual/tokenizer/tokenizer_config.json" },
    },
    // Cloudflare's clef-flash: the joint schema head and support files from
    // Cloudflare's repository, the Qwen3.5-9B backbone as bartowski's Q6_K
    // GGUF (llama.cpp b11279, imatrix) and its BF16 projector; pinned
    // 2026-10-02 by the pull.
    .{
        .name = "clef-flash",
        .repo = "Cloudflare/clef-flash",
        .file = "joint_head.safetensors",
        .revision = "17f0b0ad64efb65d273590632833508766b2aae6",
        .sha256 = "19cdcec8c81dc9212be320fff47462ab342fbc1278be4368fb3da71241cf5ba0",
        .size = 243_538_016,
        .architecture = "qwen35",
        .quantization = "Q6_K",
        .support = &.{ "config.json", "joint_head_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "processor_config.json", "generation_config.json" },
        .backbone = .{
            .repo = "bartowski/Cloudflare_clef-flash-GGUF",
            .file = "Cloudflare_clef-flash-Q6_K.gguf",
            .revision = "d7f376ea88c05e7bb1014dd5351a93df9dd8029e",
            .sha256 = "b80f7cfb803de25aade15853bab331e9f030a9ee64dee7bc91ed8c2b07312423",
            .size = 7_793_714_496,
        },
        .mmproj = .{
            .repo = "bartowski/Cloudflare_clef-flash-GGUF",
            .file = "mmproj-Cloudflare_clef-flash-bf16.gguf",
            .revision = "d7f376ea88c05e7bb1014dd5351a93df9dd8029e",
            .sha256 = "3c45b34aee6f353a0d41d6b96ba712a498a17a82f0a6152bf68a5021f9652c0f",
            .size = 921_704_928,
        },
    },
};

/// What an embedding input's parts may be.
pub const Modality = enum { text, image, audio };

/// An embedding model (EmbeddingGemma 2): a GGUF the `Embedder` opens, and
/// the projector its image and audio parts need. Separate from `entries`
/// so nothing that resolves a text model can pick one.
pub const EmbeddingEntry = struct {
    name: []const u8,
    title: []const u8,
    repo: []const u8,
    file: []const u8,
    revision: []const u8,
    sha256: []const u8,
    size: u64,
    architecture: []const u8,
    quantization: []const u8,
    /// The vision and audio encoders (`--with mmproj`), in `repo` at
    /// `revision`; text needs only the main file.
    mmproj: ?Artifact = null,
    /// The vector widths it is trained for, widest first.
    dimensions: []const usize,
    /// Every part kind the entry reads with all its files; image and audio
    /// need the projector.
    modalities: []const Modality,
};

pub const embedding_entries = [_]EmbeddingEntry{
    // Google's EmbeddingGemma 2 as unsloth's Q8_0 conversion with the BF16
    // projector (docs/models/embeddinggemma.md § Artifacts); pinned
    // 2026-10-08 by the pull.
    .{
        .name = "embeddinggemma-2",
        .title = "EmbeddingGemma 2",
        .repo = "unsloth/embeddinggemma-2-GGUF",
        .file = "embeddinggemma-2-Q8_0.gguf",
        .revision = "031f0d4b35536f69ab3509d4893c923264fcf253",
        .sha256 = "6f1bd4ac6c5df7444f9cca7ca36cafe6cfa34cd6f49fefb1e0b4be8143aed8bc",
        .size = 309_855_520,
        .architecture = inference.embed.architecture,
        .quantization = "Q8_0",
        .mmproj = .{
            .repo = "unsloth/embeddinggemma-2-GGUF",
            .file = "mmproj-BF16.gguf",
            .revision = "031f0d4b35536f69ab3509d4893c923264fcf253",
            .sha256 = "995aaa56e88b9b631f651861d659b728a625ccc02a238be37cff56cf11dc0032",
            .size = 982_074_880,
        },
        .dimensions = &inference.embed.widths,
        .modalities = &.{ .text, .image, .audio },
    },
};

/// A speech or transcription checkpoint: safetensors weights and the
/// support files beside them, one directory `nuclis speak` or `nuclis
/// transcribe` opens. Separate from `entries` so nothing that resolves a
/// text model can pick one.
pub const AudioEntry = struct {
    name: []const u8,
    title: []const u8,
    /// `speech` or `transcription`.
    kind: ModelKind,
    repo: []const u8,
    /// The weights' Hub name; its directory is the checkpoint's.
    file: []const u8,
    revision: []const u8,
    sha256: []const u8,
    size: u64,
    architecture: []const u8,
    quantization: []const u8,
    /// Hub names the checkpoint needs beside the weights (plain git files).
    support: []const []const u8,
    /// The speech codec's weights, in `repo` at `revision`, pulled with the
    /// checkpoint; `codec_support` are its own support files.
    codec: ?Artifact = null,
    codec_support: []const []const u8 = &.{},
    /// The speakers a speech model voices, as its `config.json` names them.
    speakers: []const []const u8 = &.{},
    /// Whether it performs a delivery instruction (the 1.7B CustomVoice).
    instructions: bool = false,

    /// The weights and the codec's.
    pub fn totalSize(self: *const AudioEntry) u64 {
        return self.size + if (self.codec) |c| c.size else 0;
    }
};

const qwen3_tts_speakers = [_][]const u8{ "aiden", "dylan", "eric", "ono_anna", "ryan", "serena", "sohee", "uncle_fu", "vivian" };
const qwen3_tts_support = [_][]const u8{ "config.json", "generation_config.json", "merges.txt", "preprocessor_config.json", "tokenizer_config.json", "vocab.json" };
const qwen3_tts_codec_support = [_][]const u8{ "speech_tokenizer/config.json", "speech_tokenizer/configuration.json", "speech_tokenizer/preprocessor_config.json" };

pub const audio_entries = [_]AudioEntry{
    // Qwen3-TTS 12Hz CustomVoice, both sizes as published (BF16 talker, F32
    // codec, byte-identical between them); pinned 2026-10-27 by the pull
    // (docs/models/qwen3-tts.md). Only the 1.7B performs an instruction.
    .{
        .name = "qwen3-tts-1.7b",
        .title = "Qwen3-TTS 1.7B CustomVoice",
        .kind = .speech,
        .repo = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        .file = "model.safetensors",
        .revision = "0c0e3051f131929182e2c023b9537f8b1c68adfe",
        .sha256 = "38b1d5971bdbd982b561cccec982669a53b0537c3cf5e9bd4778ed07bb2f5137",
        .size = 3_833_402_552,
        .architecture = "qwen3_tts",
        .quantization = "BF16",
        .support = &qwen3_tts_support,
        .codec = .{
            .repo = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            .file = "speech_tokenizer/model.safetensors",
            .revision = "0c0e3051f131929182e2c023b9537f8b1c68adfe",
            .sha256 = "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258",
            .size = 682_293_092,
        },
        .codec_support = &qwen3_tts_codec_support,
        .speakers = &qwen3_tts_speakers,
        .instructions = true,
    },
    .{
        .name = "qwen3-tts-0.6b",
        .title = "Qwen3-TTS 0.6B CustomVoice",
        .kind = .speech,
        .repo = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        .file = "model.safetensors",
        .revision = "85e237c12c027371202489a0ec509ded67b5e4b5",
        .sha256 = "bc3c7e785eb961179c25450d1acff03f839e0002f2f3a5aeb67b5735c0fa2adb",
        .size = 1_811_626_576,
        .architecture = "qwen3_tts",
        .quantization = "BF16",
        .support = &qwen3_tts_support,
        .codec = .{
            .repo = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            .file = "speech_tokenizer/model.safetensors",
            .revision = "85e237c12c027371202489a0ec509ded67b5e4b5",
            .sha256 = "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258",
            .size = 682_293_092,
        },
        .codec_support = &qwen3_tts_codec_support,
        .speakers = &qwen3_tts_speakers,
    },
    // NVIDIA's Nemotron 3.5 ASR streaming 0.6B, F32 as published; pinned
    // 2026-10-27 by the pull (docs/models/nemotron-asr.md).
    .{
        .name = "nemotron-asr-0.6b",
        .title = "Nemotron 3.5 ASR Streaming 0.6B",
        .kind = .transcription,
        .repo = "nvidia/nemotron-3.5-asr-streaming-0.6b",
        .file = "model.safetensors",
        .revision = "ea30d66debe3740a08b573244286791d423d6b3e",
        .sha256 = "9eebdd6590289cb3030f310858f3df93256600a800a3e8200c5993d5f967e174",
        .size = 2_552_062_944,
        .architecture = "nemotron3_5_asr",
        .quantization = "F32",
        .support = &.{ "config.json", "generation_config.json", "processor_config.json", "tokenizer.json", "tokenizer_config.json" },
    },
};

pub fn findAudio(name: []const u8) ?*const AudioEntry {
    for (&audio_entries) |*e| if (std.mem.eql(u8, e.name, name)) return e;
    return null;
}

/// The checkpoint directory of an audio entry: the weights' directory under
/// `<models>/<repo>/`.
pub fn audioDirectory(alloc: Allocator, models_dir: []const u8, entry: *const AudioEntry) ![]u8 {
    const weights = try std.fs.path.join(alloc, &.{ models_dir, entry.repo, entry.file });
    defer alloc.free(weights);
    return alloc.dupe(u8, std.fs.path.dirname(weights).?);
}

pub fn findEmbedding(name: []const u8) ?*const EmbeddingEntry {
    for (&embedding_entries) |*e| if (std.mem.eql(u8, e.name, name)) return e;
    return null;
}

/// The embedding entry whose main file has this digest, so a vector's space
/// names the checkpoint however the caller reached its file.
pub fn embeddingByDigest(sha256: []const u8) ?*const EmbeddingEntry {
    for (&embedding_entries) |*e| if (std.mem.eql(u8, e.sha256, sha256)) return e;
    return null;
}

pub fn findDecision(name: []const u8) ?*const DecisionEntry {
    for (&decision_entries) |*e| if (std.mem.eql(u8, e.name, name)) return e;
    return null;
}

/// What runs `name`: its registry entry's kind, else the catalogue that
/// lists it; anything else (a path, an unknown name) is a text model's.
pub fn kindOf(registry: config.Models, name: []const u8) ModelKind {
    if (registry.find(name)) |e| return e.kind orelse .generation;
    if (findDecision(name) != null) return .decision;
    if (findEmbedding(name) != null) return .embedding;
    if (findAudio(name)) |e| return e.kind;
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

/// The checkpoint directory of a decision entry: the weights' directory
/// under `<models>/<repo>/`.
pub fn decisionDirectory(alloc: Allocator, models_dir: []const u8, entry: *const DecisionEntry) ![]u8 {
    const weights = try std.fs.path.join(alloc, &.{ models_dir, entry.repo, entry.file });
    defer alloc.free(weights);
    return alloc.dupe(u8, std.fs.path.dirname(weights).?);
}

/// The widest entry name, so every listing lines its status column up
/// however the catalogue grows. Computed here rather than guessed at each
/// call site, which is what let `gemma-4-12b-qat` overflow a hardcoded 12.
pub const name_width = blk: {
    var widest: usize = 0;
    for (&entries) |entry| widest = @max(widest, entry.name.len);
    for (&decision_entries) |entry| widest = @max(widest, entry.name.len);
    for (&embedding_entries) |entry| widest = @max(widest, entry.name.len);
    for (&audio_entries) |entry| widest = @max(widest, entry.name.len);
    break :blk widest + 1;
};

/// Pads `text` to `name_width` in `buffer` (which must be at least that
/// wide), so padding happens before any escape sequence and colour can never
/// move a column.
pub fn padName(buffer: []u8, text: []const u8) []const u8 {
    const written = @min(text.len, buffer.len);
    @memcpy(buffer[0..written], text[0..written]);
    const total = @min(@max(written, name_width), buffer.len);
    @memset(buffer[written..total], ' ');
    return buffer[0..total];
}

/// The first catalogue model that hears audio, for a refusal that names
/// where to go instead; null when none does.
pub fn hearing() ?*const Entry {
    for (&entries) |*e| if (e.hears) return e;
    return null;
}

/// Exact name match; names never contain `/` or end in `.gguf`, so a path
/// can never be mistaken for one.
pub fn find(name: []const u8) ?*const Entry {
    for (&entries) |*e| if (std.mem.eql(u8, e.name, name)) return e;
    return null;
}

/// A file the catalogue pins, main or companion, by repository and Hub
/// name; `model inspect` compares the Hub's digest against `sha256`.
pub const FileMatch = struct {
    entry: *const Entry,
    role: Role,
    sha256: []const u8,
    /// What the companion is loaded for; null for the main file.
    used_for: ?[]const u8,
};

pub fn findFile(repo: []const u8, file: []const u8) ?FileMatch {
    for (&entries) |*e| {
        if (!std.mem.eql(u8, e.repo, repo)) continue;
        if (std.mem.eql(u8, e.file, file)) return .{ .entry = e, .role = .main, .sha256 = e.sha256, .used_for = null };
        for (e.companions) |*c| if (std.mem.eql(u8, c.file, file)) return .{ .entry = e, .role = c.role, .sha256 = c.sha256, .used_for = c.used_for };
    }
    return null;
}

/// `<models>/<repo>/<file>`, the layout `model pull` writes.
pub fn localPath(alloc: Allocator, models_dir: []const u8, entry: *const Entry, file: []const u8) ![]u8 {
    return std.fs.path.join(alloc, &.{ models_dir, entry.repo, file });
}

/// From the sidecar alone: `present` when it records the pinned digest and
/// size, `mismatch` when it records something else, `unverified` when the
/// file is there without one (copied in by hand; a `pull` verifies it),
/// `absent` otherwise.
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

test "decision entries are well formed and never text-model names" {
    for (&decision_entries) |e| {
        try std.testing.expectEqual(@as(usize, 40), e.revision.len);
        try std.testing.expectEqual(@as(usize, 64), e.sha256.len);
        try std.testing.expect(find(e.name) == null);
        try std.testing.expect(std.mem.endsWith(u8, e.file, ".safetensors"));
        // Support files sit in the weights' directory or below it.
        const prefix = if (std.fs.path.dirname(e.file)) |d| d.len + 1 else 0;
        for (e.support) |name| try std.testing.expect(prefix == 0 or std.mem.startsWith(u8, name, e.file[0..prefix]));
    }
    const dir = try decisionDirectory(std.testing.allocator, "/m", findDecision("laya").?);
    defer std.testing.allocator.free(dir);
    try std.testing.expectEqualStrings("/m/convaiinnovations/laya", dir);
    const multilingual = try decisionDirectory(std.testing.allocator, "/m", findDecision("laya-multilingual").?);
    defer std.testing.allocator.free(multilingual);
    try std.testing.expectEqualStrings("/m/convaiinnovations/laya/multilingual", multilingual);
    try std.testing.expect(findDecision("qwen3.8-27b") == null);
}

test "embedding entries are well formed and share no name with another kind" {
    for (&embedding_entries) |e| {
        try std.testing.expectEqual(@as(usize, 40), e.revision.len);
        try std.testing.expectEqual(@as(usize, 64), e.sha256.len);
        try std.testing.expect(find(e.name) == null and findDecision(e.name) == null);
        try std.testing.expect(std.mem.endsWith(u8, e.file, ".gguf") and e.title.len > 0);
        try std.testing.expectEqual(Modality.text, e.modalities[0]);
        // The pull fetches the projector from the entry's own repository.
        if (e.mmproj) |m| try std.testing.expect(std.mem.eql(u8, m.repo, e.repo) and std.mem.eql(u8, m.revision, e.revision) and m.sha256.len == 64);
        try std.testing.expect(embeddingByDigest(e.sha256) == findEmbedding(e.name));
    }
    try std.testing.expectEqualStrings("gemma-embedding2", findEmbedding("embeddinggemma-2").?.architecture);
    try std.testing.expect(findEmbedding("qwen3.8-27b") == null and findEmbedding("laya") == null);
}

test "audio entries are well formed, of their two kinds, and share no name with another kind" {
    for (&audio_entries, 0..) |*e, i| {
        try std.testing.expectEqual(@as(usize, 40), e.revision.len);
        try std.testing.expectEqual(@as(usize, 64), e.sha256.len);
        try std.testing.expect(e.kind == .speech or e.kind == .transcription);
        try std.testing.expect(find(e.name) == null and findDecision(e.name) == null and findEmbedding(e.name) == null);
        try std.testing.expect(std.mem.endsWith(u8, e.file, ".safetensors") and e.title.len > 0 and e.support.len > 0);
        try std.testing.expectEqual(e.kind, kindOf(.{}, e.name));
        // A speech model brings its codec from its own repository and commit.
        try std.testing.expectEqual(e.kind == .speech, e.codec != null);
        if (e.codec) |c| {
            try std.testing.expect(std.mem.eql(u8, c.repo, e.repo) and std.mem.eql(u8, c.revision, e.revision) and c.sha256.len == 64);
            const codec_dir = std.fs.path.dirnamePosix(c.file).?;
            for (e.codec_support) |name| try std.testing.expect(std.mem.startsWith(u8, name, codec_dir));
            try std.testing.expect(e.speakers.len > 0);
        }
        for (audio_entries[i + 1 ..]) |other| try std.testing.expect(!std.mem.eql(u8, e.name, other.name));
    }
    // Only the 1.7B performs an instruction; both speak with the same codec.
    try std.testing.expect(findAudio("qwen3-tts-1.7b").?.instructions and !findAudio("qwen3-tts-0.6b").?.instructions);
    try std.testing.expectEqualStrings(findAudio("qwen3-tts-1.7b").?.codec.?.sha256, findAudio("qwen3-tts-0.6b").?.codec.?.sha256);
    try std.testing.expectEqual(ModelKind.transcription, kindOf(.{}, "nemotron-asr-0.6b"));
    try std.testing.expectEqual(@as(u64, 3_833_402_552 + 682_293_092), findAudio("qwen3-tts-1.7b").?.totalSize());
    const dir = try audioDirectory(std.testing.allocator, "/m", findAudio("nemotron-asr-0.6b").?);
    defer std.testing.allocator.free(dir);
    try std.testing.expectEqualStrings("/m/nvidia/nemotron-3.5-asr-streaming-0.6b", dir);
    try std.testing.expect(findAudio("laya") == null and findAudio("qwen3.8-27b") == null);
}

test "the table is well formed: unique names, 40-character commits, 64-character digests" {
    for (entries) |e| try std.testing.expect(e.title.len > 0 and !std.mem.eql(u8, e.title, e.name));
    for (&entries, 0..) |e, i| {
        try std.testing.expectEqual(@as(usize, 40), e.revision.len);
        try std.testing.expectEqual(@as(usize, 64), e.sha256.len);
        try std.testing.expect(std.mem.indexOfScalar(u8, e.name, '/') == null and !std.mem.endsWith(u8, e.name, ".gguf"));
        for (e.companions) |c| {
            try std.testing.expectEqual(@as(usize, 64), c.sha256.len);
            try std.testing.expect(c.role != .main);
        }
        for (entries[i + 1 ..]) |other| try std.testing.expect(!std.mem.eql(u8, e.name, other.name));
    }
    try std.testing.expectEqualStrings("unsloth/Qwen3.8-27B-GGUF", find("qwen3.8-27b").?.repo);
    try std.testing.expectEqual(Profile.gemma4, find("gemma-4-12b-qat").?.profile);
    try std.testing.expectEqualStrings("gemma4", find("gemma-4-12b-qat").?.architecture);
    try std.testing.expectEqual(Profile.gemma4, find("gemma-4-26b-a4b").?.profile);
    try std.testing.expectEqual(Profile.gemma4_e, find("gemma-4-e4b-qat").?.profile);
    try std.testing.expectEqual(Profile.muse_glimmer, find("muse-glimmer-30b").?.profile);
    // Files outside the catalogue run through discovery, not by name.
    try std.testing.expect(find("gemma-4-12b") == null and find("bonsai-2-27b") == null);
    // The measured speculative verdicts: every drafter pays but the mixture-of-experts head.
    try std.testing.expect(find("qwen3.8-27b").?.speculative);
    try std.testing.expectEqual(@as(usize, 7), find("qwen3.8-27b").?.draft_length);
    try std.testing.expectEqual(@as(usize, 5), find("gemma-4-12b-qat").?.draft_length);
    try std.testing.expectEqual(@as(usize, 6), find("gemma-4-e4b-qat").?.draft_length);
    try std.testing.expectEqual(@as(usize, 6), find("muse-glimmer-30b").?.draft_length);
    try std.testing.expect(!find("gemma-4-26b-a4b").?.speculative);
    // Only the E4B's projector carries an audio encoder.
    try std.testing.expectEqualStrings("gemma-4-e4b-qat", hearing().?.name);
    for (&entries) |*e| if (e.hears) try std.testing.expect(e.companion(.mmproj) != null);
    for (&entries) |e| if (e.speculative) try std.testing.expect(e.draft_length >= 1 and e.draft_length <= inference.engine.max_draft_length);
    try std.testing.expectEqualStrings("muse-glimmer", find("muse-glimmer-30b").?.architecture);
    try std.testing.expectEqual(Role.mtp, findFile("unsloth/Muse-Glimmer-30B-GGUF", "dflash-kquant.gguf").?.role);
    try std.testing.expectEqual(Role.mtp, findFile("unsloth/gemma-4-12B-it-qat-GGUF", "mtp-gemma-4-12B-it.gguf").?.role);
    try std.testing.expect(find("qwen") == null);
    try std.testing.expect(find("qwen/Qwen3.8-27B-UD-Q4_K_M.gguf") == null);
    try std.testing.expectEqualStrings("speculative decoding", find("qwen3.8-27b").?.companion(.mtp).?.used_for);
    try std.testing.expect(find("qwen3.8-27b").?.companion(.imatrix) == null);
    try std.testing.expectEqual(Role.main, findFile("unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-Q4_K_M.gguf").?.role);
    try std.testing.expectEqualStrings("image input", findFile("unsloth/Qwen3.8-27B-GGUF", "mmproj-BF16.gguf").?.used_for.?);
    try std.testing.expect(findFile("unsloth/Qwen3.8-27B-GGUF", "other.gguf") == null);
    try std.testing.expect(findFile("other/repo", "Qwen3.8-27B-UD-Q4_K_M.gguf") == null);
}

test "status derives from the sidecar without hashing" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", alloc);
    defer alloc.free(root);
    const entry = find("qwen3.8-27b").?;
    const path = try localPath(alloc, root, entry, entry.file);
    defer alloc.free(path);
    try std.testing.expectEqual(Status.absent, try status(alloc, io, path, entry.sha256, entry.size));
    try tmp.dir.createDirPath(io, "unsloth/Qwen3.8-27B-GGUF");
    try tmp.dir.writeFile(io, .{ .sub_path = "unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf", .data = "GGUF" });
    try std.testing.expectEqual(Status.unverified, try status(alloc, io, path, entry.sha256, entry.size));
    var sidecar: model.Sidecar = .{ .repo = entry.repo, .file = entry.file, .revision = entry.revision, .sha256 = entry.sha256, .size = entry.size, .role = .main, .downloaded_at = "2026-09-11T00:00:00Z", .nuclis_version = model.version };
    const sidecar_path = try model.sidecarPath(alloc, path);
    defer alloc.free(sidecar_path);
    try model.writeSidecar(alloc, io, .cwd(), sidecar_path, sidecar);
    try std.testing.expectEqual(Status.present, try status(alloc, io, path, entry.sha256, entry.size));
    sidecar.size = 1;
    try model.writeSidecar(alloc, io, .cwd(), sidecar_path, sidecar);
    try std.testing.expectEqual(Status.mismatch, try status(alloc, io, path, entry.sha256, entry.size));
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = sidecar_path, .data = "{" });
    try std.testing.expectEqual(Status.mismatch, try status(alloc, io, path, entry.sha256, entry.size));
}
