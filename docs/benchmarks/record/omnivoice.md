<!-- kind: record -->
# OmniVoice / VoiceTut-TTS: the record

The dated record behind [the OmniVoice card](../../models/omnivoice.md): the
`audio.cpp` container schema, tensor table, embedded configuration files, and
forward-pass equations, each true on its date.

## Sources (2026-11-22)

- **The artifact**: `mohammedaly22/VoiceTut-TTS-GGUF`
  `voicetut-tts-q8_0.gguf` at commit
  `615457bb2e9043f468e012c159146b28fa8f5959`, 1,350,264,224 B, SHA-256
  `995a1fc8973cf376b490231aa107bd22ab1f5da003e89d11ada48f6d5058f67a`
  ([catalogue.md § Pinned commits and digests](catalogue.md#pinned-commits-and-digests-2026-09-11)).
- **The container**: `audio.cpp`'s single-GGUF packed container
  (`general.architecture = "audiocpp"`, `audiocpp.model_spec.family = "omnivoice"`),
  carrying weights, codec, and embedded JSON sidecars in one file.
- **The semantics**: `k2-fsa/OmniVoice` (bidirectional Qwen3-0.6B masked
  diffusion generator and iterative unmasking schedule) and
  `bosonai/higgs-audio-v2-tokenizer` (24 kHz DAC decoder and 16 kHz HuBERT-Base
  + DAC semantic-acoustic encoder).

## The file (2026-11-22)

The inventory is `inference/src/models/fixtures/omnivoice-voicetut-q8_0.json`
(`scripts/fixtures/gguf-inventory.py`): GGUF v3, 16 metadata keys, 840 tensors,
directory 11,568,567 B.

### Container metadata (2026-11-22)

| Key | Type | Value / count |
| --- | --- | --- |
| `general.architecture` | string | `"audiocpp"` |
| `general.name` | string | `".VoiceTut-TTS-gguf-root"` |
| `audiocpp.tensor_name_format` | string | `"native"` |
| `audiocpp.source_format` | string | `"packed"` |
| `audiocpp.weight_type` | string | `"q8_0"` |
| `audiocpp.model_spec.version` | uint32 | `1` |
| `audiocpp.model_spec.family` | string | `"omnivoice"` |
| `audiocpp.model_spec.json` | string | 3,831 B JSON manifest |
| `audiocpp.tensor_sources.names` | array\<string\> | 2 entries (`audio_tokenizer_weights`, `weights`) |
| `audiocpp.tensor_sources.paths` | array\<string\> | 2 entries |
| `audiocpp.embedded_files.names` | array\<string\> | 6 entries |
| `audiocpp.embedded_files.offsets` | array\<uint64\> | 7 offsets `[0, 2531, 2737, 6905, 9144, 11433130, 11433692]` |
| `audiocpp.embedded_files.data` | array\<uint8\> | 11,433,692 bytes (lazy descriptor; required raising `gguf.Limits.array_items` from `4_000_000` to `32_000_000`) |
| `audiocpp.tensor_names` | array\<string\> | 840 full native tensor names (resolves `_audiocpp.2`…`_audiocpp.526` aliases for names > 64 bytes) |
| `audiocpp.tensor_ranks` | array\<int32\> | 840 entries |
| `audiocpp.tensor_shapes` | array\<int64\> | 1,503 flattened shape dimensions |

### Embedded sidecar files (2026-11-22)

| File | Size (B) | Role |
| --- | ---: | --- |
| `audio_tokenizer/config.json` | 2,531 | `HiggsAudioV2TokenizerModel`: 24 kHz DAC decoder (`decoder_hidden_size = 1024`, `upsampling_ratios = [8, 5, 4, 2, 3]`, `hop_length = 960`), 16 kHz HuBERT-Base (`hidden_size = 768`, 12 layers, 12 heads, 7 feature-extractor convs), RVQ `codebook_size = 1024`, `codebook_dim = 64` |
| `audio_tokenizer/preprocessor_config.json` | 206 | `DacFeatureExtractor`: `sampling_rate = 24000`, `hop_length = 960`, `padding_side = "right"` |
| `chat_template.jinja` | 4,168 | Inherited Qwen3 template (unused by OmniVoice speech prompts) |
| `config.json` | 2,239 | `OmniVoice` (`num_audio_codebook = 8`, `audio_vocab_size = 1025`, `audio_mask_id = 1024`, `audio_codebook_weights = [8, 8, 6, 6, 4, 4, 2, 2]`) + `llm_config` (`qwen3`, 28 layers, hidden 1,024, FFN 3,072 SiLU, 16Q / 8KV × 128, `rms_norm_eps = 1e-6`, `rope_theta = 1000000`, `vocab_size = 151676`) |
| `tokenizer.json` | 11,423,986 | Hugging Face Qwen2/Qwen3 BPE tokenizer: NFC normalizer, `Sequence([Split, ByteLevel])` pre-tokenizer, `ByteLevel` decoder, 151,643 vocab, 151,387 merges, 33 added tokens (including `151669` `<\|denoise\|>` through `151675` `<\|text_end\|>`) |
| `tokenizer_config.json` | 562 | `Qwen2Tokenizer`, `pad_token = "<\|endoftext\|>"`, `eos_token = "<\|im_end\|>"`, 7 `extra_special_tokens` |

### Tensor table and encodings (2026-11-22)

All 840 tensors partition into 313 generator tensors (`weights/*`) and 527
codec tensors (`audio_tokenizer_weights/*`):

| Encoding | GGML id | Tensors | Role |
| --- | ---: | ---: | --- |
| F32 | 0 | 513 | RMS norms, LayerNorms, biases, conv weights, RVQ codebooks |
| F16 | 1 | 37 | `weights/audio_embeddings.weight` `[1024, 8200]`, `weights/llm.embed_tokens.weight` `[1024, 151676]`, codec projections |
| Q8_0 | 8 | 289 | 28 Qwen3 backbone layers (`q/k/v/o_proj`, `gate/up/down_proj`), `weights/audio_heads.weight` `[1024, 8200]`, HuBERT linear projections |
| I64 | 27 | 1 | `weights/codebook_layer_offsets` `[8]` = `[0, 1025, 2050, 3075, 4100, 5125, 6150, 7175]` (added as storage layout id 27 in `encoding.zig`) |

### Forward-pass equations (2026-11-22)

- **Input embeddings**: each sequence position is either a text token
  (`llm.embed_tokens[id]`) or an 8-codebook audio frame whose embedding is the
  sum over codebooks $c \in \{0..7\}$ of
  `audio_embeddings[codebook_layer_offsets[c] + code[c]]`, where unmasked codes
  lie in `0..1023` and masked slots hold `audio_mask_id = 1024`.
- **Bidirectional Qwen3 backbone**: 28 pre-norm transformer layers with
  **non-causal** (full bidirectional) GQA self-attention (16 query heads, 8 KV
  heads, head dim 128), per-head RMSNorm on Q and K (`eps = 1e-6`), NeoX
  half-split RoPE (`theta = 1e6`), and SwiGLU MLP (`1024 → 3072 → 1024`,
  `SiLU`), followed by final RMSNorm `llm.norm`.
- **Codebook logits and CFG**: `audio_heads` projects the normalized hidden
  state at each target audio frame to 8 × 1,025 logits (8,200 columns, sliced
  per codebook at `codebook_layer_offsets[c]`). Classifier-free guidance
  evaluates conditional and unconditional sequences at each diffusion step with
  `guidance_scale = 2.0`.
- **Iterative unmasking**: `num_step = 32` cosine-shifted steps
  (`t_shift = 0.1`), scoring masked `(frame, codebook)` slots with
  `layer_penalty_factor = 5.0`, Gumbel position selection at
  `position_temperature = 5.0`, and token selection at
  `class_temperature = 0.0` (argmax).
- **Audio codec**: HiggsAudioV2Tokenizer decodes `[8, T]` codes at 25 fps
  through 8 RVQ codebooks (1,024 × 64) and a 5-stage DAC decoder
  (`upsampling_ratios = [8, 5, 4, 2, 3]`, product 960, `output_padding = stride % 2`,
  no final `tanh`) to 24 kHz mono PCM.
  Reference clips are encoded by the dual acoustic (DAC, 24 kHz) + semantic
  (HuBERT-Base 12-layer, 16 kHz, mean over all 13 hidden states subsampled `::2`)
  encoder into `[8, T_ref]` codes.

## The codec on the CPU (2026-11-22)

`HiggsAudioV2TokenizerModel` (`transformers` 5.19.0, `torch` 2.14.1, CPU F32)
was run once on all 527 `audio_tokenizer_weights/*` tensors dequantized from
`voicetut-tts-q8_0.gguf` over the first 48,000 samples (2.00 s, 50 frames at
24 kHz) of `tests/fixtures/qwen3-tts/0.6b-clone-reference/reference.wav`
(reconstruction SNR 7.22 dB, correlation 0.9040). The fixtures live under
`tests/fixtures/omnivoice/codec/`.

`omnivoice-check MODEL.gguf codec --max-relative-rms 1e-4` (gate
`omnivoice-codec-cpu`, tier `verify-cpu`, ReleaseFast on Apple M4 Pro):

| Stage | Shape | rel RMS | max \|Δ\| |
| --- | --- | ---: | ---: |
| `enc resampled_16k` | `[32000]` | 1.131e-7 | 1.192e-7 |
| `enc hubert` | `[50, 768]` | 2.789e-6 | 1.568e-5 |
| `enc semantic` | `[50, 768]` | 4.974e-6 | 1.078e-4 |
| `enc acoustic` | `[50, 256]` | 8.231e-6 | 1.124e-4 |
| `enc fused` | `[50, 1024]` | 7.503e-6 | 2.067e-4 |
| `enc codes` | `[50, 8]` | 400 / 400 exact | — |
| `dec quantized` | `[50, 1024]` | 1.703e-7 | 3.815e-6 |
| `dec fc2` | `[50, 256]` | 5.068e-7 | 7.248e-5 |
| `dec conv1` | `[50, 1024]` | 6.790e-7 | 1.183e-4 |
| `dec block_0`–`block_4` (worst) | `[32, C]` edges | 1.288e-6 | 2.289e-4 |
| `dec samples` | `[48000]` | 1.689e-6 | 3.487e-6 |

## The backbone on the CPU (2026-11-22)

`Qwen3Model` (`transformers` 5.19.0, eager full bidirectional attention, CPU F32)
plus `audio_embeddings` and `audio_heads` dequantized from `voicetut-tts-q8_0.gguf`
was run once at step 0 on a 53-row conditional Egyptian Arabic clone sequence
(25 prompt tokens + 16 reference frames from `omnivoice/codec/codes.u16` + 12
masked target frames) and the 12-row unconditional target canvas. The fixtures
live under `tests/fixtures/omnivoice/backbone/`.

`omnivoice-check MODEL.gguf trace --max-relative-rms 1e-4 --max-abs 2e-3` (gate
`omnivoice-trace-cpu`, tier `verify-cpu`, ReleaseFast on Apple M4 Pro):

| Stage | Conditional rel RMS / max \|Δ\| | Unconditional rel RMS / max \|Δ\| |
| --- | ---: | ---: |
| `embed` | 0.000e0 / 0.000e0 | 0.000e0 / 0.000e0 |
| `layer0` | 6.991e-7 / 2.861e-5 | 4.940e-7 / 1.526e-5 |
| `layer13` | 7.967e-7 / 2.747e-4 | 3.985e-7 / 4.578e-5 |
| `layer27` | 2.169e-6 / 6.738e-2 | 1.182e-6 / 3.906e-3 |
| `norm` | 8.915e-7 / 5.493e-4 | 8.877e-7 / 1.984e-4 |
| `logits` (`[12, 8200]`, 96/96 argmax equal) | 5.954e-7 / 3.052e-4 | 6.025e-7 / 3.204e-4 |

The unnormalized layer-27 residual stream reaches magnitude ~3×10⁴ before
`llm.norm`, so `max |Δ|` is checked on `embed`, `norm`, and `logits`, while
`rel RMS` (`≤ 1e-4`) covers every stage.

**CPU synthesis smoke test (2026-11-22)**: `omnivoice-check MODEL.gguf speak
--backend cpu --text "إزيك يا باشا؟ عامل إيه النهارده؟" --duration 2.0 --seed 0`
(ReleaseFast, Apple M4 Pro, 32 diffusion steps, `guidance_scale = 2.0`) produced
50 frames / 48,000 samples (2.00 s, RMS 0.1106, finite) in 240.9 s (RTF 120.46×;
open 189 ms). Transcribed by `nuclis transcribe --language ar` (`nemotron-3.5-asr-streaming-0.6b`):
`"إزيك بش عامل إيه النهار ده"`.

## On Metal (2026-11-22)

`omnivoice_metal.Plan` wraps all mapped `Q8_0` weights in place (`Backend.wrap`)
and packs the conditional and unconditional CFG sequences (`[S_cond ‖ S_uncond]`)
into one forward pass and one Metal command buffer per diffusion step using
`Backend.attentionSegmentsGrouped`, per-segment NeoX `ropeRows`, `addRmsNorm`,
`siluMul`, and `matmulTile` (with `audio_heads` evaluated only on the packed
target tail rows). `higgs_codec_metal.Decoder` executes all five DAC decoder
blocks and the final unclamped 7-tap convolution in one Metal command buffer via
`codec_metal.Device`, folding non-causal `ConvTranspose1d` into a 3-tap gather +
`matmulTile` + `addBiasRows` with no new Metal shaders.

**Backbone trace on Metal** (`omnivoice-check MODEL.gguf trace-metal --max-relative-rms 1e-4 --max-abs 2e-3`,
gate `omnivoice-trace-metal`, tier `verify`, ReleaseSafe on Apple M4 Pro):

| Stage | Conditional vs CPU rel RMS / max \|Δ\| | Unconditional vs CPU rel RMS / max \|Δ\| |
| --- | ---: | ---: |
| `embed` | 0.000e0 / 0.000e0 | 0.000e0 / 0.000e0 |
| `layer0` | 7.031e-7 / 2.861e-5 | 4.969e-7 / 1.526e-5 |
| `layer13` | 8.306e-7 / 3.662e-4 | 3.285e-7 / 3.052e-5 |
| `layer27` | 2.363e-6 / 6.641e-2 | 1.133e-6 / 3.662e-3 |
| `norm` | 8.824e-7 / 4.883e-4 | 9.120e-7 / 2.136e-4 |
| `logits` (`[12, 8200]`, 96/96 argmax equal) | 5.970e-7 / 3.204e-4 | 6.046e-7 / 3.204e-4 |
| Packed CFG (`[cond ‖ uncond]`, `S = 65`) | 5.970e-7 / 3.204e-4 | 6.026e-7 / 3.204e-4 |

Worst relative RMS across all stages is `2.363e-6`, worst normalized `max |Δ|`
is `4.883e-4`, `generate` cancellation returns `error.Canceled`, and `Plan.deinit`
leaves `created_bytes == 0` and `wrapped.count() == 0`.

**Higgs Audio V2 codec decode on Metal** (`omnivoice-check MODEL.gguf codec-metal --max-relative-rms 1e-4 --max-abs 2e-3`,
gate `omnivoice-codec-metal`, tier `verify`, ReleaseSafe on Apple M4 Pro):
50 frames → 48,000 samples (2.00 s) in **31.3 ms** (64× faster than real time),
`rel RMS = 1.878e-6` / `max |Δ| = 3.776e-6` against CPU `higgs_codec`, and
`rel RMS = 1.915e-6` / `max |Δ| = 3.040e-6` against the author trace, with two
init/decode/deinit lifecycle passes leaving zero leaked GPU buffers.

**Metal synthesis first-look (2026-11-22)**: `omnivoice-check MODEL.gguf speak
--backend metal --text "إزيك يا باشا؟ عامل إيه النهارده؟" --duration 2.0 --seed 0`
(ReleaseSafe, Apple M4 Pro, 32 packed CFG diffusion steps, `guidance_scale = 2.0`,
open 184 ms):

| Phase | Time | Notes |
| --- | ---: | --- |
| Diffusion backbone (32 packed CFG steps) | 1,742.9 ms | 54.5 ms/step (`S_cond = 75`, `S_uncond = 50`, packed `S = 125`) |
| Higgs Audio V2 codec decode (50 frames) | 31.4 ms | 1 command buffer → 48,000 samples (2.00 s) |
| **Total synthesis** | **1,774.5 ms** | **RTF 0.89×** (136× faster than CPU's 240.9 s; RMS 0.1106, finite) |

Transcribed by `nuclis transcribe --language ar` (`nemotron-3.5-asr-streaming-0.6b`):
`"إزيك بش عامل إيه النهار ده"` (identical to CPU).

## Round-trip and speed baseline (2026-11-22)

Measured on Apple M4 Pro (48 GiB, macOS 27.2.1), `ReleaseSafe`, Metal,
`voicetut-tts` (`voicetut-tts-q8_0.gguf`, 1.35 GB) via freshly built
`./zig-out/bin/nuclis` at `6b32f42` (before the catalogue pin).

### Egyptian Arabic round-trip (`omnivoice-roundtrip-metal`, 2026-11-22)

8 pinned sentences (`tests/fixtures/omnivoice-roundtrip.json`, 6 Egyptian
Arabic with `nuclis transcribe --language ar` and 2 Arabic–English
code-switching with `--language ar,en-US`), spoken by
`nuclis speak --model voicetut-tts --backend metal --seed 0` and scored by
`scripts/gate/wer.py` with Egyptian-aware Arabic normalization (stripping
diacritics and tatweel, folding `أ/إ/آ/ٱ→ا`, `ى→ي`, `ة→ه`, `ؤ→و`, `ئ→ي`):

- **Overall**: **15.42 % CER** (35 / 227 characters), **34.69 % WER** (17 / 49
  words), 26.7 s of audio in 27.5 s wall time (two fresh processes per sentence).
- Gate bound: `--max-cer 0.22`. Most character edits come from Nemotron ASR
  transliterating code-switched English words into Arabic script (`meeting` →
  `ميتنج`, `update` → `ابديت`, `deadline` → `ديد لاين`) and Egyptian spelling
  variants (`بكره` vs `بكرا`, `النهارده` vs `النهار ده`).

### Cold and warm Metal speed baseline (2026-11-22)

| Case | Mode | Audio | Synth | Speed | Diffusion / Codec | TTFA | Load | Peak RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Short Egyptian Arabic (cold) | auto-voice | 2.16 s | 1.88 s | 1.15× | 1,856 / 23 ms | 1,880 ms | 173 ms | 408 MiB |
| Short Egyptian Arabic (warm) | auto-voice | 2.16 s | 1.81 s | 1.19× | 1,783 / 29 ms | 1,813 ms | 176 ms | 408 MiB |
| Medium Egyptian Arabic | auto-voice | 6.19 s | 4.76 s | 1.30× | 4,685 / 74 ms | 4,759 ms | 179 ms | — |
| Arabic–English code-switching | auto-voice | 7.57 s | 6.11 s | 1.24× | 6,018 / 91 ms | 6,110 ms | 179 ms | — |
| Multi-sentence paragraph (2 chunks) | auto-voice (chunk-0 anchor) | 15.10 s | 15.27 s | 0.99× | 15,105 / 165 ms | 7,707 ms | 181 ms | — |
| Voice design (`male, young adult, moderate pitch`) | `--instruct` | 3.32 s | 2.89 s | 1.15× | 2,846 / 43 ms | 2,890 ms | 182 ms | — |
| Zero-shot clone, 2.16 s ref (first use) | `--voice` | 3.20 s | 4.02 s | 0.80× | 3,940 / 81 ms | 4,021 ms | 181 ms | 2,184 MiB |
| Zero-shot clone, 2.16 s ref (cached) | `--voice` | 3.20 s | 3.81 s | 0.84× | 3,769 / 45 ms | 3,814 ms | 185 ms | — |

Reference voice encoding (lazy CPU HuBERT-Base + DAC encoder on first use):
**1.0 s** for a 2.16 s clip (2,184 MiB peak RSS during encoder init); **0.0 ms**
on subsequent calls once cached in `voice.json`.

Compared against nuclis's other speech models on the same Apple M4 Pro:
`voicetut-tts` (1.35 GB, 408 MiB RSS) is smaller than `qwen3-tts-1.7b-customvoice`
(4.5 GB) and `drama-3` (20.8 GB) and runs faster than real time (1.15–1.30×) in
auto-voice and voice-design modes, with higher first-audio latency (one full
32-step diffusion chunk before decode) than autoregressive frame-streaming models.



