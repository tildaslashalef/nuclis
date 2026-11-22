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
  (`upsampling_ratios = [8, 5, 4, 2, 3]`, product 960) to 24 kHz mono PCM.
  Reference clips are encoded by the dual acoustic (DAC, 24 kHz) + semantic
  (HuBERT-Base 12-layer, 16 kHz) encoder into `[8, T_ref]` codes.
