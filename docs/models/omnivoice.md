<!-- kind: card -->
# OmniVoice / VoiceTut-TTS

[VoiceTut-TTS](https://huggingface.co/mohammedaly22/VoiceTut-TTS-GGUF) is a
speech synthesizer built on the [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice)
architecture, fine-tuned for Egyptian Arabic (`arz`) and Arabic–English
code-switching. It ships as one self-contained `audio.cpp` GGUF
(`voicetut-tts-q8_0.gguf`, 1.26 GiB) carrying the backbone weights, the
Higgs Audio V2 audio tokenizer, and the embedded Qwen3 BPE tokenizer and
configuration files.

Licences: VoiceTut-TTS weights are Apache-2.0; the OmniVoice base is
CC-BY-NC; the Higgs Audio V2 tokenizer carries the Higgs Audio 2 Community
License.

The artifact inventory, container schema, and verified forward equations live
in [the record](../benchmarks/record/omnivoice.md#sources-2026-11-22), and its
pinned commit and SHA-256 in
[catalogue.md § Pinned commits and digests](../benchmarks/record/catalogue.md#pinned-commits-and-digests-2026-09-11).

## Architecture

| Part | Geometry and equations |
| --- | --- |
| Backbone | Qwen3-0.6B bidirectional masked-diffusion transformer: 28 layers, width 1,024, FFN 3,072 (SwiGLU / SiLU), 16 query / 8 KV heads × 128, Q and K per-head RMSNorm (`eps = 1e-6`), NeoX RoPE (`theta = 1e6`), no attention bias |
| Audio tokens | 8 codebooks × 1,025 entries (`0..1023` codes, `1024` mask id), packed into `weights/audio_embeddings.weight` `[1024, 8200]` and `weights/audio_heads.weight` `[1024, 8200]` with `weights/codebook_layer_offsets` `[0, 1025, 2050, 3075, 4100, 5125, 6150, 7175]` (`I64`) |
| Diffusion schedule | Iterative parallel unmasking: `num_step = 32`, `t_shift = 0.1`, classifier-free guidance `guidance_scale = 2.0`, `layer_penalty_factor = 5.0`, `position_temperature = 5.0`, `class_temperature = 0.0`, codebook loss weights `[8, 8, 6, 6, 4, 4, 2, 2]` |
| Audio codec | HiggsAudioV2Tokenizer (`bosonai/higgs-audio-v2-tokenizer`): 24 kHz output, 25 fps (960 samples/frame), 8 RVQ codebooks of 1,024 (codebook dim 64), DAC acoustic decoder (hidden 1,024, upsampling ratios `[8, 5, 4, 2, 3]`) and DAC + 12-layer HuBERT-Base semantic encoder (16 kHz semantic branch, 768 width) |

## Files and encodings

One GGUF, `voicetut-tts-q8_0.gguf` (1,350,264,224 B): GGUF v3, 16 metadata
keys, 840 tensors, directory 11,568,567 B. Its inventory is
`inference/src/models/fixtures/omnivoice-voicetut-q8_0.json`.

- **Container**: `general.architecture = "audiocpp"`,
  `audiocpp.model_spec.family = "omnivoice"`, `audiocpp.weight_type = "q8_0"`,
  `audiocpp.tensor_name_format = "native"`, `audiocpp.source_format = "packed"`.
- **Embedded sidecars**: six files stored as one lazy `uint8` array
  (`audiocpp.embedded_files.data`, 11,433,692 B) indexed by
  `audiocpp.embedded_files.names` and `audiocpp.embedded_files.offsets`:
  `config.json`, `tokenizer_config.json`, `tokenizer.json`,
  `chat_template.jinja`, `audio_tokenizer/config.json`, and
  `audio_tokenizer/preprocessor_config.json`.
- **Long tensor names**: codec tensor names exceeding 64 bytes are stored as
  `_audiocpp.2`…`_audiocpp.526` in the GGUF tensor table and mapped back to
  their native names via `audiocpp.tensor_names` (840 strings), paired with
  `audiocpp.tensor_ranks` and `audiocpp.tensor_shapes`.
- **Encodings**: 513 F32, 37 F16 (`weights/audio_embeddings.weight`,
  `weights/llm.embed_tokens.weight`, and codec projections), 289 Q8_0 (the
  28 Qwen3 backbone layers, `weights/audio_heads.weight`, and HuBERT
  projections), and 1 I64 (`weights/codebook_layer_offsets`).

## Voice modes

VoiceTut-TTS has no bundled speaker clips. One file serves three modes:

- **Auto-voice**: no reference clip and no voice-design tags; the model
  samples a voice per seed.
- **Voice design**: comma-separated OmniVoice attribute tags (gender, age,
  pitch, style, English accent, or Chinese dialect) inside
  `<|instruct_start|>…<|instruct_end|>`, mutually exclusive with a reference
  clip.
- **Zero-shot voice cloning**: a saved reference clip and transcript
  (`nuclis voice add`), encoded through the Higgs Audio V2 tokenizer.

## Prompt format and tokenizer

No chat profile. `omnivoice_text` builds OmniVoice's conditional prompt
verbatim (no implicit text normalizer):

```text
[<|denoise|>]<|lang_start|>{lang}<|lang_end|><|instruct_start|>{instruct}<|instruct_end|><|text_start|>[{ref_text} ]{text}<|text_end|>
```

- **Control tokens**: seven added special tokens (`151669` `<|denoise|>` through
  `151675` `<|text_end|>`), matched before NFC normalization by the embedded
  Qwen3 BPE tokenizer (`hf_json.Shape.qwen2`: `NFC` +
  `Sequence([Split(qwen2 regex, Isolated), ByteLevel(use_regex=false)])` +
  `ByteLevel` decoder, 151,643 vocab, 151,387 merges, 33 added tokens).
- **Language**: defaults to `arz` (Egyptian Arabic); `None` leaves the language
  unspecified.
- **Instruct tags**: validated strictly against OmniVoice's five categories, at
  most one tag per category — gender (`male`, `female`, `男`, `女`), age
  (`child`, `teenager`, `young adult`, `middle-aged`, `elderly`, and Chinese
  equivalents), pitch (`very low pitch`…`very high pitch`), style (`whisper`,
  `耳语`), and English accents / Chinese dialects. Unknown or conflicting tags
  return a typed error, and `instruct` is mutually exclusive with a reference
  clip.
- **Duration and chunking**: explicit `duration_sec` overrides `speed` and
  automatic script-aware frame estimation (25 fps, bounded to 10–1,500 frames).
  Long text splits on Arabic and English sentence ends (`.!?;…`, newlines,
  `؟؛۔`) without breaking `<|...|>` or `[...]` tags.
- **Gates**: `omnivoice-vocabulary` (`make gate NAME=omnivoice-vocabulary`)
  checks the full structural binding, embedded `config.json`,
  `weights/codebook_layer_offsets`, and pinned English, Egyptian Arabic,
  code-switching, diacritics, and prompt token sequences.
  `omnivoice-codec-cpu` (`make gate NAME=omnivoice-codec-cpu`, tier
  `verify-cpu`) checks `higgs_codec_encoder` and `higgs_codec` on the CPU
  against the 50-frame `HiggsAudioV2Tokenizer` trace (400/400 RVQ codes exact,
  samples within `1.7e-6` relative RMS,
  [record](../benchmarks/record/omnivoice.md#the-codec-on-the-cpu-2026-11-22)).


