# Third-party notices

nuclis is written independently: other implementations are read as format
definitions, never copied ([AGENTS.md](AGENTS.md#engineering-rules)). This
file lists the third-party material present in the repository, with its
origin and licence, and the projects consulted. Licence texts are linked,
not copied into the tree.

## Material in the repository

### Prompt templates, reimplemented in Zig

| Profile | Template from | Fixtures | Licence |
| --- | --- | --- | --- |
| `inference/src/profiles/qwen38.zig` | the template in [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/ce331efc48d5722b4b8041f72793e530eae18d9b), SHA-256 `12827f24…` | `qwen38-text.json`, `qwen38-tools.json` | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0.txt) |
| `inference/src/profiles/gemma4.zig` | the template in [unsloth/gemma-4-12b-it-GGUF](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF/tree/fc034cfff751157913579611efad8462ac1be606) and the QAT files, SHA-256 `845f1ee4…` | `gemma4-text.json`, `gemma4-tools.json` | the artifact declares Apache-2.0; the weights follow [Google's model card](https://huggingface.co/google/gemma-4-12b-it) |
| `inference/src/profiles/gemma4_e.zig` | the E-series revision in `gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf`, SHA-256 `241c50d8…` | `gemma4_e-text.json` | as Gemma 4 |
| `inference/src/profiles/muse_glimmer.zig` | the template in [unsloth/Muse-Glimmer-30B-GGUF](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF/tree/faa5b025c584459c13febfa5c59883516710ae39), SHA-256 `114f55eb…` | `muse_glimmer-text.json`, `muse_glimmer-tools.json` | none declared; the weights follow the model card |
| `inference/src/profiles/clef.zig`, `inference/src/models/clef.zig` | the record encoding, fixed prompt text, and `JointSchemaHead.forward` of Cloudflare's [`joint_schema_model.py`](https://huggingface.co/Cloudflare/clef-flash/tree/17f0b0ad64efb65d273590632833508766b2aae6) | `inference/src/models/fixtures/clef/` | [Apache-2.0](https://huggingface.co/Cloudflare/clef-flash/blob/17f0b0ad64efb65d273590632833508766b2aae6/LICENSE) |

Profile fixtures are in `inference/src/profiles/fixtures/`. The Gemma 4 and
Muse Glimmer fixtures were rendered by llama.cpp's server from the template
before 2026-10-14; the template text itself is not in the tree. Every
reimplementation adds size limits, ownership, UTF-8 checks, and typed
inputs; none implements assistant prefill.

### Format-defining data

- **GGML quantization tables** (MIT, Copyright (c) 2023-2026 The ggml
  authors, [licence](https://github.com/ggml-org/llama.cpp/blob/master/LICENSE)):
  the IQ3_S codebook (`inference/src/quant/iq3-grid.zig`, also emitted into
  the Metal source), the IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, and IQ1_S codebooks
  (`inference/src/quant/iq-codebooks.zig`; the IQ2 sign table is computed),
  and the IQ4_NL values (`inference/src/quant/decode.zig`,
  `inference/src/backends/metal/dequant.metal`), from llama.cpp `7620399f`
  `ggml/src/ggml-common.h`. Any decoder of these formats must hold the same
  numbers; the decoders are written independently.
- **PQ2_0 and PTQ1_0** (ids 142, 143) are defined by the PrismML fork of
  llama.cpp (MIT, `5d80cff0`), read as a format contract; its output is the
  `inference/src/quant/fixtures/ternary.json` fixture. No code or table from
  it is in the tree.
- **Safetensors** dtype names and header layout, from the
  [format specification](https://github.com/huggingface/safetensors)
  (Apache-2.0), in `inference/src/formats/safetensors.zig`.
- **GPT-2's byte-to-Unicode alphabet**, the order every GPT-2-style
  vocabulary is stored in, in `inference/src/tokenizer/bpe.zig`.
- **Unicode Character Database** ([Unicode License](https://www.unicode.org/license.txt),
  Copyright © Unicode, Inc.): `inference/src/tokenizer/unicode-ranges.bin`
  (character classes, by way of llama.cpp's `unicode-data.cpp` at the
  revision above); `inference/src/tokenizer/nfc_table.zig` and
  `src/tui/graphemes_table.zig`, generated from UCD 17.0.0 by
  `scripts/fixtures/tokenizer-nfc.py` and `scripts/fixtures/grapheme-table.py`;
  and `src/tui/fixtures/GraphemeBreakTest.txt`, the conformance file. The
  pre-tokenizers, NFC, and grapheme rules over that data are written
  independently.

### Presentation

- **Gruvbox dark palette** (MIT, Copyright (c) 2026 Pavel Pertsev,
  [licence](https://github.com/morhetz/gruvbox/blob/master/LICENSE.md)): the
  colour values in `src/tui/theme.zig` and the `--t-*` colours of
  `site/style.css`.
- **Fonts** ([SIL OFL 1.1](https://openfontlicense.org/open-font-license-official-text/),
  no Reserved Font Name), from [google/fonts `9710da1`](https://github.com/google/fonts/tree/9710da1eacb3be272583c3224dcb70f9da6eadbb/ofl),
  narrowed and subset with `fonttools`: `site/fonts/archivo-v1.woff2`
  (Archivo, Copyright 2020 The Archivo Project Authors) and
  `site/fonts/jetbrains-mono-v1.woff2` (JetBrains Mono, Copyright 2020 The
  JetBrains Mono Project Authors).

### Speech data

Both [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); the clips
are cut and rounded to 16-bit PCM.

- **LibriSpeech** (Panayotov, Chen, Povey, Khudanpur, 2015;
  [OpenSLR 12](https://www.openslr.org/12)):
  `tests/fixtures/nemotron-asr/librispeech/clip.wav` (test-clean
  `121-121726-0000`) and 40 test-clean transcripts in
  `tests/fixtures/speech-corpus.json`.
- **FLEURS** (Conneau et al., 2022;
  [google/fleurs `70bb2e84`](https://huggingface.co/datasets/google/fleurs/tree/70bb2e84b976b7e960aa89f1c648e09c59f894dd)):
  `tests/fixtures/nemotron-asr/fleurs-de/clip.wav` (German test sentence
  1660) and 20 transcripts in German, French, Spanish, and Russian in the
  same corpus file.

## Projects consulted, not copied

No code from these is in nuclis. Where fixtures came from running one, they
are its outputs, pinned before nuclis became its own reference
([docs/models/README.md § 4](docs/models/README.md#4-validate-against-nuclis-itself)).

| Project | Read for | Fixtures from running it |
| --- | --- | --- |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) `7620399f` (MIT) | GGUF, the quantization formats, tokenizer behaviour; the numerical and speed reference until 2026-10-14 | traces and token arrays under `tests/fixtures/` and `inference/src/**/fixtures/` |
| [ds4](https://github.com/antirez/ds4) by antirez | the Objective-C Metal bridge behind a C interface | none |
| [laya](https://github.com/NandhaKishorM/laya) 0.3.20 (Apache-2.0) | the decision model's inputs, forward, calibration, tokenizer | `inference/src/models/fixtures/laya/`, `laya-multilingual/` |
| clef-flash `joint_schema_model.py` (Apache-2.0, `17f0b0ad`) | the sequence and the head | `inference/src/models/fixtures/clef/` |
| [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) `022e286b` (Apache-2.0), with Transformers 4.57.3 | the talker, code predictor, codec; the tokenizer's split pattern (`inference/src/tokenizer/cased.zig`) | `tests/fixtures/qwen3-tts/` |
| [fish-speech](https://github.com/fishaudio/fish-speech) `0f8a12e8` (Fish Audio Community License 1.0) | Drama 3's backbone, codebook tower, prompt, sampler, codec | `tests/fixtures/drama3/` |
| [transformers](https://github.com/huggingface/transformers) `c0d024fb` (Apache-2.0) | Nemotron 3.5 ASR: features, encoder, streaming, transducer | `tests/fixtures/nemotron-asr/` |

Drama 3's weights are under the Fish Audio Community License: no hosted
service for third parties without Fish Audio's agreement, consent from
anyone whose voice is cloned, and "Built with Fish Audio" on
redistribution ([model card](https://huggingface.co/fishaudio/drama-3)).
nuclis pulls weights; it ships none.

## The project's licence

nuclis is MIT ([LICENSE](LICENSE)). The engine has no dependencies; this
file lists material, not packages.
