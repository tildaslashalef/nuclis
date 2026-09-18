# Third-party notices

nuclis is written independently. External implementations are used as
**references and validation oracles**, not as sources to copy: see the policy in
[AGENTS.md](AGENTS.md#shared-engineering-rules). This file records every piece of
third-party material that is actually present in the repository, with its
origin and license, and lists the references that were consulted but not copied.
License texts are linked on the web; no separate license text files live in the
repository.

## Material present in the repository

The material falls into three kinds: the model prompt templates reimplemented
in Zig, the format-defining data any decoder of these formats must contain, and
the presentation palette the terminal surface draws with.

### Model prompt templates

#### Qwen3.8 — Apache License 2.0

`inference/src/profiles/qwen38.zig` reimplements, in Zig, the
reasoning-preserving subset of the chat template embedded in
`Qwen3.8-27B-UD-Q4_K_M.gguf` (Qwen / Unsloth). The pinned fixtures in
`inference/src/profiles/fixtures/qwen38-text.json` and `qwen38-tools.json` were
rendered from that template. Source:
<https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/4ca720788d1e01f1bff70c033e0d0028fd02e502>,
model SHA-256 `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482`.
Modifications: explicit size limits, ownership, UTF-8 checks, and narrower typed
inputs; multimodal rendering and assistant prefill are not implemented.
License: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt).

#### Gemma 4 — as licensed by the artifact

`inference/src/profiles/gemma4.zig` reimplements, in Zig, the text and
tool-calling subset (system, user, assistant; thinking switch; tool
declarations, calls, and responses; no media) of the chat template embedded
in `gemma-4-12b-it-UD-Q4_K_XL.gguf` (Google / Unsloth; the QAT file carries
the same template). The template text is not in the tree; the pinned fixtures
in `inference/src/profiles/fixtures/gemma4-text.json` and `gemma4-tools.json`
were rendered from it by the reference server. Source:
<https://huggingface.co/unsloth/gemma-4-12b-it-GGUF/tree/fc034cfff751157913579611efad8462ac1be606>,
model SHA-256 `90fd944d227e9d9b68e7e2c7d5b57b79d4c66ed521b0919fbbd932cf834f6f8e`,
template SHA-256 `845f1ee48e39fc942fe190da9df6a1c5db229e17a96ea08966ad1c9274e73d1b`.
The artifact header declares `general.license = apache-2.0`; the upstream
Google model is distributed under the terms stated on its model card
(<https://huggingface.co/google/gemma-4-12b-it>), which govern the weights
and are not affected by this reimplementation. Modifications: explicit
size limits, ownership, UTF-8 checks, narrower typed inputs, `<bos>`
rendered as text; tools, multimodal rendering, and assistant prefill are
not implemented.

### Format-defining data

#### GGML storage-format constants — MIT (ggml authors)

Two lookup tables define GGML quantization formats and exist nowhere but the
ggml/llama.cpp sources; any decoder of these formats must contain the same
numbers. They were extracted from llama.cpp revision
`7620399f58aebfd2196b74021f9581bcf7218cb9`, `ggml/src/ggml-common.h`:

- the 512-entry IQ3_S codebook, `inference/src/quant/iq3-grid.zig`
  (also emitted into the Metal shader source at compile time);
- the 16-entry IQ4_NL value table in `inference/src/quant/decode.zig` and
  `inference/src/backends/metal/dequant.metal`.

The decoders themselves (Zig and MSL) are independent implementations of the
documented block layouts, verified against pinned fixtures. The tables are
retained with the ggml copyright notice as a conservative attribution:
Copyright (c) 2023-2026 The ggml authors, MIT License,
[llama.cpp LICENSE](https://github.com/ggml-org/llama.cpp/blob/master/LICENSE).

The group-128 ternary encodings PQ2_0 and PTQ1_0 (ids 142, 143) are defined
by the PrismML fork of llama.cpp (`PrismML-Eng/llama.cpp`, MIT, revision
`5d80cff0b8cb9f2bf823cfc4e71e3abb97f290d6`), read as a format contract and
used as the numerical oracle for the committed
[`ternary.json`](inference/src/quant/fixtures/ternary.json) fixture; no
table or code from it is present in the tree.

#### GPT-2 byte-level alphabet — format fact

`inference/src/tokenizer/bpe.zig` maps the 256 byte values onto Unicode
scalars in the order defined by OpenAI's GPT-2 byte-level BPE (the mapping every
GPT-2-style GGUF vocabulary is stored in). The mapping is a property of the
vocabulary format; the merge algorithm is independently implemented.

#### Unicode character data — Unicode License

`inference/src/tokenizer/unicode-ranges.bin` is a compact table of letter,
mark, number, and whitespace membership derived from the Unicode Character
Database, obtained through llama.cpp's generated `unicode-data.cpp` at the
revision above (`scripts/tokenizer-unicode.py`). Unicode data is
Copyright © Unicode, Inc., licensed under the Unicode License
(<https://www.unicode.org/license.txt>). The pre-tokenizer in
`inference/src/tokenizer/pre.zig` implements the `qwen35` splitting rules
declared by the model's tokenizer; it is independently written.

`src/tui/graphemes_table.zig` is generated by `scripts/grapheme-table.py` from
UCD 17.0.0 (Grapheme_Cluster_Break and Indic_Conjunct_Break, East Asian Width,
Default_Ignorable_Code_Point, Emoji_Presentation/Emoji_Modifier, and
Extended_Pictographic). `src/tui/fixtures/GraphemeBreakTest.txt` is the
conformance test file for that version. `src/tui/graphemes.zig` implements the
UAX #29 boundary rules and the UAX #11 width policy over that data; it is
independently written.

### Presentation data

#### Gruvbox dark palette — MIT (Pavel Pertsev)

`src/tui/theme.zig` carries the colour values of the Gruvbox dark palette
(morhetz/gruvbox, <https://github.com/morhetz/gruvbox>): the background and
foreground scales and the eight accent pairs, each with the truecolor value
and the 256-colour approximation published in `colors/gruvbox.vim`. A palette
is a list of colours, not code; nuclis's role table, level detection, and
sequence composition are independently written, and the values are retained
with attribution because they are recognisably one designer's palette.
Copyright (c) 2026 Pavel Pertsev, MIT License,
[gruvbox LICENSE](https://github.com/morhetz/gruvbox/blob/master/LICENSE.md).

## References consulted, not copied

These projects were read to understand formats and algorithms and are used at
development time to generate pinned fixtures and comparison traces. No source
code from them is present in nuclis.

- **llama.cpp** `7620399f58aebfd2196b74021f9581bcf7218cb9` (MIT) — GGUF format,
  quantization semantics, tokenizer behavior, Metal reference performance.
  Built under `.zig-cache/reference/` by [docs/reference/reference-baseline.md](docs/reference/reference-baseline.md);
  `scripts/*.py` and `scripts/reference-generation.cpp` call its public API.
  Fixtures under `inference/src/**/fixtures/` and traces are outputs of
  running it, not copies of it.
- **ds4** by antirez (<https://github.com/antirez/ds4>) — Objective-C Metal glue
  pattern behind a C interface, named in the nuclis specification as an
  implementation reference.
- Model documentation: the Qwen3.8 and Gemma 4 model cards and configuration
  on Hugging Face.

## The project's own licence

nuclis is MIT-licensed ([LICENSE](LICENSE)). This file lists material from
elsewhere that exists in the tree — format-defining constants, prompt
templates, and palettes — with a link to each licence; it is not a list of
dependencies, because the engine has none.
