<!-- kind: card -->
# EmbeddingGemma 2

[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2)
(Google DeepMind, Apache-2.0) maps text, images, audio, and mixtures of them
in one input to a single 768-dimensional, L2-normalized vector. It decodes
nothing: it is the embedding kind beside the language and decision models.
Catalogue entry `embeddinggemma-2`; `nuclis embed` and `POST /v1/embeddings`
([api-embeddings.md](../guide/api-embeddings.md)) run it. The binding is
[inference/src/models/embeddinggemma.zig](../../inference/src/models/embeddinggemma.zig),
the forward `embeddinggemma_runtime.zig` and `embeddinggemma_metal.zig`, the
input handling [inference/src/embed.zig](../../inference/src/embed.zig).
How every fact here was read and checked is in
[the record](../benchmarks/record/embeddinggemma.md).

## Architecture

A 24-block Gemma 4–style text model with bidirectional attention; the
vision and audio encoders are Gemma 4's, in one mmproj file.

| Fact | EmbeddingGemma 2 | Unlike Gemma 4 |
| --- | --- | --- |
| Blocks (sliding / global) | 24 (20 / 4; global at 5, 11, 17, 23) | |
| Width, FFN, output | 512, 2,048, 768 | an `output.weight` projection, not a tied head |
| Heads | 4 query; KV 2 (sliding, 256 wide) / 1 (global, 512 wide) | global layers have their own `attn_v` |
| Attention | bidirectional, no cache; sliding layers see `\|i − j\| ≤ 512` | a symmetric window (`sliding_window` 1,024 is its full width) |
| Per-layer inputs | 512 wide, from `per_layer_model_proj` alone | no token table, no `/√2` |
| RoPE | NeoX over the whole head, base 1e6 global / 1e4 sliding | no `rope_freqs` |
| Pooling | mean over every row (BOS, prefix, EOS included), then L2 | no logits, no soft-cap |
| Context | 8,192 tokens per input, every modality together | |

Per block, as Gemma 4: raw-weight RMS norms, Q/K norms, a weightless V norm,
attention scale 1.0, a tanh-GELU gated FFN, then the per-layer input
`x += rmsnorm_w(proj · (gelu(inp_gate · x) ⊙ ple[l]), post_norm)` and the
per-layer output scale. Token rows are scaled by √512; projector rows are
taken as they are.

## Files and encodings

The catalogue pins the Q8_0 text file (310 MB) and the BF16 mmproj (982 MB,
both encoders); text needs only the first. Q8_0 because embedding is one
prefill-only pass, so weight bandwidth barely matters, while stored vectors
pay quantization noise forever; the mmproj stays the original BF16. The
BF16 text file (558 MB, not catalogued) is what the `embeddinggemma-google-*`
gates measure the claim on.

**Precision: F32 only.** The activations exceed f16's range (the card warns
against float16), so residuals, activations, and attention stay F32, and
Metal runs every matrix (Q8_0, BF16, F32) on the generic tile with F32
operands; no half-operand tile runs this family. Why not bf16 either:
[record, 2026-10-09](../benchmarks/record/embeddinggemma.md#the-f16-hazard-2026-10-09).

## Tokenizer and task prefixes

The Gemma 4 vocabulary (`tokenizer.ggml.model = "gemma4"`); an input is
`<bos>` … `<eos>`. Text parts are **literal**: a control token's spelling in
a document is text, never the token, and media positions come from the
order of the parts, so content cannot change an input's framing.

No task is implied. With `--task` (or the API's `task`), the prefix opens
the input, before any media part (`Task.prefix` in
`src/embedding/request.zig`):

| Task | Prefix |
| --- | --- |
| `search_query`, `question_answering`, `fact_checking`, `code_retrieval` | `task: search result \| query: `, `task: question answering \| query: `, … |
| `classification`, `clustering`, `similarity` | `task: classification \| query: `, …, `task: sentence similarity \| query: ` |
| `document` | `title: {title or none} \| text: ` |

**Widths.** The leading 768, 512, 256, or 128 values, renormalized, are
trained embeddings (`embed.widths`); others are refused. Vectors of
different widths or files are not comparable, and each response names its
space (`embeddinggemma-2@6f1bd4ac6c5d/768` for the catalogue's file).

## The input contract

One input is an ordered list of parts and gives one vector, in one pass of
at most 8,192 tokens. An input over the limit is refused unless the request
truncates, which keeps the final EOS and reports the cut.

| Part | Becomes | Budget |
| --- | --- | --- |
| Text | its tokens | 1 per subword |
| Image | `<\|image>`, the projector rows, `<image\|>` | 70, 140, 280 (default), 560, or 1120 |
| Audio | `<\|audio>`, the projector rows, `<audio\|>` | 25 per second, at most 30 s (750) |

Inputs pack into Metal batches of at most 8,192 rows; packing changes
timing, never a vector, bit for bit.

## Images

Images follow Google's `Gemma4ImageProcessor`, not the chat path's
pipeline: the stored 8-bit values (alpha dropped, no colour management),
the largest 48-pixel grid of the image's ratio within the budget
(`gemma4.googleGrid`), torchvision's antialiased bicubic
(`preprocess.resizeWith(…, .torchvision)`), F32 patches `2x − 1`, and
`gelu_pytorch_tanh` in the encoder's FFN. The vision encoder (`gemma4v`, 16
blocks of 768, no clipped linears) is built on the first image and holds
918 MB on Metal.

## Audio

16 kHz mono, Google's log-mel frames, and the `gemma4a` conformer
([audio.md](../engine/audio.md)), whose layer-norm epsilon is read from this
file's header (1e-6). The encoder is built on the first clip and holds
658 MB on Metal. A 16 kHz mono 16-bit WAV is read exactly; other formats go
through AudioToolbox's resampler.

## What is verified

<!-- gen:gates-for:embeddinggemma -->
| Gate | Tier | Model | Evidence |
| --- | --- | --- | --- |
| `embeddinggemma-trace-cpu` | `verify-cpu` | embeddinggemma | [embeddinggemma.md](embeddinggemma.md) |
| `embeddinggemma-vectors-cpu` | `verify-cpu` | embeddinggemma | [embeddinggemma.md](embeddinggemma.md) |
| `embeddinggemma-google-cpu` | `verify-cpu` | embeddinggemma_bf16 | [embeddinggemma.md](embeddinggemma.md) |
| `embeddinggemma-image-cpu` | `verify-cpu` | embeddinggemma_bf16 | [embeddinggemma.md](embeddinggemma.md) |
| `embeddinggemma-audio-cpu` | `verify-cpu` | embeddinggemma_bf16 | [embeddinggemma.md](embeddinggemma.md) |
| `embeddinggemma-vectors-metal` | `verify` | embeddinggemma | [embeddinggemma.md](embeddinggemma.md) |
| `embeddinggemma-google-metal` | `verify` | embeddinggemma_bf16 | [embeddinggemma.md](embeddinggemma.md) |
| `embeddinggemma-image-metal` | `verify` | embeddinggemma_bf16 | [embeddinggemma.md](embeddinggemma.md) |
| `embeddinggemma-audio-metal` | `verify` | embeddinggemma_bf16 | [embeddinggemma.md](embeddinggemma.md) |
<!-- /gen -->

The accuracy claim is the BF16 file against Google's float32 pipeline:
`1 − cos` about 4e-12 on text, images, and audio, the size of summation
order. The Q8_0 file costs about 5e-5 of cosine on text.

## Numbers

| Workload | Result | Record |
| --- | --- | --- |
| Metal, Q8_0, 64 inputs × 256 tokens | 1,671 ms, 38.3 inputs/s | [2026-10-09](../benchmarks/record/embeddinggemma.md#the-metal-plan-2026-10-09) |
| Metal, Q8_0, one 512-token input | 59.2 ms | same |
| Metal, Q8_0, one 8,192-token input | 1,754 ms | same |
| One image at 280 tokens (decode, resize, encode) | about 460 ms | [2026-10-09](../benchmarks/record/embeddinggemma.md#images-2026-10-09) |
| One 4.8 s clip (decode, mel, encode) | 130 ms | [2026-10-09](../benchmarks/record/embeddinggemma.md#audio-2026-10-09) |
| Linux CPU, 4 vCPU, `serve`, 32 texts of 17,788 tokens | 49.6 s, 359 tokens/s | [2026-10-31](../benchmarks/record/embeddinggemma.md#every-core-on-linux-2026-10-31) |

M4 Pro 48 GB unless noted. About 80 % of a Metal batch is F32 matmul at the
generic tile's ceiling.

## Known limits

- Video is not embedded.
- F32 operands cost the 20–25 % a half-operand tile would save.
- On Linux, images and audio are refused (`unsupported_feature`): the
  decoders are the macOS bridges.
- A text containing a control token's spelling differs from Google's
  vector, which parses it.
- Image budgets are the five Google's processor takes; a clip over 30 s is
  refused unless truncated.
