# EmbeddingGemma 2

[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2)
(Google DeepMind, Apache-2.0) is an embedding model. It maps text, images,
audio and video (and mixtures of them in one input) to a single
768-dimensional, L2-normalized vector. It decodes nothing and answers no
question, so it is a third kind of model beside the language models and
the decision models. The text model is a 24-block Gemma 4–style
transformer with bidirectional attention; the vision and audio encoders
are Gemma 4's, in one mmproj file.

Everything here was read on 2026-10-08 and 2026-10-09 from these sources:
the pinned files, through `scripts/gguf-inventory.py`; Google's repository
at the pinned revision (`config.json`, `config_sentence_transformers.json`,
`processor_config.json`, the card); and the second llama.cpp checkout
`b11514` (`de7fa0a3c`; `src/models/gemma-embedding2.cpp`, `conversion/gemma.py`,
`src/llama-hparams.h`, `tools/mtmd/`). The llama.cpp checkout is read for
semantics and used as an oracle, never copied. Where a fact comes from a
reference rather than the file, the sentence says so.

**Contents.**

- [Artifacts](#artifacts)
- [Metadata (`gemma-embedding2.*`)](#metadata-gemma-embedding2)
- [Tensors (413)](#tensors-413)
- [Forward pass (from the reference graph, `gemma-embedding2.cpp`)](#forward-pass-from-the-reference-graph-gemma-embedding2cpp)
- [Tokenizer and task prefixes](#tokenizer-and-task-prefixes)
- [Matryoshka widths](#matryoshka-widths)
- [The f16 hazard](#the-f16-hazard)
- [The mmproj (`clip`, 963 tensors)](#the-mmproj-clip-963-tensors)
- [The input contract](#the-input-contract)
- [Reference oracle status](#reference-oracle-status)
- [CPU reference against the oracles (2026-10-09)](#cpu-reference-against-the-oracles-2026-10-09)
- [The Metal plan (2026-10-09)](#the-metal-plan-2026-10-09)
- [`nuclis embed` (2026-10-09)](#nuclis-embed-2026-10-09)
- [Images (2026-10-09)](#images-2026-10-09)
- [Audio (2026-10-09)](#audio-2026-10-09)

## Artifacts

| Repository | Commit | File | Size (B) | SHA-256 |
| --- | --- | --- | ---: | --- |
| `unsloth/embeddinggemma-2-GGUF` | `031f0d4b35536f69ab3509d4893c923264fcf253` | `embeddinggemma-2-Q8_0.gguf` | 309,855,520 | `6f1bd4ac6c5df7444f9cca7ca36cafe6cfa34cd6f49fefb1e0b4be8143aed8bc` |
| | | `embeddinggemma-2-BF16.gguf` | 557,950,240 | `f315cbbb30dd487e44d501c8902abe88808755e43753a96beed1964f0a48aa4f` |
| | | `mmproj-BF16.gguf` | 982,074,880 | `995aaa56e88b9b631f651861d659b728a625ccc02a238be37cff56cf11dc0032` |

All three were pulled and verified by `nuclis model pull <repo> --file
<name> --revision <commit>` on 2026-10-09, with their sidecars beside them
under `~/.nuclis/models/unsloth/embeddinggemma-2-GGUF/`; the same rows are in
[catalogue.md § Pinned commits and digests](catalogue.md#pinned-commits-and-digests-2026-09-11).
The original checkpoint, `google/embeddinggemma-2` (safetensors,
sentence-transformers, not gated), is pinned at
`914f7f89142e33e77833254d9c9b90c3cef7303b`. It is the semantic oracle's
input, not a file nuclis runs.

**The quantization (decided 2026-10-08): Q8_0 text, BF16 mmproj.**

- Embedding is one prefill-only pass over a small transformer, so weight
  bandwidth barely matters. Half of the Q8_0 file is `token_embd`
  (262144 × 512), which is gathered, not multiplied.
- Stored vectors outlive the run, so quantization noise is paid in every
  row of an index.
- Activations overflow f16 ([below](#the-f16-hazard)), and the K-quant
  prefill tiles hold activations as half. Q8_0 runs on the generic
  F32-operand `nu_matmul`.
- The mmproj in BF16 is the original weight, as every other catalogue
  entry's mmproj is; the repository's F16 is the format the card warns
  against.
- The BF16 text file is pinned to measure the choice against, together
  with Google's f32 vectors (the acceptance record, when it lands).

The inventories are the fixtures
`inference/src/models/fixtures/embeddinggemma-2-q8_0.json` (48 keys, 413
tensors; the tensor directory is 15,790,616 B, almost all of it the
vocabulary) and `inference/src/vision/fixtures/embeddinggemma-2-mmproj.json`
(32 keys, 963 tensors).

## Metadata (`gemma-embedding2.*`)

| Key | Value |
| --- | --- |
| `general.architecture` | `gemma-embedding2` |
| `general.size_label` | `271M` (the card: 270M text, 170M vision, 300M audio) |
| `block_count` | 24 |
| `context_length` | 262144 (the base model's; the card's limit is **8192**, shared by every modality) |
| `embedding_length` | 512 |
| `embedding_length_out` | 768 |
| `feed_forward_length` | 2048 |
| `embedding_length_per_layer_input` | 512 |
| `attention.head_count` | 4 |
| `attention.head_count_kv` | array of 24: `2` on sliding layers, `1` on global layers |
| `attention.sliding_window_pattern` | array of 24 booleans: false on layers 5, 11, 17, 23 |
| `attention.sliding_window` | 1024, the **full width of a symmetric window** (below) |
| `attention.key_length` / `value_length` | 512 (global layers) |
| `attention.key_length_swa` / `value_length_swa` | 256 (sliding layers) |
| `attention.shared_kv_layers` | 0 |
| `attention.causal` | false |
| `attention.layer_norm_rms_epsilon` | 1e-6 |
| `rope.dimension_count` / `dimension_count_swa` | 512 / 256 (rotate the whole head) |
| `rope.freq_base` / `freq_base_swa` | 1,000,000 / 10,000 |
| `pooling_type` | 1 (mean) |
| `general.file_type` | 7 (Q8_0) |

There is no `final_logit_softcapping` (there are no logits) and no
`rope_freqs` tensor. The converter (`EmbeddingGemma2Model`, a subclass of
`Gemma4Model`) writes `sliding_window` as twice Google's `sliding_window`
(512), because Google's window extends 512 positions to each side. It
also returns no extra tensors ("default rope on all layers").

**The window.** llama.cpp's `LLAMA_SWA_TYPE_SYMMETRIC` masks key `j`
from query `i` when `j − i < −1024/2` or `j − i > 1024/2`. Visible means
|i − j| ≤ 512: 512 positions apart is visible, 513 is not. This is the
`window` semantics of `Backend.attentionSegments`; our generation plans'
windows are one-sided.

## Tensors (413)

Non-block tensors:

| Tensor | Shape | Encoding (Q8_0 file) |
| --- | --- | --- |
| `token_embd.weight` | [512, 262144] | Q8_0 |
| `per_layer_model_proj.weight` | [512, 12288] = 24 × 512 | **BF16** |
| `per_layer_proj_norm.weight` | [512] | F32 |
| `output_norm.weight` | [512] | F32 |
| `output.weight` | [512, 768] | Q8_0 |

There is **no `per_layer_token_embd`**: the per-layer inputs come only from
the projection. `output.weight` is a separate projection to the
embedding width, not a tied head.

Per block (17 tensors):

| Tensor | Sliding layer | Global layer |
| --- | --- | --- |
| `attn_norm`, `post_attention_norm`, `ffn_norm`, `post_ffw_norm`, `post_norm` | [512] F32 | same |
| `attn_q.weight` | [512, 1024] = 4 × 256 | [512, 2048] = 4 × 512 |
| `attn_k.weight`, `attn_v.weight` | [512, 512] = 2 × 256 | [512, 512] = 1 × 512 |
| `attn_output.weight` | [1024, 512] | [2048, 512] |
| `attn_q_norm`, `attn_k_norm` | [256] F32 | [512] F32 |
| `ffn_gate.weight`, `ffn_up.weight` | [512, 2048] | same |
| `ffn_down.weight` | [2048, 512] | same |
| `inp_gate.weight` | [512, 512] | same |
| `proj.weight` | [512, 512] | same |
| `layer_output_scale.weight` | [1] F32 | same |

Unlike Gemma 4's global layers, these have their own `attn_v`. Every
matrix is Q8_0 except `per_layer_model_proj` (BF16). Norms and scales are
F32.

**Values read from the file.** Norm weights are stored **raw**:
`Gemma4Model.norm_shift` returns 0, and the reference's `build_norm` is
`rms_norm(x) · w`. On layer 0, `attn_norm` ranges from 4.84 to 39.25
(mean 6.57); `attn_q_norm` is the constant 1.0234 and `attn_k_norm` the
constant 0.1221. `output_norm` ranges from −1.81 to 21.5, and
`per_layer_proj_norm` from 4.88 to 52.75. `layer_output_scale` per layer
0…23 is 0.3906 0.1689 0.9375 0.8711 0.6914 0.457 0.2256 0.9141 0.8672
0.8086 0.5547 0.3809 0.7734 0.8047 0.5 0.7383 0.3594 0.9492 0.7031 0.9688
0.9609 0.9531 0.8711 0.1147.

## Forward pass (from the reference graph, `gemma-embedding2.cpp`)

Notation: `n` rows, width `d` = 512, `rmsnorm(x)` = x / √(mean(x²) + 1e-6),
`rmsnorm_w(x)` = rmsnorm(x) · w, `gelu` the tanh approximation (`ggml_gelu`;
Google's `gelu_pytorch_tanh`).

1. **Input rows.** A token row is `token_embd[t] · √512`. A projector row
   (an image or audio soft token) is taken **as is**, not scaled (the
   reference scales only the rows it gathers). Call the result `x0`.
2. **Per-layer inputs**, once from `x0` for every row, projector rows
   included: `P = per_layer_model_proj · x0 / √512` (12288 wide), viewed as
   24 slices of 512, each `ple[l] = rmsnorm_w(P[l], per_layer_proj_norm)`.
   There is no token-table term and no `/√2` (Gemma 4 E4B has both).
3. **Each block `l`**, with `x` starting at `x0`:
   - `h = rmsnorm_w(x, attn_norm)`; `q, k, v = Wq h, Wk h, Wv h`, split
     into heads (4 query heads; 2 or 1 KV heads).
   - `q = rmsnorm_w(q, attn_q_norm)` and `k = rmsnorm_w(k, attn_k_norm)`
     per head; `v = rmsnorm(v)` per head, **without a weight**.
   - NEOX RoPE (split halves, `LLAMA_ROPE_TYPE_NEOX`) over the whole head
     at the position, base 1e6 on global layers and 1e4 on sliding ones.
   - Attention with scale **1.0** (the Q/K norms do the scaling), grouped
     query, no cache. Every key is visible on global layers; on sliding
     layers, keys with |i − j| ≤ 512. Then `Wo`.
   - `x = x + rmsnorm_w(attn, post_attention_norm)`.
   - `f = Wdown (gelu(Wgate h') ⊙ Wup h')` with `h' = rmsnorm_w(x, ffn_norm)`;
     `x = x + rmsnorm_w(f, post_ffw_norm)`.
   - `x = x + rmsnorm_w(proj · (gelu(inp_gate · x) ⊙ ple[l]), post_norm)`.
   - `x = x · layer_output_scale`.
4. **The vector.** `y = output · rmsnorm_w(x, output_norm)` per row (768
   wide), then the mean over **every** row (BOS, task prefix and EOS
   included; sentence-transformers' `include_prompt: true`), then L2
   normalization. The projection has no bias, so projecting before or
   after the mean is the same; the reference projects per row.

## Tokenizer and task prefixes

`tokenizer.ggml.model` is `gemma4`: the 262144-token Gemma 4 vocabulary
with its merges, `add_space_prefix` false. Both `add_bos_token` and
`add_eos_token` are true (BOS 2, EOS 1, unknown 3, pad 0, mask 4), so an
input is `<bos>` text `<eos>`. The oracle confirms it: "hello world" is
`2 23391 1902 1`. The chat template (1016 B, SHA-256 `4b852efc…d85e`) only
concatenates text parts and places `<|image|>`, `<|audio|>` or `<|video|>`
at each media part. It adds no turn markers.

**Task prefixes.** The model is trained with short instruction prefixes
on text; the card says they improve quality, and omitting them still
works. **The prefix opens the input**, before any media part:
sentence-transformers prepends a prompt as a system message, which the
template renders first, so an image alone gets one too (checked on the
pinned revision, 2026-10-09: the logo with the query prompt is `<bos>`, the
prompt's eight tokens, BOI, …). From the card's best-practices section and
`config_sentence_transformers.json`:

| Use | Kind | Query | Document |
| --- | --- | --- | --- |
| Web and document search | asymmetric | `task: search result \| query: {query}` | `title: {title} \| text: {content}` |
| Question answering | asymmetric | `task: question answering \| query: {question}` | `title: {title} \| text: {passage}` |
| Fact checking | asymmetric | `task: fact checking \| query: {claim}` | `title: {title} \| text: {evidence}` |
| Code search | asymmetric | `task: code retrieval \| query: {query}` | `title: {title or filename} \| text: {code}` |
| Classification | symmetric | `task: classification \| query: {content}` | — |
| Clustering | symmetric | `task: clustering \| query: {content}` | — |
| Similarity | symmetric | `task: sentence similarity \| query: {content}` | — |

A document without a title is `title: none | text: {content}`, which is
what sentence-transformers' `Document` prompt renders; a titled document
is formatted by the caller. The prompt table also names aliases with the
same text (`Retrieval-query`, `STS`, `Reranking`, `BitextMining` and
others); `default_prompt_name` is null, so sentence-transformers adds
nothing unless asked.

## Matryoshka widths

The model is trained with Matryoshka representation learning: the leading
768, 512, 256 or 128 dimensions of the vector are a usable embedding once
renormalized to unit length. Other widths are not trained. Vectors of
different widths, or from different files, are not comparable.

## The f16 hazard

The card: "Run inference in `bfloat16` or `float32`. Do not use
`float16`." The activation range exceeds f16, which returns NaN or a
silently degraded vector without an error. In our terms: residuals,
activations and attention stay F32 on every backend, and no matrix path
whose tiles hold activations in half (the K-quant prefill tiles) may run
this family. BF16 weights are safe (an 8-bit exponent). On Metal, the
Q8_0, BF16 and F32 matrices run on the generic matmul tile, whose
weight and activation tiles are both F32 (`matmulGeometry`: `half =
false`); the plan calls `matmulTile` so that no batch size reaches
another kernel.

**Decided 2026-10-09: F32 only.** Half-operand tiles would make a Metal
batch about 20–25 % faster (§ The Metal plan, rates), but they are not
built for this family:
- In f16, the feed-forward down projection's input (gelu(gate) · up) is
  not normalized and may overflow.
- In bf16, the vectors would give up their agreement with Google's float32
  pass (llama.cpp's bf16-staged run reaches a cosine of 0.999995 against
  ours at 1 − 4e-12).
- Vectors from two precisions are not exactly comparable in one index.

A bf16 mode would be an opt-in, measured against Google's vectors and a
retrieval check, and only once indexing speed is shown to matter.

## The mmproj (`clip`, 963 tensors)

One file carries both encoders: `clip.has_vision_encoder` and
`clip.has_audio_encoder` are both true, and both project to 512, the text
width.

**Vision (`gemma4v`, 211 tensors).**

- 16 blocks, width 768, FFN 3072 (gated, `gelu_pytorch_tanh`), 12 heads of
  64 with Q/K norms, patch 16, `v.position_embd` [768, 10240, 2], and
  axial RoPE with base 100 (llama.cpp's `clip.cpp` sets it).
- **No clipped-linear bounds** (`use_clipped_linears` false in Google's
  `vision_config`; no `*.input_min`-style tensors in `v.*`).
- `mm.input_projection` [768 → 512] BF16. Matrices are BF16, norms and the
  patch and position embeddings F32.
- `image_mean` 0, `image_std` 1 (`do_normalize` false): the only pixel
  normalization is the 1/255 rescale. Resampling is bicubic.
- Soft tokens per image: **280** by default (Google's `max_soft_tokens` and
  `image_seq_length`, pooling kernel 3); the card allows 70 to 1120. The
  reference's `clip.cpp` caps `gemma4v` at 70…1120 tokens and, without
  `--image-max-tokens`, sizes images toward the **1120** cap, so the oracle
  must be run with `--image-max-tokens 280` to match Google's processor.
- Tokens: BOI 255999, image 258880, EOI 258882. `mtmd` frames an image
  as `<|image>` … `<image|>`.
- **The FFN activation is `gelu_pytorch_tanh`** (Google's
  `hidden_activation`). No key in the file says so, and llama.cpp's
  `clip.cpp` runs every `gemma4v` file with its default, `gelu_quick`.
  nuclis's chat path follows the reference; the embedder uses Google's
  (§ Images).

**Audio (`gemma4a`, 752 tensors).** The same tensors, names and shapes as
Gemma 4 E4B's audio encoder, except `mm.a.input_projection` (1536 → 512
here, 1536 → 2560 there):

- Two conv2d subsampling layers (`a.conv1d.0`, `a.conv1d.1`; 3 × 3
  kernels, 128 and 32 channels, each with a norm), then
  `a.input_projection` [1024, 1024] F32.
- 12 conformer blocks of width 1024 with 8 heads (128 per head) and FFN
  4096 (`silu`). Each block has two half-step FFNs (`ffn_*` and
  `ffn_*_1`, residual weight 0.5), and chunked local attention: chunk 12,
  left context 13, right context 0, logit cap 50, `attn_k_rel`, and
  `per_dim_scale` [128]. It also has a conv module (`conv_pw1` into a GLU,
  a depthwise kernel of 5, `conv_pw2`) and clipped linears
  (`input_min/max`, `output_min/max`) on every attention, FFN and
  pointwise conv matrix.
- `a.pre_encode.out` [1024 → 1536] with a bias, then
  `mm.a.input_projection` [1536 → 512].
- **`clip.audio.attention.layer_norm_epsilon` is 1e-6** here (Google's
  `audio_config.rms_norm_eps`), and 1e-5 in E4B's mmproj: read it from the
  header, do not carry E4B's.
- Front end (`Gemma4AudioFeatureExtractor`): 16 kHz mono; frame 320, hop
  160, FFT 512; 128 mel bins over 0–8000 Hz; `mel_floor` 1e-3; no
  preemphasis, no dither, no per-bin normalization; right padding.
- 40 ms per token (25 per second). Tokens: BOA 256000, audio 258881, EOA
  258883. `mtmd` frames audio as `<|audio>` … `<audio|>`.
- **Three facts the names do not give** (§ Audio): the light
  convolution's two norms are swapped in the file, `per_dim_scale` is
  stored after its softplus, and the attention mask keeps 11 rows back,
  not 12.
- **Google's processor cuts a clip at 30 s** (`max_length` 480,000
  samples).

**Video** (deferred to its own unit, 2026-10-09) reuses the vision
encoder. From Google's `EmbeddingGemma2VideoProcessor` and
`EmbeddingGemma2Processor` at the pinned revision:
- **Sampling:** frames at 1 fps, at most 32; a longer video is
  resampled uniformly (`overflow_strategy = "uniform"`).
- **Frames:** each is resized as an image is, under a 140-soft-token
  budget, and framed as BOI (255999), its rows as `<|video|>` (258884)
  placeholders, EOI. There are no timestamps (`add_timestamps` false), so
  32 frames take about 4,500 tokens.
- **The soundtrack is not embedded:** a video's audio would be a separate
  audio part.
- **What the unit would add:** an AVFoundation frame decoder beside the
  image and audio bridges, Google's sampling, oracle fixtures from a short
  generated video, gates, `--video`, and a video part in the API. Parity
  with Google is exact only up to the decoded pixels, as with resampled
  audio.

## The input contract

One input is an ordered list of parts, and gives one vector. All parts
share one bidirectional pass of at most **8192** tokens: BOS and EOS, the
text tokens, and each media part's soft tokens with its begin and end
markers.

| Part | Becomes | Budget |
| --- | --- | --- |
| Text | its tokens; the task prefix, if any, opens the input | 1 per subword |
| Image | BOI, the projector rows, EOI | 280 by default; 70, 140, 560 or 1120 |
| Audio | BOA, the projector rows, EOA | 25 per second, at most 30 s (750) |

**Text is literal (decided 2026-10-09).** nuclis tokenizes text parts with
`parse_special = false`: the spelling of a control token in a document
(`<bos>`, `<|image|>`) is text, never the token. Media positions come from
the order of the parts, so no text needs a placeholder, and content cannot
change an input's framing. Every framing token (`<bos>`, `<eos>`, the image
and audio markers, `<|video|>`) is a **control** token in this vocabulary,
which literal encoding never matches; its 7 user-defined tokens (tool-call
and channel markers) still match, as in the reference, and frame nothing.
Google's pipeline parses such spellings; the vectors differ only for inputs
that contain one. In the fixture set that is `long-8k.doc`: one token, and
a cosine of 0.99979 against Google's vector, which becomes 1.0000000 when
the spelling is parsed the oracle's way.

The card's own example interleaves parts in one input ("Waterproof running
shoes. `<|image|>` Featuring a breathable mesh upper."); the vector
represents them together and is comparable with a text-only one.

## Reference oracle status

- **llama.cpp.** The main pin `7620399` does not know `gemma-embedding2`.
  Upstream added it, with vision and audio, in `4fbc76dec` (#30054,
  2026-10-06); the newest release, `v0.6.0` (2026-10-05), predates it. The
  second checkout `.reference/llama.cpp-embed` is at build tag `b11514`,
  `de7fa0a3c6a2e1b4cd9f22eb8d6bf5b12dbdb63b` (2026-10-08), built with the
  usual recipe ([llama-cpp.md § The third oracle](../benchmarks/llama-cpp.md#the-third-oracle-embeddinggemma-2-2026-10-09)).
  Its `llama-embedding --pooling mean` loads the Q8_0 and BF16 files and
  returns unit vectors. The fixture driver is `scripts/reference-embedding.cpp`;
  it runs each input as **one mixed batch** of token rows and projector rows
  (`llama_batch_ext`), because separate decode calls on a cache-less model
  would not attend to each other.
- **sentence-transformers.** `google/embeddinggemma-2` at the pinned
  revision, in float32 on the CPU, through sentence-transformers 6.1.0,
  transformers 5.19.0 (the card was written against 5.18.0.dev0) and torch
  2.14.1, in `.reference/venv-embed`, driven by `scripts/embedding-reference.py`.
  It is the end-to-end truth, input processing included (prefixes, image
  resizing, mel features).

Both read one input set, `tests/fixtures/embeddinggemma-inputs/inputs.json`
(104 cases; [provenance](../../tests/fixtures/provenance.md)).

**The oracles against each other (2026-10-09).** Cosines per case from
`scripts/compare-embedding.py`, minimum and mean per group; the token ids
are identical in every case:

| Group (cases) | llama.cpp Q8_0 / ST f32 | llama.cpp BF16 / ST f32 | Q8_0 / BF16 |
| --- | --- | --- | --- |
| Text (96) | 0.999888 / 0.999947 | 0.999995 / 0.999998 | 0.999885 / 0.999945 |
| 2,930-token document | 0.999905 | 0.999999 | 0.999905 |
| 8,142-token document | 0.999897 | 1.000000 | 0.999897 |
| Images (2) | 0.995162 / 0.997510 | 0.995319 / 0.997609 | 0.999931 / 0.999948 |
| Text, image, text (1) | 0.999874 | 0.999924 | 0.999946 |
| Audio (3) | 0.998446 / 0.999429 | 0.998670 / 0.999547 | 0.999866 / 0.999912 |

On text, BF16 matches Google's float32 to six digits, so the reference
graph is right and the Q8_0 file's cost is about 5e-5 of cosine. The gap
on images is the reference's resampling, not the model: `image.og`
(1200 × 630) is at 0.9952, and the same picture pre-resized to its patch
grid (1104 × 576, so neither side resamples) agrees at 0.99980.
`clip.cpp`'s bicubic is not Google's (PIL's antialiased bicubic), so for
images **sentence-transformers is the oracle** and llama.cpp checks the
encoder from given pixels.

**Facts the oracles settled.**

- **Special-token text is parsed.** The HF tokenizer turns special-token
  text in the input (a literal `<bos>`, as in `long-8k.txt`) into the
  token; llama.cpp does the same only with `parse_special`, which the
  driver sets.
- **Framing.** An image is `<bos>` BOI (255999), its soft tokens, EOI
  (258882), `<eos>`; an audio clip the same with BOA (256000) and EOA
  (258883). Text around a media part keeps its whitespace: "engine: "
  ends in its own space token (236743) before BOI.
- **Soft tokens follow the aspect ratio inside the budget.** At 280, the
  1254 × 1254 logo becomes a 48 × 48 patch grid, 16 × 16 = 256 soft tokens;
  the 1200 × 630 card a 36 × 69 grid, 12 × 23 = 276.
- **Audio** is 25 tokens per second: 4.8 s gives 479 mel frames and 120
  soft tokens.
- **The 8192 limit is ours to enforce.** sentence-transformers'
  `max_seq_length` is unbounded, and it truncates nothing.

## CPU reference against the oracles (2026-10-09)

The CPU reference is `inference/src/models/embeddinggemma.zig` (the
binding), `embeddinggemma_runtime.zig` (the forward), and
`inference/src/embed.zig` (the `Embedder`: files, tokens, framing, limits,
Matryoshka). One input is one F32 pass with no cache. Each matrix is
decoded from its encoded rows into one reused scratch right before its
`cpu.dense.matmul`, so nothing is resident in F32. `token_embd` rows are
gathered one at a time. `ple` is computed per layer from the kept `x0`.
Attention is `cpu.dense.attention`, which gained grouped-query heads
(`kv_heads`, `kv_stride`) for this family. Its window was already symmetric:
on a sliding layer, `|i − j| ≤ 512`.

`inference/embeddinggemma-check.zig` (`zig build test-embeddinggemma --
MODEL traces|vectors|google`) checks it in three ways, each a gate of
tier `verify-cpu`:

| Gate | File | Against | Measured | Bound |
| --- | --- | --- | --- | --- |
| `embeddinggemma-google-cpu` | BF16 | Google's float32 vectors, 97 texts | 1 − min cosine 4.0e-12 | cosine ≥ 0.9999999 |
| `embeddinggemma-vectors-cpu` | Q8_0 | llama.cpp's Q8_0 vectors (Metal), 97 texts | 1 − min cosine 4.2e-7 | cosine ≥ 0.99999 |
| `embeddinggemma-trace-cpu` | Q8_0 | llama.cpp's CPU pass over the same weights in F32, every stage of `north.raw` and `north.query` | relative RMS ≤ 3.7e-4, relative max ≤ 3.1e-4 | 5e-4 and 2e-3 |

Token ids equal the oracle's in all 97 cases. In `long-8k.doc` they must
differ, and parsed the oracle's way they must match (the input contract).

**The accuracy claim is the BF16 line.** On the same weights, our F32 pass
and PyTorch's agree to `1 − cos ≈ ½δ²` with δ ≈ 3e-6, the size of
summation-order rounding. llama.cpp's BF16 run reaches only 0.999995 against
the same vectors (§ Reference oracle status).

**Why the traces are bounded at 5e-4, not the decode traces' 1e-4.** The
reference's own arithmetic rounds:

- On Metal, the batched matmul stages activations in threadgroup memory as
  half, or as bf16 for a BF16 matrix. The recorded Metal traces differ from
  ours by 4.6e-4 (`l_out`) and 1.6e-3 (`inp_per_layer`, the BF16
  `per_layer_model_proj`) relative RMS.
- On the CPU, a BF16 matrix rounds its activations to bf16 too, and
  `llama-quantize … F32` keeps `per_layer_model_proj` in BF16. So the
  traced reference runs on a copy with every tensor widened to F32
  (`scripts/gguf-widen-f32.py`), and `inp_scaled` and `inp_per_layer` then
  match.
- ggml's CPU GELU is an f16 lookup table (`GGML_GELU_FP16`, unconditional
  in `ggml-cpu/vec.h`): input and output rounded to f16. Emulating the table
  in our runtime brought layers 0–2 under 1e-4, but the later layers stay
  near 2e-4 from further F32 differences in the reference, such as its
  angle stepping. The vector gates, not the traces, carry the accuracy
  claim. The traces localize a defect to a stage, and any real defect
  exceeds 5e-4 by orders of magnitude.

**Cost.** 97 texts plus the 8,142-token document take about 130 s
(ReleaseFast, M4 Pro), most of it in the long document. That is a reference
rate: the Metal plan is the path that indexes a corpus.

## The Metal plan (2026-10-09)

`inference/src/models/embeddinggemma_metal.zig` runs the CPU forward over
packed batches of at most 8192 rows; `Embedder` takes it with
`Backend.metal`, and `embedBatch` packs consecutive inputs.

- **Layout.** Each input starts on an 8-row boundary. The rows that pad it
  to the boundary are a zero-input segment of their own. Each row's
  `[begin, end)` is in a bounds buffer, and its rotary row (at its position
  in its input) in a per-row table.
- **Kernels.** Norms, GELU and the residual adds are the backend's F32
  kernels. Matrices are wrapped in place for the generic F32 tile.
  Attention is `Backend.attentionSegmentsGrouped`, with materialized scores
  ([metal-backend.md § Segments attention with materialized scores](../engine/metal-backend.md#segments-attention-with-materialized-scores-2026-10-09)).
  Its query tiles never mix inputs, and each covers only the keys its rows
  can see: the input on a global layer, the window band on a sliding one.
- **Host work.** Token rows are decoded, and vectors pooled, with the CPU
  reference's own functions (`inputRows`, `pool`).
- **Memory.** About 34 KB of activations per row (277 MB at 8192 rows),
  plus a 128 MB score buffer. An 8192-row input's global layer runs in 8
  chunks of it.

**Batching changes timing, never answers.** Every tile, key range and
summation order of an input is the same wherever it sits in a batch, so its
vector is identical, bit for bit, alone or packed. The check tool holds
this on every text case.

**Accuracy.** Two gates of tier `verify` (`embeddinggemma-check MODEL
vectors|google --backend metal`; 7 s each):

| Gate | File | Against | Measured | Bound |
| --- | --- | --- | --- | --- |
| `embeddinggemma-google-metal` | BF16 | Google's float32 vectors, 97 texts | 1 − min cosine 4.2e-12 | cosine ≥ 0.9999999 |
| `embeddinggemma-vectors-metal` | Q8_0 | llama.cpp's Q8_0 vectors, 97 texts | 1 − min cosine 4.2e-7 | cosine ≥ 0.99999 |

Both also require the packed batches to equal the one-at-a-time vectors.
Against the CPU traces' references the Metal stages land within the CPU's
own bounds (`traces --backend metal`: relative RMS ≤ 3.7e-4).

**Rates.** Measured 2026-10-09 on an Apple M4 Pro (48 GB), macOS 27.0.1, Zig 0.17.0,
ReleaseFast, at `191e26c`, with nothing else running on the GPU.
`embeddinggemma-check MODEL bench --backend metal` against llama.cpp
`b11514` on its defaults (Metal, flash attention `auto`, matmul
activations staged as half), through `scripts/reference-embedding-bench.cpp`
([llama-cpp.md § The third oracle](../benchmarks/llama-cpp.md#the-third-oracle-embeddinggemma-2-2026-10-09)).

**Method.** Each input is BOS, " the" repeated, EOS (the cost does not
depend on the text), in batches of at most 8192 rows. One untimed warm-up
per shape, then the median of 7 timed runs (5 for the single inputs). Each
run ends when every vector is on the host. The two engines alternate, each
run after 60 s idle, three rounds. The table gives the median round; rounds
spread about ±5 %, and a warm GPU runs both engines up to 8 % faster.

| Input | nuclis Q8_0 | llama.cpp Q8_0 | nuclis BF16 | llama.cpp BF16 |
| --- | ---: | ---: | ---: | ---: |
| 64 × 256 tokens | 1,671 ms, 38.3 inputs/s | 1,560 ms, 41.0 inputs/s | 1,619 ms, 39.5 inputs/s | 1,606 ms, 39.9 inputs/s |
| 1 × 512 tokens | 59.2 ms | 50.8 ms | 57.8 ms | 52.9 ms |
| 1 × 8192 tokens | 1,754 ms | 1,808 ms | 1,731 ms | 1,910 ms |

**Reading.**
- **Level, at higher precision.** On the corpus batch the two engines are
  within the run-to-run spread on BF16, with llama.cpp 7 % ahead on Q8_0,
  though it rounds every matmul's activations to half. We are 3–10 %
  ahead at 8192 tokens, where attention dominates. On one 512-token input
  llama.cpp leads by 9–14 %.
- **GPU-bound.** Host work (decoding token rows, the tables, pooling) is
  about 25 ms of a 1.64 s batch.
- **Where the batch goes.** About 80 % is F32 matmul on the generic tile
  at about 3.7 TFLOP/s, the tile's ceiling. The half-operand tiles reach
  about 5 TFLOP/s, and they are what the f16 hazard rules out; that
  trade is the remaining lever. Attention takes about 11 %.

## `nuclis embed` (2026-10-09)

The command (`src/embed.zig`) and the wire types it shares with the API
(`src/embedding/`) follow [spec §5.10](../spec.md#510-the-embedding-path):
no implicit task, Google's prefixes from `--task` and `--title`, the
trained widths, truncation reported per input, and every response named
by its space (`embeddinggemma-2@6f1bd4ac6c5d/768` for the catalogue's Q8_0
file). With a task, the prefix opens the input (§ Tokenizer and task
prefixes).

**End to end against the oracles.** The fixture's `north` sentence through
the built binary on Metal, with the Q8_0 file, rendered by `--task`
rather than pre-rendered as in the fixture set. The token counts equal
the oracles' in each case.

| `--task` | Fixture case | Tokens | 1 − cos, llama.cpp Q8_0 | 1 − cos, Google's f32 |
| --- | --- | ---: | ---: | ---: |
| `search_query` | `north.query` | 25 | 1.8e-7 | 4.5e-5 |
| `document` | `north.doc` | 24 | 3.8e-7 | 5.7e-5 |
| `code_retrieval` | `north.code` | 25 | 3.6e-7 | 6.4e-5 |

These are the gates' agreements (§ The Metal plan): the command adds no
error of its own. The gap to Google's float32 is the Q8_0 file's, about
5e-5 of cosine on text (§ Reference oracle status).

## Images (2026-10-09)

An image part goes through Google's `Gemma4ImageProcessor` and vision
tower, not through the chat path's reference pipeline. Each step was read
from the pinned transformers 5.19.0 and torchvision, and checked on the
fixture images:

| Step | Google's (what nuclis does for embeddings) | llama.cpp's (the chat path's) |
| --- | --- | --- |
| Decode | PIL `convert("RGB")`: alpha dropped, straight values, no colour management | stb_image: the same |
| Grid | `get_aspect_ratio_preserving_size`: the largest 48-pixel grid of the image's ratio within `budget · 9` patches, floored (`gemma4.googleGrid`) | the smart size on the 48-pixel grid, letterboxed |
| Resize | torchvision's native uint8 antialiased bicubic: Pillow's separable passes with weights rounded to the most fractional bits that keep the largest in an i16 (`preprocess.resizeWith(…, .torchvision)`) | Pillow's, 22 fractional bits |
| Patches | F32, `2x − 1` | rounded to F16 (the convolution's im2col) |
| FFN gate | `gelu_pytorch_tanh` | `gelu_quick` |

- **Decoding.** ImageIO's drawing path premultiplied alpha: on the RGBA
  logo it changed 12 % of the values, by up to 254. The bridge now copies
  the stored values of 8-bit RGB layouts (alpha dropped, no colour
  conversion), as PIL and stb_image do; other layouts are still drawn.
  Both fixture images then equal PIL's bytes. The chat path decodes
  through the same bridge.
- **Resizing.** Pillow's own resize differs from torchvision's by one
  level in 0.05 % of the logo's values. `resizeWith(…, .torchvision)`
  equals torchvision on both fixture images at 70, 280 and 1120 tokens,
  downscaling and upscaling (`preprocess.zig` tests it against two
  recorded torchvision resizes).
- **Budgets.** Google's processor takes 70, 140, 280, 560 or 1120 soft
  tokens and nothing else, so `--image-tokens` and `image_tokens` take
  those five. The budget changes the vector, and the response records it.
- **Encoder memory.** The vision encoder is built on the first image. On
  Metal it then holds 918 MB: its weights, mapped in place, and the plan's
  buffers, which are sized for 1120 tokens. The server adds that to the
  model's share of the memory budget.

**Against the oracles** (`zig build test-embeddinggemma -- MODEL images
--mmproj mmproj-BF16.gguf`, M4 Pro):

| File | Backend | `image.logo` | `image.og` | `mix.logo` |
| --- | --- | ---: | ---: | ---: |
| BF16 | Metal | 2.0e-12 | 9.6e-13 | 3.4e-12 |
| BF16 | CPU | 3.1e-12 | 1.2e-12 | 4.9e-12 |
| Q8_0 | Metal | 3.3e-5 | 7.1e-5 | 5.3e-5 |

These are `1 − cos` against Google's float32 vectors, with the token ids
equal to Google's in each case. On the BF16 file, images agree as text
does, to summation order. The Q8_0 file costs about what it costs on text.
llama.cpp's own image vectors reach 0.995 to 0.99986 (§ Reference
oracle status). The mode also runs the vision encoder on llama.cpp's
pixels with its F16 patches and `gelu_quick` against its projector rows
(`media-0`), as a localizer: relative RMS 8.3e-3, relative max 2.7e-2.
That oracle was recorded on Metal, whose matmuls stage activations as
half, so its bound is loose (1.5e-2 and 5e-2). A packed batch of the
three gives each its vector alone, bit for bit.

| Gate | Tier | Runs |
| --- | --- | --- |
| `embeddinggemma-image-metal` | `verify` | every image case, BF16, floor 0.9999999; packed batch equality |
| `embeddinggemma-image-cpu` | `verify-cpu` | `mix.logo` only, BF16, floor 0.9999999 (6 minutes: the CPU reference takes about 3 minutes an image) |

**Rates.** On Metal, preparing one image (decode, resize, encode) takes
about 460 ms at 280 tokens, about 2 images a second. The encoder's
matmuls dominate, on the generic F32 tile: 2,304 patches through 16
blocks.

## Audio (2026-10-09)

An audio part is decoded to 16 kHz mono, turned into Google's log-mel
frames, and encoded by the `gemma4a` conformer. The derivation, the
decoder, and the three facts the file's names do not give are in
[audio.md](../engine/audio.md).

- **Front end.** The frames equal `Gemma4AudioFeatureExtractor`'s to
  4.8e-7 on all three clips.
- **The 30 s bound.** Google's extractor truncates at 30 s, so an audio
  part is at most 750 rows. A longer clip is refused (`input_too_long`,
  naming the 30 s) unless the request truncates, which keeps the first
  30 s and reports the input as cut.
- **Encoder memory.** The audio encoder is built on the first clip. On
  Metal it then holds 658 MB: its BF16 weights (300M parameters, 604 MB,
  mapped in place) and buffers for 750 rows.

**Against the oracles** (`zig build test-embeddinggemma -- MODEL audio
--mmproj mmproj-BF16.gguf`, M4 Pro):

| File | Backend | `audio.north` | `audio.sourdough` | `audio.tone` |
| --- | --- | ---: | ---: | ---: |
| BF16 | CPU | 6.0e-13 | 4.2e-13 | 2.4e-11 |
| BF16 | Metal | 6.7e-13 | 7.5e-13 | 4.8e-11 |
| Q8_0 | Metal | 5.0e-5 | 7.4e-5 | 1.5e-4 |

These are `1 − cos` against Google's float32 vectors, with the token ids
equal to Google's. On the BF16 file audio agrees as text and images do.
The Q8_0 file's cost is largest on the tone, a sound unlike speech
(llama.cpp's Q8_0 and BF16 vectors also differ by 1.3e-4 on audio).

Two more measurements back these up:
- **Google's own stages.** Every stage of the CPU reference matched a hook
  dump of Google's tower (subsampling, block 0's three sub-stages, each
  block, the output and the projection) to relative RMS ≤ 1.8e-6.
- **llama.cpp's rows** (`media-0`) are a localizer only: relative RMS
  1.4e-2 and relative max 8.2e-2, from its own front end and half-staged
  matmuls. The swapped norms alone gave 0.72.

A packed batch of the three clips gives each its vector alone, bit for
bit.

| Gate | Tier | Runs |
| --- | --- | --- |
| `embeddinggemma-audio-metal` | `verify` | every clip, BF16, floor 0.9999999; packed batch equality |
| `embeddinggemma-audio-cpu` | `verify-cpu` | every clip, BF16, floor 0.9999999 (6 s) |

**Across modalities** (recorded, not gated; Q8_0, Metal). Each spoken
sentence ranks its own text first among the 24 raw text cases:

| Clip | Its text | Next |
| --- | ---: | ---: |
| `speech-1.wav` (`north`) | 0.902 | 0.660 (`ar.raw`) |
| `speech-2.wav` (`sourdough`) | 0.763 | 0.654 (`hello.raw`) |

**Rates.** On Metal, preparing a clip (decode, mel, encode) takes 130 ms
for 4.8 s and 45 ms for 2 s.

**Other formats.** A 16 kHz mono 16-bit WAV is read exactly. Anything else
goes through AudioToolbox's resampler: a 44.1 kHz stereo copy of the
`north` clip gives cosine 0.9999 to the original, an AAC copy 0.998.

