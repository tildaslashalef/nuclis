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
`06cad0b9e` (`src/models/gemma-embedding2.cpp`, `conversion/gemma.py`,
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
works. **Prefixes apply to text only**: images, audio and video go without
one. From the card's best-practices section and
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
this family. BF16 weights are safe (an 8-bit exponent).

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

**Video** (not in this unit) reuses the vision encoder: frames sampled at
1 fps, at most 32 frames, 140 soft tokens per frame, token `<|video|>`
258884.

## The input contract

One input is an ordered list of parts, and gives one vector. All parts
share one bidirectional pass of at most **8192** tokens: BOS and EOS, the
text tokens, and each media part's soft tokens with its begin and end
markers.

| Part | Becomes | Budget |
| --- | --- | --- |
| Text | its tokens, after the task prefix if any | 1 per subword |
| Image | BOI, the projector rows, EOI | 280 by default (70–1120) |
| Audio | BOA, the projector rows, EOA | 25 per second (about 327 s alone) |

The card's own example interleaves parts in one input ("Waterproof running
shoes. `<|image|>` Featuring a breathable mesh upper."); the vector
represents them together and is comparable with a text-only one.

## Reference oracle status

- **llama.cpp.** The main pin `7620399` does not know `gemma-embedding2`.
  Upstream added it, with vision and audio, in `4fbc76dec` (#30054,
  2026-10-06). The second checkout `.reference/llama.cpp-embed` at
  `06cad0b9e77315bd2930bd4f70a6d7b37b2a01c1` (2026-10-07) builds with the
  usual recipe ([llama-cpp.md § The third oracle](../benchmarks/llama-cpp.md#the-third-oracle-embeddinggemma-2-2026-10-09)).
  On 2026-10-09 its `llama-embedding -ngl 99 --pooling mean
  --embd-normalize 2` loaded the Q8_0 and BF16 files and returned
  768-wide vectors of norm 1.0000000.
- **sentence-transformers.** `google/embeddinggemma-2` at the pinned
  revision, in float32, through sentence-transformers 6.1.0 and
  transformers 5.19.0 (the card was written against 5.18.0.dev0) in
  `.reference/venv-embed`. It is the end-to-end truth for input
  processing too (prefixes, image resizing, mel features).
