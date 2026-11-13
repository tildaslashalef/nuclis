<!-- kind: card -->
# Gemma 4

Google's Gemma 4 in three configurations behind one adapter,
[inference/src/models/gemma4.zig](../../inference/src/models/gemma4.zig),
which selects the configuration by `gemma4.block_count` and validates every
key against it. Catalogue entries: `gemma-4-26b-a4b` (the mixture of
experts, the default chat model), `gemma-4-e4b-qat` (the on-device size,
which also hears audio), and `gemma-4-12b-qat` (the dense 12B, legacy).
Each pulls its quantization-aware-trained Q4_0 file, a vision projector,
and a draft head ([catalogue.md](catalogue.md)). How every fact here was
established, the oracle comparisons, and the first-look rates are in
[the record](../benchmarks/record/gemma4.md). Images are in
[vision.md § Gemma 4's projectors](../engine/vision.md#gemma-4s-projectors).

## Architecture

| Fact | 12B | 26B-A4B | E4B |
| --- | --- | --- | --- |
| Blocks (sliding / global) | 48 (40 / 8) | 30 (25 / 5) | 42 (35 / 7) |
| Width, dense FFN | 3,840, 15,360 | 2,816, 2,112 (the shared branch) | 2,560, 10,240 |
| Query heads | 16 | 16 | 8 |
| KV heads, sliding / global | 8 / 1 | 8 / 2 | 2 / 2 |
| Global value | the key projection | the key projection | its own `attn_v` |
| Sliding window (own position included) | 1,024 | 1,024 | 512 |
| Experts | none | 128, 8 per token, 704 wide | none |
| Per-layer embeddings | none | none | 256 wide |
| Layers owning a cache | all | all | 0–23; 24–41 read 22 (sliding) or 23 (global) |
| Image span on sliding layers | bidirectional | bidirectional | causal |
| Context declared | 262,144 | 262,144 | 131,072 |

Shared by all three: period-6 layer pattern (layer 5, 11, … global); head
size 256 on sliding layers and 512 on global ones; vocabulary 262,144;
RMS epsilon 1e-6; tied output head (no `output.weight`); final logit
soft-cap 30.

## Forward pass

With `rms(x) = x / sqrt(mean(x²) + 1e-6)` and every norm `rms(x) · w`
over the **raw** stored weight (nothing added):

1. `x = E[token] · sqrt(width)` (token rows only; image and audio rows are
   not scaled).
2. Per layer: `q = rope(norm_q(Wq a))`; `k = Wk a` and `v = Wv a`, or
   `v = k` before any norm where the layer has no `attn_v`; then
   `k = rope(norm_k(k))` and `v = rms(v)` without a weight, never rotated.
   Scores `q · k` at **scale 1.0**, causal, sliding layers masked to the
   window, no attention soft-cap. `h = norm_post_attn(Wo o) + h`.
3. Dense FFN: `Wdown · (gelu_tanh(Wgate f) ⊙ Wup f)` over `f = norm_ffn(h)`,
   then `h = (norm_post_ffn(·) + h) · layer_output_scale`, a per-layer scalar
   on the whole residual stream.
4. Head: `30 · tanh(Eᵀ norm_out(h) / 30)`.

**RoPE** is NeoX pairing over the whole head: base 10,000 on sliding
layers; base 1,000,000 on global layers with `rope_freqs.weight` as per-pair
divisors (64 ones, 192 of 1e30), so 128 of the 512 dimensions rotate.

### The 26B-A4B's expert layer

With `a` the residual after the attention add: the shared branch is the
dense FFN followed by `post_ffw_norm_1`; the router takes
`rms(a) · (1/√2816) ⊙ ffn_gate_inp.scale`, a softmax over 128 logits, the 8
largest renormalized to sum one (`cpu.experts.route`); each selected expert
computes `scale[e] · Wdown[e] · (gelu(gate) ⊙ up)` over `pre_ffw_norm_2(a)`
from the fused gate-up tensor, summed by routing weight and normed by
`post_ffw_norm_2`; then `h = (post_ffw_norm(shared + experts) + a) ·
layer_output_scale`. The 3-D tensors are `[experts][rows][columns]`, experts
contiguous (`cpu.ExpertMatrix`), and stay resident; on Metal they run on the
[gathered expert kernels](../engine/metal-kernels.md#gathered-expert-kernels),
and the plan asks for a prompt chunk of 512 (`Plan.preferredChunk`).

### The E4B's per-layer embeddings and shared KV

Per token, with `x0` the scaled embedding row,
`ple = (norm_256(P · x0 / √2560) + E_pl[t] · √256) / √2`; per layer, after
the FFN residual and before the output scale,
`x += norm(proj · (gelu(inp_gate · x) ⊙ ple[il])) ⊙ post_norm`. An image or
audio row takes its unscaled row as `x0` and the padding token's per-layer
row. Layers 24–41 have no `attn_k`, `attn_v` or `attn_k_norm`, still
project and rotate Q, and attend over the cache `Config.kvSource` names;
the session holds 24 caches.

## Files and encodings

The adapter accepts matrices in F32, Q4_0, Q5_0, Q5_1, Q8_0, Q3_K, Q4_K,
Q5_K, Q6_K, IQ4_NL, IQ3_S and IQ4_XS (`gemma4.matrixEncoding`), norms in
F32, and BF16 for `per_layer_model_proj` alone (`executableEncoding`), which
the E4B's K-quant and IQ files keep unquantized. Those files also carry
`attn_k`, `attn_v` and `attn_k_norm` on the shared-KV layers; the binder
checks their shapes and never reads them. Another block count is
`UnsupportedConfiguration`. The draft head is its own architecture
(`gemma4_assistant.zig`); `bindDraft` refuses one whose KV heads or window
differ from the target's.

## Tokenizer

`tokenizer.ggml.model = "gemma4"`: SPM-style BPE over raw UTF-8 code
points, spaces escaped to `▁` before merging, runs of newlines split from
everything else, `<0xNN>` byte fallback (`Encoder.encodeSpm`,
`bpe.encodeSpmBudget`). The encoder never adds BOS: the profile writes
`<bos>` (2) as text, and `nuclis tokenize --raw` and `generate --raw` take
it in the text. The seven user-defined markers (tool calls, responses,
channels, `<|"|>`) always decode as text.

## Prompt profile

The profile is selected by the chat template's SHA-256: `gemma4` for the
12B and 26B-A4B ([prompt-profile.md § Gemma 4](../engine/prompt-profile.md#gemma-4-gemma4)),
`gemma4_e` for the E4B, whose template drops the empty thought channel
after the generation prompt when thinking is off
([§ Gemma 4 E-series](../engine/prompt-profile.md#gemma-4-e-series-gemma4_e)).
Turns are `<|turn>role\n…<turn|>\n`; thinking is a switch (`<|think|>` in
the system turn); reasoning is `<|channel>thought\n…<channel|>` and is not
kept in history. Generation stops on `<turn|>` (106) or `<eos>` (1). Tool
calls are in [tool-calling.md](../engine/tool-calling.md). Sampling defaults
are the file's hint: temperature 1.0, top-p 0.95, top-k 64, no penalties.
A file whose template matches neither digest (a fine-tune's, or the earlier
17,530-byte revision) is `UnsupportedPromptTemplate` until forced with
`--prompt-profile gemma4` or `gemma4_e`.

## Audio input on the E4B

A user message's `input_audio` part ([api.md § The request](../guide/api-chat.md#the-request))
or `nuclis generate --audio <file>`. The encoder is the `gemma4a` conformer
shared with EmbeddingGemma 2 ([audio.md](../engine/audio.md)): 751 of the
companion's 752 audio tensors are byte-identical to EmbeddingGemma 2's,
and only `mm.a.input_projection` differs (1,536 to 2,560). The embedder is
an RMS norm without a weight then that projection, computed at epsilon
1e-6 whichever value the file states. A clip renders as `<|audio>`, one
`<|audio|>` per encoder row (25 a second, at most 750: 30 s), `<audio|>`;
a longer clip is refused, never cut. A message renders its images, then
its clips, then its text; the span is causal and goes through the same
`prefillVision` as an image's. The encoder is bound on the first clip
(`Engine.loadAudio`) from the projector file. The token cache keys each
clip by its SHA-256 ([session.md](../engine/session.md#the-agents-token-cache)).

## What is verified

<!-- gen:gates-for:gemma4 -->
| Gate | Tier | Model | Evidence |
| --- | --- | --- | --- |
| `gemma4-26b-a4b-trace-cpu` | `verify-cpu` | gemma4_26b_a4b | [gemma4.md](gemma4.md) |
| `gemma4-26b-a4b-trace-f32` | `verify-release` | gemma4_26b_a4b | [gemma4.md](gemma4.md) |
| `gemma4-26b-a4b-trace-f16` | `verify` | gemma4_26b_a4b | [gemma4.md](gemma4.md) |
| `gemma4-26b-a4b-generation-metal` | `verify` | gemma4_26b_a4b | [gemma4.md](gemma4.md) |
| `gemma4-26b-a4b-vision-metal` | `verify` | gemma4_26b_a4b | [vision.md § gemma-4s-projectors](../engine/vision.md#gemma-4s-projectors) |
| `gemma4-26b-a4b-vision-cpu` | `verify-cpu` | gemma4_26b_a4b | [vision.md § gemma-4s-projectors](../engine/vision.md#gemma-4s-projectors) |
| `gemma4-26b-a4b-perplexity-full` | `verify` | gemma4_26b_a4b | [eval.md § the-reference-record-2026-09-24](../benchmarks/record/eval.md#the-reference-record-2026-09-24) |
| `gemma4-e4b-trace-cpu` | `verify-cpu` | gemma4_e4b | [gemma4.md](gemma4.md) |
| `gemma4-e4b-trace-f32` | `verify` | gemma4_e4b | [gemma4.md](gemma4.md) |
| `gemma4-e4b-trace-f16` | `verify` | gemma4_e4b | [gemma4.md](gemma4.md) |
| `gemma4-e4b-generation-metal` | `verify` | gemma4_e4b | [gemma4.md](gemma4.md) |
| `gemma4-e4b-generation-cpu` | `verify-cpu` | gemma4_e4b | [gemma4.md](gemma4.md) |
| `gemma4-e4b-vision-metal` | `verify` | gemma4_e4b | [vision.md § gemma-4s-projectors](../engine/vision.md#gemma-4s-projectors) |
| `gemma4-e4b-vision-cpu` | `verify-cpu` | gemma4_e4b | [vision.md § gemma-4s-projectors](../engine/vision.md#gemma-4s-projectors) |
| `gemma4-e4b-audio-metal` | `verify` | gemma4_e4b | [gemma4.md § audio-input-on-the-e4b](gemma4.md#audio-input-on-the-e4b) |
| `gemma4-e4b-audio-asr` | `verify` | gemma4_e4b | [gemma4.md § audio-input-on-the-e4b](gemma4.md#audio-input-on-the-e4b) |
| `gemma4-e4b-draft-trace-metal` | `verify` | gemma4_e4b | [speculative-decoding.md § the-gemma-4-assistant-heads-2026-09-21](../benchmarks/record/speculative-decoding.md#the-gemma-4-assistant-heads-2026-09-21) |
| `gemma4-e4b-draft-trace-cpu` | `verify-cpu` | gemma4_e4b | [speculative-decoding.md § the-gemma-4-assistant-heads-2026-09-21](../benchmarks/record/speculative-decoding.md#the-gemma-4-assistant-heads-2026-09-21) |
| `gemma4-e4b-vocabulary` | `verify` | gemma4_e4b | [prompt-profile.md](../engine/prompt-profile.md) |
| `gemma4-e4b-perplexity` | `verify` | gemma4_e4b | [eval.md § the-reference-record-2026-09-24](../benchmarks/record/eval.md#the-reference-record-2026-09-24) |
| `gemma4-e4b-perplexity-full` | `verify-release` | gemma4_e4b | [eval.md § the-reference-record-2026-09-24](../benchmarks/record/eval.md#the-reference-record-2026-09-24) |
| `gemma4-e4b-perplexity-4k` | `verify-long` | gemma4_e4b | [eval.md § the-long-context-record-2026-09-24](../benchmarks/record/eval.md#the-long-context-record-2026-09-24) |
<!-- /gen -->

The F16 cache's trace bound (1.0 / 5e-2) is the model's, not the kernels':
unscaled scores over 256- and 512-wide heads amplify key rounding
([record, 2026-09-11](../benchmarks/record/gemma4.md#metal-plan-2026-09-11)).
The 26B-A4B's chunked prefill and verify batches carry their own bounds
(2e1 / 5e-1), since a rounding that moves a router logit swaps an expert
([record, 2026-09-18](../benchmarks/record/gemma4.md#gemma-4-26b-a4b-the-expert-configuration-2026-09-18)).

## Numbers

Tokens per second against the pinned llama.cpp on the same arrays (Metal,
F16 cache, greedy; M4 Pro 48 GB, 2026-10-04/05;
[benchmarks § The benchmarks on macOS 27](../benchmarks/record/benchmarks.md#the-benchmarks-on-macos-27-2026-10-04)),
nuclis / reference:

| Model | Prefill 512 | Prefill 32,639 | Decode 512 | Decode 32,639 | Drafter at 512 |
| --- | ---: | ---: | ---: | ---: | --- |
| 26B-A4B | 521.06 / 584.62 | 136.86 / 325.68 | 54.02 / 68.14 | 31.23 / 43.36 | off |
| E4B QAT | 531.03 / 597.96 | 189.30 / 388.42 | 46.84 / 62.10 | 33.08 / 44.58 | 46.9 → 81.9, draft 6 |

The head-split decode attention raised long-context decode afterwards
(26B-A4B 42.32 at 32,639 from a restored prefix;
[record, 2026-10-15](../benchmarks/record/gemma4.md#long-context-decode-with-the-head-split-attention-2026-10-15)).
The E4B transcribes six corpus clips at 0.88 % WER (the `gemma4-e4b-audio-asr`
gate's set).

## Known limits

- Every cache is allocated for the full session, sliding layers included:
  at 32,768 tokens the F16 session is 11.3 GB on the 12B and 7.4 GB on the
  26B-A4B; a ring for the windowed layers would cut the 12B's to about
  0.35 GB.
- The 26B-A4B's drafter is off (0.67× at 512), and its prefill falls
  furthest behind the reference with context.
- The specialized prefill tiles round activations to half; Gemma's large
  activations make chunked prefill differ from per-token steps more than
  Qwen's (bounded, same greedy token).
- No gate pins a K-quant E4B file; its perplexity (75.56 on WikiText-2
  512×8) is a different checkpoint's, not a bound.
- The E4B does not report a speaker's tone from audio it transcribes
  correctly ([record, 2026-10-27](../benchmarks/record/gemma4.md#audio-input-on-the-e4b-2026-10-27)).
