<!-- kind: card -->
# Muse Glimmer 30B

Meta's dense 30B chat model (`general.architecture = muse-glimmer`):
catalogue entry `muse-glimmer-30b`, **legacy** since 2026-11-10. It still
pulls, verifies, and runs by name, but nothing gates it. The file is
`unsloth/Muse-Glimmer-30B-GGUF` `Muse-Glimmer-30B-UD-Q4_K_XL.gguf`
(15.9 GB), with a vision projector (`mmproj-kquant.gguf`, run by
`vision/muse_glimmer.zig`) and a DFlash drafter (`dflash-kquant.gguf`,
[speculative-decoding.md § The Muse Glimmer DFlash drafter](../benchmarks/record/speculative-decoding.md#the-muse-glimmer-dflash-drafter-2026-09-21));
digests in [catalogue.md](catalogue.md). The adapter is
[inference/src/models/muse_glimmer.zig](../../inference/src/models/muse_glimmer.zig),
the forward `muse_glimmer_runtime.zig` and `muse_glimmer_metal.zig`. The
bring-up, its oracle comparisons and rates are in
[the record](../benchmarks/record/muse-glimmer.md).

## Architecture

From the file's 13 `muse-glimmer.*` keys and the reference graph:

| Fact | Value |
| --- | --- |
| Blocks | 52: sliding where `l mod 4 < 3` (39), global where `l mod 4 = 3` (13) |
| Width, FFN | 6,656, 19,968 (SiLU-gated) |
| Attention | 32 query heads over 2 KV heads, head 128, scale 1/√128 |
| Sliding window | 2,048 positions, the query's own included |
| Position | RoPE on sliding layers only: adjacent pairs `(2i, 2i+1)` over the whole head, base 500,000; **global layers have none** |
| Attention gate | `o = concat(heads) ⊙ sigmoid(Wgate a)`, before the output projection |
| Norms | the embedding RMS-normed without a weight; pre and Q/K norms ε 1e-5; post norms ε 1e-8 (hard-coded); weights stored with the `+1` folded in |
| Head | untied `output.weight`, logits × 0.196116, then `20 · tanh(x / 20)` |
| Vocabulary, context | 202,048; 131,072 declared |

Each layer: `a = norm_attn(h)`; q, k, v and the gate projected from `a`;
Q/K norms per head; RoPE on sliding layers; causal attention (windowed on
sliding layers); the gate; `h += rms_1e-8(Wo o) · w_post_attn`; then
`h += rms_1e-8(Wdown(silu(Wgate f) ⊙ Wup f)) · w_post_ffn` over
`f = norm_ffn(h)`. The DFlash drafter reads the input residuals of layers
2, 14, 26, 38, and 50.

## Files and encodings

The adapter pins every `muse-glimmer.*` key and the vocabulary size, binds
the 731 tensors by name and shape, and rejects any other key, missing or
extra tensor, wrong shape, or encoding outside F32, Q4_0, Q8_0, Q3_K, Q4_K,
Q5_K, Q6_K, IQ4_NL, IQ3_S, and IQ4_XS (`executableEncoding`). The pinned
file is Q4_K with Q5_K on the head and the last seven attention outputs.
Every cache is allocated for the full session, sliding layers included:
1.7 GB at 32,768 tokens in F16, plus 640 MiB for the drafter's five caches.
The tiny `muse-glimmer` variants keep the load and forward paths under
`make test-tiny` ([README.md § Tiny models](README.md#tiny-models)).

## Tokenizer

`tokenizer.ggml.model = gpt2`, `pre = llama4`: 202,048 tokens, 439,802
merges; BOS `<|begin_of_text|>` 200000, EOS `<|end_of_text|>` 200001, EOT
`<|eot|>` 200008. `<|start|>` (200022) and `<|message|>` (200023) are
re-typed as user-defined, so they match in text, as the reference does;
`<|eot|>`, `<|eom|>`, and `<|end_of_text|>` end generation. The role words
are ordinary tokens.

The splitter is [gpt4o.zig](../../inference/src/tokenizer/gpt4o.zig): the
rules the reference's rewritten gpt-4o expression produces, without a
regex engine. A word is one optional non-letter, non-number prefix, then a
letter run where "upper" means *a letter that is not ASCII a–z* and
"lower" *a letter that is not ASCII A–Z* (so `HelloWORLD` → `Hello`,
`WORLD`; `ÀBC` → `À`, `BC`), then an optional ASCII contraction; numbers
go in runs of up to three; other characters as one run followed by any
CR, LF, and `/`; whitespace as qwen35's. Why these rules and how they were
read: [record, 2026-09-19](../benchmarks/record/muse-glimmer.md#tokenizer-2026-09-19).

## Chat template

`tokenizer.chat_template` is 7,167 bytes; the profile `muse_glimmer`
(`profiles/muse_glimmer.zig`) is selected by its digest (it differs from
the Hub repository's `chat_template.jinja`). Clauses in
[prompt-profile.md § Muse Glimmer](../engine/prompt-profile.md#muse-glimmer-muse_glimmer),
the ATEM tool protocol in
[tool-calling.md](../engine/tool-calling.md#muse-glimmer-atem-calls-as-their-own-messages).

- The profile writes `<|begin_of_text|>` itself; the encoder never adds it.
- Turns are `<|start|>ROLE<|message|>…<|eot|>`, content verbatim;
  `<|eom|>` ends only the last tool call of an assistant message that
  another assistant message follows.
- The system turn is the caller's text, `\n\nReasoning strength: LEVEL.`,
  then `\n\n# Valid recipients: "self", "user".`, tool namespaces between.
  The level is `low`, `medium`, `high` (the default), or `xhigh`; the
  shared `off` renders as `low`. Each leading system or developer message
  is its own system turn.
- Without a system message the profile renders the template's default
  (`You are a helpful AI assistant.`, the knowledge cutoff) without its
  date line.
- Reasoning renders as `<|start|>assistant to=self<|message|>…<|eom|>`,
  the answer as `<|start|>assistant to=user<|message|>…<|eot|>`; the
  generation prompt is `<|start|>assistant`.

## What is verified

No gates: the entry is legacy. The record holds the bring-up comparisons
(CPU and Metal within the reference's traces; F16 cache bounds 0.1 / 3e-4,
set by the large late-layer residual stream).

## Numbers

Against the pinned llama.cpp on the same arrays (Metal, F16 cache, greedy;
M4 Pro 48 GB, 2026-10-04/05;
[record](../benchmarks/record/benchmarks.md#nuclis-muse-glimmer-30b-2026-10-05)),
nuclis / reference, tokens per second:

| Prompt tokens | Prefill | Decode |
| ---: | ---: | ---: |
| 512 | 96.41 / 96.43 | 11.62 / 14.24 |
| 32,639 | 59.59 / 75.43 | 7.19 / 9.38 |

The head-split decode attention later raised decode from a restored
prefix to 11.52 at 32,639
([record, 2026-10-15](../benchmarks/record/muse-glimmer.md#metal-plan-2026-09-19)).
The DFlash pair runs at draft 6: 11.62 → 16.09 (1.42×) at 512.

## Known limits

- Decode trails the reference (0.77–0.84×): the gap is the Q4_K matvec's
  per-block cost, not launch parallelism; split-K lost
  ([record, 2026-09-21](../benchmarks/record/muse-glimmer.md#metal-plan-2026-09-19)).
- No ring layout for the 39 windowed layers' caches.
- No gate pins any of its paths.
