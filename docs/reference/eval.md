# Teacher-forced evaluation

`nuclis eval` measures a model's perplexity on a text: the one number that
ranks a quantization, a cache precision, or a kernel's rounding against the
reference over thousands of positions, where the trace gates see two or
three. The unit that built it is `APPS-14` in the
[engineering log](../engineering-log.md); `MODL-26` is the Gemma 4 `verify`
fix it surfaced.

## The method

*Teacher forcing* feeds every position the text's actual next token, never
a sampled one, and reads the model's log-probability of that token from the
position's logits. The negative log-likelihood (NLL) averaged over the
scored tokens is the cross-entropy in nats; its exponential is the
perplexity.

`eval` follows the reference's `llama-perplexity` exactly, so its numbers
compare with that tool's output on the same file:

1. The whole file is tokenized raw: no template, special markers not
   parsed. When the model has a BOS (below), it is prepended to the array.
2. The array is cut into whole windows of `ctx` tokens (default 512); the
   remainder is dropped. `--chunks N` takes the first `N`.
3. Each window runs from an empty session, with the BOS written over its
   first token (a window's first token is context, never scored).
4. Only the second half is scored: row `j` for `ctx/2 ≤ j < ctx − 1`
   predicts token `j + 1`. Every scored token has at least `ctx/2` tokens of
   context, so the number is not dominated by cold starts. At 512 that is
   255 tokens per window.
5. Per row, `−log softmax(row)[next]`: the maximum is subtracted in F32 and
   the exponentials summed in F64 (`eval.rowNll`). The perplexity is
   `exp(mean NLL)`. Its error is the NLL's sample standard error scaled
   by the perplexity (first order), the reference's formula. The running
   estimate after each window is printed as the reference prints it
   (`[1]4.3473,[2]6.1654,…`).

`--reference <json>` reads a pinned reference run
(`scripts/reference-perplexity.py`). It takes that run's `ctx` and window
count unless the flags state them (a conflict is `ReferenceMismatch`),
refuses a file whose SHA-256 differs from the one the reference ran, and
fails beyond **0.5 %** relative difference in perplexity. The report
(`--json`, schema 1) carries the text's size, digest, and token count,
the windows, the BOS id, the NLL, the perplexity and its error, the
running estimates, the time, and the comparison.

## The engine seam

`Model.prefillRows(tokens, vocabulary, rows, observer)` consumes tokens
like `prefill` but writes **every** row's logits
(`inference/src/engine.zig`). A GPU plan that declares `prefillRows`
(all three: `qwen35_metal`, `gemma4_metal`, `muse_glimmer_metal`) records
its chunk's layer stack once, then the output norm, the head over every
row, and the family's logit shaping (Bonsai's rotation, Gemma's soft-cap,
Muse's scale and soft-cap), all in one command buffer. The head's output
buffer is created on the first call
(`matmulPadded(chunk) × vocabulary`, about 250 MB for Qwen's 248,320 ids)
and kept for the plan's life, so no other load pays for it. The CPU
reference, and a plan without the method, step token by token.

`eval` feeds the unscored first half of each window as a plain `prefill`
and the scored half through `prefillRows` in pieces of one prefill chunk
(256 rows), bounding the host rows at about 250 MB. The whole run is
cancellable (Ctrl-C at a layer boundary); it has no clock bound, since its
length is its window count.

`generation-check` pins the seam: through the generic F32 tiles, every row
of `prefillRows` over the 70-token prompt must be the stepped logits at its
position within 5e-3 / 2e-4 (summation order only), for every family.

## BOS and tokenization

The encoder never adds BOS. A raw text starts with the *profile's* opening
token (`Profile.bosToken`: none for Qwen3.8 and Bonsai, `<bos>` for Gemma 4,
`<|begin_of_text|>` for Muse Glimmer), which is what the reference does:
it forces `add_bos_token` on for every Gemma 4 file, including the 26B-A4B
whose metadata says false. A file without a profile falls back to its own
`tokenizer.ggml.add_bos_token` (`Engine.textBos`). On wikitext-2 the token
arrays equal the reference's exactly: all 4,096 ids of the first eight
windows, checked against its `--kl-divergence-base` dump for Qwen3.8 and
Gemma 4 12B.

Gemma's SPM-style splitter makes each line one BPE piece, and the merge
loop rescans the piece, so its work grows with the square of the line.
Wikitext's paragraph lines exceed the chat prompt's work bounds
(`WorkLimitExceeded`), so `eval` scales both bounds with the text, which is
itself capped at 16 MiB. The 1.29 MB test text tokenizes in about 3 s on
Gemma. `nuclis tokenize` keeps the chat bounds and refuses 40 KB of
wikitext on Gemma.

## The reference and its two paths

The reference's default run decodes a batch of 2,048 tokens (four windows
of 512) per call, through its matrix-matrix kernels. On Gemma 4 12B that
batched path disagrees with its own per-token path (`-ub 1`, one token per
decode call, the path its trace harness exercises) by 1.2 %: **597.36**
against **590.13** over the same 4,096 tokens. Per scored token the two
reference runs differ by a median of 0.049 nats, p99 1.49. Our first
comparison against the batched run measured the same spread and a −1.41 %
perplexity, and looked like a Gemma bug. The localization:

- our Metal `step` path and `prefill` path agree with each other (589.23 and
  588.95), with an F16 or an F32 cache (589.20);
- the reference is stable across its own cache and attention settings
  (597.36 with an F16 or an F32 cache, 597.52 with flash attention off),
  and across batch sizes (597.25 at 512 tokens a batch);
- on the worst window (window 7 of ctx 32, a 12.7-nat difference at one
  position), `generate --kv f32 --trace-dir` and the trace harness agree at
  every one of 48 layers × 25 positions within 1e-5 relative RMS, logits
  1.4e-6;
- against the per-token reference, our per-token NLL differs by a median of
  0.006 nats, p99 0.18.

So the pinned references are **per-token runs** (`--ubatch 1`, which adds
`-b <ctx> -ub 1`), for every family. The per-token reference is slow
(it decodes one token per call) but is written once.

## The reference record (APPS-14, 2026-09-24)

wikitext-2-raw `wiki.test.raw` (1,290,590 bytes, SHA-256 `173c87a5…`,
fetched by `make eval-corpus`), eight windows of 512 (4,096 tokens, 2,040
scored), F16 cache, Metal, Apple M4 Pro 48 GB, Zig 0.16.0 ReleaseSafe;
llama.cpp `7620399` built with the recipe of
[reference-baseline.md](reference-baseline.md) plus the `llama-perplexity`
target.

| Model (file) | nuclis | reference, per-token (pinned) | difference | reference, batched | nuclis eval time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8-27B (UD-Q4_K_M) | 6.6696 ± 0.3626 | 6.6696 ± 0.3626 | −0.000 % | 6.6655 (+0.06 %) | 55.8 s |
| Gemma 4 12B (UD-Q4_K_XL) | 588.954 ± 73.12 | 590.133 ± 73.25 | −0.200 % | 597.363 (−1.41 %) | 24.1 s |
| Gemma 4 26B-A4B (QAT UD-Q4_K_XL) | 1107.838 ± 130.51 | 1113.719 ± 131.26 | −0.528 % (bound 1 %) | 1110.492 (−0.24 %) | 12.7 s |
| Muse Glimmer 30B (UD-Q4_K_XL) | 6.6250 ± 0.3672 | 6.6249 ± 0.3672 | +0.001 % | 6.6264 (−0.02 %) | 53.3 s |

The time is the evaluation after the load: 4,096 tokens fed, 2,040 scored
(Qwen 73 tok/s). The per-token references took minutes each and the
batched ones under a minute (Qwen 44 s).

The Gemma 4 perplexities are in the hundreds because the instruction-tuned
files, fed raw encyclopedia text with no chat template, spread probability
over continuations the text does not take. The reference measures the same
on the same tokens, so the comparison stands. At that level the far tail
dominates: on Gemma 4 12B, 116 of the 2,040 targets sit more than 16 nats
below their row's maximum.

Per scored token, on the rows the dump does not clamp:

| Pair | median \|Δ NLL\| | p90 | p99 | max |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.8: nuclis vs reference (batched) | 0.0018 | 0.012 | 0.037 | 0.077 |
| Gemma 4 12B: nuclis vs reference (per-token) | 0.006 | 0.033 | 0.18 | 2.83 |
| Gemma 4 12B: reference batched vs per-token | 0.049 | 0.31 | 1.49 | 3.31 |
| Gemma 4 26B-A4B: nuclis vs reference (per-token) | 0.053 | 0.38 | 1.12 | 3.66 |
| Gemma 4 26B-A4B: reference batched vs per-token | 0.051 | 0.37 | 1.18 | 3.88 |

**The 26B-A4B's bound is 1 %.** On the mixture-of-experts file a
perturbation of a router logit past a near-tie swaps one of a token's eight
experts, and the change cascades through the window (gemma4.md records the
same effect between our chunked and stepped paths). The reference disagrees
with itself there as much as we disagree with it (the last two rows), and
by 0.29 % in perplexity. Our two paths sit 0.53 % (prefill) and 0.69 %
(per-token `step`: 1106.03) below its per-token run, with a mean
per-token difference of −0.003 nats: noise that does not average out in
2,040 tokens dominated by the tail, not a bias. The fixture carries
`tolerance: 0.01` and the reason; every other family keeps 0.5 %.

Each family's figure is a gate (`make gate NAME='*-perplexity'`, the
`verify` tier): `nuclis eval --reference
tests/fixtures/perplexity/<family>-wikitext2-c512x8.json`.

## Gemma 4's verify rows (MODL-26, 2026-09-24)

Building `prefillRows` surfaced a gap in a sibling path: Gemma 4's Metal
`verify` ran the head over every row without the final soft-cap, so sampled
speculative acceptance on Gemma drew from uncapped logits (row 0 of
`<bos>Hello,` 27.8 from the stepped logits). Fixed and gated; the facts are in
[speculative-decoding.md § The Gemma 4 assistant heads](speculative-decoding.md#the-gemma-4-assistant-heads-modl-19).

## Regenerating a reference

```sh
make eval-corpus
cmake --build .zig-cache/reference/llama.cpp/build --target llama-perplexity --parallel 8
python3 scripts/reference-perplexity.py --model <file.gguf> \
  --text .zig-cache/eval/wikitext-2-raw/wiki.test.raw --ctx 512 --chunks 8 \
  --ubatch 1 --output tests/fixtures/perplexity/<family>-wikitext2-c512x8.json
```

The JSON records the tool, revision, flags, model file name and size, the
text's name and SHA-256, the windows, the perplexity, its error, and the
running estimates. To compare per token, the same binary with
`--kl-divergence-base <file>` writes the token array (`_logits_`, `n_ctx`,
`n_vocab`, `n_chunk`, then `n_chunk · n_ctx` int32 ids) followed by every
scored row's log-probabilities as uint16 steps above
`max − 16`. A target below that floor reads as the floor, so compare only
unclamped rows.

## Limitations

- Bonsai 2 has no reference record: its encodings decode only in the
  PrismML fork, which was not built with `llama-perplexity`.
- The CPU reference runs `eval` token by token (about 11 s a token on the
  Gemma 4 12B), so its record would take hours; none is pinned.
- One corpus, one window size. Long-context perplexity (windows of 4K–32K)
  and a code corpus are not recorded.
- The scored rows are read back whole (about 250 MB per 256 rows) and
  reduced on the host; a device log-softmax would remove the copy.
