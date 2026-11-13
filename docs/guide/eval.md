<!-- kind: guide -->
# Teacher-forced evaluation

`nuclis eval` measures a model's perplexity on a text: the one number that
ranks a quantization, a cache precision, or a kernel's rounding against the
reference over thousands of positions, where the trace gates see two or
three.

## The method

*Teacher forcing* feeds every position the text's actual next token, never
a sampled one, and reads the model's log-probability of that token from the
position's logits. The negative log-likelihood (NLL) averaged over the
scored tokens is the cross-entropy in nats; its exponential is the
perplexity.

`eval` follows llama.cpp's `llama-perplexity` exactly, which made the
first pinned references while nuclis was bootstrapped
([models/README.md § 4](../models/README.md#4-validate-against-nuclis-itself)),
so its numbers compare with that tool's output on the same file. Below,
*the reference* is that tool's run:

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
([§ The reference file](#the-reference-file)). It takes that run's `ctx` and window
count unless the flags state them (another `ctx`, or more windows than
it ran, is `ReferenceMismatch`; fewer compare against its running value
after that window, `chunk_ppl`, which the fast gates use), refuses a file whose SHA-256 differs from the one the reference ran, and
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
`<|begin_of_text|>` for Muse Glimmer), which is what the reference did:
it forced `add_bos_token` on for every Gemma 4 file, including the 26B-A4B
whose metadata says false. A file without a profile falls back to its own
`tokenizer.ggml.add_bos_token` (`Engine.textBos`). The token arrays equal the reference's
([record, 2026-09-24](../benchmarks/record/eval.md#bos-and-tokenization-2026-09-24)).

Gemma's SPM-style splitter makes each line one BPE piece, so wikitext's
paragraph lines are long pieces; the queued merge
([tokenizer.md § BPE](../engine/tokenizer.md#bpe-and-id-to-byte-decoding))
keeps their work linear, and `eval` widens the input and work bounds in
proportion to the text (capped at 16 MiB). The 1.29 MB test text encodes
in about 0.26 s on Gemma.

## What is pinned

The references compare at **0.5 %**, except the Gemma 4 26B-A4B's at 1 %:
on the mixture-of-experts file a router near-tie swaps an expert and the
change cascades through the window, and the reference disagrees with
itself there as much as nuclis disagrees with it (the fixture's
`tolerance` carries the reason). The gates, with the numbers they pinned
in [the reference record](../benchmarks/record/eval.md#the-reference-record-2026-09-24) and [the
long-context record](../benchmarks/record/eval.md#the-long-context-record-2026-09-24):

| Gate | Tier | Windows |
| --- | --- | --- |
| `qwen38-perplexity`, `gemma4-e4b-perplexity` | `verify` | two of 512 |
| `gemma4-26b-a4b-perplexity-full` | `verify` | eight of 512 |
| `qwen38-perplexity-full`, `gemma4-e4b-perplexity-full` | `verify-release` | eight of 512 |
| `gemma4-e4b-perplexity-4k` | `verify-long` | four of 4,096 |

The Gemma 4 12B and Muse Glimmer have no gate. The Gemma 4 26B-A4B's
difference at 4,096 (−1.79 %) is open and ungated, its fixture
`tests/fixtures/perplexity/gemma4-26b-a4b-wikitext2-c4096x4.json` kept as
the evidence.

## The reference file

A reference is one JSON object; `--reference` reads these fields and
ignores the rest:

| Field | Meaning |
| --- | --- |
| `tool`, `revision` | what ran it: `nuclis eval` at a nuclis revision, or `llama-perplexity` at a llama.cpp revision |
| `text_sha256` | the text's digest; another file is refused |
| `ctx`, `chunks` | the window and the windows run |
| `ppl`, `ppl_error` | the perplexity after every window and its standard error |
| `chunk_ppl` | the running perplexity after each window (4 decimals), so fewer windows compare |
| `tolerance` | optional: a bound wider than 0.5 % for a file whose reference disagrees with itself |

`nuclis eval --write-reference <file>` writes one, never over an existing
file, adding the provenance a pinned file carries: `schema_version`,
`executor` and `kv`, `model_file`, `model_bytes`, and `model_sha256` (the
sidecar's when it records this size, else hashed), `text_file`, the
`flags`, and the `date`. `cross_check` and `note` are written by hand.

## Regenerating a reference

A re-pinned file's reference is nuclis's own: Metal's eight windows,
cross-checked by one CPU window
([models/README.md § Re-pinning a file](../models/README.md#re-pinning-a-file));
`tests/fixtures/perplexity/qwen38-wikitext2-c512x8.json` is the first.

```sh
make eval-corpus
./zig-out/bin/nuclis eval --model <name> --backend metal \
  --file .reference/eval/wikitext-2-raw/wiki.test.raw --chunks 8 \
  --write-reference tests/fixtures/perplexity/<family>-wikitext2-c512x8.json
./zig-out/bin/nuclis eval --model <name> --backend cpu --kv f32 \
  --file .reference/eval/wikitext-2-raw/wiki.test.raw \
  --reference tests/fixtures/perplexity/<family>-wikitext2-c512x8.json --chunks 1
```

The second line is the cross-check; its result goes into `cross_check`.
The Gemma 4 references are `llama-perplexity` per-token runs
([record, 2026-09-24](../benchmarks/record/eval.md#the-reference-and-its-two-paths-2026-09-24)),
pinned before nuclis became its own reference; they stay until their file
is re-pinned. How they were made: [record](../benchmarks/record/eval.md#the-reference-and-its-two-paths-2026-09-24).

## Limitations

- Bonsai 2 has no reference record, and no gate.
- The CPU reference runs `eval` token by token (about 11 s a token on the
  Gemma 4 12B; one window of Qwen3.8 takes 28 min), so
  a whole CPU record is out of reach; one window cross-checks Metal's.
- One corpus, at windows of 512 and 4,096. Windows past 4,096 and a code
  corpus are not recorded.
- The scored rows are read back whole (about 250 MB per 256 rows) and
  reduced on the host; a device log-softmax would remove the copy.
