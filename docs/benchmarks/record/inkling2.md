<!-- kind: record -->
# Inkling 2 Mini: the record

The dated record behind [the Inkling 2 card](../../models/inkling2.md): the
CPU reference on the real file, as written on its date.

## The CPU reference on the real file (2026-11-26)

M4 Pro, 48 GB; the gates' CPU build (`ReleaseFast`, no Metal) at
`a8a7269`; the pinned `Inkling-2-Mini-UD-Q4_K_XL.gguf` (SHA-256
`e62d463c…`); an F32 cache.

- **Agreement.** On the eight ids of
  `tests/fixtures/inkling2-quick-brown/prompt-tokens.json`, fed one token
  a step, every application's output row agrees with the format's
  definition to 5 % relative RMS, and the last position's argmax is the
  same. The largest errors are at applications where the router's eighth
  expert differs at a near tie; the alternative readings the card lists
  each moved every application further away
  ([inkling2.md § Readings confirmed](../../models/inkling2.md#readings-confirmed)).
- **The pinned trace.** nuclis's own layer and logit rows for the same ids
  are the fixture `inkling2-trace-cpu` compares against; two further runs
  of the same build, and one of the `ReleaseSafe` build, matched it within
  the gate's bounds (`max_absolute` 0.002, relative RMS 1e-4).
- **Speed.** Teacher-forced, one token a step, every core busy (about 8.5
  cores of CPU time):

  | Run | Tokens | Wall time | Seconds a token |
  | --- | ---: | ---: | ---: |
  | `generate`, prompt of 8, `--ctx-size 16` | 8 | 5.33 s | 0.67 |
  | `eval`, one window of 512 | 512 | 378 s | 0.74 |

- **Perplexity**, a sanity check, not a gate: `nuclis eval --backend cpu
  --ctx-size 512 --chunks 1` on WikiText-2's `wiki.test.raw`
  (SHA-256 `173c87a5…`) is 5.238 ± 0.807 over the window's 255 scored
  tokens.

## The stepped Metal plan (2026-11-26)

M4 Pro, 48 GB; `ReleaseFast` with Metal, the plan of the commit after
`97476b9`; the same file and the `inkling2-quick-brown` prompt (8 tokens).

| Run | vs | Max abs | Relative RMS |
| --- | --- | ---: | ---: |
| Metal, F32 cache | the CPU reference's F32 trace, all 385 files | 7.6e-5 | 8.7e-7 |
| CPU, cache rows rounded to F16 | the CPU reference's F32 trace | 2.57 | 0.082 |
| Metal, F16 cache | the same | 2.57 | 0.082 |
| Metal, F16 cache | the CPU with rows rounded to F16 | 3.0e-3 | 8.8e-5 |

- The F16 rounding is the model's, not the plan's: through keys clipped to
  a norm of 22.6 it moves layer 35 by 2.57 at position 5. The logits move
  0.117 (relative 5.1e-3), and the top 5 and the greedy token hold
  (margin 2.95).
- The tiny model, five seeds: stepped against the CPU at most 9.3e-6 /
  2.2e-6. Against a reference rounded alike, the F16 cache is within
  4.3e-4 on four seeds and 1.30 / 0.34 on the fifth; Q8_0 within 6.5e-3
  on three and 0.89 / 0.23 at worst on the other two: the jump of a
  flipped code or expert.
- First speed reading, `nuclis bench`, 64 tokens after the 8-token
  prompt, F16 cache, mean of 3: decode 33.98 tok/s (29.4 ms a token);
  the stepped prefill 36.9 tok/s.

## The rounded references (2026-11-27)

The same machine, file, and prompt; the gates' CPU build at `356d69b`
plus this change, where `generate --backend cpu --kv f16|q8_0` rounds each
cache row as a Metal cache of that precision stores it. The two traces are
pinned as `tests/fixtures/inkling2-quick-brown-f16` and `-q8_0`; a second
run of each matched it exactly.

| Run | vs | Max abs | Relative RMS |
| --- | --- | ---: | ---: |
| CPU, rows rounded to Q8_0 | the CPU reference's F32 trace | 2.93 | 0.128 |
| Metal, Q8_0 cache | the CPU reference's F32 trace | 2.91 | 0.127 |
| Metal, F16 cache | the CPU with rows rounded to F16 | 2.96e-3 | 8.8e-5 |
| Metal, Q8_0 cache | the CPU with rows rounded to Q8_0 | 0.140 | 2.5e-3 |

- Q8_0 holds every position within 6e-5 through application 10. At
  application 11 or 13, positions 2 to 7 step to about 1e-3 at once: a
  block's code that a last-place difference in the key moves by one. The
  rest is that step carried forward.
- The Q8_0 rounding moves the logits 0.242; the greedy token holds.

## The chunked plan (2026-11-27)

The same machine and file, `nuclis bench` at `356d69b` plus this change,
Metal, greedy, F16 cache; the prompts are the first 512 and 4,096 tokens
of `docs/spec.md` (raw), 64 generated, context prompt + 128, mean of 3.
The machine read 10 to 15 % apart between runs this day, so each A/B
below alternates its builds.

| Prompt | Prefill tok/s | First token ms | Decode tok/s |
| ---: | ---: | ---: | ---: |
| 8 (the trace prompt) | 80.3 | 99 | 29.7 |
| 512 | 311.5 | 1,643 | 29.3 |
| 4,096 | 292.0 | 14,026 | 27.2 |

- The stepped plan at `356d69b`, same session and method: prefill 31.5
  tok/s, decode 29.1 (and 28.7 at a 65,536 context); this change's
  decode, run straight after, 29.7 and 29.3. The 33.98 of the stepped
  record was a cooler machine, not a faster plan.
- Chunk size, 4,096 tokens, builds alternated twice: 512 rows 289 and
  278 tok/s, 1,024 rows 331 and 313. A first pass read 256 rows 196 and
  180 (512 tokens). `preferredChunk` is 1,024.
- The expert path's crossover, first token over 2 generated, mean of 5,
  by threshold (the matvec below it, the gathered tile from it):

  | Prompt | 8 | 16 | 32 | 64 |
  | ---: | ---: | ---: | ---: | ---: |
  | 8 | 113 | 107 | 92 | 127 |
  | 12 | 240 | 143 | 141 | 195 |
  | 16 | 273 | 221 | 191 | 175 |
  | 24 | 402 | 360 | 280 | 320 |
  | 32 | 423 | 434 | 393 | 355 |
  | 48 | 452 | 452 | 452 | 454 |

  32 is never worst and best or near it from 8 to 24; `matvec_expert_rows`
  is 32. Under 16 rows the verify attention runs; past it the
  materialized chunk.
- Prefill profile, 512 tokens (one encoder a dispatch, so its times are
  for shares only): 2,794 dispatches a chunk. The NVFP4 expert tiles take
  51 % (gate-up 312 ms, down 160 ms of 858); the Q8_0 projections 14 %;
  the bias table 4.6 %; `l2_clip_rows` 4.5 %; the convolutions 4.1 %.
  Stored blocks 20 to 23 hold Q8_0 experts: on the generic F32 tile they
  took 62 ms of 935 for 2 of 24 applications, on the half tile added here
  47 ms, about the NVFP4 tiles' cost a block.
- Decode profile, 8-token prompt: 2,490 dispatches a token. The expert
  matvecs take 19 % (NVFP4 gate-up 216 GB/s, down 176 GB/s); the
  controller's F32 matvecs 12 % at 22 GB/s, being 192 calls of 64 × 2,304;
  the Q8_0 projections 17 %; the head 222 GB/s.
- Correctness of the chunked paths (`generation-check --metal`, the real
  file, these thresholds): the generic F32 tiles' chunk of 32 over 40
  tokens 1.8e-5 / 2.8e-7 from stepping; the half tiles 0.117 / 3.5e-3 at
  chunks of 64 and 48, and 9.7e-4 / 2.6e-5 at 32, whose experts take the
  matvec; F16 cache chunked 0.170 / 9.7e-3, Q8_0 0.262 / 1.2e-2; eight
  verify rows within 2.6e-3 / 1.9e-4 of stepping; the greedy token held in
  each.

## The pinned main file (2026-11-27)

The same machine; nuclis at `1d63ba0` plus this change; the index's new
pin, `main` at `8078d149…` (`inkling2_loop`, SHA-256 `e6427665…`).

- **The same model.** Every CPU trace pinned from the earlier file
  (`inkling2-trace-cpu`, `-f16`, `-q8_0`, 385 files each) matches at max
  abs 0; the Metal traces read 7.6e-5, 2.96e-3, and 0.140, as before.
- **Perplexity**, `nuclis eval --backend metal`, F16 cache, WikiText-2's
  `wiki.test.raw`, eight windows of 512: 7.6582 ± 0.43, every window's
  running value equal to the earlier file's to four places (window 1
  5.2454). The CPU's first window, F32 cache, is 0.137 % below
  (395 s); rounded to F16 (`--kv f16`), 0.102 %. The rest is likely the
  chunked prefill's half-precision expert tiles (§ The chunked plan), not
  measured apart.
- **Routing**, `nuclis bench --expert-counts`, Metal, the first 1,024,
  8,192, and 32,768 tokens of `wiki.test.raw` (raw), context 34,816.
  Mean distinct experts over R consecutive prompt rows:

  | Prompt | R = 2 | 4 | 8 | 9 | 16 | 1,024-row chunk |
  | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | 1,024 | 12.8 | 20.2 | 30.7 | 32.8 | 44.2 | 109.5 |
  | 8,192 | 12.5 | 19.5 | 29.4 | 31.3 | 42.1 | 110.8 |
  | 32,768 | 12.6 | 19.6 | 29.6 | 31.6 | 42.5 | 110.1 |

  A chunk's lists hold about 74 rows each, 45 to 48 % of them under 32.
  The next token shares 3.2 to 3.5 of its 8 experts with the prompt's
  last, and 2.8 to 4.2 in decode.

## The DSpark drafter, priced (2026-11-27)

The same machine; nuclis at `ec6ffcc`, Metal, F16 cache, context 4,096;
`nuclis bench`, the first R tokens of `wiki.test.raw` as a prompt, mean
of 5. A prompt of R tokens stands for a verify batch of R rows at a
shallow depth (`--verify-rows` needs a drafter):

| Rows | 1 | 3 | 4 | 5 | 9 |
| --- | ---: | ---: | ---: | ---: | ---: |
| First token, ms | 32.2 | 54.3 | 63.7 | 71.5 | 131.7 |

Decode reads 30.5 tokens/s, 32.7 ms a token. A block pays only if
`E · 32.7 > t(verify R) + t(draft) + t(recover)`, E being the tokens a
verify keeps; before the draft and the recovery, E must pass 1.66 at 3
rows, 1.95 at 4, 2.19 at 5, and 4.03 at 9. Recovery is not free: the
convolution history is recurrent, so a partial accept rewinds and replays
the kept rows. The new drafter (`3da00cf7…`), measured in session on prose
and code, greedy and at the profile's sampling, keeps too few to clear
these with room for its own pass and the replay at any draft length, so
it is not implemented. A later attempt starts from these costs, the
routing above, and a recovery that copies the history instead of
replaying it.

## The baseline (2026-11-27)

The same machine on AC power; nuclis at `0c6a67b` plus the `inkling2`
workloads row, `ReleaseSafe`, Metal, F16 cache, no drafter, greedy; the
pinned `main` file. `make workload NAME='inkling2/*'`: the acceptance
arrays made by `nuclis tokenize` into
`tests/fixtures/run-2026-11-27-inkling2/`, 128 tokens generated, context
32,768 (65,536 for the last column); three runs to 16,384, one past
([nuclis-2026-11-27-inkling2.json](../nuclis-2026-11-27-inkling2.json),
[-64k](../nuclis-2026-11-27-inkling2-64k.json)):

| Prompt tokens | 512 | 4,096 | 16,384 | 32,639 | 65,407 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Decode tok/s | 25.63 | 26.40 | 21.37 | 21.10 | 18.51 |
| Prefill tok/s | 282.43 | 325.98 | 274.39 | 281.95 | 232.03 |

The 64K run's 32,639 read 22.55 and 287.13. The shapes, mean of 3 (2 for
4,096): `prose512` decode 27.87, prefill 277.77; `prose4096` 27.35 and
339.87; `code` (a 9-token raw prompt) 31.06 decode, first token 143 ms.
The session is 2,630 MiB at a 32,768 context and 3,654 MiB at 65,536.

**Memory** (`scripts/bench/nuclis_mem_usage.py --watch 2` over a 32,639
prompt at context 32,768, peaks): the mapped weights resident 17.63 GB,
the process footprint 3.80 GB (Metal 0.90 GB, CPU 2.82 GB), 21.43 GB in
all.

**Where prefill goes** (`bench --profile-prefill`, the 32,639 array, one
run): 122.0 GPU seconds, 3,696 ms a 1,024-row chunk. The NVFP4 expert
tiles take 38.8 % (gate-up 25.7 %, down 13.1 %) and the Q8_0 experts of
stored 20–23 3.8 %; attention 24.5 % (scores with the relative bias 13.2 %,
values 8.5 %, the sink statistics 2.1 %); the Q8_0 projections 12.5 %;
`l2_clip_rows` 4.2 %, the bias table 4.1 %, the convolutions 3.3 %. At the
same length the 26B-A4B spent 236.0 seconds, 66 % of them in global
attention and 12 % in experts ([gemma4.md](gemma4.md#prefill-at-32k-todays-baseline-and-its-split-2026-11-18)):
Inkling's windows keep attention cheap and its 48 expert applications a
token cost the time instead.

## What a verify batch costs (2026-11-27)

The same machine; nuclis at `9447861`, `ReleaseSafe`, Metal, F16 cache,
context 4,096; the drafter is not bound, so every number here is the
target's own.

**Sampling.** The 512-token prose array, 128 tokens, six interleaved
pairs: greedy 28.16 to 28.46 tokens/s (mean 28.30), the profile's
sampling (temperature 1, top-k 0, top-p 1) 18.49 to 18.86 (mean 18.75),
34 % slower. Over 128 tokens the sampled path spends 18 ms a token more,
the host sorting all 200,058 logits.

**The batch split** (`bench --profile-prefill`, prompts of the first R
tokens of `wiki.test.raw`, 20 runs; profiling serializes the kernels, so
a step reads above plain decode's 28.7 GPU ms):

| Rows | 1 | 2 | 4 | 8 | ms a row |
| --- | ---: | ---: | ---: | ---: | ---: |
| Command buffer | 32.0 | 42.3 | 61.0 | 97.2 | 9.31 |
| Kernels attributed | 39.2 | 48.0 | 67.3 | 103.7 | 9.21 |
| NVFP4 experts | 7.88 | 14.89 | 29.03 | 58.45 | 7.22 |
| Q8_0 experts (stored 20–23) | 1.16 | 2.29 | 4.65 | 9.42 | 1.18 |
| Q8_0 projections and head | 11.75 | 13.75 | 14.49 | 14.83 | 0.44 |
| Everything else | 18.41 | 17.07 | 19.13 | 21.00 | 0.37 |

The experts are 91 % of what a row adds: every row reads its own eight,
and nothing else grows with the batch. Over 1,024 prose rows
(`--expert-counts`), 2 rows touch 12.8 distinct experts, 4 touch 20.2
and 8 touch 30.7: 0.80, 0.63 and 0.48 of the experts read today. An
expert kernel that read each distinct expert once would, by those
ratios (an estimate, not a measurement), take a 4-row batch's experts
from 33.7 ms to about 21 ms.

**The Q8_0 projections** (`make bench-matvec-rows ARGS='8 inkling'`,
synthetic weights of Inkling's shapes): the multi-row matmul holds
223 to 262 GB/s from 2 to 8 rows on 3072×2304, 2304×3072 and 9216×2304,
and the 201,024-row head takes 2.12 ms at 2 rows and 2.21 ms at 8,
against 1.90 ms for one. A few-row Q8_0 kernel has nothing to win. A
verify that computes the head for each row with the 1-row matvec instead
pays 1.90 ms a row: 15.3 ms at 8.

**One row.** The 1-row batch path runs the loop's 2304×64 F32 gain and
forget projections as a matvec at 24 GB/s (4.74 ms a step, 192 of them);
the 2-row path's matmul does the same in 2.08 ms. Plain decode does not
take that path: routed to the matmul, decode's GPU time stayed at 28.6 ms
a token, and the change was not kept. Decode's wall time is 31.1 ms a
token against 28.6 on the GPU.

**Depth** is not measured here: `bench --verify-rows` needs a drafter,
so the 4,096 and 16,384 tails of 2, 4 and 8 rows move to the drafter's
binding.

What it decides for the drafter: sampling from the device's top-k is
worth its own change (a third of sampled decode); the expert kernel is
the per-row cost to cut, not the Q8_0 projections; and a verify computes
its R rows' logits with the matmul.

## Sampling from the device's top-k (2026-11-27)

The same machine; nuclis at `870cec6` (before) and the same tree with
the full distribution drawn from the readback (after), `ReleaseSafe`,
Metal, F16 cache, context 4,096, no drafter. `bench --prompt-tokens
prompt-512.json --max-tokens 128 --repeat 1 --warmup 0 --temperature 1
--seed i` (the profile's options: top-k 0, top-p 1, min-p 0), eight
interleaved pairs, seeds 1 to 8, the order alternating per pair. Another
session shared the machine, so the numbers are provisional:

| Decode tok/s | Mean | Range |
| --- | ---: | ---: |
| Before (sorts 200,058 logits on the host) | 20.78 | 20.68 to 21.17 |
| After (draws from the 256-entry readback) | 31.79 | 31.66 to 31.93 |
| Greedy, after (three runs) | 34.10 | 33.87 to 34.23 |

Sampled decode is 53 % faster, 16.7 ms a token less, and 7 % below greedy
where it was 34 % below. No token of the 1,024 the pairs generated fell
back to the full logits; prefill is unchanged (305.9 to 308.9 tokens/s on
both sides).

**The same tokens.** `generate --json` on the same array, 128 tokens:
seeds 1 to 5 at temperature 1 emit the identical sequence before and
after, and so do seeds 1 and 2 at temperature 1.5 (one fallback) and 2
(101 and 114 of 128 fell back: a hot distribution puts the draw past the
256 candidates). Seeds 1 and 2 with `--logits`, which keeps the full
reference path, match the readback runs. At temperature 2 decode is 21.99
to 22.91 tokens/s after against 21.38 to 21.49 before: a fallback costs
what the old path did.
