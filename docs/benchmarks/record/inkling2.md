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
