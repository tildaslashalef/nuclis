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
