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
