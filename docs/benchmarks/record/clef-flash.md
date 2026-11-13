<!-- kind: record -->
# clef-flash: the record

The dated record behind [the clef-flash card](../../models/clef-flash.md): the bring-up checks against Cloudflare's reference head and the first time per decision, as written on 2026-10-02.

## Checked against the oracle (2026-10-02)

The backbone is the qwen35 family nuclis already validates for Qwen3.8
and the projector is Qwen3.8's at another width; neither got a Python
reference of its own. `config.json` and the GGUF headers show no flag that
changes the arithmetic (`attn_output_gate` as Qwen3.8, no MTP layer, text
mrope with identical axes). The new parts are checked against the
reference without its backbone (`.reference/clef-venv`, torch 2.11.0,
transformers 5.10.2):

| Check | How | Result |
| --- | --- | --- |
| The sequence | `clef-reference.py sequence` wrote the reference's ids and spans for 10 requests (the card's two examples, Laya's shapes, a 3,931-token log, a 6-question conversation, two image requests); `clef-check sequences` rebuilds them | ids and every span equal, image token counts (80; 64 + 192) equal |
| The head alone | seeded random inputs through both heads (`head --synthetic`) | max logit difference 1.4e-6 |
| The head on real inputs | nuclis's Metal backbone dumps each request's rows and lexical vectors; the reference head turns them into `head.json` | nuclis's head within 6e-7 |
| CPU against Metal | the CPU reference's rows against the Metal plan's | `card_invoice`: rows' cosine ≥ 0.999997, RMS ratio 6.2e-4, logits 8.5e-5; `noul_error` 7.5e-5, `noul_criteria` 5.8e-4 |
| The sanity set | every request has obvious answers (15 text questions, 2 image) | every one picked, on both backends |

For scale, the reference's own deployment runs the head in BF16, which
moves the logits by 0.007–0.03 from its F32 run on the same inputs.

The gates: `clef-sequences` (tokenizer only), `clef-metal` (every request,
images included, against `head.json` within 1e-3 and the sanity set),
`clef-cpu` (the CPU tier: the two requests under 160 tokens, about 8 min,
within 2e-3).

## Time per decision (2026-10-02)

Measured 2026-10-02 on the M4 Pro (48 GB), Metal, `nuclis decide
--json` at `ReleaseSafe` (the gates' build), one `noul` question over a
log state, load excluded (0.5 s, the head's decode included):

| State | Tokens per sequence | 1 state | 10 states | 50 states |
| --- | ---: | ---: | ---: | ---: |
| about 500 tokens of log | 637 | 2.37 s | 23.0 s (2.30 s each) | 116.1 s (2.32 s each) |
| about 2,000 tokens of log | 2,162 | 8.51 s | 82.4 s (8.24 s each) | not run (about 7 min) |

The time is the backbone's prefill, about 260–290 tok/s (with the 512-row
chunk; `clef-check` measures 14.6 s for 3,931 tokens), linear in the
states: one sequence at a time, so batching requests saves nothing.
Questions are almost free: they add their schema tokens to one pass,
where Laya pays a pass per question. An image adds its projector pass
(`image_red`, 80 image tokens: 1.24 s for 224 tokens against 0.61 s for a
143-token text request). A debug build is about 1.5× slower (the head on
the CPU).
