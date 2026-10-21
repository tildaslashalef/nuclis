# TODO — active plan

This file is the queue: the units agreed and not yet merged, each with its
design, and a *Where we are* note. A unit is one session, one branch, and
one pull request, whose description is the record of the work; when it
merges, its section is deleted here, and when the last one goes, this file
is emptied back to this header and its *Deferred* list. Requirements live in [docs/spec.md](docs/spec.md); the
engine map in [docs/architecture.md](docs/architecture.md); how to build,
test, and measure in [docs/development.md](docs/development.md).

Session protocol (also in [AGENTS.md](AGENTS.md)): read this file first. If
it lists work, summarize *Where we are* and ask the user how to continue. If
it is empty, ask what to work on and write the agreed plan here.

## Where we are

Theme: speculation for Saluki 27B, on the branch `saluki-mtp-companion`
(one session per unit, no pushes or pull requests until the user says).
Qwen's MTP companion binds and drafts on Saluki, but the verdict is off:
the 8-row verify batch costs 2.1× a single-row step (Q4_K_M: 1.45×)
(`docs/engine/speculative-decoding.md` § The Qwen MTP companion). After
this unit: re-measure, then a DFlash2 drafter unit, first on the Q4_K_M.

| Unit | State |
| --- | --- |
| A cheaper few-row verify on Saluki | in progress |

## A cheaper few-row verify on Saluki

Base: `c6ebb02`

1. **Profile.** `nuclis bench --speculative on --verify-rows 8 --accept
   0 --profile` against a single-row `--profile`, on Saluki at 512 with
   the companion (a scratch `NUCLIS_HOME` whose `saluki-27b` entry names
   it as `mtp`). Rank the verify buffer's kernels by time.
2. **The lead.** `specializedMatmulFrag` in
   `inference/src/backends/metal/root.zig` has `_f2` fragment tiles for
   IQ2_XXS and Q2_K, and none for IQ2_XS, IQ2_S, IQ3_XXS, IQ1_S, or IQ1_M
   (staged `matmul_*_8`). Add tiles for whichever the profile ranks
   highest, validated against the CPU decoders with `test-metal`, ranked
   with the tile benchmarks, and gated.
3. **Re-measure.** `scripts/spec-matrix.py --model saluki --run
   run-2026-09-06 --contexts code,512,4096,16384,32639 --drafts 3-7
   --sampling greedy,instruct --cooldown 60`. Apply the verdict rule.
   Record the result in `docs/models/saluki.md`,
   `docs/engine/speculative-decoding.md`, and
   `docs/benchmarks/history.md`.
4. **Gates.** `make verify`.

## Deferred

Agreed and not scheduled; when one is picked up, it becomes a unit above.

- **Saluki's speed goals** (restated by the user, 2026-10-15; the numbers
  and their arithmetic in `docs/models/saluki.md`, its **Goals**): decode at
  32,639 ≥ 18 tok/s, and a 32,639-token prompt prefilled in ≤ 7 minutes.
  Decode needs the 2-bit matvecs at about 200 GB/s (IQ2_XXS reads 144
  today; the limiter is instructions per weight, not bandwidth: at
  2.26 bits a weight, 200 GB/s leaves an estimated four instructions per
  weight on the 16-core GPU; `docs/engine/apple-gpu.md` § The merged
  gate+up on Saluki's encodings). The prefill goal is the entry below. Context past
  32,768 (`max_context` in `src/config.zig`) waits on the prefill goal.
  Speculation does not reach it yet. Qwen's MTP companion binds and
  drafts on Saluki, but its verdict is off (prose geometric mean ≤ 0.993×;
  `docs/models/saluki.md` § Speculation): the 8-row verify batch on the
  2-bit encodings costs 2.1× a single-row step, against the Q4_K_M's 1.45×.

- **Long-context prefill, every family** (the user's, 2026-10-15). Prefill
  at 512 is level with llama.cpp (0.89–1.02×) and falls behind as the
  prompt grows; at 32,639: Muse Glimmer 0.79×, Qwen3.8 0.71×, Saluki
  0.65×, Gemma 4 12B QAT 0.50×, E4B 0.49×, 26B-A4B 0.42×
  (`docs/benchmarks/README.md` § The benchmarks on macOS 27,
  `docs/models/saluki.md` § Speed). The suspect is the chunk attention
  (`nu_attention_chunk` / `_h`, `Backend.attentionChunk`), 30 % of a 16K
  prefill on Qwen in 2026-09; Gemma's 512-wide global layers run a
  variant that recomputes scores per 256 value columns, which may be why
  the Gemmas trail most. Two attempts closed: sharing K/V across a
  group's heads (slower) and register-level reuse (2–5 %), both at
  ~0.7 TFLOP/s and blamed on the per-tile instruction chain
  (`docs/engine/metal-backend.md` § Long-context prefill attention, and
  § second attempt). First step: a cooled `nuclis bench --profile` of
  each family's 16K prefill to confirm the attention share per family;
  then a scratch ablation as the split pass's unit did (§ The split
  pass at the memory roof found registers, guards, and a scale inside
  the products bounding decode, not bytes or ALU), and `make
  bench-attention` to rank variants. A saved prefix does not help here:
  each run pays the prefill.

- **Video input for EmbeddingGemma 2** (about one session). Frames at
  1 fps (at most 32) through the image encoder at 140 soft tokens each,
  an AVFoundation decoder beside the image and audio bridges, oracle
  fixtures from a short generated video, gates, `--video`, and a video
  part in the API. Google's processor facts: `docs/models/embeddinggemma.md`
  § The mmproj (`Video`).
