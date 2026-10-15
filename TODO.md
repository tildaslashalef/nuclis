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

Theme: **Qwen3.8-27B in under 8 GB of weights, decoding faster than the
4-bit file.** The artifact is Underdog Saluki 27B 1.0, registered locally
as `saluki-27b` (facts, lineage, validation, and every number:
[docs/models/saluki.md](docs/models/saluki.md)). Since the bring-up unit
(`saluki-2bit-encodings`) it runs on the CPU and on Metal, validated
without an external oracle (the cross-encoding check, CPU traces as its
own fixture), and chats, calls tools, and reads images through `nuclis
serve`. Since the speed unit (`saluki-2bit-speed`) its specialized
kernels decode 16.96 tok/s at 512 (llama.cpp 12.46, the Q4_K_M on nuclis
11.67) and prefill 94.18 (llama.cpp 97.19). Since the Q8_0 cache unit
(`q8-kv-cache`) `--kv q8_0` holds its 32K session in 1,238 MiB instead
of 2,198 at no measured perplexity or speed cost at 512
([docs/engine/metal-backend.md § Q8_0 KV cache](docs/engine/metal-backend.md#q8_0-kv-cache-2026-10-15)). Since the
head-split unit (`split-pass-rows`) the decode attention split pass reads
the F16 cache at the memory roof (Saluki's layer at 32,640 rows: F16
1.70 → 0.63 ms, Q8_0 2.15 → 0.61) and Saluki decodes 13.20 tok/s at
32,639 instead of 11.63
([docs/engine/metal-backend.md § The split pass at the memory roof](docs/engine/metal-backend.md#the-split-pass-at-the-memory-roof-2026-10-15)).
Saluki stays out of the catalogue for now, registered in `nuclis.json`
like Bonsai (the user's decision, 2026-10-15); the comparison unit
decides.

**North star: Saluki decodes at ≥ 22 tok/s at 512 tokens** on this
machine (`nuclis bench`, Metal, F16 cache, greedy, 128 tokens). Not
reached: it needs the 7.61 GB per token at about 200 GB/s on average,
above every 2-bit kernel alone; where the step goes is in saluki.md
§ Speed. Every unit reports against it.

**nuclis is its own reference** (since 2026-10-14; AGENTS.md, spec § 3):
new work is validated against nuclis's already-validated paths. What
replaces the oracle for a genuinely new architecture is open
(docs/models/README.md § 0).

Next: **the head-split attention measured across the families**, then
**2-bit files compared on nuclis, the catalogue decision**.

| Unit | Branch | State |
| --- | --- | --- |
| The head-split attention measured across the families | `split-pass-family-rates` | in progress |
| 2-bit files compared on nuclis, the catalogue decision | `two-bit-comparison` | next |

## The head-split attention measured across the families

Base: `115cbf7` (#21 merged). No engine code changes; measurement and
documents, plus the speed loop's options below.

#21 measured only Saluki end to end, back to back, and its 32,639 gain
(11.63 → 13.20 tok/s, about 10 ms per token) is below the 17 ms the
kernel bench predicts (16 layers × 1.07 ms). This unit attributes that
gap and measures the other families.

1. **Before/after without heat bias.** `make speed` (`scripts/speed.py`)
   with the base saved at `fcebd2b` (the commit before #21: Q8_0 cache,
   old split pass) against the tree, `--contexts 16384,32639 --tokens 128
   --pairs 5`: interleaved pairs, so heat falls on both sides equally.
   `speed.py` gains `--kv f16|q8_0` (it hard-codes F16) and `--run <dir>`
   (Saluki has no acceptance workload; its arrays are
   `run-2026-09-06` plus `boundary-2026-09-06`). Families, one at a time,
   10 minutes idle between: `saluki` (F16 and Q8_0), `muse`,
   `gemma4_qat`, `gemma4_26b_a4b`. The first run per context prefills the
   saved prefix (32K: 5–10 minutes each).
2. **Attribution.** `nuclis bench --profile --prefix-cache
   .zig-cache/speed/prefix` on Saluki at 16,384, F16 and Q8_0, base and
   tree binaries, from a cooled chip (10 minutes idle before each): the
   `attention` family's ms per step against the kernel bench (16 layers ×
   0.335 / 0.332 ms after; 0.930 / 1.158 before), and whether any other
   family moved. Reports under `.zig-cache/bench/profile/`.
3. **Record.** The rates in each family's `docs/models/*.md` (Muse
   Glimmer, Gemma 4 12B QAT and 26B-A4B in `gemma4.md`, Saluki), the
   attribution in `docs/engine/metal-backend.md` § The split pass at the
   memory roof, `speed.py`'s options in `docs/development.md` § The speed
   loop. The README's results tables (release-record method, same-night
   llama.cpp, mean of three) are re-run only if the user says so.

Gates: `make workloads-validate` (speed.py self-test), `make lint-py`,
`make docs-check`, `make verify-auto`.

## 2-bit files compared on nuclis, the catalogue decision

Saluki, Unsloth UD-IQ2_XXS (7.26 GB) and UD-IQ2_S (8.36 GB) — same
encoding family — on `nuclis eval`, `nuclis bench`, and the agent task
list (`make agent-eval`). Decides which file, if any, becomes a catalogue
entry.

## Deferred

Agreed and not scheduled; when one is picked up, it becomes a unit above.

- **Video input for EmbeddingGemma 2** (about one session). Frames at
  1 fps (at most 32) through the image encoder at 140 soft tokens each,
  an AVFoundation decoder beside the image and audio bridges, oracle
  fixtures from a short generated video, gates, `--video`, and a video
  part in the API. Google's processor facts: `docs/models/embeddinggemma.md`
  § The mmproj (`Video`).
