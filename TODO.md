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
11.67) and prefill 94.18 (llama.cpp 97.19).

**North star: Saluki decodes at ≥ 22 tok/s at 512 tokens** on this
machine (`nuclis bench`, Metal, F16 cache, greedy, 128 tokens). Not
reached: it needs the 7.61 GB per token at about 200 GB/s on average,
above every 2-bit kernel alone; where the step goes is in saluki.md
§ Speed. Every unit reports against it.

**nuclis is its own reference** (since 2026-10-14; AGENTS.md, spec § 3):
new work is validated against nuclis's already-validated paths. What
replaces the oracle for a genuinely new architecture is open
(docs/models/README.md § 0).

Next: **Q8_0 attention cache**.

| Unit | Branch | State |
| --- | --- | --- |
| Q8_0 attention cache | `q8-kv-cache` | next; facts to read first |
| 2-bit files compared on nuclis, the catalogue decision | `two-bit-comparison` | queued |

## Q8_0 attention cache

`runtime/session.zig` `Precision` gains a quantized attention cache
(Q8_0: 34 bytes per 32 values, half of F16) for every family; at 32K on
Saluki 2,048 → 1,088 MiB. Read the attention kernels' cache access first
and rewrite this section at that level. Measured by `nuclis eval` (F16 vs
Q8_0) and `make verify-long`.

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
