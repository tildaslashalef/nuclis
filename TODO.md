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
serve`. On this unit's branch the specialized kernels decode 17.1 tok/s at
512 (llama.cpp 12.46; the generic kernels 7.22) and prefill about 96
(llama.cpp 97).

**North star: Saluki decodes at ≥ 22 tok/s at 512 tokens** on this
machine (`nuclis bench`, Metal, F16 cache, greedy, 128 tokens): the 2-bit
matvecs at the ~180 GB/s the Q4_K kernel sustains, over 7.61 GB per token.
An estimate until measured; every unit reports against it.

**nuclis is its own reference** (since 2026-10-14; AGENTS.md, spec § 3):
new work is validated against nuclis's already-validated paths. What
replaces the oracle for a genuinely new architecture is open
(docs/models/README.md § 0).

Next: **Saluki: the 2-bit matvecs at the north star**.

| Unit | Branch | State |
| --- | --- | --- |
| Saluki: the 2-bit matvecs at the north star | `saluki-2bit-speed` | in progress |
| Q8_0 attention cache | `q8-kv-cache` | queued; facts to read first |
| 2-bit files compared on nuclis, the catalogue decision | `two-bit-comparison` | queued |

## Saluki: the 2-bit matvecs at the north star

Base: `2fe0752`. Delivered on the branch, measured in
[docs/engine/apple-gpu.md](docs/engine/apple-gpu.md) § Saluki's 2-bit
matvecs on the `ffn_down` shape and § The merged gate+up on Saluki's
encodings:

- `c35a6c1`: specialized matvecs `nu_matvec_<enc>` and prefill tiles for
  the seven (`make bench-kernels ARGS='<ENC> [shape]'`, GB/s: IQ2_XXS
  135–144, IQ1_S and IQ1_M 129–140, IQ2_XS 150–158, IQ2_S 158–169,
  IQ3_XXS 172–196, Q2_K 189–200).
- `7424ae3`: the backend declines a merged projection holding one of the
  seven (`Backend.mergeDeclined`, `error.MergeDeclined`) and the model
  adapters' standalone fallback runs it: they lost 11–23 % inside the
  merged kernel. Decode at 512: 16.09 → 17.10 tok/s.
- `86f9945`: IQ2_S's matvec at 16 SIMD groups per threadgroup (+2.5–4 %).
- Decode 17.1–17.2 tok/s at 512 (`nuclis bench --model saluki-27b
  --prompt-tokens tests/fixtures/run-2026-09-06/prompt-512.json
  --max-tokens 128 --ctx-size 32768 --kv f16`), prefill about 96.

Measured and closed (the probe tables in apple-gpu.md): the F16 dot
(fails `saluki-trace-f32`, 2.56e-2 and 2.5e-4 against 2e-3 and 1e-4),
sign tables, more rows per SIMD group, prefetched weights, inputs staged
in threadgroup memory, threadgroup size outside IQ2_S, keeping a plain
merge's other segments merged. The half-to-float conversions cost no
instruction (counted).

**22 tok/s is not reached**: it needs the 7.61 GB per token at about
200 GB/s on average, above every 2-bit kernel alone (apple-gpu.md § The
merged gate+up on Saluki's encodings, *Where the step goes now*: gate+up
about 35 % of the step, `ffn_down` 19 %). What remains:

1. **The exit record** in [docs/models/saluki.md](docs/models/saluki.md)
   § Speed: decode and prefill at 512, 4,096, 16,384, and 32,639 with
   the pinned prompts (the command above, `--prompt-tokens` per length),
   replacing the generic kernels' table, and the specialized kernels'
   rates in place of its matvec table; § Status to match. Any README or
   site figure citing Saluki follows (`make site-check`).
2. **`make verify` once** before the pull request. `verify-long` is
   flagged by `make verify-auto` only because kernels.metal holds the
   attention kernels; nothing here changes attention or the cache.
3. **Prefill to llama.cpp's 97 tok/s**, if the last 1 % matters: the
   matmul tiles' `_8`, `_w8`, `_f2` variants, as IQ3_S has.

Leads outside this unit: independent matvecs of a step (gate and up; q,
k, v) on a concurrent encoder, which the profile's overlap suggests
would hide the drain between them; the merged-versus-standalone findings
for Q4_K and IQ4_XS (Qwen3.8's file), proposed as their own unit.

Gates: `make verify-auto`, `make verify` (Metal kernels), `zig build
test-metal -Dmetal=true`. Not `verify-cpu`: no CPU arithmetic changes.

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
