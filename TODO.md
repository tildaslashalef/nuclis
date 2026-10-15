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
([docs/engine/metal-backend.md § Q8_0 KV cache](docs/engine/metal-backend.md#q8_0-kv-cache-2026-10-15)).
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

Next: **the split pass at the memory roof** (in progress), then **2-bit
files compared on nuclis, the catalogue decision**.

| Unit | Branch | State |
| --- | --- | --- |
| The split pass at the memory roof | `split-pass-rows` | in progress |
| 2-bit files compared on nuclis, the catalogue decision | `two-bit-comparison` | next |

## The split pass at the memory roof

Base: `fcebd2b`. The flash-decoding split pass `nu_attention_decode_t`
(`inference/src/backends/metal/kernels.metal`, dispatched by
`Backend.attentionDecode` and `attentionVerify`) reads Saluki's 32,640-row
F16 cache at 79 GB/s and Q8_0 at 33 GB/s (`metal-check --decode-bench`:
F32 1.80 ms, F16 1.70, Q8_0 2.15); about 27 of an 86 ms decode token at
32K. Goal: F16 and Q8_0 near the memory roof, Q8_0 faster than F16 at
long context.

**What limits it** (measured in a scratch harness running the same
kernels, outputs compared with the current kernel's): registers, not
ALU or bytes. One SIMD group holds eight heads' queries and accumulators
(`GROUP × CH × 2` = 128 floats per lane plus the rows), so about one
threadgroup fits per core: padding its threadgroup memory to 32 KB
changes nothing, the same kernel with GROUP 6 instead of 8 is 20 %
faster, and GROUP 2 over three head groups (reading the cache three
times) is 43 % faster. Per-channel guards (`c < width`) cost as much
again; `simd_sum`, `exp`, and explicit `fma` are noise (clang contracts
`a*b+c` even under `MTLMathModeSafe`); a lazy rescale branch and two rows
per step lose.

**Design.** A second instantiation family beside the general kernel,
`nu_attention_heads_t<KV, GROUP, CH>`, for `key_width == value_width ==
32 × CH` with CH ∈ {4, 8, 16} (widths 128, 256, 512):
- A threadgroup is (KV head, split, row) and holds `HG × RL` SIMD
  groups: SIMD group `s` owns query heads `GROUP × (s % HG)` onward
  (`HG = ceil(g / GROUP)`, g = query heads per KV head) and walks rows
  `start + s / HG`, step `RL`. Every SIMD group of the threadgroup reads
  the same rows at the same time, so DRAM sees each row once; the copies
  hit the core's cache.
- No guards: lane `l` owns channels `CH·l … CH·l + CH − 1`, loaded as
  packed `float4`/`half4`; a Q8_0 lane reads `min(CH, 8)` consecutive
  codes of one block as `ushort` pairs and one scale.
- F32/F16 keep the current per-row body (dot, `simd_sum`, online
  softmax, accumulate). Q8_0 splits the row in three phases (every head's
  dot, then the softmax, then the V accumulation) and folds the block
  scale out of both: `dot = d · Σ q·code`, `acc += (p · d) · code`.
- The `RL` row lanes merge through threadgroup memory as now (4,096
  floats, so `HG × GROUP × W ≤ 4,096` when `RL > 1`); the partial layout
  `[row][head][split][2 + W]` and `nu_attention_merge` are unchanged.
- Per (precision, width), GROUP and the preferred RL, from the sweep:
  F32/F16 128 → (4, 2), 256 → (2, 2), 512 → (1, 2); Q8_0 128 → (4, 2),
  256 → (3, 4), 512 → (1, 2). `RL` halves until `HG × RL × 32 ≤ 1,024`
  and the stage fits. Any other shape (another width, K ≠ V widths,
  `HG × 32 > 1,024`) keeps `nu_attention_decode_t` unchanged.

Scratch numbers at 32,640 rows (GPU ms, split pass plus merge):

| Shape | Precision | Now | New |
| --- | --- | ---: | ---: |
| Saluki 24 / 4 / 256 | F32 | 1.80 | 1.23 |
| | F16 | 1.70 | 0.63 |
| | Q8_0 | 2.15 | 0.61 |
| Muse Glimmer 32 / 2 / 128 | F16 | 2.46 | 0.41 |
| | Q8_0 | 2.55 | 0.49 |
| Gemma global 16 / 1 / 512 | F16 | 2.68 | 1.20 |
| | Q8_0 | 3.28 | 1.22 |
| Gemma sliding 16 / 8 / 256 | F16 | 1.31 | 1.09 |

Gates and evidence: `zig build test-metal -Dmetal=true -Doptimize=ReleaseSafe`
(decode, verify, wide, and Q8_0 checks at their bounds, plus a 128-wide
case and a fallback-width case added to the decode and Q8_0 checks);
`--decode-bench`; `nuclis bench --model saluki-27b --prompt-tokens
tests/fixtures/boundary-2026-09-06/prompt-32639.json --max-tokens 128
--ctx-size 32768 --kv f16|q8_0` and `run-2026-09-06/prompt-16384.json`; `make verify-auto`;
`make verify` (Metal kernel change); `make verify-long` (attention).
Done = the measured speedup at 16K and 32K in
docs/engine/metal-backend.md and docs/models/saluki.md.

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
