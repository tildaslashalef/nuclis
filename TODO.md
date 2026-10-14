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
serve`, at the generic kernels' speed: decode 7.22 tok/s at 512 (llama.cpp
12.46), prefill 67 (llama.cpp 97).

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
| Saluki: the 2-bit matvecs at the north star | `saluki-2bit-speed` | next |
| Q8_0 attention cache | `q8-kv-cache` | queued; facts to read first |
| 2-bit files compared on nuclis, the catalogue decision | `two-bit-comparison` | queued |

## Saluki: the 2-bit matvecs at the north star

Base: the bring-up's merge commit on `main`. The seven encodings decode
through the generic kernels (`nu_segment` in
`backends/metal/dequant.metal`: 16 values per call through a switch), at
43–83 GB/s (`zig build bench-kernels -Dmetal=true -- <ENC>`; the table in
saluki.md § Speed); the Q4_K specialized matvec runs at 177–198 GB/s on
the same shapes, its generic path at ~103.

1. **Specialized matvecs** in `backends/metal/kernels.metal`, named
   `nu_matvec_<enc>` beside `nu_matvec_iq4_xs` and instantiated with
   `NU_MATVEC_ARGS` like it; a `Kernel` tag each in `metal/root.zig` and a
   case where encoding 21 routes to `.matvec_iq3_s`. Order by bytes per
   token: IQ2_XXS (119 tensors, every `ffn_down` but 11), IQ2_S, Q2_K,
   IQ2_XS, IQ3_XXS, IQ1_S, IQ1_M. Each must equal the CPU within
   `metal-check`'s matvec bounds (the codebook fixture already runs through
   every matvec loop) and reach ≥ 170 GB/s on the three shapes above.
2. **Codebook placement** is the first experiment: the grids sit in
   `constant` memory (IQ2_XXS 2 KiB, IQ2_XS 4 KiB, IQ2_S 8 KiB, IQ3_XXS
   1 KiB, IQ1_S 16 KiB); try threadgroup copies and narrower encodings
   (IQ2 magnitudes are 3 values, so 2 bits each; IQ1 components are
   ternary). Measure one encoding before generalizing.
3. **The multi-row matvec** (`nu_matvec_rows_*`, used by verify batches)
   and the merged segments for the seven, if the profile shows them.
4. **Exit:** `nuclis bench --model saluki-27b` decode ≥ 22 tok/s at 512
   and the 4K / 16K / 32,639 rows against saluki.md § Speed; record in
   saluki.md. If 22 is not reached, record where the time goes
   (`nuclis bench --profile`) and stop at the measured result.
5. **Then prefill**: the matmul tiles (`_8`, `_w8`, `_f2` variants, as
   IQ3_S has) to at least llama.cpp's 97 tok/s at 512.

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
