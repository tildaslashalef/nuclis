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
4-bit file.** The artifact is Underdog Saluki 27B 1.0
(`ConwayResearch/Underdog-Saluki-27B-1.0` at `4f60eba0`, pulled and
digest-verified into `~/.nuclis/models/ConwayResearch/…`, with the Q8_0
projector). It is the pinned Qwen3.8 architecture (`qwen35`, 64 blocks, no
MTP head, the already-pinned template `c3cf9e34…`) in a per-tensor mix of
seven encodings nuclis does not implement. The llama.cpp record of
2026-10-14 (`.reference/saluki/RESULTS.md`, to move into
`docs/models/saluki.md`): decode 12.46 / 12.21 / 11.47 / 9.54 tok/s at
512 / 4K / 16K / 32K (Qwen Q4_K_M 11.16 / 10.97 / 8.54 / 7.84 on the same
build), prefill 97 / 97 / 84 / 79, 10.9 GiB resident at 32K with the
projector. Its 2-bit kernels reach 95 GB/s effective against 172 GB/s for
Q4_K.

**North star: Saluki decodes at ≥ 22 tok/s at 512 tokens** on this
machine (`nuclis bench`, Metal, F16 cache, greedy, 128 tokens), which is
the 2-bit matvecs sustaining the ~180 GB/s our Q4_K path already does on
7.61 GB per token. It is an estimate until measured; every unit below
reports against it.

**Direction agreed 2026-10-14: nuclis is its own reference.** New work is
validated against nuclis's already-validated paths (the CPU reference,
the pinned Q4_K_M of the same model, Metal against CPU), not against
llama.cpp; external sources are read only as a format definition where no
other exists (the IQ codebooks). The 2026-10-14 llama.cpp run is the last
external comparison.

Next: **Saluki: the seven 2-bit encodings, correct end to end** (this
branch, `saluki-2bit-encodings`).

| Unit | Branch | State |
| --- | --- | --- |
| Saluki: the seven 2-bit encodings, correct end to end | `saluki-2bit-encodings` | in progress |
| Saluki: the 2-bit matvecs at the north star | `saluki-2bit-speed` | queued |
| Q8_0 attention cache | `q8-kv-cache` | queued; facts to read first |
| 2-bit files compared on nuclis, the catalogue decision | `two-bit-comparison` | queued |

## Saluki: the seven 2-bit encodings, correct end to end

Base: `d57d2ac`. Delivers: Saluki binds, decodes, and generates on the CPU
and on Metal (generic kernels: speed is the next unit), validated without
llama.cpp, with the rule change written down.

1. **Facts.** `docs/models/saluki.md` (listed in `docs/README.md` §
   Model families and `docs/models/README.md` where families are listed):
   artifact table (commit, files, sizes, SHA-256: main `4a673518…`,
   mmproj Q8_0 `2e0b88b6…`), metadata deltas from `qwen35-27b.json`
   (block_count 64, no `nextn`), tensor encoding table, the llama.cpp
   record from `.reference/saluki/RESULTS.md`, the memory budget (weights
   7,522 MiB, KV 64 KiB/token = 2,048 MiB at 32K, DeltaNet state 150 MiB,
   compute 345 MiB, projector 848 MiB), and the comparison with ISTA
   GSQ-RCO IQ2_XS (146 tensors re-encoded; norms and `ssm_conv1d` moved by
   BF16 steps from the originals while ISTA's are bit-identical: evidence
   of post-quantization training) and Unsloth UD-IQ2_XXS / UD-IQ2_S.
   Commit the inventory fixture `inference/src/models/fixtures/saluki-27b.json`
   (`scripts/gguf-inventory.py`). Pinned row in `docs/models/catalogue.md`
   § Pinned commits; no catalogue entry (the comparison unit decides).
2. **The rule change.** `AGENTS.md` (*Shared engineering rules*, the
   bullet on external implementations; *Validation and definition of
   done*), `docs/spec.md` (the decision row "External implementations are
   references and oracles"; § on quantization fixtures; the benchmark
   claims paragraph), `docs/models/README.md` § 4 *Pin the oracle and
   compare* (becomes: validate against nuclis's own paths), and
   `docs/development.md` where it lists the reference checkouts. Existing
   pinned fixtures and gates stay: they are committed data.
3. **Layouts.** `inference/src/tensor/encoding.zig` `layout`: 10 Q2_K
   84/256, 16 IQ2_XXS 66/256, 17 IQ2_XS 74/256, 18 IQ3_XXS 98/256,
   19 IQ1_S 50/256, 22 IQ2_S 82/256, 29 IQ1_M 56/256 (sizes from
   `ggml-common.h`'s static asserts, checked 2026-10-14).
4. **Codebooks.** Beside `quant/iq3-grid.zig`: IQ2_XXS 256×u64, IQ2_XS
   512×u64, IQ2_S 1024×u64, IQ3_XXS 256×u32, IQ1_S 2048×u64, extracted by
   `scripts/quant-fixtures.py` from `ggml/src/ggml-common.h`; the
   IQ2 sign table (`ksigns_iq2xs`, 7 bits + parity) is computed, not
   copied. `THIRD_PARTY_NOTICES.md` § GGML storage-format constants lists
   them. Metal gets them the way it gets the IQ3_S codebook.
5. **CPU decoders.** `inference/src/quant/decode.zig` `row` and
   `validateRow` for the seven; unit tests on hand-built blocks that pin
   every field (scales, grid indices, sign bits and parity, the IQ1
   delta ±0.125, IQ1_M's scale spread over `scales`, Q2_K's min).
   `models/qwen35.zig` `executableEncoding` accepts them.
6. **Cross-encoding check** (the new validation): explicit step
   `zig build test-cross-encoding -- <a.gguf> <b.gguf> [--rows n]`. For
   every tensor present in both with equal dimensions, decode sampled rows
   from each and report cosine and relative RMS per tensor and per
   encoding pair, plus a negative control (the same rows against a
   shuffled row) so the margin is visible. Run Saluki against the pinned
   Q4_K_M; thresholds come from that first run and are written in
   `saluki.md`.
7. **Metal, generic path.** `backends/metal/dequant.metal` decoders for
   the seven (16 values from `first`, as the others); the embedding gather
   for IQ1_M (`token_embd`); `metal-check` CPU-vs-GPU fixtures per
   encoding (as `checkDenseEncodings`).
8. **End to end.** `nuclis generate --model <saluki>` on CPU and Metal;
   a CPU trace fixture `tests/fixtures/saluki-hello-comma` recorded by
   nuclis's CPU reference; gates `saluki-trace-f16` / `saluki-trace-f32`
   (Metal against it) and a `saluki` model entry in `gates.json`.
   `nuclis eval --file wiki.test.raw --chunks 8` for Saluki and the
   Q4_K_M, recorded in `saluki.md`. A chat, a tool call, and an image
   (`--image docs/branding/nuclis.png`) through the Q8_0 projector.
9. **Baseline speed** on the generic kernels: `nuclis bench` at 512 /
   4K / 16K / 32,639 and `zig build bench-kernels` per encoding, recorded
   as the starting point for the next unit.

Gates: `zig build test`, `zig fmt`, `make lint-py`, `make docs-check`,
`make verify-auto`, `make verify` once (the unit touches the inference
stack). `make verify-cpu` is not required: no existing CPU kernel or
forward changes; the new decoders are pinned by unit tests and the
cross-encoding check.

## Saluki: the 2-bit matvecs at the north star

Specialized matvecs in `backends/metal/kernels.metal` (`nu_matvec_<enc>`
beside `nu_matvec_iq4_xs`, routed in `metal/root.zig` where encoding 21
picks `matvec_iq3_s`), in order of bytes per token: IQ2_XXS, IQ2_S, Q2_K,
IQ2_XS, IQ3_XXS, IQ1_S, IQ1_M. Per kernel ≥ 170 GB/s on the model shapes
(`bench-kernels`); codebook placement (constant vs threadgroup memory) is
the first experiment. Exit: `nuclis bench` decode ≥ 22 tok/s at 512, and
the 4K / 16K / 32K rows against the llama.cpp record. Then the prefill
tiles (`_8`, `_w8`, `_f2` variants) to at least the record's 97 tok/s.

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
