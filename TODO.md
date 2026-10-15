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
| Q8_0 attention cache | `q8-kv-cache` | in progress; design written |
| 2-bit files compared on nuclis, the catalogue decision | `two-bit-comparison` | queued |

## Q8_0 attention cache

Base: `d70751d`. Branch `q8-kv-cache`.

`session.Precision` gains `q8_0`: GGML's Q8_0 block, 32 values as one
F16 scale `d` then 32 `i8` codes (34 bytes, value = `d * q`), 53 % of
F16. At 32K on Saluki (16 attention layers × 2 × 1,024 channels) the
attention rows go 2,048 → 1,088 MiB. GPU only, like `f16`: the CPU
reference keeps F32 and `Engine.kv_precision` reports it. The default
stays `f16`; whether `q8_0` becomes the default is decided by the
measurements below and recorded in the pull request.

Facts read (2026-10-15): every family writes its cache from F32 rows
(`k`/`v` decode scratch, `k_c`/`v_c` chunk buffers) by `copy` or
`packHalf`; decode and few-row verify read the cache through
`nu_attention_decode_t` (lane `l` owns channels `l, l+32, …`, so a
32-value block maps one byte per lane, the scale broadcast); prefill
chunks read it through `nu_attention_chunk_t` / `_reuse_t`, which
`simdgroup_load` 8×8 tiles straight from device memory and cannot read
blocks. Every cache row width is a multiple of 32 (qwen35 1,024; Gemma 4
and Muse Glimmer `kv_heads × head_size`, heads of 128/256/512).

1. **Session** (`inference/src/runtime/session.zig`): `Precision.size()`
   becomes `bytes(elements)` (Q8_0: `elements / 32 * 34`); `sizes` and
   `Rows.rowBytes` use it; `Session.init` refuses a `q8_0` row not a
   multiple of 32 (`InvalidShape`). The digest already hashes the tag.
   Tests: layout sizes, snapshot round trip with a `q8_0` layer.
2. **Reference quantizer** (`inference/src/quant/encode.zig`, new):
   `q8_0Row(input, output)`: per block `amax = max |x|`,
   `d = f16(amax / 127)`, `q = clamp(round(x / f32(d)), -127, 127)`
   (round half away from zero), `d = 0` gives zero codes. Test: the
   round trip through `decode.row` within `d / 2` per value, and the
   extremes.
3. **Metal kernels** (`kernels.metal`, `metal/root.zig`):
   - `nu_pack_q8_0`, `Backend.packQ8(pairs)`: the `packHalf` pair
     contract, one SIMD group per block, the quantizer of step 2
     (bit-identical to it under `MTLMathModeSafe`; `test-metal` checks).
   - `nu_unpack_q8_0`, `Backend.unpackQ8(pairs)`: Q8_0 → F16 for the
     chunk path. Not exact: `d * q` needs up to 18 significant bits and
     F16 holds 11, a 2^-12 relative rounding under Q8_0's own step of
     up to `d / 2`; the prefill chunk sees it, decode does not.
   - `Backend.storeRows(pairs, precision)`: `copy` / `packHalf` /
     `packQ8`, so each family's cache write is one call.
   - `nu_attention_decode_t` reads through a per-type element loader;
     new instantiations `nu_attention_decode_q8` / `_wq8`.
     `attentionDecode` and `attentionVerify` accept `q8_0` when both
     widths are multiples of 32; byte checks use `Precision.bytes`.
     `attentionChunk`/`attentionChunkReuse`/`attention` refuse `q8_0`
     (`InvalidShape`).
4. **Families** (`qwen35_metal.zig`, `gemma4_metal.zig`,
   `muse_glimmer_metal.zig`, their drafter paths): cache writes through
   `storeRows`; decode and verify pass `q8_0` through; a prefill chunk on
   a `q8_0` cache unpacks the visible rows `[first, position + count)` of
   K and V into a plan-owned F16 stage (`kv_stage`, created only for
   `q8_0`: capacity × the widest row × 2 bytes × 2) and runs the F16
   chunk kernel on it with the half queries, as the F16 path does. The
   stage is one layer's F16 rows, against the whole cache's saving.
5. **Surface**: `--kv f32|f16|q8_0` (the tag names drive the CLI,
   completion and config), help, `docs/guide/configuration.md`,
   `docs/spec.md` (the cache decision row and the command synopses),
   `docs/engine/session.md`, `docs/engine/metal-backend.md` (a *Q8_0 KV
   cache* section beside *F16 KV cache*).

Gates and evidence:
- `zig build test`; `make test-metal` (pack bit-identical to
  `q8_0Row`; unpack exact; Q8_0 decode attention against the F32 kernel
  over the dequantized rows, bit-identical by construction, and against
  the CPU over the original floats; verify rows; shape refusals).
- `test-generation --metal` (`inference/generation-check.zig`): a Q8_0
  stepped and chunked run against the F32 stepped logits, beside the
  F16 cases; argmax identical.
- A `qwen38-trace-q8_0` gate beside `qwen38-trace-f16` in `gates.json`,
  bounds set from the measurement.
- `nuclis eval` on Saluki, F16 vs Q8_0 (same file, context, chunks),
  and on the Q4_K_M; `make verify-long`; `make verify` (fast Metal tier).
- `nuclis bench` Saluki at 512 (the north-star row: no regression) and
  at a long context (session MiB and decode tok/s, F16 vs Q8_0).

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
