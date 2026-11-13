<!-- kind: reference -->
# Metal backend

The Metal backend runs a model's text schedule on the GPU: one command
buffer per token, every activation resident on the GPU, and the CPU supplying a
token ID and reading back either logits or a greedy token. It is numerically
validated against the CPU reference for every family in the catalogue
([models § 4](../models/README.md#4-validate-against-nuclis-itself)); Qwen3.8 was the first and is the one the kernels are tuned
for. For the surrounding stack see
[architecture.md](../architecture.md), and for where decode and prefill time
goes [§ 11 Where performance goes](../architecture.md#8-where-performance-goes);
the current rates are the README's [Results](../../README.md#results). What
the GPU itself is (limits, dynamic caching) is in
[apple-gpu.md](apple-gpu.md).

This document is the bridge, ownership, and the schedule. The rest of the
reference:

- [metal-kernels.md](metal-kernels.md): the kernels, their geometry and contracts;
- [metal-attention.md](metal-attention.md): the KV cache layouts and the attention kernels;
- [metal-evidence.md](metal-evidence.md): the numerical checks;
- [the record](../benchmarks/record/metal-backend.md): how each part was built
  and measured, the investigations that closed negative included.

## Build and run

Metal is opt-in at build time so default builds and tests never touch the GPU:

```sh
zig build -Dmetal=true -Doptimize=ReleaseSafe
./zig-out/bin/nuclis generate --backend metal --prompt 'Hi' --max-tokens 8 --ctx-size 64
./zig-out/bin/nuclis bench --backend metal --prompt-file prompt.txt --max-tokens 64
zig build test-metal -Dmetal=true -Doptimize=ReleaseSafe
zig build test-generation -Dmetal=true -Doptimize=ReleaseSafe -- MODEL --metal
```

Without `-Dmetal=true`, `inference.metal.Backend.init` returns
`error.MetalNotEnabled` and the CLI rejects `--backend metal`. `test-metal` runs
`inference/metal-check.zig`, an explicit GPU fixture and lifecycle check
needing a Metal device but no model. `test-generation --metal` runs the session
isolation/reset protocol on the GPU plan. Full-model layer comparison uses the
same `--logits`/`--trace-dir` oracle as the CPU backend ([sampling.md](sampling.md)).

## Files and ownership

| File | Owns |
| --- | --- |
| `backends/metal/bridge.m` | Device, queue, shader library, pipelines, buffers, one open command buffer. A generic recording API: `begin`, `dispatch(pipeline, bindings, constants, grid)`, `commit`, and `discard` (an error path drops the open command buffer unsubmitted). Opt-in profiling: timestamp counter sample buffers and one encoder per dispatch, resolved to seconds per dispatch at commit. Knows nothing about models, shapes, or encodings. |
| `backends/metal/root.zig` | `Backend`: compiles the kernel set, hands out `Buffer` handles (`create`/`wrap`/`slice`), and exposes one typed encoder per kernel that validates shapes before recording. `Profile` accumulates timed dispatches by kernel, encoding, and shape. Knows kernel contracts, not layer schedules. |
| `backends/metal/kernels.zig`, `profile.zig` | The MSL source's assembly and the `Kernel` table (`kernel_files`, the codebooks as MSL constants); per-dispatch timing types. |
| `backends/metal/ops/*.zig` | The typed encoders, one file per kind of kernel (`matvec`, `matmul`, `experts`, `rowwise`, `encoders`, `recurrent`, `attention`, `sampling`, `audio`, `asr`, `tts`): functions of `*Backend` that `root.zig` aliases as methods, so callers write `backend.matvec(…)`. |
| `backends/metal/dequant.metal` | GGUF block decoders, ported line by line from `quant/decode.zig`. Bit-exact with the CPU decoders. |
| `backends/metal/kernels/*.metal` | Compute kernels, one file per kind of operation (`common`, `matvec`, `matvec_codebook`, `matvec_split`, `matvec_rows`, `experts`, `matmul`, `rowwise`, `recurrent`, `attention`, `recurrent_chunk`, `sampling`, `encoders`, `audio`, `asr`, `tts`), joined in that order (`kernel_files` in `kernels.zig`; later files use earlier helpers). `matvec_split_k` (the split-K matvecs and the fragment layout pin) joins only the `bench_kernels` build `metal-check` imports. What each computes: [metal-kernels.md](metal-kernels.md). |
| `models/qwen35_metal.zig` | `Plan`: the Qwen schedule expressed as encoder calls. Owns the session and activation buffers; borrows weights and the backend. |
| `models/gemma4_metal.zig` | `Plan`: the Gemma 4 schedules on the same encoders; sliding-window slices, two RoPE tables, the wide global-layer attention ([gemma4.md § Metal plan](../benchmarks/record/gemma4.md#metal-plan-2026-09-11)). |
| `models/muse_glimmer_metal.zig` | `Plan`: the Muse Glimmer 30B schedule on the same encoders; adjacent-pair RoPE on sliding layers only, the sigmoid attention gate, the untied scaled head ([muse-glimmer.md § Metal plan](../benchmarks/record/muse-glimmer.md#metal-plan-2026-09-19)). |
| `models/qwen3_tts_metal.zig` | `Plan`: Qwen3-TTS's talker and code predictor over the BF16 matrices wrapped in place; a frame's 15 groups sampled on the device and checked by the host sampler ([qwen3-tts.md § On Metal](../benchmarks/record/qwen3-tts.md#on-metal-2026-10-27)). |
| `models/drama3_metal.zig`, `laya_metal.zig`, `embeddinggemma_metal.zig` | The other families' plans on the same encoders (their cards in [models](../models/README.md)). |
| `audio/*_metal.zig`, `vision/*_metal.zig` | The codec decoders (Qwen3-TTS's `Decoder` and `Stream`, streamed with their carried state on the device, bit for bit the same for any piece size), the ASR and audio encoders, and the vision encoders ([vision.md](vision.md)). |
| `metal-check.zig` | Explicit GPU checks against pinned fixtures and CPU references (the TTS kernels' in `tts-metal-check.zig`), and the kernel micro-benchmarks behind `make bench-kernels` and its siblings. |

The MSL source is assembled at compile time: the IQ3_S codebook is emitted
from the single Zig definition, then `dequant.metal` and the `kernels/`
files, in `kernel_files` order, are embedded. The shader compiles at backend
creation with `MTLMathModeSafe`, so no reassociation or contraction is
applied, and every `Kernel` tag's pipeline is compiled there. Metal's
per-user shader cache keeps that cheap: a warm start spends about 50 ms,
a library recompile 1.4–1.6 s
([record](../benchmarks/record/metal-backend.md#shader-compilation-at-startup-2026-11-12)).

## Execution model

Qwen3.8's decode step:

```text
Plan.step(token)
  session.begin()                       CPU: admission
  backend.begin()                       open command buffer + compute encoder
  embed(token) -> x
  for each of 64 layers:
    rmsnorm(x) -> normalized
    mixer: attention (merged qg+k+v, norm, rope, cache write, split pass + merge, gate, output)
        or DeltaNet   (merged qkv+z+beta+alpha, conv, silu, l2norm, gates, delta, norm*silu(z), output)
    x += projected
    rmsnorm(x) -> normalized; merged gate/up with silu*mul; down matvec; x += projected
  final rmsnorm, output matvec -> logits [, argmax -> result]
  backend.commit()                      submit, wait for completion, account GPU time
  read logits or greedy id from shared memory
  session.commit()                      CPU: position += 1
```

The layer dispatches are recorded into one serial compute pass; Metal
guarantees each dispatch sees the previous one's writes. `commit()`
**waits**, so after it the CPU may read any buffer and may `reset()` the
session. That synchronous contract is what makes the following simple:

- **Weights** are the read-only file mapping wrapped per tensor with
  `newBufferWithBytesNoCopy` (cached by address). The mapping outlives the
  backend. Per-tensor buffers keep GPU time steady where one buffer over
  the whole mapping did not
  ([record](../benchmarks/record/metal-backend.md#execution-model-2026-09-07)).
- **Session state** (`runtime/session.zig`) is one page-aligned byte block
  wrapped once. KV rows (in the layout's precision:
  [metal-attention.md § The KV cache](metal-attention.md#the-kv-cache)),
  convolution history, and DeltaNet matrices are typed views the plan binds
  by byte range. `reset()` is a CPU memset, valid only when no GPU work is
  in flight — true after every `commit()`. An asynchronous backend must
  fence before `reset`, `fail`, or observer callbacks; that invariant is
  documented at the top of `session.zig`.
- **Activations** are backend-owned shared buffers sized from the plan at
  `Plan.init`. The bridge has no model constants.
- **Failure**: a failed command buffer or a rejected dispatch returns an error;
  `errdefer` marks the session failed and it stays poisoned until `reset()`.
  The DeltaNet update is in place, so a partially executed token is not
  transactional — matching the CPU contract's outcome. An error while
  recording (a rejected shape, an observer's `check` cancelling) discards
  the open command buffer (`Backend.discard`, `errdefer` after `begin`):
  the recorded layers never run, and the backend can `begin` again.
  Destroying a backend mid-recording discards too; waiting on a buffer
  that was never committed would never return.
- **Observer.** `check` runs between recorded layers with no synchronization
  (Ctrl-C, time limit). `layer` needs the activations, so it forces a `commit`
  after every layer: traces only. The two are separate callbacks, so a
  `generate` or `bench` token stays one command buffer.
- **Profiling** (`enableProfiling`, `bench --profile`): Apple GPUs sample
  timestamps at encoder boundaries only, so each dispatch is recorded in its
  own encoder and timed by its start/end stamps (nanoseconds, same timeline as
  `GPUStartTime`). The per-dispatch encoders cost ~8 ms per token, reported
  as the gap between attributed and command-buffer time, so profiled rates
  are diagnostic. Methodology: [benchmarks](../benchmarks/README.md#per-kernel-profile).

## Prefill in chunks

`Plan.prefill(tokens, …)` consumes a prompt in chunks of up to `chunk`
tokens (`engine.prefill_chunk`, 256, or the plan's `preferredChunk`,
clamped to the session capacity), one command buffer per chunk. Inside a
chunk the projections and the feed-forward network run through
`Backend.matmul` on `[token][feature]` activation buffers sized for the
chunk (padded to a multiple of 64 rows; `step` keeps its own single-token
buffers, so the two paths never alias); the norms, RoPE (`nu_rope_rows`),
SiLU, gates, and residual adds run over `count` rows at once; the DeltaNet
convolution runs as one causal pass plus a history update; attention runs
as one causal tiled dispatch per layer after the chunk's keys and values
are projected, normalized, rotated, and stored into the cache in one go;
and DeltaNet runs as one chunkwise dispatch per layer over 32-token
sub-chunks. Nothing steps per token inside a chunk.

Contract: the whole prompt must fit the remaining context (refused before
any work), the observer's `check` runs between layers of every chunk, and a
per-layer `layer` observer is refused because it is a per-token contract —
`Model.prefill` steps token by token in that case, which is how
`--trace-dir` and the `qwen38-trace-*` gates keep their exact per-token
semantics. The tile products accumulate in a different order than the
matvec and round their operands to half, so a chunked prompt and the same
prompt stepped agree within a tolerance rather than bit for bit;
`generation-check --metal` bounds the difference per family
([metal-evidence.md § Model-level checks](metal-evidence.md#model-level-checks)).
How the chunked path was built and what each step bought:
[record](../benchmarks/record/metal-backend.md#prefill-in-chunks-2026-09-08).

## Small batches

A speculative verify batch (a seed and its drafts), a drafter's batched
commit, and a recovery replay are 2–16 rows, which the prefill path's
choices fit badly. Each part routes by row count instead:

- **Products.** `Backend.matmul` sends 2–3 rows to the multi-row matvec and
  up to `small_chunk_tokens` (24) to the register-fragment tile
  ([metal-kernels.md § Batched products](metal-kernels.md#batched-products)).
- **Attention.** Chunks of at most `verify_max_rows` (16) rows run the
  decode split pass with a row dimension
  ([metal-attention.md § Few-query verify attention](metal-attention.md#few-query-verify-attention)).
- **DeltaNet.** A verify batch runs per-token steps with a replay tape
  rather than the 32-token WY form
  ([speculative-decoding.md](speculative-decoding.md)).

The `qwen38-verify-depth-*` gates (`generation-check --verify-depth`)
restore a saved prefix of the acceptance arrays at 511, 4,095, 16,383, and
32,638 tokens and compare a 4-row verify batch's logits with stepped decode
from the same state, bounding the worst row at 4e-2 max abs and 2e-3
relative RMS, with the same greedy token on every row. A causal limit off by
the batch reads 4.7e-2 relative RMS with the greedy token unchanged, which
is why the gate bounds the RMS and not only the argmax. Where a verify
batch's cost went before these routes:
[record](../benchmarks/record/metal-backend.md#where-a-verify-batchs-cost-goes-2026-09-30).

## Limits

- The specialized matvec kernels reach 150–258 GB/s of a published 273 and
  are bound by per-block instruction issue, not bytes
  ([apple-gpu.md](apple-gpu.md)). Q8_0 and IQ4_NL (under 1 % of the Qwen
  file's bytes) use the generic kernel.
- Profiling times dispatches in separate encoders, which perturbs what it
  measures by ~8 %; there is no per-dispatch sampling on Apple GPUs, and the
  Instruments GPU limiter counters are unavailable on this device from
  `xctrace`. Counters come from Metal captures replayed in Xcode,
  and only for command buffers under its full-profiling limit (kernel
  captures, not a whole verify): [apple-gpu.md](apple-gpu.md).
- Prefill's ceiling is the matmul tile's half operands, whose rounding shows
  more on Gemma's larger activations (2.5e-3 relative RMS on the chunked
  logits against 1.1e-4 for Qwen). Attention widths are bounded at 512
  (decode: 16 channels per lane in the wide pair; prefill: two value
  splits); a wider head needs a third instantiation.
- Wrapped buffers: `Backend.wrap` caches by address and `unwrap` forgets a
  range (plans call it on `deinit` so a later session at the same address
  and another length can be wrapped), but the bridge keeps every
  `MTLBuffer` object until the backend is destroyed; a long-lived backend
  that opens and closes many plans accumulates no-copy buffer objects.
  Created buffers can be freed with `release` (the Gemma plan
  frees its chunk buffers when an image span grows them); the id is never
  reused, and a dispatch that binds a released id fails.
- The vision encoders' attention runs on the segments kernels; 72-wide
  heads waste a quarter of the values kernel's last column block
  ([../benchmarks/record/vision.md § Encoders at the materialized-score attention](../benchmarks/record/vision.md#encoders-at-the-materialized-score-attention-2026-10-27)).
- The shader is compiled from source at startup; there is no binary archive.
- GPU resource cleanup is exercised by recreating the backend across fixture
  files in `metal-check`; Zig's testing allocator cannot observe Metal objects.
