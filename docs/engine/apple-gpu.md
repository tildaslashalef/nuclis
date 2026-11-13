<!-- kind: reference -->
# The Apple M4 Pro GPU, as measured

What this engine's kernels meet on the M4 Pro's GPU (16 cores, 273 GB/s
unified memory), each fact with its measurement or its source. Captures
are taken with `make capture` or a micro-benchmark's `CAPTURE=` and read
in Xcode ([development.md § GPU counters by
capture](../development.md#gpu-counters-by-capture)); exported counter
tables stay under `.zig-cache/trace/`, the numbers that matter are copied
here.

Sources cited by number are Apple's M3-generation tech talks, which
describe the Apple family 9 shader core the M4 Pro shares: 111373 *Learn
performance best practices for Metal shaders*, 111374 *Discover new Metal
profiling tools for M3 and A17 Pro*, 111375 *Explore GPU advancements in
M3 and A17 Pro* (developer.apple.com/videos/play/tech-talks/<number>).

## The device

Queried from `MTLDevice` and `system_profiler` on 2026-09-30 (macOS 27.0),
unless a row names a measurement.

| Fact | Value | Source |
| --- | --- | --- |
| GPU | Apple M4 Pro, 16 cores, Apple family 9 (not 10), Metal 4 | `supportsFamily`, `system_profiler` |
| Unified memory | 48 GiB; `recommendedMaxWorkingSetSize` 40.2 GB; `maxBufferLength` 30.2 GB | `MTLDevice` |
| Memory bandwidth | 273 GB/s published; best kernel alone 258 GB/s (Q5_K matvec, 2026-10-03), a Qwen decode step about 63 % of peak | Apple; [record](../benchmarks/record/metal-backend.md#specialized-matvec-2026-09-07), [benchmarks](../benchmarks/history.md#the-decode-speed-baseline-2026-09-30) |
| SIMD width | 32 threads, every pipeline | `threadExecutionWidth` (`nuclis bench --kernel-stats`) |
| Threads per threadgroup | 1,024; every pipeline keeps the full 1,024 (below) | `maxThreadsPerThreadgroup`, `--kernel-stats` |
| Threadgroup memory | 32 KiB per threadgroup; occupancy falls off a cliff between 16 and 24 KB per group (the 64 × 64 prefill tile fits only with half operands); our largest static use is `nu_delta_chunk`, 28,160 B | `maxThreadgroupMemoryLength`; measured in [the tile record](../benchmarks/record/metal-backend.md#the-batched-prefill-tile-2026-09-09); `--kernel-stats` |
| Matrix unit | `simdgroup_matrix` 8 × 8; operands of one type (no half → float matrix conversion in MSL); half operands with F32 accumulation reach 4.9–5.2 TFLOP/s in 64 × 64 tiles, F32 operands 3.3–3.5; Apple publishes no peak | measured, [the tile record](../benchmarks/record/metal-backend.md#the-batched-prefill-tile-2026-09-09) |
| Per-lane arithmetic | scalar: building a `float4` value by value cost 3–4× one `uchar4 → float4` cast (Q5_K matvec 113 → 210 GB/s) | measured, [record](../benchmarks/record/metal-backend.md#specialized-matvec-2026-09-07) |
| Loads | the specialized matvecs load `uint4` (16-byte-aligned blocks), `uint2`, or `packed_ushort4` by block alignment; no controlled sweep of load width against rate has been run | [metal-kernels.md § Specialized matvec](metal-kernels.md#specialized-matvec) |
| half ↔ float | conversions are free: a half source of an F32 multiply-add adds no ALU instruction (counted, [record](../benchmarks/record/apple-gpu.md#salukis-2-bit-matvecs-on-the-ffn_down-shape-2026-10-15)); 16-bit types use fewer registers | 111373; measured |
| Clock | an isolated short dispatch measures the GPU clock's ramp-up, not the kernel: micro-benchmarks run 8 dispatches per command buffer, 64 below 8,192 rows | measured, [record](../benchmarks/record/metal-backend.md#specialized-matvec-2026-09-07) |

## Registers and occupancy under dynamic caching

On family 9 GPUs, registers, threadgroup, tile, stack, and buffer data
share on-chip caches, and "on-chip register memory is now dynamically
allocated and deallocated over the lifetime of the shader"; the maximum
register use "no longer dictates how many SIMDgroups can be run" (111375).
An **occupancy manager** watches each shader and lowers its occupancy when
its working set would spill past the L1 (111375, 111374). The FP32, FP16,
and integer pipes issue in parallel "to a greater degree than ever before"
(111375), which is why a capture reports a limiter per pipe.

**Measured consequence: the compiler reports no register pressure.** All
144 of our pipelines report `maxTotalThreadsPerThreadgroup` 1,024 and a
SIMD width of 32 (`nuclis bench --kernel-stats`, 2026-09-30), including
`nu_matvec_q4_k` at 192 registers with a spill and
`nu_attention_chunk_reuse_h` with 116 GB/s of stack traffic
([record](../benchmarks/record/apple-gpu.md#the-verify-batchs-two-largest-kernels-2026-09-30)). On
older Apple GPUs this limit drops when a kernel's registers cap its
threads; here it never does. Register pressure is read from a capture
instead: *Allocated registers* and *spilled bytes*, *L1 Register
Residency*, *Stack L1 Read/Write Bandwidth*, and occupancy against the
occupancy manager's target. What `--kernel-stats` still shows is static
threadgroup memory, which does gate occupancy ("shader cores will stall <!-- ok: quoted from Apple -->
launching new threads due to unavailability of thread group memory",
111374) and has the measured cliff above.

## Reading a capture

- Xcode's profiler reports per dispatch a **limiter** and a
  **utilization** for each unit. The limiter is the share of time the unit
  was the bottleneck; utilization is how busy it was. A high limiter with
  a low utilization means the unit's issue rate, not its throughput, is
  what stalls.
- **Occupancy** is the share of the core's SIMD-group slots in use;
  **Occupancy Manager Target** is what the hardware wanted. On family 9
  GPUs registers live in the L1 (dynamic caching), so register use shows up
  as *L1 Register Residency*, and a spill as stack traffic.
- The profiler runs at a **performance state** (Minimum / Medium /
  Maximum), chosen in the Performance view's gauge menu before
  *Profile*. The overview may still print "Medium" after a Maximum run;
  the effective GPU time is the check (it should match the benchmark's
  rate). It also reports *Sampled Cores* (9/16 and 0/16 in our runs, with
  a warning), which did not change the exported counters. Confirm a
  limiter read at Medium at Maximum before an experiment depends on it.

## What the captures found

Each capture's counters and reading are in [the record](../benchmarks/record/apple-gpu.md):

- The specialized matvecs are issue-bound on the integer and complex pipe,
  not waiting on memory: Q4_K at half the occupancy manager's target with
  192 registers ([2026-09-30](../benchmarks/record/apple-gpu.md#nu_matvec_q4_k-on-qwens-ffn_down-shape-2026-09-30)),
  IQ4_XS on its table addressing ([2026-10-03](../benchmarks/record/apple-gpu.md#nu_matvec_iq4_xs-on-qwens-merged-gate-shape-2026-10-03)),
  Saluki's 2-bit kernels on integer work, where half-to-float conversions
  cost no instruction ([2026-10-15](../benchmarks/record/apple-gpu.md#salukis-2-bit-matvecs-on-the-ffn_down-shape-2026-10-15)).
- Small-batch kernels are bound by registers or the pipe they occupy: the
  scalar multi-row matvec spilled at 8 rows ([2026-09-30](../benchmarks/record/apple-gpu.md#the-multi-row-matvec-at-2-and-8-rows-2026-09-30)),
  its word-outer rewrite compiled a run-time-indexed accumulator array to
  select chains ([2026-10-03](../benchmarks/record/apple-gpu.md#a-word-outer-multi-row-matvec-at-4-and-8-rows-2026-10-03)),
  and the register-fragment tile sits on the F32 pipe its 8 × 8 multiplies
  use ([2026-10-01](../benchmarks/record/apple-gpu.md#the-register-fragment-verify-matmul-2026-10-01)).
- A verify batch through the prefill chunk attention left the GPU nearly
  empty, 6 % occupied ([2026-09-30](../benchmarks/record/apple-gpu.md#the-verify-batchs-two-largest-kernels-2026-09-30)),
  which the few-query verify attention answered.
- The 2-bit encodings' standalone kernels beat their bodies inside the
  merged gate+up kernel, which is why the backend declines those merges
  ([2026-10-15](../benchmarks/record/apple-gpu.md#the-merged-gateup-on-salukis-encodings-2026-10-15)).
