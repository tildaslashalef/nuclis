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
| Memory bandwidth | 273 GB/s published; best kernel alone 248 GB/s (Q6_K matvec), a Qwen decode step about 63 % of peak | Apple; [metal-backend.md § Specialized matvec](metal-backend.md#specialized-matvec), [bench.md](bench.md#the-decode-speed-baseline-engn-18-2026-09-30) |
| SIMD width | 32 threads, every pipeline | `threadExecutionWidth` (`nuclis bench --kernel-stats`) |
| Threads per threadgroup | 1,024; every pipeline keeps the full 1,024 (below) | `maxThreadsPerThreadgroup`, `--kernel-stats` |
| Threadgroup memory | 32 KiB per threadgroup; occupancy falls off a cliff between 16 and 24 KB per group (the 64 × 64 prefill tile fits only with half operands); our largest static use is `nu_delta_chunk`, 28,160 B | `maxThreadgroupMemoryLength`; measured in [metal-backend.md § Kernels](metal-backend.md#kernels) (ENGN-05); `--kernel-stats` |
| Matrix unit | `simdgroup_matrix` 8 × 8; operands of one type (no half → float matrix conversion in MSL); half operands with F32 accumulation reach 4.9–5.2 TFLOP/s in 64 × 64 tiles, F32 operands 3.3–3.5; Apple publishes no peak | measured, [metal-backend.md § Kernels](metal-backend.md#kernels) |
| Per-lane arithmetic | scalar: building a `float4` value by value cost 3–4× one `uchar4 → float4` cast (Q5_K matvec 113 → 210 GB/s) | measured, [metal-backend.md § Specialized matvec](metal-backend.md#specialized-matvec) |
| Loads | the specialized matvecs load `uint4` (16-byte-aligned blocks), `uint2`, or `packed_ushort4` by block alignment; no controlled sweep of load width against rate yet | [metal-backend.md § Specialized matvec](metal-backend.md#specialized-matvec) |
| half ↔ float | conversions are free; 16-bit types use fewer registers | 111373 (not isolated by a measurement of ours) |
| Clock | an isolated short dispatch measures the GPU clock's ramp-up, not the kernel: micro-benchmarks run 8 dispatches per command buffer, 64 below 8,192 rows | measured, [metal-backend.md § Specialized matvec](metal-backend.md#specialized-matvec) |

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
`nu_attention_chunk_reuse_h` with 116 GB/s of stack traffic (below). On
older Apple GPUs this limit drops when a kernel's registers cap its
threads; here it never does. Register pressure is read from a capture
instead: *Allocated registers* and *spilled bytes*, *L1 Register
Residency*, *Stack L1 Read/Write Bandwidth*, and occupancy against the
occupancy manager's target. What `--kernel-stats` still shows is static
threadgroup memory, which does gate occupancy ("shader cores will stall
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

## `nu_matvec_q4_k` on Qwen's `ffn_down` shape (2026-09-30)

`make bench-kernels ARGS=Q4_K CAPTURE='matvec-Q4_K-5120x17408
(ffn_down)-block'`: 64 dispatches of the 5,120 × 17,408 Q4_K matvec
(50.1 MB) in one command buffer (exports
`matvec-Q4_K-5120x17408-ffn_down-block_2026-09-30T0941_medium.csv` and
`…_2026-09-30T0947_max.csv`, not committed), profiled in Xcode 27 on macOS 27.0 at
performance states Medium and Maximum. The benchmark reads 127–150 GB/s
on this shape; the replays took 31.09 ms at Medium (about 103 GB/s) and
21.51 ms at Maximum (**149 GB/s**, the benchmark's best).

| Counter | Medium | Maximum |
| --- | ---: | ---: |
| Instruction Throughput Limiter / Utilization | 75.4 / 9.9 % | 72.1 / 9.8 % |
| Integer and Complex Limiter / Utilization | 71.2 / 38.6 % | 68.9 / 38.0 % |
| F32 Limiter / Utilization | 16.0 / 13.1 % | 16.0 / 12.9 % |
| Integer and Conditional Limiter | 6.3 % | 6.2 % |
| ALU instruction mix: integer and complex / float / integer and conditional | 50.8 / 34.6 / 14.6 % | same |
| Kernel Occupancy / Occupancy Manager Target | 23.1 / 47.4 % | 23.5 / 45.2 % |
| Allocated registers / high register / spilled | 192 / 192 / 16 bytes | same |
| L1 Cache Limiter / Eviction Rate | 17.4 / 13.7 % | 18.8 / 17.4 % |
| Last Level Cache Limiter / Miss Rate | 0.2 / 95.8 % | 1.5 / 94.4 % |
| MMU Limiter / TLB Miss Rate | 0.3 / 74.6 % | 2.0 / 34.4 % |

**Reading, confirmed at full clocks.** The kernel is not waiting
on memory: the last-level cache, the L1, and the MMU are barely limiters,
and the cache misses are the expected streaming. It is **issue-bound on
the integer and complex pipe**: half its ALU instructions are integer and
complex (the Q4_K nibble unpacking, scale extraction, and integer-to-float
conversions), and that pipe is the limiter 69–71 % of the time while only 38–39 %
utilized. Occupancy is half what the occupancy manager targets, with 192
registers per thread and a small spill, so fewer SIMD groups are in
flight to hide load latency. KERN-05 cut instruction count without a gain;
what this adds is *which* instructions: the integer and complex ones.
That ranks the single-row matvec's ideas: a decode that produces floats
without integer-to-float conversions (the half magic-number form), fewer
unpacking operations per weight, and fewer live registers, each read
against these counters.

## The verify batch's two largest kernels (2026-09-30)

A whole 4-row Qwen verify at 4K replays only in Xcode's lite mode (over
the full-profiling run-time limit: 332.08 ms effective GPU time at
Maximum, the unprofiled verify's 335 ms), so its two largest kernels were
captured from the micro-benchmarks on the verify's shapes and profiled at
Maximum (exports `attention-4096-c8-reuse-f16_2026-09-30T1513_max.csv` and
`rows-IQ4_XS-17408x5120-t4-tile_2026-09-30T1513_max.csv`, not committed).
Both replays ran at the benchmarks' rates: 16 attention dispatches in
107.26 ms (6.70 ms each; the sweep reads 6.46–6.56) and 16 matmul
dispatches in 8.39 ms (0.52 ms each, the benchmark's). Sampled cores 3/16
and 8/16, with the usual warning.

| Counter | `nu_attention_chunk_reuse_h` (8 rows, 4,096 visible, F16) | `nu_matmul_iq4_xs_8` (17,408 × 5,120, 4 tokens) |
| --- | ---: | ---: |
| ALU Utilization | 4.9 % | 47.9 % |
| Instruction Throughput Limiter / Utilization | 8.6 / 2.9 % | **91.0** / 27.3 % |
| F32 Limiter / Utilization | 4.9 / 4.8 % | 68.0 / 47.8 % |
| Integer and Complex Limiter / Utilization | 6.4 / 6.3 % | 56.4 / 43.1 % |
| Integer and Conditional Limiter | 1.8 % | 35.6 % |
| ALU instruction mix: float / integer and complex / integer and conditional | 49.9 / 32.2 / 18.0 % | 50.0 / 22.5 / 27.5 % |
| Kernel Occupancy / Occupancy Manager Target | **6.2 / 85.1 %** | 37.7 / 82.1 % |
| L1 Register / Threadgroup Residency | 31.4 / 4.8 % | 1.5 / 23.3 % |
| Stack L1 Read / Write Bandwidth | **116.1 / 116.2** | 0 / 0 |
| Threadgroup Memory L1 Read Bandwidth | 4.8 | 321.1 |
| Buffer L1 Read Bandwidth / Miss Rate | 16.3 / 27.9 % | 688.6 / 19.7 % |
| L1 Cache Limiter / Eviction Rate | 1.6 / 100 % | 22.9 / 0 % |
| Last Level Cache Limiter / Miss Rate | 10.3 / 47.0 % | 9.1 / 39.2 % |
| MMU Limiter | 0.1 % | 3.1 % |

(Bandwidths as Xcode exports them, GB/s.)

**The verify attention waits on nothing: the GPU is nearly empty.**
Every limiter is at or under 10 %, ALU utilization is 5 %, and occupancy is
6 % against an 85 % target: 109,056 kernel invocations over 16
dispatches, about 6,800 threads each (the dispatch is 24 threadgroups per
value split, one per query head: ADR 0001), cannot hide the latency of
walking 4,096 cached rows. What traffic there is, is mostly the thread's own
stack: 116 GB/s of spill reads and writes against 16 GB/s of buffer
reads, with the L1 evicting every line. Both point at the design KERN-21
proposes: split the keys so hundreds of threadgroups share the walk, and
keep per-thread state small enough not to spill. It also explains the
sweep's finding that 1 and 8 rows cost the same: rows are not what the
kernel is short of; parallel work is.

**The verify matmul is issue-bound, like the matvec, on different
pipes.** The instruction-throughput limiter is 91 % while the ALUs are
48 % busy: F32 (68 %) and integer and complex (56 %) both near their
limits, occupancy at 38 % of an 82 % target, no spills. Memory is not the
limit (last-level cache 9 %, MMU 3 %). The tile stages every decoded
weight through threadgroup memory (321 GB/s of threadgroup reads for
689 GB/s of buffer reads), and its 8-wide fragment computes 8 token
columns for 4 real ones, so half its float work multiplies padding. That
ranks KERN-24's ideas: decode straight into register fragments (drop the
staging instructions), and route small row counts to a body that does not
compute padded columns: the multi-row matvec already reads 103 GB/s at 4
tokens on this shape against the tile's 90.5.

## The multi-row matvec at 2 and 8 rows (2026-09-30)

`make bench-matvec-rows CAPTURE='rows-Q4_K-17408x5120-t<n>-rows'`: the
Q4_K multi-row matvec (`nu_matvec_rows_q4_k_t2` and `_t8`) on the
17,408 × 5,120 `ffn_gate`/`ffn_up` shape, which reads 141.7 GB/s at 2
rows (0.35 ms) and 27.8 GB/s at 8 (1.80 ms; the 16 × 8 tile reads 96.5 at
any count). Profiled at Maximum (exports
`rows-Q4_K-17408x5120-t2-rows_2026-09-30T1800_max.csv` and
`…-t8-rows_…`, not committed); the replays' GPU times were not read.

| Counter | 2 rows | 8 rows |
| --- | ---: | ---: |
| ALU Utilization | 23.8 % | 12.0 % |
| Instruction Throughput Limiter / Utilization | 65.2 / 12.6 % | 23.1 / 6.3 % |
| Integer and Complex Limiter / Utilization | 60.3 / 36.5 % | 9.1 / 7.9 % |
| F32 Limiter / Utilization | 34.6 / 23.7 % | 22.4 / 18.0 % |
| ALU instruction mix: float / integer and complex / integer and conditional | 49.7 / 38.4 / 11.9 % | 75.1 / 16.6 / 8.3 % |
| Kernel ALU instructions (same invocations) | 8.70 × 10⁹ | 22.30 × 10⁹ |
| Kernel Occupancy / Occupancy Manager Target | 25.2 / 30.8 % | **17.6 / 19.7 %** |
| L1 Register / Buffer Residency | 51.3 / 26.8 % | **72.4** / 12.6 % |
| Stack L1 Read / Write Bandwidth | 146.8 / 129.2 | **309.9 / 297.2** |
| Buffer L1 Read Bandwidth / Miss Rate | 601.9 / 16.8 % | 395.0 / 19.8 % |
| L1 Cache Limiter / Eviction Rate | 25.4 / 76.3 % | 16.1 / 100 % |
| Last Level Cache Limiter / MMU Limiter | 4.0 / 1.6 % | 0.2 / 0.1 % |
| Compute Shader Launch Limiter | 66.3 % | 64.7 % |

(Bandwidths as Xcode exports them, GB/s.)

**Reading: register pressure, confirmed.** At 2 rows the body looks like
the single-row matvec: issue-bound on the integer and complex pipe (limiter
60 %), with some spill already (276 GB/s of stack traffic). At 8 rows the
live accumulators (one per row and token) fill the L1 with registers
(72 % register residency), the kernel spills 607 GB/s of stack traffic
against 395 GB/s of buffer reads, the occupancy manager lowers its own
target to 20 % to keep that working set on chip, and every ALU limiter
falls: the SIMD groups wait on their own stack, not on the weights or
the ALUs. The 2.6× instructions for 4× the tokens cost 5× the time. A
multi-row body that holds more rows than about two needs fewer live
values per thread (fewer rows per SIMD group, or accumulators in
threadgroup memory), not fewer instructions. The launch limiter reads
65 % in both runs; what it measures here is not established.

## The register-fragment verify matmul (2026-10-01)

`make bench-matvec-rows ARGS="4 frag"
CAPTURE='frag-IQ4_XS-17408x5120-t4-iq4_xs_f2'`: `nu_matmul_iq4_xs_f2`,
the tile that decodes each lane's weight segment straight into its
`simdgroup_matrix` elements (no threadgroup staging), on the same case
as the 16 × 8 tile above (export
`frag-IQ4_XS-17408x5120-t4-iq4_xs_f2_2026-10-01T0600_max.csv`, not
committed; profiled at Maximum). The benchmark reads 0.388 ms (122 GB/s)
against the tile's 0.523 (90.6).

| Counter | 16 × 8 tile | fragment tile |
| --- | ---: | ---: |
| ALU Utilization | 47.9 % | 53.1 % |
| Instruction Throughput Limiter / Utilization | 91.0 / 27.3 % | 86.1 / 29.4 % |
| F32 Limiter / Utilization | 68.0 / 47.8 % | **84.8 / 64.2 %** |
| Integer and Complex Limiter / Utilization | 56.4 / 43.1 % | 33.7 / 24.8 % |
| Integer and Conditional Limiter | 35.6 % | 38.4 % |
| Kernel ALU instructions (16 dispatches) | 25.9 × 10⁹ | 21.4 × 10⁹ |
| ALU instruction mix: float / integer and complex / integer and conditional | 50.0 / 22.5 / 27.5 % | 60.4 / 11.7 / 27.9 % |
| Kernel Occupancy / Occupancy Manager Target | 37.7 / 82.1 % | 26.1 / **33.9 %** |
| L1 Register / Threadgroup Residency | 1.5 / 23.3 % | 11.0 / 0.2 % |
| Register L1 Read / Write Bandwidth | 6.5 / 7.3 | 195.8 / 210.7 |
| Threadgroup Memory L1 Read Bandwidth | 321.1 | 5.3 |
| Stack L1 Read / Write Bandwidth | 0 / 0 | 0 / 0 |
| Buffer L1 Read Bandwidth / Miss Rate | 688.6 / 19.7 % | 762.0 / 12.5 % |
| Last Level Cache Limiter / MMU Limiter | 9.1 / 3.1 % | 5.2 / 3.6 % |

(Bandwidths as Xcode exports them, GB/s.)

**Reading: the F32 pipe, which the 8 × 8 multiplies occupy.** Dropping
the staging removed 17 % of the instructions and most integer work, and
the kernel is now bound on the F32 pipe (limiter 85 %). The registers no
longer fit the core's register file (11 % of the L1 holds registers,
400 GB/s of register traffic through it, no stack spill), and the
occupancy manager lowers its target from 82 % to 34 %. A probe with the
decode removed (`_p2`, only the MMAs and fragment moves) ran the same
case in 0.234 ms, about 6 TFLOP/s of 8 × 8 work: the multiplies alone are
60 % of the kernel. Taking the decode's float work out of the F32 pipe
(the MMAs multiplying the integer codes from a half table, the group
scale applied once to the partial product) measured no gain, nor did
F32 weight fragments, shared block headers, fewer rows per SIMD group, or
a software-pipelined decode (KERN-24's ledger). At 4 tokens half of every
8 × 8 multiply is padding columns, and nothing can fill them (a column
shares its A operand, so it cannot carry another K range or row set);
the padded multiplies set the floor of any `simdgroup_matrix` body at
small token counts on this GPU, and a body that does fewer F32
operations per weight is a scalar one, which KERN-12 found register-bound
past two rows.
