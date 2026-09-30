# The Apple M4 Pro GPU, as measured

What this engine's kernels meet on the M4 Pro's GPU (16 cores, 273 GB/s
unified memory), each fact with its measurement or its source. Captures
are taken with `make capture` or a micro-benchmark's `CAPTURE=` and read
in Xcode ([development.md § GPU counters by
capture](../development.md#gpu-counters-by-capture)); exported counter
tables stay under `.zig-cache/trace/`, the numbers that matter are copied
here.

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
