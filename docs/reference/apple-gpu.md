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
(50.1 MB) in one command buffer, profiled in Xcode 27 on macOS 27.0 at
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
| Last Level Cache Limiter / Miss Rate | 0.2 / 95.8 % | 1.5 % / — |
| MMU Limiter / TLB Miss Rate | 0.3 / 74.6 % | 2.0 / 34.4 % |

**Reading, confirmed at full clocks.** The kernel is not waiting
on memory: the last-level cache, the L1, and the MMU are barely limiters,
and the cache misses are the expected streaming. It is **issue-bound on
the integer and complex pipe**: half its ALU instructions are integer and
complex (the Q4_K nibble unpacking, scale extraction, and integer-to-float
conversions), and that pipe is the limiter 71 % of the time while only 39 %
utilized. Occupancy is half what the occupancy manager targets, with 192
registers per thread and a small spill, so fewer SIMD groups are in
flight to hide load latency. KERN-05 cut instruction count without a gain;
what this adds is *which* instructions: the integer and complex ones.
That ranks the single-row matvec's ideas: a decode that produces floats
without integer-to-float conversions (the half magic-number form), fewer
unpacking operations per weight, and fewer live registers, each read
against these counters.
