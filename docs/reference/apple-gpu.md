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
  Maximum) and may sample a subset of cores. Absolute times at Medium
  run slower than a benchmark at full clocks; ratios between units shift
  with the clock, so a limiter read at Medium is confirmed at Maximum
  before an experiment depends on it.

## `nu_matvec_q4_k` on Qwen's `ffn_down` shape (2026-09-30)

`make bench-kernels ARGS=Q4_K CAPTURE='matvec-Q4_K-5120x17408
(ffn_down)-block'`: 64 dispatches of the 5,120 × 17,408 Q4_K matvec
(50.1 MB) in one command buffer, profiled in Xcode 27 on macOS 27.0 at
**performance state Medium, 9 of 16 cores sampled**. The benchmark itself
reads 127–150 GB/s on this shape; the profiled replay took 31.09 ms, about
103 GB/s, the lower clocks.

| Counter | Value |
| --- | ---: |
| Instruction Throughput Limiter / Utilization | 75.4 % / 9.9 % |
| Integer and Complex Limiter / Utilization | 71.2 % / 38.6 % |
| F32 Limiter / Utilization | 16.0 % / 13.1 % |
| Integer and Conditional Limiter / Utilization | 6.3 % / 5.5 % |
| ALU instruction mix: integer and complex / float / integer and conditional | 50.8 / 34.6 / 14.6 % |
| Kernel Occupancy / Occupancy Manager Target | 23.1 % / 47.4 % |
| Allocated registers / high register / spilled | 192 / 192 / 16 bytes |
| L1 Register Residency / Buffer Residency / Eviction Rate | 36.2 / 41.0 / 13.7 % |
| L1 Cache Limiter | 17.4 % |
| Last Level Cache Limiter / Miss Rate | 0.2 % / 95.8 % |
| MMU Limiter / TLB Miss Rate | 0.3 % / 74.6 % |

**Reading (at Medium, to confirm at Maximum).** The kernel is not waiting
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
