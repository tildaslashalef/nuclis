<!-- kind: record -->
# Benchmarking: the record

The dated measurements behind [the benchmarking guide](../README.md), kept as written; older records are in [history.md](../history.md).

## The benchmarks on macOS 27 (2026-10-04)

Every family re-measured on both engines on one machine and OS, so the
README's columns share a date. Apple M4 Pro (12 CPU, 16 GPU cores), 48 GB,
macOS 27.0.1 (26A434), AC, no thermal warning recorded, display kept on;
`sudo purge` once before the first run. nuclis `c55604c` (0.6.0-dev,
ReleaseSafe, Zig 0.17.0); the reference at `7620399` (built with AppleClang
21.0.0), the server flags of [llama-cpp.md](../llama-cpp.md#run-the-workload-2026-09-06).
Both sides take one warmup and **three** measured requests at every length,
32,639 included (the 2026-09 records took one there), 128 greedy output
tokens, context 32,768, F16 KV. The reference **replays** the committed
token arrays (`--replay`), so both dates and both engines process identical
tokens; Gemma 4 E4B, never measured before, gets its own arrays
(`tests/fixtures/run-2026-10-04-gemma4-e4b/`). Ten minutes idle between
families. Rates are mean ± sample standard deviation; Δ is against the
family's 2026-09 record on macOS 26.

### Reference: Qwen3.8-27B (2026-10-04)

[reference-2026-10-04-qwen38.json](../reference-2026-10-04-qwen38.json),
replaying `run-2026-09-06` and `boundary-2026-09-06` (the server's own
rendering of the template and its tokens equal the fixture's):

| Prompt tokens | Prefill tok/s | Decode tok/s | 2026-09 prefill / decode | Δ decode |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 92.03 ± 0.07 | 10.21 ± 0.00 | 89.19 / 9.66 | +5.7 % |
| 4,096 | 95.49 ± 1.48 | 10.02 ± 0.21 | 89.26 / 9.21 | +8.8 % |
| 16,384 | 79.31 ± 2.06 | 7.54 ± 0.28 | 74.07 / 7.32 | +3.1 % |
| 32,639 | 69.30 ± 0.25 | 6.56 ± 0.03 | 67.28 / 6.71 | −2.2 % |

The reference got faster too on the new OS, except decode at 32K. Server
RSS 17.2–17.5 GiB between requests (weights memory-mapped).

### Reference: Gemma 4 12B QAT (2026-10-04)

[reference-2026-10-04-gemma4-qat.json](../reference-2026-10-04-gemma4-qat.json),
replaying `run-2026-09-12-gemma4-qat` (the server's rendering equals the
fixture's):

| Prompt tokens | Prefill tok/s | Decode tok/s | 2026-09 prefill / decode | Δ decode |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 226.66 ± 0.09 | 30.15 ± 0.18 | 224.46 / 27.69 | +8.9 % |
| 4,096 | 221.32 ± 2.39 | 28.05 ± 0.51 | 216.82 / 25.69 | +9.2 % |
| 16,384 | 171.55 ± 0.42 | 22.42 ± 0.46 | 158.78 / 22.10 | +1.5 % |
| 32,639 | 144.53 ± 0.58 | 19.12 ± 0.30 | 143.72 / 20.94 | −8.7 % |

The same shape as Qwen's: faster at short context, slower decode at 32K
(three samples on both dates). Server RSS 7.3–8.1 GiB.

### Reference: Gemma 4 26B-A4B (2026-10-04)

[reference-2026-10-04-gemma4-26b-a4b.json](../reference-2026-10-04-gemma4-26b-a4b.json),
replaying `run-2026-09-18-gemma4-26b-a4b` (the server's rendering equals
the fixture's):

| Prompt tokens | Prefill tok/s | Decode tok/s | 2026-09 prefill / decode | Δ decode |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 584.62 ± 0.84 | 68.14 ± 0.17 | 580.76 / 68.02 | +0.2 % |
| 4,096 | 559.48 ± 0.13 | 62.37 ± 0.04 | 548.41 / 60.82 | +2.5 % |
| 16,384 | 456.93 ± 4.34 | 53.31 ± 0.30 | 459.46 / 50.67 | +5.2 % |
| 32,639 | 325.68 ± 2.24 | 43.36 ± 0.61 | 343.38 / 44.09 | −1.6 % |

The mixture of experts moved least: within ±5 % everywhere, 32K prefill
−5 %. Server RSS 14.2–14.7 GiB.

### Reference: Muse Glimmer 30B (2026-10-04)

[reference-2026-10-04-muse-glimmer.json](../reference-2026-10-04-muse-glimmer.json),
replaying `run-2026-09-19-muse-glimmer`. The server's own rendering
differs from the fixture's, as expected: the template carries the
server's date, so the replay is what keeps the 2026-09 tokens.

| Prompt tokens | Prefill tok/s | Decode tok/s | 2026-09 prefill / decode | Δ decode |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 96.43 ± 0.46 | 14.24 ± 0.01 | 95.28 / 13.69 | +4.0 % |
| 4,096 | 89.79 ± 6.49 | 11.00 ± 1.14 | 93.01 / 12.14 | −9.4 % |
| 16,384 | 78.67 ± 1.86 | 9.63 ± 0.20 | 80.96 / 10.07 | −4.3 % |
| 32,639 | 75.43 ± 0.27 | 9.38 ± 0.08 | 76.86 / 9.98 | −6.0 % |

The 4,096 row is disturbed: its three requests slowed one after the
other (prefill 95.9 → 90.4 → 83.0, decode 12.3 → 10.6 → 10.1, 22:01–22:03
local) while 512 before them was steady and 32,639 after them holds
within ±0.3; no thermal warning was recorded. Server RSS 15.2–15.5 GiB.

Re-run alone on an idle machine after a 10-minute cool-down
([reference-2026-10-04-muse-glimmer-4k.json](../reference-2026-10-04-muse-glimmer-4k.json),
23:17): prefill **95.31 ± 1.01**, decode **12.25 ± 0.75** tok/s (2026-09:
93.01 / 12.14). The decode fell again within the run (warmup 13.96, then
12.71, 12.65, 11.38) while prefill held, so the spread at this length is
the reference's own, not the machine's. The re-run's row is the published
one (`muse/acceptance` lists its record first).

### Reference: Gemma 4 E4B QAT (first record, 2026-10-04)

[reference-2026-10-04-gemma4-e4b.json](../reference-2026-10-04-gemma4-e4b.json),
on its own arrays (`--family gemma4-e`, no replay): E4B's template renders
thinking off as a bare `<|turn>model\n`, so its prompts differ from the
12B's in the template tokens (prefix 4, suffix 12). The arrays and the run
are kept as `tests/fixtures/run-2026-10-04-gemma4-e4b/`.

| Prompt tokens | Prefill tok/s | Decode tok/s |
| ---: | ---: | ---: |
| 512 | 597.96 ± 0.14 | 62.10 ± 0.01 |
| 4,096 | 600.85 ± 0.10 | 59.26 ± 0.06 |
| 16,384 | 516.49 ± 2.83 | 52.29 ± 0.07 |
| 32,639 | 388.42 ± 2.24 | 44.58 ± 0.15 |

Server RSS 4.6–4.7 GiB.

### nuclis: Qwen3.8-27B (2026-10-04)

[nuclis-2026-10-04.json](../nuclis-2026-10-04.json), at
`d008abf` (clean tree; the binary built at `c55604c`, documents and scripts
only between), 23:31–00:40:

| Prompt tokens | Prefill tok/s | Decode tok/s | 2026-09 prefill / decode | Δ decode |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 94.21 ± 0.04 | 11.67 ± 0.01 | 90.45 / 10.62 | +9.9 % |
| 4,096 | 85.35 ± 1.94 | 10.38 ± 0.76 | 83.70 / 10.20 | +1.8 % |
| 16,384 | 62.69 ± 0.39 | 8.30 ± 0.06 | 62.70 / 8.27 | +0.3 % |
| 32,639 | 48.91 ± 0.04 | 7.53 ± 0.03 | 49.55 / 7.55 | −0.2 % |

Against today's reference:

| Prompt tokens | Decode nuclis / llama.cpp | ratio | Prefill nuclis / llama.cpp | ratio |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 11.67 / 10.21 | 1.14× | 94.21 / 92.03 | 1.02× |
| 4,096 | 10.38 / 10.02 | 1.04× | 85.35 / 95.49 | 0.89× |
| 16,384 | 8.30 / 7.54 | 1.10× | 62.69 / 79.31 | 0.79× |
| 32,639 | 7.53 / 6.56 | 1.15× | 48.91 / 69.30 | 0.71× |

Decode still leads at every length. The 2026-10-03 weight-streaming matvec work shows at 512 (+9.9 %)
and fades with context, where attention, not the matvecs, dominates.
Prefill falls behind from 4K on. Peak RSS 2.4 GiB (weights mapped),
footprint 2.6–2.7 GiB.

**The 4,096 step on the slow dense models.** Its decode fell within the
run (11.22 → 10.21 → 9.71) while 512 before it and 16K/32K after it held
within ±0.06; the Muse reference did the same at 4K (12.62 → 10.13, and
again on its re-run). The fast models' 4K rows are flat (12B QAT, 26B-A4B,
E4B within ±0.5 %). At 4K each request runs about a minute at full load
on a 27–30B model, long enough to heat the chip from its cooled start,
too short to settle; at 16K and longer the rate is the settled one. Both
engines start each family from the same 10-minute cool-down, so the
comparison holds; the 4K row carries the larger spread.

### nuclis: Gemma 4 12B QAT (2026-10-05)

[nuclis-2026-10-05-gemma4-qat.json](../nuclis-2026-10-05-gemma4-qat.json)
(dated by its start after midnight), at `5ef0544`, clean, 00:50–01:35:

| Prompt tokens | Prefill tok/s | Decode tok/s | 2026-09 prefill / decode | Δ decode |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 207.02 ± 0.07 | 25.38 ± 0.02 | 179.45 / 23.96 | +5.9 % |
| 4,096 | 159.40 ± 0.03 | 23.43 ± 0.05 | 143.25 / 20.71 | +13.1 % |
| 16,384 | 104.62 ± 1.09 | 17.93 ± 0.35 | 98.67 / 17.87 | +0.3 % |
| 32,639 | 72.42 ± 0.14 | 15.11 ± 0.23 | 73.04 / 16.24 | −6.9 % |

Against today's reference:

| Prompt tokens | Decode nuclis / llama.cpp | ratio | Prefill nuclis / llama.cpp | ratio |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 25.38 / 30.15 | 0.84× | 207.02 / 226.66 | 0.91× |
| 4,096 | 23.43 / 28.05 | 0.84× | 159.40 / 221.32 | 0.72× |
| 16,384 | 17.93 / 22.42 | 0.80× | 104.62 / 171.55 | 0.61× |
| 32,639 | 15.11 / 19.12 | 0.79× | 72.42 / 144.53 | 0.50× |

Short-context prefill gained most (+15 % at 512, +11 % at 4K); 32K decode
lost 7 %, as the reference's did (−8.7 %). The gap to the reference is
unchanged in shape: decode 0.79–0.84×, prefill halving by 32K. Peak RSS
10.8 GiB, footprint 11.0 GiB.

### nuclis: Gemma 4 26B-A4B (2026-10-05)

[nuclis-2026-10-05-gemma4-26b-a4b.json](../nuclis-2026-10-05-gemma4-26b-a4b.json),
at `781a084`, clean, 01:45–02:08:

| Prompt tokens | Prefill tok/s | Decode tok/s | 2026-09 prefill / decode | Δ decode |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 521.06 ± 0.70 | 54.02 ± 0.02 | 500.42 / 55.29 | −2.3 % |
| 4,096 | 358.23 ± 0.07 | 48.64 ± 0.03 | 346.17 / 49.44 | −1.6 % |
| 16,384 | 206.89 ± 6.33 | 37.93 ± 1.86 | 206.44 / 39.55 | −4.1 % |
| 32,639 | 136.86 ± 0.83 | 31.23 ± 0.03 | 134.74 / 31.30 | −0.2 % |

Against today's reference:

| Prompt tokens | Decode nuclis / llama.cpp | ratio | Prefill nuclis / llama.cpp | ratio |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 54.02 / 68.14 | 0.79× | 521.06 / 584.62 | 0.89× |
| 4,096 | 48.64 / 62.37 | 0.78× | 358.23 / 559.48 | 0.64× |
| 16,384 | 37.93 / 53.31 | 0.71× | 206.89 / 456.93 | 0.45× |
| 32,639 | 31.23 / 43.36 | 0.72× | 136.86 / 325.68 | 0.42× |

The 16,384 decode spread is one request (35.79 between 39.20 and 38.79;
without it the row reads 39.00); prefill fell from 214 to 203 on the last
two requests. Otherwise
unchanged from 2026-09 within ±2.3 %: the mixture of experts gained
nothing from the new OS on either engine. Peak RSS 7.3 GiB, footprint
7.5 GiB.

### nuclis: Muse Glimmer 30B (2026-10-05)

[nuclis-2026-10-05-muse-glimmer.json](../nuclis-2026-10-05-muse-glimmer.json),
at `b7c0b02`, clean, 02:18–03:18; its 4,096 reference row is the re-run's
(the record names each row's source):

| Prompt tokens | Prefill tok/s | Decode tok/s | 2026-09 prefill / decode | Δ decode |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 96.41 ± 0.02 | 11.62 ± 0.00 | 93.49 / 9.60 | +20.9 % |
| 4,096 | 84.13 ± 2.59 | 10.09 ± 0.80 | 80.51 / 8.26 | +22.1 % |
| 16,384 | 68.56 ± 0.38 | 8.08 ± 0.04 | 67.83 / 7.19 | +12.4 % |
| 32,639 | 59.59 ± 0.01 | 7.19 ± 0.01 | 59.20 / 6.62 | +8.6 % |

Against today's reference:

| Prompt tokens | Decode nuclis / llama.cpp | ratio | Prefill nuclis / llama.cpp | ratio |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 11.62 / 14.24 | 0.82× | 96.41 / 96.43 | 1.00× |
| 4,096 | 10.09 / 12.25 | 0.82× | 84.13 / 95.31 | 0.88× |
| 16,384 | 8.08 / 9.63 | 0.84× | 68.56 / 78.67 | 0.87× |
| 32,639 | 7.19 / 9.38 | 0.77× | 59.59 / 75.43 | 0.79× |

The largest gain of the night: decode +9 to +22 % over 2026-09, from
0.70× of the reference at 512 to 0.82×. Its tree moved from `2026-09-19`'s
to after the 2026-10-03 weight-streaming matvec work, which Qwen's +9.9 % also reflects; the
share due to the OS was not separated. The 4,096 step drifts as on the other slow
dense runs (10.96 → 9.90 → 9.40). Peak RSS 1.8 GiB, footprint 2.1 GiB.

### nuclis: Gemma 4 E4B QAT (first record, 2026-10-05)

[nuclis-2026-10-05-gemma4-e4b.json](../nuclis-2026-10-05-gemma4-e4b.json),
at `f72a81c`, clean, 03:28–03:45, on its own arrays:

| Prompt tokens | Prefill tok/s | Decode tok/s | Decode nuclis / llama.cpp | Prefill nuclis / llama.cpp |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 531.03 ± 0.39 | 46.84 ± 0.02 | 0.75× | 0.89× |
| 4,096 | 430.80 ± 0.06 | 45.40 ± 0.04 | 0.77× | 0.72× |
| 16,384 | 276.77 ± 4.57 | 38.02 ± 1.97 | 0.73× | 0.54× |
| 32,639 | 189.30 ± 0.18 | 33.08 ± 0.38 | 0.74× | 0.49× |

The same shape as the 12B's, a little further behind on decode. The 16K
spread is again one request (35.74 after 39.15, 39.17; without it 39.16),
as on the 26B-A4B: both Gemma runs lost about 9 % on one 16K request,
with prefill dipping too; not investigated. Peak RSS 2.0 GiB, footprint
2.2 GiB.

### Speculation pairs (2026-10-05)

[speculative-2026-10-05/](../speculative-2026-10-05/), at
`7c34501`, 03:56–04:54, in one sitting after a 10-minute cool-down and
90 s idle before each cell (`make spec-matrix ARGS='--model <key>
--contexts 512,32639 --drafts <n> --sampling greedy --cooldown 90'`):
three off/on pairs per cell, 128 greedy tokens, each family on its
acceptance arrays (E4B now on its own, not the 12B's), at the catalogue
draft length.

| Model | Draft | 512: off → on | 32,639: off → on | C / 50E at 512, 32K |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.8-27B | 7 | 11.63 → 17.44 (1.50×) | 7.86 → 9.84 (1.26×) | 1.15, 2.03 |
| Gemma 4 12B QAT | 5 | 25.38 → 46.47 (1.85×) | 15.77 → 15.48 (0.98×) | 0.43, 1.29 |
| Gemma 4 E4B QAT | 6 | 46.85 → 81.93 (1.75×) | 33.63 → 37.55 (1.12×) | 0.24, 0.53 |
| Muse Glimmer 30B | 6 | 11.62 → 16.09 (1.42×) | 7.29 → 7.82 (1.08×) | 1.24, 2.56 |
| Gemma 4 26B-A4B (off) | 4 | 53.97 → 35.94 (0.67×) | 28.93 → 19.68 (0.68×) | 0.56, 1.02 |

Speedups are the median of the pairs' on/off ratios. Qwen reproduces
2026-10-03's cooled run (1.51×, 1.26×). Muse's off rate rose with its
plain decode (8.5 → 11.6 at 512), so its ratio fell from 1.50× to 1.42×
while its on rate rose (12.6 → 16.1). E4B's 512 ratio, 2.10× on the
12B's arrays, is 1.75× on its own. The 26B-A4B stays off: 0.67–0.68×.

Short code on Qwen (`--contexts code --drafts 7 --sampling greedy
--cooldown 90`, 09:52 on 2026-10-05, the binary of the runs above; the
tree's only change was `scripts/repo/site-check.py`, so the report's revision
reads `092df76+dirty`): **11.79 → 20.87 tok/s, 1.77×**, C / 50E 0.96
([qwen38-code.json](../speculative-2026-10-05/qwen38-code.json)).
The README's "20 tokens/s on short code" holds on macOS 27.0.1 (2026-10-01: 20.26).
The draft lengths these pairs run at are each entry's catalogue default,
chosen by the rule in [history.md § The re-priced speculative
verdicts](../history.md#the-re-priced-speculative-verdicts-2026-10-01).

## Per-kernel profile (2026-09-07)

Record (2026-09-07,
Apple M4 Pro, ReleaseSafe, 255 measured steps):
[benchmarks/nuclis-profile-2026-09-07.json](../nuclis-profile-2026-09-07.json).
Grouped by encoding, ms per token of the 104.8 ms attributed (109.0 ms
command-buffer time in profile mode; 101 ms unprofiled):

| Group | ms/token | dispatches/token | Note |
| --- | ---: | ---: | --- |
| IQ4_XS (specialized) | 28.1 | 117 | 169 GB/s on 17,408×5,120 |
| Q5_K (specialized) | 27.0 | 131 | 193 GB/s |
| Q4_K (specialized) | 23.1 | 103 | 158 GB/s |
| Q6_K (specialized) | 5.3 | 24 | 241 GB/s (output head included, 0.8/token) |
| Q3_K (generic) | 6.2 | 7 | **43 GB/s** |
| IQ3_S (generic) | 3.9 | 4 | **39 GB/s** |
| IQ4_NL (generic) | 3.0 | 7 | 108 GB/s |
| Q8_0 (generic) | 1.7 | 104 | 96 of these are 48×5,120 β/α projections at 18 GB/s: launch-bound |
| Everything else | 6.6 | 739 | rmsnorm 2.7 ms (209 dispatches of ~13 µs), delta 1.3, add 0.6, attention 0.5 |

The generic kernel costs 14.8 ms per token, not the ~9 ms the previous
calculation assumed: Q3_K and IQ3_S run at ~40 GB/s, half the rate used in
that estimate.

The most informative view is time per 256-value block on the same shape,
because it removes bytes-per-block from the comparison:

| Encoding | 17,408×5,120 (1,088 threadgroups) | 5,120×17,408 (320 threadgroups) |
| --- | ---: | ---: |
| IQ4_XS | 0.81 ns/block | 0.80 |
| Q5_K | 0.91 | 0.94 |
| Q4_K | 0.91 | 1.02 |
| Q6_K | 0.89 | 0.92 |

All four specialized kernels spend the same time per block. Q4_K's lower GB/s
is only its smaller block (144 bytes against Q5_K's 176 and Q6_K's 210): the
limiter is per-block instruction work (decode, scale math, the 8-lanes-per-block
structure, input loads), shared by all four, not Q4_K's byte layout. Shapes
with 320 threadgroups are ~10 % slower per block than those with 1,088, an
occupancy effect that also explains the micro-benchmark's slower
20,480×17,408 column. Reaching the bandwidth floor (~59 ms per token) needs
~0.5 ns per block: fewer instructions per block or more work in flight, and
that is now the measured target for kernel work rather than a guess.

## Per-kernel profile after the specialized matvecs (2026-09-08)

The profile after the specialized Q3_K, IQ3_S, and IQ4_NL matvecs (2026-09-08), same workload and 255 measured steps:
[raw report](../nuclis-profile-2026-09-08.json). Attributed GPU time
is 99.38 ms/step; command-buffer GPU time is 103.31 ms/step. Dispatch count
remains 1,236 per step, with no unsampled dispatches.

| Group | ms/step |
| --- | ---: |
| IQ4_XS specialized | 27.53 |
| Q5_K specialized | 26.89 |
| Q4_K specialized | 23.08 |
| Q6_K specialized | 5.30 |
| Q3_K specialized | 2.29 |
| IQ3_S specialized | 1.34 |
| IQ4_NL generic | 3.02 |
| Q8_0 generic | 1.65 |
| Other kernels | 8.27 |

Q3_K/IQ3_S together fell from the recorded 10.1 to 3.63 ms/step; the generic
matvec group now costs 4.67 ms/step. Q3_K takes 0.940/0.944 ns/block on
17,408×5,120 and 5,120×17,408; IQ3_S takes 0.934/0.969 respectively, meeting
the ≤1.0 ns/block target on both shapes. Other kernels rose from 6.6 to
8.27 ms/step in this profile, so the removed matvec time should not be read
as an equal wall-time gain. Only the unprofiled benchmark above establishes
that gain.

## Per-kernel profile after the merged projections (2026-09-08)

The profile after the merged projections (2026-09-08), 255 measured steps:
[raw report](../nuclis-profile-c12-2026-09-08.json). Dispatches fell
from 1,236 to 932 per step (304 removed). Attributed time is 92.49 ms/step,
command-buffer time 97.61 ms/step; no dispatches were unsampled. Each merged
entry reports the combined rows and weight bytes, with encoding omitted because
one dispatch can execute several encodings.

| Group | ms/step | Dispatches/step |
| --- | ---: | ---: |
| Merged FFN gate/up + SiLU | 35.58 | 64 |
| Merged DeltaNet input projections | 15.45 | 48 |
| Merged attention input projections | 4.25 | 16 |
| Standalone specialized matvecs | 30.13 | |
| Generic matvecs | 1.37 | |
| Other kernels | 5.69 | |

These groups replace the earlier per-encoding attribution for merged matrices;
comparing one encoding's standalone total before/after would omit its work now
inside `matvec_segments`. Plain `bench` above remains the speed measurement.

## Memory of a running process (2026-09-26)

Measured on Qwen3.8-27B-UD-Q4_K_M, context 16384, KV F16, `agent`, M4 Pro
48 GB, 2026-09-26: 16.11 of 16.46 GB of weights resident, session state
1.23 GB, Metal buffers 0.14 GB, driver 0.08-0.22 GB, CPU 0.31 GB; footprint
1.75-1.90 GB (peak 1.95 GB during prefill); total 17.9-18.0 GB, steady from
ready through a 16-step turn. System wired memory rises by about 17 GB while
command buffers run (the GPU wires what it reads) and falls back between
them. On exit (Ctrl-C twice mid-turn, Ctrl-D with a tool's child running,
Ctrl-C idle) the process is gone in 0.5-0.6 s with status 0, the footprint is
returned, wired memory falls back to its idle 3 GB, and a tool's child is
killed; the GGUF stays in the page cache as reclaimable cached files, which
is what makes the next load 20 s instead of a cold read.
