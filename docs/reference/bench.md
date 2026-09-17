# Benchmarking nuclis

`nuclis bench` repeats greedy prefill and decode on one loaded model and
reports every sample plus means over the measured runs. It exists so that
every performance claim about nuclis comes from the same command, with the
same definitions, and can be reproduced.

```sh
zig build -Dmetal=true -Doptimize=ReleaseSafe --global-cache-dir .zig-cache/global
./zig-out/bin/nuclis bench --backend metal --prompt-file prompt.txt \
  --max-tokens 128 --ctx-size 4096 --repeat 3 --warmup 1 --json
```

Defaults: 1 warmup, 3 measured runs, 32 output tokens, the pinned chat
template with thinking off (`--raw` sends literal text). The model,
backend, and context come from `~/.nuclis/nuclis.json` when it exists
(8,192 context by default; `make bench` passes `--ctx-size 2048`
explicitly), and the report records the file it ran with (`Config:` /
`config`) so a published number names its configuration. The output
budget, repetitions, and sampling never come from the file.
Sampling is greedy by default so runs are comparable — `bench` never
applies a model's sampling profile or the file's overrides, unlike
`generate` and `chat`;
`--temperature`, `--top-k`, `--top-p`, `--min-p`, `--presence-penalty`,
`--repetition-penalty`, and `--seed` run a sampled benchmark, which the
report labels (`sampling`, `sampling_options`, `seed`) and, on Metal, shows
the GPU top-k fallback count per run (absent when a penalty forces the
full-logit path). Published sampled numbers must state their options; the
official profiles are in
[generation.md](generation.md#sampling-profiles-and-the-selection-chain-modl-01).
Repetitions reuse one loaded
engine and reset the session between runs; the first run therefore pays
first-touch page-in of the mapped weights and is normally the warmup.

## Definitions

| Field | Meaning |
| --- | --- |
| `load_milliseconds` | Opening the file, parsing the directory, mapping, binding, vocabulary, session allocation, and backend creation. Excludes weight page-in, which happens lazily on first use. |
| `prefill_milliseconds` | All prompt token steps; the last one computes logits. |
| `prefill_tokens_per_second` | `prompt_tokens / prefill`. |
| `first_token_milliseconds` | Time to first token: prefill plus sampling the first token. |
| `decode_milliseconds` | From the first sampled token to the last, covering `generated_tokens - 1` model steps. |
| `decode_tokens_per_second` | `(generated_tokens - 1) / decode`; absent when fewer than two tokens were generated. |
| `gpu_busy_milliseconds` | Sum of Metal `GPUEndTime - GPUStartTime` over completed command buffers during the run. Absent on the CPU backend. Excludes host encoding, copies, and waits. |
| `kv_precision`, `session_bytes` | The attention cache layout the session used (`f16` or `f32`; always `f32` on the CPU reference) and the session block's size, page-padded (KV cache plus recurrent state). Records taken in different precisions are not comparable without stating this. |
| `profile` | Present with `--profile`: per-kernel GPU time per step (see below). |
| `stop_reason` | `eos`, `token_budget`, `context_limit`, or `cancelled`; a run that stops early changes the token counts, so the reason is always reported. Ctrl-C ends the benchmark after the current run, which is reported but excluded from means. |

Means are computed over measured (non-warmup) samples only; a rate that could
not be measured is omitted from the sample and the mean. The text and JSON
renderings derive from the same `Report` value.

Cold-start measurement (weights not yet in the page cache) requires a fresh
process and an evicted cache; `bench` does not simulate it. Record it
separately by running once after a reboot or after `purge`.

## What to record with results

Hardware, macOS version, Zig version and build mode (the report includes the
mode), the artifact hash, the backend, prompt token count, output budget,
context, warmup policy, power mode, and whether other GPU work was running.
Compare against the pinned reference only at equal prompt tokens, output
tokens, context, and greedy sampling; see [reference-baseline.md](reference-baseline.md).

## Acceptance runs

The v0.1 acceptance workload is the reference's
([reference-baseline.md](reference-baseline.md)): 512, 4,096, 16,384, and
32,639 prompt tokens, 128 output tokens, greedy, one warmup, three measured
repetitions (one at 32,639, as the reference took), context 32,768, F16
KV cache. `make baseline` (`scripts/nuclis-baseline.py`) runs it and writes
`docs/benchmarks/nuclis-<date>.json`. The prompts are the reference
harness's own token arrays under `tests/fixtures/`, fed through `bench
--prompt-tokens`, so both engines processed identical tokens; text
renderings of the same prompts are written beside them for reading
(`.zig-cache/prompts/reference/`), and the script first checks that
`nuclis tokenize` turns the synthetic corpus into the reference's body
tokens at every cut (it does, through 32,620 tokens). The 512 array is not
the canonical tokenization of its own text (its corpus slice ends on a lone
space token that the pre-tokenizer merges with the suffix's newline: 511
tokens as text), which is why the runs take arrays rather than text.

### Warm record — 2026-09-10

[nuclis-2026-09-10.json](../benchmarks/nuclis-2026-09-10.json). Apple M4 Pro
(12 CPU, 16 GPU cores), 48 GiB, macOS 26.6.2 (25G83), AC power, Zig 0.16.0,
ReleaseSafe, `nuclis 0.1.0-dev` built from the ENGN-07 tree, pinned artifact
SHA-256 `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482`
re-hashed after the runs, backend metal, `--kv f16 --ctx-size 32768
--max-tokens 128`. Nothing else used the GPU; the runs went 512 → 32,639 in
one 46-minute sequence, so each length follows the previous one's load.
Mean ± sample standard deviation over the measured runs; the reference
columns are its accepted warm means (three samples; one at 32,639):

| Prompt tokens | Prefill tok/s | Reference | Decode tok/s | Reference | Decode ms/step | First token | Warmup (prefill / decode) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 512 | 90.45 ± 0.52 | 89.19 | 10.62 ± 0.02 | 9.66 | 94.2 | 5.66 s | 88.2 / 10.61 |
| 4,096 | 83.70 ± 0.37 | 89.26 | 10.20 ± 0.11 | 9.21 | 97.4–99.2 | 48.9 s | 83.0 / 10.23 |
| 16,384 | 62.70 ± 0.57 | 74.07 | 8.27 ± 0.04 | 7.32 | 120.4–121.5 | 261 s | 65.3 / 8.74 |
| 32,639 | 49.55 (one run) | 67.28 | 7.55 (one run) | 6.71 | 132.5 | 659 s | 48.9 / 8.03 |

Every sample stopped with `stop_reason token_budget` at exactly the fixture's
token count and 128 generated tokens; 32,639 + 128 = 32,767 fits the 32,768
context with one token to spare and nothing truncated (a prompt that does
not fit is `ContextFull` before any run). Decode is above the reference at
every length (+10 % at 512, +11 % at 4K, +13 % at 16K, +13 % at 32K).
Prefill is at the reference at 512 (+1 %) and below it as the prompt grows
(−6 % at 4K, −15 % at 16K, −26 % at 32K): the batched attention chunk
kernel's cost grows with the visible cache (ENGN-05 measured the matmul
tile at the reference's rate at 4K; ENGN-08 profiled the kernel at 30 % of
the 16K prefill and found it latency-bound, not cache-bound:
[metal-backend.md § ENGN-08](metal-backend.md#long-context-prefill-attention-engn-08-2026-09-10-closed-without-a-kernel-change),
[roadmap](../roadmap.md#also-deferred)). The 16K row is lower than the 66.5 / 8.92
measured after the 13,399-token docs prompt in KERN-08 because the prompt is
longer (the decode rate at 16K context is what this row states) and the
run followed the 4K row without a cool-down.

**Memory.** The session block (KV cache and recurrent state, page-padded)
is 2,304,376,832 bytes (2.15 GiB) at 32,768 capacity in every run. Peak
resident set of the whole `bench` process (`/usr/bin/time -l`) was
2.35 GiB at every length, and its peak physical footprint 2.76–2.84 GB;
sampled with `footprint` during a run, the process is 2.2 GiB of
`MALLOC_LARGE` (the session and scratch), 0.23 GB of graphics
allocations, and only 25 MB of mapped file. The 16.46 GB weight file is
memory-mapped and read by the GPU, and macOS charges those pages to
system wired memory, not to the process (`vm_stat` showed 20.5 GiB wired
during a run on a machine also running an editor and a browser). The
headroom statement is therefore a calculation, not a single measurement:
48 GiB total − 16.46 GB weights − 2.84 GB process peak ≈ 29 GiB free for
the OS and tools at 32K context. Swap in use fell from 1,188 MB before the
first run to 1,012 MB after the last; no run grew it.

### Cold-start row — 2026-09-10

[nuclis-2026-09-10-cold.json](../benchmarks/nuclis-2026-09-10-cold.json).
After `sudo purge` (`vm_stat` free pages 380K →
2.5M; the script's header-only `tokenize` check ran first and touches only
the vocabulary pages), one process, 512 prompt tokens, no warmup, one run:

| Prompt tokens | Load | Prefill tok/s | First token | Decode tok/s | Decode ms/step | GPU busy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 (cold) | 777 ms | 43.42 | 11.79 s | 10.55 | 94.8 | 17.6 s |

Against the warm row (90.45 / 5.66 s / 10.62): loading is unchanged
because mapping does not page anything in; the first prefill pays for
faulting the 16.46 GB of weights from the SSD, about 6 s more than warm;
decode is at the warm rate because every weight is resident by the first
generated token. Peak footprint 2.77 GB, as warm. This is one sample
labelled cold; it is not averaged with the warm rows.

**Not done.** The `--trace-dir` comparison at a long position (8,192): both harnesses trace every position (64 layers × 20 KB per token,
about 10 GB per side at that depth) and nuclis's tracer synchronizes the
GPU after every layer, so neither can produce a single-position trace
within a session; a positional filter is a follow-up if long-position
numerics are ever in doubt (the F16 attention kernels are checked against
the F64 reference at 32,000 rows in `test-metal`).

## Observations so far

These are smoke observations from the synchronous bring-up bridge, not
acceptance results. 2026-09-07, Apple M4 Pro 48 GiB, macOS 26.6.2, Zig 0.16.0
ReleaseSafe, pinned artifact, 22-token chat prompt, 16 output tokens, context
256, one warmup and two measured runs:

```text
run    prompt  gen  stop           prefill ms  pp tok/s  first ms   decode ms  tg tok/s   gpu ms
 1         22   16  token_budget       9221.9      2.39    9222.4      6355.5      2.36  10712.2
 2         22   16  token_budget       9774.4      2.25    9774.9      6643.9      2.26  11524.0
```

With the GPU-resident plan (one command buffer per token), the same workload:

```text
run    prompt  gen  stop           prefill ms  pp tok/s  first ms   decode ms  tg tok/s   gpu ms
 1         22   16  token_budget       4103.5      5.36    4103.5      3005.9      4.99   6640.8
 2         22   16  token_budget       4098.2      5.37    4098.2      2995.4      5.01   6627.2
```

Decode improved from 2.3 to 5.0 tok/s and GPU busy time is ~93 % of wall time,
so the remaining gap to the bandwidth floor (~17 tok/s) and to the reference
(9.66 tok/s at 512 context) is inside the kernels, chiefly `nu_matvec`
geometry (~200 ms per token to read 16.1 GB is ~80 GB/s of a published 273).

With specialized matvec kernels for Q4_K/Q5_K/Q6_K/IQ4_XS (`make bench`:
64 output tokens, context 2048, three measured runs, 2026-09-07):

```text
run    prompt  gen  stop           prefill ms  pp tok/s  first ms   decode ms  tg tok/s   gpu ms
 1         22   64  token_budget       2500.3      8.80    2500.3      7357.9      8.56   8635.9
 2         22   64  token_budget       2675.2      8.22    2675.2      7395.0      8.52   8720.7
 3         22   64  token_budget       2488.8      8.84    2488.8      7382.4      8.53   8803.8
Measured mean over 3 runs: prefill      8.62 tok/s, decode      8.54 tok/s, first token 2554.7 ms
```

The same command on the generic kernels immediately before measured 4.93 tok/s
decode. Per-kernel bandwidth is in
[metal-backend.md § Specialized matvec](metal-backend.md#specialized-matvec).

The first per-kernel profile (below) showed that this run was committing 64
command buffers per token, not one: the CLI's layer observer, which exists for
Ctrl-C and the time limit, forced a `commit` after every layer on the GPU plan.
Splitting the observer into a values-free `check` and a trace-only `layer`
callback removed those round trips with no kernel change (2026-09-07):

```text
run    prompt  gen  stop           prefill ms  pp tok/s  first ms   decode ms  tg tok/s   gpu ms
 1         22   64  token_budget       2189.4     10.05    2189.4      6495.3      9.70   8596.1
 2         22   64  token_budget       2185.8     10.07    2185.8      6500.8      9.69   8603.9
 3         22   64  token_budget       2190.1     10.05    2190.1      6498.9      9.69   8601.0
Measured mean over 3 runs: prefill     10.05 tok/s, decode      9.69 tok/s, first token 2188.4 ms
```

GPU busy time per token is unchanged (~101 ms); wall time per decode step fell
from 117 to 103 ms, so decode is now GPU-bound. This is a 22-token prompt at
context 2048; it is not the reference's 512-token workload, and the 32K
acceptance runs (ENGN-07) remain.

KERN-03, Q3_K/IQ3_S specialized kernels (2026-09-08), same `make bench`
workload, Apple M4 Pro 48 GiB, macOS 26, Zig 0.16.0 ReleaseSafe, pinned
artifact SHA-256 `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482`:

| Run | Prefill tok/s | Decode tok/s | GPU busy ms (85 steps) |
| --- | ---: | ---: | ---: |
| 1 | 10.61 | 10.23 | 8171.6 |
| 2 | 10.65 | 10.23 | 8175.7 |
| 3 | 10.65 | 10.22 | 8177.0 |
| Mean | 10.63 | 10.23 | 8174.8 |

One warmup, all runs generated 64 tokens and stopped at `token_budget`.
Mean decode is 5.6 % above the recorded 9.69 tok/s baseline; GPU busy time
is 96.2 ms/step. This remains a short-context smoke measurement, not ENGN-07
acceptance. IQ4_NL still uses the generic kernel.

KERN-04 merged projections (2026-09-08), same hardware, artifact, build mode,
and standard `make bench` workload as KERN-03, with the artifact now under
`~/.nuclis/models/qwen/`:

| Run | Prefill tok/s | Decode tok/s | GPU busy ms (85 steps) |
| --- | ---: | ---: | ---: |
| 1 | 10.90 | 10.49 | 7955.5 |
| 2 | 10.90 | 10.59 | 7910.2 |
| 3 | 11.27 | 10.81 | 7733.3 |
| Mean | 11.02 | 10.63 | 7866.3 |

All runs stopped at the 64-token budget; one warmup preceded the three measured
runs. The mean decode rate is 3.9 % above KERN-03's recorded 10.23 tok/s. Individual
samples show variation; this is a short-prompt observation, not a long-context
acceptance result. Mean GPU busy time is 92.5 ms/step. Prompt processing still
uses sequential token steps; ENGN-02–ENGN-04 address its remaining throughput limit.

KERN-05 (2026-09-08) closed without a kernel change (every hypothesis measured
below the 5 % kernel-benchmark bar; the record is in
[metal-backend.md § KERN-05](metal-backend.md#kern-05--per-block-cost-research-2026-09-08-closed-without-a-kernel-change)).
The same `make bench` workload at the end of the session, kernels identical
to KERN-04: 11.09 tok/s prefill, 10.64 decode (runs 10.63 / 10.66 / 10.64), GPU
busy 7,844 ms per 85 steps (92.3 ms/step). The per-kernel profile budget
above is unchanged and was not re-recorded.

KERN-06 (2026-09-08), GPU partial top-k sampling, same hardware, artifact,
build, and `make bench` workload (22 prompt tokens, 64 generated, context
2048, one warmup, three measured runs; decode ms/step is
`decode_milliseconds / 63`). All configurations produced 64 tokens at
`token_budget`; the GPU path's tokens were checked bit-identical to the
reference path at fixed seeds (48 tokens × four configurations, including one
with 23 fallbacks) by comparing `generate --json` with and without
`--logits`, which forces the reference path:

| Sampling | Path | Decode tok/s | Decode ms/step | vs greedy | Fallbacks / run |
| --- | --- | ---: | ---: | ---: | ---: |
| greedy | GPU argmax | 10.77 | 92.9 | — | — |
| T 0.7, top-k 40, top-p 0.95 | GPU top-k | 10.69 | 93.6 | +0.7 ms | 0 |
| T 1.0, top-k 20, top-p 0.95 (thinking profile) | GPU top-k | 10.67 | 93.7 | +0.8 ms | 0 |
| T 0.7, top-k 0, top-p 0.95 (nucleus, GPU denominator) | GPU top-k | 10.51 | 95.1 | +2.2 ms | 0 |
| T 0.7, top-k 0, top-p 1 (ineligible) | full readback + CPU sort | 8.68 | 115.2 | +22.3 ms | — |

The last row is the pre-KERN-06 cost of every sampled token (the plan's 19 ms
estimate was taken at 9.69 tok/s greedy). The nucleus row's GPU busy time was
itself 1.9 ms/step above the top-k rows in this session (7,970 vs 7,805 ms per
85 steps with identical GPU work), so its CPU-side cost is within the 2 ms
acceptance bound; the top-k rows are within it outright. Greedy measured
10.77 here against 10.64 earlier in the session with the same binary's
kernels; the spread is the usual short-prompt variation.

ENGN-02 chunked prefill (2026-09-08), same hardware and artifact. Standard
`make bench` workload: prefill 35.02 tok/s (was 11.1; the 22-token prompt is
one chunk), decode 10.79 (unchanged), first token 628 ms (was ~1,980).
Longer raw prompts with `bench --raw --prompt-file … --max-tokens 32
--ctx-size 4096 --repeat 2`: 543 tokens → 51.3 tok/s prefill, 10.50 decode;
3,547 tokens → 33.3 prefill, 7.76 decode (the decode drop is attention over
the longer cache). The reference does 89.2/89.3 prefill at 512/4,096. These
are not the acceptance workloads (ENGN-07 builds the reference's exact prompts);
details in [metal-backend.md § Prefill in chunks](metal-backend.md#prefill-in-chunks-engn-02).

ENGN-03 causal tiled attention (2026-09-09), same hardware, artifact, and
methodology (`bench --raw --prompt-file … --max-tokens 32 --ctx-size 4096
--repeat 2`, one warmup and two runs; prompts are prefixes of
`docs/architecture.md` and `docs/spec.md`, 545 and 3,657 tokens; flags given
explicitly so the record does not depend on `~/.nuclis/nuclis.json`):

| Prompt tokens | Prefill tok/s | ENGN-02 | Reference (llama.cpp) | Decode after the prompt |
| ---: | ---: | ---: | ---: | ---: |
| 22 (`make bench`, context 2,048, 64 output) | 35.3 | 35.0 | — | 10.47 |
| 545 | 51.4 | 51.3 at 543 | 89.2 at 512 | 10.36 |
| 3,657 | 50.6 | 33.3 at 3,547 | 89.3 at 4,096 | 8.48 |

Prefill no longer falls with prompt length: attention over a chunk is one
dispatch per layer instead of three per token. The rate at 512 is unchanged
because attention was a small share there. Decode is untouched by ENGN-03 (the
`step` path keeps the three-dispatch kernels); its spread here (10.36–10.47
short, 8.48 after 3.6K) is the usual run-to-run variation and the longer
visible cache.

ENGN-04 chunkwise DeltaNet (2026-09-09), same hardware, artifact, and
methodology; the 16K prompt is a 53,300-byte prefix of the docs at
`--ctx-size 16384 --repeat 1` (one warmup and one run, about five minutes
each):

| Prompt tokens | Prefill tok/s | ENGN-03 | Reference (llama.cpp) | Decode after the prompt | Reference decode |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 22 (`make bench`) | 34.9 | 35.3 | — | 10.33 | 9.66 at 512 |
| 545 | 52.6 | 51.4 | 89.2 at 512 | 10.28 | 9.66 |
| 3,657 | 53.0 | 50.6 | 89.3 at 4,096 | 8.61 | 9.21 |
| 13,399 | 42.6 | — | 74.1 at 16,384 | 4.67 | 7.32 |

Nothing steps per token inside a prefill chunk any more. The gain from ENGN-04
is small (about 2 tok/s) because the per-token DeltaNet dispatches were a
small share at these lengths once attention was batched; prefill now sits
on the matmul tile's ceiling of 42–57 tok/s (`make bench-matmul`), which
is ENGN-05. The fall at 13K (53 → 43) is `nu_attention_chunk` reading a 13K-row
F32 cache for each of the six query heads of a KV group (KERN-07, KERN-08). Decode
at 13K context is 4.67 against the reference's 7.32: the three-dispatch
decode attention over an F32 cache is the long-context lever KERN-07 and KERN-08
were reordered before ENGN-07 to pull.

ENGN-05 matmul tile ceiling (2026-09-09), same hardware, artifact, and
methodology (`bench --raw --prompt-file … --max-tokens 32 --ctx-size 4096
--repeat 2`; the 22-token row is `make bench` with `--repeat 3 --warmup 1`):

| Prompt tokens | Prefill tok/s | ENGN-04 | Reference (llama.cpp) | Decode after the prompt | First token ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 22 (`make bench`) | 39.1 | 34.9 | — | 10.30 | 563 |
| 545 | 83.8 | 52.6 | 89.2 at 512 | 10.15 | 6,507 |
| 3,657 | 81.7 | 53.0 | 89.3 at 4,096 | 8.63 | 44,783 |

The batched matmul now runs through per-encoding tiles with half operands
and F32 accumulation in 64×64 tiles (`make bench-matmul`: 4.9–5.2 TFLOP/s,
was 2.3–3.1; per-lever table in
[metal-backend.md § Kernels](metal-backend.md#kernels)). Prefill is within
6–9 % of the reference at 512 and 4K; decode is untouched. The 16K row was
not rerun (ENGN-05 does not change attention). Chunked vs stepped logits moved
from 2.8e-5 to 2.4–2.6e-3 max abs with identical argmax (bound 2e-2).

KERN-07 F16 KV cache (2026-09-10), same hardware, artifact, and methodology,
both cache precisions back to back with every flag explicit (`bench --raw
--prompt-file … --max-tokens 32 --ctx-size 4096 --repeat 2 --kv f16|f32`;
the 16K row `--ctx-size 16384 --repeat 1`, one warmup and one run; the
22-token row is `make bench` with `ARGS='--kv …'`, 64 output tokens,
three runs). Session bytes are the `bench` header's figure:

| Prompt tokens | Context | KV | Session | Prefill tok/s | Decode after the prompt | First token ms | Reference (llama.cpp, F16 KV) |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 22 | 2,048 | f16 | 278 MiB | 39.7 | 10.59 | 554 | 9.66 decode at 512 |
| 22 | 2,048 | f32 | 406 MiB | 39.7 | 10.62 | 554 | |
| 3,657 | 4,096 | f16 | 406 MiB | 83.1 | 8.88 | 44,020 | 89.3 / 9.21 at 4,096 |
| 3,657 | 4,096 | f32 | 662 MiB | 82.0 | 8.55 | 44,611 | |
| 13,399 | 16,384 | f16 | 1,174 MiB | 67.3 | 5.59 | 198,978 | 74.1 / 7.32 at 16,384 |
| 13,399 | 16,384 | f32 | 2,198 MiB | 61.3 | 4.58 | 218,720 | |

Halving the cache is worth +22 % decode and +10 % prefill after the 13K
prompt (5.59 vs 4.58, 67.3 vs 61.3), +4 % / +1 % after 3.6K, and nothing
at 2K, where attention is a small share of the token. Both precisions
were measured in the same session; the ENGN-04 row's 42.6 / 4.67 at 13K was
F32 before ENGN-05, so the F32 row here (61.3 / 4.58) is the honest
same-day baseline. The remaining gap to the reference at 16K is the
three-dispatch decode attention over a `[heads][visible]` score buffer,
which KERN-08 replaces with one split-K pass. Greedy tokens on the 22-token
and 545-token pinned prompts are identical in both precisions
([metal-backend.md § F16 KV cache](metal-backend.md#f16-kv-cache-kern-07)).

KERN-08 flash-decoding attention (2026-09-10), same hardware, artifact, and
methodology, F16 cache. The 32K row is a 30,650-token prefix of the docs
(`.zig-cache/prompts/p32k.txt`, 128,000 bytes of `docs/*.md` and
`docs/reference/*.md` concatenated in the order architecture, spec,
llm-guide, metal-backend, engineering-log, cpu-reference, development,
agent-spec, cut at a UTF-8 boundary) at `--ctx-size 32768 --max-tokens 64
--repeat 1 --warmup 0`: no warmup, so its prefill includes the process's
first-dispatch costs and the decode rate covers 63 steps. The three-pass
row was taken with the KERN-07 binary the same day, alone on the GPU:

| Prompt tokens | Context | Decode kernels | Prefill tok/s | Decode after the prompt | Decode ms/step | First token ms |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 30,650 | 32,768 | three-pass (KERN-07) | 49.2 | 2.65 | 377 | 622,694 |
| 30,650 | 32,768 | flash decoding | 51.0 | 8.09 | 124 ms | 600,510 |
| 13,399 | 16,384 | three-pass (KERN-07) | 67.3 | 5.59 | 179 | 198,978 |
| 13,399 | 16,384 | flash decoding | 66.5 | 8.92 | 112 | 201,450 |
| 22 (`make bench`) | 2,048 | three-pass (KERN-07) | 39.7 | 10.59 | 94 | 554 |
| 22 (`make bench`) | 2,048 | flash decoding | 40.0 | 10.63 | 94 | 550 |

Decode at 32K is 3.05× faster than the three-pass kernels and above the
reference's 6.71 at that context; at 16K it is 8.92 against the
reference's 7.32. The 2K rows are taken after a cool-down: run straight
after the two long rows, the same `make bench` gave 33.1 / 9.15 and then
35.3 / 9.53 (prefill / decode), a 10–15 % dip on the prefill path this
unit does not touch, which is the machine's sustained-load behaviour and
not the kernel. Record the preceding load with any short-workload row.

MODL-01 (2026-09-08), the official Qwen3.8 sampling profiles measured with
`make bench` and explicit flags (the same 22-token workload, 64 generated,
context 2048, one warmup, three measured runs; decode ms/step is
`decode_milliseconds / 63`), same hardware, artifact, and build:

| Sampling | Path | Decode tok/s | Decode ms/step | vs greedy | Fallbacks / run |
| --- | --- | ---: | ---: | ---: | ---: |
| greedy (this session) | GPU argmax | 10.60 | 94.4 | — | — |
| thinking profile: T 1.0, top-k 20, top-p 0.95 | GPU top-k | 10.57 | 94.7 | +0.3 ms | 0 |
| instruct profile: T 0.7, top-k 20, top-p 0.8, presence 1.5 | full readback + penalties + CPU sort | 8.69 | 115.1 | +20.7 ms | — |

The instruct profile pays the pre-KERN-06 cost on every token because its
presence penalty must be applied to all 248,320 logits before the sort; a
GPU penalty kernel is a measured follow-up ([roadmap](../roadmap.md)). The
thinking profile stays on the GPU path. Prefill (34.7 tok/s) and first
token (634 ms greedy, 654 ms instruct) are unchanged in definition; the
greedy row was re-measured in the same session as the profiles (10.60 vs
10.77 in the KERN-06 session, the usual short-prompt spread). Token equivalence
between the GPU path and the reference path (`generate --json` with and
without `--logits`) was re-checked with `min_p` in play: identical tokens
for `--top-k 0 --min-p 0.05` (39 tokens to EOS), `--top-k 0 --top-p 1
--min-p 0.02 --temperature 1.5` (48 tokens, decided entirely through the
`min_p` prefix, 0 fallbacks), `--top-k 20 --min-p 0.1` (48), and the
instruct profile (10 tokens to EOS, both runs on the full path).

Incremental KERN-04 experiments on that same workload:

| Version | Mean prefill tok/s | Mean decode tok/s |
| --- | ---: | ---: |
| DeltaNet projections only | 10.77 | 10.37 |
| Plus sequential gate/up per SIMD group (rejected) | 10.67 | 10.24 |
| Plus gate/up split across SIMD groups (kept) | 10.95 | 10.49 |
| Plus attention projections (final) | 11.02 | 10.63 |

## Gemma 4 12B: first look (MODL-06, 2026-09-11)

The second family's rates on the same machine, method, and build as
above (greedy, context 2,048, three measured runs after one warmup), not
an acceptance record: the reference harness run on the same token arrays
is MODL-07's. Both cache precisions, since Gemma's F16 tolerance is its own
([gemma4.md § Metal plan](gemma4.md#metal-plan-modl-06-2026-09-11)).

| Workload | Prefill tok/s | Decode tok/s | First token |
| --- | ---: | ---: | ---: |
| 9-token raw prompt (the `make bench` text through Gemma's tokenizer), 64 out, `--kv f16` | 36.8 | 21.2 | 245 ms |
| same, `--kv f32` | 36.8 | 21.2 | 245 ms |
| `tests/fixtures/run-2026-09-06/prompt-512.json` as an opaque id array, 128 out, `--kv f16` | 199.7 | 20.25 | 2,564 ms |
| same, `--kv f32` | 199.0 | 20.20 | 2,572 ms |
| `llama-bench` `7620399`, same file, `-p 512 -n 128 -ngl 99 -fa 1 -ctk f16 -ctv f16 -r 3` | 219.9 ± 2.4 | 24.9 ± 0.15 | — |

Weights are 7.35 GB, so 20.2 tok/s reads 148 GB/s against the reference's
183; the 512-token prefill is at 91 % of the reference. The K-quant file
has 43 of 48 layers' projections in Q4_K, `attn_v` mostly Q6_K, and the
Q5_K embedding as the tied output head (1.0 GB read per token for the
head alone). Nothing here was tuned for Gemma; the per-kernel profile and
the levers are a follow-up after MODL-07.

## Gemma 4 12B acceptance record (MODL-07, 2026-09-12)

The v0.1 acceptance workload on the second family, each side on its own
token arrays since the tokenizers differ: the reference harness ran on
the Gemma file with `--family gemma4`
(`tests/fixtures/run-2026-09-12-gemma4/`, summarized in
[reference-2026-09-12-gemma4.json](../benchmarks/reference-2026-09-12-gemma4.json):
512, 4,096, 16,384, and 32,639 prompt tokens, three measured repetitions
at every length after one warmup, 128 output tokens, greedy, context
32,768, F16 cache, `llama-server 7620399` with the recipe's flags on Metal),
and `make baseline-gemma4` fed those arrays through `bench --prompt-tokens`
([nuclis-2026-09-12-gemma4.json](../benchmarks/nuclis-2026-09-12-gemma4.json);
since 2026-09-12 that target is `make baseline-gemma4`, the catalogue's
file having moved to the QAT checkpoint below).
Every array starts with `<bos>` and the Gemma template's user turn; the
script verified that `nuclis tokenize` reproduces the reference's corpus
tokens through every cut (32,619 tokens) and that all four text
renderings tokenize to exactly their array's count (Qwen's 512 array does
not). Apple M4 Pro (12 CPU, 16 GPU cores), 48 GiB, macOS 26.6.2, AC power,
Zig 0.16.0, ReleaseSafe, `nuclis 0.1.0-dev` from the MODL-07 tree, artifact
SHA-256 `90fd944d…` (the K-quant file), `--kv f16 --ctx-size 32768
--max-tokens 128`, one 29-minute sequence 512 → 32,639 with nothing else on
the GPU. Mean ± sample standard deviation over the measured runs (one at
32,639); the reference columns are its warm means over three samples:

| Prompt tokens | Prefill tok/s | Reference | Decode tok/s | Reference | Decode ms/step | First token | Warmup (prefill / decode) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 512 | 192.97 ± 2.99 | 209.85 | 19.82 ± 0.31 | 24.51 | 49.7–51.3 | 2.7 s | 187.1 / 18.93 |
| 4,096 | 155.31 ± 0.11 | 200.10 | 18.91 ± 0.16 | 22.57 | 52.6–53.4 | 26.4 s | 154.5 / 19.01 |
| 16,384 | 103.04 ± 0.13 | 153.54 | 14.65 ± 0.04 | 16.34 | 68.0–68.4 | 159 s | 105.6 / 15.14 |
| 32,639 | 74.04 (one run) | 142.80 | 14.04 (one run) | 16.05 | 71.2 | 441 s | 73.7 / 13.20 |

Every sample stopped with `token_budget` at exactly the array's count and
128 generated tokens. Decode is at 81 % of the reference at 512, 84 % at
4K, 90 % at 16K, and 87 % at 32K (7.35 GB of weights at 19.8 tok/s is
146 GB/s effective; the tied Q5_K embedding is 1.0 GB of that per token).
Prefill is at 92 % at 512 and falls with length (78 % at 4K, 67 % at 16K,
52 % at 32K): the reference's own rate falls too (210 → 143), but
nuclis's falls faster, as Qwen's did before ENGN-08 found the chunk attention
kernel latency-bound. Nothing in MODL-06 or MODL-07 was tuned for Gemma; the
per-kernel profile on this file is the follow-up already in the
[roadmap](../roadmap.md#also-deferred), and the 512-wide global layers'
chunk attention (scores recomputed per value split, MODL-06) is the first
thing to profile.

**Memory.** The session block is 11,274,289,152 bytes (10.5 GiB) at
32,768 capacity: every one of the 40 sliding layers is allocated for the
full capacity although it reads only the last 1,024 rows ([gemma4.md §
Metal plan](gemma4.md#metal-plan-modl-06-2026-09-11)); a ring layout for
those layers would cut it to about 0.35 GB and is the roadmap's
session-layout unit. Peak resident set of the `bench` process was
10.75 GiB at every length and its peak footprint 11.74–11.80 GB; the
7.37 GB weight file is memory-mapped and charged to wired memory, so the
headroom is a calculation: 48 GiB − 7.37 GB − 11.8 GB ≈ 29 GiB at 32K
context, the same as Qwen's because the session is five times larger
and the weights 9 GB smaller.

## Gemma 4 12B acceptance record, QAT file (MODL-08, 2026-09-12)

The same workload on the catalogue's file after MODL-08 switched the entry to
Google's quantization-aware-trained checkpoint (every weight matrix
Q4_0, SHA-256 `90fd44e2…`, 6.72 GB). The reference harness ran again on
this file (`tests/fixtures/run-2026-09-12-gemma4-qat/`,
[reference-2026-09-12-gemma4-qat.json](../benchmarks/reference-2026-09-12-gemma4-qat.json);
its token arrays are byte-identical to the K-quant run's, since the two
files share vocabulary and template, and `nuclis tokenize` reproduces
every cut as before), and `make baseline-gemma4` fed them through `bench
--prompt-tokens` ([nuclis-2026-09-12-gemma4-qat.json](../benchmarks/nuclis-2026-09-12-gemma4-qat.json)).
Same hardware, OS, and build mode as the record above; `nuclis 0.1.0-dev`
from the MODL-08 tree; `--kv f16 --ctx-size
32768 --max-tokens 128`; one 30-minute sequence 512 → 32,639. The
reference ran on AC power; the nuclis process reports **battery power**
(the machine was unplugged between the two), which the 512-token spread
below may reflect. Mean ± sample standard deviation over three measured
runs (one at 32,639); the reference columns are its warm means:

| Prompt tokens | Prefill tok/s | Reference | Decode tok/s | Reference | First token |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 179.45 ± 19.32 | 224.46 ± 0.70 | 23.96 ± 0.97 | 27.69 ± 0.14 | 2.9 s |
| 4,096 | 143.25 ± 1.90 | 216.82 ± 2.10 | 20.71 ± 1.34 | 25.69 ± 0.31 | 28.6 s |
| 16,384 | 98.67 ± 1.06 | 158.78 ± 23.31 | 17.87 ± 0.56 | 22.10 ± 2.79 | 166 s |
| 32,639 | 73.04 (one run) | 143.72 ± 3.95 | 16.24 (one run) | 20.94 ± 1.46 | 447 s |

Every sample stopped with `token_budget` at the array's count and 128
generated tokens. Against the K-quant record above, decode rose at every
length (19.82 → 23.96 at 512, 14.04 → 16.24 at 32K: 6.7 GB of Q4_0 read
through the fastest kernel of the set, [metal-backend.md § Specialized
matvec](metal-backend.md#specialized-matvec)) and stands at 87 % of the
reference at 512, 81 % at 4K and 16K, 78 % at 32K. Prefill fell slightly
(192.97 → 179.45 at 512 inside that row's spread, 155 → 143 at 4K,
103 → 99 at 16K, 74 → 73 at 32K) while the reference's rose on the same
file (210 → 224), so nuclis is at 80 % of the reference at 512 and 51 %
at 32K: the Q4_0 matmul tile decodes two 16-value segments per 32-value
block with two 8-byte loads and a scale read each, where the K-quant
tiles amortize their loads over 256 values, and the long-context fall is
the chunk attention latency already named for the K-quant file. Both are
the per-kernel profile's first targets ([roadmap](../roadmap.md#also-deferred)).
The reference's own 16K row has a 15 % spread this time (132–175 tok/s),
so ratios at that length are indicative.

**Memory.** Session block 11,274,289,152 bytes (10.5 GiB, the same
full-capacity sliding caches), peak resident set 11.54 GiB, peak footprint
11.77–11.81 GB at every length; the weight file is 0.65 GB smaller than
the K-quant one.

## Per-kernel profile

`nuclis bench --profile` (Metal only) adds a table of GPU time per kernel and
matrix shape, per step, from GPU timestamps (`MTLCounterSampleBuffer`). Apple
GPUs sample timestamps only at encoder boundaries, so in profile mode every
dispatch is recorded in its own compute encoder. Consequences, all reported
rather than hidden:

- The profiled run's own prefill/decode rates are **not comparable** with an
  unprofiled run (the header says so). Use plain `bench` for speed claims.
- `attributed` is the sum of timed kernel durations; the command-buffer GPU
  time exceeds it by the encoder boundaries. On this model the boundaries cost
  ~4 ms per token and the kernels themselves run ~4 ms slower in total than
  unprofiled, so profile-mode numbers are ~8 % pessimistic overall.
- Timestamps are nanoseconds on the same timeline as `GPUStartTime`; this was
  measured, not assumed (the first encoder's start stamp equals the command
  buffer's start to the microsecond).
- Up to 4,096 dispatches per command buffer are timed (a Qwen token records
  1,236); the rest are counted as `unsampled` and excluded from the table.
- Warm-up runs are profiled and discarded; the table covers measured steps only.
- Matrix kernels are keyed by encoding and shape, so one kernel's time splits
  by tensor; `GB/s` is weight bytes over kernel time. Elementwise kernels
  aggregate by name.

`make bench-profile` runs it with the `bench` workload. Record (2026-09-07,
Apple M4 Pro, ReleaseSafe, 255 measured steps):
[benchmarks/nuclis-profile-2026-09-07.json](../benchmarks/nuclis-profile-2026-09-07.json).
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

KERN-03 profile (2026-09-08), same workload and 255 measured steps:
[raw report](../benchmarks/nuclis-profile-2026-09-08.json). Attributed GPU time
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

KERN-04 profile (2026-09-08), 255 measured steps:
[raw report](../benchmarks/nuclis-profile-c12-2026-09-08.json). Dispatches fell
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

### Gemma 4 12B QAT decode, first per-kernel profile (2026-09-12)

`make bench-profile MODEL=<qat file> ARGS="--kv f16"` on the catalogue's
Q4_0 file, 22-token prompt, 2,048 context, 192 measured steps: 882
dispatches per step, 40.0 ms attributed of 44.3 ms command-buffer time
(profile mode, ~8 % pessimistic). Removing the amortized prefill
(`matmul_q4_0_32`, 3.5 ms/step, which is this run's 22-token prefill
spread over the steps), a decode step is 36.5 ms of GPU time:

| Part | ms/step | Share | Note |
| --- | ---: | ---: | --- |
| Weight matvecs (7 shapes) | 30.9 | 85 % | 191–224 GB/s, 217 GB/s weighted |
| RMS norms | 2.7 | 7 % | 337 dispatches of ~8 µs: launch latency, not bytes |
| Attention (decode, wide decode, merge) | 2.1 | 6 % | 2K context, so latency-bound |
| RoPE, elementwise epilogues, sampling | 0.7 | 2 % | |

The matvec kernels run at the same rate inside the model as alone on
fixture matrices (204–230 GB/s, § Kernel micro-benchmark), so nothing is
lost to the token schedule: decode is weight reading plus about 5.5 ms of
everything else. Whole-step effective bandwidth is 184 GB/s against the
reference's 186 GB/s at 27.69 tok/s on the same file, while nuclis
measures 23.96 tok/s (41.7 ms/step) in the acceptance run, so the 5.6 ms
gap is *not* in the Q4_0 decode kernel. The two candidates it leaves are
the 337 norm launches (Gemma has six norms per layer; the same 2.7 ms
costs Qwen's 101 ms step 2.7 % and this 41.7 ms step 6.5 %, which is why
the Qwen tuning does not transfer for free to a smaller model) and the
matvecs' own 75–82 % of the published 273 GB/s, whose limiter KERN-05
measured as per-block instruction work. The 15,360-wide fused gate/up
projection alone is 38 % of the step.

Prefill in the same run: the 32-token matmul tile runs at 25–29 GB/s on a
22-token prompt, the known small-M case (ENGN-05; one token tile leaves too
few threadgroups), not a Q4_0 property.

Follow-ups from this profile are in the
[roadmap](../roadmap.md#also-deferred); none is scheduled before TERM-01.

## Kernel micro-benchmark

`make bench-kernels` runs `metal-check --matvec-bench`: each matvec kernel
alone on fixture-tiled matrices (38–1,040 MB, larger than any cache) with
eight back-to-back dispatches per command buffer, reporting GB/s of weight bytes
from GPU busy time, best and mean of five command buffers. Two methodology
points learned while building it: isolated single dispatches measure the GPU's
clock ramp from idle rather than the kernel (means fell to a third of the best),
and allocating a fresh multi-GB buffer per case made timings swing with driver
residency work, so one weight buffer is reused. KERN-05 added the model's actual
down-projection shape (5,120×17,408, 320 threadgroups) and batches 64
dispatches per command buffer for matrices under 8,192 rows, since eight of
those are too short to hold the clock. Its numbers move a few percent between
runs; rank variants with back-to-back off/on/off/on rounds, as the KERN-05 record
does, not from one run. It needs a Metal device but no model, and it isolates
a kernel from the token schedule: use it to rank kernel variants, and `bench`
to claim decode speed.

`make bench-matmul` (`metal-check --matmul-bench`) measures the batched
prefill matmul alone on the two FFN shapes for a 256-token chunk: GPU ms per
dispatch, GFLOP/s of F32 multiply-adds, and the prefill tok/s the 54
GFLOP/token model would reach if that were its only cost — a ceiling for
ENGN-02, not a prediction. Results are in
[metal-backend.md § Kernels](metal-backend.md#kernels).

`make bench-experts` (`metal-check --experts-bench`) measures the gathered
expert kernels on the Gemma 4 26B-A4B shape (128 experts, 8 selected;
gate-up 1,408 × 2,816 and down 2,816 × 704 per expert, Q4_0): GB/s of the
selected experts' bytes for the gathered gate-up, the gathered down, and
the whole decode chain (gate-up, gelu rows, down, combine), beside a dense
matvec over the same byte count — the rate a gathered kernel can at most
reach. Same methodology as `bench-kernels` (64 dispatches per command
buffer, best and mean of five). Results and the reading of them are in
[metal-backend.md § Gathered expert kernels](metal-backend.md#gathered-expert-kernels-kern-09).
`bench-kernels` takes an optional encoding name (`ARGS=Q4_0`) to measure
one kernel alone; a full run heats the GPU progressively (the encodings
measured last come out 15–30 % below their rested rates), so re-measure a
changed kernel alone on a rested machine before comparing with a record.

On the CPU reference backend, each step takes roughly 18–19 s; `bench` runs
there only with small budgets and is useful for definitions, not for speed.
