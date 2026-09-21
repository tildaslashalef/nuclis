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
| `speculative`, `draft_length` | With the switch off the drafter is not loaded at all — no weights, no scratch, no draft cache — which is the true baseline. With `--speculative on` (or the entry's `generation.speculative`) the family's source loads and every run is an off/on pair on that one loaded model, so the pair's off sample is the loaded-but-off case and the on sample carries the draft length. The catalogue entry carries the measured verdict (ENGN-17). |
| `speculative_steps`, `accepted_per_step` | Verify batches the run made and the mean accepted drafts per batch; tokens per batch is `(generated_tokens − 1) / speculative_steps`. |
| `verify_milliseconds`, `accept_milliseconds`, `recover_milliseconds` | The run's time in the model's verify batches, in the host acceptance decision (the sampled path's per-row distributions), and in recovery (rewind and replay, including the final recover to the emitted count when a batch crosses the budget); divide by `speculative_steps` for the per-batch cost. |
| `checkpoint_milliseconds`, `recover_rewind_milliseconds`, `recover_replay_milliseconds` | The verify batch's checkpoint copy, and recovery split into the copy back to the checkpoint and the forward over the accepted prefix; they sum to `recover_milliseconds` less clock overhead. |
| `recover_by_length` | Recovery calls per accepted prefix length (1..`max_draft_length + 1`) with `calls`, `rewind_milliseconds`, and `replay_milliseconds`; full acceptance recovers nothing and contributes calls with zero time. |
| `propose_milliseconds`, `commit_milliseconds` | The run's time in the drafter's proposal of the batch's drafts and in its commit of the accepted prefix; divide by `speculative_steps` for the per-batch cost. The prompt commit is inside `prefill_milliseconds`. |
| `mean_speculative_decode_tokens_per_second`, `decode_speedup` | The on samples' decode mean and its ratio to the off samples' mean, the pair's comparison. |

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
KERN-16 in [TODO.md](../../TODO.md)). The 16K row is lower than the 66.5 / 8.92
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
GPU penalty kernel is a measured follow-up (KERN-13 in [TODO.md](../../TODO.md)). The
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

KERN-11 (2026-09-19) short-chunk prefill matmul, same hardware, artifact,
build, and standard `make bench` workload (22 prompt tokens, 64 generated,
context 2048, one warmup, three measured runs). The 22-token chunk now runs
the 16-row × 8-token split tile (three token tiles) instead of one 32×32 tile;
decode is untouched. Before is the same build with `small_chunk_tokens = 8`,
which selects the 32×32 tile for this prompt:

| Version | Prefill tok/s | First token ms | Decode tok/s |
| --- | ---: | ---: | ---: |
| Before (32×32 tile) | 38.78 | 567.3 | 10.22 |
| After (16×8 split tile, threshold 24) | 43.01 | 511.5 | 10.20 |

Prefill +10.9 %, first token −9.8 %, decode within noise. The per-shape
kernel rates and the experiments behind the tile are in
[metal-backend.md § Small-chunk tile](metal-backend.md#small-chunk-tile-kern-11-2026-09-19).

KERN-12 (2026-09-20) multi-row matvec, same machine and build: the kernel
reads the weight bytes once per batch instead of once per token and was meant
to serve the verify, replay, and commit batches (2–8 rows). On the two FFN
shapes (`make bench-matvec-rows`, Apple M4 Pro, Zig 0.16.0, ReleaseSafe) it
beats the 16×8 tile at 2 rows (143–182 GB/s against 88–116) and for
Q6_K/IQ4_XS at 3, but at 5 rows streams 49–67 and at 8 rows 20–33, below the
tile, so it fails the unit's ≥ 150 / ≥ 120 GB/s targets. `Backend.matmul`
routes only 2-row batches of the specialized encodings to it
(`small_batch_rows = 2`); the verify (5 rows) is unchanged. Full table, method,
and the register/FMA analysis are in
[metal-backend.md § Multi-row matvec](metal-backend.md#multi-row-matvec-kern-12-2026-09-20-closed-below-its-target).
A repeat-1 spot run at the close revision (512 prose, draft 4, F16 KV, ctx
32768) measured verify 232–288 ms and recover 170–217 ms per batch, within the
speculative record's range for verify. Recovery here is aggregate milliseconds
per speculative step, not latency conditioned on two-row replay. ENGN-14 now
refreshes the record immediately after the recovery change, including recovery
calls/timings by accepted length; ENGN-17 retains the final defaults record.

The corrected two-row sweep (2026-09-20, `27303ed` plus REPO-08) forces the
specialized tile control and includes all four head encodings. All specialized
cases still win: FFN 127.7–181.9 GB/s vs tile 87.7–116.2, head 146.8–188.0
vs 89.2–114.1. Routing stays at two tokens. Method and individual rates:
[corrected control](metal-backend.md#corrected-two-row-control-repo-08-2026-09-20).

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
per-kernel profile on this file is the follow-up this record points at,
and the 512-wide global layers' chunk attention (scores recomputed per
value split, MODL-06) is the first thing to profile.

**Memory.** The session block is 11,274,289,152 bytes (10.5 GiB) at
32,768 capacity: every one of the 40 sliding layers is allocated for the
full capacity although it reads only the last 1,024 rows ([gemma4.md §
Metal plan](gemma4.md#metal-plan-modl-06-2026-09-11)); a ring layout for
those layers would cut it to about 0.35 GB and is a session-layout change
of its own. Peak resident set of the `bench` process was
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
the per-kernel profile's first targets.
The reference's own 16K row has a 15 % spread this time (132–175 tok/s),
so ratios at that length are indicative.

**Memory.** Session block 11,274,289,152 bytes (10.5 GiB, the same
full-capacity sliding caches), peak resident set 11.54 GiB, peak footprint
11.77–11.81 GB at every length; the weight file is 0.65 GB smaller than
the K-quant one.

## Gemma 4 26B-A4B acceptance record (MODL-10, 2026-09-18)

The v0.1 acceptance workload on the catalogue's mixture of experts,
`gemma-4-26b-a4b` (`gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf`, SHA-256
`a7c5bc71…`, 14.25 GB, every matrix Q4_0, 128 experts of which 8 per
token). The reference harness ran on this file
(`tests/fixtures/run-2026-09-18-gemma4-26b-a4b/`,
[reference-2026-09-18-gemma4-26b-a4b.json](../benchmarks/reference-2026-09-18-gemma4-26b-a4b.json):
`llama-server 7620399` with the recipe's flags at 32,768 context, F16
cache, three measured repetitions at every length after one warmup; its
token arrays are byte-identical to the two 12B runs', since the three
files share vocabulary and template, and `nuclis tokenize` reproduces the
corpus through every cut and every text rendering to its array's count),
and `make baseline-gemma4-26b-a4b` fed the arrays through `bench
--prompt-tokens` ([nuclis-2026-09-18-gemma4-26b-a4b.json](../benchmarks/nuclis-2026-09-18-gemma4-26b-a4b.json)).
Apple M4 Pro (12 CPU, 16 GPU cores), 48 GiB, macOS 26.6.2, AC power on
both sides, Zig 0.16.0, ReleaseSafe, `nuclis 0.2.0-dev` from the MODL-10
tree (`56ef7d4` plus this unit's Makefile and documents), `--kv f16
--ctx-size 32768 --max-tokens 128`, the family's prefill chunk of 512,
one 15-minute sequence 512 → 32,639 with nothing else on the GPU. Mean ±
sample standard deviation over three measured runs (one at 32,639); the
reference columns are its warm means over three samples:

| Prompt tokens | Prefill tok/s | Reference | Decode tok/s | Reference | Decode ms/step | First token | Warmup (prefill / decode) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 512 | 500.42 ± 4.65 | 580.76 ± 5.35 | 55.29 ± 0.05 | 68.02 ± 0.12 | 18.1 | 1.0 s | 416.9 / 55.29 |
| 4,096 | 346.17 ± 1.29 | 548.41 ± 4.12 | 49.44 ± 0.64 | 60.82 ± 0.28 | 20.0–20.5 | 11.8 s | 341.8 / 49.16 |
| 16,384 | 206.44 ± 1.21 | 459.46 ± 3.28 | 39.55 ± 0.18 | 50.67 ± 2.32 | 25.2–25.4 | 79.4 s | 207.8 / 39.92 |
| 32,639 | 134.74 (one run) | 343.38 ± 11.81 | 31.30 (one run) | 44.09 ± 0.10 | 31.9 | 242 s | 135.8 / 31.47 |

Every sample stopped with `token_budget` at exactly the array's count and
128 generated tokens, on both sides. The half-tile rounding that the
generation check measures on random tokens (3.5e-1 relative RMS through
the routed layers) does not show here: the workload is greedy on the
reference's own arrays and every run met its budget with the same stop,
which is what the record can say about it; per-token agreement with the
reference on real prompts remains the trace comparison's job, not this
table's.

**Decode** is at 81 % of the reference at 512 and 4K, 78 % at 16K, and
71 % at 32K. The step reads about 2.1 GB of Q4_0 weights (eight experts
of 704 plus the 2,112-wide shared FFN on every layer, the attention
projections, and the tied 262,144-row head; a sum over the tensor shapes),
so 18.1 ms is 118 GB/s effective against the reference's 146 at 14.7 ms.
The gap widens with context faster than the reference's: from 512 to 32K
nuclis adds 13.8 ms per step and the reference 8.0, for the same five
global layers reading two KV heads over the whole context (0.67 GB per
step at 32K in F16, about 3.3 ms of bytes), so the flash-decoding kernel
on the two-KV-head, 16-query-head geometry is well above its byte cost at
long context.

**Prefill** is at 86 % of the reference at 512 and falls to 63 % at 4K,
45 % at 16K, and 39 % at 32K, the steepest fall of the three Gemma
records (the 12B QAT's is 80 % → 51 %): the reference's rate falls 581 →
343 over the same lengths, nuclis's 500 → 135. Two things compound here
beyond the chunk attention latency already named for the 12B: the
gathered expert tiles are dispatched per 512-token chunk (64 chunks at
32K, each with its own routing, list build, and two gathered matmuls), and
the global layers' chunk attention with two KV heads scores the whole
prefix per chunk.

**Per-kernel profile** (`make bench-profile MODEL=<26b-a4b file>
ARGS="--kv f16"`, 22-token prompt, 2,048 context, 192 measured steps,
2026-09-18): 915 dispatches per step, 23.8 ms attributed of 28.7 ms
command-buffer time. Profile mode costs this model far more than the 12B's
8 %: the same prompt decodes at 57.9 tok/s unprofiled and 33.4 profiled,
so the shares below are indicative and the absolute times pessimistic.
Removing the amortized prefill (the gathered and dense matmul tiles, about
3.3 ms/step), a decode step attributes about 20.5 ms:

| Part | ms/step | Share | Note |
| --- | ---: | ---: | --- |
| Expert matvecs (gate-up, down) | 5.6 | 27 % | gate-up 162 GB/s, down **114 GB/s** (704-wide rows) |
| Dense matvecs (head, attention, shared FFN, router) | 8.8 | 43 % | 114–180 GB/s; the tied head 2.3 ms alone at 180 GB/s |
| RMS norms | 2.9 | 14 % | 331 dispatches of ~9 µs: launch latency (eight norms per expert layer) |
| Attention (decode, wide decode, merge) | 1.6 | 8 % | 2K context |
| Routing glue (`route`, `combine_experts`, `gelu_mul_rows`) | 0.8 | 4 % | 90 dispatches |
| RoPE, adds, packing, scales | 0.7 | 3 % | |

Against the 12B's profile, the matvecs hold less of their isolated
bandwidth (the expert down projection and the 2,112-wide shared FFN down
at 114 GB/s against 191–224 for the 12B's shapes: narrow rows leave the
per-row threadgroup lanes idle, the follow-up KERN-09 named) and the
launch-bound part grew from 337 to 421 small dispatches per step. In
order: the expert down kernel's idle lanes, a fused norm-and-scale or a
batched norm launch, and the wide flash-decoding kernel at long context.

**Memory.** Session block 7,381,975,040 bytes (6.87 GiB) at 32,768
capacity (25 sliding layers × 2 × 2,048 and 5 global layers × 2 × 1,024
halves per position, the sliding caches allocated for the full capacity as
on the 12B); peak resident set 7.21 GiB and peak footprint 7.94–8.01 GB at
every length; the 14.25 GB weight file is memory-mapped and charged to
wired memory, so the headroom is 48 GiB − 14.25 GB − 8.0 GB ≈ 29 GB at 32K.
The reference server's resident set between requests was 15.08–15.62 GB.

## Bonsai 2 27B acceptance record (MODL-17, 2026-09-18)

The v0.1 acceptance workload on the catalogue's ternary entry,
`bonsai-2-27b` (`Ternary-Bonsai-2-27B-PQ2_0.gguf`, SHA-256 `3907dc16…`,
7.21 GB, Qwen3.8-27B's architecture with every matrix PQ2_0 in a
Hadamard-rotated basis, [bonsai.md](bonsai.md)). The reference side is the
**PrismML fork** of llama.cpp at `prism-b10687-5d80cff` (commit
`5d80cff0…`, the only decoder of the file; stock llama.cpp rejects it),
its server started with the recipe's flags minus `--lazy-mode` (which its
base does not know) at 32,768 context, F16 cache, and the workload harness
run against it with `--family qwen38 --reference-revision 5d80cff0…`
(`tests/fixtures/run-2026-09-18-bonsai/`,
[reference-2026-09-18-bonsai.json](../benchmarks/reference-2026-09-18-bonsai.json):
three measured repetitions at every length after one warmup, 32,639
included). Its token arrays are byte-identical to the Qwen3.8 run's
(`run-2026-09-06` and the boundary run; the file shares the vocabulary and
its single-user-turn rendering), and `make baseline-bonsai` fed them through
`bench --prompt-tokens`
([nuclis-2026-09-18-bonsai.json](../benchmarks/nuclis-2026-09-18-bonsai.json)).
Apple M4 Pro (12 CPU, 16 GPU cores), 48 GiB, macOS 26.6.2, AC power on both
sides, Zig 0.16.0, ReleaseSafe, `nuclis 0.2.0-dev` from the MODL-17 tree
(`7177307` plus this unit's plan, Makefile, and documents), `--kv f16
--ctx-size 32768 --max-tokens 128`, the Qwen prefill chunk, one 65-minute
sequence 512 → 32,639 with nothing else on the GPU. Mean ± sample standard
deviation over three measured runs (one at 32,639); the reference columns
are the fork's warm means over three samples:

| Prompt tokens | Prefill tok/s | Fork | Decode tok/s | Fork | Decode ms/step | First token | Warmup (prefill / decode) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 512 | 93.69 ± 0.10 | 97.54 ± 0.31 | 13.87 ± 0.00 | 17.05 ± 0.08 | 72.1 | 5.5 s | 94.5 / 13.96 |
| 4,096 | 86.01 ± 0.02 | 99.07 ± 0.26 | 13.26 ± 0.00 | 16.61 ± 0.07 | 75.4 | 47.6 s | 86.8 / 13.34 |
| 16,384 | 66.35 ± 0.01 | 79.69 ± 2.88 | 11.72 ± 0.00 | 13.42 ± 0.18 | 85.3 | 247 s | 66.3 / 11.71 |
| 32,639 | 50.63 (one run) | 84.76 ± 3.90 | 10.20 (one run) | 12.83 ± 0.04 | 98.0 | 645 s | 50.6 / 10.20 |

Every sample stopped with `token_budget` at exactly the array's count and
128 generated tokens, on both sides. Beside the Qwen3.8-27B record on the
same arrays (ENGN-07: 90.45 / 83.70 / 62.70 / 49.55 prefill, 10.62 / 10.20
/ 8.27 / 7.55 decode), the ternary file decodes **31–35 % faster** at every
length and prefills at the same rate — the prefill tiles are compute-bound
and the ternary tiles match the set, so a file less than half the size
gains nothing there.

**Decode** is at 81 % of the fork at 512 and 80 % at 4K, 87 % at 16K, and
80 % at 32K. The step reads 7.2 GB of ternary weights, so 72.1 ms is 100
GB/s effective against the fork's 123 at 58.6 ms; the kernels are at the
set's multiply-rate ceiling
([metal-backend.md § Ternary matvecs and tiles](metal-backend.md#ternary-matvecs-and-tiles-kern-10-2026-09-18):
a ternary byte carries twice the values of a Q4_0 byte, and the same
values per second is half the byte rate), the transform is 2.1 % and the
gather 0.2 % of the step
([§ The rotation on the Qwen plan](metal-backend.md#the-rotation-on-the-qwen-plan-modl-17-2026-09-18)).
The byte count alone projected two to three times the Qwen rate; the
kernel set delivers 1.3×, and the fork's own kernels 1.6× over the
mainline Qwen record (17.05 against 9.66 at 512). Closing the rest is a
different ternary arithmetic in the matvec (packed integer products, or
one decoded weight across several inputs), a kernel experiment of its
own. From 512 to 32K nuclis adds 25.9 ms per step
and the fork 19.9: the same attention-and-recurrent growth as the Qwen
record (KERN-08's flash-decoding kernel over 16 attention layers, the 48
DeltaNet states), on a smaller base.

**Prefill** is at 96 % of the fork at 512, 87 % at 4K, 83 % at 16K, and
60 % at 32K, the Qwen plan's own long-context curve (the fork's rate holds
at 80–99 over the four lengths where nuclis's falls 94 → 51, the chunk
attention latency ENGN-08 measured), unchanged by the encoding.

**Agent check** (Metal, context 8,192, `--think medium`, `-p --json`,
`--model bonsai-2-27b` with the catalogue's forced `.qwen38` profile):
"create greeting.txt with hello world, then read it back" issued
`write_file` then `read_file` in three steps and answered from the
contents, `hello world` on disk; **176** prompt tokens, **210** generated
(three thinking spans of 5.6, 4.9, and 2.0 s), prefill **3.19 s**, decode
**15.1 s** (72 ms per token in the loop, the record's 512-token rate),
`replayed: true`, stop `eos`.

**Memory.** Session block 2,304,376,832 bytes (2.15 GiB) at 32,768 capacity
(16 attention layers × 2 × 1,024 halves per position plus the 48 recurrent
states), peak resident set 2.36 GiB at every length; the 7.21 GB file is
memory-mapped, so the headroom is 48 GiB − 7.2 GB − 2.4 GB ≈ 38 GB at 32K.

**PTQ1_0.** The record above is the PQ2_0 file's. The denser packing of the
same weights (`Ternary-Bonsai-2-27B-PTQ1_0.gguf`, 5.95 GB) measured on the
same plan the same day by `make bench` (22-token prompt, 64 output tokens,
context 2,048, two rounds): 13.05 / 13.01 tok/s decode against PQ2_0's
12.87, prefill 39.3 against 39.1, and it matches the PQ2_0 traces on both
cache precisions ([bonsai.md](bonsai.md#metal-plan-modl-17-2026-09-18)).
Not slower at 1.26 GB less, so the catalogue entry moved to it (decided
2026-09-18); the acceptance workload was not re-run on it.

## Muse Glimmer 30B: first look (MODL-12, 2026-09-19)

The third family's rates on the same machine, method, and build as
above (greedy, context 2,048, three measured runs after one warmup), not
an acceptance record: the reference harness run on the same token arrays
is MODL-13's. The prompt is raw because the profile does not exist yet;
the 512-token array is the first 512 ids of `docs/spec.md`'s opening
6,000 bytes through Muse's tokenizer (the Qwen arrays carry ids above
its vocabulary). Both cache precisions, since Muse's F16 tolerance is
its own ([muse-glimmer.md § Metal plan](muse-glimmer.md#metal-plan-modl-12-2026-09-19),
which also holds the per-kernel profile).

| Workload | Prefill tok/s | Decode tok/s | First token |
| --- | ---: | ---: | ---: |
| 10-token raw prompt (the `make bench` text through Muse's tokenizer), 64 out, `--kv f16` | 18.3 | 9.99 | 547 ms |
| same, `--kv f32` | 18.2 | 9.91 | 549 ms |
| 512-token array, 128 out, `--kv f16` | 93.2 | 9.59 | 5,492 ms |
| same, `--kv f32` | 93.0 | 9.52 | 5,504 ms |
| `llama-bench` `7620399`, same file, `-p 512 -n 128 -ngl 99 -fa 1 -ctk f16 -ctv f16 -r 3` | 101.9 ± 0.1 | 14.08 ± 0.11 | — |

Weights are 15.87 GB, so 9.99 tok/s reads 159 GB/s against the
reference's 223 (71 %); the 512-token prefill is at 91 %. The Qwen
`make bench` the same day: 39.75 / 10.44 tok/s, unchanged.

## Muse Glimmer 30B acceptance record (MODL-13, 2026-09-19)

The v0.1 acceptance workload on the third family, each side on its own
token arrays since the tokenizers differ: the reference harness ran on
the Muse file with `--family muse-glimmer`
(`tests/fixtures/run-2026-09-19-muse-glimmer/`, summarized in
[reference-2026-09-19-muse-glimmer.json](../benchmarks/reference-2026-09-19-muse-glimmer.json):
512, 4,096, 16,384, and 32,639 prompt tokens, three measured repetitions
at every length after one warmup, 128 output tokens, greedy, context
32,768, F16 cache, `llama-server 7620399` with the recipe's flags on Metal;
the template's default `Reasoning strength: high.` system turn, dated
2026-09-19 by the server's clock, heads every array), and
`make baseline-muse-glimmer` fed those arrays through `bench --prompt-tokens`
([nuclis-2026-09-19-muse-glimmer.json](../benchmarks/nuclis-2026-09-19-muse-glimmer.json)).
Every array starts with `<|begin_of_text|>` and that system turn; the
script verified that `nuclis tokenize` reproduces the reference's corpus
tokens through every cut (32,577 tokens) and that all four text renderings
tokenize to exactly their array's count. The first attempt refused the
191 KB corpus with `WorkLimitExceeded`: the encoder's special-token scan
charged every one of Muse's 2,048 reserved markers per byte, exhausting its
1 GiB budget at 18 KB of text; the scan now searches each marker's first
byte ([tokenizer.md](tokenizer.md)), and the run below is on that fix.
Apple M4 Pro (12 CPU, 16 GPU cores), 48 GiB, macOS 26.6.2, AC power,
Zig 0.16.0, ReleaseSafe, `nuclis 0.2.0-dev` at `bbac7ac`, artifact
SHA-256 `82bece30…`, `--kv f16 --ctx-size 32768 --max-tokens 128`, one
42-minute sequence 512 → 32,639 with nothing else on the GPU. Mean ± sample
standard deviation over the measured runs (one at 32,639); the reference
columns are its warm means over three samples:

| Prompt tokens | Prefill tok/s | Reference | Decode tok/s | Reference | Decode ms/step | First token | Warmup (prefill / decode) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 512 | 93.49 ± 0.09 | 95.28 | 9.60 ± 0.00 | 13.69 | 103.3–103.4 | 5.5 s | 91.1 / 9.61 |
| 4,096 | 80.51 ± 2.65 | 93.01 | 8.26 ± 0.22 | 12.14 | 116.5–122.3 | 49–52 s | 83.2 / 9.14 |
| 16,384 | 67.83 ± 0.17 | 80.96 | 7.19 ± 0.03 | 10.07 | 137.5–138.6 | 241–242 s | 68.8 / 7.30 |
| 32,639 | 59.20 (one run) | 76.86 | 6.62 (one run) | 9.98 | 149.9 | 551 s | 59.2 / 6.61 |

Every sample stopped with `token_budget` at exactly the array's count and
128 generated tokens. Decode is at 70 % of the reference at 512, 68 % at
4K, 71 % at 16K, and 66 % at 32K (15.87 GB of weights at 9.60 tok/s is
152 GB/s effective against the reference's 217); prefill is at 98 % at
512 and falls with length (87 % at 4K, 84 % at 16K, 77 % at 32K), the
reference's own rate falling too (95 → 77). The gap is the widest of the
four families and is spread over the large Q4_K matvecs
([muse-glimmer.md § Metal plan](muse-glimmer.md#metal-plan-modl-12-2026-09-19)
has the per-kernel profile); the 4K row's decode drifted from 9.14 tok/s
on its warmup to 8.12 on its third sample within four minutes, which no
other length showed and which was not investigated (thermal is the
obvious suspect). Both are the performance theme's material: KERN-15
measured the matvec half of the gap and closed negative
([§ Split-K matvec sweep](#split-k-matvec-sweep-kern-15-2026-09-21)),
not this unit's.

**Memory.** The session block is 1,744,830,464 bytes (1.63 GiB) at
32,768 capacity: 52 layers × 2 × 256 halves per position, every sliding
layer allocated for the full capacity although it reads only the last
2,048 rows (a ring layout would be a session-layout change of its own). Peak
resident set of the `bench` process was 1.80 GiB at every length and its
peak footprint 2.17–2.22 GB; the 15.88 GB weight file is memory-mapped and
charged to wired memory, so the headroom is a calculation: 48 GiB −
15.9 GB − 2.2 GB ≈ 30 GiB at 32K context, with the vision projector
(1.4 GB) and the DFlash drafter (1.6 GB) still to come.

## Speculative decoding record (ENGN-12, 2026-09-20)

The first measurement of speculative generation on Qwen3.8-27B with its
embedded draft head, taken to decide the catalogue entry's default and to
size the per-batch costs the performance units attack
([speculative-decoding.md](speculative-decoding.md), the plan in
[TODO.md](../../TODO.md)). `make speculative-record`
(`scripts/nuclis-speculative.py`; the reports are under
[benchmarks/speculative-2026-09-20/](../benchmarks/speculative-2026-09-20/))
ran twelve configurations, each as off/on pairs on one loaded model (the
drafter loaded in both): the reference corpus arrays at 512 and 4,096
tokens and the fixed code prompt `Write a Zig function that reverses a
string.` (`--raw`, 10 tokens), greedy and with the instruct profile's
sampling (`--temperature 0.7 --top-p 0.8 --top-k 20 --presence-penalty
1.5`, seed 0), draft lengths 2, 4, 7, 128 output tokens, context 32,768,
F16 KV, one warmup, three measured repetitions (two at 4K). Apple M4 Pro
(12 CPU, 16 GPU cores), 48 GiB, macOS 26.6.2 (25G83), AC power, Zig
0.16.0, ReleaseSafe, `nuclis 0.2.0-dev` at `3d5cb94`, artifact SHA-256
`322e194f…`, backend metal, one 62-minute sequence (10:45–11:47) with
nothing else on the GPU (metadata reads of other files ran on the CPU
during some configurations). Session block 2,595,487,744 bytes (the 32K
session, the block's own cache, and the 150 MB checkpoint region). Means
over the measured runs; per-batch costs are the run's milliseconds divided
by its verify batches; every sample stopped on `token_budget`:

| configuration | prompt | draft | accepted/step | tokens/batch | verify ms | accept ms | recover ms | prefill off → on (s) | decode off → on tok/s | speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prose 512, greedy | 512 | 2 | 1.23 | 2.23 | 224.5 | 0.0 | 99.3 | 5.63 → 16.73 | 10.43 → 6.31 | 0.60× |
| prose 512, greedy | 512 | 4 | 1.67 | 2.65 | 234.9 | 0.0 | 182.2 | 5.67 → 17.48 | 10.07 → 5.67 | 0.56× |
| prose 512, greedy | 512 | 7 | 2.28 | 3.26 | 242.4 | 0.0 | 208.0 | 6.15 → 18.87 | 9.06 → 6.09 | 0.67× |
| prose 512, instruct | 512 | 2 | 1.27 | 2.26 | 237.9 | 46.1 | 106.7 | 6.03 → 18.66 | 8.03 → 5.35 | 0.67× |
| prose 512, instruct | 512 | 4 | 1.68 | 2.67 | 244.2 | 54.1 | 184.4 | 6.08 → 19.13 | 7.95 → 5.04 | 0.63× |
| prose 512, instruct | 512 | 7 | 1.92 | 2.92 | 242.8 | 59.0 | 219.9 | 6.15 → 19.13 | 7.80 → 4.96 | 0.64× |
| code, greedy | 10 | 2 | 1.35 | 2.35 | 239.2 | 0.0 | 102.7 | 0.39 → 0.57 | 9.00 → 6.30 | 0.70× |
| code, greedy | 10 | 4 | 2.34 | 3.34 | 250.6 | 0.0 | 140.7 | 0.39 → 0.55 | 8.91 → 7.56 | 0.85× |
| code, greedy | 10 | 7 | 2.97 | 3.97 | 237.6 | 0.0 | 190.4 | 0.40 → 0.56 | 9.01 → 7.90 | 0.88× |
| code, instruct | 10 | 4 | 2.79 | 3.78 | 226.8 | 77.5 | 98.5 | 0.36 → 0.53 | 7.94 → 8.22 | 1.04× |
| prose 4K, greedy | 4,096 | 4 | 1.98 | 2.95 | 343.6 | 0.0 | 239.2 | 53.05 → 176.01 | 8.49 → 4.62 | 0.54× |
| prose 4K, instruct | 4,096 | 4 | 1.55 | 2.54 | 340.9 | 52.0 | 271.9 | 53.18 → 174.87 | 7.63 → 3.57 | 0.47× |

**Reading.** The switch loses on every row but one, and that one (code,
instruct, draft 4: 1.04×) wins only because its baseline pays the
penalty-path readback (7.94 tok/s against 8.9–9.0 greedy). The costs are
fixed per batch and the batches advance too few tokens: a verify batch
costs 225–250 ms at 512 tokens of context (2.4–2.6 ordinary steps of ≈ 95
ms) and 341–344 ms at 4K, where the small-chunk attention reads the whole
visible cache; recovery costs 99–272 ms whenever a draft is rejected (the
rewind plus a whole-stack replay of the accepted prefix through the same
small-chunk path; free on full acceptance, which is why it grows with the
draft length); the sampled path adds 46–78 ms per batch of host time for
its per-row full-vocabulary sorts (`accept ms`); proposal costs 6.2 ms per
draft and the drafter's commit 6.2 ms per accepted token (inside
`decode`, not separated in this record). Acceptance is 1.2–3.0 drafts per
batch: 42 % per proposed draft on prose at draft 4, 58–68 % on the code
prompt. The speculative prefill is 2.9–3.3× the ordinary one (16.7 s
against 5.6 s at 512; 176 s against 53 s at 4K) because the prompt is
committed through 8-row verify chunks with the output head and a
per-token drafter commit. The "off" baseline drifted within the sequence
(10.43 tok/s in the first configuration to 9.0 after ten minutes and 8.5 at
4K), which the interleaved pairs absorb: the speedup is always the pair's
ratio. Draft length 7 was never worse than 4 in this record (the extra
drafts are cheap to propose and the recover cost grows only on
rejection), but the default stays 4 until the proposal policy exists.

**Verdict.** `generation.speculative` stays off by default for the Qwen
entry, `draft_length` 4; the carried MODL-18 item "decode rate unchanged
with the drafter loaded but switched off" holds against the 2026-09-10
record within the sequence's drift (10.43 against 10.62 tok/s at 512 with
the drafter loaded, its cache and checkpoint region allocated) and gets an
in-process baseline in ENGN-17. The per-batch costs above are the plan's
cost table; the targets are in `TODO.md`.

## The speculative verdict record (ENGN-17, 2026-09-21)

The final measurement of Qwen3.8-27B's embedded draft head on the finished
path (ENGN-13 through ENGN-16, KERN-13 through KERN-18), taken to set the
catalogue entries' defaults and to close the performance theme's plan. Two
baselines are measured: the true baseline with the switch off — `bench` now
opens `.none`, so no drafter weights, scratch, or draft cache are loaded —
and the pair's loaded-but-off sample. `make speculative-record`
(`scripts/nuclis-speculative.py --baseline`; reports under
[benchmarks/speculative-2026-09-21/](../benchmarks/speculative-2026-09-21/))
ran the twelve configurations of the ENGN-12 record as off/on pairs on one
loaded model plus a no-drafter baseline pass per configuration: the
reference corpus arrays at 512 and 4,096 tokens and the fixed code prompt
`Write a Zig function that reverses a string.` (`--raw`, 10 tokens), greedy
and with the instruct profile's sampling (`--temperature 0.7 --top-p 0.8
--top-k 20 --presence-penalty 1.5`, seed 0), draft lengths 2, 4, 7, 128
output tokens, context 32,768, F16 KV, one warmup, three measured
repetitions (two at 4K). Apple M4 Pro (12 CPU, 16 GPU cores), 48 GiB,
macOS 26.6.2 (25G83), AC power, Zig 0.16.0, ReleaseSafe, `nuclis 0.2.0-dev`
at `981f74d` plus the ENGN-17 change, artifact SHA-256 `322e194f…`, backend
metal, one 52-minute sequence (18:39–19:32) with nothing else on the GPU.
Means over the measured runs; per-batch costs are the run's milliseconds
divided by its verify batches; every sample stopped on `token_budget`:

| configuration | prompt | draft | accepted/step | proposed/step | drafts/accepted | tokens/batch | propose ms | verify ms | accept ms | recover ms | checkpoint ms | commit ms | prefill off → on (s) | decode baseline → off → on tok/s | speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code, greedy | 10 | 2 | 1.35 | 1.81 | 1.34 | 2.35 | 11.2 | 218.5 | 0.00 | 6.0 | 3.15 | 4.38 | 0.35 → 0.37 | 10.27 → 10.47 → 9.67 | 0.92× |
| code, greedy | 10 | 4 | 2.17 | 2.92 | 1.34 | 3.17 | 18.0 | 225.0 | 0.00 | 6.2 | 2.81 | 4.41 | 0.35 → 0.38 | 10.11 → 10.35 → 12.39 | 1.20× |
| code, greedy | 10 | 7 | 2.53 | 3.75 | 1.48 | 3.53 | 23.1 | 227.7 | 0.00 | 8.7 | 3.00 | 4.33 | 0.35 → 0.39 | 10.16 → 10.21 → 13.22 | 1.30× |
| code, instruct | 10 | 4 | 2.53 | 3.14 | 1.24 | 3.50 | 19.3 | 229.0 | 0.03 | 6.4 | 2.88 | 4.27 | 0.35 → 0.38 | 10.26 → 10.42 → 13.37 | 1.28× |
| prose 512, greedy | 512 | 2 | 1.12 | 1.62 | 1.45 | 2.12 | 10.1 | 222.9 | 0.00 | 7.7 | 3.60 | 5.32 | 5.78 → 5.89 | 10.50 → 10.53 → 8.48 | 0.81× |
| prose 512, greedy | 512 | 4 | 1.49 | 2.27 | 1.53 | 2.49 | 14.2 | 227.2 | 0.00 | 9.8 | 3.25 | 5.31 | 5.77 → 5.89 | 10.62 → 10.54 → 9.59 | 0.91× |
| prose 512, greedy | 512 | 7 | 1.70 | 2.68 | 1.57 | 2.70 | 16.6 | 227.6 | 0.00 | 10.7 | 3.12 | 5.28 | 5.75 → 5.86 | 10.62 → 10.62 → 10.26 | 0.97× |
| prose 512, instruct | 512 | 2 | 1.22 | 1.69 | 1.39 | 2.22 | 10.4 | 226.0 | 0.02 | 6.0 | 3.22 | 4.89 | 5.75 → 5.86 | 10.56 → 10.58 → 8.85 | 0.84× |
| prose 512, instruct | 512 | 4 | 1.56 | 2.45 | 1.57 | 2.55 | 15.1 | 228.2 | 0.03 | 8.2 | 2.89 | 5.10 | 5.75 → 5.86 | 10.57 → 10.56 → 9.82 | 0.93× |
| prose 512, instruct | 512 | 7 | 1.68 | 2.90 | 1.73 | 2.68 | 18.0 | 229.1 | 0.03 | 10.5 | 3.05 | 5.14 | 5.74 → 5.85 | 10.55 → 10.56 → 10.07 | 0.95× |
| prose 4K, greedy | 4,096 | 4 | 1.51 | 2.06 | 1.36 | 2.49 | 13.2 | 309.1 | 0.00 | 6.3 | 2.90 | 9.39 | 49.83 → 50.99 | 10.09 → 10.07 → 7.30 | 0.73× |
| prose 4K, instruct | 4,096 | 4 | 1.53 | 2.40 | 1.56 | 2.51 | 15.4 | 319.2 | 0.02 | 10.0 | 3.07 | 9.52 | 49.86 → 51.07 | 10.21 → 9.63 → 7.04 | 0.73× |

**Verdict: off for the Qwen entry, `draft_length` 4.** Code greedy reads
0.92 / 1.20 / 1.30× at drafts 2 / 4 / 7 and code instruct 1.28× at draft 4;
prose 512 reads 0.81 / 0.91 / 0.97× greedy and 0.84 / 0.93 / 0.95× instruct;
4K reads 0.73× both. The bar was code ≥ 1.5× and prose ≥ 0.9× at the chosen
length: code tops at 1.30× and prose never reaches 0.9 at 4K, so the switch
stays off. What the performance units bought over the ENGN-12 record is
visible in the cost columns — the sampled decision 46–78 ms → 0.00–0.03 ms
(ENGN-15's device readback), recovery 99–272 ms → 6.0–10.7 ms (ENGN-14's
row checkpoints; the 150 MB copy is 2.8–3.6 ms of it), the prompt commit
2.9–3.3× → 1.02–1.03× (ENGN-13's plan chunk), the proposal policy's
drafts/accepted 1.24–1.73 and tokens/batch 2.12–3.53 (ENGN-16) — but the
verify batch stays the verdict's cost: 218.5–229.1 ms at 512 and 309.2–319.2
ms at 4K for 2.1–3.5 tokens, 1.6–1.8 ordinary decode steps, with the
small-chunk tiles' row-flat work and the chunk attention over the visible
cache. Every other batch cost is now negligible; the levers left are a
cheaper small-batch verify (KERN-12's 2-row route, KERN-14's tile, and
KERN-16's window were the measured attempts, all closed below their
targets) and the proposal policy, which the record shows is already near
its ceiling (63–81 % of proposed positions accepted, highest on the code
prompt with the instruct profile).

**Both baselines.** The pair's loaded-but-off decode equals the no-drafter
baseline within ±2.4 % on eleven of the twelve configurations; the twelfth
(`prose 4K, instruct`) read 9.63 against its baseline's 10.21, which a
focused repeat at 4K did not reproduce (loaded-off 10.25–10.32 vs baseline
10.24–10.27 tok/s over two pairs each, same flags): the MODL-18 item
"decode rate unchanged with the drafter loaded but switched off" holds, and
that configuration's pair-off samples were the drifted ones. Loading the
drafter costs memory and load time, not rate: the session grows 2,304 →
3,851 MB (the 640 MB draft cache and the 150 MB checkpoint region among
it) and `load_milliseconds` 790 → 1,247 ms. The sequence's baselines read
10.1–10.6 tok/s; the ENGN-12 sequence read 10.43 early and 8.5 at 4K, so
the path's rate improvement is real but partly a cooler sequence.

**The cost table** of *Where we are* in [TODO.md](../../TODO.md) is
refreshed from this record; the catalogue defaults it sets are Qwen off
(this record), Gemma off
([§ The Gemma 4 draft pair](#the-gemma-4-draft-pair-modl-19-2026-09-21)),
Muse on ([§ The Muse Glimmer DFlash draft pair](#the-muse-glimmer-dflash-draft-pair-modl-20-2026-09-21)),
each at `draft_length` 4.

## Recovery by row checkpoints (ENGN-14, 2026-09-20)

The second speculative-decoding record, taken after the whole-stack replay
was replaced by per-row recurrent checkpoints
([speculative-decoding.md § Recovery by accepted length](speculative-decoding.md#recovery-by-accepted-length-engn-14-2026-09-20),
[session.md § Row checkpoints](session.md#row-checkpoints-engn-14)). Same
methodology, corpus, prompts, sampling, draft lengths, context, and
precision as the ENGN-12 record above; `make speculative-record`
(`scripts/nuclis-speculative.py`; reports under
[benchmarks/speculative-2026-09-20-recovery/](../benchmarks/speculative-2026-09-20-recovery/))
ran twelve configurations in one 46-minute sequence (20:49–21:35), each an
off/on pair on one loaded model. Apple M4 Pro (12 CPU, 16 GPU cores),
48 GiB, macOS 26.6.2 (25G83), AC power, Zig 0.16.0, ReleaseSafe,
`nuclis 0.2.0-dev` at `417aea0`, artifact SHA-256 `322e194f…`, backend
metal, nothing else on the GPU. Session block 3,850,633,216 bytes (the 32K
session, the block's own cache, the 150 MB checkpoint region, and the
1,255,146,752-byte row region). Means over the measured runs; per-batch
costs are the run's milliseconds divided by its verify batches; every
sample stopped on `token_budget`:

| configuration | prompt | draft | accepted/step | tokens/batch | verify ms | accept ms | recover ms | prefill off → on (s) | decode off → on tok/s | speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prose 512, greedy | 512 | 2 | 1.23 | 2.23 | 254.1 | 0.0 | 9.7 | 5.93 → 6.15 | 10.31 → 7.78 | 0.75× |
| prose 512, greedy | 512 | 4 | 1.67 | 2.65 | 287.0 | 0.0 | 22.2 | 7.61 → 6.88 | 8.26 → 7.67 | 0.93× |
| prose 512, greedy | 512 | 7 | 2.28 | 3.26 | 261.2 | 0.0 | 21.2 | 6.14 → 6.23 | 9.35 → 9.70 | 1.04× |
| prose 512, instruct | 512 | 2 | 1.36 | 2.35 | 249.7 | 54.9 | 6.4 | 6.29 → 6.35 | 8.18 → 7.07 | 0.86× |
| prose 512, instruct | 512 | 4 | 1.82 | 2.80 | 256.3 | 61.4 | 11.1 | 6.28 → 6.38 | 7.99 → 7.68 | 0.96× |
| prose 512, instruct | 512 | 7 | 2.06 | 3.05 | 264.6 | 65.2 | 13.0 | 6.39 → 6.52 | 7.78 → 7.66 | 0.98× |
| code, greedy | 10 | 2 | 1.35 | 2.35 | 254.6 | 0.0 | 8.1 | 0.41 → 0.44 | 8.81 → 8.29 | 0.94× |
| code, greedy | 10 | 4 | 2.34 | 3.34 | 257.6 | 0.0 | 9.8 | 0.38 → 0.43 | 9.15 → 11.13 | 1.22× |
| code, greedy | 10 | 7 | 2.97 | 3.97 | 250.3 | 0.0 | 15.3 | 0.37 → 0.42 | 9.32 → 12.51 | 1.34× |
| code, instruct | 10 | 4 | 2.79 | 3.78 | 241.2 | 76.6 | 6.3 | 0.35 → 0.37 | 8.56 → 10.57 | 1.23× |
| prose 4K, greedy | 4,096 | 4 | 1.98 | 2.95 | 371.7 | 0.0 | 17.7 | 52.44 → 55.29 | 8.72 → 6.85 | 0.79× |
| prose 4K, instruct | 4,096 | 4 | 1.79 | 2.77 | 362.2 | 62.6 | 12.2 | 60.13 → 55.18 | 7.67 → 5.64 | 0.74× |

Per-batch cost breakdown (ms), same samples:

| configuration | propose | verify | accept | recover | checkpoint | commit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prose 512, greedy, d2 | 12.8 | 254.1 | 0.0 | 9.7 | 5.2 | 5.6 |
| prose 512, greedy, d4 | 26.3 | 287.0 | 0.0 | 22.2 | 5.0 | 5.7 |
| prose 512, greedy, d7 | 43.5 | 261.2 | 0.0 | 21.2 | 4.5 | 5.6 |
| prose 512, instruct, d2 | 13.2 | 249.7 | 54.9 | 6.4 | 3.0 | 5.8 |
| prose 512, instruct, d4 | 25.1 | 256.3 | 61.4 | 11.1 | 2.9 | 8.2 |
| prose 512, instruct, d7 | 42.8 | 264.6 | 65.2 | 13.0 | 2.9 | 9.6 |
| code, greedy, d2 | 12.7 | 254.6 | 0.0 | 8.1 | 3.8 | 4.7 |
| code, greedy, d4 | 24.9 | 257.6 | 0.0 | 9.8 | 3.4 | 4.7 |
| code, greedy, d7 | 43.0 | 250.3 | 0.0 | 15.3 | 3.8 | 4.7 |
| code, instruct, d4 | 25.3 | 241.2 | 76.6 | 6.3 | 2.8 | 4.9 |
| prose 4K, greedy, d4 | 26.2 | 371.7 | 0.0 | 17.7 | 4.6 | 10.9 |
| prose 4K, instruct, d4 | 34.0 | 362.2 | 62.5 | 12.2 | 3.0 | 15.3 |

**Reading.** Recovery is out of the picture: 6–22 ms per batch at every
accepted length (all of it one 150 MB slot copy; replay zero), against the
99–272 ms of the ENGN-12 record, and `checkpoint` is 3–5 ms. What decides
a configuration now is the verify batch over the tokens it advances: 250–287
ms at 512 for 2.2–4.0 tokens (63–126 ms per token against an ordinary step
of ≈ 95–105 ms), 362–372 ms at 4K. Verify barely scales with rows (254 ms
at 3 rows against 250 ms at 8), so proposing fewer drafts buys little; the
code prompt wins because acceptance is high (2.34–2.97 drafts per batch,
1.22–1.34× at draft 4–7) and prose loses because 1.23–2.28 accepted drafts
do not amortize a ~2.6-step batch. The sampled path still pays its host
acceptance (55–77 ms, the full-vocabulary sorts), which is ENGN-15 and
KERN-13's target; the verify itself is the new KERN-14 (small-batch tile)
and KERN-16 (long-context attention) territory. The plan was rewritten from
this table: [TODO.md](../../TODO.md).

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

Follow-ups from this profile: the norm launches became KERN-18 across all
three families, the split-K candidate for the matvecs was KERN-15 and
measured behind the single pass, and the Q4_0 prefill tile is untaken.

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
prefill matmul alone on the two FFN shapes for a chunk of `ARGS=<tokens>`
(256 by default): GPU ms per dispatch, GFLOP/s of F32 multiply-adds, GB/s of
weight bytes (read once per token tile), and the prefill tok/s the 54
GFLOP/token model would reach if that were its only cost — a ceiling for
ENGN-02, not a prediction. Since KERN-11 it issues 64 dispatches per command
buffer at t ≤ 8 and 16 above and divides by the count, because one short
dispatch measures the clock ramp. Results are in
[metal-backend.md § Kernels](metal-backend.md#kernels) and, for the
small-chunk tile, [§ Small-chunk tile](metal-backend.md#small-chunk-tile-kern-11-2026-09-19).

`make bench-experts` (`metal-check --experts-bench`) measures the gathered
expert kernels on the Gemma 4 26B-A4B shape (128 experts, 8 selected;
gate-up 1,408 × 2,816 and down 2,816 × 704 per expert, Q4_0): GB/s of the
selected experts' bytes for the gathered gate-up, the gathered down, and
the whole decode chain (gate-up, gelu rows, down, combine), beside a dense
matvec over the same byte count — the rate a gathered kernel can at most
reach; then the prefill tiles over a chunk of `ARGS=<chunk>` tokens (256
by default) routed over random logits: GB/s of the bytes the tiles read
(one expert matrix per 32-row tile), the time per chunk, and the tok/s
that time alone would allow over 30 layers. Same methodology as
`bench-kernels` (64 dispatches per command buffer, best and mean of
five). Results and the reading of them are in
[metal-backend.md § Gathered expert kernels](metal-backend.md#gathered-expert-kernels-kern-09).
`bench-kernels` takes an optional encoding name (`ARGS=Q4_0`) to measure
one kernel alone; a full run heats the GPU progressively (the encodings
measured last come out 15–30 % below their rested rates), so re-measure a
changed kernel alone on a rested machine before comparing with a record.

`make bench-attention` (`metal-check --attention-bench`) measures the
causal chunk attention alone on the model geometry (24 query heads, 4 KV
heads, 256 wide) in F32 and F16: the shipped row-split body beside the
register-reuse body at 4K–32K visible rows (256-query chunks) and at the
verify-shaped counts (1, 8, 64). Each case is one dispatch per repeat
inside one command buffer (`max(1, 65536 / visible)` repeats), two
warm-ups, best of three measured rounds; it reports GPU ms per dispatch,
GFLOP/s of the 8×8 matrix work over each query row's own causal prefix,
and the logical cache bandwidth the math implies. Results and the KERN-16
reading are in [§ Prefill attention sweep](#prefill-attention-sweep-kern-16-2026-09-21).

On the CPU reference backend, each step takes roughly 18–19 s; `bench` runs
there only with small budgets and is useful for definitions, not for speed.

## The KERN-13 quick pass (2026-09-20)

The device penalty kernel (KERN-13) landed; a full record waits for ENGN-17.
This is the unit's gate pass, `make speculative-record ARGS="--only prose512
code"`, which covers the prose-512 and code configurations (the 4K pair was
not re-measured). Same methodology as the ENGN-14 record above: Apple M4 Pro
(12 CPU, 16 GPU cores), 48 GiB, macOS 26.6.2 (25G83), AC power, Zig 0.16.0,
ReleaseSafe, `nuclis 0.2.0-dev` at `d31c5cd`, artifact SHA-256 `322e194f…`,
backend metal, F16 KV, context 32,768, 128 output tokens, one warmup and
three measured runs (two at 4K, not run here) per configuration, off/on
pairs on one loaded model; reports under `.zig-cache/bench/spec/`. Every
sample stopped on `token_budget`; `topk_fallbacks` was 0 at every measured
sample in both the off and on halves. The off half of the instruct
configurations is the new penalized GPU path; the accept half still reads
the full verify rows back (ENGN-15).

| configuration | prompt | draft | accepted/step | tokens/batch | verify ms | accept ms | recover ms | prefill off → on (s) | decode off → on tok/s | speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code, greedy | 10 | 2 | 1.35 | 2.35 | 257.3 | 0.0 | 10.3 | 0.40 → 0.45 | 8.66 → 8.11 | 0.94× |
| code, greedy | 10 | 4 | 2.34 | 3.34 | 273.1 | 0.0 | 12.7 | 0.40 → 0.45 | 8.67 → 10.41 | 1.20× |
| code, greedy | 10 | 7 | 2.97 | 3.97 | 268.8 | 0.0 | 18.9 | 0.40 → 0.44 | 8.74 → 11.63 | 1.33× |
| code, instruct | 10 | 4 | 2.79 | 3.77 | 252.4 | 91.3 | 6.8 | 0.37 → 0.45 | 8.74 → 9.83 | 1.12× |
| prose 512, greedy | 512 | 2 | 1.23 | 2.23 | 248.8 | 0.0 | 11.2 | 5.89 → 6.03 | 10.22 → 7.86 | 0.77× |
| prose 512, greedy | 512 | 4 | 1.67 | 2.65 | 262.8 | 0.0 | 20.5 | 5.99 → 6.28 | 9.50 → 8.31 | 0.87× |
| prose 512, greedy | 512 | 7 | 2.28 | 3.26 | 268.6 | 0.0 | 20.0 | 6.23 → 6.46 | 9.21 → 9.51 | 1.03× |
| prose 512, instruct | 512 | 2 | 1.36 | 2.35 | 251.1 | 62.9 | 6.7 | 6.38 → 6.61 | 9.04 → 6.87 | 0.76× |
| prose 512, instruct | 512 | 4 | 1.82 | 2.80 | 265.5 | 71.7 | 11.7 | 6.50 → 6.62 | 8.87 → 7.31 | 0.82× |
| prose 512, instruct | 512 | 7 | 2.06 | 3.05 | 264.2 | 77.9 | 13.7 | 6.52 → 6.66 | 8.80 → 7.47 | 0.85× |

**Reading.** The instruct baseline moved from the ENGN-14 record's 8.18 /
7.99 / 7.78 tok/s (prose d2/d4/d7) to 9.04 / 8.87 / 8.80, and from 8.56 to
8.74 on code — inside the spread of the greedy off baselines (code
8.66–8.74; prose 9.21–10.22 across the record's consecutive runs is clock
drift over the 25-minute sequence, not a path difference). The sampled
accept time is unchanged (62.8–91.3 ms): it still sorts the full verify
rows, which ENGN-15 moves onto the device readback. Verify and recover are
where ENGN-14 left them (249–273 ms and 6.8–20.5 ms per batch).

## The ENGN-15 quick pass (2026-09-20)

The verify batch's per-row top-k readback landed (ENGN-15): the sampled
acceptance no longer reads and sorts the full logits of every verify row.
This is the unit's gate pass, `make speculative-record ARGS="--only prose512
code"` (the 4K pair waits for ENGN-17). Same methodology and revision line as
[the KERN-13 quick pass](#the-kern-13-quick-pass-2026-09-20) — the runs above
were taken immediately before these, `nuclis 0.2.0-dev` at `d31c5cd` plus the
ENGN-15 change, artifact SHA-256 `322e194f…`; reports under
`.zig-cache/bench/spec/`. Every sample stopped on `token_budget`. `accept`
is the host acceptance per batch in microseconds now that the full-row sorts
are gone; `fallbacks` counts sampled rows whose readback could not decide
(0 everywhere: the readback decided every row).

| configuration | draft | accepted/step | tokens/batch | verify ms | accept µs | recover ms | propose ms | fallbacks | decode off → on tok/s | speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code, greedy | 2 | 1.35 | 2.35 | 263.8 | 0.1 | 9.3 | 13.0 | 0 | 8.27 → 7.97 | 0.96× |
| code, greedy | 4 | 2.34 | 3.34 | 271.8 | 0.2 | 13.4 | 25.8 | 0 | 8.67 → 10.43 | 1.20× |
| code, greedy | 7 | 2.97 | 3.97 | 271.8 | 0.1 | 18.3 | 44.6 | 0 | 8.66 → 11.53 | 1.33× |
| code, instruct | 4 | 2.79 | 3.77 | 274.5 | 36.9 | 10.6 | 25.8 | 0 | 8.64 → 11.79 | 1.36× |
| prose 512, greedy | 2 | 1.23 | 2.23 | 262.2 | 0.1 | 7.6 | 12.9 | 0 | 9.43 → 7.65 | 0.81× |
| prose 512, greedy | 4 | 1.67 | 2.65 | 291.1 | 0.2 | 16.2 | 25.8 | 0 | 8.34 → 7.72 | 0.93× |
| prose 512, greedy | 7 | 2.28 | 3.26 | 294.3 | 0.1 | 18.6 | 45.1 | 0 | 8.25 → 8.84 | 1.07× |
| prose 512, instruct | 2 | 1.36 | 2.35 | 282.8 | 18.8 | 8.1 | 13.2 | 0 | 8.20 → 7.50 | 0.92× |
| prose 512, instruct | 4 | 1.82 | 2.80 | 297.3 | 21.6 | 15.0 | 25.7 | 0 | 8.20 → 8.06 | 0.98× |
| prose 512, instruct | 7 | 2.06 | 3.05 | 297.2 | 23.8 | 18.6 | 44.0 | 0 | 8.19 → 8.24 | 1.01× |

**Reading.** Accept fell from 62.8–91.3 ms per batch (the KERN-13 pass) to
0.02–0.04 ms — 18.8–36.9 µs with penalties, 0.1–0.2 µs for the greedy
sampled path — against the ≤ 5 ms target, two orders of magnitude under it.
The instruct speedups moved with it: code 1.36× at draft 4 (was 1.12× in
the KERN-13 pass) and prose 512 0.92 / 0.98 / 1.01× (was 0.76 / 0.82 /
0.85×). Verify (262–297 ms) and propose (13–45 ms) are now the whole batch;
KERN-14 and ENGN-16 are next. The prose off baselines sit lower than the
KERN-13 pass (8.20 vs 8.80–9.04) — the same clock drift within the session's
runs, and the speedup columns are same-run pairs.

## Small-batch tile sweep (KERN-14, 2026-09-20)

The verify batch's small-batch tile experiment: a 32-output-row variant of
the 16×8 split-K tile, measured by `make bench-matvec-rows ARGS="8 head"`
against `Backend.matmulTile`'s 16×8 control (nothing routes to the
candidate; `Backend.matmulTile32` calls it directly). Apple M4 Pro (48 GiB),
Zig 0.16.0, ReleaseSafe, `d31c5cd` plus the unit's change, minimum of three
measured command buffers after one warm-up, sixteen dispatches per buffer,
synthetic quantization fixtures, no model loaded. Rates are logical weight
bytes per GPU second. 16×8 → 32×8 GB/s at 2 / 5 / 8 tokens:

| encoding | shape | 2 rows | 5 rows | 8 rows |
| --- | --- | --- | --- | --- |
| Q4_K | 17,408×5,120 | 96.3 → 87.7 | 96.8 → 88.7 | 96.7 → 87.7 |
| Q4_K | 5,120×17,408 | 93.2 → 82.0 | 94.4 → 85.0 | 92.5 → 84.9 |
| Q4_K | 248,320×5,120 | 92.3 → 86.3 | 92.2 → 85.6 | 95.3 → 87.1 |
| Q5_K | 17,408×5,120 | 111.0 → 109.5 | 110.0 → 109.8 | 112.0 → 108.0 |
| Q5_K | 5,120×17,408 | 105.6 → 104.3 | 109.4 → 100.7 | 106.2 → 102.1 |
| Q5_K | 248,320×5,120 | 108.5 → 107.2 | 106.4 → 106.9 | 106.6 → 104.7 |
| Q6_K | 17,408×5,120 | 114.4 → 90.0 | 114.0 → 84.6 | 114.1 → 90.8 |
| Q6_K | 5,120×17,408 | 109.1 → 82.6 | 105.0 → 87.8 | 106.1 → 87.0 |
| Q6_K | 248,320×5,120 | 108.6 → 88.0 | 109.2 → 89.0 | 110.2 → 90.1 |
| IQ4_XS | 17,408×5,120 | 91.0 → 89.9 | 90.5 → 92.7 | 90.1 → 88.3 |
| IQ4_XS | 5,120×17,408 | 87.4 → 89.0 | 85.9 → 89.3 | 87.1 → 89.4 |
| IQ4_XS | 248,320×5,120 | 83.2 → 89.8 | 83.0 → 89.5 | 85.6 → 88.8 |
| Q3_K | 17,408×5,120 | 63.6 → 47.7 | 63.0 → 47.1 | 58.5 → 45.2 |
| Q3_K | 5,120×17,408 | 51.6 → 38.2 | 57.2 → 46.1 | 58.2 → 45.8 |
| IQ3_S | 17,408×5,120 | 64.9 → 60.0 | 62.3 → 56.6 | 60.5 → 53.9 |
| IQ3_S | 5,120×17,408 | 60.0 → 55.4 | 60.7 → 55.0 | 59.3 → 54.7 |

**Reading.** The 32-row group loses 8–26 % on the Qwen FFN encodings at 5
rows (Q4_K 96.8→88.7, Q6_K 114.0→84.6, Q3_K 63.0→47.1), ties on Q5_K, and
wins only ~5–8 % on the wide head and IQ4_XS. No case reaches the ≤ 150
GB/s bar. The 16×8 tile stays the verify control; the full-model verify
latency is unchanged (262–297 ms per batch, the ENGN-15 pass above), since
production routing never selects the candidate. Verdict in
[metal-backend.md § The wide 32×8 tile](metal-backend.md#the-wide-328-tile-kern-14-2026-09-20-closed-negative).

## Split-K matvec sweep (KERN-15, 2026-09-21)

The row-poor decode matvec experiment: split-K twins of the Q4_K/Q5_K
specialized matvec (`nu_matvec_q4_k_split`, `nu_matvec_q5_k_split`) and of
the plain segment merge (`nu_matvec_segments_split`), each writing a
`[split][row]` partial buffer reduced by `nu_reduce_splits` /
`nu_segment_reduce_splits`, measured by `make bench-matvec-split` against
the single-pass kernel (splits 1 is the control). Nothing routes to the
split path: only `Backend.matvecSplits` / `matvecSegmentsSplits` force it,
and the `make test-metal` exactness fixture holds it against the F64 CPU
reference at 6,656 rows and at the four-segment merge for every split.
Apple M4 Pro (48 GiB), Zig 0.16.0, ReleaseSafe, `10cf6ba` plus the unit's
change; five command buffers per case (two warm-ups) with 64 dispatches per
buffer below 8,192 rows and 8 above (the clock policy of `--matvec-bench`),
best of three measured rounds, synthetic quantization fixtures, no model
loaded. Rates are logical weight bytes per GPU second; single pass →
2 / 4 / 8 splits:

| encoding | shape | 1 (control) | 2 | 4 | 8 |
| --- | --- | ---: | ---: | ---: | ---: |
| Q4_K | 6,656×19,968 (Muse ffn_down) | 151.0 | 145.9 | 148.1 | 138.8 |
| Q4_K | 6,656×4,096 (Muse attention output) | 157.7 | 150.7 | 135.5 | 124.3 |
| Q4_K | 5,120×17,408 (Qwen ffn_down) | 152.8 | 146.1 | 146.6 | 134.5 |
| Q4_K | 8,704×6,656 (Muse q/k/v/gate merge, four segments) | 152.1 | 157.4 | 150.3 | 158.5 |
| Q5_K | 6,656×19,968 | 193.0 | 193.4 | 198.4 | 193.3 |
| Q5_K | 6,656×4,096 | 175.0 | 170.3 | 159.0 | 130.6 |
| Q5_K | 5,120×17,408 | 191.2 | 190.9 | 184.8 | 172.6 |

**Reading.** The unit's hypothesis — a 6,656–8,704-row matrix launches too
few 16-row groups to keep the weight bus busy — is refuted. Splitting K
multiplies the group count and the partial traffic is 0.2 % of the weight
bytes, yet every Q4_K plain shape loses 2–10 % and the loss grows with the
split count: the signature of a per-group cost (the `simd_sum`/store tail
and the second launch), not of insufficient parallelism. Q5_K's single win
is +2.8 % at 4 splits on a shape Muse uses as Q4_K, and its other cells are
within noise. The merged projection's ~4 % at 2 and 8 splits is the only
reproducible gain — the segment kernel's groups are three times shorter
than the plain matvec's — but it is far below the unit's ≥ 190 GB/s bar,
and that bucket is global: routing it would touch the other families'
merges, which were not measured. No bucket routes. The Q4_K rate is the
kernel's ~0.9 ns per 256-value block (144 bytes) against Q5_K's 176, not
the row count; reaching the bar needs per-block arithmetic, not more K
parallelism. Verdict in
[metal-backend.md § Split-K](metal-backend.md#split-k-kern-15-2026-09-21-closed-negative).

## The ENGN-16 quick pass (2026-09-20)

The proposal policy's gate pass: `engine.draft_p_min = 0.7` (the early stop
on the block's top-candidate probability; the adaptive-length half closed
negative and is not shipped). Same methodology and revision line as the
KERN-13 and ENGN-15 quick passes above — the runs were taken after the
ENGN-15 pass, `nuclis 0.2.0-dev` at `d31c5cd` plus the ENGN-16 change,
artifact SHA-256 `322e194f…`; reports under `.zig-cache/bench/spec/`.
`proposed/step` and `drafts/accepted` are new columns from
`Sample.proposed_per_step`; the unpoliced control's drafts/accepted is
`draft_length / accepted` except for the final partial batch (measured
2.16 for prose d4 and 2.31 for code d7 in the interleaved A/B). Every
sample stopped on `token_budget`.

| configuration | draft | accepted/step | proposed/step | drafts/accepted | tokens/batch | verify ms | propose ms | recover ms | decode off → on tok/s | speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code, greedy | 2 | 1.35 | 1.81 | 1.34 | 2.35 | 249.8 | 12.1 | 12.0 | 8.44 → 8.20 | 0.97× |
| code, greedy | 4 | 2.17 | 2.92 | 1.34 | 3.17 | 255.7 | 19.5 | 13.8 | 8.47 → 10.54 | 1.24× |
| code, greedy | 7 | 2.53 | 3.75 | 1.48 | 3.53 | 258.6 | 24.9 | 17.2 | 8.50 → 11.31 | 1.33× |
| code, instruct | 4 | 2.53 | 3.14 | 1.24 | 3.50 | 259.4 | 21.0 | 13.6 | 8.50 → 11.44 | 1.35× |
| prose 512, greedy | 2 | 1.12 | 1.62 | 1.45 | 2.12 | 233.9 | 10.5 | 16.4 | 10.04 → 7.67 | 0.76× |
| prose 512, greedy | 4 | 1.49 | 2.27 | 1.53 | 2.49 | 253.6 | 15.2 | 21.0 | 8.59 → 8.23 | 0.96× |
| prose 512, greedy | 7 | 1.70 | 2.68 | 1.57 | 2.70 | 260.0 | 17.9 | 22.5 | 8.17 → 8.64 | 1.06× |
| prose 512, instruct | 2 | 1.22 | 1.69 | 1.39 | 2.22 | 250.7 | 11.3 | 13.2 | 8.48 → 7.65 | 0.90× |
| prose 512, instruct | 4 | 1.56 | 2.45 | 1.57 | 2.54 | 254.0 | 16.2 | 19.0 | 8.41 → 8.44 | 1.00× |
| prose 512, instruct | 7 | 1.68 | 2.90 | 1.73 | 2.66 | 256.4 | 19.1 | 20.3 | 8.46 → 8.68 | 1.03× |

**Reading.** The policy trims 25–45 % of the proposed positions (propose
10–25 ms per batch against 13–45 unpoliced) and gives prose instruct its
first ≥ 1× rows (1.00 / 1.03× at drafts 4 / 7 against 0.98 / 1.01× in the
ENGN-15 pass), while code holds: 1.35× instruct and 1.33× greedy at draft 7
(1.33× unpoliced). Drafts per accepted token fall on every row; prose draft
4 reads 1.57 against the interleaved control's measured 2.16 (−27 %), three
points short of the unit's 30 % bar, while its decode rate rose — the bar's
purpose. Verify (234–260 ms) is still the whole batch.

## The Gemma 4 draft pair (MODL-19, 2026-09-21)

The `gemma4-assistant` companion's off/on pair on the Gemma acceptance
workload: `gemma-4-12b-qat` (QAT, every matrix Q4_0), Metal, F16 KV, ctx
32768, `tests/fixtures/run-2026-09-12-gemma4-qat/prompt-512.json`, 128
output tokens, greedy, the companion loaded (`--speculative on`), one warmup
and three measured runs per configuration on one loaded model. Nuclis
`0.2.0-dev` at `4bc7b8d` plus the MODL-19 change; reports under
`.zig-cache/bench/gemma-modl19*.json`. Per-batch costs are the sample fields
divided by `speculative_steps`; `tokens/batch` is
`(generated_tokens − 1) / speculative_steps`; the speedup is the pair's
`decode_tokens_per_second` on/off. Every sample stopped on `token_budget`.
Ordinary decode on this workload is 25.2–25.4 tok/s (39.5 ms per token).

| draft | accepted/step | proposed/step | tokens/batch | verify ms | propose ms | accept µs | recover ms | commit ms | checkpoint ms | decode off → on tok/s | speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1.510 | 1.941 | 2.490 | 136.0 | 3.9 | 0.1 | 0.001 | 0.001 | 0.000 | 25.24 → 17.80 | 0.705× |
| 4 | 2.256 | 3.385 | 3.256 | 135.9 | 6.8 | 0.1 | 0.001 | 0.002 | 0.000 | 25.40 → 22.83 | 0.899× |
| 7 | 2.735 | 4.971 | 3.735 | 136.2 | 9.8 | 0.2 | 0.001 | 0.002 | 0.000 | 25.16 → 25.60 | 1.017× |

**Reading.** The drafter's proposals are accurate (55 % acceptance per
position at draft 7) and every non-verify cost is negligible: the head reads
the target's caches rather than owning one, so `commit` is a row copy and
recovery is the position rewind alone (1 µs per batch against Qwen's 12.7 ms
slot copy). The verify batch is flat at 136 ms across draft 2, 4, and 7 —
3 to 8 rows cost the same because that batch measures the 512-row attention
and the per-layer dispatches, not the row work — so the pair's break-even
sits at about 3.7 tokens per batch and only draft 7 reaches it. The Gemma
head is therefore a correct adapter with a negative default at the plan's
draft length; the levers are `max_draft_length` (the 8-row tile bound) or a
cheaper small-batch verify, both ENGN-17's call.

## The Muse Glimmer DFlash draft pair (MODL-20, 2026-09-21)

The `dflash-kquant` companion's off/on pair on the Muse acceptance workload:
`muse-glimmer-30b`, Metal, F16 KV, ctx 32768,
`tests/fixtures/run-2026-09-19-muse-glimmer/prompt-512.json`, 128 output
tokens, greedy, the companion loaded (`--speculative on`), one warmup and
three measured runs per configuration on one loaded model. Nuclis
`0.2.0-dev` at `316accc` plus the MODL-20 change; reports under
`.zig-cache/bench/muse-modl20-draft{4,8,15}.json`. Per-batch costs are the
sample fields divided by `speculative_steps`; `tokens/batch` is
`(generated_tokens − 1) / speculative_steps`; the speedup is the pair's
`decode_tokens_per_second` on/off. Every sample stopped on `token_budget`.
Each pair is interleaved on one loaded model, which is what the ratio uses;
the off baselines drift down across the three configurations (9.67, 9.35,
9.13 tok/s at 512) over the ten-minute sequence.

| draft | accepted/step | proposed/step | tokens/batch | verify ms | propose ms | accept µs | recover ms | commit ms | checkpoint ms | decode off → on tok/s | speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 1.415 | 1.830 | 2.396 | 172.9 | 26.0 | 0.1 | 0.001 | 1.99 | 0.001 | 9.67 → 11.93 | 1.234× |
| 8 | 1.560 | 2.120 | 2.540 | 183.0 | 48.5 | 0.1 | 0.001 | 2.04 | 0.001 | 9.35 → 10.88 | 1.163× |
| 15 | 1.723 | 2.362 | 2.702 | 188.2 | 51.8 | 0.1 | 0.001 | 2.11 | 0.001 | 9.13 → 11.16 | 1.222× |

**Reading.** The drafter pays on every length — the first positive family
verdict, 1.16–1.23× — and the proposals are accurate: 73–77 % of the
positions it does propose are accepted, and the early stop (`draft_p_min =
0.7`, ENGN-16) trims hard, so only 1.83 / 2.12 / 2.36 positions of the 4 / 8
/ 15 requested are ever forwarded. Every non-verify cost is negligible:
recovery is the position rewind alone (1 µs per batch; Muse Glimmer is
attention-only), `accept` is 0.1 µs (sampled acceptance's device readback
is not used on the greedy path), and the prompt commit adds 1.2–2.9 % to
the prefill (1.012–1.029×). The verify batch is the cost and the lever:
172.9–188.2 ms for 2.8–3.4 rows, 1.7–1.8 ordinary decode steps, because the
batch's matmuls run the padded small-chunk tiles; the block proposal grows
with the requested length (5 / 9 / 16 rows through the small-batch tiles:
26.0 / 48.5 / 51.8 ms). Draft 8 is the worst of the three — 22 ms more
proposal and 10 ms more verify than draft 4 for 0.14 more tokens per batch —
while draft 15's longer batches recover the ground (1.222×). The best
measured length is 4 (1.234×) with 15 statistically tied; ENGN-17 sets the
entry's default from this record. Memory: the session is 2,415,919,104
bytes (2304 MiB) at 32,768 with the five draft caches (640 MiB of it), the
drafter's device workspace is 149,861,504 bytes, and the verify scratch
53,862,464 bytes.

## Prefill attention sweep (KERN-16, 2026-09-21)

The register-reuse chunk attention (`nu_attention_chunk_reuse` /
`_h`: the four SIMD groups of a (head, 32-query) tile split the value
columns and publish the probability tile through threadgroup memory)
against the shipped row-split body, measured by `make bench-attention`
(`metal-check --attention-bench`). One command buffer issues
`max(1, 65536 / visible)` dispatches (a prefill chunk of 256 queries or a
verify-shaped count) after two warm-ups; best of three measured rounds;
the model geometry (24 query heads, 4 KV heads, 256-wide) with synthetic
F32 and F16 caches, no model loaded. Apple M4 Pro (48 GiB), Zig 0.16.0,
ReleaseSafe, `652a0cc` plus the unit's change. GPU ms per dispatch
(a full layer: 24 heads × the tile count):

| visible | count | row-split f16 | register-reuse f16 | Δ | row-split f32 | register-reuse f32 | Δ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 256 | 3.684 | 3.688 | +0.1 % | 3.951 | 4.101 | +3.8 % |
| 4,096 | 256 | 35.179 | 34.246 | −2.6 % | 38.054 | 37.960 | −0.2 % |
| 8,192 | 256 | 71.653 | 69.932 | −2.4 % | 76.276 | 75.120 | −1.5 % |
| 16,384 | 256 | 145.031 | 138.064 | −4.8 % | 159.276 | 149.165 | −6.3 % |
| 32,512 | 256 | 282.230 | 273.058 | −3.2 % | 307.087 | 291.255 | −5.2 % |
| 512 | 64 | 1.139 | 1.097 | −3.7 % | 1.344 | 1.329 | −1.1 % |
| 4,096 | 64 | 8.945 | 7.757 | −13.3 % | 10.477 | 8.998 | −14.1 % |
| 16,384 | 64 | 36.121 | 31.382 | −13.1 % | 41.407 | 35.191 | −15.0 % |
| 512 | 8 | 0.862 | 0.767 | −11.0 % | 0.995 | 0.872 | −12.4 % |
| 2,048 | 8 | 3.610 | 3.163 | −12.4 % | 4.652 | 3.841 | −17.4 % |
| 4,096 | 8 | 7.621 | 6.460 | −15.2 % | 9.329 | 7.702 | −17.4 % |
| 16,384 | 8 | 30.642 | 25.806 | −15.8 % | 37.239 | 30.618 | −17.8 % |
| 16,384 | 1 | 30.704 | 25.831 | −15.9 % | 37.338 | 30.777 | −17.6 % |

**Reading.** The unit's thesis was that the row-split body pairs every
8×8 multiply with ~1.5 `simdgroup_load`s and that reusing a V block over
four row blocks (and a published P block over eight column blocks) would
lift the long-context prefill the 32K acceptance needs. The reuse body is
correct and consistently ahead, but only by 2–5 % at the 256-row prefill
chunks (0 % at 512, −4.8 % at 16K): the loads it removes were not the
limiter. The kernel is flat at ~640–750 GFLOP/s F16 and ~610–700 F32
across the sweep, and the F32 cache — twice the bytes per block —
costs only 5–10 %, so neither matrix throughput nor memory traffic is the
limit; the fixed per-tile instruction and latency chain is, and the reuse
body changes 84 of ~500 instructions per SIMD group per tile. Where the
body wins by 11–16 % is the verify-shaped counts (1–64 rows), because
there each of the four groups carries a quarter of the value columns
instead of one group carrying all 256; that is the window the shipped
routing uses (`attention_reuse_max_rows = 64`, 256-wide values only).
Run-to-run spread is ~1.5 % (the same 32,512/256 case measured 281.2 ms
then 282.2 ms for the row-split body in two sweeps). Verdict in
[metal-backend.md § KERN-16](metal-backend.md#long-context-prefill-attention-second-attempt-kern-16-2026-09-21-closed-negative).

## Fused norm sweep (KERN-18, 2026-09-21)

The fused norm kernels (`nu_rmsnorm_add`, `nu_add_rmsnorm`,
`nu_rmsnorm_rope`) against the pairs they replace, forced off by
`bench --unfused-norms` (`Backend.fused_norms = false`). Apple M4 Pro
(48 GiB), Zig 0.16.0, ReleaseSafe, `1d82c18` plus the unit's change.
**Measurement note:** the profile and rate runs were taken with the
drafter *not* loaded (a local `.none` open) because the shipped `bench`
now loads Qwen's embedded block and measures the off/on pair, and because
`bench` on the Gemma and Muse entries currently fails with
`DraftSourceMissing` (their registry companions wait on MODL-19/20 while
the config's `generation.speculative` is true). This is the 2026-09-07
profile's condition, the one the unit's baselines came from; ENGN-17's
default change re-measures consistently.

`bench --profile`, canonical workload (ctx 2,048, 64 tokens, F16 KV,
greedy), dispatch counts per step (192 measured command buffers, three of
them prefill; the per-decode-step count is `dispatches − prefill·savings`
and is the design's layer count):

| model | body | total/step | rmsnorm | add / add_scale | rope | fused kernels |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Qwen3.8-27B | unfused | 938.8 | 209.0 | 128.0 | 31.5 | — |
| Qwen3.8-27B | fused | 844.3 | 114.5 | 65.0 | — | add_rmsnorm 63.0, rmsnorm_rope 31.5 |
| Gemma 4 12B QAT | unfused | 882.2 | 337.0 | 95.3 | 94.5 | — |
| Gemma 4 12B QAT | fused | 693.2 | 148.0 | — | — | rmsnorm_add 94.5, rmsnorm_rope 94.5 |
| Muse Glimmer 30B | unfused | 922.9 | 314.0 | 102.4 | 76.8 | — |
| Muse Glimmer 30B | fused | 743.8 | 134.8 | — | — | rmsnorm_add 102.4, rmsnorm_rope 76.8 |

Per decode step the fusion removes 96 dispatches on Qwen (64 post norms,
32 q/k norms), 192 on Gemma, and 182 on Muse; the blended totals move
−94.5, −189.0, and −179.2. The bars were ≤ 1,150 (−86) on Qwen, ≤ 705
(−177) on Gemma, and ≤ 737 (−185, 20 %) on Muse: Qwen and Gemma pass,
Muse misses by three because its 13 global layers do not rotate q/k and
only 182 of the assumed 208 dispatches pair.

`bench` decode at 512 (128 tokens, F16 KV, greedy, two or three
interleaved off/on pairs; the best and mean of the off path):

| model | unfused tok/s | fused tok/s | ratio | bar |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.8-27B | 10.814 / 10.742 | 10.788 / 10.744 | 1.000× | ≥ 1.01× ✗ |
| Gemma 4 12B QAT | 27.387 / 27.379 | 27.517 / 27.515 | 1.005× | ≥ 1.04× ✗ |
| Muse Glimmer 30B | 9.993 / 9.936 | 10.018 / 9.975 | 1.004× | ≥ 1.02× ✗ |

With the drafter loaded, Qwen's shipped path reads 10.814 unfused against
10.788 fused (0.998×) — the same conclusion within the run-to-run spread
(~0.5 %). Prefill is unchanged (43.8 / 104.3 / 72.9 tok/s both ways).
**Reading.** The 2026-09-07 profile's ~13 µs per `rmsnorm` dispatch was
kernel work, not a launch floor: the norm kernels move their row twice and
the fused kernel moves it the same way in one dispatch, so the saving is
the removed launch (~2–4 µs of ~7–10 µs), about 0.4 ms of Gemma's 39 ms
step. The dispatch reduction is real (10–21 %) and shipped; the speed bars
are missed and the unit closes below its target. Verdict in
[metal-backend.md § KERN-18](metal-backend.md#fused-decode-norms-kern-18-2026-09-21-closed-below-its-target).
