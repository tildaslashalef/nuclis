# 0001 — Amortize Qwen decode through a dedicated small-batch verifier

**Status:** proposed
**Date:** 2026-09-29

Source inspected at `48b55d8`; the cited paths re-checked at `e174a3e`. Throughput improvements below are targets;
the cited benchmark results are historical measurements.

## Context

### Target and constraints

Target one stream on the M4 Pro, 48 GB, at approximately 20 emitted tokens/s
at **each** acceptance length. Keep the pinned Qwen artifact, full visible
context, F16 KV, and the existing numerical and sampling contracts. A smaller
artifact or lossy cache is a separate quality/performance experiment.

The [acceptance record](../reference/bench.md#acceptance-runs), not a fresh measurement:

| Prompt tokens | Recorded decode tokens/s | Improvement to 20 |
| ---: | ---: | ---: |
| 512 | 10.62 | 1.88× |
| 4,096 | 10.20 | 1.96× |
| 16,384 | 8.27 | 2.42× |
| 32,639 | 7.55 | 2.65× |

Those 2026-09-10 numbers need re-measuring at the current revision.
Use the same token arrays, 128 outputs, 32,768 capacity and warmup policy.
Do not substitute configured capacity for actual visible prompt length.

### Why ordinary decode alone is an unlikely route

The [architecture's traffic estimate](../architecture.md#11-where-performance-goes)
is 16.1 GB of weights per forward. Apple's
[M4 Pro specification](https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/)
gives 273 GB/s peak memory bandwidth. Under the assumption that those weights
must stream from memory each step, their ideal floor is 59 ms, or 17 tokens/s.
20 tokens/s would require 322 GB/s for weights alone, before attention,
recurrent state, arithmetic and dispatch overhead. Recompute actual executed
tensor bytes before treating this estimate as a hard bound.

Improving matvec remains useful, but the route most likely to cross this
floor is amortizing a target forward across several **accepted** tokens.
Speculative decoding can preserve a target distribution with the appropriate
acceptance algorithm ([original paper](https://proceedings.mlr.press/v202/leviathan23a.html)).
Nuclis's particular numerical and sampled-acceptance contracts still need
their own gates; the paper is not proof of our implementation.

### What exists and what has failed

The [Qwen speculative verdict](../reference/bench.md#the-speculative-verdict-record-engn-17-2026-09-21)
already measures the embedded MTP head. Code reaches 13.22 tokens/s greedy
and 13.37 sampled on a short prompt; prose at 512 tops at 10.26 greedy,
and the 4K draft-4 pair is only 7.30 greedy / 7.04 sampled. Speculation stays
off for Qwen. No Qwen speculative 16K/32K success is established here.

The costly component is verification: about 227 ms at 512 and 309–319 ms
at 4K. Proposal, recovery, checkpoint and commit are already separately
measured. Per-row recurrent checkpoints and device top-k acceptance already
exist; proposing them again would duplicate completed work.

Source confirms `qwen35_metal.Plan.verify` / `verifyGreedy` run a causal
small prefill batch and the full output head over all rows. `Backend.matmulImpl`
routes eligible two-row batches to multi-row matvec; larger batches use tiles.
Scratch padding to 64 rows is an allocation contract, not execution of 64
rows: dispatch uses the actual row count, rounded to the selected token tile.
The [multi-row sweep](../reference/metal-backend.md#multi-row-matvec-kern-12-2026-09-20-closed-below-its-target)
found growing scalar work and possible register pressure; the latter remains
a hypothesis without compiler/profiler evidence. Wider tiles, split-K,
attention reuse and norm fusion also have recorded limited or negative
results in [the benchmark record](../reference/bench.md). Reopen them only with a changed
mechanism and a prediction that the experiment can falsify.

### Where the verify cost goes, from our own records

`Backend.attentionChunk` dispatches `query_heads × ceil(count/32) ×
value_splits` threadgroups, so a 3–8-row verify runs 24 threadgroups per
value split, each walking the whole visible cache. The
[KERN-16 sweep](../reference/bench.md#prefill-attention-sweep-kern-16-2026-09-21)
prices it at 8 rows, F16: 0.767 / 6.46 / 25.8 ms per layer at 512 / 4K /
16K visible, and the same at 1 row as at 8. Over Qwen's 16 full-attention
layers that is about 12 / 103 / 413 ms per verify. The 512 → 4K difference
(91 ms) matches the measured verify growth (82–92 ms), while single-row flash
decoding grows only from 94 to 98 ms per step over the same range. Verify
attention, not the matmul, decides every context from 4K up.

### External evidence (surveyed 2026-09-30)

References and oracles only, never sources to copy. **M** marks a
measurement published by the source, **C** a claim without shown data.

| Technique | Mechanism | Evidence | Our status |
| --- | --- | --- | --- |
| Split-KV few-query attention with GQA packing | One threadgroup per (KV head, key split), rows = 6 query heads × T queries, per-row causal limit, partial merge | MLX [`sdpa_vector_2pass`](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/scaled_dot_product_attention.cpp); llama.cpp `flash_attn_ext_vec`; [Open-TQ-Metal](https://arxiv.org/html/2604.16957v1) §3.2 split-K on an M1 Max (M) | Done for single-row decode (KERN-08); absent for verify |
| Register-fragment verify matmul | Decode quantized weights straight into 8×8 fragment elements (`thread_elements`), K permuted per lane with activations packed to match; no threadgroup staging | [metal-flash-attention](https://github.com/philipturner/metal-flash-attention) design (no async copies on Apple9, C); [llama.cpp #29110](https://github.com/ggml-org/llama.cpp/pull/29110) 4-row register tile 1.6–1.9× (M, M3 Ultra); MLX `qmv_wide` (C) | Untried; our tiles stage through threadgroup memory (KERN-11/14) |
| Per-token recurrent verify plus replay | Below ~64 tokens the recurrent form beats the chunk (WY) form; verify keeps a frozen state and a per-token tape, commit replays the accepted prefix once | [vLLM #58863](https://github.com/vllm-project/vllm/pull/58863) 1.3–2.3 ms/cycle (M, GB10); mlx-lm `gated_delta_step` (C) | Row checkpoints written inside `deltaChunk` |
| Block softmax, contiguous loads in decode attention | Lane owns contiguous `half8` channels; one max/rescale per key block | MLX `sdpa_vector`; metal-flash-attention | Per-key `simd_sum` and rescale; 30,650-token decode reads KV at about 67 GB/s (estimate) |
| GPU counters by capture | `MTLCaptureManager` `.gputrace`: ALU/FP16/integer utilization, limiters, occupancy, L1 eviction, MMU limiter on Apple family 9 | [Apple tech talk 111374](https://developer.apple.com/videos/play/tech-talks/111374/) | Never used; `xctrace` counters are unsupported here |
| Target-trained drafter | DFlash 2 checkpoint for this model | [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2): τ 9–20 % above MTP (C, H200) | MODL-20 DFlash adapter (Muse); DFlash 2 differs |
| Suffix / prompt-lookup drafts | Free exact drafts from the prompt and history | [SuffixDecoding](https://arxiv.org/abs/2411.04975): 1.8–4.5× on agent traces (M, vLLM) | None |
| Root-sibling verification | One extra row from S₀ carrying the head's second candidate | [GDN Tree-Scan](https://arxiv.org/html/2609.23900v1): +27 % at B = 1 (M, GB10) | Chain-only checkpoints |

Ceiling references, same class of hardware: MLX runs an M = 1 forward of a
4-bit Qwen 27B on an M4 Pro in 68 ms and M = 3 at 1.38× that
([mlx#3553](https://github.com/ml-explore/mlx/issues/3553), M); at our
estimate of its 14.4 GB of weights that is about 78 % of peak bandwidth,
against our 63 %. A third-party blog reports 18.3 tokens/s for Qwen3.6-27B with MTP on a
48 GB M4 Pro (C, unchecked). Metal 4's tensor API lowers to the same ALUs
before M5 and gives no gain here (llama.cpp #16634, C); indirect command
buffers and concurrent dispatch bound at the ~2 ms wall-versus-GPU gap.
Lossy KV (Open-TQ's int4) and activation sparsity change the target and are
out of scope.

## Decision

Qwen decode is amortized through a dedicated small-batch verification
path behind the existing model-independent verification contract. Select its kernels by
measured row count, tensor shape and encoding; preserve target semantics,
full context and recurrent recovery. Establish feasibility before committing
to the 20 tokens/s outcome.

### Experiment sequence

The units, their predictions, and the keep rule (a change lands only with
a measured decode gain) are the plan in [TODO.md](../../TODO.md) while the
theme runs, and the engineering log afterwards. In order:

1. A fast measurement loop: saved prefixes, the verify batch cost C at
   every depth and row count, interleaved A/B against a base binary; the
   current four-context baseline and the first 16K/32K acceptance counts.
2. GPU counters from Metal captures, answering the open limiter and
   register-pressure questions.
3. Few-query split-KV verify attention; then long-context decode
   attention, the single-row matvec, the verify-width matmul, and the
   DeltaNet recurrent verify, ordered by the measured cost table.
4. Re-price C and E per family and set the speculative defaults.

### Correctness coverage required before a candidate lands

The existing tiers remain required, but do not cover the proposed Qwen path
at all target contexts. At the inspected revision, `make verify-long` checks
Gemma 4 E4B at 4K; `qwen38-speculative-metal` exercises short F32 contexts.
A passing throughput workload does not establish numerical correctness.
Add explicit Qwen gates to `gates.json`, extending
`inference/generation-check.zig` and the kernel fixtures in
`inference/metal-check.zig` as needed:

- At each acceptance prompt length (512, 4,096, 16,384, 32,639), with F16 KV
  and 32,768 capacity, compare every verifier row with an ordinary stepped
  target starting from the same prefix state. Exercise 2–8-row batches using
  the family's [chunk-versus-step bound](../reference/speculative-decoding.md#the-recovery-contract-engn-11). Check greedy choices and sampled
  selection from full logits versus device top-k, with the same seed and
  correctly advanced penalty history; ordinary and speculative sampled
  streams need not be seed-identical.
- Force every accepted draft prefix, including zero drafts and full
  acceptance. Compare recovered convolution and DeltaNet state, draft-cache
  alignment, and the next correction logits against the stepped control.
  Exercise the final context tail, where fewer rows fit, including the
  single-row fallback and rejection near capacity.
- For a new attention schedule, compare each query's causal prefix against
  the CPU attention reference over the same F16 operands at 16K and near
  32K, with future cache rows poisoned to expose out-of-prefix reads.

The 512 and 4K comparisons fit the `verify` tier. Each 16K and 32K
comparison pays two long prefills (the 32,639-token prefill alone ran
about 11 minutes in the acceptance record), so those gates belong in
`verify-long` or `verify-release`. The attention comparison is a
`metal-check` kernel fixture over synthetic F16 operands, not a full-model
CPU run. Pin the inputs, bounds, tiers, and runnable gate commands in the
experiment design before implementation. These checks supplement the existing cancellation,
failure, and GPU lifetime coverage. The additional gates remain to be
implemented and passed.

### Quantitative stop rules

Use total emitted tokens divided by total decode seconds. If a batch emits
an average of E tokens and costs C milliseconds including **all** components,
20 tokens/s requires C ≤ 50E. Use ratios of summed counts and times, not
averages of per-batch rates.

| Existing configuration | Emitted tokens/batch | Total budget at 20 tokens/s |
| --- | ---: | ---: |
| Prose 512, greedy, draft 7 | 2.70 | 135 ms |
| Prose 4K, greedy, draft 4 | 2.49 | 124.5 ms |
| Short code, greedy, draft 7 | 3.53 | 176.5 ms |

These are calculated budgets using historical emitted counts, not predictions
of future acceptance. For prose 512 draft 7, non-verifier costs total about
35.7 ms, leaving approximately **99 ms for verification**, against 227.6 ms
historically. At 4K draft 4, non-verifier costs total about 31.8 ms, leaving
**93 ms**, against 309 ms. For short code draft 7, non-verifier costs total
about 39.1 ms, leaving **137 ms**, against 227.7 ms: a 40 % cut, the only
budget within reach of a verifier-only improvement.

Compare the verify budgets with today's **single-row** decode step, from the
acceptance record:

| Context | Single-row step today | Verify budget at 20 tokens/s |
| --- | ---: | ---: |
| 512 (prose, draft 7) | 94 ms | 99 ms |
| 4K (prose, draft 4) | 98 ms | 93 ms |
| 16K | 121 ms | ≈93 ms, if E and non-verifier costs held at 4K |
| 32K | 132 ms | ≈93 ms, same assumption |

At 512 a batch of about 3.7 rows must cost what one row costs today; at 4K
and beyond it must cost **less than today's single-row step**. The verifier
streams the same 16.1 GB as that step, so no verifier schedule reaches these
budgets unless the single-row path's own weight streaming (about 171 GB/s of
the 273 GB/s peak at 512) and long-context attention improve too. Ordinary
decode efficiency is therefore on the critical path, not supporting work.
There are no measured acceptance counts here from which to price 16K/32K.

Stop an individual candidate when its predicted saving does not materialize
in complete batch wall time, when it only moves costs elsewhere without a
net gain, or when numerical gates fail. An incremental gain may be retained
without reaching 20 tokens/s alone; record its remaining budget gap and
re-measure combinations rather than adding isolated kernel savings.

Apply the 20 tokens/s feasibility rule to the complete route at each context.
Re-price C and E after each retained change. Stop pursuing that target when
the measured costs and an explicit optimistic bound on the remaining work
still cannot fit C ≤ 50E; state that bound's assumptions. If no exact
same-artifact route is credible after these experiments, report that result
and propose a separately approved artifact/quality tradeoff. At 20 tokens/s,
an ordinary forward could read at most 13.65 GB at theoretical peak bandwidth;
the practical weight budget is smaller once all other costs are included.

## Consequences

### First deliverable

A current four-context baseline with the single-row step's achieved
bandwidth, and a verifier cost table, followed by one
bounded kernel experiment selected from that evidence. Commit its mechanism,
control, the Qwen correctness coverage above, numerical gate commands, and
separate candidate and whole-route pass/stop thresholds before implementation.
This proposal does not promise that 20 tokens/s is achievable on the unchanged
model at every context.

### Implementation observations to test

These are concrete differences in execution paths, not diagnosed bugs:

- `qwen35_metal.zig:recordLayers` uses separate gate/up matmuls and a
  standalone SiLU multiply. Single-row `step` uses merged projections with
  a fused SiLU epilogue and fused residual/norm. The verifier misses those
  fusions. That adds dispatches and intermediate traffic, but earlier norm
  fusion was effectively neutral for Qwen, so this is not a doubling claim.
- The small specialized matmul tile has eight token columns. Typical verify
  batches have about three or four real rows, so only three or four of the
  tile's eight token columns carry real work. The two-row special route exists, but current scalar bodies
  degrade beyond two rows. This is useful work lost to geometry, not 64-row
  execution: the larger scratch padding does not set dispatch size.
- `deltaChunk` uses a 32-token WY formulation even for tiny verify batches.
  When row checkpoints are enabled it additionally reconstructs and writes
  each row's recurrent state inside the verifier kernel. That cost is booked
  to **verification**, not the separately timed `recover` call. Ablate the
  checkpoint-writing cost with controlled replay recovery and compare a
  tiny-batch recurrent schedule; do not disable recovery in production.
- Chunk attention uses a different schedule from single-row flash decoding:
  24 threadgroups per value split, each walking the whole cache, and a 32-row
  tile at least 75 % padding for a verify. The reuse body's 11–16 % win at
  small counts does not change that grid; see *Where the verify cost goes*.
- Qwen's MTP proposals are serial forwards carrying a hidden vector and
  maintaining a private draft cache; accepted target hidden rows then advance
  that cache. [Muse's DFlash](../reference/bench.md#the-muse-glimmer-dflash-draft-pair-modl-20-2026-09-21)
  has a different proposal mechanism and no Qwen
  DeltaNet state to recover. Its positive result cannot establish that Qwen's
  drafter or verifier has the same economics.

### Confidence

Subjective engineering estimate, not a statistical probability: approximately
**15% confidence** in 20 tokens/s at all four contexts with the unchanged
artifact and current draft source, before new measurements: from 4K up, the
verifier must beat today's single-row step. Approximately **50% confidence**
for short coding workloads, where the historical best is already 13.4
tokens/s and the verify budget needs a 40 % cut. The survey supports the
shape of this: an M4 Pro running MLX at about 78 % of peak bandwidth would
take a single-row step to about 13 tokens/s, and its 3-row forward costs
1.38× a step; reaching 20 at 512 needs both that efficiency and a verify
near that ratio. There is stronger evidence for some improvement than
for a uniform doubling. Refresh these estimates after the first cost table.

The benefit is weight reuse across accepted tokens. The cost is a separate
execution schedule, more numerical validation, and workload-dependent draft
acceptance. Revisit if the profiled costs cannot meet the batch budgets, if
acceptance falls at long context, or if a compatible and materially better
trained draft source becomes available (the published DFlash 2 checkpoint
is the first candidate).

## Alternatives considered

- Ordinary matvec tuning alone: the estimated weight-streaming floor is above
  the 50 ms target, so it cannot be the route alone; but the verify budgets
  also require it (see the stop rules), so it runs alongside the verifier.
- Enable current speculation globally: the measured prose and 4K regressions
  rule it out.
- Repeat wider tiles, scalar multi-row kernels or split-K unchanged: prior
  sweeps supply negative controls; a new mechanism must explain a new outcome.
- Reduce quantization or truncate context: changes quality or target semantics
  and requires a separate decision and evaluation.
- Replace the drafter first: may improve acceptance, but retains an expensive
  verifier and adds checkpoint, integration and memory uncertainties.

## Related

- [Engineering log](../engineering-log.md#repo-24--qwen-decode-at-20-tokenss-evidence-and-experiment-proposal-2026-09-30).
- [Speculative recovery and draft sources](../reference/speculative-decoding.md).
- [Metal execution and kernel experiments](../reference/metal-backend.md).
