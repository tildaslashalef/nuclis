# TODO — active plan

This file is the only progress tracker. It holds **unfinished work only**:
where we are, what is next, and the design of each remaining unit. When a
unit closes, its outcome moves to
[docs/engineering-log.md](docs/engineering-log.md) and
its section is deleted here; when the last unit closes, this file is emptied
back to this header. Requirements live in [docs/spec.md](docs/spec.md); the
engine map in [docs/architecture.md](docs/architecture.md); how to build,
test, and measure in [docs/development.md](docs/development.md).

Session protocol (also in [AGENTS.md](AGENTS.md)): read this file first. If
it lists work, summarize *Where we are* and ask the user how to continue. If
it is empty, ask what to work on and write the agreed plan here.


## Where we are

Theme agreed 2026-09-30: **decode speed on Metal**, Qwen3.8-27B first,
toward the 20 tokens/s of [ADR 0001](docs/adr/0001-qwen-decode-verifier.md)
(proposed). ENGN-18 closed 2026-09-30: saved prefixes, `bench
--verify-rows`, and `make speed` / `make speed-base` exist, and the
opening baseline is in
[bench.md § The decode-speed baseline](docs/reference/bench.md#the-decode-speed-baseline-engn-18-2026-09-30).
The base binary for `make speed` is saved at `4dc7c70`
(`.zig-cache/speed/base/`, not committed; `make speed-base` after each
kept change). Next: KERN-20's remainder.
The verify's attention and matmul are read against their counters
(apple-gpu.md); what remains is pipeline statistics (`bench
--kernel-stats`) and the rest of `apple-gpu.md`. Then the levers in the order the cost table set
(*Order* below).

Deferred (user, 2026-09-29), until the user picks it up: AGNT-18, the
agent's `decide` tool (its design at the end). A session does not start
it on its own.

Queued (user, 2026-09-30), after the decode-speed theme unless the user
pulls it forward: AGNT-19, saved prefixes for the agent across processes
(its design after AGNT-18).

## The theme: decode speed on Metal

**Why this order.** Our own records already say where the verify batch
goes (ADR 0001, *Where the verify cost goes*): the chunk attention a
verify runs on 24 threadgroups costs about 103 ms per batch at 4K and
413 ms at 16K, and decides every context from 4K up. The single-row path
runs at 63 % of the 273 GB/s peak against about 78 % for MLX on the same
chip, and every verify budget also needs that gap closed. So: tools that
make an experiment take a minute (ENGN-18) and let us see the GPU's
limiters (KERN-20), then the levers ranked by measured cost.

**What the other families get.** Every lever below except ENGN-19 sits in
shared code:

| Lever | Unit | Qwen 27B | Bonsai 27B | Gemma 4 (E4B, 12B, 26B-A4B) | Muse 30B |
| --- | --- | :---: | :---: | :---: | :---: |
| Few-query split-KV verify attention | KERN-21 | ✓ | ✓ | ✓ (draft heads) | ✓ (speculation on) |
| Long-context decode attention | KERN-22 | ✓ | ✓ | ✓ | ✓ |
| Single-row matvec bandwidth | KERN-23 | ✓ | ✓ | ✓ (their encodings) | ✓ |
| Register-fragment verify matmul | KERN-24 | ✓ | ✓ | ✓ | ✓ |
| DeltaNet recurrent verify, replay tape | ENGN-19 | ✓ | ✓ | — | — |
| Re-price speculation per family | ENGN-20 | ✓ | ✓ | ✓ | ✓ |

Bonsai runs on the Qwen adapter (`qwen35_metal.zig` with its Hadamard
fold); Gemma and Muse verify through the same `attentionChunk` and
`mmRows` routes. Every unit re-measures each family's decode it touches
through `make speed` before it closes.

### The loop (every experiment in every unit)

Fast, cheap to abandon, and nothing lands without a measured gain.

1. **Idea → micro-bench** (no model, seconds): `make bench-attention`,
   `make bench-matvec-rows`, `make bench-kernels`, or a new sweep in
   `inference/metal-check.zig`. Write the prediction down first, in the
   unit's ledger below. A kernel that misses its micro prediction by more
   than half is dropped here.
2. **Correctness in seconds**: `make test-metal` (the kernel fixtures,
   against the F64 CPU references, poison past the visible rows).
3. **End-to-end A/B** (about a minute per context, ENGN-18): `make speed
   ARGS='--contexts 512,4096,16384,32639'` interleaves the saved base
   binary and the candidate on restored prefixes.
4. **Keep rule.** Keep a change only when its median decode (or, for a
   verify lever, the batch cost C at the unit's row counts) improves by
   **≥ 2 %** at one context or more and no context regresses by more than
   1 %, over ≥ 5 interleaved pairs; the record's run-to-run spread is about
   1.5 %. Then `make verify-auto` (plus `make verify-long` for attention or
   cache changes) and commit `perf(inference): …` with the numbers in the
   body. Refresh the base binary (`make speed-base`) after each kept commit.
5. **Otherwise revert** (`git restore`, nothing committed), and add one
   ledger line: idea, prediction, measured, why it lost. The ledger is
   committed with the next kept change or at session end, and moves into
   the log when the unit closes, so a negative result stays findable and
   is not re-tried unchanged.

**Be brave, but bounded.** A unit lists more ideas than it needs; try the
cheapest falsifying measurement of each, and spend a session on none that
has not beaten its micro-bench. ENGN-18 and KERN-20 are tooling and land
without a speed gain; every other code change lands only through step 4.

**Apple GPU knowledge** found along the way (limiters, occupancy,
register limits, load widths, measured on this M4 Pro) goes into
[docs/reference/apple-gpu.md](docs/reference/apple-gpu.md), created by
KERN-20, with the source or the measurement for each fact.

### Order

| # | Unit | Sessions | Lands when |
| --- | --- | ---: | --- |
| 1 | KERN-20 — Seeing inside the GPU: capture, pipeline statistics, `apple-gpu.md` | 1 | a decode step and a verify batch are read against their limiters |
| 2 | KERN-21 — Few-query split-KV verify attention | 1–2 | C at 4K drops ≥ 60 ms, or closed negative |
| 3 | KERN-24 — Register-fragment verify matmul | 2 | the 4-row matmul ≤ 1.3 single-row steps, or closed negative |
| 4 | ENGN-19 — DeltaNet recurrent verify with a replay tape | 1–2 | checkpoint + recover + slot writes ≤ 6 ms per batch |
| 5 | KERN-22 — Long-context decode attention | 1 | 32K decode ≥ 9.2 tok/s, or closed negative |
| 6 | KERN-23 — Single-row matvec toward MLX-class bandwidth | 2 | 512 decode ≥ 11.5 tok/s, or closed at its ledger |
| 7 | ENGN-20 — Re-price speculation per family; the defaults | 1 | the verdict table is re-measured and the catalogue follows it |

**The order is ENGN-18's cost table** (bench.md § The decode-speed
baseline, `--profile` of a 4-row verify and a decode step, 2026-09-30).
A 4-row verify batch at 4K is 399 ms of kernel time: weight matmuls 237
(2.8 decode steps' worth), verify attention 106, DeltaNet 44; at 32K
attention is 824 of 1,114. KERN-21 first: the largest term at depth, and
the drafter's commit (48 ms per batch at 32K) runs the same
`attentionChunk`. KERN-24 next: the largest term at 512 and 4K. ENGN-19:
`delta_chunk` 38 ms + recover 14 + checkpoint 3 per batch. Then decode:
KERN-22 (decode attention 5.2 ms per step at 4K, 27.5 at 32K) and KERN-23
(weight matvecs 85 ms per step at every depth).

Identifiers are provisional in this order; they are fixed in the order
the units close. Units 2–6 are independent of one another: re-rank them
when a kept change moves the cost table.

## KERN-20 — Seeing inside the GPU: capture, pipeline statistics, `apple-gpu.md`

`xctrace`'s counter profile is unsupported on this M4 Pro (KERN-05), so
every limiter question so far was answered by guessing. Metal's own
capture is a different path.

- **Capture — landed 2026-09-30.** `Backend.captureNext` (bridge
  `nu_metal_capture_next`, the queue captured until the next commit),
  `bench --capture <path>` (one decode step after the runs), `make
  capture`; procedure in `docs/development.md` § GPU counters by capture.
  Verified: a Qwen decode step at 2K context captured (15 GB document,
  weights included). Xcode 27 has no command-line reader, so the counters
  are read in Xcode's GUI. Kernel captures: `make bench-kernels |
  bench-matvec-rows | bench-attention CAPTURE=<label substring>` (about
  1 GB each, `metal-check`'s `NUCLIS_CAPTURE`). Not yet confirmed that
  Xcode opens and replays either size: the user's first check. First
  reading landed in [docs/reference/apple-gpu.md](docs/reference/apple-gpu.md):
  `nu_matvec_q4_k` on the `ffn_down` shape is issue-bound on the integer
  and complex pipe (limiter 71 %, half its instructions), occupancy 23 %
  against a 47 % target, 192 registers and a 16-byte spill; memory, cache,
  and MMU are not limiters; confirmed at performance state Maximum (149
  GB/s in the replay, the benchmark's rate). Xcode confirmed:
  replay, *Profile after replay*, Performance → Counters → export CSV.
  Verify capture taken 2026-09-30 (`bench --speculative on --verify-rows
  4 --accept 1 --capture`, 4K array, depth 4,095): it replays only in
  Xcode's lite mode (over the full-profiling run-time limit; 332.08 ms
  effective GPU time at Maximum, the unprofiled verify's 335 ms), so no
  counters; delete it once the user has closed it. Counters come from
  kernel captures instead, handed to the user 2026-09-30:
  `attention-4096-c8-reuse-f16` (`make bench-attention CAPTURE=…`, the
  verify's `attention_chunk_reuse_h`) and
  `rows-IQ4_XS-17408x5120-t4-tile` (`make bench-matvec-rows CAPTURE=…`,
  its `matmul_iq4_xs_8`, 0.52 ms, 90.5 GB/s; the multi-row path reads
  103 GB/s at t = 4). **Read 2026-09-30** into
  [apple-gpu.md § The verify batch's two largest kernels](docs/reference/apple-gpu.md#the-verify-batchs-two-largest-kernels-2026-09-30):
  the attention is latency-bound on an empty GPU (occupancy 6 % of an
  85 % target, every limiter ≤ 10 %, 116 GB/s of stack spill traffic);
  the matmul tile is issue-bound (instruction throughput limiter 91 %,
  F32 68 %, occupancy 38 % of 82 %, staging through threadgroup memory,
  half its columns padding). Remaining: a `delta_chunk` case in
  `metal-check` if the DeltaNet verify still matters then.
- **Pipeline statistics.** At pipeline creation, log per kernel
  `maxTotalThreadsPerThreadgroup` (it drops below 1024 when a kernel's
  registers limit occupancy), `threadExecutionWidth`, and
  `staticThreadgroupMemoryLength`; `nuclis bench --kernel-stats` prints
  them. A cheap register-pressure signal for every experiment after this.
- **`docs/reference/apple-gpu.md`.** The M4 Pro GPU as we measure it:
  cores, SIMD width, register file and dynamic caching, threadgroup
  memory, load widths that coalesce, `simdgroup_matrix` throughput we
  reach, half ↔ float conversion costs, the limiters of our hot kernels;
  each fact with its source (Apple tech talks 111373–111375, the Metal
  Shading Language specification, the WWDC 2025 Metal 4 sessions) or its
  measurement. Linked from `docs/architecture.md` § 11 and
  `metal-backend.md`.
- **Answer the two open questions** with the capture: KERN-05's matvec
  limiter (issue-bound or latency-bound at 171 GB/s?) and KERN-12's
  register-pressure hypothesis for the multi-row matvec.

Gates: `make check`, `make verify-auto`; the capture path is off unless
the variable is set, so no numerical gate moves.

## KERN-21 — Few-query split-KV verify attention

`Backend.attentionChunk` / `attentionChunkReuse` dispatch
`query_heads × ceil(count/32) × value_splits` threadgroups: 24 for any
Qwen verify, each walking the whole cache; 8 rows cost 6.46 ms per layer
at 4K and 25.8 ms at 16K, the same as 1 row (bench.md § Prefill attention
sweep).

- **Kernel** `nu_attention_verify` (+ `_h` for the F16 cache) in
  `kernels.metal`, `Backend.attentionVerify` in `root.zig`: grid (KV head,
  key split); a threadgroup's rows are the GQA group's query heads × T
  queries (Qwen: 6 × T, so T = 4 fills three 8-row blocks); each row's
  causal limit is `position + row`, masked only in the splits that reach
  it; softmax per key block in registers (one max and rescale per block,
  `exp2` with the scale folded in); partials merged by
  `nu_attention_merge` extended to T rows. Splits chosen as flash decoding
  chooses them (KERN-08), then swept.
- **Routing.** `qwen35_metal`, `gemma4_metal`, `muse_glimmer_metal` send
  verify batches (count ≤ 8, later ≤ 16) here; prefill chunks keep
  `attentionChunk`. Gemma's sliding-window layers keep their window.
- **Ideas to try, cheapest first:** split count sweep (16…256); K/V
  fragments shared across the 6 heads versus per head; T-major versus
  head-major row order; F16 score accumulation with F32 max/sum; staging
  K through threadgroup memory versus direct `simdgroup_load`.
- **Prediction.** 8 rows ≤ 1.5 ms per layer at 4K, ≤ 4 ms at 16K
  (`make bench-attention`, the verify-shaped counts); the 4K verify batch
  loses ≥ 60 ms in `make speed ARGS='--verify-rows 4'`. Stop below a 3×
  kernel gain. Today (`--profile`, 2026-09-30): `attention_chunk_reuse_h`
  106.0 ms per 4-row batch at 4K, 823.3 ms at 32K. Route the drafter's
  batched commit (`commitBatch`, 2–4 rows: 29 ms at 16K, 55 ms at 32K)
  through the same kernel.
- **Correctness.** A `metal-check` fixture over synthetic F16 operands at
  counts 1–8, visible 512 to 32,767, future rows poisoned (6e4 F16), each
  row against the F64 CPU attention of its own causal prefix, at the
  existing verify-shaped bounds (1e-3 F16). The ADR's Qwen gates:
  verify rows against stepped decode from the same state at 512 and 4K in
  `verify`, at 16K and 32,639 in `verify-long` (saved prefixes from
  ENGN-18 keep them affordable), in `gates.json` through
  `inference/generation-check.zig`.

Gates: `make test-metal`, `make verify-auto`, `make verify`, `make
verify-long`.

## KERN-22 — Long-context decode attention

Single-row decode is 124 ms at 30,650 tokens against 94 ms at 2K
(bench.md, flash decoding): about 30 ms for 2 GB of F16 cache, some 67 GB/s.
`nu_attention_decode` gives a lane channels `l, l+32, …` and does a
`simd_sum`, two `exp`s and a full rescale per key.

- **Ideas, cheapest first:** (a) contiguous `half8` channels per lane (one
  16-byte load); (b) a lane per key inside a 32-key block, one softmax
  reduction per block; (c) 128 or 256 splits; (d) KERN-21's kernel at
  T = 1 (6 heads padded to 8 rows), which would unify decode and verify.
- **Prediction.** The decode attention kernel ≥ 150 GB/s of cache in
  `bench --profile` at 32K (`attention_decode_h` 26.8 → ≤ 13 ms per step);
  32K decode 8.28 → ≥ 9.2 tok/s, 16K 9.25 → ≥ 9.5. (The first target,
  7.55 → 8.3, was set against the hot-chip acceptance record; the
  2026-09-30 baseline already reads 8.28.)
- **Correctness.** The decode attention fixtures in `metal-check`
  (poisoned future rows), the trace gates, `make verify-long`.

Gates: `make test-metal`, `make verify-auto`, `make verify`, `make
verify-long`.

## KERN-23 — Single-row matvec toward MLX-class bandwidth (2 sessions)

171 GB/s in the model at 512 (63 %); MLX reaches about 78 % on a 4-bit
27B on this chip. The micro-benchmarks already read 192–212 GB/s for
IQ4_XS, so part of the gap is in the model, not the kernel. Read
KERN-20's capture first; the ideas are ranked by what it says.

- **What the counters say first** (apple-gpu.md, at full clocks): the Q4_K
  matvec is issue-bound on the integer and complex pipe, not on memory, at
  half its target occupancy with 192 registers. So (b) and fewer live
  registers come first, (a) only if the in-model capture shows an MMU
  limiter the micro-bench does not.
- **Ideas:** (a) weights in Metal-allocated buffers instead of
  `newBufferWithBytesNoCopy` over the file mapping (the in-model rate
  runs about 10 % under the micro-bench; the MMU limiter decides);
  (b) the half magic-number decode (a nibble OR'd into a half of exponent
  1024, minus 1024) with float accumulation, per encoding; (c) scale and
  bias once per group, `s·Σqx + b·Σx`; (d) wider loads per lane, 16 bytes
  aligned; (e) the command buffer split in 2–4 so the GPU starts while
  the CPU encodes, and `MTLDispatchTypeConcurrent` with explicit barriers
  (bounded by the ~2 ms wall-versus-GPU gap); (f) the output head (Q6_K,
  about 1 GB) as its own tuned kernel, since the drafter pays it per
  proposal too.
- **Prediction.** Q4_K 179 → ≥ 200 GB/s in `make bench-kernels`; 512
  decode 10.62 → ≥ 11.5 tok/s. Each idea is judged by `make speed` on
  its own.
- **Correctness.** The per-encoding matvec fixtures, the trace gates of
  every family whose encodings change, `make verify`.

Gates: `make test-metal`, `make verify-auto`, `make verify`.

## KERN-24 — Register-fragment verify matmul (2 sessions)

The verify matmul decodes weights into threadgroup memory
(`nu_matmul_split_body`) and loads B strided on every k8 step; KERN-12's
scalar rows and KERN-14's wider tile both closed below target.

- **Kernel.** Each lane decodes its contiguous quantized run straight into
  its `simdgroup_half8x8` fragment elements (`thread_elements()`; check
  it is public in our MSL version, fixture-test the lane layout); K
  permuted per 128-wide tile so the lane's elements are contiguous words,
  the ≤ 8 activation rows pre-packed once in the same order; no
  threadgroup memory; one weight pass for all rows.
- **The floor.** An 8×8 fragment computes 8 token columns whatever the
  real count: about 64–92 ms of matrix work for an 8-row verify on this GPU
  (estimate), so the unit also measures a scalar body at T ≤ 4 and routes
  by measured row count; drafts of 3–4 are the likely operating point.
- **Prediction.** ≥ 150 GB/s at t = 4 on Q5_K / Q6_K (95–115 today) in
  `make bench-matvec-rows`; 4-row verify matmul ≤ 1.3 single-row steps.
  Stop below 120 GB/s.
- **Correctness.** The matmul fixtures at 1–8 rows per encoding;
  `qwen38-speculative-metal`, the generation gates.

Gates: `make test-metal`, `make verify-auto`, `make verify`.

## ENGN-19 — DeltaNet recurrent verify with a replay tape

Verify runs the 32-token WY chunk for 3–8 rows and writes every row's
recurrent state inside `deltaChunk` (about 151 MB per row); recovery
copies a slot.

- **Design.** For count ≤ 8, a per-token recurrent verify kernel reading
  a frozen state and writing a tape per row (the correction, normalized
  key, and gate the update needs, a few KB per layer); commit replays the
  accepted prefix once and writes the state once. Price it against the
  current chunk + checkpoints first with `make speed ARGS='--verify-rows
  4 --accept N'` for N = 0, 1, 3. Today, per 4-row batch at 4K:
  `delta_chunk` 38.3 ms of kernel time (1.9 ms in a decode step),
  recover 14.0 ms, checkpoint 2.8 ms.
- **Prediction.** checkpoint + recover + slot writes ≈ 32 ms → ≤ 6 ms per
  batch; the region for 8 slots 1.25 GB → tens of MB.
- **Correctness.** Replay against stepped decode, bit-identical on the
  CPU, within the chunk-versus-step bound on Metal, for every accepted
  length (`generation-check --speculative-check`); `make verify-cpu`
  because the CPU runtime's recovery changes.

Gates: `make verify-auto`, `make verify`, `make verify-cpu`.

## ENGN-20 — Re-price speculation per family; the defaults

After the kept levers: re-run the ENGN-17 matrix for Qwen, the Gemma and
Muse draft pairs, at 512 / 4K and, from saved prefixes, 16K / 32K; draft
lengths 2–7; re-tune `p_min` and the length policy at the new costs.
Update the catalogue's speculative defaults per family where the family's
acceptance rule holds, and the ADR's confidence with the measured C and E.
Proposed next from the result, not before: the DFlash 2 checkpoint for
Qwen, suffix drafts for agent edit turns, one root-sibling row.

Gates: `make verify-auto`, `make verify`; `make agent-eval` if a default
the agent uses changes.

## AGNT-18 — The agent's `decide` tool (experiment) — deferred

Deferred 2026-09-29 (user) until picked up. When it is, its baseline is
re-measured on the generated playground (REPO-21) before the tool lands.

- A tool in `src/agent/tools/` calling the in-process `Decider` (loaded
  on first use, `decide.model`): one question (choice, score, or noul
  with its options) over up to 64 candidates given as workspace file
  paths or as the items of the previous tool result; returns the
  candidates ranked with probabilities, truncation marked; limits are host
  constants; failures are results.
- Its description and the system-prompt line measured on the playground
  task list before and after (`make agent-eval VARIANT=…`), on Qwen3.8-27B
  and Gemma 4 E4B: task success, wall time, and prefill tokens saved.
  Kept only if it helps; the result is logged either way.

## AGNT-19 — Saved prefixes for the agent: the primed prefix and `/resume` across processes (2 sessions) — queued

`nuclis agent` prefills its system block and tool definitions at every
start (about 11 s for 934 tokens on Qwen, llm-guide.md § 22) and replays a whole
conversation on `/resume` (minutes at 16K). ENGN-18's `src/prefix_cache.zig`
already writes and restores a model snapshot keyed by model files, tokens,
and layout; this unit uses it for real work, where a stale state is a
correctness bug, not a timing one.

- **Key.** `prefix_cache.Key` gains the build: the binary's
  `nuclis --version` string and the git revision from `build_options`
  (a dev build's revision plus a dirty flag). A different build is a
  miss, never a restore: a saved state must equal what this build's
  prefill would compute. The file header records all four keys.
- **Session 1: the primed prefix.** `Completer.prime` (`src/agent/loop.zig`)
  first calls `prefix_cache.load` from `<NUCLIS_HOME>/cache/prefix/`
  (`src/paths.zig`); on a hit it restores and skips the prefill, on a
  miss it prefills as today, snapshots (already done), and saves.
  `restorePrimed` is unchanged. A `/ctx` change re-opens the engine and
  keys on the new capacity. Speculation on and off are different layouts
  (the draft block's cache), so each has its own file.
- **Session 2: `/resume`.** When the agent writes a session file
  (`src/agent/history.zig` / the save path), it also saves the model
  snapshot at the end of the last completed turn, keyed by the consumed
  tokens; `/resume` restores it when the rendered prefix's tokens match
  and prefills only the remainder, falling back to the replay otherwise.
- **Bounds.** A per-file bound (the session's used extent: about 150 MB
  of recurrent state plus 64 KiB per token for Qwen, 2.3 GB at 32K) and a
  directory budget (`cache.prefix_bytes` in `nuclis.json`, default 8 GB;
  0 disables), evicted oldest-accessed first after each save. `nuclis
  cache ls` and `nuclis cache clear` (APPS surface) show and empty it.
- **Limits, stated in the docs.** Exact token prefixes only (an edited
  system prompt or tool list misses); bound to capacity, KV precision, and
  the draft layout; disk cost as above.
- **Prediction.** The agent's first prompt appears with the primed prefix
  restored in ≤ 0.5 s instead of ~11 s; `/resume` of a 16K conversation
  in ≤ 2 s instead of minutes.
- **Correctness.** Unit tests: a build-key miss, a budget eviction, a
  corrupt file refused and re-prefilled (the ENGN-18 tests extended). A
  restored primed prefix gives the same first-turn tokens as a fresh
  prime (greedy, both backends' generation checks). The agent surface
  through `make shot` (a cold start, then a warm start showing the
  restore; a `/resume`), and `make agent-eval` before and after, since
  the loop's behaviour changes.

Gates: `make check`, `make lint-py`, `make verify-auto`, `make shot`,
`make agent-eval VARIANT=…`. Docs: `docs/reference/session.md`,
`docs/development.md` § User directories, the agent's help.
