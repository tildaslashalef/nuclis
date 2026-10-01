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
The base binary for `make speed` is saved at `d1f0a1b` (KERN-21's code)
(`.zig-cache/speed/base/`, not committed; `make speed-base` after each
kept change). KERN-20 closed 2026-09-30: captures read in Xcode,
`bench --kernel-stats`, and
[apple-gpu.md](docs/reference/apple-gpu.md) with four kernel readings; KERN-05
and KERN-12 answered. KERN-21 closed 2026-09-30: verify batches run the flash-decoding split
pass (Qwen's 4-row C 387 → 288 ms at 4K, 1,157 → 376 at 32K; Gemma 12B
908 → 200 at 32K), checked at depth by `qwen38-verify-depth-*`. Next:
KERN-24, the verify's weight matmuls, now 236 of 306 ms of kernel time
at 4K (*Order* below).

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
| 1 | KERN-24 — Register-fragment verify matmul | 2 | the 4-row matmul ≤ 1.3 single-row steps, or closed negative |
| 2 | ENGN-19 — DeltaNet recurrent verify with a replay tape | 1–2 | checkpoint + recover + slot writes ≤ 6 ms per batch |
| 3 | KERN-22 — Long-context decode attention | 1 | 32K decode ≥ 9.2 tok/s, or closed negative |
| 4 | KERN-23 — Single-row matvec toward MLX-class bandwidth | 2 | 512 decode ≥ 11.5 tok/s, or closed at its ledger |
| 5 | ENGN-20 — Re-price speculation per family; the defaults | 1 | the verdict table is re-measured and the catalogue follows it |

**Re-ranked after KERN-21** (`--profile`, 4-row Qwen verify, warm,
2026-09-30): at 4K weight matmuls 236.4 ms, DeltaNet 41.7, attention
14.4 (was 106.3); at 32K matmuls 235.9, attention 95.8 (was 823.6),
DeltaNet 41.3. The order below stands: KERN-24, ENGN-19, then decode.

**The order was ENGN-18's cost table** (bench.md § The decode-speed
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

## KERN-22 — Long-context decode attention

Single-row decode is 124 ms at 30,650 tokens against 94 ms at 2K
(bench.md, flash decoding): about 30 ms for 2 GB of F16 cache, some 67 GB/s.
`nu_attention_decode` gives a lane channels `l, l+32, …` and does a
`simd_sum`, two `exp`s and a full rescale per key.

- **Ideas, cheapest first:** (a) contiguous `half8` channels per lane (one
  16-byte load); (b) a lane per key inside a 32-key block, one softmax
  reduction per block; (c) 128 or 256 splits; (d) the GQA group's 6
  heads × T rows as one `simdgroup_matrix` Q tile against 8-key blocks.
  Since KERN-21 the verify batch runs this same kernel with a row
  dimension (`attentionVerify`), linear in rows at about 0.23 ms per row
  per layer at 4K and 95.8 ms per 4-row batch at 32K: every idea here is
  measured on `make bench-attention`'s verify rows too, and (d) is
  KERN-21's untried idea for them.
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

Base: `d39e3d4`

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

**Session 1 (2026-10-01), in the tree, uncommitted at this point:**
`nu_matmul_frag_body<ENC, RB, TA, TB, MODE>` (kernels.metal, after the
wide tile), `nu_fragment_layout` + `checkFragmentLayout`, the `_f2` /
`_f2hh` / `_f2ff` / `_f4` / `_p1*` / `_p2*` instantiations,
`Backend.matmulKernel` (forced kernel), `specializedMatmulFrag` (not yet
routed), `make bench-matvec-rows ARGS="<rows> frag"` (`fragBench`). The
fragment tiles pass `make test-metal` against the generic F32 tile, worst
2.02e-5 of Σ|w·x| (half tile 2.96e-5; bound 2e-4).

Where the 4-row Qwen verify's matrix time goes (`--profile`, 4K,
`d39e3d4`): `matmul_iq4_xs_8` 84 ms over its shapes, `q4_k_8` 59, `q5_k_8`
62, `q6_k_8` 16 (the head 8.8), the generic tile 31 (Q8_0 48×5120 β/α
17.1 ms at 1.5 GB/s: 96 dispatches of 2 threadgroups; IQ4_NL 13 ms at 25
GB/s), `q3_k_8` / `iq3_s_8` 10.

**Ledger** (`make bench-matvec-rows ARGS="4 frag"`, 4 rows, GB/s of weight
bytes, 17408×5120 / 5120×17408; tile today: Q4_K 96.6 / 94.5, Q5_K 112.1 /
109.0, Q6_K 116.0 / 109.9, IQ4_XS 90.6 / 88.4; the tile's ms is flat across
encodings, 0.52–0.66):

1. Fragment tile, eight activation fragments built per step and kept live
   across the decode. Predicted ≥ 150 on Q5_K/Q6_K. Measured: half × F32
   40–50 (2× slower than the tile), half × half 83, F32 × F32 46, 32 rows
   per group 52. Lost to what the next line fixed (registers or a
   non-unrolled fragment array).
2. MMA floor probe (`_p2*`: no decode, no activation loads): 0.234 ms on
   Q4_K 17408×5120, about 6 TFLOP/s of 8×8 work (173–214 GB/s
   equivalent); with activation loads (`_p1*`) 0.25 ms. The tile is not
   near its matrix floor; the decode is the larger half.
3. **Candidate (kept in the tree):** each activation fragment built just
   before its multiply, `#pragma unroll` on both loops: Q4_K 107.4 /
   104.4, Q5_K 115.3 / 111.7, Q6_K 152.6 / 137.5, IQ4_XS 122.2 / 115.4
   (half × half no faster than half × F32: 105.8 / 103.8 on Q4_K, so the
   activations stay F32). Next: a cheaper decode into the fragment
   (decode costs ~0.22 ms of Q4_K's 0.47), RB = 4, then routing and
   `make speed`.

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
