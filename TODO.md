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
The base binary for `make speed` lives in `.zig-cache/speed/base/` (not
committed; `make speed-base` after each kept change). KERN-20 closed 2026-09-30: captures read in Xcode,
`bench --kernel-stats`, and
[apple-gpu.md](docs/reference/apple-gpu.md) with four kernel readings; KERN-05
and KERN-12 answered. KERN-21 closed 2026-09-30: verify batches run the flash-decoding split
pass (Qwen's 4-row C 387 → 288 ms at 4K, 1,157 → 376 at 32K; Gemma 12B
908 → 200 at 32K), checked at depth by `qwen38-verify-depth-*`. KERN-24 closed 2026-10-01
below its target: the register-fragment tile and two routings cut Qwen's
4-row verify C at 4K from 289 to 244 ms (512: 275 → 230) and Gemma 12B
QAT's by 23 %; the base binary for `make speed` is at `2a6b1ac`.
ENGN-19, the DeltaNet replay tape, is in progress: implemented and
passing the unit and kernel fixtures, **uncommitted in the working tree**,
paused 2026-10-01 at the user's request. Pick up at its *Remaining*
list (gates first).

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
| 1 | ENGN-19 — DeltaNet recurrent verify with a replay tape | 1–2 | checkpoint + recover + slot writes ≤ 6 ms per batch |
| 2 | KERN-22 — Long-context decode attention | 1 | 32K decode ≥ 9.2 tok/s, or closed negative |
| 3 | KERN-23 — Single-row matvec toward MLX-class bandwidth | 2 | 512 decode ≥ 11.5 tok/s, or closed at its ledger |
| 4 | ENGN-20 — Re-price speculation per family; the defaults | 1 | the verdict table is re-measured and the catalogue follows it |

**Re-ranked after KERN-24** (`--profile`, 4-row Qwen verify at 4K,
2026-10-01): matrices 195.3 ms (was 236.8; 2.3 decode steps), DeltaNet
41.3, attention 14.5. The matrices stay the largest term but KERN-24
found their floor (the padded 8×8 multiplies); ENGN-19 is next, then
decode. ENGN-20 must price the fragment tile's flat cost: a 7-draft
verify pays the same matrices as a 3-draft one.

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
the units close. The units are independent of one another: re-rank them
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

## ENGN-19 — DeltaNet recurrent verify with a replay tape

Base: `e35381b`

**Session 1 (2026-10-01): implemented, not yet gated or committed.** The
code is in the working tree (uncommitted); `make metal`, `make test` and
`make test-metal` pass, `zig fmt --check` is clean.

- **Priced before code** (`bench --speculative on --verify-rows 4 --accept
  N`, saved prefixes, warm, per batch in ms; helper
  `.zig-cache/speed/phase.py <binary> <ctx> <rows> <accept> [batches]`,
  not committed):

  | ctx | accept | checkpoint | verify | recover | total |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | 512 | 0 / 1 / 3 | 3.0 | 192 | 14.7 / 14.2 / 0 | 238 / 234 / 221 |
  | 4K | 0 / 1 / 3 | 3.0 | 203–210 | 14.4 / 16.2 / 0 | 258 / 253 / 233 |

  `--profile` at 4K, 4 rows: `delta_chunk` 37.9 ms, `convolution_rows`
  3.0, `convolution_history` 0.45 per batch. `Session.restoreRow` also
  ran a leftover per-float `isFinite` scan over the 151 MB it copied.
- **Design as built.**
  - `nu_delta_rows` (`kernels.metal`, `Backend.deltaRows`): one SIMD group
    per (value head, value row), the state row in registers, `nu_delta`'s
    arithmetic per token; reads the state, never writes it. Per row it
    writes the output and a tape: keys of the 16 Q/K heads + 48 decays
    (`key_stride` 2,096 floats), then the 6,144 corrections.
  - `nu_delta_replay` (`Backend.deltaReplay`): the first `kept` tape rows
    applied to the state in place.
  - The plan's tape (`qwen35_metal.zig` `Plan.tape`, `tape_shape`,
    `tapeLayer`): per DeltaNet layer `max_draft_rows` × 10,240 convolution
    inputs + the delta tape, 28 MB total, only with a drafter. A verify of
    ≤ 8 rows copies `mixed_c` into it, skips `convolutionHistory`, runs
    `deltaRows`, and marks `Session.deferRows(count)`.
  - `Plan.replayRows(kept)`: one command buffer, per layer
    `convolutionHistory` from the taped inputs and `deltaReplay`, then
    `Session.settleRows(kept)`. `settle()` (replay all) runs first in
    `step`, `prefill`, `prefillVision`, `prefillRows`, `verify*`,
    `propose`, `commit`.
  - Session (`runtime/session.zig`): the 1.25 GB row region, `restoreRow`,
    `rowSlotLayer` and `Session.init`'s `row_checkpoints` argument are
    gone; `pending_rows`, `deferRows`, `settleRows`; `begin*`,
    `checkpoint`, `snapshot`, `truncate` refuse `ReplayPending`;
    `rewind`/`reset`/`restore` drop pending rows.
  - Engine: `Model.recover` replays the accepted prefix when rows are
    pending (`Executor.replayRows`); `Model.snapshot` / `checkpoint`
    settle first.
  - Checks: the `metal-check` fixture shows rows + replay **bit-identical**
    to 8 stepped GPU `nu_delta` calls (outputs and every prefix state,
    state untouched). `generation-check`'s recovery check re-runs the
    batch per accepted length so every length goes through the replay, plus
    the `ReplayPending` / replay-bound refusals.
  - The CPU reference is untouched, so the CPU tier is not needed.
- **First measurement** (single runs, 4K, 4 rows): verify 205 → 167 ms,
  recover 14.4 → 1.9–2.7 ms, total 253 → 199 ms at accept 1 and
  233 → 197 at accept 3. Not yet an interleaved `make speed` A/B.

**Remaining, in order:**

1. `make gate NAME=qwen38-speculative-metal` (the recovery check through
   the tape), then `make gate NAME='qwen38-verify-depth-*'` and
   `NAME=qwen38-draft-trace-metal`.
2. Delete the dead row-slot paths: `nu_delta_chunk`'s `row_states`
   block, `DeltaChunkParams.row_states/row_stride`, `deltaChunk`'s
   `slots` argument, the `RowSlots` / `row_states` part of
   `nu_convolution_history` and `convolutionHistory`, and the "Row slots"
   block of the chunk fixture in `metal-check.zig`. Update
   `Model.recover`'s doc comment (it still says rewind then forward).
3. `make speed ARGS='--contexts 512,4096,16384,32639 --verify-rows 4
   --accept 1'` (and `--accept 0`, `3`) plus a decode row (must not
   regress), against the base at `2a6b1ac`.
4. Optional experiments, ledger either way: the deferred checkpoint copy
   (2.9 ms; the live state is the checkpoint until a non-tape forward),
   and encoding the replay into the next forward's command buffer
   instead of its own (saves a submit and wait).
5. `make verify-auto`, `make verify`; docs that still describe row
   checkpoints: `docs/reference/session.md`, `docs/reference/bench.md`,
   `docs/development.md`, ADR 0001's verify cost; then the log entry and
   the commit `perf(inference): …` with the numbers.

Gates: `make test-metal`, `make verify-auto`, `make verify` (no CPU tier:
the CPU runtime is unchanged).

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
