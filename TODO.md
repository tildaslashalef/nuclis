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
QAT's by 23 %. ENGN-19 closed 2026-10-01: verify batches of up to 8 rows
step DeltaNet without writing the state and recovery replays a tape, so
Qwen's 4-row C fell 48–51 ms at every depth (4K 239 → 191 ms, 32K 329 →
279) and the 1.25 GB row-slot region is gone; the base binary for `make
speed` is at ENGN-19's commit. Next: ENGN-20, re-pricing speculation
at the new costs (*Order* below, re-ordered 2026-10-01).

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
| 1 | ENGN-20 — Re-price speculation per family; the defaults | 1 | the verdict table is re-measured and the catalogue follows it |
| 2 | KERN-23 — Single-row matvec toward MLX-class bandwidth | 2 | 512 decode ≥ 11.5 tok/s, or closed at its ledger |
| 3 | KERN-22 — Long-context decode attention | 1 | 32K decode ≥ 9.2 tok/s, or closed negative |

**Re-ordered after ENGN-19** (user, 2026-10-01): ENGN-20 first, since
speculation now pays (an estimated 1.2–1.3× from the ENGN-17 acceptance
at the new C) and its measured E decides how much each decode lever is
worth; then KERN-23, the largest per-token lever (matvecs 85 of ~95 ms
per step) and the groundwork for a multi-row scalar verify body; then
KERN-22, which pays mainly at depth (about 14 ms per step at 32K, plus
the verify's attention there). Pull KERN-22 forward if long agent
sessions become the main use.

**Re-ranked after ENGN-19** (`--profile`, 4-row Qwen verify at 4K,
2026-10-01): matrices 195 ms, attention 14.5, DeltaNet 6.4 (was 41.3),
recover 2–3 ms, checkpoint 3. The verify's remaining cost is the matrices'
floor KERN-24 found; decode (KERN-22, KERN-23) is next, and ENGN-20
re-prices speculation at the new C.

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

## ENGN-20 — Re-price speculation per family; the defaults

Base: `fec1ec3`

**What changed since the verdicts.** ENGN-17 (2026-09-21) set Qwen and
Gemma off, Muse on, each at `draft_length` 4, against a 4-row Qwen C of
257 ms at 512 and 342 at 4K. Since then C fell to 177 / 191 / 232 / 279 ms
at 512 / 4K / 16K / 32,639 (KERN-21, KERN-24, ENGN-19; bench.md § The
DeltaNet replay tape). ENGN-18's E (2.4–2.8 emitted per batch at draft 4,
every depth) predicts Qwen at 2.5 / 0.19 s ≈ 13 tok/s at 512 against 10.56
plain (~1.25×), ~1.3× at 4K, ~1.2× at 32K. Gemma's verify fell 23 % (KERN-24);
Muse's too, through the shared `attentionChunk`/`mmRows` routes.

**Session 1 — tools (code, `make check`):**

- `inference.engine.Speculative` gains `p_min: f32 = draft_p_min`;
  `speculativeBatch` reads it instead of the constant. `bench` alone gains
  `--draft-p-min <0..1>` (`src/cli.zig`, `src/help.zig`, completion; a
  measuring knob like `--verify-rows`, not a user setting: `generate` and
  `agent` reject it). The report carries `speculative_p_min`.
- `scripts/spec-matrix.py` (`make spec-matrix ARGS=…`, self-test in
  `make workloads-validate`): per cell (model, context or `code`, sampling,
  draft length, p_min) one `nuclis bench --speculative on --draft-length L
  --repeat 3 --warmup 0 --max-tokens 128 --ctx-size 32768 --kv f16 --json`
  process, contexts restoring speed.py's saved prefixes
  (`--prefix-cache .zig-cache/speed/prefix`, the family's acceptance
  arrays through `speed.prompt_for`), `code` the ENGN-17 prompt (`Write a
  Zig function that reverses a string.`, `--raw`). A pair names the
  catalogue entry (`gates.json` `entry`) so the companion resolves. It
  derives per cell: accepted/proposed per batch, **E** = (generated − 1) /
  batches, **C** = decode ms / batches with its components, off → on tok/s,
  speedup as the median of the 3 pair ratios, C / 50E; saves reports under
  `.zig-cache/spec/<model>/<rev>/`, prints a markdown table, `--json`
  the rows. Samplings: `greedy`, `instruct` (the profile's think-off
  options: Qwen 0.7 / 0.8 / 20 / presence 1.5; Gemma and Muse 1.0 / 0.95 /
  64), `thinking` (Qwen think-on 1.0 / 0.95 / 20, the agent's).

**Session 1 — the matrix (background runs, about 3 hours):**

| Family (key) | Contexts | Samplings | Draft lengths |
| --- | --- | --- | --- |
| Qwen (`qwen38`, embedded head) | 512, 4096, 16384, 32639, code | greedy, instruct | 2–7 |
| Gemma 12B QAT (`gemma4_qat`) | 512, 4096, 16384, 32639 | greedy, instruct | 2–7 |
| Gemma E4B, 26B-A4B | 512, 4096 | greedy, instruct | 2, 4, 7 (all 2–7 if any ≥ 1.0×) |
| Muse (`muse`, DFlash) | 512, 4096, 16384, 32639 | greedy, instruct | 2, 4, 6, 8, 11, 15 |

Then, per family at its best length: `p_min` ∈ {0, 0.5, 0.6, 0.7, 0.8} at
512 instruct, 4K greedy, and code greedy (Qwen), and Qwen's `thinking`
sampling at 512 and 4K. Write each table into the docs as it lands.

**The verdict rule** (one for every family, replacing ENGN-17's Qwen bar of
code ≥ 1.5× / prose ≥ 0.9×): an entry turns speculation on at draft length
L when, at L and the shipped `p_min`, no measured configuration reads below
0.98× (the drift band) and the geometric mean over the prose contexts and
samplings is ≥ 1.10×. L is the length with the best geometric mean,
ties to the shorter. `draft_p_min` changes only if one value beats 0.7 by
≥ 2 % in the geometric mean with no configuration worse by > 1 %.

**Lands:** `src/catalog.zig` (`speculative`, `draft_length` per entry and the
test at the end), each entry's verdict comment; `engine.draft_p_min` if it
moves; bench.md § a new *The re-priced speculative verdicts (ENGN-20)*
with the tables; speculative-decoding.md's verdict paragraphs per family;
ADR 0001's budget table and *Confidence* with the measured C and E; the
report JSONs (the summary rows, not every sample) under
`docs/benchmarks/speculative-2026-10-01/`. If Qwen turns on: `make
agent-eval VARIANT=spec-on` against the off run (task success, wall time),
and `make shot` of a turn showing the bar's `spec N.NN/step`.

Proposed next from the result, not before: the DFlash 2 checkpoint for
Qwen, suffix drafts for agent edit turns, one root-sibling row.

Gates: `make check`, `make lint-py`, `make verify-auto`, `make verify`
(the engine's speculative path changes); `make agent-eval` if a default
the agent uses changes.

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
