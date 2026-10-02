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
279) and the 1.25 GB row-slot region is gone. ENGN-20 closed 2026-10-02:
speculation re-priced on every family; the catalogue turns it on for Qwen
(draft 7: prose 1.25–1.57×, short code 20.3–20.9 tok/s), Gemma 12B QAT
(5), E4B (6), and Muse (6), and off for Gemma 26B-A4B; `make
spec-matrix` measures it, and `agent --print` now loads the drafter. An
existing `~/.nuclis/nuclis.json` keeps its entries' old values until the
user edits them or re-runs `config init`. The base binary for `make
speed` is at ENGN-19's commit (no engine arithmetic changed since).
REPO-27 closed 2026-10-02: an external review's fixes (a failed
recording is discarded, so metal-check reports instead of hanging; the
kernel table is derived from `Kernel`; `--file` names a support file's
weights; parser property tests). Its log entry lists what waits for
KERN-23: pruning the 56 check-only pipelines (1.70 s of a cold start)
and the GGUF type-id enum. No engine arithmetic changed, so the `make
speed` base binary stands. Next: KERN-23, re-scoped 2026-10-02 (user) to
the single-row matvec and a multi-row verify body, since speculation is
now the default path (*Order* below).

Deferred (user, 2026-09-29), until the user picks it up: AGNT-18, the
agent's `decide` tool (its design at the end). A session does not start
it on its own.

Queued (user, 2026-09-30), after the decode-speed theme unless the user
pulls it forward: AGNT-19, saved prefixes for the agent across processes
(its design after AGNT-18).

Queued (user, 2026-10-02), after the decode-speed theme unless the user
pulls it forward: APPS-19 (2 sessions), `nuclis serve`, a local decision
API that keeps decision models loaded and batches concurrent requests into
one Metal pass (its design at the end).

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
| Single-row and few-row matvec bandwidth | KERN-23 | ✓ | ✓ | ✓ (their encodings) | ✓ |
| DeltaNet recurrent verify, replay tape | ENGN-19 | ✓ | ✓ | — | — |

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
| 1 | KERN-23 — Weight streaming for one row and a few: decode and the verify body | 2 | 512 decode ≥ 11.5 tok/s, a 4-row verify C ≤ 150 ms at 512, or Qwen speculative prose 512 ≥ 18 tok/s; or closed at its ledger |
| 2 | KERN-22 — Long-context decode attention | 1 | 32K decode ≥ 9.2 tok/s, or closed negative |

**Re-ranked after ENGN-20** (2026-10-02): speculation is on for Qwen
(draft 7), Gemma 12B QAT (5), E4B (6), and Muse (6), so a verify lever now
moves the default decode rate, and every decode lever moves the plain
path the drafters fall back to. Qwen's C / 50E at draft 7 is 1.21–1.42 at
512 and 4K (C 164–182 ms against 128–141 ms budgets) and 1.6–1.9 from
16K; both Gemma entries break even at 32K, where the verify attention is
linear in rows, so KERN-22 now pays in speculation as well as decode.
The drafter's proposal is 16–18 ms per Qwen batch at draft 7, its head
the output head KERN-23's idea (f) names.

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

## KERN-23 — Weight streaming for one row and a few: decode and the verify body (2 sessions)

Base: `c2c03a4`

**Why, since ENGN-20.** Speculation is on for Qwen at draft 7, so a
default turn spends its time in verify batches: at 512, C 164 ms = propose
16.5 + checkpoint 3 + verify 138 + recover 2.5 + commit 4 (cold, greedy;
bench.md § The re-priced speculative verdicts). The verify's matrices sit
at KERN-24's floor: the fragment tile pads every 8×8 multiply and costs the
same from 1 to 8 rows, its rates at 4 rows 107 GB/s (Q4_K) to 152 (Q6_K),
against 179–212 GB/s for the single-row matvec in `make bench-kernels`.
Plain decode still runs wherever speculation is off (Gemma 26B-A4B, Bonsai,
files outside the catalogue, a turn after an image) and is 85 ms of
matvecs in a 95 ms step at 512.

**What the counters say** (apple-gpu.md, at full clocks): the Q4_K matvec
is issue-bound on the integer and complex pipe, not on memory, at half its
target occupancy with 192 registers. The multi-row matvec kernels for 2–8
tokens already exist (`nu_matvec_rows_{q4_k,q5_k,q6_k,iq4_xs}_t2..t8`,
`nu_matvec_rows_k_body` and siblings in `kernels.metal`), but only 2 rows
route to them (KERN-12): beyond that each token's accumulators push the
registers and the body loses to the tile. KERN-24's close names the gap:
a scalar body that shares activations across rows without that register
wall; none was tried. Cheaper decode arithmetic and fewer live registers
serve both halves of this unit.

**Session 1 — the single-row matvec.** Ideas in this order, each judged
first on `make bench-kernels` (Q4_K, Q5_K, Q6_K, IQ4_XS, IQ4_NL, Q8_0 at
Qwen's shapes), then by `make speed` decode at 512 and 4K: (b) the half
magic-number decode (a nibble OR'd into a half of exponent 1024, minus
1024) with float accumulation, per encoding; (c) scale and bias once per
group, `s·Σqx + b·Σx`; (d) 16-byte aligned loads per lane; then (a)
weights in Metal-allocated buffers instead of `newBufferWithBytesNoCopy`,
only if a capture shows an MMU limiter; (e) the command buffer split so
the GPU starts while the CPU encodes; (f) the Q6_K output head (about 1
GB) as its own kernel, which the drafter pays per proposed position too.
`--kernel-stats` reads each candidate's thread limit before it is timed.

**Session 2 — the multi-row body.** Carry session 1's decode into the
`_t3..t8` bodies with the accumulators bounded (rows × tokens per
SIMD group chosen so the thread limit stays at 1,024 in
`--kernel-stats`), and measure every encoding at 3–8 tokens against the
fragment tile with `make bench-matvec-rows ARGS="8 frag"`. Where the body
beats the tile at a token count, route verify batches of that count to it
(the routing that sends 2 tokens to `matvec_rows` today, in
`inference/src/backends/metal/root.zig`), per encoding. Kept only through
`make speed --verify-rows 4,8` and the speculative record below; Gemma's
Q4_0 verify gets the same body if Q4_0 joins.

**Predictions** (written before code; each idea's own goes into the
ledger):

- Q4_K single-row matvec 179 → ≥ 200 GB/s in `make bench-kernels`; 512
  plain decode 10.56 → ≥ 11.5 tok/s.
- A 4-token multi-row body ≥ 150 GB/s on Qwen's Q4_K and IQ4_XS shapes
  (the tile reads 107 and 122); Qwen's 4-row verify C at 512 177 → ≤ 150
  ms in `make speed`.
- Qwen prose 512 with speculation (`make spec-matrix ARGS='--model qwen38
  --contexts 512 --drafts 7 --cooldown 90 --rev <rev>'`, greedy and
  instruct): 16.5 / 15.9 → ≥ 18 tok/s.

**Lands when** any prediction is met with no context regressing, or the
unit closes at its ledger. **Correctness:** the per-encoding matvec and
multi-row fixtures in `metal-check` against the F64 CPU references
(poisoned rows past the visible ones), the trace gates of every family
whose encodings change, `qwen38-verify-depth-*` and
`qwen38-generation-metal` (verify rows against stepped decode), `make
verify`.

Gates: `make test-metal`, `make verify-auto`, `make verify`; the
speculative record above for the third prediction.

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
  on first use): one question (choice, score, or noul with its options)
  over up to 64 candidates given as workspace file paths or as the items
  of the previous tool result; returns the candidates ranked with
  probabilities, truncation marked; limits are host constants; failures
  are results.
- **Checkpoint: measured, not assumed.** The tool opens `decide.model`
  (default `laya`), and every measurement below runs both checkpoints;
  this unit settles the default (*Settling the default* below). The
  evidence so far points both ways: on English filter tasks (ranking logs
  by urgency, cache misses among 50 entries, relevant search hits) `laya`
  ranked correctly and `laya-multilingual` lost the rankings (README,
  2026-09-29); on a check
  that turns on comparing numbers in a JSON state (a distance against a
  target) `laya-multilingual` answered P = 0.99 where `laya` answered at
  chance (0.51, 0.54 with nothing cut). `laya-multilingual` reads 1,024
  tokens per sequence against 512 and costs about 0.4 s to open (its
  34 MB `tokenizer.json`), once per process.
- **A labeled set, scored per decision.** End-to-end task success alone
  cannot separate a good filter from a bad one: the agent may finish by
  reading the files anyway. Before the tool lands, build 30–60 cases from
  playground sessions (search hits with the relevant ones marked, build
  and test logs with the failing section marked, diffs with the hunks
  that matter marked, and cases that turn on comparing numbers against a
  threshold: sizes, counts, durations, exit codes) and score each
  decision: the right candidate in the top 1 and top 3, and whether the
  probabilities track correctness.
- **A cheap baseline on the same set:** keyword overlap between the
  question and each candidate, and a random order. The tool is worth its
  GPU time only where Laya beats both.
- **Settling the default.** Score `laya` and `laya-multilingual` on the
  labeled set, top 3 over all cases as the measure, reported per category
  (search hits, logs, diffs, numbers). The higher becomes the default; a
  margin under 3 cases keeps `laya` (upstream's root checkpoint, 75 ms to
  open). If `laya-multilingual` wins: set `Config.Decide.model` in
  `src/config.zig`, the `--model` row of `decide`'s help in
  `src/help.zig`, the catalogue table and `decide` row in `docs/spec.md`,
  the config example and `decide.model` line in `docs/development.md`,
  `docs/reference/laya.md` § `nuclis decide`,
  `docs/reference/artifacts.md` § The catalogue, and the README's Laya
  section (whose "worse on English" sentence the scores replace); then
  measure the agent's first `decide` call, and if the 0.4 s open shows,
  the parsed-vocabulary cache from `laya.md`'s limits joins the unit.
  Either way the log entry carries the per-category table.
- Its description and the system-prompt line measured on the playground
  task list before and after (`make agent-eval VARIANT=…`), on Qwen3.8-27B
  and Gemma 4 E4B: task success, wall time, and prefill tokens saved.
  Kept only if it helps on both the labeled set and the task list; the
  result is logged either way.

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

## APPS-19 — `nuclis serve`: a local decision API that batches across requests (2 sessions) — queued

Base: recorded when the unit starts.

Every `nuclis decide` call starts a process and opens its checkpoint:
about 75 ms for `laya` and 0.4 s for `laya-multilingual` (parsing its
34 MB `tokenizer.json`) before a 230–300 ms Metal encode, so a client
deciding every second spends half its time loading. A long-running
process that keeps decision models open removes that; batching requests
that arrive together into one Metal pass raises throughput the way a
multi-state call already does (one state 2,440 tokens/s, 50 states 3,500,
[laya.md § Time per call](docs/reference/laya.md#time-per-call)); an
endpoint listing the models lets clients follow the catalogue.

Where the time goes decides the design: the encode is hundreds of
milliseconds and HTTP on loopback is tens of microseconds, so the server
must add nothing measurable around the encode, never leave the GPU idle
while a request waits, and pack whatever is waiting into the next pass.

### Surface

- **Command.** `nuclis serve` (APPS surface, `src/serve.zig`, dispatched
  from `src/cli.zig` like `decide`): `--host` (default `127.0.0.1`),
  `--port` (default 8735), `--model <name|path>` (repeatable, opened at
  start; otherwise each opens on first use), `--backend cpu|metal`.
  Binding anywhere but loopback prints a warning; there is no
  authentication.
- **Routes.** JSON in and out, typed errors as
  `{"error": {"code", "message"}}` with the HTTP status that fits:
  - `POST /v1/decide`: the body `nuclis decide --request` reads
    (`questions`, `state` or `states`, an optional `model` naming a
    decision model, default `decide.model`), `?explain=1` for the explain
    fields. The response is byte-for-byte what `nuclis decide --json`
    writes for the same request, timings aside. `{"file": path}` states
    are refused: the server reads no file a request names.
  - `POST /v1/systemone`: Jev's single-state call for clients written
    against TypeSafe's API: one `state`, the response `{model, answers,
    usage}`.
  - `GET /v1/models`: every decision entry (catalogue and registry, `kind:
    "decision"`): name, repo, revision, present, loaded, and the budgets
    from its `rl_agent_config.json` (`max_len`, `head_max_len`).
  - `GET /v1/health`: version, backend, loaded models, queue depth.
- **Shared code.** The JSON request parsing (`parseJson`,
  `stateFromJson`, `questionsFromJson`, the request part of
  `buildRequest`) and the JSON writer (`writeAnswer` and the `--json`
  body of `run`) move from `src/decide.zig` to `src/decide_json.zig`, so
  the CLI and the server cannot drift.

### Session 1: the server, lean

- **I/O.** `std.Io` (Zig 0.16) net listener and `std.http.Server`: one
  accept loop, each connection its own task (`io.async`), HTTP/1.1
  keep-alive, no pipelining. Each connection owns a fixed read buffer
  (16 KiB headers) and an arena reset after every response, so a warm
  connection allocates nothing that outlives a request. Bodies are read
  into the arena up to the limit, parsed once, rendered once; the
  response goes out with `Content-Length` in one write.
- **Models.** A table of open `inference.decide.Decider`s keyed by the
  resolved directory, at most 2 open (least recently used evicted when no
  batch holds it), each opened once under a lock however many first
  requests race.
- **The GPU.** One worker task owns decisions; connections hand it a
  job and wait on its completion. In this session it runs jobs one at a
  time.
- **Limits.** Host constants: body 4 MiB, headers 16 KiB, 64
  connections, 64 queued jobs (then 503 `busy`), 30 s per request
  (`timeout`); `inference.decide`'s request limits (`max_states`,
  `max_questions`, `max_options`, `max_state_bytes`) apply unchanged.
- **Targets** (Apple M4 Pro, ReleaseFast, `ab -k`, 1,000 requests):
  `GET /v1/health` p50 ≤ 0.2 ms and p99 ≤ 1 ms at concurrency 1, ≥
  20,000 requests/s at concurrency 16; a warm `laya-multilingual`
  `POST /v1/decide` (one state, two questions) within 2 ms of the encode
  `timings_ms` reports, against 0.75 s through the subprocess.
- **Correctness.** Unit tests over in-memory connections: routing, each
  error (malformed JSON, unknown model, a file state, an oversized body,
  oversized headers, busy, timeout), keep-alive across requests, and the
  model table's eviction, on the tiny synthetic checkpoint on the CPU,
  under `std.testing.allocator` with no leaks. A server on an ephemeral
  port answers the 8 root fixture requests with exactly `nuclis decide
  --json`'s results (a new fast-tier gate, `decide-serve`).

### Session 2: batching across requests

- **Scheduler.** While a batch runs, arriving jobs queue per model. When
  the GPU frees, the worker takes every queued job for the oldest job's
  model whose sequences fit the Metal plan's 2,048 rows (in arrival
  order, a job never split), runs them as one `Decider.decide` call, and
  hands each job its own results. No waiting window: an idle GPU starts
  the first job at once, so a lone client pays nothing for batching. The
  CPU backend batches the same way.
- **Why it is safe.** A packed sequence's logits do not depend on what is
  packed beside it (bit-identical,
  [laya.md § On Metal](docs/reference/laya.md#on-metal)), so batching
  changes timing, never answers. The test asserts it: 16 concurrent
  requests return exactly the results each gets alone.
- **Fairness.** Jobs for another model wait at most one batch: the next
  batch serves the oldest waiting job's model.
- **Targets** (same machine, `ab -k -c 16`, a one-state two-question
  `laya` request, 512 requests): throughput ≥ 2× session 1's at
  concurrency 16, p99 latency ≤ 2 batches' time; concurrency 1 unchanged
  within 2 %.
- **Measured and recorded**: requests/s and p50/p99 at concurrency 1, 4,
  16 for both checkpoints, before and after batching, in the `nuclis
  serve` section of `laya.md`.

Docs: `docs/spec.md` (the command table), a `nuclis serve` section in
`docs/reference/laya.md`, `docs/development.md` (the port, the limits,
measuring with `ab`), `src/help.zig`, `docs/architecture.md` (the
decision path).

Gates: `make check`, `make verify-auto` (the new `decide-serve` gate). No
numerical behaviour changes, so no Metal or CPU tier.
