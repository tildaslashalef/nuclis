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

**Next: KERN-23 session 2** (the multi-row body), or first the rest of
session 1 (below). Session 1 (2026-10-03) met the unit's first
prediction: Qwen's plain decode at 512 10.37 → 11.78 tok/s, at 4K 10.24
→ 11.34, by three kept kernel changes (`e36b77d`, `1137eb2`, `c151ee2`);
the base binary for `make speed` is `c151ee2`. Its ledger and the
step's profile are in the KERN-23 section. REPO-29 closed 2026-10-03:
the tree is on Zig 0.17.0. REPO-30 closed 2026-10-03: REPO-29's 4K loss is not
one; `fd09aa4` under Zig 0.16 and the tree under 0.17 both read 10.0
tok/s decode and a 4-row C of 194 ms at 4K, so KERN-23's 4K
starting point is about 10.0 / 194, not REPO-29's 9.18 / 207. The 16K and 32K saved
prefixes under `.zig-cache/speed/prefix/` are gone (512 and 4K exist):
the first run at those depths prefills them again, about 3 and 11 min. MODL-34 closed 2026-10-02 in one
session: `nuclis decide --model clef-flash` and `nuclis serve` answer
with Cloudflare's 9B decision model, text and images, every question of
a state in one pass (2.3 s for a 637-token state on Metal;
[clef.md](docs/reference/clef.md)). APPS-19 closed 2026-10-02: `nuclis
serve` keeps decision models open behind the nuclis API on
127.0.0.1:8000 ([api.md](docs/reference/api.md)). The order of the units
that follow is the table at the end of this section; the history of the
speed theme comes first.

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
user edits them or re-runs `config init`.
REPO-27 closed 2026-10-02: an external review's fixes (a failed
recording is discarded, so metal-check reports instead of hanging; the
kernel table is derived from `Kernel`; `--file` names a support file's
weights; parser property tests). Its log entry lists what waits for
KERN-23: pruning the 56 check-only pipelines (1.70 s of a cold start)
and the GGUF type-id enum.

**Re-ordered 2026-10-02 (user).** Decision models move ahead of the last
speed unit: first APPS-19 (closed), `nuclis serve`, the nuclis API
(`src/api/`) with decisions as its first service, an OpenAI-compatible
service expected later; then MODL-34 (closed),
Cloudflare's clef-flash decision model; then KERN-23, which closes the
decode-speed theme. Dropped (engineering log): KERN-22, long-context
decode attention, a small win at 32K only; AGNT-18, the agent's `decide`
tool, since decision models are served by `nuclis serve` and the agent
stays a tool for language models.

| # | Unit | Sessions |
| --- | --- | ---: |
| 1 | KERN-23 — Weight streaming for one row and a few (closes the decode-speed theme) | 2 |
| — | AGNT-19 — Saved prefixes for the agent across processes | queued after KERN-23 |

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
   1.5 %. A 1–2 % gain keeps when every pair is faster and within 0.5 % of
   the median change (amended 2026-10-03; `scripts/speed.py` prints it).
   Then `make verify-auto` (plus `make verify-long` for attention or
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

Base: `8c47b56`

**Ledger** (session 1; `make bench-kernels` best GB/s at base, Qwen
shapes 4×ffn_gate / 4×ffn_down / output / ffn_down: Q4_K 178.6 / 157.5 /
179.5 / 150.3, Q5_K 211.5 / 205.4 / 209.7 / 190.6, Q6_K 251.1 / 246.2 /
251.9 / 238.5, IQ4_XS 212.2 / 188.8 / 217.2 / 180.7; idea (c) is already
the K-quant body's form, `d·sa·Σqx − dmin·ma·Σx`):

- (b) Q4_K half magic-number decode, high nibbles kept at ×16 with the
  1/16 folded into the group scale. Prediction: Q4_K ≥ 195 / 175 / 195 /
  170 GB/s. Measured 199.9 / 178.5 / 199.8 / 177.4 (+11–18 %); `make
  speed` decode 512 10.37 → 10.79 tok/s (+4.0 %), 4K 10.24 → 10.65
  (+4.0 %), 5 pairs each within +3.8..+4.6 %. **Kept.**
- (b) Q5_K, the same decode with the fifth bit OR'd in at bit 4 (low)
  and bit 8 (high, ×16). Prediction: Q5_K ≥ 225 / 220 / 225 / 205 GB/s.
  Measured 255.9 / 250.9 / 257.6 (+21–23 %); `make speed` decode 512
  11.04 → 11.35 tok/s (+2.8 %), 4K 10.64 → 10.93 (+2.7 %). **Kept.**
- IQ4_XS byte-pair table: a `constant half2[256]` indexed by a whole byte
  gives its low and high nibble's values in one lookup (half the
  lookups). Prediction: IQ4_XS ≥ 235 / 210 / 235 / 200 GB/s. Measured
  157.8 / 155.9 / 159.7 / 153.9 (−26 %): a 1 KB constant table read at
  divergent indices costs more than the halved count saves. Reverted;
  the lookups' addressing, not their number, is the cost.
- IQ4_XS table by `simd_shuffle` (one table entry per lane): 93.4 /
  92.1 / 94.0 / 86.4 GB/s, a shuffle per code costs far more. Reverted.
- IQ4_XS table in threadgroup memory (16 floats, filled behind one
  barrier; the segment and gathered kernels fill it too). Prediction
  ≥ 220 / 200 / 220 / 195 GB/s. Measured 219.6 / 209.8 / 220.7 / 199.9
  (+2–11 %); the 256-entry `half2` pair table in threadgroup memory read
  205.8 / 193.6 / 206.9 / 186.2 (bank conflicts, not kept). `make speed`
  decode 512 11.35 → 11.78 tok/s (+3.8 %), 4K 10.93 → 11.34 (+3.7 %);
  Gemma 26B-A4B 512 +0.19 %, Muse 512 −0.24 % (noise). **Kept; the
  512 decode prediction (≥ 11.5) is met.**
- (e) Command-buffer split: none needed. Unprofiled 512 decode, 63
  tokens: wall 5,347 ms, GPU busy 5,501 ms (first token included), so
  the GPU never waits on encoding. (a) Metal-allocated weights: the
  capture's MMU limiter is 0.3–2 %, so not tried.
- K-quant group scales through `nu_magic` (two scales per `half2`
  instead of four `float(uint)` conversions). No prediction was written
  before measuring. Q4_K 250.6 / 237.8 / 252.2 / 230.0 GB/s (+25–34 %),
  Q5_K +0.2–0.7 %. `make speed` 512 11.78 → 11.98 (+1.72 %, pairs
  +1.61..+1.73), 4K 11.34 → 11.52 (+1.62 %, +1.55..+1.68): below the
  2 % rule, reverted, patch at `.zig-cache/k23/kscales.patch`. Under the
  amended rule (below) it was measured again: 512 +0.94 % (pairs
  −0.39..+1.73), 4K +1.75 % (+1.55..+2.32), NOISE; then one deciding
  10-pair run, fixed in advance: 512 +1.58 % (−1.01..+5.65), 4K +1.55 %
  (+1.22..+2.09), NOISE. **Reverted.** Every median reads +0.9..+1.8 %,
  and the base binary itself read 11.60–11.78 tok/s across the runs, so
  the machine was noisier late in the session; retry only on a cool
  machine or stacked under a larger change to the same body. Profile:
  the standalone Q4_K ffn_down 3.03 → 2.28 ms, the gate+up segment
  kernel only 33.4 → 32.5 ms, since its tensors are mostly IQ4_XS (61 of
  128 gate/up; Q4_K 28, Q5_K 28, Q3_K 6, IQ4_NL 3 on the generic branch).
- Where the step goes after the kept changes (`--profile`, 512,
  `c151ee2`): matrices 76.1 ms; segment gate+up 33.4 at 194 GB/s,
  DeltaNet projections 13.2 at 182, IQ4_XS ffn_down 6.4 at 179, Q5_K
  6.1 / 4.7 at 220 / 200, Q6_K head 4.2 at 250, attention q/k/v 3.9 at
  181, Q4_K ffn_down 3.0 at 166, IQ4_NL generic 1.4 at 105, IQ3_S + Q3_K
  1.4 at 107–114.
- Keep rule amended (user, 2026-10-03): a 1–2 % gain keeps when every
  pair is faster and within 0.5 % of the median change; `scripts/speed.py`
  prints KEEP for it, development.md § The speed loop states it.

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

**Session 1 delivered** (ledger above): the half magic-number decode in
the Q4_K and Q5_K bodies (shared by the standalone, split, segment, and
gathered matvecs) and IQ4_XS's table in threadgroup memory; (c) was
already the K-quant form, (e) and (a) need no code. **Left from session
1**, each small (about 0.7 % of a step or less), so on a cool machine:
a specialized IQ4_NL matvec (1.4 ms at 105 GB/s on the generic path,
seven tensors, three inside fused segments); Q3_K and IQ3_S at 107–114
GB/s (1.4 ms); (d) wider loads for IQ4_XS (two `uint2` per lane); the
K-scale conversion (`.zig-cache/k23/kscales.patch`, +1.6 % medians,
failed the rule twice); (f) the Q6_K head, already at 250 GB/s, only
for the drafter's per-position cost. IQ4_XS is the remaining large
lever: 61 of 128 gate/up tensors, the fused gate+up kernel at 194 GB/s
for 33 ms of the step. Its capture (apple-gpu.md § `nu_matvec_iq4_xs`
on Qwen's merged gate shape, 2026-10-03) reads it issue-bound on the
integer and conditional pipe (59 % of its instructions: a nibble
extraction and a threadgroup address per code), memory not limiting,
no spill; the candidates are named there. `make verify` (fast Metal
tier) passed 40/40 on `6a0d321` (session 1's kernels, 2026-10-03); it
runs again before the unit closes if session 2 keeps a change.

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
