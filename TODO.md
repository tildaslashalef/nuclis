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

**Next: REPO-29, the Zig 0.17.0 upgrade** (added 2026-10-03; its
section is before KERN-23's). The installed compiler is 0.17.0 and the
tree does not configure under it, so it goes first. **Then KERN-23
session 1.** Before any KERN-23 change: run `make speed-base` on the
clean tree after REPO-29 (built with Zig 0.17), then record the unit's `Base:` at the first change. There is no
base binary now: `.zig-cache/` was cleaned after MODL-34 (2026-10-02),
and MODL-34 changed `qwen35_metal.zig` anyway (the shape is read from
the file; Qwen3.8's arithmetic and gates unchanged). The cleanup also
took the 16K and 32K saved prefixes under `.zig-cache/speed/prefix/`
(512 and 4K were rebuilt by the gates): the first `make speed` or
`qwen38-verify-depth-16k`/`-32k` run at those depths prefills them again,
about 3 and 11 min. MODL-34 closed 2026-10-02 in one
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
| 1 | REPO-29 — Upgrade to Zig 0.17.0 (blocks everything: the tree does not build) | 1 |
| 2 | KERN-23 — Weight streaming for one row and a few (closes the decode-speed theme) | 2 |
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

## REPO-29 — Upgrade to Zig 0.17.0 (1 session)

Base: recorded when the unit starts.

**Why first.** The installed compiler is 0.17.0 (`~/.local/opt/zig/stable`,
0.16.0 removed), and the tree does not configure under it, so no gate, no
`make speed-base`, and no KERN-23 experiment can run until this lands. A
compiler upgrade is its own unit (AGENTS.md § Versioning).
[Release notes](https://ziglang.org/download/0.17.0/release-notes.html).

**Probe (2026-10-03, scratch worktree at `fd09aa4`, discarded).** Errors
surface one class at a time; fixing each exposed the next. Not yet seen:
anything past `@typeInfo` (the build stopped there). The probe's throwaway
diff is in `.zig-cache/zig017/probe.patch` (not committed; its string
helper `__rep` is a hack, do not apply it as is).

1. **`b.args` removed** (configure). 15 sites in `build.zig`,
   `inference/build.zig`, `huggingface/build.zig`:
   `if (b.args) |args| run.addArgs(args);` → `run.addPassthruArgs();`.
   `zig build <step> -- ARGS` and the Makefile/`gates.json` callers keep
   working unchanged; check one (`make shot` or `zig build test-vocabulary --
   …`) passes its arguments.
2. **Array multiplication `**` removed** (parse). 77 sites, 18+ files.
   - `[_]T{v} ** n` (array fill, ~40, mostly tests in `inference/src/quant/decode.zig`,
     `backends/cpu/*.zig`, `generation-check.zig`, the two `failures` arrays in
     `backends/cpu/root.zig:125` and `vision/muse_glimmer.zig:399`, `bpe.zig:55`,
     `inspect.zig:164`) → `var x: [n]T = @splat(v);` (or `@as([n]T, @splat(v))`
     in expressions); `qwen35.zig:425` nested → `[64][N]bool = @splat(@splat(false))`.
   - `"s" ** n` (string repetition, ~35: `src/tui/{banner,editor,markdown,status}.zig`,
     `src/help.zig:42`, `src/api/http.zig:236-238`, `src/api/decisions/service.zig:395,469`,
     `inference/src/profiles/muse_glimmer.zig:531,938`). Single-byte ones become
     `@as([n]u8, @splat(' '))` (`&` where a slice is needed); multi-byte
     (`"─"`, `"line\n"`, `"<|patch|>"`, `"\"s\","`) need one comptime
     `repeat(comptime s, comptime n) *const [s.len * n]u8` helper. Put it in
     one place each package can reach (`inference/src/` text utilities and a
     `src/` test helper, or `src/` only if inference's two sites take `@splat`/
     a literal); no per-file copies. `src/tui/markdown.zig:926` repeats in a
     comptime loop.
   - Markdown test strings containing `**` inside literals are not operators;
     leave them.
3. **Removed std names** (sema): `EnumSet.initEmpty()` → `.empty`
   (`src/model.zig` ×5, `inference/src/models/modernbert.zig:88`);
   `std.ascii.indexOfIgnoreCase` → `findIgnoreCase` (`src/tui/theme.zig:474,487`).
4. **`@typeInfo` is struct-of-arrays** (sema). 33 sites. `.fields` is gone:
   enums have `field_names`/`field_values`, structs and unions
   `field_names`/`field_types`/`field_attrs` (defaults via
   `field_attrs[i].defaultValue(T)`). `inline for (info.field_names,
   info.field_types) |name, T|` replaces `|field| field.name / field.type`;
   `.fields.len` → `.field_names.len`. Heaviest file: `src/config.zig`
   (12 sites, including default values); then `src/tui/{theme,event}.zig`,
   `src/help.zig`, `src/completion.zig`, `inference/src/sampling/root.zig`,
   `inference/src/backends/metal/root.zig:59` (the kernel table),
   `gemma4_runtime.zig:134`, `vision/gemma4.zig:181`, `registry.zig:173`,
   `src/agent/session.zig:331`, `src/bench.zig:384`.
5. **Then** keep building (`zig build`, `zig build -Dmetal=true`, `zig build
   test`, the check tools) until clean; record each new class here.

**Silent changes to audit** (no compile error):
- `@bitCast` on arrays/vectors changed meaning: grep found only scalar
  casts (52 sites, `f32/f16 ↔ u32/u16`, `i8 ↔ u8`), unaffected. Re-grep after
  the fixes for any `@Vector`/array operand.
- `@hasDecl` is now true only for `pub` declarations, also in the same file.
  19 sites, all capability probes in `inference/src/engine.zig`,
  `registry.zig:67`, `generation-check.zig:317,2043` (`prefillRows`,
  `verify`, `replayRows`, `drafter`, `bindDraft`, `embedded_draft`,
  `preferredChunk`, …). A private decl now silently drops a path (e.g. a
  family losing speculation). Check each probed name is `pub` on every
  family's `Runtime`/`Plan`; the fast tier's speculative and vision gates
  confirm.
- `mem.eql` on float slices no longer short-circuits on identical pointers
  (no float `mem.eql` found).

**Deprecations, migrate in this unit** (removed in 0.18; cheap now):
`std.fmt.allocPrint(a, …)` → `a.print(…)` (157 sites, mechanical);
`@intFromEnum`/`@enumFromInt` → `@backingInt`/`@fromBackingInt` (47 sites;
`zig fmt` rewrites them, check the diff); `std.DynamicBitSetUnmanaged` →
`std.bit_set.Dynamic` (`sampling/root.zig:107`, `tokenizer/hf_json.zig:308`);
any `std.builtin` / `@import("builtin").os|cpu` (none found). Run `zig build`
with no deprecation warnings left. `std.posix` terminal/signal calls and
`std.fs.path` still exist in 0.17; development.md's earlier migration list
stays deferred.

**Docs.** `build.zig.zon` ×3 `minimum_zig_version = "0.17.0"`;
`docs/development.md § Toolchain` (tested compiler, release-notes link) and
replace *Zig 0.16 usage audit* with a 0.17 one (what was migrated, what is
deferred); `AGENTS.md:8` and `README.md:24` (0.17). Leave benchmark records
and README:78's measured line as they are: they name the compiler they ran
with. ZLS does not work with 0.17 yet (release notes, Build Server
Protocol); note it in development.md.

**Gates.** `make check`; `make lint-py` only if scripts change; `make
verify-auto`; `make verify` (fast Metal tier: the compiler, LLVM 22, and the
Objective-C bridge's clang all changed); `make verify-cpu` once (LLVM 22
recompiles every CPU reference kernel; ~25 min). Happy path:
`./zig-out/bin/nuclis --version`, a `nuclis chat` screenshot through `make
shot`, and `nuclis serve` + one `/v1/models` request. Speed: no 0.16 base
binary exists, so either fetch 0.16.0 into `.zig-cache/zig016/`, build `fd09aa4`
with it as the `make speed` base, and record the A/B at 512 and 4K
(decode and 4-row verify C) in bench.md; or record only the 0.17 numbers as
KERN-23's fresh `make speed-base`. Recommend the A/B (LLVM 22 still has loop
vectorization disabled; a host-side regression would otherwise be charged
to KERN-23).

**Lands when** the tree builds and every gate above passes under 0.17.0 with
no deprecation left from the list, and `make speed-base` has been taken with
the 0.17 binary for KERN-23.

## KERN-23 — Weight streaming for one row and a few: decode and the verify body (2 sessions)

Base: recorded when the unit starts.

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
