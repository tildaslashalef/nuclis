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

**Next: REPO-29 session 2** (the Zig 0.17.0 upgrade; its section is
before KERN-23's). Session 1 (2026-10-03) landed the migration: the tree
builds, `make check` and `make verify-auto` pass under 0.17. Session 2:
the 0.17 features, the speed sanity check, `make verify`, `make
verify-cpu`, docs, close. **Then KERN-23
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
| 1 | REPO-29 — Upgrade to Zig 0.17.0, adopting its features, with a speed sanity check against the 0.16 records (blocks everything: the tree does not build) | 2 |
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

## REPO-29 — Upgrade to Zig 0.17.0 (2 sessions)

Base: `8f3e6ed`

**Sessions.** 1: the tree builds and tests clean under 0.17 (removals,
silent changes, deprecations), `make check` and `make verify-auto` pass,
committed. 2: the 0.17 features below, the speed sanity check, `make verify`,
`make verify-cpu`, docs, close.

**Why first.** The installed compiler is 0.17.0 (`~/.local/opt/zig/stable`,
managed by `zigup`; 0.16.0 removed), and the tree does not configure under it, so no gate, no
`make speed-base`, and no KERN-23 experiment can run until this lands. A
compiler upgrade is its own unit (AGENTS.md § Versioning).
[Release notes](https://ziglang.org/download/0.17.0/release-notes.html).

**Session 1 (2026-10-03): the tree builds and tests clean under 0.17.**
Classes met, in the order they surfaced (each fix exposed the next):

1. **`b.args` removed:** 15 sites → `run.addPassthruArgs()`.
2. **Array `**` removed:** fills → `var x: [n]T = @splat(v)` (38 by regex,
   3 by hand: the nested `qwen35.zig` slot table, a `&@as([8]u8, …)`
   argument, `inspect.zig`'s struct fill); single-byte strings in
   `help.zig`/`status.zig` → `@splat(' ')`; multi-byte and test strings →
   `repeat(s, n)` from the new `src/text.zig` (an `inline fn`, so `++`
   sees a comptime operand); inference's two sites became a literal and an
   `@splat` array. `markdown.zig`'s 64-level comptime loop raises its
   branch quota.
3. **Removed std names:** `EnumSet.initEmpty()` → `.empty`,
   `indexOfIgnoreCase` → `findIgnoreCase`.
4. **`@typeInfo` struct-of-arrays:** `.fields` → `field_names` /
   `field_types` / `field_values`; no site read default values.
   `std.meta.fieldNames/fieldTypes` are deprecated, and a non-inline call is
   not comptime-known as an `inline for` operand in a runtime function, so
   `config.zig` uses `@typeInfo(T).@"struct".field_names` directly.
5. **`Allocator.dupeZ` removed** → `dupeSentinel(u8, s, 0)` (3 sites).
6. **`zig build --global-cache-dir` removed** → `ZIG_GLOBAL_CACHE_DIR`:
   the Makefile exports it (`$(CURDIR)/.zig-cache/global`), `gates.py`
   sets it from `gates.json`'s `build.cache` in `load()`; the reference
   docs' commands dropped the flag; development.md says so.
7. **`std.fmt.bufPrintZ` removed** → `bufPrintSentinel(…, 0)`
   (`metal-check.zig`).
8. **`std.testing.allocator` is a `SafeAllocator`**, which grows a block in
   place only while it ends its bucket, so `checkAllAllocationFailures`
   saw different allocation counts per run (`NondeterministicMemoryUsage`
   in Laya's open and Muse's prompt rendering). New
   `inference/src/alloc_check.zig` (`checkAll`, `noGrowth`: resize/remap
   refused) backs all 22 inference callers and `laya-check.zig`.
   Huggingface's two callers pass and stay on std (the package cannot
   import inference); if one turns nondeterministic, give it the same
   wrapper.

Deprecations migrated: `allocPrint`/`allocPrintSentinel` → `a.print` /
`a.printSentinel` (157); `@intFromEnum`/`@enumFromInt` → `@backingInt` /
`@fromBackingInt` (`zig fmt` rewrote them, adding `@intCast` where the
backing type must match; literal ones simplified); `DynamicBitSetUnmanaged`
→ `bit_set.Dynamic`; `builtin.os` → `builtin.target.os` (2). No other name
from the release notes' deprecation list is used (deprecations are doc
comments only; the compiler does not warn).

Silent changes audited: `@hasDecl` probes (19) all name `pub` declarations
on every family (the non-`pub` `replayRows`/`prefill`/`verify` in
`generation-check.zig` and `engine.zig` are wrappers, not probe targets);
the 52 `@bitCast`s are scalar; float `mem.eql` has 18 sites (the probe
missed them), all comparing distinct buffers, so unaffected.

Gates: `make check` passes (645/645 unit tests, `test-metal`, the
manifests); `make verify-auto` passes (diff since `8f3e6ed`): every check and the 40 fast-tier gates it selected (all families' traces, generation, vision, speculative, draft, verify-depth 512/4K, perplexity; so the `@hasDecl` paths and `addPassthruArgs` are exercised). It flags `verify-cpu` and `verify-long` for `cpu/attention.zig`, whose change is test-array syntax only; `verify-cpu` runs in session 2 anyway (LLVM 22). `zig build -Dmetal=false` and
`hf-downloader` build.

**Use what 0.17 adds** (not only what it removes). Each item names its
sites; adopt it where it reads better, and record in the development.md
audit what was adopted, what was judged not to apply, and why.
- **Configure cache (build system).** `build.zig:19` reads
  `build.zig.zon` at configure time through `std.Io.Dir.cwd()`, a
  dependency 0.17's configure cache cannot see: after `make release` bumps
  the version, `--version` could report the old one from a cached
  configuration. Read it through `b.path("build.zig.zon")` and declare
  `b.dependOnFileContents(b.path("build.zig.zon"))`; prove it by bumping
  the version in a scratch edit and checking `./zig-out/bin/nuclis
  --version` changes without `rm -rf .zig-cache`. Then run `zig build
  --cache-poison=disallowed` (and with `-Dmetal=true`) to show no other
  step poisons the cache; fix any that does (`findProgramLazy` instead of
  `findProgram`, `dependOn*` for files read while configuring).
- **Build API.** `addPassthruArgs` (above); `b.pathList` if a `Fmt` step
  is added; `--print-configuration` worth a line in development.md for
  inspecting the graph.
- **`@divCeil`.** The `(a + b - 1) / b` idiom, 26 sites:
  `inference/src/backends/metal/root.zig` (13, grid sizes),
  `src/tui/transcript.zig` (6), `vision/muse_glimmer.zig` (2),
  `metal-check.zig`, `src/{decide,discover}.zig`, `src/tui/{diff,editor}.zig`
  (1 each). Find them with `git grep -nE '\+ [a-z_.0-9()]+ - 1\) / '`;
  convert only true ceiling divisions of non-negative integers.
- **`Allocator.print`, `@backingInt`/`@fromBackingInt`, `@splat`** (the
  migrations above are the adoption).
- **`ArrayList.last()` / `lastPtr()`.** `items[list.items.len - 1]`, 10
  sites (`git grep -nE 'items\[[a-z_.]*items\.len - 1\]'`).
- **`std.bit_set.Dynamic`, `std.bit_set.Integer/Array/Static`** names, and
  `.empty`/`.full` decls instead of `init*` functions.
- **`std.heap.SafeAllocator`.** `std.process.Init.gpa` in Debug builds is
  now a `SafeAllocator` (thread-safe, never reuses memory, catches
  double-frees and cross-instance frees) without code changes. Run the
  `serve` happy path (the batcher and pool threads) and `make check` in
  Debug to exercise it, and note in development.md that leak and misuse
  reports come from it.
- **Comptime-length slices coerce to array pointers.** Simplifies the
  `[i * 4 ..][0..4]` readers only where the length is comptime; check, do
  not force.
- **`std.Io.Semaphore.waitTimeout`.** Evaluate for
  `src/api/decisions/batcher.zig:107` (`wait` with a deadline); adopt only if
  it removes code.
- **`zig fmt --complexity`.** Report the token/node change for the touched
  files in the log entry (before/after), as the release notes suggest.
- **Not applicable, say so in the audit:** incremental compilation
  (`-fincremental --watch` targets x86_64-linux; the Mach-O linker and
  aarch64 backend are not ready), `std.zon.parse` (no ZON parsing beyond the
  version line), `@SpirvType`, translate-c (no `@cImport`; the Metal bridge is
  Objective-C compiled by `addCSourceFile`), Build Server Protocol (no
  tooling consumes it yet; ZLS does not work with 0.17).

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
shot`, and `nuclis serve` + one `/v1/models` request.

**Speed sanity check against the records (re-decided 2026-10-03, user).**
No interleaved 0.16 → 0.17 A/B: the Metal kernels are compiled at run time
by Apple's compiler, so Zig/LLVM only reach host time (encoding, the
Objective-C bridge, sampling), a thin slice of a ~95 ms step. Instead:
`make speed-base` with the 0.17 binary (KERN-23's base), then `make speed
ARGS='--contexts 512,4096 --verify-rows 1,4 --model qwen38'` (base and
tree are the same build, so both sides are 0.17 readings, ≥ 5 pairs).
Compare against the latest 0.16 records in
[bench.md](docs/reference/bench.md) § The DeltaNet replay tape (ENGN-19):
plain decode 10.57 / 10.20 tok/s at 512 / 4K, 4-row (accept 1) C
177.44 / 190.75 ms. Within 3 % (run-to-run spread ~1.5 %, plus not
interleaved): record the readings and compiler in the log entry, done.
Beyond 3 %: `--profile` to tell host time from GPU time, and only then
`zigup install 0.16.0` and a real interleaved A/B on a scratch worktree at
`fd09aa4`, before the unit closes, not charged to KERN-23.

**Landed early (2026-10-03, before session 1):** AGENTS.md § Local
toolchain notes records the `zigup` layout and the local std/langref;
development.md's paths point at `stable`. The log entry includes it.

**Lands when** the tree builds and every gate above passes under 0.17.0,
no deprecation from the list is left, the features above are adopted or
recorded as not applicable, the speed sanity check is within 3 % of the
records (or a gap is explained), and `make speed-base` has been taken
with the 0.17 binary for KERN-23.

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
