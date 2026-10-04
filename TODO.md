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

ENGN-21 is planned and not started: re-measure every family against the
pinned llama.cpp on today's machine (macOS 27.0.1, Zig 0.17.0, after
KERN-23's +14 % Qwen decode), at all four acceptance lengths, then
republish the README, `docs/reference/bench.md`, and nuclis.dev. The user
runs it in a fresh session. Next: step 0 of the unit.

| Unit | What | Sessions |
| --- | --- | --- |
| ENGN-21 | Re-measure all families against llama.cpp at 512 / 4K / 16K / 32,639 on macOS 27; republish README, bench.md, nuclis.dev | 2 |

## ENGN-21: the benchmarks, measured again

Base: (set at the first change)

**Why.** The README's main table is from macOS 26.6.2, Zig 0.16.0, and
trees before KERN-23 (Qwen decode +14 % on 2026-10-03); its reference
columns are llama.cpp runs from 2026-09-06 to 09-19 on that older OS. The
speculation table mixes a cooled Qwen session (10-03) with a warm one for
the others (10-01). nuclis.dev copies both tables (`make site-check`
holds it to the README).

**Decisions (user, 2026-10-04).** Full comparison: nuclis *and* the
pinned reference (`7620399`, `.reference/llama.cpp/build`, already built)
re-run on the same machine and OS, at every acceptance length: 512,
4,096, 16,384, 32,639 prompt tokens, 128 greedy output tokens, context
32,768, F16 KV. Rows: Qwen3.8-27B, Gemma 4 12B QAT, Gemma 4 26B-A4B,
Muse Glimmer 30B, and Gemma 4 E4B QAT added. Lengths beyond 32K (the
files declare 128K–256K) are out of scope: no token arrays, reference
runs, or gates exist there; a later unit if wanted.

**Session 1: measure.** Machine on AC, nothing else on the GPU, lid
open, idle 15 min before the first run; note the start state
(`pmset -g batt`, `pmset -g therm`).

0. *Facts first, then rewrite this step list and commit it before any
   run.* Read `docs/reference/reference-baseline.md` (the reference
   harness, `scripts/reference-baseline.py` → `scripts/reference-record.py`),
   `scripts/nuclis-baseline.py`, and `workloads.json`'s `*/acceptance`
   entries. Settle: whether the reference harness can replay the existing
   token arrays (`tests/fixtures/run-*/prompt-<n>.json`) so both dates
   measure identical tokens (preferred), or must regenerate them; how the
   E4B run gets its arrays (no `gemma4-e4b/acceptance` workload or
   `reference-*-gemma4-e4b.json` exists yet; add both); the expected wall
   time per family on each side (Qwen nuclis took 46 min, 26B-A4B 65 min).
1. Build: `make metal`, record `git rev-parse --short HEAD`, `nuclis
   --version`, `zig version` (must equal `~/.local/opt/zig/stable/zig
   version`), `sw_vers`, `system_profiler SPHardwareDataType | grep -E
   'Chip|Memory'`.
2. Reference, family by family, cool-down (10 min idle) between: Qwen,
   Gemma 12B QAT, 26B-A4B, Muse, E4B → `docs/benchmarks/reference-<date>-<family>.json`.
3. nuclis acceptance, same order and cool-downs: `make workload
   NAME=<entry>/acceptance` with `reference_records` pointed at step 2's
   records → `docs/benchmarks/nuclis-<date>-<suffix>.json`; the script
   re-hashes each pinned file after its runs.
4. Speculation pairs in one session from a cooled chip, at each entry's
   catalogue draft length (README: Qwen 7, 12B QAT 5, E4B 6, Muse 6),
   512 and 32,639, plus the 26B-A4B pair that justifies keeping it off;
   reports copied to `docs/benchmarks/speculative-<date>/`.
5. After each family, write its numbers into `docs/reference/bench.md`
   (a new dated section per record, older records kept) and commit, so
   the transcript is never the only copy. Run long jobs in the
   background and keep their output out of the context (`tail`, `grep`).

**Session 2: publish.**

6. README: *Results* becomes two tables, decode and prefill, each with
   the four lengths as `nuclis / llama.cpp`, five rows; the header line
   names machine, OS, Zig, build, reference revision, method. The
   speculation table regenerated. Recheck every prose claim against the
   new numbers: "decodes faster than the reference at every length", the
   other families' gap, "short code reaches 20 tokens/s", "38 % less
   model time" (re-measure with `make agent-eval` or drop the number),
   "26B-A4B stays off".
7. `scripts/site-check.py`: `RESULTS_HEADER` and `readme_results` follow
   the new tables (self-test updated); `site/index.html` `#bench` gains
   the 4,096 and 16,384 columns and the E4B row; `site/site.js` charts
   offer the four lengths (the context toggle becomes four buttons);
   page copy that cites numbers rechecked ("Qwen3.8 is where the tuning
   went"). Screenshots at 1440 and 390 px; push deploys the site.
8. `docs/reference/metal-backend.md` or `architecture.md` § Where
   performance goes, only if a family's gap moved materially.
9. Close: log entry with every record path, the machine state, and the
   deltas against the 2026-09 records.

Gates: `make lint-py`, `make workloads-validate`, `make site-check`,
`make verify-auto`. No inference tier: the unit measures, it changes no
numerical behaviour (a code fix found along the way is its own unit).
