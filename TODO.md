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

ENGN-21 session 1 is under way: step 0's facts are settled and the step
list below is rewritten from them. Next: step 1 (the harness changes),
then the runs. The machine was checked on 2026-10-04 19:00: AC, no
thermal warning, GPU 16–23 % (the display only), `sudo purge` done, the
idle opencode server stopped, display sleep blocked by Amphetamine.

| Unit | What | Sessions |
| --- | --- | --- |
| ENGN-21 | Re-measure all families against llama.cpp at 512 / 4K / 16K / 32,639 on macOS 27; republish README, bench.md, nuclis.dev | 2 |

## ENGN-21: the benchmarks, measured again

Base: `78e7ec4`

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
open, display sleep off; note the start state (`pmset -g batt`,
`pmset -g therm`) in each record (the harnesses capture `pmset`).

*Facts (step 0, 2026-10-04).*
- nuclis replays committed arrays (`scripts/nuclis-baseline.py`,
  `load_reference_prompt`); the reference harness rebuilds them through
  the server every run. The rebuild is deterministic for Gemma (12B,
  12B QAT, 26B-A4B arrays are byte-identical) but not for Muse: its
  template carries the server's date. So the reference must **replay**
  the committed arrays (step 1a).
- Qwen's arrays: `run-2026-09-06/` holds 512–16,384 and
  `boundary-2026-09-06/` the 32,639 one (equal to the Bonsai run's).
- E4B's template (`241c50d8…`) differs from the 12B's: thinking off
  renders no empty `<|channel>thought\n<channel|>`, so the harness's
  `gemma4` check fails on it and the 12B arrays are not E4B's prompts
  (the 2026-10-01 speculation rows used them anyway). E4B gets its own
  arrays from the harness (no replay) under a new family `gemma4-e`.
  The pinned reference runs E4B (MODL-27 traces and perplexity).
- `reference-record.py` names records `reference-<date>-<family>`: the
  three Gemma files collide, so pass `--output` with the workload suffix.
- Repetitions (user, 2026-10-04): one warmup and **three** measured
  requests at every length on both sides, 32,639 included (before: the
  Qwen reference and every nuclis boundary run took one).
- Wall time from the last records (reference / nuclis, minutes, three
  at 32K): Qwen 53 / 69, 12B QAT 25 / 45, 26B-A4B 10 / 23, Muse 48 / 61,
  E4B ~10 / ~15; with 10-minute cool-downs about 7 h. A run that crosses
  midnight dates its record by its start; that is fine.

1. Harness changes, committed before any run (gates: `make lint-py`,
   `make workloads-validate`):
   a. `scripts/reference-baseline.py --replay <dir>[,<dir>…]` (fixture
      run directories under `tests/fixtures/`, the first holding a size
      wins): sends those `prompt-<n>.json` arrays instead of building
      them; still checks the server's revision, slot, template marker,
      and smoke; fails unless the server tokenizes the corpus into the
      fixture's body tokens; saves the fixture's
      `prompt-construction.json` plus `replayed_from` and whether the
      server's own prefix/suffix tokens equal it (false only for Muse).
   b. Family `gemma4-e` in `reference-baseline.py` (`FAMILIES`: BOS
      `<bos>`, marker `<turn|>\n<|turn>model\n`, and `<|think|>` must be
      absent) and `nuclis-baseline.py` (`THINKING_OFF`, same rule).
   c. `nuclis-baseline.py --boundary-repetitions` default 3.
   d. `workloads.json`: `gemma4-e4b/acceptance` (model `gemma4_e4b`, run
      `run-2026-10-04-gemma4-e4b`, suffix `gemma4-e4b`); every
      acceptance entry's `reference_records` → the new records (Qwen's
      becomes one file, `reference-<date>-qwen38.json`).
2. Build: `make metal`; record `git rev-parse --short HEAD`, `nuclis
   --version`, `zig version` (= `~/.local/opt/zig/stable/zig version`),
   `sw_vers`, `system_profiler SPHardwareDataType | grep -E 'Chip|Memory'`.
3. Reference, in order Qwen, 12B QAT, 26B-A4B, Muse, E4B, 10 min idle
   between: start `llama-server` with the flags of
   `reference-baseline.md` § Run the workload on the family's file
   (`gates.json` `models`) in the background, then
   `python3 scripts/reference-baseline.py --family <f>
   --prompt-lengths 512,4096,16384,32639 --server-pid <pid>
   --output-dir .reference/<date>-<suffix> --replay <runs>` (E4B: no
   `--replay`), stop the server, then `python3 scripts/reference-record.py
   .reference/<date>-<suffix> --family <f> --output
   docs/benchmarks/reference-<date>-<suffix>.json`. Replay runs: Qwen
   `run-2026-09-06,boundary-2026-09-06`; 12B QAT
   `run-2026-09-12-gemma4-qat`; 26B-A4B `run-2026-09-18-gemma4-26b-a4b`;
   Muse `run-2026-09-19-muse-glimmer`. E4B's output directory's
   `prompt-*.json`, `prompt-construction.json`, and the files the other
   runs keep are copied to `tests/fixtures/run-2026-10-04-gemma4-e4b/`
   (and `tests/fixtures/provenance.md` gains its line).
4. nuclis, same order and cool-downs: `make workload
   NAME=<entry>/acceptance` → `docs/benchmarks/nuclis-<date>[-<suffix>].json`;
   the script re-hashes each pinned file after its runs.
5. Speculation pairs in one sitting from a cooled chip: `make
   spec-matrix ARGS='--model <key> --contexts 512,32639 --drafts <n>
   --sampling greedy --cooldown 90'` at the catalogue draft (Qwen 7, 12B
   QAT 5, E4B 6, Muse 6; 26B-A4B 4, the pair that keeps it off); the
   arrays are each family's acceptance run (`speed.family_run`, so E4B's
   own once 1d lands); `--report --json` copied to
   `docs/benchmarks/speculative-<date>/<key>.json`.
6. After each family, its numbers go into `docs/reference/bench.md` (a
   new dated section per record, older records kept) and are committed.
   Long jobs run in the background; their output stays out of the
   context (`tail`, `grep`).

**Session 2: publish.**

7. README: *Results* becomes two tables, decode and prefill, each with
   the four lengths as `nuclis / llama.cpp`, five rows; the header line
   names machine, OS, Zig, build, reference revision, method. The
   speculation table regenerated. Recheck every prose claim against the
   new numbers: "decodes faster than the reference at every length", the
   other families' gap, "short code reaches 20 tokens/s", "38 % less
   model time" (re-measure with `make agent-eval` or drop the number),
   "26B-A4B stays off".
8. `scripts/site-check.py`: `RESULTS_HEADER` and `readme_results` follow
   the new tables (self-test updated); `site/index.html` `#bench` gains
   the 4,096 and 16,384 columns and the E4B row; `site/site.js` charts
   offer the four lengths (the context toggle becomes four buttons);
   page copy that cites numbers rechecked ("Qwen3.8 is where the tuning
   went"). Screenshots at 1440 and 390 px; push deploys the site.
9. `docs/reference/metal-backend.md` or `architecture.md` § Where
   performance goes, only if a family's gap moved materially.
10. Close: log entry with every record path, the machine state, and the
   deltas against the 2026-09 records.

Gates: `make lint-py`, `make workloads-validate`, `make site-check`,
`make verify-auto`. No inference tier: the unit measures, it changes no
numerical behaviour (a code fix found along the way is its own unit).
