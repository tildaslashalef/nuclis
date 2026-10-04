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
| REPO-35 | The documents, restructured: a docs hub, `worklog.md`, no ADRs, consistent names, current state first, a link checker | 2 |

Order: ENGN-21 first (its session 2 rewrites README, `bench.md`, and the
site); REPO-35 starts once it closes, from a tree that holds today's numbers.

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

## REPO-35: the documents, restructured

Base: (set at the first change)

**Why.** nuclis.dev links into `docs/` (`site/index.html` :203 bench,
:320 "Setup guide" → `development.md`, :348–351 architecture, llm-guide,
log, AGENTS; `site/llms.txt` :11–19). A newcomer lands in month-old lab
notebooks: `bench.md` (2,375 lines) puts its current answer last,
`development.md` opens with the author's machine, `docs/README.md` (the
hub) is linked from nowhere and misses 12 documents, and the same
2026-09-10 Qwen numbers sit in eight files. Survey of 2026-10-04 (an
agent, read-only); its findings by file:line are folded into the steps.

**Decisions (user, 2026-10-04).** `docs/engineering-log.md` becomes
`docs/worklog.md`. `docs/adr/` is removed with every reference: ADR 0001
(the Qwen decode verifier, *proposed*) was implemented with changes and no
unit followed the ADR convention; its durable knowledge moves to
`docs/reference/speculative-decoding.md` first. AGENTS.md drops ADRs from
the session protocol.

**Constraints.** The log stays append-only in content; renaming it and
repointing paths inside it is a mechanical pass, not a rewrite (step 3,
user to confirm). `CHANGELOG.md`'s 162 log links pin release tags
(`blob/v0.x/…`) and stay valid; leave them. Headings that other files
cite keep their text (anchors): `bench.md` has 24 cited anchors, 117
links (31 from the log, 28 from `workloads.json`); `metal-backend.md` 66;
`development.md` 31. Reorder sections, never retitle a cited one.

**Session 1: mechanics** (paths, links, removals; no prose rewrite).

1. `scripts/docs-check.py` + `make docs-check`, standard library: every
   relative Markdown link and anchor in `*.md`, the `docs/...md#…`
   strings in `workloads.json` / `gates.json` / `docs/benchmarks/*.json`,
   and the repo links in `site/index.html` and `site/llms.txt` resolve
   (GitHub anchor rule: lowercase, punctuation stripped, spaces →
   hyphens, `-1` suffixes for repeats); `--self-test`. Known breakage it
   must report today: `session.md#the-agents-token-cache` (development,
   spec ×2, session), `session.md#row-checkpoints-engn-14`
   (speculative-decoding), `vision.md#the-qwen3-vl-projector-modl-21`
   (gates.json ×2). Fix those; record the log's own dead links
   (`roadmap.md`, `agent-spec.md`, `architecture.md#9-adding-a-model`)
   as an allow-list the check reads. Add it to `verify-auto`'s
   model-free checks for `*.md` paths.
2. ADR removal: move ADR 0001's durable parts, condensed, before
   deleting it: the budget rule C ≤ 50E and its derivation (ADR :178–280;
   elsewhere only in `scripts/spec-matrix.py`'s docstring and the log),
   the external evidence surveyed 2026-09-30 (:91–117: MLX
   `sdpa_vector`, metal-flash-attention, llama.cpp #29110, vLLM #58863,
   DFlash2, SuffixDecoding, GDN Tree-Scan) and the alternatives
   considered (:348–361) → a `speculative-decoding.md` section "The Qwen
   verify budget"; where the verify cost goes (:78–90) and the
   implementation observations (:293–326: verify misses the merged
   gate/up and SiLU fusions; the 8-column tile runs 3–4 real rows) →
   `metal-backend.md`. Its Confidence section is plan material; drop it.
   Repoint `bench.md` :1728 and :2007, `speculative-decoding.md` :973,
   `apple-gpu.md` :155;
   delete `docs/adr/`; AGENTS.md :39 (session protocol), :71 (durable
   knowledge), :391 (unit identifiers) lose the ADR clauses;
   `docs/README.md` :43 too.
3. `git mv docs/engineering-log.md docs/worklog.md`; repoint
   `.github/workflows/ci.yml`, AGENTS.md (4), `development.md` (6),
   `llm-guide.md`, `docs/README.md`, `bench.md`, `eval.md`, `session.md`,
   `speculative-decoding.md`, `vision.md`, `spec.md`, `README.md`,
   `scripts/changelog.py` (:6, :26 `LOG`; `closed_units(previous)` reads
   the log *at the previous tag*, where `worklog.md` does not exist: fall
   back to `docs/engineering-log.md` there, with a self-test case, or the
   first release after the rename breaks), `site/index.html` (:350),
   `site/llms.txt`, `TODO.md`. Inside the log, only its links to the
   removed ADR and to renamed files (step 4) are repointed — ask the user
   before this pass; the alternative is leaving them dead and
   allow-listed.
4. Names. Ask the user first: a `docs/models/` folder for the per-model
   documents (gemma4 62 refs / 20 files incl. `gates.json` and six
   `inference/*.zig`, muse-glimmer 31/19, bonsai 39/22, laya 30/16,
   `clef.md` → `clef-flash.md` 13/9, a new `qwen3.8.md` gathering the
   Qwen facts now spread over qwen-validation, generation, tokenizer,
   gguf-inspection, prompt-profile; `new-model-guide.md` → its README),
   or the flat folder below. Either way `bench.md` (164 refs) and
   `metal-backend.md` (116) keep their paths. The flat option renames
   only the misfits, each with its inbound links: `qwen-validation.md` → `qwen3.8.md` (like `gemma4.md`,
   `muse-glimmer.md`, `bonsai.md`), `gguf-inspection.md` → `gguf.md`,
   `reference-baseline.md` → `llama-cpp.md` (the oracle and its harness).
   Count each rename's references before it (`grep -rI`), run
   `make docs-check` after.
5. The hub: `docs/README.md` lists every document, one line each,
   grouped (start here; using nuclis; how it works; per model; the
   record), and drops "for Qwen3.8-27B" (:3); `docs/reference/README.md`
   (GitHub shows it under the bare file list) points back to it, and a
   `docs/benchmarks/README.md` indexes the records by family and date
   without renaming them (scripts write those paths). Linked
   from `README.md` (:199), `site/index.html` (a "Docs" link), and
   `site/llms.txt`. AGENTS.md :405–406 and `development.md` :1431 agree:
   documents are listed in `docs/README.md`. AGENTS.md :359 drops
   `docs/roadmap.md`.

**Session 2: content** (current state first; history kept, below it).

6. `bench.md`: after *Definitions*, a *Current results* section (the
   ENGN-21 tables, the speculative defaults table now at :2296, the
   README speculation table's source, which today lives only in the
   log at KERN-23) and a table of contents; *Observations so far*
   (:183–505, 2026-09-07 bring-up) moves under a *History* heading with
   the dated sections, text unchanged. Where a dated section duplicates
   a topic document's record (KERN-13/14/15/16/18 in
   `metal-backend.md`; ENGN-12/14/15/16/20, MODL-19/20 in
   `speculative-decoding.md`), keep its heading and replace the body by
   one line and a link, or keep both — decide on reading them.
7. `metal-backend.md`: a current-state lead and TOC, as
   `speculative-decoding.md` :8–24 does; :7–8's "2026-09-07 review"
   pointer resolved; "Performance observation" (:1778) labelled as
   2026-09-07 on macOS 26.
8. `development.md` split: a newcomer `docs/getting-started.md` (install,
   `~/.nuclis`, configuration from :498, `nuclis model pull` from :573)
   becomes the site's "Setup guide" target; the agent UI internals
   (:744–1006: styled output, live region, transcript, renderer) move to
   a `docs/reference/terminal.md`; cited headings leave a one-line stub.
   Stale lines: :14 (eval shipped, APPS-14), :44 (macOS 26), :85–90
   (copied numbers → link), :389, :407.
9. One home per number: README and the site keep the headline tables;
   `architecture.md` :518–534 (stale flowchart, "the switch lost" — ENGN-20
   turned it on), `development.md` :85–90, `llm-guide.md` :33 (ask: the
   guide changes only on request), `laya.md` :27, `reference-baseline`
   (now `llama-cpp.md`) :223 link to `bench.md` instead of copying.
10. Bring-up documents, current-state lead or fold: `gguf.md` :3,
    `qwen3.8.md` :4, `generation.md` :1 ("bring-up"), :11–12, `cpu-reference.md`
    :308–313, `llama-cpp.md` :4–5 ("nuclis does not execute models yet"),
    `muse-glimmer.md` :444–445 (the projector shipped), `bonsai.md` :342
    (the catalogue entry was removed 2026-09-26), `api.md` :372
    ("Zig 0.16.0").
11. Protected documents, navigation only, with the user's yes: a TOC for
    `llm-guide.md` and its §27 heading level (:1031, H2 among H3s); the
    log's summary table (:12–177) linking each row to its entry; check
    the log renders on GitHub at ~500 KB.
12. Close: the log entry lists every rename and removal; `make
    docs-check`, `make site-check`, `make lint-py`, `make verify-auto`.

Gates: `make docs-check` (new), `make site-check`, `make lint-py`,
`make verify-auto`. Documents and scripts only; no inference tier. The
site's changed pages are looked at by the user.
