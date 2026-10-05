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

ENGN-21 session 1 is done (2026-10-04 20:01 → 10-05 04:54): every
family measured on both engines at the four lengths, three requests each,
and the speculation pairs; all records under `docs/benchmarks/`
(`reference-2026-10-04-*`, `nuclis-2026-10-04.json`,
`nuclis-2026-10-05-*`, `speculative-2026-10-05/`), every number in
`bench.md` § The benchmarks on macOS 27 with its caveats (the 4K step on
the slow dense models drifts as the chip heats; one slow 16K request on
each Gemma run; Muse's 4K reference re-run). Next: session 2, step 7.
The site's new numbers section (four lengths, a coloured table grouped
by phase, E4B, today's numbers) is merged into `main` (`f6caef7`), so
`make site-check` fails (31 problems) until step 7 gives the README the
same tables: do not push before then (a push deploys the site). Step 8
is left with `site-check.py`'s header and parser, its self-test, and the
page copy.

| Unit | What | Sessions |
| --- | --- | --- |
| ENGN-21 | Re-measure all families against llama.cpp at 512 / 4K / 16K / 32,639 on macOS 27; republish README, bench.md, nuclis.dev | 2 |
| REPO-35 | The documents, restructured: guide/models/engine/app/benchmarks, a hub, `worklog.md`, no ADRs, current state first, a link checker | 3 |

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

*Session 1 delivered* steps 1–6 as planned, plus: the reference takes
the first record holding a length (`nuclis-baseline.py`
`reference_rows`), so Muse's 4K re-run replaces that row; records crossing
midnight are dated 2026-10-05.

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
:320 "Setup guide" → `development.md`, :348–351; `site/llms.txt`
:11–19), and a newcomer lands in month-old lab notebooks: `bench.md`
(2,375 lines) puts its current answer last, `development.md` opens with
the author's machine, the hub `docs/README.md` is linked from nowhere
and misses 12 documents, `docs/reference/` is 26 files with no index and
no grouping, Qwen (the first target) has no document of its own, and the
same 2026-09-10 Qwen numbers sit in eight files. Survey of 2026-10-04
(an agent, read-only); its file:line findings are folded in below.

**Decisions (user, 2026-10-04).**
- Restructure now, once, rather than live with the layout: the tree
  below. Old paths get no stubs.
- `docs/engineering-log.md` → `docs/worklog.md`.
- `docs/adr/` removed with every reference (ADR 0001 was implemented
  with changes; no unit followed the convention); its durable knowledge
  moves first (step 3). AGENTS.md drops ADRs.
- The log's *content* stays append-only, but one mechanical pass may
  repoint its link paths and anchors to the new tree (yes, 2026-10-04).
- `llm-guide.md` may be edited in this unit: TOC, §27's level, the
  stale number (yes, 2026-10-04).
- The log's summary table links each identifier to its entry (yes,
  2026-10-04); new rows are written linked from then on.

**The tree.** One rule per folder, so a new document has one home:

```
docs/
  README.md            the hub: every document, one line each
  spec.md              requirements (stays)
  architecture.md      the map of the engine and app (stays)
  development.md       contributing: toolchain, gates, the record, CI, releases
  llm-guide.md         the companion on the inference stack (stays)
  worklog.md           ← engineering-log.md
  guide/               using nuclis
    getting-started.md   new: install, ~/.nuclis, model pull, first chat (the site's "Setup guide")
    configuration.md     ← development.md § User directories, the config file
    api.md               ← reference/api.md (nuclis serve)
    eval.md              ← reference/eval.md (nuclis eval)
  models/              one file per family, plus the catalogue
    README.md            ← reference/new-model-guide.md (adding a model) + the family list
    catalogue.md         ← reference/artifacts.md (sources, pins, the models directory)
    qwen3.8.md           ← reference/qwen-validation.md, gathering the Qwen facts now in
                           generation, tokenizer, gguf-inspection, prompt-profile
    gemma4.md  muse-glimmer.md  bonsai.md  laya.md    ← reference/<same>
    clef-flash.md        ← reference/clef.md
  engine/              the inference/ package, one file per subsystem
    gguf.md              ← reference/gguf-inspection.md
    safetensors.md  tokenizer.md  quantization.md  cpu-reference.md
    metal-backend.md  apple-gpu.md  session.md  prompt-profile.md
    tool-calling.md  vision.md  speculative-decoding.md    ← reference/<same>
    sampling.md          ← reference/generation.md
  app/                 the src/ package
    agent.md             ← reference/agent-concepts.md
    terminal.md          ← development.md :744–1006 (styled output, live region, transcript, renderer)
  benchmarks/          the record: JSON records stay where scripts write them
    README.md            ← bench.md's method, definitions, current results, and an index of the records
    history.md           ← bench.md's dated sections, text unchanged
    llama-cpp.md         ← reference/reference-baseline.md (the oracle and its harness)
  media/  branding/    (stay)
```

`docs/reference/` disappears. Costs measured by the survey: gemma4 62
references in 20 files (`gates.json`, six `inference/*.zig`), bench 164
(log 58, `workloads.json` 28), metal-backend 116 (log 47),
reference-baseline 44 (17 benchmark JSON, 4 scripts), muse-glimmer 31,
bonsai 39, laya 30, generation 28. Mechanical, with a checker.

**Session 1: the checker and the moves** (paths and links only; no prose).

1. `scripts/docs-check.py` + `make docs-check`, standard library, with
   `--self-test`:
   - *check*: every relative Markdown link and anchor in tracked `*.md`;
     the `docs/…` paths (with `#anchors`) in `*.json`, `*.py`, `*.zig`,
     `Makefile`, `.github/`; the `blob/main/…` links in `site/`. GitHub
     anchor rule: lowercase, punctuation stripped except `-`, spaces →
     hyphens, `-1`, `-2` for repeats. `CHANGELOG.md`'s `blob/v0.x/` links
     pin tags and are skipped.
   - *move*: `--move old=new[,old=new…]` does `git mv` and rewrites every
     reference above, re-relativizing Markdown links from each file's
     own directory (moved files' outgoing links too); `--move
     old#anchor=new#anchor` rewrites anchors when sections move between
     files. Reused for every later rename.
   - Joins `verify-auto`'s model-free checks for `*.md` and `site/` paths.
   First run reports today's breakage; fix it: `session.md#the-agents-token-cache`
   (development :530, spec ×2, session), `session.md#row-checkpoints-engn-14`
   (speculative-decoding), `vision.md#the-qwen3-vl-projector-modl-21`
   (`gates.json` ×2), and in the log `roadmap.md` ×2, `agent-spec.md` ×2,
   `architecture.md#9-adding-a-model` (→ §10): repoint or unlink.
2. The worklog: `--move docs/engineering-log.md=docs/worklog.md`; the
   prose mentions too (AGENTS.md :59, :391, development :28, :1108,
   bench :1061, new-model-guide :215, prompt-profile :134).
   `scripts/changelog.py` (:6, :26 `LOG`): `closed_units(previous)` reads
   the log *at the previous tag*, where `worklog.md` does not exist —
   fall back to `docs/engineering-log.md` there, with a self-test case,
   or the first release after the rename breaks.
   Then the table: each `| AREA-NN |` row's identifier becomes
   `[AREA-NN](#<its heading's anchor>)`, generated from the headings
   (`HEADING_RE`), checked by `make docs-check`. `changelog.py`'s
   `ROW_RE` (:31) accepts both the bare and the linked form (older tags
   hold bare rows), with self-test cases for both; AGENTS.md *Closing a
   unit* says the new row is linked.
3. The ADR, moved then removed. Into `engine/speculative-decoding.md` §
   "The Qwen verify budget", condensed: the rule C ≤ 50E and its
   derivation (ADR :178–280; elsewhere only in `scripts/spec-matrix.py`'s
   docstring and the log), the external evidence surveyed 2026-09-30
   (:91–117: MLX `sdpa_vector`, metal-flash-attention, llama.cpp #29110,
   vLLM #58863, DFlash2, SuffixDecoding, GDN Tree-Scan), the alternatives
   considered (:348–361). Into `engine/metal-backend.md`: where the
   verify cost goes (:78–90) and the implementation observations
   (:293–326: verify misses the merged gate/up and SiLU fusions; the
   8-column tile runs 3–4 real rows). Drop Confidence. Repoint bench
   :1728, :2007, speculative-decoding :973, apple-gpu :155, the log's
   :6356 link (to the new section); delete `docs/adr/`; AGENTS.md :39–41,
   :71, :391 and `docs/README.md` :41–47 lose the ADR text.
4. The tree, by `--move`, one commit per folder (engine, models, app +
   guide, benchmarks), `make docs-check` clean after each: the plain
   renames of the tree above. `bench.md` moves whole to
   `benchmarks/README.md` here; its split is step 7.
5. The hub: `docs/README.md` rewritten from the tree (one line per
   document, grouped as the folders), "for Qwen3.8-27B" (:3) gone;
   linked from `README.md` (:199, which links `docs/reference/` today),
   `site/index.html` (a "Docs" link), `site/llms.txt`. AGENTS.md :405–406
   and `development.md` :1431 agree: documents are listed in
   `docs/README.md`. AGENTS.md :359 drops `docs/roadmap.md`; its
   *Repository architecture* and other path mentions follow the tree.
   Gates: `make docs-check`, `make site-check`, `make lint-py`,
   `make verify-auto`, `zig build test` (the `.zig` comments changed).

**Session 2: the splits** (content moves; history kept, below it).

6. `development.md` → `guide/getting-started.md` (new: install, `~/.nuclis`,
   `nuclis model pull` from :498, a first chat), `guide/configuration.md`
   (:573…), `app/terminal.md` (:744–1006), each by `--move
   development.md#… = …` for the anchors (31 inbound, 13 distinct, 6 from
   the log). What stays is the contributor guide, with a TOC. The site's
   "Setup guide" (:320) points at `guide/getting-started.md`.
7. `benchmarks/README.md`: definitions, what to record, the acceptance
   method, *Current results* (the ENGN-21 tables, the speculative
   defaults table now at bench :2296, and the README speculation table's
   source, today only in the log at KERN-23), and an index of the JSON
   records by family and date. Every dated section, *Observations so
   far* (:183–505) first among them, moves to `benchmarks/history.md`
   with its heading text unchanged, by `--move` with anchors (24 cited,
   117 links). The `<!-- bench:… -->` region (bench :1557) moves with its
   section and `scripts/bench-report.py --write-doc` follows it.
8. `models/qwen3.8.md`: the Qwen facts (architecture, hybrid schedule,
   tokenizer, prompt profile, validation) gathered from qwen-validation,
   generation, tokenizer, gguf, prompt-profile, as `gemma4.md` is for
   Gemma; the sources keep a link.

**Session 3: current state first.**

9. Leads and TOCs on the long documents: `engine/metal-backend.md`
   (a current-state lead like speculative-decoding :8–24; :7–8's
   "2026-09-07 review" resolved; "Performance observation" :1778 labelled
   2026-09-07, macOS 26), `llm-guide.md` (TOC; §27 at :1031 is H2 among
   H3s), `models/gemma4.md`, `engine/vision.md`.
10. One home per number: README and the site keep the headline tables;
    `architecture.md` :518–534 (stale flowchart; "the switch lost" —
    ENGN-20 turned it on), `development.md` :85–90, `llm-guide.md` :33,
    `models/laya.md` :27, `benchmarks/llama-cpp.md` :223 link to
    `benchmarks/README.md` instead of copying.
11. Stale text: `engine/gguf.md` :3, `models/qwen3.8.md` (qwen-validation :4),
    `engine/sampling.md` :1 ("bring-up"), :11–12, `engine/cpu-reference.md`
    :308–313, `benchmarks/llama-cpp.md` :4–5 ("nuclis does not execute
    models yet"), `models/muse-glimmer.md` :444–445 (the projector
    shipped), `models/bonsai.md` :342–355 (catalogue entry removed
    2026-09-26), `guide/api.md` :372 ("Zig 0.16.0"), development :14
    (eval shipped), :44 (macOS 26), :389, :407. Duplicated unit records
    (KERN-13/14/15/16/18 in metal-backend and history; ENGN-12/14/15/16/20,
    MODL-19/20 in speculative-decoding and history): keep the topic
    document's, reduce the history's body to a line and a link.
12. Close: the log entry lists every move (old → new) and removal; check
    the log renders on GitHub at ~500 KB.

Gates: `make docs-check` (new), `make site-check`, `make lint-py`,
`make verify-auto`, `zig build test` when `.zig` comments change.
Documents, scripts, and comments only; no inference tier. The site's
changed pages are looked at by the user.
