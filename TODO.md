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

REPO-35 is planned and not started: the documents restructured into
`guide/`, `models/`, `engine/`, `app/`, `benchmarks/` with a hub, the log
renamed `worklog.md`, the ADR folder removed, current state first, and a
link checker that also performs the moves. ENGN-21 closed on 2026-10-05
(the benchmarks on macOS 27; the README, `bench.md`, and the site carry
them). Next: session 1 of REPO-35, step 1 (`scripts/docs-check.py`).
Nothing is pushed yet; a push deploys nuclis.dev.

| Unit | What | Sessions |
| --- | --- | --- |
| REPO-35 | The documents, restructured: guide/models/engine/app/benchmarks, a hub, `worklog.md`, no ADRs, current state first, a link checker | 3 |

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
