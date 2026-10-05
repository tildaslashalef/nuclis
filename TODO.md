# TODO — active plan

This file is the only progress tracker. It holds **unfinished work only**:
where we are, what is next, and the design of each remaining unit. When a
unit closes, its outcome moves to
[docs/worklog.md](docs/worklog.md) and
its section is deleted here; when the last unit closes, this file is emptied
back to this header. Requirements live in [docs/spec.md](docs/spec.md); the
engine map in [docs/architecture.md](docs/architecture.md); how to build,
test, and measure in [docs/development.md](docs/development.md).

Session protocol (also in [AGENTS.md](AGENTS.md)): read this file first. If
it lists work, summarize *Where we are* and ask the user how to continue. If
it is empty, ask what to work on and write the agreed plan here.

## Where we are

The worklog's retirement is in progress on branch `retire-worklog`, the
last unit queued.

| Unit | What | Sessions |
| --- | --- | --- |
| Retire the worklog and identifiers | delete `docs/worklog.md` (archived at `v0.6.0`), IDs out of headings and comments, the rest linked to the archived entry | 1 |

## Retire the worklog and identifiers

Base: `a95c4a1`

`ARCHIVE` = `https://github.com/tildaslashalef/nuclis/blob/v0.6.0/docs/worklog.md`.
Each identifier's anchor and close date come from the table of `git show
v0.6.0:docs/worklog.md` (`| [ID](#anchor) | title | date |`).

1. `docs-check` also checks `blob/<tag>/<path>#anchor` links whose tag
   exists locally: the path at that tag, and the anchor in it.
2. Headings (88): `(ID, …)` → `(…)`, `(ID)` → `(<close date>)`,
   `(ID session N, …)` → `(session N, …)`; the six with the ID in their
   text get a name (`The KERN-13 quick pass` → `The penalty-kernel quick
   pass`, ENGN-15 → sampled acceptance, ENGN-16 → proposal policy, `KERN-05
   — per-block cost research` → `Per-block cost research`, `(KERN-10
   session 2, …)`, `(AGNT-01 session 1)`). New anchors are checked unique
   per file, and every reference follows through `docs-check --move`
   anchor moves.
3. Prose (about 440 lines in `*.md`): an identifier outside code and link
   text becomes `[ID](ARCHIVE#anchor)`; links into `worklog.md` become
   `ARCHIVE#anchor`.
4. Code comments (12 files) and the notes in `gates.json` and
   `workloads.json`: the identifier removed, the sentence kept true.
5. `docs/worklog.md` deleted; the hub, README, `spec.md`,
   `development.md`, AGENTS.md § No unit identifiers, the site's
   "Worklog" link, and `llms.txt` point at the merged pull requests and
   the archive.

Gates: `make docs-check`, `make site-check`, `make lint-py`, `zig build
test` (comments), `make verify-auto`.
