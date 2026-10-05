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

Releases are tag-driven (the pull request for it is in review, branch
`tag-driven-releases`). Next: v0.6.0, cut the new way: a version pull
request (`make version V=0.6.0`, the highlights drafted in its description),
then the user's annotated tag on its merge; it archives the worklog at the
tag. Then the unit below.

| Unit | What | Sessions |
| --- | --- | --- |
| Retire the worklog and identifiers | after v0.6.0: delete `docs/worklog.md`, IDs out of headings and comments, the rest linked to the archived entry | 1 |

## Retire the worklog and identifiers

After v0.6.0 is published. `docs/worklog.md` deleted (it stays at the
`v0.6.0` tag); its links repointed to
`https://github.com/tildaslashalef/nuclis/blob/v0.6.0/docs/worklog.md#<anchor>`.
Headings: `(AREA-NN, date)` → `(date)`, `(AREA-NN)` alone → the unit's
close date from the log's table, through `docs-check --move` anchor moves
(88 headings). Code comments: identifiers removed (13 files). Remaining
prose mentions (about 500, mostly `benchmarks/history.md`,
`engine/metal-backend.md`, `engine/speculative-decoding.md`): linked to the
archived entry. `gates.json` and `workloads.json` evidence anchors follow.
AGENTS.md's area table goes; the site's "Worklog" link becomes the
releases and merged PRs.
