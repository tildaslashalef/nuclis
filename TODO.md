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

v0.6.0 is published (2026-10-05). The release helper is in progress on
branch `release-helper`; the worklog's retirement follows.

| Unit | What | Sessions |
| --- | --- | --- |
| Release helper | `make release-draft`, `make release V=`, `make tag V=`: the material for the highlights, the version pull request, the signed tag | 1 |
| Retire the worklog and identifiers | delete `docs/worklog.md` (archived at `v0.6.0`), IDs out of headings and comments, the rest linked to the archived entry | 1 |

## Release helper

Base: `f23f44a`

**Why.** v0.6.0 was cut by hand: the merged pull requests read with `gh`,
the highlights drafted from them, the version pull request, the tag. The
user wants the agent to do it on "release", with the highlights drafted for
their review.

1. `scripts/release.py` (standard library; `git` and `gh`), `--self-test`:
   - `draft`: the pull requests merged into `main` since the last tag
     (their merge commits in `<tag>..origin/main`), each with its title and
     the first paragraph of its *What and why*; the suggested next version
     (0.x: a breaking or `feat` title bumps the minor, otherwise the
     patch); where the highlights go (`.zig-cache/highlights-vX.Y.Z.txt`).
   - `open X.Y.Z --tiers "<what ran>"`: refuses a dirty tree, a `main`
     behind `origin/main`, a version not above the current one, an existing
     tag, or a missing highlights file; then the branch `release-vX.Y.Z`,
     `make version`, the commit `chore(release): vX.Y.Z`, the push, and the
     pull request with the highlights and the tiers in its description.
   - `tag X.Y.Z`: refuses unless on `main` at `origin/main`, the manifest
     says `X.Y.Z`, the highlights file is not empty, and the tag is new
     locally and on the remote; then `git tag -a -F` (signed when git signs
     tags), `git push origin vX.Y.Z`, and the release run's link.
2. `make release-draft`, `make release V= TIERS=`, `make tag V=`.
3. AGENTS.md § Versioning and releases: "release" means draft, show the
   highlights, open the version pull request, stop; merging and tagging
   only on the user's word. `docs/development.md` § Versioning: the
   commands, and what makes good highlights.

Gates: `make lint-py`, the script's self-test, `make docs-check`, `make
verify-auto`; `draft` run against the real history (v0.5.0..v0.6.0).

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
