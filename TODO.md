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

Releases move from release-please to tags (user, 2026-10-05: lean and
stable, nothing third-party). This unit is in progress on branch
`tag-driven-releases`; then v0.6.0 is cut the new way (a version PR, then
an annotated tag), and the unit after it retires the worklog.

| Unit | What | Sessions |
| --- | --- | --- |
| Tag-driven releases | release-please removed; `release.yml` on an annotated tag: its message is the highlights, GitHub generates the list of merged pull requests | 1 |
| Retire the worklog and identifiers | after v0.6.0: delete `docs/worklog.md`, IDs out of headings and comments, the rest linked to the archived entry | 1 |

## Tag-driven releases

Base: `d1d4588`

**Why.** release-please's release PR, opened by `github-actions`, needs a
manual approval to run CI (or the admin bypass), regenerates its CHANGELOG
over hand-written highlights, and is an external action with majors to
follow. A tag carries the same information with nothing to maintain.

1. Remove `release-please-config.json`, `.release-please-manifest.json`,
   `.github/workflows/release-please.yml`, and the `// x-release-please-version`
   markers (the manifests stay at `0.5.0`).
2. `make version V=X.Y.Z`: set the three `build.zig.zon` versions,
   refusing a non-semver value.
3. `release.yml` on `push: tags: v*` (dispatch kept): refuses a lightweight
   tag (no highlights), then as before (semver, manifest = tag, compiler,
   gate, build, `--version`, attest); creates a draft whose notes are the
   tag's message, GitHub's generated list of merged pull requests since
   the previous release (`releases/generate-notes`), and
   `.github/release-footer.md`; uploads the three assets; publishes.
4. `CHANGELOG.md` keeps v0.5.0 and earlier; its intro points to the
   Releases page from v0.6.0 on.
5. AGENTS.md § Versioning and releases, `docs/development.md` §
   Continuous integration and releases and § Versioning.
6. Settings (approved with this choice): the ruleset's admin bypass
   removed (`.github/ruleset-main.json`), Actions no longer allowed to open
   pull requests, and a tag ruleset (`.github/ruleset-tags.json`): `v*`
   tags cannot be deleted or moved.

Gates: `actionlint`, `make docs-check`, `make site-check`, `zig build`,
`make verify-auto`.

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
