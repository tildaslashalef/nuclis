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

The project moves to pull requests and automated releases. This unit sets
them up on branch `pr-workflow-and-release-please`, the first PR; v0.6.0 is
then cut by merging release-please's first release PR, and archives the
worklog. A second unit (planned below) retires the worklog and the unit
identifiers. Next: step 1 of the PR workflow unit.

| Unit | What | Sessions |
| --- | --- | --- |
| PR workflow and release-please | PRs as the record, squash merges, a ruleset on `main`, release-please for releases, CI that always reports | 1 |
| Retire the worklog and identifiers | after v0.6.0: delete `docs/worklog.md`, IDs out of headings and comments, the rest linked to the archived entry | 1 |

## PR workflow and release-please

Base: `178411f`

**Decisions (user, 2026-10-05).** One unit = one session = one branch = one
PR. The PR description is the record (no worklog entry); squash merge, the
PR title a Conventional Commit and the squash body the PR description. No
GitHub Issues: the queue stays in `TODO.md`. Agents push branches and open
PRs; the user merges. Releases through release-please; v0.6.0 is the
archive point of the worklog.

**Facts.** release-please-action's latest is **v5.0.0** (2026-04-22, node24;
commit `45996ed1f6d02564a971a2fa1b5860e934307cf7`). Release type `simple`
writes `CHANGELOG.md` and a `version.txt` only if one exists
(`createIfMissing: false`), so `build.zig.zon` stays the version's source,
updated through `extra-files` (generic updater, `// x-release-please-version`
on the `.version` line; `build.zig` parses that line up to its closing
quote, so the comment is safe). `draft: true` with `force-tag-creation:
true` creates the tag with a draft release. Events from `GITHUB_TOKEN` start
no workflow: a tag it creates does not trigger `release.yml`'s `push: tags`,
and its release PR gets no CI run.

1. **Versions.** The three `build.zig.zon` read `0.5.0` (the last release)
   with the marker; the `-dev` convention retires. `release-please-config.json`:
   `release-type: simple`, `include-component-in-tag: false`,
   `bump-minor-pre-major: true`, `draft: true`, `force-tag-creation: true`,
   `extra-files` (the three manifests), `changelog-sections` (feat, fix, perf,
   docs, build, refactor shown; chore, test, ci, style hidden),
   `pull-request-title-pattern` `chore: release v${version}`, a PR header
   asking for highlights. `.release-please-manifest.json` `{".": "0.5.0"}`.
2. **Release workflow.** `.github/workflows/release-please.yml` on push to
   `main` (action pinned by SHA; contents, pull-requests, issues write);
   when `release_created`, it calls `release.yml` (`workflow_call`, input
   `tag`, granting `id-token` and `attestations`). `release.yml` loses `push:
   tags` (dispatch stays, for re-runs); it checks out the tag, keeps its
   checks (semver, manifest = tag, compiler, fmt, tests, `--version`),
   builds, packages, attests, then uploads the three assets to the draft,
   appends `.github/release-footer.md` to the draft's notes, verifies three
   assets, and publishes. `.github/release-notes.sh` and CI's notes slice go.
3. **CI that always reports.** `ci.yml` drops `paths-ignore`. A `changes`
   job (ubuntu) decides whether code changed (anything outside `docs/`,
   `site/`, `*.md`); the macOS matrix runs only then. A `docs` job (ubuntu,
   standard library only): `docs-check`, `site-check`, `gates-validate`,
   `workloads-validate` with their self-tests. A final `ok` job needs all and
   fails if any failed or was cancelled; `ok` is the required check.
4. **The record's shape.** `.github/pull_request_template.md`: what and why,
   evidence (local gates and their output, records), remaining, a checklist.
5. **Retired.** `scripts/release.py`, `scripts/changelog.py`, `make release`,
   `make changelog`; `CHANGELOG.md`'s intro says how it is written now.
6. **Rules.** AGENTS.md: the session protocol (branch, plan in `TODO.md` on
   it, PR, record, user merges), commits (push branches and open or update
   PRs authorized; never push or merge `main`), versioning and releases
   (release-please), no new unit identifiers (the next unit retires the old
   ones); `docs/development.md` § Continuous integration and releases and §
   Versioning follow.
7. **Repository settings, the user's to approve** (one `gh api` call each,
   prepared in the PR body): squash merges only, title from the PR title and
   body from the PR description, branches deleted on merge; a ruleset on
   `main`: PRs required (0 approvals), `ok` required, no force push or
   deletion, the admin role may bypass (the release PR, opened by
   `GITHUB_TOKEN`, gets no CI run; a token for release-please would lift
   that, a later choice).
8. **The PR.** Push the branch, open the PR with the record; after the user
   merges, release-please opens the v0.6.0 release PR.

Gates: `make docs-check`, `make site-check`, `make lint-py`, `make
gates-validate`, `zig build` (the manifests), `actionlint` if installed.

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
