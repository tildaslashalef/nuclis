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

**Next: REPO-33, track A (CI speed)**, then B (release and changelog),
then C (the README GIF); the user may add tracks until they say to close.
REPO-32 (2026-10-04) fixed the failed v0.5.0 release: `install-zig.sh`
takes the first matching digest, and `release.yml` runs the workflow
revision's installer. v0.5.0 published by dispatch from `main` (run
`37209513321`, success; the assets verified against `SHA256SUMS`, the
binary reports `nuclis 0.5.0`); REPO-33's log entry records this, since
REPO-32's entry predates it. Track A is committed (`80791df`) and is
measured on the runners after it is pushed.

| # | Unit | Sessions |
| --- | --- | ---: |
| 1 | REPO-33 — CI speed, release notes that lead with the work, the README GIF at speculative speed, and what the user adds | open |

## REPO-33 — CI speed, release notes that lead with the work, the README GIF at speculative speed, and what the user adds (open)

Base: `037ff9d`

Agreed with the user 2026-10-04 as **one flexible unit**: further requests
during it become new tracks below, and it closes only when the user says
so. Each track commits on its own (`REPO-33` in the subject); the log
entry is written once, at the close. No inference code changes are
planned, so no Metal or CPU tier applies; `make verify-auto` before each
commit, `make lint-py` for Python.

### Track A — CI speed

**Facts.** `ci.yml` runs on every push to `main` and every PR, one job on
`macos-15`. Run `37152348051` (warm-ish, 5 m 57 s wall): install Zig 25 s
(cache miss), `zig build test` 2 m 07 s, `-Dmetal=true -Doptimize=ReleaseSafe`
1 m 18 s, `-Dmetal=false -Doptimize=ReleaseSafe` 1 m 22 s, post-cache 8 s.
Of the last 60 commits, 25 touched only `docs/`, `*.md`, or `TODO.md`.
The cache key hashes all of `build.zig.zon`, which every release bump
changes, and `restore-keys: zig-<os>-<arch>-` can never prefix-match the
key `zig-<hash>-<os>-<arch>`.

**Do.**
1. `paths-ignore` on `push` and `pull_request`: `docs/**`, `**/*.md`
   (README, TODO, CHANGELOG, AGENTS, notices). `workflow_dispatch` stays.
2. Three parallel jobs instead of one: `test` (fmt check, semver check,
   `zig build test`), `metal` (Metal build, `--version`/`--help`/
   `agent --help`), `cpu` (`-Dmetal=false` build, `--version`). Wall time
   becomes the slowest job plus setup.
3. Optimize mode: measure locally `zig build -Dmetal=true` Debug vs
   ReleaseSafe from a clean `.zig-cache` (`time`, M4 Pro, note it is not the
   runner). The ci builds only prove compile and link; `release.yml` keeps
   ReleaseSafe. Choose by the numbers; record them in the log.
   *Measured 2026-10-04, M4 Pro, empty local and global caches
   (`ZIG_GLOBAL_CACHE_DIR`; 0.17 has no `--global-cache-dir`):
   `-Dmetal=true` Debug 43 s, ReleaseSafe 83 s.*
4. Cache: toolchain key from `.github/zig-toolchain` plus the
   `minimum_zig_version` value (not the whole manifest); a separate
   `actions/cache` for `.zig-cache` and `~/.cache/zig` keyed per job on
   `hashFiles('**/*.zig', '**/build.zig.zon')` with a working prefix
   `restore-keys`. Keep it only if a warm run is measurably faster.
   *First run, 2026-10-04 (`37210607506`, `4f5d9ed`, every cache cold):
   success, wall 4 m 02 s (old layout 5 m 57 s): `test` 2 m 28 s, `cpu`
   2 m 19 s, `metal` 4 m 00 s (its ReleaseSafe build 3 m 31 s with no
   warm global cache; the old serial job built it after the tests had
   warmed it). Warm run (`37211094389`, `82d21db`): wall 2 m 16 s
   (−62 %): `test` 1 m 17 s, `cpu` 44 s, `metal` 2 m 16 s; the toolchain
   cache hit (install 0 s). `metal` stays the bound because `build.zig`
   embeds `git rev-parse HEAD` in the build options, so the executable
   recompiles on every commit whatever the cache holds; a Debug `metal`
   build would roughly halve it (locally 43 s vs 83 s cold), if the user
   wants it.*
5. Evidence: `gh run view <id> --json jobs` step times, a docs-only push
   that does not trigger, a code push before and after. Update
   `docs/development.md` § Continuous integration and releases and the
   header comment of `ci.yml`.

### Track B — release and changelog

**Facts.** `make release` → `scripts/release.py` (`make check`, strip
`-dev`, `scripts/changelog.py <tag>`, commit `chore(release): vX.Y.Z`,
annotated tag, bump to `X.(Y+1).0-dev`; never pushes).
`scripts/changelog.py` lists every commit since the previous tag under
Breaking / Features / Bug Fixes / Performance / Other, one line per
commit; v0.5.0's section has 24 lines for two agent units and two CI fixes.
`.github/release-notes.sh` slices the tag's section for the GitHub notes.
The release path is only exercised by a real tag (REPO-32's lesson).

**Do.**
1. Research how open-source projects write release notes (Keep a
   Changelog, git-cliff, release-please, changesets; Zig, Rust, Bun, Deno,
   llama.cpp releases) and report in chat with a recommendation before
   changing the format.
   *Researched 2026-10-04.* Keep a Changelog: written for people, and
   it lists "commit log diffs" as an anti-pattern. The common pattern
   (Ollama, Zig, Rust's blog plus RELEASES.md, Neovim's `news.txt`):
   a short hand-written layer stored outside the commits, above a
   generated or edited list, ending with a compare link. git-cliff
   offers `commit_parsers` and `link_parsers`, and the annotated tag
   message `{{ message }}` as the highlights; release-please needs
   PRs; changesets and towncrier need a fragment file per change,
   which the engineering log already plays the part of. GitHub's
   generated notes list merged PRs, so they show nothing in a
   push-only repository. Robustness: run the dry run in CI on every
   push; publish as a draft and attach every asset before publishing
   (immutable releases, GA 2025-10, refuse assets added later); build
   provenance with `actions/attest` (`id-token: write`,
   `attestations: write`; users check it with
   `gh attestation verify`). Recommendation: stay on the stdlib
   script, no new tool.
2. Likely direction (confirm with the user): a hand-written
   **Highlights** paragraph per release, then one bullet per closed unit
   (`AREA-NN` from commit subjects, title from the engineering log's
   table row, linked to its log anchor), then fixes outside units,
   breaking changes flagged; docs/chore/test commits collapse into a
   compare link (`/compare/vA...vB`). Where the highlights text lives and
   how `release.py` asks for it is decided with the user.
   *Decided by the user 2026-10-04:* highlights drafted at release
   (`release.py` opens `$VISUAL`/`$EDITOR` on a template listing the
   units and breaking changes, refuses empty text, and uses the text as
   the annotated tag message; `--highlights-file` for non-interactive
   runs); section = highlights, units (log heading title, linked to the
   anchor at the tag), breaking, outside units, every commit in
   `<details>`, compare link. The release workflow publishes a draft
   first, then flips it, and attests the tarballs with
   `actions/attest-build-provenance` v4.2.2
   (`4d101475d8b20a2381f78447822ac1eab6504dd8`).
   *Then, 2026-10-04:* the whole CHANGELOG is revamped to this format,
   v0.1.0–v0.5.0, each with highlights drafted from the log and approved
   by the user (e.g. v0.4.0 leads with the Zig 0.17 upgrade), and each
   published release's notes replaced with `gh release edit` (tags never
   move). Units are grouped by area (Models, Engine, Kernels, Agent,
   Application, Terminal, Repository); a unit counts for the release whose
   log first lists it (`--ref <tag>` reads the log and plan at the tag);
   follow-ups and in-progress units are marked; a unit named only in a
   commit and nowhere in the plan or log (REPO-11, dropped) is left out.
3. Make the release path testable before a tag: a `make release
   DRY_RUN=1` that prints the generated section and the notes
   `release-notes.sh` would publish; a ci step that runs
   `release-notes.sh` on the newest changelog section and the
   installer's digest lookup on a duplicated pin file.
4. Ask whether to regenerate v0.5.0's notes in the new format
   (`gh release edit`; the tag does not move).
5. Update `docs/development.md` § Versioning and AGENTS.md's release
   recipe if the command surface changes.

### Track C — the README GIF at speculative speed

**Facts.** `scripts/agent-demo.py` records `nuclis agent` in tmux on a
scratch clone of the playground (`scripts/playground.py`) with
`rect_perimeter` broken, streams an asciicast to
`.zig-cache/demo/agent.cast`, and renders `docs/media/agent.gif` with
`agg` (`--speed 2 --idle-time-limit 1.5`). The current GIF (4.4 MB,
2026-09-26) predates speculation by default; with it Qwen decodes at
about 17 tok/s. README lines 10–12 state "2×" and "2 min 35 s".

**Do.**
1. Confirm the script still drives the agent after AGNT-20's command
   surface (the `◆ ready` marker, Ctrl-C exit) and that the agent's
   default model runs with its draft; fix the script if not.
2. `zig build -Dmetal=true -Doptimize=ReleaseFast`, then
   `python3 scripts/agent-demo.py`; read the cast's last timestamp for
   the real session time. Try `--speed 1.5` or `1` if the real session is
   short enough; keep the GIF near or under 5 MB.
3. Look at frames before committing (the agent fixes the bug, tests
   pass). Update the README caption (speed, real time, speculative
   decoding) and log the session time and size.

*Delivered 2026-10-04:* `make build` (ReleaseSafe, `0.6.0-dev`), the
script unchanged but for its default `--speed 1`; Qwen's catalogue entry
has speculation on at draft 7. Cast 86.1 s: warm-up 23.0 s (1447
tokens), task submitted at 34.8 s, the answer at 70.6 s (about 36 s,
seven steps: read README, `make test`, `find`, read `rect.py`, the
edit, `make test`); status bar 21–27 tok/s. Frames read: the fix is
`2 * (width + height)`, 5 tests OK. GIF at 1× 84.7 s, 2.43 MB (2×
would be 44.3 s, 2.32 MB; the old one 4.39 MB). README caption updated.

### Track D — branding

The user added `docs/branding/` (2026-10-04): `nuclis.svg` (512×512 viewBox,
teal `#087F8C` bands, an amber diamond) and `nuclis.png` (1254×1254 RGBA,
transparent). The PNG heads the README (centred, `width="160"`). Later,
when the user asks: icon sets generated from the SVG for the nuclis.dev
website.

### Further tracks

Requests the user adds during the unit are appended here as tracks E, F…
with the same level of detail.
