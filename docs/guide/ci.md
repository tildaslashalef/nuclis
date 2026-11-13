<!-- kind: guide -->
# Continuous integration and releases

What CI checks, what it cannot, and how `release.yml` builds and publishes a
tag. The release steps an agent runs are
[development.md § Versioning](../development.md#versioning).

## Continuous integration and releases

Work reaches `main` only through pull requests, merged by the user (a
ruleset on `main` requires a pull request and the `ok` check, allows
only rebase merges, and refuses force pushes and deletion;
`.github/ruleset-main.json` is its tracked copy, and the repository's
settings allow no squash or merge commits either). Every pull request is
merged by rebase: each of its commits is one unit, a Conventional Commit
whose body is the record of that unit
(`.github/pull_request_template.md` gives the record's shape), so `git log
main` holds every record. A small unit outside any milestone is a pull
request of its one commit. CI therefore runs once
per milestone, and the gate between units is local: `make check`,
`make verify-auto`, and `make build-linux`, the release's Linux target
cross-compiled into `.zig-cache/linux` (it proves the tree compiles there;
the Linux tests are CI's)
([AGENTS.md § The session protocol](../../AGENTS.md#the-session-protocol)).

Two workflows under `.github/workflows/`. The build jobs install the
compiler with `.github/install-zig.sh`: the version comes from
`minimum_zig_version` in `build.zig.zon` and the SHA-256 from
`.github/zig-toolchain`, verified before anything is unpacked. No
marketplace action installs the toolchain and no digest is fetched at run
time; the tree pins a digest for every artifact it downloads. A Zig upgrade
edits the manifest, that file, and [code.md § Toolchain](code.md#toolchain) in one unit
of work. Every action is pinned by full commit SHA rather than a moving
tag, and checkouts keep `persist-credentials` off.

**`ci.yml`** (every push to `main`, every pull request) always reports:

| Job | Runs | What |
| --- | --- | --- |
| `changes` | always (ubuntu) | whether anything outside `docs/`, `site/`, and Markdown changed |
| `docs` | always (ubuntu) | `make docs-check site-check workloads-validate`, standard-library Python |
| `test` | code changed (macOS) | `zig fmt --check`, a semver check on the manifest's version, `make input-safety`, `zig build test` (the fuzz corpora included), `make gates-validate` |
| `metal` | code changed (macOS) | a `-Dmetal=true -Doptimize=ReleaseSafe` build, `--version` and `--help` on it |
| `cpu` | code changed (macOS) | a Debug `-Dmetal=false` build, so the non-Metal path keeps linking |
| `linux` | code changed (ubuntu) | `zig build test -Dmetal=false` and a Debug build, `--version` on it |
| `ok` | always | passes when every job passed or was skipped: the check `main` requires |

That is `make check` minus the GPU. The toolchain cache is keyed on the
compiler version and its pins; each build job caches `.zig-cache` and Zig's
global cache per commit, restoring the newest. **Not** in CI: `test-metal`,
`verify`, `bench`, and anything that pulls a model. They need the pinned
artifacts and the real M4 Pro, and a rate measured on a virtualised GPU is
a number nobody should trust; those gates run locally and their results go
into the pull request's description.

**`release.yml`** (an annotated `v*` tag pushed by the user, or a dispatch
for a tag whose release is still a draft) refuses to publish, before it
builds anything, unless:

1. the tag is `vMAJOR.MINOR.PATCH[-prerelease]`;
2. the tag is annotated with a non-empty message (the highlights);
3. the tag equals `build.zig.zon`'s `.version`;
4. the installed `zig version` equals `minimum_zig_version`;
5. the tag's release, if one exists, is still a draft.

The Zig version is the tag's, but `install-zig.sh` is the workflow
revision's and the digests are the tag's `.github/zig-toolchain` plus the
workflow revision's (the first matching line wins). It runs CI's gate
(`make ci-test`, the target CI's test job runs), builds `-Dmetal=true
-Doptimize=ReleaseSafe` and, cross-compiled beside it, the CPU-only Linux
binary (`make build-linux LINUX_OPT=ReleaseSafe`, [development.md § Linux](../development.md#linux)), asserts the macOS binary
reports the tag's version and the Linux one carries the same `nuclis
X.Y.Z` string (it cannot run on the runner), attests the tarballs
(`actions/attest-build-provenance`; check one with `gh attestation verify
<file> -R tildaslashalef/nuclis`), and creates a draft release with four
assets:
the binary tarballs (`nuclis-vX.Y.Z-aarch64-macos.tar.gz`,
`nuclis-vX.Y.Z-x86_64-linux.tar.gz`), a source
tarball (`nuclis-vX.Y.Z-src.tar.gz`, `git archive` of the tag), and a
`SHA256SUMS` covering them. The notes are the tag's message, the
Conventional Commit subjects since the previous tag grouped by type
(`scripts/repo/release.py notes`, the workflow revision's copy: milestones
merge by rebase, so each unit is one subject), and
`.github/release-footer.md`. It publishes
only once the asset count checks, so a failed upload never leaves a public
release missing files.

The binary is **unsigned and not notarized** (there is no Apple Developer ID
for this project), so macOS quarantines it on download; the notes say to run
`xattr -d com.apple.quarantine nuclis`, and building from source stays one
`zig build` away.
