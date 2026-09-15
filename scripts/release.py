#!/usr/bin/env python3
"""Cut a release from the manifest's development version.

The version is never an argument: the root `build.zig.zon` is the single
source of truth (docs/development.md § Versioning), so the release is that
version with its `-dev` suffix stripped. The package manifests under
`inference/` and `huggingface/` mirror it and are bumped in the same commits.
The script runs `make check`, writes the CHANGELOG section, commits
`chore(release): vX.Y.Z`, creates the annotated tag, then bumps the tree to
the next `X.(Y+1).0-dev` in a follow-up commit. It never pushes: publishing is
a separate action.

    make release            # cut the release
    make release DRY_RUN=1  # print the plan, change nothing

Standard library only.
"""
import argparse
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHANGELOG = ROOT / "CHANGELOG.md"
MANIFEST_PATHS = ("build.zig.zon", "inference/build.zig.zon", "huggingface/build.zig.zon")
MANIFESTS = tuple(ROOT / path for path in MANIFEST_PATHS)
VERSION_RE = re.compile(r'\.version = "([^"]+)"')
DEV_RE = re.compile(r"^(?P<x>\d+)\.(?P<y>\d+)\.(?P<z>\d+)-dev$")


def git(*args):
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True
    ).stdout


def fail(message):
    print(f"release: {message}", file=sys.stderr)
    sys.exit(1)


def manifest_version():
    versions = {read_version(path) for path in MANIFESTS}
    if len(versions) != 1:
        listed = ", ".join(f"{path.relative_to(ROOT)}={read_version(path)}" for path in MANIFESTS)
        fail(f"manifest versions disagree: {listed}")
    return versions.pop()


def read_version(path):
    match = VERSION_RE.search(path.read_text())
    if not match:
        fail(f"no .version in {path.relative_to(ROOT)}")
    return match.group(1)


def set_version(version):
    for path in MANIFESTS:
        path.write_text(VERSION_RE.sub(f'.version = "{version}"', path.read_text(), count=1))


def next_dev(release):
    x, y, _ = (int(part) for part in release.split("."))
    return f"{x}.{y + 1}.0-dev"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    current = manifest_version()
    if not DEV_RE.match(current):
        fail(f"manifest version {current!r} is not MAJOR.MINOR.PATCH-dev")
    release = current[: -len("-dev")]
    tag = f"v{release}"
    following = next_dev(release)

    if git("status", "--porcelain").strip():
        fail("the working tree is dirty; commit or stash first")
    if git("tag", "--list", tag).strip():
        fail(f"tag {tag} already exists; tags never move")
    if CHANGELOG.exists() and re.search(
        rf"^## \[{re.escape(tag)}\]", CHANGELOG.read_text(), re.MULTILINE
    ):
        fail(f"CHANGELOG.md already has a {tag} section")

    print(f"release {release} ({tag}), then begin {following}")

    if args.dry_run:
        print("dry run: nothing checked, written, committed, or tagged")
        return

    print("checking: make check")
    if subprocess.run(["make", "check"], cwd=ROOT).returncode != 0:
        fail("make check failed; nothing was written")

    set_version(release)
    subprocess.run(["python3", "scripts/changelog.py", tag], cwd=ROOT, check=True)

    git("add", *MANIFEST_PATHS, "CHANGELOG.md")
    git("commit", "-m", f"chore(release): {tag}")
    git("tag", "-a", tag, "-m", f"nuclis {tag}")

    set_version(following)
    git("add", *MANIFEST_PATHS)
    git("commit", "-m", f"chore: begin {following}")

    print(f"\ntagged {tag}; the tree is now {following}.")
    print("publish with:")
    print("  git push origin HEAD")
    print(f"  git push origin {tag}")


if __name__ == "__main__":
    main()
