#!/usr/bin/env python3
"""Cut a release from the manifest's development version.

The version is never an argument: the root `build.zig.zon` is the single
source of truth (docs/development.md § Versioning), so the release is that
version with its `-dev` suffix stripped. The package manifests under
`inference/` and `huggingface/` mirror it and are bumped in the same commits.
The script asks for the release's highlights (an editor on a template
listing the units, or `--highlights-file`), runs `make check`, writes the
CHANGELOG section (scripts/changelog.py), commits `chore(release): vX.Y.Z`,
creates the annotated tag with the highlights as its message, then bumps the
tree to the next `X.(Y+1).0-dev` in a follow-up commit. It never pushes:
publishing is a separate action.

    make release                          # cut the release
    make release HIGHLIGHTS=notes.md      # without an editor
    make release DRY_RUN=1                # print the section, change nothing

Standard library only.
"""

import argparse
import datetime
import io
import os
import pathlib
import re
import shlex
import subprocess
import sys
import tempfile
from typing import NoReturn

import changelog

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHANGELOG = ROOT / "CHANGELOG.md"
MANIFEST_PATHS = ("build.zig.zon", "inference/build.zig.zon", "huggingface/build.zig.zon")
MANIFESTS = tuple(ROOT / path for path in MANIFEST_PATHS)
VERSION_RE = re.compile(r'\.version = "([^"]+)"')
DEV_RE = re.compile(r"^(?P<x>\d+)\.(?P<y>\d+)\.(?P<z>\d+)-dev$")


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout


def fail(message) -> NoReturn:
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


def ask_highlights(tag, range_spec):
    """The highlights from $VISUAL/$EDITOR on a template; '#' lines are dropped."""
    previous = range_spec.split("..")[0] if ".." in range_spec else None
    grouped = changelog.release_units(range_spec, previous, None)
    breaking, _, _ = changelog.changes(range_spec)
    template = [
        f"# Highlights for {tag}: two to four sentences on what this release changes",
        "# for someone using nuclis, with the numbers that show it. Lines starting",
        "# with '#' are dropped; empty text cancels the release.",
        "#",
        "# Units:",
    ]
    for heading, units in grouped:
        template += [f"#   {heading}"] + [f"#     {unit}  {title}" for unit, title, _, _ in units]
    if breaking:
        template += ["#", "# Breaking:"] + [f"#   {entry}" for entry in breaking]
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"
    with tempfile.NamedTemporaryFile("w+", suffix=".md", prefix="highlights-", delete=False) as handle:
        handle.write("\n".join(template) + "\n\n")
        path = pathlib.Path(handle.name)
    try:
        if subprocess.run([*shlex.split(editor), str(path)]).returncode != 0:
            fail(f"{editor} exited with an error; nothing was written")
        return path.read_text()
    finally:
        path.unlink()


def clean_highlights(text):
    return "\n".join(line for line in text.splitlines() if not line.startswith("#")).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print the section and exit")
    parser.add_argument("--highlights-file", type=pathlib.Path, help="the highlights, instead of an editor")
    args = parser.parse_args()
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(line_buffering=True)

    current = manifest_version()
    if not DEV_RE.match(current):
        fail(f"manifest version {current!r} is not MAJOR.MINOR.PATCH-dev")
    release = current[: -len("-dev")]
    tag = f"v{release}"
    following = next_dev(release)

    if git("status", "--porcelain").strip():
        if not args.dry_run:
            fail("the working tree is dirty; commit or stash first")
        print("release: the working tree is dirty; a real release would refuse", file=sys.stderr)
    if git("tag", "--list", tag).strip():
        fail(f"tag {tag} already exists; tags never move")
    if CHANGELOG.exists() and re.search(rf"^## \[{re.escape(tag)}\]", CHANGELOG.read_text(), re.MULTILINE):
        fail(f"CHANGELOG.md already has a {tag} section")

    print(f"release {release} ({tag}), then begin {following}")
    previous = changelog.previous_tag()
    range_spec = f"{previous}..HEAD" if previous else "HEAD"
    highlights = clean_highlights(args.highlights_file.read_text()) if args.highlights_file else None
    date = datetime.date.today().isoformat()

    if args.dry_run:
        print()
        print(changelog.build_section(tag, date, range_spec, previous, highlights), end="")
        print("\ndry run: nothing checked, written, committed, or tagged")
        return

    if highlights is None:
        highlights = clean_highlights(ask_highlights(tag, range_spec))
    if not highlights:
        fail("no highlights; nothing was written")

    print("checking: make check")
    if subprocess.run(["make", "check"], cwd=ROOT).returncode != 0:
        fail("make check failed; nothing was written")

    set_version(release)
    changelog.write(changelog.build_section(tag, date, range_spec, previous, highlights))

    git("add", *MANIFEST_PATHS, "CHANGELOG.md")
    git("commit", "-m", f"chore(release): {tag}")
    git("tag", "-a", tag, "-m", f"nuclis {tag}\n\n{highlights}")

    set_version(following)
    git("add", *MANIFEST_PATHS)
    git("commit", "-m", f"chore: begin {following}")

    print(f"\ntagged {tag}; the tree is now {following}.")
    print("publish with:")
    print("  git push origin HEAD")
    print(f"  git push origin {tag}")


if __name__ == "__main__":
    main()
