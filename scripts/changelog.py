#!/usr/bin/env python3
"""Write the CHANGELOG section for a release from Conventional Commits.

Collects non-merge commits since the previous tag, groups feat/fix/perf, and
flags breaking changes (a `!` before the colon or a `BREAKING CHANGE:` footer).
If CHANGELOG.md does not exist it is created; otherwise the new section is
inserted at the top of the history, newest first. Run locally before tagging,
not in CI:

    make changelog ARGS="v0.2.0"

Standard library only.
"""
import argparse
import datetime
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHANGELOG = ROOT / "CHANGELOG.md"
TYPE_HEADINGS = [("feat", "Features"), ("fix", "Bug Fixes"), ("perf", "Performance")]
HEADER_RE = re.compile(r"^(?P<type>[a-z]+)(?:\((?P<scope>[^)]+)\))?(?P<breaking>!)?: (?P<subject>.+)$")
BREAKING_FOOTER_RE = re.compile(r"^BREAKING CHANGE: (.+)$", re.MULTILINE)


def git(*args):
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True
    ).stdout


def previous_tag():
    tags = git("tag", "--sort=-creatordate").split()
    return tags[0] if tags else None


def commits(range_spec):
    out = git("log", "--no-merges", "--format=%x1e%H%x1f%s%x1f%b", range_spec)
    for record in out.split("\x1e"):
        if not record.strip():
            continue
        parts = record.split("\x1f")
        if len(parts) >= 3:
            yield parts[1].strip(), parts[2]


def build_section(version, date, range_spec):
    sections = {heading: [] for _, heading in TYPE_HEADINGS}
    breaking, other = [], []
    for subject, body in commits(range_spec):
        match = HEADER_RE.match(subject)
        if not match:
            other.append(subject)
            continue
        scope, text = match.group("scope"), match.group("subject")
        entry = f"**{scope}:** {text}" if scope else text
        if match.group("breaking") or BREAKING_FOOTER_RE.search(body):
            breaking.append(entry)
        for kind, heading in TYPE_HEADINGS:
            if match.group("type") == kind:
                sections[heading].append(entry)
                break
        else:
            other.append(subject)

    lines = [f"## [{version}] - {date}", ""]
    if breaking:
        lines += ["### Breaking Changes", ""] + [f"- {e}" for e in breaking] + [""]
    for _, heading in TYPE_HEADINGS:
        if sections[heading]:
            lines += [f"### {heading}", ""] + [f"- {e}" for e in sections[heading]] + [""]
    if other:
        lines += ["### Other", ""] + [f"- {e}" for e in other] + [""]
    return "\n".join(lines).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="the release version, e.g. v0.2.0")
    parser.add_argument("--date", default=datetime.date.today().isoformat())
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tag = previous_tag()
    range_spec = f"{tag}..HEAD" if tag else "HEAD"
    section = build_section(args.version, args.date, range_spec)

    if args.dry_run:
        print(section, end="")
        return

    if CHANGELOG.exists():
        text = CHANGELOG.read_text()
    else:
        text = "# Changelog\n\nNotable changes, newest first.\n"
    intro, sep, rest = text.partition("\n## [")
    if sep:
        new = f"{intro}\n{section}\n{sep[1:]}{rest}"
    else:
        new = f"{text.rstrip()}\n\n{section}"
    CHANGELOG.write_text(new)
    print(f"wrote {CHANGELOG.name} for {args.version} since {tag or 'the beginning'}")


if __name__ == "__main__":
    main()
