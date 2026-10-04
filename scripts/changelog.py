#!/usr/bin/env python3
"""Write the CHANGELOG section for a release, led by the work it closed.

A section is: the hand-written highlights, one line per unit the release's
commits name (`AREA-NN` in the subject; the title is the unit's heading in
docs/engineering-log.md, linked at the tag), breaking changes, feat/fix/perf
commits outside any unit, every commit folded into a `<details>` block, and
a compare link. Commits are the non-merge ones since the previous tag. If
CHANGELOG.md does not exist it is created; otherwise the section is inserted
newest first. `scripts/release.py` calls it; by hand:

    make changelog ARGS="v0.6.0 --dry-run"
    make changelog ARGS="v0.5.0 --range v0.4.0..v0.5.0 --dry-run"

Standard library only.
"""

import argparse
import datetime
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHANGELOG = ROOT / "CHANGELOG.md"
LOG = ROOT / "docs/engineering-log.md"
TODO = ROOT / "TODO.md"
HEADER_RE = re.compile(r"^(?P<type>[a-z]+)(?:\((?P<scope>[^)]+)\))?(?P<breaking>!)?: (?P<subject>.+)$")
BREAKING_FOOTER_RE = re.compile(r"^BREAKING CHANGE: (.+)$", re.MULTILINE)
UNIT_RE = re.compile(r"\b([A-Z]{4}-\d{2})\b")
HEADING_RE = re.compile(r"^## ([A-Z]{4}-\d{2}) — (.+)$", re.MULTILINE)
DATE_SUFFIX_RE = re.compile(r" \(\d{4}-\d{2}-\d{2}[^)]*\)$")
OUTSIDE_TYPES = ("feat", "fix", "perf")
PLACEHOLDER = "_Highlights are written when the release is cut._"


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout


def previous_tag(ref="HEAD"):
    out = subprocess.run(
        ["git", "-C", str(ROOT), "describe", "--tags", "--abbrev=0", "--match", "v*", ref],
        capture_output=True,
        text=True,
    )
    return out.stdout.strip() or None


def repo_url():
    """https://github.com/<owner>/<repo> from origin, ssh or https."""
    url = git("remote", "get-url", "origin").strip()
    url = re.sub(r"^git@github\.com:", "https://github.com/", url)
    return re.sub(r"\.git$", "", url)


def commits(range_spec):
    """(hash, subject, body), oldest first."""
    out = git("log", "--reverse", "--no-merges", "--format=%x1e%H%x1f%s%x1f%b", range_spec)
    for record in out.split("\x1e"):
        parts = record.split("\x1f")
        if len(parts) >= 3:
            yield parts[0].strip(), parts[1].strip(), parts[2]


def slug(heading):
    """GitHub's anchor for a Markdown heading."""
    text = re.sub(r"[^\w\- ]", "", heading.lower())
    return text.replace(" ", "-")


def unit_titles():
    """{id: (title, anchor or None)}: closed units from the log, open ones from TODO.md."""
    titles = {}
    if TODO.exists():
        for unit, rest in HEADING_RE.findall(TODO.read_text()):
            titles[unit] = (DATE_SUFFIX_RE.sub("", re.sub(r" \((open|\d+ sessions?[^)]*)\)$", "", rest)), None)
    for unit, rest in HEADING_RE.findall(LOG.read_text()):
        titles[unit] = (DATE_SUFFIX_RE.sub("", rest), slug(f"{unit} — {rest}"))
    return titles


def gather(range_spec):
    """Commits sorted into units (first-seen order), breaking, outside units, and all."""
    units, breaking, outside, every = {}, [], [], []
    for sha, subject, body in commits(range_spec):
        every.append((sha, subject))
        match = HEADER_RE.match(subject)
        for unit in dict.fromkeys(UNIT_RE.findall(subject)):
            units.setdefault(unit, []).append(sha)
        if not match:
            continue
        scope, text = match.group("scope"), match.group("subject")
        text = UNIT_RE.sub("", text).replace("()", "").rstrip(" ,;")
        entry = f"**{scope}:** {text}" if scope else text
        if match.group("breaking") or BREAKING_FOOTER_RE.search(body):
            breaking.append(entry)
        if match.group("type") in OUTSIDE_TYPES and not UNIT_RE.search(subject):
            outside.append(entry)
    return units, breaking, outside, every


def build_section(tag, date, range_spec, previous, highlights=None):
    units, breaking, outside, every = gather(range_spec)
    titles, url = unit_titles(), repo_url()
    lines = [f"## [{tag}] - {date}", "", (highlights or PLACEHOLDER).strip(), ""]
    if units:
        lines += ["### Units", ""]
        for unit, shas in units.items():
            title, anchor = titles.get(unit, ("(no log entry)", None))
            name = f"[{unit}]({url}/blob/{tag}/docs/engineering-log.md#{anchor})" if anchor else unit
            count = f" ({len(shas)} commits)" if len(shas) > 1 else ""
            lines.append(f"- **{name}** {title}{count}")
        lines.append("")
    if breaking:
        lines += ["### Breaking Changes", ""] + [f"- {e}" for e in breaking] + [""]
    if outside:
        lines += ["### Outside units", ""] + [f"- {e}" for e in outside] + [""]
    lines += [f"<details><summary>All {len(every)} commits</summary>", ""]
    lines += [f"- [`{sha[:7]}`]({url}/commit/{sha}) {subject}" for sha, subject in reversed(every)]
    lines += ["", "</details>", ""]
    if previous:
        lines += [f"**Full diff:** [{previous}...{tag}]({url}/compare/{previous}...{tag})", ""]
    return "\n".join(lines).rstrip() + "\n"


def write(section):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("version", help="the release tag, e.g. v0.6.0")
    parser.add_argument("--date", default=datetime.date.today().isoformat())
    parser.add_argument("--range", help="commits to describe (default: previous tag..HEAD)")
    parser.add_argument("--highlights-file", type=pathlib.Path, help="the highlights paragraph")
    parser.add_argument("--dry-run", action="store_true", help="print the section, write nothing")
    args = parser.parse_args()

    if args.range:
        range_spec, previous = args.range, args.range.split("..")[0] or None
    else:
        previous = previous_tag()
        range_spec = f"{previous}..HEAD" if previous else "HEAD"
    highlights = args.highlights_file.read_text() if args.highlights_file else None
    section = build_section(args.version, args.date, range_spec, previous, highlights)

    if args.dry_run:
        print(section, end="")
        return
    write(section)
    print(f"wrote {CHANGELOG.name} for {args.version} since {previous or 'the beginning'}")


if __name__ == "__main__":
    main()
