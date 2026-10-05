#!/usr/bin/env python3
"""Write the CHANGELOG section for a release, led by the work it closed.

A section is: the hand-written highlights, one line per unit the release's
commits name (`AREA-NN` in the subject; the title is the unit's heading in
docs/worklog.md, linked at the tag), breaking changes, feat/fix/perf
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
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHANGELOG = ROOT / "CHANGELOG.md"
LOG = "docs/worklog.md"
# The log's path before the documents were restructured; tags up to v0.5.0 hold it.
OLD_LOG = "docs/engineering-log.md"
TODO = "TODO.md"
HEADER_RE = re.compile(r"^(?P<type>[a-z]+)(?:\((?P<scope>[^)]+)\))?(?P<breaking>!)?: (?P<subject>.+)$")
BREAKING_FOOTER_RE = re.compile(r"^BREAKING CHANGE: (.+)$", re.MULTILINE)
UNIT_RE = re.compile(r"\b([A-Z]{4}-\d{2})\b")
# A table row; its identifier is bare in older logs and links its entry since the restructure.
ROW_RE = re.compile(r"^\| (?:\[)?([A-Z]{4}-\d{2})(?:\]\(#[\w-]+\))? \| (.+?) \| \d{4}-\d{2}-\d{2} \|$", re.MULTILINE)
HEADING_RE = re.compile(r"^#{2,4} ([A-Z]{4}-\d{2}) — (.+)$", re.MULTILINE)
SUFFIX_RE = re.compile(r" \([^()]*\)$")
OUTSIDE_TYPES = ("feat", "fix", "perf")
# Inference first: the order a reader of an inference engine cares about.
AREAS = (
    ("MODL", "Models"),
    ("ENGN", "Engine"),
    ("KERN", "Kernels"),
    ("AGNT", "Agent"),
    ("APPS", "Application"),
    ("TERM", "Terminal"),
    ("REPO", "Repository"),
)
PLACEHOLDER = "_Highlights are written when the release is cut._"


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout


def read_at(ref, path):
    """A file at a revision, or the working tree's when ref is None; empty if absent."""
    if ref is None:
        file = ROOT / path
        return file.read_text() if file.exists() else ""
    out = subprocess.run(["git", "-C", str(ROOT), "show", f"{ref}:{path}"], capture_output=True, text=True)
    return out.stdout if out.returncode == 0 else ""


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


def closed_units(ref):
    """{id: (title, anchor or None)} for every unit the log lists at ref.

    The table is the complete registry; entries written with a heading give
    the shorter title and an anchor.
    """
    return parse_units(read_at(ref, LOG) or read_at(ref, OLD_LOG))


def parse_units(text):
    """{id: (title, anchor or None)} from a log's table rows and entry headings."""
    units: dict[str, tuple[str, str | None]] = {unit: (title, None) for unit, title in ROW_RE.findall(text)}
    for unit, rest in HEADING_RE.findall(text):
        units[unit] = (SUFFIX_RE.sub("", rest), slug(f"{unit} — {rest}"))
    return units


def open_units(ref):
    """{id: title} for the units TODO.md plans at ref."""
    return {unit: SUFFIX_RE.sub("", rest) for unit, rest in HEADING_RE.findall(read_at(ref, TODO))}


def release_units(range_spec, previous, ref):
    """[(area heading, [(id, title, anchor, note)])]: units closed since previous, by area.

    A unit is closed in this release when the log lists it at ref but not at
    previous; one the commits name that closed earlier is a follow-up, one
    not yet logged is in progress.
    """
    now = closed_units(ref)
    before = closed_units(previous) if previous else {}
    # An in-progress unit is titled by the plan at ref, else by today's log.
    planned = {unit: title for unit, (title, _) in closed_units(None).items()} | open_units(ref)
    named = {unit for _, subject, _ in commits(range_spec) for unit in UNIT_RE.findall(subject)}
    rows = {}
    for unit in set(now) - set(before):
        rows[unit] = (*now[unit], "")
    for unit in named - set(rows):
        if unit in before:
            rows[unit] = (*before[unit], "follow-up")
        elif unit not in now and unit in planned:
            # A unit named nowhere in the plan or the log was dropped, not shipped.
            rows[unit] = (planned[unit], None, "in progress")
    grouped = []
    for area, heading in AREAS:
        units = sorted(unit for unit in rows if unit.startswith(area + "-"))
        if units:
            grouped.append((heading, [(unit, *rows[unit]) for unit in units]))
    return grouped


def changes(range_spec):
    """Breaking entries, feat/fix/perf outside any unit, and every (hash, subject)."""
    breaking, outside, every = [], [], []
    for sha, subject, body in commits(range_spec):
        every.append((sha, subject))
        match = HEADER_RE.match(subject)
        if not match:
            continue
        scope, text = match.group("scope"), match.group("subject")
        text = UNIT_RE.sub("", text).replace("()", "").rstrip(" ,;")
        entry = f"**{scope}:** {text}" if scope else text
        if match.group("breaking") or BREAKING_FOOTER_RE.search(body):
            breaking.append(entry)
        if match.group("type") in OUTSIDE_TYPES and not UNIT_RE.search(subject):
            outside.append(entry)
    return breaking, outside, every


def build_section(tag, date, range_spec, previous, highlights=None, ref=None):
    """The section for tag; ref is the revision whose log and plan describe it (None: the tree)."""
    url = repo_url()
    log_url = f"{url}/blob/{tag}/{LOG}"
    breaking, outside, every = changes(range_spec)
    lines = [f"## [{tag}] - {date}", "", (highlights or PLACEHOLDER).strip(), ""]
    for heading, units in release_units(range_spec, previous, ref):
        lines += [f"### {heading}", ""]
        for unit, title, anchor, note in units:
            target = f"{log_url}#{anchor}" if anchor else log_url
            name = unit if note == "in progress" else f"[{unit}]({target})"
            suffix = f" _({note})_" if note else ""
            lines.append(f"- **{name}** {title}{suffix}".rstrip())
        lines.append("")
    if breaking:
        lines += ["### Breaking changes", ""] + [f"- {e}" for e in breaking] + [""]
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


class SelfTest(unittest.TestCase):
    def test_rows_bare_and_linked(self):
        text = (
            "| REPO-07 | The roadmap retired | 2026-09-10 |\n"
            "| [ENGN-21](#engn-21--the-benchmarks) | The benchmarks, measured again | 2026-10-05 |\n"
        )
        units = parse_units(text)
        self.assertEqual(units["REPO-07"], ("The roadmap retired", None))
        self.assertEqual(units["ENGN-21"], ("The benchmarks, measured again", None))

    def test_heading_gives_title_and_anchor(self):
        units = parse_units("| ENGN-21 | long | 2026-10-05 |\n\n## ENGN-21 — The benchmarks (2026-10-05)\n")
        self.assertEqual(units["ENGN-21"], ("The benchmarks", "engn-21--the-benchmarks-2026-10-05"))

    def test_a_tag_before_the_rename_reads_the_old_path(self):
        # v0.5.0 predates docs/worklog.md; its log is found at the old path.
        units = closed_units("v0.5.0")
        self.assertIn("REPO-31", units)


def main():
    if "--self-test" in sys.argv:
        sys.argv = sys.argv[:1]
        unittest.main(module=__name__, verbosity=1)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("version", help="the release tag, e.g. v0.6.0")
    parser.add_argument("--date", default=datetime.date.today().isoformat())
    parser.add_argument("--range", help="commits to describe (default: previous tag..HEAD)")
    parser.add_argument("--ref", help="read the log and plan at this revision (default: the tree)")
    parser.add_argument("--highlights-file", type=pathlib.Path, help="the highlights paragraph")
    parser.add_argument("--dry-run", action="store_true", help="print the section, write nothing")
    args = parser.parse_args()

    if args.range:
        range_spec = args.range
        previous = args.range.split("..")[0] or None if ".." in args.range else None
    else:
        previous = previous_tag()
        range_spec = f"{previous}..HEAD" if previous else "HEAD"
    highlights = args.highlights_file.read_text() if args.highlights_file else None
    section = build_section(args.version, args.date, range_spec, previous, highlights, args.ref)

    if args.dry_run:
        print(section, end="")
        return
    write(section)
    print(f"wrote {CHANGELOG.name} for {args.version} since {previous or 'the beginning'}")


if __name__ == "__main__":
    main()
