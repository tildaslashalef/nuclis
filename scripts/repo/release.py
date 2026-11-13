#!/usr/bin/env python3
"""The release, in three steps an agent runs when the user says "release".

    release.py draft                       what landed since the last tag, for the highlights
    release.py open X.Y.Z --tiers TEXT     the version pull request, highlights in its description
    release.py tag X.Y.Z                   the annotated (signed) tag on main, pushed
    release.py notes vX.Y.Z                the release notes' list of changes (release.yml)

Milestones merge by rebase, one commit per unit, so what a release holds is
the Conventional Commit subjects on main since the last tag, grouped by type.

The highlights are a file, `.zig-cache/highlights-vX.Y.Z.txt`: the agent
writes it from `draft`'s output, the user reads it, `open` puts it in the
pull request, and `tag` makes it the tag's message, which `release.yml`
publishes as the release notes' head (docs/development.md § Versioning).
Every step checks before it changes anything. Standard library; `git` and
`gh` do the work.
"""

import argparse
import pathlib
import re
import subprocess
import sys
import unittest

# The checkout the script sits in; `--root` names another, for a copy run from
# outside it (release.yml runs its own revision's script against the tag's).
ROOT = pathlib.Path(__file__).resolve().parents[2]
MANIFESTS = ("build.zig.zon", "inference/build.zig.zon", "huggingface/build.zig.zon")
SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:-[0-9A-Za-z.-]+)?$")
TITLE = re.compile(r"^(?P<type>[a-z]+)(?:\([^)]*\))?(?P<breaking>!)?: ")

# ---- pure functions (covered by --self-test) --------------------------------


def parse_version(text):
    """`"0.6.0"` -> (0, 6, 0); None when it is not semver."""
    m = SEMVER.match(text)
    return (int(m[1]), int(m[2]), int(m[3])) if m else None


def suggest_version(current, titles):
    """The next version from the merged titles: before 1.0, breaking or `feat`
    bumps the minor, anything else the patch; from 1.0, breaking bumps the major."""
    major, minor, patch = current
    kinds = [TITLE.match(t) for t in titles]
    breaking = any(k and k["breaking"] for k in kinds) or any("BREAKING CHANGE" in t for t in titles)
    feature = any(k and k["type"] == "feat" for k in kinds)
    if major == 0:
        return (0, minor + 1, 0) if breaking or feature else (0, minor, patch + 1)
    if breaking:
        return (major + 1, 0, 0)
    return (major, minor + 1, 0) if feature else (major, minor, patch + 1)


def what_and_why(body):
    """The first paragraph of a commit record's or pull request
    description's *What and why* (a `## ` heading or a bare line), or of the
    text itself when it has no such section."""
    text = re.sub(r"<!--.*?-->", "", body or "", flags=re.S)
    m = re.search(
        r"^(?:## )?What and why\s*\n(.*?)(?=^## |^(?:Evidence|Remaining|Checklist)\s*$|\Z)", text, flags=re.S | re.M
    )
    section = m[1] if m else text
    for para in re.split(r"\n\s*\n", section.strip()):
        if para.strip() and not para.lstrip().startswith("#"):
            return " ".join(line.strip() for line in para.strip().splitlines())
    return ""


GROUPS = (
    ("feat", "Features"),
    ("fix", "Fixes"),
    ("perf", "Performance"),
    ("refactor", "Refactoring"),
    ("docs", "Documentation"),
    ("test", "Tests"),
    ("build", "Build"),
    ("ci", "CI"),
    ("chore", "Chores"),
)


def grouped(subjects):
    """Commit subjects as Markdown sections by Conventional Commit type, in
    `GROUPS` order (anything else last); breaking ones marked; the version
    bump left out."""
    sections = {}
    for subject in subjects:
        if subject.startswith("chore(release):"):
            continue
        m = TITLE.match(subject)
        kind = m["type"] if m and m["type"] in dict(GROUPS) else None
        mark = " **(breaking)**" if m and m["breaking"] else ""
        sections.setdefault(kind, []).append(f"- {subject}{mark}")
    order = [k for k, _ in GROUPS] + [None]
    names = dict(GROUPS) | {None: "Other"}
    return "\n\n".join(f"### {names[k]}\n\n" + "\n".join(sections[k]) for k in order if k in sections)


def highlights_path(version):
    return ROOT / ".zig-cache" / f"highlights-v{version}.txt"


def pr_body(version, highlights, tiers):
    return f"""## What and why

The v{version} version bump: `make version V={version}` in the three `build.zig.zon` and `site/index.html`. After this merges, the release is the annotated tag `v{version}` on the merge commit, with the highlights below as its message (`make tag V={version}`).

### Highlights (the tag's message)

```
{highlights.strip()}
```

## Evidence

- The release tiers: {tiers}

## Remaining

After the merge: `make tag V={version}`.

## Checklist

- [ ] `make verify-auto` passes from the branch's base, and the tiers it names ran or are exempt (version strings only)
- [ ] Documents that describe the change are updated; `make docs-check` passes
"""


# ---- the repository ------------------------------------------------------------


def run(*cmd, check=True):
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if check and out.returncode != 0:
        raise SystemExit(f"release: {' '.join(cmd)} failed: {out.stderr.strip()}")
    return out.stdout.strip()


def refuse(message):
    raise SystemExit(f"release: refusing: {message}")


def manifest_version():
    text = (ROOT / "build.zig.zon").read_text()
    m = re.search(r'^\s*\.version = "([^"]*)",', text, re.M)
    return m[1] if m else refuse("build.zig.zon has no .version line")


def last_tag():
    return run("git", "describe", "--tags", "--abbrev=0", "--match", "v*", "origin/main")


def fetch():
    run("git", "fetch", "--quiet", "--tags", "origin", "main")


def commits_between(start, end):
    """(subject, body) of each commit after `start` up to `end`, oldest first."""
    return parse_log(run("git", "log", "--reverse", "--format=%s%x1f%b%x1e", f"{start}..{end}"))


def parse_log(log):
    """(subject, body) pairs from `%s%x1f%b%x1e` output. str.strip() counts the
    separators as whitespace, so the last commit's may be gone when its body is empty."""
    pairs = []
    for entry in log.split("\x1e"):
        subject, _, body = entry.strip("\n").partition("\x1f")
        if subject.strip():
            pairs.append((subject, body))
    return pairs


def draft(args):
    fetch()
    tag = args.since or last_tag()
    commits = commits_between(tag, "origin/main")
    subjects = [subject for subject, _ in commits]
    current = parse_version(tag.lstrip("v")) or refuse(f"{tag} is not semver")
    nxt = ".".join(map(str, suggest_version(current, subjects)))
    print(f"# Since {tag}: {len(commits)} commit(s)\n")
    if not commits:
        print("Nothing landed since the last tag: nothing to release.")
        return
    for subject, body in commits:
        if subject.startswith("chore(release):"):
            continue
        print(f"## {subject}\n")
        summary = what_and_why(body)
        print(f"{summary}\n" if summary else "(no record)\n")
    print(f"Suggested version: {nxt}")
    print(f"Highlights go to: {highlights_path(nxt).relative_to(ROOT)}")
    print("Highlights: two to four sentences for a user; what changed for them first; measured numbers only.")


def notes(args):
    previous = run("git", "describe", "--tags", "--abbrev=0", "--match", "v*", f"{args.tag}^", check=False)
    start = previous or run("git", "rev-list", "--max-parents=0", args.tag).splitlines()[0]
    print(f"## Changes since {previous or 'the first commit'}\n")
    print(grouped([subject for subject, _ in commits_between(start, args.tag)]))


def open_pr(args):
    version = args.version
    wanted = parse_version(version)
    if wanted is None:
        refuse(f"{version} is not X.Y.Z")
    fetch()
    if run("git", "status", "--porcelain"):
        refuse("the working tree has changes")
    if run("git", "rev-parse", "HEAD") != run("git", "rev-parse", "origin/main"):
        refuse("HEAD is not origin/main (git switch main && git pull)")
    current = parse_version(manifest_version())
    if current is None or wanted is None or wanted <= current:
        refuse(f"{version} is not above the manifest's {manifest_version()}")
    if run("git", "tag", "-l", f"v{version}"):
        refuse(f"v{version} exists")
    path = highlights_path(version)
    if not path.is_file() or not path.read_text().strip():
        refuse(f"no highlights in {path.relative_to(ROOT)}")
    branch = f"release-v{version}"
    run("git", "switch", "-c", branch)
    run("make", "version", f"V={version}")
    run("git", "commit", "-am", f"chore(release): v{version}")
    run("git", "push", "-u", "origin", branch)
    url = run(
        "gh",
        "pr",
        "create",
        "--base",
        "main",
        "--head",
        branch,
        "--title",
        f"chore(release): v{version}",
        "--body",
        pr_body(version, path.read_text(), args.tiers),
    )
    print(url)


def tag(args):
    version = args.version
    fetch()
    if run("git", "branch", "--show-current") != "main":
        refuse("not on main")
    if run("git", "rev-parse", "HEAD") != run("git", "rev-parse", "origin/main"):
        refuse("main is not origin/main (git pull)")
    if manifest_version() != version:
        refuse(f"the manifest says {manifest_version()}, not {version} (is the version pull request merged?)")
    path = highlights_path(version)
    if not path.is_file() or not path.read_text().strip():
        refuse(f"no highlights in {path.relative_to(ROOT)}")
    if run("git", "tag", "-l", f"v{version}") or run("git", "ls-remote", "--tags", "origin", f"refs/tags/v{version}"):
        refuse(f"v{version} exists")
    run("git", "tag", "-a", f"v{version}", "-F", str(path))
    run("git", "push", "origin", f"v{version}")
    print(
        f"v{version} pushed; release.yml publishes it: https://github.com/tildaslashalef/nuclis/actions/workflows/release.yml"
    )


# ---- self-test ---------------------------------------------------------------


class SelfTest(unittest.TestCase):
    def test_parse_log_keeps_a_last_commit_without_body(self):
        log = "feat: a\x1fwhy\nmore\x1e\nchore(release): v1.0.0\x1f\x1e".strip()
        self.assertEqual(parse_log(log), [("feat: a", "why\nmore"), ("chore(release): v1.0.0", "")])

    def test_parse_version(self):
        self.assertEqual(parse_version("0.6.0"), (0, 6, 0))
        self.assertEqual(parse_version("1.2.3-rc1"), (1, 2, 3))
        self.assertIsNone(parse_version("v0.6.0"))

    def test_suggest_version(self):
        self.assertEqual(suggest_version((0, 6, 0), ["fix(metal): a race", "docs: a page"]), (0, 6, 1))
        self.assertEqual(suggest_version((0, 6, 0), ["feat(agent): a tool"]), (0, 7, 0))
        self.assertEqual(suggest_version((0, 6, 0), ["refactor!: the API"]), (0, 7, 0))
        self.assertEqual(suggest_version((1, 2, 3), ["feat: x", "fix!: y"]), (2, 0, 0))
        self.assertEqual(suggest_version((1, 2, 3), ["feat: x"]), (1, 3, 0))

    def test_what_and_why(self):
        body = "<!-- the template's note -->\n## What and why\n\nThe first\nparagraph.\n\nThe second.\n\n## Evidence\n\n- x\n"
        self.assertEqual(what_and_why(body), "The first paragraph.")
        self.assertEqual(what_and_why("Plain description.\n\nMore."), "Plain description.")
        self.assertEqual(what_and_why(""), "")

    def test_what_and_why_reads_a_commit_record(self):
        body = "What and why\n\n- A fix\n  in two lines.\n- Another.\n\nEvidence\n\n- tests\n"
        self.assertEqual(what_and_why(body), "- A fix in two lines. - Another.")

    def test_grouped(self):
        text = grouped(["fix(cli): a", "feat!: b", "chore(release): v0.7.0", "wip c", "feat(x): d"])
        self.assertEqual(
            text,
            "### Features\n\n- feat!: b **(breaking)**\n- feat(x): d\n\n### Fixes\n\n- fix(cli): a\n\n### Other\n\n- wip c",
        )

    def test_pr_body_carries_the_highlights(self):
        body = pr_body("0.7.0", "Faster.\n", "skipped: docs only")
        self.assertIn("```\nFaster.\n```", body)
        self.assertIn("The release tiers: skipped: docs only", body)
        self.assertIn("make tag V=0.7.0", body)


def main():
    if "--self-test" in sys.argv:
        sys.argv = sys.argv[:1]
        unittest.main(module=__name__, verbosity=1)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=pathlib.Path, help="the repository to act on (default: the one this script is in)")
    sub = ap.add_subparsers(dest="command", required=True)
    d = sub.add_parser("draft", help="what merged since the last tag, for the highlights")
    d.add_argument("--since", help="the tag to start from (default: the last one)")
    d.set_defaults(fn=draft)
    o = sub.add_parser("open", help="the version pull request")
    o.add_argument("version")
    o.add_argument("--tiers", required=True, help="what the release tiers did, for the record: ran, or skipped and why")
    o.set_defaults(fn=open_pr)
    t = sub.add_parser("tag", help="the annotated tag on main, pushed")
    t.add_argument("version")
    t.set_defaults(fn=tag)
    n = sub.add_parser("notes", help="the release notes' list of changes, from the previous tag to TAG")
    n.add_argument("tag")
    n.set_defaults(fn=notes)
    args = ap.parse_args()
    if args.root:
        global ROOT
        ROOT = args.root.resolve()
    args.fn(args)


if __name__ == "__main__":
    main()
