#!/usr/bin/env python3
"""The release, in three steps an agent runs when the user says "release".

    release.py draft                       what merged since the last tag, for the highlights
    release.py open X.Y.Z --tiers TEXT     the version pull request, highlights in its description
    release.py tag X.Y.Z                   the annotated (signed) tag on main, pushed

The highlights are a file, `.zig-cache/highlights-vX.Y.Z.txt`: the agent
writes it from `draft`'s output, the user reads it, `open` puts it in the
pull request, and `tag` makes it the tag's message, which `release.yml`
publishes as the release notes' head (docs/development.md § Versioning).
Every step checks before it changes anything. Standard library; `git` and
`gh` do the work.
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
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
    """The first paragraph of a pull request description's *What and why*,
    or of the description itself when it has no such section."""
    text = re.sub(r"<!--.*?-->", "", body or "", flags=re.S)
    m = re.search(r"^## What and why\s*\n(.*?)(?=^## |\Z)", text, flags=re.S | re.M)
    section = m[1] if m else text
    for para in re.split(r"\n\s*\n", section.strip()):
        if para.strip() and not para.lstrip().startswith("#"):
            return " ".join(line.strip() for line in para.strip().splitlines())
    return ""


def merged_since(prs, commits):
    """The pull requests whose merge commit is in `commits`, oldest first; a
    release's own version pull request is not part of what it releases."""
    inside = [
        p
        for p in prs
        if (p.get("mergeCommit") or {}).get("oid") in commits and not p.get("title", "").startswith("chore(release):")
    ]
    return sorted(inside, key=lambda p: p.get("mergedAt") or "")


def highlights_path(version):
    return ROOT / ".zig-cache" / f"highlights-v{version}.txt"


def pr_body(version, highlights, tiers):
    return f"""## What and why

The v{version} version bump: `make version V={version}` in the three `build.zig.zon`. After this merges, the release is the annotated tag `v{version}` on the merge commit, with the highlights below as its message (`make tag V={version}`).

### Highlights (the tag's message)

```
{highlights.strip()}
```

## Evidence

- The release tiers: {tiers}

## Remaining

After the merge: `make tag V={version}`.

## Checklist

- [x] `make verify-auto` passes from the branch's base, and the tiers it names ran or are exempt (version strings only)
- [x] Documents that describe the change are updated; `make docs-check` passes
- [x] The section of `TODO.md` this work came from is removed or updated
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


def draft(args):
    fetch()
    tag = args.since or last_tag()
    commits = set(run("git", "rev-list", f"{tag}..origin/main").split())
    listing = run(
        "gh",
        "pr",
        "list",
        "--state",
        "merged",
        "--base",
        "main",
        "--limit",
        "300",
        "--json",
        "number,title,body,mergedAt,mergeCommit,url",
    )
    every = json.loads(listing or "[]")
    prs = merged_since(every, commits)
    merges = {(p.get("mergeCommit") or {}).get("oid") for p in every}
    # Commits that reached main without a pull request (before the ruleset).
    direct = [
        line.split(" ", 1)[1]
        for line in run("git", "log", "--format=%H %s", f"{tag}..origin/main").splitlines()
        if line.split(" ", 1)[0] not in merges and " " in line
    ]
    current = parse_version(tag.lstrip("v")) or refuse(f"{tag} is not semver")
    nxt = ".".join(map(str, suggest_version(current, [p["title"] for p in prs] + direct)))
    print(f"# Since {tag}: {len(prs)} merged pull request(s), {len(commits)} commit(s)\n")
    if not commits:
        print("Nothing merged since the last tag: nothing to release.")
        return
    for p in prs:
        print(f"## #{p['number']} {p['title']}\n")
        summary = what_and_why(p.get("body", ""))
        print(f"{summary}\n" if summary else "(no description)\n")
    if direct:
        print(f"## {len(direct)} commit(s) outside pull requests\n")
        print("\n".join(f"- {s}" for s in direct) + "\n")
    print(f"Suggested version: {nxt}")
    print(f"Highlights go to: {highlights_path(nxt).relative_to(ROOT)}")
    print("Highlights: two to four sentences for a user; what changed for them first; measured numbers only.")


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

    def test_merged_since(self):
        prs = [
            {"number": 2, "mergeCommit": {"oid": "b"}, "mergedAt": "2026-10-05T02:00:00Z"},
            {"number": 1, "mergeCommit": {"oid": "a"}, "mergedAt": "2026-10-05T01:00:00Z"},
            {"number": 3, "mergeCommit": {"oid": "z"}, "mergedAt": "2026-09-01T00:00:00Z"},
            {"number": 4, "mergeCommit": None, "mergedAt": None},
            {
                "number": 5,
                "title": "chore(release): v0.7.0",
                "mergeCommit": {"oid": "c"},
                "mergedAt": "2026-10-05T03:00:00Z",
            },
        ]
        self.assertEqual([p["number"] for p in merged_since(prs, {"a", "b", "c"})], [1, 2])

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
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
