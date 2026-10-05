#!/usr/bin/env python3
"""Every documentation link resolves, and documents move without breaking one.

Check (the default): each relative Markdown link and its `#anchor` in the
tracked `*.md` files; each `docs/….md[#anchor]` path named in `*.json`,
`*.py`, `*.zig`, `Makefile`, and `.github/`; each `blob/main/…` link in
`site/`. Anchors follow GitHub's rule (lowercase, punctuation but `-` and
`_` dropped, spaces to hyphens, `-1`, `-2` for repeats) plus explicit
`<a id>`/`<a name>`. `CHANGELOG.md`'s links pin release tags and are skipped.

Move (`--move old=new[,old=new…]`): `git mv` the files, then rewrite every
reference above, re-relativizing Markdown links from each file's own
directory (a moved file's outgoing links too). `old.md#a=new.md#b` moves a
section's anchor. Standard library only; reads tracked files, no network.
"""

import argparse
import os
import posixpath
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "tildaslashalef/nuclis"
SKIP = {"CHANGELOG.md"}
# Files whose `docs/…` strings are test data or a deliberate old path (the
# changelog reads the log at tags from before its rename).
NO_REWRITE = {"scripts/docs-check.py", "scripts/changelog.py"}
# The plan names old paths on purpose; only its Markdown links follow a move.
LINKS_ONLY = {"TODO.md"}
LINK = re.compile(r"(!?\[(?:[^\]\[]|\[[^\]]*\])*\]\()([^)\s]+)((?:\s+\"[^\"]*\")?\))")
REF_DEF = re.compile(r"^( {0,3}\[[^\]]+\]:\s+)(\S+)", re.MULTILINE)
CODE_SPAN = re.compile(r"`+[^`\n]*`+")
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
HTML_ID = re.compile(r"<a\s+(?:id|name)=\"([^\"]+)\"")
ROOT_PATH = re.compile(r"(?<![\w./-])(docs/[\w./-]+\.md)(#[\w-]+)?")
SITE_LINK = re.compile(rf"https://github\.com/{REPO}/(?:blob|tree)/main/([\w./-]+?)(#[\w-]+)?(?=[\"'\s)<]|$)")
TEXT_SUFFIXES = (".json", ".py", ".zig")
# Paths that tests and examples in code name as data, not as documents.
CODE_DATA_PATHS = {"docs/x.md", "docs/a.md", "docs/faq.md", "docs/design.md", "docs/history.md", "docs/fix_a_queue.md"}


# ---- pure functions (covered by --self-test) --------------------------------


def slug(heading):
    """GitHub's anchor for a heading's text (inline Markdown reduced to its text)."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading)
    text = text.replace("`", "").replace("*", "")
    text = re.sub(r"[^\w\- ]", "", text.lower())
    return text.replace(" ", "-")


def anchors(markdown):
    """Every anchor a Markdown document defines."""
    seen, out = {}, set()
    fence = None
    for line in markdown.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            marker = stripped[:3]
            fence = None if fence == marker else (fence or marker)
            continue
        if fence:
            continue
        out.update(HTML_ID.findall(line))
        m = HEADING.match(line)
        if m:
            base = slug(m[2])
            n = seen.get(base, 0)
            seen[base] = n + 1
            out.add(base if n == 0 else f"{base}-{n}")
    return out


def links(markdown):
    """(start, end, target) of each link target outside code, in document order."""
    out = []
    fence = None
    offset = 0
    for line in markdown.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            marker = stripped[:3]
            fence = None if fence == marker else (fence or marker)
        elif not fence:
            spans = [m.span() for m in CODE_SPAN.finditer(line)]
            for pattern in (LINK, REF_DEF):
                for m in pattern.finditer(line):
                    if any(a <= m.start() < b for a, b in spans):
                        continue
                    out.append((offset + m.start(2), offset + m.end(2), m[2]))
        offset += len(line)
    return sorted(out)


def is_external(target):
    return re.match(r"^[a-z][a-z0-9+.-]*:", target) is not None


def split_target(target):
    path, _, anchor = target.partition("#")
    return path, anchor


def resolve(source, target):
    """The repository path a relative link in `source` points at ('' for the file itself)."""
    path, _ = split_target(target)
    if not path:
        return source
    if path.startswith("/"):
        return posixpath.normpath(path.lstrip("/"))
    return posixpath.normpath(posixpath.join(posixpath.dirname(source), path))


def relative(source, target_path):
    """The link from `source` to `target_path`, both repository paths."""
    rel = posixpath.relpath(target_path, posixpath.dirname(source) or ".")
    return rel


def rewrite_markdown(text, old_source, new_source, moves, anchor_moves):
    """`text` with each relative link re-pointed after `moves` ({old: new} paths).

    Links are read from `old_source`'s directory and written from `new_source`'s.
    """
    parts, last = [], 0
    for start, end, target in links(text):
        if is_external(target):
            continue
        path, anchor = split_target(target)
        old_target = resolve(old_source, target)
        new_target = moves.get(old_target, old_target)
        key = (old_target, anchor)
        if anchor and key in anchor_moves:
            new_target, anchor = anchor_moves[key]
        if not path:
            same = new_target == new_source or new_target == old_source
            new = f"#{anchor}" if same else relative(new_source, new_target) + (f"#{anchor}" if anchor else "")
        else:
            trailing = "/" if path.endswith("/") and not new_target.endswith("/") else ""
            new = relative(new_source, new_target) + trailing + (f"#{anchor}" if anchor else "")
        if new != target and (new_target != old_target or new_source != old_source or key in anchor_moves):
            parts.append(text[last:start])
            parts.append(new)
            last = end
    parts.append(text[last:])
    return "".join(parts)


def rewrite_paths(text, moves, anchor_moves):
    """Root-relative `docs/….md[#a]` paths and site blob links re-pointed."""

    def swap(path, anchor):
        anchor = (anchor or "").lstrip("#")
        if anchor and (path, anchor) in anchor_moves:
            path, anchor = anchor_moves[(path, anchor)]
        else:
            path = moves.get(path, path)
        return path + (f"#{anchor}" if anchor else "")

    text = ROOT_PATH.sub(lambda m: swap(m[1], m[2]), text)
    return SITE_LINK.sub(lambda m: m[0].replace(m[1] + (m[2] or ""), swap(m[1], m[2]), 1), text)


def parse_moves(spec):
    """`old=new,…` -> ({old: new} file moves, {(old, anchor): (new, anchor)})."""
    moves, anchor_moves = {}, {}
    for item in filter(None, (s.strip() for s in spec.split(","))):
        old, _, new = item.partition("=")
        if not new:
            raise ValueError(f"--move {item!r}: expected old=new")
        if "#" in old:
            op, oa = split_target(old)
            np, na = split_target(new)
            anchor_moves[(op, oa)] = (np, na or oa)
        else:
            moves[old] = new
    return moves, anchor_moves


# ---- the repository ------------------------------------------------------------


def tracked():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [p for p in out.splitlines() if (ROOT / p).is_file()]


def is_markdown(path):
    return path.endswith(".md") and path not in SKIP and "/fixtures/" not in path


def is_other_text(path):
    return (
        path.endswith(TEXT_SUFFIXES)
        or path == "Makefile"
        or path.startswith((".github/", "site/"))
        and not path.endswith((".png", ".woff2", ".gif"))
    ) and "/fixtures/" not in path


def check(files):
    problems = []
    cache = {}

    def anchors_of(path):
        if path not in cache:
            cache[path] = anchors((ROOT / path).read_text())
        return cache[path]

    def judge(where, path, anchor):
        target = ROOT / path
        if not target.exists():
            return f"{where}: {path} does not exist"
        if anchor and path.endswith(".md") and target.is_file() and anchor not in anchors_of(path):
            return f"{where}: {path} has no #{anchor}"
        return None

    for path in files:
        if path in NO_REWRITE:
            continue
        if is_markdown(path):
            text = (ROOT / path).read_text()
            for start, _, target in links(text):
                if is_external(target):
                    continue
                line = text.count("\n", 0, start) + 1
                problem = judge(f"{path}:{line}", resolve(path, target), split_target(target)[1])
                if problem:
                    problems.append(problem)
        elif is_other_text(path):
            text = (ROOT / path).read_text(errors="replace")
            for pattern in (ROOT_PATH, SITE_LINK) if path.startswith("site/") else (ROOT_PATH,):
                for m in pattern.finditer(text):
                    if m[1] in CODE_DATA_PATHS:
                        continue
                    line = text.count("\n", 0, m.start()) + 1
                    problem = judge(f"{path}:{line}", m[1].rstrip("/"), (m[2] or "").lstrip("#"))
                    if problem:
                        problems.append(problem)
    return problems


def move(spec):
    moves, anchor_moves = parse_moves(spec)
    files = tracked()
    for old, new in moves.items():
        if old in files:
            os.makedirs(ROOT / posixpath.dirname(new), exist_ok=True)
            subprocess.run(["git", "mv", old, new], cwd=ROOT, check=True)
        elif new not in files:
            raise SystemExit(f"docs-check: {old} is not a tracked file")
    # An interrupted move is resumed: the files list after the moves, by old path.
    files = [next((o for o, n in moves.items() if n == p), p) for p in tracked()]
    changed = 0
    for old_path in files:
        if old_path in NO_REWRITE:
            continue
        new_path = moves.get(old_path, old_path)
        file = ROOT / new_path
        if is_markdown(old_path):
            text = file.read_text()
            out = rewrite_markdown(text, old_path, new_path, moves, anchor_moves)
            if old_path not in LINKS_ONLY:
                out = rewrite_paths(out, moves, anchor_moves)
        elif is_other_text(old_path):
            try:
                text = file.read_text()
            except UnicodeDecodeError:
                continue
            out = rewrite_paths(text, moves, anchor_moves)
        else:
            continue
        if out != text:
            file.write_text(out)
            changed += 1
    print(f"docs-check: moved {len(moves)} file(s), {len(anchor_moves)} anchor(s); rewrote {changed} file(s)")


def run():
    problems = check(tracked())
    for p in problems:
        print(f"docs-check: {p}", file=sys.stderr)
    if problems:
        print(f"docs-check: {len(problems)} problem(s)", file=sys.stderr)
        return 1
    print("docs-check: ok")
    return 0


# ---- self-test ---------------------------------------------------------------


class SelfTest(unittest.TestCase):
    def test_slug(self):
        self.assertEqual(
            slug("The benchmarks on macOS 27 (ENGN-21, 2026-10-04)"), "the-benchmarks-on-macos-27-engn-21-2026-10-04"
        )
        self.assertEqual(slug("ENGN-21 — The benchmarks"), "engn-21--the-benchmarks")
        self.assertEqual(slug("The `--replay` option"), "the---replay-option")
        self.assertEqual(slug("See [bench](bench.md) here"), "see-bench-here")

    def test_anchors_repeat_and_skip_code(self):
        doc = '# A\n## A\n```\n# B\n```\n<a id="x"></a>\n'
        self.assertEqual(anchors(doc), {"a", "a-1", "x"})

    def test_links_skip_code(self):
        doc = "[a](x.md) `[b](y.md)` [`c`](z.md#q)\n```\n[d](w.md)\n```\n[r]: ref.md\n"
        self.assertEqual([t for _, _, t in links(doc)], ["x.md", "z.md#q", "ref.md"])

    def test_resolve_and_relative(self):
        self.assertEqual(resolve("docs/reference/a.md", "../spec.md#x"), "docs/spec.md")
        self.assertEqual(resolve("docs/a.md", "#x"), "docs/a.md")
        self.assertEqual(relative("docs/models/gemma4.md", "docs/engine/gguf.md"), "../engine/gguf.md")

    def test_rewrite_target_moved(self):
        moves = {"docs/reference/bench.md": "docs/benchmarks/README.md"}
        out = rewrite_markdown("see [b](reference/bench.md#acceptance-runs)", "docs/spec.md", "docs/spec.md", moves, {})
        self.assertEqual(out, "see [b](benchmarks/README.md#acceptance-runs)")

    def test_rewrite_source_moved(self):
        moves = {"docs/reference/gemma4.md": "docs/models/gemma4.md"}
        text = "[s](../spec.md) [g](gguf.md#a) [self](#x)"
        out = rewrite_markdown(text, "docs/reference/gemma4.md", "docs/models/gemma4.md", moves, {})
        self.assertEqual(out, "[s](../spec.md) [g](../reference/gguf.md#a) [self](#x)")

    def test_rewrite_anchor_moved(self):
        am = {("docs/development.md", "config"): ("docs/guide/configuration.md", "config")}
        out = rewrite_markdown("[c](development.md#config) [d](development.md#gates)", "docs/a.md", "docs/a.md", {}, am)
        self.assertEqual(out, "[c](guide/configuration.md#config) [d](development.md#gates)")

    def test_rewrite_paths(self):
        moves = {"docs/engineering-log.md": "docs/worklog.md"}
        text = '"evidence": "docs/engineering-log.md#repo-07" and https://github.com/tildaslashalef/nuclis/blob/main/docs/engineering-log.md"'
        out = rewrite_paths(text, moves, {})
        self.assertIn('"docs/worklog.md#repo-07"', out)
        self.assertIn('blob/main/docs/worklog.md"', out)

    def test_parse_moves(self):
        moves, am = parse_moves("a.md=b.md, c.md#x=d.md")
        self.assertEqual(moves, {"a.md": "b.md"})
        self.assertEqual(am, {("c.md", "x"): ("d.md", "x")})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--move", help="old=new[,old=new…] files, or old.md#a=new.md[#b] sections, then rewrite references")
    ap.add_argument("--self-test", action="store_true", help="run the unit tests of the pure functions")
    args = ap.parse_args()
    if args.self_test:
        sys.argv = sys.argv[:1]
        unittest.main(module=__name__, verbosity=1)
    if args.move:
        move(args.move)
        return run()
    return run()


if __name__ == "__main__":
    sys.exit(main())
