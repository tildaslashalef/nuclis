#!/usr/bin/env python3
"""Every documentation link resolves, and documents move without breaking one.

Check (the default): each relative Markdown link and its `#anchor` in the
tracked `*.md` files; each `docs/….md[#anchor]` path named in `*.json`,
`*.py`, `*.zig`, `Makefile`, and `.github/`; each `blob/main/…` link in
`site/`. Anchors follow GitHub's rule (lowercase, punctuation but `-` and
`_` dropped, spaces to hyphens, `-1`, `-2` for repeats) plus explicit
`<a id>`/`<a name>`.

Move (`--move old=new[,old=new…]`): `git mv` the files, then rewrite every
reference above, re-relativizing Markdown links from each file's own
directory (a moved file's outgoing links too). `old.md#a=new.md#b` moves a
section's anchor. Standard library only; reads tracked files, no network.

Also: no work-unit identifier (`AREA-NN`, retired with the worklog at
v0.6.0) in a tracked Markdown file or the gate and workload manifests;
work is named by what it does.

The documents' rules (docs/README.md § Three kinds of document): each file
under `docs/`, and `README.md`, opens with its kind (`<!-- kind: reference
-->`, `card`, `guide`, or `record`); a reference, card, or guide uses no
plan words about the project and names only `make` targets, `scripts/`
paths, `nuclis` commands and flags, gates, and workloads that exist; every
section of a record is dated; each kind keeps to its line budget
(generated blocks not counted); every document is listed in
`docs/README.md`.
"""

import argparse
import os
import posixpath
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = "tildaslashalef/nuclis"
# Files whose `docs/…` strings are this script's own test data.
NO_REWRITE = {"scripts/repo/docs-check.py", "scripts/repo/docs-gen.py"}
LINK = re.compile(r"(!?\[(?:[^\]\[]|\[[^\]]*\])*\]\()([^)\s]+)((?:\s+\"[^\"]*\")?\))")
REF_DEF = re.compile(r"^( {0,3}\[[^\]]+\]:\s+)(\S+)", re.MULTILINE)
# A code span: a run of backticks, then text without that run or a blank line
# (spans may wrap lines), then the same run.
CODE_SPAN = re.compile(r"(`+)((?:(?!\n[ \t]*\n)(?!\1)[\s\S])*?)\1")
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
HTML_ID = re.compile(r"<a\s+(?:id|name)=\"([^\"]+)\"")
ROOT_PATH = re.compile(r"(?<![\w./-])(docs/[\w./-]+\.md)(#[\w-]+)?")
# A link into a tagged revision of this repository, checked against that tag.
TAG_LINK = re.compile(rf"^https://github\.com/{REPO}/blob/(v[\w.-]+)/([\w./-]+?)(?:#([\w-]+))?$")
SITE_LINK = re.compile(rf"https://github\.com/{REPO}/(?:blob|tree)/main/([\w./-]+?)(#[\w-]+)?(?=[\"'\s)<]|$)")
TEXT_SUFFIXES = (".json", ".py", ".zig", ".metal", ".cpp", ".m", ".h", ".sh")
# Paths that tests and examples in code name as data, not as documents.
# A retired work-unit identifier, and the strings of that shape that are not one.
UNIT_ID = re.compile(r"\b[A-Z]{3,5}-[0-9]{2}\b")
NOT_UNIT_IDS = {"UTF-16", "UTF-32"}
UNIT_ID_FILES = {"gates.json", "workloads.json"}
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


def mask_code(markdown):
    """`markdown` with fenced blocks and code spans blanked, offsets kept."""
    out, fence = [], None
    for line in markdown.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            marker = stripped[:3]
            fence = None if fence == marker else (fence or marker)
            out.append(re.sub(r"[^\n]", " ", line))
        elif fence:
            out.append(re.sub(r"[^\n]", " ", line))
        else:
            out.append(line)
    return CODE_SPAN.sub(lambda m: re.sub(r"[^\n]", "x", m[0]), "".join(out))


def links(markdown):
    """(start, end, target, text span or None) of each link outside code, in document order.

    Link text may wrap onto following lines, as Markdown allows.
    """
    masked = mask_code(markdown)
    out = []
    for m in LINK.finditer(masked):
        if "\n\n" in m[1]:
            continue
        lead = 2 if m[1].startswith("!") else 1
        out.append((m.start(2), m.end(2), markdown[m.start(2) : m.end(2)], (m.start(1) + lead, m.end(1) - 2)))
    for m in REF_DEF.finditer(masked):
        out.append((m.start(2), m.end(2), markdown[m.start(2) : m.end(2)], None))
    return sorted(out)


def unit_ids(text):
    """(line, identifier) for each work-unit identifier in a text."""
    return [(text.count("\n", 0, m.start()) + 1, m[0]) for m in UNIT_ID.finditer(text) if m[0] not in NOT_UNIT_IDS]


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


def display(path):
    """How link text names a document: its path under docs/, a folder for its README."""
    rel = path.removeprefix("docs/")
    return posixpath.dirname(rel) if posixpath.basename(rel) == "README.md" else rel


def rename_text(label, old, new, written=None):
    """Link text naming `old` (by root path, docs path, file name, or the link's own
    relative path `written` -> its rewritten form) renamed to `new`."""
    forms = [
        *([written] if written else []),
        (old, "docs/" + display(new)),
        (old.removeprefix("docs/"), display(new)),
        (posixpath.basename(old), posixpath.basename(display(new)) or display(new)),
    ]
    forms = sorted({f for f in forms if f[0] != f[1]}, key=lambda f: -len(f[0]))
    if not forms:
        return label
    pattern = re.compile(r"(?<![\w./-])(" + "|".join(re.escape(o) for o, _ in forms) + r")(?![\w/-])")
    lookup = dict(forms)
    return pattern.sub(lambda m: lookup[m[1]], label)


def rewrite_markdown(text, old_source, new_source, moves, anchor_moves):
    """`text` with each relative link re-pointed after `moves` ({old: new} paths).

    Links are read from `old_source`'s directory and written from `new_source`'s;
    link text that names a moved document is renamed with it.
    """
    parts, last = [], 0
    for start, end, target, span in links(text):
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
            if span and new_target != old_target:
                label = text[span[0] : span[1]]
                parts.append(text[last : span[0]])
                parts.append(rename_text(label, old_target, new_target, (path, split_target(new)[0])))
                last = span[1]
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
    """Tracked files and new ones git does not ignore."""
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [p for p in out.splitlines() if (ROOT / p).is_file()]


# The one document among the test fixtures; the rest are data.
FIXTURE_DOCS = {"tests/fixtures/provenance.md"}


def is_markdown(path):
    return path.endswith(".md") and ("/fixtures/" not in path or path in FIXTURE_DOCS)


def is_other_text(path):
    return (
        path.endswith(TEXT_SUFFIXES)
        or path == "Makefile"
        or path.startswith((".github/", "site/"))
        and not path.endswith((".png", ".woff2", ".gif"))
    ) and "/fixtures/" not in path


def at_tag(tag, path):
    """The text of `path` at `tag`, or None when the tag or the path is absent."""
    out = subprocess.run(["git", "show", f"{tag}:{path}"], cwd=ROOT, capture_output=True, text=True)
    return out.stdout if out.returncode == 0 else None


def check(files):
    problems = []
    cache = {}
    tagged = {}

    def judge_tagged(where, tag, path, anchor):
        if (tag, path) not in tagged:
            text = at_tag(tag, path)
            tagged[(tag, path)] = None if text is None else anchors(text)
        found = tagged[(tag, path)]
        if found is None:
            if (
                at_tag(tag, ".") is None
                and subprocess.run(
                    ["git", "rev-parse", "-q", "--verify", f"refs/tags/{tag}"], cwd=ROOT, capture_output=True
                ).returncode
            ):
                return f"{where}: tag {tag} is not in this clone (git fetch --tags)"
            return f"{where}: {path} does not exist at {tag}"
        if anchor and anchor not in found:
            return f"{where}: {path} at {tag} has no #{anchor}"
        return None

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
        if is_markdown(path) or path in UNIT_ID_FILES:
            text = (ROOT / path).read_text()
            for line, found in unit_ids(text):
                problems.append(f"{path}:{line}: {found} is a unit identifier; name the work by what it does")
        if is_markdown(path):
            text = (ROOT / path).read_text()
            for start, _, target, _ in links(text):
                line = text.count("\n", 0, start) + 1
                tagged_link = TAG_LINK.match(target)
                if tagged_link:
                    problem = judge_tagged(f"{path}:{line}", *tagged_link.groups())
                    if problem:
                        problems.append(problem)
                    continue
                if is_external(target):
                    continue
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
    moved_now = {}
    for old, new in moves.items():
        if old in files:
            os.makedirs(ROOT / posixpath.dirname(new), exist_ok=True)
            subprocess.run(["git", "mv", old, new], cwd=ROOT, check=True)
            moved_now[new] = old
        elif new not in files:
            raise SystemExit(f"docs-check: {old} is not a tracked file")
    # A resumed move: a file moved by an earlier run is already rewritten from
    # its new place, so only this run's moves are read from their old one.
    files = [moved_now.get(p, p) for p in tracked()]
    changed = 0
    for old_path in files:
        if old_path in NO_REWRITE:
            continue
        new_path = moves.get(old_path, old_path) if old_path in moved_now.values() else old_path
        file = ROOT / new_path
        if is_markdown(old_path):
            text = file.read_text()
            out = rewrite_markdown(text, old_path, new_path, moves, anchor_moves)
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


# ---- the documents' rules ------------------------------------------------------

KINDS = {"reference": 400, "card": 200, "guide": 500, "record": None}
KIND_LINE = re.compile(r"^<!-- kind: (\w+) -->\n")
# The teaching companion keeps its own shape (docs/README.md § Three kinds).
EXEMPT = {"docs/llm-guide.md"}
README_BUDGET = 180
PLAN_WORDS = re.compile(r"\b(will|not yet|planned|future|TODO|for now|phase \d+|step \d+)\b", re.IGNORECASE)
OK_MARK = re.compile(r"<!-- ok: [^>]+ -->")
GEN_BLOCK = re.compile(r"(?<!`)<!-- gen:[\w:.-]+ -->.*?<!-- /gen -->", re.DOTALL)
QUOTE = re.compile(r"\"[^\"\n]*\"|“[^”\n]*”")
DATE = re.compile(r"\b20\d\d-\d\d-\d\d\b")
SPAN = re.compile(r"(`+)(.+?)\1", re.DOTALL)
GLOBAL_FLAGS = {"--help", "-h", "--version", "--json"}
# Rules that fail the check; any other is reported as a warning.
FAILING = {"kind", "listed", "plan-word", "record-date", "name", "budget", "owner", "section"}


def kind_of(path, text):
    m = KIND_LINE.match(text)
    return m[1] if m else None


def governed(path):
    return path == "README.md" or (path.startswith("docs/") and path.endswith(".md"))


def strip_generated(text):
    """`text` with every generated block's body removed (the markers stay)."""
    return GEN_BLOCK.sub(lambda m: m[0].split("-->", 1)[0] + "--><!-- /gen -->", text)


def budget_lines(text):
    return strip_generated(text).count("\n")


def plan_words(text):
    """(line, word) for each plan word in prose: code, quotes, generated blocks, and `ok:` lines excluded."""
    masked = mask_code(GEN_BLOCK.sub(lambda m: re.sub(r"[^\n]", " ", m[0]), text))
    found = []
    for n, line in enumerate(masked.splitlines(), 1):
        if OK_MARK.search(line) or line.lstrip().startswith(">"):
            continue
        line = re.sub(r"<!--.*?-->", "", QUOTE.sub("", line))
        found += [(n, m[1]) for m in PLAN_WORDS.finditer(line)]
    return found


def undated_headings(text):
    """(line, heading) for each section heading of a record without a date."""
    out = []
    for n, line in enumerate(mask_code(text).splitlines(), 1):
        m = HEADING.match(line)
        if m and len(m[1]) >= 2 and not DATE.search(m[2]):
            out.append((n, m[2]))
    return out


def make_targets(makefile):
    return {m[1] for m in re.finditer(r"^([\w.-]+):(?!=)", makefile, re.MULTILINE)}


def expand(pattern):
    """A name pattern's alternatives: `{a,b}` braces expanded; `*` kept for fnmatch."""
    m = re.search(r"\{([^{}]*)\}", pattern)
    if not m:
        return [pattern]
    return [x for alt in m[1].split(",") for x in expand(pattern[: m.start()] + alt + pattern[m.end() :])]


def matches_any(pattern, names):
    import fnmatch

    return all(any(fnmatch.fnmatchcase(n, alt) for n in names) for alt in expand(pattern))


class Names:
    """What a document may name: from the Makefile, the tree, the facts, and the manifests."""

    def __init__(self, makefile, files, facts, gates, workloads, catalog):
        self.targets = make_targets(makefile)
        self.files = set(files)
        self.dirs = {posixpath.dirname(f) for f in files} | {
            "/".join(f.split("/")[:i]) for f in files for i in range(1, f.count("/") + 1)
        }
        self.cli = facts.get("cli", {})
        self.gates = [g["name"] for g in gates.get("gates", [])]
        self.families = set(gates.get("families", {})) | {g.get("family") for g in gates.get("gates", [])}
        self.models = set(gates.get("models", {}))
        self.workloads = [f"{r}/{s}" for r, m in workloads.get("models", {}).items() for s in m["shapes"]]
        self.workload_rows = set(workloads.get("models", {}))
        self.artifacts = {a["name"] for a in catalog.get("artifacts", [])}

    def path_exists(self, path):
        import fnmatch

        path = path.rstrip("/").rstrip(".,;:")
        if any(c in path for c in "*?{<"):
            return any(fnmatch.fnmatchcase(f, alt) for f in self.files for alt in expand(path.split("<")[0] + "*"))
        return path in self.files or path in self.dirs

    def problems(self, span):
        """What in one code span names something that does not exist."""
        out = []
        words = span.split()
        if not words:
            return out
        if words[0] == "make" and len(words) > 1 and re.fullmatch(r"[\w.-]+", words[1]):
            target = words[1]
            if target not in self.targets and not target.endswith("-"):
                out.append(f"make target `{target}`")
            for w in words[2:]:
                m = re.fullmatch(r"NAME=['\"]?([^'\"]+)['\"]?", w)
                if m and target == "gate" and not matches_any(m[1], self.gates):
                    out.append(f"gate `{m[1]}`")
                if m and target == "workload" and "<" not in m[1] and not matches_any(m[1], self.workloads):
                    out.append(f"workload `{m[1]}`")
        for w in words:
            if w.startswith("scripts/") and not self.path_exists(w):
                out.append(f"path `{w}`")
        binary = next((i for i, w in enumerate(words) if w in ("nuclis", "./zig-out/bin/nuclis")), None)
        if binary is not None and binary + 1 < len(words):
            cmd = words[binary + 1]
            if re.fullmatch(r"[a-z]+", cmd):
                if cmd not in self.cli:
                    out.append(f"command `nuclis {cmd}`")
                else:
                    known = set(self.cli[cmd]["flags"]) | GLOBAL_FLAGS
                    for w in words[binary + 2 :]:
                        flag = w.split("=")[0]
                        if re.fullmatch(r"--[a-z][\w-]*", flag) and flag not in known:
                            out.append(f"flag `{flag}` of `nuclis {cmd}`")
        if (
            len(words) == 1
            and re.fullmatch(r"[a-z0-9_]+(-[a-z0-9_*{},]+)+", span)
            and span.split("-")[0] in self.families
            and span not in self.artifacts
            and span not in self.models
            and span not in self.workload_rows
            and not matches_any(span, self.gates)
        ):
            out.append(f"gate `{span}`")
        return out


def named_problems(text, names):
    """(line, what) for each code span in prose that names something missing."""
    body = GEN_BLOCK.sub(lambda m: re.sub(r"[^\n]", " ", m[0]), text)
    fenced = mask_code_fences(body)
    out = []
    for m in SPAN.finditer(fenced):
        if OK_MARK.search(fenced[m.end() : fenced.find("\n", m.end())]):
            continue
        line = fenced.count("\n", 0, m.start()) + 1
        out += [(line, what) for what in names.problems(m[2].replace("\n", " "))]
    return out


def mask_code_fences(markdown):
    """`markdown` with fenced blocks blanked (code spans kept), offsets kept."""
    out, fence = [], None
    for line in markdown.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            marker = stripped[:3]
            fence = None if fence == marker else (fence or marker)
            out.append(re.sub(r"[^\n]", " ", line))
        else:
            out.append(re.sub(r"[^\n]", " ", line) if fence else line)
    return "".join(out)


def load_json(path, default):
    try:
        import json

        return json.loads((ROOT / path).read_text())
    except (OSError, ValueError):
        return default


def owner_problems(index, read):
    """Each row of docs/README.md § Who owns which fact names a block its owner must carry."""
    out = []
    section = (
        index.split("## Who owns which fact", 1)[1].split("\n## ", 1)[0] if "## Who owns which fact" in index else ""
    )
    for m in re.finditer(r"^\| ([^|]+) \| \[[^\]]*\]\(([^)#]+)[^)]*\) \(`([\w:-]+)`\) \|$", section, re.MULTILINE):
        fact, target, block = m[1].strip(), resolve("docs/README.md", m[2]), m[3]
        text = read(target)
        if text is not None and f"<!-- gen:{block} -->" not in text:
            out.append(f"docs/README.md: {target} owns {fact.lower()} but carries no `{block}` block")
    return out


SECTION_REF = re.compile(r"([\w./-]+\.md) § ([^)`\"\n,;:]+)")
SECTION_SOURCES = (".zig", ".py", ".metal", ".m", ".h", ".json", ".sh")


def section_title(heading):
    """A heading's text as code cites it: markup and a leading section number dropped."""
    return re.sub(r"^\d+(\.\d+)*\.?\s+", "", re.sub(r"[`*]", "", heading)).lower().strip()


def cites(title, headings):
    """Whether `file.md § title` in code names one of `headings` (either may be cut short)."""
    t = section_title(title)
    if re.fullmatch(r"\d+(\.\d+)*", title):
        return any(re.match(rf"^{re.escape(title)}\b", h) for h in headings)
    return any(h and (t.startswith(section_title(h)) or section_title(h).startswith(t)) for h in headings)


def section_problems(files, read):
    """(where, what) for each `docs/x.md § Title` in code whose document has no such heading."""
    docs = {
        p: [m[2] for m in map(HEADING.match, mask_code_fences(read(p)).splitlines()) if m]
        for p in files
        if is_markdown(p)
    }
    out = []
    for path in files:
        if (
            path in NO_REWRITE
            or "/fixtures/" in path
            or not (path.endswith(SECTION_SOURCES) or path in ("Makefile", "build.zig"))
        ):
            continue
        text = read(path)
        for m in SECTION_REF.finditer(text):
            name, title = m[1], m[2].strip().rstrip(".")
            found = [p for p in docs if p == name or p.endswith("/" + name.lstrip("./"))]
            line = text.count("\n", 0, m.start()) + 1
            if not found:
                out.append(f"{path}:{line}: names {name}, which does not exist")
            elif not any(cites(title, docs[p]) for p in found):
                out.append(f"{path}:{line}: {name} has no section § {title}")
    return out


def rules(files):
    """(rule, problem) for every document rule a file breaks."""
    out = []
    names = Names(
        (ROOT / "Makefile").read_text(),
        files,
        load_json("docs/facts.json", {}),
        load_json("gates.json", {}),
        load_json("workloads.json", {}),
        load_json("src/catalog.json", {}),
    )
    index = (ROOT / "docs/README.md").read_text() if (ROOT / "docs/README.md").exists() else ""
    listed = {resolve("docs/README.md", t) for _, _, t, _ in links(index)}

    def read(path):
        return (ROOT / path).read_text() if (ROOT / path).is_file() else None

    out += [("owner", p) for p in owner_problems(index, read)]

    def text_of(path):
        try:
            return (ROOT / path).read_text()
        except (OSError, UnicodeDecodeError):
            return ""

    out += [("section", p) for p in section_problems(files, text_of)]
    for path in files:
        if not governed(path) or path in EXEMPT:
            continue
        text = (ROOT / path).read_text()
        kind = kind_of(path, text)
        if path.startswith("docs/") and path != "docs/README.md" and path not in listed:
            out.append(("listed", f"{path}: not listed in docs/README.md"))
        if kind not in KINDS:
            out.append(("kind", f"{path}:1: no `<!-- kind: reference|card|guide|record -->` first line"))
            continue
        if kind == "record":
            out += [
                ("record-date", f"{path}:{n}: record heading without a date: {h}") for n, h in undated_headings(text)
            ]
            continue
        out += [
            ("plan-word", f"{path}:{n}: plan word `{w}` (state what is true, or mark `<!-- ok: why -->`)")
            for n, w in plan_words(text)
        ]
        out += [("name", f"{path}:{n}: no such {what}") for n, what in named_problems(text, names)]
        limit = README_BUDGET if path == "README.md" else KINDS[kind]
        used = budget_lines(text)
        if used > limit:
            out.append(("budget", f"{path}: {used} lines, over the {kind} budget of {limit} (two subjects?)"))
    return out


def run():
    files = tracked()
    problems = check(files)
    warnings = []
    for rule, problem in rules(files):
        (problems if rule in FAILING else warnings).append(problem)
    for w in warnings:
        print(f"docs-check: warning: {w}", file=sys.stderr)
    if warnings:
        print(f"docs-check: {len(warnings)} warning(s)", file=sys.stderr)
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

    def test_code_spans_wrap_lines(self):
        doc = "see `git show\nv0.6.0:x` (`| [ID](#anchor) |`) and [real](y.md)\n"
        self.assertEqual([t for _, _, t, _ in links(doc)], ["y.md"])

    def test_links_skip_code(self):
        doc = "[a](x.md) `[b](y.md)` [`c`](z.md#q)\n```\n[d](w.md)\n```\n[r]: ref.md\n"
        self.assertEqual([t for _, _, t, _ in links(doc)], ["x.md", "z.md#q", "ref.md"])

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

    def test_link_text_follows_the_move(self):
        moves = {
            "docs/reference/bench.md": "docs/benchmarks/README.md",
            "docs/reference/clef.md": "docs/models/clef-flash.md",
        }
        text = "[bench.md § Runs](reference/bench.md#runs) [reference/clef.md](reference/clef.md) [the clef](reference/clef.md)"
        out = rewrite_markdown(text, "docs/spec.md", "docs/spec.md", moves, {})
        self.assertEqual(
            out,
            "[benchmarks § Runs](benchmarks/README.md#runs) [models/clef-flash.md](models/clef-flash.md) [the clef](models/clef-flash.md)",
        )

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

    def test_tag_link(self):
        m = TAG_LINK.match("https://github.com/tildaslashalef/nuclis/blob/v0.6.0/docs/worklog.md#kern-21--x")
        assert m is not None
        self.assertEqual(m.groups(), ("v0.6.0", "docs/worklog.md", "kern-21--x"))
        self.assertIsNone(TAG_LINK.match("https://github.com/tildaslashalef/nuclis/blob/main/docs/x.md"))

    def test_unit_ids(self):
        text = "UTF-16 and IQ4_XS and SHA-256\nsee [KERN-03](x.md), ENGN-21\n"
        self.assertEqual(unit_ids(text), [(2, "KERN-03"), (2, "ENGN-21")])

    def test_plan_words(self):
        text = 'It will run.\n`will` in code\n> will quoted\nthe "will" word\nfuture rows <!-- ok: cache rows -->\nstep 3 here\n'
        self.assertEqual(plan_words(text), [(1, "will"), (6, "step 3")])
        self.assertEqual(plan_words("<!-- gen:x -->\nwill\n<!-- /gen -->\n"), [])

    def test_record_headings(self):
        self.assertEqual(undated_headings("# Title\n## A (2026-09-10)\n## B\n"), [(3, "B")])

    def test_budget_skips_generated(self):
        self.assertEqual(budget_lines("a\n<!-- gen:x -->\n1\n2\n3\n<!-- /gen -->\nb\n"), 3)

    def test_names(self):
        names = Names(
            "check:\n\tx\ngate:\n\ty\nworkload:\nOPT := 1\n",
            ["scripts/repo/docs-check.py"],
            {"cli": {"generate": {"usage": [], "flags": ["--prompt"]}}},
            {"gates": [{"name": "qwen38-trace-f32", "family": "qwen38"}], "families": {"qwen38": {}}},
            {"models": {"qwen38": {"model": "qwen38", "shapes": ["code"]}}},
            {"artifacts": [{"name": "qwen38-x"}]},
        )
        self.assertEqual(names.problems("make check"), [])
        self.assertEqual(names.problems("make nope"), ["make target `nope`"])
        self.assertEqual(names.problems("make OPT"), ["make target `OPT`"])
        self.assertEqual(names.problems("make gate NAME='qwen38-*'"), [])
        self.assertEqual(names.problems("make gate NAME=muse-x"), ["gate `muse-x`"])
        self.assertEqual(names.problems("make workload NAME=qwen38/code"), [])
        self.assertEqual(names.problems("scripts/repo/gone.py --x"), ["path `scripts/repo/gone.py`"])
        self.assertEqual(names.problems("scripts/repo/"), [])
        self.assertEqual(names.problems("nuclis generate --prompt x --json"), [])
        self.assertEqual(names.problems("nuclis generate --nope"), ["flag `--nope` of `nuclis generate`"])
        self.assertEqual(names.problems("nuclis chat"), ["command `nuclis chat`"])
        self.assertEqual(names.problems("qwen38-trace-{f32,f16}"), ["gate `qwen38-trace-{f32,f16}`"])
        self.assertEqual(names.problems("qwen38-trace-*"), [])
        self.assertEqual(names.problems("qwen38-x"), [])

    def test_owner(self):
        index = "## Who owns which fact\n\n| Fact | Owner |\n| --- | --- |\n| Encodings | [q](engine/q.md) (`encodings`) |\n"
        texts = {"docs/engine/q.md": "no block"}
        self.assertEqual(len(owner_problems(index, texts.get)), 1)
        texts["docs/engine/q.md"] = "<!-- gen:encodings -->\n<!-- /gen -->"
        self.assertEqual(owner_problems(index, texts.get), [])

    def test_section_refs(self):
        texts = {"docs/a.md": "# A\n## 7.4 Sessions and storage\n## The split pass (2026-09-10)\n", "x.zig": ""}
        texts["x.zig"] = (
            "// docs/a.md § Sessions and storage; a.md § 7.4; a.md § The split pass; a.md § Gone; docs/b.md § X"
        )
        found = section_problems(list(texts), texts.get)
        self.assertEqual(
            [f.split(": ", 1)[1] for f in found],
            ["a.md has no section § Gone", "names docs/b.md, which does not exist"],
        )

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
