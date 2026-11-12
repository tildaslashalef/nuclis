#!/usr/bin/env python3
"""Fails on a construct that can panic, in code that reads outside input.

Outside `test` blocks of the files in `FILES`, each of these must carry a
trailing `// checked: <where>` comment naming the guard that keeps it from
firing on any input: `catch unreachable`, `unreachable`, `@intCast(`,
`std.debug.assert(` (or an `assert(` alias), and `.?` on an optional. The
comment is the reviewable part; the script cannot prove the guard holds.
docs/development.md § Hostile input.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# The parsers of outside bytes: requests, sessions, tool arguments, terminal
# input, model output, and model, audio, and image files.
FILES = (
    "src/api/**/*.zig",
    "src/agent/session.zig",
    "src/agent/resume.zig",
    "src/agent/tools/**/*.zig",
    "src/tui/keys.zig",
    "src/tui/terminal.zig",
    "src/tui/markdown.zig",
    "inference/src/formats/**/*.zig",
    "inference/src/profiles/stream.zig",
    "inference/src/audio/audio.zig",
    "inference/src/vision/image.zig",
)
# Writes a file, never reads one.
EXCLUDED = ("inference/src/formats/gguf_writer.zig",)

PATTERNS = (
    ("catch unreachable", re.compile(r"\bcatch\s+unreachable\b")),
    ("unreachable", re.compile(r"(?<!catch )\bunreachable\b")),
    ("@intCast", re.compile(r"@intCast\(")),
    ("assert", re.compile(r"(?<![\w.])(?:std\.debug\.)?assert\(")),
    (".?", re.compile(r"\.\?")),
)
ESCAPE = re.compile(r"//\s*checked:\s*\S")
TEST_START = re.compile(r"^test\b")
DECL = re.compile(r"^(pub\s+)?(?:inline\s+|extern\s+|export\s+)?(?:fn|const|var)\s+(\w+)")


@dataclass(frozen=True)
class Hit:
    path: str
    line: int
    kind: str
    text: str


def code_of(line: str) -> tuple[str, str]:
    """The line's code with string and character literals blanked, and its comment."""
    stripped = line.lstrip()
    if stripped.startswith("\\\\"):
        return "", ""
    out: list[str] = []
    i = 0
    while i < len(line):
        c = line[i]
        if c == "/" and line.startswith("//", i):
            return "".join(out), line[i:]
        if c in "\"'":
            j = i + 1
            while j < len(line) and line[j] != c:
                j += 2 if line[j] == "\\" else 1
            out.append(c + " " * max(0, min(j, len(line)) - i - 1) + c)
            i = j + 1
            continue
        out.append(c)
        i += 1
    return "".join(out), ""


@dataclass
class Decl:
    name: str
    public: bool
    first: int
    last: int


def regions(codes: list[str]) -> tuple[list[tuple[int, int]], list[Decl]]:
    """The `test` blocks and the other top-level declarations, as 0-based
    inclusive line ranges."""
    tests: list[tuple[int, int]] = []
    decls: list[Decl] = []
    i = 0
    while i < len(codes):
        code = codes[i]
        test = TEST_START.match(code)
        decl = DECL.match(code)
        if not test and not decl:
            i += 1
            continue
        depth = 0
        j = i
        while j < len(codes):
            depth += codes[j].count("{") - codes[j].count("}")
            if depth <= 0 and codes[j].rstrip().endswith((";", "}")):
                break
            j += 1
        if test:
            tests.append((i, j))
        elif decl:
            decls.append(Decl(decl.group(2), decl.group(1) is not None, i, j))
        i = j + 1
    return tests, decls


def test_only(codes: list[str], tests: list[tuple[int, int]], decls: list[Decl]) -> list[tuple[int, int]]:
    """The ranges of the non-`pub` declarations that only tests, and other
    such declarations, refer to: test support, wherever it sits."""
    covered = list(tests)
    marked: set[str] = set()
    changed = True
    while changed:
        changed = False
        for d in decls:
            if d.public or d.name in marked:
                continue
            word = re.compile(rf"\b{re.escape(d.name)}\b")
            uses = [n for n, code in enumerate(codes) if not d.first <= n <= d.last and word.search(code)]
            if uses and all(any(a <= n <= b for a, b in covered) for n in uses):
                marked.add(d.name)
                covered.append((d.first, d.last))
                changed = True
    return covered


def scan(path: str, source: str) -> Iterator[Hit]:
    """Every unescaped hit outside tests and test support."""
    lines = source.splitlines()
    split = [code_of(line) for line in lines]
    codes = [code for code, _ in split]
    tests, decls = regions(codes)
    skipped = test_only(codes, tests, decls)
    for n, (code, comment) in enumerate(split):
        if any(a <= n <= b for a, b in skipped) or ESCAPE.search(comment):
            continue
        for kind, pattern in PATTERNS:
            if pattern.search(code):
                yield Hit(path, n + 1, kind, lines[n].strip())


def files() -> list[Path]:
    found: set[Path] = set()
    for pattern in FILES:
        found.update(ROOT.glob(pattern))
    return sorted(p for p in found if str(p.relative_to(ROOT)) not in EXCLUDED)


def self_test() -> None:
    source = "\n".join(
        [
            "const a = b.?;",
            "const c = @intCast(d); // checked: d < 256 above",
            'const s = "x.? unreachable";',
            "// a comment with .? and assert(x)",
            "x catch unreachable;",
            "std.debug.assert(ok);",
            "assert(ok);",
            "self.assert(ok);",
            "\\\\ multiline .?",
            "else => unreachable,",
            "const q: u8 = '?';",
            'test "t" {',
            "    const y = z.?;",
            "    if (w) {",
            "        unreachable;",
            "    }",
            "}",
            "const after = e.?;",
            "fn helper() void {",
            "    _ = h.?;",
            "}",
            'test "uses helper" {',
            "    helper();",
            "}",
            "fn used() void {",
            "    _ = u.?;",
            "}",
            "pub fn api() void {",
            "    used();",
            "}",
        ]
    )
    got = [(h.line, h.kind) for h in scan("t.zig", source)]
    want = [
        (1, ".?"),
        (5, "catch unreachable"),
        (6, "assert"),
        (7, "assert"),
        (10, "unreachable"),
        (18, ".?"),
        (26, ".?"),
    ]
    if got != want:
        raise SystemExit(f"self-test failed:\n got  {got}\n want {want}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("input-safety-check: self-test passed")
        return 0
    hits = [h for p in files() for h in scan(str(p.relative_to(ROOT)), p.read_text(encoding="utf-8"))]
    for h in hits:
        print(f"{h.path}:{h.line}: {h.kind} without `// checked: <where>`: {h.text}")
    if hits:
        print(f"input-safety-check: {len(hits)} unguarded construct(s) in input-facing code", file=sys.stderr)
        return 1
    print(f"input-safety-check: {len(files())} files clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
