#!/usr/bin/env python3
"""The agent's scratch project: a tiny dependency-free Python geometry package
that the agent's task list (agent-eval.py) and the README's GIF
(agent-demo.py) run against. Nothing of it is committed to nuclis: `ensure`
generates it on first use under .zig-cache/playground/, from the text below
and fixed seeds, and commits it with `git init` so the workspace's own `make
reset` returns to that baseline. A version stamp inside its `.git` rebuilds
it when this file changes what it writes.

    python3 scripts/playground.py            # build it if needed, print its path
    python3 scripts/playground.py --force    # rebuild it

It holds only what the tasks touch: the `shapes` package (whose polygon area
is off by a factor of two on purpose, a task's scope check), the CLI, the
tests, a design note whose `Error handling` heading appears twice, a seeded
release history, 400 measurements, a file over 1 MiB, and a script that
exits 7 silently.
"""

import argparse
import math
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ROOT / ".zig-cache" / "playground"
# Bump when the files below change, so existing workspaces are rebuilt.
VERSION = "1"
STAMP = "nuclis-playground"

README = """\
# playground

A tiny Python project the nuclis agent works in.

## Layout

- `src/shapes/`: a small geometry package (`circle`, `rect`, `polygon`).
- `src/cli.py`: a command-line entry point that prints shape metrics.
- `tests/`: unittest-based tests, run with `make test`.
- `docs/`: a short design document and a long release history.
- `data/`: fixtures: `measurements.txt` (400 lines) and `big.txt` (over 1 MiB).
- `scripts/`: shell situations for the bash tool.

## Running

```sh
make test
python3 src/cli.py circle 2
python3 src/cli.py rect 3 4
```

## Conventions

Functions are pure and take floats. Errors are raised as `ValueError`
with a short message. Keep the package dependency-free.
"""

MAKEFILE = """\
.PHONY: test reset

test:
\tpython3 -m unittest discover -s tests -v

# Back to the committed baseline: discards every change and removes untracked
# and ignored files (agent-written files, __pycache__).
reset:
\tgit checkout -- .
\tgit clean -fdxq
\tgit status --short
"""

INIT = '''\
"""Geometry helpers for the playground project."""

from .circle import circle_area, circle_perimeter
from .polygon import polygon_area, polygon_perimeter
from .rect import rect_area, rect_perimeter

__all__ = [
    "circle_area",
    "circle_perimeter",
    "polygon_area",
    "polygon_perimeter",
    "rect_area",
    "rect_perimeter",
]
'''

CIRCLE = '''\
"""Circle metrics."""

import math


def circle_area(radius: float) -> float:
    """Area of a circle. Raises ValueError on a negative radius."""
    if radius < 0:
        raise ValueError("radius must be non-negative")
    return math.pi * radius * radius


def circle_perimeter(radius: float) -> float:
    """Circumference of a circle. Raises ValueError on a negative radius."""
    if radius < 0:
        raise ValueError("radius must be non-negative")
    return 2 * math.pi * radius
'''

POLYGON = '''\
"""Regular polygon metrics."""

import math


def polygon_area(sides: int, length: float) -> float:
    """Area of a regular polygon. Raises ValueError below 3 sides or a negative length."""
    if sides < 3:
        raise ValueError("a polygon needs at least 3 sides")
    if length < 0:
        raise ValueError("length must be non-negative")
    return (sides * length * length) / (2 * math.tan(math.pi / sides))


def polygon_perimeter(sides: int, length: float) -> float:
    """Perimeter of a regular polygon. Raises ValueError below 3 sides or a negative length."""
    if sides < 3:
        raise ValueError("a polygon needs at least 3 sides")
    if length < 0:
        raise ValueError("length must be non-negative")
    return sides * length
'''

RECT = '''\
"""Rectangle metrics."""


def rect_area(width: float, height: float) -> float:
    """Area of a rectangle. Raises ValueError on a negative side."""
    if width < 0 or height < 0:
        raise ValueError("sides must be non-negative")
    return width * height


def rect_perimeter(width: float, height: float) -> float:
    """Perimeter of a rectangle. Raises ValueError on a negative side."""
    if width < 0 or height < 0:
        raise ValueError("sides must be non-negative")
    return 2 * (width + height)
'''

CLI = '''\
"""Command-line entry point: `python3 src/cli.py circle 2` or `rect 3 4`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from shapes import circle_area, circle_perimeter, rect_area, rect_perimeter  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: cli.py circle <r> | rect <w> <h>", file=sys.stderr)
        return 2
    shape = argv[1]
    try:
        if shape == "circle":
            r = float(argv[2])
            print(f"area={circle_area(r):.3f} perimeter={circle_perimeter(r):.3f}")
        elif shape == "rect":
            w, h = float(argv[2]), float(argv[3])
            print(f"area={rect_area(w, h):.3f} perimeter={rect_perimeter(w, h):.3f}")
        else:
            print(f"unknown shape: {shape}", file=sys.stderr)
            return 2
    except (IndexError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
'''

TESTS = """\
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from shapes import circle_area, circle_perimeter, rect_area, rect_perimeter  # noqa: E402


class CircleTests(unittest.TestCase):
    def test_area(self):
        self.assertAlmostEqual(circle_area(1), math.pi)

    def test_perimeter(self):
        self.assertAlmostEqual(circle_perimeter(1), 2 * math.pi)

    def test_negative(self):
        with self.assertRaises(ValueError):
            circle_area(-1)


class RectTests(unittest.TestCase):
    def test_area(self):
        self.assertEqual(rect_area(3, 4), 12)

    def test_perimeter(self):
        self.assertEqual(rect_perimeter(3, 4), 14)


if __name__ == "__main__":
    unittest.main()
"""

DESIGN = """\
# Design

## Goals

Keep the package tiny and dependency-free so it can be read in one sitting.

## Error handling

Every public function validates its inputs and raises `ValueError` with a
short, lowercase message. The CLI maps that to exit status 1.

## Error handling

(The heading above is repeated on purpose: it makes `edit_file` refuse a
non-unique match until more context is given.)
"""

EXIT7 = """\
#!/bin/sh
# Silent failure: no output, exit status 7.
exit 7
"""


def history() -> str:
    """48 release sections of seeded prose, newest last, two planted facts."""
    rng = random.Random(5)
    areas = ["cli", "shapes", "docs", "tests", "build"]
    verbs = ["add", "fix", "tidy", "speed up", "document", "rename"]
    things = ["the parser", "rounding", "the help text", "a test", "the Makefile", "error messages"]
    out = ["# Release history", "", "One entry per change, newest last.", ""]
    for minor in range(0, 8):
        for patch in range(0, 6):
            v = f"0.{minor}.{patch}"
            out.append(f"## v{v}")
            out.append("")
            for _ in range(rng.randint(4, 8)):
                out.append(f"- {rng.choice(areas)}: {rng.choice(verbs)} {rng.choice(things)}")
            if v == "0.4.2":
                out.append("- cli: remove the --legacy flag; it printed two decimals and nobody used it")
            if v == "0.6.3":
                out.append("- shapes: raise ValueError instead of returning NaN for negative sides")
            out.append("")
    return "\n".join(out) + "\n"


def measurements() -> str:
    """400 numbered circles: `0001 radius=3.274 area=33.675`."""
    rng = random.Random(3)
    lines = []
    for i in range(1, 401):
        r = rng.uniform(0.1, 7.0)
        lines.append(f"{i:04d} radius={r:.3f} area={math.pi * r * r:.3f}")
    return "\n".join(lines) + "\n"


def big_text() -> str:
    """Over 1 MiB and 32,000 lines, past read_file's caps and edit_file's limit,
    with one NEEDLE line."""
    rng = random.Random(7)
    lines = []
    for i in range(1, 32001):
        word = rng.choice(["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"])
        lines.append(f"{i:05d} {word} value={rng.random():.6f} tag=t{rng.randint(0, 99):02d}")
    lines[12345] = "12346 NEEDLE value=0.000000 tag=t00  # the one line with NEEDLE"
    return "\n".join(lines) + "\n"


def files() -> dict[str, str]:
    return {
        "README.md": README,
        "Makefile": MAKEFILE,
        ".gitignore": "__pycache__/\n*.pyc\n",
        "src/shapes/__init__.py": INIT,
        "src/shapes/circle.py": CIRCLE,
        "src/shapes/polygon.py": POLYGON,
        "src/shapes/rect.py": RECT,
        "src/cli.py": CLI,
        "tests/test_shapes.py": TESTS,
        "docs/design.md": DESIGN,
        "docs/history.md": history(),
        "data/measurements.txt": measurements(),
        "data/big.txt": big_text(),
        "scripts/exit7.sh": EXIT7,
    }


def git(path: Path, *args: str) -> None:
    # A fixed identity and date make the baseline commit the same everywhere;
    # the user's hooks and signing do not apply to a scratch repository.
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "playground",
        "GIT_AUTHOR_EMAIL": "playground@nuclis.invalid",
        "GIT_COMMITTER_NAME": "playground",
        "GIT_COMMITTER_EMAIL": "playground@nuclis.invalid",
        "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z",
        "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z",
    }
    config = ["-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", "-c", "init.defaultBranch=main"]
    _ = subprocess.run(["git", *config, *args], cwd=path, env=env, check=True, capture_output=True)


def current(path: Path) -> bool:
    stamp = path / ".git" / STAMP
    return stamp.is_file() and stamp.read_text().strip() == VERSION


def ensure(path: Path = DEFAULT, force: bool = False) -> Path:
    """The workspace at `path`, generated if it is missing or stale. Refuses a
    directory it did not generate, so a real project is never deleted."""
    if current(path) and not force:
        return path
    if path.exists():
        if not (path / ".git" / STAMP).is_file():
            sys.exit(f"{path} exists and is not a generated playground; not touching it")
        shutil.rmtree(path)
    path.mkdir(parents=True)
    for rel, text in files().items():
        target = path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        _ = target.write_text(text, newline="")
    (path / "scripts" / "exit7.sh").chmod(0o755)
    git(path, "init", "-q")
    git(path, "add", "-A")
    git(path, "commit", "-qm", "playground: the baseline")
    _ = (path / ".git" / STAMP).write_text(VERSION + "\n")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _ = ap.add_argument("--dir", type=Path, default=DEFAULT, help="where the workspace lives (default %(default)s)")
    _ = ap.add_argument("--force", action="store_true", help="rebuild it even if it is current")
    args = ap.parse_args()
    print(ensure(args.dir, args.force))


if __name__ == "__main__":
    main()
