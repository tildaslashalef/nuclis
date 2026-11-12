#!/usr/bin/env python3
"""Runs one fuzz test under Zig's fuzzer for a wall-clock budget.

`zig build fuzz -Dfuzz-target=NAME --fuzz=N` runs N inputs and reports; this
repeats it in chunks until SECONDS have passed, in a cache of the target's
own (`.zig-cache/fuzz/NAME`), where the corpus the fuzzer grows carries over
from one chunk and one run to the next. A crash stops the run and names the
saved input, which `zig build fuzz` reads back as a corpus entry. With
`--list`, prints the targets: every `test "fuzz NAME: ..."` in the tree.
docs/development.md § Hostile input.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEST = re.compile(r'^test "fuzz ([a-z0-9_-]+):', re.M)


def targets() -> dict[str, Path]:
    """Every fuzz target, by name, with the file that declares it."""
    found: dict[str, Path] = {}
    for top in ("src", "inference/src"):
        for path in sorted((ROOT / top).rglob("*.zig")):
            for name in TEST.findall(path.read_text(encoding="utf-8")):
                found[name] = path.relative_to(ROOT)
    return found


def chunk(zig: str, target: str, runs: str, seconds: float) -> tuple[int | None, str]:
    """One `--fuzz=runs` invocation; None for the exit code when cut at the deadline."""
    command = [
        zig,
        "build",
        "fuzz",
        f"-Dfuzz-target={target}",
        "-Dmetal=false",
        "--cache-dir",
        f".zig-cache/fuzz/{target}",
        f"--fuzz={runs}",
    ]
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=max(seconds, 1.0))
        return process.returncode, output
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
        return None, output


def report(output: str) -> str:
    """The fuzzer's closing report, or the output's tail when there is none."""
    start = output.rfind("======= FUZZING REPORT")
    return output[start:].strip() if start >= 0 else "\n".join(output.strip().splitlines()[-20:])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", help="the fuzz target's name")
    parser.add_argument("--seconds", type=float, default=60.0, help="wall-clock budget (default 60)")
    parser.add_argument("--runs", default="200K", help="inputs per chunk (default 200K)")
    parser.add_argument("--zig", default=os.environ.get("ZIG", "zig"))
    parser.add_argument("--list", action="store_true", help="print the targets and exit")
    args = parser.parse_args()

    known = targets()
    if args.list or not args.target:
        for name, path in known.items():
            print(f"{name:12} {path}")
        return 0 if args.list else 2
    if args.target not in known:
        print(f"no fuzz target {args.target!r}; `make fuzz` lists them", file=sys.stderr)
        return 2

    deadline = time.monotonic() + args.seconds
    chunks = 0
    last = ""
    while (left := deadline - time.monotonic()) > 0:
        code, output = chunk(args.zig, args.target, args.runs, left)
        if code is None:
            break
        chunks += 1
        last = output
        if code != 0:
            print(report(output))
            print(
                f"\n{args.target}: failed in chunk {chunks}; the input is under .zig-cache/fuzz/{args.target}/f/crash",
                file=sys.stderr,
            )
            return 1
    print(report(last) if last else "no chunk finished inside the budget")
    print(f"\n{args.target}: {chunks} chunks of {args.runs} in {args.seconds:.0f} s, no failure")
    return 0


if __name__ == "__main__":
    sys.exit(main())
