#!/usr/bin/env python3
"""Pinned llama-perplexity run, written as the JSON `nuclis eval --reference` reads.

Runs the pinned llama.cpp checkout's `llama-perplexity` on a text file at a
window size and count, parses its running and final estimates, and records
the text's SHA-256 (eval refuses a different file), the exact flags, and the
checkout revision. `--ubatch 1` makes the reference decode one token per
kernel call: on Gemma 4 12B its batched path disagrees with its own
per-token path by 1.2 % (docs/reference/eval.md), and the per-token path is
the one its trace harness validates.
"""

import argparse
import datetime
import hashlib
import json
import pathlib
import re
import subprocess
import sys

REVISION = "7620399f58aebfd2196b74021f9581bcf7218cb9"
ROOT = pathlib.Path(__file__).resolve().parent.parent
CHECKOUT = ROOT / ".zig-cache/reference/llama.cpp"
BINARY = CHECKOUT / "build/bin/llama-perplexity"

RUNNING = re.compile(r"\[(\d+)\](\d+\.\d+),")
FINAL = re.compile(r"Final estimate: PPL = (\d+\.\d+) \+/- (\d+\.\d+)")


def parse(log):
    """The running estimate after each window and the final one with its error."""
    running = [float(v) for _, v in RUNNING.findall(log)]
    final = FINAL.search(log)
    if final is None or not running:
        raise ValueError("no perplexity estimate in the reference output")
    return running, float(final.group(1)), float(final.group(2))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def self_test():
    log = ("perplexity: calculating perplexity over 2 chunks, n_ctx=512\n"
           "[1]4.3434,[2]6.1586,\n"
           "0.51.372.778 I Final estimate: PPL = 6.1586 +/- 0.36229\n")
    running, ppl, error = parse(log)
    assert running == [4.3434, 6.1586] and ppl == 6.1586 and error == 0.36229
    try:
        parse("no estimate")
    except ValueError:
        pass
    else:
        raise AssertionError("an output without an estimate must be refused")
    print("reference-perplexity.py self-test passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model", type=pathlib.Path)
    parser.add_argument("--text", type=pathlib.Path)
    parser.add_argument("--ctx", type=int, default=512)
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--ubatch", type=int, help="tokens per decode kernel call (1: the per-token path)")
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--tolerance", type=float, help="a bound wider than eval's 0.5 %%, for a model whose reference disagrees with itself")
    parser.add_argument("--tolerance-reason", help="why the bound is wider (required with --tolerance)")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if not (args.model and args.text and args.output):
        parser.error("--model, --text, and --output are required")
    if args.tolerance is not None and not args.tolerance_reason:
        parser.error("--tolerance needs --tolerance-reason")
    revision = subprocess.run(["git", "-C", str(CHECKOUT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if revision != REVISION:
        sys.exit(f"the reference checkout is at {revision}, not the pinned {REVISION}")
    flags = ["-c", str(args.ctx), "--chunks", str(args.chunks), "-ngl", "99"]
    if args.ubatch:
        # The batch must hold one window for the micro-batch to split it.
        flags += ["-b", str(args.ctx), "-ub", str(args.ubatch)]
    proc = subprocess.run([str(BINARY), "-m", str(args.model), "-f", str(args.text)] + flags, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(proc.stderr[-2000:])
    running, ppl, error = parse(proc.stdout + proc.stderr)
    report = {
        "schema_version": 1,
        "tool": "llama-perplexity",
        "revision": revision,
        "model_file": args.model.name,
        "model_bytes": args.model.stat().st_size,
        "text_file": args.text.name,
        "text_sha256": sha256(args.text),
        "ctx": args.ctx,
        "chunks": args.chunks,
        "flags": flags,
        "ppl": ppl,
        "ppl_error": error,
        "chunk_ppl": running,
        "date": datetime.date.today().isoformat(),
    }
    if args.tolerance is not None:
        report["tolerance"] = args.tolerance
        report["tolerance_reason"] = args.tolerance_reason
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(f"{args.output}: PPL {ppl} +/- {error} over {args.chunks} windows of {args.ctx}")


if __name__ == "__main__":
    main()
