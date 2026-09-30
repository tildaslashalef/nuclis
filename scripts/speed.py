#!/usr/bin/env python3
"""The speed loop's A/B driver: a saved base binary against the tree, interleaved.

`--save-base` (make speed-base) copies ./zig-out/bin/nuclis, kernels embedded,
to .zig-cache/speed/base/nuclis with its revision and a dirty flag. A run
(make speed) alternates base and candidate `nuclis bench` processes, pair by
pair and first-mover swapped each pair, per context in `--contexts`, on
prefixes both restore from .zig-cache/speed/prefix (`bench --prefix-cache`;
the first run at a context prefills and saves it). Plain decode compares
tokens/s; `--verify-rows` compares the verify batch cost C in ms
(`bench --verify-rows R --accept a`, speculation on). Each row prints both
medians, the change (positive is faster), the pairs' range of per-pair
changes, and the keep rule's verdict: KEEP at >= 2 % faster, REGRESS below
-1 %, NOISE between; the change is KEEP when one row keeps and none
regresses, over >= 5 pairs. `--json` prints the rows for the ledger.
The contexts name the acceptance token arrays (512, 4096, 16384, 32639);
another length takes that many tokens from the 32,639 array.
See docs/development.md § The speed loop.
"""

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gates  # noqa: E402  (model_path and the gates manifest)

ROOT = Path(__file__).resolve().parent.parent
SPEED = ROOT / ".zig-cache" / "speed"
BASE_DIR = SPEED / "base"
PREFIX_DIR = SPEED / "prefix"
CANDIDATE = ROOT / "zig-out" / "bin" / "nuclis"
FIXTURES = ROOT / "tests" / "fixtures"
ACCEPTANCE = (512, 4096, 16384, 32639)
# Qwen's 32,639-token array is the reference's boundary run, kept apart from
# the run directory the acceptance workload names.
EXTRA_RUNS = {"run-2026-09-06": ["boundary-2026-09-06"]}
KEEP_PERCENT = 2.0
REGRESS_PERCENT = -1.0
MIN_PAIRS = 5


# ---- pure functions (covered by --self-test) --------------------------------


def parse_list(text):
    """`512,4096` -> [512, 4096]; every item a positive integer."""
    values = [int(item) for item in text.split(",") if item.strip()]
    if not values or any(v <= 0 for v in values):
        raise ValueError(f"expected positive integers, got {text!r}")
    return values


def accepted_for(rows, accept):
    """The forced accepted count: `accept` when given, else half the drafts."""
    a = (rows - 1) // 2 if accept is None else accept
    if not 0 <= a < rows:
        raise ValueError(f"--accept {a} is not below {rows} rows")
    return a


def change_percent(base, candidate, higher_is_better):
    """How much faster the candidate is, in percent of the base."""
    if higher_is_better:
        return (candidate - base) / base * 100.0
    return (base - candidate) / base * 100.0


def verdict(delta):
    if delta >= KEEP_PERCENT:
        return "KEEP"
    if delta < REGRESS_PERCENT:
        return "REGRESS"
    return "NOISE"


def summarize(label, base, candidate, higher_is_better):
    """One row from the pairs' measurements (equal-length lists, pair order)."""
    if len(base) != len(candidate) or not base:
        raise ValueError("a row needs one base and one candidate value per pair")
    b, c = statistics.median(base), statistics.median(candidate)
    delta = change_percent(b, c, higher_is_better)
    pairs = [change_percent(x, y, higher_is_better) for x, y in zip(base, candidate, strict=True)]
    return {
        "row": label,
        "unit": "tok/s" if higher_is_better else "ms",
        "pairs": len(base),
        "base": b,
        "candidate": c,
        "change_percent": delta,
        "pair_min_percent": min(pairs),
        "pair_max_percent": max(pairs),
        "verdict": verdict(delta),
        "base_values": base,
        "candidate_values": candidate,
    }


def overall(rows):
    """The keep rule over every row: a keep with no regression, on enough pairs."""
    if not rows or any(r["pairs"] < MIN_PAIRS for r in rows):
        return "INCONCLUSIVE"
    verdicts = {r["verdict"] for r in rows}
    if "REGRESS" in verdicts:
        return "REGRESS"
    return "KEEP" if "KEEP" in verdicts else "NOISE"


def prompt_candidates(run, context):
    """Token-array files for `context` in the family's run, most specific first."""
    names = [run] + EXTRA_RUNS.get(run, [])
    return [FIXTURES / name / f"prompt-{context}.json" for name in names]


def bench_argv(nuclis, model, prompt, tokens, verify_rows=None, accept=0):
    argv = [str(nuclis), "bench", "--backend", "metal", "--model", model, "--prompt-tokens", str(prompt)]
    argv += ["--max-tokens", str(tokens), "--ctx-size", "32768", "--kv", "f16"]
    # No warm-up: with the weights in the page cache a process's first run
    # decodes at its second's rate; only its first token is slower.
    argv += ["--prefix-cache", str(PREFIX_DIR), "--repeat", "1", "--warmup", "0", "--json"]
    if verify_rows is None:
        return argv + ["--speculative", "off"]
    return argv + ["--speculative", "on", "--verify-rows", str(verify_rows), "--accept", str(accept)]


def metric(report, verify):
    """Decode tokens/s of a plain report, or the verify batch cost C in ms."""
    if verify:
        mean = (report.get("verify") or {}).get("mean")
        return None if mean is None else float(mean["total"])
    value = report.get("mean_decode_tokens_per_second")
    return None if value is None else float(value)


# ---- effects -----------------------------------------------------------------


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def save_base():
    if not CANDIDATE.exists():
        sys.exit(f"{CANDIDATE} is missing: build first (make speed-base builds it)")
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    target = BASE_DIR / "nuclis"
    # Removed, never overwritten: macOS keeps the old file's code signature
    # cached and kills a binary rewritten under it.
    target.unlink(missing_ok=True)
    shutil.copy2(CANDIDATE, target)
    info = {"revision": git("rev-parse", "--short", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    (BASE_DIR / "base.json").write_text(json.dumps(info) + "\n")
    print(f"base saved: {target} at {info['revision']}{' (dirty tree)' if info['dirty'] else ''}")


def family_run(model_key):
    workloads = json.loads((ROOT / "workloads.json").read_text())["workloads"]
    entry = workloads.get(f"{model_key}/acceptance")
    if entry is None:
        sys.exit(f"no {model_key}/acceptance workload names this family's token arrays")
    return entry["run"]


def prompt_for(run, context):
    for path in prompt_candidates(run, context):
        if path.exists():
            return path
    if context > max(ACCEPTANCE):
        sys.exit(f"no token array holds {context} tokens")
    source = next((p for p in prompt_candidates(run, max(ACCEPTANCE)) if p.exists()), None)
    if source is None:
        sys.exit(f"no {max(ACCEPTANCE)}-token array for {run}")
    ids = json.loads(source.read_text())[:context]
    path = SPEED / "prompts" / run / f"prompt-{context}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ids))
    return path


def measure(argv, verify):
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"{' '.join(argv)} failed:\n{result.stderr.strip()[-2000:]}")
    value = metric(json.loads(result.stdout), verify)
    if value is None:
        sys.exit(f"{' '.join(argv)} measured nothing")
    return value


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save-base", action="store_true", help="copy the built binary as the base (make speed-base)")
    ap.add_argument("--contexts", default=",".join(map(str, ACCEPTANCE)), help="prompt lengths (default: %(default)s)")
    ap.add_argument("--pairs", type=int, default=MIN_PAIRS, help="interleaved pairs per row (default: %(default)s)")
    ap.add_argument("--tokens", type=int, default=32, help="decode tokens, or verify batches, per run (%(default)s)")
    ap.add_argument("--verify-rows", help="also compare the verify batch cost at these row counts, e.g. 1,4,8")
    ap.add_argument("--accept", type=int, help="drafts each verify batch accepts (default: half the drafts)")
    ap.add_argument("--no-decode", action="store_true", help="only the verify rows")
    ap.add_argument("--model", default="qwen38", help="the gates.json model key (default: %(default)s)")
    ap.add_argument("--json", action="store_true", help="print the rows as JSON")
    ap.add_argument("--self-test", action="store_true", help="run the unit tests of the pure functions")
    args = ap.parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(SelfTest)
        sys.exit(0 if unittest.TextTestRunner(verbosity=0).run(suite).wasSuccessful() else 1)
    if args.save_base:
        save_base()
        return
    base = BASE_DIR / "nuclis"
    if not base.exists():
        sys.exit("no base binary: run `make speed-base` at the revision to compare against")
    info = json.loads((BASE_DIR / "base.json").read_text())
    gdoc = json.loads((ROOT / "gates.json").read_text())
    if args.model not in gdoc["models"]:
        sys.exit(f"unknown model {args.model}; gates.json has {', '.join(gdoc['models'])}")
    model = gdoc["models"][args.model].get("entry") or gates.model_path(gdoc, args.model)
    run = family_run(args.model)
    modes = [] if args.no_decode else [None]
    modes += parse_list(args.verify_rows) if args.verify_rows else []
    rows = []
    print(
        f"base {info['revision']}{' (dirty)' if info['dirty'] else ''} vs tree "
        f"{git('rev-parse', '--short', 'HEAD')}{' (dirty)' if git('status', '--porcelain') else ''}, "
        f"{args.model}, {args.pairs} pairs",
        file=sys.stderr,
    )
    for context in parse_list(args.contexts):
        prompt = prompt_for(run, context)
        for rows_n in modes:
            verify = rows_n is not None
            accept = accepted_for(rows_n, args.accept) if verify else 0
            label = f"{context} verify R={rows_n} a={accept}" if verify else f"{context} decode"
            # One unmeasured run first, so a missing saved prefix is prefilled
            # outside the pairs.
            subprocess.run(bench_argv(CANDIDATE, model, prompt, 2, rows_n, accept), cwd=ROOT, capture_output=True)
            base_values, candidate_values = [], []
            for i in range(args.pairs):
                order = [(base, base_values), (CANDIDATE, candidate_values)]
                for binary, values in order if i % 2 == 0 else reversed(order):
                    values.append(measure(bench_argv(binary, model, prompt, args.tokens, rows_n, accept), verify))
            row = summarize(label, base_values, candidate_values, higher_is_better=not verify)
            rows.append(row)
            print(
                f"{label:<24} base {row['base']:9.2f} cand {row['candidate']:9.2f} {row['unit']:<5} "
                f"{row['change_percent']:+6.2f} % (pairs {row['pair_min_percent']:+.2f}..{row['pair_max_percent']:+.2f})"
                f"  {row['verdict']}",
                file=sys.stderr,
            )
    result = {"base": info, "model": args.model, "rows": rows, "verdict": overall(rows)}
    print(f"verdict: {result['verdict']}", file=sys.stderr)
    if args.json:
        print(json.dumps(result, indent=2))


class SelfTest(unittest.TestCase):
    def test_parse_list(self):
        self.assertEqual(parse_list("512, 4096"), [512, 4096])
        with self.assertRaises(ValueError):
            parse_list("0")

    def test_accepted_for(self):
        self.assertEqual(accepted_for(4, None), 1)
        self.assertEqual(accepted_for(8, None), 3)
        self.assertEqual(accepted_for(1, None), 0)
        with self.assertRaises(ValueError):
            accepted_for(4, 4)

    def test_change_direction(self):
        self.assertAlmostEqual(change_percent(10.0, 10.5, True), 5.0)
        self.assertAlmostEqual(change_percent(200.0, 190.0, False), 5.0)

    def test_verdicts(self):
        self.assertEqual(verdict(2.0), "KEEP")
        self.assertEqual(verdict(1.99), "NOISE")
        self.assertEqual(verdict(-1.0), "NOISE")
        self.assertEqual(verdict(-1.01), "REGRESS")

    def test_summarize_uses_medians_and_pair_changes(self):
        row = summarize("512 decode", [10.0, 10.2, 9.8, 10.0, 10.1], [10.3, 10.4, 10.2, 10.3, 10.5], True)
        self.assertEqual(row["base"], 10.0)
        self.assertEqual(row["candidate"], 10.3)
        self.assertAlmostEqual(row["change_percent"], 3.0)
        self.assertEqual(row["verdict"], "KEEP")
        self.assertAlmostEqual(row["pair_min_percent"], (10.4 - 10.2) / 10.2 * 100)

    def test_overall_rule(self):
        def row(v, pairs=5):
            return {"verdict": v, "pairs": pairs}

        self.assertEqual(overall([row("KEEP"), row("NOISE")]), "KEEP")
        self.assertEqual(overall([row("KEEP"), row("REGRESS")]), "REGRESS")
        self.assertEqual(overall([row("NOISE")]), "NOISE")
        self.assertEqual(overall([row("KEEP", pairs=3)]), "INCONCLUSIVE")

    def test_bench_argv(self):
        plain = bench_argv("n", "m", "p.json", 32)
        self.assertEqual(plain[-2:], ["--speculative", "off"])
        self.assertIn("--prefix-cache", plain)
        verify = bench_argv("n", "m", "p.json", 16, 4, 1)
        self.assertEqual(verify[-6:], ["--speculative", "on", "--verify-rows", "4", "--accept", "1"])

    def test_metric(self):
        self.assertEqual(metric({"mean_decode_tokens_per_second": 10.5}, False), 10.5)
        self.assertEqual(metric({"verify": {"mean": {"total": 280.9}}}, True), 280.9)
        self.assertIsNone(metric({"verify": {"mean": None}}, True))

    def test_prompt_candidates(self):
        paths = prompt_candidates("run-2026-09-06", 32639)
        self.assertEqual([p.parent.name for p in paths], ["run-2026-09-06", "boundary-2026-09-06"])
        self.assertTrue(any(p.exists() for p in paths) or not FIXTURES.exists())


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
