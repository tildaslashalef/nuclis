#!/usr/bin/env python3
"""The speculative matrix: real speculation, off/on pairs, per family and cell.

A cell is (context, sampling, draft length, p_min). Each runs one `nuclis
bench --speculative on` process: `--repeat` off/on pairs on one loaded
model, 128 output tokens, F16 KV, context 32,768. A numeric context restores
the speed loop's saved prefix of the family's acceptance array
(.zig-cache/speed/prefix, prefilled by the first run that needs it); `code`
is the fixed short prompt of the ENGN-17 record. Reports are saved under
.zig-cache/spec/<model>/<rev>/ and a cell already saved there is not re-run,
so an interrupted matrix resumes. The table gives, per cell, accepted and
proposed drafts per batch, E (emitted tokens per batch), the per-batch costs
and their sum C, C / 50E (20 tok/s needs <= 1), the off and on rates, and the
speedup: the median of the pairs' on/off ratios.
See TODO.md (ENGN-20) and docs/development.md § The speed loop.
"""

import argparse
import json
import statistics
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gates  # noqa: E402
import speed  # noqa: E402  (the saved prefixes and the family token arrays)

ROOT = speed.ROOT
OUT = ROOT / ".zig-cache" / "spec"
CODE_PROMPT = "Write a Zig function that reverses a string."
# Each profile's think-off options ("instruct") and Qwen's think-on ones, the
# agent's default; greedy is the deterministic control.
SAMPLING = {
    "qwen": {
        "instruct": {"temperature": 0.7, "top-p": 0.8, "top-k": 20, "presence-penalty": 1.5},
        "thinking": {"temperature": 1.0, "top-p": 0.95, "top-k": 20},
    },
    "card": {"instruct": {"temperature": 1.0, "top-p": 0.95, "top-k": 64}},
}


# ---- pure functions (covered by --self-test) --------------------------------


def parse_ints(text):
    """`2-7` or `2,4,7` (or a mix) as a sorted list."""
    out = set()
    for part in text.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.update(range(int(lo), int(hi) + 1))
        elif part:
            out.add(int(part))
    return sorted(out)


def sampling_flags(model_key, sampling):
    if sampling == "greedy":
        return []
    table = SAMPLING["qwen" if model_key.startswith("qwen") else "card"]
    if sampling not in table:
        raise ValueError(f"{model_key} has no {sampling} sampling")
    flags = []
    for key, value in table[sampling].items():
        flags += [f"--{key}", str(value)]
    return flags + ["--seed", "0"]


def cell_name(context, sampling, draft, p_min):
    return f"{context}-{sampling}-d{draft}" + ("" if p_min is None else f"-p{p_min:g}")


def bench_argv(nuclis, model, model_key, prompt, sampling, draft, p_min, repeat, tokens):
    argv = [str(nuclis), "bench", "--backend", "metal", "--model", model]
    if prompt is None:
        argv += ["--prompt", CODE_PROMPT, "--raw"]
    else:
        argv += ["--prompt-tokens", str(prompt), "--prefix-cache", str(speed.PREFIX_DIR)]
    argv += ["--max-tokens", str(tokens), "--ctx-size", "32768", "--kv", "f16"]
    argv += ["--repeat", str(repeat), "--warmup", "0"]
    argv += ["--speculative", "on", "--draft-length", str(draft)]
    if p_min is not None:
        argv += ["--draft-p-min", str(p_min)]
    return argv + sampling_flags(model_key, sampling) + ["--json"]


def row_of(report):
    """The derived row of one cell's report (totals over its on runs)."""
    samples = report["samples"]
    off = [s for s in samples if not s["speculative"]]
    on = [s for s in samples if s["speculative"]]
    if not off or not on or len(off) != len(on):
        return None
    batches = sum(s["speculative_steps"] for s in on)
    if batches == 0:
        return None

    def per_batch(field):
        return sum(s.get(field) or 0.0 for s in on) / batches

    emitted = sum(s["generated_tokens"] - 1 for s in on)
    c = sum(s["decode_milliseconds"] for s in on) / batches
    e = emitted / batches
    ratios = [b["decode_tokens_per_second"] / a["decode_tokens_per_second"] for a, b in zip(off, on)]
    return {
        "pairs": len(on),
        "accepted": sum(s["accepted_per_step"] * s["speculative_steps"] for s in on) / batches,
        "proposed": sum(s["proposed_per_step"] * s["speculative_steps"] for s in on) / batches,
        "E": e,
        "propose": per_batch("propose_milliseconds"),
        "checkpoint": per_batch("checkpoint_milliseconds"),
        "verify": per_batch("verify_milliseconds"),
        "recover": per_batch("recover_milliseconds"),
        "commit": per_batch("commit_milliseconds"),
        "C": c,
        "C_over_50E": c / (50.0 * e),
        "off": statistics.mean(s["decode_tokens_per_second"] for s in off),
        "on": statistics.mean(s["decode_tokens_per_second"] for s in on),
        "speedup": statistics.median(ratios),
        "speedup_min": min(ratios),
        "speedup_max": max(ratios),
        "stops": sorted({s["stop_reason"] for s in samples}),
    }


HEADER = (
    "| context | sampling | draft | p_min | accepted | proposed | E | propose | verify | recover | commit | C ms "
    "| C / 50E | off → on tok/s | speedup (pairs) |\n"
    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
)


def table_line(cell, row):
    context = cell["context"] if cell["context"] == "code" else f"{int(cell['context']):,}"
    return (
        f"| {context} | {cell['sampling']} | {cell['draft']} | {cell['p_min']:g} | {row['accepted']:.2f} | {row['proposed']:.2f} "
        f"| {row['E']:.2f} | {row['propose']:.1f} | {row['verify']:.1f} | {row['recover']:.1f} | {row['commit']:.1f} "
        f"| {row['C']:.1f} | {row['C_over_50E']:.2f} | {row['off']:.2f} → {row['on']:.2f} "
        f"| {row['speedup']:.3f}× ({row['speedup_min']:.2f}–{row['speedup_max']:.2f}) |"
    )


# ---- effects -----------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen38", help="the gates.json model key (default: %(default)s)")
    ap.add_argument("--contexts", default="512,4096", help="prompt lengths and/or `code` (default: %(default)s)")
    ap.add_argument("--sampling", default="greedy,instruct", help="greedy, instruct, thinking (Qwen)")
    ap.add_argument("--drafts", default="2-7", help="draft lengths, e.g. 2-7 or 2,4,7 (default: %(default)s)")
    ap.add_argument("--p-min", help="proposal thresholds, e.g. 0,0.5,0.7 (default: the engine's)")
    ap.add_argument("--repeat", type=int, default=3, help="off/on pairs per cell (default: %(default)s)")
    ap.add_argument("--tokens", type=int, default=128, help="output tokens per run (default: %(default)s)")
    ap.add_argument("--rev", help="read or write the reports of this revision (default: HEAD, +dirty)")
    ap.add_argument("--fresh", action="store_true", help="re-run cells already saved")
    ap.add_argument("--report", action="store_true", help="only print the saved cells, run nothing")
    ap.add_argument("--json", action="store_true", help="print the rows as JSON")
    ap.add_argument("--self-test", action="store_true", help="run the unit tests of the pure functions")
    args = ap.parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(SelfTest)
        sys.exit(0 if unittest.TextTestRunner(verbosity=0).run(suite).wasSuccessful() else 1)
    gdoc = json.loads((ROOT / "gates.json").read_text())
    if args.model not in gdoc["models"]:
        sys.exit(f"unknown model {args.model}; gates.json has {', '.join(gdoc['models'])}")
    # A pair names the catalogue entry so its draft companion resolves.
    model = gdoc["models"][args.model].get("entry") or gates.model_path(gdoc, args.model)
    rev = args.rev or speed.git("rev-parse", "--short", "HEAD") + (
        "+dirty" if speed.git("status", "--porcelain") else ""
    )
    out_dir = OUT / args.model / rev
    out_dir.mkdir(parents=True, exist_ok=True)
    run = speed.family_run(args.model)
    p_mins = [None] if args.p_min is None else [float(p) for p in args.p_min.split(",")]
    rows = []
    print(HEADER)
    for context in args.contexts.split(","):
        prompt = None if context == "code" else speed.prompt_for(run, int(context))
        for sampling in args.sampling.split(","):
            for draft in parse_ints(args.drafts):
                for p_min in p_mins:
                    path = out_dir / (cell_name(context, sampling, draft, p_min) + ".json")
                    if args.fresh or not path.exists():
                        if args.report:
                            continue
                        argv = bench_argv(
                            speed.CANDIDATE, model, args.model, prompt, sampling, draft, p_min, args.repeat, args.tokens
                        )
                        result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
                        if result.returncode != 0:
                            sys.exit(f"{' '.join(argv)} failed:\n{result.stderr.strip()[-2000:]}")
                        path.write_text(result.stdout)
                    report = json.loads(path.read_text())
                    row = row_of(report)
                    if row is None:
                        print(f"{path.name}: no measured pair", file=sys.stderr)
                        continue
                    cell = {
                        "context": context,
                        "sampling": sampling,
                        "draft": draft,
                        "p_min": report.get("speculative_p_min", 0.7),
                    }
                    rows.append({**cell, **row})
                    print(table_line(cell, row), flush=True)
    if args.json:
        print(json.dumps({"model": args.model, "revision": rev, "rows": rows}, indent=2))


class SelfTest(unittest.TestCase):
    def test_parse_ints(self):
        self.assertEqual(parse_ints("2-4,7"), [2, 3, 4, 7])
        self.assertEqual(parse_ints("4,2"), [2, 4])

    def test_sampling_flags(self):
        self.assertEqual(sampling_flags("muse", "greedy"), [])
        self.assertIn("1.5", sampling_flags("qwen38", "instruct"))
        self.assertIn("64", sampling_flags("gemma4_qat", "instruct"))
        with self.assertRaises(ValueError):
            sampling_flags("muse", "thinking")

    def test_bench_argv(self):
        code = bench_argv("nuclis", "m", "qwen38", None, "greedy", 4, None, 3, 128)
        self.assertIn("--raw", code)
        self.assertNotIn("--prefix-cache", code)
        self.assertNotIn("--draft-p-min", code)
        deep = bench_argv("nuclis", "m", "qwen38", Path("p.json"), "thinking", 7, 0.5, 3, 128)
        self.assertEqual(deep[deep.index("--draft-length") + 1], "7")
        self.assertEqual(deep[deep.index("--draft-p-min") + 1], "0.5")
        self.assertIn("--prefix-cache", deep)

    def test_cell_name(self):
        self.assertEqual(cell_name("512", "greedy", 4, None), "512-greedy-d4")
        self.assertEqual(cell_name("code", "instruct", 7, 0.5), "code-instruct-d7-p0.5")

    def test_row_of(self):
        def sample(on, rate, steps=0, generated=129, ms=0.0):
            s = {
                "speculative": on,
                "decode_tokens_per_second": rate,
                "generated_tokens": generated,
                "stop_reason": "token_budget",
            }
            if on:
                s.update(
                    speculative_steps=steps,
                    decode_milliseconds=ms,
                    accepted_per_step=1.5,
                    proposed_per_step=2.0,
                    propose_milliseconds=10.0 * steps,
                    verify_milliseconds=150.0 * steps,
                    recover_milliseconds=2.0 * steps,
                    commit_milliseconds=5.0 * steps,
                    checkpoint_milliseconds=3.0 * steps,
                )
            return s

        report = {
            "samples": [
                sample(False, 10.0),
                sample(True, 12.0, 50, ms=10000.0),
                sample(False, 10.0),
                sample(True, 13.0, 50, ms=10000.0),
            ]
        }
        row = row_of(report)
        assert row is not None
        self.assertAlmostEqual(row["E"], 128 / 50)
        self.assertAlmostEqual(row["C"], 200.0)
        self.assertAlmostEqual(row["C_over_50E"], 200.0 / (50 * 128 / 50))
        self.assertAlmostEqual(row["speedup"], 1.25)
        self.assertAlmostEqual(row["verify"], 150.0)
        self.assertIsNone(row_of({"samples": [sample(False, 10.0)]}))


if __name__ == "__main__":
    main()
