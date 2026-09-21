#!/usr/bin/env python3
"""The speculative-decoding record (docs/reference/bench.md § Speculative record).

Runs `nuclis bench --speculative on` as off/on pairs on one loaded model over
the reference corpus arrays (512 and 4,096 tokens) and the fixed code prompt,
greedy and with the instruct profile's sampling, at draft lengths 2, 4, 7,
writing one JSON per configuration; `--summarize` turns a directory of them
into the markdown table the record uses (per-batch costs are the run's
milliseconds divided by its verify batches).
"""
import argparse
import json
import pathlib
import statistics
import subprocess
import sys
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROSE = {512: "tests/fixtures/run-2026-09-06/prompt-512.json", 4096: "tests/fixtures/run-2026-09-06/prompt-4096.json"}
CODE_PROMPT = "Write a Zig function that reverses a string."
INSTRUCT = ["--temperature", "0.7", "--top-p", "0.8", "--top-k", "20", "--presence-penalty", "1.5"]

# (name, prompt arguments, repeat, draft lengths, instruct)
CONFIGS = [
    ("prose512-greedy", ["--prompt-tokens", PROSE[512]], 3, (2, 4, 7), False),
    ("prose512-instruct", ["--prompt-tokens", PROSE[512]], 3, (2, 4, 7), True),
    ("code-greedy", ["--prompt", CODE_PROMPT, "--raw"], 3, (2, 4, 7), False),
    ("code-instruct", ["--prompt", CODE_PROMPT, "--raw"], 3, (4,), True),
    ("prose4k-greedy", ["--prompt-tokens", PROSE[4096]], 2, (4,), False),
    ("prose4k-instruct", ["--prompt-tokens", PROSE[4096]], 2, (4,), True),
]


def run(args):
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    common = ["--backend", "metal", "--model", args.model, "--max-tokens", "128", "--ctx-size", "32768",
              "--kv", "f16", "--warmup", "1", "--speculative", "on", "--json"]
    for name, prompt, repeat, drafts, instruct in CONFIGS:
        if args.only and not any(name.startswith(o) for o in args.only):
            continue
        for d in drafts:
            label = f"{name}-d{d}"
            cmd = [args.nuclis, "bench", *common, *prompt, "--repeat", str(repeat), "--draft-length", str(d)]
            if instruct:
                cmd += INSTRUCT
            print(f"{datetime.now():%H:%M:%S} start {label}", flush=True)
            with open(out / f"{label}.json", "w") as f, open(out / f"{label}.err", "w") as e:
                code = subprocess.call(cmd, stdout=f, stderr=e, cwd=ROOT)
            print(f"{datetime.now():%H:%M:%S} done {label} exit {code}", flush=True)


def mean(values):
    return statistics.fmean(values) if values else None


def fmt(v, digits=1):
    return "—" if v is None else f"{v:.{digits}f}"


def summarize(directory):
    rows = []
    for path in sorted(pathlib.Path(directory).glob("*.json")):
        try:
            report = json.load(open(path))
        except json.JSONDecodeError:
            print(f"skipping {path.name}: not a complete report", file=sys.stderr)
            continue
        measured = [s for s in report["samples"] if not s["warmup"] and s["stop_reason"] != "cancelled"]
        off = [s for s in measured if not s["speculative"]]
        on = [s for s in measured if s["speculative"]]
        if not on or not off:
            continue
        steps = [s["speculative_steps"] for s in on]
        per_batch = lambda key: mean([s[key] / s["speculative_steps"] for s in on if s.get(key) is not None and s["speculative_steps"]])
        rows.append({
            "config": path.stem,
            "sampling": report["sampling"].split(":")[0],
            "draft": report.get("speculative_draft_length"),
            "prompt": on[0]["prompt_tokens"],
            "runs": len(on),
            "accepted": mean([s["accepted_per_step"] for s in on]),
            "proposed": mean([s["proposed_per_step"] for s in on if s.get("proposed_per_step") is not None]),
            "tokens_per_batch": mean([(s["generated_tokens"] - 1) / s["speculative_steps"] for s in on if s["speculative_steps"]]),
            "verify": per_batch("verify_milliseconds"),
            "accept": per_batch("accept_milliseconds"),
            "recover": per_batch("recover_milliseconds"),
            "prefill_off": mean([s["prefill_milliseconds"] for s in off]),
            "prefill_on": mean([s["prefill_milliseconds"] for s in on]),
            "decode_off": mean([s["decode_tokens_per_second"] for s in off if s["decode_tokens_per_second"]]),
            "decode_on": mean([s["decode_tokens_per_second"] for s in on if s["decode_tokens_per_second"]]),
            "speedup": report.get("decode_speedup"),
        })
    print("| configuration | prompt | draft | accepted/step | proposed/step | drafts/accepted | tokens/batch | verify ms | accept ms | recover ms | prefill off → on (s) | decode off → on tok/s | speedup |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in rows:
        per_token = (r["proposed"] / r["accepted"]) if (r["proposed"] is not None and r["accepted"]) else None
        print(f"| {r['config']} | {r['prompt']:,} | {r['draft']} | {fmt(r['accepted'], 2)} | {fmt(r['proposed'], 2)} | {fmt(per_token, 2)} | {fmt(r['tokens_per_batch'], 2)} | "
              f"{fmt(r['verify'])} | {fmt(r['accept'], 2)} | {fmt(r['recover'])} | "
              f"{fmt(r['prefill_off'] / 1000, 2)} → {fmt(r['prefill_on'] / 1000, 2)} | "
              f"{fmt(r['decode_off'], 2)} → {fmt(r['decode_on'], 2)} | {fmt(r['speedup'], 2)}× |")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="path to the Qwen3.8-27B GGUF")
    parser.add_argument("--nuclis", default="./zig-out/bin/nuclis")
    parser.add_argument("--out", default=".zig-cache/bench/spec")
    parser.add_argument("--only", nargs="*", help="configuration name prefixes to run (prose512, code, prose4k)")
    parser.add_argument("--summarize", metavar="DIR", help="print the record table from a directory of reports")
    args = parser.parse_args()
    if args.summarize:
        summarize(args.summarize)
        return
    if not args.model:
        parser.error("--model is required to run")
    run(args)


if __name__ == "__main__":
    main()
