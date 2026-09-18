#!/usr/bin/env python3
"""nuclis on the reference workload: prompts, runs, and a benchmark record.

Runs `nuclis bench --prompt-tokens` on the reference harness's exact token
arrays committed under tests/fixtures/ (scripts/reference-baseline.py built
them), one process per prompt length under /usr/bin/time -l for peak resident
memory, and writes a compact record under docs/benchmarks/. Before any run it
checks with `nuclis tokenize` that nuclis tokenizes the synthetic corpus into
the reference's body tokens, and renders each prompt as text beside the
record for reading (the reference concatenated token sequences, so the text's
canonical token count can differ by one at the cut; the record states it).
Standard library only; no network; nothing outside the repository is read but
the model file.
"""

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import statistics
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
MARKER = "NUCLIS_BENCH_CONTENT"
CLOSING = "\n```\n\nGive your review."
# The reference harness's thinking-off marker per family (scripts/reference-baseline.py).
THINKING_OFF = {"qwen38": "<think>\n\n</think>", "gemma4": "<|channel>thought\n<channel|>"}


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def capture(*command, timeout=30):
    """Best-effort environment evidence; None when the tool is missing or fails."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reference_corpus():
    """The synthetic Zig corpus, imported from the reference harness so the two never drift."""
    spec = importlib.util.spec_from_file_location("reference_baseline", HERE / "reference-baseline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.corpus()


class Tokenizer:
    """`nuclis tokenize --raw --json` on a file: IDs and byte offsets, no model run."""

    def __init__(self, nuclis, model, scratch):
        self.nuclis, self.model, self.scratch = nuclis, model, scratch

    def __call__(self, text):
        path = self.scratch / "tokenize-input.txt"
        path.write_text(text)
        result = subprocess.run(
            [self.nuclis, "tokenize", "--model", self.model, "--raw", "--json", "--prompt-file", str(path)],
            capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError("nuclis tokenize failed: " + result.stderr.strip())
        report = json.loads(result.stdout)
        if not report["round_trip"]:
            raise RuntimeError("token IDs do not decode back to the prompt; offsets are not cut points")
        return report


def load_reference_prompt(fixtures, run, size):
    """The reference's token array for `size`: the named run's, else whichever fixture run holds it."""
    for directory in [fixtures / run] + sorted(fixtures.iterdir()):
        candidate = directory / f"prompt-{size}.json"
        if candidate.is_file():
            return json.loads(candidate.read_text()), candidate.relative_to(ROOT).as_posix()
    raise RuntimeError(f"no reference fixture holds a {size}-token prompt under {fixtures}")


def build_prompt(size, construction, corpus, corpus_report, reference_ids, tokenize, prompt_dir):
    """Verify the tokenizer against the reference's array and render it as text.

    The check that matters: nuclis's tokens for the corpus equal the reference's
    body tokens, and the template prefix and suffix are the recorded ones. The
    text file is for reading; its own token count is recorded, not required to
    match (the 512 array ends its body on a lone space token that canonical
    tokenization merges with the suffix's newline).
    """
    prefix_len, suffix_len = len(construction["prefix_tokens"]), len(construction["suffix_tokens"])
    prefix_text, template_suffix = construction["template"].split(MARKER)
    suffix_text = CLOSING + template_suffix
    if reference_ids[:prefix_len] != construction["prefix_tokens"] or reference_ids[-suffix_len:] != construction["suffix_tokens"]:
        raise RuntimeError(f"the {size}-token fixture does not start and end with the recorded template tokens")
    body = reference_ids[prefix_len:-suffix_len]
    if corpus_report["ids"][:len(body)] != body:
        raise RuntimeError(f"nuclis tokenizes the corpus differently from the reference within the first {len(body)} tokens")
    text = prefix_text + corpus.encode()[:corpus_report["offsets"][len(body)]].decode() + suffix_text
    report = tokenize(text)
    path = prompt_dir / f"prompt-{size}.txt"
    path.write_text(text)
    return {
        "prompt_tokens": size,
        "corpus_tokens": len(body),
        "text_file": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text_bytes": len(text.encode()),
        "text_tokens": report["tokens"],
        "text_ids_equal_reference": report["ids"] == reference_ids,
    }


def memory_snapshot():
    return {"timestamp": timestamp(), "swap": capture("sysctl", "-n", "vm.swapusage")}


def run_bench(args, prompt, repetitions, warmup):
    command = [
        "/usr/bin/time", "-l", args.nuclis, "bench", "--model", args.model, "--backend", "metal",
        "--prompt-tokens", str(ROOT / prompt["reference_fixture"]), "--max-tokens", str(args.generate),
        "--ctx-size", str(args.ctx_size), "--kv", args.kv, "--repeat", str(repetitions),
        "--warmup", str(warmup), "--json",
    ]
    print(f"Starting {args.label}: prompt={prompt['prompt_tokens']}, output={args.generate}, "
          f"warmup={warmup}, repetitions={repetitions}", flush=True)
    before = memory_snapshot()
    result = subprocess.run(command, capture_output=True, text=True, timeout=4 * 3600)
    after = memory_snapshot()
    if result.returncode != 0:
        raise RuntimeError(f"nuclis bench failed for {prompt['prompt_tokens']} tokens: {result.stderr.strip()[-2000:]}")
    report = json.loads(result.stdout)
    rss = re.search(r"(\d+)\s+maximum resident set size", result.stderr)
    footprint = re.search(r"(\d+)\s+peak memory footprint", result.stderr)
    if report["prompt_source"] != "tokens" or report["kv_precision"] != args.kv or report["context"] != args.ctx_size:
        raise RuntimeError("the report does not describe the requested run")
    for sample in report["samples"]:
        if (sample["stop_reason"] != "token_budget" or sample["prompt_tokens"] != prompt["prompt_tokens"]
                or sample["generated_tokens"] != args.generate):
            raise RuntimeError(f"a run did not meet the fixed workload: {sample}")
    measured = [s for s in report["samples"] if not s["warmup"]]
    for sample in measured:
        print(f"  prefill={sample['prefill_tokens_per_second']:.2f} tok/s, decode={sample['decode_tokens_per_second']:.2f} tok/s, "
              f"first token={sample['first_token_milliseconds'] / 1000:.1f}s", flush=True)
    return {
        "prompt_tokens": prompt["prompt_tokens"],
        "command": command[2:],
        "peak_resident_bytes": int(rss.group(1)) if rss else None,
        "peak_footprint_bytes": int(footprint.group(1)) if footprint else None,
        "memory_before": before,
        "memory_after": after,
        "report": report,
    }


def summarize(runs):
    rows = []
    for run in runs:
        measured = [s for s in run["report"]["samples"] if not s["warmup"]]
        row = {"prompt_tokens": run["prompt_tokens"], "samples": len(measured)}
        for key, metric in (("prefill_tokens_per_second", "prefill_tokens_per_second"),
                            ("decode_tokens_per_second", "decode_tokens_per_second"),
                            ("first_token_milliseconds", "first_token_milliseconds")):
            values = [s[metric] for s in measured]
            row[key] = {"mean": statistics.mean(values), "sample_stdev": statistics.stdev(values) if len(values) > 1 else None}
        row["session_bytes"] = run["report"]["session_bytes"]
        row["peak_resident_bytes"] = run["peak_resident_bytes"]
        rows.append(row)
    return rows


def reference_revision(benchmarks, names):
    """The one reference revision the committed records were measured at."""
    revisions = {json.loads((benchmarks / name).read_text())["reference_revision"] for name in names if (benchmarks / name).is_file()}
    if len(revisions) > 1:
        raise RuntimeError("the reference records were measured at different revisions: " + ", ".join(sorted(revisions)))
    return revisions.pop() if revisions else None


def reference_rows(benchmarks, names):
    """Means of the accepted reference samples, read from the committed records."""
    rows = {}
    for name in names:
        path = benchmarks / name
        if not path.is_file():
            continue
        record = json.loads(path.read_text())
        for sample in record.get("samples", []):
            if sample.get("phase") not in ("warm", "measured", "capacity"):
                continue
            if sample.get("phase") == "capacity" and not sample.get("accepted", False):
                continue
            timings = sample["timings"]
            entry = rows.setdefault(sample["prompt_tokens"], {"prefill": [], "decode": [], "source": name})
            entry["prefill"].append(timings["prompt_per_second"])
            entry["decode"].append(timings["predicted_per_second"])
    return [{"prompt_tokens": size, "samples": len(e["prefill"]), "source": e["source"],
             "prefill_tokens_per_second": statistics.mean(e["prefill"]),
             "decode_tokens_per_second": statistics.mean(e["decode"])} for size, e in sorted(rows.items())]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nuclis", default=str(ROOT / "zig-out/bin/nuclis"), help="freshly built binary (default: ./zig-out/bin/nuclis)")
    parser.add_argument("--model", default=os.path.join(os.environ.get("NUCLIS_HOME", os.path.expanduser("~/.nuclis")), "models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf"))
    parser.add_argument("--fixtures", default=str(ROOT / "tests/fixtures"), type=pathlib.Path, help="directory holding the reference run fixtures")
    parser.add_argument("--prompt-dir", default=str(ROOT / ".zig-cache/prompts/reference"), type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, help="record path (default docs/benchmarks/nuclis-<date>.json); never overwritten")
    parser.add_argument("--prompt-lengths", default="512,4096,16384,32639")
    parser.add_argument("--generate", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--boundary-repetitions", type=int, default=1, help="repetitions for a prompt that fills the context with the output (the reference took one)")
    parser.add_argument("--ctx-size", type=int, default=32768)
    parser.add_argument("--kv", default="f16", choices=("f16", "f32"))
    parser.add_argument("--label", default="warm", choices=("warm", "cold"), help="cold: one process after `sudo purge`; pass --purge to run it here")
    parser.add_argument("--purge", action="store_true", help="run `sudo purge` before the first run (asks for a password)")
    parser.add_argument("--model-sha256", help="skip hashing the model (16 GB) when the digest is known")
    parser.add_argument("--prompts-only", action="store_true", help="build and verify the prompts, run nothing")
    parser.add_argument("--run", default="run-2026-09-06", help="the reference run directory under --fixtures holding prompt-construction.json and the token arrays")
    parser.add_argument("--reference-records", default="reference-2026-09-06.json,reference-boundary-2026-09-06.json",
                        help="comma-separated record names under docs/benchmarks whose accepted samples are the reference rows")
    args = parser.parse_args()
    try:
        lengths = [int(n) for n in args.prompt_lengths.split(",")]
    except ValueError:
        parser.error("--prompt-lengths must be comma-separated integers")
    if not lengths or min(lengths) <= 0 or len(lengths) != len(set(lengths)):
        parser.error("prompt lengths must be positive and unique")
    if args.generate <= 0 or args.repetitions <= 0 or args.warmup < 0 or args.boundary_repetitions <= 0:
        parser.error("output budget and repetitions must be positive; warmup non-negative")
    if max(lengths) + args.generate > args.ctx_size:
        parser.error("the longest prompt plus the output must fit the context; nothing is truncated")
    if not pathlib.Path(args.nuclis).is_file():
        parser.error(f"{args.nuclis} is not a file; build first (make metal)")
    if not pathlib.Path(args.model).is_file():
        parser.error(f"model not found: {args.model}")
    output = args.output or ROOT / "docs/benchmarks" / f"nuclis-{datetime.date.today().isoformat()}{'-cold' if args.label == 'cold' else ''}.json"
    if output.exists() and not args.prompts_only:
        parser.error(f"{output} exists; choose another --output")

    args.prompt_dir.mkdir(parents=True, exist_ok=True)
    construction_path = args.fixtures / args.run / "prompt-construction.json"
    construction = json.loads(construction_path.read_text())
    family = construction.get("family", "qwen38")
    if THINKING_OFF[family] not in construction["template"]:
        raise RuntimeError("expected the reference's text template with reasoning disabled")
    corpus = reference_corpus()
    if hashlib.sha256(corpus.encode()).hexdigest() != construction["corpus_sha256"]:
        raise RuntimeError("the synthetic corpus no longer matches the reference run's digest")
    with tempfile.TemporaryDirectory() as scratch:
        tokenize = Tokenizer(args.nuclis, args.model, pathlib.Path(scratch))
        corpus_report = tokenize(corpus)
        prompts = []
        for size in lengths:
            reference_ids, source = load_reference_prompt(args.fixtures, args.run, size)
            if len(reference_ids) != size:
                raise RuntimeError(f"{source} holds {len(reference_ids)} tokens, not {size}")
            prompt = build_prompt(size, construction, corpus, corpus_report, reference_ids, tokenize, args.prompt_dir)
            prompt["reference_fixture"] = source
            prompt["reference_fixture_sha256"] = sha256_file(ROOT / source)
            prompts.append(prompt)
            print(f"prompt {size}: run input {source}; corpus tokens match the reference through token {prompt['corpus_tokens']}; "
                  f"text {prompt['text_file']} ({prompt['text_bytes']} bytes) tokenizes to {prompt['text_tokens']}"
                  f"{'' if prompt['text_ids_equal_reference'] else ' (IDs differ at the cut)'}", flush=True)
    if args.prompts_only:
        return

    if args.purge:
        subprocess.run(["sudo", "purge"], check=True)
    started = timestamp()
    version = capture(args.nuclis, "--version")
    runs = []
    for prompt in prompts:
        boundary = prompt["prompt_tokens"] + args.generate == args.ctx_size - 1 or prompt["prompt_tokens"] + args.generate == args.ctx_size
        repetitions = args.boundary_repetitions if boundary else args.repetitions
        runs.append(run_bench(args, prompt, repetitions, args.warmup))
    model_sha256 = args.model_sha256 or sha256_file(args.model)
    record = {
        "schema_version": 1,
        "date": datetime.date.today().isoformat(),
        "label": args.label,
        "purged_before_first_run": args.purge,
        "started_at": started,
        "finished_at": timestamp(),
        "methodology": "See ../reference/bench.md#acceptance-runs and ../reference/reference-baseline.md. Each run feeds the reference "
                       "harness's exact token array (tests/fixtures) through `bench --prompt-tokens`; every sample is greedy with a "
                       "fixed output budget and starts from an empty session; warmups are excluded from the means; peak resident "
                       "memory is /usr/bin/time -l over the whole bench process (weights are memory-mapped, so it counts resident "
                       "model pages).",
        "nuclis_version": version,
        "git_revision": capture("git", "-C", str(ROOT), "rev-parse", "HEAD"),
        "git_dirty": bool(capture("git", "-C", str(ROOT), "status", "--porcelain")),
        "build_mode": runs[0]["report"]["build_mode"] if runs else None,
        "zig_version": capture("zig", "version"),
        "hardware": {
            "cpu": capture("sysctl", "-n", "machdep.cpu.brand_string"),
            "memory_bytes": int(capture("sysctl", "-n", "hw.memsize") or 0),
            "cpu_cores": int(capture("sysctl", "-n", "hw.ncpu") or 0),
            "gpu": capture("system_profiler", "SPDisplaysDataType", "-detailLevel", "mini", timeout=60),
        },
        "os": capture("sw_vers"),
        "power": capture("pmset", "-g", "batt"),
        "model_path": args.model,
        "model_sha256": model_sha256,
        "backend": "metal",
        "kv_precision": args.kv,
        "context": args.ctx_size,
        "output_tokens": args.generate,
        "warmup_runs": args.warmup,
        "repetitions": args.repetitions,
        "boundary_repetitions": args.boundary_repetitions,
        "prompts": prompts,
        "summary": summarize(runs),
        "family": family,
        "reference": {"revision": reference_revision(ROOT / "docs/benchmarks", args.reference_records.split(",")), "run": args.run, "rows": reference_rows(ROOT / "docs/benchmarks", args.reference_records.split(","))},
        "runs": runs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n")
    print("Completed: " + str(output), flush=True)
    for row in record["summary"]:
        print(f"  {row['prompt_tokens']:>6} tokens: prefill {row['prefill_tokens_per_second']['mean']:.2f} tok/s, "
              f"decode {row['decode_tokens_per_second']['mean']:.2f} tok/s, session {row['session_bytes'] / 2**20:.0f} MiB, "
              f"peak RSS {(row['peak_resident_bytes'] or 0) / 2**30:.2f} GiB", flush=True)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
