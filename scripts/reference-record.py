#!/usr/bin/env python3
"""Summarize a reference-baseline.py run directory into a benchmark record.

Reads run-config.json, samples.json, summary.json, and server-props.json from
the untracked run directory and writes docs/benchmarks/reference-<date>-<family>.json
(the shape scripts/nuclis-baseline.py reads its reference rows from: `samples`
with `phase` warmup/warm and per-request `timings`). Hashes the model file the
server reported so the record names the artifact. Standard library only.
"""
import argparse
import datetime
import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=pathlib.Path, help="the harness output directory")
    parser.add_argument("--family", required=True, help="the profile family the run measured (qwen38, gemma4)")
    parser.add_argument("--output", type=pathlib.Path, help="record path (default docs/benchmarks/reference-<date>-<family>.json); never overwritten")
    parser.add_argument("--model-sha256", help="skip hashing when the digest is known")
    args = parser.parse_args()
    config = json.loads((args.run / "run-config.json").read_text())
    samples = json.loads((args.run / "samples.json").read_text())
    summary = json.loads((args.run / "summary.json").read_text())
    props = json.loads((args.run / "server-props.json").read_text())
    construction = json.loads((args.run / "prompt-construction.json").read_text())
    if construction.get("family", "qwen38") != args.family:
        raise SystemExit(f"the run's prompt construction is for {construction.get('family', 'qwen38')}, not {args.family}")
    # The local date, as nuclis-baseline.py names its records (started_at stays UTC).
    date = datetime.date.today().isoformat()
    output = args.output or ROOT / "docs/benchmarks" / f"reference-{date}-{args.family}.json"
    if output.exists():
        raise SystemExit(f"{output} exists; choose another --output")
    rss = [int(s[k]["server_rss_kib"]) for s in samples for k in ("memory_before", "memory_after") if s[k].get("server_rss_kib")]
    record = {
        "date": date,
        "started_at": config["started_at"],
        "reference_revision": config["reference_revision"],
        "family": args.family,
        "model_path": props["model_path"],
        "model_sha256": args.model_sha256 or sha256_file(props["model_path"]),
        "build_info": props["build_info"],
        "hardware": config["hardware"],
        "os": config["os"],
        "power": config["power"],
        "context": config["context"],
        "output_tokens": config["generate"],
        "prompt_lengths": config["prompt_lengths"],
        "repetitions": config["repetitions"],
        "methodology": "See ../reference/reference-baseline.md. Greedy, seed 1, no prefix reuse, EOS ignored for the timed "
                       "requests, one untimed warmup per length excluded from the means; rates are the server's own timings.",
        "prompt_file_sha256": {p.name: sha256_file(p) for p in sorted(args.run.glob("prompt-*.json"))},
        "smoke_response": json.loads((args.run / "smoke.json").read_text())["response"]["content"],
        "summary": summary["results"],
        "samples": [{k: s[k] for k in ("phase", "prompt_tokens", "repetition", "output_tokens", "wall_seconds", "timings")} for s in samples],
        "memory_snapshots": {"server_rss_kib_min": min(rss) if rss else None, "server_rss_kib_max": max(rss) if rss else None,
                             "limitation": "RSS sampled between requests; not peak memory. Weights are memory-mapped."},
        "raw_local_directory": str(args.run),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Wrote {output}")
    for row in summary["results"]:
        print(f"  {row['prompt_tokens']:>6} tokens ({row['phase']}, {row['samples']} samples): prefill {row['prompt_per_second']['mean']:.2f} tok/s, "
              f"decode {row['predicted_per_second']['mean']:.2f} tok/s")


if __name__ == "__main__":
    main()
