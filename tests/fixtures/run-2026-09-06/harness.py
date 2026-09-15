#!/usr/bin/env python3
"""Opt-in development harness for a separately started, pinned llama.cpp server.

Only localhost is accepted. Prompts are synthetic Zig code, never workspace
files. Each request disables prefix reuse and saves its exact token input plus
the raw response. This is tooling for comparison, not part of the Zig runtime.
"""

import argparse
import datetime
import hashlib
import json
import pathlib
import statistics
import subprocess
import time
import urllib.parse
import urllib.request


REVISION = "7620399f58aebfd2196b74021f9581bcf7218cb9"


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def capture(*command):
    """Best-effort environment evidence; never collect process arguments or env."""
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    return result.stdout.strip() if result.returncode == 0 else None


def corpus():
    header = (
        "Review this synthetic Zig module for overflow and API design. "
        "Give a concise review with concrete improvements.\n\n```zig\n"
    )
    functions = []
    for index in range(1200):
        functions.append(
            f"pub fn sum_{index:04d}(values: []const u32) u64 {{\n"
            "    var total: u64 = 0;\n"
            "    for (values) |value| {\n"
            f"        total += @as(u64, value) * {index % 17 + 1};\n"
            "    }\n"
            "    return total;\n"
            "}\n\n"
        )
    return header + "".join(functions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18087")
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--prompt-lengths", default="512,4096,16384")
    parser.add_argument("--generate", type=int, default=128)
    parser.add_argument("--capacity-check", action="store_true")
    parser.add_argument("--server-pid", type=int, help="optionally sample this server's RSS between requests")
    args = parser.parse_args()
    address = urllib.parse.urlparse(args.url)
    if (address.scheme != "http" or address.hostname != "127.0.0.1"
            or address.username or address.password or address.path not in ("", "/")
            or address.query or address.fragment):
        parser.error("--url must be an HTTP endpoint on 127.0.0.1")
    try:
        lengths = [int(n) for n in args.prompt_lengths.split(",")]
    except ValueError:
        parser.error("--prompt-lengths must be comma-separated integers")
    if not lengths or min(lengths) <= 0 or args.generate <= 0 or args.repetitions <= 0:
        parser.error("token counts and repetitions must be positive")
    if len(lengths) != len(set(lengths)) or (args.server_pid is not None and args.server_pid <= 0):
        parser.error("prompt lengths must be unique and server PID must be positive")

    # Ignore HTTP proxy configuration for this explicitly local connection.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(endpoint, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(args.url.rstrip("/") + endpoint, data=data,
                                     headers={"Content-Type": "application/json"})
        with opener.open(req, timeout=1800) as response:
            return json.load(response)

    props = request("/props")
    if REVISION[:7] not in props["build_info"]:
        raise RuntimeError("server is not the pinned reference revision")
    context = props["default_generation_settings"]["n_ctx"]
    if props["total_slots"] != 1 or max(lengths) + args.generate > context:
        raise RuntimeError("expected one slot with room for prompt and output")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        path = args.output_dir / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(path)

    def memory():
        return {
            "timestamp": timestamp(),
            "swap": capture("sysctl", "-n", "vm.swapusage"),
            "vm_stat": capture("vm_stat"),
            "server_rss_kib": capture("ps", "-p", str(args.server_pid), "-o", "rss=")
            if args.server_pid else None,
        }

    save("run-config.json", {
        "started_at": timestamp(), "reference_revision": REVISION,
        "context": context, "prompt_lengths": lengths, "generate": args.generate,
        "repetitions": args.repetitions, "capacity_check": args.capacity_check,
        "hardware": capture("sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize", "hw.ncpu"),
        "os": capture("sw_vers"), "power": capture("pmset", "-g", "batt"),
        "power_settings": capture("pmset", "-g", "custom"), "memory": memory(),
    })

    def tokenize(text, special=False):
        return request("/tokenize", {"content": text, "add_special": False,
                                     "parse_special": special})["tokens"]

    save("server-props.json", props)
    marker = "NUCLIS_BENCH_CONTENT"
    template = request("/apply-template", {"messages": [{"role": "user", "content": marker}]})["prompt"]
    if template.count(marker) != 1 or "<think>\n\n</think>" not in template:
        raise RuntimeError("expected the pinned text template with reasoning disabled")
    prefix, suffix = template.split(marker)
    prefix_tokens = tokenize(prefix, True)
    suffix_tokens = tokenize("\n```\n\nGive your review." + suffix, True)
    source = corpus()
    body_tokens = tokenize(source)
    save("prompt-construction.json", {
        "template": template,
        "corpus_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "prefix_tokens": prefix_tokens, "suffix_tokens": suffix_tokens,
        "method": "prefix + truncated synthetic-code token sequence + suffix",
    })

    def prompt(size):
        count = size - len(prefix_tokens) - len(suffix_tokens)
        if count <= 0 or count > len(body_tokens):
            raise RuntimeError("synthetic corpus cannot fill requested token count")
        return prefix_tokens + body_tokens[:count] + suffix_tokens

    def complete(tokens, n_predict, ignore_eos):
        start = time.perf_counter()
        response = request("/completion", {
            "prompt": tokens, "n_predict": n_predict, "temperature": 0,
            "samplers": ["temperature"], "seed": 1,
            "cache_prompt": False, "ignore_eos": ignore_eos,
            "return_tokens": True, "stream": False, "stop": [],
        })
        return response, time.perf_counter() - start

    # A smoke response checks actual text decoding, but is not a quality score.
    smoke_prompt = request("/apply-template", {"messages": [{
        "role": "user", "content": "Write a Python function add(a, b) that returns their sum. Return only code.",
    }]})["prompt"]
    smoke, smoke_seconds = complete(tokenize(smoke_prompt, True), 96, False)
    save("smoke.json", {"response": smoke, "wall_seconds": smoke_seconds})
    if "return" not in smoke["content"] or "a + b" not in smoke["content"]:
        raise RuntimeError("smoke output needs review; refusing to label this a working reference")
    print("Model smoke response: " + smoke["content"].strip(), flush=True)

    results = []
    cases = [(n, "warm") for n in lengths]
    if args.capacity_check:
        cases.append((context - args.generate, "capacity"))
    for size, phase in cases:
        tokens = prompt(size)
        save(f"prompt-{size}.json", tokens)
        repetitions = args.repetitions if phase == "warm" else 1
        # First request at each size establishes an untimed warmup of that shape.
        for repetition in range(repetitions + 1 if phase == "warm" else 1):
            label = "warmup" if phase == "warm" and repetition == 0 else phase
            print(f"Starting {label}: prompt={size}, output={args.generate}, repetition={repetition}", flush=True)
            memory_before = memory()
            response, wall = complete(tokens, args.generate, True)
            save(f"response-{size}-{label}-{repetition}.json", {"response": response, "wall_seconds": wall})
            timings = response["timings"]
            if response.get("truncated") or timings["prompt_n"] != size or timings["cache_n"] != 0:
                raise RuntimeError("prompt truncation or reuse invalidated the measurement")
            if response["tokens_predicted"] != args.generate or response["stop_type"] != "limit":
                raise RuntimeError("generation did not meet the fixed output budget")
            row = {"phase": label, "prompt_tokens": size, "repetition": repetition,
                   "output_tokens": args.generate, "wall_seconds": wall, "timings": timings,
                   "memory_before": memory_before, "memory_after": memory()}
            results.append(row)
            save("samples.json", results)
            print(f"  prefill={timings['prompt_per_second']:.2f} tok/s, "
                  f"decode={timings['predicted_per_second']:.2f} tok/s, wall={wall:.2f}s", flush=True)
    summary = []
    for size, phase in cases:
        rows = [r for r in results if r["prompt_tokens"] == size and r["phase"] == phase]
        item = {"prompt_tokens": size, "phase": phase, "samples": len(rows)}
        for metric in ("prompt_per_second", "predicted_per_second"):
            values = [r["timings"][metric] for r in rows]
            item[metric] = {"mean": statistics.mean(values),
                            "sample_stdev": statistics.stdev(values) if len(values) > 1 else None}
        summary.append(item)
    save("summary.json", {"reference_revision": REVISION, "context": context,
                          "generate": args.generate, "results": summary})
    print("Completed: " + str(args.output_dir / "summary.json"), flush=True)


if __name__ == "__main__":
    main()
