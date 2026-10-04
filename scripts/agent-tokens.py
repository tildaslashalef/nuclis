#!/usr/bin/env python3
"""Where an agent-eval variant's prefill went: per tool, the calls, the tokens
their results added to the context, and each tool's share of all result
tokens, counted with the engine's own tokenizer (`nuclis tokenize --raw`),
not a character estimate.

    python3 scripts/agent-tokens.py agnt19-after
    python3 scripts/agent-tokens.py agnt19-after agnt20-grep   # side by side

Reads the session files `scripts/agent-eval.py` leaves under
.zig-cache/agent-eval/<variant>/. A result's tool is the call its `call` id
answers. `prompt` is the sum of every step's newly prefilled tokens, so
`results / prompt` is how much of the turn's prefill the tools caused.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / ".zig-cache" / "agent-eval"
DEFAULT_BINARY = ROOT / "zig-out" / "bin" / "nuclis"
TOOLS = ["read_file", "edit_file", "write_file", "bash", "grep", "glob"]


class Counter:
    """Token counts through the binary, memoized: a variant repeats results."""

    def __init__(self, binary: Path, model: str | None):
        self.binary = binary
        self.model = model
        self.cache: dict[str, int] = {}

    def __call__(self, text: str) -> int:
        if not text:
            return 0
        if text in self.cache:
            return self.cache[text]
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(text)
            path = f.name
        argv = [str(self.binary), "tokenize", "--prompt-file", path, "--raw", "--json"]
        if self.model:
            argv += ["--model", self.model]
        out = subprocess.run(argv, capture_output=True, text=True, check=True)
        Path(path).unlink()
        n = json.loads(out.stdout)["tokens"]
        self.cache[text] = n
        return n


def measure(variant: str, count: Counter) -> dict:
    """Per tool: calls and result tokens; plus the variant's totals."""
    folder = OUT_DIR / variant
    sessions = sorted(folder.glob("*.session.jsonl"))
    if not sessions:
        sys.exit(f"agent-tokens: no session files under {folder}")
    calls: dict[str, int] = defaultdict(int)
    tokens: dict[str, int] = defaultdict(int)
    prompt = steps = 0
    for session in sessions:
        names: dict[str, str] = {}
        for line in session.read_text().splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("type") == "assistant":
                steps += 1
                prompt += e.get("stats", {}).get("prompt_tokens", 0)
                for call in e.get("tool_calls", []):
                    names[str(call.get("id"))] = call.get("name", "?")
            elif e.get("type") == "tool_result":
                name = names.get(str(e.get("call")), "?")
                calls[name] += 1
                tokens[name] += count(e.get("text", ""))
    return {"runs": len(sessions), "steps": steps, "prompt": prompt, "calls": calls, "tokens": tokens}


def render(variant: str, m: dict):
    total = sum(m["tokens"].values())
    print(f"{variant}: {m['runs']} runs, {m['steps']} steps, {m['prompt']} prompt tokens, {total} in tool results")
    print(f"  {'tool':<11} {'calls':>5} {'tokens':>7} {'share':>6} {'per call':>8}")
    names = [t for t in TOOLS if m["calls"].get(t)] + sorted(set(m["calls"]) - set(TOOLS))
    for name in sorted(names, key=lambda n: -m["tokens"][n]):
        n, t = m["calls"][name], m["tokens"][name]
        print(f"  {name:<11} {n:>5} {t:>7} {t / total if total else 0:>6.0%} {t / n:>8.0f}")
    if m["prompt"]:
        print(f"  results are {total / m['prompt']:.0%} of the prompt tokens")


def compare(variants: list[str], measured: list[dict]):
    print()
    print(f"  {'tool':<11}" + "".join(f" {v[:16]:>16}" for v in variants))
    names = sorted({n for m in measured for n in m["calls"]}, key=lambda n: -measured[0]["tokens"].get(n, 0))
    for name in names + ["total"]:
        row = f"  {name:<11}"
        for m in measured:
            t = sum(m["tokens"].values()) if name == "total" else m["tokens"].get(name, 0)
            row += f" {t:>16}"
        print(row)
    base = sum(measured[0]["tokens"].values())
    for v, m in zip(variants[1:], measured[1:]):
        t = sum(m["tokens"].values())
        if base:
            print(f"  {v}: tool-result tokens {t / base - 1:+.0%} against {variants[0]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("variants", nargs="+", help="agent-eval variant names (folders under .zig-cache/agent-eval/)")
    ap.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    ap.add_argument("--model", help="whose tokenizer; default engine.model")
    args = ap.parse_args()
    count = Counter(args.binary, args.model)
    measured = [measure(v, count) for v in args.variants]
    for i, (v, m) in enumerate(zip(args.variants, measured)):
        if i:
            print()
        render(v, m)
    if len(measured) > 1:
        compare(args.variants, measured)


if __name__ == "__main__":
    main()
