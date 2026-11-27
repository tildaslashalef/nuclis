#!/usr/bin/env python3
"""The workload driver: runs the benchmark workloads that workloads.json declares.

workloads.json is shapes × models (scripts/bench/manifest.py `expand`
names each `<row>/<shape>`). A `bench` workload resolves into one `nuclis
bench` command (model from gates.json and the index, prompt array or raw text, budget, context, cache precision,
sampling, the off/on pair with `--speculative on`, and an optional
no-drafter baseline run) and saves the JSON report under
`.zig-cache/bench/<workload>/<rev>-<n>.json`; an `acceptance` workload
delegates to scripts/bench/nuclis-baseline.py, which writes the dated record under
docs/benchmarks/. Tables come from scripts/bench/bench-report.py.
See docs/benchmarks/README.md § The record.
"""

import argparse
import fnmatch
import json
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/bench"))
import manifest  # noqa: E402

MANIFEST = ROOT / "workloads.json"
SAVE_ROOT = ".zig-cache/bench"
KINDS = ("bench", "acceptance")
SAMPLING_FLAGS = {
    "temperature": "--temperature",
    "top_k": "--top-k",
    "top_p": "--top-p",
    "min_p": "--min-p",
    "presence_penalty": "--presence-penalty",
    "repetition_penalty": "--repetition-penalty",
    "seed": "--seed",
}
ACCEPTANCE_LENGTHS = [512, 4096, 16384, 32639]
ACCEPTANCE_OUTPUT = 128
BAR_METRICS = ("decode_tokens_per_second", "prefill_tokens_per_second", "decode_speedup")


# ---- pure functions (covered by --self-test) --------------------------------


INDEX_SCHEMA = 2
ROLES = ("default", "extra")
SUPPORTS = ("verified", "preview", "archived")


def index_problems(index):
    """The index's own problems, as tools/gates/index.zig `validate` names them:
    a schema other than INDEX_SCHEMA, or an artifact whose role or support is
    missing or unknown. Unknown is refused, never read as verified."""
    problems = []
    version = index.get("schema_version")
    if type(version) is not int or version != INDEX_SCHEMA:
        problems.append(f"src/catalog.json: schema_version must be {INDEX_SCHEMA}; run make index")
    artifacts = index.get("artifacts")
    if not isinstance(artifacts, list):
        return problems + ['src/catalog.json: no "artifacts" list']
    for i, a in enumerate(artifacts):
        name = a.get("name") or f"artifacts[{i}]"
        if a.get("role") not in ROLES:
            problems.append(f"src/catalog.json: {name} has no known role (default or extra)")
        if a.get("support") not in SUPPORTS:
            problems.append(f"src/catalog.json: {name} has no known support (verified, preview, or archived)")
    return problems


def validate_rows(wdoc, gdoc, index):
    """The shapes × models document's own problems: an index it cannot read,
    unknown shapes and models, and a model whose artifact is archived (a
    workload measures what is gated)."""
    problems = index_problems(index)
    shapes = wdoc.get("shapes", {})
    for name, shape in shapes.items():
        if "base" in shape and shape["base"] not in shapes:
            problems.append(f"shapes.{name}: base names an unknown shape {shape['base']!r}")
    for row, m in wdoc.get("models", {}).items():
        key = m.get("model")
        if key not in gdoc["models"]:
            problems.append(f'models.{row}: model names an unknown gates.json key "{key}"')
            continue
        artifact = manifest.index_artifact(index, gdoc["models"][key].get("artifact", ""))
        if artifact is not None and artifact.get("support") == "archived":
            problems.append(f"models.{row}: {artifact['name']} is archived; remove its workloads")
        for name in m.get("shapes", []):
            if name not in shapes:
                problems.append(f"models.{row}: unknown shape {name!r}")
        for key in ("bars", "note", "evidence"):
            for name in m.get(key, {}):
                if name not in m.get("shapes", []):
                    problems.append(f"models.{row}.{key}: {name!r} is not one of its shapes")
        if "prompt_tokens" in str([shapes.get(n, {}) for n in m.get("shapes", [])]) and "run" not in m:
            problems.append(f"models.{row}: a prompt_tokens shape needs the row's run")
    return problems


def validate(workloads, models):
    problems = []
    for name, w in workloads.items():
        if "/" not in name or name != name.lower():
            problems.append(f"{name}: names are lower-case <entry>/<workload>")
        kind = w.get("kind")
        if kind not in KINDS:
            problems.append(f"{name}: kind must be one of {KINDS}")
        if w.get("model") not in models:
            problems.append(f'{name}: model names an unknown gates.json key "{w.get("model")}"')
        if not isinstance(w.get("evidence"), str):
            problems.append(f"{name}: missing evidence")
        if kind == "bench":
            if ("prompt" in w) == ("prompt_tokens" in w):
                problems.append(f"{name}: exactly one of prompt, prompt_tokens")
            for key in ("max_tokens", "ctx", "warmup", "repeat"):
                if not isinstance(w.get(key), int) or w[key] < 0:
                    problems.append(f"{name}: {key} must be a non-negative integer")
            if w.get("kv") not in manifest.KV_PRECISIONS:
                problems.append(f"{name}: kv must be one of {', '.join(manifest.KV_PRECISIONS)}")
            if w.get("pair") and not isinstance(w.get("draft_length"), int):
                problems.append(f"{name}: a pair needs draft_length")
            if w.get("baseline") and not w.get("pair"):
                problems.append(f"{name}: baseline is only meaningful for a pair")
            for key in w.get("sampling", {}):
                if key not in SAMPLING_FLAGS:
                    problems.append(f"{name}: unknown sampling option {key}")
            for metric, bar in w.get("bars", {}).items():
                if (
                    metric not in BAR_METRICS
                    or not isinstance(bar, dict)
                    or not (set(bar) <= {"min", "max"})
                    or not bar
                ):
                    problems.append(f"{name}: bars.{metric} must be {{min|max}} on one of {BAR_METRICS}")
        elif kind == "acceptance":
            for key in ("run", "record_suffix"):
                if not isinstance(w.get(key), str):
                    problems.append(f"{name}: acceptance needs {key}")
            if not isinstance(w.get("reference_records", ""), str):
                problems.append(f"{name}: reference_records must be a string")
            lengths = w.get("prompt_lengths", ACCEPTANCE_LENGTHS)
            if not isinstance(lengths, list) or not lengths or not all(isinstance(n, int) and n > 0 for n in lengths):
                problems.append(f"{name}: prompt_lengths must be positive integers")
            elif max(lengths) + ACCEPTANCE_OUTPUT > w.get("ctx", 32768):
                problems.append(f"{name}: the longest prompt and the output must fit ctx")
    return problems


def bench_argv(w, model_path, nuclis="./zig-out/bin/nuclis", speculative=None):
    """The `nuclis bench` command for a bench workload; `speculative` overrides the pair's switch."""
    argv = [nuclis, "bench", "--backend", "metal", "--model", model_path]
    if "prompt_tokens" in w:
        argv += ["--prompt-tokens", w["prompt_tokens"]]
    else:
        argv += ["--prompt", w["prompt"]]
        if w.get("raw"):
            argv.append("--raw")
    argv += [
        "--max-tokens",
        str(w["max_tokens"]),
        "--ctx-size",
        str(w["ctx"]),
        "--kv",
        w["kv"],
        "--warmup",
        str(w["warmup"]),
        "--repeat",
        str(w["repeat"]),
    ]
    for key, value in w.get("sampling", {}).items():
        argv += [SAMPLING_FLAGS[key], str(value)]
    on = w.get("pair", False) if speculative is None else speculative
    argv += ["--speculative", "on" if on else "off"]
    if on:
        argv += ["--draft-length", str(w["draft_length"])]
        if w.get("draft"):
            argv += ["--draft", w["draft"]]
    argv.append("--json")
    return argv


def acceptance_argv(w, model_path, nuclis="./zig-out/bin/nuclis", date="DATE"):
    """nuclis-baseline.py on the row's run: arrays it lacks are made, the
    ones it has held to what nuclis makes; the shape's lengths and context."""
    argv = [
        sys.executable,
        "scripts/bench/nuclis-baseline.py",
        "--model",
        model_path,
        "--nuclis",
        nuclis,
        "--run",
        w["run"],
        "--reference-records",
        w.get("reference_records", ""),
        "--make-prompts",
        "--family",
        w["family"],
        "--prompt-lengths",
        ",".join(str(n) for n in w.get("prompt_lengths", ACCEPTANCE_LENGTHS)),
        "--ctx-size",
        str(w.get("ctx", 32768)),
    ]
    for key, flag in (("warmup", "--warmup"), ("repeat", "--repetitions"), ("repeat", "--boundary-repetitions")):
        if key in w:
            argv += [flag, str(w[key])]
    if w["record_suffix"]:
        argv += ["--output", f"docs/benchmarks/nuclis-{date}-{w['record_suffix']}.json"]
    return argv


def ceiling_of(help_text):
    """The build's context ceiling, from `nuclis bench --help`'s --ctx-size row."""
    import re

    match = re.search(r"context window in tokens, 1\.\.(\d+)", help_text)
    return int(match.group(1)) if match else None


def context_ceiling(nuclis="./zig-out/bin/nuclis"):
    try:
        result = subprocess.run([nuclis, "bench", "--help"], capture_output=True, text=True, timeout=30, cwd=ROOT)
    except OSError:
        return None
    return ceiling_of(result.stdout + result.stderr)


def select(names, patterns):
    out = []
    for pattern in patterns:
        hits = [n for n in names if fnmatch.fnmatchcase(n, pattern)]
        if not hits:
            raise KeyError(pattern)
        out.extend(n for n in hits if n not in out)
    return out


def next_sequence(directory, rev):
    """One past the highest sequence already saved for this revision."""
    n = 0
    for p in Path(directory).glob(f"{rev}-*.json"):
        stem = p.name[len(rev) + 1 : -5]
        digits = stem.split("-")[0]
        if digits.isdigit():
            n = max(n, int(digits))
    return n + 1


def record_path(directory, rev, n, suffix=""):
    """`<rev>-<n><suffix>.json`: a pair's baseline shares its record's `n`,
    which is how bench-report.py finds it."""
    return Path(directory) / f"{rev}-{n}{suffix}.json"


# ---- execution ----------------------------------------------------------------


def load():
    """The expanded workloads and the resolved gates.json; exits on a problem."""
    wdoc = json.loads(MANIFEST.read_text())
    index = manifest.load_index()
    gdoc = manifest.resolve(json.loads(manifest.MANIFEST.read_text()), index)
    problems = validate_rows(wdoc, gdoc, index)
    workloads = {}
    if not problems:
        try:
            workloads = manifest.expand(wdoc, gdoc, index)
        except KeyError as e:
            problems.append(str(e))
        problems += validate(workloads, gdoc["models"])
    if problems:
        for p in problems:
            print(f"workloads.json: {p}", file=sys.stderr)
        sys.exit(2)
    return workloads, gdoc


def git_rev():
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()


def model_for(w, gdoc, env=None):
    """The pinned path, except that a pair names the catalogue entry so the
    companion (`mtp`) resolves; an explicit `<KEY>_MODEL` override wins either way."""
    env = os.environ if env is None else env
    if w.get("pair") and not env.get(w["model"].upper() + "_MODEL"):
        return gdoc["models"][w["model"]].get("entry") or manifest.model_path(gdoc, w["model"], env)
    return manifest.model_path(gdoc, w["model"], env)


def run_workload(name, w, gdoc, args):
    model = model_for(w, gdoc)
    save_dir = ROOT / args.save_dir / name
    rev = git_rev()
    if w["kind"] == "acceptance":
        ceiling = context_ceiling()
        if ceiling is not None and w.get("ctx", 32768) > ceiling:
            print(
                f"{name}: needs a {w['ctx']}-token context; this build's ceiling is {ceiling} "
                f"(build with make metal ARGS=-Dmax-context={w['ctx']})",
                file=sys.stderr,
            )
            return False
        argv = acceptance_argv(w, model, date=time.strftime("%Y-%m-%d"))
        print(f"{name}: {' '.join(argv)}", flush=True)
        if args.dry_run:
            return True
        return subprocess.run(argv, cwd=ROOT).returncode == 0
    runs = [("", bench_argv(w, model))]
    if w.get("baseline"):
        runs.append(("-baseline", bench_argv(w, model, speculative=False)))
    ok = True
    sequence = None
    for suffix, argv in runs:
        print(f"{name}{suffix}: {' '.join(argv)}", flush=True)
        if args.dry_run:
            continue
        save_dir.mkdir(parents=True, exist_ok=True)
        if sequence is None:
            sequence = next_sequence(save_dir, rev)
        out = record_path(save_dir, rev, sequence, suffix)
        started = time.monotonic()
        proc = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
        seconds = time.monotonic() - started
        if proc.returncode != 0:
            print(
                f"FAIL {name}{suffix} ({seconds:.0f} s)\n" + "\n".join(proc.stderr.strip().splitlines()[-10:]),
                flush=True,
            )
            ok = False
            continue
        out.write_text(proc.stdout)
        report = json.loads(proc.stdout)
        line = f"saved {out.relative_to(ROOT)} ({seconds:.0f} s): decode {fmt(report.get('mean_decode_tokens_per_second'))} tok/s, prefill {fmt(report.get('mean_prefill_tokens_per_second'))} tok/s"
        if report.get("speculative_draft_length"):
            line += f", speculative {fmt(report.get('mean_speculative_decode_tokens_per_second'))} tok/s, {fmt(report.get('decode_speedup'))}×"
        print(line, flush=True)
    return ok


def fmt(v):
    return "—" if v is None else f"{v:.2f}"


def model_status(path):
    return "present" if Path(path).is_file() else "absent"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--list", action="store_true", help="every workload with its kind, model and status, shape, and evidence"
    )
    ap.add_argument(
        "--validate", action="store_true", help="validate the manifest and run the self-test; no model needed"
    )
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument(
        "--run",
        action="append",
        metavar="NAME",
        help="workloads by name or glob (repeatable): qwen38/prose512-draft, 'gemma4-e4b/*'",
    )
    ap.add_argument("--save-dir", default=SAVE_ROOT, help=f"where bench reports go (default {SAVE_ROOT}/<workload>/)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)
    if args.self_test:
        sys.exit(0 if self_test() else 1)
    workloads, gdoc = load()
    if args.validate:
        print(f"workloads.json: {len(workloads)} workloads, valid")
        sys.exit(0 if self_test() else 1)
    if args.list:
        for name, w in workloads.items():
            model = manifest.model_path(gdoc, w["model"])
            shape = (
                (
                    f"{w['max_tokens']} out, ctx {w['ctx']}, {w['kv']}, {w['warmup']}+{w['repeat']}"
                    + (f", pair d{w['draft_length']}" if w.get("pair") else "")
                    + (", baseline" if w.get("baseline") else "")
                    + (", sampled" if w.get("sampling") else "")
                )
                if w["kind"] == "bench"
                else f"acceptance: {w['run']}"
            )
            print(f"{name:<34} {w['model']:<15} {model_status(model):<8} {shape:<44} {w['evidence']}")
        return
    if not args.run:
        ap.error("one of --list, --validate, --run is required")
    try:
        names = select(list(workloads), args.run)
    except KeyError as e:
        sys.exit(f"no workload matches {e.args[0]!r}; --list shows them")
    if not args.no_build and not args.dry_run:
        manifest.build(gdoc, False)
    failed = [n for n in names if not run_workload(n, workloads[n], gdoc, args)]
    if failed:
        sys.exit("failed: " + ", ".join(failed))


# ---- self-test ----------------------------------------------------------------


def sample():
    return {
        "workloads": {
            "a/prose": {
                "kind": "bench",
                "model": "a",
                "prompt_tokens": "tests/p.json",
                "max_tokens": 8,
                "ctx": 64,
                "kv": "f16",
                "warmup": 1,
                "repeat": 2,
                "evidence": "docs/x.md",
            },
            "a/code-draft": {
                "kind": "bench",
                "model": "a",
                "prompt": "hi",
                "raw": True,
                "max_tokens": 8,
                "ctx": 64,
                "kv": "f16",
                "warmup": 0,
                "repeat": 1,
                "pair": True,
                "draft_length": 4,
                "baseline": True,
                "sampling": {"temperature": 0.7, "top_k": 20},
                "bars": {"decode_speedup": {"min": 1.0}},
                "evidence": "docs/x.md",
            },
            "a/acceptance": {
                "kind": "acceptance",
                "model": "a",
                "run": "run-x",
                "reference_records": "r.json",
                "record_suffix": "a",
                "family": "a",
                "evidence": "docs/x.md",
            },
            "a/acceptance64": {
                "kind": "acceptance",
                "model": "a",
                "run": "run-x",
                "record_suffix": "a-64k",
                "family": "a",
                "prompt_lengths": [32639, 65407],
                "ctx": 65536,
                "warmup": 1,
                "repeat": 1,
                "evidence": "docs/x.md",
            },
        }
    }


SAMPLE_INDEX = {
    "schema_version": 2,
    "artifacts": [
        {
            "name": "a",
            "role": "default",
            "support": "verified",
            "defaults": {"draft": "mtp"},
            "files": [
                {"role": "main", "local": "o/a/a.gguf"},
                {"role": "draft", "variant": "mtp", "local": "o/a/mtp.gguf", "verdict": {"draft_length": 7}},
                {"role": "draft", "variant": "dflash2", "local": "o/a/d.gguf", "verdict": {"draft_length": 5}},
            ],
        },
        {
            "name": "old",
            "role": "extra",
            "support": "archived",
            "defaults": {},
            "files": [{"role": "main", "local": "o/old.gguf"}],
        },
    ],
}


def sample_rows():
    return {
        "shapes": {
            "prose": {
                "kind": "bench",
                "prompt_tokens": "p.json",
                "max_tokens": 8,
                "ctx": 64,
                "kv": "f16",
                "warmup": 1,
                "repeat": 2,
                "evidence": "docs/x.md",
            },
            "prose-draft": {"base": "prose", "pair": True},
            "acceptance": {"kind": "acceptance", "evidence": "docs/x.md"},
            "acceptance64": {"kind": "acceptance", "record_tag": "64k", "ctx": 65536, "evidence": "docs/x.md"},
        },
        "models": {
            "a": {
                "model": "a",
                "run": "run-x",
                "reference_records": "r.json",
                "record_suffix": "a",
                "shapes": ["prose", "prose-draft", "acceptance", "acceptance64"],
                "bars": {"prose-draft": {"decode_speedup": {"min": 1.0}}},
            },
            "a-dflash2": {"model": "a", "run": "run-x", "draft": "dflash2", "shapes": ["prose-draft"]},
        },
    }


class SelfTest(unittest.TestCase):
    def test_sample_valid(self):
        self.assertEqual(validate(sample()["workloads"], {"a": {}}), [])

    def test_shapes_by_models(self):
        gdoc = manifest.resolve({"models": {"a": {"artifact": "a"}, "old": {"artifact": "old"}}}, SAMPLE_INDEX)
        self.assertEqual(validate_rows(sample_rows(), gdoc, SAMPLE_INDEX), [])
        ws = manifest.expand(sample_rows(), gdoc, SAMPLE_INDEX)
        self.assertEqual(
            sorted(ws), ["a-dflash2/prose-draft", "a/acceptance", "a/acceptance64", "a/prose", "a/prose-draft"]
        )
        self.assertEqual(ws["a/acceptance64"]["record_suffix"], "a-64k")
        self.assertEqual(ws["a/prose"]["prompt_tokens"], "tests/fixtures/run-x/p.json")
        self.assertEqual(ws["a/prose-draft"]["draft_length"], 7)
        self.assertEqual(ws["a/prose-draft"]["bars"], {"decode_speedup": {"min": 1.0}})
        self.assertEqual(ws["a-dflash2/prose-draft"]["draft_length"], 5)
        self.assertEqual(ws["a/acceptance"]["run"], "run-x")
        self.assertEqual(ws["a/acceptance"]["family"], "a")
        self.assertEqual(validate(ws, gdoc["models"]), [])
        argv = bench_argv(ws["a-dflash2/prose-draft"], "a")
        self.assertEqual(argv[argv.index("--draft") + 1], "dflash2")
        self.assertNotIn("--draft", bench_argv(ws["a/prose-draft"], "a"))
        rows = sample_rows()
        rows["models"]["old"] = {"model": "old", "shapes": ["nope"]}
        rows["models"]["gone"] = {"model": "zz", "shapes": []}
        problems = validate_rows(rows, gdoc, SAMPLE_INDEX)
        for needle in ("old is archived", "unknown shape 'nope'", 'unknown gates.json key "zz"'):
            self.assertTrue(any(needle in p for p in problems), (needle, problems))

    def test_index_fails_closed(self):
        self.assertEqual(index_problems(SAMPLE_INDEX), [])
        one = {"name": "x", "role": "extra", "support": "verified"}
        cases = [
            ({"artifacts": []}, "schema_version must be 2"),
            ({"schema_version": 1, "artifacts": []}, "schema_version must be 2"),
            ({"schema_version": "2", "artifacts": []}, "schema_version must be 2"),
            ({"schema_version": True, "artifacts": []}, "schema_version must be 2"),
            ({"schema_version": 2}, 'no "artifacts" list'),
            ({"schema_version": 2, "artifacts": [{**one, "role": None}]}, "x has no known role"),
            ({"schema_version": 2, "artifacts": [{**one, "role": "current"}]}, "x has no known role"),
            (
                {"schema_version": 2, "artifacts": [{k: v for k, v in one.items() if k != "support"}]},
                "x has no known support",
            ),
            ({"schema_version": 2, "artifacts": [{**one, "support": "legacy"}]}, "x has no known support"),
        ]
        for index, needle in cases:
            problems = index_problems(index)
            self.assertEqual(len(problems), 1, (index, problems))
            self.assertIn(needle, problems[0])
        gdoc = manifest.resolve({"models": {"a": {"artifact": "a"}}}, SAMPLE_INDEX)
        stale = {**SAMPLE_INDEX, "schema_version": 1}
        self.assertTrue(any("schema_version" in p for p in validate_rows(sample_rows(), gdoc, stale)))

    def test_validation_names_problems(self):
        doc = sample()
        doc["workloads"]["a/prose"]["prompt"] = "both"
        doc["workloads"]["a/code-draft"]["bars"] = {"nope": {"min": 1}}
        doc["workloads"]["B/x"] = {"kind": "bench", "model": "zz"}
        problems = validate(doc["workloads"], {"a": {}})
        for needle in ("exactly one of prompt", "bars.nope", "lower-case", 'unknown gates.json key "zz"'):
            self.assertTrue(any(needle in p for p in problems), (needle, problems))

    def test_bench_argv(self):
        w = sample()["workloads"]["a/code-draft"]
        argv = bench_argv(w, "/m.gguf")
        self.assertEqual(argv[:6], ["./zig-out/bin/nuclis", "bench", "--backend", "metal", "--model", "/m.gguf"])
        self.assertIn("--raw", argv)
        self.assertEqual(argv[argv.index("--speculative") + 1], "on")
        self.assertEqual(argv[argv.index("--draft-length") + 1], "4")
        self.assertEqual(argv[argv.index("--temperature") + 1], "0.7")
        off = bench_argv(w, "/m.gguf", speculative=False)
        self.assertEqual(off[off.index("--speculative") + 1], "off")
        self.assertNotIn("--draft-length", off)
        plain = bench_argv(sample()["workloads"]["a/prose"], "/m.gguf")
        self.assertEqual(plain[plain.index("--prompt-tokens") + 1], "tests/p.json")
        self.assertEqual(plain[-3:], ["--speculative", "off", "--json"])

    def test_model_for(self):
        gdoc = {"models": {"a": {"path": "~/m/a.gguf", "entry": "a-entry"}}}
        ws = sample()["workloads"]
        self.assertEqual(model_for(ws["a/code-draft"], gdoc, env={}), "a-entry")
        self.assertEqual(model_for(ws["a/prose"], gdoc, env={}), os.path.expanduser("~/m/a.gguf"))
        self.assertEqual(model_for(ws["a/code-draft"], gdoc, env={"A_MODEL": "/x.gguf"}), "/x.gguf")

    def test_acceptance_argv(self):
        argv = acceptance_argv(sample()["workloads"]["a/acceptance"], "/m.gguf", date="2026-01-02")
        self.assertEqual(argv[-1], "docs/benchmarks/nuclis-2026-01-02-a.json")
        self.assertEqual(argv[argv.index("--run") + 1], "run-x")
        self.assertIn("--make-prompts", argv)
        self.assertEqual(argv[argv.index("--prompt-lengths") + 1], "512,4096,16384,32639")
        self.assertNotIn("--repetitions", argv)
        long = acceptance_argv(sample()["workloads"]["a/acceptance64"], "/m.gguf", date="2026-01-02")
        self.assertEqual(long[long.index("--reference-records") + 1], "")
        self.assertEqual(long[long.index("--prompt-lengths") + 1], "32639,65407")
        self.assertEqual(long[long.index("--ctx-size") + 1], "65536")
        self.assertEqual(long[long.index("--repetitions") + 1], "1")
        self.assertEqual(long[-1], "docs/benchmarks/nuclis-2026-01-02-a-64k.json")
        doc = sample()
        doc["workloads"]["a/acceptance64"]["prompt_lengths"] = [65500]
        self.assertTrue(any("must fit ctx" in p for p in validate(doc["workloads"], {"a": {}})))

    def test_ceiling_of(self):
        self.assertEqual(ceiling_of("  --ctx-size <n>   context window in tokens, 1..131072; default 32768"), 131072)
        self.assertIsNone(ceiling_of("no such row"))

    def test_select_and_record_paths(self):
        names = ["a/prose", "a/code-draft", "b/prose"]
        self.assertEqual(select(names, ["a/*"]), ["a/prose", "a/code-draft"])
        self.assertEqual(select(names, ["*/prose", "a/prose"]), ["a/prose", "b/prose"])
        with self.assertRaises(KeyError):
            select(names, ["c/*"])
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(next_sequence(d, "abc1234"), 1)
            (Path(d) / "abc1234-1.json").write_text("{}")
            (Path(d) / "abc1234-1-baseline.json").write_text("{}")
            (Path(d) / "abc1234-3.json").write_text("{}")
            n = next_sequence(d, "abc1234")
            self.assertEqual(record_path(d, "abc1234", n).name, "abc1234-4.json")
            # The baseline pairs with its record, whatever was saved between.
            self.assertEqual(record_path(d, "abc1234", n, "-baseline").name, "abc1234-4-baseline.json")
            self.assertEqual(next_sequence(d, "zzz9999"), 1)


def self_test():
    return (
        unittest.TextTestRunner(verbosity=0)
        .run(unittest.defaultTestLoader.loadTestsFromTestCase(SelfTest))
        .wasSuccessful()
    )


if __name__ == "__main__":
    main()
