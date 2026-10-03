#!/usr/bin/env python3
"""The gate runner: executes the model-specific checks that gates.json declares.

A gate is one command (argv with placeholders) plus a comparator: `exit`
(the command's status decides) or `trace` (its `{trace}` directory goes
through compare-generation.py against the gate's bounds). Gates carry a
tier (`verify`: Metal, minutes; `verify-release`: whole-file acceptance
before a release; `verify-long`: Metal long-context perplexity, tens of
minutes; `verify-cpu`: the CPU reference, hours) and the source globs that
make them relevant, so `--changed REV` selects by `git diff`: the Metal
tier's matches by default, another tier's with `--tier` (the long tier runs
when a change touches attention, the caches, or a windowed schedule, the
CPU tier when a change alters what the CPU reference computes, and all
three before a release). `--auto` is the one command for a change: the
model-free checks its paths select (`checks`), its fast-tier gates
cheapest first until one fails, then the tiers it requires but that did
not run (`requires`). Bounds live in the manifest and nowhere else.
See docs/development.md § Gates.
"""

import argparse
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "gates.json"
TIERS = ("verify", "verify-release", "verify-long", "verify-cpu")
COMPARATORS = ("exit", "trace")
PLACEHOLDERS = ("{nuclis}", "{nuclis-cpu}", "{zig-metal}", "{zig-cpu}", "{model}", "{mtp}", "{mmproj}", "{trace}")
TRACE_ROOT = ".zig-cache/gates/trace"
CPU_PREFIX = ".zig-cache/gates/cpu"
TIMES = ROOT / ".zig-cache/gates/times.json"
TODO = ROOT / "TODO.md"
# A gate never measured sorts after the cheap ones.
UNKNOWN_SECONDS = 60.0


# ---- pure functions (covered by --self-test) --------------------------------


def validate(doc):
    """Return a list of manifest problems; empty means valid."""
    problems = []
    build = doc.get("build", {})
    for key in ("metal_optimize", "cpu_optimize", "cache"):
        if not isinstance(build.get(key), str):
            problems.append(f"build.{key}: missing or not a string")
    models = doc.get("models", {})
    for name, entry in models.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            problems.append(f'models.{name}: needs a "path" string')
    seen = set()
    for i, gate in enumerate(doc.get("gates", [])):
        name = gate.get("name", f"#{i}")
        if not isinstance(gate.get("name"), str):
            problems.append(f"gate #{i}: missing name")
        if name in seen:
            problems.append(f"{name}: duplicate name")
        seen.add(name)
        if gate.get("tier") not in TIERS:
            problems.append(f"{name}: tier must be one of {TIERS}")
        for key in ("model", "mtp"):
            if key in gate and gate[key] not in models:
                problems.append(f'{name}: {key} names an unknown model "{gate[key]}"')
        if "model" not in gate:
            problems.append(f"{name}: missing model")
        paths = gate.get("paths")
        if not isinstance(paths, list) or not paths or not all(isinstance(p, str) for p in paths):
            problems.append(f"{name}: paths must be a non-empty list of globs")
        command = gate.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(a, str) for a in command):
            problems.append(f"{name}: command must be a non-empty argv list")
        else:
            for arg in command:
                for token in re.findall(r"\{[a-z-]+\}", arg):
                    if token not in PLACEHOLDERS:
                        problems.append(f"{name}: unknown placeholder {token}")
            if "{mtp}" in " ".join(command) and "mtp" not in gate:
                problems.append(f"{name}: command uses {{mtp}} but the gate names no mtp")
            if "{mmproj}" in " ".join(command) and "mmproj" not in models.get(gate.get("model"), {}):
                problems.append(f'{name}: command uses {{mmproj}} but model "{gate.get("model")}" names no mmproj')
        comparator = gate.get("comparator", {})
        kind = comparator.get("kind")
        if kind not in COMPARATORS:
            problems.append(f"{name}: comparator.kind must be one of {COMPARATORS}")
        if kind == "trace":
            if not isinstance(comparator.get("reference"), str):
                problems.append(f"{name}: trace comparator needs a reference directory")
            if not isinstance(comparator.get("positions"), int):
                problems.append(f"{name}: trace comparator needs integer positions")
            bounds = gate.get("bounds")
            if not isinstance(bounds, dict) or set(bounds) != {"max_absolute", "max_relative_rms"}:
                problems.append(f"{name}: trace gates need bounds {{max_absolute, max_relative_rms}}")
            if command and "{trace}" not in " ".join(command):
                problems.append(f"{name}: trace gates must write to {{trace}}")
        elif "bounds" in gate:
            problems.append(f"{name}: an exit gate carries no bounds")
        if not isinstance(gate.get("evidence"), str):
            problems.append(f"{name}: missing evidence")
    for i, check in enumerate(doc.get("checks", [])):
        name = check.get("name", f"check #{i}")
        command = check.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(a, str) for a in command):
            problems.append(f"{name}: command must be a non-empty argv list")
        if not is_globs(check.get("paths")):
            problems.append(f"{name}: paths must be a non-empty list of globs")
    for i, rule in enumerate(doc.get("requires", [])):
        if rule.get("tier") not in TIERS or rule.get("tier") == "verify":
            problems.append(f"requires #{i}: tier must be one of {TIERS[1:]}")
        if not is_globs(rule.get("paths")):
            problems.append(f"requires #{i}: paths must be a non-empty list of globs")
        if not isinstance(rule.get("reason"), str):
            problems.append(f"requires #{i}: missing reason")
    return problems


def is_globs(paths):
    return isinstance(paths, list) and bool(paths) and all(isinstance(p, str) for p in paths)


def glob_to_regex(glob):
    """`**` spans directories, `*` stays inside one path component."""
    out = ""
    i = 0
    while i < len(glob):
        if glob.startswith("**/", i):
            out += "(?:.*/)?"
            i += 3
        elif glob.startswith("**", i):
            out += ".*"
            i += 2
        elif glob[i] == "*":
            out += "[^/]*"
            i += 1
        else:
            out += re.escape(glob[i])
            i += 1
    return re.compile("^" + out + "$")


def matches(glob, path):
    return glob_to_regex(glob).match(path) is not None


def select(gates, changed):
    """The gates (or checks) whose paths match any changed file, in manifest order."""
    return [g for g in gates if any(matches(p, f) for p in g["paths"] for f in changed)]


def requirements(rules, changed):
    """(tier, the first file that matched, reason) for each `requires` rule the change matches."""
    out = []
    for rule in rules:
        hit = next((f for f in changed if any(matches(p, f) for p in rule["paths"])), None)
        if hit is not None:
            out.append((rule["tier"], hit, rule["reason"]))
    return out


def by_cost(gates, times):
    """Cheapest first, one model at a time: the files together outgrow memory,
    so interleaving models reloads each from disk. Models go in the order of
    their cheapest gate, a model's gates by their last measured seconds; ties
    keep manifest order."""

    def cost(g):
        return times.get(g["name"], UNKNOWN_SECONDS)

    first = {}
    for g in gates:
        first[g.get("model")] = min(first.get(g.get("model"), cost(g)), cost(g))
    return sorted(gates, key=lambda g: (first[g.get("model")], g.get("model") or "", cost(g)))


def unit_base(todo_text):
    """The `Base:` revision of the first unit in TODO.md (the one in progress), or None."""
    m = re.search(r"^Base: `([0-9a-f]{7,40})`", todo_text, re.M)
    return m.group(1) if m else None


def check_argv(check, changed):
    """A check's argv with `{zig-files}` expanded to the changed Zig sources; None when it has none to read."""
    zig = [f for f in changed if f.endswith((".zig", ".zon"))]
    if "{zig-files}" in check["command"] and not zig:
        return None
    return [a for arg in check["command"] for a in (zig if arg == "{zig-files}" else [arg])]


def expand(argv, gate, doc, env=None):
    """Substitute placeholders; the build placeholders expand to several argv items."""
    env = os.environ if env is None else env
    table = {
        "{nuclis}": ["./zig-out/bin/nuclis"],
        "{nuclis-cpu}": [f"./{CPU_PREFIX}/bin/nuclis"],
        "{zig-metal}": ["zig", "build"] + zig_flags(doc, "metal"),
        "{zig-cpu}": ["zig", "build"] + zig_flags(doc, "cpu"),
    }
    scalars = {"{trace}": f"{TRACE_ROOT}/{gate['name']}"}
    for key in ("model", "mtp"):
        if key in gate:
            scalars["{" + key + "}"] = model_path(doc, gate[key], env)
    entry = doc["models"].get(gate.get("model"), {})
    if "mmproj" in entry:
        scalars["{mmproj}"] = os.path.expanduser(entry["mmproj"])
    out = []
    for arg in argv:
        if arg in table:
            out.extend(table[arg])
            continue
        for token, value in scalars.items():
            arg = arg.replace(token, value)
        out.append(arg)
    return out


def zig_flags(doc, flavour):
    """The `zig build` options behind `{zig-metal}` / `{zig-cpu}`."""
    build = doc["build"]
    optimize = (
        ["-Dmetal=true", f"-Doptimize={build['metal_optimize']}"]
        if flavour == "metal"
        else [f"-Doptimize={build['cpu_optimize']}"]
    )
    return optimize


def model_path(doc, key, env=None):
    """The pinned path, or the `<KEY>_MODEL` environment override; `~` expanded."""
    env = os.environ if env is None else env
    override = env.get(key.upper() + "_MODEL")
    path = override if override else doc["models"][key]["path"]
    return os.path.expanduser(path)


def missing_files(gate, doc, env=None):
    """The model files a gate reads (`model`, `mtp`, and the entry's `mmproj`
    when its command names it) that are not on this machine."""
    paths = [model_path(doc, gate[key], env) for key in ("model", "mtp") if key in gate]
    entry = doc["models"].get(gate.get("model"), {})
    if "mmproj" in entry and any("{mmproj}" in arg for arg in gate["command"]):
        paths.append(os.path.expanduser(entry["mmproj"]))
    return [path for path in paths if not os.path.exists(path)]


def builds_needed(gates):
    """Which binaries the selected gates run: the two nuclis builds, and the check tools behind the zig placeholders (built up front so the build is timed apart from the checks)."""
    joined = [" ".join(g["command"]) for g in gates]
    return {
        "metal": any("{nuclis}" in c for c in joined),
        "cpu": any("{nuclis-cpu}" in c for c in joined),
        "checks-metal": any("{zig-metal}" in c for c in joined),
        "checks-cpu": any("{zig-cpu}" in c for c in joined),
    }


# ---- execution ----------------------------------------------------------------


def load():
    doc = json.loads(MANIFEST.read_text())
    problems = validate(doc)
    if problems:
        for p in problems:
            print(f"gates.json: {p}", file=sys.stderr)
        sys.exit(2)
    # `zig build` takes the global cache from the environment only; every
    # build and gate below inherits it.
    os.environ.setdefault("ZIG_GLOBAL_CACHE_DIR", str(ROOT / doc["build"]["cache"]))
    return doc


def changed_files(rev):
    diff = subprocess.run(
        ["git", "diff", "--name-only", rev], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    return sorted(set(diff.split()) | set(untracked.split()))


def load_times():
    try:
        return json.loads(TIMES.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def save_times(results):
    """Merge each run gate's seconds into the history `--auto` orders by."""
    times = load_times()
    times.update({r["name"]: round(r["seconds"], 1) for r in results if r.get("passed") is not None})
    TIMES.parent.mkdir(parents=True, exist_ok=True)
    TIMES.write_text(json.dumps(times, indent=1, sort_keys=True) + "\n")


def build(doc, which, dry_run):
    """Build one binary (`metal`, `cpu`) or one flavour of the check tools (`checks-metal`, `checks-cpu`); returns seconds."""
    b = doc["build"]
    if which == "metal":
        cmd = ["zig", "build", "-Dmetal=true", f"-Doptimize={b['metal_optimize']}"]
    elif which == "cpu":
        cmd = ["zig", "build", "-Dmetal=true", f"-Doptimize={b['cpu_optimize']}", "--prefix", CPU_PREFIX]
    else:
        # The placeholders' own flags, so the gates' `zig build` finds the compile cached.
        cmd = (
            ["zig", "build", "check-tools"]
            + zig_flags(doc, which.removeprefix("checks-"))
            + ["--prefix", f".zig-cache/gates/{which}"]
        )
    print("build:", " ".join(cmd), file=sys.stderr, flush=True)
    if dry_run:
        return 0.0
    started = time.monotonic()
    subprocess.run(cmd, cwd=ROOT, check=True)
    return time.monotonic() - started


def run_gate(gate, doc, dry_run):
    """Runs one gate; `passed` is None with `skipped` set when a model file it
    reads is not on this machine (the run neither passes nor fails it)."""
    argv = expand(gate["command"], gate, doc)
    trace = ROOT / TRACE_ROOT / gate["name"]
    result = {"name": gate["name"], "tier": gate["tier"], "command": argv, "passed": None, "seconds": 0.0}
    missing = missing_files(gate, doc)
    if missing:
        result["skipped"] = "no model: " + ", ".join(missing)
        if dry_run:
            print(f"{gate['name']}: skipped ({result['skipped']})")
        return result
    if dry_run:
        print(f"{gate['name']}: {' '.join(argv)}")
        if gate["comparator"]["kind"] == "trace":
            print(
                f"  then compare-generation.py {trace.relative_to(ROOT)} {gate['comparator']['reference']} "
                f"--max-absolute {gate['bounds']['max_absolute']} --max-relative-rms {gate['bounds']['max_relative_rms']}"
            )
        return result
    shutil.rmtree(trace, ignore_errors=True)
    trace.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    # Merged output read line by line, each stamped with its time since
    # launch: the phases of a check (load, then each protocol) without
    # instrumenting the tools.
    proc = subprocess.Popen(argv, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.stdout is not None  # stdout=PIPE
    timeline = []
    for line in proc.stdout:
        timeline.append((round(time.monotonic() - started, 2), line.rstrip("\n")))
    proc.wait()
    result["seconds"] = time.monotonic() - started
    result["exit_status"] = proc.returncode
    result["timeline"] = timeline
    tail = "\n".join(line for _, line in timeline[-20:]).strip()
    if proc.returncode != 0:
        result["passed"] = False
        result["detail"] = tail
        return result
    comparator = gate["comparator"]
    if comparator["kind"] == "exit":
        result["passed"] = True
        result["detail"] = tail.splitlines()[-1] if tail else ""
        return result
    cmd = [
        sys.executable,
        "scripts/compare-generation.py",
        str(trace),
        comparator["reference"],
        "--positions",
        str(comparator["positions"]),
        "--max-absolute",
        str(gate["bounds"]["max_absolute"]),
        "--max-relative-rms",
        str(gate["bounds"]["max_relative_rms"]),
    ]
    for key in ("embedding", "layers", "vocab"):
        if key in comparator:
            cmd += ["--" + key, str(comparator[key])]
    if comparator.get("draft"):
        cmd.append("--draft")
    cmp = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    result["seconds"] = time.monotonic() - started
    try:
        report = json.loads(cmp.stdout)
    except json.JSONDecodeError:
        result["passed"] = False
        result["detail"] = (cmp.stderr or cmp.stdout).strip()[-2000:]
        return result
    rows = report["comparisons"]
    result["passed"] = bool(report["passed"])
    result["measured"] = {
        "max_absolute": max(r["max_absolute"] for r in rows),
        "max_relative_rms": max(r["relative_rms"] for r in rows),
        "files": len(rows),
    }
    if "native_greedy" in report:
        result["measured"]["greedy"] = report["native_greedy"]
        result["measured"]["reference_greedy"] = report["reference_greedy"]
    return result


def print_result(r, gate, timeline=False):
    if r.get("skipped"):
        print(f"SKIP {r['name']:<34} {r['skipped']}", flush=True)
        return
    status = "PASS" if r["passed"] else "FAIL"
    line = f"{status} {r['name']:<34} {r['seconds']:7.1f} s"
    if "measured" in r:
        m, b = r["measured"], gate["bounds"]
        line += (
            f"  max abs {m['max_absolute']:.3e} ≤ {b['max_absolute']:g}"
            f"  rel rms {m['max_relative_rms']:.3e} ≤ {b['max_relative_rms']:g}  ({m['files']} files)"
        )
        if "greedy" in m:
            line += f"  greedy {m['greedy']}" + (
                "" if m["greedy"] == m["reference_greedy"] else f" ≠ {m['reference_greedy']}"
            )
    elif r.get("detail"):
        line += f"  {r['detail'] if r['passed'] else ''}"
    print(line, flush=True)
    if timeline:
        for t, text in r.get("timeline", []):
            print(f"    {t:7.1f} s  {text[:160]}")
    if not r["passed"] and r.get("detail"):
        for text in r["detail"].splitlines():
            print("    " + text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print every gate with its tier, family, model, and evidence")
    ap.add_argument(
        "--validate", action="store_true", help="validate the manifest and run the self-test; no model needed"
    )
    ap.add_argument("--self-test", action="store_true", help="run the unit tests of the pure functions")
    ap.add_argument(
        "--tier", choices=TIERS, help="run every gate of a tier; with --changed, the selected gates of that tier"
    )
    ap.add_argument(
        "--gate",
        action="append",
        metavar="NAME",
        help="run gates by name or glob (repeatable): gemma4-qat-trace-f16, 'muse-*'",
    )
    ap.add_argument(
        "--auto",
        nargs="?",
        const="",
        metavar="REV",
        help="what a change needs: the model-free checks and verify gates its paths select, cheapest first until one fails, then the tiers it requires; REV defaults to TODO.md's `Base:` line",
    )
    ap.add_argument("--keep-going", action="store_true", help="with --auto, run everything selected after a failure")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="a gate whose model is not on this machine fails instead of being skipped (before a release)",
    )
    ap.add_argument(
        "--changed",
        nargs="?",
        const="HEAD",
        metavar="REV",
        help="run the Metal-tier gates whose paths match `git diff --name-only REV` plus untracked files (default HEAD); the other tiers' matches are listed, and run with --tier verify-long or verify-cpu",
    )
    ap.add_argument("--dry-run", action="store_true", help="print the commands instead of running them")
    ap.add_argument("--no-build", action="store_true", help="do not rebuild the binaries first")
    ap.add_argument("--json", action="store_true", help="print the results as JSON")
    ap.add_argument(
        "--timeline", action="store_true", help="print each output line of a gate with its time since launch"
    )
    args = ap.parse_args()
    os.chdir(ROOT)

    if args.self_test:
        sys.exit(0 if self_test() else 1)
    doc = load()
    if args.validate:
        print(f"gates.json: {len(doc['gates'])} gates, {len(doc['models'])} models, valid")
        sys.exit(0 if self_test() else 1)
    if args.list:
        for g in doc["gates"]:
            print(f"{g['name']:<34} {g['tier']:<10} {g['family']:<8} {g['model']:<15} {g['evidence']}")
        return

    if args.auto is not None:
        sys.exit(auto(doc, args))
    if args.gate:
        selected = []
        for pattern in args.gate:
            hits = [g for g in doc["gates"] if fnmatch.fnmatchcase(g["name"], pattern)]
            if not hits:
                sys.exit(f"no gate matches {pattern!r}; --list shows them")
            selected.extend(g for g in hits if g not in selected)
    elif args.changed is not None:
        files = changed_files(args.changed)
        matched = select(doc["gates"], files)
        tier = args.tier or "verify"
        selected = [g for g in matched if g["tier"] == tier]
        others = [g for g in matched if g["tier"] != tier]
        print(
            f"{len(files)} changed file(s) since {args.changed}; {len(selected)} {tier} gate(s) selected: "
            + (", ".join(g["name"] for g in selected) or "none"),
            flush=True,
        )
        if tier == "verify":
            long = [g["name"] for g in others if g["tier"] == "verify-long"]
            cpu = [g["name"] for g in others if g["tier"] == "verify-cpu"]
            release = [g["name"] for g in others if g["tier"] == "verify-release"]
            if long:
                print(
                    "Long-tier gates matched, not run (they run when the change touches attention, the caches, "
                    "or a windowed schedule, and before a release; `--tier verify-long` runs them): " + ", ".join(long),
                    flush=True,
                )
            if cpu:
                print(
                    "CPU-tier gates matched, not run (they run when the change alters what the CPU reference "
                    "computes, and before a release; `--tier verify-cpu` runs them): " + ", ".join(cpu),
                    flush=True,
                )
            if release:
                print(
                    "Release-tier gates matched, not run (before a release; `--tier verify-release` runs them): "
                    + ", ".join(release),
                    flush=True,
                )
        if not selected:
            return
    elif args.tier:
        selected = [g for g in doc["gates"] if g["tier"] == args.tier]
    else:
        ap.error("one of --list, --validate, --auto, --tier, --gate, --changed is required")

    builds = {}
    if not args.no_build:
        need = builds_needed(selected)
        for which in ("metal", "cpu", "checks-metal", "checks-cpu"):
            if need[which]:
                builds[which] = round(build(doc, which, args.dry_run), 1)
    results = []
    for gate in selected:
        r = run_gate(gate, doc, args.dry_run)
        results.append(r)
        if not args.dry_run and not args.json:
            print_result(r, gate, args.timeline)
    if args.dry_run:
        return
    save_times(results)
    if args.json:
        print(json.dumps({"revision": git_rev(), "builds": builds, "results": results}, indent=2))
    skipped = [r["name"] for r in results if r.get("skipped")]
    failed = [r["name"] for r in results if not r["passed"] and (args.strict or not r.get("skipped"))]
    passed = len(results) - len(skipped) - len([n for n in failed if n not in skipped])
    total = sum(r["seconds"] for r in results)
    built = sum(builds.values())
    print(
        f"{passed}/{len(results)} passed in {total:.0f} s"
        + (f" after {built:.0f} s of builds" if builds else "")
        + (f"; {len(skipped)} skipped, their models not on this machine: {', '.join(skipped)}" if skipped else "")
        + (f"; failed: {', '.join(failed)}" if failed else ""),
        file=sys.stderr,
    )
    sys.exit(1 if failed else 0)


def auto(doc, args):
    """`--auto`: checks, then gates cheapest first, then the required tiers; returns the exit status."""
    base = args.auto or unit_base(TODO.read_text() if TODO.exists() else "")
    if base is None:
        print(
            "warning: no REV and no `Base:` line in TODO.md; diffing against HEAD (committed work is not seen)",
            file=sys.stderr,
        )
        base = "HEAD"
    files = changed_files(base)
    # A deleted file still selects gates; only a file that exists can be formatted.
    present = [f for f in files if (ROOT / f).exists()]
    print(f"{len(files)} changed file(s) since {base}", flush=True)
    failed = []
    skipped = []

    def stop():
        return failed and not args.keep_going

    for check in select(doc.get("checks", []), files):
        argv = check_argv(check, present)
        if argv is None:
            continue
        print(f"check {check['name']}: {' '.join(argv)}", flush=True)
        if args.dry_run:
            continue
        started = time.monotonic()
        proc = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
        ok = proc.returncode == 0
        print(f"{'PASS' if ok else 'FAIL'} check {check['name']:<28} {time.monotonic() - started:7.1f} s", flush=True)
        if not ok:
            failed.append(check["name"])
            for line in (proc.stderr or proc.stdout).strip().splitlines()[-20:]:
                print("    " + line)
            if stop():
                break

    gates = by_cost([g for g in select(doc["gates"], files) if g["tier"] == "verify"], load_times())
    if gates and not stop():
        print(f"{len(gates)} verify gate(s), cheapest first: " + ", ".join(g["name"] for g in gates), flush=True)
        if not args.no_build:
            need = builds_needed(gates)
            for which in ("metal", "cpu", "checks-metal", "checks-cpu"):
                if need[which]:
                    build(doc, which, args.dry_run)
        results = []
        for gate in gates:
            r = run_gate(gate, doc, args.dry_run)
            if args.dry_run:
                continue
            results.append(r)
            print_result(r, gate, args.timeline)
            if r.get("skipped"):
                skipped.append(r["name"])
                if not args.strict:
                    continue
            if not r["passed"]:
                failed.append(r["name"])
                if stop():
                    break
        save_times(results)

    required = requirements(doc.get("requires", []), files)
    for tier, hit, reason in required:
        print(
            f"Required, not run: {tier} ({hit}: {reason}); make verify-changed BASE={base} ARGS='--tier {tier}'",
            flush=True,
        )
    if not args.dry_run:
        if skipped:
            print("skipped, their models not on this machine: " + ", ".join(skipped), file=sys.stderr)
        if failed:
            print("failed: " + ", ".join(failed), file=sys.stderr)
        else:
            print(
                "all selected checks and gates passed" + (f" ({len(skipped)} skipped)" if skipped else ""),
                file=sys.stderr,
            )
    return 1 if failed else 0


def git_rev():
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()


# ---- self-test ----------------------------------------------------------------


def sample_doc():
    return {
        "build": {"metal_optimize": "ReleaseSafe", "cpu_optimize": "ReleaseFast", "cache": ".zig-cache/global"},
        "models": {"a": {"path": "~/m/a.gguf"}, "a_mtp": {"path": "~/m/a-mtp.gguf"}},
        "gates": [
            {
                "name": "a-trace",
                "family": "a",
                "tier": "verify",
                "model": "a",
                "paths": ["inference/src/models/a*.zig", "inference/src/quant/**"],
                "command": ["{nuclis}", "generate", "--model", "{model}", "--trace-dir", "{trace}"],
                "comparator": {"kind": "trace", "reference": "tests/fixtures/a", "positions": 2},
                "bounds": {"max_absolute": 0.002, "max_relative_rms": 0.0001},
                "evidence": "docs/a.md",
            },
            {
                "name": "a-draft-cpu",
                "family": "a",
                "tier": "verify-cpu",
                "model": "a",
                "mtp": "a_mtp",
                "paths": ["inference/src/backends/cpu/**"],
                "command": ["{zig-cpu}", "test-generation", "--", "{model}", "--draft-model", "{mtp}"],
                "comparator": {"kind": "exit"},
                "evidence": "docs/a.md",
            },
            {
                "name": "a-perplexity-4k",
                "family": "a",
                "tier": "verify-long",
                "model": "a",
                "paths": ["inference/src/backends/metal/**"],
                "command": ["{nuclis}", "eval", "--model", "{model}", "--file", "t.raw", "--reference", "r.json"],
                "comparator": {"kind": "exit"},
                "evidence": "docs/a.md",
            },
        ],
        "checks": [
            {"name": "fmt", "command": ["zig", "fmt", "--check", "{zig-files}"], "paths": ["**/*.zig"]},
            {"name": "metal", "command": ["make", "test-metal"], "paths": ["inference/src/backends/**"]},
        ],
        "requires": [
            {
                "tier": "verify-cpu",
                "paths": ["inference/src/backends/cpu/**", "inference/src/models/*_runtime.zig"],
                "reason": "the CPU reference",
            },
        ],
    }


class SelfTest(unittest.TestCase):
    def test_sample_is_valid(self):
        self.assertEqual(validate(sample_doc()), [])

    def test_validation_names_each_problem(self):
        doc = sample_doc()
        doc["gates"].append(dict(doc["gates"][0]))  # duplicate name
        doc["gates"][1]["tier"] = "cheap"
        doc["gates"][1]["model"] = "nope"
        doc["gates"][0]["command"] = ["{bogus}"]
        doc["gates"][0]["comparator"] = {"kind": "report"}
        problems = validate(doc)
        for needle in (
            "duplicate name",
            "tier must be",
            'unknown model "nope"',
            "unknown placeholder {bogus}",
            "comparator.kind",
        ):
            self.assertTrue(any(needle in p for p in problems), (needle, problems))

    def test_exit_gate_refuses_bounds_and_trace_gate_needs_them(self):
        doc = sample_doc()
        doc["gates"][1]["bounds"] = {"max_absolute": 1, "max_relative_rms": 1}
        del doc["gates"][0]["bounds"]
        problems = validate(doc)
        self.assertTrue(any("carries no bounds" in p for p in problems))
        self.assertTrue(any("need bounds" in p for p in problems))

    def test_globs(self):
        self.assertTrue(matches("inference/src/quant/**", "inference/src/quant/q4k.zig"))
        self.assertTrue(matches("inference/src/quant/**", "inference/src/quant/fixtures/x.json"))
        self.assertTrue(matches("inference/src/models/a*.zig", "inference/src/models/a_metal.zig"))
        self.assertFalse(matches("inference/src/models/a*.zig", "inference/src/models/b.zig"))
        self.assertFalse(matches("inference/src/models/a*.zig", "inference/src/models/fixtures/a.zig"))
        self.assertFalse(matches("inference/src/quant/**", "src/quant/x.zig"))

    def test_select_by_changed_paths(self):
        gates = sample_doc()["gates"]
        self.assertEqual([g["name"] for g in select(gates, ["src/tui/editor.zig"])], [])
        self.assertEqual([g["name"] for g in select(gates, ["inference/src/models/a_runtime.zig"])], ["a-trace"])
        self.assertEqual(
            [g["name"] for g in select(gates, ["inference/src/backends/cpu/vector.zig", "inference/src/quant/q4.zig"])],
            ["a-trace", "a-draft-cpu"],
        )

    def test_missing_models_are_listed(self):
        doc = sample_doc()
        gate = doc["gates"][1]
        env = {"A_MODEL": "/nowhere/a.gguf", "A_MTP_MODEL": "/nowhere/mtp.gguf"}
        self.assertEqual(missing_files(gate, doc, env=env), ["/nowhere/a.gguf", "/nowhere/mtp.gguf"])
        self.assertEqual(missing_files(gate, doc, env={"A_MODEL": __file__, "A_MTP_MODEL": __file__}), [])

    def test_expand_placeholders(self):
        doc = sample_doc()
        argv = expand(doc["gates"][1]["command"], doc["gates"][1], doc, env={"HOME": "/h"})
        self.assertEqual(argv[:3], ["zig", "build", "-Doptimize=ReleaseFast"])
        self.assertEqual(
            argv[-3:], [os.path.expanduser("~/m/a.gguf"), "--draft-model", os.path.expanduser("~/m/a-mtp.gguf")]
        )
        argv = expand(doc["gates"][0]["command"], doc["gates"][0], doc, env={"A_MODEL": "/elsewhere/a.gguf"})
        self.assertEqual(
            argv,
            [
                "./zig-out/bin/nuclis",
                "generate",
                "--model",
                "/elsewhere/a.gguf",
                "--trace-dir",
                f"{TRACE_ROOT}/a-trace",
            ],
        )

    def test_checks_and_requirements_follow_paths(self):
        doc = sample_doc()
        changed = ["inference/src/backends/cpu/vector.zig", "docs/x.md"]
        self.assertEqual([c["name"] for c in select(doc["checks"], changed)], ["fmt", "metal"])
        self.assertEqual(
            check_argv(doc["checks"][0], changed), ["zig", "fmt", "--check", "inference/src/backends/cpu/vector.zig"]
        )
        self.assertIsNone(check_argv(doc["checks"][0], ["docs/x.md"]))
        self.assertEqual(
            requirements(doc["requires"], changed),
            [("verify-cpu", "inference/src/backends/cpu/vector.zig", "the CPU reference")],
        )
        self.assertEqual(requirements(doc["requires"], ["src/tui/editor.zig"]), [])

    def test_by_cost_puts_unmeasured_gates_after_cheap_ones(self):
        gates = [{"name": n, "model": "a"} for n in ("slow", "new", "cheap", "mid")]
        order = [g["name"] for g in by_cost(gates, {"slow": 120.0, "cheap": 0.5, "mid": 10.0})]
        self.assertEqual(order, ["cheap", "mid", "new", "slow"])

    def test_by_cost_keeps_a_models_gates_together(self):
        gates = [
            {"name": "a-big", "model": "a"},
            {"name": "b-small", "model": "b"},
            {"name": "a-small", "model": "a"},
            {"name": "b-big", "model": "b"},
        ]
        times = {"a-big": 50.0, "a-small": 2.0, "b-small": 0.5, "b-big": 9.0}
        self.assertEqual([g["name"] for g in by_cost(gates, times)], ["b-small", "b-big", "a-small", "a-big"])

    def test_unit_base_reads_the_first_unit(self):
        text = "## A\n\nBase: `acce06a` (the commit)\n\n## B\n\nBase: `1234567`\n"
        self.assertEqual(unit_base(text), "acce06a")
        self.assertIsNone(unit_base("no base here, Base: `acce06a` inline"))

    def test_validation_names_bad_checks_and_rules(self):
        doc = sample_doc()
        doc["checks"].append({"name": "bad", "command": [], "paths": []})
        doc["requires"].append({"tier": "verify", "paths": ["x"], "reason": "r"})
        problems = validate(doc)
        for needle in ("bad: command", "bad: paths", "requires #1: tier"):
            self.assertTrue(any(needle in p for p in problems), (needle, problems))

    def test_builds_needed(self):
        gates = sample_doc()["gates"]
        self.assertEqual(builds_needed(gates), {"metal": True, "cpu": False, "checks-metal": False, "checks-cpu": True})
        self.assertEqual(
            builds_needed([gates[1]]), {"metal": False, "cpu": False, "checks-metal": False, "checks-cpu": True}
        )


def self_test():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SelfTest)
    return unittest.TextTestRunner(verbosity=0).run(suite).wasSuccessful()


if __name__ == "__main__":
    main()
