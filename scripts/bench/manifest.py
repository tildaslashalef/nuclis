"""Reads what the bench scripts share with the gate manifest: model paths and builds."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "gates.json"
CPU_PREFIX = ".zig-cache/gates/cpu"


def load():
    return json.loads(MANIFEST.read_text())


def model_path(doc, key, env=None):
    """The pinned path, or the `<KEY>_MODEL` environment override; `~` expanded."""
    env = os.environ if env is None else env
    override = env.get(key.upper() + "_MODEL")
    path = override if override else doc["models"][key]["path"]
    return os.path.expanduser(path)


def build(doc, which, dry_run):
    """Build the `metal` or `cpu` binary; returns seconds."""
    b = doc["build"]
    if which == "metal":
        cmd = ["zig", "build", "-Dmetal=true", f"-Doptimize={b['metal_optimize']}"]
    else:
        cmd = ["zig", "build", "-Dmetal=true", f"-Doptimize={b['cpu_optimize']}", "--prefix", CPU_PREFIX]
    print("build:", " ".join(cmd), file=sys.stderr, flush=True)
    if dry_run:
        return 0.0
    started = time.monotonic()
    subprocess.run(cmd, cwd=ROOT, check=True)
    return time.monotonic() - started
