"""Reads what the bench scripts share with the gate manifest: model paths, builds, and the workloads.

A `gates.json` model names an index artifact (`src/catalog.json`, written by
`make index`) and a file of it, resolved here the way tools/gates/index.zig
resolves it; `workloads.json` is shapes × models, expanded by `workloads()`
into the named workloads every script reads.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "gates.json"
INDEX = ROOT / "src" / "catalog.json"
WORKLOADS = ROOT / "workloads.json"
CPU_PREFIX = ".zig-cache/gates/cpu"
MODELS_ROOT = "~/.nuclis/models/"


def index_artifact(index, name):
    return next((a for a in index["artifacts"] if a["name"] == name), None)


def index_file(artifact, selector):
    """The file of `selector`: main, mmproj, draft, draft:<variant>, or directory."""
    files = artifact["files"]
    if selector == "directory":
        return {"local": os.path.dirname(files[0]["local"]), "role": "directory"}
    role, _, variant = selector.partition(":")
    wanted = variant or (artifact["defaults"].get("draft") if role == "draft" else None)
    matches = [f for f in files if f["role"] == role]
    for f in matches:
        if wanted and f.get("variant") == wanted:
            return f
    return None if variant or not matches else matches[0]


def resolve(doc, index):
    """Fills `path`, `entry`, and `mmproj` of every model that names an artifact."""
    for model in doc["models"].values():
        name = model.get("artifact")
        if name is None:
            continue
        artifact = index_artifact(index, name)
        selector = model.get("file", "main")
        f = index_file(artifact, selector) if artifact else None
        if f is None:
            raise KeyError(f"{name} has no file {selector}")
        model["path"] = MODELS_ROOT + f["local"]
        if selector in ("main", "directory"):
            model["entry"] = name
            projector = index_file(artifact, "mmproj")
            if projector:
                model["mmproj"] = MODELS_ROOT + projector["local"]
    return doc


def load_index():
    return json.loads(INDEX.read_text())


def load():
    return resolve(json.loads(MANIFEST.read_text()), load_index())


def model_path(doc, key, env=None):
    """The pinned path, or the `<KEY>_MODEL` environment override; `~` expanded."""
    env = os.environ if env is None else env
    override = env.get(key.upper() + "_MODEL")
    path = override if override else doc["models"][key]["path"]
    return os.path.expanduser(path)


def draft_length(gdoc, index, key, variant=None):
    """The measured draft length of a model's default draft (or `variant`), else 4."""
    artifact = index_artifact(index, gdoc["models"][key].get("artifact", ""))
    f = index_file(artifact, "draft:" + variant if variant else "draft") if artifact else None
    return (f.get("verdict") or {}).get("draft_length", 4) if f else 4


def expand(wdoc, gdoc, index):
    """`shapes` × `models` as {"<row>/<shape>": workload}. A shape's `base` is
    another shape it extends; `prompt_tokens` is relative to the row's run;
    the row's `bars`, `note`, and `evidence`, keyed by shape, win."""
    shapes = wdoc.get("shapes", {})

    def shape(name, seen=()):
        s = dict(shapes[name])
        base = s.pop("base", None)
        if base is not None:
            if base in seen or base not in shapes:
                raise KeyError(f"shape {name}: base {base}")
            s = {**shape(base, seen + (name,)), **s}
        return s

    out = {}
    for row, m in wdoc.get("models", {}).items():
        for name in m.get("shapes", []):
            w = shape(name)
            w["model"] = m["model"]
            if w.get("kind") == "acceptance":
                for key in ("run", "reference_records", "record_suffix"):
                    if key in m:
                        w[key] = m[key]
            if "prompt_tokens" in w:
                w["prompt_tokens"] = f"tests/fixtures/{m['run']}/{w['prompt_tokens']}"
            if w.get("pair"):
                if m.get("draft"):
                    w["draft"] = m["draft"]
                if m["model"] in gdoc["models"]:
                    w.setdefault("draft_length", draft_length(gdoc, index, m["model"], m.get("draft")))
            for key in ("bars", "note", "evidence"):
                if name in m.get(key, {}):
                    w[key] = m[key][name]
            if "draft_length" in m.get("overrides", {}).get(name, {}):
                w["draft_length"] = m["overrides"][name]["draft_length"]
            out[f"{row}/{name}"] = w
    return out


def workloads(gdoc=None, index=None):
    index = load_index() if index is None else index
    gdoc = resolve(json.loads(MANIFEST.read_text()), index) if gdoc is None else gdoc
    return expand(json.loads(WORKLOADS.read_text()), gdoc, index)


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
