#!/usr/bin/env python3
"""The documents' generated blocks: what the code and the manifests know, written once.

A block is the text between `<!-- gen:NAME -->` and `<!-- /gen -->` (or
`<!-- gen:NAME:ARG -->` for one command, kind, or family). Blocks render from
`docs/facts.json` and the committed manifests (`src/catalog.json`,
`gates.json`, `workloads.json`), so checking them needs no build.
`docs/facts.json` is collected from the source tables (encodings, profiles,
limits, `NUCLIS_*` variables) and the built binary's help pages.

  docs-gen.py                     rewrite every block from the committed facts
  docs-gen.py --facts BIN         collect the facts with BIN first, then rewrite
  docs-gen.py --check             fail when a block differs from its rendering
  docs-gen.py --check-facts BIN   fail when docs/facts.json differs from BIN's

A block is never edited by hand: prose around it explains, the block states
(docs/README.md § Generated blocks). Standard library only.
"""

import argparse
import json
import os
import posixpath
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FACTS = "docs/facts.json"
BLOCK = re.compile(r"(?<!`)<!-- gen:([\w-]+)(?::([\w.-]+))? -->(.*?)<!-- /gen -->", re.DOTALL)

# Host constants a document states, by the file and declaration that own them.
LIMITS = [
    ("context ceiling (tokens)", "inference/src/runtime/session.zig", "max_capacity"),
    ("output budget (tokens)", "src/config.zig", "max_output_tokens"),
    ("draft length", "inference/src/engine.zig", "max_draft_length"),
    ("prefill chunk (tokens)", "inference/src/engine.zig", "prefill_chunk"),
    ("config file (bytes)", "src/config.zig", "max_file_bytes"),
    ("agent steps per turn", "src/agent/loop.zig", "budget_default"),
    ("agent turns per session", "src/agent/loop.zig", "max_turns"),
    ("agent instructions file (bytes)", "src/agent/system_prompt.zig", "max_instructions_bytes"),
    ("read_file lines per call", "src/agent/tools/read_file.zig", "max_lines"),
    ("read_file default lines", "src/agent/tools/read_file.zig", "default_lines"),
    ("read_file bytes", "src/agent/tools/read_file.zig", "max_bytes"),
    ("grep matches shown", "src/agent/tools/grep.zig", "max_shown"),
    ("grep files scanned", "src/agent/tools/grep.zig", "max_files"),
    ("grep seconds", "src/agent/tools/grep.zig", "max_seconds"),
    ("glob results", "src/agent/tools/glob.zig", "max_results"),
    ("image tokens per image", "inference/src/vision/root.zig", "max_image_tokens"),
    ("server output tokens", "src/api/chat/wire.zig", "max_output_tokens"),
    ("server audio clip (bytes)", "src/api/chat/wire.zig", "max_audio_bytes"),
    ("server web rounds", "src/api/chat/service.zig", "max_web_rounds"),
    ("server speech input (bytes)", "src/api/speech/wire.zig", "max_input_bytes"),
    ("server live transcriptions", "src/api/transcription/stream.zig", "max_sessions"),
    ("decision questions per call", "inference/src/decide.zig", "max_questions"),
    ("decision states per call", "inference/src/decide.zig", "max_states"),
]
ENV_SOURCES = ("src", "inference", "huggingface")
ENV = re.compile(r'"(NUCLIS_[A-Z_]+)"')
LAYOUT = re.compile(r'(\d+) => \.\{ \.name = "(\w+)", \.elements_per_block = (\d+), \.bytes_per_block = (\d+) \}')
TEMPLATE = re.compile(r'pub const template_sha256 = "([0-9a-f]{64})";')


# ---- facts --------------------------------------------------------------------


def constant(text, name):
    """The integer a `const NAME = <literal or product>;` declares, or None."""
    m = re.search(rf"\bconst {name}\s*(?::\s*\w+)?\s*=\s*([0-9_ *]+);", text)
    if not m:
        return None
    value = 1
    for factor in m[1].split("*"):
        value *= int(factor.strip().replace("_", ""))
    return value


def help_sections(page):
    """A help page's sections by title (`Usage:` → its indented lines)."""
    sections, title = {}, None
    for line in page.splitlines():
        if line and not line.startswith(" ") and line.endswith(":"):
            title = line[:-1]
            sections[title] = []
        elif title and line.startswith("  "):
            sections[title].append(line.rstrip())
        elif not line.strip():
            continue
        else:
            title = None
    return sections


def two_columns(lines):
    """(term, text) rows of a help section; continuation lines join the text."""
    rows = []
    for line in lines:
        m = re.match(r"^  (\S.*?)\s{2,}(\S.*)$", line)
        if m and not line.startswith("   "):
            rows.append([m[1], m[2]])
        elif rows and line.startswith("   "):
            rows[-1][1] += " " + line.strip()
        else:
            rows.append([line.strip(), ""])
    return rows


def flags(lines):
    """Every `--flag` an Options section names."""
    found = []
    for term, _ in two_columns(lines):
        for flag in re.findall(r"(?<![\w-])(--[a-z][\w-]*|-[a-zA-Z])\b", term):
            if flag not in found:
                found.append(flag)
    return found


def run_help(binary, *args):
    out = subprocess.run([binary, *args], capture_output=True, text=True, env={**os.environ, "NO_COLOR": "1"})
    if out.returncode != 0:
        raise SystemExit(f"docs-gen: {binary} {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout


def collect(binary):
    """Every fact a block renders from: the source tables and BIN's help pages."""
    encoding = (ROOT / "inference/src/tensor/encoding.zig").read_text()
    encodings = [
        {"id": int(i), "name": n, "elements_per_block": int(e), "bytes_per_block": int(b)}
        for i, n, e, b in LAYOUT.findall(encoding)
    ]
    if not encodings:
        raise SystemExit("docs-gen: no layouts found in inference/src/tensor/encoding.zig `layout`")
    profiles = []
    root = (ROOT / "inference/src/profiles/root.zig").read_text()
    names = re.search(r"pub const Profile = enum \{([^}]*?)\n\n", root)
    for name in re.findall(r"^\s*(\w+),", names[1], re.MULTILINE) if names else []:
        m = TEMPLATE.search((ROOT / f"inference/src/profiles/{name}.zig").read_text())
        profiles.append({"name": name, "template_sha256": m[1] if m else None})
    limits = []
    for label, path, name in LIMITS:
        value = constant((ROOT / path).read_text(), name)
        if value is None:
            raise SystemExit(f"docs-gen: {path} declares no integer `{name}`")
        limits.append({"limit": label, "value": value, "source": f"{path} `{name}`"})
    env = {}
    for top in ENV_SOURCES:
        for path in sorted((ROOT / top).rglob("*.zig")):
            rel = path.relative_to(ROOT).as_posix()
            if "/test" in rel or rel.endswith("-check.zig") or "fixtures" in rel:
                continue
            for var in ENV.findall(path.read_text(errors="replace")):
                env.setdefault(var, [])
                if rel not in env[var]:
                    env[var].append(rel)
    overview = help_sections(run_help(binary, "--help"))
    commands = two_columns(overview.get("Commands", []))
    cli = {}
    for topic in topics(commands):
        page = help_sections(run_help(binary, topic, "--help"))
        usage = [line.strip() for line in page.get("Usage", [])]
        named = flags(page.get("Options", []))
        named += [f for f in re.findall(r"(?<![\w-])(--[a-z][\w-]*)", " ".join(usage)) if f not in named]
        cli[topic] = {"usage": usage, "flags": named}
    config_page = help_sections(run_help(binary, "config", "--help"))
    with tempfile.TemporaryDirectory() as home:
        subprocess.run(
            [binary, "config", "init"], env={**os.environ, "NUCLIS_HOME": home}, capture_output=True, check=True
        )
        config_init = (Path(home) / "nuclis.json").read_text()
    return {
        "$comment": "Written by scripts/repo/docs-gen.py --facts; never by hand (docs/README.md § Generated blocks).",
        "commands": [{"command": c, "summary": s} for c, s in commands],
        "cli": cli,
        "config_keys": config_page.get("Keys", []),
        "config_init": config_init,
        "encodings": encodings,
        "profiles": profiles,
        "limits": limits,
        "env": [{"variable": k, "read_in": v} for k, v in sorted(env.items())],
    }


def topics(commands):
    """Command names with a page: the overview's terms, `speak, voice, transcribe` split."""
    out = []
    for term, _ in commands:
        out += [t.strip() for t in term.split(",")]
    return out


# ---- rendering ----------------------------------------------------------------


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(str(c).replace("|", "\\|") for c in row) + " |" for row in rows]
    return "\n".join(lines)


def size(n):
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.1f} MB"


def link(source, target):
    """A Markdown link to a `docs/…#anchor` evidence path from the file `source`."""
    path, _, anchor = target.partition("#")
    rel = posixpath.relpath(path, posixpath.dirname(source))
    label = posixpath.basename(path) + (f" § {anchor}" if anchor else "")
    return f"[{label}]({rel}{'#' + anchor if anchor else ''})"


def shape_kind(shape):
    """A workload shape's kind; a shape with a `base` and a `pair` is a speculative pair."""
    return shape.get("kind") or ("speculative pair" if "pair" in shape else "")


def render(name, arg, source, facts, manifests):
    """The text of one block, without its surrounding newlines; ValueError if unknown."""
    catalog, gates, workloads = manifests
    if name == "cli-commands":
        return table(["Command", "What it does"], [[f"`{c['command']}`", c["summary"]] for c in facts["commands"]])
    if name == "cli-synopsis":
        cmds = [arg] if arg else list(facts["cli"])
        if arg and arg not in facts["cli"]:
            raise ValueError(f"no command {arg}")
        lines = [u for c in cmds for u in facts["cli"][c]["usage"]]
        return "```text\n" + "\n".join(lines) + "\n```"
    if name == "config-keys":
        return "```text\n" + "\n".join(line[2:] for line in facts["config_keys"]) + "\n```"
    if name == "config-init-example":
        return "```json\n" + facts["config_init"].rstrip("\n") + "\n```"
    if name == "catalogue":
        rows = []
        for a in catalog["artifacts"]:
            if arg and a["kind"] != arg:
                continue
            main = next((f for f in a["files"] if f["role"] == "main"), a["files"][0])
            rows.append([f"`{a['name']}`", a["kind"], a["quantization"], size(main["size"]), a["status"]])
        if not rows:
            raise ValueError(f"no artifacts of kind {arg}")
        return table(["Name", "Kind", "Encoding", "Main file", "Status"], rows)
    if name == "gates-by-tier":
        tiers = {}
        for g in gates["gates"]:
            tiers.setdefault(g["tier"], []).append(g["name"])
        rows = [[f"`{t}`", len(n), ", ".join(f"`{x}`" for x in n)] for t, n in tiers.items()]
        return table(["Tier", "Gates", "Names"], rows) + f"\n\n{len(gates['gates'])} gates."
    if name == "gates-for":
        rows = [
            [f"`{g['name']}`", f"`{g['tier']}`", g.get("model") or "", link(source, g["evidence"])]
            for g in gates["gates"]
            if g["family"] == arg
        ]
        if not rows:
            raise ValueError(f"no gates for family {arg}")
        return table(["Gate", "Tier", "Model", "Evidence"], rows)
    if name == "workloads":
        rows = [
            [f"`{row}/{shape}`", m["model"], shape_kind(workloads["shapes"][shape])]
            for row, m in workloads["models"].items()
            for shape in m["shapes"]
        ]
        return table(["Workload", "Model", "Kind"], rows)
    if name == "encodings":
        rows = [[e["id"], f"`{e['name']}`", e["elements_per_block"], e["bytes_per_block"]] for e in facts["encodings"]]
        return table(["ID", "Encoding", "Values per block", "Bytes per block"], rows)
    if name == "encodings-count":
        return str(len(facts["encodings"]))
    if name == "profiles":
        rows = [
            [f"`{p['name']}`", f"`{p['template_sha256']}`" if p["template_sha256"] else ""] for p in facts["profiles"]
        ]
        return table(["Profile", "Template SHA-256"], rows)
    if name == "limits":
        return table(
            ["Limit", "Value", "Declared in"], [[x["limit"], f"{x['value']:,}", x["source"]] for x in facts["limits"]]
        )
    if name == "env":
        rows = [[f"`{e['variable']}`", ", ".join(f"`{p}`" for p in e["read_in"])] for e in facts["env"]]
        return table(["Variable", "Read in"], rows)
    raise ValueError(f"unknown block {name}")


def rewrite(text, source, facts, manifests):
    """`text` with every block re-rendered; raises ValueError naming a bad block."""

    def one(m):
        body = render(m[1], m[2], source, facts, manifests)
        inner = body if "\n" not in body and "\n" not in m[3] else f"\n{body}\n"
        head = f"<!-- gen:{m[1]}{':' + m[2] if m[2] else ''} -->"
        return f"{head}{inner}<!-- /gen -->"

    return BLOCK.sub(one, text)


def markdown_files():
    out = subprocess.run(["git", "ls-files", "*.md"], cwd=ROOT, capture_output=True, text=True, check=True)
    return [p for p in out.stdout.split() if (ROOT / p).exists()]


def load():
    facts = json.loads((ROOT / FACTS).read_text())
    manifests = tuple(json.loads((ROOT / p).read_text()) for p in ("src/catalog.json", "gates.json", "workloads.json"))
    return facts, manifests


def blocks(write):
    facts, manifests = load()
    stale = []
    for path in markdown_files():
        text = (ROOT / path).read_text()
        if "<!-- gen:" not in text:
            continue
        try:
            new = rewrite(text, path, facts, manifests)
        except ValueError as e:
            stale.append(f"{path}: {e}")
            continue
        if new != text:
            if write:
                (ROOT / path).write_text(new)
            stale.append(f"{path}: generated block out of date (make docs-gen)")
    return stale


def write_facts(facts):
    (ROOT / FACTS).write_text(json.dumps(facts, indent=2, ensure_ascii=False) + "\n")


# ---- self-test ------------------------------------------------------------------


class SelfTest(unittest.TestCase):
    PAGE = (
        "nuclis x — a thing\n\nUsage:\n  nuclis x [--a] <file>\n\nOptions:\n"
        "  --alpha, -a             one\n                          two\n  --beta N                three\n\n"
        "Examples:\n  nuclis x\n"
    )

    def test_sections_and_flags(self):
        s = help_sections(self.PAGE)
        self.assertEqual(s["Usage"], ["  nuclis x [--a] <file>"])
        self.assertEqual(two_columns(s["Options"])[0], ["--alpha, -a", "one two"])
        self.assertEqual(flags(s["Options"]), ["--alpha", "-a", "--beta"])

    def test_constant(self):
        self.assertEqual(constant("pub const max_bytes: usize = 4 * 1024 * 1024;", "max_bytes"), 4 << 20)
        self.assertEqual(constant("const max_walk = 20_000;", "max_walk"), 20000)
        self.assertIsNone(constant("const other = 1;", "max_walk"))

    def test_rewrite_block_and_inline(self):
        facts = {"encodings": [{"id": 0, "name": "F32", "elements_per_block": 1, "bytes_per_block": 4}]}
        text = "n = <!-- gen:encodings-count -->9<!-- /gen -->\n<!-- gen:encodings -->\nold\n<!-- /gen -->\n"
        out = rewrite(text, "docs/a.md", facts, ({}, {}, {}))
        self.assertIn("n = <!-- gen:encodings-count -->1<!-- /gen -->", out)
        self.assertIn("<!-- gen:encodings -->\n| ID |", out)
        self.assertEqual(rewrite(out, "docs/a.md", facts, ({}, {}, {})), out)

    def test_unknown_block(self):
        with self.assertRaises(ValueError):
            rewrite("<!-- gen:nothing -->\n<!-- /gen -->", "docs/a.md", {}, ({}, {}, {}))

    def test_link(self):
        self.assertEqual(
            link("docs/models/gemma4.md", "docs/engine/sampling.md#traces"),
            "[sampling.md § traces](../engine/sampling.md#traces)",
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--facts", metavar="BIN", help="collect docs/facts.json with this nuclis binary first")
    ap.add_argument("--check", action="store_true", help="fail when a block is out of date; write nothing")
    ap.add_argument("--check-facts", metavar="BIN", help="fail when docs/facts.json differs from what BIN reports")
    ap.add_argument("--self-test", action="store_true", help="run the unit tests of the pure functions")
    args = ap.parse_args()
    if args.self_test:
        sys.argv = sys.argv[:1]
        unittest.main(module=__name__, verbosity=1)
    if args.check_facts:
        fresh = json.dumps(collect(args.check_facts), indent=2, ensure_ascii=False) + "\n"
        if fresh != (ROOT / FACTS).read_text():
            print(f"docs-gen: {FACTS} is out of date with the code (make docs-gen)", file=sys.stderr)
            return 1
        print(f"docs-gen: {FACTS} ok")
        return 0
    if args.facts:
        write_facts(collect(args.facts))
    problems = blocks(write=not args.check)
    if args.check:
        for p in problems:
            print(f"docs-gen: {p}", file=sys.stderr)
        if problems:
            return 1
        print("docs-gen: every generated block is current")
        return 0
    for p in problems:
        print(f"docs-gen: {p.replace('out of date (make docs-gen)', 'rewritten')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
