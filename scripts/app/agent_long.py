"""The agent's long tasks: one from-scratch app per task, built over several
`nuclis agent --print` turns continued through `--resume`, in an empty
workspace, then checked by running what was built.

A long task is where a harness change shows on a real build: dropped calls,
failed-edit streaks, replays, reasoning cuts, and the window filling up.
Each check of the predicate is recorded separately, so a partial app still
scores. Records go under `.zig-cache/agent-eval/<variant>/long/`.

    python3 scripts/app/agent-eval.py --variant long-base --long todo-bun \\
        --model qwen3.8-27b --seeds 1
    python3 scripts/app/agent-eval.py --long-compare long-base long-x
"""

import json
import os
import shutil
import signal
import sqlite3
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "agent-eval" / "long"
PORT = 3000
BASE = f"http://localhost:{PORT}"
CHECKS = ("build", "serves", "wal", "crud")


@dataclass
class LongTask:
    name: str
    prompt_file: Path
    # The reply to the design the prompt asks for first.
    confirm: str
    # Sent while the predicate fails, at most `continues` times.
    follow_up: str
    continues: int
    wall_seconds: int
    check: Callable[[Path], dict]


# ----- the todo-bun predicate -------------------------------------------------


def _bun_sources(ws: Path) -> list:
    skip = {"node_modules", ".git", "dist", "build"}
    out = []
    for p in sorted(ws.rglob("*.ts")):
        if not any(part in skip for part in p.relative_to(ws).parts):
            out.append(p)
    return out


def _entry(ws: Path) -> list:
    """The command that starts the app: package.json's start or dev script,
    else the first conventional entry file."""
    pkg = ws / "package.json"
    if pkg.exists():
        try:
            scripts = json.loads(pkg.read_text()).get("scripts", {}) or {}
        except (json.JSONDecodeError, AttributeError):
            scripts = {}
        for name in ("start", "dev"):
            if name in scripts:
                return ["bun", "run", name]
    for rel in ("server.ts", "index.ts", "src/server.ts", "src/index.ts", "app.ts", "src/app.ts"):
        if (ws / rel).exists():
            return ["bun", rel]
    return []


def _request(method: str, path: str, body=None, timeout=5.0):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, raw
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except (urllib.error.URLError, OSError):
        return 0, b""


def _free_port():
    """Kills whatever an earlier run left listening on the port."""
    r = subprocess.run(["lsof", "-ti", f"tcp:{PORT}", "-sTCP:LISTEN"], capture_output=True, text=True)
    for pid in r.stdout.split():
        try:
            os.kill(int(pid), signal.SIGKILL)
        except (ProcessLookupError, ValueError, PermissionError):
            pass


def check_todo_bun(ws: Path) -> dict:
    result = {c: False for c in CHECKS}
    notes = []
    sources = _bun_sources(ws)
    if not sources:
        notes.append("no .ts files")
    else:
        failed = []
        with tempfile.TemporaryDirectory() as out:
            for src in sources:
                r = subprocess.run(
                    ["bun", "build", str(src), "--target=bun", "--outdir", out],
                    cwd=ws,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if r.returncode != 0:
                    failed.append(f"{src.relative_to(ws)}: {(r.stderr or r.stdout).strip().splitlines()[-1:]}")
        result["build"] = not failed
        if failed:
            notes.append("build: " + "; ".join(failed[:3]))
    cmd = _entry(ws)
    if not cmd:
        notes.append("no entry to start")
        return {**result, "note": " | ".join(notes)}
    _free_port()
    db = ws / "todos.db"
    server = subprocess.Popen(
        cmd, cwd=ws, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True
    )
    try:
        deadline = time.monotonic() + 10
        status = 0
        while time.monotonic() < deadline:
            status, _ = _request("GET", "/api/todos", timeout=2)
            if status == 200 or server.poll() is not None:
                break
            time.sleep(0.25)
        result["serves"] = status == 200
        if not result["serves"]:
            notes.append(f"GET /api/todos: {status or 'no answer'}")
        if db.exists():
            try:
                con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=2)
                mode = con.execute("PRAGMA journal_mode").fetchone()[0]
                con.close()
                result["wal"] = str(mode).lower() == "wal"
                if not result["wal"]:
                    notes.append(f"journal_mode {mode}")
            except sqlite3.Error as exc:
                notes.append(f"todos.db: {exc}")
        else:
            notes.append("no ./todos.db")
        if result["serves"]:
            result["crud"], note = _crud()
            if note:
                notes.append(note)
    finally:
        try:
            os.killpg(server.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        out_text = server.communicate(timeout=5)[0] if server.stdout else ""
        if not result["serves"] and out_text:
            notes.append("server: " + out_text.strip().splitlines()[-1][:160] if out_text.strip() else "")
        _free_port()
    return {**result, "note": " | ".join(n for n in notes if n)}


def _crud() -> tuple:
    status, raw = _request("POST", "/api/todos", {"title": "eval probe", "priority": "high"})
    if status not in (200, 201):
        return False, f"POST: {status}"
    try:
        body = json.loads(raw or b"{}")
    except json.JSONDecodeError:
        return False, "POST: not JSON"
    todo = body.get("todo", body) if isinstance(body, dict) else {}
    tid = todo.get("id") if isinstance(todo, dict) else None
    if tid is None:
        return False, "POST: no id"
    status, _ = _request("PATCH", f"/api/todos/{tid}", {"completed": True})
    if status not in (200, 204):
        return False, f"PATCH: {status}"
    status, _ = _request("DELETE", f"/api/todos/{tid}")
    if status not in (200, 204):
        return False, f"DELETE: {status}"
    status, raw = _request("GET", "/api/todos")
    try:
        listed = json.loads(raw or b"[]")
    except json.JSONDecodeError:
        return False, "GET after DELETE: not JSON"
    items = listed.get("todos", listed) if isinstance(listed, dict) else listed
    if any(isinstance(t, dict) and t.get("id") == tid for t in items or []):
        return False, "DELETE: still listed"
    return True, ""


LONG_TASKS = {
    "todo-bun": LongTask(
        name="todo-bun",
        prompt_file=FIXTURES / "todo-bun.md",
        confirm="I confirm the design, proceed with the implementation",
        follow_up="continue: make the definition of done pass (bun build, server on :3000, ./todos.db in WAL mode, the REST API working)",
        continues=4,
        wall_seconds=3600,
        check=check_todo_bun,
    ),
}


# ----- what the session says ---------------------------------------------------


def session_counts(session: Path, events: list) -> dict:
    """Failure modes the long task exists to count, read from the session
    file and the turns' event streams."""
    c = {
        "cut_calls": 0,
        "write_errors": 0,
        "edit_errors": 0,
        "edit_streak": 0,
        "bash_errors": 0,
        "replay_by_cause": {},
        "peak_ctx": 0,
        "context_full": 0,
        "cancelled_steps": 0,
    }
    for text in events:
        for line in text.splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("type") == "notice" and "did not parse" in e.get("text", ""):
                c["cut_calls"] += 1
            if e.get("type") == "turn_end" and e.get("stop") in ("context_limit", "context_full"):
                c["context_full"] += 1
    if not session.exists():
        return c
    names = {}
    streak = 0
    cut_entries = 0
    for line in session.read_text().splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = e.get("type")
        if kind == "cut":
            cut_entries += 1
        if kind == "assistant":
            st = e.get("stats", {})
            if e.get("stop") == "cancelled":
                c["cancelled_steps"] += 1
            ctx = st.get("reused_tokens", 0) + st.get("prompt_tokens", 0) + st.get("generated", 0)
            c["peak_ctx"] = max(c["peak_ctx"], ctx)
            if st.get("replayed") or st.get("lost_tokens", 0) > 0:
                cause = st.get("replay") or st.get("reuse") or "?"
                agg = c["replay_by_cause"].setdefault(cause, [0, 0, 0.0])
                agg[0] += 1
                agg[1] += st.get("prompt_tokens", 0)
                agg[2] = round(agg[2] + st.get("prefill_seconds", 0), 1)
            for call in e.get("tool_calls", []):
                names[str(call.get("id"))] = call.get("name")
        elif kind == "tool_result":
            name = names.get(str(e.get("call")))
            if e.get("is_error"):
                if name == "write_file":
                    c["write_errors"] += 1
                if name == "bash":
                    c["bash_errors"] += 1
                if name == "edit_file":
                    c["edit_errors"] += 1
                    streak += 1
                    c["edit_streak"] = max(c["edit_streak"], streak)
            elif name == "edit_file":
                streak = 0
    # The notice and the session's `cut` entry name the same drop.
    c["cut_calls"] = max(c["cut_calls"], cut_entries)
    return c


# ----- running one -------------------------------------------------------------


def run_long(task: LongTask, seed: int, args, variant_dir: Path, parse, Run) -> dict:
    """Runs `task` once from an empty git workspace and returns its record.
    `parse` and `Run` are the task list's, so both read the session alike."""
    long_dir = variant_dir / "long"
    long_dir.mkdir(parents=True, exist_ok=True)
    ws = long_dir / f"{task.name}-s{seed}"
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=ws, check=True)
    session = long_dir / f"{task.name}-s{seed}.session.jsonl"
    if session.exists():
        session.unlink()
    started = time.monotonic()
    events, walls, turns_log = [], [], []

    def turn(prompt_args: list, resume: bool) -> tuple:
        left = task.wall_seconds - (time.monotonic() - started)
        if left <= 30:
            return "", "wall budget spent", -2
        cmd = [
            str(Path(args.binary).resolve()),
            "agent",
            *prompt_args,
            "--json",
            "--seed",
            str(seed),
            "--session",
            str(session),
        ]
        if resume:
            cmd += ["--resume", str(session)]
        if args.model:
            cmd += ["--model", args.model]
        if args.system_prompt:
            cmd += ["--system-prompt", str(Path(args.system_prompt).resolve())]
        cmd += args.extra
        t0 = time.monotonic()
        try:
            proc = subprocess.run(cmd, cwd=ws, capture_output=True, text=True, timeout=left)
            out, err, status = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            err, status = "timeout", -1
        wall = time.monotonic() - t0
        events.append(out)
        walls.append(wall)
        turns_log.append(
            {"prompt": " ".join(prompt_args)[:80], "status": status, "wall": round(wall, 1), "stderr": err[-300:]}
        )
        print(f"  turn {len(turns_log)}: exit {status}, {wall:.0f}s", flush=True)
        return out, err, status

    _, _, status = turn(["--prompt-file", str(task.prompt_file)], False)
    if status == 0:
        _, _, status = turn(["-p", task.confirm], True)
    checks = task.check(ws)
    continued = 0
    while status == 0 and not all(checks[c] for c in CHECKS) and continued < task.continues:
        _, _, status = turn(["-p", task.follow_up], True)
        continued += 1
        checks = task.check(ws)
    run = parse(
        events[-1] if events else "", session, Run(wall_seconds=time.monotonic() - started, exit_status=status), walls
    )
    record = {
        "task": task.name,
        "seed": seed,
        "variant": args.variant,
        "model": args.model,
        "extra": args.extra,
        "checks": {c: checks[c] for c in CHECKS},
        "passed": sum(1 for c in CHECKS if checks[c]),
        "note": checks.get("note", ""),
        "turns": len(turns_log),
        "turns_log": turns_log,
        "steps": run.steps,
        "calls": run.calls,
        "errors": run.errors,
        "prompt_tokens": run.prompt_tokens,
        "generated": run.generated,
        "reasoning_tokens": run.reasoning_tokens,
        "reasoning_cut": run.reasoning_cut,
        "model_seconds": round(run.prefill_seconds + run.decode_seconds, 1),
        "prefill_seconds": round(run.prefill_seconds, 1),
        "decode_seconds": round(run.decode_seconds, 1),
        "wall_seconds": round(time.monotonic() - started, 1),
        "drafted": run.drafted,
        "accepted": run.accepted,
        "dropped": run.dropped,
        "elided": run.elided,
        "tools": run.tools,
        **session_counts(session, events),
    }
    (long_dir / f"{task.name}-s{seed}.json").write_text(json.dumps(record, indent=1) + "\n")
    return record


def summary_line(r: dict) -> str:
    replay = sum(v[1] for v in r["replay_by_cause"].values())
    return (
        f"{r['variant']} {r['task']} s{r['seed']} {r['passed']}/4 {''.join('✓' if r['checks'][c] else '✗' for c in CHECKS)} "
        f"turns={r['turns']} steps={r['steps']} cut={r['cut_calls']} write_err={r['write_errors']} "
        f"edit_err={r['edit_errors']} (streak {r['edit_streak']}) rcut={r['reasoning_cut']} replay={replay} "
        f"peak={r['peak_ctx']} gen={r['generated']} model={r['model_seconds']}s wall={r['wall_seconds']}s {r['note'][:120]}"
    )


def load(variant_dir: Path) -> list:
    d = variant_dir / "long"
    if not d.is_dir():
        return []
    return [json.loads(p.read_text()) for p in sorted(d.glob("*-s*.json"))]


def compare(out_dir: Path, names: list) -> str:
    lines = [
        "| variant | task | seed | checks | turns | steps | cut | write err | edit err (streak) | bash err | reasoning cut | replayed tokens (s) | peak ctx | generated | model s | wall s |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for name in names:
        for r in load(out_dir / name):
            tok = sum(v[1] for v in r["replay_by_cause"].values())
            sec = sum(v[2] for v in r["replay_by_cause"].values())
            lines.append(
                f"| {name} | {r['task']} | {r['seed']} | {r['passed']}/4 {''.join('✓' if r['checks'][c] else '✗' for c in CHECKS)} | "
                f"{r['turns']} | {r['steps']} | {r['cut_calls']} | {r['write_errors']} | {r['edit_errors']} ({r['edit_streak']}) | "
                f"{r['bash_errors']} | {r['reasoning_cut']} | {tok} ({sec:.0f}) | {r['peak_ctx']} | {r['generated']} | "
                f"{r['model_seconds']} | {r['wall_seconds']} |"
            )
    return "\n".join(lines)
