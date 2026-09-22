#!/usr/bin/env python3
"""The agent's task list: twelve small coding tasks against the playground
project, each run as one `nuclis agent --print` turn and scored on whether it
did the job and what it cost. The list is how a change to the system prompt,
a tool description, or the loop is judged: run it before and after, compare.

    make agent-eval VARIANT=baseline                 # the built-in prompt
    make agent-eval VARIANT=cand1 ARGS='--system-prompt p.txt'
    python3 scripts/agent-eval.py --compare baseline cand1

Each task starts from the playground's committed baseline (`make reset`
there), optionally applies a setup (a planted bug), runs the turn with a
fixed seed, checks the result (file contents, a command's output, the
answer's text), and records one JSON per task and seed under
.zig-cache/agent-eval/<variant>/<task>-s<seed>.json: pass/fail, steps, tool
calls, tool errors, failed edits, prompt and generated tokens, the model's
own seconds, the answer's length, and the stop reason. `--compare` renders
the variants side by side. The playground is a separate repository
(`~/Code/playground` by default); nothing here is committed to it.
"""
import argparse
import glob
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / '.zig-cache' / 'agent-eval'
DEFAULT_WORKSPACE = Path.home() / 'Code' / 'playground'
DEFAULT_BINARY = ROOT / 'zig-out' / 'bin' / 'nuclis'
DEFAULT_INSTRUCTIONS = ROOT / 'tests' / 'fixtures' / 'agent-eval' / 'AGENTS.md'
TURN_TIMEOUT = 900


@dataclass
class Task:
    name: str
    prompt: str
    check: Callable[[Path, 'Run'], tuple[bool, str]]
    setup: Optional[Callable[[Path], None]] = None


@dataclass
class Run:
    """What one turn produced, from the event stream and the session file."""
    answer: str = ''
    answers: list = field(default_factory=list)
    steps: int = 0
    calls: int = 0
    errors: int = 0
    edit_errors: int = 0
    prompt_tokens: int = 0
    generated: int = 0
    model_seconds: float = 0.0
    stop: str = ''
    tools: list = field(default_factory=list)
    wall_seconds: float = 0.0
    exit_status: int = 0


# ----- the tasks ------------------------------------------------------------

def tests_pass(ws: Path) -> bool:
    r = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests'],
                       cwd=ws, capture_output=True, text=True, timeout=120)
    return r.returncode == 0


def run_py(ws: Path, code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, '-c', code], cwd=ws, capture_output=True, text=True, timeout=60)


def bullets(text: str) -> int:
    return sum(1 for line in text.splitlines() if re.match(r'\s*([-*•]|\d+[.)])\s', line))


def has_all(text: str, needles) -> bool:
    return all(n in text for n in needles)


def check_count(ws: Path, run: Run):
    lines = sum(1 for _ in open(ws / 'data' / 'measurements.txt'))
    largest = max(float(l.split('area=')[1]) for l in open(ws / 'data' / 'measurements.txt'))
    want = [str(lines), f'{largest:.3f}']
    return has_all(run.answer, want), f'want {want}'


def check_history(ws: Path, run: Run):
    heads = [l[3:].strip() for l in open(ws / 'docs' / 'history.md') if l.startswith('## ')]
    want = [str(len(heads)), heads[-1]]
    return has_all(run.answer, want), f'want {want}'


def check_rename(ws: Path, run: Run):
    text = (ws / 'src' / 'shapes' / 'rect.py').read_text()
    ok = ('def rect_area(w: float, h: float)' in text and 'def rect_perimeter(w: float, h: float)' in text
          and 'width' not in text.replace('"', '') and tests_pass(ws))
    return ok, 'both signatures renamed, no `width` left, tests pass'


def check_heading(ws: Path, run: Run):
    text = (ws / 'docs' / 'design.md').read_text()
    plain = text.count('## Error handling\n')
    cli = text.count('## Error handling (CLI)')
    ok = plain == 1 and cli == 1 and text.index('## Error handling\n') < text.index('## Error handling (CLI)')
    return ok, f'plain={plain} cli={cli}'


def check_newfile(ws: Path, run: Run):
    if not (ws / 'src' / 'shapes' / 'sphere.py').exists() or not (ws / 'tests' / 'test_sphere.py').exists():
        return False, 'sphere.py or test_sphere.py missing'
    probe = run_py(ws, (
        "import sys, math; sys.path.insert(0, 'src')\n"
        "from shapes import sphere_volume, sphere_surface\n"
        "assert abs(sphere_volume(1) - 4/3*math.pi) < 1e-9\n"
        "assert abs(sphere_surface(1) - 4*math.pi) < 1e-9\n"
        "try:\n    sphere_volume(-1)\nexcept ValueError:\n    pass\nelse:\n    raise SystemExit('no ValueError')\n"))
    if probe.returncode != 0:
        return False, 'probe: ' + (probe.stderr.strip().splitlines() or ['?'])[-1]
    return tests_pass(ws), 'tests pass'


def setup_bugfix(ws: Path):
    path = ws / 'src' / 'shapes' / 'polygon.py'
    path.write_text(path.read_text().replace('return sides * length', 'return sides + length'))


def check_bugfix(ws: Path, run: Run):
    text = (ws / 'src' / 'shapes' / 'polygon.py').read_text()
    return 'return sides * length' in text and tests_pass(ws), 'multiplication restored, tests pass'


def check_exit(ws: Path, run: Run):
    low = run.answer.lower()
    ok = '7' in run.answer and any(w in low for w in ('nothing', 'no output', 'none', 'silent', 'empty', 'did not print', "didn't print"))
    return ok, 'status 7 and "printed nothing"'


def check_grep(ws: Path, run: Run):
    ok = has_all(run.answer, ['circle.py', 'polygon.py', 'rect.py']) and len(run.answer) <= 600
    return ok, 'three paths, short answer'


def check_json(ws: Path, run: Run):
    j = subprocess.run([sys.executable, 'src/cli.py', 'circle', '2', '--json'], cwd=ws, capture_output=True, text=True, timeout=60)
    t = subprocess.run([sys.executable, 'src/cli.py', 'circle', '2'], cwd=ws, capture_output=True, text=True, timeout=60)
    try:
        obj = json.loads(j.stdout)
        # Rounding to the text form's three decimals is a fair reading of the task.
        ok = abs(obj['area'] - 4 * math.pi) < 1e-3 and abs(obj['perimeter'] - 4 * math.pi) < 1e-3
    except Exception as exc:  # noqa: BLE001
        return False, f'json form: {exc}: {j.stdout.strip()[:80]} {j.stderr.strip()[:80]}'
    ok = ok and t.stdout.startswith('area=') and tests_pass(ws)
    return ok, 'json and text forms, tests pass'


def check_summary(ws: Path, run: Run):
    n = bullets(run.answer)
    ok = 1 <= n <= 3 and len(run.answer) <= 600 and any(w in run.answer.lower() for w in ('valueerror', 'dependency', 'error'))
    return ok, f'{n} bullets, {len(run.answer)} chars'


def check_big(ws: Path, run: Run):
    low = run.answer.lower()
    ok = any(w in low for w in ('value', 'tag')) and len(run.answer) <= 700 and run.steps <= 6
    return ok, f'{len(run.answer)} chars, {run.steps} steps'


def check_validate(ws: Path, run: Run):
    probe = run_py(ws, (
        "import sys; sys.path.insert(0, 'src')\n"
        "from shapes import polygon_area, polygon_perimeter\n"
        "assert abs(polygon_area(4, 1) - 2) < 1e-9 and polygon_perimeter(4, 1) == 4\n"
        "for f in (polygon_area, polygon_perimeter):\n"
        "    try:\n        f(3.5, 1)\n    except ValueError as e:\n        assert 'integer' in str(e), str(e)\n"
        "    else:\n        raise SystemExit('no ValueError')\n"))
    if probe.returncode != 0:
        return False, 'probe: ' + (probe.stderr.strip().splitlines() or ['?'])[-1]
    # Scope: the task asked for validation, not a formula change (the
    # fixture's area formula is off by a factor of two on purpose).
    if '(2 * math.tan(math.pi / sides))' not in (ws / 'src' / 'shapes' / 'polygon.py').read_text():
        return False, 'formula changed: out of scope'
    test_text = (ws / 'tests' / 'test_shapes.py').read_text()
    return 'integer' in test_text and tests_pass(ws), 'validation, its test, tests pass'


TASKS = [
    Task('count', 'How many lines does data/measurements.txt have, and what is the largest area value in it?', check_count),
    Task('history', 'How many release sections does docs/history.md have, and which version is the newest? Entries are newest last.', check_history),
    Task('rename', 'In src/shapes/rect.py rename the parameters width and height to w and h in both functions. Behaviour must not change.', check_rename),
    Task('heading', "In docs/design.md the heading 'Error handling' appears twice. Rename the second one to 'Error handling (CLI)' and leave the first alone.", check_heading),
    Task('newfile', 'Add src/shapes/sphere.py with sphere_volume(radius) and sphere_surface(radius) following the package conventions (validation included), export them from src/shapes/__init__.py, add tests in tests/test_sphere.py, and run the tests.', check_newfile),
    Task('bugfix', 'polygon_perimeter(4, 2) returns 6 but should return 8. Find the bug and fix it.', check_bugfix, setup_bugfix),
    Task('exit', 'Run scripts/exit7.sh and report its exit status and whether it printed anything.', check_exit),
    Task('grep', 'Which files under src raise ValueError? Give the paths only.', check_grep),
    Task('json', 'Add a --json flag to src/cli.py: when it is the last argument, print the metrics as a JSON object with keys area and perimeter instead of the text line. Keep the text form as the default.', check_json),
    Task('summary', 'Summarize docs/design.md in at most three bullet points.', check_summary),
    Task('big', 'Describe what data/big.txt contains, in two sentences.', check_big),
    Task('validate', "Make polygon_area and polygon_perimeter in src/shapes/polygon.py raise ValueError('sides must be an integer') when sides is not an int, and add a test for it in tests/test_shapes.py.", check_validate),
]


# ----- running -------------------------------------------------------------

def reset(ws: Path):
    subprocess.run(['make', '-s', 'reset'], cwd=ws, check=True, capture_output=True)


def parse(events_text: str, session_path: Path, run: Run) -> Run:
    names = {}
    for line in events_text.splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get('type') == 'turn_end':
            run.stop = e.get('stop', '')
            stats = e.get('stats', {})
            run.model_seconds = stats.get('prefill_seconds', 0) + stats.get('decode_seconds', 0)
    if session_path.exists():
        for line in session_path.read_text().splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = e.get('type')
            if kind == 'assistant':
                run.steps += 1
                st = e.get('stats', {})
                run.prompt_tokens += st.get('prompt_tokens', 0)
                run.generated += st.get('generated', 0)
                if e.get('answer'):
                    run.answers.append(e['answer'])
                for call in e.get('tool_calls', []):
                    run.calls += 1
                    names[str(call.get('id'))] = call.get('name')
                    run.tools.append(call.get('name'))
            elif kind == 'tool_result':
                if e.get('is_error'):
                    run.errors += 1
                    if names.get(e.get('call')) in ('edit_file', 'write_file'):
                        run.edit_errors += 1
    run.answer = run.answers[-1] if run.answers else ''
    return run


BEHAVIOURS = ('regex_grep', 'pytest', 'reread', 'cd')


def behaviours(session_path: Path) -> dict:
    """Habits worth counting per run, read from the session: a regex handed to
    the literal grep, a guessed test runner, a file re-read right after its
    own edit, and a `cd` before a command."""
    counts = dict.fromkeys(BEHAVIOURS, 0)
    if not session_path.exists():
        return counts
    last_edit = None
    for line in session_path.read_text().splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get('type') != 'assistant':
            continue
        for call in e.get('tool_calls', []):
            try:
                a = json.loads(call.get('arguments') or '{}')
            except json.JSONDecodeError:
                a = {}
            name = call.get('name')
            if name == 'grep' and re.search(r'\\\||\.\*|\[\^?\w|\\[dsw]|\|', str(a.get('pattern', ''))):
                counts['regex_grep'] += 1
            if name == 'bash':
                cmd = str(a.get('command', ''))
                if 'pytest' in cmd:
                    counts['pytest'] += 1
                if cmd.lstrip().startswith('cd '):
                    counts['cd'] += 1
            if name in ('edit_file', 'write_file'):
                last_edit = a.get('path')
            elif name == 'read_file':
                if last_edit and a.get('path') == last_edit:
                    counts['reread'] += 1
                last_edit = None
            else:
                last_edit = None
    return counts


def run_task(task: Task, seed: int, args, variant_dir: Path) -> dict:
    ws = Path(args.workspace)
    reset(ws)
    if args.instructions and args.instructions != 'none':
        # Untracked, so the reset before and after the turn removes it.
        shutil.copy(args.instructions, ws / 'AGENTS.md')
    if task.setup:
        task.setup(ws)
    session = variant_dir / f'{task.name}-s{seed}.session.jsonl'
    if session.exists():
        session.unlink()
    cmd = [str(Path(args.binary).resolve()), 'agent', '-p', task.prompt, '--json', '--seed', str(seed), '--session', str(session)]
    if args.model:
        cmd += ['--model', args.model]
    if args.system_prompt:
        cmd += ['--system-prompt', str(Path(args.system_prompt).resolve())]
    cmd += args.extra
    started = time.monotonic()
    try:
        proc = subprocess.run(cmd, cwd=ws, capture_output=True, text=True, timeout=TURN_TIMEOUT)
        events, stderr, status = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as exc:
        events, stderr, status = (exc.stdout or b'').decode() if isinstance(exc.stdout, bytes) else (exc.stdout or ''), 'timeout', -1
    run = parse(events, session, Run(wall_seconds=time.monotonic() - started, exit_status=status))
    try:
        ok, note = task.check(ws, run)
    except Exception as exc:  # noqa: BLE001
        ok, note = False, f'check raised {exc!r}'
    record = {
        'task': task.name, 'seed': seed, 'variant': args.variant, 'ok': ok, 'note': note,
        'steps': run.steps, 'calls': run.calls, 'errors': run.errors, 'edit_errors': run.edit_errors,
        'prompt_tokens': run.prompt_tokens, 'generated': run.generated,
        'model_seconds': round(run.model_seconds, 1), 'wall_seconds': round(run.wall_seconds, 1),
        'answer_chars': len(run.answer), 'stop': run.stop, 'exit_status': run.exit_status,
        'tools': run.tools, 'answer': run.answer, 'stderr_tail': stderr[-400:],
        'behaviours': behaviours(session),
    }
    (variant_dir / f'{task.name}-s{seed}.json').write_text(json.dumps(record, indent=1) + '\n')
    reset(ws)
    return record


def load_variant(name: str) -> list:
    records = []
    for p in sorted((OUT_DIR / name).glob('*-s*.json')):
        if p.name.endswith('.session.jsonl'):
            continue
        r = json.loads(p.read_text())
        r['behaviours'] = behaviours(p.with_name(p.name[:-5] + '.session.jsonl'))
        records.append(r)
    return records


def aggregate(records: list) -> dict:
    if not records:
        return {}
    by_task = {}
    for r in records:
        by_task.setdefault(r['task'], []).append(r)
    return {
        'runs': len(records),
        'pass_rate': sum(r['ok'] for r in records) / len(records),
        'tasks_passed': sum(all(r['ok'] for r in rs) for rs in by_task.values()),
        'tasks': len(by_task),
        'steps': statistics.mean(r['steps'] for r in records),
        'errors': sum(r['errors'] for r in records),
        'edit_errors': sum(r['edit_errors'] for r in records),
        'prompt_tokens': statistics.mean(r['prompt_tokens'] for r in records),
        'generated': statistics.mean(r['generated'] for r in records),
        'model_seconds': statistics.mean(r['model_seconds'] for r in records),
        'answer_chars': statistics.mean(r['answer_chars'] for r in records),
        **{b: sum(r['behaviours'][b] for r in records) for b in BEHAVIOURS},
    }


def table(variant: str, records: list) -> str:
    lines = [f'{variant}: {len(records)} runs',
             '| task | seed | ok | steps | calls | err | edit err | prompt | gen | model s | chars | stop | note |',
             '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |']
    for r in records:
        lines.append(f"| {r['task']} | {r['seed']} | {'✓' if r['ok'] else '✗'} | {r['steps']} | {r['calls']} | {r['errors']} | {r['edit_errors']} | "
                     f"{r['prompt_tokens']} | {r['generated']} | {r['model_seconds']} | {r['answer_chars']} | {r['stop']} | {r['note']} |")
    a = aggregate(records)
    if a:
        lines.append(f"| **all** | | {a['pass_rate']*100:.0f} % ({a['tasks_passed']}/{a['tasks']} tasks on every seed) | {a['steps']:.2f} | | {a['errors']} | {a['edit_errors']} | "
                     f"{a['prompt_tokens']:.0f} | {a['generated']:.0f} | {a['model_seconds']:.1f} | {a['answer_chars']:.0f} | | |")
    return '\n'.join(lines)


def compare(names: list) -> str:
    cols = [(n, aggregate(load_variant(n))) for n in names]
    head = '| metric | ' + ' | '.join(n for n, _ in cols) + ' |'
    sep = '| --- | ' + ' | '.join('---:' for _ in cols) + ' |'
    rows = [head, sep]
    def fmt(key, f):
        return f'| {key} | ' + ' | '.join(f(a[key]) if a else '—' for _, a in cols) + ' |'
    rows.append(fmt('runs', lambda v: str(v)))
    rows.append('| passed | ' + ' | '.join(f"{a['pass_rate']*100:.0f} % ({a['tasks_passed']}/{a['tasks']})" if a else '—' for _, a in cols) + ' |')
    rows.append(fmt('steps', lambda v: f'{v:.2f}'))
    rows.append(fmt('errors', lambda v: str(v)))
    rows.append(fmt('edit_errors', lambda v: str(v)))
    rows.append(fmt('prompt_tokens', lambda v: f'{v:.0f}'))
    rows.append(fmt('generated', lambda v: f'{v:.0f}'))
    rows.append(fmt('model_seconds', lambda v: f'{v:.1f}'))
    rows.append(fmt('answer_chars', lambda v: f'{v:.0f}'))
    for b in BEHAVIOURS:
        rows.append(fmt(b, lambda v: str(v)))
    # Per-task pass marks across variants, seeds joined.
    rows.append('')
    rows.append('| task | ' + ' | '.join(n for n, _ in cols) + ' |')
    rows.append('| --- | ' + ' | '.join('---' for _ in cols) + ' |')
    per = {n: load_variant(n) for n in names}
    for t in TASKS:
        marks = []
        for n in names:
            rs = sorted((r for r in per[n] if r['task'] == t.name), key=lambda r: r['seed'])
            marks.append(''.join('✓' if r['ok'] else '✗' for r in rs) or '—')
        rows.append(f'| {t.name} | ' + ' | '.join(marks) + ' |')
    return '\n'.join(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--variant', default='baseline', help='name of the run; results go under .zig-cache/agent-eval/<variant>/')
    ap.add_argument('--system-prompt', help='a file whose text replaces the built-in system prompt (nuclis agent --system-prompt)')
    ap.add_argument('--instructions', default=str(DEFAULT_INSTRUCTIONS), help='a file copied to the playground as AGENTS.md before every turn (the project instructions section); `none` for no file')
    ap.add_argument('--seeds', default='1,2', help='comma-separated seeds; every task runs once per seed')
    ap.add_argument('--tasks', default='*', help='glob over task names')
    ap.add_argument('--workspace', default=str(DEFAULT_WORKSPACE))
    ap.add_argument('--binary', default=str(DEFAULT_BINARY))
    ap.add_argument('--model', help='--model for the agent; the configured default otherwise')
    ap.add_argument('--compare', nargs='+', metavar='VARIANT', help='render the aggregate table of these variants and exit')
    ap.add_argument('--list', action='store_true', help='print the task list and exit')
    ap.add_argument('extra', nargs='*', help='further nuclis agent flags after --')
    args = ap.parse_args()

    if args.list:
        for t in TASKS:
            print(f'{t.name:10} {t.prompt}')
        return
    if args.compare:
        print(compare(args.compare))
        return
    ws = Path(args.workspace)
    if not (ws / 'Makefile').exists():
        sys.exit(f'no playground at {ws}')
    if not Path(args.binary).exists():
        sys.exit(f'no binary at {args.binary}')
    variant_dir = OUT_DIR / args.variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(',') if s]
    chosen = [t for t in TASKS if glob.fnmatch.fnmatch(t.name, args.tasks)]
    rev = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    meta = {'variant': args.variant, 'rev': rev, 'binary': args.binary, 'system_prompt': args.system_prompt, 'instructions': args.instructions,
            'model': args.model, 'seeds': seeds, 'extra': args.extra, 'time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    if args.system_prompt:
        shutil.copy(args.system_prompt, variant_dir / 'system-prompt.txt')
    if args.instructions and args.instructions != 'none':
        shutil.copy(args.instructions, variant_dir / 'AGENTS.md')
    (variant_dir / 'meta.json').write_text(json.dumps(meta, indent=1) + '\n')
    records = []
    for seed in seeds:
        for task in chosen:
            r = run_task(task, seed, args, variant_dir)
            records.append(r)
            print(f"{args.variant} {task.name:10} s{seed} {'ok  ' if r['ok'] else 'FAIL'} steps={r['steps']} err={r['errors']} "
                  f"tok={r['prompt_tokens']}+{r['generated']} {r['model_seconds']}s {r['note']}", flush=True)
    print()
    print(table(args.variant, load_variant(args.variant)))


if __name__ == '__main__':
    main()
