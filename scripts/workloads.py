#!/usr/bin/env python3
"""The workload driver: runs the benchmark workloads that workloads.json declares.

A `bench` workload resolves into one `nuclis bench` command (model from
gates.json, prompt array or raw text, budget, context, cache precision,
sampling, the off/on pair with `--speculative on`, and an optional
no-drafter baseline run) and saves the JSON report under
`.zig-cache/bench/<workload>/<rev>-<n>.json`; an `acceptance` workload
delegates to scripts/nuclis-baseline.py, which writes the dated record under
docs/benchmarks/. Tables come from scripts/bench-report.py.
See docs/development.md § The record.
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
import gates  # noqa: E402  (model_path and the gates manifest)

MANIFEST = ROOT / 'workloads.json'
SAVE_ROOT = '.zig-cache/bench'
KINDS = ('bench', 'acceptance')
SAMPLING_FLAGS = {'temperature': '--temperature', 'top_k': '--top-k', 'top_p': '--top-p', 'min_p': '--min-p',
                  'presence_penalty': '--presence-penalty', 'repetition_penalty': '--repetition-penalty', 'seed': '--seed'}
BAR_METRICS = ('decode_tokens_per_second', 'prefill_tokens_per_second', 'decode_speedup')


# ---- pure functions (covered by --self-test) --------------------------------

def validate(doc, models):
    problems = []
    for name, w in doc.get('workloads', {}).items():
        if '/' not in name or name != name.lower():
            problems.append(f'{name}: names are lower-case <entry>/<workload>')
        kind = w.get('kind')
        if kind not in KINDS:
            problems.append(f'{name}: kind must be one of {KINDS}')
        if w.get('model') not in models:
            problems.append(f'{name}: model names an unknown gates.json key "{w.get("model")}"')
        if not isinstance(w.get('evidence'), str):
            problems.append(f'{name}: missing evidence')
        if kind == 'bench':
            if ('prompt' in w) == ('prompt_tokens' in w):
                problems.append(f'{name}: exactly one of prompt, prompt_tokens')
            for key in ('max_tokens', 'ctx', 'warmup', 'repeat'):
                if not isinstance(w.get(key), int) or w[key] < 0:
                    problems.append(f'{name}: {key} must be a non-negative integer')
            if w.get('kv') not in ('f16', 'f32'):
                problems.append(f'{name}: kv must be f16 or f32')
            if w.get('pair') and not isinstance(w.get('draft_length'), int):
                problems.append(f'{name}: a pair needs draft_length')
            if w.get('baseline') and not w.get('pair'):
                problems.append(f'{name}: baseline is only meaningful for a pair')
            for key in w.get('sampling', {}):
                if key not in SAMPLING_FLAGS:
                    problems.append(f'{name}: unknown sampling option {key}')
            for metric, bar in w.get('bars', {}).items():
                if metric not in BAR_METRICS or not isinstance(bar, dict) or not (set(bar) <= {'min', 'max'}) or not bar:
                    problems.append(f'{name}: bars.{metric} must be {{min|max}} on one of {BAR_METRICS}')
        elif kind == 'acceptance':
            for key in ('run', 'reference_records', 'record_suffix'):
                if not isinstance(w.get(key), str):
                    problems.append(f'{name}: acceptance needs {key}')
    return problems


def bench_argv(w, model_path, nuclis='./zig-out/bin/nuclis', speculative=None):
    """The `nuclis bench` command for a bench workload; `speculative` overrides the pair's switch."""
    argv = [nuclis, 'bench', '--backend', 'metal', '--model', model_path]
    if 'prompt_tokens' in w:
        argv += ['--prompt-tokens', w['prompt_tokens']]
    else:
        argv += ['--prompt', w['prompt']]
        if w.get('raw'):
            argv.append('--raw')
    argv += ['--max-tokens', str(w['max_tokens']), '--ctx-size', str(w['ctx']), '--kv', w['kv'],
             '--warmup', str(w['warmup']), '--repeat', str(w['repeat'])]
    for key, value in w.get('sampling', {}).items():
        argv += [SAMPLING_FLAGS[key], str(value)]
    on = w.get('pair', False) if speculative is None else speculative
    argv += ['--speculative', 'on' if on else 'off']
    if on:
        argv += ['--draft-length', str(w['draft_length'])]
    argv.append('--json')
    return argv


def acceptance_argv(w, model_path, nuclis='./zig-out/bin/nuclis', date='DATE'):
    argv = [sys.executable, 'scripts/nuclis-baseline.py', '--model', model_path, '--nuclis', nuclis, '--run', w['run'],
            '--reference-records', w['reference_records']]
    if w['record_suffix']:
        argv += ['--output', f"docs/benchmarks/nuclis-{date}-{w['record_suffix']}.json"]
    return argv


def select(names, patterns):
    out = []
    for pattern in patterns:
        hits = [n for n in names if fnmatch.fnmatchcase(n, pattern)]
        if not hits:
            raise KeyError(pattern)
        out.extend(n for n in hits if n not in out)
    return out


def next_path(directory, rev, suffix=''):
    """`<rev>-<n><suffix>.json`, n one past the highest sequence already saved for this revision."""
    n = 0
    for p in Path(directory).glob(f'{rev}-*.json'):
        stem = p.name[len(rev) + 1:-5]
        digits = stem.split('-')[0]
        if digits.isdigit():
            n = max(n, int(digits))
    return Path(directory) / f'{rev}-{n + 1}{suffix}.json'


# ---- execution ----------------------------------------------------------------

def load():
    doc = json.loads(MANIFEST.read_text())
    gdoc = json.loads(gates.MANIFEST.read_text())
    problems = validate(doc, gdoc['models'])
    if problems:
        for p in problems:
            print(f'workloads.json: {p}', file=sys.stderr)
        sys.exit(2)
    return doc, gdoc


def git_rev():
    return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def model_for(w, gdoc, env=None):
    """The pinned path, except that a pair names the catalogue entry so the
    companion (`mtp`) resolves; an explicit `<KEY>_MODEL` override wins either way."""
    env = os.environ if env is None else env
    if w.get('pair') and not env.get(w['model'].upper() + '_MODEL'):
        return gdoc['models'][w['model']].get('entry') or gates.model_path(gdoc, w['model'], env)
    return gates.model_path(gdoc, w['model'], env)


def run_workload(name, w, gdoc, args):
    model = model_for(w, gdoc)
    save_dir = ROOT / args.save_dir / name
    rev = git_rev()
    if w['kind'] == 'acceptance':
        argv = acceptance_argv(w, model, date=time.strftime('%Y-%m-%d'))
        print(f"{name}: {' '.join(argv)}", flush=True)
        if args.dry_run:
            return True
        return subprocess.run(argv, cwd=ROOT).returncode == 0
    runs = [('', bench_argv(w, model))]
    if w.get('baseline'):
        runs.append(('-baseline', bench_argv(w, model, speculative=False)))
    ok = True
    for suffix, argv in runs:
        print(f"{name}{suffix}: {' '.join(argv)}", flush=True)
        if args.dry_run:
            continue
        save_dir.mkdir(parents=True, exist_ok=True)
        out = next_path(save_dir, rev, suffix)
        started = time.monotonic()
        proc = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
        seconds = time.monotonic() - started
        if proc.returncode != 0:
            print(f"FAIL {name}{suffix} ({seconds:.0f} s)\n" + '\n'.join(proc.stderr.strip().splitlines()[-10:]), flush=True)
            ok = False
            continue
        out.write_text(proc.stdout)
        report = json.loads(proc.stdout)
        line = f"saved {out.relative_to(ROOT)} ({seconds:.0f} s): decode {fmt(report.get('mean_decode_tokens_per_second'))} tok/s, prefill {fmt(report.get('mean_prefill_tokens_per_second'))} tok/s"
        if report.get('speculative_draft_length'):
            line += f", speculative {fmt(report.get('mean_speculative_decode_tokens_per_second'))} tok/s, {fmt(report.get('decode_speedup'))}×"
        print(line, flush=True)
    return ok


def fmt(v):
    return '—' if v is None else f'{v:.2f}'


def model_status(path):
    return 'present' if Path(path).is_file() else 'absent'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--list', action='store_true', help='every workload with its kind, model and status, shape, and evidence')
    ap.add_argument('--validate', action='store_true', help='validate the manifest and run the self-test; no model needed')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--run', action='append', metavar='NAME', help="workloads by name or glob (repeatable): gemma4-qat/prose512-draft, 'qwen38/spec/*'")
    ap.add_argument('--save-dir', default=SAVE_ROOT, help=f'where bench reports go (default {SAVE_ROOT}/<workload>/)')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--no-build', action='store_true')
    args = ap.parse_args()
    os.chdir(ROOT)
    if args.self_test:
        sys.exit(0 if self_test() else 1)
    doc, gdoc = load()
    if args.validate:
        print(f"workloads.json: {len(doc['workloads'])} workloads, valid")
        sys.exit(0 if self_test() else 1)
    if args.list:
        for name, w in doc['workloads'].items():
            model = gates.model_path(gdoc, w['model'])
            shape = (f"{w['max_tokens']} out, ctx {w['ctx']}, {w['kv']}, {w['warmup']}+{w['repeat']}" +
                     (f", pair d{w['draft_length']}" if w.get('pair') else '') + (', baseline' if w.get('baseline') else '') +
                     (', sampled' if w.get('sampling') else '')) if w['kind'] == 'bench' else f"acceptance: {w['run']}"
            print(f"{name:<34} {w['model']:<15} {model_status(model):<8} {shape:<44} {w['evidence']}")
        return
    if not args.run:
        ap.error('one of --list, --validate, --run is required')
    try:
        names = select(list(doc['workloads']), args.run)
    except KeyError as e:
        sys.exit(f'no workload matches {e.args[0]!r}; --list shows them')
    if not args.no_build and not args.dry_run:
        gates.build(gdoc, 'metal', False)
    failed = [n for n in names if not run_workload(n, doc['workloads'][n], gdoc, args)]
    if failed:
        sys.exit('failed: ' + ', '.join(failed))


# ---- self-test ----------------------------------------------------------------

def sample():
    return {'workloads': {
        'a/prose': {'kind': 'bench', 'model': 'a', 'prompt_tokens': 'tests/p.json', 'max_tokens': 8, 'ctx': 64, 'kv': 'f16',
                    'warmup': 1, 'repeat': 2, 'evidence': 'docs/x.md'},
        'a/code-draft': {'kind': 'bench', 'model': 'a', 'prompt': 'hi', 'raw': True, 'max_tokens': 8, 'ctx': 64, 'kv': 'f16',
                         'warmup': 0, 'repeat': 1, 'pair': True, 'draft_length': 4, 'baseline': True,
                         'sampling': {'temperature': 0.7, 'top_k': 20}, 'bars': {'decode_speedup': {'min': 1.0}}, 'evidence': 'docs/x.md'},
        'a/acceptance': {'kind': 'acceptance', 'model': 'a', 'run': 'run-x', 'reference_records': 'r.json', 'record_suffix': 'a', 'evidence': 'docs/x.md'},
    }}


class SelfTest(unittest.TestCase):
    def test_sample_valid(self):
        self.assertEqual(validate(sample(), {'a': {}}), [])

    def test_validation_names_problems(self):
        doc = sample()
        doc['workloads']['a/prose']['prompt'] = 'both'
        doc['workloads']['a/code-draft']['bars'] = {'nope': {'min': 1}}
        doc['workloads']['B/x'] = {'kind': 'bench', 'model': 'zz'}
        problems = validate(doc, {'a': {}})
        for needle in ('exactly one of prompt', 'bars.nope', 'lower-case', 'unknown gates.json key "zz"'):
            self.assertTrue(any(needle in p for p in problems), (needle, problems))

    def test_bench_argv(self):
        w = sample()['workloads']['a/code-draft']
        argv = bench_argv(w, '/m.gguf')
        self.assertEqual(argv[:6], ['./zig-out/bin/nuclis', 'bench', '--backend', 'metal', '--model', '/m.gguf'])
        self.assertIn('--raw', argv)
        self.assertEqual(argv[argv.index('--speculative') + 1], 'on')
        self.assertEqual(argv[argv.index('--draft-length') + 1], '4')
        self.assertEqual(argv[argv.index('--temperature') + 1], '0.7')
        off = bench_argv(w, '/m.gguf', speculative=False)
        self.assertEqual(off[off.index('--speculative') + 1], 'off')
        self.assertNotIn('--draft-length', off)
        plain = bench_argv(sample()['workloads']['a/prose'], '/m.gguf')
        self.assertEqual(plain[plain.index('--prompt-tokens') + 1], 'tests/p.json')
        self.assertEqual(plain[-3:], ['--speculative', 'off', '--json'])

    def test_model_for(self):
        gdoc = {'models': {'a': {'path': '~/m/a.gguf', 'entry': 'a-entry'}}}
        ws = sample()['workloads']
        self.assertEqual(model_for(ws['a/code-draft'], gdoc, env={}), 'a-entry')
        self.assertEqual(model_for(ws['a/prose'], gdoc, env={}), os.path.expanduser('~/m/a.gguf'))
        self.assertEqual(model_for(ws['a/code-draft'], gdoc, env={'A_MODEL': '/x.gguf'}), '/x.gguf')

    def test_acceptance_argv(self):
        argv = acceptance_argv(sample()['workloads']['a/acceptance'], '/m.gguf', date='2026-01-02')
        self.assertEqual(argv[-1], 'docs/benchmarks/nuclis-2026-01-02-a.json')
        self.assertEqual(argv[argv.index('--run') + 1], 'run-x')

    def test_select_and_next_path(self):
        names = ['a/prose', 'a/code-draft', 'b/prose']
        self.assertEqual(select(names, ['a/*']), ['a/prose', 'a/code-draft'])
        self.assertEqual(select(names, ['*/prose', 'a/prose']), ['a/prose', 'b/prose'])
        with self.assertRaises(KeyError):
            select(names, ['c/*'])
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(next_path(d, 'abc1234').name, 'abc1234-1.json')
            (Path(d) / 'abc1234-1.json').write_text('{}')
            (Path(d) / 'abc1234-1-baseline.json').write_text('{}')
            (Path(d) / 'abc1234-3.json').write_text('{}')
            self.assertEqual(next_path(d, 'abc1234', '-baseline').name, 'abc1234-4-baseline.json')
            self.assertEqual(next_path(d, 'zzz9999').name, 'zzz9999-1.json')


def self_test():
    return unittest.TextTestRunner(verbosity=0).run(unittest.defaultTestLoader.loadTestsFromTestCase(SelfTest)).wasSuccessful()


if __name__ == '__main__':
    main()
