#!/usr/bin/env python3
"""The gate runner: executes the model-specific checks that gates.json declares.

A gate is one command (argv with placeholders) plus a comparator: `exit`
(the command's status decides) or `trace` (its `{trace}` directory goes
through compare-generation.py against the gate's bounds). Gates carry a
tier (`verify`: Metal, minutes; `verify-long`: Metal long-context
perplexity, tens of minutes; `verify-cpu`: the CPU reference, hours) and the
source globs that make them relevant, so `--changed REV` selects by
`git diff`: the Metal tier's matches by default, another tier's with
`--tier` (the long tier runs when a change touches attention, the caches,
or a windowed schedule, the CPU tier when a change alters what the CPU
reference computes, and both before a release). Bounds live in the manifest
and nowhere else.
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
MANIFEST = ROOT / 'gates.json'
TIERS = ('verify', 'verify-long', 'verify-cpu')
COMPARATORS = ('exit', 'trace')
PLACEHOLDERS = ('{nuclis}', '{nuclis-cpu}', '{zig-metal}', '{zig-cpu}', '{model}', '{mtp}', '{mmproj}', '{trace}')
TRACE_ROOT = '.zig-cache/gates/trace'
CPU_PREFIX = '.zig-cache/gates/cpu'


# ---- pure functions (covered by --self-test) --------------------------------

def validate(doc):
    """Return a list of manifest problems; empty means valid."""
    problems = []
    build = doc.get('build', {})
    for key in ('metal_optimize', 'cpu_optimize', 'cache'):
        if not isinstance(build.get(key), str):
            problems.append(f'build.{key}: missing or not a string')
    models = doc.get('models', {})
    for name, entry in models.items():
        if not isinstance(entry, dict) or not isinstance(entry.get('path'), str):
            problems.append(f'models.{name}: needs a "path" string')
    seen = set()
    for i, gate in enumerate(doc.get('gates', [])):
        name = gate.get('name', f'#{i}')
        if not isinstance(gate.get('name'), str):
            problems.append(f'gate #{i}: missing name')
        if name in seen:
            problems.append(f'{name}: duplicate name')
        seen.add(name)
        if gate.get('tier') not in TIERS:
            problems.append(f'{name}: tier must be one of {TIERS}')
        for key in ('model', 'mtp'):
            if key in gate and gate[key] not in models:
                problems.append(f'{name}: {key} names an unknown model "{gate[key]}"')
        if 'model' not in gate:
            problems.append(f'{name}: missing model')
        paths = gate.get('paths')
        if not isinstance(paths, list) or not paths or not all(isinstance(p, str) for p in paths):
            problems.append(f'{name}: paths must be a non-empty list of globs')
        command = gate.get('command')
        if not isinstance(command, list) or not command or not all(isinstance(a, str) for a in command):
            problems.append(f'{name}: command must be a non-empty argv list')
        else:
            for arg in command:
                for token in re.findall(r'\{[a-z-]+\}', arg):
                    if token not in PLACEHOLDERS:
                        problems.append(f'{name}: unknown placeholder {token}')
            if '{mtp}' in ' '.join(command) and 'mtp' not in gate:
                problems.append(f'{name}: command uses {{mtp}} but the gate names no mtp')
            if '{mmproj}' in ' '.join(command) and 'mmproj' not in models.get(gate.get('model'), {}):
                problems.append(f'{name}: command uses {{mmproj}} but model "{gate.get("model")}" names no mmproj')
        comparator = gate.get('comparator', {})
        kind = comparator.get('kind')
        if kind not in COMPARATORS:
            problems.append(f'{name}: comparator.kind must be one of {COMPARATORS}')
        if kind == 'trace':
            if not isinstance(comparator.get('reference'), str):
                problems.append(f'{name}: trace comparator needs a reference directory')
            if not isinstance(comparator.get('positions'), int):
                problems.append(f'{name}: trace comparator needs integer positions')
            bounds = gate.get('bounds')
            if not isinstance(bounds, dict) or set(bounds) != {'max_absolute', 'max_relative_rms'}:
                problems.append(f'{name}: trace gates need bounds {{max_absolute, max_relative_rms}}')
            if command and '{trace}' not in ' '.join(command):
                problems.append(f'{name}: trace gates must write to {{trace}}')
        elif 'bounds' in gate:
            problems.append(f'{name}: an exit gate carries no bounds')
        if not isinstance(gate.get('evidence'), str):
            problems.append(f'{name}: missing evidence')
    return problems


def glob_to_regex(glob):
    """`**` spans directories, `*` stays inside one path component."""
    out = ''
    i = 0
    while i < len(glob):
        if glob.startswith('**/', i):
            out += '(?:.*/)?'
            i += 3
        elif glob.startswith('**', i):
            out += '.*'
            i += 2
        elif glob[i] == '*':
            out += '[^/]*'
            i += 1
        else:
            out += re.escape(glob[i])
            i += 1
    return re.compile('^' + out + '$')


def matches(glob, path):
    return glob_to_regex(glob).match(path) is not None


def select(gates, changed):
    """The gates whose paths match any changed file, in manifest order."""
    return [g for g in gates if any(matches(p, f) for p in g['paths'] for f in changed)]


def expand(argv, gate, doc, env=None):
    """Substitute placeholders; the build placeholders expand to several argv items."""
    env = os.environ if env is None else env
    build = doc['build']
    zig_common = ['--global-cache-dir', build['cache']]
    table = {
        '{nuclis}': ['./zig-out/bin/nuclis'],
        '{nuclis-cpu}': [f'./{CPU_PREFIX}/bin/nuclis'],
        '{zig-metal}': ['zig', 'build', '-Dmetal=true', f"-Doptimize={build['metal_optimize']}"] + zig_common,
        '{zig-cpu}': ['zig', 'build', f"-Doptimize={build['cpu_optimize']}"] + zig_common,
    }
    scalars = {'{trace}': f"{TRACE_ROOT}/{gate['name']}"}
    for key in ('model', 'mtp'):
        if key in gate:
            scalars['{' + key + '}'] = model_path(doc, gate[key], env)
    entry = doc['models'].get(gate.get('model'), {})
    if 'mmproj' in entry:
        scalars['{mmproj}'] = os.path.expanduser(entry['mmproj'])
    out = []
    for arg in argv:
        if arg in table:
            out.extend(table[arg])
            continue
        for token, value in scalars.items():
            arg = arg.replace(token, value)
        out.append(arg)
    return out


def model_path(doc, key, env=None):
    """The pinned path, or the `<KEY>_MODEL` environment override; `~` expanded."""
    env = os.environ if env is None else env
    override = env.get(key.upper() + '_MODEL')
    path = override if override else doc['models'][key]['path']
    return os.path.expanduser(path)


def builds_needed(gates):
    """Which binaries the selected gates run directly (the zig placeholders build themselves)."""
    joined = [' '.join(g['command']) for g in gates]
    return {'metal': any('{nuclis}' in c for c in joined), 'cpu': any('{nuclis-cpu}' in c for c in joined)}


# ---- execution ----------------------------------------------------------------

def load():
    doc = json.loads(MANIFEST.read_text())
    problems = validate(doc)
    if problems:
        for p in problems:
            print(f'gates.json: {p}', file=sys.stderr)
        sys.exit(2)
    return doc


def changed_files(rev):
    diff = subprocess.run(['git', 'diff', '--name-only', rev], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    untracked = subprocess.run(['git', 'ls-files', '--others', '--exclude-standard'], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return sorted(set(diff.split()) | set(untracked.split()))


def build(doc, which, dry_run):
    b = doc['build']
    if which == 'metal':
        cmd = ['zig', 'build', '-Dmetal=true', f"-Doptimize={b['metal_optimize']}", '--global-cache-dir', b['cache']]
    else:
        cmd = ['zig', 'build', '-Dmetal=true', f"-Doptimize={b['cpu_optimize']}", '--global-cache-dir', b['cache'], '--prefix', CPU_PREFIX]
    print('build:', ' '.join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=ROOT, check=True)


def run_gate(gate, doc, dry_run):
    argv = expand(gate['command'], gate, doc)
    trace = ROOT / TRACE_ROOT / gate['name']
    result = {'name': gate['name'], 'tier': gate['tier'], 'command': argv, 'passed': None, 'seconds': 0.0}
    if dry_run:
        print(f"{gate['name']}: {' '.join(argv)}")
        if gate['comparator']['kind'] == 'trace':
            print(f"  then compare-generation.py {trace.relative_to(ROOT)} {gate['comparator']['reference']} "
                  f"--max-absolute {gate['bounds']['max_absolute']} --max-relative-rms {gate['bounds']['max_relative_rms']}")
        return result
    shutil.rmtree(trace, ignore_errors=True)
    trace.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    proc = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
    result['seconds'] = time.monotonic() - started
    result['exit_status'] = proc.returncode
    tail = '\n'.join((proc.stderr or proc.stdout).strip().splitlines()[-20:])
    if proc.returncode != 0:
        result['passed'] = False
        result['detail'] = tail
        return result
    comparator = gate['comparator']
    if comparator['kind'] == 'exit':
        result['passed'] = True
        result['detail'] = tail.splitlines()[-1] if tail else ''
        return result
    cmd = [sys.executable, 'scripts/compare-generation.py', str(trace), comparator['reference'],
           '--positions', str(comparator['positions']),
           '--max-absolute', str(gate['bounds']['max_absolute']), '--max-relative-rms', str(gate['bounds']['max_relative_rms'])]
    for key in ('embedding', 'layers', 'vocab'):
        if key in comparator:
            cmd += ['--' + key, str(comparator[key])]
    if comparator.get('draft'):
        cmd.append('--draft')
    cmp = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    result['seconds'] = time.monotonic() - started
    try:
        report = json.loads(cmp.stdout)
    except json.JSONDecodeError:
        result['passed'] = False
        result['detail'] = (cmp.stderr or cmp.stdout).strip()[-2000:]
        return result
    rows = report['comparisons']
    result['passed'] = bool(report['passed'])
    result['measured'] = {'max_absolute': max(r['max_absolute'] for r in rows),
                          'max_relative_rms': max(r['relative_rms'] for r in rows), 'files': len(rows)}
    if 'native_greedy' in report:
        result['measured']['greedy'] = report['native_greedy']
        result['measured']['reference_greedy'] = report['reference_greedy']
    return result


def print_result(r, gate):
    status = 'PASS' if r['passed'] else 'FAIL'
    line = f"{status} {r['name']:<34} {r['seconds']:7.1f} s"
    if 'measured' in r:
        m, b = r['measured'], gate['bounds']
        line += (f"  max abs {m['max_absolute']:.3e} ≤ {b['max_absolute']:g}"
                 f"  rel rms {m['max_relative_rms']:.3e} ≤ {b['max_relative_rms']:g}  ({m['files']} files)")
        if 'greedy' in m:
            line += f"  greedy {m['greedy']}" + ('' if m['greedy'] == m['reference_greedy'] else f" ≠ {m['reference_greedy']}")
    elif r.get('detail'):
        line += f"  {r['detail'] if r['passed'] else ''}"
    print(line, flush=True)
    if not r['passed'] and r.get('detail'):
        for l in r['detail'].splitlines():
            print('    ' + l)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--list', action='store_true', help='print every gate with its tier, family, model, and evidence')
    ap.add_argument('--validate', action='store_true', help='validate the manifest and run the self-test; no model needed')
    ap.add_argument('--self-test', action='store_true', help='run the unit tests of the pure functions')
    ap.add_argument('--tier', choices=TIERS, help='run every gate of a tier; with --changed, the selected gates of that tier')
    ap.add_argument('--gate', action='append', metavar='NAME', help="run gates by name or glob (repeatable): gemma4-qat-trace-f16, 'muse-*'")
    ap.add_argument('--changed', nargs='?', const='HEAD', metavar='REV',
                    help='run the Metal-tier gates whose paths match `git diff --name-only REV` plus untracked files (default HEAD); the other tiers\' matches are listed, and run with --tier verify-long or verify-cpu')
    ap.add_argument('--dry-run', action='store_true', help='print the commands instead of running them')
    ap.add_argument('--no-build', action='store_true', help='do not rebuild the binaries first')
    ap.add_argument('--json', action='store_true', help='print the results as JSON')
    args = ap.parse_args()
    os.chdir(ROOT)

    if args.self_test:
        sys.exit(0 if self_test() else 1)
    doc = load()
    if args.validate:
        print(f"gates.json: {len(doc['gates'])} gates, {len(doc['models'])} models, valid")
        sys.exit(0 if self_test() else 1)
    if args.list:
        for g in doc['gates']:
            print(f"{g['name']:<34} {g['tier']:<10} {g['family']:<8} {g['model']:<15} {g['evidence']}")
        return

    if args.gate:
        selected = []
        for pattern in args.gate:
            hits = [g for g in doc['gates'] if fnmatch.fnmatchcase(g['name'], pattern)]
            if not hits:
                sys.exit(f"no gate matches {pattern!r}; --list shows them")
            selected.extend(g for g in hits if g not in selected)
    elif args.changed is not None:
        files = changed_files(args.changed)
        matched = select(doc['gates'], files)
        tier = args.tier or 'verify'
        selected = [g for g in matched if g['tier'] == tier]
        others = [g for g in matched if g['tier'] != tier]
        print(f"{len(files)} changed file(s) since {args.changed}; {len(selected)} {tier} gate(s) selected: "
              + (', '.join(g['name'] for g in selected) or 'none'), flush=True)
        if tier == 'verify':
            long = [g['name'] for g in others if g['tier'] == 'verify-long']
            cpu = [g['name'] for g in others if g['tier'] == 'verify-cpu']
            if long:
                print("Long-tier gates matched, not run (they run when the change touches attention, the caches, "
                      "or a windowed schedule, and before a release; `--tier verify-long` runs them): " + ', '.join(long), flush=True)
            if cpu:
                print("CPU-tier gates matched, not run (they run when the change alters what the CPU reference "
                      "computes, and before a release; `--tier verify-cpu` runs them): " + ', '.join(cpu), flush=True)
        if not selected:
            return
    elif args.tier:
        selected = [g for g in doc['gates'] if g['tier'] == args.tier]
    else:
        ap.error('one of --list, --validate, --tier, --gate, --changed is required')

    if not args.no_build:
        need = builds_needed(selected)
        if need['metal']:
            build(doc, 'metal', args.dry_run)
        if need['cpu']:
            build(doc, 'cpu', args.dry_run)
    results = []
    for gate in selected:
        r = run_gate(gate, doc, args.dry_run)
        results.append(r)
        if not args.dry_run and not args.json:
            print_result(r, gate)
    if args.dry_run:
        return
    if args.json:
        print(json.dumps({'revision': git_rev(), 'results': results}, indent=2))
    failed = [r['name'] for r in results if not r['passed']]
    total = sum(r['seconds'] for r in results)
    print(f"{len(results) - len(failed)}/{len(results)} passed in {total:.0f} s" + (f"; failed: {', '.join(failed)}" if failed else ''), file=sys.stderr)
    sys.exit(1 if failed else 0)


def git_rev():
    return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()


# ---- self-test ----------------------------------------------------------------

def sample_doc():
    return {
        'build': {'metal_optimize': 'ReleaseSafe', 'cpu_optimize': 'ReleaseFast', 'cache': '.zig-cache/global'},
        'models': {'a': {'path': '~/m/a.gguf'}, 'a_mtp': {'path': '~/m/a-mtp.gguf'}},
        'gates': [
            {'name': 'a-trace', 'family': 'a', 'tier': 'verify', 'model': 'a', 'paths': ['inference/src/models/a*.zig', 'inference/src/quant/**'],
             'command': ['{nuclis}', 'generate', '--model', '{model}', '--trace-dir', '{trace}'],
             'comparator': {'kind': 'trace', 'reference': 'tests/fixtures/a', 'positions': 2},
             'bounds': {'max_absolute': 0.002, 'max_relative_rms': 0.0001}, 'evidence': 'docs/a.md'},
            {'name': 'a-draft-cpu', 'family': 'a', 'tier': 'verify-cpu', 'model': 'a', 'mtp': 'a_mtp', 'paths': ['inference/src/backends/cpu/**'],
             'command': ['{zig-cpu}', 'test-generation', '--', '{model}', '--draft-model', '{mtp}'],
             'comparator': {'kind': 'exit'}, 'evidence': 'docs/a.md'},
            {'name': 'a-perplexity-4k', 'family': 'a', 'tier': 'verify-long', 'model': 'a', 'paths': ['inference/src/backends/metal/**'],
             'command': ['{nuclis}', 'eval', '--model', '{model}', '--file', 't.raw', '--reference', 'r.json'],
             'comparator': {'kind': 'exit'}, 'evidence': 'docs/a.md'},
        ],
    }


class SelfTest(unittest.TestCase):
    def test_sample_is_valid(self):
        self.assertEqual(validate(sample_doc()), [])

    def test_validation_names_each_problem(self):
        doc = sample_doc()
        doc['gates'].append(dict(doc['gates'][0]))  # duplicate name
        doc['gates'][1]['tier'] = 'cheap'
        doc['gates'][1]['model'] = 'nope'
        doc['gates'][0]['command'] = ['{bogus}']
        doc['gates'][0]['comparator'] = {'kind': 'report'}
        problems = validate(doc)
        for needle in ('duplicate name', 'tier must be', 'unknown model "nope"', 'unknown placeholder {bogus}', 'comparator.kind'):
            self.assertTrue(any(needle in p for p in problems), (needle, problems))

    def test_exit_gate_refuses_bounds_and_trace_gate_needs_them(self):
        doc = sample_doc()
        doc['gates'][1]['bounds'] = {'max_absolute': 1, 'max_relative_rms': 1}
        del doc['gates'][0]['bounds']
        problems = validate(doc)
        self.assertTrue(any('carries no bounds' in p for p in problems))
        self.assertTrue(any('need bounds' in p for p in problems))

    def test_globs(self):
        self.assertTrue(matches('inference/src/quant/**', 'inference/src/quant/q4k.zig'))
        self.assertTrue(matches('inference/src/quant/**', 'inference/src/quant/fixtures/x.json'))
        self.assertTrue(matches('inference/src/models/a*.zig', 'inference/src/models/a_metal.zig'))
        self.assertFalse(matches('inference/src/models/a*.zig', 'inference/src/models/b.zig'))
        self.assertFalse(matches('inference/src/models/a*.zig', 'inference/src/models/fixtures/a.zig'))
        self.assertFalse(matches('inference/src/quant/**', 'src/quant/x.zig'))

    def test_select_by_changed_paths(self):
        gates = sample_doc()['gates']
        self.assertEqual([g['name'] for g in select(gates, ['src/tui/editor.zig'])], [])
        self.assertEqual([g['name'] for g in select(gates, ['inference/src/models/a_runtime.zig'])], ['a-trace'])
        self.assertEqual([g['name'] for g in select(gates, ['inference/src/backends/cpu/vector.zig', 'inference/src/quant/q4.zig'])],
                         ['a-trace', 'a-draft-cpu'])

    def test_expand_placeholders(self):
        doc = sample_doc()
        argv = expand(doc['gates'][1]['command'], doc['gates'][1], doc, env={'HOME': '/h'})
        self.assertEqual(argv[:4], ['zig', 'build', '-Doptimize=ReleaseFast', '--global-cache-dir'])
        self.assertEqual(argv[-3:], [os.path.expanduser('~/m/a.gguf'), '--draft-model', os.path.expanduser('~/m/a-mtp.gguf')])
        argv = expand(doc['gates'][0]['command'], doc['gates'][0], doc, env={'A_MODEL': '/elsewhere/a.gguf'})
        self.assertEqual(argv, ['./zig-out/bin/nuclis', 'generate', '--model', '/elsewhere/a.gguf', '--trace-dir', f'{TRACE_ROOT}/a-trace'])

    def test_builds_needed(self):
        gates = sample_doc()['gates']
        self.assertEqual(builds_needed(gates), {'metal': True, 'cpu': False})
        self.assertEqual(builds_needed([gates[1]]), {'metal': False, 'cpu': False})


def self_test():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SelfTest)
    return unittest.TextTestRunner(verbosity=0).run(suite).wasSuccessful()


if __name__ == '__main__':
    main()
