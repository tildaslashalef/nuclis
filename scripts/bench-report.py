#!/usr/bin/env python3
"""Tables and comparisons from saved `nuclis bench --json` reports.

`--table FILE...` renders one markdown row per report: the acceptance form
(prefill, decode, first token, session) for plain runs, the record form
(per-batch costs, off/on rates, the no-drafter baseline when a `-baseline`
sibling exists) for off/on pairs. `--write-doc DOC --name NAME` replaces the
region between `<!-- bench:NAME -->` and `<!-- /bench:NAME -->` in a
document with that table and its provenance line, so record tables are
generated, never transcribed. `--compare PREV CUR` prints the deltas, and
`--check WORKLOAD FILE` tests a report against the workload's `bars`
(exit 1 on a miss). See docs/development.md § The record.
"""
import argparse
import json
import re
import statistics
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAIR_HEADER = ('| report | prompt | sampling | draft | accepted/step | proposed/step | drafts/accepted | tokens/batch | propose ms | verify ms | '
               'accept ms | recover ms | checkpoint ms | commit ms | prefill off → on (s) | decode baseline → off → on tok/s | speedup |\n'
               '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
PLAIN_HEADER = ('| report | prompt | sampling | runs | prefill tok/s | decode tok/s | first token s | session MiB | load ms |\n'
                '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |')


# ---- pure functions (covered by --self-test) --------------------------------

def measured(report):
    return [s for s in report['samples'] if not s['warmup'] and s['stop_reason'] != 'cancelled']


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def mean_sd(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else None)


def fmt(v, digits=2):
    return '—' if v is None else f'{v:.{digits}f}'


def fmt_sd(pair, digits=2):
    if pair is None:
        return '—'
    m, sd = pair
    return f'{m:.{digits}f}' + (f' ± {sd:.{digits}f}' if sd is not None else '')


def is_pair(report):
    return any(s['speculative'] for s in measured(report))


def sampling_label(report):
    return report.get('sampling', 'greedy').split(':')[0]


def plain_row(label, report):
    rows = measured(report)
    return (f"| {label} | {rows[0]['prompt_tokens']:,} | {sampling_label(report)} | {len(rows)} | "
            f"{fmt_sd(mean_sd([s['prefill_tokens_per_second'] for s in rows]))} | "
            f"{fmt_sd(mean_sd([s['decode_tokens_per_second'] for s in rows]))} | "
            f"{fmt(mean([s['first_token_milliseconds'] for s in rows]) / 1000 if rows else None, 1)} | "
            f"{report.get('session_bytes', 0) / 2**20:.0f} | {report.get('load_milliseconds', 0):.0f} |") if rows else f'| {label} | — | — | 0 | — | — | — | — | — |'


def pair_row(label, report, baseline=None):
    rows = measured(report)
    off = [s for s in rows if not s['speculative']]
    on = [s for s in rows if s['speculative']]
    if not on or not off:
        return f'| {label} | — | {sampling_label(report)} | — | (no complete pair) |'

    def per_batch(key):
        return mean([s[key] / s['speculative_steps'] for s in on if s.get(key) is not None and s.get('speculative_steps')])
    accepted = mean([s['accepted_per_step'] for s in on])
    proposed = mean([s.get('proposed_per_step') for s in on])
    per_token = proposed / accepted if (proposed is not None and accepted) else None
    tokens_per_batch = mean([(s['generated_tokens'] - 1) / s['speculative_steps'] for s in on if s.get('speculative_steps')])
    decode_off = mean([s['decode_tokens_per_second'] for s in off])
    decode_on = mean([s['decode_tokens_per_second'] for s in on])
    decode_base = mean([s['decode_tokens_per_second'] for s in measured(baseline)]) if baseline else None
    speedup = report.get('decode_speedup') or (decode_on / decode_off if decode_off and decode_on else None)
    prefill_off, prefill_on = mean([s['prefill_milliseconds'] for s in off]), mean([s['prefill_milliseconds'] for s in on])
    return (f"| {label} | {on[0]['prompt_tokens']:,} | {sampling_label(report)} | {report.get('speculative_draft_length', '—')} | "
            f"{fmt(accepted)} | {fmt(proposed)} | {fmt(per_token)} | {fmt(tokens_per_batch)} | "
            f"{fmt(per_batch('propose_milliseconds'), 1)} | {fmt(per_batch('verify_milliseconds'), 1)} | {fmt(per_batch('accept_milliseconds'))} | "
            f"{fmt(per_batch('recover_milliseconds'), 1)} | {fmt(per_batch('checkpoint_milliseconds'))} | {fmt(per_batch('commit_milliseconds'))} | "
            f"{fmt(prefill_off / 1000 if prefill_off else None)} → {fmt(prefill_on / 1000 if prefill_on else None)} | "
            f"{fmt(decode_base)} → {fmt(decode_off)} → {fmt(decode_on)} | {fmt(speedup)}× |")


def table(entries):
    """`entries`: [(label, report, baseline_report_or_None)]; one table, pair or plain by the first report."""
    if not entries:
        raise ValueError('no reports')
    pair = is_pair(entries[0][1])
    lines = [PAIR_HEADER if pair else PLAIN_HEADER]
    for label, report, baseline in entries:
        lines.append(pair_row(label, report, baseline) if pair else plain_row(label, report))
    return '\n'.join(lines)


def provenance(entries, paths):
    r = entries[0][1]
    ctx = ', '.join(sorted({f"{e[1].get('context')} ctx / {e[1].get('kv_precision')} KV / {e[1].get('build_mode')}" for e in entries}))
    names = ', '.join(f'`{p}`' for p in paths)
    return f'Generated by `scripts/bench-report.py` from {names} ({r.get("backend")}, {ctx}); means over the measured runs.'


def replace_region(text, name, body):
    start, end = f'<!-- bench:{name} -->', f'<!-- /bench:{name} -->'
    if start not in text or end not in text:
        raise KeyError(f'markers {start} … {end} not found')
    pattern = re.compile(re.escape(start) + r'.*?' + re.escape(end), re.S)
    return pattern.sub(lambda m: f'{start}\n{body}\n{end}', text, count=1)


def metrics(report):
    rows = measured(report)
    out = {'prefill_tokens_per_second': report.get('mean_prefill_tokens_per_second') or mean([s['prefill_tokens_per_second'] for s in rows if not s['speculative']]),
           'decode_tokens_per_second': report.get('mean_decode_tokens_per_second') or mean([s['decode_tokens_per_second'] for s in rows if not s['speculative']])}
    if report.get('decode_speedup') is not None:
        out['decode_speedup'] = report['decode_speedup']
    return out


def compare(prev, cur):
    a, b = metrics(prev), metrics(cur)
    lines = []
    for key in ('prefill_tokens_per_second', 'decode_tokens_per_second', 'decode_speedup'):
        if a.get(key) is not None and b.get(key) is not None:
            lines.append(f'{key:<26} {a[key]:9.3f} → {b[key]:9.3f}  {100 * (b[key] / a[key] - 1):+.1f} %')
    return lines


def check_bars(bars, report):
    """(missed, lines): each bar's metric against min/max."""
    m = metrics(report)
    missed, lines = [], []
    for metric, bar in bars.items():
        value = m.get(metric)
        ok = value is not None and (bar.get('min') is None or value >= bar['min']) and (bar.get('max') is None or value <= bar['max'])
        bound = ' '.join(f'{k} {v}' for k, v in bar.items())
        lines.append(f"{'PASS' if ok else 'MISS'} {metric} {fmt(value, 3)} ({bound})")
        if not ok:
            missed.append(metric)
    return missed, lines


# ---- execution ----------------------------------------------------------------

def load_entries(paths):
    entries = []
    for p in paths:
        path = Path(p)
        report = json.loads(path.read_text())
        sibling = path.with_name(path.stem + '-baseline.json')
        baseline = json.loads(sibling.read_text()) if sibling.is_file() else None
        entries.append((path.stem, report, baseline))
    return entries


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--table', nargs='+', metavar='FILE', help='render the table of these reports')
    ap.add_argument('--write-doc', metavar='DOC', help='with --table and --name: replace the marked region of DOC')
    ap.add_argument('--name', help='the marker name for --write-doc')
    ap.add_argument('--compare', nargs=2, metavar=('PREV', 'CUR'), help='print the deltas between two reports')
    ap.add_argument('--check', nargs=2, metavar=('WORKLOAD', 'FILE'), help="test FILE against the workload's bars in workloads.json")
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        sys.exit(0 if self_test() else 1)
    status = 0
    if args.table:
        entries = load_entries(args.table)
        body = table(entries) + '\n\n' + provenance(entries, args.table)
        if args.write_doc:
            if not args.name:
                ap.error('--write-doc needs --name')
            doc = Path(args.write_doc)
            doc.write_text(replace_region(doc.read_text(), args.name, body))
            print(f'wrote bench:{args.name} in {doc}')
        else:
            print(body)
    if args.compare:
        prev, cur = (json.loads(Path(p).read_text()) for p in args.compare)
        print('\n'.join(compare(prev, cur)) or 'nothing comparable')
    if args.check:
        name, path = args.check
        workloads = json.loads((ROOT / 'workloads.json').read_text())['workloads']
        if name not in workloads:
            sys.exit(f'unknown workload {name}')
        missed, lines = check_bars(workloads[name].get('bars', {}), json.loads(Path(path).read_text()))
        print('\n'.join(lines) or f'{name}: no bars')
        status = 1 if missed else 0
    if not (args.table or args.compare or args.check):
        ap.error('one of --table, --compare, --check is required')
    sys.exit(status)


# ---- self-test ----------------------------------------------------------------

def sample_plain():
    s = lambda w, pf, dc: {'warmup': w, 'prompt_tokens': 512, 'generated_tokens': 128, 'stop_reason': 'token_budget', 'prefill_milliseconds': 2000,
                           'first_token_milliseconds': 2100, 'decode_milliseconds': 12000, 'prefill_tokens_per_second': pf,
                           'decode_tokens_per_second': dc, 'speculative': False}
    return {'backend': 'metal', 'context': 32768, 'kv_precision': 'f16', 'build_mode': 'ReleaseSafe', 'session_bytes': 3 * 2**30,
            'load_milliseconds': 800, 'sampling': 'greedy', 'samples': [s(True, 100, 9), s(False, 180, 24), s(False, 182, 26)],
            'mean_prefill_tokens_per_second': 181, 'mean_decode_tokens_per_second': 25}


def sample_pair():
    off = {'warmup': False, 'prompt_tokens': 512, 'generated_tokens': 128, 'stop_reason': 'token_budget', 'prefill_milliseconds': 2000,
           'first_token_milliseconds': 2100, 'decode_milliseconds': 5000, 'prefill_tokens_per_second': 256, 'decode_tokens_per_second': 25.4, 'speculative': False}
    on = dict(off, speculative=True, draft_length=4, speculative_steps=40, accepted_per_step=2.25, proposed_per_step=3.4, propose_milliseconds=272,
              verify_milliseconds=5440, accept_milliseconds=4, recover_milliseconds=0.04, checkpoint_milliseconds=0.0, commit_milliseconds=0.08,
              decode_tokens_per_second=22.8, prefill_milliseconds=2050)
    return {'backend': 'metal', 'context': 32768, 'kv_precision': 'f16', 'build_mode': 'ReleaseSafe', 'sampling': 'greedy',
            'samples': [dict(off, warmup=True), on | {'warmup': True}, off, on], 'speculative_draft_length': 4,
            'mean_prefill_tokens_per_second': 256, 'mean_decode_tokens_per_second': 25.4, 'mean_speculative_decode_tokens_per_second': 22.8,
            'decode_speedup': 22.8 / 25.4}


class SelfTest(unittest.TestCase):
    def test_plain_table(self):
        out = table([('r1', sample_plain(), None)])
        self.assertTrue(out.startswith('| report | prompt |'))
        self.assertIn('| r1 | 512 | greedy | 2 | 181.00 ± 1.41 | 25.00 ± 1.41 | 2.1 | 3072 | 800 |', out)

    def test_pair_table_with_and_without_baseline(self):
        out = table([('p', sample_pair(), None)])
        self.assertIn('| p | 512 | greedy | 4 | 2.25 | 3.40 | 1.51 | 3.17 | 6.8 | 136.0 | 0.10 | 0.0 | 0.00 | 0.00 | 2.00 → 2.05 | — → 25.40 → 22.80 | 0.90× |', out)
        base = sample_plain()
        out = table([('p', sample_pair(), base)])
        self.assertIn('| 25.00 → 25.40 → 22.80 |', out)

    def test_replace_region(self):
        text = 'a\n<!-- bench:x -->\nold\n<!-- /bench:x -->\nb\n'
        self.assertEqual(replace_region(text, 'x', 'new'), 'a\n<!-- bench:x -->\nnew\n<!-- /bench:x -->\nb\n')
        with self.assertRaises(KeyError):
            replace_region(text, 'y', 'new')

    def test_compare_and_bars(self):
        lines = compare(sample_plain(), sample_pair())
        self.assertTrue(any(l.startswith('decode_tokens_per_second') and '+1.6 %' in l for l in lines), lines)
        missed, lines = check_bars({'decode_speedup': {'min': 1.0}}, sample_pair())
        self.assertEqual(missed, ['decode_speedup'])
        missed, _ = check_bars({'decode_tokens_per_second': {'min': 20}}, sample_pair())
        self.assertEqual(missed, [])


def self_test():
    return unittest.TextTestRunner(verbosity=0).run(unittest.defaultTestLoader.loadTestsFromTestCase(SelfTest)).wasSuccessful()


if __name__ == '__main__':
    main()
