#!/usr/bin/env python3
"""Compare locally captured native/reference F32 traces (no model or NumPy needed)."""
import argparse
import array
import json
import math
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('native', type=Path)
parser.add_argument('reference', type=Path)
parser.add_argument('--positions', type=int, required=True)
parser.add_argument('--max-absolute', type=float, default=0.002)
parser.add_argument('--max-relative-rms', type=float, default=0.0001)
parser.add_argument('--embedding', type=int, default=5120, help='residual width per layer file (Qwen3.8 5120, Gemma 4 12B 3840)')
parser.add_argument('--layers', type=int, default=64, help='decoder layers traced (Qwen3.8 64, Gemma 4 12B 48)')
parser.add_argument('--vocab', type=int, default=248320, help='logit count (Qwen3.8 248320, Gemma 4 262144)')
parser.add_argument('--draft', action='store_true', help='compare prediction-block rows p0-h.f32/p1-h.f32 and greedy.txt instead of layers')
args = parser.parse_args()
if not 1 <= args.positions <= 32768:
    parser.error('positions must be between 1 and 32768')


def read(path, size):
    data = path.read_bytes()
    if len(data) != size * 4:
        raise ValueError(f'{path}: unexpected byte count')
    values = array.array('f', data)
    import sys
    if sys.byteorder != 'little':
        values.byteswap()
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f'{path}: nonfinite values')
    return values


def metric(name, a, b):
    size = len(a)
    rms = math.sqrt(sum((x-y)**2 for x, y in zip(a, b)) / size)
    baseline = math.sqrt(sum(y*y for y in b) / size)
    return dict(file=name, max_absolute=max(abs(x-y) for x, y in zip(a, b)), rms=rms, relative_rms=rms / max(baseline, 1e-30))


if args.draft:
    results = []
    for name, size in [(f'p{p}-h.f32', args.embedding) for p in range(args.positions)] + [('p1-hprev.f32', args.embedding)]:
        try:
            a, b = read(args.native / name, size), read(args.reference / name, size)
        except FileNotFoundError:
            continue
        results.append(metric(name, a, b))
    if not results:
        raise SystemExit('no draft rows found')
    native_greedy = [int(x) for x in (args.native / 'greedy.txt').read_text().split()]
    reference_greedy = [int(x) for x in (args.reference / 'greedy.txt').read_text().split()]
    passed = all(r['max_absolute'] <= args.max_absolute and r['relative_rms'] <= args.max_relative_rms for r in results) and native_greedy == reference_greedy
    print(json.dumps(dict(passed=passed, max_absolute_tolerance=args.max_absolute, relative_rms_tolerance=args.max_relative_rms,
                          native_greedy=native_greedy, reference_greedy=reference_greedy, comparisons=results), indent=2))
    raise SystemExit(0 if passed else 1)

results = []
for name, size in [(f'token-{p}-layer-{i}.f32', args.embedding) for p in range(args.positions) for i in range(args.layers)] + [('logits.f32', args.vocab)]:
    a, b = read(args.native / name, size), read(args.reference / name, size)
    rms = math.sqrt(sum((x-y)**2 for x, y in zip(a, b)) / size)
    baseline = math.sqrt(sum(y*y for y in b) / size)
    result = dict(file=name, max_absolute=max(abs(x-y) for x, y in zip(a, b)), rms=rms, relative_rms=rms / max(baseline, 1e-30))
    if name == 'logits.f32':
        result['native_top5'] = sorted(range(size), key=lambda i: a[i], reverse=True)[:5]
        result['reference_top5'] = sorted(range(size), key=lambda i: b[i], reverse=True)[:5]
        best, second = result['reference_top5'][:2]
        result['reference_greedy_margin'] = b[best] - b[second]
    results.append(result)
passed = all(r['max_absolute'] <= args.max_absolute and r['relative_rms'] <= args.max_relative_rms for r in results)
print(json.dumps(dict(passed=passed, max_absolute_tolerance=args.max_absolute, relative_rms_tolerance=args.max_relative_rms, comparisons=results), indent=2))
raise SystemExit(0 if passed else 1)
