#!/usr/bin/env python3
"""Extract the pinned reference's L/M/N/whitespace flags; no host Unicode DB."""
import argparse
import hashlib
import re
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--reference', type=Path, default=ROOT / '.zig-cache/reference/llama.cpp')
args = parser.parse_args()
source = (args.reference / 'src/unicode-data.cpp').read_bytes()
if hashlib.sha256(source).hexdigest() != '95170cd1c105a5b41a1b2dce73b0fae8ce8011ef7897600828bb2babe8b26e5d':
    parser.error('reference Unicode data differs from pinned llama.cpp 7620399f58aebfd2196b74021f9581bcf7218cb9')
text = source.decode()
ranges = [(int(a, 16), int(b, 16)) for a, b in re.findall(
    r'\{0x([0-9A-Fa-f]+), 0x([0-9A-Fa-f]+)\}', text.split('unicode_ranges_flags = {', 1)[1].split('};', 1)[0])]
spaces = {int(n, 16) for n in re.findall(r'0x([0-9A-Fa-f]+)', text.split('unicode_set_whitespace = {', 1)[1].split('};', 1)[0])}
assert ranges[0][0] == 0 and ranges[-1][0] == 0x110000
flags = bytearray(0x110000)
for (start, value), (end, _) in zip(ranges, ranges[1:]):
    flags[start:end] = bytes([value & 0x16]) * (end - start)
for cp in spaces:
    flags[cp] |= 0x20
out = bytearray()
last = None
for cp, value in enumerate(flags):
    if value != last:
        out.extend(struct.pack('<IB', cp, value))
        last = value
target = ROOT / 'inference/src/tokenizer/unicode-ranges.bin'
target.write_bytes(out)
print('source sha256:', hashlib.sha256(source).hexdigest())
print('ranges:', len(out) // 5, 'output sha256:', hashlib.sha256(out).hexdigest())
