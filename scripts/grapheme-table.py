#!/usr/bin/env python3
"""Generate src/tui/graphemes_table.zig from a pinned Unicode Character Database.

The table is the terminal's grapheme-breaking and width data: UAX #29 properties
(Grapheme_Cluster_Break, Indic_Conjunct_Break), UAX #11 East Asian Width, and the
emoji/legacy properties that decide emoji presentation. It is committed; this
script only regenerates it. Downloads are cached under .zig-cache/ucd/<version>/
and checked against pinned digests, so a given source tree always regenerates the
same bytes.

    python3 scripts/grapheme-table.py
"""
import argparse
import hashlib
import re
import urllib.request
from pathlib import Path

VERSION = '17.0.0'
BASE = f'https://www.unicode.org/Public/{VERSION}/ucd'
ROOT = Path(__file__).resolve().parents[1]

# name -> sha256, pinned for the version above.
SOURCES = {
    'auxiliary/GraphemeBreakProperty.txt': 'd6b51d1d2ae5c33b451b7ed994b48f1f4dc62b2272a5831e7fd418514a6bae89',
    'emoji/emoji-data.txt': '2cb2bb9455cda83e8481541ecf5b6dfda66a3bb89efa3fa7c5297eccf607b72b',
    'EastAsianWidth.txt': 'ea7ce50f3444a050333448dffef1cadd9325af55cbb764b4a2280faf52170a33',
    'DerivedCoreProperties.txt': '24c7fed1195c482faaefd5c1e7eb821c5ee1fb6de07ecdbaa64b56a99da22c08',
}

GCB = {
    'Other': 'other', 'CR': 'cr', 'LF': 'lf', 'Control': 'control',
    'Extend': 'extend', 'ZWJ': 'zwj', 'Regional_Indicator': 'ri',
    'Prepend': 'prepend', 'SpacingMark': 'spacing_mark', 'L': 'l', 'V': 'v',
    'T': 't', 'LV': 'lv', 'LVT': 'lvt',
}
INCB = {'Consonant': 'consonant', 'Extend': 'extend', 'Linker': 'linker'}

LINE = re.compile(
    r'^\s*([0-9A-Fa-f]+)(?:\.\.([0-9A-Fa-f]+))?\s*;\s*([A-Za-z_]+)'
    r'(?:\s*;\s*([A-Za-z_]+))?')

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--ucd-dir', type=Path, default=ROOT / '.zig-cache/ucd' / VERSION,
                    help='where the UCD files are cached/downloaded')
parser.add_argument('--output', type=Path, default=ROOT / 'src/tui/graphemes_table.zig')
parser.add_argument('--offline', action='store_true', help='fail instead of downloading')
args = parser.parse_args()


def source(relative):
    path = args.ucd_dir / Path(relative).name
    if not path.exists():
        if args.offline:
            parser.error(f'missing {path}; run without --offline')
        args.ucd_dir.mkdir(parents=True, exist_ok=True)
        url = f'{BASE}/{relative}'
        print('fetch', url)
        with urllib.request.urlopen(url, timeout=60) as response:
            path.write_bytes(response.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SOURCES[relative]:
        parser.error(f'{relative} sha256 {digest} != pinned {SOURCES[relative]}')
    return path.read_text(encoding='utf-8')


def assign(table, start, end, value):
    table[start:end + 1] = bytes([value]) * (end - start + 1)


# Grapheme_Cluster_Break, InCB, East Asian Width, and the emoji properties. A
# code point that no source names keeps the default (other/none/narrow/false).
gcb = bytearray(0x110000)
incb = bytearray(0x110000)
flags = bytearray(0x110000)
WIDE, IGNORABLE, EMOJI_PRESENTATION, EMOJI_MODIFIER, EXTENDED_PICTOGRAPHIC = 1, 2, 4, 8, 16

for line in source('auxiliary/GraphemeBreakProperty.txt').splitlines():
    match = LINE.match(line)
    if not match or line.lstrip().startswith('#'):
        continue
    start = int(match.group(1), 16)
    end = int(match.group(2), 16) if match.group(2) else start
    assign(gcb, start, end, list(GCB).index(match.group(3)) if match.group(3) in GCB else 0)

for line in source('DerivedCoreProperties.txt').splitlines():
    match = LINE.match(line)
    if not match or line.lstrip().startswith('#'):
        continue
    start = int(match.group(1), 16)
    end = int(match.group(2), 16) if match.group(2) else start
    prop, value = match.group(3), match.group(4)
    if prop == 'Default_Ignorable_Code_Point':
        for cp in range(start, end + 1):
            flags[cp] |= IGNORABLE
    elif prop == 'InCB' and value in INCB:
        assign(incb, start, end, list(INCB).index(value) + 1)

for line in source('EastAsianWidth.txt').splitlines():
    match = LINE.match(line)
    if not match or line.lstrip().startswith('#'):
        continue
    start = int(match.group(1), 16)
    end = int(match.group(2), 16) if match.group(2) else start
    if match.group(3) in ('W', 'F'):
        for cp in range(start, end + 1):
            flags[cp] |= WIDE

for line in source('emoji/emoji-data.txt').splitlines():
    match = LINE.match(line)
    if not match or line.lstrip().startswith('#'):
        continue
    start = int(match.group(1), 16)
    end = int(match.group(2), 16) if match.group(2) else start
    bit = {
        'Emoji_Presentation': EMOJI_PRESENTATION,
        'Emoji_Modifier': EMOJI_MODIFIER,
        'Extended_Pictographic': EXTENDED_PICTOGRAPHIC,
    }.get(match.group(3))
    if bit:
        for cp in range(start, end + 1):
            flags[cp] |= bit

# One interval per maximal run of identical properties. `props` prints only the
# fields that differ from the defaults, so most intervals are one line.
intervals = []
start = 0
last = (gcb[0], incb[0], flags[0])
for cp in range(1, 0x110000):
    value = (gcb[cp], incb[cp], flags[cp])
    if value != last:
        intervals.append((start, last))
        start, last = cp, value
intervals.append((start, last))


def emit_props(value):
    g, i, f = value
    fields = []
    if g:
        fields.append(f'.gcb = .{GCB[list(GCB)[g]]}')
    if i:
        fields.append(f'.incb = .{INCB[list(INCB)[i - 1]]}')
    if f & WIDE:
        fields.append('.wide = true')
    if f & IGNORABLE:
        fields.append('.ignorable = true')
    if f & EMOJI_PRESENTATION:
        fields.append('.emoji_presentation = true')
    if f & EMOJI_MODIFIER:
        fields.append('.emoji_modifier = true')
    if f & EXTENDED_PICTOGRAPHIC:
        fields.append('.extended_pictographic = true')
    return '.{ ' + ', '.join(fields) + ' }' if fields else '.{}'


def gcb_enum():
    return 'pub const Gcb = enum(u4) { ' + ', '.join(
        f'{name} = {index}' for index, name in enumerate(GCB.values())) + ' };'


def incb_enum():
    return 'pub const Incb = enum(u2) { none = 0, ' + ', '.join(
        f'{name} = {index + 1}' for index, name in enumerate(INCB.values())) + ' };'


header = f'''//! Generated by scripts/grapheme-table.py from the Unicode Character Database
//! {VERSION}; do not edit by hand. Unicode data is Copyright © Unicode, Inc.,
//! licensed under the Unicode License (https://www.unicode.org/license.txt);
//! see THIRD_PARTY_NOTICES.md. Regenerate with `python3 scripts/grapheme-table.py`.
//!
//! One sorted interval table. `graphemes.zig` binary-searches it; nothing else
//! reads it, so replacing it (for example with `uucode`) is a local change.
pub const unicode_version = "{VERSION}";

/// Grapheme_Cluster_Break, UAX #29.
{gcb_enum()}
/// Indic_Conjunct_Break, UAX #29 GB9c.
{incb_enum()}
pub const Props = packed struct(u16) {{
    gcb: Gcb = .other,
    incb: Incb = .none,
    /// East Asian Width `W` or `F` (UAX #11); ambiguous `A` is narrow here.
    wide: bool = false,
    ignorable: bool = false,
    emoji_presentation: bool = false,
    emoji_modifier: bool = false,
    extended_pictographic: bool = false,
    _pad: u5 = 0,
}};

pub const Interval = struct {{ start: u21, props: Props }};

pub const intervals = [_]Interval{{
'''

lines = [header]
for start_cp, value in intervals:
    lines.append(f'    .{{ .start = 0x{start_cp:05X}, .props = {emit_props(value)} }},\n')
lines.append('};\n')

args.output.write_text(''.join(lines), encoding='utf-8')
print(f'{len(intervals)} intervals -> {args.output.relative_to(ROOT)}')
