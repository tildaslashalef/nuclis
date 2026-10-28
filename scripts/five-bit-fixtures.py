#!/usr/bin/env python3
"""Write the Q5_0 and Q5_1 decoder fixture from the formats' equations.

Usage: python3 scripts/five-bit-fixtures.py

Writes inference/src/quant/fixtures/five-bit.json: one 256-value row (eight
blocks of 32) of each encoding. The first two blocks hold every code on both
half-rows (planes 0x5555aaaa and its inverse over the nibble pattern
0, 15 / 1, 14 / ...); the other six carry seeded random payloads. Scales span
negative, subnormal, and large F16 values. Values follow the formats
(docs/engine/quantization.md): code q = nibble | bit << 4, where value j < 16
takes the low nibble of byte j and bit j of the plane, value j + 16 the high
nibble and bit j + 16; Q5_0 is d * (q - 16), Q5_1 d * q + m, rounded to F32.
No external decoder is involved; the CPU decoder (`zig build test`) and
`metal-check` hold themselves to these rows.
"""

import json
import pathlib
import random
import struct

REVISION = "nuclis: the formats' equations (scripts/five-bit-fixtures.py)"
NIBBLES = bytes(((15 - j) << 4) | j for j in range(16))
PLANES = [0x5555AAAA, 0xAAAA5555]
SCALES = [0.5, -1.0, 2.0**-6, 0.0137, -3.75, 2.0**-20, 1000.0, -0.0002]
MINIMUMS = [-2.0, 0.125, -0.0071, 3.5, -100.0, 0.0, 0.001, -0.5]


def f16(value):
    return struct.pack("<e", value)


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def block(encoding, index, rng):
    if index < 2:
        plane, payload = PLANES[index], NIBBLES
    else:
        plane, payload = rng.getrandbits(32), bytes(rng.getrandbits(8) for _ in range(16))
    d = struct.unpack("<e", f16(SCALES[index]))[0]
    m = struct.unpack("<e", f16(MINIMUMS[index]))[0]
    header = f16(d) + (f16(m) if encoding == 7 else b"")
    data = header + struct.pack("<I", plane) + payload
    values = [0.0] * 32
    for j in range(16):
        low = (payload[j] & 15) | ((plane >> j) & 1) << 4
        high = (payload[j] >> 4) | ((plane >> (j + 16)) & 1) << 4
        for at, code in ((j, low), (j + 16, high)):
            values[at] = f32(d * (code - 16)) if encoding == 6 else f32(f32(d * code) + m)
    return data, values


def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    rows = []
    for encoding in (6, 7):
        rng = random.Random(encoding)
        data, values = b"", []
        for index in range(8):
            more, decoded = block(encoding, index, rng)
            data += more
            values += decoded
        rows.append({"encoding": encoding, "bytes": list(data), "values": values})
    out = root / "inference/src/quant/fixtures/five-bit.json"
    out.write_text(json.dumps({"revision": REVISION, "rows": rows}, indent=2) + "\n")
    print(f"wrote {out.relative_to(root)}")


if __name__ == "__main__":
    main()
