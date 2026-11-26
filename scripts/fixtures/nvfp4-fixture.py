#!/usr/bin/env python3
"""Write the NVFP4 decoder fixture from the format's equations.

Usage: python3 scripts/fixtures/nvfp4-fixture.py

Writes inference/src/quant/fixtures/nvfp4.json: one 1,152-value row (18
blocks of 64, an Inkling 2 expert's down row) of NVFP4, ggml id 40. The
first two blocks hold every code in every sub-block, in both nibble orders;
their scale bytes are the format's edge cases (0x00, 0x7F, bit 7, subnormal,
exponent 15). The other sixteen carry seeded random codes under scale bytes
in 0x4C-0x7E, the range the Inkling file uses. Values follow the format
(docs/engine/quantization.md): in sub-block s, byte j holds value j in its
low nibble and value j + 8 in its high nibble; a value is e2m1(code) *
ue4m3(scale). No external decoder is involved; the CPU decoder
(`zig build test`) and `metal-check` hold themselves to this row.
"""

import json
import pathlib
import random

REVISION = "nuclis: the format's equations (scripts/fixtures/nvfp4-fixture.py)"
E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]
EDGE_SCALES = [[0x00, 0x7F, 0x80, 0xFF], [0x01, 0x07, 0xB8, 0x7E]]
BLOCKS = 18


def ue4m3(byte):
    if byte in (0x00, 0x7F):
        return 0.0
    exponent, mantissa = (byte >> 3) & 15, byte & 7
    if exponent == 0:
        return mantissa * 2.0**-9
    return (1 + mantissa / 8) * 2.0 ** (exponent - 7)


def block(index, rng):
    if index < 2:
        scales = EDGE_SCALES[index]
        pattern = [j | (j + 8) << 4 if index == 0 else (j + 8) | j << 4 for j in range(8)]
        codes = pattern * 4
    else:
        scales = [rng.randint(0x4C, 0x7E) for _ in range(4)]
        codes = [rng.getrandbits(8) for _ in range(32)]
    values = [0.0] * 64
    for s in range(4):
        d = ue4m3(scales[s])
        for j in range(8):
            byte = codes[8 * s + j]
            values[16 * s + j] = E2M1[byte & 15] * d
            values[16 * s + j + 8] = E2M1[byte >> 4] * d
    return bytes(scales + codes), values


def main():
    root = pathlib.Path(__file__).resolve().parents[2]
    rng = random.Random(40)
    data, values = b"", []
    for index in range(BLOCKS):
        more, decoded = block(index, rng)
        data += more
        values += decoded
    rows = [{"encoding": 40, "bytes": list(data), "values": values}]
    out = root / "inference/src/quant/fixtures/nvfp4.json"
    out.write_text(json.dumps({"revision": REVISION, "rows": rows}, indent=2) + "\n")
    print(f"wrote {out.relative_to(root)}")


if __name__ == "__main__":
    main()
