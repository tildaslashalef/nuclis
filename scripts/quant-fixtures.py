#!/usr/bin/env python3
"""Regenerate small CPU decoder fixtures from a locally built pinned llama.cpp.

Usage: python3 scripts/quant-fixtures.py [reference-checkout] [--prism-checkout DIR]
Defaults to .zig-cache/reference/llama.cpp under the repository root, and
.zig-cache/reference/prism-llama.cpp for the PrismML fork that defines the
ternary encodings (PQ2_0, PTQ1_0; docs/reference/bonsai.md). Each checkout
must sit at its pinned revision with its Release build; no model or GPU is
used. Writes fixtures under inference/src/quant/fixtures/ (`ternary.json`
from the fork, the rest from mainline) and the attributed IQ3_S table in
inference/src/quant/iq3-grid.zig.
"""

import argparse
import ctypes
import json
import pathlib
import re
import struct
import subprocess


REVISION = "7620399f58aebfd2196b74021f9581bcf7218cb9"
# The PrismML fork at release prism-b10687-5d80cff: the only decoder of the
# Prism-private ternary encodings (docs/reference/reference-baseline.md).
PRISM_REVISION = "5d80cff0b8cb9f2bf823cfc4e71e3abb97f290d6"


def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", nargs="?", type=pathlib.Path,
                        default=root / ".zig-cache/reference/llama.cpp")
    parser.add_argument("--prism-checkout", type=pathlib.Path,
                        default=root / ".zig-cache/reference/prism-llama.cpp")
    args = parser.parse_args()

    def pinned_library(checkout, expected):
        checkout = checkout.resolve()
        revision = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != expected:
            raise SystemExit(f"{checkout} must sit at the pinned revision {expected}")
        return ctypes.CDLL(str(checkout / "build/bin/libggml-base.dylib"))

    library = pinned_library(args.checkout, REVISION)
    checkout = args.checkout.resolve()

    def reference(encoding, name, blocks, elements, source_library=None):
        encoded = b"".join(blocks)
        source = ctypes.create_string_buffer(encoded)
        output = (ctypes.c_float * (elements * len(blocks)))()
        decode = getattr(source_library or library, "dequantize_row_" + name)
        decode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
        decode.restype = None
        decode(source, output, len(output))
        return {"encoding": encoding, "bytes": list(encoded), "values": list(output)}

    def save(name, rows, revision=REVISION):
        target = root / "inference/src/quant/fixtures" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"revision": revision, "rows": rows}, indent=2) + "\n")

    # The fork's group-128 ternary blocks. Arbitrary payload bytes exercise
    # the fixed-point trit extraction on non-canonical codes too, since the
    # decoder must reproduce the reference arithmetic, not only valid packings.
    prism = pinned_library(args.prism_checkout, PRISM_REVISION)
    scales = [0.5, -2, 0, 1.5, 2**-24, 65504, -0.125, 8]
    fixtures = []
    blocks = [struct.pack("<e", scale) + bytes((block * 32 + j) * 73 % 256 for j in range(32))
              for block, scale in enumerate(scales)]
    blocks.append(struct.pack("<e", 1) + bytes([0b11100100] * 32))  # every 2-bit code, in order
    blocks.append(struct.pack("<e", 1) + bytes([0xff] * 32))
    fixtures.append(reference(142, "pq2_0", blocks, 128, prism))
    blocks = [bytes((block * 26 + j) * 73 % 256 for j in range(26)) + struct.pack("<e", scale)
              for block, scale in enumerate(scales)]
    blocks.append(bytes([0] * 26) + struct.pack("<e", 1))
    blocks.append(bytes([0xff] * 26) + struct.pack("<e", 1))
    # Canonical packings: five trits per byte scaled by 256/243 (rounded up)
    # and the four-trit tail bytes, so the fixture also holds valid codes.
    def pack(trits, scale):
        return (sum(t * 3 ** (4 - i) for i, t in enumerate(trits)) * 256 + 242) // 243
    canonical = bytes(pack([(j + k) % 3 for k in range(5)], 5) for j in range(24))
    canonical += bytes((sum(((j + k) % 3) * 3 ** (3 - k) for k in range(4)) * 256 + 242) // 243 for j in range(2))
    blocks.append(canonical + struct.pack("<e", 0.5))
    fixtures.append(reference(143, "ptq1_0", blocks, 128, prism))
    save("ternary.json", fixtures, PRISM_REVISION)

    fixtures = []
    for encoding, name, width in [(8, "q8_0", 32), (20, "iq4_nl", 16), (2, "q4_0", 16)]:
        # Exactly representable half scales, including negative, zero, tiny,
        # and large values. Q8 covers all 256 signed byte representations.
        scales = [0.5, -2, 0, 1.5, 2**-24, 65504, -0.125, 8]
        blocks = [
            struct.pack("<e", scale)
            + bytes((block * width + j) * 73 % 256 for j in range(width))
            for block, scale in enumerate(scales)
        ]
        fixtures.append(reference(encoding, name, blocks, 32))
    save("simple.json", fixtures)

    fixtures = []
    for encoding, name, size in [(12, "q4_K", 144), (13, "q5_K", 176)]:
        blocks = []
        for index, (d, dmin) in enumerate([(0.5, 0.25), (-0.125, 1.5),
                                          (2**-24, 0), (0, 65504)]):
            # Populate the packed coefficient bytes directly, independently of
            # Zig's unpacking equations. Odd stride visits every byte value.
            payload = bytes((j * 73 + index * 41) % 256 for j in range(size - 4))
            blocks.append(struct.pack("<ee", d, dmin) + payload)
        fixtures.append(reference(encoding, name, blocks, 256))
    save("k-affine.json", fixtures)

    fixtures = []
    for encoding, name, size in [(11, "q3_K", 110), (14, "q6_K", 210)]:
        blocks = []
        for index, scale in enumerate([0.5, -0.125, 2**-24, 65504, 0]):
            payload = bytes((j * 73 + index * 41) % 256 for j in range(size - 2))
            blocks.append(payload + struct.pack("<e", scale))
        fixtures.append(reference(encoding, name, blocks, 256))
    save("k-signed.json", fixtures)

    fixtures = []
    blocks = []
    for index, scale in enumerate([0.5, -0.125, 2**-24, 65504, 1, 2, 3, 4]):
        # Cover every grid index once with nonzero scales. The high-index plane
        # selects the upper 256 entries in the last four blocks.
        indices = bytes((index * 64 + j) % 256 for j in range(64))
        high = bytes([255 if index >= 4 else 0] * 8)
        signs = bytes((j * 73 + index * 41) % 256 for j in range(32))
        scales = bytes((j * 73 + index * 41) % 256 for j in range(4))
        blocks.append(struct.pack("<e", scale) + indices + high + signs + scales)
    # Mixed ninth-index bits catch swapped bit positions that all-zero/all-one
    # high planes in the exhaustive table sweep would not distinguish.
    blocks.append(struct.pack("<e", 1) + bytes((j * 73 + 19) % 256 for j in range(108)))
    blocks.append(struct.pack("<e", 0) + bytes([255] * 108))
    fixtures.append(reference(21, "iq3_s", blocks, 256))
    blocks = []
    for index, scale in enumerate([0.5, -0.125, 2**-24, 65504, 0]):
        payload = bytes((j * 73 + index * 41) % 256 for j in range(134))
        blocks.append(struct.pack("<e", scale) + payload)
    fixtures.append(reference(23, "iq4_xs", blocks, 256))
    save("iq.json", fixtures)

    # This normative codebook is format data, not learned model weights. Retain
    # its upstream license and provenance alongside the packaged Zig source.
    common = (checkout / "ggml/src/ggml-common.h").read_text()
    body = common.split("GGML_TABLE_BEGIN(uint32_t, iq3s_grid, 512)", 1)[1].split("GGML_TABLE_END()", 1)[0]
    entries = re.findall(r"0x[0-9a-fA-F]+", body)
    if len(entries) != 512:
        raise RuntimeError("expected the pinned 512-entry IQ3_S grid")
    lines = ["    " + ", ".join(entries[i:i+8]) + "," for i in range(0, 512, 8)]
    table = ("//! IQ3_S codebook: a GGML format constant obtained from llama.cpp\n"
             "//! " + REVISION + " (ggml/src/ggml-common.h); see\n"
             "//! THIRD_PARTY_NOTICES.md. Regenerate with scripts/quant-fixtures.py. Low byte\n"
             "//! is component 0. The Metal backend emits this same table into its shader.\n"
             "pub const values = [512]u32{\n" + "\n".join(lines) + "\n};\n")
    (root / "inference/src/quant/iq3-grid.zig").write_text(table)


if __name__ == "__main__":
    main()
