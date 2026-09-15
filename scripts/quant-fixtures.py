#!/usr/bin/env python3
"""Regenerate small CPU decoder fixtures from a locally built pinned llama.cpp.

Usage: python3 scripts/quant-fixtures.py [reference-checkout]
Defaults to .zig-cache/reference/llama.cpp under the repository root.
Requires that checkout's matching Release build; no model or GPU is used.
Writes fixtures under inference/src/quant/fixtures/ and the attributed
IQ3_S table in inference/src/quant/iq3-grid.zig.
"""

import argparse
import ctypes
import json
import pathlib
import re
import struct
import subprocess


REVISION = "7620399f58aebfd2196b74021f9581bcf7218cb9"


def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", nargs="?", type=pathlib.Path,
                        default=root / ".zig-cache/reference/llama.cpp")
    checkout = parser.parse_args().checkout.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise SystemExit("reference checkout must match the pinned revision")
    library = ctypes.CDLL(str(checkout / "build/bin/libggml-base.dylib"))

    def reference(encoding, name, blocks, elements):
        encoded = b"".join(blocks)
        source = ctypes.create_string_buffer(encoded)
        output = (ctypes.c_float * (elements * len(blocks)))()
        decode = getattr(library, "dequantize_row_" + name)
        decode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
        decode.restype = None
        decode(source, output, len(output))
        return {"encoding": encoding, "bytes": list(encoded), "values": list(output)}

    def save(name, rows):
        target = root / "inference/src/quant/fixtures" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"revision": REVISION, "rows": rows}, indent=2) + "\n")

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
