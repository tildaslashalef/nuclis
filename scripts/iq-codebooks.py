#!/usr/bin/env python3
"""Extract the IQ1/IQ2/IQ3_XXS codebooks from ggml-common.h into Zig.

Usage: python3 scripts/iq-codebooks.py [checkout]

Reads only the header text of a llama.cpp checkout (default
.reference/llama.cpp at the revision THIRD_PARTY_NOTICES.md cites); builds
and runs nothing. The tables are format data: the encodings index them, so
a decoder must hold the same numbers. Writes
inference/src/quant/iq-codebooks.zig. The IQ2 sign table is not copied:
it is the 7-bit index with its parity in bit 7, computed in Zig.
"""

import argparse
import pathlib
import re
import subprocess

REVISION = "7620399f58aebfd2196b74021f9581bcf7218cb9"
TABLES = [
    # (ggml declaration, Zig name, element type, count, what the entry holds)
    ("uint64_t, iq2xxs_grid, 256", "iq2_xxs", "u64", 256, "eight magnitudes in {8, 25, 43}"),
    ("uint64_t, iq2xs_grid, 512", "iq2_xs", "u64", 512, "eight magnitudes in {8, 25, 43}"),
    ("uint64_t, iq2s_grid, 1024", "iq2_s", "u64", 1024, "eight magnitudes in {8, 25, 43}"),
    ("uint32_t, iq3xxs_grid, 256", "iq3_xxs", "u32", 256, "four magnitudes in {4, 12, ..., 62}"),
    ("uint64_t, iq1s_grid, NGRID_IQ1S", "iq1_s", "u64", 2048, "eight signed bytes in {-1, 0, +1}; IQ1_M shares it"),
]


def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", nargs="?", type=pathlib.Path, default=root / ".reference/llama.cpp")
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "-C", str(args.checkout), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise SystemExit(f"{args.checkout} must sit at {REVISION}")
    common = (args.checkout / "ggml/src/ggml-common.h").read_text()

    out = [
        "//! IQ1_S/IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, and IQ3_XXS codebooks: GGML format",
        "//! constants from llama.cpp " + REVISION,
        "//! (ggml/src/ggml-common.h); see THIRD_PARTY_NOTICES.md. Regenerate with",
        "//! scripts/iq-codebooks.py. Low byte is component 0.",
        "",
    ]
    for declaration, name, element, count, holds in TABLES:
        body = common.split(f"GGML_TABLE_BEGIN({declaration})", 1)[1].split("GGML_TABLE_END()", 1)[0]
        entries = re.findall(r"0x[0-9a-fA-F]+", body)
        if len(entries) != count:
            raise SystemExit(f"{name}: expected {count} entries, found {len(entries)}")
        per_line = 4 if element == "u64" else 8
        out.append(f"/// {count} entries; each holds {holds}.")
        out.append(f"pub const {name} = [{count}]{element}{{")
        out += ["    " + ", ".join(entries[i : i + per_line]) + "," for i in range(0, count, per_line)]
        out.append("};")
        out.append("")
    (root / "inference/src/quant/iq-codebooks.zig").write_text("\n".join(out))


if __name__ == "__main__":
    main()
