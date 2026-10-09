#!/usr/bin/env python3
"""Compare two EmbeddingGemma 2 vector sets case by case (no model or NumPy needed).

Each side is a prefix: PREFIX.json (`dimensions`, `cases` with `id`, `tokens`,
and run-length `ids`) beside PREFIX.f32 (the vectors in case order), as
scripts/embedding-reference.py and scripts/reference-embedding.cpp write
them. Reports the cosine per case, its minimum and mean per modality (the
id's first component), and every case whose token ids differ. Exits 1 when
a cosine is below --min-cosine or the ids differ (unless --ignore-ids).
"""

import argparse
import array
import json
import math
import sys
from pathlib import Path


def load(prefix: str):
    meta = json.loads(Path(prefix + ".json").read_text())
    values = array.array("f", Path(prefix + ".f32").read_bytes())
    if sys.byteorder != "little":
        values.byteswap()
    dims = meta["dimensions"]
    if len(values) != dims * len(meta["cases"]):
        raise SystemExit(f"{prefix}.f32: {len(values)} floats for {len(meta['cases'])} cases of {dims}")
    return {c["id"]: (c, values[i * dims : (i + 1) * dims]) for i, c in enumerate(meta["cases"])}


def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))


def main():
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument("--min-cosine", type=float, default=0.0)
    parser.add_argument("--ignore-ids", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="print every case")
    args = parser.parse_args()
    left, right = load(args.left), load(args.right)
    shared = [k for k in left if k in right]
    if not shared:
        raise SystemExit("no case in common")
    groups: dict[str, list[float]] = {}
    failed = False
    for key in shared:
        (lc, lv), (rc, rv) = left[key], right[key]
        c = cosine(lv, rv)
        group = key.split(".")[0] if key.split(".")[0] in ("image", "audio", "mix", "long-3k", "long-8k") else "text"
        groups.setdefault(group, []).append(c)
        same = lc["ids"] == rc["ids"]
        if not same and not args.ignore_ids:
            failed = True
            print(f"{key}: token ids differ ({lc['tokens']} against {rc['tokens']} tokens)")
        if c < args.min_cosine:
            failed = True
        if args.verbose or c < args.min_cosine:
            print(f"{key:28s} cosine {c:.6f}{'' if same else '  ids differ'}")
    for group, values in groups.items():
        print(f"{group:8s} {len(values):3d} cases  min {min(values):.6f}  mean {sum(values) / len(values):.6f}")
    missing = sorted(set(left) ^ set(right))
    if missing:
        print(f"{len(missing)} cases on one side only: {', '.join(missing[:6])}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
