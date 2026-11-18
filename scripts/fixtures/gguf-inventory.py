#!/usr/bin/env python3
"""Independent GGUF directory inventory (no weights): the fixture format the
adapters' tests hydrate (`inference/src/models/fixtures/*.json`).

Reads the header with the standard library only, so it shares nothing with the
Zig parser it checks. Output: {"metadata": {...}, "tensors": [[name, dims, encoding_id], ...]}.
Scalars and short strings are kept; arrays of at most 64 numbers keep their
values, as do arrays of any element type under the `prism.hadamard.` prefix
(the Zig parser retains the same: a binding validates the rotation from
them); other large arrays and long strings become descriptors (count/length,
byte offset, element type, and for strings the SHA-256), which is what the
tests need and keeps vocabularies out of the tree.

Usage: gguf-inventory.py <file.gguf> [--out fixture.json] [--max-string 256]
                         [--against fixture.json]

`--against` compares the file with an existing family's fixture: the
metadata keys and tensor names only one side has, with the architecture
prefix and block numbers folded, the first question of a new release
(docs/models/README.md § Which kind of new).
"""

import argparse
import hashlib
import json
import re
import struct

TYPES = {
    0: ("B", 1),
    1: ("b", 1),
    2: ("H", 2),
    3: ("h", 2),
    4: ("I", 4),
    5: ("i", 4),
    6: ("f", 4),
    7: ("?", 1),
    10: ("Q", 8),
    11: ("q", 8),
    12: ("d", 8),
}
STRING, ARRAY = 8, 9


class Reader:
    def __init__(self, f):
        self.f, self.pos = f, 0

    def read(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError("truncated directory")
        self.pos += n
        return b

    def scalar(self, t):
        fmt, size = TYPES[t]
        return struct.unpack("<" + fmt, self.read(size))[0]

    def string(self, max_string):
        n = self.scalar(10)
        offset = self.pos
        raw = self.read(n)
        if n <= max_string:
            return raw.decode("utf-8", "replace")
        return {
            "length": n,
            "offset": offset,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "head": raw[:64].decode("utf-8", "replace"),
        }

    def value(self, t, max_string, retain=False):
        if t == STRING:
            return self.string(max_string)
        if t == ARRAY:
            et, n = self.scalar(4), self.scalar(10)
            offset = self.pos
            if retain and et == STRING:
                return {"count": n, "offset": offset, "type": et, "values": [self.string(1 << 16) for _ in range(n)]}
            if et in TYPES and (n <= 64 or retain):
                return {
                    "count": n,
                    "offset": offset,
                    "type": et,
                    "values": [int(self.scalar(et)) if et == 7 else self.scalar(et) for _ in range(n)],
                }
            if et in TYPES:
                self.read(n * TYPES[et][1])
            elif et == STRING:
                for _ in range(n):
                    self.read(self.scalar(10))
            else:
                raise ValueError("nested arrays unsupported")
            return {"count": n, "offset": offset, "type": et}
        return self.scalar(t)


def inventory(path, max_string):
    with open(path, "rb") as f:
        r = Reader(f)
        if r.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        version = r.scalar(4)
        n_tensors, n_kv = r.scalar(10), r.scalar(10)
        metadata = {}
        for _ in range(n_kv):
            key = r.string(1 << 16)
            if not isinstance(key, str):
                raise ValueError("a metadata key longer than 64 KiB")
            metadata[key] = r.value(r.scalar(4), max_string, retain=key.startswith("prism.hadamard."))
        tensors = []
        for _ in range(n_tensors):
            name = r.string(1 << 16)
            nd = r.scalar(4)
            dims = [r.scalar(10) for _ in range(nd)]
            enc = r.scalar(4)
            r.scalar(10)  # offset within the data section, not part of the inventory
            tensors.append([name, dims, enc])
        return {"gguf_version": version, "directory_bytes": r.pos, "metadata": metadata, "tensors": tensors}


def fold_key(key, arch):
    return re.sub(rf"^{re.escape(arch)}\.", "<arch>.", key) if arch else key


def fold_tensor(name):
    return re.sub(r"^blk\.\d+\.", "blk.N.", name)


def shape_of(inv):
    """The architecture, the folded key set, and the folded tensor-name set
    of an inventory or fixture."""
    meta = inv["metadata"]
    arch = meta.get("general.architecture", "")
    arch = arch["value"] if isinstance(arch, dict) else arch
    keys = {fold_key(k, arch) for k in meta}
    names = {fold_tensor(t[0]) for t in inv["tensors"]}
    return arch, keys, names


def comparable(value):
    """A metadata value as compared: a scalar itself, an array its values
    when retained, else its length and element type (tokenizer tables)."""
    if isinstance(value, dict):
        if "values" in value:
            return tuple(value["values"])
        return ("array", value.get("count"), value.get("type"))
    return value


def values_of(inv, arch):
    """Folded key -> comparable value, `general.*` (names, licences, links)
    left out."""
    return {fold_key(k, arch): comparable(v) for k, v in inv["metadata"].items() if not k.startswith("general.")}


def tensor_forms(inv):
    """Folded tensor name -> the set of (shape, element type) its layers use."""
    forms = {}
    for name, shape, kind, *_ in inv["tensors"]:
        forms.setdefault(fold_tensor(name), set()).add((tuple(shape), kind))
    return forms


def show(value, limit=80):
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def against(inv, path):
    """Prints what separates the file from an existing fixture: keys and
    tensors only one side has, then scalar values and tensor shapes or
    element types that differ under a shared folded name (ten of each)."""
    with open(path) as f:
        old = json.load(f)
    new_arch, new_keys, new_names = shape_of(inv)
    old_arch, old_keys, old_names = shape_of(old)
    print(f"architecture: {new_arch} against {old_arch} ({path})")
    for title, items in (
        ("metadata keys only in the file", new_keys - old_keys),
        ("metadata keys only in the fixture", old_keys - new_keys),
        ("tensors only in the file", new_names - old_names),
        ("tensors only in the fixture", old_names - new_names),
    ):
        print(f"{title}: {len(items)}")
        for item in sorted(items):
            print(f"  {item}")
    new_values, old_values = values_of(inv, new_arch), values_of(old, old_arch)
    changed = sorted(k for k in new_values.keys() & old_values.keys() if new_values[k] != old_values[k])
    print(f"metadata values that differ: {len(changed)}")
    for k in changed[:10]:
        print(f"  {k}: {show(new_values[k])} (fixture {show(old_values[k])})")
    new_forms, old_forms = tensor_forms(inv), tensor_forms(old)
    reshaped = sorted(n for n in new_forms.keys() & old_forms.keys() if new_forms[n] != old_forms[n])
    print(f"tensors whose shape or element type differs: {len(reshaped)}")
    for n in reshaped[:10]:
        print(f"  {n}: {show(sorted(new_forms[n]))} (fixture {show(sorted(old_forms[n]))})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--out")
    ap.add_argument("--max-string", type=int, default=256)
    ap.add_argument("--against", help="an existing fixture to compare the file with")
    a = ap.parse_args()
    inv = inventory(a.file, a.max_string)
    if a.against:
        against(inv, a.against)
    text = json.dumps(inv, indent=None, separators=(", ", ": "))
    if a.out:
        # Retained numeric arrays stay on one line each (a sign vector is
        # tens of thousands of entries); everything else is indented.
        compact = {}
        for key, value in inv["metadata"].items():
            if isinstance(value, dict) and "values" in value and value["type"] != STRING:
                compact[key] = value["values"]
                value["values"] = f"@@{key}@@"
        text = json.dumps({"metadata": inv["metadata"], "tensors": inv["tensors"]}, indent=1)
        for key, values in compact.items():
            text = text.replace(json.dumps(f"@@{key}@@"), json.dumps(values, separators=(",", ":")))
        with open(a.out, "w") as f:
            f.write(text + "\n")
    print(
        f"{a.file}: GGUF v{inv['gguf_version']}, {len(inv['metadata'])} keys, {len(inv['tensors'])} tensors, directory {inv['directory_bytes']} bytes"
    )


if __name__ == "__main__":
    main()
