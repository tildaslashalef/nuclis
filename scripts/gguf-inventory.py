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
"""
import argparse, hashlib, json, struct, sys

TYPES = {0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2), 4: ("I", 4), 5: ("i", 4), 6: ("f", 4), 7: ("?", 1), 10: ("Q", 8), 11: ("q", 8), 12: ("d", 8)}
STRING, ARRAY = 8, 9

class Reader:
    def __init__(self, f):
        self.f, self.pos = f, 0
    def read(self, n):
        b = self.f.read(n)
        if len(b) != n: raise EOFError("truncated directory")
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
        return {"length": n, "offset": offset, "sha256": hashlib.sha256(raw).hexdigest(), "head": raw[:64].decode("utf-8", "replace")}
    def value(self, t, max_string, retain=False):
        if t == STRING: return self.string(max_string)
        if t == ARRAY:
            et, n = self.scalar(4), self.scalar(10)
            offset = self.pos
            if retain and et == STRING:
                return {"count": n, "offset": offset, "type": et, "values": [self.string(1 << 16) for _ in range(n)]}
            if et in TYPES and (n <= 64 or retain):
                return {"count": n, "offset": offset, "type": et, "values": [int(self.scalar(et)) if et == 7 else self.scalar(et) for _ in range(n)]}
            if et in TYPES:
                self.read(n * TYPES[et][1])
            elif et == STRING:
                for _ in range(n): self.read(self.scalar(10))
            else:
                raise ValueError("nested arrays unsupported")
            return {"count": n, "offset": offset, "type": et}
        return self.scalar(t)

def inventory(path, max_string):
    with open(path, "rb") as f:
        r = Reader(f)
        if r.read(4) != b"GGUF": raise ValueError("not a GGUF file")
        version = r.scalar(4)
        n_tensors, n_kv = r.scalar(10), r.scalar(10)
        metadata = {}
        for _ in range(n_kv):
            key = r.string(1 << 16)
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file"); ap.add_argument("--out"); ap.add_argument("--max-string", type=int, default=256)
    a = ap.parse_args()
    inv = inventory(a.file, a.max_string)
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
        with open(a.out, "w") as f: f.write(text + "\n")
    print(f"{a.file}: GGUF v{inv['gguf_version']}, {len(inv['metadata'])} keys, {len(inv['tensors'])} tensors, directory {inv['directory_bytes']} bytes")

if __name__ == "__main__":
    main()
