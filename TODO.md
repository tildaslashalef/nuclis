# TODO — active plan

This file is the only progress tracker. It holds **unfinished work only**:
where we are, what is next, and the design of each remaining unit. When a
unit closes, its outcome moves to
[docs/engineering-log.md](docs/engineering-log.md) and
its section is deleted here; when the last unit closes, this file is emptied
back to this header. Requirements live in [docs/spec.md](docs/spec.md); the
engine map in [docs/architecture.md](docs/architecture.md); how to build,
test, and measure in [docs/development.md](docs/development.md).

Session protocol (also in [AGENTS.md](AGENTS.md)): read this file first. If
it lists work, summarize *Where we are* and ask the user how to continue. If
it is empty, ask what to work on and write the agreed plan here.

## Where we are

Theme agreed 2026-09-27 (user): safetensors as a second artifact format,
a new experiment with no reference comparison (the Jev-like decision work
in [docs/research/typesafe-jev.md](docs/research/typesafe-jev.md) is the
motivation; Laya, `convaiinnovations/laya` at `55cf4c4e`, is the first
target artifact). MODL-28 closed the same day
([log](docs/engineering-log.md#modl-28--safetensors-sets-through-the-hub-client-and-model-pull-2026-09-27)):
Laya's root set (`model.safetensors` + support files) and
`HuggingFaceTB/SmolLM2-135M` are pulled under `~/.nuclis/models`, ready as
MODL-29's live inputs. Next: MODL-29.

## Order

| Unit | Title | Sessions |
| --- | --- | --- |
| MODL-29 | A generic safetensors loader (`inference/src/formats/safetensors.zig`) and `nuclis inspect` on it | one |

## MODL-29 — A generic safetensors loader

Format facts (huggingface/safetensors, Apache-2.0): 8-byte little-endian
`n`, then `n` bytes of UTF-8 JSON (may be space-padded) mapping tensor
names to `{dtype, shape, data_offsets: [begin, end]}` relative to the
byte buffer at `8 + n`, plus an optional `__metadata__` string→string
map. The buffer is fully indexed: sorted by offset, tensors are contiguous
without holes or overlap and end at the file's end. Offsets need not be
aligned (Laya: an F16 tensor at offset 12). Rank 0 is a scalar; a zero
dimension is an empty tensor.

`inference/src/formats/safetensors.zig`:
- `Limits` (header 100,000,000 bytes, the format's own bound; tensors
  100,000; metadata 16,384; rank 8), `Dtype` (BOOL, U8, I8, F8_E4M3,
  F8_E5M2, F8_E8M0, I16, U16, F16, BF16, I32, U32, F32, I64, U64, F64;
  anything else `UnsupportedDtype`, named through a `Rejection` like
  GGUF's), `Tensor` (name, dtype, shape, offset, bytes, elements),
  `Document` (arena-owned; `find`, `metadataValue`), `parse(gpa, reader,
  file_bytes, limits)`, `open(gpa, io, path, limits)`. Errors typed:
  duplicate names, bad JSON shape, byte count ≠ shape × dtype size,
  holes, overlaps, out of bounds.
- `Checkpoint`: opens a single file, or an index
  (`*.safetensors.index.json`: `weight_map` names each tensor's shard;
  every shard's tensors must match the map exactly), maps every shard
  read-only, and exposes `tensor(name)` → borrowed bytes and descriptor,
  and `decode(name, out []f32)` for F32, F16, BF16 (unaligned
  little-endian reads). Mappings outlive every borrowed view.
- Exported from `inference/src/root.zig` as `safetensors`.

`src/cli.zig` + `src/inspect.zig`: `nuclis inspect --model <x>` where
`<x>` is a `.safetensors` file, an index, or a directory holding one set:
tensor count, dtype histogram (tensors, elements, bytes), metadata count,
shards, header bytes; `--json` from the same snapshot. `validate` on a
safetensors path is a typed error naming the missing adapter.

Docs: `docs/reference/safetensors.md` (format summary, validation rules,
the loader's ownership), `docs/architecture.md` (formats row),
`THIRD_PARTY_NOTICES.md` (dtype names from the format specification).

**Checks.** Unit tests from wire-format fixtures: a valid file, padded
header, scalar and empty tensors, each rejection (bad length, bad JSON,
duplicate, unknown dtype, size mismatch, hole, overlap, out of bounds,
limits before allocation), every allocation failure, an index with a
missing or extra tensor. Live: `nuclis inspect --model` on the pulled
Laya file (206 tensors: 205 F16, 1 F32) and on SmolLM2-135M; decode a
row of `temperature` equal to Laya's header values read independently
with Python. `make check`. No Metal tier (nothing in the runtime calls
the loader yet).
