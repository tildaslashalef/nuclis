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
target artifact). Nothing is implemented yet; start with MODL-28.

## Order

| Unit | Title | Sessions |
| --- | --- | --- |
| MODL-28 | Safetensors sets through `huggingface/` and `nuclis model pull`, pinned like GGUF | one |
| MODL-29 | A generic safetensors loader (`inference/src/formats/safetensors.zig`) and `nuclis inspect` on it | one |

## MODL-28 — Safetensors sets through the Hub client and `model pull`

Facts (checked 2026-09-27 on `convaiinnovations/laya` and
`HuggingFaceTB/SmolLM2-135M`): `?blobs=true` gives every sibling `size`
and `blobId`; LFS files add `lfs.sha256`; plain files (config, small
tokenizers) have only `blobId`, which is the git blob SHA-1
(`sha1("blob <size>\0" ++ bytes)`, checked with `git hash-object`). A
plain file's `resolve` answers 307 to `/api/resolve-cache/...`, whose
range GET is a 206: the existing direct path fetches it.

**Artifact sets.** A GGUF artifact is unchanged (a file, or a
`-NNNNN-of-MMMMM.gguf` shard set). A safetensors artifact is a single
`.safetensors` file or a `<prefix>-NNNNN-of-MMMMM.safetensors` shard set,
plus its *support files*: files with extension `.json`, `.txt`, `.model`,
`.jinja` in the artifact's directory and its subdirectories, excluding
subdirectories that hold other `.safetensors` files (Laya's `multilingual/`
and `typed-decisions/` are separate artifacts) and `*.safetensors.index.json`
files whose prefix is not the set's. Never `.py`, `.bin`, `.pt`, `.md`,
images.

`huggingface/src/hub.zig`:
- `File`: `sha256: ?[32]u8` (LFS) and `git_oid: [20]u8`; a file needs one
  of the two to be verifiable (`MissingChecksum` otherwise).
- `Catalog.files` keeps weights only (`.gguf`, `.safetensors`), so choice
  lists and companion lookups stay weights; new `Catalog.support`.
- `validate`: a filename ends in `.gguf` or `.safetensors`, unless
  `Request.exact` (below), which accepts any listed file.
- `select(a, catalog, filename)`: groups weights into artifacts (shard
  parsing generalized over the extension); a repo-only request with one
  artifact selects it, several return null; a safetensors selection
  returns its shards, then its support files.
- `Request.exact: bool = false`: download only the named file, no shard or
  set expansion (`model pull` has already expanded; this also removes the
  N² re-verification of GGUF shard sets that per-job downloads caused).

`huggingface/src/root.zig`: `FileSink` checks the container by extension
(GGUF magic; safetensors header length `8 + n <= size` and byte 8 `{`;
support files unchecked); hashes SHA-256 always and git SHA-1 when the
catalogue has no SHA-256; `publish`/`verifyExisting` verify whichever
digest is known. `LocalFile.sha256` stays the computed SHA-256.
`hf-downloader` text says "model files", not "GGUF files".

`src/model.zig`:
- `Role.support` (config, tokenizer, index); safetensors weights are
  `main`. `roleFromHeader` only opens `.gguf`.
- `Sidecar.git_blob: ?[]const u8 = null` (40 hex) for plain files; the
  step-2 conflict check compares it when the Hub gives no SHA-256.
- Raw-repository pulls fetch every selected file with `exact = true`.
- `--register` on a safetensors artifact is refused (`NotRunnable`: no
  runtime family reads safetensors yet).
- `ls` lists `.safetensors` beside `.gguf` (support files have sidecars
  but are not listed rows).
- `src/help.zig`, `huggingface/README.md`, `docs/development.md` (models
  layout) updated.

**Checks.** Unit tests: catalogue parse of a mixed fixture (LFS, plain,
code, images), artifact grouping (Laya's three sets, a shard set, Mistral's
`consolidated.safetensors` beside shards → selection required), git-blob
verification, sidecar round trip with and without `git_blob`. Live, fresh
binary: `nuclis model pull convaiinnovations/laya --file model.safetensors`
(root set: weights + `rl_agent_config.json`, `encoder/`, `tokenizer/`,
`eval/*.json`), a second pull reuses every file, `nuclis model ls` shows
it; `nuclis model pull HuggingFaceTB/SmolLM2-135M` (sole artifact);
an existing GGUF pull still reuses its verified files. `make check`. No
Metal tier (no numerical change).

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
