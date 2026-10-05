# Safetensors

nuclis reads safetensors checkpoints as a second artifact format beside
GGUF: `nuclis model pull` fetches a set with its configuration and
tokenizer files ([catalogue.md § Safetensors artifacts](../models/catalogue.md#safetensors-artifacts)),
and `inference/src/formats/safetensors.zig` reads and maps it. No model
family runs one yet; this is the container layer an experimental family
(Laya, a typed-decision encoder, planned in `TODO.md`) would bind to. There is no llama.cpp oracle here: the format contract and
independent reads of real files are the checks.

## The format

The contract is the format's own specification (huggingface/safetensors,
Apache-2.0):

1. 8 bytes: `n`, the header length, little-endian u64.
2. `n` bytes of UTF-8 JSON, beginning with `{`, possibly padded with
   spaces. Each key is a tensor name mapping to
   `{"dtype": "BF16", "shape": [576], "data_offsets": [begin, end]}`;
   the optional `__metadata__` key maps strings to strings.
3. The byte buffer. Offsets are relative to its start (`8 + n`), not to
   the file. Tensors are row-major, outermost dimension first (the reverse
   of GGUF's order), little-endian, and **need not be aligned**: Laya
   stores an F16 tensor at offset 12.

A sharded checkpoint adds `<prefix>.safetensors.index.json`, whose
`weight_map` names each tensor's shard file.

## What the reader checks

`parse` reads the length and the header only, never weights:

| Check | Error |
| --- | --- |
| header length over 100,000,000 bytes (the format's bound), before allocating | `LimitExceeded` |
| header length under 2 or past the file | `InvalidHeader` |
| not an object, trailing tokens, a tensor entry without exactly `dtype`, `shape`, `data_offsets`, non-integer or negative numbers, non-string metadata | `InvalidHeader` |
| a tensor name twice | `DuplicateTensor` |
| a dtype outside BOOL, U8, I8, F8_E5M2, F8_E4M3, F8_E8M0, I16, U16, F16, BF16, I32, U32, F32, I64, U64, F64 (the sub-byte `F4`/`F6_*` and complex `C64` included) | `UnsupportedDtype` |
| `end - begin` differs from shape × dtype size, `end < begin`, rank over 8 | `InvalidShape` |
| shape product overflowing u64 | `Overflow` |
| sorted by offset, the tensors must tile the buffer exactly | `UnindexedBytes` (a hole or trailing bytes), `OverlappingTensors`, `TensorOutOfBounds` |

A per-tensor rejection names the tensor through `Rejection`, as the GGUF
reader does. Rank 0 (a scalar, one element) and a zero dimension (an empty
tensor, zero bytes) are valid.

## The checkpoint

`Checkpoint.open(gpa, io, path, limits)` takes a `.safetensors` file, an
`*.safetensors.index.json`, or a directory, which resolves to
`model.safetensors.index.json`, then `model.safetensors`, then a sole index,
then a sole `.safetensors` file (`NoCheckpoint`, `AmbiguousCheckpoint`
otherwise). An index's shard names must be plain file names beside it
(`InvalidIndex` for a path), and every shard's tensors must be exactly the
ones the index places there (`IndexMismatch`). Each shard is mapped
read-only; `get(name)` returns a `Ref` (descriptor and bytes) borrowing the
mapping, valid until `deinit`. `Ref.decode(first, out)` reads F32, F16, and
BF16 into f32 with unaligned little-endian loads; other dtypes are
`UnsupportedDtype` (FP8 and packed quantizations are bytes whose meaning a
family supplies).

## Commands

```sh
nuclis inspect --model ~/.nuclis/models/convaiinnovations/laya          # a directory
nuclis inspect --model <file>.safetensors --json                        # shards, dtypes, metadata
nuclis inspect --model <checkpoint> --tensor type_emb.weight            # one tensor, its first 8 values
```

`nuclis validate` on a safetensors path is `NotRunnable`.

## Evidence (MODL-29, 2026-09-27)

- Laya (`convaiinnovations/laya` at `55cf4c4e`, root set): 206 tensors,
  205 F16 and 1 F32, header 21,536 bytes, 842,587,666 tensor bytes in a
  842,609,210-byte file, the same counts a Python read of the remote
  header gave. SmolLM2-135M: 272 BF16 tensors.
- Decoded values equal an independent Python `struct` read: Laya
  `temperature` (F32) `1 1 1`, `type_emb.weight` (F16) `-1.4257812
  -0.22570801 0.2331543 -1.34375 …`, SmolLM2 `model.norm.weight` (BF16)
  `1.75 1.8046875 1.765625 1.59375 …`.
- Real headers of five other repositories, written into sparse files of
  the true size so the tiling check runs against the real byte count, all
  parse: `Qwen/Qwen3-0.6B` (311 BF16), `Qwen/Qwen3-0.6B-FP8` (196 F8_E4M3,
  311 BF16), `openai/gpt-oss-20b` shard 0 (40 U8 MXFP4 blocks and scales,
  156 BF16), `answerdotai/ModernBERT-large` (173 F32), and
  `Qwen/Qwen2.5-7B-Instruct` shard 1 of 4 (82 BF16).
