<!-- kind: reference -->
# Safetensors

nuclis reads safetensors checkpoints as a second artifact format beside
GGUF: `nuclis model pull` fetches a set with its configuration and
tokenizer files ([catalogue.md § Safetensors artifacts](../models/catalogue.md#safetensors-artifacts)),
and `inference/src/formats/safetensors.zig` reads and maps it. The families
that run from one bind through it: Laya's typed-decision encoder
([laya.md](../models/laya.md)), Clef's head
([clef-flash.md](../models/clef-flash.md)), Qwen3-TTS
([qwen3-tts.md](../models/qwen3-tts.md)), and Nemotron ASR
([nemotron-asr.md](../models/nemotron-asr.md)); no text-generation family
does. The format contract and independent reads of real files are the checks.

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

## Evidence

The reader's counts and decoded values were checked against an independent
read of Laya's and SmolLM2's files, and real headers of five other
repositories parse ([record, 2026-09-27](../benchmarks/record/safetensors.md#evidence-2026-09-27)).
