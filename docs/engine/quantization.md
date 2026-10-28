# CPU quantization decoding

The inference library exposes `quant.row(id, bytes, output)` through
[root.zig](../../inference/src/root.zig). It reconstructs one contiguous GGUF
row into caller-owned `f32` storage. This is a numerical reference for future
kernels; the CLI still provides structural inspection and validation only.

Implemented encodings are F32 (ID 0), F16 (1), Q4_0 (2), Q5_0 (6), Q5_1 (7), Q8_0 (8), Q3_K (11),
Q4_K (12), Q5_K (13), Q6_K (14), IQ4_NL (20), IQ3_S (21), and IQ4_XS (23).
This covers all nine encodings in the pinned Qwen artifact, plus F16, plus
Q4_0 ([MODL-08](https://github.com/tildaslashalef/nuclis/blob/v0.6.0/docs/worklog.md#modl-08--q4_0-path-and-the-qat-catalogue-entry-2026-09-12), 2026-09-12: the only weight encoding of the catalogue's Gemma 4
12B file), plus Q5_0 and Q5_1 (2026-10-28: K-quant files' fallback where
a row is not a multiple of 256, below). A [CPU matrix-vector reference](cpu-reference.md) consumes these
rows. Inspection also recognizes the BF16 layout; row decoding rejects it
explicitly.

## Interface and ownership

[decode.zig](../../inference/src/quant/decode.zig) takes borrowed input bytes
and an independently owned output slice; they must not overlap. The call retains
neither. It performs no allocation or I/O, so no allocator or Io parameter is
needed. Callers choose a bounded scratch size and supply one complete row.

The output length specifies the number of elements. A nonempty row must contain
whole storage blocks, and the input must contain exactly the corresponding
bytes. Unsupported IDs, partial blocks, truncated bytes, extra bytes, and
mismatched output lengths return typed errors before modifying output. Size
checks use division instead of potentially overflowing multiplication.
`quant.validateRow(id, byte_count, element_count)` exposes the same checks
without touching buffers, allowing numerical callers to validate complete
operations before writing their results.

Bytes are read explicitly as little-endian integers; an unaligned source is
valid. Zig's `@bitCast` interprets those bits as a floating-point number or signed
integer without changing them. `@floatCast` widens an F16 value to F32, while
`@floatFromInt` converts a signed quantized integer to its numerical F32 value.
IEEE nonfinite values are preserved or propagated; successful decoding is not
model-weight validation.

## Equations and evidence

Q8_0 stores an F16 scale followed by 32 signed bytes: each reconstructed value
is the scale multiplied by the signed byte. It is also the one format nuclis
writes at run time, as the `q8_0` attention cache: `quant/encode.zig`
`q8_0Row` sets `d = f16(max |x| / 127)` and `q = clamp(round(x / d), -127,
127)` per block, the arithmetic Metal's `nu_pack_q8_0` repeats bit for bit
([metal-backend.md § Q8_0 KV cache](metal-backend.md#q8_0-kv-cache-2026-10-15)). IQ4_NL stores an F16 scale and 16
bytes containing 32 four-bit lookup indices. The low nibbles describe the first
16 values, and the high nibbles describe the next 16. The lookup values are
nonuniform: treating IQ4_NL as ordinary signed four-bit integers is incorrect.
Q4_0 is that ordinary case: the same 18-byte block and nibble order, and the
value is `d * (q - 8)` for the four-bit code `q`, no table.

Q5_0 (22 bytes) and Q5_1 (24 bytes) add a fifth bit to Q4_0's 32 values: an
F16 `d` (Q5_1 then an F16 `m`), a little-endian 32-bit plane `qh`, and Q4_0's
sixteen nibble bytes. Value `j` below 16 is the low nibble of byte `j` with
bit `j` of `qh` as its fifth bit; value `j + 16` the high nibble with bit
`j + 16`. Q5_0 is `d * (q - 16)`, Q5_1 `d * q + m`, for the five-bit code
`q`. They reach nuclis through K-quant files: a K-quant row is a multiple of
256 values, and llama.cpp's quantizer stores a tensor whose rows are not in
a 32-value block of similar size instead (Q4_K becomes Q5_0, Q5_K Q5_1, Q6_K
Q8_0, IQ4_XS IQ4_NL). Gemma 4 26B-A4B's `ffn_down` rows (2,112 dense, 704
per expert) are such tensors, so its Q4_K_M and Q4_K_S files carry Q5_0 and
its Q5_K files Q5_1.

Q4_K and Q5_K store 256 values in eight groups of 32. Each group reconstructs
`value = (d * scale) * q - (dmin * minimum)`, where `d` and `dmin` are F16
block coefficients and `scale` and `minimum` are unsigned six-bit group
coefficients. All sixteen group coefficients fit into twelve bytes. The first
four groups use the low six bits of separate bytes; the later groups combine
nibbles with the high bits left over in those bytes.

Q4_K uses 128 bytes of four-bit values. Each 32-byte span holds one group's
values in its low nibbles and the next group's values in its high nibbles.
Q5_K adds 32 high-bit bytes before that payload. Each high-bit byte supplies
one bit to the same column in all eight groups. These layouts share a private
`kBlock(comptime fifth_bit, ...)` implementation; the compile-time argument
selects storage offsets while keeping their common equations together.

Q3_K and Q6_K use sixteen groups of sixteen values per 256-value block,
with a shared F16 multiplier and signed group scales. Q3_K packs six-bit
scale codes biased by 32 into twelve bytes, alongside two-bit value planes
and an inverted sign mask: a clear mask bit subtracts four from the value.
Q6_K stores signed byte scales directly; its four low bits and two high bits
reconstruct a code biased by 32. The F16 multiplier comes last in both layouts.
The byte-indexed helpers avoid dependence on host endianness or aligned loads.

IQ4_XS reuses the IQ4_NL nonlinear values in eight 32-value groups, each with
a six-bit scale biased by 32. Scale low nibbles occupy four bytes; their upper
two bits occupy a little-endian 16-bit field. IQ3_S uses a 512-entry codebook:
each nine-bit index selects four positive magnitudes, then separate per-value
sign bits and a group multiplier `d * (1 + 2 * scale)` reconstruct the values.
Eight groups of 32 values have four-bit scales. Its codebook is fixed format
data, not learned model weights. The attributed
[table](../../inference/src/quant/iq3-grid.zig) is regenerated from the pinned
reference header; integer shifts extract components without host-endian aliasing.

PQ2_0 and PTQ1_0 (ids 142 and 143) are the PrismML fork's group-128 ternary
blocks for Bonsai 2 27B: `w = d * t` with `t` in {-1, 0, +1} and one F16
scale per 128 weights. PQ2_0 stores the scale then 32 bytes of four two-bit
codes each, low bits first, value `d * (q - 1)` (code 3 decodes to +2 as
the reference does). PTQ1_0 stores 24 base-3 bytes, two four-trit tail
bytes, then the scale: digit `n` of a byte is `((byte * 3^n mod 256) * 3)
>> 8`, the 24 bytes are runs of 16 and 8 emitted digit-major, and the tail
bytes hold four trits at the leading positions. BF16 (id 30) widens the
high half of a single. The contracts, the fork's revision, and the
rotation these files also need are in [bonsai.md](../models/bonsai.md#encodings-from-the-forks-ggml-commonh-and-ggml-quantsc).

The block layouts and equations are facts of the GGML storage formats, learned
from the pinned llama.cpp
[decoder source](https://github.com/ggml-org/llama.cpp/blob/7620399f58aebfd2196b74021f9581bcf7218cb9/ggml/src/ggml-quants.c)
and [block definitions](https://github.com/ggml-org/llama.cpp/blob/7620399f58aebfd2196b74021f9581bcf7218cb9/ggml/src/ggml-common.h)
and implemented independently here; the ternary blocks from the PrismML
fork's at `5d80cff0b8cb9f2bf823cfc4e71e3abb97f290d6` the same way. The two
lookup tables (IQ3_S codebook, IQ4_NL values) are format constants whose
provenance is recorded in [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md).

Colocated tests cover hand-calculated blocks, signed extremes, every IQ4 lookup
index, half-row ordering, independent block scales, F16 subnormals/signed zero/
infinity/NaN, little-endian F32 values, and invalid inputs. A small committed
[fixture](../../inference/src/quant/fixtures/simple.json) contains eight Q8_0,
eight IQ4_NL, and (since [MODL-08](https://github.com/tildaslashalef/nuclis/blob/v0.6.0/docs/worklog.md#modl-08--q4_0-path-and-the-qat-catalogue-entry-2026-09-12)) eight Q4_0 blocks with outputs from the pinned
C CPU decoders. The Zig results match exactly; these simple equations do not
require a tolerance. The Q8 fixture covers all 256 byte representations; the
Q4_0 and IQ4_NL payloads are the same bytes, so the two decoders are pinned
against each other's ordering as well. Scales include zero, negative values,
the smallest positive F16 subnormal, and the largest finite F16.

The separate [K-format fixture](../../inference/src/quant/fixtures/k-affine.json)
contains four blocks each of Q4_K and Q5_K, generated from packed coefficient
bytes and nonuniform payloads. It checks exact agreement with the pinned C
functions across multiple blocks, including negative, zero, and subnormal block
scales. Independent hand-calculated tests check all eight groups' six-bit scale
and minimum fields, affine subtraction, nibble order, and every Q5_K high-bit
position. Invalid-size tests apply to both new formats and leave output intact.

The [signed K-format fixture](../../inference/src/quant/fixtures/k-signed.json)
adds five blocks each of Q3_K and Q6_K with exact pinned CPU outputs. Independent
tests exercise biased scale packing, Q3_K's inverted sign mask and all two-bit
planes, Q6_K's quarter ordering, and signed byte scale extremes. Both formats
share the invalid-input tests, including extra complete blocks and partial rows.

The [IQ fixture](../../inference/src/quant/fixtures/iq.json) covers every one
of IQ3_S's 512 grid entries with nonzero scales, plus mixed high-index bits and
a zero-scale block. IQ4_XS
has five varied blocks. Both match pinned CPU outputs exactly. Hand-calculated
tests isolate IQ3_S component/sign ordering and odd scales, and IQ4_XS six-bit
scales and half-row order. All fixture comparisons use deliberately offset byte
slices to exercise unaligned input; invalid-size tests cover both new encodings.

The [ternary fixture](../../inference/src/quant/fixtures/ternary.json) is
produced by the fork's own `dequantize_row_pq2_0` and `dequantize_row_ptq1_0`
(`scripts/quant-fixtures.py --prism-checkout`, its `revision` the fork's
pin): eight blocks of each with arbitrary payload bytes and the eight scales
above, so non-canonical codes reproduce the reference's fixed-point
arithmetic too, plus all-zero and all-one blocks, an every-code PQ2_0 block,
and a canonically packed PTQ1_0 block. Hand-calculated tests cover PQ2_0's
code order within bytes and the +2 code, PTQ1_0's digit-major runs and
four-trit tail from a canonical packing, and BF16's widening of a subnormal,
signed zero, infinity, and NaN.

Q2_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, and IQ3_XXS (2026-10-14) were
validated without an external decoder: hand-calculated tests pin every field
of each block (scale and minimum nibbles, grid indices and their high bits,
the 7-bit sign indices with parity as the eighth sign, the IQ1 shift of ±1/8
and its sign, IQ1_M's scale rebuilt from four nibbles), and the
cross-encoding check compares whole models' rows with an already-validated
encoding of the same weights ([saluki.md § Validation](../models/saluki.md#validation-without-an-external-reference)).
Their codebooks come from `scripts/iq-codebooks.py`. The
[codebook fixture](../../inference/src/quant/fixtures/codebook.json) holds
two real blocks of each from Saluki 27B, written by `zig build
test-cross-encoding -- --fixture <out> <saluki.gguf>` with the CPU
decoders' values: a regression pin for the CPU and the rows `metal-check`
holds the GPU decoders to, bit for bit.

Q5_0 and Q5_1 (2026-10-28) were validated without an external decoder and
without the cross-encoding check. Hand-calculated tests pin both halves of
the plane, every code on each half-row, the bias of 16 and Q5_1's offset.
The [five-bit fixture](../../inference/src/quant/fixtures/five-bit.json)
(eight blocks of each, written by `scripts/five-bit-fixtures.py` from the
equations above, with negative, subnormal and large scales) holds the CPU
decoder and `metal-check` holds the GPU to it: exact decode through every
matvec and matmul path, and the 26B-A4B expert chains with a Q5_0 or Q5_1
down projection against the F64 CPU reference. End to end, a Gemma 4
26B-A4B Q4_K_M file with 32 Q5_0 `ffn_down` tensors gave the same greedy
text on Metal and the CPU, and the same top five after a 368-token prefill
(final logits max abs 5.2e-2, relative RMS 1.9e-3, inside the family's
chunked-prefill bounds).

Default tests consume only the committed fixtures and need no external checkout or model:

```sh
zig build test
```

To regenerate on macOS, build the pinned reference following
[llama-cpp.md](../benchmarks/llama-cpp.md), then run:

```sh
python3 scripts/quant-fixtures.py
```

The default checkout is `.reference/llama.cpp`, resolved relative to
the repository rather than the working directory. An optional positional path
selects another checkout. The generator checks the checkout revision and calls
the built `libggml-base` CPU functions through Python `ctypes`; it does not load a model or initialize
Metal. Rebuild that library from the pinned checkout before regeneration; a Git
revision check cannot prove the provenance of a stale local binary.
