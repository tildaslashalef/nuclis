# CPU quantization decoding

The inference library exposes `quant.row(id, bytes, output)` through
[root.zig](../../inference/src/root.zig). It reconstructs one contiguous GGUF
row into caller-owned `f32` storage. This is a numerical reference for future
kernels; the CLI still provides structural inspection and validation only.

Implemented encodings are F32 (ID 0), F16 (1), Q4_0 (2), Q8_0 (8), Q3_K (11),
Q4_K (12), Q5_K (13), Q6_K (14), IQ4_NL (20), IQ3_S (21), and IQ4_XS (23).
This covers all nine encodings in the pinned Qwen artifact, plus F16, plus
Q4_0 (MODL-08, 2026-09-12: the only weight encoding of the catalogue's Gemma 4
12B file). A [CPU matrix-vector reference](cpu-reference.md) consumes these
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
is the scale multiplied by the signed byte. IQ4_NL stores an F16 scale and 16
bytes containing 32 four-bit lookup indices. The low nibbles describe the first
16 values, and the high nibbles describe the next 16. The lookup values are
nonuniform: treating IQ4_NL as ordinary signed four-bit integers is incorrect.
Q4_0 is that ordinary case: the same 18-byte block and nibble order, and the
value is `d * (q - 8)` for the four-bit code `q`, no table.

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

The block layouts and equations are facts of the GGML storage formats, learned
from the pinned llama.cpp
[decoder source](https://github.com/ggml-org/llama.cpp/blob/7620399f58aebfd2196b74021f9581bcf7218cb9/ggml/src/ggml-quants.c)
and [block definitions](https://github.com/ggml-org/llama.cpp/blob/7620399f58aebfd2196b74021f9581bcf7218cb9/ggml/src/ggml-common.h)
and implemented independently here. The two lookup tables (IQ3_S codebook,
IQ4_NL values) are format constants whose provenance is recorded in
[THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md).

Colocated tests cover hand-calculated blocks, signed extremes, every IQ4 lookup
index, half-row ordering, independent block scales, F16 subnormals/signed zero/
infinity/NaN, little-endian F32 values, and invalid inputs. A small committed
[fixture](../../inference/src/quant/fixtures/simple.json) contains eight Q8_0,
eight IQ4_NL, and (since MODL-08) eight Q4_0 blocks with outputs from the pinned
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

Default tests consume only the committed fixtures and need no external checkout or model:

```sh
zig build test --global-cache-dir .zig-cache/global
```

To regenerate on macOS, build the pinned reference following
[reference-baseline.md](reference-baseline.md), then run:

```sh
python3 scripts/quant-fixtures.py
```

The default checkout is `.zig-cache/reference/llama.cpp`, resolved relative to
the repository rather than the working directory. An optional positional path
selects another checkout. The generator checks the checkout revision and calls
the built `libggml-base` CPU functions through Python `ctypes`; it does not load a model or initialize
Metal. Rebuild that library from the pinned checkout before regeneration; a Git
revision check cannot prove the provenance of a stale local binary.
