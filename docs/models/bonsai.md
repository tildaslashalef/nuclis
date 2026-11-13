<!-- kind: card -->
# Bonsai 2 27B

Prism ML's ternary re-encoding of Qwen3.8-27B
(`prism-ml/Ternary-Bonsai-2-27B-gguf`, `general.architecture = qwen35`).
It has no catalogue entry and no gates: it binds and runs by repository
and file name, and the tiny `bonsai` variant keeps its load and forward
paths under `make test-tiny` ([README.md § Tiny models](README.md#tiny-models)).
The Qwen adapter, tokenizer, and runtime schedule apply unchanged; the work
sits below the adapter: two weight encodings, a BF16 pair per DeltaNet
layer, and a Hadamard rotation of every projection input that the
converter folded into the stored weights. The bring-up against the PrismML
fork is [the record](../benchmarks/record/bonsai.md).

## Architecture

Qwen3.8-27B's ([qwen3.8.md](qwen3.8.md)), with these differences:

| Fact | Bonsai 2 | Qwen3.8-27B |
| --- | --- | --- |
| Blocks | 64, no `nextn_predict_layers` (no MTP block) | 65 with one prediction block |
| Matrices | ternary, group 128 (PQ2_0 or PTQ1_0) | K-quants and IQ quants |
| `ssm_alpha`, `ssm_beta` | BF16 | Q8_0 |
| Projection inputs | rotated by a signed Hadamard transform | plain |
| Output head | untied, rotated | untied |

## Files and encodings

Two files at commit `6ed5e12b`: `Ternary-Bonsai-2-27B-PQ2_0.gguf` (7.2 GB)
and `Ternary-Bonsai-2-27B-PTQ1_0.gguf` (5.9 GB, the same scales and trits
in a denser packing), with a Q8_0 projector that the Qwen3-VL adapter
binds. The Qwen adapter's executable set includes ids 142, 143, and 30.

## Encodings (from the fork's `ggml-common.h` and `ggml-quants.c`)

Both are group-128 ternary: `w = d · t` with `t ∈ {−1, 0, +1}` and one F16
scale `d` per 128 weights, decoded in
[`quant/decode.zig`](../../inference/src/quant/decode.zig) and pinned
bit-identical to the fork's decoders by
[`ternary.json`](../../inference/src/quant/fixtures/ternary.json)
(arbitrary payload bytes, so non-canonical codes reproduce the reference
arithmetic too).

- **PQ2_0** (id 142): 34 bytes, `ggml_half d` then `uint8_t qs[32]`.
  Element `j` is the two bits at `(j % 4) * 2` of byte `j / 4`, low bits
  first; the value is `d · (q − 1)`, so code 3 decodes to +2 (a ternary
  file never stores it). 2.125 bits per weight.
- **PTQ1_0** (id 143): 28 bytes, `uint8_t qs[24]`, `uint8_t qh[2]`, then
  `ggml_half d`. TQ1_0's trit packing at group 128: a byte holds five trits
  in base 3 with the leading digit in the top bits; digit `n` is
  `((byte · 3ⁿ mod 256) · 3) >> 8` and the value `d · (digit − 1)`. The 24
  bytes are two runs of 16 and 8 bytes, each **digit-major** (all first
  digits of the run, then all second digits, …: 80 + 40 values); the two
  `qh` bytes hold four trits each, digit-major too (8 values). 1.75 bits
  per weight.
- **BF16** (id 30): the high half of an IEEE single, widened without
  rounding.

## Rotation (`prism.hadamard.*`, as the fork's loader reads it)

The converter rotated every projection's input basis by
`R = (1/√n) · Hₙ · S` (n = 1024, S a diagonal of ±1 signs), folded `R` into
the stored weights, and stored the embedding table in the rotated basis.
So inference applies the transform to the **activation** before each
rotated projection, and the inverse to the embedding row after lookup.

| Key | Value | Meaning |
| --- | --- | --- |
| `version` | 1 | the only version accepted |
| `block_size` | 1024 | the transform acts per 1024 consecutive input elements |
| `transform` | `normalized-sylvester-walsh-hadamard` | `H[row][col] = (1/√1024) · (−1)^popcount(row & col)`, so `H⁻¹ = H` |
| `axis` | `input-last-dimension` | along the projection's input |
| `sign_mode` | `explicit` | a stored sign vector per width |
| `sign_widths`, `sign_values` | [5120, 6144, 17408]; 28,672 int32 of ±1 | the three vectors concatenated in that order |
| `weight_names` | 401 strings | every matrix but the embedding and `ssm_alpha` / `ssm_beta` |
| `inverse_weight_names` | [`token_embd.weight`] | the row lookup gets the inverse |
| `gdn_v_grouped` | true | `ssm_out`'s input is regathered first (below) |

**Forward:** rows of 1024, `x' = x ⊙ signs[width]`, then `y = H · x'`;
the projection runs on the stored weights. Applied once per activation:
the normed residual before the mixer and before the FFN pair, the mixer
output before `ssm_out` / `attn_output`, the FFN hidden before `ffn_down`,
and the head's input. `ssm_alpha`, `ssm_beta`, and every norm see the
untransformed activation. **Inverse after lookup:** `h = signs[5120] ⊙ (H · z)`.

**`gdn_v_grouped`:** the mixer emits the 48 value heads tiled (value head
`rep · 16 + nk`), the fold used the grouped order (`nk · 3 + rep`), so
`ssm_out`'s input is gathered into the grouped order before the signs.

`qwen35.zig` `validateRotation` pins exactly this contract: any other
value, any other `prism.hadamard.*` key, a weight name outside the set or
claimed twice, or rotation keys without the version are
`UnsupportedConfiguration`. The transform is
[`cpu/hadamard.zig`](../../inference/src/backends/cpu/hadamard.zig) on the
CPU and `Backend.hadamard` (`nu_hadamard`) on Metal, plus a row-gather
kernel for the regathering
([metal-kernels.md § The rotation on the Qwen plan](../engine/metal-kernels.md#the-rotation-on-the-qwen-plan)).
A plain Qwen file takes none of these paths. As separate dispatches the
transform is 2.1 % of a decode token, so it is not fused into the loads.

## Chat template

The file's template (8,952 bytes, `c3cf9e34…`) is **the upstream Qwen3.8
template**, not the pinned `qwen38` profile's, which carries Unsloth's
fixes. They render every conversation the upstream template accepts
identically; the upstream one refuses a leading `system`, `developer`,
`system` history that `qwen38` merges. So the digest is not an alias, and
the file runs under `qwen38` forced: `--prompt-profile qwen38` by path, or
`config init` / `config sync`, which force the family's `qwen38`. The
header's sampling hints (1.0 / 0.95 / 20) are the profile's.

## What is verified

No gates. At bring-up the CPU reference and the Metal plan matched the
fork's traces in both cache precisions, and PTQ1_0 matched the PQ2_0
traces at the same bounds
([record, 2026-09-18](../benchmarks/record/bonsai.md#metal-plan-2026-09-18)).

## Numbers

Metal, F16 cache, 22-token prompt, 64 output tokens (M4 Pro, 2026-09-18;
[record](../benchmarks/record/bonsai.md#metal-plan-2026-09-18)):

| File | Decode tok/s | Prefill tok/s |
| --- | ---: | ---: |
| PQ2_0 | 12.87 | 39.1 |
| PTQ1_0 | 13.05 | 39.3 |
| Qwen3.8-27B on the same plan | 10.42 | 39.3 |

## Known limits

- The ternary matvecs move about 100 GB/s of weights (PQ2_0 119, PTQ1_0 88
  on the head) where the Q4_K kernels move 170–200: they are at the kernel
  set's multiply rate, so the rate the byte count promises (two to three
  times Qwen's) needs a different ternary arithmetic.
- The fork's other formats (group-64 `Q2_0`, TQ1_0 at group 256) are not
  read.
