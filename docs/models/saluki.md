# Underdog Saluki 27B: Qwen3.8-27B in 2.26 bits per weight

Underdog's 2-bit re-encoding of Qwen3.8-27B (`general.architecture =
qwen35`), built on ISTA DASLab's GSQ-RCO quantization. The pinned Qwen
adapter, tokenizer, template, and runtime schedule apply unchanged; the
work sits below the adapter, in seven weight encodings nuclis did not
implement. It is the first model brought up without an external oracle:
its validation is nuclis's own (§ [Validation](#validation-without-an-external-reference)).
Everything here was read on 2026-10-14 from the pulled file with
`scripts/gguf-inventory.py` (the committed inventory is
[`fixtures/saluki-27b.json`](../../inference/src/models/fixtures/saluki-27b.json))
and by range reads of the comparison files' headers and tensors.

**Goal.** Decode at ≥ 22 tok/s at 512 tokens on the M4 Pro: the 2-bit
matvecs at the ~180 GB/s the Q4_K path sustains, over 7.61 GB per token.
llama.cpp reaches 12.46 (§ [The llama.cpp record](#the-llamacpp-record-2026-10-14)).

## Artifacts

| | Main | Projector |
| --- | --- | --- |
| Repository | `ConwayResearch/Underdog-Saluki-27B-1.0` (Apache-2.0) | same |
| Commit | `4f60eba09f4b6db63c72915d328d97e1d93edb8c` | same |
| File | `Underdog-Saluki-27B-1.0-IQ2-mix.gguf` | `mmproj-Underdog-Saluki-27B-1.0-Q8_0.gguf` |
| Size | 7,898,369,152 B | 629,246,976 B |
| SHA-256 | `4a673518b11b1c4445f9b9a9d3c40356f5ba39f6a6260dcb25f3fbfd475d9efb` | `2e0b88b680a45a58eea666b9776a89601b16f4ef575b871a686d82ee65938756` |
| Pulled | 2026-10-14, `nuclis model pull … --file … --revision 4f60eba0…` | same, `--role mmproj` |

The repository also has `mmproj-…-F16.gguf` (927,606,912 B, SHA-256
`f3aa5781…`, not pulled). The Q8_0 projector has the layout of Bonsai 2's
(Q8_0 matrices, F16 `ffn_down`, F32 vectors;
[vision.md § The Qwen3-VL projector](../engine/vision.md#the-qwen3-vl-projector-2026-09-22)),
which the Qwen3-VL adapter already binds. No catalogue entry: the
[catalogue](catalogue.md) decision waits for the comparison with
Unsloth's 2-bit files on nuclis.

## Metadata

Equal to the pinned Qwen3.8 file
([fixtures/qwen35-27b.json](../../inference/src/models/fixtures/qwen35-27b.json))
in every `qwen35.*` dimension, with two exceptions: `qwen35.block_count`
is 64 (65 there) and `qwen35.nextn_predict_layers` is absent. The file
has no MTP head, as ISTA's IQ2_XS it was built from has none (ISTA ships
heads only in separate `-mtp` files). `general.name` is `Qwen3.8-27B
GSQ-RCO` and `general.quantized_by` `ISTA DASLab`: Underdog left ISTA's
header in place. `general.sampling.*` says temperature 1.0, top-p 0.95,
top-k 20; the card recommends 0.6 for general use and 0 for tool calls.

The chat template is 8,952 bytes, SHA-256
`c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`: the
upstream Qwen3.8 template already checked for Bonsai 2 against the pinned
profile (4 of 68 cases differ, the `merged_system_*` histories;
[prompt-profile.md](../engine/prompt-profile.md)).

## Tensors and encodings

851 tensors, 7.89 GB; the shapes equal the pinned Qwen3.8 file's tensor
for tensor. Excluding the embedding (one row per token is gathered), a
decoded token reads 7.61 GB, 2.26 bits per weight (the Q4_K_M: 15.39 GB,
4.58). The encoding is chosen per tensor, so one layer mixes several:

| Encoding | Tensors | In nuclis before this unit |
| --- | ---: | --- |
| F32 | 353 | yes |
| IQ2_XXS | 119 | no |
| Q8_0 | 96 (`ssm_alpha`, `ssm_beta`) | yes |
| IQ2_S | 46 | no |
| Q2_K | 46 | no |
| IQ3_S | 44 | yes |
| IQ2_XS | 39 | no |
| IQ1_S | 36 | no |
| IQ1_M | 31 (`token_embd` among them) | no |
| IQ3_XXS | 30 | no |
| IQ4_XS | 7 | yes |
| Q4_K | 4 | yes |

`ffn_down` is IQ2_XXS in 53 of 64 layers; `attn_q` is IQ2_XXS or Q2_K;
`attn_k` and `attn_v` keep the most bits (IQ3_S, IQ4_XS, Q4_K); the output
head is IQ3_S.

## Lineage: what Underdog changed

Compared with ISTA's `Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf` (8.41 GB; the
NOTICE names it as the base), by header and by range reads of tensor
bytes:

- **146 tensors re-encoded**, 0.52 GB saved: the 96 `ssm_alpha` /
  `ssm_beta` from BF16 to Q8_0; 50 down to IQ2_XXS (41 `ffn_down`, 6
  `ffn_up`, 2 `ffn_gate`, plus others from IQ2_XS, IQ2_S, Q2_K, IQ3_XXS,
  IQ3_S, IQ4_XS); the output head from IQ4_XS to IQ3_S.
- **The weights changed, not only their encodings.** ISTA's F32 tensors
  equal the original model's (the pinned Q4_K_M's F32 tensors) bit for
  bit. Saluki's norms and `ssm_conv1d` differ from them by BF16-sized
  steps (max 2⁻¹⁰ to 2⁻⁹; relative RMS 2–3 × 10⁻⁴ on the norms, 8 × 10⁻³
  on `blk.0.ssm_conv1d`), while `ssm_a`, `ssm_dt.bias`, and `output_norm`
  are unchanged; quantized tensors whose encoding is unchanged also differ
  in bytes. That is the signature of training after quantization (the
  full-precision parameters tuned to absorb the rounding error, the codes
  re-chosen); the card says only "tuned to keep tool calling intact".
  This is a reading of the evidence, not a documented method.

Unsloth's 2-bit files of the same model use the same encoding family:
`UD-IQ2_XXS` (7.26 GB, 6.84 GB per token, leaning on IQ1_S: 100 tensors)
and `UD-IQ2_S` (8.36 GB, 7.94 GB per token); the former adds Q3_K, which
nuclis runs. The decoders and kernels built for Saluki run all three.

## Memory at 32K context

From llama.cpp's buffer report (`llama-server -c 32768 -np 1 -fa on`,
Q8_0 projector), which matches the arithmetic of the layout:

| Part | MiB | |
| --- | ---: | --- |
| Weights | 7,522 | 265 CPU-mapped embedding + 7,257 GPU |
| Attention cache, F16 | 2,048 | 16 layers × 4 KV heads × 256 × K, V × 2 B = 64 KiB per token |
| DeltaNet state | 150 | 48 layers × 48 heads × 128 × 128 × F32, constant in context |
| Compute buffers | 345 | |
| Projector and its buffer | 848 | |
| Total | ≈ 10,900 | |

"Under 8 GB" is the weights. The attention cache is the part that grows
and the part a quantized cache shrinks.

## The llama.cpp record (2026-10-14)

The last external comparison. llama.cpp master
`529ac893e754e95a50469b2122f5efe28dfc08a0` (2026-10-14), Release, Metal,
embedded library, built under `.reference/llama.cpp-latest`; Apple M4 Pro
(12 CPU, 16 GPU cores), 48 GiB, macOS 27.0.1. `llama-bench -ngl 99 -fa 1
-ctk f16 -ctv f16`: decode is 128 tokens at a depth (three runs), prefill
a prompt of that length at depth 0 (one run). Saluki ran first from a
cool chip, the Q4_K_M after five minutes idle. Tokens per second, Saluki /
Qwen3.8 UD-Q4_K_M:

| | 512 | 4,096 | 16,384 | 32,639 |
| --- | ---: | ---: | ---: | ---: |
| Decode | 12.46 / 11.16 | 12.21 / 10.97 | 11.47 / 8.54 | 9.54 / 7.84 |
| Prefill | 97.19 / 92.06 | 96.58 / 90.06 | 84.40 / 82.43 | 78.87 / 75.74 |

Half the bytes buy 1.12× at 512: 95 GB/s effective against 172 GB/s for
the Q4_K_M, so llama.cpp's 2-bit matvecs are bound by decoding work, not
memory. nuclis decodes the Q4_K_M at 11.67 tok/s, 17.4 with speculation
([README § Results](../../README.md#results)).

Through `llama-server` (thinking off unless noted): a chat answer, a
two-step arithmetic question with thinking on (correct, 370 characters of
reasoning), three parallel tool calls with correct arguments, and an
accurate description of `docs/branding/nuclis.png` through the Q8_0
projector. A smoke test, not a benchmark.

## Validation without an external reference

To be filled by this unit: the cross-encoding check against the pinned
Q4_K_M (per-encoding cosine and relative RMS, with the negative control),
Metal against the CPU reference, and perplexity against the Q4_K_M on
nuclis.

## Status

In bring-up (2026-10-14): the file is pinned and inventoried; nuclis
rejects it until the seven encodings land.
