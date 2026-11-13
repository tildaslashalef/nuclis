<!-- kind: record -->
# GGUF inspection: the record

The dated record behind [the reference](../../engine/gguf.md).

## Verified artifact (2026-09-05)

The local artifact was checked on 2026-09-05 against the revision and SHA-256
recorded in the specification. Its full-file checksum matched. The initial
inventory was also cross-checked with an independent Python wire-format reader.

| Property | Observed value |
| --- | --- |
| Container | Little-endian GGUF v3 |
| Architecture / name | `qwen35` / `Qwen3.8-27B` |
| File size | 16,464,440,224 bytes |
| Directory end / aligned data offset | 10,996,621 / 10,996,640 bytes |
| Metadata entries / tensors | 50 / 866 |
| Stored tensor elements | 27,320,697,856 |
| Tensor payload bytes | 16,453,443,584 |
| Declared blocks / next-token prediction layers | 65 / 1 |
| Embedding / feed-forward length | 5,120 / 17,408 |
| Attention heads / KV heads | 24 / 4 |
| Attention key / value length | 256 / 256 |
| Declared context | 262,144 tokens |

Stored elements include auxiliary tensors; this is not a count of parameters
used by the text execution path. The [Qwen adapter](../../models/qwen3.8.md)
validates this tensor mapping and distinguishes 64 decoder layers from the
auxiliary block; generic inspection does not perform that architecture-specific check.

| Storage encoding | Tensors | Payload bytes |
| --- | ---: | ---: |
| F32 | 360 | 10,686,464 |
| Q8_0 | 106 | 80,773,120 |
| Q3_K | 7 | 268,083,200 |
| Q4_K | 104 | 4,203,970,560 |
| Q5_K | 131 | 4,833,607,680 |
| Q6_K | 30 | 1,812,787,200 |
| IQ4_NL | 7 | 330,301,440 |
| IQ3_S | 4 | 153,190,400 |
| IQ4_XS | 117 | 4,760,043,520 |

The filename's Q4 label is insufficient to select kernels: the artifact runs
only because each of the nine observed encodings has its own decoder
([quantization.md](../../engine/quantization.md)).

## Companion files in models/qwen (2026-09-08)

(The directory was `models/qwen/` when this was recorded; since
2026-09-11 the same files live under `models/unsloth/Qwen3.8-27B-GGUF/`,
pulled by `nuclis model pull qwen3.8-27b --all` with provenance sidecars,
and the imatrix, `config.json`, and `README.md` are no longer kept locally.)

These files sat beside the main GGUF; text generation does not depend on them.
`nuclis inspect --model <file> --json` successfully inspected the imatrix and
projector containers; a separate read of their metadata established the fields
below. Container inspection does not mean the engine can execute them.

| File | Observed contents | Use in nuclis |
| --- | --- | --- |
| `config.json` | Qwen3.5-family architecture; 64 text layers, hidden 5120, FFN 17408, 24 query/4 KV heads, one MTP layer, no dedicated MTP embeddings; vision depth 27, hidden 1152, output 5120 | Cross-checked the GGUF adapter, the MTP weight sharing, and the vision design. Runtime dimensions and encodings come from the validated GGUF. |
| `imatrix_unsloth.gguf` | 13,642,656 bytes; `general.type=imatrix`; 992 F32 tensors; dataset label `unsloth_calibration_dataset`, 1,251 chunks of 8,192 tokens | Calibration statistics for quantization studies. It is not additional inference weights, a speed optimization, or a numerical trace oracle. |
| `mmproj-BF16.gguf` | 931,146,432 bytes; architecture `clip`, type `mmproj`; 334 tensors (224 F32, 110 BF16); vision encoder present, 27 blocks, patch size 16, output width 5120, projector type `qwen3vl_merger` | A vision encoder and merger, not merely a small linear projection; it is the projector of [vision.md § The Qwen3-VL projector](../../engine/vision.md#the-qwen3-vl-projector). Text-only generation does not load it. |
| `mtp-Qwen3.8-27B-Q4_0.gguf` | Separate downloaded prediction head | The `mtp` companion: the same prediction block at lower precision ([speculative-decoding.md § The Qwen MTP companion](../../benchmarks/record/speculative-decoding.md#the-qwen-mtp-companion-2026-10-21)). |
| `README.md` | Upstream architecture overview, thinking controls, suggested sampling parameters, vision/context claims | Human reference for product/evaluation choices. Its settings are not automatically applied; its claimed context is not our measured capacity. |

An importance matrix summarizes activation magnitudes collected on calibration
inputs and is used when choosing quantized weights. Its influence is baked into
the resulting quantized artifact; loading it alongside that artifact would not
improve its existing values. The dataset label and chunk counts alone are not
enough to reproduce the calibration corpus or prove quality on our coding tasks.
See the [llama.cpp imatrix documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md)
for its quantization role and statistical limitations.

The local README suggests thinking sampling of temperature 1.0/top-p 0.95/top-k
20 and non-thinking 0.7/0.80/20, with presence penalty 1.5 in the latter mode.
These are upstream recommendations, not measured nuclis defaults
([sampling.md](../../engine/sampling.md) has the sampler and its presence penalty). Keep benchmark sampling fixed for
comparability. The config's `unsloth_fixed_mtp` flag is provenance information,
not evidence that nuclis's MTP path is correct.

Local SHA-256 identities (not independently verified against remote hashes):

| File | SHA-256 |
| --- | --- |
| `config.json` | `45d9009888b40b71af58ac82f7151393c113b4eabb870ca3c3bbd237667d18cc` |
| `README.md` | `66088b22a4db15353288f45a570279f76d7d8b7f349a64dd28698b867c9bd2bb` |
| `imatrix_unsloth.gguf` | `0ee5b10bd0c2fa2127c6f4b43dbfe1efd71e383b63217af9dade1de36599f1c1` |
| `mmproj-BF16.gguf` | `83ee4f4f205fa514161778c41df1ea14144faa0f713510893b63c2395f5c2d53` |
| `mtp-Qwen3.8-27B-Q4_0.gguf` | `50d9ce5a6da381bbcfb31061cf73df94a90e6faf8efeddee379a9cb8f1501c6e` |
