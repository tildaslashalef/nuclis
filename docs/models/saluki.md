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
and the part a quantized cache shrinks: with `--kv q8_0` nuclis's 32K
session is 1,238 MiB instead of 2,198 (attention rows 1,088 instead of
2,048, plus the 150 MB of DeltaNet state;
[metal-backend.md § Q8_0 KV cache](../engine/metal-backend.md#q8_0-kv-cache-2026-10-15)).

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

**Decoders.** Unit tests in `quant/decode.zig` pin every field of the
seven blocks on hand-built data. Then the cross-encoding check (gate
`saluki-cross-encoding`; `zig build test-cross-encoding -- <saluki>
<qwen3.8 Q4_K_M> --rows 16`): every matrix both files hold, 16 rows each,
decoded from Saluki by the new decoders and from the Q4_K_M by decoders
pinned to llama.cpp's fixtures before 2026-10-14. 2026-10-14, per Saluki
encoding: the mean over tensors of each tensor's worst-row cosine, the
worst tensor, the largest relative RMS, and the control (the largest
|cosine| of a row against a distant row of the other file):

| Encoding | Tensors | Mean cosine | Min | Max rel. RMS | Control |
| --- | ---: | ---: | ---: | ---: | ---: |
| IQ1_S | 36 | 0.840 | 0.752 | 0.860 | 0.109 |
| IQ1_M | 31 | 0.850 | 0.770 | 0.817 | 0.135 |
| Q2_K | 46 | 0.924 | 0.834 | 0.615 | 0.384 |
| IQ2_XXS | 119 | 0.916 | 0.857 | 0.609 | 0.383 |
| IQ2_XS | 39 | 0.942 | 0.930 | 0.421 | 0.211 |
| IQ2_S | 46 | 0.954 | 0.925 | 0.407 | 0.177 |
| IQ3_XXS | 30 | 0.967 | 0.931 | 0.367 | 0.349 |
| IQ3_S | 44 | 0.977 | 0.946 | 0.329 | 0.442 |
| IQ4_XS | 7 | 0.991 | 0.983 | 0.275 | 0.302 |
| Q4_K | 4 | 0.989 | 0.983 | 0.185 | 0.119 |
| Q8_0 | 96 | 0.999 | 0.994 | 0.105 | 0.848 |

Agreement rises with bits per weight, as it must (the Q4_K_M side carries
its own error, so even Q8_0 stops short of 1). The worst tensors are
`attn_gate` at IQ1. The control is high only where neighbouring rows of
small matrices correlate (`ssm_alpha`, `ssm_beta` in Q8_0).

The margin, by breaking one detail at a time (mean cosine, correct →
broken):

| Broken detail | Encodings | Correct → broken |
| --- | --- | --- |
| the eighth sign (parity) dropped | IQ2_XXS, IQ2_XS, IQ3_XXS | 0.916 → 0.790, 0.942 → 0.808, 0.967 → 0.829 |
| the IQ1 shift's sign inverted (±1/8) | IQ1_S, IQ1_M | 0.840 → 0.779, 0.850 → 0.772 |
| IQ2_S's high index bits in reverse order | IQ2_S | 0.954 → 0.698 |
| Q2_K's scale and minimum nibbles swapped | Q2_K | 0.924 → 0.464 |
| IQ1_M's two per-16 scales swapped | IQ1_M | 0.850 → 0.756 |

The gate's thresholds sit between: IQ1_S and IQ1_M 0.81, Q2_K 0.85,
IQ2_XXS 0.87, IQ2_XS and IQ2_S 0.88, IQ3_XXS 0.91. They hold for this pair
of files only.

**GPU decoders.** `dequant.metal` decodes the seven encodings with the
CPU's operation order (the shaders compile in safe math mode), from the
same codebooks emitted into the shader source. `metal-check` holds them to
the [codebook fixture](../../inference/src/quant/fixtures/codebook.json)
(two real blocks of each, the CPU decoders' values): the embedding gather
bit-identical, the matvec, merged-segment, multi-row, and matmul paths
within their usual bounds.

**The forward.** It is the Qwen3.8 schedule, unchanged. nuclis's CPU
reference recorded `tests/fixtures/saluki-hello-comma` (`Hello,`, greedy
token 353); the gates hold Metal to it:

| Gate | Max abs | Rel. RMS | Bound |
| --- | ---: | ---: | --- |
| `saluki-trace-cpu` | 0 | 0 | 2e-3 / 1e-4 (a regression pin) |
| `saluki-trace-f32` | 1.68e-4 | 8.5e-7 | 2e-3 / 1e-4 |
| `saluki-trace-f16` | 2.42e-2 | 2.63e-4 | 3e-2 / 4e-4 |

The F16 bound is wider than Qwen3.8's 2e-4, and the deviation is the
cache's: the F16 and F32 runs are equal through layers 0–2 (DeltaNet),
part at layer 3, the first attention layer (relative RMS 9.5e-6), and
grow to 1.6e-4 by layer 63; the logits stay within 3.7e-5 with the same
greedy token. The F16 attention kernels themselves are checked against the
CPU on rounded operands by `metal-check`.

**Quality.** `nuclis eval --file wiki.test.raw --chunks 8 --prompt-profile
qwen38` (8 windows of 512, the last 255 tokens of each scored), Metal, F16
cache, 2026-10-14: Saluki **7.996 ± 0.455**, the Qwen3.8 Q4_K_M 6.670 ±
0.363 (+19.9 %).

**In use.** Registered as `saluki-27b` with its projector and the Qwen3.8
profile (`nuclis model pull … --register saluki-27b --profile qwen38`; the
template is not an alias of the pinned profile, as for Bonsai 2). Through
`nuclis serve`, the four requests sent to llama-server above: a chat
answer, the arithmetic question (141), three parallel tool calls with
correct arguments, and the logo described, "on a black background": the
PNG is transparent and the preprocessor drops alpha unpremultiplied
([vision.md](../engine/vision.md)). The requests' llama.cpp switch
`chat_template_kwargs.enable_thinking` is not read by nuclis, which takes
OpenAI's `reasoning_effort: "none"` ([api.md](../guide/api.md)), so they
reasoned first.

## Speed

`nuclis bench --model saluki-27b --prompt-tokens <array> --max-tokens 128
--ctx-size 32768 --kv f16` with the pinned arrays
(`tests/fixtures/run-2026-09-06/prompt-<n>.json`,
`boundary-2026-09-06/prompt-32639.json`), Metal, ReleaseSafe, the
specialized kernels (`86f9945`), 2026-10-15. At 512 one warmup and three
runs, at 4,096 one warmup and one run, at the longer prompts one run.
Tokens per second, nuclis / the generic kernels (2026-10-14) / the
llama.cpp record:

| | 512 | 4,096 | 16,384 | 32,639 |
| --- | ---: | ---: | ---: | ---: |
| Decode | **16.96** / 7.22 / 12.46 | **16.05** / 6.92 / 12.21 | **14.02** / 6.28 / 11.47 | **11.99** / 6.15 / 9.54 |
| Prefill | **94.18** / 66.96 / 97.19 | **87.60** / 62.67 / 96.58 | **67.34** / 50.35 / 84.40 | **51.52** / 42.11 / 78.87 |

Decode at 512 ranges 16.96–17.23 across this day's runs. It is 36 %
ahead of llama.cpp, and 45 % ahead of the Q4_K_M on nuclis (11.67),
from half the bytes. Prefill trails llama.cpp from 3 % at 512 to 35 % at
32,639.

The matvecs alone (`make bench-kernels ARGS=<ENC>`, GPU busy time, best
of five rounds, GB/s of weights read), the specialized kernels against
Q4_K's the same day:

| Encoding | 69632×5120 | 20480×17408 | 5120×17408 |
| --- | ---: | ---: | ---: |
| IQ1_S | 137.7 | 135.5 | 128.1 |
| IQ1_M | 130.3 | 130.2 | 126.2 |
| Q2_K | 199.7 | 198.2 | 189.2 |
| IQ2_XXS | 144.0 | 141.5 | 135.1 |
| IQ2_XS | 157.9 | 156.8 | 149.6 |
| IQ2_S | 168.7 | 168.5 | 161.6 |
| IQ3_XXS | 194.4 | 186.6 | 172.4 |
| Q4_K | 197.4 | 178.1 | 177.6 |

What took decode from 7.2 to 17 tok/s, in the order measured:
specialized matvecs for the seven (subnormal-pair decode, no per-value
integer-to-float conversion); the merged projections declined for them
(their bodies lost 11–23 % inside the merged kernel); and IQ2_S's grid
shared by 16 SIMD groups. The counters, the probes that moved nothing,
and the F16 dot that fails the F32 trace are in
[apple-gpu.md](../engine/apple-gpu.md#salukis-2-bit-matvecs-on-the-ffn_down-shape-2026-10-15).

**The Q8_0 cache** (`--kv q8_0`, the same method, 2026-10-15, `q8-kv-cache`
branch): decode 17.04 / 13.38 / 11.02 at 512 / 16,384 / 32,639 against
F16's 16.66 / 13.78 / 11.63 the same day, prefill equal (93.82 / 67.20 /
51.59), the session 1,238 MiB instead of 2,198; perplexity unchanged
(7.996 at 8 × 512, 6.113 at 4 × 4,096).

**The head-split decode attention** (the same method, 2026-10-15,
`split-pass-rows` branch; at 16,384 F16 one warmup and three runs, the
others one run): the split pass reads the F16 cache at the memory roof
and Q8_0 level with it
([metal-backend.md § The split pass at the memory roof](../engine/metal-backend.md#the-split-pass-at-the-memory-roof-2026-10-15)).
Decode, tokens per second, before (the Q8_0 cache's runs above) → after:

| Cache | 16,384 | 32,639 |
| --- | ---: | ---: |
| F16 | 13.78 → **16.01** | 11.63 → **13.20** |
| Q8_0 | 13.38 → **14.89** | 11.02 → **13.26** |

Prefill is unchanged (67.0 and 50.3). Those runs were back to back and
heat cost them part of the gain; the interleaved A/B below replaces
them.

**Measured without heat bias** (`make speed`, `scripts/speed.py
--contexts 16384,32639 --tokens 128 --pairs 5 --run run-2026-09-06`,
2026-10-15): `fcebd2b` (the commit before the head split) against
`115cbf7`, five pairs alternating the two binaries on a restored
prefix, 10 minutes idle before each cache. Decode, tokens per second,
medians:

| Cache | 16,384 | 32,639 |
| --- | ---: | ---: |
| F16 | 13.98 → **16.12** (+15.3 %) | 11.93 → **14.97** (+25.5 %) |
| Q8_0 | 13.29 → **16.13** (+21.4 %) | 10.96 → **15.03** (+37.2 %) |

Each row's five pairs agree within 0.3 %. At 32,639 F16 saves 17.0 ms
per token, all that 16 layers of the kernel bench predict; the cooled
profile finds the split pass 1–7 % faster than its bench time and nothing
else in the step moved
([metal-backend.md § The split pass at the memory roof](../engine/metal-backend.md#the-split-pass-at-the-memory-roof-2026-10-15)).
Q8_0 now decodes level with F16 at 16,384 and 0.4 % ahead at 32,639,
in 56 % of the session memory. The llama.cpp record (9.54 at 32,639)
decodes right after its own prefill, as the acceptance method does, so
these rates are not set against it.

**The goal of 22 tok/s at 512 is not reached.** A decode step is about
58.5 ms: gate and up about 35 %, `ffn_down` 19 %, the DeltaNet
projections 13 %, everything outside the matvecs about 8 ms. 22 tok/s
needs the 7.61 GB per token at about 200 GB/s on average, above every
2-bit kernel here alone
([apple-gpu.md § The merged gate+up](../engine/apple-gpu.md#the-merged-gateup-on-salukis-encodings-2026-10-15)).

## Status

Bring-up (2026-10-14): the file validates and binds, the seven encodings
decode on the CPU and on Metal, Metal matches the CPU reference's traces,
and the model chats, calls tools, and reads images through `nuclis
serve`.

Speed (2026-10-15): specialized kernels for the seven, decode 16.96 tok/s
at 512 (llama.cpp 12.46) and prefill 94.18 (97.19). The 22 tok/s goal is
not reached; § Speed records where the step goes. The head-split decode attention
(2026-10-15) took decode at 32,639 from 11.93 to 14.97 (F16, interleaved
A/B on a restored prefix). Open: prefill at long
context (35 % behind llama.cpp at 32,639), and which 2-bit file becomes a
catalogue entry (the comparison with Unsloth's UD-IQ2_XXS and UD-IQ2_S).
