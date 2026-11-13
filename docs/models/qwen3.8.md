<!-- kind: card -->
# Qwen3.8-27B: the first target

Qwen3.8-27B is the family nuclis was built around and the one its kernels
are tuned for: catalogue entry `qwen3.8-27b`, file
`unsloth/Qwen3.8-27B-GGUF` `Qwen3.8-27B-UD-Q4_K_M.gguf` (16.6 GB, Unsloth's re-made file of
2026-11-11), adapter
`qwen35` ([inference/src/models/qwen35.zig](../../inference/src/models/qwen35.zig)).
Its speed is in the README's [Results](../../README.md#results) and
[history.md](../benchmarks/history.md).

## Architecture

From the file's `qwen35.*` metadata (the [descriptor
fixture](../../inference/src/models/fixtures/qwen35-27b.json)):

| Fact | Value |
| --- | --- |
| Blocks | 64 text layers plus one auxiliary prediction block (the MTP head) |
| Schedule | hybrid: full attention where layer % 4 == 3 (16 layers), Gated DeltaNet elsewhere (48) |
| Width | embedding 5,120, feed-forward 17,408 |
| Attention | 24 query heads, 4 KV heads, key and value length 256; RoPE base 10,000,000 over 64 dimensions in four sections |
| DeltaNet | inner size 6,144, 16 groups, state 128, 48 time-step heads, convolution kernel 4 |
| Vocabulary | 248,320 tokens; RMS norm epsilon 1e-6 |
| Context | 262,144 declared; nuclis measures up to 32,768 |
| Weights | nine encodings (F32, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ3_S, IQ4_XS); 16.2 GB in the text decoder |

## Prompt profile and where its facts live

The profile is `qwen38`, selected by the template digest (SHA-256
`12827f24…`, Unsloth's fixed template); the engine's other documents carry
the rest:

| Topic | Document |
| --- | --- |
| Binding and structural validation | below |
| Tokenizer (BPE over the GGUF vocabulary) | [engine/tokenizer.md](../engine/tokenizer.md) |
| Chat template, reasoning markers, sampling profile | [engine/prompt-profile.md § Qwen3.8](../engine/prompt-profile.md#qwen38-qwen38), [engine/sampling.md § Sampling profiles](../engine/sampling.md#sampling-profiles-and-the-selection-chain) |
| Tool calls | [engine/tool-calling.md § Qwen](../engine/tool-calling.md#qwen-xml-like-calls-json-declarations) |
| CPU forward, DeltaNet, traces against the reference | [engine/cpu-reference.md](../engine/cpu-reference.md), [engine/sampling.md § Numerical traces](../engine/sampling.md#numerical-traces) |
| Metal plan and its kernels | [engine/metal-backend.md](../engine/metal-backend.md) |
| Draft head and speculation budget | [benchmarks/record/speculative-decoding.md § The Qwen3.8 draft head](../benchmarks/record/speculative-decoding.md#the-qwen38-draft-head-2026-09-20), [§ The Qwen verify budget](../engine/speculative-decoding.md#the-qwen-verify-budget) |
| Vision projector | [engine/vision.md § The Qwen3-VL projector](../engine/vision.md#the-qwen3-vl-projector); its speed (1,024 tokens in 2.4–2.7 s) in [§ Encoders at the materialized-score attention](../benchmarks/record/vision.md#encoders-at-the-materialized-score-attention-2026-10-27) |
| Companion files and the catalogue entry | [catalogue.md](catalogue.md) |

## Structural validation

`nuclis validate` checks the Qwen3.8-27B GGUF profile and constructs typed
weight bindings. A successful result makes no claim about tokenizer
correctness, numerical accuracy, or speed; those are the documents above.

```sh
zig build -Doptimize=ReleaseSafe
./zig-out/bin/nuclis validate
./zig-out/bin/nuclis validate --model /path/to/model.gguf --json
```

Path selection follows [inspection](../engine/gguf.md): an explicit path, then
the shared root selected by `NUCLIS_HOME` or `HOME`. The command writes no user
files. Errors produce a nonzero exit code. `inspect` remains a generic container
tool and can inspect other architectures with recognized tensor layouts.

### Accepted profile

The adapter accepts the metadata dimensions and tensor forms observed in the
[pinned artifact](../spec.md#4-supported-models-and-artifacts). It verifies required
metadata types and values, including attention/SSM dimensions, normalization
epsilon, RoPE base and sections, vocabulary size, and quantization version.
Unknown `qwen35.*` metadata is rejected so an unimplemented schedule or RoPE
override cannot silently alter the interpretation. Other model sizes, tied output
embeddings, alternate projection packing, and other auxiliary layouts need an
explicit extension.

| Part | Layers | Tensors | Stored bytes |
| --- | ---: | ---: | ---: |
| Main text decoder and embedding/output tensors | 64 | 851 | 16,202,254,336 |
| Auxiliary prediction block | 1 | 15 | 351,008,768 |

The main schedule consists of 48 DeltaNet and 16 full-attention layers. In
zero-based indices, full attention occurs at 3, 7, …, 63. Block 64 belongs to
auxiliary prediction, uses full attention, and is excluded from the text binding.
Its tensors are still validated. Stored byte totals exclude session state,
scratch space, and platform allocations.

The adapter checks every required tensor's dimensions and encoding, then rejects
unconsumed tensors. Normalization, convolution, and recurrent scalar tensors
require F32. Matrix tensors accept the encodings the Qwen kernels execute
(`executableEncoding`): F32, BF16, Q8_0, Q2_K to Q6_K, IQ1_S, IQ1_M,
IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, and the ternary
PQ2_0 and PTQ1_0; the pinned file uses nine of them.
This is a structural allowlist; numerical acceptance is evidence from [sampling.md](../engine/sampling.md) and [metal-backend.md](../engine/metal-backend.md), not from binding.
The validator does not enforce the exact per-tensor quantization choices or
establish artifact identity. The full-file hash remains a separate check.

Tokenizer vocabulary length is checked against embedding dimensions. Tokenizer
algorithms, merges, special-token behavior, and prompt rendering belong to the
tokenizer and the profile, not the binding. A matching display name is neither
required nor evidence of compatibility.

### Implementation and ownership

[qwen35.zig](../../inference/src/models/qwen35.zig) exposes `bind(allocator,
document)`. The document must already have passed generic GGUF parsing. The
adapter creates a temporary name index and consumes each required entry once.
All temporary allocations are freed on success or error.

The returned `Binding` holds embedding/output tensor pointers and 64 `Layer`
values. Every layer has common normalization and feed-forward weights plus a
tagged `mixer` union containing either `FullAttention` or `DeltaNet`. This lets
the numerical code work with named fields instead of repeating string lookups
or guessing which weights belong to a layer. The union also prevents treating a
recurrent layer as if it only needed an attention KV cache.

Tensor pointers borrow the original `Document`. Keep it alive until all users of
the binding finish. The binding does not own or allocate weight buffers, and
has no independent cleanup method. The fixed 64-layer profile is intentional;
generalizing it should follow another concrete supported model.

The GGUF module remains independent of Qwen. It retains numeric metadata
arrays of at most 64 elements, while larger arrays retain descriptors. That
shared facility lets this adapter validate the four RoPE sections without
materializing a large vocabulary or adding Qwen key names to the container parser.

## What is verified

<!-- gen:gates-for:qwen38 -->
| Gate | Tier | Model | Evidence |
| --- | --- | --- | --- |
| `qwen38-trace-f32` | `verify` | qwen38 | [sampling.md § numerical-traces](../engine/sampling.md#numerical-traces) |
| `qwen38-trace-f16` | `verify` | qwen38 | [sampling.md § numerical-traces](../engine/sampling.md#numerical-traces) |
| `qwen38-trace-q8_0` | `verify` | qwen38 | [metal-attention.md § q8_0-kv-cache](../engine/metal-attention.md#q8_0-kv-cache) |
| `qwen38-generation-metal` | `verify` | qwen38 | [session.md](../engine/session.md) |
| `qwen38-generation-cpu` | `verify-cpu` | qwen38 | [session.md](../engine/session.md) |
| `qwen38-vision-metal` | `verify` | qwen38 | [vision.md § the-qwen3-vl-projector](../engine/vision.md#the-qwen3-vl-projector) |
| `qwen38-vision-cpu` | `verify-cpu` | qwen38 | [vision.md § the-qwen3-vl-projector](../engine/vision.md#the-qwen3-vl-projector) |
| `qwen38-speculative-metal` | `verify` | qwen38 | [speculative-decoding.md](../engine/speculative-decoding.md) |
| `qwen38-verify-depth-512` | `verify` | qwen38 | [metal-backend.md](../engine/metal-backend.md) |
| `qwen38-verify-depth-4k` | `verify` | qwen38 | [metal-backend.md](../engine/metal-backend.md) |
| `qwen38-speculative-cpu` | `verify-cpu` | qwen38 | [speculative-decoding.md](../engine/speculative-decoding.md) |
| `qwen38-dflash2-trace-metal` | `verify` | qwen38 | [speculative-decoding.md § the-qwen-dflash2-drafter-2026-10-21](../benchmarks/record/speculative-decoding.md#the-qwen-dflash2-drafter-2026-10-21) |
| `qwen38-dflash2-trace-cpu` | `verify-cpu` | qwen38 | [speculative-decoding.md § the-qwen-dflash2-drafter-2026-10-21](../benchmarks/record/speculative-decoding.md#the-qwen-dflash2-drafter-2026-10-21) |
| `qwen38-dflash2-speculative-metal` | `verify` | qwen38 | [speculative-decoding.md § the-qwen-dflash2-drafter-2026-10-21](../benchmarks/record/speculative-decoding.md#the-qwen-dflash2-drafter-2026-10-21) |
| `qwen38-dflash2-speculative-cpu` | `verify-cpu` | qwen38 | [speculative-decoding.md § the-qwen-dflash2-drafter-2026-10-21](../benchmarks/record/speculative-decoding.md#the-qwen-dflash2-drafter-2026-10-21) |
| `qwen38-draft-trace-metal` | `verify` | qwen38 | [speculative-decoding.md § the-qwen38-draft-head-2026-09-20](../benchmarks/record/speculative-decoding.md#the-qwen38-draft-head-2026-09-20) |
| `qwen38-draft-trace-cpu` | `verify-cpu` | qwen38 | [speculative-decoding.md § the-qwen38-draft-head-2026-09-20](../benchmarks/record/speculative-decoding.md#the-qwen38-draft-head-2026-09-20) |
| `qwen38-draft-stats` | `verify-release` | qwen38 | [speculative-decoding.md § the-qwen38-draft-head-2026-09-20](../benchmarks/record/speculative-decoding.md#the-qwen38-draft-head-2026-09-20) |
| `qwen38-vocabulary` | `verify` | qwen38 | [prompt-profile.md](../engine/prompt-profile.md) |
| `qwen38-mtp-companion-trace-metal` | `verify` | qwen38 | [speculative-decoding.md § the-qwen-mtp-companion-2026-10-21](../benchmarks/record/speculative-decoding.md#the-qwen-mtp-companion-2026-10-21) |
| `qwen38-mtp-companion-trace-cpu` | `verify-cpu` | qwen38 | [speculative-decoding.md § the-qwen-mtp-companion-2026-10-21](../benchmarks/record/speculative-decoding.md#the-qwen-mtp-companion-2026-10-21) |
| `qwen38-perplexity` | `verify` | qwen38 | [eval.md § the-reference-record-2026-09-24](../benchmarks/record/eval.md#the-reference-record-2026-09-24) |
| `qwen38-perplexity-full` | `verify-release` | qwen38 | [eval.md § the-reference-record-2026-09-24](../benchmarks/record/eval.md#the-reference-record-2026-09-24) |
| `qwen38-verify-depth-16k` | `verify-long` | qwen38 | [metal-backend.md](../engine/metal-backend.md) |
| `qwen38-verify-depth-32k` | `verify-long` | qwen38 | [metal-backend.md](../engine/metal-backend.md) |
<!-- /gen -->

## Numbers

The re-made file's baseline (Metal, F16 cache, M4 Pro, 2026-11-12;
[history.md § The re-made Qwen3.8 file](../benchmarks/history.md#the-re-made-qwen38-file-2026-11-12)):

| Prompt tokens | Prefill tok/s | Decode tok/s | MTP at draft 7 | DFlash2 at draft 7 |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 94.22 | 11.79 | 1.61× | 1.63× |
| 4,096 | 86.46 | 11.41 | 1.42× | 1.53× |

WikiText-2 perplexity over eight windows of 512 is 6.6320 ± 0.3595. Why
the file was re-pinned and how it was checked:
[record, 2026-11-12](../benchmarks/record/qwen3.8.md#the-re-made-file-2026-11-12).

## Known limits

- The binding is fixed at 64 text layers: other sizes, tied output
  embeddings, alternate projection packing, and other auxiliary layouts
  are refused until a concrete file asks for them.
- The file declares 262,144 tokens of context; nuclis's session stops at
  32,768.
- The dense 27B decodes at about 12 tok/s, too slow for agent loops on
  this machine next to the Gemma 4 models.
