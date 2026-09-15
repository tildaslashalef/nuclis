# Qwen structural validation

`nuclis validate` checks the initial Qwen3.8-27B GGUF profile and constructs typed
weight bindings. It is a prerequisite for implementing execution. A successful
result makes no claim about tokenizer correctness, numerical accuracy, or speed.

```sh
zig build -Doptimize=ReleaseSafe
./zig-out/bin/nuclis validate
./zig-out/bin/nuclis validate --model /path/to/model.gguf --json
```

Path selection follows [inspection](gguf-inspection.md): an explicit path, then
the shared root selected by `NUCLIS_HOME` or `HOME`. The command writes no user
files. Errors produce a nonzero exit code. `inspect` remains a generic container
tool and can inspect other architectures with recognized tensor layouts.

## Accepted profile

The adapter accepts the metadata dimensions and tensor forms observed in the
[pinned artifact](../spec.md#target-model-and-download). It verifies required
metadata types and values, including attention/SSM dimensions, normalization
epsilon, RoPE base and sections, vocabulary size, and quantization version.
Unknown `qwen35.*` metadata is rejected so an unimplemented schedule or RoPE
override cannot silently alter the interpretation. Other model sizes, tied output
embeddings, alternate projection packing, and other auxiliary layouts need an
explicit future extension.

| Part | Layers | Tensors | Stored bytes |
| --- | ---: | ---: | ---: |
| Main text decoder and embedding/output tensors | 64 | 851 | 16,102,434,816 |
| Auxiliary prediction block | 1 | 15 | 351,008,768 |

The main schedule consists of 48 DeltaNet and 16 full-attention layers. In
zero-based indices, full attention occurs at 3, 7, …, 63. Block 64 belongs to
auxiliary prediction, uses full attention, and is excluded from the text binding.
Its tensors are still validated. Stored byte totals exclude session state,
scratch space, and platform allocations.

The adapter checks every required tensor's dimensions and encoding, then rejects
unconsumed tensors. Normalization, convolution, and recurrent scalar tensors
require F32. Matrix tensors accept the nine layouts needed by this initial
artifact: F32, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ3_S, and IQ4_XS.
This is a structural allowlist; numerical acceptance is evidence from [generation.md](generation.md) and [metal-backend.md](metal-backend.md), not from binding.
The validator does not enforce the exact per-tensor quantization choices or
establish artifact identity. The full-file hash remains a separate check.

Tokenizer vocabulary length is checked against embedding dimensions. Tokenizer
algorithms, merges, special-token behavior, and prompt rendering remain separate
work. A matching display name is neither required nor evidence of compatibility.

## Implementation and ownership

[qwen35.zig](../../inference/src/models/qwen35.zig) exposes `bind(allocator,
document)`. The document must already have passed generic GGUF parsing. The
adapter creates a temporary name index and consumes each required entry once.
All temporary allocations are freed on success or error.

The returned `Binding` holds embedding/output tensor pointers and 64 `Layer`
values. Every layer has common normalization and feed-forward weights plus a
tagged `mixer` union containing either `FullAttention` or `DeltaNet`. This lets
later numerical code work with named fields instead of repeating string lookups
or guessing which weights belong to a layer. The union also prevents treating a
recurrent layer as if it only needed an attention KV cache.

Tensor pointers borrow the original `Document`. Keep it alive until all users of
the binding finish. The binding does not own or allocate weight buffers, and
has no independent cleanup method. The fixed 64-layer profile is intentional;
generalizing it should follow another concrete supported model.

The GGUF module remains independent of Qwen. It now retains numeric metadata
arrays of at most 16 elements, while larger arrays retain descriptors. That
shared facility lets this adapter validate the four RoPE sections without
materializing a large vocabulary or adding Qwen key names to the container parser.

## Evidence and references

On 2026-09-06, the downloaded artifact passed the adapter and produced the counts
above. A 39 KB [descriptor fixture](../../inference/src/models/fixtures/qwen35-27b.json)
captures actual metadata and every tensor's name, dimensions, and encoding.
It contains no weight values, tokens, or private prompts. Tests hydrate this
independent inventory rather than generating the expected layout from the binder.

Debug and ReleaseSafe validation both pass on the actual file. A process-level
negative check used a temporary sparse file with the same directory but an
altered attention interval: generic `inspect` succeeded, while `validate`
returned `UnsupportedConfiguration` with no success JSON. No weights were copied.

Tests cover the hybrid schedule, borrowed references, text/auxiliary accounting,
changed dimensions, wrong types, missing metadata, RoPE overrides, missing and
unexpected tensors, altered query/gate shapes, forbidden encodings, auxiliary
tensor errors, and allocation failure cleanup. Default tests require no model
download or GPU. The root suite has 25 tests after this increment.

Layout interpretation was checked against the
[Qwen35 implementation at llama.cpp revision 7620399](https://github.com/ggml-org/llama.cpp/blob/7620399f58aebfd2196b74021f9581bcf7218cb9/src/models/qwen35.cpp)
and the [upstream model configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json).
The former distinguishes main and auxiliary blocks and accounts for gated query
dimensions. This revision is a source reference, not yet a tested performance
baseline. No external implementation code was copied into this adapter.
