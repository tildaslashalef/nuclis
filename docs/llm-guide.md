# Learning the inference stack through nuclis

This is the growing companion to implementation of the inference stack,
starting with the concepts explained on 2026-09-06. Read sections in order:
later explanations build on these terms. Append substantial new concepts as
we build their components; correct earlier explanations when necessary.
Agent and terminal concepts live in
[reference/agent-concepts.md](reference/agent-concepts.md). Product
requirements stay in the [spec](spec.md), and implementation status stays in
[TODO.md](../TODO.md) (in progress) and the
[engineering log](engineering-log.md) (closed).

## 1. What an inference engine does

A trained model contains **weights**: arrays of learned numerical parameters.
A **tensor** is a multidimensional array; its **shape** gives the length of each
axis. A weight matrix might transform an input vector into a different set of
features. Weights alone do not execute anything. The engine supplies the model's
operations, manages memory and state, and runs those operations on hardware.

Our intended text path is:

```text
model file → validated weights + architecture configuration
messages → prompt formatting → tokenizer → input token IDs
input IDs + weights + session state → model operations → logits
logits → sampler → next token ID → decoded text
                   ↳ feed the token back into the model and repeat
```

**Logits** are scores over the vocabulary for the next token. The **sampler**
chooses a token from these scores. Greedy sampling selects the highest score;
other strategies allow controlled variation. Turning an output token ID into
text is also called decoding by tokenizer libraries. Benchmark “decode speed”
means the whole autoregressive generation step, not just this string conversion.

## 2. What we have built: the model-file reader

GGUF is an inference-oriented container from the GGML ecosystem. It carries
named tensors, their shapes and storage encodings, and metadata that can include
architecture settings, tokenizer information, and a chat template. Its layout
supports memory mapping: accessing file-backed data without first copying the
whole file into a separate heap allocation. See the [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).

Our [GGUF parser](../inference/src/formats/gguf.zig) reads the directory
and validates its structure. Think of this as checking a book's table of
contents and page ranges before reading its chapters. It checks dimensions,
encoding sizes, duplicate entries, offsets, alignment, overlap, and file bounds.
It does not yet map weights for computation or evaluate model equations.

The initial supported GGUF subset is complete for inspection of our pinned
artifact; this is not universal support for every GGUF version or encoding.
The artifact has 866 tensors in nine storage encodings. The
[Qwen adapter](../inference/src/models/qwen35.zig) goes further: it checks
what the tensors mean for this architecture and binds them to typed layer
structures. It separates 851 main text tensors from 15 auxiliary tensors.
Structural compatibility does not establish numerical correctness.

### GGUF and safetensors

Safetensors is a tensor-storage format with a JSON header describing names,
dtypes, shapes and byte offsets, followed by tensor data. It avoids pickle's
executable-object mechanism and supports efficient access. See its
[format documentation](https://github.com/safetensors/safetensors/blob/main/README.md#format).

| Question | GGUF | safetensors |
| --- | --- | --- |
| Typical packaging | Inference metadata and tensors together | Tensors, commonly accompanied by separate model/tokenizer configuration files |
| Quantization | Established GGML block encodings | Representation and quantization conventions depend on the producer/runtime |
| Our current support | Bounded parser and initial Qwen validation | None |

Neither extension determines model quality or generation speed by itself.
The numerical representation and executing kernels matter. Safetensors does
not mean “always full precision,” and GGUF does not mean “always four bits.”

Supporting safetensors later is compatible with our modular direction, but is
not a first-release commitment. We would add a loader and normalize its tensors
and accompanying configuration before architecture-specific binding. Today's
Qwen binder accepts a GGUF document directly; that format-neutral boundary still
needs to be built when a second format provides a concrete requirement.

## 3. Quantization: fitting the weights in memory

**Quantization** represents numbers with fewer bits, plus information such as
scales used to interpret them. A block encoding groups multiple values with
shared metadata. Its name and tensor dimensions determine how many bytes must
exist; its decoding equations determine what numbers those bytes represent.
Our encoding module implements byte sizing; the separate
[quantization module](reference/quantization.md) now decodes every encoding in the pinned artifact, plus F16. The first numerical tensor
operation, matrix-vector multiplication, is now implemented.

Roughly 27 billion parameters at two bytes each would occupy 54 GB before
session state and temporary buffers. That already exceeds 48 GiB. Our downloaded
GGUF is about 16.46 GB because it uses mixed quantization. The `Q4_K_M` filename
does not imply every tensor uses a single four-bit representation.

The CPU reference routines reconstruct values and now multiply matrices by
vectors; additional tensor operations remain pending. These give us readable correctness oracles for GPU kernels. The
fast GPU path can unpack weights while multiplying, avoiding a full expanded
floating-point copy of the model.

## 4. Tokens, prefill, and decode

A **token** is a vocabulary entry representing a piece of text or a special
control marker. It can correspond to a word, part of one, punctuation, or
whitespace. Code and natural language tokenize differently; token counts are
not word counts. The **tokenizer** maps between text and token IDs.

**Prompt tokens** count everything the model receives: instructions, conversation
history, code, tool results, and template markers. **Prefill** is the phase that
processes these tokens and prepares session state. Prompt tokens are the amount
of input; prefill is the work done on it. Processing many prompt tokens together
lets the GPU reuse weights across more work.

**Decode** generates the continuation autoregressively: each selected token
becomes input to the next step. **Output tokens** or **generated tokens** count
the answer; decode tokens/s measures how quickly it grows. For one ordinary
sequence, future tokens are not all known in advance, limiting parallelism.

**Context capacity** includes input and generated history, not just the initial
prompt. A 32,768-token allocation cannot accept a 32,768-token prompt and then
produce another lengthy answer without a different context-management policy.
Runtime boundary conventions may require an additional token of margin.

The [reference baseline](reference/reference-baseline.md) measured Qwen through llama.cpp
and its GGML Metal backend. “llama” here names the software, not the model family.
These are not nuclis performance measurements.

| Input tokens | Prefill tokens/s | Decode tokens/s | Approximate prefill duration |
| ---: | ---: | ---: | ---: |
| 512 | 89.19 | 9.66 | 5.7 seconds |
| 4,096 | 89.26 | 9.21 | 46 seconds |
| 16,384 | 74.07 | 7.32 | 3 minutes 41 seconds |

These are mean rates from three warm requests per size, with prefix reuse off.
Each produces 128 output tokens. At 7.32 tokens/s, 1,000 generated tokens would
roughly take 137 seconds if that rate held; this extrapolation is not a measured
1,000-token run. Reference decode timing counts the intervals after the first
generated token, so its rate uses 127 intervals in our 128-token runs.

**Time to first token** also includes startup, tokenization, prefill and serving
overheads. We did not measure streaming first-token latency in this experiment.
Warm means prior execution warmed the relevant path; it does not mean the
prompt was reused. The exact-boundary capacity result is flagged and explained
in the baseline report; it is not a clean capacity acceptance pass. The later
one-token-margin follow-up passed the reference token-count/truncation check.
It does not establish nuclis execution or zero-swap memory acceptance.

## 5. Why weights and session state are separate

Weights are immutable during inference. Each conversation needs mutable state.
An attention **KV cache** retains keys and values computed from earlier tokens,
so future steps can reuse them. Its size grows with the retained context.

Our Qwen text model alternates two kinds of layer: 16 full-attention layers and
48 DeltaNet layers. DeltaNet maintains a recurrent numerical state updated as
tokens arrive. That state is not an ordinary token-indexed KV cache. Rewinding a
conversation requires restoring or reconstructing the relevant recurrent state;
simply shortening an attention cache is insufficient.

Reusing an unchanged conversation prefix can avoid processing it from scratch.
This primarily improves follow-up prefill and first-token latency. It does not
make each new decode step inherently faster. Persistent prefix caching remains
deferred in the spec; state ownership must nevertheless support correct reuse.

## 6. Where performance can come from

Practical local coding is the goal. The reference is one pinned configuration,
not a proven hardware ceiling. It is also a mature implementation that already
uses many optimizations. Zig alone cannot guarantee beating it.

Dense-model decode repeatedly accesses a large weight set. **Memory bandwidth**
is how quickly data can reach computation; enough RAM to fit a model does not
imply enough bandwidth to generate rapidly. Longer attention histories add
more state access. Profiling must establish the limiting operations on this Mac.

Our correctness-first implementation and optimization work should examine:

- **Quantized matrix operations:** unpack inside the GPU operation, specialize
  for actual shapes/encodings, and reuse data efficiently across GPU threads.
- **Separate prefill and decode kernels:** batched matrix multiplication and
  single-token matrix-vector work have different efficient implementations.
- **Kernel fusion:** combine compatible normalization, gates, or residual
  operations to reduce intermediate memory traffic and dispatches.
- **GPU scheduling:** reuse buffers and submit useful groups of operations,
  avoiding unnecessary CPU/GPU waits while preserving resource lifetimes.
- **Attention and recurrent state:** efficient cache access, attention tiling,
  and DeltaNet updates; evaluate lower state precision only with accuracy checks.

A **kernel** is a GPU program performing a numerical operation. Zig owns the
execution plan, Objective-C connects to Metal's host API, and Metal Shading
Language expresses GPU kernels. The first, synchronous version of this boundary
exists; see [metal-backend.md](reference/metal-backend.md) and section 22 below.

Later candidates include prompt reuse and **speculative decoding**: draft several
tokens cheaply, verify them with the main model, and retain accepted tokens.
Our artifact has auxiliary next-token-prediction weights, but using them for
MTP/speculation requires additional implementation and validation. Acceptance
rate and verification cost determine whether it helps. This remains deferred;
we should not promise its speedup or silently add it to initial release scope.

Future measurements must separate gains in prefill, decode, memory, and quality.
A faster run at shorter context or more aggressive quantization is a changed
configuration, not proof that an equivalent kernel became faster.

## 7. From stored bytes to numerical values

The first [row decoders](reference/quantization.md) connect the file layout to numbers.
Q8_0 uses a shared scale and signed integers. IQ4_NL uses four-bit indices into
a nonuniform table, then applies its scale. Both use F16 scales, illustrating
why a quantized format can still contain floating-point metadata.

Packing order is part of numerical correctness. For IQ4_NL, the two nibbles in
one byte refer to values 16 positions apart. Reading them as adjacent values
would produce valid-looking numbers in the wrong positions, corrupting later
matrix multiplication. Known blocks and pinned CPU fixtures test this ordering.

In Zig, a slice borrows an array plus its length; receiving a slice does not
transfer ownership. Our decoder receives bytes and a caller-owned F32 output
slice, checks both lengths, then fills it without allocating. That lets a future
caller reuse row-sized scratch. `@bitCast` reinterprets stored bits, whereas
`@floatFromInt` performs a numerical conversion; confusing them would turn a
quantized byte into the wrong value. These routines establish storage decoding,
not correct model execution or optimized CPU inference.

## 8. Quantizing the scales themselves

Q4_K and Q5_K add a second level of quantization. A 256-value **super-block**
contains eight groups, each with its own scale and offset. Those group
coefficients are stored as six-bit integers and interpreted using two shared
F16 coefficients. This lets different parts of a row represent different value
ranges without storing a floating-point scale for every weight.

Our new decoders reconstruct `(d * scale) * q - (dmin * minimum)`. The offset
means a zero integer code can represent a negative weight. Q5_K extends the
four-bit code with a fifth bit stored in a separate byte array, called a
**bit plane**. The plane groups bits by their role instead of storing each
weight's five bits together. Correct decoding must restore their association.

The shared Zig helper uses a `comptime` Boolean to select Q4_K or Q5_K storage.
The compiler knows that choice when compiling the call and can discard the
irrelevant path. This is a small specialization around two concrete formats;
callers still use the same checked row interface and own the output buffer.
The exact CPU fixture matches validate these decoders, not GPU kernels or the
full Qwen execution graph.


## 9. Signed values can have different stored representations

Q3_K and Q6_K illustrate why counting bits does not fully describe an encoding.
Q3_K's two low value bits become negative when a separate mask bit is clear:
it subtracts four, giving codes from -4 through 3. Q6_K combines six bits into
an unsigned code, then subtracts 32, giving values from -32 through 31.

Their group scales differ too. Q3_K packs unsigned six-bit codes and subtracts
32 from them; Q6_K stores ordinary signed eight-bit integers. Our Zig code uses
`@bitCast` only for the latter. It widens the unsigned biased codes to `i16`
before subtracting, so valid negative results cannot underflow an unsigned type.
A negative scale can reverse a code's sign; both fields must be decoded correctly.

The CPU routines now decode these formats with byte indexing, explicit biases,
and bounded shifts. Hand-calculated tests isolate signs and ordering, while
multi-block fixtures compare the complete operation with pinned C decoders.
These are row-level references; the CPU module now composes them into
matrix-vector multiplication, but model-layer execution remains pending.


## 10. A code can represent several values at once

IQ3_S uses a **codebook** of 512 four-value patterns. A nine-bit index selects
one pattern of positive magnitudes, after which sign bits and a group scale
produce four weights. Unlike IQ4_NL's scalar lookup, the index describes a
small vector. The table is fixed by the storage format; it is not an additional
set of learned weights or something generated during model loading.

The stored 32-bit table entries contain four component bytes. Our decoder
extracts those bytes with integer shifts so the interpretation does not depend
on the machine's byte order. Tests compare every table entry against the pinned
reference, while a small hand-calculated case checks component and sign order.
IQ4_XS combines the earlier IQ4_NL scalar table with signed group scales.

All nine encodings in our artifact now have CPU row decoders. This closes the
bytes-to-values step. The CPU matrix-vector operation now combines them with
input vectors. Complete model execution still requires additional operations,
state handling, tokenization, and the architecture schedule.


## 11. Combining weights with an input vector

A matrix-vector multiplication produces one output for every weight row. Each
output is a **dot product**: multiply corresponding weight and input values,
then add the products. For row `[1, 2, 3]` and input `[2, -1, 0.5]`, that gives
`2 - 2 + 1.5 = 1.5`. Multiple rows transform the same input into multiple output
features. This is a building block for a model's learned linear projections.

Our [CPU reference](reference/cpu-reference.md) decodes one row into reusable F32 scratch.
It then widens both operands to F64, multiplies and accumulates in column order,
and converts the final sum to F32. This separates the precision of stored
weights from the precision used to accumulate their products. The test
`16777216 + 1 - 16777216` retains `1` with our F64 accumulator; ordinary F32
accumulation loses that term at the first addition.

A Zig `Matrix` here is only a borrowed descriptor: bytes, encoding, and two
lengths. It owns no memory and does not imply that its contents are valid.
The operation validates the complete descriptor before modifying buffers, then
borrows caller-owned scratch for each row. This makes memory use explicit and
avoids a full floating-point copy of the weights. Future GPU kernels may unpack
inside a multiplication instead, but must satisfy the same mathematical result
within a justified tolerance. No Qwen layer schedule is implemented by this
single operation.


## 12. Controlling scale and introducing nonlinear behavior

Matrix multiplication alone is linear. **Activations** introduce nonlinear
behavior, allowing successive layers to express more than a single linear
transformation. Our scalar references now include sigmoid, SiLU, and softplus.
Sigmoid maps values into the interval from zero to one and is useful as a gate.
SiLU multiplies an input by that gate. Softplus is a smooth positive function
that behaves approximately like its input when the input is large and positive.

**RMSNorm** divides a vector by the square root of its mean square plus epsilon.
Epsilon prevents division by zero; it belongs inside the square root in our
implemented equation. This operation controls overall magnitude without
subtracting the mean. Our primitive applies no learned weights: model-specific
weight offsets and gating must remain explicit in the adapter.

**Softmax** turns scores into nonnegative probabilities that sum approximately
to one. Directly exponentiating large scores can overflow. Subtracting their
maximum preserves the mathematical probabilities while keeping exponentials
bounded by one. Our implementation treats negative infinity as a masked score;
it rejects an all-masked vector because there is no supported outcome to
normalize. Maximum and sum are **reductions**: each combines a whole vector into
one value. They finish before output writes, allowing exact in-place use.

The [CPU references](reference/cpu-reference.md) use F64 intermediates and explicit
nonfinite-input policies. Stable formulas matter even before optimization:
`log1p` in softplus retains tiny positive results that forming `1 + tiny` first
could round away. These are tested numerical building blocks; the complete
attention and recurrent layers still need implementation.


## 13. Prompt profiles are part of checkpoint compatibility

The same numerical architecture can be paired with different conversation
formats. Our new [Qwen text profile](reference/prompt-profile.md) supplies the exact role
markers, separators, reasoning instructions, and assistant prefix expected by
the pinned checkpoint. It is separate from model equations and from tokenization.
For this checkpoint, prior assistant reasoning is a separate field that remains
in history even when the next answer's thinking mode is off.

A prompt's **byte length** and **token length** are different constraints. The
renderer bounds UTF-8 bytes before tokenization. The tokenizer must later map
those bytes to IDs and enforce any token budget. Special-token parsing also
matters: a literal role marker can become one control token or ordinary text
pieces, depending on the requested mode. The captured fixtures test both choices.

We now compare 20 native rendered prompts byte-for-byte with the pinned
reference. We also saved exact token IDs; native prompt encoding and
decoding now match every captured sequence (sections 15–16). Its vocabulary and merge lists can now be loaded separately from lazy
GGUF descriptors (see the next section). The
metadata's `gpt2` label does not imply that every GPT-2 tokenizer is compatible:
the `qwen35` pre-tokenization rules and checkpoint vocabulary determine the
actual behavior. These fixtures let the next increment test compatibility
without depending on a live model server during ordinary Zig tests.


## 14. Vocabulary ownership and merge ranks

The [vocabulary loader](reference/tokenizer.md) turns selected lazy GGUF arrays into owned
runtime data. A **token ID** is an index into the vocabulary, so sorting its
strings would change model meaning. A **merge rank** is the position of a pair
in the merge list. Our byte-pair encoding (BPE) core uses those ranks to choose
which adjacent pieces to combine; a dictionary lookup alone is not tokenization.

There are two deliberate lifetimes. `gguf.ArrayReader` returns string slices
that borrow the caller's directory bytes. `vocabulary.load` copies those strings
into an arena and builds lookup tables whose keys borrow that arena. The caller
can then free the source bytes and parsed document without invalidating token
lookups. Deinitializing the vocabulary releases the whole arena; copying the
owning struct does not create another independently owned vocabulary.

Bounds apply before allocation and copying: entry counts bound tables and arrays,
per-string limits reject oversized entries, and an aggregate text budget bounds
copied strings. The text budget is not a total memory limit, because entries,
hash-table capacity, and arena overhead also occupy memory. These distinctions
matter when loading an untrusted file, even before any GPU weights are created.


## 15. Bytes, merge order, and incomplete characters

The [BPE core](reference/tokenizer.md) starts from bytes, not user-visible characters.
UTF-8 encodes many characters as several bytes; each byte receives a reversible
symbol in the vocabulary's byte alphabet. A printed `Ġ` in a stored token means
an encoded space byte. It does not mean the original user typed that character.

BPE repeatedly combines adjacent symbols according to learned merge ranks.
For a synthetic input `abc`, if `b c` outranks `a b`, the first merge produces
`a` plus `bc`. Even an available `abc` token does not justify choosing it unless
the merge rules reach it. Our tests exercise this distinction and leftmost
selection when the same pair appears at more than one position.

The current core deliberately handles only one piece. A pre-tokenizer must
first separate a prompt using the checkpoint's Unicode rules, and special-token
recognition must identify control markers. Those stages are now implemented by
the full encoder described in section 16.
Passing the whole prompt to BPE would allow merges across required boundaries.
The explicit artifact check therefore distinguishes single-piece encoding tests
from decoding all captured token sequences.

Decoding reverses token IDs into bytes. A single token may contain only the
first byte of a multibyte character; that is valid tokenizer output even though
it is not independently valid UTF-8 text. The new decoder preserves those bytes.
A future streaming text renderer must retain incomplete characters until later
tokens complete them. This decoder does not yet implement that renderer.


## 16. Pre-tokenization controls which merges are legal

Our full [qwen35 encoder](reference/tokenizer.md) now composes the earlier building blocks.
First it reserves special-marker spans. Then a Unicode splitter identifies
ordinary text pieces. Finally BPE runs independently inside each piece. In this
checkpoint's rules, each Unicode number is a separate piece: even if a merge
rule could combine `1` and `2`, the full encoder cannot merge them across that
boundary. Combining marks, however, remain in the same run as letters.

Special markers have their own precedence. Longer marker types reserve matching
spans before shorter types, and claimed spans cannot be split afterward. With
special parsing disabled, control markers become ordinary text; user-defined
markers still receive their IDs. These are vocabulary categories, separate from
conversation roles such as user and assistant.

The Unicode category table is pinned data, like the vocabulary. Using whatever
Unicode version happens to be installed on the host could change boundaries
for newly assigned characters. Our generated range table keeps behavior stable
and small enough for ordinary tests without an external checkout.

`Encoder` borrows the immutable vocabulary and owns only its marker index. Each
encoding call owns its temporary buffers and returned IDs. BPE work is charged
to one budget across all pieces, so splitting an input into many pieces does not
reset the allowed work. The actual-artifact check now reproduces all 40 saved
standalone and prompt token sequences; this establishes tested text compatibility,
not model generation or GPU execution.


## 17. Direction, magnitude, and position

The new [L2 reference](reference/cpu-reference.md) divides a vector by its length, with
epsilon as a lower bound on the denominator. This differs from RMSNorm, which
uses a mean square and adds epsilon inside the square root. The pinned Qwen
DeltaNet path applies L2 normalization to its query and key vectors; the full
recurrent operation is now available as a CPU reference (section 19). Keeping the primitive separate prevents
that model-specific schedule from leaking into shared vector math.

**Rotary position embedding**, or RoPE, represents position by rotating pairs
of query/key features. Each pair uses a different angular frequency. Our
split-half layout pairs channel `i` with `i + rotary_width/2`, rather than its
immediate neighbor. The pinned text path rotates a 64-channel prefix and leaves
the remaining 192 channels unchanged. Because its three active position streams
are identical for text, the multi-axis reference reduces to this one-position
operator. Different image/video positions are outside this implementation.

Ideal rotations preserve length, and opposite positions undo one another.
Floating-point arithmetic introduces rounding: our F64 direct frequencies and
the pinned CPU backend's repeated F32 updates increasingly differ at large
positions. Tests therefore separate mathematical invariants from comparisons
with another implementation. The documented tolerance for a saved reference
case is evidence about that case, not permission for an arbitrary error in
future GPU kernels or the whole model.


## 18. Attention reads a visible prefix of memory

The new [attention reference](reference/cpu-reference.md) answers one query position at a
time. A query is compared with stored keys to produce scores; softmax turns the
scores into weights; those weights combine the corresponding value vectors.
Keys determine relevance, while values supply the content that is combined.
The result is still a head output, not a generated token or complete layer.

**Grouped-query attention (GQA)** lets several query heads share a KV head.
Our explicit mapping uses consecutive groups. For the pinned model, 24 query
heads share four KV heads, six queries per group. This reduces the number of
key/value vectors that a future cache must retain. The reference accepts existing
arrays; cache allocation, appending, and session lifecycle are still pending.

**Causal masking** means a query cannot read future tokens. Here the caller gives
the visible prefix length, and the operation excludes the suffix from both
softmax and the weighted sum. Repeated calls can use growing prefixes during
prefill or the current cache length during decode. A masked element is excluded;
it is not given a zero score, which would still contribute to softmax.

We keep attention scores in F64 through normalization. The public softmax
primitive takes F32, so routing huge dot products through it would lose finite
scores before stabilization could help. Explicit buffer types preserve that
numerical boundary. One caller-owned scratch row is reused for every query head,
and the reference validates all shapes and finite values before modifying it.


## 19. DeltaNet updates memory instead of appending token records

In full attention, each token contributes separate key/value vectors to a cache.
A later query scores those individual keys. Gated DeltaNet instead keeps a
fixed-size matrix per head and changes that matrix on each token. The matrix
acts like an associative memory: applying a key predicts a value; the delta
rule writes a correction proportional to the difference between that prediction
and the incoming value. A separate decay gate controls how much old memory
survives. The query then reads the updated memory.

This is architecturally novel **within our implementation**, rather than a new
invention. The [Gated Delta Networks paper](https://arxiv.org/abs/2412.06464)
combines gated forgetting with delta-rule memory updates. Our pinned llama.cpp
already has a CPU `ggml_gated_delta_net` operator, so we can compare both outputs
and recurrent matrices directly. Testing it before Metal separates errors in
recurrence/state layout from GPU scheduling or synchronization mistakes.

There is also short-term state: causal convolution retains a small window of
previous projected inputs. The current input uses only that history and itself,
never future tokens. Our new [recurrent references](reference/cpu-reference.md) implement
both the history shift and the scalar-gated DeltaNet update, with caller-owned
state and explicit numerical checks before committing a step.

Deleting the last KV record can remove one token from ordinary attention's
stored history. DeltaNet has already mixed that token into its matrix; some old
information may have been forgotten or overwritten. It cannot generally undo
that update by reducing a length, or reliably invert it after floating-point
rounding. Restoring a checkpoint or replaying from earlier state is the practical
route. Both the convolution history and recurrent matrix must be restored.

The tested operators are building blocks. To generate text, we still need mapped
weights, a session that owns both kinds of memory, the complete hybrid layer
schedule, logits, and a token-selection loop. Metal then supplies the practical
execution backend; none of these stateful operators alone is a language model.


## 20. Weight lifetimes and session failure

`weights.Mapped` now keeps a read-only file mapping and its parsed GGUF document
alive together. Views borrow the map, so no full floating-point copy of the model
is required. The file must remain unchanged while mapped. Row decoding and small
F32 reads still validate tensor ranges before accessing data.

`session.Session` separately owns attention buffers, convolution histories, and
recurrent matrices described by a model-supplied layout. A token moves through
begin, commit, or failure. Failure poisons the session until reset because some
layers may already have updated recurrent state. Reset clears all state kinds;
there is deliberately no method that pretends changing a KV length rewinds them.
The full layer schedule is the next composition step.


## 21. From one layer to a generated token

`models/qwen35_runtime.zig` now composes the complete CPU text schedule. An input
token selects one embedding row. Each layer normalizes that vector, mixes it
with its attention or recurrent history, adds the residual, and runs a gated
feed-forward block. After the final layer, normalization and the vocabulary
projection produce one logit per token. Prefill updates the same session state
as decode; intermediate prompt tokens skip the expensive vocabulary projection.

A logit is an unnormalized score. Greedy selection takes the largest; temperature
sampling rescales scores before exponentiation, with top-k and top-p restricting
the candidate set. Our sampler owns its seeded RNG and borrows caller-provided
scratch storage. It knows nothing about Qwen or prompt markers. The selected ID
becomes the next input, causing every recurrent layer to update again.

The runtime borrows immutable mapped weights and owns activation workspace plus
session state. A callback observes completed layers and can cancel a step. Since
earlier layers have already mutated state, that error poisons the session until
reset. The explicit full-model check now verifies independent sessions and reset
after cancellation by comparing two-token logits exactly.

Decoded token bytes are not necessarily complete UTF-8 characters. The new stream
keeps up to one incomplete scalar between pieces and flushes completed text after
each generated token. It replaces invalid or unfinished characters with U+FFFD;
JSON token IDs retain the exact sequence. This presentation state is separate
from both the model's recurrent matrices and its tokenizer vocabulary.

The CPU graph now agrees closely with the pinned Metal reference on short
layer/logit traces. This verifies wiring that isolated operators could not check,
such as Q/gate packing and DeltaNet head mapping. It does not establish long
context quality or practical speed; those remain separate acceptance work.

## 22. Crossing to the GPU: what the first Metal bridge teaches

A GPU does not run our Zig functions. We write **kernels** in Metal Shading
Language, hand them to the driver as source text, and receive compiled
**pipeline states**. Work is described with a **command buffer**: an encoder
records "bind these buffers, run this pipeline over this many threads", and
`commit` hands the buffer to the GPU. Nothing has executed until then, and the
CPU learns about completion only by waiting or by a callback.

The bridge in `backends/metal/bridge.m` keeps every Metal object behind one
opaque C handle. Zig sees `nu_metal_create`, `nu_metal_register`, a handful of
operation functions, and `nu_metal_destroy`. This is the seam the specification
requires: Objective-C owns object lifetimes and exceptions; Zig owns what the
numbers mean. The `-fno-objc-arc` build flag makes every `retain`/`release`
explicit so ownership is readable rather than implied by the compiler.

**Unified memory** on Apple silicon lets the GPU read the same bytes the CPU
mapped from the model file. `newBufferWithBytesNoCopy` wraps existing pages in
an `MTLBuffer` without copying, but only whole, page-aligned ranges; the bridge
rounds each tensor's address down to a page and remembers the offset. Because
the buffer borrows our mapping, the mapping must outlive the backend. This is
the same ownership sentence we wrote for `weights.View`, now enforced across a
language boundary.

The first `nu_matvec` kernel makes the parallel structure concrete. A **SIMD
group** is 32 GPU lanes executing in lockstep. One SIMD group takes one output
row: each lane dequantizes 16 consecutive weights straight from the quantized
bytes, multiplies them with the input, and the group combines its 32 partial
sums with `simd_sum`. No F32 copy of the weights ever exists — the decoding
happens on the way into the multiply, which is the whole point of keeping
weights quantized in memory.

Why is this version slow despite being ~40× faster than the CPU reference?
Every operation is its own command buffer, and the bridge waits for each.
A decode token needs several hundred operations, so the CPU spends most of the
step asking the GPU "are you done?" instead of streaming 16 GB of weights.
The GPU also sits idle between dispatches while norms, activations, and gating
run on the CPU. The lesson is that **synchronization is a cost like bandwidth**;
the next design records a whole token's work into one command buffer and keeps
activations on the GPU.

The GPU accumulates in F32 where the CPU reference used F64. Comparing the two
requires a stated tolerance, not equality. For a dot product of 17,408 terms we
bound the error relative to the L1 mass `Σ|w·x|`, because rounding error scales
with the magnitudes being summed rather than with the final result, which can
be small through cancellation. Full-model traces then confirm that these
per-operation differences stay far below the bring-up thresholds.

## 23. Making a matvec kernel fast: bytes, lanes, and arithmetic

Decode is the loop "read all 16 GB of weights, once, per token", so the
kernel that turns quantized bytes into a dot product decides the speed of
everything. Section 22's kernel gave one SIMD group per row and decoded 16
values per lane with byte loads and a runtime `switch` on the encoding. It was
correct and reached ~100 GB/s of the M4 Pro's published 273. The specialized
kernels in `kernels.metal` (`nu_matvec_q4_k` and friends) reach 150–250 GB/s
and took decode from 4.9 to 8.5 tok/s. Four ideas did the work.

**Vector loads that match the block layout.** GGUF blocks are byte streams
with structure: in Q4_K the sixteen bytes at `16 + 32·pair + 16·half` hold the
low nibbles of one 32-value group and the high nibbles of the next. A lane
that loads those sixteen bytes as one `uint4` owns 32 values from a single
memory instruction. The trick is choosing, per encoding, which lane owns which
slice so that one aligned load yields a whole set of values: eight lanes cover
a 256-value block, four blocks fit a 32-lane SIMD group. Alignment is a fact
of the format, not a hope: 144- and 176-byte blocks are 16-byte aligned,
136-byte blocks only 8-byte, 210-byte Q6_K blocks only 2-byte, so each kernel
uses the widest load its block permits, and the Zig encoder
(`Backend.specializedMatvec`) checks the row offset and stride before it picks
the specialized kernel.

**Factor the scale out of the sum.** A group's values are `d·s·q − dmin·m`.
Instead of computing that for each value and multiplying by `x`, accumulate
`Σq·x` and `Σx` and apply the scale once:
`Σ(d·s·q − dmin·m)·x = d·s·Σ(q·x) − dmin·m·Σx`. This is algebra, not
approximation, and it has a property we rely on: with a one-hot input the
right-hand side is literally the CPU decoder's expression, so the pinned
fixture columns stay exact while dense inputs differ only by rounding order.
Q6_K's bias of 32 folds in the same way (`Σ(q−32)·x = Σq·x − 32·Σx`), and the
`Σx` terms are shared by the four rows a SIMD group handles.

**Work in the packed byte domain.** A GPU lane is a scalar processor; a
`float4` operation is four instructions. Building `float4(w & 15, (w >> 8) &
15, …)` costs two integer instructions per value before the conversion.
Assembling four codes at once with word-wide masks —
`(v & 0x0f0f0f0f) | (((h >> bit) & 0x01010101) << 4)` completes four Q5_K
values in four instructions — and converting with one `uchar4 → float4` cast
cut the per-value cost enough to take Q5_K from 113 to 210 GB/s. Explicit
`fma` matters for the same reason: `MTLMathModeSafe` never contracts a
multiply-add on its own, and we want the safe mode for comparability with the
CPU reference.

**Branches diverge, selects do not.** In a Q4_K block, groups 0–3 and 4–7
store their six-bit scales differently, and a SIMD group always contains lanes
of both kinds. An `if` on the group index makes the whole SIMD group execute
both paths; writing the decode with `?:` selects computes both once and picks.

Two measurement lessons came with the kernels. A single short dispatch per
command buffer measures the GPU's clock ramping up from idle, not the kernel;
`make bench-kernels` issues eight dispatches back to back, as a token does.
And several experiments that "should" have helped — more or fewer rows per
SIMD group, larger threadgroups, hoisting loads — changed nothing or hurt, while
Q4_K, which does strictly less arithmetic than Q5_K, stays slowest per byte.
The kernel is not arithmetic-bound in the simple sense; what limits Q4_K is a
question for GPU performance counters, which is why the plan keeps a profiling
unit (KERN-02) rather than more guesses.

## 24. Profiling a GPU: timestamps, boundaries, and what the numbers said

Section 23 ended with a question — what limits Q4_K? — and a rule: answer it
with counters, not guesses. `nuclis bench --profile` is that answer. It is
worth knowing how GPU timing works, because the constraints shaped both the
tool and its interpretation.

**You cannot put a stopwatch inside a command buffer.** The CPU only learns
when a command buffer as a whole started and finished (`GPUStartTime`,
`GPUEndTime`); that is what `gpu ms` in `bench` has always reported. To time
individual kernels the GPU itself must write timestamps into a
`MTLCounterSampleBuffer` as it executes. Apple GPUs do this only at *encoder*
boundaries, never between dispatches inside an encoder (a probe confirmed
`supportsCounterSampling` is true for stage boundaries alone). So in profile
mode `bridge.m` gives every dispatch its own compute encoder with a start and
an end sample. The bridge stays model-blind: it returns seconds per dispatch
in recording order, and the Zig `Backend`, which recorded the kernels in that
order, pairs them up and accumulates by kernel, encoding, and matrix shape
(`Profile`). Kernel names never enter Objective-C.

**Measure your measuring instrument.** Two facts had to be established before
a single number could be trusted. The timestamp unit is undocumented; a probe
showed the stamps are nanoseconds on the same timeline as `GPUStartTime` (the
first encoder's start stamp divided by 1e9 equals it to the microsecond), after
a first guess of mach ticks produced kernel sums 23× larger than the command
buffer. And the encoders themselves cost time: attributed kernel time (104.8
ms per token) is below command-buffer time in profile mode (109.0 ms), which
is itself above unprofiled (101 ms). The report prints both so the ~8 %
perturbation is visible, and `metal-check` asserts the invariant that stamps
never sum to more than the buffer that contains them. A profiled run's tok/s
is labeled not comparable; plain `bench` remains the only source of speed
claims.

**The first thing a profiler finds is usually not a kernel.** The very first
run reported 1,864 command buffers for 29 token steps: 64 per token. The
CLI's layer observer — installed on every run for Ctrl-C and the time limit —
received activations after each layer, and on the GPU plan that means commit,
wait, read, begin, sixty-four times a token. The plan was "one command buffer
per token"; the program never was. Splitting the observer into a values-free
`check` (runs while recording, costs nothing) and a trace-only `layer` took
decode from 8.54 to 9.69 tok/s with no kernel change: GPU time per token was
already ~101 ms, and wall time simply fell to meet it. Synchronization hides
in plain sight because it is nobody's kernel.

**Compare per block, not per byte.** With real numbers in hand, the Q4_K
question dissolved. GB/s differs by encoding (Q4_K 158, Q5_K 193, Q6_K 241),
but each of those matvecs on the same 17,408×5,120 shape takes 0.81–0.91 ns
per 256-value block. Q4_K is not slower; it has fewer bytes per block (144
against 176 and 210) to show for the same instruction work. The limiter the
four kernels share is per-block instructions — decode, scale arithmetic, the
eight-lanes-per-block structure, input loads — and the bandwidth floor
corresponds to about 0.5 ns per block. Section 23's suspicion about Q4_K's
access pattern was wrong, and the table that refuted it fits in six rows.
Meanwhile the "4 % of bytes on the generic kernel, ~9 ms" estimate turned out
to be 14.8 ms, because Q3_K and IQ3_S run at 40 GB/s rather than the assumed
90; and 209 RMSNorm dispatches of ~13 µs each cost 2.7 ms, launch-bound work
that only merging can remove. Estimates from micro-benchmarks were off by 2×
in places; the measured budget is now what the remaining plan is built on.

## 25. Prefix reuse: continuing a session without rewinding

`nuclis agent` is the first code that holds a *conversation* against the engine
rather than a single prompt, and it exposes a constraint hidden until now.
A transformer with only a KV cache can "forget" the last turn by truncating
the cache to an earlier position. Qwen3.8's DeltaNet layers cannot: their
recurrent state is a fixed-size matrix folded over every token so far, and no
earlier value survives to truncate back to. So the chat never edits the
session's past. It only ever appends, or resets and replays.

Appending is enough for the common case because the chat template is
prefix-stable. Rendering the conversation `[u1, a1, u2]` yields the text for
`[u1, a1]` plus a suffix (`<|im_end|>`, the new user block, the assistant
header). The chat therefore keeps `seen`, the exact text the session has
consumed, and on the next turn computes `increment(seen, full)`: the suffix
if `full` starts with `seen`, otherwise "replay". Two details decide whether
the prefix matches. First, the assistant's reasoning must travel through
`reasoning_content`, not inside `content`, because the template renders those
differently for past turns than the model emitted them live. Second, the last
token sampled is never fed back — the loop stops after choosing the stop token
or the budget's final token — so `seen` excludes it (`generated[0..n-1]`);
otherwise the prefix would include a token whose state the model never
computed. When effort changes, the system-side text differs and the
comparison fails honestly: the status bar shows "replaying".

The other structural move is where interactivity lives. Raw terminal mode
turns off ISIG, so Ctrl-C arrives as byte 3 instead of a signal; the chat
calls `interrupt.request()`, the same flag the signal handler sets, and the
model's `check` observer sees it at the next layer boundary. Polling keys and
redrawing happen in a new `Hooks.step` callback that runs after every model
step — a *hook after* the step, not an observer *inside* it — so the
production path keeps one command buffer per token (section 24's lesson) and
the UI still refreshes at token rate. Nothing in `inference` learned
what a terminal is.

## 26. Three-bit weights: geometry can be shared, decoding cannot

Q3_K and IQ3_S both store 256 weights in 110 bytes, but those bytes describe
different mathematics. Q3_K combines two low bits with an inverted sign bit
and a signed scale for each group of sixteen. IQ3_S uses a nine-bit index
into a grid of four magnitudes, separate signs, and an odd scale for each
group of thirty-two. Equal storage size does not imply equal decoding.

The new Metal kernels share the work assignment: eight lanes divide a block,
each lane reads thirty-two activations once and reuses them for four output
rows. A compile-time boolean selects the decoding body, producing two named
kernels without a runtime encoding branch. Their 110-byte blocks also mean
that a block can begin only two bytes past a wider alignment boundary. Packed
16-bit vector loads respect that contract; an ordinary `uint4` would not.

Q3_K factors its offset out of the dot product: `Σ(q−4)x = Σqx−4Σx`,
then applies each sixteen-value scale. IQ3_S instead looks up signed grid
values and applies its thirty-two-value scale once. A one-hot input isolates
a single decoded weight, so exact fixture checks prove the bit layout, signs,
and scales; dense random inputs test the changed floating-point reduction
order against the F64 CPU reference with the existing error bound.

## 27. Calibration data is used before inference

The downloaded `imatrix_unsloth.gguf` is an importance matrix: statistics from
activations observed while running calibration inputs. Quantization can use
those statistics to penalize weight errors more heavily in input channels that
carry more signal. The output of that process is the quantized model we run.
The statistics are not another layer to execute and do not restore precision
when loaded next to already-quantized weights.

Our inspector can read this file because GGUF is a container, not a guarantee
that its tensors form a language model. It reports 992 F32 tensors with no
language-model architecture. The companion vision projector is another example:
its GGUF contains a vision encoder and merger, which need their own execution
schedule. See the [companion inventory](reference/gguf-inspection.md#companion-files-in-modelsqwen-2026-09-08)
for the inspected contents and their current uses.

## 28. Merging GPU work without changing the math

Several projections in a Qwen layer read the same normalized activation vector.
Before KERN-04, each weight matrix had a separate GPU dispatch even when there was
no dependency between them. A segment table now describes their row ranges,
encodings, and buffer offsets in one dispatch. A threadgroup selects one segment
and runs the same arithmetic body that the standalone kernel uses.

This is a useful distinction between a uniform branch and divergent work. All
lanes of a SIMD group select the same weight encoding, so they execute one
decoder together. Different groups can select different encodings without
forcing every lane to perform every decoder. Binding selection works the same
way; sharing a buffer binding does not require copying its slices together.

The feed-forward pair adds a dependency: the output is `silu(gate) * up`. Our
first version computed both projections in each SIMD group and reduced the
number of dispatches, but measured slower than the DeltaNet-only merge. The
kept version assigns two SIMD groups to each projection. They exchange just
sixteen reduced floats through threadgroup memory, synchronize at a barrier,
and write the combined result. The temporary full up vector is avoided on this
path. Each projection keeps its original accumulation order, and tests compare
merged results with standalone operations exactly. Fewer dispatches are a
useful mechanism; measured token latency decides whether a layout helps.

## 29. Negative results: why fewer instructions can be slower on a GPU

KERN-05 asked a narrow question about the specialized matvec kernels: they spend
0.8–0.9 ns per 256-value block whatever the encoding, and the bandwidth floor
is about 0.5, so what is the per-block work that costs the difference? Five
hypotheses were built and measured with `make bench-kernels`; none reached
the 5 % bar, and the kernels shipped unchanged. The results are still worth
understanding, because each one rules out a mental model.

**Fewer ALU instructions made Q4_K slower.** The scale decode of a Q4_K block
is a chain of shifts, masks, and selects. One variant decoded each group on a
single lane and shared the results with four `simd_shuffle`s: strictly less
arithmetic per lane, and 12 % slower. A sibling variant that decoded on half
the lanes behind an `if` — which on a SIMD machine executes the same
instructions, masked — and shuffled the results was a few percent faster.
If instruction count decided the time, the first variant would have won. It
lost, so the compiler's schedule (which loads and shuffles wait on which) is
what the time is made of, and a source-level count of operations does not
predict it.

**Latency hiding is paid for in registers.** The classic way to hide a load's
latency is to give the core independent work: process two blocks per loop
iteration into separate accumulators. That variant was 35 % slower. Every
value that is live across the two blocks — decoded `float4`s, input vectors,
two accumulator sets — needs a register, and a GPU core has a fixed register
file that it divides among the threadgroups it keeps resident. More registers
per thread means fewer resident threadgroups, and resident *other* threadgroups
are how this GPU hides latency already. In-lane parallelism competed with
occupancy and lost. The same lesson appears in the input-staging experiment:
copying the 5,120-float input into 20 KB of threadgroup memory replaced cached
loads with a barrier and a memory reservation that limits how many groups
share a core, and it cost 7–9 %.

**"Diagnostic builds" tell you what the limiter is not.** Compiling with
`MTLMathModeFast` — never shipped, since comparability with the CPU reference
needs safe mode — changed nothing for five encodings and made Q5_K 12 %
slower. The dot products are already explicit `fma`, so contraction had
nothing to add; the reassociation it allows changed Q5_K's schedule for the
worse. That one experiment removes "ALU issue" from the list of candidates
without any counter.

**What is left.** The limiter is neither instruction count, nor input traffic,
nor arithmetic; it is sensitive to register footprint and to instruction
scheduling within the eight-lanes-per-block structure. Apple's GPU limiter
counters would name it — `xctrace` runs from the command line here, but on
this machine that counter profile is reported unsupported and its tables come
back empty. The next step is not another instruction cut; it is a different
geometry (for example one SIMD group per block with a wider reduction), which
changes what each lane keeps live. Negative results are recorded with their
numbers in [reference/metal-backend.md](reference/metal-backend.md#kern-05--per-block-cost-research-2026-09-08-closed-without-a-kernel-change)
so the next session starts from what is known instead of re-running it.

## 30. Sampling without reading the vocabulary back

Greedy decoding on the GPU already avoided the 1 MB logit readback with a
two-pass argmax (section 24 measured what that saved). Sampled decoding
could not: temperature, top-k, and top-p need a *distribution*, and the
reference sampler builds it by sorting all 248,320 logits on the CPU. That
cost 19 ms per token — a fifth of a decode step — for a decision that in
practice touches a few dozen candidates. KERN-06 moves the candidate search to
the GPU while keeping one hard promise: the token is **bit-identical** to the
reference sampler's for the same seed. Three ideas make that possible.

**Separate what needs the whole vocabulary from what touches the token.**
The reference path does four things: sort, exponentiate the retained set,
walk it until the running mass reaches `top_p · sum`, and draw. Only the
sort needs every logit. The GPU therefore returns the best 256 logits in the
sort's exact order (ties break on the lower id in the kernel's comparison,
the same rule as the CPU comparator), and the CPU redoes exponentiation,
walk, and draw on those values in the reference order with the reference
F64 arithmetic. Identical inputs and identical operations give identical
bits; no tolerance is involved on this path. With `top_k ≤ 256` the retained
set is entirely inside the readback, so this covers the official Qwen
profiles (top-k 20) with nothing left to approximate.

**When a number cannot be exact, bound it and guard the decision.** With
`top_k = 0` the nucleus walk compares against the sum over the *whole*
vocabulary, which the GPU must compute — as F32 partial sums, not the F64
descending-order sum the CPU would take. The two differ by parts in 10⁸
(measured: 9.6e-8 on a flat vector). Instead of pretending that is zero, the
sampler treats the GPU sum as an interval `total · (1 ± 1e-5)` and accepts a
comparison only when the running mass lies clearly outside it; a comparison
inside the band, or a nucleus larger than 256 candidates, asks for the full
logits and runs the reference path. Crucially the random draw happens only
after the decision, so a deferred token consumes no randomness and the
fallback produces exactly the token the reference would have. The test that
matters perturbs the sum by more than the real error in both directions and
checks that the sampler either agrees with the reference or defers — never
disagrees. In the model, at temperature 0.7 the fallback rarely triggers; at
1.5 with a wide nucleus it triggered on roughly half the tokens and every
token still matched.

**A k-round selection is cheaper than a sort when k is tiny.** Sorting
248,320 values on the GPU is real work; selecting 256 is not. Each partial
threadgroup keeps its slice in registers and runs 256 rounds of "best
untaken value" — a SIMD shuffle reduction and an eight-way pick per round —
and the final pass merges 64 sorted lists with one cursor each. The three
dispatches cost far less than the 1 MB copy and the CPU sort they replace,
and the readback is 2 KB. What limits the design is register residency: the
partial pass holds 16 values per thread, which bounds the vocabulary it
accepts at 262,144; a larger vocabulary would need a second level, not a
different idea.

The practical rule this leaves behind for MODL-01: anything that must adjust
*every* logit before selection (presence and repetition penalties) is a
fallback case by construction, because the readback happens after the
projection and before the CPU sees the vector.

## 31. Why prefill is a different kernel

Decode reads every weight once per token and does about two floating-point
operations per weight byte read: it is bound by memory bandwidth, and the
whole kernel story so far (sections 23–29) was about reading bytes faster.
Prefill has the same weights and many tokens at once, and that changes which
resource is scarce.

**The arithmetic intensity flips.** A 512-token prompt multiplied one token
at a time reads 16 GB five hundred and twelve times. Multiplied as a matrix,
it reads 16 GB once and does 512 times the arithmetic on each byte. The
memory bus that bounded decode at ~59 ms per token is now far from the
limit; the ALUs are. That is why the same model that decodes at 11 tok/s can
prefill at 89 in the reference: not a faster memory path, a different ratio.
Before ENGN-02, nuclis prefilled at decode speed because it *was* decode: the
prompt loop called `step` per token.

**A tile kernel amortizes the decode of the weights.** `nu_matmul` decodes a
32-row slice of the weight matrix once into threadgroup memory and multiplies
it against 32 tokens with `simdgroup_float8x8`, the SIMD-group matrix
instructions that keep an 8×8 block in registers across the multiply. Each
weight byte is decoded once per 32 tokens instead of once per token, and
the multiply-accumulate runs on operands already in registers. Measured
alone on the feed-forward shapes it reaches 2.3–3.1 TFLOP/s, a ceiling of
42–57 tok/s for the whole model with the generic decoder in the tile.

**Batching changes the numerics, not the semantics.** The tile accumulates
products in a different order than the matvec, so a chunked prompt and the
same prompt stepped token by token do not agree bit for bit: max abs 3.1e-5
on the final logits, same argmax. The trace comparison against the reference
(`make compare`) therefore keeps the per-token path, and the equivalence
between the two paths is its own recorded check. This is the general rule
from section 7 again: a faster path is accepted when its difference from the
slow one is measured and bounded, never because the output reads well.

**What did not get batched in ENGN-02.** Attention for a chunk still ran the
decode kernels per token over the growing cache, and DeltaNet still updates
its matrix one token at a time, because both have a sequential dependency
that a plain matmul cannot express. That is why prefill measured 51 tok/s at
543 tokens but 33 at 3,547 after ENGN-02: the per-token mixer loops grow with
the cache. Tiled causal attention (ENGN-03, section 34) and the chunkwise
DeltaNet form (ENGN-04) are the batched versions of those two dependencies, and
each is a new piece of math with its own CPU reference, not a kernel tuning.

## 32. Penalties and min-p: how defaults distort a distribution

Sections 21 and 30 treated the sampler as a way to draw from the
distribution the model produced. Real deployments rarely draw from that
distribution unchanged: a model card ships *recommended* settings per mode,
and those settings reshape the logits before any draw. MODL-01 implemented the
two reshaping controls the official Qwen3.8 profiles use and made the
profiles the defaults, which forced three questions to be answered
precisely rather than by habit.

**A penalty is state, not policy.** Presence and repetition penalties lower
the logits of tokens that have already appeared. "Already appeared" is a
property of the *session* — the prompt tokens and everything generated
since the last reset — not of the sampler, which is a policy that could be
swapped mid-conversation. So the history lives in its own object, a bitset
over the vocabulary (one bit per token, 31 KB), owned by the caller and
reset exactly where the session is reset: a new chat session, a replay, a
cancelled turn. The engine loop observes tokens into it; the sampler only
reads it. Keeping state and policy apart is what lets the chat change the
sampling profile with Ctrl-T without losing what the model has seen, and
what makes the test for "the history mirrors the session" a one-line
invariant instead of a search through the code.

**The order of operations is the specification.** "Repetition penalty
1.5" means nothing until you say whether it divides before or after the
presence penalty subtracts, whether it acts before or after temperature,
and whether `min_p` filters the top-k set or the whole vocabulary. Every
implementation makes these choices, few state them, and two engines that
agree on the numbers can still produce different tokens. nuclis pins the
chain — penalties on raw logits, then sort and temperature, then top-k,
then `min_p`, then top-p over the survivors — with tests that would fail if
any two stages swapped: a repetition-then-presence case whose value
(`2/4 − 1 = −0.5`) differs from the other order (`(2 − 1)/4 = 0.25`), and a
`min_p`/top-p pair whose drawable set differs depending on which sum top-p
uses. The chain is documented in
[generation.md](reference/generation.md#sampling-profiles-and-the-selection-chain-modl-01)
because the reference oracle semantics (llama.cpp's) are a contract to
match, not code to copy.

**`min_p` is a ratio, and that makes it cheap.** The rule "discard
candidates below `min_p` times the top probability" looks like it needs
the softmax, which needs the sum over the vocabulary. It does not: after
the sampler exponentiates relative to the maximum, each candidate's weight
is *exactly* `p / p_max`, so the survivors are simply the leading candidates
whose weight is at least `min_p` — a prefix of the already-sorted list, no
denominator involved. That observation is what keeps `min_p` on the GPU
path of section 30: the readback holds the first 256 candidates in order,
and the prefix is either inside it (decide exactly, with the CPU sum over
the survivors) or covers all of it (defer to the full path). The one
configuration that was never GPU-eligible — `top_k = 0` with `top_p = 1`
— becomes eligible as soon as `min_p > 0`, and the equivalence check
confirmed bit-identical tokens in exactly that configuration.

**What the profiles cost.** The thinking profile (temperature 1, top-k 20,
top-p 0.95) runs on the GPU path at greedy speed. The instruct profile adds
a presence penalty of 1.5, and a penalty must touch *every* logit of a
token in the history before the sort — but the GPU already sorted before
the CPU could touch anything. So the instruct profile falls back to the
full 1 MB readback on every token and measured 8.69 tok/s against 10.60
greedy. That is not a defect of the design; it is the honest cost of a
default the model card asks for, measured and recorded, with the fix (a
kernel that applies the history on the device) planned as KERN-13. The
engineering rule this leaves behind: when a default changes the output
distribution, make its effect on both correctness and speed a measured
number before calling it the default.

## 33. Configuration as a schema: comptime reflection over a struct

Every command line up to APPS-03 restated the same facts: which model, which
backend, how much context, how many tokens, which reasoning effort, which
sampling overrides. APPS-03 moved them into `~/.nuclis/nuclis.json`. The
interesting part is not the file but how little code it takes when the
schema *is* a Zig struct.

**The struct is the schema.** `src/config.zig` declares `Config` with
three section structs (`engine`, `generation`, `agent`, then `chat`) whose fields carry
their defaults. Everything else derives from it: the JSON written by
`config init` is `std.json.Stringify` over `Config{}`; loading walks the
parsed `std.json.Value` tree against the struct with an `inline for` over
`@typeInfo(T).@"struct".fields`, recursing into fields whose type is a
struct and treating every other field as a key; `config show` walks the
same fields to print them. Adding a setting is adding one field with a
default. Nothing is registered twice, so the file, the loader, the
validator, and the printer cannot disagree about which keys exist.

**Keys are addressed by path, resolved at compile time.** Each key gets
an index in a pre-order walk of the struct (`leafIndex(Config,
"engine.ctx_size")`), computed at `comptime`, and the source of every
value (`default`, `file`, `flag`) is a plain array indexed that way. A
misspelled path in the code does not compile; a misspelled key in the file
is rejected at load with its dotted path in the message. The same
`inline for` that reads a section rejects unknown keys first, so a typo is
reported even when the rest of the section is valid.

**Errors carry no payload, so a diagnostic travels beside them.** Zig
errors are names, not values. A validation failure returns a typed error
(`UnknownConfigKey`, `InvalidConfigValue`, `UnsupportedConfigVersion`) and
writes the human reason — the key, the accepted form or range, the found
value — into a fixed-size `Diagnostic` the caller passed in. `main` prints
both. The same pattern already existed in the Metal bridge; the rule is
that the error decides control flow and the diagnostic decides the message.

**Overrides compose before they apply.** The file's sampling entries and
the flags are both `sampling.Overrides` (all-optional mirrors of the
sampler's options). `Overrides.merge` layers them — a non-null flag beats
a non-null file value — and only then does the profile of the reasoning
mode (§32) take the merged overrides. `null` in the file therefore means
"whatever the profile says", which is why the defaults file is written
with explicit nulls: it documents the knobs without freezing the model's
recommended values into user state. `bench` deliberately ignores the file's
sampling and budget so a benchmark is reproducible from its command line
and the report, which now names the configuration file it read.

**A run-time map inside a compile-time schema (MODL-03).** The `models`
registry broke the rule that every key is known at compile time: its keys
are the user's entry names (`gemma`, `qwen-q3`). The fix is one exception
in the walk, not a second loader. `isSection` and `leafCount` treat the
`Models` type as having no key slots, `Models` prints itself through a
`jsonStringify` method so the file shows a map rather than a list of
pairs, and each entry is parsed by the same section walker with a
run-time path prefix (`models.gemma.`) and no origin slots — the walker's
`base` became a `comptime ?usize`, and `null` prunes the origin write at
compile time instead of guarding it at run time. Resolution then has five
layers, defaults < profile < file < entry < flags, each key remembering
which one produced it, and `config show` prints the effective value with
that layer's name: a `null` in the file is shown as the profile's number
labelled `profile`, which closed the question of what `null` means better
than any sentence in the docs did.

**What was left out on purpose.** No per-key environment variables, no
per-project files, no `agent` section until the code that reads it
exists: a key with a default can be added without a schema bump (the
registry was), so the file grows with the engine rather than ahead of it.

## 34. Attention without the score matrix: online softmax in tiles

Section 18 described attention as reading a visible prefix: for one query,
score every cached key, softmax the scores, and average the values by those
weights. The decode kernels do exactly that in three dispatches with a
`[head][visible]` score buffer between them. For a prefill chunk of `C`
queries that buffer would be `C × visible` per head, and the three-dispatch
shape would run once per query row. ENGN-03 replaces it with one dispatch per
layer that never materializes the scores. Two ideas make that possible.

**Softmax can be computed in one pass with a running max.** The softmax
weight of key `j` is `exp(s_j − m) / Σ exp(s_i − m)`, where `m` is the
maximum score; subtracting `m` is what keeps `exp` finite. Written as one
pass over key tiles, you do not know `m` until the end. The online form
keeps, per query row, the max seen so far `m` and the sum `l` under that
max, and the running output `o = Σ exp(s_i − m) · v_i`. When a new tile
raises the max to `m'`, every term computed so far is too large by
`exp(m' − m)`, so multiply `l` and `o` by `α = exp(m − m')` and continue.
The final answer is `o / l`. Nothing is stored per key; the tile is
consumed and forgotten. This is the algorithm behind "flash attention"; in
nuclis it is written from the identity above and checked against the F64
reference, which computes the softmax the plain way per query row.

**Tiles map onto the SIMD-group matrix unit.** Section 31 introduced
`simdgroup_float8x8`, the 8×8 block a SIMD group multiplies in registers.
Attention has two products with the same shape: scores `S = Q·Kᵀ` and the
weighted values `P·V`. A SIMD group owns 8 query rows; for a tile of 32
keys it accumulates four 8×8 score blocks over the 256-wide head, writes
them to threadgroup memory, and then the plain threads do the row-wise
work — mask, max, `exp`, sum — with each lane owning one row and eight
columns and combining across the four lanes of a row with SIMD shuffles.
The probabilities go back through the same threadgroup tile as the left
operand of `P·V`, accumulated into 32 register-resident 8×8 blocks per
SIMD group (the 8 rows × 256 value channels of the output). Rescaling
those blocks by the per-row `α` has no direct matrix instruction, but a
diagonal matrix with `α` on its diagonal does exactly that as a multiply,
so the kernel builds one in threadgroup memory and applies it only when a
row's max actually moved.

**Causality is a bound on the loop plus a mask inside the last tile.** Row
`t` of the chunk sits at cache position `position + t` and may see keys
`0 .. position + t`. A SIMD group's eight rows have limits that differ by
at most seven, so it walks key tiles up to the largest and sets the score
to −∞ for any key beyond a row's own limit; `exp(−∞) = 0`, and a weight
that is exactly zero cannot leak whatever value sits in that row. The test
places `1e30` in every key and value row after one row's horizon and
requires that row and its predecessors to match the clean reference. The
last 8-key block may also run past the end of the visible range, and a
matrix load has no bounds check, so those blocks are staged through
threadgroup memory with the missing rows zeroed: the kernel reads exactly
the rows it is allowed to and nothing depends on what lies past them.

**What the measurement says.** The per-row F64 comparison over 256 rows and
a 2,048-row cache gives max abs 3.9e-7, tighter than the three-dispatch
decode path, because the online form subtracts a max that is never smaller
than the plain one and the products accumulate in the matrix unit. The
prefill rate no longer falls with prompt length: the attention loop that
grew with the cache is gone, and what remains per token inside a chunk is
DeltaNet's recurrence, the subject of ENGN-04.

## 35. SIMD groups: which layer they live in

The short answer: SIMD groups exist only inside the Metal Shading Language
kernels in `kernels.metal`. Zig never sees one, and the Objective-C bridge
never sees one. Both of them only agree on numbers that the kernels depend
on. Nothing in the CPU reference uses Zig's own vector types either; that
is by design, since the F64 loops are the oracle and stay plain.

**The hierarchy a GPU dispatch creates.** When the bridge calls
`dispatchThreadgroups`, it asks for a grid of threadgroups and says how
many threads each one has. The hardware then runs each threadgroup as a set
of SIMD groups of 32 threads, called lanes, that execute the same
instruction at the same time on different data. So there are four levels:
the grid, the threadgroup, the SIMD group, the lane. A kernel picks its
geometry by choosing the thread count. Ask for 32 threads and the
threadgroup is one SIMD group. Ask for 128 and it is four. Ask for 256 and
it is eight.

**What each layer knows.**

- **Zig** (`backends/metal/root.zig`) is the host. It chooses the grid and
  the thread count in every `dispatch` call, validates buffer sizes, and
  holds the constants the kernels assume: four rows per SIMD group in the
  matvec, four SIMD groups per matvec threadgroup, 32-row padding for the
  matmul, 8-row padding for chunk attention. It does not know what a lane
  is. When a kernel comment says "must match `rows_per_simdgroup` in
  root.zig", that is the whole contract between the two layers: two numbers
  that have to agree.
- **The Objective-C bridge** (`bridge.m`) compiles the kernel source into
  pipelines, binds buffers, passes the parameter struct, and issues the
  dispatch with the geometry Zig gave it. Its one SIMD-adjacent duty is a
  check that the thread count does not exceed the pipeline's maximum. It
  never mentions a SIMD group.
- **The kernels** (`kernels.metal`) are where the concept is real. Two
  attributes tell a thread where it sits: `thread_index_in_simdgroup`, its
  lane from 0 to 31, and `simdgroup_index_in_threadgroup`, which SIMD group
  of the threadgroup it belongs to. Everything else is built from those two
  numbers.

**Three things a SIMD group can do that a plain thread cannot.**

1. **Reduce across lanes without memory.** `simd_sum(x)` returns the sum of
   `x` over all 32 lanes to every lane, in a few cycles, with no shared
   memory and no barrier. The decode matvec is built on this: one SIMD
   group owns an output row, each lane multiplies a stride of 32 columns,
   and one `simd_sum` finishes the dot product. The decode attention scores
   and the softmax use the same pattern with `simd_max`.
2. **Exchange values between specific lanes.** `simd_shuffle_xor(x, 1)`
   gives each lane the value from the lane whose index differs in bit 0.
   Two of these combine four lanes. The chunk attention kernel uses that to
   compute a row's max and sum, where four lanes own the 32 columns of one
   query row. `simd_shuffle(x, lane)` reads a value from any named lane,
   which is how the same kernel broadcasts a row's rescale factor when it
   builds the diagonal matrix.
3. **Multiply 8×8 matrices held across the group.** A `simdgroup_float8x8`
   is 64 floats spread over the 32 lanes, two per lane. `simdgroup_load`
   fills one from memory, `simdgroup_multiply_accumulate` does an 8×8 by
   8×8 product on the matrix unit, and `simdgroup_store` writes it back.
   The individual lane never sees which two elements it holds, and the
   layout is deliberately undocumented. That is why the attention kernel
   round-trips its score tile through threadgroup memory: row-wise work
   like the softmax has to happen on plain threads, and the matrix unit
   only does products. The matmul tile and the chunk attention kernel are
   the two users in the tree.

**Barriers, and why there are two.** Lanes in a SIMD group execute in
lockstep, so when one SIMD group writes a tile to threadgroup memory and
then reads it back, it only needs `simdgroup_barrier`, which is close to
free. When SIMD groups share data with each other, as the matmul does when
every thread decodes part of the weight tile all four groups then
multiply, the kernel needs `threadgroup_barrier`, which stops the whole
threadgroup. The chunk attention kernel gives every SIMD group its own
scratch region for exactly this reason: no group ever waits for another,
and a group whose query rows are all past the end of the chunk can simply
return.

**The rule for choosing the geometry.** Put in one SIMD group the work that
wants to be reduced or exchanged, put in one threadgroup the work that
wants to share a tile of memory, and leave the rest to the grid. The matvec
reduces along a row, so a row is a SIMD group. The matmul shares a decoded
weight tile, so a tile is a threadgroup. Chunk attention reduces along a
query row and needs no sharing, so a row belongs to four lanes and a SIMD
group is independent. None of this reaches Zig except as the thread count
in the dispatch. The geometry of every kernel is tabulated in
[reference/metal-backend.md § Kernel geometry](reference/metal-backend.md#kernel-geometry).

**What Zig SIMD would be, and why it is absent.** Zig has `@Vector` types
that map to CPU vector instructions, a separate mechanism on a different
processor. The CPU reference in `backends/cpu/` does not use them. It is
written as scalar F64 loops so it is the plain statement of the math the
GPU kernels are checked against, and speed there is not a goal.

## 36. Chunkwise recurrence: paying for parallelism with a triangular solve

Section 19 explained DeltaNet as a memory that is updated in place: each
token decays the 128×128 matrix, writes in the prediction error for its
key, and reads the result with its query. That is inherently sequential.
Token `t` cannot be processed until token `t−1` has updated the matrix,
which is why ENGN-02 and ENGN-03 left 48 layers stepping one token at a time inside
a prefill chunk while everything else was batched. ENGN-04 removes that loop,
and the trick is worth understanding because it is the same move that
makes attention parallel: turn a chain of matrix updates into inner
products between tokens.

**Unroll the recurrence.** Write one step as `S_t = a_t S_{t−1} + u_t k_tᵀ`,
where `u_t` is the correction the token writes and `k_t` its key. Apply it
twice and the previous matrix drops out: `S_t` is the starting matrix
decayed by the product of all decays so far, plus every earlier
correction, each decayed by the decays that came after it. Nothing in that
sum needs the intermediate matrices. But the correction `u_t` was defined
using `S_{t−1} k_t`, the prediction, and that seems to bring the chain
back.

**Except the prediction only needs inner products.** Substitute the
unrolled `S_{t−1}` into `S_{t−1} k_t` and every term is `u_s (k_s · k_t)`,
a correction scaled by how similar two keys are. So `u_t` equals its "fresh"
value, `β_t (v_t − γ_t S_0 k_t)`, minus a weighted sum of the earlier
corrections in the chunk, where the weights are decays times key
similarities times `β_t`. Written for all `C` tokens at once that is a
matrix equation `(I + A) U = B` with `A` strictly lower triangular, since
row `t` only refers to rows before it. A triangular system is solved by
forward substitution, one row after another, and each row is a dot product
of things already known. That is the WY form, named after a factorization
trick from numerical linear algebra, and it is exactly what
`recurrent.deltaChunk` does in F64.

**What was bought and what it costs.** The sequential form does `C` updates
of a 128×128 matrix, each a dependency on the last. The chunkwise form does
one `C × C` table of key similarities, a forward substitution over it, and
then reads the outputs and the final matrix as sums over the chunk. The
matrix is touched twice per chunk instead of once per token, and the
per-token work turned into small matrix products, which is what
`simdgroup_float8x8` is for. The cost is `C²` inner products that the
sequential form never computed. For `C = 64` that is cheap; for a whole
prompt it would not be, which is why the chunk is a fixed size and the F32
state is carried between chunks exactly as it is between tokens.

**Why decays are handled as logarithms.** The decay ratio between two
tokens is a product of up to 64 factors, each between 0 and 1. Multiplying
them out and dividing is fine on paper and fragile in floating point. The
reference keeps a running sum of log decays and takes `exp` of a
difference, which is always at most 1 and never divides by a number that
underflowed. The GPU kernel will inherit that choice.

**How it is proven.** The chunk function is checked three ways, in
increasing distance from the mathematics. Against a sequential loop that
keeps its state in F64, a model-shaped random chunk is bit-identical after
the final cast to F32: the algebra is exact. Against the production
`delta`, which rounds the state to F32 between every token, the difference
is 2.4e-7, and it is the same whether the 64 tokens run as one chunk or as
40 then 24 with the state carried: chunk boundaries are invisible. And the
three pinned steps recorded from the reference implementation, run as one
chunk, reproduce their pinned outputs. The kernel that follows in the next
session will be measured against this function, and this function remains
measured against the per-token step. A faster form of a recurrence is
accepted only when the slow form says it is right.

**The kernel, in one paragraph.** `nu_delta_chunk` keeps nothing large in
threadgroup memory. The 128×128 state of a head is 64 KB against a 32 KB
limit, so instead of holding it, four threadgroups split a head into blocks
of 32 state rows and stream those rows through the matrix unit twice per
sub-chunk: once to read the starting matrix's contribution (`S₀·Kᵀ` and
`S₀·Qᵀ`) and once to write the carry (`γ S₀ + Wᵀ K`). The row split is
sound because, given the chunk's keys, queries, decays, and betas, each
value row of the state evolves independently. Everything token-by-token
lives in six 32×32 tiles of threadgroup memory, and the only sequential
part, the forward substitution, is 32 rows long with one thread per value
column. Measured against the F64 chunk reference the kernel's outputs
differ by 3.4e-8 and its carried state by 1.2e-7.

## 37. Arithmetic intensity inside a tile: decoder, operand width, and threadgroup memory

§ 31 explained why prefill needs a tile kernel: a weight decoded once must
be multiplied against many tokens, or the decode dominates. ENGN-05 asked the
next question: once every weight is decoded once per chunk, what bounds the
tile? The plan's guess was the decoder, because the tile's speed differed
by encoding (Q4_K 3.1 TFLOP/s, IQ3_S 2.3) and the decoder is the only part
that knows the encoding. That guess was half right, and the way it was
wrong is the lesson.

**Measure one lever at a time.** The specialized decoders (the same
vector loads and packed-byte tricks the decode matvecs use, § 29)
collapsed the spread: every encoding landed at 13.1–14.4 ms for the same
product. So the decoder *was* the difference between encodings. But the
common floor it revealed, about 3.4 TFLOP/s, was not much above the best
encoding's old number: the decoder had been a tax on top of a slower
matrix phase, not the matrix phase itself. Had the levers been combined
first, the table would have shown one number and no explanation.

**Arithmetic intensity is a per-tile property.** A 32×32 output tile with
a 64-column K step does 2·32·32·64 multiply-adds per step and loads a
32×64 weight tile plus a 32×64 activation tile: 16 loads of 8×8 per SIMD
group against 32 matrix MACs. Doubling one edge of the tile to 64×32 makes
that 24 loads against 64 MACs, a better ratio, and measured *no change*.
That is a strong statement: the loads were not the bound either. What
remained was the matrix unit's F32 rate and the number of tiles the GPU
can keep in flight at once.

**Half operands are worth 1.5×, and not for the reason expected.** Loading
the decoded weights and the activations into threadgroup memory as `half`
and multiplying `simdgroup_half8x8` inputs into `simdgroup_float8x8`
accumulators (the sums stay F32) took the 32×32 tile from 12.3 to 11.2 ms:
a real gain, not the 2× a doubled ALU rate would give. The larger effect
came through memory: a half tile is half the bytes, and this GPU has a
cliff in how many threadgroups a core can hold as a function of their
threadgroup memory. 16 KB per group runs at full speed; 24 KB and 32 KB run
five times slower, on every encoding, with nothing else changed. In F32 a
64×64 tile needs 32 KB and collapses; in half it needs 16 KB and is the
fastest shape measured, 9.0 ms, 5.1 TFLOP/s. The operand width bought the
tile shape.

**What did not help, recorded.** Prefetching the next K step's segments
into registers before this step's MACs, the classic way to hide load
latency, was slower: the 16 extra `float4` per thread cost registers, and
registers are the other occupancy currency. Predicating the MACs and
stores on the 8×8 blocks that hold real rows and tokens, to skip padding
on the last tile of a chunk, cost 30 % on full tiles and gained nothing on
short prompts. Both negatives are in the reference table so nobody
re-measures them by accident.

**Short prompts are a different regime.** A 22-token prompt is one token
tile. The 5,120-row down projection then dispatches only 80 threadgroups of
64 rows across 20 cores, each walking 272 K steps with a device round trip
per step; the weights stream at 20–25 GB/s where the matvec path streams
them at 170. The 32×32 set for chunks of at most 32 tokens recovers the
lost threadgroups (39 tok/s, above the 35 before ENGN-05), but the honest
number is that 22 tokens take 563 ms and one token takes 95 ms: a
small-M product wants a different design (split the K dimension across
threadgroups and reduce), which is planned as KERN-14 rather than
squeezed into a unit about the large-tile ceiling.

**The numerics contract moved, and the record says by how much.** The F32
tiles were bit-identical to the generic tile and 2.8e-5 from the stepped
path on the final logits. Rounding both operands to half is a different
computation: the specialized tile is now compared with the generic F32
tile on the same inputs (worst difference 3.8e-5 of Σ|w·x| in the
fixtures), and chunked prefill differs from stepped by 2.4–2.6e-3 max abs
with the same argmax, against the accepted bound of 2e-2. The generic tile
stays F32, so dense F32/F16 rows of any magnitude remain exact and the
fixture rows with block scales near 65,504 (which would overflow half)
still decode bit for bit. The half path assumes the model's activations
stay below half's range; the generation check on real prompts is the
evidence, and the fallback is one constant.

## 38. Half the bytes: what an F16 cache rounds, and what it must not

Built in KERN-07. Every full-attention layer appends one key row and one value
row per token, 1,024 F32 channels each, and every later token reads all of
them back. At 32K tokens that is 4 GiB read per decoded token and 4 GiB
held per session. Storing the rows as 16-bit floats halves both, which is
why every production engine offers it, and why the interesting question is
not whether to do it but exactly which numbers get rounded.

**Where the rounding is placed.** A key is a projection output after the
key norm and RoPE; a value is a projection output. The GPU computes both in
F32 exactly as before and rounds once, when the row is written into the
cache (`nu_pack_half`: one thread per element, nearest even, bit-identical
to Zig's `@floatCast`). Nothing upstream changes, so an F16 cache and an
F32 cache hold the same rows up to that one rounding. The reading side
decides how much more precision is lost, and the two attention paths
differ:

- *Decode* reads one key row at a time in a plain loop, so the kernel
  converts each half back to F32 as it loads it. Queries, scores, the
  softmax, and the weighted sum stay F32: the cache is the only thing that
  is 16-bit.
- *Prefill* multiplies 8×8 tiles on the matrix unit, which takes both
  operands in one type. A half key tile cannot be multiplied by an F32
  query tile, and Metal has no conversion from a half tile to a float tile
  (the constructor is protected; assigning the storage vector crashes the
  compiler). So the half chunk kernel packs the queries to half as well and
  rounds its probability tile to half before multiplying it with the value
  tile, accumulating in F32. This is the same contract the ENGN-05 matmul
  tiles use for activations, and the same discipline: when the operands
  are rounded, compute the normalizer from the rounded values, so the
  weights still sum to exactly what the products used.

**How to measure it honestly.** A kernel that reads rounded data must be
compared two ways. Against the CPU reference *over the rounded values*, it
should be as exact as the F32 kernel; that isolates the kernel's own error
(1.1e-8 for the decode kernel, the same as F32). Against the reference over
the *original* values, the difference is the rounding's cost, which is a
property of the data, not of the kernel (1.2e-5 absolute, 2.1e-4 relative
RMS on Gaussian rows). The chunk kernel has a third term, the half
probability tile, whose worst case is a 2^-11 relative perturbation of
every softmax weight; it only shows when few keys are visible, because
over a long prefix the perturbations are independent and average out.

**Why the full-model tolerance had to move, and why only for F16.** The
bring-up comparison against the reference traces holds every layer output
to 2e-3 absolute. The residual stream of a 27B model has outlier channels
in the hundreds, so a 2e-4 relative change of an attention output that
feeds them shows up as 2.5e-2 absolute in the deep layers, while the logits
move by 9e-4 and the greedy token does not move at all. Loosening the
global threshold would hide regressions in the F32 path, which is the
reference the GPU is checked against; so the F16 mode gets its own
tolerance (3e-2 / 2e-4 on layer files, logits inside the bring-up numbers),
`make compare` runs both precisions, and the F32 numbers stay where they
were. The rule generalizes: a documented tolerance per numerical mode,
never one loosened number that covers all of them.

**What half cannot hold.** 65,504 is the largest finite half. Keys are
normalized and rotated, values are a linear projection; on this model both
stay far below it, but the check is the generation check and the pinned
prompts, not a per-element guard. The same assumption already underlies
the half matmul tiles; both would fail loudly (infinities, a non-finite
result error) rather than silently.

## 39. Checkpoints, not rewinds: why a hybrid model copies its state

Built in ENGN-06. A pure-attention engine can "undo" the last *k* tokens for
free: the cache is append-only, so forgetting them is setting a length.
That is how most servers implement retries and cancelled turns. This model
has 48 recurrent layers among its 64, and each holds a 48 × 128 × 128
matrix that is overwritten in place every token; after token *n* it is a
function of every token so far, and no earlier value survives. Set the
attention length back to *m* and the recurrent layers still remember *n*:
the model would attend to one history and recur over another, silently.
The spec forbids that rewind for exactly this reason.

What exists instead is a checkpoint: copy the used state out, and copy it
back later. "Used" matters. A 32K session holds 2 GiB of cache, but after
300 tokens only 300 rows of each layer's keys and values carry information,
so the snapshot is the recurrent state (150 MB, always whole) plus 64 KiB
per token. Restoring copies the same bytes back and sets the position;
rows beyond it are stale and never read, because attention always looks at
`[0, position + 1)` after writing its own row.

Two guards make the copy safe. The snapshot carries a digest of the layouts
and the capacity, so bytes from an F16 session cannot land in an F32 one
or a 4K checkpoint in an 8K session; a mismatch is a typed error that
touches nothing. And neither direction is allowed while a step is in
flight or after one failed: a failed step leaves half-updated matrices,
which are not a state anyone should keep. On the GPU the same rule is the
existing one: the CPU may touch session memory only after `commit()` has
waited for the command buffer.

The proof is the simplest possible: take the checkpoint after token one,
compute token two, restore, compute token two again, and demand the logits
be identical bit for bit on both backends. Anything less (a tolerance)
would mean some state was not captured. The chat's replay after a
cancelled turn, and the agent's cancelled tool loops, are what this
replaces.

## 40. Flash decoding: one query against a long memory

Built in KERN-08. Prefill attention (§ 34) has many queries and many keys, so
tiles of both amortize every load through the matrix unit. Decode has one
query per head and tens of thousands of keys: there is nothing to tile on
the query side, and the whole cost is streaming the cache once. The
three-pass kernels that served bring-up read it far more than once. Six
query heads share each KV head in this model, and a kernel that assigns
one threadgroup per (query head, position) fetches every key row six
times, one per head; the values pass does the same per channel. Add a
softmax that walks each head's 30,000 scores on one SIMD group, and a
token at 32K context spent about 280 of its 377 ms in attention.

The fix has three parts, each of which is a general rule for the regime.

**Read each row once for everything that needs it.** A threadgroup owns
one KV head and all the query heads that share it, so a key row is loaded
into registers once and dotted with six queries. This is the GQA
structure of the model turned into a memory-traffic argument: the cache
is 2 GB at 32K with F16 rows, and that is now the number of bytes the
kernel reads per token, not six times it.

**Keep the softmax state in registers, split the rows across the GPU.**
The online softmax from § 34 needs, per head, a running max, a running
sum, and an accumulator the width of a value row. With lane l holding
channels l, l+32, … that accumulator is eight floats per lane per head,
so the whole state for six heads fits in registers and no score buffer is
written. The visible range is cut into up to 64 slices, one threadgroup
each, so a 32K context becomes 256 threadgroups of four SIMD groups
instead of 24 serial ones; each slice produces a partial (max, sum,
accumulator) per head.

**Merge partials with the log-sum-exp identity.** Two partial softmaxes
over disjoint key sets combine exactly: rescale each by exp(m_i − M) for
the joint max M, add the sums and the accumulators, divide once at the
end. The same identity merges the four SIMD groups inside a threadgroup
(through 8 KB of threadgroup memory, three rounds) and the 64 slices in a
tiny second dispatch. An empty slice is a partial with max −∞ and sum
zero, and the identity handles it without a special case, since
exp(−∞) is zero.

What the checks say about this shape of kernel: against the F64
reference the result is within 2e-8 at 32,000 rows, closer than the old
three-pass kernel, because each product is accumulated once in F32
instead of being rounded into a stored score and read back. The
full-model comparison tightened from 1.2e-4 to 6.1e-5 for the same
reason. The remaining question, which the bench answers, is whether the
kernel reaches the bandwidth floor: 2 GB per token at this GPU's rate is
about 10 ms, against the 280 it replaced.

## 41. Comparing engines: the same tokens, separate clocks, and what "memory" means

Built in ENGN-07, the unit that turned the reference baseline into a
comparison. Three things had to be true before a nuclis number could sit
beside a llama.cpp number in one table.

**The same tokens, not the same text.** The reference harness built its
prompts by concatenating token arrays: a template prefix, the first *n*
tokens of a synthetic corpus, and a closing instruction, each tokenized
separately. That is a perfectly good way to hit an exact length, but the
result is not always the canonical tokenization of any text. The 512
prompt ends its corpus slice on a lone space token followed by the
suffix's newline; the byte-level pre-tokenizer (§ 5) groups whitespace
runs with the newline that follows them, so tokenizing the joined text
yields ` \n` as one token and 511 in total, while one more corpus token
yields 513. There is no text that tokenizes to that array. `bench` therefore
takes `--prompt-tokens`, a JSON array fed untokenized, and the acceptance
runs use the committed fixture arrays themselves. `nuclis tokenize`
exists for the other direction: it renders a text prompt exactly as
`generate` would (raw, or one turn through the pinned template), reports
the IDs and each token's byte offset in the rendered text, and reads only
the artifact's header (the vocabulary and template live there), so a
script can cut a corpus at a token boundary and count tokens without a
model run. The check it enabled is worth stating: nuclis tokenizes the
83,384-token corpus into the reference's body tokens at every cut used
(the longest, 32,620 tokens, included), which is tokenizer equivalence on
a much larger input than the pinned fixtures.

**Separate clocks.** Both sides report prefill and decode as distinct
rates, but the definitions have to line up. nuclis's decode rate is
`(generated - 1) / decode`, the steps after the first sampled token,
because the first token's latency is prefill; the reference's
`predicted_per_second` also counts intervals after the first token. Both
tokenize outside the timed region. Neither streams to a client during the
measurement. What differs is the process: the reference is a server
answering HTTP requests, nuclis is one process running the loop directly,
so wall time per request is not comparable and is not compared; only the
two rates are.

**Memory.** Weights are memory-mapped (§ 3), so the process's resident set
counts model pages the kernel has faulted in, and it grows with use rather
than being allocated. Peak resident memory over a whole `bench` process
(`/usr/bin/time -l`) is therefore an honest upper bound on what the run
touched, not the engine's allocation. The session block is the part nuclis
allocates for context (the KV cache plus the recurrent state, page-padded),
reported as `session_bytes`; at 32K with the F16 cache it is 2.15 GiB. The
headroom question the spec asks is answered by the pair: peak resident
memory beside the machine's 48 GiB, with swap counters sampled before and
after each run so a measurement that leaned on swap cannot pass silently.

The record that closes the unit is a dated JSON file under
`docs/benchmarks/` with the hardware, OS, compiler, build mode, artifact
hash, git revision, power state, the fixture arrays' hashes, every sample,
and the reference's accepted rows read from its own record, so the
comparison is reproducible from the tree plus the model file.

## 42. Judging a file from its head: the directory as a contract

A GGUF file is a directory followed by weights. The directory — magic,
version, counts, the metadata key-value pairs, then one descriptor per
tensor with its name, shape, encoding id, and offset — says everything an
engine needs to decide whether it can run the file, and none of it needs
the weights. `nuclis model inspect` (MODL-03) makes that decision over the
network before a 16 GB download, and the way it does so is a small study in
reading exactly enough.

**Read until the parser is satisfied.** The directory's length is not
stored anywhere; it is known only once the last descriptor has been read.
So the command fetches the head of the file in 8 MiB windows through the
download package's range reads and hands the bytes it has to the ordinary
GGUF parser with the *real* file size. The parser cannot tell a truncated
buffer from a short file except by running out — `EndOfStream` — so that
one error means "fetch another window" while every other error is a
verdict about the file. Because the size passed is the real one, the
bounds checks (every tensor must end inside the file) are the same the
local path runs; because the parser's own 64 MiB directory bound caps the
loop, a file that never resolves cannot make the command download
indefinitely. Qwen3.8-27B's directory is 11.0 MB (the vocabulary lives in
it), Gemma 4 12B's 15.8 MB: two windows and about four seconds each.

**Four levels of "can I run this".** The verdict separates claims the
code base used to blur. *Storable*: the encoding id is in the layout table,
so the parser can size the tensor; Q4_0 and BF16 are storable. *Executable*:
the adapter has kernels for it (`Adapter.executableEncoding`); Q4_0 is not.
*Bindable*: the adapter's binder accepts the whole directory — every
expected tensor present with its shape, no stranger, the metadata pinned
to the profile the runtime implements. *Supported*: all of that, and the
catalogue pins the Hub's digest for the file. The first two are per
tensor, so the report names the first offender (`token_embd.weight uses
Q4_0 (id 2), outside the qwen35 adapter's executable set`); the third is
per file; the fourth is a table lookup. A `gemma4` file today stops at the
first gate — no adapter — which is exactly the state the Gemma 4 12B adapter changes.

**Errors carry no payload, again.** The parser rejected unknown encodings
with a bare `UnsupportedTensorType`, which was enough for `inspect` on a
local file and useless for a verdict. Rather than widen the error set, the
parser gained an optional `Rejection` out-parameter that copies the
offending tensor's name and encoding id into a fixed buffer before the
error unwinds (the directory storage is freed on the way out, so the name
must be copied, not borrowed). `parse` is unchanged: it passes `null`.
The same pattern serves the configuration loader (§33) and the Metal
bridge; the rule stays that the error decides control flow and the
diagnostic decides the message.

**Testing without the artifact.** The adapter's tests already hydrate the
pinned model's directory inventory from a JSON fixture (every tensor's
name, shape, and encoding; no weights). The inspection test serializes
that inventory back into GGUF bytes — the vocabulary array written with
its real count and empty strings, since the binder checks the count and
never reads the strings — and serves them through a stub range reader in
256 KiB pieces. The same bytes the parser would see from the Hub, without
the Hub: supported with the pinned digest, runnable with another, not
runnable with one tensor re-encoded as Q4_0, rejected with an id the
layout table lacks, no adapter with the architecture renamed. The first
element of the adapter registry (`models.adapterFor`) arrived here, ahead
of the engine's own dispatch through it (MODL-04).

## 43. A registry built from a table: the seam before the second model

The spec's extension rule says a second architecture must cost a new
adapter, its profile, its tests, and any genuinely new mathematics, and
nothing else. Before MODL-04 the engine could not honour that: `Engine.open`
called the Qwen adapter's `bind`, held a Qwen `Binding`, and its `Model`
union named the Qwen runtime and plan; the executable named the Qwen
profile for its `--think` levels and sampling defaults in seven files. A
Gemma adapter would have meant editing all of them. MODL-04 moves that
knowledge into two tables and derives everything else from them.

**The adapter table.** An adapter now publishes a *family*: a namespace
(a Zig `type` used only for its declarations) with the names the engine
needs — `architecture`, `executableEncoding`, `Binding` and `bind`,
`Runtime` (the CPU reference), `Plan` (the Metal executor). Qwen's is six
lines at the end of `models/qwen35.zig`. `models/root.zig` lists the
families in `table`; that list is the registration. The generic
`Registry(families)` in `models/registry.zig` then builds the `Adapter`
enum *from the table* with `@Enum`, so a tag exists exactly when a family
does, and the engine builds its executor union the same way with `@Union`:
one field per family, each holding `Executor(Family)`, the cpu/metal union
the old `Model` was, now generic over the family. Dispatch is an
`inline else` switch on the tag: at load, `select` compares the
architecture string once; per token, `Model.step` is a jump on an enum.
The compile-time check `checkFamily` names a missing declaration at the
table rather than at the first use site.

Why derive the types instead of writing `enum { qwen35, gemma4 }` by hand?
Because the hand-written version is a second list that must agree with the
first, and the rule the tree is defending is precisely "one place". The
test over stub families (`alpha`, `beta`) proves the derivation: tags,
lookup, the `known` list, encoding dispatch, and `validate` all come from
the stub table with no reference to Qwen.

**The profile table.** Prompt profiles register the same way in
`profiles/root.zig`: the `Profile` enum lists modules, and `forDocument`
selects one by the SHA-256 of the artifact's chat template, never by
architecture, so a conversation is never rendered with a template the
fixtures did not verify. The conversation types (`Role`, `Message`,
`Effort`, `Limits`) moved to the root and are shared: the executable's
configuration keeps one `think` vocabulary, and a profile whose template
only switches thinking on or off maps `off` to off and the rest to on. That
is a decision recorded here, not a fact from a model card.

**What the executable sees.** Nothing under `src/` imports an adapter or a
profile module any more. `nuclis validate` renders a shared
`models.Summary` whose layer composition is a list of named kinds (`16
full_attention, 48 delta_net`) instead of fields named after one
architecture's layers (its JSON is schema 2 for that reason); `model
inspect`'s verdict, the CLI's diagnostic for an architecture without an
adapter (`known: qwen35`), and the trace observer all go through the
registries. The cost of the second adapter is now what the spec asked for:
its files, one line in `table`, one tag in `Profile`.

**What this does not prove yet.** The registry is exercised by one real
family and two stubs. MODL-05–MODL-07 bring the second real family through it, and
the spec still wants a small synthetic dense-attention model in the default
tests before the seam is called stable. Bench after MODL-04: decode 10.66,
prefill 40.0 tok/s on the standard workload, unchanged.

## 44. The second architecture on the CPU: what changed and what did not

Gemma 4 12B is the first model to go through the seam after the registry
(§ 43), and the interesting part is how little of the numerical layer it
needed. The CPU reference reuses `rmsNorm`, the quantized row decoders and
`matvec`, `attention.apply`, and `rope.apply` unchanged. Two things were
genuinely new mathematics and got their own fixtures: the tanh GELU
(`0.5·x·(1 + tanh(√(2/π)·x·(1 + 0.044715·x²)))`, evaluated in F64 so the
F32 result is the rounded formula) and RoPE frequency *factors*, a divisor
per rotated pair; the checkpoint stores 64 ones and 192 values of 1e30, so
three quarters of a global head's pairs get an angle that underflows to
nothing and copy through. That is how "rotate only the first 25 % of the
dimensions" is expressed in the file without a second code path.

**Sliding window without a mask.** Forty of the 48 layers attend only to
the last 1024 positions. The Qwen attention reference takes borrowed key
and value arrays plus a visible-row count, no mask argument, and it did
not need one: the visible rows of a sliding layer are a contiguous suffix
of the cache, so the runtime slices `keys.floats(first, visible)` with
`first = position + 1 − 1024` and calls the same function. The reference
implementation masks; the contract, "attend to rows `first..position`",
is the same either way, and the trace comparison proves it.

**Reading the reference, not copying it.** Every fact in
[gemma4.md](reference/gemma4.md) names where it came from: the file
(shapes, encodings, the 1e30 factors, the raw norm weights), the upstream
configuration, or the pinned llama.cpp source (the order of the value
norm relative to the key norm, the scale of 1.0 on attention scores, the
NeoX pairing, the window inequality). The implementation was written from
that record; then the reference's per-layer outputs on `<bos>Hello,` were
captured and compared. The first run agreed to a relative RMS of 8e-6 on
every layer and reproduced the greedy token. When a port disagrees, the
per-layer trace tells you which operation, and the record tells you which
fact to re-read.

**Two tokenizers, one merge loop.** Qwen's vocabulary is byte-level BPE:
every input byte becomes a symbol of a 256-letter alphabet and merges
join symbols. Gemma's is SentencePiece-style BPE: symbols are Unicode
code points, spaces are first rewritten to `▁` (U+2581), the only
pre-splitting is around runs of newlines, and a symbol the vocabulary
lacks falls back to `<0xNN>` byte tokens. The rank-ordered merge scan is
identical, so it became one function (`bpe.mergeSpans`) with two callers
that differ only in how they make symbols and resolve leftovers. Twelve
strings against the reference tokenizer agreed exactly, including the
chat markers written inside text, an emoji, CJK, and a four-newline run.

## 45. The second architecture on the GPU: parameters, instantiations, and one clamp

§ 44 showed how little of the CPU layer Gemma 4 needed. The Metal plan
(`models/gemma4_metal.zig`, MODL-06) is the same story with a twist: a GPU
kernel is written for a *shape*, and the shapes a second model brings are
where the reuse is tested. The Qwen kernels were written for 24 query
heads over 4 KV heads, 256 channels per head, one attention scale, and
no window. Gemma has two layer kinds: forty layers of 16 heads over 8 KV
heads at 256 channels that see only the last 1,024 positions, and eight
layers of 16 heads over *one* KV head at 512 channels that see everything.

**What became a parameter.** The sliding window on prefill chunks is a
`window` field on the chunk attention kernel: each row hides keys below
`position − window + 1`, the key loop starts at the first tile a SIMD
group can see, and one guard keeps the running max at −∞ for a row whose
keys are all hidden in a tile (the old kernel assumed key 0 was visible
to everyone). On decode the window is not a kernel change at all: the
plan slices the cache to the last 1,024 rows, exactly as the CPU
reference does. The 512-wide value output exceeded the chunk kernel's
32 accumulator matrices per SIMD group, so the grid gained a third axis:
one threadgroup per 256 value columns, each recomputing the scores it
needs. Redundant work on eight layers, no new kernel.

**What became an instantiation.** The flash-decoding kernel keeps each
query head's running max, sum, and accumulator in registers: 8 heads × 8
channels per lane was its budget. 16 heads × 16 channels would be four
times that and spill. The kernel became a template over both numbers,
instantiated twice: `8 × 8` (the Qwen shape, unchanged) and `4 × 16`
(512 channels, four heads at a time), the same 128 floats per lane either
way, with a KV head's 16 query heads covered by four threadgroups per
split that each read the slice once. Register budgets are why GPU code
has instantiations where CPU code has a loop bound.

**What was genuinely new.** Four scalar epilogues (`x *= s` for the
embedding scale, `x = (x + y)·s` for the per-layer output scale,
`cap·tanh(x/cap)` for the logits) and a GELU variant of the fused
gate-pair projection. Each is a dozen lines and a fixture against
`cpu.*`. And one lesson that was not in any reference: the first Gemma
step on the GPU produced NaN in nine of 15,360 gate values. Metal's
`tanh` is computed through `exp` and overflows to NaN past about ±44,
where the CPU's F64 `tanh` quietly saturates. Since F32 tanh is exactly
±1 from ±20 on, clamping the argument there changes no finite result and
removes the failure. A per-layer probe (commit after each operation, scan
for non-finite values) found it in one run; the trace comparison alone
would only have said "layer 0 is wrong".

**Reading a tolerance instead of blessing it.** With the F32 cache the
plan matched the pinned llama.cpp traces to 3.4e-4 at the first try
after the clamp. With the F16 cache it deviated by 0.73 on the logits,
thirty times Qwen's F16 deviation. Before writing that number down as
"Gemma's tolerance", two experiments separated *kernel error* from
*model sensitivity*: the same half kernels on the Gemma geometry against
the CPU over the rounded operands (2e-4: the kernels are right), and the
CPU reference itself with its keys rounded to F16 before the cache write
(0.75: the same deviation without any GPU in the loop). Gemma's attention
scores are unscaled, so a 2⁻¹¹ rounding of a key moves a score by
hundredths and a softmax weight by percents; on a short prompt with a
BOS sink that compounds through 48 layers. The tolerance is recorded
with that explanation, and `--kv f32` exists for numerical work.

**The check that found a backend bug.** Generalizing `generation-check`
to both families added one more 70-token pass per family — and Qwen's
run failed with `InvalidShape` where it had passed for weeks. The Metal
backend caches wrapped buffers by address and never forgot one; a plan's
freed session memory came back from the allocator at the same address
with a different length, and the wrap refused it. The fix (`unwrap` on
plan `deinit`) is two lines; the lesson is that a check whose order of
allocations changes is a new check.

## 46. A prompt profile is a contract: template, stop set, and reasoning markers

The MODL-07 implementation described here supplied textual markers to the chat;
token-aware profile decoding replaces that last step
([agent-concepts § 2](reference/agent-concepts.md#2-why-a-streaming-parser-needs-token-boundaries)).

§ 44 and § 45 took Gemma 4 through the numerical seam. The last piece
(MODL-07) is the conversational one: the chat template, and everything that
quietly depends on it. The spec calls this the *prompt profile* and keeps
it apart from the architecture for a reason that the second model made
concrete: two checkpoints of the same architecture can ship different
templates, and one template can front several architectures (the K-quant
and QAT Gemma files share theirs). So a profile is selected by the SHA-256
of the template text, never by `general.architecture`.

**Reading a template as a contract, not running it.** A Jinja template is
a small program; the reference runs it with an interpreter. nuclis does
not: the profile is a Zig function that renders the *subset* of
conversations the playground produces (system, user, assistant, a thinking
switch), and the evidence that it renders them correctly is a fixture the
reference server produced from the artifact's own template. Reading the
19 KB Gemma template for that subset reduces it to a dozen clauses (the
module comment of `profiles/gemma4.zig` lists them), and several were not
what the model card says. The reference strips the template's leading
`<bos>` because its tokenizer adds one; nuclis's encoder never adds BOS
(a decision from MODL-05), so the profile writes `<bos>` as text and the test
prepends it to the fixture. The reference sends `developer` messages as
`system`. Two assistant messages in a row continue one `model` turn.
Reasoning is rendered only after the last user message, so in a
conversation that ends with the user it is never rendered at all — the
opposite of Qwen3.8, whose fixture keeps every `<think>` block. None of
this is guesswork once the fixture exists; all of it would have been
guesswork without it.

**Thinking as a switch versus a level.** The shared `Effort` enum has four
values because Qwen3.8 has four. Gemma's template has a boolean. The
registry's rule (§ 43) is that a profile *collapses* what it cannot
express: `off` is off, everything else is on, and a test asserts that
`low` and `xhigh` render exactly like `medium`. The alternative — a
per-profile effort type — would have pushed the difference into every
caller (the config schema, the CLI flag, the chat's Ctrl-T) for no
benefit the user can see.

**The two assumptions the second model found.** The engine's stop test
was `token == eos or token == bos`, with a comment explaining that both
`<|im_end|>` and `<|endoftext|>` end a Qwen turn. On Gemma, BOS is
`<bos>` (2): the model would have stopped the moment it emitted a BOS,
and would *not* have stopped on `<eos>` (1) at all, since the K-quant
file's `eos_token_id` is `<turn|>`. And the chat split every turn on the
literal `</think>`. Both were Qwen facts living outside the profile —
exactly what the spec's extension rule forbids, and invisible until a
second template arrived. The fix is the profile owning both: a
`stop_tokens` list of *texts* the engine resolves to ids in the loaded
vocabulary (refusing to load when one is missing, since a template whose
end marker the tokenizer does not know is a broken pairing), and a
`reasoning` pair of open/close markers the chat's `parts` splits on. The
open marker matters for Gemma because the model emits it
(`<|channel>thought\n` is generated text), whereas Qwen's prompt supplies
`<think>` and the model only closes it; `parts` strips the opener when
present and treats an unclosed channel as all thinking, so one function
serves both.

**Whose profile is it?** The configuration resolves a profile before any
file is open, from the catalogue entry's name, so that `config show` can
print effective sampling values instantly. A bare path or a registry entry
the catalogue does not know gets the first profile. That is a guess, and
Qwen's instruct defaults (presence penalty 1.5, which also forces the
slow full-logit sampling path) are a bad guess for Gemma. The rule is
therefore: the configuration's profile is for *display*; the engine's
profile, selected from the opened file's template, is what `generate` and
`chat` sample with, corrected once the model is open. One line each, and
the display stays honest about being a guess.

**What the fixture capture taught about the reference.** The server
reports the template in `/props` without the file's trailing newline, so
the digest of the text it serves differs from the file's by one byte
while rendering identically (the last tag strips trailing whitespace).
The capture script accepts both forms and the profile pins the file's
digest, because the file is what `forDocument` hashes. Small, but the
kind of thing that costs an hour when a check fails for the wrong reason.

Read: `inference/src/profiles/gemma4.zig` (the module comment is the
contract), `inference/src/profiles/root.zig` (`Reasoning`, `stopTokens`),
`inference/src/engine.zig` (`stop_ids`), `src/tui/view.zig` (`parts`),
[reference/prompt-profile.md](reference/prompt-profile.md), and
[reference/new-model-guide.md](reference/new-model-guide.md), the
starter guide written from MODL-05–MODL-07.

## 47. A new storage encoding, end to end: Q4_0

MODL-08 closed the second architecture on the file it was decided for: Google's
quantization-aware-trained Gemma 4 12B, whose 329 weight matrices are all
Q4_0. Until this sub-unit the parser could *size* a Q4_0 tensor and the
decoder refused to *read* one, which is the honest state for an encoding
without kernels. Adding one is a good tour of every layer a number passes
through, because an encoding is not one function; it is the same contract
restated five times, each pinned to the layer below.

**What Q4_0 is, and what it is not.** A block of 32 values: an F16 scale
`d`, then sixteen bytes; the low nibble of byte `j` is value `j`, the high
nibble value `16 + j`, and the value is `d · (q − 8)`. That is IQ4_NL's
block byte for byte (§ 6 met the nibble order there) with the lookup table
replaced by subtraction, and it is *not* a small Q4_K: no super-block, no
six-bit group scales, no minimum. The QAT checkpoint uses it because that
is the arithmetic its training simulated, and the loader's rule that it
never requantizes is what makes the choice of file meaningful at all.

**Layer 1, the storage fact.** `tensor/encoding.zig` already said "18 bytes
per 32 elements" for id 2; the parser had needed that to lay out the file.
Nothing changed there, which is the point of keeping layout apart from
arithmetic.

**Layer 2, the CPU truth.** `quant.row` gained a `2 =>` arm of five lines.
Its evidence is not the test I wrote by hand (though there is one, with
the same nibble bytes as the IQ4_NL test so the two decoders pin each
other's ordering); it is the fixture regenerated by
`scripts/quant-fixtures.py`, eight Q4_0 blocks decoded by the pinned
llama.cpp C function through `ctypes`. Regenerating showed a small
hazard: the script rewrites every fixture and the IQ3_S grid file, and
its header text for the grid had drifted from the committed one, so a
regeneration that touched nothing numerically would still have produced
a diff. The script's header now matches the file; a generator must be
idempotent on what it does not intend to change.

**Layer 3, the generic GPU decoder.** `dequant.metal` got
`nu_dequant_q4_0`, written as `d * float(int(nibble) - 8)`: the CPU's
operations in the CPU's order, because `test-metal` checks the embedding
row bit for bit against `quant.row`. The generic matvec, the embedding
kernel, and the generic matmul tile all go through it, so at this point
the QAT file would already *run* on Metal, slowly.

**Layer 4, the specialized kernels, by reuse.** The K-quant matvecs
walk 256 values per lane octet, one lane per 32-value slice. A Q4_0
block is exactly one such slice, so lane `l` takes blocks `l`, `l + 32`,
… of the row the way IQ4_XS's lanes take a group (the loop walks blocks
rather than 256-value strides, so a row of any block count serves). Two things carried over
from neighbours: the loads are `packed_ushort4` because an 18-byte block
is only 2-byte aligned (Q6_K's problem; a `uint` load at an odd-word
address is undefined on the GPU, which shows up as garbage in a few
rows, not as a fault), and the bias folds into the input sum as Q6_K's
32 does, `Σ d·(q−8)·x = d·(Σq·x − 8·Σx)`, which is exact for a one-hot
input and so keeps the fixture columns exact. One rule is new:
`specializedMatvec` refuses a Q4_0 row whose stride is not a multiple of
144 bytes, because the kernel's `blocks` parameter counts 256-value
strides; a row of 42 blocks would silently drop the tail. The tile for
prefill needed one generalization, "segments per block" (two, not
sixteen), and its decode is the generic expression in the same order,
so the F32 view of the tile is bit-identical to the generic tile and
only the half rounding remains, which `test-metal` bounds.

**Layer 5, the adapter's claim.** `gemma4.executableEncoding` lists id 2;
`qwen35`'s does not. The executable set is not "what the kernels can do"
but "what this adapter will bind", a claim the adapter makes about the
files it has been validated on. Qwen's Q4_0 draft head (the MTP unit) will add it
there when that unit runs on it.

**What the file then said.** Against its own traces from the reference
harness the QAT file is tighter than the K-quant file on every path (CPU
4.4e-5, Metal F32 7.7e-5, F16 cache 1.3e-2 against 0.73 on the K-quant),
so the F16 key-rounding sensitivity of § 45 is a property of a checkpoint
and a prompt, not of Gemma. The generation check told the opposite story
for prefill: the chunked run deviates from the stepped one by 0.44 in the
logits where the K-quant file gave 0.086, and the same check through the
generic F32 tiles gives 7.8e-4, so the schedule is right and the
half-operand rounding of the prefill tiles (§ 24's ceiling) is what this
checkpoint amplifies. The family's bound moved to cover both files, with
both numbers written next to it. A bound is only as good as the sentence
that says what it covers.

**The extension rule, checked on the diff.** Outside the adapter, its
kernels, the fixtures, the catalogue, and the docs, the change touched
one line in `quant.row`'s validation list and the enumerations of
`metal-check`; no edit to the GGUF parser, the sampling policy, the
generation loop, or the Metal object lifecycle. That is what the spec
asked the second architecture to prove, and Q4_0 was the last piece of
the proof.

Read: `inference/src/quant/decode.zig` (the `2 =>` arm and its test),
`inference/src/backends/metal/dequant.metal` (`nu_dequant_q4_0`),
`inference/src/backends/metal/kernels.metal` (`nu_matvec_q4_0_body`,
`nu_tile_q4_0`, `nu_tile_segment`), `Backend.specializedMatvec` in
`inference/src/backends/metal/root.zig`,
[reference/gemma4.md § Q4_0 path](reference/gemma4.md#q4_0-path-and-the-qat-file-modl-08-2026-09-12),
and [reference/quantization.md](reference/quantization.md).

## 48. Mixture of experts: sparsity is a gift at decode and a bill at prefill

KERN-09 built the kernels for the first sparse model in the tree, Gemma 4
26B-A4B, before its adapter exists. The name says the whole idea: 26
billion parameters stored, about 4 billion *active* per token. Every
layer's feed-forward block is replaced by 128 small ones (the "experts",
each a gated-GELU FFN of width 704 instead of one of width 2,112 beside
them) and a router that picks 8 per token. § 6 said decode is bound by
the bytes read per token; a token here reads 8 of 128 expert matrices, so
the expert bytes per token are one sixteenth of what a dense model of the
same size would read. That arithmetic is why this family comes before the
dense 30B in the plan: 2.1 GB per token at Q4_0 against 16 GB for the
dense 27B, on the same memory bus.

**Routing is a softmax you never finish.** The router is a small matrix
(2,816 × 128) whose output is a logit per expert. The reference takes the
softmax, keeps the 8 largest probabilities, and renormalizes them to sum
one. Two details carry over from § 30's sampler. Selection compares the
*logits*, not the probabilities: softmax is monotone, so the order is the
same, but `exp` rounds, and two logits that are equal (ties happen; the
fixture has three at the top) or a hair apart could swap order in a
kernel that rounds differently from the reference — comparing the raw
values with a lowest-index tie-break makes the indices exact, bit for
bit, and only the weights carry rounding. And the renormalizing sum is
clamped below at the smallest F16 normal (`weight_sum_floor`, about
6.1e-5): a token whose selected probabilities all underflow would divide
by zero, and the reference's floor is part of the contract, not a
kernel's choice. The GPU kernel is one 256-thread group per token, one
logit per thread, and `k` rounds of "best untaken" — the shuffle
reduction § 30 used for top-k sampling, on 128 values instead of 248,320.

**A 3-D tensor is a row of matrices.** GGUF stores the experts of one
projection as a single tensor `[experts][rows][columns]`, the experts
contiguous. `cpu.ExpertMatrix.expert(e)` is a slice: `bytes[e ·
per_expert ..]` viewed as the dense `Matrix` of § 11. That is the whole
reason the decode kernels were cheap. `nu_matvec_experts` is
`nu_segment_sums` — the same lane arithmetic as the merged projections of
§ 28, specialized or generic by encoding — with the weight pointer
offset by `expert · rows · stride` before the loop; a selected expert
costs exactly the bytes of a dense matrix of its size, and the other 120
are never touched. `test-metal` proves the "never" by writing NaN into
every scale of the unselected experts and checking the outputs are
bit-identical. One rule is new: the expert index comes from a buffer the
GPU wrote a moment earlier, so the kernel clamps it to the tensor. A
corrupt routing buffer then yields a wrong answer, which the comparison
catches, instead of a read past the tensor, which nothing would.

**Decode: four dispatches and one geometry lesson.** Route, gathered
gate-up (all 8 slots share the token's input), the gated GELU over the 8
hidden rows, gathered down (each slot its own hidden row), then a
weighted sum of the 8 projections with the routing weights and the
per-expert down scale the checkpoint carries. Measured on the 26B-A4B
shape, the gathered gate-up runs at 215 GB/s of selected bytes — the
dense matvec's rate over the same byte count, so gathering costs
nothing. The down projection runs at 154. Its row is 704 values, 22 Q4_0
blocks, and § 47's kernel gives one block per lane: 10 of 32 lanes idle
in its single pass. Nothing about sparsity; the row is short. The lane
mapping for short rows is a follow-up because at 8 experts × 30 layers
the whole chain reads 0.8 GB per token, about 5 ms, and the profile of
a real token decides whether that 25 % matters.

**Prefill: the same problem as § 31, harder.** A dense prefill reads
each weight once per tile of 32 or 64 tokens; that is the entire gain of
chunking. In a sparse layer each token wants its own 8 experts, so "the
tokens that use expert 17" are scattered through the chunk, and a tile
of consecutive tokens would need 8 × 32 different matrices. The rows
have to be regrouped by expert first — a sort by key — and it has to
happen on the GPU, because a host round trip per layer per chunk (30 of
them per 256 tokens) would cost more than the matmul. `nu_expert_lists`
is that sort in one threadgroup: a histogram over the experts with
threadgroup atomics, an exclusive prefix sum, a scatter of each slot row
(token, slot) into its expert's segment. It also writes the *tile list*:
one entry per 32 rows of each expert. Here is the constraint a CPU
programmer does not have: the matmul's grid must be sized when it is
*recorded*, before the counts exist. So the host dispatches a bound —
Σ ceil(count / 32) ≤ n / 32 + experts, 192 tiles for 2,048 rows over 128
experts — and tiles past the real count exit at their first instruction.
The bound is the price of never reading the counts back. Atomics make
the order of rows within an expert nondeterministic; that is harmless
here because each row's result is computed on its own (a reduction
across rows would not be, which is why the router avoids atomics).

**The gathered tile is the dense tile with two indirections.** The
prefill kernel is § 37's `nu_matmul_t` body with a `GATHER` flag: the
weight base is the tile's expert, the 32 activation rows are read
through the row list (`row / k` for the gate-up projection, since a
token's 8 slots share its input; the row itself for the down
projection), rows past the tile's count stage zeros, and the result goes
through threadgroup memory — the 64 × 64 half tile is exactly 32 × 64
floats, so the buffer is reused — to be scattered to each row's slot in
the output. Nothing is written past the count, which is why the scratch
holds exactly `k · chunk` rows with no padding. The dense instantiations
are the same body with the flag off, so the refactor could not change
them silently: `test-metal` runs the dense tiles against the CPU before
it runs the gathered ones.

**What the measurement said, and what it means for sparse models.** The
gathered gate-up tiles run at 4.8 TFLOP/s executed, against 5.0 for the
dense 64 × 64 Q4_0 tile: per executed multiply, the gathered tile is the
dense tile. But at 256 tokens the 2,048 slot rows spread over 128
experts give 16 rows per expert, and a 32-row tile is half zeros; the
useful rate is half the executed one. The chunk fills the tiles: 50 % at
256 tokens, 69 % at 512, 81 % at 1,024, and the per-token cost falls
accordingly (10.5 → 27.0 ms per layer-chunk for 4× the tokens) while the
GB/s of tile bytes stays at 41. Read that beside § 31 and § 37. At
decode, sparsity is pure gain: fewer bytes, same rate. At prefill, the
tile's economics depend on how many *rows share a matrix*, and sparsity
divides that number by `experts / k` — sixteen here. A sparse model's
prefill is compute-bound at a lower fill than a dense one's, and the
chunk size becomes a lever it was not before (a 1,024-token chunk needs
92 MB of down-projection scratch, the memory/latency trade the adapter's
Metal plan makes). The alternative — looping the decode kernels over the
chunk — would read 6.8 GB per layer at 256 tokens; the tiles are four
times faster there and six at 1,024, which is the gain that justifies
the sort.

Read: `inference/src/backends/cpu/experts.zig` (`route`, `Ffn`, `ffn`),
`inference/src/backends/metal/kernels.metal` (`nu_route`,
`nu_matvec_experts`, `nu_expert_lists`, `nu_matmul_body` and
`nu_matmul_experts_t`), the encoders `route`, `matvecExperts`,
`expertLists`, and `matmulExperts` in
`inference/src/backends/metal/root.zig`, `checkExperts` and
`expertsBench` in `inference/metal-check.zig`,
[reference/cpu-reference.md § Mixture of experts](reference/cpu-reference.md#mixture-of-experts-routing-and-the-gathered-ffn),
and [reference/metal-backend.md § Gathered expert kernels](reference/metal-backend.md#gathered-expert-kernels-kern-09).

## 49. Ternary weights and a rotated basis: what Bonsai 2 27B stores

Bonsai 2 27B is Qwen3.8-27B again — the same 64 layers, the same DeltaNet
and attention mixers, the same tokenizer, the same adapter binding the
same 851 text tensors — at 7.2 GB instead of 16.5. MODL-16's first session
put its two weight encodings through the CPU decoder, pinned its numerical
oracle, and read its one new idea into the binding without yet executing
it. The idea is that the *cheapest* weights are only usable after a change
of basis, and this section is about why.

**Ternary is a quantization, not a compression.** A ternary weight is one
of three values, `{−1, 0, +1}`, times a scale shared by a group of 128:
`w = d · t`. Information-theoretically a trit is log₂ 3 ≈ 1.58 bits, so a
model "at 1.58 bits" is one whose matrices were *trained or re-trained* to
survive that rounding; Prism ML calls the file `folded` and the process
distillation. Nothing in nuclis decides whether that worked. As with the
QAT file of § 47, the engine's job is to reproduce the arithmetic the file
implies, bit for bit against the reference, and leave quality to the
benchmarks. The two files differ only in how the trits are packed:

- **PQ2_0** spends a whole two-bit slot per trit (`q − 1`, so the fourth
  code, +2, exists but is never stored): 32 bytes plus the F16 scale per
  128 weights, 2.125 bits per weight. Four codes per byte, low bits first —
  the nibble path of § 47 with a smaller nibble, which is why it is the
  bring-up file.
- **PTQ1_0** packs five trits per byte in base 3, because 3⁵ = 243 ≤ 256:
  24 bytes hold 120 trits, two more bytes hold the last 8, 28 bytes per
  128, 1.75 bits per weight — 17 % fewer bytes to read than PQ2_0. The
  price is that extracting a digit is no longer a shift and a mask. The
  format scales the packed number by 256/243 so that the *leading* digit
  is what `(byte · 3) >> 8` returns, and multiplying the byte by 3 (mod
  256) discards that digit and promotes the next: five multiplies, five
  shifts, no division. It is mainline llama.cpp's TQ1_0 trick at group
  128 instead of 256. The values also come out digit-major — all first
  digits of a run of bytes, then all second digits — so element `j` of a
  block is *not* in byte `j / 5`. Our decoder reproduces the order and the
  fixed-point arithmetic exactly; the fixture from the fork's own
  decoder feeds it arbitrary bytes on purpose, so non-canonical codes
  (which a valid file never contains) match too. Matching on those is
  what proves the arithmetic rather than the packing.

Why one scale per 128 rather than per 256 or per row? A ternary group has
only three levels, so the scale is the entire dynamic range; a smaller
group tracks local magnitude better at the cost of two bytes per 128
weights. The fork's comment explains PTQ1_0's existence with exactly this:
a 256-wide scale "has to discard one of the two group scales it
straddles", so a group-128 checkpoint cannot be stored losslessly in
TQ1_0.

**Why a rotation.** Rounding to three levels is brutal on a matrix whose
columns have a few large entries and many small ones: the scale follows
the outliers, and everything else rounds to zero. Activations have the
same shape — a handful of channels carry most of the energy. The remedy
in Bonsai (and in QuaRot, SpinQuant, and their relatives) is to multiply
by an **orthogonal** matrix before quantizing. For a projection `W · x`
and any orthogonal `R`, `W · x = (W Rᵀ) · (R x)`: rotate the weight's
input axis, rotate the activation the same way, and the product is
unchanged. If `R` mixes every coordinate into every other, the outliers
are spread thin, both `W Rᵀ` and `R x` look Gaussian, and ternary
rounding of `W Rᵀ` loses far less. The converter did the weight half once
(`W' = W Rᵀ`, then quantized), which is what "folded" means; the engine
must do the activation half on every token.

**Why this rotation.** A dense random orthogonal matrix on a 17,408-wide
input would cost more than the projection it protects. Bonsai's `R` is
`(1/√1024) · H₁₀₂₄ · S`: `S` a diagonal of ±1 signs, `H` the Sylvester
Walsh-Hadamard matrix, `H[r][c] = (−1)^popcount(r & c)`, applied
independently to each block of 1024 consecutive elements (5120, 6144, and
17408 are all multiples of 1024; the power of two is Sylvester's
requirement). The Hadamard matrix has two gifts: it needs no storage (a
bit-parity rule), and `H · v` costs 10 butterfly stages of 512 additions
each — the fast Walsh-Hadamard transform — instead of a million multiplies.
The sign flip is what makes it a *random* rotation rather than a fixed
one: `H` alone is a known matrix, and a known matrix can be defeated by an
adversarial or merely structured input (a constant activation is an
eigenvector of `H`); a random ±1 per coordinate first, and every input
looks generic to `H`. The signs are three vectors in the header, one per
input width, 28,672 numbers in all. Because `H` is symmetric and
orthonormal, `H⁻¹ = H`, which is why the embedding table can be stored in
the rotated basis and undone after lookup by the same transform in the
opposite order: `h = S · (H · z)`.

**What it costs.** Count the distinct activations that meet a rotated
weight per layer: the residual before the mixer (shared by `attn_qkv` and
`attn_gate`, or by `q`, `k`, `v`), the mixer output before `ssm_out` or
`attn_output`, the residual before the FFN (shared by `gate` and `up`),
and the FFN hidden before `down` — four transforms of 5120, 6144, 5120,
and 17,408 elements. The fork memoizes exactly this (one transform per
activation, however many folded weights read it). Over 64 layers plus the
output head that is about 2.2 million elements × 10 stages per token:
tens of millions of additions against the 27 billion multiply-adds of the
matvecs, negligible as arithmetic. Prism nevertheless calls it "one of
the larger non-matmul costs of a decode step" on Metal, and § 24 and § 28
taught why: at batch 1 the cost of a small kernel is its launch and its
dependency, not its FLOPs, and this is 258 more small dependent kernels
per token. Whether the sign flip and the transform fuse into the matvec's
input load, as Prism fuses the sign flip, is what KERN-10 measures.

**The bytes are the point.** Decode is memory-bound (§ 6): at 2.125 bits
per weight the language model is 7.2 GB against the 16.5 GB of the
`Q4_K_M` file, so the same M4 Pro that reads Qwen3.8's weights at 10
tokens/s should read these at two to three times that. The fork's smoke
run on our machine gave 17 tokens/s on a 16-token completion, before any
of our own work; the transform is the tax on that gain.

**Two things the binding had to learn.** First, this file has 64 blocks
and no auxiliary prediction layer: the Qwen release's 65th block (the
draft head of § 43's registry summary) was not re-encoded, so the adapter
now accepts both declarations and reports which it found. Second, the
`ssm_alpha` and `ssm_beta` projections of every DeltaNet layer are *not*
in the rotated list and are stored as BF16, not ternary: they are tiny
(5120 × 48) and feed the gating scalars of § 19, where a rounding error
compounds across the recurrence. The converter left them in the original
basis at high precision; our schedule must feed them the untransformed
activation while every neighbour takes the rotated one. A rotation
contract is a list of *which* inputs rotate, and the adapter pins the
list rather than trusting the file's: 401 names, one slot each, the
embedding inverse and nothing else.

**And one convention.** The value part of DeltaNet's output enters
`ssm_out` as 48 heads of 128, and the fork records that the fold was
computed with those heads in a *grouped* order (the three value heads of
a key group adjacent) while its mixer produces them *tiled* (the sixteen
groups adjacent), so it permutes the activation before the transform. The
two orders are the same numbers in a different sequence, and a
Hadamard transform over blocks of 1024 is not invariant to that
sequence, so the permutation is part of the contract. Whether nuclis's
mixer already produces the grouped order or the tiled one is a fact of
`qwen35_runtime.zig` that session 2 reads before it applies anything.

**Where it stands.** `quant.row` decodes ids 142, 143, and 30 against the
fork's fixture; the parser retains the rotation arrays; the adapter
validates the contract and hands a `Rotation` to the runtimes; the fork's
traces for `Hello,` are committed (greedy token ` I`, the same as
Qwen3.8's). The CPU runtime applies the transform — `cpu.hadamard` is
forty lines of butterflies in F64 — and matched the fork's traces on the
first run at max abs 2.4e-4, with the same top three logits to three
decimals. That first-run match is worth a sentence: the encodings, the
sign vectors, the block size, the four activations per layer, the
embedding inverse, and the value-head regathering were each read from a
contract and pinned before any number was compared, so the comparison
had one thing to say and said it. The Metal plan still refuses a rotated
binding with a typed error, because a model in the wrong basis produces
fluent nonsense, not a crash; the ternary kernels and the transform
kernel are KERN-10, the plan is MODL-17.

Read: `inference/src/quant/decode.zig` (the `142 =>` arm, `ptq1Block`,
`trit`), `inference/src/models/qwen35.zig` (`Rotation`,
`validateRotation`, `rotatedSlot`), `gguf.retained_key_prefixes` in
`inference/src/formats/gguf.zig`,
`inference/src/backends/cpu/hadamard.zig`, `rotate` and `rotateGrouped`
in `inference/src/models/qwen35_runtime.zig`, and
[reference/bonsai.md](reference/bonsai.md).
