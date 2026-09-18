# CPU numerical references

The inference library exposes `cpu.Matrix` and `cpu.matvec(matrix, input,
output, scratch)` from [the CPU module](../../inference/src/backends/cpu/root.zig).
It also exposes `rmsNorm`, `l2Norm`, `softmax`, `sigmoid`, `silu`, `softplus`, and
`rope.apply`, `attention.apply`, `recurrent.convolution`/`recurrent.delta`, and
`experts.route`/`experts.ffn` over `cpu.ExpertMatrix`. These
references do not load a model, run layers, or provide an optimized CPU inference
backend. Model composition and session/cache orchestration remain pending.

## Matrix-vector contract

For a matrix with `rows` output features and `columns` input features:

```text
output[r] = sum over c of weights[r,c] * input[c]
```

`Matrix` is a borrowed descriptor containing an encoding ID, dimensions, and
contiguous encoded bytes. Each row must hold complete quantization blocks,
without padding. GGUF dimension 0 corresponds to `columns`, and dimension 1
to `rows`. Constructing a descriptor does not validate it; each call does.

Input must have exactly `columns` F32 values and output exactly `rows` values.
Both dimensions must be nonzero. Scratch needs at least `columns` F32 values;
only that prefix is modified. Output and scratch must not overlap each other
or either input buffer. The function retains no slices, performs no I/O, and
allocates nothing. The caller can reuse one row of scratch across a large matrix.

Validation rejects inconsistent dimensions, insufficient scratch, unsupported
encodings, partial rows, and mismatched byte counts before modifying writable
buffers. It divides the total byte length by the row count after checking exact
divisibility, avoiding potentially overflowing products of dimensions. The
quantization module's `validateRow` centralizes encoding support and block-size
rules and is also used by `quant.row` itself.

Each row is decoded to F32 using the [row decoders](quantization.md). The
operation widens both weights and inputs to F64 before multiplying, sums from
the first column to the last, then converts the result to F32. This reduces
reference accumulation error; it is not a promise of bit-identical results from
future parallel GPU reductions. Per-operation tolerances will be established
with GPU implementations. NaNs and infinities propagate; a finite F64 sum that
exceeds F32 range can become infinity in the output.

## Normalization, probabilities, and activations

[vector.zig](../../inference/src/backends/cpu/vector.zig) uses F32 inputs and
outputs with F64 intermediate arithmetic. It performs no allocation or I/O.
The vector functions require nonempty, equally sized input/output slices. Exact
in-place use is supported; partial overlap is forbidden. They validate all
expected error conditions before changing output.

| Function | Equation | Domain and failure behavior |
| --- | --- | --- |
| `rmsNorm(input, output, epsilon)` | `x / sqrt(mean(x*x) + epsilon)` | Finite input; finite positive epsilon; typed error otherwise |
| `l2Norm(input, output, epsilon)` | `x / max(sqrt(sum(x*x)), epsilon)` | Finite input; finite positive epsilon; zero vectors stay zero |
| `softmax(input, output)` | `exp(x-max(x)) / sum(exp(x-max(x)))` | Negative infinity masks an entry; NaN/+infinity rejected; all-masked returns `EmptySupport` |
| `sigmoid(x)` | `1 / (1 + exp(-x))` | Scalar; -/+infinity map to 0/1; NaN propagates |
| `silu(x)` | `x * sigmoid(x)` | Scalar; -infinity maps to negative zero; +infinity preserved; NaN propagates |
| `softplus(x)` | `log(1 + exp(x))` | Scalar; -infinity maps to zero; +infinity preserved; NaN propagates |

L2 normalization uses the sum of squares and clamps the norm after taking the
square root. It does not add epsilon inside the root or average the squares.
For `[3,4]`, epsilon 10 gives `[0.3,0.4]`; epsilon below 5 gives `[0.6,0.8]`.
F64 squaring also preserves tiny F32 inputs and avoids F32 square overflow.

RMSNorm does not include learned weights, gating, or a model-specific weight
offset. Adapters must apply those explicitly. It squares in F64 to avoid F32
overflow on large finite inputs. Softmax subtracts in F64 before exponentiation,
so even opposite F32 extremes are safe. It finishes its maximum and sum
reductions before writing and recomputes exponentials rather than allocating
an F64 temporary vector. This is a clarity/memory choice, not an optimization.

Sigmoid splits its computation by input sign to avoid exponential overflow.
Softplus uses `max(x,0) + log1p(exp(-abs(x)))`; `log1p` preserves the small
negative-input tail that `log(1 + tiny)` can lose to rounding.

Six numerical test blocks check exact normalization examples, zeros and large
inputs, softmax probabilities and translation invariance, masks and all-masked
errors, in-place operation, activations' finite values and nonfinite limits,
and invalid inputs without partial writes. The nontrivial constants were checked
independently using Python `decimal` with 80-digit precision; they are formula
references, not measurements of llama.cpp kernels. F32 transcendental results
use absolute tolerance `1e-7`, with relative `1e-6` for the tiny softplus tail.
These CPU test tolerances do not establish future GPU acceptance tolerances.

## Unscaled text RoPE

[rope.zig](../../inference/src/backends/cpu/rope.zig) exposes
`cpu.rope.apply(input, output, options)` for one head. Options explicitly supply
an even, nonzero `dimensions` no larger than the head width, a finite frequency
`base >= 1`, and an `i32` token `position`. Let `half = dimensions/2`:

```text
theta[i] = position * base^(-i/half)
y[i]      = x[i] * cos(theta[i]) - x[i+half] * sin(theta[i])
y[i+half] = x[i] * sin(theta[i]) + x[i+half] * cos(theta[i])
```

Only the rotary prefix changes; the tail is copied unchanged. Position zero
preserves all bits, including signed zero. Negative positions apply the inverse
rotation up to rounding. Slices must be equally sized and nonempty, and every
input value must be finite. Exact alias is supported; partial overlap is
forbidden. Shape, base, and input checks finish before writes. No allocation or
I/O occurs. Intermediate arithmetic is F64, with final F32 rounding; rotation
of extreme finite inputs can overflow F32 and produce infinity.

The pinned model rotates 64 of each head's 256 channels with base 10,000,000.
Its text positions repeat the same index across the temporal/height/width
streams, and the fourth rotary section is empty. Therefore its unscaled text
MRoPE reduces to this ordinary split-half rotation. This operator does not
implement different multimodal positions, adjacent-pair layout, frequency
scaling, YaRN, or learned frequency factors. The future adapter must supply
positions and traverse Q/K heads; calling this operator does not run attention.

Synthetic tests cover hand-calculated angles with two frequencies, rotation
followed by its inverse, pair-norm preservation, exact alias, unchanged tails,
identity, overflow behavior, and invalid inputs. The fixed angle example uses
absolute tolerance `5e-7`; norm preservation uses relative tolerance `2e-7`.

## Pinned vector fixtures

The committed [fixtures](../../inference/src/backends/cpu/fixtures/vector.json)
come from llama.cpp `7620399f58aebfd2196b74021f9581bcf7218cb9`, its matching local
Release build, and one CPU thread. Reproduce them without a model or GPU:

```sh
python3 scripts/cpu-vector-fixtures.py
```

The script uses the local C ABI to build and compute small graphs. Four L2 cases
cover a 3–4 vector, denominator clamping, zero, and tiny values. Four RoPE cases
use a synthetic 256-channel head at positions 0, 1, 127, and 32,767, with rotary
width 64, base 10,000,000, IMRoPE mode 40, sections `[11,11,10,0]`, positions
`[p,p,p,0]`, frequency scale 1, extension factor 0, and attention factor 1.
The [reference guide](reference-baseline.md) documents rebuilding the checkout.

L2 fixtures use absolute tolerance `1e-7`. RoPE's rotated channels use `5e-6`
through position 127 and `1e-3` at 32,767; tails match exactly. The largest
observed discrepancy against the latter fixture is about `9.4e-4`. The reference
advances F32 angles by repeated multiplication, while ours evaluates each
frequency directly in F64. Differences grow with position, so this comparison
is not bitwise equivalence. It does not set future GPU tolerances or establish
end-to-end inference accuracy. The fixtures avoid extreme values where the
reference's F32 squaring overflows or underflows but our F64 L2 reference does not.

## Single-position grouped-query attention

[attention.zig](../../inference/src/backends/cpu/attention.zig) exposes
`cpu.attention.apply(input, output, scratch)` for one query position. Its input
records dimensions, explicit positive finite score scale, borrowed arrays, and
the number of visible KV tokens. The operation computes:

```text
kv_head = query_head / (query_heads / kv_heads)
score[t] = scale * dot(query[query_head], key[t, kv_head])
probability = softmax(score[0..visible_tokens])
output[query_head, c] = sum(probability[t] * value[t, kv_head, c])
```

Queries use `[query_head][key_channel]` order. Keys and values use
`[token][kv_head][channel]` order; output uses `[query_head][value_channel]`.
Query heads must be a positive multiple of KV heads, with consecutive query
head groups sharing one KV head. Key and value widths may differ. For the
pinned model's 24 query heads and four KV heads, each group has six query heads;
the future adapter supplies scale `1/sqrt(256)` after Q/K normalization and RoPE.
The primitive itself chooses neither scale nor model equations.

Only the leading `visible_tokens` KV entries participate. The caller can choose
a different prefix for each prefill position or expose the full current cache
for decode. A zero visible prefix returns `EmptySupport`; a prefix longer than
storage is invalid. `tokens` counts initialized records, not unused cache capacity.
This operation does not create or append a cache and does
not infer a causal position. Arbitrary additive masks, sliding windows,
position biases, and attention sinks are outside its contract.

All dimensions must be nonzero, all buffers must have their exact declared
sizes, and products are overflow-checked. Every Q/K/V value, including masked
storage, must be finite. Validation finishes before output or scratch changes.
The scratch buffer holds at least `visible_tokens` F64 values; its unused suffix
is untouched. It is reused across heads. Writable buffers must not overlap each
other or any input. No allocation or I/O occurs; the function borrows all buffers.

Dot products, scaled scores, exponentials, and weighted sums remain F64. This
avoids narrowing large finite dot products into F32 infinity before softmax.
The reference's cache is F32 by decision (KERN-07): `Session.Rows.floats` asserts
an `f32` layout, the CPU runtime never asks for another, and an `f16`
request on `--backend cpu` runs F32 and reports it. The F16 cache is a GPU
layout whose tolerance against this reference is recorded in
[metal-backend.md § F16 KV cache](metal-backend.md#f16-kv-cache-kern-07).
Subtracting the maximum makes exponentials at most one, with at least one
nonzero term. Values are accumulated with these unnormalized weights and divided
by their sum before the final F32 conversion. The existing F32 softmax API is
therefore not used as an intermediate narrowing step. Work is proportional to
query heads times visible tokens times the combined key/value widths, in addition
to validation of all declared storage. This is a correctness reference, not a
measured throughput implementation.

Five test blocks cover analytical probabilities and scaling, visible-prefix
masking, distinct GQA groups and channels, scores beyond F32 range, malformed
shapes, overflow, masked nonfinite storage, and validation before writes.
The [attention fixtures](../../inference/src/backends/cpu/fixtures/attention.json)
add six pinned one-thread CPU graphs: four query/two KV heads or three query/one
KV head, key width three, value width two, and visible prefixes one, two, or four.
The reference composes `ggml_mul_mat`, `ggml_soft_max_ext`, and `ggml_mul_mat`,
materializing only the visible KV prefix. Absolute tolerance is `2e-6` for these
small cases; this is not an end-to-end attention-layer or future GPU tolerance.
Regenerate them with the same `scripts/cpu-vector-fixtures.py` command above.
No model or GPU is needed to generate or run these fixtures.

## Causal convolution and DeltaNet state

[recurrent.zig](../../inference/src/backends/cpu/recurrent.zig) provides two
single-step references with caller-owned F32 state and F64 calculations.
Neither allocates memory nor performs I/O. The caller supplies sequence order,
initial state (normally zeros), head mapping, and any checkpoint copies.

`recurrent.convolution(input, weights, history, output, kernel)` is depthwise:
each channel has independent weights and history. Weights are `[channel][tap]`,
oldest tap first and current-input tap last. History is `[channel][kernel-1]`,
oldest first. After computing a successful output, history drops the oldest
sample and appends the current input. Kernel one needs no history. No bias or
SiLU is included. Buffers must be disjoint. A validation pass and a numerical
check pass ensure every failure leaves both history and output unchanged.

`recurrent.delta(parameters, state, next_state, output, scratch)` handles one
scalar-gated head. State uses `[value_channel][key_channel]` order, permitting
rectangular matrices. The supplied query/key vectors have equal width; value
and output widths match the number of state rows. With `a = exp(log_decay)`:

```text
D[j,i]    = a * state[j,i]
error[j]  = value[j] - sum_i(D[j,i] * key[i])
next[j,i] = D[j,i] + beta * error[j] * key[i]
output[j] = scale * sum_i(next[j,i] * query[i])
```

The prediction uses the **decayed** state, and output reads the **updated**
state. `log_decay` must be finite and nonpositive, `beta` finite in `[0,1]`, and
query scale finite and positive. Q/K normalization, sigmoid beta, log-decay
preparation, and output normalization/gating are caller responsibilities. KDA
per-channel decay and automatic head broadcasting are not implemented here;
the chunkwise form is `deltaChunk` below. The pinned fused reference maps repeated Q/K heads
with modulo indexing; the future adapter must not reuse attention's consecutive
GQA grouping blindly (the pinned model has 16 Q/K and 48 value heads).

`next_state` may exactly alias `state`; other overlap is forbidden. Scratch
requires `state.len + value.len` F64 elements, and its suffix is untouched.
Candidates are validated before committing any next state or output. Shape,
gate, and nonfinite-input errors modify no buffers. Unrepresentable F32 results
return `NonFiniteResult`; scratch may change, but durable state and output do
not. State rounds to F32 between steps, with F64 calculations within each step.
This differs from the reference backend's intermediate F32 rounding and uses
explicit tolerances rather than claiming bitwise agreement.

Eleven test blocks cover convolution orientation and independent channels,
updated history, kernel one, the DeltaNet equations, beta/decay extremes,
rectangular state, replay from an explicitly copied checkpoint, and transactional
failure including overflow in a later channel. Three consecutive convolution and
DeltaNet steps are checked against [pinned fixtures](../../inference/src/backends/cpu/fixtures/recurrent.json),
carrying our own state between steps. DeltaNet outputs and updated matrices use
absolute tolerance `1e-6`; the exactly representable convolution examples and
history shifts match exactly. The fixture generator calls `ggml_ssm_conv` and
`ggml_gated_delta_net` on the pinned one-thread CPU backend; the latter exports
both output and final state. Expected convolution history is the explicit window
shift, not an output of `ggml_ssm_conv`. Regenerate with
`scripts/cpu-vector-fixtures.py`; no model or GPU is needed.

### Chunkwise DeltaNet (ENGN-04, stage 1)

`recurrent.deltaChunk(chunk, state, next_state, output, scratch)` computes
`C` consecutive steps of the same head without stepping the matrix per
token. With `a_t = exp(log_decay_t)`, `γ_t = Π_{r≤t} a_r`, and
`r(t,s) = γ_t / γ_s` (computed as `exp(L_t − L_s)` from cumulative log
decays so long chunks never divide by an underflowed product), the
sequential update `S_t = a_t S_{t−1} + u_t k_tᵀ` with
`u_t = β_t (v_t − a_t S_{t−1} k_t)` unrolls to
`S_t = γ_t S_0 + Σ_{s≤t} r(t,s) u_s k_sᵀ`. Substituting that into `u_t`
gives a strictly lower triangular system, the WY form:

```text
A[t,s] = β_t · r(t,s) · (k_s · k_t)          for s < t
B[t]   = β_t · (v_t − γ_t · S_0 k_t)
(I + A) U = B                                 forward substitution: u_t = B_t − Σ_{s<t} A[t,s] u_s
o_t    = scale · (γ_t · S_0 q_t + Σ_{s≤t} r(t,s) · (k_s · q_t) · u_s)
S_C    = γ_C · S_0 + Σ_s r(C,s) · u_s k_sᵀ
```

Each `u_t` depends on earlier `u_s` only through inner products of keys,
never through the matrix, which is what a GPU kernel can batch. Inputs are
`[token][channel]` rows plus per-token log decays and betas; the same
validation rules as `delta` apply per token, everything inside the chunk is
F64, `state`/`next_state` are F32 like the session (`next_state` may alias
`state`), and validation or overflow leaves both outputs untouched. Scratch
is `2C² + 3CV + VK + C` F64 elements (`deltaChunkScratch`).

Evidence (2026-09-09, four test blocks): against an F64-state sequential
loop on a model-shaped chunk (C = 64, 128×128 state, random gates in the
model's ranges) the outputs and carry are **bit-identical after the F32
cast** (bound 1e-9); against the production `delta`, which rounds state to
F32 per step, one chunk of 64 and two chunks of 40 + 24 with the state
carried between them agree to 2.4e-7 (bound 1e-4); the three pinned
sequential fixture steps run as one chunk reproduce the pinned outputs and
final state to 1e-6; validation failures and the single-token case match
`delta` exactly. The Metal kernel (`nu_delta_chunk`, stage 2) is checked
against this function and against the sequential steps.

A checkpoint test demonstrates buffer restoration, not a shipped session or
rewind API. Session rollback must restore convolution history and DeltaNet
matrices along with attention KV and token position. The next implementation
work is computational weight access and model/session composition, followed by
full-layer and logit comparisons and a minimal token loop. These primitive tests
do not yet generate text.

## Mixture-of-experts routing and the gathered FFN

[experts.zig](../../inference/src/backends/cpu/experts.zig) is the reference
for a mixture-of-experts layer's routing and expert projections; the chain
a model applies before the router logits (norms, scales) and after the sum
(post norms) belongs to its adapter (Gemma 4 26B-A4B's is in
[gemma4.md](gemma4.md)).

`ExpertMatrix` is a borrowed 3-D encoded tensor `[experts][rows][columns]`
(GGUF dimension 2 is the expert): `experts` contiguous `rows × columns`
matrices of equal byte length. `expert(index)` returns one as a `Matrix`
after validating that the bytes divide into whole rows of whole blocks.

`route(logits, indices, weights)` selects `indices.len` experts for one
token: the largest **logits** by (value desc, index asc), then their
softmax probabilities (F64) renormalized to sum one, with the sum clamped
below at the smallest F16 normal (`weight_sum_floor`, the reference's
clamp against a zero denominator). Selection compares the logits rather
than the probabilities because softmax is monotone, so a kernel that
rounds `exp` differently still selects the same set — the GPU router's
indices are checked exact against this. Logits must be finite; at most
`max_experts` (1,024).

`ffn(spec, input, indices, weights, output, scratch, accumulator)` is the
gathered gated-GELU FFN of one token: for each slot `s`, `gate_up[e_s] ·
input` (the expert's gate rows followed by its up rows, the reference's
fused layout), `gelu(gate) ⊙ up`, `down[e_s] ·` that, times the optional
per-expert `down_scale[e_s]`, times `weights[s]`, summed over the slots
in F64. Scratch is `Ffn.scratchLen()` floats plus one F64 per output. The
Metal decode chain (`route`, `matvecExperts`, `geluMulRows`,
`matvecExperts`, `combineExperts`) is checked against it in `test-metal`
([metal-backend.md § Gathered expert kernels](metal-backend.md#gathered-expert-kernels-kern-09)).

## The Hadamard rotation's activation side

[hadamard.zig](../../inference/src/backends/cpu/hadamard.zig) is the
reference for the transform a Hadamard-folded file (Bonsai 2 27B,
[bonsai.md](bonsai.md#rotation-prismhadamard-as-the-forks-loader-reads-it))
needs on every projection input: `forward(x, signs, block)` multiplies each
element by its ±1 sign and then applies the normalized Sylvester
Walsh-Hadamard transform in place to every `block` consecutive elements
(`H[r][c] = (-1)^popcount(r & c) / sqrt(block)`, ten butterfly stages in
F64 scratch for a 1024 block); `inverse` is the same butterflies followed
by the signs, `(H S)^-1 = S H`, which is what an embedding row stored in
the rotated basis needs after lookup. Blocks must be powers of two up to
1024 and divide the width; the sign vector has the width's length.
Tests check the butterflies against the parity-defined matrix for blocks
1, 2, 8, and 1024, the round trip, a constant and a delta block by hand,
and that rejected shapes leave the input untouched. Which activations
take it, and the value-head regathering before `ssm_out`, are the
adapter's runtime's business ([bonsai.md](bonsai.md#cpu-reference-against-the-fork-modl-16-2026-09-18)).

## Validation and limits

Five matrix-vector test blocks cover:

- Hand-calculated rectangular F32 and F16 examples and unused scratch capacity.
- Cancellation that distinguishes F64 accumulation from F32 accumulation.
- Multiple independently scaled Q8_0 blocks in each of two matrix rows.
- Invalid dimensions, insufficient buffers, malformed storage, and unsupported
  formats, with output and scratch unchanged on error.
- Basis vectors selecting first, interior, and last columns from pinned C
  decoder fixtures for every quantized encoding.

The basis-vector cases test matrix orientation and row offsets against known
decoded values; they are not an end-to-end comparison against llama.cpp's matrix
kernels. The independent hand-calculated dot products test summation and signed
contributions. Default tests require no reference checkout, model, or GPU:

```sh
zig build test --global-cache-dir .zig-cache/global
```

The freshly built `nuclis validate --json` reports `validation: structure_only`
and `inference_available: true`: a successful binding is the profile `generate`
executes on the CPU and Metal backends.

## Full-model composition

The CPU references now execute the full pinned Qwen text schedule through
`models/qwen35_runtime.zig`. See [generation](generation.md) for CLI usage,
full-layer/logit comparisons, and explicit session isolation/reset checks.
Metal remains pending; this implementation is for numerical bring-up.
