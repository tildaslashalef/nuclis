# Building an inference engine: the nuclis guide

This is the companion to the inference stack: what it took to make a 27
billion parameter model answer on one laptop, told in the order the ideas
matter rather than the order they were built. Every technique in it is
implemented in this tree, and every number is a measurement from it. It is
written to be read end to end in an evening and to be revisited one section
at a time; each section ends with the files to read next.

It is not about the agent or the terminal (that is
[reference/agent-concepts.md](reference/agent-concepts.md)), not the map of
the code ([architecture.md](architecture.md)), and not the requirements
([spec.md](spec.md)). It is about the ideas, and about the moments where an
idea turned out to be wrong and a measurement said so.

A note on the hardware, because it is a character in the story: an Apple
M4 Pro with 48 GB of unified memory and a 273 GB/s memory bus. The models
are Qwen3.8-27B (the pinned target: 64 layers, 16 of them attention and 48
recurrent), Gemma 4 in a dense 12B and a sparse 26B, Muse Glimmer 30B, and
Bonsai 2, which is Qwen3.8 again at 1.58 bits per weight.

---

## Part One — The problem

### 1. The numbers that decide everything

Start with a division. The Qwen file is 16.1 GB of weights. To produce one
token, a dense model reads every weight once. At 273 GB/s that read takes
59 ms, so the fastest this machine can ever decode this model is about 17
tokens per second, whatever the code looks like. llama.cpp, a mature
engine with years of Metal work in it, reaches 9.66 on the same file.
nuclis reaches 10.62. The gap between 10 and 17 is the entire subject of
Part Five.

The second number is the prompt. Reading the prompt (*prefill*) is a
different problem from producing the answer (*decode*): the same 16 GB of
weights can be multiplied against 512 prompt tokens at once, so the bytes
are amortized and the arithmetic units become the limit. The reference
prefills at 89 tokens per second and decodes at 9.66; both numbers come
from the same weights on the same bus, and the ratio is Part Six.

The third is memory. 27 billion parameters at two bytes each would be 54
GB, more than the machine has, so the weights are stored at about four bits
on average and never expanded. The conversation's memory, the cache of
keys and values every attention layer keeps, is 64 KiB per token at F32 on
this model: a 32K-token context is 4 GiB of state beside the weights, or 2
at F16. That is Part Seven.

Everything else follows from those three: bytes per token, tokens per
weight read, and bytes per token of context.

### 2. What an engine is

A trained model is a set of numbers. An engine is everything that makes
them do something:

```text
model file        → validated weights and architecture facts
messages          → the prompt template → the tokenizer → token ids
ids + weights + session state → the layer schedule → logits
logits            → the sampler → the next id → text
                    ↳ feed the id back and repeat
```

Three kinds of memory live in that loop, with different owners and
lifetimes, and getting them apart is most of the design. **Weights** are
immutable and memory-mapped from the file; no floating-point copy of them
ever exists. **Session state** is per conversation and mutable: the
attention cache rows and the recurrent matrices, carved out of one
page-aligned block so the GPU can address it without a copy. **Activations**
are scratch, overwritten every token. When a later section says "session"
it means the middle kind, and when it says a layer "writes its row" it
means into that block.

Read: `inference/src/engine.zig` (`runLoop` is the whole loop),
[architecture.md § 3](architecture.md#3-three-kinds-of-memory).

---

## Part Two — The file

### 3. A directory you can trust

A GGUF file is a directory followed by weights: a magic number, counts, the
metadata (architecture facts, the vocabulary, the chat template), then one
descriptor per tensor with its name, shape, storage encoding, and offset.
The directory says everything an engine needs to decide whether it can run
the file, and none of it needs the weights. Qwen's directory is 11 MB; the
vocabulary lives in it.

The parser's job is refusal. It checks dimensions, encoding sizes,
duplicate names, offsets, alignment, overlap, and file bounds before it
lets anyone read a byte of weight, and it bounds its own allocations
against a file that lies about its counts. `nuclis model inspect` runs the
same parser over the head of a file fetched in 8 MiB windows from the Hub,
with the real file size for the bounds checks, and stops when the parser
stops asking for more. Four seconds decide whether a 16 GB download is
worth starting.

That decision has four levels, and the code base once blurred them.
*Storable*: the parser knows the encoding's block size. *Executable*: the
adapter has kernels for it. *Bindable*: every tensor the architecture
expects is present with its shape and nothing unexpected is. *Supported*:
all of that, and the catalogue pins the Hub's digest for the file. The
verdict names the first offender by tensor name, because "unsupported" with
no noun costs an afternoon.

Read: `inference/src/formats/gguf.zig`, `src/model.zig` (`inspect`),
[reference/gguf-inspection.md](reference/gguf-inspection.md).

### 4. Eleven ways to store a weight

Quantization is the reason the model fits. A block encoding groups values
(32 or 256 of them) under shared metadata, and its decoding equations say
what the bytes mean. The Qwen file mixes nine encodings; the tree decodes
eleven, and each one taught something about the difference between
counting bits and knowing a format.

**A scale and a code.** Q8_0 is the simplest: an F16 scale and 32 signed
bytes. IQ4_NL keeps four-bit indices into a fixed table of sixteen
non-uniform values, then scales. Even here packing order is part of
correctness: the two nibbles of one byte are values 16 positions apart,
and reading them as neighbours produces plausible numbers in the wrong
places, which is worse than a crash.

**Scales that are themselves quantized.** Q4_K and Q5_K store 256 values as
eight groups, each with a six-bit scale and a six-bit offset that are
interpreted through two shared F16 coefficients: `(d·scale)·q − (dmin·min)`.
An offset means a zero code can be a negative weight. Q5_K's fifth bit
lives in a separate array, a *bit plane*, and restoring each weight's five
bits from two places is the whole decoder.

**Signs are a representation choice.** Q3_K's two low bits go negative when
a separate mask bit is clear; Q6_K adds a bias of 32 to an unsigned code.
Both scales differ too. Two encodings of the same size, 110 bytes per 256
values, can describe different arithmetic: IQ3_S spends a nine-bit index on
a *codebook* of 512 four-value patterns plus signs, so one code is a small
vector, not a number.

**Q4_0** is 32 values, an F16 scale, and sixteen bytes of nibbles with the
value `d·(q − 8)`: IQ4_NL's block with the table replaced by a subtraction.
It matters because Google's quantization-aware-trained Gemma is stored in
it, and that file is only meaningful if the engine reproduces exactly the
arithmetic its training simulated. The loader never requantizes, which is
what makes the choice of file mean something.

**Ternary.** Bonsai 2 stores each weight as one of `{−1, 0, +1}` times a
scale per 128. PQ2_0 spends two bits per trit. PTQ1_0 packs five trits in a
byte in base 3, because 3⁵ = 243 fits in 256: the byte is pre-scaled so the
leading digit is `(byte·3) >> 8`, and multiplying by 3 modulo 256 promotes
the next. Five multiplies, no division, and the values come out
digit-major, so element `j` is not in byte `j/5`. The fixture feeds the
decoder arbitrary bytes on purpose: matching on codes a valid file never
contains proves the arithmetic, not just the packing.

One file in the download directory is none of these: the importance
matrix, 992 F32 tensors of activation statistics used *by the quantizer*
to decide which channels deserve precision. It is not a layer to run and
does not restore anything when loaded beside the weights it shaped.

Three rules survived all eleven. The CPU decoder is the truth, pinned
against the reference implementation's own C decoders through fixtures.
Every GPU decoder is bit-identical to it, because they index the same bytes
with the same equations. And a specialized kernel folds the scale out of
the sum (`Σ(d·s·q − dmin·m)·x = d·s·Σq·x − dmin·m·Σx`) as algebra, never
as approximation, so a one-hot input reproduces the decoder's expression
exactly and the fixture columns stay exact on the GPU.

Read: `inference/src/quant/decode.zig`, `inference/src/tensor/encoding.zig`,
[reference/quantization.md](reference/quantization.md).

### 5. The rotated basis

Rounding to three levels is brutal on a matrix whose columns have a few
large entries and many small ones: the scale follows the outliers and the
rest round to zero. Activations have the same shape. Bonsai's remedy, the
one QuaRot and SpinQuant use, is a change of basis. For any orthogonal
`R`, `W·x = (W·Rᵀ)·(R·x)`: rotate the weight's input axis once at
conversion time, rotate the activation the same way at every token, and
the product is unchanged while both operands look Gaussian.

The rotation is `(1/√1024)·H·S` on each block of 1024 elements: `S` a
diagonal of random signs shipped in the file header, `H` the Sylvester
Walsh-Hadamard matrix, `H[r][c] = (−1)^popcount(r & c)`. `H` needs no
storage and costs ten butterfly stages of additions, and the random signs
are what stop a structured input from being an eigenvector of a known
matrix. Because `H` is its own inverse, the embedding table can be stored
rotated and undone after lookup by the same transform.

Four activations per layer meet rotated weights; the transform is
memoized per activation. Two projections, the DeltaNet gates, are stored
unrotated at BF16 because a rounding error there compounds through the
recurrence, and the adapter pins the list of *which* inputs rotate rather
than trusting the file. On the GPU the transform costs 1.66 ms of a 78 ms
token: at batch one, a small kernel's cost is its launch and its
dependency, not its arithmetic, and this is 258 more of them. The bytes are
the point: 7.2 GB against 16.5, and 12.87 tokens per second where the
forked reference reaches 17.05.

Read: `inference/src/backends/cpu/hadamard.zig`, `models/qwen35.zig`
(`Rotation`), [reference/bonsai.md](reference/bonsai.md).

---

## Part Three — Text

### 6. Tokens are bytes with a history

A tokenizer is not a dictionary. Byte-pair encoding starts from bytes, each
mapped to a reversible symbol, and repeatedly merges adjacent pairs in the
order the training assigned them: `abc` becomes `a` + `bc` if the pair
`bc` outranks `ab`, even when a token `abc` exists. Merge rank is the
algorithm; the vocabulary's order is the model's meaning, so nothing is
ever sorted.

Before any merge runs, the *pre-tokenizer* decides where merges are legal.
Qwen's rules split every Unicode number into its own piece, so `1` and `2`
can never merge across that boundary however the ranks fall; combining
marks stay with their letters. Those rules need Unicode categories, and
using the host's Unicode tables would let a system update move a boundary,
so the category table is pinned data like the vocabulary. Special markers
are claimed first, longer types before shorter, and a claimed span is
never split.

Three families meant three splitters and two vocabulary models. Gemma's is
SentencePiece-style: symbols are code points, spaces become `▁`, the only
pre-split is around newline runs, and an unknown symbol falls back to
`<0xNN>` byte tokens. Muse uses the gpt-4o pattern as the reference
realizes it. The rank-ordered merge loop is one function with three
callers.

Decoding is the reverse, with one trap: a token may hold the first byte of
a multibyte character, valid tokenizer output that is not valid text. The
stream keeps up to one incomplete scalar between tokens and flushes
complete text; the ids stay exact. And the way you know a tokenizer is
right is not that the words look right: nuclis reproduces the reference's
token arrays on an 83,384-token corpus at every cut the benchmarks use,
including a 32,620-token one.

Read: `inference/src/tokenizer/encode.zig`, `bpe.zig`, `stream.zig`,
[reference/tokenizer.md](reference/tokenizer.md).

### 7. The prompt profile is a contract

The same weights are useless without the exact text the checkpoint was
trained to see: role markers, separators, the reasoning switch, the
assistant prefix. nuclis calls that the *prompt profile*, keeps it apart
from the architecture, and selects it by the SHA-256 of the chat template,
never by the architecture name. Two files of one family can ship different
templates; the K-quant and QAT Gemma files share one.

A Jinja template is a small program, and the reference runs it with an
interpreter. nuclis does not: the profile is a Zig function that renders
the subset of conversations the agent produces, and the evidence that it
renders them correctly is a fixture the reference server produced from
the artifact's own template. Reading Gemma's 19 KB template for that
subset reduced it to a dozen clauses, several of which the model card
gets wrong: reasoning is rendered only after the last user message,
`developer` is sent as `system`, two assistant messages continue one turn.

The second family found two Qwen facts hiding outside the profile. The
engine stopped on `eos or bos`, which on Gemma would have stopped at the
first `<bos>` and never at `<turn|>`; and the chat split every turn on the
literal `</think>`. Both moved into the profile: a stop set of texts
resolved to ids at load (a template whose end marker the vocabulary lacks
refuses to load), and a pair of reasoning markers the surface splits on.
When the template defines tool calling, the profile owns that too, in
both directions: it renders the tool definitions and results into the
template's own grammar and decodes the model's output into typed calls at
token boundaries. Qwen's XML-like calls, Gemma's `call:NAME{…}`, Muse's
ATEM grammar: three wire formats, and nothing outside `profiles/` names
any of them.

One more collapse is deliberate. The shared effort enum has four levels
because Qwen has four; Gemma's template has a switch. A profile maps what
it cannot express (`off` is off, everything else is on) rather than
pushing a per-profile type into every caller.

Read: `inference/src/profiles/root.zig`, `profiles/gemma4.zig` (the
module comment is the contract),
[reference/prompt-profile.md](reference/prompt-profile.md),
[reference/tool-calling.md](reference/tool-calling.md).

---

## Part Four — One token on the CPU

### 8. Slow on purpose

Every operation the GPU runs exists first as a plain function in
`backends/cpu/` with F64 accumulation and its own tests, and the CPU
schedule that composes them takes about eighteen seconds per token. That
is not a prototype; it is the oracle, and its value is being obviously
correct. When the GPU and the CPU disagree, the CPU is presumed right until
proven otherwise, and the trace comparison says at which layer.

The matvec decodes a row into scratch and sums products in F64; the test
`16777216 + 1 − 16777216` keeps the `1` where F32 loses it at the first
addition. RMSNorm puts epsilon inside the square root; L2 normalization
bounds the denominator instead. Softmax subtracts the maximum before
exponentiating and refuses an all-masked vector rather than returning
zeros. Softplus uses `log1p` so `1 + tiny` does not round the tiny away.
RoPE rotates pairs at split-half stride with F64 frequencies, and the
tests separate mathematical invariants (rotations preserve length; opposite
positions cancel) from agreement with another implementation, because the
reference's repeated F32 updates drift at large positions and a tolerance
against it is evidence about that case only.

The habit this layer sets for the whole tree: validate every shape and
every finite value first, then write. A failure leaves the caller's buffers
untouched, and tests assert that.

Read: `inference/src/backends/cpu/vector.zig`, `rope.zig`,
[reference/cpu-reference.md](reference/cpu-reference.md).

### 9. Attention reads, DeltaNet writes

Attention is a lookup over everything seen so far. For one query, score
every stored key, softmax the scores, and combine the values by those
weights; keys decide relevance, values carry content. Two structural
facts shape every kernel later. *Grouped-query attention*: 24 query heads
share 4 key/value heads on Qwen, so a cache row is read by six queries.
*Causality*: a query sees the visible prefix and nothing after it, and a
masked key is excluded from the softmax, not given a zero score.

Qwen's other 48 layers are Gated DeltaNet, and they do the opposite. Each
head keeps a 128 × 128 matrix, an associative memory: applying a key
predicts a value, the delta rule writes back the prediction error scaled
by a learning rate, a decay gate decides how much of the old memory
survives, and the query reads the updated matrix.

```text
S ← a·S
S ← S + β·(v − S·k)·kᵀ
out = scale·S·q
```

A short causal convolution over the last four inputs sits in front of it.
The consequence that shapes the session design: this state is a function
of every token ever fed. Attention forgets the last token by setting a
length. DeltaNet has already mixed it in, and no arithmetic undoes that
after rounding. The spec forbids rewinding a hybrid model by truncating the
cache, and Part Seven and Part Eight are largely about what to do instead.

Read: `inference/src/backends/cpu/attention.zig`, `recurrent.zig`,
[architecture.md § 4](architecture.md#4-one-qwen-layer).

### 10. A layer, a token, a session

A layer is norm, mixer, residual, norm, feed-forward, residual, and the
schedule is that 64 times, then a final norm and the vocabulary
projection. The first working CPU run agreed with the reference's
per-layer traces on the first try in one respect and taught wiring in
another: the query-and-gate packing and the DeltaNet head mapping (16
key heads broadcast to 48 value heads by `h % 16`, unlike attention's
consecutive groups) are the kind of fact no isolated operator test can
catch. Traces catch them.

Three families, three layer menus. Gemma alternates sliding-window layers
that see the last 1,024 positions with global layers whose heads are 512
wide, applies a tanh GELU, scales its residual stream, and in the 26B
replaces the feed-forward with 128 experts of which 8 fire. Muse is dense,
30B, with windowed layers and 13 global ones. Bonsai is Qwen's schedule on
rotated weights. What did not change across them is the point of Part Nine.

The session is where a token becomes durable. It is one page-aligned block
with typed views per layer: `Rows` of keys and values for attention (F32
or F16), `[]f32` for recurrent state, and a tiny state machine: `ready →
updating → ready`, or `failed` until `reset`, because a step that fails
halfway has already mutated some recurrent layers and half-updated
matrices are not a state anyone should keep. `snapshot` copies the used
extent (the recurrent state whole, 64 KiB per token of cache) behind a
digest of the layout and capacity, so F16 bytes cannot land in an F32
session. The proof is bitwise: checkpoint after token one, compute token
two, restore, compute it again, identical logits on both backends.

Read: `models/qwen35_runtime.zig` (`fullAttention`, `linearAttention`),
`inference/src/runtime/session.zig`,
[reference/session.md](reference/session.md).

---

## Part Five — The GPU

### 11. Crossing the bridge

A GPU does not run Zig. Kernels are written in Metal Shading Language,
compiled by the driver into pipeline states, and work is described into a
command buffer: an encoder records "bind these buffers, run this pipeline
over this many threads", `commit` hands the buffer over, and nothing has
executed until then. The bridge in `bridge.m` keeps every Metal object
behind one opaque handle; Zig sees a dozen C functions and never a Metal
type. It is compiled without automatic reference counting so every retain
and release is a line you can read.

Unified memory is the gift of this hardware: the GPU addresses the same
RAM the CPU mapped the file into. `newBufferWithBytesNoCopy` wraps a
page-aligned range with no copy, which is why weights are never copied and
why the session is one page-aligned block. The obligation that comes with
it is lifetime: the mapping must outlive the backend, and the CPU may
touch session memory only after the command buffer that writes it has
completed.

The first bridge was correct and slow: a command buffer and a wait per
operation, several hundred per token, 2.3 tokens per second. Recording a
whole token into one buffer gave 5.0 with the same kernels. Later the
profiler found the CLI's layer observer quietly committing 64 buffers per
token on every run for the sake of Ctrl-C, and splitting it into a
values-free `check` took decode from 8.5 to 9.7 with no kernel change.
Synchronization is a cost like bandwidth, and it hides because it is
nobody's kernel.

The wait has one refinement, added for the agent: `commit` waits on a
semaphore the buffer's completion handler signals, with a 100 ms timeout
that calls back into whoever installed a `tick`. A 95 ms decode step never
pays for it; a three-second prefill chunk lets the terminal repaint thirty
times.

Read: `inference/src/backends/metal/bridge.m` (200 lines), `root.zig`
(`Backend.commit`), [architecture.md § 7](architecture.md#7-metal-for-a-zig-programmer).

### 12. Lanes, SIMD groups, and where they live

A dispatch makes a grid of threadgroups; each threadgroup runs as SIMD
groups of 32 lanes that execute the same instruction on different data.
Four levels: grid, threadgroup, SIMD group, lane. Zig chooses the grid and
the thread count and holds the constants the kernels assume (rows per SIMD
group, tile padding); the bridge forwards the geometry; only
`kernels.metal` ever names a lane. The contract between the layers is a
handful of numbers that must agree, and a comment on each side says so.

A SIMD group can do three things a plain thread cannot, and the kernel
shapes in the tree are built from exactly those. It can reduce across
lanes without memory: `simd_sum` finishes a dot product in a few cycles,
which is why the decode matvec gives one output row to one SIMD group. It
can exchange values between named lanes: `simd_shuffle_xor` combines the
four lanes that own one query row in the attention kernel. And it can
multiply 8 × 8 matrices held across the group: `simdgroup_float8x8`, 64
floats spread two per lane in an undocumented layout, which is why
row-wise work like a softmax goes through threadgroup memory to plain
threads and the matrix unit only ever multiplies.

The rule for geometry: put in one SIMD group the work that wants to be
reduced or exchanged, in one threadgroup the work that wants to share a
tile of memory, and leave the rest to the grid. Two barriers exist because
lanes in one SIMD group are already in lockstep (`simdgroup_barrier` is
nearly free) and SIMD groups sharing a tile are not
(`threadgroup_barrier` stops the group). The CPU reference uses none of
this and no Zig vector types either; it is scalar F64 loops so that it
stays the plain statement of the math.

Read: `kernels.metal` starting with `nu_add` and `nu_rmsnorm`,
[reference/metal-backend.md § Kernel geometry](reference/metal-backend.md#kernel-geometry).

### 13. The matvec that reads 16 GB

Decode is "read every weight once per token", so the kernel that turns
quantized bytes into a dot product decides everything. The first version,
one SIMD group per row, sixteen values per lane by byte loads and a
runtime switch on the encoding, reached about 100 GB/s of the bus's 273.
The specialized kernels reach 150 to 250, and four ideas did the work.

**Vector loads that match the block.** In Q4_K the sixteen bytes at
`16 + 32·pair + 16·half` hold the low nibbles of one 32-value group and the
high nibbles of the next; a lane that loads them as one `uint4` owns 32
values from one instruction. Which lane owns which slice is chosen per
encoding so that aligned loads yield whole value sets, and alignment is a
fact of the format: 144- and 176-byte blocks allow 16-byte loads, 110-byte
blocks only 2-byte ones, and the Zig side refuses a specialized kernel
when the row offset disagrees. **Factor the scale out.** Accumulate `Σq·x`
and `Σx` and apply the group scale once. **Work in the packed byte
domain.** Assemble four codes with word-wide masks and convert with one
`uchar4 → float4` cast; that alone took Q5_K from 113 to 210 GB/s.
**Select, never branch.** Half the lanes in a Q4_K SIMD group decode
scales one way and half the other; an `if` makes every lane execute both
paths, a `?:` computes both once and picks.

Then the profiler said the question was wrong. A dispatch cannot be timed
from the CPU, so profile mode gives every dispatch its own encoder and lets
the GPU stamp encoder boundaries; the stamps turned out to be nanoseconds
on the command buffer's own timeline, established by a probe after a first
guess made kernels sum to 23 times their buffer. With real numbers the
encodings that differed by GB/s all cost 0.81 to 0.91 ns per 256-value
block. Q4_K was not slow; it had fewer bytes per block to show for the same
instruction work. The limiter was per-block work, and the bandwidth floor
sat at about 0.5 ns per block.

So five variants attacked the per-block work, and none reached the 5 %
bar. The one with strictly fewer instructions, decoding each group on one
lane and sharing by shuffle, was 12 % slower. Processing two blocks per
iteration to hide load latency was 35 % slower: every live value costs a
register, the register file is divided among the threadgroups a core keeps
resident, and resident *other* groups are how this GPU already hides
latency. Staging the input in threadgroup memory cost 7 to 9 %. Fast math
changed nothing and made Q5_K worse. Every one of those is written down
with its number so nobody measures it twice, and the kernels shipped
unchanged. The lesson is not that the kernel is optimal. It is that a
source-level count of operations does not predict the time, and the next
lever is a different geometry, not a shorter loop.

Read: `kernels.metal` (`nu_matvec_q4_k` and its neighbours),
`Backend.specializedMatvec`,
[reference/metal-backend.md § KERN-05](reference/metal-backend.md#kern-05--per-block-cost-research-2026-09-08-closed-without-a-kernel-change).

### 14. Fewer dispatches, same math

Several projections in a layer read the same normalized vector, and each
was its own dispatch. A segment table now describes their row ranges,
encodings, and offsets in one dispatch; a threadgroup selects its segment
and runs the standalone kernel's body. All lanes of a SIMD group pick the
same encoding, so there is a uniform branch and no divergence. The
feed-forward pair has a dependency, `silu(gate)·up`, and the version that
computed both projections in one SIMD group measured slower than the plain
merge; the kept version gives each projection its own SIMD groups and
exchanges sixteen floats through threadgroup memory at a barrier.

The same idea run past its limit is instructive. Fusing the residual add
with the norm that follows it, and the query and key norms with their
RoPE, removed 96 to 192 dispatches per decode step across the three
families. The speed bars were missed by all three: decode at 512 moved
by 0.0 to 0.5 %. The profile had said each small dispatch cost about 13 µs,
and that was true, but it was kernel work, not a launch floor, so merging
two memory-bound passes saved only the launch. The fused kernels shipped
for the dispatch count with the pairs kept behind a switch, and the record
says where the real win would be: the norm inside the kernel that produces
its input, an epilogue, not a merge.

Read: `Backend.matvecSegments`, `rmsNormAdd` and `rmsNormRope` in
`root.zig`, [reference/bench.md § Fused norm sweep](reference/bench.md#fused-norm-sweep-kern-18-2026-09-21).

### 15. Sampling on the device

Greedy decoding never needs the logits on the CPU: a two-pass argmax
returns one id and skips a 1 MB readback. Sampling did need them, because
temperature, top-k, and top-p want a distribution and the reference builds
it by sorting all 248,320 logits, 19 ms a token. The fix keeps one hard
promise: the token is bit-identical to the reference sampler's for the same
seed.

Only the sort needs every logit. The GPU returns the best 256 in the
sort's exact order, with the same tie rule, and the CPU redoes
exponentiation, the nucleus walk, and the draw on those values in the
reference's F64 order. Identical inputs and operations give identical bits.
When `top_k = 0` the nucleus walk compares against the whole vocabulary's
sum, which the GPU computes in F32 partials that differ from the F64 sum by
parts in 10⁸; so the sampler treats it as an interval and *defers* to the
full readback whenever a comparison lands inside the band, and the random
draw happens only after the decision, so a deferred token consumes no
randomness. The test perturbs the sum beyond its real error in both
directions and demands that the sampler either agrees or defers, never
disagrees. Selecting 256 of 248,320 is 256 rounds of "best untaken" per
threadgroup and a 64-way merge, cheaper than the copy it replaces; its
bound is register residency, 16 values per thread, which caps the
vocabulary at 262,144.

`min_p` looked like it needed a softmax and does not: after exponentiating
relative to the maximum, each weight is exactly `p/p_max`, so the
survivors are a prefix of the sorted list. And penalties looked like the
one thing that could never go to the device, because presence and
repetition adjust every logit of every token in the history before the
sort, and the readback happens after the projection. The model card's
instruct profile carries a presence penalty of 1.5, so it ran the full
path at 8.69 tokens per second against 10.60 greedy for a while, measured
and recorded as the honest cost of a default. The penalty kernel closed
that: the history is a bitset over the vocabulary, 31 KB, and applying it
on the device before the top-k puts the instruct profile within 2.5 % of
greedy.

Read: `inference/src/sampling/root.zig`, `nu_topk_partial` and
`nu_penalize` in `kernels.metal`,
[reference/generation.md § Sampling profiles](reference/generation.md#sampling-profiles-and-the-selection-chain-modl-01).

---

## Part Six — Prefill is a different problem

### 16. Arithmetic intensity flips

Before chunked prefill, nuclis read a 512-token prompt at decode speed,
because the prompt loop called `step` per token: 16 GB read 512 times.
Multiplied as a matrix, the weights are read once and every byte does 512
times the arithmetic, and the resource that runs out is the matrix unit,
not the bus. That is the entire reason the same model prefills at 90 and
decodes at 10.

`nu_matmul` decodes a slice of the weight matrix once into threadgroup
memory and multiplies it against a tile of tokens with `simdgroup_float8x8`,
so each weight byte is decoded once per 32 or 64 tokens. Then the tile
asked its own question: once the decode is amortized, what bounds it? The
guess was the decoder, because the tile's speed differed by encoding.
Measured one lever at a time, the specialized decoders collapsed the spread
and revealed a common floor; doubling the tile's edge to improve the
load-to-multiply ratio changed nothing, so the loads were not it either.

Half operands were worth 1.5×, and not for the reason expected. Loading the
decoded weights and activations as `half` and accumulating in F32 helped a
little through the ALU and a lot through memory: this GPU has a cliff in
how many threadgroups a core can hold as a function of threadgroup memory,
16 KB runs at full speed and 24 or 32 KB five times slower. In F32 a 64 × 64
tile needs 32 KB and collapses; in half it needs 16 and is the fastest
shape measured, 5.1 TFLOP/s. The operand width bought the tile shape. The
numerics contract moved with it and the record says by how much: chunked
prefill differs from the stepped path by about 2.5e-3 on the logits with
the same argmax, and the generic F32 tile stays for the rows whose scales
would overflow half.

Short prompts are a third regime. A 22-token prompt is one token tile, so a
5,120-row projection dispatches 80 threadgroups across 20 cores and streams
weights at 20 to 25 GB/s where the matvec streams 170. A 16-row × 8-token
split-K tile for chunks of at most 8 tokens took the short-prompt prefill
from 38.8 to 43.0 tokens per second and the first token from 567 to 511
ms; the same split-K idea applied to the *decode* matvecs later measured
behind the single pass at every split count and was closed with its
numbers. Threadgroup count is not what bounds those kernels.

Read: `nu_matmul_t` in `kernels.metal`, `Backend.matmul`,
[reference/metal-backend.md](reference/metal-backend.md) (the tile sweeps).

### 17. Attention in tiles: online softmax

A prefill chunk of `C` queries against a cache of thousands of keys would
need a `C × visible` score buffer per head if softmax were computed the
plain way. The identity that removes it: keep, per query row, the maximum
`m` seen so far, the sum `l` under that maximum, and the running output
`o = Σ exp(s_i − m)·v_i`; when a new key tile raises the maximum to `m'`,
every term so far is too large by `exp(m' − m)`, so scale `l` and `o` by
`exp(m − m')` and continue. The answer is `o/l` and no score is ever stored.
This is flash attention, written from the identity and checked against the
F64 reference that computes softmax the slow way.

On the matrix unit a SIMD group owns 8 query rows, accumulates score
blocks against 32-key tiles, writes them to threadgroup memory for the
row-wise mask, max, and exp on plain lanes, and sends the probabilities
back as the left operand of the value product into 32 register-resident
8 × 8 accumulators. Rescaling those by a per-row factor has no matrix
instruction, but a diagonal matrix does it as a multiply, so the kernel
builds one and applies it only when a row's maximum actually moved.
Causality is a bound on the tile loop plus `−∞` inside the last tile, and
the test puts `1e30` in every row past one query's horizon and demands the
clean answer. Result: 3.9e-7 against F64, tighter than the decode path,
and a prefill rate that no longer falls with prompt length.

A second attempt at this kernel for long contexts, keeping the key tile in
registers instead of threadgroup memory, measured 2 to 5 % ahead at
prefill chunk sizes and 11 to 16 % ahead on the 1 to 64-row batches that
speculation verifies, so it ships only for those. The finding underneath
was that F16 and F32 caches differ by 5 to 10 % in this kernel while it
sits flat at 640 to 750 GFLOP/s, so the long-context deficit is not a load
problem, and the next attempt would start from a different lever.

Read: `nu_attention_chunk_t` in `kernels.metal`,
[reference/bench.md § Prefill attention sweep](reference/bench.md#prefill-attention-sweep-kern-16-2026-09-21).

### 18. DeltaNet in chunks: the triangular solve

The recurrence `S_t = a_t·S_{t−1} + u_t·k_tᵀ` is sequential by definition:
token `t` needs the matrix after `t − 1`. Unroll it and the intermediate
matrices drop out: `S_t` is the starting matrix decayed by every decay so
far plus every earlier correction decayed by the decays after it. The
correction `u_t` was defined through the prediction `S_{t−1}·k_t`, which
seems to bring the chain back, except that substituting the unrolled form
makes every term `u_s·(k_s·k_t)`: a correction scaled by how similar two
keys are. For all `C` tokens at once that is `(I + A)·U = B` with `A`
strictly lower triangular, solved by forward substitution row by row. The
per-token matrix update became inner products between the chunk's keys,
which is the same move that makes attention parallel, and it is exactly
what `deltaChunk` does in F64.

The cost is `C²` inner products the sequential form never computed, cheap
at `C = 64` and ruinous for a whole prompt, so the chunk is fixed and the
state is carried between chunks exactly as between tokens. Decays are
handled as logarithms because a product of 64 factors below one and a
division by it are fine on paper and fragile in F32. Three checks in
increasing distance from the mathematics: bit-identical to a sequential
F64 loop after the final cast, 2.4e-7 from the production per-token step
whether run as one chunk or 40 then 24, and the reference's three pinned
steps reproduced as one chunk.

The kernel keeps nothing large in threadgroup memory: a head's 128 × 128
state is 64 KB against a 32 KB limit, so four threadgroups each take 32
state rows (given the chunk's keys and decays, each value row of the state
evolves independently) and stream them through the matrix unit twice per
sub-chunk. The only sequential part, the substitution, is 32 rows long
with one thread per column. Outputs within 3.4e-8 of the F64 chunk, the
carried state within 1.2e-7.

Read: `inference/src/backends/cpu/recurrent.zig` (`deltaChunk`),
`nu_delta_chunk` in `kernels.metal`,
[reference/cpu-reference.md](reference/cpu-reference.md).

### 19. Experts: sparsity's gift and bill

Gemma 4 26B-A4B stores 26 billion parameters and uses about 4 per token:
each layer's feed-forward is 128 small experts of width 704 and a router
that picks 8. At decode this is pure gain, 2.1 GB read per token at Q4_0
against 16 for the dense 27B on the same bus, and the model decodes at
about 58 tokens per second.

The router is a softmax you never finish. Selection compares logits, not
probabilities, because softmax is monotone but `exp` rounds and a tie at
the top (the fixture has three) could swap under a different rounding;
lowest index breaks ties, so the expert ids are exact bit for bit and only
the weights carry rounding. The renormalizing sum is floored at the
smallest F16 normal because the reference floors it, and a floor from the
reference is part of the contract. A 3-D expert tensor is a row of
matrices, so the gathered decode matvec is the segment kernel with the
weight pointer offset by `expert·rows·stride`; unselected experts are never
touched, and the test proves "never" by writing NaN into every scale of
them. The expert index comes from a buffer the GPU wrote a moment earlier,
so the kernel clamps it: a corrupt route gives a wrong answer the
comparison catches, not a read past the tensor that nothing would.

Prefill is where the bill comes. A dense tile reads each weight once per
32 or 64 tokens; in a sparse layer the tokens that want expert 17 are
scattered through the chunk. So the rows are sorted by expert on the GPU,
a histogram with threadgroup atomics, a prefix sum, and a scatter, and the
gathered tile is the dense tile with two indirections. The grid must be
sized when it is recorded, before the counts exist, so the host dispatches
a bound and tiles past the real count exit at their first instruction.
Per executed multiply the gathered tile *is* the dense tile, 4.8 against
5.0 TFLOP/s, but at 256 tokens spread over 128 experts each tile is half
zeros. The fill rises with the chunk, 50 % at 256 tokens to 81 % at 1,024,
and the chunk size becomes a lever it never was for a dense model.
Sparsity divides the rows that share a matrix by `experts/k`, sixteen
here, so a sparse model's prefill is compute-bound at a lower fill than a
dense one's.

Read: `inference/src/backends/cpu/experts.zig`, `nu_route`,
`nu_expert_lists`, `nu_matmul_experts_t` in `kernels.metal`,
[reference/metal-backend.md § Gathered expert kernels](reference/metal-backend.md#gathered-expert-kernels-kern-09).

---

## Part Seven — Long context

### 20. Half the bytes: the F16 cache

Every attention layer appends one key row and one value row per token and
every later token reads them all back: 4 GiB per decoded token at 32K on
Qwen with F32 rows, and 4 GiB held. Storing the rows as F16 halves both.
The interesting question was never whether, but exactly which numbers get
rounded and where.

The rounding happens once, at the write: the GPU computes keys and values
in F32 as before and packs the row on its way into the cache, nearest
even, bit-identical to Zig's `@floatCast`. Decode converts each half back
to F32 as it loads it, so the cache is the only 16-bit thing. Prefill
multiplies tiles on the matrix unit, which takes both operands in one
type, and Metal has no conversion from a half tile to a float tile; so the
half chunk kernel packs its queries to half as well and rounds its
probability tile before the value product, computing the normalizer from
the rounded values so the weights still sum to what the products used.

How to measure that honestly: compare the kernel against the CPU *over the
rounded values* to isolate its own error (1.1e-8, the same as F32), and
against the *original* values to see the rounding's cost, which is a
property of the data (1.2e-5 absolute). At the model level the residual
stream has outlier channels in the hundreds, so a 2e-4 relative change in
an attention output shows up as 2.5e-2 in deep layers while the greedy
token does not move. Loosening the one global threshold would hide
regressions in the F32 path, so F16 gets its own tolerance (3e-2 on layer
files) and the gates run both. A documented tolerance per numerical mode,
never one number loosened to cover them all.

Gemma showed the same rounding cost thirty times more on its K-quant
file: its attention scores are unscaled, so a 2⁻¹¹ rounding of a key
moves a score by hundredths and a softmax weight by percents, compounding
through 48 layers on a short prompt with a BOS sink. Two experiments
separated kernel error from model sensitivity before the number was
written down: the same half kernels against the CPU over rounded operands
(2e-4: the kernels are right), and the CPU reference itself with keys
rounded to F16 before the write (0.75: the same deviation with no GPU in
the loop). The QAT file, incidentally, was tighter than the K-quant on
every path. `--kv f32` exists for numerical work.

Read: `nu_pack_half`, the `_h` instantiations in `kernels.metal`,
[reference/session.md](reference/session.md),
[reference/gemma4.md](reference/gemma4.md).

### 21. Flash decoding

Prefill attention tiles queries against keys. Decode has one query per
head and tens of thousands of keys: nothing to tile on the query side, and
the whole cost is streaming the cache once. The bring-up kernels streamed
it far more than once, one threadgroup per (query head, position), so each
key row was fetched six times for the six heads that share it; a token at
32K spent 280 of its 377 ms in attention.

Three rules fixed it. Read each row once for everything that needs it: a
threadgroup owns one KV head and all the query heads sharing it, so the
GQA structure of the model becomes a memory-traffic argument and the cache
is read once per token, 2 GB at 32K with F16 rows. Keep the softmax state
in registers and split the rows across the GPU: with lane `l` holding
channels `l, l + 32, …` the running max, sum, and accumulator for six heads
fit in registers, and the visible range is cut into up to 64 slices of one
threadgroup each, 256 threadgroups instead of 24 serial ones. Merge
partials with the log-sum-exp identity: two softmaxes over disjoint key
sets combine exactly by rescaling each to the joint maximum, and an empty
slice is a partial with max `−∞` that the identity handles with no special
case. Against F64 the result is within 2e-8 at 32,000 rows, closer than
the old kernel because each product is accumulated once instead of being
rounded into a stored score. Decode at 32K went from 2.65 to 8.09 tokens
per second, and the widest Gemma head became a second instantiation of the
same template rather than a second kernel, because register budgets are
why GPU code has instantiations where CPU code has a loop bound.

Read: `nu_attention_decode_t`, `nu_attention_merge` in `kernels.metal`,
`Backend.attentionDecode`.

### 22. Checkpoints, not rewinds

A pure-attention engine undoes the last `k` tokens by setting a length.
This model cannot, because 48 layers have folded every token into a matrix
that keeps no history. Set the attention length back and the recurrent
layers still remember; the model would attend to one past and recur over
another, silently. What exists instead is a copy: the used extent of the
session, restored behind a digest, never while a step is in flight.

That constraint reaches into the conversation. The agent never edits the
session's past; it appends, or resets and replays. Appending works because
the chat template is prefix-stable: rendering `[u1, a1, u2]` yields the
text for `[u1, a1]` plus a suffix, so the agent keeps the exact text the
session has consumed and prefills only the increment. Two details decide
whether the prefix matches: reasoning must travel through its own field,
because the template renders past thoughts differently from how the model
emitted them live, and the last sampled token is never fed back, so the
consumed text excludes it. When the effort changes the system block
differs and the comparison fails honestly; the bar says *replayed*.

The system prompt and the tool definitions are the same for every turn, so
the engine prefills them once at startup, records them as consumed, and
keeps a snapshot; a new session or a resume restores the snapshot rather
than prefilling again. The wait a user felt after the first Enter was that
prefill, about eleven seconds for 934 tokens, and it now happens before
the prompt appears.

Read: `Session.snapshot`, `restore`, `checkpoint`, `rewind`, `truncate` in
`session.zig`, `src/agent/loop.zig` (`increment`, `Completer.prime`),
[reference/session.md](reference/session.md).

---

## Part Eight — Speculation

### 23. Guessing ahead, and paying to check

Decode reads 16 GB to produce one token. If something cheap could guess
the next four, the main model could score all five in one batched forward
(one more weight read, five rows of activations) and keep the prefix it
agrees with. That is speculative decoding, and the whole tree's memory
model, verification path, and sampling rule had to bend to make it exact.

**Three ways to guess.** Each family names its own draft source behind one
contract, `propose(token, out)` and `commit(tokens, hidden)`: the Qwen
release ships a 65th block that predicts the next token from the main
model's hidden state, run as a chain; Gemma has a companion file of
assistant heads that read the target's own caches; Muse has a DFlash
drafter that proposes a block at once. The drafter's cache is one more
layout in the session block, so snapshots and checkpoints cover it for
free. The Qwen block's chain accepts 90/80/69/64 % of positions at depths
0 to 3 on the pinned prompt, and a `p_min` early stop ends a chain whose
top candidate falls below 0.7.

**One batch, two recoveries.** The verify batch feeds the last chosen
token and the `k` drafts through the prefill path with every row's logits
kept. Attention rewinds by position: rows past the accepted prefix are
never read, so nothing is copied. Recurrent state cannot, so it is
checkpointed before the batch and, on partial acceptance, restored. The
first recovery replayed the accepted prefix through the whole stack, 150
to 182 ms; per-row recurrent checkpoints inside the batch made it a slot
copy, 6 to 22 ms. Never rewind DeltaNet by truncating the position alone.

**Accepting without cheating.** Greedy acceptance compares each draft with
its row's argmax. Sampled acceptance draws the target's own token from the
row's shaped distribution, with penalties and history advanced through the
earlier drafts of the batch, and accepts the draft when the draw equals
it; every emitted token is a target draw whatever proposed it, which makes
the rule exact for greedy chains. The textbook `min(1, p/q)` rejection
rule is exact only for drafts sampled from `q` and was replaced for that
reason. The draws happen on the device's per-row top-k readback, so
`accept` fell from 63 to 91 ms per batch to 19 to 37 µs.

**What the record said.** The costs per batch on Qwen at 512 tokens, after
every lever: propose 10 to 23 ms, checkpoint 3, verify 218 to 229, recover
6 to 11, commit 4 to 10. The verify is the batch, and it is row-flat: 3 to
8 rows through the small-batch tiles cost nearly what 1 row costs through
the matvecs, so the speedup is bounded by how many drafts are accepted per
batch, 1.1 to 2.5 here. The verdict: code greedy 1.20 to 1.30× at drafts 4
to 7, prose 0.81 to 0.97×, 4K context 0.73×; the switch stays off for Qwen.
Gemma's heads accept 2.26 per batch but its verify is 136 ms at 3 to 8
rows, 0.899× at draft 4; off. Muse's DFlash drafter accepts 73 to 77 % of
positions and the pair runs 1.16 to 1.23×; on. Each catalogue entry carries
its own verdict and `bench` opens with no drafter at all when the switch is
off, so the baseline is true.

Every kernel lever the plan ordered for the verify closed at or below its
target with its numbers on record: a multi-row matvec that wins only at 2
rows, a wider small-batch tile, split-K, register-reuse attention, fused
norms. The honest summary is that speculation is exact on all three
families, a speedup on one, and the remaining cost is a small-batch matrix
product this GPU does not do well at 3 to 8 rows.

Read: `inference/src/runtime/draft.zig`, `speculativeBatch` in `engine.zig`,
`models/dflash.zig`, `models/gemma4_assistant.zig`,
[reference/speculative-decoding.md](reference/speculative-decoding.md),
[reference/bench.md § The speculative verdict record](reference/bench.md#the-speculative-verdict-record-engn-17-2026-09-21).

---

## Part Nine — Families

### 24. The registry table

Before the second model, `Engine.open` called the Qwen binder, held a Qwen
binding, and its union named the Qwen runtime and plan; the executable
named the Qwen profile in seven files. A second architecture would have
meant editing all of them, which is exactly what the spec's extension rule
forbids: a family must cost its adapter, its profile, its tests, and any
genuinely new mathematics, and nothing else.

An adapter now publishes a namespace with the names the engine needs
(`architecture`, `executableEncoding`, `Binding` and `bind`, `Runtime`,
`Plan`), and one list in `models/root.zig` is the registration. The
`Adapter` enum is built *from the list* with `@Enum`, the executor union
with `@Union`, so a tag exists exactly when a family does and the two
cannot disagree; dispatch is an `inline else` on the tag. Profiles register
the same way and are selected by template digest. A test over two stub
families proves the derivation with no reference to Qwen. Nothing under
`src/` names an adapter or a profile module any more.

Read: `inference/src/models/registry.zig`, `models/root.zig`,
`profiles/root.zig`, [architecture.md § 10](architecture.md#10-adding-a-model).

### 25. What a second architecture costs

Gemma 4 12B on the CPU reused the norm, the decoders, the matvec, the
attention, and RoPE unchanged. Two things were new mathematics with their
own fixtures: a tanh GELU, and RoPE frequency *factors*, a divisor per
rotated pair, stored as 64 ones and 192 values of 1e30 so three quarters of
a global head's pairs get an angle that underflows to nothing. That is how
"rotate the first quarter of the dimensions" is expressed by the file with
no second code path. The sliding window needed no mask on the CPU: the
visible rows of a windowed layer are a contiguous suffix of the cache, so
the runtime slices the view and calls the same function. Every fact came
from the file, the upstream configuration, or the pinned reference source,
each with its provenance written down, and the first trace comparison
agreed to a relative RMS of 8e-6 on every layer.

On the GPU the reuse is tested by shapes. The window became a parameter on
the chunk attention kernel and a cache slice at decode. The 512-wide heads
exceeded the accumulator budget, so the grid gained an axis (one
threadgroup per 256 value columns, redundant scores, no new kernel) and
the flash-decoding kernel became a template instantiated twice. Four
scalar epilogues and a GELU pair mode were genuinely new, a dozen lines
each. And one lesson no reference contained: the first Gemma step produced
NaN in nine of 15,360 gate values, because Metal's `tanh` goes through
`exp` and overflows past about ±44 where the CPU's saturates. F32 tanh is
exactly ±1 from ±20 on, so a clamp there changes no finite result. A
per-layer probe found it in one run; the trace alone would only have said
"layer 0 is wrong".

The expert configuration was not a second adapter: the same family pins
two configurations selected by block count, and one runtime and one plan
run either with the expert layer as a branch. Muse Glimmer brought a
second tokenizer splitter, windowed attention with global layers, a
reasoning channel, and its own tool grammar, and no new kernel. Bonsai
brought two encodings, a transform, and a permutation contract for the
value heads, on Qwen's schedule. The count that matters is what each did
not touch: the parser, the sampler, the session, the bridge, the existing
kernels, the generation loop.

Read: [reference/gemma4.md](reference/gemma4.md),
[reference/muse-glimmer.md](reference/muse-glimmer.md),
[reference/new-model-guide.md](reference/new-model-guide.md).

---

## Part Ten — Honesty

### 26. Measuring, and what counts as knowing

Fluent output is not evidence. The oracle chain runs from the pinned
llama.cpp build, used as a reference and never linked, through fixtures
(decoders must match exactly; F32 GPU reductions match F64 sums within
stated bounds), to per-layer traces of the real model on real prompts,
with thresholds written down and passes recorded with dates and observed
maxima. Every model-specific check is a gate in `gates.json`, tiered by
cost and selected by the paths a change touched; every benchmark is a
workload in `workloads.json` with its report saved by revision. The record
is data, and the tables in the documents are generated from it.

Comparing two engines needed three things to be true. The same *tokens*,
not the same text: the reference harness concatenated token arrays that no
text tokenizes to, so `bench` takes token arrays and the acceptance runs
use the committed ones. Separate clocks with aligned definitions: decode is
`(generated − 1)/seconds`, the intervals after the first token, on both
sides. And an honest meaning of memory: weights are mapped, so resident
memory is what the run touched, reported beside the session block and
with swap counters sampled before and after so a run that leaned on swap
cannot pass silently.

The habit that makes negative results useful is writing them down with
their numbers next to the positive ones. The per-block matvec study, the
wide tile, split-K, the register-reuse attention, the fused norms, and two
speculative defaults all closed below their targets, and each closed with
a table that says what was measured and what it rules out. A plan that
starts from those tables asks a different question than one that starts
from a hunch.

Read: [reference/bench.md](reference/bench.md),
[development.md § Gates](development.md#gates),
[engineering-log.md](engineering-log.md).

---

## 27. What it took

Twelve things, in the order they became true.

1. The bus sets the ceiling. Bytes per token is the first number to
   compute and the last to forget: 16 GB at 273 GB/s is 17 tokens per
   second before a line of code.
2. Never expand the weights. Decode on the way into the multiply, on both
   processors, bit-identical between them.
3. Refuse before you read. A directory that validates before the first
   weight byte, and a verdict that names its offender.
4. A tokenizer is a merge order plus the rules for where merges are
   legal, pinned against the reference on a corpus, not a sentence.
5. The template is a contract, selected by its digest, owning its stop set,
   its reasoning markers, and its tool grammar.
6. Write the math twice. A slow F64 reference that is obviously right,
   and a fast schedule measured against it at every layer.
7. Session state is not a cache. Recurrent layers remember everything
   and rewind nothing; copy, never truncate.
8. Synchronization is a cost like bandwidth. One command buffer per token,
   no hidden waits, a tick that costs nothing.
9. Kernels are geometry. Vector loads that match the block, scales
   factored out, selects not branches; and when the profiler speaks,
   believe it over the instruction count.
10. Prefill is a different problem. Tiles, half operands, online softmax,
    a triangular solve, and a sort by expert; the resource that runs out
    changes, and so does the kernel.
11. Half the bytes at a documented cost. F16 rows with a tolerance per
    mode, flash decoding that reads the cache once.
12. Guess ahead, but exactly. A batch, two recoveries, a target draw per
    row, and a switch set per family by its own record.

And one more that is not a technique: every number in this guide is in a
table somewhere with a date, a revision, and the command that produced
it. That is what made it possible to write the guide at all.
