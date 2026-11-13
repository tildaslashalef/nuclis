<!-- kind: reference -->
# Metal kernels

The compute kernels the [Metal backend](metal-backend.md) records: how each
maps onto the GPU, what it computes, and the contract its typed encoder in
`backends/metal/ops/` enforces before recording. Every kernel binds data
buffers at indices 0–6 and a small parameter struct at 7; shapes come from
parameters, nothing is hard-coded to one model. The numerical checks are in
[metal-evidence.md](metal-evidence.md); how each kernel was built and
measured, including the variants that lost, is in
[the record](../benchmarks/record/metal-backend.md).

## Kernel geometry

How each kernel maps onto threadgroups and SIMD groups (32 lanes), and
which cross-lane operations it relies on. The concepts are explained in
[llm-guide.md § 12](../llm-guide.md#12-lanes-simd-groups-and-where-they-live).
Thread counts are the numbers `Backend` passes to `dispatch`.

| Kernel | Threads / group | One threadgroup owns | One SIMD group owns | Cross-lane ops | Threadgroup memory |
| --- | ---: | --- | --- | --- | --- |
| `nu_matvec` (generic) | 32 | one output row | the row: lanes stride 16-value segments | `simd_sum` | none |
| `nu_matvec_*` (specialized) | 128 | 16 output rows | 4 rows; 8 lanes per 256-value block | `simd_sum` | none (IQ4_XS: 64 B table) |
| `nu_matvec_segments` plain / pair | 128 | 16 rows of one segment / 8 gate+up pairs | 4 rows | `simd_sum`; pairs share 16 floats | 64 B (pair mode) |
| `nu_matvec_experts` | 128 | 16 rows of one slot's expert (the segment bodies) | 4 rows | `simd_sum` | none |
| `nu_route` | 256 | one row of ≤ 256 router logits | 32 logits; k rounds of "best untaken" | `simd_max`, `simd_sum`, `simd_shuffle_down`, 8-way threadgroup pick | 8 + 8 + 64 entries |
| `nu_matmul_*` (specialized) | 128 | 64-row × 64-token output tile (32×32 for chunks of ≤ 32 tokens) | a 32×32 quarter as 4×4 `simdgroup_float8x8` (2×2 in the small tile) | matrix loads and MACs | 8 KB half weight tile + 8 KB half activation tile (small tile: 4 KB, activations from device), `threadgroup_barrier` |
| `nu_matmul_*_8` | 128 | 16-row × 8-token output tile (two token tiles for 9..16) | 16 rows × 8 tokens over one K slice: two 8×8 accumulators sharing one B load, four groups split K | matrix loads and MACs | 8 KB half tile (16 rows × 64 k per group), `simdgroup_barrier` only |
| `nu_matmul_*_f2` | 128 | 16-row × 8-token output tile (token tiles as `_8`) | 16 rows × 8 tokens over one K slice: each lane decodes one 16-value segment per row straight into its `simdgroup_matrix` elements (K permuted per 64-column step), two float4 activation loads per token | matrix MACs; four K partials summed in a fixed order | 2 KB of K partials, one `threadgroup_barrier` |
| `nu_matvec_rows_*_t<n>` | 128 | 32 output rows × up to 8 tokens | 8 rows: each 8-lane group owns 2 and the four walk the same block; one accumulator per (row, token) | `simd_shuffle_xor` across 8 lanes per (row, token) | 64 B IQ4 table (IQ4_XS), one `threadgroup_barrier` |
| `nu_matmul` (generic) | 128 | 32-row × 32-token output tile | a 16×16 quarter as 2×2 `simdgroup_float8x8` | matrix loads and MACs | 8 KB F32 weight tile + 8 KB F32 activation tile, `threadgroup_barrier` |
| `nu_attention_chunk` / `_h` | 128 | (query head, 32-query tile, 256 value columns) | 8 query rows: 4 score blocks, 32 output blocks | `simd_shuffle_xor`, `simd_shuffle`, `simd_any`, matrix MACs | 6 KB (per-group score tile, diagonal, staging); 7.5 KB in the half instantiation (its own probability tile), `simdgroup_barrier` only |
| `nu_attention_chunk_reuse` / `_h` | 128 | (query head, 32-query tile) | all 32 query rows × 64 value columns; scores for its own 8 rows | matrix MACs; probabilities published through threadgroup memory | double-buffered probability tile and rescale row, one `threadgroup_barrier` per 32-key tile |
| `nu_segments_scores` | 128 | (query tile of ≤ 32 rows, head, 32-key block) | a 16×16 quarter as 2×2 `simdgroup_float8x8` | matrix loads and MACs | 8 KB Q tile + 8 KB transposed K tile (F32), `threadgroup_barrier` |
| `nu_segments_stats` | 256 | (query tile, head) | one strip row at a time, an online (max, sum) per lane | `simd_max`, `simd_sum` | none |
| `nu_segments_values` | 128 | (query tile, head, 32 value columns) | a 16×16 quarter as 2×2 `simdgroup_float8x8` | matrix loads and MACs | 4 KB P tile + 4 KB V tile (F32), `threadgroup_barrier` |
| `nu_delta_chunk` | 128 | (value head, 32 value rows), all sub-chunks | an 8-row block of every 32×32 tile; column `tid` in the triangular solve | matrix MACs; no shuffles | 24 KB of 32×32 tiles, `threadgroup_barrier` per phase, `mem_device` per sub-chunk |
| `nu_attention_decode` / `_h`, `_w` / `_wh`, `_q8` / `_wq8` | 128 | (KV head, group of ≤ 8 query heads — ≤ 4 in the wide pair —, split of the visible rows) | every fourth row of the slice; lane l owns channels l, l+32, … (8 per lane, 16 in the wide pair) with per-head running max, sum, and accumulator in registers | `simd_sum` per (row, head); 3-round merge of the SIMD groups through 8 KB | 8 KB stage, `threadgroup_barrier` |
| `nu_attention_heads_*` | `HG × RL` SIMD groups, ≤ 1,024 threads | (KV head, split, row) | GROUP query heads, rows `start + s / HG` step RL; lane l owns channels `CH·l … CH·l + CH − 1` | `simd_sum` per (row, head); merge through the 4,096-float stage | 16 KB stage |
| `nu_attention_merge` | 256 | one query head | — (threads stride the value channels, log-sum-exp over ≤ 64 splits) | none | none |
| `nu_attention_scores` / `_values` (and `_h`; the three-pass decode, kept as `test-metal`'s oracle) | 32 | one (head, position) / (head, channel) | the dot product / weighted sum | `simd_sum` | none |
| `nu_attention_softmax` | 32 | one head's score row | the whole row | `simd_max`, `simd_sum` | none |
| `nu_delta` | 32 | one (value head, coordinate) | the 128-wide contraction | `simd_sum` | none |
| `nu_l2norm` | 32 | one head vector | the norm | `simd_sum` | none |
| `nu_shared_expert_add` | 256 | one row | a strided slice of the gate dot | `simd_sum` then 8-way combine | 8 floats |
| `nu_rmsnorm` (and the fused norms) | 256 | one row | a 32-value strip | `simd_sum` then 8-way combine | 8 floats |
| `nu_hadamard` | 256 | one 1,024-block of one row | 128 values, four per thread | none: two butterfly stages in registers, eight through threadgroup memory | 4 KB, a barrier per stage |
| `nu_argmax_partial`, `nu_topk_partial`, `nu_expsum_partial` | 256 | a strided slice of the vocabulary | its 32 values | `simd_shuffle_down` reduction, 8-way threadgroup pick | 8–16 entries |
| `nu_argmax_final` | 32 | the partials | all of them | `simd_shuffle_down` | none |
| `nu_topk_final` | 256 | the 64 sorted lists | cursors | k-way merge on plain threads | list cursors |
| elementwise (`nu_add`, `nu_silu_*`, `nu_gelu_mul`, `nu_gelu_mul_rows`, `nu_combine_experts`, `nu_scale`, `nu_add_scale`, `nu_softcap`, `nu_rope*`, `nu_convolution*`, `nu_copy`, `nu_pack_half`, `nu_unpack_q8_0`, `nu_sigmoid_gate`, `nu_delta_gates`, `nu_embed`, `nu_gather_rows`, `nu_penalize`) | 256 | 256 elements | 32 elements, one per lane | none | none |

Three patterns cover the table: a *reduction* kernel gives a SIMD group one
output and reduces with `simd_sum`; a *tile* kernel gives a threadgroup a
tile of memory that its SIMD groups multiply with the matrix unit; an
*elementwise* kernel uses threads as independent workers and never
communicates. The choice per kernel follows one rule: put in a SIMD group
what must be reduced or exchanged, put in a threadgroup what must share
memory, and leave the rest to the grid. Threadgroup memory sets occupancy,
with a cliff between 16 and 24 KB per group
([apple-gpu.md](apple-gpu.md#the-device)).

## Matrix-vector products

`nu_matvec` is the generic kernel: one SIMD group per output row, each lane
decoding 16-value segments straight from quantized bytes through the
runtime encoding switch. It serves F32, F16, BF16, Q8_0, IQ4_NL, and any
misaligned range. `Backend.matvec` records a specialized kernel instead
when `specializedMatvec` accepts the encoding and the row start, row
stride, and input are aligned for its vector loads.

### Specialized matvec

Decode reads every weight once per token, so these kernels decide decode
speed. They exist for Q4_K, Q5_K, Q6_K, Q3_K, Q2_K, IQ4_XS, IQ3_S,
IQ3_XXS, IQ2_XXS, IQ2_XS, IQ2_S, IQ1_S, IQ1_M, Q4_0, Q5_0, Q5_1, PQ2_0,
and PTQ1_0, as templates instantiated with `[[host_name]]`, so the bridge
needs no function constants.

- **Geometry.** 128-thread groups of four SIMD groups; each SIMD group owns
  four consecutive output rows (`Backend.rows_per_simdgroup`) and walks their
  256-value blocks four at a time, eight lanes per block. A lane decodes 32
  values of its block from one to three vector loads and reads its 32 inputs
  as eight `float4` from device memory, shared by the four rows. Q4_0 walks
  32-value blocks instead (lane `l` takes blocks `l`,
  `l + 32`, …), so any whole-block row serves, including a 22-block expert
  row; the ternary encodings give four lanes to each 128-value block.
- **Factored scales.** Per 32-value group a lane accumulates `Σq·x` and `Σx`
  and applies the group's scale and minimum once:
  `Σ(d·s·q − dmin·m)·x = d·s·Σ(q·x) − dmin·m·Σx`. With a one-hot input this is
  the CPU decoder's expression exactly, which keeps the pinned fixture columns
  exact; dense inputs differ only by F32 rounding order. Biased codes (Q6_K's
  32, Q4_0's 8, Q3_K's 4, the ternary codes' 1) fold into `Σx` the same way.
- **Packed-byte decoding.** Codes are assembled four at a time in the packed
  byte domain (`(v & 0x0f0f0f0f) | (((h >> bit) & 0x01010101) << 4)` completes
  four Q5_K values) and converted with one `uchar4 → float4` cast; the GPU is
  scalar per lane, so value-by-value construction costs several times more.
- **Half magic-number decode.** Q4_K and Q5_K codes are OR'd into the
  mantissas of halves of exponent 1024 (two codes per `half2`), minus 1024:
  exact, and on the FP16 pipe instead of an integer-to-float conversion per
  value; the group's scale carries the 1/16 of a high nibble (`nu_magic`,
  `nu_magic_nibbles`, `nu_magic_fives`). The codebook encodings (IQ2_XXS,
  IQ2_XS, IQ2_S, IQ3_XXS, `matvec_codebook.metal`) mask grid bytes into
  F16 subnormals the same way and XOR their signs into the half sign bits.
  IQ4_XS reads its 16-entry table from a threadgroup copy (`nu_iq4_table`).
  These kernels are issue-bound on the integer and complex pipe, which is
  why the conversions were moved off it
  ([the counters](../benchmarks/record/apple-gpu.md#nu_matvec_q4_k-on-qwens-ffn_down-shape-2026-09-30)).
- **Alignment rules.** Vector-load widths follow block sizes: Q4_K (144 B) and
  Q5_K (176 B) use `uint4`, IQ4_XS (136 B) `uint2`, and Q3_K/IQ3_S
  (110 B), Q6_K (210 B), and Q4_0 (18 B) use `packed_ushort4`.
  `specializedMatvec` requires the row offset and stride to be multiples of
  the load width and the input to be float4-aligned; otherwise the generic
  kernel is recorded. Every tensor of the pinned artifacts qualifies.
- Scale decoding for Q4_K/Q5_K group pairs is written with selects: half of
  each SIMD group's lanes need the "direct" form and half the "split" form, so
  a branch would execute both.

`make bench-kernels` (`metal-check --matvec-bench`) measures each kernel
alone on fixture-tiled matrices, eight back-to-back dispatches per command
buffer (64 under 8,192 rows: an isolated dispatch measures the GPU clock's
ramp-up), best and mean of five command buffers. The kernels reach 150–258
GB/s of the published 273; their tables and the variants that did not
pay (two blocks per lane, staged inputs, two or eight rows per SIMD group,
hoisted scale decode, fast math, split-K) are in the record:
[specialized matvec](../benchmarks/record/metal-backend.md#specialized-matvec-2026-09-07),
[per-block cost research](../benchmarks/record/metal-backend.md#per-block-cost-research-2026-09-08-closed-without-a-kernel-change),
[split-K](../benchmarks/record/metal-backend.md#split-k-2026-09-21-closed-negative).
The split-K matvecs (`kernels/matvec_split_k.metal`, `Backend.matvecSplits`
and `matvecSegmentsSplits`) compile only into the `bench_kernels` build
`metal-check` imports; the application neither embeds nor compiles them.

### Merged projections

`Backend.matvecSegments` accepts up to four borrowed matrix/output
descriptors sharing one input. Each segment has a row count, encoding,
stride, and weight and output byte offsets. All rows must be multiples of
sixteen, columns must match, output ranges must fit and be disjoint from
reads, and the group must fit seven data bindings. Validation finishes
before dispatch, so the adapter can fall back to standalone matvecs on
`InvalidShape` without replaying partially recorded work. Input uses
binding 0; bindings 1–6 hold the distinct weight/output buffers, and slices
of one buffer share a binding. DeltaNet packs `mixed|z|beta|alpha` into one
activation allocation; attention writes directly to its query/gate buffer
and the current key/value session slots.

Plain mode assigns each sixteen-row threadgroup to one segment, with the
same lane partials as the standalone kernels (a generic body keeps the
standalone traversal for other encodings or unaligned weights, marked by an
internal high bit in the encoding field). Pair mode computes eight gate/up
output rows per threadgroup: two SIMD groups compute gate and two up, share
sixteen reduced floats through threadgroup memory, and apply `silu(gate) ·
up` (mode 1) or `gelu(gate) · up` (mode 2) after one barrier, so no
intermediate FFN vector is written. The profile books it as
`matvec_segments`, with the sum of the segments' rows and bytes. A merge holding
one of the seven 2-bit encodings' specialized kernels (IQ2_XXS, IQ2_XS,
IQ2_S, IQ3_XXS, Q2_K, IQ1_S, IQ1_M; `mergeDeclined`) returns
`error.MergeDeclined` with nothing recorded, because those standalone
kernels beat their merged bodies; the adapter then runs the segments
standalone ([the counters](../benchmarks/record/apple-gpu.md#the-merged-gateup-on-salukis-encodings-2026-10-15)).

### Multi-row matvec

A batch of 2 to 8 activation rows (a verify batch, a drafter's commit, a
recovery replay) reads its weights once through `nu_matvec_rows_*`
(`Backend.matvecRows`): each 8-lane group of a SIMD group owns two rows and
the four groups walk the same 256-value block, so one input load serves
eight rows. Per 4-byte word, each row's eight codes are dequantized once
with scale and bias folded in (`fma(q, d·s, −dmin·m)`), then each token's
eight inputs are loaded once and dotted into both rows. The token count is
a template parameter, one host name per encoding and count
(`nu_matvec_rows_q4_k_t2` … `_t8`): a run-time token loop would move the
accumulators to thread-local memory. Q4_K, Q5_K, Q6_K, IQ4_XS, and IQ2_XXS
have specialized bodies; `nu_matvec_rows` (one SIMD group per row) serves
the rest. `Backend.matmul` routes here by `usesMatvecRows`: 2–3 rows for
Q4_K, Q5_K, and Q6_K (`small_batch_rows`), 2 rows for the other specialized
bodies; `matvec_rows_max` (8) is the kernel range `metal-check` sweeps.
Past those counts the fragment tile below wins
([record](../benchmarks/record/metal-backend.md#the-word-outer-body-2026-10-03)).

### Ternary matvecs and tiles

Bonsai 2 27B's PQ2_0 (id 142, 34 B per 128 values) and PTQ1_0 (id 143, 28 B;
[bonsai.md](../models/bonsai.md#encodings-from-the-forks-ggml-commonh-and-ggml-quantsc))
have generic decoders (`nu_dequant_pq2_0`, `nu_dequant_ptq1_0`, and
`nu_dequant_bf16` for the file's `ssm_alpha` / `ssm_beta` rows),
specialized matvecs, and `nu_tile_*` decoders for the batched tiles. The
matvecs give four lanes to each 128-value block, eight blocks per
iteration, every lane running the same code on its quarter so PTQ1_0's
digit-major layout costs no divergence. `w = d · t` factors as
`d · (Σ t·x − Σ x)` with the raw code as `t`. Two-bit fields are masked
four bytes at once (`(w >> 2k) & 0x03030303`) and paired with the inputs in
that strided order; trits come out of 16-bit slots two bytes at a time.
These kernels run at the set's ceiling in values per second, which halves
the byte rate
([record](../benchmarks/record/metal-backend.md#ternary-matvecs-and-tiles-2026-09-18)).

## Batched products

`Backend.matmul` computes `out[t][r] = Σ W[r][k]·X[t][k]` for a chunk of
tokens and selects by encoding, weight alignment (the matvec rules), and
chunk length (`specializedMatmul`, `specializedMatmulFrag`,
`matmulGeometry`):

| Tokens | Product | Operands |
| --- | --- | --- |
| 2 to 3 (by encoding, above) | the multi-row matvec | F32 |
| 2 to 8, no specialized tile, ≤ `small_matrix_rows` (1,024) rows | the generic multi-row matvec | F32 |
| ≤ `small_chunk_tokens` (24) | `nu_matmul_*_f2`, the register-fragment 16×8 tile | half weights, F32 activations |
| ≤ 32 | `nu_matmul_*_32`, 32×32, activations from device | half weights, F32 activations |
| above | `nu_matmul_*`, 64×64, both operands staged | half, F32 accumulation |
| any, no specialized tile | `nu_matmul`, 32×32 | F32 |

The specialized tiles decode a segment with the vector loads and
packed-byte helpers of the specialized matvecs (`nu_tile_*`), evaluating the
generic decoder's expression in the same F32 operation order, so a
specialized F32 tile is bit-identical to the generic one before the half
rounding. The 64×64 tile stages the activation tile transposed to
`[k][token]`, so the B loads are plain loads; it fits only with half
operands, at 8 KB + 8 KB of threadgroup memory. The fragment tile stages
nothing: a lane holds elements (r, c) and (r, c + 1) of each
`simdgroup_matrix` (`nu_fragment_row/col`, pinned by `nu_fragment_layout`
in the `bench_kernels` build), decodes segment q of its own weight row into
them, and builds each activation fragment from two float4 loads just before
its multiply; the four SIMD groups split K and sum their partials in a
fixed order. `nu_matmul_*_8`, the staged 16×8 tile, is kept as the
benchmark control (`matmulTile`); `matmulKernel` forces any instantiation
for measurement.

Contract: rows % 8 == 0, columns % 64 == 0, float4-aligned input (offset and
stride), activation buffers padded to `matmulPadded` (a multiple of 64
token rows); rows past `tokens` are computed and stored on the padding.
Half operands assume |w| and |x| below 65,504; decoded weights are O(1),
the activations are checked by `generation-check --metal` (chunked against
stepped), never per element, and `generic_only` runs every product through
the F32 tile to separate the rounding from the schedule. The tile
measurements are in the record
([the 64×64 tile](../benchmarks/record/metal-backend.md#the-batched-prefill-tile-2026-09-09),
[small chunks](../benchmarks/record/metal-backend.md#small-chunk-tile-2026-09-19),
[the fragment tile](../benchmarks/record/metal-backend.md#the-register-fragment-tile-2026-10-01-below-its-target)).

## Attention

The decode split pass, verify, prefill chunk, and segments attention
kernels, and the cache layouts they read, are in
[metal-attention.md](metal-attention.md).

## Recurrent state

- `nu_delta`: one SIMD group per (value head, value coordinate): decayed
  prediction, error, in-place update, scaled output. Q/K heads broadcast
  modulo the Q/K head count (the tiled order), or with `DeltaShape.grouped`
  value head `h` reads Q/K head `h / (vheads / qheads)` (the order Drama 3's
  GGUF keeps; `nu_delta_chunk` takes the same flag).
- `nu_convolution`: one thread per channel, history shifted in place;
  `nu_convolution_rows` and `nu_convolution_history` are the chunk forms,
  bit-identical to the sequential kernel by construction.
- `nu_delta_chunk`: chunkwise DeltaNet for a prefill chunk in one dispatch
  per layer, the WY form of `cpu.recurrent.deltaChunk`
  ([cpu-reference.md § Chunkwise DeltaNet](cpu-reference.md#chunkwise-deltanet)).
  A threadgroup per (value head, block of 32 value rows) loops over 32-token
  sub-chunks with the state carried in the session buffer; each group reads
  and writes only its own 32 state rows, and a `mem_device` barrier orders
  the carry before the next sub-chunk. Per sub-chunk: `K·Kᵀ` and `Q·Kᵀ` on
  the matrix unit, `S₀·Kᵀ` and `S₀·Qᵀ` streamed from the state rows, decay
  ratios as `exp(L_t − L_s)` of cumulative log decays, forward substitution
  with one thread per value column, the output on the matrix unit, and the
  carry `S_new = γ_n S₀ + Wᵀ K` stored in place. The state never sits in
  threadgroup memory (64 KB per head against 32 KB); the six 32×32 tiles
  take 24 KB. Token rows past `count` go through the zero-filling masked
  loader, so the caller's padding is never read. Contract
  (`Backend.deltaChunk`): `keys % 8 == 0`, `values % 32 == 0`, `count` ≤
  4,096, `qkv` rows padded to a multiple of 32 (`deltaChunkRows`).
  Nontransactional like `nu_delta`.

## Gathered expert kernels

A mixture-of-experts layer stores each expert projection as a 3-D tensor
`[experts][rows][columns]` (GGUF dimension 2 is the expert), and a token
reads only the `k` experts its router selected. Decode is four dispatches
per layer:

- `Backend.route`: one 256-thread group per row of logits; `k` rounds of
  "best untaken" select by **logit** (value desc, index asc), so `exp`
  rounding cannot reorder near-ties, and the selected probabilities are
  divided by their sum clamped below at the smallest F16 normal, as the
  CPU reference does. At most 256 experts and 64 selected.
- `Backend.matvecExperts`: threadgroup `g` serves slot `g / row_groups` and
  16-row block `g % row_groups` of that slot's expert, through the segment
  bodies, so a selected expert costs the bytes of a dense matrix of its size.
  `in_stride` 0 shares one input across the slots (gate-up). Expert indices
  are clamped to the tensor, so a corrupt routing buffer cannot read past it.
- `Backend.geluMulRows`: `gelu(gate[r][i]) · up[r][i]` over strided rows.
- `Backend.combineExperts`: `out[row][c] = Σ_s w[row][s] · scale[e_s] ·
  y[row·k + s][c]`, the per-expert down scale optional.

Prefill reads each expert's weights once per 32 of its rows.
`Backend.expertLists` turns the chunk's routing into a row list grouped by
expert and a tile list of (expert, first, count) entries per 32 rows, in
one threadgroup (histogram by atomics, prefix sum, scatter); the bound
`expertTileBound` lets the host size the grid before the counts exist.
`Backend.matmulExperts` is the dense tile's body with a `GATHER` flag: it
reads a tile's activation rows through the row list (input row `slot_row /
in_group`, `in_group` = k on gate-up) and scatters each valid result to
its slot row. Q4_0, Q5_0, and Q5_1 have a half 64×32 instantiation; every
other encoding takes the generic F32 tile. The chain per layer is `route` →
`expertLists` → `matmulExperts` → `geluMulRows` → `matmulExperts` →
`combineExperts`. Qwen3.5-MoE's gated shared expert adds
`Backend.sharedExpertAdd`: `x[r] += sigmoid(g · x[r]) · y[r]` in one dispatch
per row. Measurements and the adapter's routing sensitivity:
[record](../benchmarks/record/metal-backend.md#gathered-expert-kernels-2026-09-18).

## Row-wise kernels and epilogues

- `nu_embed` decodes one row of the embedding matrix.
- `nu_rmsnorm`: one 256-thread group per row, strided rows, weighted, with
  an optional `silu(multiplier)` epilogue (DeltaNet's output gate).
- `nu_l2norm`, `nu_rope` (from an F64-computed cos/sin table uploaded at
  init; `Backend.ropeTable` takes optional per-pair frequency factors; a
  `pairing` parameter rotates split-half or adjacent pairs,
  `cpu.rope.Pairing`), `nu_rope_rows`, `nu_add`, `nu_silu_mul`,
  `nu_silu_inplace`, `nu_delta_gates`, `nu_sigmoid_gate`.
- Gemma 4's epilogues: `nu_gelu_mul` (tanh GELU of `cpu.gelu`), `nu_scale`,
  `nu_add_scale` (x = (x + y)·s, one rounding after the add), `nu_softcap`
  (cap·tanh(x/cap)). Their `tanh` is `nu_tanh`, clamped at ±20: Metal's
  `tanh` returns NaN past about ±44, while F32 tanh is exactly ±1 from ±20
  on, so the clamp changes no finite result.
- `nu_pack_half`, `nu_pack_q8_0`, and `nu_unpack_q8_0` write and expand
  the cache rows ([metal-attention.md § The KV
  cache](metal-attention.md#the-kv-cache)).

### Fused decode norms

`nu_rmsnorm_add` (`destination = (destination + norm(input)·w) · scale`,
every post norm), `nu_add_rmsnorm` (`residual += input`, then `output =
norm(residual)·w`; Qwen's post-attention norm), and `nu_rmsnorm_rope` (the
norm, then the rotation of the leading `dims` channels after a `mem_device`
barrier; every q/k norm) each replace a pair of dispatches. The plans call
the helpers `Backend.rmsNormAdd`, `addRmsNorm`, and `rmsNormRope`, which run
the unfused pair when `Backend.fused_norms` is false (`bench
--unfused-norms`). The saving is the removed launch, under 1 % of a step
([record](../benchmarks/record/metal-backend.md#fused-decode-norms-2026-09-21-closed-below-its-target)).

### The Hadamard transform kernel

`nu_hadamard` is the activation side of a folded rotation (`cpu.hadamard`,
[bonsai.md § Rotation](../models/bonsai.md#rotation-prismhadamard-as-the-forks-loader-reads-it)):
in place over `rows` rows of `width` floats at a stride, per 1,024-block,
forward `x = H (signs ⊙ x)` or inverse `x = signs ⊙ (H x)`, scale 1/32
exactly. `Backend.hadamard(data, signs, width, rows, stride, inverse)`
requires float4-aligned data and sign buffers and a width that is a
multiple of the block. Its cost is launches, about 2 % of a Bonsai token
([record](../benchmarks/record/metal-backend.md#the-hadamard-transform-kernel-2026-09-18)).

### The rotation on the Qwen plan

`qwen35_metal.zig` applies the transform where the CPU reference does
([bonsai.md § Metal plan](../benchmarks/record/bonsai.md#metal-plan-2026-09-18)): the
inverse over the embedding rows after the gather; the forward over the
normed residual before the mixer's projections, over the mixer output
before `attn_output` / `ssm_out`, over the normed residual before the FFN
pair, over the FFN hidden before `ffn_down`, and over the output-head
input, each in place. The DeltaNet `ssm_alpha` / `ssm_beta` projections are
dispatched before the residual is transformed, so the plain file's
four-way merged matvec becomes two two-way merges around it. The `ssm_out`
input is first regathered from the mixer's tiled head order into the
fold's grouped order by `nu_gather_rows` (`Backend.gatherRows`:
`dst[r][g] = src[r][map[g]]`, the map uploaded once at plan init) into a
scratch of one decode row and `padded` prefill rows.

## Sampling

- `nu_argmax_partial` + `nu_argmax_final`: greedy selection with
  lowest-index tie breaking; the CPU reads back 4 bytes instead of the
  logits.
- `nu_topk_partial` + `nu_topk_final` + `nu_expsum_partial`: the best 256
  logits by (value desc, index asc) and Σ exp((l − max) / T) as 64 F32
  partials with non-finite flags. The partial pass keeps 16
  register-resident values per thread and runs k rounds of "best untaken";
  the final pass is a 64-way merge of sorted lists. 2 KB read back per
  sampled token; the contract is in
  [sampling.md](sampling.md#sampling-on-the-gpu-without-reading-the-vocabulary-back).

### History penalties

`nu_penalize` applies the sampler's presence and repetition penalties to
the logits in place, one thread per logit: for a token in the history bit
set, `l / r` when `l > 0` and `l · r` otherwise, then `l − presence`. The
set arrives as `vocabulary / 32` little-endian u32 words; the plan uploads
it only when `History.revision` changed. `Backend.penalize` validates the
count, the word range, and a finite positive `repetition`; the plans record
it between the output head and the argmax or top-k, only when a selection
readback was requested, so a raw `logits` readback stays unpenalized for a
CPU sampler. The device numbers equal `Sampling.Sampler.penalize`
operation for operation, except that subnormal operands flush to zero.
