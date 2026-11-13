<!-- kind: reference -->
# Metal attention and the KV cache

How the [Metal backend](metal-backend.md) stores attention rows and the
kernels that read them: the decode split pass, its verify form, the prefill
chunk kernel, and the cache-free segments kernel. Thread geometry is in
[metal-kernels.md § Kernel geometry](metal-kernels.md#kernel-geometry), the
checks in [metal-evidence.md](metal-evidence.md), and how each was built and
measured in [the record](../benchmarks/record/metal-backend.md).

## The KV cache

`session.Layout.attention` carries a `precision` (`f32`, `f16`, or `q8_0`);
the session block is bytes, and each attention layer exposes `Rows` views
(`range(first, count)` for a GPU binding, `floats(first, count)` for the
CPU reference, which asserts `f32`). The plan builds its layouts from
`Plan.init`'s `kv` argument; `engine.Engine.open` takes it from the command
(`--kv f16|f32|q8_0`, `engine.kv_precision` in `nuclis.json`, default
`f16`), and `Engine.kv_precision` reports what the session uses. The CPU
reference never carves another layout: a request on `--backend cpu` runs
F32 and the report says so. `Backend.storeRows(pairs, precision)` (a copy,
`packHalf`, or `packQ8`) is each plan's one cache write.

### F16 KV cache

A 32,768-token Qwen3.8 session holds 2.15 GiB with `f16` and 4.15 GiB with
`f32` (16 attention layers × 2 × 1,024 channels per token, plus 150 MB of
recurrent state).

- **Decode.** The merged attention projection writes F32 into the plan's
  `k`/`v` scratch rows, the key norm and RoPE run there, and one
  `nu_pack_half` dispatch (round to nearest even, bit-identical to Zig's
  `@floatCast`) writes both slots. The decode kernels read `half` and
  convert per element; queries, scores, the softmax, and the output stay F32.
- **Prefill.** The chunk's rotated keys and values are packed straight into
  the cache range and the rotated queries into `q_c_h`; the chunk kernel's
  half instantiation multiplies half operands, so the F16 chunk path rounds
  Q and P as well as K and V.

Half assumes |k|, |v| < 65,504 after the key norm and RoPE, checked through
the generation check and the pinned prompts, never per element. Tolerances
and the bring-up:
[record](../benchmarks/record/metal-backend.md#f16-kv-cache-2026-09-10).

### Q8_0 KV cache

`q8_0` stores each attention row as GGML's Q8_0 blocks (an F16 scale `d`,
then 32 `i8` codes; value `d * q`; 34 bytes per 32 values, 53 % of F16),
for every family. `Session.init` refuses a `q8_0` row that is not a whole
number of blocks; every cache row in the catalogue is a multiple of 32.

- **Writing.** `nu_pack_q8_0` (`Backend.packQ8`) quantizes the F32 rows one
  SIMD group per block: `d = half(amax / 127)`, `q = clamp(round(x / d),
  -127, 127)`, zero codes when `d` is zero or not finite, NaN coded as zero.
  The same arithmetic is `quant/encode.zig` `q8_0Row`, and the two are
  bit-identical (`precise::divide` under `MTLMathModeSafe`).
- **Decode and verify.** The kernels find each cache row once (`nu_kv_row`)
  and fill a lane's channels with `nu_kv_load`, overloaded for F32, F16, and
  `nu_q8_0`; `d * q` is exact in F32, so the kernel's own error is the F32
  one.
- **Prefill.** The chunk bodies `simdgroup_load` 8×8 tiles from device
  memory and cannot read blocks, so a plan with a `q8_0` cache expands the
  visible rows `[first, position + count)` of K and V into a plan-owned F16
  stage (`kv_stage`: capacity × the widest row × 2 bytes, keys then values)
  with `nu_unpack_q8_0` and runs the F16 chunk kernel on it. A chunk
  therefore sees K and V rounded once more to half than decode does.
  `attentionChunk`, `attentionChunkReuse`, and the three-pass `attention`
  refuse `q8_0`.

Q8_0's logit error is 40–50× F16's, the ratio of an 8-bit block code to an
11-bit mantissa, and perplexity does not measurably change; on Gemma 4
26B-A4B routing amplifies it past the greedy token, so that configuration's
Q8_0 logits are checked finite only. The per-family bounds are
`inference/generation-check.zig` `Bounds`; the measurements:
[record](../benchmarks/record/metal-backend.md#q8_0-kv-cache-2026-10-15).

## Decode attention

`Backend.attentionDecode` runs a split pass and a merge: no
`[heads][visible]` score buffer, and every cache row read once per KV head,
shared by the query heads of its group. The grid is (KV head, split) with
`splits = min(64, ceil(visible / 256))`; each split keeps per head a running
max, sum, and accumulator (online softmax in F32; F16 and Q8_0 rows convert
on load; queries and probabilities are F32) and writes one partial
`(m, l, acc[value_width])` per head into `[query_heads][splits][2 +
value_width]`. `nu_attention_merge` rescales every split to the global max
and divides by the merged sum; empty splits (`m = −∞`, `l = 0`) drop out.
Two kernels share that partial layout:

- `nu_attention_heads_t<KV, GROUP, CH>` takes keys and values of one width
  `32 × CH` (128, 256, 512): SIMD group `s` owns GROUP query heads and
  walks every RL-th row, the head groups reading each row together; lane
  `l` owns `CH` consecutive channels read as packed `float4`/`half4`
  without guards. Q8_0 rows run in three phases (every head's dot, the
  softmax, the V accumulation) with the block scale outside the per-code
  products. `Backend.decodePlan` picks GROUP and RL per cache type and
  width by measured optimum, halving row lanes until the threadgroup fits
  1,024 threads and its merge the 4,096-float stage.
- `nu_attention_decode_t` takes every other shape, templated on the cache
  type, query heads per threadgroup, and channels per lane (8 × 8 up to 256
  wide, 4 × 16 up to 512); a KV head with more query heads than an
  instantiation holds takes several threadgroups per split
  (`attentionDecodeHeadGroups`). Its Q8_0 instantiations give lane `l`
  four consecutive channels per 128-wide stripe (`nu_kv_channel`), so a
  row costs a lane one scale and two aligned half-words of codes per stripe.

The three-pass kernels (`nu_attention_scores`, `_softmax`, `_values`) stay
in the library as `test-metal`'s oracle. Why the head split:
[record](../benchmarks/record/metal-backend.md#the-split-pass-at-the-memory-roof-2026-10-15).

## Few-query verify attention

`Backend.attentionVerify` runs a batch of at most `verify_max_rows` (16)
query rows (a verify batch, a drafter's batched commit, a prefill's short
tail) through the decode split pass with a row dimension: the grid is (KV
head, head group, split, row) and the merge runs over (row, head). Row r of
`rows` sees the first `visible − (rows − 1 − r)` cache rows (and, with a
window, only its last `window`); the split boundaries are the last row's,
so a split past a row's end stays empty and the merge skips it. Queries are
F32 over every cache precision; decode is the rows = 1 case. Rows share
nothing but the dispatch: each re-reads its prefix. The models route every
chunk of at most 16 rows here (Gemma only without an image span); the
partial buffers are sized for 16 rows. Its cost is linear in rows and bound
by the per-(row, head) arithmetic, not the bus
([record](../benchmarks/record/metal-backend.md#few-query-verify-attention-2026-09-30)).

## Prefill: chunk attention

`nu_attention_chunk` is causal attention for a chunk in one
dispatch per layer. A threadgroup per (query head, 32-query tile); each SIMD
group owns 8 query rows and walks key tiles of 32 cache positions up to its
own causal limit with an online softmax, computing Q·Kᵀ and P·V with
`simdgroup_float8x8`. The running output is rescaled by exp(m_old − m_new)
as a multiply by a diagonal matrix, only when some row's max moved. Key
sub-blocks that would cross the end of the visible range are staged with
rows past the end zeroed, so no cache row beyond `position + count` is
read; masked keys score −∞. Row `t` attends to `cache[0 .. position + t]`,
or with a nonzero `window` to its last `window` keys; a row whose keys are
all hidden in a tile keeps its max at −∞ until one shows; the Gemma plan
slices the cache at the earliest key its chunk can see. `span { begin,
end }` makes an image bidirectional: chunk rows in the span see keys up to
`position + end` (Gemma 4 passes an image chunk's span on its sliding
layers). The F16 instantiation `_h` loads half queries, keys, and
values into `simdgroup_half8x8` operands, rounds the probability tile to
half before P·V, and accumulates in F32 (the matrix unit multiplies
operands of one type). Contract (`Backend.attentionChunk`): widths multiples
of 8, value width ≤ 512 (above 256 the grid adds one threadgroup per 256
value columns, each recomputing the scores), `count` ≤ `max_chunk_rows`
(16,384), query and output buffers padded to a multiple of 8 rows
(`attentionChunkRows`), the cache covering `position + count` rows.
Chunks of at most `attention_reuse_max_rows` (64) rows on the 256-wide
geometry, with no span, take `nu_attention_chunk_reuse`, whose SIMD groups
split the value columns and share the probability tiles (11–16 % faster at
those counts, level at full chunks:
[record](../benchmarks/record/metal-backend.md#long-context-prefill-attention-second-attempt-2026-09-21-closed-negative)).

## Segments attention with materialized scores

`Backend.attentionSegmentsGrouped` is bidirectional attention with no
cache over packed rows with per-row `[begin, end)` bounds and an optional
symmetric window, grouped-query heads of any multiple of 8 up to 512, and
inputs up to 8,192 rows: EmbeddingGemma 2's encoder and the vision
encoders. It runs over a list of query tiles of at most 32 rows, each with
the key range its rows can see rounded up to 32 and a score strip of
`rows · span` floats (`attentionTiles` builds the list, `chunkTiles` splits
it to fit a score buffer). Three F32 dispatches per chunk:
`nu_segments_scores` (S = scale · Q·Kᵀ per tile, head, and 32-key block,
invisible keys −∞), `nu_segments_stats` (each strip row's max and 1 / sum,
stored after the strips), and `nu_segments_values` (O = P·V, P formed while
the strip is staged). Key rows past `rows` stage as zeros; a row that sees
no key comes out zero. A row's result depends only on its tile's rows and
keys, so batching is invisible in the output. Materialized scores run the
matrix tiles at 2.9–4.3 TFLOP/s where the online-softmax bodies stayed near
0.7 ([record](../benchmarks/record/metal-backend.md#segments-attention-with-materialized-scores-2026-10-09)).
