# Metal backend

The Metal backend runs the pinned Qwen text schedule on the GPU: one command
buffer per token, every activation resident on the GPU, and the CPU supplying a
token ID and reading back either logits or a greedy token. It is numerically
validated against the CPU reference and llama.cpp traces. For the surrounding
stack see [architecture.md](../architecture.md); for the performance plan see
the 2026-09-07 review.

## Build and run

Metal is opt-in at build time so default builds and tests never touch the GPU:

```sh
zig build -Dmetal=true -Doptimize=ReleaseSafe --global-cache-dir .zig-cache/global
./zig-out/bin/nuclis generate --backend metal --prompt 'Hi' --max-tokens 8 --ctx-size 64
./zig-out/bin/nuclis bench --backend metal --prompt-file prompt.txt --max-tokens 64
zig build test-metal -Dmetal=true -Doptimize=ReleaseSafe --global-cache-dir .zig-cache/global
zig build test-generation -Dmetal=true -Doptimize=ReleaseSafe --global-cache-dir .zig-cache/global -- MODEL --metal
```

Without `-Dmetal=true`, `inference.metal.Backend.init` returns
`error.MetalNotEnabled` and the CLI rejects `--backend metal`. `test-metal` runs
`inference/metal-check.zig`, an explicit GPU fixture and lifecycle check
needing a Metal device but no model. `test-generation --metal` runs the session
isolation/reset protocol on the GPU plan. Full-model layer comparison uses the
same `--logits`/`--trace-dir` oracle as the CPU backend ([generation.md](generation.md)).

## Files and ownership

| File | Owns |
| --- | --- |
| `backends/metal/bridge.m` | Device, queue, shader library, pipelines, buffers, one open command buffer. A generic recording API: `begin`, `dispatch(pipeline, bindings, constants, grid)`, `commit`. Opt-in profiling: timestamp counter sample buffers and one encoder per dispatch, resolved to seconds per dispatch at commit. Knows nothing about models, shapes, or encodings. |
| `backends/metal/root.zig` | `Backend`: compiles the kernel set, hands out `Buffer` handles (`create`/`wrap`/`slice`), and exposes one typed encoder per kernel that validates shapes before recording. `Profile` accumulates timed dispatches by kernel, encoding, and shape. Knows kernel contracts, not layer schedules. |
| `backends/metal/dequant.metal` | GGUF block decoders, ported line by line from `quant/decode.zig`. Bit-exact with the CPU decoders. |
| `backends/metal/kernels.metal` | Compute kernels: generic matvec, specialized matvec for Q3_K/Q4_K/Q5_K/Q6_K/IQ3_S/IQ4_XS/Q4_0, merged projections (plain, SiLU pair, GELU pair), embed, rmsnorm, l2norm, rope, add, silu·mul, silu, gelu·mul, scale, add·scale, softcap, delta gates, sigmoid gate, DeltaNet, convolution, three-pass decode attention (templated on the cache type), flash-decoding attention (templated on cache type, heads per group, channels per lane: two instantiation pairs) + merge, argmax (2), partial top-k + exp-sum (3), batched prefill matmul, chunk forms (rope rows, convolution rows + history, copy), causal chunk attention with window and value splits (F32 and half instantiations), chunkwise DeltaNet, F16 packing. |
| `models/qwen35_metal.zig` | `Plan`: the Qwen schedule expressed as encoder calls. Owns the session and activation buffers; borrows weights and the backend. |
| `models/gemma4_metal.zig` | `Plan`: the Gemma 4 12B schedule (MODL-06) on the same encoders; sliding-window slices, two RoPE tables, the wide global-layer attention ([gemma4.md § Metal plan](gemma4.md#metal-plan-modl-06-2026-09-11)). |
| `models/muse_glimmer_metal.zig` | `Plan`: the Muse Glimmer 30B schedule (MODL-12) on the same encoders; adjacent-pair RoPE on sliding layers only, the sigmoid attention gate, the untied scaled head ([muse-glimmer.md § Metal plan](muse-glimmer.md#metal-plan-modl-12-2026-09-19)). |
| `metal-check.zig` | Explicit GPU checks against pinned fixtures and CPU references; `--matvec-bench` measures the matvec kernels' achieved bandwidth. |

The MSL source is assembled at compile time: the IQ3_S codebook is emitted from
the single Zig definition, then `dequant.metal` and `kernels.metal` are embedded.
The shader compiles at backend creation (about one second) with
`MTLMathModeSafe`, so no reassociation or contraction is applied.

## Execution model

```text
Plan.step(token)
  session.begin()                       CPU: admission
  backend.begin()                       open command buffer + compute encoder
  embed(token) -> x
  for each of 64 layers:
    rmsnorm(x) -> normalized
    mixer: attention (merged qg+k+v, norm, rope, cache write, scores/softmax/values, gate, output)
        or DeltaNet   (merged qkv+z+beta+alpha, conv, silu, l2norm, gates, delta, norm*silu(z), output)
    x += projected
    rmsnorm(x) -> normalized; merged gate/up with silu*mul; down matvec; x += projected
  final rmsnorm, output matvec -> logits [, argmax -> result]
  backend.commit()                      submit, wait for completion, account GPU time
  read logits or greedy id from shared memory
  session.commit()                      CPU: position += 1
```

KERN-04 records about 932 dispatches per step (including final output work), down
from 1,236. The layer dispatches are recorded into one serial compute pass; Metal
guarantees each dispatch sees the previous one's writes. `commit()` **waits**,
so after it the CPU may read any buffer and may `reset()` the session. That
synchronous contract is what makes the following simple:

- **Weights** are the read-only file mapping wrapped per tensor with
  `newBufferWithBytesNoCopy` (cached by address). The mapping outlives the
  backend. Wrapping the whole 16 GB mapping as one buffer was measured to make
  GPU time swing 3–4× between runs; per-tensor buffers are steady.
- **Session state** (`runtime/session.zig`) is one page-aligned byte block
  wrapped once. KV rows (F32 or F16 per the layout, KERN-07), convolution
  history, and DeltaNet matrices are typed views the plan binds by byte
  range. `reset()` is a CPU memset, valid only when no GPU
  work is in flight — true after every `commit()`. An asynchronous backend must
  fence before `reset`, `fail`, or observer callbacks; that invariant is
  documented at the top of `session.zig`.
- **Activations** are backend-owned shared buffers sized from the plan
  (5120, 17408, 12288, …) at `Plan.init`. The bridge has no model constants.
- **Failure**: a failed command buffer or a rejected dispatch returns an error;
  `errdefer` marks the session failed and it stays poisoned until `reset()`.
  The DeltaNet update is in place, so a partially executed token is not
  transactional — matching the CPU contract's outcome.
- **Observer.** `check` runs between recorded layers with no synchronization
  (Ctrl-C, time limit). `layer` needs the activations, so it forces a `commit`
  after every layer: traces only. Until 2026-09-07 the CLI installed a single
  callback for both purposes and every `generate`/`bench` token ran as 64
  command buffers; the first per-kernel profile exposed it, and splitting the
  callbacks took decode from 8.54 to 9.69 tok/s without touching a kernel.
- **Profiling** (`enableProfiling`, `bench --profile`): Apple GPUs sample
  timestamps at encoder boundaries only, so each dispatch is recorded in its
  own encoder and timed by its start/end stamps (nanoseconds, same timeline as
  `GPUStartTime`; measured). The per-dispatch encoders cost ~8 ms per token,
  reported as the gap between attributed and command-buffer time, so profiled
  rates are diagnostic. Methodology and results: [bench.md](bench.md#per-kernel-profile).

## Kernels

All kernels bind data buffers at indices 0–6 and a small parameter struct at 7.
Shapes come from parameters; nothing is hard-coded to Qwen.

### Kernel geometry

How each kernel maps onto threadgroups and SIMD groups (32 lanes), and
which cross-lane operations it relies on. The concepts are explained in
[llm-guide.md § 35](../llm-guide.md#35-simd-groups-which-layer-they-live-in);
the SIMD-level optimizations of the decode matvec are in
[§ Specialized matvec](#specialized-matvec) and the negative results in
[§ KERN-05](#kern-05--per-block-cost-research-2026-09-08-closed-without-a-kernel-change).
Thread counts are the numbers `Backend` passes to `dispatch`.

| Kernel | Threads / group | One threadgroup owns | One SIMD group owns | Cross-lane ops | Threadgroup memory |
| --- | ---: | --- | --- | --- | --- |
| `nu_matvec` (generic) | 32 | one output row | the row: lanes stride 16-value segments | `simd_sum` | none |
| `nu_matvec_*` (specialized) | 128 | 16 output rows | 4 rows; 8 lanes per 256-value block | `simd_sum` | none |
| `nu_matvec_segments` plain / pair | 128 | 16 rows of one segment / 8 gate+up pairs | 4 rows | `simd_sum`; pairs share 16 floats | 64 B (pair mode) |
| `nu_matvec_experts` (KERN-09) | 128 | 16 rows of one slot's expert (the segment bodies) | 4 rows | `simd_sum` | none |
| `nu_route` (KERN-09) | 256 | one row of ≤ 256 router logits | 32 logits; k rounds of "best untaken" | `simd_max`, `simd_sum`, `simd_shuffle_down`, 8-way threadgroup pick | 8 + 8 + 64 entries |
| `nu_matmul_*` (specialized, ENGN-05) | 128 | 64-row × 64-token output tile (32×32 for chunks of ≤ 32 tokens) | a 32×32 quarter as 4×4 `simdgroup_float8x8` (2×2 in the small tile) | matrix loads and MACs | 8 KB half weight tile + 8 KB half activation tile (small tile: 4 KB, activations from device), `threadgroup_barrier` |
| `nu_matmul_*_8` (KERN-11) | 128 | 16-row × 8-token output tile (two token tiles for 9..16) | 16 rows × 8 tokens over one K slice: two 8×8 accumulators sharing one B load, four groups split K | matrix loads and MACs | 8 KB half tile (16 rows × 64 k per group), `simdgroup_barrier` only |
| `nu_matvec_rows_*_t<n>` (KERN-12) | 128 | 16 output rows × up to 8 tokens | 4 rows; 8 lanes per 256-value block; one accumulator per (row, token) | `simd_sum` across 32 lanes per (row, token) | none |
| `nu_matmul` (generic) | 128 | 32-row × 32-token output tile | a 16×16 quarter as 2×2 `simdgroup_float8x8` | matrix loads and MACs | 8 KB F32 weight tile + 8 KB F32 activation tile, `threadgroup_barrier` |
| `nu_attention_chunk` / `_h` | 128 | (query head, 32-query tile, 256 value columns) | 8 query rows: 4 score blocks, 32 output blocks | `simd_shuffle_xor`, `simd_shuffle`, `simd_any`, matrix MACs | 6 KB (per-group score tile, diagonal, staging); 7.5 KB in the half instantiation (its own probability tile), `simdgroup_barrier` only |
| `nu_delta_chunk` | 128 | (value head, 32 value rows), all sub-chunks | an 8-row block of every 32×32 tile; column `tid` in the triangular solve | matrix MACs; no shuffles | 24 KB of 32×32 tiles, `threadgroup_barrier` per phase, `mem_device` per sub-chunk |
| `nu_attention_decode` / `_h` (KERN-08), `_w` / `_wh` (MODL-06) | 128 | (KV head, group of ≤ 8 query heads — ≤ 4 in the wide pair —, split of the visible rows) | every fourth row of the slice; lane l owns channels l, l+32, … (8 per lane, 16 in the wide pair) with per-head running max, sum, and accumulator in registers | `simd_sum` per (row, head); 3-round merge of the SIMD groups through 8 KB | 8 KB stage, `threadgroup_barrier` |
| `nu_attention_merge` | 256 | one query head | — (threads stride the value channels, log-sum-exp over ≤ 64 splits) | none | none |
| `nu_attention_scores` / `_values` (and `_h`; the three-pass decode kept as `test-metal`'s oracle) | 32 | one (head, position) / (head, channel) | the dot product / weighted sum | `simd_sum` | none |
| `nu_attention_softmax` | 32 | one head's score row | the whole row | `simd_max`, `simd_sum` | none |
| `nu_delta` | 32 | one (value head, coordinate) | the 128-wide contraction | `simd_sum` | none |
| `nu_l2norm` | 32 | one head vector | the norm | `simd_sum` | none |
| `nu_rmsnorm` | 256 | one row | a 32-value strip | `simd_sum` then 8-way combine | 8 floats |
| `nu_argmax_partial`, `nu_topk_partial`, `nu_expsum_partial` | 256 | a strided slice of the vocabulary | its 32 values | `simd_shuffle_down` reduction, 8-way threadgroup pick | 8–16 entries |
| `nu_argmax_final` | 32 | the partials | all of them | `simd_shuffle_down` | none |
| `nu_topk_final` | 256 | the 64 sorted lists | cursors | k-way merge on plain threads | list cursors |
| elementwise (`nu_add`, `nu_silu_*`, `nu_gelu_mul`, `nu_gelu_mul_rows`, `nu_combine_experts`, `nu_scale`, `nu_add_scale`, `nu_softcap`, `nu_rope*`, `nu_convolution*`, `nu_copy`, `nu_pack_half`, `nu_sigmoid_gate`, `nu_delta_gates`, `nu_embed`) | 256 | 256 elements | 32 elements, one per lane | none | none |

Three patterns cover the table: a *reduction* kernel gives a SIMD group one
output and reduces with `simd_sum`; a *tile* kernel gives a threadgroup a
tile of memory that its SIMD groups multiply with the matrix unit; an
*elementwise* kernel uses threads as independent workers and never
communicates. The choice per kernel follows one rule: put in a SIMD group
what must be reduced or exchanged, put in a threadgroup what must share
memory, and leave the rest to the grid.

- `nu_matvec`: the generic kernel, one SIMD group (32 lanes) per output row;
  each lane decodes 16-value segments straight from quantized bytes through the
  runtime encoding switch. It serves F32, F16, Q8_0, IQ4_NL, and any
  misaligned range.
- `nu_matvec_q3_k`, `nu_matvec_q4_k`, `nu_matvec_q5_k`, `nu_matvec_q6_k`,
  `nu_matvec_iq3_s`, `nu_matvec_iq4_xs`, `nu_matvec_q4_0` (MODL-08): specialized kernels,
  selected by `Backend.matvec` when the row start, row stride, and input are
  aligned for their vector loads. See "Specialized matvec" below.
- `nu_embed`: decodes one row of the embedding matrix.
- `nu_rmsnorm`: 256-thread group per row, strided input/output rows, weighted,
  optional `silu(multiplier)` epilogue (used for DeltaNet's output gate).
- `nu_l2norm`, `nu_rope` (from an F64-computed cos/sin table uploaded at init;
  `Backend.ropeTable` takes optional per-pair frequency factors, MODL-06;
  a `pairing` parameter rotates split-half `(i, i + dims/2)` or adjacent
  `(2i, 2i + 1)` pairs, `cpu.rope.Pairing`, MODL-12),
  `nu_add`, `nu_silu_mul`, `nu_silu_inplace`, `nu_delta_gates`, `nu_sigmoid_gate`.
- MODL-06 epilogues for Gemma 4: `nu_gelu_mul` (tanh GELU of `cpu.gelu`, also
  pair mode 2 of `nu_matvec_segments`), `nu_scale` (x *= s), `nu_add_scale`
  (x = (x + y)·s, one rounding after the add), `nu_softcap` (cap·tanh(x/cap)).
  Their `tanh` is `nu_tanh`, clamped at ±20: Metal's `tanh` goes through
  `exp` and returns NaN past about ±44, while F32 tanh is exactly ±1 from
  ±20 on, so the clamp changes no finite result (the first Gemma step
  produced NaN in nine gate values before it).
- `nu_delta`: one SIMD group per (value head, value coordinate); decayed
  prediction, error, in-place update, scaled output; Q/K heads broadcast modulo.
- `nu_convolution`: one thread per channel, history shifted in place.
- Attention (decode): scores, softmax, values as three dispatches over a
  `[head][visible]` score buffer.
- `nu_attention_chunk` (ENGN-03): causal attention for a prefill chunk in one
  dispatch per layer, no score buffer. Threadgroup per (query head, 32-query
  tile); each of its four SIMD groups owns 8 query rows and walks key tiles
  of 32 cache positions up to its own causal limit with an online softmax
  (running max and sum per row), computing Q·Kᵀ and P·V with
  `simdgroup_float8x8`. The 8×32 score tile round-trips through threadgroup
  memory for the row-wise max/exp/sum on plain threads; the running output
  is rescaled by exp(m_old − m_new) as a multiply by a diagonal matrix, only
  when some row's max moved. Key sub-blocks that would cross the end of the
  visible range are staged through threadgroup memory with rows past the end
  zeroed, so no cache row beyond `position + count` is read; masked keys get
  score −∞ (weight exactly 0). Contract (`Backend.attentionChunk`): widths
  multiples of 8, value width ≤ 512 (above 256 the grid adds one
  threadgroup per 256 value columns, each recomputing the scores; MODL-06),
  `count` ≤ 4,096, query/output buffers
  padded to a multiple of 8 rows (`attentionChunkRows`), the cache buffers
  cover `position + count` rows. Row `t` attends to `cache[0 .. position +
  t]`, or with a nonzero `window` (MODL-06) to `cache[position + t + 1 −
  window .. position + t]`: the key loop starts at the tile holding the
  SIMD group's first visible key, hidden keys score −∞, and a row whose
  keys are all hidden in a tile keeps its running max at −∞ (no rescale,
  zero weights) until one shows; the Gemma plan slices the cache at the
  earliest key its chunk can see so row 0 of the slice is that key. The
  output gate (Qwen) is applied afterwards by `nu_sigmoid_gate` over the
  chunk. Evidence: `test-metal` compares 256 query rows at positions
  1,792..2,047 over a 2,048-row cache against the F64 CPU reference per row
  (each row with its own visible prefix): max abs 3.9e-7, bound 1e-5; a
  37-row chunk at position 5 (count and total not multiples of 8 or 32) with
  1e30 in every key/value row after query row 20's horizon matches the
  clean reference for rows 0..20 and stays finite after; single- and
  three-token prompts from an empty cache; padded-row and width contract
  violations are refused before dispatch. MODL-06 adds the Gemma geometry:
  windows of 1,024 before and inside a chunk, a window of 8 with fully
  hidden tiles, and 16 heads of 512 over one KV head (two value splits),
  F32 within 3.0e-6 (bound 1e-5) and F16 over the rounded operands within
  1.9e-4 (bound 2e-3) of the F64 reference per row with the window applied
  as a key slice.
- `nu_delta_chunk` (ENGN-04): chunkwise DeltaNet for a prefill chunk in one
  dispatch per layer, the WY form of `cpu.recurrent.deltaChunk`
  ([cpu-reference.md § Chunkwise DeltaNet](cpu-reference.md#chunkwise-deltanet-engn-04-stage-1)).
  Threadgroup per (value head, block of 32 value rows), four per head, 128
  threads; it loops over 32-token sub-chunks with the state carried in the
  session buffer (each group reads and writes only its own 32 state rows,
  so no cross-group synchronization exists; a `mem_device` barrier orders
  the carry before the next sub-chunk). Per sub-chunk: `K·Kᵀ` and `Q·Kᵀ`
  as `simdgroup_float8x8` products over the tokens; `S₀·Kᵀ` and `S₀·Qᵀ`
  streamed from the state rows; decay ratios as `exp(L_t − L_s)` of
  cumulative log decays (thread 0 prefix-sums 32 values); the strictly
  lower `A = β·r·KK` and `B = β(v − γ S₀k)` on plain threads; forward
  substitution with one thread per value column (32 sequential rows, no
  barriers inside); `O = r·KQ × U` on the matrix unit plus the `γ S₀q` term;
  and the carry `S_new = γ_n S₀ + Wᵀ K` as a diagonal multiply of the old
  block plus four MACs per 8×8 state block, stored in place. The state is
  never held in threadgroup memory (64 KB per head against the 32 KB
  limit); the six 32×32 tiles it does hold total 24 KB. Token rows past
  `count` in the last sub-chunk go through the zero-filling masked loader,
  so the caller's padding is never read (the fixture fills it with NaN).
  Contract (`Backend.deltaChunk`): `keys % 8 == 0`, `values % 32 == 0`,
  `count` ≤ 4,096, `qkv` rows padded to a multiple of 32
  (`deltaChunkRows`), input row layout as `nu_delta` (q, k, v per head; the
  Q/K head of value head `h` is `h % qheads`), gate rows `[token][gate_stride]`,
  output rows `[token][out_stride]`. Nontransactional like `nu_delta`.
  Evidence: `test-metal` runs a 70-token chunk (sub-chunks 32, 32, 6) on
  the model shape with L2-normalized q/k against the F64 chunk reference
  per head (output max abs 3.4e-8, bound 1e-5; state 1.2e-7, bound 1e-4)
  and against 70 sequential CPU steps per head (1.2e-7, bound 1e-4);
  contract violations are refused before dispatch.
- `nu_argmax_partial` + `nu_argmax_final`: greedy selection with lowest-index
  tie breaking; the CPU reads back 4 bytes instead of 1 MB of logits.
- `nu_matmul` (ENGN-02, ENGN-05): batched prefill product `out[t][r] = Σ W[r][k]·X[t][k]`
  for a chunk of tokens, one template `nu_matmul_t<encoding, rows, tokens,
  weight type, activation type, staged>` with thirteen instantiations. A
  128-thread group owns a rows × tokens output tile as four SIMD groups of
  `simdgroup_float8x8` accumulators; each 64-column K step decodes one
  16-value segment per thread and row pass into a threadgroup weight tile,
  stages the activation tile transposed to `[k][token]` (so the B loads are
  plain loads), and multiplies. Weights are decoded once per token tile
  instead of once per token. The specialized instantiations
  (`nu_matmul_q3_k`, `_q4_k`, `_q5_k`, `_q6_k`, `_iq3_s`, `_iq4_xs`) decode
  a segment with the vector loads and packed-byte helpers of the specialized
  matvecs (`nu_tile_*`), evaluating the generic decoder's expression in the
  same F32 operation order, so a specialized F32 tile is bit-identical to the
  generic one (Q4_0's tile, MODL-08, addresses two segments per 32-value block
  where the K-quant tiles address sixteen per 256-value block); they hold both operands as **half** in 64×64 tiles (8 KB +
  8 KB) and accumulate in F32. A second set (`nu_matmul_*_32`, 32×32, half
  weight tile, activations loaded from device memory as F32, 4 KB) serves
  chunks of at most 32 tokens, where the 64-row tiles leave a 5,120-row
  projection with only 80 threadgroups. A third set (`nu_matmul_*_8`, KERN-11:
  16 rows × 8 tokens, half weight tile, activations from device as F32, 8 KB)
  splits the K range across the four SIMD groups with a `simdgroup_barrier`-
  only loop and serves chunks of at most `small_chunk_tokens` (24) tokens,
  where the 32-row tile's barrier-bound loop streams weight bytes far below
  the matvec floor; see [§ Small-chunk tile](#small-chunk-tile-kern-11-2026-09-19).
  The generic instantiation
  (`nu_matmul`) keeps F32 operands in 32×32 tiles for F32, F16, Q8_0,
  IQ4_NL, and misaligned ranges (exact for dense rows of any magnitude).
  `Backend.matmul` selects by encoding, weight alignment (the matvec
  rules), and chunk length (`specializedMatmul`, `matmulGeometry`).
  Contract: rows % 8 == 0, columns % 64 == 0, float4-aligned input (offset
  and stride), activation buffers padded to `matmulPadded` (a multiple of
  64 token rows); rows past `tokens` are computed and stored on the padding.
  Half operands assume |w| and |x| below 65,504; the model's decoded weights
  are O(1) and its activations are checked by `generation-check --metal`
  (chunked vs stepped), while the generic tile is one constant away as a
  fallback. `make bench-matmul` (256 tokens, ReleaseSafe, M4 Pro, best of
  five, 2026-09-09), each lever measured alone on the Q4_K and IQ3_S gate
  shape (17,408×5,120), ms / GFLOP/s:

  | Variant (threadgroup memory) | Q4_K | IQ3_S | Kept |
  | --- | ---: | ---: | --- |
  | Baseline: generic decoder, F32, 32×32, activations from device (8 KB) | 14.71 / 3,102 | 19.60 / 2,328 | — |
  | 1. Specialized tile decode (8 KB) | 13.73 / 3,323 | 13.29 / 3,433 | yes |
  | 1b. + activation tile staged in F32 (16 KB) | 12.55 / 3,635 | 12.33 / 3,700 | yes |
  | 3. F32 64×32 (24 KB) / 32×64 (24 KB) / 64×64 (32 KB) | 12.57 / 13.03 / 63.08 | 12.33 / 13.25 / 61.57 | no |
  | 2. Half operands, F32 accumulation, 32×32 (8 KB) | 11.24 / 4,060 | 10.71 / 4,262 | — |
  | 2+3. Half 64×32 (12 KB) / 64×64 (16 KB) | 10.51 / 9.00 | 10.04 / 8.97 | 64×64 |
  | Half weights, F32 activations: 64×32 (16 KB) / 64×64 (24 KB) | 11.02 / 56.09 | 10.49 / 54.89 | no |
  | Prefetch of the next step's segments before the MACs | 10.50 | 10.36 | no (registers) |
  | Activation tile staged transposed, plain B loads | 9.07 | 9.08 | yes |
  | MACs and stores predicated on live 8×8 blocks | 11.91 | 11.74 | no (+30 %) |
  | **Final: specialized half 64×64, transposed staging** | **8.87 / 5,144** | **8.85 / 5,158** | |

  Final table over the six encodings and both shapes: specialized 8.77–9.37
  ms, 4,868–5,205 GFLOP/s (a ceiling of 90–96 prefill tok/s at 54 GFLOP per
  token); generic F32 tile 13.08–18.06 ms, 2,527–3,488 GFLOP/s. Two facts
  decided the design: the per-encoding spread of the baseline (Q4_K 3.1 vs
  IQ3_S 2.3 TFLOP/s) was the generic decoder and lever 1 removed it, but
  the common floor it exposed (3.3–3.5 TFLOP/s) was the F32 matrix phase;
  and threadgroup memory sets occupancy on this GPU with a cliff between
  16 KB and 24 KB per group, which is why the 64×64 tile only works with
  half operands. At 22 tokens the large tiles run at 20–25 GB/s of weight
  traffic (one token tile, too few threadgroups, a latency-bound K loop),
  hence the 32×32 set for short chunks; short prompts remain far from the
  weight-bandwidth floor (KERN-12's multi-row matvec closes below its target —
  see [§ Multi-row matvec](#multi-row-matvec-kern-12-2026-09-20-closed-below-its-target)
  — and KERN-15's split-K remains in [TODO.md](../../TODO.md)).
- Mixture of experts (KERN-09; see [§ Gathered expert kernels](#gathered-expert-kernels-kern-09)):
  `nu_route` (softmax and top-k with renormalized weights per logit row),
  `nu_matvec_experts` (a matvec over the selected experts' slices of a 3-D
  tensor, any encoding through the segment bodies), `nu_gelu_mul_rows`
  (the gated GELU over strided rows, the up half of a fused gate-up row
  bound at its offset), and `nu_combine_experts` (the weighted sum of the
  slots' down projections with an optional per-expert scale). Prefill
  adds `nu_expert_lists` (the chunk's slot rows grouped by expert, with
  the 32-row tile list) and `nu_matmul_experts` / `nu_matmul_experts_q4_0`
  (the matmul body gathering its activation rows through the lists and
  scattering the results back).
- `nu_topk_partial` + `nu_topk_final` + `nu_expsum_partial` (KERN-06): the best
  256 logits by (value desc, index asc) and Σ exp((l − max) / T) as 64 F32
  partials with non-finite flags. The partial pass keeps 16 register-resident
  values per thread and runs k rounds of "best untaken" (SIMD shuffle
  reduction, 8-way threadgroup pick, owner retires the winner); the final
  pass is a 64-way merge of sorted lists with one cursor per list. 2 KB
  read back per sampled token; contract in
  [generation.md](generation.md#sampling-on-the-gpu-without-reading-the-vocabulary-back-kern-06).

## Specialized matvec

Decode reads every weight once per token, so the matvec kernels for Q5_K,
IQ4_XS, Q4_K, and Q6_K decide decode speed. Their design, in
`kernels.metal` (templates instantiated with `[[host_name]]`, so the bridge
needs no function constants):

- **Geometry.** 128-thread groups of four SIMD groups; each SIMD group owns
  four consecutive output rows (`Backend.rows_per_simdgroup`) and walks their
  256-value blocks four at a time, eight lanes per block. A lane decodes 32
  values of its block from one to three vector loads and reads its 32 inputs
  as eight `float4` from device memory, shared by the four rows.
- **Factored scales.** Per 32-value group a lane accumulates `Σq·x` and `Σx`
  and applies the group's scale and minimum once:
  `Σ(d·s·q − dmin·m)·x = d·s·Σ(q·x) − dmin·m·Σx`. With a one-hot input this is
  the CPU decoder's expression exactly, which keeps the pinned fixture columns
  exact; dense inputs differ only by F32 rounding order. Q6_K folds its bias of
  32 into `Σx` the same way.
- **Packed-byte decoding.** Codes are assembled four at a time in the packed
  byte domain (`(v & 0x0f0f0f0f) | (((h >> bit) & 0x01010101) << 4)` completes
  four Q5_K values) and converted with one `uchar4 → float4` cast. The GPU is
  scalar per lane, so building float4s value by value cost 3–4× more; this
  change alone took Q5_K from 113 to 210 GB/s.
- **Alignment rules.** Vector-load widths follow block sizes: Q4_K (144 B) and
  Q5_K (176 B) use `uint4`, IQ4_XS (136 B) `uint2`, and Q3_K/IQ3_S
  (110 B), Q6_K (210 B), and Q4_0 (18 B) use `packed_ushort4`. `Backend.specializedMatvec` requires the row offset and
  stride to be multiples of 16/8/2 respectively and the input to be
  float4-aligned; otherwise the generic kernel is recorded. Every tensor of the
  pinned artifacts qualifies. The K-quant and IQ matvecs walk 256-value
  strides per lane octet; the Q4_0 matvec walks 32-value blocks (lane
  `l` takes blocks `l`, `l + 32`, …), so any whole-block Q4_0 row serves,
  including the 704-column expert down projection of Gemma 4 26B-A4B.
- Scale decoding for Q4_K/Q5_K group pairs is written with selects: half of
  each SIMD group's lanes need the "direct" form and half the "split" form, so
  a branch would execute both.

`make bench-kernels` (`metal-check --matvec-bench`) measures each kernel alone
on fixture-tiled matrices of 38–1,040 MB, eight back-to-back dispatches per
command buffer (64 for matrices under 8,192 rows, whose single dispatch is too
short to hold the clock; isolated dispatches measure GPU clock ramp-up, not
the kernel), best and mean of five command buffers, from
`GPUEndTime − GPUStartTime`.
2026-09-07, Apple M4 Pro, ReleaseSafe, best of five, GB/s of weight bytes:

| Encoding | 69,632×5,120 | 20,480×17,408 | 248,320×5,120 | generic kernel |
| --- | ---: | ---: | ---: | ---: |
| Q4_K | 172 | 150 | 169 | 105 |
| Q5_K | 211 | 206 | 213 | 96 |
| Q6_K | 248 | 241 | 246 | 78 |
| IQ4_XS | 212 | 191 | 212 | 95 |

Published bandwidth is 273 GB/s. Experiments that did **not** move these
numbers and were reverted: 2 or 8 rows per SIMD group, 2 or 8 SIMD groups per
threadgroup, issuing all rows' loads before arithmetic (worse: register
pressure), and a float lookup table in place of `uchar4 → float4` conversion
(mixed).

The in-model profile ([bench.md](bench.md#per-kernel-profile)) named the
limiter: on the same shape all four kernels take 0.8–0.9 ns per 256-value
block, so Q4_K's lower GB/s is only its smaller block, not its access pattern.
The kernels are bound by per-block instruction work shared by all four
(decode, scale math, eight lanes per block, input loads); shapes with 320
threadgroups run ~10 % slower per block than shapes with 1,088 (occupancy).
The bandwidth floor corresponds to ~0.5 ns per block.

KERN-03 adds Q3_K and IQ3_S with the same four-row geometry. Both use a
compile-time variant of `nu_matvec_three`; the encoding choice introduces no
runtime branch. Q3_K assembles four biased codes in a word and folds the
bias of four into each 16-value input sum. IQ3_S loads eight grid indices
and their ninth bits, applies the per-value signs, and factors the odd group
scale out of the dot product. Both follow the local dequantizers.

2026-09-08, same micro-benchmark methodology and hardware, ReleaseSafe,
best GB/s (generic column is the 248,320×5,120 case):

| Encoding | 69,632×5,120 | 20,480×17,408 | 248,320×5,120 | generic kernel |
| --- | ---: | ---: | ---: | ---: |
| Q3_K | 124.1 | 121.5 | 121.0 | 43.2 |
| IQ3_S | 122.6 | 121.0 | 119.0 | 42.5 |

MODL-08 (2026-09-12) adds Q4_0, the only weight encoding of the catalogue's
Gemma 4 12B file, with the same 128-thread geometry: lane `l` of a SIMD
group takes the 18-byte blocks `l`, `l + 32`, … of its rows (the same
block order as an octet taking block `8·kb + g` of stride `kb`, which is
how it was first written; KERN-09 made the loop walk blocks so a row of
22 blocks — the 26B-A4B's expert down projection — no longer falls back
to `nu_matvec`), two `packed_ushort4` loads (2-byte alignment) give the
four nibble words, and the bias of eight folds into the per-block input
sum exactly as Q6_K's 32 does (`Σ d·(q−8)·x = d·(Σq·x − 8·Σx)`). The
one-hot fixture columns are exact through it, the randomized rows
(including a 22-block row) within 4e-6 of the F64 reference through
both kernels. The block walk costs nothing on whole-stride rows: on
2026-09-17 the 5,120 × 17,408 shape alternated stride loop / block loop /
stride / block at 173 / 182 / 199 / 206 GB/s as the GPU warmed (each pair
at or above its predecessor), the three larger shapes at 230 / 211 / 229
against the 2026-09-12 record above, and the QAT F32 comparison was
unchanged (max abs 7.7e-5, relative RMS 3.2e-6).

2026-09-12, the same micro-benchmark, hardware, and build mode (the other
encodings re-measured in the same run: Q4_K 179 / 158 / 180 / 149, Q6_K
252 / 245 / 252 / 239, IQ4_XS 214 / 192 / 218 / 187 on the four shapes,
within noise of the tables above), best GB/s of weight bytes:

| Encoding | 69,632×5,120 | 20,480×17,408 | 248,320×5,120 | 5,120×17,408 | generic kernel |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q4_0 | 228.8 | 211.7 | 229.6 | 206.5 | 114.8–119.7 |

Per byte the Q4_0 kernel is the fastest of the set (18 bytes carry 32
values with one scale and no group coefficients to unpack: 0.9 ns per
256-value stride is 144 bytes here against 176 for Q5_K), and the
generic kernel on Q4_0 is also the fastest generic case for the same
reason. In the QAT Gemma file every matrix takes this kernel.

### Ternary matvecs and tiles (KERN-10, 2026-09-18)

Bonsai 2 27B's PQ2_0 (id 142, 34 B per 128 values) and PTQ1_0 (id 143,
28 B; [bonsai.md](bonsai.md#encodings-from-the-forks-ggml-commonh-and-ggml-quantsc))
got `nu_dequant_pq2_0` / `nu_dequant_ptq1_0` (and `nu_dequant_bf16` for
the file's `ssm_alpha` / `ssm_beta` rows) in the generic library,
`nu_matvec_pq2_0` / `nu_matvec_ptq1_0`, and the `nu_tile_*` decoders
behind `nu_matmul_pq2_0`, `nu_matmul_ptq1_0`, and their `_32`
instantiations. The matvecs keep the set's geometry (four SIMD groups of
four rows) with **four lanes per 128-value block, eight blocks per
iteration**, every lane running the same code on its quarter so the
digit-major PTQ1_0 layout costs no divergence: a PQ2_0 quarter is 32
consecutive values (eight bytes, one `packed_ushort4` at two-byte
alignment); a PTQ1_0 quarter is four bytes of the 16-byte run, two of the
8-byte run, and one digit of the two tail bytes (one `uint` at four-byte
alignment, two `ushort`), 20 + 10 + 2 values in the strided order the
layout gives them. `w = d · t` factors as `d · (Σ t·x − Σ x)` with the raw
code as `t`, exact for a one-hot input, which is what keeps the fixture
columns exact. Two-bit fields are masked four bytes at once
(`(w >> 2k) & 0x03030303`: elements k, 4+k, 8+k, 12+k) and paired with the
inputs in that strided order — a free re-labelling of registers — which
took PQ2_0 from 105 to 119 GB/s; trits come out of 16-bit slots two
bytes at a time (`((w & 0x00ff00ff) · 3ⁿ) & 0x00ff00ff`, then bits 8–9 of
`q · 3`). `specializedMatvec` accepts any whole-block row at the two
alignments; the tiles decode eight segments per block with the generic
expression in the generic order (bit-identical to the F32 tile before the
half rounding, as the others).

2026-09-18, the same micro-benchmark (Q4_0 re-measured in the run: 230.3
/ 211.2 / 230.7 / 207.0), best GB/s of weight bytes:

| Encoding | 69,632×5,120 | 20,480×17,408 | 248,320×5,120 | 5,120×17,408 | generic kernel |
| --- | ---: | ---: | ---: | ---: | ---: |
| PQ2_0 | 118.5 | 101.1 | 118.7 | 96.9 | 47.6–49.4 |
| PTQ1_0 | 88.0 | 87.6 | 88.5 | 83.3 | 29.6–30.3 |

**The bytes are not the bound.** The design's target was the Q4_0 set's
200 GB/s; these kernels reach about half. In values per second they
are at the set's ceiling — PQ2_0 448 G values/s on the output head,
PTQ1_0 405, Q4_0 410 — and a ternary byte carries twice the values of
a Q4_0 byte (3.8 against 1.8), so at the same multiply rate the byte
rate halves. The per-value cost of this design (a field mask shared by
four values, one integer-to-float conversion, one FMA) is the floor
the structure has; going past it needs different arithmetic (packed
integer products, or sharing one decoded value across several inputs),
which is a follow-up measured against the model rate MODL-17 records,
not against this table. PTQ1_0 pays its five-digit extraction: 12 % fewer
bytes than PQ2_0 at 25 % fewer bytes per second, so it decodes slower;
whether the catalogue moves to it is decided by MODL-17's measurement.
At batch 1 the transform of the activations (KERN-10 session 2) adds
to the token, not to these numbers.

`make bench-matmul` (256 tokens), best ms and the GFLOP/s of the tile:
PQ2_0 9.18 ms / 4,971 on 17,408×5,120 and 9.73 / 4,689 on 5,120×17,408,
PTQ1_0 9.69 / 4,711 and 10.78 / 4,234, Q4_0 in the same run 9.66 / 4,724
and 10.05 / 4,540: the prefill tiles are compute-bound and the ternary
tiles match the set.

### The Hadamard transform kernel (KERN-10 session 2, 2026-09-18)

`nu_hadamard` is the activation side of the folded rotation
(`cpu.hadamard`, [bonsai.md § Rotation](bonsai.md#rotation-prismhadamard-as-the-forks-loader-reads-it)):
in place over `rows` rows of `width` floats at a stride, per 1,024-block,
forward `x = H (signs ⊙ x)` or inverse `x = signs ⊙ (H x)`, scale 1/32
exactly. One 256-thread group per block of one row; each thread owns four
consecutive values, does the first two butterfly stages in registers and
the other eight through 4 KB of threadgroup memory, two pairs per stage
with a barrier between stages. `Backend.hadamard(data, signs, width,
rows, stride, inverse)` requires a float4-aligned data and sign buffer
and a width that is a multiple of the block. `metal-check` compares it
with `cpu.hadamard` (F64 butterflies) on 5,120 / 6,144 / 17,408 over
strided rows, forward, inverse alone, and the round trip: max abs
7.2e-7 (bound 3e-5).

`make bench-hadamard` (`metal-check --hadamard-bench`) issues one
Bonsai token's transforms at batch 1 — per layer the four rotated
activations (5,120, 6,144, 5,120, 17,408) over 64 layers, plus the
embedding inverse and the output-head input: 258 dispatches, 2,197,504
elements — in one command buffer. 2026-09-18, M4 Pro, ReleaseSafe, best
/ mean of five: **1.34 / 2.20 ms per token**, against 0.051 µs per
1,024-block inside a 64-row 17,408-wide dispatch (0.11 ms of arithmetic
for the whole token). The cost is launch and dependency, about 5 µs per
dispatch, and at the ternary matvecs' rate it is 2–3 % of a token. The
fusion the plan asked to measure — the sign flip and transform inside
the matvec's input load — is decided by that arithmetic: every SIMD
group of a matvec would recompute its block's transform (thousands of
times per projection against once), so it is not attempted. What a
fusion can save is launches, by folding the transform into the norm
that precedes two of the four activations per layer; whether the 2–3 %
justifies it is read from MODL-17's per-kernel profile of the whole
token, not from this number.

### The rotation on the Qwen plan (MODL-17, 2026-09-18)

`qwen35_metal.zig` applies the transform where the CPU reference does
([bonsai.md § Metal plan](bonsai.md#metal-plan-modl-17-2026-09-18)): the
inverse over the embedding row(s) after the gather, the forward over the
normed residual before the mixer's projections, over the mixer output
before `attn_output` / `ssm_out`, over the normed residual before the FFN
pair, over the FFN hidden before `ffn_down`, and over the output-head
input — in place, since nothing reads those buffers afterwards except the
projection itself; the DeltaNet `ssm_alpha` / `ssm_beta` projections are
dispatched before the residual is transformed (the four-way merged matvec
of the plain file splits into two two-way merges around the transform,
the plain file keeps its one dispatch). The `ssm_out` input is regathered
first from the mixer's tiled head order into the fold's grouped order by
`nu_gather_rows` (`Backend.gatherRows`: `dst[r][g] = src[r][map[g]]` over
rows of `groups` vectors, one thread per element, the 48-entry map
uploaded once at plan init), into a scratch of one decode row and
`padded` prefill rows, then transformed there. `metal-check` proves the
gather exact on the 48 × 128 regrouping over strided rows and its
refusals (overlap, a short map, a short stride).

On the whole token (`make bench-profile MODEL=bonsai-2-27b`, 22-token
prompt, 2,048 context, 192 measured steps): 1,292 dispatches per step,
79.9 ms attributed of 85.5 ms command-buffer time; the 258 transforms take
**1.66 ms (2.1 %)** and the 48 gathers 0.16 ms (0.2 %), the matvecs about
80 % at 97–116 GB/s of weight bytes (`matvec_segments` on the merged
projections 102–113, the PQ2_0 FFN down 97, the output head 116), the
norms 2.1 ms (2.6 %). So the fusion question the transform kernel left
open is answered for now: folding the transform into the norm that
precedes two of the four activations would save at most about 1 % of the
token, and the token is bound by the ternary matvecs' multiply rate
(§ Ternary matvecs and tiles), which is where the next kernel work goes.

### Gathered expert kernels (KERN-09)

A mixture-of-experts layer stores each expert projection as a 3-D tensor
`[experts][rows][columns]` (GGUF dimension 2 is the expert), the experts
contiguous, and a token reads only the `k` experts its router selected.
The decode path is four dispatches per layer, every one a parameter of an
existing contract:

- `Backend.route`: one 256-thread group per row of logits. The row's max
  and exp-sum come from `simd_max`/`simd_sum` and an 8-way threadgroup
  combine; then `k` rounds of "best untaken" select by **logit** (value
  desc, index asc; the comparison is on the logits, not the
  probabilities, so `exp` rounding cannot reorder near-ties and the
  indices match the F64 reference exactly), the winner records its
  probability, and the selected probabilities are divided by their sum
  clamped below at the smallest F16 normal, as the reference does. At
  most 256 experts (one logit per thread) and 64 selected.
- `Backend.matvecExperts`: threadgroup `g` serves slot `g / row_groups`
  and 16-row block `g % row_groups` of that slot's expert, whose bytes
  start at `expert · rows · stride`; the lane arithmetic is
  `nu_segment_sums` (the specialized body of the encoding under the
  matvec alignment rules, otherwise the generic decoder), so a selected
  expert costs the bytes of a dense matrix of its size and nothing of
  the other experts is read. `in_stride` 0 shares one input across the
  slots (gate-up); the down projection gives each slot its own hidden
  row. Expert indices are clamped to the tensor so a corrupt routing
  buffer cannot read past it.
- `Backend.geluMulRows`: `gelu(gate[r][i]) · up[r][i]` over strided rows;
  with the reference's fused gate-up rows (gate rows first, then up rows,
  per expert) `up` is the gate buffer sliced at the up half.
- `Backend.combineExperts`: `out[row][c] = Σ_s w[row][s] · scale[e_s] ·
  y[row·k + s][c]`, the per-expert down scale optional; F32 `fma` over
  the slots.

Evidence (`test-metal`): the router on 128 experts (k 8, six strided rows
with ties at the top and at the k boundary, an all-equal row, a row of
40× spread) and on 37 experts (k 5): indices exact, weights within
**1.5e-8** (bound 1e-6). The gathered Q4_0 matvec on six experts of
40 × 1,280 with slots `[5, 0, 5, 2]`, shared and per-slot inputs, both
kernel paths, within the dense matvec tolerance (4e-6 of Σ|w·x|); with
NaN written into every scale of the unselected experts the selected
outputs are bit-identical. A dense F32 tensor of 3 × 8 rows takes the
generic branch (rows below one group). The whole decode chain over three
tokens (six experts, width 256, ff 704, k 3, with and without the down
scale) against `cpu.experts.ffn` in F64: worst **1.5e-8** of
(1 + max|y|), bound 2e-5. Contract violations (k > experts, more than
256 experts, unaligned indices, a slot input stride that is not a
multiple of four floats, zero slots, an expert count that does not
divide the tensor, the pair output aliasing its inputs, short weight
or scale buffers) are refused before dispatch.

`make bench-experts` (`metal-check --experts-bench`, ReleaseSafe, M4 Pro,
2026-09-17): the 26B-A4B shape — 128 experts, 8 selected, gate-up
1,408 × 2,816 and down 2,816 × 704 per expert, Q4_0 — 64 dispatches per
command buffer, best / mean of five, GB/s of the **selected** experts'
bytes, beside a dense matvec over the same byte count:

| Case | MB read | best | mean |
| --- | ---: | ---: | ---: |
| gathered gate-up, 8 × (1,408 × 2,816) | 17.8 | 215.1 | 203.5 |
| gathered down, 8 × (2,816 × 704) | 8.9 | 154.2 | 151.5 |
| chain: gate-up, gelu rows, down, combine | 26.8 | 171.5 | 171.2 |
| dense 11,264 × 2,816 (the gate-up bytes) | 17.8 | 216.6 | 214.4 |
| dense 2,816 × 5,632 (the down bytes) | 8.9 | 206.7 | 203.3 |

The gathered gate-up runs at the dense rate. The down projection is at
three quarters of it because a 704-column row is 22 blocks: the Q4_0
body gives one block per lane, so 10 of 32 lanes idle in its single
pass (before the block walk it fell back to the generic decoder at
74 GB/s). A lane mapping for short rows (two rows per octet, or lanes
splitting a block) is the follow-up if the per-token profile puts the
down projection above its floor: at 8 selected experts × 30 layers the
chain reads about 0.8 GB per token, roughly 5 ms at this rate.

**Prefill.** A chunk of `chunk` tokens has `n = k · chunk` slot rows
(token `t`, slot `s` is slot row `t · k + s`, the order `route` writes),
and the point of the batched path is to read each expert's weights once
per group of its rows instead of once per row. Two kernels, no host
round trip inside the chunk:

- `Backend.expertLists`: one 256-thread group turns `indices` into the
  *lists* buffer (`expertListsLayout`): a threadgroup histogram over the
  experts (atomics), the exclusive prefix sum, a serial pass by one
  thread that writes the per-expert offsets and the **tile list** — for
  each expert, one entry (expert, first, count) per 32 of its rows — and
  a scatter that writes the **row list**, each expert's slot rows
  contiguous from its offset. Σ ceil(count / 32) ≤ n / 32 + experts
  (`expertTileBound`), so the host sizes the matmul grid before the
  counts exist and tiles past the count exit at their first instruction.
  The order within an expert comes from the atomics and is not
  deterministic; every row's result is computed on its own, so the output
  is. Indices are clamped as in the decode kernel.
- `Backend.matmulExperts`: `nu_matmul_body`, the dense tile's body with a
  `GATHER` flag. Threadgroup `g` serves tile `g / row_tiles` and row tile
  `g % row_tiles` of that tile's expert (bytes at `expert · rows · stride`).
  The tile's 32 activation rows are the slot rows `row_list[first ..
  first + count]`, read from input row `slot_row / in_group` (`in_group`
  = k on the gate-up projection, where a token's k slots share its
  input; 1 on the down projection, one hidden row per slot); rows past
  `count` stage zeros. The accumulators go through threadgroup memory
  (`tile` reused: 32 × 64 F32 is exactly 64 × 64 half) and each valid
  row is scattered to output row `slot_row`; nothing is written past the
  count, so the scratch holds exactly `n` rows and needs no padding. Two
  instantiations: 64 × 32 with half operands for Q4_0 under the matvec
  alignment rules, the generic F32 32 × 32 tile for every other encoding
  or alignment. The dense `nu_matmul` instantiations are the same body
  with the flag off. The profile attributes no bytes to this kernel: what
  it reads depends on the routing.

The chain per layer is then `route` → `expertLists` → `matmulExperts`
(gate-up, `in_group` k) → `geluMulRows` over `n` rows → `matmulExperts`
(down, `in_group` 1) → `combineExperts` with `rows = chunk`.

Evidence (`test-metal`): over a chunk of 45 tokens (135 slot rows, not a
multiple of 32) with six experts and k 3, expert 0 forced onto every
token (two tiles, one partial) and expert 5 onto none: the lists checked
on the host (the skew, the offsets, every tile's expert / first / count,
the row list a permutation of the slot rows grouped by expert); the
outputs against `cpu.experts.ffn` per token and against the decode path
row by row — the generic F32 tile within **1.1e-6** of max|y| (bound
2e-5), the half tile **4.4e-4** (bound 2e-3: both operands of both
projections rounded to half); a chunk of one token (three one-row tiles);
rows not a multiple of 8, a short lists buffer, an unaligned input
stride, a zero input group, and 257 experts refused before dispatch.

`make bench-experts ARGS=<chunk>` (ReleaseSafe, M4 Pro, 2026-09-18), the same
26B-A4B shape, the router over random logits (every expert touched), GB/s
of the bytes the tiles read (one expert matrix per tile) and the time per
chunk, 64 chains per command buffer, best of five:

| Chunk | Slot rows | Tiles | Fill | gate-up GB/s | down GB/s | chain per chunk | tok/s if 30 layers were the whole cost |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 2,048 | 128 | 50 % | 41.8 | 41.1 | 10.5 ms | 809 |
| 512 | 4,096 | 186 | 69 % | 42.0 | 41.2 | 15.9 ms | 1,074 |
| 1024 | 8,192 | 315 | 81 % | 40.7 | 39.9 | 27.0 ms | 1,264 |

The rate per tile byte is the dense tile's: at 256 tokens the gate-up
tiles execute 32.5 GFLOP in 6.8 ms, 4.8 TFLOP/s, against 5.0 TFLOP/s for
the dense 64 × 64 Q4_0 tile (`make bench-matmul`); the tile is
compute-bound, not bandwidth-bound, at 32 tokens. What the number hides
is the **fill**: with 2,048 slot rows over 128 experts an expert averages
16 rows, so half of a 32-row tile multiplies zeros, and the useful rate
is half the executed one. Larger chunks fill the tiles (the third column
is `n / (32 · tiles)`), which is why the per-token cost falls with the
chunk while the GB/s does not move; the adapter's chunk size for this
family is a memory/latency trade the Metal plan decides (512, measured in
[gemma4.md § 26B-A4B](gemma4.md#gemma-4-26b-a4b-the-expert-configuration-modl-09)). Against the
alternative of looping the decode kernels over the chunk (2,048 × 3.3 MB
= 6.8 GB per layer at the decode rate, about 40 ms), the tiles are four
times faster at 256 tokens and six at 1,024. Follow-ups, not in scope: a
16-row token tile for sparse chunks (the dense tile's rate falls with
the token width, so the gain is unclear), a 64-token tile for chunks
where experts exceed 32 rows, and Q4_K / Q6_K instantiations for other
families.

**In the adapter** (MODL-09, 2026-09-18). `gemma4_metal.zig` records the
decode chain after the dense FFN of every 26B-A4B layer and the prefill
chain over each chunk's rows, both from the same `feedForward` shape as
the CPU reference ([gemma4.md § 26B-A4B](gemma4.md#gemma-4-26b-a4b-the-expert-configuration-modl-09)).
The three expert tensors are wrapped whole (no copy, 14.2 GB resident)
and the router, an F32 128 × 2,816 matrix, is the first F32 matrix a plan
dispatches: the generic matvec and the generic F32 tile decode it
exactly. One property of the model surfaced through the generation
check: a mixture of experts routes **discretely**, so the half-operand
rounding of the specialized tiles, which moves a dense model's chunked
logits by percents, here moves router logits past near-ties and swaps a
token's eighth expert for another, replacing that token's whole expert
output. On the 70-random-token prompt the chunked prefill sits at
1.1e1 / 3.5e-1 (max abs / relative RMS) from the per-token steps with the
specialized tiles, 1.1e-1 relative RMS with the expert projections
alone through F32 matvecs (the dense tiles' rounding still routes), and
**2.8e-4 / 1.2e-5** with every tile generic: the chunk schedule (row
routing, lists, gathered tiles, combine) is exact, and the gap is the
rounding class shared with the dense tiles, amplified by routing. The
greedy token agreed in every comparison. The check records the expert
configuration's own bounds (2e1 / 5e-1) beside the 12B's and keeps the
F32-tile bound (5e-3 / 2e-4) unchanged, which is the one that proves the
schedule.

### KERN-05 — per-block cost research (2026-09-08, closed without a kernel change)

The unit spent one session on the plan's five hypotheses against the 0.8–0.9
ns/block observation. Method per hypothesis: fixtures (`make test-metal`) →
`make bench-kernels` → keep only a ≥ 5 % gain. Nothing met the bar, so the
kernels are byte-identical to KERN-04; the outputs are the harness changes above
(the true 5,120×17,408 down-projection shape and the longer batches for it),
the `make trace` target, and the results below. Baseline for the session
(HEAD kernels, new harness, best GB/s of five, block kernel / generic):

| Encoding | 69,632×5,120 | 20,480×17,408 | 248,320×5,120 | 5,120×17,408 |
| --- | ---: | ---: | ---: | ---: |
| Q4_K | 179.2 / 109.0 | 157.7 / 108.2 | 179.6 / 109.4 | 152.4 / 104.3 |
| Q5_K | 211.1 / 99.9 | 202.8 / 99.3 | 214.3 / 100.2 | 190.2 / 96.5 |
| Q6_K | 251.9 / 81.8 | 245.7 / 80.5 | 252.6 / 81.8 | 239.4 / 79.5 |
| IQ4_XS | 214.3 / 99.1 | 192.2 / 98.6 | 218.3 / 99.4 | 186.5 / 95.8 |
| Q3_K | 123.5 / 44.9 | 120.8 / 44.7 | 124.1 / 44.8 | 117.6 / 43.8 |
| IQ3_S | 122.6 / 44.1 | 121.0 / 43.2 | 123.4 / 43.9 | 114.4 / 42.8 |

Rejected, with the Q4_K/Q5_K best GB/s on the same four shapes:

1. **ILP — two blocks per lane iteration with independent accumulators.**
   Q4_K 115.5 / 99.7 / 111.1 / 90.6 and Q5_K 157.3 / 146.8 / 149.6 / 137.9:
   about 35 % slower. Two blocks' decoded `float4`s and two accumulator sets
   are live at once; the register footprint costs more occupancy than the
   second dependency chain hides. The kernels are already latency-hidden by
   the four rows and by other SIMD groups, not by in-lane parallelism.
2. **Input staging in threadgroup memory.** The whole vector staged once per
   threadgroup when it fits (5,120 columns = 20 KB; 17,408-column shapes left
   unstaged). Staged shapes: Q4_K 170.6 / – / 171.3 / – against 184–186 in
   the same tree, Q5_K 198.3 / – / 199.1 / – against 204–211; the unstaged
   shapes did not move. The copy, the barrier, and 20 KB of threadgroup
   memory per 128-thread group (which caps resident groups) cost more than
   the cached `float4` input loads they replace. Input traffic is measured,
   not inferred, to be off the critical path.
3. **Two rows per SIMD group for shapes under 8,192 rows** (`nu_matvec_q4_k<2>`
   selected by the encoder; the pre-migration working state). On 5,120×17,408
   Q4_K: 142.8 against 152.4 baseline and 159–160 with the four-row body in
   the same tree, −6 %. Doubling the threadgroups (320 → 640) halves the
   input reuse per lane, and that loss exceeds the occupancy gain on this
   shape.
4. **Scale-decode hoisting.** Two forms. (a) The even lane of each lane pair
   decodes the pair's two group scales and the odd lane takes them by four
   `simd_shuffle`s (the pre-migration form, Q4_K only): the one positive
   signal, a repeatable +3.3 / +5.6 / +4.0 / +3.9 % over two off/on/off/on
   rounds (178.2→184.3, 160.5→168.7, 178.7→184.7, 154.4→159.9), below the
   5 % bar on three of four shapes and at most ~1 ms/token end to end;
   reverted by the rule. (b) Each of the eight lanes of a block decodes one
   group with half the select work and the pair's groups arrive by four
   shuffles with computed lane indices: Q4_K 158.1 / 139.4 / 153.1 / 133.6,
   −12 %, while Q5_K moved +0.3 / +4.3 / −1.2 / +5.1 % (noise). Strictly fewer
   ALU instructions made Q4_K slower, so instruction count is not what bounds
   it; the masked decode in (a) costs the same ALU as decoding on every lane
   and still won, which points at scheduling/latency, not issue.
5. **Diagnostic: `MTLMathModeFast`** (throwaway build, never shipped). Q4_K,
   Q6_K, Q3_K, IQ3_S, and IQ4_XS unchanged within noise; Q5_K fell to
   185.1 / 166.0 / 180.1 / 159.0 (−12 %). Contraction has nothing left to
   contract (the dot products are explicit `fma`), and the reassociation it
   licenses hurt Q5_K. ALU issue is not the limiter, and safe mode stays.

Instruments: `xctrace` runs from the command line through `DEVELOPER_DIR`
(see [AGENTS.md](../../AGENTS.md) § Local toolchain notes, `make trace`).
On this M4 Pro the "Metal GPU Counters" instrument reports *Selected counter
profile is not supported on target device*; the `gpu-counter-value` and
`metal-gpu-counter-intervals` tables export empty, while `metal-gpu-intervals`
(the dispatch timeline) records normally. The limiter question therefore
stays open at the counter level: what the session established is that it is
neither ALU count, nor input traffic, nor contraction, and that the kernels
are sensitive to register footprint (H1) and to instruction scheduling (H4b
vs H4a). The remaining candidates are load-latency exposure per block and
the eight-lanes-per-block reduction structure, which only a different
geometry (e.g. one SIMD group per block with a wider reduction) would test.

`make bench` after the session, kernels unchanged: 10.64 tok/s decode
([bench.md](bench.md)).

## Prefill in chunks (ENGN-02)

`Plan.prefill(tokens, …)` consumes a prompt in chunks of up to `chunk` tokens
(256 from the engine, clamped to the session capacity), one command buffer
per chunk. Inside a chunk the projections and the feed-forward network run
through `nu_matmul` on `[token][feature]` activation buffers sized for the
chunk (about 0.4 MB per token, padded to a multiple of 64 rows; `step` keeps
its own single-token buffers, so the two paths never alias), the norms, RoPE (`nu_rope_rows`), SiLU, gates,
and residual adds run over `count` rows at once, the DeltaNet convolution runs
as one causal pass over the chunk plus a history update
(`nu_convolution_rows`, `nu_convolution_history`; bit-identical to the
sequential kernel by construction), attention runs as **one causal tiled
dispatch per layer** (`nu_attention_chunk`, ENGN-03) after the chunk's keys and
values were projected, normalized, rotated, and copied into the cache in one
go, and DeltaNet runs as **one chunkwise dispatch per layer**
(`nu_delta_chunk`, ENGN-04) over 32-token sub-chunks. Nothing steps per token
inside a chunk any more.

Contract: the whole prompt must fit the remaining context (refused before any
work), the observer's `check` runs between layers of every chunk, and a
per-layer `layer` observer is refused because it is a per-token contract —
`Model.prefill` steps token by token in that case, which is how `--trace-dir`
and `make compare` keep their exact per-token semantics. The tile matmul
accumulates in a different order than the matvec, so a chunked prompt and the
same prompt stepped agree within a tolerance rather than bit for bit:
`generation-check --metal` runs a 70-token prompt both ways with 32-token
chunks (two full chunks and a partial one, so chunk boundaries fall inside
attention and DeltaNet state) and measured max abs 3.1e-5, relative RMS
1.1e-6, identical argmax (2026-09-08; bound 2e-2 / 1e-3).

Measured (2026-09-08, M4 Pro, ReleaseSafe, `bench --raw`, 32 output tokens,
context 4096, one warmup and two runs; the 22-token row is the standard
`make bench` workload at context 2048):

| Prompt tokens | Prefill tok/s | Before ENGN-02 | Reference (llama.cpp) | Decode after the prompt |
| ---: | ---: | ---: | ---: | ---: |
| 22 | 35.0 | 11.1 | — | 10.79 |
| 543 | 51.3 | ~11 | 89.2 at 512 | 10.50 |
| 3,547 | 33.3 | ~11 | 89.3 at 4,096 | 7.76 |

The rate fell with prompt length because the per-token attention loop grew
with the visible cache (3 dispatches per token per layer). ENGN-03 (2026-09-09,
same methodology, prompts of 545 and 3,657 tokens) removed that fall-off:

| Prompt tokens | Prefill tok/s | ENGN-02 | Reference (llama.cpp) | Decode after the prompt |
| ---: | ---: | ---: | ---: | ---: |
| 22 | 35.3 | 35.0 | — | 10.47 |
| 545 | 51.4 | 51.3 | 89.2 at 512 | 10.36 |
| 3,657 | 50.6 | 33.3 | 89.3 at 4,096 | 8.48 |

`generation-check --metal` (70-token prompt, 32-token chunks) after ENGN-03:
max abs 2.5e-5, relative RMS 1.2e-6, identical argmax (bound 2e-2 / 1e-3).

ENGN-04 (2026-09-09) replaced the last per-token loop, `nu_delta`, with
`nu_delta_chunk`; `generation-check --metal` after it: max abs 2.8e-5,
relative RMS 1.2e-6, identical argmax. Same methodology, plus a 13,399-token
prompt at context 16,384 (`--repeat 1`):

| Prompt tokens | Prefill tok/s | ENGN-03 | Reference (llama.cpp) | Decode after the prompt |
| ---: | ---: | ---: | ---: | ---: |
| 22 | 34.9 | 35.3 | — | 10.33 |
| 545 | 52.6 | 51.4 | 89.2 at 512 | 10.28 |
| 3,657 | 53.0 | 50.6 | 89.3 at 4,096 | 8.61 |
| 13,399 | 42.6 | — | 74.1 at 16,384 | 4.67 |

Prefill then sat on the matmul tile's ceiling of 42–57 tok/s
(`make bench-matmul`, generic decoder in the tile). The fall at 13K is
chunk attention over a 13K-row F32 cache (KERN-07/KERN-08); decode at 13K context
(4.67 vs the reference's 7.32) is the decode attention over the same cache.

ENGN-05 (2026-09-09) raised that ceiling to 90–96 tok/s with specialized
half-operand 64×64 tiles ([§ Kernels](#kernels)). Same methodology and
prompts; the 16K row was not rerun because ENGN-05 does not touch attention:

| Prompt tokens | Prefill tok/s | ENGN-04 | Reference (llama.cpp) | Decode after the prompt |
| ---: | ---: | ---: | ---: | ---: |
| 22 | 39.1 | 34.9 | — | 10.30 |
| 545 | 83.8 | 52.6 | 89.2 at 512 | 10.15 |
| 3,657 | 81.7 | 53.0 | 89.3 at 4,096 | 8.63 |

`generation-check --metal` after ENGN-05 (70-token prompt, chunks of 64, 48,
and 32 so both tile sets are exercised): max abs 2.37e-3 / 2.32e-3 /
2.59e-3, relative RMS 1.1–1.2e-4, identical argmax (bound 2e-2 / 1e-3). The
step from 2.8e-5 to 2.5e-3 is the half rounding of both operands; the
CPU-vs-GPU `make compare` (stepped, matvec path) is unchanged at 1.22e-4.
Prefill is within 6–9 % of the reference at 512 and 4K; the remaining gap
is spread over attention, DeltaNet, norms, and the generic-tile tensors
(7 IQ4_NL and 106 Q8_0 tensors in this artifact).

### Small-chunk tile (KERN-11, 2026-09-19)

A prompt's first tokens run through `Backend.matmul` as one short chunk. The
32×32 tile gives a 5,120-row projection 160 threadgroups whose K loop carries
two `threadgroup_barrier`s per 64-column step, so at 22–32 tokens it streams
weight bytes at 10–41 GB/s while the specialized matvecs on the same encodings
reach 88–246. The `nu_matmul_*_8` set gives one 128-thread group **16 rows ×
8 tokens**: two 8×8 accumulators per SIMD group share one activation block per
step, the four SIMD groups partition the 64-column K steps, each lane decodes
two 16-value segments into its own group's 16×64 half tile, and only
`simdgroup_barrier` orders the loop — no `threadgroup_barrier` inside it. The
four K partials reduce once at the end in a fixed SIMD-group order.
`specializedMatmul` takes the `_8` tile for `tokens <= small_chunk_tokens`
(24), the `_32` tile above it; at 9–24 tokens the `_8` tile runs two or three
token tiles and still beats one 32-token tile in wall-clock (22 tokens: 1.55 ms
against 1.83 ms for Q4_K, measured with the threshold at 32).

Method: `make bench-matmul ARGS=<t>` (Apple M4 Pro, Zig 0.16.0, ReleaseSafe).
Each row is five measured command buffers after two warm-ups, each buffer
issuing 64 dispatches at t ≤ 8 and 16 above, timed with `gpuSeconds()` and
divided by the dispatch count — the batch keeps the GPU clocked where an
isolated dispatch measures ramp-up. Weights are read once per token tile
(`region.len × ceil(t / tile_tokens)`), the count `Backend.matmul`'s profile
attributes. The matvec column is the same encoding from `make bench-kernels`
run immediately before on the same machine (69,632×5,120 for the gate shape,
5,120×17,408 for the down shape). GB/s of weight bytes:

**ffn_gate (17,408×5,120).** t=1..22 take the 16×8 tile, t=32 the 32×32.

| Encoding | matvec | t=1 | t=4 | t=8 | t=9 | t=16 | t=22 | t=32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q4_K | 179 | 93 | 96 | 95 | 98 | 95 | 97 | 28 |
| Q5_K | 211 | 108 | 109 | 108 | 111 | 113 | 114 | 33 |
| Q3_K | 122 | 64 | 62 | 62 | 64 | 64 | 63 | 21 |
| Q6_K | 246 | 115 | 115 | 113 | 116 | 115 | 114 | 40 |
| IQ3_S | 121 | 66 | 66 | 65 | 66 | 66 | 67 | 21 |
| IQ4_XS | 214 | 87 | 90 | 86 | 88 | 92 | 91 | 28 |
| Q4_0 | 230 | 90 | 89 | 89 | 91 | 92 | 92 | 29 |
| PQ2_0 | 116 | 47 | 48 | 48 | 50 | 50 | 49 | 14 |
| PTQ1_0 | 88 | 32 | 33 | 33 | 34 | 34 | 34 | 11 |

**ffn_down (5,120×17,408).** Same tile assignment.

| Encoding | matvec | t=1 | t=4 | t=8 | t=9 | t=16 | t=22 | t=32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q4_K | 151 | 91 | 94 | 92 | 95 | 97 | 97 | 26 |
| Q5_K | 189 | 105 | 106 | 106 | 110 | 110 | 111 | 32 |
| Q3_K | 117 | 58 | 58 | 59 | 61 | 61 | 60 | 20 |
| Q6_K | 238 | 106 | 106 | 109 | 112 | 110 | 113 | 38 |
| IQ3_S | 113 | 62 | 60 | 61 | 65 | 65 | 65 | 21 |
| IQ4_XS | 184 | 87 | 86 | 84 | 90 | 88 | 91 | 26 |
| Q4_0 | 205 | 87 | 88 | 88 | 92 | 90 | 91 | 28 |
| PQ2_0 | 97 | 46 | 46 | 47 | 49 | 49 | 50 | 13 |
| PTQ1_0 | 84 | 31 | 31 | 32 | 33 | 33 | 33 | 10 |

**Activation-operand experiments.** Each was measured alone under the batched
bench at 8 tokens before the threshold moved:
- **16 rows per threadgroup, kept.** Two accumulators per SIMD group share one
  B load, halving the gathered activation traffic per weight byte. It moved
  Q4_K from 75 to 95 GB/s and every encoding up 4–32 %, most where the weight
  bytes per activation byte are fewest (the K-quants). The shipped `_8` tile is
  this geometry.
- **Packed `[k][token]` half activations, dropped.** A transposing sibling of
  `nu_pack_half` (one per projection input) reached 62–67 GB/s at 8 tokens
  against the 16-row tile's 90+ on the same encodings, before its one-time pack
  cost of 34–112 µs per input. A plain `[k][token]` layout leaves each 8×8 B
  block as eight 16-byte strided half loads, which the transposed float gather
  already served better; a blocked layout that makes the block one contiguous
  read is left as a follow-up.

**Reading.** The tile is flat across 1–22 tokens: 32–116 GB/s, 37–64 % of the
same encoding's matvec rate. At 9–24 tokens it runs two or three token tiles
and still matches t=8's bytes-per-second because the attribution counts every
pass; its wall-clock stays below the 32×32 tile's until the fourth tile
(t=25), which is why `small_chunk_tokens` is 24. It is still far below the 70 %
floor: the best rows are Q4_K's 52–64 % on the down shape; the ternary and
Q3/IQ3 rows sit at 37–52 %. The reason is the activation operand — a group
gathers 8 tokens × columns × 4 B (160 KB at 5,120 columns) against 16 × columns
of weights (46 KB of Q4_K) — and the 16-row tile only halved that ratio while
the packed layout made it worse. The remaining levers (32 rows per group, a
blocked activation read, or keeping the activations in threadgroup memory
across a row strip) are recorded in the log, not shipped; the unit closes
below its floor on this record.

### Multi-row matvec (KERN-12, 2026-09-20; closed below its target)

A verify batch (`1 + k` ≤ 8 rows), the recovery replay (`a + 1` rows), and the
decode-time commit (`a + 1` rows) are small batches that `Backend.matmul`
served with the 16×8 split tile. The `nu_matvec_rows_*` set is the one-weight-
pass alternative: each SIMD group owns four output rows (the matvec's mapping,
`kb = lane >> 3` selecting the 256-value block and `pair`/`half` the 32-value
slice), the token loop is innermost so a decoded slice serves every activation
row, and `acc[row][token]` is indexed by constants (`#pragma unroll`), which is
what keeps it in registers. Q4_K, Q5_K, Q6_K, and IQ4_XS have specialized
bodies; `nu_matvec_rows` (one SIMD group per row, a runtime token bound) serves
Q4_0 and the ternary encodings and is slower than the tile at every count. The
token count is a template parameter, one host name per encoding and count
(`nu_matvec_rows_q4_k_t2` … `_t8`): a runtime `tokens` loop with `break` moves
the accumulators to thread-local memory. `Backend.matmul` routes 2-row batches
to the specialized bodies (`small_batch_rows = 2`, `route_small_batch`);
`matvec_rows_max = 8` is the kernel range and `metal-check`'s sweep covers it.

Method: `make bench-matvec-rows ARGS=8` (Apple M4 Pro, Zig 0.16.0,
ReleaseSafe), two FFN shapes, three measured command buffers after a warm-up,
16 dispatches each, GB/s of weight bytes. Each cell is the gate shape / the
down shape. The historical tile column predates production routing.
The repaired sweep uses `Backend.matmulTile` explicitly at the same count;
`Backend.matmul` would now select the multi-row kernel at two tokens.

| Encoding | 2 rows | 3 | 4 | 5 | 8 | tile |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q4_K | **143 / 128** | 86 / 84 | 64 / 62 | 53 / 52 | 28 / 27 | 96 / 94 |
| Q5_K | **148 / 137** | 98 / 92 | 73 / 73 | 61 / 60 | 30 / 28 | 112 / 108 |
| Q6_K | **182 / 179** | **125 / 121** | 88 / 87 | 67 / 63 | 33 / 31 | 116 / 108 |
| IQ4_XS | **172 / 154** | **109 / 94** | 102 / 82 | 65 / 49 | 23 / 20 | 90 / 87 |

Bold beats the tile. The tested body wins at 2 rows for every encoding and,
for Q6_K and IQ4_XS only, at 3; at 5–8 rows it is slower than the tile and
misses acceptance. Both paths make one logical weight pass at these counts.
The scalar body's arithmetic and activation-load work grow with token count,
while the matrix tile computes eight token columns regardless.

A probe replacing the per-token input offset with a constant measured
181 GB/s flat from 2 to 8 rows. It does not isolate input-load cost: the compiler
can also merge identical dot products and accumulator recurrences. The real
body's marginal cost is ~0.19–0.22 ms per token on the 50 MB FFN
(~0.9 TFLOP/s of useful arithmetic). Tested layouts retaining more inputs or
decoded rows were slower, consistent with register pressure, but register
counts, spills and occupancy need compiler/profiler evidence. These experiments
justify stopping work on these scalar bodies, not an impossibility claim for
all scalar register tiles. The unit shipped two-row routing below its original
acceptance; the matrix tile and eliminating recovery replay are the next levers.

A reference-style `1 row per lane group × NT tokens` body was also measured and
rejected: it decodes a 256-value block header once per 16-value segment instead
of once per 32 values, and Q4_K fell to 55 GB/s at 2 rows.

#### Corrected two-row control (REPO-08, 2026-09-20)

`make bench-matvec-rows ARGS="2 head"`, Apple M4 Pro (48 GiB), Zig 0.16.0,
ReleaseSafe, revision `27303ed` plus the REPO-08 repair committed with this
record. Minimum of three measured command buffers after one warm-up, sixteen
dispatches per buffer; synthetic quantization fixtures, no model loaded. Rates
are logical weight bytes per GPU second, not measured DRAM traffic. No other
GPU benchmark ran concurrently.

The control calls `Backend.matmulTile`, which bypasses multi-row routing while
sharing production validation and tile selection. A profiler fixture asserts
that two-token Q4_K control and production calls dispatch `matmul_q4_k_8` and
`matvec_rows_q4_k_t2`, respectively. Head allocation and selection use the actual
aligned row stride; the previous `stride=1` capability check skipped every head.

| Encoding | FFN gate tile / rows GB/s | FFN down tile / rows GB/s | Head tile / rows GB/s |
| --- | ---: | ---: | ---: |
| Q4_K | 96.8 / 143.7 | 93.8 / 127.7 | 95.4 / 146.8 |
| Q5_K | 111.8 / 147.9 | 106.7 / 137.6 | 110.6 / 150.8 |
| Q6_K | 116.2 / 181.9 | 109.3 / 178.1 | 114.1 / 188.0 |
| IQ4_XS | 90.8 / 172.3 | 87.7 / 154.8 | 89.2 / 169.4 |

Shapes: gate 17,408×5,120; down 5,120×17,408; head 248,320×5,120.
All twelve specialized cases win; production routing remains two tokens only.
The generic paths remain slower on both FFN shapes (Q3_K, IQ3_S, Q4_0,
PQ2_0, PTQ1_0). This run revalidates two-token routing; it does not replace
the historical 3–8-token measurements or measure full-model recovery latency.

**Session 1 table (single dispatch per command buffer — the rate is the GPU's
clock ramp, not the tile; kept for the record and superseded by the batched
table above).**

**ffn_gate (17,408×5,120).** t=1/4/8 take the 8×8 tile, t=9..32 the 32×32.

| Encoding | matvec | t=1 | t=4 | t=8 | t=9 | t=16 | t=22 | t=32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q4_K | 178 | 51.7 | 30.1 | 44.7 | 23.6 | 10.0 | 23.6 | 10.7 |
| Q5_K | 211 | 44.3 | 87.9 | 87.9 | 34.0 | 21.3 | 34.1 | 17.6 |
| Q3_K | 124 | 31.3 | 59.7 | 60.5 | 19.2 | 21.9 | 20.5 | 11.4 |
| Q6_K | 252 | 59.2 | 67.2 | 59.4 | 28.3 | 40.9 | 23.3 | 24.4 |
| IQ3_S | 123 | 31.7 | 63.1 | 63.2 | 22.2 | 22.2 | 18.2 | 22.3 |
| IQ4_XS | 214 | 49.5 | 84.0 | 42.2 | 20.6 | 22.6 | 18.5 | 27.3 |
| Q4_0 | 231 | 44.6 | 60.5 | 44.6 | 27.4 | 21.9 | 11.6 | 21.7 |
| PQ2_0 | 118 | 45.7 | 45.7 | 45.7 | 14.5 | 8.3 | 7.2 | 14.5 |
| PTQ1_0 | 88 | 29.4 | 29.4 | 29.4 | 10.6 | 7.3 | 6.4 | 11.0 |

**ffn_down (5,120×17,408).**

| Encoding | matvec | t=1 | t=4 | t=8 | t=9 | t=16 | t=22 | t=32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q4_K | 152 | 44.6 | 64.1 | 76.8 | 26.7 | 10.1 | 26.7 | 13.5 |
| Q5_K | 189 | 43.7 | 43.0 | 87.3 | 12.7 | 16.2 | 31.7 | 17.3 |
| Q3_K | 118 | 56.6 | 57.0 | 57.4 | 20.6 | 11.3 | 17.2 | 17.0 |
| Q6_K | 236 | 53.4 | 74.8 | 64.2 | 26.4 | 22.3 | 35.9 | 24.4 |
| IQ3_S | 115 | 34.3 | 61.6 | 61.6 | 20.9 | 20.9 | 17.5 | 20.7 |
| IQ4_XS | 186 | 41.5 | 61.5 | 42.0 | 26.5 | 22.1 | 14.3 | 21.4 |
| Q4_0 | 205 | 67.8 | 58.9 | 71.7 | 20.4 | 18.8 | 13.0 | 21.0 |
| PQ2_0 | 97 | 45.3 | 45.3 | 45.3 | 13.7 | 9.2 | 7.7 | 13.6 |
| PTQ1_0 | 83 | 21.8 | 29.0 | 29.0 | 10.4 | 6.7 | 6.1 | 8.6 |

## Merged projections (KERN-04)

`Backend.matvecSegments` accepts up to four borrowed matrix/output descriptors
sharing one input. Each segment has a row count, encoding, stride, and weight
and output byte offsets. All rows must be multiples of sixteen, columns must
match, output ranges must fit and be disjoint from reads, and the complete group
must fit seven data bindings. Validation finishes before dispatch, so the model
adapter can fall back to standalone matvecs on `InvalidShape` without replaying
partially recorded work.

Input uses binding 0; bindings 1–6 hold the distinct weight/output buffers,
selected by a uniform switch. Slices of the same buffer share a binding at
byte offset zero and retain their offsets in the segment table. DeltaNet packs
`mixed|z|beta|alpha` into one activation allocation; attention writes directly
to its query/gate buffer and the current key/value session slots. No extra GPU
allocation or host copy occurs per token.

Plain mode assigns each sixteen-row threadgroup to one segment. The specialized
bodies produce exactly the same lane partials as standalone kernels; a generic
body preserves the standalone sixteen-value traversal for other encodings or
unaligned weights. The encoder marks that fallback with an internal high bit
in the encoding field. Profile entries are `matvec_segments`, with no single
encoding and with the sum of all segment rows and weight bytes.

Pair mode computes eight gate/up output rows per threadgroup: two SIMD groups
compute gate and two compute up, then share sixteen reduced F32 values (64 bytes)
through threadgroup memory. A barrier precedes `silu(gate) * up`; no intermediate
FFN vector needs a GPU write/read. The existing `up` scratch buffer is retained
for the standalone fallback. Each SIMD group still computes four rows with the
same reduction order. This layout replaces the plan's initial sequential pair
per SIMD group, which measured a regression; the outcome records both variants.

## F16 KV cache (KERN-07)

`session.Layout.attention` carries a `precision` (`f32` or `f16`); the
session block is bytes and each attention layer exposes `Rows` views
(`range(first, count)` for a GPU binding, `floats(first, count)` for the
CPU reference, which asserts `f32`). The plan builds its layouts from
`Plan.init`'s `kv` argument; `engine.Engine.open` takes it from the
command (`--kv f16|f32`, `engine.kv_precision` in `nuclis.json`, default
`f16`), and `Engine.kv_precision` reports what the session uses (`f32` on
the CPU reference, whatever was asked). A 32,768-token session holds
2.15 GiB with `f16` and 4.15 GiB with `f32` (16 attention layers × 2 ×
1,024 channels per token, plus 150 MB of recurrent state).

Where the halves come from and who reads them:

- **Decode.** The merged attention projection keeps writing F32; with an
  F16 cache it writes the plan's `k`/`v` scratch rows instead of the slot,
  the key norm and RoPE run there, and one `nu_pack_half` dispatch
  (two vectors, one thread per element, round to nearest even, bit-identical
  to Zig's `@floatCast`) writes both slots. `nu_attention_scores_h` and
  `nu_attention_values_h` read `half` and convert per element; queries,
  scores, the softmax, and the output stay F32. One extra dispatch per
  attention layer (sixteen per token) and no extra traffic: the F32 rows
  it reads were written by the projection in the same command buffer.
- **Prefill.** The chunk's rotated keys and values are packed straight from
  `k_c`/`v_c` into the cache range (replacing the F32 copies), and the
  rotated queries are packed into `q_c_h`. `nu_attention_chunk_h` loads
  half keys, values, and queries into `simdgroup_half8x8` operands and
  accumulates in `simdgroup_float8x8`, because the matrix unit multiplies
  operands of one type (the ENGN-05 matmul's contract; a half→float matrix
  conversion does not exist in MSL and the storage-vector route crashes
  the compiler). The probability tile is therefore rounded to half before
  P·V, and the running softmax sum is taken over the stored halves so the
  weights remain a convex combination. Scores, the online softmax, the
  accumulators, and the output are F32. This is a design deviation from
  the plan's "read half via a flag": the F16 chunk path rounds Q and P as
  well as K and V, which the tolerances below measure.
- **The CPU reference** never carves an F16 layout (decided); an `f16`
  request on `--backend cpu` runs F32 and the report says so.

Evidence (`test-metal`, 2026-09-10): packed halves bit-identical to
`@floatCast`; half decode attention at 1,021 visible against the CPU over
the rounded cache 1.1e-8 (the kernel's own error, bound 2e-5) and
against the CPU over the original floats 1.2e-5 max abs, 2.1e-4 relative
RMS (the rounding's cost on N(0, 0.5²) data); half chunk attention on the
ENGN-03 cases against the CPU over the rounded operands 1.8e-4 max abs (bound
1e-3: a 2^-11 weight perturbation over values up to about 2, reached only
when few keys are visible). Full model: `make compare-f16` (raw `Hello,`,
position 2) passes at its own tolerance, **max abs 3e-2 and relative RMS
2e-4 per layer file** (measured 2.5e-2 at layer 59 of token 1 and 1.9e-4 at
layer 51 of token 0; the residual stream's outlier channels carry the
absolute error), with the logits inside the bring-up thresholds (9.2e-4 /
5.3e-5) and the greedy token unchanged (353). The bring-up thresholds
(2e-3 / 1e-4) stay the accepted numbers for the F32 cache
(`make compare-f32`, unchanged at 1.22e-4); `make compare` runs both.
`test-generation --metal`: 70 tokens through an F16 cache against the F32
stepped logits, 4.3e-4 max abs / 2.3e-5 relative RMS stepped and
2.6e-3 / 1.3e-4 chunked (chunk 32), argmax identical; greedy tokens on the
22-token bench prompt (64 out) and the 545-token raw prompt (32 out) are
identical in both precisions. Half assumes |k|, |v| < 65,504 after the
key norm and RoPE, checked like ENGN-05's activation assumption: through the
generation check and the pinned prompts, never per element.

Performance is in [bench.md § Observations](bench.md#observations-so-far)
(KERN-07 rows).

## Flash-decoding attention (KERN-08)

`Backend.attentionDecode` replaces the three-pass decode attention in the
plan: two dispatches, no `[heads][visible]` score buffer, and every cache
row read **once per KV head** instead of once per query head (the six
query heads of a group share the row). Grid = `(kv_head, split)` with
`splits = min(64, ceil(visible / 256))`; a threadgroup of four SIMD groups
walks its slice of rows, each SIMD group every fourth row so the four
rows touched per step are adjacent in memory, lane l owning channels
l, l+32, … of the key and value rows. The kernel is a template over the
query heads per threadgroup and the channels per lane, instantiated as
8 × 8 (widths up to 256) and, since MODL-06, 4 × 16 (`_w` / `_wh`, widths up
to 512), the same 128 floats of registers per lane either way; a KV head
with more query heads than the instantiation holds takes several
threadgroups per split (`attentionDecodeHeadGroups`), each reading the
slice once — Gemma's global layers put 16 query heads of 512 channels
over one KV head, four threadgroups per split. Per head it keeps a
running max, sum, and accumulator over its channels (online softmax in
F32; F16 rows convert on load; queries and probabilities are F32 in both
precisions, unlike the prefill chunk kernel). The SIMD groups merge into
the first through an 8 KB stage in three rounds, and the threadgroup
writes one partial `(m, l, acc[value_width])` per head into
`[query_heads][splits][2 + value_width]` (1.6 MB for the model, allocated
once, independent of the capacity: the score buffer was 3 MiB at 32K).
`nu_attention_merge` (one threadgroup per head, one thread per channel)
rescales every split to the global max and divides by the merged sum;
empty SIMD groups or splits (`m = −∞`, `l = 0`) drop out of both merges.
The three-pass kernels stay in the library as `test-metal`'s oracle and
are no longer dispatched by the plan.

Evidence (`test-metal`, 2026-09-10): the six pinned attention fixtures
(tiny widths, one split, mostly empty SIMD groups) within 1e-5; the model
shape at 257 visible (two splits, the second short), 1,021, 16,385 (64
uneven splits), and 32,000 rows against the F64 CPU reference: F32 cache 2.2e-8 / 1.2e-8 / 7.5e-9 / 4.7e-9 max abs, F16 cache over the rounded rows 1.9e-8 / 1.3e-8 / 5.1e-9 / 4.2e-9 (bounds 2e-5 up to 1,021 rows and 1e-4 above); shape rejections (a group of 9, a width of 264, a short partial buffer, a short cache).
Full model: `make compare-f32` 129 files, max abs 6.1e-5 (was 1.22e-4: each row is now accumulated once in F32 rather than through a stored score), `make compare-f16` 2.50e-2 / 1.9e-4 at the F16 tolerance, greedy unchanged. `test-generation --metal` unchanged
(bit-identical sessions, snapshot round trip). Performance:
[bench.md § Observations](bench.md#observations-so-far) (KERN-08 rows).

## Long-context prefill attention (ENGN-08, 2026-09-10, closed without a kernel change)

ENGN-07 measured prefill at the reference's rate at 512 tokens and −6 / −15 /
−26 % at 4K / 16K / 32,639, with the matmul tiles at the reference's speed
(ENGN-05). ENGN-08 asked whether `nu_attention_chunk` re-reading the cache once
per query head (ENGN-03's recorded deviation) was the cause, and found it is
not; the record is here so the next attempt starts from the evidence.

**Profile** (`bench --profile --prompt-tokens` on the 16,384-token
reference array, `--max-tokens 1`, F16 cache, one run;
[nuclis-profile-16k-2026-09-10.json](../benchmarks/nuclis-profile-16k-2026-09-10.json)):
prefill 256.5 s (63.9 tok/s under profiling), 64 chunks of 256 tokens.
`nu_attention_chunk_h` is the largest single kernel at 75.5 s (29.6 %;
74 ms per layer and chunk at an average of ~8K visible rows), the
batched matmuls together about 60 %, the chunkwise DeltaNet 3.6 %. At
74 ms per layer-chunk the kernel runs at about 0.7 TFLOP/s of 8×8 matrix
work against the 5 TFLOP/s the ENGN-05 matmul tiles reach, and the sub-block
loads it issues per multiply are cache hits (see below), so it is neither
bandwidth- nor matrix-unit-bound: instruction and latency per SIMD group.

**Variants tried, all correct, all slower** (same profile, attention
seconds of the same 16K prefill; every variant passed the ENGN-03 and KERN-07
fixtures and the new 16,384-row case, F32 3.87e-7 / half 1.81e-4):

| Kernel | Attention | Prefill |
| --- | ---: | ---: |
| ENGN-03 as shipped: threadgroup per (query head, 32-query tile), each SIMD group loads its own K/V sub-blocks from the cache | 75.5 s | 63.9 tok/s |
| Shared stage: threadgroup per (KV head, 8-query tile), six SIMD groups (one per query head), each 8-row K/V sub-block staged once into threadgroup memory by a scalar copy, 16 threadgroup barriers per 32-key tile | 90.4 s | 60.8 |
| Same with 8-wide vector staging, 16 half rows per stage (8 float), Q loads hoisted out of the sub-block loop, scratch sized for six groups | 85.3 s | 61.4 |
| Same plus the 8 query rows held in registers for the whole key walk (32 `simdgroup_matrix` per SIMD group) | 96.8 s | 58.6 |

Sharing the cache reads across the six heads of a KV group therefore does
not pay: the per-head re-reads were served from the GPU's caches, and the
barriers and copies cost more than the loads saved. Holding queries in
registers hurt through register pressure (the kernel already carries 32
output accumulators per SIMD group). What the numbers say instead is
that each 8×8 multiply in this kernel is paired with about 1.5
`simdgroup_load`s and that no loaded block serves more than one multiply,
whereas the matmul tiles reuse every staged block across several. The
untried lever is reuse at the register level: a threadgroup per (head,
32-query tile) whose four SIMD groups split the value columns (each
owning 4 row blocks × 8 column blocks, still 32 accumulators) and share
the probability tiles through threadgroup memory, so a V block serves four
multiplies; the score half needs the same for K (a d-slice per SIMD group
with a partial-score reduction) or accepts 1:1. That is a register-budget
study first (48 live matrices per SIMD group), then a kernel; it was not
started in ENGN-08's session.

## Numerical evidence

`test-metal` (ReleaseSafe, Apple M4 Pro, last checked 2026-09-10):

| Check | Comparison | Tolerance |
| --- | --- | --- |
| Every pinned quantized fixture, all 10 encodings, via `nu_embed` | vs CPU decoder | **exact** |
| Every pinned fixture column via one-hot matvec (specialized kernels for Q4_K/Q5_K/Q6_K/IQ4_XS, Q3_K/IQ3_S, Q4_0) | vs fixture value | 1e-5 relative |
| Merged projections and fused SiLU, all fixture encodings, forced generic and misaligned fallback, separate/packed outputs | vs standalone GPU operations | **exact** |
| Merged shape/range/alias/binding rejection, profile row/byte totals | API invariants | **exact** |
| Q3_K/IQ3_S pinned fixture columns via one-hot specialized matvec | vs fixture value | **exact** |
| Randomized 1,280/5,120/17,408-column rows, 35 per encoding, specialized and generic kernels | F32 GPU vs F64 CPU `matvec` | `4e-6 × Σ|w·x| + 1e-6` |
| Misaligned Q4_K weight slice through the generic fallback; `specializedMatvec` selection rules | vs CPU `matvec`; exact | same; exact |
| Q4_K half-subnormal scales | vs CPU decoder | 1e-9 |
| Dense F32/F16, three rows | vs CPU `matvec` | 1e-6 |
| Pinned DeltaNet steps, GPU state carried | vs llama.cpp fixtures | 1e-5 |
| Pinned convolution steps | output 1e-6; history exact | |
| Two heads sharing Q/K, three updates | vs CPU `recurrent.delta` | 1e-6 |
| Six pinned attention graphs | vs llama.cpp fixtures | 1e-5 |
| Model-shaped attention, 1,021 visible of 1,024 | vs CPU `attention.apply` | 2e-5 |
| `nu_pack_half` over 2 × 1,048,576 floats | vs `@floatCast` | **exact** |
| Half decode attention, 1,021 visible (KERN-07) | vs CPU over the rounded cache / over the F32 cache | 2e-5 (measured 1.1e-8) / measured 1.2e-5, 2.1e-4 relative RMS |
| Half chunk attention, the ENGN-03 cases with half Q, K, V, and P (KERN-07) | vs CPU over the rounded operands per row | 1e-3 (measured 1.8e-4) |
| Flash-decoding attention (KERN-08): six pinned fixtures; model shape at 257 / 1,021 / 16,385 / 32,000 visible, F32 and F16 cache | vs fixtures; vs F64 CPU `attention.apply` (F16: over the rounded rows) | 1e-5; 2e-5 up to 1,021 and 1e-4 above (measured ≤ 2.3e-8) |
| Windowed and wide chunk attention (MODL-06): windows of 1,024 and 8 on the 16/8/256 geometry, 16/1/512 with two value splits, F32 and F16 | vs F64 CPU per row over the window's key slice (F16: rounded operands) | 1e-5 (measured 3.0e-6); 2e-3 (measured 1.9e-4) |
| Wide (`_w`/`_wh`) and grouped decode attention (MODL-06), 16/1/512 and 16/8/256 at 257 and 1,021 visible, both precisions | vs F64 CPU `attention.apply` (F16: rounded rows) | 5e-5 (measured 2.4e-7) |
| RoPE over a 512-wide head with factors (64 ones, 192 × 1e30) at 32,767 | vs CPU `rope.apply` with factors; unrotated pairs exact | 2e-6 relative; **exact** |
| `nu_gelu_mul` (with ±60, 200, −3e3), `nu_scale`, `nu_add_scale`, `nu_softcap` (with ±3e3) | vs `cpu.gelu`, F32 arithmetic, `std.math.tanh` | 2e-6 relative; exact; exact; 2e-6 relative |
| Merged GELU pair (`gelu_mul_pair`), every fixture encoding | vs standalone matvecs + `nu_gelu_mul` | **exact** |
| RMSNorm (5120; 48×128 with silu gate; 24 strided heads) | vs CPU `rmsNorm` | 1e-5 |
| RoPE at position 32,767 | vs CPU `rope.apply` | 2e-6 relative |
| RoPE with adjacent pairing, 32 heads of 128 at base 5e5, position 32,767 (MODL-12) | vs CPU `rope.apply` in that mode | 2e-6 relative |
| L2 norm, silu·mul, add, silu, gates, sigmoid gate | vs `cpu.*` | 1e-6 |
| Argmax over 248,320 with a tie | lowest index | exact |
| Partial top-k (256) over 248,320 with ties, vs the CPU sort | (value, index) pairs | **exact** |
| Batched matmul, 40 rows × 1,280/5,120 columns × 37 tokens, every encoding | per-token row vs F64 CPU `matvec` | `4e-6 × Σ|w·x| + 1e-6` |
| Chunk kernels (rope rows in both pairings, convolution rows + history, grouped L2 norm, repeated delta gates, copy) | vs their sequential single-token forms | **exact** |
| Exp-sum over 248,320 flat logits, T = 1.5 | vs F64 | ≤ 2e-6 relative (measured 9.6e-8) |
| Non-finite logit raises the partition flag and never enters the list; shape rejection | API invariants | exact |

Full-model comparison, raw `Hello,`, GPU-resident plan, against the llama.cpp
`7620399` traces: all 128 layer outputs across two positions and the logits
pass the bring-up thresholds (max absolute 0.002, relative RMS 1e-4). Observed
maxima with the generic matvec: absolute `7.0e-4`, relative RMS `2.9e-6`;
logits absolute `1.6e-5`, identical top five, greedy `[353, 2688]` (` I'm`).
With the specialized matvec kernels (`make compare`, 2026-09-07): maximum
absolute `9.2e-5` over the 129 files — tighter, since the factored form
rounds fewer times. With KERN-03 Q3_K/IQ3_S (2026-09-08), all 129 files
still pass, maximum absolute error `0.0001220703125`. KERN-04 merged projections
retain that maximum; `make test-generation-metal` was rerun on 2026-09-08.
With the F16 cache (KERN-07, 2026-09-10) the same comparison holds at the F16
tolerance stated in [§ F16 KV cache](#f16-kv-cache-kern-07): layer files up
to 2.5e-2 absolute / 1.9e-4 relative RMS, logits 9.2e-4 / 5.3e-5, greedy
unchanged.
`test-generation --metal`
passes: independent sessions and reset after injected cancellation produce
bit-identical logits, and (ENGN-06) a step after `restore` reproduces the
snapshotted session's next logits bit for bit. Since MODL-06 the check selects
the family from the file, carries per-family tolerances, and repeats the
chunked comparison through the generic F32 tiles (`generic_only`) to
separate the half-tile rounding from the chunk schedule: Qwen 2.4e-3 /
1.1e-4 with the specialized tiles, 2.8e-5 / 1.2e-6 with the F32 tiles;
F16 cache 5.5e-4 / 2.4e-5 stepped, 2.6e-3 / 1.3e-4 chunked (2026-09-11).
The second model's numbers are in
[gemma4.md § Metal plan](gemma4.md#metal-plan-modl-06-2026-09-11).

## Performance observation

`make bench`: 22-token chat prompt, 64 output tokens, context 2048, greedy,
ReleaseSafe, Apple M4 Pro 48 GiB, macOS 26, one warmup and three measured runs.

Generic matvec (before KERN-01):

```text
run    prompt  gen  stop           prefill ms  pp tok/s  first ms   decode ms  tg tok/s   gpu ms
 1         22   64  token_budget       4124.9      5.33    4124.9     12908.9      4.88  15827.2
 2         22   64  token_budget       4365.9      5.04    4365.9     13043.5      4.83  16177.6
 3         22   64  token_budget       4004.4      5.49    4004.4     12387.0      5.09  15313.1
Measured mean over 3 runs: prefill      5.29 tok/s, decode      4.93 tok/s, first token 4165.1 ms
```

Specialized matvec (KERN-01, 2026-09-07):

```text
run    prompt  gen  stop           prefill ms  pp tok/s  first ms   decode ms  tg tok/s   gpu ms
 1         22   64  token_budget       2500.3      8.80    2500.3      7357.9      8.56   8635.9
 2         22   64  token_budget       2675.2      8.22    2675.2      7395.0      8.52   8720.7
 3         22   64  token_budget       2488.8      8.84    2488.8      7382.4      8.53   8803.8
Measured mean over 3 runs: prefill      8.62 tok/s, decode      8.54 tok/s, first token 2554.7 ms
```

One command buffer per token in the CLI as well (observer split, KERN-02,
2026-09-07):

```text
run    prompt  gen  stop           prefill ms  pp tok/s  first ms   decode ms  tg tok/s   gpu ms
 1         22   64  token_budget       2189.4     10.05    2189.4      6495.3      9.70   8596.1
 2         22   64  token_budget       2185.8     10.07    2185.8      6500.8      9.69   8603.9
 3         22   64  token_budget       2190.1     10.05    2190.1      6498.9      9.69   8601.0
Measured mean over 3 runs: prefill     10.05 tok/s, decode      9.69 tok/s, first token 2188.4 ms
```

Decode history: 2.3 tok/s (synchronous per-operation bridge) → 5.0 (one
command buffer per token in the plan) → 8.5 (specialized matvec) → 9.7 (one
command buffer per token in the CLI too). GPU busy time is ~101 ms per token
and wall time per decode step ~103 ms: decode is GPU-bound. The measured
budget per token (profile mode): specialized matvec 83.7 ms, generic matvec
14.8 ms (Q3_K and IQ3_S at ~40 GB/s are 10 ms of it), everything else 6.6 ms
(rmsnorm 2.7 ms in 209 launch-bound dispatches). The reference decodes at 9.66
tok/s at 512 context; the bandwidth floor is ~59 ms per token. Prefill ran
one token per command buffer at decode speed until ENGN-02 (see [Prefill in
chunks](#prefill-in-chunks-engn-02)). These are smoke observations on a short
prompt, not the acceptance runs (ENGN-07).

## Limits

- The specialized matvec kernels reach 150–250 GB/s of a published 273 and
  are bound at ~0.9 ns per block by per-block instruction work, not bytes.
  Q8_0 and IQ4_NL (under 1 % of the Qwen file's bytes) still use the
  generic kernel; KERN-03 gave Q3_K and IQ3_S their own, MODL-08 Q4_0.
- Profiling times dispatches in separate encoders, which perturbs what it
  measures by ~8 %; there is no per-dispatch sampling on Apple GPUs, and the
  Instruments GPU limiter counters are unavailable on this device from
  `xctrace` (KERN-05).
- Prefill is fully batched (ENGN-02–ENGN-04); its ceiling is the matmul tile's
  half operands (ENGN-05), whose rounding shows more on Gemma's larger
  activations (2.5e-3 relative RMS on the chunked logits against 1.1e-4
  for Qwen). Attention widths are bounded at 512 (decode: 16 channels per
  lane in the wide pair; prefill: two value splits); a wider head needs a
  third instantiation.
- Wrapped buffers: `Backend.wrap` caches by address and `unwrap` forgets a
  range (plans call it on `deinit` so a later session at the same address
  and another length can be wrapped), but the bridge keeps every
  `MTLBuffer` object until the backend is destroyed; a long-lived backend
  that opens and closes many plans accumulates no-copy buffer objects.
- Shader compiled from source at startup; no binary archive.
- GPU resource cleanup is exercised by recreating the backend across fixture
  files in `metal-check`; Zig's testing allocator cannot observe Metal objects.
