<!-- kind: reference -->
# Metal numerical evidence

What holds the [Metal backend](metal-backend.md) to the CPU reference
([cpu-reference.md](cpu-reference.md)): the kernel checks `make test-metal`
runs on pinned fixtures and synthetic shapes, and the model-level checks
that compare a whole plan against the CPU executor. The bounds below are
what the checks enforce; the values each measured, and when, are in
[the record](../benchmarks/record/metal-backend.md#numerical-evidence-2026-09-10).

## What test-metal checks

`inference/metal-check.zig` (the TTS kernels' checks are in
`tts-metal-check.zig`) needs a Metal device and no model. Every check
compares against the F64 CPU reference unless the row names another, and
every contract violation it lists is refused before dispatch.

| Check | Comparison | Bound |
| --- | --- | --- |
| Every pinned quantized fixture via `nu_embed` | vs CPU decoder | **exact** |
| Every pinned fixture column via one-hot matvec, specialized and generic kernels | vs fixture value | **exact** for the factored kernels; 1e-5 relative |
| Randomized 1,280/5,120/17,408-column rows, 35 per encoding, specialized and generic kernels; a 22-block Q4_0 row | F32 GPU vs F64 CPU `matvec` | `4e-6 × Σ|w·x| + 1e-6` |
| Misaligned weight slices through the generic fallback; `specializedMatvec` selection rules | vs CPU `matvec`; exact | same; exact |
| Q4_K half-subnormal scales | vs CPU decoder | 1e-9 |
| Dense F32/F16, three rows | vs CPU `matvec` | 1e-6 |
| Merged projections and fused SiLU/GELU pairs, every fixture encoding, forced generic and misaligned fallback, separate and packed outputs | vs standalone GPU operations | **exact** |
| Merged shape, range, alias, and binding rejection; profile row and byte totals | API invariants | **exact** |
| Split-K matvecs at 2/4/8 splits (`bench_kernels` build) | vs F64 CPU `matvec` | the matvec bound |
| Batched matmul, 40 rows × 1,280/5,120 columns × 37 tokens, every encoding | per-token row vs F64 CPU `matvec` | `4e-6 × Σ|w·x| + 1e-6` |
| Multi-row matvec, 2 to `matvec_rows_max` rows, specialized and generic bodies | per-token row vs F64 CPU `matvec` | `4e-6 × Σ|w·x| + 1e-6` |
| Fragment tile, every instantiation; `nu_fragment_layout` | vs the generic F32 tile; the pinned element layout | 2e-4 of Σ|w·x|; exact |
| Two-token production and control routing | the dispatched kernel names | exact |
| Pinned DeltaNet steps with the GPU state carried; two heads sharing Q/K | vs pinned fixtures; vs CPU `recurrent.delta` | 1e-5; 1e-6 |
| Pinned convolution steps | output; history | 1e-6; exact |
| Chunkwise DeltaNet, 70 tokens (sub-chunks 32, 32, 6), NaN in the padding | vs F64 chunk reference per head; vs 70 sequential steps | output 1e-5, state 1e-4; 1e-4 |
| Chunk kernels (rope rows in both pairings, convolution rows and history, grouped L2 norm, repeated delta gates, copy) | vs their sequential single-token forms | **exact** |
| Six pinned attention graphs; model-shaped attention at 1,021 of 1,024 visible | vs pinned fixtures; vs CPU `attention.apply` | 1e-5; 2e-5 |
| Decode split pass: the pinned fixtures; the model shape at 257 / 1,021 / 16,385 / 32,000 visible, F32 and F16 caches | vs F64 CPU `attention.apply` (F16: over the rounded rows) | 2e-5 up to 1,021 rows, 1e-4 above |
| Wide and grouped decode (16/1/512, 16/8/256), Muse Glimmer's shape, a width only the general kernel takes (12 heads of 96 over 4) | same | 5e-5 |
| Q8_0 cache: `packQ8` on 1,021 to 16,385 rows and edge blocks (all zero, subnormal scale, NaN, infinity); `unpackQ8`; the split pass at 257 / 1,021 / 16,385 visible, a 5-row verify at 4,099, 16 heads of 512 over one at 2,049 | vs `q8_0Row`; vs the CPU decoder rounded to half; vs F64 over the dequantized rows | exact; exact; 2e-5 to 1e-4 |
| `nu_pack_half` over 2 × 1,048,576 floats | vs `@floatCast` | **exact** |
| Half decode attention, 1,021 visible | vs CPU over the rounded cache | 2e-5 |
| Verify attention, 1 to 16 rows over a poisoned range past the batch, to the 32K cache; windowed and wide geometries; 17 rows and a span refused | vs F64 per row over its causal prefix | 1e-5 |
| Chunk attention: 256 rows at 1,792..2,047 over 2,048; a 37-row chunk with 1e30 past a row's horizon; one- and three-token prompts; windows of 1,024 and 8; 16 heads of 512 over one (two value splits) | vs F64 per row over its visible keys | F32 1e-5; F16 over the rounded operands 1e-3 to 2e-3 |
| Bidirectional spans: a span in a 300-row chunk with poisoned rows after it, a span past a window of 8, F16, 16×72 heads at 200 and 9,900 rows | vs F64 per row, the span's rows seeing to its end | 1e-5 relative past 1; 2e-3 F16 |
| Register-reuse chunk attention: the windowed and wide cases and counts 1–256 over a poisoned range past `position + count` | vs F64 per row, both bodies | 1e-5; 1e-3 F16 |
| Segments attention, 1 to 8,192 rows, packed and windowed, widths 256 and 512, long inputs in 8 and 64 chunks | vs F64 over each row's visible keys | 2e-4 |
| Router on 128 experts (k 8: ties at the top and the k boundary, an all-equal row, a 40× spread) and 37 (k 5) | indices; weights | exact; 1e-6 |
| Gathered Q4_0 matvec, slots `[5, 0, 5, 2]`, shared and per-slot inputs, NaN in unselected experts' scales | vs dense matvec; the selected outputs | the matvec bound; bit-identical |
| Expert decode chain, three tokens; prefill chain over 45 tokens (135 slot rows) with one expert on every token and one on none | vs `cpu.experts.ffn`; the lists on the host | 2e-5 of (1 + max|y|) F32 tile, 2e-3 half tile; exact |
| Fused norms: `rmsNormAdd`, `addRmsNorm`, `rmsNormRope` (24 packed heads at stride 512, 8 adjacent in place) | vs the unfused pair; vs `cpu.rmsNorm`/`cpu.rope` | 2e-5; 2e-4 |
| RMSNorm (5,120; 48×128 with silu gate; 24 strided heads) | vs CPU `rmsNorm` | 1e-5 |
| RoPE at 32,767, both pairings; a 512-wide head with factors (64 ones, 192 × 1e30) | vs CPU `rope.apply`; unrotated pairs | 2e-6 relative; **exact** |
| `nu_gelu_mul` (with ±60, 200, −3e3), `nu_scale`, `nu_add_scale`, `nu_softcap` (with ±3e3) | vs `cpu.gelu`, F32 arithmetic, `std.math.tanh` | 2e-6 relative; exact; exact; 2e-6 relative |
| L2 norm, silu·mul, add, silu, gates, sigmoid gate | vs `cpu.*` | 1e-6 |
| Hadamard on 5,120 / 6,144 / 17,408 over strided rows, forward, inverse, round trip; the 48 × 128 row gather and its refusals | vs `cpu.hadamard`; exact | 3e-5; exact |
| History penalties over every logit sign (positive, negative, zero, negative zero) | vs `Sampling.Sampler.penalize` | exact but for flushed subnormals |
| Argmax over 248,320 with a tie; partial top-k (256) with ties | lowest index; vs the CPU sort | exact; **exact** |
| Exp-sum over 248,320 flat logits, T = 1.5 | vs F64 | 2e-6 relative |
| A non-finite logit raises the partition flag and never enters the list; shape rejection | API invariants | exact |

GPU resource cleanup is exercised by recreating the backend across fixture
files; Zig's testing allocator cannot observe Metal objects.

## Model-level checks

These need a model and run as gates
([development.md § Gates](../development.md#gates)):

- **Layer traces.** `qwen38-trace-f32`, `qwen38-trace-f16`, and
  `qwen38-trace-q8_0` run raw `Hello,` through the GPU plan and compare
  every layer output of two positions and the logits with the pinned
  traces. Bounds per layer file: F32 2e-3 max abs and 1e-4 relative RMS;
  F16 3e-2 and 2e-4 (the residual stream's outlier channels carry the
  absolute error); Q8_0 1.0 and 2.5e-2; the greedy token unchanged.
- **Generation.** `generation-check --metal` (`qwen38-generation-metal`
  and each family's `*-generation-metal`) selects the family from the
  file and checks that independent sessions and a reset after injected
  cancellation give bit-identical logits, that a step after `restore`
  reproduces the snapshotted session's next logits bit for bit, and that a
  70-token prompt chunked (chunks of 64, 48, and 32) agrees with the same
  prompt stepped within the family's `Bounds` in
  `inference/generation-check.zig`, in each cache precision. It repeats the
  chunked comparison through the generic F32 tiles (`generic_only`), which
  separates the half tiles' rounding from the chunk schedule.
- **Verify at depth.** The `qwen38-verify-depth-*` gates
  ([metal-backend.md § Small batches](metal-backend.md#small-batches))
  bound a 4-row verify batch's logits against stepped decode at depth.
