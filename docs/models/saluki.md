<!-- kind: card -->
# Underdog Saluki 27B

**Parked.** Underdog's 2.26-bit re-encoding of Qwen3.8-27B
(`ConwayResearch/Underdog-Saluki-27B-1.0`, `general.architecture =
qwen35`) has no catalogue entry, no gates, and no scheduled work. Its
bring-up of 2026-10-14 to 2026-10-21 is [the record](../benchmarks/record/saluki.md).

## Architecture

Qwen3.8-27B's ([qwen3.8.md](qwen3.8.md)), dimension for dimension, without
the MTP block (`qwen35.block_count` 64). The Qwen adapter binds the file
(the committed inventory is
[`fixtures/saluki-27b.json`](../../inference/src/models/fixtures/saluki-27b.json)),
and Qwen's `MTP/` companion binds onto it as a draft
([speculative-decoding.md § The Qwen MTP companion](../benchmarks/record/speculative-decoding.md#the-qwen-mtp-companion-2026-10-21)).

## Files and encodings

The file mixes encodings per tensor; the seven it brought, IQ1_S, IQ1_M,
Q2_K, IQ2_XXS, IQ2_XS, IQ2_S, and IQ3_XXS, have CPU decoders
(`quant/decode.zig`), Metal decoders, and specialized matvecs, and run any
qwen35 file that uses them. The tiny `saluki` and `saluki-mix` variants
keep them under `make test-tiny` ([README.md § Tiny models](README.md#tiny-models)).

## Prompt profile

The upstream Qwen3.8 template, not an alias of the pinned profile: run it
with `--prompt-profile qwen38`, or register the pull with
`--profile qwen38`.

## What is verified

No gates. It was the first model validated without an external
reference, by cross-encoding the same weights against the Q4_K_M
([record, 2026-10-14](../benchmarks/record/saluki.md#validation-without-an-external-reference-2026-10-14)).

## Numbers

Metal, F16 cache, M4 Pro, 2026-10-15
([record](../benchmarks/record/saluki.md#speed-2026-10-15)): decode
16.96 tok/s at 512 and 14.97 at 32,639 (from a restored prefix); prefill
94.18 at 512 and 51.52 at 32,639. WikiText-2 perplexity 7.996 at 8 × 512,
against the Q4_K_M's 6.670.

## Known limits

Prefill at 32,639 trails llama.cpp by 35 %, and the verify batch on the
2-bit encodings costs as much as on 4-bit ones, so speculation does not
pay on prose.
