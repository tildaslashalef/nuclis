<!-- kind: record -->
# Generation and sampling: the record

The dated record behind [the reference](../../engine/sampling.md).

## Trace provenance (2026-09-06)

The same command with `--backend cpu` writes the fixture itself; it was
re-made that way on 2026-11-12, when the file was re-pinned. The first
traces came from llama.cpp `7620399` on 2026-09-06, through
the llama.cpp generation harness (retired; at tag `v0.11.0`), while
llama.cpp was the reference
([models § 4](../../models/README.md#4-validate-against-nuclis-itself)).

The first rows came from the retired helper's `--mtp-draft` mode, which
read the block's pair from llama.cpp's MTP context; Muse Glimmer's DFlash
rows came from its `--dflash-draft` mode and left the tree with the Muse
gates (the last tag that held them is `v0.11.0`).

## Evidence (2026-09-07)

On 2026-09-07, Zig 0.16.0 ReleaseSafe on the target Apple M4 Pro, the pinned
artifact and llama.cpp revision `7620399f58aebfd2196b74021f9581bcf7218cb9`:

- Raw `Hello`: all 64 layers compared; logits maximum absolute error `4.77e-6`,
  RMS `9.80e-7`; identical top five, greedy token 11 (`,`).
- Raw `Hello,`: all 128 layer outputs compared across two positions; identical
  top five logits and greedy token 353 (` I`). Native two-token generation
  continued with IDs `[353, 2688]`, text ` I'm`.

- Chat `Hi` with the thinking-off template: all 832 layer outputs across 13
  prompt positions pass the same comparison thresholds. Native output begins
  `Hello! How` (IDs `[9419, 0, 2500]`).
- Full-model session isolation and reset after injected layer-3 cancellation pass
  with bit-identical two-token CPU logits. All 95 default tests pass in Debug and
  ReleaseSafe, including partial-allocation cleanup, sampling, and UTF-8 splits.

The first single-token native run took 39 whole seconds excluding model loading.
That is a smoke-run observation, not a warm throughput benchmark. Full-model
checks run explicitly and are distinct from default allocator/unit tests.
