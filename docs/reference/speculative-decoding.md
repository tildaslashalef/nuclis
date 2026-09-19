# Speculative decoding: recovery contract, draft sources, measurements

The reference document of the speculative-decoding theme. Its requirements
are in the [spec](../spec.md#speculative-decoding); the units that fill it
in are planned in [TODO.md](../../TODO.md) (ENGN-11, MODL-18, ENGN-12,
MODL-19, MODL-20); closed outcomes are cited from the
[engineering log](../engineering-log.md).

As of 2026-09-19 the recovery contract is implemented and measured
(ENGN-11). What exists: the Qwen adapter binds and validates the 15
embedded `nextn` tensors without executing them
([qwen-validation.md](qwen-validation.md)); the session has a host-side
snapshot and restore and, now, an in-block checkpoint
([session.md](session.md)); every family's draft companion is pinned and
pulled ([artifacts.md](artifacts.md)).

Sections to come, one per unit: the draft contract, each family's draft
source with its facts and provenance, and the measurements behind each
catalogue verdict.

## The recovery contract (ENGN-11)

A verify batch feeds the main model the last chosen token followed by `k`
drafts in one forward and keeps every row's logits. If `a` drafts are
accepted, the session must end at the state after them — the accepted
prefix. The two state kinds recover differently:

- **Attention caches rewind by position.** A row's content never depends on
  later rows, so `truncate(P + a + 1)` sets the position and nothing is
  copied; rows past it are ignored by contract. This is what Gemma 4 and
  Muse Glimmer (attention-only) use.
- **Recurrent state is replayed.** Qwen 27B and Bonsai 2 hold 48 DeltaNet
  layers whose matrices are a function of every token fed. The batch's
  checkpoint is taken before the forward; on partial acceptance the session
  rewinds to it and re-runs the accepted prefix as a second batched
  forward, which rewrites the same attention rows. Truncating the position
  alone would leave the matrices at `P + k + 1`.

The session owns the mechanism: a page-aligned region inside the byte
block, one recurrent copy, `checkpoint`/`rewind`/`truncate` with their
refusals, all described in [session.md § Checkpoint and
rewind](session.md#checkpoint-and-rewind-engn-11). `engine.Model.recover`
is the accepted-prefix operation, and `generation-check` exercises it per
accepted length on every family, on both executors: replay is bit-identical
to sequential decoding on the CPU, and within the family's
chunk-versus-step bound on Metal (`make test-generation` /
`test-generation-metal`, recorded in the log).

Measured costs on Qwen 27B (2026-09-19): region 156,893,184 bytes;
checkpoint and rewind 3 ms per batch on Metal and 2 ms on the CPU
reference (one 150 MB copy each way); the replay is a short-chunk prefill
(KERN-11's tile). Losing `k − a` drafts
therefore costs one 150 MB copy plus an `a + 1`-token prefill, not a
context-long replay. A bounded alternative — per-token recurrent
checkpoints written by the DeltaNet chunk kernel, `(k + 1) × 150 MB` of
device scratch on Qwen — was not needed by these numbers and is not built;
the decision and its measurement are here for the next revisiter.
