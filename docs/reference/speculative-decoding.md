# Speculative decoding: recovery contract, draft sources, measurements

The reference document of the speculative-decoding theme. Its requirements
are in the [spec](../spec.md#speculative-decoding); the units that fill it
in are planned in [TODO.md](../../TODO.md) (ENGN-11, MODL-18, ENGN-12,
MODL-19, MODL-20); closed outcomes are cited from the
[engineering log](../engineering-log.md).

Nothing of the theme is implemented as of 2026-09-19. What exists: the
Qwen adapter binds and validates the 15 embedded `nextn` tensors without
executing them ([qwen-validation.md](qwen-validation.md)); the session has
a host-side snapshot and restore ([session.md](session.md)); every family's
draft companion is pinned and pulled ([artifacts.md](artifacts.md)).

Sections to come, one per unit: the recovery contract (checkpoint, rewind,
replay, on both state kinds), the draft contract, each family's draft
source with its facts and provenance, and the measurements behind each
catalogue verdict.
