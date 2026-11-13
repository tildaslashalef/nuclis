<!-- kind: record -->
# Session state: the record

The dated record behind [the reference](../../engine/session.md).

## Snapshot and restore: evidence (2026-09-10)

**Evidence** (`make gate NAME=qwen38-generation-cpu` and `make gate NAME=qwen38-generation-metal`,
2026-09-10): step token 1, snapshot (157,024,256 bytes at position 1),
step token 2 → logits A; restore, step token 2 → logits B; A equals B bit
for bit on both backends; a third step after the restore equals the same
step of a never-snapshotted session; a session of capacity 8 refuses the
capacity-4 snapshot and keeps its position; a session poisoned by a
cancelled step refuses to snapshot until reset. Unit tests cover the
round trip under every allocation failure, the used-extent size, the
untouched unused row, precision and capacity mismatches, and a truncated
snapshot.

## Checkpoint and rewind: costs (2026-09-19)

**Costs** (the generation check, `make gate NAME='qwen38-generation-*'`, Qwen 27B,
2026-09-19): the region is 156,893,184 bytes (150 MB) on Qwen and on
Bonsai; `checkpoint` and `rewind` 3 ms each on Metal and 2 ms each on the
CPU reference (one 150 MB copy in each direction, the block quiescent
between steps). On the attention-only Gemma 4 and Muse Glimmer the region
is zero bytes and both calls are free. The recovery check confirms the
accepted-prefix contract per accepted length: on the CPU the replay's
logits equal the sequential run bit for bit; on Metal the batch is chunked,
so the family's chunk-versus-step bound applies (Qwen 27B ≤ 2.9e-3 max abs
and 1.5e-4 relative RMS over a 4- and an 8-row batch, argmax equal; Bonsai
≤ 8.6e-6; Gemma 4 and Muse exactly zero at these small tiles).

## The row checkpoints the tape replaced (2026-10-01)

This replaced the row checkpoints recovery used from 2026-09-20 (a 150 MB recurrent copy per batch
row, 1.25 GB for eight, written by the chunk kernel and restored by a host
copy); the [worklog archived at v0.6.0](https://github.com/tildaslashalef/nuclis/blob/v0.6.0/docs/worklog.md) records both.
