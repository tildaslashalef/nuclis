<!-- kind: record -->
# Audio: the record

The dated record behind [the reference](../../engine/audio.md).

## The hook dump of Google's tower (2026-10-09)

Every stage of the CPU reference also matched a hook dump of Google's
tower (subsampling, each block, the output, the projection) to relative
RMS ≤ 1.8e-6. `scripts/audio-trace.py` (retired; at tag `v0.11.0`) wrote that
dump to `.zig-cache/audio-trace/`, and the `audio` mode traces the CPU reference
against it stage by stage whenever it is there: the localizer that found
the swapped norms and the mask width. Results are in
[embeddinggemma.md § Audio](../../models/embeddinggemma.md#audio).
