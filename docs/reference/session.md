# Session state: layouts, the byte block, and snapshots

`inference/src/runtime/session.zig` owns every mutable byte of an inference
session and nothing else: no weights, no activations, no model knowledge.
Model adapters describe what they need as `Layout`s; the session allocates
one block, carves typed views, and enforces the one rule the spec insists
on: recurrent state is never rewound by changing a length.

## Layouts and views

| Layout | Regions | View the layer sees |
| --- | --- | --- |
| `attention { key_row, value_row, precision }` | `capacity` key rows and `capacity` value rows, stored as F32 or F16 (`precision`, KERN-07) | `Rows`: `range(first, count)` is a byte range for a GPU binding; `floats(first, count)` is the F32 view the CPU reference reads (asserts `f32`) |
| `recurrent { history, matrix }` | that many F32 values each, whole | `[]f32` |

`Session.memory` is one page-aligned byte block (`[]align(16384) u8`), so a
GPU backend wraps it once with no copy; every region starts on a 16-byte
boundary and each view is exact (padding belongs to nobody).
`Session.bytes()` is the block, page padding included: 2.15 GiB at 32,768
tokens with the F16 cache, 4.15 with F32, of which 150 MB is recurrent
state. `layout_digest` hashes the layouts and the capacity; equal digests
mean byte-compatible state.

## The state machine

```text
ready --begin/beginChunk--> updating --commit/commitChunk--> ready
                              |
                             fail  -->  failed --reset--> ready
```

A step admits its positions (`ContextFull` if they do not fit), writes, and
commits `position += count`. A failure between the two leaves the session
`failed`: DeltaNet updates are in place and nontransactional, so partially
updated state must never be mistaken for committed state. Only `reset()`
(a memset of the whole block) leaves that state.

**GPU invariant.** `reset`, `fail`, `snapshot`, `restore`, and any CPU read
of a view are valid only while no submitted GPU work can still write the
block. The synchronous Metal backend waits on every command buffer, so
these are always valid between steps; an asynchronous backend would have to
fence first.

## Snapshot and restore (ENGN-06)

`Session.snapshot(gpa)` copies the **used** extent into a caller-owned
`Snapshot`: each attention layer's rows `[0, position)` (keys then values,
at the layout's precision) and each recurrent layer's history and matrix
whole, packed in layer order with the position, capacity, and layout
digest. It is refused on an updating or failed session (`SessionNotReady`):
there is no committed state to keep. Its size is 150 MB plus the cache up
to the position (64 KiB per token with F16), never the capacity.

`Session.restore(&snap)` copies the bytes back and sets the position. It
requires a ready session (reset a failed one first) of the same capacity
and layout digest and a snapshot whose byte count matches its position;
anything else is `SnapshotMismatch` with the session untouched. Rows past
the restored position are not cleared, and nothing reads them: attention
sees `[0, position + 1)` after its own write, recurrent state is whole.

Why copy instead of rewind: 48 of the 64 layers are recurrent, and their
matrix after token *n* is a function of every token before it. Truncating a
KV length would leave those matrices at token *n* while attention believed
it was at *m < n*. A snapshot is the only rewind that exists, and it is
explicit about its cost.

`engine.Model.snapshot`/`restore` expose the pair on both executors; the
chat and the agent loop are the intended callers (turn-boundary
checkpoints instead of replaying the conversation after a cancel), wired
in the agent's phase 2.

**Evidence** (`make test-generation` and `make test-generation-metal`,
2026-09-10): step token 1, snapshot (157,024,256 bytes at position 1),
step token 2 → logits A; restore, step token 2 → logits B; A equals B bit
for bit on both backends; a third step after the restore equals the same
step of a never-snapshotted session; a session of capacity 8 refuses the
capacity-4 snapshot and keeps its position; a session poisoned by a
cancelled step refuses to snapshot until reset. Unit tests cover the
round trip under every allocation failure, the used-extent size, the
untouched unused row, precision and capacity mismatches, and a truncated
snapshot.
