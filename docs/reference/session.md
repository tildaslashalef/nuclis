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
state. A session built with a checkpoint region adds another 150 MB on
Qwen (`checkpoint_region`, below). `layout_digest` hashes the layouts and
the capacity; equal digests mean byte-compatible state.

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

The agent uses one snapshot as its **primed prefix** (ENGN-09): at startup
the completer prefills the system block and tool definitions, snapshots
the session, and whenever a conversation must start from that prefix again
— a new session, a `/resume`, a replay after elision — restores it and
prefills only what follows. A re-opened engine (a `/ctx` change) primes
anew, since the snapshot is bound to its capacity and layout.

Why copy instead of rewind: 48 of the 64 layers are recurrent, and their
matrix after token *n* is a function of every token before it. Truncating a
KV length would leave those matrices at token *n* while attention believed
it was at *m < n*. A snapshot is the only rewind that exists, and it is
explicit about its cost.

`engine.Model.snapshot`/`restore` expose the pair on both executors; the
chat and the agent loop are the intended callers (turn-boundary
checkpoints instead of replaying the conversation after a cancel), wired
in the agent's phase 2. The model's pair also carries what a loaded
drafter keeps outside the session (`Drafter.carried`: an MTP head's
pending target hidden row, 20 KB for Qwen), in `Snapshot.carried`, and
refuses a snapshot whose carried length differs before touching the
session: without it a restored prefix proposed its first draft from a
zeroed row. `bench --prefix-cache` writes the model's snapshot to disk
([development.md § The speed loop](../development.md#the-speed-loop)).

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

## Checkpoint and rewind (ENGN-11)

A snapshot copies at `snapshot()` time whatever the caller then holds in
the heap; the recovery of a speculative verify batch cannot pay that: it
must be taken and undone inside one decode step, and the batch is fed
before the accept decision. `Session.init`'s `checkpoint` flag therefore
reserves one **page-aligned region inside the block**, after the layer
regions, holding one copy of every recurrent layer's history and matrix in
layer order (F32, on the same 16-byte boundaries the layer views use).
Attention-only layouts reserve zero bytes. The region is not part of
`layout_digest` and is never packed into a `Snapshot`; `bytes()` includes
it, and a GPU binds it with the rest of the shared block, no second
binding.

| Call | Effect |
| --- | --- |
| `checkpoint()` | `SessionNotReady` unless ready; `NoCheckpointRegion` if the session was built without one; else copies the recurrent state into the region and records the position. A no-op copy on attention-only layouts, but the position is still recorded. |
| `rewind()` | Returns to the recorded checkpoint: restores the recurrent copy and sets the position. `NoCheckpoint` without one; `SessionNotReady` on an updating or failed session. |
| `truncate(position)` | Moves the position back without touching layer memory. Allowed only when no layer is recurrent (`RecurrentStateNotRewindable`) and only within `[checkpoint_position, position]` (`RewindOutOfRange`). |

Two state kinds, two rewinds. Attention rows `[0, position)` are
independent of later rows, so a verify batch writes `[P, P + k + 1)` and
accepting `a` drafts costs `truncate(P + a + 1)` and nothing else; rows
past the position are left as they are, exactly as `restore` documents.
Recurrent state is a function of every token fed, so `rewind` restores the
copy and the accepted prefix is then **replayed** through `prefill` — the
batch's rows are rewritten by the same forward, and never by truncating a
position alone. Qwen's Metal plan avoids both the rewind and the forward:
its verify leaves the recurrent state at the batch's start and keeps a
tape that replays the accepted rows
([§ Pending rows and the verify tape](#pending-rows-and-the-verify-tape-engn-19)).

`reset()` and `restore()` clear the recorded position: both rewrite the
state, so the region's bytes are stale and are only read after a fresh
`checkpoint()`. `engine.Model.checkpoint/rewind/truncate` forward on both
executors, and `engine.Model.recover(accepted)` is the accepted-prefix
operation used by the loop: the tape replay when the batch kept one, else
rewind then `prefill(accepted)` on a recurrent model, and
`truncate(checkpoint + accepted.len)` on an attention-only one.

A model with an embedded draft block (Qwen's `blk.64`, MODL-18) adds the
block's attention cache as **one more layout in the same session** while the
drafter is loaded, so the checkpoint/rewind/truncate contract above covers
it with no second mechanism: the block's rows are rewritten by `commit`
after a rewind, and `reset` memsets them with the rest.

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

## Pending rows and the verify tape (ENGN-19)

A verify batch of at most `max_draft_rows` (8) on Qwen's Metal plan does
not advance the recurrent layers. Per DeltaNet layer it copies the rows'
convolution inputs to the plan's tape, skips the history shift, and runs
`nu_delta_rows` (`Backend.deltaRows`): `nu_delta`'s per-token arithmetic
with each state row in registers, reading the state once and never writing
it. Each row also writes its normalized keys, decays, and corrections to the
tape. The plan then calls `Session.deferRows(count)`, and
`Session.pending_rows` says the recurrent layers still hold the state at
`position - pending_rows`.

`Plan.replayRows(kept)` applies the kept prefix in one command buffer,
`nu_convolution_history` over the taped inputs and `nu_delta_replay` per
layer, and `Session.settleRows(kept)` drops the rest of the batch from the
position. `engine.Model.recover` calls it when rows are pending; every other
plan entry (`step`, the prefills, `verify*`, `propose`, `commit`) and
`Model.snapshot`/`checkpoint` first replay all pending rows, so a caller
that never recovers still sees the batch applied. While rows are pending,
`begin`/`beginChunk`, `checkpoint`, `snapshot`, and `truncate` refuse with
`ReplayPending`; `rewind`, `reset`, and `restore` drop the rows with the
rest of the state they rewrite.

The tape is plan memory, not session memory: per DeltaNet layer 8 × 10,240
convolution inputs, 8 × (16 × 128 keys + 48 decays), and 8 × 6,144
corrections, 591,360 bytes; 28,385,280 bytes for Qwen's 48 layers, only
with a drafter. A batch past 8 rows updates the state in place
(`nu_delta_chunk`) and recovery rewinds and replays. The replayed state is
bit-identical to the same rows stepped through `nu_delta` (`make
test-metal`); `make gate NAME=qwen38-generation-metal` checks the recovery
at every accepted length of a 4- and an 8-row batch against sequential
steps and exercises the refusals.

This replaced ENGN-14's row checkpoints (a 150 MB recurrent copy per batch
row, 1.25 GB for eight, written by the chunk kernel and restored by a host
copy); see the [engineering log](../engineering-log.md) for both.
