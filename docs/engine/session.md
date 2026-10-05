# Session state: layouts, the byte block, and snapshots

`inference/src/runtime/session.zig` owns every mutable byte of an inference
session and nothing else: no weights, no activations, no model knowledge.
Model adapters describe what they need as `Layout`s; the session allocates
one block, carves typed views, and enforces the one rule the spec insists
on: recurrent state is never rewound by changing a length.

## Layouts and views

| Layout | Regions | View the layer sees |
| --- | --- | --- |
| `attention { key_row, value_row, precision }` | `capacity` key rows and `capacity` value rows, stored as F32 or F16 (`precision`, [KERN-07](https://github.com/tildaslashalef/nuclis/blob/v0.6.0/docs/worklog.md#kern-07--f16-kv-cache-as-a-session-layout-option-2026-09-10)) | `Rows`: `range(first, count)` is a byte range for a GPU binding; `floats(first, count)` is the F32 view the CPU reference reads (asserts `f32`) |
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

## Snapshot and restore (2026-09-10)

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

The agent keeps snapshots at turn boundaries and restores the longest one
a render starts with, in memory and on disk: see
[§ The agent's token cache](#the-agents-token-cache-2026-10-04).

Why copy instead of rewind: 48 of the 64 layers are recurrent, and their
matrix after token *n* is a function of every token before it. Truncating a
KV length would leave those matrices at token *n* while attention believed
it was at *m < n*. A snapshot is the only rewind that exists, and it is
explicit about its cost.

`engine.Model.snapshot`/`restore` expose the pair on both executors; the
agent's token cache is their caller. The model's pair also carries what a loaded
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

## Checkpoint and rewind (2026-09-19)

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
([§ Pending rows and the verify tape](#pending-rows-and-the-verify-tape-2026-10-01)).

`reset()` and `restore()` clear the recorded position: both rewrite the
state, so the region's bytes are stale and are only read after a fresh
`checkpoint()`. `engine.Model.checkpoint/rewind/truncate` forward on both
executors, and `engine.Model.recover(accepted)` is the accepted-prefix
operation used by the loop: the tape replay when the batch kept one, else
rewind then `prefill(accepted)` on a recurrent model, and
`truncate(checkpoint + accepted.len)` on an attention-only one.

A model with an embedded draft block (Qwen's `blk.64`, [MODL-18](https://github.com/tildaslashalef/nuclis/blob/v0.6.0/docs/worklog.md#modl-18--qwen38-draft-head-the-embedded-prediction-block-on-the-cpu-reference-and-the-metal-plan-2026-09-19--2026-09-20-two-sessions)) adds the
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

## Pending rows and the verify tape (2026-10-01)

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

This replaced [ENGN-14](https://github.com/tildaslashalef/nuclis/blob/v0.6.0/docs/worklog.md#engn-14--recovery-without-the-whole-stack-replay-2026-09-20-two-sessions)'s row checkpoints (a 150 MB recurrent copy per batch
row, 1.25 GB for eight, written by the chunk kernel and restored by a host
copy); see the [worklog](https://github.com/tildaslashalef/nuclis/blob/v0.6.0/docs/worklog.md) for both.

## The agent's token cache (2026-10-04)

The agent re-prefilled tokens it had already computed whenever the session
could not continue where it stood: every start (the system block and tools),
a cancelled step, an effort change, compaction. `src/agent/cache.zig` keeps
snapshots at **boundaries** and `Completer` (`src/agent/loop.zig`) restores
the longest one a render starts with, then prefills the rest. New tokens
(a tool result, the user's message) are never cached; only re-prefill is
removed.

- **Boundaries.** The primed prefix, one per effort (the effort is part
  of the system block), and the end of every turn that ends in an answer
  (`Model.checkpoint`, called by the loop). Both sit before a control token,
  which BPE never merges across, so the remainder encodes the same alone.
  A state with an image in it is not kept: the placeholder text is the same
  for every image.
- **Matching** is on the rendered text the state consumed, not on tokens:
  a turn's generated tokens need not be the canonical encoding of their
  text, so re-encoding the render would miss them. A hit must be a proper
  prefix, leaving a token whose logits the next sample needs. A cached
  state further along than the session wins even when the session could
  continue (after a re-prime under a conversation, the session stands at
  the system block). The penalty history is a 31 KB bitset copied with the
  state.
- **Memory tier**: in the process, least recently used out first under
  `cache.memory_bytes` (4 GiB; 0 keeps none). A Qwen state is 150 MB plus
  64 KiB per token, about 250 MB primed and 1.2 GB at 16K. A context or
  model change (`/model`, Ctrl-W) drops it (the states are bound to the
  engine that made them).
- **Disk tier**: the primed prefix, and the last turn end of each
  conversation that has a session file (each save replaces that
  conversation's previous one), under `<root>/cache/prefix/` through
  `inference.prefix_cache` (header with the keys, content digest; a file
  that fails either is deleted and reads as a miss). The key's token half
  is the digest of the rendered text the state consumed, which is also the
  middle of the file name, so a session's states are found without loading
  a model (`cache.removeDigest`, used by `/resume`'s Ctrl-D and `agent rm`). The
  key's model half hashes the model files (and the draft companion when a
  drafter is loaded), the executor, and the build: version and commit,
  plus the executable's size and modification time for a build from a
  modified tree, since such builds share a commit (`cache.buildId`). A
  different build misses, never restores. `cache.disk_bytes` (8 GiB; 0
  disables) bounds the directory; a hit refreshes a file's modification
  time and each save evicts the oldest beyond the budget.
- **Resume.** The turn's last `assistant` entry records its `boundary`
  (the text's byte length and digest). A resume (`Completer.resumeFrom`)
  tries the file's last boundary before the memory tier: when the render
  starts with it, the state is loaded from disk and the penalty history
  is rebuilt from the text's encoding. The system block carries the date,
  so a conversation resumed on another day misses and replays.
- **Model switch.** `/model` unloads the engine, opens the chosen file,
  empties the memory tier, and re-opens the disk tier under the new
  model's key, so a model used before restores its primed prefix from disk
  (switching back to Qwen3.8 after Gemma 4 12B: 0.1 s, 2026-10-04). The
  conversation is rendered through the new profile and prefilled whole
  (cause `model`), and the session file records a `model` entry beside
  `context` and `effort`.
- **Accounting.** A step that could not continue the session reports a
  `Replay` with its cause (`resumed`, `cancel`, `rewrite` for compaction or
  elision, `effort`, `context`, `model`, `failure`) and the tokens restored; the
  bar shows `replayed <cause> <prefilled>, <restored> restored`, and the
  session file records `replay` and `restored_tokens` beside `replayed`.

**Evidence** (Qwen3.8-27B Q4_K_M, Metal, M4 Pro, 16K window, dirty build of
2026-10-04 on `9f45bf0`). Start-up warm-up of the 1,333-token prefix at
`low` (`make shot`, `.zig-cache/tui/c1-start`, `c8-warm-start`, `c6-low`):
16.3 s prefilled, 1.0 s from disk, 0.4 s from memory. `agent -p` wall time
with the weights in the page cache, three pairs: 17.4 s cold, 2.6 s with
the disk entry. A cancel then a prompt (`c4-after-cancel`): `replayed
cancel 41, 1369 restored`; low → xhigh → low then a prompt
(`c7-after-effort`): `replayed effort 17, 1427 restored`. A cold and a
warm `agent -p` at temperature 0 give the same text with the drafter and
without it. A primed entry on disk is 242–251 MB.

