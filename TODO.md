# TODO — active plan

This file is the only progress tracker. It holds **unfinished work only**:
where we are, what is next, and the design of each remaining unit. When a
unit closes, its outcome moves to
[docs/engineering-log.md](docs/engineering-log.md) and
its section is deleted here; when the last unit closes, this file is emptied
back to this header. Requirements live in [docs/spec.md](docs/spec.md); the
engine map in [docs/architecture.md](docs/architecture.md); how to build,
test, and measure in [docs/development.md](docs/development.md).

Session protocol (also in [AGENTS.md](AGENTS.md)): read this file first. If
it lists work, summarize *Where we are* and ask the user how to continue. If
it is empty, ask what to work on and write the agreed plan here.


## Where we are

Planned on 2026-09-19, after AGNT-10 closed and emptied the plan: the
roadmap's first theme, speculative decoding across the families, as
five units. Nothing of it is implemented. What exists today: the Qwen
adapter binds and validates the 15 embedded `nextn` tensors (351 MB)
and never executes them; the session has a host-side snapshot and
restore (ENGN-06, a 150 MB copy on Qwen) and no in-place checkpoint;
both executors return logits for the last token of a prefill only; the
catalogue pins every family's draft companion and every file is pulled
(Qwen's separate head, both Gemma heads, Muse's DFlash drafter), but
`Engine.open` loads one GGUF. The accepted configuration (the file per
registry entry, the per-command switch, the draft length) is in
[docs/spec.md § Speculative decoding](docs/spec.md#speculative-decoding).
Next is ENGN-11 session 1: the checkpoint and rewind on the session, and
the recovery test on both backends.

Order: ENGN-11 → MODL-18 → ENGN-12 → MODL-19 → MODL-20. After MODL-20 the
roadmap continues with the performance follow-ups, then vision, then agent
expansion ([docs/roadmap.md](docs/roadmap.md)).

| Unit | Title | Sessions |
| --- | --- | --- |
| ENGN-11 | Speculative state recovery: checkpoint, batch, accept-prefix, rewind on both state kinds | 1–2 |
| MODL-18 | Qwen3.8 draft head: the embedded prediction block on the CPU reference and the Metal plan | 2 |
| ENGN-12 | Batched verification, speculative generation (greedy and sampled), the switch and the draft length, benchmark | 2 |
| MODL-19 | Gemma 4 draft heads: the companion file as a second GGUF, 12B and 26B-A4B | 1–2 |
| MODL-20 | Muse Glimmer DFlash drafter: facts, contract fit, acceptance loop | 2 |

## The theme — fixed before the units (decided 2026-09-19)

**Words.** A *draft source* proposes tokens (an MTP head predicts the next
few from the main model's hidden state; a DFlash drafter proposes a block
at once). A *verify batch* feeds the main model the last chosen token
followed by the `k` drafts in one batched forward and keeps the logits of
every row. The *accepted prefix* is the longest run of drafts the main
model agrees with; the *correction* is the main model's own token at the
first disagreement (or its *bonus* token after the last row when every
draft is accepted). *Recovery* puts the session at the state after the
accepted prefix.

**Two state kinds, two recoveries.** Attention caches rewind by position:
the batch writes rows `[P, P + k + 1)`, a row's content never depends on
later rows, and rows past the position are ignored by contract (already
how `restore` leaves them), so accepting `a` drafts sets the position to
`P + a + 1` and nothing is copied. Recurrent state (Qwen3.8's and Bonsai
2's 48 DeltaNet layers: history and matrix) is a function of every token
fed, so it is *checkpointed* before the batch and, on partial acceptance,
restored and *replayed* over the accepted prefix — a second batched
forward of `a + 1` tokens, which also rewrites the same attention rows.
Never rewind DeltaNet by truncating the position alone. Whether a bounded
scheme beats replay (per-token recurrent checkpoints written by the batch,
`(k + 1) × 150 MB` of device scratch on Qwen) is measured in ENGN-11, not
assumed. Gemma 4 and Muse Glimmer are attention-only, so their recovery is
the position rewind alone.

**The draft contract** is model-independent and lives in the runtime:
propose up to `k` tokens from the state after the last committed token,
commit the accepted prefix (the drafter advances its own state — an MTP
head has its own KV cache over every committed position), rewind with the
session, reset, and report its memory into the load plan. Each adapter
implements it with its family's source: Qwen3.8's embedded block first
(the separate `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` only if inspection proves
the embedded weights insufficient), Gemma 4's companion heads, Muse's
DFlash drafter. Bonsai 2's file drops the block and its catalogue entry
pins no draft companion; whether Qwen's separate head drafts for it (the
residual stream is the same basis, the rotation is folded into the
projections) is an experiment recorded under MODL-18's remaining
limitations, not a promise of this plan.

**Verification** returns the main model's logits for every row of the
batch on both executors; the GPU plan computes the head on all rows of
the chunk and reads them back (9 MB at `k = 8` on Qwen), bypassing the
device top-k path while speculation is on. Greedy acceptance compares the
draft with the row's argmax; sampled acceptance uses the sampler's shaped
distributions (temperature, top-k/p, min-p, penalties with the history
advanced through the earlier drafts of the batch) for both the target `p`
and the draft `q`: accept `d_i` with probability `min(1, p_i(d_i) /
q_i(d_i))`, on rejection sample the correction from
`normalize(max(0, p_i − q_i))`, on full acceptance sample the bonus from
the last row. Identical seeded streams versus ordinary decoding are not a
requirement.

**Configuration** is the accepted design of 2026-09-17, now in
[spec § Speculative decoding](docs/spec.md#speculative-decoding): the
file per registry entry (`models.<name>.mtp`, a typed load error when
missing or mismatched), `generate.speculative` and `--speculative on|off`
on `generate`, `agent`, and `bench`, `generate.draft_length` and
`--draft-length` capped by a host constant; the acceptance rule and the
recovery scheme are not exposed. Whether the `mtp` role is renamed to a
mechanism-neutral `draft` (sidecars, `--with`, the config key) is decided
in MODL-19, when the first companion file is loaded.

**Where the detail goes.** A new
[docs/reference/speculative-decoding.md](docs/reference/speculative-decoding.md)
(started by ENGN-11) holds the recovery contract, the draft contract, each
family's source with its facts and provenance, and the measurements; the
session, Metal, generation, and bench references gain their sections;
[llm-guide.md](docs/llm-guide.md) is extended only when the user asks.
No unit claims a speedup before ENGN-12's benchmark, and negative results
are recorded per family.

## ENGN-11 — Speculative state recovery

**Design.**
- Session 1: `runtime/session.zig` gains a checkpoint region — a second
  copy of every recurrent layer's history and matrix, allocated at `init`
  when the caller asks for it (the memory plan: 150 MB on Qwen, nothing on
  attention-only layouts), plus the checkpointed position. `checkpoint()`
  copies recurrent state and records the position; `rewind(position)`
  restores the recurrent copy and sets the position, refusing a position
  outside `[checkpointed, current]` or a session that is updating or
  failed (`SessionNotReady`), and a rewind without a checkpoint
  (`NoCheckpoint`); a `reset` or `restore` invalidates the checkpoint.
  On the CPU the copy is a `memcpy`; on Metal the session's byte block is
  the shared buffer, the copy is a blit encoded in order with the plan's
  command buffers, and the executor waits for the batch's GPU work before
  a rewind touches host-visible state. `engine.Model` exposes the pair on
  both executors (`checkpoint`, `rewind`) and the accepted-prefix
  operation on top of them: `recover(base, tokens[0..a+1])` — position
  rewind when `a = k` or the layouts are attention-only, otherwise
  rewind to the base and replay the accepted tokens through `prefill`.
  The existing chunked prefill is the verify batch's stand-in: the test
  feeds `k` tokens as one chunk at a checkpointed position.
- Session 2 (if needed): the ordering on Metal, the measurements, and
  the bounded-scheme question. Measure at position 512 and 32,639 on
  Qwen and Gemma 4: checkpoint cost, rewind cost, replay cost per
  accepted length `a ∈ {0, 2, 4, 7}` at `k = 8`, peak memory. If replay
  costs a full step, note it as the price a partial rejection pays and
  leave per-token recurrent checkpoints to the benchmark's verdict in
  ENGN-12.

**Acceptance.** On both backends (`make test-generation` and
`test-generation-metal` extended): step to a position, checkpoint,
feed `k = 8` tokens as one batch, and for every `a` in `0..8` recover to
`a` and step one more token; the logits equal a never-speculated session
that stepped the same tokens, bit for bit on the CPU and within the
documented chunk-versus-step contract on Metal (the difference is the
prefill tile against the matvec, recorded, not hidden). Unit tests: full
acceptance (no copy), immediate rejection (`a = 0`), a cancelled batch
(the session poisons, the checkpoint dies with the reset), errors,
a batch that would exceed the capacity refused before any write,
allocation failure of the checkpoint region, two independent sessions
(one rewound, the other untouched), and the interplay with snapshot and
restore. Memory and costs recorded in the new reference document and
session.md; `make check`.

## MODL-18 — Qwen3.8 draft head: the embedded prediction block

**Design.**
- Session 1: facts. From the inventory fixture and the separate head's
  header (`scripts/gguf-inventory.py` on `MTP/mtp-Qwen3.8-27B-Q4_0.gguf`,
  never loaded by the engine unless the embedded block proves
  insufficient): the 15 tensors of block 64 (`eh_proj` 10240×5120,
  `enorm`, `hnorm`, `shared_head_norm`, and one full-attention layer's
  attention and FFN) and what the block shares with the main model (the
  token embedding and the output head, per `config.json`'s "no dedicated
  MTP embeddings"). The mapping from the architecture's authoritative
  source, confirmed by the reference: the block's input at position `i`
  is `eh_proj([enorm(embed(t_{i+1})); hnorm(h_i)])` with `h_i` the main
  model's residual before its final norm, the layer runs with its own KV
  cache over every committed position, and `shared_head_norm` feeds the
  shared head. The oracle for pinned draft-logit traces is the pinned
  mainline revision if it executes the block, else a second pinned
  revision or implementation recorded in `reference-baseline.md` the way
  MODL-16 pinned the fork. Facts and provenance go to
  `speculative-decoding.md`. Then the draft contract in `inference/`
  (`runtime/draft.zig`, or the name the code suggests), the session
  layout gaining the block's attention cache (one more attention layer,
  checkpointed and rewound with the rest), the runtime keeping `h` of
  the last committed token (and of every row of a batch, for ENGN-12),
  and the CPU reference in `qwen35_runtime.zig`; `make compare` gains
  the draft rows at several positions of `Hello,`.
- Session 2: `qwen35_metal.zig` runs the block on the existing kernel
  set (the concatenation is a layout step; `eh_proj` a 10240-wide matvec;
  the head matvec on the draft's hidden state, which dominates the draft's
  cost — `k` drafts cost `k` head matvecs plus `k` block forwards), with
  the draft's argmax on the device; draft state reset and rewind tests;
  the offline acceptance statistic: ordinary greedy decoding of the
  fixed coding prompts with a draft taken at every step and compared to
  the token the main model then chose, per draft depth.

**Acceptance.** CPU and GPU draft logits match the pinned traces at the
stated tolerances at several positions; tests cover reset and recovery;
draft latency per token, extra memory, and the acceptance statistic per
depth recorded. No speedup claimed. `make check`, `compare` (main-model
rows unchanged), `test-metal`.

## ENGN-12 — Batched verification, speculative generation, the switch, the benchmark

**Design.**
- Session 1: `verify(tokens, rows)` on both executors — the CPU runtime
  keeps each step's logits; the Metal plan's chunk path computes the head
  on every row of the chunk (the prefill tile already runs every row
  through the layers; the head is the exception computed for the last
  row) and reads the rows back, argmax per row on the device for the
  greedy path. `engine.runLoop` gains the speculative step when a drafter
  is loaded and the switch is on: propose `k` (shortened to what the
  capacity and the token budget allow), verify, accept the longest
  matching prefix, take the correction or the bonus, recover per
  ENGN-11, commit the drafter, and emit the committed tokens in order
  through the existing hooks — stop tokens, the budget, the context
  limit, and the penalty history are per committed token, so a stop
  inside the batch discards what follows it. Sampled acceptance as the
  theme fixes it, in `sampling/`: the shaped `p_i` and `q_i` on the host
  from full rows, the draws from the seeded sampler's stream, the
  residual distribution for the correction.
- Session 2: configuration — `generate.speculative` (default from the
  catalogue entry's verdict, off for a bare path), `generate.draft_length`
  (per-family default from the entry, capped by `max_draft_length`), the
  flags on `generate`, `agent`, and `bench`, layered as `think` is
  (defaults, entry, flag) with `config show` provenance and `--help`;
  `bench` measures ordinary and speculative decode on the same loaded
  model in one process and reports draft length, accepted drafts per
  step, verification and recovery cost, memory, and end-to-end tok/s; the
  record in bench.md against the Qwen acceptance record's workload (the
  fixed coding prompts at 512, 4K, 16K, 32,639); the Qwen entry's default
  set from the measurement.

**Acceptance.** Greedy speculative output equals ordinary greedy
decoding on the pinned prompts, token for token on the CPU and within
the backend contract on Metal (a divergence caused by chunk-versus-step
numerics is recorded with its position); synthetic tests of the
acceptance and correction distribution over fixed `p` and `q` (counts
over many seeds, including a zero-probability draft, full rejection,
full acceptance); tests of EOS inside a batch, the budget inside a
batch, partial acceptance, cancellation mid-batch, and context capacity;
`make check`, `make compare`, `make test-generation-metal`; the benchmark
record with the verdict, positive or negative.

## MODL-19 — Gemma 4 draft heads: the companion file as a second GGUF

**Design.** Facts first: `scripts/gguf-inventory.py` on the 12B's
`mtp-gemma-4-12B-it.gguf` and the 26B-A4B's
`MTP/mtp-gemma-4-26B-A4B-it-Q4_0.gguf` — architecture key, tensors, the
shared tensors they reference, whether the 26B-A4B head carries experts —
and the reference's loader for how the head is driven; recorded with
provenance before any code. Loading: `Engine.open` gains the optional
draft-source path (the accepted seam), resolved from `models.<name>.mtp`
(`config init` fills it from the catalogue) and verified at load — a
missing file, a foreign architecture, or a vocabulary or width that does
not match the main file is a typed load error, never a fallback or a
download; the head's memory joins the load plan. The Gemma adapter
implements the draft contract for both entries, CPU reference then Metal,
traces from the reference at several positions. The role name (`mtp` or
`draft`) is decided here for the sidecars, `--with`, and the config key,
with the recommendation to keep `mtp` and document that it names the
draft source, since a rename migrates every sidecar for no behavior.
Measurements as ENGN-12's benchmark on both entries; each catalogue
verdict set from its own numbers.

**Acceptance.** Traces at the tolerances on both backends for both
entries; the load errors tested with fixtures; greedy equivalence on the
pinned prompts; the benchmark record and the verdicts; `make check` and
the Gemma compare targets unchanged for the main model.

## MODL-20 — Muse Glimmer DFlash drafter

**Design.**
- Session 1: facts from `dflash-kquant.gguf` and the reference —
  architecture, tensors, the block size it proposes, which of the
  target's per-layer input residuals it consumes (the reference exposes
  every layer's `t_layer_inp` for it), whether one forward proposes the
  block or a denoising loop does, and how the reference accepts the
  block; recorded with provenance. Then the fit decision, written into
  `speculative-decoding.md`: if one forward proposes a block, it is the
  shared contract's `propose` with `k` the block size; if it needs its own
  loop, the contract gains only what the facts require or Muse keeps its
  own acceptance loop beside the shared one. The Muse runtime and plan
  keep the residuals the drafter reads for the last committed token (and
  the batch's rows), device-resident on Metal.
- Session 2: the drafter on the CPU reference and Metal, traces from the
  reference at several positions, the benchmark on the Muse acceptance
  workload, the catalogue verdict.

**Acceptance.** Traces at the tolerances on both backends; the contract
decision documented; the benchmark record; the verdict, negative if the
drafter does not pay for its verification; `make check`, the Muse compare
targets unchanged.
