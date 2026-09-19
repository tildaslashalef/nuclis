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
On the same day KERN-11 was pulled in front of the theme: the verify
batch of ENGN-12 (up to nine token rows) runs through the prefill matmul
tiles, which the Metal reference records at 20–29 GB/s of weight traffic
on short chunks against the matvecs' 200+, so a speedup measured on the
tile as it is would be measured against a known-bad kernel. KERN-11
session 1 (2026-09-19) landed the 8-row × 8-token split-K tile, its
`metal-check` entries, the bench's weight-byte GB/s column, and the first
numbers: the `_8` tile streams 22–88 GB/s at 1–8 tokens against the 32×32
tile's 6–41 at 9–32, roughly 20–50 % of the matvec ceilings (83–252),
below the unit's 70 % floor, and the short-t runs are noisy enough that
the crossover is not yet pinned. Next is KERN-11 session 2: re-measure the
8×8/32×32 crossover with the rounds interleaved, decide
`small_chunk_tokens`, instantiate a `_16` variant if the numbers ask, and
close the unit (the docs tables, the `make bench` records, the log).
Nothing of the speculative-decoding theme is implemented.

Order: KERN-11 → ENGN-11 → MODL-18 → ENGN-12 → MODL-19 → MODL-20. After
MODL-20 the roadmap continues with the performance follow-ups, then
vision, then agent expansion ([docs/roadmap.md](docs/roadmap.md)).

| Unit | Title | Sessions |
| --- | --- | --- |
| KERN-11 | Small-chunk prefill matmul near the weight-bandwidth floor | 2 |
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

## KERN-11 — Small-chunk prefill matmul near the weight-bandwidth floor

**The problem, with numbers.** A chunk of `t ≤ 32` tokens goes through
the `nu_matmul_*_32` tiles (`inference/src/backends/metal/kernels.metal`,
template `nu_matmul_body`, instantiated with `[[host_name]]` near the end
of the file): one threadgroup of 128 threads owns a 32-row × 32-token
output tile, and its K loop takes 64 columns per step with two
`threadgroup_barrier`s per step (after staging, after the multiplies).
On a 5,120-row projection that is 160 threadgroups, each walking 80
sequential steps that move 1 KB of Q4_K weights apiece; the GPU (16
cores) is mostly waiting on latency. Measured: 20–29 GB/s of weight
bytes on the 22-token bench prompt, where the specialized matvecs on the
same encodings reach 200+ GB/s (`make bench-kernels`). Decode is
bandwidth-bound, so a chunk of 1–8 tokens *could* cost about one matvec
pass; today it costs several. This is first-token latency in every chat
turn, and it is the cost of ENGN-12's verify batch.

**Design: a tile shaped like the matvec.** A new instantiation set,
`nu_matmul_*_8` (one per specialized encoding: Q3_K, Q4_K, Q5_K, Q6_K,
IQ3_S, IQ4_XS, Q4_0, PQ2_0, PTQ1_0, plus the generic F32 tile if it
measures well), with this geometry:
- One threadgroup of 128 threads owns **8 rows × 8 tokens** of output.
  Grid: `rows / 8` row tiles (rows % 8 == 0 is already the contract) ×
  `ceil(tokens / 8)` token tiles, so a 5,120-row projection launches 640
  threadgroups and a 17,408-row one 2,176 — the matvec's order of
  parallelism (it launches `rows / 16`).
- **The four SIMD groups split the K range**: SIMD group `sg` takes the
  64-column steps `k0 = 64 · (4j + sg)`. Per step, each lane decodes one
  16-value segment (`row = lane >> 2`, `segment = lane & 3`) through the
  existing `nu_tile_segment<ENC>` into the SIMD group's **own** 8×64 half
  tile in threadgroup memory (1 KB per SIMD group, 4 KB per threadgroup),
  then eight `simdgroup_load` / `simdgroup_multiply_accumulate` pairs
  against B blocks loaded straight from device memory transposed (the
  existing `STAGE == false` path of the body) into one
  `simdgroup_float8x8` accumulator. Because each SIMD group writes and
  reads only its own tile, the K loop needs `simdgroup_barrier` only —
  **no `threadgroup_barrier` in the loop**, which is the latency fix.
- At the end, the four accumulators are stored to threadgroup memory
  (4 × 256 B, reusing the tiles), summed by 64 threads in a fixed order
  (deterministic), and stored with the existing row bound
  (`row0 + r < p.rows`); token rows past `tokens` are computed on the
  padding as today. Columns % 64 == 0 stays the contract; a K range that
  does not divide by 4 steps leaves SIMD groups with one step fewer.
- Host side (`inference/src/backends/metal/root.zig`): the new names
  appended to `kernel_names` **and** to the `Kernel` enum in the same
  order (the pipelines are indexed by enum value); `matmulGeometry`
  returns `{ .rows = 8, .tokens = 8, .half = true }` for them;
  `specializedMatmul` gains a first tier, `tokens <= small_chunk_tokens`
  (a `pub const`, initially 8, set by the measurement), before the ≤ 32
  tier; `matmul` needs no other change (`matmulPadded` stays 64; the
  buffers already hold that many rows).

**What is measured, and the decision it makes.**
- `inference/metal-check.zig` `matmulBench` reports, beside GFLOP/s, the
  GB/s of weight bytes (`region.len / best`) — the number that matters at
  small `t` — and is run as `make bench-matmul ARGS=<t>` for `t` in 1, 4,
  8, 9, 16, 22, 32 on both FFN shapes, all encodings, specialized and
  generic. The ceiling is the same encoding's matvec rate from
  `make bench-kernels` on a rested machine (the reference's methodology
  notes apply: heat, clock ramp, back-to-back rounds).
- The threshold: the `_8` tile serves the range where it beats the 32×32
  tile; if 8 tokens sit at the floor and 9–16 do not, a `_16` variant
  (8 rows × 16 tokens, two B blocks per step, the same split) is
  instantiated and measured before the threshold is set. ENGN-12 picks
  `max_draft_length` so its verify batch (`k + 1` rows) fits one token
  tile of whatever this unit ships.
- `make bench` on Qwen (22-token prompt): prefill tok/s and first-token
  latency before and after, recorded in bench.md; the decode row must not
  move (the tile is not on the decode path).

**Correctness gates.** `metal-check`'s matmul exactness check (the block
that loops `token_counts` over the fixtures, comparing the specialized
tile with the generic F32 tile under the half-rounding bound) gains
token counts 1, 5, and 8 so the new tile is checked on every encoding and
on a partial token tile; the selection checks gain `tokens = 8 →
geometry.tokens == 8` and `tokens = 9 → 32` (or the measured threshold).
Then `make test-metal`, `make check`, `make compare` (the `Hello,` trace
is a short chunk, so the F32/F16 traces exercise the new tile at the
documented tolerances), `make test-generation-metal` (chunked against
stepped), and the Gemma, Bonsai, and Muse compare targets, all of which
share `Backend.matmul`.

**Session 1 (done 2026-09-19).** Landed `nu_matmul_split_t` and the nine
`nu_matmul_*_8` instantiations (Q3_K, Q4_K, Q5_K, Q6_K, IQ3_S, IQ4_XS,
Q4_0, PQ2_0, PTQ1_0), the host names and `Kernel` entries, the 8×8
geometry, and `specializedMatmul`'s `tokens <= small_chunk_tokens` (8)
tier; `metal-check`'s exactness block now covers token counts 1, 5, and 8
and pins the t=8/t=9 selection; `matmulBench` reports weight-byte GB/s and
the selected geometry. First numbers (`make bench-matmul ARGS=<t>`, Apple
M4 Pro, Zig 0.16.0, ReleaseSafe, best of five; `make bench-kernels` for
the ceilings): the `_8` tile streams 22–88 GB/s at t=1–8 against the 32×32
tile's 6–41 at t=9–32, matvec ceilings 83–252, so roughly 20–50 % — below
the 70 % acceptance floor; the t=1/4/8 rows do identical work and vary up
to 2×, and a second run moved them 13 GB/s on average, so the crossover is
not yet pinned. Full table in
[metal-backend.md](docs/reference/metal-backend.md#small-chunk-tile-kern-11-session-1-2026-09-19).
`make check`, `make compare` (f32/f16), `make test-generation-metal`, and
the Gemma QAT / 26B-A4B, Bonsai, and Muse Glimmer compares pass. A first
version stored all four K partials in one shared region of the tile,
racing the still-running K loops of the other SIMD groups; it surfaced as
`NonFiniteResult` in the 6-token remainder chunk of
`test-generation-metal` (never in the fixture exactness test) and is fixed
by storing each partial in its own group's tile.

**Review of session 1 (2026-09-19).** Accepted: the kernel, the host
tiers, the checks, and all gates re-run and passing (`make check`,
`compare` f16 max abs 0.025, `test-generation-metal` chunk-32 max abs
2.6e-3, Gemma QAT f16, Bonsai f16, Muse f16 at their tolerances). Three
findings, which set session 2's order:
1. **The bench measures one dispatch per command buffer.** `matmulBench`
   wraps each `matmul` in its own `begin`/`commit`, and the reference's
   own methodology note (`bench-kernels`, KERN-05) records that isolated
   dispatches measure the GPU's clock ramp, not the kernel; at t ≤ 8 the
   whole matmul is about a millisecond, so the recorded 22–88 GB/s and the
   2× spread between identical rows are the ramp, not the tile. Batch
   16–64 dispatches per command buffer (as `bench-kernels` does) before
   any number is compared with the matvec column or a threshold chosen.
   When a token count spans several token tiles (t = 9 on an `_8` tile),
   the bytes must be multiplied by the token-tile count, as `matmul`'s
   profile attribution already does.
2. **The activation operand costs more traffic than the weights.** Each
   threadgroup loads the chunk's whole activation block for its K range
   transposed and strided straight from device memory (8 tokens × columns
   × 4 B = 160 KB at 5,120 columns) and there are `rows / 8` threadgroups,
   so the gate shape reads about 100 MB of gathered activations against
   44 MB of Q4_K weights, in 8-row × 32-byte gathers per k8 step. Two
   experiments, cheapest first: (a) 16 rows per threadgroup (each SIMD
   group keeps two accumulators; one B load serves two A loads), which
   halves the activation traffic per weight byte and keeps `rows / 16`
   groups, still the matvec's grid; (b) pack the chunk's activations once
   per projection input into a `[k][token]` half layout (a transposing
   sibling of `nu_pack_half`, run once for the projections that share an
   input) so the B loads are contiguous 128-byte rows. Measure each alone
   under the batched bench.
3. **Order the tile's reuse across steps by the spec.** Step i+1's lane
   writes into `own` follow step i's `simdgroup_load` reads of the same
   region, and the final `simdgroup_store` follows the last reads, with
   no barrier between them; lockstep execution makes this hold in
   practice, but a `simdgroup_barrier(mem_threadgroup)` before the writes
   (at the top of the step and before the final store) is the ordering
   the language guarantees and costs nothing measurable. Add it.

**Session 2 (next).** In this order: the batched bench (finding 1) and a
re-measured table; the barrier (finding 3); the two activation-operand
experiments (finding 2), each measured alone; then decide
`small_chunk_tokens` from the 8×8/32×32 crossover and instantiate a `_16`
variant only if the numbers ask for it; `make bench` prefill and
first-token records on Qwen; the Metal reference's kernel and geometry
table rows and the final subsection (the session-1 table stays, marked
as taken before the methodology fix); remove the roadmap's short-prompt
bullet; the log entry. `make compare-gemma4` (the K-quant 12B file)
cannot run while that artifact is absent; the QAT and 26B-A4B Gemma
targets cover the family meanwhile.

**Acceptance.** The `_8` tile computes within the half-rounding bound on
every specialized encoding at 1, 5, and 8 tokens; every gate above
passes; on the two FFN shapes at 1–8 tokens the tile streams weight
bytes at no less than 70 % of the same encoding's matvec rate (the floor
the design claims; below it the unit records why and what is left);
prefill on the 22-token prompt and first-token latency recorded before
and after.

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
