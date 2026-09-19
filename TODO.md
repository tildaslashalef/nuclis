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
roadmap's first theme, speculative decoding across the families, as five
units. ENGN-11 closed the same day in one session (its session 2 was not
needed): `Session.init` can reserve a page-aligned checkpoint region, and
`checkpoint`/`rewind`/`truncate`, `engine.Model.recover`, and the
`recoveryCheck` on every family and both backends implement the
accepted-prefix recovery, documented in
[speculative-decoding.md](docs/reference/speculative-decoding.md) and
[session.md](docs/reference/session.md). On Qwen the region is 150 MB and
checkpoint/rewind cost 3 ms on Metal and 2 ms on CPU; the replay is
bit-identical to sequential decoding on the CPU and within the family's
chunk-versus-step bound on Metal. What still exists from before: the Qwen
adapter binds and validates the 15 embedded `nextn` tensors (351 MB) and
never executes them; both executors return logits for the last token of a
prefill only; the catalogue pins every family's draft companion and every
file is pulled (Qwen's separate head, both Gemma heads, Muse's DFlash
drafter), but `Engine.open` loads one GGUF and sizes no checkpoint region
yet. The accepted configuration (the file per registry entry, the
per-command switch, the draft length) is in
[docs/spec.md § Speculative decoding](docs/spec.md#speculative-decoding).
Nothing of the speculative-decoding theme beyond the recovery contract is
implemented; its five units were rewritten at implementation level on
2026-09-19 after the companions' headers and the reference's speculative
driver were read (the oracle fact is in the theme section). MODL-18
session 1 is in progress: the dense Qwen35 MTP graph and the `draft-mtp`
driver are read and their facts recorded with provenance in
[speculative-decoding.md](docs/reference/speculative-decoding.md#the-qwen38-draft-head-modl-18)
(one correction: the block's `h` input is the target's post-`output_norm`
hidden, not the pre-norm residual); `Binding.draft: ?DraftBlock` binds the
15 tensors on the 65-block file and is null on Bonsai; the CPU runtime
keeps the last token's `h`, sizes the 65-layout session, runs the block
(`draftForward`) and exposes `propose`/`commit` behind
[runtime/draft.zig](inference/src/runtime/draft.zig)'s contract;
`Engine.open` takes a `DraftRequest` (`.embedded` on CPU Qwen; `.file` and
Metal are MODL-18 session 2 / MODL-19). The block matches a pinned
reference trace (positions 0 and 1 of `Hello,`, max abs ≤ 1.1e-5, greedy
tokens equal) via `checkDraft` in `make test-generation`. Remaining in
session 1: `inspect`/`validate` reporting the block as *draft head:
embedded*. Session 2: the Metal block, the compare rows, and `--draft-stats`.

Order: MODL-18 → ENGN-12 → MODL-19 → MODL-20. After MODL-20 the roadmap
continues with the performance follow-ups, then vision, then agent
expansion ([docs/roadmap.md](docs/roadmap.md)).

| Unit | Title | Sessions |
| --- | --- | --- |
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

**The oracle.** The pinned llama.cpp checkout (`7620399f5`, the one
`make compare` already runs) implements speculative decoding with both
draft kinds in `common/speculative.cpp` (`--spec-type draft-mtp` and
`draft-dflash`, `-md <draft file>`, `--spec-draft-n-max`), opens a main
file as an MTP context that runs only the `nextn` layer with its own
attention cache (`LLAMA_CONTEXT_TYPE_MTP`), exposes the target's
pre-norm residual as `llama_get_embeddings_nextn`, and knows the
`gemma4-assistant` and `dflash` architectures. One pinned reference
therefore serves every family's draft traces; no second pin is needed.
Read the driver before implementing each source: `process()` (how the
drafter's cache is filled for accepted tokens), `draft()` (how positions
are proposed), and the model's graph for the block.

**Where the detail goes.** A new
[docs/reference/speculative-decoding.md](docs/reference/speculative-decoding.md)
(started by ENGN-11) holds the recovery contract, the draft contract, each
family's source with its facts and provenance, and the measurements; the
session, Metal, generation, and bench references gain their sections;
[llm-guide.md](docs/llm-guide.md) is extended only when the user asks.
No unit claims a speedup before ENGN-12's benchmark, and negative results
are recorded per family.

## MODL-18 — Qwen3.8 draft head: the embedded prediction block

**Progress (2026-09-19, session 1 partial).** The reference's dense MTP
graph and driver were read and the facts recorded with provenance in
[speculative-decoding.md](docs/reference/speculative-decoding.md#the-qwen38-draft-head-modl-18).
Delivered: `Binding.draft: ?DraftBlock` (15 tensors, null on Bonsai);
`runtime/draft.zig`'s `Drafter` contract; the CPU runtime's 65-layout
session, `draftForward`, `propose`/`commit`, and `drafter()`; a pinned
reference trace (`scripts/reference-generation.cpp --mtp-draft`, the
fixture under `inference/src/models/fixtures/qwen35-mtp/`) checked by
`checkDraft` in `generation-check`; and `Engine.open`'s `DraftRequest`
(`.embedded` on CPU; `.file` and Metal are later units). Remaining:
`inspect`/`validate` reporting the block as *draft head: embedded*, then
session 2 (the Metal block, the compare rows, `--draft-stats`).

**Facts read on 2026-09-19** (from `inference/src/models/fixtures/qwen35-27b.json`
and `scripts/gguf-inventory.py` on the separate head; to be confirmed
from the reference's graph in session 1 and recorded in
`speculative-decoding.md` with provenance):
- Block 64 of the main file is the whole draft head: `nextn.eh_proj`
  [10240 → 5120] Q6_K, `nextn.enorm`, `nextn.hnorm`,
  `nextn.shared_head_norm` [5120] F32, and one full-attention layer of
  the main model's shape — `attn_norm`, `attn_q` [5120 → 12288] Q6_K (the
  merged query and gate, as the main full-attention layers), `attn_k` and
  `attn_v` [5120 → 1024] **Q8_0**, `attn_q_norm`/`attn_k_norm` [256],
  `attn_output` [6144 → 5120] Q6_K, `post_attention_norm`, `ffn_gate`/
  `ffn_up` [5120 → 17408] and `ffn_down` Q6_K. The separate
  `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` holds the same 15 tensors and adds
  only `token_embd`/`output` at **Q3_K** and `output_norm`, i.e. lower
  quantized copies of what the main file already has. **Decision: the
  embedded block is the source; the separate file is not loaded** (the
  catalogue keeps it pinned; the log records why).
- The reference (pinned llama.cpp, `draft-mtp`) opens the *same* main
  file as a second context of type MTP that runs only the `nextn` layer
  with a plain attention cache of its own; the target exposes its
  residual after the last layer and before `output_norm` for every
  position (`t_h_nextn`, `llama_get_embeddings_nextn`); the drafter's
  batch carries both a token id and an embedding row per position: at MTP
  position `p + 1` the pair `(h_p, x_{p+1})`, the block computing
  `eh_proj([enorm(embed(x_{p+1})); hnorm(h_p)])` → the layer → 
  `shared_head_norm` → the shared output head → logits for `x_{p+2}`.
  `process()` fills the drafter's cache for accepted tokens by copying the
  target's `h` rows for the batch (plus one pending row carried across
  batches); `draft()` chains: the drafted token and the block's own
  output hidden state feed the next draft position (qwen35 is the
  single-head mode). Read those two functions and the qwen35 graph
  builder for the exact alignment before writing the CPU reference.
- Q8_0 must be in `qwen35.executableEncoding` for the block's `attn_k`
  and `attn_v` (the generic matvec and the generic F32 tile serve it);
  confirm, and reject the block otherwise as the binder already rejects
  unsupported encodings.

**Session 1 — contract and CPU reference.**
- `inference/src/runtime/draft.zig`: the model-independent contract.
  `pub const Drafter = struct { ptr, vtable }` or a comptime-generic
  family member, whichever the registry's pattern suggests
  (`models/registry.zig`), with: `propose(self, max: usize, out:
  []u32) !usize` (greedy candidates from the state after the last
  committed token; also returns the draft's logits row per position for
  ENGN-12's sampled acceptance, into a caller-owned `[]f32` of `max ×
  vocabulary`), `commit(self, tokens: []const u32, h_rows: ...)`
  (advance the drafter's own state over accepted tokens), `checkpoint`/
  `rewind` (its cache is one more attention layout in the *same*
  `Session`, so it rides on ENGN-11's checkpoint), `reset`, and
  `bytes()` for the load plan.
- `qwen35.zig`: the block's tensors move from "validated, excluded" to a
  `Binding.draft: ?DraftBlock` (the 15 tensors); the text binding is
  unchanged. `qwen35_runtime.zig`: the session's layouts gain one
  attention layout for the block (  `Session.init` on 65 layouts when the
  drafter is requested); `step`/the prefill path keep the target hidden
  of the last token (`h` = `self.normalized` after the `output_norm`
  before `self.mm(self.binding.output, …)`, confirmed from the
  reference's `t_h_nextn`; the block applies its own `hnorm` on top,
  [speculative-decoding.md](docs/reference/speculative-decoding.md#the-qwen38-draft-head-modl-18))
  and, for a batch, every row's `h`; `Drafter.propose` runs the block for one position at
  a time, chained as the reference does; `commit` runs the block over
  the accepted tokens with their `h` rows to fill its cache (the same
  forward, logits discarded). Tests: the block on the CPU against a
  pinned trace from the reference at positions 1..3 of `Hello,` (the
  trace harness of `reference-baseline.md` extended to dump the MTP
  context's `nextn` outputs and draft logits; if the harness cannot reach
  them, the pinned fixture is the reference's greedy draft *tokens* at
  those positions from `llama-completion --spec-type draft-mtp -md
  <main file> --spec-draft-n-max 4` with tracing on, and our CPU logits
  become the pinned oracle for Metal).
- `Engine.open` gains `draft: DraftRequest = .none | .embedded | .{ .file
  = path }`; for `.embedded` the Qwen adapter binds the block, sizes the
  session for it, and constructs the drafter; `nuclis inspect`/`validate`
  report the block as *draft head: embedded*.

**Session 2 — Metal and the statistic.** `qwen35_metal.zig` runs the block
with the existing kernels (the concatenation is two `nu_copy`/pack
dispatches or a strided input; `eh_proj` a 10240-wide matvec; the layer
is the main model's full-attention layer code path over the block's
cache; the head matvec on the block's normalized output, argmax on the
device via `nu_argmax_partial/final`, logits read back only when ENGN-12
samples); reset and rewind tests through `generation-check.zig` (the
drafter's cache rides the recovery check); `make compare` gains the
draft rows (`compare-generation.py` compares the new trace files). The
acceptance statistic, offline: `nuclis generate --draft-stats` (or a
`generation-check` mode) decodes the fixed coding prompts greedily and,
at every step, proposes 4 drafts and counts, per depth, whether draft
`i` equals the token the main model chose `i` steps later; reported as a
per-depth acceptance table in `speculative-decoding.md`. Draft latency
per position and the block's bytes recorded.

**Acceptance.** CPU block outputs match the pinned trace at the stated
tolerance at 3 positions; Metal matches the CPU within the F16-cache
contract (`compare-f32`/`-f16` tolerances for the new rows); reset and
rewind tests pass on both executors; the per-depth acceptance statistic
and draft latency recorded; `make check`, `compare` (main rows
unchanged), `test-generation-metal`; the Qwen decode rate in `make bench`
unchanged with the drafter loaded but switched off.

## ENGN-12 — Batched verification, speculative generation, the switch, the benchmark

**Files.** `inference/src/engine.zig` (`Executor.prefill` line 60,
`runLoop` line 493 with its `gpu_greedy`/`gpu_topk` selection),
`inference/src/models/*_metal.zig` (`Plan.prefill`, which computes the
head for the last row only), `inference/src/models/*_runtime.zig`
(`Runtime.step` already yields full logits per token),
`inference/src/sampling/root.zig` (`Sampler.select`, `Candidate`,
`History`), `src/config.zig` (`Config.Generate`, `ModelEntry`, `Flags`,
`Resolved`, `resolve`, the `leafIndex` provenance), `src/cli.zig` (the
`--think` parsing pattern around line 198 and its tests around 541),
`src/catalog.zig` (`Entry`), `src/bench.zig` (`Sample`, `Report`, `run`),
`src/help.zig`, `docs/spec.md § Speculative decoding`,
`docs/reference/generation.md`, `docs/reference/bench.md`,
`docs/development.md § Configuration file`.

**Session 1 — verify and the loop.**
- `Executor.verify(self, tokens: []const u32, rows: []f32) !void`:
  logits for every token, `rows.len == tokens.len × vocabulary`. CPU: a
  loop of `Runtime.step` with each row's logits. Metal: `Plan.verify`
  runs the chunk path once and the output head over all `tokens.len`
  rows through `Backend.matmul` (the tile KERN-11 shipped; the head is
  `output.weight`, 248,320 × 5,120 on Qwen, one weight pass), then reads
  the rows back (`tokens.len × vocabulary × 4` bytes, ≤ 8 MB at eight
  rows). A `verifyGreedy` variant that returns only per-row argmax
  (`nu_argmax_partial/final` per row) serves the greedy path without the
  readback. `Model.verify`/`verifyGreedy` forward on both executors.
- `runLoop` gains the speculative step, taken when `eng.drafter != null`
  and `settings.speculative`: with `t0` the last chosen token not yet
  fed, `k = min(draft_length, capacity − position − 1, budget_left)`;
  `drafter.propose(k, drafts, q_rows)`; `checkpoint()`; `verify([t0] ++
  drafts[0..k])` (greedy: `verifyGreedy`); accept the longest prefix
  where row `i`'s choice equals `drafts[i]`; the token after the prefix
  is row `a`'s choice (the correction, or the bonus when `a == k`);
  `recover([t0] ++ drafts[0..a])`; `drafter.commit(...)`; then the
  committed tokens `drafts[0..a]` and the new token go through the
  existing per-token path in order — `history.observe`, the `token`
  hook, `isStop`, the budget, the context limit — so a stop token inside
  the batch ends the turn there and the rest is discarded (recover to
  that prefix). Cancellation during `verify` poisons the session exactly
  as a cancelled step does and `resetAll` handles it. The observer's
  `before_step`/`step` hooks fire once per batch with the position.
- Sampled acceptance, `inference/src/sampling/speculative.zig`:
  `Sampler.distribution(logits, scratch, history) → []Candidate` (the
  shaped, normalized distribution the existing `select` draws from,
  factored out of it); `accept(p, q, draft, rng) bool` with probability
  `min(1, p(draft) / q(draft))` (a draft absent from `p`'s candidates is
  probability 0 → rejected); `residual(p, q, scratch) → []Candidate`
  normalizing `max(0, p − q)` (if it is empty, the correction is drawn
  from `p`); the history for row `i` includes the accepted drafts before
  it. The drafter's `q` rows are its logits through the same shaping.
  Unit tests with fixed `p`/`q` tables: counts over 20,000 seeded draws
  match the target distribution within 3 σ, including a zero-probability
  draft, full rejection, and full acceptance.

**Session 2 — configuration and the benchmark.**
- `Config.Generate` gains `speculative: bool = false` and `draft_length:
  usize = 4`; `ModelEntry` gains `speculative: ?bool` and `draft_length:
  ?usize`; `Flags` gains both; `Resolved` gains both, resolved as
  `think` is (defaults, entry, flag) with `config show` provenance;
  `cli.zig` parses `--speculative on|off` and `--draft-length N` for
  `generate`, `agent`, and `bench` (tests beside the `--think` ones);
  `config.max_draft_length` is a host constant equal to KERN-11's token
  tile minus one (7 while the tile is 8 rows), a larger value is
  `InvalidNumber`; `Entry` gains `speculative: bool` and `draft_length:
  u8` (the verdict, written by the benchmark; `config init` copies them
  into the entry). A drafter is loaded whenever the family has a source
  (embedded or the entry's `mtp` file) and either the switch is on or
  `bench` will measure both ways; `generate` with the switch off and no
  drafter needed loads nothing extra.
- `bench`: when a drafter is loaded, each measured run is done twice on
  the same loaded model, switch off then on, and the report carries both;
  `Sample` gains `speculative: bool`, `draft_length: ?usize`,
  `accepted_per_step: ?f64`, `verify_milliseconds: ?f64`,
  `recover_milliseconds: ?f64`; `schema_version` becomes 2 and the text
  report prints the pair with the ratio. The record in `bench.md`: the
  Qwen acceptance workload (512, 4K, 16K, 32,639) greedy and with the
  instruct sampling profile, draft length 2, 4, 7, memory, end-to-end
  tok/s; the Qwen entry's `speculative`/`draft_length` set from it;
  `spec.md`'s section updated with the measured result.

**Acceptance.** Greedy speculative output equals ordinary greedy decoding
token for token on the CPU on the pinned prompts (`generation-check`
gains this) and on Metal within the chunk-versus-step contract with any
divergence recorded by position; the sampled-acceptance unit tests; EOS
inside a batch, the budget inside a batch, partial acceptance,
cancellation mid-batch, and a batch at the context limit, each a test;
`make check`, `make compare`, `make test-generation-metal`; the benchmark
record with the verdict, positive or negative, and the entry updated.

## MODL-19 — Gemma 4 draft heads: the companion file as a second GGUF

**Facts read on 2026-09-19** (`nuclis inspect` and the inventory script on
the 12B's `mtp-gemma-4-12B-it.gguf`; the 26B-A4B's
`MTP/mtp-gemma-4-26B-A4B-it-Q4_0.gguf` is inspected in session 1):
- `general.architecture = gemma4-assistant`, 4 blocks, embedding 1024,
  FFN 8192, 16 query heads, `head_count_kv = [8, 8, 8, 1]`, key/value
  length 512 (256 on sliding), sliding window 1024 with pattern
  `[1, 1, 1, 0]`, RoPE bases 1e6 / 1e4, `nextn_predict_layers = 4`,
  `attention.shared_kv_layers = 4`, `embedding_length_out = 3840`
  (the target's width), `rope_freqs.weight` [256]. 49 tensors, Q4_0:
  `nextn.pre_projection` [7680 → 1024] (the concatenation of the 3840-wide
  token embedding and the 3840-wide target residual, projected into the
  head's width), `nextn.post_projection` [1024 → 3840] (back to the
  target's width for its output head), an own `token_embd` [1024 ×
  262144], `output_norm`, and per block `attn_norm`, `attn_q` [1024 →
  4096], `attn_q_norm` [256], `attn_output` [4096 → 1024], the three FFN
  matrices and their norms, `layer_output_scale` [1] — and **no `attn_k`
  or `attn_v`**: with `shared_kv_layers = 4` the head's layers attend
  through the target's own key/value cache, which is why the KV head
  counts mirror the target's layer kinds. In the reference this is the
  `chain_heads` mode: four trained heads, one per draft step, selected
  by `llama_set_nextn_layer_offset`, so the drafter proposes at most four
  positions per step and each position is one layer's forward.
- What session 1 must read from the reference's `gemma4-assistant` graph
  before code: which target layers' caches each head layer reads (the
  mapping of 4 head layers onto the 48 target layers), what the 1024-wide
  `token_embd` is for (the drafted token's input at the head's width,
  or a tied output), where `layer_output_scale` applies, and whether the
  head reuses the target's `output` head after `post_projection`.

**Design.**
- Loading a second GGUF: `Engine.open`'s `draft = .{ .file = path }` opens
  it with `inference.weights.Mapped.open`, validates `general.architecture
  == "gemma4-assistant"`, `embedding_length_out ==` the main file's
  embedding width, and the vocabulary size, else `DraftSourceMismatch`;
  a missing file is `DraftSourceMissing`; both are typed load errors
  reported by `generate`/`agent`/`bench` and never a fallback. The path
  comes from `models.<name>.mtp` (already a registry field, filled by
  `config init` from the catalogue); `nuclis model ls` shows the
  companion as loaded-by "the draft unit".
- `gemma4.zig` gains `bindDraft(doc) !DraftBinding` for the head; the
  Gemma runtime and plan implement the draft contract: the drafter reads
  the target session's attention rows (read-only) for the shared layers,
  keeps the target's pre-norm residual per position as the Qwen drafter
  does, runs one head layer per proposed position, and shares the
  target's embedding and output head. CPU reference first with pinned
  traces from the reference (`llama-completion --spec-type draft-mtp -md
  mtp-gemma-4-12B-it.gguf`), then Metal on the existing kernel set.
- The 26B-A4B head: inspected first; if it is the same architecture at
  its own width it is the same code with the MoE main model's residual;
  if not, the difference is recorded and the 26B-A4B is measured only if
  it fits the unit's second session.
- The role name: decided here. Recommendation: keep `mtp` (the registry
  key, the sidecar role, `--with mtp`) and document that it names the
  draft source of any mechanism; a rename migrates every sidecar for no
  behavior.

**Acceptance.** Draft traces at the tolerances on both backends for the
12B (and the 26B-A4B if in scope); the load errors tested with a missing
path, a wrong architecture (the projector file), and a mismatched width
(the Qwen head); greedy equivalence on the pinned prompts on both
backends; the benchmark record on the Gemma acceptance workload and the
entries' verdicts; `make check`, the Gemma compare targets unchanged for
the main model.

## MODL-20 — Muse Glimmer DFlash drafter

**Facts read on 2026-09-19** (`nuclis inspect` and the inventory script on
`dflash-kquant.gguf`; to be confirmed from the reference's `dflash`
graph and `draft-dflash` driver in session 1):
- `general.architecture = dflash`, 5 blocks at the target's width 6656
  (a 2.6B drafter), FFN 19968, 32 query heads, 8 KV heads, head 128,
  RoPE 5e5, sliding window 2048 on all five layers, `dflash.block_size =
  16`, `dflash.target_layers = [2, 14, 26, 38, 50]`. Tensors: `fc.weight`
  [33280 → 6656] (the five target layers' input residuals concatenated,
  5 × 6656, projected to one feature per position), `enc.output_norm`
  [6656], per block `attn_norm`, `attn_q` [6656 → 4096] with
  `attn_q_norm` [128], `attn_k`/`attn_v` [6656 → 1024] with
  `attn_k_norm`, `attn_output` [4096 → 6656], `ffn_norm`, the three FFN
  matrices, and a final `output_norm`; **no token embedding and no
  output head**: it shares the target's `token_embd` and `output.weight`.
- The reference's `draft-dflash` driver (not the DFlash2 selector: the
  file has no selector keys) proposes a block in one forward: a batch of
  `n_draft + 1` rows at positions `n .. n + n_draft` whose first token is
  the last committed token and the rest a mask token, one decode on the
  draft context, then the draft tokens are sampled per row from the
  block's logits; `llama_set_embeddings_nextn(ctx_dft, true, masked)`
  supplies the target features. The drafter keeps its own attention
  cache over past positions' features (window 2048). So it fits the
  shared contract as `propose(k)` with `k ≤ 15` in one forward; the
  verify batch is then up to 16 rows, which is the case for KERN-11's
  `_16` tile if it exists, otherwise two 8-row tiles.
- What session 1 must read before code: the mask token id and where it
  comes from (`dflash.*` keys or the tokenizer's reserved tokens), whether
  the block rows attend causally (`dflash.attention.causal`, default
  when absent), `sample_from_anchor`, exactly which residual each target
  layer index contributes (the input of layer `l`, `t_layer_inp`, as
  `muse-glimmer.md` notes), and how the cache is filled for accepted
  positions (`process()`).

**Design.** Session 1: the facts above confirmed and recorded; the fit
decision written into `speculative-decoding.md` (expected: the shared
`propose` with a block; if the driver needs something the contract lacks,
add only that); the Muse runtime and plan retain the input residuals of
layers 2, 14, 26, 38, 50 for the last committed token and for every row
of a batch (device-resident on Metal, one 5 × 6656 row per position);
loading through the same `draft = .{ .file }` path as MODL-19 with
`DraftSourceMismatch` on width or vocabulary; `muse_glimmer.zig`
`bindDraft`; the CPU reference of the drafter with pinned traces from the
reference (`llama-completion --spec-type draft-dflash -md
dflash-kquant.gguf`). Session 2: Metal, the benchmark on the Muse
acceptance workload with draft lengths 4, 8, 15, the catalogue verdict.

**Acceptance.** Traces at the tolerances on both backends; the contract
decision documented; the benchmark record; the verdict, negative if the
drafter does not pay for its verification; `make check`, the Muse
compare targets unchanged.
