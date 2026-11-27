<!-- kind: reference -->
# Speculative decoding: recovery contract, draft sources, measurements

The draft contract, how the session recovers a partly accepted batch, the
five draft sources, the verify batch, the loop, and the switch. The
requirements: [spec § 5.6](../spec.md#56-speculative-decoding); how each
piece was built and measured: [the record](../benchmarks/record/speculative-decoding.md).

Every catalogue entry with a draft source is measured, and its defaults
follow the record
([re-priced verdicts, 2026-10-01](../benchmarks/record/speculative-decoding.md#the-re-priced-verdicts-2026-10-01),
[the DFlash2 drafter, 2026-10-21](../benchmarks/record/speculative-decoding.md#the-qwen-dflash2-drafter-2026-10-21),
[the 26B-A4B's experts, 2026-11-18](../benchmarks/record/gemma4.md#what-the-experts-cost-2026-11-18),
[the two chat models, 2026-11-20](../benchmarks/record/benchmarks.md#the-two-chat-models-with-their-drafters-2026-11-20)):

| Entry | Default | Draft source |
| --- | --- | --- |
| Qwen3.8-27B | on at draft 7 | the DFlash2 drafter; `--draft mtp` selects its `nextn` block, also on at 7 |
| Gemma 4 E4B | on at draft 6 | its assistant head |
| Gemma 4 26B-A4B (archived) | on at draft 4 | its assistant head |
| Gemma 4 12B QAT (archived) | on at draft 5 | its assistant head |
| Muse Glimmer | on at draft 6 | its DFlash drafter |
| Inkling 2 Mini | off | its DSpark drafter |

## The draft contract

`inference/src/runtime/draft.zig` owns the model-independent interface. A
family exposes a `Drafter`: a host pointer, the target hidden width
`hidden` it consumes, its block bound `max_proposals`, and five function
pointers.

- `propose(token, out)` returns greedy candidates chained from the state
  after the last committed token. There is no draft distribution:
  acceptance draws from the target alone ([§ The loop](#the-loop)).
- `commit(tokens, h_rows)` advances the drafter over tokens the main model
  committed, using their target hidden rows (`hidden` floats per token).
- `reset()` clears the drafter's state.
- `carried()` is the state it keeps outside the session (the MTP heads'
  pending target hidden row; empty for DFlash), which
  `Model.snapshot`/`restore` copy with the session.
- `bytes()` reports the workspace it owns beyond the session's.

There is no `rewind`: a drafter's own attention cache is one more layout in
the same `Session`, so the session's checkpoint, rewind, and truncate cover
it, and the generation loop only calls `reset` on a session reset.
`max_proposals` is 7 for the MTP blocks, the Gemma heads, DFlash2, and
DSpark, and 15 for Muse's DFlash block; `propose` never returns more and `verify`
refuses a batch above `max_proposals + 1` rows.

## The recovery contract

A verify batch feeds the main model the last chosen token followed by `k`
drafts in one forward and keeps every row's logits. If `a` drafts are
accepted, the session must end at the state after them. The two state kinds
recover differently:

- **Attention caches rewind by position.** A row's content never depends
  on later rows, so `truncate(P + a + 1)` sets the position and nothing is
  copied; rows past it are ignored by contract. Gemma 4 and Muse Glimmer
  (attention-only) recover this way, in about a microsecond.
- **Recurrent state is restored from the verify tape.** Qwen3.8's 48
  DeltaNet layers and Inkling 2's 48 convolution histories are a function
  of every token fed; truncating the position alone would leave them at
  `P + k + 1`. The verify batch keeps a frozen state and a per-token tape,
  and recovery replays the accepted prefix from it
  ([session.md § Pending rows and the verify tape](session.md#pending-rows-and-the-verify-tape)).

`engine.Model.recover` is the accepted-prefix operation; it does nothing
when every draft was accepted (the batch fed exactly the accepted prefix).
`generation-check` exercises it per accepted length on every family with a
draft source, on both executors: bit-identical to sequential decoding on
the CPU, and within the family's chunk-versus-step bound on Metal
(`make gate NAME='qwen38-generation-*'`).

## Draft sources

`Engine.open` takes a `DraftRequest`: `.none` (no drafter weights, scratch,
or cache), `.embedded`, `.file = path`, or `.preferred` (the family's own
choice: an embedded block, else the entry's companion). A companion's
binding carries its own `weights.View`, never the target's. A missing
source is `DraftSourceMissing`; a companion of the wrong architecture,
width, vocabulary, or configuration is `DraftSourceMismatch`.

### Qwen3.8's prediction block

The main file's 65th block is one dense full-attention decoder layer of the
main model's shape plus four `nextn` tensors, sharing the main
`token_embd` and `output` head.

- **The pair.** At position `p` the block consumes the token `x_p` and the
  target hidden `h_{p-1}`, the final hidden after `output_norm` (the input
  to the target's head):
  `concat = [enorm(embed(x_p)); hnorm(h_{p-1})]`, projected by
  `nextn.eh_proj` [2·5120 → 5120].
- **The layer.** The main model's full-attention layer (merged query and
  gate, per-head q/k norms, 24 query and 4 KV heads of 256, RoPE base 1e7
  with sections `[11, 11, 10, 0]`, `sigmoid(gate)` on the attention output)
  over the block's own cache layout, then the dense SiLU-gated FFN
  [5120 → 17408 → 5120], `shared_head_norm`, and the shared head.
- **Drafting and commit.** The first draft pairs the seed with the carried
  target hidden of the last committed token; each later draft pairs the
  drafted token with the block's own output hidden. `commit` fills the
  block's cache from the target's hidden rows, never the block's, and
  carries the last one. On Metal a commit of two or more tokens is one
  command buffer.

The separate `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` companion carries the same
block at lower precision (`nextn.eh_proj` and `ffn_*` Q4_K, `attn_*` Q6_K).
`qwen35.bindDraft` binds its `blk.*` tensors onto an unrotated target of the
same shape and leaves its embedding and head unbound, so it shares the
target's as the embedded block does; `Binding.viewOf` picks the companion's
mapping for the block's tensors, so the draft forward has one path.
`.preferred` keeps a file's own block and takes the companion only for a
file without one; `.file` replaces an embedded block. A rotated (Bonsai)
target is refused with `DraftSourceMismatch`.

### Gemma 4's assistant heads

The companion is a GGUF with `general.architecture = gemma4-assistant`
(`inference/src/models/gemma4_assistant.zig`): four trained blocks, one per
chain step, with three pinned widths keyed by the target's (the 12B's 3840,
the 26B-A4B's 2816, the E4B's 2560). A file whose keys differ from the
pinned configuration is refused.

- **The pair.** `[sqrt(n_embd_out)·embed(x_p); h_{p-1}]`, the target's
  token embedding and its post-`output_norm` hidden, through
  `nextn.pre_projection`; neither half is normalized.
- **The layer.** The head has no `attn_k`/`attn_v`: each block reads the
  target's own cache, the sliding blocks layer `n_layer − 2` (visible rows
  `P − 1023 … P − 1`) and the global block layer `n_layer − 1`
  (`0 … P − 1`), and writes neither. Attention scores are unscaled; the FFN
  is tanh-GELU gated; `layer_output_scale` multiplies each block's output.
- **The outputs.** `output_norm` feeds the head's own tied `token_embd`
  (no soft-cap) for the greedy candidate and `nextn.post_projection` for
  the next step's `h`. Every step runs at the proposal position.
- **Commit and recovery.** `commit` copies the last target hidden into the
  pending row; the drafter owns no cache, so recovery is the attention
  rewind alone.

`Engine.open` requires the architecture, the target's width and vocabulary
count, and a global-layer `rope_freqs.weight` equal to the target's (the
Metal plan reuses the target's rope tables). The Metal `verify` applies
Gemma's final soft-cap `30 · tanh(x / 30)` to every row, as `step` does.

### Muse Glimmer's DFlash drafter

The companion `dflash-kquant.gguf` (`inference/src/models/dflash.zig`) is a
2.6B drafter of architecture `dflash`: 5 blocks at width 6656, 32 query
heads of 128 over 8 KV heads, `block_size` 16, target layers
`[2, 14, 26, 38, 50]`, every layer sliding at 2048, NeoX RoPE at base 5e5,
mask token 201818. It has no embedding and no head: it uses the target's.
The adapter refuses a file that carries any `dflash.*` key outside the
pinned set (a causal or anchor key, a selector or convolution key, a rope
dimension count), since each would change the equations.

- **Commit is the encoder.** The drafter's cache is filled from the
  target's residuals: the inputs of the five target layers, concatenated
  (`Drafter.hidden` = 33,280), through `fc.weight` and `enc.output_norm`,
  then each block's `attn_k`/`attn_v`, key norm, and RoPE at the target
  position.
- **Propose is one block.** One forward over 16 rows at `P … P + 15`, the
  anchor then 15 mask tokens, non-causal inside the block and windowed
  back over the injected prefix; the target's `output` head gives the
  per-row argmax. `propose(k)` is one forward for any `k ≤ 15`.
- **No removal.** The proposal's attention ends at the block's last
  position, so rows a previous proposal left past it are never read and
  `commit` overwrites the accepted ones; `reset` is a no-op and recovery is
  the position rewind.

On Metal the language model captures each target layer's input rows into
one device slab per slot during `prefill`/`verify`/`verifyGreedy`; the
block's attention is one `attentionDecode` per row, since the causal chunk
kernels cannot express its mask. The five draft caches hold full capacity
in the session's `kv` precision.

### Qwen3.8's DFlash2 drafter

`Qwen3.8-27B-DFlash2-Q8_0.gguf`, the index's `qwen3.8-27b` draft variant
`dflash2` (`--with draft=dflash2`), is a block drafter trained against
Qwen3.8-27B (`inference/src/models/dflash2.zig`). Its architecture id is
`dflash`, Muse's; the selector keys tell the two apart. The equations come
from `z-lab/dflash` `dflash/model.py`, the format's only definition.

- **Base.** Muse's drafter at Qwen's width: five Qwen3 blocks of 32 query
  and 8 KV heads of 128, q/k norms, RoPE θ 1e7 over the whole head,
  non-causal within a 2048 window both ways, the residuals entering target
  layers 6, 20, 34, 48, and 62 encoded into keys and values (25,600 floats
  per commit row), and an 8-row block `[anchor, mask × 7]` (mask 248070)
  through the target's embedding and head.
- **Two dynamic convolutions per block**, around attention and around the
  FFN: the taps are `conv_proj(x)` viewed `[half][tap][group of 16]`, added
  to `*_conv_base` (stored `[embedding, 2, 2]` in the GGUF); row −1 is
  zero and the context rows take no convolution.
- **The candidate selector.** At each draft position it scores the 16
  largest logits `u_k` as `u_k + Σ_r pred[prev][r] · h[r] · succ[cand_k][r]`
  with `h = selector_hidden(row)` (rank 256), and the greedy chain takes
  the highest.

`qwen35.attachDraft` takes a DFlash2 companion over the file's own block;
the family declares `attachDraft` instead of `bindDraft` so it can choose
by content. The CPU reference is `dflash2_runtime.zig`; on Metal (`Flash`
in `qwen35_metal.zig`) the convolution is `nu_dflash_conv`, the drafter's
Q8_0 projections take `matmul`'s Q8_0 tiles as any family's do, and the
selector runs on the host over a device top-16 per row.

### Inkling 2's DSpark drafter

`inkling-2-mini`'s draft variant `dspark` (`dspark.zig`, architecture
`dflash`; `markov_w1` and `mask_embd` tell it from Muse's and Qwen's) binds
through `inkling2.attachDraft` when its taps `[5, 23, 41, 48]` fit.

- **Block.** `[anchor, mask × 7]` at `P … P + 7`: the anchor's target
  embedding through `token_embd_norm`, then `mask_embd`'s rows as stored.
  Three blocks of 24 query and 8 KV heads of 128, q/k norms, NeoX RoPE θ 1e6
  over the whole head, non-causal inside the block; blocks 0 and 2 slide at
  1,025 counted from the anchor for every row (the anchor and the 1,024
  committed rows before it), block 1 every one; 8 rows for any `k ≤ 7`.
- **Commit.** A row is `cur` entering applications 5, 23 and 41 (after a
  loop pass's reset) and the head's normed input: 9,216 floats. Each block
  keys `k_ctx_norm(attn_k_ctx(c))` with RoPE and values `attn_v_ctx(c)`, for
  `c = ctx_norm(Σ ctx_mix[i] · row_i)`; `f_rnn = markov_ctx_norm(Σ
  markov_ctx_mix[i] · row_i)` of the last committed row is `carried`, tagged
  with its position. A block whose previous row was not committed drafts
  nothing.
- **The Markov chain.** Draft `i` (rows 1 … 7) starts from the target's
  head on row `i`'s hidden, uncapped; `joint_proj · [state; markov_w1[prev];
  hidden_i; f_rnn]` gives a gate, a candidate and the Markov feature
  (`state ← cand + σ(g) · (state − cand)`), `markov_w2` its bias over the
  full vocabulary, and `30 · tanh((base + bias) / 30)` the draft's logits;
  `prev` is the anchor, then the draft before.

Three full-capacity caches follow the target's 96 (a block writes 8 rows
past a window's compaction). The CPU reference (`dspark_runtime.zig`)
recovers by rewind and replay, Metal (`dspark_metal.zig`) from the verify
tape. The Metal plan captures commit rows in `prefill` and `verify*`
(`nu_dspark_mix` mixes them); one command buffer runs the block, its 8 rows
as 192 query heads of one `attentionDecode` (`gatherRows` folds them into
each KV head's group: the mask is not causal), and the chain, per draft
`ttsEmbedRow` of `markov_w1[prev]` by the device-held id, `nu_dspark_markov`,
the BF16 `markov_w2` matvec, the cap, and `argmax`. The entry is off ([record](../benchmarks/record/inkling2.md#the-dspark-drafter-on-metal-2026-11-27)).

## The verify batch

`Model.verify(tokens, vocabulary, output, hidden, observer)` runs the batch
and, when `hidden` is given, reports per row what `Drafter.commit` consumes
(the post-`output_norm` hidden, or the DFlash residuals). Its output is one
of two modes (`engine.VerifyOutput`):

- `rows`: every row's target logits (`tokens.len × vocabulary`);
- `topk`: one `sampling.TopK` per row from the device partial top-k, with
  every row's logits left resident so `Plan.readVerifyRow(row, out)` serves
  a fallback.

`verifyGreedy` returns each row's argmax from the device without a logit
readback. The CPU reference loops `Runtime.step`; a Metal plan records the
layer stack once (shared with `prefillChunk`) and runs the head over all
rows through the batched matmul. A batch is at most the plan's chunk and
its `max_verify_rows` (16 on Qwen; `max_draft_proposals + 1` on Gemma and
Muse). `Backend.matmul` routes batches of up to `small_batch_rows` (3) rows
of the specialized encodings to the multi-row matvec and larger ones to the
tile.

## The loop

`runLoop` takes `Speculative{ enabled, draft_length }`. When a drafter is
loaded and the switch is on, it commits the prompt to the drafter through
`engine.commitPrompt` in `prefill_chunk` chunks, then each step:

1. proposes `k = min(draft_length, max_proposals, capacity − position − 1,
   budget_left)` drafts, stopping the chain after the first position whose
   top-candidate probability (the full-vocabulary softmax at the argmax) is
   below `engine.draft_p_min = 0.7` (that position is still proposed; `0`
   proposes every requested position);
2. checkpoints, verifies `[seed] ++ drafts`, and accepts the longest
   prefix;
3. recovers the accepted prefix, commits it to the drafter, and emits the
   accepted drafts and the correction through the ordinary per-token
   checks. The correction is carried as the next step's seed.

EOS, the budget, or the context limit inside a batch ends the turn there
and the session recovers to the emitted prefix; a cancelled `verify`
resets the session and reports `.cancelled`; a context with no room for a
batch stops at `context_limit`. A per-layer observer disables speculation.
The drafter is asked only while `Speculative.enabled` is set, so a loaded
block costs nothing when the switch is off.

**Acceptance.** Greedy compares `verifyGreedy`'s argmax with the draft.
Sampled acceptance (`inference/src/sampling/speculative.zig`) draws the
target's own token from row `i`'s shaped distribution and accepts the
draft when they agree; otherwise the draw is the correction, and after the
last accepted row it is the bonus. Every emitted token is a draw from the
target, which makes the rule exact for greedy draft chains. The history
advances through the accepted drafts, so row `i`'s penalties see the
drafts before it. The sampled path decides from the `topk` readback with
`Sampler.selectFromHistory`, applying penalties on the host to the 256
candidates; it falls back to the resident row and `Sampler.distribution`
(counted in `Timing.topk_fallbacks`) for `repetition_penalty < 1`,
`presence_penalty < 0`, a penalized greedy draw, a nucleus or full
distribution without `top_k` under a penalty, a retained set that fails
the bound, or, without penalties, any row `selectFrom` defers (a nucleus
or a full-distribution draw past the 256 candidates or inside
`total_band`). Ordinary decoding with
penalties applies them on the device (`nu_penalize`, see
[sampling.md](sampling.md)).

The agent's `Completer.prime` commits the primed prefix through
`commitPrompt`, so the primed snapshot carries the drafter's cache rows,
and the status bar shows the turn's mean accepted drafts per batch
(`spec N.NN/step`).

## The switch and the bench pair

`generation.speculative` (default off) and `generation.draft_length`
(default 4, 1 ≤ n ≤ `engine.max_draft_length` = 15, the largest block any
family ships) in `nuclis.json`, with per-entry overrides under
`models.<name>.generation` and `models.<name>.draft` (a draft variant or a
file), and `--speculative on|off`, `--draft-length N`, and `--draft` on
`generate`, `agent`, and `bench`, resolved defaults → entry → flag with
`config show` provenance. Above the host bound is `InvalidNumber`; above
the loaded drafter's `max_proposals` is `InvalidDraftLength` at run time.
A configuration file keeps the values it was written with; `nuclis config
init` writes the current defaults.

`generation.draft_max_depth` stops drafting by depth, because a verify
batch's rows each attend over the whole context while the tokens it saves
do not grow: both Gemma 4 drafters fall below 1× past 32K
([record](../benchmarks/record/gemma4.md#supporting-65536-2026-11-19)).
`Speculative.max_position` is the cap; a run whose prompt ends at or past
it decodes plainly, with the GPU's selection shortcuts, and one that starts
below drafts to its end, since plain steps after a batch would sample every
token from full logits. The value resolves flag → entry → global → the
draft file's verdict (`Verdict.max_depth`, never copied into an entry), 0
meaning none. The verdict is this machine's measurement and a default: on
other hardware `make spec-matrix` (which passes `--draft-max-depth 0`)
measures the crossover again. Skipping the drafter past the cap is exact:
Gemma 4's heads keep no rows of their own, and their one carried hidden
row is rewritten by the next prompt's commit; a drafter with its own rows
would only accept less after a rewind below the cap.

The commands open the model with `.preferred` when the switch is on and
`.none` otherwise. `bench` with the switch on measures every run as an
off/on pair on the one loaded model, so the pair's ratio is the speedup
claim; `--draft-p-min` overrides the stop threshold there. Its samples
carry `speculative_steps`, `proposed_per_step`, `accepted_per_step`, and the
per-phase milliseconds (`propose`, `verify`, `accept`, `checkpoint`,
`recover`, its rewind/replay split and `recover_by_length`, `commit`); its
report carries `decode_speedup`. `make spec-matrix` runs real speculation
on saved prefixes across contexts, draft lengths, and samplings, and is
how the defaults above are set.

## The Qwen verify budget

Rate is total emitted tokens over total decode seconds. A batch that emits
E tokens on average and costs C milliseconds in all (propose, verify,
recover, commit, the loop) reaches 20 tokens/s only when **C ≤ 50E**. Sum
counts and times; never average per-batch rates. The verify budget is 50E
less the batch's measured non-verify costs. `scripts/bench/spec-matrix.py`
prints C / 50E for every cell. Its last pricing, the alternatives weighed,
and the external evidence surveyed are in
[the record, 2026-10-01](../benchmarks/record/speculative-decoding.md#the-qwen-verify-budget-2026-10-01).

## Evidence

| Gate | What it holds |
| --- | --- |
| `qwen38-generation-{metal,cpu}` | recovery per accepted length; the block's checkpoint and reset (`draftRecoveryCheck`) |
| `qwen38-{draft,mtp-companion}-trace-{metal,cpu}` | the embedded block's rows against `inference/src/models/fixtures/qwen35-mtp/`; the companion's on the Q4_K_M against the same rows, equal greedy drafts |
| `qwen38-draft-stats` | per-depth and per-`p_max` acceptance on two fixed code prompts |
| `qwen38-speculative-{metal,cpu}` | greedy equivalence and the loop's edges: partial acceptance, budget, EOS, cancellation, context |
| `qwen38-dflash2-{trace,speculative}-{metal,cpu}` | DFlash2 on Metal against the CPU rows in `tests/fixtures/qwen38-dflash2`; the same loop checks |
| `gemma4-e4b-draft-trace-{metal,cpu}` | the E4B head's rows (`inference/src/models/fixtures/gemma4-mtp-e4b/`) and every `verify` row against the stepped logits |
| `inkling2-dspark-{trace-cpu,trace-metal,trace-metal-generic,speculative-cpu,speculative-metal}` | the DSpark block's rows and greedy chain against `tests/fixtures/inkling2-dspark`; the loop checks |

`generation-check --metal` also runs `verifyTopKCheck`, which requires the
`topk` readback's decisions to equal `select` on the `rows`-mode row, and
`sampling/root.zig` tests `selectFromHistory` against `select`.

## Why it is this way

- Why recurrent state is not truncated, and what recovery cost before the
  tape: [record, 2026-09-19](../benchmarks/record/speculative-decoding.md#the-recovery-contract-2026-09-19),
  [2026-09-20](../benchmarks/record/speculative-decoding.md#recovery-by-accepted-length-2026-09-20).
- Why acceptance draws from the target rather than `min(1, p/q)`:
  [record, 2026-09-20](../benchmarks/record/speculative-decoding.md#the-verify-batch-and-the-loop-2026-09-20).
- Why `draft_p_min` is 0.7 and the adaptive length was dropped:
  [record, 2026-09-20](../benchmarks/record/speculative-decoding.md#the-proposal-policy-2026-09-20).
- Why the multi-row matvec takes only small batches and the wide tile was
  not adopted: [record, 2026-09-20](../benchmarks/record/speculative-decoding.md#small-batches-the-multi-row-matvec-and-the-2-row-routing-2026-09-20),
  [the tile experiment](../benchmarks/record/speculative-decoding.md#the-small-batch-tile-experiment-2026-09-20-closed-negative).
- Why DFlash2 is not the default:
  [record, 2026-10-21](../benchmarks/record/speculative-decoding.md#the-qwen-dflash2-drafter-2026-10-21).
- Why Gemma 4 26B-A4B was off, and what turned it on (a verify batch's
  experts through the gathered matvec):
  [record, 2026-10-01](../benchmarks/record/speculative-decoding.md#the-re-priced-verdicts-2026-10-01),
  [2026-11-18](../benchmarks/record/gemma4.md#what-the-experts-cost-2026-11-18).
- How each draft source was traced and validated:
  [Qwen's block](../benchmarks/record/speculative-decoding.md#the-qwen38-draft-head-2026-09-20),
  [the MTP companion](../benchmarks/record/speculative-decoding.md#the-qwen-mtp-companion-2026-10-21),
  [Gemma's heads](../benchmarks/record/speculative-decoding.md#the-gemma-4-assistant-heads-2026-09-21),
  [Muse's drafter](../benchmarks/record/speculative-decoding.md#the-muse-glimmer-dflash-drafter-2026-09-21).
