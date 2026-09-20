# Speculative decoding: recovery contract, draft sources, measurements

The reference document of the speculative-decoding theme. Its requirements
are in the [spec](../spec.md#speculative-decoding); the units that fill it
in are planned in [TODO.md](../../TODO.md); closed outcomes are cited from
the [engineering log](../engineering-log.md).

As of 2026-09-20 the recovery contract (ENGN-11), the Qwen3.8 embedded
prediction block (MODL-18), and speculative generation with its switch,
draft length, and benchmark record (ENGN-12) are implemented and measured.
What exists: the draft contract in `runtime/draft.zig`; the Qwen adapter
binds the 15 embedded `nextn` tensors and both executors run the block
([qwen-validation.md](qwen-validation.md)); the session has a host-side
snapshot and restore and an in-block checkpoint ([session.md](session.md));
every family's draft companion is pinned and pulled
([artifacts.md](artifacts.md)); the verify batch, the sampled acceptance
module, and the `runLoop` speculative step are in place, checked for greedy
equivalence on both executors, and switched by `generation.speculative` /
`--speculative on|off` with `generation.draft_length` / `--draft-length`;
`bench` measures the off/on pair on one loaded model. The record
([bench.md](bench.md#speculative-decoding-record-engn-12-2026-09-20)) keeps
the switch off by default: the per-batch costs the plan's performance units
attack are tabulated in `TODO.md`. Gemma 4's and Muse Glimmer's own draft
sources are still to come (MODL-19, MODL-20).

Sections to come, one per unit: each family's draft source with its facts
and provenance, and the measurements behind each catalogue verdict.

## The draft contract (MODL-18)

`inference/src/runtime/draft.zig` owns the model-independent interface: a
family exposes a `Drafter` (a host pointer and four function pointers)
whose `propose(token, out)` returns greedy candidates chained from the
state after the last committed token (no draft distribution: acceptance
draws from the target alone, below); `commit(tokens, h_rows)`
advances the drafter over tokens the main model committed using their
target hidden; `reset()` clears the drafter's state; `bytes()` reports the
workspace it owns beyond the session's. There is deliberately no
`rewind`: the drafter's own attention cache is one more layout in the
*same* `Session`, so ENGN-11's checkpoint/rewind/truncate cover it and the
generation loop only needs `reset` on a session reset.

## The recovery contract (ENGN-11)

A verify batch feeds the main model the last chosen token followed by `k`
drafts in one forward and keeps every row's logits. If `a` drafts are
accepted, the session must end at the state after them — the accepted
prefix. The two state kinds recover differently:

- **Attention caches rewind by position.** A row's content never depends on
  later rows, so `truncate(P + a + 1)` sets the position and nothing is
  copied; rows past it are ignored by contract. This is what Gemma 4 and
  Muse Glimmer (attention-only) use.
- **Recurrent state is replayed.** Qwen 27B and Bonsai 2 hold 48 DeltaNet
  layers whose matrices are a function of every token fed. The batch's
  checkpoint is taken before the forward; on partial acceptance the session
  rewinds to it and re-runs the accepted prefix as a second batched
  forward, which rewrites the same attention rows. Truncating the position
  alone would leave the matrices at `P + k + 1`.

The session owns the mechanism: a page-aligned region inside the byte
block, one recurrent copy, `checkpoint`/`rewind`/`truncate` with their
refusals, all described in [session.md § Checkpoint and
rewind](session.md#checkpoint-and-rewind-engn-11). `engine.Model.recover`
is the accepted-prefix operation, and `generation-check` exercises it per
accepted length on every family, on both executors: replay is bit-identical
to sequential decoding on the CPU, and within the family's
chunk-versus-step bound on Metal (`make test-generation` /
`test-generation-metal`, recorded in the log).

Measured costs on Qwen 27B (2026-09-19): region 156,893,184 bytes;
checkpoint and rewind 3 ms per batch on Metal and 2 ms on the CPU
reference (one 150 MB copy each way); the replay is a short-chunk prefill
(KERN-11's tile). Losing `k − a` drafts
therefore costs one 150 MB copy plus an `a + 1`-token prefill, not a
context-long replay. A bounded alternative — per-token recurrent
checkpoints written by the DeltaNet chunk kernel, `(k + 1) × 150 MB` of
device scratch on Qwen — was not needed by these numbers and is not built;
ENGN-14 measures recovery by accepted length and reopens it
([§ Recovery by accepted length](#recovery-by-accepted-length-engn-14-2026-09-20)).

## The Qwen3.8 draft head (MODL-18)

Facts confirmed on 2026-09-19 from the pinned reference
(`7620399f5`): the dense MTP graph `src/models/qwen35.cpp:485-639`, the
driver `common/speculative.cpp:1324-1760`
(`common_speculative_impl_draft_mtp`), and the example loop
`examples/speculative-simple/speculative-simple.cpp`.

**The block.** The main file's 65th block is one dense full-attention
decoder layer of the main model's shape plus four `nextn` tensors. There
is no separate draft model: the reference opens the *same* file as a
second context of type `LLAMA_CONTEXT_TYPE_MTP`
(`llama_context_params.ctx_type`), which executes only that block with its
own full-attention cache (`llama_set_embeddings_nextn(ctx, true, true)`).
The separate `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` adds only lower-quantized
copies of `token_embd`/`output`/`output_norm` under `nextn.embed_tokens` /
`nextn.shared_head_head` / `nextn.shared_head_norm`; the main file's block
carries `shared_head_norm` and falls back to the main `token_embd` and
`output` (`qwen35.cpp:518,621-635`). The embedded block is therefore the
source.

**The pair.** At MTP position `p` the block consumes the token `x_p` and
the target hidden `h_{p-1}`: `concat = [enorm(embed(x_p)); hnorm(h_{p-1})]`
along the feature axis (`e_norm` first, `h_norm` second), projected by
`nextn.eh_proj` `[2·5120 -> 5120]` (`qwen35.cpp:539-549`,
`ggml_concat(..., dim=0)`). `h` is the target's `t_h_nextn`, i.e. the
final hidden **after** `output_norm` and the input to the target's own
head (`qwen35.cpp:204-209`), not the pre-norm residual; the block applies
its own `hnorm` on top. In our runtime this is exactly the `self.normalized`
row produced by `norm(self.x, self.normalized, self.output_norm)` before
`self.mm(self.binding.output, ...)`.

**The layer.** `attn_norm`; `attn_q` is the merged query+gate `[5120 ->
12288]` with each head storing `[query(256); gate(256)]`; `attn_q_norm` /
`attn_k_norm` are applied per head before RoPE; `attn_k`/`attn_v` are
`[5120 -> 1024]` (4 KV heads, Q8_0 in the file, a supported encoding);
24 query heads, key/value width 256, RoPE base 1e7 with sections
`[11, 11, 10, 0]`; the attention output is multiplied elementwise by
`sigmoid(gate)` and projected by `attn_output` `[6144 -> 5120]`; residual
to the `eh_proj` output; `post_attention_norm`; dense FFN `ffn_gate` /
`ffn_up` `[5120 -> 17408]`, `ffn_down` `[17408 -> 5120]`, `silu(gate)·up`;
residual; `shared_head_norm`; the shared `output` head.

**Drafting.** `draft()` seeds the block at `n_past` with token `id_last`
and the carried `pending_h` (the target `h` of the last committed token),
decodes one row, and takes the greedy top-1 (sampler `top_k = 10`,
candidate 0) as the first draft; each later step feeds the drafted token
and the block's own output hidden `h_nextn` as the next `h`, at position
`n_past + i + 1`. Drafting stops at `n_max` or when the top-1 probability
falls below `p_min` (`speculative.cpp:1596-1745`).

**Advancing the cache.** After the target decodes `[id_last, drafts...]`,
`process()` runs the block over the whole batch (token + shifted target
`h`: row `k` gets `h` of the previous row; the first row gets `pending_h`),
writing its cache rows at the batch positions. It stores every target `h`
row for acceptance and carries the last one as `pending_h`
(`speculative.cpp:1478-1594`). On partial acceptance the target and the
draft cache are both rewound and `accept(n)` sets `pending_h` to the
target `h` at index `n` (`speculative.cpp:1747-1760`). The block's own
output hidden is used to chain drafts only; the cache is filled from the
target's `h`, never the block's.

**Which encoding.** The block's `attn_k`/`attn_v` are Q8_0; `qwen35`'s
`executableEncoding` already admits id 8 (generic matvec and generic F32
tile), and the binder validates every block tensor's shape and encoding
before it is bound.

**Measured.** The CPU block (`qwen35_runtime.draftForward`) runs against a
trace captured from the pinned reference through the extended harness
(`scripts/reference-generation.cpp --mtp-draft`): at positions 0 and 1 of
`Hello,`, its `h_nextn` matches to 7.2e-6 and 1.1e-5 max abs (3.1e-7 and
4.9e-7 relative RMS) and its greedy draft token equals the reference's
(9419, 271). The vectors are pinned under
`inference/src/models/fixtures/qwen35-mtp/` and checked by `checkDraft` in
`make test-generation` / `test-generation-metal`. As a sanity figure for
the acceptance the theme cares about, the reference's own
`llama-speculative-simple --spec-type draft-mtp --spec-draft-n-max 4` on
`Hello,` accepted 8 of 33 drafts (24.2 %) over 16 generated tokens.

**Metal (2026-09-20).** `qwen35_metal.zig` runs the same block with the
existing kernels: the pair is two copies into a 10,240-wide buffer and an
`eh_proj` matvec; the layer is the main full-attention path over the
block's own cache layout (the plan sizes 65 layouts when a drafter is
asked for); the head is the shared `output` matvec on the normalized
hidden, argmaxed on the device. `Plan.init`'s `draft` flag is honored and
`Engine.open(.embedded)` builds the block on Metal as on the CPU. Against
the same pinned trace the Metal rows match at 1.5e-5 max abs / 5.8e-7
relative RMS at position 0 and 1.5e-5 / 7.2e-7 at position 1 (F32 cache);
with the F16 cache the keys and values round within the family's recorded
`half_*` bound (8.5e-3 / 3.1e-4 and 2.5e-3 / 2.4e-4). `compare-draft-metal` / `-cpu` write the native rows and
run `compare-generation.py --draft` against the pinned directory; both pass
on all three rows (block `h` at positions 0 and 1, and the target `hprev`).
`draftRecoveryCheck` (generation-check, both executors) resets the block
and takes a checkpoint/rewind across a block row: the hidden is
reproduced byte for byte and two independently reset drafters propose the
same greedy chain.

## The verify batch and the loop (ENGN-12, session 1)

Implemented and checked on 2026-09-20; the configuration and the benchmark
are session 2.

**Verify.** `Model.verify(tokens, vocabulary, rows, hidden, observer)` fills
`rows` (`tokens.len × vocabulary`) with the target logits of every row and,
when `hidden` is given, the post-`output_norm` hidden per row that
`Drafter.commit` consumes. The CPU reference loops `Runtime.step`; the Metal
plan records the layer stack once (`recordLayers`, shared with
`prefillChunk`) and runs the output head over all rows through the batched
matmul tile into a device buffer sized by the tile
(`matmulPadded(max_verify_rows) × vocabulary`, about 64 MB), reading back
only `tokens.len` rows. `verifyGreedy` returns each row's argmax from the
device (`nu_argmax_partial`/`final` per row) without a logit readback. A
batch is at most the plan's chunk and `max_verify_rows = 16`; the loop uses
at most `max_draft_length + 1 = 8`.

**The loop.** `runLoop` takes a `Speculative{ enabled, draft_length }`
setting; `max_draft_length = 7` is the host bound (KERN-11's 8-row tile less
the seed). When a drafter is loaded and the switch is on, it first commits
the prompt to the drafter in verify-sized chunks (the block's cache is
filled from the target hidden of every committed position), then each step:
proposes `k = min(draft_length, capacity − position − 1, budget_left)`
drafts; checkpoints; verifies `[seed] ++ drafts`; accepts the longest prefix
(greedy: `verifyGreedy` argmax equals the draft; sampled: the row's own
draw equals the draft); recovers the accepted prefix; commits it to the
drafter; and emits the accepted drafts and the correction through the
ordinary per-token checks. The correction is emitted once and carried as the
next step's seed. EOS, the budget, or the context limit inside a batch ends
the turn there and the session recovers to the emitted prefix; a cancelled
`verify` poisons the session and `resetAll` handles it. The per-token GPU
greedy/top-k shortcuts are off while speculating (every seed row is
materialized), and a per-layer observer disables speculation.

**Sampled acceptance.** `inference/src/sampling/speculative.zig`:
`Sampler.distribution` is the shaped, normalized nucleus (penalties, sort,
top-k, temperature and `min_p`, top-p) factored from `select`; `decide`
draws the target's own token from row `i`'s distribution and accepts the
draft when they agree, else the draw is the correction; after the last
accepted row the draw is the bonus. Every emitted token is a draw from the
target whatever proposed the drafts, which is what makes the rule exact for
the greedy chains the drafters produce (the `min(1, p/q)` rejection rule with
a residual correction is exact only for drafts sampled from `q`, so an
earlier version of this module that used it against argmax drafts leaned
toward the drafter's choice and was replaced on 2026-09-20). The history
advances through the accepted drafts, so row `i`'s penalties see the drafts
before it. Unit tests draw 20,000 seeded decisions against a fixed `p` for
a draft the target never emits and for its most likely token and hold the
emitted counts and the acceptance rate to `p` within 3 σ. The decision costs
one full-vocabulary sort per row on the host (`Timing.accept`, the bench's
`accept_milliseconds`), which the per-token GPU top-k path avoids in
ordinary decoding; a per-row top-k readback in `verify` is the follow-up.

**Evidence.** `generation-check --speculative-check`
(`make speculative-check`, `make speculative-check-metal`) runs the model
primitives and, on Metal, the engine loop: 12 greedy tokens equal ordinary
greedy decoding token for token on both executors (7 of 24 drafts accepted on
the pinned `Hello,` seed on Metal). `make check`, `make compare`, and
`make test-generation-metal` pass; `make test-generation` is unchanged (the
speculative check is its own target because the CPU reference is slow).

**Acceptance statistic (2026-09-20).** `generation-check --draft-stats`
(the `make draft-stats` target) decodes each fixed coding prompt greedily,
records every step's target hidden, commits the drafter over the prefix,
proposes four chained candidates at every step and compares draft `i` to
the token the target chose `i + 1` steps after the seed. On Metal, four
drafts per position cost 6.2 ms (about 1.6 ms per block forward) and the
block's workspace is 1,116,160 bytes. The per-depth acceptance was:

| Prompt | depth 0 | depth 1 | depth 2 | depth 3 |
| --- | ---: | ---: | ---: | ---: |
| `Write a Zig function that reverses a string.` | 28/31 = 90.3 % | 24/30 = 80.0 % | 20/29 = 69.0 % | 18/28 = 64.3 % |
| `def fibonacci(n):` | 29/31 = 93.5 % | 25/30 = 83.3 % | 24/29 = 82.8 % | 24/28 = 85.7 % |

The reference's own driver on the first prompt accepted 24 of 40 drafts
(60 %) with its top-k sampler, so the rates are credible rather than an
artifact of the alignment. The block's weights add no decode cost when
speculation is off: `runLoop` asks the drafter only while
`Speculative.enabled` is set.
The block's own `attn_k`/`attn_v` are Q8_0, already in
`qwen35.executableEncoding`.

## The switch, the bench pair, and the record (ENGN-12, session 2)

**Configuration.** `generation.speculative` (default off) and
`generation.draft_length` (default 4, 1 ≤ n ≤ `engine.max_draft_length` =
7, `InvalidNumber` above it) in `nuclis.json`, per-entry overrides under
`models.<name>.generation`, and `--speculative on|off` / `--draft-length N`
on `generate`, `agent`, and `bench`, resolved defaults → entry → flag with
`config show` provenance (`src/config.zig`, `src/cli.zig`). `generate` and
`agent` open the model with `DraftRequest.embedded` when the switch is on
(`DraftSourceMissing` when the family has no embedded block: Gemma, Muse,
Bonsai) and `.none` otherwise; `bench` opens with `.optional_embedded`
whatever the switch and measures every run as an off/on pair on the one
loaded model, so the pair's ratio is the speedup claim
(`src/bench.zig`: `speculative`, `draft_length`, `speculative_steps`,
`accepted_per_step`, `verify_milliseconds`, `accept_milliseconds`,
`recover_milliseconds` per sample; `speculative_draft_length`,
`mean_speculative_decode_tokens_per_second`, `mean_accepted_per_step`,
`decode_speedup` per report; `schema_version` stays 1).

**The loop's edges** (`generation-check --speculative-check` on Metal):
partial acceptance recovers and continues; the budget inside a batch stops
at exactly the budget with the extra accepted drafts discarded
(`Model.recover` to the emitted prefix); EOS inside a batch stops after
the EOS with the session at the tokens before it, as ordinary decoding
leaves it; cancellation mid-batch (the interrupt's `error.Cancelled`)
resets the session and reports `.cancelled`; a context with no room for a
batch stops at `context_limit`. `Model.recover` skips the rewind and the
replay when every draft was accepted (the batch fed exactly the accepted
prefix).

**The sampled rule** was corrected in this session (above, *Sampled
acceptance*): the target-draw rule replaces `min(1, p/q)`, and the draft
contract lost its logits rows. `Timing.accept` isolates the host decision.

**The record** is in [bench.md § Speculative decoding record](bench.md#speculative-decoding-record-engn-12-2026-09-20)
(`make speculative-record`): off by default for the Qwen entry; the
per-batch costs — verify 225–250 ms at 512 and ≈ 342 ms at 4K, recovery
99–272 ms on rejection, the sampled decision 46–78 ms, proposal 6.2 ms per
draft, commit 6.2 ms per token, the prefill at 2.9–3.3× — are the cost
table the plan's performance units (ENGN-13 to ENGN-17, KERN-12, KERN-13)
target.

## The prompt commit and the batched drafter commit (ENGN-13)

**Prompt commit through `prefill` (2026-09-20).** `Plan.prefill` gained a
`hidden_rows` readback — every row's post-`output_norm` hidden, checked
bit-identical to `verify`'s rows on the pinned `Hello,` trace — and
`runLoop`'s speculative prompt branch now consumes the prompt through
`engine.commitPrompt` in `prefill_chunk` (256) chunks instead of `verify` in
8-row chunks.

**Batched drafter commit (2026-09-20).** `Plan.commit` over two or more
tokens is one command buffer: it embeds each token, pairs it with the
previous row's target hidden in a `padded × 10240` `[enorm; hnorm]` buffer,
projects `eh_proj`, runs the block's attention chunk at the batch's cache
positions, and adds the FFN — the head norm is skipped because a commit only
fills the block's cache rows. One token keeps the standalone `draftForward`;
the CPU `Runtime.commit` stays per token as the reference. The scratch
(`draft_hprev_c`, `draft_concat_c`) is sized by the padded chunk, since
`eh_proj`'s matmul reads padded rows.

**Measured** at revision `9a5d3cf`, draft 4, F16 KV, ctx 32768, prose 512 and
4,096, one warmup and repeated measured pairs on one loaded model:

| workload | ordinary prefill | speculative prefill | ratio | propose/batch | verify/batch | recover/batch | commit/batch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 (three runs) | 5,807.8 ms | 5,939.4 ms | **1.02×** | 25.7 ms | 236.5 ms | 185.5 ms | **5.4 ms** |
| 4,096 (two runs) | 51,831 ms | 54,399 ms | **1.05×** | 26.5 ms | 345.0 ms | 243.2 ms | 10.8 ms |

Both meet the unit's targets (≤ 1.10× at 512, ≤ 1.15× at 4K, and
`commit_milliseconds / speculative_steps` ≤ 8 ms at 512). Accepted drafts per
batch was 1.667 at 512 and 1.977 at 4K. The serial prompt commit measured
1.59× at `db9cf80` and the ENGN-12 record's old `verify`-per-8-rows path
2.85×. `make draft-stats` reproduces MODL-18 (28/31, 24/30, 20/29, 18/28 and
29/31, 25/30, 24/29, 24/28), `make speculative-check-metal` is unchanged
(7/24 and 6/21 accepted), `make compare-draft-metal` unchanged (3 rows,
1.5e-5 / 7.9e-7), and `make check`, `make compare` (f32 6.1e-5 / 7.7e-7,
f16 2.5e-2 / 1.9e-4), `make test-generation-metal` pass. `engine.Timing`
now separates `propose` and `commit`, and `bench.Sample` carries
`propose_milliseconds` and `commit_milliseconds`.

**The agent (2026-09-20).** `Completer.prime` commits the primed prefix
through `commitPrompt` when a drafter is loaded, so the primed snapshot
carries the block's cache rows (the agent's first turn no longer attends over
prefix rows the block never wrote). The loop surfaces the turn's mean
accepted drafts per batch in the status bar (`spec N.NN/step`), from the
engine timing. On a bounded pty at revision `ccb191b`, the first turn of
`nuclis agent --speculative on` on `Write a one-line Python function that
doubles a number.` (ctx 4096, 32 tokens) settled at **spec 2.56/step**,
within the range `bench` measures for the code prompt (1.2–3.0 accepted at
draft 4).

## Small batches: the multi-row matvec and the 2-row routing (KERN-12)

The verify batch (`1 + k` rows), the recovery replay (`a + 1`), and the
decode-time commit (`a + 1`) all reach `Backend.matmul`, which served them with
the 16×8 split tile. KERN-12 added `nu_matvec_rows_*_t2..t8`: one SIMD group
per four output rows, a decoded weight slice multiplied against every activation
row before the next slice is fetched, one accumulator per (row, token). The
sweep (`make bench-matvec-rows`, both FFN shapes, in
[metal-backend.md § Multi-row matvec](metal-backend.md#multi-row-matvec-kern-12-2026-09-20-closed-below-its-target))
shows it wins at **2 rows only** for every specialized encoding (143–182 GB/s
against the tile's 88–116) and for Q6_K/IQ4_XS at 3; at 5 rows it is 49–67 and
at 8 rows 20–33, below the tile. `Backend.matmul` therefore routes only 2-row
batches of the specialized encodings (`small_batch_rows = 2`); the tile keeps
3–24, so the 5-row verify is unchanged and the plan's verify target
(≥ 150 GB/s at 5, ≤ 130 ms) is **not met**. Growing arithmetic and
activation-load work plausibly explain the tested scalar bodies' decline. The constant-input probe also permits
arithmetic elimination, so it cannot isolate load cost or prove all scalar
layouts incapable of meeting the target.

A spot run at the close revision (512 prose, draft 4, F16 KV, ctx 32768,
warmup 1, repeat 1, one loaded model) measured `verify_milliseconds /
speculative_steps` 232–288 ms and `recover_milliseconds / speculative_steps`
170–217 ms. These are aggregate costs per speculative step, mixing accepted
lengths and steps without replay; they do not establish two-row recovery latency
or the effect of its routing. ENGN-14 must measure recovery by accepted length
before and after replacing replay with recurrent-state copies. `make compare`
(f32 6.1e-5 / 7.7e-7, f16 2.5e-2 / 1.9e-4), `make test-generation-metal`,
`make speculative-check-metal` (12 tokens greedy, the loop edge cases), and
`make draft-stats` (28/31, 24/30, 20/29, 18/28 and 29/31, 25/30, 24/29, 24/28)
are unchanged.

## Recovery by accepted length (ENGN-14, 2026-09-20)

The aggregate `recover_milliseconds / speculative_steps` mixed accepted
lengths together, which hid that the replay's cost is governed by its row
count. `engine.Timing` now carries the split and the histogram — `recover`
(aggregate, including the final recover to the emitted count when a batch
crosses the budget), `recover_rewind` (the checkpoint copy), `recover_replay`,
`recover_by_length[accepted]` (calls, rewind, replay per accepted prefix of
1..`max_draft_length + 1`), and `checkpoint` (the verify batch's copy, which
no other field covered). `bench`'s samples expose the same as
`checkpoint_milliseconds`, `recover_rewind_milliseconds`,
`recover_replay_milliseconds`, and `recover_by_length` (cells with `accepted`,
`calls`, `rewind_milliseconds`, `replay_milliseconds`). Full acceptance
recovers nothing: its cells count calls with zero time.

Session 1 measured the replay by accepted length and moved the seed-only
case to the per-token `step` path; session 2 replaced the replay with
per-row recurrent checkpoints. Measured on Qwen3.8-27B UD-Q4_K_M, Metal,
F16 KV, ctx 32768, the 512-token prose prompt, greedy, 128 output tokens,
draft 4, warmup 1 and three measured runs on one loaded model (48 verify
batches per run, 144 total; per-run files under `.zig-cache/bench/engn14/`).
`calls` and milliseconds **per call**, summed over the three runs:

| accepted prefix | calls | replay via prefill ms (session 1) | replay via step ms (session 1) | restore by slot ms (session 2) |
| ---: | ---: | ---: | ---: | ---: |
| 1 (seed only) | 36 | 217.8 | 96.4 | 15.6 |
| 2 | 42 | 179.8 | 178.4 | 14.9 |
| 3 | 27 | 220.4 | 219.8 | 16.5 |
| 4 | 18 | 182.8 | 182.4 | 13.0 |
| 5 (all four drafts) | 24 | 0 | 0 | 0 |

The session-1 `recover` mean was 182 ms per batch on the prefill path and
150 ms with the seed-only branch; session 2's is **12.7 ms per batch**
(610.8 ms over 48 batches), all of it the slot copy. The write side costs
`verify` 12,330 ms per run against 11,307 without slots — **+21 ms per
batch** for up to five 150 MB slot writes and their recompute — and
`checkpoint` still copies the batch's checkpoint (150 ms per run, ~3 ms per
batch). `session_bytes` is 3,850,633,216 with the 1,255,146,752-byte row
region. The off/on pair at draft 4 greedy reads 0.85× (8.56 vs 10.04 tok/s)
against session 1's 0.62×; prose still loses, now to the verify batch
rather than recovery: sampled acceptance (ENGN-15), the proposal policy
(ENGN-16), the small-batch tile (KERN-14), and long-context attention
(KERN-16) are the remaining levers.

Two facts for the next revisiter. The kernel computes each row's state as
`S_r = γ_r S₀ + Σ_{s≤r} r(r,s) U[s]ᵀK[s]` with every exponent ≤ 0; the
first implementation rescaled the full-chunk `W` by `r(r, n−1)` and
overflowed on a layer whose chunk decay reached `cum = −114`, writing `inf`
into slots and NaN into the state on restore. And the slot writes are only
gated by `row_states > 0`: ordinary prefill, decode, and the prompt commit
pass 0, and the kernels take the same code path as before. The layout and
the refusals are in [session.md § Row checkpoints](session.md#row-checkpoints-engn-14).

## The device penalty kernel (KERN-13, 2026-09-20)

A sampler with a presence or repetition penalty changes the sort itself, so
before this unit the loop turned both GPU selection paths off while one was
active: every such token read the full 248,320-logit row back and sorted it
on the host, and the ordinary decode step measured 7.7–8.6 tok/s on the
instruct profile against 8.8–10.3 greedy (the ENGN-14 record). `nu_penalize`
([metal-backend.md § History penalties](metal-backend.md#history-penalties-kern-13-2026-09-20))
applies the penalties on the device in place — before the argmax and before
the three top-k passes — so the readback is the vector the CPU sampler
would sort: repetition first (`l / r` for positive, `l · r` for negative
logits), then `l − presence`, for every token in the history bit set. The
host uploads the set as `vocabulary / 32` little-endian u32 words when its
revision changed (`History.revision`, bumped by `observe` and `reset`), so
the 31 KB copy happens on a change, not per step.

The sampler side: `TopK` carries a `penalized` flag; `Sampler.selectFrom`
decides a readback with penalties active only when the device marked it
`penalized` (an unpenalized readback still defers to the full path), and
`gpuEligible()` no longer excludes penalties. `Sampler`'s profile defaults
to the same operation as before when a full readback is requested: raw
`logits` are never penalized in place by the plan, and the loop's `readLogits`
fallback skips the sampler's own penalties when the vector already carries
them. The Qwen, Gemma 4, and Muse Glimmer plans all upload and apply the
penalty (each on the final logits, after Gemma's and Muse's soft-cap).

Measured 2026-09-20, `make speculative-record ARGS="--only prose512 code"`,
Qwen3.8-27B UD-Q4_K_M, Metal, F16 KV, ctx 32768, 128 output tokens, one
warmup and three measured runs per configuration, off/on pairs on one
loaded model (`d31c5cd`; the full table is in
[bench.md § The KERN-13 quick pass](bench.md#the-kern-13-quick-pass-2026-09-20)):

| configuration | decode off tok/s (before KERN-13) | decode off tok/s (now) |
| --- | ---: | ---: |
| code, greedy, d2/d4/d7 | 8.81 / 9.15 / 9.32 | 8.66 / 8.67 / 8.74 |
| code, instruct, d4 | 8.56 | 8.74 |
| prose 512, greedy, d2/d4/d7 | 10.31 / 8.26 / 9.35 | 10.22 / 9.50 / 9.21 |
| prose 512, instruct, d2/d4/d7 | 8.18 / 7.99 / 7.78 | 9.04 / 8.87 / 8.80 |

The instruct baseline now sits inside the greedy band within a run and
across the record (code instruct 8.74 against code greedy 8.66–8.74;
prose instruct 8.80–9.04 against greedy 9.21–10.22, whose spread across the
record's three consecutive runs is the session's clock drift, not the
kernel). `topk_fallbacks` is 0 at every configuration: the penalized
readback decided every token. The speculative speedups move little
(code greedy 1.20× at draft 4 against 1.22×, prose 512 greedy 0.88 against
0.93), because the acceptance decision still sorts the full rows
(ENGN-15); what this unit fixes is ordinary decode with penalties, which is
no longer 12–25 % below greedy.

## Sampled acceptance on the device readback (ENGN-15, 2026-09-20)

The sampled acceptance used to call `Sampler.distribution` on every verify
row: a full-vocabulary sort per row on the host, measured at 62.8–91.3 ms
per batch in the KERN-13 quick pass (17–22 % of an instruct batch). Now the
verify batch has two output modes (`engine.VerifyOutput`): full `rows`, as
before, and `topk` — one `sampling.TopK` per row from the device partial
top-k over that row, with every row's logits left resident in
`verify_logits` so `Plan.readVerifyRow(row, out)` serves a fallback. The
plan owns one `TopKBuffers` set per verify row (16 × ~130 KB, allocated
with the drafter) because the rows' dispatches share one command buffer.

The sampled path in `speculativeBatch` picks a row's token with
`Sampler.selectFromHistory(&tops[i], scratch, history)`: the penalties,
when active, are applied on the host to the 256 candidates with the *live*
history — which the accepted drafts of the batch advance, as
`distribution`'s call sequence did — and the penalized top-k is exact when
its last entry clears the readback's smallest raw value. That bound holds
because the penalties only lower values; `repetition_penalty < 1`,
`presence_penalty < 0`, a penalized greedy draw (`temperature == 0`), a
nucleus (`top_k == 0`), or a retained set that fails the bound returns
`null` without advancing the RNG, and the loop reads the row back and takes
`distribution` (counted in `topk_fallbacks`). Without penalties
`selectFromHistory` delegates to `selectFrom`, so eligible nucleus sampling
decides from the readback as it already did on the step path. The greedy
sampled path still uses `verifyGreedy`; only the temperature-0-with-penalty
configuration falls back to full rows.

Measured 2026-09-20, `make speculative-record ARGS="--only prose512 code"`,
same workload and methodology as the KERN-13 pass above
([bench.md § The ENGN-15 quick pass](bench.md#the-engn-15-quick-pass-2026-09-20)):

| configuration | accept before (KERN-13 pass) | accept now | fallbacks | speedup before → now |
| --- | ---: | ---: | ---: | ---: |
| code, instruct, d4 | 91.3 ms | 36.9 µs | 0 | 1.12× → 1.36× |
| prose 512, instruct, d2 | 62.9 ms | 18.8 µs | 0 | 0.76× → 0.92× |
| prose 512, instruct, d4 | 71.7 ms | 21.6 µs | 0 | 0.82× → 0.98× |
| prose 512, instruct, d7 | 77.9 ms | 23.8 µs | 0 | 0.85× → 1.01× |

The ≤ 5 ms target is met by two orders of magnitude, and the readback
decided every row: the fallback path exists for the guard's edge cases, not
for this profile. `Timing.topk_fallbacks` now counts the verify fallbacks
too, not just the step-path ones. Exactness is checked twice:
`generation-check --metal` runs `verifyTopKCheck`, which verifies the same
four-token batch with both modes on real logits and requires every row's
readback decision (with and without penalties) to equal `select` on the
rows-mode row, reading the resident row back when the readback defers
(8/8 rows decided, 0 fallbacks, all equal); and `sampling/root.zig` tests
`selectFromHistory` against `select` for hundreds of random vectors and
option sets, including the deferrals.
