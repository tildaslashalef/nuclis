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

KERN-14 closed on 2026-09-20 **negative**: the wide 32×8 variant of the
small-batch tile loses 8–26 % on the Qwen FFN encodings at 5 rows (Q4_K
96.8 → 88.7, Q6_K 114.0 → 84.6, Q3_K 63.0 → 47.1 GB/s), ties on Q5_K, and
wins only 5–8 % on the wide head — the best tile at 5 rows is 110 GB/s
against the ≤ 150 bar. Nothing routes to it; the 16×8 tile and the two-row
matvec routing stand, so full-model verify latency is unchanged (262–297 ms
per batch). The wide kernels stay as the measured fixture. Verdict:
[metal-backend.md § The wide 32×8 tile](docs/reference/metal-backend.md#the-wide-328-tile-kern-14-2026-09-20-closed-negative),
sweep in
[bench.md § Small-batch tile sweep](docs/reference/bench.md#small-batch-tile-sweep-kern-14-2026-09-20),
[log](docs/engineering-log.md#kern-14--the-wide-328-small-batch-tile-measured-closed-negative-2026-09-20).
Before it in the same session, KERN-13 (the device penalty kernel) and
ENGN-15 (sampled acceptance on the per-row top-k) closed positive: `accept`
fell from 62.8–91.3 ms per batch to **18.8–36.9 µs**, code instruct (draft
4) rose to **1.36×**, prose 512 instruct to **0.92 / 0.98 / 1.01×** at
drafts 2 / 4 / 7. The deferred CPU speculative check was run and passed (12
tokens identical to ordinary greedy).

**Next: ENGN-16** (the proposal policy: `p_min` early stop and an adaptive
length), then KERN-15 and KERN-16, then ENGN-17's full record.

Speculative decoding works end to end on Qwen3.8-27B and is not yet a
speedup worth switching on by default. ENGN-11 (recovery), MODL-18 (the
embedded draft head), ENGN-12 (batched verification, the speculative
loop, greedy and sampled acceptance, the switch and the draft length, the
benchmark record), and ENGN-13 (the prompt commit at the plan's chunk and
the batched drafter commit) closed on 2026-09-19/20. The switch stays
**off** by default with `draft_length 4`; the current record's verdict is in
the cost table below. KERN-12 (the multi-row matvec) closed on 2026-09-20
**below its target**: the register-tiled body wins at 2 rows (143–182 GB/s
against the 16×8 tile's 88–116) but at 5 rows streams 49–67 and at 8 rows
20–33, so `matmul` routes only 2-row batches of the specialized encodings
(`small_batch_rows = 2`) and the tile keeps the verify
([metal-backend.md § Multi-row matvec](docs/reference/metal-backend.md#multi-row-matvec-kern-12-2026-09-20-closed-below-its-target)).
REPO-08 repaired that sweep's controls; all twelve two-row cases beat the
tile.

This plan is the path to the speed benefit, as measured costs per verify
batch on Metal (Qwen 27B, F16 KV, 512-token context unless noted; ordinary
decode step ≈ 95–105 ms; the ENGN-14 record's numbers, see *Working a unit
here* for how to refresh them):

| cost per batch | measured (record, 2026-09-20) | cause | unit | target |
| --- | ---: | --- | --- | ---: |
| propose `k` drafts | 12.7–43.5 ms (≈ 6.3 ms per draft) | one block forward per draft | ENGN-16 | fewer forwards, same accepted tokens |
| checkpoint | 2.8–5.2 ms | one 150 MB copy | — | — |
| verify `1 + k` rows | 262–297 ms at 512, 362–372 ms at 4K | the 16×8 prefill tile at small row counts; the chunk attention over the visible cache | KERN-16 at long context (KERN-14 closed negative) | ≤ 130 ms |
| accept (sampled) | 18.8–36.9 µs (quick pass) | one draw per row on the device readback | ENGN-15 ✓ | ≤ 5 ms |
| recover (on rejection) | 6–22 ms (was 150–182) | one 150 MB slot copy | ENGN-14 ✓ | ≤ 40 ms |
| commit `a + 1` tokens | 4.7–15.3 ms | one batched forward per committed prefix | ENGN-13 ✓ | ≤ 8 ms |
| prompt commit (prefill) | 1.02× ordinary prefill | the plan's own chunk, not 8-row verify chunks | ENGN-13 ✓ | ≤ 1.10 × |
| tokens per batch | 2.23–3.97 (1.23–2.97 accepted) | acceptance 42 % per draft on prose, 58–68 % on code | ENGN-16 | more accepted per proposed |

Measured speedups (ENGN-15 quick pass, `d31c5cd` plus the unit's change):
code greedy 0.96 / 1.20 / 1.33× at drafts 2 / 4 / 7, code instruct 1.36× at
draft 4; prose 512 greedy 0.81 / 0.93 / 1.07×, instruct 0.92 / 0.98 / 1.01×.
Both sampled paths are now free of host work; what remains is the batch's
model time. At draft 4 the code prompt advances 3.34 tokens for a 272 ms
verify (81 ms/token against ~115), prose 2.65 for 291 (110 against ~120).
KERN-14 closed negative, so the 512-token verify stays where it is;
KERN-16's long-context attention is the remaining kernel lever, and
ENGN-16 trims the proposal and the wasted rows on prose. Nothing here
claims a final speedup before ENGN-17 measures it.

Order: ENGN-16 → KERN-15 → KERN-16 →
ENGN-17 → ENGN-18 → ENGN-19 → KERN-17 → TERM-10 → MODL-19 → MODL-20 →
MODL-21 → AGNT-11 → MODL-22 → MODL-23. KERN-13 and ENGN-15 landed first
because the instruct profile's presence penalty defeated the GPU top-k
readback path otherwise. KERN-14's small-batch tile closed negative, so
verify stays on the 16×8 tile at 512; ENGN-16 is cheap and trims the prose
waste, and KERN-15/KERN-16 are the remaining kernel levers. **KERN-15 and KERN-16 sit before ENGN-17
because they change the Qwen path the verdict measures**: the split-K
matvec touches Qwen's row-poor shapes, and the long-context attention is
every verify batch at 16K–32K (and the 32K acceptance's largest deficit).
ENGN-18 (Gemma's launch-bound decode), ENGN-19 (the ring layout: memory,
not speed), and KERN-17 (the ternary experiment) are other-family or
experimental and sit after the verdict, grouped so the performance theme
finishes in one stretch; TERM-10 (chat polish) and the vision units follow.
The performance group closes in the order of measured leverage: the
row-poor matvecs (two thirds of Muse's gap, and every family's small
projections), the long-context prefill attention, Gemma's decode, the ring
layout, and last the ternary arithmetic, which is the least certain and may
close negative; the small-batch tile led the order and closed negative. AGNT-12 (background commands) and
APPS-14 (teacher-forced `eval`) are drafted for decision, not ordered.

| Unit | Title | Sessions |
| --- | --- | --- |
| ENGN-16 | Draft proposal policy: `p_min` early stop and an adaptive length | 1 |
| KERN-15 | Split-K decode matvec for row-poor shapes (Muse's gap, every family's small projections) | 1–2 |
| KERN-16 | Long-context prefill attention, second attempt (register-level reuse) | 2 |
| ENGN-17 | The verdict, the defaults, and the bench baseline without the drafter | 1 |
| ENGN-18 | Gemma 4 12B decode: fused norms and fewer launches | 1–2 |
| ENGN-19 | A ring layout for windowed attention caches | 2 |
| KERN-17 | The ternary matvec's arithmetic (experiment; may close negative) | 1 |
| TERM-10 | Chat polish: operation dots, the running pulse, write summaries with a file view | 1 |
| MODL-19 | Gemma 4 draft heads: the companion file as a second GGUF, 12B and 26B-A4B | 1–2 |
| MODL-20 | Muse Glimmer DFlash drafter: facts, contract fit, acceptance loop | 2 |
| MODL-21 | The vision contract, image input, and the Qwen3.8 projector | 2–3 |
| AGNT-11 | Images in the chat: drop, paste, `/image`, the `[image #N]` chip | 1 |
| MODL-22 | Gemma 4 vision: the unified embedder (12B) and the SigLIP projector (26B-A4B) | 2 |
| MODL-23 | Muse Glimmer's windowed vision encoder | 2 |
| AGNT-12 | Background commands (drafted for decision; see its section) | — |
| APPS-14 | Teacher-forced `eval` (drafted for decision; see its section) | — |

## Working a unit here

**Build and gates.** `make build` writes `./zig-out/bin/nuclis` (Metal on).
Every unit keeps these green and says so in its log entry:

- `make check` — fmt, unit tests, Metal fixtures.
- `make compare` — the pinned llama.cpp traces of the main model (f32 max
  abs 6.1e-5 / rel RMS 7.7e-7; f16 2.5e-2 / 1.9e-4).
- `make test-generation-metal` — chunked prefill vs steps, the recovery
  check at every accepted length (bounds 2e-2 max abs / 1e-3 rel RMS).
- `make speculative-check-metal` — 12 greedy tokens identical to ordinary
  greedy decoding through the primitives and through `engine.runLoop` on the
  pinned `Hello,` seed, plus the loop edge cases (budget, EOS, cancellation,
  context limit). `make speculative-check` is the same on the CPU reference
  and takes about 20 minutes: run it once per unit that touches
  `engine.zig` or `qwen35_runtime.zig`.
- `make draft-stats` — the per-depth acceptance table of the embedded head
  on the two fixed coding prompts (MODL-18's 90/80/69/64 % and 94/83/83/86 %
  at depths 0–3); a unit that touches the block's forward or its commit must
  reproduce it within one draft per cell.
- `make compare-draft-metal` — the block's own pinned trace.

**The record.** `make speculative-record` runs
`scripts/nuclis-speculative.py`: the reference corpus arrays at 512 and 4,096
tokens and the fixed code prompt, greedy and with the instruct profile's
sampling (`--temperature 0.7 --top-p 0.8 --top-k 20 --presence-penalty
1.5`), draft lengths 2, 4, 7, 128 output tokens, context 32,768, F16 KV,
one warmup, three measured repetitions (two at 4K), each run as an off/on
pair on one loaded model; JSON per configuration under
`.zig-cache/bench/spec/`. Per-batch costs are the sample's
`verify_milliseconds`, `accept_milliseconds`, `recover_milliseconds` divided
by `speculative_steps`; tokens per batch is `(generated_tokens − 1) /
speculative_steps`; the speedup is `decode_tokens_per_second` on vs off in
the same pair. **The full 12-configuration record runs once per path, at
ENGN-17.** Units before it gate on a quick pass —
`make speculative-record ARGS="--only prose512 code"` — and their measured
numbers go into the reference docs as they are taken. A single
configuration by hand:

```sh
M=$HOME/.nuclis/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf
./zig-out/bin/nuclis bench --backend metal --model "$M" \
  --prompt-tokens tests/fixtures/run-2026-09-06/prompt-512.json \
  --max-tokens 128 --ctx-size 32768 --kv f16 --warmup 1 --repeat 3 \
  --speculative on --draft-length 4 --json
```

Ordinary decode on this workload is 10.62 tok/s at 512 (94.2 ms/step) and
10.20 at 4K, prefill 90.45 and 83.70 tok/s (the 2026-09-10 record). Nothing
else may use the GPU during a record; state the git revision, and never
present an estimate as a measurement.

**Where the facts go.** Each unit's measured numbers go into
[speculative-decoding.md](docs/reference/speculative-decoding.md) (a section
per unit) and its record rows into [bench.md](docs/reference/bench.md) as
they are taken; kernels into
[metal-backend.md](docs/reference/metal-backend.md); the session layout into
[session.md](docs/reference/session.md). The log entry cites them.

## The theme — fixed before the units (decided 2026-09-19, sampled rule revised 2026-09-20)

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
later rows, and rows past the position are ignored by contract, so
accepting `a` drafts sets the position to `P + a + 1` and nothing is
copied. Recurrent state (Qwen3.8's and Bonsai 2's 48 DeltaNet layers:
history and matrix) is a function of every token fed, so it is
*checkpointed* before the batch and, on partial acceptance, restored and
*replayed* over the accepted prefix — a second batched forward of `a + 1`
tokens, which also rewrites the same attention rows. Never rewind DeltaNet
by truncating the position alone. Gemma 4 and Muse Glimmer are
attention-only, so their recovery is the position rewind alone. ENGN-14
replaces the replay with per-row checkpoints if they measure cheaper.

**The draft contract** is model-independent and lives in
`inference/src/runtime/draft.zig`: `propose(token, out)` chains greedy
candidates from the state after the last committed token, `commit(tokens,
h_rows)` advances the drafter over the accepted prefix with the target
hidden of each token, `reset`, and `bytes` for the load plan; the drafter's
cache is one more layout in the session, so checkpoint/rewind cover it.
Each adapter implements it with its family's source: Qwen3.8's embedded
block (the separate `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` stays pinned and
unloaded), Gemma 4's companion heads, Muse's DFlash drafter. Bonsai 2's
file drops the block and its entry pins no draft companion.

**Verification** returns the main model's logits for every row of the batch
on both executors, or (sampled, eligible options) each row's device partial
top-k with the logits resident for a fallback. Greedy acceptance compares
the draft with the row's argmax. Sampled acceptance draws the target's own
token from the row's shaped distribution (temperature, top-k/p, min-p,
penalties with the history advanced through the earlier drafts of the
batch) and accepts `d_i` when the draw equals it, else the draw is the
correction; on full acceptance the last row's draw is the bonus. Every
emitted token is a
target draw whatever proposed the drafts, which is what makes the rule
exact for greedy chains; the `min(1, p/q)` rejection rule with a residual
correction is exact only for drafts sampled from `q` and was replaced on
2026-09-20 for that reason. Identical seeded streams versus ordinary
decoding are not a requirement.

**Configuration** is in [spec § Speculative decoding](docs/spec.md#speculative-decoding):
the file per registry entry (`models.<name>.mtp`, a typed load error when
missing or mismatched), `generation.speculative` and `--speculative on|off`
on `generate`, `agent`, and `bench`, `generation.draft_length` and
`--draft-length` capped by `engine.max_draft_length` = 7; the acceptance
rule and the recovery scheme are not exposed. Whether the `mtp` role is
renamed to a mechanism-neutral `draft` is decided in MODL-19.

**The oracle.** The pinned llama.cpp checkout (`7620399f5`, built by `make
compare` under `.zig-cache/reference/llama.cpp`) implements speculative
decoding with both draft kinds in `common/speculative.cpp` (the MTP driver
at lines 1324–1760, `draft()` at 1602–1700; `--spec-type draft-mtp` and
`draft-dflash`, `-md <draft file>`, `--spec-draft-n-max`,
`--spec-draft-p-min`), opens a main file as an MTP context that runs only
the `nextn` layer with its own attention cache, exposes the target's hidden
as `llama_get_embeddings_nextn`, and knows the `gemma4-assistant` and
`dflash` architectures. Read the driver before implementing each source.

**Where the detail goes.** [speculative-decoding.md](docs/reference/speculative-decoding.md)
holds the recovery contract, the draft contract, each family's source with
its facts and provenance, and the measurements; the session, Metal,
generation, and bench references gain their sections;
[llm-guide.md](docs/llm-guide.md) is extended only when the user asks.

## ENGN-16 — Draft proposal policy: `p_min` early stop and an adaptive length

**Facts (read 2026-09-20; updated from the ENGN-14 record).**
- `propose` (`qwen35_metal.zig` / `qwen35_runtime.zig`) always proposes
  `k = draft_length` positions at ≈ 6.3 ms per block forward: 12.7–43.5 ms
  per batch at drafts 2–7 (5–12 % of a batch). Per-depth acceptance on code
  falls 90 → 64 % across depths 0–3 (MODL-18) and is 22 % overall on prose.
- The reference MTP driver (`common/speculative.cpp:1602–1700`) stops
  drafting when the block's top candidate has `p < p_min` (default 0 =
  off, `--spec-draft-p-min`; a top-k 10 sampler on the block's logits) and
  has no adaptive length.
- `Backend.topk` returns Σ exp((l − max) / T) as partial sums (`sums`) with
  the top ids and values, so `p_max = 1 / total` at T = 1 with no logit
  readback; `sampling.TopK.total` is the F64 sum of the partials.
- Verify barely scales with rows, so the policy's value is the skipped
  block forwards and the avoided hopeless rows on prose, not a big verify
  cut.

**Design.**
1. `Drafter.propose(token, out, p_min: f32)`: on Metal, when `p_min > 0`,
   record `b.topk(draft_logits, vocabulary, 1, 1.0, self.topk)` in the
   same command buffer as the block forward and read `total`; stop after a
   draft whose `p_max = 1 / total < p_min`. On the CPU, the softmax
   maximum over `draft_logits` (exact). The first draft is always proposed
   (count ≥ 1 when `out.len ≥ 1`).
2. The adaptive length in `runLoop`: `k_next = draft_length` after full
   acceptance, `k_next = clamp(accepted + 1, 1, draft_length)` after a
   rejection; `settings.draft_length` stays the cap and the only exposed
   knob (spec).
3. `p_min` is a host constant `engine.draft_p_min`, chosen from data:
   extend `generation-check --draft-stats` to bin each proposed draft by
   the block's `p_max` (≥ 0.9, 0.7–0.9, 0.5–0.7, < 0.5) and report the
   acceptance rate per bin and depth; set the threshold at the bin where
   acceptance falls below 50 % (a draft costs one block forward plus one
   verify row; at 50 % it pays). Record the table in
   `speculative-decoding.md`.
4. `bench.Sample` gains `proposed_per_step`, so drafts per accepted token
   is derivable from the record.

**Acceptance.** Prose 512, draft 4: proposed drafts per accepted token
falls by ≥ 30 % and `decode_tokens_per_second` (on) is not lower; code 512,
draft 7: within 2 % of the unpoliced run. `make speculative-check-metal`
still passes (the policy changes which drafts are proposed, never the
tokens emitted); `make check`, `make draft-stats`. Gate with
`make speculative-record ARGS="--only prose512 code"`.

## ENGN-17 — The verdict, the defaults, and the bench baseline without the drafter

**Facts.** `bench` opens the model with `DraftRequest.optional_embedded`
whatever the switch, so there is no in-process measurement without the
drafter loaded, and the MODL-18 acceptance item "decode rate unchanged with
the drafter loaded but switched off" (carried through ENGN-12) is only
comparable across records. The per-entry verdict lives in `src/catalog.zig`
(`Entry`) and is written into `models.<name>.generation` by `config init`
(`src/config.zig`); the built-in defaults are `Config.Generation.speculative
= false`, `draft_length = 4`.

**Design.**
1. `bench`: the default opens with `.none` and runs no pair; `--speculative
   on` opens with `.embedded` and runs the off/on pair. The "off" sample of
   a pair is then the loaded-but-off case, and the default the true
   baseline; report both in the record.
2. Re-run `make speculative-record` on the finished path (after ENGN-13
   through KERN-16), write the record in `bench.md` with the per-batch cost
   table of *Where we are* refreshed, and the spec's measured result.
3. `catalog.Entry` gains `speculative: bool` and `draft_length: usize`;
   `config init` writes them into the entry's `generation`; set Qwen's from
   the record: on if code ≥ 1.5× and prose ≥ 0.9× at the chosen length,
   else off with the reason in the log. Gemma and Muse entries stay off
   until MODL-19/20 measure them.
4. Documentation: `docs/development.md § Configuration file`,
   `docs/reference/bench.md § Definitions` (the speculative fields),
   `docs/spec.md § Speculative decoding` (the measured result and the
   defaults).

**Acceptance.** The record with both baselines; the entry's defaults from
it; `make check`; `config init` / `config show` tests cover the new entry
fields; the spec's measured result cites the record.

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
  mtp-gemma-4-12B-it.gguf`), then Metal on the existing kernel set. The
  prompt commit and the batched commit follow ENGN-13's shape (hidden
  rows from `prefill`, one chunked forward per commit).
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
backends; the benchmark record on the Gemma acceptance workload with the
per-batch cost table and the entries' verdicts; `make check`, the Gemma
compare targets unchanged for the main model.

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
  verify batch is then up to 16 rows, which KERN-12's multi-row path
  serves at ≤ 8 rows and the 16×8 tile beyond (a 16-row variant of the
  multi-row matvec is this unit's call, measured).
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
decision documented; the benchmark record with the per-batch cost table;
the verdict, negative if the drafter does not pay for its verification;
`make check`, the Muse compare targets unchanged.

## TERM-10 — Chat polish: operation dots, the running pulse, write summaries with a file view

**Facts (read 2026-09-20).** The agent emits typed events
(`src/tui/event.zig`: `tool_call { id, name, summary, detail }`,
`tool_result { id, text, truncated, is_error, summary }`, `diff`, `status`,
`notice`, `turn_end`); `src/tui/transcript.zig` turns them into blocks
(`tool_call` carries `running: bool`, cleared when its result arrives) and
renders a settled call as `glyphs.done` + the summary with the detail rows
under `glyphs.detail`; the `Ui` in `src/agent/root.zig` advances one
spinner frame per repaint while a turn runs (`spinnerFrame`, shared by the
thinking label and a running call). Themes are `src/tui/theme.zig` (`Style`
roles, `Glyphs` per `GlyphSet` unicode/ascii; the escape stream is pinned by
golden tests without a TTY). The tools are `bash`, `edit_file`, `glob`,
`grep`, `read_file`, `write_file` (`src/agent/tools/`); `bash` bounds output
at 1 MiB and 300 s. The model of the polish is the screenshot of 2026-09-20:
a coloured dot per operation, a pulsing dot while one runs, `Write(path)`
with `Wrote N lines to path` and a numbered, truncated file view, and
`Read 1 file, ran 1 shell command` summaries between assistant blocks.

**Design.**
1. Operation dots. `Glyphs` gains `dot` (`●` / `*`) and `Style` gains
   `op_ok`, `op_error`, `op_running`, `op_write` (green, red, dim/accent,
   blue in the truecolor and c256 themes; the c16 theme maps them onto its
   palette, `plain` drops colour). The transcript's tool-call row becomes
   `dot name(argument)`: `Read(src/main.zig)`, `Write(TODO.md)`,
   `Bash(make check)`, `Grep("pattern", src/)`, `Edit(path)`, `Glob(*.zig)`;
   the tool's own `summary` moves to the detail row. The dot's style is
   `op_running` while `running`, then `op_ok` or `op_error` from
   `is_error`; a `write_file`/`edit_file` result uses `op_write`.
2. The running pulse. A running call's dot alternates between `op_running`
   and `dim` on the repaint cadence the spinner already uses (the `frame`
   counter; two phases, not the ten-frame spinner), so a long `bash` shows
   a slow blink; the thinking label keeps the spinner. The status bar's
   running-tool text is unchanged.
3. Write summaries. `write_file` returns `summary = "Wrote {lines} lines to
   {path}"` (and `edit_file` `"Edited {path}: +{added} −{removed} lines"`),
   and the transcript renders the result of a write as the summary row plus
   a numbered file view of the first `write_preview_lines = 10` lines
   (`dim` line numbers, code style) and `… +N lines` when longer; a
   `read_file` result keeps today's rows. The view is derived from the
   result text the model already receives, never from a second read.
4. Turn summaries. On `turn_end`, when a turn had tool calls, the transcript
   writes one `dim` row `Read 2 files, ran 1 shell command, wrote 1 file`
   (counts by tool kind, in that order, singular/plural) between the tool
   blocks and the next assistant text; a turn without tools writes nothing.
5. Golden tests in `transcript.zig` for each row form at unicode and ascii,
   a running-then-settled call at both pulse phases, a write with 3 and
   with 40 lines, and the turn summary; the theme test asserts every new
   style has a value in all four kinds.
6. The edit diff. Today `transcript.renderDiff` draws `diff.Row`s
   (`old_line`, `new_line`, `kind` context/add/remove, `text`, a changed
   `Span`) side by side from 96 columns (`side_by_side_min_width`) and
   unified below, with `diff_add` green, `diff_remove` red, `diff_change`
   reverse video, and no line numbers or markers. The polish: a gutter of
   right-aligned old and new line numbers in `dim` (width from the largest
   number in the rows, per side in the side-by-side form), a marker cell
   after the gutter (`Glyphs.diff_add` `+`, `diff_remove` `−`, context a
   space; the ascii set uses `+`/`-`), and two new background styles
   `diff_add_bg`/`diff_remove_bg` (a dark green and a dark red in truecolor
   and c256; the c16 theme keeps foreground colour only; `plain` none) that
   fill the row to the pane width so a change reads as a band, with the
   changed span still highlighted inside it (`diff_change` becomes a
   brighter background of the same hue instead of reverse video); a header
   row `path` with `+N −M` counts in `diff_header`; the side-by-side form
   pads both panes to equal width and separates them with `Glyphs.table_bar`;
   the unified form keeps one gutter with both numbers. Goldens: an
   insertion, a deletion, a paired replacement with a changed span, CRLF
   context, at 80 and at 120 columns, unicode and ascii.
7. The markdown renderer (`src/tui/markdown.zig`, 818 lines, eight tests).
   Resilience: a fuzz-style test feeds truncated prefixes of every fixture
   document (every byte boundary) through `split` and `render` and asserts
   no error, no control byte in the output, and a bounded row count;
   pathological inputs get pinned goldens: a 10,000-character line without
   spaces, 64 nested list levels, an unclosed fence at end of stream, a
   table with 40 columns, a heading of only `#`, a link whose URL contains
   an escape, and mixed CRLF. Efficiency: `render` is called on every
   token for the streaming tail, so the closed part must not be re-rendered
   — the transcript caches the rows of the blocks `split` has closed (by
   the byte offset `split` returns) and renders only the open tail; a test
   counts `render` calls over a streamed fixture and the cache's hit rate.
   Behaviour: inline code inside headings and list items, `***bold
   italic***`, nested quotes, ordered lists that start at a number other
   than 1, and a table cell that is empty, each with a golden.

**Acceptance.** `make check` with the golden tests and the prefix fuzz; a
manual `nuclis agent` session on a small task shows the four elements of
the screenshot and an edit's banded diff with numbers; the streaming test
shows the closed-block cache holds (no re-render of closed blocks);
`docs/development.md § The agent's transcript` and `docs/agent-spec.md`
(the rendering section) describe the rows. No tool contract changes except
the summaries' text.

## Vision through the companion projectors — fixed before the units (decided 2026-09-20)

**Words.** A *projector file* is the catalogue's `mmproj` companion: a
GGUF of architecture `clip` holding a vision encoder and/or a projection
into the language model's width. *Preprocessing* turns a decoded image into
the projector's input (resize to a patch-aligned grid, normalize by
`clip.vision.image_mean`/`image_std`). The projector returns one *feature
row* per output token at the model's width; the *image span* is the run of
placeholder tokens in the prompt whose embedding rows are replaced by those
features. The chat shows an attached image as the chip `[image #N]`.

**The shared contract** lives in a new `inference/src/vision/` package
(`root.zig`): `Projector` (a family adapter's value: `bytes()` for the load
plan, `prepare(alloc, image: Rgb8) !Prepared` — the preprocessed tensor and
the output grid `{ width_tokens, height_tokens }` — and `encode(prepared,
out: []f32) !void` producing `grid.count × hidden` feature rows), a
`Preprocess` module (the reference's resize algorithms per family, exact),
and `Rgb8 { width, height, pixels }` decoded by `image.zig`: a P6 PPM
parser for fixtures and tests, and on macOS an ImageIO bridge
(`inference/src/vision/image_bridge.m`, `CGImageSourceCreateWithData` →
RGB8; PNG, JPEG, HEIC, WebP, TIFF) behind an opaque handle, as the Metal
bridge is. Bounds are host constants: image bytes ≤ 32 MiB, decoded pixels
≤ 64 M, at most 8 images per turn.

**The prompt seam.** `profiles.Message` gains `images: []const ImageRef`
(`{ index, grid }`), and the chip text `[image #N]` in `content` is what a
profile renders into its family's marker tokens: Qwen3.8 `<|vision_start|>`
+ `count × <|image_pad|>` + `<|vision_end|>` (ids 248053, 248056, 248054);
Gemma 4 `<|image>` … `<image|>` (ids 255999, 258882; the placeholder id
inside and any newline layout are read from the reference's `mtmd.cpp`
gemma4 branch in MODL-22's session 1); Muse `<|image_start|>` …
`<|image_end|>` with `<|patch|>` placeholders (ids 200080, 200081, 200092;
the exact layout read in MODL-23). `Engine.encode` stays text-only; after
encoding, `engine.locateImageSpans(tokens, placeholder_id)` finds the runs
and pairs them in order with the features, giving `Prompt { tokens, spans:
[]ImageSpan { start, count, features, grid } }`. `Executor.prefill` /
`Plan.prefill` / `Runtime.step` take the spans: the plan's `recordLayers`
overwrites `x_c` rows of a span with its features (one device copy per
span); the CPU runtime substitutes the row per token. A span never
straddles a prefill chunk (the engine aligns chunk boundaries to span
edges). Session snapshots, checkpoints, and speculation are unaffected:
features are consumed at prefill, and the drafter's commit takes the
target hidden as always.

**Positions and masks.** Qwen3.8 uses M-RoPE (`qwen35.rope.dimension_sections`
= [11, 11, 10, 0]; the reference's `LLAMA_ROPE_TYPE_IMROPE` for `qwen35`,
`llama-model.cpp:3023`): text tokens carry equal (t, h, w); an image span's
rows carry (t, t + h, t + w) over its grid and the span advances the text
position by `max(count_h, count_w)` (the reference's
`set_position_mrope_2d`, `mtmd-helper.cpp:139-158`, and `mtmd.cpp`'s
`MTMD_POS_TYPE_MROPE`). So MODL-21 extends the RoPE step to per-section
positions on both executors. Gemma 4's language model attends
bidirectionally inside an image span on its sliding-window layers only
(`hparams.non_causal_type = LLAMA_NON_CAUSAL_TYPE_SWA_ONLY`, the reference's
`gemma4.cpp:23-25`); MODL-22 adds a span mask to the chunk attention.
Muse's language model is unchanged by images.

**The oracle.** The pinned checkout's `tools/mtmd/` (`clip.cpp`, the
projector graphs under `models/qwen3vl.cpp`, `gemma4uv.cpp`, `gemma4v.cpp`,
`muse-glimmer.cpp`, the preprocessors in `mtmd-image.cpp`, the marker and
position logic in `mtmd.cpp`/`mtmd-helper.cpp`) and `llama-mtmd-cli -m
<main> --mmproj <mmproj> --image <file> -p <prompt>`. Each unit pins, per
family, one fixture image (a small synthetic P6 PPM under
`tests/fixtures/vision/`, committed) with the reference's projector output
rows and the first generated tokens; tolerances are set by the first trace
(BF16 weights computed in F32 on the CPU reference, F16 on Metal). The
projector files' facts, read on 2026-09-20 with `scripts/gguf-inventory.py`:

| family | file | projector type | encoder | patch | merge | output width | notes |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| Qwen3.8-27B | `mmproj-BF16.gguf` (931 MB; 224 F32, 110 BF16) | `qwen3vl_merger` | 27 blocks, 1152 wide, FFN 4304, 16 heads, GELU, eps 1e-6, image_size 768 | 16 | 2 | 5120 | `is_deepstack_layers` all 0: no deepstack in this file; mean/std 0.5 |
| Gemma 4 12B | `mmproj-BF16.gguf` (175 MB; 11 tensors) | `gemma4uv` (+ `gemma4ua` audio, not planned) | none: patches → LayerNorm → linear 768→3840 → LayerNorm → learned x/y tables → LayerNorm → RMSNorm → `mm_input_proj` | 16 | — | 3840 | image_size 224; the language model does the vision work (bidirectional SWA layers) |
| Gemma 4 26B-A4B | `mmproj-BF16.gguf` (1.19 GB; 356 tensors) | `gemma4v` | 27 blocks, 1152 wide, FFN 4304 (SigLIP), avg-pool by merge, RMSNorm, `mm_input_proj` | 16 | read | 2816 | image_size 224 |
| Muse Glimmer 30B | `mmproj-kquant.gguf` (1.4 GB; Q4_K 200, Q6_K 100, BF16 3, F32 506) | `muse-glimmer` | 50 blocks, 1536 wide, FFN 8960, 16 heads, 2D RoPE base 1e4, windowed attention (every 4th and the last layer global), pixel-shuffle ×2, adapter 6144→4096→4096 GELU, projection 4096→6656 | 14 | 2 | 6656 | image_size 896; grid by aspect-preserving search under a token cap |

None of the four files carries `image_min_pixels`/`image_max_pixels`; the
reference's per-projector defaults (`clip.cpp` ≈ 1627 for gemma4v, 1653
for qwen3vl, 1683 for muse-glimmer) are read in each unit's session 1.
Output tokens for Qwen and Muse are `(w / patch / 2) × (h / patch / 2)`.

**Where the detail goes.** A new `docs/reference/vision.md` holds the
contract, each family's projector facts and provenance, the preprocessing
per family, the traces, and the memory; `docs/spec.md` gains the vision
requirements (moved from the deferred list when MODL-21 opens);
`docs/agent-spec.md` drops "image input" from *Not in scope* when AGNT-11
opens.

## MODL-21 — The vision contract, image input, and the Qwen3.8 projector

**Session 1 (facts, then the section rewritten before code).** Read the
reference's `qwen3vl.cpp` graph (patch embedding conv, the 2D position
embedding resize, the 27 blocks, the 2×2 spatial merge into `n_embd × 4`
then the merger MLP `mm_0`/`mm_1` to 5120), `clip.cpp`'s QWEN3VL hparams
defaults (min/max pixels, `image_resize_algo`) and `set_input` positions
(the merge-ordered `(y, x)` pairs at ≈ 4781), `mtmd-image.cpp`'s
`calc_size_preserved_ratio` (align 32 = patch × merge, then min/max
pixels), and the M-RoPE position build. Dump the reference's projector
output for the fixture image (the `tools/mtmd/debug` tooling or a `cb`
hook in `clip.cpp`), and its first 8 greedy tokens for `describe this
image` on the fixture. Record everything in `vision.md` and rewrite this
section with the tensor names, shapes, and the tolerance.

**Design.**
1. `inference/src/vision/`: `root.zig` (the contract above), `image.zig`
   (P6 PPM; the ImageIO bridge with `-framework ImageIO -framework
   CoreGraphics` added to `build.zig` beside the Metal bridge), `preprocess.zig`
   (the smart resize and the reference's resize algorithm for this family,
   normalize, the `[channel][y][x]` layout the conv expects), `qwen3vl.zig`
   (the CPU reference: `bind(doc)` validates `clip.projector_type ==
   "qwen3vl_merger"` and the shapes, `Runtime` runs the conv as a matmul
   over patch vectors, the blocks with the existing `cpu` kernels, the
   merge and the MLP), `qwen3vl_metal.zig` (the plan on the existing
   kernel set: batched matmul tiles over the `n_patches` rows, the chunk
   attention kernel with a full mask over the image, RMS/LayerNorm, GELU;
   BF16 weights converted at load to F16 device buffers, recorded in the
   memory plan).
2. `Engine.open` gains `vision: ?[]const u8` (the projector path from
   `models.<name>.mmproj`; `nuclis model ls` marks it loaded-by "the vision
   unit"); `VisionSourceMismatch` on a projector whose `projection_dim`
   differs from the model's width. Loading is a load-time decision like
   the drafter.
3. The prompt seam and M-RoPE: `Prompt`/`ImageSpan` in `engine.zig`,
   `locateImageSpans`, the chunk alignment; `Runtime.step`/`Plan.prefill`
   accept feature rows; `Backend.rope`/`ropeRows` gain a per-section
   position triple (`(t, h, w)` with the sections [11, 11, 10, 0]; the
   text path passes `(p, p, p)`, bit-identical to today by construction —
   `make compare` proves it); the Qwen adapter computes the span positions.
4. `generate --image <path>` (repeatable, ≤ 8) and the rendered prompt:
   the profile's `render` replaces `[image #N]` with the marker tokens;
   `tokenize --image` shows the spans; `nuclis validate` reports the
   projector.
5. `generation-check --vision-check` (Metal and CPU): the projector's
   output rows for the fixture against the pinned trace, then greedy
   tokens against the pinned first 8.

**Acceptance.** The projector trace within the tolerance set in session 1
on both executors; the 8 greedy tokens identical; `make compare` unchanged
(text positions bit-identical); `make check` with unit tests for the PPM
parser (bounds, malformed files), the smart resize against values computed
by hand, `locateImageSpans`, and the profile's rendering; the memory record
(projector weights, activation scratch at 768 × 768: 2,304 patches) in
`vision.md`; `generate --image` on a photo produces a sensible caption
(recorded, not asserted).

## AGNT-11 — Images in the chat: drop, paste, `/image`, the `[image #N]` chip

**Facts (read 2026-09-20).** The editor (`src/tui/editor.zig`) already turns
a bracketed paste of ≥ 4 lines or ≥ 400 bytes into a `Chip { start, end,
lines }` over the buffer, rendered as one token (`[pasted 96 lines, 6.1
KB]`), deleted as one unit by backspace; a file dropped onto Terminal.app
or iTerm2 arrives as its path (spaces backslash-escaped), inside a
bracketed paste on terminals that support it and as plain typed text
otherwise. Slash commands are parsed in `src/agent/commands.zig`
(`Command`: `new`, `resume_session`, `ctx`, `think`, `save`, `help`). The
agent renders `profiles.Message`s (`content` text) through the profile and
`Completer.run` prefills the remainder; sessions persist messages
(`src/agent/session.zig`). The screenshot's behaviour is the model: an
attached image shows as `[Image #1]` in the prompt and the transcript
labels it with its source path.

**Design.**
1. Attachment model: `Editor` gains `attachments: []Attachment { path, index
   }` and an `image` chip kind (`Chip.kind: enum { paste, image }`), drawn
   as `[image #N]` with the `op_write`-style accent; the chip's bytes in the
   buffer are literally `[image #N]`, so the submitted text carries the
   marker and nothing else changes downstream. Backspace removes the chip
   and its attachment; indices are per prompt, in attachment order.
2. Three ways in, one path: (a) a bracketed paste whose trimmed content is
   one existing regular file with an image extension (`.png .jpg .jpeg
   .gif .webp .heic .tiff .bmp`, case-insensitive; `file://` prefix and
   backslash escapes removed) becomes an image chip instead of text; (b)
   `/image <path>` (with the completion the other path commands use)
   attaches and inserts the chip at the cursor; (c) at submit, any
   whitespace-separated token that resolves the same way is attached and
   replaced by a chip in the recorded prompt, for terminals that do not
   bracket pastes — the transcript shows what was attached. Ctrl-V clipboard
   images are not in this unit (macOS pasteboard access is a bridge call;
   record as a follow-up).
3. The turn: `Agent` decodes each attachment through `vision.image`
   (a typed error becomes a `notice` — `could not read image #2: …` — and
   the turn is not sent), runs the projector, and builds the user
   `Message` with `images` and the `[image #N]` content; the profile
   renders the markers; the engine attaches the spans. Bounds: 8 images per
   turn, the host limits of the vision contract. The transcript renders
   the user turn with the chip and a `dim` detail row `image #1:
   /path/to/file.png (1024×768 → 24×18 tokens)`. Sessions persist the path
   and the grid, not the pixels; a resumed session re-decodes on replay
   and reports a missing file as a notice.
4. The model without a projector (`models.<name>.mmproj` unset or the
   family has none yet): the chip is refused at attach time with a notice
   naming the reason, never silently dropped.

**Acceptance.** Editor tests for the three ways in, chip deletion, index
renumbering, and the bounds; a session round trip with an attachment;
`nuclis agent` with Qwen3.8 and a dropped screenshot answers a question
about it (recorded in the log); `docs/agent-spec.md` and
`docs/development.md § The agent's transcript` describe the chip.

## MODL-22 — Gemma 4 vision: the unified embedder (12B) and the SigLIP projector (26B-A4B)

**Session 1 (facts).** Read `gemma4uv.cpp` (the 12B: im2col patches →
LayerNorm `patch_norm_1` → `patch_embeddings_0` + `patch_bias` → LayerNorm
`patch_norm_2` → learned `position_embeddings` tables (x then y) →
LayerNorm `patch_norm_3` → RMSNorm → `mm_input_proj_w`; eps 1e-5 LayerNorm),
`gemma4v.cpp` (the 26B-A4B: SigLIP blocks, `ggml_pool_2d` average by
`n_merge`, RMSNorm, `mm_input_proj`), the GEMMA4V hparams defaults
(`clip.cpp` ≈ 1627), `mtmd.cpp`'s gemma4 branch for the placeholder token
inside `<|image>` … `<image|>` and the position handling, and the language
model's `non_causal_type = SWA_ONLY` mask (`llama-graph.cpp`, the
`LLAMA_NON_CAUSAL_TYPE_SWA_ONLY` case). Pin the fixture traces for both
entries; rewrite this section.

**Design.** `vision/gemma4.zig` with two adapters selected by
`clip.vision.projector_type` (`gemma4uv`, `gemma4v`); the 12B's is a few
matmuls and norms (no encoder), the 26B-A4B's reuses the SigLIP block code
of MODL-21 (same shapes: 1152 / 4304 / 16 heads) with the pooling; the
Gemma language model gains the span mask on sliding layers: the CPU
`fullAttention` and the chunk attention kernel take an optional
`bidirectional_spans` list (rows inside a span attend to every row of the
same span on SWA layers; global layers stay causal), tested by a fixture
with a poisoned future row outside the span. The profile renders the
markers; `generate`/`agent` as MODL-21. The audio projector in the 12B
file (`gemma4ua`) is not loaded.

**Acceptance.** Traces within tolerance for both entries on both executors;
the greedy tokens; the Gemma compare targets unchanged (text path); the
mask fixture; `make check`; captions recorded.

## MODL-23 — Muse Glimmer's windowed vision encoder

**Session 1 (facts).** Read `muse-glimmer.cpp` (the host-computed inputs:
`pos_w/pos_h` 1-indexed 2D RoPE positions in window order, `sp_perm` /
`inv_perm` window grouping, `ds_perm` pixel-shuffle gather, `sp_mask`
block-diagonal window mask on the sparse layers; the adapter and the
projection), `clip.cpp`'s MUSE_GLIMMER branch (≈ 1683: `patch_temporal =
2`, `sparse_factor = 4`; `set_input` ≈ 4582–4650 for the permutations),
`mtmd-image.cpp:1678-1760` (`muse_glimmer_grid_size`: the aspect-preserving
grid under `max_tokens = image_max_pixels / (14 · 14 · 4)`, a stretch
resize without padding), and `mtmd.cpp`'s template with `<|patch|>`
placeholders. Pin the fixture trace; rewrite this section.

**Design.** `vision/muse_glimmer.zig`: preprocessing per the grid search;
the window permutation and mask computed on the host as the reference
does; 50 blocks with 2D RoPE (the existing RoPE kernel with a per-row
position pair) and the chunk attention kernel with the block-diagonal
mask on sparse layers and a full mask on global ones (the same optional
mask input MODL-22 adds); pixel shuffle as a gather; the adapter MLP and
the projection through the existing Q4_K/Q6_K matmul tiles (the file is
K-quant). Memory: the 896 × 896 grid is 4,096 patches before the shuffle.

**Acceptance.** Trace within tolerance on both executors; greedy tokens;
the Muse compare targets unchanged; the mask fixture; captions recorded;
`make check`.

## AGNT-12 — Background commands (drafted 2026-09-20 for decision)

**What the screenshot shows.** A long-running command moves to the
background; the transcript keeps its row with a pulsing dot, the model gets
a handle, and the result arrives later as a tool result.

**Assessment.** It is worth doing, after TERM-10 and separately from it,
because it changes the loop, not the rendering: today `bash` runs to
completion or its 300 s timeout inside one tool step, and the agent loop
has one event source (the model's stream). Backgrounding needs a second
one: a job table whose completions are delivered as tool results at the
next step boundary. The parts, all bounded by the agent rules:
1. `bash` gains `background: bool` (a model-supplied flag, not a limit):
   the tool spawns, redirects both streams to a workspace-scoped temp
   file, and returns at once with `{ job: N, pid }`; at most
   `max_jobs = 2` jobs, else a typed `TooManyJobs` result.
2. `jobs` tool: `read(job, offset)` (bounded by the existing result budget,
   truncation marked), `wait(job, seconds ≤ 60)`, `kill(job)`; the
   workspace kills every job at turn cancellation and process exit (the
   `bash` tool's kill-and-reap path already exists).
3. Delivery: when a job exits, the loop injects a `tool` message
   (`job N exited with status S; last 40 lines: …`) before the next model
   step, so the model never polls; the transcript row pulses while the job
   runs and settles with the exit status.
4. The UI: `/jobs` lists them; a job's row keeps its dot; the status bar
   counts running jobs.

**Cost.** Roughly the size of AGNT-11: the job table and delivery in
`src/agent/loop.zig` and `tools/bash.zig`, a new tool definition in every
profile's tool fixtures (`scripts/profile-tools-fixtures.py`), transcript
rows, and cancellation tests. The risk is an orphaned process; the
mitigation is the workspace owning every pid. Decide after TERM-10.

## KERN-15 — Split-K decode matvec for row-poor shapes

**Facts (from the Muse Glimmer profile, MODL-12/13).** The decode matvecs'
bandwidth tracks the matrix's row count, not its encoding: the 202,048-row
head streams at 208 GB/s, the 39,936-row FFN pair at 175, and the two
shapes with 6,656 to 8,704 rows (the FFN down projection over 19,968
columns, the merged q/k/v/gate) at 145 to 147, because a matrix with few
rows launches too few threadgroups to keep the bus busy; Muse's 6,656-wide
residual puts more than a third of its bytes in those shapes, and at the
head's rate its token would take 84 ms instead of 100 (two thirds of its
66–71 % gap to the reference). Qwen's 5,120-row shapes (`ffn_down`,
`attn_output`, the DeltaNet output) and Gemma's have the same profile at a
smaller share. Launch count is the rest: 922 dispatches per Muse token
with six RMS-norm launches per layer (≈ 9 ms).
([muse-glimmer.md § Metal plan](docs/reference/muse-glimmer.md#metal-plan-modl-12-2026-09-19),
[bench.md](docs/reference/bench.md#muse-glimmer-30b-acceptance-record-modl-13-2026-09-19)).
The 4K row's decode drift within one run (9.14 to 8.12 tok/s in four
minutes) is reproduced first, with the GPU clock watched
(`powermetrics --samplers gpu_power`), before anything is measured.

**Design.** Split-K on the specialized matvecs for shapes below
`split_k_rows = 16,384` rows: several threadgroups per row block, each
summing a column range, reduced by a second tiny kernel or an atomic-free
partial buffer (`[splits][rows]`, reduced in the consumer's next launch
where one exists, else `nu_reduce_splits`); the split count from a
`bench-kernels` sweep (2, 4, 8) per shape. Shares the per-lane decode
loops with KERN-12's multi-row kernel (one template, `NU_ROWS` and
`NU_SPLITS`). Files: `kernels.metal`, `backends/metal/root.zig`
(`specializedMatvec` gains the split tier), `metal-check.zig` (exactness
at every encoding for a 6,656-row shape; the sweep), the family plans
unchanged.

**Acceptance.** The two row-poor Muse shapes ≥ 190 GB/s on
`bench-kernels`; Muse decode at 512 ≥ 10.5 tok/s (from 9.60; the
reference 13.69); Qwen decode not slower; `make compare` and every family's
compare target unchanged; the acceptance records re-run for Muse.

## KERN-16 — Long-context prefill attention, second attempt

**Facts (ENGN-05, ENGN-08).** The 32K acceptance record measured prefill
below the reference (−6 % at 4K, −15 % at 16K, −26 % at 32,639) and the
profile put the chunk attention kernel at 30 % of the 16K prefill, running
at 0.7 TFLOP/s: latency-bound with one `simdgroup_load` per multiply, not
cache-bound. Sharing the cache tiles across the six heads of a KV group
did not help (three variants, all slower; the re-reads were cache hits).
The speculative verify batch reads the whole visible cache once per batch
through the same kernel, so at 16K–32K context this kernel bounds
speculation too ([metal-backend.md § ENGN-08](docs/reference/metal-backend.md#long-context-prefill-attention-engn-08-2026-09-10-closed-without-a-kernel-change)).

**Design.** Register-level reuse as in the matmul tiles: split the value
columns across the four SIMD groups of a (head, 32-query) tile so each V
block serves four row blocks, share the probability tiles through
threadgroup memory, and budget the registers (48 live matrices per SIMD
group) on paper before writing it; `attention_chunk` and its half
variant; the F64 fixture with poisoned future rows; `bench-attention`
(new, no model) sweeping visible rows 4K–32K. Measure at 16K and 32,639
on the reference arrays (`make baseline`).

**Acceptance.** Prefill at 32,639 within 10 % of the reference (from
−26 %); 16K within 8 %; `make test-metal`, `make compare`, the acceptance
record re-run; the verify batch at 16K context measurably faster on the
speculative record's 4K/16K rows.

## ENGN-18 — Gemma 4 12B decode: fused norms and fewer launches

**Facts (MODL-07/08 profile).** On the QAT file decode is 78–87 % of the
reference and prefill 80 % at 512 falling to 51 % at 32,639, with nothing
tuned for Gemma; the per-kernel profile says where the 5.6 ms/step gap is
not: the Q4_0 matvecs hold their isolated bandwidth and are 85 % of the
step. What is left, in order: the RMS-norm launches per step
(launch-bound), the matvecs' share of the bus, the Q4_0 prefill tile (two
16-value segments per block), the tied Q4_0 head, the sliding decode, and
the wide global-layer attention; a prefill tile keeping activations in F32
belongs here too (the QAT checkpoint amplifies half-operand rounding about
five times more than the K-quant file) ([bench.md](docs/reference/bench.md),
[gemma4.md](docs/reference/gemma4.md)).

**Design.** One experiment per session, each judged on the acceptance
record: a fused norm-and-scale kernel (the post norms into the residual
add, the q/k head norms into the RoPE launch) counted by dispatches per
token in `bench --profile`; then the Q4_0 tile's staged block decode. Stop
when a step is within 5 % of the reference or the remaining items are the
matvecs themselves.

**Acceptance.** Dispatches per token reduced by ≥ 30 %; decode at 512 ≥
93 % of the reference; the Gemma compare targets unchanged; the QAT
acceptance record re-run.

## ENGN-19 — A ring layout for windowed attention caches

**Facts.** Gemma 4's 40 sliding layers see 1,024 positions but their caches
are allocated for the full session capacity on both backends: 11.3 GB of a
32K F16 session, of which a ring would keep 0.35 GB; Muse's 2,048-row
windows are the same shape at 1.63 GiB per 32K session
([gemma4.md](docs/reference/gemma4.md), [muse-glimmer.md](docs/reference/muse-glimmer.md)).

**Design.** A `session.Layout` variant `.windowed_attention { window }`
with a modulo row index, touching the CPU reference's slice, the decode
slice, the chunk kernel's key loop (wraparound inside a chunk: two ranges
per chunk), `snapshot`/`restore`, `checkpoint`/`truncate` (a rewind past
the window's oldest row is refused), and `bytes()`; a session-layout unit
with its own fixtures (a window smaller than a chunk, a chunk straddling
the wrap, a rewind at the wrap). Memory is the deliverable; the speed
must not regress.

**Acceptance.** The Gemma 32K session under 3 GB and Muse's under 1 GB
with decode and prefill within 2 % of before; every compare and
generation check unchanged; the memory rows of both records refreshed.

## KERN-17 — The ternary matvec's arithmetic (experiment)

**Facts (KERN-10, MODL-17).** Bonsai 2 27B decodes at 1.3× the
Qwen3.8-27B rate where its byte count promises 2–3×, and at 80–87 % of
the PrismML fork: the PQ2_0 / PTQ1_0 matvecs move about 100 GB/s of
weight bytes because they are at the kernel set's multiply-rate ceiling
(one field mask per four values, one integer-to-float conversion, one FMA
per value; a ternary byte carries twice the values of a 4-bit byte). The
transform and the gather are 2.3 % of the step and not the lever
([metal-backend.md](docs/reference/metal-backend.md#ternary-matvecs-and-tiles-kern-10-2026-09-18),
[bench.md](docs/reference/bench.md#bonsai-2-27b-acceptance-record-modl-17-2026-09-18)).

**Design.** Two experiments in one session, on `bench-kernels`: packed
integer products (several trits against several inputs in one
integer multiply-add, the inputs pre-quantized to int8 per block) and one
decoded weight shared across several inputs (the multi-row idea of
KERN-12 applied to the ternary decode). Keep the one that measures above
150 GB/s; close negative with the numbers if neither does — Bonsai is one
entry and the payoff is uncertain.

**Acceptance.** A measured verdict either way; if positive, Bonsai decode
≥ 1.6× Qwen's on the record; the Bonsai compare targets unchanged.

## APPS-14 — Teacher-forced `eval` (drafted 2026-09-20 for decision)

**What it is.** `nuclis eval --model <m> --file <text> [--ctx N]` feeds a
fixed text through the model with *teacher forcing*: every position is
fed the reference's actual next token, never a sampled one, and the
command records the model's log-probability of that token from the logits
of each step. The output is the mean negative log-likelihood and its
exponential, the perplexity, over the text (the reference's
`llama-perplexity`), optionally per chunk.

**Why it matters here.** Every numerical check in the tree today is a
pinned trace at two or three positions (`make compare`) or a greedy
token-for-token equality on one seed; both catch a wrong kernel and both
are blind to small systematic drift over long text, which is exactly what
a quantization choice, an F16 cache, a half-operand tile, or a new matvec
introduces. Perplexity on a fixed corpus is the one number that ranks
those choices against each other and against the reference on the same
text, independent of sampling; it is how the catalogue's quantization
verdicts, KERN-12's multi-row kernel, the F16 cache bound, and the
projector's effect on the language model would be judged at scale rather
than at three positions. It is cheap to build: prefill already yields the
logits of every row on Metal (`verify`), and the CPU reference steps token
by token.

**Design, if accepted.** `src/eval.zig`: tokenize the file, run it in
prefill chunks with all-rows logits (`Executor.verify` without a drafter,
or a `prefillLogits` sibling), sum `−log softmax(row)[next]` in F64, report
NLL, perplexity, tokens, and the per-chunk series as text and JSON;
`--reference <json>` compares against a pinned `llama-perplexity` run on
the same text (the pinned checkout builds it). Fixture: the reference
corpus's first 4,096 tokens. One session.

**Acceptance.** Perplexity on the corpus within 0.5 % of the reference's
on the same file and context; `make check`; the spec's CLI section names
the command.
