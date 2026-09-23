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

ENGN-16 closed on 2026-09-20: the drafter's `p_min` early stop is shipped
at `engine.draft_p_min = 0.7` (a position whose top candidate is below it
ends the chain), while the adaptive length closed negative and is not. The
gate pass (`--only prose512 code`) reads code instruct **1.35×**, code
greedy **1.33×** at draft 7 (within 1 % of the control), prose instruct
**1.00 / 1.03×** at drafts 4 / 7 (was 0.98 / 1.01×), and drafts/accepted
1.24–1.73; the unit's prose 30 % bar is missed by three points (−27 %
against the interleaved control) while its decode rate rises. Facts:
[speculative-decoding.md § The proposal policy](docs/reference/speculative-decoding.md#the-proposal-policy-engn-16-2026-09-20),
the table in
[bench.md § The ENGN-16 quick pass](docs/reference/bench.md#the-engn-16-quick-pass-2026-09-20),
and the [log](docs/engineering-log.md#engn-16--draft-proposal-policy-the-p_min-early-stop-shipped-the-adaptive-length-dropped-2026-09-20).

Before it in the same session: KERN-13 (device penalty kernel) and ENGN-15
(sampled acceptance on the per-row top-k) closed positive — `accept` fell
from 62.8–91.3 ms per batch to **18.8–36.9 µs**, code instruct 1.36×,
prose instruct 0.92 / 0.98 / 1.01× at drafts 2 / 4 / 7 — and KERN-14 (the
wide 32×8 small-batch tile) closed negative, keeping the 16×8 control. The
deferred CPU speculative check was run and passed (12 tokens identical to
ordinary greedy).

KERN-15 (split-K decode matvec for the row-poor shapes) closed negative on
2026-09-21: the split bodies measure behind the single pass on every row
bucket (Q4_K 6,656×19,968 151.0 → 145.9 / 148.1 / 138.8 GB/s at 2/4/8
splits; the four-segment merge 152.1 → 157.4 / 150.3 / 158.5, below the
≥ 190 bar), so nothing routes to them and the Q4_K kernels' ~0.9 ns per
256-value block stands as Muse's decode limiter
([bench.md § Split-K matvec sweep](docs/reference/bench.md#split-k-matvec-sweep-kern-15-2026-09-21)).

KERN-16 (long-context prefill attention, second attempt) closed negative
on 2026-09-21: the register-reuse body is correct and 2–5 % ahead at the
256-row prefill chunks (0 % at 512, −4.8 % at 16K, −3.2 % at 32,512 in
F16) against the attention cut its acceptance needed, and 11–16 % ahead on
the verify-shaped counts (1–64 rows), which ship as
`attention_reuse_max_rows = 64` on the 256-wide geometry while the prefill
chunks keep the row-split body. The loads were not the limiter (F16 vs F32
caches differ 5–10 %, the kernel is flat at ~640–750 GFLOP/s F16), so the
long-context deficit is not attention-load-bound; a third attempt would
start from the untried levers in
[metal-backend.md § KERN-16](docs/reference/metal-backend.md#long-context-prefill-attention-second-attempt-kern-16-2026-09-21-closed-negative)
([bench.md § Prefill attention sweep](docs/reference/bench.md#prefill-attention-sweep-kern-16-2026-09-21)).

KERN-18 (fused decode norms) closed below its target on 2026-09-21: the
three fused kernels ship (Qwen's add+post norm and full-attention q/k
norms, Gemma's post norms and q/k norms, Muse's post norms and sliding q/k
norms) at −96 / −192 / −182 dispatches per decode step, but the speed bars
miss — decode at 512 reads 1.000× Qwen, 1.005× Gemma, 1.004× Muse against
≥ 1.01 / 1.04 / 1.02× — because the profile's per-dispatch time was kernel
work, not a launch floor, so fusing two memory-bound passes saves only the
removed launch. The pairs stay behind `Backend.fused_norms` /
`bench --unfused-norms`, and ENGN-17's record measures the shipped path
([bench.md § Fused norm sweep](docs/reference/bench.md#fused-norm-sweep-kern-18-2026-09-21)).
**Two facts for MODL-19/20:** the route to a real decode win is epilogue
fusion (the norm inside the kernel that produces its input), not merging
dispatches, and `bench` on a family whose companion has no adapter fails
with `DraftSourceMissing` while the switch is on (fixed for Gemma by
MODL-19 and for Muse by MODL-20).

MODL-19 (Gemma 4 draft heads) closed on 2026-09-21: the
`gemma4-assistant` companion is a second GGUF that `Engine.open` maps and
validates, the CPU and Metal plans read the target's layer-46/47 caches
without owning one, and the pinned trace matches at 1.1e-4 max abs / 3.6e-6
rel RMS (CPU) and 1.5e-4 / 5.4e-6 (Metal) with greedy 2613 and 236764. The
verdict is **negative at draft 4**: 2.26 accepted per batch, every non-verify
cost negligible, but the verify batch is 136 ms at 3–8 rows, so the pair
runs 0.899× at draft 4 and only 1.017× at draft 7 against 25.2–25.4 tok/s
ordinary decode. Facts and the table:
[speculative-decoding.md § The Gemma 4 assistant heads](docs/reference/speculative-decoding.md#the-gemma-4-assistant-heads-modl-19)
and [bench.md § The Gemma 4 draft pair](docs/reference/bench.md#the-gemma-4-draft-pair-modl-19-2026-09-21).
The 26B-A4B head is bound and width-checked but not measured. The verifier's
row-flat cost is the lever, not the drafter: a cheaper small-batch verify is
an untaken lever (ENGN-17's record kept the default off).

REPO-13 closed on 2026-09-22: `spec.md` is rewritten as the one
technical specification with the agent spec merged in and removed
([log](docs/engineering-log.md#repo-13--one-specification-specmd-rewritten-as-a-technical-specification-with-the-agent-spec-merged-in-2026-09-22)).
REPO-12 closed on 2026-09-22: the architecture guide follows the KV cache
end to end and is current with the tree, and the inference guide is
rewritten as a 27-section narrative
([log](docs/engineering-log.md#repo-12--the-architecture-guide-follows-the-kv-cache-end-to-end-the-inference-guide-rewritten-as-one-narrative-2026-09-22)).

TERM-10 closed on 2026-09-22 across its two sessions (eight commits on
2026-09-21, three on 2026-09-22): the repaint tick and the visible
warm-up, the boxed header, the framed input box, the two-group status
bar, the operation dots, then the banded diff with its gutter and counts,
the markdown hardening (a prefix fuzz over ten fixtures, pathological
goldens, nested quotes, `***bold italic***`, soft line breaks kept, no row
ever past the edge, one render per closed block), `!`/`!!` shell commands
from the editor, Ctrl-O folding tool output, Ctrl-X copying the last
answer, and Ctrl-G editing the input in `$VISUAL`/`$EDITOR`. 487 tests,
the Metal tier green, the captures under `.zig-cache/tui/`
([log](docs/engineering-log.md#term-10--chat-polish-the-repaint-tick-the-frame-the-operation-rows-the-banded-diff-and-the-editors-shell-and-keys-2026-09-21--2026-09-22)).

AGNT-13 closed on 2026-09-22: the system prompt is sections in
`src/agent/system_prompt.zig`, pinned byte for byte to a fixture, with
`agent.instructions` (`auto`/`off`/a path, 8 KiB cap) and `--system-prompt
<file>` for tuning; the playground task list (`scripts/agent-eval.py`,
`make agent-eval VARIANT=…`, twelve tasks, two seeds, counted habits) is
how prompt changes are judged from now on (AGENTS.md § Validation item 5).
Five variants were measured; the shipped text removed the regex-to-`grep`,
`pytest`, and `cd` habits and cut the answer length a fifth, at 23–24 of
24 passes on every variant
([log](docs/engineering-log.md#agnt-13--the-system-prompt-as-sections-measured-the-playground-task-list-the-guidelines-that-changed-behaviour-the-instructions-file-2026-09-22)).
AGNT-14 closed on 2026-09-22: the Qwen decoder strips only the two
delimiter newlines (a value keeps its trailing newline), `read_file`
serves the first MiB of a larger file with the size stated, and Enter
steers a running turn (Alt-Enter queues); `newfile` fell from 10–12 steps
and 240–346 s to 8–9 steps and 138–190 s
([log](docs/engineering-log.md#agnt-14--three-measured-fixes-the-qwen-decoder-keeps-a-values-trailing-newline-read_file-serves-the-first-mib-enter-steers-a-running-turn-2026-09-22)).
MODL-21 closed on 2026-09-22: the `inference/src/vision/` contract, the
image decoders, the exact preprocessing, and the Qwen3.8 `qwen3vl_merger`
projector on both executors, with image spans through the language model
(M-RoPE per row, feature rows at prefill) and `generate --image`. The
projector matches the pinned oracle at 0.39 max abs / 3.9e-3 relative RMS
(the reference's own CPU/Metal spread is 0.36 / 4.6e-3), the eight greedy
tokens on the pinned features are identical on both executors, `make
verify` is 27/27 with every text trace gate unchanged, and `qwen38-vision-cpu`
passed
([log](docs/engineering-log.md#modl-21--the-vision-contract-image-input-and-the-qwen38-projector-on-both-executors-2026-09-22),
[vision.md](docs/reference/vision.md)).
AGNT-15 closed on 2026-09-23 (taken before AGNT-16, the user's call):
images and text files drop into the chat as chips, `/image <path>`
attaches one, the projector runs at submit, the transcript shows a detail
row and (on Ghostty/kitty) a preview, sessions record path and grid and
resume by re-encoding; the caption of a dropped scene, the Gemma refusal,
and the resume were captured in the harness, which gained a `paste=` step
([log](docs/engineering-log.md#agnt-15--images-and-text-files-in-the-chat-drop-image-the-chips-the-projector-turn-the-detail-row-and-preview-sessions-2026-09-23),
[vision.md § Images in the chat](docs/reference/vision.md#images-in-the-chat-agnt-15-2026-09-23)).
MODL-24 closed the same day: every Qwen decode step after an image wrote
and read the wrong cache rows (the image-adjusted rotary position was also
used as the cache row and the visible count), so a real photo was
described with the wrong hair, skin, and window side; fixed, and the
vision gate now compares decode after an image with one prefill. The
reference's answer and ours now agree on 182 of 183 teacher-forced tokens.
TERM-12 the same day: a spinner row while the projector loads and images
encode. TERM-11 closed the same day from the user's screenshots: the region
re-anchors on a resize from the terminal's cursor report, the preview is
capped to the room above the region and is off under tmux (tmux cannot
scroll it). **One check is the user's:** in plain Ghostty (no tmux), drop
an image, ask, and see that the picture scrolls with the text.
**Next: AGNT-16** (AGNT-14's three follow-ups; its section below).
**AGNT-12 (background commands) was dropped on 2026-09-22**, the user's
call after AGNT-13's measurement: a step costs 10–100 s on this engine, so
the model has nothing to do while a command runs in the background, no
task on the list ran a command longer than a few seconds, and a second tool
in the primed block is a cost every turn; its one valuable part, steering,
moved into AGNT-14. The images unit is renumbered AGNT-15: AGNT-11 and
AGNT-12 are closed identifiers in the log (2026-09-17/18) and are never
reused.
v0.2.0 was tagged on 2026-09-21
(`142fa81`, 133 changelog entries since v0.1.0), pushed, and published by
`release.yml` with its three assets; the tree is `0.3.0-dev`. APPS-16
(output budget default 4096, cap 16384) closed the same day
([log](docs/engineering-log.md#apps-16--output-budget-default-4096-cap-16384-2026-09-21)).
APPS-15 closed on 2026-09-21: `nuclis config init --discover [--dry-run]
[--json]` registers the runnable GGUF files the catalogue does not name
(the HauhauCS Gemma 4 finetune with its projector and a forced `gemma4`
profile, the Bonsai PQ2_0 bring-up file), and every `--help` page is
self-contained in the Usage / Options / Examples / Notes form with no
pointer at the repository's documents
([log](docs/engineering-log.md#apps-15--config-init---discover-and-self-contained-help-pages-2026-09-21)).
REPO-11 (the bench.md split) was dropped on 2026-09-21, the user's call.
REPO-10 closed on 2026-09-21: the 30
benchmark workloads are data in `workloads.json` (the acceptance runs per
pinned file, the prose/code runs and their draft pairs per family, the
twelve `qwen38/spec/*` configurations of the speculative record), run by
`scripts/workloads.py` through `make workload NAME=…` with reports saved
under `.zig-cache/bench/<workload>/<rev>-<n>.json`, and
`scripts/bench-report.py` renders the record tables from saved JSON and
writes them between `<!-- bench:NAME -->` markers; the Gemma QAT draft pair
reproduced MODL-19's row (0.90× against 0.899×) and its table in bench.md
is the first generated one. REPO-09 closed the same day: the 37 model-specific checks are
gates in `gates.json` with a tier and source globs, run by
`scripts/gates.py` through `make verify`, `verify-cpu`, `verify-changed`,
and `gate NAME=`; the Makefile is down to 40 targets
([development.md § Gates](docs/development.md#gates) and
[§ The record](docs/development.md#the-record); the log entries
[REPO-09](docs/engineering-log.md#repo-09--one-gate-registry-tiers-change-triggers-and-the-model-specific-checks-as-data-2026-09-21)
and
[REPO-10](docs/engineering-log.md#repo-10--benchmark-workloads-as-data-and-generated-record-tables-2026-09-21)).
ENGN-17 closed on 2026-09-21 with the speculative plan's
verdict and the full record: Qwen **off** (code greedy 1.20–1.30×, prose
512 0.81–0.97×, 4K 0.73×), Gemma **off** (0.899× at draft 4, 1.017× at 7),
Muse **on** (1.163–1.234×); the catalogue entries carry the verdicts and
`config init` writes them into each entry's `generation`. `bench` opens with
no drafter at all when the switch is off — the true baseline — and the
loaded-but-off rate equals it within ±2.4 % on eleven of twelve
configurations (session 2,304 → 3,851 MB, load 790 → 1,247 ms). Facts:
[bench.md § The speculative verdict record](docs/reference/bench.md#the-speculative-verdict-record-engn-17-2026-09-21),
[spec.md § Speculative decoding](docs/spec.md#speculative-decoding), and the
[log](docs/engineering-log.md#engn-17--the-speculative-verdict-qwen-and-gemma-off-muse-on-the-full-record-and-benchs-true-baseline-2026-09-21).

Speculative decoding works end to end on all three families; it is a
speedup worth switching on by default only for Muse (ENGN-17's verdict).
ENGN-11 (recovery), MODL-18 (the
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
decode step ≈ 95–105 ms; ENGN-17's final record, see *Working a unit here*
for how to refresh them):

| cost per batch | measured (record, 2026-09-21) | cause | unit | target |
| --- | ---: | --- | --- | ---: |
| propose `k` drafts | 10.1–23.1 ms | one block forward per draft, trimmed by `p_min` | ENGN-16 ✓ (early stop; adaptive dropped) | fewer forwards, same accepted tokens |
| checkpoint | 2.8–3.6 ms | one 150 MB copy | — | — |
| verify `1 + k` rows | 218.5–229.1 ms at 512, 309.2–319.2 ms at 4K | the small-chunk tiles' row-flat work and the chunk attention over the visible cache | KERN-16 ✗ (attention −2–5 % at chunk sizes; the verify-shaped routing lives) | ≤ 130 ms |
| accept (sampled) | 0.00–0.03 ms | one draw per row on the device readback | ENGN-15 ✓ | ≤ 5 ms |
| recover (on rejection) | 6.0–10.7 ms (was 150–182) | one 150 MB slot copy | ENGN-14 ✓ | ≤ 40 ms |
| commit `a + 1` tokens | 4.3–9.5 ms | one batched forward per committed prefix | ENGN-13 ✓ | ≤ 8 ms |
| prompt commit (prefill) | 1.02–1.03× ordinary prefill (was 2.9–3.3×) | the plan's own chunk, not 8-row verify chunks | ENGN-13 ✓ | ≤ 1.10 × |
| tokens per batch | 2.12–3.53 (1.12–2.53 accepted) | acceptance 42 % per draft on prose, 58–68 % on code | ENGN-16 ✓ (drafts/accepted 1.24–1.73) | more accepted per proposed |

The verdict is ENGN-17's record: code greedy 0.92 / 1.20 / 1.30× at drafts
2 / 4 / 7, code instruct 1.28× at 4, prose 512 greedy 0.81 / 0.91 / 0.97×,
instruct 0.84 / 0.93 / 0.95×, 4K 0.73× both — below the code ≥ 1.5× and
prose ≥ 0.9× bars, so the Qwen entry stays off at `draft_length` 4. The
batch's model time is what remains (the verify), and every kernel lever the
plan ordered closed below its target; the record's numbers and the
per-family defaults are in
[bench.md § The speculative verdict record](docs/reference/bench.md#the-speculative-verdict-record-engn-17-2026-09-21).

Order: AGNT-16 → MODL-22 → MODL-23 (MODL-21 and AGNT-15 closed first, so
the chat has its image path; the Gemma 4 and Muse projectors follow). APPS-14
stays drafted for decision on its own. KERN-13, ENGN-15, and ENGN-16 landed
first (the penalty kernel, the sampled readback, the proposal policy).
KERN-14's small-batch tile, KERN-15's split-K matvec, and KERN-16's
register-reuse attention closed negative, so
verify stays on the 16×8 tile at 512 and the row-poor shapes on the
single-pass kernel; KERN-18 closed below its target with the fused norms
shipped, and its dispatch counts are live in every verify batch's layer
passes. **KERN-16's
verify-shaped routing landed before ENGN-17 because the verdict measures
the Qwen path it changes**: the reuse body takes the 1–64-row chunk
attention of every verify batch at 16K–32K context, measured 11–16 %
faster at the kernel.
**MODL-19 and MODL-20 closed on 2026-09-21**: Gemma's heads measured
negative at the plan's draft length, Muse's DFlash drafter positive
(1.16–1.23×), and ENGN-17 set each entry's default from its own record
(Qwen and Gemma off, Muse on).
The vision units follow the agent ones.
**Three units were dropped from the plan on 2026-09-21** (the user's call):
Gemma's launch-bound decode and prefill (the Q4_0 tile and fewer launches),
the ring layout for windowed caches, and the ternary matvec arithmetic.
Their observations stay in the reference docs as measured limits — the
Q4_0 prefill tile is untaken, the sliding caches are still allocated for
the full capacity
([gemma4.md § Metal plan](docs/reference/gemma4.md#metal-plan-modl-06-2026-09-11)),
and the ternary matvec keeps its current arithmetic
([bench.md § Bonsai 2](docs/reference/bench.md#bonsai-2-27b-acceptance-record-modl-17-2026-09-18))
— but no unit carries them. The performance group has no unit left; the small-batch tile, the split-K matvec, the
long-context attention, and the fused norms led the order and closed below
their targets. APPS-14 (teacher-forced `eval`) is drafted for decision, not ordered.
**REPO-09 and REPO-10 were ordered on 2026-09-21** (the user's call): the
gate registry and the benchmark workloads are written once against the
finished set of families and records, so they follow ENGN-17 rather than
preceding it; REPO-10's workloads take their model paths from REPO-09's
manifest.

| Unit | Title | Sessions |
| --- | --- | --- |
| AGNT-16 | AGNT-14's follow-ups: the Muse value contract measured, steering that interrupts reasoning, the guessed-path guideline (ordered 2026-09-22; see its section) | 1 |
| MODL-22 | Gemma 4 vision: the unified embedder (12B) and the SigLIP projector (26B-A4B) | 2 |
| MODL-23 | Muse Glimmer's windowed vision encoder | 2 |
| APPS-14 | Teacher-forced `eval` (drafted for decision; see its section) | — |

## Working a unit here

**Build and gates.** `make build` writes `./zig-out/bin/nuclis` (Metal on).
`make check` (seconds) at every commit; `make verify` (the Metal tier of
[gates.json](gates.json), about 7.5 minutes) once per unit, and `make
verify-cpu` only when `make verify-changed BASE=<the unit's base commit>`
selects a CPU gate; a single gate by name with `make gate NAME=…`
([development.md § Gates](docs/development.md#gates)). A unit that touches
the block's forward or its commit reproduces `make gate
NAME=qwen38-draft-stats` within one draft per cell (MODL-18's 90/80/69/64 %
and 94/83/83/86 % at depths 0–3). The log entry names the tiers run.

**The record.** Every benchmark is a workload in [workloads.json](workloads.json):
`make workload NAME=…` runs it (a name or a glob: `qwen38/prose512`,
`gemma4-qat/prose512-draft`, `'qwen38/spec/*'` for the twelve
configurations of the speculative record, `muse/acceptance` for the
four-length acceptance run) and saves the report under
`.zig-cache/bench/<workload>/<rev>-<n>.json`; `scripts/bench-report.py
--table` renders the table and `--write-doc DOC --name NAME` writes it
between the document's `<!-- bench:NAME -->` markers, with the JSON copied
to `docs/benchmarks/` when the document cites it
([development.md § The record](docs/development.md#the-record)). Ordinary
decode on the Qwen 512 workload is 10.62 tok/s (94.2 ms/step) and 10.20 at
4K, prefill 90.45 and 83.70 tok/s (the 2026-09-10 record). Nothing else may
use the GPU during a record; state the git revision, and never present an
estimate as a measurement.

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
rule and the recovery scheme are not exposed. The role name stays `mtp`
(MODL-19): the spec's "the `mtp` role names the draft source whatever its
mechanism" already carries the meaning, and a rename would migrate every
sidecar for no behavior.

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

**The lesson of MODL-24 (2026-09-23), binding on every vision unit.** A
fixture that passes can still hide a broken decode. Qwen's vision gate
matched eight greedy tokens on a 4×3-token image while every step *after*
a 32×32 image wrote and read the wrong cache rows: the step used the
image-adjusted rotary position as its cache row. On a 12-token image the
two differ by 8 and the tokens happened not to flip; on a real photo the
model described the wrong person. So each family's acceptance includes,
beyond its trace:
1. **Decode after an image equals one prefill** (the generation check's
   step-versus-prefill comparison, already in `visionCheck` for Qwen):
   reset, prefill the prompt, step through the greedy tokens, and compare
   the logits with one batched prefill of prompt plus tokens, bound 2e-2.
   Run it on both executors and prove it catches a planted fault once.
2. **A large real image against the reference**, not only the synthetic
   fixture: `test-generation -- <model> --metal --vision-check <mmproj>
   --vision-image <file> --vision-oracle <dir>` with the oracle directory
   from `scripts/reference-vision.cpp`, at the projector's largest grid.
   Report the projector rows' relative RMS and per-block map, the last
   position's logits, and the teacher-forced agreement with the reference
   CLI's greedy answer; agreement below 97 % is a defect until explained
   by margins under 0.3.
3. **Anything that lets a row's position differ from its cache index**
   (M-RoPE advances, Gemma's bidirectional span, Muse's windows or
   permutations) names, in the unit's section, which index each kernel
   receives: the cache row, the visible count, and the rotary position
   are separate parameters, never one value reused.

**Where the detail goes.** A new `docs/reference/vision.md` holds the
contract, each family's projector facts and provenance, the preprocessing
per family, the traces, and the memory; `docs/spec.md` carries the vision
requirements (MODL-21) and the chat's attachment rules (AGNT-15).

## AGNT-16 — AGNT-14's follow-ups: the Muse value contract measured, steering that interrupts reasoning, the guessed-path guideline (ordered 2026-09-22)

**1. Muse's ATEM values.** Read on 2026-09-22: `muse_glimmer.zig`'s
`parseTool` passes a parameter value verbatim (`value_start[0..value_end]`)
and the reference's ATEM grammar (`common/chat.cpp:2261-2267`) takes a
string value as `until(PARAM_END)`, so Muse never had the trimming defect;
AGNT-14's note was wrong. Pin it: a test that a `content` value ending in
`\n` round-trips through `renderCall` and `parseTool`; measure it: the
`newfile` task on `muse-glimmer-30b`, seed 1, recorded in the log (the
model's `write_file` content ends with a newline or not, and the steps).

**2. Steering interrupts reasoning.** Today a steered message waits for the
step boundary, so during a long reasoning step (4,000 tokens at `think
low` happened on the list) it sits. Rule: reasoning is interruptible,
output is not. In `Agent.feed`'s `.thinking` arm, when steering is pending
and the step has produced no answer text and decoded no call, request the
interrupt (`interrupt.request()`, the same flag Ctrl-C sets) and mark
`steer_break`; in `steps()`, a `.cancelled` outcome with `steer_break` set
is not the turn's end: clear the flag and the interrupt, drop the partial
step (not appended to history, not recorded — the transcript already
showed its deltas, and a notice `— steering: reasoning restarted` follows),
deliver the steering, and continue; the step counts against the budget.
Once an answer or a call has begun, the step runs to its boundary as
before. The next completion replays from the primed prefix (a cancelled
step clears the consumed text), which is the cost. Test: a stub whose
`run` sends thinking, then consults `interrupt.requested()` and returns
`.cancelled`; a steer placed before the turn lands after a restarted step
whose partial reasoning is not in history; and a steer with answer text
already streamed waits for the boundary. Print mode unaffected.

**3. The guessed-path guideline.** `newfile` seed 2 read
`tests/test_circle.py`, a name the model invented (a `FileNotFound`
result, one step). Add to the `read_file` guideline: "never guess a path:
when you are not sure a file exists, glob first". The pinned text is
re-pinned with a full pass of the task list (`--variant agnt16`, seeds 1
and 2) against AGNT-13's `final`; the change is judged on the habit
(guessed reads, counted from the sessions as a `FileNotFound` on
`read_file`) and the pass marks.

**Also.** `agent.instructions` is written into the user's `nuclis.json`
(`config set agent.instructions auto`, done 2026-09-22; user data, not a
commit).

**Check.** `make check`; the agnt16 pass; the Muse newfile run; a harness
capture of a steer typed during reasoning (`think high` on a question,
then a steer) showing the restart notice.

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

**Acceptance.** The three checks of *The lesson of MODL-24* above (decode
after an image equals one prefill on both executors; a large real image
against the reference with teacher-forced agreement ≥ 97 %; the index
each attention kernel receives named in this section). Traces within tolerance for both entries on both executors;
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

**Acceptance.** The three checks of *The lesson of MODL-24* above, at the
896×896 grid (4,096 patches before the shuffle), with the window
permutation's cache and rotary indices named in this section. Trace within tolerance on both executors; greedy tokens;
the Muse compare targets unchanged; the mask fixture; captions recorded;
`make check`.

## APPS-14 — Teacher-forced `eval` (drafted 2026-09-20 for decision)

**What it is.** `nuclis eval --model <m> --file <text> [--ctx N]` feeds a
fixed text through the model with *teacher forcing*: every position is
fed the reference's actual next token, never a sampled one, and the
command records the model's log-probability of that token from the logits
of each step. The output is the mean negative log-likelihood and its
exponential, the perplexity, over the text (the reference's
`llama-perplexity`), optionally per chunk.

**Why it matters here.** Every numerical check in the tree today is a
pinned trace at two or three positions (the trace gates) or a greedy
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
