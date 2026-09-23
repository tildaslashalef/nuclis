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

MODL-23 closed on 2026-09-23 (two sessions): Muse Glimmer reads images on
both executors and in the chat, so every catalogue family with a working
projector has vision, and the vision theme is done. The windowed encoder
runs its rows in window order; our patches equal the reference's bit for
bit, the residual after block 30 is 1.8e-3 from the reference (its own
CPU/Metal spread there is 1.8e-2), and on the aerial photo as a PNG the
teacher-forced agreement with the reference's answer is 198/200. The
reference disagrees with itself past block 32 (sink patches) and between
its batched and one-row prompt, so the fixture pins the one-row logits
(6.9e-3). A JPEG differs from the reference's pixels (ImageIO against
stb_image), which the sink blocks amplify (193/200). The encode is 47 s at
the full 4,096 tokens (67 s before the windows moved to the chunk
kernel), limited by the global blocks' attention. Folded into the unit:
`generation.image_max_tokens` (`"auto"`, a per-model override,
`--image-max-tokens`) for all three families, and `generate --json`'s
`image_milliseconds`. The `muse-vision-cpu` gate was stopped before its
end at the user's call; the rest of the CPU tier's selection was not run
([log](docs/engineering-log.md#modl-23--muse-glimmers-windowed-vision-encoder-on-both-executors-the-image-token-cap-2026-09-23),
[vision.md § Muse Glimmer's projector](docs/reference/vision.md#muse-glimmers-projector-modl-23-2026-09-23)).
MODL-25 closed the same day: Bonsai 2's projector (Qwen3.8's, re-encoded
Q8_0 with an F16 `ffn_down`) binds through the Qwen3-VL adapter, which now
accepts F32/F16/BF16/Q8_0 matrices, and Bonsai captions the photo; no
gate was run, the user's call
([log](docs/engineering-log.md#modl-25--bonsai-2s-projector-through-the-qwen3-vl-adapter-q8_0-and-f16-weights-2026-09-23)).
**TERM-13 was dropped on 2026-09-23**, the user's call; the harness keeps
`--direct`. **Nothing is ordered next**: APPS-14 stays drafted for
decision.

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
AGNT-16 closed the same day
([log](docs/engineering-log.md#agnt-16--agnt-14s-follow-ups-the-muse-value-contract-pinned-and-measured-steering-that-restarts-a-reasoning-only-step-the-guessed-path-guideline-measured-and-dropped-2026-09-23)):
Muse's values were already verbatim (now pinned, `newfile` passes), a
steer during a reasoning-only step restarts it, and the guessed-path
guideline changed nothing and was reverted. TERM-13 (a stale preview
and header edge after exit, from the user's Ghostty screenshot) was
drafted, then dropped; the harness gained `--direct` for it.
MODL-22 closed the same day, in one session
([log](docs/engineering-log.md#modl-22--gemma-4-vision-the-unified-embedder-12b-and-the-siglip-encoder-26b-a4b-on-both-executors-bidirectional-image-spans-2026-09-23),
[vision.md § Gemma 4's projectors](docs/reference/vision.md#gemma-4s-projectors-modl-22-2026-09-23)):
both Gemma 4 entries read images on both executors and in the chat. On a
3840×2160 photo (1,100 tokens) the teacher-forced agreement with the
reference's answer is 120/121 (12B) and 89/91 (26B-A4B) on our own
projector rows. Every Gemma chat turn after the first re-prefills the
conversation (`replayed`), images included.
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

Order: nothing (the vision units MODL-21, AGNT-15, MODL-22, MODL-23, and
MODL-25 closed; TERM-13 dropped). APPS-14
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
