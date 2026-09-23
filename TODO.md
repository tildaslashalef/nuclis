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
AGNT-16 closed the same day
([log](docs/engineering-log.md#agnt-16--agnt-14s-follow-ups-the-muse-value-contract-pinned-and-measured-steering-that-restarts-a-reasoning-only-step-the-guessed-path-guideline-measured-and-dropped-2026-09-23)):
Muse's values were already verbatim (now pinned, `newfile` passes), a
steer during a reasoning-only step restarts it, and the guessed-path
guideline changed nothing and was reverted. **TERM-13** (a stale preview
and header edge after exit, from the user's Ghostty screenshot) is
drafted and deferred by the user; the harness gained `--direct` for it.
MODL-22 closed the same day, in one session
([log](docs/engineering-log.md#modl-22--gemma-4-vision-the-unified-embedder-12b-and-the-siglip-encoder-26b-a4b-on-both-executors-bidirectional-image-spans-2026-09-23),
[vision.md § Gemma 4's projectors](docs/reference/vision.md#gemma-4s-projectors-modl-22-2026-09-23)):
both Gemma 4 entries read images on both executors and in the chat. On a
3840×2160 photo (1,100 tokens) the teacher-forced agreement with the
reference's answer is 120/121 (12B) and 89/91 (26B-A4B) on our own
projector rows. **Two facts for MODL-23:** the chunk attention's `span`
is one contiguous bidirectional range per dispatch, and the SigLIP
attention at 9,900 rows of 72-wide heads runs under 1 TFLOP/s (13.7 of a
17.1 s encode). Every Gemma chat turn after the first re-prefills the
conversation (`replayed`), images included.
**Next: MODL-23 session 2**, the implementation. Session 1 (the facts,
2026-09-23) rewrote its section below, pinned the synthetic fixture, and
left the aerial photo's oracle under `.zig-cache/vision/muse/photo-metal/`.
Three facts change the earlier sketch: the cap is 4,096 tokens (16,384
patches, not the 896² warm-up), the pixel shuffle interleaves channels
(`c·4 + s`), and the reference's own CPU and Metal encoders part at block
33, where the network turns one or two patches into sinks, so rows are
compared tightly only through block 32.
Folded in by the user on 2026-09-23: `generation.image_max_tokens`
(`"auto"` = each projector's maximum; a per-model override and
`--image-max-tokens`), for all three families; the section's design says
where. If the session runs long, the setting is the part to land first.
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

Order: MODL-23, TERM-13 when the user takes it up (MODL-21, AGNT-15, and
MODL-22 closed first, so the chat has its image path and two families'
projectors; Muse's follows). APPS-14
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
| MODL-23 | Muse Glimmer's windowed vision encoder (session 1, the facts, done) | 2 |
| APPS-14 | Teacher-forced `eval` (drafted for decision; see its section) | — |
| TERM-13 | Exiting after an image preview leaves the picture and the header's top edge on screen (reported 2026-09-23, deferred by the user; see its section) | 1 |

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
Gemma 4 `<|image>` + `count × <|image|>` + `<image|>` (ids 255999,
258880, 258882; no newlines, MODL-22); Muse `<|image_start|>` +
`count × <|patch|>` + `<|image_end|>` (ids 200080, 200092, 200081; no
newlines, MODL-23's facts). `Engine.encode` stays text-only; after
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
`gemma4.cpp:23-25`); MODL-22 added the chunk attention's `span`.
Muse's language model is causal with plain positions over an image; its
weightless input norm applies to the feature rows too (MODL-23's facts).

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
| Muse Glimmer 30B | `mmproj-kquant.gguf` (1.4 GB; Q4_K 200, Q6_K 100, BF16 3, F32 506) | `muse-glimmer` | 50 blocks, 1536 wide, FFN 8960, 16 heads, 2D RoPE base 1e4, windowed attention (every 4th and the last layer global), pixel-shuffle ×2, adapter 6144→4096→4096 erf GELU, projection 4096→6656 | 14 | 2 | 6656 | image_size 896 is the warm-up only; grid by aspect-preserving search under 4,096 tokens (16,384 patches), Lanczos stretch |

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

## MODL-23 — Muse Glimmer's windowed vision encoder

**Facts (read 2026-09-23 from the pinned `7620399f5` and the file).**

*The file.* `mmproj-kquant.gguf` (1.40 GB), `clip.projector_type =
muse-glimmer`, mean/std 0.5, LayerNorm eps **1e-5**
(`clip.vision.attention.layer_norm_epsilon`), `image_size` 896 (the
warm-up only), patch 14, merge 2 (`clip.vision.spatial_merge_size`).
50 blocks × 16 tensors: `ln1`/`ln2` weight + bias F32 [1536];
`attn_q`/`attn_k`/`attn_out` **Q4_K** and `attn_v` **Q6_K** [1536, 1536],
each with an F32 bias; `ffn_up` Q4_K [1536, 8960] + bias, `ffn_down` Q6_K
[8960, 1536] + bias, **no gate**. `v.pre_ln`/`v.post_ln` weight + bias;
`v.patch_embd.weight` F32 [14, 14, 3, 1536], **no patch bias**;
`v.position_embd.weight` F32 [1536, 1024] (a 32×32 grid); `mm.0` BF16
[6144, 4096], `mm.1` BF16 [4096, 4096], `mm.2` BF16 [4096, 6656], no
biases. `muse_glimmer_patch_temporal = 2` is set in `clip.cpp:1688` and
read nowhere (the kernel is already 2-D).

*Preprocessing* (`mtmd-image.cpp:1682-1739`). `max_tokens =
image_max_pixels / (14·14·2·2)` with `set_limit_image_tokens(1, 4096)`:
**up to 4,096 tokens, 16,384 patches** (not the 896² warm-up). The grid is
transformers' `get_aspect_ratio_preserving_size` on 28-pixel units: scale
`(h/28, w/28)` down to `max_tokens` keeping the ratio, try floor/ceil of
each side, keep the pair under the cap closest to `h/w` (ties to more
tokens; none fits → round and clamp at 1). Then a **stretch** resize to
`(28·w, 28·h)` with Pillow's **Lanczos** (support 3, `sinc(x)·sinc(x/3)`,
the same 22-bit separable resampler as our bicubic), no padding;
`(v/255 − 0.5)/0.5` on the host; the im2col rounds the patch values to
**F16** (`node_0 [588, …] f16`), channel-planar (588 = 3·14·14), raster.
The synthetic 96×64 → 84×56 px, 6×4 patches, **3×2 tokens**; the aerial
photo 3840×2160 → 2380×1344, 170×96 patches, **85×48 = 4,080 tokens**.

*The graph* (`models/muse-glimmer.cpp`, `clip.cpp` `build_vit` and the
`set_input` branch at 4582–4645). Per patch: conv (no bias) + the position
table **bilinearly resized half-pixel** (`pixel_offset 0.5`, edges clamped,
no antialias; not Qwen's aligned corners; returned as is when the grid is
32×32), raster order. Then `sp_perm` groups the patches into **32×32-patch
windows** (the table's side), windows in raster order and patches raster
within each (edge windows partial), and every block runs in that order:
`pre_ln` (LayerNorm + bias), then 50 × { `h = LN(x)·ln1 + b`; q/k/v =
W·h + bias; **2-D RoPE, GGUF normal (adjacent pairs)**, base 1e4: channels
[0,48) pairs (2i, 2i+1) turn by `pos_w·10000^(−i/24)`, channels [48,96)
pairs (48+2i, 49+2i) by `pos_h·10000^(−i/24)`, i < 24, positions
**1-indexed** from the patch's original grid cell; attention scale
1/√96 over the **window's rows** on sparse layers and **all rows** on the
13 global ones (`il = 49` or `(il + 1) % 4 == 0`: 3, 7, …, 47, 49);
`x += W_o·attn + b`; `x += W_down·gelu_erf(W_up·LN(x)·ln2 + b) + b` }
(exact **erf** GELU, not our tanh form); `post_ln`; `inv_perm` back to
raster. The **pixel shuffle is interleaved**: merged token o = (oy, ox)
has element **`c·4 + s`** = channel c of patch (2oy + ry, 2ox + rx),
s = 2ry + rx (checked against the trace: exact, while Qwen's
concatenation `s·1536 + c` is off by 7.4). Adapter: `mm.0` 6144→4096,
erf GELU, `mm.1` 4096→4096, erf GELU, `mm.2` 4096→6656.

*The language model.* The reference writes `<|image_start|>` (200080) +
the rows + `<|image_end|>` (200081) where the media marker was, no
newlines; the GGUF template's own image part (a single `<|patch|>`,
200092) is not what the reference renders. Causal (`mtmd_decode_use_non_causal`
is false), plain positions (`MTMD_POS_TYPE_NORMAL`: the synthetic span sat
at 43–48): the cache row, the rotary position, and the visible count stay
one value. `build_inp_embd` feeds the image rows through the same
weightless **`embd_norm`** as token rows (`src/models/muse-glimmer.cpp:73-74`),
so our feature rows go into `x_c` *before* that RMS norm.

*The oracle.* `scripts/reference-vision.cpp` gains `--n-ctx N` (the photo
needs 4,130 positions) and `--vision-flash` (the projector's attention as
the reference's flash attention: without it a global layer over 16K
patches materializes a 17 GB score matrix); rebuild as in
[vision.md § Gemma 4's projectors](docs/reference/vision.md#gemma-4s-projectors-modl-22-2026-09-23).
Prompt (our rendering, thinking off → `low`):
`<|begin_of_text|><|start|>system<|message|>You are a helpful AI assistant.\nKnowledge cutoff: 2026-01-04.\n\nReasoning strength: low.\n\n# Valid recipients: "self", "user".<|eot|><|start|>user<|message|><__media__>describe this image<|eot|><|start|>assistant`.
`inference/src/vision/fixtures/muse-glimmer-synthetic/` holds the
reference's Metal rows (`features.f32`, 6 × 6656), `prompt-tokens.json`
(43 text tokens ending 200080, the 6-row span at 43–48, then `200081 30402
544 3371 200008 200022 140680`), the 16 best last logits (328 at 18.89,
next −5.05: the first token barely depends on the image), 8 greedy tokens
`328 19669 200023 30402 544 3371 368 954`, and `layer-out-30.f32` (24 ×
1536, window order = raster here). `fixtures/muse-glimmer-mmproj.json` is
the file's inventory. The photo's oracle directory is
`.zig-cache/vision/muse/photo-metal/` (`--n-ctx 8192 --vision-flash`,
200 greedy tokens; the answer names Lake Tahoe, turquoise water, granite
boulders, snow-capped mountains, a pine on the shore; 116 s end to end,
22 GB resident); not committed.

*The reference disagrees with itself past layer 32.* Its CPU and Metal
encoders on the synthetic fixture (relative RMS of the difference):
`pre_ln` 7.6e-4, block 0 1.4e-2, 3 1.2e-2, 10 1.0e-2, 20 1.3e-2, 30
1.8e-2, 32 7.6e-2, **33 0.63**, 49 0.18, `encoder_out` 0.44, rows 0.26 —
yet the last logits agree to 3.8e-3 and all 12 greedy tokens match. At
block 33 the FFN makes one or two patches **sinks** on channel 1082
(Metal: patches 0 and 17 at −420/−396; CPU: patch 0 alone at −910), and
which patches become sinks flips with rounding (the CPU's Q8_K
activations against the Metal's F16). So rows can be compared tightly only
through block 32; past it, the check is the language model's output.

**Design.**
- `preprocess.zig`: the resampler's filter becomes a parameter
  (`bicubic`, `lanczos`); `resizeLanczos` beside `resizeBicubic`.
- `vision/muse_glimmer.zig`: the constants above; `bind` (the types
  listed; any other encoding is refused); `gridFor(size, max_tokens)` (the
  search, with a caller's cap for the checks); `Layout` built once per
  grid on the host: `order` (window order → raster patch), `windows`
  (`{ begin, count }` in window order), and the pixel-shuffle gather;
  `positionRows` (half-pixel bilinear); `ropeTable` (48 pairs per row:
  pair j < 24 turns by `pos_w·10000^(−j/24)`, pair j ≥ 24 by
  `pos_h·10000^(−(j−24)/24)`, `Pairing.adjacent`); `Runtime`, the CPU
  reference in F64 accumulation, rows in window order throughout.
- `vision/muse_glimmer_metal.zig`: `Plan`. The host writes the patch rows
  and the position rows **already in window order** (the conv and the
  table add are per row, so permuting their inputs replaces `sp_perm`);
  the patch matmul (F32 kernel, F16-rounded values); 50 blocks of the
  existing `layerNorm`, K-quant `matmul` + `addBiasRows` (q, k, v, out,
  up, down), `ropeRows(.adjacent)`, attention (sparse: `attentionFull`
  once per window on slices, ≤ 1,024 rows, width 96; global:
  `attentionChunk` with `span = { 0, n }` over all rows, as Gemma's
  SigLIP), and a new erf `geluErf` kernel (with `cpu.geluErf`);
  `post_ln`; `inv_perm` + the interleaved shuffle as one host gather
  between two command buffers (as Gemma's pool; a kernel only if it
  measures slow); the BF16 adapter. Buffers sized for 16,384 patches:
  x, h, q, k, v, attention 101 MB each, up 587 MB (≈ 1.2 GB with the
  weights' 1.40 GB, an estimate).
- `vision/root.zig`: `Projector` gains `.muse_glimmer`; `loadVision`
  needs no row reservation (causal spans chunk like Qwen's).
- Profile: `image_placeholder = "<|patch|>"`; a user message with images
  renders `<|image_start|>` + `count × <|patch|>` + `<|image_end|>` before
  its text (as Gemma's), and the three markers join the rejected content
  markers.
- `muse_glimmer_metal.zig` / `muse_glimmer_runtime.zig` gain
  `prefillVision`: text runs and spans chunk separately (Qwen's causal
  path); a span's rows copy the features into `x_c` in place of `embed`,
  then the same weightless `rmsNorm` runs over every row.
- **The image token cap as a setting** (the user's call, 2026-09-23:
  folded into this unit, default `auto`). `generation.image_max_tokens`
  in `src/config.zig`: a new leaf type `ImageMaxTokens = union(enum) {
  auto, tokens: usize }` (JSON `"auto"` or an integer), so `describe`,
  `parseValue`, and `formatLeaf` gain a union case; global default
  `.auto`, validated 1..4,096 (the largest family maximum); the entry's
  `ModelEntry.Generation.image_max_tokens: ?ImageMaxTokens = null`
  overrides it, and `--image-max-tokens auto|N` on `generate` and `agent`
  (`src/cli.zig`, beside `--draft-length`) overrides both; `resolve` and
  its origin marks (`leafIndex`, `.model` / `.flag`) as for
  `draft_length`; `config init` writes `"auto"` into the global block and
  nothing per entry. `Projector` (`inference/src/vision/projector.zig`)
  gains `max_tokens`, set after `loadVision` from the resolved value:
  `auto` is the family's maximum (Qwen 1,024, Gemma 1,120, Muse 4,096), a
  number is clamped to `[min_tokens, maximum]` (Gemma's 70 floor
  included); `qwen3vl.gridFor`, `gemma4.gridFor`, and
  `muse_glimmer.gridFor` take the cap; a bidirectional family reserves
  rows for the effective cap, not the maximum. A resumed session
  re-encodes at its **recorded grid** (the cap may have changed since),
  refused only above the family maximum. Callers: `src/generate.zig:202`
  and `src/agent/root.zig:551`. Documented in `docs/spec.md`'s
  configuration keys, the `generate`/`agent` help pages, and vision.md.
- **Indices each kernel receives** (lesson 3). Encoder: the row index is
  the window-order index r, used for every matmul, norm, and attention
  row; the rotary position of row r is `(pos_w, pos_h)` = `(o % grid_w + 1,
  o / grid_w + 1)` with `o = order[r]`, never r; a sparse layer's keys are
  its window's `[begin, begin + count)`, a global layer's `[0, n)`. The
  gather reads row `order⁻¹[patch]`. Language model: cache row = rotary
  position = visible count = `state.position` + row, as for text.
- Speculation: `images_fed` (`src/agent/loop.zig`) already keeps it off
  after an image until `reset`, so Muse (on by default) decodes an image
  conversation without its drafter; keep that, record it in vision.md.
  Feeding the image prefill's hidden rows to the drafter is a later unit.

**Acceptance.**
1. The synthetic fixture on both executors: prompt tokens equal to the
   oracle's; `layer-out-30` within the reference's own spread there
   (≤ 2e-2 relative RMS); the rows' relative RMS reported, not gated
   (the sinks); the 16 top logits and the 8 greedy tokens on the pinned
   rows and on our own rows.
2. Decode after an image equals one prefill (bound 2e-2) on both
   executors, the planted fault caught once.
3. The aerial photo at 4,080 tokens (16,320 patches, 18 windows, the last
   column 10 patches wide) against `photo-metal`: teacher-forced agreement
   with the reference's 200-token answer ≥ 97 %, or explained by margins
   under 0.3; the rows' relative RMS and the encode time recorded.
4. Host-logic tests: `gridFor` against the reference's choices (the two
   fixtures, a tall image, one over the cap), `Layout` on a 66×34-patch
   grid (partial windows) against a hand-computed permutation, Lanczos
   against a Pillow-computed fixture; the window attention on the kernel
   with a poisoned row outside the window.
5. The cap: config tests (`"auto"`, an integer, `0` and a string other
   than `auto` refused with the key's message, the model override and the
   flag winning in that order, `config show` origins); grid tests at a
   lowered cap per family (the photo at `--image-max-tokens 1024` on Muse
   is 1,008 tokens); a resumed session keeps its recorded grid after the
   cap changes.
6. The Muse text gates unchanged (`make verify`), `make check`, a caption
   in the chat through the harness.

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

## TERM-13 — Exiting after an image preview leaves the picture and the header's top edge on screen (reported 2026-09-23, deferred)

**The report.** The user's screenshot (plain Ghostty 1.3.1, no tmux, a
session of about ten minutes with the photo previewed): after exit the
screen holds the shell prompt at the top, a single box top edge on row 1
(the banner's, by its indent and width), the photo still placed around rows
15–26, and nothing else. The transcript is gone from the screen and the
image outlived it.

**What is known (2026-09-23).** Not reproduced on a short run. The tmux
harness (preview forced on with `env -u TMUX TERM_PROGRAM=ghostty`): drop,
one answer, Ctrl-C leaves the transcript intact (`.zig-cache/tui/x-exit.txt`).
In a direct Ghostty window (`scripts/tui-shot.py --direct`, added for this
unit; see [development.md § Looking at the agent without a person at the
keyboard](docs/development.md#looking-at-the-agent-without-a-person-at-the-keyboard))
the same run's recorded stream ends with `bye` on row 34 and `finish`
walking 18 rows up from row 53 before `ESC[0J`, so only the region is
erased. The photographs failed (`screencapture`: no screen-recording
permission for the terminal running the agent session). The user's
session-level details (several turns, a resize, exiting while busy, a
fold toggle) are not yet known.

**Suspects, in order.**
1. *Ghostty does not move kitty placements when a partial scroll region
   scrolls.* `Screen.insertAbove` scrolls with `DECSTBM` (top 1, bottom
   above the region); if Ghostty only carries placements (and creates
   scrollback) on a full-screen scroll, the picture stays on its rows while
   the text moves past it, and later turns write over and around it.
2. *The exit walks too far.* `Screen.finish` goes up `cursor_row + slack`;
   the screenshot's cursor landed on row 2, i.e. the screen believed nearly
   every row above the region was slack or region. Candidates: `resized`
   adding slack from a cursor report, `rewriteAbove` (a replay after a
   resize or a fold) re-sending the preview's raw row with its relative
   `CSI n A`/`CSI n B`, or a busy region filling the screen at exit.
3. *Erasing never deletes a placement.* `ESC[0J` leaves kitty images in
   place, so any erase over a preview (exit, a replay, a resize) strands it.

**Session 1.** Get the screen-recording permission (or the user's
photograph) and reproduce with `--direct`: several turns after the image
until the transcript scrolls past it, then a fold toggle, a resize, and an
exit while busy, a capture after each; read `<name>.stream` for the escape
sequence of each step. Fix what reproduces; the likely shape is to give
each preview an image id, delete the placements whose rows an erase or
rewrite covers (`a=d` by id, or `d=y` per row), and, if suspect 1 holds,
not rely on a scroll region to carry a picture (scroll the whole screen
while a preview is on it, or re-place it). **Check:** the `--direct`
captures of each step, the golden escape tests of `finish` and the replay
with a preview row, `make check`.
