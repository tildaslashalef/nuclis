<!-- kind: guide -->
# The gates

How the model-specific checks are defined, selected, remembered, and run.
Which gates exist, by tier, and when to run each tier is
[development.md § Gates](../development.md#gates).

## The registry

Every model-specific numerical check is a *gate* in [`gates.json`](../../gates.json),
run by `nuclis-gates` (`tools/gates`); bounds, model paths, and evidence anchors live
there and nowhere else. A gate is one command (an argv with the
placeholders `{nuclis}`, `{nuclis-cpu}`, `{zig-metal}`, `{zig-cpu}`,
`{model}`, `{mtp}`, `{baseline}`, `{mmproj}`, `{entry}`, `{trace}`) and a comparator: `exit` (the command's
status decides: the generation checks, the speculative checks, the Gemma
draft check that compares its own rows, `draft-stats`, the
vocabulary checks, the perplexity checks, the cross-encoding check against
its `baseline` file) or `trace` (the `{trace}` directory goes through
`nuclis-gates compare` against the gate's `bounds`). A gate whose
model files (`model`, `mtp`, `baseline`, or the `mmproj` its command names) are not on
this machine is skipped, not failed: `SKIP <gate> no model: <path>`, a
separate count in the summary, `skipped` in `--json`. So is a gate whose
data is not: a gate names the sets it reads in `data`, and the manifest's
`data` registry gives each set's files (`paths`, or every clip `file` of a
`listed_in` manifest under `root`) and the command that fetches it
(`SKIP <gate> no data: speech-corpus (60 of 60 files missing; make
speech-corpus)`). The wikitext text is fetched by the make targets
themselves; the speech corpus (about 1.7 GB of archives) only by `make
speech-corpus`. A skip is not a
pass; `--strict` turns it into a failure, and `make verify-release` (so
every release) runs strict. The tiers (their gates are listed in
[development.md § Gates](../development.md#gates)):

| | tier `verify` | tier `verify-release` | tier `verify-long` | tier `verify-cpu` |
| --- | --- | --- | --- | --- |
| executor | the Metal plan (and the tokenizer) | the Metal plan | the Metal plan | the CPU reference |
| covers | one representative file per family and the paths only a variant has (§ What each gate protects) | whole-file acceptance: the 8-window perplexities of Qwen3.8 and the E4B (`*-perplexity-full`), the F32 trace of the 26B-A4B, the 1.7B speech round trip, Laya multilingual, `qwen38-draft-stats` | positions past 512 and the sliding windows | the CPU reference of every family, projector, and draft source |
| budget | 480 s; the canaries 60 s | 300 s | 120 s (the depth gates restore saved prefixes under `.zig-cache/speed/prefix/`; a missing file costs one prefill) | none |
| build | `ReleaseSafe`, `./zig-out/bin/nuclis` | the same | the same | `ReleaseFast` into `.zig-cache/gates/cpu/` (the reference exists to be exact, not safe, and gives identical numbers faster) |
| when | selected by `make verify-auto`; all of it before a release | once before a release | when a unit changes attention, the KV cache, or a windowed schedule (what only positions past 512 and past the 1,024/2,048-token windows exercise), and once before a release | when a unit changes what the CPU reference computes (an existing CPU kernel's or decoder's arithmetic, a family's `*_runtime.zig` forward, a projector's CPU `Runtime`), when a family or a draft source is brought up, to tell a wrong kernel from wrong model semantics after a Metal trace fails, and once before a release; not for additions nothing calls, refactors a unit test pins, the check tool, or Metal code |

The fast tier trims what its gates repeat, not what they cover: a Metal
draft trace runs the plan alone (its CPU half is the `*-draft-trace-cpu`
gate), `checkChunkedPrefill` takes its F32-tile and `prefillRows`
comparisons over 40 tokens (chunk 32 still splits), and the Qwen and E4B
`*-perplexity` gates run `--chunks 2` against the reference's running
value after the second window (`chunk_ppl`; −0.005 %, −0.019 %
at `f5982c4`), the whole run being `*-perplexity-full`. The 26B-A4B runs
only the whole run, in `verify` (13 s): its running difference swings
from −4.1 % after the first window to −1.2 % after the fourth before
settling at −0.53 %, so its 1 % bound holds only for the whole run (two
windows read −0.850 % at `79fe2c9`, inside the bound by luck of the
prompt, not by design).

**What a tier costs.** `nuclis-gates --costs` (`make gates-costs`) sums
each tier's and the canaries' last passing seconds from
`$NUCLIS_HOME/gates/times.json` against the manifest's `budgets`; a gate
never timed on this machine counts as zero and shows in the `timed`
column, so a total is a floor. A tier run that passes its budget warns
on stderr, and fails under `--strict`; a gate that pushes `verify` over
moves to `verify-release` or gets cheaper, it does not raise the budget.
`make gates-list` prints each gate's seconds; the last measured costs are in [the record](../benchmarks/record/development.md#gate-costs-2026-11-13).

**Models are index files.** A `gates.json` model names an `artifact` of
the catalogue's index and, optionally, a `file`: `main` (the default),
`mmproj`, `draft` (the artifact's default draft), `draft:<variant>`, or
`directory` (a safetensors checkpoint's folder). The runner reads the
index from `src/catalog.json`, which `make index` writes from
`src/catalog.zig` and a unit test holds equal to it (the runner imports
none of nuclis, and CI's docs job has no Zig). A literal `path` is
allowed only with a `note` saying why the file is not in the index. The
manifest is invalid when a model names a legacy artifact, or when a
current chat artifact has no gate. The manifest's `retired` list names
fixture prefixes under `tests/fixtures/` that must not come back; the
tree lint fails on any tracked file under one.

**Three lanes.** Always: `make check`, on every commit, and `zig build
test` in CI. Focus: `make verify-auto`, before a pull request, when the
engine changed. Release: `make verify-release` and the other tiers, strict,
before a tag. No scheduled job runs them; `make sweep` runs every gate not
already remembered, whenever the Mac is free, and those passes count
towards the release.

**`make verify-auto`** diffs from where the branch left `origin/main` (or
`BASE=<rev>`). It runs the model-free `checks` the changed paths select,
then the selected `verify` gates cheapest first, one model at a time (the
pinned files together outgrow the 48 GB, so interleaving models reloads
each from disk: the same 27 gates took 298 s interleaved, 220 s grouped)
until one fails (`ARGS=--keep-going` runs the rest), and ends by naming the
tiers the manifest's `requires` rules ask for. Those rules name files, not
intent: an addition nothing calls yet or a refactor a unit test pins still
skips the CPU tier, and the long rule's line asks for a judgment of the
diff. `ARGS=--dry-run` prints the plan, and says why each gate was chosen.

**Selection.** `tools/gates` (`nuclis-gates`) fingerprints each engine file:
a Zig file by its tokens, so comments and spacing do not count, anything
else by its bytes. The root `build.zig` is left out on purpose. It walks
`@import` and `@embedFile` forward from each of the manifest's `families`
seeds, stopping at `aggregators`. A changed file reached by one or a few
families selects those families' gates; a file no family reaches alone, or
more than `family_cutoff` reach, selects the `canaries`, 17 cheap gates
that cover the shared plans within their 60 s budget. Other tiers are not run unless
asked: `make verify-changed BASE=<rev> ARGS='--tier verify-long'`.

**Remembered passes.** A pass is keyed by the gate's definition, the
fingerprints of every engine file except those exclusive to other families,
the build files (`build.zig`, `build.zig.zon`, `inference/build.zig.zon`),
the application (`src/`, `huggingface/`) when the command runs `{nuclis}`
or `{nuclis-cpu}`, the folder `scripts/gate/` when the command runs one of
its scripts, each model file's path, size, and modification time, the
manifest's `build` block, and `zig version`. Passes live in `$NUCLIS_HOME/gates/`, so they outlast a cleared
`.zig-cache` and are shared by worktrees. A skip is not a pass.
`ARGS=--no-cache` runs a gate anyway.

The other commands: `make verify` / `make verify-release` / `make
verify-long` / `make verify-cpu` (a tier), `make sweep`, `make gate`
with `NAME=` a gate's name or a glob (`make gate NAME=gemma4-e4b-trace-f16`,
`make gate NAME='qwen38-*'`; `ARGS=--dry-run` prints the commands), `make gates-list`, and `make gates-validate` (part of
`make check`: the manifest's schema, no model).
`nuclis-gates --json` returns the results with the git
revision for a log entry, each with its output lines stamped by time
since launch (`--timeline` prints them); the check tools behind
`{zig-*}` are compiled first (`zig build check-tools`), so the builds'
seconds are reported apart from the gates'. Gate names are `<entry>-<check>-<executor>`:
`qwen38`, `gemma4-e4b`, `gemma4-26b-a4b`, `laya`, `clef`, `embeddinggemma`,
`qwen3-tts-0.6b`, `drama3`, `nemotron-asr`;
`trace-{cpu,f32,f16}`, `generation-{cpu,metal}`, `speculative-{cpu,metal}`,
`draft-trace-{cpu,metal}`, `draft-stats`, `vocabulary`, `perplexity` (two
windows), `perplexity-full`, `perplexity-4k`. A
model path is overridden per key by `<KEY>_MODEL` (`GEMMA4_E4B_MODEL=…`),
except in a command that names `{entry}`, the model's catalogue name: that
gate runs the pinned files the catalogue resolves (an audio or vision gate
through `nuclis generate`, which finds the projector by entry). A `trace`
comparator names the model's shapes (`embedding`, and `layers` and `vocab`
outside `draft` mode); nothing defaults them.
Traces are written under `.zig-cache/gates/trace/<gate>/`. The perplexity
gates read wikitext-2-raw's test text, fetched into `.reference/eval/` and
checked against its SHA-256 by `make eval-corpus`, which `verify`,
`verify-changed`, and `gate` run first (a no-op once the file is there);
their references are pinned under `tests/fixtures/perplexity/`
([guide/eval.md](eval.md)).

## What each gate protects

Every check executable aggregates its differences through
`inference/check_stats.zig`, whose maxima and minima latch a NaN or an
infinity instead of dropping it, and fails when it compared nothing (an
unknown `test-tiny` variant, an empty fixture, a filter that matched no
case). `NUCLIS_CHECK_INJECT_NAN=1 make test-metal` poisons one kernel
output and must fail: the self-test of that rule.

What can break independently, and what protects it. *Model-free* means
`make check` (unit tests, `test-metal`, and `test-tiny`, no weights):

| Axis | Model-free | Gates that alone cover it |
| --- | --- | --- |
| matvec, matmul tiles, embed gather, per encoding | every quant fixture encoding (Q4_0, Q8_0, IQ4_NL, Q3–Q6_K, IQ3_S, IQ4_XS, PQ2_0, PTQ1_0, and the codebook fixture's Q2_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS) against the CPU, the CPU decoders against fixture blocks pinned from llama.cpp's decoders during bring-up (the codebook encodings against the cross-encoding check instead), and F32/F16/BF16 matvec and matmul tiles on random weights (`checkDenseEncodings`) | — |
| attention: decode, chunk, windowed, F16 cache | fixtures to 16K keys, and Gemma's geometries (16 heads of 512 over one or two KV heads, 16 of 256 over 8) | — |
| DeltaNet, Hadamard, RoPE, norms, softcap, sampling kernels | fixtures | — |
| Metal `layerNorm`, `addBiasRows` | strided rows against the CPU (`checkVisionNorms`) | — |
| Metal attention over packed sequences, the erf GeGLU (Laya) | packed sequences and windows against F64 (`checkSegmentAttention`), fused rows against the CPU (`checkGeluErfRows`) | `laya-metal` (the plan, its batching, and its cleanup) |
| Laya multilingual: mmBERT's geometry (22 layers of 768, intermediate 1,152), positions 512 to 1,023, the checkpoint's own special tokens | — | `laya-multilingual-cpu`, `laya-multilingual-metal` |
| experts (MoE) | Metal Q4_0 chains; CPU FFN on a 2×2 F32 tensor | `gemma4-26b-a4b-*` |
| Qwen 3.5 plan (DeltaNet, gated attention, MTP block) | the runtime's admission tests | `qwen38-*` |
| Qwen 3.5 with Hadamard rotations, PTQ1_0, BF16 | decoders, kernels | none (Bonsai is legacy in the index) |
| Gemma 4: sliding/global, softcap | — | any Gemma file |
| Gemma 4: shared KV, per-layer embeddings, own V on global layers, window 512 | — | `gemma4-e4b-*` |
| Gemma 4: K=V global layers, window 1024, the `gemma4` profile | — | `gemma4-26b-a4b-*` |
| Gemma 4: one global KV head (16:1), 48 layers | the 16:1 attention geometry | none (the 12B QAT file is legacy in the index; the same code as the 26B's 8:1) |
| Muse Glimmer plan, its tokenizer (GPT-4o splitter) and projector | — | none (legacy in the index) |
| session layout: recurrent with the verify tape (Metal, drafter) | session unit tests | `qwen38-generation-metal` |
| session layout: recurrent without a tape (rewind, replay) | session unit tests | every CPU generation gate |
| session layout: attention only (truncate) | session unit tests | any Gemma generation gate |
| engine loop: EOS, budget, context, cancellation, speculative loop, penalties, top-k readback | none (`engine.zig`'s one test covers companion selection) | `qwen38-speculative-metal` |
| draft sources: Qwen MTP and DFlash, Gemma assistant | — | `qwen38-draft-trace-*`, `qwen38-dflash2-*`, `gemma4-e4b-draft-trace-*` |
| tokenizers: Qwen BPE, Gemma SPM, Laya byte-level and Metaspace | the Metaspace rules and rejections on a synthetic vocabulary | one vocabulary gate each |
| vision: Qwen3-VL, SigLIP small with clamps, SigLIP large with standardization | preprocessing, grids | one vision gate each |
| batched prefill at 256 rows, F16 cache over 511 keys, real prose | — | the perplexity gates |
| a verify batch at depth (few-query attention, the multi-row weight path) against stepped decode from the same state | `checkAttentionVerify` against F64 to the 32K cache | `qwen38-verify-depth-{512,4k}`; `-16k`, `-32k` in `verify-long` |
| past 512 tokens and the sliding windows | — | `verify-long` |
| speech, audio, and embeddings: Qwen3-TTS, Drama 3, Nemotron ASR, EmbeddingGemma 2, Gemma 4 audio | — | their own gates (`qwen3-tts-*`, `drama3-*`, `nemotron-asr-*`, `embeddinggemma-*`, `gemma4-e4b-audio-*`) |

Not covered by any gate: the 26B-A4B's assistant head, and the PQ2_0 encoding (decoder and kernel
fixtures only). On Metal a speculative greedy divergence is printed, not
failed (chunk-versus-step rounding may flip a token); the CPU run fails
on it. `qwen38-draft-stats` reports acceptance rates and fails only on
an error.

Adding a family adds its gates to the manifest and nothing to the
Makefile ([models](../models/README.md)). The
manifest's numbers are the accepted tolerances the reference documents
justify (Qwen's F16 bound in [metal-backend.md](../engine/metal-backend.md),
Gemma's in [gemma4.md](../models/gemma4.md)); the CPU rows and the F32 cache run at
the bring-up thresholds (max abs 2e-3, relative RMS 1e-4).
