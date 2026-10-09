# Development guide

Working on nuclis: the toolchain, the build and test commands, the gates,
the record, CI, and releases. Installing and using nuclis is in
[guide/getting-started.md](guide/getting-started.md); product behavior and
acceptance criteria belong in [spec.md](spec.md); agent-specific working
instructions belong in [../AGENTS.md](../AGENTS.md).

- [Current state and layout](#current-state-and-layout)
- [Environment](#environment)
- [Gates](#gates)
- [The record](#the-record)
- [The speed loop](#the-speed-loop)
- [Toolchain](#toolchain)
- [Working on the agent](#working-on-the-agent)
- [The website](#the-website)
- [Continuous integration and releases](#continuous-integration-and-releases)
- [Build and format contract](#build-and-format-contract)
- [Versioning](#versioning)
- [Commits, progress, and code explanations](#commits-progress-and-code-explanations)
- [Testing](#testing)
- [Measurements and artifacts](#measurements-and-artifacts)
- [Reference implementations and third-party material](#reference-implementations-and-third-party-material)
- [Documentation conventions](#documentation-conventions)

## Current state and layout

Every command in `nuclis --help` works: `agent`, `generate`, `bench`,
`tokenize`, `eval`, `inspect`, `validate`, `model`, `cache`, `decide`,
`serve`, `config`, and `completion`. The engine runs on the CPU reference
and, built with `-Dmetal=true`, on the Metal backend. [../TODO.md](../TODO.md) says what is in progress, the
[merged pull requests](https://github.com/tildaslashalef/nuclis/pulls?q=is%3Apr+is%3Amerged) what changed and why, and
[worklog.md](https://github.com/tildaslashalef/nuclis/blob/v0.6.0/docs/worklog.md) the work before v0.6.0.

```text
src/                 executable: CLI, model commands, config
  src/tui/           terminal surface, engine-free: screen, editor, theme,
                     markdown, keys
  src/completer.zig  the completion side of a conversation: render, continue
                     or restore the session, complete (src/completer/cache.zig)
  src/agent/         agent composition: engine, conversation, sessions, tools
  src/decision/      the decision request and response JSON, model names
  src/api/           nuclis serve: HTTP transport, router, GPU executor,
                     services (decisions/, chat/)
inference/           engine library: runtime, quantization, tokenizer,
                     sampling, backends, profiles
huggingface/         Hub download library (Xet) and its standalone binary;
                     imported by src/ only
docs/                the hub README.md: guide/, models/, engine/, app/, benchmarks/
site/                the nuclis.dev website: static HTML, CSS, JS, no build
TODO.md              active plan: unfinished units only
build.zig            build
build.zig.zon        manifest; single source of the version
```

The executable composes the `inference` and `huggingface` modules through
the root build. Keep kernels with the inference library that owns them.

## Environment

Facts every unit depends on; keep them here, not in `TODO.md`.

- Zig 0.17.0 (on the author's machine under `~/.local/opt/zig/stable`, managed
  by `zigup`; any install of that version works); consult its installed std source for API
  details. Apple M4 Pro, 48 GiB, macOS 27. Metal compiles
  shaders at runtime from the Command Line Tools; Xcode-only tools are reached
  per process (see [../AGENTS.md § Local toolchain notes](../AGENTS.md#local-toolchain-notes)).
- Model `~/.nuclis/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf`
  (catalogue name `qwen3.8-27b`, the default `engine.model`), SHA-256
  `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482`,
  from [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
  at commit `4ca720788d1e01f1bff70c033e0d0028fd02e502`, pulled from a
  clean state by `nuclis model pull qwen3.8-27b --all` on 2026-09-11 ([MODL-03](https://github.com/tildaslashalef/nuclis/blob/v0.6.0/docs/worklog.md#modl-03--models-directory-catalogue-and-registry-2026-09-11)).
  Its companions `mmproj-BF16.gguf` and `MTP/mtp-Qwen3.8-27B-Q4_0.gguf`
  sit beside it, verified and not executed
  ([models/catalogue.md](models/catalogue.md#pinned-commits-and-digests-2026-09-11),
  [engine/gguf.md](engine/gguf.md#companion-files-in-modelsqwen-2026-09-08)).
- Two ignored directories hold local state. `.zig-cache/` is disposable:
  Zig's build cache (`make clean-cache` drops it; it grows with every
  build), gate traces and results, the generated playground; `make
  distclean` or `rm -rf` loses nothing that does not come back by itself.
  `.reference/` is durable: the oracles' checkouts, builds, and venvs, their
  staged models, and pinned downloads (`eval/`, `ucd/`), all slow to set up
  again; nothing in the Makefile deletes it.
- Reference llama.cpp `7620399` builds on demand under
  `.reference/llama.cpp`; the
  PrismML fork `5d80cff` (release `prism-b10687-5d80cff`), the only decoder
  of Bonsai 2 27B's ternary encodings, beside it under
  `.reference/prism-llama.cpp` with the same recipe
  ([llama-cpp.md § The second oracle](benchmarks/llama-cpp.md#the-second-oracle-the-prismml-fork-2026-09-18)),
  and mainline `06cad0b9e`, the first that runs EmbeddingGemma 2, under
  `.reference/llama.cpp-embed`
  ([llama-cpp.md § The third oracle](benchmarks/llama-cpp.md#the-third-oracle-embeddinggemma-2-2026-10-09)).
  EmbeddingGemma 2's semantic oracle is sentence-transformers 6.1 in a venv
  at `.reference/venv-embed` (the recipe heads
  `scripts/embedding-reference.py`), with Google's checkpoint in float32.
  The decision model's oracle is the `laya` 0.3.20 Python package in a venv
  at `.reference/laya-venv` (Python 3.12 through `uv`; the recipe
  heads `scripts/laya-reference.py`; `--subfolder multilingual` for the
  multilingual set), run on the CPU in F32 against the pulled checkpoint
  from a staged copy, because the package may rewrite
  `tokenizer_config.json` in place.
  clef-flash's oracle is Cloudflare's `joint_schema_model.py` at the
  pinned commit in a venv at `.reference/clef-venv` (torch 2.11.0,
  transformers 5.10.2; the recipe heads `scripts/clef-reference.py`),
  which never loads the backbone: `sequence` mode writes the reference's
  ids and spans, `head` mode runs the head on the rows `clef-check run
  --dump` writes from nuclis's backbone.
  The reference oracle itself is committed under `tests/fixtures/`
  ([provenance](../tests/fixtures/provenance.md)); `make gate NAME='qwen38-trace-*'` reads
  `tests/fixtures/reference-hello-comma`. The rates of both engines on the
  same token arrays are the README's [Results](../README.md#results); the
  method is [benchmarks § Acceptance runs](benchmarks/README.md#acceptance-runs),
  the reference's harness [llama-cpp.md](benchmarks/llama-cpp.md).
- `make help` lists all tasks. Per-commit: `make check` (fmt-check, unit
  tests, `test-metal`, the gate manifest's validation; about 75 s). Per
  unit: `make verify`, the Metal tier of the gate registry
  ([§ Gates](#gates)); `make verify-cpu` only when the unit changes what
  the CPU reference computes, and once before a release (AGENTS.md §
  Validation). `make bench` when performance is claimed; a
  workload of [§ The record](#the-record) (`make workload NAME=…`) when a
  record is claimed. The kernel micro-benchmarks (`bench-kernels`,
  `bench-matmul` with `ARGS=<tokens>`, `bench-matvec-split`,
  `bench-matvec-rows`, `bench-hadamard`, `bench-experts`,
  `bench-attention`, `bench-profile`, `trace`) measure and never gate;
  [benchmarks](benchmarks/README.md) says what each records. `generate
  --prompt-tokens <json>` feeds a token array untokenized, as `bench` does.

## Gates

Every model-specific numerical check is a *gate* in [`gates.json`](../gates.json),
run by `scripts/gates.py`; bounds, model paths, and evidence anchors live
there and nowhere else. A gate is one command (an argv with the
placeholders `{nuclis}`, `{nuclis-cpu}`, `{zig-metal}`, `{zig-cpu}`,
`{model}`, `{mtp}`, `{trace}`) and a comparator: `exit` (the command's
status decides: the generation checks, the speculative checks, the Gemma
and Muse draft checks that compare their own rows, `draft-stats`, the
vocabulary checks, the perplexity checks) or `trace` (the `{trace}` directory goes through
`scripts/compare-generation.py` against the gate's `bounds`). A gate whose
model files (`model`, `mtp`, or the `mmproj` its command names) are not on
this machine is skipped, not failed: `SKIP <gate> no model: <path>`, a
separate count in the summary, `skipped` in `--json`. A skip is not a
pass; `--strict` turns it into a failure, and `make verify-release` (so
every release) runs strict. Two things
make the registry cheaper than the recipes it replaced:

| | tier `verify` | tier `verify-release` | tier `verify-long` | tier `verify-cpu` |
| --- | --- | --- | --- | --- |
| executor | the Metal plan (and the tokenizer) | the Metal plan, and the 12B QAT file's CPU gates | the Metal plan | the CPU reference |
| covers | one representative file per family and the paths only a variant has (§ What each gate protects) | whole-file acceptance: the 8-window perplexities (`*-perplexity-full`), the Gemma 12B QAT file, `qwen38-draft-stats` | positions past 512 and the sliding windows | the CPU reference of every family, projector, and draft source |
| cost | minutes (40 gates in 311 s measured 2026-10-02 with `clef-sequences` and `clef-metal`, [MODL-34](https://github.com/tildaslashalef/nuclis/blob/v0.6.0/docs/worklog.md#modl-34--clef-flash-cloudflares-9b-decision-model-text-and-vision-on-both-backends-2026-10-02-four-planned-sessions-in-one); 38 gates; 258 s measured 2026-09-29 for 36 with the three `laya-multilingual-*`, 254 s for the 33 before them, down from 38 gates in 704 s; worklog, [REPO-20](https://github.com/tildaslashalef/nuclis/blob/v0.6.0/docs/worklog.md#repo-20--fast-verification-gates-re-derived-from-code-paths-a-release-tier-make-verify-auto-2026-09-29), [MODL-31](https://github.com/tildaslashalef/nuclis/blob/v0.6.0/docs/worklog.md#modl-31--laya-on-metal-packed-batches-bidirectional-windowed-attention-over-sequence-bounds-2026-09-29), [MODL-33](https://github.com/tildaslashalef/nuclis/blob/v0.6.0/docs/worklog.md#modl-33--laya-multilingual-the-metaspace-tokenizer-the-checkpoints-own-special-tokens-checked-on-both-backends-2026-09-29); then `qwen38-verify-depth-512` and `-4k`, 3–4 s each from saved prefixes) | minutes of Metal (8 gates, 156 s) and the 12B QAT file's two CPU gates (tens of minutes) | minutes (3 gates: `gemma4-e4b-perplexity-4k` 53 s; `qwen38-verify-depth-16k` and `-32k`, 4–6 s each from the saved prefixes under `.zig-cache/speed/prefix/`, which a missing file costs one prefill: about 3 and 11 min; the other families wait for their references) | 25 min (15 gates, 1,523 s measured 2026-10-02 with `clef-cpu`, 368 s; 14 gates in 1,327 s on 2026-09-30 with the reference's `matvec` on every core, from hours; `muse-vision-cpu` 447 s and `qwen38-speculative-cpu` 308 s the longest; engineering log, [KERN-19](https://github.com/tildaslashalef/nuclis/blob/v0.6.0/docs/worklog.md#kern-19--a-threaded-cpu-reference-bit-identical-the-cpu-tier-in-22-minutes-2026-09-30), [MODL-34](https://github.com/tildaslashalef/nuclis/blob/v0.6.0/docs/worklog.md#modl-34--clef-flash-cloudflares-9b-decision-model-text-and-vision-on-both-backends-2026-10-02-four-planned-sessions-in-one)) |
| build | `ReleaseSafe`, `./zig-out/bin/nuclis` | the same (CPU gates as `verify-cpu`) | the same | `ReleaseFast` into `.zig-cache/gates/cpu/` (the reference exists to be exact, not safe; the Gemma QAT CPU trace measured 29.4 s against 34.7 s at ReleaseSafe with identical numbers, 2026-09-21) |
| when | every unit that touched the inference stack | once before a release | when a unit changes attention, the KV cache, or a windowed schedule (what only positions past 512 and past the 1,024/2,048-token windows exercise), and once before a release | when a unit changes what the CPU reference computes (an existing CPU kernel's or decoder's arithmetic, a family's `*_runtime.zig` forward, a projector's CPU `Runtime`), when a family or a draft source is brought up, to tell a wrong kernel from wrong model semantics after a Metal trace fails, and once before a release; not for additions nothing calls, refactors a unit test pins, the check tool, or Metal code |

The fast tier trims what its gates repeat, not what they cover: a Metal
draft trace runs the plan alone (its CPU half is the `*-draft-trace-cpu`
gate), `checkChunkedPrefill` takes its F32-tile and `prefillRows`
comparisons over 40 tokens (chunk 32 still splits),
`bonsai-generation-metal` passes `--half-tiles-only` (the F32-tile and
F16-cache comparisons check the Qwen 3.5 plan's schedule, which
`qwen38-generation-metal` covers), and the Qwen, Muse, and E4B
`*-perplexity` gates run `--chunks 2` against the reference's running
value after the second window (`chunk_ppl`; −0.005 %, −0.001 %, −0.019 %
at `f5982c4`), the whole run being `*-perplexity-full`. The 26B-A4B keeps
all eight windows: its running difference swings from −4.1 % after the
first to −1.2 % after the fourth before settling at −0.53 %, so its 1 %
bound holds only for the whole run.

**`make verify-auto`** is the one command while working: it diffs from
the `Base:` line of the unit in progress in `TODO.md` (or `BASE=<rev>`),
runs the model-free `checks` of `gates.json` the changed paths select
(`fmt` over the changed Zig files, `unit` = `make test`, `test-metal`,
`manifests`, `python`, `site`), then the matched `verify` gates cheapest first, one model
at a time (the pinned files together outgrow the 48 GB, so interleaving
models reloads each from disk: the same 27 gates took 298 s interleaved,
220 s grouped; models
go in the order of their cheapest gate, each gate by its last measured
seconds, kept in `.zig-cache/gates/times.json` by every run) until one
fails (`ARGS=--keep-going` runs the rest), and ends
by naming the tiers the change requires but that did not run, from the
manifest's `requires` rules, with the file that matched and the rule's
reason (`verify-cpu` for the CPU kernels, decoders, `*_runtime.zig`
forwards, and CPU projectors; `verify-long` for the files that hold
attention, the KV cache, and the windowed schedules). The rules name
files, not intent: an addition nothing calls yet or a refactor a unit
test pins still skips the CPU tier (the table above), and the long rule
matches files that hold more than attention, so its line asks for a
judgment of the diff. `ARGS=--dry-run` prints the plan.

Each gate lists the source globs (`paths`) that make it relevant, so
`make verify-changed BASE=<rev>` runs the Metal-tier gates whose paths
match `git diff --name-only <rev>` plus untracked files: a commit under
`src/tui/` selects nothing, one under `inference/src/models/gemma4*.zig`
the Gemma gates; a CPU kernel file selects only the families that call it
(`backends/cpu/experts.zig` the 26B-A4B, `hadamard.zig` Bonsai,
`recurrent.zig` Qwen and Bonsai). The other tiers' gates that match are
listed, not run; `make verify-changed BASE=<rev> ARGS='--tier
verify-long'` (or `verify-cpu`, `verify-release`) runs them when the
change calls for that tier (the table above). The other commands: `make
verify` / `make verify-release` / `make verify-long` / `make verify-cpu`
(a tier), `make gate
NAME=<name or glob>` (`gemma4-qat-trace-f16`, `'muse-*'`; `ARGS=--dry-run`
prints the commands), `make gates-list`, and `make gates-validate` (part of
`make check`: the manifest's schema and the runner's self-test, no model).
`python3 scripts/gates.py --json` returns the results with the git
revision for a log entry, each with its output lines stamped by time
since launch (`--timeline` prints them); the check tools behind
`{zig-*}` are compiled first (`zig build check-tools`), so the builds'
seconds are reported apart from the gates'. Gate names are `<entry>-<check>-<executor>`:
`qwen38`, `gemma4`, `gemma4-qat`, `gemma4-26b-a4b`, `muse`, `bonsai`;
`trace-{cpu,f32,f16}`, `generation-{cpu,metal}`, `speculative-{cpu,metal}`,
`draft-trace-{cpu,metal}`, `draft-stats`, `vocabulary`, `perplexity` (two
windows), `perplexity-full`, `perplexity-4k`. A
model path is overridden per key by `<KEY>_MODEL` (`GEMMA4_QAT_MODEL=…`).
Traces are written under `.zig-cache/gates/trace/<gate>/`. The perplexity
gates read wikitext-2-raw's test text, fetched into `.reference/eval/` and
checked against its SHA-256 by `make eval-corpus`, which `verify`,
`verify-changed`, and `gate` run first (a no-op once the file is there);
their references are pinned under `tests/fixtures/perplexity/`
([guide/eval.md](guide/eval.md)).

### What each gate protects

Measured 2026-09-29 at `acce06a`, before the fast tier (M4 Pro, `python3 scripts/gates.py
--tier verify --json`, whose per-gate `timeline` stamps every output
line): the Metal tier took 704 s after 9 s of builds. Loading is not
the cost: a model's first gate pays the cold page cache (2–8 s), every
later one maps it in 0.3–1 s. The cost is in four places:

| Where | Seconds | Why |
| --- | --- | --- |
| the CPU reference inside Metal-tier draft traces | 184 (Muse), 46 (Gemma 12B QAT), 15 (E4B), 5 (Qwen's draft block) | `--draft-trace` and `checkDraft` always run the CPU forward, at the Metal build's ReleaseSafe, then the plan; the Metal half compares against the pinned fixtures on its own |
| `checkChunkedPrefill`'s F32 tiles and `prefillRows`, 70 tokens | 25–31 per 27–30B file | the generic F32 tile kernels, slowest path by design |
| perplexity, 8 windows of 512 | 56 (Qwen), 53 (Muse), 19 (26B-A4B), 12 (E4B) | the references store the cumulative value per window (`chunk_ppl`), so fewer windows compare against the same file |
| the Gemma 12B QAT file's five gates | 74 | a third Gemma file, covering only its dimensions (below) |

What can break independently, and what protects it. *Model-free* means
`make check` (unit tests and `test-metal`, no weights):

| Axis | Model-free | Gates that alone cover it |
| --- | --- | --- |
| matvec, matmul tiles, embed gather, per encoding | every quant fixture encoding (Q4_0, Q8_0, IQ4_NL, Q3–Q6_K, IQ3_S, IQ4_XS, PQ2_0, PTQ1_0) against the CPU, the CPU decoders against llama.cpp's blocks, and F32/F16/BF16 matvec and matmul tiles on random weights (`checkDenseEncodings`) | — |
| attention: decode, chunk, windowed, F16 cache | fixtures to 16K keys, and Gemma's geometries (16 heads of 512 over one or two KV heads, 16 of 256 over 8) | — |
| DeltaNet, Hadamard, RoPE, norms, softcap, sampling kernels | fixtures | — |
| Metal `layerNorm`, `addBiasRows` | strided rows against the CPU (`checkVisionNorms`) | — |
| Metal attention over packed sequences, the erf GeGLU (Laya) | packed sequences and windows against F64 (`checkSegmentAttention`), fused rows against the CPU (`checkGeluErfRows`) | `laya-metal` (the plan, its batching, and its cleanup) |
| Laya multilingual: mmBERT's geometry (22 layers of 768, intermediate 1,152), positions 512 to 1,023, the checkpoint's own special tokens | — | `laya-multilingual-cpu`, `laya-multilingual-metal` |
| experts (MoE) | Metal Q4_0 chains; CPU FFN on a 2×2 F32 tensor | `gemma4-26b-a4b-*` |
| Qwen 3.5 plan (DeltaNet, gated attention, MTP block) | the runtime's admission tests | `qwen38-*` |
| Qwen 3.5 with Hadamard rotations, PTQ1_0, BF16 | decoders, kernels | `bonsai-*` |
| Gemma 4: sliding/global, softcap | — | any Gemma file |
| Gemma 4: shared KV, per-layer embeddings, own V on global layers, window 512 | — | `gemma4-e4b-*` |
| Gemma 4: K=V global layers, window 1024, the `gemma4` profile | — | `gemma4-26b-a4b-*` (and the 12B QAT) |
| Gemma 4: one global KV head (16:1), 48 layers | the 16:1 attention geometry | the 12B QAT, release tier (a shape, through the same code as the 26B's 8:1) |
| Muse Glimmer plan | — | `muse-*` |
| session layout: recurrent with the verify tape (Metal, drafter) | session unit tests | `qwen38-generation-metal` |
| session layout: recurrent without a tape (rewind, replay) | session unit tests | `bonsai-generation-metal`; every CPU generation gate |
| session layout: attention only (truncate) | session unit tests | any Gemma or Muse generation gate |
| engine loop: EOS, budget, context, cancellation, speculative loop, penalties, top-k readback | none (`engine.zig` has no tests) | `qwen38-speculative-metal` |
| draft sources: Qwen MTP, Gemma assistant, Muse DFlash | — | `qwen38-draft-trace-*`, `gemma4-{e4b,qat}-draft-trace-*` (the same code, two shapes), `muse-draft-trace-*` |
| tokenizers: Qwen BPE, Muse (GPT-4o splitter), Gemma SPM, Laya byte-level and Metaspace | the Metaspace rules and rejections on a synthetic vocabulary | one vocabulary gate each (the 12B QAT file carries E4B's tokenizer) |
| vision: Qwen3-VL, SigLIP small with clamps, SigLIP large with standardization, Muse | preprocessing, grids | one vision gate each |
| batched prefill at 256 rows, F16 cache over 511 keys, real prose | — | the perplexity gates |
| a verify batch at depth (few-query attention, the multi-row weight path) against stepped decode from the same state | `checkAttentionVerify` against F64 to the 32K cache | `qwen38-verify-depth-{512,4k}`; `-16k`, `-32k` in `verify-long` |
| past 512 tokens and the sliding windows | — | `verify-long` |

Not covered by any gate: the 12B QAT's `gemma4uv` unified projector, the
26B-A4B's assistant head, and the PQ2_0 encoding (decoder and kernel
fixtures only). On Metal a speculative greedy divergence is printed, not
failed (chunk-versus-step rounding may flip a token); the CPU run fails
on it. `qwen38-draft-stats` reports acceptance rates and fails only on
an error.

Adding a family adds its gates to the manifest and nothing to the
Makefile ([models](models/README.md)). The
manifest's numbers are the accepted tolerances the reference documents
justify (Qwen's F16 bound in [metal-backend.md](engine/metal-backend.md),
Gemma's in [gemma4.md](models/gemma4.md), Muse's in
[muse-glimmer.md](models/muse-glimmer.md), Bonsai's in
[bonsai.md](models/bonsai.md)); the CPU rows and the F32 cache run at
the bring-up thresholds (max abs 2e-3, relative RMS 1e-4).

## The record

Every benchmark workload is a named entry in [`workloads.json`](../workloads.json),
run by `scripts/workloads.py` (`make workload NAME=<name or glob>`,
`make workloads-list`, `make workloads-validate` inside `make check`), and
its model path is one lookup into `gates.json`'s `models`. A `bench`
workload is one `nuclis bench --json` invocation: the prompt array
(`tests/fixtures/run-*/prompt-<n>.json`) or a raw text prompt, `max_tokens`,
`ctx`, `kv`, `warmup`, `repeat`, an optional `sampling` map (the instruct
profile's flags), `pair: true` with `draft_length` for the off/on pair on
one loaded model (`--speculative on`, which is how `bench` already
measures both ways; the pair names the catalogue entry so the `mtp`
companion resolves), and `baseline: true` for a second run with the switch
off, the no-drafter baseline. Reports are saved as
`.zig-cache/bench/<workload>/<rev>-<n>[-baseline].json` (`<rev>` the short
git revision, `<n>` a sequence per revision); nothing under `.zig-cache`
is committed, so a record's JSON is copied to `docs/benchmarks/` when the
document cites it. An `acceptance` workload delegates to
`scripts/nuclis-baseline.py` (the four-length arrays, `/usr/bin/time`,
hardware, the reference rows) and writes the dated record under
`docs/benchmarks/` as before. The workloads: `qwen38/{prose512,
prose4096, code}` and their `-draft` pairs, `qwen38/spec/*` (the twelve
configurations of the speculative record, pair plus baseline each),
`gemma4-qat/prose512{,-draft}`,
`gemma4-26b-a4b/prose512`, `muse/prose512{,-draft}`, `bonsai/prose512`,
and one `<entry>/acceptance` per pinned file.

`scripts/bench-report.py` turns saved reports into the documents' tables:
`--table FILE...` prints one row per report, the acceptance form (prefill
and decode with sample standard deviation, first token, session, load) for
plain runs and the record form (accepted and proposed per batch, the
per-batch costs, prefill and decode off → on, the no-drafter baseline when
a `-baseline` sibling exists, the speedup) for pairs; `--write-doc DOC
--name NAME` replaces the region between `<!-- bench:NAME -->` and
`<!-- /bench:NAME -->` in a document with that table and a provenance line,
so a generated table is committed and never transcribed; `--compare PREV
CUR` prints the deltas; `--check WORKLOAD FILE` tests a report against the
workload's `bars` (`{metric: {min, max}}` on the decode rate, the prefill
rate, or the speedup) and exits non-zero on a miss. Bars are sparse and
mean a verdict's condition (`muse/prose512-draft` carries `decode_speedup
≥ 1.0`, the reason its entry turns speculation on), not a regression
threshold: run noise is stated by the record, not enforced by the driver.

## The speed loop

A decode-speed change lands only through an interleaved A/B against a saved
base binary, at every context it could move. Three tools make that take
about a minute per context instead of an eleven-minute 32K prefill:

- **Saved prefixes.** `nuclis bench --prefix-cache <dir>` prefills the
  prompt less its last token once, writes the model's snapshot (the session
  and the drafter's carried row) to
  `<dir>/<model>-<tokens>-<layout>.snap`, and every later run with the same
  model files (size and first MiB of the target and draft source), prefix
  tokens, and session layout (layouts, KV precision, capacity) restores it
  and feeds the last token; prefill rates are then omitted. A file that
  does not match its header, its size, or its content digest is refused
  and re-prefilled. A file is the session's used extent (150 MB of
  recurrent state plus 64 KiB of F16 cache per token for Qwen); they live under `.zig-cache/speed/prefix/` and are never
  committed. The `qwen38-verify-depth-*` gates key theirs the same way
  (`inference.prefix_cache`), so the speed loop's files serve them. A layout change misses by digest; a numerical change to
  prefill leaves a saved prefix a valid state to time from.
- **Verify cost at depth.** `bench --speculative on --verify-rows R
  --accept a` times `--max-tokens` verify batches of R rows per run from
  the prompt's depth, with fixed drafts (the prompt's first tokens) and `a`
  accepted: the calls a speculative batch makes, in its order
  (`inference.engine.forcedBatch`), so the batch cost C(R, depth) is read
  without acceptance noise. The report splits C into propose, checkpoint,
  verify, recover, and commit per batch; `--profile` then times only the
  verify command buffer, and `--capture` records one verify batch's.
- **The A/B driver.** `make speed-base` builds and saves
  `./zig-out/bin/nuclis` (kernels embedded) with its revision and a dirty
  flag under `.zig-cache/speed/base/`. `make speed ARGS='--contexts
  512,4096,16384,32639 --verify-rows 1,4,8 --model qwen38'`
  (`scripts/speed.py`) builds the tree and, per context and mode, runs
  base and candidate `bench` processes in alternating order for
  `--pairs` (5) pairs of one run each (no warm-up: with the weights in the
  page cache a first run decodes at its second's rate), on the family's
  acceptance token arrays (another length slices the 32,639 array),
  restoring the shared saved prefix. It prints both medians, the change
  (positive is faster: decode tokens/s up, C down), the range of per-pair
  changes, and the keep rule's verdict per row and overall; `--json` prints
  the rows for a ledger. An A/A run (base = candidate) read −0.01 % and
  +0.07 % with per-pair changes within ±0.23 % at 512 (decode and a 4-row
  verify, 2026-09-30).

- **The speculative matrix.** `make spec-matrix ARGS='--model qwen38
  --contexts code,512,4096,16384,32639 --sampling greedy,instruct --drafts
  2-7'` (`scripts/spec-matrix.py`) runs real speculation per cell (context,
  sampling, draft length, and `--p-min` through `bench --draft-p-min`): one
  `bench --speculative on` process of three off/on pairs, on the same saved
  prefixes. It prints E, C and its parts, C / 50E, and the median pair
  speedup; reports land in `.zig-cache/spec/<model>/<rev>/` and a saved
  cell is not re-run, so a stopped matrix resumes (`--report` only reads
  them). The chip heats over a long sequence: plain decode at 512 drifts
  from 10.5 to 9.1 tok/s within 15 minutes, and the verify with it, so the
  paired ratio holds while the absolute rates do not; `--cooldown 90`
  (seconds idle before each cell) gives cold-chip rates. This is how a
  catalogue verdict is set
  ([benchmarks § The re-priced speculative verdicts](benchmarks/history.md#the-re-priced-speculative-verdicts-2026-10-01)).

**The keep rule** ([TODO.md](../TODO.md) while the decode-speed theme
runs): keep a change when its median decode, or for a verify lever the
batch cost C at the unit's row counts, improves by ≥ 2 % at one context or
more and no context regresses by more than 1 %, over ≥ 5 interleaved
pairs. A gain of 1–2 % keeps too when every pair is faster and none lies
more than 0.5 % from the median change: interleaved pairs agree far more
closely than the 1.5 % between separate runs the 2 % allows for (amended
2026-10-03, user). Then `make verify-auto` (and `make verify-long` for
attention or cache changes), commit `perf(inference): …` with the rows in the body, and
`make speed-base`. Otherwise revert and write the negative result down.

## Toolchain

The tested compiler is Zig 0.17.0; the three manifests require at least
0.17.0 ([release notes](https://ziglang.org/download/0.17.0/release-notes.html)).
Verify standard-library calls against that toolchain's installed source and
language reference rather than assuming older Zig examples still apply.
ZLS does not work with 0.17 yet: the release split `zig build` into a
configurer and a maker process, and ZLS waits on the new Build Server
Protocol. `zig build --print-configuration` prints the configured build
graph as ZON, the quickest way to see which steps and options exist.

Local language reference: `doc/langref.html` inside the Zig install
(`~/.local/opt/zig/stable/doc/langref.html` on the author's machine).

The Metal backend additionally needs Apple's SDK/frameworks and the
Objective-C compiler. The [reference baseline](benchmarks/llama-cpp.md)
compiles embedded shader source through Metal at runtime using Command Line
Tools; a standalone `metal` compiler is needed only for an offline shader
build path; nuclis compiles its kernels the same way. Swift is not a
project build requirement.

Pass allocator and Io dependencies explicitly. Prefer a separate pure function
for decisions that can be tested without file, network, process, or GPU access.
Document ownership on returned allocations and the lifetime of borrowed data.
Use scoped cleanup for normal paths and partial-initialization errors.

`main(init: std.process.Init)` receives the allocator, environment, arguments,
and Io implementation. Pass `init.io` through file operations; keep decisions
and in-memory parsing independent of OS access. Use `std.Io.Reader`/`Writer`
and `std.Io.Dir`/`File`, checking APIs against the installed standard library.

Zig's startup supplies `Io.Threaded`. The experimental `Io.Evented` backend
selects io_uring on Linux and Dispatch on macOS; io_uring does not run on the
target Mac. Keep the injected default until measurements justify changing it.
See the installed `lib/std/start.zig`, `lib/std/Io.zig` (`Evented`), and
`lib/std/Io/Dispatch.zig`. GPU work is scheduled by the Metal backend, not
by `Io` ([engine/metal-backend.md](engine/metal-backend.md)).

ReleaseSafe is the initial release configuration; ordinary `zig build` uses
Zig's Debug default. A faster optimization mode is justified only by numerical
validation and measured improvements; benchmark results must name the actual
mode used.

### Zig 0.17 usage audit

Audited 2026-10-03 against the
[0.17.0 release notes](https://ziglang.org/download/0.17.0/release-notes.html)
([REPO-29](https://github.com/tildaslashalef/nuclis/blob/v0.6.0/docs/worklog.md#repo-29--zig-0170-the-tree-migrated-and-its-features-adopted-2026-10-03-two-sessions)); counts are grep results over the
tree. 0.16's adoption of the `Io` interface stands, as does the decision to
skip its concurrency layer.

- **Removed, migrated:** `b.args` (→ `addPassthruArgs`), array `**`
  (→ `@splat`, or `repeat` in `src/text.zig` for multi-byte strings),
  `@typeInfo(...).fields` (→ `field_names`/`field_types`/`field_values`),
  `Allocator.dupeZ` and `fmt.bufPrintZ` (→ the `Sentinel` variants),
  `EnumSet.initEmpty`, `ascii.indexOfIgnoreCase`, and `zig build
  --global-cache-dir` (→ `ZIG_GLOBAL_CACHE_DIR`, set by the Makefile and
  `scripts/gates.py`).
- **Deprecated in 0.17, migrated:** `fmt.allocPrint` (→ `Allocator.print`,
  157 sites), `@intFromEnum`/`@enumFromInt` (→ `@backingInt` /
  `@fromBackingInt`, rewritten by `zig fmt`), `DynamicBitSetUnmanaged`
  (→ `bit_set.Dynamic`), `builtin.os` (→ `builtin.target.os`). The tree
  uses no other deprecated name. Deprecations are doc comments only; the
  compiler does not warn.
- **Silent semantic changes, checked:** every `@hasDecl` probe names a
  `pub` declaration on each family; every `@bitCast` is scalar; the float
  `mem.eql` calls compare distinct buffers.
- **Adopted:** `@divCeil` (40 ceiling divisions), `ArrayList.lastPtr` and
  `addOne` where they remove an index (4 sites; the rest read as well with
  `len > 0`), `--cache-poison=disallowed` passes for the default, Metal
  release, test, and check-tool graphs (no `build.zig` reads the working
  directory or calls `findProgram`; the root build reads `build.zig.zon`
  from the build root, which 0.17's configure cache tracks by itself).
- **`std.heap.SafeAllocator`:** `std.testing.allocator` and, in Debug,
  `std.process.Init.gpa`. It is thread-safe, never reuses memory, and panics
  on double frees and frees from another instance; leak reports come from
  it. It grows a block in place only while the block ends its bucket, so
  `std.testing.checkAllAllocationFailures` over it sees run-dependent
  allocation counts; the inference package checks through
  `inference.alloc_check`, which refuses in-place growth.
- **Not applicable:** incremental compilation and the self-hosted aarch64
  backend (Mach-O is not ready), `std.zon.parse` (the version line is the
  only ZON the build reads), `@SpirvType`, translate-c (no `@cImport`; the
  bridge is Objective-C through `addCSourceFile`), the Build Server
  Protocol (no tool here consumes it), `Io.Semaphore.waitTimeout` (the
  batcher's one-shot completion is an `Io.Event` with `waitTimeout`
  already), and comptime-length slice coercion (no site needs it).

Still deferred (0.17 keeps these calls; none sits in a numerical path):

| Use | Count | File | Migration |
| --- | ---: | --- | --- |
| `std.fs.path.join` / `isAbsolute` / `dirname` | 93 | 28 files across `src/`, `inference/`, `huggingface/` (most in `src/paths.zig`, `src/model.zig`, `src/decision/catalog.zig`) | `std.Io.Dir.path` equivalents |
| `std.posix.sigaction`, `Sigaction`, `SIG`, `SA.RESETHAND`, `sigemptyset` | 5 | `src/interrupt.zig` | `std.posix.system` calls, or an `Io`-level signal facility if a later release adds one |
| `std.posix.tcgetattr` / `tcsetattr` / `termios` / `poll` / `pollfd` / `winsize` / `system.ioctl` | 11 | `src/tui/terminal.zig` | `std.posix.system` for raw mode and size; `Io` polling once `Io.Evented` is usable |

The compiler is a separate versioning axis, so the upgrade is its own unit;
every benchmark record names the exact Zig used.

## Working on the agent

How the agent is looked at and measured while it changes; what it does
for a user is in [guide/](guide/), how its terminal works in
[app/terminal.md](app/terminal.md).

### Looking at the agent without a person at the keyboard

`make shot ARGS='<steps>'` (`scripts/tui-shot.py`) starts `nuclis agent`
in a detached tmux session (`tmux -L nuclis`, 160×45 by default, the status
bar off so the pane is the whole window, `COLORTERM=truecolor` as Ghostty
sets it so the theme resolves to the same palette) and runs the steps in
order: `keys=<text>`, `enter`, `key=<tmux key>` (`C-c`, `Tab`, `Escape`),
`paste=<text>` (a bracketed paste, as a dropped file's path arrives),
`wait=<s>`, `until=<text>,<s>` (wait for the text on screen), `capture=<name>`,
`burst=<name>,<seconds>,<hz>`, and `buffer=<name>` (tmux's paste buffer,
where an OSC 52 copy lands, to `<name>.buffer.txt`); `--env KEY=VALUE`
sets a variable for the agent (`EDITOR`, to drive Ctrl-G). A step with a
space is one quoted argument. A capture writes `<name>.txt`,
`<name>.ansi`, and `<name>.tagged.txt` (every styled run as
`[fg=#hex bg=#hex bold]…[/]`, so a colour is readable as text) under
`.zig-cache/tui/`; a burst captures at the given rate, keeps the frames that
differ from their predecessor, and reports the mean interval between them
in `<name>.burst.json` — the way an animation's cadence is measured rather
than eyeballed. `--ghostty` also opens one Ghostty window attached to the
session and photographs it to `<name>.png` on each capture (`screencapture`
needs the screen-recording permission once). `--direct` runs the agent in a
Ghostty window of its own with no tmux in between, for what only the
terminal does (kitty image placements, its cursor reports and resizes): a
pty relay in the window forwards the steps (through a FIFO) and the
window's own keys, records every byte the agent writes to
`<session>.stream`, `until=` searches that stream, and a capture writes the
PNG and `<name>.stream`; the window stays open after the agent exits so its
last screen can be photographed. The session stays up unless
`--stop` is given; `tmux -L nuclis attach -t shot` joins it. Captures are
evidence for the log, never fixtures: the goldens in `src/tui/` stay the
contract.

This is the iteration loop for every change to the surface: `make build`,
one `make shot` run that drives the feature (a `paste=` for a drop, `keys=`
for a prompt, `until=ready,<s>` to wait out a turn, `capture=` at each
state worth looking at), the `.txt` for the layout and the `.tagged.txt`
for the colours, then the fix and the same run again. A crash lands in the
capture as the panic and its trace, since tmux keeps the dead pane
(`remain-on-exit`), so a failure is read the same way as a success. To look
at a run in a real terminal, pass `--ghostty` (the harness opens the window
attached) or leave `--stop` off and attach from Ghostty. One thing the
harness cannot show is the inline image preview: it is off under tmux,
which cannot scroll a picture it does not know about, so that one is looked
at by running `./zig-out/bin/nuclis agent` in Ghostty directly.

### The agent's task list

`scripts/agent-eval.py` (`make agent-eval VARIANT=<name> ARGS='…'`) runs
thirteen small tasks against the playground: a tiny Python geometry package
that `scripts/playground.py` generates on first use under
`.zig-cache/playground/` (`make playground` builds it and prints the path)
and commits with `git init` as the baseline its own `make reset` returns
to. Nothing of it is committed to nuclis; it holds only what the tasks and
the README's GIF (`scripts/agent-demo.py`, which clones it) touch, from
text and fixed seeds, so every machine gets the same baseline commit, and
a version stamp in its `.git` rebuilds it when the generator changes
(`--workspace <dir>` runs another project with `make reset` and `make
test`). The tasks: whole-file questions (a line count
and a maximum, the release list), edits (a rename, the repeated heading,
a planted bug, a validation with its test), a new module with tests, a
flag added to the CLI, a silent `exit 7`, a tree-wide `grep`, a three-bullet
summary, the 1 MiB file, and "what is this project about". Each task is one
`nuclis agent --print` turn with a fixed seed from the committed baseline, checked by its own
predicate (file contents, a command's output, the answer's text) and
scored on steps, tool calls, tool errors, failed edits, prompt and
generated tokens, the model's seconds, the answer's length, and four
counted habits read from the session (a regex handed to the literal
`grep`, a guessed `pytest`, a file re-read right after its own edit, a
`cd` before a command); the records land under
`.zig-cache/agent-eval/<variant>/` with their session files, and
`--compare a b …` renders the variants side by side. `--system-prompt
<file>` runs a candidate prompt without a rebuild (the instructions
section still follows it), `--instructions <file>` plants the playground's
`AGENTS.md` (`tests/fixtures/agent-eval/AGENTS.md` by default, `none` for
no file), and `--seeds 1,2` is the default because a sampled turn varies:
two seeds cannot resolve a 10 % difference in seconds, so a change is
judged on the pass marks, the habit counts, and the medians rather than
the means. This is how a change to the system prompt, a tool description,
or the loop is judged: before and after, on the same list and seeds, and
the log entry cites the table; the first record is
[AGNT-13](https://github.com/tildaslashalef/nuclis/blob/v0.6.0/docs/worklog.md#agnt-13--the-system-prompt-as-sections-measured-the-playground-task-list-the-guidelines-that-changed-behaviour-the-instructions-file-2026-09-22).
`scripts/agent-tokens.py <variant> [<variant> …]` says where a variant's
prefill went: per tool, the calls, the tokens their results added (counted
by `nuclis tokenize --raw`, the engine's own tokenizer), and each tool's
share, with the variants side by side when given several.

## The website

`site/` is nuclis.dev: one page (`index.html`, `style.css`, `site.js`), a
`404.html`, and the files crawlers and Cloudflare read. There is no build
step and no dependency; what is in the folder is what is served.

| File | Holds |
| --- | --- |
| `index.html` | the page; its head carries the canonical URL, Open Graph and Twitter tags, JSON-LD, and the `#bench` block of measured figures |
| `site.js` | the agent replay, the engine strip, the drafter, the charts, the Decision Dungeons rooms, and the latest-release lookup |
| `fonts/` | Archivo and JetBrains Mono, subset WOFF2 ([THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md#website-fonts)); a new subset gets a new file name, since `_headers` caches the folder as immutable |
| `_headers` | Cloudflare's response headers: the CSP (no inline script or style anywhere), security headers, caching |
| `wrangler.jsonc`, `.assetsignore` | the Worker that serves the folder: its name, the 404 page, the `nuclis.dev` route |
| `og.png` | the 1200×630 share card, rendered from `scripts/site-og.html` |
| `robots.txt`, `sitemap.xml`, `site.webmanifest`, `llms.txt` | crawlers, installs, and language-model readers |

**Figures are copied, never typed twice.** The `#bench` block holds the
README's *Results* and *Speculative decoding* tables, speedup ratios
included (the README computes them from unrounded rates, so the page shows
them as given rather than dividing its rounded ones). `make site-check`
fails when the two differ, when the JSON-LD `softwareVersion` or the
download button's fallback is not the newest `CHANGELOG.md` release, when a
reference or `#fragment` does not resolve, when anything would load from
another origin, or when an inline style or script would break the CSP.
`gates.json` runs it when `site/`, `README.md`, or `CHANGELOG.md` changes,
so a release or a new benchmark table that the page does not follow fails
`make verify-auto`. The download button asks GitHub's API for the latest
release when the page loads and keeps the checked fallback when it cannot.

**Preview and look.** `make site-serve` serves the folder at
<http://localhost:8000> (without `_headers`). Check a change at 1440 and
390 px wide; any Chromium takes screenshots without a window:

```sh
chromium --headless --hide-scrollbars --window-size=1440,7800 \
  --virtual-time-budget=9000 --screenshot=.zig-cache/site.png http://localhost:8000/
```

The share card is rendered the same way, from the template beside the
scripts:

```sh
chromium --headless --hide-scrollbars --window-size=1200,630 --allow-file-access-from-files \
  --screenshot=site/og.png "file://$PWD/scripts/site-og.html"
```

To test the CSP locally, put `_headers`' policy (less `frame-ancestors`)
into a `<meta http-equiv="Content-Security-Policy">` of a copy and load it
with `--enable-logging=stderr`: a violation prints as a `CONSOLE` line.

**Deploy.** A Cloudflare Worker named `nuclis` with static assets and no
code (`site/wrangler.jsonc`; `.assetsignore` keeps the config itself from
being served), connected to this repository through Workers Builds: root
directory `site`, no build command, deploy command `npx wrangler deploy`,
production branch `main`, builds only when a push touches `site/*`. The
config routes the apex `nuclis.dev` as a custom domain (the zone is on the
same account, so the deploy writes its DNS record and Cloudflare issues the
certificate) and turns `workers.dev` and preview URLs off, so the site has
one address. The app was created in the dashboard (Workers & Pages →
Create → Continue with GitHub), which also mints the build token; the
watch path was narrowed with `cf builds triggers update <trigger>
--path-includes 'site/*'`, and `cf workers list` and `cf builds triggers
list --external-script-id <worker id>` show the state. Pushing `main` is
the deploy; nothing in CI deploys. `npx
wrangler deploy --dry-run` in `site/` validates the config without
uploading.

## Continuous integration and releases

Work reaches `main` only through pull requests, squash-merged by the user
(a ruleset on `main` requires a pull request and the `ok` check, and
refuses force pushes and deletion). The pull request's title, a Conventional
Commit, becomes the squash commit's subject, and its description, the
record of the work (`.github/pull_request_template.md`), becomes the body.

Two workflows under `.github/workflows/`. The macOS jobs install the
compiler with `.github/install-zig.sh`: the version comes from
`minimum_zig_version` in `build.zig.zon` and the SHA-256 from
`.github/zig-toolchain`, verified before anything is unpacked. No
marketplace action installs the toolchain and no digest is fetched at run
time; the tree pins a digest for every artifact it downloads. A Zig upgrade
edits the manifest, that file, and the toolchain section above in one unit
of work. Every action is pinned by full commit SHA rather than a moving
tag, and checkouts keep `persist-credentials` off.

**`ci.yml`** (every push to `main`, every pull request) always reports:

| Job | Runs | What |
| --- | --- | --- |
| `changes` | always (ubuntu) | whether anything outside `docs/`, `site/`, and Markdown changed |
| `docs` | always (ubuntu) | `make docs-check site-check gates-validate workloads-validate`, standard-library Python |
| `test` | code changed (macOS) | `zig fmt --check`, a semver check on the manifest's version, `zig build test` |
| `metal` | code changed (macOS) | a `-Dmetal=true -Doptimize=ReleaseSafe` build, `--version` and `--help` on it |
| `cpu` | code changed (macOS) | a Debug `-Dmetal=false` build, so the non-Metal path keeps linking |
| `ok` | always | passes when every job passed or was skipped: the check `main` requires |

That is `make check` minus the GPU. The toolchain cache is keyed on the
compiler version and its pins; each macOS job caches `.zig-cache` and Zig's
global cache per commit, restoring the newest. **Not** in CI: `test-metal`,
`verify`, `bench`, and anything that pulls a model. They need the pinned
artifacts and the real M4 Pro, and a rate measured on a virtualised GPU is
a number nobody should trust; those gates run locally and their results go
into the pull request's description.

**`release.yml`** (an annotated `v*` tag pushed by the user, or a dispatch
for a tag whose release is still a draft) refuses to publish, before it
builds anything, unless:

1. the tag is `vMAJOR.MINOR.PATCH[-prerelease]`;
2. the tag is annotated with a non-empty message (the highlights);
3. the tag equals `build.zig.zon`'s `.version`;
4. the installed `zig version` equals `minimum_zig_version`;
5. the tag's release, if one exists, is still a draft.

The Zig version is the tag's, but `install-zig.sh` is the workflow
revision's and the digests are the tag's `.github/zig-toolchain` plus the
workflow revision's (the first matching line wins). It runs the same gate,
builds `-Dmetal=true -Doptimize=ReleaseSafe`, asserts the binary reports
the tag's version, attests both tarballs
(`actions/attest-build-provenance`; check one with `gh attestation verify
<file> -R tildaslashalef/nuclis`), and creates a draft release with three
assets:
the binary tarball (`nuclis-vX.Y.Z-aarch64-macos.tar.gz`), a source
tarball (`nuclis-vX.Y.Z-src.tar.gz`, `git archive` of the tag), and a
`SHA256SUMS` covering both. The notes are the tag's message, GitHub's
generated list of the pull requests merged since the previous release
(`releases/generate-notes`), and `.github/release-footer.md`. It publishes
only once the asset count checks, so a failed upload never leaves a public
release missing files.

The binary is **unsigned and not notarized** (there is no Apple Developer ID
for this project), so macOS quarantines it on download; the notes say to run
`xattr -d com.apple.quarantine nuclis`, and building from source stays one
`zig build` away.

## Build and format contract

The root build provides:

```sh
zig build                   # Metal is on by default for macOS aarch64
zig build test              # src/, inference/, and huggingface/ unit tests
zig build -Doptimize=ReleaseSafe
zig build -Dmetal=false     # a CPU-only binary (the default off Apple Silicon)
zig build test-hf           # the huggingface package's tests alone
zig build hf-downloader     # its standalone binary, zig-out/bin/hf-downloader
```

`-Dmetal` defaults to **on for macOS on aarch64** and off everywhere else
(2026-09-12): the configuration's default backend is `metal`, so a plain
build that left it out produced a binary that failed at run time with
`MetalNotEnabled` against its own defaults. A build without the backend now
says so and names both ways out (rebuild, or `--backend cpu`). Building the
bridge does not make the default tests need a GPU: the fixtures that execute
kernels are the separate `test-metal` step.

The root `Makefile` wraps these and the explicit model/GPU targets with the
local cache flag already applied; `make help` lists every target. Targets that
run the CLI always use the freshly built `zig-out/bin/nuclis`.

The Python scripts under `scripts/` (tooling, never part of the build) follow
the root `pyproject.toml`: basedpyright in `standard` mode, the type checker
editors such as Zed run on it, and ruff with lines up to 120 columns. `make
lint-py` runs `ruff check`, `ruff format --check`, and basedpyright, each
pinned and fetched by `uvx`; `make fmt-py` formats; the `python` check of
`gates.json` runs `lint-py` whenever a script or `pyproject.toml` changes.

`make install` builds the Metal release and copies the binary to
`$(PREFIX)/bin/nuclis` (`PREFIX` defaults to `~/.local`); `make uninstall`
removes it. The binary is self-contained (the Metal shader source is
embedded and compiled at run time; models live under `~/.nuclis`). The old
file is removed before the copy, never overwritten in place: on Apple
Silicon the kernel keeps a replaced file's code signature cached and kills
the binary rewritten under it at launch. Shell completion is the user's to
install, from `nuclis completion fish|bash|zsh` (`nuclis completion --help`);
the scripts are shims that ask the binary on every Tab (`nuclis __complete`,
`src/completion.zig`), so they never need regenerating after an install.

The root build installs the executable under `zig-out/bin/` and tests the
executable, the inference module, and the download package. Default tests
must not initialize a GPU, load full model weights, or reach the network.

`zig build run -- inspect --json` runs nuclis directly.
In a sandbox that cannot write Zig's default global cache, set
`ZIG_GLOBAL_CACHE_DIR=/absolute/writable/cache` (Zig 0.17's `zig build` has no
`--global-cache-dir`); the Makefile and `scripts/gates.py` point it at the root
`.zig-cache/global` directory.

Format changed Zig source and build files with `zig fmt`. The repository check is:

```sh
zig fmt --check build.zig build.zig.zon src/ inference/ huggingface/
```

Explicit Metal/full-model test commands must be documented when introduced.
Verify behavior using the newly built binary.

## Versioning

The version lives in the root `build.zig.zon`: the last released version.
The manifests under `inference/` and `huggingface/` mirror it (the Zig
package format requires a `.version`; the path dependencies ignore it), and
`make version V=X.Y.Z` sets all three. The executable reports it as `nuclis
--version`, fed from the root manifest through build options.

- **SemVer with the 0.x convention.** Before 1.0 the minor number is the
  breaking axis: a release with breaking changes or features bumps the
  minor, one with only fixes the patch.
- **Releasing** (`scripts/release.py`; an agent runs it on "release"):

  | Step | Command | What it does |
  | --- | --- | --- |
  | Gather | `make release-draft` | the pull requests merged since the last tag, each with the first paragraph of its *What and why*; the suggested version; where the highlights go |
  | Write | (the agent) | the highlights, in `.zig-cache/highlights-vX.Y.Z.txt` |
  | Verify | `make verify`, `verify-cpu`, `verify-long`, `verify-release` | the release tiers on `main` (the pinned models, about an hour); skipped when no engine code changed |
  | Propose | `make release V=X.Y.Z TIERS='…'` | from an up-to-date `main`: the branch, `make version`, the commit, the push, and the pull request `chore(release): vX.Y.Z` with the highlights and the tiers in its description |
  | Merge | the user | |
  | Tag | `make tag V=X.Y.Z` | on `main` at `origin/main` with the manifest at `X.Y.Z`: the annotated tag with the highlights as its message, signed when git signs tags (`tag.gpgSign`), pushed; `release.yml` publishes |

  Each step refuses before changing anything when its conditions fail (a
  dirty tree, `main` behind, a version not above the last, a tag that
  exists, no highlights). **Good highlights** are two to four sentences for
  someone who uses nuclis: what changed for them first (a model, a speed, a
  command), measured numbers only, each one in the benchmark record; no
  internal names, unit jargon, or file paths. They head the release notes,
  above the generated list of pull requests.
- **A tag that exists on the remote never moves.** A ruleset refuses
  updating or deleting a `v*` tag; the fix is the next number.
- `CHANGELOG.md` keeps the releases up to v0.5.0; from v0.6.0 the
  [Releases page](https://github.com/tildaslashalef/nuclis/releases) is the
  changelog.
- Benchmark records cite the git revision, and the release tag once one
  exists.
- **The Zig toolchain is a separate axis.** `minimum_zig_version` pins source
  compatibility, the exact compiler is recorded in benchmark records, and Zig
  upgrades are their own unit of work.

## Commits, progress, and code explanations

Follow the commit, branch, and pull request conventions in
[../AGENTS.md](../AGENTS.md). A unit's tests, documentation, and its
[../TODO.md](../TODO.md) update travel in its pull request, whose
description is the record of the work.

The project is also intended to deepen the user's understanding of Zig. Add
module-level explanations of data flow and ownership, document public interfaces,
and explain subtle invariants beside the code that enforces them. Avoid comments
that merely translate obvious statements into English. Write implementation
walkthroughs as modules become stable, grounded in the working code.

## Testing

- Colocate Zig unit tests with the source they exercise.
- Use `std.testing.allocator` to catch leaks, including error-path leaks.
  Exhaustive allocation-failure checks in `inference/` go through
  `alloc_check.checkAll` (see the 0.17 audit above).
- Test through module interfaces. Use small CPU reference operations to verify
  GPU numerical work; do not require full-model runs for each source edit.
- Keep parsing, agent state, and memory-planning decisions deterministic and
  independent of I/O.
- Keep model and GPU tests opt-in, with an explicit external model path.
  Missing required hardware/artifacts must be reported as a skip or unmet
  prerequisite, never as a passing full-model check.
- Verify Objective-C and GPU resource lifetimes separately: Zig's testing
  allocator does not observe all platform allocations.

Use focused checks while implementing. Expand validation for shared-module
changes and unresolved concerns. Documentation-only changes need link and
consistency checks rather than unrelated code tests.

## Measurements and artifacts

Benchmark equivalent inputs and settings. Record toolchain and engine revisions,
hardware, model hash, context length, output length, precision, and warm/cold
conditions. Separate nuclis load time, prompt processing, and generated-token
latency. Preserve raw measurements outside the source tree unless deliberately
publishing a small, reviewed benchmark report; recorded artifacts live under
[benchmarks/](benchmarks/). Comparable runs against the reference use its
exact token arrays (`bench --prompt-tokens`; the `<entry>/acceptance`
workloads run `scripts/nuclis-baseline.py` on them, [§ The record](#the-record)),
never a re-tokenized rendering of them; `nuclis tokenize` shows what a text
prompt becomes before a model runs.

Store model weights and large datasets outside the repository. The initial
download and checksum are in [spec.md](spec.md#4-supported-models-and-artifacts).
Default build and test commands must not fetch them.

Secrets are environment-only and excluded from logs and fixtures. Real session
data and private source snippets must not become test or benchmark fixtures.

### Measuring the API

`nuclis serve` listens on `127.0.0.1:8000` by default (`serve.port`,
`--port`); start it with the models measured open (`--model laya --model
laya-multilingual`) so no request pays an open, and with `--quiet`: the
request log writes and flushes a line per response. Rates come from ApacheBench (`/usr/sbin/ab`,
in macOS) with keep-alive, and percentiles from its CSV, which keeps
fractions of a millisecond where the summary rounds to whole ones:

```sh
ab -k -n 512 -c 16 -p request.json -T application/json -e out.csv \
  http://127.0.0.1:8000/v1/decisions
```

`ab` speaks HTTP/1.0: with `-k` it keeps a connection only when the reply
says `connection: keep-alive`, which the server sends to a 1.0 request
that asked. Its "Failed requests" counts responses whose length differs
from the first, so `GET /v1/health` (whose counters change length) reports
nearly all as failed while every one is a 200: read the
`Length:` breakdown and `Non-2xx responses`. Compare served responses
with `nuclis decide --json` after masking `timings_ms`, and read the
batching counters from `GET /v1/health` before and after a run.
[guide/api.md § Measured rates](guide/api.md#measured-rates)
holds the record.

### GPU counters by capture

`xctrace`'s GPU counter profile is unsupported on the M4 Pro, but a Metal
capture replayed in Xcode reports the counters: ALU (FP32, FP16, integer)
utilization and limiters, occupancy, caches, and the MMU limiter per
dispatch. `make capture` runs `bench --capture` with
`MTL_CAPTURE_ENABLED=1`: after the runs, one more decode step is recorded
into `.zig-cache/trace/decode.gputrace` (`CAPTURE_OUT=` to move it,
`MODEL=`, `PROMPT=`, and `ARGS=` as for `bench`; the budget's context sets
the visible cache). The document holds every buffer the step reads, the
weights included: about 15 GB for Qwen3.8-27B, so delete it when done.
Xcode 27 ships no command-line reader for it, and its MCP tools (`xcrun
mcpbridge`) include none: open it in Xcode, tick *Profile after replay*
(the first time, Xcode asks to download its Metal Toolchain component,
which shader profiling needs), replay,
and read *Performance* (counters per dispatch, limiters). Never commit a
capture.

**Counters need a short command buffer.** Xcode 27 profiles a command
buffer in full only below a GPU run-time limit; above it the profiler runs
in *lite* mode ("the Metal workload exceeds the maximum size supported for
full profiling"): the timeline and the effective GPU time, with Shaders,
Heat Map, Cost Graph, and Counters disabled, and a Counters export of a
header alone. A 21 ms micro-benchmark buffer profiled in full; a 4-row
Qwen verify at 4K (332 ms at Maximum, 2026-09-30) did not. A whole-model
capture therefore confirms GPU time and dispatch order; counters come from
kernel captures.

**Pipeline limits without a capture.** `nuclis bench --kernel-stats`
(no model; `--json` for a machine-readable list) compiles the kernels and
prints each pipeline's `maxTotalThreadsPerThreadgroup`,
`threadExecutionWidth`, and static threadgroup memory. On this family 9
GPU the thread limit stays 1,024 whatever a kernel's registers
(dynamic caching), so register pressure is read only from a capture;
the threadgroup-memory column is the cheap check
([apple-gpu.md § Registers and occupancy](engine/apple-gpu.md#registers-and-occupancy-under-dynamic-caching)).

For one kernel on one shape, capture a micro-benchmark instead: `make
bench-kernels ARGS=Q4_K CAPTURE='matvec-Q4_K-5120x17408 (ffn_down)-block'`
(also `bench-matvec-rows` and `bench-attention`) records the last warm-up
command buffer of each case whose printed label contains `CAPTURE` into
`.zig-cache/trace/kernels/` (spaces become `-`, parentheses are
dropped), replacing the previous set.

**The division of work.** The agent makes the capture and asks the user
to profile it (gauge menu: Maximum, *Profile*) and export Counters as CSV
into `.zig-cache/trace/` under any name. When the user says it is done,
the agent renames the export to `<capture name>_<YYYY-MM-DDTHHMM>_<performance
state>.csv` (the time is the file's modification time; the state is
checked against the replay's GPU time, which at Maximum matches the
benchmark's rate), for example
`matvec-Q4_K-5120x17408-ffn_down-block_2026-09-30T0947_max.csv`, reads
it, copies the numbers that matter into
[apple-gpu.md](engine/apple-gpu.md) citing the file name, and deletes
the capture (the `.gputrace`), keeping the CSV. About 1 GB each:
the benchmark's buffers are sized for its largest shape, and the capture
holds whole buffers.

### Memory of a running process

`python3 scripts/nuclis_mem_usage.py` reports where a running `nuclis`
process's memory is (`--watch 2` samples until Ctrl-C and prints the peaks,
`--json` for one machine-readable report). Run it outside a sandbox: it reads
`footprint -j`, `vmmap -w`, and `mincore` over the mapped GGUF. The weights are
a file mapping the GPU reads through no-copy buffers, so they are clean page
cache outside the process footprint and RSS (Activity Monitor's *Memory*
column leaves them out); the script adds them back. The session state (KV
cache and recurrent state) is the heap shared with the GPU (`SM=SHM`).
Measured on Qwen3.8-27B-UD-Q4_K_M, context 16384, KV F16, `agent`, M4 Pro
48 GB, 2026-09-26: 16.11 of 16.46 GB of weights resident, session state
1.23 GB, Metal buffers 0.14 GB, driver 0.08-0.22 GB, CPU 0.31 GB; footprint
1.75-1.90 GB (peak 1.95 GB during prefill); total 17.9-18.0 GB, steady from
ready through a 16-step turn. System wired memory rises by about 17 GB while
command buffers run (the GPU wires what it reads) and falls back between
them. On exit (Ctrl-C twice mid-turn, Ctrl-D with a tool's child running,
Ctrl-C idle) the process is gone in 0.5-0.6 s with status 0, the footprint is
returned, wired memory falls back to its idle 3 GB, and a tool's child is
killed; the GGUF stays in the page cache as reclaimable cached files, which
is what makes the next load 20 s instead of a cold read.

## Reference implementations and third-party material

llama.cpp and other engines are references: read them to understand a format or
algorithm, run them to produce pinned fixtures and comparison traces, and measure
them as baselines. Do not copy their code into nuclis. New kernels and decoders
are written from the format contract and our own CPU references, then proven
equivalent by fixtures (bit-exact where the operation is exact, stated tolerances
otherwise). Outputs of running a reference are ours to commit as fixtures with
provenance; its source is not. Constants that define a format (codebooks,
lookup tables) and any other third-party material that must be in the tree are
listed in [../THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md), which links
license texts on the web. Do not commit license text files.

## Documentation conventions

- Keep product requirements and acceptance criteria in [spec.md](spec.md); the
  the active plan lives in [../TODO.md](../TODO.md) and each finished
  unit's record in its pull request.
- Add a document under `docs/` when there is concrete information to record,
  in the folder whose rule it fits, and give it a line in
  [README.md](README.md), the hub.
- Label proposed behavior, implemented behavior, and measured results accurately.
- Update existing authoritative guidance instead of copying it into a new file.
- Use relative repository links. Move a document with `make docs-check
  ARGS='--move old=new'`, which rewrites every reference; `make docs-check`
  fails on any link or anchor that does not resolve.
