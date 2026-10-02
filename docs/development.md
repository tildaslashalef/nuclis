# Development guide

Product behavior and acceptance criteria belong in [spec.md](spec.md);
agent-specific working instructions belong in [../AGENTS.md](../AGENTS.md).

## Current state and layout

Inspection, Qwen structural validation, native CPU generation, the opt-in
GPU-resident Metal backend (`-Dmetal=true`), `generate`, `bench`,
`tokenize`, `config`, `model pull` / `model ls`, and the `nuclis agent`
surface work; see
[../TODO.md](../TODO.md) for what is in progress and
[engineering-log.md](engineering-log.md) for what
closed. Teacher-forced scoring (`eval`) is a future increment.

```text
src/                 executable: CLI, model commands, config
  src/tui/           terminal surface, engine-free: screen, editor, theme,
                     markdown, keys
  src/agent/         agent composition: engine, conversation, sessions, tools
inference/           engine library: runtime, quantization, tokenizer,
                     sampling, backends, profiles
huggingface/         Hub download library (Xet) and its standalone binary;
                     imported by src/ only
docs/                spec, architecture, guides, reference, engineering log
TODO.md              active plan: unfinished units only
build.zig            build
build.zig.zon        manifest; single source of the version
```

The executable composes the `inference` and `huggingface` modules through
the root build. Keep kernels with the inference library that owns them.

## Environment

Facts every unit depends on; keep them here, not in `TODO.md`.

- Zig 0.16.0 (on the author's machine under `~/.local/opt/zig/stable`; any
  install of that version works); consult its installed std source for API
  details. Apple M4 Pro, 48 GiB, macOS 26. Metal compiles
  shaders at runtime from the Command Line Tools; Xcode-only tools are reached
  per process (see [../AGENTS.md § Local toolchain notes](../AGENTS.md#local-toolchain-notes)).
- Model `~/.nuclis/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf`
  (catalogue name `qwen3.8-27b`, the default `engine.model`), SHA-256
  `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482`,
  from [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
  at commit `4ca720788d1e01f1bff70c033e0d0028fd02e502`, pulled from a
  clean state by `nuclis model pull qwen3.8-27b --all` on 2026-09-11 (MODL-03).
  Its companions `mmproj-BF16.gguf` and `MTP/mtp-Qwen3.8-27B-Q4_0.gguf`
  sit beside it, verified and not executed
  ([reference/artifacts.md](reference/artifacts.md#pinned-commits-and-digests-modl-02-2026-09-11),
  [reference/gguf-inspection.md](reference/gguf-inspection.md#companion-files-in-modelsqwen-2026-09-08)).
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
  ([reference-baseline.md § The second oracle](reference/reference-baseline.md#the-second-oracle-the-prismml-fork-modl-16-2026-09-18)).
  The decision model's oracle is the `laya` 0.3.20 Python package in a venv
  at `.reference/laya-venv` (Python 3.12 through `uv`; the recipe
  heads `scripts/laya-reference.py`; `--subfolder multilingual` for the
  multilingual set), run on the CPU in F32 against the pulled checkpoint
  from a staged copy, because the package may rewrite
  `tokenizer_config.json` in place.
  The reference oracle itself is committed under `tests/fixtures/`
  ([provenance](../tests/fixtures/provenance.md)); `make gate NAME='qwen38-trace-*'` reads
  `tests/fixtures/reference-hello-comma`. Accepted reference warm rates
  (prefill/decode tok/s): 512 in = 89.19/9.66; 4K = 89.26/9.21;
  16K = 74.07/7.32; 32,639 = 67.28/6.71
  ([reference-baseline.md](reference/reference-baseline.md)). nuclis on the
  same token arrays (ENGN-07 record, 2026-09-10): 512 = 90.45/10.62; 4K =
  83.70/10.20; 16K = 62.70/8.27; 32,639 = 49.55/7.55
  ([bench.md § Acceptance runs](reference/bench.md#acceptance-runs)).
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
  [bench.md](reference/bench.md) says what each records. `generate
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
| cost | minutes (38 gates; 258 s measured 2026-09-29 for 36 with the three `laya-multilingual-*`, 254 s for the 33 before them, down from 38 gates in 704 s; engineering log, REPO-20, MODL-31, MODL-33; then `qwen38-verify-depth-512` and `-4k`, 3–4 s each from saved prefixes) | minutes of Metal (8 gates, 156 s) and the 12B QAT file's two CPU gates (tens of minutes) | minutes (3 gates: `gemma4-e4b-perplexity-4k` 53 s; `qwen38-verify-depth-16k` and `-32k`, 4–6 s each from the saved prefixes under `.zig-cache/speed/prefix/`, which a missing file costs one prefill: about 3 and 11 min; the other families wait for their references) | 22 min (14 gates, 1,327 s measured 2026-09-30 with the reference's `matvec` on every core, from hours; `muse-vision-cpu` 447 s and `qwen38-speculative-cpu` 308 s the longest; engineering log, KERN-19) |
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
`manifests`), then the matched `verify` gates cheapest first, one model
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
([reference/eval.md](reference/eval.md)).

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
Makefile ([new-model-guide.md](reference/new-model-guide.md)). The
manifest's numbers are the accepted tolerances the reference documents
justify (Qwen's F16 bound in [metal-backend.md](reference/metal-backend.md),
Gemma's in [gemma4.md](reference/gemma4.md), Muse's in
[muse-glimmer.md](reference/muse-glimmer.md), Bonsai's in
[bonsai.md](reference/bonsai.md)); the CPU rows and the F32 cache run at
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
  ([bench.md § The re-priced speculative verdicts](reference/bench.md#the-re-priced-speculative-verdicts-engn-20-2026-10-01)).

**The keep rule** ([TODO.md](../TODO.md) while the decode-speed theme
runs): keep a change when its median decode, or for a verify lever the
batch cost C at the unit's row counts, improves by ≥ 2 % at one context or
more and no context regresses by more than 1 %, over ≥ 5 interleaved
pairs; then `make verify-auto` (and `make verify-long` for attention or
cache changes), commit `perf(inference): …` with the rows in the body, and
`make speed-base`. Otherwise revert and write the negative result down.

## Toolchain

The tested compiler is Zig 0.16.0; manifests require at least 0.16.0.
Verify standard-library calls against that toolchain's installed source and
language reference rather than assuming older Zig examples still apply.

Local language reference: `doc/langref.html` inside the Zig install
(`~/.local/opt/zig/stable/doc/langref.html` on the author's machine).

The Metal backend additionally needs Apple's SDK/frameworks and the
Objective-C compiler. The [reference baseline](reference/reference-baseline.md)
compiles embedded shader source through Metal at runtime using Command Line
Tools; a standalone `metal` compiler is needed only for an offline shader
build path. Pin nuclis's tested macOS/SDK requirements during backend
bring-up. Swift is not a project build requirement.

Pass allocator and Io dependencies explicitly. Prefer a separate pure function
for decisions that can be tested without file, network, process, or GPU access.
Document ownership on returned allocations and the lifetime of borrowed data.
Use scoped cleanup for normal paths and partial-initialization errors.

`main(init: std.process.Init)` receives the allocator, environment, arguments,
and Io implementation. Pass `init.io` through file operations; keep decisions
and in-memory parsing independent of OS access. Use `std.Io.Reader`/`Writer`
and `std.Io.Dir`/`File`, checking APIs against the installed standard library.

Zig 0.16's startup supplies `Io.Threaded`. The experimental `Io.Evented` backend
selects io_uring on Linux and Dispatch on macOS; io_uring does not run on the
target Mac. Keep the injected default until measurements justify changing it.
See the [0.16 release notes](https://ziglang.org/download/0.16.0/release-notes.html)
and installed `lib/std/start.zig` and `lib/std/Io/Evented.zig`.
GPU scheduling will be a separate Metal backend responsibility.

ReleaseSafe is the initial release configuration; ordinary `zig build` uses
Zig's Debug default. A faster optimization mode is justified only by numerical
validation and measured improvements; benchmark results must name the actual
mode used.

### Zig 0.16 usage audit

Audited 2026-09-09 against the
[0.16.0 release notes](https://ziglang.org/download/0.16.0/release-notes.html);
"uses"/"does not use" are grep results over `src/` and `inference/`. The tree
adopted the `Io` interface end to end (`std.process.Init` with `io`,
`std.Io.Dir`/`File`, `std.Io.Reader`/`Writer`, `File.createMemoryMap`,
`std.Io.Clock`) and deliberately skips the concurrency layer (`io.async`,
`Io.Mutex`, `Io.Evented`): the engine is one sequential GPU pipeline,
cancellation is a flag read between layers in `src/interrupt.zig`, and the chat
polls with `std.posix.poll` because `Io.Evented` is experimental. It is clean
on the other 0.16 removals (`@cImport`, `@Type`, `@Vector` indexing,
`Thread.Pool`, `SegmentedList`, `ArenaAllocator` locking).

Two spots to migrate at the next compiler upgrade, both isolated from engine
code:

| Use | Count | File | Migration |
| --- | ---: | --- | --- |
| `std.fs.path.join` / `isAbsolute` / `dirname` | 7 | `src/paths.zig`, `src/config.zig` | `std.Io.Dir.path` equivalents |
| `std.posix.sigaction`, `Sigaction`, `SIG`, `SA.RESETHAND`, `sigemptyset` | 5 | `src/interrupt.zig` | `std.posix.system` calls, or an `Io`-level signal facility if a later release adds one |
| `std.posix.tcgetattr` / `tcsetattr` / `termios` / `poll` / `pollfd` / `winsize` / `system.ioctl` | 11 | `src/tui/terminal.zig` | `std.posix.system` for raw mode and size; `Io` polling once `Io.Evented` is usable |

The compiler is a separate versioning axis, so the upgrade is its own unit;
every benchmark record names the exact Zig used.

## User directories

nuclis uses `~/.nuclis` as its user configuration/data root. `NUCLIS_HOME`,
when set, overrides it and must be an absolute path. An empty or relative
override is an error. Otherwise resolve the root from `HOME`.

```text
~/.nuclis/
  models/                  model artifacts: <owner>/<repo>/<file> plus a provenance
                           sidecar <file>.nuclis.json per verified file ([reference/artifacts.md](reference/artifacts.md))
  nuclis.json              engine configuration (APPS-03; `nuclis config init` writes it)
  agent/                   the agent's data root (`paths.agentPath`; TERM-01)
    history.jsonl          submitted prompts across sessions (TERM-01; append-only,
                           one JSON object per line, tail-read at startup)
    sessions/<cwd-slug>/   one append-only JSONL file per session
    exports/               `/save` markdown exports
                           ([spec.md § Sessions and storage](spec.md#74-sessions-and-storage))
  cache/                   regenerable runtime data (planned)
```

Create directories only when an operation needs to write them. Inspection/help
must not create settings or directories. Do not migrate or overwrite existing
user files implicitly. Credentials remain environment-only (`HF_TOKEN` for
gated Hub repositories; public ones need none).

### Model download

`nuclis model pull <name> [--with mmproj,mtp | --all]` fetches a catalogue
entry (`src/catalog.zig`, the only source of "supported": name, repository,
file, pinned commit, digests, companions with the unit that will load them;
`qwen3.8-27b` today) into `<root>/models/<owner>/<repo>/<file>` through the
`huggingface` package ([its README](../huggingface/README.md): native Xet
reconstruction, SHA-256 verified, atomic publication, verified reuse of a
file already in place) and writes the sidecar beside each file; the Hub's
digest at the pinned commit must equal the catalogue's (`CatalogMismatch`
otherwise, and no sidecar). `nuclis model pull <owner/repo> [--file <name>]
[--revision <rev>] [--role main|mmproj|mtp|imatrix]` fetches any other
artifact by repository id (a GGUF, or a safetensors set with its config
and tokenizer files, [reference/artifacts.md § Safetensors
artifacts](reference/artifacts.md#safetensors-artifacts)): the revision (`main` by default; a tag, branch, or
commit) is resolved once, printed as the 40-character commit, and that
commit pins the transfer and is what the sidecar records; a repository
with several artifacts and no `--file` prints the choices with sizes and fails
with `SelectionRequired`; exact names match in full, subdirectories
included (`--file MTP/mtp-Qwen3.8-27B-Q4_0.gguf` keeps the subdirectory).
Both forms take `--force` and `--json` and need the root and nothing from
`nuclis.json`. A registry entry name (`nuclis model pull gemma`, see
[§ Configuration file](#configuration-file)) is the one form that reads the
file: it pulls the entry's `repo`/`file` at its `revision` (`main` when
unset) with the Hub's digest, `--with mmproj,mtp` or `--all` adding the
entry's companion names; an entry that names a `path` has nothing to pull
(`NotPullable`), and a name that is none of the three forms is
`UnknownModel`. A second pull of the same file hashes it, downloads nothing,
and rewrites the sidecar. A sidecar recording other content is `ExistingFileMismatch` unless
`--force` replaces file and sidecar; a differing file nuclis never verified
is the same error from the package, and `--force` replaces it too. Progress
is one updating line on stderr (bytes, rate, ETA) on a terminal, phase lines
otherwise; Ctrl-C cancels through the package's sink, which removes the
partial file (a second Ctrl-C kills the process the ordinary way and may
leave the temporary file). `nuclis model ls [--json]` prints the catalogue
with each entry's local status from its sidecar alone (`present`,
`absent`, `mismatch`, `unverified`: the file is there without a sidecar)
and its companions beneath with a "not loaded yet" note, then the other
model files (GGUF, safetensors weights) in the layout with their sidecar facts (runnable only if their
architecture has an adapter); files above `<owner>/<repo>/` are counted,
not listed; every listed file that a registry entry locates (`path`, or
`repo` and `file`) says `registered as <name>`, with the profile when the
entry forces one, and entries whose file is absent are listed last (a
`nuclis.json` that fails to load leaves the listing unannotated with one
warning line). `make model-ls` wraps it. `nuclis model pull <owner/repo>
--file <name> --register <name> [--profile <p>]` also writes the pull as a
registry entry (`repo`, `file`, the resolved commit, and the forced
profile) once every file is verified, so `--model <name>` and `config set
engine.model <name>` work from then on; a companion role fills the same
entry's `mmproj`/`mtp`, a name that locates other content is refused, a
catalogue name is refused before the transfer unless it names the
catalogue's own file (the registry resolves first, so such an entry would
shadow the catalogue; the loader rejects one however it got there), and a
registry-entry pull refuses `--register`. `--model` and `engine.model`
accept a registry entry, a catalogue name, or a path; a missing file
fails before anything opens, naming the resolved path.

`nuclis model inspect (<name> | <owner/repo> --file <name>) [--revision <rev>]
[--json]` answers "will this quantization load" before a download. It lists
the repository at the resolved commit, fetches the head of the file through
the package's `readRange` in 8 MiB windows until the GGUF directory parses
(never past the parser's 64 MiB directory bound; the Qwen directory is
11.0 MB and Gemma 4 12B's 15.8 MB, two requests and about 4 s each on
2026-09-11), prints what `inspect` prints for a local file, and ends with
a verdict: `supported` (the catalogue pins the Hub's digest for the file
and the adapter binds the directory), `runnable` (an adapter for
`general.architecture` binds it, but the file is not in the catalogue or
carries another digest), or `not runnable` naming the first offending
tensor and its encoding (a layout nuclis does not store, or one outside
the adapter's executable set), the missing adapter (`gemma4`, `clip`), or
the adapter's rejection; a catalogue companion (`mmproj-BF16.gguf`) is
reported as the companion it is. Nothing is written and no weights are
downloaded; a repository with several GGUFs and no `--file` lists them as
`pull` does.

### Configuration file

`nuclis.json` is one sectioned document (`engine`, `generation`, `agent`,
`decide`, and the `models` registry) with a `schema_version`. The sections name a
*scope*, not a command: `engine` (the artifact and its session) and
`generation` (how tokens are produced: budget, effort, speculative decoding,
sampling) are shared by `generate` and `agent`; `agent` holds only the chat
surface's own settings (`think`, `fold_thinking`, `theme`, `instructions`,
`thinking_budget`, and was named
`chat` until 2026-09-11, see
[spec.md § Configuration](spec.md#58-configuration)); `bench` reads `engine` plus
its own flags. The section was named `generate` until 2026-09-20, when the
rename made the scope explicit;
`src/config.zig` is its schema and the built-in defaults. `nuclis config
init` writes the defaults with every catalogue model as a registry entry
(one today), so the file shows the entry shape with the catalogue's facts
(the entries are optional: a catalogue name resolves without one), then
prints the effective engine keys, each catalogue model's local status,
and the `nuclis model pull <name>` to run next (the example below).

`nuclis config init --discover [--dry-run] [--json]` registers what the
catalogue does not name: it walks `<root>/models` as `model ls` does,
skips the files a registry entry already locates and the companions
(sidecar role, or a name carrying `mmproj`, `mtp`, or `dflash`), reads each
remaining file's GGUF directory and judges it as `model inspect` does (an
adapter for its architecture, every tensor in the executable set, the
binding), and writes one entry per runnable file: the name is the
repository's last path segment lower-cased (`-gguf` dropped; a taken name
gains the quantization suffix, then a counter; never a catalogue name),
the entry is `repo` + `file` + `revision` from the sidecar (or `path` when
there is none), companions beside the file fill `mmproj` and `mtp`,
`profile` is forced to the family's when the template digest matches no
profile (the finetune case, otherwise left to the digest), and
`generation.speculative` / `draft_length` take the catalogue's verdict for
the same architecture (off when the family drafts from a companion that
is absent). Every skipped file is reported with its reason, a header that
fails to parse included; the file is created first when absent and kept
when present, and `--dry-run` prints the report without writing
(APPS-15). Discovered on 2026-09-21: the HauhauCS Gemma 4 12B finetune
with its projector and a forced `gemma4` profile, and the Bonsai 2 PQ2_0
bring-up file. The example:

```json
{
  "schema_version": 1,
  "engine":   { "model": "qwen3.8-27b", "backend": "metal", "ctx_size": 16384,
                "kv_precision": "f16" },
  "generation": { "max_tokens": 4096, "think": "off", "speculative": false, "draft_length": 4,
                "image_max_tokens": "auto",
                "sampling": { "temperature": null, "top_k": null, "top_p": null, "min_p": null,
                              "presence_penalty": null, "repetition_penalty": null } },
  "agent":    { "think": "low", "fold_thinking": true, "theme": "gruvbox-dark", "instructions": "auto",
                "thinking_budget": 1024 },
  "decide":   { "model": "laya" },
  "models":   { "qwen3.8-27b": { "kind": null, "path": null,
                                 "repo": "unsloth/Qwen3.8-27B-GGUF", "file": "Qwen3.8-27B-UD-Q4_K_M.gguf",
                                 "revision": "4ca720788d1e01f1bff70c033e0d0028fd02e502",
                                 "mmproj": "mmproj-BF16.gguf", "mtp": "MTP/mtp-Qwen3.8-27B-Q4_0.gguf",
                                 "profile": null, "ctx_size": null,
                                 "generation": { "max_tokens": null, "think": null, "speculative": false, "draft_length": 4, "image_max_tokens": null, "sampling": { "…": null } },
                                 "agent": { "think": null, "fold_thinking": null } } }
}
```

Each entry's `generation.speculative` / `generation.draft_length` is the
family's measured verdict (`src/catalog.zig`; [bench.md § Definitions](reference/bench.md#definitions)),
so a fresh file already turns speculation on for the family whose record
pays and off for the rest; a user's global `generation.speculative` still
applies to models with no entry, and `--speculative` overrides either.

- Precedence: built-in defaults < the model's sampling profile < the
  file's global sections < the registry entry the model names < command-line
  flags. `NUCLIS_HOME` only moves the root; there are no per-key environment
  overrides and no per-project files. `nuclis config show [--json]` prints
  the effective value of every key with its source (`default`, `profile`,
  `file`, `model`, `flag`), one line above the table naming the model, how
  it resolved (registry entry, catalogue name, path), its profile, and that
  a `null` sampling key takes the profile's value for the configured
  `generation.think`; then each registry entry's stated keys. The profile
  named there is the catalogue entry's (the first profile for a bare path
  or an unknown registry name), chosen without opening the file; a run
  samples with the opened file's own profile, selected by its template
  digest, so a Gemma file reached through a path still gets Gemma's
  defaults (MODL-07). An entry's `profile` (`qwen38`, `gemma4`) or the
  `--prompt-profile` flag forces that profile on the file whatever its
  template digest — the way to run a finetune converted with another
  revision of the template, which the engine would otherwise refuse as
  `UnsupportedPromptTemplate`; the agent prints a notice at startup, and
  the rendering is the pinned protocol's, not necessarily the file's own
  ([prompt-profile.md § Evidence](reference/prompt-profile.md#evidence-and-reproduction)).
  `nuclis config set <key> <value>` changes one key by its dotted name
  (`engine.model hauhau`, `generation.sampling.temperature 0.7`,
  `models.<name>.profile gemma4`; `null` clears an override): the file's
  own text is edited so stated keys and their order survive, the result
  goes through the same loader before it is written (a refused value
  leaves the file untouched and names the key), `engine.model` must
  resolve to a file that exists, and a missing file is created as `init`
  writes it. Entries are created by `model pull --register`, never by
  `set` (`models.<name>.<key>` on an unknown name says so). The JSON form
  carries the file as loaded (`config`, the registry as a map), the
  `effective` view, and the `sources` map. The flag layer is visible in each
  command's own report (`generate --json`, the `bench` report's `config`
  and settings fields).
- `engine.model` is a registry entry name, a catalogue name (`qwen3.8-27b`,
  the default), or a path, tried in that order; a path resolves under
  `<root>/models` unless absolute. `--model` takes the same forms, a path
  being as given. `engine.backend` defaults to `metal` when the build
  has it, `cpu` otherwise. `engine.kv_precision` (`f16` default, `f32`;
  flag `--kv`) is the attention cache layout on the GPU: `f16` halves the
  cache's memory and the bytes attention reads per token (KERN-07); the CPU
  reference always keeps F32, and every report (`generate --json`, the
  `bench` header and JSON) states the precision the session actually used
  beside its `session_bytes`. `bench` takes model, backend, and context
  (the entry's `ctx_size` included) from the file and keeps its output
  budget (32), repetitions, and greedy sampling on the command line so runs
  stay comparable; its report records the file it ran with.
- The registry: `models` maps a name (1..64 printable characters, no `/`,
  not ending in `.gguf`) to an entry that locates a model one way, `path`
  (relative to `<root>/models` unless absolute) or `repo` + `file` (the
  layout `model pull` writes, `<root>/models/<repo>/<file>`) with an
  optional `revision`; optional `mmproj` and `mtp` companion file names
  in the same directory (recorded for the units that will load them, the vision unit
  and the MTP unit, and used by `model pull <name> --with`); and optional
  `ctx_size`, `generation` (`max_tokens`, `think`, `speculative`, `draft_length`, `image_max_tokens`, `sampling`), and `agent`
  (`think`, `fold_thinking`) overrides that apply only while that entry is
  the model, `null` meaning the global value. Entries pin no digest (a
  pull by entry name takes the Hub's). A registry name shadows a catalogue
  name for `--model`/`engine.model`; `model pull` tries the catalogue
  first, since it needs nothing from the file. An entry with `"kind":
  "decision"` (written by `model pull --register` for a Laya layout) is a
  decision checkpoint: `repo` + `file` or `path` name its weights, whose
  directory `nuclis decide` opens; text commands refuse it by name.
- `decide.model` (default `laya`) is the checkpoint `nuclis decide` opens:
  a registry entry of kind `decision`, a decision catalogue name
  ([artifacts.md § The catalogue](reference/artifacts.md#the-catalogue)),
  or a directory (under `<root>/models` unless absolute); `--model` takes
  the same forms. A text model's name is refused.
- Sampling entries are overrides: `null` means the official profile of the
  reasoning mode ([generation.md](reference/generation.md#sampling-profiles-and-the-selection-chain-modl-01)),
  so the file never freezes a model's recommended settings. The profile is
  the adapter's (`qwen38` for the one adapter; the catalogue records it per
  entry so `config show` names it without opening the file; the adapter registry dispatches
  per architecture).
- Validation: unknown keys are rejected with their dotted path (registry
  keys as `models.<name>.<key>`, so an unknown companion such as
  `imatrix` is named), wrong types and enum values name the key and the
  accepted form, ranges are the same as for flags (context 1..32768,
  tokens 1..16384, sampling options through the sampler's rules), an entry
  must locate its model one way, the file is bounded at 64 KiB, and a
  `schema_version` other than 1 is an error that states the migration
  (move the file aside, `config init`, copy settings back). Adding a key
  with a default does not bump the version (the registry was added to
  schema 1; a file without it loads with an empty one); renaming or
  re-typing one does, except the `chat` → `agent` section rename, whose
  keys and defaults are unchanged: a file with a `chat` section is
  rejected with a message that says to rename it. Keys arrive with the features that read them
  (`engine.kv_precision` arrived with KERN-07; an `agent` section comes with
  the embedded agent).

### Styled output

Every text report (`config show`, `model ls|pull|inspect`, `inspect`,
`validate`, `tokenize`, `bench`) and the `error:` line on stderr are
colored with the agent's palette (gruvbox dark by default) when the stream
is a terminal
and the environment advertises color (`COLORTERM=truecolor` for 24-bit,
a `256color` or `direct` `TERM` for the approximations, any other terminal
for the sixteen ANSI slots). `--json`, a pipe, `NO_COLOR`,
or an unset or `dumb` `TERM` disable styling completely, so scripts and the tests
see exactly the same bytes; the tests pin the plain form with
`style.Style.none`. The renderers take a `style.Style` explicitly and pad
text before wrapping it in escapes, so alignment never depends on them.
The palette lives in `src/tui/theme.zig` (TERM-01): named palettes selected by
`agent.theme`, behind a semantic style enum that a theme cannot change, so
a theme changes colour and never layout. `src/tui/style.zig` is the same
palette applied to one-shot reports.

Dim is a colour, not a faded one: the `dim` role paints gruvbox's grey and
adds the SGR dim attribute only at the plain level, where there is no colour
to carry the meaning. Stacking both halves an already low-contrast foreground
and made notices and the help page unreadable on a translucent terminal
(reported and fixed 2026-09-12).

Glyphs are a separate axis from colour (TERM-01 step 6). Every decoration the
agent draws — bullets, task boxes, rules, table joints, fold arrows, the
spinner, the status-bar labels — is named in `theme.Glyphs`, with a Unicode
table and an ASCII one. The ASCII table is selected when the locale does not
claim UTF-8 (`LC_ALL`, then `LC_CTYPE`, then `LANG`, none of which contains
`utf-8`/`utf8`) or when `NUCLIS_ASCII=1` is set, which is also how to check
the fallback on a UTF-8 terminal. Colour and glyphs never influence each
other: `NO_COLOR` keeps the Unicode drawing, and an ASCII terminal keeps its
colours.

### The agent's live region

`src/tui/screen.zig` is the only module that emits a movement escape, and
its geometry is the contract the rest of the surface is written against: a
repaint rewrites the live region in place inside synchronized output
(`CSI ? 2026 h/l`), an insertion above it narrows the scrolling region to
the rows above (`DECSTBM`, top margin row 1 so scrolled-off rows still
reach the scrollback) and scrolls only those, and the region's *bottom*
stays anchored, so a turn that grows the region pushes the transcript up
and one that shrinks it releases rows above the editor rather than
leaving blanks beneath it. A resize replays only the last turn at the new
width; older turns are left as the terminal reflowed them (spec § The agent:
completed turns are immutable), and the region's bottom follows the new
last row: a taller terminal adds blank rows under the region that become
slack above it, a shorter one is taken to have kept its last rows in view,
as tmux and Ghostty do. `NUCLIS_NO_SCROLL_REGION=1` forces the
cursor-up rewrite fallback for a terminal that mishandles `DECSTBM`, and a
`dumb` or unset `TERM` turns both capabilities off. The escape stream of
every operation is pinned by golden tests that need no TTY.

### The agent's transcript

Between the renderer and the screen sits `src/tui/transcript.zig`, the
answer to "what is on the screen, and who may rewrite it" (TERM-01 step 7). The
agent produces typed events (`src/tui/event.zig`: user, thinking, answer,
tool call, tool result, diff, notice, status, turn end); the transcript turns
them into blocks and offers three views of those blocks:

- `takeClosed` — rows for everything that closed since the last call, marked
  written. The agent inserts them above the live region. **Exactly once**:
  the same rows can never be handed out twice, which is what makes a
  scrollback both append-only and correct. An answer flushes block by block
  as its markdown blocks close, so a long answer scrolls away while it is
  written.
- `liveRows` — what is still open, tail-clamped under `… N lines above`.
- `replayRows` — every written block again, for the two events that may
  rewrite the scrollback (a fold toggle, a resize). The agent skips the
  rewrite when the turn is taller than the space above the region: those
  rows are in the scrollback and cannot be reached.

The transcript is pure: no `Io`, no clock (the animated thinking label and
the running dot's pulse phase are passed in by the agent, which has both),
and no knowledge of tokens or models. That is what lets its tests drive a
whole turn — including a byte-by-byte stream — with `std.testing.allocator`
and no TTY. The rows it produces for a tool call are the dot in the call's
state colour (`op_running` pulsing against `dim`, then `op_ok`, `op_error`,
or `op_write`) and `Name(argument)` from `tools.describe`, the tool's
one-sentence result under `└`, and, where the model's text resumes, the
dim `ops` row counting the run (`Read 2 files, ran 1 shell command`). The
repaint cadence comes from the Metal backend's `tick`: `commit` waits on a
semaphore its completion handler signals and calls back every 100 ms, and
the surface installs its poll-and-draw there (`installTick`), so a prefill
chunk repaints ten times a second instead of once. `Screen.paintFrom`
rewrites only from the first row that differs from the last frame.

An attachment is an `attachment` event after the prompt's `user` event:
the transcript keeps it as a detail of that user block and renders a dim
`└ image #1: shot.png (320×240 → 10×8 tokens)` row under the prompt. When
the event carries a preview (the chat builds one where
`tui.graphics.enabled` says the terminal draws kitty graphics, never under
tmux), the block also emits the preview's rows blank and then one raw row
that climbs over them, places the image with the cursor kept, and comes
back — the row model stays text, and the goldens never contain a sequence.
The chat caps the picture to the rows above the live region. A drop, a
`/image`, and a typed path all reach the same `Editor.attachImage`; the
chat's `dropProbe` is the editor's only view of the file system.

A mutation's diff is a header row (the path, `+N −M`), then rows of a
`dim` gutter (old and new line numbers, right-aligned), a marker cell
(`+`, `−`, or a space), and the text on its band — `diff_add` and
`diff_remove` are the accent on a dark shade of the same hue, the changed
bytes of a paired line on the brighter `diff_add_change`/`diff_remove_change`
tint — padded to the width so the band reads as one. Side by side from
`side_by_side_min_width` (96) columns, the two panes separated by the table
bar; unified below that. A theme role marked `wide_bg` drops its
background at sixteen colours, where a dark band cannot be painted, and
takes its `plain` attributes instead.

Two folds, both applied to whatever is rendered next and replayed over the
last turn: Tab folds thinking, Ctrl-O cycles the tool view
(`transcript.ToolView`: `summary`, one result row under each call;
`output`, the result's text under it as the model received it, dim, cut at
`max_output_rows`; `folded`, a call keeps its row and loses the detail
under it, the result rows, and the diff's rows). A `!` line from the editor runs through the `bash` tool and
shows as the same `Bash(cmd)` block followed by the output as an `info`
block (40 rows, then `… N more lines`); with `!` the output is also the
next user message, which the surface does not echo (`quiet_user`).

The answer's markdown is rendered once per closed block: the transcript
keeps the byte offset up to which the answer has been flushed to the
scrollback (`Answer.flushed`), hands `markdown.split` only the remainder,
and renders the closed prefix it returns; the open tail is shown raw. A
test streams a document byte by byte and counts the renders — one per
block boundary, never one per token — and a prefix fuzz feeds every byte
prefix of every fixture through `split` and `render`, asserting no error,
no control byte, a bounded row count, and no row wider than the width.

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
twelve small tasks against the playground: a tiny Python geometry package
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
summary, and the 1 MiB file. Each task is one `nuclis agent --print` turn
with a fixed seed from the committed baseline, checked by its own
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
[AGNT-13](engineering-log.md#agnt-13--the-system-prompt-as-sections-measured-the-playground-task-list-the-guidelines-that-changed-behaviour-the-instructions-file-2026-09-22).

### The agent without a terminal

`nuclis agent -p "<prompt>"` (or `--print --prompt-file <path>`) runs one
turn with no TTY: the text form streams the answer, `--json` writes every
event as one object per line, and `--session <path>` is the only way a
printed turn records anything. It is the scripting entry point today and the
harness the agent loop will be tested through in phase 2, where a stream of
JSON lines is something a test can assert on and a terminal is not.

### The agent's session files

`src/agent/session.zig` writes one append-only JSONL file per conversation
under `~/.nuclis/agent/sessions/<cwd-slug>/<stamp>_<id>.jsonl` (TERM-01 step 8).
The first line is a header (format `version`, session id, time, working
directory, the model path and the digest its pull sidecar recorded, effort,
context size); every later line is an entry carrying `id` and `parent`, so a
branch or a cancel-rewind is representable later without a migration. The
file is created at the first entry, so starting the agent and quitting writes
nothing. On load, a truncated *last* line is dropped — that is where a crash
lands — while any other unparsable line, or an unknown entry type, is a typed
error naming the line number; a file from a newer `version` is refused
outright. `/save` derives markdown from the same entries, so there is no
second transcript format.

### The agent's renderer

`src/tui/markdown.zig` turns a turn's text into pre-styled, pre-wrapped rows.
Two rules matter when reading it (TERM-01 step 6):

- **Streaming.** `markdown.split(text)` divides a partial turn into the
  blocks that can no longer change and the one still being written: a blank
  line ends a block, a heading/rule/quote/list item ends one at its newline,
  and a paragraph, a table, or an open fence keeps its block open. The agent
  renders the closed part and shows the open part as raw text, repainting
  both on every token, so styling appears block by block instead of at the
  end of the turn. A trailing newline is the end of the last block, not an
  empty one after it — which is what makes the rendered prefix of a partial
  turn a prefix of the finished one, a property a test pins byte by byte.
- **Wrapping.** `view.lines` and `view.wrapStyled` take a `Wrap` mode.
  Everything a reader reads wraps at the last space that fits (`.word`, the
  editor's rule since step 3); only rows that are truncated to one line
  anyway — the status bar, the shortcut hint — use `.character`. A word
  longer than the row still breaks, and a space that lands past the edge
  becomes the next break point instead of breaking the row, so a word ending
  exactly at the last column keeps it.

## Continuous integration and releases

Two workflows under `.github/workflows/`, both on `macos-15` (Apple Silicon),
both installing the compiler with `.github/install-zig.sh`: the version comes
from `minimum_zig_version` in `build.zig.zon` and the SHA-256 from
`.github/zig-toolchain`, verified before anything is unpacked. No marketplace
action installs the toolchain and no digest is fetched at run time — this
tree pins a digest for every artifact it downloads, and the compiler is the
one it cannot do without. A Zig upgrade edits the manifest, that file, and
the toolchain section above in one unit of work; the script refuses any
version the two do not agree on. The actions that do run
(`actions/checkout`, `actions/cache`) are pinned by full commit SHA rather
than a moving tag, and `persist-credentials` is off, so the checkout cannot
push back into the repository.

Both scripts are POSIX shell and awk. A workflow step should depend on the
tools that are on every machine and nothing else, which is why neither `jq`
nor Python appears in CI even though the repository's local tooling
(`scripts/*.py`) is written in Python.

**`ci.yml`** (push to `main`, pull requests): `zig fmt --check`, a semver
check on `build.zig.zon`'s version, `zig build test`, a
`-Dmetal=true -Doptimize=ReleaseSafe` build with `--version`/`--help`
on the result, and a `-Dmetal=false` build so the non-Metal path keeps
linking. That is `make check` minus the GPU. **Not** in CI: `test-metal`,
`verify`, `bench`, and anything that pulls a model — they need the pinned
artifacts and the real M4 Pro, and a rate measured on a virtualised GPU is a
number nobody should trust. Those gates stay local and their evidence stays
in the engineering log.

**`release.yml`** (a `v*` tag) refuses to publish, before it builds anything,
unless:

1. the tag is `vMAJOR.MINOR.PATCH[-prerelease]`;
2. the tag equals `build.zig.zon`'s `.version` — the check that catches a tag
   cut before the `-dev` suffix was stripped;
3. the installed `zig version` equals `minimum_zig_version`;
4. `CHANGELOG.md` has a section for the tag.

It then runs the same gate, builds `-Dmetal=true -Doptimize=ReleaseSafe`,
asserts the binary reports the tag's version, and publishes three assets: the
binary tarball (`nuclis-vX.Y.Z-aarch64-macos.tar.gz`), a source tarball
(`nuclis-vX.Y.Z-src.tar.gz`, cut with `git archive` from the tag), and a
`SHA256SUMS` covering both. The release notes are the tag's own section of
`CHANGELOG.md` plus a fixed footer, sliced by `.github/release-notes.sh`, so
notes and changelog cannot drift.

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
In a sandbox that cannot write Zig's default global cache, append
`--global-cache-dir /absolute/writable/cache`; local verification uses the root
`.zig-cache/global` directory.

Format changed Zig source and build files with `zig fmt`. The repository check is:

```sh
zig fmt --check build.zig build.zig.zon src/ inference/ huggingface/
```

Explicit Metal/full-model test commands must be documented when introduced.
Verify behavior using the newly built binary.

## Versioning

The version lives in the root `build.zig.zon`, the single source of truth. The
package manifests under `inference/` and `huggingface/` mirror it (the Zig
package format requires a `.version`, but the path dependencies ignore it) and
`make release` bumps all three together, refusing if they disagree. The
executable exposes `nuclis --version`, fed from the root manifest through
build options.

- **SemVer with the 0.x convention.** Before 1.0 the minor number is the
  breaking axis: breaking changes bump the minor, compatible additions and
  fixes bump the patch.
- **0.1.0** marks the spec's v0.1 acceptance (the 32K context record), measured
  in [reference/bench.md § Acceptance runs](reference/bench.md#acceptance-runs).
  The tree stays on `0.1.0-dev` until that release is tagged, then returns to
  `0.2.0-dev` in the commit after the tag.
- **A tag that exists on the remote never moves.** Once a release is published
  the rule is absolute: the fix is the next number, never a moved tag.
- **Release recipe.** `make release` runs the mechanical steps; `make release
  DRY_RUN=1` previews them without touching anything. The version is derived
  by stripping `-dev` from `build.zig.zon`, never passed in: the manifest is
  the source of truth, and `release.yml` rejects a tag that disagrees with it.
  It runs `make check`, writes the release's section into `CHANGELOG.md` from
  Conventional Commits, commits `chore(release): vX.Y.Z`, creates the
  annotated tag, then commits the next `X.(Y+1).0-dev`. It never pushes.
  1. `make verify` passes on the tree (needs the pinned models); commit any
     fix that turns up.
  2. `make release`.
  3. Push the branch and the tag to publish. Tags are annotated; sign them
     (`git tag -s`) where a signing key is configured.
- Tags move only forward; never add a tag retroactively to an earlier commit.
- Benchmark records cite the git revision, and the release tag once one
  exists.
- `CHANGELOG.md` is assembled from Conventional Commits
  starting at the first tag: `feat` → minor, `fix` → patch,
  `!`/`BREAKING CHANGE` flagged as breaking. `make changelog` writes the
  section since the previous tag locally before tagging.
- **The Zig toolchain is a separate axis.** `minimum_zig_version` pins source
  compatibility, the exact compiler is recorded in benchmark records, and Zig
  upgrades are their own unit of work.

## Commits, progress, and code explanations

Follow the commit convention in [../AGENTS.md](../AGENTS.md). As coherent
increments pass their acceptance checks, record the outcome in
[engineering-log.md](engineering-log.md) and remove the
unit from [../TODO.md](../TODO.md); keep tests, relevant documentation, and
that tracker update in the code commit.

The project is also intended to deepen the user's understanding of Zig. Add
module-level explanations of data flow and ownership, document public interfaces,
and explain subtle invariants beside the code that enforces them. Avoid comments
that merely translate obvious statements into English. Write implementation
walkthroughs as modules become stable, grounded in the working code.

## Testing

- Colocate Zig unit tests with the source they exercise.
- Use `std.testing.allocator` to catch leaks, including error-path leaks.
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
([apple-gpu.md § Registers and occupancy](reference/apple-gpu.md#registers-and-occupancy-under-dynamic-caching)).

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
[apple-gpu.md](reference/apple-gpu.md) citing the file name, and deletes
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
  the active plan lives in [../TODO.md](../TODO.md) and closed units in
  [engineering-log.md](engineering-log.md).
- Add supporting architecture decisions, benchmark reports, and operating guides
  under `docs/` when there is concrete information to record.
- Link new documents from [README.md](README.md).
- Label proposed behavior, implemented behavior, and measured results accurately.
- Update existing authoritative guidance instead of copying it into a new file.
- Use relative repository links and remove stale links when moving documents.
