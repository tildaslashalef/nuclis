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

Theme agreed 2026-09-27 (user): Laya (`convaiinnovations/laya`, a
ModernBERT encoder with a typed-decision head, Apache-2.0) end to end in
nuclis, behind a decision API any client can call: a person writing the
state and questions by hand, a script, or later an LLM. Why it is worth
having beside an LLM: every question re-reads its state (the encoder is
bidirectional, so nothing is shared across questions), and an LLM must
spend decode tokens writing each question, so delegating one decision the
LLM could make itself saves nothing. It pays as a **filter**: one question
fanned out over many states the LLM never reads (30 search hits, every
log section, each diff hunk), where the LLM's prefill is the cost avoided
(docs/reference/laya.md gives the measured numbers). Landed before the
plan: safetensors pull (MODL-28) and the safetensors loader (MODL-29).
Kept to three units on purpose, to iterate fast.

MODL-30 closed 2026-09-29 (engineering log): `nuclis decide` runs Laya on
the CPU end to end, matching the `laya` 0.3.20 package exactly on its 8
oracle requests (gates `laya-vocabulary`, `laya-cpu`); the `laya`
catalogue entry and `decide.model`.

Added 2026-09-29 (user), ahead of the Laya units: REPO-20, cut debugging
and verification time. The user is not happy with the gate design (one
full set per model, so the Gemma family is checked three times; 12 min
for `make verify`, hours for `make verify-cpu`) and wants the gates
re-derived from what they must protect, plus a CPU reference fast enough
to use while debugging. REPO-20 session 1 is done except the CPU
tier's baseline times (running); the gate set is approved, and session
2 applies it (sessions swapped: the threaded reference is session 3).

## Order

| Unit | Title | Sessions |
| --- | --- | --- |
| REPO-20 | Fast verification: gates re-derived from code paths, a threaded bit-identical CPU reference | three |
| MODL-31 | Laya on Metal: bidirectional windowed attention, the encoder plan, measured | one or two |
| AGNT-18 | The agent's `decide` tool: LLM-written questions over tool-supplied states (the experiment) | one |

Decisions (user, 2026-09-27): the oracle is Laya's own Python package;
the root checkpoint first (multilingual later, same family code); CPU
first, then Metal; a new command, `nuclis decide`, with three input tiers,
fan-out over many states, a styled terminal view, and `--json`; each
`results[i]` a complete Jev response; LLM-written questions are the last
unit, an experiment.

## REPO-20 — Fast verification

Goal (user, 2026-09-29): debugging and verification runs short enough to
use freely. Targets: `make verify` ≤ 4 min (704 s measured), the CPU
tier ≤ 30 min (hours today), one family's CPU trace ≤ 1 min.

### Session 1 — measured, matrix written; the set approved

Delivered: `scripts/gates.py` stamps every output line of a gate with
its time since launch (`timeline` in `--json`, `--timeline` to print)
and compiles the check tools up front (new `zig build check-tools`
step in `build.zig`) so builds are timed apart; the coverage matrix and
where the time goes are in `docs/development.md § What each gate
protects`. Baseline JSON: `.zig-cache/gates/baseline/verify.json`
(Metal, `acce06a`, 704 s + 9 s builds, 38/38) and `verify-cpu.json`
(the CPU tier, running at the end of session 1; copy its per-gate times
into the section above when it lands, and into the log on close).

Gate set approved by the user 2026-09-29, with three additions and the
sessions swapped (the gate set first: it pays on every unit; the threaded
reference helps only the rarely run CPU tier):

- **`verify`** (fast, every unit touching the inference stack), 32
  gates: `qwen38-{trace-f32,trace-f16,generation-metal,
  speculative-metal,draft-trace-metal,vision-metal,vocabulary,
  perplexity}`; `bonsai-{trace-f32,trace-f16,generation-metal}`;
  `muse-{trace-f32,trace-f16,generation-metal,draft-trace-metal,
  vision-metal,vocabulary,perplexity}`; `gemma4-e4b-{trace-f32,
  trace-f16,generation-metal,draft-trace-metal,vision-metal,vocabulary,
  perplexity}`; `gemma4-26b-a4b-{trace-f32,trace-f16,generation-metal,
  vision-metal,perplexity}`; `laya-{vocabulary,cpu}`. With four tool
  changes: (1) `--draft-trace --metal` and `checkDraft` under `--metal`
  run the plan only (the CPU halves stay in the `*-draft-trace-cpu`
  gates; about −200 s); (2) `checkChunkedPrefill` over 40 tokens instead of
  70 (chunk 32 still splits; the 64-token tile kernel is pinned by
  `test-metal`; −30 s); (3) `nuclis eval --chunks 2` against the same
  reference file's `chunk_ppl[1]` (first two windows measured 0.002 % and
  0.005 % from llama's on Qwen; −100 s; the 8-window runs move to the
  release tier); (4) `bonsai-generation-metal` without the F32-tile,
  `prefillRows`, and F16-KV sub-checks, which test the Qwen 3.5 plan's
  schedule that `qwen38-generation-metal` covers (a `--session-only`
  style flag keeping isolation, cancellation, snapshot, recovery, and the
  half-tile chunks; −30 s). Projected 225–260 s including builds.
- **`verify-release`** (new; `make release` runs it with `verify-cpu`
  and `verify-long`): the four 8-window perplexities; the Gemma 12B QAT
  file's `trace-f32`, `trace-f16`, `generation-metal`,
  `draft-trace-metal`, and its two CPU gates (it covers one global KV
  head and 48 layers, a shape through the 26B's code; its tokenizer is
  E4B's); `qwen38-draft-stats` (a report: it fails only on an error).
  `gemma4-qat-vocabulary` is removed (the same tokenizer as
  `gemma4-e4b-vocabulary`).
- **`verify-cpu`**: unchanged in content minus the 12B QAT gates;
  session 2 makes it fast.
- **Globs**: `inference/src/backends/cpu/**` replaced per file:
  `experts.zig` → the 26B-A4B gates, `hadamard.zig` → Bonsai,
  `recurrent.zig` → Qwen and Bonsai, `dense.zig` → Laya; the others
  (`root.zig`, `attention.zig`, `vector.zig`, `rope.zig`) → every CPU
  gate. The Metal side stays whole-directory (`kernels.metal` is one
  file every plan uses).

Gaps found, not fixed here: no gate covers the 12B QAT's `gemma4uv`
unified projector, the 26B-A4B's assistant head, or PQ2_0 beyond
fixtures; Metal BF16 numerics and `layerNorm`/`addBiasRows` have no
model-free test; `engine.zig` has no unit tests.

### Session 2 — apply the approved set

- The four tool changes above (`inference/generation-check.zig`
  `museDraftTrace`, `gemmaDraftTrace`, `checkDraft`,
  `checkChunkedPrefill`, a Bonsai flag; `src/eval.zig` accepting
  `--chunks` below the reference's and comparing `chunk_ppl[chunks-1]`,
  with a unit test), the manifest (`gates.json`: tiers, the new
  `verify-release` tier in `scripts/gates.py` `TIERS`, the globs), the
  Makefile (`verify-release`, `release` running it, help), and
  `docs/development.md § Gates` (tier table, counts, times),
  `AGENTS.md` § Validation and definition of done (the tier rules).
- The additions (model-free, in `inference/metal-check.zig`, so `make
  check` protects what the moved gates covered by accident): an attention
  fixture at the 12B QAT's geometry (16 query heads on one global KV head,
  width 512: four wide-decode head groups per KV head); BF16 matvec and
  matmul tiles against the CPU; `layerNorm` and `addBiasRows` against the
  CPU.
- Checks: every remaining gate passes; `make verify` and the CPU tier
  timed against the targets; the log records before and after. If
  `make verify` measures above 4 min, the next cut is the 70-token
  stepped F32 reference inside `checkChunkedPrefill` (about 8 s per
  27–30B file).

### Session 3 — a threaded CPU reference, bit-identical

- Split independent work across `Io` tasks without changing any sum's
  order, the pattern of `backends/cpu/dense.zig` (`std.Io.Group`,
  `taskCount`): `backends/cpu/root.zig` `matvec` by rows (each row still
  one F64 sum in column order; the decode scratch becomes one `columns`
  slice per task, so the callers' workspaces grow to `taskCount ×
  columns`), `attention.apply` by head, `experts.ffn` by expert, and the
  vision encoders' matvecs. The kernels take `std.Io`; the family
  runtimes (`*_runtime.zig`, `vision/*.zig` CPU paths) pass theirs.
- Proof of no change: every `*-trace-cpu` directory and logits file
  `cmp`-identical before and after, all `verify-cpu` gates passing; times
  before (the session 1 baseline) and after per gate.
- `make verify-cpu` once (this changes how the reference runs, not what
  it computes; the byte comparison is the evidence).

## MODL-31 — Laya on Metal

Where the CPU path stands (docs/reference/laya.md § On the CPU, § Time
per call): `models/laya.zig` `Laya.logits(io, gpa, ids, markers, kind,
out, trace)` over `models/modernbert_runtime.zig` and
`backends/cpu/dense.zig`; `inference.decide.Backend` is `enum { cpu }`
and `Decider.open` ignores it. CPU encode times to beat (ReleaseSafe,
M4 Pro): 56 tokens 269 ms, 164 tokens over three questions 801 ms, 512
tokens 2,582 ms; load 0.6 s. The Metal plan is checked against the CPU
forward with the `laya-cpu` bounds (relative 1e-5 per stage; the trace's
`Stage`s give the rows) and the fixtures.

- The encoder and head as a Metal plan beside the CPU runtime: F16
  weights on the GPU (existing F16 matmul/matvec kernels where they fit;
  a batched-sequence matmul is the main shape), LayerNorm without bias,
  GeGLU, and a **new attention kernel: bidirectional, full or windowed
  (`|i − j| ≤ 64`)**, over one or several sequences padded to a common
  length with a key mask (the existing kernels are causal).
- `Decider` backend `metal` by default when built with Metal;
  `--backend cpu|metal` on `nuclis decide`.
- Checks: Metal against the CPU and the fixtures (F32 accumulation;
  bounds recorded per layer); resource lifetime and cleanup tests like
  the other plans; measured on the M4 Pro: 1, 10, 50 questions at 64 and
  512 state tokens, load time separate, written into
  `docs/reference/laya.md` with hardware, build, and commit.
  `make verify` once (shared kernels touched).

## AGNT-18 — The agent's `decide` tool (experiment)

- A tool in `src/agent/tools/` calling the in-process `Decider` (loaded
  on first use, `decide.model`): one question (choice, score, or noul
  with its options) over up to 64 candidates given as workspace file
  paths or as the items of the previous tool result; returns the
  candidates ranked with probabilities, truncation marked; limits are host
  constants; failures are results.
- Its description and the system-prompt line measured on the playground
  task list before and after (`make agent-eval VARIANT=…`), on Qwen3.8-27B
  and Gemma 4 E4B: task success, wall time, and prefill tokens saved.
  Kept only if it helps; the result is logged either way.
