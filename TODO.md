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

Added 2026-09-29 (user), ahead of the Laya units: cut debugging and
verification time. REPO-20 closed 2026-09-29 (engineering log): the
gates re-derived from what they protect (`make verify` 704 s → 246 s,
a `verify-release` tier) and `make verify-auto`, which runs the checks
and gates a diff selects and names the tiers it requires.

Reordered 2026-09-29 (user): Laya end to end first, MODL-31 then AGNT-18;
KERN-19 (the threaded CPU reference) follows them, its "before" times
not yet taken. Next is MODL-31, Laya on Metal.

## Order

| Unit | Title | Sessions |
| --- | --- | --- |
| MODL-31 | Laya on Metal: bidirectional windowed attention, the encoder plan, measured | one or two |
| AGNT-18 | The agent's `decide` tool: LLM-written questions over tool-supplied states (the experiment) | one |
| KERN-19 | A threaded CPU reference, bit-identical: the CPU tier ≤ 30 min | one |

Decisions (user, 2026-09-27): the oracle is Laya's own Python package;
the root checkpoint first (multilingual later, same family code); CPU
first, then Metal; a new command, `nuclis decide`, with three input tiers,
fan-out over many states, a styled terminal view, and `--json`; each
`results[i]` a complete Jev response; LLM-written questions are the last
unit, an experiment.

## MODL-31 — Laya on Metal

Base: `26eccf1`

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

## KERN-19 — A threaded CPU reference, bit-identical

Split from REPO-20 at its close (user, 2026-09-29), which delivered the
fast gate tiers and `make verify-auto` (engineering log). Goal: the CPU
tier ≤ 30 min (hours today) and one family's CPU trace ≤ 1 min, with
not one bit of the reference's output changed.

Base: to be recorded when it starts (`d46e907` was recorded before the
reorder; take the commit before its first change).

- First: `python3 scripts/gates.py --tier verify-cpu --json >
  .zig-cache/gates/kern19-before.json` in the background, the per-gate
  "before" times (14 gates, hours; nothing else running meanwhile, since
  compiles and GPU runs distort it), then copy
  `.zig-cache/gates/trace/*-trace-cpu/` to `.zig-cache/gates/kern19-before/`
  for the byte comparison.

- Split independent work across `Io` tasks without changing any sum's
  order, the pattern of `backends/cpu/dense.zig` (`std.Io.Group`,
  `taskCount`): `backends/cpu/root.zig` `matvec` by rows (each row still
  one F64 sum in column order; the decode scratch becomes one `columns`
  slice per task, so the callers' workspaces grow to `taskCount ×
  columns`), `attention.apply` by head, `experts.ffn` by expert, and the
  vision encoders' matvecs. The kernels take `std.Io`; the family
  runtimes (`*_runtime.zig`, `vision/*.zig` CPU paths) pass theirs.
- Proof of no change: every `*-trace-cpu` directory and logits file
  `cmp`-identical before and after (`cmp -r` against
  `.zig-cache/gates/kern19-before/`), all `verify-cpu` gates passing;
  times before and after per gate.
- `make verify-cpu` once (this changes how the reference runs, not what
  it computes; the byte comparison is the evidence).
