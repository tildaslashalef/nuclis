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

Reordered 2026-09-29 (user): Laya end to end first, then KERN-19 (the
threaded CPU reference), its "before" times not yet taken. MODL-31
closed 2026-09-29 (engineering log): Laya on Metal, `nuclis decide
--backend cpu|metal` (metal by default), packed batches, 13–20× the CPU
(a 512-token state in about 0.2 s), gate `laya-metal`. Next is AGNT-18,
the agent's `decide` tool; `Decider` opens on Metal in the agent's
process beside the text model, taking the GPU in turn.

## Order

| Unit | Title | Sessions |
| --- | --- | --- |
| AGNT-18 | The agent's `decide` tool: LLM-written questions over tool-supplied states (the experiment) | one |
| KERN-19 | A threaded CPU reference, bit-identical: the CPU tier ≤ 30 min | one |

Decisions (user, 2026-09-27): the oracle is Laya's own Python package;
the root checkpoint first (multilingual later, same family code); CPU
first, then Metal; a new command, `nuclis decide`, with three input tiers,
fan-out over many states, a styled terminal view, and `--json`; each
`results[i]` a complete Jev response; LLM-written questions are the last
unit, an experiment.

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
