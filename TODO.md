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
to use while debugging. Start with REPO-20, session 1.

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
use freely. Targets, to confirm or revise in session 1: `make verify`
≤ 4 min (722 s on 2026-09-29), the CPU tier ≤ 30 min (hours today), one
family's CPU trace ≤ 1 min.

Facts (2026-09-29, `make verify` at `792c47f`, 43 gates then, 38 now):
the Metal tier's time is Muse 303 s (its draft trace alone 181 s), Qwen
188 s, Gemma QAT 72 s, Bonsai 61 s, E4B 42 s, 26B-A4B 39 s, Laya 16 s.
By kind, the `*-generation-metal` checks (session isolation, reset,
snapshot, prefill against per-token steps) cost 10–81 s each and the
`*-perplexity` checks 11–55 s each; traces are 0.4–10 s. Every gate is
its own process that loads its model. The CPU reference is single
threaded and exact by design: Gemma 4 E4B prefills 19 tokens in 67 s and
decodes at 3–4 s per token (Metal: 0.16 s, 18 ms per token). Its kernels
are few call sites: `backends/cpu/root.zig` `matvec` (F64 sums per row),
`attention.apply` per head, `experts.ffn`, `recurrent.delta*`, and the
vision encoders' matvecs.

### Session 1 — measure and re-derive the gate set (no gate removed yet)

- Baselines: `make verify-cpu` per-gate times (run it in the background,
  once) and `make verify` per-gate times split into build, model load,
  and check (add the split to `scripts/gates.py` output if it is not
  there).
- A coverage matrix, written into `docs/development.md § Gates`: for each
  gate, the code paths only it exercises. The axes are what can break
  independently: each kernel × weight encoding used by a catalogue file,
  each schedule feature (DeltaNet, windowed and global attention, shared
  KV layers, per-layer embeddings, mixture of experts), each session
  layout (for isolation, reset, snapshot), each prompt profile and
  tokenizer, each vision projector, each draft source, the long-context
  paths. Read each gate's command and what its check tool actually
  compares.
- From the matrix, the proposed set, on these rules: a gate stays only if
  it covers a path no cheaper gate covers; a family is checked once on
  its smallest representative file, and a variant file only for the paths
  it alone has (for Gemma, the questions are what the 12B QAT file covers
  that E4B does not, and what 26B-A4B adds beyond its experts);
  whole-file acceptance of every catalogue entry moves to the release
  checks; engine-level properties (session isolation, cancellation)
  are checked once per session layout, not once per model. Also: gates
  of one model share one process and one model load where the tools
  allow it; why Muse's draft trace takes 181 s and whether fewer steps
  prove the same thing; whether `verify-changed`'s path globs select
  precisely (today `backends/cpu/**` selects every CPU gate).
- End the session by presenting the proposed set, its projected times,
  and what each removed gate's coverage moves to; the user approves it
  before session 3 removes anything. Rewrite sessions 2 and 3 below with
  the findings.

### Session 2 — a threaded CPU reference, bit-identical

- Split the reference's independent work across `Io` tasks without
  changing any sum's order: `matvec` rows (each row still one task's F64
  sum in column order), attention heads, the experts of `experts.ffn`,
  the vision encoders' matvecs. The kernels take `std.Io` (as
  `backends/cpu/dense.zig` does); the family runtimes pass theirs.
- Proof of no change: every CPU trace directory and logits file byte
  identical before and after (`cmp` over the `*-trace-cpu` outputs), all
  `verify-cpu` gates passing; times before and after per gate.
- `make verify-cpu` once (this changes how the reference runs, not what
  it computes; the byte comparison is the evidence).

### Session 3 — apply the new gate set

- Remove and merge gates as approved, regroup the tiers (a fast default
  tier and the release checks), fix the path globs, and update
  `docs/development.md § Gates`, the tier rules in `AGENTS.md`
  (§ Validation and definition of done), and the Makefile help.
- Checks: every remaining gate passes; `make verify` and the CPU tier
  timed against the targets; the log records before and after.

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
