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

Nothing is in progress. The Laya theme (agreed 2026-09-27) landed:
`nuclis decide` on the CPU and on Metal, the English and multilingual
checkpoints, matching Laya's own package (MODL-30, MODL-31, MODL-33).
Beside it closed REPO-20 to REPO-23 and REPO-25 (fast gate tiers, a
generated agent playground, typed Python scripts, `.reference/`, the
README's Laya section, gates that skip an absent model) and KERN-19 (the
CPU reference on every core, bit-identical: the CPU tier in 22 minutes).
All in the engineering log.

Deferred (user, 2026-09-29), until the user picks it up: AGNT-18, the
agent's `decide` tool, the Laya theme's last unit (its design below).
A fresh session asks what to work on; it does not start AGNT-18 on its own.

## AGNT-18 — The agent's `decide` tool (experiment) — deferred

Deferred 2026-09-29 (user) until picked up. When it is, its baseline is
re-measured on the generated playground (REPO-21) before the tool lands.

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
