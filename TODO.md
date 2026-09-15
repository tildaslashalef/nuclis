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

TERM-05 closed on 2026-09-15: a tool call is two rows (call and detail) and
the result text stays with the model and the session file. What remains is
the failure that started both units: a live `nuclis agent` turn at context
8192 read a few documents and ended in a bare `ContextFull`. AGNT-08 makes
the loop survive that; its first session bounds results by the window and
fixes the failure message, its second adds in-turn elision. The measured
sizes it relies on are in the engineering log (AGNT-07: about 20 tokens per
source line, a 720-token tools block).

Order: AGNT-08 (two sessions).

| Unit | Title | Sessions |
| --- | --- | --- |
| AGNT-08 | Context budget: bounded results, in-turn elision, honest failure | 2 |

## AGNT-08 — Context budget: bounded results, in-turn elision, honest failure

**Why.** `read_file` allows 2000 lines / 1 MiB and `bash` 1 MiB: about
40,000 tokens for one full read (log, AGNT-07: ~20 tokens per source line),
five times an 8K window and larger than a 32K one. `dropOldestTurn` only
drops whole earlier turns, so a single long turn cannot be rescued, and the
overflow is discovered by the engine after the prompt was built.

**Session 1 — bounded results and an honest failure.**
- A context-relative result budget: each tool result is capped at
  `ctx_size / 8` tokens (1,024 at 8K, 4,096 at 32K), counted with the
  model's tokenizer through the completion seam, with the host constants as
  absolute ceilings. The cut tells the model how to continue (`truncated
  after line 240; call read_file with offset=240`). `read_file`'s default
  `count` drops to 200 lines.
- When nothing fits, the transcript names the window and the flag to raise
  (`context window full (8192 tokens): raise --ctx-size or start a new
  session`); the status bar keeps the numbers from the moment of overflow
  instead of resetting to `ctx 0`.
- Fix the status bar's prefill rate, which shows `—` during agent turns
  although every step prefills.

**Session 2 — in-turn elision.** Before each step, estimate the prompt from
the last measured `prompt_tokens` plus the tokens appended since. If it
would exceed `ctx_size` minus a reserve for generation, replace the oldest
tool results of the current turn with one-line stubs naming the call and
its size, keeping the user's task message and the last two results verbatim;
only when nothing is left to elide fall back to `dropOldestTurn`. Elision
invalidates the prefix KV cache from that point and re-prefills the turn
(50–90 tok/s on the bench), so it cuts once per turn in one large step,
never one item at a time. Record the elision as a `compaction` entry.

**Acceptance.** Fake-model tests for the budget cut, the failure message,
and the elision order; a live 8K turn that reads three reference documents
completes instead of ending in `ContextFull`.
