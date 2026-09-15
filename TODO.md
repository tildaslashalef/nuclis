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
source line, a 720-token tools block). Two more
units were opened on 2026-09-15 from the first playground turns: the wait
after Enter is the first turn's prefill of the system prompt and tools block
(ENGN-09), and a step that ends in a tool call never closes its thinking
block while the last one reports the whole turn's time (TERM-06).

Order: AGNT-08 (two sessions), TERM-06, ENGN-09.

| Unit | Title | Sessions |
| --- | --- | --- |
| AGNT-08 | Context budget: bounded results, in-turn elision, honest failure | 2 |
| TERM-06 | Per-step thinking blocks with their own duration | 1 |
| ENGN-09 | Primed sessions: prefill the prefix at startup, restore it on new | 1 |

## AGNT-08 — Context budget: bounded results, in-turn elision, honest failure

**Why.** `read_file` allows 2000 lines / 1 MiB and `bash` 1 MiB: about
40,000 tokens for one full read (log, AGNT-07: ~20 tokens per source line),
five times an 8K window and larger than a 32K one. `dropOldestTurn` only
drops whole earlier turns, so a single long turn cannot be rescued, and the
overflow is discovered by the engine after the prompt was built.

**Session 1 — done (2026-09-15, uncommitted evidence below; the unit closes
with session 2).** `loop.resultBudget(capacity)` = `capacity / 8`, never
below 256 tokens; `Agent.fit` counts a result through the new `Model.count`
seam (the engine's tokenizer; the test stub uses four bytes per token),
scales the byte cut by the measured density, backs up to a line boundary
(or a code-point boundary when one line is over budget), and appends
`[truncated to fit the context: A of B lines shown; continue with read_file
offset=N]` (other tools: `narrow the request for the rest`). The detail row
gains `· cut to A lines for the context`. `read_file` defaults to 200 lines.
`Completer.overflow` records what did not fit; the surface says `context
window full: the step needed N tokens (prompt plus output budget) of C;
raise it with /ctx <n> (or --ctx-size), or start over with /new`, print mode
puts the same in the error diagnostic. The status bar's prefill rate is
measured per step from the step's first beat and kept beside the decode
rate. Evidence: 373 tests; live on the playground at 8K, a 400-line read of
`data/measurements.txt` was cut to 45 lines (1,440 bytes) with `offset=46`,
and the model reported exactly that; at `--ctx-size 1024` the same turn
ended with `the step needed 2907 tokens (prompt plus output budget) of
1024`.

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

## TERM-06 — Per-step thinking blocks with their own duration

**Why.** Each completion step has its own reasoning channel, so a turn with
two tool calls shows three thinking blocks; that is right. But `thinking_end`
is only sent when answer text starts (`Agent.feed` in `src/agent/loop.zig`),
so a step that ends in a tool call never closes its block and its header
stays at "thinking", and the one block that does close measures from the
*turn's* start, so "Thought for 44.5s" includes the earlier steps, the tool
runs, and the searches.

**Design.**
- `Agent` keeps a `step_started` timestamp set in `beginStep`; the seconds
  in `thinking_end` measure from it, never from `turn_started`.
- The block closes at the first of: the reasoning channel's end (the first
  answer byte, as now), or the end of a step whose reply carried tool calls
  and no answer. Both send `thinking_end` once; `thinking_ended` guards it.
- A step with no reasoning text sends nothing, so no empty header appears.
- The session file already stores per-step `thinking`; the resume replay
  closes each block it re-applies (it passes `0` today; it keeps that, since
  no duration was stored, and the label reads "Thought" without a time when
  seconds are zero — or store `thinking_seconds` in the assistant entry's
  stats, the cleaner choice; decide at implementation and record it).

**Acceptance.** A stub-model test with two tool-call steps and an answer
step sees three `thinking_end` events, each with its own duration, in order;
a live playground turn with two tools shows three "Thought for" headers whose
sum is below the turn's elapsed time.

## ENGN-09 — Primed sessions: prefill the prefix at startup, restore it on new

**Why.** The engine is opened before the prompt appears, but nothing is
prefilled until Enter. The first turn then pays the system prompt, the
tools block, and the message — 850 to 1,200 tokens at the measured 50 to 90
tok/s (10 to 25 s), plus the cold first pass that pages the weights into the
GPU. Later turns prefill only their suffix (`Completer.increment`), so the
cost returns on every new session (`/new`, Ctrl-N) and every `/ctx` change,
which reopens the engine. The first progress beat arrives only after the
first 256-token chunk, so the screen is still until then.

**Design.**
- **Prime.** After the engine opens, render the system message with the
  tool definitions through the profile, encode, prefill it into the session
  with no sampling, and record it as the `Completer`'s `seen` text. A test
  on the profile pins that the rendering of `[system]` is a byte prefix of
  the rendering of `[system, user, …]`, which is what makes the increment
  path hit on the first turn.
- **Snapshot and restore.** Take a session snapshot (`Model.snapshot`,
  from ENGN-06) once primed; `/new` and Ctrl-N restore it and reset `seen`
  to the primed text instead of resetting the model. A `/ctx` change
  reopens the engine and primes again. The snapshot is freed with the engine.
- **Feedback.** Priming runs during startup with a status row (`warming up
  · 512/850`) driven by the prefill observer, before the prompt is enabled.
  A turn's first status beat is sent at its start with position 0 and the
  token target, so the bar shows the count and an estimate at once.
- **Bounds.** The primed prefix counts against the context like any prompt;
  a window smaller than the prefix is a typed error at startup, not a
  `ContextFull` on the first Enter.

**Acceptance.** Measured on the playground with the pinned Qwen artifact
before and after: time from Enter to the first thinking byte on the first
turn, and after `/new`; the prefill token count of the first step drops
from ~850+ to the message's own tokens. Fake-model tests for the increment
hit, the restore path, and the too-small-window error.
