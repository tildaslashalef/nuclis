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

**Next: AGNT-20 session 2, reading less** (below; the baseline is
measured and committed, the tool changes are next). Session 1 (the
command surface and `/model`) is delivered and committed (2026-10-04):
see its section for what landed and what it measured. AGNT-19 (the token
cache, session management, and the fixes from the user's testing) closed
on 2026-10-04: the task list's wall time per task fell 23 % (the primed
prefix restored instead of prefilled), a 9.8K-token resume 126.5 → 2.05 s;
its "after" task-list run (`.zig-cache/agent-eval/agnt19-after/`) is
AGNT-20 session 2's baseline. The decode-speed theme closed earlier with
KERN-23 (Qwen plain decode at 512 11.78 tok/s, speculative prose about
17 tok/s); the base binary for `make speed` is `2f1fe98`
(`.zig-cache/speed/base/`).

| # | Unit | Sessions |
| --- | --- | ---: |
| 1 | AGNT-20 — A faster agent step: the command surface and `/model`, reading less, faster prefill | 3+ |

## AGNT-20 — A faster agent step: the command surface and `/model`, reading less, faster prefill (3 sessions, the third may take more)

Base: `6bfae8c`

Agreed with the user 2026-10-04 as one unit (no separate units for its
parts). What the user's testing turns up during it is fixed inside it
(user, 2026-10-04).

**Why.** With speculation Qwen decodes at about 17 tok/s, and an agent
step now waits mostly on prefill of *new* tokens, which the token cache
cannot remove: in the AGNT-19 "before" task-list run (24 runs, seeds 1
and 2) steps prefilled 20.5K tokens, about 13K of them tool results
(63 %): `read_file` 54 % of those (37 calls), `edit_file` 20 % (25 calls,
its confirmation diff), `bash` 15 %, `grep` 6 % (5 calls: the model
reads rather than greps), `write_file` 5 %, `glob` under 1 %. In the
user's own repository one step that read five files prefilled 3,637
tokens at about 67 tok/s, about 54 s. Two levers multiply: fewer tokens
read (session 2) and faster prefill per token (session 3). Session 1 is
the command surface the user asked for in the same discussion.

**Session 1: delivered 2026-10-04.** `src/tui/picker.zig` (the one
chooser: title and count, fuzzy search over `src/tui/fuzzy.zig`, current
mark, ←/→ option rows in the ↑/↓ focus ring, a cost note, in-picker
`y`-only confirmation; 6 tests), the command table rewritten
(`/model [name]`, `/resume [id]`, `/clear`, `/help`; actions refuse an
argument; retired commands answer with where their job went),
`Profile.efforts()`/`nearestEffort` per profile (Qwen off/low/medium/xhigh,
Gemma 4 off/low/medium, Muse low…xhigh; Ctrl-T cycles only those),
`model.runnableModels` (registry entries with a present file, then
unshadowed present catalogue names), the agent's model half as `Open`
(engine, vocabulary-sized buffers, paths, digest) replaced in place by
`switchModel` for `/model` and Ctrl-W alike (fallback to the previous model
on a failed open; memory tier dropped, disk tier re-keyed;
`Agent.reencodeImages` re-runs the new projector over the conversation's
images), the session `model` entry, `nuclis agent export <id> [path]`
(replacing `/save`), spec § 7.2/7.4, session.md, help, completion. Fixes
from the session's own runs: a `/command` or `!` line typed during a turn
is queued for the turn's end instead of being steered into the model; a
typed `@path` to an image attaches (it did not; only `/`, `~`, `.` paths
did). Evidence (`make shot`, `.zig-cache/tui/`): `model-open`,
`model-filtered`, `switch-picker`, `switched`, `gemma-answer` (Gemma
recalled the Qwen turn's prompt; `replayed model 34`), `back` (Qwen's
primed prefix restored from disk in 0.1 s), `t1`–`t3` (Ctrl-T on Gemma:
medium, off, low), `ctxw` (Ctrl-W to 32K through the same path),
`resume-open`, `resume-confirm`, `resume-deleted`, `resumed`, `cleared`,
`queued-command`, `after-turn`, `at-image`. `make verify-auto` passed
(fmt, unit, the 11 fast gates its paths selected). Remaining limitations,
for the log at close: `/model`'s option rows guess an uncatalogued
registry file's profile (`qwen38` unless the entry forces one) until it is
opened, after which the effort is clamped to the real profile; a resume
replays through the running model, whatever `model` entries the file
holds; command-line sampling flags do not carry across a switch.

**Session 1 (as planned): the command surface and `/model`.** Files:
`src/agent/commands.zig` (the table), `src/agent/root.zig` (pickers,
`runCommand`, the engine reopen that `/ctx` uses today), `src/tui/choice.zig`
(the picker), `src/help.zig`, `src/completion.zig`, `docs/spec.md`
(§ Editor, the agent's commands), `src/agent/session.zig` (a `model`
entry).

- **Interface rules**, written into `docs/spec.md` and enforced by the
  command table: a command is an *action* (no argument, acts now:
  `/clear`, `/help`) or a *chooser* (no argument opens a picker; the same
  choice typed as the argument skips it: `/model qwen`, `/resume 1a2b`).
  Every chooser uses one picker component: a title, a fuzzy search box
  (the `@` matcher, `commands.fuzzyScore`), the list with the current item
  marked, optional option rows adjusted with ←/→, and the footer `↑↓
  move · ←→ adjust · Enter choose · Esc back`; destructive actions confirm
  inside it. Keys are accelerators for commands, listed beside them in
  `/help`; no setting is reachable by a key alone. Results are one-line
  `  — …` notices; anything longer is an info block.
- **`/model [name]`**: the runnable models of `nuclis.json` (registry
  entries and catalogue names whose files are present, as `nuclis model
  ls` reports them; `config.Models`, `catalog.zig`), the current one
  marked; option rows for the effort (only the levels the model's profile
  supports) and the context window. Enter unloads the engine cleanly,
  opens the chosen file with that entry's settings (context, speculation,
  draft length, thinking budget), primes (the token cache serves a model
  used before), and continues the conversation rendered through the new
  profile, which is a full prefill: the picker says so before Enter. The
  session file records a `model` entry (a new entry type: older readers
  refuse it; note it in session.md and spec § 7.4).
- **Removed**: `/think` (Ctrl-T and `/model`), `/ctx` (Ctrl-W and
  `/model`), `/save` (becomes `nuclis agent export <id> [path]`, beside
  `agent ls` / `agent rm`), `/list` and `/delete` (folded into `/resume`:
  Enter resumes, Ctrl-D deletes with the in-picker confirmation, Esc
  leaves). **Renamed**: `/new` → `/clear` (drops the conversation; the old
  session stays resumable; Ctrl-N stays its key). **Kept**: `/resume`,
  `/model`, `/clear`, `/help`, `!`/`!!`, and `/image` only if `@image.png`
  does not already attach an image (check first; drop it if it does).
- **Tests**: parsing and the table; the picker's filtering and option
  rows; the model list from a fixture `nuclis.json`; `agent export`.
  `make shot`: `/model` open, filtered, an effort change, a switch to
  another model and back (Gemma 4 12B QAT and Qwen are both local), the
  `/resume` picker deleting a session, `/clear`.

**Session 2: in progress.** Measured first (committed with the
script): `scripts/agent-tokens.py`, and the task `about` ("What is this
project about?") added to `scripts/agent-eval.py`. Baseline variant
`agnt20-base` (`.zig-cache/agent-eval/agnt20-base/`: `agnt19-after`'s
24 runs, identical calls and results to `agnt19-before` since the cache
changes no decision, plus `about` on `1007b08`; 11/13 tasks on every
seed): 116 steps, 22,678 prompt tokens, 18,067 of them tool results
(80 %): `read_file` 48 calls 10,041 (56 %), `bash` 28 / 3,189 (18 %),
`edit_file` 25 / 3,154 (17 %), `grep` 5 / 820, `write_file` 4 / 775,
`glob` 6 / 88. Facts that correct the design below: `read_file` adds no
line numbers (the `00001` was `big.txt`'s content), and the model never
sees a tool's `summary` (only the transcript does): a read cut at 200
lines, or a grep cut at 200 matches, is not marked to the model. No file
in the baseline is re-read unchanged (the two re-reads follow the model's
own edit), so the repeated-read check saves nothing on this list. The
largest results: `history.md` page one, 1,163 tokens (both seeds); a
30-line sample of `big.txt`, 644.

Fix from the user's testing (2026-10-04): Enter on the open `/` list
sent `/` to the model; it now runs the highlighted command (`/` Enter
opens `/model`'s picker, `/cl` Enter clears), and a bare `/` is a notice
listing the commands. Evidence: `slash-open`, `slash-enter`, `slash-cl`,
`slash-bare`.

**Session 2 (as planned): reading less.** Files: `src/agent/tools/grep.zig`,
`read_file.zig`, `edit_file.zig`, `bash.zig`, `src/agent/loop.zig` (the
repeated-read check), `src/agent/system_prompt.zig`, a new
`scripts/agent-tokens.py` (the per-tool table above for any
`agent-eval` variant), `scripts/agent-eval.py` (one more task).

- **Measure first**: `scripts/agent-tokens.py <variant>` prints, per
  tool, calls, result tokens (the engine's tokenizer through `nuclis
  tokenize`, not a character estimate), and share; run it on
  `agnt19-after` (`.zig-cache/agent-eval/agnt19-after/`) as this
  session's baseline. Add a task in the user's
  style: "what is this project about" in a workspace with a long history
  file (the playground has `docs/history.md`, 431 lines).
- **`grep`**: `context` (0–5 lines around a hit), `path` (a directory or
  a glob to search under), results grouped by file (the path once), 50
  matches by default with `N more` marked (200 today), `.gitignore`
  entries skipped (native, not `rg`: the search costs milliseconds, the
  result's tokens are the cost, and one rule on every machine keeps the
  task list reproducible).
- **`read_file`**: line numbers only as wide as the file needs (today
  `00001 ` on every line); a 120-line default page (200 today) whose
  header always gives the total (`lines 1–120 of 431`); `outline: true`
  for markdown headings and code definitions with their line numbers.
- **A repeated read is not paid twice**: a read of a range whose file is
  unchanged since an earlier step of the same turn returns `unchanged
  since step N (lines a–b)`; the earlier text is still in the context.
- **`edit_file`**: the confirmation diff with 1 line of context (3
  today), or the changed lines only; measured, since the diff may be what
  catches a wrong edit.
- **`bash`**: a long output keeps its head and its tail with `N lines
  omitted` between (test failures are at the end).
- **System prompt**: one rule, locate with `grep` before reading, read
  only the lines needed.
- **Judged** by `make agent-eval VARIANT=…` per lever group against
  `agnt19-after`: pass rate not lower (11/12 tasks on every seed today),
  tool-result tokens down ≥ 30 % (a target, not a measurement), steps not
  up by more than one per task, wall seconds per task down. Changes to the
  system prompt or a tool description re-pin the prompt text only with
  the new table (AGENTS.md § Validation, item 5).

**Session 3 (or more): faster prefill.** A design that waits on facts:
the session reads them first and rewrites this part at file and kernel
level, committed before any code.

- **Facts to read**: `bench --raw --prompt-file … --ctx-size 16384` on
  Qwen at prompts of 512, 2K, 4K, 8K, and 16K tokens (prefixes of
  `docs/architecture.md` and `docs/spec.md`, as bench.md's prefill tables
  use), prefill tok/s each; `bench --profile` at 4K and 16K for the
  per-kernel split (matmul tiles per quant type, attention, the DeltaNet
  chunk, norms); llama.cpp's prefill at the same lengths as the oracle
  (reference-baseline tooling, scripts/reference-baseline.py). Known so
  far: 88 tok/s at 512 (2026-09-19, llama.cpp 89), 67–70 tok/s at 3–4K in
  the agent's bar, 32K whole-prompt 640 s (about 51 tok/s); KERN-16's
  attention rewrite gained only 2–5 % (the loads it removed were not the
  limiter).
- **Then** the largest kernel share at 4K first, one change per
  measurement, `make speed`-style A/B against a saved base, each kept
  change committed with its number; a negative result written down.
- **Gates**: `make verify` (the fast Metal tier) once, since this touches
  the inference stack; `make verify-long` if attention or the KV cache
  changes; the CPU tier only if a CPU reference's arithmetic changes.

**Close**: the agent-eval comparison (tokens, steps, wall seconds, pass
rate) and the prefill table (before → after at each length) in the log;
`docs/reference/bench.md` gets the prefill measurements,
`docs/reference/` the tools' new contracts.

