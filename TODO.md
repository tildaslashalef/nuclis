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

**Next: AGNT-19 session 2** (below): session 1 landed on 2026-10-04
(the cache, both tiers for the primed prefix, turn-end states in memory,
the replay accounting; see *Session 1 delivered*). The decode-speed theme (agreed
2026-09-30, toward ADR 0001's 20 tokens/s) closed with KERN-23 on
2026-10-03: Qwen's plain decode at 512 10.37 → 11.78 tok/s, 2–3-row
verify batches 6–12 % cheaper, speculative prose at 512 17.42 / 16.55
tok/s (greedy / instruct, draft 7). Its units, ledgers, and the levers
left open are in the engineering log (ENGN-18 to ENGN-20, KERN-20, KERN-21,
KERN-23, KERN-24). The base binary for `make speed` is `2f1fe98`
(`.zig-cache/speed/base/`); the saved prefixes at 512, 4K, and 32,639
exist under `.zig-cache/speed/prefix/`, 16K does not.

| # | Unit | Sessions |
| --- | --- | ---: |
| 1 | AGNT-19 — Token caching for the agent: turn-boundary snapshots in memory and on disk; `/list`, `/delete`, `agent rm` | 2 |
| 2 | AGNT-20 — A faster agent step: the command surface and `/model`, reading less, faster prefill | 3+ |

## AGNT-19 — Token caching for the agent: snapshots at turn boundaries, in memory and on disk (2 sessions)

Base: `9f45bf0`

**Why.** Prefill runs at about 90 tok/s on Qwen, and the agent re-prefills
tokens it has already computed in five places: every start (the system
block and tools, about 934 tokens, about 11 s; llm-guide.md § 22), every
`agent --print` task of `make agent-eval`, `/resume` (the whole
conversation, minutes at 16K), a cancelled step (`Completer.run` clears
`seen`, so the next step replays the conversation from the primed
prefix), and an effort change (`low` / `xhigh` rewrite the system block,
so even the primed prefix misses). Compaction and elision rewrite an
early message, so a cache saves only up to the first changed token there.
New tokens (tool results, the user's message) are not cacheable: this
unit removes re-prefill, not prefill.

**The constraint.** 48 of Qwen's 64 layers are recurrent, so a state
cannot be resumed at an arbitrary token by truncating the KV cache
(session.md § Snapshot and restore): reuse exists only where a snapshot
was taken. The cache is therefore a set of snapshots at chosen
boundaries with a longest-prefix lookup, not per-token prefix caching.

**Design.**

- **Entry.** A `Snapshot` (`inference.engine.Model.snapshot`, the drafter's
  carried row included) plus its token prefix. Key: `prefix_cache.Key`
  (model files, layout digest, token digest) extended with the **build**:
  the version and a git revision with a dirty flag, a new
  `build_options` field in `build.zig` (only `version` exists). A
  different build is a miss, never a restore: a restored state must equal
  what this build's prefill computes.
- **Boundaries.** The primed prefix, one entry per effort level, and the
  end of every completed turn (the model's final answer, before the next
  user message). Both sit at the template's control tokens, which BPE
  never merges across, so a boundary's tokens are the same whatever is
  rendered after it.
- **Lookup.** `Completer.run`'s fallback (today `restorePrimed`, else a
  reset and a full render) becomes: encode the full render, find the
  longest entry whose tokens prefix it, `restore`, and prefill the rest.
  The incremental path (`increment(seen, full)`) is unchanged; matching on
  divergence moves from text to tokens.
- **Memory tier** (in-process): cancel, compaction, effort switches, `/new`
  in the same process. Bounded by `cache.memory_bytes` (default 4 GB):
  a snapshot is 150 MB plus 64 KiB per token on Qwen (about 1.2 GB at
  16K), so it holds the last few turns, oldest evicted first.
- **Disk tier** (`<NUCLIS_HOME>/cache/prefix/`, `src/paths.zig`): start,
  `agent --print`, `/resume`. Writes through `prefix_cache.save` / `load`
  (header with all keys, content hash; a corrupt file is refused and
  deleted). Bounded per file (the used extent) and by `cache.disk_bytes`
  in `nuclis.json` (default 8 GB; 0 disables), evicted oldest-accessed
  first after each save.
- **Engine.** Only `snapshot`, `restore`, and `prefix_cache`, which the
  existing gates already cover. A cheaper turn checkpoint (copy only the
  150 MB recurrent state and rewind attention by position) is a new engine
  API: its own ENGN unit, only if the memory tier's copies measure too
  slow.
- **Out of scope.** Arbitrary-position reuse, sharing across models or
  capacities (a `/ctx` change misses), and making compaction itself
  cache-friendly (a loop behaviour change; a later AGNT unit if the replay
  counts below show it matters).

**Session 1: the cache and the memory tier.** A new `src/agent/cache.zig`
(pure logic: entries, longest-prefix lookup by tokens, budget and
eviction; an injected store so tests need no model), wired into
`Completer` (`src/agent/loop.zig`) in place of `primed`. The status
line's `replayed` gains the cause and the tokens re-prefilled (start,
resume, cancel, compaction, effort), so the gains are counted, not
estimated. The disk tier for the primed prefix lands here too (start and
`agent --print`); the build revision in `build.zig`.

*Session 1 delivered (2026-10-04).* `src/agent/cache.zig` (`Memory`,
`Disk`, `modelKey`, `buildId`, `openDisk`; four tests), `Completer` on it
(`prime` returns `Primed{tokens, from}`; `run` restores the longest
cached state, even over a session that could continue; `checkpoint`, a
new `Model` hook the loop calls when a turn ends in an answer;
`reset(cause)`, `dropCache`), `loop.Replay{cause, restored}` through
`TurnStats.replay` to the bar and the session file (`replay`,
`restored_tokens`), `cache.memory_bytes` / `cache.disk_bytes` in
`src/config.zig`, `paths.prefixCachePath`, `revision` and `dirty` in
`build.zig`'s options. Design changes from the text above, recorded in
[session.md § The agent's token cache](docs/reference/session.md#the-agents-token-cache):
matching is on rendered text, not tokens (generated tokens need not be
the canonical encoding), states holding an image are not kept, a dirty
build adds the executable's size and mtime to its key, and the key
stays `prefix_cache.Key` with the build folded into its model half (no
engine change). Measured: warm-up 16.3 s cold, 1.0 s from disk (the
prediction said ≤ 0.5 s; reading 250 MB is the rest), 0.4 s from
memory; cancel and effort replays restore 1,369 and 1,427 tokens
(`.zig-cache/tui/c1`–`c8`); cold and warm greedy text identical with and
without the drafter. Not yet measured: the turn-end snapshot's cost at
16K (a 1.2 GB copy on the turn's path), to time in session 2 alongside
`/resume` at 16K. `make agent-eval` "before" runs against the `Base:`
binary (build `9f45bf0` in a worktree), "after" against the closing one.

*Fixes from the user's testing, inside this unit (user, 2026-10-04: what
their playground testing turns up is investigated and fixed here, no
separate units).*
- `bash` always failed in the live agent (`could not start a shell:
  Unexpected`; a debug build panics on `EBADF`): Zig 0.17's Darwin spawn
  passes `.cwd = .{ .dir = Dir.cwd() }` to
  `posix_spawn_file_actions_addfchdir_np` as `AT_FDCWD`, which it refuses.
  Broken since the Zig 0.17 migration (REPO-29, shipped in v0.4.0); the
  tests used a temporary directory, a real descriptor. The child now
  inherits the process's directory when the workspace is it
  (`src/agent/tools/bash.zig`), with a test from `Dir.cwd()`.
- A second prefill after a step that read five files was the results
  (3,637 new tokens), not a replay: expected, nothing to fix.
- `@` completion is fuzzy for a word without `/` (`commands.completePaths`,
  `fuzzyPaths`, `fuzzyScore`; two tests; `.zig-cache/tui/f1-faq`,
  `f2-poly`, `f3-accepted` in `~/Code/playground`).

**Session 2: `/resume` and the surface.** The session writer
(`src/agent/session.zig`) saves the last completed turn's snapshot to the
disk tier keyed by its tokens; `/resume` and `agent --resume` look it up
and prefill only the remainder, falling back to the replay on a miss.
`nuclis cache ls` / `nuclis cache clear` (APPS surface, help and
completion). Session management (user, 2026-10-04), since a deleted
session must take its disk snapshot with it:

- `/list`: prints `resume.Listing` (as `nuclis agent ls`) into the
  transcript, read-only, the current session marked.
- `/delete [id]`: an id or a unique id prefix (an ambiguous prefix is a
  usage result listing the matches); no argument opens the `/resume`
  picker in delete mode. Asks y/n; refuses the current session. Removes
  the JSONL file and the session's disk-tier snapshot, never `/save`
  exports.
- `nuclis agent rm <id>`: the CLI counterpart of `agent ls`, same
  resolution and refusal rules, no model load.

One delete function (in `src/agent/resume.zig` beside `find`) serves
both; `/list` and `/delete` are rows of `commands.zig`'s table, so help
and completion follow; `agent rm` goes into `src/cli.zig` and
`src/completion.zig`. Docs: `docs/reference/session.md` (the cache),
`docs/spec.md` § Sessions and storage and the CLI synopsis,
`docs/development.md` § User directories, the agent's help.

**Predictions.** First prompt after start ≤ 0.5 s instead of about 11 s
(warm disk entry); a cancel at 16K resumes the next step in ≤ 2 s
instead of a full replay; `/resume` of a 16K conversation ≤ 2 s instead
of minutes; `make agent-eval` model time per task down by about the
primed prefill (about 11 s).

**Correctness.** Unit tests in `cache.zig`: longest-prefix choice, a
build-key miss, budget eviction in both tiers, a corrupt file refused.
Unit tests for session management: `/list` and `/delete` parsing, an
ambiguous prefix, refusing the current session, the snapshot removed
with its session, `agent rm` argument errors.
One hand-run end-to-end check on Qwen (not a gate): a fresh and a
restored session give the same greedy tokens for one prompt, with and
without the drafter. `make shot`: a cold start, a warm start showing the
restore, a cancel then a prompt, a `/resume`, a `/list`, a `/delete`
(picker and by id).

Gates: `make check`, `make lint-py`, `make shot`, `make agent-eval
VARIANT=…` (once before and once after, session 2). No inference gate
tiers (user, 2026-10-03): this is agent work, so `make verify-auto`'s
Metal gates, `make verify`, and the CPU, long, and release tiers do not
run for it; the unit uses only engine APIs those tiers already cover, and
an engine change it turns out to need becomes its own ENGN unit.

## AGNT-20 — A faster agent step: the command surface and `/model`, reading less, faster prefill (3 sessions, the third may take more)

Agreed with the user 2026-10-04 as one unit (no separate units for its
parts). Starts when AGNT-19 closes; its section gets ``Base: `<rev>` ``
then.

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

**Session 1: the command surface and `/model`.** Files:
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

**Session 2: reading less.** Files: `src/agent/tools/grep.zig`,
`read_file.zig`, `edit_file.zig`, `bash.zig`, `src/agent/loop.zig` (the
repeated-read check), `src/agent/system_prompt.zig`, a new
`scripts/agent-tokens.py` (the per-tool table above for any
`agent-eval` variant), `scripts/agent-eval.py` (one more task).

- **Measure first**: `scripts/agent-tokens.py <variant>` prints, per
  tool, calls, result tokens (the engine's tokenizer through `nuclis
  tokenize`, not a character estimate), and share; run it on
  `agnt19-after` as this session's baseline. Add a task in the user's
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

