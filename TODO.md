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
