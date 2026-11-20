<!-- kind: reference -->
# The agent's surface and loop

What a user can do in `nuclis agent` and what the loop does with it: the
screen, the editor and its attachments, the keys, commands, and pickers,
the transcript and status bar, the session files, and the rules of a turn.
The requirements these meet are [spec.md § 7](../spec.md#7-the-agent);
how the surface draws is [terminal.md](terminal.md), and the ideas behind
it are [agent.md](agent.md).

## Surface

- Requires a TTY; `-p`/`--print` runs one turn without one ([spec § 7.7](../spec.md#77-print-mode)).
- Startup clears the visible screen (scrollback kept), prints the welcome
  in a box when the terminal is wide enough for the wordmark, primes the
  session ([§ The loop](#the-loop)), and paints the live region right under
  the welcome: the active turn, the framed input box, a hint row, and the
  status bar.
- Completed turns are written once above the region and never repainted;
  the region is the only thing repainted, inside synchronized output.
  Insertion above the region uses a scrolling region with the top margin
  at row 1 so scrolled-off rows reach the scrollback; a terminal without
  the capability gets a cursor-up rewrite. The region follows the content:
  its top is the row after the transcript, a shorter region keeps its top
  and leaves the rows it released blank under the status bar, and a taller
  one or an insertion takes those rows back before the transcript scrolls.
  Rows that close in the region are painted out and inserted in one frame,
  so they fill the rows they free. At a turn's end the answer's last row
  sits one blank row above the input box.
- A resize replays the last turn at the new width when it is still
  wholly on screen once the terminal has reflowed it; a taller turn, and
  every older one, is left as the terminal reflowed it, with no message.
  Exit erases the region and restores the terminal, leaving only the
  transcript.
- An answer streams into the scrollback about a line at a time, so the
  live region holds the line being written and not a growing block: a
  paragraph line, heading, rule, quote, or list item goes out at its
  newline, styled, as one render of the whole answer would draw it; so
  does each line of a fenced block (its caption with the first line, the
  highlighting carried across). A paragraph is held while an inline span
  opened in it (`**`, `` ` ``, a link) waits for its closer on a later
  line. A table goes out whole once a line that is not a row ends it,
  because its column widths need every row; until then it is the one
  thing that stays in the region, tail-clamped when taller.
- Colour levels: truecolor, 256-colour, the sixteen ANSI slots, or plain
  attributes; `NO_COLOR` wins. Glyphs: Unicode, or ASCII when the locale
  is not UTF-8 or `NUCLIS_ASCII=1`. Code blocks draw no border and no
  padding (the band is erase-to-end), so a selection copies code alone. The default theme is Gruvbox dark; a theme
  supplies a palette only and can never change layout.
- The surface repaints at the engine tick's cadence through every GPU
  wait, so a prefill chunk and a long tool call both animate.

## Editor

- Enter sends; Shift-Enter (kitty keyboard protocol, requested on entry;
  Ctrl-J is the fallback) inserts a newline. Word wrap at spaces; the
  cursor row is always visible; a scrolled editor shows how many rows are
  above.
- A bracketed paste of at least 4 lines or 400 bytes becomes a chip
  (`[pasted 96 lines, 6.1 KB]`) that moves and deletes as one unit; Ctrl-E
  expands it; the text itself is what is sent. Input limit 128 KiB.
- A paste that is one file's path (a file dropped onto the window) becomes
  an attachment chip: an image (`.png .jpg .jpeg .gif .webp .heic .tiff
  .bmp .ppm`, decoded by the vision contract) is `[image #N]`, whose bytes
  in the prompt are that marker. The file is encoded at once, in the
  background while the prompt is typed, one image at a time; the chip
  shows `[image #1 · queued]`, `· encoding…`, `· 676 tokens`, or
  `· failed: <reason>`, and the turn takes the encoded rows when the file's
  digest is unchanged (encoding it at submit otherwise, and waiting for an
  encode still running). A deleted chip's encode finishes and is dropped;
  quitting waits for it. An audio clip (`.wav .m4a .mp3 .flac .aiff`,
  decoded by AudioToolbox) on a model that hears (the catalogue marks it:
  Gemma 4 E4B with its projector) is `[audio #N]`, encoded the same way,
  its chip showing `· 10.4 s → 261 tokens` when ready. A clip is measured
  at drop from its header: one over 30 s is refused with the limit named
  (`long.wav is 42.0 s: a clip is at most 30 s, and is not cut`), never cut;
  a file no decoder reads is refused naming the formats; a model that does
  not hear refuses by name (`gemma-4-12b-qat does not hear audio;
  gemma-4-e4b-qat does`). A UTF-8
  text file within the input limit is `[file name, N lines]` over its
  content, fenced and headed by the path. A typed image or clip path in a
  prompt is attached at submit: an `@path` (workspace-relative, as
  completion offers it), or a path starting `/`, `~`, or `.` (what a drop
  types on a terminal that does not bracket pastes).
  At most 8 images and 8 clips per prompt, each kind numbered in attachment
  order and renumbered on deletion; Backspace removes a chip and its file;
  Ctrl-E turns an image or audio chip back into its path. A model without a
  working projector or audio encoder refuses the chip with a notice naming
  the reason; the paste stays text.
- The transcript shows an attached image or clip as a dim detail row under
  the prompt (`image #1: shot.png (320×240 → 10×8 tokens)`, `audio #1:
  memo.m4a (12.4 s → 310 tokens)`) and, for an image on a terminal
  that draws kitty graphics (Ghostty, kitty; by environment, never under
  tmux, off with `NUCLIS_NO_PREVIEW=1`), a preview of at most 12 rows under
  it, fewer when less room stands above the input box. Sessions
  record an image's path and grid, never its pixels, and a clip's path,
  SHA-256, length, and rows, never its samples; a resumed session decodes
  and encodes each again (an image at the current cap; the recorded grid
  binds nothing), leaves a missing file out with a notice, and says when a
  clip's file changed since it was sent. A clip's digest keys the token
  cache as an image's does, so a follow-up turn reads only its new tokens.
- Up/Down move inside a multi-line input and recall history from the
  first and last rows; history persists across sessions (200 entries).
- Tab completes a `/command` or an `@path`, else folds thinking; Enter on an open command list
  runs the highlighted command, and a bare `/` is answered with the command names, never sent to
  the model. An `@word` without a `/` matches fuzzily across the workspace tree (`@faq` offers
  `docs/faq.md`; characters in order, case-insensitive, basename and word starts first; `.git`,
  `node_modules`, `zig-out`, `.zig-cache`, and `__pycache__` are not walked, at most 20,000
  entries); one with a `/` lists that directory by prefix, so Tab walks into it. The `@path`
  list closes when Tab takes a file, or when the word already names the one file it offers; a
  directory keeps it open. Ctrl-O cycles the tool rows of the last turn through three views: the
  summary (each call's row, detail, and one result row), the output (the result's text under it,
  40 rows then `… N more lines`), and folded (the call rows stay, their detail, result, and diff
  rows go). The last turn is rewritten in the new view when it is still wholly on screen; a
  taller one keeps its rows, and the status bar says the view applies from the next turn
  (`tools: folded · from the next turn`). Ctrl-T cycles the efforts the model's profile renders
  distinctly and Ctrl-W the context window (2K up to the build's ceiling, re-opening the engine), both accelerators
  for `/model`; Ctrl-N is `/clear`; Ctrl-C cancels a turn, twice quits, or quits when idle;
  Ctrl-D quits.
- Ctrl-X copies the last answer to the clipboard through OSC 52 (at most
  256 KiB); Ctrl-G opens the input in `$VISUAL` or `$EDITOR` with the
  terminal released around the child and takes the edited text back.
- A line that starts with `!` runs the rest through the `bash` tool (same
  bounds, same workspace, Ctrl-C cancels): the transcript shows a
  `Bash(cmd)` block with the output under it, bounded to 40 rows, and the
  full output goes to the model as the next message, `$ cmd` first. `!!`
  runs and shows without sending. Both forms enter the prompt history.
- The editor stays live while a turn runs. Enter *steers*: the text is
  delivered as a user message before the model's next step, after the
  tool calls in flight, and appears in the transcript and the session
  where the model saw it; at most four wait at once, and what a turn ends
  without delivering goes out as the next message. Reasoning is
  interruptible, output is not: a steer that arrives while the step has
  only reasoned (no answer text, no call) stops the step, drops it from
  the history and the session (the transcript keeps what it showed, then
  `— steering: reasoning restarted`), delivers the message, and runs the
  step again against the step budget; Ctrl-C in that window still
  cancels the turn. Alt-Enter *queues* the
  text for after the turn; a queued message is not in the transcript
  until it is sent. Print mode has no steering.
- The input box is framed; the frame's colour is the reasoning effort and
  its top edge carries the spinner while a turn runs.
- **Commands.** A line is a command only when it starts with `/` and its
  first word is ASCII letters (`/usr/bin/env` stays a prompt). A command is
  an *action* (no argument, acts now; an argument is a usage error) or a
  *chooser* (no argument opens its picker; the choice typed as the argument
  skips it). The commands are `/model [name]`, `/resume [id]`, `/compact [focus]`, `/clear`,
  and `/help`, plus `!`/`!!`; a retired command (`/new`, `/think`, `/ctx`,
  `/list`, `/delete`, `/save`, `/image`) answers with where its job went.
  A command typed during a turn waits for the turn's end instead of
  steering it. Results are one-line `  — …` notices; anything longer is an
  info block.
- **Pickers.** Every chooser uses one picker: a title with the match count,
  a search box (the `@` matcher), the list with the item in use marked,
  optional option rows adjusted with ←/→, a dim line saying what Enter
  costs, and the footer `↑↓ move · ←→ adjust · Enter choose · Esc back`;
  ↑/↓ move through the list and then the option rows. A destructive action
  asks inside the picker, and only `y` confirms. Keys are accelerators for
  commands and listed beside them in `/help`; no setting is reachable by a
  key alone.
- **`/model [name]`** offers the runnable models: the registry entries of `nuclis.json` whose
  file is present, then catalogue names no entry shadows (as `nuclis model ls` reports them),
  each with its family, quantization, and size (`gemma4 · Q4_0 (QAT) · 15.70 GB`): the
  catalogue's, or for a file outside it the GGUF header's `general.architecture` and
  `general.file_type`, read within the directory limits when the listing walks the models; a
  file whose header does not parse shows its name. The picker has option rows for the effort
  (the levels the profile renders distinctly) and the context window, at the running values for
  the running model and at the entry's (the effort carried over) for another. A name, or a query
  only one model matches, switches without the picker. Enter on another model unloads the
  engine, opens the chosen file with its entry's settings (context, speculation, draft length,
  thinking budget, sampling; backend, cache precision, output ceiling, and image cap from the
  command line hold), primes it (the token cache's disk tier serves a model used before), and
  continues the conversation rendered through the new profile, which is a full prefill; the
  conversation's images and clips are encoded again by the new model's encoders, and one it
  cannot read or hear leaves the model's view with a notice. A failed open falls back to the
  previous model. On the running model, Enter applies a new effort (re-priming) or window
  (re-opening).
- **`/resume [id]`**: an id or a prefix of exactly one id resumes it;
  otherwise the picker (also a bare `nuclis agent --resume`, at start)
  lists the workspace's sessions of any size, newest first with their
  bytes, the one in use marked (the argument is the search; none, a notice
  naming the directory); Ctrl-D deletes one after a confirmation, refusing
  the one in use. **`/clear`** starts a fresh session file; the old stays resumable.

## Transcript and status

- The agent produces typed events (user, thinking, answer, tool call, tool
  result, diff, notice, info, status, turn end); the transcript turns them
  into blocks and hands each closed block to the screen exactly once. An
  answer flushes as it streams ([§ Surface](#surface)).
- Thinking folds and unfolds without leaving the model conversation; a
  step's block closes with its own reasoning time, from its first
  reasoning token (`Thought for 6.0s`).
- While a turn runs, the live region names the step's phase, timed from
  its start: the prefill (`reading the prompt… 3s`, `reading results…
  1.2k/2.3k · 18s`, or `re-reading the conversation (the effort
  changed)…` when the step replays, by `Replay.Cause`), reasoning
  (`thinking… 6s (Tab to unfold)`), and a call being written (`writing a
  tool call… 1.4 KB`).
- A tool call is a dot and `Name(argument)`, the dot coloured by state
  (running pulses; settled well, failed, or a write), with the tool's
  one-sentence result under it; a failed call shows its message, bounded.
  A running call counts its seconds after its row (`● Bash(sleep 20) ·
  6s`), and the bar's phase cell names it (`running Bash · 6s`) in place
  of the last model phase; the settled row keeps the tool's own time
  (`└ no output · 20.0s`), left off for a call under a second. `--print
  --json` carries it as the `tool_result` event's `seconds`.
  A call Ctrl-C stopped is cancelled, not failed (dim dot, `└ cancelled
  after 6.1s · 3 lines of output`, `└ cancelled` if it never ran; the
  event and session entry carry `cancelled`; the model's text is the
  tool's). A run of calls is summed up in one dim row where the model's
  text resumes, a cancel apart (`Ran 2 shell commands, 1 cancelled`). A mutation is followed by its diff: a header with the path and
  the `+N −M` counts, a gutter of old and new line numbers, a marker cell,
  and the text on a band of its hue (dark green, dark red; the changed
  bytes on a brighter tint) padded to the width; side by side from 96
  columns and unified below that, folded when long. Sixteen-colour
  terminals keep the accent without the band.
- Answers render as markdown (headings, emphasis, inline code, links as
  hyperlinks, nested quotes, lists with task boxes, fenced code with a
  heuristic highlighter, rules, tables); a soft line break inside a
  paragraph stays a line break; model-supplied control bytes are stripped;
  no rendered row is ever wider than the terminal, whatever the input.
- An answer that rests on partial reads ends with one dim row under it,
  because the tool rows that said `cut` have scrolled away:
  `— this answer used partial reads: 5 results cut, 4 elided · /clear or
  a larger --ctx-size gives it more room`, naming only the counts that
  are not zero: results cut to the result budget, results compaction
  elided, and earlier turns dropped, all within the turn. `--print
  --json` writes the row as a `notice` event before `turn_end`, whose
  `stats` carry the three counts (`results_cut`, `results_elided`,
  `turns_dropped`); a turn whose reads were whole adds nothing.
- The status bar shows measurements on the left (state, progress, the loop step, context use,
  token counts, prefill and decode rates) and settings on the right (effort, the speculative
  switch and draft length, cache precision, backend, model). A rate that was not measured prints
  as absent; a countdown is marked as an estimate. A narrow bar drops whole cells, never part of
  one: the settings from the right (model, backend, cache precision, the switch, effort), then
  the measurements from the right (the replay, `tg`, `pp`, `in/out`); only the state cell and
  `ctx` are cut at the column when even they do not fit. The replay outlasts the settings
  because it is news of the last turn and they are not. Context use is `ctx used/ceiling`: the
  tokens the session holds against the prompt's ceiling, the window less the step's reserve
  (`ctx 7694/13312` at 16K, `low`, and a 1,024 budget), so the bar fills where compaction
  starts; `/model`'s note for the model in use names the window, the ceiling, and the reserve.
  After a cancel, until the next step, `ctx` is the conversation's size at the last step. A
  replay of 1,024 tokens or more (`status.replay_shown`) reads `re-read 2.3k tokens (after a
  cancel)`, its cause in plain words; a smaller one has no cell. `/help` explains it; the
  session file and `--print --json` keep every replay's cause and tokens prefilled and restored.

## Sessions and storage

```text
~/.nuclis/agent/
  sessions/<cwd-slug>/<timestamp>_<id>.jsonl   one file per session
  history.jsonl                                 prompt history
  exports/                                      `agent export` markdown
```

- One JSON object per line: a header (format version, id, time, working
  directory, the model's path and verified digest, effort, context size),
  then entries with `id` and `parent`: `user`, `assistant` (thinking,
  answer, tool calls, stop, stats, and on a turn's last step the
  `boundary` of its state saved in the token cache), `tool_result` (with its summary),
  `effort`, `context`, `model` (a `/model` switch: the name, file, digest,
  context, and effort the conversation continued on; readers older than
  the entry refuse the file, as any unknown type), `compaction`, `notice`, `cut` (a dropped call), `handoff`, `system` (the system block, before the first user entry).
- Append-only, created at the first entry; a resumed file's ids continue
  past its last. A file stops recording at 64 MiB, the most a resume
  reads, with a notice past 51 MiB naming `/clear`. A truncated last line is
  dropped on load; any other unparsable line or unknown entry type is a
  typed error naming the line; a newer format version is refused.
- Entries store what the model saw and produced, never terminal styling.
  `nuclis agent export` derives markdown from the same entries. A failed write is a dim
  notice, never a lost turn.
- Resuming (`/resume`, `--resume [<id>]`; print mode's default `latest`) renders
  the conversation the model last saw through the profile into a fresh
  session: cancelled steps left out and every `compaction` applied where
  it was written (`context_full`, `results_elided`, `turn_aborted`, `reasoning_dropped`, `arguments_stubbed`), from the latest `handoff`, under the latest `system` block when only its date or instructions differ (those wait for `/new`). When the
  file's last `boundary` starts that render and its state is on disk (same
  model files, backend, context size, and build), the state is restored and
  only the rest is prefilled; otherwise the whole conversation is
  ([agent.md § The agent's token cache](agent.md#12-the-agents-token-cache)).
- `agent ls` and the `/resume` picker list a workspace's sessions (the
  picker marks the one in use); `agent rm <id>` and Ctrl-D in the picker
  delete one, named by its id or a prefix of exactly one id (the picker
  asks first and refuses the session in use), together with the
  token-cache states its boundaries name. `agent export <id> [path]`
  writes one as markdown, to `path` or under `exports/`; exports stay.
- `nuclis cache` lists the token cache (each state's size and last use,
  the total against `cache.disk_bytes`); `nuclis cache clear` empties it.

## The loop

- The system block is built from sections (`src/agent/system_prompt.zig`):
  identity and workspace, the working rules, one guideline per tool under
  the rendered tool definitions, the cost rule, the date and the `$ `
  convention for the user's own shell commands, and the project's
  instructions file (`agent.instructions`: `auto` reads `AGENTS.md`, then
  `CLAUDE.md`; `off`; or a path), cut at 8 KiB with a marked cut. Every
  sentence is judged on the playground task list
  ([development.md § The agent's task list](../development.md#the-agents-task-list));
  `--system-prompt <file>` replaces the built sections for that purpose.
  The session is primed with the block before the first prompt, restored
  from the token cache when a state of it is kept; a window too
  small for it is a notice at startup, and the warm-up notice counts the
  instructions file's tokens and says where a restored prefix came from.
- Multi-turn without replay: the loop keeps the text the session has
  consumed and prefills only the increment; an effort change, a cancelled
  turn, or compaction restarts from the longest cached state the
  conversation begins with (the primed prefix or a turn's end) and
  replays the rest ([agent.md § The agent's token cache](agent.md#12-the-agents-token-cache)).
- Context budget: a step starts when its reserve is left after the
  prompt, its thinking cap at the current effort plus `answer_reserve`
  (2,048) and at most `max_tokens` (`completer.Reserve.step`), and may
  generate on to `max_tokens` while the window has room. The prompt may
  grow to the window less that reserve, its ceiling (`Window.ceiling`):
  12,288 at 16K and 28,672 at 32K at `low` with the default budget, where
  reserving all 8,192 left 8,192 and 24,576. One tool result may take a
  sixth of the ceiling in tokens (`resultBudget`, never below 256); it is
  cut at a line boundary with a note saying how to ask for the rest.
  A `bash` result is cut in the middle instead, since a failure reports
  at the end: a third of the room for its head and two thirds for its
  tail, the trailing `[bash: …]` notes (the exit status) always kept, and
  `[… N lines cut to fit the context …]` at the cut; the row says the
  same. Other tools keep the prefix, since a `read_file` page continues
  with `offset`; a cut `web_fetch` page states the characters it kept
  and the `offset` that continues it, the way a cut read restates its
  lines. A web page is asked for at the budget's size
  (`web.tools.pageChars`: 2.75 characters a token, at most 16,000), so
  that cut is rare
  ([record, 2026-11-17](../benchmarks/record/agent.md#web-pages-sized-from-the-result-budget-2026-11-17)).
  The window is 32K by default: Qwen3.8-27B with its drafter peaks at
  6.8 GB beside its 17.9 GB of mapped weights after a 26K-token prompt.
- Compaction: one decision when a step does not fit, runs into the
  window's end (dropped and run once more; a second time is
  `ContextFull`), or a turn starts past 85 % of the ceiling. Until the
  render (`Model.measure`) is at or below half the ceiling
  (`low_water_percent`), `Agent.compact` empties the reasoning of every
  step but the last two, then shortens their `write_file` and `edit_file`
  arguments to the path and a stub, then hands off (also `/compact
  [focus]`): one step without tools summarizes a condensed render, and the
  conversation becomes one message: the host's ledger (every request, the
  answer replied to, the files written, the last failing command's end),
  the summary, and the request in progress. When it cannot run, earlier
  turns go oldest first, then the turn's results become stubs, keeping the
  last two. Each cut is an entry a resume replays, announced once per kind
  (`— 3 older turns dropped from context to fit`); past them the turn ends
  naming the tokens needed. The reserve and the mark:
  [record, 2026-11-17](../benchmarks/record/agent.md#the-context-budget-2026-11-17);
  30 % or 25 % change nothing on the task list: [2026-11-18](../benchmarks/record/agent.md#how-deep-compaction-cuts-low-water-50-30-25-2026-11-18).
- Reasoning budget: one step reasons for at most `agent.thinking_budget`
  tokens at `low` (2048 by default and in every catalogue entry `config
  init` writes, 0 for no cap), twice that at `medium`, three times at
  `high`, four times at `xhigh`, and never more than the step's
  `generation.max_tokens` (8192 by default) less `answer_reserve`
  (2,048; half of a `max_tokens` below 4,096), so the answer and a long
  call keep the rest: 2,048, 4,096, 6,144, and 6,144 at the defaults.
  The default and the cap were measured on Qwen3.8 and Gemma 4:
  [record, 2026-11-17](../benchmarks/record/agent.md#the-thinking-budget-2026-11-17). Past it
  the engine closes the reasoning itself (Muse Glimmer's `<|eom|>`), as
  history renders it (Qwen3.8: a newline, `</think>`, a blank line), so
  the next step continues the session; the transcript and the session
  step say it was cut. A reasoning reopened past the budget is closed
  again after its header; the model's own late close is dropped.
  Effort is otherwise the template's own instruction, which the model may
  exceed.
- A tool call the model is still writing shows its size in the region; a
  call the step stopped inside (cancellation, the output budget) is
  dropped with a notice, never executed and never shown as answer text.
  Arguments are checked against the tool's schema before it runs, and a
  missing or mistyped one is named in the result.
- Failures are results: an unknown tool, invalid arguments, a timeout, a
  truncated call, an empty result, or an ordinary tool failure returns a
  typed result to the model. Only an inference-transport failure ends the
  turn.

## Tools

The requirements every tool meets are [spec.md § 7.6](../spec.md#76-tools).

| Tool | Contract |
| --- | --- |
| `read_file` | a line-addressed region: offset and count, 120 lines by default, at most 2,000 lines; a partial page ends with a note the model reads (its range, the total, where to continue), a whole file with none; a file over 1 MiB serves its first MiB, whole code points only, with the size stated and the shell named for the rest; `outline` lists a markdown file's headings or a source file's definitions (Python, Zig, JS/TS, Go, Rust, C-family, shell) with line numbers, at most 400; non-UTF-8 is a typed error |
| `write_file` | create or replace a UTF-8 text file, at most 1 MiB, by atomic replacement that keeps a replaced file's mode; a new file's missing parent directories are created, once the nearest existing one resolves inside the workspace and each missing name is plain (not `.` or `..`), and the result names them; `append: true` adds to the end of an existing file through the same replacement, so a large file goes in parts, with the 1 MiB bound on the result; the model reads a replacement's diff, a new file's or an append's line count only |
| `edit_file` | replace one unique, non-empty `old_string` with `new_string`, or `edits`, at most 16 such pairs applied in order to the running content, all or none; the match is tried exactly, then by whole lines ignoring trailing whitespace and CRLF (the file's CRLF kept), then ignoring indentation too (`new_string` shifted by the one indentation delta found, or refused), and must be unique at the first tier that matches, the result naming a tolerant tier; `line` (the call's, or an edit's own) picks the match starting nearest it among several and the result says which; a batch that repeats an `old_string` with no `line` takes its matches in file order, each repeat past the previous one's replacement; identical strings the file already holds are answered with their line; a miss names the most similar region of `old_string`'s line count (mean per-line LCS ratio of trimmed lines, at least 0.6), what differs (text, backslashes, indentation with both counts, line endings, or trailing whitespace) and its text, at most 40 lines and 4 KiB with the cut marked, or says there is none or the search was cut short; overlapping matches count as several, and matching that would take too long is refused; any failure, a result over 1 MiB included, writes nothing; the file's mode is kept; the model reads the diff with one line of context (the transcript shows three) |
| `glob` | one pattern (`*`, `?`, classes, `**`); hidden entries only when named; at most 200 results in stable order, of at most 10,000 collected; more, or unreadable directories, are marked; symlinks are not followed, and the ones the pattern could reach are counted (`[no files · 1 symlink skipped]`) |
| `grep` | literal, case-sensitive; `path` narrows it to a directory, a file, or a glob inside the workspace, `context` adds 0–5 lines around a match; grouped by file in path order; at most 50 matches with the total stated; at most 20,000 files and 10 s a search, a stop stated; skips hidden entries, symlinks, binary-looking files, generated trees, and what each directory's `.gitignore` excludes (a subset: negation, directory-only, anchoring, the glob wildcards) |
| `bash` | one command with a minimal environment; combined output at most 1 MiB; 300 s; cancellation, the timeout, and the output bound stop the command's process group (`SIGTERM`, then `SIGKILL` after 2 s) and reap the child; a shell that exits while a `&` job still holds the output is read 1 s more, then the job's group is stopped and the note says so; a command still running whose output has been quiet 15 s is asked of `lsof` every 5 s whether a process of its group listens on TCP, and if so is stopped as a server in the foreground, the note naming the addresses and how to run it in the background (where `lsof` is missing, the timeout stands); an output over 120 lines reaches the model as its first 40 and last 80 with the omitted count between |
| `web_search` | offered only with `agent.web` and `web.url`; one query (≤ 400 bytes) to the user's search service, optionally limited by `freshness` (`day`, `week`, `month`, `year`); at most 8 results, each title, URL, and snippet, with up to three more passages from the page, each one line of at most 300 characters; the corrected query and a skipped provider noted; 15 s; Ctrl-C cancels |
| `web_fetch` | offered as `web_search`; one `http`/`https` URL (≤ 2,048 bytes) of an HTML page or a PDF, fetched by the service, which refuses non-public addresses; its Markdown text a page at a time, sized from the result budget (5,632 code points at 16K and 13,139 at 32K with the default budget, at most 16,000), the range noted with the `offset` that continues it; or, with `question` (≤ 400 bytes, not with `offset`), only the parts that answer it, a few thousand characters; a page behind a bot challenge named as such; 30 s; Ctrl-C cancels |
| `web_context` | offered as `web_search` when the service's health lists `/v1/context`, probed once at startup; one query and optional `freshness`; passages from at most 8 pages, a 2,048-token budget, each source's title, URL, and date above its passages, at most 12,000 characters, or the page size when smaller, cut at a source boundary; 25 s; Ctrl-C cancels |

The web tools are off by default: without `agent.web` the model is shown
the six tools and the same system prompt, byte for byte. Searches and
contexts are metered upstream, so a turn makes at most six uncached ones;
past that the call answers that the budget is spent. The service and
its contract: [agent.md § 11](agent.md#11-the-web-tools-a-hosts-search-service-opt-in).

A read of lines a file returned earlier in the same turn, from unchanged
content and with that result still in the context, is answered with a
one-line pointer to it rather than the lines again (after a failed `edit_file` of the file, the lines).
