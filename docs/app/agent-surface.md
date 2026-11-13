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
  session ([§ The loop](#the-loop)), and anchors the live region at the bottom: the active
  turn, the framed input box, a hint row, and the status bar.
- Completed turns are written once above the region and never repainted;
  the region is the only thing repainted, inside synchronized output.
  Insertion above the region uses a scrolling region with the top margin
  at row 1 so scrolled-off rows reach the scrollback; a terminal without
  the capability gets a cursor-up rewrite. The region's bottom is the
  anchor; blank rows between the transcript and the region are slack that
  insertions fill and a growing region takes back before anything scrolls.
- A resize replays the last turn at the new width; older turns are left
  as printed. Exit erases the region and restores the terminal, leaving
  only the transcript.
- Colour levels: truecolor, 256-colour, the sixteen ANSI slots, or plain
  attributes; `NO_COLOR` wins. Glyphs: Unicode, or ASCII when the locale
  is not UTF-8 or `NUCLIS_ASCII=1`. Code blocks draw no border, so a
  selection copies code alone. The default theme is Gruvbox dark; a theme
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
- Tab completes a `/command` or an `@path`, else folds thinking; Enter
  on an open command list runs the highlighted command, and a bare `/` is
  answered with the command names, never sent to the model. An
  `@word` without a `/` matches fuzzily across the workspace tree (`@faq`
  offers `docs/faq.md`; characters in order, case-insensitive, basename and
  word starts first; `.git`, `node_modules`, `zig-out`, `.zig-cache`, and
  `__pycache__` are not walked, at most 20,000 entries); one with a `/`
  lists that directory by prefix, so Tab walks into it. Ctrl-O
  cycles the tool rows of the last turn through three views: the summary
  (each call's row, detail, and one result row), the output (the result's
  text under it, 40 rows then `… N more lines`), and folded (the call rows
  stay, their detail, result, and diff rows go); Ctrl-T cycles the efforts
  the model's profile renders distinctly and Ctrl-W the context window (2K
  to 32K, re-opening the engine), both accelerators for `/model`; Ctrl-N
  is `/clear`; Ctrl-C cancels a turn, twice quits, or quits when idle;
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
  skips it). The commands are `/model [name]`, `/resume [id]`, `/clear`,
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
- **`/model [name]`** offers the runnable models: the registry entries of
  `nuclis.json` whose file is present, then catalogue names no entry
  shadows (as `nuclis model ls` reports them), with option rows for the
  effort (the levels the profile renders distinctly) and the context
  window, at the running values for the running model and at the entry's
  (the effort carried over) for another. A name, or a query only one model
  matches, switches without the picker. Enter on another model unloads the
  engine, opens the chosen file with its entry's settings (context,
  speculation, draft length, thinking budget, sampling; backend, cache
  precision, output ceiling, and image cap from the command line hold),
  primes it (the token cache's disk tier serves a model used before), and
  continues the conversation rendered through the new profile, which is a
  full prefill; the conversation's images and clips are encoded again by
  the new model's encoders, and one it cannot read or hear leaves the
  model's view with a notice.
  A failed open falls back to the previous model. On the running model,
  Enter applies a new effort (re-priming) or window (re-opening).
- **`/resume [id]`**: an id or a prefix of exactly one id resumes it;
  otherwise the picker lists the workspace's sessions (newest first, the
  one in use marked; the argument becomes the search) and Ctrl-D deletes
  the selected one after the in-picker confirmation, refusing the session
  in use. **`/clear`** drops the conversation into a fresh session file; the
  old one stays resumable.

## Transcript and status

- The agent produces typed events (user, thinking, answer, tool call, tool
  result, diff, notice, info, status, turn end); the transcript turns them
  into blocks and hands each closed block to the screen exactly once. An
  answer flushes block by block as its markdown blocks close.
- Thinking folds and unfolds without leaving the model conversation; a
  step's block closes with its own measured time.
- A tool call is a dot and `Name(argument)`, the dot coloured by state
  (running pulses; settled well, failed, or a write), with the tool's
  one-sentence result under it; a failed call shows its message, bounded.
  A run of calls is summed up in one dim row where the model's text
  resumes. A mutation is followed by its diff: a header with the path and
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
- The status bar shows measurements on the left (state, progress, the
  loop step, context use, token counts, prefill and decode rates) and
  settings on the right (effort, the speculative switch and draft length,
  cache precision, backend, model). A rate that was not measured prints as
  absent; a countdown is marked as an estimate. A narrow bar drops
  settings from the right.

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
  the entry refuse the file, as any unknown type), `compaction`, `notice`.
- Append-only, created at the first entry; a resumed file's ids continue
  past its last. A file stops recording at 64 MiB, the most a resume
  reads, with a notice past 51 MiB naming `/clear`. A truncated last line is
  dropped on load; any other unparsable line or unknown entry type is a
  typed error naming the line; a newer format version is refused.
- Entries store what the model saw and produced, never terminal styling.
  `nuclis agent export` derives markdown from the same entries. A failed write is a dim
  notice, never a lost turn.
- Resuming (`/resume`, `--resume [<id>]`, `latest` by default) renders
  the conversation the model last saw through the profile into a fresh
  session: cancelled steps left out and every `compaction` applied where
  it was written (`context_full`, `results_elided`, `turn_aborted`). When the
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
  replays the rest, and the bar names the cause and the tokens prefilled
  and restored ([agent.md § The agent's token cache](agent.md#12-the-agents-token-cache)).
- Compaction: one tool result may not exceed an eighth of the context
  window in tokens (never below 256); it is cut at a line boundary with a
  note saying how to ask for the rest. When a step still does not fit,
  older results of the turn become one-line stubs, then whole earlier
  turns leave the model's view, each recorded as a `compaction` entry
  (a failed turn leaves it too, recorded the same way);
  only then does the turn end with a message naming the tokens needed
  and how to raise the window. A turn that alone cannot fit fails with a
  named reason.
- Reasoning budget: one step reasons for at most `agent.thinking_budget`
  tokens at `low` (1024 by default and in every catalogue entry `config
  init` writes, 0 for no cap), twice that at `medium`, three times at
  `high`, four times at `xhigh`, and never more than half of the step's
  `generation.max_tokens` (8192 by default), so the answer and a long
  call keep the rest; past it
  the engine emits the profile's close marker itself (Muse Glimmer's
  `<|eom|>`), so the model answers or calls a tool, and the transcript and
  the session step say the reasoning was cut. A model that reopens its
  reasoning past the budget is closed again at once, after the block's
  header, and the model's own late close is dropped, never shown as text.
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
| `write_file` | create or replace a UTF-8 text file, at most 1 MiB, by atomic replacement that keeps a replaced file's mode; the model reads a replacement's diff, a new file's line count only |
| `edit_file` | replace one exact, unique, non-empty sequence; zero or several matches change nothing, nor does a result over 1 MiB; the file's mode is kept; the model reads the diff with one line of context (the transcript shows three) |
| `glob` | one pattern (`*`, `?`, classes, `**`); hidden entries only when named; at most 200 results in stable order, of at most 10,000 collected; more, or unreadable directories, are marked |
| `grep` | literal, case-sensitive; `path` narrows it to a directory, a file, or a glob inside the workspace, `context` adds 0–5 lines around a match; grouped by file in path order; at most 50 matches with the total stated; at most 20,000 files and 10 s a search, a stop stated; skips hidden entries, symlinks, binary-looking files, generated trees, and what each directory's `.gitignore` excludes (a subset: negation, directory-only, anchoring, the glob wildcards) |
| `bash` | one command with a minimal environment; combined output at most 1 MiB; 300 s; cancellation, the timeout, and the output bound stop the command's process group (`SIGTERM`, then `SIGKILL` after 2 s) and reap the child; an output over 120 lines reaches the model as its first 40 and last 80 with the omitted count between |
| `web_search` | offered only with `agent.web` and `web.url`; one query (≤ 400 bytes) to the user's search service, optionally limited by `freshness` (`day`, `week`, `month`, `year`); at most 8 results, each title, URL, and snippet, up to two more snippets under the first three, the corrected query and the engines that failed noted; 15 s; Ctrl-C cancels |
| `web_fetch` | offered as `web_search`; one `http`/`https` URL (≤ 2,048 bytes) fetched by the service, which refuses non-public addresses; its Markdown text 16,000 code points at a time, the range noted with the `offset` that continues it; a page behind a bot challenge named as such; 30 s; Ctrl-C cancels |
| `web_context` | offered as `web_search` when the service's health lists `/v1/context` (it has Brave), probed once at startup; one query and optional `freshness`; passages from at most 8 pages, a 2,048-token budget, each source's title, URL, and date above its passages, at most 12,000 characters cut at a source boundary; 25 s; Ctrl-C cancels |

The web tools are off by default: without `agent.web` the model is shown
the six tools and the same system prompt, byte for byte. Searches and
contexts are metered upstream, so a turn makes at most six uncached ones;
past that the call answers that the budget is spent. The service and
its contract: [agent.md § 11](agent.md#11-the-web-tools-a-hosts-search-service-opt-in).

A read of lines a file returned earlier in the same turn, from unchanged
content and with that result still in the context, is answered with a
one-line pointer to it rather than the lines again.
