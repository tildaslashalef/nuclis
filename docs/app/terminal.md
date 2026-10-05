# The agent's terminal

How `nuclis agent` draws: styled output, the live region, the transcript,
the renderer, the session files, and the agent without a terminal. The
concepts behind them are in [agent.md](agent.md); how the agent is looked
at and measured while it changes is in
[development.md § Working on the agent](../development.md#working-on-the-agent).

## Styled output

Every text report (`config show`, `model ls|pull|inspect`, `inspect`,
`validate`, `tokenize`, `bench`) and the `error:` line on stderr are
colored with the agent's palette (gruvbox dark by default) when the stream
is a terminal
and the environment advertises color (`COLORTERM=truecolor` for 24-bit,
a `256color` or `direct` `TERM` for the approximations, any other terminal
for the sixteen ANSI slots). `--json`, a pipe, `NO_COLOR`,
or an unset or `dumb` `TERM` disable styling completely, so scripts and the tests
see exactly the same bytes; the tests pin the plain form with
`style.Style.none`. The renderers take a `style.Style` explicitly and pad
text before wrapping it in escapes, so alignment never depends on them.
The palette lives in `src/tui/theme.zig` ([TERM-01](https://github.com/tildaslashalef/nuclis/blob/v0.6.0/docs/worklog.md#term-01--agent-terminal-surface-2026-09-12)): named palettes selected by
`agent.theme`, behind a semantic style enum that a theme cannot change, so
a theme changes colour and never layout. `src/tui/style.zig` is the same
palette applied to one-shot reports.

Dim is a colour, not a faded one: the `dim` role paints gruvbox's grey and
adds the SGR dim attribute only at the plain level, where there is no colour
to carry the meaning. Stacking both halves an already low-contrast foreground
and made notices and the help page unreadable on a translucent terminal
(reported and fixed 2026-09-12).

Glyphs are a separate axis from colour ([TERM-01](https://github.com/tildaslashalef/nuclis/blob/v0.6.0/docs/worklog.md#term-01--agent-terminal-surface-2026-09-12) step 6). Every decoration the
agent draws — bullets, task boxes, rules, table joints, fold arrows, the
spinner, the status-bar labels — is named in `theme.Glyphs`, with a Unicode
table and an ASCII one. The ASCII table is selected when the locale does not
claim UTF-8 (`LC_ALL`, then `LC_CTYPE`, then `LANG`, none of which contains
`utf-8`/`utf8`) or when `NUCLIS_ASCII=1` is set, which is also how to check
the fallback on a UTF-8 terminal. Colour and glyphs never influence each
other: `NO_COLOR` keeps the Unicode drawing, and an ASCII terminal keeps its
colours.

## The agent's live region

`src/tui/screen.zig` is the only module that emits a movement escape, and
its geometry is the contract the rest of the surface is written against: a
repaint rewrites the live region in place inside synchronized output
(`CSI ? 2026 h/l`), an insertion above it narrows the scrolling region to
the rows above (`DECSTBM`, top margin row 1 so scrolled-off rows still
reach the scrollback) and scrolls only those, and the region's *bottom*
stays anchored, so a turn that grows the region pushes the transcript up
and one that shrinks it releases rows above the editor rather than
leaving blanks beneath it. A resize replays only the last turn at the new
width; older turns are left as the terminal reflowed them (spec § The agent:
completed turns are immutable), and the region's bottom follows the new
last row: a taller terminal adds blank rows under the region that become
slack above it, a shorter one is taken to have kept its last rows in view,
as tmux and Ghostty do. `NUCLIS_NO_SCROLL_REGION=1` forces the
cursor-up rewrite fallback for a terminal that mishandles `DECSTBM`, and a
`dumb` or unset `TERM` turns both capabilities off. The escape stream of
every operation is pinned by golden tests that need no TTY.

## The agent's transcript

Between the renderer and the screen sits `src/tui/transcript.zig`, the
answer to "what is on the screen, and who may rewrite it" ([TERM-01](https://github.com/tildaslashalef/nuclis/blob/v0.6.0/docs/worklog.md#term-01--agent-terminal-surface-2026-09-12) step 7). The
agent produces typed events (`src/tui/event.zig`: user, thinking, answer,
tool call, tool result, diff, notice, status, turn end); the transcript turns
them into blocks and offers three views of those blocks:

- `takeClosed` — rows for everything that closed since the last call, marked
  written. The agent inserts them above the live region. **Exactly once**:
  the same rows can never be handed out twice, which is what makes a
  scrollback both append-only and correct. An answer flushes block by block
  as its markdown blocks close, so a long answer scrolls away while it is
  written.
- `liveRows` — what is still open, tail-clamped under `… N lines above`.
- `replayRows` — every written block again, for the two events that may
  rewrite the scrollback (a fold toggle, a resize). The agent skips the
  rewrite when the turn is taller than the space above the region: those
  rows are in the scrollback and cannot be reached.

The transcript is pure: no `Io`, no clock (the animated thinking label and
the running dot's pulse phase are passed in by the agent, which has both),
and no knowledge of tokens or models. That is what lets its tests drive a
whole turn — including a byte-by-byte stream — with `std.testing.allocator`
and no TTY. The rows it produces for a tool call are the dot in the call's
state colour (`op_running` pulsing against `dim`, then `op_ok`, `op_error`,
or `op_write`) and `Name(argument)` from `tools.describe`, the tool's
one-sentence result under `└`, and, where the model's text resumes, the
dim `ops` row counting the run (`Read 2 files, ran 1 shell command`). The
repaint cadence comes from the Metal backend's `tick`: `commit` waits on a
semaphore its completion handler signals and calls back every 100 ms, and
the surface installs its poll-and-draw there (`installTick`), so a prefill
chunk repaints ten times a second instead of once. `Screen.paintFrom`
rewrites only from the first row that differs from the last frame.

An attachment is an `attachment` event after the prompt's `user` event:
the transcript keeps it as a detail of that user block and renders a dim
`└ image #1: shot.png (320×240 → 10×8 tokens)` row under the prompt. When
the event carries a preview (the chat builds one where
`tui.graphics.enabled` says the terminal draws kitty graphics, never under
tmux), the block also emits the preview's rows blank and then one raw row
that climbs over them, places the image with the cursor kept, and comes
back — the row model stays text, and the goldens never contain a sequence.
The chat caps the picture to the rows above the live region. A drop and a
typed path both reach the same `Editor.attachImage`; the
chat's `dropProbe` is the editor's only view of the file system.

A mutation's diff is a header row (the path, `+N −M`), then rows of a
`dim` gutter (old and new line numbers, right-aligned), a marker cell
(`+`, `−`, or a space), and the text on its band — `diff_add` and
`diff_remove` are the accent on a dark shade of the same hue, the changed
bytes of a paired line on the brighter `diff_add_change`/`diff_remove_change`
tint — padded to the width so the band reads as one. Side by side from
`side_by_side_min_width` (96) columns, the two panes separated by the table
bar; unified below that. A theme role marked `wide_bg` drops its
background at sixteen colours, where a dark band cannot be painted, and
takes its `plain` attributes instead.

Two folds, both applied to whatever is rendered next and replayed over the
last turn: Tab folds thinking, Ctrl-O cycles the tool view
(`transcript.ToolView`: `summary`, one result row under each call;
`output`, the result's text under it as the model received it, dim, cut at
`max_output_rows`; `folded`, a call keeps its row and loses the detail
under it, the result rows, and the diff's rows). A `!` line from the editor runs through the `bash` tool and
shows as the same `Bash(cmd)` block followed by the output as an `info`
block (40 rows, then `… N more lines`); with `!` the output is also the
next user message, which the surface does not echo (`quiet_user`).

The answer's markdown is rendered once per closed block: the transcript
keeps the byte offset up to which the answer has been flushed to the
scrollback (`Answer.flushed`), hands `markdown.split` only the remainder,
and renders the closed prefix it returns; the open tail is shown raw. A
test streams a document byte by byte and counts the renders — one per
block boundary, never one per token — and a prefix fuzz feeds every byte
prefix of every fixture through `split` and `render`, asserting no error,
no control byte, a bounded row count, and no row wider than the width.

## The agent without a terminal

`nuclis agent -p "<prompt>"` (or `--print --prompt-file <path>`) runs one
turn with no TTY: the text form streams the answer, `--json` writes every
event as one object per line, and `--session <path>` is the only way a
printed turn records anything. It is the scripting entry point today and the
harness the agent loop will be tested through in phase 2, where a stream of
JSON lines is something a test can assert on and a terminal is not.

## The agent's session files

`src/agent/session.zig` writes one append-only JSONL file per conversation
under `~/.nuclis/agent/sessions/<cwd-slug>/<stamp>_<id>.jsonl` ([TERM-01](https://github.com/tildaslashalef/nuclis/blob/v0.6.0/docs/worklog.md#term-01--agent-terminal-surface-2026-09-12) step 8).
The first line is a header (format `version`, session id, time, working
directory, the model path and the digest its pull sidecar recorded, effort,
context size); every later line is an entry carrying `id` and `parent`, so a
branch or a cancel-rewind is representable later without a migration. The
file is created at the first entry, so starting the agent and quitting writes
nothing. On load, a truncated *last* line is dropped — that is where a crash
lands — while any other unparsable line, or an unknown entry type, is a typed
error naming the line number; a file from a newer `version` is refused
outright. `nuclis agent export` derives markdown from the same entries, so there is no
second transcript format.

## The agent's renderer

`src/tui/markdown.zig` turns a turn's text into pre-styled, pre-wrapped rows.
Two rules matter when reading it ([TERM-01](https://github.com/tildaslashalef/nuclis/blob/v0.6.0/docs/worklog.md#term-01--agent-terminal-surface-2026-09-12) step 6):

- **Streaming.** `markdown.split(text)` divides a partial turn into the
  blocks that can no longer change and the one still being written: a blank
  line ends a block, a heading/rule/quote/list item ends one at its newline,
  and a paragraph, a table, or an open fence keeps its block open. The agent
  renders the closed part and shows the open part as raw text, repainting
  both on every token, so styling appears block by block instead of at the
  end of the turn. A trailing newline is the end of the last block, not an
  empty one after it — which is what makes the rendered prefix of a partial
  turn a prefix of the finished one, a property a test pins byte by byte.
- **Wrapping.** `view.lines` and `view.wrapStyled` take a `Wrap` mode.
  Everything a reader reads wraps at the last space that fits (`.word`, the
  editor's rule since step 3); only rows that are truncated to one line
  anyway — the status bar, the shortcut hint — use `.character`. A word
  longer than the row still breaks, and a space that lands past the edge
  becomes the next break point instead of breaking the row, so a word ending
  exactly at the last column keeps it.

