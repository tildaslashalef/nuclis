<!-- kind: reference -->
# The agent's terminal

How `nuclis agent` draws: styled output, the live region, the transcript,
the renderer, the session files, and the agent without a terminal, then
the terminal concepts behind them (§1, §4, and §7 of the concepts whose
others are in [agent.md](agent.md) and
[agent-tool-calls.md](agent-tool-calls.md)). How the agent is looked
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
The palette lives in `src/tui/theme.zig`: named palettes selected by
`agent.theme`, behind a semantic style enum that a theme cannot change, so
a theme changes colour and never layout. `src/tui/style.zig` is the same
palette applied to one-shot reports.

Dim is a colour, not a faded one: the `dim` role paints gruvbox's grey and
adds the SGR dim attribute only at the plain level, where there is no colour
to carry the meaning: stacking both halves an already low-contrast
foreground, which makes notices and the help page unreadable on a
translucent terminal.

Glyphs are a separate axis from colour. Every decoration the
agent draws — bullets, task boxes, rules, table joints, fold arrows, the
spinner, the status-bar labels — is named in `theme.Glyphs`, with a Unicode
table and an ASCII one. The ASCII table is selected when the locale does not
claim UTF-8 (`LC_ALL`, then `LC_CTYPE`, then `LANG`, none of which contains
`utf-8`/`utf8`) or when `NUCLIS_ASCII=1` is set, which is also how to check
the fallback on a UTF-8 terminal. Colour and glyphs never influence each
other: `NO_COLOR` keeps the Unicode drawing, and an ASCII terminal keeps its
colours.

A finished turn sends a desktop notification (OSC 9, `nuclis: response
ready`) when the window lost focus during it and the turn was not
cancelled; terminals without OSC 9 ignore it. `NUCLIS_NO_NOTIFY=1` (any
value but `0`) or a `dumb` `TERM` turns it off (`src/tui/terminal.zig`).

Every `NUCLIS_*` variable the code reads, and where (`NUCLIS_HOME` moves
the user root, [getting-started.md § User
directories](../guide/getting-started.md#user-directories)):

<!-- gen:env -->
| Variable | Read in |
| --- | --- |
| `NUCLIS_ASCII` | `src/tui/theme.zig` |
| `NUCLIS_HOME` | `src/agent/print.zig`, `src/cli.zig`, `src/completion.zig`, `huggingface/src/root.zig` |
| `NUCLIS_NO_NOTIFY` | `src/tui/terminal.zig` |
| `NUCLIS_NO_PREVIEW` | `src/tui/graphics.zig` |
| `NUCLIS_NO_SCROLL_REGION` | `src/tui/screen.zig` |
| `NUCLIS_PANIC_HOOK` | `src/panic_hook.zig` |
<!-- /gen -->

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

The lease is given back on every way out. A `defer` right after the lease
is taken covers any startup failure; a panic restores the terminal before
the default report (the root `panic` in `src/main.zig`), and `SIGTERM` and
`SIGHUP` restore it and then end the process as their default would, since
a model step can hold the loop for seconds. The restore also ends a
synchronized frame and resets the scroll region, both of which a failure
in the middle of a paint leaves set. A bracketed paste holds at most the
editor's 128 KiB: past that it stops growing and is refused at its end
marker, with a notice.

## The agent's transcript

Between the renderer and the screen sits `src/tui/transcript.zig`, the
answer to "what is on the screen, and who may rewrite it". The
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
dim `ops` row counting the run's calls by their results (`Read 2 files,
ran 1 shell command, 1 call failed`: a red call counts only as failed). The
repaint cadence comes from the Metal backend's `tick`: `commit` waits on a
semaphore its completion handler signals and calls back every 100 ms, and
the surface installs its poll-and-draw there (`installTick`), so a prefill
chunk repaints ten times a second instead of once. `Screen.paintFrom`
rewrites only from the first row that differs from the last frame.

An attachment is an `attachment` event after the prompt's `user` event:
the transcript keeps it as a detail of that user block and renders a dim
`└ image #1: shot.png (320×240 → 10×8 tokens)` row under the prompt (a
clip's is `└ audio #1: memo.m4a (12.4 s → 310 tokens)`, no preview). When
the event carries a preview (the chat builds one where
`tui.graphics.enabled` says the terminal draws kitty graphics, never under
tmux), the block also emits the preview's rows blank and then one raw row
that climbs over them, places the image with the cursor kept, and comes
back — the row model stays text, and the goldens never contain a sequence.
The chat caps the picture to the rows above the live region. A drop and a
typed path both reach the same `Editor.attachImage` or `Editor.attachAudio`
(`[audio #N]`, numbered apart from the images); the chat's `dropProbe` is
the editor's only view of the file system. For a clip it loads the audio
encoder on first use (`audioAvailable`, refusing by name on a model that
does not hear, the hearing one taken from the catalogue's `hears`) and
measures the file from its header (`src/agent/clip.zig`): over 30 s or
undecodable is a notice and the paste stays text. The encoder never loads
from inside the GPU wait's tick (a drop read during a turn): that drop is
refused with a notice to drop it again after the turn.

An image or audio chip is encoded while the prompt is typed
(`src/agent/ahead.zig`, keyed by the attachment's path for the chip and by
the file's SHA-256 for the turn). The idle loop's `syncAhead` queues
a new chip's file, drops a deleted chip's entry, writes each chip's note
(`Editor.setNote`: `queued`, `encoding…`, `676 tokens` or `10.4 s → 261
tokens`, `failed: …`), and
starts the next queued file on a task (`Io.concurrent`) when none runs.
The task uses the model's own Metal backend, so the driver calls
`Ahead.settle` (wait for the running task) before anything else touches
the engine: a turn (`loadMedia`, under its `encoding image #1` label), a
command, a model or context switch, a resume, quitting. The backend's
`tick` runs on whichever thread commits, so `toolTick` returns at once off
the driver's thread: the task never reads keys or draws. Measured on the
E4B: [vision.md § Encoders at the materialized-score attention](../benchmarks/record/vision.md#encoders-at-the-materialized-score-attention-2026-10-27).

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

Text from the model, a tool, or a model file is drawn as data. Control
bytes are dropped, the UTF-8 encoded C1 controls (`U+0080`–`U+009F`, which
a terminal may read as CSI or ST) included, and a tab becomes spaces to the
next stop of four cells before the text is wrapped, so a tab-indented code
block keeps its shape. The status bar is a raw row, so its free text (the
status word, and a model's name, which comes from the file's own metadata)
is cleaned before it is painted. A table row is cut or padded to its
header's cell count, and a link target is looked for once per run of `[`,
so neither a ragged table nor a page of brackets can stall or stop a frame.

## The agent without a terminal

`nuclis agent -p "<prompt>"` (or `--print --prompt-file <path>`) runs one
turn with no TTY: the text form streams the answer, `--json` writes every
event as one object per line, and `--session <path>` is the only way a
printed turn records anything. It is the scripting entry point and the
harness the agent's task list runs through (`make agent-eval`), where a
stream of JSON lines is something a test can assert on and a terminal is
not.

## The agent's session files

`src/agent/session.zig` writes one append-only JSONL file per conversation
under `~/.nuclis/agent/sessions/<cwd-slug>/<stamp>_<id>.jsonl`.
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
Two rules matter when reading it:

- **Streaming.** `markdown.split(text)` divides a partial turn into the
  blocks that can no longer change and the one still being written: a blank
  line ends a block, a heading/rule/quote/list item ends one at its newline,
  and a paragraph, a table, or an open fence keeps its block open. The agent
  renders each closed block once, as it closes, and shows the open part
  raw in the live region, so styling appears block by block instead of at
  the end of the turn. A trailing newline is the end of the last block, not an
  empty one after it — which is what makes the rendered prefix of a partial
  turn a prefix of the finished one, a property a test pins byte by byte.
- **Wrapping.** `view.lines` and `view.wrapStyled` take a `Wrap` mode.
  Everything a reader reads wraps at the last space that fits (`.word`, the
  editor's rule too); only rows that are truncated to one line
  anyway — the status bar, the shortcut hint — use `.character`. A word
  longer than the row still breaks, and a space that lands past the edge
  becomes the next break point instead of breaking the row, so a word ending
  exactly at the last column keeps it.

## 1. A terminal that is written once: events, blocks, and the line between them

The inference companion is about producing tokens; this one is about the other
end — how a turn reaches a person — because the terminal surface is what the
tool loop hangs from, and its rules are not obvious.

**The rendering model.** nuclis does not use the alternate screen (the
full-window mode `vim` and `less` switch into). Finished output is written into
the terminal's *own* scrollback exactly once and never touched again; a small
**live region** at the bottom — the streaming block, the editor, the status bar
— is the only thing repainted. The payoff is that scrolling, searching,
selecting, and copying are the terminal's job, already correct, and the
conversation survives exit. The price is that a printed row is gone: it can be
rewritten only while it is still on screen, which is why a fold toggle on a
turn taller than the screen does nothing rather than half-rewriting it.

Two escape sequences carry the model. **Synchronized output** (`CSI ? 2026 h`
/ `l`) brackets a repaint so a terminal shows the frame whole instead of
tearing mid-write. A **scrolling region** (`DECSTBM`, `CSI 1 ; bottom r`)
narrows scrolling to the rows *above* the live region, so inserting a finished
row scrolls only those and the region below is never erased — the flicker of
erase-and-repaint disappears. The top margin must stay at row 1, because that
is the condition under which terminals feed scrolled-off rows to the
scrollback.

**When is a block finished?** While a turn streams, the last markdown block is
still being written: a paragraph may gain words, a fenced block its closing
fence, a table another row. So the renderer answers one question —
`markdown.split(text)` returns the prefix whose rendering can no longer change
and the tail that can — and the surface styles the first while showing the
second raw. A blank line ends a block; a heading, a rule, a quote, a list item,
or a closing fence ends one at its newline; a paragraph, a table, and an open
fence keep theirs open. One property makes this safe, and is pinned by a test
that feeds a document a byte at a time: *rendering a closed prefix produces a
prefix of rendering the whole*. Without it, text would shift on the screen as
later bytes arrived.

**Events, and who owns their strings.** Between the generation loop and the
screen sits a tagged union — user, thinking delta, thinking end, answer delta,
tool call, tool result, diff, status, turn end, notice. The transcript builds
blocks from it, the status bar folds the beats into a row, print mode writes
it as text or JSON, and the engine and the tool loop produce the same kinds. An
`answer_delta` is emitted once per generated token, so its
string is **borrowed for the duration of the call**: a consumer that keeps text
copies it. Owning every event's bytes would mean an allocation and a free per
token for text that is copied anyway.

Zig makes two parts of this cheap. The consumer is `anytype` with a single
method — `sink.send(event)` — so the splitter that turns one stream into two
works for the transcript and for print mode with no interface value, no vtable,
and no allocation: each call site is compiled against its own concrete type.
And the union is `union(enum)`, so `switch` over it is exhaustive: adding an
event kind without rendering it is a compile error rather than a row that
silently never appears.

**The original hold-back.** A model writes one stream with its reasoning channel
delimited by markers ([§7 of the inference companion](../llm-guide.md#7-the-prompt-profile-is-a-contract)). Splitting it *incrementally* has a trap: while the
channel is open, the first bytes of a half-arrived `</think>` look exactly like
reasoning text, and a naive splitter puts `</thi` on the screen for one token
before reclassing it. Keeping back one marker's length minus one byte makes
that impossible, and costs nothing — the held bytes arrive with the next token.
But then the *end* of a turn must release them explicitly, or a turn cancelled
mid-thought loses its last characters. That bug lived in the inline version
until the splitter became a module with tests of its own; it is the kind of
thing that only shows up when a component can be driven byte by byte without a
model, a clock, or a terminal.

## 4. A cell is not a code point: grapheme clusters and terminal width

A terminal is addressed in fixed cells, and the naive way to count them is one
per Unicode code point (or worse, one per byte). That is wrong for exactly the
text a chat UI sees most: emoji. `👋🏿` is a wave plus a skin-tone modifier,
`👩‍🚀` is three code points joined by a zero-width joiner, `❤️` is a heart plus
U+FE0F, `🇨🇭` is two regional indicators, `e` + U+0301 is a base plus a
combining mark. Measured per code point, the first three are four cells wide
and can be *split across a row* mid-cluster; the editor's Backspace can delete
half a family.

The unit is the **grapheme cluster**, defined by UAX #29. Clusters are found by
a boundary rule, not by a per-character property: "no break before an Extend",
"break around Control", "join a regional-indicator pair", "join an
extended-pictographic ZWJ sequence", and a handful of Hangul and Indic rules.
Several of those need *state* — the parity of a regional-indicator run, or
whether an `Extended_Pictographic Extend* ZWJ` preceded the current character —
which is the same lesson as [§2](agent-tool-calls.md#2-why-a-streaming-parser-needs-token-boundaries): the meaning of a byte depends on what came
before it. `src/tui/graphemes.zig` carries that state through a cluster and
resets it at each boundary.

Width is a second, separate policy on top of segmentation. UAX #11 gives each
code point an East Asian Width (`W`/`F` are two cells, `A` ambiguous is a
judgement call), but a cluster's width is not the sum of its parts: a joiner,
a variation selector, a skin-tone modifier, and Hangul V/T jamo all contribute
zero, and an emoji-presentation cluster is two cells regardless of how many
code points it took. nuclis keeps one documented policy in `clusterWidth`:
VS15 keeps a cluster at one cell, an emoji presentation / modifier / flag pair
is two, everything else sums its non-zero code points, and an unpaired
regional indicator stays one. That is a terminal approximation, pinned by
fixtures rather than asserted universally.

The data is the other half. Unicode properties are large tables; `uucode`
exists to generate them, but adding a dependency for one table is a lot of
surface for a project that keeps its build offline. `scripts/fixtures/grapheme-table.py`
instead downloads a **pinned UCD** once, checks its digests, and emits one
sorted interval table (`graphemes_table.zig`) that a binary search reads. The
generator is committed, the artifact is committed, and the data layer sits
behind `graphemes.zig`, so swapping in `uucode` later is a change to one file.
The correctness evidence is the Unicode project's own conformance file:
`GraphemeBreakTest.txt` (766 cases at 17.0.0) runs through the same iterator
the editor and the wrapper use.

## 7. A diff is one computation, rendered two ways

A diff looks like text, and the first instinct is to render it as text:
compute a unified string, then colour the lines whose first byte is `+` or
`-`. That works until the second consumer appears. The model and the session
need the unified form; the terminal, at a wide width, wants a *two-column*
view that shows an old line beside the replacement, with the bytes that
changed marked. If the renderer recovers line numbers and pairing by parsing
the unified string, the computation exists twice — once producing the string,
once interpreting it — and the two can drift.

So the unit of a diff here is a **row**: old line number, new line number,
kind, text, and the byte span that changed. `tui.diff.compute` produces rows
from an old and a new byte string and derives the unified text from the same
edit script. The renderer consumes rows and never parses; the model and the
session consume the unified string. Neither is a description of the other.

The algorithm is a plain line diff with two practical concessions. Lines
split on `\n` alone, so a CRLF file diffs cleanly against itself (`\r` stays
in the line and is invisible once rendered). Common leading and trailing lines
are stripped first, because the common case is a small edit in a large file;
the remaining middle goes through an LCS table, and a middle too large for the
table degrades to "all removed, all added" rather than spending unbounded
time. Only changed hunks plus three lines of context are materialised, so a
one-line change in a thousand-line file stays eight rows; the unified text the
model reads keeps one line on each side, since it wrote the change and pays
for every line it reads back. Pairing a removal with
the addition that replaced it is what lets the two-column view put them on one
display row and highlight the bytes that differ, and it falls out of the same
edit script.

Zig note: the module has no theme and no I/O. It allocates owned bytes and is
tested against fixtures (insertion, deletion, replacement, context, CRLF, a
missing trailing newline, the bounded fallback), which is what lets the layout
be a golden test instead of a screenshot.
