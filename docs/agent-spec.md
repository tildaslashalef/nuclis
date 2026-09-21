# nuclis agent — specification

Status: accepted product scope, supporting [spec.md](spec.md), which stays
the authoritative engine specification and links here. Extracted on
2026-09-09 from the spec's chat and embedded-agent sections and from the
roadmap's agent units, with the decisions of that day (name, rendering
model, session storage, the TUI module API). **Phase 1 closed as unit TERM-01 on
2026-09-12** (two sessions;
[engineering log](engineering-log.md#term-01--agent-terminal-surface-2026-09-12));
**phase 2 closed as unit AGNT-07 on 2026-09-14**, the tool loop and resume
([engineering log](engineering-log.md#agnt-07--polish-resume-and-the-agent-plan-closed-2026-09-14)).

Wording: *decided* marks an accepted decision, *proposed* a target the
implementing unit may still change with a recorded reason, *implemented*
what the tree does today. Nothing here claims a measurement.

## Purpose

`nuclis agent` is the interactive surface of the engine: a terminal
program that holds a conversation with the local model and, in its second
phase, completes small coding tasks in the current working directory with a
handful of bounded tools. It is one command and one product surface. The
evaluation CLI (`generate`, `bench`, `inspect`, `validate`, `config`) stays
separate and never acquires tools ([spec § Local serving](spec.md#local-serving-and-deferred-work)).

It is also the project's playground: every engine feature that a user can
feel (prefill speed, long context, sampling profiles, snapshots) is
exercised here first, and the terminal surface is where the engine's typed
events get their first consumer.

## Decisions

Dated so a later reader can tell a decision from its context.

- **Name (2026-09-09, decided; implemented 2026-09-12, TERM-01 step 1).** The
  command is `nuclis agent`. The configuration section is `agent` (renamed
  from `chat` on 2026-09-11, ahead of the command: `think`,
  `fold_thinking`, and now `theme`), the data root is `~/.nuclis/agent/`
  (`paths.agentPath`, created only when something is written), and the
  composition module is `src/agent/`. No alias is kept and no data is
  migrated (the tree is `0.1.0-dev`, nothing external depends on the name);
  `nuclis chat` is an unknown command. A file with a `chat` *section* still
  gets the rename message from the configuration loader.
- **Inline rendering with native scrollback (2026-09-08 for the chat,
  reaffirmed 2026-09-09 for the agent, decided).** No alternate screen.
  Completed turns are written once into the terminal's scrollback and never
  repainted; a bounded live region at the bottom holds the active turn, the
  editor, and the status bar. The terminal's own scrolling, search, mouse
  selection, and copy work natively, and the transcript survives exit.
  Accepted limits: completed turns are immutable (no re-fold, no reflow on
  resize; only the last turn is replayed), overlays live inside the live
  region, and reading a long transcript is the terminal's job. An
  on-demand alternate-screen overlay for reading the transcript is
  compatible with this decision and stays in reserve; it is not planned.
- **Sessions are append-only JSONL files (2026-09-09, decided).** One file
  per session under `~/.nuclis/agent/sessions/`, one JSON object per line,
  the same event kinds the transcript renders. No database: nuclis has no
  third-party code and a session log needs none of a database's
  properties; a truncated last line is dropped on load and that is the
  whole crash story. This reverses the earlier "no sessions, no event log"
  tradeoff; see [Sessions and storage](#sessions-and-storage).
- **No permission system (2026-09-08, decided).** Every tool executes
  immediately; the user sits at the terminal and sees each mutation (exact
  command, unified diff) as it happens. Supervision is visibility plus the
  workspace boundary. The inline choice prompt exists as a generic
  component (pickers, completion) and may later host an approval step; that
  step is not planned.
- **No MCP, skills, extensions, or subagents (decided).** Six fixed tools.
- **The TUI is a module, not a package (2026-09-09, decided).** `src/tui/`
  imports nothing from `inference`; it is promoted to a top-level package
  only when something outside this executable consumes it.
- **Theme (2026-09-09, decided; implemented 2026-09-12, TERM-01 step 1).**
  Gruvbox dark is the default theme. `src/tui/theme.zig` paints through a
  semantic style enum at three levels (truecolor, 256-colour, plain
  attributes; `NO_COLOR` wins). That enum is the contract between renderer
  and palette, and it is now backed by three layers: a named `Palette` of
  colours (the `fg`/`bg` scales and the eight accent pairs of
  [morhetz/gruvbox](https://github.com/morhetz/gruvbox), each carrying its
  truecolor value and the community 256-colour approximation), a `Role`
  table saying what each semantic style asks the palette for, and the
  level. A theme supplies a palette only — it can never change a role, so
  it changes colour and glyph weight, never layout. `agent.theme` selects
  it (default `gruvbox-dark`); every (palette, level, style) triple is a
  static string built at compile time, so `paint` stays a table lookup.
  TERM-01 added the block kinds the later steps render (tool call, tool
  result, diff add/remove/header, chip, progress, choice selection). Two
  tests pin the result: every palette value against the canonical source,
  and the sequences each level emits.

## Phases and units

```text
phase 1  terminal surface     TERM-01 (two sessions)             closed 2026-09-12
phase 2  engine events        AGNT-01 (typed events; the adapter registry landed in MODL-04, 2026-09-11)  closed 2026-09-14
         agent loop           AGNT-02 (closed 2026-09-14) → AGNT-03 → AGNT-04 → AGNT-05 (closed) → AGNT-06 (closed) → AGNT-07 (closed 2026-09-14)
```

Phase 1 is engine-free except for one progress hook, which is why it can
close the plan after the kernel units. Phase 2 waits for the v0.1
acceptance because every agent step is a fresh completion with a
growing prompt (needs the batched prefill of ENGN-02–ENGN-05), a cancelled tool
loop needs a turn-boundary checkpoint (ENGN-06), and reliable tool calls need
the official sampling profiles (MODL-01, closed). The seam between the phases
is the typed event union defined in phase 1: the agent loop is a producer
of the same events the transcript already consumes.

## Phase 1 — the terminal surface

### Implemented today (`nuclis chat`, APPS-02)

Recorded here because phase 1 builds on it; the [engineering log](engineering-log.md)
holds the APPS-02 outcome.

- Takes its backend, context, output ceiling, starting effort (`low`),
  sampling overrides, and initial thinking fold from `nuclis.json`; shares
  the generation flags (`--backend`, `--ctx-size`, `--max-tokens`,
  sampling, `--seed`, `--think`), which override the file. Requires a TTY.
  Startup clears the visible screen (scrollback preserved), prints the
  welcome (`tui/banner.zig`: the ASCII wordmark when the terminal is at
  least 60 columns wide, then the version, the model reached through its
  registry or catalogue name with the artifact's own name, backend, profile
  and whether it was forced, context, effort, and the workspace with the
  home directory as `~`, all of it in a rounded box; the plain lines below
  that width), and anchors the live region (active turn, the framed input
  box, hint, status bar) at the bottom. The welcome scrolls away like any
  transcript row; the status bar carries the model's short name as its last
  segment. The input box's frame carries the reasoning effort as its colour
  and, while a turn runs, the spinner in its top edge.
- Multi-turn without replay: the process keeps the text the session has
  consumed (rendered prompt plus generated tokens except the final
  stop/budget token). Each turn renders the whole conversation with the
  profile; when the result starts with the consumed text only the remainder
  is prefilled. Otherwise (effort change, cancelled turn, template trimming)
  the session resets and replays, and the status bar says so. Recurrent
  state is never rewound by hand.
- Compaction: before a turn, the rendered conversation plus the output
  budget is measured by encoding it; when it cannot fit, the oldest turn
  pairs leave the model's view (`context_start`) until it fits. They stay
  visible in the transcript under a dim "older turns dropped" line.
  Dropping diverges from the consumed text, so the session replays the kept
  conversation. A single turn that alone cannot fit fails with
  `ContextFull`. Summarized compaction is a later upgrade on the same
  message-list seam.
- Keys: Enter sends; Shift-Enter inserts a newline (kitty keyboard
  protocol, requested on entry; Ctrl-J is the fallback); bracketed paste
  inserts verbatim with tabs as four spaces, up to the 16 KiB input limit;
  Left/Right/Home/End move by grapheme cluster (emoji and combining marks
  are one unit); Delete/Backspace; Up/Down recall submitted prompts (the
  pre-recall input is kept as a draft); Tab folds thinking; Ctrl-T cycles
  effort off/low/medium/high/xhigh; Ctrl-W cycles the context window
  2k/4k/8k/16k/32k (the engine re-opens at the new capacity, the
  conversation replays into it on the next turn, a failed re-open falls
  back); Ctrl-N starts a new session; Ctrl-C cancels a running turn (second
  Ctrl-C quits) or quits when idle; Ctrl-D quits. The lease also asks for
  focus in/out reports; a turn that finishes while the window is unfocused
  sends one OSC 9 notification (off with `NUCLIS_NO_NOTIFY`, and for a
  cancellation or a dumb terminal).
- Thinking folds and unfolds without leaving the model conversation; the
  active turn re-renders as it streams, the last printed turn is replayed
  with the toggled state, older turns keep the state they were printed
  with. While a step runs the fold label animates ("⠙ thinking… 3s"); a
  step's block closes when its reasoning ends (the first answer byte, or
  the step ending in tool calls) with that step's own time from its start
  ("Thought for 3.4s"), so a turn with two tool calls shows three timed
  blocks whose sum is below the turn's elapsed time; the time is kept in
  the session's step stats so a resumed session shows it (TERM-06).
- Completed answers render as small markdown (headings, emphasis, inline
  code, `[label](http(s)://…)` links as OSC 8 hyperlinks, quotes, lists with task boxes,
  fenced code with a heuristic highlighter, rules, pipe tables) into
  pre-styled, pre-wrapped lines; model-supplied control bytes are stripped.
  A streaming tail shows as plain text.
- Rates follow `bench`: prefill covers the turn's new prompt tokens, decode
  covers `generated − 1` steps; elapsed-time estimates during a turn,
  measured timings afterwards. Unavailable measurements never appear as
  fabricated rates.

### What phase 1 changes

Motivation, from the first long-prompt session (2026-09-09): a
1,226-token paste was cut mid-word with no sign of the lines above, the
live region hid the prompt's top while the turn ran, and the spinner froze
during a 15 s prefill.

#### Rendering model (proposed; the four items below implemented in TERM-01 step 2)

- **Turn commit.** Enter prints the user turn into the scrollback in full,
  at once, and clears the editor. The live region then holds only the
  thinking preview and the streaming answer. When they exceed the region's
  budget, the tail is shown under a `… N lines above` indicator, never
  clamped silently. Flushing closed answer blocks into the scrollback as
  they complete is the session-2 refinement once the renderer knows block
  boundaries (*implemented in step 7*, below).
  *Implemented (TERM-01 step 4)*, including the failure path: the
  prompt is committed before anything can fail, so a turn that cannot fit
  the context leaves what was asked in the transcript with a dim reason
  under it. A completed turn's scrollback block is the answer alone, which
  is also what a resize or a fold toggle replays.
- **Atomic repaints.** Every live-region repaint is wrapped in synchronized
  output (`CSI ? 2026 h/l`) so the region never tears; terminals without it
  ignore the sequences. *Implemented.*
- **Insertion above the region.** Completed lines are inserted above the
  live region through a scrolling region (`DECSTBM`) rather than by
  repainting the region after each print; the fallback is today's
  cursor-up rewrite. This is the technique Ratatui adopted for its inline
  viewport to stop `insert_before` flicker. *Implemented*: the top margin
  stays at row 1 so scrolled-off rows still reach the scrollback, and
  `NUCLIS_NO_SCROLL_REGION=1` selects the fallback. TERM-01 added one rule the
  proposal did not state: the region's **bottom** is the anchor, so a
  region that shrinks releases rows above the editor instead of leaving
  blank rows under it. Those released rows are slack, not transcript: the
  next growth takes them back and the next insertion fills them before it
  scrolls, so a settled tool call or a folded thinking block never leaves
  a gap in the scrollback (TERM-07).
- **Resize.** The live region recomputes its layout; the last printed turn
  is replayed at the new width; older turns are left as printed.
  *Implemented*, replacing APPS-02's clear-and-reprint-everything. Known
  limit: the replay walks back over the row count the turn had at the old
  width, which a terminal that reflows has already changed, so a resize can
  leave a partial duplicate of the last turn — bounded to that one turn.
- **Capabilities.** `NO_COLOR` and a 16-colour fallback; an ASCII glyph set
  when the locale is not UTF-8 or `NUCLIS_ASCII=1`; box glyphs never appear
  inside copyable code text. *Implemented (TERM-01 step 6).* The colour level
  gained `c16` (the palette's ANSI slots, which are nuclis's mapping and not
  the theme's, since a sixteen-colour terminal paints from the user's own
  palette), and `TERM` now has to advertise `256color` or `direct` for the
  wide palette instead of any terminal getting it. Glyphs became a third
  axis beside palette and level: `theme.Glyphs` names every decoration once
  with a Unicode and an ASCII table, selected by the locale's codeset or
  `NUCLIS_ASCII`. A code block draws no border at all — the language is a
  caption row above it — so selecting one copies the code and nothing else.
- **Exit.** Terminal state restored; only the transcript remains in the
  scrollback, no banner residue. *Implemented*: the region is erased and
  the scrolling region reset before the terminal lease is released.

#### Editor (proposed; implemented in TERM-01 step 3 as `src/tui/editor.zig`)

- Word wrap at spaces; a scrolled editor shows `… N lines above`; the
  cursor row is always visible.
- A paste longer than a few lines becomes an inline chip
  (`[pasted 96 lines, 6.1 KB]`). The chip is a range over the flat input
  buffer: cursor motion skips it, Backspace at its edge deletes it whole,
  and an expand key drops the range so the text is edited in place. The
  text itself is what is sent.
- Up/Down move inside a multi-line input; history recall happens only from
  the first line (Up) and the last line (Down).
- Input limit 128 KiB (today 16 KiB).
- Prompt history persists across sessions in
  `~/.nuclis/agent/history.jsonl`.
- Session 2: completion lists inside the live region for slash commands on
  `/` and workspace paths on `@`; typing while a turn runs is allowed and
  the message is queued for the next step (steering; the spec's earlier
  open question is thereby answered "yes, in phase 1 session 2").

*Implemented (TERM-01 step 3).* Decisions the unit had to make and the spec did
not state: the expand key is **Ctrl-E**, and any edit that lands strictly
inside a chip expands it too, since a chip whose bytes changed is no longer
what was pasted; a paste becomes a chip at **4 lines or 400 bytes**; a
bracketed paste accumulates until its end marker, so the whole paste is
weighed at once rather than appearing and then collapsing; a word longer
than the row still breaks mid-word, because the alternative is a row that
cannot be drawn; the history keeps **200** entries and does not repeat the
newest one. Persistence lives in `src/agent/history.zig`, not in the
editor, because `src/tui/` does no file I/O: one JSON object per line,
append-only, a truncated or malformed line dropped on load, only the last
256 KiB of the file read at startup, and prompts over 8 KiB kept in memory
only (a 100 KB paste is not something anyone recalls with Up).

#### Transcript and events (proposed)

The transcript consumes a typed event union owned by the TUI module. The
engine (phase 2, AGNT-01) and the agent loop produce the same kinds; phase 1
produces them from the existing generation loop.

```zig
pub const Event = union(enum) {
    user: Text,                       // a committed prompt
    thinking_delta: Text,
    answer_delta: Text,
    tool_call: struct { id: Id, name: []const u8, summary: []const u8, detail: ?[]const u8 },
    tool_result: struct { id: Id, text: []const u8, truncated: bool, is_error: bool, summary: []const u8 },
    diff: struct { path: []const u8, unified: []const u8 },
    status: struct { phase: Phase, position: usize, target: usize, rates: Rates, elapsed_ns: u64 },
    turn_end: struct { stop: StopReason, stats: TurnStats },
    notice: Text,                     // dim system lines: replay, dropped turns, errors
};
```

Strings are owned by the event and freed by the consumer; an allocator is
passed explicitly. Block kinds rendered from these events: user, thinking
(foldable), answer, tool call (name, argument summary, folded result with a
truncation marker), diff (`+`/`-` coloured), notice, and a choice prompt.

*Implemented (TERM-01 step 7)* as `src/tui/event.zig` and `src/tui/transcript.zig`,
with four deviations, all of them recorded here:

- **Events borrow their strings; the consumer copies what it keeps.**
  Owning them would mean an allocation and a free per generated token for
  text the transcript immediately copies into its own buffer. The producer
  keeps the bytes alive for the duration of the call, which is all any
  consumer needs — the session writer serializes on the spot.
- **`thinking_end` is an event.** The transcript takes no `Io` and so has no
  clock; the producer states the measured seconds instead of the consumer
  taking them.
- **`turn_end` is a block**, not only a status: it renders the stop marker
  (`— cancelled`, `— output budget reached`, `— context full`) and the blank
  row that separates turns, so every row on the screen belongs to a block. A
  failure adds no row of its own, because it always arrives with a notice
  naming it.
- **`info` is a kind beside `notice`** (added 2026-09-12, after the first
  `/help` on a translucent terminal proved unreadable). A notice is an aside —
  a replay, dropped turns, a failure under a prompt — and is dim; an `info`
  block is what a command *answered*, and carries no style at all, so it is
  painted at the terminal's own foreground. The same report fixed the
  mechanism behind it: the `dim` role stacked the SGR dim attribute on a grey
  foreground, halving an already low-contrast colour, and now uses the
  attribute only at the plain level where there is no colour to carry it.
- **The rewrite is bounded by the screen.** A fold toggle or a resize can
  only reach rows that are still on the screen, so a turn taller than the
  space above the live region is left as printed rather than half-rewritten.
  The fold state still applies to whatever is printed next.

This is also where the spec's *session-2 refinement* landed: `takeClosed`
hands each closed block to the screen exactly once, and an answer flushes
**block by block** as its markdown blocks close, so a long answer moves into
the scrollback while it is being written and the live region holds only the
paragraph in progress. Verified on the QAT Gemma file: one insertion for the
prompt, one per paragraph, one for the code block, one for the stop marker.

Rendering is incremental while streaming: closed markdown blocks are
styled, the open block is raw. Nested lists, code blocks with a language
label, table column alignment.

*Implemented (TERM-01 step 6), except the event union itself, which is step 7.*
`markdown.split` draws the boundary (a blank line, or a heading, rule,
quote, or list item at its newline, closes a block; a paragraph, a table,
and an open fence keep theirs open), and the agent renders the closed part
and shows the open part as raw text on every repaint. Two rules the
proposal did not state, both of them tested: a trailing newline is the end
of the last block rather than an empty one after it, which is what makes
the rendering of a partial turn a prefix of the finished turn's; and list
nesting is read from the raw line's indentation at two spaces per level
(a tab counts as four), so four-space indentation reads as two levels — a
difference of indent only, since the bullet glyphs cycle. The transcript
also joined the editor's word wrap here: everything a reader reads breaks
at the last space that fits, and only rows truncated to one line (the
status bar, the hint) still break at the column.

#### Status and progress (proposed)

*The bar's layout (2026-09-21).* Two groups: the measurements on the left
(the state word, prefill or decode progress, the loop step, context, token
counts, the prefill and decode rates, `replayed`) and the settings on the
right, right-aligned (`✦ think low │ spec on 4 · 3.13/step` or `spec off`
`│ kv f16 │ metal │ qwen3.8-27b`). A narrow bar drops settings from the
right end one cell at a time, then truncates the measurements at the
column. `spec` states the switch as the session runs it: on only when a
draft source is loaded, with the draft length, and the accepted drafts per
step after a speculative turn.

The engine's between-layer `check` callback gains a sibling `progress`
callback carrying phase, position, and target (additive; `generate` and
`bench` do not register it). The status bar shows `prefill 768/1226` per
chunk with an ETA from the measured rate, the spinner animates from that
callback, and decode shows live tok/s and elapsed time. The step-based
counter that exists today goes.

*Implemented (TERM-01 step 5).* `inference.observer.Progress` is
`{phase, position, target}`; the two Metal plans call it after every
prefill chunk, `runLoop` after every prompt token on the unchunked path and
after every generated token. `generate.Trace` grew an optional `Sink` so
the observer stays one value and `generate`/`bench` install nothing. The
agent's per-token `step` hook is gone: `progress` is now its only beat, and
the prefill counter is measured rather than guessed at one token per call.
Measured on the QAT Gemma file, 993 prompt tokens: the bar steps
`256/993 ~4s`, `512/993 ~2s`, `768/993 ~1s`.

#### Keys and commands (proposed)

Keys as implemented today plus: an expand key on a chip, Up/Down inside a
multi-line input as above, and Ctrl-O reserved (transcript overlay, not
planned). Slash commands, session 2:

| Command | Effect |
| --- | --- |
| `/new` | New session (as Ctrl-N) |
| `/ctx <n>` | Context window (as Ctrl-W, with a value) |
| `/think <effort>` | Reasoning effort (as Ctrl-T, with a value) |
| `/save [path]` | Export the transcript as markdown (default under `~/.nuclis/agent/exports/`) |
| `/help` | Keys and commands inside the live region |
| `/resume` | Session picker: replay a saved session (AGNT-07) |

*Implemented (TERM-01 step 9), `/resume` included (AGNT-07).* The decisions the
proposal left open:

- **What is a command.** A submitted line is a prompt unless it starts with
  `/` *and* its first word is nothing but ASCII letters. `/ctx 16384` is an
  instruction; `/usr/bin/env is fine` and `what is 1/2?` are questions. A
  `/word` that names nothing is a dim notice listing the known commands
  rather than a turn spent on the model.
- **Tab completes, Enter sends.** With a list open, Up/Down move the
  selection and Tab takes it; Enter always submits, so a message can never be
  eaten by a completion. Accepting a command adds the space its argument
  needs; accepting a directory keeps its `/`, so the next Tab walks into it.
  The list is rebuilt on every key rather than opened and closed, so it can
  never disagree with what is typed.
- **`/help` is a notice block, not an overlay.** It goes into the scrollback
  with everything else, where it can be scrolled back to and copied. Its text
  is generated from the same table the parser and the completion list read.
- **Steering.** The editor stays live while a turn runs — including pastes,
  which used to be dropped — and the cursor stays visible in it. Enter queues
  the message and clears the editor; when the turn ends, the queued text goes
  back into the editor and is submitted without a keystroke. A queued message
  is not in the transcript until it is actually sent, so it can still be
  edited if the turn is cancelled first.
- **`@path` completion** lists the working directory's entries,
  workspace-relative, hidden ones only when the prefix asks, at most 50.
- **The picker.** `/resume` (`AGNT-07`) opens the module's `Choice` list over the
  workspace's saved sessions, newest first (id prefix, time, effort, context
  size, directory); Up/Down move, Enter or Tab picks, Ctrl-C closes. The
  chosen conversation is replayed into a **fresh session** — loader, then
  `loop.Agent.restore`, then a transcript replay — and the fresh file is
  seeded with the saved entries. `--resume [<id>]` does the same before the
  first prompt — alone it continues the workspace's newest session — and an
  argument that is a path is used as given, which is how print mode scripts
  a specific file. `nuclis agent ls [--json]` lists the workspace's sessions
  (id, time, effort, window, first prompt), the same rows the picker shows.

#### Print mode (proposed)

`nuclis agent -p "<prompt>"` (or `--print` with `--prompt-file`) runs one
turn without a TTY and writes events as plain text, or as one JSON object
per line with `--json`. It is the model-free test harness for the agent
loop once tools exist and the scripting entry point for the engine. Print
mode does not write a session file unless `--session` names one.

*Implemented (TERM-01 step 10)* as `src/agent/print.zig`. The text form is the
**answer alone**, streamed as it arrives — which is what the terminal
surface showed, since thinking is folded there by default and the markdown
is the model's own. A turn that produced no answer at all (the budget spent
on reasoning, a cancellation) says so on stderr, where it cannot disturb
what a script is reading. The `--json` form is every event, including the
reasoning deltas and the progress beats, one object per line with the union
tag as `type`; that serializer lives on `tui.event` (writing to a writer is
not file I/O), while the session file's different shape stays in
`src/agent/session.zig`. `--json` without `--print` is an error: an
interactive surface has no JSON form.

The two surfaces consume `engine.complete` events (AGNT-01 session 1).
`src/agent/stream.zig` maps the inference union to the TUI union; the profile
owns reasoning-channel decoding and UTF-8 boundaries. History and session
logging retain these same channels without parsing accumulated output.

### Module API (`src/tui/`)

The rule: `src/tui/` imports nothing from `inference` and knows nothing
about tokens, sessions, or models. `src/agent/` owns the engine, the
session log, the tool registry (phase 2), and drives the TUI with events
and a tick.

| Type | Owns | Pure surface for tests |
| --- | --- | --- |
| `Event` | the union above; owned strings | serialization to and from JSON (the session file) |
| `diff` | a pure old/new content → structured rows + unified text; owned bytes, no theme, no I/O | the diff fixtures and hunk/span computation |
| `Transcript` | blocks (closed and open), fold state; writes a closed block to the scrollback sink exactly once | `render(block, width) → lines` |
| `Editor` | buffer, chip ranges, cursor, draft, history; `handleKey(key) → Action` (`none`, `submit`, `command`, `cancel`, `quit`, `fold`, …) | `layout(width, max_rows) → lines + cursor` |
| `Status` | phase, position, target, rates, elapsed, effort, context use, step budget | `paint(width) → line` |
| `Choice` | an inline list with a selection; used by pickers and completion | `layout(width, max_rows)` |
| `Theme` | the semantic style enum, named palettes (`gruvbox-dark` first), the colour level detected from the environment, the glyph set (Unicode or ASCII) | `paint(style) → sgr` as a pure table lookup |
| `Screen` | the terminal lease (raw mode, kitty keys, bracketed paste, resize), the input loop yielding `key`, `paste`, `resize`, `tick`, the live-region repaint with synchronized output, insertion above the region | writes to a generic writer so golden tests capture the escape stream |

Only `Screen` performs I/O and takes an `Io`. Everything above it is a
value type over an allocator. *Implemented (TERM-01 step 1):* the existing
files (`terminal`, `keys`, `view`, `markdown`, `highlight`, `theme`, plus
`style`, the same palette for one-shot command reports) moved into
`src/tui/` behind a `root.zig` that re-exports them, and `src/agent/`
holds the composition; the types above are folded in as the later steps
touch them.

*Step 7 added `event`, `transcript`, `status`, and `choice`, and closed the
module's defining rule: `src/tui/` now imports nothing from `inference` at
all.* AGNT-01 session 1 subsequently removed `view.parts` and its marker pair:
profiles now deliver semantic channels. The engine's `Phase` and `StopReason`
are still mirrored by terminal types and mapped by the agent. `Status` is a
value folded from the turn's `status` and `turn_end` events plus the
settings only the agent knows, with a pure `paint(width, palette, frame,
elapsed)` — so the bar's layout, its `—` for an unmeasured rate, and its
`~` countdown are tested without a model or a clock. `Status` also carries the
loop's step in progress against its budget, which the bar paints as
`step N/M` while a turn runs. `Choice` exists
before its first user (completion in step 9, `/resume` in phase 2, both now
shipped): an
inline list that copies its items, keeps the selection when the list is
rebuilt under it, wraps at both ends, and scrolls only as far as the
selection needs.

### Phase 1 acceptance

- Golden tests without a TTY: editor layout (wrap, chip, scroll indicator,
  cursor placement), renderer output for a fixture document covering every
  construct, transcript rows for each event kind, the escape stream of one
  repaint and one insertion, a session file round trip.
- `make check`; `generate` and `bench` unchanged.
- Manual checklist recorded in the engineering log: the 1,226-token test
  prompt pastes as one chip and sends as one turn with progress advancing
  per chunk; the prompt is visible in full above the live region while the
  turn runs; a resize mid-turn recovers; the transcript is intact in the
  scrollback after exit; a print-mode turn writes the same text the TUI
  showed.

## Sessions and storage

Decided 2026-09-09; the layout is proposed and the implementing sub-unit
fixes it.

```text
~/.nuclis/agent/
  sessions/<cwd-slug>/<timestamp>_<id>.jsonl   one file per session
  history.jsonl                                 submitted prompts across sessions
  exports/                                      /save markdown exports
```

`<cwd-slug>` is the canonical working directory with the leading separator
removed and every `/` replaced by `-`, truncated with a hash suffix when it
exceeds 200 bytes; `<id>` is a random 128-bit identifier in hex.

The first line is a header; every later line is an entry:

```jsonl
{"type":"session","version":1,"id":"…","time":"2026-09-09T10:12:03Z","cwd":"/Users/…/nuclis","model":{"path":"…","sha256":"…"},"effort":"low","ctx_size":8192}
{"type":"user","id":1,"parent":null,"time":"…","text":"…"}
{"type":"assistant","id":2,"parent":1,"time":"…","thinking":"…","answer":"…","tool_calls":[],"stop":"eos","stats":{…}}
{"type":"tool_result","id":3,"parent":2,"call":"…","text":"…","truncated":false,"is_error":false,"summary":"lines 1 to 40 of 96"}
{"type":"effort","id":4,"parent":3,"effort":"medium"}
{"type":"context","id":5,"parent":4,"ctx_size":16384}
{"type":"compaction","id":6,"parent":5,"first_kept":2,"reason":"context_full"}
{"type":"compaction","id":7,"parent":6,"first_kept":8,"reason":"results_elided"}
{"type":"notice","id":8,"parent":7,"text":"…"}
```

Rules:

- Append-only; one `write` per entry, flushed at turn boundaries. A file
  whose last line does not parse is loaded without that line; anything
  else that does not parse is a typed error naming the line.
- Every entry carries `id` and `parent` even though phase 1 writes a
  linear chain. They make branching and a cancel-rewind representable
  later without a format migration.
- The header `version` is the migration key; a reader rejects a newer
  version with a typed error and never rewrites a file it did not create.
- Entries store what the model saw and produced (thinking included, so a
  resumed session renders the same prompt), never terminal styling. An
  assistant entry's `tool_calls` carry the host correlation ids the following
  `tool_result` entries answer (`AGNT-07`); a cancelled step records none, because
  a call without its result would not load.
- Resuming (implemented by `AGNT-07` as `src/agent/resume.zig`: `/resume` and
  `--resume [<id>]`) replays the kept conversation through the profile into a
  fresh session — the loader rebuilds the message list, `Agent.restore`
  continues the host ids, and the fresh file is seeded with the saved entries.
  It is a prefill, not a state restore. ENGN-06's in-memory snapshot is the other
  mechanism and is never persisted.
- `/save` derives markdown from the same entries; there is no second
  transcript format.
- Retention is the user's: nuclis never deletes a session file. Print mode
  and tests write nothing unless asked.
- The files are plaintext under the user's file permissions; tool output
  lands in them verbatim.

*Implemented (TERM-01 step 8)* as `src/agent/session.zig`, with four decisions
the layout above left open and one deviation:

- **The file is created lazily**, at the first entry. Starting the agent and
  quitting writes nothing, which is also how print mode and the tests stay
  silent without a flag.
- **The timestamp in the name is compact** (`20260912T101203Z`): the same
  instant as the header's RFC 3339 value, in a name that sorts and needs no
  quoting.
- **An unknown entry `type` is malformed**, not skipped. The header's
  `version` is the migration key, so a reader that meets a type it does not
  know is reading a file it should not be reading — the one exception stays
  the truncated *last* line, where a crash lands.
- **A failed write never costs a turn.** The reason is said once, dimly, in
  the transcript, and the conversation continues: retention is the user's,
  and a full disk is not a reason to lose what the model just wrote.
- *Deviation:* serialization lives in the session module rather than on
  `tui.event` as the module table said. An entry is not an event — a turn
  produces hundreds of `answer_delta` events and exactly one `assistant`
  entry — and the file records the turn, not the keystrokes of the model.

The header's `model.sha256` is the digest the pull sidecar recorded
(`<file>.nuclis.json`), read at startup; a model nuclis never verified simply
omits the key rather than paying for a digest of six gigabytes.

## Phase 2 — the agent loop

Borrows the shape of pi's coding agent (earendil-works/pi): a small state
machine, a handful of tools, a minimal system prompt, and no configuration
surface beyond what the model needs.

### Goal

Give `nuclis agent` the ability to complete small coding tasks in the
current working directory with the local model: inspect files, run
commands, apply edits. It reuses the phase-1 surface unchanged (transcript,
editor, compaction, sessions) and adds a tool layer under it.

### Design principles

1. **Minimal system prompt.** Identity, workspace note, the rendered tool
   definitions, and a few behavioural lines: call tools with the model's
   native tool-call syntax, batch independent calls, never fabricate
   results, treat tool errors as information.
2. **Few, strong tools.** `read_file`, `write_file`, `edit_file`, `glob`,
   `grep`, `bash`. Names and contracts mirror what an external agent
   product would use, so prompts and transcripts transfer.
3. **Bounded everything.** Output sizes, result counts, execution time, and
   step count are host constants. Truncation is always marked. Exceeding a
   limit is a typed tool result the model can respond to, not an abort. On
   top of the tools' ceilings, one result may not exceed an eighth of the
   context window in tokens (never below 256): the loop counts it with the
   model's tokenizer, cuts it at a line boundary, and appends a note that
   says how many lines were shown and how to ask for the rest (`read_file`
   gets the offset to continue from). `read_file` reads 200 lines unless
   asked for more, so a whole file is a choice, not a default. When a step
   still does not fit, the loop first replaces the turn's older tool
   results (all but the last two) with one-line stubs naming the call and
   its size, then drops whole earlier turns, each recorded as a
   `compaction` entry; only then does the turn end, with a message that
   names the tokens needed and the window, and the flag or command that
   raises it (AGNT-08).
4. **Failures are results.** Unknown tool, invalid arguments, timeout, and
   ordinary tool failure return structured errors to the model. Only
   inference-transport failure ends the turn.

### Architecture: two boundaries

```text
src/agent/                                inference
  agent loop state machine  ◄──────────►  qwen38 profile
     │  typed events (Event)                │  tools block + tool roles (encode)
     │                                      │  streaming tool-call parser (decode)
     ▼                                      │  pinned template fixtures
  tool registry: validate → execute → typed result
     │
     ▼
  src/tui/ (transcript, editor, status, screen)
```

#### Format boundary — inference side (AGNT-01)

The model's native tool-call syntax is profile knowledge and stays in the
prompt profile. AGNT-01 session 1 implements `engine.complete` and typed
thinking/answer/stop events; `tool_call` was reserved until the profile could
produce it. Session 2 landed the render/tool-history inputs and their shared
validation; the profile now renders Qwen's native tools and decodes generated
calls into `tool_call` events. The [model evidence](reference/tool-calling.md)
distinguishes Qwen's XML-like calls from Gemma's own grammar and tool-response
handoff. Neither is a generic Hermes JSON payload.

- **Engine API.** `render(messages, tools, effort)` through the profile;
  `complete(eng, tokens, limit, sampler, history, buffers, observer, sink)` streaming typed events
  (`thinking`, `answer`, `tool_call {id, name, arguments}`, `stop` with
  reason and metrics) plus cancellation and the GPU top-k path that
  `runLoop` has today. Adapters register explicitly and the engine
  dispatches on `general.architecture` (landed as MODL-04 on 2026-09-11:
  `inference.models.table`, `profiles.Profile`;
  [architecture § 9](architecture.md#9-adding-a-model)). The executable keeps presentation
  only: `generate` writes text, `bench` aggregates timings, the agent
  renders and executes. As of AGNT-01 session 2, `tools` are the shared
  `profiles.ToolDefinition` list. `profiles/qwen38.zig` renders the
  native tool path (AGNT-05) and `profiles/gemma4.zig` its own (AGNT-09:
  the `<|tool>` declarations, `call:NAME{…}` calls, results inside the model
  turn, and `<|tool_response>` as the handoff stop token).
- **Encode.** `profiles.validate` (shared) enforces the conversation rules
  before any profile renders: system/developer only lead, reasoning only on
  the assistant, and tool results answer the pending assistant calls in
  execution order, matched by host ID, with orphan, duplicate, out-of-order,
  and missing results rejected. A completion-ready conversation ends with a
  user message or a fully answered tool-result group. `profiles/qwen38.zig`
  renders tool definitions into the system block, assistant history carrying
  tool calls, and tool results as the pinned GGUF template's folded
  `<tool_response>` user turn, byte for byte against pinned fixtures.
- **Decode.** A streaming parser converts generated output into typed
  events. It knows the token-boundary facts (special-token detection) that
  app code cannot access. The shared profile decoder reads the two bracket
  control tokens, collects the body, and parses it with the profile's grammar;
  a call still open at EOS, a budget stop, or a cancellation — and a body the
  grammar refuses — is released as answer text and never surfaced for
  execution. Malformed arguments become a typed error result.
- **Fixtures.** Tool-path rendering and parsing are pinned like the text
  path: expected strings captured once by executing the artifact's chat
  template on representative inputs (tools block, zero/one/two calls, tool
  responses, × reasoning effort), stored beside `qwen38-text.json`. The
  fixtures are `qwen38-tools.json`; the parse tests reuse them as round-trip
  inputs.

Acceptance for AGNT-01: the event stream is tested against a stubbed
completion without a model; `generate`/`bench`/agent unchanged; no `src/`
file imports a model adapter directly.

#### Agent loop — application side

The agent owns orchestration only. A turn is a loop over steps; a step is
one completion request plus the sequential execution of every tool call in
the response:

1. Render instructions, history, and tool schemas through the profile.
2. Stream the completion; forward events to the transcript as they arrive.
3. After the response completes, execute each tool call in order: validate
   arguments against the registry, execute, append the typed result to
   history and to the session file.
4. Send results back to the model, or finish when the response contains no
   tool calls.
5. Stop on cancellation, an unrecoverable failure, or the step budget
   (default 16); publish the stop reason to the transcript.

A tool call renders as two rows and its result text never reaches the
transcript: the call row is a dot and `Name(argument)` (`● Read(TODO.md)`,
`● Bash(make test)`, `● Grep(needle)`; an argument longer than 72 cells is
cut on the row and repeated in full on a detail row, `└ $ …` for a command),
and the detail row is one sentence the tool supplies with its result
(`└ lines 1 to 40 of 96 · truncated, continue with offset=41`, `└ 16
files`, `└ 3 matches in 2 files`, `└ Wrote 12 lines to a.py`, `└ Edited
a.py: +12 −3 lines`; a clean `bash` run adds nothing). The dot's colour is
the call's state: green settled well, red failed, blue for a write or an
edit, and while the call runs it pulses between yellow and dim on the
repaint cadence, so a long command stays visibly alive and cancellable from
the keyboard. Where the model's text resumes after a run of calls, and at
the end of a turn that made any, one dim row sums them up (`Read 2 files,
searched 1 time, ran 1 shell command, wrote 1 file`). A failed call shows
its message in error style, bounded to three rows. The result text
goes to the model and the session file, whose `tool_result` entries keep the
summary so a resumed session renders the same rows (TERM-05). For
`write_file`/`edit_file` a diff derived by a pure function before execution
follows. Edit arguments are validated (exact, unique, nonempty replacement)
before anything is touched. The transcript renders the diff width-adaptively:
side-by-side (old | new, line numbers, changed span highlighted) when the
terminal is wide enough and the unified single-column form below that, with
long diffs folded; the unified text is what the model and session record
(AGNT-04). A queued steering message is inserted as the next user message when the
current step ends.

The session is **primed** before the first prompt (ENGN-09): the completer
renders the profile's prefix — the system block with the tool definitions,
without a generation prompt, pinned as a byte prefix of every rendering by
each profile — prefills it, records it as consumed, and keeps a snapshot of
the session. The first turn then prefills only its own message; a new
session, a resume, and the replay after elision restore the snapshot instead
of prefilling the prefix again. The warm-up is visible: the live region
shows `⠋ warming up… system prompt and tools 256/934 ~14s` with the bar's
own estimate, repainted at the engine tick's cadence (about 10 Hz through
every GPU wait, not once per prefill chunk), and the transcript gets `—
warmed up in 11.2s · 934 tokens` when it is done; Enter queues meanwhile,
and a window too small for the prefix
plus the output budget is a notice at startup (an error diagnostic in print
mode), not a `ContextFull` on the first Enter. A `/ctx` change re-opens the
engine and primes again; a `/think` change alters the system block and
primes again.

*Implemented (AGNT-02; profile decode in AGNT-06)* as `src/agent/loop.zig`, with the
engine completion in `loop.Completer`. Decode lives in the profile now: the
loop consumes typed `tool_call` events and keeps native history (assistant
messages with `tool_calls`, `.tool` result messages), so the in-app parser it
started with is gone. Two seams — `Model` (completion) and
`Events` (display and session) — let the loop be tested against a stubbed
completion with no model and no terminal; both the interactive surface and
print mode are `Events` sinks over the same agent. The single-completion
`stream.Dispatcher` and the driver's `generateTurn` were removed; the agent
now owns the message history, the consumed-text bookkeeping, the step budget,
and compaction (drop the oldest prior turn and retry when a render no longer
fits). A cancelled step is recorded for display but not appended to history,
since the model never saw its end.

### Tools

Workspace root is the process's working directory, canonicalized at
startup. Limits are host constants, not model-supplied.

| Tool | Contract |
| --- | --- |
| `read_file` | Line-addressed bounded region: offset + count, 2,000 lines / 1 MiB per call. Non-UTF-8 files produce a typed error. |
| `write_file` | Create or replace a text file, 1 MiB content limit; temp-file replacement, no symlink escape. |
| `edit_file` | Replace one exact, nonempty byte sequence; fail without writing on zero or multiple matches. |
| `glob` | One `*`/`?`/class/`**` pattern; hidden entries excluded unless named; ≤ 200 workspace-relative results, stable order. |
| `grep` | Literal, case-sensitive search; skips binary-looking files and hidden dirs; ≤ 200 matches per call. Native Zig. |
| `bash` | One command; bounded combined output (1 MiB), 300 s timeout, cancellation reaps the child; minimal child environment. |

*Implemented.* `read_file` and `glob` landed with the tool registry; AGNT-03 added
`grep` and `bash` ([engineering log](engineering-log.md#agnt-03--read-tools-and-bash-2026-09-14)).
`grep` is literal and case-sensitive and skips hidden entries, symlinks,
binary-looking files, and a fixed generated-tree list, bounding matches and
bytes. `bash` folds the child's stderr into its stdout at the shell level,
bounds combined output and wall time, kills and reaps on cancellation, timeout,
or the output bound, and passes only a minimal environment (`PATH`, `HOME`,
`LANG`, `TERM`, `TMPDIR`) so a secret in the parent environment is not
reachable from a model-issued command.

AGNT-04 added the two mutations ([engineering log](engineering-log.md#agnt-04--mutations-and-diffs-2026-09-14)).
`write_file` creates or replaces a UTF-8 text file and writes it by
temp-file replacement, so a crash leaves either the old file or the new one;
`edit_file` replaces one exact, non-empty byte sequence and fails without
writing on zero or multiple matches. Both resolve the target through
`Workspace.resolveTarget`, which permits a not-yet-existing file by
canonicalizing its parent and refuses an existing path that canonicalizes
outside the workspace — including through a symlink. A mutation returns a
structured diff (`tui.diff`), which the loop turns into the `diff` event and,
for the model and session, the unified text.

### Compaction and context

Tool results are ordinary turn content: the turn-dropping compaction
applies unchanged. Risks specific to agentic use, now measured (`AGNT-07`;
[log](engineering-log.md#agnt-07--polish-resume-and-the-agent-plan-closed-2026-09-14)):
the six-tool block the Qwen profile renders is 2,896 bytes / 720 prompt tokens
(about 9% of an 8,192-token context before any conversation); a representative
source read runs about 20 tokens per line, so a 200-line read is ~4,000 tokens
and `read_file`'s 2,000-line cap is on the order of 40,000 tokens, five times
an 8K context. `read_file` defaults stay modest and truncation is marked; each
executed call is a new completion request, so prefill cost grows per step.

### Relationship to external agent products

Nuclis depends on no external agent product. What it fixes deliberately is
vocabulary and contracts that mirror such products (tool names, limits,
the division of labour: native tool-call conversion in `inference`,
execution in the agent), so prompts and transcripts transfer. When nuclis
serves an OpenAI-compatible endpoint, the tool-call parser lands under it
and an external agent consumes typed calls through the protocol.

### Delivery sequence (AGNT-02–AGNT-07)

One unit each, after AGNT-01:

1. **AGNT-02 Loop prototype** (closed 2026-09-14). State machine in `src/agent/`
   with a provisional in-app parser and a read-only pair (`read_file`,
   `glob`); an echo-style self-check call exercises the loop end-to-end;
   print mode drives the tests. See the
   [engineering log](engineering-log.md#agnt-02--the-agent-loop-state-machine-provisional-parser-read-tools-2026-09-14).
2. **AGNT-03 Read tools and bash** (closed 2026-09-14). `read_file`, `glob`,
   `grep`, `bash` with the contracts above; transcript rendering for calls
   and results.
3. **AGNT-04 Mutations** (closed 2026-09-14). `write_file`, `edit_file`, the
   structured diff event and its width-adaptive rendering, humanized tool-call
   lines, and the polling `Tick` that lets a running `bash` read Ctrl-C. See
   the [engineering log](engineering-log.md#agnt-04--mutations-and-diffs-2026-09-14).
4. **AGNT-05 Tool rendering** (closed 2026-09-14). `profiles/qwen38.zig` renders
   the artifact's tools block, native assistant calls, and folded tool
   results, pinned by `qwen38-tools.json` captured from the reference; Gemma
   recorded its `ToolsUnsupported` position until AGNT-09 (2026-09-16) gave it
   the same treatment with `gemma4-tools.json`. See the
   [engineering log](engineering-log.md#agnt-05--qwen-tool-rendering-and-pinned-fixtures-2026-09-14).
5. **AGNT-06 Tool decoding and the loop** (closed 2026-09-14). The profile
   decoder parses generated calls at token boundaries; the agent loop consumes
   them and carries native tool history; the provisional parser was removed.
   See the
   [engineering log](engineering-log.md#agnt-06--tool-call-decoding-and-the-profile-driven-loop-2026-09-14).
6. **AGNT-07 Polish and resume** (closed 2026-09-14). The assistant entry records
   the decoded calls; `/resume` and `--resume <id>` replay a session; the
   status bar shows the step budget; the tools block and the tool-output fill
   are measured. See the
   [engineering log](engineering-log.md#agnt-07--polish-resume-and-the-agent-plan-closed-2026-09-14).

The plan closes here. The engine work that follows is planned in
[TODO.md](../TODO.md); the one-time repository reset is a separate,
post-plan action, not a unit of this plan.

Default tests need no model: parser tests, tool contracts against temporary
workspaces, loop tests against a stubbed completion stream through print
mode.

### Open questions

- Whether tool steps use different sampling defaults for reliability;
  measure before changing anything.
- Whether a `bash` allow-list or an approval step is wanted once the loop
  has run on real tasks; the `Choice` component exists either way.

### Future consideration: docs and API lookup (2026-09-14; recorded, not planned)

Discussed while planning AGNT-03/AGNT-04; the assumed goal is a coding agent looking
up docs and APIs. Nothing here is scheduled, and the six-tool decision stands
until a unit changes it.

**Findings.** The loop is format-agnostic, so a seventh tool is a registry
entry and a system-prompt line — the cost is not mechanical. It is network
access, credentials, untrusted content, and reproducibility, and it intersects
two standing decisions: **no permission system** (every tool executes
immediately) and **default tests require neither network access nor a large
model**. Web text is the strongest prompt-injection vector the agent would
have, and paired with `bash` (AGNT-03) an injected page becomes command execution.
`bash` already reaches the network through its own commands, with the same risk
and none of the structure or bounds a dedicated tool would add.

**Options, cheapest first, for the "coding agent looks up docs" goal.**

- **A. Read-only docs roots.** Keep the workspace the only writable root and
  add configured read-only roots (vendored docs, a checked-out library, the
  Zig std tree) that `read_file`/`glob`/`grep` can address. No network, no new
  index, offline and reproducible, and it reuses the AGNT-03 tools. The first thing
  to build if *local* API lookup is the need.
- **B. Bounded `fetch_url`.** HTTPS GET with host/scheme/redirect/timeout/
  content-type/byte limits, HTML-to-text, results marked untrusted and logged
  with provenance, transport injected so default tests stub it. This is what
  reaches external docs; it carries the security and reproducibility decisions
  above, so it is its own unit, not a AGNT-03/AGNT-04 add-on.
- **C. Lexical local index.** Only when a large, stable corpus on disk needs
  *ranking* that exact match cannot give: a pure-Zig inverted index (BM25) over
  configured docs roots, returning ranked chunks with path and line ranges,
  built offline and pinnable as a fixture. Pairs with A.
- **D. Embedding (semantic) index — ruled out.** It needs a separate embedding
  model class (loader, numerical path, quantization) and ranks
  non-deterministically across artifacts, against the project's pinned-fixture
  posture.

**Recommendation for a future session.** Choose by corpus: workspace code needs
only A plus `grep`; external docs need B; a mirrored doc set needs A and then
C; D is out. If B lands, revisit the no-permission-system decision for its
interaction with `bash`.

## Configuration

The `agent` section of `nuclis.json` (renamed from `chat` on 2026-09-11,
schema version unchanged because the keys keep their defaults; a `chat`
section is rejected with a message naming the rename): `think`,
`fold_thinking`, and `theme` (default `gruvbox-dark`; an unknown name is a
typed configuration error naming the known themes). Phase 2 adds nothing
that the model does not need; the step budget and tool limits are host
constants. Print mode and `--no-session` are flags, not configuration.

## Not in scope

HTTP serving, a permission system, MCP or extensions, subagents, image
input (the Gemma 4 vision unit gives the agent no image tool), and a
second product surface. The spec's [deferred list](spec.md#local-serving-and-deferred-work)
remains the boundary. A docs/API lookup tool (read-only docs roots, a bounded
fetch, or a local lexical index) is recorded under
[Future consideration](#future-consideration-docs-and-api-lookup-2026-09-14-recorded-not-planned);
it is not scheduled.
