# Agent and terminal concepts

Companion to implementation, teaching the agent and terminal concepts behind
`nuclis agent`: how a turn reaches the terminal, how model output becomes typed
events, and how the tool loop executes a call. The design is in
[agent-spec.md](../agent-spec.md); the inference stack is taught in the
companion [llm-guide.md](../llm-guide.md).

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
delimited by markers ([§46 of the inference companion](../llm-guide.md#46-a-prompt-profile-is-a-contract-template-stop-set-and-reasoning-markers)). Splitting it *incrementally* has a trap: while the
channel is open, the first bytes of a half-arrived `</think>` look exactly like
reasoning text, and a naive splitter puts `</thi` on the screen for one token
before reclassing it. Keeping back one marker's length minus one byte makes
that impossible, and costs nothing — the held bytes arrive with the next token.
But then the *end* of a turn must release them explicitly, or a turn cancelled
mid-thought loses its last characters. That bug lived in the inline version
until the splitter became a module with tests of its own; it is the kind of
thing that only shows up when a component can be driven byte by byte without a
model, a clock, or a terminal.

## 2. Why a streaming parser needs token boundaries

The final interpretation of model output belongs to the inference profile. A
sampled token arrives as an integer ID; decoding it to
bytes discards useful information. A real `</think>` token and several
ordinary tokens spelling `</think>` can produce identical text, but only the
former closes the reasoning channel. The application-level splitter (§1)
could not distinguish them, however carefully it held back partial text.

`engine.complete` runs the existing numerical loop and feeds each ID plus its
decoded piece into `Profile.decoder`. Qwen's closing token and Gemma's channel
closing token select the answer channel. Gemma's opener also carries ordinary
`thought\n` text after the special token; only that short header suffix needs
buffering across pieces. Reasoning content can otherwise be delivered at once.
UTF-8 repair runs inside each channel, so half a character cannot join bytes
on the other side of a control token. Tests exercise every split of a
multibyte string and literal marker text using fake token IDs.

The output is an `events.Event` union: thinking, answer, a reserved tool-call
variant, or stop with the loop's metrics. In Zig, `sink: anytype` means the
compiler checks and specializes the call to `sink.send(event)` for its actual
consumer. Event strings borrow the decoder's reusable scratch only until that
call returns. The agent copies the two text channels into owned writers, then
uses those same values for the transcript, session log, and next prompt.
There is no second interpretation at history replay time.

A tool call uses the same boundary: the profile converts its native
syntax into a complete call with host correlation ID, name, and JSON-object
arguments. The agent validates the registered tool and executes it only after
the completion hands back control. Qwen and Gemma serialize calls and results
differently ([the evidence](tool-calling.md)); Qwen's native path is
implemented (§8), and Gemma's remains unsupported until its own fixtures exist.
`generate` and `bench` retain the raw `runLoop` path.

## 3. A tool call is host data until a profile serializes it

The other half of the tool seam is the conversation types a profile will
encode. The interesting decision is what is *shared* and what is
*profile-owned*. A model's wire syntax — Qwen's `<tool_call>` blocks, Gemma's
`call:NAME{...}` grammar — is profile knowledge, because only the profile's
pinned template can say how a call is spelled. But the agent needs to reason
about calls before any of that: which call a result answers, whether a result
is missing, whether the conversation is ready for another completion. That
reasoning must not depend on either wire format, so it lives in shared types:
`ToolCall {id, name, arguments}`, `ToolDefinition {name, description,
parameters}`, `Role.tool`, and a `tool_call_id` on the result message.

The `id` deserves a note. Neither Qwen nor Gemma promises to put a stable
identifier on a call, and the model has no way to echo the host's notion of
one. A short numeric ID (`u32`, matching the transcript) is therefore a
*host* correlation token, assigned when the assistant's calls are parsed and
matched back, by execution order, to the results. Making it explicit is what
lets the validator reject an orphan result (nothing pending), a duplicate
(already answered), a missing one (a user turn arrives mid-group), or an
out-of-order pair — errors that a bare role tag cannot express. The same type
serves the completion side: `events.ToolCall` *is* `profiles.ToolCall`, so
the decoder and the history cannot drift.

Validation is deliberately separate from support. `profiles.validate` accepts
a structurally valid tools input; it does not decide whether a profile can
render it. Each profile asks that second question. Qwen answered yes once its
template was pinned; Gemma still answers no with `error.ToolsUnsupported`
until its distinct grammar and result handoff get their own fixtures. That
ordering is the point: the loop was built and tested against the shared
contract while the profile refused to guess at a wire format it had no
fixtures for — the same rule that keeps a mismatched template from being
rendered at all.

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
which is the same lesson as §2: the meaning of a byte depends on what came
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
surface for a project that keeps its build offline. `scripts/grapheme-table.py`
instead downloads a **pinned UCD** once, checks its digests, and emits one
sorted interval table (`graphemes_table.zig`) that a binary search reads. The
generator is committed, the artifact is committed, and the data layer sits
behind `graphemes.zig`, so swapping in `uucode` later is a change to one file.
The correctness evidence is the Unicode project's own conformance file:
`GraphemeBreakTest.txt` (766 cases at 17.0.0) runs through the same iterator
the editor and the wrapper use.

## 5. A turn is a loop: steps, the two seams, and a parser that must hold back

Chat is a single completion: render the conversation, stream an answer, stop.
An *agent* turn is a loop. The model may answer with a request to run a tool
instead of (or before) an answer, and the host has to execute it and hand the
result back before the model can continue. `src/agent/loop.zig` is that loop:
render the history, complete one step, parse the response for calls, execute
every call in order, append the results, and repeat — until a response carries
no calls, a step budget is spent (16), or the step is cancelled. A step that
was cancelled is *not* appended to the history: the model never saw its end,
so it is display-only.

The loop is deliberately the one piece the terminal surface and `--print`
share, and neither should be able to make it behave differently. So it is
written against two small seams rather than against an engine and a terminal.
`Model.run(messages)` is the completion side: the real `Completer` renders
through `inference.engine` and, crucially, keeps the bookkeeping that lets a
growing conversation prefill only its *new* suffix (`seen`), replaying from an
empty session when the render no longer starts with what the session consumed.
`Events` is the presentation side: `send` for terminal events and `record` for
session entries. A test implements both with a scripted answer and a recorder,
which is why the loop's budget, cancellation, and tool-execution paths are
pinned without a model or a GPU.

The parse lives in the prompt profile. Native tool decoding belongs to the
same place that owns the tokenizer's special-token boundaries (§2), and it now
lives there: the shared `profiles/stream.zig` decoder holds the call's
bracket token ids and a body parser, and the provisional in-app splitter,
`src/agent/parse.zig`, is gone. Streaming still shapes the problem, and the
state machine is the same idea the provisional parser had: the decoder collects
the body between the two control tokens and never emits a call that is still
open when the stream ends — after EOS, a budget stop, or a cancellation the
bytes are released as plain answer text. It cannot emit `<tool_call>` as
ordinary text and then discover the call, because the opening token is
recognized by id, which is the only signal a byte-stream consumer cannot see.
That is the parser analog of §2's rule: a byte's meaning depends on what came
after it, and the only safe move is to delay the decision. §8 follows the
decoder in detail.

The loop now carries **native** tool history: an assistant message with its
`tool_calls` and one `.tool` result message per call, which the profile renders
in the artifact's own syntax. The earlier folded `<tool_response>` user turn
was the placeholder while the profile boundary caught up; it is gone, so the
wire format lives in the profile and nowhere else.

## 6. Running a command is a lifetime problem

A read tool borrows bytes; a shell tool creates a *process*. That process has
a lifetime, an output stream, and an environment, and all three are the host's
responsibility the moment a model can ask for one.

The bounds are not decoration. Output must stop at a byte cap, because a
command can print forever and the transcript would grow without limit; wall
time must stop at a deadline, because a command can block forever. Both are
enforced by *not reading* once the cap is hit or the deadline passes, and then
killing the child. Cancellation is the third bound and the subtle one: the
engine notices Ctrl-C between layers of the model, but a running child is not
in that loop, so the tool has to poll the same `interrupt` flag between reads
and kill the child itself. The three paths share one cleanup: `kill` blocks
until the process is gone and reaps it, so cancel, timeout, and overflow all
leave no orphan.

Reading output is a second lifetime hazard. stdout and stderr are separate
pipes, and reading one to end before touching the other deadlocks as soon as
the child fills the pipe you are ignoring — the classic reason a
`MultiReader` waits on both at once. But waiting on both loses the *order*
between them, which a build log needs (an error printed between two progress
lines should stay between them). The fix is to fold stderr into stdout at the
shell level, `exec 2>&1`, so there is one ordered stream by construction,
while the empty second pipe is still drained so no writer can block.

The environment is the quiet security boundary. A tool that inherits the
parent's environment hands the model whatever secrets are in it; a command as
innocent as `env` would print them into the conversation. nuclis forwards a
short allowlist — `PATH`, `HOME`, `LANG`, `TERM`, `TMPDIR` — and drops the
rest, so the child is useful and the model still cannot read the host's
tokens. `expand_arg0` is off for the same reason: `/bin/sh` is a path, not a
name resolved through anything the child can influence.

Finally, the split between `grep` and `bash` is deliberate. `grep` is literal,
native, and bounded: one predictable contract that the tool layer can test
without a shell. `bash` is the escape hatch for everything the fixed tools do
not cover — regex, pipeline, `.gitignore` semantics — and it pays for that
power with the process lifetime above.

One more turn of the same screw. When the terminal is in raw mode, Ctrl-C is a
*key*, not a signal — ISIG is off — so the interrupt the engine's observer
checks is never set while a blocking tool runs. The tool cannot poll a flag
that nothing will raise, and the driver cannot read the keyboard while the
tool holds the loop. The seam that closes the gap is a beat: an optional
`Tick` the agent installs on the workspace, invoked by `bash` once per poll
iteration, on which the driver reads the keyboard, repaints, and sets the same
`interrupt` flag the tool is already watching. A tool that never blocks pays
nothing, and print mode installs no tick at all.

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
one-line edit in a thousand-line file stays seven rows. Pairing a removal with
the addition that replaced it is what lets the two-column view put them on one
display row and highlight the bytes that differ, and it falls out of the same
edit script.

Zig note: the module has no theme and no I/O. It allocates owned bytes and is
tested against fixtures (insertion, deletion, replacement, context, CRLF, a
missing trailing newline, the bounded fallback), which is what lets the layout
be a golden test instead of a screenshot.

## 8. Decoding a tool call at the token boundary

A model does not emit JSON. Qwen3.8 emits a call as ordinary tokens wrapped in
two *control* tokens: `<tool_call>` and `</tool_call>`. The wrapper is the only
part the vocabulary marks as special; everything inside — `<function=…>`, each
`<parameter=…>`, the value — is spelled with ordinary tokens that could appear
in prose. That asymmetry is the whole design: the profile can detect a call's
boundaries only while it still has token IDs, and it must then treat the body
as opaque text and parse it separately.

`engine.complete` already hands each generated token to the profile decoder as
`(id, piece)`. The shared decoder (`profiles/stream.zig`) now also holds an
optional tool grammar: the two bracket IDs and a `parse(alloc, body)` function.
On the opening ID it flushes whatever answer text was pending, switches to
collecting the body, and stops emitting text; on the closing ID it parses the
collected body with the profile and emits one `tool_call` event. The body
parser is the profile's (`qwen38.parseTool`), so the wire syntax never leaves
`profiles/` — the agent loop sees typed `ToolCall`s and nothing else.

Two failure modes are decided by the same state machine. A call that is still
open when the stream ends — EOS, the token budget, a cancellation — never
reached its closing token, so it is released to the transcript as plain answer
text, the same bytes the model wrote. A body that arrives complete but that the
parser refuses (no function name, malformed parameters) is likewise released as
text. Neither can reach execution: an incomplete or malformed call is a
message, never an action. This is the tool analogue of §6's rule that a
process is not an instruction until the host has actually built it.

The arguments themselves obey the template's own literal-versus-JSON rule. The
prompt profile renders a string argument literally and anything else as JSON;
the decoder reverses it by trying to parse each value as JSON and falling back
to the literal text. The host still validates the fields against the tool it
names — the profile normalizes the envelope, it does not vouch for the
contents. And because the outer markers are resolved from the artifact's
vocabulary, a template whose controls are not standalone tokens (Gemma's inner
calls, for one) simply declares no tool grammar and keeps rejecting tool input
rather than approximating a format its model never sees.

## 9. Two tool grammars, one loop

A decoder that recognises *a* tool call is not yet a design that recognises
*any* model's tool call, and the two checkpoints nuclis runs make that concrete.
Qwen3.8 wraps a call in two control tokens (`<tool_call>`/`</tool_call>`) and
puts a small XML-like body inside; the bracket tokens are what makes decoding
possible (§8). Gemma 4's template uses an ordinary-token grammar,
`<|tool_call>call:NAME{…}<tool_call|>`, and its result handoff resumes the model
turn and can reopen the thought channel — neither is the same shape as Qwen's
`<tool_response>` fold. ([The evidence](tool-calling.md) records the
two formats.)

The lesson is where the difference is allowed to live. The loop sees only
`profiles.ToolCall {id, name, arguments}`, `Role.tool`, and a shared
`ToolDefinition`; it never reads a marker or a body. What varies per checkpoint
— the bracket texts, the body grammar, the result framing, whether the
reasoning channel can reopen — is a `StreamMarkers`/`parse` pair and a render
function in the profile module. Gemma contributes no pair, so it returns
`error.ToolsUnsupported` rather than guessing at a format its model was never
trained on; a second format is a second fixture set, not a branch in the loop.
This is the same trade as [§43 of the inference companion](../llm-guide.md#43-a-registry-built-from-a-table-the-seam-before-the-second-model): the variation is named once, at the
seam, and the code above the seam stays single.

## 10. Resuming a session is a replay, not a restore

The intuition that saving a conversation means saving the model's state is the
one to resist. A `Session` is a page-aligned block of attention rows and, for
this hybrid, recurrent vectors; it cannot survive a process exit, and it cannot
be rewound by truncating a cache ([§39 of the inference companion](../llm-guide.md#39-checkpoints-not-rewinds-why-a-hybrid-model-copies-its-state)). So `~/.nuclis/agent/sessions/*.jsonl`
stores the other thing: the **conversation** — what the user typed, what the
model reasoned and answered, which call it made, what the tool returned. That
is exactly the input the render function needs, and nothing else.

`/resume` and `--resume <id>` therefore **replay**: the loader parses the
entries back into `profiles.Message`s (user, assistant-with-`tool_calls`, tool
result with its `tool_call_id`), `Agent.restore` appends them to a fresh loop
and continues the host correlation ids past the highest restored one, and the
next turn renders the whole thing through the profile and prefills it. The
transcript is replayed from the same entries, so what the terminal shows after
a resume is what it showed before. The fresh session file is seeded with the
entries so it can itself be resumed.

Two details make it work. The host ids stored on assistant calls are what let a
result match its call on load — the model never echoes them, so without them the
conversation would not validate. And a **cancelled step is not a message** the
model ever finished, so it is recorded for display but neither appended to
history nor given stored calls; a file whose assistant call has no following
result would not load. The cost of the replay is that a long conversation is
prefilled again on resume, and a session that had grown past the context window
must be compacted again — which is the honest trade: the alternative is
persisting opaque numerical state that the format cannot describe and a future
change cannot safely reinterpret.

This is also why the resumable unit is the *entry*, not the event. A turn emits
hundreds of `answer_delta` events and writes one `assistant` entry; the file
records the turn, so a resume re-renders rather than replays keystrokes.
