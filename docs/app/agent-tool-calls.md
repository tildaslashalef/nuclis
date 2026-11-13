<!-- kind: reference -->
# Tool calls at the token boundary

How model output becomes typed events and a tool call becomes host data:
the profile's side of the agent's tool seam. These are §2, §3, §8, and §9 of
the concepts behind `nuclis agent`; the loop that runs the calls is in
[agent.md](agent.md), the terminal they reach in [terminal.md](terminal.md).

## 2. Why a streaming parser needs token boundaries

The final interpretation of model output belongs to the inference profile. A
sampled token arrives as an integer ID; decoding it to
bytes discards useful information. A real `</think>` token and several
ordinary tokens spelling `</think>` can produce identical text, but only the
former closes the reasoning channel. The application-level splitter ([§1](terminal.md#1-a-terminal-that-is-written-once-events-blocks-and-the-line-between-them))
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
differently ([the evidence](../engine/tool-calling.md)); each profile
decodes its own (§8, §9).
`generate` and `bench` retain the raw `runLoop` path.

## 3. A tool call is host data until a profile serializes it

The other half of the tool seam is the conversation types a profile
encodes. The interesting decision is what is *shared* and what is
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
render it. Each profile asks that second question, and renders a tool
grammar only once its fixtures pin it: the loop is built and tested against
the shared contract, never against a wire format a profile would have to
guess — the same rule that keeps a mismatched template from being rendered
at all.

## 8. Decoding a tool call at the token boundary

A model does not emit JSON. Qwen3.8 emits a call as ordinary tokens wrapped in
two *control* tokens: `<tool_call>` and `</tool_call>`. The wrapper is the only
part the vocabulary marks as special; everything inside — `<function=…>`, each
`<parameter=…>`, the value — is spelled with ordinary tokens that could appear
in prose. That asymmetry is the whole design: the profile can detect a call's
boundaries only while it still has token IDs, and it must then treat the body
as opaque text and parse it separately.

`engine.complete` already hands each generated token to the profile decoder as
`(id, piece)`. The shared decoder (`profiles/stream.zig`) also holds an
optional tool grammar: the two bracket IDs and a `parse(alloc, body)` function.
On the opening ID it flushes whatever answer text was pending, switches to
collecting the body, and stops emitting text; on the closing ID it parses the
collected body with the profile and emits one `tool_call` event. The body
parser is the profile's (`qwen38.parseTool`), so the wire syntax never leaves
`profiles/` — the agent loop sees typed `ToolCall`s and nothing else.

Two failure modes are decided by the same state machine. A call that is still
open when the stream ends never reached its closing token: at EOS a body that
parses is still one call, and anything else — a body that does not parse, a
budget stop, a cancellation — is dropped with a notice (`tool_cut`), never
shown as answer text. A body that arrives complete but that the parser refuses
(no function name, malformed parameters) is released as text under Qwen, the
same bytes the model wrote, and dropped with a notice under Gemma 4
(`Tool.unparsed`). Neither can reach execution: an incomplete or malformed call is a
message, never an action. This is the tool analogue of [§6](agent.md#6-running-a-command-is-a-lifetime-problem)'s rule that a
process is not an instruction until the host has actually built it.

The arguments themselves obey the template's own literal-versus-JSON rule. The
prompt profile renders a string argument literally and anything else as JSON;
the decoder reverses it by trying to parse each value as JSON and falling back
to the literal text. The host still validates the fields against the tool it
names — the profile normalizes the envelope, it does not vouch for the
contents. And because the outer markers are resolved from the artifact's
vocabulary, a template whose call brackets are not standalone tokens declares
no tool grammar and keeps rejecting tool input rather than approximating a
format its model never sees.

## 9. Two tool grammars, one loop

A decoder that recognises *a* tool call does not by that recognise
*any* model's tool call, and Qwen3.8 and Gemma 4 make that concrete.
Qwen3.8 wraps a call in two control tokens (`<tool_call>`/`</tool_call>`) and
puts a small XML-like body inside; the bracket tokens are what makes decoding
possible (§8). Gemma 4's brackets are control tokens too, around an
ordinary-token body, `<|tool_call>call:NAME{…}<tool_call|>`, and its result handoff resumes the model
turn and can reopen the thought channel — neither is the same shape as Qwen's
`<tool_response>` fold. ([The evidence](../engine/tool-calling.md) records the
two formats.)

The lesson is where the difference is allowed to live. The loop sees only
`profiles.ToolCall {id, name, arguments}`, `Role.tool`, and a shared
`ToolDefinition`; it never reads a marker or a body. What varies per checkpoint
— the bracket texts, the body grammar, the result framing, whether the
reasoning channel can reopen — is a `StreamMarkers`/`parse` pair and a render
function in the profile module. Gemma's pair arrived with its own fixture set
and no change to the loop: a second format is a second fixture set, not a
branch in the loop. Its handoff is even a different mechanism — the model
emits `<|tool_response>` after its calls, so that token is simply one more
stop token in the profile's set.
This is the same trade as [§24 of the inference companion](../llm-guide.md#24-the-registry-table): the variation is named once, at the
seam, and the code above the seam stays single.
