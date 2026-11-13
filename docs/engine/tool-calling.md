<!-- kind: reference -->
# Tool calling: model evidence and the engine seam

Each family's native tool format, the evidence it was read from (Qwen on
2026-09-13, Gemma 4 on 2026-09-16, Muse Glimmer on 2026-09-19), and the
engine seam it constrains: the completion events. Every profile implements
its native path, the E-series through Gemma 4's
([prompt-profile.md](prompt-profile.md)).

## Model cards and artifact identity

The requested [Qwen3.8-27B GGUF card](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
advertises tool-calling improvements and configurable thinking. The requested
[Gemma 4 26B-A4B QAT GGUF card](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF/raw/main/README.md)
advertises native function calling; it is a mixture-of-experts checkpoint,
with configurable thinking and no previous-turn thoughts in ordinary history.
Neither card alone defines the complete wire format.

The Gemma evidence below was read on the 12B artifacts pinned when it was
gathered; the 26B-A4B has run since 2026-09-18 and carries the same
template digest. The
[artifact record](../benchmarks/record/catalogue.md#pinned-commits-and-digests-2026-09-11)
and [profile fixtures](prompt-profile.md#evidence-and-reproduction) remain
authoritative for implemented behavior. A live upstream template is evidence
for interface design, not permission to replace a pinned profile.

## Qwen: XML-like calls, JSON declarations

The [upstream template](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/chat_template.jinja)
places JSON tool declarations inside a system tools block. Calls use an outer
`<tool_call>` block, an inner `<function=NAME>` block, and one
`<parameter=NAME>` block per argument. String values are literal content;
non-string values are JSON. The value's delimiters are exactly one newline
after `>` and one before `</parameter>` (the reference's grammar,
`common/chat.cpp`, read 2026-09-22); the decoder removes those two and
nothing else, so a file's content keeps its trailing newline and its first
line's indentation, while the JSON typing looks at the whitespace-trimmed
text (` 20 ` is the number 20). This is not a Hermes JSON call payload.
Consecutive logical tool messages become response blocks in one user turn.
Assistant history preserves structured calls separately from answer and
reasoning; calls do not serialize a call ID. Thinking can remain across
steps of a tool loop.

Direct metadata inspection of the pinned local Qwen GGUF confirmed template
SHA-256 `12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce`
and the same call structure. Its template explicitly rejects nonempty JSON
strings as history arguments: callers must deserialize them to objects.
Vocabulary inspection found `<tool_call>` / `</tool_call>` at 248058 / 248059
and `<think>` / `</think>` at 248068 / 248069. The inner function/parameter
delimiters are not standalone vocabulary entries. Exact outer token boundaries
therefore help, but do not eliminate the need for a bounded incremental parser
inside a call. These facts were read with `scripts/fixtures/gguf-inventory.py` and a
standard-library reader over its recorded vocabulary offset, without weights.

A tool result is framed content ([prompt-profile.md § Content is
data](prompt-profile.md#content-is-data)): a file that spells
`</tool_response>` or `<|im_end|>` reaches the model as text, not as the
end of its response block or its turn. A tool or parameter name is spelled
bare inside `<function=…>` and `<parameter=…>`, so one carrying a marker, a
`>`, or a newline is refused.

## Gemma: calls and results inside a model turn

The [26B-A4B upstream template](https://huggingface.co/google/gemma-4-26B-A4B-it/raw/main/chat_template.jinja)
uses declarations bracketed by `<|tool>` and `<tool|>`. A call has the form
`<|tool_call>call:NAME{key:VALUE}<tool_call|>`, with `<|"|>` string delimiters.
Arguments support nested objects and arrays, rather than being JSON text on
the wire. Results use `<|tool_response>response:NAME{…}<tool_response|>`.
Logical tool messages are folded into the preceding model turn; the template
can resolve their IDs against preceding calls. It preserves current-loop
reasoning after the last user, and may resume a thought channel after results.
This differs from dropping thoughts before a new user turn.

Google's [function-calling guide](https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4)
shows generation handing control back at `<|tool_response>`, after a complete
call. Tool execution belongs to the host, which returns results and resumes
generation. Its illustrative regex is not a parser contract for nested values.

The cached template matching the pinned 12B digest
`845f1ee48e39fc942fe190da9df6a1c5db229e17a96ea08966ad1c9274e73d1b`
confirms these structural rules. Direct inspection of the pinned 12B vocabulary
found call markers 48/49, response markers 50/51, string delimiter 52, and
channel markers 100/101 (all user-defined tokens, rendered as text by the
decoder, so the body between the call brackets carries its own string
delimiters). The ids are not assumed: the decoder resolves each marker by
its text in the loaded file's vocabulary (`MissingToolToken` when one is
absent), and `gemma4-e4b-vocabulary` asserts them on the E4B. The tool
fixtures, `gemma4-tools.json`, were captured on the 12B.

Three facts from the pinned reference (`7620399`) shaped the Gemma
implementation. Its vocabulary loader marks `<|tool_response>` end-of-generation
by name, matching Google's "additional stop sequence": the model emits every
call of a step and then that token, so the profile lists it as a stop token
and the shared decoder needs no handoff state. Its chat layer appends
`<|turn>model\n` whenever the template leaves the prompt at a closed turn,
which happens when results are followed by content at the end of the
conversation; `/apply-template` renders through that layer, so the fixtures
include the repair and the profile reproduces it. And its server enables
reasoning preservation by default for templates that support it, so a
call-bearing assistant message keeps its `<|channel>thought` at any age while
a plain answer keeps it only after the last user message. Two deliberate
departures, both pinned by tests: reasoning is trimmed before rendering so the
model's own bytes are reproduced for the incremental prefill, and numbers are
rendered as their JSON text where the reference reformats floats
(`1e10` → `10000000000.0`).

The models leave the grammar in one way seen so far: the 26B-A4B wrote
`call:remember{fact: "…"}`, a JSON string in plain quotes, in 2 of 4
replays of one turn (Muon, 2026-10-28). The parser reads a JSON string,
as value or key (`{"fact":…}`), as JSON reads it. A body that still does
not parse is dropped with `tool_cut`, not released as answer text as the
other profiles do (`Tool.unparsed`): its brackets are control tokens, so
the text would show the reader the model's call syntax; the API answers
`malformed_tool_call`.

A result's content sits between two `<|"|>` delimiters and the DSL has no
escape. It is framed content, so a delimiter or a marker it spells is text
tokens and cannot end the string, the response, or the turn; it is
rendered, not refused, and a file that mentions one no longer ends the
agent's turn. An argument string or a description carrying a marker is
still refused, since those are the model's or the host's own strings.

## Muse Glimmer: ATEM calls as their own messages

The pinned template (digest `114f55eb…`, [muse-glimmer.md](../models/muse-glimmer.md#chat-template))
puts declarations into every system turn as prose plus JSON lines: after
the strength line, the template's instructions, `// Tool metadata` with one
`{"name": NS, "description": ""}` per tool namespace (a name's part before
the first `.`, the whole name without one), `// Function schemas` with one
`{"name": …, "description": …, "parameters": …}` per tool in the
reference's `tojson` style (a space after `:` and `,`, keys in insertion
order, quotes, backslashes, and control characters escaped, non-ASCII
literal), and a fixed example; the recipients line then lists `"self"`,
one `"NS.*"` per namespace, and `"user"`. A call is its own message,
`<|start|>assistant to=NAME<|message|>` holding one
`<atem:function_calls>` block with one `<atem:invoke name="NAME">` and one
`<atem:parameter name="KEY">VALUE</atem:parameter>` line per argument:
strings verbatim (spaces and newlines included, no escaping), booleans and
`null` as words, numbers as written, lists and objects as JSON. Several
calls of one step are `<|eom|>`-separated messages and the last ends the
turn with `<|eot|>`; the message's content is not rendered beside its
calls. A result is its own turn, `<|start|>tool NAME<|message|><tool_output name="NAME">\nCONTENT\n</tool_output><|eot|>`,
the name resolved from the call it answers. Reasoning stays its own
`to=self` message at any age. The reference parses the body with a
schema-aware grammar (string-typed parameters verbatim, others as JSON) and
treats `<|eot|>` as the end of the call step.

Three facts shaped the implementation (`muse_glimmer-tools.json`, 52 cases
captured 2026-09-19 through the reference's chat layer). The markup is
text: no ATEM tag is a token, so the decoder's only structural signals are
the channel tokens (`<|start|>` 200022, `<|message|>` 200023, `<|eom|>`
200007) and the header text between them; the profile's `parseTool` reads
the body of any `to=NAME` message and, like Qwen's, keeps a value that
parses as JSON and takes anything else as a literal string (the reference's
schema-aware typing needs the declaration, which the decoder does not
have). A call body is complete at the turn's end, not at a bracket: the
decoder completes an open body when the turn stopped on `<|eot|>` and
cuts it (a `tool_cut` event, never text) on a budget stop or a cancellation. And a body with
more than one invoke, or anything but parameters inside the invoke, is not
a call: it is released as text rather than guessed at. Content, results,
names, and argument strings carrying the ATEM markup or a control marker
are rejected (`error.UnsupportedContent`) because the reference parses with
regular expressions and the markup would be structure; the model's own
released text loses the markup instead.

## Consequences for implementation

The shared boundary: profiles receive token IDs plus decoded pieces and
emit thinking, answer, complete tool calls, and completion stop events. The
agent consumes semantic events and owns execution. `runLoop` remains the
raw primitive for measured generation paths.

A tool definition carries a name, a description, and a parameter schema;
history carries assistant call records and tool-result correlation, not
only a role tag. Arguments are normalized to a JSON object at the boundary,
and each profile serializes its native format. IDs are host correlation
data when the model format provides none. Unsupported tool inputs are
refused (`error.UnsupportedContent`), never rendered as ordinary text.

Qwen rendering and decoding are both pinned
(`fixtures/qwen38-tools.json`): the tools block, the native
`<tool_call>`/`<function>`/`<parameter>` history, and the folded
`<tool_response>` user turn match the artifact's template byte for byte, and
the shared streaming decoder reads the outer control tokens from the
artifact's vocabulary, parses the body with the profile's grammar, releases
a malformed call as text, and cuts a truncated one (`tool_cut`), executing
neither. Gemma's grammar
(`fixtures/gemma4-tools.json`, `gemma4.parseTool`) rides the same decoder;
its handoff needs no mechanism of its own, because the model's own
`<|tool_response>` token after the last call of a step is a stop token, so
several calls in one step all arrive before generation ends.

Muse's grammar (`fixtures/muse_glimmer-tools.json`, `muse_glimmer.parseTool`)
rides the decoder's second grammar, the message headers; its handoff is the
stop token `<|eot|>` after the step's last call, the same shape as Gemma's.

Under both bracket grammars a second opening token before a close settles
the first call as cut (`tool_cut`) and starts the next one; no assert is
reachable from model output.

The application can validate call IDs, pending result order, schemas, and
execution limits without knowing any wire format. The profile alone decides
whether results become user turns or remain inside a model turn. The
parsers' and the stream decoder's tests cover multiple calls, nested values,
literal marker-like text, truncated controls, cancellation, and reasoning
across tool steps.

## Completion events

`engine.complete` wraps `runLoop` with profile decoding. Its caller supplies
sampling scratch (`CompletionBuffers`), the effort used to render the prompt,
an optional observer, and a synchronous `sink.send(events.Event)`. Thinking
and answer strings borrow per-piece scratch until `send` returns. The sink
must copy anything it retains. UTF-8 fragments are joined within a channel;
an incomplete scalar at a channel boundary or completion end becomes U+FFFD.
A successful completion emits one `stop` carrying the loop's outcome and
metrics, including cancellation during prefill or decode. Inference and sink
errors propagate, without a fabricated stop. `runLoop` and its raw token hooks
remain available to `generate` and `bench`.

The union's `tool_call {id: u32, name, arguments}` is the same type as the
history `ToolCall`; the ID is host correlation data (the profile sets 0 and
the agent assigns it) and arguments are a normalized JSON object. The Qwen
decoder produces these: the shared `profiles/stream.zig` recognizes the
`<tool_call>` / `</tool_call>` control tokens by vocabulary ID, collects the
body as ordinary pieces, and hands it to `qwen38.parseTool`, which emits one
call when the closing token arrives, with `tool_progress` events (the body's
size) while it grows. A body the parser refuses after its closing token is
released as answer text and never executed, bracket text included (copied
when the bracket arrives: a piece is the caller's per-token buffer). A call
still open when the completion stops is one call at EOS if it parses;
otherwise, and on a budget stop or a cancellation, it is dropped with a
`tool_cut` event: neither an action nor answer text. A close token that
arrives while no reasoning is open (the model ending a thought the engine
already cut) is dropped. A Qwen parameter value ends at the first
`</parameter>` followed by the next parameter, `</function>`, or the body's
end, so content that mentions the marker survives. A reasoning channel the model
opens while answering is thinking from there, shown as a further block,
whatever the effort (with thinking off, Gemma still opens an empty channel
after a tool result); a closing token with no channel open stays literal. The Gemma decoder is the same machinery with its own
brackets (`<|tool_call>` / `<tool_call|>`) and `gemma4.parseTool`; its
handoff is the `<|tool_response>` stop token. The model-card and pinned-template evidence is in the sections above.
