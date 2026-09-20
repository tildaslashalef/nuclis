# Tool calling: model evidence and the engine seam

Research checked 2026-09-13 for AGNT-01, extended 2026-09-16 for AGNT-09
and 2026-09-19 for AGNT-10. These are format findings and design
constraints; all three profiles implement their native path
([prompt-profile.md](prompt-profile.md)).

## Model cards and artifact identity

The requested [Qwen3.8-27B GGUF card](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
advertises tool-calling improvements and configurable thinking. The requested
[Gemma 4 26B-A4B QAT GGUF card](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF/raw/main/README.md)
advertises native function calling; it is a mixture-of-experts checkpoint,
with configurable thinking and no previous-turn thoughts in ordinary history.
Neither card alone defines the complete wire format.

The engine's pinned Gemma artifacts are **12B**, not the requested 26B-A4B.
The latter's numerical implementation landed as MODL-10 (2026-09-18). The existing
[artifact record](artifacts.md#pinned-commits-and-digests-modl-02-2026-09-11)
and [profile fixtures](prompt-profile.md#evidence-and-reproduction) remain
authoritative for implemented behavior. A live upstream template is evidence
for interface design, not permission to replace a pinned profile.

## Qwen: XML-like calls, JSON declarations

The [upstream template](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/chat_template.jinja)
places JSON tool declarations inside a system tools block. Calls use an outer
`<tool_call>` block, an inner `<function=NAME>` block, and one
`<parameter=NAME>` block per argument. String values are literal content;
non-string values are JSON. This is not a Hermes JSON call payload.
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
inside a call. These facts were read with `scripts/gguf-inventory.py` and a
standard-library reader over its recorded vocabulary offset, without weights.

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
delimiters). These IDs are **12B evidence**; they have not been verified on
the requested 26B QAT artifact. The 12B tool fixtures are pinned
(`gemma4-tools.json`).

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

## Muse Glimmer: ATEM calls as their own messages

The pinned template (digest `114f55eb…`, [muse-glimmer.md](muse-glimmer.md#chat-template))
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
releases it as text on a budget stop or a cancellation. And a body with
more than one invoke, or anything but parameters inside the invoke, is not
a call: it is released as text rather than guessed at. Content, results,
names, and argument strings carrying the ATEM markup or a control marker
are rejected (`error.UnsupportedContent`) because the reference parses with
regular expressions and the markup would be structure; the model's own
released text loses the markup instead.

## Consequences for implementation

AGNT-01 establishes the shared boundary: profiles receive token IDs plus decoded
pieces and emit thinking, answer, complete tool calls, and completion stop
events. The agent consumes semantic events and owns execution. `runLoop`
remains the raw primitive for measured generation paths.

The tools parameter should carry names, descriptions, and parameter schemas;
history needs assistant call records and tool-result correlation, not only a
new role tag. Normalize arguments to a JSON object at the boundary, while
letting each profile serialize its native format. IDs are host correlation
data when the model format provides none. Reject unsupported tool inputs
explicitly in AGNT-01; neither existing profile should silently render them as
ordinary text.

Qwen rendering and decoding are both pinned
(`fixtures/qwen38-tools.json`): the tools block, the native
`<tool_call>`/`<function>`/`<parameter>` history, and the folded
`<tool_response>` user turn match the artifact's template byte for byte, and
the shared streaming decoder reads the outer control tokens from the
artifact's vocabulary, parses the body with the profile's grammar, and releases
a truncated or malformed call as text rather than executing it. Gemma's grammar
(`fixtures/gemma4-tools.json`, `gemma4.parseTool`) rides the same decoder;
its handoff turned out to need no new mechanism, because the model's own
`<|tool_response>` token after the last call of a step is a stop token, so
several calls in one step all arrive before generation ends.

Muse's grammar (`fixtures/muse_glimmer-tools.json`, `muse_glimmer.parseTool`)
rides the decoder's second grammar, the message headers; its handoff is the
stop token `<|eot|>` after the step's last call, the same shape as Gemma's.

The application can validate call IDs, pending result order, schemas, and
execution limits without knowing any wire format. The profile alone decides
whether results become user turns or remain inside a model turn. Test multiple
calls, nested values, literal marker-like text, truncated controls, cancellation,
and reasoning across tool steps when implementing the native parsers.
