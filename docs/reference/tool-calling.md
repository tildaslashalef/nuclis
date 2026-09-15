# Tool calling: model evidence and the engine seam

Research checked 2026-09-13 for AGNT-01. These are format findings and design
constraints, not a claim that native tool calling is implemented.

## Model cards and artifact identity

The requested [Qwen3.8-27B GGUF card](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
advertises tool-calling improvements and configurable thinking. The requested
[Gemma 4 26B-A4B QAT GGUF card](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF/raw/main/README.md)
advertises native function calling; it is a mixture-of-experts checkpoint,
with configurable thinking and no previous-turn thoughts in ordinary history.
Neither card alone defines the complete wire format.

The engine's pinned Gemma artifacts are **12B**, not the requested 26B-A4B.
The latter's numerical implementation remains the 26B-A4B mixture-of-experts unit in the roadmap. The existing
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
channel markers 100/101. These IDs are **12B evidence**; they have not been
verified on the requested 26B QAT artifact. Tool fixtures are still required.

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
a truncated or malformed call as text rather than executing it. Gemma still
requires its own grammar and a profile-owned handoff condition distinct from
ordinary end-of-turn stopping; stopping at the first call would lose additional
calls, so its tools stay explicitly unsupported until its own fixtures pass.

The application can validate call IDs, pending result order, schemas, and
execution limits without knowing either wire format. The profile alone decides
whether results become user turns or remain inside a model turn. Test multiple
calls, nested values, literal marker-like text, truncated controls, cancellation,
and reasoning across tool steps when implementing the native parsers.
