# TODO — active plan

This file is the only progress tracker. It holds **unfinished work only**:
where we are, what is next, and the design of each remaining unit. When a
unit closes, its outcome moves to
[docs/engineering-log.md](docs/engineering-log.md) and
its section is deleted here; when the last unit closes, this file is emptied
back to this header. Requirements live in [docs/spec.md](docs/spec.md); the
engine map in [docs/architecture.md](docs/architecture.md); how to build,
test, and measure in [docs/development.md](docs/development.md).

Session protocol (also in [AGENTS.md](AGENTS.md)): read this file first. If
it lists work, summarize *Where we are* and ask the user how to continue. If
it is empty, ask what to work on and write the agreed plan here.

## Where we are

APPS-06 closed on 2026-09-16. One unit is open: Gemma 4 has no tool
calling, so `nuclis agent` on `gemma-4-12b` skips its warm-up with
`ToolsUnsupported` and runs without tools. The reference oracle for the
fixtures is the pinned llama.cpp build under `.zig-cache/reference/`
(rebuild recipe in [docs/reference/reference-baseline.md](docs/reference/reference-baseline.md)).

Order: AGNT-09.

| Unit | Title | Sessions |
| --- | --- | --- |
| AGNT-09 | Gemma 4 native tool calling: rendering, decoding, handoff, fixtures | 1 |

## AGNT-09 — Gemma 4 native tool calling: rendering, decoding, handoff, fixtures

**Why.** Feature parity with Qwen: the agent loop is profile-driven, so the
only missing piece is the Gemma profile's tool path. Today `gemma4.zig`
returns `error.ToolsUnsupported` for any tool input and the agent runs
text-only on Gemma.

**Evidence read (2026-09-16).** The pinned 12B template (digest `845f1ee4…`,
extracted from the local GGUF, not in the tree), Google's
[function-calling](https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4)
and [prompt-formatting](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4)
guides, and the pinned llama.cpp `7620399` (`common/chat.cpp`,
`src/llama-vocab.cpp`, `tools/server/*`). Facts the design rests on:
- Control tokens in the pinned vocabulary: `<|tool>` 46 / `<tool|>` 47
  (control), `<|tool_call>` 48 / `<tool_call|>` 49, `<|tool_response>` 50 /
  `<tool_response|>` 51, `<|"|>` 52 (all user-defined, rendered as text by
  the decoder). Same template digest on the QAT file.
- **Handoff.** The model ends a call step by emitting `<|tool_response>`:
  Google calls it "an additional stop sequence", and the reference marks it
  end-of-generation by name. So the Gemma stop set becomes `<turn|>`,
  `<eos>`, `<|tool_response>` (the engine already stops on any profile stop
  token, `max_stop_tokens` 4). Several calls in one step are emitted
  back to back before the handoff token, so stopping there loses none.
- **Wire grammar** (reference PEG, `common_chat_params_init_gemma4`):
  `<|tool_call>call:NAME{key:value,…}<tool_call|>`; values are
  `<|"|>…<|"|>` strings (no escaping; the string ends at the next
  delimiter), JSON numbers, `true`/`false`/`null`, `{key:value,…}` with
  bare keys `[^:}]+`, and `[value,…]`; optional whitespace after `{`, `[`,
  `,`, and `:`.
- **Template rendering.** System turn: `<|think|>\n`? then the first
  system message, then one `<|tool>declaration:NAME{description:<|"|>…<|"|>,parameters:{properties:{…},required:[<|"|>a<|"|>],type:<|"|>OBJECT<|"|>}}<tool|>`
  per tool, no separators, then `<turn|>\n`. Property keys are sorted
  (`dictsort`), types uppercased, each property `KEY:{description?,enum?|items?,nullable?,properties?/required?,type}`.
  An assistant message with calls renders, inside one `model` turn and in
  this order: reasoning as `<|channel>thought\n…\n<channel|>` only when the
  message follows the last user message (the template's gate; the
  current tool loop keeps its thoughts, older turns drop them), the calls
  (`dictsort`ed keys), the answering results as
  `<|tool_response>response:NAME{value:<|"|>CONTENT<|"|>}<tool_response|>`,
  then the content. `<turn|>\n` follows unless the turn continues into the
  next assistant message or the conversation ends on results without
  content. Generation prompt: `<|turn>model\n` (plus the pre-closed
  channel when thinking is off), except after results, where the turn is
  still open and only `<|channel>thought\n` is added when thinking is on.
  The reference then appends `<|turn>model\n` whenever the prompt ends on
  `<turn|>\n` (content plus results at the end); `/apply-template` goes
  through that path, so the fixtures include it and the profile reproduces
  it.

**Design.**
- **Render** (`inference/src/profiles/gemma4.zig`). The tools block in
  `prefix` and `render`; a bounded JSON-Schema subset for declarations
  (object → properties/required; each property needs a string `type`;
  optional `description`, `enum` on strings, `items` on arrays, nested
  object `properties`/`required`, `nullable`); anything else is
  `error.UnsupportedContent`, never approximated. Calls and results in the
  DSL from the normalized JSON arguments. A string that contains `<|"|>`,
  or content/arguments carrying any of the six tool markers, is
  `error.UnsupportedContent` (structure smuggling, as Qwen's marker check).
  Reasoning is trimmed before rendering (the template does not trim; the
  model writes one newline before `<channel|>`, and trimming plus the
  template's `\n<channel|>` reproduces the model's own bytes so the
  incremental prefill hits) — documented as the one deliberate deviation,
  pinned by a test that padded and unpadded reasoning render alike.
- **Decode** (`stream_markers`): `tool_open`/`tool_close` are
  `<|tool_call>`/`<tool_call|>`; `parseTool` is a bounded recursive-descent
  parser of the DSL (nesting depth 16, body size already bounded by the
  decoder) into a JSON object, null for anything else. `stop_tokens` gains
  `<|tool_response>`.
- **Fixtures.** `scripts/profile-tools-fixtures.py --profile gemma4` writes
  `inference/src/profiles/fixtures/gemma4-tools.json` (same shape as the
  Qwen file; the reference strips `<bos>`, the test prepends it). Cases:
  the five Qwen cases, plus a richer declaration (description, integer,
  boolean, enum, array of strings, nested object), a call with nested
  object/array/number/boolean/null arguments, a loop that ends on results
  (no trailing user) with and without content and reasoning, and two
  assistant steps in one loop; efforts `off` and `medium`. Captured on the
  K-quant 12B (template digest and full checksum verified).
- **Artifact check.** `inference/vocabulary-check.zig` gains the seven tool
  tokens for Gemma.
- **Agent.** No loop change expected: the warm-up prefills the tools block,
  the decoder emits calls, `<|tool_response>` ends the step, and the next
  render continues the model turn with the results. Verify the increment
  path hits across a tool step (`replayed` false in `--print --json`).
- **Docs.** `prompt-profile.md` (Gemma section, evidence, capture recipe),
  `tool-calling.md` (implemented, the deviation, the reference workaround),
  `architecture.md` § profiles, `agent-spec.md` and `agent-concepts.md`
  clauses that record `ToolsUnsupported`, `spec.md` line on Gemma,
  `THIRD_PARTY_NOTICES.md` (the reimplementation is no longer text-only),
  the engineering log.

**Acceptance.** Default tests pin every Gemma tool prompt byte for byte;
`parseTool` round-trips rendered calls (strings with quotes and newlines,
nested values) and refuses truncated or malformed bodies; the stream test
drives the Gemma grammar with fake ids; `make check` passes; `make
test-vocabulary MODEL=<gemma>` passes. Live: `nuclis agent` on
`gemma-4-12b` warms up without a skip notice, and a `--print --json` turn
executes a `write_file` then a `read_file` call with the file on disk as
asked; record whether the second step replayed.
