<!-- kind: reference -->
# Prompt profiles and tokenizer fixtures

A prompt profile is the checkpoint's conversation contract, kept apart from
its numerical architecture: the chat template rendered as a bounded Zig
function, the tokens that end a turn, the markers that delimit the model's
reasoning in generated text, and the checkpoint's official sampling
defaults. The registry is `inference/src/profiles/root.zig`: one tag per
profile module, selected at load by the SHA-256 of the artifact's
`tokenizer.chat_template` (`profiles.forDocument`), never by architecture,
so a conversation is never interpreted with a template the profile's
fixtures did not verify. The profiles and their template digests:

<!-- gen:profiles -->
| Profile | Template SHA-256 |
| --- | --- |
| `qwen38` | `12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce` |
| `gemma4` | `845f1ee48e39fc942fe190da9df6a1c5db229e17a96ea08966ad1c9274e73d1b` |
| `gemma4_e` | `241c50d86bdfe5e43307da87f559cd2416aacd67a8de46c15acc0105ef2200b7` |
| `muse_glimmer` | `114f55ebdc1804c1af371197b9fdf2d6bb925966c9dfe46b73782a71bc07965e` |
<!-- /gen -->

| Profile | Module | Stop tokens | Reasoning markers |
| --- | --- | --- | --- |
| `qwen38` | [qwen38.zig](../../inference/src/profiles/qwen38.zig) | `<|im_end|>`, `<|endoftext|>` | `<think>` … `</think>` |
| `gemma4` | [gemma4.zig](../../inference/src/profiles/gemma4.zig) | `<turn|>`, `<eos>`, `<|tool_response>` | `<|channel>thought\n` … `<channel|>` |
| `gemma4_e` | [gemma4_e.zig](../../inference/src/profiles/gemma4_e.zig) | as `gemma4` | as `gemma4` |
| `muse_glimmer` | [muse_glimmer.zig](../../inference/src/profiles/muse_glimmer.zig) | `<|eot|>`, `<|end_of_text|>` | `<|start|>assistant to=self<|message|>` … `<|eom|>` |

The catalogue's `qwen3.8-27b` uses `qwen38`; `gemma-4-26b-a4b` and
`gemma-4-12b-qat` use `gemma4`; `gemma-4-e4b-qat` uses `gemma4_e`;
`muse-glimmer-30b` uses `muse_glimmer`. The same directory holds the
decision models' input contracts (`laya.zig`, `clef.zig`, `decision.zig`),
which are not chat templates and not in the registry.

Every module exposes the same surface: `template_sha256`,
`template_aliases`, `render`, `prefix`, `samplingDefaults`, `efforts`,
`stop_tokens`, `bos_token`, `reasoning`, `stream_markers`,
`image_placeholder`, `generation_prompts`, and `rewritesTurn`, with
`audio_placeholder` where the family hears. The last two serve the session
cache ([agent.md § The agent's token cache](../app/agent.md#12-the-agents-token-cache)):
`Profile.promptEnd` cuts the generation prompt off a render, leaving the
bytes the next turn's render starts with, and `rewritesTurn(effort)` says
whether the history render of a generated answer differs from what its
generation consumed (Gemma 4: thinking off, the empty thought channel is
absent from past turns; thinking on, the reasoning gate drops it; the
E-series only the latter). A test in `profiles/root.zig` pins both per
profile. The engine resolves the stop
tokens to ids in the artifact's vocabulary at load (`MissingStopToken` when
one is absent: a template/tokenizer mismatch is refused, never guessed) and
ends generation on any of them; without a matching profile it stops on the
file's `eos_token_id` alone and refuses chat rendering. `engine.complete`
uses `Profile.decoder` to resolve reasoning controls in the vocabulary and
stream semantic events. Actual control IDs delimit channels; ordinary tokens
spelling the same text do not. Gemma's `thought\n` suffix is ordinary text
following the opening control, so the decoder handles its partial arrival.
A completion starts in reasoning only where the prompt's last reasoning
control is the opening one (`stream.opensInReasoning`): Qwen's thinking
prompt ends inside `<think>`, while Gemma's ends at `<|turn>model\n` and
the model may answer without opening a channel, which is then answer text.
The agent stores the channels directly for history and session logging;
`generate`/`agent` sample with the profile's defaults for the
`--think` effort ([sampling.md](sampling.md#sampling-profiles-and-the-selection-chain)).
Nothing under `src/` names a profile module; the executable uses the tag.

## Shared contract

`profiles.Profile.render(allocator, messages, tools, effort, limits)`
returns an owned UTF-8 prompt that the caller frees with the same allocator.
Messages borrow `role`, `content`, and optional `reasoning_content`. Roles
are system, developer, user, assistant, and tool. A completion-ready
conversation ends with a user message or a fully answered tool-result group;
system/developer messages are allowed only at the beginning. All strings
must be valid UTF-8; reasoning on a non-assistant message is
`UnsupportedContent`.

A user message may carry images and audio clips (`Message.images`,
`Message.audio`). Every profile renders each image before the text as the
`width_tokens × height_tokens` copies of its `image_placeholder` between
the family's markers (Qwen3.8 `<|vision_start|>` … `<|vision_end|>`,
Gemma 4 `<|image>` … `<image|>`, Muse Glimmer `<|image_start|>` …
`<|image_end|>`); the Gemma profiles render each clip after the images as
one `<|audio|>` per encoder row between `<|audio>` and `<audio|>`. The
projector's or encoder's rows replace the placeholders at prefill
([vision.md](vision.md), [audio.md](audio.md)).

### Content is data

A control marker spelled as text in system, user, or tool content stays
text. `render` with `Limits.literal_marks` set (every render the engine
encodes: `Engine.render`, `Engine.prefix`, `nuclis tokenize`) frames that
content between the tokenizer's literal marks, `0xFE` and `0xFD`, two bytes
UTF-8 never uses; the encoder drops them and matches no special token of
any kind inside or across a framed span
([tokenizer.md § Literal spans](tokenizer.md#literal-spans)). A file the
agent read that spells `<|im_end|>` or `</tool_response>`, a user who types
`<|turn>`, or a client that sends `<|image_pad|>` gets those characters as
text tokens, and the turn structure the model sees is the template's
alone. Nothing is refused for it, so a tool result never ends the agent's
turn for what a file says. Framing does not split the merges: a framed
span tokenises exactly as the same bytes unframed, so text without a
marker encodes as before. The fixtures compare unframed renders.

Tool and parameter names are spelled bare inside the template's syntax, so
a name carrying a marker (or the syntax's own delimiters) is refused with
`UnsupportedContent`. The assistant's own content and reasoning are not
framed: that text is the model's past output (a call released as text
when its closing token never came, a channel opened as text), and its
markers are **removed**, since re-encoding them would make control tokens
and refusing them would end every later turn of the session. The
template's own `strip_thinking` is the same idea. Muse Glimmer also
refuses its markers and the ATEM markup in content (§ Muse Glimmer): the
markup is text the model reads by pattern, which framing cannot defuse.

Tool inputs are shared types, not profile syntax. `ToolDefinition` carries a
name, a description, and a parameter schema as a JSON object;
`Message.tool_calls` carries the assistant's calls in execution order, and a
`role = .tool` message answers one of them by `tool_call_id`. The host assigns
the `u32` ID (`events.ToolCall` and `profiles.ToolCall` are one type): a
native wire format need not serialize one. `profiles.validate` checks all of
this before any rendering, so the rules cannot drift between profiles: call
arguments and parameter schemas must be JSON objects; a group's results
answer the pending calls in order, matched by ID; orphan, duplicate,
out-of-order, and missing results are rejected; nothing but results may
interleave a pending group. Defaults bound messages to 1,024, tool
definitions to 64, total borrowed bytes (content, reasoning, tool names,
descriptions, schemas, and arguments) to 1 MiB, and rendered output to 2 MiB
(UTF-8 bytes, not tokens). The renderer retains no borrowed data and releases
partial output on errors, including allocation failures
(`checkAllAllocationFailures`). `output_bytes` bounds the logical output; the
buffer may reserve more.

The effort argument is the shared `Effort` (`off`, `low`, `medium`,
`high`, `xhigh`), named after Qwen3.8's levels because it came first (its
template folds `high` into `xhigh`); a profile with fewer levels collapses
them, and one without `off` renders it as its lowest level. Every profile
renders and decodes its native tool path against pinned fixtures. No
profile implements assistant prefill or a general Jinja interpreter.

## Qwen3.8 (`qwen38`)

Leading system/developer messages are trimmed and merged with newlines;
empty leading system content is omitted. `low` and `xhigh` prepend the
artifact's reasoning instructions to the system block; `medium` adds none,
and `high` renders the `xhigh` line, as the template resolves it.
All modes append the assistant generation prefix `<|im_start|>assistant\n<think>\n`;
`off` closes it at once with an empty thinking block. Previous assistant
reasoning is preserved in history as `<think>…</think>` before the answer,
in every mode. Assistant answers lose a `<think>` or `</think>` they spell:
callers supply reasoning separately. The renderer trims ASCII
whitespace as the template's `trim` did under the Jinja implementation the
fixtures were captured with; nonbreaking spaces and other Unicode stay
intact.

**Tools.** With tool definitions the system turn becomes the artifact's tools
block: the reasoning instruction (when the mode has one), `# Tools`, one
OpenAI-shaped declaration per tool inside `<tools>…</tools>`, the template's
format reminder, then any merged system text. Declarations and non-string
argument values are serialized in the reference's `tojson` style (a space
after `:` and between items, keys in insertion order, non-ASCII literal).
Assistant calls render as `<tool_call>\n<function=NAME>…` with one
`<parameter=KEY>` block per argument (string values literal, others JSON);
consecutive tool results fold into one `<|im_start|>user` turn of
`<tool_response>` blocks, exactly as the template does. A tool or parameter
name carrying a marker, a `>`, or a newline is refused. Gemma's native tool grammar is a different shape and its result
handoff resumes the model turn; see its section below and
[tool-calling.md](tool-calling.md).

## Gemma 4 (`gemma4`)

Turns are `<|turn>role\n…<turn|>\n`; the assistant's role name is `model`.
The prompt starts with `<bos>` as text: the tree's encoder never adds BOS,
so the profile writes the marker the template emits (the captured
fixtures lack it, and the fixture test prepends it).
The first message, when system or developer, goes into the system turn;
later system and developer messages render as `system` turns of their own,
empty ones included (the reference sends `developer` as `system`). Thinking
is a switch, not a level: `off` renders no `<|think|>` and pre-closes an
empty thought channel in the generation prompt
(`<|turn>model\n<|channel>thought\n<channel|>`) so the model answers
directly; any other effort puts `<|think|>\n` at the top of the system
turn (which then exists even without a system message) and ends the prompt
at `<|turn>model\n`, after which the model opens its own channel or
answers without one. Assistant
reasoning renders as `<|channel>thought\n…\n<channel|>` only where the
template's gate passes: on a message after the last user message (the
current tool loop) or on any message that calls tools (the reference server
preserves reasoning by default for this template, and the fixtures were
captured with that default); everywhere else `reasoning_content` is
accepted and dropped. The profile trims the reasoning first, the one
departure from the template (which does not): the model writes at most one
newline before `<channel|>`, so trimmed text plus the template's own
`\n<channel|>` reproduces the model's bytes when it wrote that newline and the
session's incremental prefill hits; when the model closes the channel with no
newline the next step replays from the primed prefix instead. Two assistant
messages in a row continue one `model` turn (contents trimmed and
concatenated, no marker between them). Assistant content and reasoning
lose any control marker they spell (`<|channel>`, `<channel|>`, the six
tool markers, the `<|"|>` delimiter, `<|turn>`, `<turn|>`, `<bos>`), as the
template strips thoughts.

**Tools.** Declarations go into the system turn after the system text with
no separator, one `<|tool>declaration:NAME{…}<tool|>` each: the description,
then `parameters:{properties:{…},required:[…],type:<|"|>OBJECT<|"|>}` for a
nonempty schema. The profile renders the JSON-Schema subset the template's
macro understands — object schemas with `properties` (keys in byte order,
as the reference's case-sensitive `dictsort`), `required`, a mandatory
string `type` (uppercased), and per property an optional `description`,
`enum` on strings, `items` on arrays, nested `properties`/`required` on
objects, and `nullable` — and rejects anything else with
`error.UnsupportedContent` rather than approximating (a schema without a
type would leave the template's braces unclosed). An assistant call is
`<|tool_call>call:NAME{key:value,…}<tool_call|>`, keys in byte order, values
in the template's DSL: strings literal between `<|"|>` delimiters (there is
no escape, so a string containing the delimiter or any marker is rejected),
numbers as their JSON text, `true`/`false`/`null`, nested objects with bare
keys, and arrays. The results that answer the calls follow inside the same
model turn as `<|tool_response>response:NAME{value:<|"|>CONTENT<|"|>}<tool_response|>`
(the content untrimmed and framed, so a delimiter it spells stays text), then the message's content; `<turn|>\n` closes the
turn unless it continues into the next assistant message or the conversation
ends on results with no content, in which case the turn stays open and the
generation prompt adds only `<|channel>thought\n` when thinking is on. When
results are followed by content at the end of the conversation the template
closes the turn and adds no model turn; the reference's chat layer then
appends `<|turn>model\n`, and the profile reproduces that (its
`/apply-template` renders through the same layer, so the fixtures pin it).
Decoding uses the `<|tool_call>` / `<tool_call|>` control tokens (48/49)
through the shared stream decoder and `gemma4.parseTool`, a bounded
recursive-descent reader of the DSL (nesting depth 16, whitespace where the
reference grammar allows it, and JSON strings in plain quotes, which the
26B-A4B sometimes writes) that writes the JSON object the agent consumes;
a malformed body is cut like a truncated one (below), never released as
text ([tool-calling.md § Gemma](tool-calling.md#gemma-calls-and-results-inside-a-model-turn)). The model hands off by
emitting `<|tool_response>` after its calls — Google's guide calls it an
additional stop sequence and the reference marks it end-of-generation by
name — so it is the profile's third stop token and several calls in one step
all arrive before it.

Sampling defaults are the file's own hint, the same in both modes
(`general.sampling.temp` 1.0, `top_p` 0.95, `top_k` 64; no penalties, no
`min_p`), recorded as the file's claim: no published per-mode table was
pinned for Gemma 4, unlike Qwen3.8's. The stop set is `<turn|>` (106, the
K-quant file's `eos_token_id`), `<eos>` (1, the QAT file's), and
`<|tool_response>` (50, the tool handoff); both files carry all three tokens
and the same template.

## Gemma 4 E-series (`gemma4_e`)

The E4B's template is Gemma 4's with one rendered difference, read from the
two templates' diff: with thinking off the generation prompt is a bare
`<|turn>model\n`, with no empty thought channel, so the model may open one
itself. Everything else (turns, tools, reasoning gate, stop tokens, stream
markers, sampling) is Gemma 4's: `gemma4_e.render` is `gemma4.renderWith`
with `empty_thought_when_off = false`, and `rewritesTurn` is true only with
thinking on. Its fixture, `gemma4_e-text.json`, holds 14 text cases
captured from the E4B file's own template.

## Muse Glimmer (`muse_glimmer`)

Turns are `<|start|>ROLE<|message|>…<|eot|>` with nothing trimmed: the
template writes content and reasoning verbatim, so the profile does too.
The prompt starts with `<|begin_of_text|>` as text (the encoder never adds
BOS; the reference server stripped it from `/apply-template` output and
the fixture test prepends it). Every leading system or developer message is
its own system turn (the reference sends `developer` as `system`), each
closed by `\n\nReasoning strength: LEVEL.` and
`\n\n# Valid recipients: "self", "user".`; without one the profile writes
the template's synthesized turn, `You are a helpful AI assistant.` and
`Knowledge cutoff: 2026-01-04.`, **without** the `Current date:` line the
template adds from the server's clock (decided 2026-09-16: the profile
takes no clock; the fixture test removes the captured line). The level is
the template's `reasoning_strength`: `low`, `medium`, `high`, `xhigh`, and
the shared `off` renders as `low` because the model always opens a
reasoning message. Assistant history keeps `reasoning_content` wherever
it appears, as its own message `<|start|>assistant to=self<|message|>…<|eom|>`
(the template has no last-user gate), then the answer as
`<|start|>assistant to=user<|message|>…<|eot|>`; two assistant messages in
a row are two such turns. The generation prompt is `<|start|>assistant`.
Content carrying a control marker (`<|start|>`, `<|message|>`, `<|eom|>`,
`<|eot|>`, `<|end_of_text|>`, `<|begin_of_text|>`) is rejected; the
assistant's own past text loses them instead.

**Decoding** is the stream decoder's channel grammar (`stream.Channel`):
the model's output is a sequence of messages `HEADER<|message|>BODY`
ended by `<|eom|>` (200007) or the next `<|start|>` (200022), the first
header arriving as ordinary text (` to=self`) because the prompt ends
inside it. The header routes the body: `assistant to=self` is thinking,
`assistant` or `assistant to=user` the answer, any other `to=NAME` a tool
body when the profile parses one and answer text otherwise. A
header still open at a stop, or longer than 256 bytes, is released as
answer text. `<|eot|>` (200008) and `<|end_of_text|>` (200001) are the
stop tokens; `<|eom|>` is not, since the model continues after it.

**Tools.** Declarations go into every system turn between the strength
line and the recipients line: the template's instructions, one metadata
line per namespace, one JSON schema line per tool in the reference's
`tojson` style (insertion-ordered keys, `", "` and `": "` separators,
non-ASCII literal), and the fixed example; the recipients line gains one
`"NS.*"` per namespace. An assistant call is its own message
`<|start|>assistant to=NAME<|message|>` with the ATEM block (one invoke,
one `<atem:parameter name="KEY">` per argument: strings verbatim, `true`/
`false`/`null`, numbers as written, lists and objects as JSON), ended by
`<|eom|>` before another call and `<|eot|>` after the last; content beside
calls is not rendered. A result is its own `<|start|>tool NAME<|message|>`
turn wrapping the content in `<tool_output name="NAME">`. Names cannot
carry a quote, an angle bracket, or whitespace; content, results, names,
and argument strings carrying the ATEM markup or a control marker are
rejected. Decoding is the channel grammar above with `muse_glimmer.parseTool`
on `to=NAME` bodies: a body that is one well-formed block yields the call
(a value that parses as JSON keeps its type, anything else is a literal
string), anything else is released as text; the body completes at
`<|eom|>`, the next `<|start|>`, or the turn's `<|eot|>` stop, and is cut
on a budget stop or a cancellation ([tool-calling.md](tool-calling.md)).

Sampling defaults are the model card's "Best Practices" (temperature 1.0,
top-p 0.95, top-k 64; no `min_p`, no penalties), the same for every
strength; the file declares no `general.sampling.*` keys.

## Evidence and reproduction

The committed fixtures
([qwen38-text.json](../../inference/src/profiles/fixtures/qwen38-text.json),
[gemma4-text.json](../../inference/src/profiles/fixtures/gemma4-text.json),
[gemma4_e-text.json](../../inference/src/profiles/fixtures/gemma4_e-text.json),
[muse_glimmer-text.json](../../inference/src/profiles/fixtures/muse_glimmer-text.json),
described in [muse-glimmer.md](../models/muse-glimmer.md#chat-template))
hold prompts llama.cpp's server rendered from each artifact's own template
when the fixtures were captured, for seven conversations (a single turn, a system message, later
system/developer messages, assistant history with reasoning, two assistant
messages in a row, Unicode with nonbreaking spaces, and whitespace-only
user text): 35 cases across five efforts for Qwen3.8 (re-captured
2026-09-19 with `high`), 14 across `off` and `medium` for Gemma 4 (the
Zig test asserts `low`, `high`, and `xhigh` render like `medium`), 14 for
the E4B's template, and 28
across four strengths for Muse Glimmer (its test removes the synthesized
turn's captured date line and prepends `<|begin_of_text|>`). Native
rendering matches every prompt byte for byte (default `zig build test`). Each file also retains the exact token ids of every
rendered prompt and of ten standalone strings captured with special-token
parsing both off and on (`add_special=false`): code, numbers,
contractions, leading whitespace, CRLF, multilingual text, emoji, and the
profile's own control markers. Error tests cover invalid UTF-8,
unsupported representations, ordering, exact limits, and every allocation
failure point.

The tool fixtures
([qwen38-tools.json](../../inference/src/profiles/fixtures/qwen38-tools.json),
captured from llama.cpp's server like the text fixtures) add 20 Qwen cases across the
four efforts: a tools block alone; one call with content and reasoning; two
calls in one assistant message; a call with empty arguments; and a call whose
string argument contains a newline and a quote. Each carries the same tools
and, where present, the assistant/reasoning/tool-result history, so the tools
block, the rendered calls, and the folded `<tool_response>` user turn are all
pinned byte for byte through the same `render` path.
[gemma4-tools.json](../../inference/src/profiles/fixtures/gemma4-tools.json)
(`--profile gemma4`, captured 2026-09-16 on the K-quant 12B) adds 24 Gemma
cases across `off` and `medium`: the same five shapes with a third,
schema-rich declaration (description, integer, boolean, enum, array of
strings, nested object); a system message with tools; a call with nested
object, array, float, boolean, and null arguments; a loop ending on results
with and without reasoning and with content (the reference's reopened turn);
two steps in one loop; and a final answer after results followed by a user
turn. The Zig test prepends `<bos>` as the text test does.
[muse_glimmer-tools.json](../../inference/src/profiles/fixtures/muse_glimmer-tools.json)
(`--profile muse_glimmer`, captured 2026-09-19) adds 52 Muse cases across
the four strengths: Gemma's shapes with a fourth declaration whose texts
need JSON escaping (quotes, a backslash, a tab, a newline, non-ASCII, `<`,
`&`), plus a multi-line string argument; the Zig test prepends
`<|begin_of_text|>` and removes the synthesized turn's date line as the
text test does.

`make gate NAME=qwen38-vocabulary` and `make gate NAME=gemma4-e4b-vocabulary`
(`zig build test-vocabulary`, `inference/vocabulary-check.zig`) are the
checks against the real artifact: each selects the profile from the file's
template, asserts the vocabulary facts (size, merges, BOS/EOS, selected ids,
the stop set resolving), and encodes every captured string and prompt to
the pinned ids (Qwen3.8: 20 standalone and 35 prompts, plus the byte-level
single-piece checks; Gemma 4 E4B: 20 and 14). The check knows the `gemma4`
facts too; `muse_glimmer` has no pinned facts there and is refused
(`UnsupportedProfile`). Default tests exercise committed fixtures and
synthetic vocabularies only.

The captures came from llama.cpp's server on the pinned files, the
reference while the first families were brought up
([record](../benchmarks/record/prompt-profile.md#how-the-fixtures-were-captured-2026-09-07-to-2026-09-19));
a re-pinned file's fixtures are made by nuclis
([models § Which executor makes which fixture](../models/README.md#which-executor-makes-which-fixture)).
The templates are adapted, not copied; their origins and licenses are in
[THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md).

**Other template revisions.** A file converted with another revision of a
profile's template (a finetune, an older converter) carries another
digest and is refused. A revision becomes an alias when its lowercase hex
digest is added to the module's `template_aliases`; `profiles.forTemplate`
then selects the profile for it as for the pinned digest (the root test
"profiles are selected by template digest" holds that). The code checks
nothing else, so the change that adds the digest carries the evidence in
its record: the revision renders every case of the profile's text and tool
fixtures byte for byte like the pinned template, and its file encodes them
to the same ids. Every profile's alias list is empty. A file whose revision
is not an alias runs with `--prompt-profile <p>` (or a registry entry's
`profile`), which renders the pinned protocol onto it; the agent says so at
startup.

Two revisions are not aliases: the Gemma 4 one community 12B finetunes
ship (`dc311bb0…`) and Bonsai 2 27B's (`c3cf9e34…`;
[bonsai.md § Chat template](../models/bonsai.md#chat-template)); the cases
that differ are in the [record](../benchmarks/record/prompt-profile.md#template-revisions-checked-2026-09-17-2026-09-18).
