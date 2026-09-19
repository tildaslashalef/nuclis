# Prompt profiles and tokenizer fixtures

A prompt profile is the checkpoint's conversation contract, kept apart from
its numerical architecture: the chat template rendered as a bounded Zig
function, the tokens that end a turn, the markers that delimit the model's
reasoning in generated text, and the checkpoint's official sampling
defaults. The registry is `inference/src/profiles/root.zig`: one tag per
profile module, selected at load by the SHA-256 of the artifact's
`tokenizer.chat_template` (`profiles.forDocument`), never by architecture,
so a conversation is never interpreted with a template the profile's
fixtures did not verify. Two profiles exist:

| Profile | Module | Template SHA-256 | Stop tokens | Reasoning markers |
| --- | --- | --- | --- | --- |
| `qwen38` | [qwen38.zig](../../inference/src/profiles/qwen38.zig) | `12827f24…` | `<|im_end|>`, `<|endoftext|>` | `<think>` … `</think>` |
| `gemma4` | [gemma4.zig](../../inference/src/profiles/gemma4.zig) | `845f1ee4…` | `<turn|>`, `<eos>`, `<|tool_response>` | `<|channel>thought\n` … `<channel|>` |

Every module exposes the same surface: `template_sha256`, `render`,
`samplingDefaults`, `stop_tokens`, `reasoning`, and `stream_markers`. The engine resolves the stop
tokens to ids in the artifact's vocabulary at load (`MissingStopToken` when
one is absent: a template/tokenizer mismatch is refused, never guessed) and
ends generation on any of them; without a matching profile it stops on the
file's `eos_token_id` alone and refuses chat rendering. `engine.complete`
uses `Profile.decoder` to resolve reasoning controls in the vocabulary and
stream semantic events. Actual control IDs delimit channels; ordinary tokens
spelling the same text do not. Gemma's `thought\n` suffix is ordinary text
following the opening control, so the decoder handles its partial arrival.
The agent stores the channels directly for history and session logging;
`generate`/`agent` sample with the profile's defaults for the
`--think` effort ([generation.md](generation.md#sampling-profiles-and-the-selection-chain-modl-01)).
Nothing under `src/` names a profile module; the executable uses the tag.

## Shared contract

`profiles.Profile.render(allocator, messages, tools, effort, limits)`
returns an owned UTF-8 prompt that the caller frees with the same allocator.
Messages borrow `role`, `content`, and optional `reasoning_content`. Roles
are system, developer, user, assistant, and tool. A completion-ready
conversation ends with a user message or a fully answered tool-result group;
system/developer messages are allowed only at the beginning. All strings
must be valid UTF-8; reasoning on a non-assistant message is
`UnsupportedContent`. A control marker spelled as text is refused in user
and tool content and in tool names (structure smuggled past the profile)
but **removed** from the assistant's own content and reasoning: that text
is the model's past output (a call released as text when its closing token
never came, a channel opened as text), and re-encoding it would turn the
marker into a control token, while refusing it would end every later turn
of the session. The template's own `strip_thinking` is the same idea.

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
them, and one without `off` renders it as its lowest level. Validation is
separate from support: a structurally valid tools input still passes
`validate`, and a profile whose template defined no tool grammar rejects it
with `error.ToolsUnsupported`; Qwen3.8 and Gemma 4 render and decode their
native tool path against pinned fixtures. No profile implements multimodal
content, assistant prefill, or a general Jinja interpreter.

## Qwen3.8 (`qwen38`)

Leading system/developer messages are trimmed and merged with newlines;
empty leading system content is omitted. `low` and `xhigh` prepend the
artifact's reasoning instructions to the system block; `medium` adds none,
and `high` renders the `xhigh` line, as the template resolves it.
All modes append the assistant generation prefix `<|im_start|>assistant\n<think>\n`;
`off` closes it at once with an empty thinking block. Previous assistant
reasoning is preserved in history as `<think>…</think>` before the answer,
in every mode. Assistant answers containing `<think>` or `</think>` are
rejected: callers supply reasoning separately. The renderer trims ASCII
whitespace as the pinned llama.cpp Jinja implementation does; nonbreaking
spaces and other Unicode stay intact.

**Tools.** With tool definitions the system turn becomes the artifact's tools
block: the reasoning instruction (when the mode has one), `# Tools`, one
OpenAI-shaped declaration per tool inside `<tools>…</tools>`, the template's
format reminder, then any merged system text. Declarations and non-string
argument values are serialized in the reference's `tojson` style (a space
after `:` and between items, keys in insertion order, non-ASCII literal).
Assistant calls render as `<tool_call>\n<function=NAME>…` with one
`<parameter=KEY>` block per argument (string values literal, others JSON);
consecutive tool results fold into one `<|im_start|>user` turn of
`<tool_response>` blocks, exactly as the template does. Assistant content
carrying a control marker is rejected so structure cannot be smuggled into the
answer. Gemma's native tool grammar is a different shape and its result
handoff resumes the model turn; see its section below and
[tool-calling.md](tool-calling.md).

## Gemma 4 (`gemma4`)

Turns are `<|turn>role\n…<turn|>\n`; the assistant's role name is `model`.
The prompt starts with `<bos>` as text: the tree's encoder never adds BOS
(MODL-05), so the profile writes the marker the template emits (the reference
server strips it from `/apply-template` output because its tokenizer adds
it back; the token streams are identical, and the fixture test prepends it).
The first message, when system or developer, goes into the system turn;
later system and developer messages render as `system` turns of their own,
empty ones included (the reference sends `developer` as `system`). Thinking
is a switch, not a level: `off` renders no `<|think|>` and pre-closes an
empty thought channel in the generation prompt
(`<|turn>model\n<|channel>thought\n<channel|>`) so the model answers
directly; any other effort puts `<|think|>\n` at the top of the system
turn (which then exists even without a system message) and ends the prompt
at `<|turn>model\n`, after which the model opens its own channel. Assistant
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
newline the next step replays from the primed prefix instead (measured on
the first step of the live check in the engineering log). Two assistant
messages in a row continue one `model` turn (contents trimmed and
concatenated, no marker between them). Assistant content or reasoning
carrying any control marker (`<|channel>`, `<channel|>`, the six tool
markers, or the `<|"|>` delimiter) is rejected rather than stripped as the
template would.

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
(the content untrimmed), then the message's content; `<turn|>\n` closes the
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
reference grammar allows it) that writes the JSON object the agent consumes;
a malformed or truncated body is released as text. The model hands off by
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

## Muse Glimmer (`muse_glimmer`)

Turns are `<|start|>ROLE<|message|>…<|eot|>` with nothing trimmed: the
template writes content and reasoning verbatim, so the profile does too.
The prompt starts with `<|begin_of_text|>` as text (the encoder never adds
BOS; the reference server strips it from `/apply-template` output and the
fixture test prepends it). Every leading system or developer message is
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
body when the profile parses one (AGNT-10) and answer text otherwise. A
header still open at a stop, or longer than 256 bytes, is released as
answer text. `<|eot|>` (200008) and `<|end_of_text|>` (200001) are the
stop tokens; `<|eom|>` is not, since the model continues after it.

**Tools** are a later unit (AGNT-10): a tool definition or tool history is
`error.ToolsUnsupported`. Sampling defaults are the model card's "Best
Practices" (temperature 1.0, top-p 0.95, top-k 64; no `min_p`, no
penalties), the same for every strength; the file declares no
`general.sampling.*` keys.

## Evidence and reproduction

The committed fixtures
([qwen38-text.json](../../inference/src/profiles/fixtures/qwen38-text.json),
[gemma4-text.json](../../inference/src/profiles/fixtures/gemma4-text.json),
[muse_glimmer-text.json](../../inference/src/profiles/fixtures/muse_glimmer-text.json),
described in [muse-glimmer.md](muse-glimmer.md#chat-template))
hold prompts the pinned reference server rendered from the artifact's own
template for seven conversations (a single turn, a system message, later
system/developer messages, assistant history with reasoning, two assistant
messages in a row, Unicode with nonbreaking spaces, and whitespace-only
user text): 35 cases across five efforts for Qwen3.8 (re-captured
2026-09-19 with `high`), 14 across `off` and `medium` for Gemma 4 (the
Zig test asserts `low`, `high`, and `xhigh` render like `medium`), and 28
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
captured by `scripts/profile-tools-fixtures.py`) add 20 Qwen cases across the
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

`make test-vocabulary` (`inference/vocabulary-check.zig`; `MODEL=<path>`
for the Gemma file) is the opt-in check against the real artifact: it
selects the profile from the file's template, asserts the vocabulary
facts (size, merges, BOS/EOS, selected ids, the stop set resolving), and
encodes every captured string and prompt to the reference's ids (Qwen3.8:
20 standalone and 35 prompts, plus the byte-level single-piece checks;
Gemma 4: 20 and 14; Muse Glimmer: 20 and 28). Default tests exercise committed fixtures and
synthetic vocabularies only.

Capture, with the reference built as in
[reference-baseline.md](reference-baseline.md): start the server on the
artifact from the repository root (`--reasoning off` is the Qwen recipe's
and does not affect `/apply-template`):

```sh
.zig-cache/reference/llama.cpp/build/bin/llama-server \
  --model "$HOME/.nuclis/models/unsloth/gemma-4-12b-it-GGUF/gemma-4-12b-it-UD-Q4_K_XL.gguf" \
  --ctx-size 2048 --parallel 1 --device MTL0 --n-gpu-layers 99 \
  --host 127.0.0.1 --port 18087 --no-webui
```

In another terminal:

```sh
python3 scripts/tokenizer-fixtures.py --profile gemma4   # or --profile qwen38, --profile muse_glimmer
python3 scripts/profile-tools-fixtures.py --profile gemma4   # or --profile qwen38 (default)
zig build test --global-cache-dir .zig-cache/global
```

The script accepts loopback only, ignores proxy configuration, verifies
the model's full checksum, size, and template digest (the server reports
the template without the file's trailing newline; both forms are
accepted), and checks the reference build id. It passes the thinking
settings per request as the template's own keyword arguments
(`enable_thinking` and, for Qwen3.8, `reasoning_effort` and
`preserve_thinking`) and uses `/apply-template`, `/tokenize`, and
`/detokenize`. It sends only synthetic content and performs no
generation. Stop the server afterward.

Captures: Qwen3.8 on 2026-09-07 (re-captured 2026-09-12 with the seven
conversations, 2026-09-19 with `high`), Gemma 4 on 2026-09-12, Muse
Glimmer on 2026-09-19, all with llama.cpp
`7620399f58aebfd2196b74021f9581bcf7218cb9` on the pinned files
([artifacts.md](artifacts.md#pinned-commits-and-digests-modl-02-2026-09-11)).
The templates are adapted, not copied; their origins and licenses are in
[THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md).

**Other template revisions.** A file converted with another revision of a
profile's template (a finetune, an older converter) carries another
digest and is refused. `scripts/profile-alias-check.py --profile <p>`
is the gate for accepting one: with the reference server holding that
file, it replays every pinned text, tool, and token case through
`/apply-template` and `/tokenize` and, only when every prompt and token
stream is byte-identical, records the digest with the file's identity in
`fixtures/<profile>-aliases.json`; the profile's `template_aliases` lists
the same digests and `profiles.forTemplate` accepts them. Checked on
2026-09-17 for the 17,530-byte Gemma 4 revision that finetunes such as
`Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf` ship
(`dc311bb0…`): **18 of 38 cases differ**, all in history rendering — a
thought on an earlier tool-call step is dropped (the pinned template keeps
it), two consecutive assistant messages become two turns, and a tool-result
group before a user turn is left without its `<turn|>` — so it is not an
alias, and both profiles' alias lists are empty. Such a file runs with
`--prompt-profile <p>` (or a registry entry's `profile`), which renders the
pinned protocol onto it; the agent says so at startup. Checked on
2026-09-18 for Bonsai 2 27B's template (`c3cf9e34…`, 8,952 bytes: the
upstream Qwen3.8 template without Unsloth's fixes) on the PrismML fork's
server (`--reference-revision 5d80cff0…`; the script reports a template's
`raise_exception` as a mismatch rather than crashing): **4 of 68 cases
differ**, the `merged_system_*` histories, which that template refuses
(`System message must be at the beginning.`) where the pinned one merges
them; every other prompt and token stream is byte-identical
([bonsai.md § Chat template](bonsai.md#chat-template)). Not an alias
either; MODL-17 chooses the entry's profile from that evidence.

## Completion events (AGNT-01 session 1)

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
call when the closing token arrives. A call still open at EOS, a budget stop,
or a cancellation — and a body the parser refuses — is released as answer text
and never executed, bracket text included (copied when the bracket arrives:
a piece is the caller's per-token buffer). A reasoning channel the model
opens while answering is thinking from there, shown as a further block,
whatever the effort (with thinking off, Gemma still opens an empty channel
after a tool result); a closing token with no channel open stays literal. The Gemma decoder is the same machinery with its own
brackets (`<|tool_call>` / `<tool_call|>`) and `gemma4.parseTool`; its
handoff is the `<|tool_response>` stop token. See
[tool-calling.md](tool-calling.md) for the model-card and pinned-template
evidence.
