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
| `gemma4` | [gemma4.zig](../../inference/src/profiles/gemma4.zig) | `845f1ee4…` | `<turn|>`, `<eos>` | `<|channel>thought\n` … `<channel|>` |

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
`UnsupportedContent`.

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
`xhigh`), named after Qwen3.8's levels because it came first; a profile
with fewer levels collapses them. Validation is separate from support: a
structurally valid tools input still passes `validate`, and both profiles
reject it with `error.ToolsUnsupported` until AGNT-05–AGNT-06 pin the native tools block,
call grammar, and result role. Neither profile implements native tools,
multimodal content, assistant prefill, or a general Jinja interpreter.

## Qwen3.8 (`qwen38`)

Leading system/developer messages are trimmed and merged with newlines;
empty leading system content is omitted. `low` and `xhigh` prepend the
artifact's reasoning instructions to the system block; `medium` adds none.
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
handoff resumes the model turn; it is deliberately rendered as
`error.ToolsUnsupported` rather than approximated (see
[tool-calling.md](tool-calling.md)).

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
reasoning is never rendered in history (the template's gate only passes
for turns after the last user message, and a conversation always ends with
one): `reasoning_content` is accepted and dropped. Two assistant messages
in a row continue one `model` turn (contents trimmed and concatenated, no
marker between them). Assistant content carrying `<|channel>` or
`<channel|>` is rejected rather than stripped as the template would.

Sampling defaults are the file's own hint, the same in both modes
(`general.sampling.temp` 1.0, `top_p` 0.95, `top_k` 64; no penalties, no
`min_p`), recorded as the file's claim: no published per-mode table was
pinned for Gemma 4, unlike Qwen3.8's. The stop set is `<turn|>` (106, the
K-quant file's `eos_token_id`) and `<eos>` (1, the QAT file's); both files
carry both tokens and the same template.

## Evidence and reproduction

The committed fixtures
([qwen38-text.json](../../inference/src/profiles/fixtures/qwen38-text.json),
[gemma4-text.json](../../inference/src/profiles/fixtures/gemma4-text.json))
hold prompts the pinned reference server rendered from the artifact's own
template for seven conversations (a single turn, a system message, later
system/developer messages, assistant history with reasoning, two assistant
messages in a row, Unicode with nonbreaking spaces, and whitespace-only
user text): 28 cases across four efforts for Qwen3.8, 14 across `off` and
`medium` for Gemma 4 (the Zig test asserts `low` and `xhigh` render like
`medium`). Native rendering matches every prompt byte for byte (default
`zig build test`). Each file also retains the exact token ids of every
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

`make test-vocabulary` (`inference/vocabulary-check.zig`; `MODEL=<path>`
for the Gemma file) is the opt-in check against the real artifact: it
selects the profile from the file's template, asserts the vocabulary
facts (size, merges, BOS/EOS, selected ids, the stop set resolving), and
encodes every captured string and prompt to the reference's ids (Qwen3.8:
20 standalone and 28 prompts, plus the byte-level single-piece checks;
Gemma 4: 20 and 14). Default tests exercise committed fixtures and
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
python3 scripts/tokenizer-fixtures.py --profile gemma4   # or --profile qwen38
python3 scripts/profile-tools-fixtures.py                # Qwen native tool prompts
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
conversations), Gemma 4 on 2026-09-12, both with llama.cpp
`7620399f58aebfd2196b74021f9581bcf7218cb9` on the pinned files
([artifacts.md](artifacts.md#pinned-commits-and-digests-modl-02-2026-09-11)).
The templates are adapted, not copied; their origins and licenses are in
[THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md).

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
and never executed. Gemma declares no tool grammar and keeps rejecting tool
inputs with `error.ToolsUnsupported`. See [tool-calling.md](tool-calling.md)
for the model-card and pinned-template evidence, including Gemma's distinct
handoff.
