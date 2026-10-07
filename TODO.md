# TODO — active plan

This file is the queue: the units agreed and not yet merged, each with its
design, and a *Where we are* note. A unit is one session, one branch, and
one pull request, whose description is the record of the work; when it
merges, its section is deleted here, and when the last one goes, this file
is emptied back to this header. Requirements live in [docs/spec.md](docs/spec.md); the
engine map in [docs/architecture.md](docs/architecture.md); how to build,
test, and measure in [docs/development.md](docs/development.md).

Session protocol (also in [AGENTS.md](AGENTS.md)): read this file first. If
it lists work, summarize *Where we are* and ask the user how to continue. If
it is empty, ask what to work on and write the agreed plan here.

## Where we are

The plan below was agreed on 2026-10-07; nothing is built yet. `nuclis
serve` learns OpenAI's wire formats for the language models, so agents
and apps written for OpenAI-compatible servers connect by changing a base
URL. Next: **the completer and a streaming transport**, on branch
`serve-streaming-and-completer` (created, holding only this plan).

| Unit | Branch | What |
| --- | --- | --- |
| The completer and a streaming transport | `serve-streaming-and-completer` | `Completer` leaves the agent; the transport streams and takes per-route body limits; no route yet |
| Chat Completions, whole responses | `chat-completions` | `POST /v1/chat/completions` without streaming, language models in `GET /v1/models`, the API guide and spec |
| Chat Completions, streamed | `chat-completions-stream` | `stream: true`, cancellation by disconnect, keepalives, pi as the acceptance client |
| Responses, stateless | `responses-stateless` | `POST /v1/responses` with `store: false`, over the same core |

## The theme: an OpenAI-compatible service for the language models

Shared by every unit below; unit 2 writes it into
[docs/guide/api.md](docs/guide/api.md) and [docs/spec.md](docs/spec.md)
(§6 `serve`, §10, which lists the service as deferred).

**Decisions (user, 2026-10-07).**
- **The target is clients of OpenAI-compatible local servers.** The
  acceptance client is pi (`.reference/pi-mono`, read at `70759f48b`):
  its local-server providers use the `openai-completions` API (its llama.cpp
  provider, `packages/coding-agent/src/extensions/llama/provider.ts`), and
  its docs call Chat Completions the most compatible. So Chat Completions
  first; Responses second, stateless.
- **No engine change.** `inference/` is not touched. The service drives
  the agent's `Completer` (render through the profile, continue or restore
  the session, `engine.complete`, typed events), so a served conversation
  runs exactly as `nuclis chat` runs it.
- **The KV cache is invisible to clients.** Clients resend the whole
  conversation every request (pi's Responses client sends `store: false`
  and no `previous_response_id`). Reuse is the server's: `Completer`
  continues the live session when the new render extends what it consumed,
  else restores the longest cached prefix (`cache.Memory`), else prefills.
  The only visible trace is `usage.prompt_tokens_details.cached_tokens`.
  No stored responses, no response-id state, no `previous_response_id`.
- **Extensions are top-level fields, named as llama.cpp and vLLM name
  them** (`top_k`, `min_p`, `repetition_penalty`, `presence_penalty`,
  `seed`), not a `nuclis` object: clients pass extra sampling keys at the
  top level (pi's `samplingParams` merges them into the body).
- **Refused with `400 unsupported_feature`, because only an engine change
  would honour them:** `response_format` other than `text`
  (`json_schema`, `json_object`), `tool_choice` `required` or a named
  function, `n` > 1, `logprobs`/`top_logprobs`, audio. Fields that are
  only advice (`user`, `store`, `metadata`, `prompt_cache_key`,
  `service_tier`, `parallel_tool_calls: true`) are accepted and ignored.
- **One language model open**, beside the decision pool's two
  (`decisions/pool.zig` `capacity`). A request naming another language
  model closes it and opens that one on the GPU worker, as the agent's
  `/model` does; the log line says so. The model runs with the settings
  `nuclis chat` would resolve for it (`config.Resolved`: `ctx_size`,
  `kv_precision`, `think`, sampling, speculative, `cache`,
  `agent.thinking_budget`).
- **Status codes follow OpenAI's SDKs, not the decision routes:** 400 for
  validation (the decision routes keep 422), 404 `model_not_found`, 529
  `busy`/`timeout`, 503 `shutting_down`. The error body gains `type`
  (`invalid_request_error` for 4xx, `server_error` for 5xx) and `param`
  (null or the field), which the SDKs read; decision clients ignore them.
  The SDKs retry 409, 429 and 5xx, so no route answers 409.

## The completer and a streaming transport

Base: `40b69f1`

**Why.** The two prerequisites of a language-model route, landed with no
route: the completion core the agent owns becomes shared, and the
transport can stream. Behaviour of `nuclis chat` and of the decision
routes does not change.

1. **`Completer` leaves the agent.** Move `Completer`, the `Model` seam
   and what it carries (`Model`, `Reply`, `Replay`, `Image`, `Overflow`,
   `increment`, `placeholderRuns`) from `src/agent/loop.zig` into a new
   `src/completer.zig`, and `src/agent/cache.zig` to
   `src/completer/cache.zig` (its `../tui/style.zig` import serves
   `nuclis cache ls` and `clear`, `Listing.render`, `ls`, `clear`; the
   path changes, nothing else). `loop.zig` re-exports the
   names it uses (`pub const Completer = completer.Completer;` …) so the
   agent, `print.zig` and `root.zig` compile unchanged. Their tests move
   with them; the test count does not drop. Update the references in
   `docs/engine/session.md` § The agent's token cache and
   `docs/app/agent.md` (`make docs-check`).
2. **The output budget clamps.** `Completer.run` reserves the whole
   `buffers.generated` slice and returns `ContextFull` when prompt plus
   slice exceed the window. Clients size the budget from their own idea
   of the window (pi sends `min(maxTokens, contextWindow − estimate −
   4096)`, its defaults 16,384 and 128,000; its llama.cpp provider sets
   `maxTokens` to the whole window), so a server must cap rather than
   refuse. Add `clamp_budget: bool = false`:
   when set and the prompt fits, the limit passed to `engine.complete`
   (and to `config.thinkingBudget`) is the space left, at least 1;
   `ContextFull` only when the prompt itself does not fit. The agent
   keeps `false`. Test with a stub-free unit test on the arithmetic
   (extract `fn budget(position, prompt, slice, capacity, clamp) ?usize`).
3. **Streaming responses.** `http.Response` gains
   `stream: ?Stream = null`, `Stream = struct { context: *anyopaque,
   write: *const fn (*anyopaque, std.Io, *std.http.BodyWriter) bool }`.
   `Exchange.respond` calls `request.respondStreaming` (chunked, keep-alive
   kept) with the status and `content-type`, then `write`; `false` from it,
   or a failed `end`, closes the connection. The log line records bytes
   written and the duration to the end of the stream. Tests in
   `http.zig`'s in-memory style: a stream of three chunks round-trips as
   chunked, a keep-alive request after it is served, a write that fails
   midway closes.
4. **Per-route body limits.** Images arrive base64 in the body, so the
   chat routes need more than 4 MiB. `Router.add` takes an optional body
   limit; `http.Handler` gains `body_limit: ?*const fn (*anyopaque, path)
   usize`, which `serveOne` asks before reading the body. Limits:
   `limits.transport.max_body` (4 MiB) by default, 32 MiB for the chat
   routes (8 images at `decide.max_image_bytes`-sized inputs do not fit;
   the per-image bound stays the decision one, 16 MiB). Test: a 5 MiB body
   is 413 on `/v1/decisions` and read on a route registered at 32 MiB.
5. **The cross-thread pipe** the streamed route will need: `src/api/pipe.zig`,
   a bounded byte queue (mutex, `std.Io.Condition`, 1 MiB) the GPU worker
   appends to and the connection task drains, with `close` (producer done)
   and `abandon` (consumer gone: later appends return `error.Abandoned`).
   The worker never blocks on the socket: an append past the bound waits
   at most until the consumer drains or abandons. Tests with two tasks.

**Gates.** `zig build test`, `zig fmt --check`, `make docs-check`, `make
verify-auto` (no inference path changes, so no Metal tier). Exercise:
`./zig-out/bin/nuclis serve` answers `/v1/decisions` and `/v1/health` as
before; `nuclis agent --print` runs one turn (the moved `Completer`).

## Chat Completions, whole responses

Base: set when the branch is cut from `main` after the unit above merges.

**Why.** The route itself, without streaming: everything a request means
is decided here, and the streamed unit only changes how it is delivered.

1. **The service.** `src/api/chat/service.zig`, registered beside the
   decision service in `src/api/root.zig` `Server.register`: `POST
   /v1/chat/completions` (32 MiB body). The handler parses and validates
   on the connection task, submits one `gpu.Item` that runs the
   conversation, and waits as the decision service does (`serve.timeout`
   before it starts; a started item is waited for). The item owns no
   socket.
2. **The language model on the worker.** `src/api/chat/model.zig`: the
   open model (`engine.Engine`), its `Completer` with `clamp_budget =
   true`, the sampler, history and buffers, built as `agent/print.zig`
   `run` builds them, from `config.Resolved` for the model name
   (`engine.model` when the request names none). Opened lazily by the first
   item (or at start with `serve --chat-model <name>`; decide the flag
   name against `cli.zig`), closed and reopened when a request names
   another. `cache.Memory` at `cache.memory_bytes`; no disk tier in this
   unit (`save_turns = false`).
3. **Request → `Completer.run`.** Per request, on the worker:
   `completer.effort` from `reasoning_effort` (`none` → `off`, `minimal` →
   `low`, others by name; absent → the resolved `think`), sampler options
   from `profile.samplingOptions(effort, overrides)` with the request's
   `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`,
   `repetition_penalty`, `seed` (absent → `0`), `buffers.generated`
   sliced to `max_completion_tokens` or `max_tokens` (absent → the
   resolved `max_tokens`; at most `config.max_output_tokens`), then
   `run(messages, definitions, images, sink)`; `checkpoint` after a
   response that ends in an answer, as the loop does at a turn end.
   Mapping (`src/api/chat/wire.zig`, pure, unit-tested without a model):

   | Wire | `Profile` |
   | --- | --- |
   | `messages[].role` `system`, `developer`, `user`, `assistant`, `tool` | `Role`, one to one |
   | `content`: a string, or parts `text` and `image_url` (`data:` URL) | `content`, joined; images through `Model.encode_image`, `ImageRef` on the message; a remote URL is `400` |
   | assistant `reasoning_content` (also read: `reasoning`) | `reasoning_content` |
   | assistant `tool_calls[{id, type: "function", function{name, arguments}}]` | `tool_calls`, ids mapped to `u32` by first appearance in this request |
   | tool `tool_call_id` | `tool_call_id` through the same map; an unknown id is `400` |
   | `tools[{type: "function", function{name, description, parameters}}]` | `ToolDefinition`, `parameters` re-serialized as JSON text |
   | `tool_choice` `auto` / `none` | definitions passed / none |

   What pi sends that the mapping must take (`openai-completions.ts`
   `buildParams`, `convertMessages`): the system prompt as role
   `developer` for a reasoning model; assistant `content: null` when the
   turn had only calls; `tools: []` when the history has calls but no
   tools; tool results as plain strings (`(no tool output)` when empty),
   and a tool's images as a following user message; `store: false`;
   `max_completion_tokens` (or `max_tokens`); no `tool_choice`, no
   `parallel_tool_calls`; ids passed through unchanged.

   Bounds: `profiles.Limits` (1,024 messages, 64 tools), 8 images
   (`decide.max_images`), a model without a projector answers
   `400 images_unsupported`.
4. **Events → the response.** The sink accumulates: `thinking` →
   `message.reasoning_content`; `answer` → `message.content`;
   `tool_call` → `message.tool_calls[]` with a fresh id
   (`call_` + 24 random base62 characters, unique across requests, mapped
   to the engine's `u32` only inside a request); `tool_progress` and
   `tool_cut` → nothing. `finish_reason`: `tool_calls` when calls were
   decoded, else `stop` for `eos`, `length` for `token_budget` and
   `context_limit`; `failure` is `500 model_failed`. `usage`:
   `prompt_tokens` (rendered prompt), `completion_tokens`,
   `prompt_tokens_details.cached_tokens` (tokens continued or restored:
   the `Replay`'s `restored`, or the consumed length when the session
   continued), `completion_tokens_details.reasoning_tokens`. The object:
   `id` (`chatcmpl-` + random), `object: "chat.completion"`, `created`,
   `model`, `choices[0]{index, message, finish_reason}`, `usage`. Model
   text is UTF-8 checked before it is written.
5. **`GET /v1/models`.** (pi reads its `models.json` list and never calls
   this; other clients build their pickers from it.) A `models.Source`
   for language models: registry
   and catalogue entries of kind language, `owned_by: "nuclis"`, and in the
   `nuclis` object `kind: "language"`, `present`, `loaded`,
   `context_length` (the resolved `ctx_size`), `images` (projector
   present), `efforts`.
6. **Errors.** `errors.zig` gains `type` and `param`, written for every
   route; the decision routes' tests updated for the two new fields.
   `ContextFull` with the prompt alone over the window is
   `400 context_length_exceeded` with the counts; pi matches
   `context_length_exceeded` in the error text and compacts rather than
   retrying (`packages/ai/src/utils/overflow.ts`). The 529 messages say
   "overloaded", which pi's agent-level retry matches
   (`RETRYABLE_PROVIDER_ERROR_PATTERN`).
7. **Documents.** `docs/guide/api.md` § Chat Completions (the request
   fields, the mapping, what is refused and why, a `curl` example, the
   pi `models.json` entry); `docs/spec.md` §6 `serve` row and §10;
   `nuclis serve --help`.

**Gates.** `zig build test` (the wire mapping and the sink without a
model), `make docs-check`, `make verify-auto`. Exercise with the built
binary: a `curl` conversation of three requests against Qwen3.8 (the
second and third report `cached_tokens` near the previous prompt), one
tool round trip (`tools` → `tool_calls` → a `tool` message → the answer),
one image with Gemma 4 12B, a refused `response_format`. A stdlib-only
check, `scripts/api-check.py` (`make api-check`, not a gate: it needs a
running server and a model), sends those requests and asserts the shapes
the OpenAI SDKs parse.

## Chat Completions, streamed

Base: set when the branch is cut.

**Why.** Every agent streams; pi always sends `stream: true`. This unit
delivers the same events as server-sent events and makes a long request
safe to abandon.

1. **SSE.** `stream: true` answers through `http.Response.stream`
   (`content-type: text/event-stream`). The GPU item's sink writes
   `data: {chunk}\n\n` lines into the `pipe.zig` queue; the connection
   task's `write` drains it to the body writer and flushes per chunk.
   Chunks: `object: "chat.completion.chunk"`, the first with
   `delta.role: "assistant"`; `thinking` → `delta.reasoning_content`;
   `answer` → `delta.content`; each `tool_call` → one chunk with
   `delta.tool_calls[{index, id, type, function{name, arguments}}]`, the
   arguments whole (the engine yields them only once parsed); the last
   with `finish_reason`; with `stream_options.include_usage`, a chunk with
   empty `choices` and `usage`; then `data: [DONE]`. The last choice
   chunk always carries a `finish_reason`: pi fails a stream without one
   ("Stream ended without finish_reason"). A failure after the head is a
   `data: {"error": {…}}` line and the end of the stream (the SDK throws
   on it).
2. **The head at once, then keepalives.** pi's HTTP idle timeout is 300 s
   for the head and for silence in the body (`http-dispatcher.ts`,
   `headersTimeout` and `bodyTimeout`); the SDKs' default is 600 s. A
   request can wait `serve.timeout` (300 s) in the queue and then prefill
   a 32K prompt for about 11 minutes (48.91 tok/s,
   docs/benchmarks/README.md). So a validated streamed request sends the
   200 head and flushes it before it is queued, and an SSE comment
   (`: queued`, `: prefill 8192/32639`) goes out every 10 s while it waits
   and while it prefills (`Observer.progress`); every SSE parser ignores
   comments. A queue timeout after the head is an in-stream error whose
   text says "timeout", which pi's agent-level retry matches.
3. **Cancellation.** A failed write, or the consumer gone, abandons the
   pipe and sets the item's cancel flag; the item's `Observer.check`
   returns `error.Cancelled`, `engine.complete` ends with `stop =
   cancelled`, and `Completer.run` resets the session (cause `cancel`).
   A write into a closed socket can succeed once before it fails, so
   every request, streamed or not, also watches its socket's read side
   while it waits; end of stream sets the same flag. Then an SDK's timeout
   and retry does not leave the abandoned generation running ahead of its
   own retry. pi's Esc aborts the fetch, which closes the connection
   (`Agent.abort` → the request's `signal`); its next request drops the
   aborted message from the history. Process Ctrl-C (`interrupt.zig`) is
   not used per request.
4. **Decisions behind a generation.** The executor is one queue, so a
   decision waits for a running generation. Measure it (a decision request
   while Qwen3.8 decodes 2,000 tokens) and record it in
   `docs/guide/api.md`; if the wait matters, the item's `progress` hook may
   run queued decision items between tokens, without an engine change —
   a decision for the follow-up list, not this unit's.
5. **Acceptance with pi.** `~/.pi/agent/models.json` provider `nuclis`
   (`baseUrl: http://127.0.0.1:8000/v1`, `api: "openai-completions"`, a
   dummy `apiKey`, which pi requires), and per model: `reasoning: true`
   (else pi sends no effort and replays no reasoning), `thinkingLevelMap`
   with `off: "none"` (else thinking off sends nothing and the server's
   default effort applies) and `xhigh: "xhigh"`, `contextWindow` and
   `maxTokens` from the model's `ctx_size`, `input: ["text", "image"]` for
   a model with a projector. pi's unknown-host compat defaults fit
   (`developer` role, `reasoning_effort`, `max_completion_tokens`,
   `include_usage`, no `strict`); the entry goes into `docs/guide/api.md`.
   A
   multi-step coding task in the agent playground with reasoning and tool
   calls; Esc mid-generation cancels it (the server log shows `cancelled`);
   per request, `cached_tokens` against `prompt_tokens`, written into the
   pull request: every step after the first should continue or restore,
   and a miss names its cause.

**Gates.** `zig build test` (chunk encoding from a scripted event list,
the pipe under a slow consumer), `make verify-auto`, `make api-check`
extended with a streamed request, the pi session above.

## Responses, stateless

Base: set when the branch is cut.

**Why.** Clients built on OpenAI's newer API (and pi's
`openai-responses` provider, which sends `store: false` and replays
every item) work too. A second translator over the same core, as
`/v1/systemone` and `/v1/decisions` are two wire formats over one
decision request.

1. `POST /v1/responses`: `input` (a string, or items `message`,
   `reasoning`, `function_call`, `function_call_output`), `instructions`
   (the first system message), `reasoning.effort`, `tools` (`type:
   "function"`, flat), `tool_choice`, `max_output_tokens`, sampling as
   above, `stream`. `store` is accepted only as `false` or absent (then
   treated as `false`): nothing is stored, and `previous_response_id`,
   `conversation`, `background`, `prompt` are `400 unsupported_feature`.
2. Output: a `reasoning` item (`content: [{type: "reasoning_text"}]`,
   `summary: []`, no `encrypted_content`), a `message` item
   (`output_text`), `function_call` items (`call_id`, `arguments`);
   `status` `completed` / `incomplete` (`max_output_tokens`) / `failed`;
   `usage` with `input_tokens_details.cached_tokens`.
3. Streamed: `response.created`, `response.in_progress`, per item
   `output_item.added` / deltas (`reasoning_text.delta`,
   `output_text.delta`) / `function_call_arguments.done` /
   `output_item.done`, then `response.completed` or
   `response.incomplete`, each with `sequence_number`.
4. What pi sends and needs (`openai-responses.ts`,
   `openai-responses-shared.ts`): `store: false`, `prompt_cache_key`,
   `reasoning: {effort, summary: "auto"}`, `include:
   ["reasoning.encrypted_content"]`, `session_id` headers (all accepted and
   ignored); replayed items carry `id`, `status` and `phase`, and a
   reasoning item comes back verbatim as nuclis sent it (so a `reasoning`
   input item without `encrypted_content` is the normal case). It routes
   deltas by `output_index`, needs an `output_item.done` for every
   function call, and a terminal event on every stream.
5. Acceptance: pi with `api: "openai-responses"` on the same task;
   `make api-check` extended.
