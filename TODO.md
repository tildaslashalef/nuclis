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

The plan below was agreed on 2026-10-07. The completer and the streaming
transport merged as #7, Chat Completions without streaming as #8. In
progress: **Chat Completions, streamed**, on branch
`chat-completions-stream`.

| Unit | Branch | What |
| --- | --- | --- |
| Chat Completions, streamed | `chat-completions-stream` | `stream: true`, cancellation by disconnect, keepalives, pi as the acceptance client |
| Responses, stateless | `responses-stateless` | `POST /v1/responses` with `store: false`, over the same core |

## The theme: an OpenAI-compatible service for the language models

Shared by the units below. What is served is written in
[docs/guide/api.md § Chat Completions](docs/guide/api.md#chat-completions)
and [docs/spec.md](docs/spec.md) §6 `serve`; each unit updates them.

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

## Chat Completions, streamed

Base: `5abd749`

**Why.** Every agent streams; pi always sends `stream: true`. This unit
delivers the same events as server-sent events and makes a long request
safe to abandon.

It builds on `src/api/chat/`: `wire.parse` stops refusing `stream`
(reads `stream_options.include_usage`), `Language.run` takes the sink it
writes through instead of a `wire.Collector`, and the `Job` carries the
`Pipe` and a cancel flag its `Observer` reads (`Completer.observer`).

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
