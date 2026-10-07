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
transport merged as #7, Chat Completions as #8; streaming, keepalives,
and cancellation are done on branch `chat-completions-stream` (its pull
request awaiting review), with pi 1.1.0 as the accepted client. Next:
**Responses, stateless**, on a branch cut from `main` once it merges.
Measured and left open (docs/guide/api.md § Conversations and the
cache, docs/spec.md §10): a decision waits behind a running generation,
136.5 s behind 2,000 Qwen3.8 tokens.

| Unit | Branch | What |
| --- | --- | --- |
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
