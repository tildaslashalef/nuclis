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

The plan below was agreed on 2026-10-07; the completer and streaming
transport (#7), Chat Completions (#8), and their streaming with pi as the
accepted client (#9) merged. In progress: **decisions beside a
generation**, on branch `decisions-beside-generation` (agreed 2026-10-08,
before the Responses API, which waits on a concrete Responses-only
client).

| Unit | Branch | What |
| --- | --- | --- |
| Decisions beside a generation | `decisions-beside-generation` | one memory budget over both kinds of model, evicting the least recently used; queued decisions run between a generation's steps |
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

## Decisions beside a generation

Base: `31114b3`

**Why.** One executor serves both kinds of model, and a chat item holds
the worker for its whole prefill and decode: a `laya-multilingual`
decision sent 5 s into a 2,000-token Qwen3.8 generation was answered
after 136.5 s, and one behind a long prefill would reach `serve.timeout`
and fail `529 timeout`. And nothing bounds the two kinds together: one
language model and two decision models may be resident at once (about
34 GB in the worst catalogue case), with no limit against the machine.
Decisions (user, 2026-10-08): a shared budget with eviction across kinds,
default physical memory minus 16 GiB; decisions interleaved with a
generation; no engine change.

1. **The budget** (`src/api/memory.zig`, pure, tested with fake models).
   `Budget{ limit }` holds the resident models: kind, name, bytes, last
   use, pinned. `reserve(bytes, owner)` closes the least recently used
   unpinned others (through each entry's `close` callback, on the worker)
   until `bytes` fit; `ModelTooLarge` when `bytes` exceed the limit alone,
   `Pinned` when only pinned models stand in the way. `add`, `resize`,
   `touch`, `remove`, `pin`/`unpin`; a snapshot for `/v1/health` under a
   lock (the rest is worker-only).
2. **Footprints.** A decision model: the regular files of its checkpoint
   directory plus its backbone and projector (`Location`), reserved before
   `Decider.open` in `Pool.acquire`; the pool's two slots stay, and an
   evicted slot is closed through the budget. A language model: before
   `Open.init`, its weights file, its draft file when speculative decoding
   is on, and `cache.memory_bytes` (the states it may keep); after,
   `resize` to the same plus `Session.bytes()` (the KV cache), evicting
   more if the estimate was short; plus the projector's file when
   `loadVision` runs. `Language.close` removes it.
3. **The limit.** `serve.memory_bytes` (`config.zig` `Serve`, null = auto:
   physical memory, `hw.memsize`, minus 16 GiB, at least 4 GiB); flag
   `--memory <GiB>`. A model alone over the limit is refused: `400
   model_too_large` (chat), `422 model_too_large` (decisions), naming the
   bytes and the limit. `/v1/health` gains `memory {limit, resident,
   models[{name, kind, bytes}]}`.
4. **Decisions between steps.** `gpu.Item` gains `short: bool` (the
   decision batcher's drain item and the decision preload set it).
   `Executor.runShort(io)`, called only on the worker from inside a
   running item, unlinks the short items queued at that moment and runs
   each once (`.again` requeues at the tail; `done` finishes it as `run`
   does). The chat job pins its language model for its run and calls
   `runShort` from its `Observer.progress` (after each generated token,
   after each prefill chunk). Each model has its own Metal backend and
   queue, and decode reports progress after sampling, so a decision pass
   there touches nothing the generation holds. A batch whose model would
   need the pinned language model evicted gets `Pinned` from the pool and
   puts its jobs back (`.again`): it runs after the generation, as today.
5. **Documents.** `docs/guide/api.md` (§ Running the server: the budget
   and `--memory`; § Conversations and the cache: the decision wait
   replaced by the measurement; Errors; `GET /v1/health`),
   `docs/guide/configuration.md` (`serve.memory_bytes`), `docs/spec.md`
   §6 and §10, `nuclis serve --help`, `docs/architecture.md` § The API
   layer.

**Gates.** `zig build test` (the budget's eviction order, pins,
too-large, resize; `runShort` on queued short items only, `.again`
requeued), `make verify-auto`, `make api-check`. Measured with the built
binary and written into the guide: the decision sent 5 s into a
2,000-token Qwen3.8 generation (target: answered within 0.3 s; the
generation's slowdown); `--memory 20` with Qwen3.8 open, then a
clef-flash decision (evicts Qwen3.8 when the generation is not running;
waits for it when it is); a model over the limit refused.

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
