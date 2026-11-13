<!-- kind: guide -->
# Chat Completions

The chat route of [the nuclis API](api.md), which covers running the
server, its memory budget, its limits, and its errors.

## `POST /v1/chat/completions`

OpenAI's Chat Completions for the language models `nuclis agent` runs. A
program written against an OpenAI SDK, or an agent configured for an
OpenAI-compatible server, works with the base URL
`http://127.0.0.1:9000/v1` and any API key.

```sh
curl -s localhost:9000/v1/chat/completions -d '{
  "model": "qwen3.8-27b",
  "messages": [{"role": "user", "content": "Name a prime number above 50."}],
  "reasoning_effort": "none"
}'
```

```json
{
  "id": "chatcmpl-iteSmCuNOu0Uuhmq6YCwPU8k",
  "object": "chat.completion",
  "created": 1791411276,
  "model": "qwen3.8-27b",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "A prime number above 50 is **53**.\n\nOther examples include 59, 61, 67, and 71."},
    "logprobs": null,
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 21, "completion_tokens": 34, "total_tokens": 55,
    "prompt_tokens_details": {"cached_tokens": 0},
    "completion_tokens_details": {"reasoning_tokens": 0}
  }
}
```

One language model is open at a time, beside the decision models. A
request runs with the settings `nuclis agent` resolves for its model
(context window, attention cache precision, speculative decoding, the
thinking budget `agent.thinking_budget`), and one request runs on the
GPU at a time, decisions included.

## The request

| Field | In nuclis |
| --- | --- |
| `model` | a registry entry or catalogue name of a language model, never a path. Absent: the open model, else `engine.model`. Another model closes the open one and opens itself; that request pays the load |
| `messages` | roles `system`, `developer`, `user`, `assistant`, `tool`. `content` is a string, null (an assistant turn of calls only), or parts: `text`, and on user messages `image_url` with a base64 data URL (at most 8 images, 16 MiB each) and `input_audio` (below) |
| `input_audio` parts | `{"type": "input_audio", "input_audio": {"data": "<base64>", "format": "wav"}}`: `data` base64 or a base64 data URL, `format` one of `wav`, `mp3`, `aiff`, `flac`, `m4a` (the decoder reads the bytes; the name is checked, not trusted). At most 8 clips per request, 16 MiB and **30 s** each (750 rows; a longer clip is `400 input_too_long`, never cut). Only Gemma 4 E4B hears them (its projector pulled); another model answers `400 audio_unsupported`, a clip that does not decode `400 invalid_audio`. The model reads a message's images, then its clips, then its text ([gemma4.md § Audio input](../models/gemma4.md#audio-input-on-the-e4b)) |
| assistant `reasoning_content` | the turn's reasoning, given back as it was returned (`reasoning` is read too); the model's template keeps or drops it |
| assistant `tool_calls`, tool `tool_call_id` | the calls and their results; a result must answer an earlier call |
| `tools` | `type: "function"` with `name`, `description`, `parameters` (a JSON Schema); at most 64. The model's own template renders them. With `serve.web`, also `{"type": "web_search"}`, `{"type": "web_fetch"}`, and `{"type": "web_context"}`, which the server runs ([hosted web tools](#hosted-web-tools)) |
| `tool_choice` | `auto` (the default) or `none`, which renders no tools |
| `reasoning_effort` | `none` (no reasoning), `minimal`, `low`, `medium`, `high`, `xhigh` (`max` reads as `xhigh`); the model's nearest level. `minimal` is `low` with its reasoning closed at half the budget (512 tokens at the default `agent.thinking_budget`, [below](#a-short-reasoning-minimal)). Absent: `agent.think` |
| `max_completion_tokens`, `max_tokens` | the output budget, reasoning included; absent: the model's `max_tokens` setting. Larger values are capped at 16,384 and at what the window leaves after the prompt |
| `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `repetition_penalty` | each overrides one option of the model's sampling for the effort; `temperature: 0` is greedy |
| `seed` | the sampler's seed for this request |
| `nuclis_web_text` | `true`: each finished entry of `nuclis_web_calls` adds what the model read ([hosted web tools](#hosted-web-tools)) |
| `stream`, `stream_options.include_usage` | server-sent events instead of one body ([streaming](#streaming)); with `include_usage`, a last chunk carries the usage |

Accepted and ignored, because they change nothing here: `user`,
`store`, `metadata`, `prompt_cache_key`, `service_tier`,
`parallel_tool_calls`, and `frequency_penalty: 0`.

### A short reasoning: `minimal`

`minimal` renders as `low` and closes the reasoning at half of `low`'s
budget, as the budget always closes it: the model then answers or calls
a tool. It is for a retry that needs some reasoning but not a long one,
such as Muon's: with `none`, Gemma 4 26B-A4B answered "I've added that
to your tasks" without calling the tool, and `low` fixes that slowly.
The half budget was chosen on Muon's requests: [record, 2026-11-01](../benchmarks/record/api.md#a-short-reasoning-minimal-2026-11-01).

Closed early, the model mostly still makes the call it had settled on;
the shorter the budget, the more often it writes the spoken answer
instead. A switch between `none` and `low` (or `minimal`) misses the
cache once: Gemma 4's template marks thinking at the start of the
system turn, so the two prompts differ from the first token; after
that, each effort restores its own system block.

## What is refused

`400 unsupported_feature`, with `param` naming the field, for what would
change the result and cannot be honoured:

| Field | Why |
| --- | --- |
| `response_format` other than `text`, `tool_choice: "required"` or a named function | they need the sampler to mask tokens against a grammar, which the engine does not have |
| `n` above 1, `logprobs`, `top_logprobs` | one choice per request; log probabilities are not returned |
| `stop`, `logit_bias`, `frequency_penalty` other than 0 | not implemented; the model's own end of turn stops it |
| `audio`, `modalities` beyond `text`, `prediction`, part types other than `text`, `image_url`, and `input_audio`, an audio `format` outside the five above | text out; text, images, and audio in |
| an `image_url` that is not a data URL | the server fetches no URL |
| a `web_search`, `web_fetch`, or `web_context` tool type without `serve.web`; a function of the same name beside one (`invalid_request`) | the server runs no web tools unless configured to; a clash would make its calls ambiguous |

## The response

`message.content` is the answer, null when the turn is only calls;
`message.reasoning_content` the reasoning, when there was any;
`message.tool_calls` the calls, each with a fresh `call_` id, its
`arguments` a JSON object as text. `finish_reason` is `tool_calls` when
the turn made calls, `length` when it reached the output budget or the
window, `no_answer` when [hosted calls](#hosted-web-tools) ran and the model
ended its turn without a word, `stop` otherwise. `usage.prompt_tokens` is the whole rendered
conversation, `completion_tokens` everything generated (reasoning
included, `reasoning_tokens` of it).

## Streaming

With `"stream": true` the answer is `text/event-stream`: one `data:` line
per `chat.completion.chunk`, then `data: [DONE]`. The first chunk carries
`delta.role`, then each piece of reasoning is a `delta.reasoning_content`
and each piece of the answer a `delta.content`, as the model writes them
(a character split across tokens is held until it is whole). Each call is
one chunk with `delta.tool_calls[{index, id, type, function{name,
arguments}}]`, the arguments whole: the model's call syntax is decoded
only once it is complete. The last choice chunk carries the
`finish_reason`; with `include_usage`, a chunk with no choices carries the
usage. A hosted web call is reported as it starts and as it finishes
([hosted web tools](#hosted-web-tools)). An error after the stream began (the model fails, or the request
waited `serve.timeout` without reaching the GPU) is one
`data: {"error": {…}}` line, and the stream ends.

The status and headers are sent at once, before the request waits for
the GPU, and a stream that has been silent for 10 s gets an SSE comment
saying where the request stands (`: queued`, `: prefill 4096/6635`,
`: generating 812`); clients ignore comments, and their idle timeouts
(300 s for pi, 600 s for OpenAI's SDKs) do not fire during a long
prefill. A 6,635-token prompt on Qwen3.8 prefilled for 85 s with a
comment every 10 s.

A client that closes the connection stops its request, streamed or not:
the server notices within a second, the model stops at its next step,
and the log line says `cancelled`. A request still waiting for the GPU is
taken out of the queue. Cancelling a streamed request after 22 tokens and
a whole one after 3 s (its client's timeout) both stopped within the
second.

## Conversations and the cache

A client sends the whole conversation every request; nothing is stored
between requests, and no response id carries state. The model's state is
reused anyway: the server continues the session it holds when the new
conversation extends what it last consumed, otherwise restores the
longest earlier state it kept at the end of an answer or of a system
prompt of 64 tokens or more (in memory, within `cache.memory_bytes`,
4 GiB), and prefills only the rest.
`usage.prompt_tokens_details.cached_tokens` says how much was reused. On
Qwen3.8, three requests of one conversation reported 0, 37 of 55, and 65
of 86 tokens cached, and a tool round trip reused 360 of 390. Gemma 4's
template rewrites the model's past turns (an empty thought block with
reasoning off on the 12B and 26B-A4B, reasoning dropped once a user
message follows), so there the session goes back to where the last
prompt ended and continues from it: the answer as rendered and the new
message are prefilled. A request sent again reuses the same point on
Gemma 4 and Muse Glimmer. Where the last prompt ended in a tool result,
that point is the whole prompt: the request sent again re-runs its last
token for the next one's logits. A
model with recurrent state (Qwen3.8) cannot move back a token, so there
the last turn's prompt is prefilled again from the longest shorter state.
Images and clips count by their bytes: a
conversation resent with the same ones reuses its state, another image or
clip in the same place does not (measured on the E4B with a 4.8 s clip:
133 of 136 tokens cached for the same request again, 152 of 171 for a
follow-up, 0 for another clip). Changing the model empties the cache.

What the cache saves per turn: [record, 2026-10-08](../benchmarks/record/api.md#conversations-and-the-cache-2026-10-08).

A decision does not wait for a generation that is running: queued
decisions run between its steps (after each generated token, after each
prefill chunk), each model on its own Metal queue. Five
`laya-multilingual` decisions sent during a 2,000-token Qwen3.8
generation were answered in 51 to 164 ms, and the
generation took 139.5 s (141.4 s without them). During a long prefill a
decision waits for the current chunk, 0.45 s on Qwen3.8. Answers are the
same as on an idle server, logits included. A decision whose model does
not fit beside the generation in memory waits for it to end
([memory](api.md#memory)).

## Agents

An agent built for OpenAI-compatible servers connects as a custom
provider. For pi, `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "nuclis": {
      "baseUrl": "http://127.0.0.1:9000/v1",
      "api": "openai-completions",
      "apiKey": "nuclis",
      "models": [{
        "id": "qwen3.8-27b",
        "reasoning": true,
        "thinkingLevelMap": {"off": "none", "xhigh": "xhigh"},
        "input": ["text", "image"],
        "contextWindow": 16384,
        "maxTokens": 8192
      }]
    }
  }
}
```

pi wants an API key and accepts any. `reasoning: true` makes it send
`reasoning_effort` and give the reasoning back each turn; the
`thinkingLevelMap` sends `none` when thinking is off (otherwise nothing is
sent and `agent.think` applies) and offers `xhigh`. `contextWindow` is the
model's `ctx_size`. Its defaults for an unknown server fit nuclis as they
are (the `developer` role, `max_completion_tokens`, streamed usage).

## Hosted web tools

With `serve.web` on ([configuration](configuration.md#web)), a request may
list `{"type": "web_search"}`, `{"type": "web_fetch"}`, and
`{"type": "web_context"}` among its tools.
They render through the model's template like any tool, with the
definitions the agent uses, and the server runs their calls itself through
the search service `web.url` names.

The command line decides without the file: `nuclis serve --web-url <url>`
turns the tools on against that service (the same rule as `web.url`:
`https`, or `http` on a loopback host; no credentials, query, or fragment),
whatever `serve.web` says, and `--no-web` turns them off; the two exclude
each other. An app that starts the server (Muon) passes the address this
way rather than rewriting `nuclis.json`.

```json
{"messages": [{"role": "user", "content": "What does PEP 8 say about line length?"}],
 "tools": [{"type": "web_search"}, {"type": "web_fetch"}]}
```

- The three tools: `web_search` (`query`, optional `freshness`) returns at
  most 8 results, a title, URL, and snippet each, with up to two more
  snippets under the first three; `web_fetch` (`url`, optional `offset`)
  reads one page as Markdown, 16,000 characters at a time; `web_context`
  (`query`, optional `freshness`) returns passages from up to 8 pages that
  answer the question, about 2,000 tokens, each under its page's title and
  URL. `freshness` is `day`, `week`, `month`, or `year`.
- `web_context` needs a search service with Brave. The server accepts the
  type either way and its start line names the tools the service offered
  when it started; on a service without it, a call answers "web_context is
  unavailable on the search service now; use web_search" as its result, and
  the model goes on.
- Searches are metered upstream (Brave), so one request makes at most six
  uncached `web_search` and `web_context` calls; a seventh answers that the
  budget is spent, without reaching the service. Answers the service had
  cached, and fetches, are free.
- In a request with hosted tools, a call to a web tool the request did not
  enable (and defines no function of that name) is not returned either:
  `web_context`'s description names the other two, so the model may call
  one, and the server answers it as a failed result, "not a tool in this
  request", like the agent's "not a tool I have".
- A completion whose calls are all hosted is not returned: the server runs
  them, appends the assistant message and one tool message per result, and
  completes again, up to six completions per request. The sixth runs without
  the web tools, so it answers from what it has.
- When a completion also calls one of the request's functions, its hosted
  calls run and the function calls are returned as usual
  (`finish_reason: "tool_calls"`).
- The client never sees the hosted calls and results as messages, so the
  conversation it sends next does not carry them. The server keeps them:
  when a later request resends the assistant message a request returned (its
  text and function calls; reasoning and call ids may differ, outer
  whitespace too) after the same conversation, the hosted rounds are put back
  in front of it, so the model reads the results again and the prompt
  matches what it processed, which keeps its cached state usable. Kept are
  the last 64 such requests, up to 8 MiB, for an hour; a restart, an edited
  message, or a conversation that no longer fits the window with them gets
  the client's conversation as sent.
- Every hosted call is reported in `message.nuclis_web_calls`, an extension
  OpenAI clients ignore: `name`, `arguments`, `summary` (`3 results`,
  `characters 0 to 642 of 642`, or the failure's sentence), and `is_error`.
  Streamed, each call is two chunks whose `delta` carries
  `nuclis_web_calls` with the call's `index` and a `status`: `started`
  as the server is about to run it (its `name` and `arguments`, no
  outcome yet), then `finished` with the `summary` and `is_error`. A
  client can say it is searching from the first (a search takes a second
  or more); one that reads only finished entries skips `started`, and an
  entry without `status` (not streamed) is finished. With
  `"nuclis_web_text": true` on the request, each finished entry adds
  `text`: what the model read, its titles, URLs, published dates, and
  snippets as rendered, or the page. It is for debugging an answer; the
  request log never holds it.

  ```json
  {"delta": {"nuclis_web_calls": [{"index": 0, "status": "started", "name": "web_search", "arguments": "{\"query\":\"PEP 8 line length\"}"}]}}
  {"delta": {"nuclis_web_calls": [{"index": 0, "status": "finished", "name": "web_search", "arguments": "{\"query\":\"PEP 8 line length\"}", "summary": "8 results", "is_error": false}]}}
  ```
- A round after hosted calls that ends the turn with no text and no call
  runs once more on the same prompt (a `seed` moves on by one), which
  reuses the conversation before the turn and reads the results again.
  Still empty, the `finish_reason` is `no_answer`. A token the vocabulary marks as anything
  but text never reaches `content` or `reasoning_content`, wherever the
  model writes it; the exception is a Qwen3.8 call that does not parse,
  released as the model wrote it.
- The usage counts every completion; the log line adds `2 web calls in 3
  rounds`, `1 web exchange put back` when earlier hosted rounds were
  restored, and `empty answer retried`. The model's state carries over between completions, so a round
  reuses the conversation before it ([record, 2026-10-16](../benchmarks/record/api.md#hosted-rounds-reuse-the-state-2026-10-16)).
- Text the model writes before a call (Muon's "Let me check.") is rendered
  by Gemma 4's template after the call's results, where the next round's
  prompt closes the turn and opens another; history keeps the answer in
  that turn. The state is kept where the two still agree, after the
  results, so the turn after a search prefills only the answer and the new
  message. [Record, 2026-10-28](../benchmarks/record/api.md#text-before-a-call-2026-10-28).
- A web call waits on the GPU worker; decision and embedding items queued
  meanwhile run during its waits. A client that leaves cancels it.
- Without `serve.web` (or `--web-url`), and under `--no-web`, any of the
  three types is `400 unsupported_feature` naming `tools[i]`.
