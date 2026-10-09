# The nuclis API

`nuclis serve` runs the nuclis API: an HTTP/1.1 server on this machine
with three services. **Decisions** keep decision models open and answer
typed questions about text or JSON, in TypeSafe's Jev protocol, so a
client written for `api.typesafe.ai` switches by changing its base URL.
**Chat Completions** serve the language models in OpenAI's format, so a
client written for an OpenAI-compatible server does the same.
**Embeddings** turn text, images and audio into vectors in OpenAI's embeddings format. This page
is the reference for client authors; it assumes nothing about the code.

- [Running the server](#running-the-server), [memory](#memory)
- [Conventions](#conventions)
- [Errors](#errors)
- [Decisions](#decisions): [`POST /v1/systemone`](#post-v1systemone),
  [`POST /v1/decisions`](#post-v1decisions), [questions and
  answers](#questions-and-answers), [batching](#batching)
- [Chat Completions](#chat-completions): [the request](#the-request),
  [what is refused](#what-is-refused), [the response](#the-response),
  [streaming](#streaming), [conversations and the
  cache](#conversations-and-the-cache), [agents](#agents)
- [Embeddings](#embeddings): [`POST /v1/embeddings`](#post-v1embeddings),
  [the request](#the-embeddings-request), [the
  response](#the-embeddings-response), [passes and
  memory](#passes-and-memory)
- [`GET /v1/models`](#get-v1models), [`GET /v1/health`](#get-v1health)
- [Measured rates](#measured-rates)

## Running the server

```sh
nuclis model pull laya                 # once: the default decision model
nuclis serve                           # http://127.0.0.1:9000/v1, decide.model open
```

| Option | Meaning |
| --- | --- |
| `--host <ip>` | the address to listen on: an IP literal (`127.0.0.1`, `::1`, `0.0.0.0`) or `localhost`; default `serve.host`, `127.0.0.1`. Any address beyond loopback prints a warning: there is no authentication. |
| `--port <n>` | default `serve.port`, `9000` |
| `--model <name>` | a decision model opened before the server listens (at most 2); without one, `decide.model` opens; others open on their first request |
| `--chat-model <name>` | a language model opened before the server listens; without one, the first chat request opens one |
| `--memory <GiB>` | what every open model may hold together; default `serve.memory_bytes`, else physical memory less 16 GiB ([memory](#memory)) |
| `--backend cpu\|metal` | where the models run; default `metal` in a Metal build |
| `--quiet` | no request log (`serve.log: false` does the same) |
| `--timeout <s>` | how long a request may wait for the GPU before `529 timeout`; default `serve.timeout`, `300` |

The configuration file (`~/.nuclis/nuclis.json`) holds the defaults:

```json
"decide": { "model": "laya" },
"serve":  { "host": "127.0.0.1", "port": 9000, "log": true, "timeout": 300 }
```

`nuclis config set serve.port 9000` changes one; the flags override them
for a run.

A model is named as `nuclis decide --model` names it: a registry entry of
kind `decision` in `~/.nuclis/nuclis.json`, a decision catalogue name
(`laya`, `laya-multilingual`, `clef-flash`), or a checkpoint directory. A
request that names none gets the configuration's `decide.model` (default
`laya`). The server keeps 2 models open and closes the least recently used
when a third is asked for; opening takes 0.07 s (`laya`) to 0.15 s
(`laya-multilingual`) on Metal, and 0.5 s for `clef-flash`, paid by the
request that asks. Each model validates questions by its own rules:
`clef-flash` takes questions without `instructions`, Laya does not.

### Memory

Every open model counts against one budget, whatever its kind:
32 GiB on a 48 GB machine by default (physical memory less 16 GiB, for
the system and other applications). A model counts its files (a decision
checkpoint's weights, backbone, and projector; a language model's
weights, its draft file when speculative decoding is on, and its image
projector once an image arrives; an embedding model's file) plus, for a
language model, its attention cache and the saved states it may keep (`cache.memory_bytes`,
4 GiB). Qwen3.8 counts 22.0 GiB, `laya-multilingual` 0.6 GiB, clef-flash
8.4 GiB.

Before a model opens, the least recently used others close until it
fits, whichever kind they are. A language model is never closed while a
request runs on it: a decision that would need its memory waits for the
generation to end. A model larger than the whole budget is refused
(`model_too_large`). With `--memory 30` and Qwen3.8 open, an idle
clef-flash decision closed Qwen3.8 (and the older Laya) and answered in
4.2 s; the next chat request reopened Qwen3.8 in 1.2 s and closed
clef-flash; a clef-flash decision sent during a generation was answered
0.95 s after it ended. `GET /v1/health` lists what is resident.

`clef-flash` also reads images: a request's `"images"` is a list of data
URLs (`data:image/png;base64,…`) or bare base64, at most 8, each read
before every state (`422 images_unsupported` from Laya, or from
`clef-flash` without its projector pulled; `422 invalid_image` for bytes
that are not an image). The body limit (4 MiB) bounds them. Its answers
are `systemone`'s own: uncalibrated, `confidence` the top probability. A
clef decision is a 9B prefill, 2.3 s for a 637-token state
([clef-flash.md § Time per decision](../models/clef-flash.md#time-per-decision)).

Ctrl-C stops the server gracefully: it accepts no new connections,
waits up to 10 s for decisions already queued or running, closes the
connections (idle keep-alive ones included), and exits 0, so a Debug
build ends with the allocator's leak check. A second Ctrl-C ends the
process at once. The server reads no files on a client's behalf and
writes nothing but its log: one line per response on stdout, coloured on
a terminal (time, method, path, status, latency, size, and what was done
or the error):

```text
17:34:35.751 POST   /v1/systemone    200   51.6 ms    175 B  laya · 1 state × 1 question · 43 tokens
17:34:36.278 POST   /v1/decisions    200  201.8 ms   1.3 KB  laya · 1 state × 2 questions · 104 tokens · pass of 6
17:34:36.008 POST   /v1/decisions    422  0.062 ms    116 B  invalid_request: question a: unknown type "maybe"; use one of choice, noul, score
```

## Conventions

- Every route is under `/v1`. Request and response bodies are JSON
  (`content-type: application/json`), UTF-8; responses carry
  `content-length` and are indented.
- HTTP/1.1 keep-alive is supported and recommended (HTTP/1.0 clients get
  it when they send `connection: keep-alive`); requests on one
  connection are answered in order, without pipelining.
- `authorization` and other headers are accepted and ignored.
- Limits, fixed by the host:

| Limit | Value | Past it |
| --- | --- | --- |
| request line and headers | 16 KiB | `431 headers_too_large`, connection closed |
| request body | 4 MiB; 32 MiB for `/v1/chat/completions` and `/v1/embeddings` | `413 payload_too_large`, connection closed |
| open connections | 64 | `529 busy`, connection closed |
| decision requests waiting for the GPU | 64 | `529 busy` |
| embedding requests waiting for the GPU | 64 | `529 busy` |
| inputs per embedding request | 2,048 | `400 request_too_large` |
| text per embedding input | 1 MiB | `400 request_too_large` |
| wait before a request starts on the GPU | 300 s (`serve.timeout`) | `529 timeout` (a started pass always finishes) |
| states per request | 64 | `422 request_too_large` |
| questions per request | 32 | `422 request_too_large` |
| options per question | 255 | `422 request_too_large` |
| bytes per state | 1 MiB | `422 request_too_large` |

The option limit is the protocol's. A model may hold fewer: Laya fits a
question's options into its head budget (192 tokens for `laya`, 256 for
`laya-multilingual`) and refuses a question whose options do not fit with
`422 options_exceed_budget`; `clef-flash` refuses a request whose
questions alone overflow its 16,384 tokens the same way.

## Errors

Every error has the same body, with the HTTP status that fits:

```json
{"error": {"code": "model_not_found", "message": "jev-2: not pulled yet (…)", "type": "invalid_request_error", "param": "model"}}
```

`code` is stable and meant for branching; `message` is for people;
`type` (`invalid_request_error` for a 4xx, `server_error` otherwise) and
`param` (the request field at fault, or null) are the fields OpenAI's
SDKs read. The decision routes' statuses follow TypeSafe's: `422` for a
body that fails validation. The chat route answers `400` instead, which
OpenAI's SDKs raise as a bad request. Both answer `529` when the server
is overloaded (retry with backoff, as both families of SDKs do by default).

| Status | Code | When |
| --- | --- | --- |
| 400 | `bad_request` | not HTTP: a malformed request line or header, an unknown method; the connection closes |
| 404 | `not_found` | no route at that path |
| 405 | `method_not_allowed` | the path exists, the method does not (`HEAD` is answered as `GET`) |
| 413 | `payload_too_large` | the body exceeds 4 MiB (32 MiB for chat and embeddings) |
| 415 | `unsupported_encoding` | a compressed body (`content-encoding`) |
| 417 | `bad_request` | an `expect` other than `100-continue` |
| 422 | `invalid_json` | the body is not JSON |
| 422 | `invalid_request` | the JSON is not a valid request: a missing or extra field, a malformed question, `states` sent to `systemone`, a text that cannot be tokenized |
| 422 | `file_state_refused` | a state of the form `{"file": path}`; send the text |
| 422 | `request_too_large` | past a count or size limit above |
| 422 | `options_exceed_budget` | a question's options do not fit the model's budget |
| 422 | `model_not_found` | `model` names nothing that is pulled |
| 422 | `not_a_decision_model` | `model` names a language model |
| 422 | `model_too_large` | the decision model alone needs more than the memory budget ([memory](#memory)) |
| 400 | `invalid_json`, `invalid_request` | chat: the body is not JSON, or a field is malformed (`param` names it) |
| 400 | `unsupported_feature` | chat: a field nuclis cannot honour ([what is refused](#what-is-refused)) |
| 400 | `not_a_language_model` | chat: `model` names a decision or embedding model |
| 400 | `images_unsupported` | chat: an image for a model whose projector is not pulled |
| 400 | `context_length_exceeded` | chat: the rendered conversation does not fit the model's window |
| 404 | `model_not_found` | chat: `model` names no pulled language model |
| 400 | `model_too_large` | chat: the language model alone needs more than the memory budget, with the sizes |
| 400 | `invalid_json`, `invalid_request` | embeddings: the body is not JSON, or a field is malformed (`param` names it) |
| 400 | `unsupported_feature` | embeddings: token arrays, an image by URL, an audio format nuclis does not name, media without the projector pulled, a width other than 768, 512, 256, or 128 ([what is refused](#what-an-embedding-request-may-not-carry)) |
| 400 | `invalid_image`, `invalid_audio` | embeddings: an image or audio part that does not decode (or a clip under 10 ms); the message names the input |
| 400 | `input_too_long` | embeddings: an input over 8,192 tokens, or a clip over 30 s, without `"truncate": true`; the message gives its index and count |
| 400 | `request_too_large` | embeddings: over 2,048 inputs, or over 1 MiB of text in one |
| 400 | `not_an_embedding_model` | embeddings: `model` names a language or decision model |
| 404 | `model_not_found` | embeddings: `model` names no pulled embedding model |
| 400 | `model_too_large` | embeddings: the model alone needs more than the memory budget |
| 431 | `headers_too_large` | the request line and headers exceed 16 KiB |
| 500 | `model_failed` | the model could not be opened, or a completion failed on the GPU |
| 500 | `internal`, `invalid_registry_entry` | a fault on the server's side |
| 503 | `shutting_down` | the server is stopping |
| 529 | `busy` | 64 requests wait already, or 64 connections are open |
| 529 | `timeout` | the request waited `serve.timeout` (300 s) without reaching the GPU |

## Decisions

A decision model answers typed questions about a **state** (text, or any
JSON: an object, a record, a conversation as a list). There are three
question types: `noul` (yes or no, as a probability), `choice` (one
option of several), and `score` (a value along ordered levels). Each
question is answered on its own; nothing is generated.

### `POST /v1/systemone`

TypeSafe's Jev call: one state, Jev's answers. A Jev client needs no
change beyond the base URL.

```sh
curl -s localhost:9000/v1/systemone -d '{
  "model": "jev-latest",
  "state": "Help! My payouts have been failing for 3 days.",
  "questions": {
    "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]}
  }
}'
```

```json
{
  "model": "laya",
  "answers": {
    "is_urgent": {"type": "noul", "noul": 0.7419},
    "frustration": {
      "type": "score",
      "score": 1.0889,
      "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
      "probabilities": {"0": 0.0317, "1": 0.8477, "2": 0.1206},
      "confidence": 0.5407
    }
  },
  "usage": {"input_tokens": 87, "output_tokens": 0}
}
```

Request fields:

| Field | Type | |
| --- | --- | --- |
| `state` | string, object, or array | required; what the questions are about |
| `questions` | object: id → question | required; at least one ([questions](#questions-and-answers)) |
| `model` | string | optional; a nuclis decision model, or a TypeSafe id |

A TypeSafe model id (`jev-latest`, `jev-1.13.0`, any `jev-…`) means the
server's default decision model, unless the registry has an entry by that
exact name. The response's `model` is the nuclis model that answered.

Response fields: `model`, `answers` (one per question id, exactly Jev's
fields, nothing else), and `usage`: `input_tokens` is the tokens the
model read over every question's sequence, `output_tokens` is always 0
(nothing is generated).

### `POST /v1/decisions`

The nuclis call: one state or many, and everything the model computed.
The body is what `nuclis decide --request` reads; the response is
byte for byte what `nuclis decide --json` writes for it, timings aside.

```sh
curl -s 'localhost:9000/v1/decisions?explain=1' -d '{
  "model": "laya-multilingual",
  "questions": {
    "team": {"type": "choice", "instructions": "Which team should handle this?",
             "criteria": {"billing": "invoices, refunds", "technical": "bugs, outages"}}
  },
  "states": ["I was charged twice!", {"customer": "acme", "error": "500 on checkout"}]
}'
```

```json
{
  "schema_version": 1,
  "model": "laya-multilingual",
  "repo": "convaiinnovations/laya",
  "revision": "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
  "timings_ms": {"load": 0, "tokenize": 0.2, "encode": 43.6},
  "results": [
    {
      "answers": {
        "team": {
          "type": "choice",
          "choice": "billing",
          "probabilities": {"billing": 0.7182, "technical": 0.2818},
          "confidence": 0.1421,
          "nuclis": {
            "answer_confidence": 0.7182,
            "logits": [-0.4003656506538391, -1.3360267877578735],
            "temperature": 1,
            "bucket": "choice:2",
            "sequence_tokens": 30,
            "state_kept": 5
          }
        }
      },
      "usage": {"input_tokens": 30, "output_tokens": 0},
      "nuclis": {"state": "state[0]", "state_tokens": 5, "truncated": false}
    }
  ]
}
```

(the second state's result is omitted; the server indents every line)

Request fields: `questions` as above; exactly one of `state` (one) or
`states` (a non-empty list, at most 64); `model` as above. Query:
`?explain=1` adds `sequence_tokens` and `state_kept` to each answer's
`nuclis` object.

Response fields:

| Field | |
| --- | --- |
| `schema_version` | 1; a breaking change to this body raises it |
| `model`, `repo`, `revision` | the model as the request named it, and the Hub repository and commit its weights came from (null for a local directory) |
| `timings_ms.load` | opening the model for this request; 0 when it was open |
| `timings_ms.tokenize` | building this request's sequences |
| `timings_ms.encode` | the GPU pass this request ran in; in a [batch](#batching), the whole batch's |
| `results` | one per state, in order: a complete Jev response (`answers`, `usage`) plus `nuclis` |
| `results[].nuclis` | `state` (a label: `state[i]`), `state_tokens` (the state's length), `truncated` (whether a question's sequence cut the state) |
| `answers.<id>` | Jev's fields, then `nuclis`: `answer_confidence` (the top probability), `logits` (one per option, in order), `temperature` (applied before the softmax), `bucket` (the temperature's bucket) |

### Questions and answers

Each question is `{"type", "instructions", "criteria"}`; `instructions`
and every criterion may be a string, an object, or an array (structured
values are rendered as JSON text for the model).

| Type | `criteria` | Answer fields |
| --- | --- | --- |
| `noul` | optional `{"true": …, "false": …}`, what a yes and a no mean | `type`, `noul`: P(yes), 0 to 1 |
| `choice` | an object, option → description (or `null`), or a list of option names | `type`, `choice` (the most probable option), `probabilities` (option → p, summing to 1), `confidence` |
| `score` | a list of level descriptions, level 0 first | `type`, `score` (Σ level × p, may fall between levels), `legend` (level → description), `probabilities` (level → p), `confidence` |

`confidence` is 1 − H(p)/ln k, the distribution's normalized entropy
taken from 1: 1 for a certain answer, 0 for a uniform one. Numbers are
rounded to 4 places. A state longer than the model's sequence (512 tokens
for `laya`, 1,024 for `laya-multilingual`, the question and options
included) is cut, a list state at its start and any other at its end,
and flagged in `truncated`.

### Batching

Requests run on the GPU one pass at a time. Decision requests that wait
together share a pass: when the GPU frees, the server takes the oldest
waiting request's model and every request waiting for that model, in
arrival order, as long as their sequences fit one pass (2,048 tokens; a
request is never split, and the first always runs). There is no waiting
window, so a lone request starts at once. Requests for another model wait
at most one pass. A sequence's answer does not depend on what shares its
pass: batching changes timing, never answers.

## Chat Completions

### `POST /v1/chat/completions`

OpenAI's Chat Completions for the language models `nuclis chat` runs. A
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
request runs with the settings `nuclis chat` resolves for its model
(context window, attention cache precision, speculative decoding, the
thinking budget `agent.thinking_budget`), and one request runs on the
GPU at a time, decisions included.

### The request

| Field | In nuclis |
| --- | --- |
| `model` | a registry entry or catalogue name of a language model, never a path. Absent: the open model, else `engine.model`. Another model closes the open one and opens itself; that request pays the load |
| `messages` | roles `system`, `developer`, `user`, `assistant`, `tool`. `content` is a string, null (an assistant turn of calls only), or parts: `text`, and on user messages `image_url` with a base64 data URL (at most 8 images, 16 MiB each) |
| assistant `reasoning_content` | the turn's reasoning, given back as it was returned (`reasoning` is read too); the model's template keeps or drops it |
| assistant `tool_calls`, tool `tool_call_id` | the calls and their results; a result must answer an earlier call |
| `tools` | `type: "function"` with `name`, `description`, `parameters` (a JSON Schema); at most 64. The model's own template renders them |
| `tool_choice` | `auto` (the default) or `none`, which renders no tools |
| `reasoning_effort` | `none` (no reasoning), `minimal` and `low`, `medium`, `high`, `xhigh` (`max` reads as `xhigh`); the model's nearest level. Absent: `agent.think` |
| `max_completion_tokens`, `max_tokens` | the output budget, reasoning included; absent: the model's `max_tokens` setting. Larger values are capped at 16,384 and at what the window leaves after the prompt |
| `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `repetition_penalty` | each overrides one option of the model's sampling for the effort; `temperature: 0` is greedy |
| `seed` | the sampler's seed for this request |
| `stream`, `stream_options.include_usage` | server-sent events instead of one body ([streaming](#streaming)); with `include_usage`, a last chunk carries the usage |

Accepted and ignored, because they change nothing here: `user`,
`store`, `metadata`, `prompt_cache_key`, `service_tier`,
`parallel_tool_calls`, and `frequency_penalty: 0`.

### What is refused

`400 unsupported_feature`, with `param` naming the field, for what would
change the result and cannot be honoured:

| Field | Why |
| --- | --- |
| `response_format` other than `text`, `tool_choice: "required"` or a named function | they need the sampler to mask tokens against a grammar, which the engine does not have |
| `n` above 1, `logprobs`, `top_logprobs` | one choice per request; log probabilities are not returned |
| `stop`, `logit_bias`, `frequency_penalty` other than 0 | not implemented; the model's own end of turn stops it |
| `audio`, `modalities` beyond `text`, `prediction`, part types other than `text` and `image_url` | text out, text and images in |
| an `image_url` that is not a data URL | the server fetches no URL |

### The response

`message.content` is the answer, null when the turn is only calls;
`message.reasoning_content` the reasoning, when there was any;
`message.tool_calls` the calls, each with a fresh `call_` id, its
`arguments` a JSON object as text. `finish_reason` is `tool_calls` when
the turn made calls, `length` when it reached the output budget or the
window, `stop` otherwise. `usage.prompt_tokens` is the whole rendered
conversation, `completion_tokens` everything generated (reasoning
included, `reasoning_tokens` of it).

### Streaming

With `"stream": true` the answer is `text/event-stream`: one `data:` line
per `chat.completion.chunk`, then `data: [DONE]`. The first chunk carries
`delta.role`, then each piece of reasoning is a `delta.reasoning_content`
and each piece of the answer a `delta.content`, as the model writes them
(a character split across tokens is held until it is whole). Each call is
one chunk with `delta.tool_calls[{index, id, type, function{name,
arguments}}]`, the arguments whole: the model's call syntax is decoded
only once it is complete. The last choice chunk carries the
`finish_reason`; with `include_usage`, a chunk with no choices carries the
usage. An error after the stream began (the model fails, or the request
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

### Conversations and the cache

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
Gemma 4 and Muse Glimmer. A conversation with an image in it is
prefilled whole on every request, and changing the model empties the
cache.

Measured with `nuclis serve` (M4 Pro, temperature 0, 2026-10-08, the
same answers before and after; `cached_tokens` and request time):

| Request | Before | After |
| --- | --- | --- |
| Gemma 4 12B, second turn, reasoning `none` | 0 of 58 | 30 |
| Gemma 4 12B, second turn, `low` | 0 of 56 | 32 |
| Gemma 4 E4B, second turn, `low` | 0 of 49 | 32 |
| Gemma 4 12B, the same request again | 0 of 37 | 30 |
| a new question under a 500-token system prompt (E4B, 12B, Qwen3.8) | 0 | 486–492; 3.96 → 1.75 s on the 12B, 10.3 → 5.2 s on Qwen3.8 |
| Gemma 4 12B, third and fourth turns of that conversation | 0 of 573, 0 of 636 (3.9, 4.3 s) | 498, 566 (1.7, 1.8 s) |

The first request under a new system prompt pays for keeping it: 0.25 s
on the 12B, 0.5 s on Qwen3.8.

A decision does not wait for a generation that is running: queued
decisions run between its steps (after each generated token, after each
prefill chunk), each model on its own Metal queue. Five
`laya-multilingual` decisions sent during a 2,000-token Qwen3.8
generation were answered in 51 to 164 ms (136.5 s before this), and the
generation took 139.5 s (141.4 s without them). During a long prefill a
decision waits for the current chunk, 0.45 s on Qwen3.8. Answers are the
same as on an idle server, logits included. A decision whose model does
not fit beside the generation in memory waits for it to end
([memory](#memory)).

### Agents

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

With this entry, pi 1.1.0 on Qwen3.8 did one of the agent playground's
tasks (a new module, its export, its tests, and a test run) in five
requests. The first prefilled pi's 7,297-token system prompt and tools in
109 s; each later one continued the session, reusing everything the
previous one had consumed (7,582 of 8,050 tokens, then 8,337 of 9,181,
10,251 of 10,357, and 10,419 of 10,591). Esc in pi cancelled a request
in its prefill.

## Embeddings

### `POST /v1/embeddings`

OpenAI's embeddings call for the embedding models `nuclis embed` runs
(`embeddinggemma-2`, [embeddinggemma.md](../models/embeddinggemma.md)):
one unit vector per input, in input order. OpenAI's SDKs work unchanged:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:9000/v1", api_key="none")
r = client.embeddings.create(model="embeddinggemma-2", input=["a cat", "a kitten"])
q = client.embeddings.create(model="embeddinggemma-2", input="how do auroras form?",
                             extra_body={"task": "search_query"})
```

```sh
curl -s localhost:9000/v1/embeddings -d '{"input": "how do auroras form?", "task": "search_query"}'
```

### The embeddings request

| Field | Meaning |
| --- | --- |
| `model` | a registry entry of kind `embedding` or an embedding catalogue name; default `embed.model` (`embeddinggemma-2`). Never a path |
| `input` | a string (one input), or a list whose items are strings or lists of content parts, at most 2,048. A part is `{"type": "text", "text": "…"}`, `{"type": "image_url", "image_url": {"url": "data:image/png;base64,…"}}` (a base64 data URL or bare base64; PNG, JPEG, HEIC, WebP, TIFF, GIF, BMP), or `{"type": "input_audio", "input_audio": {"data": "<base64>", "format": "wav"}}` (`wav`, `mp3`, `aiff`, `flac`, or `m4a`; at most 30 s). Each item is one input and gives one vector; its parts are read in order, in one pass. The body limit (32 MiB) bounds the media |
| `dimensions` | `768` (default), `512`, `256`, or `128`: the vector's leading values, renormalized (the widths the model is trained for) |
| `encoding_format` | `float` (default) or `base64`: the vector's little-endian f32 bytes, which OpenAI's Python SDK asks for and decodes |
| `user` | accepted and ignored |
| `task` | nuclis's: the use the vectors are for. Text is embedded exactly as given unless a task is named; `search_query`, `question_answering`, `fact_checking`, `code_retrieval`, `classification`, `clustering`, and `similarity` open each input with the model's `task: … \| query: ` prefix, before any image or clip, `document` with `title: {title or none} \| text: ` ([embeddinggemma.md § Tokenizer and task prefixes](../models/embeddinggemma.md#tokenizer-and-task-prefixes)) |
| `title` | nuclis's: a document's title, with `"task": "document"` only |
| `truncate` | nuclis's: `true` cuts an input over 8,192 tokens at its end instead of refusing it, and a clip over 30 s to its first 30 s (Google's processor's cut); the response lists the inputs cut. An image or clip the cut would split is dropped whole |
| `image_tokens` | nuclis's: soft tokens per image, `70`, `140`, `280` (default), `560`, or `1120`, Google's processor's budgets. More is finer and slower; it changes the vector, and the response records it |

For retrieval, embed the corpus with `"task": "document"` and the queries
with `"task": "search_query"`: the two are trained to meet.

#### What an embedding request may not carry

- **Token arrays** (`[1, 2, 3]` or `[[1, 2], [3]]`): a client that sends
  them tokenized with another model's vocabulary, so the vectors would be
  noise. LangChain's `OpenAIEmbeddings` does this by default; set
  `check_embedding_ctx_length=False` and it sends text.
- **An image by URL** (`https://…`): the server fetches nothing; send a
  data URL. An image that does not decode is `invalid_image`, a clip
  `invalid_audio`, each naming the input. An audio format other than the
  five is `unsupported_feature`.
- **A clip over 30 s** without `truncate` (`input_too_long`).
- **Images and audio without the projector**: `nuclis model pull
  embeddinggemma-2 --with mmproj` fetches it (`unsupported_feature` until
  then).
- **Other widths**, and an input over 8,192 tokens without `truncate`
  (`input_too_long`, with the input's index and token count).

### The embeddings response

OpenAI's shape on one line, then the `nuclis` object:

```json
{"object": "list",
 "data": [{"object": "embedding", "index": 0, "embedding": [0.014093, -0.0013, …]}],
 "model": "embeddinggemma-2",
 "usage": {"prompt_tokens": 25, "total_tokens": 25},
 "nuclis": {"space": "embeddinggemma-2@6f1bd4ac6c5d/768", "dimensions": 768, "task": "search_query",
            "image_tokens": null, "tokens": [25], "truncated": [],
            "timings_ms": {"load": 0.0, "tokenize": 0.3, "embed": 58.9}}}
```

- A `float` value is the shortest decimal that reads back to the same
  f32; `base64` carries those bits exactly. Read as f32, the two are
  identical.
- `usage` counts every token the model read, the `<bos>` and `<eos>` each
  input is framed with included.
- `space` names where the vectors live: the checkpoint, the first 12 hex
  digits of its file's SHA-256, and the width. Vectors compare only within
  one space; an index should store it beside them.
- `image_tokens` is the image budget when the request held an image, else
  null. `tokens` is each input's token count (an image's soft tokens and
  its two markers included), `truncated` the indexes of the inputs that
  were cut, `timings_ms` the request's open (when it opened the model),
  tokenizing and image encoding, and the passes it was in.

### Passes and memory

Embedding requests share GPU passes of at most 2,048 rows (an input's
tokens, aligned to 8). When the GPU frees, the server takes the oldest
waiting request's model and fills a pass with the inputs of every request
waiting for that model, in arrival order; a request whose inputs do not
fit continues in the next pass, ahead of later ones, and an input longer
than a pass runs alone. A pass runs between a chat generation's steps
(about 0.2 s at 2,048 rows), so a generation and an indexing job share
the GPU. An input's vector does not depend on what shares its pass:
batching changes timing, never vectors (`scripts/api-check.py --only
embeddings` checks it bit for bit).

One embedding model is open at a time; it opens on its first request
(0.7 s) and counts its file against the [memory](#memory) budget (310 MB
for the Q8_0 file). Its projector is opened with it when pulled; the
vision encoder is built on the first image (918 MB on Metal) and the audio
encoder on the first clip (658 MB), and the budget then adds each to the
model's share. Media are encoded one at a time on the GPU before their
pass: about 460 ms an image at 280 tokens, 130 ms for 4.8 s of speech.

## `GET /v1/models`

OpenAI's list shape, so an OpenAI client reads the ids; the `nuclis`
object is for nuclis clients.

```json
{
  "object": "list",
  "data": [
    {
      "id": "laya",
      "object": "model",
      "created": 0,
      "owned_by": "convaiinnovations",
      "nuclis": {
        "kind": "decision",
        "present": true,
        "loaded": true,
        "default": true,
        "family": "laya",
        "packs": true,
        "images": false,
        "max_len": 512,
        "head_max_len": 192,
        "repo": "convaiinnovations/laya",
        "revision": "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
      }
    }
  ]
}
```

Every decision model of the catalogue and the registry is listed, pulled
or not, then every language model that is present (the ones `/model`
offers in `nuclis chat`), then every embedding model of the catalogue and
the registry, pulled or not. A decision model's fields:

| Field | Meaning |
| --- | --- |
| `present` | its weights are on disk |
| `loaded` | it is open now |
| `default` | a request without `model` gets it |
| `family` | `laya` or `clef`, known before the model is pulled |
| `packs` | `true`: the states of a request, and the requests waiting with it, share one GPU pass (Laya). `false`: each state is its own pass, its cost linear in tokens (clef-flash: send one state per request with every question in it, and keep one or two requests in flight; [clef-flash.md § Time per decision](../models/clef-flash.md#time-per-decision)) |
| `images` | it reads a request's `images` (clef-flash with its projector pulled) |
| `max_len`, `head_max_len` | its sequence and question budgets in tokens; `head_max_len` is null for clef-flash (no per-question budget), and both are null for a Laya model not yet pulled |
| `owned_by` | the repository's owner, `local` for a directory |
| `created` | always 0 |

A language model's (`owned_by` is `nuclis`):

| Field | Meaning |
| --- | --- |
| `kind` | `language` |
| `name` | the model as a person names it: the catalogue's title ("Qwen3.8 27B") for a file the catalogue pins, else the GGUF's `general.name`, else the file's name |
| `architecture` | the GGUF's `general.architecture` (`qwen35`, `gemma4`, `muse-glimmer`) |
| `profile` | the prompt template it renders with (`qwen38`, `gemma4`, `gemma4_e`, `muse_glimmer`) |
| `quantization` | the catalogue's label (`UD-Q4_K_M`, `Q4_0 (QAT)`); for another file, the encoding holding most of its tensor bytes (`Q4_K`); null when the header does not say |
| `size_bytes` | the main file's size |
| `present`, `loaded`, `default` | as above; `default` is `engine.model` |
| `context_length` | the window it opens with (`ctx_size`) |
| `images` | its entry names a projector |
| `efforts` | the reasoning levels its template renders, in the profile's names: `off` is the request's `none` (`reasoning_effort` accepts both) |
| `detail` | the same in one line for people: architecture, quantization, and size, or the file's name |

An embedding model's (`owned_by` is the repository's owner, `local`
without one):

| Field | Meaning |
| --- | --- |
| `kind` | `embedding` |
| `name` | the catalogue's title ("EmbeddingGemma 2"), else the entry's name |
| `architecture`, `quantization` | `gemma-embedding2`; the catalogue's label (`Q8_0`), null for another file |
| `size_bytes` | the file's size |
| `present`, `loaded`, `default` | as above; `default` is `embed.model` |
| `dimensions` | the widths it is trained for, `[768, 512, 256, 128]` |
| `modalities` | what this server embeds with it: `["text"]`, and `"image"` and `"audio"` once the projector is pulled |
| `max_tokens` | tokens per input, 8,192 |
| `tasks` | the `task` values it takes |
| `repo`, `revision` | where its file came from |

## `GET /v1/health`

```json
{
  "status": "ok",
  "version": "0.4.0-dev",
  "backend": "metal",
  "loaded": ["laya", "laya-multilingual"],
  "language": "qwen3.8-27b",
  "embedding": "embeddinggemma-2",
  "memory": {"limit": 34359738368, "resident": 24227472014,
             "models": [{"name": "qwen3.8-27b", "kind": "language", "bytes": 23583635232},
                        {"name": "laya-multilingual", "kind": "decision", "bytes": 643836782}]},
  "queue": {"queued": 0, "running": false, "completed": 1709},
  "decisions": {"waiting": 0, "batches": 1707, "requests": 3136},
  "embeddings": {"waiting": 0, "passes": 42, "requests": 37},
  "connections": 1
}
```

`loaded` lists the open decision models, `language` the open language
model (null before the first chat request), `memory` the budget
(`limit`, `resident`, and each resident model's `name`, `kind`, and
`bytes`, most recently used first). `queue` is the GPU's
(`completed` counts GPU items, a batch or a completion being one);
`decisions.waiting` is decision requests not yet in a pass,
`batches` and `requests` the passes run and the requests they answered
since the start. `embedding` is the open embedding model (null before the
first embedding request), and `embeddings` counts its waiting requests,
passes, and answered requests the same way.

## Measured rates

Apple M4 Pro (48 GB), ReleaseFast, Zig 0.16.0, macOS 27.0; `nuclis serve
--model laya --model laya-multilingual` on Metal, on the port of the
time (8735) and before the request log existed; ApacheBench 2.3 with
keep-alive (`ab -k`) on loopback; checkpoints `convaiinnovations/laya` at
`55cf4c4e`. The decision request is one state (93 characters, a billing
complaint) and two questions (a two-option choice and a noul);
latencies are ab's percentiles (`-e`, which keeps fractions of a
millisecond). The request:

```json
{"questions": {"team": {"type": "choice", "instructions": "Which team should handle this?",
   "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages, errors"}},
  "urgent": {"type": "noul", "instructions": "Is this urgent?"}},
 "state": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."}
```

with `"model": "laya-multilingual"` added for that checkpoint.

### One GPU job per request (before batching), 2026-10-02, commit `cb79cde`

| Route | Model | Concurrency | Requests | Requests/s | p50 ms | p99 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `GET /v1/health` | — | 1 | 1,000 | 25,905 | 0.019 | 0.082 |
| `GET /v1/health` | — | 16 | 20,000 | 146,801–166,589 | 0.096 | 0.187 |
| `POST /v1/decisions` | laya | 1 | 512 | 28.8 | 34.4 | 36.4 |
| `POST /v1/decisions` | laya | 4 | 512 | 28.9 | 136.4 | 153.0 |
| `POST /v1/decisions` | laya | 16 | 512 | 29.4 | 543.2 | 544.2 |
| `POST /v1/decisions` | laya-multilingual | 1 | 512 | 69.2 | 14.5 | 14.9 |
| `POST /v1/decisions` | laya-multilingual | 4 | 512 | 71.1 | 55.6 | 58.5 |
| `POST /v1/decisions` | laya-multilingual | 16 | 512 | 72.0 | 222.2 | 222.9 |

What the server adds: a warm request's HTTP p50 against the encode the
response reports (median of 40, `timings_ms.encode`), tokenizing 0.1 ms:
laya 34.05 ms against 34.5 ms, laya-multilingual 13.97 ms against
14.3 ms, so nothing measurable (the targets were within 2 ms). The same
request through `nuclis decide --request … --json`, a process per call:
0.09–0.10 s (laya), 0.20–0.21 s (laya-multilingual), wall clock.

### Batched across requests, 2026-10-02, commit `cd35ed8`

Same machine, build mode, request, and protocol:

| Model | Concurrency | Requests/s before → after | p50 ms before → after | p99 ms before → after |
| --- | ---: | --- | --- | --- |
| laya | 1 | 28.8 → 29.2 | 34.4 → 34.2 | 36.4 → 34.7 |
| laya | 4 | 28.9 → 33.8 | 136.4 → 118.3 | 153.0 → 118.9 |
| laya | 16 | 29.4 → 36.4 (1.24×) | 543.2 → 439.3 | 544.2 → 444.3 |
| laya-multilingual | 1 | 69.2 → 71.2 | 14.5 → 14.0 | 14.9 → 14.5 |
| laya-multilingual | 4 | 71.1 → 85.1 | 55.6 → 46.9 | 58.5 → 47.5 |
| laya-multilingual | 16 | 72.0 → 93.4 (1.30×) | 222.2 → 171.0 | 222.9 → 172.1 |

At concurrency 16 a pass held 7.9 requests on average (512 in 65 passes,
`GET /v1/health` counters), about 820 rows. The gain is the model's: one
request already encodes at 3,020 rows/s (laya, 104 rows in 34.4 ms) and
7,160 (laya-multilingual, 101 in 14.1 ms), against the packed rates
[laya.md § Time per call](../models/laya.md#time-per-call) measures, about 3,700
and 8,660, so packing can add about 1.2× here, and does. The planned 2×
would need a faster Laya encode, not a different server. p99 is about two
passes (laya 444 ms against 2 × 217 ms, laya-multilingual 172 against
2 × 85): a request that just missed a pass waits for it and runs in the
next.

Batching changes timing, never answers: 16 concurrent requests with
different questions (2 states each, choice, noul, and score questions)
returned the same bytes as each sent alone, timings aside, on both
checkpoints on Metal (16 in 3 and in 4 passes), and the unit test
asserts the same on the CPU.

### Embeddings, 2026-10-09

Apple M4 Pro (48 GB), ReleaseSafe (`make metal`), Zig 0.17.0, macOS 27.0;
`nuclis serve` on Metal with `embeddinggemma-2` (the Q8_0 file and
`unsloth/embeddinggemma-2-GGUF` at `031f0d4b`), commit `7d37e93`. Each input is `<bos>`,
" the" repeated, `<eos>` (the cost does not depend on the text), sent as
one request with `encoding_format: base64`; one warm-up, then the median
of 7, from Python's `urllib` on loopback. `embed` is the response's
`timings_ms.embed`, the passes alone.

| Request | Tokens | Wall ms | `embed` ms | Inputs/s |
| --- | ---: | ---: | ---: | ---: |
| 64 inputs × 256 tokens | 16,384 | 1,686 | 1,674 | 38.0 |
| 1 input × 512 tokens | 512 | 60 | 59 | 16.8 |
| 1 input × 8,192 tokens | 8,192 | 1,756 | 1,748 | 0.6 |

The server adds about 12 ms to the 64-input request (parsing, the queue,
and 64 base64 vectors), and the batch runs at the rate the encoder has
alone (1,671 ms for the same inputs in
[embeddinggemma.md § The Metal plan](../models/embeddinggemma.md#the-metal-plan-2026-10-09)).
Passes of 2,048 rows cost no throughput: the same server with 8,192-row
passes gave 1,669 ms and 38.3 inputs/s, within the spread, while a
2,048-row pass holds the GPU for about 0.2 s instead of 1.7 s.

