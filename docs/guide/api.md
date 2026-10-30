# The nuclis API

`nuclis serve` runs the nuclis API: an HTTP/1.1 server on this machine
with three services. **Decisions** keep decision models open and answer
typed questions about text or JSON, in TypeSafe's Jev protocol, so a
client written for `api.typesafe.ai` switches by changing its base URL.
**Chat Completions** serve the language models in OpenAI's format, so a
client written for an OpenAI-compatible server does the same.
**Embeddings** turn text, images and audio into vectors in OpenAI's embeddings format.
**Speech** speaks text in OpenAI's speech format, streamed as it is
made, beside a chat generation, in a built-in speaker, a saved voice
cloned from a clip, or a voice described in words. This page
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
- [Speech](#speech): [`POST /v1/audio/speech`](#post-v1audiospeech), [the
  request](#the-speech-request), [voices](#voices), [streaming and
  cancellation](#streaming-and-cancellation), [scheduling](#scheduling)
- [Live transcription](#live-transcription): [`GET /v1/audio/transcriptions/stream`](#get-v1audiotranscriptionsstream), a WebSocket
- [Transcription (files)](#transcription-files): [the request](#the-file-transcription-request), [the response](#the-file-transcription-response), [segments](#segments), [long recordings](#long-recordings)
- [`GET /v1/models`](#get-v1models), [`GET /v1/health`](#get-v1health)
- [Measured rates](#measured-rates)

## Running the server

```sh
nuclis model pull laya                 # once: the default decision model
nuclis serve                           # http://127.0.0.1:9000/v1, nothing open yet
```

| Option | Meaning |
| --- | --- |
| `--host <ip>` | the address to listen on: an IP literal (`127.0.0.1`, `::1`, `0.0.0.0`) or `localhost`; default `serve.host`, `127.0.0.1`. Any address beyond loopback prints a warning: there is no authentication. |
| `--port <n>` | default `serve.port`, `9000` |
| `--model <name>` | a model opened before the server listens, of any kind: a decision, language, embedding, speech, or transcription model by registry or catalogue name, or a decision checkpoint's directory; repeated, at most 2 decision, 1 language, 1 embedding, 1 speech, and 1 transcription model (what the server keeps open). Without it nothing opens at start: each model opens on its first request (`decide.model` for a decision request that names none, `embed.model` for an embeddings request, `speech.model` for a speech request) |
| `--pin <name>` | a model opened at start, as `--model`, that the memory budget never closes ([memory](#memory)); repeated, at most 8; replaces `serve.pin` |
| `--memory <GiB>` | what every open model may hold together; default `serve.memory_bytes`, else physical memory less a third, at most 16 GiB ([memory](#memory)) |
| `--backend cpu\|metal` | where the models run; default `metal` in a Metal build |
| `--quiet` | no request log (`serve.log: false` does the same) |
| `--timeout <s>` | how long a request may wait for the GPU before `529 timeout`; default `serve.timeout`, `300` |

The configuration file (`~/.nuclis/nuclis.json`) holds the defaults:

```json
"decide": { "model": "laya" },
"speech": { "model": "qwen3-tts-1.7b-customvoice", "voice": "ryan" },
"serve":  { "host": "127.0.0.1", "port": 9000, "log": true, "timeout": 300,
            "pin": ["gemma-4-e4b-qat", "qwen3-tts-1.7b-customvoice", "nemotron-asr-0.6b"] }
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
32 GiB on a 48 GB machine by default (physical memory less a third of it, at most 16 GiB, for
the system and other applications). A model counts its files (a decision
checkpoint's weights, backbone, and projector; a language model's
weights, its draft file when speculative decoding is on, and its image
projector once an image arrives; an embedding model's file; a speech
model's weights and codec; a transcription model's weights) plus, for a
language model, its attention cache and the saved states it may keep (`cache.memory_bytes`,
4 GiB), and for a speech or transcription model the device memory it
created on opening (its converted weights, caches, and scratch). Qwen3.8 counts 22.0 GiB, `laya-multilingual` 0.6 GiB, clef-flash
8.4 GiB, Gemma 4 E4B 8.9 GiB, `qwen3-tts-1.7b-customvoice` 4.7 GiB (5,095,675,552 B:
the BF16 talker and the F32 codec, 4,515,695,644 B, and 553 MiB of
device buffers), `nemotron-asr-0.6b` 2.4 GiB (2,602,777,056 B: 2,552,062,944
and 48 MiB). What a request holds while it runs is not counted: a
speech request's synthesizer (about 33 MiB on the device), a live
transcription session (about 70 MB), a file's upload.

Before a model opens, the least recently used others close until it
fits, whichever kind they are. A language model is never closed while a
request runs on it: a decision that would need its memory waits for the
generation to end. A model larger than the whole budget is refused
(`model_too_large`). With `--memory 30` and Qwen3.8 open, an idle
clef-flash decision closed Qwen3.8 (and the older Laya) and answered in
4.2 s; the next chat request reopened Qwen3.8 in 1.2 s and closed
clef-flash; a clef-flash decision sent during a generation was answered
0.95 s after it ended. `GET /v1/health` lists what is resident.

A **pinned** model (`--pin`, `serve.pin`) opens at start and is never
closed to make room, so a voice assistant's three models stay ready
whatever else is asked:

```sh
nuclis serve --pin gemma-4-e4b-qat --pin qwen3-tts-1.7b-customvoice --pin nemotron-asr-0.6b
```

A model that does not fit beside the pinned ones is refused at once
(`model_too_large`), however long it would wait. A pin is by the name the
model is opened by, and it guards against the budget, not against a
request: the server keeps one model of each kind but decisions, so a
request naming another language or speech model still replaces the
pinned one in its slot, and the pin applies again when its name reopens.
`GET /v1/health` marks each resident model's `pinned`.

**Before a model opens**, `GET /v1/models` says what it will count in
each entry's `memory_bytes`, with `memory_source` saying how it knows:

| `memory_source` | `memory_bytes` |
| --- | --- |
| `open` | what it counts now, the number `/v1/health` reports |
| `measured` | what it counted the last time it opened, with the same files and, for a language model, the same `ctx_size`, `kv_precision`, `cache.memory_bytes`, and speculative decoding; kept in `~/.nuclis/cache/memory.json` across starts |
| `estimate` | what it counts before it opens. Exact for an embedding and a decision model; for a speech or transcription model its files, under the count by the device memory (none for Drama 3 and Qwen3-TTS, 49 MiB for Nemotron ASR); for a language model its files and `cache.memory_bytes`, under the count by its attention cache and draft file |

A model opened once on this machine is `measured`, exact until its
files or those settings change. Measured on the M4 Pro (2026-10-30)
with four models pinned at `--memory 47`, every listed number equalled
`/v1/health`'s: the Gemma 4 26B-A4B with its draft 24,781,970,336 B
(before its first open the estimate was 21,090,982,816 B, the file and the 4 GiB of saved states: 15 % under),
drama-3 20,787,264,352 B, `nemotron-asr-0.6b` 2,603,784,672 B, and
embeddinggemma-2 309,855,520 B. Opened once each, the others counted:

| Model | Counted | Estimate before |
| --- | ---: | ---: |
| `qwen3.8-27b` | 23,583,635,232 | 20,759,407,520 |
| `gemma-4-26b-a4b` | 22,235,001,920 | 18,544,014,400 |
| `gemma-4-12b-qat` | 16,902,177,472 | 11,011,324,096 |
| `gemma-4-e4b-qat` | 9,509,865,184 | 8,510,663,072 |
| `qwen3-tts-1.7b-base` | 4,539,706,836 | the same |
| `qwen3-tts-1.7b-customvoice`, `-voicedesign` | 4,515,695,644 | the same |
| `qwen3-tts-0.6b-base` | 2,511,637,364 | the same |
| `qwen3-tts-0.6b-customvoice` | 2,493,919,668 | the same |

A model refused at start (`--model`, `--pin`) says the sizes, the
needed and the free beside the models pinned before it:

```text
error: ModelTooLarge: the speech model drama-3 needs 19.4 GiB; 16.6 of 40.0 GiB is free beside the pinned models (--memory, serve.memory_bytes)
```

The pins open in the order decision, language, embedding, speech,
transcription, so a later kind is the one refused; a limit that fits
them all is at least the sum of their `memory_bytes`.

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
| request body | 4 MiB; 32 MiB for `/v1/chat/completions` and `/v1/embeddings`; 256 MiB for `/v1/audio/transcriptions` (`serve.transcription_bytes`) | `413 payload_too_large`, connection closed |
| open connections | 64 | `529 busy`, connection closed |
| decision requests waiting for the GPU | 64 | `529 busy` |
| embedding requests waiting for the GPU | 64 | `529 busy` |
| file transcriptions waiting for the GPU (each holds its upload) | 16 | `529 busy` |
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
| 413 | `payload_too_large` | the body exceeds 4 MiB (32 MiB for chat and embeddings, 256 MiB or `serve.transcription_bytes` for file transcriptions) |
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
| 400 | `not_a_language_model` | chat: `model` names a decision, embedding, speech, or transcription model |
| 400 | `images_unsupported` | chat: an image for a model whose projector is not pulled |
| 400 | `audio_unsupported` | chat: an `input_audio` part for a model that does not hear (only Gemma 4 E4B does), or whose projector is not pulled |
| 400 | `invalid_audio` | chat: an `input_audio` part that does not decode |
| 400 | `input_too_long` | chat: a clip over 30 s (never cut) |
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
| 400 | `invalid_request` | file transcriptions: not `multipart/form-data`, no `file` part or two, a malformed form, a field out of range (`param` names it), a `language` the model does not know |
| 400 | `unsupported_feature` | file transcriptions: a `response_format` other than the five, `prompt`, `include[]`, `stream` with `srt` or `vtt` |
| 400 | `invalid_audio` | file transcriptions: the file is not audio nuclis reads, or stops decoding midway (the message gives where) |
| 400 | `not_a_transcription_model` | file transcriptions: `model` names a model of another kind |
| 404 | `model_not_found` | file transcriptions: `model` names no pulled transcription model |
| 400 | `model_too_large` | file transcriptions: the model needs more than the budget leaves beside the pinned models |
| 431 | `headers_too_large` | the request line and headers exceed 16 KiB |
| 500 | `model_failed` | the model could not be opened, or a completion failed on the GPU |
| 500 | `malformed_tool_call` | chat: the model wrote a tool call that does not parse (Gemma 4), so nothing ran and there is no answer; streamed, it is the stream's error line. Sending the request again usually works |
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
| `messages` | roles `system`, `developer`, `user`, `assistant`, `tool`. `content` is a string, null (an assistant turn of calls only), or parts: `text`, and on user messages `image_url` with a base64 data URL (at most 8 images, 16 MiB each) and `input_audio` (below) |
| `input_audio` parts | `{"type": "input_audio", "input_audio": {"data": "<base64>", "format": "wav"}}`: `data` base64 or a base64 data URL, `format` one of `wav`, `mp3`, `aiff`, `flac`, `m4a` (the decoder reads the bytes; the name is checked, not trusted). At most 8 clips per request, 16 MiB and **30 s** each (750 rows; a longer clip is `400 input_too_long`, never cut). Only Gemma 4 E4B hears them (its projector pulled); another model answers `400 audio_unsupported`, a clip that does not decode `400 invalid_audio`. The model reads a message's images, then its clips, then its text ([gemma4.md § Audio input](../models/gemma4.md#audio-input-on-the-e4b-2026-10-27)) |
| assistant `reasoning_content` | the turn's reasoning, given back as it was returned (`reasoning` is read too); the model's template keeps or drops it |
| assistant `tool_calls`, tool `tool_call_id` | the calls and their results; a result must answer an earlier call |
| `tools` | `type: "function"` with `name`, `description`, `parameters` (a JSON Schema); at most 64. The model's own template renders them. With `serve.web`, also `{"type": "web_search"}` and `{"type": "web_fetch"}`, which the server runs ([hosted web tools](#hosted-web-tools)) |
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
| `audio`, `modalities` beyond `text`, `prediction`, part types other than `text`, `image_url`, and `input_audio`, an audio `format` outside the five above | text out; text, images, and audio in |
| an `image_url` that is not a data URL | the server fetches no URL |
| a `web_search` or `web_fetch` tool type without `serve.web`; a function of the same name beside one (`invalid_request`) | the server runs no web tools unless configured to; a clash would make its calls ambiguous |

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
Gemma 4 and Muse Glimmer. Images and clips count by their bytes: a
conversation resent with the same ones reuses its state, another image or
clip in the same place does not (measured on the E4B with a 4.8 s clip:
133 of 136 tokens cached for the same request again, 152 of 171 for a
follow-up, 0 for another clip). Changing the model empties the cache.

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

### Hosted web tools

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
  entry without `status` (not streamed) is finished.

  ```json
  {"delta": {"nuclis_web_calls": [{"index": 0, "status": "started", "name": "web_search", "arguments": "{\"query\":\"PEP 8 line length\"}"}]}}
  {"delta": {"nuclis_web_calls": [{"index": 0, "status": "finished", "name": "web_search", "arguments": "{\"query\":\"PEP 8 line length\"}", "summary": "8 results", "is_error": false}]}}
  ```
- The usage counts every completion; the log line adds `2 web calls in 3
  rounds`, and `1 web exchange put back` when earlier hosted rounds were
  restored. The model's state carries over between completions, so a round
  reuses the conversation before it (measured: 1,257 of 2,084 prompt tokens
  over three rounds, Saluki, against the stub, 2026-10-16).
- Text the model writes before a call (Muon's "Let me check.") is rendered
  by Gemma 4's template after the call's results, where the next round's
  prompt closes the turn and opens another; history keeps the answer in
  that turn. The state is kept where the two still agree, after the
  results, so the turn after a search prefills only the answer and the new
  message. Measured on Muon's conversation (E4B, reasoning `none`, against
  the stub, 2026-10-28): the turn after a search reused 1,214 of 1,784
  prompt tokens in 2.22 s before, 1,723 of 1,785 in 1.04 s after; the
  26B-A4B, which rewrites its turns anyway, reused 1,701 of 1,773 either
  way.
- A web call waits on the GPU worker; decision and embedding items queued
  meanwhile run during its waits. A client that leaves cancels it.
- Without `serve.web` (or `--web-url`), and under `--no-web`, any of the
  three types is `400 unsupported_feature` naming `tools[i]`.

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

## Speech

### `POST /v1/audio/speech`

OpenAI's speech call for the speech models `nuclis speak` runs, Qwen3-TTS
in three kinds: CustomVoice (two sizes) speaks in nine built-in speakers,
Base (two sizes) in a saved voice cloned from a clip ([voices](#voices)),
and the 1.7B VoiceDesign in a voice its `instructions` describe; and
Fish Audio's Drama 3 (`drama-3`, [below](#drama-3)), which clones a saved
voice at 44.1 kHz with its delivery cued in the text. The
audio streams as it is made: the first
80 ms leave after one frame, then chunks of one, two, and four frames
(320 ms), each no longer than the audio sent before it, so a listener
who plays from the first byte never waits between them.

```sh
curl -s localhost:9000/v1/audio/speech -o hello.wav -d '{
  "input": "Hello there. This is nuclis speaking through the API.",
  "voice": "ryan"
}'
```

The response is `200`, `transfer-encoding: chunked`, `content-type:
audio/wav`: a WAV header whose sizes are the largest a field holds (the
length is not known when it is sent; players read to the end of the
stream), then 16-bit mono samples at the model's rate (24 kHz; 44.1 kHz
for Drama 3, `sample_rate` in `GET /v1/models`). With `"response_format":
"pcm"` it is the samples alone (`audio/pcm`, little-endian, at that rate).

One speech model is open at a time beside the others; it opens on its
first request (0.35 s for the 1.7B with its files in the page cache) or
at start with `--model` or `--pin`. A request's text is spoken whole: send
a sentence or two at a time, as a client splitting a chat answer at
sentence ends does.

### The speech request

| Field | In nuclis |
| --- | --- |
| `input` | the text, UTF-8, at most 16 KiB (`400 request_too_large`) |
| `model` | a registry entry of kind `speech` or a catalogue speech name (`qwen3-tts-1.7b-customvoice`, `qwen3-tts-0.6b-customvoice`, `qwen3-tts-1.7b-base`, `qwen3-tts-0.6b-base`, `qwen3-tts-1.7b-voicedesign`, `drama-3`); never a path. The names before 0.11.0 (`qwen3-tts-1.7b`, `qwen3-tts-0.6b`) are `404 model_not_found`, the message naming the new one. Absent: the open one, else `speech.model` (`qwen3-tts-1.7b-customvoice`). Another speech model replaces the open one; that request pays the open |
| `voice` | on CustomVoice one of its speakers: `aiden`, `dylan`, `eric`, `ono_anna`, `ryan`, `serena`, `sohee`, `uncle_fu`, `vivian` (OpenAI's voice names are not speakers here: `400 invalid_request`, listing them). On a Base a saved voice, by name or as OpenAI's custom voice `{"id": "<name>"}`: `404 voice_not_found` for one not saved, `400 invalid_request` for a speaker's name. On VoiceDesign ignored (OpenAI's clients always send one). A custom voice on CustomVoice or VoiceDesign is `400 voice_cloning_unsupported`. Absent: `speech.voice` (`ryan`) |
| `instructions` | on the 1.7B CustomVoice how to say it (`"Speak with excited, breathless joy."`); on VoiceDesign the voice itself (`"A calm elderly man with a deep, gravelly voice."`), sampled anew each request; absent, the model's own choice. On the 0.6B CustomVoice, a Base, and Drama 3 (whose cues are in `input`), `400 instructions_unsupported` |
| `response_format` | `wav` (the default; OpenAI's is `mp3`) or `pcm` |
| `stream_format` | `audio` (the default): the bytes as they are made. `sse`: server-sent events, each `{"type": "speech.audio.delta", "audio": "<base64>"}` carrying the next bytes of the same stream (the WAV header in the first, for `wav`), then `{"type": "speech.audio.done", "usage": {"input_tokens", "output_tokens", "total_tokens"}, "finish_reason"}` (the prompt's rows in, the frames out; `eos` when the model ended the speech, `length` when [the frame budget](../models/qwen3-tts.md#the-frame-budget) stopped it, which a client may treat as a failed line) |
| `speed` | `1` only (`400 unsupported_feature` otherwise): the model has no rate control |
| `seed` | nuclis's: the sampler's seed, default 0. The same text, voice, and seed give the same samples, alone or beside a chat generation |
| `language` | nuclis's: `auto` (the default), or one of the model's (`english`, `chinese`, `german`, …; `400 invalid_request` lists them). Drama 3 takes only `auto` |
| `context` | nuclis's, Drama 3 only: an earlier response's `x-speech-context`, whose last sentence this request continues ([below](#drama-3)); ignored when unknown or expired |

Other fields OpenAI defines are ignored. Refused with `400
unsupported_feature`: `mp3`, `opus`, `aac`, and `flac` (no encoder here),
a `speed` other than 1.
`model` naming another kind is `400 not_a_speech_model`; a name nothing
pulled answers is `404 model_not_found`. A refusal on the GPU worker (an
unknown speaker or language) is still a status: the response's head
waits until the request's prompt is prefilled, which is before its first
audio anyway. A model that fails to open is `500 model_failed`; one too
large for the budget beside the pinned models `400 model_too_large`.

### Voices

A Base speaks in a voice saved from a clip of 1–30 s: its timbre from the
clip alone, or, with the clip's exact words, the closest likeness (Qwen's
in-context cloning). A voice is saved once and named; the first speech
request in it on a Base encodes it on the CPU (under a second; the
connection waits, the GPU does not) and keeps the result for later
requests. A clip-alone voice then starts as fast as a speaker's; with a
transcript the clip's frames are part of every prompt, so the first audio
grows with the clip (on the 0.6B about 260 ms for 4.6 s, 430 ms for 10 s,
650 ms for 15 s; [qwen3-tts.md § Speed](../models/qwen3-tts.md#speed-2026-10-29)).
Either Base size speaks any saved voice. The routes follow OpenAI's custom voices; nuclis has no consent
flow, so clone only a voice you have the right to use.

```sh
curl -s localhost:9000/v1/audio/voices -F name=anna -F audio_sample=@anna.m4a \
  -F 'reference_text=Hello, it is Anna. I will be reading the news today.'
curl -s localhost:9000/v1/audio/speech -o hi.wav -d '{
  "model": "qwen3-tts-1.7b-base", "input": "Good morning.", "voice": {"id": "anna"}
}'
```

| Route | Does |
| --- | --- |
| `POST /v1/audio/voices` | saves a voice, `201` with its object. Multipart: `audio_sample` (the clip: WAV, AIFF, MP3, M4A, FLAC, CAF; the body at most 32 MiB), `name` (`[a-z0-9][a-z0-9_-]{0,63}`, not one of the speakers), and nuclis's `reference_text` (what the clip says, at most 2 KiB; trimmed, empty is none). OpenAI's `consent` is ignored. `409 voice_exists`, `400 too_many_voices` past 256, `400 invalid_request` with the field for the rest |
| `GET /v1/audio/voices` | `{"object": "list", "data": [voice…]}`, by name |
| `GET /v1/audio/voices/{name}` | one voice, or `404 voice_not_found` |
| `DELETE /v1/audio/voices/{name}` | `{"id", "object": "audio.voice.deleted", "deleted": true}`; the clip and all made of it go |

A voice object is OpenAI's `{"id", "object": "audio.voice", "name",
"created_at"}` (`id` is the name; `created_at` Unix seconds) with
nuclis's `reference_text` (or null) and `seconds`. Voices live under the
server's user root, `~/.nuclis/voices/<name>/`, shared with `nuclis voice`.

For a voice that stays the same, VoiceDesign's output can be saved: speak
a line or two in a described voice, then save that audio with its text as
a voice and speak in it on a Base (Qwen's *voice design then clone*).

### Drama 3

`drama-3` (Fish Audio's Drama 3, 35B-A3B, one 20.8 GB Q4_K GGUF;
[models/drama3.md](../models/drama3.md)) speaks in a saved voice with its
transcript: a voice saved without `reference_text` is `400
invalid_request` (param `voice`), so save Drama 3's voices with their
words. Any saved voice serves it and the Bases alike; a voice
saved since Drama 3 also keeps its clip at 44.1 kHz, which Drama 3
encodes (an older one is resampled from 24 kHz, losing the top of the
band). Its first use encodes the clip on the CPU with Fish's codec.

The delivery is cued in `input` itself:

- free-form cues in brackets before the words they shape: `[whispering]
  Don't wake him.`, `[laughs] No way.`, `[sighs]`;
- paired tags around words: `<whisper>…</whisper>`, `<emphasis>`,
  `<soft>`, `<fast>`, `<slow>`, `<stress>`, `<forceful>`;
- speakers `<|speaker:0|>` to `<|speaker:15|>` before their lines;
  untagged text is speaker 0, the saved voice.

Refused with `400 invalid_request` (param `input`, the message naming
the fault): a bracket left open or a cue inside a cue, an unclosed tag,
a speaker tag outside 0–15, and any other `<|…|>` token. Line breaks are
spaces. A request's text is split at sentence ends (Arabic's `؟` and `؛`
included) and spoken sentence by sentence, each in the voice of the one
before it. A sentence that has not started speaking within 1.5 s of audio
(steady noise does not count as speaking) is drawn again with the next
seed, twice at most. A sentence that falls silent for 1.5 s ends there,
keeping 0.28 s of the silence: [drama3.md § Stalls, and what
nuclis adds](../models/drama3.md#stalls-and-what-nuclis-adds).

**One sentence a request.** Every Drama 3 response carries an
`x-speech-context` header, an id for its last sentence. A later request
in the same model and voice that passes it as `"context": "<id>"` takes
that sentence (its text and the audio codes it was spoken with) as
history. So it continues the earlier delivery, as a request's own
sentences do. The sentence is kept for 5 minutes once its request ends,
at most 32 of them. An id that is unknown, expired, or from another voice
is ignored: the request is spoken without it, not refused, and the log's
`context` says which (`hit`, `missing`, `expired`, `other_voice`). A
request sent while the one it continues is still speaking waits behind
it in the queue, so the sentence is kept by the time it starts.

```sh
curl -s -D head.txt -o one.pcm localhost:9000/v1/audio/speech -d '{"model": "drama-3",
  "voice": {"id": "anna"}, "response_format": "pcm", "input": "[tired] يا عم ده إحنا بقالنا ساعة مستنيين."}'
curl -s -o two.pcm localhost:9000/v1/audio/speech -d '{"model": "drama-3", "voice": {"id": "anna"},
  "response_format": "pcm", "input": "والزحمة دي مش طبيعية.", "context": "'"$(grep -i x-speech-context head.txt | cut -d" " -f2 | tr -d "\r")"'"}'
```

Every prompt starts with the same rows for a voice: the system turn and
the voice's clip with its transcript (328 rows for a 12 s clip). The model
keeps the state after them for its 4 most recent voices. A request then
prefills only its own text (and a context's sentence); the first in a
voice since the model opened prefills both. On the M4 Pro with a 12 s
clip, a one-sentence request's prefill went from 1,612 ms to 243 ms
(first audio 393 ms), or 700 ms with a context (first audio 854 ms). The
take is the same bit for bit either way:
[drama3.md § The prompt's prefix](../models/drama3.md#the-prompts-prefix).

### Streaming and cancellation

The head is sent once the prompt is prefilled; the samples follow chunk
by chunk. An event stream silent for 10 s gets an SSE comment (`:
speaking`). An error after the head (the device fails mid-utterance) is
one `data: {"error": {…}}` event in an event stream; a raw stream ends
without its last chunk, which an HTTP client reports as a truncated
body.

A client that closes the connection stops its request between chunks:
the server notices within a second, frees the synthesizer, and the next
request starts. A request still waiting is taken out of the queue; one
that waits `serve.timeout` without starting is `529 timeout`. A
1.5 s client timeout on a long sentence stopped it after 2.96 s of
audio had been made, and `GET /v1/health` showed nothing speaking a
second later.

### Scheduling

Speech runs on the one GPU worker as short items: each run makes one
chunk, so a chunk runs between the steps of a chat generation (and of
its hosted tool waits) and speech never waits for a generation to end.
Between a generation's steps speech takes only what playback needs: one
frame a run (about 37 ms on the 1.7B), and none while the audio sent is
a second or more ahead of its playback, counted from the request's
first byte at real time. So beside a generation a request is made at
playback speed with a second in hand, and the generation keeps the rest
of the GPU (on the 1.7B about half its rate, on the 0.6B 60 %: § The
voice loop below); alone, with nothing waiting, it runs at full speed in
chunks of four. Opening a voice is the exception: a request whose voice
is not open waits for the running generation to end, since opening may
close the language model it runs on.

A voice speaks one request at a time (its talker cache and the codec's
scratch are one synthesizer's), so concurrent speech requests queue in
arrival order, at most 64 waiting (`529 busy` past them); each one's head
is sent when its turn begins. Three sent together answered at 0.14,
1.57, and 2.83 s, each when the one before it ended. A client that does
not read keeps at most about 5 s of audio queued on the server; the
request then waits for it rather than hold the GPU.

The speech models are listed by `GET /v1/models` (`kind` `speech`, with
`voices`, `instructions`, `sample_rate` (24000; 44100 for `drama-3`),
and `formats`), the
transcription models too (`kind` `transcription`, `sample_rate` 16000), each
with `present`, `loaded`, `default`, `size_bytes` (weights and codec),
`repo`, and `revision`. `GET /v1/health` gives the open `speech` and
`transcription` models and `speech_requests`: `waiting`, whether one is
`speaking`, and those `answered` since the start.

## Live transcription

### `GET /v1/audio/transcriptions/stream`

A WebSocket ([RFC 6455](https://www.rfc-editor.org/rfc/rfc6455)) for a
voice client's microphone: the client streams audio as it is captured,
the server answers with the transcript as it forms, and with a final
transcript and word times when the client says the utterance ended. It
runs Nemotron 3.5 ASR streaming 0.6B, the model `nuclis transcribe`
runs ([nemotron-asr.md](../models/nemotron-asr.md)).

```sh
python3 scripts/ws-transcribe.py clip.wav --language en-US   # a client in the standard library
```

The handshake is a `GET` with `upgrade: websocket`, `connection:
upgrade`, a `sec-websocket-key`, and `sec-websocket-version: 13`; the
server answers `101` and negotiates no subprotocol and no extension.
The session is set by the query:

| Parameter | In nuclis |
| --- | --- |
| `model` | a registry entry of kind `transcription` or `nemotron-asr-0.6b`; never a path. Absent: the open one, else `transcribe.model` |
| `language` | a locale of the model's prompt dictionary (`en-US`, `de-DE`, `fr`, `es-419`, …) or `auto` (the default: the model tags the language it heard), or a comma-separated list of up to four, `auto` only last ([several languages](#several-languages)) |
| `lookahead` | `0`, `3` (the default), `6`, or `13` rows of 80 ms the encoder waits for: a chunk, and so a possible partial, every 80, 320, 560, or 1,120 ms of audio |

A refusal before the upgrade is an ordinary HTTP response with the
shared error body: `426 upgrade_required` (with `upgrade: websocket`) to
a request that is not a WebSocket handshake, `426 unsupported_version`
(with `sec-websocket-version: 13`) to another version, `400
bad_handshake` (no `connection: upgrade`, a key that is not 16 bytes in
base64, not a `GET`), `400 invalid_request` (a malformed parameter,
`param` names it), `400 not_a_transcription_model`, `404
model_not_found`, `529 busy` (16 live transcriptions run already).

**The client sends** binary messages of 16 kHz mono 16-bit
little-endian samples, of any size (a sample may straddle two messages),
and text messages, each a JSON object:

| Message | Meaning |
| --- | --- |
| `{"type": "finish"}` | the utterance ended: the audio sent before it is transcribed to its end, a `final` follows, and the next audio begins a new utterance (the recognizer's state is reset) |
| `{"type": "reset"}` | drop the utterance in progress and every sample not yet transcribed, without a `final` |
| `{"type": "language", "language": "de-DE"}` | the language prompt from the next chunk on (`auto` the default's), or a list (`"ar,en,auto"`); an unknown locale is an `error` event and the language stays |
| `{"type": "phrases", "phrases": ["Siobhan", "Dun Laoghaire"]}` | names or words to expect, from the next chunk on, until the next `phrases` (`[]` clears them); send it first to have them from the start. At most 64, each at most 48 bytes; more is an `error` event and the list stays ([phrases](#phrases-to-expect)) |

A command applies after every sample sent before it. Pings are answered
with pongs; a close is answered once what was asked before it (a
`finish`) has been answered, within 5 s.

**The server sends** text messages, each a JSON object:

```json
{"type": "ready", "model": "nemotron-asr-0.6b", "lookahead": 3, "language": "auto", "sample_rate": 16000}
{"type": "partial", "text": "Kenneth and Beth refrain from telling", "audio_seconds": 2.813}
{"type": "final", "text": "Kenneth and Beth refrain from telling the other girls or Uncle John of Old Will Rogers' visit, but they got Mister Watson in the library and questioned him closely about the penalty for forging a check.",
 "language": "en-US", "words": [{"word": "Kenneth", "start": 0.800, "end": 1.200}, …], "audio_seconds": 11.700}
{"type": "error", "code": "invalid_message", "message": "a text message is a JSON object with a \"type\""}
```

- `ready` once the model is open and the session's recognizer exists
  (audio sent before it waits, within the backlog limit).
- `partial` when a chunk changed the utterance's text: the whole text so
  far, never a delta, so a stale one is simply replaced; `audio_seconds`
  is the audio of the utterance it covers. Chunks that add nothing send
  nothing.
- `final` after `finish`: the text (equal to `nuclis transcribe --json`'s
  for the same samples, lookahead, and language), the language tag the
  model emitted (`null` if none), the words, and the utterance's length.
  Words are grouped as a file's are ([segments](#segments)): a word
  starts at a piece that begins with a space; its `start` and `end` are
  the 80 ms encoder rows its pieces were emitted at (a late punctuation
  mark's row left out), from the utterance's start, which trail the
  speech by up to the lookahead. An
  utterance with no audio has an empty final.
- `error` with a stable `code` (below); the session continues unless a
  close follows.

**Limits**, fixed by the host:

| Limit | Value | Past it |
| --- | --- | --- |
| a message (its fragments together) | 1 MiB | close `1009` |
| a text message | 4 KiB | `error` `invalid_message` |
| audio received and not yet transcribed | 10 s | `error` `audio_backlog`, close `1008` |
| events queued for a client that does not read | 1 MiB | close `1008` |
| silence: no message, ping, or pong received | 60 s | close `1008` |
| live transcriptions at once | 16 (each holds 72 MB of encoder state on the device) | `529 busy` at the handshake (`1013` if two race for the last) |
| open connections | 64, shared with every route | `529 busy` |

**Scheduling.** Each connection is one short item on the GPU worker,
which pushes one encoder chunk a run (4 rows of 80 ms at lookahead 3,
about 20–35 ms on Metal) and is queued again while audio waits, so live
transcription runs between a chat generation's steps, and several
connections take turns a chunk at a time. Audio is buffered on the
connection until a chunk is complete, whatever the message sizes: the
transcript does not depend on how the client splits its audio, or how
fast it sends it. Opening the model is the exception: a connection
whose model is not open waits for a running generation to end, since
opening may close the language model it runs on. Every connection has
its own recognizer state over the one open model, which the memory
budget never closes while a connection uses it.

**Cancellation.** A client that disconnects, with or without a close
frame, stops its session at once: nothing more of its audio is
transcribed, and its recognizer is freed at the worker's next turn. On
Ctrl-C the server closes every session with `1001`.

**Errors and closes:**

| Event `code` / close | When |
| --- | --- |
| `invalid_message` | a text message that is not one of the commands, or over 4 KiB; the session continues |
| `invalid_language` | a `language` command naming no locale of the model (the session continues), or a `language` query that does not (close `1008`) |
| `audio_backlog`, close `1008` | more than 10 s of audio waits for the GPU |
| `model_failed` | the transcription failed on the device: the utterance is dropped and the session continues; if the model did not open, close `1011` |
| `model_too_large`, close `1011` | the model needs more than the budget leaves beside the pinned models |
| `busy`, close `1013` | the memory it needs is held by a running request, or another transcription model is in use; retry |
| close `1002` | a frame RFC 6455 forbids (unmasked, a reserved bit or opcode, a control frame fragmented or over 125 bytes) |
| close `1007` | a text message that is not UTF-8 |
| close `1001` | the server is stopping |

**For a voice assistant.** Stream the microphone continuously; send
`finish` when the turn ends (the client's own voice activity detection
decides), and read the `final` about 40 ms later; the next audio starts
the next utterance with no further message. Send `reset` to drop an
utterance (a barge-in the assistant ignores). The server transcribes
what it is sent: echo cancellation, so the assistant does not hear its
own voice, is the client's.

## Transcription (files)

### `POST /v1/audio/transcriptions`

OpenAI's transcription call for a whole recording, built for long ones (a
lecture, a meeting, an hour or more): the transcript as text, JSON with
segments and words, or subtitles, at once or streamed as segments are
final. The model is Nemotron 3.5 ASR, the one `nuclis transcribe` runs,
and the route's text is the command's, byte for byte (the gate
`nemotron-asr-file-route`).

```sh
curl -s localhost:9000/v1/audio/transcriptions \
  -F file=@meeting.m4a -F response_format=verbose_json \
  -F 'timestamp_granularities[]=segment' -F 'timestamp_granularities[]=word'
curl -s localhost:9000/v1/audio/transcriptions -F file=@lecture.wav -F response_format=srt > lecture.srt
```

The upload is held in memory while it is transcribed and is limited to
256 MiB (`serve.transcription_bytes`; past it `413
payload_too_large`): two hours of 16 kHz 16-bit mono WAV (230 MB), or
many hours of a compressed format. Files queue in arrival order, one
transcribed at a time; at most 16 wait (`529 busy`), and one that waits
`serve.timeout` without starting is `529 timeout`.

### The file transcription request

`multipart/form-data`, as OpenAI's SDKs send it. Parts are bounded (at
most 32, each part's headers 1 KiB, a field's value 1 KiB), and exactly
one is a file (`phrases` may be 8 KiB).

| Field | In nuclis |
| --- | --- |
| `file` | the recording: WAV, AIFF, MP3, M4A/AAC, FLAC, ALAC, or CAF (AudioToolbox, macOS), resampled to 16 kHz and mixed to mono; a 16 kHz mono WAV, 16-bit or float, is read directly on any platform |
| `model` | a registry entry of kind `transcription` or `nemotron-asr-0.6b`; never a path. Absent: the open one, else `transcribe.model`. OpenAI's names (`whisper-1`, `gpt-4o-transcribe`) are `404 model_not_found` |
| `language` | `auto` (the default), or a locale or two-letter code the model knows (`en-US`, `de-DE`, `fr`, …; 121 keys), or a comma-separated list of up to four, `auto` only last ([several languages](#several-languages)); an unknown one is `400 invalid_request` |
| `response_format` | `json` (the default), `text`, `verbose_json`, `srt`, `vtt` |
| `timestamp_granularities[]` | `segment`, `word`, or both (repeat the field); with `verbose_json` or `stream`. Absent: segments |
| `stream` | `true`: server-sent events (below); not with `srt` or `vtt` |
| `lookahead` | nuclis's: rows of 80 ms the encoder waits for, `0`, `3` (the default), `6`, `13`; as `nuclis transcribe --lookahead` |
| `phrases` | nuclis's: a JSON array of names or words to expect, `["Siobhan", "Dun Laoghaire"]` (at most 64, each at most 48 bytes, the field 8 KiB); as `nuclis transcribe --phrase` ([phrases](#phrases-to-expect)) |

Refused with `400 unsupported_feature`: a `prompt` (the model takes no
text; `phrases` and `language` are its hints) and `include[]` (the decoder is
greedy and keeps no log probabilities). Other fields OpenAI defines are
ignored (`temperature`: decoding is greedy).

### Phrases to expect

Nemotron ASR knows no names it was not trained on, and takes no text
prompt. A client that knows the names in its user's life lists them in
`phrases`; the decoder then leans toward them: a piece that starts one
gains 2 logits over the blank and the other pieces, a piece that continues
one 6 (`docs/models/nemotron-asr.md` § Phrase boosting). Write each as it
should be spelled. Without phrases nothing changes.

Measured on three recorded utterances of a name the model misheard each
time (2026-10-28, `en-US`, r = 3, the same on both routes): two came back
spelled right; the third stayed wrong, its word never beginning with the
name's first piece. Twenty unrelated names listed on the 60-clip speech
corpus moved the word error rate from 6.11 % to 6.19 % (74 to 75 errors
in 1,211 words).

### Several languages

A user who speaks more than one language lists them: `language:
"ar,en,auto"` on either route (`nuclis transcribe --language
ar,en-US,auto`). The audio is encoded once and decoded under each; each
stretch of words up to a pause takes the language that wrote it,
because a fixed locale writes nothing for most other languages. `auto`,
only last, is the fallback: it is used when none of the others wrote, or
when it wrote the same script better (Spanish that `en` spelled in
English). The response's `language` (a final's on the live route) names
the language chosen, from its tag or, without one, the locale. Partials
may change language until the pause; segments and finals do not.

On 40 short Egyptian Arabic clips and 13 English turns, `ar,en,auto`
wrote every English turn as `auto` does and the Arabic with a mean
character error of 0.229 (`auto` 0.252, with three in another script:
Devanagari, Hangul, Latin; `ar` 0.230, with every English turn empty).
Two languages cost about 8 % over one, three with `auto` about 33 %
(0.021× real time). docs/models/nemotron-asr.md § Several languages.

### The file transcription response

- `json` (`application/json`): `{"text", "usage": {"type": "duration",
  "seconds"}}`.
- `text` (`text/plain`): the transcript and a newline.
- `verbose_json`: `{"task": "transcribe", "language", "duration",
  "text", "segments", "words", "usage"}`, `segments` and `words` as the
  granularities ask. `language` is the locale of the last language tag
  the model emitted (`en-US`, `de-DE`), or the request's, or with a list
  the one chosen, where OpenAI writes a name (`english`).
- `srt` (`text/plain`) and `vtt` (`text/vtt`): one cue per segment,
  `HH:MM:SS,mmm` and `HH:MM:SS.mmm`, numbered from 1 in SRT, under a
  `WEBVTT` header in VTT.

A segment is `{"id", "start", "end", "text", "language"}`, a word
`{"word", "start", "end"}`, times in seconds. The text is the segments'
texts joined by spaces.

With `stream=true` the response is `text/event-stream`, in OpenAI's
events, a segment at a time as each becomes final:

```
data: {"type": "transcript.text.delta", "delta": "The thrall's eyes flashed."}

data: {"type": "transcript.text.segment", "id": 0, "start": 3.6, "end": 4.96, "text": "The thrall's eyes flashed.", "language": "en-US"}

: progress 61.44 of 1802.25 s

data: {"type": "transcript.text.done", "text": "…", "usage": {"type": "duration", "seconds": 1802.25}}
```

Each final segment is a `transcript.text.delta` (its text, with a
leading space after the first: the deltas joined are the text) then a
`transcript.text.segment` (OpenAI's diarized event, with `language`
for `speaker`, and `words` when asked). A comment `: progress <done> of
<total> s` comes at most once a second, and `: transcribing` after 10 s
of silence. The head is sent once the model is open and the language
checked, so those refusals are still statuses; an error after it (the
file stops decoding) is one `data: {"error": {…}}` event before the end.
A client that closes the connection stops its transcription before the
next 2.56 s piece; a client that does not read keeps at most 64 KiB of
events queued, and the transcription waits for it rather than hold the
GPU.

### Segments

Segments come from the transcriber's tokens and their rows (80 ms each),
by one function `nuclis transcribe --segments` shares
(`src/audio/transcript.zig`). A new segment starts at a word that follows
terminal punctuation (`.`, `?`, `!`, `…`, `。`, `？`, `！`, then perhaps
closing quotes), a pause of 1.04 s or more since the last word's sound,
or 30 s into a segment. The model emits punctuation late, after the pause
it marks (up to 2.6 s after the word in the speech corpus), so times and
pauses follow the pieces holding a letter or digit, and a segment without
terminal punctuation stays open 4 s for a late comma or period. A
segment's language is the tag the model emitted at its end, else the
last before it, else the request's locale. Times are the rows that
emitted the tokens, not a forced alignment: a word's start is its first
piece's row, its end one row past its last sounding piece.

The pause was proposed at 0.8 s and measured on the speech corpus's 60
clips, each one or two sentences (703 word boundaries): at 0.8 s, 14
sentences split mid-way (5 at a comma); at 1.04 s, 4; at 2 s, none.
Mid-sentence pauses of 0.8–1 s are common in read speech, so the
constant is 1.04 s (13 rows). On the 30-minute recording below, every
join of 1.5 s or more is a segment boundary, and the pause carries much
of the work: in a continuous stream the model ends fewer sentences with
punctuation than it does a clip alone (110 of the 258 segments end in
`.`, `?`, or `!`, each followed by its language tag).

### Long recordings

The upload is decoded in pieces (`inference/src/audio/audio.zig`
`Reader`, AudioToolbox read 4,096 frames at a time) and fed to one
stream 2.56 s (32 rows) at a time, each piece a short GPU item: it runs
between a chat generation's steps and speech's chunks, never waiting for
a generation to end. Beside a generation a file takes at most a quarter
of the GPU's time (a piece runs only once three times its length has
passed since the last), since it has no deadline and the generation
waits for each piece; alone it runs back to back. Memory is the upload, the transcriber's state (on
Metal 72 MB of device buffers, the front end's pending samples), and the
segments, never the decoded samples: peak RSS rose 46 MiB from a
5-minute to a 30-minute recording whose upload is 45.7 MiB larger
(measured below). A 2-hour 16 kHz 16-bit WAV would hold its 230 MB plus
the same fixed part, about 100 MB above an idle server (an estimate from
those two, not measured). The tokens (8 bytes each) and the segments
grow with the length, a few hundred kilobytes for two hours.

Fed in 2.56 s pieces, every clip of the corpus gives the same tokens and
rows as pushed whole (60 of 60, Metal), so `nuclis transcribe` feeds
files the same way. On a 30-minute recording the piece size changes the
transcript by a few characters (20,647 to 20,651 for pieces of 8 to 64
rows): the encoder's batches round differently. The state carries from
one utterance into the next, which changes a few first words (2 % of the
words against each clip transcribed alone, measured below). A recording
that **changes language** loses words after each switch: the 30-minute
recording of the corpus's five languages, under `auto`, scored 19.0 %
WER against 6.3 % for its clips alone, the first seconds after a switch
often missing (the CPU reference drops the same words from the first
45 s of it), and two segments ran across a join. One language per
recording is what it transcribes well.

## `GET /v1/models`

OpenAI's list shape, so an OpenAI client reads the ids; the `nuclis`
object describes each model well enough to choose one, build a request
for it, or offer its pull, without the CLI. `GET /v1/models/{id}` is
OpenAI's retrieve-model: one entry alone, `404 model_not_found` for an id
the list does not hold.

Both answer with a strong `ETag` of the body; a request whose
`If-None-Match` holds it (or `W/` it, or `*`) gets an empty `304`. The tag
changes whenever the listing does: a pull, an `rm`, a registration, the
settings, or a model opening or closing (`loaded`).

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
        "schema_version": 1,
        "kind": "decision",
        "name": "laya",
        "architecture": "modernbert",
        "quantization": "F16",
        "size_bytes": 842609210,
        "source": "catalogue",
        "download_bytes": 842609210,
        "present": true,
        "loaded": false,
        "memory_bytes": 842609210,
        "memory_source": "estimate",
        "default": false,
        "family": "laya",
        "packs": true,
        "images": false,
        "max_len": 512,
        "head_max_len": 192,
        "repo": "convaiinnovations/laya",
        "revision": "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
        "endpoints": ["/v1/decisions", "/v1/systemone"],
        "pinned": false,
        "input_modalities": ["text"],
        "output_modalities": ["decision"],
        "detail": "modernbert · F16 · 842.6 MB"
      }
    }
  ]
}
```

The list holds every decision model of the catalogue and the registry,
pulled or not; every language model that is present (the ones `/model`
offers in `nuclis chat`), then the catalogue's language models not on
disk; every embedding, speech, and transcription model of the catalogue
and the registry, pulled or not.

Every kind's fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | 1; raised when a field changes incompatibly (new fields do not raise it) |
| `kind` | `decision`, `language`, `embedding`, `speech`, or `transcription` |
| `name` | the model as a person names it (below, per kind) |
| `architecture`, `quantization`, `size_bytes` | what the weights are and what they hold on disk (per kind below; null or 0 when unknown) |
| `source` | `catalogue` (a name nuclis pins) or `registry` (an entry of `nuclis.json`) |
| `download_bytes` | what `nuclis model pull <id>` fetches without `--with`/`--all`, from the catalogue's pinned sizes; null when nuclis cannot know it (a registry entry naming a file the catalogue does not pin) |
| `present` | its weights are on disk |
| `loaded` | it is open now |
| `memory_bytes`, `memory_source` | what it counts against the budget once open, and how that is known: `open`, `measured`, or `estimate` ([memory](#memory)); `memory_bytes` is null when unknown |
| `pinned` | the server keeps it open for its whole life (`serve.pin`, `--pin`) |
| `default` | a request without `model` gets it |
| `endpoints` | the routes that serve it |
| `input_modalities`, `output_modalities` | what a request may carry and what comes back: `text`, `image`, `audio` in; `text`, `audio`, `embedding`, `decision` out |
| `detail` | one line for people: architecture, quantization, and size |
| `options` | the request fields it takes, by name (below) |
| `repo`, `revision` | where its files came from; null for a local directory |

Each `options` field is an object a client can build a control or check a
request from:

| Key | Meaning |
| --- | --- |
| `type` | `enum` (one of `values`), `enum_list` (several), `number`, `integer`, `string`, `string_list`, `boolean` |
| `values` | an enum's choices |
| `min`, `max` | a number's bounds, inclusive; left out when there is none |
| `max_items`, `max_bytes` | a list's length and each item's size |
| `default` | what this server uses for a request that leaves the field out, from its settings as they are now; null when it uses nothing (no seed, no task) or the request must name it (a speech model without `speech.voice` among its speakers) |
| `by_effort` | a language model's default under another `reasoning_effort`, where it differs (Qwen3.8 at `off`: `temperature` 0.7) |

A field the model does not take is left out (`instructions` on the 0.6B);
the request formats' own fields (`messages`, `input`, `file`) are not
listed.
| `owned_by` | the repository's owner, `nuclis` for a language model, `local` for a directory |
| `created` | always 0 |

A decision model's:

| Field | Meaning |
| --- | --- |
| `name`, `architecture`, `quantization`, `size_bytes` | the id; the catalogue's (`modernbert`, `F16`, the weights with clef's backbone), null for a registry entry |
| `options` | `{}`: a decision request's fields describe the task, not the model |
| `family` | `laya` or `clef`, known before the model is pulled |
| `packs` | `true`: the states of a request, and the requests waiting with it, share one GPU pass (Laya). `false`: each state is its own pass, its cost linear in tokens (clef-flash: send one state per request with every question in it, and keep one or two requests in flight; [clef-flash.md § Time per decision](../models/clef-flash.md#time-per-decision)) |
| `images` | it reads a request's `images` (clef-flash with its projector pulled) |
| `max_len`, `head_max_len` | its sequence and question budgets in tokens; `head_max_len` is null for clef-flash (no per-question budget), and both are null for a Laya model not yet pulled |

A language model's:

| Field | Meaning |
| --- | --- |
| `name` | the catalogue's title ("Qwen3.8 27B") for a file the catalogue pins, else the GGUF's `general.name`, else the file's name |
| `architecture` | the GGUF's `general.architecture` (`qwen35`, `gemma4`, `muse-glimmer`) |
| `profile` | the prompt template it renders with (`qwen38`, `gemma4`, `gemma4_e`, `muse_glimmer`) |
| `quantization` | the catalogue's label (`UD-Q4_K_M`, `Q4_0 (QAT)`); for another file, the encoding holding most of its tensor bytes (`Q4_K`); null when the header does not say |
| `size_bytes` | the main file's size (the catalogue's for a model not on disk) |
| `default` | `engine.model` |
| `context_length` | the window it opens with (`ctx_size`) |
| `images` | its entry names a projector |
| `audio` | it hears a request's `input_audio`: its projector is on disk and carries an audio encoder (Gemma 4 E4B's does, the 12B's does not) |
| `efforts` | the reasoning levels its template renders, in the profile's names: `off` is the request's `none` (`reasoning_effort` accepts both) |
| `features` | `tools` (native function calling; every profile renders it), `hosted_tools` (`["web_search", "web_fetch"]` when the server has web, else `[]`), `reasoning` (more than one effort), `streaming`, `structured_output` (`false`: a JSON `response_format` needs constrained decoding, which nuclis has not) |
| `limits` | what a request may carry: `max_output_tokens` (16,384), `max_tools` (64), `max_images` and `max_image_bytes` (8, 16 MiB), `max_audio_clips`, `max_audio_bytes`, and `max_audio_seconds` (8, 16 MiB, 30) |
| `detail` | for a model on disk, the same as `nuclis model ls`: architecture, quantization, and size, or the file's name |
| `options` | `reasoning_effort` (its efforts; default `agent.think`, the profile's nearest), `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `repetition_penalty` (the parser's bounds; defaults the profile's for that effort after `generation.sampling` and the entry's overrides, with `by_effort`), `max_tokens` (1–16,384; default the model's `max_tokens`), `seed`, `tool_choice`, `tools` (the tool types: `function`, and `web_search`, `web_fetch` with web), `stream` |

An embedding model's:

| Field | Meaning |
| --- | --- |
| `name` | the catalogue's title ("EmbeddingGemma 2"), else the entry's name |
| `architecture`, `quantization` | `gemma-embedding2`; the catalogue's label (`Q8_0`), null for another file |
| `size_bytes` | the file's size |
| `default` | `embed.model` |
| `dimensions` | the widths it is trained for, `[768, 512, 256, 128]` |
| `modalities` | what this server embeds with it: `["text"]`, and `"image"` and `"audio"` once the projector is pulled (the same as `input_modalities`) |
| `max_tokens` | tokens per input, 8,192 |
| `tasks` | the `task` values it takes |
| `limits` | `max_inputs` (2,048), `max_tokens` (8,192), `max_audio_seconds` (30), `max_body_bytes` (32 MiB) |
| `options` | `dimensions` (default 768), `task` (no default: text as given), `title`, `encoding_format` (`float`), `truncate` (`false`), `image_tokens` (280) |

A speech or transcription model's:

| Field | Meaning |
| --- | --- |
| `name` | the catalogue's title ("Qwen3-TTS 1.7B CustomVoice"), else the entry's repository name |
| `architecture`, `quantization` | the catalogue's (`qwen3_tts`, `BF16`); for a registered speech checkpoint, read from it once pulled (`quantization` is the weights' dtype, `mixed` when they differ); null before |
| `size_bytes` | the weights and, for speech, the codec: the catalogue's sizes, else the files on disk (0 before the pull) |
| `default` | `speech.model` or `transcribe.model` |
| `voices` | speech: what a request's `voice` takes: on CustomVoice the speakers, in the checkpoint's order (`spk_id`; for a model not pulled, the catalogue's; for a registered one, empty before the pull); on a Base the saved voices' names; on VoiceDesign none |
| `speakers` | speech: the same as objects, `{"id": "eric", "dialect": "sichuan_dialect"}`: a dialect speaker speaks its dialect for `auto` and `chinese` (`dialect` null before the pull) |
| `languages` | speech: the `language` values besides `auto`, the codec's (`english`, …, and the dialects `beijing_dialect`, `sichuan_dialect` on Qwen's); transcription: the locales of its `prompt_dictionary` (121 on Nemotron, `auto` among them). Empty before the pull |
| `instructions` | speech: whether it performs a request's `instructions` (the 1.7B CustomVoice, and VoiceDesign, whose instructions describe the voice); null before the pull |
| `cloning` | speech: on a Base `{"modes": ["audio", "audio_and_transcript"], "voices_route": "/v1/audio/voices"}` (a voice from a clip alone, or a clip and its words), else null; null for a registered model not pulled |
| `design` | speech: whether `instructions` describe the voice (VoiceDesign); null for a registered model not pulled |
| `sample_rate` | 24,000 for speech output, 16,000 for transcription input |
| `formats` | speech: the `response_format` values, `["pcm", "wav"]` |
| `limits` | speech: `max_input_bytes` (16 KiB). Transcription: `max_phrases` and `max_phrase_bytes` (64, 48), `max_file_bytes` (`serve.transcription_bytes`) |
| `options` | speech: `voice` (its values the `voices`; default `speech.voice` when it is one of them, else null; absent on VoiceDesign), `language` (default `auto`), `instructions` (the 1.7B), `response_format` (`wav`), `stream_format` (`audio`), `seed` (0). Transcription, the file route's: `language` (`auto`), `response_format` (`json`), `timestamp_granularities` (`["segment"]`), `stream`, `lookahead` (`0`, `3`, `6`, `13`; default 3), `phrases` |

## `GET /v1/health`

```json
{
  "status": "ok",
  "version": "0.4.0-dev",
  "backend": "metal",
  "loaded": ["laya", "laya-multilingual"],
  "language": "qwen3.8-27b",
  "embedding": "embeddinggemma-2",
  "web": "off",
  "memory": {"limit": 34359738368, "resident": 24227472014,
             "models": [{"name": "qwen3.8-27b", "kind": "language", "bytes": 23583635232},
                        {"name": "laya-multilingual", "kind": "decision", "bytes": 643836782}]},
  "queue": {"queued": 0, "running": false, "completed": 1709},
  "decisions": {"waiting": 0, "batches": 1707, "requests": 3136},
  "embeddings": {"waiting": 0, "passes": 42, "requests": 37},
  "connections": 1,
  "in_flight": [{"id": "019a3412-6f1e-7c2a-9b4d-0e5f6a7b8c9d", "client_id": "turn-41",
                 "method": "POST", "path": "/v1/audio/speech", "running_ms": 1505.6,
                 "model": "qwen3-tts-0.6b-base", "voice_kind": "clone", "voice": "isa"}]
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
passes, and answered requests the same way. `web` is `off` when the tools
are (no `serve.web` or `--web-url`, or `--no-web`); otherwise `ok` when the search service and every provider
behind it answer its own health call, `down` when not (that call waits at
most 5 s). `in_flight` lists the requests under way, newest first: the
id, the client's id when it sent one, the method, the path, how long it has
run, and the [request log](#the-request-log) fields its route has recorded
so far (a route records most of them when it ends).

## The request log

`nuclis serve` writes two JSON lines a request to
`~/.nuclis/logs/` (under `NUCLIS_HOME` when set), whoever started it: a
`request.start` once the head is read, so a request still running is in the
file, and a `request.end` once the response is written, its stream ended,
or its WebSocket closed. `GET` and `HEAD /v1/health` are not logged
(pollers). The stdout line (`serve.log`, `--quiet`) is separate and
unchanged; `serve.request_log: false` turns the file off.

**Never a payload.** No text, message, image, clip, vector, transcript, or
audio, in or out, and no query string: sizes, counts, durations, the names
nuclis resolved (model, voice), and nuclis's own error codes and messages.

**Files.** One a UTC day, `serve-2026-10-30.jsonl`, never renamed: any
number of servers append to it safely (opened `O_APPEND`, one write a line,
a line at most 4 KiB). Past 64 MiB a day continues in
`serve-2026-10-30.1.jsonl`, `.2`, …. At start a server deletes the files
older than 14 days, then the oldest until those left hold at most 256 MiB
(never today's). Files are created `0600`. A failed write never fails a
request.

**Ids.** Every request gets a UUIDv7 (time-ordered), returned as
`x-request-id` on every response, errors, streams, and a WebSocket's `101`
included, and logged as `id`. A client's `X-Client-Request-Id` (OpenAI's
SDKs send it; printable ASCII, at most 128 bytes) is echoed back and
logged as `client_id`, so one client turn reads as one thread across
routes; one that breaks those rules is ignored, and the start line says
`client_id_refused: true`.

**A line.** An object: `time` (RFC 3339, UTC, milliseconds), `schema`
(1), `event`, `id`, then snake_case fields whose names carry their units,
numbers as numbers. A field a request did not reach is absent.

```json
{"time":"2026-10-30T09:12:04.318Z","schema":1,"event":"request.start","id":"019a3412-6f1e-7c2a-9b4d-0e5f6a7b8c9d","client_id":"turn-41","method":"POST","path":"/v1/audio/speech"}
{"time":"2026-10-30T09:12:06.904Z","schema":1,"event":"request.end","id":"019a3412-6f1e-7c2a-9b4d-0e5f6a7b8c9d","client_id":"turn-41","status":200,"duration_ms":2586.2,"bytes_in":212,"bytes_out":214080,"model":"qwen3-tts-0.6b-base","voice_kind":"clone","voice":"isa","voice_ready_ms":0.4,"format":"pcm","stream_format":"audio","frames":56,"audio_s":4.46,"first_audio_ms":431.0,"finish_reason":"eos"}
```

| Event | Fields |
| --- | --- |
| `request.start` | `client_id`, `client_id_refused`, `method`, `path` |
| `request.end`, every route | `client_id`, `status`, `duration_ms`, `bytes_in`, `bytes_out`; an error's `error` (its code) and `message` |

What each route adds to its `request.end`:

| Route | Fields |
| --- | --- |
| `POST /v1/chat/completions` | `model`, `effort`, `stream`, `prompt_tokens`, `cached_tokens`, `completion_tokens`, `finish_reason` (`stop`, `length`, `tool_calls`, `cancelled`), `web_calls`, `web_rounds`, `web_restored` |
| `POST /v1/decisions`, `POST /v1/systemone` | `model`, `states`, `questions`, `input_tokens`, `batch_requests` (the requests that shared the pass), `open_ms` (when this request opened the model) |
| `POST /v1/embeddings` | `model`, `inputs`, `input_tokens`, `passes`, `batch_requests`, `open_ms` |
| `POST /v1/audio/speech` | `model`, `voice` (none for VoiceDesign), `voice_kind` (`speakers`, `clone`, `design`), `voice_ready_ms` (a saved voice read back, or encoded on its first use with the model), `format`, `stream_format`, `frames`, `seed`, `audio_s`, `first_audio_ms`, `open_ms`, `finish_reason` (`eos`, `length`, `cancelled`); for Drama 3 also `sentences`, `redraws` (silent starts drawn again, each with the next seed), `silence_stops` (sentences its silence stop ended), `prefill_ms`, `prefix_restored` and `prefix_prefilled` (the takes whose voice prefix was kept or prefilled), `context_id` (the response's `x-speech-context`) and `context` (`hit`, `missing`, `expired`, `other_voice`, when the request named one) |
| `POST`, `GET`, `DELETE /v1/audio/voices` | `voice`; a list's `voices` (its count); a create's `clip_s` and `transcript` (whether one came) |
| `POST /v1/audio/transcriptions` | `model`, `stream`, `audio_s`, `real_time` (audio seconds per second), `segments`, `first_segment_ms`, `open_ms`, `finish_reason` (`done`, `cancelled`) |
| `GET /v1/audio/transcriptions/stream` | `model`, `lookahead`, `audio_s`, `finals`, `closed_by` (`client`, `server`, `gone`), `close_code`, and the server's `close_reason` |
| `GET /v1/models` | `models` (the count); a retrieve's `model` |

An error a stream reports after its `200` is the end line's `error` and
`message` beside the `200`. `GET /v1/health` lists the requests in flight, each with
its fields so far (`GET /v1/health` above).

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

### Speech beside a chat stream, 2026-10-27, commit `07dac34`

Apple M4 Pro (48 GB), ReleaseSafe (`make build`), Zig 0.17.0, macOS
27.0; `nuclis serve --quiet --model gemma-4-e4b-qat --pin qwen3-tts-1.7b-customvoice
--pin nemotron-asr-0.6b` on Metal (`unsloth/gemma-4-E4B-it-qat-GGUF` at
`8c5a9e4f`, speculative decoding on at draft 6;
`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` at `0c0e3051`). **The machine was
not quiet**: another session's builds held the CPU (load average 3–4.6),
and `nuclis speak --stream` of the first sentence ran at 1.65× real time
then against 2.3× on a quiet machine
([qwen3-tts.md § Speed](../models/qwen3-tts.md#speed)); the server's
speech alone ran at 1.72×, so the server adds nothing measurable, and
every rate below is low by that margin. A Python client (`http.client`
on loopback) sent five sentences of 3–5 s of audio, one after another,
each `"response_format": "pcm"`, `"seed": 1`; first byte is the first
audio byte of the body after the request was sent, the real-time factor
the audio's length over the request's wall time. The chat is one streamed
request, `"reasoning_effort": "none"`, temperature 0, 1,200 tokens of a
story; its rate is content chunks per second (a chunk is a token but for
a held split character), between the first and the last of a window.

| | First audio byte (median, range) | Real-time factor (median, range) | E4B decode |
| --- | --- | --- | --- |
| speech alone | 99 ms (87–107) | 1.72× (1.32–1.78) | — |
| E4B alone | — | — | 43.7 tokens/s |
| speech while E4B streams | 135 ms (129–152) | 1.49× (1.43–1.53) | 5.4 tokens/s during the speech, 44.3 after |

Speech beside a generation stays faster than real time and starts within
about 40 ms of alone; the generation pays: each of its steps waits for a
four-frame chunk, so it ran at an eighth of its rate while the 21.1 s of
speech were made (in 14.2 s), and at its own rate after. Audio made beside the
generation was byte-identical to the same request alone. How to share the
GPU more evenly is measured in § The voice loop below, whose policy
replaced this one.

### Live transcription alone and beside a chat stream, 2026-10-27, commit `18216d7`

Apple M4 Pro (48 GB), ReleaseSafe (`make build`), Zig 0.17.0, macOS
27.0.1; `nuclis serve --quiet --model gemma-4-e4b-qat --pin
nemotron-asr-0.6b` on Metal (`unsloth/gemma-4-E4B-it-qat-GGUF` at
`8c5a9e4f`, speculative decoding off as this machine's
`generation.speculative` sets it; `nvidia/nemotron-3.5-asr-streaming-0.6b`
at `ea30d66d`). **The machine was not quiet**: another worktree's
`nuclis serve` and `nuclis transcribe` (file transcription on the same
GPU) ran for about 80 % of each run, sampled once a second, and its
compiler for some of it. `python3 scripts/ws-transcribe.py --measure
--clips librispeech-6829-68769-0000,fleurs-de_de-1660 --beside-chat
gemma-4-e4b-qat --rounds 3`: one connection, lookahead 3, each clip under
its locale, sent in 20 ms messages each when its audio has ended at real
time (as a microphone delivers it), then `finish`; one warm-up, then each
clip three times (159 partials). A partial's delay is its arrival behind
the moment the audio it covers had ended; the final's is its arrival
after `finish` (sent with the last message). The chat is one streamed
request, `"reasoning_effort": "none"`, temperature 0, 1,200 tokens of a
story; its rate is content chunks per second between the first and the
last of a window (alone: the whole answer; beside: from the clip's first
message to its final). Two runs:

| | Partial behind the audio, median / p90 / max | Final after `finish`, median (range) | E4B decode |
| --- | --- | --- | --- |
| transcription alone | 42 / 47 / 60 ms; 29 / 31 / 33 ms | 31 ms (24–34); 20 ms (16–20) | — |
| E4B alone | — | — | 48.1; 45.8 tokens/s |
| transcription while E4B streams | 50 / 66 / 74 ms; 49 / 65 / 81 ms | 48 ms (33–60); 41 ms (33–60) | 41.8; 42.1 tokens/s during the clip (34.4–43.3 over the six), 44.7; 44.3 after |

Live transcription keeps up with the speaker beside a generation: every
partial arrived at most 81 ms after the audio it covers (medians 29 to
50 ms), every final at most 60 ms after the turn ended. The generation
pays 8 to 13 % of its rate while a clip streams (a step may wait for one
chunk of about 20–35 ms every 320 ms), against speech's eightfold
slowdown ([above](#speech-beside-a-chat-stream-2026-10-27-commit-07dac34)).
An earlier run on this branch, whose other jobs were not logged,
measured E4B at 56.4 tokens/s alone and 54.8 while a clip streamed. Both
clips are among the `nemotron-asr-stream-route` gate's four, whose
finals equal `nuclis transcribe`'s text.

### File transcription, 2026-10-27, commit `c4fe892`

Apple M4 Pro (48 GB), ReleaseSafe (`make build`), Zig 0.17.0, macOS 27.0;
measured on the same file-route code before its rebase onto the live
route (then `4390fa8`); a fresh `nuclis serve` with `nemotron-asr-0.6b`
open on Metal (`nvidia/nemotron-3.5-asr-streaming-0.6b` at `ea30d66d`), lookahead 3,
`auto`. **The machine was shared**: another session's builds and tests
held it (load average 2–7), and the same run measured 42–64× real time
from one run to the next. The recording is `scripts/transcription-check.py
long`'s: the 40 LibriSpeech clips of the speech corpus in a seeded order,
each followed by 1.5–3 s of silence, 16-bit 16 kHz mono WAV (30 minutes:
1,802.3 s, 55.0 MiB). Rate is the audio's length over the request's
wall time (the upload over loopback included); first segment is the
time from sending the request to the first `transcript.text.segment`
event with `stream=true`; RSS is `ps`'s, sampled every 50 ms.

| | 5 minutes | 30 minutes |
| --- | --- | --- |
| rate, alone (runs) | 42.7–62.5× (3) | 42.4–63.8× (6) |
| first segment, `stream=true` | 0.17–0.26 s | 0.20–0.29 s |
| peak RSS (idle 212–217 MiB) | 322–327 MiB | 368–374 MiB |
| segments; words differing from the clips alone | 44; 2.18 % | 258; 1.98 % |
| WER against the references (the clips alone) | 5.47 % (4.06 %) | 5.04 % (4.08 %) |

Beside a chat: the same server with `gemma-4-e4b-qat` open too, one
streamed chat (1,200 tokens of a story, temperature 0, `reasoning_effort`
`none`) started 2 s before the 30-minute file and running through it,
then the same chat alone. Its rate is content chunks per second, as in
the speech measurement.

| | Transcription | E4B decode |
| --- | --- | --- |
| alone | 42–64× real time | 45.8–51.5 tokens/s |
| together | 28.4–40.9× real time | 12.0–17.9 tokens/s |

The transcript beside the chat equals the one alone. The chat pays as
it does for speech: each of its steps waits for a 2.56 s piece, 40–64 ms
at the rates above. Smaller pieces cost nearly as much each: through
`nuclis transcribe` on the same machine and file, pieces of 8, 16, 32,
and 64 rows ran at 19, 25, 40, and 68× real time, so a piece of 8 rows
still took about 34 ms (each emitted token adds a command buffer,
[nemotron-asr.md § On Metal](../models/nemotron-asr.md#on-metal-2026-10-27)). § The voice loop below measures the share
that replaced this.

### The voice loop, 2026-10-27, commit `f96f750`

Apple M4 Pro (48 GB), ReleaseSafe (`make build`), Zig 0.17.0, macOS
27.0, a quiet machine (no other GPU job; load average 3–4 from the
desktop); `nuclis serve --quiet --model gemma-4-e4b-qat --pin <voice>
--pin nemotron-asr-0.6b` on Metal, through `scripts/voice-loop.py` and
`scripts/ws-transcribe.py --measure`, `scripts/transcription-check.py
long --chat`. Chat rates are streamed content chunks per second as in
the records above (`reasoning_effort` `none`, temperature 0, 1,200
tokens of a story). Playback is modelled: audio plays at real time from
its first byte, each speech request after the one before it; the lead is
the audio in hand when a chunk arrives, an underrun a chunk that arrives
after it ran out.

**Beside a chat.** Five sentences (28.1 s of audio on the 1.7B, 30.8 s on
the 0.6B), each its own request, sent together while the chat streams;
two rounds each.

| | E4B alone | E4B beside it | It beside E4B |
| --- | --- | --- | --- |
| speech, 1.7B, before (1 frame then chunks of 4, every run) | 55.9 tokens/s | 7.1–7.3 tokens/s | 2.06–2.09× real time; a 50–80 ms underrun after the first 80 ms of each utterance, alone too |
| speech, 1.7B | 53.9–56.4 tokens/s | 23.7–26.8 tokens/s | real time with a second in hand (lead median 2.0 s, min 40 ms), no underrun; first byte 96–118 ms |
| speech, 0.6B | 55.3 tokens/s | 32.5–33.6 tokens/s | the same, no underrun; first byte 45–103 ms |
| live transcription (2 clips × 3 rounds at real time) | 55.2 tokens/s | 52.2 tokens/s (48.1–53.1 per clip) | partials 41 ms behind the audio (median; p90 54, max 61), final 33 ms after `finish` (alone 53 and 37) |
| a 5-minute file | 52.5 tokens/s | 40.6 tokens/s | 12.1× real time (2.18 % of words differ from the clips alone, as before) |

Speech needs what playback needs: one frame of the 1.7B takes about
37 ms a run for 80 ms of audio, 46 % of the GPU, so E4B keeps about half
its rate while the 1.7B speaks (the proposed "at most 25 % slower" is out
of reach for it; the 0.6B leaves 60 %). Two frames a run gave E4B no more
(23.6–25.1 tokens/s) and a thinner lead. The chat runs at its own rate
again once the speech is made.

**A spoken question answered aloud.** The question ("What is the
tallest mountain in Africa, and how long does it take to climb it?")
spoken once by the server's voice, resampled to 16 kHz, streamed at real
time to the live route with 0.4 s of silence after it, then `finish`; the
final transcript to chat (a system prompt asking for three or four
spoken sentences); each sentence of the streamed answer to speech as
soon as it ends. Times after `finish`, three rounds:

| | 1.7B voice | 0.6B voice |
| --- | --- | --- |
| final transcript | 27 ms (19–29) | 25 ms (23–27) |
| first token | 287 ms (175–287) | 287 ms (167–297) |
| first sentence | 387 ms (285–388) | 388 ms (275–393) |
| first audio byte | 523 ms (418–524) | 482 ms (364–482) |
| E4B over the answer (4 sentences) | 20.0–20.2 tokens/s | 26.9–27.4 tokens/s |
| playback | 15.6 s, no underrun, lead min 40 ms | 22.5 s, no underrun, lead min 50 ms |

The transcript was the question word for word. Without the silence the
recognizer at lookahead 3 drops the last word ("it"), as `nuclis
transcribe` does on the same file: a clip that ends within 100 ms of its
last word leaves the encoder no right context, which a turn that ends in
silence always has. The first speech request after the server starts is
about 90 ms slower than the next (158 against 71 ms to the first byte).
