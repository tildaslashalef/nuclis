<!-- kind: guide -->
# The nuclis API

`nuclis serve` runs the nuclis API: an HTTP/1.1 server on this machine.
Each service follows a format clients already speak, so a client written
for it switches by changing its base URL:

| Service | Routes | Format | Guide |
| --- | --- | --- | --- |
| Decisions: typed questions about text or JSON | `POST /v1/systemone`, `POST /v1/decisions` | TypeSafe's Jev | [api-decisions.md](api-decisions.md) |
| Chat Completions: the language models | `POST /v1/chat/completions` | OpenAI's | [api-chat.md](api-chat.md) |
| Embeddings: text, images and audio into vectors | `POST /v1/embeddings` | OpenAI's | [api-embeddings.md](api-embeddings.md) |
| Speech: text spoken, streamed as it is made | `POST /v1/audio/speech`, `/v1/audio/voices` | OpenAI's | [api-speech.md](api-speech.md) |
| Transcription: live from a microphone, or a whole file | `GET /v1/audio/transcriptions/stream`, `POST /v1/audio/transcriptions` | a WebSocket; OpenAI's | [api-transcription.md](api-transcription.md) |
| The models on offer | `GET /v1/models` | OpenAI's | [api-models.md](api-models.md) |

This page covers what every route shares: [running the
server](#running-the-server), [memory](#memory), [who may use
it](#who-may-use-the-server), [conventions](#conventions),
[errors](#errors), [`GET /v1/health`](#get-v1health), and [the request
log](#the-request-log). The guides are the reference for client authors;
they assume nothing about the code. What the server was measured to
sustain is [the record](../benchmarks/record/api.md).

## Running the server

```sh
nuclis model pull laya                 # once: the default decision model
nuclis serve                           # http://127.0.0.1:9000/v1, nothing open yet
```

| Option | Meaning |
| --- | --- |
| `--host <ip>` | the address to listen on: an IP literal (`127.0.0.1`, `::1`, `0.0.0.0`) or `localhost`; default `serve.host`, `127.0.0.1`. Any address beyond loopback prints a warning: there is no authentication ([trust](#who-may-use-the-server)). |
| `--port <n>` | default `serve.port`, `9000` |
| `--model <name>` | a model opened before the server listens, of any kind: a decision, language, embedding, speech, or transcription model by registry or catalogue name, or a decision checkpoint's directory; repeated, at most 2 decision, 1 language, 1 embedding, 1 speech, and 1 transcription model (what the server keeps open). Without it nothing opens at start: each model opens on its first request (`decide.model` for a decision request that names none, `embed.model` for an embeddings request, `speech.model` for a speech request) |
| `--pin <name>` | a model opened at start, as `--model`, that the memory budget never closes ([memory](#memory)); repeated, at most 8; replaces `serve.pin` |
| `--memory <GiB>` | what every open model may hold together; default `serve.memory_bytes`, else physical memory less a third, at most 16 GiB ([memory](#memory)) |
| `--backend cpu\|metal` | where the models run; default `metal` in a Metal build |
| `--quiet` | no line per request on stdout (`serve.log: false` does the same); the [request log](#the-request-log) file is unaffected |
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
`GET /v1/health` marks each resident model's `pinned`, and `in_use` while a
request holds it (a chat request holds its model across every hosted web
round).

**Before a model opens**, `GET /v1/models` says what it counts once open in
each entry's `memory_bytes`, with `memory_source` saying how it knows:

| `memory_source` | `memory_bytes` |
| --- | --- |
| `open` | what it counts now, the number `/v1/health` reports |
| `measured` | what it counted the last time it opened, with the same files and, for a language model, the same `ctx_size`, `kv_precision`, `cache.memory_bytes`, and speculative decoding; kept in `~/.nuclis/cache/memory.json` across starts |
| `estimate` | what it counts before it opens. Exact for an embedding and a decision model; for a speech or transcription model its files, under the count by the device memory (none for Drama 3 and Qwen3-TTS, 49 MiB for Nemotron ASR); for a language model its files and `cache.memory_bytes`, under the count by its attention cache and draft file |

A model opened once on this machine is `measured`, exact until its
files or those settings change. How close the estimate comes: [record, 2026-10-30](../benchmarks/record/api.md#memory-counted-against-the-estimate-2026-10-30).

A model refused at start (`--model`, `--pin`) says the sizes, the
needed and the free beside the models pinned before it:

```text
error: ModelTooLarge: the speech model drama-3 needs 19.4 GiB; 16.6 of 40.0 GiB is free beside the pinned models (--memory, serve.memory_bytes)
```

The pins open in the order decision, language, embedding, speech,
transcription, so a later kind is the one refused; a limit that fits
them all is at least the sum of their `memory_bytes`.

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

### Who may use the server

`nuclis serve` has no authentication: it listens on loopback by default,
and any client that can connect may call every route and name any model
of the registry or the catalogue (never a path). That is the contract,
not an oversight: bind it beyond loopback only on a network you trust,
and put a reverse proxy that authenticates in front of it for anything
else.

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
| the request line and headers to arrive, from the connection's start or the previous response (an idle keep-alive connection included) | 10 s | connection closed, nothing sent; the log says `408` |
| a pause inside a request body | 30 s | connection closed, nothing sent |
| one write to a client that does not read (a response, a stream's chunk, a WebSocket frame) | 30 s | connection closed; a WebSocket session ends without a close frame, which a client that stopped reading could not take |
| decision requests waiting for the GPU | 64 | `529 busy` |
| embedding requests waiting for the GPU | 64 | `529 busy` |
| file transcriptions waiting for the GPU (each holds its upload) | 16 | `529 busy` |
| inputs per embedding request | 2,048 | `400 request_too_large` |
| text per embedding input | 1 MiB | `400 request_too_large` |
| wait before a request starts on the GPU | 300 s (`serve.timeout`) | `529 timeout` (a started pass always finishes) |
| decision states and questions per request | the block below | `422 request_too_large` |
| options per question | 255 | `422 request_too_large` |
| bytes per state | 1 MiB | `422 request_too_large` |

The host constants a request meets, read from the code (the agent's are
beside them; the server's are the `server` and `decision` rows):

<!-- gen:limits -->
| Limit | Value | Declared in |
| --- | --- | --- |
| context ceiling (tokens) | 32,768 | inference/src/runtime/session.zig `max_capacity` |
| output budget (tokens) | 16,384 | src/config.zig `max_output_tokens` |
| draft length | 15 | inference/src/engine.zig `max_draft_length` |
| prefill chunk (tokens) | 256 | inference/src/engine.zig `prefill_chunk` |
| config file (bytes) | 65,536 | src/config.zig `max_file_bytes` |
| agent steps per turn | 16 | src/agent/loop.zig `budget_default` |
| agent turns per session | 256 | src/agent/loop.zig `max_turns` |
| agent instructions file (bytes) | 8,192 | src/agent/system_prompt.zig `max_instructions_bytes` |
| read_file lines per call | 2,000 | src/agent/tools/read_file.zig `max_lines` |
| read_file default lines | 120 | src/agent/tools/read_file.zig `default_lines` |
| read_file bytes | 1,048,576 | src/agent/tools/read_file.zig `max_bytes` |
| grep matches shown | 50 | src/agent/tools/grep.zig `max_shown` |
| grep files scanned | 20,000 | src/agent/tools/grep.zig `max_files` |
| grep seconds | 10 | src/agent/tools/grep.zig `max_seconds` |
| glob results | 200 | src/agent/tools/glob.zig `max_results` |
| image tokens per image | 4,096 | inference/src/vision/root.zig `max_image_tokens` |
| server output tokens | 16,384 | src/api/chat/wire.zig `max_output_tokens` |
| server audio clip (bytes) | 16,777,216 | src/api/chat/wire.zig `max_audio_bytes` |
| server web rounds | 6 | src/api/chat/service.zig `max_web_rounds` |
| server speech input (bytes) | 16,384 | src/api/speech/wire.zig `max_input_bytes` |
| server live transcriptions | 16 | src/api/transcription/stream.zig `max_sessions` |
| decision questions per call | 32 | inference/src/decide.zig `max_questions` |
| decision states per call | 64 | inference/src/decide.zig `max_states` |
<!-- /gen -->

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
| 417 | `bad_request` | an `expect` other than `100-continue`, on any method; the connection closes |
| 422 | `invalid_json` | the body is not JSON |
| 422 | `invalid_request` | the JSON is not a valid request: a missing or extra field, a malformed question, `states` sent to `systemone`, a text that cannot be tokenized |
| 422 | `file_state_refused` | a state of the form `{"file": path}`; send the text |
| 422 | `request_too_large` | past a count or size limit above |
| 422 | `options_exceed_budget` | a question's options do not fit the model's budget |
| 422 | `model_not_found` | `model` names nothing that is pulled |
| 422 | `not_a_decision_model` | `model` names a language model |
| 422 | `model_too_large` | the decision model alone needs more than the memory budget ([memory](#memory)) |
| 400 | `invalid_json`, `invalid_request` | chat: the body is not JSON, or a field is malformed (`param` names it) |
| 400 | `unsupported_feature` | chat: a field nuclis cannot honour ([what is refused](api-chat.md#what-is-refused)) |
| 400 | `not_a_language_model` | chat: `model` names a decision, embedding, speech, or transcription model |
| 400 | `images_unsupported` | chat: an image for a model whose projector is not pulled |
| 400 | `audio_unsupported` | chat: an `input_audio` part for a model that does not hear (only Gemma 4 E4B does), or whose projector is not pulled |
| 400 | `invalid_audio` | chat: an `input_audio` part that does not decode |
| 400 | `input_too_long` | chat: a clip over 30 s (never cut) |
| 400 | `context_length_exceeded` | chat: the rendered conversation does not fit the model's window |
| 404 | `model_not_found` | chat: `model` names no pulled language model |
| 400 | `model_too_large` | chat: the language model alone needs more than the memory budget, with the sizes |
| 400 | `invalid_json`, `invalid_request` | embeddings: the body is not JSON, or a field is malformed (`param` names it) |
| 400 | `unsupported_feature` | embeddings: token arrays, an image by URL, an audio format nuclis does not name, media without the projector pulled, a width other than 768, 512, 256, or 128 ([what is refused](api-embeddings.md#what-an-embedding-request-may-not-carry)) |
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
| 529 | `busy` | 64 requests wait already, 64 connections are open, or the memory a model needs is held by a running request (retry shortly) |
| 529 | `timeout` | the request waited `serve.timeout` (300 s) without reaching the GPU |

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
`bytes`, most recently used first; `pinned` when `--pin` keeps it,
`in_use` while a request holds it, so the budget cannot close it). `queue` is the GPU's
(`completed` counts GPU items, a batch or a completion being one);
`decisions.waiting` is decision requests waiting for a pass,
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
| `POST /v1/chat/completions` | `model`, `effort`, `stream`, `prompt_tokens`, `cached_tokens`, `completion_tokens`, `finish_reason` (`stop`, `length`, `tool_calls`, `no_answer`, `cancelled`), `web_calls`, `web_rounds`, `web_restored` |
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

**A panic.** A panic means the server's state cannot be trusted, so the
process ends: every connection drops, and the models it held close. Its
last line is a `server.panic`: `id`, the request the panicking thread was
serving (null when it was the GPU worker's), `in_flight`, the id of every
request still running, and `message`, the panic's. Run `nuclis serve`
under a supervisor that restarts it (launchd's `KeepAlive`, systemd's
`Restart=on-failure`); a client sees its connection close without a
response and retries.

```json
{"time":"2026-11-12T09:00:33.688Z","schema":1,"event":"server.panic","id":"01a1cefc-c218-7957-80b5-152e361eb8ab","in_flight":["01a1cefc-c218-7957-80b5-152e361eb8ab"],"message":"integer overflow"}
```

