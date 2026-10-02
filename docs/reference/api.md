# The nuclis API

`nuclis serve` runs the nuclis API: an HTTP/1.1 server on this machine
that keeps decision models open and answers typed questions about text
or JSON. It speaks TypeSafe's Jev protocol, so a client written for
`api.typesafe.ai` switches by changing its base URL. This page is the
reference for client authors; it assumes nothing about the code.

Decisions are the first service. An OpenAI-compatible service for the
language models (`/v1/chat/completions`) is planned beside them on the
same server and is not served yet.

- [Running the server](#running-the-server)
- [Conventions](#conventions)
- [Errors](#errors)
- [Decisions](#decisions): [`POST /v1/systemone`](#post-v1systemone),
  [`POST /v1/decisions`](#post-v1decisions), [questions and
  answers](#questions-and-answers), [batching](#batching)
- [`GET /v1/models`](#get-v1models), [`GET /v1/health`](#get-v1health)
- [Measured rates](#measured-rates)

## Running the server

```sh
nuclis model pull laya                 # once: the default decision model
nuclis serve                           # http://127.0.0.1:8000/v1, decide.model open
```

| Option | Meaning |
| --- | --- |
| `--host <ip>` | the address to listen on: an IP literal (`127.0.0.1`, `::1`, `0.0.0.0`) or `localhost`; default `serve.host`, `127.0.0.1`. Any address beyond loopback prints a warning: there is no authentication. |
| `--port <n>` | default `serve.port`, `8000` |
| `--model <name>` | a decision model opened before the server listens (at most 2); without one, `decide.model` opens; others open on their first request |
| `--backend cpu\|metal` | where the models run; default `metal` in a Metal build |
| `--quiet` | no request log (`serve.log: false` does the same) |
| `--timeout <s>` | how long a request may wait for the GPU before `529 timeout`; default `serve.timeout`, `300` |

The configuration file (`~/.nuclis/nuclis.json`) holds the defaults:

```json
"decide": { "model": "laya" },
"serve":  { "host": "127.0.0.1", "port": 8000, "log": true, "timeout": 300 }
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

`clef-flash` also reads images: a request's `"images"` is a list of data
URLs (`data:image/png;base64,…`) or bare base64, at most 8, each read
before every state (`422 images_unsupported` from Laya, or from
`clef-flash` without its projector pulled; `422 invalid_image` for bytes
that are not an image). The body limit (4 MiB) bounds them. Its answers
are `systemone`'s own: uncalibrated, `confidence` the top probability. A
clef decision is a 9B prefill, 2.3 s for a 637-token state
([clef.md § Time per decision](clef.md#time-per-decision)).

Ctrl-C stops the server. It reads no files on a client's behalf and
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
| request body | 4 MiB | `413 payload_too_large`, connection closed |
| open connections | 64 | `529 busy`, connection closed |
| decision requests waiting for the GPU | 64 | `529 busy` |
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
{"error": {"code": "model_not_found", "message": "jev-2: not pulled yet (…)"}}
```

`code` is stable and meant for branching; `message` is for people. The
statuses follow TypeSafe's: `422` for a body that fails validation, `529`
when the server is overloaded (retry with backoff, as TypeSafe's SDKs do
by default).

| Status | Code | When |
| --- | --- | --- |
| 400 | `bad_request` | not HTTP: a malformed request line or header, an unknown method; the connection closes |
| 404 | `not_found` | no route at that path |
| 405 | `method_not_allowed` | the path exists, the method does not (`HEAD` is answered as `GET`) |
| 413 | `payload_too_large` | the body exceeds 4 MiB |
| 415 | `unsupported_encoding` | a compressed body (`content-encoding`) |
| 417 | `bad_request` | an `expect` other than `100-continue` |
| 422 | `invalid_json` | the body is not JSON |
| 422 | `invalid_request` | the JSON is not a valid request: a missing or extra field, a malformed question, `states` sent to `systemone`, a text that cannot be tokenized |
| 422 | `file_state_refused` | a state of the form `{"file": path}`; send the text |
| 422 | `request_too_large` | past a count or size limit above |
| 422 | `options_exceed_budget` | a question's options do not fit the model's budget |
| 422 | `model_not_found` | `model` names nothing that is pulled |
| 422 | `not_a_decision_model` | `model` names a language model |
| 431 | `headers_too_large` | the request line and headers exceed 16 KiB |
| 500 | `model_failed` | the model could not be opened |
| 500 | `internal`, `invalid_registry_entry` | a fault on the server's side |
| 503 | `shutting_down` | the server is stopping |
| 529 | `busy` | 64 decision requests wait already, or 64 connections are open |
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
curl -s localhost:8000/v1/systemone -d '{
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
curl -s 'localhost:8000/v1/decisions?explain=1' -d '{
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
or not. `present`: its weights are on disk; `loaded`: it is open now;
`default`: a request without `model` gets it; `max_len` and
`head_max_len`: its sequence and question budgets in tokens (null when
not pulled); `owned_by`: the repository's owner, `local` for a directory;
`created` is always 0.

## `GET /v1/health`

```json
{
  "status": "ok",
  "version": "0.4.0-dev",
  "backend": "metal",
  "loaded": ["laya", "laya-multilingual"],
  "queue": {"queued": 0, "running": false, "completed": 1709},
  "decisions": {"waiting": 0, "batches": 1707, "requests": 3136},
  "connections": 1
}
```

`queue` is the GPU's (`completed` counts GPU items, a batch being one);
`decisions.waiting` is decision requests not yet in a pass,
`batches` and `requests` the passes run and the requests they answered
since the start.

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
[laya.md § Time per call](laya.md#time-per-call) measures, about 3,700
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
