<!-- kind: guide -->
# Decisions

The decision routes of [the nuclis API](api.md), which covers running
the server, its limits, and its errors. They speak TypeSafe's Jev
protocol, so a client written for `api.typesafe.ai` switches by changing
its base URL.

A decision model answers typed questions about a **state** (text, or any
JSON: an object, a record, a conversation as a list). There are three
question types: `noul` (yes or no, as a probability), `choice` (one
option of several), and `score` (a value along ordered levels). Each
question is answered on its own; nothing is generated.

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
([clef-flash.md § Time per decision](../models/clef-flash.md#numbers)).

## `POST /v1/systemone`

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

## `POST /v1/decisions`

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

## Questions and answers

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

## Batching

Requests run on the GPU one pass at a time. Decision requests that wait
together share a pass: when the GPU frees, the server takes the oldest
waiting request's model and every request waiting for that model, in
arrival order, as long as their sequences fit one pass (2,048 tokens; a
request is never split, and the first always runs). There is no waiting
window, so a lone request starts at once. Requests for another model wait
at most one pass. A sequence's answer does not depend on what shares its
pass: batching changes timing, never answers.

