<!-- kind: guide -->
# `GET /v1/models`

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
offers in `nuclis agent`), then the catalogue's language models not on
disk; every embedding, speech, and transcription model of the catalogue
and the registry, pulled or not.

Beside `id` (the name a request sends) and `object`, an entry carries
`created`, always 0 (OpenAI's schema requires it; nuclis has no creation
time), and `owned_by`: the repository's owner, `nuclis` for a language
model, `local` for a directory. The rest is under `nuclis`.

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
| `memory_bytes`, `memory_source` | what it counts against the budget once open, and how that is known: `open`, `measured`, or `estimate` ([memory](api.md#memory)); `memory_bytes` is null when unknown |
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

A decision model's:

| Field | Meaning |
| --- | --- |
| `name`, `architecture`, `quantization`, `size_bytes` | the id; the catalogue's (`modernbert`, `F16`, the weights with clef's backbone), null for a registry entry |
| `options` | `{}`: a decision request's fields describe the task, not the model |
| `family` | `laya` or `clef`, known before the model is pulled |
| `packs` | `true`: the states of a request, and the requests waiting with it, share one GPU pass (Laya). `false`: each state is its own pass, its cost linear in tokens (clef-flash: send one state per request with every question in it, and keep one or two requests in flight; [clef-flash.md § Time per decision](../models/clef-flash.md#numbers)) |
| `images` | it reads a request's `images` (clef-flash with its projector pulled) |
| `max_len`, `head_max_len` | its sequence and question budgets in tokens; `head_max_len` is null for clef-flash (no per-question budget), and both are null for a Laya model not pulled |

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
| `features` | `tools` (native function calling; every profile renders it), `hosted_tools` (`["web_search", "web_fetch"]` when the server has web, with `"web_context"` when its search service offers it, else `[]`), `reasoning` (more than one effort), `streaming`, `structured_output` (`false`: a JSON `response_format` needs constrained decoding, which nuclis has not) |
| `limits` | what a request may carry: `max_output_tokens` (16,384), `max_tools` (64), `max_images` and `max_image_bytes` (8, 16 MiB), `max_audio_clips`, `max_audio_bytes`, and `max_audio_seconds` (8, 16 MiB, 30) |
| `detail` | for a model on disk, the same as `nuclis model ls`: architecture, quantization, and size, or the file's name |
| `options` | `reasoning_effort` (its efforts; default `agent.think`, the profile's nearest), `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `repetition_penalty` (the parser's bounds; defaults the profile's for that effort after `generation.sampling` and the entry's overrides, with `by_effort`), `max_tokens` (1–16,384; default the model's `max_tokens`), `seed`, `tool_choice`, `tools` (the tool types: `function`, and `web_search`, `web_fetch`, `web_context` with web), `stream` |

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

