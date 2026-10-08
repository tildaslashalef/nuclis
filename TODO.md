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

Agreed 2026-10-08, from Seshat's client feedback: language models in
`GET /v1/models` carry structured fields, so a client can show a proper
name and size; a token-count route was declined. In progress: **model
fields**, on branch `model-fields`.

| Unit | Branch | What |
| --- | --- | --- |
| Model fields | `model-fields` | `name`, `architecture`, `profile`, `quantization`, `size_bytes` on each language model in `GET /v1/models` |

## Model fields

Base: `c7a64e9`

**Why.** A language model's `nuclis` object has only `detail`, free text
that is a file name for a file outside the catalogue, so a client cannot
print a proper name or size.

1. **Catalogue titles.** `catalog.Entry` gains `title` (a person's name
   for the checkpoint: "Qwen3.8 27B", "Gemma 4 12B QAT", "Gemma 4
   26B-A4B", "Gemma 4 E4B QAT", "Muse Glimmer 30B"); a test holds every
   language entry to a non-empty one.
2. **The fields** (`src/api/chat/service.zig` `list`), per runnable
   language model, resolved through its file: a registry entry whose
   repo and file the catalogue pins, or a catalogue name, takes the
   catalogue's `title`, `architecture`, and `quantization`; any other
   file reads its GGUF header (`inference.gguf.open`): `general.name`
   (else the file's name without `.gguf`), `general.architecture`, and
   the encoding holding the most tensor bytes as `quantization`. Also
   `profile` (the prompt profile it renders with, as `/model` resolves
   it) and `size_bytes` (the main file's size). `detail` stays. Measured
   2026-10-08: a header read is 10 to 30 ms, so no cache.
3. **Documents.** `docs/guide/api.md` § `GET /v1/models` (the language
   model's fields), `docs/models/catalogue.md` if it lists the entry's
   fields.

**Gates.** `zig build test`, `make verify-auto`; with the built binary,
`GET /v1/models` shows the fields for a catalogue model and for a file
outside the catalogue (Ternary Bonsai 2, the uncensored Gemma 4 12B).
