<!-- kind: guide -->
# nuclis documentation

nuclis is a local inference engine for Apple silicon with a CLI, a small
coding agent, and decision models: an engine library under `inference/`,
an executable under `src/`. Every document is listed here, one folder per
kind; a new document goes in the folder whose rule it fits and gets a line
below.

## Three kinds of document

Each document states its kind on its first line (`<!-- kind: reference -->`),
and `make docs-check` holds each kind to its rules.

| Kind | Promise | Where | Tense | Budget |
| --- | --- | --- | --- | ---: |
| `reference` | true at this commit | `engine/`, `app/`, `spec.md`, `architecture*.md`, `models/catalogue.md` | present | 400 lines |
| `card` | a model family's reference, one template | `models/<family>.md` | present | 200 lines |
| `guide` | commands that work at this commit | `guide/`, `development.md`, `README.md` (this index), `models/README.md`, `benchmarks/README.md`, `../README.md` (180) | imperative | 500 lines |
| `record` | true on its date, never edited after | `benchmarks/history.md`, `benchmarks/record/`, `benchmarks/llama-cpp.md` | past, dated | none |

A reference, card, or guide uses no plan words about the project (`will`,
`not yet`, `planned`, `future`, `for now`, `phase N`, `step N`, `TODO`)
outside code and quotes; a true use carries `<!-- ok: reason -->` on its
line. Every `make` target, `scripts/` path, `nuclis` command and flag, gate,
and workload it names in a code span exists. Every section heading of a
record carries its date. Budgets count lines outside generated blocks; a
file that cannot fit is two subjects. A reference links the record entry
for the why instead of retelling it. [llm-guide.md](llm-guide.md), the
maintainer's teaching companion, keeps its own shape.

## Generated blocks

What the code already knows is written once, between `<!-- gen:NAME -->`
and `<!-- /gen -->`, by `make docs-gen`; a block is never edited by hand,
and `make docs-check` fails when one is out of date. The blocks render from
[facts.json](facts.json) (the help pages, `config init`'s file, and the
source tables of encodings, profiles, limits, and `NUCLIS_*` variables,
collected by `scripts/repo/docs-gen.py`) and from the committed manifests
(`src/catalog.json`, `gates.json`, `workloads.json`). `make check` fails
when `facts.json` is out of date with the code, so a new flag, key, limit,
or encoding that forgets `make docs-gen` does not pass.

| Block | Source |
| --- | --- |
| `cli-commands`, `cli-synopsis`, `cli-synopsis:CMD` | `nuclis --help`, `nuclis CMD --help` |
| `config-keys`, `config-init-example` | `nuclis config --help`, `nuclis config init` in an empty home |
| `catalogue`, `catalogue:KIND` | `src/catalog.json` |
| `gates-by-tier`, `gates-for:FAMILY` | `gates.json` |
| `workloads` | `workloads.json` |
| `encodings`, `encodings-count` | `inference/src/tensor/encoding.zig` `layout` |
| `profiles` | `inference/src/profiles/` `template_sha256` |
| `limits` | the host constants `docs-gen.py` names, in their files |
| `env` | every `NUCLIS_*` the code reads |

## Who owns which fact

Each recurring fact has one owning document; others link it or carry its
generated block.

| Fact | Owner |
| --- | --- |
| Commands and flags | [spec.md § 6](spec.md) (`cli-synopsis`) |
| Settings, defaults, and bounds | [guide/configuration.md](guide/configuration.md) (`config-keys`) |
| Host limits | [engine/sampling.md](engine/sampling.md) (`limits`) |
| The model index | [models/catalogue.md](models/catalogue.md) (`catalogue`) |
| Gates and tiers | [development.md § Gates](development.md#gates) (`gates-by-tier`) |
| Encodings | [engine/quantization.md](engine/quantization.md) (`encodings`) |
| Prompt profiles | [engine/prompt-profile.md](engine/prompt-profile.md) (`profiles`) |
| How a change is validated | [models/README.md § 4](models/README.md#4-validate-against-nuclis-itself) |
| Environment variables | [app/terminal.md](app/terminal.md) (`env`) |

## Start here

| Document | Purpose |
| --- | --- |
| [../README.md](../README.md) | What nuclis is, the quick start, the models, the results |
| [guide/getting-started.md](guide/getting-started.md) | From nothing to a chat with a local model |
| [architecture.md](architecture.md) | The map: the engine and the app in twelve short sections with diagrams |
| [architecture-paths.md](architecture-paths.md) | The decision path, the embedding path, and the API layer |
| [architecture-rules.md](architecture-rules.md) | How correctness is established, the Zig constructs the code relies on, adding a model, structural facts |
| [spec.md](spec.md) | **The authoritative spec**: requirements, scope, acceptance criteria |
| [llm-guide.md](llm-guide.md) | The inference stack's concepts, taught as they are implemented here |
| [development.md](development.md) | Contributing: toolchain, build and test, gates, the record, CI, releases |
| [guide/code.md](guide/code.md) | The toolchain, the build and format contract, Python lint, install, testing, hostile input |
| [guide/gates.md](guide/gates.md) | How gates are defined, skipped, selected, remembered, and costed, and what each protects |
| [guide/ci.md](guide/ci.md) | The CI and release workflows |
| [Merged pull requests](https://github.com/tildaslashalef/nuclis/pulls?q=is%3Apr+is%3Amerged) | Every change since v0.6.0, each with its record: what and why, the evidence, what remains |
| [The worklog, archived](https://github.com/tildaslashalef/nuclis/blob/v0.6.0/docs/worklog.md) | Work before v0.6.0 is recorded in the worklog archived at the `v0.6.0` tag |

## Using nuclis: `guide/`

| Document | Purpose |
| --- | --- |
| [getting-started.md](guide/getting-started.md) | Install, a first run, where nuclis keeps things, model downloads |
| [configuration.md](guide/configuration.md) | `nuclis.json`: every setting, its scope, and what overrides it |
| [api.md](guide/api.md) | `nuclis serve`: the HTTP API: decisions, chat completions, embeddings, speech, and transcription |
| [api-chat.md](guide/api-chat.md) | Chat Completions: the request, what is refused, streaming, the cache, agents, hosted web tools |
| [api-decisions.md](guide/api-decisions.md) | The decision routes: Jev's `/v1/systemone` and `/v1/decisions`, naming the model, images, batching |
| [api-embeddings.md](guide/api-embeddings.md) | Embeddings: the request and response, passes, memory |
| [api-speech.md](guide/api-speech.md) | Speech: the request, saved voices, Drama 3, streaming, scheduling |
| [api-transcription.md](guide/api-transcription.md) | Transcription: the live WebSocket and whole files |
| [api-models.md](guide/api-models.md) | `GET /v1/models`: every entry's fields by kind |
| [eval.md](guide/eval.md) | `nuclis eval`: teacher-forced perplexity on a text |

## Model families: `models/`

| Document | Purpose |
| --- | --- |
| [README.md](models/README.md) | Bringing a new model family in: the order of work, the gates, mistakes already made |
| [catalogue.md](models/catalogue.md) | Where the files come from, quantization conventions, companion files, the models directory |
| [qwen3.8.md](models/qwen3.8.md) | Qwen3.8-27B, the dense model the kernels were first tuned on: the architecture at a glance, where each of its facts lives, validation, the re-made file |
| [gemma4.md](models/gemma4.md) | Gemma 4 (26B-A4B, E4B, and the legacy 12B): artifact facts, forward pass, profile |
| [muse-glimmer.md](models/muse-glimmer.md) | Muse Glimmer 30B (legacy): artifact facts, the bring-up record |
| [bonsai.md](models/bonsai.md) | Bonsai 2 27B: the ternary file and its fork |
| [saluki.md](models/saluki.md) | Underdog Saluki 27B: Qwen3.8-27B in seven 2-bit encodings, its lineage, memory, and the last record against llama.cpp |
| [laya.md](models/laya.md) | Laya: the decision encoder behind `nuclis decide` |
| [clef-flash.md](models/clef-flash.md) | clef-flash: Cloudflare's decision model |
| [embeddinggemma.md](models/embeddinggemma.md) | EmbeddingGemma 2: the embedding model, its files, forward pass, prefixes, and encoders |
| [nemotron-asr.md](models/nemotron-asr.md) | Nemotron 3.5 ASR streaming 0.6B: the speech recognizer, its front end, cache-aware encoder, and transducer |
| [drama3.md](models/drama3.md) | Fish Audio Drama 3 (35B-A3B): the dialogue speech synthesizer, its MoE backbone, codebook tower, and 44.1 kHz codec |
| [qwen3-tts.md](models/qwen3-tts.md) | Qwen3-TTS (0.6B and 1.7B): the speech synthesizer, its talker, code predictor, and codec decoder; CustomVoice's built-in voices, the Base's cloned voices, VoiceDesign's described ones |

## The engine: `engine/` (the `inference/` package)

| Document | Purpose |
| --- | --- |
| [gguf.md](engine/gguf.md) | Reading GGUF: inspection, inventory, validation boundaries |
| [safetensors.md](engine/safetensors.md) | Reading safetensors sets |
| [tokenizer.md](engine/tokenizer.md) | The native tokenizer and vocabulary ownership |
| [quantization.md](engine/quantization.md) | CPU row decoders and their pinned fixtures |
| [cpu-reference.md](engine/cpu-reference.md) | The CPU numerical references and their precision |
| [metal-backend.md](engine/metal-backend.md) | The Metal backend: build, files and ownership, the execution model, prefill in chunks, limits |
| [metal-kernels.md](engine/metal-kernels.md) | The Metal compute kernels: thread geometry, matvec, batched tiles, recurrent, experts, row-wise and sampling kernels |
| [metal-attention.md](engine/metal-attention.md) | The Metal KV cache layouts and attention kernels: decode, few-query verify, prefill chunk, segments |
| [metal-evidence.md](engine/metal-evidence.md) | What `make test-metal` checks, with bounds, and the gates that hold Metal to the CPU reference |
| [apple-gpu.md](engine/apple-gpu.md) | The M4 Pro GPU as measured: counters on our kernels |
| [session.md](engine/session.md) | Session state: layouts, snapshots, checkpoints |
| [sampling.md](engine/sampling.md) | Generation, streaming, sampling, full-model traces |
| [speculative-decoding.md](engine/speculative-decoding.md) | Speculative decoding: contracts, draft sources, the Qwen verify budget |
| [prompt-profile.md](engine/prompt-profile.md) | Prompt profiles: chat renderers, stop sets, reasoning markers |
| [tool-calling.md](engine/tool-calling.md) | Tool calling: each family's format and the engine seam |
| [vision.md](engine/vision.md) | Vision through the companion projectors |
| [audio.md](engine/audio.md) | Audio through the companion encoders: decoding, the log-mel front end, Gemma 4's conformer |

## The app: `app/` (the `src/` package)

| Document | Purpose |
| --- | --- |
| [agent.md](app/agent.md) | The agent behind `nuclis agent`: the loop, its tools, web tools, and token cache |
| [agent-surface.md](app/agent-surface.md) | The agent's surface and loop: editor, keys, pickers, transcript, session files, tool contracts |
| [agent-tool-calls.md](app/agent-tool-calls.md) | Tool calls at the token boundary: each profile's calls, streaming, and what is cut |
| [terminal.md](app/terminal.md) | How the agent draws: styled output, live region, transcript, renderer, session files |

## Benchmarks: `benchmarks/`

| Document | Purpose |
| --- | --- |
| [README.md](benchmarks/README.md) | How nuclis is measured, the current results, and the index of the JSON records |
| [history.md](benchmarks/history.md) | Every dated benchmark record, oldest first, kept as written |
| [llama-cpp.md](benchmarks/llama-cpp.md) | The llama.cpp build that was the reference until 2026-10-14: its revision, build flags, and where its fixtures are |

## The record: `benchmarks/record/`

What each reference, card, and guide was before it was true: bring-up
narrative, investigations, superseded designs, and measurements, dated and
never edited after, one file per document it came from.

| Document | Purpose |
| --- | --- |
| [agent.md](benchmarks/record/agent.md) | Agent concepts: the record |
| [api.md](benchmarks/record/api.md) | The nuclis API: the record |
| [apple-gpu.md](benchmarks/record/apple-gpu.md) | The Apple M4 Pro GPU: the record |
| [architecture.md](benchmarks/record/architecture.md) | Architecture: the record |
| [audio.md](benchmarks/record/audio.md) | Audio: the record |
| [benchmarks.md](benchmarks/record/benchmarks.md) | Benchmarking: the record |
| [bonsai.md](benchmarks/record/bonsai.md) | Bonsai 2 27B: the record |
| [catalogue.md](benchmarks/record/catalogue.md) | Model artifacts: the record |
| [clef-flash.md](benchmarks/record/clef-flash.md) | clef-flash: the record |
| [cpu-reference.md](benchmarks/record/cpu-reference.md) | CPU numerical references: the record |
| [development.md](benchmarks/record/development.md) | Development: the record |
| [drama3.md](benchmarks/record/drama3.md) | Drama 3: the record |
| [embeddinggemma.md](benchmarks/record/embeddinggemma.md) | EmbeddingGemma 2: the record |
| [eval.md](benchmarks/record/eval.md) | Teacher-forced evaluation: the record |
| [gemma4.md](benchmarks/record/gemma4.md) | Gemma 4: the record |
| [gguf.md](benchmarks/record/gguf.md) | GGUF inspection: the record |
| [laya.md](benchmarks/record/laya.md) | Laya: the record |
| [metal-backend.md](benchmarks/record/metal-backend.md) | Metal backend: the record |
| [muse-glimmer.md](benchmarks/record/muse-glimmer.md) | Muse Glimmer 30B: the record |
| [nemotron-asr.md](benchmarks/record/nemotron-asr.md) | Nemotron 3.5 ASR: the record |
| [prompt-profile.md](benchmarks/record/prompt-profile.md) | Prompt profiles: the record |
| [qwen3-tts.md](benchmarks/record/qwen3-tts.md) | Qwen3-TTS: the record |
| [qwen3.8.md](benchmarks/record/qwen3.8.md) | Qwen3.8-27B: the record |
| [safetensors.md](benchmarks/record/safetensors.md) | Safetensors: the record |
| [saluki.md](benchmarks/record/saluki.md) | Underdog Saluki 27B: the record |
| [sampling.md](benchmarks/record/sampling.md) | Generation and sampling: the record |
| [session.md](benchmarks/record/session.md) | Session state: the record |
| [speculative-decoding.md](benchmarks/record/speculative-decoding.md) | Speculative decoding: the record |
| [vision.md](benchmarks/record/vision.md) | Vision: the record |

The JSON records sit beside them in [benchmarks/](benchmarks/), mostly
`nuclis-…` and `reference-…` by date and family, and the pinned traces and raw runs in
[../tests/fixtures/](../tests/fixtures/) (see its
[provenance.md](../tests/fixtures/provenance.md)).

## Elsewhere

There is no roadmap and no decision-record folder: what comes next is
agreed in session and written into the maintainer's tracker, a unit's
record is its pull request, and a design that outlives its unit goes to
the document above that owns it. Agent working instructions:
[../AGENTS.md](../AGENTS.md). Third-party material:
[../THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
