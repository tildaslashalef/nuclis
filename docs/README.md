# nuclis documentation

nuclis is a local inference engine for Apple silicon with a CLI, a small
coding agent, and decision models: an engine library under `inference/`,
an executable under `src/`. Every document is listed here, one folder per
kind; a new document goes in the folder whose rule it fits and gets a line
below.

## Start here

| Document | Purpose |
| --- | --- |
| [../README.md](../README.md) | What nuclis is, the quick start, the models, the results |
| [guide/getting-started.md](guide/getting-started.md) | From nothing to a chat with a local model |
| [architecture.md](architecture.md) | The map: the engine and the app in twelve short sections with diagrams |
| [spec.md](spec.md) | **The authoritative spec**: requirements, scope, acceptance criteria |
| [llm-guide.md](llm-guide.md) | The inference stack's concepts, taught as they are implemented here |
| [development.md](development.md) | Contributing: toolchain, build and test, gates, the record, CI, releases |
| [Merged pull requests](https://github.com/tildaslashalef/nuclis/pulls?q=is%3Apr+is%3Amerged) | Every change since v0.6.0, each with its record: what and why, the evidence, what remains |
| [The worklog, archived](https://github.com/tildaslashalef/nuclis/blob/v0.6.0/docs/worklog.md) | Work before v0.6.0 is recorded in the worklog archived at the `v0.6.0` tag |

## Using nuclis: `guide/`

| Document | Purpose |
| --- | --- |
| [getting-started.md](guide/getting-started.md) | Install, a first run, where nuclis keeps things, model downloads |
| [configuration.md](guide/configuration.md) | `nuclis.json`: every setting, its scope, and what overrides it |
| [api.md](guide/api.md) | `nuclis serve`: the HTTP API: decisions, chat completions, embeddings, speech, and transcription |
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
| [metal-backend.md](engine/metal-backend.md) | The Metal backend: bridge, kernels, ownership, evidence |
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
| [agent.md](app/agent.md) | The agent and terminal concepts behind `nuclis agent`, its web tools and token cache |
| [terminal.md](app/terminal.md) | How the agent draws: styled output, live region, transcript, renderer, session files |

## Benchmarks: `benchmarks/`

| Document | Purpose |
| --- | --- |
| [README.md](benchmarks/README.md) | How nuclis is measured, the current results, and the index of the JSON records |
| [history.md](benchmarks/history.md) | Every dated benchmark record, oldest first, kept as written |
| [llama-cpp.md](benchmarks/llama-cpp.md) | The llama.cpp build that was the reference until 2026-10-14: its revision, build flags, and where its fixtures are |

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
