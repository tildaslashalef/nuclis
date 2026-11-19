<!-- kind: guide -->
<h3 align="center"><img src="docs/branding/nuclis.png" alt="nuclis logo: two teal bands forming an N around an amber diamond" width="160"></h3>

# NUCLIS Engine

A local inference engine for open-weight models, written in Zig with a Metal
backend for Apple Silicon, and a small coding agent on top of it. It is tuned
for **Gemma 4 26B-A4B** and **Gemma 4 E4B** on an M4 Pro with 48 GB, and runs
**Qwen3.8-27B**, the dense hybrid its kernels were first written for.
Decoding reads every weight a token uses once per token, so on 273 GB/s of
memory bandwidth speed follows the parameters a token activates, not the
model's size: the 26B-A4B wakes 8 of its 128 experts, about 4B parameters,
and decodes at about 50 tokens/s where the dense Qwen3.8-27B manages 8–12.

How it is verified: every model runs first on a plain CPU reference, the
Metal kernels are checked against it, generated tiny models exercise every
weight encoding with no download (`make test-tiny`), and gates per model run
in lanes, from `make check` (seconds, no model) through `make verify-auto`
(what a change reaches) to the release tiers
([docs/development.md § Gates](docs/development.md#gates)).

![nuclis agent with Qwen3.8-27B finding and fixing a failing test](docs/media/agent.gif)

*An earlier run, on Qwen3.8-27B's first pinned file: `nuclis agent` on an M4
Pro fixing a broken function in a small Python project, speculative decoding
on, pauses over 1.5 s shortened (`scripts/app/agent-demo.py`).
[engine.nuclis.dev](https://engine.nuclis.dev) replays it.*

**Status**: an experimental project, built in my free time to learn how an
inference stack works, and to learn Zig. It runs on one machine, the version
is 0.x, and breaking changes arrive without notice.

## Quick start

From source: macOS on Apple Silicon, Zig 0.17.1, and the Command Line Tools.

```sh
make metal                                              # release build, Metal backend
./zig-out/bin/nuclis model pull gemma-4-26b-a4b --all   # pinned commit, SHA-256 verified
./zig-out/bin/nuclis config init                        # ~/.nuclis/nuclis.json, from what is installed
./zig-out/bin/nuclis agent                              # the chat, with tools, in this directory
```

[Releases](https://github.com/tildaslashalef/nuclis/releases) ship the macOS
binary and an x86-64 Linux build that runs on the CPU only, for the small
models ([getting-started.md](docs/guide/getting-started.md)). `nuclis --help`
lists every command, `make help` the development targets.

## Models

The index pins each model's repository, commit, and SHA-256, so `nuclis model
pull <name>` fetches exactly the bytes that were measured.

| Kind | Name | What it is | Size |
| --- | --- | --- | ---: |
| chat | `gemma-4-26b-a4b` | Gemma 4 mixture of experts, 8 of 128 per token; the default | 14.2 GB |
| chat | `gemma-4-e4b-qat` | Gemma 4 at on-device size: per-layer embeddings, shared KV layers; hears audio | 4.2 GB |
| chat | `qwen3.8-27b` | dense hybrid: 16 attention and 48 DeltaNet layers; drafters `mtp` and `dflash2` | 16.6 GB |
| decision | `laya`, `laya-multilingual` | a small encoder that answers typed questions in one pass | 0.8, 0.6 GB |
| decision | `clef-flash` | Cloudflare's 9B decision model: long states and images | 8.0 GB |
| embedding | `embeddinggemma-2` | text, images, and audio into one vector space | 0.3 GB |
| speech | `qwen3-tts-{1.7b,0.6b}-{customvoice,base}`, `qwen3-tts-1.7b-voicedesign` | nine built-in voices, a voice cloned from a clip, or one described in words | 2.5–4.5 GB |
| speech | `drama-3` | Fish Audio's Drama 3: dialogue, delivery cues, cloned voices | 20.8 GB |
| transcription | `nemotron-asr-0.6b` | streaming recognition with punctuation and word times | 2.6 GB |

Every chat model reads images and carries a drafter, switched on where it
measured faster. **Anything else** runs when its architecture has an adapter
(Qwen3.8, Gemma 4, Muse Glimmer): finetunes, other quantizations, Prism ML's
ternary Bonsai 2. `nuclis model inspect` judges a file before the download,
and `nuclis config sync` registers what was pulled.

## Results

nuclis today (2026-11-19): Apple M4 Pro, 48 GiB, macOS 27.2.1, ReleaseSafe,
Metal, F16 cache, 128 greedy output tokens; the mean of three runs after a
warmup to 16,384, one run each at 32,639 and 65,407 on a 65,536 context, with
no swap ([2026-11-19](docs/benchmarks/record/gemma4.md#sliding-layers-on-the-materialized-body-2026-11-19)).
Tokens per second by prompt length; each record is measured against nuclis's
own previous one
([the October 2026 comparison with llama.cpp](docs/benchmarks/record/benchmarks.md#the-benchmarks-on-macos-27-2026-10-04)
is history):

| Decode | 512 | 4,096 | 16,384 | 32,639 | 65,407 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemma 4 26B-A4B | 57.54 | 54.11 | 48.71 | 41.94 | 32.30 |
| Gemma 4 E4B QAT | 53.19 | 50.13 | 47.22 | 41.90 | 33.03 |
| Qwen3.8-27B | 11.16 | 9.23 | 9.02 | 9.17 | 8.41 |

| Prefill | 512 | 4,096 | 16,384 | 32,639 | 65,407 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemma 4 26B-A4B | 587.14 | 550.06 | 469.89 | 391.81 | 282.76 |
| Gemma 4 E4B QAT | 585.95 | 551.42 | 500.96 | 437.29 | 337.20 |
| Qwen3.8-27B | 93.16 | 84.02 | 77.48 | 73.17 | 61.19 |

**Speculative decoding**: a small drafter proposes tokens and the model checks
them in one batch, keeping what it would have produced itself. Decode off → on,
greedy, the entry's draft length, median of three (2026-10-05; 26B-A4B 11-18):

| Model | Draft | 512 | 32,639 |
| --- | ---: | ---: | ---: |
| Gemma 4 26B-A4B | 4 | 57.4 → 66.3 (1.16×) | 42.1 → 44.1 (1.05×) |
| Gemma 4 E4B QAT | 6 | 46.9 → 81.9 (1.75×) | 33.6 → 37.5 (1.12×) |
| Qwen3.8-27B | 7 | 11.6 → 17.4 (1.50×) | 7.9 → 9.8 (1.26×) |

Qwen3.8-27B's re-made file drafts at draft 7 to 18.46 tok/s with MTP and
18.18 with DFlash2 at 512 (1.61×, 1.63×), 15.98 and 16.81 at 4,096
([2026-11-12](docs/benchmarks/history.md#the-re-made-qwen38-file-2026-11-12)).

An agent re-reads a growing prompt and writes a few hundred tokens a step. On
the agent's eighteen tasks ([2026-11-18](docs/benchmarks/record/agent.md#where-an-agent-turn-goes-2026-11-18)),
drafters on, a task takes 35 s in the model on the E4B, 51 s on the 26B-A4B,
and 108 s on Qwen3.8-27B. The 26B-A4B's drafter saves 15 % there, and prefill
is then 61 % of its time. Past 32K the drafters stop paying: greedy, the
26B-A4B's is 1.05× at 32,639 and 0.94× at 65,407, the E4B's 1.14× and
0.97×, and below 1× sooner when sampling, so drafting stops at a
32,768-token prompt by default (`generation.draft_max_depth`, 0 never). Method and every record: [docs/benchmarks](docs/benchmarks/README.md).

**How it got here.** nuclis began with Qwen3.8-27B, whose attention and Gated
DeltaNet layers are where the kernels were first written and tuned; it decoded
faster than llama.cpp at every length in October 2026, and stays supported,
if slow.

## Beyond chat

[docs/README.md](docs/README.md) lists every document; start with
[architecture.md](docs/architecture.md) and [llm-guide.md](docs/llm-guide.md).
Every change is a [merged pull request](https://github.com/tildaslashalef/nuclis/pulls?q=is%3Apr+is%3Amerged) with its evidence.

**Decisions.** `nuclis decide` asks Laya or clef-flash a typed question
(yes/no, a choice, a level) and gets the answer in one pass: Laya judges a
500-token log in 0.1–0.2 s ([laya.md](docs/models/laya.md), [clef-flash.md](docs/models/clef-flash.md)).

**HTTP.** `nuclis serve` keeps models open behind a local API: OpenAI's Chat
Completions (tools, images, streaming, a cache of what a conversation already
sent), TypeSafe's Jev protocol for decisions, embeddings, speech, and
transcription ([api.md](docs/guide/api.md)).

**Embeddings.** `nuclis embed` maps text, images, audio, or a mix to one
768-number vector with EmbeddingGemma 2, within `1 − cos` ≤ 5e-11 of Google's
pipeline on the BF16 file ([embeddinggemma.md](docs/models/embeddinggemma.md)).

**Speech.** `nuclis speak` (Qwen3-TTS; Drama 3 over HTTP) and `nuclis
transcribe` (Nemotron ASR, live or whole recordings) are a voice and ears
beside the language model ([api-speech.md](docs/guide/api-speech.md), [qwen3-tts.md](docs/models/qwen3-tts.md), [drama3.md](docs/models/drama3.md), [nemotron-asr.md](docs/models/nemotron-asr.md)).

## Design rules

Explicit allocators and I/O, typed errors instead of panics, bounded inputs
and outputs, and numerical correctness before optimization. Nothing is
requantized: a file the engine cannot execute exactly is refused. nuclis is
its own reference: new work is checked against its CPU reference, Metal
against that, and the same model in an encoding already validated; other
implementations are never sources to copy
([docs/models/README.md § 4](docs/models/README.md#4-validate-against-nuclis-itself)).

## Contributions

Issues are off, and the project takes no issue reports or pull requests: the
work is set by what I want to understand next. Fork freely.

## Disclosure

This code is written with heavy assistance from AI coding agents, with me
directing what gets built, deciding what counts as evidence, and keeping or
discarding the result. Their instructions are in [AGENTS.md](AGENTS.md), and
every number in the documentation was measured, never estimated.

None of it would exist without [llama.cpp](https://github.com/ggml-org/llama.cpp)
and GGML: the GGUF format is theirs, and while nuclis bootstrapped, a pinned
llama.cpp build was the oracle its first families were checked against. The
Metal bridge follows antirez's [ds4](https://github.com/antirez/ds4), which
also set the example of saying the AI part plainly. nuclis copies no code
from them; third-party material is listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Name and licence

*nuclis* plays on *nucleus*, Latin for "kernel": the small, dense core where
the mass is. MIT, see [LICENSE](LICENSE); model weights are not covered by
it, and each artifact carries its publisher's licence.
