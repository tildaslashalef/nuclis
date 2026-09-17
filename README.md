# nuclis

A local inference engine and evaluation CLI for large language models in
GGUF format, written in Zig with a Metal backend for Apple Silicon. The
first target is Qwen3.8-27B on an M4 Pro with 48 GB of unified memory; a
second architecture, Gemma 4 12B, runs through the same adapter seam.

## What it does

- Loads a GGUF file, validates its metadata and tensor layout against a
  pinned architecture profile, and refuses anything it cannot execute
  exactly. Nothing is requantized.
- Runs prefill and decode on the GPU (Metal) or on a CPU reference
  implementation that the GPU path is checked against, layer by layer,
  with traces pinned from llama.cpp.
- Provides `generate`, `bench`, `tokenize`, the interactive `agent`
  surface, and `model pull` / `model inspect` for fetching and judging
  artifacts from the Hugging Face Hub before download.
- Keeps configuration in `~/.nuclis/nuclis.json`: default model, backend,
  context, sampling, and a registry of named models with per-model
  overrides.

## Status

All numbers in this repository were measured on one machine: a Mac with
an Apple M4 Pro (12 CPU cores, 16 GPU cores), 48 GiB of unified memory,
macOS 26.6.2, Zig 0.16.0, release builds, on AC power. Nothing has been
run on other hardware yet.

Qwen3.8-27B on Metal against the pinned llama.cpp reference (build
`7620399`, Metal, flash attention, F16 cache), both engines on the same
token arrays, 128 output tokens, greedy, warm means of three runs (one run
at 32,639). Prefill and decode in tokens per second, 2026-09-10:

| Prompt tokens | Prefill, nuclis | Prefill, llama.cpp | Decode, nuclis | Decode, llama.cpp |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 90.45 | 89.19 | 10.62 | 9.66 |
| 4,096 | 83.70 | 89.26 | 10.20 | 9.21 |
| 16,384 | 62.70 | 74.07 | 8.27 | 7.32 |
| 32,639 | 49.55 | 67.28 | 7.55 | 6.71 |

Decode is above the reference at every length; prefill is at the
reference at 512 tokens and 6–26 % below it from 4K to 32K. Methodology,
variance, and the raw records:
[docs/reference/bench.md](docs/reference/bench.md#acceptance-runs).

Gemma 4 12B, in Google's quantization-aware-trained checkpoint (every
weight matrix Q4_0), runs on both backends, matches llama.cpp's per-layer
traces, and has its own acceptance record against the reference
([docs/reference/gemma4.md](docs/reference/gemma4.md),
[bench.md](docs/reference/bench.md#gemma-4-12b-acceptance-record-qat-file-modl-08-2026-09-12)).

## Supported models

"Supported" means one thing here: the model is in the built-in catalogue
with its repository, commit, and SHA-256 pinned, and `nuclis model pull
<name>` verifies those digests before the file is published into place.
A GGUF outside the catalogue still runs when its architecture has an
adapter — `nuclis model inspect` says which of the two it is.

| Name | Architecture | File | Size |
| --- | --- | --- | ---: |
| `qwen3.8-27b` | qwen35 | `unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf` | 16.5 GB |
| `gemma-4-12b` | gemma4 | `unsloth/gemma-4-12b-it-GGUF/gemma-4-12b-it-UD-Q4_K_XL.gguf` | 7.37 GB |
| `gemma-4-12b-qat` | gemma4 | `unsloth/gemma-4-12B-it-qat-GGUF/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf` | 6.72 GB |
| `gemma-4-26b-a4b` | gemma4 (mixture of experts) | `unsloth/gemma-4-26B-A4B-it-qat-GGUF/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` | 14.2 GB |
| `muse-glimmer-30b` | muse-glimmer | `unsloth/Muse-Glimmer-30B-GGUF/Muse-Glimmer-30B-UD-Q4_K_XL.gguf` | 15.9 GB |

Each entry carries the vision projector and the draft head of its
repository as companions (`--all` fetches them; they are verified now and
loaded by later units). The two 12B Gemma entries are the same model in
two quantizations: the plain K-quant release, and Google's
quantization-aware-trained checkpoint, whose every weight matrix is Q4_0
— the encoding it was trained for, which is why it is smaller *and*
decodes faster. The 26B-A4B mixture of experts runs on both backends (its
acceptance record is the next unit); the Muse entry is pinned ahead of its
adapter, and `nuclis model inspect` reports it *not runnable* until the
architecture is implemented.

```sh
nuclis model ls                       # the catalogue and what is present
nuclis model pull gemma-4-12b-qat --all
nuclis model inspect qwen3.8-27b      # judge a file from its head, before downloading
```

## Requirements

- macOS on Apple Silicon. Only the M4 Pro with 48 GiB above has been
  tested; the Qwen artifact needs about 20 GB of memory at 32K context,
  so smaller configurations are untried, not unsupported.
- Zig 0.16.0 and the Command Line Tools (`xcode-select -p`).
- About 17 GB of disk for the Qwen artifact, 6.7–7.4 GB for a Gemma one.

## Build and run

```sh
make metal                      # release build with the Metal backend
make check                      # format check, unit tests, GPU kernel fixtures

./zig-out/bin/nuclis config init                 # writes ~/.nuclis/nuclis.json
./zig-out/bin/nuclis model pull qwen3.8-27b      # pinned commit, SHA-256 verified
./zig-out/bin/nuclis generate --prompt "Write a Zig function that reverses a string."
./zig-out/bin/nuclis agent
./zig-out/bin/nuclis bench --prompt-file prompt.txt --max-tokens 64
```

`nuclis --help` lists every command and flag. `make help` lists the
development targets (tests against the real artifact, kernel benchmarks,
trace comparison, profiling).

## Repository layout

| Path | Contents |
| --- | --- |
| `inference/` | The engine: GGUF parsing, quantized tensor decoding, tokenizer, sampling, session state, model adapters, CPU reference, Metal backend and kernels |
| `src/` | The executable: CLI, configuration, model management, and `nuclis agent` (`src/tui/` terminal surface, `src/agent/` composition) |
| `huggingface/` | Hub downloads over Xet, used by `nuclis model pull` |
| `docs/` | Specification, architecture, development guide, reference documents, engineering log |
| `tests/fixtures/` | Pinned reference traces and provenance |
| `scripts/` | Reference capture and comparison tools |

## Documentation

- [docs/architecture.md](docs/architecture.md): how a token flows through
  the engine, the adapter seam, the GPU path, and where performance goes.
- [docs/spec.md](docs/spec.md): scope, requirements, and acceptance
  criteria.
- [docs/development.md](docs/development.md): environment, build and
  test contract, model download, configuration.
- [docs/reference/](docs/reference/): benchmarks, the Metal backend,
  GGUF, model facts, the engineering log.
- [docs/reference/tool-calling.md](docs/reference/tool-calling.md): model tool formats and the planned engine boundary.
- [docs/llm-guide.md](docs/llm-guide.md): the concepts behind each
  component, written as they were built.

## Name

*nuclis* is a play on *nucleus*, from the Latin for "kernel", the core of
a nut: the small, dense part where the mass is, as in the
[atomic nucleus](https://en.wikipedia.org/wiki/Atomic_nucleus). An
inference engine is that kind of core, a few kernels doing the work with
everything else built around them. Pronounce it like *nucleus*. The
project's home will be [nuclis.dev](https://nuclis.dev) (not live yet).

## Design rules

Explicit allocators and I/O, typed errors instead of panics, bounded
inputs and outputs, and numerical correctness before optimization.
External implementations are references and validation oracles, never
sources to copy. Third-party material in the tree is listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Licence

MIT — see [LICENSE](LICENSE). Model weights are not covered by it: each
artifact carries the licence of its own publisher, which
`nuclis model inspect` does not check and you should.
