# nuclis

A local inference engine for open-weight language models, written in Zig
with a Metal backend for Apple Silicon, and a small coding agent on top of
it. Every layer is checked against a pinned llama.cpp build before it is
made fast.

![nuclis agent with Qwen3.8-27B finding and fixing a failing test](docs/media/agent.gif)

*`nuclis agent` with Qwen3.8-27B on an M4 Pro, fixing a broken function
in a small Python project. Played back at 2× with pauses shortened; the
real session took 2 min 35 s. Recorded by `scripts/agent-demo.py`.*

## Status

An experimental project, built in my free time to learn how an inference
stack works, from the GGUF bytes to the sampled token, and to learn Zig.
It has run on one machine, the version is 0.x, and breaking changes arrive
without notice. The documents record what was measured, not what was
intended. Read it as a worked notebook with a working engine inside.

## Quick start

Requires macOS on Apple Silicon, Zig 0.16.0, and the Command Line Tools.

```sh
make metal                                      # release build, Metal backend
./zig-out/bin/nuclis model pull qwen3.8-27b --all   # pinned commit, SHA-256 verified
./zig-out/bin/nuclis config init --discover     # ~/.nuclis/nuclis.json, every model found
./zig-out/bin/nuclis agent                      # the chat, with tools, in this directory
```

`nuclis generate`, `bench`, `eval`, and `tokenize` cover the rest;
`nuclis --help` lists every command and `make help` the development targets.

`make install` puts the binary in `~/.local/bin` (`PREFIX=` elsewhere), and
`nuclis completion fish|bash|zsh` prints Tab completion for that shell:

```sh
nuclis completion fish > ~/.config/fish/completions/nuclis.fish
echo 'source <(nuclis completion zsh)' >> ~/.zshrc     # after compinit
echo 'source <(nuclis completion bash)' >> ~/.bashrc
```

## Models

The catalogue pins each model's repository, commit, and SHA-256, so
`nuclis model pull <name>` fetches exactly the bytes that were measured.
Every entry reads images and carries a draft head for speculative decoding.

| Family | Name | What is distinctive | Size |
| --- | --- | --- | ---: |
| Qwen3.8 | `qwen3.8-27b` | hybrid attention: 16 attention and 48 DeltaNet layers | 16.5 GB |
| Gemma 4 | `gemma-4-12b-qat` | dense, quantization-aware trained: every matrix Q4_0 | 6.7 GB |
| | `gemma-4-26b-a4b` | mixture of experts, 8 of 128 per token | 14.2 GB |
| | `gemma-4-e4b-qat` | on-device size: per-layer embeddings, shared KV layers | 4.2 GB |
| Muse Glimmer | `muse-glimmer-30b` | dense 30B with a DFlash drafter, on by default | 15.9 GB |

**Anything else** runs when its architecture has an adapter (Qwen3.8,
Gemma 4, Muse Glimmer): finetunes, other quantizations, Gemma 4's K-quant
release, Prism ML's ternary Bonsai 2. Judge a file before downloading it,
pull it by repository, and let discovery register it:

```sh
nuclis model inspect prism-ml/Ternary-Bonsai-2-27B-gguf --file Ternary-Bonsai-2-27B-PTQ1_0.gguf
nuclis model pull prism-ml/Ternary-Bonsai-2-27B-gguf --file Ternary-Bonsai-2-27B-PTQ1_0.gguf
nuclis config init --discover   # names it, picks its prompt profile, finds its companions
```

## Results

Measured on one machine: an Apple M4 Pro (12 CPU, 16 GPU cores), 48 GiB
of unified memory, macOS 26.6.2, Zig 0.16.0, release builds, against the
pinned llama.cpp (`7620399`, Metal, flash attention, F16 cache) on the same
token arrays: 128 greedy output tokens, warm means of three runs. Tokens
per second, nuclis / llama.cpp:

| Model | Decode, 512 | Decode, 32,639 | Prefill, 512 | Prefill, 32,639 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.8-27B | 10.62 / 9.66 | 7.55 / 6.71 | 90.45 / 89.19 | 49.55 / 67.28 |
| Gemma 4 12B QAT | 23.96 / 27.69 | 16.24 / 20.94 | 179.45 / 224.46 | 73.04 / 143.72 |
| Gemma 4 26B-A4B | 55.29 / 68.02 | 31.30 / 44.09 | 500.42 / 580.76 | 134.74 / 343.38 |
| Muse Glimmer 30B | 9.60 / 13.69 | 6.62 / 9.98 | 93.49 / 95.28 | 59.20 / 76.86 |

Qwen3.8, the first target, decodes faster than the reference at every
length. The other families run on kernels written for Qwen and have not
been tuned yet; the per-kernel profiles say where the gap is. Every file
matches the reference's per-layer traces on the CPU and on Metal before
its rate is recorded. Methodology, variance, and every record:
[docs/reference/bench.md](docs/reference/bench.md).

## Experiment: decisions with Laya

`nuclis decide` runs [Laya](https://huggingface.co/convaiinnovations/laya),
a small encoder with a typed-decision head, beside the language models. It
answers a typed question about a state (text, a JSON document, or a
conversation) in one forward pass, with no decoding: `noul` (the
probability that something holds), `choice` (one option), or `score` (a
level on a scale).

```sh
nuclis model pull laya
nuclis decide --noul 'Does this log show a failure that needs a person now?' \
  --state-file quiet.log --state-file disk.log --state-file payments.log
```

The idea under test is Laya as a **filter**. A language model that needs
one fact from 30 search hits or 50 log sections pays to read all of them:
at 90 tokens/s of prefill, Qwen3.8-27B spends about 5.7 s on each
512-token state. Laya judges such a state in about 0.2 s on Metal, so one
question fanned out over the states can hand the model only the few that
matter. Whether an agent actually gains from that is an experiment still
to run, not a result.

Two checkpoints are pinned, both run on the CPU and on Metal:

| Name | Encoder | Tokens per sequence | Reads |
| --- | --- | ---: | --- |
| `laya` (the default) | ModernBERT-large | 512 | English |
| `laya-multilingual` | mmBERT-base | 1,024 | other languages too |

**Correctness.** The forward matches Laya's own Python package (`laya`
0.3.20, F32 on the CPU), the oracle: every token id of every request, the
residual stream at seven stages within 1e-5 of its largest value, the
logits within 1e-4, the calibrated answers equal after rounding. That
holds for 8 reference requests on `laya` and 16 on `laya-multilingual`,
the multilingual ones including French, German, Spanish, Arabic, Chinese,
and Hindi states and positions past 512. On Metal, a packed batch gives
each sequence exactly the logits it gets alone.

**Speed** (M4 Pro, one `noul` question over synthetic logs, encode time):

| States × tokens | `laya`, Metal | `laya`, CPU | `laya-multilingual`, Metal |
| --- | ---: | ---: | ---: |
| 1 × ~500 | 0.21 s | 2.9 s | 0.11 s |
| 50 × ~500 | 9.8 s | — | 4.1 s |
| 50 × ~60 | 1.3 s | 26.5 s | 0.6 s |

Metal is 8–20× the CPU. The multilingual checkpoint is smaller and faster,
but it takes 0.4 s to load against 0.1 s.

**What the answers are like**, from 20 hand-written cases run on both
checkpoints:

- As a filter, `laya` works. Four logs ranked by "needs a person now" come
  out disk full (0.92), payment errors (0.70), slow queries (0.49), healthy
  (0.00). Among 50 short entries, the 8 cache misses rank first; among five
  search hits, the flaky login test and the cookie race rank above the
  README and the changelog.
- Plain judgements are mostly right (a cancel threat 0.84, a crash ticket
  routed to the technical team at 0.95). Some are flat or off: a scam email
  at 0.48, and 40 minutes of database replication lag judged not urgent.
  Rewording the question or describing the options changes such answers.
- `laya-multilingual` reads other languages where `laya` cannot: a German
  cancellation 0.61 against 0.14, a Hindi request to close an account
  1.00 against 0.00. It also reads to 1,024 tokens, so it finds an error
  that `laya` cuts away. On English it is clearly worse: it loses the
  filter rankings above and is confidently wrong on several plain
  questions. `laya` stays the default.
- Neither checkpoint identifies languages; that is not what they were
  trained for.

A longer state than the budget is cut, and the cut is always flagged.
Details, measurements, and limits: [docs/reference/laya.md](docs/reference/laya.md).

## Documentation

- [docs/architecture.md](docs/architecture.md): how a token flows through
  the engine, the adapter seam, and where performance goes.
- [docs/spec.md](docs/spec.md): scope, requirements, acceptance criteria.
- [docs/development.md](docs/development.md): build, test gates,
  configuration, model downloads.
- [docs/reference/](docs/reference/): benchmarks, the Metal backend, GGUF,
  each model family's facts.
- [docs/llm-guide.md](docs/llm-guide.md): the concepts behind each
  component, written as they were built.
- [docs/engineering-log.md](docs/engineering-log.md): every closed unit of
  work and its evidence.

## Design rules

Explicit allocators and I/O, typed errors instead of panics, bounded
inputs and outputs, and numerical correctness before optimization.
Nothing is requantized: a file the engine cannot execute exactly is
refused. External implementations are references and validation oracles,
never sources to copy.

## Contributions

Issues are welcome: a bug with the command and artifact that reproduce it,
or a measurement from other hardware. Pull requests are not being reviewed
for now; the plan is set by what I want to understand next. Fork freely.

## Disclosure

This code is written with heavy assistance from AI coding agents, with me
directing what gets built, deciding what counts as evidence, and keeping
or discarding the result. The instructions those agents work from are
committed in [AGENTS.md](AGENTS.md), so the process is inspectable, and
every number in the documentation was measured on the machine named above,
never estimated by anyone or anything. If software built this way is not
for you, this repository is not for you.

None of it would exist without [llama.cpp](https://github.com/ggml-org/llama.cpp)
and GGML: the GGUF format is theirs, and a pinned build of llama.cpp is the
oracle every layer and kernel here is checked against. The shape of the
Metal bridge follows antirez's [ds4](https://github.com/antirez/ds4), which
also set the example of saying the AI part plainly. nuclis copies no code
from any of them; third-party material in the tree is listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Name and licence

*nuclis* is a play on *nucleus*, Latin for "kernel": the small, dense core
where the mass is. Pronounce it like *nucleus*.

MIT, see [LICENSE](LICENSE). Model weights are not covered by it: each
artifact carries its publisher's licence, which you should check.
