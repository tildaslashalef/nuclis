# nuclis documentation

nuclis is a standalone local inference engine and evaluation CLI for
Qwen3.8-27B on Apple silicon: an engine library under `inference/` and an
executable under `src/` with a `generate`/`bench` CLI and an embedded chat
agent.

## Core documents

| Document | Purpose |
| --- | --- |
| [spec.md](spec.md) | **Authoritative spec.** Requirements, scope, and acceptance criteria |
| [roadmap.md](roadmap.md) | Accepted themes not yet planned (performance follow-ups, vision, agent expansion) and the deferred list |
| [architecture.md](architecture.md) | **Start here.** The engine stack in ten short sections with diagrams |
| [development.md](development.md) | Toolchain, build/test commands, user directories, versioning, conventions |
| [llm-guide.md](llm-guide.md) | LLM and inference concepts, taught as they are implemented here |
| [engineering-log.md](engineering-log.md) | Dated outcomes and evidence of every closed unit |

## Reference

Detailed engineering documents under [reference/](reference/):

| Document | Purpose |
| --- | --- |
| [agent-concepts.md](reference/agent-concepts.md) | Agent and terminal concepts behind `nuclis agent`, taught as they are implemented here |
| [artifacts.md](reference/artifacts.md) | Model artifact sources, Unsloth quantization conventions, companion files, the models directory |
| [bench.md](reference/bench.md) | `bench`, `bench --profile`, `make bench-kernels`: timing definitions and methodology |
| [cpu-reference.md](reference/cpu-reference.md) | CPU linear/vector/attention/recurrent references and numerical precision |
| [gemma4.md](reference/gemma4.md) | Gemma 4 12B: artifact facts with provenance, forward pass, tokenizer, CPU reference and Metal plan evidence, profile |
| [generation.md](reference/generation.md) | Native CPU generation, streaming/sampling, full-model traces |
| [gguf-inspection.md](reference/gguf-inspection.md) | Inspection commands, artifact inventory, validation boundaries |
| [metal-backend.md](reference/metal-backend.md) | GPU-resident Metal backend: bridge, kernels, ownership, evidence |
| [prompt-profile.md](reference/prompt-profile.md) | Prompt profiles (Qwen3.8, Gemma 4): bounded chat renderers, stop sets, reasoning markers, pinned fixtures |
| [new-model-guide.md](reference/new-model-guide.md) | Bringing a new model family through the seam: order of work, gates, mistakes already made |
| [quantization.md](reference/quantization.md) | Row decoders, ownership, pinned numerical fixtures |
| [qwen-validation.md](reference/qwen-validation.md) | Qwen profile constraints, weight bindings, validation evidence |
| [reference-baseline.md](reference/reference-baseline.md) | Pinned llama.cpp baseline and comparison measurements |
| [speculative-decoding.md](reference/speculative-decoding.md) | Speculative decoding: recovery and draft contracts, per-family draft sources, measurements |
| [tokenizer.md](reference/tokenizer.md) | Native qwen35 encoding/decoding and vocabulary ownership |

## Decisions

Point decisions that are hard to reverse or likely to be questioned live in
[adr/](adr/), one file per decision (see the [template](adr/TEMPLATE.md)); a
change supersedes rather than edits.

## Data directories

- [benchmarks/](benchmarks/) — recorded measurement artifacts (JSON benchmark records)
- [../tests/fixtures/](../tests/fixtures/) — committed reference oracle: traces, raw run records, and provenance (see its [provenance.md](../tests/fixtures/provenance.md))

Third-party material and provenance: [../THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

Active plan (unfinished units only): [../TODO.md](../TODO.md).
Agent working instructions: [../AGENTS.md](../AGENTS.md).
