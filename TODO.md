# TODO — active plan

This file is the queue: the units agreed and not yet merged, each with its
design, and a *Where we are* note. A unit is one session, one branch, and
one pull request, whose description is the record of the work; when it
merges, its section is deleted here, and when the last one goes, this file
is emptied back to this header and its *Deferred* list. Requirements live in [docs/spec.md](docs/spec.md); the
engine map in [docs/architecture.md](docs/architecture.md); how to build,
test, and measure in [docs/development.md](docs/development.md).

Session protocol (also in [AGENTS.md](AGENTS.md)): read this file first. If
it lists work, summarize *Where we are* and ask the user how to continue. If
it is empty, ask what to work on and write the agreed plan here.

## Where we are

One unit in progress: **serve opens nothing until asked** (branch
`serve-lazy-models`).

| Unit | State |
| --- | --- |
| serve opens nothing until asked | in progress |

## serve opens nothing until asked

Base: `4449064`. Asked by the user, 2026-10-18.

`nuclis serve` today opens `decide.model` before it listens (or the
`--model` decision models), a language model with `--chat-model`, and the
embedding model on its first request. Two flags for one idea confuse.

- **No model opens at start** unless named. Every kind opens on its first
  request, as embeddings already do. Remove the default-model preload and
  its warning in `serve` (`src/api/root.zig`).
- **One flag, `--model <name>`, repeatable, any kind.** Its kind comes
  from the registry entry's `kind`, else the decision or embedding
  catalogue, else generation: the rule `cli.zig` uses for `generate`,
  moved to `catalog.kindOf(registry, name)` and shared. Per kind at most
  `decisions/pool.capacity` decision models, one language model, one
  embedding model (each service keeps that many open); more is
  `InvalidOptionValue` naming the kind. Each opens before the server
  listens, the start line saying its kind and time.
- **`--chat-model` is removed** (0.x: the minor is the breaking axis; the
  pull request title carries `!`). `Options.chat_model` goes;
  `Options.models` holds every kind.
- **Embeddings gain `Service.preload`** (`src/api/embeddings/service.zig`),
  the decision service's shape: locate, an `Opening` item on the worker
  that calls `pool.acquire`.
- **Completion**: a `Value.served` kind whose candidates are the registry
  entries and the three catalogues, each described by its kind
  (`decision`, `chat`, `embedding`) first; serve's `--model` uses it.
- Documents: `src/help.zig` serve page, `docs/guide/api.md` options,
  `docs/spec.md` serve row, `README.md`, `site/index.html` snippet,
  `scripts/api-check.py` (`--model` for `--chat-model`).
- Gates: `zig build test`, `zig fmt --check`, `make lint-py`,
  `make docs-check`, `make site-check`, `make verify-auto`; exercise
  `./zig-out/bin/nuclis serve` with no model (nothing opened, a decision
  and an embeddings request open theirs) and with `--model` of each kind,
  and `nuclis __complete serve --model ""`. Executable only: no tier.

## Deferred

Agreed and not scheduled; when one is picked up, it becomes a unit above.

- **Release v0.8.0** (the user, 2026-10-17), once the web tools' pull
  request merges: the release steps in AGENTS.md (*Versioning and
  releases*). `make version` now moves the site's download with the
  manifests, and the archive no longer ships `CHANGELOG.md`.

- **Saluki's speed goals** (restated by the user, 2026-10-15; the numbers
  and their arithmetic in `docs/models/saluki.md`, its **Goals**): decode at
  32,639 ≥ 18 tok/s, and a 32,639-token prompt prefilled in ≤ 7 minutes.
  Decode needs the 2-bit matvecs at about 200 GB/s (IQ2_XXS reads 144
  today; the limiter is instructions per weight, not bandwidth: at
  2.26 bits a weight, 200 GB/s leaves an estimated four instructions per
  weight on the 16-core GPU; `docs/engine/apple-gpu.md` § The merged
  gate+up on Saluki's encodings). The prefill goal is the entry below. Context past
  32,768 (`max_context` in `src/config.zig`) waits on the prefill goal.

- **Long-context prefill, every family** (the user's, 2026-10-15). Prefill
  at 512 is level with llama.cpp (0.89–1.02×) and falls behind as the
  prompt grows; at 32,639: Muse Glimmer 0.79×, Qwen3.8 0.71×, Saluki
  0.65×, Gemma 4 12B QAT 0.50×, E4B 0.49×, 26B-A4B 0.42×
  (`docs/benchmarks/README.md` § The benchmarks on macOS 27,
  `docs/models/saluki.md` § Speed). The suspect is the chunk attention
  (`nu_attention_chunk` / `_h`, `Backend.attentionChunk`), 30 % of a 16K
  prefill on Qwen in 2026-09; Gemma's 512-wide global layers run a
  variant that recomputes scores per 256 value columns, which may be why
  the Gemmas trail most. Two attempts closed: sharing K/V across a
  group's heads (slower) and register-level reuse (2–5 %), both at
  ~0.7 TFLOP/s and blamed on the per-tile instruction chain
  (`docs/engine/metal-backend.md` § Long-context prefill attention, and
  § second attempt). First step: a cooled `nuclis bench --profile` of
  each family's 16K prefill to confirm the attention share per family;
  then a scratch ablation as the split pass's unit did (§ The split
  pass at the memory roof found registers, guards, and a scale inside
  the products bounding decode, not bytes or ALU), and `make
  bench-attention` to rank variants. A saved prefix does not help here:
  each run pays the prefill.

- **Video input for EmbeddingGemma 2** (about one session). Frames at
  1 fps (at most 32) through the image encoder at 140 soft tokens each,
  an AVFoundation decoder beside the image and audio bridges, oracle
  fixtures from a short generated video, gates, `--video`, and a video
  part in the API. Google's processor facts: `docs/models/embeddinggemma.md`
  § The mmproj (`Video`).
