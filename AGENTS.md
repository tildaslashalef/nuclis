# AGENTS.md — nuclis

Instructions for coding agents working in this repository.

## Project and sources of truth

nuclis is a modular local inference engine and evaluation CLI built with
Zig 0.17, for open-weight models on an Apple M4 Pro with 48 GB of unified
memory. A lean agent loop embedded in `nuclis agent` (see the agent section
of [docs/spec.md](docs/spec.md)) gives the playground small coding task
abilities; it is not a second product.

Read [docs/architecture.md](docs/architecture.md) for the stack overview and
[docs/spec.md](docs/spec.md) for scope and acceptance criteria. Workflow and
toolchain conventions live in [docs/development.md](docs/development.md);
every other document is listed in [docs/README.md](docs/README.md), one
folder per kind (`guide/`, `models/`, `engine/`, `app/`, `benchmarks/`).

## The session protocol

Work arrives as **units**, grouped into **milestones**: one unit is one
session and one commit, however many tasks it holds; one milestone is one
branch and one pull request. If `AGENTS.local.md` exists, read it: it names
the tracker and overrides the queue rules here. Without it, ask the user
what to work on.

| Where | Holds |
| --- | --- |
| the maintainer's tracker | the queue: the units agreed and not yet done, each with its design, in order, and a hand-off note |
| the unit's commit | the record, as its body: what changed and why, the evidence, what remains (`.github/pull_request_template.md` gives the shape) |
| the milestone's pull request | the list of its units; the user reviews and merges it |

**A fresh session reads the queue first.** If it lists work, summarize the
hand-off and the next unit, ask how the user wants to continue, then
continue that unit; do not replan. If it is empty, ask what to work on,
agree the theme, and write the plan into the tracker before touching code.
There is no roadmap, and GitHub's issues, projects, and wiki are off.

**A unit's design is written for whoever implements it next**, which may
be another model: files, functions, numbers, and gate commands, not intent.
Refer to code and documents by name and heading, never by line number. A
unit whose design waits on facts reads them first and rewrites its design
at that level before any code.

**A milestone has one branch**, named for its theme (`hostile-input`), from
an up-to-date `main`, or from the previous milestone's branch when that one
has not merged yet (rebased onto `main` before its pull request). A unit
starts as soon as the unit before it is committed: nothing waits for a
merge or for CI.

**Finishing a unit**: the checks pass, here and for Linux (`make
build-linux`; CI's Linux job runs once per milestone); the documents it
changed are updated; the unit is one commit with a Conventional Commit
subject and the record as its body; the hand-off names the next unit. A
session that ends early leaves its work committed and the hand-off saying
what was delivered and what remains; the next session folds that into the
unit's one commit.

**Finishing a milestone**: after its last unit, on the user's word, the
branch is pushed and one pull request opened, titled for the milestone,
its description listing the units. The user merges it by rebase, so each
unit stays one commit and `git log main` is the history of the work.

**Nothing lands silently.** A side fix found inside a unit, and work the
user asks for during a session, belong to that unit's commit and its
record. Only work outside any milestone is a small unit of its own, with
its own branch and a pull request of that one commit. Every pull request
is merged by rebase; GitHub allows no other method.

**Durable knowledge never lives only in the tracker or a commit message**:
requirements go to `docs/spec.md`, environment facts to
`docs/development.md`, designs and the decisions behind them to the
document under `docs/` that owns them.

**Session hygiene.** One session per unit. Compact only between units,
with a note of what to keep: the unit, the changed files, the gate
commands, and measured numbers not yet written into the docs. Never let
auto-compaction fire mid-measurement. Keep tool output out of the context
(pipe builds and benchmarks through `grep`/`tail`, read documents by
section) and write measurements into the docs as soon as they are taken.

## Working mode

Act as an implementation collaborator: design, implement, test, review, and
document the requested work. Complete authorized work without unnecessary
confirmation; respect requests to discuss or draft first. Resolve routine
choices within the task; raise material ambiguity with a concrete
recommendation. Inspect files and diffs before changing them, and preserve
unrelated user work.

This project is also the user's way to understand Zig deeply. Teach in
chat, not in code comments: periodically explain the component being built
and its role in the inference stack, building on earlier explanations. The
agent writes complete code; there is no lesson or exercise workflow.

[docs/llm-guide.md](docs/llm-guide.md) is the user's companion on the
inference stack. **Extend it only when the user asks**; then read it first
and keep its examples grounded in what is implemented.

Use [docs/spec.md](docs/spec.md) to control scope. Distinguish accepted
decisions, proposed targets, and measured results. Update documentation
when behavior or architecture changes, and remove stale guidance instead of
keeping competing sources of truth.

## Repository architecture

- Three packages compose the build: `src/` (the executable: CLI, agent
  loop, tools, server, `nuclis model`), `inference/` (the engine: runtime,
  quantization, tokenizer, sampling, backends, kernels), and `huggingface/`
  (Hub downloads over Xet; imported by `src/` only, never by `inference/`).
  Build only from the root, so the root's `.zig-cache` and `zig-out` are
  the only ones.
- `tools/gates/` is the gate runner (`nuclis-gates`), a separate executable
  that imports none of the three packages. `scripts/` holds the rest of the
  tooling by purpose: `gate/`, `bench/`, `repo/`, `app/`, `fixtures/`,
  `oracle/`.
- Root `build.zig` and `build.zig.zon` compose the packages through path
  dependencies. Do not add fictitious dependencies for unimplemented
  packages.
- Executable `main.zig` files only parse arguments and dispatch.
- Keep application utilities inside `src/` until reuse justifies moving
  them to `inference/`.
- `site/` is the engine.nuclis.dev website: static files, no build,
  deployed from `main`
  ([docs/development.md § The website](docs/development.md#the-website)).
  Its figures are copies of the README's that `make site-check` holds
  equal; a change to it is looked at in screenshots, not gates.

## Engineering rules

- Pass allocators and Io explicitly; no global allocator. Document
  ownership and borrowed lifetimes, and clean up partially initialized
  state. Keep pure logic independent of I/O.
- Prefer small interfaces that hide substantial implementation. Add
  abstractions around concrete variation, not hypothetical plugins.
- Validate untrusted input (files, sockets, model output) and bound
  allocations, output, and execution time. Use typed errors for expected
  failures; never panic on bad input.
- Human-readable and machine-readable output derive from the same result
  types.
- **nuclis is its own reference.** Implement from the format or the
  mathematical contract, and validate new work against nuclis's
  already-validated paths: the CPU reference, Metal against it, and the
  same model in an encoding already validated
  ([docs/models/README.md § 4](docs/models/README.md#4-validate-against-nuclis-itself)).
  External implementations are never sources to copy and are not oracles;
  read one only as the definition of a format that has no other. Fixtures
  pinned from them earlier stay. Third-party material that must be in the
  tree is recorded in `THIRD_PARTY_NOTICES.md`, with no per-directory
  license files.

### The inference engine

- Zig owns model semantics and execution planning. Objective-C exposes a
  C-compatible Metal interface behind opaque handles; Metal Shading
  Language implements the GPU kernels.
- Shared runtime, quantization, tokenizer, sampling, and backend modules
  must not embed a family's layer schedule or prompt assumptions. Model
  adapters own weight mapping, graph construction, and state layout; prompt
  profiles are separate from numerical architecture.
- Keep immutable weights separate from mutable session state. Recurrent
  state cannot be reset or rewound by truncating an attention KV cache.
- Validate GGUF metadata, dimensions, offsets, and actual tensor encodings.
  Reject unsupported combinations explicitly; never requantize silently.
- Numerical correctness precedes optimization. Measure prefill and decode
  separately, against nuclis's own previous record on the same machine and
  method.
- GPU resources must outlive submitted work. Test synchronization,
  cancellation, and cleanup separately from allocator leak checks.
- Benchmarks identify hardware, build, artifact, context, and method, and
  cite the git revision (the release tag once one exists). Never present an
  estimate as a measurement.

### The embedded agent

The agent in `nuclis agent` is a bounded playground loop, not a platform.

- Native tool-call conversion (encode and streaming decode) belongs to the
  model profile in `inference/`; execution belongs to the agent.
- Few, strong tools; bounded output sizes, result counts, and step budgets;
  truncation is always marked. Exceeding a limit is a typed tool result,
  not an abort. Failures are results; only inference-transport failure ends
  a turn.
- The workspace is the process's working directory; limits are host
  constants, never model-supplied.

## Validation and definition of done

1. Build the affected package and run the relevant tests, error paths
   included. Unit tests live beside the source in `test` blocks and use
   `std.testing.allocator`; default tests need no network, credentials,
   GPU, or model download.
2. Format changed Zig sources and build files; lint changed Python
   (`make lint-py`).
3. Exercise the happy path with the freshly built `./zig-out/bin/nuclis`,
   never an installed copy. A change to the chat surface (`src/tui/`,
   `src/agent/root.zig`) is exercised through the screenshot harness,
   `make shot ARGS='<steps>'`, and the record cites the captures
   ([docs/development.md § Looking at the agent without a person at the
   keyboard](docs/development.md#looking-at-the-agent-without-a-person-at-the-keyboard)).
4. Run the gates ([docs/development.md § Gates](docs/development.md#gates)).
   Always: `make check` and `make build-linux`. Before a unit's commit:
   `make verify-auto`, which selects the families the change reaches (the
   canaries for shared code, nothing outside the engine) and names any
   further tier. Run a named tier only when it applies: `make verify-cpu`
   when a unit changes what the CPU reference computes (an existing CPU
   kernel's or decoder's arithmetic, a family's `*_runtime.zig` forward, a
   projector's CPU `Runtime`) or brings up a family or draft source;
   `make verify-long` when it changes attention, the KV cache, or a
   windowed schedule. Every tier, strict, before a tag.
5. A change to the agent's system prompt, a tool description, or the
   loop's behaviour is measured on the task list before and after
   (`make agent-eval VARIANT=…`,
   [docs/development.md § The agent's task list](docs/development.md#the-agents-task-list));
   the pinned prompt is re-pinned only with the new table.
6. Update the specifications, command help, and guides the change touches.
   For documentation-only work, `make docs-check`; do not claim a build or
   test that did not run.
7. Report what changed, what was validated, and what remains.

## Local toolchain

- The project's Zig is `~/.local/opt/zig/stable`, managed by the user's
  fish function `zigup`. Read the standard library at `stable/lib/std/` and
  the language reference at `stable/doc/langref.html`, never from the web.
  `~/.local/bin/zig` follows `current`, which the user may point elsewhere:
  check `zig version` against `stable/zig version` before building and say
  so if they differ. Switching links is the user's call; `zigup install
  x.y.z` fetches another version without switching, for A/B builds.
- Zig 0.17 has no `**` array-repetition operator: repeat with `@splat`
  (`&@as([64]u8, @splat('m'))`); `++` still concatenates. Check any
  construct remembered from older Zig against the langref before writing it.
- The active developer directory is the Command Line Tools, which is what
  `zig build` uses for the Objective-C bridge. Do not switch it globally.
  Tools that ship only with Xcode are reached per process
  (`DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun …`);
  `make trace` and `make capture` wrap them
  ([docs/development.md § GPU counters by capture](docs/development.md#gpu-counters-by-capture)).
  Keep traces under `.zig-cache/trace/`; never commit them.

## Versioning and releases

The root `build.zig.zon` `.version` is the last released version;
`make version V=X.Y.Z` sets it everywhere it must agree. Semantic
Versioning with the 0.x convention: the minor is the breaking axis. Tags
never move. The Zig toolchain is its own axis, and a Zig upgrade is its own
unit ([docs/development.md § Versioning](docs/development.md#versioning)).

When the user says "release", an agent:

1. runs `make release-draft` and writes the highlights to
   `.zig-cache/highlights-vX.Y.Z.txt`, at the version it suggests unless
   the user names one;
2. runs the release tiers the user wants (`make verify`, `make verify-cpu`,
   `make verify-long`, `make verify-release`; all may be skipped when no
   engine code changed, and the record says so);
3. shows the user the highlights, opens the version pull request with
   `make release V=X.Y.Z TIERS='<what ran>'`, and stops.

Merging it and `make tag V=X.Y.Z` happen only on the user's word for that
release; `release.yml` then builds, attests, and publishes.

## Repository hygiene

### Commits and pull requests

[Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/#specification):
`type(optional-scope): short description`, with `feat`, `fix`, `docs`,
`build`, `test`, `refactor`, `perf`, `ci`, or `chore`, and `!` or a
`BREAKING CHANGE:` footer for a breaking change. A unit's commit subject
and a pull request's title follow the same form. Work is named by what it
does, never by an invented identifier.

Commit each coherent, verified step while a unit is in progress; when it is
finished, fold the steps into the unit's one commit. Inspect the staged
diff and run the applicable checks before committing.

**Authorized without asking:** commits on the milestone's branch. **Ask
first, every time:** pushing a branch, opening a pull request, updating its
description. **Never:** pushing to `main`, merging a pull request,
force-pushing a branch others may have fetched, rewriting `main`'s history,
or opening issues on GitHub.

### Code comments

A comment earns its place by saying what the code cannot: an invariant, an
ownership rule, a non-obvious reason, a hazard.

- **Module doc (`//!`):** what the module is, where it sits, and the one or
  two facts needed to use or change it safely; about a dozen lines. Design
  history and format specifications belong in a document under `docs/`:
  summarize and link.
- **Declarations (`///`):** the contract in one or two sentences: what it
  does, what it owns, what it may return.
- **Inline (`//`):** the non-obvious why, never the what.
- **No history or plan narration** ("was X, now Y", "step 7", "phase 2"): a
  comment must stay true after the plan is gone.
- Prefer one to three lines. A paragraph almost always belongs in a
  document.

### User data and documents

`~/.nuclis` is the user's configuration and data root (`NUCLIS_HOME`, an
absolute path, overrides it): models under `models/`, settings and sessions
in tool-specific subdirectories. Do not commit build caches, binaries,
model weights, downloaded datasets, credentials, or session data. Use
small, attributable test fixtures.

`docs/` holds exactly one authoritative spec
([docs/spec.md](docs/spec.md)). Every document is listed in
[docs/README.md](docs/README.md) and lives in the folder its kind names.
`make docs-check` holds every link; `make docs-check ARGS='--move old=new'`
moves a document with its references.
