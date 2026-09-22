# AGENTS.md — nuclis

Instructions for coding agents working in this repository.

## Project and sources of truth

nuclis is a modular local inference engine and evaluation CLI built with
Zig 0.16, initially targeting Qwen3.8-27B on an Apple M4 Pro with 48 GB
unified memory. A lean agent loop embedded in `nuclis chat` (see the agent
section of [docs/spec.md](docs/spec.md)) gives the playground small coding
task abilities; it is not a second product.

Read [docs/architecture.md](docs/architecture.md) for the stack overview and
[docs/spec.md](docs/spec.md) for scope and acceptance criteria. Shared
workflow and toolchain conventions live in
[docs/development.md](docs/development.md); detailed reference documents
(benchmarks, Metal backend, CPU reference, GGUF, prompt profile) live under
[docs/reference/](docs/reference/). The plan and the log are described in
the session protocol below.

## The session protocol

Two files carry the project's state, and every session starts from them:

| File | Holds | Changes when |
| --- | --- | --- |
| [TODO.md](TODO.md) | the active plan: unfinished units only, with a *Where we are* hand-off note | a plan is written, a session ends, a unit closes |
| [docs/engineering-log.md](docs/engineering-log.md) | the durable, append-only record of every unit ever closed, with its evidence | a unit closes (entry **and** table row, together) |

**A fresh session is in one of two states.** Read `TODO.md` first; it
tells you which.

1. **It lists work.** Summarize *Where we are* and the next unit, ask how
   the user wants to continue, then continue that unit. Do not replan.
2. **It is empty.** Nothing is in progress. Ask what to work on, agree the
   theme with the user, and write the plan into `TODO.md` in its format
   (where-we-are note, order, unit table, unit designs) before touching
   code. There is no roadmap file. A theme that changes the architecture
   and spans several units is recorded as an ADR under `docs/adr/` **when
   the user asks for one** (never by default), and the plan cites it;
   ordinary units need no record beyond the plan and the log.

**Ending a session** leaves `TODO.md` able to restart the next one on
its own: the current unit's section says what the session delivered and
what remains, *Where we are* says where to pick up. A unit the plan marks
as several sessions closes once, at its end.

**A unit's section is written for whoever implements it next**, which may
be another model: files, functions, numbers, and gate commands, not
intent. A unit whose design waits on facts (a file's header, a
reference's driver) reads them in its first session and ends that session
by rewriting its section at that level, committed before any code.

**Closing a unit** happens in one commit: append its outcome, evidence,
files, and remaining limitations to the engineering log and add its row to
the log's table; update the documents it changed; delete its section and
row from `TODO.md`; refresh *Where we are*. When the last unit closes,
empty `TODO.md` back to its header.

**Nothing closes silently.** Work that lands outside a planned unit (a
catalogue entry, a process decision, a side fix) is still a unit: give it
the next identifier of its area and log it in the commit that lands it.

**Durable knowledge never lives only in `TODO.md`**: requirements go to
`docs/spec.md`, environment facts to `docs/development.md`, designs that
outlive a unit to a reference document, decisions that outlive the plan
to an ADR when the user asks for one.

## Session hygiene

One session per unit of `TODO.md` (two for units the plan marks as two
sessions). Inside a unit, compact at sub-unit boundaries (after a commit)
with a note of what to keep: the current unit, the changed files, the gate
commands, and any measured numbers not yet written into the docs. Never let
auto-compaction fire mid-measurement. Keep tool output out of the context:
pipe builds and benchmarks through `grep`/`tail`, read documents by section,
and write measurements into the docs as soon as they are taken so the
transcript is not the only copy. The hand-off is `TODO.md`, never the
conversation.

## Working mode

Act as an implementation collaborator: design, implement, test, review, and
document the requested work. Complete authorized work without unnecessary
confirmation. Respect requests to discuss or draft before implementation.

This project is also the user's way to understand Zig deeply. Write clear but
terse module docs and comments (see *Code comments*): ownership, invariants,
and non-obvious constructs, not prose. Keep documentation current with each
completed increment. Teach in chat, not in code comments; the agent is
authorized to write complete code, without a lesson or exercise workflow.
Periodically explain the component being built and its role in the inference
stack in chat, building on previous explanations.

[docs/llm-guide.md](docs/llm-guide.md) is the user's companion on the
inference stack itself (kernels, quantization, the runtime, model
architectures), not on the application or the agent. **Extend it only when
the user asks**, never as a routine part of closing a unit; when they do,
keep examples grounded in actual implementation status, and read the guide
before repeating or extending an explanation.

Use [docs/spec.md](docs/spec.md) to control scope. Distinguish accepted
decisions, proposed targets, and measured results. Resolve routine choices
within the task; raise material ambiguity with a concrete recommendation.

Inspect existing files and diffs before changing them. Preserve unrelated user
work. Update documentation when behavior or architecture changes, and remove
stale guidance instead of maintaining competing sources of truth.

## Repository architecture

- Three packages compose the build: `src/` (the executable: CLI, chat, agent
  loop, tools, `nuclis model`), `inference/` (the engine: runtime,
  quantization, tokenizer, sampling, backends, and its kernels), and
  `huggingface/` (Hub downloads over Xet; imported by `src/` only, never by
  `inference/`; its tests run under `zig build test`, its standalone binary
  under `make hf-downloader`). Build only from the root so the root's
  `.zig-cache` and `zig-out` are the only ones.
- Root `build.zig` and `build.zig.zon` compose them through path
  dependencies. Do not add fictitious dependencies for unimplemented packages.
- Executable `main.zig` files only parse arguments and compose/dispatch
  modules.
- Keep application-specific utilities inside `src/` until reuse justifies
  moving them to `inference/`.
- The tool name `nuclis` and the package name `.nuclis` are decided.

## Shared engineering rules

- Pass allocators explicitly; no global allocator. Document ownership and
  borrowed lifetimes, and clean up partially initialized state.
- Pass Io explicitly to I/O-capable code. Keep pure logic independent of I/O.
- Prefer small interfaces that hide substantial implementation detail.
  Add abstractions around concrete variation rather than hypothetical plugins.
- Validate untrusted input and bound allocations, output, and execution time.
- Use typed errors/results for expected failures; do not panic on bad input.
- Human-readable and machine-readable output derive from the same result types.
- External implementations (llama.cpp, ds4, ...) are references and validation
  oracles, never sources to copy. Implement from the format or mathematical
  contract and from our own CPU references; prove equivalence with pinned
  fixtures and traces. Format-defining constants and any third-party material
  that must exist in the tree are recorded in `THIRD_PARTY_NOTICES.md`, which
  links license texts on the web. Do not add per-directory license files.

## Inference engine rules

- Zig owns model semantics and execution planning.
- Objective-C exposes a C-compatible Metal interface; Metal Shading Language
  implements GPU kernels. Keep platform objects behind opaque handles.
- Shared runtime, tensor/quantization, tokenizer, sampling, and backend modules
  must not embed Qwen-specific layer schedules or prompt assumptions.
- Model adapters own weight mapping, graph construction, and state layout.
  Prompt profiles are separate from numerical architecture.
- Keep immutable weights separate from mutable session state. Recurrent state
  cannot be reset or rewound merely by truncating an attention KV cache.
- Validate GGUF metadata, dimensions, offsets, and actual tensor encodings.
  Reject unsupported combinations explicitly; do not silently requantize.
- Numerical correctness precedes optimization. Measure prefill and decode
  separately against an equivalent, pinned reference configuration.
- GPU resources must outlive submitted work. Test synchronization, cancellation,
  and cleanup separately from Zig allocator leak checks.
- Benchmarks must identify hardware, build, artifact, context, and methodology.
  Do not present estimates as measurements.

## Embedded agent rules

The agent in `nuclis chat` is a bounded playground loop, not a platform:
- Native tool-call conversion (encode and streaming decode) belongs to the
  model profile in `inference/`; execution belongs to the agent.
- Few, strong tools; bounded output sizes, result counts, and step budgets;
  truncation is always marked. Exceeding a limit is a typed tool result, not
  an abort. Failures are results; only inference-transport failure ends a turn.
- The workspace is the process's working directory; limits are host constants,
  never model-supplied.

## Validation and definition of done

For code changes, once build scaffolding exists:

1. Build the affected package and run relevant tests, including error paths.
2. Format/check changed Zig sources and build files.
3. Exercise the applicable happy path with the freshly built
   `./zig-out/bin/nuclis`, never an installed copy.
4. Run broader checks when shared code or unresolved risks justify them.
   The model-specific checks are gates in `gates.json`, tiered by cost and
   selected by changed paths (`make verify-changed`; the CPU tier only when
   selected), see [docs/development.md § Gates](docs/development.md#gates).
   `make verify` (the Metal tier) once per unit that touched the inference
   stack — judge by the work: a unit confined to the executable, the
   documents, or the scripts changes no numerical behaviour and needs no
   tier.
5. A change to the agent's system prompt (`src/agent/system_prompt.zig`),
   a tool description, or the loop's behaviour is measured on the
   playground task list before and after (`make agent-eval VARIANT=…`,
   [docs/development.md § The agent's task list](docs/development.md#the-agents-task-list));
   the pinned prompt text is re-pinned only with the new table.
6. Update relevant specifications, command help, and operational documentation.
7. Report what changed, validation performed, and material remaining limitations.

Unit tests live alongside source in `test` blocks and use
`std.testing.allocator`. Default tests must not require live network, real
credentials, GPU hardware, or large model downloads; full-model and Metal
tests are explicit targets using external artifacts.

For documentation-only changes, validate links, consistency, and removal of
stale references. Do not claim build/test execution when no code or build exists.

## Local toolchain notes

- The active developer directory is the Command Line Tools
  (`xcode-select -p`), which is what `zig build` uses for the Objective-C
  bridge. Do not switch it globally. Tools that only ship with Xcode
  (`xctrace`, Instruments) are reached per process through the environment:

  ```sh
  DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun xctrace record \
    --template 'Metal System Trace' --instrument 'Metal GPU Counters' \
    --output .zig-cache/trace/run.trace --no-prompt --launch -- <binary> <args>
  DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun xctrace export \
    --input .zig-cache/trace/run.trace --toc
  ```

  `make trace` wraps this for `bench`. On the M4 Pro the GPU limiter counter
  profile is reported unsupported and its tables export empty; the GPU
  timeline (`metal-gpu-intervals`) does record. Keep traces under
  `.zig-cache/trace/`; never commit them.

## Versioning and releases

- The root `build.zig.zon` `.version` is the single source of truth. The
  package manifests under `inference/` and `huggingface/` carry the same
  version because the Zig package format requires one, but the path
  dependencies ignore it; `make release` bumps all three together so they
  cannot drift. The executable exposes the version as `nuclis --version`, fed
  from the root manifest through build options.
- [Semantic Versioning](https://semver.org/) with the 0.x convention: while
  `0.y.z`, the **minor** is the breaking axis. `0.1.0` is the spec's v0.1
  acceptance (the 32K context record); until it passes, the tree stays on
  `0.1.0-dev`. Planned work never bumps the version; acceptance does.
- Release recipe: `make release` strips the `-dev` suffix (the version is
  derived from `build.zig.zon`, never passed in), writes the CHANGELOG
  section, commits `chore(release): vX.Y.Z`, tags it (annotated), and bumps
  to the next `X.(Y+1).0-dev` in a follow-up commit. It never pushes.
- Tag only forward, never retroactively. Benchmarks and test records cite the
  git revision, and published numbers cite the release tag once one exists.
- `CHANGELOG.md` starts at the first tag, assembled from Conventional
  Commits since the previous tag (`feat` → minor, `fix` → patch,
  `!`/`BREAKING CHANGE` → flagged).
- The Zig toolchain is a separate axis: `minimum_zig_version` pins source
  compatibility, the exact compiler used is recorded in every benchmark
  record, and a Zig upgrade is its own unit of work.

## Repository hygiene

### Commits and implementation tracking

Use [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/#specification):
`type(optional-scope): short description`. Use `feat` for features, `fix` for
bug fixes, and `docs`, `build`, `test`, `refactor`, `perf`, or `chore` when those
better describe the change. Scopes such as `nuclis`, `inference`, and `gguf`
are optional. Mark breaking changes with `!` or a `BREAKING CHANGE:` footer.

Commit each coherent, verified unit of work: a feature with its tests and docs,
a focused fix, or a self-contained documentation update. Do not commit every
small edit, and do not combine unrelated completed work into a large end-of-task
commit. Inspect the staged diff and run applicable checks before committing.
Use short subjects; add a body only when it explains a useful reason or tradeoff.
Routine local commits are authorized. Publishing or pushing is a separate action.
Do not add `Co-Authored-By:` or any other co-author/attribution trailer to
commits.

[TODO.md](TODO.md) tracks the units in progress and
[docs/engineering-log.md](docs/engineering-log.md) the
units that closed. Update them in the same commit as the work they describe.
Close a unit only when its acceptance checks pass; record partial work and
missing prerequisites explicitly in `TODO.md`. Keep requirements in
[docs/spec.md](docs/spec.md), and link evidence or supporting docs instead of
duplicating the specification.

### Code comments

A comment earns its place by saying what the code cannot: an invariant, an
ownership rule, a non-obvious reason, a hazard. Write the fewest words that
save the next reader the most time.

- **Module doc (`//!`): a short orientation.** What the module is, where it
  sits, and the one or two facts needed to use or change it safely — roughly a
  dozen lines. Design history, format specifications, and teaching belong in
  `docs/reference/` or `docs/llm-guide.md`; summarize and link, do not restate.
- **Declarations (`///`): the contract.** One or two sentences: what it does,
  what it owns, what it may return. Rationale only where it is surprising.
- **Inline (`//`): the non-obvious why, never the what.** If a comment restates
  the next line, fix the code or delete the comment.
- **No history or plan narration.** Do not write "was X, now Y", "step 7", or a
  unit identifier as the subject of a sentence (see *Unit identifiers*). A
  comment must stay true after the plan is gone.
- **No duplicated specifications.** If the text lives in `docs/`, the comment
  is one line and a link.
- Prefer one to three lines. A paragraph is a smell; a multi-paragraph comment
  block almost always belongs in a document.

### Unit identifiers

`TODO.md`, `docs/roadmap.md`, and `docs/engineering-log.md` label
units with an `AREA-NN` identifier (`AGNT-07`, `MODL-03`): a frozen
four-letter area and a zero-padded sequence number within that area.

The areas are frozen. Adding one means recording it in this table first,
never retrofitting an identifier onto closed work.

| Area | Covers |
| --- | --- |
| `APPS` | CLI surface, configuration, command behavior, user-facing application features |
| `AGNT` | the embedded agent loop, tools, and tool-call handling |
| `ENGN` | engine, runtime, scheduling, session state, prefill and decode performance |
| `KERN` | compute kernels and the KV cache |
| `MODL` | model adapters, quantization, tokenizer, prompt profiles, model artifacts |
| `TERM` | terminal rendering and interaction |
| `REPO` | repository, toolchain, versioning, release, and documentation infrastructure |

Numbers are assigned within an area in the order units close, are never
reused, and a unit that spans sessions keeps one number. The area letters are
arbitrary; the number carries no meaning beyond sequence.

`docs/engineering-log.md` is append-only and is never emptied, so a
closed unit's heading is its identifier's **permanent anchor**. Treat an
identifier as a *citation*, never as the explanation:

- **Code comments must stand alone.** Deleting every identifier from a comment
  must not lose what it says or why. Prefer "the profile will own decoding"
  over "tool-call decoding lives in the profile"; cite the identifier, at
  most, as an extra pointer.
- **Current docs** (architecture, llm-guide, reference) may cite an identifier
  as a link to its log entry, but the sentence must read correctly if the
  identifier is removed.
- **The log and the live plan** (the log itself, `TODO.md`, the ADRs) keep
  identifiers as keys. On close, the identifier migrates to the log and its
  anchor becomes permanent; never reuse an identifier for a different unit.

### User data

`~/.nuclis` is the user configuration/data root. `NUCLIS_HOME` may override
it with an absolute path. Models live under `~/.nuclis/models`; settings and
sessions live in tool-specific subdirectories beneath it.

Do not commit build caches, binaries, model weights, downloaded datasets,
benchmark artifacts containing private data, API keys, or user session data.
Use small, attributable test fixtures. Keep model files outside the repository.

`docs/` is the project documentation hub. Keep exactly one authoritative spec
([docs/spec.md](docs/spec.md)), link supporting docs from
[docs/architecture.md](docs/architecture.md), and record durable decisions
there when needed. Do not require an external project checkout to work here.
