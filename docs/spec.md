# nuclis — Inference Engine Specification

Status: implementation in progress. GGUF inspection, the initial Qwen structural
adapter, CPU numerical references, bounded vocabulary loading, native qwen35
text encoding/decoding, and a bounded text prompt profile are implemented.
Native CPU generation and short full-model comparisons are implemented; the
GPU-resident Metal backend passes the same comparisons
([reference/metal-backend.md](reference/metal-backend.md)); the 32K
acceptance workload is measured against the reference
([reference/bench.md § Acceptance runs](reference/bench.md#acceptance-runs)). Name and implementation languages are decided. The
interactive agent is specified in [agent-spec.md](agent-spec.md): its
terminal surface is implemented as `nuclis agent`, its tool layer has not
started. See [reference/gguf-inspection.md](reference/gguf-inspection.md)
for inspection behavior and [reference/generation.md](reference/generation.md)
for CPU generation.

## Purpose

**nuclis runs language models locally with custom compute kernels, initially
optimized for coding with Qwen3.8-27B on an Apple M4 Pro with 48 GB unified memory.**

Its first executable is a model evaluation CLI; the interactive agent
([agent-spec.md](agent-spec.md)) is also product scope. Shared engineering conventions are
documented in [development.md](development.md).

## Decisions and initial scope

- Zig 0.16.x owns loading, model execution, memory planning, tokenization,
  sampling, the evaluation CLI, and benchmarks.
- Objective-C provides a C-compatible interface to Metal. It owns Metal object
  lifetimes and command encoding; model semantics remain in Zig.
- Metal Shading Language implements GPU kernels.
- First supported model: the text path of Qwen3.8-27B, using one pinned GGUF.
- First production backend: Metal on macOS arm64, tuned on the target M4 Pro.
- CPU reference calculations support numerical tests and diagnosis. A fast CPU
  inference backend is not a first-release requirement.
- One active generation session initially. Weights are immutable and separate
  from session state so future multiple sessions do not require copying weights.
- Future architectures must have a defined integration path from the beginning.
  Supporting arbitrary GGUF models is not implied by parsing the container.

## Target model and download

Use **Qwen3.8-27B-UD-Q4_K_M.gguf**, approximately **16.5 GB**, from
[Unsloth's model repository](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-UD-Q4_K_M.gguf).
This is the initial speed/memory compromise; coding quality must be evaluated,
not inferred from the quantization name.

Published file SHA-256:
`322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482`.

Download with nuclis itself (MODL-02/MODL-03; the `huggingface` package: pinned
commit, SHA-256 verified, atomic publication, a provenance sidecar beside
the file). The catalogue name `qwen3.8-27b` pins repository, file, commit
`4ca720788d1e01f1bff70c033e0d0028fd02e502`, and the digest above:

```sh
nuclis model pull qwen3.8-27b          # --all adds the vision projector and MTP head
nuclis model ls
```

The file lands in `$HOME/.nuclis/models/unsloth/Qwen3.8-27B-GGUF/`
([development.md § Model download](development.md#model-download)) and is
the default `engine.model`; `--model` and the configuration take a
registry entry name, the catalogue name, or a path.

Separate vision projector, MTP, importance-matrix, and alternative quantization
downloads are not needed for the initial text inference path. The pinned GGUF
itself contains an auxiliary prediction block: its metadata declares 65 blocks
and one next-token prediction layer. The adapter must account for this explicitly
while implementing the 64 main decoder layers. Metadata inspection must
enumerate the actual tensor encodings: a filename containing Q4 does not mean
every tensor is Q4. Implement all encodings required by this exact artifact,
and reject unsupported encodings explicitly. Do not silently requantize it.

Use the shared user root and `NUCLIS_HOME` override documented in
[development.md](development.md#user-directories). `--model` (a registry
entry, a catalogue name, or a path) takes precedence over the configured
model, which defaults to the catalogue entry under that root's `models/`
directory. The prompt profile is selected by the file's chat-template
digest; `--prompt-profile <p>`, a registry entry's `profile`, or the
catalogue entry's `profile` forces one on a file whose template is not
pinned, with a notice (the catalogue's pin is how `bonsai-2-27b` renders
the Qwen3.8 protocol its upstream template does not carry the digest of).

The [published configuration](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/main/config.json)
uses the Qwen3.5 architecture family despite the Qwen3.8 model name. It describes
64 layers with full attention every fourth layer, hidden size 5120, and dense
feed-forward networks. The
[model card](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/main/README.md)
identifies the other layers as Gated DeltaNet. The loader must validate the GGUF
architecture identifier, dimensions, metadata, and tensors against the supported
implementation rather than dispatching from the file's display name.

The [upstream Qwen model card](https://huggingface.co/Qwen/Qwen3.8-27B)
enables thinking and preservation of thinking history by default, with configurable
reasoning effort. Preserve complete assistant turns in the future agent renderer
and test both thinking and non-thinking modes against pinned prompt fixtures.
Sampling defaults differ by mode and must be recorded explicitly in evaluations
(implemented: the per-mode profiles in
[reference/generation.md](reference/generation.md#sampling-profiles-and-the-selection-chain-modl-01)).
The published 262,144-token native context is model metadata, not a promise that
nuclis can serve that context on the target Mac.

## Repository layout

Target layout. The executable package, inference package, GGUF/tensor modules,
and root build exist; other modules below are planned:

```text
docs/spec.md                     product and architecture specification
src/                             executable package: CLI, terminal surface, agent loop, tools
  build.zig
  build.zig.zon
  src/main.zig                   argument parsing and dispatch
  src/model.zig                  `model pull` / `model ls`: download, provenance sidecars, listing
  src/tui/                       terminal surface, engine-free (TERM-01)
  src/agent/                     agent composition: session, engine, tools (TERM-01)
huggingface/                     Hub download library (Xet), Zig package; imported by src/ only
inference/                       reusable inference library, Zig package
  build.zig
  build.zig.zon
  src/
    root.zig                     public engine interface
    runtime/                     model and session lifecycle, execution plans
    formats/gguf.zig             container and tensor metadata
    tensor/                      views, shapes, storage encodings
    quant/                       quantization layouts and CPU reference decoding
    tokenizer/                   supported tokenization algorithms
    sampling/                    sampling and stop policies
    models/qwen35/               first architecture adapter
    backends/cpu/                reference execution for tests
    backends/metal/              Zig backend and Objective-C bridge
  kernels/metal/                 shared kernels and specialized kernel modules
build.zig                        aggregate package build and test steps
build.zig.zon                    local path dependencies
```

The executable package lives in `src/`. Reusable inference code belongs in
`inference/`; it does not depend on the executable or on `huggingface/`.
Kernels belong to that library. Do not split every internal module into a
separate Zig package. The root build registers the executable, inference,
and huggingface packages.

## Module ownership

| Shared module | Owns | Does not own |
| --- | --- | --- |
| GGUF | Bounds-checked metadata and tensor directory parsing, backing storage lifetime | Architecture equations or Qwen tensor-name interpretation |
| Tensor and quantization | Shapes, strides, encoded blocks, validated views, reference decoding | Layer order, model-specific weight choices |
| Tokenizer | Supported algorithms, vocabulary handling, streaming detokenization | A universal assumption about prompt templates |
| Runtime | Model/session lifecycle, cancellation, memory budgets, execution-plan submission | Assumption that every model uses transformer layers or only a KV cache |
| Sampling | Explicit seeded RNG, greedy and configured sampling, stop conditions | Tool execution or agent policy |
| Metal backend | Device capabilities, buffers, pipelines, kernel dispatch, synchronization, timing | Chat formats, GGUF model naming, layer semantics |
| Model adapter | Metadata validation, weight mapping, execution plan, state layout, prompt profile | CLI rendering or platform object management |

Qwen-specific work includes its hybrid layer schedule, gated attention,
DeltaNet recurrence and convolution, normalization variants, RoPE configuration,
weight mappings, and its text prompt profile. The prompt profile owns supported
chat rendering, special tokens, thinking controls, and eventual tool-call
encoding. It is separate from the numerical architecture: another checkpoint
can share the architecture while using different prompt conventions.

Matrix operations, reductions, normalization, positional transforms, attention,
and recurrent mathematical operations may be shared where their contracts match.
Qwen-specific fused kernels remain explicitly specialized until another model
demonstrates reuse. Sharing a name such as RMSNorm is insufficient evidence that
two models use identical equations or weight conventions.

## Interfaces and extension rules

The public engine interface should let callers load a model, create a session,
evaluate input, stream generation, reset state, and release resources. Return
typed results and metrics; callers choose how to render them. Allocators and Io
are explicit. Document which returned data is owned and which is borrowed.
Status: `inference.engine` (`Engine.open`/`deinit`, `Model.step`/`reset`,
`runLoop` with `Hooks` and the layer observer) is that interface as of
2026-09-08; explicit adapter and profile registration (`models.table`,
`profiles.Profile`, selected at load) landed 2026-09-11 (MODL-04); the second
production family (Gemma 4 12B, MODL-05–MODL-08, 2026-09-11/12) went through the
seam with the profile owning its stop set and reasoning markers and one
new storage encoding (Q4_0) added as kernels plus fixtures without an
edit to the parser, the sampler, or the generation loop; the starter
guide is [reference/new-model-guide.md](reference/new-model-guide.md);
typed thinking/answer/stop completion events landed in AGNT-01 session 1, and
the shared tool definitions/history, their validation, and explicit
`ToolsUnsupported` landed in session 2; both profiles now render and decode
their native tool syntax against pinned fixtures (Qwen in AGNT-05/06, Gemma in
AGNT-09), so the agent's tool loop is model knowledge end to end
([agent-spec § Format boundary](agent-spec.md#format-boundary--inference-side-agnt-01)).

Internally, separate three seams:

1. **Architecture:** a model adapter validates weights and produces executable
   work with an architecture-owned state layout. Use explicit registration;
   no dynamic plugin system is needed initially.
2. **Execution:** a backend prepares and runs that work. CPU reference execution
   and Metal provide concrete implementations for the operations under test.
3. **Prompt profile:** tokenization configuration and message rendering are
   selected for the checkpoint, independently of kernel dispatch.

Use a limited, typed execution representation sufficient for current operations.
Allow architecture-specific operations with explicit inputs, outputs, state
effects, and capability checks. Do not build a general tensor compiler first.
Resolve architecture dispatch and pipeline selection during loading/preparation,
and keep per-token work free of repeated string-based discovery.

An additional architecture should require a new adapter, its tests and prompt
profile, and any genuinely new mathematical operations. It should not require
editing the GGUF parser, sampling policy, CLI generation loop, or Metal object
lifecycle. Verify this rule with a small independent dense-attention test model
before declaring the architecture seam stable; this does not promise a second
production model in v0.1.

Session state is opaque to callers. Full-attention KV buffers and recurrent
state have different lifetimes and update semantics. Never implement generic
rewind or prefix reuse by truncating KV alone. Initial reset rebuilds all state.
Rollback is an explicit snapshot/restore contract (accepted, ENGN-06): a
caller-owned copy of the used state, restorable only into a session of the
same capacity and layout ([reference/session.md](reference/session.md)).
Prefix reuse beyond that remains a future contract.

## Objective-C / Metal interface

Expose C-compatible functions and opaque handles to Zig. Keep Objective-C
objects, exceptions, and ownership conventions behind this interface. Return
explicit errors, with matched creation/destruction operations and cleanup on
partial initialization failure. No process-global model or device singleton.

The bridge manages devices, queues, buffers, pipeline creation, command encoding,
submission, completion, and error reporting. Zig owns the model execution plan;
the backend chooses dispatch geometry and legal kernel fusion. GPU-visible
weights and buffers must outlive submitted work. Synchronize before CPU access
or destruction, and document whether each operation records, submits, or waits.

Batch dispatches to avoid waiting after every operation. Reuse scratch storage
and compiled pipelines. Prefer mapped or shared weight storage where validated
against Metal alignment, buffer-size, residency, and lifetime requirements;
unified memory alone does not make every file mapping a valid Metal buffer.

ds4 uses Objective-C Metal glue exposed to a C engine, as shown in
[ds4_metal.m](https://github.com/antirez/ds4/blob/main/ds4_metal.m).
It is an implementation reference, not a code-size target. Any adapted code
must retain its applicable license and attribution.

## CLI proposal

`inspect`, `validate`, `generate`, `bench`, `tokenize`, `config`, `model`, `agent`, `--help`, and `--version` are implemented. Inspection currently reports
the directory, declared dimensions, storage histogram, and validated tensor ranges;
`validate` checks the initial Qwen structural profile and returns a summary of
text and auxiliary tensors. Tokenizer compatibility, numerical validation, and
context-dependent memory estimates remain pending. See [reference/qwen-validation.md](reference/qwen-validation.md).
`generate` accepts `--prompt` or `--prompt-file` with raw/chat, streaming,
seeded sampling (the official per-mode profiles by default; per-option
flags including `min_p` and penalties), `--think` reasoning effort,
structured stop reasons including
Ctrl-C cancellation, and timings. `bench` is implemented as described in
[reference/bench.md](reference/bench.md); `agent` as
specified in [agent-spec.md](agent-spec.md); `config` reads and writes the engine
configuration file described in
[development.md § Configuration file](development.md#configuration-file)
(precedence: built-in defaults < profile < file < registry entry < flags);
`config set` changes one key of it in place; `model pull`, `model ls`, and
`model inspect` download (and, with `--register`, name in the file), list,
and judge artifacts as described in
[development.md § Model download](development.md#model-download); `eval`
describes the target interface:

```text
nuclis inspect --model <path> --json
nuclis validate --model <path> --json
nuclis generate --model <path> --prompt-file <path> --max-tokens 256
nuclis bench --model <path> --prompt-file <path> --max-tokens 256 --json
nuclis bench --model <path> --prompt-tokens <json-path> --max-tokens 128 --ctx-size 32768 --json
nuclis tokenize --model <path> --prompt-file <path> [--raw] --json
nuclis agent --model <path> --think low
nuclis agent -p "<prompt>" [--json] [--session <path>]   (print mode)
nuclis config init | show [--json]
nuclis model pull <owner/repo> [--file <name>] [--revision <rev>] [--role <role>] [--force] [--json]
nuclis model inspect (<name> | <owner/repo> --file <name>) [--revision <rev>] [--json]
nuclis model ls [--json]
nuclis eval --model <path> --dataset <jsonl-path> --json
nuclis --help
nuclis --version
```

- `inspect`: artifact identity, architecture, dimensions, tensor-type histogram,
  supported/unsupported features, and estimated memory for a requested context.
- `generate`: text streaming, raw-prompt or supported chat mode, explicit context
  and output budgets, seed, sampling controls, and thinking on/off where supported.
- `bench`: repeated cold and warm measurements with explicit prompt token counts,
  output length, warmups, and separate load/prefill/decode timings; a prompt
  may be given as a token array so a reference measurement's exact input is
  reproduced rather than re-tokenized.
- `tokenize`: the prompt as `generate`/`bench` would feed it (raw or one
  rendered turn), its token IDs and each token's byte offset in the rendered
  text; reads the artifact's header only.
- `eval`: teacher-forced scoring of explicit prompt/continuation pairs, token
  counts, negative log likelihood, and optional reference-logit comparisons.
  An evaluation dataset must define tokenization and scored spans. It does not
  execute model-generated code.

Generation text goes to stdout and diagnostics to stderr. JSON output uses
versioned result types shared with human-readable metrics. Text reports are
styled with the agent's palette (`src/tui/style.zig` over
`src/tui/theme.zig`, gruvbox dark by default) only on a terminal that
advertises color, at the strongest level it advertises (truecolor,
256-colour, or the sixteen ANSI slots); `--json`,
pipes, `NO_COLOR`, and an unset or `dumb` `TERM` get plain bytes, and the error
line on stderr follows the same rule. Report stop reasons
such as EOS, token budget, context limit, cancellation, or execution failure.
Never silently truncate prompts to fit context. Stream valid UTF-8 even when a
token ends inside a character. Full arbitrary Jinja support is not required;
the initial text renderer must match fixtures from the pinned template.

## Interactive agent

Accepted interface requirement: `nuclis agent` is the interactive surface
of the engine (named `nuclis chat` until TERM-01). It owns a prompt
editor and conversation presentation, streams model output, supports
folding thinking without deleting it from the model conversation, shows a
status bar with generation state, token counts, context usage, and
measured prefill/decode rates (never fabricated ones), and handles
cancellation, new-session reset, terminal resize, and clean terminal
restoration. It records each conversation in an append-only JSONL file
under `~/.nuclis/agent/sessions/` and exports it as markdown on request,
takes slash commands (`/new`, `/resume`, `/ctx`, `/think`, `/save`, `/help`)
with completion for those and for workspace paths, accepts a message typed
while a turn runs and sends it when the turn ends, resumes a saved session
(`/resume`, `--resume <id>`: a replay through the profile, not a state
restore), and runs one turn without a
terminal at all (`--print`, text or JSON event lines). Its second phase adds a bounded tool layer (six fixed tools,
no permission system, no extension mechanism) so the local model can
complete small coding tasks in the working directory. Native tool-call
conversion belongs to the model profile in `inference/`; execution belongs
to the agent.

The full specification — decisions (name, inline rendering with native
scrollback, JSONL sessions, theme), the phase-1 terminal surface and its
module API, session storage, the phase-2 loop, tools, and delivery
sequence — is [agent-spec.md](agent-spec.md). Phase 1 closed with TERM-01 on
2026-09-12; phase 2 closed with AGNT-07 on 2026-09-14.

## Performance and memory targets

Primary workload: one user performing coding tasks. Measure both time to first
token after a substantial code prompt and steady generation latency.

- Bring-up context: 8,192 total tokens, including generated output.
- Proposed v0.1 acceptance context: 32,768 total tokens on the 48 GB target.
- Larger contexts are later targets, subject to measurements and memory budgets.
- Keep weights quantized during the normal GPU path; avoid a permanent full
  floating-point duplicate. Use chunked prefill and planned scratch storage.
- Reserve headroom for macOS and the user's development tools. Validate actual
  memory pressure, GPU allocation limits, and swap behavior on the target Mac.

Using the published attention dimensions, an F16 KV cache for the 16 full-attention
layers is approximately 64 KiB per token, or 2 GiB at 32,768 tokens. This is a
planning calculation, not total memory consumption: recurrent state, weights,
activations, scratch buffers, and allocation overhead are additional.

Establish a pinned llama.cpp Metal baseline using the exact GGUF and equivalent
input tokens, context, cache precision, sampling, and output budget. Confirm that
the reference revision supports this checkpoint before treating it as an oracle.
Compare 512-, 4,096-, and 16,384-token coding prompts, then exercise the 32K limit.
Record chip/GPU configuration, OS, compiler versions, engine revisions, artifact
hash, power mode, warmup policy, and memory use with each result.

Initial optimization objective: approach reference performance, then improve
identified bottlenecks without reducing numerical correctness or coding quality.
Numeric throughput acceptance thresholds will be set after the baseline exists.
No absolute tokens/second or speedup claim is made by this specification.

Measured (2026-09-10, [reference/bench.md § Acceptance runs](reference/bench.md#acceptance-runs),
[benchmarks/nuclis-2026-09-10.json](benchmarks/nuclis-2026-09-10.json)): on
the reference's own token arrays, 128 outputs, F16 KV, context 32,768, all
four lengths complete on the token budget. Prefill / decode tok/s against
the reference: 512 = 90.45 / 10.62 vs 89.19 / 9.66; 4,096 = 83.70 / 10.20
vs 89.26 / 9.21; 16,384 = 62.70 / 8.27 vs 74.07 / 7.32; 32,639 = 49.55 /
7.55 vs 67.28 / 6.71. Session block 2.15 GiB at 32K; process peak footprint
2.84 GB with the 16.46 GB weights mapped for the GPU; swap did not grow.
Cold start after `purge` (512 tokens, one run): first token 11.79 s against
5.66 s warm, decode unchanged at 10.55.

## Verification and milestones

1. **Artifact and baseline:** verify the hash, inventory encodings and metadata,
   validate reference support, record timings, and establish numerical fixtures.
2. **Loading and correctness tools:** bounds-checked GGUF parsing, tokenizer and
   prompt fixtures, reference quantization operations, and `inspect`.
3. **First text generation:** the full Qwen text graph, correct hybrid state,
   straightforward Metal kernels, generation, scoring, and cancellation.
4. **Optimization:** specialized decode and chunked-prefill kernels, reduced
   synchronization, memory reuse, profiling, and repeatable benchmark comparison.
5. **v0.1 evaluation CLI:** context/memory validation, extension-seam test,
   documented commands, and published correctness/performance results.

Tests must cover malformed/truncated GGUF data and arithmetic overflow, encoded
block decoding, tokenization, prompt formatting, reference-versus-GPU operations,
and model-state behavior. Compare teacher-forced logits or intermediate values
with documented per-operation tolerances. Fluent output alone is not validation.
Greedy disagreements require examining margins and numerical errors rather than
assuming bit-identical GPU arithmetic.

Verify that chunked prefill and incremental processing agree within tolerance,
session reset reproduces a fresh session, and different sessions do not share
mutable state. Exercise cleanup after load errors, cancellation, and GPU errors.
Pure Zig tests use `std.testing.allocator`; Metal resource cleanup needs separate
lifecycle checks because that allocator cannot observe Objective-C/GPU allocations.

Default tests require neither network access nor a 16.5 GB model. Use small
fixtures locally; full-model and Metal tests are explicit targets with an external
model path. Run the freshly built CLI for each applicable happy-path check.
Use `zig fmt --check src inference/src` once these paths exist.

## Local serving and deferred work

A later local server can wrap the inference library and expose an
OpenAI-compatible chat-completions protocol with SSE. Model prompt/tool-call
encoding belongs with model profiles and serving; tool execution and permissions
remain the responsibility of the consuming agent, not the engine. The evaluation
CLI must not acquire filesystem-editing or shell-execution tools; tool execution
lives in the agent ([agent-spec.md](agent-spec.md)).

Deferred: HTTP serving, vision/video, persistent prefix caches, concurrent
request batching, additional production architectures, additional GPU
backends, training, and model conversion/quantization tooling. These
extensions should use the defined seams as concrete requirements emerge.
Speculative decoding left this list on 2026-09-19; its requirements follow.

## Speculative decoding

Accepted 2026-09-17 (configuration) and 2026-09-19 (the contracts); the
units are planned in [../TODO.md](../TODO.md) and the design detail, once
implemented, is recorded in
[reference/speculative-decoding.md](reference/speculative-decoding.md).

A draft source proposes tokens; the main model verifies them in one batched
forward and commits only the accepted prefix. The draft source is
model-specific and chosen by the adapter (an MTP head, a DFlash drafter);
the verification and recovery protocol is shared. Recovery never rewinds
recurrent state by truncating an attention position: recurrent state is
checkpointed before a batch and restored, then replayed over the accepted
prefix, and GPU work completes before host-visible state is restored.
Greedy acceptance follows when sampling is off, sampled acceptance with
the rejection correction that preserves the target distribution when it is
on; identical seeded token streams against ordinary decoding are not
required.

Three settings, because they answer three different questions:

- **The file**: `models.<name>.mtp` in `~/.nuclis/nuclis.json`, one draft
  companion per registry entry (`config init` fills it from the
  catalogue). Resolved and verified at load, never an implicit download; a
  missing or mismatched file is a typed load error, not a silent fallback.
  Loading the drafter is a load-time decision because its weights and the
  checkpoint scratch of the recovery contract belong to the memory plan.
  The `mtp` role names the draft source whatever its mechanism.
- **The switch**: a generation setting, `generation.speculative` in the
  configuration and `--speculative on|off` on `generate`, `agent`, and
  `bench`, layered like `think` and sampling (defaults, then the entry,
  then the flag). Per command rather than per load: it changes nothing in
  the model's state layout, and `bench` must measure the same loaded model
  both ways in one process, which is how a speedup claim is made. The
  default is on only for a family whose measured acceptance rate pays;
  the catalogue entry carries that verdict, not the user. Off is the
  ordinary loop, bit-for-bit: every speculative path (the drafter, the
  checkpoint copy, recovery, per-row state writes) is gated behind the
  switch and must not run, or change an ordinary dispatch, when it is off.
- **The draft length** (positions proposed per step): a second generation
  setting with a per-family default from the same measurement, capped by
  a host constant. Its best value depends on the prompt mix, so it sits
  beside the switch, not in the load plan.

Not exposed: the acceptance rule (greedy or sampled follows from whether
sampling is on) and the recovery scheme (an internal correctness
contract). Engine seam: load options gain an optional draft-source path
and the session its checkpoint scratch; the generation loop is what asks
the drafter, so a drafter loaded but switched off costs memory only, as
`think` already works for reasoning. Speculation ships enabled per family
only where its measured acceptance rate pays for verification; negative
results are recorded.

Measured (2026-09-20, [reference/bench.md § Speculative decoding record](reference/bench.md#speculative-decoding-record-engn-12-2026-09-20)):
on Qwen3.8-27B with its embedded draft head, greedy speculation decodes at
0.56–0.67× the ordinary rate on the 512-token corpus prompt and 0.85–0.88×
on the code prompt at draft lengths 4 and 7, 0.54× at 4K; with the
instruct profile's sampling 0.63–0.67× on prose and 1.04× on code (the
baseline there pays the penalty readback). Acceptance is 1.2–3.0 drafts
per batch; a verify batch costs 2.4–2.6 ordinary steps at 512 and 3.6 at
4K, recovery up to 2.9 steps on rejection, and the speculative prefill
2.9–3.3× the ordinary one. The Qwen entry therefore ships with the switch
off and `draft_length` 4; the performance units planned from these costs
are in `TODO.md`, and ENGN-17 re-measures and sets the defaults.

## Remaining discussion

- The 32K acceptance context is measured on the reference workload
  ([reference/bench.md § Acceptance runs](reference/bench.md#acceptance-runs));
  confirming it against normal coding workloads (real prompts, the agent)
  remains.
- Set speed and quality thresholds after measuring the pinned baseline.
- Finalize CLI flag names, minimum macOS/SDK requirements, and numerical tolerances
  during the corresponding implementation milestones.
