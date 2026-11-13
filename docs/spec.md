<!-- kind: reference -->
# nuclis — technical specification

The authoritative statement of what nuclis is, what it must do, and the
decisions that bound how it does it: requirements and accepted decisions,
not status. History is in the
[merged pull requests](https://github.com/tildaslashalef/nuclis/pulls?q=is%3Apr+is%3Amerged) (before v0.6.0, the [worklog](https://github.com/tildaslashalef/nuclis/blob/v0.6.0/docs/worklog.md)); the build and
measurement in [development.md](development.md); the structure in
[architecture.md](architecture.md); each component in the documents
[README.md](README.md) lists. Numbers here are acceptance criteria; lists
the code holds are generated from it. *Must* is a requirement, *should* a
strong default a unit may depart from with a recorded reason, *may* a
permission; changing a decision is a change to this document.

## 1. Purpose and scope

nuclis runs language models locally with its own compute kernels, on Apple
silicon, for one user doing coding work. It is **an inference library**
(`inference/`: loading, validation, tokenization, the model schedules, the
CPU reference, the Metal backend, sampling, speculative decoding); **an
evaluation CLI** (`src/`, § 6), which also answers typed questions, embeds,
speaks, transcribes, and serves every kind of model over HTTP; and **an
interactive agent** (`nuclis agent`, § 7) with a bounded tool layer for
small coding tasks. The target is an Apple M4 Pro with 48 GB of unified
memory; Linux is a CPU-only build target that serves the models the CPU
reference runs fast enough ([development.md § Linux](development.md#linux)).
Language models come in three families through one seam, the adapter
table (§ 5.7); the decision, embedding, and speech models have their own
paths (§ 5.9–§ 5.11). Everything else is deferred (§ 10).

## 2. Definitions

| Term | Meaning |
| --- | --- |
| artifact | one GGUF file, identified by its SHA-256; or a safetensors set (weights and their configuration and tokenizer files), each file identified by its digest |
| family | a language-model architecture the engine has an adapter for in `inference/src/models/root.zig`'s table (`qwen35`, `gemma4`, `muse_glimmer`) |
| adapter | the family's binding, CPU runtime, and Metal plan |
| profile | the prompt contract of a chat template: rendering, stop set, reasoning markers, tool grammar; selected by the template's digest |
| session | the mutable state of one conversation: attention cache rows and recurrent state, in one block |
| prefill, decode | consuming prompt tokens, in chunks on the GPU; producing one token per step from the previous one |
| draft source, verify batch | a model-specific component that proposes tokens for speculative decoding; one forward over the last committed token and `k` drafts, keeping every row's logits |
| turn, step | in the agent: one user message and everything the model does until it answers; one completion request plus the execution of its tool calls |
| gate, workload | a model-specific check in `gates.json`; a benchmark in `workloads.json` |
| decision model | a model with a typed-decision head that scores a question's options about a state in one forward, with no decoding: Laya (an encoder, one pass per question) or clef-flash (a Qwen backbone, every question in one pass) |
| state, question | what a decision model reads (text, or JSON rendered as text) and what it answers: `choice` (one of named options), `score` (a level of an ordered scale), or `noul` (the probability a statement holds) |

## 3. Decisions

| Decision | Reason |
| --- | --- |
| Zig owns loading, execution planning, tokenization, sampling, the CLI, and the agent; Objective-C exposes a C-compatible Metal interface behind opaque handles; Metal Shading Language implements kernels. | One language for semantics, one thin bridge for the platform, no framework between them. |
| The Metal backend is the production backend; the CPU path is a reference, slow by design, never optimized. The one exception is the decision model's encoder, whose F32 CPU forward is threaded and vectorized (§5.9). | A fast CPU backend is not a product need for text generation; an obviously correct oracle is. A decision model is small enough, and useful enough without a GPU, that its CPU path is a product; its oracle is the reference package's F32 run. |
| Decision checkpoints have their own catalogue table and registry kind; text commands refuse them by name. | They are directories of safetensors, run by another path; nothing that resolves a text model may pick one. |
| Embedding models have their own catalogue table and registry kind (`embedding`); text and decision commands refuse them by name, and `nuclis embed` refuses theirs. | An embedding model decodes nothing and answers no question; its vectors compare only with vectors of the same file and width. |
| Weights are memory-mapped, stay quantized, and are never requantized or expanded. | The 27B model fits only at about four bits; the file's arithmetic is what the checkpoint was validated for. |
| One active session per process; weights are immutable and separate from session state. | Multiple sessions later must not copy weights. |
| Session state is opaque and never rewound by truncating an attention position alone; rollback is snapshot/restore, and inside a speculative batch checkpoint/rewind. | Recurrent layers keep no history to truncate back to. |
| The attention cache is F16 on the GPU by default, Q8_0 on request (`--kv q8_0`); the CPU reference keeps F32. | Half the memory and half the bytes per step, at a rounding bounded per mode by the gates; Q8_0 nearly halves them again for long contexts, at a larger rounding. |
| Adapters and profiles register through tables; the enum, the executor union, and the known-architecture list derive from them. | A family costs its own files and one line, and two lists cannot disagree. |
| A profile is selected by the chat template's SHA-256, never by architecture; the profile owns the stop set, the reasoning markers, and tool rendering and decoding in both directions. | Two files of one family can ship different templates; nothing outside `profiles/` may name a template's tokens. |
| Every adapter writes its schedule twice, CPU and Metal; no shared op list. | Three families showed the variation is in kernel parameters and instantiations, not operation order. |
| nuclis is its own reference: new work is validated against its already-validated paths, not an external implementation. External code is never a source to copy and is read only as the definition of a format that has no other. | The engine is implemented from the format and the mathematics. Its CPU reference, its Metal plans, and its files already matched to llama.cpp's traces are a mature base (decided 2026-10-14); the fixtures and traces pinned from llama.cpp before then stay as committed data. |
| A genuinely new architecture, which no validated nuclis path computes, is validated by data: its permanent gates measure against human ground truth (word error rate on pinned public clips with their transcripts; synthesized speech recognized back by an already-validated recognizer). Its bring-up adds traces from the authors' published code, run once and pinned as fixtures; from then on its CPU reference is the oracle for its Metal plan, as for every family. | Data outlives any implementation and is what the model was trained to match; a trace taken once catches what a word-error bound lets through (a wrong norm epsilon, a misplaced scale) without making the external code a standing oracle (decided 2026-10-27). |
| Speculative decoding is exact by construction, switched per family by its measured verdict, and off changes nothing in the ordinary path. | A speedup claim needs a true baseline in the same process; the ordinary loop must not pay for a feature it does not use. |
| The catalogue pins repository, file, revision, digest, profile, and companions per entry. | A supported model is a specific file, not a name. |
| Configuration is one JSON file whose schema is a Zig struct; precedence is defaults < profile < file < registry entry < flags. | The file, the loader, the validator, and the printer cannot disagree about which keys exist. |
| `nuclis agent` renders inline with the terminal's native scrollback; no alternate screen. | The terminal's scrolling, search, selection, and copy work natively, and the transcript survives exit. |
| Sessions are append-only JSONL files; nuclis never deletes one. | A crash story of one truncated line, no database, and the user owns retention. |
| The agent has six fixed tools, and up to three web tools when the configuration opts in; no permission system, no extension mechanism, no subagents. | Supervision is visibility plus the workspace boundary; the agent is a playground for the engine, not a platform. The web tools reach a service the user runs, so they are the user's choice, never a default. |
| Native tool-call conversion belongs to the profile in `inference/`; execution belongs to the agent. | The loop stays format-agnostic; the model's wire syntax is model knowledge. |
| The evaluation CLI never acquires filesystem-editing or shell tools. | Tool execution is the agent's responsibility alone. |
| The terminal surface (`src/tui/`) imports nothing from `inference`. | Its golden tests need no model, no GPU, and no TTY. |
| Every measurement names its hardware, build, artifact, context, and method; an estimate is never presented as one. | Performance claims are the record's, and only `bench` produces them. |
| Nothing lands silently: every change is a unit, one commit whose body is its record, reached through its milestone's pull request. | The commits and pull requests are the durable history; this document carries none. |
## 4. Supported models and artifacts

The index (`src/catalog.json`, read by `src/catalog.zig`) is the list of
supported artifacts. An entry pins the Hub repository, revision, and every
file's SHA-256 and size; its kind, profile, and lifecycle; and its
companions by role, each draft with the speculative verdict a measurement
set. A `current` entry is gated and written by `config init`; a `legacy`
one pulls, verifies, and runs by name, with no gates. The index:

<!-- gen:catalogue -->
| Name | Kind | Encoding | Main file | Status |
| --- | --- | --- | --- | --- |
| `qwen3.8-27b` | generation | UD-Q4_K_M | 16.56 GB | current |
| `gemma-4-12b-qat` | generation | Q4_0 (QAT) | 6.72 GB | legacy |
| `gemma-4-26b-a4b` | generation | Q4_0 (QAT) | 14.25 GB | current |
| `gemma-4-e4b-qat` | generation | Q4_0 (QAT) | 4.22 GB | current |
| `muse-glimmer-30b` | generation | UD-Q4_K_XL | 15.88 GB | legacy |
| `laya` | decision | F16 | 842.6 MB | current |
| `laya-multilingual` | decision | F16 | 643.8 MB | current |
| `clef-flash` | decision | Q6_K | 243.5 MB | current |
| `embeddinggemma-2` | embedding | Q8_0 | 309.9 MB | current |
| `qwen3-tts-1.7b-customvoice` | speech | BF16 | 3.83 GB | current |
| `qwen3-tts-0.6b-customvoice` | speech | BF16 | 1.81 GB | current |
| `qwen3-tts-1.7b-base` | speech | BF16 | 3.86 GB | current |
| `qwen3-tts-0.6b-base` | speech | BF16 | 1.83 GB | current |
| `qwen3-tts-1.7b-voicedesign` | speech | BF16 | 3.83 GB | current |
| `drama-3` | speech | Q4_K | 20.79 GB | current |
| `nemotron-asr-0.6b` | transcription | F32 | 2.55 GB | current |
<!-- /gen -->

A command refuses another kind's model by name. What each entry is:
[models/catalogue.md](models/catalogue.md) and the family cards. A file
outside the index whose architecture has an adapter is *runnable*: `config
init` and `config sync` register it, its profile chosen by template digest.

- The loader **must** validate the architecture, dimensions, metadata,
  tensor names, shapes, offsets, and actual encodings against the adapter,
  reject an unsupported combination with a typed error naming the first
  offending tensor, and never dispatch from a display name or requantize
  ([engine/gguf.md](engine/gguf.md), [engine/safetensors.md](engine/safetensors.md)).
- `model pull` **must** verify the digest before publishing a file
  atomically, beside a provenance sidecar. `model inspect` **must** judge a
  file from its directory alone: storable, executable, bindable,
  supported. `model rm` **must** delete only under the models directory and
  keep a file another entry uses.
- A companion is loaded only by the unit that consumes it; a missing or
  mismatched one is a typed load error, never an implicit download.
- The user root is `~/.nuclis` (`NUCLIS_HOME`, an absolute path, overrides
  it), models under `models/<owner>/<repo>/<file>`.

## 5. Engine requirements

### 5.1 Interface

The library **must** let a caller open a model, create a session, prefill,
step, verify a batch, stream, reset, and release, with typed results; the
caller renders. Allocators and `Io` are explicit; every returned slice says
whether it is owned or borrowed. Adapters, profiles, kernels, and stop
tokens are resolved at load, never per token.

### 5.2 Loading and validation

Parsing **must** be bounds-checked against the file size and the parser's
limits, with overflow-checked sizes, before any weight byte is read; a bad
file is a typed error, never a panic. Every encoding **must** have a CPU
decoder pinned by tests on hand-built blocks and agreeing with the same
model in an already-validated encoding
([models/README.md § 4](models/README.md#4-validate-against-nuclis-itself)),
and a bit-identical GPU decoder ([engine/quantization.md](engine/quantization.md)).

### 5.3 Memory model

Immutable mapped weights, mutable session state in one page-aligned block,
and per-step scratch. GPU-visible memory **must** outlive submitted work,
and the CPU **must not** touch session memory while a command buffer that
writes it is in flight. The adapter declares the session's layouts; a
draft source's cache is one more layout.

### 5.4 Session contract

A step is `ready → updating → ready`; a failed step leaves the session
`failed` until `reset()`. `snapshot`/`restore` copy the used extent behind
a digest of layouts and capacity, a mismatch touching nothing. Attention
rewinds by position, recurrent state from a checkpoint; no operation
**may** rewind a hybrid session by position alone. Sessions share no
mutable state ([engine/session.md](engine/session.md)).

### 5.5 Execution

- Chunked prefill **must** agree with the stepped path within the family's
  tolerance; decode records one token into one command buffer, with no
  per-layer observer on the production path.
- The Metal bridge has matched create/destroy pairs, cleans up partial
  initialization, and documents whether each operation records, submits,
  or waits; Zig owns the plan. An optional tick runs on the calling thread
  during a long wait and **must not** touch the backend or the model
  ([engine/metal-backend.md](engine/metal-backend.md)).
- Sampling is a policy over a caller-owned history (penalties,
  temperature, top-k, min-p, top-p, the draw), with per-mode defaults
  owned by the profile; a GPU path **must** produce the reference's token
  for the same seed or defer to it ([engine/sampling.md](engine/sampling.md)).

### 5.6 Speculative decoding

A draft source proposes tokens, the main model verifies them in one batch,
and the session is recovered to the accepted prefix
([engine/speculative-decoding.md](engine/speculative-decoding.md)). The
draft contract (`propose`, `commit`, `reset`, `bytes`) is
family-independent; recurrent state is checkpointed before the batch, with
GPU work complete before host-visible state is touched; every emitted token
is a target draw (greedy acceptance when sampling is off). The switch
defaults to the entry's measured verdict. Off **must** be the ordinary loop
bit for bit, and `bench` **must** open no drafter when the switch is off.

### 5.7 Extension rules

An additional family **must** cost a new adapter, its profile, its tests
and fixtures, and any genuinely new mathematics (a `cpu.*` function with
fixtures, a kernel with a `metal-check` entry), and no edit to the GGUF
parser, sampling, the session, the loop, or the Metal lifecycle. A new
encoding costs a CPU decoder arm, a bit-identical GPU decoder, specialized
kernels where wanted, and the adapter's claim ([models](models/README.md)).

### 5.8 Configuration

One file, `~/.nuclis/nuclis.json`, whose schema is the struct `Config` in
`src/config.zig`:

<!-- gen:config-keys -->
```text
engine.*                model, backend, ctx_size, kv_precision
generation.*            max_tokens, think, speculative, draft_length,
                        image_max_tokens, sampling.{temperature, top_k, top_p,
                        min_p, presence_penalty, repetition_penalty}
agent.*                 think, fold_thinking, theme, instructions,
                        thinking_budget, web
decide.*                model
embed.*                 model
speech.*                model, voice
transcribe.*            model
serve.*                 host, port, log, request_log, timeout, memory_bytes,
                        web, pin, transcription_bytes
web.*                   url
cache.*                 memory_bytes, disk_bytes
models.<name>.*         path, or repo + file + revision; mmproj, draft (a
                        file; mtp is its old name), profile, ctx_size,
                        generation.*, agent.* for that model only, and
                        generation.draft: an index model's draft variant
                        (qwen3.8-27b: mtp, dflash2). Entries are created by
                        init, sync, and model pull --register, never by set.
A null sampling key takes the profile's value. agent.instructions is
auto (AGENTS.md then CLAUDE.md), off, or a path.
```
<!-- /gen -->

A `models` entry locates a file, names its companions, may force a
profile, and overrides keys for that model; its `kind` says which command
opens it. Precedence is defaults < profile < file < registry entry <
flags; an unknown key or an out-of-range value **must** be a typed error
naming the key, and `bench` ignores the file's sampling and budget
([guide/configuration.md](guide/configuration.md#configuration-file)).

### 5.9 The decision path

`inference.decide.Decider` answers typed questions about states for two
families picked by the checkpoint's files: Laya, an encoder, and
clef-flash, the qwen35 backbone and Qwen3-VL projector with a joint head
([models/laya.md](models/laya.md), [models/clef-flash.md](models/clef-flash.md)).

- The input contract, calibration, and answer fields **must** be the
  reference package's (`laya` 0.3.20; Cloudflare's `joint_schema_model.py`
  at the pinned revision, its sequence equal to the reference's), and
  Laya's tokenizer the reference library's two shapes.
- Laya **must** match the reference's F32 run within 1e-5 relative per
  stored stage and logits within 1e-4 of max(1, |logit|) on the CPU and on
  Metal; clef-flash's head within 1e-4, the whole model within 1e-3 on
  Metal and 2e-3 on the CPU.
- Host limits: 64 states, 32 questions, 255 options, 1 MiB per state, 8
  images of 16 MiB; calls batched together **must** each get the answers
  they get alone.

### 5.10 The embedding path

`inference.embed.Embedder` turns an input's ordered parts (text, images,
clips) into one unit vector, in one bidirectional pass of at most 8,192
tokens, with no session ([models/embeddinggemma.md](models/embeddinggemma.md)).

- An over-long input or a clip over 30 s **must** be refused unless the
  request truncates, which is reported; text is literal, no task prefix is
  implicit, and only the trained widths (768, 512, 256, 128) are served.
- Every response names its **space**, `<checkpoint>@<sha256[0:12]>/<width>`.
- The BF16 file **must** match Google's float32 vectors at cosine
  ≥ 0.9999999 for text, images, and audio, and the Q8_0 file its pinned
  vectors at ≥ 0.99999, on the CPU and on Metal; an input **must** get the
  same vector whatever shares its pass.
- Host limits: 2,048 inputs, 1 MiB of text an input, 32 MiB an image,
  64 MiB a clip; media are never fetched, and the API refuses token arrays.

### 5.11 The speech path

`inference.models.qwen3_tts_runtime` speaks and
`inference.audio.nemotron_asr` hears, on the CPU or a Metal plan, validated
by data (§ 3): word error rates against human transcripts and a round trip
through the recognizer ([models/qwen3-tts.md](models/qwen3-tts.md),
[models/drama3.md](models/drama3.md), [models/nemotron-asr.md](models/nemotron-asr.md)).
A voice **must** decode causally, its samples independent of chunking and
of other work for the same text, voice, and seed; an instruction a model
cannot take is refused. A clip streamed in any pieces **must** give the
same text as the whole clip, Metal the same as the CPU, in memory that does
not grow with the samples.

## 6. Command-line interface

The commands, and their usage as their help pages print it:

<!-- gen:cli-commands -->
| Command | What it does |
| --- | --- |
| `agent` | the interactive chat; -p <text> runs one turn |
| `generate` | one completion from a prompt |
| `bench` | repeated prefill and decode measurements |
| `tokenize` | the prompt's token ids and byte offsets, no model run |
| `eval` | perplexity of a text file, teacher-forced |
| `inspect` | an artifact's identity, dimensions, tensor encodings |
| `validate` | whether a file binds to its architecture's adapter |
| `model` | pull, list, judge, and remove Hugging Face Hub artifacts |
| `cache` | list or clear the agent's saved model states |
| `decide` | typed questions about text or JSON, answered by Laya |
| `embed` | text, images, audio into vectors, by EmbeddingGemma 2 |
| `speak, voice, transcribe` | speech by Qwen3-TTS, saved voices, Nemotron ASR |
| `serve` | the nuclis API over HTTP: decisions, models kept open |
| `config` | write, show, or set a key of ~/.nuclis/nuclis.json |
| `completion` | the shell completion script for fish, bash, or zsh |
<!-- /gen -->

<!-- gen:cli-synopsis -->
```text
nuclis agent [options] [--resume [<id>]]
nuclis agent -p <text> [--json] [--session <path>]
nuclis agent --print --prompt-file <path> [--json] [--session <path>]
nuclis agent ls [--json]
nuclis agent rm <id>
nuclis agent export <id> [path]
nuclis generate --prompt <text> [options]
nuclis generate --prompt-file <path> [options]
nuclis generate --prompt-tokens <json> [options]
nuclis bench --prompt <text> [options]
nuclis bench --prompt-file <path> [options]
nuclis bench --prompt-tokens <json> [options]
nuclis tokenize --prompt <text> [options]
nuclis tokenize --prompt-file <path> [options]
nuclis eval --file <path> [options]
nuclis inspect [--model <name|path>] [--tensor <name>] [--json]
nuclis validate [--model <name|path>] [--json]
nuclis model ls [-l] [--available [--all]] [--json]
nuclis model pull <name> [--with mmproj,draft[=<v>] | --all [--variants]]
[--force] [--json]
nuclis model pull <owner/repo> [--file <name>] [--revision <rev>] [--role <r>]
[--register <name> [--profile <p>]] [--force] [--json]
nuclis model inspect <name> [--json]
nuclis model inspect <owner/repo> --file <name> [--revision <rev>] [--json]
nuclis model adopt <path> [--force] [--json]
nuclis model rm <name> [--with mmproj,draft | --all] [--json]
nuclis cache [ls] [--json]
nuclis cache clear
nuclis decide --request <file|-> [options]
nuclis decide --questions <file> <states> [options]
nuclis decide <inline questions> <states> [options]
<states>: --state <text> and --state-file <path>, repeatable
<inline questions>: --choice <text> --option <key[=desc]>…,
--score <text> --level <text>…, --noul <text>; each [--id <name>]
nuclis embed <text>… [options]
nuclis embed --image <file>… --audio <file>… [options]
nuclis embed --input-file <file|-> [options]
each argument is one input: quote a sentence
nuclis speak <text>… -o <file.wav> [options]
the words are joined by spaces: quote a sentence
nuclis voice add <name> <audio> [--text <transcript>] [--json]
nuclis voice list [--json]
nuclis voice rm <name> [--json]
nuclis transcribe <file> [options]
nuclis serve [--host <ip>] [--port <n>] [--model <name>]… [--pin <name>]…
[--memory <GiB>] [--backend cpu|metal] [--timeout <s>]
[--web-url <url> | --no-web]
nuclis config init [--dry-run] [--json]
nuclis config sync [--prune] [--dry-run] [--json]
nuclis config show [--json]
nuclis config set <key> <value>
nuclis completion fish|bash|zsh
```
<!-- /gen -->

What each command **must** do beyond its summary; `decide`, `embed`, `speak`,
`voice`, and `transcribe` are § 5.9–§ 5.11, and `agent` is § 7:

| Command | Contract |
| --- | --- |
| `generate` | one completion: raw prompt or one rendered turn, streamed, seeded, the profile's per-mode sampling defaults with per-flag overrides, a structured stop reason, timings |
| `bench` | repeated cold and warm measurements with separate load, prefill, and decode timings; a prompt may be a token array so an exact input is reproduced; the speculative pair on one loaded model in one process ([benchmarks](benchmarks/README.md)) |
| `tokenize` | the prompt as `generate` would feed it, its ids and byte offsets, from the artifact's header only |
| `model` | pull with digest verification and sidecars; list what is installed, grouped by kind; adopt a file placed by hand; judge a file at the four levels of § 4 |
| `serve` | the nuclis API over HTTP/1.1, loopback by default (another address warns: no authentication): Jev's call, OpenAI's chat completions, embeddings, speech, custom voices, and transcription, live transcription, the model list, and health ([guide/api.md](guide/api.md)); models opened by name, never by path, within one memory budget, the least recently used closed to make room and a model in use never; one GPU pass at a time; every limit's refusal a typed error body |
| `config` | `init` writes the file from what is installed; `sync` adds what was installed since and, with `--prune`, drops entries whose files are gone; `show` prints effective values with their source layer; `set` sets one key |
| `completion` | a thin script per shell that runs the hidden `nuclis __complete <words…>`, which answers from one command table held to the parser and the help pages by tests |

Output rules: text to stdout, diagnostics to stderr; `--json` is a
versioned result type shared with the human-readable form; styling only on
a terminal that advertises colour, never under `NO_COLOR` or a pipe; every
stop a named reason, and no prompt silently truncated; streamed text
valid UTF-8; every `--help` page
self-contained.

## 7. The agent

`nuclis agent` is the interactive surface of the engine and the project's
playground: every feature a user can feel is exercised here first; the
evaluation CLI stays separate. What it does, key by key, is
[app/agent-surface.md](app/agent-surface.md).

### 7.1 Surface

It **must** require a TTY outside print mode, render inline with the
terminal's native scrollback, write each completed turn once above a live
region, and repaint only that region. Colour degrades to plain attributes,
`NO_COLOR` wins, and a theme supplies a palette only, never layout.

### 7.2 Editor

The editor stays live while a turn runs (Enter steers it, Alt-Enter
queues). Pastes and dropped files become chips encoded in the background;
what the model cannot read is refused by name, never cut or ignored. A
line is a command only when it starts with `/` and an ASCII word; keys are
accelerators for commands, and a destructive action is confirmed inside
its picker.

### 7.3 Transcript and status

Typed events become blocks, each handed to the screen exactly once;
model-supplied control bytes are stripped and no row is wider than the
terminal. A rate that was not measured prints as absent.

### 7.4 Sessions and storage

Sessions are append-only JSONL files under
`~/.nuclis/agent/sessions/<cwd-slug>/` that store what the model saw and
produced; a truncated last line is dropped, any other bad line is a typed
error naming it, and a failed write is a notice, never a lost turn.
Resuming restores a saved state from the token cache when one matches.
nuclis deletes a session only when asked (`agent rm`).

### 7.5 The loop

A turn is a loop of at most 16 steps: render the system block, the history,
and the tool definitions through the profile; stream the completion;
execute each decoded call in order, validated against the registry; send
the results back, or finish when the response holds no calls. A turn
stops on cancellation, an unrecoverable failure, or the step budget, with
its reason. Failures are results: an unknown tool, invalid arguments, a
timeout, a truncated call, or a tool failure returns a typed result; only
an inference-transport failure ends the turn. The loop prefills only each
turn's increment; a result over an eighth of the window is cut, and older
results, then earlier turns, leave the model's view before a turn fails.
Every sentence of the system block is judged on the task list
([development.md § The agent's task list](development.md#the-agents-task-list)).

### 7.6 Tools

Six fixed tools (`read_file`, `write_file`, `edit_file`, `glob`, `grep`,
`bash`) and, only when the configuration opts in, three web tools; their
contracts are [app/agent-surface.md § Tools](app/agent-surface.md#tools).
Every path is resolved under the workspace, the process's working
directory, and a symlink escape is refused. Limits are host constants,
never model-supplied; truncation is always marked; exceeding a limit is a
typed result. Without `agent.web` the model sees the six tools and the
same system prompt, byte for byte. The host constants:

<!-- gen:limits -->
| Limit | Value | Declared in |
| --- | --- | --- |
| context ceiling (tokens) | 32,768 | inference/src/runtime/session.zig `max_capacity` |
| output budget (tokens) | 16,384 | src/config.zig `max_output_tokens` |
| draft length | 15 | inference/src/engine.zig `max_draft_length` |
| prefill chunk (tokens) | 256 | inference/src/engine.zig `prefill_chunk` |
| config file (bytes) | 65,536 | src/config.zig `max_file_bytes` |
| agent steps per turn | 16 | src/agent/loop.zig `budget_default` |
| agent turns per session | 256 | src/agent/loop.zig `max_turns` |
| agent instructions file (bytes) | 8,192 | src/agent/system_prompt.zig `max_instructions_bytes` |
| read_file lines per call | 2,000 | src/agent/tools/read_file.zig `max_lines` |
| read_file default lines | 120 | src/agent/tools/read_file.zig `default_lines` |
| read_file bytes | 1,048,576 | src/agent/tools/read_file.zig `max_bytes` |
| grep matches shown | 50 | src/agent/tools/grep.zig `max_shown` |
| grep files scanned | 20,000 | src/agent/tools/grep.zig `max_files` |
| grep seconds | 10 | src/agent/tools/grep.zig `max_seconds` |
| glob results | 200 | src/agent/tools/glob.zig `max_results` |
| image tokens per image | 4,096 | inference/src/vision/root.zig `max_image_tokens` |
| server output tokens | 16,384 | src/api/chat/wire.zig `max_output_tokens` |
| server audio clip (bytes) | 16,777,216 | src/api/chat/wire.zig `max_audio_bytes` |
| server web rounds | 6 | src/api/chat/service.zig `max_web_rounds` |
| server speech input (bytes) | 16,384 | src/api/speech/wire.zig `max_input_bytes` |
| server live transcriptions | 16 | src/api/transcription/stream.zig `max_sessions` |
| decision questions per call | 32 | inference/src/decide.zig `max_questions` |
| decision states per call | 64 | inference/src/decide.zig `max_states` |
<!-- /gen -->

### 7.7 Print mode

`nuclis agent -p "<prompt>"` runs one turn without a TTY: the answer alone,
or with `--json` every event as a line; nothing is recorded unless
`--session <path>` names a file.

### 7.8 Not in scope

A permission system, MCP or extensions, subagents, a lookup tool beyond
the opt-in web tools, and a second product surface.

## 8. Performance requirements

- The primary workload is one user doing coding work: time to first token
  after a substantial code prompt, and steady decode. Every length of the
  acceptance workload, to 32,768 tokens on the 48 GB target, **must**
  complete on its budget with no swap growth
  ([benchmarks § Acceptance runs](benchmarks/README.md#acceptance-runs)).
- Claims are made against nuclis's own previous record on the same machine
  and method, at equal context, cache precision, sampling, and output
  budget, naming chip, OS, compiler, revision, artifact hash, power mode,
  warm-up, and memory; prefill and decode separately.
- A kernel or loop change ships when it is at least as fast as its control
  and the gates hold; a negative result is recorded and closed.
  Speculation ships on only when it beats the ordinary rate on the
  family's acceptance workload.

## 9. Verification

- **Default tests** (`make test`) need no network, credentials, GPU, or
  model, run under a leak-checking allocator, and **must** cover malformed
  GGUF data, overflow, block decoding, tokenization, prompt rendering, the
  tools in temporary workspaces, the loop against a stubbed completion, and
  the surface's escape streams, error paths included.
- **Fixtures** are a reference's outputs on small inputs, with provenance
  ([models/README.md § 4](models/README.md#4-validate-against-nuclis-itself));
  **GPU checks** (`make test-metal`) compare every kernel with the CPU
  reference, and lifecycle, cancellation, and cleanup separately.
- **Gates** (`gates.json`, selected by changed paths through `make
  verify-auto`, every tier strict before a release) hold the traces, the
  generation checks, the bounds of § 5.9–§ 5.11, and each language model's
  perplexity ([guide/eval.md](guide/eval.md)), every threshold per
  numerical mode ([development.md § Gates](development.md#gates)).
- **Done** is build, tests with error paths, formatting, the happy path on
  the fresh binary, the gates its paths select, its documents, and a report
  of what changed, what was verified, and what remains.

## 10. Deferred work and non-goals

Deferred until a concrete requirement: constrained decoding (structured
output, forced tool choice), additional GPU backends, and video input to
the embedding model. Outside nuclis: PDF attachments; OpenAI's Responses
API (the clients `nuclis serve` serves speak Chat Completions; decided
2026-10-08); tool execution in the server; a docs lookup tool for the
agent; and a personal index, which belongs to the application that calls
`POST /v1/embeddings` (decided 2026-10-09). Non-goals: training, model
conversion or quantization tooling, universal GGUF support, a general
tensor compiler, and matching any external product's feature list.

## 11. Open questions

- Whether tool steps want different sampling defaults for reliability.
- Whether `bash` wants an allow-list or an approval step.
- Whether the sliding-window caches should be ring buffers.

