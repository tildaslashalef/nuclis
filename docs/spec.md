# nuclis — technical specification

This is the authoritative statement of what nuclis is, what it must do, and
the decisions that bound how it does it. It states requirements and
accepted decisions, not status: what the tree does today is in the
[engineering log](engineering-log.md), what it does next in
[TODO.md](../TODO.md), how it is built and measured in
[development.md](development.md), how it is structured in
[architecture.md](architecture.md), and the facts behind each component in
[reference/](reference/). Numbers appear here only as acceptance criteria;
the measurements that meet them live in the record they cite.

*Must* is a requirement, *should* a strong default that a unit may depart
from with a recorded reason, *may* a permission. A decision is a choice
this document has made; changing one is a change to this document.

1. [Purpose and scope](#1-purpose-and-scope)
2. [Definitions](#2-definitions)
3. [Decisions](#3-decisions)
4. [Supported models and artifacts](#4-supported-models-and-artifacts)
5. [Engine requirements](#5-engine-requirements)
6. [Command-line interface](#6-command-line-interface)
7. [The agent](#7-the-agent)
8. [Performance requirements](#8-performance-requirements)
9. [Verification](#9-verification)
10. [Deferred work and non-goals](#10-deferred-work-and-non-goals)
11. [Open questions](#11-open-questions)

## 1. Purpose and scope

nuclis runs language models locally with its own compute kernels, on Apple
silicon, for one user doing coding work. It is three things in one
repository:

- **an inference library** (`inference/`): loading, validation,
  tokenization, the model schedules, the CPU reference, the Metal backend,
  sampling, and speculative decoding;
- **an evaluation CLI** (`src/`): `generate`, `bench`, `tokenize`,
  `inspect`, `validate`, `config`, and `model`;
- **an interactive agent** (`nuclis agent`): a terminal surface over the
  same engine with a bounded tool layer for small coding tasks in the
  working directory.

The target hardware is an Apple M4 Pro with 48 GB of unified memory. The
first model is Qwen3.8-27B; the engine supports four families through one
extension seam (§4, §5.7). Everything outside these three surfaces is
deferred (§10).

## 2. Definitions

| Term | Meaning |
| --- | --- |
| artifact | one GGUF file, identified by its SHA-256 |
| family | an architecture the engine has an adapter for (`qwen35`, `gemma4`, `muse_glimmer`) |
| adapter | the family's binding, CPU runtime, and Metal plan |
| profile | the prompt contract of a chat template: rendering, stop set, reasoning markers, tool grammar; selected by the template's digest |
| session | the mutable state of one conversation: attention cache rows and recurrent state, in one block |
| prefill | consuming prompt tokens, in chunks on the GPU |
| decode | producing one token per step from the previous one |
| draft source | a model-specific component that proposes tokens for speculative decoding |
| verify batch | one forward over the last committed token and `k` drafts, keeping every row's logits |
| turn | in the agent: one user message and everything the model does until it answers |
| step | in the agent: one completion request plus the execution of the tool calls it contains |
| gate | a model-specific check in `gates.json`; workload: a benchmark in `workloads.json` |

## 3. Decisions

| Decision | Reason |
| --- | --- |
| Zig owns loading, execution planning, tokenization, sampling, the CLI, and the agent; Objective-C exposes a C-compatible Metal interface behind opaque handles; Metal Shading Language implements kernels. | One language for semantics, one thin bridge for the platform, no framework between them. |
| The Metal backend is the production backend; the CPU path is a reference, slow by design, never optimized. | A fast CPU backend is not a product need; an obviously correct oracle is. |
| Weights are memory-mapped, stay quantized, and are never requantized or expanded. | The 27B model fits only at about four bits; the file's arithmetic is what the checkpoint was validated for. |
| One active session per process; weights are immutable and separate from session state. | Multiple sessions later must not copy weights. |
| Session state is opaque and never rewound by truncating an attention position alone; rollback is snapshot/restore, and inside a speculative batch checkpoint/rewind. | Recurrent layers keep no history to truncate back to. |
| The attention cache is F16 on the GPU by default; the CPU reference keeps F32. | Half the memory and half the bytes per step, at a rounding bounded per mode by the gates. |
| Adapters and profiles register through tables; the enum, the executor union, and the known-architecture list derive from them. | A family costs its own files and one line, and two lists cannot disagree. |
| A profile is selected by the chat template's SHA-256, never by architecture; the profile owns the stop set, the reasoning markers, and tool rendering and decoding in both directions. | Two files of one family can ship different templates; nothing outside `profiles/` may name a template's tokens. |
| Every adapter writes its schedule twice, CPU and Metal; no shared op list. | Three families showed the variation is in kernel parameters and instantiations, not operation order. |
| External implementations are references and oracles, never sources to copy. | The engine is implemented from the format and the mathematics; equivalence is proved by fixtures and traces. |
| Speculative decoding is exact by construction, switched per family by its measured verdict, and off changes nothing in the ordinary path. | A speedup claim needs a true baseline in the same process; the ordinary loop must not pay for a feature it does not use. |
| The catalogue pins repository, file, revision, digest, profile, and companions per entry. | A supported model is a specific file, not a name. |
| Configuration is one JSON file whose schema is a Zig struct; precedence is defaults < profile < file < registry entry < flags. | The file, the loader, the validator, and the printer cannot disagree about which keys exist. |
| `nuclis agent` renders inline with the terminal's native scrollback; no alternate screen. | The terminal's scrolling, search, selection, and copy work natively, and the transcript survives exit. |
| Sessions are append-only JSONL files; nuclis never deletes one. | A crash story of one truncated line, no database, and the user owns retention. |
| The agent has six fixed tools, no permission system, no extension mechanism, no subagents. | Supervision is visibility plus the workspace boundary; the agent is a playground for the engine, not a platform. |
| Native tool-call conversion belongs to the profile in `inference/`; execution belongs to the agent. | The loop stays format-agnostic; the model's wire syntax is model knowledge. |
| The evaluation CLI never acquires filesystem-editing or shell tools. | Tool execution is the agent's responsibility alone. |
| The terminal surface (`src/tui/`) imports nothing from `inference`. | Its golden tests need no model, no GPU, and no TTY. |
| Every measurement names its hardware, build, artifact, context, and method; an estimate is never presented as one. | Performance claims are the record's, and only `bench` produces them. |
| Nothing closes silently: every change is a unit with a log entry. | The log is the durable history; this document carries none. |

## 4. Supported models and artifacts

The catalogue (`src/catalog.zig`) is the list of supported artifacts. An
entry pins the Hub repository, file, revision, and SHA-256; the profile;
the companion files by role (`mmproj`, `mtp`); and the per-entry generation
defaults that a measurement set (the speculative switch and draft length).

| Entry | Family | Artifact | Notes |
| --- | --- | --- | --- |
| `qwen3.8-27b` | `qwen35` | `unsloth/Qwen3.8-27B-GGUF` `Qwen3.8-27B-UD-Q4_K_M.gguf` | the first and default model; 64 layers, 16 attention and 48 DeltaNet; an embedded draft block |
| `gemma-4-12b`, `gemma-4-12b-qat` | `gemma4` | `unsloth/gemma-4-12b-it-GGUF`, `unsloth/gemma-4-12B-it-qat-GGUF` | sliding-window and global attention; the QAT file is Q4_0 throughout |
| `gemma-4-26b-a4b` | `gemma4` | `unsloth/gemma-4-26B-A4B-it-qat-GGUF` | the expert configuration, 8 of 128 experts per token |
| `muse-glimmer-30b` | `muse_glimmer` | `unsloth/Muse-Glimmer-30B-GGUF` | dense; windowed and global attention; a DFlash draft companion |
| `bonsai-2-27b` | `qwen35` | `prism-ml/Ternary-Bonsai-2-27B-gguf` | Qwen3.8 at ternary precision in a rotated basis; renders the Qwen profile by the catalogue's pin |

Requirements:

- The loader **must** validate the architecture identifier, dimensions,
  metadata, tensor names, shapes, offsets, and actual encodings against the
  adapter, and **must** reject an unsupported combination with a typed
  error naming the first offending tensor. It **must not** dispatch from a
  display name or requantize.
- `model pull` **must** verify the digest before publishing a file
  atomically, and **must** write a provenance sidecar beside it; a file
  whose encoding this build cannot execute keeps its sidecar and is
  reported, not deleted.
- `model inspect` **must** judge a file from its directory alone, over the
  network if needed, at four levels: storable (the encoding has a layout),
  executable (the adapter has kernels for it), bindable (every expected
  tensor present with its shape, nothing unexpected), supported (the
  catalogue pins the digest).
- Companion files are loaded only by the unit that consumes them; a
  missing or mismatched companion is a typed load error, never an implicit
  download or a silent fallback.
- The user root is `~/.nuclis` (`NUCLIS_HOME` overrides it with an absolute
  path); models live under `models/<owner>/<repo>/<file>`.

Read: [reference/artifacts.md](reference/artifacts.md),
[reference/gguf-inspection.md](reference/gguf-inspection.md), and the
family documents [reference/gemma4.md](reference/gemma4.md),
[reference/muse-glimmer.md](reference/muse-glimmer.md),
[reference/bonsai.md](reference/bonsai.md).

## 5. Engine requirements

### 5.1 Interface

The library **must** let a caller open a model, create a session, prefill,
step, verify a batch, stream generation, reset, and release, returning
typed results and metrics; the caller renders. Allocators and `Io` are
explicit; every returned slice documents whether it is owned or borrowed.
Per-token work **must** be free of string-based discovery: adapters,
profiles, kernels, and stop tokens are resolved at load.

### 5.2 Loading and validation

Parsing **must** be bounds-checked against the file size and the
parser's own limits before any weight byte is read, with overflow-checked
arithmetic on sizes. A malformed or truncated file is a typed error, never
a panic. Tensor encodings **must** be decoded by CPU reference decoders
pinned against the reference implementation's fixtures; every GPU decoder
**must** be bit-identical to its CPU decoder.

### 5.3 Memory model

Three kinds of memory with different owners: immutable mapped weights,
mutable per-conversation session state in one page-aligned block, and
per-step scratch. GPU-visible memory **must** outlive submitted work, and
the CPU **must not** touch session memory while a command buffer that
writes it is in flight. The session's layouts are declared by the adapter
(attention rows at F32 or F16, recurrent state); a draft source's cache is
one more layout in the same block.

### 5.4 Session contract

- A step is `ready → updating → ready`; a failed step leaves the session
  `failed` until `reset()`, because some layers may already have updated.
- `snapshot`/`restore` copy the used extent behind a digest of layouts and
  capacity; a mismatch is a typed error that touches nothing.
- Attention state rewinds by position; recurrent state is restored from a
  checkpoint. No operation **may** rewind a hybrid session by position
  alone.
- `reset()` rebuilds every kind of state. Different sessions share no
  mutable state.

Read: [reference/session.md](reference/session.md).

### 5.5 Execution

- Prefill runs in chunks through batched kernels; the chunk size is the
  plan's, and a chunked prefill **must** agree with the stepped path within
  the family's documented tolerance.
- Decode records one token's work into one command buffer; the production
  path installs no per-layer observer.
- The Metal bridge exposes matched create/destroy pairs, cleans up partial
  initialization, and documents for each operation whether it records,
  submits, or waits. The backend chooses dispatch geometry and legal
  fusion; Zig owns the plan. An optional tick **may** be installed on the
  wait; it runs on the calling thread, only during a wait longer than its
  interval, and **must not** touch the backend or the model.
- Sampling is a policy over a caller-owned history: the selection chain is
  penalties, then temperature and sort, top-k, min-p, top-p, then the draw,
  with the per-mode defaults owned by the profile. A GPU selection path
  **must** produce the same token as the reference sampler for the same
  seed or defer to it; it **must not** disagree.

Read: [reference/generation.md](reference/generation.md),
[reference/metal-backend.md](reference/metal-backend.md).

### 5.6 Speculative decoding

A draft source proposes tokens; the main model verifies them in one
batch; the accepted prefix is kept and the session recovered to it.

- The draft contract is family-independent (`propose`, `commit`, `reset`,
  `bytes`); each adapter names its source (an embedded block, a companion
  file of heads, a block drafter) and the loop never learns which.
- Recovery: attention by position; recurrent state checkpointed before
  the batch and restored on partial acceptance, with GPU work complete
  before host-visible state is touched.
- Acceptance is greedy when sampling is off (draft equals the row's
  argmax) and, when it is on, a draw of the target's own token per row
  with penalties and history advanced through the batch; every emitted
  token is a target draw. Identical seeded streams against ordinary
  decoding are not required.
- Three settings: the file (`models.<name>.mtp`, resolved and verified at
  load); the switch (`generation.speculative`, `--speculative on|off` on
  `generate`, `agent`, and `bench`, defaulting to the entry's measured
  verdict); the draft length (`generation.draft_length`, `--draft-length`,
  capped by a host constant and by the loaded drafter's block bound).
- Off **must** be the ordinary loop bit for bit; a drafter loaded but
  switched off costs memory and load time only. `bench` **must** open
  with no drafter at all when the switch is off, so the baseline is true.
- The acceptance rule and the recovery scheme are not user-visible
  settings.

Read: [reference/speculative-decoding.md](reference/speculative-decoding.md).

### 5.7 Extension rules

An additional family **must** cost a new adapter, its profile, its tests
and fixtures, and any genuinely new mathematics (a `cpu.*` function with
fixtures plus a kernel with a `metal-check` entry), and nothing else: no
edit to the GGUF parser, the sampling policy, the session, the generation
loop, or the Metal object lifecycle. A new storage encoding costs a CPU
decoder arm with its fixture, a bit-identical generic GPU decoder,
specialized kernels where wanted, and the adapter's executable claim.
Shared kernels **may** gain parameters and instantiations; a kernel that
exists for one family's shape stays explicitly specialized until another
family shows reuse.

Read: [reference/new-model-guide.md](reference/new-model-guide.md).

### 5.8 Configuration

One file, `~/.nuclis/nuclis.json`, with sections `engine` (model, backend,
`ctx_size`, `kv_precision`), `generation` (`max_tokens`, `think`,
`speculative`, `draft_length`, sampling overrides), `agent` (`think`,
`fold_thinking`, `theme`, `instructions`), and a `models` registry of named entries that
locate a file (path, or repository and file with a pinned revision), name
its companions, force a profile, and override any generation or agent key
for that model only. Precedence is defaults < profile < file < registry
entry < flags; `null` in the file means the profile's value. An unknown
key or an out-of-range value is a typed error naming the key. `bench`
ignores the file's sampling and budget so a measurement is reproducible
from its command line.

Read: [development.md § Configuration file](development.md#configuration-file).

## 6. Command-line interface

```text
nuclis agent [--model <m>] [--think <e>] [--resume [<id>]] [--system-prompt <path>]
nuclis agent -p "<prompt>" [--json] [--session <path>]
nuclis agent ls [--json]
nuclis generate --model <m> (--prompt <text> | --prompt-file <path>) [--raw] [--image <path>]... [--max-tokens <n>] [--think <e>] [--speculative on|off] [sampling flags] [--json]
nuclis bench --model <m> (--prompt-file <path> | --prompt-tokens <json>) --max-tokens <n> [--ctx-size <n>] [--kv f16|f32] [--speculative on|off] [--json]
nuclis tokenize --model <m> --prompt-file <path> [--raw] [--json]
nuclis inspect --model <m> [--json]
nuclis validate --model <m> [--json]
nuclis model pull (<name> | <owner/repo> --file <f>) [--revision <r>] [--role <role>] [--all] [--register] [--force] [--json]
nuclis model inspect (<name> | <owner/repo> --file <f>) [--revision <r>] [--json]
nuclis model ls [--json]
nuclis config init [--discover [--dry-run]] [--json] | show [--json] | set <key> <value>
nuclis --help | <command> --help | --version
```

| Command | Contract |
| --- | --- |
| `generate` | one completion: raw prompt or one rendered turn, streamed, seeded, the profile's per-mode sampling defaults with per-flag overrides, a structured stop reason, timings |
| `bench` | repeated cold and warm measurements with separate load, prefill, and decode timings; a prompt may be a token array so a reference's exact input is reproduced; the speculative pair measured on one loaded model in one process |
| `tokenize` | the prompt as `generate` would feed it, its ids and each token's byte offset, from the artifact's header only |
| `inspect` | identity, architecture, dimensions, the encoding histogram, validated ranges |
| `validate` | whether the file binds to its family's adapter, with the layer composition |
| `model` | pull with digest verification and sidecars, list the artifacts under the root, judge a file at the four levels of §4 |
| `config` | write the file with every catalogue model registered (`--discover` adds runnable files the catalogue does not name), show effective values with their source layer, set one key |
| `agent` | §7 |

Output rules:

- Generated text goes to stdout, diagnostics to stderr. `--json` uses
  versioned result types shared with the human-readable form.
- Reports are styled only on a terminal that advertises colour, at the
  strongest level it advertises; `--json`, pipes, `NO_COLOR`, and an unset
  or `dumb` `TERM` get plain bytes.
- Every stop is a named reason: EOS, token budget, context limit,
  cancellation, execution failure. Prompts are never silently truncated.
- Streamed text **must** be valid UTF-8 even when a token ends inside a
  character; ids stay exact.
- Every `--help` page is self-contained: usage, options, examples, notes,
  with no pointer at the repository's documents.

## 7. The agent

`nuclis agent` is the interactive surface of the engine and the project's
playground: every feature a user can feel is exercised here first. It is
one command; the evaluation CLI stays separate.

### 7.1 Surface

- Requires a TTY; `-p`/`--print` runs one turn without one (§7.7).
- Startup clears the visible screen (scrollback kept), prints the welcome
  in a box when the terminal is wide enough for the wordmark, primes the
  session (§7.5), and anchors the live region at the bottom: the active
  turn, the framed input box, a hint row, and the status bar.
- Completed turns are written once above the region and never repainted;
  the region is the only thing repainted, inside synchronized output.
  Insertion above the region uses a scrolling region with the top margin
  at row 1 so scrolled-off rows reach the scrollback; a terminal without
  the capability gets a cursor-up rewrite. The region's bottom is the
  anchor; blank rows between the transcript and the region are slack that
  insertions fill and a growing region takes back before anything scrolls.
- A resize replays the last turn at the new width; older turns are left
  as printed. Exit erases the region and restores the terminal, leaving
  only the transcript.
- Colour levels: truecolor, 256-colour, the sixteen ANSI slots, or plain
  attributes; `NO_COLOR` wins. Glyphs: Unicode, or ASCII when the locale
  is not UTF-8 or `NUCLIS_ASCII=1`. Code blocks draw no border, so a
  selection copies code alone. The default theme is Gruvbox dark; a theme
  supplies a palette only and can never change layout.
- The surface repaints at the engine tick's cadence through every GPU
  wait, so a prefill chunk and a long tool call both animate.

### 7.2 Editor

- Enter sends; Shift-Enter (kitty keyboard protocol, requested on entry;
  Ctrl-J is the fallback) inserts a newline. Word wrap at spaces; the
  cursor row is always visible; a scrolled editor shows how many rows are
  above.
- A bracketed paste of at least 4 lines or 400 bytes becomes a chip
  (`[pasted 96 lines, 6.1 KB]`) that moves and deletes as one unit; Ctrl-E
  expands it; the text itself is what is sent. Input limit 128 KiB.
- A paste that is one file's path (a file dropped onto the window) becomes
  an attachment chip: an image (`.png .jpg .jpeg .gif .webp .heic .tiff
  .bmp .ppm`, decoded by the vision contract) is `[image #N]`, whose bytes
  in the prompt are that marker and whose file is read at submit; a UTF-8
  text file within the input limit is `[file name, N lines]` over its
  content, fenced and headed by the path. `/image <path>` attaches an image
  the same way (completes like `@path`), and a typed image path in a
  prompt is attached at submit for terminals that do not bracket pastes.
  At most 8 images per prompt, numbered in attachment order and renumbered
  on deletion; Backspace removes a chip and its file; Ctrl-E turns an
  image chip back into its path. A model without a working projector
  refuses the chip with a notice naming the reason; the paste stays text.
- The transcript shows an attached image as a dim detail row under the
  prompt (`image #1: shot.png (320×240 → 10×8 tokens)`) and, on a terminal
  that draws kitty graphics (Ghostty, kitty; by environment, never under
  tmux, off with `NUCLIS_NO_PREVIEW=1`), a preview of at most 12 rows under
  it, fewer when less room stands above the input box. Sessions
  record an image's path and grid, never its pixels; a resumed session
  decodes it again and leaves a missing file out with a notice.
- Up/Down move inside a multi-line input and recall history from the
  first and last rows; history persists across sessions (200 entries).
- Tab completes a `/command` or an `@path`, else folds thinking; Ctrl-O
  folds the tool output of the last turn (the call rows stay, their detail,
  result, and diff rows go); Ctrl-T cycles effort; Ctrl-W cycles the
  context window (2K to 32K, re-opening the engine); Ctrl-N starts a new
  session; Ctrl-C cancels a turn, twice quits, or quits when idle; Ctrl-D
  quits.
- Ctrl-X copies the last answer to the clipboard through OSC 52 (at most
  256 KiB); Ctrl-G opens the input in `$VISUAL` or `$EDITOR` with the
  terminal released around the child and takes the edited text back.
- A line that starts with `!` runs the rest through the `bash` tool (same
  bounds, same workspace, Ctrl-C cancels): the transcript shows a
  `Bash(cmd)` block with the output under it, bounded to 40 rows, and the
  full output goes to the model as the next message, `$ cmd` first. `!!`
  runs and shows without sending. Both forms enter the prompt history.
- The editor stays live while a turn runs. Enter *steers*: the text is
  delivered as a user message before the model's next step, after the
  tool calls in flight, and appears in the transcript and the session
  where the model saw it; at most four wait at once, and what a turn ends
  without delivering goes out as the next message. Reasoning is
  interruptible, output is not: a steer that arrives while the step has
  only reasoned (no answer text, no call) stops the step, drops it from
  the history and the session (the transcript keeps what it showed, then
  `— steering: reasoning restarted`), delivers the message, and runs the
  step again against the step budget; Ctrl-C in that window still
  cancels the turn. Alt-Enter *queues* the
  text for after the turn; a queued message is not in the transcript
  until it is sent. Print mode has no steering.
- The input box is framed; the frame's colour is the reasoning effort and
  its top edge carries the spinner while a turn runs.

### 7.3 Transcript and status

- The agent produces typed events (user, thinking, answer, tool call, tool
  result, diff, notice, info, status, turn end); the transcript turns them
  into blocks and hands each closed block to the screen exactly once. An
  answer flushes block by block as its markdown blocks close.
- Thinking folds and unfolds without leaving the model conversation; a
  step's block closes with its own measured time.
- A tool call is a dot and `Name(argument)`, the dot coloured by state
  (running pulses; settled well, failed, or a write), with the tool's
  one-sentence result under it; a failed call shows its message, bounded.
  A run of calls is summed up in one dim row where the model's text
  resumes. A mutation is followed by its diff: a header with the path and
  the `+N −M` counts, a gutter of old and new line numbers, a marker cell,
  and the text on a band of its hue (dark green, dark red; the changed
  bytes on a brighter tint) padded to the width; side by side from 96
  columns and unified below that, folded when long. Sixteen-colour
  terminals keep the accent without the band.
- Answers render as markdown (headings, emphasis, inline code, links as
  hyperlinks, nested quotes, lists with task boxes, fenced code with a
  heuristic highlighter, rules, tables); a soft line break inside a
  paragraph stays a line break; model-supplied control bytes are stripped;
  no rendered row is ever wider than the terminal, whatever the input.
- The status bar shows measurements on the left (state, progress, the
  loop step, context use, token counts, prefill and decode rates) and
  settings on the right (effort, the speculative switch and draft length,
  cache precision, backend, model). A rate that was not measured prints as
  absent; a countdown is marked as an estimate. A narrow bar drops
  settings from the right.

### 7.4 Sessions and storage

```text
~/.nuclis/agent/
  sessions/<cwd-slug>/<timestamp>_<id>.jsonl   one file per session
  history.jsonl                                 prompt history
  exports/                                      /save markdown exports
```

- One JSON object per line: a header (format version, id, time, working
  directory, the model's path and verified digest, effort, context size),
  then entries with `id` and `parent`: `user`, `assistant` (thinking,
  answer, tool calls, stop, stats), `tool_result` (with its summary),
  `effort`, `context`, `compaction`, `notice`.
- Append-only, created at the first entry. A truncated last line is
  dropped on load; any other unparsable line or unknown entry type is a
  typed error naming the line; a newer format version is refused.
- Entries store what the model saw and produced, never terminal styling.
  `/save` derives markdown from the same entries. A failed write is a dim
  notice, never a lost turn.
- Resuming (`/resume`, `--resume [<id>]`, `latest` by default) replays
  the kept conversation through the profile into a fresh session: a
  prefill, never a state restore. `agent ls` lists a workspace's sessions.

### 7.5 The loop

A turn is a loop over steps, at most 16 per turn:

1. Render the system block, the history, and the tool definitions through
   the profile.
2. Stream the completion, forwarding events as they arrive.
3. Execute each decoded tool call in order: validate against the
   registry, run, append the typed result to history and the session.
4. Send the results back, or finish when the response holds no calls.
5. Stop on cancellation, an unrecoverable failure, or the step budget,
   and publish the reason.

- The system block is built from sections (`src/agent/system_prompt.zig`):
  identity and workspace, the working rules, one guideline per tool under
  the rendered tool definitions, the cost rule, the date and the `$ `
  convention for the user's own shell commands, and the project's
  instructions file (`agent.instructions`: `auto` reads `AGENTS.md`, then
  `CLAUDE.md`; `off`; or a path), cut at 8 KiB with a marked cut. Every
  sentence is judged on the playground task list
  ([development.md § The agent's task list](development.md#the-agents-task-list));
  `--system-prompt <file>` replaces the built sections for that purpose.
  The session is primed with the block before the first prompt and the
  snapshot restored on a new session, a resume, or a replay; a window too
  small for it is a notice at startup, and the warm-up notice counts the
  instructions file's tokens.
- Multi-turn without replay: the loop keeps the text the session has
  consumed and prefills only the increment; an effort change, a cancelled
  turn, or compaction resets and replays, and the bar says so.
- Compaction: one tool result may not exceed an eighth of the context
  window in tokens (never below 256); it is cut at a line boundary with a
  note saying how to ask for the rest. When a step still does not fit,
  older results of the turn become one-line stubs, then whole earlier
  turns leave the model's view, each recorded as a `compaction` entry;
  only then does the turn end with a message naming the tokens needed
  and how to raise the window. A turn that alone cannot fit fails with a
  named reason.
- Failures are results: an unknown tool, invalid arguments, a timeout, a
  truncated call, an empty result, or an ordinary tool failure returns a
  typed result to the model. Only an inference-transport failure ends the
  turn.

### 7.6 Tools

The workspace is the process's working directory, canonicalized at
startup; every path is resolved under it and a symlink escape is refused.
Limits are host constants, never model-supplied; truncation is always
marked; exceeding a limit is a typed result, not an abort.

| Tool | Contract |
| --- | --- |
| `read_file` | a line-addressed region: offset and count, 200 lines by default, at most 2,000 lines; a file over 1 MiB serves its first MiB with the size stated and the shell named for the rest; the result says where to continue; non-UTF-8 is a typed error |
| `write_file` | create or replace a UTF-8 text file, at most 1 MiB, by atomic replacement |
| `edit_file` | replace one exact, unique, non-empty sequence; zero or several matches change nothing |
| `glob` | one pattern (`*`, `?`, classes, `**`); hidden entries only when named; at most 200 results in stable order |
| `grep` | literal, case-sensitive; skips hidden entries, symlinks, binary-looking files, and generated trees; at most 200 matches |
| `bash` | one command with a minimal environment; combined output at most 1 MiB; 300 s; cancellation, the timeout, and the output bound kill and reap the child |

Read: [reference/agent-concepts.md](reference/agent-concepts.md),
[reference/tool-calling.md](reference/tool-calling.md).

### 7.7 Print mode

`nuclis agent -p "<prompt>"` (or `--print --prompt-file <path>`) runs one
turn without a TTY: the text form streams the answer alone; `--json`
writes every event as one object per line. A turn that produced no answer
says so on stderr. Nothing is recorded unless `--session <path>` names a
file. `--json` without print mode is an error.

### 7.8 Not in scope

HTTP serving, a permission system, MCP or extensions, subagents, a docs or
API lookup tool, and a second product surface.

## 8. Performance requirements

- The primary workload is one user doing coding work: measure time to
  first token after a substantial code prompt and steady decode.
- The acceptance context is 32,768 tokens on the 48 GB target: every
  length of the reference workload **must** complete on its token budget
  with no swap growth, with headroom left for the operating system and
  the user's tools. The record is
  [reference/bench.md § Acceptance runs](reference/bench.md#acceptance-runs).
- Weights stay quantized on the GPU path; prefill is chunked with planned
  scratch.
- Claims are made against the pinned llama.cpp build on the exact token
  arrays, with equivalent context, cache precision, sampling, and output
  budget, and each record names chip, OS, compiler, revision, artifact
  hash, power mode, warm-up policy, and memory use. Prefill and decode are
  reported separately; no absolute throughput target is set by this
  document.
- A kernel or loop change ships when it is at least as fast as its
  control on the workload it targets and the correctness gates hold;
  a negative result is recorded with its numbers and closed.
- Speculation ships on for a family only when its measured decode rate
  exceeds the ordinary rate on the family's acceptance workload.

## 9. Verification

- **Default tests** (`make test`) need no network, credentials, GPU, or
  model, and run under a leak-checking allocator with error paths covered.
  They **must** cover malformed and truncated GGUF data, overflow, block
  decoding, tokenization, prompt rendering, the tools against temporary
  workspaces, the loop against a stubbed completion, and the surface's
  escape streams.
- **Fixtures** are outputs of the reference on small inputs, committed
  with the revision that produced them; decoders match exactly, F32 GPU
  reductions match F64 sums within stated tolerances.
- **GPU checks** (`make test-metal`) compare every kernel with the CPU
  reference or the fixtures and exercise lifecycle, cancellation, and
  cleanup separately, since the Zig allocator cannot see GPU allocations.
- **Gates** (`make verify`, `verify-cpu`, `verify-changed`) are the
  model-specific checks in `gates.json`, tiered by cost and selected by
  changed paths: per-layer traces against the reference for every family
  and cache precision, generation checks that chunked prefill agrees with
  the stepped path, that a reset reproduces a fresh session, and that
  sessions share nothing, and the speculative equivalence checks. Every
  threshold is written down per numerical mode; passes are recorded with
  dates and observed maxima.
- **Workloads** (`make workload`) are the benchmarks in `workloads.json`;
  their reports are saved by revision and the record tables are generated
  from them.
- **Definition of done** for a change: build, tests including error
  paths, formatting, the happy path on the freshly built binary, the gates
  its paths select, the documents it affects, and a report of what
  changed, what was verified, and what remains.

Read: [development.md § Gates](development.md#gates) and
[§ The record](development.md#the-record).

## 10. Deferred work and non-goals

Vision input through the companion projectors is in progress: the Qwen3.8
projector and Gemma 4's two (the 12B's unified embedder, the 26B-A4B's
SigLIP encoder, with the language model's bidirectional image spans) ship
(`generate --image`, the chat's image chip, the shared
`inference/src/vision/` contract, [reference/vision.md](reference/vision.md));
the Muse Glimmer projector is planned in [TODO.md](../TODO.md).
PDF attachments are not planned: nothing in the tree extracts their text. Deferred, to be taken through the existing seams as
concrete requirements arrive: HTTP serving with an OpenAI-compatible protocol,
persistent prefix caches, concurrent request batching, additional GPU
backends, and a teacher-forced `eval` command. A local server would wrap
the library; tool execution and permissions would stay with the consuming
agent. A docs or API lookup tool for the agent was assessed and not
scheduled: read-only docs roots would be the cheapest form, a bounded
fetch would reopen the no-permission decision, an embedding index is
ruled out.

Non-goals: training, model conversion or quantization tooling, universal
GGUF support, a general tensor compiler, and matching any external
product's feature list.

## 11. Open questions

- Whether tool steps want different sampling defaults for reliability;
  measure before changing anything.
- Whether a `bash` allow-list or an approval step is wanted once the
  agent has run on more real tasks; the inline choice component exists
  either way.
- Whether the sliding-window caches should be ring buffers; the full
  allocation was kept with the numbers on record.
