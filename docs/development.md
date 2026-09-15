# Development guide

Product behavior and acceptance criteria belong in [spec.md](spec.md);
agent-specific working instructions belong in [../AGENTS.md](../AGENTS.md).

## Current state and layout

Inspection, Qwen structural validation, native CPU generation, the opt-in
GPU-resident Metal backend (`-Dmetal=true`), `generate`, `bench`,
`tokenize`, `config`, `model pull` / `model ls`, and the `nuclis agent`
surface work; see
[../TODO.md](../TODO.md) for what is in progress and
[engineering-log.md](engineering-log.md) for what
closed. Teacher-forced scoring (`eval`) is a future increment.

```text
src/                 executable: CLI, model commands, config
  src/tui/           terminal surface, engine-free: screen, editor, theme,
                     markdown, keys
  src/agent/         agent composition: engine, conversation, sessions, tools
inference/           engine library: runtime, quantization, tokenizer,
                     sampling, backends, profiles
huggingface/         Hub download library (Xet) and its standalone binary;
                     imported by src/ only
docs/                spec, architecture, guides, reference, engineering log
TODO.md              active plan: unfinished units only
build.zig            build
build.zig.zon        manifest; single source of the version
```

The executable composes the `inference` and `huggingface` modules through
the root build. Keep kernels with the inference library that owns them.

## Environment

Facts every unit depends on; keep them here, not in `TODO.md`.

- Zig 0.16.0 at `/Users/alef/.local/opt/zig/stable`; consult its installed std
  source for API details. Apple M4 Pro, 48 GiB, macOS 26. Metal compiles
  shaders at runtime from the Command Line Tools; Xcode-only tools are reached
  per process (see [../AGENTS.md § Local toolchain notes](../AGENTS.md#local-toolchain-notes)).
- Model `~/.nuclis/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf`
  (catalogue name `qwen3.8-27b`, the default `engine.model`), SHA-256
  `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482`,
  from [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
  at commit `4ca720788d1e01f1bff70c033e0d0028fd02e502`, pulled from a
  clean state by `nuclis model pull qwen3.8-27b --all` on 2026-09-11 (MODL-03).
  Its companions `mmproj-BF16.gguf` and `MTP/mtp-Qwen3.8-27B-Q4_0.gguf`
  sit beside it, verified and not executed
  ([reference/artifacts.md](reference/artifacts.md#pinned-commits-and-digests-modl-02-2026-09-11),
  [reference/gguf-inspection.md](reference/gguf-inspection.md#companion-files-in-modelsqwen-2026-09-08)).
- Reference llama.cpp `7620399` builds on demand under
  `.zig-cache/reference/llama.cpp` (vanishes with `make distclean`). The
  reference oracle itself is committed under `tests/fixtures/`
  ([provenance](../tests/fixtures/provenance.md)); `make compare` reads
  `tests/fixtures/reference-hello-comma`. Accepted reference warm rates
  (prefill/decode tok/s): 512 in = 89.19/9.66; 4K = 89.26/9.21;
  16K = 74.07/7.32; 32,639 = 67.28/6.71
  ([reference-baseline.md](reference/reference-baseline.md)). nuclis on the
  same token arrays (ENGN-07 record, 2026-09-10): 512 = 90.45/10.62; 4K =
  83.70/10.20; 16K = 62.70/8.27; 32,639 = 49.55/7.55
  ([bench.md § Acceptance runs](reference/bench.md#acceptance-runs)).
- `make help` lists all tasks. Per-commit gate: `make check` (fmt-check, unit
  tests, `test-metal`), `make compare` (both cache precisions: `compare-f32`,
  129 trace files, measured max abs 6.1e-5 at the bring-up
  thresholds; `compare-f16` at the F16 tolerance, max abs 0.03 / relative
  RMS 0.0002, measured 0.0250 / 0.00019), `make bench` when performance is claimed,
  `make test-generation-metal` when session state or the step contract
  changed; `make compare-gemma4` (`compare-gemma4-cpu`, `-f32`, `-f16`: the
  Gemma 4 12B CPU reference and the Metal plan in both cache precisions
  against the pinned traces of the catalogue's `gemma-4-12b`, three positions;
  the CPU and F32 rows at the bring-up thresholds, F16 at the model's own
  tolerance 1.0 / 0.05, [gemma4.md](reference/gemma4.md#q4_0-path-and-the-qat-file-modl-08-2026-09-12);
  `GEMMA_MODEL` overrides the file's path) and `make compare-gemma4-qat`
  (the same three on `gemma-4-12b-qat` and its own traces,
  `GEMMA_QAT_MODEL`) when the second adapter or shared math changed,
  and `make test-generation-gemma4-metal`
  (the generation check on the Gemma plan; `-qat-metal` runs it on the QAT
  entry, whose Q4_0 path amplifies the chunk rounding, and
  `test-generation-gemma4` is the CPU run at ~35 s per token) with it; `make test-vocabulary` on either
  pinned file (`MODEL=<gemma path>`) when the tokenizer or a profile
  fixture changed, and `make baseline-gemma4-qat` for the Gemma acceptance
  workload on the QAT entry
  ([bench.md](reference/bench.md#gemma-4-12b-acceptance-record-qat-file-modl-08-2026-09-12);
  `baseline-gemma4-qat` for the QAT entry's own record). `generate --prompt-tokens <json>` feeds a token
  array untokenized, as `bench` does. `make bench-kernels` ranks matvec kernel variants without a model,
  `make bench-matmul` the prefill tile per encoding (generic and specialized;
  `ARGS=<tokens>` for a chunk other than 256);
  `make bench-profile` times every dispatch inside real tokens (diagnostic,
  ~8 % perturbation); `make trace` records a Metal System Trace;
  `make baseline` runs the reference workload on the committed token arrays
  and writes the dated record under `docs/benchmarks/`
  ([bench.md § Acceptance runs](reference/bench.md#acceptance-runs)).

## Toolchain

The tested compiler is Zig 0.16.0; manifests require at least 0.16.0.
Verify standard-library calls against that toolchain's installed source and
language reference rather than assuming older Zig examples still apply.

Local language reference:
[/Users/alef/.local/opt/zig/stable/doc/langref.html](/Users/alef/.local/opt/zig/stable/doc/langref.html).

The Metal backend additionally needs Apple's SDK/frameworks and the
Objective-C compiler. The [reference baseline](reference/reference-baseline.md)
compiles embedded shader source through Metal at runtime using Command Line
Tools; a standalone `metal` compiler is needed only for an offline shader
build path. Pin nuclis's tested macOS/SDK requirements during backend
bring-up. Swift is not a project build requirement.

Pass allocator and Io dependencies explicitly. Prefer a separate pure function
for decisions that can be tested without file, network, process, or GPU access.
Document ownership on returned allocations and the lifetime of borrowed data.
Use scoped cleanup for normal paths and partial-initialization errors.

`main(init: std.process.Init)` receives the allocator, environment, arguments,
and Io implementation. Pass `init.io` through file operations; keep decisions
and in-memory parsing independent of OS access. Use `std.Io.Reader`/`Writer`
and `std.Io.Dir`/`File`, checking APIs against the installed standard library.

Zig 0.16's startup supplies `Io.Threaded`. The experimental `Io.Evented` backend
selects io_uring on Linux and Dispatch on macOS; io_uring does not run on the
target Mac. Keep the injected default until measurements justify changing it.
See the [0.16 release notes](https://ziglang.org/download/0.16.0/release-notes.html)
and installed `lib/std/start.zig` and `lib/std/Io/Evented.zig`.
GPU scheduling will be a separate Metal backend responsibility.

ReleaseSafe is the initial release configuration; ordinary `zig build` uses
Zig's Debug default. A faster optimization mode is justified only by numerical
validation and measured improvements; benchmark results must name the actual
mode used.

### Zig 0.16 usage audit

Audited 2026-09-09 against the
[0.16.0 release notes](https://ziglang.org/download/0.16.0/release-notes.html);
"uses"/"does not use" are grep results over `src/` and `inference/`. The tree
adopted the `Io` interface end to end (`std.process.Init` with `io`,
`std.Io.Dir`/`File`, `std.Io.Reader`/`Writer`, `File.createMemoryMap`,
`std.Io.Clock`) and deliberately skips the concurrency layer (`io.async`,
`Io.Mutex`, `Io.Evented`): the engine is one sequential GPU pipeline,
cancellation is a flag read between layers in `src/interrupt.zig`, and the chat
polls with `std.posix.poll` because `Io.Evented` is experimental. It is clean
on the other 0.16 removals (`@cImport`, `@Type`, `@Vector` indexing,
`Thread.Pool`, `SegmentedList`, `ArenaAllocator` locking).

Two spots to migrate at the next compiler upgrade, both isolated from engine
code:

| Use | Count | File | Migration |
| --- | ---: | --- | --- |
| `std.fs.path.join` / `isAbsolute` / `dirname` | 7 | `src/paths.zig`, `src/config.zig` | `std.Io.Dir.path` equivalents |
| `std.posix.sigaction`, `Sigaction`, `SIG`, `SA.RESETHAND`, `sigemptyset` | 5 | `src/interrupt.zig` | `std.posix.system` calls, or an `Io`-level signal facility if a later release adds one |
| `std.posix.tcgetattr` / `tcsetattr` / `termios` / `poll` / `pollfd` / `winsize` / `system.ioctl` | 11 | `src/tui/terminal.zig` | `std.posix.system` for raw mode and size; `Io` polling once `Io.Evented` is usable |

The compiler is a separate versioning axis, so the upgrade is its own unit;
every benchmark record names the exact Zig used.

## User directories

nuclis uses `~/.nuclis` as its user configuration/data root. `NUCLIS_HOME`,
when set, overrides it and must be an absolute path. An empty or relative
override is an error. Otherwise resolve the root from `HOME`.

```text
~/.nuclis/
  models/                  model artifacts: <owner>/<repo>/<file> plus a provenance
                           sidecar <file>.nuclis.json per verified file ([reference/artifacts.md](reference/artifacts.md))
  nuclis.json              engine configuration (APPS-03; `nuclis config init` writes it)
  agent/                   the agent's data root (`paths.agentPath`; TERM-01)
    history.jsonl          submitted prompts across sessions (TERM-01; append-only,
                           one JSON object per line, tail-read at startup)
    sessions/<cwd-slug>/   one append-only JSONL file per session
    exports/               `/save` markdown exports
                           ([agent-spec.md](agent-spec.md#sessions-and-storage))
  cache/                   regenerable runtime data (planned)
```

Create directories only when an operation needs to write them. Inspection/help
must not create settings or directories. Do not migrate or overwrite existing
user files implicitly. Credentials remain environment-only (`HF_TOKEN` for
gated Hub repositories; public ones need none).

### Model download

`nuclis model pull <name> [--with mmproj,mtp | --all]` fetches a catalogue
entry (`src/catalog.zig`, the only source of "supported": name, repository,
file, pinned commit, digests, companions with the unit that will load them;
`qwen3.8-27b` today) into `<root>/models/<owner>/<repo>/<file>` through the
`huggingface` package ([its README](../huggingface/README.md): native Xet
reconstruction, SHA-256 verified, atomic publication, verified reuse of a
file already in place) and writes the sidecar beside each file; the Hub's
digest at the pinned commit must equal the catalogue's (`CatalogMismatch`
otherwise, and no sidecar). `nuclis model pull <owner/repo> [--file <name>]
[--revision <rev>] [--role main|mmproj|mtp|imatrix]` fetches any other
GGUF by repository id: the revision (`main` by default; a tag, branch, or
commit) is resolved once, printed as the 40-character commit, and that
commit pins the transfer and is what the sidecar records; a repository
with several GGUFs and no `--file` prints the choices with sizes and fails
with `SelectionRequired`; exact names match in full, subdirectories
included (`--file MTP/mtp-Qwen3.8-27B-Q4_0.gguf` keeps the subdirectory).
Both forms take `--force` and `--json` and need the root and nothing from
`nuclis.json`. A registry entry name (`nuclis model pull gemma`, see
[§ Configuration file](#configuration-file)) is the one form that reads the
file: it pulls the entry's `repo`/`file` at its `revision` (`main` when
unset) with the Hub's digest, `--with mmproj,mtp` or `--all` adding the
entry's companion names; an entry that names a `path` has nothing to pull
(`NotPullable`), and a name that is none of the three forms is
`UnknownModel`. A second pull of the same file hashes it, downloads nothing,
and rewrites the sidecar. A sidecar recording other content is `ExistingFileMismatch` unless
`--force` replaces file and sidecar; a differing file nuclis never verified
is the same error from the package, and `--force` replaces it too. Progress
is one updating line on stderr (bytes, rate, ETA) on a terminal, phase lines
otherwise; Ctrl-C cancels through the package's sink, which removes the
partial file (a second Ctrl-C kills the process the ordinary way and may
leave the temporary file). `nuclis model ls [--json]` prints the catalogue
with each entry's local status from its sidecar alone (`present`,
`absent`, `mismatch`, `unverified`: the file is there without a sidecar)
and its companions beneath with a "not loaded yet" note, then the other
GGUF files in the layout with their sidecar facts (runnable only if their
architecture has an adapter); files above `<owner>/<repo>/` are counted,
not listed. `make model-ls` wraps it. `--model` and `engine.model` accept
a registry entry, a catalogue name, or a path; a missing file fails before
anything opens, naming the resolved path.

`nuclis model inspect (<name> | <owner/repo> --file <name>) [--revision <rev>]
[--json]` answers "will this quantization load" before a download. It lists
the repository at the resolved commit, fetches the head of the file through
the package's `readRange` in 8 MiB windows until the GGUF directory parses
(never past the parser's 64 MiB directory bound; the Qwen directory is
11.0 MB and Gemma 4 12B's 15.8 MB, two requests and about 4 s each on
2026-09-11), prints what `inspect` prints for a local file, and ends with
a verdict: `supported` (the catalogue pins the Hub's digest for the file
and the adapter binds the directory), `runnable` (an adapter for
`general.architecture` binds it, but the file is not in the catalogue or
carries another digest), or `not runnable` naming the first offending
tensor and its encoding (a layout nuclis does not store, or one outside
the adapter's executable set), the missing adapter (`gemma4`, `clip`), or
the adapter's rejection; a catalogue companion (`mmproj-BF16.gguf`) is
reported as the companion it is. Nothing is written and no weights are
downloaded; a repository with several GGUFs and no `--file` lists them as
`pull` does.

### Configuration file

`nuclis.json` is one sectioned document (`engine`, `generate`, `agent`,
and the `models` registry; the `agent` section holds the agent surface's
settings (`think`, `fold_thinking`, `theme`) and was named `chat` until
2026-09-11, see
[agent-spec.md](agent-spec.md#configuration)) with a `schema_version`;
`src/config.zig` is its schema and the built-in defaults. `nuclis config
init` writes the defaults with every catalogue model as a registry entry
(one today), so the file shows the entry shape with the catalogue's facts
(the entries are optional: a catalogue name resolves without one), then
prints the effective engine keys, each catalogue model's local status,
and the `nuclis model pull <name>` to run next:

```json
{
  "schema_version": 1,
  "engine":   { "model": "qwen3.8-27b", "backend": "metal", "ctx_size": 8192,
                "kv_precision": "f16" },
  "generate": { "max_tokens": 2048, "think": "off",
                "sampling": { "temperature": null, "top_k": null, "top_p": null, "min_p": null,
                              "presence_penalty": null, "repetition_penalty": null } },
  "agent":    { "think": "low", "fold_thinking": true },
  "models":   { "qwen3.8-27b": { "path": null,
                                 "repo": "unsloth/Qwen3.8-27B-GGUF", "file": "Qwen3.8-27B-UD-Q4_K_M.gguf",
                                 "revision": "4ca720788d1e01f1bff70c033e0d0028fd02e502",
                                 "mmproj": "mmproj-BF16.gguf", "mtp": "MTP/mtp-Qwen3.8-27B-Q4_0.gguf",
                                 "ctx_size": null,
                                 "generate": { "max_tokens": null, "think": null, "sampling": { "…": null } },
                                 "agent": { "think": null, "fold_thinking": null } } }
}
```

- Precedence: built-in defaults < the model's sampling profile < the
  file's global sections < the registry entry the model names < command-line
  flags. `NUCLIS_HOME` only moves the root; there are no per-key environment
  overrides and no per-project files. `nuclis config show [--json]` prints
  the effective value of every key with its source (`default`, `profile`,
  `file`, `model`, `flag`), one line above the table naming the model, how
  it resolved (registry entry, catalogue name, path), its profile, and that
  a `null` sampling key takes the profile's value for the configured
  `generate.think`; then each registry entry's stated keys. The profile
  named there is the catalogue entry's (the first profile for a bare path
  or an unknown registry name), chosen without opening the file; a run
  samples with the opened file's own profile, selected by its template
  digest, so a Gemma file reached through a path still gets Gemma's
  defaults (MODL-07). The JSON form
  carries the file as loaded (`config`, the registry as a map), the
  `effective` view, and the `sources` map. The flag layer is visible in each
  command's own report (`generate --json`, the `bench` report's `config`
  and settings fields).
- `engine.model` is a registry entry name, a catalogue name (`qwen3.8-27b`,
  the default), or a path, tried in that order; a path resolves under
  `<root>/models` unless absolute. `--model` takes the same forms, a path
  being as given. `engine.backend` defaults to `metal` when the build
  has it, `cpu` otherwise. `engine.kv_precision` (`f16` default, `f32`;
  flag `--kv`) is the attention cache layout on the GPU: `f16` halves the
  cache's memory and the bytes attention reads per token (KERN-07); the CPU
  reference always keeps F32, and every report (`generate --json`, the
  `bench` header and JSON) states the precision the session actually used
  beside its `session_bytes`. `bench` takes model, backend, and context
  (the entry's `ctx_size` included) from the file and keeps its output
  budget (32), repetitions, and greedy sampling on the command line so runs
  stay comparable; its report records the file it ran with.
- The registry: `models` maps a name (1..64 printable characters, no `/`,
  not ending in `.gguf`) to an entry that locates a model one way, `path`
  (relative to `<root>/models` unless absolute) or `repo` + `file` (the
  layout `model pull` writes, `<root>/models/<repo>/<file>`) with an
  optional `revision`; optional `mmproj` and `mtp` companion file names
  in the same directory (recorded for the units that will load them, the vision unit
  and the MTP unit, and used by `model pull <name> --with`); and optional
  `ctx_size`, `generate` (`max_tokens`, `think`, `sampling`), and `agent`
  (`think`, `fold_thinking`) overrides that apply only while that entry is
  the model, `null` meaning the global value. Entries pin no digest (a
  pull by entry name takes the Hub's). A registry name shadows a catalogue
  name for `--model`/`engine.model`; `model pull` tries the catalogue
  first, since it needs nothing from the file.
- Sampling entries are overrides: `null` means the official profile of the
  reasoning mode ([generation.md](reference/generation.md#sampling-profiles-and-the-selection-chain-modl-01)),
  so the file never freezes a model's recommended settings. The profile is
  the adapter's (`qwen38` for the one adapter; the catalogue records it per
  entry so `config show` names it without opening the file; the adapter registry dispatches
  per architecture).
- Validation: unknown keys are rejected with their dotted path (registry
  keys as `models.<name>.<key>`, so an unknown companion such as
  `imatrix` is named), wrong types and enum values name the key and the
  accepted form, ranges are the same as for flags (context 1..32768,
  tokens 1..4096, sampling options through the sampler's rules), an entry
  must locate its model one way, the file is bounded at 64 KiB, and a
  `schema_version` other than 1 is an error that states the migration
  (move the file aside, `config init`, copy settings back). Adding a key
  with a default does not bump the version (the registry was added to
  schema 1; a file without it loads with an empty one); renaming or
  re-typing one does, except the `chat` → `agent` section rename, whose
  keys and defaults are unchanged: a file with a `chat` section is
  rejected with a message that says to rename it. Keys arrive with the features that read them
  (`engine.kv_precision` arrived with KERN-07; an `agent` section comes with
  the embedded agent).

### Styled output

Every text report (`config show`, `model ls|pull|inspect`, `inspect`,
`validate`, `tokenize`, `bench`) and the `error:` line on stderr are
colored with the agent's palette (gruvbox dark by default) when the stream
is a terminal
and the environment advertises color (`COLORTERM=truecolor` for 24-bit,
a `256color` or `direct` `TERM` for the approximations, any other terminal
for the sixteen ANSI slots). `--json`, a pipe, `NO_COLOR`,
or an unset or `dumb` `TERM` disable styling completely, so scripts and the tests
see exactly the same bytes; the tests pin the plain form with
`style.Style.none`. The renderers take a `style.Style` explicitly and pad
text before wrapping it in escapes, so alignment never depends on them.
The palette lives in `src/tui/theme.zig` (TERM-01): named palettes selected by
`agent.theme`, behind a semantic style enum that a theme cannot change, so
a theme changes colour and never layout. `src/tui/style.zig` is the same
palette applied to one-shot reports.

Dim is a colour, not a faded one: the `dim` role paints gruvbox's grey and
adds the SGR dim attribute only at the plain level, where there is no colour
to carry the meaning. Stacking both halves an already low-contrast foreground
and made notices and the help page unreadable on a translucent terminal
(reported and fixed 2026-09-12).

Glyphs are a separate axis from colour (TERM-01 step 6). Every decoration the
agent draws — bullets, task boxes, rules, table joints, fold arrows, the
spinner, the status-bar labels — is named in `theme.Glyphs`, with a Unicode
table and an ASCII one. The ASCII table is selected when the locale does not
claim UTF-8 (`LC_ALL`, then `LC_CTYPE`, then `LANG`, none of which contains
`utf-8`/`utf8`) or when `NUCLIS_ASCII=1` is set, which is also how to check
the fallback on a UTF-8 terminal. Colour and glyphs never influence each
other: `NO_COLOR` keeps the Unicode drawing, and an ASCII terminal keeps its
colours.

### The agent's live region

`src/tui/screen.zig` is the only module that emits a movement escape, and
its geometry is the contract the rest of the surface is written against: a
repaint rewrites the live region in place inside synchronized output
(`CSI ? 2026 h/l`), an insertion above it narrows the scrolling region to
the rows above (`DECSTBM`, top margin row 1 so scrolled-off rows still
reach the scrollback) and scrolls only those, and the region's *bottom*
stays anchored, so a turn that grows the region pushes the transcript up
and one that shrinks it releases rows above the editor rather than
leaving blanks beneath it. A resize replays only the last turn at the new
width; older turns are left as the terminal reflowed them (agent-spec:
completed turns are immutable). `NUCLIS_NO_SCROLL_REGION=1` forces the
cursor-up rewrite fallback for a terminal that mishandles `DECSTBM`, and a
`dumb` or unset `TERM` turns both capabilities off. The escape stream of
every operation is pinned by golden tests that need no TTY.

### The agent's transcript

Between the renderer and the screen sits `src/tui/transcript.zig`, the
answer to "what is on the screen, and who may rewrite it" (TERM-01 step 7). The
agent produces typed events (`src/tui/event.zig`: user, thinking, answer,
tool call, tool result, diff, notice, status, turn end); the transcript turns
them into blocks and offers three views of those blocks:

- `takeClosed` — rows for everything that closed since the last call, marked
  written. The agent inserts them above the live region. **Exactly once**:
  the same rows can never be handed out twice, which is what makes a
  scrollback both append-only and correct. An answer flushes block by block
  as its markdown blocks close, so a long answer scrolls away while it is
  written.
- `liveRows` — what is still open, tail-clamped under `… N lines above`.
- `replayRows` — every written block again, for the two events that may
  rewrite the scrollback (a fold toggle, a resize). The agent skips the
  rewrite when the turn is taller than the space above the region: those
  rows are in the scrollback and cannot be reached.

The transcript is pure: no `Io`, no clock (the animated thinking label is
passed in by the agent, which has both), and no knowledge of tokens or
models. That is what lets its tests drive a whole turn — including a
byte-by-byte stream — with `std.testing.allocator` and no TTY.

### The agent without a terminal

`nuclis agent -p "<prompt>"` (or `--print --prompt-file <path>`) runs one
turn with no TTY: the text form streams the answer, `--json` writes every
event as one object per line, and `--session <path>` is the only way a
printed turn records anything. It is the scripting entry point today and the
harness the agent loop will be tested through in phase 2, where a stream of
JSON lines is something a test can assert on and a terminal is not.

### The agent's session files

`src/agent/session.zig` writes one append-only JSONL file per conversation
under `~/.nuclis/agent/sessions/<cwd-slug>/<stamp>_<id>.jsonl` (TERM-01 step 8).
The first line is a header (format `version`, session id, time, working
directory, the model path and the digest its pull sidecar recorded, effort,
context size); every later line is an entry carrying `id` and `parent`, so a
branch or a cancel-rewind is representable later without a migration. The
file is created at the first entry, so starting the agent and quitting writes
nothing. On load, a truncated *last* line is dropped — that is where a crash
lands — while any other unparsable line, or an unknown entry type, is a typed
error naming the line number; a file from a newer `version` is refused
outright. `/save` derives markdown from the same entries, so there is no
second transcript format.

### The agent's renderer

`src/tui/markdown.zig` turns a turn's text into pre-styled, pre-wrapped rows.
Two rules matter when reading it (TERM-01 step 6):

- **Streaming.** `markdown.split(text)` divides a partial turn into the
  blocks that can no longer change and the one still being written: a blank
  line ends a block, a heading/rule/quote/list item ends one at its newline,
  and a paragraph, a table, or an open fence keeps its block open. The agent
  renders the closed part and shows the open part as raw text, repainting
  both on every token, so styling appears block by block instead of at the
  end of the turn. A trailing newline is the end of the last block, not an
  empty one after it — which is what makes the rendered prefix of a partial
  turn a prefix of the finished one, a property a test pins byte by byte.
- **Wrapping.** `view.lines` and `view.wrapStyled` take a `Wrap` mode.
  Everything a reader reads wraps at the last space that fits (`.word`, the
  editor's rule since step 3); only rows that are truncated to one line
  anyway — the status bar, the shortcut hint — use `.character`. A word
  longer than the row still breaks, and a space that lands past the edge
  becomes the next break point instead of breaking the row, so a word ending
  exactly at the last column keeps it.

## Continuous integration and releases

Two workflows under `.github/workflows/`, both on `macos-15` (Apple Silicon),
both installing the compiler with `.github/install-zig.sh`: the version comes
from `minimum_zig_version` in `build.zig.zon` and the SHA-256 from
`.github/zig-toolchain`, verified before anything is unpacked. No marketplace
action installs the toolchain and no digest is fetched at run time — this
tree pins a digest for every artifact it downloads, and the compiler is the
one it cannot do without. A Zig upgrade edits the manifest, that file, and
the toolchain section above in one unit of work; the script refuses any
version the two do not agree on. The actions that do run
(`actions/checkout`, `actions/cache`) are pinned by full commit SHA rather
than a moving tag, and `persist-credentials` is off, so the checkout cannot
push back into the repository.

Both scripts are POSIX shell and awk. A workflow step should depend on the
tools that are on every machine and nothing else, which is why neither `jq`
nor Python appears in CI even though the repository's local tooling
(`scripts/*.py`) is written in Python.

**`ci.yml`** (push to `main`, pull requests): `zig fmt --check`, a semver
check on `build.zig.zon`'s version, `zig build test`, a
`-Dmetal=true -Doptimize=ReleaseSafe` build with `--version`/`--help`
on the result, and a `-Dmetal=false` build so the non-Metal path keeps
linking. That is `make check` minus the GPU. **Not** in CI: `test-metal`,
`compare`, `bench`, and anything that pulls a model — they need the pinned
artifacts and the real M4 Pro, and a rate measured on a virtualised GPU is a
number nobody should trust. Those gates stay local and their evidence stays
in the engineering log.

**`release.yml`** (a `v*` tag) refuses to publish, before it builds anything,
unless:

1. the tag is `vMAJOR.MINOR.PATCH[-prerelease]`;
2. the tag equals `build.zig.zon`'s `.version` — the check that catches a tag
   cut before the `-dev` suffix was stripped;
3. the installed `zig version` equals `minimum_zig_version`;
4. `CHANGELOG.md` has a section for the tag.

It then runs the same gate, builds `-Dmetal=true -Doptimize=ReleaseSafe`,
asserts the binary reports the tag's version, and publishes three assets: the
binary tarball (`nuclis-vX.Y.Z-aarch64-macos.tar.gz`), a source tarball
(`nuclis-vX.Y.Z-src.tar.gz`, cut with `git archive` from the tag), and a
`SHA256SUMS` covering both. The release notes are the tag's own section of
`CHANGELOG.md` plus a fixed footer, sliced by `.github/release-notes.sh`, so
notes and changelog cannot drift.

The binary is **unsigned and not notarized** (there is no Apple Developer ID
for this project), so macOS quarantines it on download; the notes say to run
`xattr -d com.apple.quarantine nuclis`, and building from source stays one
`zig build` away.

## Build and format contract

The root build provides:

```sh
zig build                   # Metal is on by default for macOS aarch64
zig build test              # src/, inference/, and huggingface/ unit tests
zig build -Doptimize=ReleaseSafe
zig build -Dmetal=false     # a CPU-only binary (the default off Apple Silicon)
zig build test-hf           # the huggingface package's tests alone
zig build hf-downloader     # its standalone binary, zig-out/bin/hf-downloader
```

`-Dmetal` defaults to **on for macOS on aarch64** and off everywhere else
(2026-09-12): the configuration's default backend is `metal`, so a plain
build that left it out produced a binary that failed at run time with
`MetalNotEnabled` against its own defaults. A build without the backend now
says so and names both ways out (rebuild, or `--backend cpu`). Building the
bridge does not make the default tests need a GPU: the fixtures that execute
kernels are the separate `test-metal` step.

The root `Makefile` wraps these and the explicit model/GPU targets with the
local cache flag already applied; `make help` lists every target. Targets that
run the CLI always use the freshly built `zig-out/bin/nuclis`.

The root build installs the executable under `zig-out/bin/` and tests the
executable, the inference module, and the download package. Default tests
must not initialize a GPU, load full model weights, or reach the network.

`zig build run -- inspect --json` runs nuclis directly.
In a sandbox that cannot write Zig's default global cache, append
`--global-cache-dir /absolute/writable/cache`; local verification uses the root
`.zig-cache/global` directory.

Format changed Zig source and build files with `zig fmt`. The repository check is:

```sh
zig fmt --check build.zig build.zig.zon src/ inference/ huggingface/
```

Explicit Metal/full-model test commands must be documented when introduced.
Verify behavior using the newly built binary.

## Versioning

The version lives in the root `build.zig.zon`, the single source of truth. The
package manifests under `inference/` and `huggingface/` mirror it (the Zig
package format requires a `.version`, but the path dependencies ignore it) and
`make release` bumps all three together, refusing if they disagree. The
executable exposes `nuclis --version`, fed from the root manifest through
build options.

- **SemVer with the 0.x convention.** Before 1.0 the minor number is the
  breaking axis: breaking changes bump the minor, compatible additions and
  fixes bump the patch.
- **0.1.0** marks the spec's v0.1 acceptance (the 32K context record), measured
  in [reference/bench.md § Acceptance runs](reference/bench.md#acceptance-runs).
  The tree stays on `0.1.0-dev` until that release is tagged, then returns to
  `0.2.0-dev` in the commit after the tag.
- **A tag that exists on the remote never moves.** Once a release is published
  the rule is absolute: the fix is the next number, never a moved tag.
- **Release recipe.** `make release` runs the mechanical steps; `make release
  DRY_RUN=1` previews them without touching anything. The version is derived
  by stripping `-dev` from `build.zig.zon`, never passed in: the manifest is
  the source of truth, and `release.yml` rejects a tag that disagrees with it.
  It runs `make check`, writes the release's section into `CHANGELOG.md` from
  Conventional Commits, commits `chore(release): vX.Y.Z`, creates the
  annotated tag, then commits the next `X.(Y+1).0-dev`. It never pushes.
  1. `make compare` passes on the tree (needs the pinned model); commit any
     fix that turns up.
  2. `make release`.
  3. Push the branch and the tag to publish. Tags are annotated; sign them
     (`git tag -s`) where a signing key is configured.
- Tags move only forward; never add a tag retroactively to an earlier commit.
- Benchmark records cite the git revision, and the release tag once one
  exists.
- `CHANGELOG.md` is assembled from Conventional Commits
  starting at the first tag: `feat` → minor, `fix` → patch,
  `!`/`BREAKING CHANGE` flagged as breaking. `make changelog` writes the
  section since the previous tag locally before tagging.
- **The Zig toolchain is a separate axis.** `minimum_zig_version` pins source
  compatibility, the exact compiler is recorded in benchmark records, and Zig
  upgrades are their own unit of work.

## Commits, progress, and code explanations

Follow the commit convention in [../AGENTS.md](../AGENTS.md). As coherent
increments pass their acceptance checks, record the outcome in
[engineering-log.md](engineering-log.md) and remove the
unit from [../TODO.md](../TODO.md); keep tests, relevant documentation, and
that tracker update in the code commit.

The project is also intended to deepen the user's understanding of Zig. Add
module-level explanations of data flow and ownership, document public interfaces,
and explain subtle invariants beside the code that enforces them. Avoid comments
that merely translate obvious statements into English. Write implementation
walkthroughs as modules become stable, grounded in the working code.

## Testing

- Colocate Zig unit tests with the source they exercise.
- Use `std.testing.allocator` to catch leaks, including error-path leaks.
- Test through module interfaces. Use small CPU reference operations to verify
  GPU numerical work; do not require full-model runs for each source edit.
- Keep parsing, agent state, and memory-planning decisions deterministic and
  independent of I/O.
- Keep model and GPU tests opt-in, with an explicit external model path.
  Missing required hardware/artifacts must be reported as a skip or unmet
  prerequisite, never as a passing full-model check.
- Verify Objective-C and GPU resource lifetimes separately: Zig's testing
  allocator does not observe all platform allocations.

Use focused checks while implementing. Expand validation for shared-module
changes and unresolved concerns. Documentation-only changes need link and
consistency checks rather than unrelated code tests.

## Measurements and artifacts

Benchmark equivalent inputs and settings. Record toolchain and engine revisions,
hardware, model hash, context length, output length, precision, and warm/cold
conditions. Separate nuclis load time, prompt processing, and generated-token
latency. Preserve raw measurements outside the source tree unless deliberately
publishing a small, reviewed benchmark report; recorded artifacts live under
[benchmarks/](benchmarks/). Comparable runs against the reference use its
exact token arrays (`bench --prompt-tokens`, `scripts/nuclis-baseline.py`),
never a re-tokenized rendering of them; `nuclis tokenize` shows what a text
prompt becomes before a model runs.

Store model weights and large datasets outside the repository. The initial
download and checksum are in [spec.md](spec.md#target-model-and-download).
Default build and test commands must not fetch them.

Secrets are environment-only and excluded from logs and fixtures. Real session
data and private source snippets must not become test or benchmark fixtures.

## Reference implementations and third-party material

llama.cpp and other engines are references: read them to understand a format or
algorithm, run them to produce pinned fixtures and comparison traces, and measure
them as baselines. Do not copy their code into nuclis. New kernels and decoders
are written from the format contract and our own CPU references, then proven
equivalent by fixtures (bit-exact where the operation is exact, stated tolerances
otherwise). Outputs of running a reference are ours to commit as fixtures with
provenance; its source is not. Constants that define a format (codebooks,
lookup tables) and any other third-party material that must be in the tree are
listed in [../THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md), which links
license texts on the web. Do not commit license text files.

## Documentation conventions

- Keep product requirements and acceptance criteria in [spec.md](spec.md); the
  the active plan lives in [../TODO.md](../TODO.md) and closed units in
  [engineering-log.md](engineering-log.md).
- Add supporting architecture decisions, benchmark reports, and operating guides
  under `docs/` when there is concrete information to record.
- Link new documents from [README.md](README.md).
- Label proposed behavior, implemented behavior, and measured results accurately.
- Update existing authoritative guidance instead of copying it into a new file.
- Use relative repository links and remove stale links when moving documents.
