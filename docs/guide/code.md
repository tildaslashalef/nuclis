<!-- kind: guide -->
# Writing code for nuclis

The toolchain, the build and format commands, the test rules, and the
checks that hold every input-facing parser. The gates are
[gates.md](gates.md); how a change reaches `main` is
[development.md](../development.md) and [ci.md](ci.md).

## Toolchain

The tested compiler is Zig 0.17.1; the three manifests require at least
0.17.1 (a patch release of 0.17.0,
[release notes](https://ziglang.org/download/0.17.0/release-notes.html)).
Verify standard-library calls against that toolchain's installed source and
language reference rather than assuming older Zig examples still apply.
ZLS does not work with 0.17: the release split `zig build` into a
configurer and a maker process, and ZLS waits on the new Build Server
Protocol. `zig build --print-configuration` prints the configured build
graph as ZON, the quickest way to see which steps and options exist.

Local language reference: `doc/langref.html` inside the Zig install
(`~/.local/opt/zig/stable/doc/langref.html` on the author's machine).

The Metal backend additionally needs Apple's SDK/frameworks and the
Objective-C compiler. nuclis compiles its embedded shader source through
Metal at runtime using the Command Line Tools; a standalone `metal` compiler
is needed only for an offline shader build path. Swift is not a
project build requirement.

Pass allocator and Io dependencies explicitly. Prefer a separate pure function
for decisions that can be tested without file, network, process, or GPU access.
Document ownership on returned allocations and the lifetime of borrowed data.
Use scoped cleanup for normal paths and partial-initialization errors.

`main(init: std.process.Init)` receives the allocator, environment, arguments,
and Io implementation. Pass `init.io` through file operations; keep decisions
and in-memory parsing independent of OS access. Use `std.Io.Reader`/`Writer`
and `std.Io.Dir`/`File`, checking APIs against the installed standard library.

Zig's startup supplies `Io.Threaded`. The experimental `Io.Evented` backend
selects io_uring on Linux and Dispatch on macOS; io_uring does not run on the
target Mac. Keep the injected default until measurements justify changing it.
See the installed `lib/std/start.zig`, `lib/std/Io.zig` (`Evented`), and
`lib/std/Io/Dispatch.zig`. GPU work is scheduled by the Metal backend, not
by `Io` ([engine/metal-backend.md](../engine/metal-backend.md)).

ReleaseSafe is the release configuration; ordinary `zig build` uses
Zig's Debug default. A faster optimization mode is justified only by numerical
validation and measured improvements; benchmark results must name the actual
mode used.

`zig build check` compiles the executable and the three packages' test
roots without emitting a binary, and `make watch` runs it under
`-fincremental --watch`, reporting compile errors about a quarter of a
second after each save. Binaries are not built incrementally: Zig 0.17.1's
Mach-O linker panics on an incremental build, so `make check`, the tests,
and the gates keep their full builds. `std.testing.allocator` (and, in
Debug, `std.process.Init.gpa`) is `std.heap.SafeAllocator`: thread-safe,
never reusing memory, panicking on double frees; because it grows a block
in place only at its bucket's end, exhaustive allocation-failure checks in
`inference/` go through `inference.alloc_check`, which refuses in-place
growth.

The 0.17 migration, what it adopted, and the calls it left on their
older names are in [the record](../benchmarks/record/development.md#zig-017-usage-audit-2026-10-03).

## Build and format contract

The root build provides:

```sh
zig build                   # Metal is on by default for macOS aarch64
zig build test              # src/, inference/, huggingface/, and tools/gates unit tests
zig build test-tiny -Dmetal=true  # generated models, Metal against the CPU reference
zig build -Doptimize=ReleaseSafe
zig build -Dmetal=false     # a CPU-only binary (the default off Apple Silicon)
zig build test-hf           # the huggingface package's tests alone
zig build hf-downloader     # its standalone binary, zig-out/bin/hf-downloader
zig build gates-tool        # the gate runner, zig-out/bin/nuclis-gates
zig build check             # compile errors only, no binary (make watch: incremental)
```

`-Dmetal` defaults to **on for macOS on aarch64** and off everywhere else,
because the configuration's default backend is `metal`; a binary built
without the backend says so and names both ways out (rebuild, or
`--backend cpu`). Building the
bridge does not make the default tests need a GPU: the fixtures that execute
kernels are the separate `test-metal` step.

The root `Makefile` wraps these and the explicit model/GPU targets with the
local cache flag already applied; `make help` lists every target. Targets that
run the CLI always use the freshly built `zig-out/bin/nuclis`.

The Python scripts under `scripts/` (tooling, never part of the build) follow
the root `pyproject.toml`: basedpyright in `standard` mode, the type checker
editors such as Zed run on it, and ruff with lines up to 120 columns. `make
lint-py` runs `ruff check`, `ruff format --check`, and basedpyright, each
pinned and fetched by `uvx`; `make fmt-py` formats; the `python` check of
`gates.json` runs `lint-py` whenever a script or `pyproject.toml` changes.

`make install` builds the Metal release and copies the binary to
`$(PREFIX)/bin/nuclis` (`PREFIX` defaults to `~/.local`); `make uninstall`
removes it. The binary is self-contained (the Metal shader source is
embedded and compiled at run time; models live under `~/.nuclis`). The old
file is removed before the copy, never overwritten in place: on Apple
Silicon the kernel keeps a replaced file's code signature cached and kills
the binary rewritten under it at launch. Shell completion is the user's to
install, from `nuclis completion fish|bash|zsh` (`nuclis completion --help`);
the scripts are shims that ask the binary on every Tab (`nuclis __complete`,
`src/completion.zig`), so they never need regenerating after an install.

The root build installs the executable under `zig-out/bin/` and tests the
executable, the inference module, and the download package. Default tests
must not initialize a GPU, load full model weights, or reach the network.

`zig build run -- inspect --json` runs nuclis directly.
In a sandbox that cannot write Zig's default global cache, set
`ZIG_GLOBAL_CACHE_DIR=/absolute/writable/cache` (Zig 0.17's `zig build` has no
`--global-cache-dir`); the Makefile and `tools/gates` point it at the root
`.zig-cache/global` directory.

Format changed Zig source and build files with `zig fmt`. The repository check is:

```sh
zig fmt --check build.zig build.zig.zon src/ inference/ huggingface/ tools/
```

Explicit Metal/full-model test commands must be documented when introduced.
Verify behavior using the newly built binary.

## Testing

- Colocate Zig unit tests with the source they exercise.
- Use `std.testing.allocator` to catch leaks, including error-path leaks.
  Exhaustive allocation-failure checks in `inference/` go through
  `alloc_check.checkAll` ([§ Toolchain](#toolchain)).
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

## Hostile input

Every byte from outside the process (a request, a session file, a tool's
arguments, the terminal, a model's output, a model, audio, or image file)
either becomes a value or a typed error: never a panic, an out-of-bounds
read, or a leak (AGENTS.md § Engineering rules). Two checks hold it.

**Fuzz tests.** Each parser of outside bytes has one test named
`fuzz <target>: …` beside its source, calling `std.testing.fuzz` with a seed
corpus; `make fuzz` lists them. Its body draws one byte string
(`inference.fuzz.bytes`) and lays any structure over it itself (a first
byte that picks a profile, a NUL between two parts, a nibble per token), so
a corpus entry is readable: `inference.fuzz.corpus` encodes plain strings
as the length-prefixed form the `Smith` reads. It asserts what holds for
every input, under `std.testing.allocator`: a value or a typed error, and,
where the surface promises more, that too (a WebSocket text is UTF-8, a
markdown row fits the width, a file tool reads and writes nothing outside
the workspace).

| Target | Entry point |
| --- | --- |
| `http` | `src/api/http.zig` `serve` over an in-memory connection, WebSocket upgrade included |
| `multipart` | `src/api/multipart.zig` `parse` |
| `websocket` | `src/api/websocket.zig` `Reader.next` |
| `chat` | `src/api/chat/wire.zig` `parseWith` |
| `session` | `src/agent/session.zig` `parseText` and `parseHeader`, then `src/agent/resume.zig` `conversation` and the export |
| `tools` | `read_file`, `glob`, `grep`, `write_file`, `edit_file`, `Workspace.resolve`, and `resolveTarget` in a temporary workspace with a link out of it |
| `keys` | `src/tui/keys.zig` `next` key by key, and `terminal.findCursorReport` |
| `markdown` | `src/tui/markdown.zig` `split` and `render` at widths 1 to 64 |
| `diff` | `src/tui/diff.zig` `compute` |
| `decoder` | `inference/src/profiles/stream.zig` `Decoder`, either grammar, over a 16-token vocabulary |
| `render` | every profile's `render` with arbitrary content, names, arguments, and schemas |
| `tokenizer` | `Encoder.encode` on any bytes, `bpe.decode` on any ids, the UTF-8 `Stream` |
| `gguf`, `safetensors` | `inference/src/formats/*.zig` `parse` |
| `wav`, `ppm` | `audio.decodeWavAt`, `image.decodePpm` |

`zig build test`, and so `make check`, runs each over its corpus.
`make fuzz TARGET=<name> SECONDS=60` runs one under Zig's fuzzer, which
works on macOS on Apple Silicon as on Linux, in a cache of its own
(`.zig-cache/fuzz/<name>`) where the corpus it grows carries over between
runs. A failure stops it and saves the input to
`.zig-cache/fuzz/<name>/f/crash`, in the corpus's own form: minimise it,
add it to the test's corpus (or a file under the module's `fixtures/`
through `@embedFile`), and fix the code in the same unit. To add a parser,
write its test beside it, add it to the table above, and add its file to
the check below.

**The input-safety check.** `scripts/repo/input-safety-check.py`
(`make input-safety`, part of `make check`) reads the input-facing files
its `FILES` lists (`src/api/`, the session and resume, the agent's tools,
the terminal's keys, terminal, and markdown, the formats, the stream
decoder, audio, and image) and fails on `catch unreachable`,
`unreachable`, `@intCast(`, `std.debug.assert(`, and `.?` unless the line
ends in `// checked: <where>`, naming the guard that keeps it from firing.
Tests, and declarations only tests reach, are skipped. The comment is the
review: a reason that does not hold is a bug, and a construct with no
reason is either given one or removed.

**A panic.** The root panic handler (`src/main.zig`) puts a leased terminal
back and, under `nuclis serve`, writes the request log's `server.panic` line
before the default report ([api.md § The request log](api.md#the-request-log)).
`zig build -Dpanic-hook=true` builds a binary that panics on purpose when
`NUCLIS_PANIC_HOOK` names a place (`serve`: a request to
`/v1/debug/panic`; `terminal`: once the chat surface holds the terminal),
so a check can watch both; no other build has the hooks.
