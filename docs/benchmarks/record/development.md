<!-- kind: record -->
# Development: the record

The dated measurements and decisions behind [the development guide](../../development.md), [the gates guide](../../guide/gates.md), and [the code guide](../../guide/code.md), kept as written.

## Gate costs (2026-11-13)

Measured 2026-11-13 at
`3df9497` (M4 Pro; clef not on this machine, so its gates are untimed;
the two depth gates' seconds include the prefill of a missing saved
prefix):

| tier | gates | timed | seconds | budget | slowest |
| --- | ---: | ---: | ---: | ---: | --- |
| `verify` | 52 | 42 | 411 | 480 | `drama3-roundtrip-metal` 77 s, `qwen38-generation-metal` 52 s, `qwen3-tts-0.6b-roundtrip-metal` 46 s, `drama3-metal` 36 s, `qwen3-tts-codec-metal` 26 s |
| `verify-release` | 7 | 7 | 144 | 300 | `qwen3-tts-1.7b-roundtrip-metal` 56 s, `qwen38-perplexity-full` 48 s, `gemma4-e4b-perplexity-full` 13 s, `qwen38-draft-stats` 10 s, `laya-multilingual-cpu` 9 s |
| `verify-long` | 3 | 2 | 957 | 120 (over) | `qwen38-verify-depth-32k` 696 s, `qwen38-verify-depth-16k` 261 s |
| `verify-cpu` | 29 | 18 | 1308 | — | `qwen38-dflash2-speculative-cpu` 439 s, `qwen38-generation-cpu` 268 s, `qwen38-vision-cpu` 234 s, `qwen38-speculative-cpu` 197 s, `gemma4-e4b-generation-cpu` 34 s |
| `canaries` | 17 | 15 | 47 | 60 | `qwen38-speculative-metal` 14 s, `gemma4-e4b-generation-metal` 9 s, `nemotron-asr-trace-metal` 5 s, `qwen38-vision-metal` 5 s, `gemma4-e4b-perplexity` 3 s |

## What the Metal tier cost (2026-09-29)

Measured 2026-09-29 at `acce06a`, before the fast tier (M4 Pro, `--tier verify --json`, whose per-gate `timeline` stamps every output
line): the Metal tier took 704 s after 9 s of builds. Loading is not
the cost: a model's first gate pays the cold page cache (2–8 s), every
later one maps it in 0.3–1 s. The cost is in four places:

| Where | Seconds | Why |
| --- | --- | --- |
| the CPU reference inside Metal-tier draft traces | 184 (Muse), 46 (Gemma 12B QAT), 15 (E4B), 5 (Qwen's draft block) | `--draft-trace` and `checkDraft` always run the CPU forward, at the Metal build's ReleaseSafe, then the plan; the Metal half compares against the pinned fixtures on its own |
| `checkChunkedPrefill`'s F32 tiles and `prefillRows`, 70 tokens | 25–31 per 27–30B file | the generic F32 tile kernels, slowest path by design |
| perplexity, 8 windows of 512 | 56 (Qwen), 53 (Muse), 19 (26B-A4B), 12 (E4B) | the references store the cumulative value per window (`chunk_ppl`), so fewer windows compare against the same file |
| the Gemma 12B QAT file's five gates | 74 | a third Gemma file, covering only its dimensions (below) |

## Zig 0.17 usage audit (2026-10-03)

Audited 2026-10-03 against the
[0.17.0 release notes](https://ziglang.org/download/0.17.0/release-notes.html),
when the tree migrated to 0.17.0; counts are grep results over the
tree. 0.16's adoption of the `Io` interface stands, as does the decision to
skip its concurrency layer.

- **Removed, migrated:** `b.args` (→ `addPassthruArgs`), array `**`
  (→ `@splat`, or `repeat` in `src/text.zig` for multi-byte strings),
  `@typeInfo(...).fields` (→ `field_names`/`field_types`/`field_values`),
  `Allocator.dupeZ` and `fmt.bufPrintZ` (→ the `Sentinel` variants),
  `EnumSet.initEmpty`, `ascii.indexOfIgnoreCase`, and `zig build
  --global-cache-dir` (→ `ZIG_GLOBAL_CACHE_DIR`, set by the Makefile and
  `tools/gates`).
- **Deprecated in 0.17, migrated:** `fmt.allocPrint` (→ `Allocator.print`,
  157 sites), `@intFromEnum`/`@enumFromInt` (→ `@backingInt` /
  `@fromBackingInt`, rewritten by `zig fmt`), `DynamicBitSetUnmanaged`
  (→ `bit_set.Dynamic`), `builtin.os` (→ `builtin.target.os`). The tree
  uses no other deprecated name. Deprecations are doc comments only; the
  compiler does not warn.
- **Silent semantic changes, checked:** every `@hasDecl` probe names a
  `pub` declaration on each family; every `@bitCast` is scalar; the float
  `mem.eql` calls compare distinct buffers.
- **Adopted:** `@divCeil` (40 ceiling divisions), `ArrayList.lastPtr` and
  `addOne` where they remove an index (4 sites; the rest read as well with
  `len > 0`), `--cache-poison=disallowed` passes for the default, Metal
  release, test, and check-tool graphs (no `build.zig` reads the working
  directory or calls `findProgram`; the root build reads `build.zig.zon`
  from the build root, which 0.17's configure cache tracks by itself).
- **`std.heap.SafeAllocator`:** `std.testing.allocator` and, in Debug,
  `std.process.Init.gpa`. It is thread-safe, never reuses memory, and panics
  on double frees and frees from another instance; leak reports come from
  it. It grows a block in place only while the block ends its bucket, so
  `std.testing.checkAllAllocationFailures` over it sees run-dependent
  allocation counts; the inference package checks through
  `inference.alloc_check`, which refuses in-place growth.
- **Incremental compilation, for compile errors only.** `zig build check`
  compiles the executable and the three packages' test roots without
  emitting a binary, and `make watch` runs it under `-fincremental
  --watch`. Measured 2026-11-12, Zig 0.17.1, M4 Pro, a one-line edit to
  `src/agent/root.zig`:

  | Build | Time after the edit |
  | --- | ---: |
  | `zig build` (analysis, codegen, link) | 24.3 s |
  | check-only build (`-fno-emit-bin`), not incremental | 14.3 s |
  | check-only build, `-fincremental --watch` | 0.24 s per edit (first build 23.6 s) |

  The build runner keeps the compiler process alive only under `--watch`
  (or `--webui`); without either, `-fincremental` changes nothing.
  Binaries cannot be built incrementally on aarch64-macos in 0.17.1: even
  a hello-world panics in the Mach-O linker (`TODO(MachO2): load
  archive`), with either backend. So `make check`, the tests, and the
  gates keep their full builds.
- **Not applicable:** the self-hosted aarch64 backend (Mach-O is not
  ready), `std.zon.parse` (the version line is the
  only ZON the build reads), `@SpirvType`, translate-c (no `@cImport`; the
  bridge is Objective-C through `addCSourceFile`), the Build Server
  Protocol (no tool here consumes it), `Io.Semaphore.waitTimeout` (the
  batcher's one-shot completion is an `Io.Event` with `waitTimeout`
  already), and comptime-length slice coercion (no site needs it).

Still deferred (0.17 keeps these calls; none sits in a numerical path):

| Use | Count | File | Migration |
| --- | ---: | --- | --- |
| `std.fs.path.join` / `isAbsolute` / `dirname` | 93 | 28 files across `src/`, `inference/`, `huggingface/` (most in `src/paths.zig`, `src/model.zig`, `src/decision/catalog.zig`) | `std.Io.Dir.path` equivalents |
| `std.posix.sigaction`, `Sigaction`, `SIG`, `SA.RESETHAND`, `sigemptyset` | 5 | `src/interrupt.zig` | `std.posix.system` calls, or an `Io`-level signal facility if a later release adds one |
| `std.posix.tcgetattr` / `tcsetattr` / `termios` / `poll` / `pollfd` / `winsize` / `system.ioctl` | 11 | `src/tui/terminal.zig` | `std.posix.system` for raw mode and size; `Io` polling once `Io.Evented` is usable |

The compiler is a separate versioning axis, so the upgrade is its own unit;
every benchmark record names the exact Zig used.

## The Metal default (2026-09-12)

`-Dmetal` defaults to **on for macOS on aarch64** and off everywhere else
(2026-09-12): the configuration's default backend is `metal`, so a plain
build that left it out produced a binary that failed at run time with
`MetalNotEnabled` against its own defaults. A build without the backend now
says so and names both ways out (rebuild, or `--backend cpu`).

## External oracles (2026-10-14)

- External oracles are history: the first families were brought up
  against a pinned llama.cpp build (`7620399`), with the PrismML fork
  `5d80cff` for Bonsai 2 27B, build `b11514` for EmbeddingGemma 2, and
  Python oracles for the decision model (`laya` 0.3.20), clef-flash's head,
  and EmbeddingGemma 2's semantic vectors (sentence-transformers 6.1).
  Master `529ac893e` took the last external record on 2026-10-14
  ([saluki.md](saluki.md#the-llamacpp-record-2026-10-14)); since
  then nuclis is its own reference
  ([models § 4](../../models/README.md#4-validate-against-nuclis-itself)). The
  recipes and drivers are at tag `v0.11.0` under `scripts/`; the build
  flags and dated runs are in [llama-cpp.md](../llama-cpp.md).
  The fixtures they pinned stay under `tests/fixtures/`
  ([provenance](../../../tests/fixtures/provenance.md)); `make gate
  NAME='qwen38-trace-*'` reads `tests/fixtures/reference-hello-comma`,
  which nuclis's CPU reference re-made for the re-pinned file. The
  2026-10 rates of both engines on the same token arrays are the README's
  [Results](../../../README.md#results); the method is
  [benchmarks § Acceptance runs](../README.md#acceptance-runs).
