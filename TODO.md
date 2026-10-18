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

Theme: **nuclis builds and serves on Linux, CPU only.** Metal stays the
only GPU backend; a Linux build (`-Dmetal=false`) serves the small models
the CPU reference runs fast enough, first EmbeddingGemma 2, on the
user's homelab VM `dalal` (Ubuntu 26.04, x86_64 Ryzen 9 9955HX with
AVX-512, 2 vCPU, 4 GiB, glibc 2.43; reached with `artoo vm ssh dalal`,
files copied with `artoo vm cp`).

Next: **Linux CPU build**.

| Unit | Branch | State |
| --- | --- | --- |
| Linux CPU build | `linux-cpu-build` | in progress |

## Linux CPU build

Base: `b8b76b6`

`zig build -Dtarget=x86_64-linux-gnu -Dmetal=false` fails with three
errors today; the unit makes it build, test, and serve.

1. **libc.** `build.zig`: the `nuclis` executable's `createModule` sets
   `.link_libc = true` on every target (macOS links libSystem anyway; the
   inference module sets it only under `metal`). Fixes `std.c.recv` in
   `Peer.gone` (`src/api/http.zig`) and `std.posix.system.ioctl` in
   `Terminal.size` (`src/tui/terminal.zig`), which resolves to the raw
   `std.os.linux.ioctl` (a `usize` argument) without libc.
2. **Physical memory on Linux.** `src/api/memory.zig` `physicalMemory()`:
   `sysctlbyname("hw.memsize")` on macOS; on Linux
   `std.os.linux.sysinfo` (`totalram * mem_unit`), capped by cgroup v2
   `/sys/fs/cgroup/memory.max` when it holds a number. `defaultLimit()`
   keeps the macOS formula (physical less 16 GiB, at least 4 GiB); its
   reserve becomes `min(16 GiB, physical / 4)` so a 4 GiB machine gets
   3 GiB, not all of it. Unit tests for the formula at 4, 48 GiB.
3. **Media without bridges.** Image and audio decoding already return
   `UnsupportedImageFormat` / `UnsupportedAudioFormat` when
   `build_options.metal` is false; check the API maps them to a 4xx with
   a message, not a 500.
4. **CI.** `.github/workflows/ci.yml`: a `linux` job on `ubuntu-latest`
   that runs `zig build -Dmetal=false`, `zig build test -Dmetal=false`,
   and `./zig-out/bin/nuclis --version`, gated like `check`.
5. **Docs.** `docs/spec.md` § 1: Linux is a CPU-only build target;
   `docs/development.md`: the cross-compile command
   (`-Dtarget=x86_64-linux-gnu.2.43 -Dcpu=x86_64_v4 -Doptimize=ReleaseSafe`)
   and deploying to a Linux host.
6. **On dalal.** Copy the binary and `embeddinggemma-2` over, run
   `nuclis serve --model embeddinggemma-2`, request embeddings, and
   compare them with the Mac's CPU path (`-Dmetal=false`) on the same
   inputs: cosine ≥ 0.9999 per vector. Record the speed (texts/s) in
   `docs/models/embeddinggemma.md`.

Deferred from this unit: a Linux archive in `release.yml` and the site's
download.

## Deferred

Agreed and not scheduled; when one is picked up, it becomes a unit above.

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
