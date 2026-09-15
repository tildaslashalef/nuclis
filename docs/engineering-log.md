# Engineering log

Dated outcomes of closed units, with the evidence that closed them. Each entry
is Outcome / Evidence / Files / Remaining — what now exists, the measurements
and checks that closed it, the paths that carry the change, and what was left
undone. The active plan lives in [../TODO.md](../TODO.md); requirements
stay in [spec.md](spec.md); the design each unit was built to lives in the
reference document its entry links. Entries are appended when a unit closes and
never rewritten, and numbers are as measured on the stated workload (see
[bench.md](reference/bench.md)).

| Unit | Title | Closed |
| --- | --- | --- |
| KERN-01 | Specialized matvec for Q4_K, Q5_K, Q6_K, IQ4_XS | 2026-09-07 |
| KERN-02 | Per-kernel GPU profile, `Observer` check/layer split | 2026-09-07 |
| KERN-03 | Specialized matvec for Q3_K, IQ3_S (IQ4_NL deferred) | 2026-09-08 |
| KERN-04 | Merged projections via a segment table | 2026-09-08 |
| KERN-05 | Per-block cost of the specialized kernels | 2026-09-08 |
| APPS-01 | `--think` reasoning effort | 2026-09-08 |
| APPS-02 | `nuclis chat` playground | 2026-09-08 |
| REPO-01 | Repository restructure: standalone `src/` + `inference/` | 2026-09-08 |
| REPO-02 | Reference oracle promoted to committed `tests/fixtures/` | 2026-09-08 |
| REPO-03 | Version wiring: `nuclis --version` from the manifest | 2026-09-08 |
| KERN-06 | GPU partial top-k for sampled decoding | 2026-09-08 |
| ENGN-01 | Library engine API: the engine seam moved into `inference` | 2026-09-08 |
| ENGN-02 | Chunked prefill with the batched matmul | 2026-09-08 |
| MODL-01 | Qwen3.8 sampling profiles, `min_p`, penalties | 2026-09-08 |
| ENGN-03 | Causal tiled attention for prefill | 2026-09-09 |
| ENGN-04 | Chunkwise DeltaNet (WY form) | 2026-09-09 |
| APPS-03 | Engine configuration `~/.nuclis/nuclis.json` | 2026-09-09 |
| ENGN-05 | Matmul tile ceiling: specialized half-operand 64×64 tiles | 2026-09-09 |
| KERN-07 | F16 KV cache as a session layout option | 2026-09-10 |
| ENGN-06 | Session snapshot and restore | 2026-09-10 |
| KERN-08 | Flash-decoding attention | 2026-09-10 |
| ENGN-07 | 32K acceptance run and benchmark record | 2026-09-10 |
| ENGN-08 | Long-context prefill attention | 2026-09-10 |
| MODL-02 | Model download through the `huggingface` package | 2026-09-11 |
| MODL-03 | Models directory, catalogue, and registry | 2026-09-11 |
| APPS-04 | Styled command output | 2026-09-11 |
| APPS-05 | `config init` registers the catalogue; `chat` → `agent` | 2026-09-11 |
| MODL-04 | Adapter registry | 2026-09-11 |
| MODL-05 | Gemma 4 12B facts, binding, CPU reference, tokenizer | 2026-09-11 |
| MODL-06 | Gemma 4 12B Metal plan | 2026-09-11 |
| MODL-07 | Gemma 4 12B profile, catalogue, acceptance, new-model guide | 2026-09-12 |
| MODL-08 | Q4_0 path and the QAT catalogue entry | 2026-09-12 |
| TERM-01 | Agent terminal surface | 2026-09-12 |
| AGNT-01 | Completion events and the tool seam | 2026-09-13 / 2026-09-14 |
| TERM-02 | Grapheme-correct width, wrapping, and cursor motion | 2026-09-14 |
| TERM-03 | OSC 8 hyperlinks in markdown | 2026-09-14 |
| TERM-04 | Focus tracking and turn-complete notifications | 2026-09-14 |
| AGNT-02 | The agent loop: state machine, provisional parser, read tools | 2026-09-14 |
| AGNT-03 | Read tools and `bash` | 2026-09-14 |
| AGNT-04 | Mutations and diffs | 2026-09-14 |
| AGNT-05 | Qwen tool rendering and pinned fixtures | 2026-09-14 |
| AGNT-06 | Tool-call decoding and the profile-driven loop | 2026-09-14 |
| REPO-04 | Comment and identifier hygiene | 2026-09-14 |
| AGNT-07 | Polish, resume, and the agent plan closed | 2026-09-14 |

## Context

### Starting point for the kernel work (measured 2026-09-07)

Decode on the pinned Qwen3.8-27B-UD-Q4_K_M, Apple M4 Pro, ReleaseSafe, 22-token
prompt, context 2048, greedy: **9.69 tok/s**, GPU busy ~101 ms/token, wall
~103 ms/token (GPU-bound). Sampled decoding (temperature 0.7, top-k 40, top-p
0.95) costs 19 ms/token more on the CPU. Per-token GPU budget from
`make bench-profile` ([bench.md § Per-kernel profile](reference/bench.md#per-kernel-profile)):

| Group | ms/token | Notes |
| --- | ---: | --- |
| Specialized matvec (IQ4_XS, Q5_K, Q4_K, Q6_K) | 83.7 | all at 0.8–0.9 ns per 256-value block; floor ≈ 0.5 |
| Generic matvec | 14.8 | Q3_K 6.2 + IQ3_S 3.9 at ~40 GB/s; IQ4_NL 3.0; Q8_0 1.7 (96 launch-bound 48-row dispatches) |
| Everything else | 6.6 | rmsnorm 2.7 (209 dispatches), delta 1.3, add 0.6, attention 0.5 |

1,236 dispatches per token. Prefill is one token per command buffer and runs
at decode speed. Reference: 9.66 tok/s at 512 context; 6.71 at 32K.

### Where decode time went after the merged-projection kernels (profile, 2026-09-08)

Per step, 92.49 ms attributed in profile mode (97.61 ms command-buffer time;
92.5 ms unprofiled), 932 dispatches: merged FFN gate/up 35.58 ms,
DeltaNet input projections 15.45, attention input projections 4.25;
standalone specialized matvecs 30.13, generic matvecs 1.37, other kernels
5.69. Merged entries report summed rows and bytes without a single encoding;
their work is no longer included in the per-encoding standalone totals.
KERN-04 removed 304 dispatches/step and measured 10.63 tok/s decode versus
KERN-03's 10.23. All stages and raw profiles are in
[bench.md](reference/bench.md#per-kernel-profile).

## Unit records

### KERN-01 — Specialized matvec for Q4_K, Q5_K, Q6_K, IQ4_XS (2026-09-07)

**Outcome.** The four decode-dominant weight encodings moved from the generic
`nu_matvec` to specialized kernels: 128-thread groups of four SIMD groups, four
output rows per SIMD group and eight lanes per 256-value block, factored scales
(`Σ(d·s·q − dmin·m)·x = d·s·Σ(q·x) − dmin·m·Σx`), packed-byte code assembly, and
per-block vector loads. Design and geometry in
[metal-backend.md § Specialized matvec](reference/metal-backend.md#specialized-matvec).

**Evidence.** `make bench` (64 output tokens, context 2048, three measured runs,
Apple M4 Pro, ReleaseSafe, pinned artifact) took decode from **4.93 → 8.54
tok/s**; `make compare` passed with max abs **9.2e-5**. All four kernels landed
at **0.8–0.9 ns per 256-value block** against the ~0.5 ns/block bandwidth
floor; best-of-five bandwidth figures are in
[metal-backend.md](reference/metal-backend.md#specialized-matvec).

**Files.** `inference/src/backends/metal/kernels.metal`,
`inference/src/backends/metal/root.zig`, `inference/metal-check.zig`,
`docs/reference/metal-backend.md`.

**Remaining.** Q3_K, IQ3_S, and IQ4_NL still ran through the generic kernel;
the remaining encodings and the per-block cost question were picked up by later
kernel units.

### KERN-02 — Per-kernel GPU profile, `Observer` check/layer split (2026-09-07)

**Outcome.** The Metal backend gained per-dispatch GPU timing (`nuclis bench
--profile`, one encoder per dispatch from `MTLCounterSampleBuffer`
timestamps), and the CLI observer split into a values-free `check` for Ctrl-C
and the time limit and a trace-only `layer` that forces a commit. The split
removed 64 command buffers per token with no kernel change. Method and limits
in [bench.md § Per-kernel profile](reference/bench.md#per-kernel-profile).

**Evidence.** `make bench` took decode from **8.54 → 9.69 tok/s** (prefill
10.05, first token 2188.4 ms; mean of three runs, Apple M4 Pro, ReleaseSafe).
The first profile (255 measured steps) attributed **104.8 ms** of the **109.0
ms** command-buffer time (101 ms unprofiled) over **1,236 dispatches/token**;
grouped by encoding: IQ4_XS 28.1, Q5_K 27.0, Q4_K 23.1, Q6_K 5.3, generic 14.8,
everything else 6.6 ms/token. GPU busy time was unchanged (~101 ms/token), so
decode became GPU-bound.

**Files.** `inference/src/backends/metal/root.zig`,
`inference/src/runtime/observer.zig`, `src/bench.zig`, `docs/reference/bench.md`,
`docs/reference/metal-backend.md`.

**Remaining.** The profile named the per-block instruction work as the limiter
(the bandwidth floor is ≈ 0.5 ns/block), which set the target for the
specialized-kernel work that followed.

### KERN-03 — Specialized matvec for Q3_K, IQ3_S (IQ4_NL deferred) (2026-09-08)

**Outcome.** Q3_K and IQ3_S moved off the generic kernel (~40 GB/s, 10.1
ms/token) onto a compile-time variant of the specialized four-row template,
with separate decode bodies and per-block vector loads. IQ4_NL was deferred and
stays generic; Q8_0 stays generic. Details in
[metal-backend.md § Specialized matvec](reference/metal-backend.md#specialized-matvec).

**Evidence.** Strict one-hot fixture equality and the wide-row tolerance
passed; `make check` passed 114 unit tests plus Metal fixtures; `make compare`
passed 129 files, max abs **0.0001220703125**, thresholds unchanged. `make
bench` measured **10.23 tok/s** decode and **10.63** prefill versus the recorded
9.69 decode; the profile put Q3_K at 2.29 ms/step and IQ3_S at 1.34,
**0.934–0.969 ns/block** across both model shapes, meeting the ≤ 1.0 target. See
[bench.md](reference/bench.md) and
[metal-backend.md](reference/metal-backend.md#specialized-matvec).

**Files.** `inference/src/backends/metal/kernels.metal`,
`inference/src/backends/metal/root.zig`, `inference/metal-check.zig`,
`docs/reference/bench.md`, `docs/reference/metal-backend.md`.

**Remaining.** IQ4_NL stayed on the generic path; no session or step contract
changed.

### KERN-04 — Merged projections via a segment table (2026-09-08)

**Outcome.** Three projection groups now dispatch once per layer instead of
once per matrix: FFN `gate+up` with the `silu(g)·u` epilogue, DeltaNet
`qkv+z+β+α`, and attention `qg+k+v`. A `MatvecSegments` parameter struct selects
up to four segments by a uniform prefix scan and branches uniformly to the
existing bodies; a compact buffer table resolves the binding slots. Design in
[metal-backend.md § Merged projections](reference/metal-backend.md#merged-projections-kern-04).

**Evidence.** `make check` passed 114 unit tests and Metal fixtures, including
exact merged/separate equality across every fixture encoding, generic
fallbacks, SiLU pairs, packed and separate outputs, rejection paths, and
profile byte totals. `make compare` passed 129 files, max abs
**0.0001220703125**; `make test-generation-metal` passed
isolation/reset/cancellation. `make bench` measured **11.02 tok/s** prefill and
**10.63** decode (versus 10.23 recorded); profile: **932.01 dispatches/step**
over 255 steps, **92.49 ms** attributed and **97.61 ms** command-buffer. Raw
profile: [nuclis-profile-c12-2026-09-08.json](benchmarks/nuclis-profile-c12-2026-09-08.json);
[bench.md](reference/bench.md).

**Files.** `inference/src/backends/metal/kernels.metal`,
`inference/src/backends/metal/root.zig`,
`inference/src/models/qwen35_metal.zig`, `inference/metal-check.zig`,
`docs/reference/metal-backend.md`,
`docs/benchmarks/nuclis-profile-c12-2026-09-08.json`.

**Remaining.** The initial sequential FFN pair regressed 10.37 → 10.24 tok/s;
separate gate/up SIMD groups measured 10.49 and were kept. Reduction and SiLU
arithmetic unchanged; no session layout or public step contract changed.

### KERN-05 — Per-block cost of the specialized kernels (2026-09-08)

**Outcome.** A time-boxed investigation of why the specialized matvec kernels
spend 0.8–0.9 ns per 256-value block. Five hypotheses were each measured with
fixtures → `make bench-kernels` → `make bench` → `make compare`; none moved the
needle and none was kept, so the kernels are byte-identical to KERN-04.
Negatives and the mechanisms in
[metal-backend.md § Per-block cost](reference/metal-backend.md#kern-05--per-block-cost-research-2026-09-08-closed-without-a-kernel-change).

**Evidence.** Against Q4_K baselines of **179 / 158 / 180 / 152 GB/s**: (1) two
blocks per lane iteration **−35 %** (register footprint); (2) threadgroup input
staging **−7 to −9 %** on the 5,120-column shapes; (3) two rows per SIMD group
under 8,192 rows **−6 %** on 5,120×17,408; (4) pair-shuffled scale decode
**+3.3 / +5.6 / +4.0 / +3.9 %** over four A/B rounds, below the 5 % bar on three
shapes and reverted, while the one-group-per-lane form was **−12 %**; (5)
`MTLMathModeFast` (diagnostic only) changed nothing except Q5_K **−12 %**. Gate
at close: `make check` 116 tests plus Metal fixtures; `make compare` 129 files,
max abs **0.0001220703125**; `make bench` **10.64 tok/s** decode (unchanged).
The session established that the limiter is register footprint and instruction
scheduling, not ALU count, input traffic, or contraction; `make trace` runs
`xctrace` from the command line via `DEVELOPER_DIR`, but the GPU limiter profile
is unsupported on this device and exports empty tables.

**Files.** `inference/src/backends/metal/kernels.metal`,
`inference/src/backends/metal/root.zig`, `inference/metal-check.zig`, `Makefile`,
`AGENTS.md`, `docs/reference/metal-backend.md`.

**Remaining.** The bench harness gained the real 5,120×17,408 shape,
64-dispatch batches for small matrices, and a Q4_K exact-fixture check; one
wider-reduction geometry remains untried.

### APPS-01 — `--think` reasoning effort (2026-09-08)

**Outcome.** `--think off|low|medium|xhigh` on `generate` and `chat` sets
`generate.Options.effort`; `Engine.render(messages, effort)` renders whole
conversations and `prompt()` wraps it for one turn. `generate` keeps writing the
raw stream (including `<think>` tags) to stdout; presentation is the chat's job.
No JSON `reasoning`/`content` split was added.

**Evidence.** CLI parse tests cover the flag on both commands and its rejection
on `bench`.

**Files.** `src/cli.zig`, `src/generate.zig`, `src/chat/view.zig`,
`inference/src/engine.zig`.

**Remaining.** The `</think>` split lives in `src/chat/view.zig`; it can move
into the engine when `eval` needs it.

### APPS-02 — `nuclis chat` playground (2026-09-08)

**Outcome.** A terminal chat playground over `engine.Engine` replaced the
earlier proposal of a separate TUI package, golden frames, and a themed screen.
`src/chat/root.zig` runs `generate.runLoop` with `Hooks.step`/`Hooks.token`;
multi-turn reuses the session by text prefix (`increment(seen, full)`); effort
changes and cancelled turns reset and re-prefill with a visible status, and
cancelled turns stay on screen but leave the model's conversation.
`terminal.zig` owns the raw-mode lease, alternate screen, per-frame
`TIOCGWINSZ` resize, and non-TTY rejection; `view.zig` splits thinking/answer,
wraps cell-width, and strips control bytes; `keys.zig` is a pure decoder for
legacy and kitty `CSI u` sequences.

**Evidence.** `make check` 106/106 plus `test-metal`, and
`make test-generation-metal`. A manual pty session on the model prefilled 18
tokens to position 25 on turn 1 and on turn 2 prefilled only its 20 new tokens
(prefix reuse); Ctrl-D restored the terminal and echoed both turns. The spec
note is [agent-spec.md](agent-spec.md#implemented-today-nuclis-chat-apps-02).

**Files.** `src/chat/root.zig`, `src/chat/terminal.zig`, `src/chat/view.zig`,
`src/chat/keys.zig`, `src/generate.zig`.

**Remaining.** Deliberately out: scrolling, cursor editing beyond backspace,
colour theme, `/save`, paste handling, mouse.

### REPO-01 — Repository restructure: standalone `src/` + `inference/` (2026-09-08)

**Outcome.** The tree is the standalone `src/` executable plus the `inference/`
engine package, composed by the root `build.zig`/`build.zig.zon` through path
dependencies; the engine no longer lives inside the executable, and `zig-out`
and `.zig-cache` are root-only. The layout is documented in
[architecture.md](architecture.md).

**Evidence.** The gate was revalidated at the restructured head: `make check`
**116 tests**, `make compare` **129 files max abs 1.22e-4**, `make bench`
**10.61 tok/s** decode.

**Files.** `build.zig`, `build.zig.zon`, `src/`, `inference/`.

**Remaining.** The `inference` package carries no version of its own; the
executable was wired to the manifest version in a follow-up unit.

### REPO-02 — Reference oracle promoted to committed `tests/fixtures/` (2026-09-08)

**Outcome.** The llama.cpp reference traces that prove kernel equivalence were
promoted from a rebuildable local checkout to committed fixtures under
`tests/fixtures/`, with provenance recorded per artifact.

**Evidence.** [tests/fixtures/provenance.md](../tests/fixtures/provenance.md)
records each fixture's producer, revision, and artifact; the `make compare`
acceptance value is **max abs 0.0001220703125 over 129 files**.

**Files.** `tests/fixtures/`, `tests/fixtures/provenance.md`,
`scripts/reference-generation.cpp`, `scripts/reference-baseline.py`.

**Remaining.** Regenerable pipeline outputs (the llama.cpp checkout and live
runs) stay out of git under `.zig-cache/reference/`.

### REPO-03 — Version wiring: `nuclis --version` from the manifest (2026-09-08)

**Outcome.** The root `build.zig.zon` is the single source of the version;
`build.zig` reads it and passes it as a build option, so `nuclis --version` and
the package version cannot drift.

**Evidence.** `nuclis --version` prints the manifest version; the rule is
documented in [development.md § Versioning](development.md#versioning).

**Files.** `build.zig`, `build.zig.zon`, `src/cli.zig`, `src/help.zig`,
`docs/development.md`.

**Remaining.** None.

### KERN-06 — GPU partial top-k for sampled decoding (2026-09-08)

**Outcome.** Sampled decoding no longer reads back 1 MB of logits and sorts
248,320 candidates on the CPU. `nu_topk_partial` keeps the top `K ≤ 256` of each
slice by `(value desc, index asc)` and `nu_topk_final` merges them; the CPU runs
the existing `Sampler.select` on the `K` candidates plus a GPU exp-sum. The
contract stays exact through an uncertainty band (`total_band = 1e-5`, measured
GPU error 9.6e-8) that defers any decision inside it before the RNG advances,
so a fallback token is the reference token by construction. Contract in
[generation.md](reference/generation.md#sampling-on-the-gpu-without-reading-the-vocabulary-back-kern-06).

**Evidence.** Sampled decode went **8.68 → 10.69 tok/s** with bit-identical
tokens. Unit tests covered the option grid (top-k {0, 1, 40, 256, 300} × top-p
{0.5, 0.95, 1} × T {0.3, 0.7, 1.5}, three seeds, the sum perturbed by ±4e-6, 20
random vocabularies of 1,024) with zero fallbacks on the `1 ≤ top_k ≤ 256` path
and equality everywhere; `test-metal` checked the 256 pairs over the 248,320
vocabulary with ties exactly against the CPU sort, the exp-sum against F64, the
flag, and shape rejection. `bench` gained sampling flags and a fallback column:
greedy **10.77**, GPU top-k **10.69** (0.7/0.95/40) and **10.67**
(1.0/0.95/20), nucleus-only **10.51**, ineligible full readback **8.68**. Gate:
`make check` 118 tests plus Metal fixtures, `make compare` 129 files max abs
**1.22e-4**.

**Files.** `inference/src/backends/metal/kernels.metal`,
`inference/src/backends/metal/root.zig`, `inference/src/sampling/root.zig`,
`inference/src/models/qwen35_metal.zig`, `inference/src/engine.zig`,
`src/generate.zig`, `inference/metal-check.zig`,
`docs/reference/generation.md`, `docs/reference/bench.md`.

**Remaining.** The plan's 1,000-vector × 5-seed grid runs at reduced size (20 ×
3) in the default unit test to keep the suite fast; the GPU fixture covers the
full vocabulary. `--top-k 0 --top-p 1` disables the GPU path.

### ENGN-01 — Library engine API: the engine seam moved into `inference` (2026-09-08)

**Outcome.** The engine seam (`Engine`, the CPU/GPU `Model` union, `runLoop`,
`Timing`, `Outcome`, `Hooks`) moved from the executable into the library as
`inference.engine`. `src/engine.zig` became a re-export shim plus the
command-line helpers (`readPrompt`, `milliseconds`); `generate.runLoop` adapts
the executable's `Trace` to the library's observer and the new
`Hooks.before_step`, which carries the session position the trace names files
by.

**Evidence.** Greedy `generate` text was byte-identical to a pre-move capture;
sampled `--json` (seed 5) and the CPU `Hello,` raw run were identical on every
non-timing key; `make check` 118 tests plus Metal fixtures; `make compare` 129
files, max abs **0.0001220703125**; `make test-generation-metal` passed; `make
bench` unchanged within noise. See
[architecture.md § 1](architecture.md#1-what-happens-when-you-run-nuclis-generate).

**Files.** `inference/src/engine.zig`, `src/engine.zig`, `src/generate.zig`,
`src/bench.zig`.

**Remaining.** Typed events and the adapter registry were deferred to the
follow-on completion/tool-seam unit.

### ENGN-02 — Chunked prefill with the batched matmul (2026-09-08)

**Outcome.** Prefill processes the prompt in chunks of `C` tokens (256–512)
with matrix–matrix kernels for the projections and FFN:
`Plan.prefill(tokens, logits_for_last)` alongside `step`, `[C × dim]`
activation buffers, `rows = C` norms, `nu_matmul_<enc>` tiles through the
generic decoder, and batched KV writes. Mixers stayed per token inside the
chunk. Design in
[metal-backend.md § Prefill in chunks](reference/metal-backend.md#prefill-in-chunks-engn-02).

**Evidence.** Prefill went **11 → 51 tok/s** at 543 tokens (**35** on the
22-token `make bench`); chunked vs stepped max abs **3.1e-5**, relative RMS
1.1e-6, same argmax. `make bench` measured prefill **35.0** (was 11.1) and
decode **10.79**; raw prompts of 543 and 3,547 tokens prefilled at **51.3** and
**33.3** tok/s against the reference's 89 at 512 and 4,096. `make compare`
unchanged (129 files, max abs 1.22e-4). The tile kernel measured **2.3–3.1
TFLOP/s** on the FFN shapes (`make bench-matmul`).

**Files.** `inference/src/backends/metal/kernels.metal`,
`inference/src/backends/metal/root.zig`,
`inference/src/models/qwen35_metal.zig`, `inference/generation-check.zig`,
`inference/metal-check.zig`, `docs/reference/metal-backend.md`,
`docs/reference/bench.md`.

**Remaining.** Chunked-vs-incremental was covered by the 32-token chunk check
plus the 256-token production path, not three separate runs; per-encoding tile
decode specialization and the per-token attention/DeltaNet loops were left as
later levers.

### MODL-01 — Qwen3.8 sampling profiles, `min_p`, penalties (2026-09-08)

**Outcome.** The shared sampler gained `min_p`, `presence_penalty`, and
`repetition_penalty` with neutral defaults and validation, plus
`sampling.History` (a `DynamicBitSetUnmanaged` over the vocabulary) as
conversation state observed by `engine.runLoop`. The Qwen adapter's
`samplingDefaults(effort)` holds the official per-mode table and `generate`/`chat`
default to it; `bench` never applies it. Table and chain order in
[generation.md](reference/generation.md#sampling-profiles-and-the-selection-chain-modl-01).

**Evidence.** `make check` passed 125 default tests (was 118): boundary
validation, hand-computed penalty cases pinning repetition-before-presence,
`min_p` as a top-1-relative prefix after top-k and before top-p, and the GPU
grid extended with `min_p` ∈ {0, 0.05} (equality everywhere, zero fallbacks on
the exact path). `generate --json` reported exactly the table's options for
`--think off` and `--think low`; `--temperature 0 --presence-penalty 0`
reproduced the greedy tokens `Hello! How`; GPU-path vs reference-path tokens
were identical in four configurations including `--top-k 0 --top-p 1 --min-p
0.02 --temperature 1.5` (48 tokens, 0 fallbacks). Gate: `make check` 125 tests,
`make compare` 129 files max abs **1.22e-4**; `make bench` greedy **10.60**
decode / **34.70** prefill, instruct profile **8.69** (+20.7 ms/token, full
readback for the presence penalty), thinking profile **10.57** with 0 fallbacks
([bench.md](reference/bench.md#observations-so-far)).

**Files.** `inference/src/sampling/root.zig`, `inference/src/profiles/qwen38.zig`,
`inference/src/engine.zig`, `src/generate.zig`, `src/cli.zig`,
`docs/reference/generation.md`, `docs/reference/bench.md`.

**Remaining.** A GPU penalty kernel is deferred ([roadmap](roadmap.md#also-deferred));
the chat's per-turn option rebuild was reviewed, not exercised interactively.
Behavior change: `generate` was greedy by default and now samples with the
instruct profile (`--temperature 0` restores greedy).

### ENGN-03 — Causal tiled attention for prefill (2026-09-09)

**Outcome.** `nu_attention_chunk` batches a chunk of `C` queries attending
causally over `cache[0 .. position + C]` in one dispatch per layer, without a
`C × visible` score buffer: threadgroup per (query head, 32-query tile), four
SIMD groups of 8 query rows, 32-position key tiles, online softmax with a
diagonal-matrix rescale, `simdgroup_float8x8` for Q·Kᵀ and P·V, and zero-padded
tail sub-blocks. Wired into `Plan.attentionChunk`; the `step` path is
unchanged. Contract in [metal-backend.md § Kernels](reference/metal-backend.md#kernels).

**Evidence.** `test-metal`: 256 query rows at positions 1,792..2,047 over a
2,048-row cache vs the F64 CPU reference per row gave max abs **3.9e-7** (bound
1e-5); a 37-row chunk at position 5 with 1e30 poison past each row's horizon
matched the clean reference and stayed finite; single- and three-token prompts
from an empty cache passed; contract violations were refused. `make check` 135
tests; `make compare` unchanged (129 files, max abs 1.22e-4);
`make test-generation-metal` chunked vs stepped max abs **2.5e-5**, relative RMS
1.2e-6, identical argmax. `make bench` prefill **35.3**, decode **10.47**; raw
prompts: 545 tokens **51.4**, 3,657 tokens **50.6**, against the reference's
89.2/89.3 at 512/4,096 ([bench.md](reference/bench.md#observations-so-far)).

**Files.** `inference/src/backends/metal/kernels.metal`,
`inference/src/backends/metal/root.zig`,
`inference/src/models/qwen35_metal.zig`, `inference/metal-check.zig`,
`inference/generation-check.zig`, `docs/reference/metal-backend.md`.

**Remaining.** The kernel uses a threadgroup per query head (not per KV head) so
each SIMD group's 32 output accumulators stay in registers; six query heads
re-read the same key/value rows. KV is read as F32, so the F16 cache needed a
second load path.

### ENGN-04 — Chunkwise DeltaNet (WY form) (2026-09-09)

**Outcome.** The per-token `nu_delta` loop inside a prefill chunk is gone. Stage
1 added the F64 CPU reference `cpu.recurrent.deltaChunk` (cumulative log decays,
the strictly lower system `(I + A) U = B` by forward substitution, outputs and
state as chunk sums). Stage 2 added `nu_delta_chunk` behind `Backend.deltaChunk`,
a threadgroup per (value head, 32 value rows) looping over 32-token sub-chunks
with the state streamed from the session buffer; `Plan.deltaChunk` replaced the
per-token loop and `step` is unchanged. Equations in
[cpu-reference.md](reference/cpu-reference.md#chunkwise-deltanet-engn-04-stage-1) and
[metal-backend.md § Kernels](reference/metal-backend.md#kernels).

**Evidence.** `test-metal`: a 70-token chunk (sub-chunks 32, 32, 6) on the model
shape with L2-normalized q/k and NaN past the chunk gave output max abs
**3.4e-8** (bound 1e-5) and state **1.2e-7** (bound 1e-4) vs the F64 chunk
reference per head, and **1.2e-7** vs 70 sequential CPU steps per head. `make
check` 139 tests; `make compare` unchanged (129 files, max abs 1.22e-4); `make
test-generation-metal` chunked vs stepped max abs **2.8e-5**, relative RMS
1.2e-6, identical argmax. `make bench` prefill **34.9**, decode **10.33**; raw
prompts: 545 tokens **52.6**, 3,657 tokens **53.0**, 13,399 tokens at context
16,384 **42.6** with decode **4.67** (reference 74.1 / 7.32)
([bench.md](reference/bench.md#observations-so-far)).

**Files.** `inference/src/backends/cpu/recurrent.zig`,
`inference/src/backends/metal/kernels.metal`,
`inference/src/backends/metal/root.zig`,
`inference/src/models/qwen35_metal.zig`, `inference/metal-check.zig`,
`docs/reference/cpu-reference.md`, `docs/reference/metal-backend.md`.

**Remaining.** Sub-chunks are 32 tokens, not the planned 64, because the state
is streamed from device memory and the sub-chunk only sizes the threadgroup
tiles; the two-pass/half-state variants were unnecessary and not built. The
gain is small (about 2 tok/s at 4K) because once attention was batched the
per-token DeltaNet dispatches were a small share; prefill now sits on the matmul
tile ceiling.

### APPS-03 — Engine configuration `~/.nuclis/nuclis.json` (2026-09-09)

**Outcome.** `src/config.zig` is a typed configuration file with
`engine`/`generate`/`chat` sections: a comptime schema walk over
`std.json.Value`, `resolve` layering defaults < file < flags with per-key
sources, `init` (exclusive create), and `show` (text and JSON). `nuclis config
init | show [--json]`, and `generate`, `bench`, and the chat consume
`config.Resolved`. Documented in
[development.md § Configuration file](development.md#configuration-file).

**Evidence.** Missing file → defaults with no write; `init` → `show --json`
round-trips every key and the written defaults parse back to `Config{}`; unknown
key, wrong type, bad range, and wrong version each name the key (18 literal
cases plus the size bound). `generate --json` with the settings as flags
(`--ctx-size 256 --max-tokens 6 --think low --top-k 5 --seed 7`) and as file
values was identical on every non-timing key (tokens `760 1156 369 9859 728
310`); `bench` prints `Config: <path>` and took context from the file while
keeping the flagged budget and greedy sampling. Gate: `make check` 135 tests
(was 125), `make compare` 129 files max abs **1.22e-4**.

**Files.** `src/config.zig`, `src/cli.zig`, `src/paths.zig`, `src/generate.zig`,
`src/bench.zig`, `docs/development.md`.

**Remaining.** `engine.kv_precision` and the `agent` section were deferred to the
units that read them; `config show` reports `default`/`file` and the flag layer
is proven by `resolve` tests rather than `show` accepting generation flags.
Behavior changes: `generate` defaults moved to the configured engine (Metal when
built in, 2,048 tokens, 8,192 context) and the chat starts at effort `low`.

### ENGN-05 — Matmul tile ceiling: specialized half-operand 64×64 tiles (2026-09-09)

**Outcome.** `nu_matmul` became a template on encoding, tile shape, operand
types, and staging, with thirteen instantiations: per-encoding tile decoders
(`nu_tile_*`) that evaluate the generic decoder's expression in the same F32
order, half operands with F32 accumulation, and 64×64 tiles with the activation
tile staged transposed. `Backend.matmul` selects by encoding, alignment, and
chunk length, with a 32×32 set for short chunks and the generic F32 tile as
fallback. Full table in [metal-backend.md § Kernels](reference/metal-backend.md#kernels).

**Evidence.** Lever-by-lever on the Q4_K gate shape (ms per 256-token chunk):
baseline 14.71; per-encoding decoders 13.73; staged F32 activations 12.55; F32
64×32 12.57, 64×64 63.08 (threadgroup-memory collapse); half 32×32 11.24, 64×32
10.51, 64×64 9.00; half weights with F32 activations 64×32 11.02; transposed
staging 9.07 vs 9.29; predicating MACs 11.91. Final **8.77–9.37 ms**,
**4,868–5,205 GFLOP/s**. `test-metal` compared every specialized tile with the
generic F32 tile (worst **3.79e-5** of Σ|w·x|, bound 2e-4); `make check` 139
tests; `make compare` unchanged (129 files, max abs 1.22e-4);
`make test-generation-metal` at chunks 64 / 48 / 32 gave max abs **2.37e-3 /
2.32e-3 / 2.59e-3**, relative RMS 1.1–1.2e-4, identical argmax. `make bench`
prefill **39.1** (was 34.9), decode **10.30**; raw prompts: 545 tokens **83.8**
(was 52.6; reference 89.2), 3,657 tokens **81.7** (was 53.0; reference 89.3)
([bench.md](reference/bench.md#observations-so-far)).

**Files.** `inference/src/backends/metal/kernels.metal`,
`inference/src/backends/metal/root.zig`, `inference/generation-check.zig`,
`inference/metal-check.zig`, `docs/reference/metal-backend.md`,
`docs/reference/bench.md`.

**Remaining.** The reference was approached, not matched: within 6–9 % at 512
and 4K. Half activations assume |x| < 65,504, checked only through the
generation check on real prompts; the pinned fixtures' extreme F16 block scales
are tamed to 2^-6 in the tiled matrices. A split-K or bandwidth-bound small-M
design for short prompts (22 tokens take 563 ms where one token through the
matvec takes 95 ms) is a follow-up ([roadmap](roadmap.md#also-deferred)).

### KERN-07 — F16 KV cache as a session layout option (2026-09-10)

**Outcome.** Two commits. `Session.memory` became a page-aligned byte block with
`Rows` views (`range` for GPU bindings, `floats` for the CPU reference), regions
on 16-byte boundaries; and `Layout.attention.precision` (`f32` | `f16`) landed
with `--kv`, `engine.kv_precision` (default `f16`), decode attention templated
on the cache type, `nu_pack_half`, and a half chunk kernel. `make compare` split
into `compare-f32` (unchanged thresholds) and `compare-f16` (its own tolerance).
Design in [metal-backend.md § F16 KV cache](reference/metal-backend.md#f16-kv-cache-kern-07).

**Evidence.** `test-metal`: packed halves exact; half decode attention at 1,021
visible **1.1e-8** against the CPU over the rounded cache (bound 2e-5) and
**1.2e-5** max abs / 2.1e-4 relative RMS against the CPU over the original
floats; half chunk attention on the causal-tiled cases **1.8e-4** max abs (bound
1e-3). `make check` 140 tests (was 139); `make compare-f32` 129 files max abs
**1.22e-4**; `make compare-f16` 33 of 129 layer files exceed the bring-up 2e-3
and 11 exceed 1e-4 relative RMS (max **2.5e-2** at layer 59 of token 1,
**1.9e-4** at layer 51 of token 0), logits **9.2e-4 / 5.3e-5**, greedy **353**
unchanged; F16 got its own tolerance (**3e-2 / 2e-4 on layer files**). `make
test-generation-metal`: F16 stepped vs F32 stepped after 70 tokens **4.3e-4**
max abs / 2.3e-5 relative RMS, F16 chunked **2.6e-3 / 1.3e-4**, argmax
identical. Session bytes at 32,768: **2,304,376,832** (`f16`) vs **4,451,860,480**
(`f32`). Measured (Apple M4 Pro, ReleaseSafe, pinned artifact, greedy, flags
explicit): 22 tokens / 2,048 context f16 **39.7 / 10.59 / 554 ms** and f32
**39.7 / 10.62 / 554**; 3,657 / 4,096 f16 **83.1 / 8.88 / 44,020** and f32
**82.0 / 8.55 / 44,611**; 13,399 / 16,384 f16 **67.3 / 5.59 / 198,978** and f32
**61.3 / 4.58 / 218,720**. Reference (llama.cpp, F16 KV): **89.3 / 9.21** at
4,096 and **74.1 / 7.32** at 16,384. The F16 cache is worth **+22 %** decode and
**+10 %** prefill after the 13K prompt, **+4 % / +1 %** after 3.6K, and nothing
at 2K; the remaining gap at 16K (5.59 vs 7.32 decode) is the three-dispatch
decode attention over the score buffer that KERN-08 replaces. Prefill at 13K was
42.6 before the matmul tiles landed and 61.3 with the F32 cache at this point.
Session bytes: **278 / 406 MiB** at 2,048, **406 / 662** at 4,096, **1,174 /
2,198** at 16,384 (f16 / f32)
([bench.md](reference/bench.md#observations-so-far)).

**Files.** `inference/src/runtime/session.zig`,
`inference/src/backends/metal/kernels.metal`,
`inference/src/backends/metal/root.zig`,
`inference/src/models/qwen35_metal.zig`, `inference/src/engine.zig`,
`src/config.zig`, `src/cli.zig`, `inference/metal-check.zig`,
`docs/reference/generation.md`, `docs/reference/metal-backend.md`.

**Remaining.** No F16-output matvec variant; the decode kernels still made three
dispatches over a score buffer (replaced by KERN-08); the CPU reference has no
F16 mode. Half assumes |k|, |v| < 65,504, checked through the generation check
and the pinned prompts only.

### ENGN-06 — Session snapshot and restore (2026-09-10)

**Outcome.** `Session.snapshot(gpa)` copies the used extent in layer order
(attention rows `[0, position)` of keys then values at the layout's precision;
recurrent history and matrix whole) and `Session.restore` requires a matching
capacity, `layout_digest` (a Wyhash of every layout field and the capacity), and
byte count, else `SnapshotMismatch` with the session untouched.
`engine.Model.snapshot`/`restore` dispatch to either executor. The new reference
is [session.md](reference/session.md).

**Evidence.** `make test-generation` and `make test-generation-metal`: step
token 1, snapshot (**157,024,256 bytes**), step token 2 → A; restore, step token
2 → B; A == B bit for bit; the third step after restore equals a
never-snapshotted session's; a capacity-8 session refuses the capacity-4
snapshot and keeps position 0; a poisoned session refuses to snapshot until
reset. `make check` 142 tests (round trip under `checkAllAllocationFailures`,
used-extent size, unused row preserved, capacity and precision mismatch, unready
sessions, a truncated snapshot). No kernel or plan change: `make compare` and
`make bench` not rerun.

**Files.** `inference/src/runtime/session.zig`, `inference/src/engine.zig`,
`inference/src/models/qwen35_metal.zig`,
`inference/src/models/qwen35_runtime.zig`, `docs/reference/session.md`.

**Remaining.** The snapshot is plain `[]u8`, not page-aligned `f32`; the chat is
not wired to it (that lands with the agent's turn-boundary checkpoint).

### KERN-08 — Flash-decoding attention (2026-09-10)

**Outcome.** The three-dispatch decode attention (scores over a
`[heads][visible]` buffer, softmax, values) was replaced by
`nu_attention_decode` (F32 and half cache instantiations) plus
`nu_attention_merge`: grid (KV head, split), `splits = min(64, ceil(visible /
256))`, four SIMD groups per threadgroup with running max/sum/accumulator in
registers, an 8 KB cross-SIMD stage, and a merge that rescales the splits per
head. The plan's 3 MiB scores buffer became a 1.6 MB partials buffer. The
three-pass kernels stay as `test-metal`'s oracle.

**Evidence.** `test-metal`: the six pinned fixtures within **1e-5**; the model
shape at 257, 1,021, 16,385, and 32,000 rows against the F64 CPU reference gave
F32 cache **2.2e-8 / 1.2e-8 / 7.5e-9 / 4.7e-9** and F16 cache over the rounded
rows **1.9e-8 / 1.3e-8 / 5.1e-9 / 4.2e-9** (bounds 2e-5 and 1e-4); shape
rejections passed. `make check` 142 tests; `make compare-f32` 129 files max abs
**6.1e-5** (was 1.22e-4); `make compare-f16` **2.50e-2 / 1.9e-4**, greedy
unchanged. `make test-generation-metal` unchanged. Measured at 30,650 tokens /
32,768 context: decode **2.65 (377 ms) → 8.09 (124 ms)**, prefill **49.2 →
51.0**; at 13,399 / 16,384 decode **5.59 → 8.92**, prefill **67.3 → 66.5**
(reference 74.1 / 7.32); at 22 / 2,048 decode **10.59 → 10.63**, prefill
**39.7 → 40.0**. Decode at 32K is **3.05×** the three-pass kernel and above the
reference's 6.71; attention's share of a 32K step fell from about **280 ms** to
about **30 ms**. The same command run straight after 40 minutes of sustained GPU
load gave **33–35 / 9.2–9.5** on prefill and decode alike, including the prefill
path this unit does not touch, so short-workload rows are taken on a rested
machine and long runs are labelled with what preceded them
([bench.md](reference/bench.md#observations-so-far)).

**Files.** `inference/src/backends/metal/kernels.metal`,
`inference/src/backends/metal/root.zig`,
`inference/src/models/qwen35_metal.zig`, `inference/metal-check.zig`,
`src/main.zig`, `docs/reference/metal-backend.md`, `docs/reference/bench.md`.

**Remaining.** No vectorized (`float4`/`half4`) row loads and no
simdgroup-matrix variant; the kernel is above the ~10 ms a 2 GB read would take
at this GPU's bandwidth, so a vector variant is the first lever if a profile
puts it above the floor. Short-workload rows are taken on a rested machine;
sustained-load runs are labelled with what preceded them. Side fix: the CLI's
positional stdout writer became a streaming writer.

### ENGN-07 — 32K acceptance run and benchmark record (2026-09-10)

**Outcome.** The v0.1 acceptance context was demonstrated on the target Mac with
the reference methodology. `bench --prompt-tokens <json>` feeds token arrays
untokenized and reports `prompt_source`; `nuclis tokenize` prints IDs and
per-token byte offsets from the GGUF header only; `scripts/nuclis-baseline.py` /
`make baseline` run the reference workload on the committed fixture arrays; the
records are `docs/benchmarks/nuclis-2026-09-10.json` and `-cold.json`. Detail in
[bench.md § Acceptance runs](reference/bench.md#acceptance-runs).

**Evidence.** Reference token arrays through `bench --prompt-tokens`, all four
lengths on the token budget (Apple M4 Pro 48 GiB, macOS 26.6.2, Zig 0.16.0
ReleaseSafe, pinned artifact, metal, F16 KV, context 32,768, 128 outputs,
greedy; one warmup and three runs, one at 32,639; mean ± sample stdev):

| Prompt tokens | Prefill tok/s | Reference | Decode tok/s | Reference |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 90.45 ± 0.52 | 89.19 | 10.62 ± 0.02 | 9.66 |
| 4,096 | 83.70 ± 0.37 | 89.26 | 10.20 ± 0.11 | 9.21 |
| 16,384 | 62.70 ± 0.57 | 74.07 | 8.27 ± 0.04 | 7.32 |
| 32,639 | 49.55 | 67.28 | 7.55 | 6.71 |

Every sample stopped at the fixture's exact count; 32,639 + 128 = 32,767 in a
32,768 context, nothing truncated. Decode was above the reference at every
length (+10 to +13 %); prefill was at the reference at 512 and **−6 / −15 /
−26 %** at 4K / 16K / 32K. Session block
**2,304,376,832 bytes**; process peak footprint **2.76–2.84 GB**; headroom
stated as a calculation, ≈ **29 GiB**; swap fell **1,188 → 1,012 MB**. Cold
start after `purge`, 512 tokens: load **777 ms**, first token **11.79 s**
(**5.66** warm), decode **10.55**. Gate: `make check` 146 tests, fmt clean,
`test-metal` passed; `make compare` F32 129 files max abs **6.1e-5**, F16
**2.50e-2 / 1.9e-4**.

**Files.** `src/bench.zig`, `src/tokenize.zig`, `inference/src/engine.zig`,
`scripts/nuclis-baseline.py`, `docs/benchmarks/nuclis-2026-09-10.json`,
`docs/benchmarks/nuclis-2026-09-10-cold.json`, `docs/reference/bench.md`,
`docs/spec.md`.

**Remaining.** The `--trace-dir` comparison at position 8,192 was left out
(about 10 GB per side); no prefill kernel work, so the 32K prefill gap is
measured, not addressed.

### ENGN-08 — Long-context prefill attention (2026-09-10)

**Outcome.** Closed without a kernel change: the prefill chunk attention kernel
is unchanged. The first per-kernel profile of a long prefill showed the chunk
attention as the growing prefill cost, and three shared-stage variants were
measured and rejected. Detail in
[metal-backend.md § Long-context prefill attention](reference/metal-backend.md#long-context-prefill-attention-engn-08-2026-09-10-closed-without-a-kernel-change).

**Evidence.** The 16,384-token reference array with an F16 cache:
`nu_attention_chunk_h` **75.5 s of 256.5 s (29.6 %)**, the matmuls about 60 %,
DeltaNet 3.6 %; **74 ms per layer and chunk**, about **0.7 TFLOP/s** against the
tiles' 5. Three shared-stage variants (threadgroup per KV head with one SIMD
group per query head; scalar then 8-wide vector staging; queries held in
registers) all passed the fixtures and ran slower (**90.4, 85.3, 96.8 s**). What
landed: the 16,384-row chunk attention fixture in both precisions (`test-metal`:
F32 **3.87e-7**, half **1.81e-4**, unchanged bounds). `make check` passes;
`make compare` and `bench` untouched by definition. Profile:
[nuclis-profile-16k-2026-09-10.json](benchmarks/nuclis-profile-16k-2026-09-10.json).

**Files.** `inference/src/backends/metal/kernels.metal`,
`inference/metal-check.zig`, `docs/reference/metal-backend.md`,
`docs/benchmarks/nuclis-profile-16k-2026-09-10.json`.

**Remaining.** The per-head cache re-reads were cache hits; the kernel is
latency-bound per SIMD group with one load per multiply, and the untried lever
is register-level reuse (value columns split across a head tile's SIMD groups,
probability tiles shared), recorded in the roadmap.

### MODL-02 — Model download through the `huggingface` package (2026-09-11)

**Outcome.** `huggingface` is a path dependency of the root build, imported by
`src/` only; its 16 offline tests run under `zig build test` (169 in all),
`zig build test-hf` runs them alone, and `zig build hf-downloader` installs the
standalone binary. `src/model.zig` adds `pull <owner/repo> [--file] [--revision]
[--role] [--force] [--json]` (resolves the revision once, pins the transfer to
the 40-character commit, verifies SHA-256, publishes atomically, then writes a
`<file>.nuclis.json` sidecar) and `ls [--json]`. Docs:
[development.md § Model download](development.md), [artifacts.md](reference/artifacts.md),
[spec.md](spec.md), [architecture.md](architecture.md).

**Evidence.** `make check` fmt clean, **169 tests**, `test-metal` passed.
Acceptance in a scratch `NUCLIS_HOME` (2026-09-11, Apple M4 Pro, ReleaseSafe
`-Dmetal=true`, one run each): no `--file` → 30 choices, `SelectionRequired`,
nothing created; Ctrl-C after 25 s of a 16.5 GB transfer → `Cancelled`,
directory empty, no temporary file; `Qwen3.8-27B-UD-Q4_K_M.gguf`
(**16,464,440,224 B**, commit `4ca72078…`) in **210.8 s** at **78 MB/s**,
SHA-256 `322e194f…` = spec, `shasum` agrees; second pull verified and reused in
**13.9 s**; `gemma-4-12b-it-UD-Q4_K_XL.gguf` (**7,366,423,360 B**, commit
`fc034cff…`) in **141 s** at **52 MB/s**, SHA-256 `90fd944d…`; roles `imatrix`,
`mmproj`, and `mtp` resolved correctly; a tampered sidecar gave
`ExistingFileMismatch` naming both digests; `model ls` listed five files with
commit and digest per row.

**Files.** `huggingface/`, `src/model.zig`, `src/main.zig`, `src/paths.zig`,
`build.zig`, `build.zig.zon`, `docs/development.md`, `docs/reference/artifacts.md`,
`docs/spec.md`, `docs/architecture.md`.

**Remaining.** No catalogue names yet (that is the models directory/catalogue
unit); `--concurrency` is not exposed (the package default of 8 measured best);
the scratch home was deleted after the record.

### MODL-03 — Models directory, catalogue, and registry (2026-09-11)

**Outcome.** `src/catalog.zig` is the only source of "supported"
(`qwen3.8-27b`: repository, file, pinned commit, digests, size, quantization,
architecture, profile, `mmproj`/`mtp` companions); `nuclis model pull <name>
[--with mmproj,mtp | --all]` verifies the Hub digest at the pinned commit
(`CatalogMismatch` otherwise); `model ls` groups by
`present`/`absent`/`mismatch`/`unverified`; `--model`/`engine.model` take a
catalogue name or a `models` registry entry with per-model overrides; and
`model inspect` reads the directory over the package's `readRange` and prints a
`supported`/`runnable`/`not runnable` verdict. Config layering became defaults <
profile < file < entry < flags with per-key sources; `config show` prints the
effective value of every key with its source. Detail in
[development.md § Configuration file](development.md#configuration-file) and
[artifacts.md](reference/artifacts.md).

**Evidence.** `make check` fmt clean, **179 tests**, `test-metal` passed. Config
tests (11) covered registry parse, precedence per model, unknown companion key,
ranges, and the effective view. `nuclis model inspect qwen3.8-27b` (Hub):
supported, directory **10,996,621 B** read in 2 requests (16 MiB), **4.0 s**
wall; the Gemma file: not runnable, no adapter for `gemma4` (48 blocks, 667
tensors, embedding 3840, context 262144, F32/Q4_K/Q5_K/Q6_K), directory
**15,824,191 B**, 2 requests. The clean-state cut-over rebuilt
`models/unsloth/Qwen3.8-27B-GGUF/` in **4 min 07 s**, main file **76.6 MB/s**,
digest = spec; `make bench` after the move **40.01 / 10.60** prefill / decode
(39.7 / 10.63 before).

**Files.** `src/catalog.zig`, `src/config.zig`, `src/model.zig`, `src/paths.zig`,
`inference/src/models/registry.zig`, `inference/src/models/qwen35.zig`,
`docs/development.md`, `docs/reference/artifacts.md`, `docs/spec.md`.

**Remaining.** `model ls` does not annotate registry entries; the chat turn was
not driven (no TTY); `bench` reads the entry's `ctx_size` and nothing else.

### APPS-04 — Styled command output (2026-09-11)

**Outcome.** `src/style.zig` adds a `Style` value (theme + enabled) passed
explicitly to every text renderer, enabled only when the stream is a terminal
and the theme's detection finds color (`NO_COLOR`, an unknown `TERM`, `--json`,
and pipes give plain bytes). The theme gained `success`, `warning`, and `label`,
and the command output (`config show`, `model ls`, `model pull`, `model
inspect`, `inspect`, `validate`, `tokenize`, `bench`, and the `error:` line) is
styled; text is padded before wrapping in escapes so columns align as before.

**Evidence.** `make check` fmt clean, **180 tests**, `test-metal` passed; the
tests pin the plain form with `Style.none` and one styled row.

**Files.** `src/style.zig`, `src/tui/theme.zig`, `src/cli.zig`, `src/help.zig`.

**Remaining.** The palette should be one module for the whole executable; it
moves with `style.zig` into `src/tui/` during the terminal-surface work.

### APPS-05 — `config init` registers the catalogue; `chat` → `agent` (2026-09-11)

**Outcome.** The configuration's `chat` section is now `agent` (`Config.Agent`,
`Command.agent`, paths `agent.think` and `agent.fold_thinking`, likewise in
registry entries), matching the agent spec's decided name ahead of the command
rename; the schema version is unchanged (same keys and defaults) and a file that
still says `chat` is rejected with `unknown key chat: the section is now
agent`. The command stays `nuclis chat` until the terminal surface lands.
`config init` writes every catalogue model as a registry entry and prints the
effective engine keys, each model's local status from its sidecar, and the
`nuclis model pull <name>` to run next.

**Evidence.** `make check`, **180 tests**, `test-metal` passed; an existing file
is still never touched.

**Files.** `src/config.zig`, `src/cli.zig`, `src/catalog.zig`, `docs/spec.md`,
`docs/agent-spec.md`.

**Remaining.** None.

### MODL-04 — Adapter registry (2026-09-11)

**Outcome.** The engine selects the adapter from `general.architecture` through
an explicit table: `inference/src/models/registry.zig` builds the `Adapter` enum
from the table, exposes `adapterFor`/`select` (`error.UnknownArchitecture`),
`executableEncoding`, and `validate`; `models/root.zig` composes the concrete
registry; `runtime/observer.zig` holds the shared `Observer`; `profiles/root.zig`
holds the shared profile types with digest-based lookup; and `engine.zig`
dispatches `Executor(Family)`/`Executors` with `Engine.open` selecting once and
`Engine.profile: ?Profile` replacing `template_supported`. Under `src/`,
`validate`, `inspect`, `tokenize`, `generate`, `bench`, `chat`, `config`,
`catalog`, and `cli` use only the registries and `engine.Observer`.

**Evidence.** `make check` **184 tests** and `test-metal` passed; `generate` on
the pinned Qwen file unchanged; `make bench` (2 repeats, F16 KV) **40.00 /
10.66** prefill / decode against the recorded 39.7 / 10.63; `validate` and
`generate` on the Gemma 4 12B file reported `UnknownArchitecture` naming
`gemma4` and the known `qwen35` (the pre-tokenizer gap recorded in
[architecture.md § 9](architecture.md#9-adding-a-model)). `nuclis
validate` JSON is schema 2 (`layer_kinds` replaces `full_attention_layers` /
`delta_net_layers`).

**Files.** `inference/src/models/registry.zig`,
`inference/src/models/root.zig`, `inference/src/models/qwen35.zig`,
`inference/src/runtime/observer.zig`, `inference/src/profiles/root.zig`,
`inference/src/engine.zig`.

**Remaining.** The profile the sampling defaults come from remains the
configuration's resolved one, while rendering uses the artifact's own, selected
by digest.

### MODL-05 — Gemma 4 12B facts, binding, CPU reference, tokenizer (2026-09-11)

**Outcome.** The Gemma 4 12B facts were read from the two pinned files with a
new independent header reader (`scripts/gguf-inventory.py`) and from the pinned
llama.cpp `7620399` source, recorded in [gemma4.md](reference/gemma4.md): 48 layers in a
period-6 pattern (five sliding, window 1024, head 256, 8 KV heads, RoPE base
1e4; one global, head 512, one KV head, base 1e6 with `rope_freqs` factors
rotating 128 of 512 dimensions), no value projection on global layers, four
raw-stored residual norms per layer plus per-head query/key norms, unscaled
attention scores, tanh GELU, a scalar output scale per layer, tied embeddings
scaled by sqrt(3840), logits soft-capped at 30, `tokenizer.ggml.model =
"gemma4"` (SPM-style BPE), template SHA-256 `845f1ee4…`. Code: the GGUF parser
retains arrays up to 64 elements; `cpu.gelu` and `cpu.rope.apply` factors;
`models/gemma4.zig` (validation, binder, tests over the 50 KB inventory
fixture); `models/gemma4_runtime.zig` (per-layer layouts and schedule, sliding
window as a cache-row slice); a Metal stub; and the `gemma4` tokenizer path.

**Evidence.** `validate` on the real file: 667 tensors, 40 sliding + 8 global;
tokenizer identical to the reference server's `/tokenize` on twelve strings; CPU
reference vs the oracle on `<bos>Hello,` at three positions gave max abs
**8.0e-5** on layers, **1.8e-4** on logits, relative RMS ≤ **8.3e-6**, same
greedy token **45518** and top-5 (thresholds 2e-3 / 1e-4); one CPU step **33.7
s**. Gate: `make check` **195 tests** and `test-metal` passed.
`tests/fixtures/gemma4-hello-comma/` holds 144 layer files + logits (3.3 MB).

**Files.** `inference/src/formats/gguf.zig`,
`inference/src/backends/cpu/root.zig`, `inference/src/backends/cpu/rope.zig`,
`inference/src/models/gemma4.zig`,
`inference/src/models/gemma4_runtime.zig`,
`inference/src/models/gemma4_metal.zig`, `inference/src/tokenizer/bpe.zig`,
`inference/src/tokenizer/encode.zig`, `scripts/gguf-inventory.py`,
`tests/fixtures/gemma4-hello-comma/`, `docs/reference/gemma4.md`.

**Remaining.** The Metal plan (MODL-06), the profile and template fixtures
(MODL-07), and the Q4_0/QAT path (MODL-08); the `--raw` BOS convention was
deferred.

### MODL-06 — Gemma 4 12B Metal plan (2026-09-11)

**Outcome.** `models/gemma4_metal.zig` encodes the runtime schedule (step and
chunked prefill, GPU argmax/top-k, per-layer observer, snapshot-compatible
session, sliding window as a cache-row slice on decode and a mask on prefill,
two RoPE tables with the checkpoint's factors, and the raw key projection copied
as the value on global layers). The backend gained `nu_gelu_mul`, `nu_scale`,
`nu_add_scale`, `nu_softcap`, a `gelu_mul_pair` segment mode, a clamped
`nu_tanh`, `ropeTable` with per-pair factors, `nu_attention_decode_t<KV, GROUP,
CH>` with the `_w`/`_wh` pair, a striding `nu_attention_merge`, and a
window-masked `nu_attention_chunk` with value-column splits. `Backend.unwrap`
fixed a cached-by-address wrapper bug surfaced by the reordered generation
check.

**Evidence.** `test-metal`: windowed/wide chunk attention **3.0e-6 F32** and
**1.9e-4 F16** over rounded operands (bounds 1e-5 / 2e-3), wide and grouped
decode **2.4e-7** (bound 5e-5), factored RoPE **2e-6** relative with unrotated
pairs exact, epilogues within **2e-6** or exact, GELU pair exact; Qwen fixtures
unchanged. `make compare`: Qwen f32 **6.1e-5**, f16 **2.50e-2 / 1.9e-4**. `make
compare-gemma4`: CPU **1.8e-4 / 8.3e-6**; Metal F32 **3.4e-4 / 1.5e-5**; Metal
F16 **0.73 / 3.2e-2** at its own tolerance (1.0 / 0.05), greedy **45518** and
top-5 identical on all three. `make check`: **196 tests**. First-look rates:
Gemma **199.7 / 20.25** tok/s at 512+128 (F16), **199.0 / 20.20** (F32), **36.8
/ 21.2** on the 9-token raw prompt; `llama-bench 7620399` on the same file
**219.9 / 24.9**; Qwen `make bench` **40.27 / 10.80**.

**Files.** `inference/src/models/gemma4_metal.zig`,
`inference/src/backends/metal/kernels.metal`,
`inference/src/backends/metal/root.zig`, `inference/generation-check.zig`,
`Makefile`.

**Remaining.** A ring layout for the windowed caches (11.3 GB of a 32K F16
session; roadmap), Gemma performance work, the acceptance record (MODL-07), and
the Q4_0/QAT path (MODL-08).

### MODL-07 — Gemma 4 12B profile, catalogue, acceptance, new-model guide (2026-09-12)

**Outcome.** `profiles/gemma4.zig` renders the text-only subset of the GGUF
template (`<bos>`, `<|turn>role\n…<turn|>\n` turns, a system turn carrying
`<|think|>` when thinking is on, `developer` as `system`, merged consecutive
assistants, reasoning never in history; `samplingDefaults` 1.0 / 0.95 / 64 in
every mode; `stop_tokens` `<turn|>`, `<eos>`; `reasoning` `<|channel>thought\n`
… `<channel|>`), registered as `profiles.Profile.gemma4`. The profile surface
gained `stop_tokens` and `reasoning`, `Engine.open` resolves the stop set to
ids, and a catalogue entry `gemma-4-12b` plus the new-model guide landed.

**Evidence.** Fixtures captured from `llama-server 7620399`: Gemma 14 prompts
and 20 token strings, Qwen re-captured with the same seven conversations (28
prompts, byte-identical rendering). `make check` **203 tests`;
`make test-vocabulary` matched every captured prompt and string (Gemma 262,144
tokens / 514,906 merges). Live: `generate --model gemma-4-12b` greedy ended on
`<turn|>`; a two-turn `chat` under `expect` folded "Thought for 29.1s" and
answered 144 then 145. Acceptance record: prefill / decode **192.97 / 19.82** at
512, **155.31 / 18.91** at 4K, **103.04 / 14.65** at 16K, **74.04 / 14.04** at
32,639 against the reference's **209.85 / 24.51**, **200.10 / 22.57**, **153.54
/ 16.34**, **142.80 / 16.05**; session **10.5 GiB** at 32K, peak footprint
**11.8 GB** ([bench.md](reference/bench.md#gemma-4-12b-acceptance-record-modl-07-2026-09-12)).

**Files.** `inference/src/profiles/gemma4.zig`, `inference/src/profiles/root.zig`,
`inference/src/engine.zig`, `src/catalog.zig`, `src/cli.zig`,
`scripts/tokenizer-fixtures.py`, `scripts/reference-record.py`,
`docs/reference/prompt-profile.md`, `docs/reference/bench.md`.

**Remaining.** The Q4_0 path and the QAT catalogue entry (MODL-08), Gemma
performance work and the windowed-cache ring layout (roadmap), the seam test
model, and tools/media in the profile.

### MODL-08 — Q4_0 path and the QAT catalogue entry (2026-09-12)

**Outcome.** `quant.row` decodes Q4_0 (id 2: IQ4_NL's 18-byte block and nibble
order with the code itself as the value, `d · (q − 8)`); `dequant.metal` gained
`nu_dequant_q4_0`; `kernels.metal` gained `nu_matvec_q4_0` and `nu_tile_q4_0`
with the `nu_matmul_q4_0` / `_32` instantiations; `Backend.specializedMatvec`
accepts id 2 at 2-byte alignment; the catalogue names the QAT file
`gemma-4-12B-it-qat-UD-Q4_K_XL.gguf` (commit `980b060c…`, SHA-256 `90fd44e2…`,
**6,716,356,800 B**) with its companions. This closes the Gemma adapter effort:
registry, facts, CPU reference, Metal plan, profile/catalogue/acceptance guide,
and the Q4_0 path.

**Evidence.** `make check` **204 tests**; `test-metal` passed with the Q4_0
fixture row exact through the specialized matvec and the embedding kernel,
randomized rows within **4e-6**, and the half tiles against the generic F32 tile
at **3.8e-5** worst (bound 2e-4). `make compare-gemma4`: CPU **4.4e-5 / 2.6e-6**,
Metal F32 **7.7e-5 / 3.2e-6**, Metal F16 **1.3e-2 / 6.2e-4**, greedy **107** and
top-5 identical. `make bench` (Qwen) **39.37 / 10.55**, unchanged. `make
bench-kernels`: Q4_0 **228.8 / 211.7 / 229.6 / 206.5 GB/s** on the four shapes
(generic 115–120). Acceptance record on the QAT file: prefill / decode **179.45
/ 23.96** at 512, **143.25 / 20.71** at 4K, **98.67 / 17.87** at 16K, **73.04 /
16.24** at 32,639 against the reference's **224.46 / 27.69**, **216.82 / 25.69**,
**158.78 / 22.10**, **143.72 / 20.94**; session **10.5 GiB**, footprint **11.8
GB**
([bench.md](reference/bench.md#gemma-4-12b-acceptance-record-qat-file-modl-08-2026-09-12)).

**Files.** `inference/src/quant/decode.zig`,
`inference/src/backends/metal/dequant.metal`,
`inference/src/backends/metal/kernels.metal`,
`inference/src/backends/metal/root.zig`, `inference/src/models/gemma4.zig`,
`src/catalog.zig`, `Makefile`, `docs/reference/gemma4.md`,
`docs/reference/bench.md`.

**Remaining.** Gemma performance work (per-kernel profile, windowed-cache ring
layout, an F32-activation prefill tile for the QAT file's sensitivity), the seam
test model, the 26B-A4B mixture, vision, and the MTP companions.

### TERM-01 — Agent terminal surface (2026-09-12)

**Outcome.** The v0.1 agent surface landed in two sessions. `src/chat/` became
`src/tui/` (engine-free) and `src/agent/` (composition), the command became
`nuclis agent` with data root `~/.nuclis/agent/`, and the palette a named table
behind the semantic style enum. `tui/screen.zig` took the live region
(synchronized output, `DECSTBM` insertion above it, the region's bottom as
anchor); `tui/editor.zig` brought word wrap, paste chips, history, and a 128 KiB
limit. `tui/event.zig` plus `tui/transcript.zig` turn typed events into blocks,
flushing markdown block by block while an answer is written, so `src/tui/`
imports nothing from `inference`. `src/agent/session.zig` is one append-only
JSONL file per conversation under `~/.nuclis/agent/sessions/<cwd-slug>/`.
Commands `/new`, `/ctx`, `/think`, `/save`, `/help` are a table that is also the
help text and completion list, and `nuclis agent -p "<prompt>"` runs one turn
with no terminal. Details in [agent-spec.md](agent-spec.md).

**Evidence.** `make check` **291 tests**, `test-metal` passed; `make compare`
**6.1e-5 / 0.0250** unchanged; `make bench` **40.24 / 10.81** tok/s, within the
recorded band. Golden tests without a TTY cover the editor layout, the renderer,
the transcript's rows for every event kind, a repaint/insertion escape stream,
and a session round trip. Manual checklist on the QAT Gemma file (100×30 then
80×24): a 1,228-token paste arrived as one chip (`[pasted 88 lines, 4.2 KB]`),
sent as one turn whose prompt was committed in full (89 rows), with the prefill
counter stepping per chunk and an ETA from the measured rate (256/1228 ~5s, 512
~4s, 768 ~2s, 1024 ~1s); a resize mid-turn recovered; the transcript survived
`Ctrl-D`. The same prompt through `--print` produced the same answer.

**Files.** `src/tui/`, `src/agent/`, `src/cli.zig`, `docs/agent-spec.md`.

**Remaining.** `/resume` and the session picker (the resume unit), an
alternate-screen transcript overlay (in reserve, not planned), and a dismiss key
for the completion list (a lone Escape is not decodable without a timeout, so
the list closes by no longer matching).

### AGNT-01 — Completion events and the tool seam (2026-09-13 / 2026-09-14)

**Outcome.** Two sessions. Session 1 added `inference.events.Event` (thinking,
answer, reserved `tool_call`, stop with the loop's outcome) and `engine.complete`
over the unchanged numerical `runLoop`; profile decoders receive IDs and decoded
pieces, so actual reasoning control IDs change channels while ordinary tokens
spelling marker text stay literal, and UTF-8 repair stays within a channel. Both
agent surfaces use this path; `src/tui/view.zig` no longer parses model output.
Session 2 added the tool seam: `profiles.Role.tool`, `Message.tool_calls` and
`tool_call_id`, `ToolDefinition`, one shared `events.ToolCall`/`profiles.ToolCall`
type, a bounded `Limits.tools` (default 64) and a shared 1 MiB input budget, and
`profiles.validate` as the single conversation check. Both profiles accept a
structurally valid tool input at `validate` and then reject it with
`error.ToolsUnsupported` until the native syntax lands. Sources and limits in
[tool-calling.md](reference/tool-calling.md); explanation in
[agent-concepts § 2](reference/agent-concepts.md#2-why-a-streaming-parser-needs-token-boundaries).

**Evidence.** Session 1: `make check` **298 default tests** plus Metal fixtures;
`make compare` 129 files per precision, Qwen F32 max abs **6.103515625e-5** /
relative RMS **7.696976647442415e-7**, F16 **0.02496337890625 /
0.00019148817840006747**; `make compare-gemma4-f32`/`-f16` passed (145 files
each), F32 **0.00033974647521972656 / 1.4959925821807746e-5**, F16
**0.7261533737182617 / 0.031866950982630704**. Live `agent --json -p` on both
pinned families produced `OK` with no thinking (effort `off`) and `4` with
separate thinking (Qwen 68 UTF-8 bytes / 19 generated tokens; Gemma 88 bytes /
37 tokens). Session 2: `make check` **304 default tests** (up from 298) plus
Metal fixtures; the Qwen3.8 and Gemma 4 render fixtures still matched byte for
byte; `make compare` 129 files, Qwen F32 **6.103515625e-05 /
7.696976647442433e-07**, F16 **0.02496337890625 / 0.00019148817840006818**;
`make compare-gemma4-f32`/`-f16` (145 files each) F32 **0.00033974647521972656 /
1.4959925821808762e-05**, F16 **0.7261533737182617 / 0.03186695098266467**,
unchanged. The freshly built `generate` returned `OK` on both files; `agent
--json -p` produced one `answer_delta` `OK` and one EOS `turn_end` (Qwen 17
prompt tokens, Gemma 18).

**Files.** `inference/src/events.zig`, `inference/src/engine.zig`,
`inference/src/profiles/root.zig`, `inference/src/profiles/qwen38.zig`,
`inference/src/profiles/gemma4.zig`, `src/agent/stream.zig`,
`src/agent/print.zig`, `src/tui/view.zig`, `docs/reference/tool-calling.md`,
`docs/reference/prompt-profile.md`, `docs/agent-spec.md`.

**Remaining.** Native rendering and parsing remain the open tool work (AGNT-05):
render definitions into the Qwen system block, assistant calls, and tool results
in the pinned template's role; add the Gemma grammar and its handoff distinct
from EOS; capture fixtures. The shared types and `profiles.validate` are the
boundary the loop builds on.

### TERM-02 — Grapheme-correct width, wrapping, and cursor motion (2026-09-14)

**Outcome.** Text is measured and broken by Unicode grapheme cluster.
`scripts/grapheme-table.py` downloads the pinned UCD 17.0.0, checks each file's
SHA-256, and emits `src/tui/graphemes_table.zig`: 2,327 sorted intervals
carrying Grapheme_Cluster_Break, Indic_Conjunct_Break, East Asian Width `W`/`F`,
Default_Ignorable_Code_Point, and emoji/property flags. `src/tui/graphemes.zig`
is the seam: binary search, a UAX #29 boundary state machine (GB1–GB13,
including GB9c and GB11), forward/backward cluster stepping, and one documented
width policy (VS15 → 1; emoji presentation, skin-tone modifier, or a
two-indicator flag → 2; otherwise the sum of non-zero code points; East Asian
Ambiguous → 1). `view.zig` kept its public surface and now iterates clusters,
its escape skipper factored into `escapeSpan` and taught OSC as zero width.

**Evidence.** Zig 0.16.0, M4 Pro/48 GiB. `make check`: **309 default tests** (up
from 304) plus the Metal fixture suite. New tests exercise the full **766-case**
`GraphemeBreakTest.txt` (17.0.0), pinned width cases (ZWJ families, skin tone,
VS15/VS16, flags, combining marks, Hangul, CJK, zero-width), an editor test that
Left crosses a ZWJ cluster in one move and Backspace deletes it whole with the
base+mark pair, and a wrap test that keeps a cluster intact; every pre-existing
ASCII golden test is unchanged. Provenance recorded in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

**Files.** `scripts/grapheme-table.py`, `src/tui/graphemes_table.zig`,
`src/tui/graphemes.zig`, `src/tui/view.zig`, `src/tui/editor.zig`,
`src/tui/root.zig`, `src/tui/fixtures/GraphemeBreakTest.txt`,
`THIRD_PARTY_NOTICES.md`, `docs/roadmap.md`,
`docs/reference/agent-concepts.md` §4.

**Remaining.** OSC 8 hyperlinks and focus tracking/notifications followed before
the agent loop; cursor motion now never splits a cluster.

### TERM-03 — OSC 8 hyperlinks in markdown (2026-09-14)

**Outcome.** `[label](url)` renders as a clickable hyperlink when the target is
an http(s) URL. `markdown.zig` keeps the target and wraps the styled label in
OSC 8 (`ESC ] 8 ; ; url ST … ESC ] 8 ; ; ST`); a non-http target, an over-long
target, or one carrying a control byte keeps the `.link` style but is not
clickable, so a model cannot smuggle an arbitrary scheme into the terminal. The
zero-width groundwork landed with the grapheme unit: `view.escapeSpan`
recognizes OSC (`ESC ] … BEL|ST`), so `styledWidth` counts neither the sequence
nor the URL and `wrapStyled` carries it with its text across a wrap.

**Evidence.** Zig 0.16.0, M4 Pro/48 GiB, based on `8e1db40`. `make check`:
**310 default tests** (up from 309) plus the Metal fixtures. New tests assert the
OSC open/close around a styled label, that a non-http target and a target with a
control byte are not linked, and that an OSC sequence is zero width and survives
a wrap unsplit; existing markdown and wrapping tests are otherwise unchanged.

**Files.** `src/tui/markdown.zig`, `src/tui/view.zig`.

**Remaining.** None.

### TERM-04 — Focus tracking and turn-complete notifications (2026-09-14)

**Outcome.** The terminal lease requests focus in/out reports (`CSI ?1004h`,
released as `CSI ?1004l` on exit); `keys.next` decodes `CSI I` and `CSI O` as
`focus_in`/`focus_out`; the agent tracks focus and, when a turn finishes while
the window is elsewhere, writes one OSC 9 notification (`terminal.notify`). The
decision is a pure pair (`notificationsEnabled`, `shouldNotify`): off for a dumb
terminal or `NUCLIS_NO_NOTIFY` (anything but `0`), and never for a cancellation
the user asked for. SSH `ESC O` prefixes are unaffected.

**Evidence.** Zig 0.16.0, M4 Pro/48 GiB, based on `a933123`. `make check`:
**312 default tests** (up from 310) plus the Metal fixtures. New tests cover the
focus sequences (`CSI I`/`CSI O`, and that SS3 `ESC O A` stays arrow up) and the
notification decision across focused/unfocused, enabled/disabled, and
cancelled.

**Files.** `src/tui/terminal.zig`, `src/tui/keys.zig`, `src/agent/root.zig`,
`docs/agent-spec.md`.

**Remaining.** None.

### AGNT-02 — The agent loop: state machine, provisional parser, read tools (2026-09-14)

**Outcome.** A turn is now a loop over steps. `src/agent/loop.zig` owns the state
machine: a leading system message (identity, workspace, rendered tool
definitions, call format), an owned message history, a step budget (16), and the
correlation ids matching a result to its call. Each step renders through the
artifact's profile, completes, parses for calls, executes them in order, and
folds the results into the next user message. Two seams keep it free of a model
and a terminal: a `Model` completion side (`Completer` in production, a stub in
tests) and an `Events` presentation side (`send`, `record`). Tool syntax is
provisional and isolated in `src/agent/parse.zig` (Qwen3.8's
`<tool_call><function=…><parameter=…>` form, streaming-safe); because neither
profile could yet render a tools block, definitions were rendered as text and
results returned as a `.user` message wrapped in `<tool_response>`.

**Evidence.** Zig 0.16.0, M4 Pro/48 GiB. `make check`: **330 default tests** (up
from 318; 6 parser tests and 6 loop tests over the stubbed model) plus the Metal
fixture suite. Two live `--print` turns: on the pinned Qwen3.8-27B (Metal, F16
KV, context 8192, output budget 512, `--think low`) the JSON stream showed the
`tool_call` `read_file` `{"path":"AGENTS.md","offset":1,"count":1}`, its
`tool_result` `# AGENTS.md — nuclis`, then the streamed answer and an `eos`
`turn_end`; on Gemma 4 12B a `glob` call produced six `.md` paths and
`--session` recorded the turns.

**Files.** `src/agent/loop.zig`, `src/agent/parse.zig`, `src/agent/stream.zig`,
`src/agent/print.zig`, `src/agent/root.zig`, `docs/agent-spec.md`,
`docs/reference/agent-concepts.md` §5.

**Remaining.** The provisional parser lived in its own file so its later
deletion would be a one-import diff; the `Model.run` seam took rendered
messages, not tools, because the profiles still refused tool inputs.

### AGNT-03 — Read tools and `bash` (2026-09-14)

**Outcome.** The four bounded read-only tools are complete. `grep` (native Zig)
searches the workspace for a literal, case-sensitive string, returns `path:line:
text` sorted, skips hidden entries, symlinks, binary-looking files, and a
generated-tree list (`node_modules`, `target`, `zig-out`, `vendor`, `dist`,
`__pycache__`), and bounds the match count (200) and rendered bytes (1 MiB).
`bash` runs one command through `/bin/sh -c` with a 1 MiB combined-output cap
and a 300 s wall-clock cap, folds stderr into stdout (`exec 2>&1`), kills and
reaps the child on cancellation/timeout/output-bound, and gives the child a
minimal environment (`PATH`, `HOME`, `LANG`, `TERM`, `TMPDIR`). `read_file` and
`glob` had landed earlier; this unit added `grep`, `bash`, the registry entries,
and a symlink-escape boundary test.

**Evidence.** Zig 0.16.0, M4 Pro/48 GiB. `make check`: **341 default tests** (up
from 330; 3 `grep` tests, 7 `bash` tests, and 1 boundary test) plus the Metal
fixture suite. A live `--print` turn on the pinned Qwen3.8-27B (Metal, F16 KV,
context 8192, output budget 1024, `--think low`) ran `grep "pub fn readPrompt"`
→ `src/engine.zig:16` and `:43`, then `bash "wc -l src/engine.zig"` → `110`.

**Files.** `src/agent/tools/grep.zig`, `src/agent/tools/bash.zig`,
`src/agent/tools/root.zig`, `src/agent/print.zig`, `src/cli.zig`,
`docs/agent-spec.md`, `docs/reference/agent-concepts.md` §6.

**Remaining.** The interactive surface had still not been driven on a real TTY
(none in this environment); print mode exercises the same loop and tools, so the
remaining item is the manual checklist (paste, resize, Ctrl-C cancellation,
scrollback).

### AGNT-04 — Mutations and diffs (2026-09-14)

**Outcome.** `write_file` and `edit_file` complete the six-tool registry, and a
mutation renders a structured diff. `write_file` creates or replaces a UTF-8
text file up to 1 MiB by temp-file replacement, refusing oversized or non-UTF-8
input; `edit_file` replaces one exact, non-empty byte sequence and fails without
writing on zero/multiple matches, an empty `old_string`, or an identical
replacement. `src/tui/diff.zig` is one pure function producing structured rows
(line numbers, kind, text, changed span) and the unified text from the same edit
script (LCS bounded at 1,000,000 cells; changed hunks plus three context lines).
The transcript renders side by side at 96 columns and above and unified below,
with a fold past 40 rows. A tool call renders as a humanized line (`→ Read
TODO.md [offset=126, count=24]`) built by `tools.describe`. `Workspace` gained a
`tick` so `bash` polls and the interactive driver can deliver Ctrl-C through the
raw-mode path.

**Evidence.** Zig 0.16.0, M4 Pro/48 GiB. `make check`: **364 default tests** (up
from 341) plus the Metal fixture suite: 9 diff fixtures, tool-call and
side-by-side/unified goldens, a long-diff fold, 4 `write_file` tests, 3
`edit_file` tests, a registry `describe` test, and a `bash` test whose tick
cancels a `sleep 30` well under ten seconds. Live `--print` turns created
`greeting.txt` containing `hello world` and later `edit_file`d it to `goodbye
world`, with the `--json` stream showing the diff.

**Files.** `src/tui/diff.zig`, `src/tui/transcript.zig`, `src/tui/event.zig`,
`src/tui/theme.zig`, `src/tui/root.zig`, `src/agent/tools/write_file.zig`,
`src/agent/tools/edit_file.zig`, `src/agent/tools/root.zig`,
`src/agent/tools/bash.zig`, `src/agent/loop.zig`, `src/agent/root.zig`,
`docs/agent-spec.md`, `docs/reference/agent-concepts.md` §6–7.

**Remaining.** The interactive tick has not been driven on a real TTY; the
manual checklist is carried forward. Decoding the native call syntax was still
the provisional in-app parser (AGNT-06 moved it into the profile).

### AGNT-05 — Qwen tool rendering and pinned fixtures (2026-09-14)

**Outcome.** The Qwen3.8 profile now renders the artifact's native tool path.
With tool definitions, the system turn becomes the pinned tools block: the
mode's reasoning instruction, `# Tools`, one OpenAI-shaped declaration per tool
inside `<tools>…</tools>`, the template's format reminder, then any merged
system text; declarations and non-string values use the reference's `tojson`
style. Assistant calls render as `<tool_call>\n<function=NAME>` with one
`<parameter=KEY>` block per argument, and consecutive tool results fold into one
`<|im_start|>user` turn of `<tool_response>` blocks. Assistant content carrying
a control marker is rejected. The text-only path is byte-for-byte unchanged.
Gemma 4's grammar and result handoff keep `error.ToolsUnsupported` until their
own fixtures land.

**Evidence.** Zig 0.16.0, M4 Pro/48 GiB. `make check`: **365 default tests** (up
from 364; the new fixture test asserts all **20 tool prompts** byte for byte, and
the profile-registry test expects Qwen to render and Gemma to reject) plus the
Metal fixture suite. The fixtures were captured on 2026-09-14 with llama.cpp
`7620399f58aebfd2196b74021f9581bcf7218cb9` against the pinned Qwen3.8-27B file
(template digest and full model checksum verified), one set per effort.

**Files.** `inference/src/profiles/qwen38.zig`,
`inference/src/profiles/gemma4.zig`, `inference/src/profiles/root.zig`,
`inference/src/profiles/fixtures/qwen38-tools.json`,
`scripts/profile-tools-fixtures.py`, `docs/reference/prompt-profile.md`,
`docs/reference/tool-calling.md`.

**Remaining.** The streaming tool-call decoder, the profile-driven loop, and the
deletion of `src/agent/parse.zig` were the next unit (AGNT-06).

### AGNT-06 — Tool-call decoding and the profile-driven loop (2026-09-14)

**Outcome.** Tool calling is now model knowledge end to end and the provisional
in-app parser is gone. `profiles/stream.zig` gained an optional tool grammar
(`Markers.tool`) that recognises the bracket token ids, collects the body as
ordinary pieces, and emits one `tool_call` event on the closing token; a call
still open at EOS, a budget stop, a cancellation, or a body the parser refuses
is released as plain answer text. `qwen38.parseTool` is the inverse of the
renderer and round-trips `renderCall`. The agent loop consumes typed events,
carries assistant `tool_calls` and one `.tool` result message per call, and no
longer spells the definitions or call format in its system prompt;
`src/agent/parse.zig` is deleted and nothing under `src/` names the wire syntax.

**Evidence.** Zig 0.16.0, M4 Pro/48 GiB. `make check`: **362 default tests** (the
6 provisional-parser tests left with the file; 2 new stream tests and 1 Qwen
round-trip test) plus the Metal fixture suite. `make compare` unchanged: f32 129
files max abs **6.10e-5** (bound 2e-3), max relative RMS **7.70e-7** (bound
1e-4); f16 129 files max abs **2.50e-2** (bound 3e-2), max relative RMS
**1.91e-4** (bound 2e-4). A live `--print --json` turn on the pinned
Qwen3.8-27B (Metal, context 8192) issued `write_file greeting.txt` then
`read_file greeting.txt`, and the file on disk was `hello world`.

**Files.** `inference/src/profiles/stream.zig`,
`inference/src/profiles/qwen38.zig`, `inference/src/profiles/gemma4.zig`,
`inference/src/profiles/root.zig`, `src/agent/loop.zig`,
`src/agent/tools/root.zig`, `src/agent/parse.zig` (deleted),
`docs/reference/prompt-profile.md`, `docs/reference/tool-calling.md`,
`docs/reference/agent-concepts.md` §8.

**Remaining.** Session assistant entries still record only the answer text, not
the decoded calls, so `/resume` could not yet re-render a tool conversation
through the profile; persisting them belongs with the resume unit (AGNT-07).

### REPO-04 — Comment and identifier hygiene (2026-09-14)

**Outcome.** A code-comment standard and a documentation tidy-up, with no
behavior change. `AGENTS.md` gained a *Code comments* section: module docs are a
short orientation, declaration docs a one- or two-sentence contract, inline
comments give the non-obvious why and never the what, and no history or plan
identifiers may appear. The "Working mode" note moved teaching to chat and the
llm-guide. The sweep removed **110 unit-identifier citations across 35 Zig
files**; each was replaced by the fact it stood for or dropped where the
sentence already said it. The eight longest module docs (`profiles/gemma4`,
`tui/editor`, `tui/theme`, `tui/screen`, `tui/transcript`, `agent/root`,
`agent/session`, `model`, `config`) were cut to an orientation stating the
invariants and linking the reference document.

**Evidence.** Zig 0.16.0. `zig build test`: **362 default tests** pass, so the
comment-only rewrite changed no behavior; a scan confirmed zero unit-identifier
tokens remained in any Zig comment. `zig fmt --check` and `git diff --check`
pass.

**Files.** `AGENTS.md`, and the 35 Zig sources under `src/` and `inference/src/`.

**Remaining.** None.

### AGNT-07 — Polish, resume, and the agent plan closed (2026-09-14)

**Outcome.** The tool loop is complete and a conversation survives a restart;
the agent plan closes with it. The assistant session entry now records the
profile-decoded calls with the host correlation ids their results answer
(`loop.Step.calls`, `session.Entry.assistant.tool_calls`); a cancelled step
records no calls, because an assistant entry carrying calls whose result never
followed would not load again. `/save` renders each call and labels each result.
`/resume` and `--resume <id>` re-render a stored conversation into a fresh
session: the loader lists the workspace's sessions from their headers, rebuilds
the message list, `loop.Agent.restore` continues the host ids past the highest
restored one, and the resume driver paints the transcript (with
`transcript.closeOpen`). The engine session is reset first and the fresh file is
seeded with the saved entries. The status bar paints `step N/M`, and a test pins
the six-tool block's rendered size.

**Measurements (pinned Qwen3.8-27B, M4 Pro/48 GiB, 2026-09-14).** The tools
block for the six registered tools is **2,896 bytes**, or **720 prompt tokens**
(rendering with and without tools and tokenizing both on the artifact); a live
first turn measured **850 prompt tokens**, of which 720 are the tools block. A
representative 200-line source read is **3,999 tokens** (14,399 bytes), 400
lines **7,999 tokens**, and 800 lines **15,999 tokens** (about 20 tokens per
line); `read_file`'s cap is 2,000 lines / 1 MiB, so one full-limit result is on
the order of **40,000 tokens**, roughly five times an 8,192-token context. The
tools block is about **9%** of an 8K context before any conversation.

**Evidence.** Zig 0.16.0, M4 Pro/48 GiB. `make check`: **369 default tests**
plus the Metal fixture suite (session tool-call round trip and markdown export;
resume list/find/messages/repaint; `Agent.restore`; the `--resume` parse; the
step bar). Live on the pinned Qwen artifact (Metal, context 8192): a `--print`
turn created `greeting.txt` with `write_file` then `edit_file`d it to `hello,
world`; resuming that file with `--resume <path>` and the prompt "change it to
say 'goodbye'" produced `goodbye`; resuming a read-only session by id answered
from the restored history without re-reading. `make compare` was not run (no
kernel or plan changed).

**Files.** `src/agent/session.zig`, `src/agent/loop.zig`, `src/agent/resume.zig`,
`src/agent/root.zig`, `src/agent/print.zig`, `src/agent/commands.zig`,
`src/cli.zig`, `src/tui/status.zig`, `src/tui/transcript.zig`,
`docs/agent-spec.md`, `docs/spec.md`, `docs/reference/agent-concepts.md`.

**Remaining.** The interactive surface is still undriven on a real TTY; the
manual checklist (paste, resize, Ctrl-C, scrollback, the `/resume` picker) is
carried forward, and print mode exercises the same loop. The synthetic seam test
model the spec asks for before the seam is called stable is still owed. Session
files written by reusing a `--session` path append to the existing file, which
can duplicate the first user entry; a resume seeds a fresh file, so this only
affects reusing one print-mode path across runs.
