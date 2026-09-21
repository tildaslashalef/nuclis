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
| TERM-05 | Two-row tool lines: call and detail | 2026-09-15 |
| AGNT-08 | Context budget: bounded results, in-turn elision, honest failure | 2026-09-15 |
| TERM-06 | Per-step thinking blocks with their own duration | 2026-09-15 |
| ENGN-09 | Primed sessions: prefill the prefix at startup, restore it on new | 2026-09-15 |
| APPS-06 | `nuclis agent ls` and `--resume` to the newest session | 2026-09-16 |
| AGNT-09 | Gemma 4 native tool calling | 2026-09-16 |
| MODL-14 | Catalogue entries ahead of their adapters; roadmap reordered around speculation | 2026-09-17 |
| TERM-07 | Rows released by a shrinking live region are reused, not left as gaps | 2026-09-17 |
| APPS-07 | `--prompt-profile` and the registry's `profile`; the template-alias gate | 2026-09-17 |
| APPS-08 | `model pull`: a leaked path per file and a leading slash in the sidecar's file name | 2026-09-17 |
| APPS-09 | `config set`, `model pull --register`, registry names in `model ls` | 2026-09-17 |
| AGNT-11 | A truncated tool call no longer bricks the session; reopened thought channels; copied bracket pieces | 2026-09-17 |
| APPS-10 | `model ls` as one aligned grid | 2026-09-17 |
| TERM-08 | The welcome: ASCII wordmark and session facts; the model's name on the status bar | 2026-09-17 |
| KERN-09 | Expert routing and gathered expert kernels (decode and prefill) | 2026-09-17 / 2026-09-18 |
| MODL-09 | Gemma 4 26B-A4B: artifact pin, facts, adapter, CPU reference, Metal plan | 2026-09-18 (two sessions) |
| MODL-10 | Gemma 4 26B-A4B: catalogue verdict, acceptance record, agent check | 2026-09-18 |
| MODL-15 | Bonsai 2 27B accepted ahead of Muse: artifact pinned, facts read, three units planned | 2026-09-18 |
| MODL-16 | Bonsai 2 27B: oracle, facts, ternary encodings, Hadamard transform, CPU reference | 2026-09-18 |
| KERN-10 | Ternary matvec and matmul tiles, the Walsh-Hadamard kernel | 2026-09-18 |
| APPS-11 | `model pull`: a verified file whose encoding this build does not store keeps its sidecar | 2026-09-18 |
| REPO-05 | README for a public repository: project status, contributions, disclosure | 2026-09-18 |
| MODL-17 | Bonsai 2 27B: the Qwen plan on rotated weights, catalogue, acceptance | 2026-09-18 |
| AGNT-12 | An empty tool result no longer aborts the turn | 2026-09-18 |
| APPS-12 | The default context window is 16K | 2026-09-18 |
| TERM-09 | A step that answers and then calls a tool no longer holds the turn in the live region | 2026-09-18 |
| MODL-11 | Muse Glimmer 30B: artifact pin, facts, tokenizer, binding, CPU reference | 2026-09-19 |
| MODL-12 | Muse Glimmer 30B: Metal plan | 2026-09-19 |
| MODL-13 | Muse Glimmer 30B: profile (text, reasoning channel), catalogue, acceptance | 2026-09-19 |
| AGNT-10 | Muse Glimmer ATEM tool calling: rendering, decoding, fixtures | 2026-09-19 |
| ENGN-10 | Muse Glimmer decode gap: the experiments accepted into the performance theme | 2026-09-19 |
| KERN-11 | Small-chunk prefill matmul near the weight-bandwidth floor | 2026-09-19 |
| ENGN-11 | Speculative state recovery: checkpoint, rewind, truncate, recover | 2026-09-19 |
| MODL-18 | Qwen3.8 draft head: the embedded prediction block on the CPU reference and the Metal plan | 2026-09-20 |
| APPS-13 | The configuration section is `generation`, not `generate` | 2026-09-20 |
| ENGN-12 | Batched verification, speculative generation (greedy and sampled), the switch and the draft length, the benchmark record | 2026-09-20 |
| REPO-07 | The roadmap file retired; themes are agreed in session and, when architectural, recorded as ADRs on request | 2026-09-20 |

| REPO-06 | DiffusionGemma structured reads and kev research | 2026-09-20 |
| ENGN-13 | Prompt commit at the plan's chunk and the batched drafter commit | 2026-09-20 |
| KERN-12 | Multi-row matvec for 2–8 rows: 2-row routing shipped, closed below its target | 2026-09-20 (two sessions) |
| REPO-08 | Repair the multi-row benchmark controls and hand-off | 2026-09-20 |
| ENGN-14 | Recovery without the whole-stack replay: per-row recurrent checkpoints | 2026-09-20 (two sessions) |
| KERN-13 | A GPU penalty kernel: the token history applied on the device before the top-k | 2026-09-20 |
| ENGN-15 | Sampled acceptance on the GPU top-k readback | 2026-09-20 |
| KERN-14 | The wide 32×8 small-batch tile: measured, closed negative | 2026-09-20 |
| ENGN-16 | Draft proposal policy: the `p_min` early stop shipped, the adaptive length dropped | 2026-09-20 |
| KERN-15 | Split-K decode matvec for row-poor shapes: measured behind the single pass, closed negative | 2026-09-21 |
| KERN-16 | Long-context prefill attention: register-level reuse measured 2–5 % at chunk sizes, closed negative; the verify-shaped window shipped | 2026-09-21 |
| KERN-18 | Fused decode norms: −182…−192 dispatches per decode step shipped, the speed bars missed; closed below its target | 2026-09-21 |
| MODL-19 | Gemma 4 draft heads: the `gemma4-assistant` companion adapter, traces at 1.1e-4, a negative default at draft 4 (1.017× only at draft 7) | 2026-09-21 |
| MODL-20 | Muse Glimmer DFlash drafter: the companion, the CPU reference and its trace, the Metal plan, a positive verdict at 1.16–1.23× | 2026-09-21 (two sessions) |
| ENGN-17 | The speculative verdict: Qwen and Gemma off, Muse on; the full record and `bench`'s true baseline | 2026-09-21 |

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

### TERM-05 — Two-row tool lines: call and detail (2026-09-15)

**Outcome.** A tool call renders as two rows and its result text no longer
reaches the transcript. The call row is a gerund and the subject (`● Reading
README.md`, `● Listing docs/**/*.md`; `● Running command` with the command on
a detail row of its own, `└ $ git log --oneline -1`), settled from the spinner
by the result. The detail row is one sentence each tool supplies from the
numbers it already has (`Result.summary`): `lines 1 to 12 of 153 · truncated,
continue with offset=13`, `16 files`, `3 matches in 3 files`, `exit 7 · 0
lines`, `+1 −1`, `new file, 2 lines`; a clean `bash` run adds nothing. A
failed call shows its message in error style, bounded to three rows. The
`k=v` bracket list and the `params` table left the registry; `describe`
returns the call row and the optional detail. The `tool_call` event carries
`detail`, the `tool_result` event and the session's `tool_result` entry carry
`summary` (absent in older files, loaded as empty), so `--resume` and the
markdown export render the same rows. Theme glyphs `done` (`●`/`*`) and
`detail` (`└`/`\`) replace the call arrow.

**Evidence.** Zig 0.16.0, M4 Pro/48 GiB. `zig build test`: **371 default
tests** (registry: subject-first schema and the two-row `describe`; each
tool's summary; transcript: the two rows, a running bash call with its
command, a failed call bounded to three rows, an empty row skipped unless
truncated; session round trip and markdown export of `summary`; resume
replay). Live on the pinned Qwen3.8-27B (Metal, context 8192, `agent --model
qwen3.8-27b -p … --json --session`): a turn that globbed `docs/reference/*.md`
(16 files), read `README.md` lines 1 to 12 of 153, and ran `git log
--oneline -1` produced the three call/result pairs above with the expected
`summary` and `detail` fields, and the session file stored the summaries;
1,161 prompt tokens, 230 generated, prefill 24.3 s, decode 25.7 s. The
interactive surface was not driven on a real TTY in this session; the
transcript tests pin its rows.

**Files.** `src/agent/tools/{root,read_file,glob,grep,bash,edit_file,write_file}.zig`,
`src/agent/{loop,print,resume,root,session}.zig`,
`src/tui/{event,theme,transcript}.zig`, `docs/agent-spec.md`.

**Remaining.** The tool bounds are still host constants unrelated to the
context window, and a long turn still ends in a bare `ContextFull`; that is
AGNT-08 in `TODO.md`.

### AGNT-08 — Context budget: bounded results, in-turn elision, honest failure (2026-09-15)

**Outcome.** The loop now survives a window that is too small for its
results. Three layers. *Budget:* one tool result may not exceed
`resultBudget(capacity)` = an eighth of the window, never below 256 tokens;
`Agent.fit` counts it through the new `Model.count` seam (the engine's
tokenizer; the test stub uses four bytes per token), scales the byte cut by
the measured density, backs up to a line boundary (a code-point boundary when
one line is over budget), and appends `[truncated to fit the context: A of B
lines shown; continue with read_file offset=N]` (other tools: `narrow the
request for the rest`); the detail row gains `· cut to A lines for the
context`, and `read_file` reads 200 lines unless asked for more. *Elision:*
when a completion reports `ContextFull` (the check runs before anything is
fed, so a retry costs a render), `elideResults` replaces every tool result of
the turn in progress but the last two with `[result elided to fit the
context: <tool>, N lines]` in one cut, recorded as a `compaction` entry with
reason `results_elided`; only then does `dropOldestTurn` remove earlier turns
(now also recorded, reason `context_full`), and only then is the error the
caller's. *Failure:* `Completer.overflow` carries what did not fit; the
surface says `context window full: the step needed N tokens (prompt plus
output budget) of C; raise it with /ctx <n> (or --ctx-size), or start over
with /new`, and print mode puts the same in the error diagnostic. The status
bar's prefill rate is measured per step from the step's first beat and kept
beside the decode rate. The plan's estimate-before-each-step was not built:
the completion's own pre-feed check plus retry is exact and costs no prefill.

**Evidence.** Zig 0.16.0, M4 Pro/48 GiB. `zig build test`: **375 default
tests** (budget cut at a line with the offset hint; a single over-budget line
cut at a code point; elision keeps the last two results and records one
compaction; nothing to elide is the caller's error; a dropped turn records
its compaction). Live on the pinned Qwen3.8-27B (Metal, the playground
repository): at 8K, a 400-line read of `data/measurements.txt` was cut to
**45 lines** (1,440 bytes) with `offset=46`, and the model reported exactly
that; at `--ctx-size 1024` the same turn ended with `the step needed 2907
tokens (prompt plus output budget) of 1024`; at `--ctx-size 3072
--max-tokens 256`, five 80-line pages were each cut to 16–17 lines, the sixth
step overflowed, **3 results were elided**, the session replayed, and the
turn answered correctly (4,638 prompt tokens over the turn including the
replay, 476 generated). The tools block grew from 2,896 to **2,959 bytes**
with `read_file`'s new description; the pin was updated.

**Files.** `src/agent/loop.zig`, `src/agent/root.zig`, `src/agent/print.zig`,
`src/cli.zig`, `src/agent/tools/read_file.zig`, `docs/agent-spec.md`.

**Remaining.** Elision replays the whole conversation (the prefix changes at
the first stub), which at 50–90 tok/s is the visible cost of the cut; a
primed prefix (ENGN-09) removes the system-and-tools part of that replay.
Elision reads a stub's prefix to tell it from a verbatim result; a tool
result that happens to start with that text would be miscounted.

### TERM-06 — Per-step thinking blocks with their own duration (2026-09-15)

**Outcome.** Each completion step times its reasoning from its own start
(`Agent.step_started`, set in `beginStep`) and closes the block once, at the
first of: the first answer byte, or the end of a step that produced tool
calls or stopped without an answer (`endThinking`); a cancelled step keeps
the bare label, and a step without reasoning sends nothing. Before, only an
answer closed a block, so a tool-call step's header stayed at "thinking" and
the last block's time ran from the turn's start, tool runs included. The
seconds are kept as `thinking_seconds` in the session's step stats, so a
resumed session labels its blocks; older files load it as zero, which the
label renders as "Thought" without a time.

**Evidence.** Zig 0.16.0, M4 Pro/48 GiB. `zig build test`: **376 default
tests** (a scripted turn of two tool-call steps and an answer step sends three
`thinking_end` events; a step without reasoning sends none). Live on the
pinned Qwen3.8-27B (Metal, context 8192, the playground): a turn that batched
a search and a read into one step, then answered, closed two blocks at
**22.3 s** and **13.5 s**, both stored in the session file; their sum
(35.8 s) is below the turn's prefill plus decode (41.2 s). A step's time
starts at its prefill, so the label is the wait the user saw, not decode
alone.

**Files.** `src/agent/loop.zig`, `src/agent/root.zig`, `src/agent/print.zig`,
`src/agent/session.zig`, `src/agent/resume.zig`, `src/tui/transcript.zig`,
`docs/agent-spec.md`.

**Remaining.** The interactive fold label was not driven on a real TTY in this
session; the transcript's label logic is pinned by its tests.

### ENGN-09 — Primed sessions: prefill the prefix at startup, restore it on new (2026-09-15)

**Outcome.** The wait after the first Enter was the prefill of the system
block and tools (about 850 tokens) plus the cold first pass, paid again on
every new session. Each profile now exposes `prefix`: the bytes every
rendering that starts with the same leading system messages (and tools)
begins with, without a generation prompt — Qwen's system block, Gemma's
`<bos>` and system turn — pinned by tests as a byte prefix of `render` that
ends at a turn boundary, so the remainder encodes as the whole would.
`Completer.prime` renders it, encodes, prefills through the engine (the
existing progress observer makes it visible), observes the tokens into the
sampler history, records it as consumed, and keeps a `Session` snapshot
(ENGN-06). `Completer.run` starts from that snapshot (`restorePrimed`)
whenever the session holds nothing usable and the rendering begins with the
prefix: the first turn, a new session, a resume, and the replay after
elision; only a rendering that does not start with it falls back to a full
replay. The interactive surface primes at startup under a "warming up" bar
(Enter queues), after a `/ctx` re-open (the snapshot is bound to the
engine's capacity and layout), and after `/think` (the effort is part of the
system block); print mode primes before its turn. A window that cannot hold
the prefix plus the output budget is a startup notice, or print mode's
error diagnostic, instead of a `ContextFull` on the first Enter.

**Evidence.** Zig 0.16.0, M4 Pro/48 GiB. `zig build test`: **379 default
tests** (the Qwen and Gemma prefixes are byte prefixes of their renderings for
every effort with and without tools; the agent's own system prompt and tool
definitions render as an increment over the prefix, never a replay). Live on
the pinned Qwen3.8-27B (Metal, context 8192, the playground), the same
two-step turn as TERM-06's check: the first step's prompt went from **876 to
35 tokens** and its prefill from **10.4 s to 0.8 s** (the 841-token prefix
was consumed at startup instead); the second step was unchanged (437 tokens,
5.9 s). The 3K elision scenario of AGNT-08, replayed after its cut, prefilled
**1,366 tokens instead of 2,207** on the restored snapshot and gave the
identical answer, which is the restore's correctness check. `--ctx-size
1024` now ends at startup with `context window too small for the system
prompt and tools: 2889 tokens (prefix plus output budget) of 1024`.

**Files.** `inference/src/profiles/{root,qwen38,gemma4}.zig`,
`inference/src/engine.zig`, `src/agent/loop.zig`, `src/agent/root.zig`,
`src/agent/print.zig`, `docs/agent-spec.md`, `docs/reference/session.md`.

**Remaining.** The interactive warm-up bar, Enter queueing during it, and the
`/new` restore were not driven on a real TTY in this session; print mode
exercised the same completer paths. The primed snapshot holds the prefix's
cache (about 64 KiB per token with F16, some 50 MB here) for the life of the
engine.

### APPS-06 — `nuclis agent ls` and `--resume` to the newest session (2026-09-16)

**Outcome.** `--resume` no longer needs the 32-digit id: alone (or as
`--resume latest`) it continues the workspace's newest session, and a
following flag is parsed as itself rather than taken for an id. `nuclis
agent ls [--json]` lists the workspace's sessions newest first — short id,
time, effort, window, and the first prompt's first line — from one
`Listing` type that renders both forms; the `/resume` picker shows the same
first prompt in its detail row instead of the directory name. A missing
session names the listing command in its diagnostic.

**Evidence.** Zig 0.16.0. `zig build test`: **379 default tests** (the parse
of a bare `--resume`, of `--resume -p …`, and of `agent ls --json`). Live:
`nuclis agent ls` in the playground printed its two sessions with their
first prompts, and the JSON form the same fields; in a workspace without
sessions it says so.

**Files.** `src/agent/resume.zig`, `src/agent/root.zig`, `src/cli.zig`,
`src/help.zig`, `docs/agent-spec.md`.

### AGNT-09 — Gemma 4 native tool calling (2026-09-16)

**Outcome.** The Gemma 4 profile renders and decodes its own tool path, so
`nuclis agent` on `gemma-4-12b` warms up with the tools block and runs the
same loop as on Qwen. Declarations render into the system turn as
`<|tool>declaration:NAME{…}<tool|>` from the JSON-Schema subset the
template's macro understands (anything outside it is
`error.UnsupportedContent`, never approximated); an assistant call is
`<|tool_call>call:NAME{key:value,…}<tool_call|>` in the template's DSL, its
results follow inside the same model turn as
`<|tool_response>response:NAME{value:<|"|>…<|"|>}<tool_response|>`, and the
turn stays open when the conversation ends on results. Reasoning renders where
the template's gate passes (after the last user message, or on any
call-bearing message, the reference server's default). The reference's own
repair — a fresh `<|turn>model\n` when results followed by content close the
turn — is reproduced because its `/apply-template` renders through the same
layer. `gemma4.parseTool` is a bounded recursive-descent reader of the DSL
behind the shared stream decoder, and the handoff needed no new mechanism:
the model emits `<|tool_response>` after the last call of a step, which the
reference marks end-of-generation and Google calls an additional stop
sequence, so it is the profile's third stop token. Two departures are pinned
by tests: reasoning is trimmed before rendering and numbers keep their JSON
text. `scripts/profile-tools-fixtures.py` takes `--profile`. The one shared
fix: both profiles now report an allocation failure during JSON parsing as
`OutOfMemory` rather than `InvalidConversation`.

**Evidence.** Zig 0.16.0, M4 Pro/48 GiB. `zig build test`: **385 default
tests** (up from 379: the fixture test asserts all **24 Gemma tool prompts**
byte for byte; the parser round-trips nested values and refuses 13 malformed
or truncated bodies; the stream decoder drives the Gemma grammar with two
calls in one step; the reasoning-trim pin; marker and schema rejections; the
registry test now expects both profiles to render). `make fmt-check` clean.
`make test-vocabulary MODEL=<gemma-4-12b>` resolves the seven tool tokens
(46–52). Fixtures captured with llama.cpp `7620399` on the K-quant 12B
(template digest and full checksum verified). Live, Metal, context 8192,
`--think medium`, `--print --json`: "create greeting.txt with hello world, then
read it back" issued `write_file` then `read_file` in three steps and answered
with the contents; the file on disk was `hello world`. Turn stats: 225 prompt
tokens, 168 generated, prefill 1.91 s, decode 8.50 s, `replayed: true` —
the model closed its first thought without a newline, so the second step
replayed from the primed prefix rather than incrementing (the third step
incremented).

**Files.** `inference/src/profiles/gemma4.zig`,
`inference/src/profiles/qwen38.zig`, `inference/src/profiles/root.zig`,
`inference/src/profiles/fixtures/gemma4-tools.json`,
`inference/vocabulary-check.zig`, `scripts/profile-tools-fixtures.py`,
`docs/reference/prompt-profile.md`, `docs/reference/tool-calling.md`,
`docs/reference/gemma4.md`, `docs/reference/agent-concepts.md`,
`docs/architecture.md`, `docs/agent-spec.md`, `docs/spec.md`,
`THIRD_PARTY_NOTICES.md`.

**Remaining.** The incremental prefill misses a step whenever Gemma closes
its thought channel without the newline the template renders before
`<channel|>`; the cost is a replay from the primed prefix. The 26B-A4B
artifact's tool tokens are unverified until that unit pulls it.

### MODL-14 — Catalogue entries ahead of their adapters; roadmap reordered around speculation (2026-09-17)

**Outcome.** Both planned families are pinned and registered before their
adapters exist: `gemma-4-26b-a4b` (the QAT file with `mmproj-BF16.gguf` and
`MTP/mtp-gemma-4-26B-A4B-it-Q4_0.gguf`, profile `gemma4`) and
`muse-glimmer-30b` (`Muse-Glimmer-30B-UD-Q4_K_XL.gguf` with
`mmproj-kquant.gguf` and `dflash-kquant.gguf`). Two decisions: the
catalogue entry's profile is optional (`null` until MODL-13; configuration
falls back as for a bare path), and a draft companion of any mechanism
takes the `mtp` role, which now means "the draft source the
speculative-decoding unit loads" (Muse's is a DFlash block-diffusion
drafter, not an MTP head). The roadmap's order after Muse Glimmer became
speculative decoding across the families, performance follow-ups, vision
through the companion projectors, agent expansion, with the accepted
configuration of speculation (the file per registry entry, a per-command
`--speculative` switch and `generate.speculative` key, a draft-length
setting; acceptance rule and recovery scheme not exposed). The roadmap's
own 26B-A4B section, superseded by the plan in `TODO.md`, was removed.

**Evidence.** Every file pulled and verified by `nuclis model pull --file`
on 2026-09-17 (digests in
[artifacts.md](reference/artifacts.md#pinned-commits-and-digests-modl-02-2026-09-11));
`nuclis model ls` lists both entries and their companions *present*;
`nuclis model inspect` says *not runnable: the gemma4 adapter rejects the
file: UnsupportedConfiguration* for the 26B-A4B and *no adapter for
architecture "muse-glimmer"* for Muse; `zig build test` (the catalogue's
well-formedness test extended to the new entries, the null profile, and
the drafter's role). No kernel or engine code changed.

**Files.** `src/catalog.zig`, `src/config.zig`, `README.md`,
`docs/reference/artifacts.md`, `docs/roadmap.md`, `TODO.md`.

**Remaining.** The entries turn *supported* when MODL-10 and MODL-13 close;
MODL-13 fills the Muse profile in and may make the profile field required
again. Whether `mtp` should be renamed to a mechanism-neutral `draft`
(sidecars, `--with`, docs) is for the speculative-decoding unit to decide
when it loads the first companion.

### TERM-07 — Rows released by a shrinking live region are reused, not left as gaps (2026-09-17)

**Outcome.** Every turn with a tool call, and every thinking block folded
or closed while the region showed its text, left blank rows in the
scrollback between the tool line and the next label: two after a settled
call, up to half the region's budget after a fold. The live region is
bottom-anchored, so a frame shorter than the previous one erased the rows
it no longer needed and moved its top down, leaving them blank *above*
itself; the next `insertAbove` scrolled the area above the region, blanks
included, and wrote below them, so the gap became permanent. `Screen`
now counts those rows as `slack`: a growing frame takes them back before
it pushes the transcript up, and an insertion fills them with absolute
cursor moves (`CSI row;1 H`) before it falls back to the scrolling
region, so the transcript stays contiguous; the rewrite fallback and
`finish` walk over the slack as well, leaving the cursor right after the
transcript on exit. The bottom anchor is unchanged: the editor and the
status bar never move away from the last row.

**Evidence.** Zig 0.16.0, M4 Pro/48 GiB. `zig build test`: **391 default
tests** (the shrink-then-grow sequence now stays on the same rows; an
insertion after a two-row shrink writes `ESC[6;1H` in place with no
scrolling region, and a longer one fills the slack then scrolls; the
fallback rewrite and `finish` walk over the slack). Live on
`gemma-4-12b` (Metal, context 4096) under a pseudo-terminal of 30 × 100:
the escape stream after a settled tool call (a two-row shrink) shows the
thought label written with `ESC[row;1H` onto the released row instead of
a scroll; confirmed by the user on their own terminal the same day.

**Files.** `src/tui/screen.zig`, `docs/agent-spec.md`.

**Remaining.** None known.

### APPS-07 — `--prompt-profile` and the registry's `profile`; the template-alias gate (2026-09-17)

**Outcome.** A Gemma 4 finetune
(`Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf`, Q4_K/Q6_K, the
12B structure, accepted by `validate`) was refused as
`UnsupportedPromptTemplate`: it ships the earlier 17,530-byte revision of
the Gemma 4 template (`dc311bb0…`), not the pinned one. The plan was to
accept that revision as a second digest of the `gemma4` profile once
proven equivalent. `scripts/profile-alias-check.py` is that proof: with
the reference server holding the file, it replays every pinned text, tool,
and token fixture case and requires byte-identical output before writing
`fixtures/<profile>-aliases.json`; profiles carry a `template_aliases` list
that `profiles.forTemplate` accepts. The proof **failed** (18 of 38 cases:
a thought on an earlier call step dropped, consecutive assistant messages
split into two turns, a result group before a user turn left without its
`<turn|>`), so both alias lists are empty and the finetune is not aliased.
What lets it run is explicit: `--prompt-profile <qwen38|gemma4>` on
`generate`, `bench`, `tokenize`, and `agent`, and a registry entry's
`profile` key, force the profile at `Engine.open` (`forced`;
`Engine.profile_forced` records that the digest did not select it), the
forced profile also names the sampling defaults, and the agent prints a
startup notice. `--model <path>` already loaded the file; no new file flag
was needed.

**Evidence.** Zig 0.16.0, M4 Pro/48 GiB. `zig build test`: 391 default
tests (the flag on every rendering command and its errors; a registry
entry's `profile` and the flag over it in `resolve`; aliases resolve
through `forTemplate`). The alias check on the finetune under llama.cpp
`7620399`: 18 of 38 cases differ, the diffs classified above.
`nuclis tokenize` on the file: refused without the flag, 14 tokens with
`--prompt-profile gemma4` (the same ids as the 12B renders). Live:
`nuclis agent -p … --prompt-profile gemma4` on the file ran the `bash`
tool and answered (Metal, context 4096).

**Files.** `inference/src/engine.zig`, `inference/src/profiles/root.zig`,
`inference/src/profiles/gemma4.zig`, `inference/src/profiles/qwen38.zig`,
`scripts/profile-alias-check.py`, `src/config.zig`, `src/cli.zig`,
`src/help.zig`, `src/tokenize.zig`, `src/generate.zig`, `src/bench.zig`,
`src/agent/root.zig`, `src/agent/print.zig`, `docs/spec.md`,
`docs/development.md`, `docs/reference/prompt-profile.md`,
`docs/reference/gemma4.md`.

**Remaining.** A forced profile renders the pinned protocol, so a file
whose template is a genuinely different protocol produces prompts its
model was not trained on; the flag is the user's statement that it is the
same one. `config show` names the forced profile but not that it is
forced. Print mode prints no notice.

### APPS-08 — `model pull`: a leaked path per file and a leading slash in the sidecar's file name (2026-09-17)

**Outcome.** Two defects in `src/model.zig`'s pull, both seen on a pull of
a non-catalogue repository. The destination check allocated each job's
local path from the package client's allocator and never freed it, which
the debug allocator reported on exit after every pull. And every sidecar
since APPS-04 recorded `file` with a leading `/`
(`"/Qwen3.8-27B-UD-Q4_K_M.gguf"`): the repository directory was joined
with a trailing empty component, which `std.fs.path.join` does not turn
into a separator, so slicing the path below it kept the slash. Nothing
read the field strictly, so no behaviour depended on it; the path is now
freed on every exit of the loop and the separator is trimmed. Existing
sidecars keep their slash until a re-pull rewrites them (a verified reuse,
no download).

**Evidence.** `zig build test` (391); a re-pull of
`HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced` verified the
existing 7.4 GB file in 23.8 s with no leak report and wrote
`"file": "Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf"`.

**Files.** `src/model.zig`.

**Remaining.** Pull has no offline test that exercises the loop, so the
leak was caught by a live run, not by the suite.

### APPS-09 — `config set`, `model pull --register`, registry names in `model ls` (2026-09-17)

**Outcome.** The registry was reachable only by editing JSON, which a
hand-pulled finetune made visible. `nuclis config set <key> <value>`
edits one key of the file's own JSON tree (stated keys and their order
survive; a missing file starts from what `init` writes), checks the
dotted key against the schema at run time (`engine.model`,
`generate.sampling.temperature`, `models.<name>.profile`; sections are
created, `schema_version` and `models` itself are refused, an unknown
entry name points at `--register`), takes the value as JSON when it
parses as JSON and as a string otherwise, and runs the result through
`fromText` before writing, so a refused value leaves the file untouched
with the key named; `engine.model` must also resolve to a file that
exists. `nuclis model pull <owner/repo> --file … --register <name>
[--profile <p>]` writes the entry (`repo`, `file`, the resolved commit,
companions by role, the forced profile) once every file is verified; the
same repository's entry gains a companion or a profile, other content
under the name is refused, and a registry-entry pull refuses the flag.
`model ls` says `registered as <name>` (with the profile when forced)
under every file an entry locates and lists entries whose file is absent;
a configuration that fails to load leaves the listing unannotated with a
warning. The live check found a hole the design had not stated: a pull
registered under a **catalogue name** succeeded and would have shadowed
the catalogue (the registry resolves first). Now `registrable` refuses a
catalogue name before any transfer unless it names the catalogue's own
file, and the loader rejects such an entry however it got into the file.

**Evidence.** Zig 0.16.0, M4 Pro/48 GiB. `zig build test`: **394 default
tests** (set on a missing file, JSON versus string values, `null`
clearing, registry keys and sections, every refusal with the file
unchanged; register new, companion, no file, three conflicts, the
catalogue-name rule in both `register` and `validate`; `ls` naming
entries and listing a missing one in both forms; the pull report line;
the flags). Live on the user's file: `config set engine.model hauhau`
(reported the previous value), `engine.ctx_size 99999`, `engine.model
nope`, and `models.hauhau.speed 1` refused with the key named; `model
ls` shows the finetune `registered as hauhau (profile gemma4 forced)`; a
`--register hauhau --profile gemma4` re-pull verified the 7.4 GB file
and rewrote the entry; `--register gemma-4-12b` refused before the
transfer.

**Files.** `src/config.zig`, `src/model.zig`, `src/cli.zig`,
`src/help.zig`, `docs/development.md`, `docs/spec.md`, `TODO.md`.

**Remaining.** `set` cannot remove a registry entry or create one; both
are a text edit or a `--register` away. The registration happens after
the sidecars, so a pull that fails only at registration (a conflict
introduced by a hand edit during the download) keeps its verified files
and reports the conflict.

### AGNT-11 — A truncated tool call no longer bricks the session; reopened thought channels; copied bracket pieces (2026-09-17)

**Outcome.** A session on the Gemma 4 finetune (`hauhau`, APPS-07) showed
a turn whose `write_file` call, carrying a long body after 87 s of
thinking, hit the 2048-token output budget before its closing bracket;
the decoder released it as answer text, as designed, but three things
went wrong around that. The released opening bracket was printed from
stale bytes (twelve U+FFFD where `<|tool_call>` belonged): the decoder
kept the bracket's *piece* as a borrowed slice into the caller's
per-token buffer, which later tokens overwrote; both bracket texts are now
copied. The model had also reopened `<|channel>thought` twice after the
answer began, and the decoder consumed only the first opening, so the
later markers landed in the answer as text. And once the stored answer
carried marker text (the released body's `<|"|>` quotes alone would do
it), every later turn's render refused the history with
`UnsupportedContent`, so the two following prompts of the session got no
assistant record at all. Now a channel opened while answering is thinking again (a
further block in the transcript) whatever the effort — with thinking off
the model still opens an empty channel after a tool result — and both
profiles remove control markers from the assistant's own content and
reasoning instead of refusing them; user and tool content and tool names
are still refused (the template's `strip_thinking` is the model for the
rule, [prompt-profile.md § Shared contract](reference/prompt-profile.md#shared-contract)).
The output budget itself is a setting: `--max-tokens` and
`generate.max_tokens` go to 4096, and a long file body inside a call
needs it. Related: the turn's end reported `eos` when its step had run out
of output budget (and `token_budget` for the *step* budget, which already
has its own notice); the turn now ends with the last step's stop.

**Evidence.** Zig 0.16.0, M4 Pro/48 GiB. `zig build test`: **396 default
tests** (a reopened channel with the effort on and off, an empty reopened
channel; the bracket copied out of a buffer the test overwrites; Gemma
and Qwen render assistant text with markers removed and still refuse them
elsewhere). Live on `hauhau` (Metal, context 4096, print mode): a
`write_file` call truncated by `--max-tokens 48` was released as text
with its bracket intact, and the next turn on the same session file
rendered, ran `bash`, and answered.

**Files.** `inference/src/profiles/stream.zig`,
`inference/src/profiles/gemma4.zig`, `inference/src/profiles/qwen38.zig`,
`src/agent/loop.zig`, `docs/reference/prompt-profile.md`.

**Remaining.** The strip is not pinned by a reference fixture (the fixture
scripts synthesize clean cases); a truncated call is still text the model
sees as its own broken output next turn, which is the honest history.

### APPS-10 — `model ls` as one aligned grid (2026-09-17)

**Outcome.** The catalogue section padded names to a fixed width but let
the status word and the path start wherever the name ended, printed
companions with their own narrower columns, and put a main file's size
under its name; sizes were left-aligned. The listing is now one grid: the
name column fits the widest catalogue name (a companion's role sits two
cells in), the status column the widest status word, the path column the
widest main or companion path, and every size is right-aligned after it;
a main file's detail row (size, encoding, commit, digest) and its
`registered as` line start at the path column; the "other files" and
"missing entries" sections align their own columns the same way. Sizes
read in decimal units (`16.46 GB`, `931.1 MB`); the JSON form and the pull
report keep exact bytes.

**Evidence.** `zig build test` (397; the `ls` test now derives the
expected column positions from `catalog.name_width`; the unit formatter
has its own); `nuclis model ls` on the user's five catalogue entries and
two other files.

**Files.** `src/model.zig`.

**Remaining.** Rows are wide (a companion row runs past 150 columns with
today's paths); wrapping to the terminal width would be a next step if it
bothers anyone.

### TERM-08 — The welcome: ASCII wordmark and session facts; the model's name on the status bar (2026-09-17)

**Outcome.** The agent's one-line header (`nuclis agent · <name> · metal`)
became a welcome (`src/tui/banner.zig`): the six-row ASCII `NUCLIS`
wordmark the user chose when the terminal is 60 columns or wider, then
`nuclis agent <version>`, a model line (the registry or catalogue name
the model was reached through when it was one, the artifact's own
`general.name`, the backend, the profile and `(forced)` when a flag or
the entry forced it), and a settings line (context, effort, the workspace
with `$HOME` shortened to `~`); below that width the one-line header
stays. It is transcript, so it scrolls away with the conversation (the
rendering model has no title bar), which is why the status bar now ends
with the model's short name. The rows are plain ASCII: no glyph-set
fallback is needed.

**Evidence.** Zig 0.16.0, M4 Pro/48 GiB. `zig build test`: **400 default
tests** (the wordmark's shape and width; wide and narrow renderings with
and without a registry name and a forced profile; the tilde rule at a
path boundary; the bar's last segment). Seen by the user on their
terminal on the default model.

**Files.** `src/tui/banner.zig`, `src/tui/root.zig`, `src/tui/status.zig`,
`src/agent/root.zig`, `docs/agent-spec.md`.

**Remaining.** The hint row still sits under the editor as before; the
startup notice for a forced profile is printed in addition to the
welcome's `(forced)`.

### KERN-09 — Expert routing and gathered expert kernels (decode and prefill) (2026-09-17 / 2026-09-18)

**Outcome.** The one thing the Gemma 4 26B-A4B needs that the tree
lacked: selecting experts per token and running the selected experts'
quantized weights without touching the other 120. The CPU reference
`cpu.experts` (`ExpertMatrix` over a 3-D tensor of contiguous expert
matrices, `route` with softmax, top-k by logit and renormalized weights
under the F16-normal floor, the gathered gated-GELU `ffn` in F64). On
Metal, the decode path in four dispatches per layer — `route` (one
256-thread group per logit row), `matvecExperts` (the segment bodies over
a selected expert's slice, any encoding, shared or per-slot inputs,
indices clamped), `geluMulRows`, `combineExperts` (optional per-expert
down scale) — and the prefill path in two more: `expertLists` (one
threadgroup groups the chunk's slot rows by expert and writes the 32-row
tile list, bounded by n / 32 + experts before the counts exist) and
`matmulExperts` (the dense tile body with a gather flag: activation rows
gathered through the row list, results scattered back through
threadgroup memory, no padding rows; a 64 × 32 half tile for Q4_0 and
the generic F32 32 × 32 tile otherwise). Along the way the Q4_0 matvec
body walks 32-value blocks rather than 256-value strides, so a
704-column row (22 blocks) takes the specialized kernel.

**Evidence.** Zig 0.16.0, M4 Pro/48 GiB, `make check` (`zig build test`
400, `test-metal`). Router vs the F64 reference on 128 experts (k 8, ties
at the top and the k boundary, an all-equal row, a 40× spread) and 37
experts (k 5): indices exact, weights within 1.5e-8. Gathered Q4_0 matvec
on both kernel paths with NaN in every unselected expert: selected
outputs bit-identical. Decode chain over three tokens vs `cpu.experts.ffn`:
1.5e-8 of (1 + max|y|). Prefill chain over a skewed 45-token chunk (135
slot rows; one expert on every token, one on none) vs the reference and
vs the decode path row by row: F32 tile 1.1e-6 of max|y|, half tile
4.4e-4; the lists checked on the host; a one-token chunk; the contract
rejections. `make bench-experts` (ReleaseSafe, 2026-09-17/18): decode
gate-up 215 GB/s (= dense), down 154 GB/s (22-block rows leave 10 of 32
lanes idle), chain 172 GB/s of the selected experts' bytes; prefill
tiles at the dense tile's compute rate (4.8 TFLOP/s executed against
5.0 for the dense 64 × 64 Q4_0 tile), 41 GB/s of tile bytes at every
chunk, with the useful rate set by the tile fill — 50 % at 256 tokens
(10.5 ms per layer-chunk), 69 % at 512, 81 % at 1,024 (27.0 ms) — four
to six times faster than looping the decode kernels over the chunk.
Details and the tables in
[metal-backend.md § Gathered expert kernels](reference/metal-backend.md#gathered-expert-kernels-kern-09).

**Files.** `inference/src/backends/cpu/experts.zig`,
`inference/src/backends/cpu/root.zig`,
`inference/src/backends/metal/kernels.metal`,
`inference/src/backends/metal/root.zig`, `inference/metal-check.zig`,
`inference/build.zig`, `build.zig`, `Makefile`,
`docs/reference/cpu-reference.md`, `docs/reference/metal-backend.md`,
`docs/reference/bench.md`, `docs/development.md`, `docs/llm-guide.md`
(§ 48).

**Remaining.** A short-row lane mapping for the 22-block down projection
at decode; a 16-row or 64-row token tile for chunks whose experts are far
from 32 rows; Q4_K / Q6_K gathered tile instantiations for other
families (the segment bodies already serve them at decode); the order of
rows within an expert is atomic, not deterministic (the outputs are).

### MODL-09 — Gemma 4 26B-A4B: artifact pin, facts, adapter, CPU reference, Metal plan (2026-09-18, two sessions)

**Outcome.** The first mixture of experts runs on both backends. The
Gemma adapter binds two pinned configurations selected by `block_count`
(`gemma4.Config`: the 12B and the 26B-A4B with its expert block — 128
experts, 8 used, 704 wide — and the global layers' KV heads, one or
two), validated against a committed inventory fixture with mutation
tests on both; `Layer` carries its KV heads, `weights.View.expertMatrix`
views the 3-D tensors. The CPU runtime executes the expert layer
(`feedForward`: the shared branch with its own post norm, the router
over the unweighted RMS norm of the residual scaled by 1/√2816 and the
router scale, the gathered experts over `pre_ffw_norm_2`, their post
norm, the sum's post norm). The Metal plan runs the same schedule from
the gathered kernels of KERN-09: at decode `route` → `matvecExperts`
(gate-up, shared input) → `geluMulRows` → `matvecExperts` (down) →
`combineExperts` with the per-expert down scale; per prefill chunk the
router through the generic F32 tile, `route` over the rows,
`expertLists`, `matmulExperts` twice, `combineExperts` with the chunk's
rows. Every expert tensor is wrapped whole and resident (14.2 GB); the
wide attention kernels index two KV heads without change. The chunk for
the family is 512 (`Plan.preferredChunk`, an engine override per binding).
Facts, the forward pass, and the artifact's provenance are in
[gemma4.md § 26B-A4B](reference/gemma4.md#gemma-4-26b-a4b-the-expert-configuration-modl-09).

**Evidence.** Zig 0.16.0, M4 Pro/48 GiB, `make check`, `test-metal`
(the 16-over-2, width-512 geometry added: chunk 3.0e-6 / 1.9e-4 F16,
decode 2.4e-7). `make compare-gemma4-26b-a4b` on the pinned `<bos>Hello,`
traces (91 files, three positions): CPU 5.8e-5 / 3.1e-6, Metal F32
7.7e-5 / 3.1e-6 (bring-up thresholds 2e-3 / 1e-4), Metal F16 1.9e-2 /
1.0e-3 (the family's 1.0 / 5e-2), greedy token 29104 and the same top-5
on every path. `make test-generation-gemma4-26b-a4b-metal`: sessions
bit-identical, snapshot/restore bit-exact; chunked prefill vs steps
2.8e-4 / 1.2e-5 through the generic F32 tiles (the schedule is exact)
and 1.1e1 / 3.5e-1 through the specialized half tiles, with the same
greedy token — the rounding class of the dense tiles amplified by the
discrete routing (a perturbed router logit swaps a token's eighth
expert), recorded as the expert configuration's own bounds beside the
12B's. The 12B and QAT comparisons and their generation check unchanged
to the digit; Qwen `make bench` unchanged (40.05 / 10.67). First-look
rates (`--kv f16`): 22-token prompt 122.6 / 57.9 tok/s; 512-token array
525.9 / 55.3 at chunk 512 (463.3 at 256); 4,096-token array 361.0 / 49.6
(329.0 at 256, 382.5 at 1,024).

**Files.** `inference/src/models/gemma4.zig`,
`inference/src/models/gemma4_runtime.zig`,
`inference/src/models/gemma4_metal.zig`,
`inference/src/models/fixtures/gemma4-26b-a4b.json`,
`inference/src/runtime/weights.zig`, `inference/src/engine.zig`,
`inference/metal-check.zig`, `inference/generation-check.zig`,
`tests/fixtures/gemma4-26b-a4b-hello-comma/`, `Makefile`, `README.md`,
`docs/reference/gemma4.md`, `docs/reference/metal-backend.md`,
`docs/reference/artifacts.md`, `docs/architecture.md`,
`docs/development.md`.

**Remaining.** The chunked prefill's half-tile rounding on a routed model
(3.5e-1 relative RMS on random tokens): how it shows on real prompts is
what MODL-10's acceptance record measures against the reference; the
options if it matters are an F32-activation tile variant or the
generic tile for the router's input path. The down projection's idle
lanes at decode and the 64-token expert tile at prefill (KERN-09's
follow-ups); the ring layout for windowed caches; the catalogue verdict,
acceptance record, and agent check are MODL-10.

### MODL-10 — Gemma 4 26B-A4B: catalogue verdict, acceptance record, agent check (2026-09-18)

**Outcome.** The mixture of experts is *supported*: `model inspect
gemma-4-26b-a4b` reads the Hub's digest at the pinned commit equal to the
catalogue's and the `gemma4` adapter binding it (the entry, pinned ahead of
its adapter, needed no change). The acceptance workload ran on both sides:
the reference harness on the file with `--family gemma4`
(`tests/fixtures/run-2026-09-18-gemma4-26b-a4b/`, token arrays
byte-identical to the two 12B runs', summarized in
`docs/benchmarks/reference-2026-09-18-gemma4-26b-a4b.json`) and `make
baseline-gemma4-26b-a4b` (new target) feeding those arrays through `bench
--prompt-tokens` (`docs/benchmarks/nuclis-2026-09-18-gemma4-26b-a4b.json`).
The per-kernel profile of the decode step was taken and ranked the
follow-ups. The write-then-read agent check ran on the entry. Documents:
the record and profile in bench.md, the pointers in gemma4.md, the
catalogue and provenance notes in artifacts.md, the expert configuration
as a worked example in architecture.md § Adding a model, the gate list in
development.md, the fixture row in `tests/fixtures/provenance.md` (whose
two 12B rows named stale `baseline` targets, corrected), and the README.

**Evidence.** Zig 0.16.0, M4 Pro/48 GiB, AC power, ReleaseSafe, tree
`56ef7d4` plus this unit's Makefile and documents. Acceptance record:
prefill / decode **500.42 / 55.29** at 512, **346.17 / 49.44** at 4K,
**206.44 / 39.55** at 16K, **134.74 / 31.30** at 32,639 against the
reference's **580.76 / 68.02**, **548.41 / 60.82**, **459.46 / 50.67**,
**343.38 / 44.09** (decode 81 / 81 / 78 / 71 %, prefill 86 / 63 / 45 /
39 %); every sample on both sides stopped on the token budget; session
**6.87 GiB** at 32K, peak footprint **8.0 GB**
([bench.md](reference/bench.md#gemma-4-26b-a4b-acceptance-record-modl-10-2026-09-18)).
Profile (22-token prompt, 2K context): 915 dispatches per step, a decode
step about 20.5 ms attributed of which expert matvecs 27 % (down
projection **114 GB/s** on 704-wide rows), dense matvecs 43 %, norms 14 %
(331 launches), attention 8 %, routing glue 4 %; profile mode costs this
model 42 % (57.9 → 33.4 tok/s), so shares are indicative. Agent check
(Metal, context 8,192, `--think medium`, `--print --json`): `write_file`
then `read_file`, the answer from the contents, `hello world` on disk;
**221** prompt tokens, **71** generated, prefill **0.99 s**, decode
**1.30 s**, `replayed: true`. Decode in the loop 18–19 ms per token
against the 12B's 8.50 s for 168 tokens (51 ms) and the Qwen3.8-27B's
about 94 ms. `make check` **401 tests** and `test-metal` passed after the
runs.

**Files.** `Makefile`, `tests/fixtures/run-2026-09-18-gemma4-26b-a4b/`,
`tests/fixtures/provenance.md`,
`docs/benchmarks/reference-2026-09-18-gemma4-26b-a4b.json`,
`docs/benchmarks/nuclis-2026-09-18-gemma4-26b-a4b.json`,
`docs/reference/bench.md`, `docs/reference/gemma4.md`,
`docs/reference/artifacts.md`, `docs/architecture.md`,
`docs/development.md`, `README.md`.

**Remaining.** Decode at 71–81 % and prefill falling to 39 % of the
reference at 32K, with the ranked levers in bench.md: the expert down
kernel's idle lanes on 704-wide rows, a fused or batched norm launch (421
small dispatches per step), the wide flash-decoding kernel at long
context (13.8 ms added per step from 512 to 32K against the reference's
8.0), and the per-chunk cost of the gathered prefill tiles (64 routed
chunks at 32K). The half-tile rounding on routed layers is unmeasured
against the reference on real prompts beyond the greedy budget runs; the
ring layout for windowed caches stays a roadmap item.

### MODL-15 — Bonsai 2 27B accepted ahead of Muse: artifact pinned, facts read, three units planned (2026-09-18)

**Outcome.** Prism ML's Bonsai 2 27B (released 2026-09-17) is Qwen3.8-27B
re-encoded: the GGUF declares `qwen35` with every architecture key equal
to the pinned file's and the same tokenizer, so the adapter, tokenizer,
runtime, and plan apply, and the work is a new weight layer: two ternary
encodings at group 128 (`PQ2_0` id 142, 34 bytes per block; `PTQ1_0` id
143, 28 bytes, mainline `TQ1_0`'s trit packing), 96 BF16 tensors the
engine stores but does not decode, and a blockwise Walsh-Hadamard rotation
(block 1024, explicit signs per input width, `prism.hadamard.*`) applied to
activations before 401 projections and inverted on the embedding rows.
Stock llama.cpp rejects the files; the PrismML fork at its release tag is
the second pinned oracle. Decided: planned ahead of Muse Glimmer (three
units on existing seams that improve the primary target, against five for
a new architecture); the catalogue entry `bonsai-2-27b` pins the PQ2_0
file (the bring-up packing: a 2-bit unpack of the Q4_0 kernel's shape,
and the faster prompt processing by Prism's table) with the Q8_0
projector under `mmproj`; PTQ1_0 (5.95 GB) follows in the same units. The
facts and the three unit designs (MODL-16, KERN-10, MODL-17) are in
`TODO.md` until `docs/reference/bonsai.md` records them.

**Evidence.** Facts read from the PTQ1_0 header (first 16 MiB parsed
directly), the model card, the whitepaper, and the fork's `prism-v7`
`ggml.h` / `ggml-common.h`; digests read by `model inspect` at commit
`6ed5e12b…` (`PQ2_0` 7,206,168,928 B `3907dc16…`, `PTQ1_0` 5,946,648,928 B
`53107f53…`, `mmproj-Q8_0` 629,246,976 B `6807ede6…`, `mmproj-BF16`
931,145,856 B `e287342d…`), the entry and its projector pulled and
verified by `nuclis model pull bonsai-2-27b --all`; `inspect` reports the
main file *not runnable: tensor output.weight uses encoding id 142, which
nuclis does not store*, as intended. The chat template's digest
(`c3cf9e34…`, 8,952 bytes) differs from the pinned `qwen38` one. `zig
build test` with the catalogue assertions; `make fmt-check`.

**Files.** `src/catalog.zig`, `TODO.md`, `docs/roadmap.md`,
`docs/reference/artifacts.md`, `README.md`.

**Remaining.** Everything the three units design; `gdn_v_grouped` and the
fork's exact transform contract are read from its loader in MODL-16, not
inferred.

### APPS-11 — `model pull`: a verified file whose encoding this build does not store keeps its sidecar (2026-09-18)

**Outcome.** After publishing a digest-verified download, `pull` reads the
GGUF header to learn the file's role (`imatrix`, `mmproj`), and that read
parsed the whole tensor directory: a file with a tensor encoding this
build does not store (the Bonsai files, ids 142 and 143) failed there,
which left a verified 7.2 GB file without its sidecar (so `ls` called it
unverified) and abandoned the companions after it. The role question is a
metadata one, so a directory the build cannot read now answers *no role
from the header* and the request's role is recorded, as the contract
already said for a header that says nothing; every other parse failure
still propagates.

**Evidence.** A test serializes the Qwen inventory with `output.weight`
at encoding 142 and checks the null role and its resolution to `main` or
the flag's role; `zig build test`. `nuclis model pull bonsai-2-27b --all`
then reused the published file (hashed, verified) and wrote its sidecar,
pulled the projector, and `model ls` shows both *present*.

**Files.** `src/model.zig`.

**Remaining.** None.

### REPO-05 — README for a public repository: project status, contributions, disclosure (2026-09-18)

**Outcome.** Three sections for the repository going public, decided
2026-09-18. *Project status* states what this is: an experimental,
free-time project for learning the inference stack and Zig together, not
production software, one machine, 0.x with breaking changes, understanding
before features. *Contributions* states the policy plainly: issues with a
reproduction or a measurement from other hardware are welcome, pull
requests are not being reviewed, forks are encouraged; no `CONTRIBUTING.md`,
because a contribution process nobody runs would be a false promise.
*Disclosure* says the code is written with heavy assistance from AI
coding agents under human direction, points at the committed `AGENTS.md`
as the inspectable process, states that every number was measured, and
acknowledges llama.cpp and GGML as the format's origin and the oracle
every kernel is proven against, with no code copied and the third-party
notices as the record. The wording is this project's own (its rule
against copying from references covers prose about the project too). The
former *Status* section, the measured results, is *Results so far*, so
the two headings no longer collide; the intro names all three families.

**Evidence.** Documentation only: links checked, the licence file and the
notices present. Absolute paths to the developer's toolchain remain in
`docs/development.md` and `docs/architecture.md` and the benchmark
records name the model path under the home directory; environment facts,
left as recorded.

**Files.** `README.md`.

**Remaining.** The GitHub repository's description and topics are set
outside the tree; the project page named in *Name* is not live.

### MODL-16 — Bonsai 2 27B: oracle, facts, ternary encodings, Hadamard transform, CPU reference (2026-09-18)

**Outcome.** Bonsai 2 27B runs on the CPU reference and matches its
oracle. The PrismML llama.cpp fork at release `prism-b10687-5d80cff`
(`5d80cff0…`, MIT) is the second pinned oracle, built beside mainline
with the same recipe; two pins, never one moving one
(`quant-fixtures.py`'s `PRISM_REVISION`, `profile-alias-check.py
--reference-revision`). `docs/reference/bonsai.md` records the artifact,
header, tensors, both block layouts, and the rotation contract as the
fork's loader implements it: `R = (1/√1024) H S` per 1024-block, explicit
±1 signs per input width (5120 / 6144 / 17408), 401 rotated matrices
(everything but the embedding, which stores rotated rows and gets
`S · H` after lookup, and the BF16 `ssm_alpha` / `ssm_beta`), and
`gdn_v_grouped` = the `ssm_out` input regathered from the mixer's tiled
value-head order to the grouped order the fold used. `quant.row` decodes
PQ2_0 (id 142), PTQ1_0 (id 143, base-3 trits extracted by the fixed-point
multiply, digit-major runs), and BF16 (id 30); the GGUF parser retains
`prism.hadamard.*` arrays whole; the Qwen adapter lists the three
encodings as executable, accepts the file's 64 blocks without the draft
head (the header read in MODL-15 was wrong on this and on `file_type`
141), validates the rotation against the pinned contract, and carries it
in `Binding.rotation` (`Summary.rotated_basis`, shown by `validate`).
`cpu.hadamard` implements the transform; the Qwen runtime applies the
inverse after the embedding lookup and the forward transform to the four
activations per layer that meet rotated weights plus the output-head
input, with the value-head gather before `ssm_out`. The Metal plan
refuses a rotated binding (`UnsupportedRotation`) until KERN-10 and
MODL-17. The Bonsai chat template (`c3cf9e34…`, the upstream Qwen3.8
template) is not an alias of `qwen38`: 4 of 68 fixture cases differ, all
the merged-system histories it refuses; every other prompt and token
stream is byte-identical. MODL-17 chooses the profile from that evidence.

**Evidence.** The fork's `llama-completion -ngl 99` on the PQ2_0 file
(`Hello, I'm a student in the University of the West of England (UWE)`,
17.1 tok/s on 16 tokens, a smoke run) and its loader line `loaded 402
Hadamard-folded weight(s) (1 inverse-lookup) using 1 rotation(s) and 3
sign vector(s)`; `tests/fixtures/bonsai-hello-comma/` (129 files, greedy
token 353, the same as Qwen3.8's) from `reference-generation.cpp` built
against the fork; `inference/src/quant/fixtures/ternary.json` from the
fork's own decoders, matched exactly, plus hand tests for code order, the
+2 code, the canonical trit packing, and BF16 edge values;
`fixtures/bonsai-2-27b.json` (the inventory with rotation arrays) bound
in tests with 851 / 0 tensors and the negative cases (version, sign mode,
a sign of 0, a never-rotated name, a slot claimed twice, an unknown key,
rotation keys without the version, 65 blocks); `cpu.hadamard` against the
parity-defined matrix and as a round trip; `nuclis validate --model
bonsai-2-27b` (851 tensors, 7,195,047,936 bytes, the rotated basis
printed) and `model inspect` (*supported*); **`make compare-bonsai-cpu`
passed on the first run: 129 files, max abs 2.44e-4, max relative RMS
5.4e-6, greedy 353 with the fork's top three logits to three decimals**
(56 s for two positions); the alias gate's 64 / 68 on the fork's server;
`make check` (unit tests, `test-metal` unchanged) and `make compare`
(Qwen unchanged).

**Files.** `inference/src/tensor/encoding.zig`,
`inference/src/quant/decode.zig`, `inference/src/quant/fixtures/ternary.json`,
`inference/src/formats/gguf.zig`, `inference/src/models/inventory.zig`,
`inference/src/models/fixtures/bonsai-2-27b.json`,
`inference/src/models/qwen35.zig`, `qwen35_runtime.zig`, `qwen35_metal.zig`,
`registry.zig`, `inference/src/backends/cpu/hadamard.zig`, `cpu/root.zig`,
`src/catalog.zig`, `src/validate.zig`, `src/model.zig`, `Makefile`,
`scripts/quant-fixtures.py`, `scripts/gguf-inventory.py`,
`scripts/profile-alias-check.py`, `tests/fixtures/bonsai-hello-comma/`,
`tests/fixtures/provenance.md`, `docs/reference/bonsai.md`,
`reference-baseline.md`, `quantization.md`, `cpu-reference.md`,
`generation.md`, `artifacts.md`, `prompt-profile.md`, `docs/development.md`,
`docs/llm-guide.md` (§ 49), `README.md`, `THIRD_PARTY_NOTICES.md`.

**Remaining.** Metal: the ternary matvecs and tiles, BF16 rows (not in
KERN-10's original design; added to its plan), and the transform kernel
(KERN-10); the Metal plan, the profile decision, the catalogue's
`profile`, and the acceptance record against the fork's server (MODL-17).
PTQ1_0 is decoded but no file of it is pulled or traced. The fork's
`llama-completion --jinja` aborts on the Bonsai template's start-up
self-test (its server renders it). The CPU rotation copies each rotated
activation into scratch; the reference favours clarity over the copy.

### KERN-10 — Ternary matvec and matmul tiles, the Walsh-Hadamard kernel (2026-09-18)

**Outcome.** The Metal backend executes Bonsai 2 27B's three encodings
and its rotation. `dequant.metal` decodes PQ2_0, PTQ1_0, and BF16 in the
CPU decoder's order; `nu_matvec_pq2_0` / `nu_matvec_ptq1_0` keep the
set's geometry with four lanes per 128-value block and uniform lane work
(a PTQ1_0 quarter is four bytes of the 16-byte run, two of the 8-byte
run, one digit of the tail, in the digit-major layout's strided order),
factoring `w = d · t` as `d · (Σt·x − Σx)`; two-bit fields are masked
four bytes at once and paired with the inputs by a free register
re-labelling, trits come out of 16-bit slots two bytes at a time. The
prefill tiles `nu_matmul_pq2_0` / `nu_matmul_ptq1_0` and their `_32`
forms decode eight segments per block with the generic expression.
`specializedMatvec` / `specializedMatmul` accept any whole-block row at
two-byte (PQ2_0) and four-byte (PTQ1_0) alignment. `nu_hadamard` is the
signed blockwise transform, forward and inverse, one 256-thread group per
1,024-block with two register stages and eight threadgroup stages;
`Backend.hadamard` validates alignment and width. `make bench-hadamard`
is the per-token cost of the separate kernel.

**Evidence.** `test-metal`: the ternary fixture rows decoded exactly
through the matvec (one-hot columns) and the embedding kernel; randomized
1,280 / 5,120 / 17,408-column rows through the specialized and generic
paths against the F64 CPU; the half tiles against the generic F32 tile
within the set's bound; the selection rules; `nu_hadamard` against
`cpu.hadamard` on the three rotated widths over strided rows, forward,
inverse, and round trip at max abs 7.2e-7. `make bench-kernels` (output
head, best GB/s): PQ2_0 118.7, PTQ1_0 88.5, Q4_0 230.7 in the same run —
**the design's 200 GB/s target is not met**: the kernels are at the
set's multiply rate (448 / 405 G values/s against Q4_0's 410) and a
ternary byte carries twice the values, so the byte rate halves; the
extraction rewrite took PQ2_0 from 105 to 119. `make bench-matmul`
(256 tokens): the ternary tiles at Q4_0's GFLOP/s (4,971 / 4,711 against
4,724). `make bench-hadamard`: 1.34 ms best, 2.20 ms mean per token for
258 dispatches, 0.051 µs per block of arithmetic — launch overhead, 2–3 %
of a projected token; fusion into the matvec input load is ruled out by
arithmetic (each SIMD group would recompute the block), a norm-fused
variant is left to MODL-17's profile. `make check`.

**Files.** `inference/src/backends/metal/dequant.metal`, `kernels.metal`,
`root.zig`, `inference/metal-check.zig`, `build.zig`, `Makefile`,
`docs/reference/metal-backend.md`, `docs/reference/bonsai.md`,
`docs/development.md`.

**Remaining.** The Metal plan does not yet dispatch the transform or run
a rotated binding (MODL-17). A faster ternary matvec needs different
arithmetic (packed integer products, decoded-value sharing across
inputs) and is judged against MODL-17's model rate; PTQ1_0 decodes 25 %
slower per byte than PQ2_0 here. No BF16 matvec beyond the generic path
(96 rows of 5,120 × 48 per token: negligible).

### MODL-17 — Bonsai 2 27B: the Qwen plan on rotated weights, catalogue, acceptance (2026-09-18)

**Outcome.** The Qwen Metal plan runs the rotated file: the three sign
vectors and a 48-entry value-head gather map go to the GPU at plan init,
the embedding row gets the inverse transform after the gather, and every
activation a folded weight reads gets the forward transform in place
(`Backend.hadamard`) before its projection — the normed residual before
the mixer, the mixer output before `attn_output` / `ssm_out`, the normed
residual before the FFN pair, the FFN hidden before `ffn_down`, the
output-head input — on both the decode and the chunked-prefill paths;
`ssm_alpha` / `ssm_beta` read the residual first (the four-way merged
DeltaNet projection splits in two around the transform on a rotated file
only). `ssm_out`'s input is regathered from the tiled to the grouped head
order by a new row-gather kernel (`nu_gather_rows`, `Backend.gatherRows`)
into a scratch and transformed there. The profile is decided: the
catalogue entry's `profile = .qwen38`, and a catalogue entry's profile is
now forced at open like a registry entry's, so a file whose own template
digest is not pinned renders the entry's protocol without configuration;
the deviation (nuclis merges leading system messages the upstream
template refuses) is recorded in bonsai.md. The entry moved to the
PTQ1_0 packing (5,946,648,928 B, SHA-256 `53107f53…`), measured not
slower than PQ2_0 on the whole token at 1.26 GB less; `make
compare-bonsai` (three rows), `test-generation-bonsai-metal`, and
`baseline-bonsai` exist; the workload harness takes
`--reference-revision` and the nuclis record reads the reference revision
from the records it cites.

**Evidence.** `make compare-bonsai` against the fork's traces (129 files
each): cpu max abs 2.44e-4 / relative RMS 5.4e-6, **f32 4.27e-4 / 5.4e-6**,
**f16 1.34e-2 / 7.4e-5** (the Qwen F16 tolerance), greedy token 353 on
every row; the PTQ1_0 file on the same plan 2.29e-4 / 1.29e-2 and its
CPU row 2.44e-4. `make test-generation-bonsai-metal` passed on both
packings (chunk 64 against per-token steps max abs 7.9e-4, relative RMS
4.0e-5, argmax 278/278). `metal-check` proves the gather exact on the
48 × 128 regrouping over strided rows. `make bench` (22-token prompt, 64
output tokens, context 2,048, F16 cache, warm): PQ2_0 **12.87 tok/s**
decode / 39.1 prefill, PTQ1_0 **13.05** / 39.3; `qwen3.8-27b` on the same
plan after the change 10.42 / 39.3, unchanged. Per-kernel profile: 1,292
dispatches per step, the transform **1.66 ms (2.1 %)**, the gather 0.16
ms, the ternary matvecs about 80 % at 97–116 GB/s — so no norm fusion,
and the rate the byte count promised (2–3×; delivered 1.3×) is a matvec
arithmetic question. Acceptance record against the PrismML fork's server
(`prism-b10687-5d80cff`, its `--lazy-mode` dropped) on the Qwen token
arrays (byte-identical, verified): decode **13.87 / 13.26 / 11.72 /
10.20** tok/s at 512 / 4K / 16K / 32,639 against the fork's 17.05 /
16.61 / 13.42 / 12.83 (80–87 %), prefill 93.69 / 86.01 / 66.35 / 50.63
against 97.54 / 99.07 / 79.69 / 84.76 (96 % → 60 %); every sample stopped
on `token_budget` on both sides; session 2.15 GiB, peak RSS 2.36 GiB
([bench.md](reference/bench.md#bonsai-2-27b-acceptance-record-modl-17-2026-09-18)).
Agent check (Metal, context 8,192, `--think medium`, `-p --json`):
`write_file` then `read_file` in three steps, `hello world` on disk, 176
prompt tokens, 210 generated, prefill 3.19 s, decode 15.1 s, `replayed:
true`, stop `eos`. `make check`: 413 tests and `test-metal`.

**Files.** `inference/src/models/qwen35_metal.zig`,
`inference/src/backends/metal/kernels.metal`, `root.zig`,
`inference/metal-check.zig`, `src/catalog.zig`, `src/config.zig`,
`Makefile`, `scripts/reference-baseline.py`, `scripts/nuclis-baseline.py`,
`tests/fixtures/run-2026-09-18-bonsai/`, `tests/fixtures/provenance.md`,
`docs/benchmarks/reference-2026-09-18-bonsai.json`,
`docs/benchmarks/nuclis-2026-09-18-bonsai.json`, `docs/reference/bonsai.md`,
`docs/reference/bench.md`, `docs/reference/metal-backend.md`,
`docs/reference/artifacts.md`, `docs/reference/reference-baseline.md`,
`docs/development.md`, `docs/spec.md`, `README.md`.

**Remaining.** Decode is at 80–87 % of the fork and 1.3× the Qwen3.8
rate where the bytes promise 2–3×: the ternary matvecs are at the kernel
set's multiply-rate ceiling, and closing the gap needs a different
arithmetic (packed integer products, or one decoded weight shared across
several inputs) — a kernel unit for the roadmap's performance theme,
judged against this record. The acceptance workload was measured on the
PQ2_0 file before the entry moved; the PTQ1_0 file has the short bench
and the traces. The transform stays 258 separate dispatches (2.1 %).

### AGNT-12 — An empty tool result no longer aborts the turn (2026-09-18)

**Outcome.** A tool result with no text (a `glob` with no match, a `grep`
with none, a command with no output) ended the turn with `EmptyPrompt`:
the loop counts every result's tokens to fit it to the context, and the
engine's encoder refuses an empty text as it refuses an empty prompt.
`Completer.count` now costs an empty text zero tokens, and the `Model`
contract says so. Found on `bonsai-2-27b` in the playground: the model's
second step called `glob` with `**/*\.py` (nothing matches the literal
backslash) and `bash` together, and the turn died before either result
was recorded.

**Evidence.** A loop test with a stub model whose `glob` returns empty
text reaches the answer with the empty `.tool` message in the history
(414 tests). Live on `bonsai-2-27b` (Metal, `--think low`, `-p --json`):
"Are there any Rust files (*.rs) in this project? Use glob." fed two
empty `glob` results back (`no files`) before a third listing and the
answer, stop `eos`, 331 generated tokens.

**Files.** `src/agent/loop.zig`.

**Remaining.** The model reads an empty `<tool_response>` as ambiguous
and retried the pattern twice; a tool text that says "no files match
`*.rs`" would save those steps (the summary row already says it).

### APPS-12 — The default context window is 16K (2026-09-18)

**Outcome.** `engine.ctx_size` defaults to 16,384 instead of 8,192 for
every command (the agent runs on the engine defaults): an agent turn's
system prompt, tool definitions, and a few results on top of the prompt
ran out of 8K in ordinary use. A `ctx_size` in `nuclis.json` or
`--ctx-size` still wins, so a file written by an earlier `config init`
keeps its 8,192 until edited. The cost is the session block: Qwen3.8 and
Bonsai about 1.1 GiB at 16K with the F16 cache, Gemma 4 12B about 5.6 GB
(its sliding caches are allocated for the full capacity, the ring layout
being a roadmap item).

**Evidence.** The resolution tests assert the new default (414 tests);
`nuclis --help` and the `config init` example in development.md say 16384.

**Files.** `src/config.zig`, `src/help.zig`, `docs/development.md`.

### TERM-09 — A step that answers and then calls a tool no longer holds the turn in the live region (2026-09-18)

**Outcome.** When a step streamed answer text and then made a tool call,
its answer block stayed open (only the turn's end closed answers), the
transcript's in-order writer stopped at it, and every later block of the
turn stayed in the live region — where a thinking block was painted with
the animated busy label whether or not it had closed. The screen showed
several "thinking… 207s" labels ticking on the turn's clock after the tool
rows of steps that had long finished. A `tool_call` event now closes the
step's open text (a call ends what the step had to say), and a closed
thinking block still in the live region wears its fold label
("Thought for 3.0s") rather than the busy one.

**Evidence.** A transcript test: thinking, answer, call → the whole step
is written at once and nothing live says "thinking…"; the next step's
thought shows the busy label while open and its fold label once closed
before it is written (415 tests). Seen on `bonsai-2-27b` in the
playground with "Show me all of data/measurements.txt and tell me the
largest radius in it." (a 6-step turn whose second step answered and
called `bash`).

**Files.** `src/tui/transcript.zig`.

### MODL-11 — Muse Glimmer 30B: artifact pin, facts, tokenizer, binding, CPU reference (2026-09-19)

**Outcome.** The third architecture is bound and numerically matched on
the CPU. Session 1: the pulled file and its two companions verified
against their sidecars; the pinned llama.cpp reference rebuilt and shown
to run the file (plain and `--jinja --single-turn`);
`docs/reference/muse-glimmer.md` written from the inventory and the
reference source (metadata, 731 tensors, the forward pass in equations,
the tokenizer, the template facts); the inventory fixture committed; the
`llama4` splitter (`tokenizer/gpt4o.zig`) written and selected by
`encode.zig` from the vocabulary's `pre` label, with `pre.zig` exporting
the shared character classes; the vocabulary loader re-typing
`<|start|>`/`<|message|>` as user-defined like the reference;
`vocabulary-check.zig` selecting the Muse expectations by template
digest ahead of its profile (and accepting Gemma's two EOS ids, a latent
mismatch that made the documented QAT invocation fail);
`scripts/tokenizer-fixtures.py --profile muse_glimmer` and the captured
`muse_glimmer-text.json`; `scripts/reference-split.cpp`, a harness
printing the reference's own `unicode_regex_split` pieces. Session 2:
`models/muse_glimmer.zig` (pinned keys, named binding, typed rejections
over eighteen mutations of the inventory), `muse_glimmer_runtime.zig`
(the CPU schedule: weightless embedding norm, gated attention, sandwich
norms at two epsilons, NoPE globals, the 2048 window as a cache-row
slice, logit scale and soft-cap), an adjacent-pair mode on `cpu.rope`,
a placeholder Metal `Plan` that fails at open, the family in
`models.table` (the adapter tag is `@"muse-glimmer"`), the oracle traces
`tests/fixtures/muse-glimmer-hello-comma/`, and
`make compare-muse-glimmer-cpu`; `nuclis model inspect` says *supported*.

Findings that changed the design: the reference does not run the
declared gpt-4o regex but a rewritten form through its collapsed generic
path, where every letter class is one class, so case is ASCII-only and
combining marks are not letters — the splitter implements that realized
behavior and no Unicode table change was needed; the reference re-types
the two channel markers as user-defined (matched with `parse_special`
off); the server's synthesized system turn carries a `Current date:`
line from its clock, which the profile unit's fixture test must account
for; the file's `,` is token 24, not Qwen's 11.

**Evidence.** `zig build test-vocabulary` on the file: 202,048 tokens,
439,802 merges, all 20 standalone strings and 28 rendered prompts match
the reference's ids (Qwen and both Gemma files still pass). The splitter
unit test pins the pieces of some sixty adversarial strings read from
the harness. `make compare-muse-glimmer-cpu` against the reference's
traces (`<|begin_of_text|>Hello,` = `[200000, 19873, 24]`, Metal, F32
cache): **157 files, max abs 1.53e-4, relative RMS 6.35e-7**, logits
within 6.0e-6, greedy token 372 on both sides with identical top-5, in
1 min 41 s. `make check`: 426 tests and `test-metal`.

**Files.** `inference/src/tokenizer/{gpt4o,pre,encode,vocabulary}.zig`,
`inference/vocabulary-check.zig`, `inference/src/models/{muse_glimmer,muse_glimmer_runtime,muse_glimmer_metal,root}.zig`,
`inference/src/models/fixtures/muse-glimmer-30b.json`,
`inference/src/profiles/fixtures/muse_glimmer-text.json`,
`inference/src/backends/cpu/rope.zig`, `scripts/tokenizer-fixtures.py`,
`scripts/reference-split.cpp`, `tests/fixtures/muse-glimmer-hello-comma/`,
`tests/fixtures/provenance.md`, `Makefile`, `src/catalog.zig`,
`docs/reference/muse-glimmer.md`, `docs/reference/tokenizer.md`,
`docs/reference/new-model-guide.md`, `docs/reference/prompt-profile.md`,
`docs/reference/artifacts.md`, `docs/architecture.md`,
`docs/development.md`, `THIRD_PARTY_NOTICES.md`.

**Remaining.** The Metal plan (MODL-12: the RoPE kernel's adjacent
pairing, the gate epilogue, the window), the profile with the channel
decoder and the acceptance record (MODL-13), ATEM tool calling (AGNT-10);
the CPU decode policy still omits user-defined tokens with `special`
off, where the reference renders them.

### MODL-12 — Muse Glimmer 30B: Metal plan (2026-09-19)

**Outcome.** The third architecture runs on the GPU.
`models/muse_glimmer_metal.zig` replaces the placeholder: the CPU
reference's schedule as encoder calls, one command buffer per token or
per prompt chunk, with the weightless embedding norm bound to a row of
ones, the post norms at their own epsilon, the four attention
projections merged into one segment dispatch (the query packed with
the gate and the scratch key with the value, so the four weights and
two outputs fit the kernel's seven bindings), RoPE on sliding layers
only, the sigmoid gate as the Qwen3.5 epilogue, the 2048 window as a
cache-row slice on decode and the chunk kernel's mask on prefill, and
the untied head scaled then soft-capped. The one backend change is a
`pairing` parameter on `nu_rope` and `nu_rope_rows` (`Backend.Pairing`,
the CPU's enum; the table is shared), with the Qwen and Gemma plans
passing `.split_half`. `generation-check` gained the family's spec
and `test-metal` the adjacent-pairing cases; `make compare-muse-glimmer`
runs the CPU reference and both cache precisions, `make
test-generation-muse-glimmer-metal` the protocol.

**Evidence.** `make compare-muse-glimmer` against the pinned traces
(157 files): Metal F32 **max abs 1.53e-4, relative RMS 8.0e-7** at the
bring-up thresholds, logits within 6.0e-6; Metal F16 7.3e-2 / 1.97e-4
at the family's tolerance of 0.1 / 3e-4 (seventeen late-layer files
above Qwen's 3e-2 bound, the logits within 2.6e-3); greedy 372 with
identical top-5 on every path. The generation check: sessions
bit-identical, cancellation and reset, snapshot 106,496 bytes bit-exact,
chunked prefill 1.39e-2 / 4.5e-3 against 5e-2 / 1e-2 (the F32 tiles at
3.7e-5 / 4.1e-6), the F16 cache 5.0e-3 / 6.0e-4 against 2e-2 / 2e-3,
argmax 75 of 75 everywhere. `test-metal`: adjacent RoPE within 2e-6
relative of `cpu.rope.apply`, the rows form bit-identical. First-look
rates (`nuclis bench`, raw prompt): **9.99 tok/s decode**, 93.2 tok/s
prefill on 512 tokens, against the reference's 14.08 / 101.9 on the
same file (71 % / 91 %); the per-kernel profile is in muse-glimmer.md.
The Qwen `make bench` is unchanged (39.75 / 10.44 tok/s). `make check`:
427 tests and `test-metal`.

**Files.** `inference/src/models/muse_glimmer_metal.zig`,
`inference/src/models/{gemma4_metal,qwen35_metal}.zig`,
`inference/src/backends/metal/{root.zig,kernels.metal}`,
`inference/metal-check.zig`, `inference/generation-check.zig`,
`Makefile`, `docs/reference/muse-glimmer.md`,
`docs/reference/metal-backend.md`, `docs/reference/bench.md`,
`docs/architecture.md`, `docs/development.md`.

**Remaining.** Decode at 71 % of the reference: the Q4_K matvecs read
at 145–175 GB/s against the head's 208, spread over the large matrices
(the performance theme). The full-context cache on sliding layers (a
ring layout is the roadmap's). The profile, the channel decoder, and
the acceptance record are MODL-13; tool calling AGNT-10.

### MODL-13 — Muse Glimmer 30B: profile (text, reasoning channel), catalogue, acceptance (2026-09-19)

**Outcome.** The third family renders, decodes, and is measured.
`profiles/muse_glimmer.zig` implements the pinned template's text
protocol: `<|begin_of_text|>` as text, one system turn per leading
system or developer message (each with the strength and recipients
lines) or the synthesized one without its date line, content verbatim,
reasoning kept as its own `to=self` message, `off` rendered as `low`.
`stream.zig` gained the channel grammar (`Markers.channel`): messages
`HEADER<|message|>BODY` ended by `<|eom|>` or the next `<|start|>`, the
first header arriving as text, the header routing the body to thinking,
the answer, or a tool parser, a header longer than 256 bytes released as
text, a tool body completed on EOS and released as text on any other
stop. The shared `Effort` gained `high`: Qwen's fixture re-captured with
35 cases (its template folds `high` into `xhigh`, pinned by the test),
Gemma treats it as on, and the agent's cycle, help, and configuration
messages list it. The catalogue pins `.muse_glimmer` (the field is
non-optional again), `vocabulary-check` selects the Muse expectations
through the profile, and `nuclis --help` names the profile. The
acceptance tooling learned the family: `reference-baseline.py` checks
the template's default strength line, writes `<|begin_of_text|>`, and
gives the smoke request `low` strength with a 384-token budget;
`nuclis-baseline.py` the same marker; `make baseline-muse-glimmer`.

Findings that changed the design: the reference sends `developer` as
`system` (the facts document claimed the role rendered nothing); the
first acceptance attempt refused the 191 KB corpus, because the
encoder's special-token scan charged every one of Muse's 2,048 reserved
markers per byte and exhausted its 1 GiB budget at 18 KB — the scan now
searches each marker's first byte with the same longest-first
precedence (`fix(tokenizer)`, its own test with 2,048 markers over
64 KiB), and every pinned file's vocabulary check still matches.

**Evidence.** The 28 text fixtures match byte for byte (date line
removed, BOS prepended); `make test-vocabulary` on the Muse file (20
standalone strings, 28 prompts), the Qwen file (20 and 35), and the QAT
Gemma file (20 and 14). The acceptance run
([bench.md](reference/bench.md#muse-glimmer-30b-acceptance-record-modl-13-2026-09-19)):
**9.60 / 8.26 / 7.19 / 6.62 tok/s decode** and **93.49 / 80.51 /
67.83 / 59.20 prefill** at 512 / 4,096 / 16,384 / 32,639 tokens against
the reference's 13.69 / 12.14 / 10.07 / 9.98 and 95.28 / 93.01 / 80.96 /
76.86, every sample at `token_budget` with 128 tokens; the harness
matched nuclis's corpus tokens to the reference's through 32,577
tokens. A live `nuclis agent --model muse-glimmer-30b --think medium
--print --json` turn shows three thinking blocks (32.7 s, 7.0 s, 6.7 s)
around its calls and answers in 460 tokens at 9.3 tok/s. `make check`:
442 tests and `test-metal`.

**Files.** `inference/src/profiles/{muse_glimmer,stream,root,qwen38,gemma4}.zig`,
`inference/src/profiles/fixtures/{qwen38-text,muse_glimmer-tools}.json`,
`inference/src/tokenizer/encode.zig`, `inference/vocabulary-check.zig`,
`src/{catalog,config,help}.zig`, `src/agent/{root,commands}.zig`,
`scripts/{tokenizer-fixtures,reference-baseline,nuclis-baseline,reference-record}.py`,
`Makefile`, `tests/fixtures/run-2026-09-19-muse-glimmer/`,
`docs/benchmarks/{reference,nuclis}-2026-09-19-muse-glimmer.json`,
`README.md`, `TODO.md`, `docs/agent-spec.md`, `docs/development.md`,
`docs/roadmap.md`, `docs/reference/{muse-glimmer,prompt-profile,bench,reference-baseline,tokenizer,generation,artifacts}.md`.

**Remaining.** Decode at 66–71 % of the reference and prefill falling
to 77 % at 32K, and the 4K row's decode drift within one run (the
performance theme); the full-capacity sliding cache (1.63 GiB at 32K);
the Hub's `chat_template.jinja` revision unchecked as an alias; the
vision projector and the DFlash drafter (roadmap).

### AGNT-10 — Muse Glimmer ATEM tool calling: rendering, decoding, fixtures (2026-09-19)

**Outcome.** The profile renders the template's ATEM protocol and the
loop drives it. Declarations go into every system turn as the
template's prose plus one metadata line per namespace and one JSON
schema line per tool in the reference's `tojson` style (insertion-
ordered keys, `", "` and `": "`, control characters escaped, non-ASCII
literal), with the recipients line listing each namespace; a call is
its own `assistant to=NAME` message holding one `<atem:function_calls>`
block (strings verbatim, `true`/`false`/`null`, numbers as written,
lists and objects as JSON), `<|eom|>` between a step's calls and
`<|eot|>` after the last; a result is its own `tool NAME` turn. The
channel decoder hands `to=NAME` bodies to `muse_glimmer.parseTool`
(one well-formed block with one invoke; a value that parses as JSON
keeps its type, anything else is a literal string), completes an open
body when the turn stops on `<|eot|>`, and releases it as text on a
budget stop, a cancellation, or a malformed body. Names cannot carry a
quote, an angle bracket, or whitespace; content, results, names, and
argument strings carrying the markup or a control marker are rejected.
`scripts/profile-tools-fixtures.py --profile muse_glimmer` captures
Gemma's shapes plus a declaration whose texts need JSON escaping and a
multi-line string argument; the fixture was captured while the
reference server was up for MODL-13 and landed in that unit's commit.
[tool-calling.md](reference/tool-calling.md) records the format and
the three facts that shaped the implementation.

**Evidence.** The 52 tool fixtures match byte for byte; the parser
round-trips a rendered call with every value type and refuses nine
malformed, truncated, or multi-invoke bodies; the decoder test delivers
two calls in one turn and releases the second as text under a budget
stop; smuggled markup in arguments, results, names, keys, content, and
tool names is rejected. The live `--print --json` turn on the pulled
file (Metal, context 8,192, `--think medium`): `write_file` then
`read_file` as asked, `greeting.txt` on disk with `hello world`, the
answer quoting the contents, stop `eos` after 460 tokens. `make check`
passes.

**Files.** `inference/src/profiles/muse_glimmer.zig`,
`inference/src/profiles/fixtures/muse_glimmer-tools.json`,
`scripts/profile-tools-fixtures.py`,
`docs/reference/{tool-calling,prompt-profile,muse-glimmer}.md`.

**Remaining.** Values are typed by whether they parse as JSON (Qwen's
rule), not by the declaration as the reference does, so a string-typed
parameter written as `123` or `true` reaches the agent as a number or
a boolean; a body with several invokes is released as text rather than
split into calls; a tool result carrying the ATEM markup (a file that
quotes it) fails the turn instead of rendering, as Gemma's `<|"|>` does.

### ENGN-10 — Muse Glimmer decode gap: the experiments accepted into the performance theme (2026-09-19)

**Outcome.** A roadmap decision, no code. The performance theme gained
a Muse Glimmer bullet written from MODL-12's per-kernel profile and
MODL-13's acceptance record: decode at 66–71 % of the reference is two
thirds an occupancy problem (the matvecs' bandwidth tracks the matrix's
row count — 208 GB/s on the 202,048-row head, 175 on the 39,936-row FFN
pair, 145–147 on the 6,656- to 8,704-row shapes that hold more than a
third of the bytes) and one third launch count (922 dispatches per
token, six RMS-norm launches per layer). Three experiments in order:
split-K on the decode matvec for narrow-row shapes, fusing the post
norms into the residual add and the head norms into the RoPE launch,
and the 2-KV-head flash-decoding split pass at long context; the 4K
row's decode drift is to be reproduced first.

**Evidence.** The profile and the record cited in the bullet; nothing
measured anew.

**Files.** `docs/roadmap.md`.

**Remaining.** The experiments themselves, when the performance theme
is planned after speculative decoding.

### KERN-11 — Small-chunk prefill matmul near the weight-bandwidth floor (2026-09-19, two sessions)

**Outcome.** A 16-row × 8-token split-K prefill tile (`nu_matmul_*_8`, one per
specialized encoding) serves chunks of at most `small_chunk_tokens` (24)
tokens, the 32×32 tile the rest of the range to 32 and the 64×64 tile beyond.
The four SIMD groups partition the 64-column K steps (`sg` takes steps sg,
sg+4, …); each lane decodes two 16-value segments into its own group's 16×64
half tile, two 8×8 accumulators per group share one activation block per step,
so one B load serves two weight loads; `simdgroup_barrier` orders write and
read (no `threadgroup_barrier` in the loop) and the four K partials reduce once
in a fixed order. `matmulGeometry` returns 16×8 half and `specializedMatmul`
gains the first tier. `matmulBench` now batches 64 (t ≤ 8) or 16 dispatches
per command buffer, divides the GPU time by the count, and attributes weight
bytes per token tile — the session-1 table measured the clock ramp and is kept
marked as such.

**Evidence.** `test-metal`: the half tile within the set's bound at 1, 5, 8, 9,
and 16 tokens for every fixture encoding (worst |Δ|/Σ|w·x| 3.02e-5, bound
2e-4), the 24/25 selection pinned, chunked prefill unchanged. `make check`,
`make compare` (f32 max abs 6.1e-5, rel RMS 7.7e-7; f16 2.5e-2 / 1.9e-4),
`make test-generation-metal` (chunk 32 max abs 2.6e-3, argmax equal), and the
Gemma QAT, Gemma K-quant, Bonsai, and Muse compare targets pass. `make
bench-matmul` at 1–32 tokens on both FFN shapes (batched, every encoding): the
tile streams 32–116 GB/s, 37–64 % of the same encoding's matvec rate — **the
design's 70 % floor is not met**. The gap is the activation operand: a group
gathers 8 tokens × columns × 4 B (160 KB at 5,120 columns) against 16 × columns
of weights (46 KB of Q4_K). The 16-row geometry was kept from the two
activation experiments (Q4_K 75 → 95 GB/s at 8 tokens, every encoding up 4–32 %);
the packed `[k][token]` half layout measured 62–67 GB/s before its one-time
pack cost (34–112 µs per input) and was dropped. `make bench` (22-token
prompt): prefill 38.78 → 43.01 tok/s, first token 567.3 → 511.5 ms, decode
10.22 → 10.20 (unchanged).

**Files.** `inference/src/backends/metal/kernels.metal`, `root.zig`,
`inference/metal-check.zig`, `docs/reference/metal-backend.md`,
`docs/reference/bench.md`, `docs/roadmap.md`.

**Remaining.** The tile is 37–64 % of the matvec rate, not 70 %. Levers not
shipped: 32 rows per threadgroup, a blocked activation read that makes an 8×8 B
block one contiguous 128-byte load, or holding the activations in threadgroup
memory across a row strip. The tile is 8 tokens, so a 9–16 row verify batch
spans two or three token tiles; ENGN-12 sets `max_draft_length` to 7 unless a
16-token tile is added (the design's `_16` was not asked for: the 8-token tile
is not at the floor and a 16-token tile re-reads no fewer weights per token).
The session-1 race (the four partials stored into a shared region) is fixed and
is checked only by `test-generation-metal`, not the fixture exactness test.

### ENGN-11 — Speculative state recovery: checkpoint, rewind, truncate, recover (2026-09-19, one session)

**Outcome.** The session can undo one verify batch without a host snapshot.
`Session.init` gains a `want_checkpoint: bool`; when set the block grows by a
page-aligned region, after the layer regions, holding one copy of every
recurrent layer's history and matrix in layer order (zero bytes and a still
valid position on attention-only layouts). `checkpoint()` copies the recurrent
state and records the position (`NoCheckpointRegion` without a region),
`rewind()` restores it (`NoCheckpoint` when none is recorded),
`truncate(position)` moves the position for attention-only layouts
(`RecurrentStateNotRewindable` on a recurrent one) within
`[checkpoint, position]` (`RewindOutOfRange`). `reset` and `restore` clear the
recorded position; the region stays out of `layout_digest` and `snapshot`. The
region is a slice of the same block the Metal plans wrap, so it is GPU-visible
with no new binding. `Runtime.init`/`Plan.init` (all three families) forward the
flag, and `engine.Model.checkpoint/rewind/truncate` and `hasRecurrentState`
forward on both executors; `Model.recover(accepted)` rewinds and replays the
accepted prefix on a recurrent model, truncating to `checkpoint + accepted.len`
otherwise. CPU `Executor.prefill` now admits the whole batch before the first
write (`ContextFull`, position unchanged), matching the Metal plans.
`generation-check` gains a `recoveryCheck` on every family and both backends:
it steps a header token, checkpoints, consumes a 4-row batch (and 8 on Metal),
then for every accepted length replays and steps, comparing against a second
executor that stepped the same tokens one by one — plus the refusals (rewind
without a checkpoint, truncate on a recurrent layout, a cancelled batch
poisoning the session, an over-capacity batch). Session 2 was not needed: Metal
ordering is correct and replay is a short-chunk prefill.

**Evidence.** `make check` (fmt, unit tests, Metal fixtures) passes; the new
session unit tests cover the round trip (under every allocation failure), the
region offset and `bytes()` accounting, checkpoint/rewind on the session state
machine, the range refusals, attention-only truncate, and a rewound session
leaving a peer untouched. `make test-generation` (CPU, Qwen 27B): the replay
equals sequential decoding bit for bit at every accepted length; region
156,893,184 bytes, checkpoint 2 ms, rewind 2 ms. `make test-generation-metal`
(Qwen 27B): a 4-row batch within 2.851e-3 max abs / 1.476e-4 relative RMS and
an 8-row batch within 2.244e-3 / 9.592e-5, argmax equal at every accepted
length (bounds 2e-2 / 1e-3); checkpoint 3 ms, rewind 3 ms. Bonsai 2 (Metal): a
recurrent replay within 8.583e-6 / 4.306e-7, checkpoint/rewind 2 ms. Gemma 4
12B QAT and Muse Glimmer (Metal, attention-only): region 0 bytes of state,
checkpoint/rewind free, batch and sequential logits identical at the small
tiles (bounds 6e-1 / 2e-2 and 5e-2 / 1e-2).

**Files.** `inference/src/runtime/session.zig`, `inference/src/engine.zig`,
`inference/{generation-check.zig,src/models/{qwen35,gemma4,muse_glimmer}_{runtime,metal}.zig}`,
`docs/reference/{session,speculative-decoding}.md`, `docs/architecture.md`,
`docs/engineering-log.md`, `TODO.md`.

**Remaining.** `Engine.open` still sizes no region (`openExecutor` passes
`false`): the drafter request that enables it lands with MODL-18, and ENGN-12
is `recover`'s first caller. Per-token recurrent checkpoints written by the
DeltaNet chunk kernel were not built: the measured replay cost (one 150 MB copy
plus an `a + 1`-token prefill) did not argue for the `(k + 1) × 150 MB` of
device scratch. The CPU recovery pass on Qwen runs about 20 minutes; the Metal
recovery check for a 4- and an 8-row batch is the cheap one.

### MODL-18 — Qwen3.8 draft head: the embedded prediction block on the CPU reference and the Metal plan (2026-09-19 / 2026-09-20, two sessions)

**Outcome.** The Qwen3.8 release's 65th block is the family's draft source:
the 15 embedded `nextn` tensors (351,008,768 B) around one dense
full-attention layer of the main shape. Session 1 read the pinned
reference's graph and driver and recorded the facts with provenance; session
2 ran the block on Metal and measured it. `Binding.draft: ?DraftBlock` binds
the block (null on Bonsai); `runtime/draft.zig` is the model-independent
contract (`propose`/`commit`/`reset`/`bytes`; no `rewind`, because the
block's cache is one more `Session` layout that checkpoint/rewind already
cover); `Engine.open`'s `DraftRequest.embedded` sizes the 65-layout session
and builds a drafter on both executors; `nuclis validate` reports *draft
head: embedded*. The block consumes `[enorm(embed(x_p)); hnorm(h_{p-1})]`
projected by `eh_proj`, runs the layer over its own cache at the main token
position, and heads through `shared_head_norm` and the shared `output`. The
CPU reference keeps the target hidden (`self.h`, post-`output_norm`) and
chains the block's own hidden; the Metal plan runs the same kernels (two
copies into a 10,240-wide buffer plus an `eh_proj` matvec, the full-attention
path over a 65th layout, the shared-head matvec, device argmax). The draft is
only ever run when asked: `runLoop` still calls no drafter until ENGN-12.

`generation-check` gained the checks and diagnostics: `checkDraft` (CPU and
Metal, F32 and F16 caches) against the pinned trace; `draftRecoveryCheck` on
both executors; a `--draft-trace DIRECTORY` writer of the native
`p0-h`/`p1-h`/`p1-hprev`/`greedy.txt` rows; `--draft-stats` (the
`make draft-stats` target) for the per-depth acceptance and latency;
`compare-generation.py --draft` and the `make compare-draft` targets.

**Evidence.** Session 1 (`make test-generation`, CPU, Qwen 27B): positions 0
and 1 of `Hello,` match the pinned reference at 7.2e-6 and 1.1e-5 max abs
(3.1e-7 / 4.9e-7 relative RMS), greedy tokens 9419 and 271 equal. Session 2
also ran the new checks on the CPU reference: the same block rows match,
`draftRecoveryCheck` passes, and a draft-loaded session decodes the two pinned
tokens identically to the no-drafter session. The CPU run's long tail is the
pre-existing accepted-prefix replay pass (about 20 minutes on Qwen; ENGN-11),
which the Metal run covers in seconds, so the full CPU recovery pass was not
re-waited on. Session 2 (`make test-generation-metal`): the Metal F32 rows
match at 1.5e-5 / 5.8e-7 (position 0) and 1.5e-5 / 7.2e-7 (position 1); the
F16 cache rows are at 8.5e-3 / 3.1e-4 and 2.5e-3 / 2.4e-4, within the family's
recorded `half_*` bound; a draft-loaded session never asked to propose decodes
the two pinned tokens identically to the no-drafter session; the main recovery
and prefill checks are unchanged (recovery 4 and 8 rows within 2.9e-3 / 1.5e-4,
prefill within 2.6e-3 / 1.3e-4, F16 within 5.5e-4 / 2.4e-5). `make compare-draft-metal` and
`-cpu` pass all three native rows against
`inference/src/models/fixtures/qwen35-mtp/` (Metal 1.5e-5 max abs / 7.9e-7
relative RMS, greedy `[9419, 271]`). `make draft-stats` (Metal): depth 0-3 at
28/31, 24/30, 20/29, 18/28 (90.3 / 80.0 / 69.0 / 64.3 %) and 29/31, 25/30,
24/29, 24/28 (93.5 / 83.3 / 82.8 / 85.7 %) on the two fixed prompts; 128
drafts in 797.1 and 798.5 ms (6.2 ms per position, about 1.6 ms per block
forward), block workspace 1,116,160 B; the reference's own driver accepts
24/40 (60 %) on the first prompt, so the rates are credible. `make check`
passes (204 unit tests and the Metal fixtures); `make compare`'s main rows are
unchanged (129 files, 6.1e-5 max abs F32, 2.5e-2 F16).

**Files.** `inference/src/runtime/draft.zig`,
`inference/src/models/qwen35_metal.zig`, `inference/src/models/qwen35.zig`,
`inference/src/engine.zig`, `inference/generation-check.zig`,
`scripts/compare-generation.py`, `Makefile`,
`docs/reference/{speculative-decoding,generation,session}.md`,
`docs/architecture.md`, `docs/engineering-log.md`, `TODO.md`.

**Remaining.** The pinned trace has 2 positions, not the plan's 3: the
reference harness captures one MTP row per prompt token and `Hello,` has two;
the acceptance statistic now exercises 64 positions instead. The "decode rate
in `make bench` unchanged with the drafter loaded but switched off" check has
no caller until ENGN-12 adds the load switch and is folded into ENGN-12's
acceptance. The separate `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` stays pinned but
unloaded: the embedded block is the source (the log records why).
`DraftRequest.file` and the Gemma/Muse sources remain MODL-19/20; sampled
acceptance and batched verification are ENGN-12, so this unit measures
drafting quality only, not a speedup.

### APPS-13 — The configuration section is `generation`, not `generate` (2026-09-20)

**Outcome.** The shared generation scope of `nuclis.json` was renamed from
`generate` to `generation`. The section was named after a command while it
holds settings every generation path shares: `max_tokens`, `think`,
`speculative`, `draft_length`, and `sampling` all apply to `agent` as well as
`generate`, and `bench` reads only `engine` plus its own flags. The scope rule
is now explicit — `engine` and `generation` are shared, `agent` is the chat
surface's own — and per-model entries mirror it
(`models.<name>.generation.*`). A decision recorded here, not a compatibility
shim: the tree is pre-1.0 and the schema is internal.

**Evidence.** `make check` (451 unit tests and the Metal fixtures) with every
config path, test needle, and help line on `generation.*`; `config init`
writes the full schema under the new key and `config show` prints it with the
same sources; a hand-written file with a custom `models` map loads and
resolves unchanged.

**Files.** `src/config.zig`, `src/cli.zig`, `src/help.zig`,
`docs/development.md`, `docs/reference/generation.md`, `docs/spec.md`,
`docs/llm-guide.md`, `docs/engineering-log.md`, `TODO.md`.

### REPO-06 — DiffusionGemma structured reads and kev research (2026-09-20)

**Outcome.** Enriched the TypeSafe/Jev research with the DiffusionGemma
artifact and runtime requirements, pinned open llama.cpp/vLLM proposals,
the no-training structured-read experiment, and kev's separate trained
pointer-head approach. Distinguished typed validity, question isolation,
label conditioning, seed sensitivity and empirical calibration. Recorded a
proposed model-first sequence and common evaluation gates without accepting
implementation work or changing the active speculative-decoding sequence.

**Evidence.** Read official Google architecture/configuration, Unsloth's
card and repository metadata, the llama.cpp graph at `12e0a9627d02`, vLLM
PR body/diff/discussion at `ceb8eebf3eed`, and kev README/model at
`20fa6268c8ce`; permanent source links are in the research note. Checked
local Markdown targets and `git diff --check`. Documentation only; no model
execution, benchmarks or build/tests performed.

**Files.** `docs/research/typesafe-jev.md`, `docs/engineering-log.md`,
`TODO.md`.

**Remaining.** GGUF inventory and digest verification, CPU/Metal numerical
parity, local memory/performance measurements and held-out decision quality
are future experiments. Public sources do not disclose Jev's architecture
or training recipe; neither comparison establishes equivalent calibration.

### ENGN-12 — Batched verification, speculative generation (greedy and sampled), the switch and the draft length, the benchmark record (2026-09-20, two sessions)

**Outcome.** Speculative generation runs end to end on Qwen3.8-27B with its
embedded draft head, on both executors, behind a switch, and is measured.
Session 1: `Model.verify`/`verifyGreedy` and the Qwen Metal
`Plan.verify`/`verifyGreedy` (the layer stack once, the output head over
every row of the batch, `max_verify_rows` 16; the greedy sibling reads back
one argmax per row), the `runLoop` speculative step (the prompt committed
to the drafter in verify-sized chunks, then propose / checkpoint / verify /
accept / recover / commit with the correction carried as the next seed and
the accepted tokens passed through the ordinary per-token checks), and the
sampled module `inference/src/sampling/speculative.zig`. Session 2: the
`generation.speculative` / `generation.draft_length` configuration with
entry overrides and `--speculative on|off` / `--draft-length N` on
`generate`, `agent`, and `bench` (`max_draft_length` 7); the load switch
(`DraftRequest.embedded` required by `generate`/`agent`, `DraftSourceMissing`
on a family without a block, `optional_embedded` for `bench`); the bench
off/on pair with its sample fields; the loop edge tests; the
full-acceptance skip in `Model.recover`; the `generate` → `generation`
rename (APPS-13). A review before the close found and fixed two defects:
the speculative batch caught `error.Canceled` while the interrupt raises
`error.Cancelled`, so Ctrl-C during a batch escaped the loop (the harness
injected the wrong spelling too); and the sampled rule was not
distribution-preserving — the drafters chain greedy candidates, and
`min(1, p/q)` with a residual correction is exact only for drafts sampled
from `q` — so each verify row now draws the target's own token from its
shaped distribution and accepts the draft when they agree, exact for any
drafting policy; the draft contract lost its logits rows, `Timing` gained
`accept`, and the bench sample gained `accept_milliseconds` and
`speculative_steps`. The record (`make speculative-record`,
`scripts/nuclis-speculative.py`) decides the entry: the switch stays off,
`draft_length` 4.

**Evidence.** `make check` (450 tests and the Metal fixtures, including
the seeded sampled tests: emitted counts and the acceptance rate within 3 σ
of `p` for a draft the target never emits and for its likeliest token);
`make speculative-check` (CPU) and `make speculative-check-metal`: 12
greedy tokens identical to ordinary greedy through the primitives and
through `engine.runLoop` (7/24 and 6/21 drafts accepted on `Hello,`), the
budget, EOS, cancellation, and context-limit edges; `make compare` and
`make test-generation-metal` green; a sampled speculative run with the
instruct profile's sampling produces coherent code (36 of 48 tokens from
accepted drafts), and a real SIGINT during a speculative decode ends with
"generation cancelled after 383 tokens" and exit 0. The record
([bench.md § Speculative decoding record](reference/bench.md#speculative-decoding-record-engn-12-2026-09-20),
[benchmarks/speculative-2026-09-20/](benchmarks/speculative-2026-09-20/)),
twelve off/on configurations at `3d5cb94`: greedy speculation at
0.56–0.67× on the 512-token corpus prompt, 0.85–0.88× on the code prompt
at draft 4 and 7, 0.54× at 4K; the instruct sampling 0.63–0.67× on prose
and 1.04× on code (its baseline pays the penalty readback); per batch:
verify 225–250 ms at 512 and ≈ 342 ms at 4K, recover 99–272 ms on
rejection, the sampled decision 46–78 ms, acceptance 1.2–3.0 drafts;
speculative prefill 2.9–3.3× the ordinary one. The decode rate with the
drafter loaded but off is 10.43 tok/s at 512 against the 2026-09-10
record's 10.62, within the sequence's own drift.

**Files.** `inference/src/engine.zig`, `inference/src/sampling/{root,speculative}.zig`,
`inference/src/runtime/draft.zig`,
`inference/src/models/qwen35_{runtime,metal}.zig`,
`inference/generation-check.zig`, `src/{bench,cli,config,generate,help}.zig`,
`src/agent/{loop,root}.zig`, `scripts/nuclis-speculative.py`, `Makefile`,
`docs/benchmarks/speculative-2026-09-20/`,
`docs/reference/{speculative-decoding,generation,bench}.md`,
`docs/{spec,development,roadmap}.md`, `docs/engineering-log.md`, `TODO.md`.

**Remaining.** No speedup: every per-batch cost is fixed and too high for
the tokens a batch advances, and the plan's performance units carry the
targets (the prompt and drafter commit at the plan's chunk, a multi-row
matvec for 2–8 rows, recovery without the whole-stack replay, the proposal
policy, the GPU penalty kernel, the sampled readback, the verdict). The
agent's primed prefix is never committed to the drafter (`Completer.prime`
uses the ordinary prefill), so its acceptance rate is below the bench's
until ENGN-13; after an EOS or budget inside a batch the drafter has been
advanced past the session's emitted prefix, which self-heals at the next
prompt commit and is documented, not fixed. `bench` cannot measure a
baseline without the drafter loaded (ENGN-17). The 16K and 32,639 rows were
not run: the prefill path makes them minutes long and the 4K row already
shows the trend. Draft length 7 was never worse than 4 in the record; the
default waits for the proposal policy.

### REPO-07 — The roadmap file retired; themes are agreed in session and, when architectural, recorded as ADRs on request (2026-09-20)

**Outcome.** `docs/roadmap.md` is deleted. Its role, the queue of accepted
themes not yet planned, no longer exists as a file: what comes next is
agreed with the user in session and written straight into `TODO.md`, and a
theme that changes the architecture and spans several units is recorded as
an ADR under `docs/adr/` only when the user asks for one; ordinary units
need no record beyond the plan and the log. `AGENTS.md`'s session protocol
(two state files, the empty-plan state) and `docs/README.md` say so. Every
reference to the roadmap outside this log now points at the plan's unit
that carries the item (KERN-13, KERN-14, KERN-15, KERN-16, ENGN-18,
ENGN-19, MODL-20, MODL-23) or at the closed unit it became (MODL-10); four
code comments were reworded. The roadmap's content had already moved into
`TODO.md` on 2026-09-20; HTTP serving on the agent loop is not planned, and
`docs/agent-spec.md` keeps the contract an endpoint would reuse.

**Evidence.** No reference to `roadmap.md` remains outside this log and the
dated research notes (`grep -rn roadmap` over the tree); `zig fmt --check`
on the touched sources (comment edits only).

**Files.** `docs/roadmap.md` (deleted), `AGENTS.md`, `docs/README.md`,
`docs/agent-spec.md`, `docs/llm-guide.md`, `docs/reference/bench.md`,
`docs/reference/gemma4.md`, `docs/reference/muse-glimmer.md`,
`docs/reference/tool-calling.md`, `docs/reference/metal-backend.md`,
`TODO.md`, `src/catalog.zig`, `inference/src/formats/gguf.zig`,
`inference/src/models/muse_glimmer_metal.zig`,
`inference/src/models/gemma4_metal.zig`, `docs/engineering-log.md`.

**Remaining.** None; the ADR template stays as it was.

### ENGN-13 — Prompt commit at the plan's chunk and the batched drafter commit (2026-09-20)

**Outcome.** The speculative fixed costs the ENGN-12 record blamed for the
missing speedup are gone: the prompt is committed to the drafter at the
plan's own chunk (256 tokens), and an accepted prefix is committed in one
batched forward instead of one per token. `Plan.prefill` gained a
`hidden_rows` readback — every row's post-`output_norm` hidden, the rows the
drafter's `commit` consumes — and `Executor.prefill`/`Model.prefill` forward
it; Gemma 4 and Muse refuse it (`error.HiddenUnsupported`) and the CPU
reference copies `Runtime.h`. `runLoop`'s speculative prompt branch is now
`engine.commitPrompt`: chunks of `prefill_chunk` through `prefill` (which
also admits the session and reads the last chunk's logits), then
`commitDraft`; `SpeculativeScratch.hidden` grew to `prefill_chunk × hidden`.
`Plan.commit` over two or more tokens is one command buffer — embed, the
`[enorm; hnorm]` pair into a `padded × 10240` buffer, `eh_proj`, the block's
attention chunk at the batch's cache positions, the FFN (the head norm is
skipped) — reusing the chunk activation buffers; a single token keeps the
standalone `draftForward`, and the CPU `Runtime.commit` stays per token as
the reference. `attentionChunk` takes an explicit position. `Completer.prime`
commits the primed prefix through `commitPrompt` when a drafter is loaded, so
the primed snapshot carries the block's cache rows and the agent's first turn
no longer attends over prefix rows the block never wrote.
`engine.Timing` separates `propose` and `commit`; `bench.Sample` carries
`propose_milliseconds` and `commit_milliseconds`; the agent shows the turn's
mean accepted drafts per batch as `spec N.NN/step`.

**Evidence.** At revision `9a5d3cf`, draft 4, F16 KV, ctx 32768, on the
pinned Qwen 27B: speculative prefill 1.02× ordinary at 512 (5,939.4 ms
against 5,807.8, three runs; the old path measured 2.85× and the serial
prompt commit 1.59×) and 1.05× at 4,096 (54,399 against 51,831 ms, two
runs); `commit_milliseconds / speculative_steps` 5.4 ms at 512 (the target
was ≤ 8 ms) and 10.8 ms at 4K; propose 25.7 ms, verify 236.5 ms, recover
185.5 ms per batch at 512; accepted 1.667 drafts/batch at 512 and 1.977 at
4K. `make draft-stats` reproduced MODL-18 exactly (28/31, 24/30, 20/29,
18/28 and 29/31, 25/30, 24/29, 24/28). `make speculative-check-metal` was
unchanged (12 tokens identical; 7/24 and 6/21 accepted) and the new
16-token case passes (batched versus serial commit: 4 drafts identical,
block logits 4.2e-4 max abs / 3.1e-5 relative RMS, bounds 2e-2 / 1e-3);
`make speculative-check` passed on the CPU reference (12 tokens identical);
`make compare-draft-metal` unchanged (3 rows, 1.5e-5 / 7.9e-7, greedy
9419/271); `make check` (450 tests), `make compare` (f32 6.1e-5 / 7.7e-7,
f16 2.5e-2 / 1.9e-4), and `make test-generation-metal` are green. On a
bounded pty at `ccb191b`, the first turn of `nuclis agent --speculative on`
on `Write a one-line Python function that doubles a number.` (ctx 4096)
settled at `spec 2.56/step`, inside the range bench measures on the code
prompt.

**Files.** `inference/src/models/qwen35_metal.zig`,
`inference/src/engine.zig`, `inference/src/models/{gemma4,muse_glimmer}_metal.zig`,
`inference/generation-check.zig`, `src/agent/loop.zig`,
`src/tui/{status,event}.zig`, `src/bench.zig`,
`docs/reference/{speculative-decoding,bench}.md`,
`docs/engineering-log.md`, `TODO.md`.

**Remaining.** The 4K commit is 10.8 ms per batch, above the 512 target: the
16-row commit still runs its matmuls through the 16×8 tile and its attention
through the chunk kernel over the visible cache, which KERN-12 and KERN-15
lower. `Plan.commit` sub-chunks internally when the plan's chunk is smaller
than the prompt chunk (small contexts); the batched path is not itself
measured against KERN-12's multi-row matvec yet. `bench` still cannot
measure a baseline without the drafter loaded (ENGN-17).

### KERN-12 — A multi-row matvec for 2–8 rows: the 2-row routing (2026-09-20, two sessions, closed below its target)

**Outcome.** A specialized multi-row matvec (`nu_matvec_rows_*_t2` … `_t8`,
Q4_K/Q5_K/Q6_K/IQ4_XS) reads each weight slice once and multiplies it against
every activation row of a batch; a generic `nu_matvec_rows` serves the other
encodings. Each SIMD group owns four output rows (the matvec's `pair`/`half`
mapping over the 256-value block), the token loop is innermost, and the token
count is a template parameter so every `acc[row][token]` index is constant —
a runtime token bound moves the accumulators to thread-local memory and fell
to 4 GB/s at 8 rows in session 1. `Backend.matmul` routes 2-row batches of the
specialized encodings to the kernel (`small_batch_rows = 2`,
`route_small_batch`); `matvec_rows_max = 8` is the kernel range the sweep
covers, the tile keeps 3–24. Session 2 rewrote the four bodies after session
1's scalar form spilled: extracted the per-row scale products, `#pragma unroll`
on the row and token loops, and per-encoding, per-count host names.

**Finding.** The register-tiled body wins at **2 rows** (143–182 GB/s of
weight bytes against the 16×8 tile's 88–116) and for Q6_K/IQ4_XS at 3, but at
5 rows streams 49–67 and at 8 rows 20–33, below the tile. The unit's
acceptance (≥150 GB/s at 5, ≥120 at 8) is **not met**, so the verify batch
stays on the tile. The wall is scalar-FMA and per-(row, token) input-load
issue, not weight traffic: a probe replacing the per-token input offset with a
constant (the compiler eliminates the loads) measured **181 GB/s flat from 2
to 8 rows**, and every layout that shares a token's input across rows needs
128–272 registers and spills. The reference-style one-row-per-lane-group body
was also measured and rejected: its 16-value segment decode reads the block
header twice as often as the matvec's 32-value slice.

**Evidence.** `make bench-matvec-rows` (Apple M4 Pro, Zig 0.16.0, ReleaseSafe,
two FFN shapes 17,408×5,120 and 5,120×17,408, three measured command buffers
after a warm-up, 16 dispatches each) is the table in
[metal-backend.md § Multi-row matvec](reference/metal-backend.md#multi-row-matvec-kern-12-2026-09-20-closed-below-its-target):
Q4_K 143/128, 86/84, 53/52, 28/27 at 2/3/5/8 rows (gate/down); Q5_K 148/137,
98/92, 61/60, 30/28; Q6_K 182/179, 125/121, 67/63, 33/31; IQ4_XS 172/154,
109/94, 65/49, 23/20. `make test-metal` exactness at 2/5/8 rows for every
encoding, worst `|Δ|/Σ|w·x|` 5.64e-8 (bound 4e-6), the 1-row refusal, and the
selection pinned (`small_batch_rows == 2`, `matvec_rows_max == 8`). `make
compare` unchanged (f32 6.1e-5 / 7.7e-7, f16 2.5e-2 / 1.9e-4), `make
test-generation-metal` green at every accepted length (4- and 8-row recovery
max abs 0.0), `make speculative-check-metal` green (12 tokens, 7/24 and 6/21
accepted, every loop edge), `make draft-stats` reproduces MODL-18 (28/31,
24/30, 20/29, 18/28; 29/31, 25/30, 24/29, 24/28). A repeat-1 spot run at the
close revision (512 prose, draft 4, F16 KV, ctx 32768) measured
`verify_milliseconds / speculative_steps` 232–288 ms and
`recover_milliseconds / speculative_steps` 170–217 ms, in the record's range
for verify.

**Files.** `inference/src/backends/metal/kernels.metal`,
`inference/src/backends/metal/root.zig`, `inference/metal-check.zig`,
`docs/reference/{metal-backend,speculative-decoding,bench}.md`,
`docs/engineering-log.md`, `TODO.md`.

**Remaining.** The verify (5 rows) and the commit beyond 2 rows stay on the
16×8 tile; the ≤ 130 ms verify and ≤ 110 ms 2-row replay targets are unmet.
The ≤ 110 ms replay is left to ENGN-14: per-row recurrent checkpoints remove
the replay rather than accelerate it, and a full stack pass dominates the
recover time regardless of the matvec. If the tile itself is to move, KERN-11's
unfinished levers (32 rows per threadgroup, a blocked activation read, or
holding activations in threadgroup across a row strip) are the path, not a
scalar body; the 16×8 tile is already flat at 88–116 GB/s because it uses
half-precision matrix units.


### REPO-08 — Repair the multi-row benchmark controls and hand-off (2026-09-20)

**Outcome.** The multi-row sweep uses an explicit `Backend.matmulTile` control,
sharing validation and tile dispatch with production `matmul` but bypassing
small-batch routing. Previously both columns selected the scalar kernel at two
tokens. Head allocation and selection now use the actual row stride instead of
`1`, which failed every specialized alignment check and silently skipped the head.
A profiler fixture pins the actual control and production kernel identities.

**Evidence.** `make bench-matvec-rows ARGS="2 head"` on Apple M4 Pro 48 GiB,
Zig 0.16.0, ReleaseSafe, `27303ed` plus this repair: all four specialized
encodings beat the tile on both FFN shapes and the head (twelve cases). FFN
127.7–181.9 vs 87.7–116.2 GB/s; head 146.8–188.0 vs 89.2–114.1.
[Method and table](reference/metal-backend.md#corrected-two-row-control-repo-08-2026-09-20).
Formatting and 450 CPU tests passed via `make check`; its Metal stage could not
access the device in the sandbox, then `make test-metal` passed outside it,
including the dispatch regression and 2/5/8-row error 5.64e-8 (bound 4e-6).
`make build` and the fresh `./zig-out/bin/nuclis --version` passed; current
document link targets and `git diff --check` passed.

**Corrections to KERN-12's interpretation.** The constant-input probe can remove
arithmetic and accumulator state as well as loads; it does not isolate load issue
or prove every scalar layout incapable of the target. The tested bodies miss the
original target, and two-row routing remains the measured deliverable. The
reduction is whole-SIMD `simd_sum`, not row-group shuffles. Aggregate recovery
milliseconds per speculative step cannot establish two-row replay latency.
The original log entry is retained as historical evidence; current references
and TODO carry these corrections.

**Files.** `inference/src/backends/metal/root.zig`, `inference/metal-check.zig`,
`docs/reference/{metal-backend,speculative-decoding,bench}.md`, `TODO.md`, and
this log. ENGN-14 is next; its hand-off specifies recovery timings by length
and an immediate record refresh. The separate small-batch matrix-tile experiment
is conditional on that record; ENGN-17 still owns final defaults.

**Remaining.** No new full-model or 3–8-token performance record in this repair.
The original five/eight-row bandwidth and verify targets remain unmet; two-row
recovery latency must be measured separately by ENGN-14. No scalar kernel or
production routing threshold changed.

### ENGN-14 — Recovery without the whole-stack replay (2026-09-20, two sessions)

**Outcome.** A verify batch now leaves the recurrent state after every row
behind, so recovery copies the accepted row instead of rewinding and replaying
it. Session 1 measured the replay per accepted length (the aggregate hid it)
and moved the seed-only case to the per-token `step` path: 218 → 96 ms per
call, lengths 2–4 unchanged. Session 2 added `Session.row_checkpoints` (eight
page-aligned slots of one 156,893,184-byte recurrent copy), `restoreRow`,
`rowSlotLayer`, and the refusals; `nu_delta_chunk` writes each row's state and
`nu_convolution_history` each row's history, both only when the verifier
passes `row_states > 0`, so ordinary prefill, decode, and the prompt commit
pass 0 and are unchanged; `Plan.verify`/`verifyGreedy` pass the batch count
and mark the slots; `Model.recover` restores `accepted.len - 1` and replays
nothing. `Timing` gained `recover_rewind`/`recover_replay`/
`recover_by_length`/`checkpoint`, exposed per sample by `bench`.

**The overflow found and fixed.** The first kernel form rescaled the
full-chunk `W` (base `cum[n−1]`) by `r(r, n−1)` to get prefix `r`; when a
layer's chunk decay is large — measured `cum = −114` on one Qwen layer — the
scale is `exp(97) = inf`, the slot fills with `inf`, and `0 · inf` turns the
restored state into NaNs (`NonFiniteResult` at token 58 of the first record
attempt). The shipped form builds each prefix's rescaled `U` rows directly
from the threadgroup's solved `U`, where every exponent `cum[r] − cum[s] ≤ 0`.
A second bug, `restoreRow` reading only one layer copy per slot, was caught by
the backend's bounds validation before any measurement.

**Evidence.** The full record (12 configurations, 46 minutes, 20:49–21:35,
`417aea0`, `nuclis 0.2.0-dev`, Apple M4 Pro 48 GiB, macOS 26.6.2, artifact
SHA-256 `322e194f…`, one loaded model, nothing else on the GPU;
[bench.md § Recovery record](reference/bench.md#recovery-by-row-checkpoints-engn-14-2026-09-20),
reports under
[benchmarks/speculative-2026-09-20-recovery/](benchmarks/speculative-2026-09-20-recovery/)).
`recover` fell from 150–182 ms to **6–22 ms per batch** at every accepted
length (all of it the slot copy; replay zero), the ≤ 40 ms target met;
`checkpoint` 3–5 ms; the slot writes add 21 ms per verify batch (`verify`
241–372 ms, now 70–80 % of the batch); `session_bytes` 3,850,633,216 (row
region 1,255,146,752). Speedups: code greedy 1.22× (draft 4) and 1.34×
(draft 7), code instruct 1.23×; prose 0.75–1.04× by draft length; prose 4K
0.74–0.79×. `make check`, `make compare` (f32 6.1e-5 / 7.7e-7, f16 2.5e-2 /
1.9e-4), `make test-metal` (every slot against the CPU's sequential state,
7.7e-7 worst; the per-row history exact), `make test-generation-metal`
(restore-by-slot exact against the CPU replay at 4 and 8 rows, refusals
included), `make speculative-check-metal`, and `make draft-stats` (28/31,
24/30, 20/29, 18/28 and 29/31, 25/30, 24/29, 24/28) passed. `make
speculative-check` (CPU) was deferred to the next unit that needs it: the CPU
reference is unchanged (`row_checkpoints == 0`, `Model.recover` takes the
step/prefill replay branch exactly as before, and `Session.init` adds only a
zero-sized region there).

**Files.** `inference/src/runtime/session.zig`,
`inference/src/backends/metal/kernels.metal`,
`inference/src/backends/metal/root.zig`,
`inference/src/models/qwen35_metal.zig` (and the `Session.init` signature in
the other families), `inference/src/engine.zig`,
`inference/generation-check.zig`, `inference/metal-check.zig`,
`src/bench.zig`, `docs/reference/{session,speculative-decoding,bench}.md`,
`docs/benchmarks/speculative-2026-09-20-recovery/`, `TODO.md`, and this log.

**Remaining.** The verify batch now dominates (241–372 ms, ~2.6–3.0 ordinary
steps at 512, 3.6 at 4K) and is what keeps prose below 1×; the record replans
the remaining speculative units around it (the small-batch matrix tile becomes
its own unit). The checkpoint copy is still taken every batch although the row
slots make it unnecessary for recovery; dropping it is a follow-up. The
session grows by 1.26 GB whenever a drafter is loaded, recorded here and in
`session_bytes`. KERN-12's 3–8-row verify target remains unmet. The CPU
speculative check is outstanding (deferred, not failed).

### KERN-13 — A GPU penalty kernel: the token history applied on the device before the top-k (2026-09-20)

**Outcome.** A sampler with an active presence or repetition penalty no
longer forces the full-logit readback: `nu_penalize` applies the penalties
in place on the device, before the output head's argmax and before the top-k
partial pass, so the 2 KB readback is the vector the CPU sampler would sort.
The kernel takes the history as `vocabulary / 32` little-endian u32 words
(bit `id` in word `id / 32`), the repetition and presence values; for a
token in the set it computes `l / r` for positive and `l · r` for negative
logits, then `l − presence` — `Sampler.penalize` operation for operation.
The plans keep a `penalty_history` buffer and upload the words only when
`History.revision` changed (an `observe` of an unset id, a non-empty
`reset`), one 31 KB copy per change. `TopK` gained a `penalized` flag and
`Sampler.selectFrom` decides a penalized readback only when the device
marked it so (an unpenalized one still defers to the full path);
`gpuEligible()` no longer excludes penalties. The raw `logits` readback is
never penalized in place: the kernel is recorded only when a selection
(greedy or top-k) was requested, and the loop's `readLogits` fallback skips
the sampler's own penalties when the retained vector already carries them —
a first implementation penalized in place unconditionally, which `make
compare` caught as a 1.5-logit shift in the raw trace. `Penalties`
(`{ history, repetition, presence }`) threads through
`Executor.step`/`prefill`; Qwen, Gemma 4, and Muse Glimmer all apply it
(Gemma and Muse after their soft-cap). The CPU reference ignores it and
keeps penalizing in `select`.

**Evidence.** `make test-metal`: the device vector equals the CPU sampler
for every logit sign (positive, negative, zero, negative zero), both
penalties alone and together including a rewarding pair, on a word layout
from `writeWords`; exact except that Metal's fast math flushes subnormal
operands to zero (a residual below the smallest normal F32, unresolvable in
a softmax). `make speculative-check-metal` now also runs a penalty check on
the real model: two plans consume the same tokens, one steps with the
penalized top-k readback, the other reads full logits and lets the sampler
penalize; 8 sampled and 9 greedy tokens are identical with one seed each.
`make check`, `make compare` (f32 6.1e-5 / 7.7e-7, f16 2.5e-2 / 1.9e-4),
and `make speculative-check-metal`'s existing phases pass. The unit's gate
pass `make speculative-record ARGS="--only prose512 code"` (`d31c5cd`,
`nuclis 0.2.0-dev`, Apple M4 Pro 48 GiB, macOS 26.6.2, artifact SHA-256
`322e194f…`, reports under `.zig-cache/bench/spec/`; table in
[bench.md § The KERN-13 quick pass](reference/bench.md#the-kern-13-quick-pass-2026-09-20))
puts the instruct off baselines at 9.04 / 8.87 / 8.80 tok/s (prose d2/d4/d7)
and 8.74 (code d4), against 8.18 / 7.99 / 7.78 and 8.56 in the ENGN-14
record and inside the greedy off band (code 8.66–8.74; prose 9.21–10.22,
whose spread is the session's clock drift). `topk_fallbacks` is 0
everywhere. A same-session pair measured prose instruct at 9.90 against
greedy 10.156 tok/s (2.5 % below) and, in the reverse order, 9.159 against
10.156 — the cross-process spread dominates the difference the acceptance's
2 % band would test. The sampled accept time is unchanged (62.8–91.3 ms per
batch): that is ENGN-15's target.

**Files.** `inference/src/backends/metal/kernels.metal`,
`inference/src/backends/metal/root.zig`,
`inference/src/sampling/root.zig`,
`inference/src/models/{qwen35_metal,gemma4_metal,muse_glimmer_metal}.zig`,
`inference/src/engine.zig`, `inference/metal-check.zig`,
`inference/generation-check.zig`, `src/agent/loop.zig`,
`docs/reference/{speculative-decoding,metal-backend,bench}.md`, `TODO.md`,
and this log.

**Remaining.** The speculative sampled acceptance still reads every verify
row back and sorts it (ENGN-15 now has the device path it needs). The
subnormal flush is accepted, not compensated; a logit whose magnitude is
below `floatMin` is numerically zero under any softmax step. The CPU
speculative check deferred from ENGN-14 was started in this session and
reported with ENGN-15.

### ENGN-15 — Sampled acceptance on the GPU top-k readback (2026-09-20)

**Outcome.** The sampled acceptance no longer sorts the full logits of every
verify row. `engine.VerifyOutput` gives `verify` two modes: full `rows`, as
before, and `topk` — one `sampling.TopK` per row from the device partial
top-k over that row, with the logits left resident (`verify_logits`) so
`Plan.readVerifyRow` serves a fallback. The plan owns one `TopKBuffers` set
per verify row (16 × ~130 KB, allocated with the drafter) because the rows'
dispatches share one command buffer. The sampled path decides each row with
`Sampler.selectFromHistory`: the penalties, when active, are applied on the
host to the 256 candidates with the live history that the accepted drafts
advance, and the penalized top-k is exact when its last entry clears the
readback's smallest raw value (the penalties only lower values). Raising
penalties (`repetition < 1`, `presence < 0`), a penalized greedy draw, a
nucleus, or a failed bound return `null` without advancing the RNG, and the
loop reads the row back and takes `distribution`, counted in
`topk_fallbacks` (which `Timing` now fills for verify as well as the step
path). Without penalties `selectFromHistory` delegates to `selectFrom`, so
eligible nucleus sampling decides from the readback as on the step path.
The seeded tests in `sampling/speculative.zig` are unchanged.

**Evidence.** `make speculative-check-metal` passes all phases plus a new
`verifyTopKCheck`: the same four-token batch verified with both modes on
real logits, every row's readback decision (with and without penalties)
equal to `select` on the rows-mode row, the resident rows bit-identical when
the readback defers — 8/8 rows decided, 0 fallbacks. `sampling/root.zig`
tests `selectFromHistory` against `select` over random vectors and option
sets, including the deferrals. `make check`, `make compare` (f32 6.1e-5 /
7.7e-7, f16 2.5e-2 / 1.9e-4). The unit's gate pass
`make speculative-record ARGS="--only prose512 code"` (same revision and
workload as the KERN-13 pass, reports under `.zig-cache/bench/spec/`; table
in [bench.md § The ENGN-15 quick pass](reference/bench.md#the-engn-15-quick-pass-2026-09-20))
measures `accept` at **18.8–36.9 µs per batch** with penalties and
0.1–0.2 µs for the greedy sampled path, against the ≤ 5 ms target and the
62.8–91.3 ms of the KERN-13 pass, with `topk_fallbacks` 0 at every
configuration. Code instruct draft 4 rose 1.12× → **1.36×**; prose 512
instruct 0.76 / 0.82 / 0.85× → **0.92 / 0.98 / 1.01×** at drafts 2 / 4 / 7.
The deferred CPU speculative check was run in this session: `make
speculative-check` finished exit 0, 12 tokens identical to ordinary greedy
decoding, 7/24 drafts accepted (the engine-loop edge cases are Metal-only
by design).

**Files.** `inference/src/engine.zig`, `inference/src/sampling/root.zig`,
`inference/src/models/qwen35_metal.zig`, `inference/generation-check.zig`,
`docs/reference/{speculative-decoding,bench,generation}.md`, `TODO.md`, and
this log.

**Remaining.** Verify (262–297 ms) and propose (13–45 ms) are now the whole
batch; KERN-14 (small-batch tile) and ENGN-16 (proposal policy) are next.
A row whose penalties can raise values, a penalized greedy verify
(`temperature 0` with penalties), and a `top_k = 0` nucleus under penalties
still take the full-row fallback by design; none of the record's
configurations hits them.

### KERN-14 — The wide 32×8 small-batch tile: measured, closed negative (2026-09-20)

**Outcome.** The verify batch's small-batch tile experiment closed negative
and nothing routes to it. One new variant of the 16×8 split-K tile: 32
output rows per 128-thread group (`nu_matmul_wide_body`, the `_w8`
instantiations for the seven specialized encodings), where each lane owns a
row and four 8-row accumulator blocks share one B load per K step, halving
the gathered activation traffic per weight byte. `Backend.matmulTile32`
forces it, `specializedMatmulWide` picks it, `matmulGeometry` labels it
32×8, and the production `matmul` policy is untouched (`.auto`, the 16×8
tile, and the two-row matvec routing as before). The design's second
experiment (a blocked activation read) was not run: the first variant's
regression is a tile-shape effect, not an activation-load effect, and the
head/FFN split shows no promising half.

**Evidence.** `make bench-matvec-rows ARGS="8 head"` (Apple M4 Pro 48 GiB,
Zig 0.16.0, ReleaseSafe, `d31c5cd` plus the change; minimum of three
measured command buffers after one warm-up, sixteen dispatches per buffer;
[bench.md § Small-batch tile sweep](reference/bench.md#small-batch-tile-sweep-kern-14-2026-09-20)).
At 5 rows, 16×8 → 32×8 GB/s: Q4_K 96.8 → 88.7 (gate), 94.4 → 85.0 (down);
Q6_K 114.0 → 84.6 and 105.0 → 87.8; Q3_K 63.0 → 47.1 and 57.2 → 46.1;
IQ3_S 62.3 → 56.6 and 60.7 → 55.0; Q5_K 110.0 → 109.8 and 109.4 → 100.7;
IQ4_XS 90.5 → 92.7 and 85.9 → 89.3; head IQ4_XS 83.0 → 89.5, Q6_K 109.2 →
89.0. No shape reaches the ≤ 150 GB/s bar (the best at 5 rows is 110). The
same trend holds at 2 and 8 rows. `make test-metal` passes with the wide
tile exactness-gated against the generic F32 tile at the half-tile bound
(2e-4 relative to Σ|w·x|) on both FFN shapes at 1/5/8/9/16/20/37 tokens,
and `make check` passes. The full-model verify latency is unchanged from
the ENGN-15 pass (262–297 ms per batch) because production never selects
the candidate; the 2-row routing is not regressed (nothing changed in it).

**Files.** `inference/src/backends/metal/kernels.metal`,
`inference/src/backends/metal/root.zig`, `inference/metal-check.zig`,
`docs/reference/{metal-backend,speculative-decoding,bench}.md`, `TODO.md`,
and this log.

**Remaining.** Verify stays bounded by the 16×8 tile (and the visible-cache
attention) at 512; KERN-16 attacks the long-context case and ENGN-16 trims
what the verify is asked to do. The wide kernels stay instantiated as the
measured fixture (`matmulTile32`); removing them would remove the record
from the tree, which the repo's convention keeps (KERN-12's unused token
counts).

### ENGN-16 — Draft proposal policy: the `p_min` early stop shipped, the adaptive length dropped (2026-09-20)

**Outcome.** `Drafter.propose` gained a `p_min` threshold: on Metal the
block's row records the argmax and one top-1 top-k pass (`1 / Σ exp(l −
max)` at T = 1) in the same command buffer and stops after the first
position whose top-candidate probability falls below the threshold; on the
CPU the same quantity is the host softmax maximum. `engine.draft_p_min =
0.7`, chosen from the `--draft-stats` p_max bins; 0 proposes every
requested position. The plan's second half, an adaptive length
(`k = accepted + 1` after a rejection), **closed negative and is not
shipped**: it loses the chain's tail, whose positions are still accepted at
the per-depth rates. `bench.Sample` gained `proposed_per_step` and the
record script two columns (proposed/step, drafts/accepted);
`--draft-stats` now proposes position by position, bins each draft by its
`p_max` (≥ 0.9, 0.7–0.9, 0.5–0.7, < 0.5), and reports the device-versus-host
`p_max` difference.

**Evidence.** `make draft-stats` (Metal, `d31c5cd` plus the change): code
depth acceptance 90.3/80.0/69.0/64.3 %, bins ≥ 0.9 93.1 % (81/87), 0.7–0.9
42.9 % (6/14), 0.5–0.7 22.2 % (2/9), < 0.5 12.5 % (1/8); `def
fibonacci(n):` 93.5/83.3/82.8/85.7 % and 92.5/100/14.3/33.3 %; device p_max
against the host softmax within 2.6e-7. Interleaved A/B in one session
(control `p_min = 0`): prose 512 instruct draft 4 drafts/accepted 2.16 →
1.57 (−27 %), speedup 0.872/0.907 → 0.951/0.942; code greedy draft 7 2.31 →
1.48 (−36 %), speedup 1.312/1.323 → 1.330/1.332 (+1.3 %). `p_min = 0.8`
trims more prose (−34 %) but costs code 4.4 %. The adaptive length alone
(code draft 7): accepted/step 1.91 against 2.97 and speedup 1.02 against
1.32. The unit's gate pass (`--only prose512 code`, reports under
`.zig-cache/bench/spec/`; table in
[bench.md § The ENGN-16 quick pass](reference/bench.md#the-engn-16-quick-pass-2026-09-20)):
code instruct 1.35×, code greedy 1.33× at draft 7 (within 1 % of the
control), prose instruct 1.00 / 1.03× at drafts 4 / 7 (was 0.98 / 1.01×),
drafts/accepted 1.24–1.73. `make check`, `make compare` (f32 6.1e-5 /
7.7e-7, f16 2.5e-2 / 1.9e-4), and `make speculative-check-metal` (all
phases; the loop check still shows partial acceptance) pass.

**The near miss, stated plainly.** The acceptance asked prose 512 draft 4
for a ≥ 30 % fall in drafts per accepted token; the measured control value
is 2.16 (not the arithmetic 4/1.82 = 2.20, since the final batch is
truncated) and the policed value 1.57, a 27 % fall — three points short.
The other half of that criterion, the decode rate not lower, is met with an
8–10 % gain (0.87 → 0.94/0.95 interleaved; 1.00× in the gate pass), and
every code criterion passes. Shipped at 0.7; 0.8 would pass prose and fail
code.

**Files.** `inference/src/runtime/draft.zig`,
`inference/src/models/{qwen35_metal,qwen35_runtime}.zig`,
`inference/src/engine.zig`, `inference/generation-check.zig`,
`src/bench.zig`, `scripts/nuclis-speculative.py`,
`docs/reference/{speculative-decoding,bench,generation}.md`, `TODO.md`, and
this log.

**Remaining.** Prose stays below 1× at draft 2 and near 1× at 4–7; verify
is the batch (234–260 ms). A better proposal policy than "stop at the first
low-probability position" — one that keeps the chain's tail without paying
its verify rows — is unmeasured; the adaptive length's negative result is
the evidence against the obvious variant.

### KERN-15 — Split-K decode matvec: measured behind the single pass, closed negative (2026-09-21)

**Outcome.** The row-poor matvec experiment closed negative and nothing
routes to the split path. The unit's premise — a 6,656–8,704-row matrix
launches too few 16-row threadgroups to keep the weight bus busy, so
splitting K should recover the head's rate — was refuted by measurement:
the split bodies run behind the single-pass kernel on every Q4_K shape at
2/4/8 splits, and the loss grows with the split count, the signature of the
per-group reduction tail and the extra dispatch rather than of insufficient
parallelism. Shipped as the measured fixture: `nu_matvec_q4_k_split`,
`nu_matvec_q5_k_split`, `nu_matvec_segments_split`, `nu_reduce_splits`,
`nu_segment_reduce_splits`, reachable only through `Backend.matvecSplits` /
`matvecSegmentsSplits` (both refuse a split for an encoding without a split
body), a lazily created 512 KB row-partial scratch, `make bench-matvec-split`
(`metal-check --matvec-split [ENCODING]`), and the `make test-metal`
exactness fixture at 2/4/8 splits. `MatvecBlockParams` keeps its four
fields: the Q4_K/Q5_K decode body takes the `[first, last)` K bounds as
arguments, so the standalone kernels pass compile-time constants and their
code is unchanged; the three bodies without a split twin were left
untouched. The auto-routing the WIP carried (Qwen's 5,120-row shapes and
the merged projections) was removed with the negative verdict.

**Evidence.** `make bench-matvec-split` (Apple M4 Pro 48 GiB, Zig 0.16.0,
ReleaseSafe, `10cf6ba` plus the change; five command buffers, 64 dispatches
per buffer below 8,192 rows and 8 above, best of three measured rounds;
table and reading in
[bench.md § Split-K matvec sweep](reference/bench.md#split-k-matvec-sweep-kern-15-2026-09-21)).
Single pass → 2/4/8 splits, GB/s of weight bytes: Q4_K 6,656×19,968 151.0 →
145.9/148.1/138.8; Q4_K 6,656×4,096 157.7 → 150.7/135.5/124.3; Q4_K
5,120×17,408 152.8 → 146.1/146.6/134.5; the four-segment 8,704×6,656 merge
152.1 → 157.4/150.3/158.5; Q5_K 6,656×19,968 193.0 → 193.4/198.4/193.3;
Q5_K 6,656×4,096 175.0 → 170.3/159.0/130.6; Q5_K 5,120×17,408 191.2 →
190.9/184.8/172.6. The acceptance bar (the two row-poor Muse shapes
≥ 190 GB/s) is missed: Q4_K's 144 bytes per 256-value block cap the kernel
at 150–175 where Q5_K's 176 reach 190–210 at the same geometry, so the
lever is per-block arithmetic, not K parallelism. Muse decode at 512 is
9.65 tok/s (prefill 93.14) against the MODL-13 record's 9.60/93.2 — the
acceptance's ≥ 10.5 is not met, and the record stands; the full four-length
acceptance record was not re-run because production is unchanged (the
negative close keeps the KERN-14 precedent). Qwen decode is
10.49 tok/s at 512 (prefill 88.19; the record is 10.62, the 2026-09-19
same-day check 10.44), not slower. Gates: `make check`; `make test-metal`
holds the split path at 6,656 rows for Q4_K/Q5_K at 2/4/8 splits and the
four-segment merge at 2/4/8 against the F64 CPU reference within
Σ|w·x|·4e-6; `make test-generation-metal` and
`make test-generation-muse-glimmer-metal` pass; `make speculative-check-metal`
passes (12 tokens identical, loop, budget, EOS, cancellation, context); all
twelve family compares pass unchanged (Qwen f32 6.1e-5 / 7.7e-7, f16
2.5e-2 / 1.9e-4; Muse f32 1.5e-4 / 8.0e-7, f16 7.3e-2 / 2.0e-4; Gemma 4
QAT, Gemma 4, Gemma 4 26B-A4B, and Bonsai at their pinned bounds).

**Files.** `inference/src/backends/metal/kernels.metal`,
`inference/src/backends/metal/root.zig`, `inference/metal-check.zig`,
`build.zig`, `Makefile`,
`docs/reference/{bench,metal-backend,muse-glimmer}.md`, `docs/llm-guide.md`,
`TODO.md`, and this log.

**Remaining.** The Q4_K kernel's ~0.9 ns per 256-value block is Muse's
decode limiter (159 GB/s effective against the reference's 223); the lever
this unit leaves untested is per-block arithmetic, and the catalogue's Q4_K
files are what it would pay for. The merged projection's ~4 % at 2 and 8
splits is the one lead, unshipped because the bar was not met and the
segment bucket is global (it would route the other families' merges
unmeasured). The split machinery stays as the measured fixture, as the
wide 32×8 tile does, in case an encoding or a fused-consumer reduction
changes the balance.

### KERN-16 — Long-context prefill attention: register-level reuse measured 2–5 % at chunk sizes, closed negative (2026-09-21)

**Outcome.** The second attempt at the long-context prefill attention
closed negative against its acceptance. ENGN-08's untried lever — a
(head, 32-query) threadgroup whose four SIMD groups split the 256 value
columns and publish the probability tile through threadgroup memory, so a
V block serves four row-block multiplies — was built
(`nu_attention_chunk_reuse` / `_h`), is numerically identical in accuracy
to the shipped body, and is consistently ahead, but only by 2–5 % at the
256-row prefill chunks (0 % at 512 visible, −4.8 % at 16K, −3.2 % at
32,512 in F16); the unit's acceptance needed that attention cut to reach
prefill within 10 % of the reference at 32,639. The reuse body is
nevertheless the 11–16 % faster one on the verify-shaped counts (1–64
rows), where each group carries a quarter of the value columns instead of
one group carrying all 256, so it is shipped behind
`Backend.attention_reuse_max_rows = 64` on the 256-wide geometry the
window was measured on; other widths and the 256-row prefill chunks keep
the row-split body. The mechanism: the removed loads were not the
limiter. F16 and F32 caches differ by only 5–10 % in time and the kernel
sits flat at ~640–750 GFLOP/s F16 across 512–32,512 visible, so the
per-tile instruction and latency chain dominates and removing 84 of ~500
per-group instructions per tile buys little. The design, the register
budget, and the untried levers (bank-conflict padding of the score and P
tiles, vectorized staged K/V with double buffering, and a phase ablation)
are recorded in
[metal-backend.md § KERN-16](reference/metal-backend.md#long-context-prefill-attention-second-attempt-kern-16-2026-09-21-closed-negative).

**Evidence.** `make bench-attention` (`metal-check --attention-bench`;
Apple M4 Pro 48 GiB, Zig 0.16.0, ReleaseSafe, `652a0cc` plus the change;
one command buffer of `max(1, 65536/visible)` dispatches after two
warm-ups, best of three measured rounds, model geometry 24/4/256 on
synthetic F32 and F16 caches, no model): at count 256, row-split →
reuse F16 ms 3.684 → 3.688 at 512, 35.179 → 34.246 at 4,096, 71.653 →
69.932 at 8,192, 145.031 → 138.064 at 16,384, 282.230 → 273.058 at
32,512; at count 8, 0.862 → 0.767, 3.610 → 3.163, 7.621 → 6.460, 30.642 →
25.806; at count 64, 1.139 → 1.097, 8.945 → 7.757, 36.121 → 31.382; at
count 1 and 16,384 visible, 30.704 → 25.831. F32 in the table in
[bench.md § Prefill attention sweep](reference/bench.md#prefill-attention-sweep-kern-16-2026-09-21);
run-to-run spread ~1.5 %. `make test-metal` passes with both bodies on the
MODL-06 windowed/wide cases (F32 2.980e-6, F16 1.929e-4) and on a new
poisoned-future-range fixture: counts 1–256, F32 and F16, every cache row
after `position + count` set to 1e30 (F32) or 6e4 (F16), all compared
rows match the F64 CPU reference (F32 2.384e-7, F16 2.683e-4; bounds 1e-5
and 1e-3). `make build` passes. Per the session's gate policy the model
gates (`make compare`, the generation and speculative checks) and the
speculative record were deferred to ENGN-17, which re-measures the routed
path.

**Files.** `inference/src/backends/metal/kernels.metal`,
`inference/src/backends/metal/root.zig`, `inference/metal-check.zig`,
`build.zig`, `inference/build.zig`, `Makefile`,
`docs/reference/{bench,metal-backend}.md`, `docs/development.md`,
`TODO.md`, and this log.

**Remaining.** The KERN-16 acceptance's prefill bars (32,639 within 10 %,
16K within 8 %) are not met and the Qwen long-context deficit is therefore
not attention-load-bound; the 2–5 % at chunk sizes is live in the tree but
worth ~1 % of prefill. The verify-shaped routing needs ENGN-17's record to
confirm it on the real batch (the kernel-level window is −11 % at 512 and
−15 % at 16K context). The three untried levers above are the material for
a third attempt, which the plan does not currently order; the benchmark
harness and the fixture make it cheap to test them.

### KERN-18 — Fused decode norms: −182…−192 dispatches per decode step shipped, the speed bars missed; closed below its target (2026-09-21)

**Outcome.** Three fused norm kernels replace the plan pairs the three
families dispatch every layer, and the unit closes below its target: the
dispatch-count bars pass on Qwen and Gemma and miss narrowly on Muse, and
all three decode-rate bars miss because the unit's premise — that the norm
dispatches pay a launch floor rather than bytes — was refuted by
measurement. `nu_rmsnorm_add` (`(destination + norm(input)·w) · scale`, the
`rmsNorm` + `addScale` pair of every post norm), `nu_add_rmsnorm`
(`residual += input` then `output = norm(residual)·w`, the `add` +
`rmsNorm` pair where the norm consumes the updated residual, Qwen's
post-attention norm), and `nu_rmsnorm_rope` (the `rmsNorm` + `rope` pair of
every q/k norm, split-half or adjacent, strided input or in place). Each is
reached through a Backend helper (`rmsNormAdd`, `addRmsNorm`,
`rmsNormRope`) that runs the pair it replaces when `Backend.fused_norms` is
false; `bench --unfused-norms` forces that control. The plans wired: Qwen's
step add+RMSNorm (64 per token) and full-attention q/k norms (32), Gemma's
attention and FFN post norms (96) and q/k norms (96), Muse's post norms
(104) and sliding q/k norms (78; its 13 global layers have no rotation).
The design, the kernels, and the reading are in
[metal-backend.md § KERN-18](reference/metal-backend.md#fused-decode-norms-kern-18-2026-09-21-closed-below-its-target).

**Evidence.** `make test-metal` (new `checkFusedNorms`): every fused kernel
against the pair it replaces and against `cpu.rmsNorm`/`cpu.rope` over the
model widths and strides — max abs **4.8e-7** for both comparisons (bounds
2e-5 fused-vs-unfused, 2e-4 against the CPU); a fused `generate` on
Qwen3.8-27B reproduces the greedy text. `bench --profile` (canonical
workload, drafter not loaded, `1d82c18` plus the change) dispatches per
step: Qwen 938.8 → 844.3 (rmsnorm 209.0 → 114.5, add 128.0 → 65.0,
add_rmsnorm 63.0, rmsnorm_rope 31.5), Gemma 4 12B QAT 882.2 → 693.2
(rmsnorm 337.0 → 148.0, add_scale 95.3 → 0, rmsnorm_add 94.5, rmsnorm_rope
94.5), Muse 922.9 → 743.8 (rmsnorm 314.0 → 134.8, add_scale 102.4 → 0,
rmsnorm_add 102.4, rmsnorm_rope 76.8): −94.5 / −189.0 / −179.2 per step
blended by three prefill command buffers, −96 / −192 / −182 per decode step
by design. The bars (≥ −86 / −177 / −185) are met on Qwen and Gemma and
missed on Muse by three dispatches, whose 20 % bar assumed all 52 layers
rotate q/k. `bench` decode at 512 (128 tokens, F16, greedy, interleaved
off/on pairs, drafter not loaded): Qwen 10.814/10.742 → 10.788/10.744
(**1.000×**), Gemma 27.387/27.379 → 27.517/27.515 (**1.005×**), Muse
9.993/9.936 → 10.018/9.975 (**1.004×**) against the ≥ 1.01 / 1.04 / 1.02
bars; prefill unchanged. The reading: the 2026-09-07 profile's ~13 µs per
`rmsnorm` dispatch was kernel time, not launch overhead — a norm dispatch
costs ~7–10 µs of which ~2–4 µs is the launch, and the fused kernel does
the same memory passes in one dispatch, so the win is the removed launch
(~0.4 ms of Gemma's 39 ms step). The fusion ships: correct, a small
consistent win, and the pairs stay one flag away. `make check` passes; per
the session's gate policy `make compare`, the generation and speculative
checks, and the Muse/Gemma acceptance records were deferred to ENGN-17.

**Files.** `inference/src/backends/metal/kernels.metal`,
`inference/src/backends/metal/root.zig`,
`inference/src/models/{qwen35,gemma4,muse_glimmer}_metal.zig`,
`inference/metal-check.zig`, `src/bench.zig`, `src/cli.zig`, `src/help.zig`,
`docs/reference/{bench,metal-backend}.md`, `TODO.md`, and this log.

**Remaining.** ENGN-17's record measures the shipped path end to end; the
Qwen profile there will be taken with the default open the unit defines.
Two facts to carry into MODL-19/20: `bench` on the Gemma and Muse entries
fails with `DraftSourceMissing` while the config's
`generation.speculative` is true and their registry companions have no
adapter yet, and the route to a real decode win is epilogue fusion (the
post norm inside the projection or attention kernel that produces its
input), not merging two memory-bound dispatches.


### MODL-19 — Gemma 4 draft heads: the `gemma4-assistant` companion as a second GGUF, correct but a negative default at the plan's draft length (2026-09-21)

**Outcome.** The Gemma 4 entries can speculate with their pinned companion
heads, and the unit closes with a **negative verdict at the plan's draft
length**: the adapter is correct (its rows match the reference trace at
1.1e-4 max abs on the CPU and 1.5e-4 on Metal, and greedy decoding with the
switch on is token-identical to ordinary greedy), acceptance is high (2.26
drafts per batch at draft 4, 2.74 at draft 7, ~55 % per proposed position),
and every non-verify cost is negligible (propose 6.8 ms, accept 0.1 µs,
recover/commit/checkpoint ≤ 0.002 ms per batch) — but the verify batch is
136 ms whether it carries 3 or 8 rows, so the pair breaks even only at
draft 7 (1.017×: 25.16 → 25.60 tok/s) and loses at drafts 2 and 4 (0.705×,
0.899×) against 25.2–25.4 tok/s ordinary decode. The row-flat verify says
the lever is `max_draft_length` (the 8-row tile bound), not the drafter;
that is ENGN-17's call, not this unit's.

**What shipped.** `inference/src/models/gemma4_assistant.zig`: the
companion's own binder (49 tensors, `embedding_length_out` 3840/2816, four
blocks of pattern `[1,1,1,0]`, no `attn_k`/`attn_v`). `Engine.open`'
`DraftRequest.{file,preferred}` maps a second GGUF, checks architecture,
target width, vocabulary count, and the global-layer RoPE factors the Metal
plan reuses, and reports `DraftSourceMissing`/`DraftSourceMismatch`;
`generate`, `agent`, and `bench` resolve `.preferred` from
`models.<name>.mtp` (Qwen keeps its embedded block via
`family.embedded_draft`). The head itself: `gemma4_runtime.Runtime` and
`gemma4_metal.Plan` gain `propose`/`commit`/`reset`/`bytes`/`drafter` and a
`draftForwardTrace`; the head reads the target's layer-46 (sliding: visible
`position − 1023 … position − 1`) or layer-47 (global: `0 … position − 1`)
cache rows and writes none, pairs `[sqrt(3840)·embed_target; h]` through
`nextn.pre_projection`, and classifies with its own tied `token_embd`;
`commit` is the last target hidden and recovery is the position rewind
alone. Gemma's runtime gained `prefill` hidden rows, `verify`, and
`verifyGreedy`; the plan gained `verify`/`verifyGreedy`/`readVerifyRow` and
`prefill` hidden rows over a shared `recordLayers`. The role name stays
`mtp`. `scripts/reference-generation.cpp` gained `--assistant-draft` (runs
each head row before the target decodes that position, the driver's
`draft()` moment) and the trace is pinned under
`inference/src/models/fixtures/gemma4-mtp/` (2 rows, greedy 2613 and
236764); `make compare-draft-gemma4` checks it.

**Evidence.** `make compare-draft-gemma4` (CPU and Metal):
max abs 1.097e-4 / 9.829e-5 (CPU positions 1/2) and 1.535e-4 / 1.202e-4
(Metal) against rel RMS 3.6e-6 / 3.1e-6 / 5.4e-6 / 4.5e-6, bounds 1e-2 /
1e-4; `propose` from the committed prefix returns the pinned draft.
Greedy off/on identity over 64 tokens (`Write a haiku about the sea.`,
Metal, template prompt). Load errors through a temporary `NUCLIS_HOME`:
missing companion → `DraftSourceMissing`, `clip` projector → mismatch, 26B
head on the 12B target → mismatch. Bench pair in
[bench.md § The Gemma 4 draft pair](reference/bench.md#the-gemma-4-draft-pair-modl-19-2026-09-21);
the facts, trace, and verdict in
[speculative-decoding.md § The Gemma 4 assistant heads](reference/speculative-decoding.md#the-gemma-4-assistant-heads-modl-19).
`make check` passes (455/455) with the machine quiet; note that the Metal
profile fixture's 20 µs encoder-jitter allowance tripped twice under load
(`profile check` read 95 µs over the command-buffer span at load average
~6) before passing at 58.5 vs 48.9 µs — pre-existing, not this change.
Per the session's gate policy `make compare`, the generation and speculative
checks, `draft-stats`, and the Gemma acceptance records are deferred to
ENGN-17.

**Files.** `inference/src/models/gemma4_assistant.zig` (new),
`inference/src/models/{gemma4,gemma4_runtime,gemma4_metal}.zig`,
`inference/src/engine.zig`, `inference/generation-check.zig`,
`inference/src/models/fixtures/gemma4-{mtp,head-12b}` (new),
`scripts/reference-generation.cpp`, `src/{engine,generate,bench}.zig`,
`src/agent/root.zig`, `Makefile`, `docs/reference/{speculative-decoding,bench}.md`,
`TODO.md`, and this log.

**Remaining.** The 26B-A4B head is bound and width-checked by the same code
but never measured (it needs the MoE target's 2816-wide residual); its trace
is MODL-19's unfinished second half if the user wants it before ENGN-17.
The verifier's row-flat cost and `max_draft_length = 7` bound the Gemma
speedup, so a longer draft window is the measured lever.

### MODL-20 — Muse Glimmer DFlash drafter: the companion, the CPU reference and its trace, the Metal plan, a positive verdict (2026-09-21, two sessions)

**Outcome.** Muse Glimmer 30B can speculate with its pinned `dflash-kquant`
companion, and the unit closes with a **positive verdict at every measured
length**: on the acceptance workload's 512-token prompt (Metal, F16 KV, ctx
32768, greedy, one warmup and three measured off/on pairs per configuration
on one loaded model) the pair decodes at 1.234× ordinary at draft 4, 1.163×
at 8, and 1.222× at 15, with 73–77 % of proposed positions accepted and the
early stop trimming proposals to 1.83 / 2.12 / 2.36 per step. The verify
batch is the cost (172.9–188.2 ms for 2.8–3.4 rows, 1.7–1.8 ordinary
steps); recovery is the position rewind alone (1 µs per batch; the family is
attention-only) and the prompt commit adds 1.2–2.9 % to prefill. Session 1
(CPU reference, contract fit, pinned trace) is recorded in
[speculative-decoding.md § The Muse Glimmer DFlash drafter](reference/speculative-decoding.md#the-muse-glimmer-dflash-drafter-modl-20);
session 2 delivered the Metal plan, the trace on both executors, the
record, and this verdict. ENGN-17 sets the entry's default from the record.

**What shipped (session 2).** `muse_glimmer_metal.zig`: the plan opens five
draft attention layouts when a companion is bound (the same `kv` precision
as the language model); `prefill`/`verify`/`verifyGreedy` capture each
target layer's input residual rows into one device slab per slot (the
shared `recordChunkLayers`), the plan assembles them into the row-major
`h_rows` the contract's `commit` consumes, and `commit` uploads them back
through a staging buffer for the encoder's batched `fc` and per-layer key
projections (chunked when the prefix is longer than the plan's chunk). The
block is one batched forward over 1 + k rows (batched q/k/v, per-head
norms, split-half RoPE at each row's own position, batched output and FFN)
with the **non-causal** block attention as one `attentionDecode` per row —
the causal chunk kernels' mask cannot express it and the block is 16 rows;
the proposal's head runs over every block row and reads each draft row's
top candidate plus its full-vocabulary softmax denominator through `topk`
with `k = 1`, so `p_min` is the same full-vocabulary probability as the
CPU's. `generation-check`'s `museDraftTrace` runs the CPU reference and the
Metal plan over the same pinned rows in one `--metal` invocation (the
trace's capture uses the generic F32 tile; the production half tiles would
mask the drafter's own numerics); `make compare-draft-muse` now runs both
`compare-draft-muse-cpu` and `compare-draft-muse-metal`. The draft-length
cap moved with the family: `Drafter` gained `max_proposals` (Qwen 7, Gemma
7, Muse 15) and the host `engine.max_draft_length` is 15, with `runLoop`
enforcing the loaded family's own bound and sizing the speculative scratch
from it; `config`/`--draft-length` validate against the host bound. A
latent `@min`-narrowing overflow in the Muse proposers (`u4` `count_max`
from a comptime operand, 15 + 1) was found by the draft-15 run and fixed on
both executors.

**Evidence.** `make compare-draft-muse-metal` (also runs the CPU pass):
residuals 1.5e-4…1.1e-3 max abs / 9.2e-7…3.1e-6 rel RMS (bounds 5e-3 /
1e-5), encoder 2.9e-4 / 8.2e-5 and 2.2e-4 / 1.4e-4 (bounds 1e-2 / 2e-3,
the `fc` half tile over large residual rows), block hidden 1.02e-2 /
4.4e-3 and 4.7e-3 / 1.5e-3 (bounds 2e-2 / 1e-2), all 30 pinned greedy rows
on both executors; `make compare-draft-muse-cpu` unchanged. Greedy
`generate --speculative on` vs `off` on the registry entry is byte-identical
over 12 tokens on Metal, and both sampled paths (the device top-k readback
and the penalized full-rows fallback) complete on the same entry.
`make test-generation-muse-glimmer-metal` passes with the plan's changes,
`make compare-muse-glimmer` is unchanged (f32 2.4e-4 / 8.1e-7, f16 7.3e-2 /
2.0e-4, also with the drafter loaded and the switch on), and the Qwen gates
the shared cap change touches stay green: `make compare` (f32 6.1e-5 /
7.7e-7, f16 2.5e-2 / 1.9e-4), `make test-generation-metal`,
`make speculative-check-metal` (12 greedy tokens identical, the loop edge
cases, the penalty and verify-top-k checks), `make compare-draft-metal`
(1.5e-5 / 7.9e-7, greedy 9419/271), and `make check` (all green).
Record and table:
[bench.md § The Muse Glimmer DFlash draft pair](reference/bench.md#the-muse-glimmer-dflash-draft-pair-modl-20-2026-09-21);
memory: session 2,415,919,104 bytes (2304 MiB, the five draft caches 640 MiB
of it), workspace 149,861,504 bytes, verify scratch 53,862,464 bytes.
Reports under `.zig-cache/bench/muse-modl20-draft{4,8,15}.json`.

**Files.** `inference/src/models/{dflash.zig,muse_glimmer.zig,muse_glimmer_runtime.zig,muse_glimmer_metal.zig}`,
`inference/src/models/fixtures/muse-dflash/`, `inference/src/runtime/draft.zig`,
`inference/src/models/{qwen35,qwen35_metal,qwen35_runtime,gemma4,gemma4_metal,gemma4_runtime}.zig`,
`inference/src/engine.zig`, `inference/generation-check.zig`,
`scripts/reference-generation.cpp`, `src/cli.zig`, `Makefile`,
`docs/reference/{speculative-decoding,bench,muse-glimmer}.md`, `docs/spec.md`,
`TODO.md`, and this log.

**Remaining.** The verify batch's row-flat cost is the speedup's lever (the
same observation as MODL-19): 172.9–188.2 ms for 2.8–3.4 rows because the
small-chunk matmul tiles pad to their tile bound, so a cheaper small-batch
verify or a longer draft window is ENGN-17's call. The block's non-causal
attention runs 16 `attentionDecode` calls per layer per proposal; a batched
non-causal block kernel is an untried kernel lever, recorded for a future
attempt. The 26B-A4B assistant head and the vision projectors are other
units' work.

### ENGN-17 — The speculative verdict: Qwen and Gemma off, Muse on; the full record and `bench`'s true baseline (2026-09-21)

**Outcome.** The speculative plan closes with measured defaults, and the
answer to its question is family-shaped: the Qwen entry's embedded head does
not pay (code greedy 0.92 / 1.20 / 1.30× at drafts 2 / 4 / 7, code instruct
1.28× at 4, prose 512 0.81–0.97× greedy and 0.84–0.95× instruct, 4K 0.73×
both, against the code ≥ 1.5× and prose ≥ 0.9× bar), the Gemma heads do not
pay at the plan's length (0.899× at draft 4, 1.017× at 7), and the Muse
DFlash drafter does (1.234 / 1.163 / 1.222× at drafts 4 / 8 / 15). The
catalogue entries therefore ship Qwen and Gemma off and Muse on, each at
`draft_length` 4, and a fresh `config init` writes those verdicts into the
entries' `generation` sections. The record also settles the MODL-18 item
carried through ENGN-12: with the drafter loaded but the switch off, decode
equals a no-drafter baseline within ±2.4 % on eleven of the twelve
configurations (the twelfth, prose 4K instruct, was drift and a focused
repeat did not reproduce it: loaded-off 10.25–10.32 vs baseline 10.24–10.27
tok/s over two pairs each); what loading costs is memory and time, not rate
(session 2,304 → 3,851 MB, `load_milliseconds` 790 → 1,247 ms).

**What shipped.** `bench` opens the model with `DraftRequest.none` when the
switch is off — no drafter weights, scratch, or draft cache, which is the
true baseline — and with the family's source (`.preferred`) when it is on,
so each measured run is still an off/on pair on one loaded model and the
pair's off sample is the loaded-but-off case. `catalog.Entry` gained
`speculative: bool` and `draft_length: usize` (the measured verdict, with
the reason in each entry's comment); `config.registryEntry` fills the
entry's `generation.speculative`/`generation.draft_length`, so `config init`
writes them and `config show` reports them with `model` provenance; a
user's `--speculative` still overrides. `scripts/nuclis-speculative.py`
gained `--baseline` (a no-drafter pass per configuration) and its
`--summarize` table now carries the baseline, propose, checkpoint, and
commit columns; `make speculative-record` passes `--baseline`. The record's
twelve pairs and twelve baselines are committed under
`docs/benchmarks/speculative-2026-09-21/`.

**Evidence.** The record in
[bench.md § The speculative verdict record](reference/bench.md#the-speculative-verdict-record-engn-17-2026-09-21)
(`981f74d` plus the change, one 52-minute sequence): per-batch costs propose
10.1–23.1 ms, checkpoint 2.8–3.6 ms, verify 218.5–229.1 ms at 512 and
309.2–319.2 ms at 4K, accept 0.00–0.03 ms (was 46–78), recover 6.0–10.7 ms
(was 99–272), commit 4.3–9.5 ms, prompt commit 1.02–1.03× (was 2.9–3.3),
tokens/batch 2.12–3.53; the speedups above. `make check` green with the new
catalog/config tests (`config init`/`config show` cover the entry fields);
the per-family records and the plan's cost table are refreshed in
`TODO.md`. The verify batch remains the verdict's cost (1.6–1.8 ordinary
steps for 2.1–3.5 tokens) and every kernel attempt at it (KERN-12's 2-row
route, KERN-14's tile, KERN-16's window) closed below its target.

**Files.** `src/{bench,catalog,config,help}.zig`, `scripts/nuclis-speculative.py`,
`Makefile`, `docs/benchmarks/speculative-2026-09-21/` (new),
`docs/reference/{bench,speculative-decoding}.md`, `docs/development.md`,
`docs/spec.md`, `TODO.md`, and this log.

**Remaining.** The Qwen default can be revisited if a future small-batch
verify or a longer draft window changes the arithmetic; the 26B-A4B Gemma
head is still bound but unmeasured; `bench`'s per-entry defaults only reach
a fresh `config init` (an existing file keeps its own values, by design).
REPO-09 and REPO-10 turn the gates and records into data next.
