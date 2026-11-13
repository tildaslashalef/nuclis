# Bringing a new model family into nuclis

The starter guide the spec asked for
([spec § Interfaces and extension rules](../spec.md#57-extension-rules)):
the order of work, the commands, the gates, and the mistakes already made
once. It was first written while Gemma 4 12B went through the adapter seam
(2026-09-11 to 2026-09-12). The extension rule it enforces: a new family is
a new adapter, its tests and prompt profile, and any genuinely new
mathematical operation, with no edits to the GGUF parser, the sampling
policy, the CLI generation loop, or the Metal object lifecycle. Check the
diff against that list before closing.

Numbers and file names below are Gemma's; substitute yours. Every step
ends with something committed as evidence (a fixture, a trace directory,
a table in the family's document), because the next step needs it and the
transcript will not survive.

## 1. Container and inventory

- Pick the checkpoint and quantization by `nuclis model inspect
  <owner/repo> --file <name>` (no download): it prints the directory, the
  encodings per tensor, and the verdict (`supported`, `runnable`, `not
  runnable` naming the first tensor whose layout no kernel executes).
  An encoding the tree does not execute is a kernel unit of its own;
  never plan to requantize. Gemma's QAT file (Q4_0) took one session: the
  CPU row decoder, the generic GPU dequantizer (bit-identical to the CPU),
  a specialized matvec and matmul tile built from the nearest existing
  block kernel, the adapter's `executableEncoding`, and the file's own
  traces and acceptance record
  ([gemma4.md § Q4_0 path](gemma4.md#q4_0-path-and-the-qat-file-2026-09-12)).
  Bring the adapter up on a file the kernels already execute first, if the
  repository offers one; the new encoding is then checked against it
  ([§ 6](#6-a-second-encoding)).
- Pull it (`nuclis model pull <owner/repo> --file <name>`) and record the
  commit, size, and SHA-256 from the sidecar in
  [catalogue.md](catalogue.md#pinned-commits-and-digests-2026-09-11). The
  companions (`mmproj`, `draft`) go into the same table; the index entry
  needs their digests. `nuclis inspect --model <file>` reads the local
  directory: identity, dimensions, the encoding histogram.
- `scripts/fixtures/gguf-inventory.py <file> --out
  inference/src/models/fixtures/<family>.json` is an independent header
  reader (standard library only): every metadata key, arrays up to 64
  elements, every tensor's name, shape, and encoding, and the chat
  template as a length/offset/SHA descriptor. That JSON is the binder's
  test fixture.
- Write the facts down in `docs/models/<family>.md` as
  [gemma4.md § Forward pass](gemma4.md#forward-pass-from-the-reference-graph-gemma4cpp)
  does: notation, the forward pass in equations, norms (raw or shifted
  weights?), RoPE pairing and factors, attention scale, activation, soft
  caps, tied heads, layer patterns, and where each fact came from (the
  file, the model card, the paper, the authors' code read as the
  definition of the architecture).

Watch for format capacities disguised as model facts: Gemma's 48-element
per-layer arrays exceeded the parser's retained-array bound (16, a
Qwen-era limit). Raising a *bound* in `formats/gguf` is allowed; adding a
Gemma key to it is not.

## 2. Tokenizer

Implement from the format's definition: the GGUF vocabulary
(`tokenizer.ggml.*`) or the checkpoint's `tokenizer.json`.
`inference/src/tokenizer/vocabulary.zig` lists the `tokenizer.ggml.model`
values it accepts and `tokenizer/encode.zig` the splitters. A new model or
`pre` value needs its own splitter and possibly its own merge alphabet
(Gemma: SPM-style BPE over code points with U+2581 spaces and `<0xNN>` byte
fallback; Muse: the `llama4` label). The splitter's unit tests pin the
pieces of the regex the label names, case by case.

Pin the expected ids from the checkpoint's own published tokenizer test
vectors where they exist, otherwise from strings checked by hand: newline
runs, doubled spaces, CRLF, tabs, emoji, the chat markers inside text.
`nuclis tokenize --raw --json --model <file> --prompt <text>` prints the
ids and byte offsets. Note whether the family needs BOS: nuclis's encoder
never adds it, so a family that needs it writes `<bos>` as text (raw
prompts and the profile alike). A new vocabulary needs its tokens typed:
anything but `normal` and `byte` is structure the stream decoder keeps out
of the text.

The artifact check (`inference/vocabulary-check.zig`) selects its
expectations through the profile the template digest names, so the
`<family>-vocabulary` gate (`make gate NAME=<family>-vocabulary`) runs once
§ 3 has registered the profile.

## 3. Prompt profile

Extract the template (`gguf-inventory.py --max-string 30000`, then the
`tokenizer.chat_template` value; keep it under `.zig-cache/`, never in the
tree) and implement from its text the subset the playground renders:
system and developer, user, assistant with separate reasoning, the
thinking control, tools, and media placeholders.

`inference/src/profiles/<family>.zig` exposes `template_sha256` (the
digest of the exact template, which selects the profile; never the
architecture), `template_aliases` (other revisions proven to render every
fixture identically), `render`, `samplingDefaults`, `stop_tokens`,
`reasoning`, and `stream_markers`; register the tag in
`profiles/root.zig`. The stop tokens are texts (the engine resolves them at
load and refuses a vocabulary without them); the reasoning markers are
what the model *emits*, not what the prompt contains. `stream_markers`
separates actual control-token text from any ordinary opening suffix
(Gemma's `thought\n`). `Profile.decoder` resolves these controls in the
vocabulary and `engine.complete` emits semantic channels; add fake-token
decoder tests before exercising a live completion. Sampling defaults come
with provenance (a model card table, or the file's `general.sampling.*`
hint recorded as a claim).

The fixtures, `inference/src/profiles/fixtures/<family>-text.json` and
`-tools.json` (and media cases), are written by hand from the template:
each case's messages, the rendered prompt, and its ids. Read the template
for what it actually sends rather than what the card says; for Gemma, the
template renders `developer` as `system`, merges consecutive assistant
messages, and never renders reasoning before the last user turn. Add the
family's expectations to `inference/vocabulary-check.zig`: every fixture
prompt must encode to its ids.

## 4. Validate against nuclis itself

nuclis is its own reference. The CPU reference, written from the
mathematical contract, is the root; Metal is checked against it on generated
tiny models and on the real file ([§ 5](#5-metal)); a second encoding of
the same model is checked against the first by the cross-encoding check
([§ 6](#6-a-second-encoding)). No external implementation is an oracle.

The first families (Qwen3.8, Gemma 4, Muse Glimmer, and Bonsai, the last
against Prism ML's fork) were brought up layer by layer against a pinned
llama.cpp build (`7620399`): its server rendered the profile fixtures, its
tokenizer gave the ids, and a harness linked against it wrote the layer
traces, the acceptance runs' token arrays, and the perplexity references.
The fixtures pinned then remain in the tree
([provenance.md](../../tests/fixtures/provenance.md)); the scripts that
made them were removed and are at tag `v0.11.0`. Since 2026-10-14 new work
is validated against nuclis's own CPU reference, Metal against it, and the
cross-encoding check ([spec § 3](../spec.md#3-decisions)); Saluki, that
day, was the first model brought up that way.

A genuinely new architecture has no validated path to compare with, so it
is validated by data, with traces from the authors' code pinned once at
bring-up ([spec § 3](../spec.md#3-decisions), decided 2026-10-27): its
permanent gates compare against human ground truth (word error rate on
pinned clips, a synthesis round trip through a validated recognizer), and
the traces are fixtures, not a standing oracle.

### The CPU reference

- `inference/src/models/<family>.zig`: the `family` namespace
  (`architecture`, `executableEncoding`, `Binding` and `bind`, `Runtime`,
  `Plan`; `models/registry.zig` lists the contract), a narrow validation of
  every metadata key it reads (including per-layer arrays), and tensor
  binding by name, shape, and encoding. Tests run over the inventory
  fixture: a successful bind, one typed rejection per validated fact
  (Gemma has 15), and `alloc_check.checkAll`. Register it with one
  line in `models/root.zig` `table`; `Adapter`, the engine's executor
  union, `validate`, `inspect`, and the "known architectures" diagnostic
  derive from it.
- `inference/src/models/<family>_runtime.zig`: the forward pass from § 1
  over `cpu.*`. Reuse what exists (`rmsNorm`, `matvec` with the quantized
  row decoders, `attention.apply`, `rope.apply`); a new mathematical
  operation is a new `cpu.*` function with its own unit test (Gemma added
  `cpu.gelu` and per-pair RoPE `factors`). Declare the session layout per
  layer (`session.Layout`): what state each layer has, at what width and
  precision.
- A tiny generated model of the family ([§ Tiny models](#tiny-models))
  runs its load and forward paths in every encoding the adapter accepts,
  with no download.
- Record the trace fixture on the real file with the CPU reference
  (`ReleaseFast`), commit it under `tests/fixtures/<family>-<prompt>/` with
  a row in [provenance.md](../../tests/fixtures/provenance.md), and add a
  `<family>-trace-cpu` gate to `gates.json` (tier `verify-cpu`, the
  family's seed globs in `families`;
  [development.md § Gates](../development.md#gates)):

```sh
nuclis generate --backend cpu --kv f32 --model <file> --prompt-tokens <ids.json> \
  --max-tokens 1 --ctx-size 8 --temperature 0 --logits <dir>/logits.f32 --trace-dir <dir>
./zig-out/bin/nuclis-gates compare <dir> tests/fixtures/<family>-… \
  --positions 3 --embedding 3840 --layers 48 --vocab 262144
```

Compare layer 0 at position 0 first: if it is off, the embedding scale or a
norm convention is wrong, not the attention. One CPU step of a 12B model is
about 35 s; that is a number for the plan, not a benchmark.

### Which executor makes which fixture

| Fixture | Made by | Why |
| --- | --- | --- |
| layer and logit traces, draft rows (MTP, DFlash), projector rows, greedy tokens | the CPU reference only (`ReleaseFast`) | they prove Metal right; a Metal-made row would agree with Metal's own mistake. Minutes at most |
| perplexity and other corpus statistics (`*-draft-stats`) | Metal, only once the file's CPU and Metal trace gates pass at that revision | the CPU runs `eval` a token at a time (one window of a 27B takes 28 min); these pins catch regressions, the traces judge correctness |
| speed and memory records | Metal | a new file starts a new baseline; never a difference against another file's record |

Fixtures pinned from llama.cpp earlier stay until their file is
re-pinned, and are then replaced by these, never re-made by an outside
tool. A pin nuclis made is re-made only when the file changes, never to
absorb an engine change.

## 5. Metal

`inference/src/models/<family>_metal.zig` records the same schedule as
`Backend` encoder calls, one command buffer per token (`step`) and per
prompt chunk (`prefill`). Before writing a kernel, ask what shape or
parameter the existing one lacks: Gemma needed four scalar epilogues, a
GELU mode of the fused gate pair, per-pair RoPE factors, a wider
instantiation of the decode attention template, and a window mask plus
value-column splits on the chunk attention kernel, and not one
Gemma-specific kernel ([metal-backend.md](../engine/metal-backend.md)).
Every new kernel or mode gets a `metal-check` fixture against `cpu.*` over
the new family's geometry.

On the tiny model first: `make test-tiny` holds the plan to the CPU
reference in every encoding, stepped and chunked, with no download. Then
on the real file: when the first step produces garbage, scan for
non-finite values before comparing traces (Metal's `tanh` overflows past
about ±44; the clamp at ±20 came from that probe). Then the
`<family>-trace-f32` gate at the bring-up thresholds (max abs 2e-3 and
relative RMS 1e-4, plus the same greedy token and top-5) and
`<family>-trace-f16` at a tolerance you *justify*: run the CPU reference
with its keys and values rounded to F16 and show the deviation is the
model's, not the kernels' (Gemma: 0.75 on the CPU with rounded keys, 0.73
on Metal). Both compare Metal against the CPU trace of § 4
(`make gate NAME='<family>-trace-*'`). Extend `generation-check` (the
`<family>-generation-metal` gate) with per-family tolerances for the
chunked-versus-stepped comparison; it also exercises snapshot/restore,
cancellation, and isolation, and it found a backend bug on its first run
(`Backend.unwrap`).

Take first-look rates with `nuclis bench` and record them as first look,
not as the acceptance record.

## 6. A second encoding

For a new file or encoding of a family nuclis already runs
([saluki.md](saluki.md) was the first):

1. **Decoders.** Unit tests on hand-built blocks pin every field of each
   new encoding (scales, indices, signs, offsets). Then the
   **cross-encoding check**, `zig build test-cross-encoding -- <new.gguf>
   <validated.gguf>`: the same model's tensors decoded from both files
   agree to the precision of the coarser encoding, and a negative control
   (a mismatched row) shows the margin. A wrong codebook or bit order
   lands near the control, not near the threshold.
2. **The forward.** It is the family's, already validated; record a CPU
   trace fixture for the new file as in § 4 and gate Metal against it
   (`<file>-trace-f32`, `<file>-trace-f16`).
3. **Quality.** `nuclis eval` perplexity of the new file next to the
   validated file's, both on nuclis.

### Re-pinning a file

For a new file of a model nuclis runs (Qwen3.8's re-made UD-Q4_K_M,
2026-11-12, is the first; [qwen3.8.md § The re-made file](qwen3.8.md#the-re-made-file-2026-11-12)):

1. Pull it, check the tokenizer gate and the chat template's SHA-256
   against the profile's, and run the cross-encoding check if the
   validated file is on the machine (record it as not run otherwise).
2. Record the CPU fixtures: `nuclis generate --backend cpu … --trace-dir`,
   `zig build -Doptimize=ReleaseFast test-generation -- <file>
   --draft-trace DIR` (and `--flash-trace DIR --draft-model <drafter>`),
   and `--vision-check <mmproj>`; replace the fixtures in place, with
   their `provenance.md` rows.
3. Run the family's Metal and CPU gates against them.
4. Pin Metal's eight-window perplexity (`nuclis eval --backend metal
   --chunks 8 --write-reference <json>`, which records the result, the
   text's and the model's digests, the revision, and the flags; add the
   `cross_check`), then run one window on the CPU against it (`eval
   --backend cpu --reference <it> --chunks 1`): more than about 0.05 %
   apart is a Metal bug, not a pin. The CPU window is the only independent
   check of batched prefill and the F16 cache on real prose for the new
   weights.
5. Record the new perplexity beside the old file's, measure speed as a
   new baseline, and run `make verify-long` once.

## 7. Acceptance

- `src/catalog.zig`: the index entry with the digests from § 1 (main
  file and companions), the architecture id, and the profile tag; `make
  index` rewrites `src/catalog.json` from it. `config init` writes an
  installed current entry as a registry entry; a user's existing registry
  entry of the same name shadows it.
- `nuclis --help` and the commands' pages name the profiles and their
  defaults; update them.
- Quality: a perplexity reference written by nuclis on the validated
  encoding (`nuclis eval --backend metal --chunks 8 --write-reference
  tests/fixtures/perplexity/<file>.json`, cross-checked by one CPU window
  as in [§ Re-pinning a file](#re-pinning-a-file)) and a
  `<family>-perplexity` gate that compares against it.
- Speed: a row in `workloads.json` and `make workload NAME=<row>/<shape>`
  (`code` takes a text prompt; `prose512`, `prose4096`, and `acceptance`
  replay a row's `run`, the token arrays of
  [benchmarks § Acceptance runs](../benchmarks/README.md#acceptance-runs)),
  recorded against nuclis's own previous record on the same machine and
  method, never against another file's. A new family's first record is
  its baseline.

Close with the pull request's record (design deviations, measured numbers,
gate results, what was left out), the documents the change touched, a
llm-guide section if a concept was new and the user asks for one,
`THIRD_PARTY_NOTICES.md` for the template's origin, and the diff checked
against the extension rule. Gates: `make check`, then `make verify-auto`,
which runs the canaries for shared code and names any further tier (the
other families' gates must be unchanged; the new family's
`<family>-trace-*`, `<family>-generation-metal`, `<family>-vocabulary`,
and `<family>-perplexity` gates, one set per pinned file when the family
has two, as Gemma's `gemma4-26b-a4b-*` and `gemma4-e4b-*`), and `make
verify-cpu` for the new CPU reference.

## Tiny models

Generated random-weight models run every language family's load and forward
paths on every change, with no download. `inference/src/models/tiny.zig`
builds a GGUF per variant at test time (`formats/gguf_writer.zig`,
`quant/sample.zig`), seeded and deterministic, in a temporary directory.

- **`zig build test`** (`inference/tiny-check.zig`, also `-Dmetal=false` on
  Linux): each family through `Engine.open` on the CPU, two rows plus a
  rerun, in each variant's first encoding: finite, the right shapes, same seed
  same logits; tokenizer round trips, GPT-4o's cuts on Muse. Adds 2 s.
- **`zig build test-tiny -Dmetal=true -- [variant] [seed]`** (`make
  test-tiny`): every variant in every encoding against the CPU reference,
  stepped and chunked, with the F16 and Q8_0 caches on the first encoding,
  and a control that corrupts 64 bytes of one matrix and must leave the
  stepped bound. It takes 2.5 minutes.

They prove the code paths, the plan's agreement with the reference, and the
layouts; never accuracy, since these weights mean nothing. Blocks are random
legal codes with a calibrated scale (`quant/sample.zig` says why), so an
encoder round trip is decoding them: finite, centred, the right RMS, and each
scale field linear. Drafters, DFlash, and vision are not covered.

**The tiny shapes live in a test-only build.** The adapters pin their
dimensions and the vocabulary (Muse all of them, Gemma three `Config` rows and
262,144, Qwen 248,320, the rotation the 27B shape), so no tiny file can pass
the production `bind`. The inference module is built a second time with the
build option `tiny_models`, which only `tiny-check` imports: it swaps those
constants for the smaller ones below. The validators are the same code; the
production module takes no new shape and has no bypass, and two tests in
`tiny.zig` hold that (the tiny Qwen shape passes production apart from the
vocabulary, and production refuses a shape only the tiny build pins).

| Variant | Shape (tiny build) | Encodings |
| --- | --- | --- |
| `qwen35` | 4 layers, 256 wide, DeltaNet and attention | F32, BF16, Q8_0, Q3_K to Q6_K, IQ3_S, IQ4_NL, IQ4_XS, PQ2_0, PTQ1_0 |
| `saluki`, `saluki-mix` | the same, a file per encoding and a mixed file | Q2_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS |
| `bonsai` | 4 layers, 1024 wide, rotated with BF16 `ssm_alpha` and `ssm_beta` | PQ2_0, PTQ1_0 |
| `muse-glimmer`, `muse-mix` | 4 layers, 256 wide, real attention geometry, GPT-4o tokenizer | F32, Q4_0, Q8_0, Q3_K to Q6_K, IQ3_S, IQ4_NL, IQ4_XS |
| `gemma4-dense`, `gemma4-mix` | 6 layers | F32, Q4_0, Q5_0, Q5_1, Q8_0, Q3_K to Q6_K, IQ3_S, IQ4_NL, IQ4_XS |
| `gemma4-experts` | 7 layers, 16 experts of 64 | the same, `ffn_down_exps` in Q5_0 under a 256-block file |
| `gemma4-e4b` | 12 layers, per-layer embedding | the same, its projection BF16 under Q4_K |

That is every encoding each adapter accepts. Bounds are about three times the
worst of three seeds over every encoding on the M4 Pro, in absolute logit
error and RMS relative to the reference's logits:

| Family | Stepped | Chunked | F16 cache | Q8_0 cache |
| --- | --- | --- | --- | --- |
| Qwen, Saluki, Bonsai | 6e-5, 1.5e-5 | 4e-2, 1e-2 | 3e-2, 1e-2 | 4e-2, 1e-2 |
| Muse Glimmer | 5e-6, 1e-5 | 3e-3, 5e-3 | 2e-3, 3e-3 | 3e-2, 5e-2 |
| Gemma 4 | 1.5e-4, 3e-5 | 2e-2, 6e-3 | 3e-2, 6e-3 | 1.5e-1, 5e-2 |
| Gemma 4 experts | 4e-4, 1.2e-4 | 2, 5e-1 | 2, 5e-1 | 2, 5e-1 |

Stepped runs F32 activations on both sides, so what is left is F32 rounding,
amplified by a random network. Chunked adds the half-operand tiles of the
encodings that use them, and the caches add their own. The experts' row is
wide because the router's top-k is discontinuous: rounding across a close
pair swaps an expert, which moves the logits far more than the rounding did.
The Gemma generator sets `attn_q_norm` near `1/sqrt(head)`, since its scores
are unscaled and at unit gain the softmax is one-hot, which amplified rounding
a thousandfold. A real model's numbers are the gates', never these.

To add a family, write its emitter (what `bind` requires, in the file's
tensor order and metadata), add a row to `variants` in `tiny.zig`, a test to
`tiny-check.zig`, its tiny constants behind `tiny_models` if the adapter pins
them, and its bounds.

`make test-tiny` also holds a family to what the server's scheduling rests
on: `prefill` and `prefillVision` report progress after every chunk
(`PrefillUnreported`), since speech and decisions run on those reports. The
image span's branch has no tiny projector, so a new family checks that one by
reading.

## Mistakes already made once

| Symptom | Cause | Where it is handled now |
| --- | --- | --- |
| `validate` rejects a 48-layer file | retained-array bound of 16 in the parser | bound raised to 64 (a format capacity) |
| `UnknownArchitecture` although the adapter exists | not registered in `models.table` | one line in `models/root.zig` |
| NaN in the first GPU step | `tanh` through `exp` overflowing in Metal | `nu_tanh` clamps at ±20 |
| `InvalidShape` from a plan that ran yesterday | wrapped buffers cached by address across a free | `Backend.unwrap` on plan `deinit` |
| F16 comparison thirty times worse than the first family's | unscaled attention amplifies key rounding | proven on the CPU with rounded keys; tolerance recorded with the reason |
| Model stops on `<bos>` or never on `<eos>` | the engine's stop rule was Qwen's (`eos or bos`) | `stop_tokens` on the profile, resolved at load |
| Thinking never folds in the chat | the split marker was the literal `</think>` | `reasoning` markers on the profile |
| Gemma sampled with Qwen's presence penalty | the configuration guessed the profile from a bare path | the engine's profile corrects the sampler after load |
