# Bringing a new model family into nuclis

The starter guide the spec asked for
([spec § Interfaces and extension rules](../spec.md#57-extension-rules)),
written while Gemma 4 12B went through the seam ([MODL-04](https://github.com/tildaslashalef/nuclis/blob/v0.6.0/docs/worklog.md#modl-04--adapter-registry-2026-09-11)–[MODL-07](https://github.com/tildaslashalef/nuclis/blob/v0.6.0/docs/worklog.md#modl-07--gemma-4-12b-profile-catalogue-acceptance-new-model-guide-2026-09-12), 2026-09-11 to
2026-09-12) and validated by that use. It is the order of work, the
commands, the gates, and the mistakes already made once. The extension
rule it enforces: a new family is a new adapter, its tests and prompt
profile, and any genuinely new mathematical operation, with no edits to
the GGUF parser, the sampling policy, the CLI generation loop, or the
Metal object lifecycle. Check the diff against that list before closing.

Numbers and file names below are Gemma's; substitute yours. Every step
ends with something committed as evidence (a fixture, a trace directory,
a table in a reference document), because the next step will need it and
the transcript will not survive.

## 0. Before writing code: decide the artifact and pin it

- Pick the checkpoint and quantization by `nuclis model inspect
  <owner/repo> --file <name>` (no download): it prints the directory, the
  encodings per tensor, and the verdict (`supported`, `runnable`, `not
  runnable` naming the first tensor whose layout no kernel executes).
  An encoding the tree does not execute is a kernel unit of its own;
  never plan to requantize. Gemma's QAT file (Q4_0) took one session,
  [MODL-08](https://github.com/tildaslashalef/nuclis/blob/v0.6.0/docs/worklog.md#modl-08--q4_0-path-and-the-qat-catalogue-entry-2026-09-12): a fixture from the pinned reference decoder, the CPU row decoder,
  the generic GPU dequantizer (bit-identical to the CPU), a specialized
  matvec and matmul tile built from the nearest existing block kernel,
  the adapter's `executableEncoding`, and the file's own traces and
  acceptance record ([gemma4.md § Q4_0 path](gemma4.md#q4_0-path-and-the-qat-file-2026-09-12)).
  Bring the adapter up on a file the kernels already execute first, if
  the repository offers one; the encoding unit then has an oracle of its
  own on the same architecture.
- Pull it (`nuclis model pull <owner/repo> --file <name>`) and record the
  commit, size, and SHA-256 from the sidecar in
  [catalogue.md](catalogue.md#pinned-commits-and-digests-2026-09-11).
  The companions (`mmproj`, `mtp`) go into the same table; the catalogue
  entry will need their digests.
- Since 2026-10-14 nuclis is its own reference ([spec § 3](../spec.md#3-decisions)): a new
  file or encoding of a family nuclis already runs is validated by § 4
  against nuclis's own paths, with no external oracle. The steps below that
  name the pinned llama.cpp reference (`7620399`,
  [llama-cpp.md](../benchmarks/llama-cpp.md)) are how the families before
  that date were brought up. A genuinely new architecture is validated by
  data, with traces from the authors' code pinned once at bring-up
  ([spec § 3](../spec.md#3-decisions), decided 2026-10-27): its permanent
  gates compare against human ground truth (word error rate on pinned
  clips, a synthesis round trip through a validated recognizer), and the
  traces are fixtures, not a standing oracle.

## 1. Facts with provenance (`docs/models/<family>.md`)

Read the file, not the model card. `scripts/fixtures/gguf-inventory.py <file>
--out inference/src/models/fixtures/<family>.json` is an independent
header reader (standard library only): every metadata key, arrays up to
64 elements, every tensor's name, shape, and encoding, and the chat
template as a length/offset/SHA descriptor. That JSON is the binder's
test fixture. Then read the reference's model source
(`src/models/<arch>.cpp`, `llama-vocab.cpp`, the converter) for the
*semantics*: the forward pass in equations, norms (raw or shifted
weights?), RoPE pairing and factors, attention scale, activation, soft
caps, tied heads, layer patterns. Write it down as
[gemma4.md § Forward pass](gemma4.md#forward-pass-from-the-reference-graph-gemma4cpp)
does: notation, per-layer steps, and where each fact came from. Anything
that came from the reference source rather than the file says so.

Watch for format capacities disguised as model facts: Gemma's 48-element
per-layer arrays exceeded the parser's retained-array bound (16, a
Qwen-era limit). Raising a *bound* in `formats/gguf` is allowed; adding
a Gemma key to it is not.

## 2. Tokenizer

`inference/src/tokenizer/vocabulary.zig` lists the `tokenizer.ggml.model`
values it accepts and `tokenizer/encode.zig` the splitters. A new model
or `pre` value needs its own splitter and possibly its own merge alphabet
(Gemma: SPM-style BPE over code points with U+2581 spaces and `<0xNN>`
byte fallback, [MODL-05](https://github.com/tildaslashalef/nuclis/blob/v0.6.0/docs/worklog.md#modl-05--gemma-4-12b-facts-binding-cpu-reference-tokenizer-2026-09-11); Muse: the `llama4` label, [MODL-11](https://github.com/tildaslashalef/nuclis/blob/v0.6.0/docs/worklog.md#modl-11--muse-glimmer-30b-artifact-pin-facts-tokenizer-binding-cpu-reference-2026-09-19)). Read the
regex the reference *runs* for the label, not the one the tokenizer
declares: `llama-vocab.cpp` may substitute a rewritten form, and the
generic path in `unicode.cpp` collapses every Unicode class it does not
know into one, so `\p{Lu}` and `\p{Ll}` both mean "letter" there. `scripts/reference-split.cpp`, a harness linking the reference's
`unicode.cpp` and printing `unicode_regex_split` pieces, settles
boundaries that token ids cannot (BPE merges hide them); the splitter's
unit tests pin those pieces.
The artifact check (`inference/vocabulary-check.zig`) selects its
expectations by template digest, so a family can have its tokenizer
checked before its profile exists. Check it against the reference server's `/tokenize`
on a dozen adversarial strings (newline runs, doubled spaces, CRLF, tabs,
emoji, the chat markers inside text) before anything else, because the
traces in step 4 need the ids. Note whether the reference forces BOS for
this vocabulary: nuclis's encoder never adds BOS, so a family that needs
it writes `<bos>` as text (raw prompts and the profile alike).

## 3. Binding and the CPU reference

- `inference/src/models/<family>.zig`: the `family` namespace
  (`architecture`, `executableEncoding`, `Binding` and `bind`, `Runtime`,
  `Plan`), a narrow validation of every metadata key it reads (including
  per-layer arrays), and tensor binding by name, shape, and encoding.
  Tests run over the inventory fixture: a successful bind, one typed
  rejection per validated fact (Gemma has 15), and
  `checkAllAllocationFailures`. Register it with one line in
  `models/root.zig` `table`; `Adapter`, the engine's executor union,
  `validate`, `model inspect`, and the "known architectures" diagnostic
  derive from it. The Metal plan is a stub returning `BackendUnavailable`
  until step 5.
- `inference/src/models/<family>_runtime.zig`: the forward pass from step
  1 over `cpu.*`. Reuse what exists (`rmsNorm`, `matvec` with the
  quantized row decoders, `attention.apply`, `rope.apply`); a new
  mathematical operation is a new `cpu.*` function with its own fixture
  (Gemma added `cpu.gelu` and per-pair RoPE `factors`). Declare the
  session layout per layer (`session.Layout`): what state each layer has,
  at what width and precision.

## 4. Validate against nuclis itself

For a new file or encoding of a family nuclis already runs
([saluki.md](saluki.md) is the first):

1. **Decoders.** Unit tests on hand-built blocks pin every field of each
   new encoding (scales, indices, signs, offsets). Then the
   **cross-encoding check**, `zig build test-cross-encoding -- <new.gguf>
   <validated.gguf>`: the same model's tensors decoded from both files
   agree to the precision of the coarser encoding, and a negative control
   (a mismatched row) shows the margin. A wrong codebook or bit order
   lands near the control, not near the threshold.
2. **The forward.** It is the family's, already validated; record a CPU
   trace fixture for the new file with nuclis's CPU reference
   (`nuclis generate --backend cpu … --trace-dir`, as below) and gate
   Metal against it (`<file>-trace-f32`, `<file>-trace-f16`).
3. **Quality.** `nuclis eval` perplexity of the new file next to the
   validated file's, both on nuclis.

The trace procedure the earlier families used against llama.cpp, kept
for them:

`scripts/reference-generation.cpp` (built as in
[sampling.md § Numerical traces](../engine/sampling.md#numerical-traces))
writes every layer's output and the final logits for a short prompt; it
does not add BOS, so write `<bos>` in the prompt text. Run it once,
commit the traces under `tests/fixtures/<family>-<prompt>/` with a row in
[provenance.md](../../tests/fixtures/provenance.md), and add a
`<family>-trace-cpu` gate to `gates.json` (tier `verify-cpu`, the family's
seed globs in `families`; [development.md § Gates](../development.md#gates)):

```sh
nuclis generate --backend cpu --model <file> --prompt-tokens <ids.json> \
  --max-tokens 1 --ctx-size 8 --temperature 0 --logits <dir>/logits.f32 --trace-dir <dir>
./zig-out/bin/nuclis-gates compare <dir> tests/fixtures/<family>-… \
  --positions 3 --embedding 3840 --layers 48 --vocab 262144
```

Bring-up thresholds are max abs 2e-3 and relative RMS 1e-4 on every
file, plus the same greedy token and top-5. Compare layer 0 at position 0
first: if it is off, the embedding scale or a norm convention is wrong,
not the attention. One CPU step of a 12B model is about 35 s; that is a
number for the plan, not a benchmark.

## 5. The Metal plan

`inference/src/models/<family>_metal.zig` records the same schedule as
`Backend` encoder calls, one command buffer per token (`step`) and per
prompt chunk (`prefill`). Before writing a kernel, ask what shape or
parameter the existing one lacks: Gemma needed four scalar epilogues, a
GELU mode of the fused gate pair, per-pair RoPE factors, a wider
instantiation of the decode attention template, and a window mask plus
value-column splits on the chunk attention kernel, and not one
Gemma-specific kernel ([metal-backend.md](../engine/metal-backend.md)). Every new
kernel or mode gets a `metal-check` fixture against `cpu.*` over the new
family's geometry.

When the first step produces garbage, commit after each operation and
scan for non-finite values before comparing traces (Metal's `tanh`
overflows past about ±44; the clamp at ±20 came from that probe). Then
the `<family>-trace-f32` gate at the bring-up thresholds and
`<family>-trace-f16` at a tolerance you *justify*: run the CPU
reference with its keys and values rounded to F16 and show the deviation
is the model's, not the kernels' (Gemma: 0.75 on the CPU with rounded
keys, 0.73 on Metal). Extend `generation-check` (the
`<family>-generation-metal` gate) with per-family tolerances for the
chunked-versus-stepped comparison; it also exercises snapshot/restore,
cancellation, and isolation, and it found a backend bug on the first run
(`Backend.unwrap`).

Take first-look rates with `nuclis bench` and record them as first look,
not as the acceptance record.

## 6. The prompt profile

Extract the template (`gguf-inventory.py --max-string 30000`, then the
`tokenizer.chat_template` value; keep it under `.zig-cache/`, never in
the tree) and read it for the subset the playground renders: system and
developer, user, assistant with separate reasoning, the thinking control.
Then capture before implementing: start the reference server on the file
and run `scripts/fixtures/tokenizer-fixtures.py --profile <family>` (add the
profile's digests, efforts, and template keyword arguments to its
`PROFILES` table). The fixture shows what the reference actually sends
the model, which is not always what the template or the card says: for
Gemma, no `<bos>` in the server's output (its tokenizer adds one),
`developer` as `system`, consecutive assistant messages merged, reasoning
never rendered before the last user turn.

`inference/src/profiles/<family>.zig` exposes `template_sha256`,
`render`, `samplingDefaults`, `stop_tokens`, `reasoning`, and
`stream_markers`; register
the tag in `profiles/root.zig`. The stop tokens are texts (the engine
resolves them at load and refuses a vocabulary without them); the
reasoning markers are what the model *emits*, not what the prompt
contains. `stream_markers` separates actual control-token text from any
ordinary opening suffix (Gemma's `thought\n`). `Profile.decoder` resolves
these controls in the vocabulary and `engine.complete` emits semantic
channels; add fake-token decoder tests before exercising a live completion.
Sampling defaults come with provenance (a model card table, or
the file's `general.sampling.*` hint recorded as a claim). Add the
family's expectations to `inference/vocabulary-check.zig` and run
a `<family>-vocabulary` gate: every captured prompt must encode to
the reference's ids.

## 7. Catalogue, CLI, and the acceptance record

- `src/catalog.zig`: the entry with the digests from step 0 (main file and
  companions), the architecture id, and the profile tag. `config init`
  writes it as a registry entry; a user's existing registry entry of the
  same name shadows it.
- `nuclis --help` names the profiles and their defaults; update it.
- The acceptance record: the reference harness on the file
  (`scripts/bench/reference-baseline.py --family <family> --output-dir
  .reference/<family>-<date> --prompt-lengths
  512,4096,16384,32639` against the server started with the
  [reference recipe](../benchmarks/llama-cpp.md#run-the-workload) at 32K),
  its token arrays committed under `tests/fixtures/run-<date>-<family>/`,
  a `docs/benchmarks/reference-<date>-<family>.json` summary, then
  `scripts/bench/nuclis-baseline.py --run run-<date>-<family>
  --reference-records reference-<date>-<family>.json` (a `make
  baseline-<family>` target) and the table in
  [benchmarks](../benchmarks/README.md#acceptance-runs) with hardware, build, artifact,
  context, and methodology stated.

## 8. Close

The pull request's record (design deviations, measured numbers, gate
results, what was left out), the reference documents the unit changed,
a llm-guide section if a concept was new, `THIRD_PARTY_NOTICES.md` for
the template's origin, and the diff checked against the extension rule.
Gates: `make check`, then `make verify` (the first family's gates must be
unchanged; the new family's `<family>-trace-*` and `<family>-generation-metal`
gates, one set per pinned file when the family has two, as Gemma's
`gemma4-*` and `gemma4-qat-*`), and `make bench` (the first family's rate
unchanged).

## Tiny models

Generated random-weight models run every language family's load and forward
paths on every change, with no download. `inference/src/models/tiny.zig`
builds a GGUF per variant at test time (`formats/gguf_writer.zig`,
`quant/sample.zig`), seeded and deterministic, in a temporary directory.

- **`zig build test`** (`inference/tiny-check.zig`, also `-Dmetal=false` on
  Linux): each family through `Engine.open` on the CPU, two rows plus a
  rerun, in each variant's first encoding: finite, the right shapes, same seed
  same logits; tokenizer round trips, GPT-4o's cuts on Muse. Adds 2 s.
- **`zig build test-tiny -Dmetal=true -- [variant] [seed]`**: every variant in
  every encoding against the CPU reference, stepped and chunked, with the F16
  and Q8_0 caches on the first encoding, and a control that corrupts 64 bytes
  of one matrix and must leave the stepped bound. It takes 2.5 minutes.

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
reading. A new vocabulary needs its tokens typed: anything but `normal` and
`byte` is structure the stream decoder keeps out of the text.

## Mistakes already made once

| Symptom | Cause | Where it is handled now |
| --- | --- | --- |
| `validate` rejects a 48-layer file | retained-array bound of 16 in the parser | bound raised to 64 (a format capacity) |
| `UnknownArchitecture` although the adapter exists | not registered in `models.table` | one line in `models/root.zig` |
| NaN in the first GPU step | `tanh` through `exp` overflowing in Metal | `nu_tanh` clamps at ±20 |
| `InvalidShape` from a plan that ran yesterday | wrapped buffers cached by address across a free | `Backend.unwrap` on plan `deinit` |
| F16 comparison thirty times worse than the first family's | unscaled attention amplifies key rounding | proven on the CPU with rounded keys; tolerance recorded with the reason |
| Fixture capture refuses the server's template | `/props` omits the file's trailing newline | the script accepts both digests |
| Model stops on `<bos>` or never on `<eos>` | the engine's stop rule was Qwen's (`eos or bos`) | `stop_tokens` on the profile, resolved at load |
| Thinking never folds in the chat | the split marker was the literal `</think>` | `reasoning` markers on the profile |
| Gemma sampled with Qwen's presence penalty | the configuration guessed the profile from a bare path | the engine's profile corrects the sampler after load |
