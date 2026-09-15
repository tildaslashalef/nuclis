# Bringing a new model family into nuclis

The starter guide the spec asked for
([spec § Interfaces and extension rules](../spec.md#interfaces-and-extension-rules)),
written while Gemma 4 12B went through the seam (MODL-04–MODL-07, 2026-09-11 to
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
  MODL-08: a fixture from the pinned reference decoder, the CPU row decoder,
  the generic GPU dequantizer (bit-identical to the CPU), a specialized
  matvec and matmul tile built from the nearest existing block kernel,
  the adapter's `executableEncoding`, and the file's own traces and
  acceptance record ([gemma4.md § Q4_0 path](gemma4.md#q4_0-path-and-the-qat-file-modl-08-2026-09-12)).
  Bring the adapter up on a file the kernels already execute first, if
  the repository offers one; the encoding unit then has an oracle of its
  own on the same architecture.
- Pull it (`nuclis model pull <owner/repo> --file <name>`) and record the
  commit, size, and SHA-256 from the sidecar in
  [artifacts.md](artifacts.md#pinned-commits-and-digests-modl-02-2026-09-11).
  The companions (`mmproj`, `mtp`) go into the same table; the catalogue
  entry will need their digests.
- Confirm the pinned llama.cpp reference (`7620399`,
  [reference-baseline.md](reference-baseline.md)) loads and runs the file
  (`llama-completion -ngl 99 -no-cnv`, then `--jinja --single-turn` for a
  chat answer). If it does not, there is no oracle and the unit stops here.

## 1. Facts with provenance (`docs/reference/<family>.md`)

Read the file, not the model card. `scripts/gguf-inventory.py <file>
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
byte fallback, MODL-05). Check it against the reference server's `/tokenize`
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

## 4. Pin the oracle and compare

`scripts/reference-generation.cpp` (built as in
[generation.md § Numerical traces](generation.md#numerical-traces))
writes every layer's output and the final logits for a short prompt; it
does not add BOS, so write `<bos>` in the prompt text. Run it once,
commit the traces under `tests/fixtures/<family>-<prompt>/` with a row in
[provenance.md](../../tests/fixtures/provenance.md), and add a
`make compare-<family>-cpu` target:

```sh
nuclis generate --backend cpu --model <file> --prompt-tokens <ids.json> \
  --max-tokens 1 --ctx-size 8 --temperature 0 --logits <dir>/logits.f32 --trace-dir <dir>
python3 scripts/compare-generation.py <dir> tests/fixtures/<family>-… \
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
Gemma-specific kernel ([metal-backend.md](metal-backend.md)). Every new
kernel or mode gets a `metal-check` fixture against `cpu.*` over the new
family's geometry.

When the first step produces garbage, commit after each operation and
scan for non-finite values before comparing traces (Metal's `tanh`
overflows past about ±44; the clamp at ±20 came from that probe). Then
`make compare-<family>-f32` at the bring-up thresholds and
`compare-<family>-f16` at a tolerance you *justify*: run the CPU
reference with its keys and values rounded to F16 and show the deviation
is the model's, not the kernels' (Gemma: 0.75 on the CPU with rounded
keys, 0.73 on Metal). Extend `generation-check` (`make
test-generation-<family>-metal`) with per-family tolerances for the
chunked-versus-stepped comparison; it also exercises snapshot/restore,
cancellation, and isolation, and it found a backend bug on the first run
(`Backend.unwrap`).

Take first-look rates with `nuclis bench` and `llama-bench` on the same
file and record them as first look, not as the acceptance record.

## 6. The prompt profile

Extract the template (`gguf-inventory.py --max-string 30000`, then the
`tokenizer.chat_template` value; keep it under `.zig-cache/`, never in
the tree) and read it for the subset the playground renders: system and
developer, user, assistant with separate reasoning, the thinking control.
Then capture before implementing: start the reference server on the file
and run `scripts/tokenizer-fixtures.py --profile <family>` (add the
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
`make test-vocabulary MODEL=<file>`: every captured prompt must encode to
the reference's ids.

## 7. Catalogue, CLI, and the acceptance record

- `src/catalog.zig`: the entry with the digests from step 0 (main file and
  companions), the architecture id, and the profile tag. `config init`
  writes it as a registry entry; a user's existing registry entry of the
  same name shadows it.
- `nuclis --help` names the profiles and their defaults; update it.
- The acceptance record: the reference harness on the file
  (`scripts/reference-baseline.py --family <family> --output-dir
  .zig-cache/reference/<family>-<date> --prompt-lengths
  512,4096,16384,32639` against the server started with the
  [reference recipe](reference-baseline.md#run-the-workload) at 32K),
  its token arrays committed under `tests/fixtures/run-<date>-<family>/`,
  a `docs/benchmarks/reference-<date>-<family>.json` summary, then
  `scripts/nuclis-baseline.py --run run-<date>-<family>
  --reference-records reference-<date>-<family>.json` (a `make
  baseline-<family>` target) and the table in
  [bench.md](bench.md#acceptance-runs) with hardware, build, artifact,
  context, and methodology stated.

## 8. Close

The engineering log entry (design deviations, measured numbers, gate
results, what was left out), the reference documents the unit changed,
a llm-guide section if a concept was new, `THIRD_PARTY_NOTICES.md` for
the template's origin, and the diff checked against the extension rule.
Gates: `make check`, `make compare` (the first family must be unchanged),
`make compare-<family>` (one target per pinned file when the family has
two, as Gemma's `compare-gemma4` and `compare-gemma4-qat`), `make
test-generation-metal` and `-<family>-metal`, `make bench` (the first
family's rate unchanged).

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
