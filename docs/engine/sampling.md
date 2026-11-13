<!-- kind: reference -->
# Generation and sampling

The complete 64-layer Qwen text schedule runs natively in Zig: mapped quantized
weights, embeddings, 48 convolution/DeltaNet layers, 16 gated attention layers,
residuals, feed-forward blocks, final normalization, and vocabulary logits.
There is no llama.cpp dependency in nuclis.

This is the CPU reference, the slow correctness backend every other path
is checked against
([models § 4](../models/README.md#4-validate-against-nuclis-itself)). `--backend metal` runs the same schedule
GPU-resident through the [Metal backend](metal-backend.md), with chunked
prefill, the measured rates of the README's [Results](../../README.md#results),
and teacher-forced scoring ([guide/eval.md](../guide/eval.md)). Default
tests need neither weights nor GPU access.

## Run

```sh
zig build -Doptimize=ReleaseSafe
./zig-out/bin/nuclis generate --prompt 'Hi' --max-tokens 3 --ctx-size 64
./zig-out/bin/nuclis generate --raw --prompt 'Hello,' --max-tokens 2 --ctx-size 8 --json
```

The default model is the catalogue entry `qwen3.8-27b` under `~/.nuclis/models/unsloth/Qwen3.8-27B-GGUF/`; `--model` and
`NUCLIS_HOME` work as for inspection. Keep the file unchanged while mapped.
The backend, context, output budget, reasoning effort, and sampling
overrides come from `~/.nuclis/nuclis.json` when it exists
([guide/configuration.md § Configuration file](../guide/configuration.md#configuration-file));
flags override the file per option.
The default prompt profile renders one user turn with thinking disabled;
`--think off|low|medium|high|xhigh` selects the reasoning effort the template
encodes. `--raw` supplies literal prompt text, recognizing special-token markers without
inserting BOS or EOS. `--prompt` is limited to 64 KiB and `--prompt-file` to
4 MiB of valid UTF-8; the two are exclusive and neither is ever truncated.

Defaults (without a configuration file) are the Metal backend when built
in, 8,192 output tokens, 16,384 total context, and seed zero. Sampling
defaults come from the official Qwen3.8 profile of the reasoning mode (next
section); the file's `generation.sampling` entries and then `--temperature`,
`--top-k`, `--top-p`, `--min-p`, `--presence-penalty`, and
`--repetition-penalty` each override one option, and `--temperature 0`
selects greedy. Greedy ties select the lowest token
ID. Nonfinite logits are errors.

### Sampling profiles and the selection chain

`generate` and `agent` sample with the checkpoint's official settings of the
reasoning mode `--think` selects. The tables live in the profile modules
(`samplingDefaults`) because they are part of the checkpoint's recommended
usage, not of the shared sampler, whose own defaults are neutral (greedy).
The profile is the file's, selected by its template digest at load; the
configuration's `config show` names the catalogue entry's profile (or the
first profile for a bare path) without opening the file, and a run corrects
the sampler to the file's profile once the model is open. Qwen3.8, from the
[Unsloth Qwen3.8 guide](https://unsloth.ai/docs/models/qwen3.8.md) (read
2026-09-08):

| Option | Thinking (`low`, `medium`, `high`, `xhigh`) | Instruct (`off`) |
| --- | ---: | ---: |
| `temperature` | 1.0 | 0.7 |
| `top_p` | 0.95 | 0.80 |
| `top_k` | 20 | 20 |
| `min_p` | 0.0 | 0.0 |
| `presence_penalty` | 0.0 | 1.5 |
| `repetition_penalty` | 1.0 | 1.0 |

Gemma 4 (`inference/src/profiles/gemma4.zig`) uses the file's own
sampling hint in every mode: temperature 1.0, `top_p` 0.95, `top_k` 64, no
`min_p`, no penalties (`general.sampling.*` in the header; no per-mode
table was pinned from a model card).

Flags override per option, so `--temperature 0` under `--think off` is a
greedy argmax over logits that still carry the instruct presence penalty;
add `--presence-penalty 0` for the plain argmax (the GPU argmax path). The
agent rebuilds the options from the current effort at every turn (Ctrl-T
switches profiles) and never resets the RNG. `bench` never applies a
profile: unflagged runs are greedy so its numbers stay comparable, and a
sampled `bench` states its options in the report label.

The chain, pinned by the sampler's unit tests rather than by convention:

1. **Penalties** on the raw logits of every token in the session's
   `History` (prompt and generated tokens; the engine loop observes them
   and resets the history wherever it resets the session): repetition
   first (`l / r` for positive, `l · r` for negative logits), then
   `l − presence_penalty`. They apply to greedy decoding too, since they
   change the argmax.
2. **Sort** by (value descending, id ascending), **temperature** in the
   exponentiation, the **top-k** cut.
3. **`min_p`**: after exponentiation a candidate's weight is exactly
   `p / p_max`, so the survivors are the leading candidates with
   `weight ≥ min_p` (a prefix of the sorted set; the top candidate always
   survives).
4. **top-p** over the survivors' sum, then one seeded draw.

Validation (`InvalidSamplingOptions`, before the model loads): temperature
finite and ≥ 0; top-p in (0, 1]; `min_p` in [0, 1]; presence penalty finite;
repetition penalty finite and > 0.

### Prompt processing

On the Metal backend the prompt is consumed in chunks of up to 256 tokens
through batched matrix kernels (see
[metal-backend.md § Prefill in chunks](metal-backend.md#prefill-in-chunks));
`--trace-dir` requests per-layer activations per token and therefore steps
the prompt token by token, as the CPU reference always does. Chunked and
stepped prompts agree within the tolerance recorded there, not bit for bit.
`prefill_milliseconds` keeps its definition (all prompt tokens, logits for
the last).

### Sampling on the GPU without reading the vocabulary back

The reference sampler (`inference/src/sampling/root.zig`, `Sampler.select`)
sorts all 248,320 logits by (value descending, id ascending), exponentiates
the retained top-k set relative to the maximum in F64, walks it until the
running sum reaches `top_p · sum`, and draws once. On the Metal backend the
same token is produced from a 2 KB readback instead of 1 MB of logits:

- Three kernels after the output projection (`nu_topk_partial`,
  `nu_topk_final`, `nu_expsum_partial`) produce the best 256 logits in the
  reference order (ties resolve to the lower id on the GPU exactly as in the
  sort) and Σ exp((l − max) / T) over every logit as 64 F32 partial sums; the
  CPU adds the partials in F64 in a fixed order.
- `Sampler.selectFrom` rebuilds the reference computation from the readback.
  Every floating-point step that touches the chosen token — exponentiation,
  the running sum, the threshold comparison, the draw — runs on the CPU in
  the reference order on the same values, so the result is bit-identical
  wherever it is produced. The GPU contributes only the candidate set and
  the denominator.
- **Eligibility.** With a penalty active (`presence_penalty ≠ 0` or
  `repetition_penalty ≠ 1`, the instruct profile) the readback must be taken
  *after* the device applied it: `nu_penalize` runs between the output head
  and the argmax/top-k, the plan marks the `TopK` `penalized`, and
  `selectFrom` then treats it as the penalized vector. A readback without
  that mark defers. Then `1 ≤ top_k ≤ 256`: the retained set is inside the
  readback and its denominator is computed on the CPU; always exact, never
  falls back, with or without `min_p`. `top_k = 0, min_p > 0`: the survivors are a prefix of
  the readback and their sum is the exact CPU sum, so the decision is
  exact; only when all 256 candidates survive the filter does the sampler
  **defer** (the prefix may continue past the readback). `top_k = 0, top_p
  < 1, min_p = 0`: the nucleus ranges over the vocabulary, so the GPU
  denominator is used; its relative error is bounded (measured 9.6e-8 on a
  flat vector by `test-metal`, asserted ≤ 2e-6) and the sampler only
  accepts a threshold comparison whose margin exceeds `total_band = 1e-5`;
  a comparison inside the band, or a nucleus that runs past the 256
  candidates, **falls back**. Not eligible (the full path runs every token):
  `top_k > 256` and `top_k = 0` with `top_p = 1, min_p = 0`.
- **Fallback** reads the full logits from the GPU's shared buffer
  (`Plan.readLogits`, valid until the next step) and runs `select`; the RNG
  has not advanced, so the token equals the reference path's. A non-finite
  logit anywhere raises a per-partition flag and also falls back, where the
  reference path reports `NonFiniteResult` as before. `generate --json`
  reports `gpu_topk_fallbacks`; `bench` shows a per-run fallback column.
- The `--logits` option needs the full vector and therefore keeps the
  reference path; comparing a `--logits` run with a plain run at the same
  seed is the end-to-end equivalence check (recorded in [benchmarks](../benchmarks/README.md)).

Greedy decoding with a penalty active takes the same device argmax, after
`nu_penalize`; the raw logits are only read back when a full vector was
asked for (a trace or `--logits`), and then it is unpenalized for the CPU
sampler.

### Speculative verification

With a drafter loaded and the speculative switch on, the loop batches the
step: it proposes drafts, verifies them in one forward, accepts the longest
prefix, and emits the accepted drafts and the correction through the
ordinary per-token path (history, hooks, stop, budget, context limit).
Every emitted token is a draw from the target, and greedy speculation
equals ordinary greedy token for token. The verify modes, the acceptance
rule, and the device-readback decision are in
[speculative-decoding.md](speculative-decoding.md#the-loop).

### Bounds, output, and cancellation

The output budget and the context are bounded by host constants, with
prompt plus output budget required to fit:

<!-- gen:limits -->
| Limit | Value | Declared in |
| --- | --- | --- |
| context ceiling (tokens) | 32,768 | inference/src/runtime/session.zig `max_capacity` |
| output budget (tokens) | 16,384 | src/config.zig `max_output_tokens` |
| draft length | 15 | inference/src/engine.zig `max_draft_length` |
| prefill chunk (tokens) | 256 | inference/src/engine.zig `prefill_chunk` |
| config file (bytes) | 65,536 | src/config.zig `max_file_bytes` |
| agent steps per turn | 16 | src/agent/loop.zig `budget_default` |
| agent turns per session | 256 | src/agent/loop.zig `max_turns` |
| agent instructions file (bytes) | 8,192 | src/agent/system_prompt.zig `max_instructions_bytes` |
| read_file lines per call | 2,000 | src/agent/tools/read_file.zig `max_lines` |
| read_file default lines | 120 | src/agent/tools/read_file.zig `default_lines` |
| read_file bytes | 1,048,576 | src/agent/tools/read_file.zig `max_bytes` |
| grep matches shown | 50 | src/agent/tools/grep.zig `max_shown` |
| grep files scanned | 20,000 | src/agent/tools/grep.zig `max_files` |
| grep seconds | 10 | src/agent/tools/grep.zig `max_seconds` |
| glob results | 200 | src/agent/tools/glob.zig `max_results` |
| image tokens per image | 4,096 | inference/src/vision/root.zig `max_image_tokens` |
| server output tokens | 16,384 | src/api/chat/wire.zig `max_output_tokens` |
| server audio clip (bytes) | 16,777,216 | src/api/chat/wire.zig `max_audio_bytes` |
| server web rounds | 6 | src/api/chat/service.zig `max_web_rounds` |
| server speech input (bytes) | 16,384 | src/api/speech/wire.zig `max_input_bytes` |
| server live transcriptions | 16 | src/api/transcription/stream.zig `max_sessions` |
| decision questions per call | 32 | inference/src/decide.zig `max_questions` |
| decision states per call | 64 | inference/src/decide.zig `max_states` |
<!-- /gen -->

These are allocation/execution bounds, not a validated 32K performance claim. Both `<|im_end|>` and `<|endoftext|>` stop
the pinned profile. Special tokens are omitted from presented text.

Text is flushed after each token. The UTF-8 stream buffers incomplete characters
across tokens; malformed bytes and an incomplete character at termination become
U+FFFD. JSON (schema 2) buffers the result and includes backend, sampling
settings, seed, prompt count, exact generated IDs, text, `stop_reason`
(`eos`, `token_budget`, `context_limit`, or `cancelled`), and load/prefill/first-token/decode
milliseconds with GPU busy time on Metal; definitions are in [benchmarks](../benchmarks/README.md).
Token IDs preserve information lost by text presentation.

Cancellation is checked at each layer boundary through the observer's `check`
callback (injected Zig Io and a Ctrl-C flag, with a one-hour execution bound);
it needs no activations, so on the GPU plan it runs while the token is being
recorded and costs no synchronization. The `layer` callback, which receives
activations and forces a GPU commit per layer, is installed only with
`--trace-dir`. The third callback, `progress`, reports how far a turn
has got — `{phase, position, target}` after every prefill chunk on the GPU,
every prompt token on the CPU, and every generated token. It exists because
a chunked prefill is one `step` for the loop's hooks: a caller counting
calls would see a 1,226-token prompt as a single event. `generate` and
`bench` do not install it. Runtime callback errors poison the session until reset. Reset
clears KV, convolution history, and recurrent matrices. Ctrl-C sets a flag in
a one-shot signal handler; the loop observes it at the next layer boundary
(on the GPU, the partially recorded token still executes before the session
is reset), resets the poisoned session, and reports
`stop_reason: cancelled` with the tokens completed so far (text mode prints the
completed text and an info line on stderr; JSON is emitted normally). A second
Ctrl-C terminates the process. Text already flushed before an unexpected error
remains visible; JSON is emitted only on success or cancellation.

## Numerical traces

`--logits PATH` writes the final prompt's 248,320 logits as little-endian F32.
`--trace-dir EXISTING_DIRECTORY` writes each complete layer's 5,120 F32 values,
named `token-POSITION-layer-LAYER.f32`. These diagnostics may contain prompt
information; keep them in ignored local storage. Later generated-token steps
produce layer traces too, but do not overwrite the initial prompt logits file.

The reference is nuclis's own CPU path. `tests/fixtures/reference-hello-comma/`
holds the CPU reference's traces of the raw `Hello,` prompt on the pinned
file (provenance in
[tests/fixtures/provenance.md](../../tests/fixtures/provenance.md)), and
`make gate NAME=qwen38-trace-f32` checks the Metal plan against them:

```sh
mkdir -p .zig-cache/generation/native-check
./zig-out/bin/nuclis generate --backend metal --kv f32 --raw --prompt 'Hello,' \
  --max-tokens 1 --ctx-size 8 \
  --logits .zig-cache/generation/native-check/logits.f32 \
  --trace-dir .zig-cache/generation/native-check
./zig-out/bin/nuclis-gates compare \
  .zig-cache/generation/native-check tests/fixtures/reference-hello-comma --positions 2 \
  --embedding 5120 --layers 64 --vocab 248320
```

The same command with `--backend cpu` writes the fixture itself
([provenance: record](../benchmarks/record/sampling.md#trace-provenance-2026-09-06)).
`compare` takes the geometry as flags (`--embedding`, `--layers`,
`--vocab`), so the same pair serves Gemma 4: `gemma4-26b-a4b-trace-*` and
`gemma4-e4b-trace-*` compare against their own `*-hello-comma` fixtures
([gemma4.md](../benchmarks/record/gemma4.md#cpu-reference-against-the-oracle-2026-09-11)).
Top-five IDs and the reference's greedy margin are reported for diagnosis.

The draft block has the same kind of check. `generation-check` (`zig build
test-generation`) with `--draft-trace DIR` steps the pinned `Hello,` tokens
through the block on the CPU reference or the Metal plan (`--metal`) and
writes `p0-h.f32` (the block's `h_nextn` at position 0, `h_prev` zeros),
`p1-hprev.f32` (the target's `output_norm`-applied hidden after position
0, which is what the block consumes), `p1-h.f32`, and `greedy.txt`.
`nuclis-gates compare --draft` compares those rows and the greedy tokens
against `inference/src/models/fixtures/qwen35-mtp/`, the CPU reference's
rows of the pinned file (`make gate NAME=qwen38-draft-trace-cpu`,
`qwen38-draft-trace-metal`); `make gate NAME=qwen38-generation-cpu` checks
the same vectors inside `generation-check` (its `checkDraft` pass).
`--draft-stats` instead reports the per-depth acceptance statistic and
draft latency
([speculative-decoding.md § Evidence](speculative-decoding.md#evidence)).

The comparison checks every requested layer and final logits, rejecting missing,
wrong-sized, or nonfinite files. The bounds are maximum absolute error 0.002
and relative RMS error 0.0001 for every tensor: local full-model smoke
thresholds, not general per-kernel or quality acceptance limits. They hold
for the F32 cache (`make gate NAME=qwen38-trace-f32`); the F16 cache has its
own documented tolerance, 0.03 / 0.0002 on the layer files with the logits
inside the F32 numbers (`make gate NAME=qwen38-trace-f16`;
[metal-attention.md § F16 KV cache](metal-attention.md#f16-kv-cache)).
The `.zig-cache/generation/` paths are working copies, never committed.

The explicit session test runs the real two-token sequence in two independent
sessions and after reset following injected cancellation in layer 3. Logits must
be bit-identical within the same CPU backend:

```sh
zig build test-generation -Doptimize=ReleaseSafe -- \
  "$HOME/.nuclis/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf"
```
