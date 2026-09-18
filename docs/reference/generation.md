# Native CPU generation bring-up

The complete 64-layer Qwen text schedule runs natively in Zig: mapped quantized
weights, embeddings, 48 convolution/DeltaNet layers, 16 gated attention layers,
residuals, feed-forward blocks, final normalization, and vocabulary logits.
There is no llama.cpp dependency in nuclis. The external pinned reference is
only used for comparisons.

This is the slow correctness backend. `--backend metal` runs the same schedule
GPU-resident through the [Metal backend](metal-backend.md). Efficient batched prefill, scoring, and production
performance acceptance remain pending. Default tests
need neither weights nor GPU access.

## Run

```sh
zig build -Doptimize=ReleaseSafe --global-cache-dir .zig-cache/global
./zig-out/bin/nuclis generate --prompt 'Hi' --max-tokens 3 --ctx-size 64
./zig-out/bin/nuclis generate --raw --prompt 'Hello,' --max-tokens 2 --ctx-size 8 --json
```

The default model is the catalogue entry `qwen3.8-27b` under `~/.nuclis/models/unsloth/Qwen3.8-27B-GGUF/`; `--model` and
`NUCLIS_HOME` work as for inspection. Keep the file unchanged while mapped.
The backend, context, output budget, reasoning effort, and sampling
overrides come from `~/.nuclis/nuclis.json` when it exists
([development.md § Configuration file](../development.md#configuration-file));
flags override the file per option.
The default prompt profile renders one user turn with thinking disabled;
`--think off|low|medium|xhigh` selects the reasoning effort the template
encodes. `--raw` supplies literal prompt text, recognizing special-token markers without
inserting BOS or EOS. `--prompt` is limited to 64 KiB and `--prompt-file` to
4 MiB of valid UTF-8; the two are exclusive and neither is ever truncated.

Defaults (without a configuration file) are the Metal backend when built
in, 2,048 output tokens, 8,192 total context, and seed zero (before APPS-03:
CPU, 16 tokens, 2,048 context). Sampling
defaults come from the official Qwen3.8 profile of the reasoning mode (next
section); the file's `generate.sampling` entries and then `--temperature`,
`--top-k`, `--top-p`, `--min-p`, `--presence-penalty`, and
`--repetition-penalty` each override one option, and `--temperature 0`
selects greedy. Greedy ties select the lowest token
ID. Nonfinite logits are errors.

### Sampling profiles and the selection chain (MODL-01)

`generate` and `chat` sample with the checkpoint's official settings of the
reasoning mode `--think` selects. The tables live in the profile modules
(`samplingDefaults`) because they are part of the checkpoint's recommended
usage, not of the shared sampler, whose own defaults are neutral (greedy).
The profile is the file's, selected by its template digest at load; the
configuration's `config show` names the catalogue entry's profile (or the
first profile for a bare path) without opening the file, and a run corrects
the sampler to the file's profile once the model is open. Qwen3.8, from the
[Unsloth Qwen3.8 guide](https://unsloth.ai/docs/models/qwen3.8.md) (read
2026-09-08):

| Option | Thinking (`low`, `medium`, `xhigh`) | Instruct (`off`) |
| --- | ---: | ---: |
| `temperature` | 1.0 | 0.7 |
| `top_p` | 0.95 | 0.80 |
| `top_k` | 20 | 20 |
| `min_p` | 0.0 | 0.0 |
| `presence_penalty` | 0.0 | 1.5 |
| `repetition_penalty` | 1.0 | 1.0 |

Gemma 4 (`inference/src/profiles/gemma4.zig`, MODL-07) uses the file's own
sampling hint in every mode: temperature 1.0, `top_p` 0.95, `top_k` 64, no
`min_p`, no penalties (`general.sampling.*` in the header; no per-mode
table was pinned from a model card).

Flags override per option, so `--temperature 0` under `--think off` is a
greedy argmax over logits that still carry the instruct presence penalty;
add `--presence-penalty 0` for the plain argmax (the GPU argmax path). The
chat rebuilds the options from the current effort at every turn (Ctrl-T
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
through batched matrix kernels (ENGN-02; see
[metal-backend.md § Prefill in chunks](metal-backend.md#prefill-in-chunks-engn-02));
`--trace-dir` requests per-layer activations per token and therefore steps
the prompt token by token, as the CPU reference always does. Chunked and
stepped prompts agree within the tolerance recorded there, not bit for bit.
`prefill_milliseconds` keeps its definition (all prompt tokens, logits for
the last).

### Sampling on the GPU without reading the vocabulary back (KERN-06)

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
- **Eligibility.** No penalty may be active: the readback is taken after
  the projection, before the CPU could apply the history, so
  `presence_penalty ≠ 0` or `repetition_penalty ≠ 1` (the instruct profile)
  runs the full path every token — the documented cost of that profile
  until a GPU penalty kernel exists (measured in [bench.md](bench.md)).
  Then `1 ≤ top_k ≤ 256`: the retained set is inside the readback and its
  denominator is computed on the CPU; always exact, never falls back, with
  or without `min_p`. `top_k = 0, min_p > 0`: the survivors are a prefix of
  the readback and their sum is the exact CPU sum, so the decision is
  exact; only when all 256 candidates survive the filter does the sampler
  **defer** (the prefix may continue past the readback). `top_k = 0, top_p
  < 1, min_p = 0`: the nucleus ranges over the vocabulary, so the GPU
  denominator is used; its relative error is bounded (measured 9.6e-8 on a
  flat vector by `test-metal`, asserted ≤ 2e-6) and the sampler only
  accepts a threshold comparison whose margin exceeds `total_band = 1e-5`;
  a comparison inside the band, or a nucleus that runs past the 256
  candidates, **falls back**. Not eligible (the full path runs every token):
  any penalty, `top_k > 256`, and `top_k = 0` with `top_p = 1, min_p = 0`.
- **Fallback** reads the full logits from the GPU's shared buffer
  (`Plan.readLogits`, valid until the next step) and runs `select`; the RNG
  has not advanced, so the token equals the reference path's. A non-finite
  logit anywhere raises a per-partition flag and also falls back, where the
  reference path reports `NonFiniteResult` as before. `generate --json`
  reports `gpu_topk_fallbacks`; `bench` shows a per-run fallback column.
- The `--logits` option needs the full vector and therefore keeps the
  reference path; comparing a `--logits` run with a plain run at the same
  seed is the end-to-end equivalence check (recorded in [bench.md](bench.md)).

Greedy decoding with a penalty active takes the CPU argmax over the
penalized logits (full readback) instead of the GPU argmax.

The output budget is 1–4,096 tokens and context is 1–32,768, with
prompt plus output budget required to fit. These are allocation/execution bounds,
not a validated 32K performance claim. Both `<|im_end|>` and `<|endoftext|>` stop
the pinned profile. Special tokens are omitted from presented text.

Text is flushed after each token. The UTF-8 stream buffers incomplete characters
across tokens; malformed bytes and an incomplete character at termination become
U+FFFD. JSON (schema 2) buffers the result and includes backend, sampling
settings, seed, prompt count, exact generated IDs, text, `stop_reason`
(`eos`, `token_budget`, `context_limit`, or `cancelled`), and load/prefill/first-token/decode
milliseconds with GPU busy time on Metal; definitions are in [bench.md](bench.md).
Token IDs preserve information lost by text presentation.

Cancellation is checked at each layer boundary through the observer's `check`
callback (injected Zig Io and a Ctrl-C flag, with a one-hour execution bound);
it needs no activations, so on the GPU plan it runs while the token is being
recorded and costs no synchronization. The `layer` callback, which receives
activations and forces a GPU commit per layer, is installed only with
`--trace-dir`. The third callback, `progress` (TERM-01), reports how far a turn
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

The reference helper requires the pinned checkout from
[reference-baseline.md](reference-baseline.md). It uses public APIs and selects
GPU offload, F32 KV, disabled flash attention, and one token per decode call.
It refuses to run without a GPU. Build against the relocated checkout:

```sh
mkdir -p .zig-cache/generation
c++ -std=c++17 \
  -I.zig-cache/reference/llama.cpp/include \
  -I.zig-cache/reference/llama.cpp/ggml/include \
  scripts/reference-generation.cpp \
  -L.zig-cache/reference/llama.cpp/build/bin -lllama -lggml -lggml-base \
  -Wl,-rpath,"$PWD/.zig-cache/reference/llama.cpp/build/bin" \
  -o .zig-cache/generation/reference-generation
mkdir -p .zig-cache/generation/native-check .zig-cache/generation/reference-check
./zig-out/bin/nuclis generate --raw --prompt 'Hello,' --max-tokens 1 --ctx-size 8 \
  --logits .zig-cache/generation/native-check/logits.f32 \
  --trace-dir .zig-cache/generation/native-check
.zig-cache/generation/reference-generation \
  "$HOME/.nuclis/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf" 'Hello,' \
  .zig-cache/generation/reference-check
python3 scripts/compare-generation.py \
  .zig-cache/generation/native-check .zig-cache/generation/reference-check --positions 2
```

The comparison checks every requested layer and final logits, rejecting missing,
wrong-sized, or nonfinite files. Initial bring-up tolerances are maximum absolute
error 0.002 and relative RMS error 0.0001 for every tensor. These are local
full-model smoke thresholds, not general per-kernel or quality acceptance limits.
They are the accepted numbers for the F32 cache (`make compare-f32`); the
F16 cache (KERN-07) has its own documented tolerance, 0.03 / 0.0002 on the layer
files with the logits inside the bring-up numbers (`make compare-f16`;
[metal-backend.md § F16 KV cache](metal-backend.md#f16-kv-cache-kern-07)).

The accepted outputs of this procedure are committed as fixtures under
[tests/fixtures/](../../tests/fixtures/) (`reference-hello-comma/` and
`reference-chat/`, with provenance in
[tests/fixtures/provenance.md](../../tests/fixtures/provenance.md)); `make
compare` runs the same comparison against `tests/fixtures/reference-hello-comma`
without regenerating anything. The `.zig-cache/generation/` paths above are the
working copies for reproducing or extending the traces; the compiled
`reference-generation` harness itself is a cache artifact, never committed.
Since MODL-05 the harness accepts any F32 layer width and the comparison
script takes the geometry as flags (`--embedding`, `--layers`, `--vocab`;
the defaults are Qwen3.8's), so the same pair serves Gemma 4
(`make compare-gemma4` on the K-quant entry, `compare-gemma4-qat`
on the bring-up file, each against its own traces;
[gemma4.md](gemma4.md#cpu-reference-against-the-oracle-modl-05-2026-09-11)).
Top-five IDs and reference greedy margin are reported for diagnosis. The
harness also builds unchanged against the PrismML fork (the include and
library paths of `.zig-cache/reference/prism-llama.cpp`, output
`.zig-cache/generation/prism-reference-generation`), which is how the Bonsai
2 27B traces were captured ([bonsai.md](bonsai.md#oracle-the-prismml-fork)).

The explicit session test runs the real two-token sequence in two independent
sessions and after reset following injected cancellation in layer 3. Logits must
be bit-identical within the same CPU backend:

```sh
zig build test-generation -Doptimize=ReleaseSafe --global-cache-dir .zig-cache/global -- \
  "$HOME/.nuclis/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf"
```

## Evidence

On 2026-09-07, Zig 0.16.0 ReleaseSafe on the target Apple M4 Pro, the pinned
artifact and llama.cpp revision `7620399f58aebfd2196b74021f9581bcf7218cb9`:

- Raw `Hello`: all 64 layers compared; logits maximum absolute error `4.77e-6`,
  RMS `9.80e-7`; identical top five, greedy token 11 (`,`).
- Raw `Hello,`: all 128 layer outputs compared across two positions; identical
  top five logits and greedy token 353 (` I`). Native two-token generation
  continued with IDs `[353, 2688]`, text ` I'm`.

- Chat `Hi` with the thinking-off template: all 832 layer outputs across 13
  prompt positions pass the same comparison thresholds. Native output begins
  `Hello! How` (IDs `[9419, 0, 2500]`).
- Full-model session isolation and reset after injected layer-3 cancellation pass
  with bit-identical two-token CPU logits. All 95 default tests pass in Debug and
  ReleaseSafe, including partial-allocation cleanup, sampling, and UTF-8 splits.

The first single-token native run took 39 whole seconds excluding model loading.
That is a smoke-run observation, not a warm throughput benchmark. Full-model
checks run explicitly and are distinct from default allocator/unit tests.
