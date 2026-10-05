# Laya

Laya (`convaiinnovations/laya`, Apache-2.0) is a decision model: a
ModernBERT-large encoder with a typed-decision head. It answers a typed
question (`choice`, `score`, or `noul`, a yes/no probability) about a state
(text, or a JSON document) in one forward pass, with no decoding. nuclis
runs the English root checkpoint (commit `55cf4c4e`, pulled under
`~/.nuclis/models/convaiinnovations/laya/`) and, with the same code, the
multilingual one beside it (an mmBERT-base encoder,
[below](#the-multilingual-checkpoint)), in these parts:

| Part | Files | Status |
| --- | --- | --- |
| Tokenizer | `inference/src/tokenizer/hf_json.zig`, `gpt2.zig`, `nfc.zig`, `bpe.zig` | landed ([tokenizer.md § Hugging Face `tokenizer.json`](../engine/tokenizer.md#hugging-face-tokenizerjson)) |
| Encoder and head, CPU | `inference/src/models/modernbert.zig`, `modernbert_runtime.zig`, `laya.zig`, `inference/src/backends/cpu/dense.zig` | landed (below) |
| Encoder and head, Metal | `inference/src/models/laya_metal.zig`, `inference/src/backends/metal/` (`attentionSegments`, `geluErfMulRows`) | landed (below) |
| Input contract, calibration, `Decider` | `inference/src/profiles/laya.zig`, `inference/src/decide.zig` | landed (below) |
| `nuclis decide`, the `laya` and `laya-multilingual` catalogue entries, `decide.model` | `src/decide.zig`, `src/catalog.zig`, `src/config.zig` | landed (below) |

**Why it is worth having beside a language model.** Every question
re-reads its state (the encoder is bidirectional, so nothing is shared
between questions), and a language model must spend decode tokens writing
each question, so delegating one decision the language model could make
itself saves nothing. It pays as a **filter**: one question fanned out over
many states the language model never reads (30 search hits, every log
section, each diff hunk), where the language model's prefill is the cost
avoided: nuclis prefills Qwen3.8-27B at 90.45 tok/s at 512 tokens on
Metal ([benchmarks § Acceptance runs](../benchmarks/README.md#acceptance-runs)), so each
512-token state it skips saves about 5.7 s of prefill and its context,
against about 0.2 s for Laya to judge it on Metal, or 2.9 s on the CPU
while the GPU stays free (below).

## The encoder

`encoder/config.json` (`model_type` `modernbert`) gives 28 layers, hidden
1024, 16 heads of 64, intermediate 2624, vocabulary 50368, no biases,
LayerNorm eps 1e-5. Layer `i` attends globally when `i % 3 == 0` (RoPE
theta 160000), else within a window (theta 10000): query `i` sees keys
`|i − j| ≤ 64`, half of `local_attention` 128. `modernbert.parseConfig`
also reads the Transformers 4.x spellings (`global_attn_every_n_layers`,
`global_rope_theta`, `local_rope_theta`) and rejects biases, an activation
other than exact GELU, head widths not a multiple of 8, and other layer
types.

Tensors, PyTorch `[out, in]`, under `encoder.`:
`embeddings.tok_embeddings.weight [50368,1024]`, `embeddings.norm.weight`,
per layer `attn_norm.weight` (absent on layer 0), `attn.Wqkv.weight
[3072,1024]`, `attn.Wo.weight`, `mlp_norm.weight`, `mlp.Wi.weight
[5248,1024]`, `mlp.Wo.weight [1024,2624]`, then `final_norm.weight`. The
root set stores them in F16.

The forward, as Transformers 5.17's `modeling_modernbert.py` (the oracle's)
computes it:

```
x = LayerNorm(embed[ids])
per layer:  x += Wo(attention(rope(Wqkv(attn_norm(x)))))     # attn_norm: identity on layer 0
            a, g = halves of Wi(mlp_norm(x));  x += mlp.Wo(gelu(a) * g)
x = LayerNorm_final(x)
```

`Wqkv` rows are q, k, v, each `[16][64]`. RoPE is split-half (pairs `d` and
`d + 32`) with inverse frequencies `theta^(−2i/64)`; the reference forms
each angle as an F32 product, so `modernbert_runtime.zig` rounds the angle
to F32 before its cosine and sine. Attention is bidirectional, scale 1/8.
GELU is the exact erf form. LayerNorm has a weight and no bias.

## The head

In `model.safetensors`, unprefixed: `type_emb.weight [3,1024]` is added to
every row (choice 0, score 1, noul 2); then two PyTorch
`TransformerEncoderLayer`s (`d=1024`, 16 heads, feed-forward 4096,
`norm_first`, ReLU, LayerNorm eps 1e-5 with bias, every linear with bias):
`x += out_proj(mha(norm1(x)))`, `x += linear2(relu(linear1(norm2(x))))`,
`in_proj_weight [3072,1024]` stacking q, k, v. At each option's `[MASK]`
row the scorer gives the option's logit: `LayerNorm(scorer.0) →
Linear(scorer.1) → GELU → Linear(scorer.3, → 1)`. `act_head.*` and the
`temperature` buffer are known and unused; `Laya.open` rejects any other
tensor, and any missing or misshapen one.

## On the CPU

`Laya.open(gpa, io, dir)` parses the config, maps the checkpoint, binds
every tensor, and decodes all but the embedding table to F32 (about 1.5 GB
for the root set; embedding rows decode as ids need them, so the mapping
stays open). `Laya.logits(io, gpa, ids, markers, kind, out, trace)` scores
one built sequence; `trace` sees the residual stream after each encoder
layer, after `final_norm`, and after each head layer.

The kernels are `backends/cpu/dense.zig`: `Y = X·Wᵀ + b` in 4×4 output
tiles of eight-lane F32 vectors, output columns split across `Io` tasks;
LayerNorm with F64 statistics; bidirectional attention, full or windowed,
heads split across tasks. Each output is summed by one task in a fixed
order, so results do not depend on the core count.

### Checked against the oracle

`scripts/laya-reference.py` runs the `laya` 0.3.20 package (PyTorch 2.14.0,
Transformers 5.17.0) in F32 on the CPU and writes
`inference/src/models/fixtures/laya/`: for 8 requests the ids, markers, raw
logits, and answer, and in `activations.f32` rows of the residual stream
(row 0, up to three markers, the last row) after encoder layers 0, 1, 3,
and 27, after `final_norm`, and after both head layers (marker rows only).

```sh
zig build test-laya -Doptimize=ReleaseFast -- ~/.nuclis/models/convaiinnovations/laya
```

(gate `laya-cpu`, tier verify). The bounds are relative, because F32
summation-order differences scale with the values summed and the
pre-norm residual carries outlier dimensions up to 1.65·10⁴: per tensor,
the largest difference over the largest reference value ≤ 1e-5 and the
difference's RMS over the reference's ≤ 1e-5; per logit, the difference
over max(1, |logit|) ≤ 1e-4. (The first plan's absolute 1e-4 held through
layer 3 and after `final_norm`, not at layer 27 or in the head, whose
values reach 10²–10⁴.)

Measured 2026-09-29, Apple M4 Pro (12 cores), Zig 0.16.0, ReleaseFast,
the MODL-30 session 2 commit; worst stage per request:

| Request | Tokens | Options | ms | scaled max \|Δ\| (stage) | rel RMS | scaled logit \|Δ\| |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `choice_described` | 59 | 3 | 279 | 3.1e-6 (encoder.27) | 2.3e-6 | 2.1e-6 |
| `choice_labels` | 36 | 3 | 169 | 2.6e-6 (final) | 1.7e-6 | 5.2e-6 |
| `score` | 49 | 3 | 230 | 2.4e-6 (final) | 2.5e-6 | 6.0e-6 |
| `noul` | 56 | 2 | 256 | 6.6e-6 (head.0) | 4.8e-6 | 4.0e-6 |
| `json_state` | 119 | 2 | 552 | 4.0e-6 (final) | 5.4e-6 | 4.8e-6 |
| `long_text` | 512 | 3 | 2410 | 1.7e-6 (head.0) | 2.8e-6 | 7.9e-6 |
| `long_list` | 512 | 2 | 2408 | 8.3e-6 (head.0) | 7.0e-6 | 9.0e-6 |
| `choice_20` | 193 | 20 | 906 | 2.9e-6 (encoder.27) | 2.1e-6 | 7.5e-6 |

The times are the forward alone (encoder and head, the trace's comparison
included); opening took 212 ms with the checkpoint in the page cache. A
512-token sequence is about 2·10¹¹ multiply-adds, so 2.4 s is roughly 170
GFLOP/s across the cores.

## On Metal

`models/laya_metal.zig` `Plan` runs the CPU forward's schedule on the GPU;
`Laya.open(gpa, io, dir, .metal)` builds it instead of the CPU runtime.
It owns its `Backend`, so every device buffer goes with `destroy`.

- **Weights.** Each matrix is the checkpoint mapping's F16 (or F32) bytes
  wrapped in place and read by the generic F32 `matmul` tile, which
  converts each weight and keeps F32 operands and sums; BF16 matrices are
  rejected. Vectors (norms, biases, `type_emb`) are decoded to F32
  buffers; `scorer.3` stays on the host. Opening decodes no matrix, so it
  is the shader pipelines and the mapping: 53 ms (ReleaseFast) against
  the CPU's 212 ms.
- **Packed batches.** Sequences sit back to back, at most `max_rows`
  (2048) rows, with no padding: the matmuls, norms, and elementwise
  kernels run over every row at once; a bounds buffer gives each row its
  sequence's `[begin, end)`, and a per-row RoPE table its position within
  the sequence, copied from the CPU forward's own F32-angle tables.
  Activations take about 56 KB a row (115 MB for 2048 rows).
  `Laya.logitsBatch` cuts a list of sequences into such batches;
  `Decider.decide` hands it every question about every state at once.
- **Kernels.** `attentionSegments` (`nu_attention_segments`) is
  `attention_full`'s body over a key range: row i sees the keys of its
  sequence within `|i − j| ≤ 64` on sliding layers, all of them on
  global layers and in the head; one 256-thread group per (row, head),
  bounds clamped on the device. `geluErfMulRows` is the GeGLU,
  `gelu_erf(a)·g` over the two halves of `Wi`'s output rows. The rest
  existed: `layerNorm` (with a zero bias for the encoder's bias-free
  norms), `ropeRows` split-half over the q and k thirds of the fused row,
  `addBiasRows` (every head bias, and `type_emb` per sequence), `clamp(0,
  ∞)` as ReLU, `geluErf`, `add`.
- **The scorer** runs its LayerNorm, Linear, and GELU on every row (the
  marker rows are a few of them; about 0.3 % of the forward) in the same
  command buffer; after the one commit the host takes each marker row's
  dot product with `scorer.3` in F64.

### Checked against the oracle and the CPU

```sh
zig build test-laya -Doptimize=ReleaseSafe -- ~/.nuclis/models/convaiinnovations/laya --backend metal
```

(gate `laya-metal`, tier verify). The same fixtures and the same bounds
as the CPU forward; with a trace the plan commits after every stage and
the host reads the residual rows. Measured 2026-09-29, Apple M4 Pro,
Zig 0.16.0, ReleaseFast, the MODL-31 commit; worst stage per request, the
times with a commit per traced stage (31 command buffers; the first
includes pipeline warm-up):

| Request | Tokens | Options | ms | scaled max \|Δ\| (stage) | rel RMS | scaled logit \|Δ\| |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `choice_described` | 59 | 3 | 383 | 1.5e-6 (encoder.27) | 1.4e-6 | 1.1e-6 |
| `choice_labels` | 36 | 3 | 38 | 4.0e-6 (encoder.27) | 1.9e-6 | 8.3e-7 |
| `score` | 49 | 3 | 25 | 2.9e-6 (final) | 2.5e-6 | 2.5e-6 |
| `noul` | 56 | 2 | 27 | 2.9e-6 (head.0) | 2.4e-6 | 1.7e-6 |
| `json_state` | 119 | 2 | 48 | 7.6e-6 (final) | 8.9e-6 | 2.3e-6 |
| `long_text` | 512 | 3 | 210 | 1.4e-6 (head.0) | 2.1e-6 | 6.5e-6 |
| `long_list` | 512 | 2 | 209 | 6.0e-7 (final) | 1.1e-6 | 1.2e-6 |
| `choice_20` | 193 | 20 | 76 | 4.7e-6 (head.0) | 3.7e-6 | 6.5e-6 |

The check then runs all 8 requests as one packed batch (1,536 rows, 538
ms): each logit equals the request's run alone to the bit, since no
row's arithmetic depends on what is packed beside it. On the tiny
synthetic checkpoint it opens, runs, and closes the plan three times, and
fails every allocation of `open` in turn with no leak. The two kernels
have their own model-free checks in `make test-metal`: attention over
packed sequences of 1, 37, and 90 rows, full and within windows 0, 3,
and 64, with NaN past the batch, against F64 (worst 7.2e-7); the GeGLU
over strided fused rows against `cpu.geluErf`.

## The input contract

As the `laya` 0.3.20 package defines it (`laya/common.py`
`build_sequence`, `render_options`; `laya/agent.py` `_check_question`,
`_to_internal`, `_encode_state`), implemented in `profiles/laya.zig`:

- **Questions.** `{"type", "instructions", "criteria", "labels"}`. Type
  `choice` takes criteria as an object of key → description or a list of
  keys; `score` a non-empty list of levels, lowest first; `noul` an
  optional object keyed only `true`/`false` (case-insensitive), with
  optional `labels` (two distinct non-empty strings, stripped) that word
  the options without changing which is true. Non-string instructions are
  JSON. Every failure names the question.
- **Options.** choice `"key: description"`, or `"key"` when the
  description is null or `""`; score `"level i: text"`; noul
  `"<false label>: <text or 'no, the statement does not hold'>"` then
  `"<true label>: <text or 'yes, the statement holds'>"`. A non-string
  description is compact JSON.
- **JSON rendering.** Object and list states, non-string instructions,
  and structured descriptions are Python's `json.dumps(…, ensure_ascii=False)`:
  separators `", "` and `": "`, non-ASCII kept, floats in `repr` form
  (`1.0`, `1e-05`, `1e+16`), keys in their order.
- **The sequence.** `[CLS] tok("<type> question: <instructions>") [SEP]`,
  then per option `[MASK] tok(" " + text)` (first 48 tokens), `[SEP]`,
  the state's tokens, `[SEP]`; the mask token spelled in any text becomes
  a space first. The three are the tokens `tokenizer_config.json` names
  (`cls_token`, `sep_token`, `mask_token`; `[CLS]`, `[SEP]`, `[MASK]`
  when absent): the multilingual set's are `<bos>`, `<eos>`, `<mask>`, so
  there `[MASK]` in a text stays text. Budgets, from
  `rl_agent_config.json` (`max_len` and `head_max_len`, 512 and 192 when
  absent; the root set's are those, the multilingual set's 1,024 and 256),
  in order: if the options leave under 16 of `head_max_len`, each is cut
  to `max(4, (head_max_len − 16) / count)`; the question text keeps
  `max(8, what remains)`; the state gets the rest of `max_len` less the
  final `[SEP]`, cut at its tail, or at its head for a list state (a
  conversation, newest last). Options that still do not fit are
  `OptionsExceedBudget`.

Checked by the Laya mode of `zig build test-vocabulary` (gates
`laya-vocabulary`, `laya-multilingual-vocabulary`): each oracle request (8
of the root set, 16 of the multilingual one), rebuilt from its raw
question and state, gives exactly the package's ids and markers.

## Calibration and answers

`p = softmax(logits / T)` in F32, as NumPy computes it on the F32 logits.
`T` comes from `rl_agent_config.json`: `temperature_by_options` by bucket
`"<type>:<2|3-5|6-10|11+>"` (k options), else `temperature[type]`,
**clamped to [0.5, 5.0]** (a non-number applies as 1.0): the shipped
`choice:11+` is 0.1006 and runs as 0.5. `--uncalibrated` uses T = 1.

| Type | Value | `confidence` | `answer_confidence` |
| --- | --- | --- | --- |
| choice | the argmax key | `1 − H(p) / ln k`, clipped (1 when k < 2) | `max(p)` |
| score | `Σ i·p[i]` (zero-based) | the same | `max(p)` |
| noul | `p[1]` = P(true) | `max(p1, 1 − p1)` in the package | `max(p)` |

`answer_confidence` is the one temperature scaling fits (the package's
own note), so the terminal view shows it. Numbers are rounded to 4 places
as Python's `round` does. After rounding, the recorded logits of the 8
oracle requests calibrate to exactly the package's answers.

## `nuclis decide`

Three input tiers build one request: `--request <file|->` (Jev's shape,
`{"questions": {id: definition}, "state": …}`, plus `"states": [...]` and
`{"file": path}` states; a request's `"model"` is ignored),
`--questions <file>` with `--state`/`--state-file` (repeatable), or
questions inline (`--choice <text> --option key[=description]…`,
`--score <text> --level <text>…`, `--noul <text>`, each named by `--id`,
default its type). One state renders every answer with its distribution;
several render ranked by the first question (P(true), the expected score,
or the first option's probability), a cut state flagged. `--explain`
shows each sequence decoded, its budget split, bucket, and logits;
`--truncate head|tail` chooses the end of a long state that is cut;
`--backend cpu|metal` where the model runs (default metal in a Metal
build).

`--json`: `{schema_version, model, repo, revision, timings_ms {load,
tokenize, encode}, results: [...]}`, each result a complete Jev response,
`{answers, usage {input_tokens, output_tokens: 0}}`, plus `nuclis
{state, state_tokens, truncated}`. Each answer's top level is exactly the
fields TypeSafe documents for Jev (choice `type, choice, probabilities,
confidence`; score `type, score, confidence, probabilities, legend`; noul
`type, noul`, no confidence), and everything else sits under its `nuclis`
object: `answer_confidence`, `logits`, `temperature`, `bucket`.

The checkpoint is `--model` or `decide.model` (default `laya`): a registry
entry of kind `decision` (written by `model pull --register` for a Laya
layout), the decision catalogue's `laya` or `laya-multilingual`
([catalogue.md § The catalogue](catalogue.md#the-catalogue)), or a
directory. Text commands refuse a decision model by name, and `decide`
refuses a text model.

`nuclis serve` answers the same request over HTTP with the models kept
open: `POST /v1/decisions` returns these bytes, timings aside, and `POST
/v1/systemone` is Jev's own call ([api.md](../guide/api.md)); a served request's
`"model"` chooses the checkpoint.

Checked through the fresh binary, 2026-09-29: the 8 fixture requests
(`--json`) give exactly the package's answers and usage; the card's
quickstart gives identical JSON through `--request` (file and standard
input), `--questions`, and inline flags; one question fanned out over
four log files ranks them, a long one flagged as cut; malformed requests
fail with typed errors naming the question.

### Time per call

Apple M4 Pro, Zig 0.16.0, commit `792c47f`, best of three, the checkpoint
in the page cache; `load` is opening the directory (tokenizer, config,
weights decoded to F32), separate from the call:

| Call | Tokens | Load (ReleaseSafe / Fast) | Tokenize | Encode (ReleaseSafe / Fast) |
| --- | ---: | --- | --- | --- |
| one noul, the quickstart ticket | 56 | 647 / 197 ms | 0.1 ms | 269 / 259 ms |
| the quickstart's three questions | 164 | 613 / 194 ms | 0.2 ms | 801 / 772 ms |
| one question, a 512-token state | 512 | 596 / 193 ms | 0.9 ms | 2582 / 2458 ms |

Encoding is the cost; every question is its own sequence, so a call's time
is about the sum of its sequences' (a short sequence runs at a lower rate:
the matrices are too narrow to fill the cores).

**Metal against the CPU**, as a filter runs: one noul question over 1,
10, and 50 synthetic log states of about 60 and about 500 tokens
(`nuclis decide --backend … --json --request`, `timings_ms`). Apple M4
Pro, Zig 0.16.0, ReleaseSafe, the MODL-31 commit, best of three, the
pipelines compiled and the checkpoint in the page cache:

| States × state tokens | Input tokens | Encode, Metal | Encode, CPU | Metal, tokens/s |
| --- | ---: | ---: | ---: | ---: |
| 1 × 57 | 95 | 39 ms | 519 ms | 2,440 |
| 10 × 54–60 | 947 | 278 ms | 5,047 ms | 3,410 |
| 50 × 54–60 | 4,704 | 1,344 ms | 26,485 ms | 3,500 |
| 1 × 492 | 512 | 207 ms | 2,866 ms | 2,470 |
| 10 × 487–500 | 5,120 | 1,970 ms | 30,020 ms | 2,600 |
| 50 × 484–504 | 25,600 | 9,801 ms | — | 2,610 |

Load: 69–75 ms on Metal (the tokenizer, the config, the pipelines, the
wrapped mapping), 650–690 ms on the CPU. Metal is 13–20× the CPU.

Where the GPU time goes (`laya-check … --backend metal --profile`, each
dispatch timestamped, ReleaseSafe): for the 512-token `long_text`
request, 197 ms in kernels, attention 45 % (88 ms), the matmuls 52 %
(about 3.5 TFLOP/s), everything else 3 %; for all 8 requests packed
(1,536 rows, 539 ms), attention 39 %. The attention kernel runs one
threadgroup per (row, head) that reads every visible key itself, so its
cost grows with the sequence, which is why short states run faster per
token.

## The multilingual checkpoint

`multilingual/` in the same repository at the same commit (`nuclis model
pull laya-multilingual`, `nuclis decide --model laya-multilingual`) is
the same head on an mmBERT-base encoder: 22 layers of hidden 768, 12
heads of 64, intermediate 1,152, a 256,000-token vocabulary, global
attention every third layer from 0 and windows of 128 (64 each side)
between, both RoPE thetas 160,000 (`rope_parameters`, which
`modernbert.parseConfig` reads); `model.safetensors` is 643,835,514 B, 169
F16 tensors and the F32 `temperature`, with the root set's tensor names.
`rl_agent_config.json` gives `max_len` 1,024 and `head_max_len` 256 and
temperatures of 1 (`max_prefixes` 6 is read by nothing in the package).
Nothing in the model code changed: the forward, the Metal plan, and the
2048-row batch already covered the shape. What differs is outside it:

- **Tokenizer.** Gemma's vocabulary as a Metaspace `tokenizer.json`
  (byte fallback, 580,604 merges, 249 added tokens), the second shape of
  `hf_json.zig` ([tokenizer.md § Hugging Face `tokenizer.json`](../engine/tokenizer.md#hugging-face-tokenizerjson)).
  Numbers split to single digits, so the same English log line is about
  40 % more tokens than under the root set.
- **Special tokens.** `tokenizer_config.json` names `<bos>` (2) as
  `cls_token`, `<eos>` (1) as `sep_token`, and `<mask>` (4); the
  package's `build_sequence` uses those, not the encoder config's
  `cls_token_id` (1). `<mask>` in a text becomes a space, `[MASK]` stays.
- **Budget.** 1,024 tokens per sequence, 256 for the question and options.

Checked by the same tools on the oracle's 16 requests
(`scripts/laya-reference.py --subfolder multilingual`,
`fixtures/laya-multilingual/`): the root set's 8 shapes, where
`long_text` and `long_list` now fill 1,024 tokens, then states in
French, German, Spanish, Arabic, Chinese, and Hindi (questions in the
state's language or in English), a language question, and an 889-token
French log whose one error line sits past position 512. The tensors kept
are encoder layers 0, 1, 3, and 21, `final`, and both head layers; the
bounds are the root set's. Gates `laya-multilingual-vocabulary`,
`laya-multilingual-cpu`, `laya-multilingual-metal`, tier verify. Measured
2026-09-29, Apple M4 Pro, Zig 0.16.0, ReleaseFast, the MODL-33 commit;
worst stage per request, CPU then Metal:

| Request | Tokens | Options | CPU ms | CPU scaled max \|Δ\| (stage) | CPU scaled logit \|Δ\| | Metal ms | Metal scaled max \|Δ\| (stage) | Metal scaled logit \|Δ\| |
| --- | ---: | ---: | ---: | --- | --- | ---: | --- | --- |
| `choice_described` | 56 | 3 | 99 | 1.5e-6 (head.0) | 4.7e-6 | 35 | 1.2e-6 (head.0) | 2.3e-6 |
| `choice_labels` | 36 | 3 | 65 | 1.9e-6 (head.1) | 1.1e-6 | 10 | 1.6e-6 (head.1) | 1.6e-6 |
| `score` | 52 | 3 | 90 | 1.2e-6 (head.0) | 2.5e-6 | 11 | 2.9e-7 (head.0) | 1.8e-7 |
| `noul` | 56 | 2 | 97 | 1.0e-6 (head.0) | 7.7e-7 | 11 | 6.8e-7 (final) | 7.7e-7 |
| `json_state` | 124 | 2 | 212 | 8.9e-7 (head.0) | 1.7e-6 | 21 | 5.0e-7 (final) | 4.3e-7 |
| `long_text` | 1024 | 3 | 1957 | 2.5e-6 (head.0) | 9.2e-6 | 287 | 8.7e-7 (head.0) | 2.1e-6 |
| `long_list` | 1024 | 2 | 1947 | 7.6e-6 (head.1) | 9.2e-6 | 285 | 1.2e-6 (head.0) | 1.5e-6 |
| `choice_20` | 273 | 20 | 473 | 1.4e-6 (head.0) | 5.0e-5 | 47 | 6.4e-7 (encoder.21) | 2.7e-6 |
| `french_choice` | 77 | 3 | 134 | 1.7e-6 (head.0) | 1.4e-6 | 16 | 5.8e-7 (head.0) | 4.4e-7 |
| `german_noul` | 66 | 2 | 116 | 1.8e-6 (encoder.21) | 4.1e-6 | 15 | 5.8e-7 (final) | 2.6e-6 |
| `spanish_score` | 53 | 3 | 92 | 1.5e-6 (head.0) | 4.3e-6 | 11 | 5.2e-7 (head.0) | 2.4e-6 |
| `arabic_choice` | 44 | 3 | 78 | 2.4e-6 (head.0) | 2.1e-5 | 11 | 1.3e-6 (head.0) | 1.1e-5 |
| `chinese_choice` | 56 | 3 | 97 | 1.1e-6 (head.0) | 9.9e-7 | 11 | 4.9e-7 (encoder.21) | 3.2e-7 |
| `hindi_noul` | 58 | 2 | 102 | 8.2e-7 (head.0) | 1.2e-7 | 11 | 4.6e-7 (encoder.21) | 1.2e-7 |
| `language_id` | 36 | 7 | 65 | 1.4e-6 (head.0) | 5.2e-6 | 10 | 9.7e-7 (head.0) | 1.7e-5 |
| `long_state` | 889 | 3 | 1686 | 1.4e-6 (head.0) | 6.6e-7 | 234 | 6.0e-7 (head.0) | 1.1e-7 |

Worst relative RMS: 6.5e-6 on the CPU, 1.8e-6 on Metal. All 16 as one
packed Metal batch (3,924 rows, over two batches, 908 ms) give each
logit bit-identical to its run alone. Opening the weights alone took 63
ms on the CPU and 28 ms on Metal.

**Time per call**, measured the way the root set's was (one noul over
synthetic log states, `nuclis decide --json`, `timings_ms`, ReleaseSafe,
best of three, the MODL-33 commit, Apple M4 Pro), with states sized to
this tokenizer. The root set on the same states, Metal, is on the right.
The CPU skipped the 50-state rows of long states:

| States × state tokens | Input tokens | Encode, Metal | Encode, CPU | Metal, tokens/s | Root set, Metal: input, encode, tokens/s |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 × 88 | 121 | 31 ms | 246 ms | 3,950 | 94, 39 ms, 2,420 |
| 10 × 55–88 | 1,018 | 124 ms | 2,080 ms | 8,200 | 809, 230 ms, 3,510 |
| 50 × 55–88 | 4,998 | 577 ms | 10,223 ms | 8,660 | 3,984, 1,084 ms, 3,680 |
| 1 × 483 | 516 | 108 ms | 1,112 ms | 4,790 | 367, 140 ms, 2,620 |
| 10 × 450–486 | 4,965 | 858 ms | 10,786 ms | 5,790 | 3,552, 1,213 ms, 2,930 |
| 50 × 448–486 | 24,722 | 4,143 ms | — | 5,970 | 17,709, 6,037 ms, 2,930 |
| 1 × 936 | 969 | 286 ms | 2,289 ms | 3,390 | 512 (cut), 201 ms, 2,550 |
| 10 × 900–939 | 9,481 | 2,547 ms | 22,286 ms | 3,720 | 5,120 (cut), 1,926 ms, 2,660 |
| 50 × 898–939 | 47,295 | 12,612 ms | — | 3,750 | 25,600 (cut), 9,562 ms, 2,680 |

Metal is 8–18× the CPU. Per token the multilingual set runs 1.3–2.4×
the root set (its encoder is 22 layers of 768 against 28 of 1,024), less
for long states, where attention grows with the sequence. Load is 370–425 ms on Metal
and 570–600 ms on the CPU, against 73–87 ms for the root set: parsing
the 34 MB `tokenizer.json`.

**What it is for.** The oracle's language requests through both
checkpoints (`nuclis decide --json`, the answer and its confidence):

| Request | `laya` | `laya-multilingual` |
| --- | --- | --- |
| `french_choice` (department) | billing, 0.83 | billing, 1.00 |
| `german_noul` (threatens to cancel) | 0.06 | 0.93 |
| `spanish_score` (urgency, 0–2) | 1.56 | 1.54 |
| `arabic_choice` (sentiment) | neutral, 0.51 | negative, 0.13 |
| `chinese_choice` (department, question in Chinese) | billing, 0.32 | billing, 1.00 |
| `hindi_noul` (wants to close the account) | 0.00 | 1.00 |
| `long_state` (French log, most severe level) | info, 0.04 | error, 0.99 |

It reads the decision in other languages where the root set cannot.
Language identification is another matter: asked which of English,
French, German, or Spanish a one-sentence state is written in, the
multilingual set names French (0.34), German (0.35), and English right
and calls Spanish English (0.47); the root set names Spanish (0.91) and English
right and calls French Spanish and German English. The manual test's
three-option French question gives the multilingual set french 0.42
(the root set german 0.37).

## Limits and what is not done

- Metal's attention is the scalar per-(row, head) kernel, 45 % of a
  512-token forward; a kernel that shares each key block across a tile of
  queries is the first headroom. The matmuls are the generic F32 tile
  (about 3.5 TFLOP/s); the half-operand tiles would round the
  activations, whose outliers reach 1.65·10⁴, to F16 and break the
  bounds. A batch holds at most 2048 rows, which also bounds one
  sequence on Metal (the contracts' are 512 and 1,024). BF16 matrices are
  refused on Metal (both sets are F16).
- On Metal Laya shares the GPU with a language model in the same process;
  the two do not run at once (every `commit` waits), so a decision
  delays the model's next step by its encode time.
- Opening the multilingual set takes about 0.4 s, most of it parsing its
  34 MB `tokenizer.json` (580,604 merges); the root set opens in about
  75 ms on Metal. `nuclis serve` pays it once; a binary cache of the
  parsed vocabulary would remove it for `nuclis decide`.
- Language identification is not what either checkpoint was trained for:
  asked which language a sentence is in, the multilingual set names
  French and German right and Spanish wrong, the root set the reverse
  ([above](#the-multilingual-checkpoint)).
- The package's `action.act_probability` is not computed (the card says
  it carries no signal); its language-specific temperatures and its
  `Router` (automatic choice of checkpoint by language) are not
  implemented: the caller picks `--model laya-multilingual`.
- `decide --model` completes directories, not decision entry names.
