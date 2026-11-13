<!-- kind: card -->
# Laya

Laya (`convaiinnovations/laya`, Apache-2.0) is a decision model: a
ModernBERT encoder with a typed-decision head. It answers a typed question
(`choice`, `score`, or `noul`, a yes/no probability) about a state (text,
or a JSON document) in one forward pass, with no decoding. Catalogue
entries: `laya`, the English root checkpoint (ModernBERT-large), and
`laya-multilingual`, the same head on an mmBERT-base encoder
([below](#the-multilingual-checkpoint)), both from commit `55cf4c4e` and run
by the same code: `inference/src/models/modernbert.zig`,
`modernbert_runtime.zig`, `laya.zig`, `laya_metal.zig`, the input contract
in `inference/src/profiles/laya.zig`, the `Decider` in
`inference/src/decide.zig`, and the command in `src/decide.zig`. The
measurements are in [the record](../benchmarks/record/laya.md).

**Why it is worth having beside a language model.** Every question
re-reads its state (the encoder is bidirectional, so nothing is shared
between questions), and a language model must spend decode tokens writing
each question, so delegating one decision it could make itself saves
nothing. It pays as a **filter**: one question fanned out over many states
the language model never reads (search hits, log sections, diff hunks),
where the language model's prefill is the cost avoided: each 512-token
state Qwen3.8-27B skips saves about 5.4 s of prefill (the README's
[Results](../../README.md#results)), against about 0.2 s for Laya to judge
it on Metal.

## Architecture

| Fact | `laya` | `laya-multilingual` |
| --- | --- | --- |
| Encoder layers, hidden, heads | 28, 1,024, 16 of 64 | 22, 768, 12 of 64 |
| Intermediate (GeGLU) | 2,624 | 1,152 |
| Vocabulary | 50,368 (byte-level BPE) | 256,000 (Gemma's, Metaspace) |
| Attention | global where `i % 3 == 0`, else `\|i − j\| ≤ 64` | the same |
| RoPE theta, global / local | 160,000 / 10,000 | 160,000 / 160,000 |
| Sequence and head budgets | 512, 192 | 1,024, 256 |
| Special tokens | `[CLS]`, `[SEP]`, `[MASK]` | `<bos>`, `<eos>`, `<mask>` |

**The encoder**, from `encoder/config.json` (`modernbert.parseConfig` reads
the Transformers 4.x and 5.x spellings and rejects biases, an activation
other than exact GELU, head widths not a multiple of 8, and other layer
types). No biases; LayerNorm eps 1e-5 with a weight and no bias:

```
x = LayerNorm(embed[ids])
per layer:  x += Wo(attention(rope(Wqkv(attn_norm(x)))))     # attn_norm: identity on layer 0
            a, g = halves of Wi(mlp_norm(x));  x += mlp.Wo(gelu(a) * g)
x = LayerNorm_final(x)
```

`Wqkv` rows are q, k, v, each `[heads][64]`. RoPE is split-half (pairs `d`
and `d + 32`), each angle rounded to F32 before its cosine and sine, as
the reference forms it. Attention is bidirectional, scale 1/8. GELU is the
exact erf form.

**The head** (`model.safetensors`, unprefixed): `type_emb.weight [3,1024]`
added to every row (choice 0, score 1, noul 2); two PyTorch
`TransformerEncoderLayer`s (`d=1024`, 16 heads, feed-forward 4096,
`norm_first`, ReLU, LayerNorm eps 1e-5 with bias, every linear with bias).
At each option's `[MASK]` row the scorer gives the option's logit:
`LayerNorm → Linear → GELU → Linear(→ 1)`. `act_head.*` and `temperature`
are known and unused; `Laya.open` rejects any other tensor, and any
missing or misshapen one.

## Files and encodings

A directory: `encoder/config.json`, `tokenizer.json` and
`tokenizer_config.json` ([tokenizer.md § Hugging Face `tokenizer.json`](../engine/tokenizer.md#hugging-face-tokenizerjson)),
`rl_agent_config.json`, and `model.safetensors` in F16 (root set
842,609,210 B, multilingual 643,835,514 B). The CPU decodes every matrix
but the embedding table to F32 at open (about 1.5 GB for the root set);
Metal wraps the F16 or F32 bytes in place for the generic F32 tile and
refuses BF16.

**On the CPU** the kernels are `backends/cpu/dense.zig`, each output summed
by one task in a fixed order, so results do not depend on the core count.
**On Metal** (`laya_metal.zig` `Plan`) sequences pack back to back into
batches of at most 2,048 rows with no padding; a bounds buffer gives each
row its sequence, `attentionSegments` restricts it to that sequence's keys
(and the window), and `Decider.decide` hands every question about every
state to one `Laya.logitsBatch`. A logit is bit-identical alone or packed.

## The input contract

As the `laya` 0.3.20 package defines it, in `profiles/laya.zig`:

- **Questions.** `{"type", "instructions", "criteria", "labels"}`. `choice`
  takes criteria as an object of key → description or a list of keys;
  `score` a non-empty list of levels, lowest first; `noul` an optional
  object keyed only `true`/`false` (case-insensitive), with optional
  `labels` (two distinct non-empty strings) that word the options without
  changing which is true. Every failure names the question.
- **Options.** choice `"key: description"` (or `"key"`); score
  `"level i: text"`; noul `"<false label>: <text or 'no, the statement does
  not hold'>"` then the true one. Structured values render as Python's
  `json.dumps(…, ensure_ascii=False)`.
- **The sequence.** `[CLS] tok("<type> question: <instructions>") [SEP]`,
  then per option `[MASK] tok(" " + text)` (first 48 tokens), `[SEP]`, the
  state's tokens, `[SEP]`, with the checkpoint's own three special tokens;
  the mask token spelled in any text becomes a space first. If the options
  leave under 16 of the head budget, each is cut to
  `max(4, (head − 16) / count)`; the question keeps `max(8, what remains)`;
  the state gets the rest, cut at its tail, or at its head for a list state
  (a conversation, newest last). Options that still do not fit are
  `OptionsExceedBudget`.

## Calibration and answers

`p = softmax(logits / T)` in F32. `T` comes from `rl_agent_config.json`:
`temperature_by_options` by bucket `"<type>:<2|3-5|6-10|11+>"`, else
`temperature[type]`, **clamped to [0.5, 5.0]** (a non-number applies as
1.0); `--uncalibrated` uses T = 1.

| Type | Value | `confidence` | `answer_confidence` |
| --- | --- | --- | --- |
| choice | the argmax key | `1 − H(p) / ln k`, clipped (1 when k < 2) | `max(p)` |
| score | `Σ i·p[i]` (zero-based) | the same | `max(p)` |
| noul | `p[1]` = P(true) | `max(p1, 1 − p1)` | `max(p)` |

`answer_confidence` is the one temperature scaling fits, so the terminal
view shows it. Numbers are rounded to 4 places as Python's `round` does.

## `nuclis decide`

<!-- gen:cli-synopsis:decide -->
```text
nuclis decide --request <file|-> [options]
nuclis decide --questions <file> <states> [options]
nuclis decide <inline questions> <states> [options]
<states>: --state <text> and --state-file <path>, repeatable
<inline questions>: --choice <text> --option <key[=desc]>…,
--score <text> --level <text>…, --noul <text>; each [--id <name>]
```
<!-- /gen -->

`--request` takes Jev's shape (`{"questions": {id: definition}, "state":
…}`, plus `"states": [...]` and `{"file": path}` states). One state renders
every answer with its distribution; several render ranked by the first
question, a cut state flagged. `--explain` shows each sequence decoded,
its budget split, bucket, and logits; `--truncate head|tail` picks the end
of a long state that is cut; `--backend cpu|metal` (default metal in a
Metal build). `--json` gives `{schema_version, model, repo, revision,
timings_ms, results}`, each result a complete Jev response whose answers
carry exactly the fields TypeSafe documents, everything else under a
`nuclis` object. The checkpoint is `--model` or `decide.model` (default
`laya`): a registry entry of kind `decision`, a catalogue entry
([catalogue.md § The catalogue](catalogue.md#the-catalogue)), or a
directory. `nuclis serve` answers the same request as `POST /v1/decisions`
and Jev's `POST /v1/systemone`
([api-decisions.md](../guide/api-decisions.md)).

## The multilingual checkpoint

`multilingual/` in the same repository and commit, with the root set's
tensor names (169 F16 tensors and the F32 `temperature`) and temperatures
of 1. Nothing in the model code differs; what does is outside it: the
Metaspace tokenizer (byte fallback, 580,604 merges, numbers split to single
digits, so an English log line is about 40 % more tokens), the special
tokens `tokenizer_config.json` names (`<mask>` in a text becomes a space,
`[MASK]` stays text), and the 1,024-token budget. It reads decisions in
French, German, Spanish, Arabic, Chinese, and Hindi where the root set
cannot ([record](../benchmarks/record/laya.md#the-multilingual-checkpoint-checked-2026-09-29)).

## What is verified

<!-- gen:gates-for:laya -->
| Gate | Tier | Model | Evidence |
| --- | --- | --- | --- |
| `laya-vocabulary` | `verify` | laya | [tokenizer.md](../engine/tokenizer.md) |
| `laya-cpu` | `verify` | laya | [laya.md](laya.md) |
| `laya-metal` | `verify` | laya | [laya.md](laya.md) |
| `laya-multilingual-vocabulary` | `verify` | laya_multilingual | [tokenizer.md](../engine/tokenizer.md) |
| `laya-multilingual-cpu` | `verify-release` | laya_multilingual | [laya.md](laya.md) |
| `laya-multilingual-metal` | `verify-release` | laya_multilingual | [laya.md](laya.md) |
<!-- /gen -->

Bounds are relative, because the pre-norm residual carries outliers up to
1.65·10⁴: per tensor, the largest difference and the difference's RMS over
the reference's at most 1e-5; per logit, at most 1e-4 of max(1, |logit|).

## Numbers

One noul question over synthetic log states, Metal, encode time (M4 Pro,
ReleaseSafe, 2026-09-29; [record](../benchmarks/record/laya.md#time-per-call-2026-09-29)):

| States × tokens | `laya` | `laya` tokens/s | `laya-multilingual` | tokens/s |
| --- | ---: | ---: | ---: | ---: |
| 1 × about 60 | 39 ms | 2,440 | 31 ms | 3,950 |
| 50 × about 60 | 1,344 ms | 3,500 | 577 ms | 8,660 |
| 1 × about 500 | 207 ms | 2,470 | 108 ms | 4,790 |
| 50 × about 500 | 9,801 ms | 2,610 | 4,143 ms | 5,970 |

Metal is 8–20× the CPU. Opening takes about 75 ms on Metal for the root
set and 0.4 s for the multilingual one.

## Known limits

- Metal's attention is a scalar per-(row, head) kernel, 45 % of a
  512-token forward, its cost growing with the sequence. The matmuls are
  the generic F32 tile (about 3.5 TFLOP/s): half-operand tiles would round
  the outliers to F16 and break the bounds.
- A batch holds at most 2,048 rows, which also bounds one sequence on Metal.
- On Metal Laya shares the GPU with a language model in the same process;
  a decision delays the model's next step by its encode time.
- Opening the multilingual set spends most of its 0.4 s parsing its 34 MB
  `tokenizer.json`; `nuclis serve` pays it once.
- Neither checkpoint identifies a sentence's language reliably.
- `action.act_probability`, the package's language-specific temperatures,
  and its `Router` are absent: the caller picks the checkpoint.
- `decide --model` completes directories, not decision entry names.
