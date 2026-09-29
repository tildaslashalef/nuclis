# TODO — active plan

This file is the only progress tracker. It holds **unfinished work only**:
where we are, what is next, and the design of each remaining unit. When a
unit closes, its outcome moves to
[docs/engineering-log.md](docs/engineering-log.md) and
its section is deleted here; when the last unit closes, this file is emptied
back to this header. Requirements live in [docs/spec.md](docs/spec.md); the
engine map in [docs/architecture.md](docs/architecture.md); how to build,
test, and measure in [docs/development.md](docs/development.md).

Session protocol (also in [AGENTS.md](AGENTS.md)): read this file first. If
it lists work, summarize *Where we are* and ask the user how to continue. If
it is empty, ask what to work on and write the agreed plan here.


## Where we are

Theme agreed 2026-09-27 (user): Laya (`convaiinnovations/laya`, a
ModernBERT encoder with a typed-decision head, Apache-2.0) end to end in
nuclis, behind a decision API any client can call: a person writing the
state and questions by hand, a script, or later an LLM. Why it is worth
having beside an LLM: every question re-reads its state (the encoder is
bidirectional, so nothing is shared across questions), and an LLM must
spend decode tokens writing each question, so delegating one decision the
LLM could make itself saves nothing. It pays as a **filter**: one question
fanned out over many states the LLM never reads (30 search hits, every
log section, each diff hunk), where the LLM's prefill is the cost avoided
(Qwen3.8-27B prefills at about 44 tok/s). Already landed: safetensors pull (MODL-28) and the safetensors
loader (MODL-29); the root checkpoint is pulled at
`~/.nuclis/models/convaiinnovations/laya/` (commit `55cf4c4e`). Kept to
three units on purpose, to iterate fast.

MODL-30 sessions 1 and 2 landed 2026-09-29: the oracle's fixtures, the
Hugging Face tokenizer (gate `laya-vocabulary`), and the ModernBERT
encoder and decision head on the CPU matching the oracle at every stored
stage (gate `laya-cpu`). Start with MODL-30, session 3: the profile (input
contract and calibration), `Decider`, and `nuclis decide`; the unit closes
at its end.

## Order

| Unit | Title | Sessions |
| --- | --- | --- |
| MODL-30 | Laya on the CPU: oracle, `tokenizer.json`, ModernBERT, the decision head, `nuclis decide` | three (2 done); closes once |
| MODL-31 | Laya on Metal: bidirectional windowed attention, the encoder plan, measured | one or two |
| AGNT-18 | The agent's `decide` tool: LLM-written questions over tool-supplied states (the experiment) | one |

Decisions (user, 2026-09-27): the oracle is Laya's own Python package;
the root checkpoint first (multilingual later, same family code); CPU
first, then Metal; a new command, `nuclis decide`, with three input tiers,
fan-out over many states, a styled terminal view, and `--json`; each
`results[i]` a complete Jev response; LLM-written questions are the last
unit, an experiment.

## MODL-30 — Laya on the CPU, end to end

### Session 1 — delivered 2026-09-29 (oracle and tokenizer)

- Oracle: `scripts/laya-reference.py` (venv recipe in its docstring and in
  docs/development.md § Environment) wrote
  `inference/src/models/fixtures/laya/` (989 KB): `requests.json` (8
  requests: `choice_described`, `choice_labels`, `score`, `noul`,
  `json_state`, `long_text` (512 tokens, tail cut), `long_list` (list
  state, head cut), `choice_20` (options shrunk to 8 tokens, bucket
  `choice:11+` clamped); per request the ids, markers, raw logits, the
  package's answer and usage, the texts it tokenized, and a `tensors`
  index into `activations.f32`), `tokens.json` (32 texts → ids). The
  script checks that the package's (padded, batched) answer equals
  calibration applied to its own batch-1 logits.
- Tokenizer: `inference/src/tokenizer/hf_json.zig` (`hf_tokenizer.parse`,
  `Tokenizer.encode`), `gpt2.zig` (splitter), `nfc.zig` + generated
  `nfc_table.zig` (`scripts/tokenizer-nfc.py`, UCD 17.0.0; passes every
  NFC column of `NormalizationTest.txt`). Design in
  docs/reference/tokenizer.md § Hugging Face `tokenizer.json`.
- Check: `zig build test-vocabulary -- ~/.nuclis/models/convaiinnovations/laya`
  (a directory selects the Laya mode of `inference/vocabulary-check.zig`),
  gate `laya-vocabulary` (tier verify, model `laya` in `gates.json`):
  passes, 32 text cases and every text of the 8 requests.
- Docs touched: tokenizer.md, development.md, THIRD_PARTY_NOTICES.md.

### Facts (read 2026-09-27 at `55cf4c4e`; corrected 2026-09-29 from the package and the fixtures)

Files under the set: `model.safetensors` (206 tensors, 205 F16 + 1 F32),
`encoder/config.json`, `rl_agent_config.json`, `tokenizer/tokenizer.json`,
`tokenizer/tokenizer_config.json`.

**Encoder** (`encoder/config.json`, `model_type` `modernbert`): 28 layers,
hidden 1024, 16 heads (head dim 64), intermediate 2624, vocabulary 50368,
`norm_eps` 1e-5, no biases anywhere, layer types `full_attention` at
`i % 3 == 0`, else `sliding_attention` with `local_attention` 128; RoPE
theta 160000 (full) and 10000 (sliding); `hidden_activation` `gelu`.
Tensors (`encoder.` prefix): `embeddings.tok_embeddings.weight
[50368,1024]`, `embeddings.norm.weight`, per layer `attn_norm.weight`
(**absent on layer 0**: identity), `attn.Wqkv.weight [3072,1024]`,
`attn.Wo.weight [1024,1024]`, `mlp_norm.weight`, `mlp.Wi.weight
[5248,1024]`, `mlp.Wo.weight [1024,2624]`, then `final_norm.weight`.
Shapes are PyTorch `[out, in]`, row-major.

The forward, confirmed in Transformers 5.17's `modeling_modernbert.py`
(the oracle's): `x = LayerNorm(embed[ids])`; per layer `x +=
Wo(attn(attn_norm(x)))`, `x += mlp.Wo(gelu(a) * g)` where `a, g` are the
first and second halves of `Wi(mlp_norm(x))`; `Wqkv` output rows are q,
k, v, each `[16 heads][64]`; RoPE rotate-half (pairs `d` and `d + 32`),
inverse frequencies `theta^(-2i/64)`, positions 0.., applied in F32 to q
and k with the layer type's theta; attention bidirectional, scale 1/8;
sliding layers see keys with `|i − j| ≤ 64` (inclusive); GELU exact erf;
`final_norm` last. LayerNorm has weight and no bias, eps 1e-5.

**Head** (`model.safetensors`, no prefix): `type_emb.weight [3,1024]`
added to every position (choice 0, score 1, noul 2); two `head.layers.N`,
each PyTorch `TransformerEncoderLayer(d=1024, nhead=16, ff=4096,
norm_first=True)`, ReLU, LayerNorm eps 1e-5 with bias, all linears with
bias: `x += out_proj(mha(norm1(x)))`, `x += linear2(relu(linear1(norm2(x))))`,
`in_proj_weight [3072,1024]` stacking q, k, v, scale 1/8; then at each
option's `[MASK]` position `scorer`: `LayerNorm(scorer.0) →
Linear(scorer.1) → GELU (erf) → Linear(scorer.3, → 1)`. `act_head.*` and
the `temperature` buffer are not used (the package still reports
`action.act_probability`; nuclis omits it: the card says it carries no
signal).

**Input contract** (`laya/common.py` `build_sequence`, `render_options`;
`laya/agent.py` `_check_question`, `_to_internal`, `_encode_state`):
`[MASK]` spelled in instructions, options, or state is replaced by a
space before tokenizing. `[CLS] tok("<type> question: <instructions>")
[SEP]`, then per option `[MASK] tok(" " + text)[:48]`, `[SEP]`, the
state's tokens, `[SEP]`; `tok` adds no specials. Non-string instructions
are `json.dumps(ensure_ascii=False)` (separators `", "`, `": "`). Option
texts: choice `"key: description"`, or `"key"` when the description is
null or `""` (criteria as a list are keys without descriptions); score
`"level i: text"`; noul `"<false label>: <false text or 'no, the
statement does not hold'>"`, `"<true label>: <… or 'yes, the statement
holds'>"`, labels default `false`/`true` (optional `labels` object;
criteria keys lowercased); non-string criterion values are compact JSON
(separators `", "`, `": "`). Budgets: `max_len` 512, `head_max_len` 192;
if the options leave under 16 tokens, each option is cut to `max(4,
(head_max_len − 16) / count)` tokens; the question text is cut to `max(8,
remaining)`; the state gets `max(0, max_len − len − 1)` tokens, its head
kept, except a **list** state (a conversation), which keeps its tail.
Markers past `max_len` are dropped and the question is then an error
(`options exceed head_max_len`). A state object is Python
`json.dumps(state, ensure_ascii=False)`.
Question validation to mirror: type one of choice/score/noul;
`instructions` present; choice criteria a non-empty dict or list; score a
non-empty list; noul criteria absent or a dict keyed only true/false;
`labels` only on noul, two distinct non-empty stripped strings.

**Calibration**: `p = softmax(logits / T)`, `T` from
`temperature_by_options` by bucket `"<type>:<2|3-5|6-10|11+>"` (k = option
count), else `temperature[type]`, then **clamped to [0.5, 5.0]** (a
non-number is 1.0): the shipped `choice:11+` is 0.1006 and runs as 0.5.

**Answers** (`agent.py` `_decode_answers`, every number rounded to 4
places): choice `{type, choice, probabilities {key: p}, confidence,
answer_confidence}`; score `{type, score = Σ i·p[i], legend {"i": text},
probabilities {"i": p}, confidence, answer_confidence}`; noul `{type, noul
= p[1], confidence = max(p1, 1 − p1), answer_confidence}`. `confidence` is
`1 − H(p) / ln k` clipped to [0, 1] (1 when k < 2) for choice and score;
`answer_confidence` is max(p) on every type (the calibrated one). Usage
`{input_tokens: sequence length summed over questions, output_tokens: 0}`.

**Tokenizer** (landed in session 1): 50280 model entries and 116 added
tokens, 88 of them new ids (50280–50367), so 50368 ids; `[CLS]` 50281, `[SEP]` 50282, `[PAD]` 50283, `[MASK]`
50284 (`lstrip`); whitespace runs of 2–24 spaces are added tokens.

### Session 2 — delivered 2026-09-29 (encoder and head on the CPU)

- `inference/src/models/modernbert.zig` (`parseConfig`, `bind`, `Binder`),
  `modernbert_runtime.zig` (`Model.load`, `Model.forward`, F32; all but the
  embedding table decoded at load), `laya.zig` (`Head`, `Laya.open`,
  `Laya.logits(io, gpa, ids, markers, kind, out, trace)`, `Kind`, `Stage`,
  `Trace`; unknown, missing, and misshapen tensors rejected), and
  `inference/src/backends/cpu/dense.zig` (threaded F32 `matmul`,
  `layerNorm`, windowed `attention` over `Io` tasks). Unit tests: the
  kernels against F64 and the existing attention reference; a tiny
  synthetic checkpoint through `Laya.open` (stages, rejections, allocation
  failures).
- Check: `zig build test-laya -Doptimize=ReleaseFast -- <dir>`
  (`inference/laya-check.zig`), gate `laya-cpu` (tier verify, 16 s):
  passes all 8 requests at every stored stage. Bounds became relative
  (the plan's absolute 1e-4 fails where the residual reaches 10²–10⁴):
  worst scaled max 8.3e-6, rel RMS 7.0e-6, scaled logit 9.0e-6; 512 tokens
  in 2.4 s, open in 0.2 s warm. Same numbers in a Debug build, no leaks.
  Table in docs/reference/laya.md § On the CPU (created; session 3 adds
  the contract, calibration, the command, and the filter argument).
- Docs touched: architecture.md (the safetensors row, `backends/cpu`,
  the `*_runtime.zig` exception), reference/laya.md (new).

### Session 3 — profile, `Decider`, `nuclis decide`

- `inference/src/profiles/laya.zig`: the input contract above (render,
  `[MASK]` replacement, budgets, list-state head cut, truncation flag, the
  question validation), `rl_agent_config.json` read (temperatures,
  `max_len`, `head_max_len`), calibration with the clamp, both
  confidences. Pure; tested on every `requests.json` request: its ids and
  markers from `question` and `state` (Python `json.dumps` of the object
  states, which `head_text`/`options`/`state_text` spell), and its answer
  from `logits`.
- `inference/src/decide.zig`, the library API over `models.laya.Laya`
  (CPU until MODL-31 adds Metal):
  `Question { kind: choice|score|noul, instructions, options, labels }`
  (`kind` is `models.laya.Kind`),
  `Answer { probabilities, logits, confidence, answer_confidence,
  temperature, bucket }`, `StateResult { answers, state_tokens, truncated
  }`, `Decider.open(gpa, io, dir, backend)` (reads `tokenizer.json` through
  `hf_tokenizer.parse`, bounded at 64 MiB), `decide(arena, states,
  questions) → [states][questions] answers` (every pair one sequence),
  `deinit`. Host limits as constants: 64 states, 32 questions, 64
  options, 1 MiB per state.
- `src/decide.zig` + `src/cli.zig` + `src/help.zig`: `nuclis decide`.
  Input tiers, all building the same request: `--request <file|->` (Jev
  shape, plus `states: [...]` and `{"file": path}` states); `--questions
  <file>` with `--state <text>` / `--state-file <path>` (repeatable);
  inline `--choice <text> --option key[=desc]…`, `--score <text> --level
  <text>…`, `--noul <text>`, `--id <name>`. Output: the styled view (one
  state: per question the answer, bars, `answer_confidence`; several
  states: ranked by the first question, truncation flagged per row) and
  `--json` (`schema_version`, model, repo, revision, `timings_ms` {load,
  tokenize, encode}, `results[i]` = a complete Jev response `{answers,
  usage}` with the answer fields above, plus `nuclis` {state,
  state_tokens, truncated}; each answer's extras under `nuclis` {logits,
  temperature, bucket}). `--explain` (the rendered sequence decoded with
  `bpe.decode(…, special=true)`, budget split, bucket, raw logits),
  `--uncalibrated`, `--truncate head|tail`. Before closing, check the
  answer field names against TypeSafe's published API page, not only
  Laya's server.
- Model selection: `decide.model` config key and `--model`; a catalogue
  entry `laya` pinning the root set (commit `55cf4c4e`, weights SHA-256
  `891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c`,
  support files by name), which needs the catalogue to describe a
  safetensors set; the registry's `kind: decision`, `--register`
  accepting safetensors for that kind only; then give `gates.json`'s
  `laya` model its `entry`.
- Docs: finish `docs/reference/laya.md` (the input contract, calibration,
  answers, `nuclis decide`, CPU time per request with tokenizing and
  calibration included, and the filter argument from *Where we are*; move
  the *Facts* above into it), `docs/spec.md`
  (the decide surface and its requirements), `docs/architecture.md` (the
  decision path beside `Engine`).
- Checks: `make check`; the eight fixture requests through the fresh
  binary (`--json`) equal the package's answers within 1e-4; the card's
  quickstart (department / urgency / churn) through each input tier gives
  the same JSON; a fan-out over 4 files renders ranked; malformed
  requests fail with the question named. CPU time per request recorded
  (load separate). `make verify` once at close (the unit added CPU
  kernels and touched shared modules, though no existing numerics); no CPU
  tier (no existing CPU kernel or family forward changed).

## MODL-31 — Laya on Metal

- The encoder and head as a Metal plan beside the CPU runtime: F16
  weights on the GPU (existing F16 matmul/matvec kernels where they fit;
  a batched-sequence matmul is the main shape), LayerNorm without bias,
  GeGLU, and a **new attention kernel: bidirectional, full or windowed
  (`|i − j| ≤ 64`)**, over one or several sequences padded to a common
  length with a key mask (the existing kernels are causal).
- `Decider` backend `metal` by default when built with Metal;
  `--backend cpu|metal` on `nuclis decide`.
- Checks: Metal against the CPU and the fixtures (F32 accumulation;
  bounds recorded per layer); resource lifetime and cleanup tests like
  the other plans; measured on the M4 Pro: 1, 10, 50 questions at 64 and
  512 state tokens, load time separate, written into
  `docs/reference/laya.md` with hardware, build, and commit.
  `make verify` once (shared kernels touched).

## AGNT-18 — The agent's `decide` tool (experiment)

- A tool in `src/agent/tools/` calling the in-process `Decider` (loaded
  on first use, `decide.model`): one question (choice, score, or noul
  with its options) over up to 64 candidates given as workspace file
  paths or as the items of the previous tool result; returns the
  candidates ranked with probabilities, truncation marked; limits are host
  constants; failures are results.
- Its description and the system-prompt line measured on the playground
  task list before and after (`make agent-eval VARIANT=…`), on Qwen3.8-27B
  and Gemma 4 E4B: task success, wall time, and prefill tokens saved.
  Kept only if it helps; the result is logged either way.
