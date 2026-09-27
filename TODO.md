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
three units on purpose, to iterate fast. Start with MODL-30, session 1.

## Order

| Unit | Title | Sessions |
| --- | --- | --- |
| MODL-30 | Laya on the CPU: oracle, `tokenizer.json`, ModernBERT, the decision head, `nuclis decide` | three; closes once |
| MODL-31 | Laya on Metal: bidirectional windowed attention, the encoder plan, measured | one or two |
| AGNT-18 | The agent's `decide` tool: LLM-written questions over tool-supplied states (the experiment) | one |

Decisions (user, 2026-09-27): the oracle is Laya's own Python package;
the root checkpoint first (multilingual later, same family code); CPU
first, then Metal; a new command, `nuclis decide`, with three input tiers,
fan-out over many states, a styled terminal view, and `--json`; each
`results[i]` a complete Jev response; LLM-written questions are the last
unit, an experiment.

## MODL-30 — Laya on the CPU, end to end

### Facts (read 2026-09-27 at `55cf4c4e`)

Files under the set: `model.safetensors` (206 tensors, 205 F16 + 1 F32),
`encoder/config.json`, `rl_agent_config.json`, `tokenizer/tokenizer.json`,
`tokenizer/tokenizer_config.json`.

**Encoder** (`encoder/config.json`, `model_type` `modernbert`): 28 layers,
hidden 1024, 16 heads (head dim 64), intermediate 2624, vocabulary 50368
(padded; the tokenizer has 50285 ids), `norm_eps` 1e-5, no biases
anywhere (`attention_bias`, `mlp_bias`, `norm_bias` false), layer types
`full_attention` at `i % 3 == 0`, else `sliding_attention` with
`local_attention` 128; RoPE theta 160000 (full) and 10000 (sliding);
`hidden_activation` `gelu`. Tensors (`encoder.` prefix):
`embeddings.tok_embeddings.weight [50368,1024]`, `embeddings.norm.weight`,
per layer `attn_norm.weight` (**absent on layer 0**: identity),
`attn.Wqkv.weight [3072,1024]`, `attn.Wo.weight [1024,1024]`,
`mlp_norm.weight`, `mlp.Wi.weight [5248,1024]` (input and gate halves of
2624), `mlp.Wo.weight [1024,2624]`, then `final_norm.weight`. Shapes are
PyTorch `[out, in]`, row-major.

The forward, to confirm against the fixtures in session 2 (the details
marked † come from the reference implementation's structure, not yet
from a fixture): `x = LayerNorm(embed[ids])`; per layer
`x += Wo(attn(attn_norm(x)))`, `x += mlp.Wo(gelu(a) * g)` where
`a, g = split(Wi(mlp_norm(x)))` (a first †); attention is bidirectional,
RoPE on q and k (rotate-half, the layer type's theta †), scale 1/8,
sliding layers see keys with `|i − j| ≤ 64` † (half the window); GELU is
exact erf †; `final_norm` last. LayerNorm here has weight and no bias.

**Head** (`model.safetensors`, no prefix): `type_emb.weight [3,1024]`
added to every position (question type index: choice 0, score 1, noul 2);
two `head.layers.N` layers, each PyTorch `TransformerEncoderLayer(d=1024,
nhead=16, ff=4096, norm_first=True)`, activation ReLU (the default †):
`x += out_proj(mha(norm1(x)))`, `x += linear2(relu(linear1(norm2(x))))`,
all with biases, LayerNorm with bias, `in_proj_weight [3072,1024]` stacking
q, k, v; then at each option's `[MASK]` position `scorer`:
`LayerNorm(scorer.0) → Linear(scorer.1) → GELU → Linear(scorer.3, → 1)`.
`act_head.*` and the `temperature` buffer (all ones) are not used: the
card says the act probability carries no signal, and the temperatures come
from `rl_agent_config.json`.

**Input contract** (`rl_common.py` `build_sequence`, `render_options`):
`[CLS] tok("<type> question: <instructions>") [SEP]`, then per option
`[MASK] tok(" " + text)[:48]`, `[SEP]`, the state's tokens, `[SEP]`; no
special tokens inside `tok`. Option texts: choice `"key: description"` (or
`"key"`), score `"level i: text"`, noul `"false: <false text or 'no, the
statement does not hold'>"`, `"true: <… or 'yes, the statement holds'>"`.
Budgets: `max_len` 512, `head_max_len` 192 (`rl_agent_config.json`); if
the options leave under 16 tokens, each option is cut to
`max(4, (head_max_len − 16) / count)` tokens; the question text is cut to
`max(8, remaining)`; the state gets `max_len − len − 1` tokens, cut at the
tail (head kept). A state object is serialized as Python
`json.dumps(state, ensure_ascii=False)` (separators `", "` and `": "`).
Options that still do not all fit are an error in Laya.
Calibration: `p = softmax(logits / T)`, `T` from `temperature_by_options`
by bucket `"<type>:<2|3-5|6-10|11+>"`, else `temperature[type]`;
confidence `1 − H(p) / ln k` (choice and score); noul is `p[1]`; score is
`Σ i·p[i]` (zero-based).

**Tokenizer** (`tokenizer/tokenizer.json`): BPE, 50280 vocabulary entries
+ added tokens, 50009 merges, `byte_fallback` false, normalizer NFC,
pre-tokenizer ByteLevel (`add_prefix_space` false, `use_regex` true: the
GPT-2 split), special ids `[CLS]` 50281, `[SEP]` 50282, `[PAD]` 50283,
`[MASK]` 50284. The same algorithm family as Qwen's byte-level BPE: reuse
`inference/src/tokenizer/bpe.zig` and `vocabulary.zig`; what is new is
reading `tokenizer.json` and NFC.

### Session 1 — oracle and tokenizer

1. Oracle: `scripts/laya-reference.py` (never imported by the build) run
   in a venv at `.zig-cache/reference/laya-venv` with `pip install
   laya==0.3.20` on CPU, `USE_TF=0`; it loads the pulled directory (not a
   second download), runs `build_sequence` and the model in F32 on CPU,
   and writes `inference/src/models/fixtures/laya/`: for 6 fixed requests
   (choice with and without descriptions, score, noul, a JSON-object
   state, a state long enough to truncate, a 20-option choice that
   shrinks) the token ids and marker positions, the encoder output at
   layers 0, 1, 3, and 27 for rows 0, the markers, and the last row, the
   head output at the markers, the raw logits, and the calibrated answer.
   Record the oracle's versions (laya, torch, transformers) in the
   fixture and in `THIRD_PARTY_NOTICES.md` under references consulted.
   Keep committed fixtures under 1 MB.
2. `inference/src/tokenizer/hf_json.zig`: `tokenizer.json` (BPE model,
   ByteLevel pre-tokenizer, NFC normalizer, added tokens) → the existing
   vocabulary/BPE types; unsupported models, pre-tokenizers, or
   normalizers are typed errors naming them. NFC: check first whether
   the Unicode data already in the tree covers composition; if not, add
   it the way `unicode-ranges.bin` was added (recorded in the notices).
   Check: the fixture's token ids for every request, plus a text set with
   accents, CJK, emoji, and code.
3. End the session by rewriting sessions 2–3 below at this level with
   anything the fixtures contradicted.

### Session 2 — encoder and head on the CPU

- `inference/src/models/modernbert.zig`: `Config` from `config.json`
  (bounded read; unknown `model_type` or unsupported values rejected),
  `bind(checkpoint, config) → Weights` validating every tensor name,
  shape, and dtype (F16/BF16/F32) and rejecting extras.
- `inference/src/models/modernbert_runtime.zig`: F32 CPU forward for one
  unpadded sequence (`[len]` ids → `[len][1024]` hidden), reusing
  `backends/cpu` matmul, norm, and RoPE where their shapes allow; weights
  decoded from the checkpoint through `safetensors.Ref.decode` once at
  load (about 1.7 GB of F32 for the root set; state this in the doc).
- `inference/src/models/laya.zig`: the head: type embedding, the two
  layers, the scorer at marker rows → one logit per option.
- Check: every layer the fixtures hold, max abs error ≤ 1e-4 and relative
  RMS ≤ 1e-5 against the oracle's F32; logits within 1e-4.

### Session 3 — profile, `Decider`, `nuclis decide`

- `inference/src/profiles/laya.zig`: the input contract above (render,
  budgets, truncation flag), `rl_agent_config.json` read (temperatures,
  `max_len`, `head_max_len`), calibration and confidence. Pure; tested on
  the fixtures' token ids and answers.
- `inference/src/decide.zig`, the library API:
  `Question { kind: choice|score|noul, instructions, options }`,
  `Answer { probabilities, logits, confidence, temperature, bucket }`,
  `StateResult { answers, state_tokens, truncated }`,
  `Decider.open(gpa, io, dir, backend)`,
  `decide(arena, states, questions) → [states][questions] answers`
  (every pair one sequence), `deinit`. Host limits as constants: 64
  states, 32 questions, 64 options, 1 MiB per state.
- `src/decide.zig` + `src/cli.zig` + `src/help.zig`: `nuclis decide`.
  Input tiers, all building the same request: `--request <file|->` (Jev
  shape, plus `states: [...]` and `{"file": path}` states); `--questions
  <file>` with `--state <text>` / `--state-file <path>` (repeatable);
  inline `--choice <text> --option key[=desc]…`, `--score <text> --level
  <text>…`, `--noul <text>`, `--id <name>`. Output: the styled view (one
  state: per question the answer, bars, confidence; several states:
  ranked by the first question, truncation flagged per row) and `--json`
  (`schema_version`, model, repo, revision, `timings_ms` {load, tokenize,
  encode}, `results[i]` = a complete Jev response `{answers, usage}` plus
  `nuclis` {state, state_tokens, truncated}; each answer's extras under
  `nuclis` {logits, temperature, bucket}). `--explain` (the rendered
  sequence decoded, budget split, bucket, raw logits), `--uncalibrated`,
  `--truncate head|tail`. Before closing, check the answer field names
  against TypeSafe's published API page, not only Laya's server.
- Model selection: `decide.model` config key and `--model`; a catalogue
  entry `laya` pinning the root set (commit `55cf4c4e`, weights SHA-256
  `891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c`,
  support files by name), which needs the catalogue to describe a
  safetensors set; the registry's `kind: decision`, `--register`
  accepting safetensors for that kind only.
- Docs: `docs/reference/laya.md` (the family, the contract, the oracle,
  numbers, and the filter argument from *Where we are*), `docs/spec.md` (the decide surface and its requirements),
  `docs/architecture.md` (the decision path beside `Engine`).
- Checks: `make check`; the six fixture requests through the fresh binary
  (`--json`) equal the oracle's answers within 1e-4; the card's quickstart
  (department / urgency / churn) through each input tier gives the same
  JSON; a fan-out over 4 files renders ranked; malformed requests fail
  with the question named. CPU time per request recorded (load separate).
  No Metal tier (no change to existing numerics).

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
