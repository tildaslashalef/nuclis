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

AGNT-09 closed on 2026-09-16 (Gemma 4 tool calling). The next family is
Meta's **Muse Glimmer 30B** (`unsloth/Muse-Glimmer-30B-GGUF`), a dense
30B agentic model with vision and a DFlash drafter, planned as four units
that mirror the Gemma bring-up (MODL-05..07) plus one for its tool protocol.
Nothing is pulled or coded yet; the facts below were read on 2026-09-16 from
the model card, the base repository's `config.json`, the remote GGUF header
(`nuclis model inspect`, no weights), and the pinned llama.cpp `7620399`,
which already implements the architecture (`src/models/muse-glimmer.cpp`)
and its chat format (`common_chat_params_init_muse_glimmer`), so the oracle
exists without a reference upgrade.

Order: MODL-09 → MODL-10 → MODL-11 → AGNT-10.

| Unit | Title | Sessions |
| --- | --- | --- |
| MODL-09 | Muse Glimmer 30B: artifact pin, facts, tokenizer, binding, CPU reference | 2 |
| MODL-10 | Muse Glimmer 30B: Metal plan | 1 |
| MODL-11 | Muse Glimmer 30B: profile (text, reasoning channel), catalogue, acceptance | 1 |
| AGNT-10 | Muse Glimmer ATEM tool calling: rendering, decoding, fixtures | 1 |

## The artifact (decided 2026-09-16)

**`Muse-Glimmer-30B-UD-Q4_K_XL.gguf`** at commit
`faa5b025c584459c13febfa5c59883516710ae39`, 15,878,222,368 B, SHA-256
`82bece304887a313ece08400bc030f6066c7bff5b906b0cd40308ec8a409fd38` (from
the listing; verified by the pull in MODL-09). Header: architecture
`muse-glimmer`, 52 blocks, embedding 6656, context 131072, 731 tensors:
313 F32, 410 Q4_K, 8 Q5_K (`output.weight` among them). Verdict today:
*not runnable: no adapter* — every encoding already has CPU and Metal
kernels, so the encoding side needs nothing.

Why this quantization: it is Meta's "K-Quant-17GB" tier (1.0 % measured
degradation across 15 benchmarks) and Unsloth's recommended starting point;
at 15.9 GB it leaves the 48 GB machine wide headroom for a 32K F16 KV cache
(52 layers × 1 KB/token ≈ 1.7 GB), prefill scratch, and later the vision
projector (`mmproj-kquant.gguf`, 1.40 GB) and the DFlash drafter
(`dflash-kquant.gguf`, 1.63 GB). Decode is memory-bound, so the projection
is the Qwen3.8-27B rate at the same byte count (~10 tok/s). The larger
files are quality upgrades to measure afterwards, not bring-up targets:
`UD-Q5_K_M` 19.2 GB, `UD-Q6_K_XL` 26.3 GB (Unsloth's "Mac 48 GB" row;
check its encodings by `inspect` first), `Q8_0` 29.6 GB (too tight beside
the cache and companions).

Companions in the repository, sizes from the listing, digests to be read
by `inspect` in MODL-09: `mmproj-kquant.gguf` 1,400,328,928 B,
`mmproj-Muse-Glimmer-30B-BF16.gguf` 3,849,173,728 B,
`dflash-kquant.gguf` 1,631,205,312 B.

## MODL-09 — Muse Glimmer 30B: artifact pin, facts, tokenizer, binding, CPU reference

**Facts read so far** (to be re-read from the pulled file with
`scripts/gguf-inventory.py` and recorded in `docs/reference/muse-glimmer.md`
with provenance):
- Dense causal transformer, 52 layers in a period-4 pattern
  (`attention.sliding_window_pattern = 4`: layers 0,1,2 sliding, 3 global;
  39 sliding + 13 global), window 2048, 32 query heads / 2 KV heads
  (GQA 16:1), head 128 (`attn_q` 6656×4096, `attn_k`/`attn_v` 6656×256),
  SwiGLU FFN 19968, vocabulary 202,048, embeddings untied (`output.weight`
  separate).
- RoPE θ 500,000 on sliding layers only; **NoPE on global layers**
  (`layer_rope_theta` is 0 there); rope type NORM (adjacent pairs).
- Per layer: `attn_norm`, `post_attention_norm`, `ffn_norm`,
  `post_ffw_norm` (the post norms use eps **1e-8**, the pre norms
  `rms_epsilon` 1e-5; the `weight + 1` is folded at conversion), per-head
  `attn_q_norm`/`attn_k_norm` of size 128 (`qk_scale_factor` 3.87 is
  folded into `attn_q_norm`; `attn_k_norm` is ones), and an **attention
  output gate** `attn_gate` 6656×4096: `sigmoid(gate(x)) ⊙ attn_out`
  before `attn_output`. Attention scale 1/√128.
- The input embedding is RMS-normalized **without a weight** before layer
  0; logits are scaled by `logit_scale` 0.196116 then soft-capped with
  tanh at 20 (`final_logit_softcapping`).
- Tokenizer: `tokenizer.ggml.model = gpt2`, `pre = llama4`, 439,802
  merges, BOS `<|begin_of_text|>` 200000, EOS `<|end_of_text|>` 200001,
  EOT `<|eot|>` 200008, `<|eom|>` 200007, `<|start|>` 200022,
  `<|message|>` 200023, `add_bos_token` true; 2,048 special tokens
  (200000–202047), most reserved. The `llama4` pre-type is the
  **gpt-4o regex** in the reference (case-aware letter runs with
  `\p{Lu}`/`\p{Ll}` classes and contractions, digits in runs of ≤3,
  punctuation with `[\r\n/]*`), which our `qwen35` splitter does not
  implement.
- Template SHA-256 `114f55ebdc1804c1af371197b9fdf2d6bb925966c9dfe46b73782a71bc07965e`
  (7,167 bytes, no trailing newline): the ATEM protocol (see AGNT-10). It
  differs from the Hub repository's `chat_template.jinja` (9,992 bytes,
  which normalizes a "Reasoning effort" line in the system text); the
  profile pins the GGUF digest as always.
- Sampling from the card: temperature 1.0, top-p 0.95, top-k 64; no
  `general.sampling.*` keys in the header. Reasoning strength
  low/medium/high/xhigh is a system-prompt line, not a template switch.

**Design.**
- Session 1: pull the main file (`nuclis model pull unsloth/Muse-Glimmer-30B-GGUF --file …`),
  record commit/size/digest and the companions' digests in
  [artifacts.md](docs/reference/artifacts.md); confirm the reference runs
  it (`llama-completion -ngl 99`, then `--jinja --single-turn`); write
  `docs/reference/muse-glimmer.md` from the inventory. Tokenizer: a
  `gpt4o` splitter beside `pre.zig` (the category table gains the
  uppercase/lowercase letter bits it needs — `scripts/tokenizer-unicode.py`
  regenerates `unicode-ranges.bin`), `encode.zig` selects it for
  `pre == "llama4"`, `tokenizer-fixtures.py` captures the strings and
  prompts, `vocabulary-check.zig` gains the expectations.
- Session 2: `models/muse_glimmer.zig` (validation over a committed
  inventory fixture, binder, layer kinds), `muse_glimmer_runtime.zig`
  (CPU schedule: unweighted embedding norm, gated attention, sandwich
  norms with two epsilons, NoPE globals, sliding window as a cache-row
  slice as Gemma does, logit scale + softcap), CPU primitives that are
  new (a sigmoid-gate multiply if `cpu` lacks one; a weightless RMS norm),
  the `<|begin_of_text|>Hello,` traces from the reference
  (`tests/fixtures/muse-glimmer-hello-comma/`, three positions) and a
  `make compare-muse-glimmer-cpu` at the bring-up thresholds (max abs
  2e-3, relative RMS 1e-4).

**Acceptance.** The file validates (731 tensors, 39 sliding + 13 global);
the tokenizer matches the reference's `/tokenize` on the captured strings
and prompts; the CPU reference matches the oracle traces at the thresholds
with the same greedy token; `make check` and `test-metal` unchanged.

## MODL-10 — Muse Glimmer 30B: Metal plan

**Design.** `muse_glimmer_metal.zig` composing existing kernels: Q4_K/Q5_K
matvec and the batched prefill tiles, RMS norm (a weightless variant for
the embedding norm, or a ones vector), RoPE NORM at θ 5e5 on sliding
layers only, flash-decoding attention with head 128 and 2 KV heads, the
window mask on prefill and the cache-row slice on decode (window 2048),
the sigmoid gate epilogue before the output projection (Qwen3.5's
attention gate path is the nearest existing kernel), SwiGLU, the logit
scale and tanh soft-cap (Gemma's). The KV cache stays full-context on
sliding layers (the ring layout remains the roadmap follow-up; 1.7 GB at
32K F16 is affordable).

**Acceptance.** `make compare-muse-glimmer` (CPU, Metal F32, Metal F16 at
their tolerances) on the pinned traces; `make test-generation-muse-glimmer-metal`;
`make bench` on Qwen unchanged; a first decode/prefill number recorded.

## MODL-11 — Muse Glimmer 30B: profile (text, reasoning channel), catalogue, acceptance

**Design.**
- `profiles/muse_glimmer.zig` pinned to the GGUF template digest. The
  prompt starts with `<|begin_of_text|>` as text (the encoder never adds
  BOS). Turns are `<|start|>ROLE<|message|>…<|eot|>`; two consecutive
  messages of one role end the first with `<|eom|>`. The system turn is
  the caller's text followed by `\n\nReasoning strength: LEVEL.` and
  `\n\n# Valid recipients: "self", "user".` (with tools, the tool block
  and namespaces come in between — AGNT-10). Without a system message the
  template synthesizes one ("You are a helpful AI assistant.", the
  knowledge cutoff 2026-01-04, and a current-date line when the engine
  defines `strftime_now`): **decided 2026-09-16** — the profile renders
  that default without the date line (it takes no clock), documented as a
  deviation and pinned with a fixture captured with `current_date` unset;
  the agent always sends a system message anyway.
  Assistant history: `reasoning_content` renders as
  `<|start|>assistant to=self<|message|>…<|eom|>` (kept everywhere the
  template keeps it — the template has no last-user gate), then the
  answer as `<|start|>assistant to=user<|message|>…<|eot|>`; the
  generation prompt is `<|start|>assistant`.
- **Effort.** Muse has low/medium/high/xhigh and no off. Add `high` to
  the shared `Effort` (Qwen's template also knows it: re-capture
  `qwen38-text.json` with a `high` case and pin its instruction line;
  Gemma treats it as on); Muse renders `off` as `low` (the model always
  opens its reasoning message) and the help text says so.
- **Decoder.** A new channel grammar in `profiles/stream.zig`: the
  generation prompt ends after `<|start|>assistant`, so the model's
  stream is a sequence of messages `HEADER<|message|>BODY(<|eom|>|<|eot|>)`
  where the first header arrives as ordinary text (` to=self`) and later
  ones follow a `<|start|>` control token. The header routes the body:
  `assistant to=self` → thinking, `assistant` / `assistant to=user` →
  answer, `assistant to=NAME` → a tool body handed to the profile's parser
  (AGNT-10). `<|eom|>` ends a message, `<|eot|>` ends the turn. This is
  a `StreamMarkers` variant selected by the profile, with the existing
  bracket grammar untouched; tests drive it with fake ids at every split
  point as the Gemma header test does.
- Stop tokens `<|eot|>`, `<|end_of_text|>`; sampling defaults from the
  card; `stream_markers` for the reasoning header.
- Catalogue entry `muse-glimmer-30b` with the companions (`mmproj`,
  and a new `dflash` role or the existing `mtp` role — decide: the
  companion table names what loads it); `nuclis --help`; the acceptance
  record (`scripts/reference-baseline.py --family muse-glimmer`,
  `make baseline-muse-glimmer`, the table in bench.md).

**Acceptance.** Text fixtures match byte for byte across the four levels;
`make test-vocabulary` on the file; `nuclis agent --model muse-glimmer-30b`
answers with its thinking shown; the acceptance run recorded.

## AGNT-10 — Muse Glimmer ATEM tool calling: rendering, decoding, fixtures

**Facts.** Declarations go into the system turn as prose plus one JSON
line per tool (`{"name": …, "description": …, "parameters": …}` in the
reference's `tojson` style, preceded by a `// Tool metadata` line per
namespace — a name's part before the first `.`, empty description — and
followed by a fixed example); the recipients line lists `"self"`, each
namespace as `"NS.*"`, and `"user"`. A call is its own message:
`<|start|>assistant to=NAME<|message|><atem:function_calls>\n<atem:invoke name="NAME">\n<atem:parameter name="K">V</atem:parameter>\n…</atem:invoke>\n</atem:function_calls>`
ended by `<|eom|>` when another message follows, else `<|eot|>`;
scalars are written as is, `true`/`false`/`null`, lists and objects as
JSON, strings verbatim (spaces not stripped). A result is its own turn:
`<|start|>tool NAME<|message|><tool_output name="NAME">\nCONTENT\n</tool_output><|eot|>`,
the name resolved from the call id. The reference parses the body with a
schema-aware grammar (string-typed parameters verbatim, others as JSON)
and treats `<|eot|>` as the end of the call step.

**Design.** Rendering in the profile from the shared `ToolDefinition` and
history; `parseTool` for the ATEM body (the inverse of the renderer: a
value that parses as JSON keeps its type, anything else is a literal
string — Qwen's rule); the header decoder from MODL-11 routes
`assistant to=NAME` bodies to it, and `<|eot|>` after a call ends the
step (several calls arrive as `<|eom|>`-separated messages before it).
`scripts/profile-tools-fixtures.py --profile muse_glimmer` captures the
same case shapes as Gemma's (a schema-rich declaration, nested values,
loops ending on results, two steps), and the ATEM markup inside content
or arguments is rejected as structure smuggling.

**Acceptance.** Tool fixtures byte for byte; parser round trip and
malformed-body rejections; a decoder test with two calls in one turn;
`make check`; a live `--print --json` turn on the pulled file executes
`write_file` then `read_file` with the file on disk as asked.
