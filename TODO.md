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

KERN-09 closed on 2026-09-18: the expert kernels exist on both paths
(`cpu.experts` reference; decode `route`, `matvecExperts`, `geluMulRows`,
`combineExperts`; prefill `expertLists` and the gathered `matmulExperts`
tiles), checked in `test-metal` and measured by `make bench-experts`
([metal-backend.md § Gathered expert kernels](docs/reference/metal-backend.md#gathered-expert-kernels-kern-09)).
Next is MODL-09 session 1: the 26B-A4B adapter and CPU reference (the
file is already pulled and verified; its facts are below). Two families
are planned, in this order: **Gemma 4 26B-A4B** (the first mixture of
experts; the kernels now exist, everything else reused) and then Meta's
**Muse Glimmer 30B** (a dense agentic model with a new tokenizer splitter
and a new chat-protocol decoder). The 26B-A4B facts were read on
2026-09-16 from the remote QAT header (`nuclis model inspect`, no
weights) and the pinned llama.cpp `7620399` (`src/models/gemma4.cpp`);
the Muse facts from its model card, the base repository's `config.json`,
the remote GGUF header, and the same reference, which already implements
both architectures and chat formats, so the oracles exist without a
reference upgrade.

Order: MODL-09 → MODL-10 → MODL-11 → MODL-12 → MODL-13 → AGNT-10.
All four files of both families are pulled and verified under
`~/.nuclis/models` (2026-09-17) and both have catalogue entries ahead of
their adapters; after AGNT-10 the roadmap continues with speculative
decoding across the families, then performance, then vision
([docs/roadmap.md](docs/roadmap.md)).

| Unit | Title | Sessions |
| --- | --- | --- |
| MODL-09 | Gemma 4 26B-A4B: artifact pin, facts, adapter, CPU reference, Metal plan | 2 |
| MODL-10 | Gemma 4 26B-A4B: catalogue, acceptance record, agent check | 1 |
| MODL-11 | Muse Glimmer 30B: artifact pin, facts, tokenizer, binding, CPU reference | 2 |
| MODL-12 | Muse Glimmer 30B: Metal plan | 1 |
| MODL-13 | Muse Glimmer 30B: profile (text, reasoning channel), catalogue, acceptance | 1 |
| AGNT-10 | Muse Glimmer ATEM tool calling: rendering, decoding, fixtures | 1 |

## Gemma 4 26B-A4B — the artifact and its facts (read 2026-09-16)

**`gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf`** from
`unsloth/gemma-4-26B-A4B-it-qat-GGUF` at commit
`7b92b5b28818151e8669af2e45e88d6086f490dd`, 14,249,047,104 B, SHA-256
`a7c5bc715f5ff8e99a3e8901ce7d2b42b402c669bf24f7c5250747633d0f5891`
(verified by the pull in MODL-09). Architecture `gemma4`, 30 blocks,
embedding 2816, context 262144, 658 tensors: 392 F32 and 266 Q4_0 — the
Q4_0 path from MODL-08 executes every weight. The K-quant sibling
(`unsloth/gemma-4-26B-A4B-it-GGUF`, 17.0 GB) stores its expert
down-projections as Q5_1, which the engine does not have, so the QAT file
is the only target. Verdict today: *not runnable: the gemma4 adapter
rejects the file (UnsupportedConfiguration)* — the expert tensors.

- **Template digest `845f1ee4…`, the 12B's**: the Gemma profile, its
  fixtures, and the tool calling from AGNT-09 apply unchanged. Sampling
  hint in the header: temperature 1.0, top-p 0.95, top-k 64 (the 12B's).
- **Attention** is the 12B's period-6 pattern (five sliding layers, window
  1024, head 256, 8 KV heads, RoPE base 1e4; one global layer, head 512,
  base 1e6) with two differences to read carefully: the global layers have
  **2 KV heads** (`head_count_kv` array alternates 8 and 2; the 12B had
  one), and `rope.dimension_count` is 512 for global layers (the 12B
  rotated 128 of 512 with `rope_freqs` factors) — whether factors exist
  here is read from the full inventory when the unit starts. Shapes: 16
  query heads; sliding `attn_q` 2816×4096, `attn_k`/`attn_v` 2816×2048;
  global `attn_q` 2816×8192, `attn_k` 2816×1024. Logits soft-capped at 30,
  `layer_output_scale` per layer, tied embeddings (262144 × 2816).
- **Every layer is an expert layer** (`expert_count` 128, `expert_used_count`
  8, `expert_feed_forward_length` 704) with a shared dense FFN
  (`feed_forward_length` 2112) beside it. Per layer: `ffn_gate_inp.weight`
  2816×128 (F32) with `ffn_gate_inp.scale` [2816]; fused
  `ffn_gate_up_exps.weight` [2816, 1408, 128] and `ffn_down_exps.weight`
  [704, 2816, 128] (Q4_0) with `ffn_down_exps.scale` [128]; the dense
  `ffn_gate`/`ffn_up`/`ffn_down`; norms `ffn_norm`, `post_ffw_norm_1`,
  `pre_ffw_norm_2`, `post_ffw_norm_2`, `post_ffw_norm`, `attn_norm`,
  `post_attention_norm`, and the per-head q/k norms.
- **The reference's FFN block on an expert layer** (`gemma4.cpp`):
  shared branch `post_ffw_norm_1(GELU-FFN(ffn_norm(x)))`; expert branch:
  router logits = `ffn_gate_inp · (rms_norm(x, eps) / sqrt(2816) ⊙ gate_inp.scale)`
  over the *pre-norm* attention output, softmax over 128, top-8 with the
  selected weights renormalized to sum 1 (`norm_w`), each expert a
  gated-GELU FFN over `pre_ffw_norm_2(x)` with the fused gate-up rows and
  the per-expert down scale, weighted sum, then `post_ffw_norm_2`; the
  layer adds both branches and continues as the 12B (post norm, residual,
  output scale).
- **Active bytes per token**: about 3.8B parameters (8 experts × 30
  layers ≈ 1.4B, shared FFNs ≈ 0.5B, attention ≈ 1.1B, the tied head
  ≈ 0.7B) ≈ 2.1 GB at Q4_0, against 16 GB for the dense 27B — the reason
  this family comes first.

## MODL-09 — Gemma 4 26B-A4B: artifact pin, facts, adapter, CPU reference, Metal plan

**Design.**
- Session 1: pull the QAT file and the companions it ships
  (`nuclis model pull unsloth/gemma-4-26B-A4B-it-qat-GGUF --file …`,
  digests into [artifacts.md](docs/reference/artifacts.md)); confirm the
  reference runs it; extend `docs/reference/gemma4.md` with the 26B-A4B
  section (inventory, layer pattern, the expert block, the global-layer
  KV/rope differences). Adapter: `models/gemma4.zig` accepts the expert
  configuration (validation of the 3-D expert tensors, the scales, the
  extra norms, 2 KV heads on global layers, the rope dimension count),
  keeps the 12B's validation exact, and binds both layouts; the CPU
  runtime gains the expert FFN block (router chain, top-8, gathered
  experts from KERN-09's CPU reference, shared branch, the two post norms
  and their sum); `<bos>Hello,` traces from the reference at three
  positions under `tests/fixtures/gemma4-26b-a4b-hello-comma/` and
  `make compare-gemma4-26b-a4b-cpu` at the bring-up thresholds.
- Session 2: the Metal plan (`gemma4_metal.zig` extended, not forked):
  the expert FFN block on decode and in the chunked prefill from KERN-09,
  the global attention with 2 KV heads, memory plan with all experts
  resident (14.2 GB) and no paging. `make compare-gemma4-26b-a4b`
  (CPU, F32, F16) and `make test-generation-gemma4-26b-a4b-metal`.

**Acceptance.** Traces match the oracle at the thresholds with the same
greedy token; the 12B and QAT-12B comparisons unchanged; `make bench` on
Qwen unchanged; first decode/prefill numbers recorded.

## MODL-10 — Gemma 4 26B-A4B: catalogue, acceptance record, agent check

**Design.** The catalogue entry `gemma-4-26b-a4b` exists since 2026-09-17
(the QAT file with its `mmproj-BF16.gguf` and `MTP/…-Q4_0.gguf` companions,
profile `gemma4`, all pulled and verified); what remains is its verdict
turning *supported* once the adapter binds, `nuclis --help`, the acceptance record (`scripts/reference-baseline.py
--family gemma4-26b-a4b`, token arrays under `tests/fixtures/run-<date>-gemma4-26b-a4b/`,
`make baseline-gemma4-26b-a4b`, the table in bench.md); the ring layout
for windowed caches stays a roadmap follow-up. Live: `nuclis agent
--model gemma-4-26b-a4b` runs the AGNT-09 write-then-read check, and the
decode rate against the 12B and the 27B is recorded in the log.

**Acceptance.** The record's four prompt lengths at 32K on the token
budget; the live tool turn; documents updated (gemma4.md, bench.md,
artifacts.md, architecture.md § adding a model).

## Muse Glimmer 30B — the artifact (decided 2026-09-16)

**`Muse-Glimmer-30B-UD-Q4_K_XL.gguf`** at commit
`faa5b025c584459c13febfa5c59883516710ae39`, 15,878,222,368 B, SHA-256
`82bece304887a313ece08400bc030f6066c7bff5b906b0cd40308ec8a409fd38` (from
the listing; verified by the pull in MODL-11). Header: architecture
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
by `inspect` in MODL-11: `mmproj-kquant.gguf` 1,400,328,928 B,
`mmproj-Muse-Glimmer-30B-BF16.gguf` 3,849,173,728 B,
`dflash-kquant.gguf` 1,631,205,312 B.

## MODL-11 — Muse Glimmer 30B: artifact pin, facts, tokenizer, binding, CPU reference

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

## MODL-12 — Muse Glimmer 30B: Metal plan

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

## MODL-13 — Muse Glimmer 30B: profile (text, reasoning channel), catalogue, acceptance

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
- The catalogue entry `muse-glimmer-30b` exists since 2026-09-17 with
  `mmproj-kquant.gguf` and `dflash-kquant.gguf` under the `mtp` role
  (decided: the role names the draft source the speculative-decoding unit
  loads, whatever its mechanism) and a `null` profile: this unit fills the
  profile in (`profile = .muse_glimmer`, and `catalog.Entry.profile` may
  become non-optional again); `nuclis --help`; the acceptance
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
string — Qwen's rule); the header decoder from MODL-13 routes
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
