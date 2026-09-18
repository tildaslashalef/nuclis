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

MODL-15 closed on 2026-09-18: Bonsai 2 27B is accepted and planned
ahead of Muse Glimmer. It is Prism ML's ternary re-encoding of
Qwen3.8-27B (`general.architecture = qwen35`, the pinned adapter,
tokenizer, and runtime apply unchanged), so the work is below the
adapter: two weight encodings (PQ2_0 id 142, PTQ1_0 id 143), a
Hadamard rotation of activations before each rotated projection, BF16
rows, and the PrismML llama.cpp fork as a second pinned oracle. The
catalogue entry `bonsai-2-27b` pins the PQ2_0 file with its Q8_0
projector (pulled 2026-09-18). Next is MODL-16 session 1: the fork as
an oracle, the facts document, the encodings and the transform on the
CPU. Muse Glimmer follows the three Bonsai units; its facts were read
on 2026-09-16 and its files are pulled.

Order: MODL-16 → KERN-10 → MODL-17 → MODL-11 → MODL-12 → MODL-13 → AGNT-10.
After AGNT-10 the roadmap continues with speculative decoding across the
families, then performance, then vision
([docs/roadmap.md](docs/roadmap.md)).

| Unit | Title | Sessions |
| --- | --- | --- |
| MODL-16 | Bonsai 2 27B: oracle, facts, ternary encodings, Hadamard transform, CPU reference | 2 |
| KERN-10 | Ternary matvec and matmul tiles, the Walsh-Hadamard kernel | 2 |
| MODL-17 | Bonsai 2 27B: the Qwen plan on rotated weights, catalogue, acceptance | 1–2 |
| MODL-11 | Muse Glimmer 30B: artifact pin, facts, tokenizer, binding, CPU reference | 2 |
| MODL-12 | Muse Glimmer 30B: Metal plan | 1 |
| MODL-13 | Muse Glimmer 30B: profile (text, reasoning channel), catalogue, acceptance | 1 |
| AGNT-10 | Muse Glimmer ATEM tool calling: rendering, decoding, fixtures | 1 |

## Bonsai 2 27B — the artifact (decided 2026-09-18)

`prism-ml/Ternary-Bonsai-2-27B-gguf` at commit
`6ed5e12bf84b7a63069882c91dd9e9218647d17b` (Apache-2.0, released
2026-09-17). **`Ternary-Bonsai-2-27B-PQ2_0.gguf`**, 7,206,168,928 B,
SHA-256 `3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1`,
is the catalogue's file: each trit in a 2-bit slot, 34 bytes per 128
weights, the packing Prism measures on Apple Silicon and the faster
prompt processing everywhere by its own table; the bring-up target
because its unpack is the Q4_0 nibble path's shape. The denser
**`Ternary-Bonsai-2-27B-PTQ1_0.gguf`** (5,946,648,928 B, SHA-256
`53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3`, 28
bytes per 128) is the footprint win and follows in the same units once
the plan runs; whether the entry moves to it is measured, not assumed.
Companions: `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` 629,246,976 B
(SHA-256 `6807ede6…`, pinned under `mmproj`), `…-mmproj-BF16.gguf`
931,145,856 B (`e287342d…`, the reference projector, not pinned), and an
F16 language file of 53.8 GB (not pulled). Stock llama.cpp rejects the
files (ids past its type count); the oracle is the PrismML fork at its
release `prism-b10687-5d80cff` (commit `5d80cff0b8cb…`, 2026-09-17,
MIT, tracks mainline) — pinned by tag, since the fork has already
retired one format (its legacy group-128 `Q2_0`).

Why: the same architecture as the flagship at 7.2 GB (5.95 GB in
PTQ1_0) instead of 16.5 GB, and decode is memory-bound, so the byte
count alone projects two to three times the Qwen3.8 rate; Prism's own
M4 Pro number is 18.0 tok/s decode and 125 pp512 on a pre-rotation
build, the M5 Pro 27.7 tok/s at about 201 GB/s. Quality claims (98.2 %
of the FP16 average over 14 thinking-mode benchmarks, measured through
vLLM on H100) are Prism's; nuclis proves equivalence to the fork, not
quality.

**Facts read on 2026-09-18** from the PTQ1_0 header (first 16 MiB by
range request, parsed directly), the model card, the whitepaper, and the
fork's `prism-v7` headers; to be re-read from the pulled file with
`scripts/gguf-inventory.py` and recorded in `docs/reference/bonsai.md`
with provenance:
- Header: 851 tensors (PTQ1_0 file): 402 ternary (every matrix: the
  embedding `token_embd.weight` [5120, 248320], the untied head
  `output.weight`, `attn_qkv`, `attn_gate`, `ssm_out`, `ffn_gate`,
  `ffn_up`, `ffn_down`, and the attention layers' projections), 353 F32
  (norms, `ssm_a`, `ssm_conv1d`, `ssm_dt.bias`, …), 96 **BF16** (id 30:
  `ssm_alpha.weight` and `ssm_beta.weight` [5120, 48] on the 48 DeltaNet
  layers — stored today, not decoded; the Qwen3.8 file holds them in
  another encoding). `general.file_type` 143; `general.name` "Hf",
  `general.version` "v5", `general.basename` "folded".
- Every `qwen35.*` key equals the pinned Qwen3.8-27B's (64 blocks, 5120,
  FFN 17408, 24 / 4 heads, key and value length 256, rope sections
  [11, 11, 10, 0], θ 1e7, epsilon 1e-6, `full_attention_interval` 4,
  ssm conv 4 / state 128 / groups 16 / rank 48 / inner 6144, context
  262,144). Sampling hints `general.sampling.temp` 1.0, `top_p` 0.95,
  `top_k` 20 (the card adds instruct-mode 0.7 / 0.80 / 20 with presence
  penalty 1.5; reasoning effort `xhigh` by default, `medium` supported,
  `low` behaves like `xhigh`).
- Tokenizer identical in kind: `gpt2` / `pre = qwen35`, 248,320 tokens,
  247,587 merges, BOS = PAD 248044, EOS 248046, `add_bos_token` false.
  Template SHA-256 `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`
  (8,952 bytes) — **not** the pinned `qwen38` digest (`12827f24…`): the
  alias gate proves it on every fixture or MODL-17 writes a variant.
- **Encodings** (fork `ggml-common.h`, branch `prism-v7`): `PQ2_0` id
  142, block 128: `ggml_half d; uint8_t qs[32]` (2 bits per element, 34
  B, 2.125 bpw). `PTQ1_0` id 143, block 128: `uint8_t qs[24]` (5 trits
  per byte, base-3, 120 values), `uint8_t qh[2]` (4 trits per byte, 8
  values), `ggml_half d` (28 B, 1.75 bpw) — the trit packing of
  mainline's `TQ1_0` (id 34, group 256) at group 128, so the unpacking
  contract is mainline's. Both: `w = d · t`, `t ∈ {−1, 0, +1}`; the
  fork's file types `MOSTLY_PQ2_0` 128 / `MOSTLY_PTQ1_0` 129. The fork
  also carries `GGML_HINT_SRC0_IS_HADAMARD` and `fwht` kernels (CUDA,
  Vulkan, SYCL; the Metal one lives inside its Metal backend sources).
- **Rotation** (`prism.hadamard.*`): `version` 1, `block_size` 1024,
  `transform` `normalized-sylvester-walsh-hadamard`, `axis`
  `input-last-dimension`, `sign_mode` `explicit`, `sign_widths`
  [5120, 6144, 17408] with `sign_values` (28,672 = 5120 + 6144 + 17408
  entries of ±1: one sign vector per input width), `weight_names` (401
  tensors: `output.weight` and per block `attn_qkv`, `attn_gate`,
  `ssm_out`, `ffn_down`, …), `inverse_weight_names` [`token_embd.weight`],
  `gdn_v_grouped` true. Per the whitepaper: `R = (1/√n) · Hₙ · S` with
  n = 1024 and S the diagonal of signs, folded into the stored weights;
  inference computes `W · (R x)` — a sign flip then a blockwise fast
  Walsh-Hadamard transform of the activation before each rotated
  projection, and the inverse on the embedding row after lookup. At
  batch 1 Prism calls the transform "one of the larger non-matmul costs
  of a decode step" on Metal and fuses the sign flip into its load
  path. What `gdn_v_grouped` changes (the value part of the DeltaNet
  `attn_qkv` rotated per group?) is read from the fork's loader in
  MODL-16, never guessed.
- Prism's Apple numbers (pre-rotation build, "pending re-measurement"):
  M4 Pro 18.0 tg128 / 125 pp512 at 7.2 GB; M5 Pro 28.7 / 393; M5 Max
  47.0 / 765; current build M5 Pro PQ2_0 27.7 / 397, PTQ1_0 27.1 / 369.

## MODL-16 — Bonsai 2 27B: oracle, facts, ternary encodings, Hadamard transform, CPU reference

**Design.**
- Session 1: the fork as the second oracle — clone at the release tag
  into `.zig-cache/reference/prism-llama.cpp`, build as the mainline
  recipe does, record the recipe and revision in
  `docs/reference/reference-baseline.md` beside the mainline pin (the
  trace harness and `reference-baseline.py` take the revision as a
  parameter or a second constant; two pins, never one moving one);
  confirm it runs the pulled file (`llama-completion -ngl 99`, then
  `--jinja`); `scripts/gguf-inventory.py` on the file into a committed
  fixture; write `docs/reference/bonsai.md` (artifact, header, the two
  block layouts from the format contract, the rotation contract as the
  fork's loader implements it — `gdn_v_grouped` included — with the
  whitepaper as the mathematical source); capture `Hello,` traces from
  the fork (`tests/fixtures/bonsai-hello-comma/`). `quant.row` arms for
  ids 142 and 143 with dequantization fixtures produced by the fork's
  own dequantize (pinned rows, as Q4_0's), `quant.row` for BF16 (id 30)
  and `executableEncoding` for all three in `qwen35`; the digest
  comparison of the chat template through the alias gate.
- Session 2: `cpu.hadamard` (sign flip and normalized blockwise FWHT of
  block 1024, in place over an activation of width 5120 / 6144 / 17408,
  against an F64 reference and a fixture from the fork), the inverse on
  the embedding row; `qwen35_runtime.zig` applies the transform before
  every rotated projection as the metadata names them (the adapter
  validates the `prism.hadamard.*` keys against a pinned contract and
  refuses unknown versions, sign modes, or names); `make
  compare-bonsai-cpu` at the bring-up thresholds (2e-3 / 1e-4) with the
  same greedy token.

**Acceptance.** The file validates (851 tensors, three encodings in the
executable set); the fork's traces match on the CPU at the thresholds;
`make check`, `compare` (Qwen unchanged), `test-metal` unchanged.

## KERN-10 — Ternary matvec and matmul tiles, the Walsh-Hadamard kernel

**Design.** `dequant.metal` gains `nu_dequant_pq2_0` and
`nu_dequant_ptq1_0` bit-identical to `quant.row`; `kernels.metal` gains
`nu_matvec_pq2_0` / `nu_matvec_ptq1_0` derived from the Q4_0 kernel (a
32-byte 2-bit block per 128 values; the base-3 unpack by the fixed-point
multiply of mainline's contract) with the bandwidth target of the Q4_0
set (≥ 200 GB/s on the model's shapes: 10240 × 5120, 6144 × 5120,
17408 × 5120, 5120 × 17408, 248320 × 5120), `nu_tile_*` and the
`nu_matmul_*` / `_32` instantiations for prefill; `nu_fwht_signed` (one
threadgroup per 1024-block, signs applied on load, in place; a rows
variant for chunks) and the inverse for the embedding; `metal-check`
entries (exact decode of the fixture rows, randomized rows through the
specialized and generic kernels, tiles against the generic F32 tile, the
transform against the CPU at F32 exactness); `make bench-kernels` and
`bench-matmul` rows for both encodings. Fusing the transform into the
matvec's input load is measured as a second step, not assumed.

**Acceptance.** `test-metal` with the new entries; `bench-kernels`
numbers for both encodings recorded in metal-backend.md; the existing
kernels' numbers unchanged.

## MODL-17 — Bonsai 2 27B: the Qwen plan on rotated weights, catalogue, acceptance

**Design.** `qwen35_metal.zig` dispatches the transform before each
rotated projection (and the inverse after the embedding gather) when the
binding carries the rotation, with no change on the plain Qwen file;
`make compare-bonsai` (CPU, Metal F32, Metal F16 at their tolerances)
and `make test-generation-bonsai-metal`; the profile: `.qwen38` through
the alias if the fixtures prove the template identical, else
`profiles/bonsai.zig` (the sampling defaults from the header, `xhigh`
default); the catalogue entry's `profile` filled; `nuclis --help`; the
acceptance record against the fork's server on the Qwen arrays
(`tests/fixtures/run-2026-09-06` tokens apply since the vocabulary is
identical, re-verified by the script's tokenizer check; the reference
side is the fork's run, recorded with its own revision;
`make baseline-bonsai`); the agent check; PTQ1_0 measured on the same
plan (`--model <path>`) and the entry moved only if it is not slower;
`make bench` on Qwen unchanged.

**Acceptance.** Traces at the thresholds on every path; the generation
check; the acceptance table in bench.md; the live tool turn; the Qwen
rate unchanged.

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
