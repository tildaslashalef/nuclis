# Gemma 4: facts from the artifacts and the reference

The second architecture through the adapter seam (MODL-04–MODL-08), and
since MODL-09 also the 26B-A4B mixture of experts
([§ Gemma 4 26B-A4B](#gemma-4-26b-a4b-the-expert-configuration-modl-09)),
bound by the same adapter as a second pinned configuration. Everything here was
read on 2026-09-11 from the pinned files with `scripts/gguf-inventory.py`
(an independent header reader, no weights), from `nuclis model inspect`,
and from the pinned llama.cpp reference `7620399` (`src/models/gemma4.cpp`,
`src/llama-vocab.cpp`, `conversion/gemma.py`, `ggml-cpu/ops.cpp`), which
is read as a format and semantics reference and used as the numerical
oracle, never copied. Where a fact comes from the reference source rather
than the file, the sentence says so. Nothing below is from memory of the
model card.

## Artifacts

| | K-quant (bring-up, MODL-05–MODL-07) | QAT (the catalogue's file since MODL-08) |
| --- | --- | --- |
| Repository | `unsloth/gemma-4-12b-it-GGUF` | `unsloth/gemma-4-12B-it-qat-GGUF` |
| File | `gemma-4-12b-it-UD-Q4_K_XL.gguf` | `gemma-4-12B-it-qat-UD-Q4_K_XL.gguf` |
| Commit | `fc034cfff751…` | `980b060c40a8…` |
| Size | 7,366,423,360 B | 6,716,356,800 B |
| Encodings | F32 338, Q4_K 242, Q5_K 31, Q6_K 56 | F32 338, Q4_0 329 |
| `general.name` | `Gemma-4-12B-It` | `Gemma-4 12B IT (smart Q4_0, QAT-lossless)` |
| `tokenizer.ggml.eos_token_id` | 106 (`<turn|>`) | 1 (`<eos>`), plus `eot_token_id` |

Both files have the same 58/48 architecture keys where they overlap, the
same 667 tensor names and shapes, the same vocabulary, and the same chat
template (SHA-256 `845f1ee48e39fc942fe190da9df6a1c5db229e17a96ea08966ad1c9274e73d1b`,
18,924 bytes). Digests and companions are pinned in
[artifacts.md](artifacts.md#pinned-commits-and-digests-modl-02-2026-09-11). The
inventory of the K-quant file is the fixture
`inference/src/models/fixtures/gemma4-12b.json` (all keys, arrays up to
64 elements retained, the template as a length/offset/SHA descriptor).

## Metadata (`gemma4.*`)

| Key | Value |
| --- | --- |
| `block_count` | 48 |
| `context_length` | 262144 |
| `embedding_length` | 3840 |
| `feed_forward_length` | 15360 |
| `attention.head_count` | 16 |
| `attention.head_count_kv` | array of 48: `8` on sliding layers, `1` on global layers |
| `attention.sliding_window_pattern` | array of 48 booleans: true on sliding layers |
| `attention.sliding_window` | 1024 |
| `attention.key_length` / `value_length` | 512 (global layers) |
| `attention.key_length_swa` / `value_length_swa` | 256 (sliding layers) |
| `rope.dimension_count` / `dimension_count_swa` | 512 / 256 |
| `rope.freq_base` / `freq_base_swa` | 1,000,000 / 10,000 |
| `attention.layer_norm_rms_epsilon` | 1e-6 |
| `final_logit_softcapping` | 30.0 |
| `attention.shared_kv_layers` | 0 (every layer has its own cache) |
| `embedding_length_per_layer_input` | 0 (no per-layer embeddings) |
| `general.sampling.temp` / `top_p` / `top_k` | 1.0 / 0.95 / 64 (the file's own sampling hint) |

**Layer pattern.** Both 48-element arrays are the same period-6 pattern:
layers 5, 11, 17, 23, 29, 35, 41, 47 are global (full causal attention,
head size 512, one KV head); the other 40 are sliding (window 1024, head
size 256, 8 KV heads). The reference derives the same from the pattern
array; the upstream `config.json` says `layer_types` alternate
`sliding_attention` ×5, `full_attention` ×1, with `global_head_dim` 512.

The parser must retain 48-element arrays: `formats/gguf` keeps values for
numeric arrays of at most 16 elements today (a Qwen-era bound), so MODL-05
raises that bound (a format-level capacity, not a Gemma key).

## Tensors (667)

Non-block: `token_embd.weight` [3840, 262144] Q4_K, `output_norm.weight`
[3840] F32, `rope_freqs.weight` [256] F32. There is **no `output.weight`**:
the output projection is the embedding matrix (tied; the reference creates
`output` as a duplicate of `token_embd`).

Per block (14 tensors on sliding layers, 13 on global ones):

| Tensor | Sliding layer | Global layer |
| --- | --- | --- |
| `attn_norm.weight`, `post_attention_norm.weight`, `ffn_norm.weight`, `post_ffw_norm.weight` | [3840] F32 | same |
| `attn_q.weight` | [3840, 4096] = 16 × 256 | [3840, 8192] = 16 × 512 |
| `attn_k.weight` | [3840, 2048] = 8 × 256 | [3840, 512] = 1 × 512 |
| `attn_v.weight` | [3840, 2048] | **absent** |
| `attn_output.weight` | [4096, 3840] | [8192, 3840] |
| `attn_q_norm.weight`, `attn_k_norm.weight` | [256] F32 | [512] F32 |
| `ffn_gate.weight`, `ffn_up.weight` | [3840, 15360] | same |
| `ffn_down.weight` | [15360, 3840] | same |
| `layer_output_scale.weight` | [1] F32 | same |

Encodings per role in the K-quant file: attention and gate/up matrices
Q4_K on 43 layers and Q5_K on 5; `attn_v` Q6_K on 35 layers, Q5_K on 5;
`ffn_down` Q4_K/Q5_K/Q6_K (22/5/21); `attn_output` Q4_K everywhere;
`token_embd` Q5_K. All are in the executable set, as is the QAT file's
Q4_0 since MODL-08 ([§ Q4_0 path](#q4_0-path-and-the-qat-file-modl-08-2026-09-12)).

**Values read from the file.** `rope_freqs.weight` is 64 ones followed by
192 values of 1e30. `layer_output_scale` is a per-layer scalar between
0.0045 and 0.89 (0.053 on layer 0, 0.048 on layer 47, 0.0045 on layer 11).
Norm weights are stored **raw**: the converter's `Gemma4Model.norm_shift`
returns 0 (Gemma 3's added 1), so `attn_norm` values range −143…193 with
mean 6.6 and `attn_q_norm` on layer 0 is the constant 1.0234, `attn_k_norm`
the constant 0.1221. The reference's `build_norm` is `rms_norm(x) * w`,
nothing added.

## Forward pass (from the reference graph, `gemma4.cpp`)

Notation: `d` = 3840, `rms(x) = x / sqrt(mean(x²) + 1e-6)`, `norm_w(x) =
rms(x) * w`.

1. **Embedding.** `x = E[token] * sqrt(d)` (the scale is applied to token
   embeddings only, not to injected image embeddings).
2. **Per layer** (`h` is the residual stream):
   - `a = norm_attn(h)`.
   - `q = Wq a` reshaped [heads=16][hd]; `q = norm_q(q)` per head over `hd`;
     `q = rope(q)`.
   - `k = Wk a` reshaped [kv][hd]; `v = Wv a` when the layer has `attn_v`,
     **otherwise `v = k` before any norm** (global layers). Then
     `k = norm_k(k)` per head, `v = rms(v)` per head **without a weight**,
     and `k = rope(k)`. Order matters: V is the raw K projection, RMS-normed
     without weight, never rotated.
   - Attention scores are `q · k` with **scale 1.0** (`f_attention_scale =
     1.0`, "no pre-attn scaling"; the norms carry the scaling), causal, and
     on sliding layers masked to `p_key > p_query − 1024` (the reference's
     `LLAMA_SWA_TYPE_STANDARD`: masked when `p1 − p0 ≥ n_swa`, so exactly
     1024 positions including the query's own are visible). No attention
     logit softcap (`attn_soft_cap` is false). GQA: 16 query heads over 8 KV
     heads (groups of 2) on sliding layers, over 1 KV head on global layers.
   - `o = Wo · concat(heads)`; `h = norm_post_attn(o) + h`.
   - `f = norm_ffn(h)`; `f = Wdown · (gelu_tanh(Wgate f) ⊙ (Wup f))`
     (`LLM_FFN_GELU` + `LLM_FFN_PAR`, i.e. `ggml_geglu_split`; `ggml_gelu` is
     the tanh approximation `0.5·x·(1 + tanh(√(2/π)·x·(1 + 0.044715·x²)))`).
   - `h = (norm_post_ffn(f) + h) * layer_output_scale` (the scalar
     multiplies the whole new residual stream, after the add).
3. **Head.** `y = norm_out(h)`; `logits = Eᵀ y` (tied); `logits =
   30 · tanh(logits / 30)`.

**RoPE.** NeoX pairing (`LLAMA_ROPE_TYPE_NEOX` for `LLM_ARCH_GEMMA4`): pair
`i` is dimensions `(i, i + n_rot/2)`, rotated by angle `pos · θ_i`,
`θ_i = base^(−2i/n_rot) / ff_i`. Sliding layers: `n_rot = 256` (the whole
head), base 10,000, no factors. Global layers: `n_rot = 512` (the whole
head), base 1,000,000, factors `rope_freqs.weight`: pairs 0–63 rotate
(`ff = 1`), pairs 64–255 get `ff = 1e30`, so `θ ≈ 0` and those dimensions
are unrotated. That is the upstream "proportional" RoPE with
`partial_rotary_factor 0.25`: 128 of 512 dimensions (0–63 and 256–319)
rotate on global layers. `freq_scale` is 1 and no YaRN terms apply
(`ext_factor 0`).

## Tokenizer

`tokenizer.ggml.model = "gemma4"`, no `tokenizer.ggml.pre` key, 262,144
tokens with scores (−1000 on control tokens) and 514,906 merges;
`add_bos_token` true, `add_space_prefix` false. Token types: 261,865
normal, 256 byte (`<0x00>`…`<0xFF>`, ids 238–493), 16 control, 7
user-defined. The reference treats it as **SPM-style BPE**
(`LLAMA_VOCAB_TYPE_BPE`, `LLAMA_VOCAB_PRE_TYPE_GEMMA4`):

- no word pre-splitting except the regex `[^\n]+|[\n]+` (runs of newlines
  are separated from everything else; a newline run that is itself a token,
  such as `\n` 107, `\n\n` 108, `\n\n\n` 109, is emitted whole);
- spaces are escaped to `▁` (U+2581) before merging (`escape_whitespaces`),
  so vocabulary entries look like `▁hello` (29104) and `▁` (236743);
  decoding unescapes them;
- merges run over raw UTF-8 code points, not the GPT-2 byte alphabet
  (`byte_encode = false`); a symbol with no token falls back to its bytes as
  `<0xXX>` tokens;
- the reference forces `add_bos` on for this pre-type; BOS is `<bos>` 2.

Special tokens: `<pad>` 0, `<eos>` 1, `<bos>` 2, `<unk>` 3, `<mask>` 4,
`<|tool>` 46, `<tool|>` 47, `<|tool_call>` 48, `<tool_call|>` 49,
`<|tool_response>` 50, `<tool_response|>` 51, `<|"|>` 52, `<|think|>` 98,
`<|channel>` 100, `<channel|>` 101, `<|turn>` 105, `<turn|>` 106; image and
audio markers at 255999–258884. The seven user-defined ones (tool call and
response markers, the channel markers, `<|"|>`) are rendered as text by the
reference so a chat parser can read them.

The tree's `tokenizer/vocabulary.zig` accepted only `model = "gpt2"` and
`tokenizer/pre.zig` the `qwen35` splitter until MODL-05 added Gemma's
vocabulary model and splitter (below); the stop set is `<turn|>` 106 and
`<eos>` 1 (the QAT file names 1 as EOS and 106 as EOT), owned by the
profile since MODL-07.

## Chat template and profile (MODL-07, 2026-09-12)

Turns are `<|turn>role\n…<turn|>\n`; roles `system`, `user`, `model`,
`tool`. The template emits `bos_token` first; a system turn exists when the
first message is system/developer, when tools are given, or when
`enable_thinking` is set, in which case the system turn starts with
`<|think|>\n`. Assistant reasoning is rendered as
`<|channel>thought\n…<channel|>` and only for turns after the last user
turn (or with `preserve_thinking` on tool-call turns); earlier turns are
stripped of thinking with `strip_thinking`. Tool calls use
`<|tool_call>…<tool_call|>` and responses `<|tool_response>…<tool_response|>`.
The template text is extracted from the file at capture time, never
embedded in the tree (18,922 bytes ending in a newline, the digest above).
Finetunes converted with the earlier 17,530-byte revision (digest
`dc311bb0…`) render history differently and are refused; they run under
`--prompt-profile gemma4`
([prompt-profile.md § Evidence](prompt-profile.md#evidence-and-reproduction)).

`inference/src/profiles/gemma4.zig` implements the text and tool-calling
subset (system/developer, user, assistant, tool declarations, calls, and
responses; no images or prefill) and is registered as
`profiles.Profile.gemma4`, selected by the template digest. The tool path is
in [prompt-profile.md § Gemma 4](prompt-profile.md#gemma-4-gemma4) and
[tool-calling.md](tool-calling.md).
What the reference's own rendering established, each with a fixture case
(`inference/src/profiles/fixtures/gemma4-text.json`, 14 prompts and 20
token strings captured from `llama-server 7620399` on the K-quant file,
[prompt-profile.md](prompt-profile.md#gemma-4-gemma4)):

- The server's `/apply-template` output has no `<bos>` (its tokenizer adds
  BOS); the profile renders `<bos>` as text because the tree's encoder
  never adds it (MODL-05). Token streams are identical: `[2, 105, …]`.
- `developer` arrives at the template as `system`; later system messages
  are `<|turn>system\n…<turn|>\n` turns of their own, empty ones included.
- Thinking is a switch: `off` pre-closes an empty channel in the
  generation prompt, any other effort adds `<|think|>` and lets the model
  open its channel. Reasoning never appears in history under this API.
- Two assistant messages in a row continue one `model` turn.

**Stop set and reasoning markers.** Generation ends on `<turn|>` (106) or
`<eos>` (1), resolved from the profile in the vocabulary at load
(`Engine.stop_ids`); the K-quant file names 106 as EOS and the QAT file 1,
both carry both. The chat splits the turn on `<|channel>thought\n` …
`<channel|>` (both user-defined tokens, always decoded as text). The engine
no longer assumes Qwen's "EOS or BOS" pair or the chat `</think>`: both
were latent Qwen assumptions outside the profile, found by the second
model ([llm-guide § 7](../llm-guide.md#7-the-prompt-profile-is-a-contract)).

**Sampling.** The profile's defaults are the file's own hint,
`general.sampling.temp` 1.0 / `top_p` 0.95 / `top_k` 64, the same in both
modes, no penalties. Recorded as the file's claim (no per-mode table was
pinned from a model card). `generate` and `chat` use the file's profile
even when the configuration guessed another from a bare path.

**Catalogue.** Two entries, one per quantization (2026-09-12):
`gemma-4-12b` is the K-quant file (commit `fc034cff…`, SHA-256
`90fd944d…`) with its companions `mmproj-BF16.gguf` (175,115,840 B,
`2e269f90…`) and `mtp-gemma-4-12b-it.gguf` (465,109,248 B, `145db909…`),
and `gemma-4-12b-qat` is the QAT file
([§ Q4_0 path](#q4_0-path-and-the-qat-file-modl-08-2026-09-12)). Between MODL-07
and that day one name pointed at whichever file was newest, which made
`nuclis model pull gemma-4-12b` mean different bytes in different weeks.
A registry entry of the same name still shadows either.

**Exercised.** `nuclis generate --model gemma-4-12b` in both modes (greedy:
a one-sentence answer ending on 106 with thinking off; a thought channel
with thinking on), `tokenize` (17 tokens for `Hello` at `--think medium`,
ids as the fixture), and a two-turn `nuclis agent` session under a
pseudo-terminal (`expect`): thought folded as "Thought for 29.1s", answers
144 then 145, the second turn a prefix extension of the first (ctx 87 →
16x tokens), Ctrl-D exit. The acceptance record against the reference
harness is in [bench.md](bench.md#gemma-4-12b-acceptance-record-modl-07-2026-09-12).

## Reference oracle status

The pinned llama.cpp `7620399` loads and runs both files on Metal
(`llama-completion -ngl 99 -no-cnv`, 2026-09-11): the K-quant file
continues a raw prompt in a repetition loop and the QAT file emits digits,
which is what an instruction-tuned model does without its template; with
`--jinja --single-turn` both files answer a chat prompt with a
`<|channel>thought` block (the template enables thinking by default in the
reference's default arguments). First-look rates from those runs, not a
benchmark record (default context, 6-token prompt): K-quant prompt 18.8 /
decode 25.2 tok/s, QAT 26.9 / 31.5 tok/s. The trace harness
`scripts/reference-generation.cpp` compiles against this build unchanged
(it names layers by the reference's `l_out-N` tensors); it tokenizes
without adding BOS, so Gemma traces pass `<bos>` in the prompt text
(`parse_special` is on).

## CPU reference against the oracle (MODL-05, 2026-09-11)

`inference/src/models/gemma4.zig` binds the pinned directory (the fixture
and the real file: 667 tensors, 7,350,597,824 bytes, 40 sliding and 8
global layers) and `gemma4_runtime.zig` executes the forward pass above on
the CPU. Compared against the reference's per-layer outputs on
`<bos>Hello,` (tokens `[2, 9259, 236764]`, three positions, the traces
pinned under `tests/fixtures/gemma4-hello-comma/`; since MODL-08 that target
is `make gate NAME='gemma4-trace-f*'`):

| Comparison | Measured | Threshold |
| --- | --- | --- |
| 144 layer files, max absolute | 8.0e-5 (layer 43, position 2) | 2e-3 |
| 144 layer files, max relative RMS | 8.3e-6 (layer 41, position 2) | 1e-4 |
| Logits, max absolute / relative RMS | 1.8e-4 / 5.3e-6 | 2e-3 / 1e-4 |
| Greedy token | 45518 (`thought`) on both sides; top-5 identical | equal |

Layer 0 at position 0 agrees to 3.8e-6 absolute, so the embedding scale,
the raw-weight norms, and the unscaled attention are the reference's; the
error grows slowly with depth and position as expected of F32 summation
order. One CPU step is 33.7 s for the three-token prompt (ReleaseSafe, M4
Pro), a reference number for the Metal plan of MODL-06, not a benchmark.

Recipe (the harness compiled as in
[generation.md](generation.md#numerical-traces)): `reference-generation
<file> '<bos>Hello,' <dir>` (the harness does not add BOS; `parse_special`
is on, so the marker in the text supplies it), then `nuclis generate
--backend cpu --prompt-tokens <ids.json> --max-tokens 1 --ctx-size 8
--logits … --trace-dir …`, then `compare-generation.py --positions 3
--embedding 3840 --layers 48 --vocab 262144`.

**Shared math reused, and what was added.** `cpu.rmsNorm`, `cpu.matvec`
and the quantized row decoders, `cpu.attention.apply` (the sliding window
is a contiguous slice of the cache rows: the last 1024 positions), and
`cpu.rope.apply` (already split-half) are the Qwen reference's; two
additions have their own fixtures: `cpu.gelu` (the tanh form) and the
`factors` option on RoPE (per-pair divisors, the 1e30 entries leave a pair
in place within F32). The session layout is per layer (2048-wide rows on
sliding layers, 512 on global ones). The Metal plan is
[§ Metal plan](#metal-plan-modl-06-2026-09-11).

**Tokenizer (implemented in MODL-05).** `tokenizer/vocabulary.zig` accepts
`tokenizer.ggml.model = "gemma4"` (`pre` is implied), `bpe.encodeSpmBudget`
runs the shared merge scan over code points with `<0xNN>` byte fallback,
`Encoder.encodeSpm` is the newline-run splitter with U+2581 escaping, and
`bpe.decode` unescapes, rebuilds bytes, and always renders the seven
user-defined markers. Checked against the reference server's `/tokenize`
on the real vocabulary for twelve strings (mixed punctuation and newlines,
leading and doubled spaces, four-newline runs, `émoji 😀 € 中文字`, the chat
markers `<bos>`/`<|turn>`/`<turn|>` inside text, tabs and CRLF, indented
code, digits, a literal `▁`, two spaces, one newline, one letter):
identical ids on all twelve. `nuclis tokenize --raw` and `generate --raw`
never add BOS; write `<bos>` in the text, as the traces do. Decoding does
not apply the reference's `clean_spaces` heuristics (neither does the Qwen
path).

## Metal plan (MODL-06, 2026-09-11)

`inference/src/models/gemma4_metal.zig` records the forward pass above
as `Backend` encoder calls, one command buffer per token (`step`) or per
prompt chunk (`prefill`), the same shape as the Qwen plan. What the
schedule needed from the backend, and what it reused unchanged
([metal-backend.md](metal-backend.md)):

| Operation | Kernel | Status |
| --- | --- | --- |
| Projections, FFN, tied output head | `nu_matvec_*`, `nu_matmul_*`, `nu_matvec_segments` | reused; every Gemma matrix meets the alignment and row/column rules |
| Residual and per-head norms (raw weights) | `nu_rmsnorm` | reused; the weightless value norm binds a row of ones |
| Embedding scale √3840, per-layer output scale, logit soft-cap | `nu_scale`, `nu_add_scale`, `nu_softcap` | **new** scalar epilogues; `tanh` clamped at ±20 (Metal's overflows past ±44, the CPU's saturates) |
| Tanh-GELU gate | `nu_gelu_mul`; pair mode 2 of `nu_matvec_segments` | **new**, matches `cpu.gelu` to 2e-6 relative |
| RoPE (split-half over the whole head; base 1e4, or 1e6 with `rope_freqs`) | `nu_rope`, `nu_rope_rows` | reused; `Backend.ropeTable` gained the per-pair `factors` (two tables per plan: 128 and 256 pairs per position) |
| Decode attention, sliding layers (16 over 8, width 256) | `nu_attention_decode` / `_h` | reused; the window is a cache-row slice (`firstVisible`), as in the CPU reference |
| Decode attention, global layers (16 over 1, width 512) | `nu_attention_decode_w` / `_wh` | **new instantiation** of the same template (4 heads × 16 channels per lane; a KV head's 16 query heads take four threadgroups per split) |
| Prefill attention, sliding layers | `nu_attention_chunk` / `_h` with `window = 1024` | **extended**: per-row window mask, key loop starts at the first visible tile, −∞ guard for rows with no visible key in a tile |
| Prefill attention, global layers | same, value width 512 | **extended**: one threadgroup per 256 value columns, scores recomputed per split |

Every cache is allocated for the full session capacity, sliding layers
included (the CPU reference does the same): at 32,768 tokens the F16
session is 11.3 GB (40 × 2 × 2,048 + 8 × 2 × 512 halves per position). A
ring layout for the 40 windowed layers would cut that to about 0.35 GB and
is a session-layout change of its own.

**Against the pinned traces** (`make gate NAME='gemma4-trace-f*'`,
three positions of `<bos>Hello,`, 145 files, 2026-09-11):

| Path | Max absolute | Max relative RMS | Threshold | Greedy / top-5 |
| --- | --- | --- | --- | --- |
| CPU reference | 1.8e-4 | 8.3e-6 | 2e-3 / 1e-4 | 45518, identical |
| Metal, F32 cache | 3.4e-4 (logits; layers below) | 1.5e-5 | 2e-3 / 1e-4 | 45518, identical |
| Metal, F16 cache | 0.73 (logits), 0.18 (layers) | 3.2e-2 | 1.0 / 5e-2 (its own) | 45518, identical |

The F16 tolerance is the model's, not the kernels'. Gemma's attention
scores are unscaled (`scale = 1`) over 256- and 512-wide heads, so a
2⁻¹¹ relative rounding of a key moves a score by up to a few hundredths
and a softmax weight by percents; on a short prompt with a BOS sink the
attention is peaked and the effect compounds through 48 layers. Two
independent checks pin this down: `test-metal` runs the same half
kernels on the Gemma geometry against the CPU over the *rounded*
operands (decode 2.4e-7, chunk 1.9e-4), and the CPU reference itself
with its keys and values rounded to F16 before the cache write deviates
from the F32 traces by 0.75 / 3.3e-2, the same as the Metal F16 plan
(0.73 / 3.2e-2); Metal F16 against that rounded CPU run is 0.030 /
1.3e-3 (rounding-boundary flips, amplified the same way). On the
70-random-token prompt of `generation-check` the F16 cache is 2.7e-2 /
1.1e-3, so the sensitivity is prompt-dependent. `--kv f32` is available
for numerical work; the default stays `f16` as for Qwen.

**Generation check** (`make gate NAME=gemma4-generation-metal`, 2026-09-11):
sessions bit-identical, cancellation and reset, snapshot/restore bit-exact
(688,128 bytes at position 1); chunked prefill vs per-token steps on 70
tokens: chunks of 64 / 48 / 32 at 8.6e-2 / 8.6e-2 / 2.6e-2 max abs and
2.5e-3 / 2.6e-3 / 7.5e-4 relative RMS (recorded bounds 1e-1 / 4e-3; Qwen's
are 2e-2 / 1e-3), same greedy token. Forcing the generic F32 tiles brings
the chunked run to 4.8e-4 / 1.2e-5, so the gap is the half-operand
rounding of the specialized matmul tiles (ENGN-05) on Gemma's larger
activations, not the chunk schedule (window mask, wide heads, the
value-as-key copy).

**First-look rates** (`nuclis bench`, Metal, greedy, context 2,048, three
measured runs; Apple M4 Pro 48 GB, macOS 26.6.2, Zig 0.16.0 ReleaseSafe,
2026-09-11; not the acceptance record, which MODL-07 takes against the
reference harness):

| Workload | Prefill tok/s | Decode tok/s | First token |
| --- | ---: | ---: | ---: |
| 9-token raw prompt, 64 out, `--kv f16` | 36.8 | 21.2 | 245 ms |
| same, `--kv f32` | 36.8 | 21.2 | 245 ms |
| 512-token array (`tests/fixtures/run-2026-09-06/prompt-512.json`, Qwen's ids as an opaque array), 128 out, `--kv f16` | 199.7 | 20.25 | 2,564 ms |
| same, `--kv f32` | 199.0 | 20.20 | 2,572 ms |
| Reference `llama-bench` `7620399` on the same file (`-p 512 -n 128 -ngl 99 -fa 1 -ctk f16 -ctv f16 -r 3`) | 219.9 ± 2.4 | 24.9 ± 0.15 | — |

Decode is 81 % of the reference (7.35 GB of weights at 20.2 tok/s is
148 GB/s effective) and prefill 91 %; both are follow-ups after MODL-07, not
part of MODL-06. The Qwen `make bench` is unchanged by the shared-kernel
changes (40.27 / 10.80 tok/s the same day).

## Q4_0 path and the QAT file (MODL-08, 2026-09-12)

The catalogue entry `gemma-4-12b` was decided on the quantization-aware-
trained file ([artifacts](#artifacts)): every one of its 329 weight
matrices is Q4_0, an encoding the parser stored and the CPU decoder
refused until this sub-unit. Q4_0 is IQ4_NL's block (an F16 scale `d`,
then sixteen bytes whose low nibbles are values 0–15 and high nibbles
16–31) with the code itself as the value: `d · (q − 8)`, no table
([quantization.md](quantization.md#equations-and-evidence)). What the
tree gained, each pinned by the reference's own fixture:

- `quant.row` id 2 and the eight-block Q4_0 row in
  `quant/fixtures/simple.json` from the pinned C decoder
  (`scripts/quant-fixtures.py`; the same payload bytes as the IQ4_NL row,
  so the two decoders pin each other's nibble order).
- `dequant.metal` `nu_dequant_q4_0` for the generic kernels
  (`nu_matvec`, `nu_embed`, the generic tile), bit-identical to the CPU
  decoder (`test-metal` checks the embedding row exactly).
- `nu_matvec_q4_0`: eight 18-byte blocks make the 256-value stride the
  specialized matvecs walk, so lane `g` of an octet takes block `8·kb + g`
  as IQ4_XS's lanes take a group; two `packed_ushort4` loads (blocks are
  2-byte aligned) give four words, `nu_low_nibbles`/`nu_high_nibbles`
  the codes, and the bias folds into the input sum as Q6_K's does:
  `Σ d·(q−8)·x = d·(Σq·x − 8·Σx)`, exact for a one-hot input. Selected
  by `specializedMatvec` when the row offset and stride are even (the
  body walks 32-value blocks, so any whole-block row serves), also inside
  `nu_matvec_segments`.
- `nu_tile_q4_0` and the `nu_matmul_q4_0` / `_32` instantiations: the
  tile template's segment addressing learned that a block can hold two
  segments rather than sixteen; the decode is the generic decoder's
  expression in the same operation order, so the F32 view is
  bit-identical and only the half rounding remains
  ([metal-backend.md § Prefill in chunks](metal-backend.md#prefill-in-chunks-engn-02)).
- `gemma4.executableEncoding` lists id 2; Qwen's does not (its executable
  set is that adapter's claim about the files it binds, not the kernels'
  capability). `model inspect` on the QAT file: `supported`.

**Against its own pinned traces** (`tests/fixtures/gemma4-qat-hello-comma/`,
the reference harness on the QAT file, `<bos>Hello,`, three positions,
145 files; `make gate NAME='gemma4-trace-f*'`, 2026-09-12):

| Path | Max absolute | Max relative RMS | Threshold | Greedy / top-5 |
| --- | --- | --- | --- | --- |
| CPU reference | 4.4e-5 | 2.6e-6 | 2e-3 / 1e-4 | 107 (`<|channel>`), identical |
| Metal, F32 cache | 7.7e-5 | 3.2e-6 | 2e-3 / 1e-4 | 107, identical |
| Metal, F16 cache | 1.3e-2 | 6.2e-4 | 1.0 / 5e-2 (the family's) | 107, identical |

Tighter than the K-quant file on every path (1.8e-4 / 3.4e-4 / 0.73),
and the F16 cache in particular: the key-rounding sensitivity measured on
the K-quant file is prompt- and checkpoint-dependent, and the tolerance
stays the family's recorded one. The K-quant comparisons are unchanged
and run as `make gate NAME='gemma4-trace-*'` (one make target per entry until
the gate registry of 2026-09-21).

**Generation check** (`make gate NAME=gemma4-qat-generation-metal` on the QAT
file, 2026-09-12): sessions bit-identical, cancellation and reset,
snapshot/restore bit-exact (688,128 bytes at position 1). Chunked prefill
against per-token steps on the 70 random tokens: chunks of 64 / 48 / 32
at 4.4e-1 / 4.2e-1 / 1.5e-1 max abs and 1.4e-2 / 1.3e-2 / 4.7e-3
relative RMS, same greedy token; through the generic F32 tiles 7.8e-4 /
3.2e-5 (bound 5e-3 / 2e-4, unchanged), so the schedule and the Q4_0
decode are exact and the gap is the half-operand rounding of the
specialized tiles (ENGN-05), which this checkpoint amplifies about five times
more than the K-quant file (8.6e-2 / 2.6e-2 at chunks 64 / 32 there).
The family's chunk bound moved from 1e-1 / 4e-3 to 6e-1 / 2e-2 to cover
both files, recorded in `generation-check.zig`; the F16 cache on this
prompt is 8.2e-2 / 3.1e-3 stepped and 2.2e-1 / 7.1e-3 chunked (bound
2.0 / 6e-2). A prefill tile that keeps Gemma's activations in F32 is a
performance-versus-precision choice, not a correctness gap.

**Catalogue and pull.** `gemma-4-12b` names the QAT file (commit
`980b060c…`, SHA-256 `90fd44e2…`, 6,716,356,800 B) with `mmproj-BF16.gguf`
(175,115,840 B, `dcb8103a…`) and `mtp-gemma-4-12B-it.gguf` (253,708,800 B,
`fcb35dea…`); `nuclis model pull gemma-4-12b --all` verified all three
against the pinned digests on 2026-09-12 (3.8 s, existing files reused),
and `model ls` lists the K-quant directory in its second group. The
profile fixtures apply unchanged (same template digest). The acceptance
record on the QAT file is in
[bench.md](bench.md#gemma-4-12b-acceptance-record-qat-file-modl-08-2026-09-12).

## Gemma 4 26B-A4B: the expert configuration (MODL-09)

Read on 2026-09-18 from the pulled file with `scripts/gguf-inventory.py`
(the fixture `inference/src/models/fixtures/gemma4-26b-a4b.json`) and from
the pinned reference's `src/models/gemma4.cpp` and the `build_moe_ffn`
helper of `src/llama-graph.cpp`.

**Artifact.** `unsloth/gemma-4-26B-A4B-it-qat-GGUF` at commit
`7b92b5b28818151e8669af2e45e88d6086f490dd`,
`gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf`, 14,249,047,104 B, SHA-256
`a7c5bc71…` ([artifacts.md](artifacts.md)); `general.name` `Gemma-4 26B-A4B
IT (smart Q4_0, QAT-lossless)`; 658 tensors, F32 392 and Q4_0 266 (every
matrix, the expert tensors included). The same vocabulary (262,144), the
same chat template digest (`845f1ee4…`), and the same sampling hint as
the 12B files; `tokenizer.ggml.add_bos_token` is `false` here (the 12B's
is `true`), which changes nothing: the encoder never adds BOS and the
profile writes it. The catalogue entry `gemma-4-26b-a4b` and its
companions (`mmproj-BF16.gguf`, `MTP/mtp-gemma-4-26B-A4B-it-Q4_0.gguf`)
were pinned ahead of this unit (MODL-14).

**Metadata against the 12B.** Shared and identical: `context_length`,
`attention.head_count` 16, `sliding_window` 1024, `key_length` /
`value_length` 512 and the `_swa` pair 256, `rope.dimension_count` 512 /
`_swa` 256, both `freq_base`s, the epsilon, the soft-cap, `shared_kv_layers`
0, `embedding_length_per_layer_input` 0. Different:

| Key | 12B | 26B-A4B |
| --- | --- | --- |
| `block_count` | 48 | 30 |
| `embedding_length` | 3840 | 2816 |
| `feed_forward_length` | 15360 | 2112 (the shared dense FFN) |
| `attention.head_count_kv` | 8 / **1** | 8 / **2** (global layers) |
| `expert_count` / `expert_used_count` / `expert_feed_forward_length` | absent | 128 / 8 / 704 |

The layer pattern is the same period 6 (`kindOf`): layers 5, 11, 17, 23,
29 are global, the other 25 sliding. `rope_freqs.weight` [256] is present
as on the 12B, and the global layers' `rope.dimension_count` is 512 with
the same factors, so the global RoPE is the 12B's (128 of 512 dimensions
rotate). The adapter selects the configuration by `block_count`
(`gemma4.configs`), validates every key against it, and treats the three
expert keys as known only to the expert configuration.

**Tensors.** Non-block as the 12B at width 2816. Per block, the 12B's set
(global layers: `attn_k` [2816, 1024] = 2 × 512, no `attn_v`,
`attn_output` [8192, 2816]) plus the expert block on **every** layer:

| Tensor | Shape (GGUF order) | Encoding |
| --- | --- | --- |
| `ffn_gate_inp.weight` | [2816, 128] | F32 |
| `ffn_gate_inp.scale` | [2816] | F32 |
| `ffn_gate_up_exps.weight` | [2816, 1408, 128] | Q4_0 |
| `ffn_down_exps.weight` | [704, 2816, 128] | Q4_0 |
| `ffn_down_exps.scale` | [128] | F32 |
| `post_ffw_norm_1.weight`, `pre_ffw_norm_2.weight`, `post_ffw_norm_2.weight` | [2816] | F32 |

The 3-D tensors are `[experts][rows][columns]` with the experts
contiguous (`cpu.ExpertMatrix`; `weights.View.expertMatrix`); the fused
gate-up rows are the gate rows then the up rows of one expert.

**The expert layer's FFN (from the reference graph).** With `a` the
residual after the attention add (`attn_out`):

1. Shared branch: `m = post_ffw_norm_1(Wdown · (gelu(Wgate · ffn_norm(a)) ⊙ (Wup · ffn_norm(a))))`,
   the 12B's FFN followed by its own post norm.
2. Router: `t = rms(a) · (1 / sqrt(2816)) ⊙ ffn_gate_inp.scale` — the
   *unweighted* RMS norm of the residual, scaled in that order — and
   `logits = ffn_gate_inp · t` (128). Softmax, the 8 largest, the selected
   probabilities renormalized to sum one with the sum clamped below at the
   smallest F16 normal (`norm_w`; `cpu.experts.route`).
3. Experts: over `pre_ffw_norm_2(a)`, each selected expert `e` computes
   `scale[e] · Wdown[e] · (gelu(gate) ⊙ up)` with `(gate, up)` the halves
   of `Wgate_up[e]` · input, and the outputs sum weighted by the routing
   weights (`cpu.experts.ffn`); then `x = post_ffw_norm_2(Σ)`.
4. `f = post_ffw_norm(m + x)`; `h = (f + a) · layer_output_scale`, as the 12B.

Everything else in the layer is the 12B's forward pass above.

**Reference oracle status.** The pinned llama.cpp `7620399` runs the file
on Metal; the trace harness captured `<bos>Hello,` (tokens `[2, 9259,
236764]`) on 2026-09-18 with greedy token 29104 (` hello`); the traces
are pinned under `tests/fixtures/gemma4-26b-a4b-hello-comma/` (90 layer
files of 2,816 floats and the 262,144 logits).

**CPU reference against the oracle (session 1, 2026-09-18).**
`gemma4_runtime.zig` runs both configurations from one schedule; the
expert layer is `Runtime.feedForward`. `make gate NAME=gemma4-26b-a4b-trace-cpu`
(`--positions 3 --embedding 2816 --layers 30`):

| Comparison | Measured | Threshold |
| --- | --- | --- |
| 90 layer files, max absolute | 4.6e-5 (layer 10, position 2) | 2e-3 |
| 90 layer files, max relative RMS | 1.8e-6 (layer 28, position 2) | 1e-4 |
| Logits, max absolute / relative RMS | 5.8e-5 / 3.1e-6 | 2e-3 / 1e-4 |
| Greedy token and top-5 | 29104, 26352, 1852, 8349, 144673 on both sides, logits equal to four decimals | equal |

Layer 0 at position 0 agrees to 4.0e-5, and the error does not grow
through the 30 expert layers, so the router input, the selection, the
renormalized weights, the per-expert down scale, and the three extra
norms are the reference's. Greedy continuation on the CPU: ` hello!
*waves`; a step is about 3 s (ReleaseSafe, M4 Pro; the 12B's is 11 s:
a token touches 8 experts of 704 and a 2,112-wide shared FFN instead of
a 15,360-wide FFN). `nuclis validate` reports the binding
(`gemma4_26b_a4b`, 25 sliding and 5 global layers, 658 tensors).

**Metal plan (session 2, 2026-09-18).** `gemma4_metal.zig` runs both
configurations from one schedule: `feedForward` (decode) and
`feedForwardChunk` (prefill) mirror the CPU reference's `feedForward`,
and the expert branch is the gathered kernels of
[metal-backend.md § Gathered expert kernels](metal-backend.md#gathered-expert-kernels-kern-09):

| Operation | Decode (`step`) | Prefill (`prefill`, per chunk of `count` rows) |
| --- | --- | --- |
| Shared branch post norm | `rmsNorm` with `post_ffw_norm_1` | same over `count` rows |
| Router input `rms(a) · (1/√2816) ⊙ scale` | `rmsNorm` with `ffn_gate_inp.scale` as its weight, then `scale` by 1/√2816 (the reference multiplies in the other order: one F32 rounding apart) | same over `count` rows |
| Router logits (F32 128 × 2,816) | generic `nu_matvec` | generic F32 `nu_matmul` tile |
| Selection | `route` (1 row) | `route` over `count` rows, then `expertLists` |
| Expert input | `rmsNorm` with `pre_ffw_norm_2` | same |
| Gate-up (Q4_0, 1,408 × 2,816 per expert) | `matvecExperts`, shared input | `matmulExperts`, `in_group` 8 |
| Gate | `geluMulRows` over 8 slot rows | over `count · 8` rows |
| Down (Q4_0, 2,816 × 704 per expert) | `matvecExperts`, one hidden row per slot | `matmulExperts`, `in_group` 1 |
| Weighted sum with the per-expert down scale | `combineExperts` | `combineExperts`, `rows = count` |
| Expert post norm, add to the shared branch, ordinary post norm | `rmsNorm` with `post_ffw_norm_2`, `add`, `rmsNorm` | same |
| Global attention, 16 heads of 512 over **two** KV heads | `nu_attention_decode_w` / `_wh` (two head groups per KV head) | `nu_attention_chunk` / `_h` |

The three expert tensors are wrapped whole and stay resident (14.2 GB of
weights, no paging); a token reads its eight experts' bytes. The
decode workspace is 8 slot rows; the chunk workspace holds `chunk · 8`
slot rows of 1,408, 704, and 2,816 floats plus the routing buffers
(40 MB at 256 tokens). The wide decode kernel and the chunk kernel index
the KV head generically; `test-metal` now runs the 16-over-2, width-512
geometry through both (F32 3.0e-6, F16 over rounded operands 1.9e-4;
decode 2.4e-7). The session at 32,768 tokens is 7.4 GB in F16
(25 × 2 × 2,048 + 5 × 2 × 1,024 halves per position).

**Against the pinned traces** (`make gate NAME='gemma4-26b-a4b-trace-f*'`, three
positions of `<bos>Hello,`, 91 files, 2026-09-18):

| Path | Max absolute | Max relative RMS | Threshold | Greedy / top-5 |
| --- | --- | --- | --- | --- |
| CPU reference | 5.8e-5 | 3.1e-6 | 2e-3 / 1e-4 | 29104; 26352, 1852, 8349, 144673 |
| Metal, F32 cache | 7.7e-5 | 3.1e-6 | 2e-3 / 1e-4 | identical |
| Metal, F16 cache | 1.9e-2 | 1.0e-3 | 1.0 / 5e-2 (the family's) | identical |

The F16 cache is far less sensitive here than on the 12B (0.73 / 3.2e-2):
30 layers instead of 48, and two global KV heads instead of one.

**Generation check** (`make gate NAME=gemma4-26b-a4b-generation-metal`,
2026-09-18): sessions bit-identical, cancellation and reset,
snapshot/restore bit-exact (450,560 bytes at position 1). Chunked prefill
vs per-token steps on the 70-random-token prompt: chunks of 64 / 48 / 32
at 1.11e1 / 1.08e1 / 1.10e1 max abs and 3.46e-1 / 3.40e-1 / 3.45e-1
relative RMS, the same greedy token in every case; the F16 cache stepped
at 6.8e-1 / 2.4e-2 and chunked at 1.0e1 / 3.4e-1. Through the generic F32
tiles the chunked run is **2.8e-4 / 1.2e-5** (bound 5e-3 / 2e-4
unchanged), so the chunk schedule is exact and the gap is the
specialized tiles' half-operand rounding amplified by the discrete
routing: a perturbed router logit swaps a token's eighth expert. With
the expert projections alone through per-token F32 matvecs the gap is
still 1.1e-1 relative RMS, because the dense tiles' rounding already
moves the router. The check records the expert configuration's own
bounds (2e1 / 5e-1) beside the 12B's. The acceptance record
([bench.md](bench.md#gemma-4-26b-a4b-acceptance-record-modl-10-2026-09-18))
ran the reference's arrays greedy to the token budget at every length,
which is what a rate record can say about it; per-token agreement on
real prompts stays the trace comparison's job.

**First-look rates** (`nuclis bench`, Metal, greedy, `--kv f16`, three
measured runs; Apple M4 Pro 48 GB, macOS 26.6.2, Zig 0.16.0 ReleaseSafe,
2026-09-18; the acceptance record against the reference harness is in
[bench.md](bench.md#gemma-4-26b-a4b-acceptance-record-modl-10-2026-09-18)):

| Workload | Chunk | Prefill tok/s | Decode tok/s | First token |
| --- | ---: | ---: | ---: | ---: |
| 22-token text prompt, 64 out, context 2,048 | 256 | 122.6 | 57.9 | 179 ms |
| 512-token array (`tests/fixtures/run-2026-09-06/prompt-512.json`), 128 out, context 2,048 | 256 | 463.3 | 54.1 | 1,105 ms |
| same | **512** | 525.9 | 55.3 | 974 ms |
| same | 1,024 | 515.8 | 54.4 | 993 ms |
| 4,096-token array (`prompt-4096.json`), 32 out, context 4,608 | 256 | 329.0 | 48.5 | 12,452 ms |
| same | **512** | 361.0 | 49.6 | 11,346 ms |
| same | 1,024 | 382.5 | 49.9 | 10,709 ms |

**The chunk for this family is 512** (`Plan.preferredChunk`; the engine's
default stays 256 for the dense configurations): 10–14 % more prefill
than 256 because a chunk's 4,096 slot rows fill the gathered 32-row
tiles better ([metal-backend.md](metal-backend.md#gathered-expert-kernels-kern-09):
69 % against 50 %), for 80 MB of expert workspace and 0.15 GB of chunk
activations; 1,024 buys 6 % more only on long prompts for twice that
again and a coarser cancellation grain (about a second per chunk), and
stays a follow-up beside the 64-token expert tile. Decode is 55 tok/s
against the 12B's 20.2 and the Qwen3.8-27B's 10.7 on the same machine:
a token touches eight experts of 704 and a 2,112-wide shared FFN instead
of a 15,360-wide FFN, about 2.2 GB of Q4_0 weights with the tied head
(an estimate from the tensor shapes, not a measurement), so the
effective rate is roughly 120 GB/s against the 12B's 148. The per-kernel
profile ranked the follow-ups (the expert down projection at 114 GB/s on
its 704-wide rows, then the launch-bound norms, then the wide
flash-decoding kernel at long context;
[bench.md](bench.md#gemma-4-26b-a4b-acceptance-record-modl-10-2026-09-18)).
The Qwen `make bench` is unchanged the same day (40.05 / 10.67 tok/s
against 40.27 / 10.80).

**Acceptance record and agent check (MODL-10, 2026-09-18).** Against the
reference harness on its own arrays: prefill / decode 500.42 / 55.29 at
512, 346.17 / 49.44 at 4K, 206.44 / 39.55 at 16K, 134.74 / 31.30 at 32,639
tok/s, the reference at 580.76 / 68.02, 548.41 / 60.82, 459.46 / 50.67,
343.38 / 44.09; session 6.87 GiB, peak footprint 8.0 GB
([bench.md](bench.md#gemma-4-26b-a4b-acceptance-record-modl-10-2026-09-18)).
`nuclis agent --model gemma-4-26b-a4b` (Metal, context 8,192, `--think
medium`, `--print --json`) on "create greeting.txt with hello world, then
read it back" issued `write_file` then `read_file` and answered from the
contents, the file on disk `hello world`: 221 prompt tokens, 71 generated,
prefill 0.99 s, decode 1.30 s (18–19 ms per token in the loop, against
the 12B's 8.5 s for 168 tokens under the same check), `replayed: true`
as on the 12B. The catalogue entry's verdict is *supported* (`model
inspect`: the Hub's digest at the pinned commit and the adapter's
binding).
