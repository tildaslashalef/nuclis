# Qwen3-TTS 12Hz CustomVoice (0.6B and 1.7B)

[Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)
(Qwen, Apache-2.0: the card's `license: apache-2.0` and the package's
`LICENSE`) is a text-to-speech model. In nuclis it is the voice of a local
voice assistant, served later by `POST /v1/audio/speech`. It has three
networks, in two files:

- the **talker**, a 28-layer Qwen3-style decoder over a codec vocabulary,
  which predicts one code per 80 ms frame (code group 0);
- the **code predictor**, a 5-layer decoder run 15 steps per frame, which
  predicts code groups 1–15 of that frame;
- the **codec decoder** (`speech_tokenizer/`), which turns frames of 16
  codes into 24 kHz mono PCM, 1,920 samples per frame (12.5 frames/s).

The *CustomVoice* variant speaks with one of nine built-in speakers chosen
by a codec-vocabulary id. The *Base* variant has no speakers: it clones a
voice from a reference clip through two more networks, a speaker encoder
and the codec's **encoder**
([Voice cloning (the Base)](#voice-cloning-the-base)).

Everything here was read on 2026-10-27 from these sources: the pinned
files' safetensors headers and JSON files (read over HTTP at the pinned
revision); Qwen's package `QwenLM/Qwen3-TTS` at
`022e286b98fbec7e1e916cb940cdf532cd9f488e`, unpacked at
`.reference/qwen3-tts-022e286/` (`qwen_tts/core/models/modeling_qwen3_tts.py`,
`configuration_qwen3_tts.py`,
`qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py` and its
configuration, `qwen_tts/inference/qwen3_tts_model.py`,
`qwen3_tts_tokenizer.py`). The package pins `transformers==4.57.3`
(`pyproject.toml`); where behaviour comes from `transformers` rather than
the package (generation's logits processors, the sliding-window mask), the
sentence says so; those were read in `.reference/transformers-c0d024f/`
and an installed `transformers` 5.x, not in 4.57.3 itself. The package is
read as the definition, never copied.

**Contents.**

- [Artifacts](#artifacts)
- [Configuration](#configuration)
- [Tensors](#tensors)
- [Forward pass (from `modeling_qwen3_tts.py`)](#forward-pass-from-modeling_qwen3_ttspy)
- [The codec decoder (from `modeling_qwen3_tts_tokenizer_v2.py`)](#the-codec-decoder-from-modeling_qwen3_tts_tokenizer_v2py)
- [Tokenizer and the rendered text](#tokenizer-and-the-rendered-text)
- [The 1.7B CustomVoice](#the-17b-customvoice)
- [Voice cloning (the Base)](#voice-cloning-the-base)
- [Codec decoder against the trace (2026-10-27)](#codec-decoder-against-the-trace-2026-10-27)
- [Talker and code predictor against the trace (2026-10-27)](#talker-and-code-predictor-against-the-trace-2026-10-27)
- [On Metal (2026-10-27)](#on-metal-2026-10-27)
- [Open for the implementation](#open-for-the-implementation)

## Artifacts

Repository `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`, revision
`85e237c12c027371202489a0ec509ded67b5e4b5`. Sizes from the Hub's tree at
that revision; the two safetensors sizes equal 8 + header + data from
their headers.

| File | Size (B) | SHA-256 |
| --- | ---: | --- |
| `model.safetensors` (talker + code predictor) | 1,811,626,576 | `bc3c7e785eb961179c25450d1acff03f839e0002f2f3a5aeb67b5735c0fa2adb` |
| `config.json` | 4,908 | `81aca2b6fac304944d8acf345272d8a9a727d5fc2e2e66b222ab4729340c7455` |
| `generation_config.json` | 245 | `f1b90b4513f3b34c62851049e2492d7b4c5940daf1276f89c82b8ef04127f3aa` |
| `tokenizer_config.json` | 7,344 | `dc3c31c3bdaedd5016382bb3cbe07323026775ad51f5a4fb564505992ae4a670` |
| `vocab.json` | 2,776,833 | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |
| `merges.txt` | 1,671,839 | `599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3` |
| `preprocessor_config.json` | 127 | `efdde1022ea9d76928bf7a9cd53139138f5ba2e466e837f08f6105ab1af1c119` |
| `speech_tokenizer/model.safetensors` (codec) | 682,293,092 | `836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258` |
| `speech_tokenizer/config.json` | 2,336 | `ee65bb901c876664ab8707c487157aa1a6ee57c65969b28fb5ec9dc211e68167` |
| `speech_tokenizer/preprocessor_config.json` | 234 | `fcb3805e597e786d4067706e602f6688524640f8d3396790e2e09b5942fcbdfb` |
| `speech_tokenizer/configuration.json` | 76 | `6bc26d64eb5024b4d1dab5a52371958b429256d6c9d59787f1f5294a54e0cebd` |

`speech_tokenizer/configuration.json` is a ModelScope descriptor
(`framework`, `task`), read by nothing in the package. The weights run as
published: the talker BF16, the codec F32 ([spec § 3](../spec.md#3-decisions): never requantized or expanded).

## Configuration

**Top level** (`config.json`): `tts_model_type` `custom_voice`,
`tts_model_size` `0b6`, `tokenizer_type` `qwen3_tts_tokenizer_12hz`;
text-vocabulary ids `im_start_token_id` 151644, `im_end_token_id` 151645,
`assistant_token_id` 77091, `tts_pad_token_id` 151671,
`tts_bos_token_id` 151672, `tts_eos_token_id` 151673. There is no
`speaker_encoder_config`, and `Qwen3TTSForConditionalGeneration.__init__`
builds no speaker encoder unless `tts_model_type` is `base`.

**Talker** (`talker_config`):

| Key | Value |
| --- | --- |
| `num_hidden_layers` | 28 |
| `hidden_size` | 1024 |
| `intermediate_size` | 3072, `hidden_act` `silu` |
| `num_attention_heads` / `num_key_value_heads` / `head_dim` | 16 / 8 / 128 (q width 2048 > hidden) |
| `rms_norm_eps` | 1e-6 |
| `rope_theta` | 1,000,000 |
| `rope_scaling` | `interleaved` true, `mrope_section` [24, 20, 20], `rope_type` `default` |
| `position_id_per_seconds` | 13 (read by no code in the package; see below) |
| `max_position_embeddings` | 32768 |
| `use_sliding_window` / `sliding_window` | false / null: full causal attention |
| `vocab_size` | 3072 (the codec vocabulary) |
| `text_vocab_size` / `text_hidden_size` | 151936 / 2048 |
| `num_code_groups` | 16 |
| `codec_pad_id` / `codec_bos_id` / `codec_eos_token_id` | 2148 / 2149 / 2150 |
| `codec_think_id` / `codec_nothink_id` | 2154 / 2155 |
| `codec_think_bos_id` / `codec_think_eos_id` | 2156 / 2157 |

`codec_language_id`: english 2050, german 2053, spanish 2054, chinese 2055,
japanese 2058, french 2061, sichuan_dialect 2062, korean 2064, russian 2069,
italian 2070, portuguese 2071, beijing_dialect 2074. `spk_id`: aiden 2861,
sohee 2864, ono_anna 2873, eric 2875, dylan 2878, uncle_fu 3010, ryan 3061,
vivian 3065, serena 3066. `spk_is_dialect`: eric `sichuan_dialect`, dylan
`beijing_dialect`, the rest false.

**Code predictor** (`talker_config.code_predictor_config`): 5 layers,
`hidden_size` 1024, `intermediate_size` 3072 SiLU, 16 / 8 heads of 128,
`rms_norm_eps` 1e-6, `rope_theta` 1e6, `rope_scaling` null (plain 1-D
RoPE), every `layer_types` entry `full_attention`, `vocab_size` 2048,
`num_code_groups` 16, `repetition_penalty` 1.0, `top_k` 50, `top_p` 1.0,
`temperature` 1.0 (the last three are overridden at generation, below).

**Generation** (`generation_config.json`, loaded by
`Qwen3TTSForConditionalGeneration.from_pretrained` as `generate_config`
and merged by `Qwen3TTSModel._merge_generate_kwargs`): talker `do_sample`
true, `temperature` 0.9, `top_k` 50, `top_p` 1.0, `repetition_penalty`
1.05; predictor (`subtalker_*`) `dosample` true, `temperature` 0.9, `top_k`
50, `top_p` 1.0; `max_new_tokens` 8192. The fallbacks when the file is
absent: `_merge_generate_kwargs` hard defaults (the same, but
`max_new_tokens` 2048) and `Qwen3TTSForConditionalGeneration.generate`'s
signature (4096).

nuclis takes the file as written. A key it leaves out keeps the
fallback; `do_sample` (or `subtalker_dosample`) false makes that network
greedy (temperature 0) while the other keeps sampling, and the repetition
penalty and the processors of [§ (c)](#c-the-talkers-head-group-0) still
apply, as in transformers' greedy search. Fine-tunes ship such files: the
Egyptian Arabic 1.7B `itshamdi404/Egy_Arabic_Qwen3-TTS-12Hz-1.7B-Base` (a
`custom_voice` checkpoint despite its name, one speaker,
`egyptian_speaker`) has `{"do_sample": false, "max_new_tokens": 256}`, a
greedy talker and at most about 20 s per request.

**Codec decoder** (`speech_tokenizer/config.json` `decoder_config`):
`num_quantizers` 16, `num_semantic_quantizers` 1, `codebook_size` 2048,
`codebook_dim` 512, `latent_dim` 1024, `hidden_size` 512,
`intermediate_size` 1024 SiLU, 8 layers, 16 heads of 64 (MHA,
`num_key_value_heads` 16), `rms_norm_eps` 1e-5, `rope_theta` 10000,
`sliding_window` 72, `layer_scale_initial_scale` 0.01, `decoder_dim` 1536,
`upsample_rates` [8, 5, 4, 3], `upsampling_ratios` [2, 2]. Top level:
`output_sample_rate` 24000, `decode_upsample_rate` 1920.
`semantic_codebook_size` is 4096 in the file, but **no decoder code reads
it**: `Qwen3TTSTokenizerV2Decoder` builds every codebook with
`bins=codebook_size` (2048), and the file's semantic codebook is
[2048, 256]. `max_position_embeddings` 8000 only bounds RoPE caching.

## Tensors

Shapes are as stored, PyTorch order: a `Linear` is [out, in], a `Conv1d`
[out, in/groups, k], a `ConvTranspose1d` [in, out, k].

### `model.safetensors`: 402 tensors, all BF16

`__metadata__` `{"format": "pt"}`; data 1,811,577,344 B.

| Tensor | Shape | Bytes |
| --- | --- | ---: |
| `talker.model.text_embedding.weight` | [151936, 2048] | 622,329,856 |
| `talker.text_projection.linear_fc1.weight` / `.bias` | [2048, 2048] / [2048] | 8,392,704 |
| `talker.text_projection.linear_fc2.weight` / `.bias` | [1024, 2048] / [1024] | 4,196,352 |
| `talker.model.codec_embedding.weight` | [3072, 1024] | 6,291,456 |
| `talker.model.norm.weight` | [1024] | 2,048 |
| `talker.codec_head.weight` | [3072, 1024] | 6,291,456 |
| `talker.code_predictor.model.codec_embedding.{0..14}.weight` | [2048, 1024] each | 62,914,560 |
| `talker.code_predictor.lm_head.{0..14}.weight` | [2048, 1024] each | 62,914,560 |
| `talker.code_predictor.model.norm.weight` | [1024] | 2,048 |

Per layer, 11 tensors, the same in the talker (`talker.model.layers.{0..27}.`,
880,932,864 B in all) and the predictor
(`talker.code_predictor.model.layers.{0..4}.`, 157,309,440 B):

| Tensor | Shape |
| --- | --- |
| `input_layernorm.weight`, `post_attention_layernorm.weight` | [1024] |
| `self_attn.q_proj.weight` | [2048, 1024] = 16 × 128 |
| `self_attn.k_proj.weight`, `self_attn.v_proj.weight` | [1024, 1024] = 8 × 128 |
| `self_attn.o_proj.weight` | [1024, 2048] |
| `self_attn.q_norm.weight`, `self_attn.k_norm.weight` | [128] |
| `mlp.gate_proj.weight`, `mlp.up_proj.weight` | [3072, 1024] |
| `mlp.down_proj.weight` | [1024, 3072] |

No attention biases (`attention_bias` false). There is no
`small_to_mtp_projection` tensor: the predictor's width equals the
talker's, so `Qwen3TTSTalkerCodePredictorModelForConditionalGeneration`
makes it an identity. No speaker-encoder tensors. 308 + 8 + 86 = 402.

### `speech_tokenizer/model.safetensors`: 496 tensors, all F32

No `__metadata__`; data 682,229,764 B. **Encoder: 225 tensors,
224,937,216 B under `encoder.`** (`Qwen3TTSTokenizerV2Encoder`, a
`MimiModel`); not read further. **Decoder: 271 tensors, 457,292,548 B
under `decoder.`:**

| Group | Tensors | Shapes |
| --- | --- | --- |
| `quantizer.rvq_first.vq.layers.0._codebook.{embedding_sum, cluster_usage}` | 2 | [2048, 256], [2048] |
| `quantizer.rvq_rest.vq.layers.{0..14}._codebook.{embedding_sum, cluster_usage}` | 30 | [2048, 256], [2048] |
| `quantizer.rvq_{first,rest}.output_proj.weight` | 2 | [512, 256, 1] |
| `quantizer.rvq_{first,rest}.input_proj.weight` (encode side; unused by decode) | 2 | [256, 512, 1] |
| `pre_conv.conv.{weight, bias}` | 2 | [1024, 512, 3], [1024] |
| `pre_transformer.input_proj.{weight, bias}` | 2 | [512, 1024], [512] |
| `pre_transformer.output_proj.{weight, bias}` | 2 | [1024, 512], [1024] |
| `pre_transformer.norm.weight` | 1 | [512] |
| `upsample.{0,1}.0.conv.{weight, bias}` (transposed) | 4 | [1024, 1024, 2], [1024] |
| `upsample.{0,1}.1.dwconv.conv.{weight, bias}` | 4 | [1024, 1, 7], [1024] |
| `upsample.{0,1}.1.norm.{weight, bias}`, `.gamma` | 6 | [1024] |
| `upsample.{0,1}.1.pwconv1.{weight, bias}` | 4 | [4096, 1024], [4096] |
| `upsample.{0,1}.1.pwconv2.{weight, bias}` | 4 | [1024, 4096], [1024] |
| `decoder.0.conv.{weight, bias}` | 2 | [1536, 1024, 7], [1536] |
| `decoder.{1..4}.block.0.{alpha, beta}` | 8 | [C_in] |
| `decoder.{1..4}.block.1.conv.{weight, bias}` (transposed) | 8 | [C_in, C_out, 2r], [C_out] |
| `decoder.{1..4}.block.{2,3,4}.act{1,2}.{alpha, beta}` | 48 | [C_out] |
| `decoder.{1..4}.block.{2,3,4}.conv1.conv.{weight, bias}` | 24 | [C_out, C_out, 7], [C_out] |
| `decoder.{1..4}.block.{2,3,4}.conv2.conv.{weight, bias}` | 24 | [C_out, C_out, 1], [C_out] |
| `decoder.5.{alpha, beta}` | 2 | [96] |
| `decoder.6.conv.{weight, bias}` | 2 | [1, 96, 7], [1] |

with (C_in, C_out, r) = (1536, 768, 8), (768, 384, 5), (384, 192, 4),
(192, 96, 3) for blocks 1–4 (transposed kernels 16, 10, 8, 6). Per
`pre_transformer.layers.{0..7}.` (11 tensors): `input_layernorm.weight`,
`post_attention_layernorm.weight` [512]; `self_attn.{q,k,v}_proj.weight`
[1024, 512]; `self_attn.o_proj.weight` [512, 1024];
`mlp.{gate,up}_proj.weight` [1024, 512]; `mlp.down_proj.weight`
[512, 1024]; `self_attn_layer_scale.scale`, `mlp_layer_scale.scale` [512].
No attention biases, no Q/K norms.

**Values read from the file.** `cluster_usage` lies in 0.0217–0.578
(semantic) and 0.0358–5.15 (acoustic), so the decoder's `clamp(min=1e-5)`
never binds.

## Forward pass (from `modeling_qwen3_tts.py`)

Notation: `rms_w(x)` = x / √(mean(x²) + ε) · w with the weight **raw**
(`Qwen3TTSRMSNorm`: `weight * x`, initialized to ones), computed in F32.
`T(t)` = `text_embedding[t]` (2048). `P(t)` = `linear_fc2 · silu(linear_fc1
· T(t) + b1) + b2` (1024; `Qwen3TTSTalkerResizeMLP` with
`hidden_act` silu and biases); it depends on the token only.
`C(c)` = `talker.model.codec_embedding[c]`. `E_g(c)` =
`code_predictor.model.codec_embedding.(g−1)[c]` and `H_g` =
`code_predictor.lm_head.(g−1)`, g = 1…15.

### (a) The prompt

`Qwen3TTSModel.generate_custom_voice` tokenizes
`<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n`
(`_build_assistant_text`) to ids `x`. The code slices it by index: `x[0:3]`
is the role (`<|im_start|>` 151644, `assistant` 77091, `\n` 198), `x[3:-5]`
the text t₁…t_N, and the last 5 the closing `<|im_end|>\n<|im_start|>assistant\n`,
which is never embedded. `Qwen3TTSForConditionalGeneration.generate` then
builds the talker's input rows. Every row is one 1024-wide vector, **the
sum of a text-track term and a codec-track term** where both exist
("dual track"); there is no concatenation.

With a language and a speaker (here `english`, `ryan`), N text tokens:

| Row | Text track | Codec track |
| --- | --- | --- |
| 0–2 | P(151644), P(77091), P(198) | none |
| 3 | P(tts_pad 151671) | C(think 2154) |
| 4 | P(tts_pad) | C(think_bos 2156) |
| 5 | P(tts_pad) | C(language 2050) |
| 6 | P(tts_pad) | C(think_eos 2157) |
| 7 | P(tts_pad) | C(speaker 3061) |
| 8 | P(tts_bos 151672) | C(codec_pad 2148) |

then, by `non_streaming_mode`:

- **Non-streaming** (`generate_custom_voice`'s default, `True`): rows
  9…8+N are P(tᵢ) + C(codec_pad); row 9+N is P(tts_eos 151673) +
  C(codec_pad); row 10+N is P(tts_pad) + C(codec_bos 2149). Length N + 11.
  The trailing text track is the single row P(tts_pad).
- **Streaming text** (`generate`'s own default, `False`): row 9 is P(t₁) +
  C(codec_bos). Length 10. The trailing text track is P(t₂), …, P(t_N),
  P(tts_eos), N rows.

Variants, from `generate`:
- Language `auto`: the codec prefix is three ids, nothink 2155, think_bos,
  think_eos (one row fewer, rows 3–5), still followed by speaker, then
  tts_bos + codec_pad.
- Speakers `eric` and `dylan` with language `chinese` or `auto` replace the
  language id by their dialect's (2062, 2074), so the think form is used.
- No speaker (an empty string): the speaker row is absent.
- **Instruction.** `generate_custom_voice` sets `instruct = None` when
  `tts_model_size` is `0b6`: **the 0.6B model takes no instruction**. Were
  `instruct_ids` passed to `generate` directly, the rows
  P(`<|im_start|>user\n{instruct}<|im_end|>\n`) would be prepended before
  row 0, text track only.
- Batches are left-padded with zero rows and an attention mask; with one
  request there is no padding.

### (b) One talker layer

Qwen3's layer (`Qwen3TTSTalkerDecoderLayer`, `Qwen3TTSTalkerAttention`):

- `h = rms_w(x, input_layernorm)`; `q, k, v = Wq h, Wk h, Wv h`, 16 query
  heads and 8 KV heads of 128.
- `q = rms_w(q, q_norm)`, `k = rms_w(k, k_norm)` per head (ε 1e-6).
- RoPE on q and k (below). Causal attention over all earlier rows, scale
  1/√128, grouped query (2 query heads per KV head). Then `Wo`.
- `x = x + attn`; `x = x + Wdown(silu(Wgate h') ⊙ Wup h')` with
  `h' = rms_w(x, post_attention_layernorm)`.

After 28 layers, `h = rms_w(x, talker.model.norm)`.

**RoPE: it reduces to 1-D NEOX RoPE.** `Qwen3TTSTalkerRotaryEmbedding`
builds 64 frequencies `θ^(−2i/128)` (the `default` init over `head_dim`
128) for each of three position components;
`apply_multimodal_rotary_pos_emb` with `interleaved` true takes frequency
i from component 1 (H) when i ≡ 1 (mod 3) and i < 60, from component 2
(W) when i ≡ 2 (mod 3) and i < 60, else from component 0 (T): 24, 20, 20
frequencies. The halves are then duplicated and applied by `rotate_half`
(element i pairs with i + 64). The positions come from
`Qwen3TTSTalkerForConditionalGeneration.get_rope_index`: `cumsum(mask) −
1`, **the same for all three components**; at decode, `cache_position +
rope_deltas`, again expanded to three equal rows. With one unpadded
request `rope_deltas` is 0, so the prompt has positions 0…L−1 and the
frame fed at decode step k has position L + k: one position per frame,
not per code. With equal components every frequency sees the same
position, so this is plain NEOX RoPE, base 1e6, over the whole head.
**`position_id_per_seconds` (13) is read nowhere** in the package
(`grep` over `qwen_tts/`): it has no effect on inference.

### (c) The talker's head: group 0

`logits = codec_head · h` (3072) at the last row. `generate` passes the
talker to `transformers`' `generate` with `min_new_tokens` 2,
`eos_token_id` 2150, and `suppress_tokens` = 2048…3071 except 2150
(`range(vocab_size − 1024, vocab_size)`). So group 0 is a code 0…2047 or
the end. In `transformers`' processor order (`_get_logits_processor`):

1. repetition penalty 1.05 over the group-0 ids sampled so far: zᵢ/1.05
   if zᵢ > 0, else zᵢ·1.05. The prompt is embeddings only, so
   `transformers` starts `input_ids` empty
   (`_maybe_initialize_input_ids_for_generation`) and only generated
   codes count;
2. z₂₁₅₀ = −∞ while fewer than 2 codes have been generated;
3. the suppressed ids = −∞;
4. temperature 0.9, top-k 50, top-p 1.0 (no-op), then a sample from the
   softmax.

### (d) The code predictor: groups 1–15

For each frame whose group-0 code c₀ was just sampled from the talker
state h (the post-norm state of the row that produced c₀'s logits),
`Qwen3TTSTalkerForConditionalGeneration.forward` runs
`code_predictor.generate` on the sequence, with a **fresh KV cache per
frame**:

| Position | Input row | Head | Output |
| --- | --- | --- | --- |
| 0 | h | none | none |
| 1 | C(c₀) (the **talker's** codec table) | H₁ | c₁ |
| g (2…15) | E_{g−1}(c_{g−1}) | H_g | c_g |

The predictor's layers are the talker's with plain 1-D RoPE
(`Qwen3TTSRotaryEmbedding`, NEOX, base 1e6, positions 0…15), then
`rms_w(·, code_predictor.model.norm)`, then the head. The prefill computes
positions 0–1 together; `generation_steps` selects the head (H₁ after the
prefill, then H₂…H₁₅) and the input table. `E_15` is never a predictor
input; it is used only for the talker's next row. Sampling: temperature
0.9, top-k 50, top-p 1.0, no repetition penalty (the predictor config's
1.0), no end token, exactly 15 steps.

### (e) The next talker row

`forward` sums the frame's 16 code embeddings and adds the text track:

    row_k = C(c₀) + Σ_{g=1..15} E_g(c_g) + (trail[k] if k < len(trail) else P(tts_pad))

where k counts decode steps from 0 and `trail` is the trailing text track
from (a). The talker runs this row at position L + k; its state gives the
next frame's c₀ (and is the next predictor's h).

### (f) Stopping and length

Generation stops when c₀ = 2150 (`codec_eos_token_id`) or after
`max_new_tokens` talker samples (8192 from `generation_config.json`,
≈ 655 s of audio). The returned frames are those whose predictor ran:
the codes are cut at the first frame whose c₀ is 2150, and a final c₀
sampled at the length limit has no groups 1–15 and is dropped.
`min_new_tokens` 2 guarantees at least two frames.

### (g) Streaming

`non_streaming_mode=False` is the only "streaming" in the package, and it
streams the *text into the talker*, not audio out: the prompt carries t₁,
and each generated frame adds the next text token (one token per 80 ms
frame) until the text and `tts_eos` are spent, then `tts_pad` forever. All
text is known before generation starts; `generate_*`'s docstrings say the
option "only simulates streaming text input". Audio is not streamed:
`generate` returns all codes, then `speech_tokenizer.decode` decodes them
at once (below). An incremental text feed or streamed audio are nuclis's
own design, on top of these semantics.

## The codec decoder (from `modeling_qwen3_tts_tokenizer_v2.py`)

Input: codes [T, 16] (frame-major, as the talker emits them). Output:
T × 1920 samples at 24 kHz in [−1, 1]. `Qwen3TTSTokenizerV2Model.decode`
clamps codes to ≥ 0 (batch padding is −1) and calls
`Qwen3TTSTokenizerV2Decoder.chunked_decode`.

1. **Quantizer** (`SplitResidualVectorQuantizer.decode`). Each codebook
   vector is `embedding_sum[c] / max(cluster_usage[c], 1e-5)` (256 wide,
   `EuclideanCodebook.decode`; no per-codebook projection since
   `codebook_dim` is None there). Per frame:
   `q = Wout_first · cb₀(c₀) + Wout_rest · Σ_{g=1..15} cb_g(c_g)`, each
   `output_proj` a 1×1 conv 256 → 512 without bias. All 16 groups,
   including group 0, use 2048-entry codebooks.
2. **`pre_conv`**: causal conv 512 → 1024, kernel 3.
3. **`pre_transformer`** (`Qwen3TTSTokenizerV2DecoderTransformerModel`):
   `x = input_proj · y + b` (1024 → 512); 8 layers of
   - `a = Attn(rms_w(x, input_layernorm))`: 16 heads of 64, MHA, no Q/K
     norm, NEOX RoPE base 1e4 at positions 0…T−1 of the chunk, scale 1/8,
     causal with a sliding window: key j is visible from query i when
     i − 72 < j ≤ i, 72 keys including itself (`transformers`
     `masking_utils.sliding_window_overlay`, `kv_idx > q_idx −
     sliding_window`; every layer is `sliding_attention` by the config's
     `layer_types` property);
   - `x = x + s_attn ⊙ a` (`self_attn_layer_scale.scale`);
   - `x = x + s_mlp ⊙ Wdown(silu(Wgate h) ⊙ Wup h)` with
     `h = rms_w(x, post_attention_layernorm)`;

   then `rms_w(·, norm)` (ε 1e-5 throughout) and `output_proj` (512 → 1024,
   bias).
4. **`upsample`**, twice: a causal transposed conv 1024 → 1024 (kernel 2,
   stride 2), then a ConvNeXt block (`Qwen3TTSTokenizerV2ConvNeXtBlock`):
   `x + γ ⊙ pw2(gelu(pw1(LayerNorm(dwconv(x)))))`, with `dwconv` a causal
   depthwise conv of kernel 7, `LayerNorm` ε 1e-6 with weight and bias,
   `pw1` 1024 → 4096, `pw2` 4096 → 1024 (both biased), and `nn.GELU()`
   (exact erf form). T → 4T.
5. **`decoder`**: causal conv 1024 → 1536, kernel 7; four
   `Qwen3TTSTokenizerV2DecoderDecoderBlock`s, each SnakeBeta(C_in), a
   causal transposed conv C_in → C_out (kernel 2r, stride r), then three
   residual units with dilations 1, 3, 9:
   `x + conv2(snake₂(conv1(snake₁(x))))`, `conv1` kernel 7 dilated,
   `conv2` kernel 1; then SnakeBeta(96), a causal conv 96 → 1 of kernel
   7, and `clamp(−1, 1)`. Rates 8, 5, 4, 3: 4T → 1920T.

`SnakeBeta(x) = x + sin²(e^α x) / (e^β + 1e-9)` per channel: α and β are
stored as logarithms.

**Causal padding.** `Qwen3TTSTokenizerV2CausalConvNet` pads
(k − 1)·d zeros on the left; with stride 1 its `extra_padding` is 0, so
the length is kept. `Qwen3TTSTokenizerV2CausalTransConvNet` runs
`ConvTranspose1d` and trims `kernel − stride` samples on the right
(0 for kernel 2 stride 2; r for kernel 2r stride r), giving L·stride
outputs. Output n of a decoder block's transposed conv (kernel 2s) reads
inputs ⌊n/s⌋ and ⌊n/s⌋ − 1; an `upsample` one (kernel = stride) reads
⌊n/s⌋ only.
**The decoder is causal**: the samples of frame t depend only on frames
≤ t, so decoding a prefix and decoding the whole give the same samples for
the prefix, in exact arithmetic. Its reach back is long (8 layers of a
72-frame window, plus the convolutions).

**Chunking in the authors' decode.** `chunked_decode(codes,
chunk_size=300, left_context_size=25)` decodes frames [s, e) for s = 0,
300, 600, … (e = min(s + 300, T)) by running the whole decoder **from a
fresh state** (zero conv padding, RoPE positions from 0) on frames
[s − c, e) with c = 25 for s > 0 (0 for s = 0), and discards the first
c × 1920 samples. So the authors' audio for T ≤ 300 frames (24 s) is one
full causal decode; beyond that, each 300-frame chunk sees only 25 frames
of history and differs from a full decode near its start. A nuclis
decoder that streams frame by frame with carried state (per layer the 71
keys and values before the new row, since a query sees 72 including
itself; a left history per conv) reproduces the authors' decode up to
frame 300, to summation order; to keep reproducing it, it would reset at
each 300-frame boundary and re-warm on the previous 25 frames (it does
not: see the last section).

## Tokenizer and the rendered text

`tokenizer_config.json`: `tokenizer_class` `Qwen2Tokenizer` (the processor
accepts `Qwen2TokenizerFast`), byte-level BPE from `vocab.json` (151,643
entries, ids 0…151642) and `merges.txt` (151,387 lines, no `#version`
header). `add_bos_token` false, `add_prefix_space` false, no BOS, EOS
`<|im_end|>`, pad `<|endoftext|>`; `Qwen3TTSProcessor` adds nothing.
There is no `tokenizer.json`; `AutoProcessor` builds `Qwen2TokenizerFast`
from the slow files (NFC normalizer, then the split, then byte level).
**The split is not `Qwen2Tokenizer`'s pattern.** `from_pretrained` loads
the processor with `fix_mistral_regex=True`, and in the pinned
`transformers` 4.57.3 (`tokenization_utils_base.py`,
`_patch_mistral_regex`) a local checkpoint whose `config.json` names a
`transformers_version` after 4.57.2 and before 5.0 is treated as
Mistral-like whatever its `model_type`; this one names 4.57.3, so the
flag replaces the pattern with the case-seam one
(`[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|…`,
in full in `inference/src/tokenizer/cased.zig`). Checked 2026-10-27 by
printing the loaded processor's pre-tokenizer in the pinned venv: the
authors' run splits `McDonald's iPhone` as `Mc`, `Donald`, `'s`, `i`,
`Phone`, where `Qwen2Tokenizer`'s pattern keeps `McDonald` and `iPhone`
whole. (In `transformers` 5.x the check returns before replacing, so the
same files tokenize differently there.) nuclis follows the authors' run:
[tokenizer.md § Vocabulary and merges files](../engine/tokenizer.md#vocabulary-and-merges-files).
The text embedding has 151,936 rows, more than the vocabulary.

Added tokens (151643…151675) include `<|endoftext|>` 151643,
`<|im_start|>` 151644, `<|im_end|>` 151645, `<tts_pad>` 151671,
`<tts_text_bos>` 151672, `<tts_text_eod>` 151673 (the config's
`tts_eos_token_id`), `<tts_text_bos_single>` 151674, all special; the
Qwen2 tool and think markers (151657…151668) are not special. Ids checked
in `vocab.json`: `assistant` 77091, `Ċ` (`\n`) 198, `ĊĊ` 271, `user` 872.

The rendered texts (`Qwen3TTSModel`):

    _build_assistant_text: <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
    _build_instruct_text:  <|im_start|>user\n{instruct}<|im_end|>\n
    _build_ref_text:       <|im_start|>assistant\n{text}<|im_end|>\n   (cloning only)

The index slicing in (a) assumes the role is exactly 3 tokens and the tail
exactly 5. A text that begins with a newline breaks the first assumption
(`\n\n` is one token, 271), which the authors' code does not check.

## The 1.7B CustomVoice

`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` at
`0c0e3051f131929182e2c023b9537f8b1c68adfe`, pulled 2026-10-27:

| File | Size (B) | SHA-256 |
| --- | ---: | --- |
| `model.safetensors` | 3,833,402,552 | `38b1d5971bdbd982b561cccec982669a53b0537c3cf5e9bd4778ed07bb2f5137` |
| `speech_tokenizer/model.safetensors` | 682,293,092 | `836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258` |

The codec file is byte-identical to the 0.6B's (`cmp`). It is the only
size that performs an instruction: `generate_custom_voice` in
`qwen3_tts_model.py` drops `instruct` when `tts_model_size` is `0b6`, and
the authors' README marks the 0.6B CustomVoice without instruction
control.

What differs, from the two headers (404 tensors against 402, all BF16) and
`config.json` (`tts_model_size` `1b7`):

- **Talker width 2048, FFN 6144**, still 28 layers, 16 query and 8 KV
  heads of 128: `q_proj` [2048, 2048], `k_proj`/`v_proj` [1024, 2048],
  `o_proj` [2048, 2048], gate/up [6144, 2048], down [2048, 6144], norms
  [2048]; `codec_embedding` [3072, 2048], `codec_head` [3072, 2048];
  `text_projection.linear_fc2` [2048, 2048] (the text track projects to
  2048, not 1024).
- **The code predictor is unchanged** (width 1024, 5 layers, FFN 3072)
  except that its 15 codec tables are talker-width, [2048, 2048], and a
  new `small_to_mtp_projection` (Linear 2048 → 1024 with bias) maps
  **every** predictor input row into its width: the talker's final hidden
  state, the group-0 code through the talker's table, and group g − 1
  through predictor table g − 2 (`Qwen3TTSTalkerCodePredictorModelForConditionalGeneration`,
  both its prefill and step paths). On the 0.6B the projection is the
  identity and absent from the file.
- The prompt layout, speakers, languages, sampling defaults, and the
  codec are the 0.6B's; the prompt additionally carries the instruction
  (§ (a)).

## Voice cloning (the Base)

Read 2026-10-29: the two Base checkpoints' JSON files and safetensors
headers over HTTP at the pinned revisions; Qwen's package at the commit
above (`qwen3_tts_model.py` `create_voice_clone_prompt`,
`generate_voice_clone`; `modeling_qwen3_tts.py` the speaker encoder,
`mel_spectrogram`, `extract_speaker_embedding`, `generate_icl_prompt`,
`generate`; `qwen3_tts_tokenizer.py` `encode`;
`modeling_qwen3_tts_tokenizer_v2.py` `Qwen3TTSTokenizerV2Model.encode`);
and `transformers` v4.57.3, the version the package pins,
`models/mimi/modeling_mimi.py` and `masking_utils.py` for the codec
encoder, which is a `MimiModel`.

### Artifacts

`Qwen/Qwen3-TTS-12Hz-0.6B-Base` at
`5d83992436eae1d760afd27aff78a71d676296fc` and
`Qwen/Qwen3-TTS-12Hz-1.7B-Base` at
`fd4b254389122332181a7c3db7f27e918eec64e3`. The codec files, the
tokenizer files, `generation_config.json`, and `preprocessor_config.json`
are byte-identical between the two sizes and to the CustomVoice files;
only `model.safetensors` and `config.json` differ:

| File | Size (B) | SHA-256 |
| --- | ---: | --- |
| 0.6B `model.safetensors` | 1,829,344,272 | `180b3b10eb1c9f1b4db7806d5475bae3071c0243c299d49926bab1da3b6946f6` |
| 0.6B `config.json` | 4,494 | `2e714c787c8edb98b05432685cddb634add2de4d4e645f653d68251ef72ba011` |
| 1.7B `model.safetensors` | 3,857,413,744 | `38fc7fc51c5e776e840414b6fd443962e9411b9654888fd7913e4da643cb857c` |
| 1.7B `config.json` | 4,494 | `b4f01752d15a488abde3e1ab44723ae4f4b9e68a4037257b098b3737893cc1f9` |
| `speech_tokenizer/model.safetensors` | 682,293,092 | `836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258` |

The other eight files are the CustomVoice table's (same sizes and
hashes).

**Configuration.** `tts_model_type` `base`; `speaker_encoder_config`
`{enc_dim, sample_rate: 24000}` with `enc_dim` 1024 (0.6B) or 2048
(1.7B), the talker's width; every other speaker-encoder key takes the
package's default (`Qwen3TTSSpeakerEncoderConfig`: `mel_dim` 128,
`enc_channels` [512, 512, 512, 512, 1536], `enc_kernel_sizes`
[5, 3, 3, 3, 1], `enc_dilations` [1, 2, 3, 4, 1],
`enc_attention_channels` 128, `enc_res2net_scale` 8, `enc_se_channels`
128). `talker_config` is the CustomVoice's of the same size except
`spk_id` and `spk_is_dialect` empty and `codec_language_id` without the
two dialects (ten languages). `generation_config.json` is the
CustomVoice's (§ Configuration).

**Tensors.** `model.safetensors`, all BF16: the CustomVoice's tensors of
the same size, names, shapes, and dtypes equal (402 on the 0.6B, 404 on
the 1.7B), plus 76 under `speaker_encoder.` (8,854,336 parameters on the
0.6B, 12,001,088 on the 1.7B, the difference being `fc`):

| Group | Tensors | Shapes (weight, bias) |
| --- | --- | --- |
| `blocks.0.conv` | 2 | [512, 128, 5], [512] |
| `blocks.{1,2,3}.tdnn1.conv`, `.tdnn2.conv` | 12 | [512, 512, 1], [512] |
| `blocks.{1,2,3}.res2net_block.blocks.{0..6}.conv` | 42 | [64, 64, 3], [64] |
| `blocks.{1,2,3}.se_block.conv1` | 6 | [128, 512, 1], [128] |
| `blocks.{1,2,3}.se_block.conv2` | 6 | [512, 128, 1], [512] |
| `mfa.conv` | 2 | [1536, 1536, 1], [1536] |
| `asp.tdnn.conv` | 2 | [128, 4608, 1], [128] |
| `asp.conv` | 2 | [1536, 128, 1], [1536] |
| `fc` | 2 | [enc_dim, 3072, 1], [enc_dim] |

The codec file's 225 `encoder.*` tensors (224,937,216 B, F32) are listed
under [The codec encoder](#the-codec-encoder).

### The voice-clone item

`create_voice_clone_prompt(ref_audio, ref_text, x_vector_only_mode)`
returns a `VoiceClonePromptItem` per clip: `ref_spk_embedding` (enc_dim
floats), `ref_code` (T × 16 codes, absent in x-vector-only mode),
`ref_text`, and the mode. `generate_voice_clone(voice_clone_prompt=…)`
takes the item instead of the audio, so the encoders run once per clip.
`ref_text` is required unless `x_vector_only_mode`; that mode is ICL
(in-context) otherwise. The clip is read by `librosa.load(sr=None,
mono=True)` or `soundfile` (channels averaged), unnormalized; the speaker
encoder resamples it to 24 kHz with `librosa.resample`, and the codec's
`encode` does the same.

### The speaker encoder

`extract_speaker_embedding` (24 kHz only):

1. **Mel.** Reflect-pad (n_fft − hop) / 2 = 384 samples at each end;
   STFT with `n_fft` 1024, hop 256, a periodic Hann window of 1024,
   `center` false; magnitude `sqrt(re² + im² + 1e-9)`; librosa's Slaney
   mel filters (`htk` false, `norm` `slaney`), 128 bins over 0–12 kHz;
   `log(max(x, 1e-5))`. Frames: ⌊(L + 768 − 1024) / 256⌋ + 1.
2. **ECAPA-TDNN** (`Qwen3TTSSpeakerEncoder`). Every convolution is
   `padding="same"` with reflect padding, d·(k − 1) / 2 frames at each
   end (symmetric for every kernel here), and every TDNN is conv + ReLU:
   - `blocks.0`: TDNN 128 → 512, k 5.
   - `blocks.1..3` (SE-Res2Net, dilation 2, 3, 4): TDNN 1×1 →
     Res2Net (the 512 channels in 8 chunks of 64: chunk 0 passes;
     chunk 1 → `blocks.0`; chunk i ≥ 2 → `blocks.(i−1)` of chunk i plus
     the previous chunk's output; concatenated; each a TDNN k 3 at the
     block's dilation) → TDNN 1×1 → squeeze-excitation (the time mean →
     `conv1` → ReLU → `conv2` → sigmoid, scaling every channel) → plus
     the block's input.
   - `mfa`: the three blocks' outputs concatenated (1536) → TDNN 1×1.
   - `asp` (attentive statistics pooling, `eps` 1e-12): the time mean μ
     and standard deviation σ = √max(mean((x − μ)²), 1e-12) per channel;
     [x; μ; σ] (4608 channels, μ and σ repeated over time) → TDNN 1×1 to
     128 → tanh → `asp.conv` to 1536 → softmax over time per channel →
     the weighted mean and standard deviation → [μ_w; σ_w] (3072).
   - `fc`: 1×1 conv 3072 → enc_dim, with bias. The output is
     `ref_spk_embedding`.

It runs in the model's dtype (the mel is cast to it); the file is BF16.

### The codec encoder

`Qwen3TTSTokenizer.encode`: the `EncodecFeatureExtractor`
(`speech_tokenizer/preprocessor_config.json`: 24 kHz, right-padded with
zeros, no normalization, no chunking) then
`Qwen3TTSTokenizerV2Model.encode`, which runs `MimiModel.encode` (a
subclass without the decoder halves) and keeps the first
`encoder_valid_num_quantizers` = 16 of 32 quantizers, cut to
⌈samples / 1920⌉ frames (`encode_downsample_rate`). `encoder_config`
(a `MimiConfig`): `use_causal_conv` true, `pad_mode` `constant`,
`num_filters` 64, `upsampling_ratios` [8, 6, 5, 4] (applied reversed),
`num_residual_layers` 1, `residual_kernel_size` 3, `compress` 2,
`kernel_size` 7, `last_kernel_size` 3, `hidden_size` 512, 8 layers of
8 heads × 64, `intermediate_size` 2048, `hidden_act` `gelu` (erf),
`norm_eps` 1e-5, `rope_theta` 1e4, `sliding_window` 250, `num_quantizers`
32, `num_semantic_quantizers` 1, `codebook_size` 2048, `codebook_dim`
256, `_frame_rate` 12.5 against an encodec frame rate of 24000 / 960 = 25.

**Causal convolutions** (`MimiConv1d`): zeros on the left of
k_eff − stride (k_eff = (k − 1)·d + 1), and on the right as many as bring
the length to a whole number of strides, so a stride-s layer gives
⌈L / s⌉ frames. The downsample alone pads by replicating the edge
frames. ELU has α = 1.

| Stage (`encoder.encoder.layers.N`) | Shape |
| --- | --- |
| 0: conv 1 → 64, k 7 | [64, 1, 7] + bias |
| 1: residual (ELU, conv 64 → 32 k 3, ELU, conv 32 → 64 k 1, plus input) | `block.1` [32, 64, 3], `block.3` [64, 32, 1], biases |
| 2: ELU; 3: conv 64 → 128, k 8, stride 4 | [128, 64, 8] + bias |
| 4: residual at 128 (via 64) | [64, 128, 3], [128, 64, 1] |
| 5: ELU; 6: conv 128 → 256, k 10, stride 5 | [256, 128, 10] + bias |
| 7: residual at 256 (via 128) | [128, 256, 3], [256, 128, 1] |
| 8: ELU; 9: conv 256 → 512, k 12, stride 6 | [512, 256, 12] + bias |
| 10: residual at 512 (via 256) | [256, 512, 3], [512, 256, 1] |
| 11: ELU; 12: conv 512 → 1024, k 16, stride 8 | [1024, 512, 16] + bias |
| 13: ELU; 14: conv 1024 → 512, k 3 | [512, 1024, 3] + bias |

That is ⌈L / 960⌉ frames of 512 at 25 Hz. Then:

- **`encoder.encoder_transformer.layers.{0..7}`**, pre-norm:
  `input_layernorm` (LayerNorm with bias, eps 1e-5) → attention
  (`q,k,v,o_proj` [512, 512], no bias; NEOX RoPE, θ 1e4, over positions
  0…T−1; scale 1/8) → times `self_attn_layer_scale.scale` → residual;
  `post_attention_layernorm` → `mlp.fc1` [2048, 512] → GELU →
  `mlp.fc2` [512, 2048] → times `mlp_layer_scale.scale` → residual. No
  final norm. The mask is `create_causal_mask`, which under the default
  SDPA and eager attention is plain causal: `sliding_window` 250 is not
  applied (only the flash-attention class reads it). The two agree for
  clips up to 250 frames (10 s).
- **`encoder.downsample.conv`** [512, 512, 4], stride 2, no bias,
  replicate padding: ⌈L / 1920⌉ frames at 12.5 Hz.
- **Split RVQ.** The semantic quantizer: `input_proj` [256, 512, 1] (no
  bias), then the nearest of 2048 codewords by Euclidean distance, a
  codebook being `embed_sum / max(cluster_usage, 1e-5)`: code group 0.
  The acoustic quantizer applies its own `input_proj` to the **same**
  512-wide input (not the semantic residual) and quantizes 15 times
  residually (each stage subtracts its codeword): groups 1–15. Its layers
  15–30 are never used (layer 30's `initialized` is 0).

The encoder's quantizer tensors are bit-identical to the decoder's, which
nuclis already binds: `semantic_residual_vector_quantizer` to
`decoder.quantizer.rvq_first` and the acoustic to `rvq_rest`
(`embed_sum` = `embedding_sum`, `cluster_usage`, and both `input_proj`
weights, compared value for value). What the encoder adds is the SEANet
stack (28 tensors), the transformer (96), and the downsample (1); the
other 100 are the quantizer copies.

It runs in the codec model's dtype, the caller's choice; the file is F32.

### The prompt

Built in `Qwen3TTSForConditionalGeneration.generate`. Both cloning modes
put `ref_spk_embedding` itself as the speaker row, where CustomVoice puts
the codec table's row for an `spk_id`: the codec track's prefix is
`think, think_bos, language, think_eos` (or `nothink, think_bos,
think_eos` for `auto`), then the speaker embedding, then `pad, bos`.
There are no dialects (no speakers to carry one).

`generate_voice_clone` defaults to `non_streaming_mode=False`, where
`generate_custom_voice` and `generate_voice_design` default to true: the
text is summed into the rows the talker generates, one text row per
step, instead of all prefilled.

- **x-vector only.** The role rows (`<|im_start|>assistant\n`), then the
  prefix with the text track `tts_pad` and `tts_bos` on its last row
  (`pad`), then one row: the target's first text token plus `codec_bos`.
  The trailing text is the rest of the target text, then `tts_eos`; each
  generated step adds the next trailing row, and `tts_pad` once they run
  out.
- **ICL.** The role rows, then the prefix rows **without** the
  `codec_bos` row, then rows summing two tracks: the text track
  `ref_text ids ++ text ids ++ tts_eos` (the reference rendered by
  `_build_ref_text`, `<|im_start|>assistant\n{ref}<|im_end|>\n`, sliced
  `[3:-2]`; the target by `_build_assistant_text`, sliced `[3:-5]`), and
  the codec track `codec_bos ++` one row per reference frame, the sum of
  its 16 groups' embeddings (group 0 through the talker's codec table,
  group g through predictor table g − 1). If the text track is longer,
  the prompt takes its first `codec_lens` rows and the rest is the
  trailing text; otherwise it is padded with `tts_pad` to the codec
  track's length and the trailing text is `tts_pad` alone. For a typical
  10 s reference (126 codec rows) and a sentence or two of text, the
  whole text lies within the reference rows and generation sees only
  `tts_pad`. A reference frame's row is the row nuclis already computes
  for a generated frame (§ (e)).

Generation then runs as for CustomVoice (§ (c)–(f)), with the same
sampling defaults.

### The decode

`generate_voice_clone` prepends `ref_code` to the generated codes,
decodes the whole, and cuts the first `ref_len / total` of the samples,
which is exactly `ref_len × 1920` (every frame is 1920 samples). The
decoder is causal, so nuclis decodes the reference frames first, carries
their state, and discards their samples: the same samples, without the
authors' 300-frame chunking (§ Open for the implementation). In
x-vector-only mode nothing is prepended.

### The trace (2026-10-29)

`qwen3_tts_trace.py --clone` runs in the same venv (Python 3.12.15,
`torch` 2.14.1 CPU, `transformers` 4.57.3, `numpy` 2.5.4), F32, seed 0,
and adds three cases under `tests/fixtures/qwen3-tts/` (1.7 MB):

- **`0.6b-clone-reference`.** The clip: the 0.6B CustomVoice, speaker
  `vivian`, says "Every morning I walk along the river and listen to the
  birds." (4.6 s), written as 24 kHz 16-bit PCM (`reference.wav`) and
  read back from it, so nuclis's WAV reader (÷ 32768, as `soundfile`)
  sees the same samples. Then `create_voice_clone_prompt` on the 0.6B
  Base: the speaker encoder's input mel (the last 64 of 435 frames),
  `blocks.0`, `blocks.3`, `mfa` (last 8 rows each), the pooled statistics,
  the embedding; the codec encoder's SEANet, transformer, and downsample
  outputs (last 16 rows each) and the 58 frames of codes. The encoder ran
  under `sdpa`.
- **`0.6b-clone-xvector-fox`, `0.6b-clone-icl-fox`.**
  `generate_voice_clone` on that item, x-vector only and ICL, saying the
  `fox` sentence (39 and 40 frames). Each pins the prompt's layout
  (`[text id, codec id, reference frame]` per row, −2 for the speaker
  row) and the trailing text ids, both derived from the config by the
  rules above and checked by rebuilding the talker's input rows from them
  (max |difference| 3.6e-7 and 6e-8); the talker's input rows (prompt
  then 4 decode rows), layers 0, 13, 27 and the final norm (the last 4
  prompt rows and the decode rows), group-0 logits of frames 0–4, the
  predictor's logits of frames 0–1, the generated codes, and the samples
  after the reference's are cut (the first 2 frames and the last).

The ICL prompt is 68 rows: 3 role, 6 prefix, and 59 summed rows
(`codec_bos` and the 58 reference frames) that hold the whole text track
(13 reference and 10 target ids and `tts_eos`, then 35 `tts_pad`), so the
trailing text is `tts_pad` alone. An item keeps its transcript in either
mode, so `generate_voice_clone` tokenizes it after the text in x-vector
mode too, and does not use it. nuclis's transcriber reads back the
reference and both clones word for word.

## Codec decoder against the trace (2026-10-27)

**The implementation.** `inference/src/audio/qwen3_codec.zig`: `bind`
(the 271 `decoder.*` tensors by name and shape, the encode side's two
`input_proj`s included so that nothing in `decoder.*` is unknown; the
`encoder.*` ones skipped by name), `Runtime` (every tensor decoded to F32
at load, 457 MB, 0.15 s), and `Stream`. Each causal convolution is one
matrix over gathered input rows, tap-major (`[out][k · in]`), and each
transposed convolution one matrix `[stride · out][taps · in]` whose
product for input row m is output rows m·stride … m·stride + stride − 1,
the right trim implied. So every output element is a single dot product
whose order does not depend on how frames are grouped, and a stream fed
in pieces gives `decode`'s samples bit for bit. A stream carries the left
context of each convolution ((k − 1)·d rows: 2 for `pre_conv`, 6 for each
ConvNeXt depthwise conv, `decoder.0`, and the output conv, 6, 18, 54 for
the residual units' dilated convs; one input row for each decoder block's
transposed conv, none for `upsample`'s), the transformer's 71 previous
rows of keys (after RoPE) and values per layer, and the frame count as
the RoPE position. Attention scores and sums are F64, norms take F64
statistics; everything else is F32.

**The trace.** `scripts/speech-traces/qwen3_tts_trace.py` runs the
authors' package (`022e286b`) in `.reference/venv-qwen3-tts/` (Python
3.12.15, `torch` 2.14.1 CPU, `transformers` 4.57.3, `numpy` 2.5.4) on the
0.6B in F32 at seed 0, speaker `ryan`, language `english`, with the
codec's default `sdpa` attention, and pins its outputs under
`tests/fixtures/qwen3-tts/` (`index.json`: shapes, rows, digests,
revision, versions; 1.2 MB). Two cases, both one causal decode (under 300
frames):

- `fox`, "The quick brown fox jumps over the lazy dog.": 46 frames
  sampled; the codec decodes the first 24 and every stage boundary is
  pinned (the wide stages' last rows only).
- `window`, a two-sentence text: 171 frames sampled; the codec decodes the
  first 112, 40 past the transformer's window, and the transformer's last
  16 rows and the last two frames' samples are pinned.

In the authors' own decode the prefix's samples equal the whole clip's to
6.7e-7 and 9.8e-7: the decoder is causal, as read. The run reproduces
itself: two runs wrote identical files.

**Measured** (`make gate NAME=qwen3-tts-codec-cpu`, ReleaseFast, Apple M4
Pro). Relative RMS is ‖Δ‖ / ‖reference‖ over the rows compared:

| Case | Stage | Rows | max \|Δ\| | max \|ref\| | rel RMS |
| --- | --- | --- | ---: | ---: | ---: |
| fox | `quantized` | 0–24 | 3.43e-5 | 89.6 | 3.54e-7 |
| fox | `pre_conv` | 0–24 | 4.77e-6 | 5.00 | 7.52e-7 |
| fox | `transformer` | 0–24 | 1.27e-7 | 0.210 | 1.03e-6 |
| fox | `upsample.0` | 16–48 | 3.70e-6 | 2.22 | 1.61e-6 |
| fox | `upsample.1` | 64–96 | 2.68e-5 | 17.1 | 1.99e-6 |
| fox | `decoder.0` | 80–96 | 3.43e-5 | 11.4 | 1.84e-6 |
| fox | `decoder.1` | 736–768 | 8.94e-6 | 3.16 | 1.62e-6 |
| fox | `decoder.2` | 3,776–3,840 | 2.69e-5 | 3.28 | 3.16e-6 |
| fox | `decoder.3` | 15,232–15,360 | 1.57e-5 | 2.04 | 3.77e-6 |
| fox | `decoder.4` | 45,824–46,080 | 6.87e-5 | 67.0 | 8.59e-7 |
| fox | samples | 0–46,080 | 1.90e-6 | 0.379 | 2.40e-6 |
| window | `transformer` | 96–112 | 1.19e-7 | 0.133 | 9.94e-7 |
| window | samples | 211,200–215,040 | 6.15e-8 | 0.0193 | 2.61e-6 |

The gate's bounds, about five times the worst: relative RMS 2e-5 for
every stage and the samples, and 1e-5 for the samples' largest
difference. Both cases also stream in pieces of five frames to `decode`'s
samples bit for bit.

**The window is 72 keys.** Rebuilt with a 73-key window, the `window`
case's transformer rows differ by relative RMS 6.8e-3 and its samples by
3.2e-3, so the gate tells the two apart; with 72 they agree to 1e-6. The
mask read in a newer `transformers` holds under 4.57.3.

**Speed, not a target.** The CPU reference decodes at 3.5× real time
(24 frames in 549 ms, 112 in 2,561 ms); real-time streaming is the Metal
plan's.

## Talker and code predictor against the trace (2026-10-27)

**The implementation.** `inference/src/models/qwen3_tts.zig`: `parseConfig`
(`config.json` and `generation_config.json`; any `tts_model_type` but
`custom_voice` is `UnsupportedModelType`), `bind` (the 402 or 404 tensors
by name and shape, the 1.7B's two `small_to_mtp_projection` tensors
required exactly when the predictor is narrower than the talker; any
other tensor is `UnexpectedTensor`, named), `encodeText` (the rendered
texts of [§ Tokenizer](#tokenizer-and-the-rendered-text), sliced by
position as the authors slice them), and `Prompt.build` (§ (a),
non-streaming; an instruction on the 0.6B is `InstructionUnsupported`).
`inference/src/models/qwen3_tts_runtime.zig`: `Model` (every tensor but
the text embedding decoded to F32 at load: 2,378 MB for the 0.6B, 6,422 MB
for the 1.7B, against 1,811 and 3,833 MB as published; text rows are
decoded from the mapped BF16 file per id), `Talker` (28 layers over a
growing cache), `Predictor` (a fresh cache each frame, inputs through the
projection on the 1.7B), `Sampler` (§ (c), (d)), `Synthesizer` (prefill,
frame by frame until the end code or `max_new_tokens`, decoded through the
codec's `Stream`), and `Voice` (a checkpoint directory opened whole).
Projections are `dense.zig`'s F32 matmul; norms, RoPE, and attention take
F64 statistics. The tokenizer: [tokenizer.md § Vocabulary and merges files](../engine/tokenizer.md#vocabulary-and-merges-files).

**The sampler.** The processors and their order were read in the pinned
`transformers` 4.57.3 (`generation/utils.py`, `_get_logits_processor`):
repetition penalty, then `MinNewTokensLengthLogitsProcessor`, then
`SuppressTokensLogitsProcessor`, then temperature, top-k, top-p, as § (c)
says. nuclis applies the penalty to the codes 0…2047 it has generated
(the generated ids are the only `input_ids`), drops the suppressed ids
and, for the first two frames, the end code from the candidates instead
of setting them to −∞, and draws with `sampling.Sampler`. Where the
k-th and (k+1)-th logits tie exactly, `TopKLogitsWarper` keeps both and
nuclis keeps the lower id. torch's generator is not reproduced: a seed
gives nuclis's own stream of draws.

**The trace.** `qwen3_tts_trace.py`'s `talker` section hooks one
`generate_custom_voice` run (the run that samples the codes): the
processor's token ids, the talker's input rows (`inputs_embeds` of each
call), its residual stream after layers 0, 13, 27 and after the final
norm, the `codec_head` logits, and the predictor's `lm_head` outputs. The
prompt's (text id, codec id) rows are derived from the config by
`generate`'s rules and checked in the script: rebuilt from them, the rows
equal those the talker received exactly (max |Δ| 0 in all three cases).
Cases, all speaker `ryan`, language `english`, the `fox` sentence, seed 0,
F32: the 0.6B (21 prompt rows, every row pinned), the 1.7B (21), and the
1.7B with the instruction "Speak in a warm, cheerful tone." (13
instruction ids, 34 prompt rows); on the 1.7B the hidden stages keep the
last 4 prompt rows and the 4 decode rows. The fixtures grew to 3.6 MB;
the 0.6B case's existing files were rewritten bit for bit.

**Measured** (`make gate NAME='qwen3-tts-*'`, ReleaseFast, Apple M4 Pro),
replaying the trace's codes: the token ids and the prompt rows are equal
in every case; relative RMS ‖Δ‖ / ‖reference‖ over the rows compared:

| Stage | 0.6B | 1.7B | 1.7B, instruction |
| --- | ---: | ---: | ---: |
| talker inputs (prompt, 4 decode rows) | 3.45e-7 | 5.05e-7 | 5.81e-7 |
| layer 0 | 2.27e-6 | 1.08e-6 | 9.62e-7 |
| layer 13 | 9.63e-7 | 2.45e-6 | 2.98e-6 |
| layer 27 | 9.61e-7 | 2.76e-6 | 2.83e-6 |
| final norm | 2.90e-6 | 2.59e-6 | 3.02e-6 |
| group-0 logits, frames 0–4 | 1.58e-6 | 1.27e-6 | 1.52e-6 |
| predictor logits, groups 1–15 of frames 0–1 | 1.47e-6 | 1.34e-6 | 1.59e-6 |

Every one of the 5 group-0 and 30 predictor logit rows has the trace's
argmax in every case. The gates' bound is relative RMS 1.5e-5 per stage,
five times the worst (3.02e-6).

**Time per frame, the CPU reference** (Apple M4 Pro, 14 cores, F32
weights; the speed is not a target, real time is the Metal plan's). The
gate's replay at ReleaseFast, predictor and talker step without the codec:
69.7 ms a frame for the 0.6B, 84.0 and 90.7 ms for the 1.7B (prefill of
21 rows 131 ms and 366 ms). `nuclis speak` end to end (`make build`,
ReleaseSafe, the codec decoding 16 frames at a time), over the listening
samples below: 106–125 ms a frame on the 0.6B (0.64–0.76× real time) and
130–157 ms on the 1.7B (0.51–0.62×). Loading takes 0.7–0.8 s for the 0.6B
and 1.5–1.6 s for the 1.7B with the files in the page cache (2.8 s and
5.7 s cold).

**Listening samples** (`nuclis speak --language english`, speaker `ryan`,
seed 0; written to `.zig-cache/speak/`, not kept). Five sentences on each
size (the `fox` sentence, a question, one with a time and prices, a
36-word sentence, a confirmation) and, on the 1.7B, one sentence without
an instruction and with three (cheerful, gentle, serious). Checked by
script, not by ear: every file is 24 kHz mono 16-bit, none is silent
(RMS −26.3 to −19.4 dBFS, peaks 0.32–0.64), none has a run of three or
more samples at full scale, and the pace is speech's: 2.2–3.9 words per
second of active audio (50 ms windows above −40 dBFS), the time-and-prices
sentence the slowest on the 1.7B (16 words in 8.96 s). The 0.6B leaves
about 0.45 s of silence before and after; the 1.7B starts within 0.1 s.
The instructions change the delivery measurably on the same sentence and
seed: 6.08 s without, 5.04 s cheerful (−19.4 dBFS), 3.84 s gentle
(−24.2 dBFS, the quietest), 4.96 s serious. Whether they sound right is for
a listener; the round trip through the ASR is the permanent gate (the
unit's decision 1).

## On Metal (2026-10-27)

**The talker and the code predictor** (`inference/src/models/qwen3_tts_metal.zig`,
`Plan`) run the CPU reference's forward with the checkpoint's BF16 matrices
wrapped in place: nothing is widened, so the device holds 1.8 GB (0.6B) or
3.8 GB (1.7B) of weights as published, against the CPU reference's 2.4 and
6.4 GB of F32. Norms and biases are decoded to F32 buffers; text rows and
code rows are BF16 rows widened where they are needed. A decode row runs
each layer as the RMSNorm, one dispatch for q, k, and v
(`nu_tts_matvec3_bf16`), the q/k norm fused with RoPE (`rmsNormRope`,
NEOX pairing), the flash-decoding attention, the output projection with
the residual add fused into the next norm (`addRmsNorm`), gate·up with
SiLU in one dispatch (`nu_tts_gate_up_bf16`), and the down projection. A
prompt runs as chunks of up to 128 rows through the generic F32 matmul
tile with a BF16 segment decode (`nu_tts_matmul_bf16`: the same values in
the same order as the generic tile, which reads BF16 through the
all-encodings switch at a third of the rate) and the causal chunk
attention. The talker's cache is F16 (the family convention, `--kv` in
the check) or F32, grown on demand to 16,384 rows; the predictor's 16 rows
are F32.

**Sampling stays the host's, and the frame stays on the device.** Group 0
(the repetition penalty over generated codes, the end blocked for two
frames, the markers suppressed) is sampled on the CPU from the talker's
logits, as before. A frame's groups 1–15 each depend on the code just
sampled, so sampling them on the host costs a command buffer and a round
trip per group: 16 per frame. Instead `Plan.frameOnDevice` records the
whole frame in one command buffer: each group's logits are sampled by
`nu_tts_sample` (a bitonic sort of the 2,048 logits to the CPU sampler's
order, value descending and id ascending, then its temperature, top-k,
top-p walk in F32) with the uniform the host sampler would draw next
(copied from its generator before the frame), the next group's input row
is gathered from the BF16 table at the device's code (`nu_tts_embed_row`),
and after group 15 the next talker row (§ (e)) and the talker's step
follow. After the commit the host sampler decides every group again from
the same logits, consuming its draws as the CPU path does; where it
disagrees (a draw that F32 rounding moves across a boundary) its code
stands, and the rest of the frame and the talker step are run again a
group at a time. So the codes are always the CPU sampler's: `nuclis speak
--host-sampling` (a command buffer per group) writes the same WAV byte for
byte (checked on both sizes at two seeds), and over every run here the
host overruled the device's draw 0 times. `make test-metal` checks the
sampler against `sampling.Sampler` with the same draw: 380 of 380 equal,
over top-k 50, a nucleus, greedy, and ties at the top.

**The codec decoder** (`inference/src/audio/qwen3_codec_metal.zig`,
`Decoder` and `Stream`) copies the CPU runtime's F32 matrices (the same
tap-major convolution layout, columns padded with zeros to the tile's
64; 543 MB of device buffers) and runs the CPU stream's schedule as one
command buffer per piece of at most 8 frames, its carried state (every
convolution's history, the attention window's 71 keys and values per
layer, the RoPE position) in one device buffer. Convolutions gather their
taps (`nu_tts_gather`) into rows for the generic F32 tile; the window
attention (`nu_tts_window_attention`) gives each key the lane of its
distance from the window's start, and every other kernel sums an output
in an order fixed by the output's own index, so the samples do not depend
on how frames are grouped: pieces of 1, 3, and 7 frames equal one decode
bit for bit (the gate checks it). The new kernels are in
`backends/metal/kernels/tts.metal`, their encoders in `ops/tts.zig`, and
each is a `make test-metal` case against the CPU function it replaces
(`inference/tts-metal-check.zig`); no shared kernel or encoder changed.

**Against the trace and the CPU** (`make gate NAME='qwen3-tts-*'`,
ReleaseSafe, Apple M4 Pro), replaying the trace's codes, relative RMS per
stage:

| Stage | 0.6B, F32 cache | 0.6B, F16 | 1.7B, F32 | 1.7B, F16 | 1.7B instruction, F32 | 1.7B instruction, F16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| talker inputs | 3.67e-7 | 3.67e-7 | 4.99e-7 | 4.99e-7 | 5.74e-7 | 5.74e-7 |
| layer 0 | 8.62e-7 | 4.90e-5 | 8.37e-7 | 2.78e-4 | 7.93e-7 | 3.44e-4 |
| layer 13 | 4.32e-7 | 1.97e-5 | 2.51e-6 | 3.69e-4 | 2.62e-6 | 4.67e-4 |
| layer 27 | 4.14e-7 | 4.58e-5 | 2.99e-6 | 1.87e-4 | 2.80e-6 | 3.46e-4 |
| final norm | 2.94e-6 | 5.28e-4 | 2.59e-6 | 5.04e-4 | 2.83e-6 | 4.22e-4 |
| group-0 logits | 1.21e-6 | 4.37e-4 | 1.09e-6 | 2.74e-4 | 1.21e-6 | 2.33e-4 |
| predictor logits | 1.35e-6 | 2.06e-4 | 1.58e-6 | 1.27e-4 | 1.66e-6 | 9.58e-5 |

Every logit row keeps the trace's argmax in every case and either cache.
With the F32 cache Metal is as close to the trace as the CPU reference
(worst 2.99e-6 against the CPU's 3.02e-6); the F16 cache costs about
5e-4. The gates `qwen3-tts-{0.6b,1.7b}-trace-metal` run both caches:
relative RMS 1.5e-5 per stage with F32, 3e-3 with F16. The codec on Metal
against the CPU decode of the same codes, every frame each case sampled
(46, 171, 141, 86): relative RMS 1.8e-6 to 2.2e-6, max |Δ| 2.9e-6; against
the trace's samples 2.5e-6 and 2.8e-6. `qwen3-tts-codec-metal` bounds both
at 1.5e-5.

### The round trip

`scripts/wer.py --speak tests/fixtures/tts-roundtrip.json` speaks 20
pinned sentences (a question, names, a time, prices, counts spelled as
they are said; speaker `ryan`, English, each its own seed) with `nuclis
speak` on Metal and hears them with `nuclis transcribe --backend cpu`,
then scores the words against the text with the corpus's normalization.
First run, 2026-10-27: on both sizes 5 errors in 190 words, 2.63 %. Most
are the recognizer's spelling, not the voice: "dr" for "Doctor" and
"heres" for "here is" on both, "and" dropped before "fifteen cents" on
both; then "Pitzell" for "Patel" on the 0.6B and "said" for "set" on the
1.7B. The gates `qwen3-tts-{0.6b,1.7b}-roundtrip-metal` hold the rate at
6 % (six more errors).

### Speed

`nuclis speak --stream` (ReleaseSafe, Apple M4 Pro, 48 GB) on "Is it
twenty past nine already? Doctor Okafor said the results would arrive by
Tuesday.", speaker `ryan`, English, seed 0, F16 talker cache, files in
the page cache. Task 7 (the recognizer on Metal) was benchmarking on the
same GPU in parallel: every run below started with no other GPU job
running, and runs another job overlapped are left out; the main session
re-measures after merging. "Codes" is a frame's talker and predictor
wall time, sampling included; the device split comes from `--split`,
which commits between the predictor and the talker (one more command
buffer a frame); the codec is per frame in chunks of 4.

| | 0.6B | 1.7B |
| --- | ---: | ---: |
| frames, audio | 88, 7.04 s | 86, 6.88 s |
| real time (runs) | 3.16×, 3.19×, 3.24× | 2.17×, 2.17× |
| codes a frame | 20.9–21.4 ms | 32.3–32.4 ms |
| of it on the device: talker / predictor | 5.2–5.5 / 14.3–14.9 ms | 14.4–14.6 / 15.9–16.0 ms |
| codec a frame | 3.2–3.3 ms | 3.6 ms |
| prefill (embedding and prompt) | 30–45 ms | 65–68 ms |
| first audio after the text, first chunk 1 frame | 65–80 ms | 111–115 ms |
| first audio, first chunk 4 frames | 139–157 ms (a run before the first-chunk option, chunks of 4) | 209–222 ms |
| with `--host-sampling` (16 command buffers a frame) | 2.86×, 2.88× | 2.03×, 2.05× |
| loading (`Voice.open`, warm) | 0.4 s | 0.4 s |

The first 1.7B run after a build took 1,165 ms to prefill (the shader
library and pipelines compiling once) and is left out of the table.

**Against the targets.** First audio is within the proposed 150 ms on
both sizes once the first chunk is one frame (`Synthesizer.first_chunk`,
default 1; the codec's samples do not depend on the pieces, so nothing
else changes). After it a chunk is at most the frames made before it
(1, 1, 2, 4, …): a second chunk of four would take longer than the first
frame's 80 ms to make, and a listener playing from the first byte would
hear a gap (50–80 ms on the 1.7B, found by the server's voice-loop
measurement, `docs/guide/api.md` § The voice loop). Real time is 3.2× on the 0.6B, at the proposed 3×, and
2.2× on the 1.7B, below it. Sampling the frame on the device saves 2–3 ms
a frame against a command buffer per group.

**Where the 1.7B's frame goes, and why 3× is near the roof.** The bytes a
frame reads, from the tensor shapes (not measured): the 1.7B's talker
2.70 GB (28 layers of 96 MB, the head 13 MB), its predictor 2.37 GB (15
groups of 5 layers of 30 MB, a head and the 2048 → 1024 projection of
4 MB each), 5.07 GB; the 0.6B's talker 0.85 GB and predictor 2.31 GB,
3.16 GB (the unit's estimates were 5.2 and 3.3 GB). At the device times
above the talker reads 186 GB/s (1.7B) and 160 GB/s (0.6B), the predictor
about 150 GB/s; `metal-check --tts-bench` puts one BF16 matvec at 200–250
GB/s in isolation. 3× real time on the 1.7B needs the 5.07 GB in 26.7 ms
less the codec's 3.6, 220 GB/s sustained across about 1,100 dependent dispatches
a frame: the predictor's 1,024-wide matrices (2–6 MB each) are too small
to reach the roof one at a time. Fusing q/k/v, gate·up with SiLU, the
residual adds into the next norm, and writing the F32 cache rows in place
took the predictor's dispatches from about 90 to 55 a group and left its
device time where it was (14.5 ms a frame before, 14.3–14.9 after, 0.6B):
latency, not dispatch count, bounds it. What would move it: the predictor's 5 layers as one kernel per
group, or several frames' predictors batched (they are sequential within
a frame only).

**Re-measured on a quiet machine** (2026-10-27, `519a898`, both Metal plans
merged, nothing else on the GPU; the same sentence, speaker, seed, and
build; three runs each, ranges):

| | 0.6B | 1.7B |
| --- | ---: | ---: |
| real time | 3.09–3.14× | 2.33–2.34× |
| a frame: codes / codec | 21.6–21.8 / 3.2–3.3 ms | 29.8 / 3.3–3.4 ms |
| prefill | 49–59 ms | 82–85 ms |
| first audio after the text (first chunk of 1 frame) | 82–101 ms | 123–126 ms |

The 1.7B stays below the proposed 3×, as above (bandwidth over about
1,100 dependent dispatches a frame), and above real time by 2.3×, which
is what streaming needs.

### For serving

What `POST /v1/audio/speech` can call: `qwen3_tts_runtime.Voice.open(…,
.{ .backend = .metal })` once (it owns its own `metal.Backend`; 0.4 s for
either size with the files in the page cache), then
per request `voice.prompt(text, speaker, language, instruct)` and
`voice.synthesizer(prompt.rows, seed)`, and `read` until it returns 0,
each call one chunk of 16-bit-ready samples (`first_chunk` frames, then
`chunk`). Between `read` calls no command buffer is in flight (every
commit waits), so a request is cancelled by not calling `read` again and
calling `deinit`; a `read` itself runs to the end of its chunk (at most
`chunk` frames). One synthesizer at a time per voice: the plan holds the
talker's cache and the scratch, and the codec decoder its scratch. The
route that does so, its queue, and its rates beside a chat generation:
[api.md § Speech](../guide/api.md#speech).

## Open for the implementation

- **Past 300 frames nuclis decodes causally with carried state**
  (decided 2026-10-27): the decoder is fully causal, so a stream that
  carries its convolution and attention state is the true decode; the
  authors' 25-frame re-warm at each 300-frame chunk is an artifact of
  their batch decode. Traces are compared over the first 300 frames, where
  both agree; past them the CPU–Metal gate compares nuclis with itself.
- **Text mode: non-streaming, a sentence at a time** (decided 2026-10-27).
  The client sends whole sentences (Muon splits Gemma's stream at
  sentence ends), so every request knows its whole text and uses the
  authors' default, the best-tested path. Feeding text as it arrives
  (streaming text, `tts_pad` before the text is complete) is not
  implemented; it waits for a measurement showing that a sentence's
  wait matters.
- **Precision of the authors' runs.** The talker is BF16 and the codec F32
  in the files; the dtype the authors' examples load them with is set by
  the caller, not by the files. The trace loads F32 (the BF16 weights
  widened exactly), as nuclis's CPU reference computes.
- **The tokenizer's case seam is ASCII-only**
  ([tokenizer.md](../engine/tokenizer.md#vocabulary-and-merges-files)): a
  word of a non-ASCII capital then ASCII capitals splits differently from
  the authors' run. A case table would close it if a language needs it.
