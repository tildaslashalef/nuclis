# Fish Audio Drama 3 (35B-A3B)

[fishaudio/drama-3](https://huggingface.co/fishaudio/drama-3) is a
text-to-speech model for 83 languages, with dialogue between up to 16
speakers, free-form `[bracketed]` delivery cues, and cloning from a clip and
its transcript. It has three networks:

- a **backbone**: a Qwen3.5-MoE hybrid (DeltaNet and gated full attention,
  256 experts) that reads the prompt and predicts each frame's main token
  (the semantic code, or the end of speech);
- a **tower**: 4 shared plus 9 residual small transformer layers that
  predict codebooks 1–9 of the frame from the backbone's hidden state;
- a **codec**: Fish's modded DAC at 44.1 kHz, 2,048 samples a frame
  (about 21.5 frames a second), with the encoder that turns a reference
  clip into codes.

Licence: Fish Audio Community License 1.0. No hosted service for third
parties without Fish's agreement, no cloning of a person without their
consent, and "Built with Fish Audio" on redistribution
([THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md)).

## Sources

- **The artifact**: `sakasegawa/drama-3-GGUF` `Drama-35B-A3B-v3-Q4_K.gguf`
  ([catalogue.md § Pinned commits and digests](catalogue.md#pinned-commits-and-digests-2026-09-11)).
  This is speech.cpp's own GGUF layout, architecture `drama3`, converted from
  `fishaudio/drama-3` at `e0a9218a` (`general.source.url`). speech.cpp is read
  only as the definition of this layout: the tensor names and the metadata it
  writes.
- **The semantics**: Fish's code, `fishaudio/fish-speech` at `0f8a12e8`,
  unpacked at `.reference/fish-speech-0f8a12e/`:
  - `fish_speech/models/text2semantic/fish_omni/`: `backbone.py`, `layers.py`,
    `moe.py`, `tower.py`, `generate.py`, `chunking.py`;
  - `fish_speech/models/text2semantic/sampling.py`;
  - `fish_speech/models/dac/`: `modded_dac.py`, `rvq.py`;
  - `fish_speech/configs/modded_dac_vq.yaml`.
- **The traces**: pinned once from those modules running on weights
  dequantized from this GGUF (§ The traces).

Why the GGUF rather than the MLX 4-bit conversion: every encoding in it
(Q4_K, F16, F32) already runs on the CPU and on Metal, so no new
quantization unit is needed. speech.cpp reports its layout matching Fish's
PyTorch at 96–133 dB. The mlx-audio port matches PyTorch at only 86 %
semantic top-1, so it cannot serve as a trace source.

## The file

The inventory is `inference/src/models/fixtures/drama3-q4_k.json`
(`scripts/gguf-inventory.py`). It holds 1,531 tensors: 522 Q4_K, 216 F16
(the codec's convolutions and projections), and 793 F32 (norms, small
projections, the DeltaNet parameters, the codebooks). It also carries
these metadata groups:

| Prefix | Holds |
| --- | --- |
| `drama3.text.*` | the backbone's geometry; `layer_types` (40 strings, `full_attention` at `layer % 4 == 3`) |
| `drama3.audio_decoder.*` | the tower: dim 2048, 16 heads / 2 KV × 256, MLP 6144, 10 codebooks of 4096, 4 shared layers, `vq_backbone_codebooks` 1, RoPE base 1e7 over 64 dims |
| `drama3.eos_token_id` … `pad_token_id` | EOS 248046 (`<\|im_end\|>`), semantic band 248099–252194, pad 248044 |
| `drama3.output.token_ids` | the 4,097 ids the `output` matrix's rows produce: EOS, then the semantic band in order |
| `drama3.sampling.*`, `drama3.repeat.*` | temperature 1.0, top-p 0.9, top-k 30; the repeat window 16 with its second draw at 1.0 / top-k 30 / top-p 0.95 |
| `drama3.generation.*` | default 1,024 frames, max 2,048, prompt at most 8,192 rows, chunks of 300 bytes and 3 speakers, 1 recent turn, 16 speakers |
| `drama3.prompt.*` | template `reference-bracket-v1` and its literal strings (§ The prompt); `reserved`, the six strings user text may not contain |
| `drama3.tokenizer.*` | 252,216 tokens, 247,587 merges, 4,172 added ids, the pre-tokenizer regex (Qwen3.5's) |
| `drama3.codec.*` | the codec's geometry (§ The codec), and the SHA-256 of its tensors |

### Tensor layout, and what differs from Fish's checkpoint

- **The backbone** uses llama.cpp's Qwen3.5 names: `token_embd` [2048, 252288],
  `blk.N.{attn_norm, post_attn_norm}`, and `blk.N.ffn_{gate_inp, gate_exps,
  up_exps, down_exps, gate_shexp, up_shexp, down_shexp, gate_inp_shexp}`.
  - DeltaNet layers add `ssm_in_qkv` [2048, 8192], `ssm_in_z` [2048, 4096],
    `ssm_in_b`, `ssm_in_a` [2048, 32], `ssm_conv1d` [4, 8192], `ssm_dt_bias`,
    `ssm_a_log` [32], `ssm_norm` [128], and `ssm_out`.
  - Full attention layers add `attn_q` and `attn_gate` [2048, 4096],
    `attn_k` and `attn_v` [2048, 512], `attn_q_norm` and `attn_k_norm` [256],
    and `attn_output`.
- **Fish packs `q_proj` per head as [query, gate].** The GGUF splits it into
  `attn_q` and `attn_gate`, each 16 heads × 256 in head order.
- **The lm head is cut to 4,097 rows** (`output` [2048, 4097]): only EOS
  and the semantic band are ever sampled, and `drama3.output.token_ids`
  maps each row back to its id.
- **The zero-centred norms are stored as 1 + w.** Fish's `ZeroCenteredRMSNorm`
  scales by (1 + w). The GGUF's values sit near 1 where Fish's sit near 0:
  the tower's `enorm` has mean 0.993 and range 0.96–1.04, and none of the
  backbone's norms are negative. The binder therefore multiplies by the
  stored weight directly. This applies to `attn_norm`, `post_attn_norm`,
  `output_norm`, `attn_q_norm` and `attn_k_norm` in the backbone, and to
  every `ad.*` norm. `ssm_norm`, the DeltaNet's GatedRMSNorm, is a plain
  weight and is not shifted. The traces confirm this by likelihood
  (§ The traces).
- **The tower** (`ad.*`):
  - `ad.codebook_embd` [2048, 36864] is Fish's `codebook_embeddings`
    without codebook 0, whose rows Fish never reads. Codebook k ≥ 1, code c
    is row (k − 1)·4096 + c.
  - `ad.shared.N.*` and `ad.layer.N.*` are the shared and residual
    `TowerLayer`s, each with `embd` [2048, 4096], `enorm`, `hnorm`,
    `eh_proj` [4096, 2048], `attn_norm`, `ffn_norm`, and `final_norm`.
  - Fish's `wqkv` rows [Q | Gate | K | V] are split into `attn_q`,
    `attn_gate`, `attn_k`, and `attn_v`; the projections are completed by
    `attn_output` and `ffn_{gate, up, down}`.
  - `ad.head.N` [2048, 4096] is codebook head N, which predicts code N + 1.
- **The codec** (`codec.*`) is F16 and F32 with the weight norm already
  folded, as in Fish's `load_codec_model` (§ The codec).

## The backbone

There are 40 layers. Layer i is full attention when i % 4 == 3 and DeltaNet
otherwise. Each layer is pre-norm: x += mixer(norm(x)), then
x += moe(post_attn_norm(x)). The norms are RMS with eps 1e-6, in fp32. A
final `output_norm` precedes the `output` head. The residual stream exceeds
the fp16 range, so it stays in fp32.

- **Full attention** (`layers.py`):
  - Projections: q = attn_q·x, gate = attn_gate·x (16 × 256 each), and
    k, v (2 × 256 each).
  - q and k get a per-head RMS norm, then NeoX (half-split) RoPE over the
    first 64 of 256 dims, θ = 1e7.
  - Causal GQA attention (8 query heads per KV head), scaled by 256^-½.
  - out = attn_output·(attn ⊙ sigmoid(gate)).
- **DeltaNet** (Qwen3.5's gated delta rule):
  - Projections: qkv = ssm_in_qkv·x (8192 = q 2048 | k 2048 | v 4096),
    z = ssm_in_z·x (4096), b = ssm_in_b·x and a = ssm_in_a·x (32 each).
  - qkv passes through a causal depthwise conv (k = 4, from `ssm_conv1d`),
    then SiLU.
  - Heads: 16 key heads × 128 and 32 value heads × 128; q and k are
    repeated to the 32 value heads.
  - q = l2norm(q)·128^-½ and k = l2norm(k).
  - β = sigmoid(b); g = −exp(a_log)·softplus(a + dt_bias). The state
    (32 × 128 × 128, fp32) decays by exp(g) and takes the delta update.
  - Output: ssm_out·(ssm_norm ⊙ rmsnorm(o) ⊙ silu(z)), per value head.
- **MoE** (`moe.py`):
  - Router: softmax over 256 experts, keep the top 8, renormalize
    (`norm_topk_prob`).
  - Each expert computes down(silu(gate·x) ⊙ up·x), with intermediate 512.
  - A shared expert of the same shape is added, scaled by
    sigmoid(ffn_gate_inp_shexp·x).

This is the Qwen3.5-MoE forward nuclis already runs
([qwen3.8.md](qwen3.8.md)). What is new is the input rows and the cut head.

## The frame and the backbone's input

A frame row is [main token, code 0, …, code 9]:

- For a semantic row, code 0 = main − 248099, and codes 1–9 come from the
  tower.
- A text row holds only its token id.

The backbone's input at a row (`FishOmniAudioDecoder.embed_backbone`):

- **text row**: token_embd[id];
- **semantic row**: (token_embd[main] + codebook_embd[0·4096 + code 1]) / √2.
  Here `vq_backbone_codebooks` is 1, and GGUF row 0·4096 is Fish's
  codebook 1.

## The tower

`TowerLayer(hidden, code, semantic, positions, shared_input)`:

- e = embd[code] if the row is semantic, else 0; then e += shared_input
  when one is given.
- mixed = eh_proj·[enorm(e) ‖ hnorm(hidden)].
- One pre-norm block on mixed:
  - attention: q, gate, k, v as in the backbone's full attention, with
    always-on q and k norms;
  - **interleaved** (even/odd pair) RoPE over the first 64 dims, base 1e7;
  - causal attention over time, with its own KV cache per layer;
  - sigmoid gate on the output;
  - then a SwiGLU MLP of 6,144.
- out = final_norm(block).

The decode step at backbone row p, with hidden h_p (`decode_step`):

1. Sample the main token from the 4,097 logits (§ Sampling). Then
   c0 = main − 248099, and semantic = main lies in the band.
2. Run the shared layers 0–3 on (h, c0) with `ad.shared.N.embd`. Layer 0
   also adds `unseen`: the sum of codebook_embd rows for codes 2–9 of the
   *input* frame at p (when that frame is semantic).
3. For j = 0…8: h = residual layer j on (h, c_j) with `ad.layer.j.embd`;
   then c_{j+1} = sample(ad.head.j·h).
4. The new frame is the input of row p + 1.

Every tower layer attends over all earlier positions of the sequence. Its
KV cache is teacher-forced over the prompt (`prefill`): row p is labelled
with frame p + 1's codes, and the last prompt row is filled by the first
decode step. Residual codes are sampled over all 4,096 columns; the codec
clamps them to 1,023.

## Sampling

`sample_top_k_top_p` in `tower.py`:

1. Divide the logits by the temperature, floored at 1e-5.
2. Sort descending. Drop entries where the cumulative softmax exceeds
   top-p or the rank is at least top-k; rank 0 is always kept.
3. Draw by argmax of logits − log(Exp(1) noise), which is exponential-race
   sampling.

**The repeat window (RAS)** applies to the main token. If the first draw
is semantic and appears among the last 16 main tokens (prompt rows
included), a second draw is taken at temperature 1.0, top-k 30, top-p 0.95
and replaces it. EOS is never redrawn.

Defaults differ by source. The card and the GGUF use 1.0 / 0.9 / 30;
Fish's SDK uses 0.7 / 0.7 / 30. Fish's `reference_sampling` (an epsilon
cutoff, and residual codes legal below 1,024) is off by default and not
implemented.

## The prompt

The `reference-bracket-v1` template. Each literal is tokenized on its own,
and codes enter as frame rows:

```
<|im_start|>system\n
convert the provided text to speech                       (no reference)
convert the provided text to speech reference to the following:
  per reference N (1-based): "\n\n[Reference N]\nText: " <|speaker:k|>transcript "\nSpeech: " <code rows>
<|im_end|>\n
history turns: <|im_start|>user\n <tagged text> <|im_end|>\n <|im_start|>assistant\n<|voice|> <code rows> <|im_end|>\n
<|im_start|>user\n <tagged text> <|im_end|>\n <|im_start|>assistant\n<|voice|>
```

- *Tagged text* is the user's text, prefixed `<|speaker:0|> ` unless it
  already opens with a speaker tag.
- `<|speaker:0|>`…`<|speaker:15|>` are single added tokens.
- Text may not contain the six `drama3.prompt.reserved` strings
  (`<|im_start|>`, `<|im_end|>`, `<|voice|>`, `<|text|>`, `<|interleave|>`,
  `<|semantic:`).
- A prompt is at most 8,192 rows.

## Chunking and history

`chunking.py` splits long text into chunks of about 300 UTF-8 bytes, with
at most 3 distinct speakers per chunk. The splitter tries, in order:

1. sentence ends `.!?;。！？；…` and newlines (but not a `.` inside a
   number);
2. clause marks `,:，：、`;
3. spaces;
4. raw length.

It never splits inside `<|…|>`, a paired tag, or `[…]`.

Each chunk is a user turn. The prompt carries the first chunk's turn
(pinned) plus the latest 1 earlier turn as history. When the prompt would
be too long, history is dropped. Each chunk generates at most 600 frames
(about 27.9 s).

## The codec

`fish_speech/models/dac/modded_dac.py` with `configs/modded_dac_vq.yaml`.
The GGUF's `drama3.codec.*` carries its geometry:

- encoder: dim 64, rates 2·4·8·8, with a 4-layer windowed (512)
  transformer after the last block;
- quantizer: downsample ×2·2 with ConvNeXt blocks, then 8-layer causal
  windowed (128) pre- and post-transformers (16 heads × 64, RoPE base 1e4,
  eps 1e-5);
- codebooks: a semantic codebook of 4,096 and 9 residual codebooks of
  1,024, each with codebook dim 8;
- decoder: dim 1,536, rates 8·8·4·2;
- Snake activations throughout.

The stage-by-stage contract is written when the codec is implemented
(§ Open).

## Stalls, and what nuclis adds

Measured in the lab (2026-10-30, speech.cpp, this file). On short Egyptian
Arabic phrases, about one take in five stalls:

- **the model speaks the first sentence, then 10–28 s of silence**
  ("إزيك يا باشا؟ عامل إيه؟");
- **a three-word phrase comes out empty or as silence.**

The bf16 checkpoint stalls too (a 500-clip study by fcmeyer: 4 % of
clips), so this is the model's behaviour, not the quantization's.

| Configuration (4 worst phrases, 2 voices × 4 seeds) | Stalls |
| --- | --- |
| temperature 1.0, top-p 0.9 (top-k 30 or off) | 13 / 32 |
| temperature 0.7, top-p 0.7 | 13 / 32 |
| split at sentence ends | 5 / 32 |

nuclis therefore splits at every sentence end, not only past 300 bytes.
The stop on silent frames and the retry are designed with the sampler
(§ Open).

## The traces

Not yet pinned (§ Open).

## Open

- The trace script, its fixtures, and the confirmation of the norm and
  codebook conventions.
- The codec's stage contract.
- The silence stop and retry.
