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
  weight and is not shifted. Confirmed by likelihood (§ Confirmed by
  likelihood). The codec's RMS norms are plain weights too.
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
- **The DeltaNet value heads are in Fish's grouped order**: value head h
  reads key head h / 2 (`repeat_interleave`). llama.cpp's own Qwen3.5
  converter reorders them to the tiled order, where head h reads key head
  h % 16, but speech.cpp did not. So `ssm_in_qkv`'s V rows, `ssm_in_z`,
  `ssm_in_b`, `ssm_in_a`, `ssm_dt_bias`, `ssm_a_log`, the V channels of
  `ssm_conv1d`, and `ssm_out`'s columns are read as stored, and q and k are
  repeated to the value heads **interleaved** (k₀ k₀ k₁ k₁ …), not tiled.
- `ssm_a_log` is Fish's raw `A_log`, with values of both signs; it is not
  llama.cpp's −exp(A_log). `ssm_conv1d` [4, 8192] is [channel][tap].
- **The codec** (`codec.*`) is F16 and F32 with the weight norm already
  folded, as in Fish's `load_codec_model` (§ The codec, § The GGUF's codec
  layout).

### Confirmed by likelihood

The `clone-fox` trace (§ The traces) runs Fish's modules under each
reading. It measures the mean log-probability of the reference clip's own
100 codes, teacher-forced. For the backbone that is the main token over
the 4,097-wide head; for the tower it is codes 1–9 over each 4,096-wide
head. A uniform guess scores −8.3. The reading above wins every question,
and each alternative collapses:

| Question | Chosen | Backbone | Tower |
| --- | --- | --- | --- |
| zero-centred norms stored as 1 + w | yes | −4.40 (unshifted −28.03) | −5.27 (unshifted −23.22) |
| `ad.codebook_embd` row (k − 1)·4096 + c | yes | −4.40 (rows k·4096 + c: −8.51) | −5.27 (−6.37) |
| `attn_q` / `attn_gate` as named | yes | −4.40 (swapped −9.22) | −5.27 (swapped −11.20) |
| DeltaNet value heads grouped, as stored | yes | −4.40 (tiled −11.11) | — |

Under the chosen reading the backbone ranks the true code first 17 % of
the time over the reference, and the continuation frames (the clip's
opening, which the prompt already holds) score −2.01. The tower's
continuation scores −3.43. Every wrong reading has top-1 0 % on the
backbone.

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
    repeated to the 32 value heads interleaved: value head h reads key
    head h / 2.
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

The `reference-bracket-v1` template (`build_drama3_prompt`). Each literal
below is tokenized on its own, and codes enter as frame rows:

```
<|im_start|>system\n
convert the provided text to speech                       (no reference)
convert the provided text to speech reference to the following:
  per reference N (1-based): "\n\n[Reference N]\nText: <|speaker:k|> transcript\nSpeech: " <code rows>
<|im_end|>\n
history turns: <|im_start|>user\n <tagged text> <|im_end|>\n <|im_start|>assistant\n<|voice|> <code rows> <|im_end|>\n
<|im_start|>user\n <tagged text> <|im_end|>\n <|im_start|>assistant\n<|voice|>
```

- *Tagged text* is the user's text, prefixed `<|speaker:0|> ` unless it
  already opens with a speaker tag.
- A reference's block is **one** literal: Fish tokenizes
  `f"\n\n[Reference {N}]\nText: {transcript}\nSpeech: "` whole, where the
  transcript is tagged like the text (`<|speaker:0|> Every morning…`, the
  space included) and leading white space is moved behind the tag. The
  GGUF's `reference_open`, `reference_text` and `reference_speech` are the
  pieces of that one string, not separate literals.
- A text row's frame is [id, 0, …, 0]; a code row's is
  [248099 + c0, c0, …, c9].
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

Every stage is causal (`DAC(causal=True)`, the default the yaml keeps);
Fish's own names are in parentheses. The traces pin each boundary below
(§ The traces).

### Building blocks

- **Causal convolution** (`CausalConvNet`): kernel K, dilation d, stride s,
  effective span K′ = (K − 1)·d + 1. The input gets K′ − s zeros on the left,
  and on the right just enough zeros to complete the last stride, so a length
  L becomes ⌈L / s⌉. For s = 1 that is K′ − 1 zeros on the left and none on
  the right.
- **Causal transposed convolution** (`CausalTransConvNet`): kernel 2s,
  stride s; `ConvTranspose1d` gives (L + 1)·s samples, and the last s are cut,
  leaving L·s.
- **Snake** (`dac.nn.layers.snake`): y = x + sin²(α·x) / (α + 1e-9), α per
  channel. The GGUF stores α and `inv_alpha`, which is bit-identical to the
  F32 1 / (α + 1e-9).
- **Residual unit** (`ResidualUnit`, dilation d): y = conv₁ₓ₁(snake(conv₇,d(snake(x)))),
  out = x + y. The 7-tap conv has dilation d ∈ {1, 3, 9}, so 6d zeros on the
  left.
- **ConvNeXt** (`ConvNeXtBlock`): out = x + γ ⊙ pw2(GELU(pw1(LN(dw(x))))).
  Here dw is a causal depthwise 7-tap conv (6 zeros on the left), LN a
  LayerNorm over channels (eps 1e-6, with weight and bias), GELU the exact
  erf form, pw1 1024 → 4096 and pw2 4096 → 1024 linears with biases, and γ
  per channel.
- **Windowed transformer** (`WindowLimitedTransformer`): dim 1024, 16 heads
  × 64, plain multi-head attention (no GQA), no input or output projection.
  - Each layer is h = x + γₐ ⊙ attn(rms(x)), then out = h + γ_f ⊙ ffn(rms(h)),
    with a final RMS norm after the last layer. The norms are plain RMS
    norms (weight × x̂, eps 1e-5, not zero-centred); γₐ and γ_f are the
    `LayerScale`s (`attn_scale`, `ffn_scale`).
  - ffn = w2(silu(w1·x) ⊙ w3·x), 1024 → 3072 → 1024, without biases.
  - RoPE rotates **interleaved** pairs (2i, 2i + 1) over all 64 dims of each
    head, base 1e4, angle t · 10000^(−2i/64). Fish rounds the cos and sin
    tables to **bfloat16** (`precompute_freqs_cis`'s default dtype) before
    rotating in F32. Positions restart at 0 for every call.
  - The mask is causal and windowed: row t attends to columns
    max(0, t − W + 1) … t. W is 512 in the encoder and 128 in the quantizer.
    Attention is scaled by 1/8.

### Encode (`encode_audio`, `DAC.encode`)

1. Average the channels, then resample to 44.1 kHz with
   `torchaudio.functional.resample`'s defaults (`sinc_interp_hann`, lowpass
   width 6, rolloff 0.99). Pad the end with zeros to a multiple of 2,048
   samples; frames = ⌈samples / 2048⌉.
2. **Encoder** (`encoder.block`):
   - `in_conv`: causal 7-tap conv, 1 → 64 channels.
   - Four blocks (`EncoderBlock`) with output dims 128, 256, 512, 1024 and
     strides 2, 4, 8, 8. Each block runs residual units with d = 1, 3, 9 at
     half its dim, then Snake, then a causal conv with kernel 2s and stride
     s. Block 3 then runs the 4-layer transformer with W = 512.
   - Snake(1024), then `out_conv`, a causal 3-tap conv from 1024 to 1024
     channels. This is z, at 1/512 of the sample rate.
3. **Downsample** (`quantizer.downsample`): twice, a causal conv with
   kernel 2 and stride 2 (1024 → 1024), then a ConvNeXt block. This reaches
   the frame rate, 1/2048 of the sample rate.
4. **Pre-transformer** (`quantizer.pre_module`): 8 layers, W = 128.
5. **Vector quantizers** (`dac.nn.quantize`). There is the semantic one
   (4,096 codes) and 9 residual ones (1,024 codes each), all with codebook
   dim 8. Each works on its input r as follows:
   - z_e = in_proj·r + b, a 1 × 1 conv from 1024 to 8;
   - code = argmax_j n(z_e) · n(c_j), the cosine nearest codeword, with n
     the L2 normalization (eps 1e-12);
   - z_q = out_proj·c_code + b, from 8 to 1024. The raw codeword is used
     here, not its normalized form.

   The semantic VQ reads the pre-transformer's output z and gives code 0.
   Residual VQ k reads r_k, where r₀ = z − z_q(semantic) and r_{k+1} =
   r_k − z_q,k. This gives codes 1–9. A frame's codes are [semantic,
   residual 0, …, residual 8].

### Decode (`DAC.from_indices`)

1. Clamp code 0 to ≤ 4095 and codes 1–9 to ≤ 1023 (the tower samples
   residual codes over 4,096 columns).
2. z = out_proj_sem(codebook_sem[c0]) + Σ_k out_proj_k(codebook_k[c_{k+1}]).
3. **Post-transformer** (`quantizer.post_module`): 8 layers, W = 128.
4. **Upsample** (`quantizer.upsample`): twice, a causal transposed conv
   with kernel 2 and stride 2 (1024 → 1024), then a ConvNeXt block. This
   gives 4 rows per frame.
5. **Decoder** (`decoder.model`):
   - `in_conv`: causal 7-tap conv, 1024 → 1536.
   - Four blocks (`DecoderBlock`) with rates 8, 8, 4, 2 and dims
     1536 → 768 → 384 → 192 → 96. Each block runs Snake, then a causal
     transposed conv with kernel 2s and stride s that halves the channels,
     then residual units with d = 1, 3, 9.
   - Snake(96), a causal 7-tap conv from 96 to 1, then tanh.

   The output is exactly frames × 2,048 samples.

The yaml's `decoder_transformer_layers: [4, 0, 0, 0]` has no effect.
`DecoderBlock` builds that transformer as a local variable, but the line
that would put it into `self.block` is commented out (`# transformer_module,`).
So it is never a submodule, never runs, and has no state-dict entry. Fish
loads the codec `strict=True`, so the checkpoint has none either, and that
is why the GGUF has no `codec.dec.*.tf` tensors.

### The GGUF's codec layout

Decided by reconstruction (§ The traces). Encoding then decoding the
reference clip under each of the 16 readings the names and shapes leave
open gives 10.6 dB SNR for the layout below. Every other reading gives
−1.2 to −0.1 dB.

- **Convolutions** (`*.weight` [ne0 = in, ne1 = out, ne2 = k]) are tap
  matrices [k][out][in]. Output t takes tap k times x[t·s + k·d − (K′ − s)], with zeros outside the input. Transposed
  convolutions (`dec.blk.N.up`, `q.up.N.conv`) are stored the same way,
  [k][out][in]; Fish's `ConvTranspose1d` weight is [in][out][k].
- **Stride convolutions as matrices**:
  - `enc.blk.N.down.first` and `.second` [s·in, out] are the encoder conv
    with kernel 2s cut in two: `first` holds taps 0…s−1 and `second` holds
    taps s…2s−1, each tap-major (column k·in + i).
  - So out[t] = first · x[(t−1)s … ts−1] + second · x[ts … ts+s−1] + bias.
    Each slice is s frames of `in` channels flattened frame by frame, and
    frame −1 is zero.
  - `q.down.N.conv.weight` [2·in, out] is the whole stride-2 kernel in the
    same tap-major order.
- `q.up.N` is Fish's `upsample.N`, the N-th applied.
- The ConvNeXt `dwconv.weight` [ne0 = C, ne1 = 7] is [k][C].
- The `pw1` and `pw2` linears, VQ `in_proj` [8][1024], `out_proj`
  [1024][8], and `codebook` [codes][8] are row-major [out][in].

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

`scripts/speech-traces/drama3_trace.py` runs Fish's own modules (the
`DecoderLayer`s, `FishOmniAudioDecoder`, the DAC built from
`modded_dac_vq.yaml` through hydra, `build_drama3_prompt`) on the CPU in
F32. The weights are dequantized from this GGUF: Q4_K by ggml's formula,
F16 widened. The fixtures are under `tests/fixtures/drama3/`, 6.3 MB.
`index.json` names every file's shape, dtype, SHA-256 and rows, and also
holds the GGUF→Fish name map, every layout check with its numbers
(`verdicts`), and the versions. Each array is little-endian and time
first; `rows` gives the [start, end) ranges of the whole stage that a
file holds.

| Case | What is pinned |
| --- | --- |
| `tokenizer` | the ids and tokens of 12 strings (English, Egyptian Arabic, Chinese, Russian, numbers, `[sighs]` cues, `<whisper>` tags, speaker tags, newlines and CRLF, a decomposed `é`, runs of spaces and tabs, a chat turn). The tokenizer is built from `drama3.tokenizer.*` alone: NFC, the regex split isolated, byte level, BPE without `ignore_merges`, and the added ids matched whole |
| `codec-reference` | the clip (Qwen3-TTS's pinned 24 kHz reference, 4.64 s) resampled to 44.1 kHz, whole (204,624 samples). Every encoder stage at its first and last rows; each VQ's 8-dim latent and the codes [100, 10], whole. Then `from_indices` on those codes: each stage at its first and last rows, and the first and last 2 frames of the 204,800 output samples (the whole's SHA-256 and RMS are in the index). Also the 16-way layout search (§ The GGUF's codec layout) |
| `codec-window` | the clip twice over: 800 encoder-transformer rows and 200 frames, past both windows (512 and 128). The last 16 rows of the encoder transformer, the pre-transformer and the post-transformer, the codes, and the last 2 frames of samples |
| `clone-fox` | a 164-row prompt with the clip as Reference 1 and the text `The quick brown fox jumps over the lazy dog.`, then the clip's first 24 frames teacher-forced (188 rows). Pinned: `frames` [188, 11] whole; the backbone's input rows, residual after layers 0, 3, 19 and 39, `output_norm`, and the 4,097 logits at rows 0–7 and 156–187; the tower's `prefill` at 6 rows (a text row, a reference row, the last prompt row, and three continuation rows): the shared stack's output, each residual layer's output, and each head's 4,096 logits. Also the likelihood checks (§ Confirmed by likelihood) |

The clip's encoder transformer sees 400 rows and its quantizer
transformers 100 frames, so only `codec-window` exercises the window
masks. The backbone runs layer-streamed. It builds one Fish `DecoderLayer`
at a time with all 256 experts dequantized (3 GB), runs all eight readings
through it, and frees it. One teacher-forced prefill from position 0 is
the same computation as the decode steps, since the model is causal. The
whole run takes 5.6 minutes on the M4 Pro, with a peak resident size of
32 GB that counts the GGUF's mapped pages. A second run reproduces every
file byte for byte.

To regenerate, use a throwaway venv. It needs `descript-audio-codec`
installed `--no-deps`, since its training-only imports are not needed and
the script stubs `audiotools`:

```sh
uv venv --python 3.12 .reference/venv-drama3
VIRTUAL_ENV=.reference/venv-drama3 uv pip install torch==2.14.1 torchaudio==2.11.1 \
  numpy==2.5.4 tokenizers==0.22.2 einops==0.8.2 loguru hydra-core omegaconf tqdm safetensors
VIRTUAL_ENV=.reference/venv-drama3 uv pip install --no-deps descript-audio-codec==1.0.0
.reference/venv-drama3/bin/python scripts/speech-traces/drama3_trace.py [--cases NAME,...]
```

The clip is read as `torchaudio.load` reads it (int16 / 32768); torchaudio
2.11 needs torchcodec for `load`, so the script uses `wave`. Fish loads
the codec in bfloat16 by default (`load_codec_model`), and the traces are
F32.

### Agreement on the CPU (2026-10-30)

`drama3-check MODEL.gguf trace` replays `clone-fox` through the CPU
runtime: prefill of the 164 prompt rows, the tower's prefill over rows
0–162, then each continuation row as a tower step fed the trace's codes and
a backbone decode of the next frame. The gate `drama3-trace-cpu` holds
every stage at relative RMS ≤ 1e-5 and every logits argmax equal. Measured
on the M4 Pro, ReleaseFast, the Q4_K file, against a trace computed with the
delta rule exact (below):

| Stage | Rows | rel RMS |
| --- | --- | --- |
| backbone input | 40 | 5.4e-8 |
| after layer 0 / 3 / 19 / 39 | 40 | 4.1e-7 / 5.3e-7 / 4.5e-7 / 1.0e-6 |
| `output_norm` | 40 | 2.1e-6 |
| logits (argmax 40/40) | 40 | 2.5e-6 |
| tower, shared stack | 6 | 3.2e-6 |
| tower, residual layers | 54 | 3.1e-6 |
| tower, head logits (argmax 54/54) | 54 | 1.7e-6 |

Fish's prefill runs `chunk_gated_delta_rule` (64-row chunks, two F32
triangular solves). Its rounding, about 2e-6 at layer 2, changes the input
of layer 3's router at row 34. The eighth and ninth experts there are
0.41913772 and 0.41913646 apart by 1.3e-6, so the chunked trace takes the
other expert. From that row on it differs from the exact computation by
2e-3 at layer 39, 3.8e-2 at `output_norm`, and one logits argmax in 40
(two in 54 for the tower). Fish's own layers rerun with
`recurrent_gated_delta_rule` in F64, everything else unchanged, agree with
nuclis to 2.2e-6 through layer 19 at all 188 rows. The `clone-fox`
fixtures therefore come from that run: `section_clone` replaces
`layers.chunk_gated_delta_rule` with `exact_delta_rule`, the recurrent rule
in F64 (the index's `dtype` says so). The
backbone and tower verdicts are unchanged under it. The best reading's
reference log-probability goes from −4.4039 to −4.3752.

## Open

- The silence stop and retry.
