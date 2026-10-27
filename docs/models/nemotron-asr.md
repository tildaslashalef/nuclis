# Nemotron 3.5 ASR streaming 0.6B

[Nemotron 3.5 ASR streaming 0.6B](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
(NVIDIA, OpenMDW-1.1) is a streaming speech recognizer: 16 kHz mono audio
in, punctuated and capitalized text out, in 40 locales. It is a
cache-aware FastConformer encoder (24 blocks of width 1024) with an RNN-T
transducer (an LSTM prediction network and a joint network over 13,088
outputs), conditioned on a language prompt. In nuclis it is the ears of a
local voice assistant: `nuclis transcribe` first, later a streaming
WebSocket transcription route on `nuclis serve`.

Everything here was read on 2026-10-27 from these sources: the pinned
repository's `config.json`, `generation_config.json`,
`processor_config.json`, `tokenizer_config.json`, `tokenizer.json` and
card; the `model.safetensors` header, read over HTTP (the file was not yet
pulled); `model_config.yaml` inside the `.nemo` archive (NeMo's own
configuration); and Hugging Face transformers at `c0d024fb`, unpacked
under `.reference/transformers-c0d024f/` (`nemotron3_5_asr/`,
`nemotron_asr_streaming/`, `parakeet/`, and `cache_utils.py`,
`masking_utils.py` from the same commit). The transformers code is the
definition, read and never copied. Where a fact comes from it rather than
from a file, the sentence names the class or function.

**Contents.**

- [Artifacts](#artifacts)
- [Configuration](#configuration)
- [Tensors (655)](#tensors-655)
- [The front end](#the-front-end)
- [The encoder](#the-encoder)
- [Cache-aware streaming](#cache-aware-streaming)
- [RNN-T decoding](#rnn-t-decoding)
- [Tokenizer](#tokenizer)
- [Front end and encoder against the trace (2026-10-27)](#front-end-and-encoder-against-the-trace-2026-10-27)
- [Transducer against the trace, and word error rate (2026-10-27)](#transducer-against-the-trace-and-word-error-rate-2026-10-27)
- [On Metal (2026-10-27)](#on-metal-2026-10-27)
- [Decisions and open items](#decisions-and-open-items)

## Artifacts

| Repository | Commit | File | Size (B) | SHA-256 |
| --- | --- | --- | ---: | --- |
| `nvidia/nemotron-3.5-asr-streaming-0.6b` | `ea30d66debe3740a08b573244286791d423d6b3e` | `model.safetensors` | 2,552,062,944 | `9eebdd6590289cb3030f310858f3df93256600a800a3e8200c5993d5f967e174` |
| | | `config.json` | 1,376 | `62d186fd91f518e00e7867500f1f5819225e8ee95ea3e21b546514bf2048e845` |
| | | `generation_config.json` | 193 | `993e5d4cb74a6fe9d6e7084a76b3313c1446740679be4676570c23b664fdc07e` |
| | | `processor_config.json` | 2,519 | `ec47870f1091ea4f25539208387b45b902c92d0e3f997a30061ef88f73437ab0` |
| | | `tokenizer.json` | 752,051 | `3f3d481deb073b64c2082e8c7860d487a3a62774bf4e9e4faac83007e181f246` |
| | | `tokenizer_config.json` | 881 | `5c641c5b3f50702a60082690d27c1ce7fcb5a92c4a624793bcae0f21eda3d6e0` |

The Hub lists `model.safetensors` with the LFS object id
`9eebdd6590289cb3030f310858f3df93256600a800a3e8200c5993d5f967e174`, which
the pull should reproduce. The rows go to
[catalogue.md § Safetensors artifacts](catalogue.md#safetensors-artifacts)
once pulled.

Not used: the `.nemo` (2,368,284,501 B, NeMo's gzipped tar of the
tokenizer, `model_config.yaml` and weights) and the `.q8_0.gguf`
(742,090,464 B, for NVIDIA's NeMo-Speech.cpp). License: OpenMDW-1.1 (the
card's `license_name` and *License/Terms of Use*).

## Configuration

From `config.json` (`Nemotron3_5AsrForRNNT`, `model_type`
`nemotron3_5_asr`) and its `encoder_config` (`nemotron_asr_streaming_encoder`):

| Key | Value |
| --- | --- |
| `encoder_config.hidden_size` / `intermediate_size` | 1024 / 4096 |
| `num_hidden_layers` / `num_attention_heads` / `num_key_value_heads` | 24 / 8 / 8 (head width 128) |
| `hidden_act` | `silu` (feed-forward and convolution module) |
| `attention_bias` / `convolution_bias` | false / false |
| `conv_kernel_size` | 9 |
| `subsampling_factor` / `subsampling_conv_channels` / kernel / stride | 8 / 256 / 3 / 2 |
| `num_mel_bins` | 128 |
| `scale_input` | false |
| `max_position_embeddings` | 5000 |
| `sliding_window` | 57 = left context 56 + 1 |
| `supported_num_lookahead_tokens` / `default_num_lookahead_tokens` | [3, 0, 6, 13] / 3 |
| `decoder_hidden_size` / `num_decoder_layers` | 640 / 2 |
| `hidden_act` (joint) | `relu` |
| `vocab_size` / `blank_token_id` / `pad_token_id` | 13088 / 13087 / 0 |
| `max_symbols_per_step` | 10 |
| `num_prompts` / `prompt_intermediate_size` / `default_prompt_id` | 128 / 2048 / 101 (`auto`) |
| `dtype` | `float32` |

`generation_config.json`: `decoder_start_token_id` 13087 (the blank).
`processor_config.json`: the feature extractor
(`NemotronAsrStreamingFeatureExtractor`: `feature_size` 128, `sampling_rate`
16000, `n_fft` 512, `win_length` 400, `hop_length` 160, `preemphasis` 0.97,
`padding_value` 0.0), the same lookahead set, and `prompt_dictionary`: 121
locale strings onto 84 prompt ids up to 104 (`en-US` and `en` 0, `de-DE` 9,
`auto` 101, …).

NeMo's `model_config.yaml` agrees and says more: `att_context_size`
[[56, 3], [56, 0], [56, 6], [56, 13]], `att_context_style`
`chunked_limited`, `self_attention_model` `rel_pos`, `untie_biases` true,
`xscaling` false, `conv_norm_type` `layer_norm`, `conv_context_size`
`causal`, `subsampling` `dw_striding`, `causal_downsampling` true,
`use_bias` false; preprocessor `normalize: NA`, `window: hann`, `log: true`,
`dither: 1.0e-05`, `pad_to: 0`; decoder `blank_as_pad: true`; decoding
`greedy_batch` with `max_symbols: 10`.

## Tensors (655)

The header is 74,584 B; the data 2,551,988,352 B; every tensor is F32,
637,997,088 parameters. Shapes are PyTorch's, `[out, in, …]`.

| Tensor | Shape |
| --- | --- |
| `encoder.subsampling.conv_in.weight` / `.bias` | [256, 1, 3, 3] / [256] |
| `encoder.subsampling.layers.{0,1}.depthwise_conv.weight` / `.bias` | [256, 1, 3, 3] / [256] |
| `encoder.subsampling.layers.{0,1}.pointwise_conv.weight` / `.bias` | [256, 256, 1, 1] / [256] |
| `encoder.subsampling.linear.weight` / `.bias` | [1024, 4352] / [1024] |
| `prompt_projector.linear_1.weight` / `.bias` | [2048, 1152] / [2048] |
| `prompt_projector.linear_2.weight` / `.bias` | [1024, 2048] / [1024] |
| `encoder_projector.weight` / `.bias` | [640, 1024] / [640] |
| `decoder.embedding.weight` | [13088, 640] |
| `decoder.lstm.weight_ih_l{0,1}` / `weight_hh_l{0,1}` | [2560, 640] / [2560, 640] |
| `decoder.lstm.bias_ih_l{0,1}` / `bias_hh_l{0,1}` | [2560] / [2560] |
| `decoder.decoder_projector.weight` / `.bias` | [640, 640] / [640] |
| `joint.head.weight` / `.bias` | [13088, 640] / [13088] |

Per block `encoder.layers.N.` (N = 0…23, 26 tensors, 31 + 24 × 26 = 655):

| Tensor | Shape |
| --- | --- |
| `norm_feed_forward1`, `norm_self_att`, `norm_conv`, `norm_feed_forward2`, `norm_out` (`.weight`, `.bias`) | [1024] each |
| `feed_forward{1,2}.linear1.weight` / `linear2.weight` | [4096, 1024] / [1024, 4096] |
| `self_attn.{q,k,v,o}_proj.weight` | [1024, 1024] |
| `self_attn.relative_k_proj.weight` | [1024, 1024] |
| `self_attn.bias_u`, `self_attn.bias_v` | [8, 128] |
| `conv.pointwise_conv1.weight` | [2048, 1024, 1] |
| `conv.depthwise_conv.weight` | [1024, 1, 9] |
| `conv.norm.weight`, `conv.norm.bias` | [1024] |
| `conv.pointwise_conv2.weight` | [1024, 1024, 1] |

No block matrix has a bias. `conv.norm` is a LayerNorm: NeMo stores it as
`conv.batch_norm` whatever its type, and the converter renames it (the
mapping comment in `convert_nemotron_asr_streaming_to_hf.py`); there is no
running mean or variance, so nothing to fold. The four LSTM matrices stack
the gates PyTorch's way, rows `[i | f | g | o]`, 640 each. The prompt
matrix's 1152 columns are the encoder row (0…1023) then the one-hot
(1024…1151), in `torch.cat([hidden_states, one_hot])` order.

**Values read from the file.** `decoder.embedding.weight` row 13087 (the
blank) is all zeros (`blank_as_pad`: the blank is the padding index); row
13086 is not (max |x| 2.57). The positional table is not stored: it is
computed (below).

## The front end

`NemotronAsrStreamingFeatureExtractor._torch_extract_fbank_features` and
`__call__`, on a clip `x` of L samples at 16 kHz:

1. **Pre-emphasis** over the clip: `y[0] = x[0]`, `y[n] = x[n] − 0.97 · x[n−1]`.
2. **Centering**: pad `y` with 256 zeros on each side (`torch.stft`,
   `center=True`, `pad_mode="constant"`).
3. **Frame k** is samples `160k − 256 … 160k + 255` of `y` (zero outside
   `[0, L)`), times a **symmetric** Hann window of 400
   (`torch.hann_window(400, periodic=False)`, `w[n] = ½ − ½ cos(2πn/399)`),
   zero-padded to 512 with 56 zeros each side. So only samples
   `160k − 200 … 160k + 199` count, and the end ones get weight 0.
4. **Power** of the 512-point real FFT: 257 bins `|X|²`.
5. **Mel**: `librosa.filters.mel(sr=16000, n_fft=512, n_mels=128, fmin=0,
   fmax=8000, norm="slaney")`, cast to F32. That is the **Slaney** scale
   (librosa's default `htk=False`): `mel(f) = 3f/200` below 1000 Hz and
   `15 + 27 ln(f/1000)/ln 6.4` above. 130 edges `f_0 … f_129` are equally
   spaced in mel over 0–8000 Hz. Filter m is the triangle `f_m → f_{m+1} →
   f_{m+2}` at bin frequencies `31.25 b`, times the **Slaney area norm**
   `2/(f_{m+2} − f_m)`.
6. `log(mel + 2⁻²⁴)` (`LOG_ZERO_GUARD_VALUE`). No per-feature
   normalization, no dither.
7. **Frames kept**: STFT gives `⌊L/160⌋ + 1` frames. The mask keeps
   `⌊L/160⌋`, and the last frame is multiplied by the mask, so it becomes
   **0**, not `log 2⁻²⁴`.

The HF converter drops NeMo's `normalize` (`NA`), `dither` (1e-5 in
NeMo's config), `window`, `log` and `pad_to`. The feature extractor
implements none of them beyond what is listed above, and nuclis follows it.

Against Gemma's front end, which `inference/src/audio/mel.zig` implements
today ([audio.md § The front end](../engine/audio.md#the-front-end)):

| | Gemma 4 (`mel.zig`) | Nemotron |
| --- | --- | --- |
| Pre-emphasis | none | 0.97, first sample kept |
| Padding | 160 zeros on the left | 256 zeros on each side |
| Frame | 320 samples at `160 i − 160` | 512 at `160k − 256`, 400 of them windowed |
| Window | periodic Hann 320 | symmetric Hann 400, centered in 512 |
| Spectrum | magnitude | power |
| FFT, hop, filters, range | 512, 160, 128, 0–8000 Hz | same |
| Mel scale and norm | HTK, unnormalized | Slaney scale, Slaney area norm |
| Log | `log(m + 1e-3)` | `log(m + 2⁻²⁴)` |
| Frame count | `⌈(L − 160)/160⌉` | `⌊L/160⌋` (+ one zeroed frame) |
| Normalization, dither | none | none |

**Streaming features.** A mel frame depends only on its own samples, so
frame k is final once sample `160k + 199` exists and the previous sample
(for pre-emphasis) is known. The HF processor cuts audio differently but
reproduces the same frames. The first chunk is `center=True` over
`(n − 1)·160 + 200` samples. Each later chunk is `center=False` over
`n·160 + 400` samples starting at `160m − 256` (`num_samples_*_audio_chunk`,
the card's streaming example). A chunk's unfiltered first sample sits
under the window's zero padding. At the end, `is_last_audio_chunk` pads
`n_fft/2 − hop` = 96 zeros so that the last whole-clip frame is kept.

## The encoder

Notation: N mel frames, rows `x_t` of width 1024, `LN` a LayerNorm with
weight and bias and ε 1e-5 (the `nn.LayerNorm` default), `silu(z) =
z·σ(z)`.

**Subsampling** (`NemotronAsrStreamingEncoderSubsamplingConv2D`, NeMo
`dw_striding` with causal downsampling). The mel frames form a one-channel
image, time × 128. Every 3×3 convolution has stride 2 on both axes and is
**causal on both axes**. The time axis gets 2 zero rows before and 1
after (`time_pad`). The frequency axis gets 2 zero columns before and 1
after (`freq_pad`). So output `(t, f)` reads padded rows `2t … 2t+2`,
which are input rows `2t − 2 … 2t`, and likewise for columns.

1. `conv_in` (1 → 256, bias), then ReLU: 128 → 65 frequency columns.
2. Twice: `depthwise_conv` (3×3 per channel, bias), then `pointwise_conv`
   (1×1, 256 → 256, bias), then ReLU. There is no activation between the
   two. Frequency 65 → 33 → 17.
3. Flatten row t **channel-major**, `c · 17 + f` (4352 = 256 × 17, the
   `subsampling_out_hidden_size` property). Then `linear`
   (4352 → 1024, bias). Gemma's flatten is frequency-major.

Rows: `T = f(f(f(N)))` with `f(n) = ⌊n/2⌋ + 1`
(`_get_subsampling_output_length`). One row is 80 ms, and row t depends
on mel frames `≤ 8t` only. `scale_input` is false, so there is no √1024.

**Each of the 24 blocks** (`NemotronAsrStreamingEncoderBlock`):

```
x += ½ · W2₁ silu(W1₁ LN_ff1(x))                 # feed_forward1
x += W_o · MHSA(LN_att(x))                        # self_attn
x += Conv(LN_conv(x))                             # conv
x += ½ · W2₂ silu(W1₂ LN_ff2(x))                 # feed_forward2
x  = LN_out(x)
```

The encoder output is block 23's `LN_out`; there is no further norm.

**Attention**: Transformer-XL relative attention with untied per-layer
biases (`NemotronAsrStreamingEncoderAttention`; NeMo `rel_pos`,
`pos_bias_u/v`, `linear_pos`). Per head h (8 heads of 128):
`q_i, k_j, v_j` from `q/k/v_proj` (no bias). With `u = bias_u[h]`,
`v' = bias_v[h]`, and `p_d` = `relative_k_proj · pe(d)` split per head:

```
s_ij = ((q_i + u)·k_j + (q_i + v')·p_{i−j}) / √128     for visible j, else −∞
out_i = Σ_j softmax_j(s_ij) v_j ;  then o_proj
```

The relative distance is `i − j` in absolute frames: positive for keys in
the past. This is what `_rel_shift` does: with C cached keys and a
positional table spanning `L − 1 … −(L − 1)` over `L = C + q` keys, the
shifted entry `(i, j)` is distance `C + i − j`. Softmax runs in F32.
Visibility is the chunk mask under
[Cache-aware streaming](#cache-aware-streaming). Every visible distance
lies in `[−r, 56 + r]`, so only 83 positional rows (r = 13) are ever
read. Like Gemma's relative keys, `p_d` is constant per layer.

**Positional table** (`NemotronAsrStreamingEncoderRelPositionalEncoding`):
`pe(d)[2m] = sin(d · ω_m)`, `pe(d)[2m+1] = cos(d · ω_m)`, `ω_m =
10000^(−2m/1024)`, m = 0…511. Sine and cosine are **interleaved**, where
Gemma's are halves. It is computed in F32 from a float position. HF
refuses more than `max_position_embeddings` = 5000 positions per forward,
so whole-clip input is limited to 400 s. A stream never comes near it.

**Convolution module** (`NemotronAsrStreamingEncoderConvolutionModule`),
on `z = LN_conv(x)`:

```
a | b = pointwise_conv1 · z          # 2048 → two halves of 1024
g     = a ⊙ σ(b)                     # GLU: first half gated by the second
y_t   = Σ_{k=0..8} w[c,k] · g_{t−8+k} # depthwise, causal, left pad 8, no bias
out   = pointwise_conv2 · silu(LN(y)) # conv.norm, then SiLU
```

**Language prompt** (`Nemotron3_5AsrForRNNT.get_audio_features`,
`Nemotron3_5AsrPromptProjector`). The prompt enters **after** the encoder,
per row. It is concatenated, not added, and it replaces the row rather
than joining a residual:

```
h'_t = linear_2 · relu(linear_1 · [h_t ; onehot₁₂₈(p)] + b₁) + b₂     # 1152 → 2048 → 1024
e_t  = encoder_projector · h'_t + b                                   # 1024 → 640
```

Since the prompt is one-hot, `linear_1`'s column `1024 + p` is a
per-language bias. The prompt touches no encoder state, so a stream can
change language between chunks without re-encoding. With no prompt given,
the model uses `default_prompt_id` 101 (`auto`).

## Cache-aware streaming

**The mask is chunked, not sliding** (`chunked_limited_mask_function`,
NeMo `att_context_style: chunked_limited`). Lookahead r gives chunks of
`r + 1` rows aligned at row 0, and `56/(r + 1)` left chunks
(`_resolve_attn_context`: left = `sliding_window − 1` = 56). Query i sees
key j when `0 ≤ ⌊i/(r+1)⌋ − ⌊j/(r+1)⌋ ≤ 56/(r+1)`: its own whole chunk
plus exactly 56 rows before it, `57 + r` keys in all. The whole-clip pass
uses the **same** mask: the processor sends `num_lookahead_tokens`
(default 3) to every forward. This checkpoint has no full-context mode.

| r | Rows per chunk | Left chunks | First chunk: mel / samples | Later chunks: mel / samples read / new audio | Latency (`(r+1)·80 ms`) |
| ---: | ---: | ---: | --- | --- | ---: |
| 0 | 1 | 56 | 1 / 200 | 8 / 1,680 / 1,280 | 80 ms |
| 3 | 4 | 14 | 25 / 4,040 | 32 / 5,520 / 5,120 | 320 ms |
| 6 | 7 | 8 | 49 / 7,880 | 56 / 9,360 / 8,960 | 560 ms |
| 13 | 14 | 4 | 105 / 16,840 | 112 / 18,320 / 17,920 | 1120 ms |

Mel counts come from `_required_stream_chunk_frames`: `1 + 8r` first,
`8(r + 1)` after. Sample counts are the processor's. Each chunk yields
exactly `r + 1` rows. The card's NeMo command also lists r = 1, and so
does the transformers processor's fallback `DEFAULT_NUM_LOOKAHEAD_TOKENS`.
This checkpoint's `att_context_size` does not, so nuclis accepts only
{0, 3, 6, 13}.

**State carried between chunks**, per stream (set up by
`NemotronAsrStreamingEncoder.forward` with `use_cache=True`):

- **Attention**: per layer, the keys and values (`k_proj`/`v_proj`
  outputs, before `bias_u`) of the last **56** rows
  (`DynamicSlidingWindowLayer` with `sliding_window` 57 keeps `−56:`
  after each update). The chunk attends to cache + itself, and the mask
  then admits all of them. The positional table spans `C + q` keys with
  `C = min(rows so far, 56)`. Size: 24 × 2 × 56 × 1024 F32 = 11.0 MB.
- **Depthwise convolution**: per layer, the last **8** GLU rows before
  the depthwise convolution (`NemotronAsrStreamingEncoderCausalConv1dCacheLayer`:
  `kernel − stride`; a chunk shorter than 8 keeps the older tail).
  Size: 24 × 8 × 1024 F32.
- **Subsampling**: per convolution (`conv_in`, `layers.0`, `layers.1`),
  the last **1** time row of its input (`NemotronAsrStreamingEncoderCausalConv2dCacheLayer`,
  `left_pad = kernel − stride`). That is a mel row (128 values), the stem's
  ReLU output (256 × 65), and the first layer's (256 × 33); the cached
  frequency padding is zeros.
- **Front end**: none in HF (it re-reads 400 overlapping samples per
  chunk); nuclis keeps the samples of open frames and the last sample.
- **Decoder**: LSTM `h, c` (2 × 640 each), the last `g` (640), and the
  rows decoded so far. Nothing resets between chunks. No symbol count
  crosses a chunk: a row is decoded to its blank or its tenth symbol as
  soon as it arrives, and every row of a chunk is final when the chunk is.

**The first chunk differs.** Its features are computed with
`center=True` (256 left zeros). Each subsampling convolution prepends one
extra zero row (`left_pad_init − left_pad`), giving the whole-clip pass's
two left zero rows. Its rows attend only to themselves (empty cache, as
chunk 0 in the whole clip). It is `1 + 8r` mel frames, because row t needs
mel frames `≤ 8t` and the first chunk ends at row r.

**Chunked equals whole-clip** at the same r, in exact arithmetic, when N =
`1 + 8r + 8(r+1)m`. In that case every convolution layer's input count is
odd, so the whole-clip pass never reads its right pad row. Every key is
valid. The cache holds exactly the rows the mask admits. Positions are
relative. The stream's rows are then the whole-clip rows, `(m + 1)(r + 1)`
of them. Frames are the same as long as the stream's mel frames are the
whole clip's, including the 256-zero tail at the end (step 3 above). For
any other N, the whole-clip pass ends with a short chunk. That chunk
attends only to its valid rows and reads one zero row of right padding
per convolution. HF's streaming refuses a short chunk ("pad the final
chunk"), and the card's example drops the tail.

## RNN-T decoding

**Prediction network** (`Nemotron3_5AsrRNNTDecoder`): `x = embedding[k]`
(row 13087, the blank, is zero), two `nn.LSTM` layers of 640, then
`decoder_projector` (640 → 640, bias). Per layer, with `z = W_ih x + b_ih
+ W_hh h + b_hh` split `[i | f | g | o]`: `c' = σ(f)c + σ(i)tanh(g)`,
`h' = σ(o) tanh(c')`. Layer 2's input is layer 1's `h'`. The start state
is zeros, and the first input is the blank (`decoder_start_token_id`
13087), so the first output is the LSTM of a zero vector from a zero
state; the state that step leaves (not zeros) is the one the first token
runs from.

**Joint** (`Nemotron3_5AsrRNNTJointNetwork`): `logits = head · relu(e_t +
g) + b_head`, 13,088 wide, where index 13087 is the blank. Greedy decoding
needs only the argmax, so no softmax is required.

**Greedy loop** (`ParakeetRNNTGenerationMixin._update_model_kwargs_for_generation`,
mirroring NeMo `greedy_batch`, `max_symbols: 10`):

```
t = 0; n = 0; g, state = pred(blank)          # zero input, zero state
while t < T:
    k = argmax joint(e_t, g)
    if k == 13087: t += 1; n = 0              # blank: next row, prediction state kept
    else:
        emit (k, t); g, state = pred(k, state); n += 1
        if n == 10: t += 1; n = 0             # the 10th symbol on one row forces an advance
```

On a blank the prediction network is not run (the decoder's
"all-blank fast path" returns the cached `g`). In streaming, the loop
pulls the next chunk when `t` reaches the rows encoded so far, so the
emitted tokens are the whole-clip ones whenever the rows are.

**Language.** The prompt id comes from `prompt_dictionary` (processor
`language`, NeMo `target_lang`), and the same id is applied to every
chunk. With `auto` (101), the card says the model emits a language tag
token (`<en-US>` …) after the transcript's terminal punctuation. It is
stripped by id: the 39 tag ids (below). The card's pipeline note says the
default prompt is index 0 (`en-US`), but the code's default is 101
(`config.json` and the warning in `get_audio_features`).

**Timestamps** (`Nemotron3_5AsrProcessor.decode` with `durations`, after
NeMo's `compute_rnnt_timestamps`): a token emitted at row t starts at
`t · 0.08 s` and ends at `(t + 1) · 0.08 s` (`hop/16000 · 8`).
Punctuation keeps its own frame.

## Tokenizer

`tokenizer.json` is a `tokenizers` **BPE** model: 13,087 pieces (ids
0…13086), 3,621 merges, `byte_fallback` true with no `<0xNN>` pieces,
`fuse_unk`. Its normalizer is `Precompiled` (NeMo's SentencePiece
`nmt_nfkc`), and its pre-tokenizer and decoder are `Metaspace` (`▁`,
`prepend_scheme: always`). There are 9,296 single-character pieces and
2,300 starting with `▁`; id 0 is `<unk>`, 1 `<bg-BG>`, 2 `▁`. HF converted
it (`ParakeetConverter`) from the `.nemo`'s SentencePiece model
`universal_merged_tokenizer_latest_40` (NeMo `type: bpe`, 13,087 pieces),
whose embedded trainer spec (UNIGRAM, vocabulary 256) is one merged
source's (`bg`), not the whole's. Transcription only maps ids to pieces,
so the encoding algorithm does not matter here.

Added tokens: `<pad>` 13087 and `<blank>` 13088 were appended by the
converter, and `<unk>` and 39 language tags (`<ar-AR>` … `<zh-CN>`,
ordinary pieces in NeMo, e.g. `<en-US>` 2947, `<de-DE>` 518) are marked
special. **The model's blank, 13087, is the tokenizer's `<pad>`**:
`ParakeetTokenizer._decode` drops it as padding. The tokenizer's
`<blank>` (13088, the processor's `blank_token_id`) lies outside the
joint's output and is never produced.

**Detokenization** (the `Metaspace` decoder): concatenate the pieces of
the non-special ids, replace `▁` with a space, and drop the leading
space. With `skip_special_tokens` the language tags and `<unk>` go too.
A tag has no `▁`, so if it is kept it attaches to the preceding
punctuation. Scripts without spaces (Chinese, Japanese, Thai) simply
concatenate.

## Front end and encoder against the trace (2026-10-27)

**The implementation.** `inference/src/audio/mel.zig` computes the front
end as `Spec.nemotron` ([audio.md § The front end](../engine/audio.md#the-front-end)),
in F64 with the products and pre-emphasis in F32, as for Gemma.
`inference/src/audio/nemotron_encoder.zig` is the CPU reference: `bind`
views the checkpoint's F32 tensors in place, `Encoder` holds the relative
keys of distances −13 … 69 per block, and `Stream` is the only path. A
whole clip is one stream fed at once and finished, so the chunked mask,
the caches, and the end of stream are one implementation. A stream keeps
the samples of open frames and the one before them, one input row per
subsampling convolution, each block's last 56 keys and values and last 8
GLU rows, and the subsampled rows of an unfinished chunk. Rows leave in
whole chunks of `r + 1`, and `finish` flushes as the whole-clip pass ends.
With synthetic weights, the unit tests show the authors' chunks
(`1 + 8r`, then `8(r + 1)` frames, at r = 0 and 3), pieces of 7 frames,
and samples in 80 ms pieces giving the whole clip's rows **bit for bit**.

**The trace.** `scripts/speech-traces/nemotron_trace.py` ran the authors'
`Nemotron3_5AsrForRNNT` (transformers `c0d024fb`, torch 2.14.1, librosa
1.0.0, CPU, F32) on two clips of the speech corpus, each cut to 80,880
samples (505 mel frames, 64 rows): a LibriSpeech test-clean utterance
under the prompt `auto` (101), and a FLEURS German one under `de-DE` (9).
At N = 25 + 32m frames both lookaheads end on a whole chunk. It pinned the
features, the subsampling's rows, and the rows after blocks 0, 11, 23 and
the prompt block at r = 0 and 3, from the whole clip and from the authors'
streaming path (the processor's chunk features, the encoder's caches).
Rows 0–7 and 56–63 are pinned. Rows 57–63 at r = 0, and 60–63 at r = 3,
are the first that no longer see row 0. The files, 2.8 MB, are in `tests/fixtures/nemotron-asr/` with
their digests in `index.json`.

The authors' stream gives their whole clip's features exactly. Their rows
differ only by F32 rounding: at most 1.6e-3 in the subsampling (values up
to 1.2e3), 1.8e-4 after block 0, 7.7e-7 after block 23, and 3.8e-5 after
the prompt block (`index.json`, `stream_minus_whole_max_abs`).

**Against it.** The gate `nemotron-asr-encoder-cpu`
(`zig build test-nemotron-asr -- MODEL_DIR`, 9.9 s with the build cached,
M4 Pro) runs the following. Each clip's WAV goes through the front end.
The pinned features go through the whole clip and through the authors'
chunks. The samples go in 80 ms pieces, which adds the front end's own
error. Each stage gets two figures: the largest difference over the
largest reference value ("max"), and the RMS of the difference over the
reference's ("RMS"). Worst over the two clips:

| Stage | r = 0 max | r = 0 RMS | r = 3 max | r = 3 RMS |
| --- | ---: | ---: | ---: | ---: |
| features (r-independent) | 1.5e-5 | 4.5e-7 | | |
| subsampling (r-independent) | 1.5e-6 | 1.2e-6 | | |
| whole, block 0 | 5.2e-7 | 1.6e-6 | 5.0e-7 | 1.6e-6 |
| whole, block 11 | 3.6e-7 | 9.7e-7 | 3.6e-7 | 9.4e-7 |
| whole, block 23 | 9.5e-7 | 1.2e-6 | 2.8e-7 | 4.8e-7 |
| whole, prompt block | 1.9e-6 | 3.1e-6 | 1.2e-6 | 1.8e-6 |
| chunked, block 0 | 2.3e-7 | 1.4e-6 | 5.2e-7 | 1.6e-6 |
| chunked, block 23 | 5.6e-7 | 8.3e-7 | 5.9e-7 | 8.0e-7 |
| chunked, prompt block | 1.5e-6 | 2.1e-6 | 1.6e-6 | 2.7e-6 |
| samples, block 23 | 4.3e-7 | 6.6e-7 | 3.3e-7 | 6.7e-7 |
| samples, prompt block | 1.1e-6 | 1.7e-6 | 1.4e-6 | 2.4e-6 |

The features' largest difference, 2.4e-4 on values up to 16.6, is in a
near-silent bin of the LibriSpeech clip. There `log(mel + 2⁻²⁴)` magnifies
the F32 STFT's rounding: the FLEURS clip's is 1.7e-5. The encoder's
differences are F32 summation order and do not grow through the 24
blocks. The bounds sit about five times above the worst: 1e-4 max and
5e-6 RMS on the features, 1e-5 and 2e-5 on every encoder stage
(`inference/nemotron-asr-check.zig`). A whole clip of 64 rows took 0.41 to
0.45 s on the CPU reference (ReleaseFast, M4 Pro).

**The speech corpus.** `make speech-corpus` (`scripts/speech-corpus.py`)
fetches the clips that `tests/fixtures/speech-corpus.json` pins into
`.reference/speech/`:

- 40 LibriSpeech test-clean utterances, one per speaker, the first of each
  speaker between 4 and 12 s;
- 5 FLEURS test utterances in each of German, French, Latin American
  Spanish and Russian, the first five sentences by id between 4 and 12 s.

Each clip is a member of its source archive: OpenSLR's
`test-clean.tar.gz`, or FLEURS's `test.tar.gz` per language at `70bb2e84`.
The JSON records the archive's URL and SHA-256 and the clip's own SHA-256
and transcript. The fetch streams each archive, keeps only the listed
members, stops once it has them, and refuses any clip whose digest
differs. Clips already present and correct are not fetched again.

**Found at bring-up.**

- At r = 0 the second chunk starts at sample `160 − 256 = −96`, before
  the clip. The processor's chunking then needs 96 leading zeros (the
  trace supplies them). nuclis never re-reads samples, so it has no such
  case.
- The last kept frame reads up to 40 samples past the clip when
  `L < 160⌊L/160⌋ + 40`. `finish` reads them as zeros, as the whole-clip
  STFT's padding does, so the decision under *End of stream* holds. The
  unit test with a 1.03 s clip reaches this case. The trace's clips end
  80 samples into a frame, so they do not.
- FLEURS stores its audio as 32-bit float WAV. `audio.zig`'s own RIFF
  reader took 16-bit PCM only, so those clips went through AudioToolbox,
  on macOS only; it reads float WAV too since the transducer's bring-up.
  The trace's cut clips are 16-bit. LibriSpeech's clips are FLAC, which
  only AudioToolbox reads.

## Transducer against the trace, and word error rate (2026-10-27)

**The implementation.** `inference/src/audio/nemotron_rnnt.zig` is the
transducer's CPU reference. `bind` views `encoder_projector.*`,
`decoder.*` and `joint.*` in place. `Decoder` holds the LSTM state, the
prediction `g` and the rows decoded. `push` projects a batch of encoder
rows (1024 → 640) and decodes each row in full: the joint's argmax, the
prediction network run on each emitted token, and the next row on the
blank or the tenth symbol. `reset` starts an utterance, the start blank
included. A row never waits for a later one, so rows pushed chunk by chunk
emit the whole clip's tokens. `nemotron_asr.zig` opens the directory:
`Model` holds the encoder, the transducer, the piece table
([tokenizer.md § Decode-only piece tables](../engine/tokenizer.md#decode-only-piece-tables)),
the `prompt_dictionary` and the language tag of each special id.
`Transcriber` is one utterance from samples to tokens, text and
language. `nuclis transcribe` drives it, whole or in 80 ms pieces. Unit
tests with synthetic weights cover the gate order `[i | f | g | o]`, the
blank and the ten-symbol cap, the start state, the tag pattern, and rows
in pieces against one push.

**The trace, extended.** `scripts/speech-traces/nemotron_trace.py` also
pins, for the same two cut clips at r = 0 and 3, whole and streamed: the
`encoder_projector` rows (0–7, 56–63); every step of `generate`'s greedy
decoding (the row read, and the joint's five largest logits with their
ids); the prediction network's output for the start blank; and each run's
token ids, rows and text. The regeneration reproduced every earlier array
byte for byte. The fixtures grew to 3.1 MB. The authors' own streamed
decode emits the same tokens as their whole clip. Their projector rows
differ by up to 1.6e-5.

**Against it.** The gate `nemotron-asr-trace-cpu`
(`zig build test-nemotron-asr -- MODEL_DIR --transducer`, 6.1 s with the
build cached, M4 Pro) decodes each clip twice. The first pass is the
pinned features through the encoder in one pass. The second is the
clip's samples in 80 ms pieces through a `Transcriber`. Every step's row
and argmax, and the final ids, rows and text, equal the trace's exactly
at both lookaheads in both modes: 92 and 95 steps for the LibriSpeech
clip (28 and 31 tokens), and 88 for the German one (24 tokens). The
values below are the worst over the two clips. "max" is the largest
difference over the largest reference value, "RMS" the relative RMS.
The logits run to about 1.1e3 in magnitude.

| Stage | whole r = 0 | whole r = 3 | 80 ms pieces r = 0 | 80 ms pieces r = 3 |
| --- | --- | --- | --- | --- |
| projector rows, max / RMS | 3.2e-6 / 2.4e-6 | 2.2e-6 / 1.4e-6 | 1.9e-6 / 1.3e-6 | 2.2e-6 / 1.9e-6 |
| logits at the trace's top five, max / RMS | 2.1e-6 / 1.1e-6 | 4.0e-6 / 2.0e-6 | 8.7e-7 / 3.8e-7 | 1.4e-5 / 6.8e-6 |
| margin (top 1 − top 2) vs the trace's, max abs | 1.4e-4 | 4.3e-4 | 2.4e-4 | 1.7e-3 |

The prediction network's start output is within 1.6e-7. The smallest
margins, ours and the trace's alike, are 0.060 and 0.247 (LibriSpeech,
r = 0, 3) and 0.015 and 0.011 (German). These are several times the
margins' own disagreement, so the exact argmax is not luck. The bounds
in `inference/nemotron-asr-check.zig` are 2e-5 on the projector and 1e-4
max / 5e-5 RMS on the logits. Each margin must lie within 1e-2 of the
trace's and the smallest must exceed that gap. The pinned transcripts are
those of the 5.05 s cuts: "Also a popular contrivance whereby love making may be
su" (r = 0) and "Also a popular can drivance whereby love making may be
suspende" (r = 3); "Die Romantik hatte ein starkes kulturell
deterministisches Element" (both). The whole clips through
`nuclis transcribe` (r = 3, `auto`): "Also a popular can drivance whereby
love making may be suspended but not stopped during the picnic season."
(`en-US`) and "Die Romantik hatte ein starkes kulturell
deterministisches Element, das sich aus Schriftstellern wie Goethe,
Fichte und Schlägel schöpfte." (`de-DE`).

**Speed (CPU, M4 Pro).** In the check tool (ReleaseFast), a 5.05 s clip
(64 rows) is encoded and decoded whole in 0.43–0.47 s; the transducer adds
little to the encoder's 0.41–0.45 s. Fed in 80 ms pieces it takes 1.33–1.38 s at
r = 0 (one-row chunks) and 0.73–0.75 s at r = 3. Through `nuclis
transcribe` (the ReleaseSafe build, whole clips, r = 3) the 60-clip corpus,
512.2 s of audio, transcribed in 64.5 s: 0.13× real time.

**Word error rate.** `scripts/wer.py` runs the fresh binary's
`transcribe --json --backend cpu` over corpus clips. Each clip runs under
its locale, or `--auto`. Both sides are normalized the same way: NFKC,
case folded, `ё` as `е`, apostrophes dropped, other punctuation a space.
For FLEURS the reference is its normalized transcript, what the speaker
said ("dass", a repeated "sie"), not the written sentence. The script sums
edits over clips before dividing. The whole corpus at r = 3 (`nuclis`
at this unit's `1f0be96`, 67 s wall time, one process per clip):

| Language | Clips | Words | WER, locale prompt | WER, `auto` | Card, FLEURS test, LangID / auto, 320 ms |
| --- | ---: | ---: | ---: | ---: | --- |
| English (LibriSpeech test-clean) | 40 | 828 | 3.86 % | 4.11 % | 8.27 / 8.84 (FLEURS, not LibriSpeech) |
| German (FLEURS) | 5 | 96 | 14.58 % | 14.58 % | 8.83 / 8.87 |
| French (FLEURS) | 5 | 97 | 15.46 % | 15.46 % | 9.79 / 9.84 |
| Spanish, Latin American (FLEURS) | 5 | 95 | 5.26 % | 5.26 % | 4.39 / 4.48 |
| Russian (FLEURS) | 5 | 95 | 8.42 % | 7.37 % | 9.87 / 11.01 |
| all | 60 | 1211 | 6.11 % | 6.19 % | |

The card's figures are on the whole FLEURS test sets, with its own
normalization. Ours are five sentences per language under ours, so they
are a different test set and not a comparison of accuracy. The German
and French errors are concentrated in two sentences each, mostly proper
names ("Cochamó", "Goethe, Fichte").

The gate `nemotron-asr-wer-cpu` runs six clips: three LibriSpeech, and
FLEURS German 1660, French 1661 and Russian 1660. Its first run had 10
errors in 123 words, 8.13 %. The bound is 11 %, three more errors. It
runs `./zig-out/bin/nuclis`, the Metal build, because LibriSpeech's FLAC
needs AudioToolbox. The transcription itself runs on the CPU.

**Language tags, observed.** A tagged sentence ends `.`, `▁`, `<de-DE>`:
the text then ends in a space, which `transcribe` trims. Five of the 40
LibriSpeech clips end without a tag under either prompt. Under an
explicit prompt the model still emits a tag, the prompt's own locale, in
each of the other 55 clips. Under `auto`, the Latin American Spanish clips are tagged
`<es-ES>`, and under `es-US` they are tagged `<es-US>`.

## On Metal (2026-10-27)

**What runs where.** `inference/src/audio/nemotron_metal.zig` holds a
`Plan` (the device weights and the scratch its streams share) and a
`Stream` (one utterance). The front end stays on the host
(`nemotron_encoder.FrontEnd`, shared with the CPU `Stream`): it writes mel
frames straight into the stream's first device queue. Everything after
it runs on the device: the three subsampling convolutions, the 24 blocks,
the prompt block, `encoder_projector` and the joint. Only the prediction
network's LSTM runs on the CPU (`nemotron_rnnt.Decoder.predict`): two
layers and a projection, about 28 MB of weights per emitted token,
measured below at a few percent of a chunk. Matrices are the mapped F32
bytes, wrapped in place, as published. Two are copies: the subsampling's
linear with its columns permuted to `[column][channel]` (so it reads the
last convolution's rows as they are), and `prompt_projector.linear_1`'s
first 1024 columns (its one-hot column joins the bias, per stream and
prompt). The relative keys are the CPU encoder's, uploaded once.

**A stream's state lives on the device.** The bookkeeping is the CPU
`Stream`'s, run on the host so a whole push is recorded before it is
committed:

- each subsampling convolution keeps its input rows from `2·next − 2` on
  (one or two), shifted to the queue's start after each batch;
- subsampled rows wait in a pending buffer until a whole chunk of `r + 1`
  is there;
- per block, the keys and values of the last 56 rows and the last 8 GLU
  rows, each followed by room for a batch; after a batch one dispatch
  (`asr_shift`) moves every block's last 56 keys and values, and another
  every block's last 8 GLU rows, to the front.

By size, a stream's device buffers take 72 MB (the key, value and GLU
caches 50 MB); the plan creates 51 MB beside the 2.55 GB of mapped
weights. A push is one command buffer. It holds the subsampling of the new frames,
every whole chunk through the blocks, the prompt block, the projector,
and the first batch of joint logits. `finish` ends as the CPU stream
does, so a whole clip is one stream fed at once. With 512 mel frames per
batch, a batch runs at most 80 rows through the blocks: the chunks of a
whole clip share each read of the weights. At the stream's pace, a chunk
is a batch.

**Greedy decoding** (`Stream.decode`) keeps `Decoder.decodeRow`'s loop on
the host and speculates on the joint. One prediction `g` scores up to 16
rows at once (a 13,088 × 640 product over those rows), valid until the
next emitted token changes `g`. A blank moves to the next row's logits,
already computed. A token runs the LSTM on the CPU, and the next row needs
a new joint batch. So a chunk with no token is one command buffer, and
each token adds one.

**Kernels** (`kernels/asr.metal`, `ops/asr.zig`), each a `make
test-metal` case against the CPU function it mirrors (the CPU encoder's
were factored out for this, its rows bit for bit as before):

- `asr_conv2d`: a subsampling convolution, the stem or a depthwise one,
  rows from a queue, with bias, and ReLU for the stem. The pointwise
  1×1 that follows is the generic tile, then `asr_bias_relu`.
- `asr_attention`: the Transformer-XL scores `(q + u)·k + (q + v)·p`
  over the cached keys and the chunk's under the chunked mask. One SIMD
  group per (row, head), with the softmax computed online.
- `asr_glu`, `asr_depthwise` (9 taps over the held rows and the new
  ones), `asr_joint` (`relu(e + g)`), `asr_add_scaled` (the macaron
  half step), and `asr_shift` (the caches' row moves; each thread copies
  one column in increasing row order, so an overlapping move toward the
  start is safe).
- `asr_matmul_rows_t{1,2,4,8}`: a few-rows F32 product. Each weight row
  is read once for up to 8 activation rows (a chunk has 1 to 7). It takes
  up to three matrices of one shape in one dispatch (q, k, v), and its
  epilogue stores, applies SiLU, adds to the residual (`x += ½·W·h`), or
  forms the GLU from rows `r` and `r + 1024`. Batches of more than 8 rows
  (a whole clip) take the generic 32×32 F32 tile instead.

**`nuclis transcribe`** runs on Metal by default when built with it, `--backend cpu` on the reference. `Transcriber.init` takes the engine (`Model.openMetal` builds the
plan once), and both engines give the same tokens. `scripts/wer.py`
passes `--backend` through.

**Against the trace.** The gate `nemotron-asr-trace-metal`
(`zig build test-nemotron-asr -- MODEL_DIR --metal`) runs the
`--transducer` check on the plan and adds the encoder's stages. Each clip
runs at r = 0 and 3, from the pinned features in one pass and from the
samples in 80 ms pieces. Every greedy step's row and argmax, every token
id, row and text equals the trace's, in all eight runs. Worst over the
two clips, relative to the largest reference value (max) and as relative
RMS:

| Stage | max | RMS | bound |
| --- | ---: | ---: | --- |
| subsampling | 1.6e-6 | 1.2e-6 | 2e-5 |
| blocks 0, 11, 23 | 1.2e-6 | 1.6e-6 | 2e-5 |
| prompt block (the encoder's output) | 2.4e-6 | 3.7e-6 | 2e-5 |
| projector rows | 4.1e-6 | 2.9e-6 | 2e-5 (the CPU's) |
| logits at the trace's top five | 8.0e-6 | 3.9e-6 | 1e-4, 5e-5 (the CPU's) |
| margins vs the trace's, max abs | 9.2e-4 | | 1e-2 |

The logits differ from the trace's by up to 7e-3 in absolute terms (they
run to 1.1e3). Their margins, which decide the argmax, agree within
9.2e-4, against a smallest margin of 0.0107. These are the CPU
reference's magnitudes or smaller. Whole clips (batches of 64 rows) take
the 32×32 tile, the 80 ms pieces the few-rows product: both are in the
table. The check also resets each streamed transcriber and feeds the clip
again in 1000-sample pieces (ending mid-frame), which gives the traced
tokens and rows again.

**Word error rate.** The gate `nemotron-asr-wer-metal` runs the whole
corpus through `nuclis transcribe --backend metal` at r = 3, each clip
under its locale. Every one of the 60 transcripts is word for word the
CPU's. The rates are therefore the CPU's:

| Language | Clips | Words | WER, Metal | WER, CPU |
| --- | ---: | ---: | ---: | ---: |
| English (LibriSpeech test-clean) | 40 | 828 | 3.86 % | 3.86 % |
| German (FLEURS) | 5 | 96 | 14.58 % | 14.58 % |
| French (FLEURS) | 5 | 97 | 15.46 % | 15.46 % |
| Spanish, Latin American (FLEURS) | 5 | 95 | 5.26 % | 5.26 % |
| Russian (FLEURS) | 5 | 95 | 8.42 % | 8.42 % |
| all | 60 | 1211 | 6.11 % | 6.11 % |

The bound is 7 %, eleven errors more than the first run. The 512.2 s of
audio took 12.2 s of transcription on Metal (0.024× real time, each clip
a fresh process, whole clips) and 71.5 s on the CPU.

**Speed.** `nemotron-asr-check MODEL_DIR [--metal] --bench --clip FILE`
(ReleaseSafe, Zig 0.17.0, M4 Pro 48 GB) times each clip after a warm-up
pass, three rounds each. It runs the whole clip at once. Then it feeds
80 ms pieces twice, back to back and paced at real time (each piece
pushed when its audio would have ended), and times every push. A
*chunk push* is one that completes a chunk and runs the encoder; the
*final delay* is the last push plus `finish`, the time from the end of
the audio to the final transcript. Measured on this unit's branch on
2026-10-27, while another worktree ran Metal benchmarks on the same GPU.
Repeated runs moved the medians by up to 1.5×, so these are a shared
machine's figures; the main session re-measures them on a quiet one.
A LibriSpeech clip of 11.70 s (`6829-68769-0000`, prompt `auto`):

| | Metal r = 0 | Metal r = 3 | CPU r = 0 | CPU r = 3 |
| --- | ---: | ---: | ---: | ---: |
| whole clip | 196 ms (0.017×) | 235 ms (0.020×) | 1,432 ms (0.122×) | 1,480 ms (0.127×) |
| chunk push, median / p90, back to back | 13.8 / 15.3 ms | 21.3 / 24.7 ms | 24.4 / 25.4 ms | 55.7 / 59.3 ms |
| chunk push, median / p90, at real time | 17.2 / 24.7 ms | 35.3 / 46.1 ms | 25.0 / 26.9 ms | 54.2 / 57.2 ms |
| other pushes, median (back to back) | — | 0.53 ms | — | 1.30 ms |
| streamed, busy time over audio (back to back) | 0.173× | 0.073× | 0.303× | 0.187× |
| final delay, median, back to back / at real time | 14.5 / 14.5 ms | 24.8 / 33.2 ms | 24.4 / 24.6 ms | 52.6 / 51.9 ms |

A FLEURS German clip of 10.38 s (`1191248392415945452`, prompt `auto`)
ran under heavier load. On Metal: whole clip 189 ms at r = 0 and 175 ms
at r = 3 (0.018× and 0.017×). The chunk push had a median of 17.7 and
24.1 ms back to back, and 25.5 and 36.9 ms at real time. The final delay
was 38.8 and 36.0 ms back to back: this clip ends inside a chunk, so
`finish` runs one more command buffer. On the CPU: whole clip 1,235 and
1,311 ms; chunk push 25.3 and 54.5 ms; final delay 51.0 and 53.1 ms.

**Where a chunk's time goes.** At r = 3 one round of the LibriSpeech
clip spent, back to back, 748 ms in the encoder's command buffers (37
chunks, each with its first joint batch, and the subsampling-only pushes
between them), 31 ms in later joint batches and 37 ms in the LSTM. The
joint's later batches and the LSTM are 8 % of the time, so the LSTM
stays on the CPU.
Each dispatch timed on its own (`Backend.enableProfiling`, which adds an
encoder boundary per dispatch), a round of the clip at r = 3 spent most
of its 720 ms in the products, the 2.5 GB of F32 weights a chunk reads:
the feed-forward matrices 352 ms (150 to 200 GB/s), q, k and v 78 ms,
`o_proj` and the convolution module's two 109 ms. LayerNorm took 62 ms
(144 dispatches a chunk), the attention 36 ms, and everything else about
75 ms together.
At about 230 GB/s the weights alone take about 11 ms a chunk. In one run
over the trace's clips, the product read its weights about as fast for
four rows as for one: 180 to 195 GB/s on the feed-forward matrices at
r = 3 and at r = 0. So an r = 3 chunk costs the device about what an
r = 0 chunk does, for four times the audio. The rest of its wall time is
its tokens' joint batches and LSTM steps, and the shared GPU.

Against the proposed target of 15 ms per r = 3 chunk: back to back the
chunk takes 17 to 25 ms across runs on this shared machine. Paced at real
time it takes 32 to 39 ms, because the GPU and CPU clock down in the 320
ms between chunks. At r = 0, with a chunk every 80 ms, the paced chunk
takes 1.2 to 1.5× the back-to-back one. What is left is in the products'
bandwidth (the few-rows kernel reaches 200 to 230 GB/s on the large
matrices when the GPU is not shared, less on the 1024-row ones) and in
about 220 small dispatches a chunk beside the 190 products (six
LayerNorms a block, the attention, the depthwise convolution, a SiLU). The next steps, measured first: LayerNorm folded into the
product that follows it, and fewer, larger dispatches for the 1024-row
matrices.

## Decisions and open items

Decided at the plan (2026-10-27); each is checked at bring-up:

- **End of stream** is flushed as the whole-clip pass ends: zero samples
  past the end for the open frames, one zero row of right padding per
  subsampling convolution, attention limited to the valid rows, so the
  rows equal a whole-clip pass on the same audio. The authors' streaming
  code does not define it (HF refuses short chunks; the card's example
  drops the tail); a trace confirms the equality.
- **State lives as long as the stream.** Nothing in HF resets the decoder
  or the encoder caches between utterances; nuclis carries both for the
  life of a stream and clears both when the client asks (a reset message
  on the streaming route, sent by a client after each final transcript).
- **Default language `auto`** (101), the code's default; a client names a
  locale to fix it. Language tags are stripped from the text by id and
  reported beside it. (The card's pipeline note says `en-US`.)
- **No dither**, as HF: transcription is deterministic. NeMo's config
  has `dither: 1.0e-05`; whether NeMo applies it at inference was not read.

Open, observed at bring-up rather than decided:

- **Tags in a long stream.** The card describes one tag after the
  terminal punctuation; whether `auto` emits one per sentence in a
  continuous stream is not settled by the code. A tag does appear under
  an explicit language (observed on 55 of 60 clips:
  [Transducer against the trace](#transducer-against-the-trace-and-word-error-rate-2026-10-27)).
- **Tag ↔ locale.** The tag `<sl-SL>` has no prompt key (Slovenian is
  `sl-SI`, 62); `mt-MT` (102) has a prompt but no tag, so what `auto`
  emits for Maltese is unknown.
- **Precision.** HF computes the STFT and the mel product in F32 with a
  librosa F32 filter bank, the encoder in F32; `mel.zig` works in F64 and
  casts. Measured against the trace:
  [Front end and encoder against the trace](#front-end-and-encoder-against-the-trace-2026-10-27).
- **Metal chunk latency.** The r = 3 chunk is above the proposed 15 ms on
  a shared machine, and twice its back-to-back time when paced at real
  time, as clocks drop between chunks
  ([On Metal](#on-metal-2026-10-27)). On Metal a whole clip and the same
  clip streamed are not bit for bit equal: batches of more than 8 rows
  take the 32×32 tile, chunks the few-rows product. Both match the trace's
  tokens.
