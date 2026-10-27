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
- **Decoder**: LSTM `h, c` (2 × 640 each), the last `g` (640), the frame
  pointer and symbol count (below). Nothing resets between chunks.

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
state.

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
  continuous stream, and whether a tag can appear under an explicit
  language, is not settled by the code.
- **Tag ↔ locale.** The tag `<sl-SL>` has no prompt key (Slovenian is
  `sl-SI`, 62); `mt-MT` (102) has a prompt but no tag, so what `auto`
  emits for Maltese is unknown.
- **Precision.** HF computes the STFT and the mel product in F32 with a
  librosa F32 filter bank, the encoder in F32; `mel.zig` works in F64 and
  casts. The bounds against HF's features and encoder rows are measured
  at bring-up, not assumed.
