# Audio through the companion encoders

How nuclis turns a clip into embedding rows: the decoder, the log-mel front
end, and Gemma 4's audio encoder (`gemma4a`) on the CPU and on Metal. Today
EmbeddingGemma 2 is the only family that uses it
([embeddinggemma.md § Audio](../models/embeddinggemma.md#audio-2026-10-09)).
Gemma 4 E4B carries the same encoder, and chat still refuses audio.
Everything here was written from Google's code at the pinned revisions
(transformers 5.19.0: `feature_extraction_gemma4.py`, `modeling_gemma4.py`
`Gemma4Audio*`, `processing_embedding_gemma2.py`). llama.cpp's
`tools/mtmd/models/gemma4a.cpp` and `clip.cpp` (build `b11514`) were read
as a cross-check, never copied.

**Contents.**

- [The package](#the-package)
- [Decoding](#decoding)
- [The front end](#the-front-end)
- [The encoder](#the-encoder)
- [What the file does not say](#what-the-file-does-not-say)
- [Metal](#metal)
- [Against the oracles (2026-10-09)](#against-the-oracles-2026-10-09)

## The package

`inference/src/audio/`:

- `audio.zig`: `decode` to 16 kHz mono f32 (`Pcm`, with the clip's whole
  length when it was cut). A RIFF/WAVE parser reads 16-bit mono 16 kHz PCM
  on any platform. Everything else goes through `audio_bridge.m`, built
  with the Metal bridge. Encoded bytes are bounded at 64 MiB.
- `mel.zig`: `Bank` (the window, the filter bank, the FFT twiddles) and
  `features`. Pure, no I/O.
- `gemma4a.zig`: `bind` (the companion file's `a.*` and `mm.a.*`
  tensors) and `Runtime`, the CPU reference. It also holds the pieces the
  Metal plan shares: `subsample`, `relativeKeys`, `queryScale`.
- `gemma4a_metal.zig`: `Plan`, the same schedule on the device.

## Decoding

AudioToolbox reads the bytes through `AudioFileOpenWithCallbacks` and
`ExtAudioFileWrapAudioFileID`. The formats are WAV, AIFF, CAF, MP3,
M4A/AAC, FLAC and ALAC. The client format is 16 kHz float in the file's
channel count, and the converter's sample-rate quality is set to its
maximum (`Mastering` complexity). Channels are averaged afterwards, as
librosa's `mono=True` does. A 16 kHz mono 16-bit WAV never reaches the
bridge: each sample over 32768 is what every decoder gives, so it is exact
on any platform.

Other clips match Google's only as far as two resamplers agree. On the
`north` clip, a 44.1 kHz stereo copy gives a vector at cosine 0.9999 to the
16 kHz original, and an AAC copy 0.998 (the codec's own loss).

## The front end

Google's `Gemma4AudioFeatureExtractor`, with the pinned
`processor_config.json`:

1. Pad the clip on the left with `frame_length / 2` = 160 zeros.
2. Frame i covers padded samples `[160 i, 160 i + 320)`. It is kept only
   when its last sample falls inside the clip (the extractor's mask), so a
   clip of L samples gives `⌈(L − 160) / 160⌉` frames: 4.8 s gives 479.
3. Multiply by the periodic Hann window, `np.hanning(321)[:320]`, in F32.
4. Take the magnitude (not the power) of a 512-point real FFT: 257 bins.
5. Apply 128 triangular HTK mel filters over 0–8000 Hz, unnormalized. The
   filter edges are equally spaced in mel (`2595 · log10(1 + f/700)`) and
   the bins equally spaced in hertz.
6. Take `log(mel + 1e-3)`. There is no preemphasis, no dither and no
   per-bin normalization.

`mel.Bank.features` does this in F64 with an iterative radix-2 FFT and
casts to F32. Against the extractor's own features
(`tests/fixtures/embeddinggemma-audio-features/`) the largest difference
is 4.8e-7 on all three clips.

**The 30 s bound.** The extractor's call has `max_length = 480000` and
`truncation = True`, so Google's pipeline cuts every clip at 30 s (2,999
frames, 750 rows). The embedder takes the same bound. A longer clip is
refused unless the request truncates, which keeps its first 30 s and
reports the cut.

## The encoder

Rows per clip: two stride-2 halvings, `⌈⌈frames / 2⌉ / 2⌉` (479 frames
give 120 rows, one per 40 ms). The processor counts its placeholder tokens
the same way.

1. **Subsampling.** The frames form a one-channel image (time × 128 mel).
   Two `Conv2d` layers follow (3×3, stride 2, padding 1, no bias; 1 → 128
   → 32 channels), each followed by a LayerNorm over its channels (weight,
   no bias, ε 1e-6) and a ReLU. The output row is frequency-major,
   `f · 32 + c` (32 frequency columns × 32 channels = 1024). Then
   `a.input_projection` (F32, 1024 → 1024).
2. **12 conformer blocks.** RMS norms (ε 1e-6, with a weight) throughout:
   - **First half FFN:** `x += ½ · post(W_down · silu(W_up · pre(x)))`.
   - **Attention:** `q, k, v = W · pre(x)`. Then
     `q ·= 128^−½ / ln 2 · softplus(per_dim_scale)` and
     `k ·= ln(1 + e) / ln 2`. Query i sees keys `i − 11 … i`. Each score
     is `q·k_j + q·relk[12 − (i − j)]`, capped by `50 · tanh(s / 50)`, then
     softmaxed. The relative keys are `attn_k_rel` times 13 sinusoid rows:
     distances 12 down to 0, sines then cosines over 512 timescales from 1
     to 10⁴. Finally `x += post(W_out · attention)`.
   - **Light convolution:** `x += W_pw2 · silu(post(dwconv(glu(W_pw1 ·
     pre(x)))))`. The GLU keeps the first half times the sigmoid of the
     second. The depthwise convolution is causal with 5 taps:
     `y[t] = Σ_k w[k] · g[t − 4 + k]`.
   - **Second half FFN**, as the first.
   - **Out norm.**
3. **Output.** `a.pre_encode.out` (1024 → 1536, with a bias), then the
   embedder: an RMS norm without a weight and `mm.a.input_projection`
   (1536 → the text width).

Every attention, FFN and pointwise-convolution matrix is a **clipped
linear**: `y = clamp(W · clamp(x, in), out)`, with the four F32 scalars
from the file (±20 to ±35 here, so they bind). Q, K and V clamp their
shared input to different bounds, so each clamps its own copy. The
`gradient_clipping` clamps in Google's code are ±1e10, so they never bind
in F32, and nuclis leaves them out.

## What the file does not say

Two facts were settled by comparing the GGUF's values with the
safetensors, not by name:

- **The light convolution's norms are swapped.** The file's
  `a.blk.N.conv_norm` holds Google's `lconv1d.pre_layer_norm`, and its
  `norm_conv` holds the norm after the convolution. llama.cpp's `clip.cpp`
  loads them in reverse for the same reason (its comment blames
  `tensor_mapping.py`). Read by name, they cost a relative RMS of 0.72 on
  the rows.
- **`per_dim_scale` is stored after `softplus`.** The file holds
  `softplus(p)` and the checkpoint holds `p`; they agree to 6e-9.

And one fact about the mask: Google's `sliding_window_mask_function` keeps
distances `0 ≤ d < 12`, so a query sees 11 rows back and itself, although
each 12-row block is given 12 rows of left context and 13 relative rows.
Seeing 12 rows back costs a relative RMS of 3.8e-2 on block 0's attention.

## Metal

`gemma4a_metal.Plan` runs the subsampling convolutions on the host
(`gemma4a.subsample`, about 0.2 GFLOP for 30 s) and everything after them
on the device. The pieces:

- **Matrices** are the mapped BF16 and F32 bytes on the generic F32 matmul
  tile, with F32 activations, as the text plan's.
- **Two new kernels**, each tested against the CPU reference in
  `zig build test-metal`:
  - `audio_attention`: one SIMD group per (row, head), four dimensions a
    lane. It scales q and k as it reads them and softmaxes the 12 visible
    scores.
  - `glu_conv`: the GLU fused into the depthwise convolution.
- **The half step** is folded into the post-norm weights: a host copy
  times ½, through `rmsNormAdd`.
- **Clamps:** a clamped input that another matrix also reads goes through
  a copy.

The relative keys are constant per block, so they are computed once at
init on the host. Buffers are sized for 750 rows. On the BF16 file the
plan matches the CPU reference's vectors to the same `1 − cos` against
Google (about 1e-12).

## Against the oracles (2026-10-09)

`zig build test-embeddinggemma -- MODEL audio --mmproj mmproj-BF16.gguf`
runs four checks:

- the three clips' mel frames against the extractor's (bound 1e-4);
- the `north` clip's encoder rows against llama.cpp's `media-0`, as a
  localizer;
- each clip through the whole pipeline against Google's float32 vector;
- on Metal, a packed batch against one at a time.

Every stage of the CPU reference also matched a hook dump of Google's
tower (subsampling, each block, the output, the projection) to relative
RMS ≤ 1.8e-6. `scripts/audio-trace.py` writes that dump to
`.zig-cache/audio-trace/`, and the `audio` mode traces the CPU reference
against it stage by stage whenever it is there: the localizer that found
the swapped norms and the mask width. Results are in
[embeddinggemma.md § Audio](../models/embeddinggemma.md#audio-2026-10-09).
