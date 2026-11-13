<!-- kind: card -->
# Qwen3-TTS 12Hz: CustomVoice, Base, and VoiceDesign (0.6B and 1.7B)

[Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)
(Qwen, Apache-2.0) is a text-to-speech model. In nuclis it is the voice of
a local voice assistant: `nuclis speak` and `POST /v1/audio/speech`
([api-speech.md](../guide/api-speech.md)). Five catalogue entries, all
as published (BF16 talker, F32 codec), revisions and digests in
`src/catalog.json`:

| Name | Voice | Instruction |
| --- | --- | --- |
| `qwen3-tts-0.6b-customvoice` | nine built-in speakers | no |
| `qwen3-tts-1.7b-customvoice` | nine built-in speakers | yes |
| `qwen3-tts-0.6b-base`, `qwen3-tts-1.7b-base` | cloned from a reference clip | no |
| `qwen3-tts-1.7b-voicedesign` | described by the instruction | yes (the voice) |

Three networks in two files: the **talker**, a 28-layer Qwen3-style
decoder over a 3,072-id codec vocabulary that predicts code group 0 of
each 80 ms frame; the **code predictor**, a 5-layer decoder run 15 steps a
frame for groups 1–15; and the **codec decoder** (`speech_tokenizer/`),
which turns frames of 16 codes into 24 kHz mono PCM, 1,920 samples a
frame. The Base adds a speaker encoder and the codec's encoder. The format
as read from the authors' package, tensor by tensor:
[record, 2026-10-27](../benchmarks/record/qwen3-tts.md#forward-pass-from-modeling_qwen3_ttspy-2026-10-27).

## Architecture

| | 0.6B | 1.7B |
| --- | --- | --- |
| Talker | 28 layers, width 1024, FFN 3072 SiLU | 28 layers, width 2048, FFN 6144 |
| Talker attention | 16 query / 8 KV heads of 128, Q/K RMSNorm, no biases, full causal | the same |
| Talker RoPE | `mrope_section` [24, 20, 20] with three equal position components: plain NEOX RoPE, base 1e6, one position per frame | the same |
| Code predictor | 5 layers, width 1024, 1-D RoPE at positions 0–15, a fresh cache each frame | the same, its 15 codec tables talker-wide and `small_to_mtp_projection` (2048 → 1024) on every input row |
| Input row | the sum of a text track (`text_embedding` 2048 wide, then a 2-layer SiLU MLP) and a codec track; never concatenated | text track projects to 2048 |
| Next frame's row | the 16 groups' embeddings summed, plus the trailing text row (or `tts_pad`) | the same |
| Codec decoder | split RVQ (16 codebooks of 2048), causal conv, 8-layer transformer (16 × 64 heads, a 72-key sliding window, layer scale), two ×2 ConvNeXt upsamplers, four SnakeBeta blocks (×8, 5, 4, 3) | byte-identical file |

Norm weights are applied raw (`weight * x`); `position_id_per_seconds` is
read by nothing. The codec decoder is causal: a stream carrying each
convolution's history, 71 keys and values per transformer layer, and the
frame count gives the whole-clip decode, bit for bit however the frames
are grouped. Past 300 frames nuclis keeps decoding causally rather than
re-warming every 300 frames as the authors' batch decode does
([record, 2026-10-27](../benchmarks/record/qwen3-tts.md#open-for-the-implementation-2026-10-27)).

### Voice cloning (the Base)

A Base voice is a speaker embedding and, with a transcript, the clip's
codes; `nuclis voice add` computes them once and saves them.

- **Speaker encoder** (`inference/src/audio/ecapa.zig`, CPU only): a
  24 kHz log-mel (1024-point FFT, hop 256, reflection padding, Slaney
  filters) into an ECAPA-TDNN (three SE-Res2Net blocks, attentive
  statistics pooling) and `fc` to the talker's width. Its embedding is the
  prompt's speaker row.
- **Codec encoder** (`inference/src/audio/qwen3_codec_encoder.zig`, CPU
  only): Mimi's causal SEANet, an 8-layer transformer (plain causal; the
  config's 250-frame window is not applied by the authors' attention), a
  stride-2 downsample to 12.5 Hz, and the first 16 of 32 quantizers. Its
  quantizer tensors must equal the decoder's byte for byte, or `bind`
  refuses the file.
- **Two modes.** Clip alone (x-vector): one speaker row, whatever the
  clip's length. Clip and transcript (ICL): the reference frames join the
  prompt with the reference text, and the decoder is primed with them and
  their samples discarded, so the first audio grows with the clip.

Against the authors' trace: [speaker encoder](../benchmarks/record/qwen3-tts.md#the-speaker-encoder-against-the-trace-2026-10-29), [codec encoder](../benchmarks/record/qwen3-tts.md#the-codec-encoder-against-the-trace-2026-10-29).

### VoiceDesign

The 1.7B CustomVoice's tensors (other values) with no speakers:
`parseConfig` accepts `voice_design`, and the prompt is CustomVoice's
without a speaker row, the instruction describing the voice. A designed
voice is sampled anew each time; to keep one, save a clip of it and clone
it on the 1.7B Base.

## Files and encodings

A checkpoint directory: `config.json`, `generation_config.json`,
`vocab.json`, `merges.txt`, `model.safetensors` (talker and predictor,
BF16: 402 tensors on the 0.6B, 404 on the 1.7B, 76 more under
`speaker_encoder.` on a Base) and `speech_tokenizer/model.safetensors`
(the codec, F32, 496 tensors, shared byte for byte by all five). The
weights run as published, never requantized
([spec § 3](../spec.md#3-decisions)).

- `parseConfig` takes `tts_model_type` `custom_voice`, `base`, or
  `voice_design` (anything else is `UnsupportedModelType`) and
  `generation_config.json` as written; a key it lacks keeps the authors'
  fallback, and `do_sample` false makes that network greedy.
- `bind` checks every tensor by name and shape; an unknown one is
  `UnexpectedTensor`, named; `small_to_mtp_projection` is required exactly
  when the predictor is narrower than the talker.
- The CPU reference decodes every tensor but the text embedding to F32 at
  load (2.4 GB for the 0.6B, 6.4 GB for the 1.7B); text rows are widened
  from the mapped BF16 file per id. Metal wraps the BF16 matrices in place
  (1.8 and 3.8 GB) and copies the codec's F32 matrices (543 MB).
- Tokenizer: byte-level BPE from `vocab.json` and `merges.txt` with the
  case-seam split the authors' pinned `transformers` applies
  ([tokenizer.md § Vocabulary and merges files](../engine/tokenizer.md#vocabulary-and-merges-files)).

## Prompt profile

No chat profile (`profile` null in the catalogue). `qwen3_tts.encodeText`
renders the authors' texts and slices them by position as they do:
`<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n`, the
instruction as `<|im_start|>user\n{instruct}<|im_end|>\n`, a Base's
reference as `<|im_start|>assistant\n{ref}<|im_end|>\n`. `Prompt.build`
lays out the rows: the role, then the codec prefix (`think`, `think_bos`,
the language, `think_eos`, or `nothink` … for `auto`; a dialect speaker
swaps in its dialect), the speaker, then `tts_bos`; CustomVoice and
VoiceDesign prefill the whole text (non-streaming), a Base sums it into
the generated rows. `Prompt.build` refuses a Base (`VoiceRequired`),
`buildClone` a CustomVoice (`CloningUnsupported`), and an instruction to a
model without one is `InstructionUnsupported`.

Sampling is the authors': group 0 with repetition penalty 1.05 over the
generated codes, the end code (2150) blocked for two frames and ids
2048–3071 suppressed, temperature 0.9, top-k 50; groups 1–15 temperature
0.9, top-k 50, no penalty. A seed gives nuclis's own stream of draws, not
torch's.

### The frame budget

A model that never samples the end code would run to `max_new_tokens`
(8,192 frames, 655 s). Every synthesizer stops at
`min(max_new_tokens, budget_floor_frames + budget_frames_per_token ×
text tokens)`, 50 + 20 per token of the target text, in
`qwen3_tts_runtime.zig`, and `Synthesizer.finish` says why it stopped
(`eos`, or `length` at the budget). `nuclis speak` warns at `length`;
`nuclis serve` reports it as `finish_reason` in `speech.audio.done` and
the request log. Why the margin is that wide:
[record, 2026-10-29](../benchmarks/record/qwen3-tts.md#the-frame-budget-2026-10-29).

### On Metal

`qwen3_tts_metal.Plan` runs the talker and predictor on the BF16 weights,
the talker's cache F16 (or F32) and grown on demand to 16,384 rows.
`Plan.frameOnDevice` records a whole frame in one command buffer,
sampling groups 1–15 on the device (`nu_tts_sample`) with the uniforms
the host sampler would draw; the host then decides every group again from
the same logits and its code stands where they differ, so the codes are
always the CPU sampler's (`--host-sampling` writes the same WAV). The
codec decoder (`qwen3_codec_metal.zig`) runs the CPU stream's schedule a
piece of at most 8 frames per command buffer. The first chunk is one
frame (`default_first_chunk`), then each chunk at most the frames made
before it. Between `read` calls no command buffer is in flight, so a
request is cancelled by calling `deinit`; one synthesizer at a time per
voice.

## Verified

<!-- gen:gates-for:qwen3-tts -->
| Gate | Tier | Model | Evidence |
| --- | --- | --- | --- |
| `qwen3-tts-codec-cpu` | `verify-cpu` | qwen3_tts_06b | [qwen3-tts.md § codec-decoder-against-the-trace-2026-10-27](../benchmarks/record/qwen3-tts.md#codec-decoder-against-the-trace-2026-10-27) |
| `qwen3-tts-0.6b-trace-cpu` | `verify-cpu` | qwen3_tts_06b | [qwen3-tts.md § talker-and-code-predictor-against-the-trace-2026-10-27](../benchmarks/record/qwen3-tts.md#talker-and-code-predictor-against-the-trace-2026-10-27) |
| `qwen3-tts-0.6b-clone-trace-cpu` | `verify-cpu` | qwen3_tts_06b_base | [qwen3-tts.md § the-talker-and-the-decode-against-the-trace-2026-10-29](../benchmarks/record/qwen3-tts.md#the-talker-and-the-decode-against-the-trace-2026-10-29) |
| `qwen3-tts-1.7b-trace-cpu` | `verify-cpu` | qwen3_tts_17b | [qwen3-tts.md § talker-and-code-predictor-against-the-trace-2026-10-27](../benchmarks/record/qwen3-tts.md#talker-and-code-predictor-against-the-trace-2026-10-27) |
| `qwen3-tts-1.7b-design-trace-cpu` | `verify-cpu` | qwen3_tts_17b_design | [qwen3-tts.md § voicedesign](qwen3-tts.md#voicedesign) |
| `qwen3-tts-0.6b-trace-metal` | `verify` | qwen3_tts_06b | [qwen3-tts.md § on-metal-2026-10-27](../benchmarks/record/qwen3-tts.md#on-metal-2026-10-27) |
| `qwen3-tts-0.6b-clone-cpu` | `verify` | qwen3_tts_06b_base | [qwen3-tts.md § the-speaker-encoder-against-the-trace-2026-10-29](../benchmarks/record/qwen3-tts.md#the-speaker-encoder-against-the-trace-2026-10-29) |
| `qwen3-tts-0.6b-clone-metal` | `verify` | qwen3_tts_06b_base | [qwen3-tts.md § the-talker-and-the-decode-against-the-trace-2026-10-29](../benchmarks/record/qwen3-tts.md#the-talker-and-the-decode-against-the-trace-2026-10-29) |
| `qwen3-tts-1.7b-trace-metal` | `verify` | qwen3_tts_17b | [qwen3-tts.md § on-metal-2026-10-27](../benchmarks/record/qwen3-tts.md#on-metal-2026-10-27) |
| `qwen3-tts-1.7b-design-metal` | `verify` | qwen3_tts_17b_design | [qwen3-tts.md § voicedesign](qwen3-tts.md#voicedesign) |
| `qwen3-tts-codec-metal` | `verify` | qwen3_tts_06b | [qwen3-tts.md § on-metal-2026-10-27](../benchmarks/record/qwen3-tts.md#on-metal-2026-10-27) |
| `qwen3-tts-0.6b-roundtrip-metal` | `verify` | qwen3_tts_06b | [qwen3-tts.md § on-metal-2026-10-27](../benchmarks/record/qwen3-tts.md#on-metal-2026-10-27) |
| `qwen3-tts-1.7b-roundtrip-metal` | `verify-release` | qwen3_tts_17b | [qwen3-tts.md § on-metal-2026-10-27](../benchmarks/record/qwen3-tts.md#on-metal-2026-10-27) |
<!-- /gen -->

## Current numbers

`nuclis speak`, ReleaseSafe, Metal, Apple M4 Pro 48 GB, files in the page
cache, F16 talker cache. CustomVoice: speaker `ryan`, seed 0, a 6.9–7.0 s
sentence, nothing else on the GPU (2026-10-27, `519a898`). Base and
VoiceDesign: a 3–4.4 s sentence, the 4.6 s reference clip (2026-10-29).

| Model | Real time | First audio | Prefill | Per frame |
| --- | ---: | ---: | ---: | ---: |
| 0.6B CustomVoice | 3.09–3.14× | 82–101 ms | 49–59 ms | codes 21.6–21.8 ms, codec 3.2–3.3 ms |
| 1.7B CustomVoice | 2.33–2.34× | 123–126 ms | 82–85 ms | codes 29.8 ms, codec 3.3–3.4 ms |
| 0.6B Base, clip and transcript | 2.69× | 259 ms | 227 ms | |
| 0.6B Base, clip alone | 3.06× | 83 ms | 46 ms | |
| 1.7B Base, clip and transcript | 1.17× | 1,748 ms | 1,514 ms | |
| 1.7B Base, clip alone | 2.22× | 122 ms | 76 ms | |
| 1.7B VoiceDesign, 12-word description | 1.28× | 1,110 ms | 961 ms | |

A new voice costs 751 ms with both encoders, 63 ms with the speaker
encoder alone, and 0 ms when read back. Round trip through the
transcriber: 5 word errors in 190 (2.63 %) on both CustomVoice sizes.
How each was measured, and the CPU reference's speed:
[the record](../benchmarks/record/qwen3-tts.md#speed-2026-10-27),
[cloning](../benchmarks/record/qwen3-tts.md#speed-2026-10-29),
[the round trip](../benchmarks/record/qwen3-tts.md#the-round-trip-2026-10-27).

## Known limits

- **The 1.7B's real time is about 2.3×**, below 3×: a frame reads
  5.07 GB over about 1,100 dependent dispatches, and the predictor's small
  matrices do not reach the bandwidth roof one at a time
  ([record](../benchmarks/record/qwen3-tts.md#speed-2026-10-27)).
- **The 1.7B's prefill grows far faster with the prompt's rows** than the
  0.6B's (68 rows: 1,514 ms against 227 ms), so a cloned voice with a
  transcript, or a long description, speaks about a second later; the
  cause is unmeasured.
- **The first audio of an ICL voice grows with its clip** (12.5 frames a
  second of reference): prefer a short clip with its exact words.
- **The 0.6B takes no instruction**, as in the authors' package.
- **Text is fed whole, a sentence at a time** (the authors' non-streaming
  mode); feeding text as it arrives is not implemented.
- **The tokenizer's case seam is ASCII-only**: a word of a non-ASCII
  capital then ASCII capitals splits differently from the authors' run.
- A text that begins with a newline shifts the authors' positional
  slicing (`\n\n` is one token); nuclis slices by position as they do.
