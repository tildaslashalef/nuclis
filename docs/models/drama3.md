<!-- kind: card -->
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

In nuclis it is the second speech model beside [Qwen3-TTS](qwen3-tts.md):
`nuclis speak --model drama-3` and `POST /v1/audio/speech`
([api-speech.md § Drama 3](../guide/api-speech.md#drama-3)), in a voice
saved with `nuclis voice add` or the model's own. One catalogue entry,
`drama-3`: speech.cpp's GGUF layout (architecture `drama3`), converted
from `fishaudio/drama-3`, revision and digest in `src/catalog.json`. The
layout and semantics as read from the GGUF and Fish's code:
[record, 2026-10-30](../benchmarks/record/drama3.md#the-file-2026-10-30).

## Architecture

| Part | What differs from the other families |
| --- | --- |
| Backbone | the Qwen3.5-MoE forward of [qwen3.8.md](qwen3.8.md): 40 layers, full attention at `layer % 4 == 3` (16 query / 2 KV heads of 256, a sigmoid output gate, NeoX RoPE over 64 of 256 dims, base 1e7), DeltaNet otherwise, 256 experts top 8 plus a gated shared expert |
| DeltaNet value heads | Fish's grouped order: value head h reads key head h / 2 (q and k repeated interleaved), not llama.cpp's tiled order |
| Zero-centred norms | stored as 1 + w, so multiplied by the stored weight directly; `ssm_norm` and the codec's RMS norms are plain weights |
| Input rows | a text row is `token_embd[id]`; a semantic row `(token_embd[main] + codebook_embd[code 1]) / √2` |
| Head | `output` cut to 4,097 rows: the end of speech (248046) and the semantic band 248099–252194, mapped back by `drama3.output.token_ids` |
| Tower | 4 shared plus 9 residual `TowerLayer`s, width 2048, 16 / 2 heads of 256, interleaved RoPE over 64 dims, a KV cache per layer over the whole sequence, teacher-forced over the prompt; layer 0 adds the input frame's codes 2–9; head j predicts code j + 1 over 4,096 columns |
| Codec | Fish's modded DAC at 44.1 kHz, 2,048 samples a frame (about 21.5 frames a second): a semantic codebook of 4,096 and 9 residual of 1,024 (dim 8, cosine nearest on encode), windowed causal transformers (512 in the encoder, 128 in the quantizer, BF16-rounded RoPE tables), Snake throughout, all causal |

Every stage of the codec is causal, so a stream carrying each
convolution's history, the transposed convolutions' tails, and the
window's 127 keys and values per layer gives the whole-clip samples.

## Files and encodings

One GGUF, `Drama-35B-A3B-v3-Q4_K.gguf`: 1,531 tensors, 522 Q4_K, 216 F16
(the codec's convolutions and projections) and 793 F32. Its metadata
carries the geometry (`drama3.text.*`, `drama3.audio_decoder.*`,
`drama3.codec.*`), the sampling defaults, the generation limits, the
prompt template's literal strings, and the tokenizer (252,216 tokens, the
Qwen3.5 split). The inventory is `inference/src/models/fixtures/drama3-q4_k.json`.

- Weights are wrapped in place from the mapping on Metal, Q4_K and F32
  alike; nothing is requantized. The codec decoder's weights are copied
  to F32 (823 MB on Metal), already laid out tap-major by the CPU
  `fish_codec.Runtime`.
- The tokenizer is built from `drama3.tokenizer.*` alone: NFC, the regex
  split, byte level, BPE, the added ids matched whole.

## Prompt profile

No chat profile. `drama3_text` builds Fish's `reference-bracket-v1`
template, each literal tokenized on its own and codes entering as frame
rows: a system turn (with one `[Reference N]` block per reference: its
tagged transcript, then its code rows), the history turns, then the
user's tagged text and `<|im_start|>assistant\n<|voice|>`. Text is
prefixed `<|speaker:0|>` unless it opens with a speaker tag (up to 16
speakers), and may not contain the six `drama3.prompt.reserved` strings.
A prompt is at most 8,192 rows.

**Chunking.** nuclis starts a chunk at every sentence end (`.!?;。！？；…`,
newlines, Arabic `؟؛۔`), and Fish's splitter packs by about 300 bytes and
at most 3 speakers past that; it never splits inside a tag or `[…]`.
Each chunk is a user turn; the prompt keeps the first turn and the latest
earlier one. A request's `context` brings an earlier request's last
sentence into the history (`Request.context`).

**Sampling** is the GGUF's defaults: temperature 1.0, top-p 0.9, top-k 30,
by exponential race. The repeat window: a semantic main token among the
last 16 is redrawn at 1.0 / top-k 30 / top-p 0.95; the end is never
redrawn. Residual codes are clamped to 1,023 by the codec.

### The prompt's prefix

Every prompt of a voice begins with the same rows (`Text.prefix`: the
system turn and its references), which `drama3_speech` prefills apart
for the backbone and the tower. The voice keeps the state reached for its
`max_prefixes` = 4 most recent prefixes (`Session.snapshot`: attention
rows, each DeltaNet layer's history and matrix; the tower's one row
behind; the backbone's last hidden), about 90 MB each with a 12 s
reference. A take whose prefix is kept restores it (`Session.restore`)
and prefills only the rest; a restored take computes exactly what an
unrestored one does.

### Stalls, and what nuclis adds

The model sometimes speaks a first sentence then falls silent, or speaks
nothing, in the bf16 checkpoint too. `drama3_speech.Synthesizer` adds,
beside the split at every sentence end:

- **When a take has spoken**: once its last `level_window` = 16 frame
  levels (RMS, the codec's first frame left out) peak at `spoken_rms`
  0.01 or more and at `onset_ratio` 3× their quietest; steady hiss never
  counts.
- **The stop on silence**: after that, a frame is silent under a tenth of
  the take's loudest, clamped to 0.003–0.02; 32 silent frames (1.5 s) end
  the chunk, 6 (0.28 s) are kept and the rest cut.
- **The retry**: a take's leading silence is held back; one that has not
  spoken after 32 frames is dropped and the chunk restarts on the next
  seed, at most twice; the third is released as it is.
- **The frame budget**: 64 frames plus 3 a character, at most 600 a chunk.

Held-back and cut samples are never streamed. `nuclis speak` reports the
retries and how each chunk stopped (`eos`, `silence`, `length`). The
study behind the rule:
[record, 2026-10-30](../benchmarks/record/drama3.md#stalls-and-what-nuclis-adds-2026-10-30).

### On Metal

`drama3_metal.Plan` runs the backbone and tower: one command buffer per
prefill chunk and per backbone row, and ten commits a frame for the tower
(the picks are sampled on the host). The MoE routes on the device.
Prefill projections take the generic F32 tile, not the half-operand Q4_K
tiles, because rounding the router's operands moves past near-ties.
`fish_codec_metal` decodes one command buffer per piece of at most 8
frames, the same samples bit for bit for any push size. The speaker's
codes come from `fish_codec_encoder` on the CPU.

## Verified

<!-- gen:gates-for:drama3 -->
| Gate | Tier | Model | Evidence |
| --- | --- | --- | --- |
| `drama3-metal` | `verify` | drama3_q4_k | [drama3.md § on-metal-2026-10-30](../benchmarks/record/drama3.md#on-metal-2026-10-30) |
| `drama3-trace-cpu` | `verify-cpu` | drama3_q4_k | [drama3.md § the-traces-2026-10-30](../benchmarks/record/drama3.md#the-traces-2026-10-30) |
| `drama3-roundtrip-metal` | `verify` | drama3_q4_k | [drama3.md § the-round-trip-2026-10-30](../benchmarks/record/drama3.md#the-round-trip-2026-10-30) |
| `drama3-codec-cpu` | `verify-cpu` | drama3_q4_k | [drama3.md § the-traces-2026-10-30](../benchmarks/record/drama3.md#the-traces-2026-10-30) |
| `drama3-codec-metal` | `verify` | drama3_q4_k | [drama3.md § the-codec-on-metal-2026-10-30](../benchmarks/record/drama3.md#the-codec-on-metal-2026-10-30) |
<!-- /gen -->

## Current numbers

`drama3-check speak --backend metal`, no reference voice, ReleaseFast,
Apple M4 Pro 48 GB (2026-10-30):

| Text | Frames | Audio | Prompt | Per frame (tower and decode / codec) | First audio | Wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| the `fox` sentence | 84 | 3.90 s | 0.5 s | 20.2 / 4.8 ms | 0.72 s | 2.6 s |
| a 26-word sentence | 175 | 8.13 s | 0.6 s | 20.6 / 4.6 ms | 0.84 s | 5.0 s |

A frame is 46.4 ms of audio, so a take runs at about 0.65× real time.
Through `/v1/audio/speech` with a saved 12 s voice: first audio 1,988 ms
for the voice's first request, 393 ms with its prefix restored. Round
trip through the transcriber: 2 errors in 190 words (1.05 %). On short
Egyptian Arabic phrases with the stall rule, 1 take in 64 stalls. How
each was measured:
[speed](../benchmarks/record/drama3.md#speed-2026-10-30),
[the prefix](../benchmarks/record/drama3.md#the-prompts-prefix-2026-10-30),
[the round trip](../benchmarks/record/drama3.md#the-round-trip-2026-10-30).

## Known limits

- **A cue costs takes**: a `[bracketed]` cue in the phrase makes stalls
  more likely, more so mid-phrase than before it.
- **Steady noise after speech is not stopped**: a whisper is nearly as
  flat as hiss, so such a take runs to its frame budget.
- **The codec on Metal pays about 6 ms a piece**: a take pushes one frame
  at a time until it speaks, at 9 ms a frame.
- **The tower commits ten command buffers a frame**, one per codebook
  head, since the picks are sampled on the host.
- **Memory beside other models**: the server's budget counts files, not
  Drama 3's own buffers (the F32 codec, about 0.8 GB; a synthesizer's
  plan, about 1.2 GB while it speaks; each kept prefix), so the machine
  holds about 2.5 GB more than the budget shows.
- Fish's `reference_sampling` (an epsilon cutoff) is not implemented.
