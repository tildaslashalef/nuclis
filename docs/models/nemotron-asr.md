<!-- kind: card -->
# Nemotron 3.5 ASR streaming 0.6B

[Nemotron 3.5 ASR streaming 0.6B](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
(NVIDIA, OpenMDW-1.1) is a streaming speech recognizer: 16 kHz mono audio
in, punctuated and capitalized text out, in 40 locales. In nuclis it is
the ears of a local voice assistant: `nuclis transcribe`, the file route
and the streaming WebSocket route on `nuclis serve`
([api-transcription.md](../guide/api-transcription.md)). One catalogue
entry, `nemotron-asr-0.6b`, the published F32 `model.safetensors`
(revision and digest in `src/catalog.json`). The format as read from the
checkpoint and the transformers definition:
[record, 2026-10-27](../benchmarks/record/nemotron-asr.md#the-encoder-2026-10-27).

## Architecture

| Part | What it is |
| --- | --- |
| Front end | pre-emphasis 0.97; 512-point FFT of a symmetric Hann 400 centred in 512, hop 160; power spectrum; 128 Slaney-scale, Slaney-normed mel bins over 0–8 kHz; `log(m + 2⁻²⁴)`; `⌊L/160⌋` frames, the last zeroed; no normalization, no dither (`mel.zig`, `Spec.nemotron`) |
| Subsampling | three 3×3 stride-2 convolutions, causal on time and frequency (`dw_striding`), flattened channel-major (256 × 17), then 4352 → 1024: one row per 80 ms |
| Encoder | 24 FastConformer blocks, width 1024, FFN 4096 SiLU in two macaron halves, 8 heads of 128, no matrix biases, LayerNorm ε 1e-5, a final `norm_out` per block |
| Attention | Transformer-XL relative attention with untied `bias_u`/`bias_v`, an interleaved sin/cos table; a chunked mask (`chunked_limited`): chunks of `r + 1` rows, each seeing its chunk and the 56 rows before it, the same mask whole or streamed |
| Convolution module | GLU (2048 → 1024), causal depthwise kernel 9, LayerNorm, SiLU, pointwise |
| Language prompt | after the encoder, per row: `[h ; onehot₁₂₈(p)]` → 2048 ReLU → 1024, then `encoder_projector` → 640; touches no encoder state |
| Transducer | RNN-T: a 2-layer LSTM of 640 over a 13,088-row embedding (the blank, 13087, a zero row), joint `head · relu(e + g)` over 13,088 outputs, greedy |

**Lookahead** r is 0, 3 (the default), 6, or 13 rows (80–1,120 ms of
latency); the checkpoint has no full-context mode. A stream carries per
block the last 56 keys and values and the last 8 GLU rows, one input row
per subsampling convolution, the open mel frames, and the LSTM state.
Chunked equals whole-clip at the same r; at the end of a stream nuclis
flushes as the whole-clip pass ends (zero samples, one zero row of right
padding per convolution, attention over valid rows), so the rows equal a
whole-clip pass on the same audio.

**Greedy decoding** is the authors' loop (a blank advances the row; the
tenth symbol on one row forces an advance) with one nuclis rule: a token
that starts a word (its piece begins with `▁`), emitted again on the row
that just emitted it, ends the row as a blank would (`Weights.endsRow`).
Why: [record, 2026-11-01](../benchmarks/record/nemotron-asr.md#decisions-and-open-items-2026-10-27).

## Files and encodings

The checkpoint directory: `config.json`, `generation_config.json`,
`processor_config.json` (the feature extractor and `prompt_dictionary`,
121 locale strings onto 84 prompt ids), `tokenizer.json`, and
`model.safetensors` (655 tensors, all F32, 638 M parameters). The `.nemo`
archive and the `.q8_0.gguf` are not used.

- `nemotron_encoder.bind` and `nemotron_rnnt.bind` view the F32 tensors
  in place; the CPU reference computes from the mapped file.
- On Metal the matrices are the mapped F32 bytes, wrapped in place; two
  are copies (the subsampling linear with its columns permuted, and
  `prompt_projector.linear_1`'s first 1024 columns).
- Tokenizer: decode only, a piece table from `tokenizer.json`'s BPE
  (13,087 pieces, `Metaspace`)
  ([tokenizer.md § Decode-only piece tables](../engine/tokenizer.md#decode-only-piece-tables)).
  The model's blank, 13087, is the tokenizer's `<pad>`; the 39 language
  tags are stripped from the text by id and reported beside it.

## Prompt profile

No chat profile. The language prompt is an id from `prompt_dictionary`
(`--language`, a locale such as `en-US` or `de-DE`); the default is
`auto` (101), the code's default, under which the model emits a language
tag after terminal punctuation. A fixed locale transcribes its own
language and returns nothing for most others.

**Several languages.** `--language ar,en-US,auto` takes up to
`max_languages` = 4 prompts for one encoder pass, then per language the
prompt block, the projector, and its own greedy `Decoder`
(`Transcriber.setLanguages`). Words come in stretches that end
`choose_again_rows` = 13 rows (1.04 s) after the last text of the chosen
language; within the open stretch the language that wrote text with the
highest score gain wins. `auto` may come only last, as the fallback, and
replaces another only while writing the same script. `Transcriber.settled`
counts the tokens that are final; each language's next stretch
begins `echo_rows` = 4 rows past the last text.

**Phrase boosting** (`--phrase`, at most 64; `inference/src/audio/boost.zig`).
Each phrase is spelled as the pieces spell it and matched as text against
the piece table; at each greedy step a piece that starts a phrase gets
`default_bonus` 2 added, one that continues it `default_follow` 6, and
the argmax is taken over the plain logits and those candidates. CPU and
Metal share `Decoder.choose`.

**Engines.** `nuclis transcribe` runs on Metal by default when built with
it, `--backend cpu` on the reference; both give the same tokens. On Metal
the front end and the LSTM stay on the host, everything else on the
device; a push is one command buffer, and a chunk with no token is one
command buffer, each token adding one. A file is decoded and fed in
2.56 s pieces, so a recording of any length holds no decoded samples.
State lives as long as the utterance: the streaming route clears it after
each final transcript and on `reset`.

## Verified

<!-- gen:gates-for:nemotron-asr -->
| Gate | Tier | Model | Evidence |
| --- | --- | --- | --- |
| `nemotron-asr-encoder-cpu` | `verify-cpu` | nemotron_asr | [nemotron-asr.md](nemotron-asr.md) |
| `nemotron-asr-trace-cpu` | `verify-cpu` | nemotron_asr | [nemotron-asr.md § transducer-against-the-trace-and-word-error-rate-2026-10-27](../benchmarks/record/nemotron-asr.md#transducer-against-the-trace-and-word-error-rate-2026-10-27) |
| `nemotron-asr-wer-cpu` | `verify-cpu` | nemotron_asr | [nemotron-asr.md § transducer-against-the-trace-and-word-error-rate-2026-10-27](../benchmarks/record/nemotron-asr.md#transducer-against-the-trace-and-word-error-rate-2026-10-27) |
| `nemotron-asr-trace-metal` | `verify` | nemotron_asr | [nemotron-asr.md § on-metal-2026-10-27](../benchmarks/record/nemotron-asr.md#on-metal-2026-10-27) |
| `nemotron-asr-wer-metal` | `verify` | nemotron_asr | [nemotron-asr.md § on-metal-2026-10-27](../benchmarks/record/nemotron-asr.md#on-metal-2026-10-27) |
| `nemotron-asr-file-route` | `verify` | nemotron_asr | [api-transcription.md § transcription-files](../guide/api-transcription.md#transcription-files) |
| `nemotron-asr-long` | `verify` | nemotron_asr | [api-transcription.md § transcription-files](../guide/api-transcription.md#transcription-files) |
| `nemotron-asr-stream-route` | `verify` | nemotron_asr | [api-transcription.md § live-transcription](../guide/api-transcription.md#live-transcription) |
<!-- /gen -->

## Current numbers

Word error rate over the 60-clip corpus at r = 3, each clip under its
locale, `nuclis transcribe` (2026-10-27; Metal's transcripts are word for
word the CPU's):

| Language | Clips | Words | WER |
| --- | ---: | ---: | ---: |
| English (LibriSpeech test-clean) | 40 | 828 | 3.86 % |
| German (FLEURS) | 5 | 96 | 14.58 % |
| French (FLEURS) | 5 | 97 | 15.46 % |
| Spanish, Latin American (FLEURS) | 5 | 95 | 5.26 % |
| Russian (FLEURS) | 5 | 95 | 8.42 % |
| all | 60 | 1,211 | 6.11 % |

With the repeated-word rule (2026-11-01, Metal) the corpus gives 6.52,
6.11, 6.44, and 6.36 % at r = 0, 3, 6, 13. Speed, Metal, an 11.70 s
LibriSpeech clip, nothing else on the GPU (2026-10-27, `519a898`,
ReleaseSafe, M4 Pro 48 GB):

| | r = 0 | r = 3 |
| --- | ---: | ---: |
| whole clip | 189 ms (0.016×) | 219–220 ms (0.019×) |
| chunk push, median, back to back / at real time | 12.6 / 19.7–19.9 ms | 32.1–33.4 / 36.4–36.9 ms |
| final transcript after the audio ends, at real time | 19.0–23.0 ms | 37.0–41.1 ms |
| streamed, busy time over audio | 0.16× | 0.12× |

The corpus's 512.2 s took 12.2 s on Metal (0.024×) and 71.5 s on the CPU.
Several languages cost 0.016× (`auto`) to 0.023× (four prompts) on a
190.9 s recording. How each was measured:
[record](../benchmarks/record/nemotron-asr.md#on-metal-2026-10-27),
[phrase boosting](../benchmarks/record/nemotron-asr.md#phrase-boosting-2026-10-28),
[several languages](../benchmarks/record/nemotron-asr.md#several-languages-2026-10-30).

## Known limits

- **A short word after silence is held or lost** ("Hey", "Hi" after
  seconds of digital zeros): its first piece comes within 0.5–2 logits of
  the blank. It matches the trace, so it is the checkpoint's behaviour; a
  blank penalty brings words back but changes their spelling.
- **A change of language in one stream loses words** at the switch:
  19.0 % WER over a 30-minute mixed recording against 6.3 % for its clips
  alone ([api-transcription.md](../guide/api-transcription.md)).
- **A tag appears under an explicit language too** (55 of 60 clips), the
  prompt's own locale; `<sl-SL>` has no prompt key, and `mt-MT` has a
  prompt but no tag.
- **The r = 3 chunk takes 32–37 ms on Metal**, above 15 ms: its floor is
  the 2.5 GB of F32 weights read per chunk, then about 220 small
  dispatches. The final transcript still arrives within about 40 ms of
  the end of the audio.
- **On Metal a whole clip and the same clip streamed are not bit for bit
  equal** (batches of more than 8 rows take the 32×32 tile, chunks the
  few-rows product); both match the trace's tokens.
- **A short name said into the word before it** ("يا مون") is not
  lifted by phrase boosting without looping; it needs a beam.
- The authors' code refuses more than 5,000 positions (400 s) per
  forward; nuclis feeds every input as a stream, which reads only the 83
  relative positions a chunk can see.
