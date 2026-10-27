# TODO — active plan

This file is the queue: the units agreed and not yet merged, each with its
design, and a *Where we are* note. A unit is one session, one branch, and
one pull request, whose description is the record of the work; when it
merges, its section is deleted here, and when the last one goes, this file
is emptied back to this header and its *Deferred* list. Requirements live in [docs/spec.md](docs/spec.md); the
engine map in [docs/architecture.md](docs/architecture.md); how to build,
test, and measure in [docs/development.md](docs/development.md).

Session protocol (also in [AGENTS.md](AGENTS.md)): read this file first. If
it lists work, summarize *Where we are* and ask the user how to continue. If
it is empty, ask what to work on and write the agreed plan here.

## Where we are

Theme: **speech in and speech out, served locally, for a voice
assistant.** nuclis gains two model families, streaming speech
recognition (NVIDIA Nemotron 3.5 ASR streaming 0.6B) and speech synthesis
(Qwen3-TTS 12Hz 0.6B CustomVoice), and the API to run them beside Gemma 4
E4B: the ear, the brain, and the voice of **Muon**, an experimental macOS
app in `~/Code/muon` (its own repository and plan; it calls nuclis over
HTTP and adds nothing to this tree). Agreed with the user 2026-10-27.

**One unit, one branch, many commits** (the user's decision, 2026-10-27):
the whole theme is the unit below, on `speech-in-and-out`, merged as one
pull request. Independent tasks run in parallel worktrees whose branches
are rebased onto `speech-in-and-out`; the table says which.

**Validate what the change touches, nothing more** (the user's, 2026-10-27):
each task names its gates. The new families bring their own gates; no
existing family's numerics change, so `make verify` (the whole fast tier),
`make verify-long`, and `make verify-release` are not run for this unit
unless a task edits a shared kernel or backend file, in which case
`make verify-auto` names what that file selects and only that runs. The
CPU tier runs as the new families' gates only (`make gate NAME='…-cpu'`),
not `make verify-cpu`.

Delivered: tasks 1–7 (both families pinned, described, and running on
the CPU and on Metal against the authors' traces) and 13–14 (the Metal
backend split by operation). `nuclis speak` and `nuclis transcribe`
default to Metal. Measured on a quiet machine (2026-10-27, `519a898`):

- **Voice:** 0.6B 3.1× real time, first audio 82–101 ms after the text;
  1.7B 2.3×, first audio 123–126 ms (`docs/models/qwen3-tts.md` § Speed).
  The 1.7B is below the proposed 3× (bandwidth over ~1,100 dependent
  dispatches a frame) and comfortably above real time. Codes on Metal
  equal the CPU sampler's exactly. Round trip (20 sentences, spoken on
  Metal, heard by the CPU recognizer): 2.6 % WER on both sizes.
- **Ears:** a whole 11.7 s clip in 189–220 ms; streamed at r = 3, a chunk
  push takes 32–33 ms idle (the GPU clocks down between chunks; 21 ms
  beside another GPU job), the final transcript 37–41 ms after the audio
  ends; the proposed ≤ 15 ms per chunk is not met (deferred below).
  Metal transcripts equal the CPU's word for word on all 60 clips
  (6.11 % WER).

Checks: `make verify` once with both Metal plans in (57 passed, 0
failed, the 3 Bonsai gates skipped, its file absent), every speech gate
on both tiers, `make test`, `make test-metal`. The WER gates read
`.reference/speech/`, which only the main checkout has: run them there,
or symlink it into a worktree.

Next: **task 8** (E4B hears audio) and **task 9** (serving speech) in
parallel, then 10 and 15 (the transcription routes), then 11 (all three
models together), then 12 (documents and the site).

| Unit | Branch | State |
| --- | --- | --- |
| Speech in and out: Nemotron ASR, Qwen3-TTS, the voice API | `speech-in-and-out` | next: tasks 8 and 9 |

## Speech in and out: Nemotron ASR, Qwen3-TTS, the voice API

Base: `35737c3`

Pull request title: `feat: speech in and out: Nemotron streaming ASR, Qwen3-TTS, and the voice API`.

### Decisions

1. **How a new architecture is validated**: both of the below (the
   user's, 2026-10-27; `docs/spec.md` § 3, `docs/models/README.md` § 0).
   Neither family has a nuclis path to compare against.
   - *Data, not implementations, as the permanent gates.* ASR: word error
     rate on a pinned set of public clips with human transcripts (a few
     dozen LibriSpeech test-clean and FLEURS clips, digests pinned like the
     wikitext corpus), bound set from the first measured run. TTS: the
     round trip, ASR(TTS(text)) against text, on a pinned sentence list,
     once the ASR passes its own gate; plus the codec's frames
     reconstruct their own decode exactly between CPU and Metal.
   - *One-time bring-up traces* from the authors' published code
     (`transformers` at `c0d024fb`, `Nemotron3_5AsrForRNNT`; Qwen's
     `QwenLM/Qwen3-TTS` at `022e286b`; both sources unpacked under
     `.reference/`), run once in a venv there, pinned as small
     fixtures under `tests/fixtures/`, as EmbeddingGemma 2's audio was
     (`docs/engine/audio.md` § Against the oracles). They catch the subtle
     errors a WER bound passes. Afterwards the CPU reference is the
     oracle for Metal, as for every family.
2. **Weights as published** (settled by task 1, 2026-10-27). The ASR file
   is F32 (2.55 GB), the TTS talker BF16 (1.81 GB), its codec F32 (0.68 GB,
   only the decoder used). The generic dequantizer already reads F32, F16,
   and BF16 on both backends (`inference/src/quant/decode.zig`,
   `backends/metal/dequant.metal`, encodings 0, 1, 30), and Laya runs its
   safetensors matrices through the generic tile in place
   (`laya_metal.zig`). So both families run as published, nothing
   converted (spec § 3, weights never requantized or expanded). Estimated
   bytes read (not measured): a TTS frame ≈ 0.9 GB talker + 15 predictor
   steps × ≈ 0.16 GB ≈ 3.3 GB, ≈ 41 GB/s at real time (12.5 frames/s);
   an ASR chunk reads the encoder once, ≈ 2.4 GB. Specialized F32/BF16
   matvecs are a speed task, measured first, never assumed.
3. **Both voice sizes, the 1.7B for emotion** (the user's, 2026-10-27).
   The authors' code drops the instruction on the 0.6B
   (`generate_custom_voice` in `qwen_tts/inference/qwen3_tts_model.py`,
   "for 0b6 model, instruct is not supported"; their README's model table
   agrees): only `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` performs a mood.
   Same 9 speakers, codec, and code predictor (width 1024, 5 layers); the
   talker is 28 layers of width 2048, FFN 6144, 3,833,402,552 B BF16. One
   implementation serves both, parameterized by the talker width; bring-up
   and the first traces on the 0.6B, then the 1.7B with its own trace and
   round-trip gate. Muon defaults to the 1.7B. Estimated (not measured)
   bytes per frame ≈ 2.8 GB talker + 2.4 GB predictor ≈ 5.2 GB, ≈ 65 GB/s
   at real time.

### Facts already read (2026-10-27, from the files' `config.json`)

- **Qwen3-TTS 0.6B CustomVoice**, `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` at
  `85e237c1`: `model.safetensors` 1,811,626,576 B, `speech_tokenizer/`
  682,293,092 B, Qwen2 BPE (`vocab.json`, `merges.txt`). Talker: 28 layers,
  width 1024, 16 query / 8 KV heads of 128, FFN 3072 SiLU, RMSNorm 1e-6,
  RoPE θ 1e6 with interleaved M-RoPE sections [24, 20, 20], codec vocab
  3072; text embeddings 151,936 × 2048 projected to 1024. Code predictor:
  5 layers, same widths, vocab 2048, 16 code groups (the talker predicts
  group 0, the predictor groups 1–15, sequentially, per frame). Speakers
  are codec-vocab ids (`spk_id`: vivian, serena, uncle_fu, dylan, eric,
  ryan, aiden, ono_anna, sohee); languages likewise (`codec_language_id`,
  10 plus 2 dialects). Codec decoder (`speech_tokenizer/config.json`):
  16 codebooks of 2048 (the config's `semantic_codebook_size` 4096 is
  never read; `docs/models/qwen3-tts.md`), an 8-layer
  transformer of width 512 with a 72-frame sliding window, then
  upsampling [8, 5, 4, 3] × [2, 2] = 1,920 samples per frame: 12.5 frames/s
  to 24 kHz mono. CustomVoice needs no codec *encoder* (that is for
  cloning).
- **Nemotron 3.5 ASR streaming 0.6B**, `nvidia/nemotron-3.5-asr-streaming-0.6b`
  at `ea30d66d`: `model.safetensors` 2,552,062,944 B (F32), `tokenizer.json`.
  Front end: 16 kHz, n_fft 512, window 400, hop 160, pre-emphasis 0.97,
  128 mel bins. Encoder (FastConformer, cache-aware): subsampling ×8 (conv,
  256 channels, kernel 3, stride 2), 24 blocks of width 1024, 8 heads, FFN
  4096 SiLU, depthwise conv kernel 9, attention over its own chunk plus
  56 rows before it (`sliding_window` 57 counts the row itself),
  lookahead 0, 3, 6, or 13 frames (80 ms each; default 3). RNN-T: a 2-layer
  prediction network of width 640, ReLU joint, vocab 13,088 (blank
  13,087), at most 10 symbols per frame; a language prompt (128 slots;
  `prompt_dictionary`, `auto` = 101).

### Tasks

| # | Task | Track | Depends on | Parallel |
| --- | --- | --- | --- | --- |
| 1 | Facts and pins | — | — | done |
| 2 | TTS codec decoder, CPU | A | 1 | done |
| 3 | TTS talker and code predictor, CPU; `nuclis speak` | A | 2 | done |
| 4 | TTS on Metal, streaming, real-time factor | A | 3 | done |
| 5 | ASR front end and streaming encoder, CPU | B | 1 | done |
| 6 | ASR transducer and tokenizer, CPU; `nuclis transcribe` | B | 5 | done |
| 7 | ASR on Metal, chunk latency | B | 6 | done |
| 8 | Gemma 4 E4B audio input in chat | D | — | worktree D, any time |
| 9 | Serve: speech and transcription kinds, pinning, `POST /v1/audio/speech` | C | 4 | main |
| 10 | Serve: WebSocket transport, streaming transcription | C | 7, 9 | main |
| 11 | Voice scheduling measured: E4B, TTS, and ASR together | C | 9, 10 | main |
| 12 | Documents, README, website (links Muon) | — | all | main |
| 13 | Split `kernels.metal` by operation (before 4 and 7) | — | — | done |
| 14 | Split the Metal backend's `root.zig` by operation (before 4 and 7) | — | — | done |
| 15 | Serve: file transcription for long recordings (`POST /v1/audio/transcriptions`) | C | 7, 9 | main, beside 10 |

Worktree rule: a track owns its new files. Shared registrations
(`src/catalog.zig`, `gates.json`, `docs/spec.md`, `docs/README.md`,
`THIRD_PARTY_NOTICES.md`) are small appends a track may make; the main
session rebases each track onto `speech-in-and-out` and resolves them.

#### 1. Facts and pins (delivered 2026-10-27)

Pulled and pinned (`docs/models/catalogue.md`): both Qwen3-TTS sizes and
Nemotron at the revisions above. Written: `docs/models/qwen3-tts.md` and
`docs/models/nemotron-asr.md` (tensors from the headers, forward passes
in equations from the authors' code, decisions and open items), the
validation rule in `docs/spec.md` § 3 and `docs/models/README.md` § 0.
Settled there: the talker's M-RoPE is plain 1-D NEOX RoPE (all three
components equal), base 1e6, one position per frame; the codec decodes
causally with carried state (traces compared over the first 300
frames); text goes in a sentence at a time (non-streaming text mode);
the ASR mask is chunked (a row sees its chunk plus 56 rows), the stream
flushes as the whole-clip pass ends, state lives as long as the stream,
language `auto` by default, no dither. Sources for the traces:
`.reference/qwen3-tts-022e286/` (package pins `transformers==4.57.3`)
and `.reference/transformers-c0d024f/`.

#### 14. Split the Metal backend's `root.zig` by operation (delivered 2026-10-27)

`inference/src/backends/metal/root.zig` (2,258 lines) became `root.zig`
(`Backend`'s core: device, buffers, recording, dispatch, profiling; about
470 lines), `kernels.zig` (the MSL source's assembly and `Kernel`),
`profile.zig` (timing types), and `ops/<kind>.zig`, one per
`kernels/<kind>.metal` (`matvec`, `matmul`, `experts`, `rowwise`,
`encoders`, `recurrent`, `attention`, `sampling`, `audio`). Each encoder is
a function of `*Backend` that `root.zig` aliases as a method, so no caller
changed. `Backend`'s public declarations are the same 202 plus `dispatch`,
`dispatch2`, and `perElement`, made public for `ops/` (documented as
such); `overlaps` became `Buffer.overlaps`. Checked: `make build`, `make
test`, `make test-metal`, `zig build check-tools`, and `make verify` once
(50 of 53 passed, 0 failed; the 3 Bonsai gates skipped, its file not on
this machine). Tasks 4 and 7 add their encoders in new `ops/` files and
one alias line each.

#### 13. Split `kernels.metal` by operation (delivered 2026-10-27)

The 4,249-line `inference/src/backends/metal/kernels.metal` became 14 files
under `backends/metal/kernels/` along its own sections (`common`, `matvec`,
`matvec_codebook`, `matvec_split`, `matvec_rows`, `experts`, `matmul`,
`rowwise`, `recurrent`, `attention`, `recurrent_chunk`, `sampling`,
`encoders`, `audio`), joined at compile time in that order (`kernel_files`
in `root.zig`), so tasks 4 and 7 add their kernels in separate files
instead of colliding in one. Proof of no change: the joined files are
byte-identical to the old file (so the compiled library is the same);
`make build` and `make test-metal` pass; no model gate needed.

#### 2. TTS codec decoder, CPU (worktree A)

- **Trace fixtures first.** `scripts/speech-traces/` (Python, `make
  lint-py`): `qwen3_tts_trace.py` runs the authors' package in a venv at
  `.reference/venv-qwen3-tts/` (`qwen-tts` at `022e286b`,
  `transformers==4.57.3`, CPU, F32) and writes, for two fixed sentences
  and one speaker per size at seed 0: the prompt's token and codec ids,
  the talker's hidden rows at layers 0, 13, 27 for the prompt and the
  first 4 frames, the 16 codes of the first 24 frames, the codec
  decoder's quantizer output, transformer output, and the first 4,800
  samples. Pinned as small F32 files under
  `tests/fixtures/qwen3-tts/` with a JSON index (digests, revision,
  versions); `THIRD_PARTY_NOTICES.md` names the source. Sampling is
  replaced by the trace's own codes, so nuclis replays them.
- `inference/src/audio/qwen3_codec.zig`: `bind(checkpoint)` over the
  decoder's 271 tensors (`decoder.*` of
  `speech_tokenizer/model.safetensors`; the encoder's 225 are ignored by
  name, not read), `Runtime` with `decode(codes: [n][16]u16, out: []f32)`
  and `Stream` (`push(frames)` → samples, carrying every convolution's
  left context and the transformer's last 71 rows of K/V per layer: a query sees 72 keys including itself), the
  qwen3-tts.md § The codec decoder pipeline: codebook lookup as
  `embedding_sum / max(cluster_usage, 1e-5)`, the two projected sums,
  the causal input convolution, 8 layers with layer scale and RoPE base
  1e4, the two ×2 transposed-convolution plus ConvNeXt stages, the rates
  8, 5, 4, 3 with SnakeBeta (α, β stored as logs), the clamp to [−1, 1].
- Unit tests: a stream pushed 1, 3, and 7 frames at a time gives the
  samples of one `decode` call, bit for bit.
- Gate `qwen3-tts-codec-cpu`: the fixture's 24 frames through `decode`,
  every stage against the trace (bounds from the first run, recorded in
  `gates.json`). `make test`, `make verify-auto` (selects nothing else:
  new files only).

#### 3. TTS talker and code predictor, CPU; `nuclis speak` (worktree A)

From task 2: the trace's codes are sampled with torch's generator, which
nuclis cannot reproduce; compare the talker's and predictor's hidden rows
and logits while replaying the trace's codes, never sampled ids. The
script's cases are `Spec` entries; talker hooks go on
`tts.model.talker.model.layers[i]` (with the cache they fire once for the
prefill, then once per step). The `window` sentence can be the second
case.

- **Tokenizer.** Qwen2's byte-level BPE from `vocab.json` and `merges.txt`
  (no `tokenizer.json`): `inference/src/tokenizer/vocab_merges.zig` loads
  them and the added tokens of `tokenizer_config.json` into the shared
  `Vocabulary`. Check whether Qwen2's split pattern equals the `qwen35`
  splitter's (`encode.zig`); add a `qwen2` splitter only if it does not.
  Test: the trace's token ids for both sentences.
- `inference/src/models/qwen3_tts.zig`: `Config` from `config.json` (both
  sizes; width, FFN, the predictor projection present or not), `bind`
  (402 or 404 tensors, refusing any other set by name), `Prompt.build`
  (qwen3-tts.md § (a): role rows, the codec prefix of think, language,
  speaker; non-streaming text rows; the instruction rows on the 1.7B,
  refused on the 0.6B as `InstructionUnsupported`), the text projection
  (2048 → 2048 → width, SiLU, biases).
- `inference/src/models/qwen3_tts_runtime.zig`: the CPU forward. The
  talker: 28 Qwen3 layers (q/k RMSNorm, NEOX RoPE base 1e6, GQA 16/8,
  SwiGLU), `codec_head`, the group-0 sampler (§ (c): repetition 1.05
  over generated codes, end blocked for 2 frames, suppression of
  2048–3071 but 2150, temperature 0.9, top-k 50). The predictor: a fresh
  16-row cache per frame, inputs projected by `small_to_mtp_projection`
  on the 1.7B, heads `lm_head[g−1]`, temperature 0.9, top-k 50, exactly
  15 steps. The next talker row: the 16 code embeddings summed plus the
  next text-track row (§ (e)); stop at 2150 or 8,192 frames.
  `inference/src/sampling/` is reused for top-k and temperature.
- `src/speak.zig` and its line in `src/help.zig`: `nuclis speak <text>
  [--voice <speaker>] [--language <l>] [--instruct <s>] [--model <path>]
  [--seed <n>] [--backend cpu|metal] -o <file.wav>`; 24 kHz mono 16-bit
  WAV. `main.zig` only dispatches.
- Gates: `qwen3-tts-0.6b-trace-cpu` and `qwen3-tts-1.7b-trace-cpu` (the
  trace's prompt rows, hidden rows, and, replaying its codes, the
  predictor's logits). Listen to five sentences per size, moods on the
  1.7B, and say so in the record.

#### 4. TTS on Metal, streaming (worktree A)

From task 3: replay the trace's codes (`talker` mode, the `Replay` pick)
and compare logits against the CPU reference; torch's generator is not
reproduced. Keep the weights BF16 in place on Metal (the CPU widens to F32
only as a reference; 6.4 GB for the 1.7B). The predictor's 15 sequential
steps through 5 layers per frame are about 40–55 ms of the CPU's frame:
launch overhead decides the Metal frame time. `Synthesizer.read` pushes up
to 32 frames per codec call; streaming needs small pushes (the codec is
exact for any piece size). New kernels go in their own
`backends/metal/kernels/*.metal` file appended to `kernel_files`, their
encoders in `backends/metal/ops/` with one alias line each in `root.zig`.

- `inference/src/models/qwen3_tts_metal.zig`: the talker from the
  existing pieces (`rmsNormRope` with `.split_half`, the decode
  attention, generic matvec over BF16, `siluMul`), the predictor's 15
  steps encoded into one command buffer per frame (the per-step
  dispatch is the risk: measure it), sampling on the CPU from the
  logits as the other families do.
- `inference/src/audio/qwen3_codec_metal.zig`: the decoder's plan.
  Missing kernels (transposed convolution, SnakeBeta, the 72-row windowed
  attention if `attentionSegments` does not cover it) are added to
  `backends/metal/kernels/` (the voice's and the ears' kernels in `audio.metal`, or a file of their own appended to `kernel_files`); each new kernel gets a `make
  test-metal` case against the CPU function.
- Streaming: `Synthesizer` (in `qwen3_tts.zig`) yields PCM in chunks of a
  configurable number of frames (default 4, 320 ms).
- Measure on this machine, both sizes, recorded in qwen3-tts.md: first
  audio after the text (proposed target ≤ 150 ms), real-time factor alone
  (proposed ≥ 3×), bytes per frame against the estimate.
- Gates: `qwen3-tts-{0.6b,1.7b}-trace-metal` (Metal against the CPU
  fixture), `qwen3-tts-codec-metal` (Metal decode against CPU decode on
  the same codes); `qwen3-tts-roundtrip-metal` once task 7 lands
  (word error rate of the recognized speech against the text, on a
  pinned list of 20 sentences, both sizes). If a kernel file shared with
  other families changed, the gates `make verify-auto` names, and only
  those.

#### 5. ASR front end and streaming encoder, CPU (worktree B)

- **Trace fixtures first.** `scripts/speech-traces/nemotron_trace.py`
  in `.reference/venv-transformers/` (transformers at `c0d024fb`, CPU,
  F32): for two clips (one LibriSpeech test-clean, one FLEURS non-English,
  public, licences in `THIRD_PARTY_NOTICES.md`), the features, the
  subsampling output, encoder rows at blocks 0, 11, 23, the prompt
  block's output, the joint's first 20 logit rows and the token ids, at
  lookaheads 0 and 3; plus the same clip streamed chunk by chunk.
  Pinned under `tests/fixtures/nemotron-asr/` with an index.
- **The speech corpus.** `scripts/speech-corpus.py` fetches a pinned set
  (about 40 LibriSpeech test-clean and 20 FLEURS clips across 4
  languages, each by URL and SHA-256 in `tests/fixtures/speech-corpus.json`
  with its reference transcript) into `.reference/speech/`, as `make
  eval-corpus` does for wikitext; a `make speech-corpus` target.
- **Front end.** `inference/src/audio/mel.zig` `Bank` gains a `Spec`
  (window shape and length, centring, power or magnitude, Slaney or HTK
  scale and area normalization, log guard, pre-emphasis) with Gemma's as
  one value and Nemotron's as another (nemotron-asr.md § The front end);
  `features` keeps Gemma's output bit for bit (EmbeddingGemma's unit
  tests and gates `embeddinggemma-audio-*` prove it).
- `inference/src/audio/nemotron_encoder.zig`: `bind`, the subsampling
  (three causal 3×3 stride-2 convolutions, channel-major flatten, 4352 →
  1024), 24 blocks (macaron FFN halves, Transformer-XL attention with
  per-layer `bias_u`/`bias_v` and `relative_k_proj`, the chunked mask,
  the GLU and causal depthwise convolution with LayerNorm), the prompt
  block (one-hot 128, 1152 → 2048 → 1024, ReLU). `Stream.push(pcm)`
  returns the rows whose chunk is complete, carrying the state of
  nemotron-asr.md § Cache-aware streaming; `finish()` flushes as the
  whole-clip pass ends. Compare `gemma4a.zig` first: the Gemma 4 audio
  encoder is a conformer too; share what is truly the same, nothing by
  default.
- Unit test: a clip of N = 1 + 8r + 8(r+1)m mel frames gives the same
  rows streamed in chunks as in one pass, at r = 0 and 3.
- Gate `nemotron-asr-encoder-cpu` (features, subsampling, block rows
  against the trace).

#### 6. ASR transducer and tokenizer, CPU; `nuclis transcribe` (worktree B)

From task 5: `encoder_projector` (1024 → 640) is not implemented yet and
belongs here. The fixtures under `tests/fixtures/nemotron-asr/` are at
2.8 MB; pin the joint's output as argmax ids with their margins (or the
top 5 per row), not whole logit rows. `audio.zig`'s own WAV reader takes
16-bit PCM only, and the FLEURS corpus clips are 32-bit float WAV, so the
WER gates decode them through AudioToolbox (macOS); add a float-WAV path
to the reader so they also run on Linux. `scripts/speech-traces/
nemotron_trace.py` is extended by returning more arrays from `whole()`
and `streamed()`.

- `inference/src/audio/nemotron_rnnt.zig`: the prediction network (zero
  start, embedding, two LSTM layers of 640, gates stored `[i|f|g|o]`),
  the joint `head(relu(e + g))` to 13,088, greedy decoding of at most 10
  symbols per row, blank 13087, timestamps `t · 0.08 s`.
- Tokenizer: decode only, a piece table from `tokenizer.json`
  (`inference/src/tokenizer/hf_json.zig` refuses the `Precompiled`
  normalizer by name; a decode-only loader for Metaspace pieces is
  enough), the 39 language tags stripped by id and reported.
- `src/transcribe.zig` and its help: `nuclis transcribe <file>
  [--stream] [--lookahead 0|3|6|13] [--language <code>] [--backend
  cpu|metal] [--json]`; `--stream` feeds the file in 80 ms pieces and
  prints partials as they change.
- Gates: `nemotron-asr-trace-cpu` (the joint logits and token ids against
  the trace), `nemotron-asr-wer-cpu` (6 corpus clips; bound from the
  first run).

#### 7. ASR on Metal (worktree B)

From task 6: `--backend metal` returns `BackendNotYetSupported` today.
The joint is a 13,088 × 640 matvec per symbol step and the LSTM four
matvecs of 2,560 per emitted token, neither profiled. The trace's
smallest margin is 0.0107 (German, r = 3): Metal must stay within about
1e-3 on the logits to keep the token ids exact. r = 0 streaming costs
0.26× real time on the CPU (one-row chunks batch poorly). `scripts/wer.py`
takes a backend once `transcribe` passes `--backend` through. Kernels and
encoders go in their own files, as task 4's note says.

- `inference/src/audio/nemotron_metal.zig`: the encoder's chunk step and
  the joint; the LSTM steps on the CPU unless measured slow. New kernels
  (relative-position scores, GLU, the 2-D subsampling convolution) each
  with a `make test-metal` case against the CPU function.
- Measure, recorded in nemotron-asr.md: time per chunk at lookahead 3
  (4 rows, 320 ms of audio; proposed target ≤ 15 ms), the final
  transcript's delay after the speech ends, word error rate per corpus
  language against the card's figures.
- Gates: `nemotron-asr-trace-metal`, `nemotron-asr-wer-metal` (the whole
  corpus).

#### 8. Gemma 4 E4B audio input in chat

The E4B projector's 745 audio tensors are left unbound today
(`docs/models/gemma4.md`, the E4B projector line); the encoder is
`inference/src/audio/gemma4a.zig` and its Metal plan, already validated
for EmbeddingGemma 2. Bind them for E4B, accept `input_audio` parts on
user messages in `/v1/chat/completions` (OpenAI's shape: base64 `data`,
`format` `wav` or `mp3`), at most 30 s each, `<|audio>` rows `<audio|>` as
the template renders them. Muon sends the clip of each finished turn
beside its transcript, so Gemma hears the tone. Gates: the E4B audio rows on the CPU against Metal (`gemma4-e4b-audio-metal`);
the encoder's code is EmbeddingGemma 2's but its weights are E4B's own, so
rows are not compared across the two models. Behaviourally, E4B transcribes
corpus clips (`gemma4-e4b-audio-asr`, WER with `scripts/wer.py`'s
normalization), and an observation, not a gate: whether it names the tone
of clips the 1.7B voice spoke in four moods, against transcript-only.

#### 9. Serve: kinds, pinning, `POST /v1/audio/speech`

`catalog.ModelKind` gains `speech` and `transcription`, and each family a
catalogue entry (names proposed: `qwen3-tts-0.6b`, `nemotron-asr-0.6b`) so
`nuclis model pull <name>` fetches the pinned set: Muon, which bundles the
nuclis binary, sets a new user up with nothing but those names. Check that
`model pull --json` reports progress as parseable lines while it
downloads; add it if not. The server keeps
one of each open beside the language model (`src/api/memory.zig` budget);
`serve.pin` (and `--pin <name>`, repeated) names models the budget never
closes. `src/api/speech/`: OpenAI's `/v1/audio/speech` (`model`, `input`,
`voice`, `instructions`, `response_format` `pcm` or `wav`, `stream_format`
`audio` or `sse`), each codec chunk a short GPU item (`gpu.Item.short`) so
speech advances between a generation's steps (`runShort`). Gates: unit
tests of the wire and the scheduler; an end-to-end request through the
freshly built binary.

#### 10. Serve: WebSocket, streaming transcription

`src/api/http.zig` gains the RFC 6455 upgrade (bounded frames, ping,
close; text and binary messages). `GET /v1/audio/transcriptions/stream`:
binary messages carry 16 kHz mono s16le PCM; the server answers JSON
events `partial`, `final` (with word timings), `error`; each chunk a short
GPU item. (Whole files are task 15's route.) Gates: unit tests of the framing
(fragments, masking, oversize, abrupt close) and the event stream.

#### 15. Serve: file transcription for long recordings

Asked by the user (2026-10-27) for note-taking clients beside Muon.
`POST /v1/audio/transcriptions` in OpenAI's shape, under `src/api/transcription/`:

- **Request:** `multipart/form-data` (a parser in `src/api/`, bounded:
  part count, header size, one file part) with `file`, `model`,
  `language` (a locale or `auto`), `response_format` (`json` default,
  `text`, `verbose_json`, `srt`, `vtt`), `timestamp_granularities[]`
  (`word`, `segment`), `stream` (bool). This route alone takes a larger,
  explicit body limit (proposed 256 MiB, `serve.transcription_bytes`;
  `413 payload_too_large` past it), documented beside the others in
  `docs/guide/api.md` § Conventions.
- **Segments:** the transcriber's timed tokens grouped into segments at
  terminal punctuation or a pause of at least 0.8 s (proposed; a
  constant, measured on the corpus), each `{id, start, end, text,
  language}`, words `{word, start, end}`. One function in
  `inference/src/audio/nemotron_asr.zig` (or `src/`), shared with the
  CLI.
- **Long audio:** decode the upload, then feed the `Transcriber` in
  chunks of a few seconds as short GPU items (`gpu.Item.short`), so a
  two-hour file never blocks a chat turn or speech; memory stays bounded
  by the encoder's caches plus the decoded PCM. With `stream=true`, the
  response is server-sent events in OpenAI's form (`transcript.text.delta`,
  then `transcript.text.done`) plus nuclis's own progress comments, a
  segment per event as it is finalized. A client that leaves cancels it.
- **Formats:** `srt` and `vtt` written from the segments (times as
  `HH:MM:SS,mmm` / `HH:MM:SS.mmm`), `text` the joined text, `json`
  `{text}`, `verbose_json` `{text, language, duration, segments, words}`.
- **CLI parity:** `nuclis transcribe <file> --format text|json|srt|vtt`
  and `--segments`, from the same functions.
- **Gates:** the corpus WER through the HTTP route equals the CLI's
  exactly (same text per clip); a 30-minute synthetic recording (corpus
  clips joined by silences, built by a script, not committed) checks
  segment times against the joins within one 80 ms row and peak memory
  flat in duration; unit tests of the multipart parser (boundaries,
  oversize, missing file, extra parts) and of the `srt`/`vtt` writers.
  `docs/guide/api.md` § Transcription with a measured rate (audio
  seconds per second on Metal).

#### 11. Voice scheduling measured

All three models open and pinned; a script under `scripts/` plays a
recorded question into the stream, sends the transcript to chat, and
streams the answer's sentences to speech, as Muon will. Measure on this
machine and write into `docs/guide/api.md` § Measured rates: speech end to
final transcript, to first token, to first audio; E4B's decode rate with
and without speech streaming beside it (proposed: ≤ 25 % slower); TTS
staying ahead of playback. Adjust the short-item policy if speech falls
behind.

#### 12. Documents, README, website

`docs/spec.md` (§ 4 the two families, § 6 `speak` and `transcribe`, the
API routes), `docs/guide/api.md` (the speech and transcription sections),
`docs/engine/audio.md` (the new front end and encoders),
`docs/architecture.md` § The API layer, `docs/README.md`. README: a
"Speech in and out" section with the measured numbers, linking Muon as
the experimental app built on it (the link added when Muon's repository
is public). After the merge, a nuclis release carrying this unit is what
Muon's public release bundles (its tarball pinned by SHA-256 in Muon's
repository); the release itself follows AGENTS.md § Versioning, on the
user's word. Website (`site/index.html`): a section of the same, with
figures copied from the README so `make site-check` holds. Gates:
`make docs-check`, `make site-check`, screenshots of the site section.

Last, for the user (asked 2026-10-27): a starter prompt for a fresh
session that builds Muon, written to `~/Code/muon/docs/starter-prompt.md`
and committed there (locally). It points the session at Muon's
`AGENTS.md`, `TODO.md`, and `docs/design.md`; states nuclis's final API as
built (routes, catalogue names, config keys, `--pin`, `model pull --json`
progress lines, measured latencies) with links into
`docs/guide/api.md`; and names the first unit (M1) and its done-criteria.

## Deferred

Agreed and not scheduled; when one is picked up, it becomes a unit above.

- **Nemotron ASR's chunk at r = 3 under 15 ms** (from the speech unit,
  2026-10-27). Quiet machine: 32–33 ms a chunk push (an idle GPU clocks
  down between chunks), 21 ms beside another GPU job; the F32 weights'
  floor is 12.5 ms at 200 GB/s. Measure first under task 11's load (E4B
  generating beside it); then the candidates `docs/models/nemotron-asr.md`
  § On Metal names: fold LayerNorm into the following product (5,328
  dispatches a round), stage activations in threadgroup memory, fewer
  dispatches for the 1024-row matrices. Weights stay F32 (spec § 3).
- **Qwen3-TTS 1.7B at 3× real time** (same unit). 2.3× today, bandwidth
  over ~1,100 dependent dispatches a frame; candidates: one kernel per
  predictor group, or predictors batched across frames
  (`docs/models/qwen3-tts.md` § Speed).

- **Who said what: speaker diarization** (the user's, 2026-10-27, for
  note-taking). NVIDIA's `nvidia/Nemotron-3-Diarization`, a streaming
  model tracking up to 8 speakers with a speaker cache, defined in
  `transformers` (`models/nemotron3_diarization/`, at the `c0d024fb` this
  unit pinned). A new family, brought up as Nemotron ASR was (facts,
  traces, CPU, Metal); then the transcription routes' segments gain a
  `speaker` field, and the streaming route speaker changes.

- **Saluki's speed goals** (restated by the user, 2026-10-15; the numbers
  and their arithmetic in `docs/models/saluki.md`, its **Goals**): decode at
  32,639 ≥ 18 tok/s, and a 32,639-token prompt prefilled in ≤ 7 minutes.
  Decode needs the 2-bit matvecs at about 200 GB/s (IQ2_XXS reads 144
  today; the limiter is instructions per weight, not bandwidth: at
  2.26 bits a weight, 200 GB/s leaves an estimated four instructions per
  weight on the 16-core GPU; `docs/engine/apple-gpu.md` § The merged
  gate+up on Saluki's encodings). The prefill goal is the entry below. Context past
  32,768 (`max_context` in `src/config.zig`) waits on the prefill goal.
  Speculation does not reach it yet. Qwen's MTP companion binds and
  drafts on Saluki, but its verdict is off (prose geometric mean ≤ 0.993×;
  `docs/models/saluki.md` § Speculation): the 8-row verify batch on the
  2-bit encodings costs 2.1× a single-row step, against the Q4_K_M's 1.45×.

- **Long-context prefill, every family** (the user's, 2026-10-15). Prefill
  at 512 is level with llama.cpp (0.89–1.02×) and falls behind as the
  prompt grows; at 32,639: Muse Glimmer 0.79×, Qwen3.8 0.71×, Saluki
  0.65×, Gemma 4 12B QAT 0.50×, E4B 0.49×, 26B-A4B 0.42×
  (`docs/benchmarks/README.md` § The benchmarks on macOS 27,
  `docs/models/saluki.md` § Speed). The suspect is the chunk attention
  (`nu_attention_chunk` / `_h`, `Backend.attentionChunk`), 30 % of a 16K
  prefill on Qwen in 2026-09; Gemma's 512-wide global layers run a
  variant that recomputes scores per 256 value columns, which may be why
  the Gemmas trail most. Two attempts closed: sharing K/V across a
  group's heads (slower) and register-level reuse (2–5 %), both at
  ~0.7 TFLOP/s and blamed on the per-tile instruction chain
  (`docs/engine/metal-backend.md` § Long-context prefill attention, and
  § second attempt). First step: a cooled `nuclis bench --profile` of
  each family's 16K prefill to confirm the attention share per family;
  then a scratch ablation as the split pass's unit did (§ The split
  pass at the memory roof found registers, guards, and a scale inside
  the products bounding decode, not bytes or ALU), and `make
  bench-attention` to rank variants. A saved prefix does not help here:
  each run pays the prefill.

- **Video input for EmbeddingGemma 2** (about one session). Frames at
  1 fps (at most 32) through the image encoder at 140 soft tokens each,
  an AVFoundation decoder beside the image and audio bridges, oracle
  fixtures from a short generated video, gates, `--video`, and a video
  part in the API. Google's processor facts: `docs/models/embeddinggemma.md`
  § The mmproj (`Video`).
