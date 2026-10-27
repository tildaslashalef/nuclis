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

Next: **task 1, facts and pins**; nothing is implemented yet.

| Unit | Branch | State |
| --- | --- | --- |
| Speech in and out: Nemotron ASR, Qwen3-TTS, the voice API | `speech-in-and-out` | next: task 1 |

## Speech in and out: Nemotron ASR, Qwen3-TTS, the voice API

Base: `35737c3`

Pull request title: `feat: speech in and out: Nemotron streaming ASR, Qwen3-TTS, and the voice API`.

### Open decisions (the user's, before task 2)

1. **How a new architecture is validated** (open since 2026-10-14,
   `docs/models/README.md` § 0). Neither family has a nuclis path to
   compare against. Recommended, both:
   - *Data, not implementations, as the permanent gates.* ASR: word error
     rate on a pinned set of public clips with human transcripts (a few
     dozen LibriSpeech test-clean and FLEURS clips, digests pinned like the
     wikitext corpus), bound set from the first measured run. TTS: the
     round trip, ASR(TTS(text)) against text, on a pinned sentence list,
     once the ASR passes its own gate; plus the codec's frames
     reconstruct their own decode exactly between CPU and Metal.
   - *One-time bring-up traces* from the authors' published code
     (`transformers` 5.13 `Nemotron3_5AsrForRNNT`; Qwen's `qwen-tts`
     package), run once in a venv under `.reference/`, pinned as small
     fixtures under `tests/fixtures/`, as EmbeddingGemma 2's audio was
     (`docs/engine/audio.md` § Against the oracles). They catch the subtle
     errors a WER bound passes. Afterwards the CPU reference is the
     oracle for Metal, as for every family.
2. **Weights as published, or converted once.** The ASR file is F32
   (2.55 GB, `model.safetensors`); the TTS talker BF16 (1.81 GB), its codec
   F32 (0.68 GB, of which only the decoder is used). Task 1 reads which
   dtypes the Metal matvec and matmul tiles execute. Recommended: run them
   as published if the kernels exist; otherwise an explicit, documented
   `nuclis model convert`-style F16 copy, never a silent requantization
   (AGENTS.md, Inference engine rules).

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
  16 quantizers (1 semantic of 4096, 15 of 2048, dim 512), an 8-layer
  transformer of width 512 with a 72-frame sliding window, then
  upsampling [8, 5, 4, 3] × [2, 2] = 1,920 samples per frame: 12.5 frames/s
  to 24 kHz mono. CustomVoice needs no codec *encoder* (that is for
  cloning).
- **Nemotron 3.5 ASR streaming 0.6B**, `nvidia/nemotron-3.5-asr-streaming-0.6b`
  at `ea30d66d`: `model.safetensors` 2,552,062,944 B (F32), `tokenizer.json`.
  Front end: 16 kHz, n_fft 512, window 400, hop 160, pre-emphasis 0.97,
  128 mel bins. Encoder (FastConformer, cache-aware): subsampling ×8 (conv,
  256 channels, kernel 3, stride 2), 24 blocks of width 1024, 8 heads, FFN
  4096 SiLU, depthwise conv kernel 9, attention left context 57 frames,
  lookahead 0, 3, 6, or 13 frames (80 ms each; default 3). RNN-T: a 2-layer
  prediction network of width 640, ReLU joint, vocab 13,088 (blank
  13,087), at most 10 symbols per frame; a language prompt (128 slots;
  `prompt_dictionary`, `auto` = 101).

### Tasks

| # | Task | Track | Depends on | Parallel |
| --- | --- | --- | --- | --- |
| 1 | Facts and pins | — | — | no: first |
| 2 | TTS codec decoder, CPU | A | 1 | worktree A |
| 3 | TTS talker and code predictor, CPU; `nuclis speak` | A | 2 | worktree A |
| 4 | TTS on Metal, streaming, real-time factor | A | 3 | worktree A |
| 5 | ASR front end and streaming encoder, CPU | B | 1 | worktree B |
| 6 | ASR transducer and tokenizer, CPU; `nuclis transcribe` | B | 5 | worktree B |
| 7 | ASR on Metal, chunk latency | B | 6 | worktree B |
| 8 | Gemma 4 E4B audio input in chat | D | — | worktree D, any time |
| 9 | Serve: speech and transcription kinds, pinning, `POST /v1/audio/speech` | C | 4 | main |
| 10 | Serve: WebSocket transport, streaming transcription | C | 7, 9 | main |
| 11 | Voice scheduling measured: E4B, TTS, and ASR together | C | 9, 10 | main |
| 12 | Documents, README, website (links Muon) | — | all | main |

Worktree rule: a track owns its new files. Shared registrations
(`src/catalog.zig`, `gates.json`, `docs/spec.md`, `docs/README.md`,
`THIRD_PARTY_NOTICES.md`) are small appends a track may make; the main
session rebases each track onto `speech-in-and-out` and resolves them.

#### 1. Facts and pins

- Pull both repositories at the revisions above (`nuclis model pull`;
  record commit, size, SHA-256 in `docs/models/catalogue.md` § Safetensors
  artifacts) and write `docs/models/qwen3-tts.md` and
  `docs/models/nemotron-asr.md` as `docs/models/README.md` § 1 asks: every
  tensor name and shape from the files (`inference/src/formats/safetensors.zig`
  `Checkpoint`), the forward pass in equations from the authors' code
  (read as the definition, never copied), provenance per fact.
- Settle what the configs leave open: whether the talker's M-RoPE reduces
  to one-dimensional RoPE for text-and-codec sequences (all three
  position components equal) and so reuses the Qwen 3.5 path
  (`qwen35_metal.zig` notes its M-RoPE table); the talker's prompt layout
  (`im_start`, speaker, language, instruction, text, `tts_bos` …
  `codec_eos`); the predictor's sampling (`top_k` 50, temperature 1); the
  encoder's relative positional encoding and its cache layout per
  lookahead; which tokenizer model `tokenizer.json` holds.
- Read which dtypes the Metal matvec and tiles execute for safetensors
  weights (Laya's path, `laya_metal.zig`), and take decision 2.
- Write the user's validation decision into `docs/spec.md` § 3 and close
  the open line in `docs/models/README.md` § 0. Rewrite tasks 2–7 below at
  the level of files and functions with what was read; commit before code.
- Gate: `make docs-check`.

#### 2. TTS codec decoder, CPU

`inference/src/audio/qwen3_codec.zig`: `bind` (the decoder's tensors in
`speech_tokenizer/model.safetensors`) and `Runtime.decode(frames)`, codes
[n × 16] to 24 kHz f32, causal, with a streaming state so frames decode in
chunks with the same samples as one call (a unit test asserts it). Gates:
`qwen3-tts-codec-cpu` (trace fixture, per decision 1); `make test`.

#### 3. TTS talker and code predictor, CPU

`inference/src/models/qwen3_tts.zig` (binding, prompt layout) and
`qwen3_tts_runtime.zig` (CPU forward): per frame, one talker step for
group 0, then 15 predictor steps; text tokens through the Qwen2 BPE the
tree already has. `nuclis speak <text> --voice <speaker> [--instruct <s>]
[--language <l>] -o <file.wav>` in `src/speak.zig` (the CLI's `main.zig`
only dispatches). Gates: `qwen3-tts-trace-cpu`; listen to five sentences
and say so in the record.

#### 4. TTS on Metal, streaming

`qwen3_tts_metal.zig` and `qwen3_codec_metal.zig`; the talker reuses the
existing attention and matvec kernels; the predictor's 15 short steps per
frame go in one command buffer. Streaming: frames out as produced, the
codec decoding every 4 frames (320 ms) or as configured. Measure, on this
machine: first audio after the text (target ≤ 150 ms, proposed), real-time
factor alone (target ≥ 3× real time, proposed). Gates:
`qwen3-tts-trace-metal`, `qwen3-tts-codec-metal` (CPU against Metal),
`qwen3-tts-roundtrip-metal` once task 7 lands.

#### 5. ASR front end and streaming encoder, CPU

The mel front end with Nemotron's parameters: parameterize
`inference/src/audio/mel.zig` `Bank` rather than copy it (it is shared
with EmbeddingGemma 2, so its gates `embeddinggemma-audio-*` run).
`inference/src/audio/nemotron_encoder.zig`: subsampling, 24 conformer
blocks, the cache-aware state (`Stream.push(pcm)` returns the frames whose
lookahead is complete). A unit test: pushing a clip in 80 ms pieces gives
the rows one whole-clip pass gives at the same lookahead. Gates:
`nemotron-asr-encoder-cpu`.

#### 6. ASR transducer and tokenizer, CPU

`inference/src/audio/nemotron_rnnt.zig`: the prediction network, the
joint, greedy decoding (≤ 10 symbols per frame), the language prompt;
the tokenizer from `tokenizer.json`. `nuclis transcribe <file> [--stream]
[--language <code>]` in `src/transcribe.zig`, partials printed as they
come with `--stream`. Gates: `nemotron-asr-trace-cpu`, `nemotron-asr-wer-cpu`
(on a handful of the pinned clips; the CPU is slow).

#### 7. ASR on Metal

`nemotron_metal.zig`: the encoder's chunk step and the transducer.
Measure: time per 80 ms chunk at lookahead 3 (target ≤ 15 ms, proposed),
the final transcript's delay after the speech ends. Gates:
`nemotron-asr-trace-metal`, `nemotron-asr-wer-metal` (the whole pinned
set).

#### 8. Gemma 4 E4B audio input in chat

The E4B projector's 745 audio tensors are left unbound today
(`docs/models/gemma4.md`, the E4B projector line); the encoder is
`inference/src/audio/gemma4a.zig` and its Metal plan, already validated
for EmbeddingGemma 2. Bind them for E4B, accept `input_audio` parts on
user messages in `/v1/chat/completions` (OpenAI's shape: base64 `data`,
`format` `wav` or `mp3`), at most 30 s each, `<|audio>` rows `<audio|>` as
the template renders them. Muon sends the clip of each finished turn
beside its transcript, so Gemma hears the tone. Gates: an E4B audio trace
against EmbeddingGemma 2's encoder rows on the same clip (the encoder is
shared, so its rows must match), `gemma4-e4b-*` gates verify-auto selects.

#### 9. Serve: kinds, pinning, `POST /v1/audio/speech`

`catalog.ModelKind` gains `speech` and `transcription`; the server keeps
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
GPU item. Also the plain `POST /v1/audio/transcriptions` (OpenAI's
multipart form) for whole files. Gates: unit tests of the framing
(fragments, masking, oversize, abrupt close) and the event stream.

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
is public). Website (`site/index.html`): a section of the same, with
figures copied from the README so `make site-check` holds. Gates:
`make docs-check`, `make site-check`, screenshots of the site section.

## Deferred

Agreed and not scheduled; when one is picked up, it becomes a unit above.

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
