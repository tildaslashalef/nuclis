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

`qwen3-tts-base-voice-cloning` is in progress (below): tasks 1–11 (the
facts, the traces, the config and binding, audio at 24 kHz, both
encoders, the clone prompt and synthesizer, the voice store, the
catalogue and the rename, the CLI, VoiceDesign) are done and committed.
Next: task 12, the API. A side fix in the
branch: `AGENTS.md` notes that Zig 0.17 has no `**` operator (`0aabc3a`). The next release
(0.11.0) carries K-quant E4B files
([nuclis#39](https://github.com/tildaslashalef/nuclis/pull/39)), Q5_0 and
Q5_1 ([nuclis#38](https://github.com/tildaslashalef/nuclis/pull/38)),
Muon's voice loop, the model-management unit
([nuclis#36](https://github.com/tildaslashalef/nuclis/pull/36)), the
Qwen3-TTS fine-tunes ([nuclis#41](https://github.com/tildaslashalef/nuclis/pull/41)),
and `GET /v1/models` ([nuclis#42](https://github.com/tildaslashalef/nuclis/pull/42));
this unit, if it lands first.

| Unit | Branch | State |
| --- | --- | --- |
| Qwen3-TTS Base: the two Base checkpoints, voice cloning, saved voices | `qwen3-tts-base-voice-cloning` | in progress: task 12 next |

## Qwen3-TTS Base: voice cloning with saved voices

Base: `26a006a`. The user's, 2026-10-29: the Base checkpoints clone a
voice from a reference recording (and its transcript); the catalogue's
two checkpoints are CustomVoice and their names should say so; cloning is
shown in `GET /v1/models`; a voice is encoded once and saved. One unit,
one branch.

**The facts** are in [docs/models/qwen3-tts.md § Voice cloning (the
Base)](docs/models/qwen3-tts.md#voice-cloning-the-base): the pinned
revisions, sizes, and hashes; the speaker encoder (mel and ECAPA-TDNN);
the codec encoder (Mimi); both prompts; the decode. What shapes the work:

- The talker, code predictor, tokenizer, generation config, and codec
  file are the CustomVoice's of the same size, byte for byte or name,
  shape, and dtype. New: 76 BF16 `speaker_encoder.*` tensors, and the
  codec's `encoder.*`, of which only 125 (SEANet, transformer,
  downsample) are new to nuclis: the encoder's quantizer tensors equal the
  decoder's that `qwen3_codec.bind` already binds.
- A reference frame's prompt row is the row `Model.nextRow` /
  `Plan.nextRow` already make for a generated frame; ICL needs no new
  embedding path beyond the speaker vector.
- The codec encoder's attention is plain causal under the pinned
  `transformers` (its `sliding_window` 250 is read only by the flash
  class); nuclis follows that. Clips up to 10 s are the same either way.
- Nothing local: `~/.nuclis/models/Qwen/` holds only the two CustomVoice
  checkpoints; `itshamdi404/Egy_Arabic_…-1.7B-Base` is `custom_voice`.

**Decisions.**

- Cloning is decided by the checkpoint's `tts_model_type`, never by a
  name: `base` clones and has no speakers; `custom_voice` has speakers
  and refuses a saved voice. A pulled Base fine-tune clones too.
- Both modes ship: with a transcript, ICL (closer likeness); without one,
  x-vector only. The authors' streaming-text default is followed for
  both.
- A voice is model-independent and saved once under
  `$NUCLIS_HOME/voices/<name>/`: `reference.wav` (the decoded clip, 24
  kHz mono 16-bit), `voice.json` (`name`, `reference_text` or null,
  `audio_sha256`, `seconds`, `created`, `codes` keyed by the codec's
  SHA-256 as base64 little-endian u16 T×16, `embeddings` keyed by the
  model file's SHA-256 (the catalogue's, or the one the pull recorded) as
  base64 little-endian f32). An
  entry missing for the open model is computed on first use from
  `reference.wav` and written back, so one voice serves both sizes.
- Both encoders run on the CPU reference only, once per voice per model;
  no Metal kernels. The talker and codec decoder run as today.
- Bounds (host constants): clip 1–30 s after decoding, `reference_text`
  ≤ 2 KiB, name `[a-z0-9][a-z0-9_-]{0,63}` and not a catalogue speaker
  name, at most 256 voices.
- The catalogue names say the variant: `qwen3-tts-1.7b` →
  `qwen3-tts-1.7b-customvoice`, `qwen3-tts-0.6b` →
  `qwen3-tts-0.6b-customvoice`; new `qwen3-tts-0.6b-base`,
  `qwen3-tts-1.7b-base`. The old names are refused with a message naming
  the new one (no aliases). `speech.model`'s default becomes
  `qwen3-tts-1.7b-customvoice`. Breaking: the record says so, and the
  user updates their `nuclis.json` and Muon's settings.

**Tasks.**

1. *Facts.* Done: `docs/models/qwen3-tts.md` § Voice cloning (the
   Base).
2. *Traces.* Done: `qwen3_tts_trace.py --clone`, cases
   `0.6b-clone-reference`, `0.6b-clone-xvector-fox`, `0.6b-clone-icl-fox`
   (`docs/models/qwen3-tts.md` § The trace (2026-10-29)). The venv is
   `.reference/venv-qwen3-tts`; the 0.6B Base is pulled (not registered:
   `parseConfig` refuses it until task 3).
3. *Config and binding.* Done: `qwen3_tts.parseConfig` accepts `base`
   (`Config.speaker_encoder`, `Config.clones()`; the package's defaults
   when `speaker_encoder_config` is absent; `enc_dim` must equal the
   talker's width, the front end 128 bins at 24 kHz); `bind` binds the 76
   `speaker_encoder.*` tensors (`Binding.speaker_encoder`). Both Base
   sizes are pulled at the pinned revisions and open through `nuclis speak
   --model <dir>` (every tensor bound), then stop at `VoiceRequired` (a
   Base has no speakers) until task 10.
4. *Audio at 24 kHz.* Done: `audio.decodeAt(alloc, bytes, max, rate)` and
   `decodeWavAt`; `decode` and `decodeWav` stay the encoders' 16 kHz; the
   bridge (`nu_audio_decode`, `nu_audio_open`) takes the rate. `Reader`
   stays 16 kHz (transcription only).
5. *Speaker encoder.* Done: `inference/src/audio/ecapa.zig` (`Front`, its
   own 24 kHz mel; `Encoder`, the ECAPA-TDNN, CPU only, 60 ms for the
   4.6 s clip); `qwen3-tts-check <Base dir> clone` against the trace
   (worst stage relative RMS 1.7e-6); gate `qwen3-tts-0.6b-clone-cpu`
   (tier `verify`); `gates.json` models `qwen3_tts_06b_base`,
   `qwen3_tts_17b_base`.
6. *Codec encoder.* Done: `inference/src/audio/qwen3_codec_encoder.zig`
   (`bind` over the 125 tensors of its own, the file's quantizer copies
   checked equal to the decoder's; `Encoder.encode`, 658 ms for the 4.6 s
   clip); `qwen3_codec.Binding.input_proj` kept, `decodeCodebooks`,
   `Binder.at` shared. The `clone` check now holds every code equal (58 ×
   16 on the first run) and both encoders' stages within relative RMS
   1e-4 (codec encoder worst 3.3e-5, SEANet output).
7. *Prompt and synthesizer.* Done: `qwen3_tts.Row.voice`, `Clone`,
   `Prompt.trailing`/`clone`/`primer`, `Prompt.buildClone`,
   `encodeReference`; `Model.embed`/`Plan.embed` take the clone;
   `Synthesizer.init` takes `*const Prompt` (trailing text per step,
   decoder primed with the reference frames); `Voice.clonePrompt`,
   `Voice.synthesizer(gpa, *const Prompt, seed)`. The talker check replays
   both clone cases on CPU and Metal (worst 1.4e-6; F16 cache 3.3e-4); gates
   `qwen3-tts-0.6b-clone-trace-cpu`, `qwen3-tts-0.6b-clone-metal`, and the
   primed decode in `qwen3-tts-0.6b-clone-cpu`. Building a voice from audio
   (the encoders on an open `Voice`) is task 8's `ensure`.
8. *Voice store.* Done: `src/voices.zig` (`Store.add`, `list`,
   `remove`, `ensure`; `checkName`, `modelKeys`, `encodersOf`;
   `paths.voicesPath`); `Voice.speakerEmbedding`, `Voice.referenceCodes`
   in `qwen3_tts_runtime.zig`. Unit tests with fake encoders (caching per
   model and codec key, x-vector voices make no codes, every bound, a
   changed clip refused). The real-model exercise is task 10's (`voice
   add` then `speak --voice` on both Base sizes, heard back by the
   transcriber). The module doc links `docs/guide/api.md § Voices`,
   which task 12 writes.
9. *Catalogue and rename.* Done: `audio_entries` names
   `qwen3-tts-{1.7b,0.6b}-customvoice` and adds `qwen3-tts-{1.7b,0.6b}-base`
   (pinned 2026-10-29); `catalog.renamedAudio` refuses the old names in
   `audio/catalog.resolve` (speak, transcribe, serve), `model pull`, and
   `model rm`, naming the new one; a registry entry may still take them.
   Every reference renamed (code, tests, help, scripts, site, README,
   docs), the Base rows added to the README, getting-started, spec, and
   catalogue tables. Checked: `model pull qwen3-tts-0.6b-base` verifies the
   files pulled by repository; `model ls` lists all four. The user's
   `nuclis.json` names no speech model; Muon does (see below).
10. *CLI.* Done: `nuclis voice add|list|rm` (`voices.parseArgs`,
    `voices.run`, `voices.explain`), `speak --voice` on a Base (the voice
    made ready before the first-audio clock; an instruction, a speaker
    name, or an unknown voice each refused by name); help pages (`voice`;
    `speak`; the overview's speech row) and completion (`.voice`: the
    speakers once and the saved voices; `.saved_voice` for `rm`); spec and
    README. Exercised with the built binary against a scratch
    `NUCLIS_HOME`: ICL and x-vector on both Base sizes, a sentence heard
    back word for word by the transcriber each time; first use 751 ms on
    the 0.6B (both encoders), 93 ms on the 1.7B (the codes reused, its own
    embedding made), 0 ms after; an AAC clip through AudioToolbox.
11. *VoiceDesign.* Done: `qwen3-tts-1.7b-voicedesign` pinned at
    `5ecdb67` and pulled; `parseConfig` accepts `voice_design`
    (`Config.designs()`), and the 1.7B CustomVoice's path runs it with no
    speaker; `speak --instruct` describes the voice (`--voice` refused);
    trace case `1.7b-design-keeper-fox`; gates
    `qwen3-tts-1.7b-design-trace-cpu`, `qwen3-tts-1.7b-design-metal`
    (worst 1.6e-6; F16 cache 2.9e-4); docs (`qwen3-tts.md § VoiceDesign`,
    tables, help). Design then clone checked end to end: a described voice
    saved with `voice add` and cloned on the 1.7B Base, both heard back.
    The provenance line also gained task 2's missing Base description.
12. *API.* `POST /v1/audio/voices` (multipart `name`, `audio`,
    `reference_text`; saves as `voice add` does, no model opened: the
    first request in the voice encodes it; `201` with the voice object),
    `GET /v1/audio/voices`, `GET` and `DELETE /v1/audio/voices/{name}`;
    check OpenAI's custom-voice API for field names first and match where
    it has one. `/v1/audio/speech`: `voice` a saved voice's name or
    `{"id": name}` on a Base (made ready off the GPU worker before the
    request's prefill); `400 voice_cloning_unsupported` for a saved voice
    on CustomVoice or VoiceDesign; `400 invalid_request` for a speaker on
    a Base; on VoiceDesign `instructions` is the voice's description and
    `voice` is refused. `GET /v1/models`: a speech model's `cloning`
    (`{"modes": ["audio", "audio_and_transcript"]}` on a Base, null
    otherwise and before the pull), `design` (true on VoiceDesign),
    `voices` on a Base the saved voices' names, `speakers` empty on a Base
    and VoiceDesign, `instructions` false on a Base, `options.voice`
    default null where there are no speakers.
13. *Measure and document.* Encode time for a 10 s clip on both sizes;
    first-chunk latency and real-time factor with a saved voice against a
    speaker, 0.6B and 1.7B; the priming's cost per request (cache the
    primed decoder state per voice only if it exceeds 50 ms). Explain the
    1.7B's ICL prefill first: 1,514 ms on Metal for a 4.6 s reference
    (2026-10-29, `speak`), against 227 ms on the 0.6B ICL and 76 ms on the
    1.7B x-vector; the prefill timing includes the decoder's priming. The
    1.7B VoiceDesign shows it too (961 ms for about 45 rows, no priming),
    so suspect the 1.7B's Metal prefill per row, not cloning. Into
    `docs/models/qwen3-tts.md`, `docs/guide/api.md` § Speech and
    § `GET /v1/models`, `docs/guide/configuration.md`, `docs/spec.md`,
    `docs/models/catalogue.md`, the README's *Speech in and out*.

**When the unit is done** (the user's, 2026-10-29): `make install` so
Muon (`~/Code/muon`) runs the new build; then message the agent developing
Muon (its session, "Nuclis TTS base models and voice cloning") with the
changes, the API (the voice routes, `/v1/audio/speech`'s `voice`, and
`GET /v1/models`' `cloning`, `voices`, `speakers`, `instructions`), and the
answers to its three questions: a clip is registered once as a named voice
(`POST /v1/audio/voices`), not sent with every request; the clip's limits
(1–30 s after decoding, the formats `audio.readable` names, a transcript
≤ 2 KiB); a Base takes no instructions (Qwen's cloning call has none), so
moods show on the face only, as on the 0.6B CustomVoice. Tell it about
VoiceDesign (task 11): the brain (Gemma 4) can write the voice description
per character or mood; a description is sampled anew each request, so for
a voice that must stay the same, design once and save the clip as a Base
voice (`POST /v1/audio/voices` with its text). Also tell it
the rename: Muon names `qwen3-tts-1.7b` and `qwen3-tts-0.6b` in
`Sources/MuonCore/Engine/EngineModels.swift` (and its tests), which nuclis
now refuses with `404 model_not_found`; they are
`qwen3-tts-1.7b-customvoice` and `qwen3-tts-0.6b-customvoice`. **No pull
request until the user says so.**

**Gates.** `zig build test`; `zig build test-qwen3-tts -- <dir> clone` (a new mode beside `codec` and `talker`); new gates
`qwen3-tts-0.6b-clone-cpu` (embedding, codes, rows, 8 frames against the
trace) and `qwen3-tts-0.6b-clone-metal` (Metal against the CPU, then a
saved-voice round trip through `nuclis speak` whose transcription by
Nemotron ASR matches the text); `make verify-auto` before each commit;
`make verify` once; `make verify-cpu` once (the unit changes
`qwen3_tts_runtime.zig`'s forward and adds two CPU encoders); `make
api-check` with the voice routes; `make docs-check`, `make site-check`.

## Deferred

Agreed and not scheduled; when one is picked up, it becomes a unit above.

- **MTP slows Gemma 4 26B-A4B down** (the user's, 2026-10-28). The
  catalogue's verdict is off: 0.63–0.76× on prose at drafts 2–7, the
  verify growing from 60 to 98 ms with its rows
  (`docs/engine/speculative-decoding.md`, the 26B-A4B verdict). Again on a
  K-quant 26B-A4B file with the same `mtp-gemma-4-26B-A4B-it.gguf` head,
  2026-10-28, M4 Pro, greedy, 300-token story, draft 5: 50.5 tok/s off,
  27.6 on (0.55×), 164 of 368 drafts accepted. The head's card reports
  about +35 % under llama.cpp. First step: measure llama.cpp on the same
  machine, files, and prompt (`llama-server -md <head> --spec-type
  draft-mtp`, `docs/benchmarks/llama-cpp.md` for the build) to tell
  acceptance from verify cost; then a cooled `nuclis bench --profile` of
  one verify step at 2, 4, and 6 rows. Suspect: the verify's expert work
  reads an expert's weights once per row that selects it rather than once
  per batch (the gathered tile groups rows by expert, the per-token
  matvecs do not). llama.cpp is a speed comparison here, never a source to
  copy.

- **A short word after silence is lost** (from Muon's voice loop,
  2026-10-28): the checkpoint's own borderline call, the same on CPU and
  Metal, live and whole-file (`docs/models/nemotron-asr.md` § Decisions
  and open items). A blank penalty recovers the words but changes heard
  spellings ("Hello." → "Hallo."), so it is not shipped. Candidate: only
  when an utterance's final would be empty, decode its rows again with the
  penalty, and mark the final as recovered; measure on Muon's recordings
  (kept on the user's Mac, never in the repository) and with
  `scripts/wer.py`.

- **A language change within one recording loses words** (from the speech
  unit, 2026-10-27): a mixed 30-minute file scored 19 % WER against 6.3 %
  for its clips alone, on the CPU and Metal alike, and two segments
  crossed a join (`docs/guide/api.md` § Long recordings). Candidate:
  reset the transcriber after a long silence; measure with
  `scripts/transcription-check.py long --languages all`.
- **The vision encoders' scores kernel** (same unit): ~1.7 TFLOP/s, its
  32×32 tile with one 64-wide head as the inner dimension, now the
  largest piece of E4B's encode (0.54 s of 1.64 s); 72-wide heads waste a
  quarter of the values kernel's last block; Muse's encode is 6.0 s of
  K-quant matmuls and 0.86 s of unfused bias adds
  (`docs/engine/vision.md`).
- **Link Muon** from the README's *Speech in and out* and the site's
  *Voice* section once its repository is public.

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
