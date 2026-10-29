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

`qwen3-tts-base-voice-cloning` is planned (below) and not started: its
branch holds only this plan. Next: task 1, the facts. The next release
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
| Qwen3-TTS Base: the two Base checkpoints, voice cloning, saved voices | `qwen3-tts-base-voice-cloning` | planned |

## Qwen3-TTS Base: voice cloning with saved voices

Base: `26a006a`. The user's, 2026-10-29: the Base checkpoints clone a
voice from a reference recording (and its transcript); the catalogue's
two checkpoints are CustomVoice and their names should say so; cloning is
shown in `GET /v1/models`; a voice is encoded once and saved. One unit,
one branch.

**What exists.** `Qwen/Qwen3-TTS-12Hz-0.6B-Base` (revision
`5d83992436eae1d760afd27aff78a71d676296fc`) and `…-1.7B-Base`
(`fd4b254389122332181a7c3db7f27e918eec64e3`), read 2026-10-29 over HTTP:

- `config.json`: `tts_model_type` `base`, `speaker_encoder_config`
  `{enc_dim: 1024 | 2048, sample_rate: 24000}`; `talker_config` as the
  CustomVoice of the same size except `spk_id` and `spk_is_dialect` empty
  and the ten languages without the two dialects.
- `model.safetensors`: all BF16; the talker, text projection, head and
  code predictor tensors of the CustomVoice of the same size (402 for
  0.6B, 404 for 1.7B), plus 80 under `speaker_encoder.` (an ECAPA-TDNN:
  `blocks.0.conv` [512, 128, 5]; `blocks.{1,2,3}` each `tdnn1`, 7
  `res2net_block.blocks.N.conv` [64, 64, 3], `tdnn2`, `se_block.conv1`
  [128, 512, 1], `conv2` [512, 128, 1]; `mfa.conv` [1536, 1536, 1];
  `asp.tdnn.conv` [128, 4608, 1], `asp.conv` [1536, 128, 1]; `fc`
  [enc_dim, 3072, 1]).
- `speech_tokenizer/model.safetensors`: the catalogue's codec, SHA-256
  `836b7b35…1258`, byte-identical. Its 225 `encoder.*` tensors (a Mimi
  encoder) are bound by nobody today (`qwen3_codec.bind` skips them).
- `generation_config.json`: sampled, temperature 0.9, top-k 50, top-p
  1.0, repetition penalty 1.05, the predictor the same, `max_new_tokens`
  8192.
- Nothing local: `~/.nuclis/models/Qwen/` holds only the two CustomVoice
  checkpoints; `itshamdi404/Egy_Arabic_…-1.7B-Base` is `custom_voice`
  despite its name.

**The definition** (Qwen's package at `022e286b`, read as the definition,
never copied): `Qwen3TTSModel.create_voice_clone_prompt` makes a
`VoiceClonePromptItem` (`ref_code` T×16, `ref_spk_embedding`, `ref_text`,
`x_vector_only_mode`) that `generate_voice_clone(voice_clone_prompt=…)`
reuses. That item is what nuclis saves as a voice.

- *Speaker embedding* (`extract_speaker_embedding`): 24 kHz mono; mel
  with `n_fft` 1024, hop 256, Hann window 1024, `center` false after a
  reflect pad of 384 each side; magnitude `sqrt(re² + im² + 1e-9)`;
  librosa Slaney mel, 128 bins, 0–12 kHz; `log(max(x, 1e-5))`. Then the
  ECAPA-TDNN, every convolution `padding="same"` with reflect padding:
  TDNN (k 5) → three SE-Res2Net blocks (k 3, dilations 2, 3, 4, scale 8,
  residual) → concatenation of the three blocks' outputs → `mfa` → ReLU
  → attentive statistics pooling (`eps` 1e-12) → `fc` (1×1). Every TDNN
  is conv + ReLU. Runs in the model's dtype (BF16 in Qwen's examples).
- *Reference codes*: `speech_tokenizer.encode` at 24 kHz, the first 16 of
  32 quantizers, trimmed to `ceil(samples / 1920)` frames.
- *Prompt*: rows as CustomVoice's, with the speaker row (between
  `think_eos` and `pad`) the speaker embedding itself instead of an
  `spk_id` row. Cloning's default is `non_streaming_mode=False`
  (CustomVoice's is true), so text streams in: x-vector only, the prompt
  ends with the target's first text token + `codec_bos`, and the rest of
  the target text then `tts_eos` is added to the next rows, then
  `tts_pad`. ICL: the role and prefix rows without `codec_bos`, then the
  text track `ref_text ids ++ text ids ++ tts_eos` (`ref_text` rendered by
  `_build_ref_text`, sliced `[3:-2]`; text `[3:-5]`) summed row by row
  with the codec track `codec_bos ++ Σ_groups embed(ref_code frame)`
  (group 0 from the talker's table, groups 1–15 from the predictor's);
  the longer track's excess becomes trailing text (or `tts_pad` rows pad
  the text track when the codec track is longer).
- *Decode*: the reference codes are prepended to the generated ones and
  the first `ref_len / total` of the samples cut. nuclis's decoder is
  causal, so it runs the reference frames first and discards their
  samples: exactly `ref_len × 1920`.

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
  speaker encoder's weights SHA-256 as base64 little-endian f32). An
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

1. *Facts.* Read the Mimi encoder (`transformers` 4.57.3
   `models/mimi/modeling_mimi.py`: `MimiEncoder`, `MimiTransformerModel`
   with sliding window 250, `MimiConv1d` downsample, split RVQ encode)
   and `speech_tokenizer/config.json`'s `encoder_config`, and write
   § Voice cloning (the Base) into `docs/models/qwen3-tts.md`: the
   artifacts table for both Base checkpoints, the speaker encoder and mel
   above, the encoder's layer list with shapes, the prompt in both modes.
   Rewrite this section where the facts differ; commit before code.
2. *Traces.* Extend `scripts/speech-traces/qwen3_tts_trace.py` with a
   `clone` mode on the 0.6B Base: a short public-domain or generated clip
   (provenance in `tests/fixtures/provenance.md`) → mel, embedding,
   reference codes, both prompts' rows, the first 8 frames greedy, under
   `tests/fixtures/qwen3-tts/`.
3. *Config and binding.* `qwen3_tts.parseConfig` accepts `base` (a
   `Config.clones()`; `speaker_encoder` dims); `qwen3_tts.bind` binds
   `speaker_encoder.*` on a Base and still refuses it elsewhere; the
   tests at "parseConfig refuses…" change to accept `base`.
4. *Audio at 24 kHz.* `inference/src/audio/audio.zig` `decode` takes the
   target rate (16,000 for the encoders, 24,000 here); the WAV fast path
   reads either rate, the bridge resamples.
5. *Speaker encoder* (new `inference/src/audio/ecapa.zig`): the mel as a
   `mel.zig` `Spec` if it fits, else its own; the network over the bound
   tensors widened to F32. Against the trace's mel and embedding.
6. *Codec encoder* (`inference/src/audio/qwen3_codec.zig`: bind
   `encoder.*`, `Runtime.encode(samples) → []Frame`). Against the trace's
   codes; codes are integers, so the gate is equality, with any
   disagreement explained by a distance at a quantizer boundary.
7. *Prompt and synthesizer.* `qwen3_tts.Row` gains the speaker-vector
   row and the summed reference frame; `Prompt` gains `trailing` (text
   ids, then `tts_eos`); `Prompt.buildClone(config, request, voice)` for
   both modes; `Model.embed`, `Plan.embed` (`qwen3_tts_metal.zig`), and
   `Synthesizer` (`qwen3_tts_runtime.zig`, the per-step text row,
   including `frameOnDevice`'s `pad`) use them. `Synthesizer` primes the
   decoder with the reference frames and discards their samples. Against
   the trace's rows and frames, CPU and Metal.
8. *Voice store* (new `src/voices.zig`): add, list, remove, load,
   `ensure(model)` filling a missing codes or embedding entry; atomic
   writes; the bounds above; `NUCLIS_HOME`.
9. *Catalogue and rename.* `src/catalog.zig` `audio_entries`: the two
   renamed CustomVoice entries, the two Base entries (sizes and SHA-256
   from the Hub at the pinned revisions; `speakers` empty; a `clones`
   flag); old names refused in `src/audio/catalog.zig` `resolve`; every
   reference: `src/config.zig`, `src/help.zig`, `src/api/speech/*`,
   `src/api/root.zig`, `src/api/transcription/file.zig`, `gates.json`
   (models `qwen3_tts_06b_base`, `qwen3_tts_17b_base`), `scripts/`,
   `site/index.html`, `README.md`, `docs/`.
10. *CLI.* `nuclis voice add <name> <audio> [--text <transcript>]`,
    `voice list`, `voice rm <name>`; `nuclis speak --voice <name>` on a
    Base model loads the saved voice (`ensure`, then prompt). Help and
    completion.
11. *API.* `POST /v1/audio/voices` (multipart `name`, `audio`,
    `reference_text`; encodes on the CPU for the open or default Base
    model, `201` with the voice object), `GET /v1/audio/voices`, `GET` and
    `DELETE /v1/audio/voices/{name}`; check OpenAI's custom-voice API for
    field names first and match where it has one. `/v1/audio/speech`:
    `voice` a saved voice's name or `{"id": name}` on a Base model;
    `400 voice_cloning_unsupported` for a saved voice on CustomVoice;
    `400 invalid_request` for a speaker on a Base. `GET /v1/models`: a
    speech model's `cloning` (`{"modes": ["audio",
    "audio_and_transcript"]}` on a Base, null on CustomVoice, null before
    the pull), `voices` on a Base the saved voices' names, `speakers`
    empty, `instructions` false, `options.voice` default null. Encoding
    a voice runs off the GPU worker.
12. *Measure and document.* Encode time for a 10 s clip on both sizes;
    first-chunk latency and real-time factor with a saved voice against a
    speaker, 0.6B and 1.7B; the priming's cost per request (cache the
    primed decoder state per voice only if it exceeds 50 ms). Into
    `docs/models/qwen3-tts.md`, `docs/guide/api.md` § Speech and
    § `GET /v1/models`, `docs/guide/configuration.md`, `docs/spec.md`,
    `docs/models/catalogue.md`, the README's *Speech in and out*.

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
