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

`models-listing-for-settings` is in progress: `GET /v1/models` as the
data a client (Muon's settings) renders its model pages from. Delivered:
a registered speech checkpoint lists its voices, instructions, dtype,
name, and size; every entry carries the shared fields, modalities,
limits, and features, and absent catalogue language models are listed.
Next: `options` and `speakers` (below), once the user agrees. K-quant E4B files
(`gemma4-bf16-per-layer-projection`) waits for its pull request. The next
release (0.11.0) carries it with Q5_0 and Q5_1
([nuclis#38](https://github.com/tildaslashalef/nuclis/pull/38)), Muon's
voice loop, the model-management unit
([nuclis#36](https://github.com/tildaslashalef/nuclis/pull/36)), and the
Qwen3-TTS fine-tunes ([nuclis#41](https://github.com/tildaslashalef/nuclis/pull/41)).

| Unit | What | Sessions |
| --- | --- | --- |
| models-listing-for-settings | `GET /v1/models` describes every model well enough to render a settings page from it | 1 |

## models-listing-for-settings

Base: `8d0a941`

**Delivered.** `listAudio` (`src/api/speech/service.zig`) describes a
registry speech entry from its own files once pulled
(`audio_catalog.describeSpeech`: `qwen3_tts.parseConfig` and the
safetensors header): `voices`, `instructions`, `architecture`,
`quantization`, `name` (the repository's), `size_bytes` (weights and
codec on disk). `docs/guide/api.md § GET /v1/models` documents the speech
and transcription fields.

Then the cheap part of the listing for every client: `GET /v1/models/{id}`
(a subtree route, `Router.addSubtree`); a strong `ETag` and `304` on both
(`If-None-Match` read in `http.zig`); in every entry `schema_version`,
`source`, `download_bytes`, `pinned` (`Models.pins`), `endpoints`,
`input_modalities`/`output_modalities`, and `detail` (completed in
`models.zig`); `limits` per kind from the parsers' constants; on language
models `features` (`tools`, `hosted_tools`, `reasoning`, `streaming`,
`structured_output`), `audio` (the projector's audio encoder, as
`loadAudio` checks), `repo`/`revision`, and the catalogue's absent ones.

**Proposed, waiting on the user's review** (the sample payload was shown
in the session, 2026-10-29):

- Every kind carries `kind`, `name`, `architecture`, `quantization`,
  `size_bytes`, `present`, `loaded`, `default`, `repo`, `revision`, and
  `endpoint` (the route it serves). Decision entries lack the first four
  today, language entries `repo` and `revision`.
- `options`: one object per request field the model takes, `type`
  (`enum`, `enum_list`, `number`, `integer`, `string`, `string_list`,
  `boolean`), `values` or `min`/`max` (and `max_items`/`max_bytes`),
  `default` as this server resolves it for a request without the field,
  and `by_effort` for a language model's sampling defaults that differ by
  effort. Ranges come from the parsers: `src/api/chat/wire.zig`
  (`temperature` 0–2, `top_p` and `min_p` 0–1, `presence_penalty` −2–2,
  `repetition_penalty` 0–10, `top_k` ≥ 0, `max_tokens` 1–`max_output_tokens`),
  `src/api/speech/wire.zig`, the embeddings and transcription services;
  sampling defaults from each profile's `samplingDefaults(effort)` with
  `generation.sampling` and the registry's overrides applied.
- Speech adds `languages` (the checkpoint's `codec_language_id`);
  `voice.default` is `speech.voice` when the model has it, else null.
- Existing fields stay, so current clients are unaffected.

Open questions put to the user: marking languages a fine-tune speaks only
through `auto` (Arabic on `qwen3-tts-1.7b-eg`); per-model input limits;
pinned/memory state; the `nuclis.json` key a default comes from; a pull
hint for an absent model.

Gates: `make verify-auto` (executable and documents only; no tier).

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
