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

`serve-request-log-and-speech-budget` is planned (below) and not started:
its branch holds only this plan. Next: task 1, the speech frame budget.

| Unit | Branch | State |
| --- | --- | --- |
| The speech frame budget, and a request log for `nuclis serve` | `serve-request-log-and-speech-budget` | planned |

## The speech frame budget, and a request log for `nuclis serve`

Base: `55e015c`. Raised by Muon's agent, 2026-10-29, on
0.11.0-dev+8dbc2f7: a clip-only voice on the 0.6B Base turned "Hello."
into 655.28 s of audio (8,191 frames: the model never sampled the end
code, and the only cap is `generation_config.json`'s `max_new_tokens`
8192, every speech model's), visible only when the request ended four
minutes later. The user, 2026-10-29: a separate pull request from #43; the
log covers every API route but `/v1/health`, never a request's or a
response's payload, only structured data on what nuclis did; requests
carry ids and clients' correlation ids.

**The budget, measured** (2026-10-29, `nuclis speak`, Metal, M4 Pro;
frames of 80 ms per text token of the input):

| Input | Frames per token | Most frames |
| --- | ---: | ---: |
| 20 English sentences, 0.6B CustomVoice `ryan` | 3.2–11.6 | 104 (9 tokens) |
| 3 Chinese sentences, 0.6B CustomVoice `vivian` | 4.9–7.5 | 49 |
| 8 sentences, 1.7B VoiceDesign "very, very slowly" | 6.4–14.5 | 145 (10 tokens) |
| "Hi.", "Okay.", "Hello.", "Yes.", "Thanks!" (2 tokens), both | 6.5–37.5 | 75 ("Hello.") |

**Decisions.**

- A synthesizer stops at `min(max_new_tokens, 100 + 20 × text tokens)`
  frames: 11.2 s for "Hello." (1.9× the longest short line measured), 24 s
  for 10 tokens (2× the slowest sentence). The constants live beside
  `default_chunk` in `qwen3_tts_runtime.zig`; the text tokens are the
  target's (`Prompt.text_tokens`), not a reference transcript's.
- The finish reason, `eos`, `length` (the budget or the cap), or
  `cancelled`, is the synthesizer's (`Synthesizer.finish`), shown by
  `nuclis speak`, in `speech.audio.done` (`finish_reason`), and logged.
- The request log is `<root>/logs/serve.jsonl`, written by `nuclis serve`
  whoever started it, JSON lines, rotated at 16 MiB to `serve.1.jsonl` …
  `serve.4.jsonl`. The stdout line stays as it is (`serve.log`); a new
  `serve.request_log` (default true) turns the file off. A write failure
  never fails a request.
- Two lines a request: `start` (id, correlation id, time, method, route)
  once it is routed, so a request still running is in the file; `end`
  (status, duration, bytes out, the error's code, the route's fields).
  `GET` and `HEAD /v1/health` are not logged (pollers); every other route
  is. A WebSocket logs its `start`, then its `end` when the session closes.
- Ids: every request gets a UUIDv7 (time-ordered), returned as
  `x-request-id` on every response (OpenAI's header; errors and streams
  included) and logged as `id`. A client's `X-Client-Request-Id`
  (OpenAI's SDK header; printable ASCII, at most 128 bytes, else ignored
  and marked) is logged as `client_id` and echoed back, so one client turn
  (transcription, chat, speech) reads as one thread.
- Never a payload: no text, message, image, clip, vector, or audio, in or
  out. Sizes, counts, durations, names nuclis resolved (model, voice,
  language), and nuclis's own error codes and messages. Opt-in bodies are
  not built.
- `GET /v1/health` gains `in_flight`: id, client id, route, model, running
  for, and the route's progress (audio seconds so far, tokens so far).

**Tasks.**

1. *The speech budget.* `Prompt.text_tokens` (set by `build` and
   `buildClone`), `Synthesizer.budget` and `finish` (`eos`, `length`;
   `cancelled` set by the service); `nuclis speak` prints a stop at the
   budget; `speech.audio.done` gains `finish_reason`; the log note shows
   it. Tests: a tiny model stops at the budget with `length`; the trace
   gates unchanged.
2. *Route inventory.* Read every handler (`chat`, `decisions` and
   `systemone`, `embeddings`, `models`, `audio/speech`, `audio/voices`,
   both transcription routes) and write, into `docs/guide/api.md` § The
   request log, each route's fields: what it resolved (model, kind,
   voice, language, mode), sizes in and out (tokens, inputs, questions,
   audio seconds, frames, bytes), timings (queued, prefill, first output),
   and outcomes (finish reason, cache reuse, web calls). Rewrite tasks 3
   and 4 at that level, committed before code.
3. *The log.* `src/api/request_log.zig`: the file, rotation, the two
   lines; the ids (UUIDv7 from `io.random` and the clock; the client's
   header read and checked) in `http.zig`, `x-request-id` and the echo on
   every response; a typed `Fields` each handler fills (a fixed set of
   named numbers, strings, and booleans, nothing else), carried by the
   response and its stream's end; `serve.request_log` in `config.zig`.
4. *Every route fills its fields* (task 2's list); the in-flight table
   behind `GET /v1/health`.
5. *Docs and checks.* `docs/guide/api.md` § The request log and §
   `GET /v1/health`, `docs/guide/configuration.md`, `docs/spec.md`; the
   correction of § Speed in `docs/models/qwen3-tts.md` and § Voices: with
   a transcript, the first audio grows with the reference clip (Muon's
   0.6B: 430 ms for 10 s, 630–680 ms for 15 s; 260 ms for 4.6 s), a
   clip-only voice does not (45 ms). `make api-check` gains the ids and a
   read of the log.

**When the unit is done:** tell Muon's agent ("Nuclis TTS base models and
voice cloning") that it is in, where the log is, and the ids. **No pull
request until the user says so.**

**Gates.** `zig build test`; `make verify-auto` before each commit (the
speech gates, the API's); `make api-check`; `make docs-check`. No numeric
change to any model: `make verify-cpu` and `make verify` are not called
for, the budget only stops generating sooner.

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
