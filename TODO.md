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

One unit is in progress: Muon's three reports of 2026-11-01 (nuclis
0.11.0), on one branch and one pull request at the user's word. Its
three parts follow, in the order they are worked.

| Unit | Branch | Title |
| --- | --- | --- |
| Muon's three reports | `muon-cached-prompt-asr-loop-minimal-effort` | `fix: a cached prompt, a looping final, and a minimal reasoning effort for Muon` |

Base: `1f45a9f`.

## Muon's three reports

### A fully cached prompt is answered

**Symptom.** The same `/v1/chat/completions` request sent twice on
`gemma-4-26b-a4b`: 200, then `500 model_failed … EmptyPrompt`, and every
repeat after. Muon's report has the repro (`reasoning_effort` `none`,
system + user).

**Cause** (read, not yet reproduced). `Completer.checkpoint`
(`src/completer.zig`) rewinds to the `Mark` on a profile that
`rewritesTurn` and keeps the state under `seen`, which is now the whole
render. On the repeat, `increment(seen, full)` returns `""` (non-null),
the remainder is empty, and `Completer.encodeKeyed` → `Engine.encode`
(`inference/src/engine.zig`) returns `error.EmptyPrompt`. The same can
happen through `cache.longest(full)` returning an entry whose text is all
of `full`. The `Mark` path already guards `m.bytes < full.len`.

**Fix.** A step always feeds at least one token:
1. When the remainder would be empty and the model has no recurrent state
   (`hasRecurrentState`), step back one token by position: record the
   prompt's last two tokens when the mark is made (`Mark` gains the token
   at `position - 2`, null when the prompt fed one token), keep them past
   `checkpoint`'s rewind, and on an empty remainder call
   `inference.engine.rewindSession(eng, position - 1, before)` and feed
   `[last]` directly instead of encoding text. `seen` and
   `prompt_tokens`/`reused_tokens` stay consistent (`cached` reports
   `position - 1`).
2. Otherwise (recurrent state, or the tail is unknown): treat an exact
   match as no match: `increment` and the restore choice need a non-empty
   remainder, so the step falls back to a strictly shorter cache entry
   (the system block's save) or a reset and full prefill.

**Tests.** A unit test at the `Completer` level if one can drive a tiny
model; else in the chat service tests: the same request twice, 200 both
times, and the second's `cached_tokens` is the prompt length minus one on
an attention-only model. Exercise the repro with `./zig-out/bin/nuclis
serve` on `gemma-4-26b-a4b`, both Muon's plain case and one ending in a
`tool` message; and Qwen3.8 (recurrent) for the fallback path.

**Gates.** `make verify-auto`. No tier: the change is in `src/`, the
engine's arithmetic is untouched.

### Live transcription stops looping at lookahead 3

Waits on a fact; rewrite this section once it is read, before code.

**Symptom.** `nemotron-asr-0.6b`, the user's clip
`~/Library/Logs/Muon/turns/20261101T010020.wav` (never copied into the
repository): lookahead 3 gives "One is two plus" and "plus" ten times;
6 and 13 are correct. The ten is `max_symbols` (`nemotron_rnnt.Config
.nemotron`, from `max_symbols_per_step`): one encoder row on which the
joint never picks the blank, so the greedy loop in
`inference/src/audio/nemotron_rnnt.zig` emits the same token until the
cap.

**First, the fact.** Is the predictor state advanced after each symbol
within a row on that path (CPU and Metal, `ops/asr.zig`)? Run the clip at
lookahead 3 on the CPU and on Metal (`scripts/ws-transcribe.py --speed 0
--lookahead 3`, and `nuclis transcribe` with the CPU backend). If the
two differ, or the predictor is not stepped, it is a bug and the fix is
there. If both agree and the predictor is stepped, it is the checkpoint
with little right context, and the candidate is a guard in the greedy
loop: the same non-blank token emitted on one row more than twice ends
that row (as the blank would). Measure it against the cap alone.

**Done when.** That clip at lookahead 3 has no word more than twice in a
row; a test (the `tiny` config's test beside "Never the blank") feeds a
joint that loops and checks the guard; the transcription gate's word
error rate at lookahead 3 does not get worse
(`scripts/transcription-check.py`, and the gate in `gates.json`).
`docs/models/nemotron-asr.md` § Decisions and open items records the
finding.

### A short thinking budget per request

**Asked.** Muon retries at `low` when Gemma claims a change without a tool
call; the retry spends ~800 reasoning tokens (21 s) where the call was
settled by ~80–180. Today `effortOf` (`src/api/chat/wire.zig`) maps
`minimal` onto `low`, and the budget is the server's
`agent.thinking_budget` (1,024 at `low`, `config.thinkingBudget`).

**Design (recommended).** `reasoning_effort: "minimal"` keeps `low`'s
template and caps the reasoning at a quarter of `low`'s budget (256 at the
default), closed the way the budget already closes it. `minimal` becomes
its own value through the request (`wire.zig`'s request carries it beside
`effort`) to `src/api/chat/model.zig`, where `thinking_budget` is set;
the profile's efforts stay `off`/`low`/`medium`. `GET /v1/models`'
`reasoning_effort` option lists `minimal` where `low` exists. A
`max_reasoning_tokens` field is the alternative; not chosen, since one
named level is enough for Muon and needs no new field.

**Measure.** The four prompts in Muon's report ("Remind me to buy bread on
the way home", "Could buy bread on the way home", "I need to call the
plumber tomorrow", "Don't let me forget the dentist on Tuesday"), with
`tasks_add` among the tools, 4 seeds each, at `low` and `minimal`: tool
calls out of 16, duration, reasoning tokens; and how many replies had the
reasoning closed by the budget, and whether those still called the tool
or answered in text. The numbers go to `docs/guide/api.md` (the
`reasoning_effort` entry). Also check Muon's aside: the cache miss
between `none` and `low` is the template's thinking marker in the system
turn, so the prefix differs from the first token; say so in the same
section if it holds.

**Gates.** `make verify-auto`; the wire tests for `minimal`
(`effortOf` test, the models listing).

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

- **A short name said into the word before it** (from the several
  languages work, 2026-10-30): "يا مون" is heard as يمن or يمون under
  every prompt, and the greedy phrase boost loops ("يا يا يا…") before
  it lifts the name (docs/models/nemotron-asr.md § Several languages).
  Candidate: a small beam with the phrases as a context graph; measure on
  the 16 name clips of § Several languages.

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
