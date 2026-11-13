<!-- kind: record -->
# The nuclis API: the record

The dated record behind [the API guides](../../guide/api.md): measurements
and the reasons behind constants, each true on its date.

## Sections moved from the guides, 2026-10-08 to 2026-11-01

### Memory: counted against the estimate, 2026-10-30

Measured on the M4 Pro (2026-10-30)
with four models pinned at `--memory 47`, every listed number equalled
`/v1/health`'s: the Gemma 4 26B-A4B with its draft 24,781,970,336 B
(before its first open the estimate was 21,090,982,816 B, the file and the 4 GiB of saved states: 15 % under),
drama-3 20,787,264,352 B, `nemotron-asr-0.6b` 2,603,784,672 B, and
embeddinggemma-2 309,855,520 B. Opened once each, the others counted:

| Model | Counted | Estimate before |
| --- | ---: | ---: |
| `qwen3.8-27b` | 23,583,635,232 | 20,759,407,520 |
| `gemma-4-26b-a4b` | 22,235,001,920 | 18,544,014,400 |
| `gemma-4-12b-qat` | 16,902,177,472 | 11,011,324,096 |
| `gemma-4-e4b-qat` | 9,509,865,184 | 8,510,663,072 |
| `qwen3-tts-1.7b-base` | 4,539,706,836 | the same |
| `qwen3-tts-1.7b-customvoice`, `-voicedesign` | 4,515,695,644 | the same |
| `qwen3-tts-0.6b-base` | 2,511,637,364 | the same |
| `qwen3-tts-0.6b-customvoice` | 2,493,919,668 | the same |

### A short reasoning: `minimal`, 2026-11-01

Measured 2026-11-01 (M4 Pro, `gemma-4-26b-a4b`, Muon's real requests:
a ~1,430- and a ~1,910-token prompt with seven tools, four prompts such
as "Remind me to buy bread on the way home", four seeds each, every
request after the same one at `none`); useful means it called
`tasks_add` or asked back:

| Effort | Useful | Median time | Reasoning closed by the budget |
| --- | --- | --- | --- |
| `low` (1,024) | 31 of 32 | 14–20 s | 13 times; 12 still called or asked |
| `minimal` at 256 | 20 of 32 | 5.5 s | every time; 12 answered in text, claiming a change it never made |
| `minimal` at 512 (shipped) | 28 of 32 | 10 s | 27 times; 4 answered in text |

### Conversations and the cache, 2026-10-08

Measured with `nuclis serve` (M4 Pro, temperature 0, 2026-10-08, the
same answers before and after; `cached_tokens` and request time):

| Request | Before | After |
| --- | --- | --- |
| Gemma 4 12B, second turn, reasoning `none` | 0 of 58 | 30 |
| Gemma 4 12B, second turn, `low` | 0 of 56 | 32 |
| Gemma 4 E4B, second turn, `low` | 0 of 49 | 32 |
| Gemma 4 12B, the same request again | 0 of 37 | 30 |
| a new question under a 500-token system prompt (E4B, 12B, Qwen3.8) | 0 | 486–492; 3.96 → 1.75 s on the 12B, 10.3 → 5.2 s on Qwen3.8 |
| Gemma 4 12B, third and fourth turns of that conversation | 0 of 573, 0 of 636 (3.9, 4.3 s) | 498, 566 (1.7, 1.8 s) |

The first request under a new system prompt pays for keeping it: 0.25 s
on the 12B, 0.5 s on Qwen3.8.

### A request sent again after a tool result, 2026-11-01

Where the last prompt ended in a tool result, that point is the whole prompt: the request sent again re-runs its last token for the next one's logits (Gemma 4 26B-A4B, 2026-11-01: 108 of 109
tokens cached; on 0.11.0 it was `500 model_failed`, `EmptyPrompt`).

### A decision beside a generation, 2026-10-08

Five
`laya-multilingual` decisions sent during a 2,000-token Qwen3.8
generation were answered in 51 to 164 ms (136.5 s before this), and the
generation took 139.5 s (141.4 s without them).

### pi on Qwen3.8, 2026-10-08

With this entry, pi 1.1.0 on Qwen3.8 did one of the agent playground's
tasks (a new module, its export, its tests, and a test run) in five
requests. The first prefilled pi's 7,297-token system prompt and tools in
109 s; each later one continued the session, reusing everything the
previous one had consumed (7,582 of 8,050 tokens, then 8,337 of 9,181,
10,251 of 10,357, and 10,419 of 10,591). Esc in pi cancelled a request
in its prefill.

### Hosted rounds reuse the state, 2026-10-16

The model's state carries over between completions, so a round
reuses the conversation before it (measured: 1,257 of 2,084 prompt tokens
over three rounds, Saluki, against the stub, 2026-10-16).

### Text before a call, 2026-10-28

Text the model writes before a call is rendered after the call's results;
the state is kept where the two still agree. Measured on Muon's conversation (E4B, reasoning `none`, against
the stub, 2026-10-28): the turn after a search reused 1,214 of 1,784
prompt tokens in 2.22 s before, 1,723 of 1,785 in 1.04 s after; the
26B-A4B, which rewrites its turns anyway, reused 1,701 of 1,773 either
way.

### Phrases to expect, 2026-10-28

Measured on three recorded utterances of a name the model misheard each
time (2026-10-28, `en-US`, r = 3, the same on both routes): two came back
spelled right; the third stayed wrong, its word never beginning with the
name's first piece. Twenty unrelated names listed on the 60-clip speech
corpus moved the word error rate from 6.11 % to 6.19 % (74 to 75 errors
in 1,211 words).

### Several languages, 2026-10-30

On 40 short Egyptian Arabic clips and 13 English turns, `ar,en,auto`
wrote every English turn as `auto` does and the Arabic with a mean
character error of 0.229 (`auto` 0.252, with three in another script:
Devanagari, Hangul, Latin; `ar` 0.230, with every English turn empty).
Two languages cost about 8 % over one, three with `auto` about 33 %
(0.021× real time). docs/models/nemotron-asr.md § Several languages.

### Segments, 2026-10-27

The pause was proposed at 0.8 s and measured on the speech corpus's 60
clips, each one or two sentences (703 word boundaries): at 0.8 s, 14
sentences split mid-way (5 at a comma); at 1.04 s, 4; at 2 s, none.
Mid-sentence pauses of 0.8–1 s are common in read speech, so the
constant is 1.04 s (13 rows). On the 30-minute recording below, every
join of 1.5 s or more is a segment boundary, and the pause carries much
of the work: in a continuous stream the model ends fewer sentences with
punctuation than it does a clip alone (110 of the 258 segments end in
`.`, `?`, or `!`, each followed by its language tag).

## Measured rates, 2026-10-02 to 2026-11-07

Apple M4 Pro (48 GB), ReleaseFast, Zig 0.16.0, macOS 27.0; `nuclis serve
--model laya --model laya-multilingual` on Metal, on the port of the
time (8735) and before the request log existed; ApacheBench 2.3 with
keep-alive (`ab -k`) on loopback; checkpoints `convaiinnovations/laya` at
`55cf4c4e`. The decision request is one state (93 characters, a billing
complaint) and two questions (a two-option choice and a noul);
latencies are ab's percentiles (`-e`, which keeps fractions of a
millisecond). The request:

```json
{"questions": {"team": {"type": "choice", "instructions": "Which team should handle this?",
   "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages, errors"}},
  "urgent": {"type": "noul", "instructions": "Is this urgent?"}},
 "state": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."}
```

with `"model": "laya-multilingual"` added for that checkpoint.

### One GPU job per request (before batching), 2026-10-02, commit `cb79cde`

| Route | Model | Concurrency | Requests | Requests/s | p50 ms | p99 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `GET /v1/health` | — | 1 | 1,000 | 25,905 | 0.019 | 0.082 |
| `GET /v1/health` | — | 16 | 20,000 | 146,801–166,589 | 0.096 | 0.187 |
| `POST /v1/decisions` | laya | 1 | 512 | 28.8 | 34.4 | 36.4 |
| `POST /v1/decisions` | laya | 4 | 512 | 28.9 | 136.4 | 153.0 |
| `POST /v1/decisions` | laya | 16 | 512 | 29.4 | 543.2 | 544.2 |
| `POST /v1/decisions` | laya-multilingual | 1 | 512 | 69.2 | 14.5 | 14.9 |
| `POST /v1/decisions` | laya-multilingual | 4 | 512 | 71.1 | 55.6 | 58.5 |
| `POST /v1/decisions` | laya-multilingual | 16 | 512 | 72.0 | 222.2 | 222.9 |

What the server adds: a warm request's HTTP p50 against the encode the
response reports (median of 40, `timings_ms.encode`), tokenizing 0.1 ms:
laya 34.05 ms against 34.5 ms, laya-multilingual 13.97 ms against
14.3 ms, so nothing measurable (the targets were within 2 ms). The same
request through `nuclis decide --request … --json`, a process per call:
0.09–0.10 s (laya), 0.20–0.21 s (laya-multilingual), wall clock.

### Batched across requests, 2026-10-02, commit `cd35ed8`

Same machine, build mode, request, and protocol:

| Model | Concurrency | Requests/s before → after | p50 ms before → after | p99 ms before → after |
| --- | ---: | --- | --- | --- |
| laya | 1 | 28.8 → 29.2 | 34.4 → 34.2 | 36.4 → 34.7 |
| laya | 4 | 28.9 → 33.8 | 136.4 → 118.3 | 153.0 → 118.9 |
| laya | 16 | 29.4 → 36.4 (1.24×) | 543.2 → 439.3 | 544.2 → 444.3 |
| laya-multilingual | 1 | 69.2 → 71.2 | 14.5 → 14.0 | 14.9 → 14.5 |
| laya-multilingual | 4 | 71.1 → 85.1 | 55.6 → 46.9 | 58.5 → 47.5 |
| laya-multilingual | 16 | 72.0 → 93.4 (1.30×) | 222.2 → 171.0 | 222.9 → 172.1 |

At concurrency 16 a pass held 7.9 requests on average (512 in 65 passes,
`GET /v1/health` counters), about 820 rows. The gain is the model's: one
request already encodes at 3,020 rows/s (laya, 104 rows in 34.4 ms) and
7,160 (laya-multilingual, 101 in 14.1 ms), against the packed rates
[laya.md § Time per call](laya.md#time-per-call-2026-09-29) measures, about 3,700
and 8,660, so packing can add about 1.2× here, and does. A 2× gain
would need a faster Laya encode, not a different server. p99 is about two
passes (laya 444 ms against 2 × 217 ms, laya-multilingual 172 against
2 × 85): a request that just missed a pass waits for it and runs in the
next.

Batching changes timing, never answers: 16 concurrent requests with
different questions (2 states each, choice, noul, and score questions)
returned the same bytes as each sent alone, timings aside, on both
checkpoints on Metal (16 in 3 and in 4 passes), and the unit test
asserts the same on the CPU.

### Embeddings, 2026-10-09

Apple M4 Pro (48 GB), ReleaseSafe (`make metal`), Zig 0.17.0, macOS 27.0;
`nuclis serve` on Metal with `embeddinggemma-2` (the Q8_0 file and
`unsloth/embeddinggemma-2-GGUF` at `031f0d4b`), commit `7d37e93`. Each input is `<bos>`,
" the" repeated, `<eos>` (the cost does not depend on the text), sent as
one request with `encoding_format: base64`; one warm-up, then the median
of 7, from Python's `urllib` on loopback. `embed` is the response's
`timings_ms.embed`, the passes alone.

| Request | Tokens | Wall ms | `embed` ms | Inputs/s |
| --- | ---: | ---: | ---: | ---: |
| 64 inputs × 256 tokens | 16,384 | 1,686 | 1,674 | 38.0 |
| 1 input × 512 tokens | 512 | 60 | 59 | 16.8 |
| 1 input × 8,192 tokens | 8,192 | 1,756 | 1,748 | 0.6 |

The server adds about 12 ms to the 64-input request (parsing, the queue,
and 64 base64 vectors), and the batch runs at the rate the encoder has
alone (1,671 ms for the same inputs in
[embeddinggemma.md § The Metal plan](embeddinggemma.md#the-metal-plan-2026-10-09)).
Passes of 2,048 rows cost no throughput: the same server with 8,192-row
passes gave 1,669 ms and 38.3 inputs/s, within the spread, while a
2,048-row pass holds the GPU for about 0.2 s instead of 1.7 s.

### Speech beside a chat stream, 2026-10-27, commit `07dac34`

Apple M4 Pro (48 GB), ReleaseSafe (`make build`), Zig 0.17.0, macOS
27.0; `nuclis serve --quiet --model gemma-4-e4b-qat --pin qwen3-tts-1.7b-customvoice
--pin nemotron-asr-0.6b` on Metal (`unsloth/gemma-4-E4B-it-qat-GGUF` at
`8c5a9e4f`, speculative decoding on at draft 6;
`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` at `0c0e3051`). **The machine was
not quiet**: another session's builds held the CPU (load average 3–4.6),
and `nuclis speak --stream` of the first sentence ran at 1.65× real time
then against 2.3× on a quiet machine
([qwen3-tts.md § Speed](qwen3-tts.md#speed-2026-10-27)); the server's
speech alone ran at 1.72×, so the server adds nothing measurable, and
every rate below is low by that margin. A Python client (`http.client`
on loopback) sent five sentences of 3–5 s of audio, one after another,
each `"response_format": "pcm"`, `"seed": 1`; first byte is the first
audio byte of the body after the request was sent, the real-time factor
the audio's length over the request's wall time. The chat is one streamed
request, `"reasoning_effort": "none"`, temperature 0, 1,200 tokens of a
story; its rate is content chunks per second (a chunk is a token but for
a held split character), between the first and the last of a window.

| | First audio byte (median, range) | Real-time factor (median, range) | E4B decode |
| --- | --- | --- | --- |
| speech alone | 99 ms (87–107) | 1.72× (1.32–1.78) | — |
| E4B alone | — | — | 43.7 tokens/s |
| speech while E4B streams | 135 ms (129–152) | 1.49× (1.43–1.53) | 5.4 tokens/s during the speech, 44.3 after |

Speech beside a generation stays faster than real time and starts within
about 40 ms of alone; the generation pays: each of its steps waits for a
four-frame chunk, so it ran at an eighth of its rate while the 21.1 s of
speech were made (in 14.2 s), and at its own rate after. Audio made beside the
generation was byte-identical to the same request alone. How to share the
GPU more evenly is measured in § The voice loop below, whose policy
replaced this one.

### Live transcription alone and beside a chat stream, 2026-10-27, commit `18216d7`

Apple M4 Pro (48 GB), ReleaseSafe (`make build`), Zig 0.17.0, macOS
27.0.1; `nuclis serve --quiet --model gemma-4-e4b-qat --pin
nemotron-asr-0.6b` on Metal (`unsloth/gemma-4-E4B-it-qat-GGUF` at
`8c5a9e4f`, speculative decoding off as this machine's
`generation.speculative` sets it; `nvidia/nemotron-3.5-asr-streaming-0.6b`
at `ea30d66d`). **The machine was not quiet**: another worktree's
`nuclis serve` and `nuclis transcribe` (file transcription on the same
GPU) ran for about 80 % of each run, sampled once a second, and its
compiler for some of it. `python3 scripts/gate/ws-transcribe.py --measure
--clips librispeech-6829-68769-0000,fleurs-de_de-1660 --beside-chat
gemma-4-e4b-qat --rounds 3`: one connection, lookahead 3, each clip under
its locale, sent in 20 ms messages each when its audio has ended at real
time (as a microphone delivers it), then `finish`; one warm-up, then each
clip three times (159 partials). A partial's delay is its arrival behind
the moment the audio it covers had ended; the final's is its arrival
after `finish` (sent with the last message). The chat is one streamed
request, `"reasoning_effort": "none"`, temperature 0, 1,200 tokens of a
story; its rate is content chunks per second between the first and the
last of a window (alone: the whole answer; beside: from the clip's first
message to its final). Two runs:

| | Partial behind the audio, median / p90 / max | Final after `finish`, median (range) | E4B decode |
| --- | --- | --- | --- |
| transcription alone | 42 / 47 / 60 ms; 29 / 31 / 33 ms | 31 ms (24–34); 20 ms (16–20) | — |
| E4B alone | — | — | 48.1; 45.8 tokens/s |
| transcription while E4B streams | 50 / 66 / 74 ms; 49 / 65 / 81 ms | 48 ms (33–60); 41 ms (33–60) | 41.8; 42.1 tokens/s during the clip (34.4–43.3 over the six), 44.7; 44.3 after |

Live transcription keeps up with the speaker beside a generation: every
partial arrived at most 81 ms after the audio it covers (medians 29 to
50 ms), every final at most 60 ms after the turn ended. The generation
pays 8 to 13 % of its rate while a clip streams (a step may wait for one
chunk of about 20–35 ms every 320 ms), against speech's eightfold
slowdown ([above](#speech-beside-a-chat-stream-2026-10-27-commit-07dac34)).
An earlier run on this branch, whose other jobs were not logged,
measured E4B at 56.4 tokens/s alone and 54.8 while a clip streamed. Both
clips are among the `nemotron-asr-stream-route` gate's four, whose
finals equal `nuclis transcribe`'s text.

### File transcription, 2026-10-27, commit `c4fe892`

Apple M4 Pro (48 GB), ReleaseSafe (`make build`), Zig 0.17.0, macOS 27.0;
measured on the same file-route code before its rebase onto the live
route (then `4390fa8`); a fresh `nuclis serve` with `nemotron-asr-0.6b`
open on Metal (`nvidia/nemotron-3.5-asr-streaming-0.6b` at `ea30d66d`), lookahead 3,
`auto`. **The machine was shared**: another session's builds and tests
held it (load average 2–7), and the same run measured 42–64× real time
from one run to the next. The recording is `scripts/gate/transcription-check.py
long`'s: the 40 LibriSpeech clips of the speech corpus in a seeded order,
each followed by 1.5–3 s of silence, 16-bit 16 kHz mono WAV (30 minutes:
1,802.3 s, 55.0 MiB). Rate is the audio's length over the request's
wall time (the upload over loopback included); first segment is the
time from sending the request to the first `transcript.text.segment`
event with `stream=true`; RSS is `ps`'s, sampled every 50 ms.

| | 5 minutes | 30 minutes |
| --- | --- | --- |
| rate, alone (runs) | 42.7–62.5× (3) | 42.4–63.8× (6) |
| first segment, `stream=true` | 0.17–0.26 s | 0.20–0.29 s |
| peak RSS (idle 212–217 MiB) | 322–327 MiB | 368–374 MiB |
| segments; words differing from the clips alone | 44; 2.18 % | 258; 1.98 % |
| WER against the references (the clips alone) | 5.47 % (4.06 %) | 5.04 % (4.08 %) |

Beside a chat: the same server with `gemma-4-e4b-qat` open too, one
streamed chat (1,200 tokens of a story, temperature 0, `reasoning_effort`
`none`) started 2 s before the 30-minute file and running through it,
then the same chat alone. Its rate is content chunks per second, as in
the speech measurement.

| | Transcription | E4B decode |
| --- | --- | --- |
| alone | 42–64× real time | 45.8–51.5 tokens/s |
| together | 28.4–40.9× real time | 12.0–17.9 tokens/s |

The transcript beside the chat equals the one alone. The chat pays as
it does for speech: each of its steps waits for a 2.56 s piece, 40–64 ms
at the rates above. Smaller pieces cost nearly as much each: through
`nuclis transcribe` on the same machine and file, pieces of 8, 16, 32,
and 64 rows ran at 19, 25, 40, and 68× real time, so a piece of 8 rows
still took about 34 ms (each emitted token adds a command buffer,
[nemotron-asr.md § On Metal](nemotron-asr.md#on-metal-2026-10-27)). § The voice loop below measures the share
that replaced this.

### The voice loop, 2026-10-27, commit `f96f750`

Apple M4 Pro (48 GB), ReleaseSafe (`make build`), Zig 0.17.0, macOS
27.0, a quiet machine (no other GPU job; load average 3–4 from the
desktop); `nuclis serve --quiet --model gemma-4-e4b-qat --pin <voice>
--pin nemotron-asr-0.6b` on Metal, through `scripts/app/voice-loop.py` and
`scripts/gate/ws-transcribe.py --measure`, `scripts/gate/transcription-check.py
long --chat`. Chat rates are streamed content chunks per second as in
the records above (`reasoning_effort` `none`, temperature 0, 1,200
tokens of a story). Playback is modelled: audio plays at real time from
its first byte, each speech request after the one before it; the lead is
the audio in hand when a chunk arrives, an underrun a chunk that arrives
after it ran out.

**Beside a chat.** Five sentences (28.1 s of audio on the 1.7B, 30.8 s on
the 0.6B), each its own request, sent together while the chat streams;
two rounds each.

| | E4B alone | E4B beside it | It beside E4B |
| --- | --- | --- | --- |
| speech, 1.7B, before (1 frame then chunks of 4, every run) | 55.9 tokens/s | 7.1–7.3 tokens/s | 2.06–2.09× real time; a 50–80 ms underrun after the first 80 ms of each utterance, alone too |
| speech, 1.7B | 53.9–56.4 tokens/s | 23.7–26.8 tokens/s | real time with a second in hand (lead median 2.0 s, min 40 ms), no underrun; first byte 96–118 ms |
| speech, 0.6B | 55.3 tokens/s | 32.5–33.6 tokens/s | the same, no underrun; first byte 45–103 ms |
| live transcription (2 clips × 3 rounds at real time) | 55.2 tokens/s | 52.2 tokens/s (48.1–53.1 per clip) | partials 41 ms behind the audio (median; p90 54, max 61), final 33 ms after `finish` (alone 53 and 37) |
| a 5-minute file | 52.5 tokens/s | 40.6 tokens/s | 12.1× real time (2.18 % of words differ from the clips alone, as before) |

Speech needs what playback needs: one frame of the 1.7B takes about
37 ms a run for 80 ms of audio, 46 % of the GPU, so E4B keeps about half
its rate while the 1.7B speaks (the proposed "at most 25 % slower" is out
of reach for it; the 0.6B leaves 60 %). Two frames a run gave E4B no more
(23.6–25.1 tokens/s) and a thinner lead. The chat runs at its own rate
again once the speech is made.

**A spoken question answered aloud.** The question ("What is the
tallest mountain in Africa, and how long does it take to climb it?")
spoken once by the server's voice, resampled to 16 kHz, streamed at real
time to the live route with 0.4 s of silence after it, then `finish`; the
final transcript to chat (a system prompt asking for three or four
spoken sentences); each sentence of the streamed answer to speech as
soon as it ends. Times after `finish`, three rounds:

| | 1.7B voice | 0.6B voice |
| --- | --- | --- |
| final transcript | 27 ms (19–29) | 25 ms (23–27) |
| first token | 287 ms (175–287) | 287 ms (167–297) |
| first sentence | 387 ms (285–388) | 388 ms (275–393) |
| first audio byte | 523 ms (418–524) | 482 ms (364–482) |
| E4B over the answer (4 sentences) | 20.0–20.2 tokens/s | 26.9–27.4 tokens/s |
| playback | 15.6 s, no underrun, lead min 40 ms | 22.5 s, no underrun, lead min 50 ms |

The transcript was the question word for word. Without the silence the
recognizer at lookahead 3 drops the last word ("it"), as `nuclis
transcribe` does on the same file: a clip that ends within 100 ms of its
last word leaves the encoder no right context, which a turn that ends in
silence always has. The first speech request after the server starts is
about 90 ms slower than the next (158 against 71 ms to the first byte).
### Speech beside a prefill, 2026-11-07

Apple M4 Pro (48 GB), ReleaseSafe, Zig 0.17.0, on the branch after
`8960c8a`, `scripts/bench/speech-beside-chat.py`: a fresh `nuclis serve
--pin <brain> --pin qwen3-tts-1.7b-customvoice --memory 40`; a chat with
nothing cached, and 0.3 s later one sentence of 11.0 s of audio (`sohee`,
seed 0) read as a stream; playback modelled from the first byte, a stall
any moment it has nothing. Before is one frame a run (4 runs of each
effort); after, the pace in § Scheduling (8 runs of each at 2,700 tokens,
4 at 4,500, `none` only).

| Prompt | Before: speech, stalls, silent, worst | After: speech, stalls | First token alone, before, after |
| --- | --- | --- | --- |
| E4B, 2,700 | 0.94–0.99×, 11, 5.1–5.5 s, 0.63 s | 0.91–0.93×, none in 16 runs | 6.2–6.5, 6.6–7.0, 10.7–13.9 s |
| 26B-A4B, 2,700 | 1.03–1.31×, 6–7, 5.8–7.9 s, 2.0 s | 0.79–0.93×, none in 14 of 16 (0.04 s, 0.37 s) | 6.9–7.4, 7.8–10.6, 12.0–14.1 s |
| E4B, 4,500 | 1.37–1.39×, 18, 9.7–9.8 s, 0.69 s | 0.91–0.97×, none in 3 of 4 (0.19 s) | 10.9–11.1, 11.7–13.6, 15.7–16.8 s |
| 26B-A4B, 4,500 | not measured | 0.81–0.93×, none in 4 | 12.8–13.0, not measured, 18.1–21.2 s |

Speech alone is unchanged (0.41×), as is speech beside a cached prompt
(0.41–0.42×, no stall), and a chat's generation beside speech (24.4–25.5
tokens/s, `voice-loop.py beside`). Each of the three stalls left is in a
run whose first token came late, by 0.6 s for the 0.04 s stall and a
second or more for the others: a step longer than the lead. What made
those steps long is not known.

The prefill pays with what speech needs, about 46 % of the GPU while it
speaks, whichever way it is sliced: about 5 s here, for 11 s of speech,
and less for a shorter sentence. Prefill chunks keep their size, since a
shorter one would lose batching and speech would need the same share.
