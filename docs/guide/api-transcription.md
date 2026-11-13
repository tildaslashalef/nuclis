<!-- kind: guide -->
# Transcription

The transcription routes of [the nuclis API](api.md), which covers
running the server, its memory budget, its limits, and its errors: a
WebSocket for a live microphone, and OpenAI's call for a whole file.

## Live transcription

### `GET /v1/audio/transcriptions/stream`

A WebSocket ([RFC 6455](https://www.rfc-editor.org/rfc/rfc6455)) for a
voice client's microphone: the client streams audio as it is captured,
the server answers with the transcript as it forms, and with a final
transcript and word times when the client says the utterance ended. It
runs Nemotron 3.5 ASR streaming 0.6B, the model `nuclis transcribe`
runs ([nemotron-asr.md](../models/nemotron-asr.md)).

```sh
python3 scripts/gate/ws-transcribe.py clip.wav --language en-US   # a client in the standard library
```

The handshake is a `GET` with `upgrade: websocket`, `connection:
upgrade`, a `sec-websocket-key`, and `sec-websocket-version: 13`; the
server answers `101` and negotiates no subprotocol and no extension.
The session is set by the query:

| Parameter | In nuclis |
| --- | --- |
| `model` | a registry entry of kind `transcription` or `nemotron-asr-0.6b`; never a path. Absent: the open one, else `transcribe.model` |
| `language` | a locale of the model's prompt dictionary (`en-US`, `de-DE`, `fr`, `es-419`, …) or `auto` (the default: the model tags the language it heard), or a comma-separated list of up to four, `auto` only last ([several languages](#several-languages)) |
| `lookahead` | `0`, `3` (the default), `6`, or `13` rows of 80 ms the encoder waits for: a chunk, and so a possible partial, every 80, 320, 560, or 1,120 ms of audio |

A refusal before the upgrade is an ordinary HTTP response with the
shared error body: `426 upgrade_required` (with `upgrade: websocket`) to
a request that is not a WebSocket handshake, `426 unsupported_version`
(with `sec-websocket-version: 13`) to another version, `400
bad_handshake` (no `connection: upgrade`, a key that is not 16 bytes in
base64, not a `GET`), `400 invalid_request` (a malformed parameter,
`param` names it), `400 not_a_transcription_model`, `404
model_not_found`, `529 busy` (16 live transcriptions run already).

**The client sends** binary messages of 16 kHz mono 16-bit
little-endian samples, of any size (a sample may straddle two messages),
and text messages, each a JSON object:

| Message | Meaning |
| --- | --- |
| `{"type": "finish"}` | the utterance ended: the audio sent before it is transcribed to its end, a `final` follows, and the next audio begins a new utterance (the recognizer's state is reset) |
| `{"type": "reset"}` | drop the utterance in progress and every sample waiting to be transcribed, without a `final` |
| `{"type": "language", "language": "de-DE"}` | the language prompt from the next chunk on (`auto` the default's), or a list (`"ar,en,auto"`); an unknown locale is an `error` event and the language stays |
| `{"type": "phrases", "phrases": ["Siobhan", "Dun Laoghaire"]}` | names or words to expect, from the next chunk on, until the next `phrases` (`[]` clears them); send it first to have them from the start. At most 64, each at most 48 bytes; more is an `error` event and the list stays ([phrases](#phrases-to-expect)) |

A command applies after every sample sent before it. Pings are answered
with pongs; a close is answered once what was asked before it (a
`finish`) has been answered, within 5 s.

**The server sends** text messages, each a JSON object:

```json
{"type": "ready", "model": "nemotron-asr-0.6b", "lookahead": 3, "language": "auto", "sample_rate": 16000}
{"type": "partial", "text": "Kenneth and Beth refrain from telling", "audio_seconds": 2.813}
{"type": "final", "text": "Kenneth and Beth refrain from telling the other girls or Uncle John of Old Will Rogers' visit, but they got Mister Watson in the library and questioned him closely about the penalty for forging a check.",
 "language": "en-US", "words": [{"word": "Kenneth", "start": 0.800, "end": 1.200}, …], "audio_seconds": 11.700}
{"type": "error", "code": "invalid_message", "message": "a text message is a JSON object with a \"type\""}
```

- `ready` once the model is open and the session's recognizer exists
  (audio sent before it waits, within the backlog limit).
- `partial` when a chunk changed the utterance's text: the whole text so
  far, never a delta, so a stale one is simply replaced; `audio_seconds`
  is the audio of the utterance it covers. Chunks that add nothing send
  nothing.
- `final` after `finish`: the text (equal to `nuclis transcribe --json`'s
  for the same samples, lookahead, and language), the language tag the
  model emitted (`null` if none), the words, and the utterance's length.
  Words are grouped as a file's are ([segments](#segments)): a word
  starts at a piece that begins with a space; its `start` and `end` are
  the 80 ms encoder rows its pieces were emitted at (a late punctuation
  mark's row left out), from the utterance's start, which trail the
  speech by up to the lookahead. An
  utterance with no audio has an empty final.
- `error` with a stable `code` (below); the session continues unless a
  close follows.

**Limits**, fixed by the host:

| Limit | Value | Past it |
| --- | --- | --- |
| a message (its fragments together) | 1 MiB | close `1009` |
| a text message | 4 KiB | `error` `invalid_message` |
| audio received and waiting to be transcribed | 10 s | `error` `audio_backlog`, close `1008` |
| events queued for a client that does not read | 1 MiB | close `1008` |
| silence: no message, ping, or pong received | 60 s | close `1008` |
| live transcriptions at once | 16 (each holds 72 MB of encoder state on the device) | `529 busy` at the handshake (`1013` if two race for the last) |
| open connections | 64, shared with every route | `529 busy` |

**Scheduling.** Each connection is one short item on the GPU worker,
which pushes one encoder chunk a run (4 rows of 80 ms at lookahead 3,
about 20–35 ms on Metal) and is queued again while audio waits, so live
transcription runs between a chat generation's steps, and several
connections take turns a chunk at a time. Audio is buffered on the
connection until a chunk is complete, whatever the message sizes: the
transcript does not depend on how the client splits its audio, or how
fast it sends it. Opening the model is the exception: a connection
whose model is not open waits for a running generation to end, since
opening may close the language model it runs on. Every connection has
its own recognizer state over the one open model, which the memory
budget never closes while a connection uses it.

**Cancellation.** A client that disconnects, with or without a close
frame, stops its session at once: nothing more of its audio is
transcribed, and its recognizer is freed at the worker's next turn. On
Ctrl-C the server closes every session with `1001`.

**Errors and closes:**

| Event `code` / close | When |
| --- | --- |
| `invalid_message` | a text message that is not one of the commands, or over 4 KiB; the session continues |
| `invalid_language` | a `language` command naming no locale of the model (the session continues), or a `language` query that does not (close `1008`) |
| `audio_backlog`, close `1008` | more than 10 s of audio waits for the GPU |
| `model_failed` | the transcription failed on the device: the utterance is dropped and the session continues; if the model did not open, close `1011` |
| `model_too_large`, close `1011` | the model needs more than the budget leaves beside the pinned models |
| `busy`, close `1013` | the memory it needs is held by a running request, or another transcription model is in use; retry |
| close `1002` | a frame RFC 6455 forbids (unmasked, a reserved bit or opcode, a control frame fragmented or over 125 bytes) |
| close `1007` | a text message that is not UTF-8 |
| close `1001` | the server is stopping |

**For a voice assistant.** Stream the microphone continuously; send
`finish` when the turn ends (the client's own voice activity detection
decides), and read the `final` about 40 ms later; the next audio starts
the next utterance with no further message. Send `reset` to drop an
utterance (a barge-in the assistant ignores). The server transcribes
what it is sent: echo cancellation, so the assistant does not hear its
own voice, is the client's.

## Transcription (files)

### `POST /v1/audio/transcriptions`

OpenAI's transcription call for a whole recording, built for long ones (a
lecture, a meeting, an hour or more): the transcript as text, JSON with
segments and words, or subtitles, at once or streamed as segments are
final. The model is Nemotron 3.5 ASR, the one `nuclis transcribe` runs,
and the route's text is the command's, byte for byte (the gate
`nemotron-asr-file-route`).

```sh
curl -s localhost:9000/v1/audio/transcriptions \
  -F file=@meeting.m4a -F response_format=verbose_json \
  -F 'timestamp_granularities[]=segment' -F 'timestamp_granularities[]=word'
curl -s localhost:9000/v1/audio/transcriptions -F file=@lecture.wav -F response_format=srt > lecture.srt
```

The upload is held in memory while it is transcribed and is limited to
256 MiB (`serve.transcription_bytes`; past it `413
payload_too_large`): two hours of 16 kHz 16-bit mono WAV (230 MB), or
many hours of a compressed format. Files queue in arrival order, one
transcribed at a time; at most 16 wait (`529 busy`), and one that waits
`serve.timeout` without starting is `529 timeout`.

### The file transcription request

`multipart/form-data`, as OpenAI's SDKs send it. Parts are bounded (at
most 32, each part's headers 1 KiB, a field's value 1 KiB), and exactly
one is a file (`phrases` may be 8 KiB).

| Field | In nuclis |
| --- | --- |
| `file` | the recording: WAV, AIFF, MP3, M4A/AAC, FLAC, ALAC, or CAF (AudioToolbox, macOS), resampled to 16 kHz and mixed to mono; a 16 kHz mono WAV, 16-bit or float, is read directly on any platform |
| `model` | a registry entry of kind `transcription` or `nemotron-asr-0.6b`; never a path. Absent: the open one, else `transcribe.model`. OpenAI's names (`whisper-1`, `gpt-4o-transcribe`) are `404 model_not_found` |
| `language` | `auto` (the default), or a locale or two-letter code the model knows (`en-US`, `de-DE`, `fr`, …; 121 keys), or a comma-separated list of up to four, `auto` only last ([several languages](#several-languages)); an unknown one is `400 invalid_request` |
| `response_format` | `json` (the default), `text`, `verbose_json`, `srt`, `vtt` |
| `timestamp_granularities[]` | `segment`, `word`, or both (repeat the field); with `verbose_json` or `stream`. Absent: segments |
| `stream` | `true`: server-sent events (below); not with `srt` or `vtt` |
| `lookahead` | nuclis's: rows of 80 ms the encoder waits for, `0`, `3` (the default), `6`, `13`; as `nuclis transcribe --lookahead` |
| `phrases` | nuclis's: a JSON array of names or words to expect, `["Siobhan", "Dun Laoghaire"]` (at most 64, each at most 48 bytes, the field 8 KiB); as `nuclis transcribe --phrase` ([phrases](#phrases-to-expect)) |

Refused with `400 unsupported_feature`: a `prompt` (the model takes no
text; `phrases` and `language` are its hints) and `include[]` (the decoder is
greedy and keeps no log probabilities). Other fields OpenAI defines are
ignored (`temperature`: decoding is greedy).

### Phrases to expect

Nemotron ASR knows no names it was not trained on, and takes no text
prompt. A client that knows the names in its user's life lists them in
`phrases`; the decoder then leans toward them: a piece that starts one
gains 2 logits over the blank and the other pieces, a piece that continues
one 6 (`docs/models/nemotron-asr.md` § Phrase boosting). Write each as it
should be spelled. Without phrases nothing changes.

What phrases fix and cost: [record, 2026-10-28](../benchmarks/record/api.md#phrases-to-expect-2026-10-28).

### Several languages

A user who speaks more than one language lists them: `language:
"ar,en,auto"` on either route (`nuclis transcribe --language
ar,en-US,auto`). The audio is encoded once and decoded under each; each
stretch of words up to a pause takes the language that wrote it,
because a fixed locale writes nothing for most other languages. `auto`,
only last, is the fallback: it is used when none of the others wrote, or
when it wrote the same script better (Spanish that `en` spelled in
English). The response's `language` (a final's on the live route) names
the language chosen, from its tag or, without one, the locale. Partials
may change language until the pause; segments and finals do not.

What a list costs and gains: [record, 2026-10-30](../benchmarks/record/api.md#several-languages-2026-10-30).

### The file transcription response

- `json` (`application/json`): `{"text", "usage": {"type": "duration",
  "seconds"}}`.
- `text` (`text/plain`): the transcript and a newline.
- `verbose_json`: `{"task": "transcribe", "language", "duration",
  "text", "segments", "words", "usage"}`, `segments` and `words` as the
  granularities ask. `language` is the locale of the last language tag
  the model emitted (`en-US`, `de-DE`), or the request's, or with a list
  the one chosen, where OpenAI writes a name (`english`).
- `srt` (`text/plain`) and `vtt` (`text/vtt`): one cue per segment,
  `HH:MM:SS,mmm` and `HH:MM:SS.mmm`, numbered from 1 in SRT, under a
  `WEBVTT` header in VTT.

A segment is `{"id", "start", "end", "text", "language"}`, a word
`{"word", "start", "end"}`, times in seconds. The text is the segments'
texts joined by spaces.

With `stream=true` the response is `text/event-stream`, in OpenAI's
events, a segment at a time as each becomes final:

```
data: {"type": "transcript.text.delta", "delta": "The thrall's eyes flashed."}

data: {"type": "transcript.text.segment", "id": 0, "start": 3.6, "end": 4.96, "text": "The thrall's eyes flashed.", "language": "en-US"}

: progress 61.44 of 1802.25 s

data: {"type": "transcript.text.done", "text": "…", "usage": {"type": "duration", "seconds": 1802.25}}
```

Each final segment is a `transcript.text.delta` (its text, with a
leading space after the first: the deltas joined are the text) then a
`transcript.text.segment` (OpenAI's diarized event, with `language`
for `speaker`, and `words` when asked). A comment `: progress <done> of
<total> s` comes at most once a second, and `: transcribing` after 10 s
of silence. The head is sent once the model is open and the language
checked, so those refusals are still statuses; an error after it (the
file stops decoding) is one `data: {"error": {…}}` event before the end.
A client that closes the connection stops its transcription before the
next 2.56 s piece; a client that does not read keeps at most 64 KiB of
events queued, and the transcription waits for it rather than hold the
GPU.

### Segments

Segments come from the transcriber's tokens and their rows (80 ms each),
by one function `nuclis transcribe --segments` shares
(`src/audio/transcript.zig`). A new segment starts at a word that follows
terminal punctuation (`.`, `?`, `!`, `…`, `。`, `？`, `！`, then perhaps
closing quotes), a pause of 1.04 s or more since the last word's sound,
or 30 s into a segment. The model emits punctuation late, after the pause
it marks (up to 2.6 s after the word in the speech corpus), so times and
pauses follow the pieces holding a letter or digit, and a segment without
terminal punctuation stays open 4 s for a late comma or period. A
segment's language is the tag the model emitted at its end, else the
last before it, else the request's locale. Times are the rows that
emitted the tokens, not a forced alignment: a word's start is its first
piece's row, its end one row past its last sounding piece.

Why the pause is 1.04 s: [record, 2026-10-27](../benchmarks/record/api.md#segments-2026-10-27).

### Long recordings

The upload is decoded in pieces (`inference/src/audio/audio.zig`
`Reader`, AudioToolbox read 4,096 frames at a time) and fed to one
stream 2.56 s (32 rows) at a time, each piece a short GPU item: it runs
between a chat generation's steps and speech's chunks, never waiting for
a generation to end. Beside a generation a file takes at most a quarter
of the GPU's time (a piece runs only once three times its length has
passed since the last), since it has no deadline and the generation
waits for each piece; alone it runs back to back. Memory is the upload, the transcriber's state (on
Metal 72 MB of device buffers, the front end's pending samples), and the
segments, never the decoded samples: peak RSS rose 46 MiB from a
5-minute to a 30-minute recording whose upload is 45.7 MiB larger
([record](../benchmarks/record/api.md#file-transcription-2026-10-27-commit-c4fe892)). A 2-hour 16 kHz 16-bit WAV would hold its 230 MB plus
the same fixed part, about 100 MB above an idle server (an estimate from
those two, not measured). The tokens (8 bytes each) and the segments
grow with the length, a few hundred kilobytes for two hours.

Fed in 2.56 s pieces, every clip of the corpus gives the same tokens and
rows as pushed whole (60 of 60, Metal), so `nuclis transcribe` feeds
files the same way. On a 30-minute recording the piece size changes the
transcript by a few characters (20,647 to 20,651 for pieces of 8 to 64
rows): the encoder's batches round differently. The state carries from
one utterance into the next, which changes a few first words (2 % of the
words against each clip transcribed alone, [record](../benchmarks/record/api.md#file-transcription-2026-10-27-commit-c4fe892)). A recording
that **changes language** loses words after each switch: the 30-minute
recording of the corpus's five languages, under `auto`, scored 19.0 %
WER against 6.3 % for its clips alone, the first seconds after a switch
often missing (the CPU reference drops the same words from the first
45 s of it), and two segments ran across a join. One language per
recording is what it transcribes well.

