<!-- kind: guide -->
# Speech

The speech routes of [the nuclis API](api.md), which covers running the
server, its memory budget, its limits, and its errors.

## `POST /v1/audio/speech`

OpenAI's speech call for the speech models `nuclis speak` runs, Qwen3-TTS
in three kinds: CustomVoice (two sizes) speaks in nine built-in speakers,
Base (two sizes) in a saved voice cloned from a clip ([voices](#voices)),
and the 1.7B VoiceDesign in a voice its `instructions` describe;
Fish Audio's Drama 3 (`drama-3`, [below](#drama-3)), which clones a saved
voice at 44.1 kHz with its delivery cued in the text; and
VoiceTut-TTS (`voicetut-tts`, [below](#voicetut-tts)), an OmniVoice model
specialized in Egyptian Arabic and Arabic–English code-switching with
auto-voice, attribute-based voice design, and zero-shot voice cloning at
24 kHz. The audio streams as it is made: the first
80 ms leave after one frame, then chunks of one, two, and four frames
(320 ms), each no longer than the audio sent before it, so a listener
who plays from the first byte never waits between them.

```sh
curl -s localhost:9000/v1/audio/speech -o hello.wav -d '{
  "input": "Hello there. This is nuclis speaking through the API.",
  "voice": "ryan"
}'
```

The response is `200`, `transfer-encoding: chunked`, `content-type:
audio/wav`: a WAV header whose sizes are the largest a field holds (the
length is not known when it is sent; players read to the end of the
stream), then 16-bit mono samples at the model's rate (24 kHz; 44.1 kHz
for Drama 3, `sample_rate` in `GET /v1/models`). With `"response_format":
"pcm"` it is the samples alone (`audio/pcm`, little-endian, at that rate).

One speech model is open at a time beside the others; it opens on its
first request (0.35 s for the 1.7B with its files in the page cache) or
at start with `--model` or `--pin`. A request's text is spoken whole: send
a sentence or two at a time, as a client splitting a chat answer at
sentence ends does.

## The speech request

| Field | In nuclis |
| --- | --- |
| `input` | the text, UTF-8, at most 16 KiB (`400 request_too_large`) |
| `model` | a registry entry of kind `speech` or a catalogue speech name (`qwen3-tts-1.7b-customvoice`, `qwen3-tts-0.6b-customvoice`, `qwen3-tts-1.7b-base`, `qwen3-tts-0.6b-base`, `qwen3-tts-1.7b-voicedesign`, `drama-3`, `voicetut-tts`); never a path. The names before 0.11.0 (`qwen3-tts-1.7b`, `qwen3-tts-0.6b`) are `404 model_not_found`, the message naming the new one. Absent: the open one, else `speech.model` (`qwen3-tts-1.7b-customvoice`). Another speech model replaces the open one; that request pays the open |
| `voice` | on CustomVoice one of its speakers: `aiden`, `dylan`, `eric`, `ono_anna`, `ryan`, `serena`, `sohee`, `uncle_fu`, `vivian` (OpenAI's voice names are not speakers here: `400 invalid_request`, listing them). On a Base or Drama 3 a saved voice, by name or as OpenAI's custom voice `{"id": "<name>"}`: `404 voice_not_found` for one not saved, `400 invalid_request` for a speaker's name. On VoiceTut-TTS omitted/`null` uses auto-voice (or voice design when `instructions` is given), and a saved voice (`"<name>"` or `{"id": "<name>"}`) clones it with its caption. On VoiceDesign ignored (OpenAI's clients always send one). A custom voice on CustomVoice or VoiceDesign is `400 voice_cloning_unsupported`. Absent on non-OmniVoice models: `speech.voice` (`ryan`) |
| `instructions` | on the 1.7B CustomVoice how to say it (`"Speak with excited, breathless joy."`); on VoiceDesign the voice itself (`"A calm elderly man with a deep, gravelly voice."`), sampled anew each request; on VoiceTut-TTS comma-separated OmniVoice attribute tags (`"male, young adult, moderate pitch"`) when `voice` is omitted (`400 invalid_request` for unknown/conflicting tags or when combined with a saved `voice`). On the 0.6B CustomVoice, a Base, and Drama 3 (whose cues are in `input`), `400 instructions_unsupported` |
| `response_format` | `wav` (the default; OpenAI's is `mp3`) or `pcm` |
| `stream_format` | `audio` (the default): the bytes as they are made. `sse`: server-sent events, each `{"type": "speech.audio.delta", "audio": "<base64>"}` carrying the next bytes of the same stream (the WAV header in the first, for `wav`), then `{"type": "speech.audio.done", "usage": {"input_tokens", "output_tokens", "total_tokens"}, "finish_reason"}` (the prompt's rows in, the frames out; `eos` when the model ended the speech, `length` when [the frame budget](../models/qwen3-tts.md#the-frame-budget) stopped it, which a client may treat as a failed line) |
| `speed` | positive speaking rate multiplier on `voicetut-tts` (default `1.0`); `1` only (`400 unsupported_feature` otherwise) on models without rate control |
| `seed` | nuclis's: the sampler's seed, default 0. The same text, voice, and seed give the same samples, alone or beside a chat generation |
| `language` | nuclis's: `auto` (the default; on `voicetut-tts` defaults to `arz` for Egyptian Arabic), or one of the model's (`english`, `chinese`, `german`, …; `400 invalid_request` lists them). Drama 3 takes only `auto` |
| `context` | nuclis's, Drama 3 only: an earlier response's `x-speech-context`, whose last sentence this request continues ([below](#drama-3)); ignored when unknown or expired |

Other fields OpenAI defines are ignored. Refused with `400
unsupported_feature`: `mp3`, `opus`, `aac`, and `flac` (no encoder here),
a `speed` other than 1.
`model` naming another kind is `400 not_a_speech_model`; a name nothing
pulled answers is `404 model_not_found`. A refusal on the GPU worker (an
unknown speaker or language) is still a status: the response's head
waits until the request's prompt is prefilled, which is before its first
audio anyway. A model that fails to open is `500 model_failed`; one too
large for the budget beside the pinned models `400 model_too_large`.

## Voices

A Base speaks in a voice saved from a clip of 1–30 s: its timbre from the
clip alone, or, with the clip's exact words, the closest likeness (Qwen's
in-context cloning). A voice is saved once and named; the first speech
request in it on a Base encodes it on the CPU (under a second; the
connection waits, the GPU does not) and keeps the result for later
requests. A clip-alone voice then starts as fast as a speaker's; with a
transcript the clip's frames are part of every prompt, so the first audio
grows with the clip (on the 0.6B about 260 ms for 4.6 s, 430 ms for 10 s,
650 ms for 15 s; [qwen3-tts.md § Speed](../benchmarks/record/qwen3-tts.md#speed-2026-10-29)).
Either Base size speaks any saved voice. The routes follow OpenAI's custom voices; nuclis has no consent
flow, so clone only a voice you have the right to use.

```sh
curl -s localhost:9000/v1/audio/voices -F name=anna -F audio_sample=@anna.m4a \
  -F 'reference_text=Hello, it is Anna. I will be reading the news today.'
curl -s localhost:9000/v1/audio/speech -o hi.wav -d '{
  "model": "qwen3-tts-1.7b-base", "input": "Good morning.", "voice": {"id": "anna"}
}'
```

| Route | Does |
| --- | --- |
| `POST /v1/audio/voices` | saves a voice, `201` with its object. Multipart: `audio_sample` (the clip: WAV, AIFF, MP3, M4A, FLAC, CAF; the body at most 32 MiB), `name` (`[a-z0-9][a-z0-9_-]{0,63}`, not one of the speakers), and nuclis's `reference_text` (what the clip says, at most 2 KiB; trimmed, empty is none). OpenAI's `consent` is ignored. `409 voice_exists`, `400 too_many_voices` past 256, `400 invalid_request` with the field for the rest |
| `GET /v1/audio/voices` | `{"object": "list", "data": [voice…]}`, by name |
| `GET /v1/audio/voices/{name}` | one voice, or `404 voice_not_found` |
| `DELETE /v1/audio/voices/{name}` | `{"id", "object": "audio.voice.deleted", "deleted": true}`; the clip and all made of it go |

A voice object is OpenAI's `{"id", "object": "audio.voice", "name",
"created_at"}` (`id` is the name; `created_at` Unix seconds) with
nuclis's `reference_text` (or null) and `seconds`. Voices live under the
server's user root, `~/.nuclis/voices/<name>/`, shared with `nuclis voice`.

For a voice that stays the same, VoiceDesign's output can be saved: speak
a line or two in a described voice, then save that audio with its text as
a voice and speak in it on a Base (Qwen's *voice design then clone*).

## Drama 3

`drama-3` (Fish Audio's Drama 3, 35B-A3B, one 20.8 GB Q4_K GGUF;
[models/drama3.md](../models/drama3.md)) speaks in a saved voice with its
transcript: a voice saved without `reference_text` is `400
invalid_request` (param `voice`), so save Drama 3's voices with their
words. Any saved voice serves it and the Bases alike; a voice
saved since Drama 3 also keeps its clip at 44.1 kHz, which Drama 3
encodes (an older one is resampled from 24 kHz, losing the top of the
band). Its first use encodes the clip on the CPU with Fish's codec.

The delivery is cued in `input` itself:

- free-form cues in brackets before the words they shape: `[whispering]
  Don't wake him.`, `[laughs] No way.`, `[sighs]`;
- paired tags around words: `<whisper>…</whisper>`, `<emphasis>`,
  `<soft>`, `<fast>`, `<slow>`, `<stress>`, `<forceful>`;
- speakers `<|speaker:0|>` to `<|speaker:15|>` before their lines;
  untagged text is speaker 0, the saved voice.

Refused with `400 invalid_request` (param `input`, the message naming
the fault): a bracket left open or a cue inside a cue, an unclosed tag,
a speaker tag outside 0–15, and any other `<|…|>` token. Line breaks are
spaces. A request's text is split at sentence ends (Arabic's `؟` and `؛`
included) and spoken sentence by sentence, each in the voice of the one
before it. A sentence that has not started speaking within 1.5 s of audio
(steady noise does not count as speaking) is drawn again with the next
seed, twice at most. A sentence that falls silent for 1.5 s ends there,
keeping 0.28 s of the silence: [drama3.md § Stalls, and what
nuclis adds](../models/drama3.md#stalls-and-what-nuclis-adds).

**One sentence a request.** Every Drama 3 response carries an
`x-speech-context` header, an id for its last sentence. A later request
in the same model and voice that passes it as `"context": "<id>"` takes
that sentence (its text and the audio codes it was spoken with) as
history. So it continues the earlier delivery, as a request's own
sentences do. The sentence is kept for 5 minutes once its request ends,
at most 32 of them. An id that is unknown, expired, or from another voice
is ignored: the request is spoken without it, not refused, and the log's
`context` says which (`hit`, `missing`, `expired`, `other_voice`). A
request sent while the one it continues is still speaking waits behind
it in the queue, so the sentence is kept by the time it starts.

```sh
curl -s -D head.txt -o one.pcm localhost:9000/v1/audio/speech -d '{"model": "drama-3",
  "voice": {"id": "anna"}, "response_format": "pcm", "input": "[tired] يا عم ده إحنا بقالنا ساعة مستنيين."}'
curl -s -o two.pcm localhost:9000/v1/audio/speech -d '{"model": "drama-3", "voice": {"id": "anna"},
  "response_format": "pcm", "input": "والزحمة دي مش طبيعية.", "context": "'"$(grep -i x-speech-context head.txt | cut -d" " -f2 | tr -d "\r")"'"}'
```

Every prompt starts with the same rows for a voice: the system turn and
the voice's clip with its transcript (328 rows for a 12 s clip). The model
keeps the state after them for its 4 most recent voices. A request then
prefills only its own text (and a context's sentence); the first in a
voice since the model opened prefills both. On the M4 Pro with a 12 s
clip, a one-sentence request's prefill went from 1,612 ms to 243 ms
(first audio 393 ms), or 700 ms with a context (first audio 854 ms). The
take is the same bit for bit either way:
[drama3.md § The prompt's prefix](../models/drama3.md#the-prompts-prefix).

## VoiceTut-TTS

`voicetut-tts` (OmniVoice fine-tuned for Egyptian Arabic and Arabic–English
code-switching, one 1.35 GB `Q8_0` GGUF; [models/omnivoice.md](../models/omnivoice.md))
outputs 24 kHz mono audio and runs in three voice modes from the same file:

- **Auto-voice**: omit `voice` (or set it to `null`) and omit `instructions`.
  Speaks immediately with no saved voice; the voice timbre is determined by
  `seed`.
- **Voice design**: omit `voice` and set `instructions` to comma-separated
  OmniVoice attribute tags, at most one per category: gender (`male`, `female`),
  age (`child`, `teenager`, `young adult`, `middle-aged`, `elderly`), pitch
  (`very low pitch`, `low pitch`, `moderate pitch`, `high pitch`,
  `very high pitch`), style (`whisper`), and English accent / Chinese dialect.
  Unknown tags, conflicting categories, or combining `instructions` with a
  saved `voice` return `400 invalid_request` (param `instructions`).
- **Zero-shot voice cloning**: pass `"voice": "<name>"` or
  `"voice": {"id": "<name>"}` for a voice saved via `nuclis voice add` or
  `POST /v1/audio/voices` with its transcript (`reference_text`). A voice saved
  without `reference_text` is `400 invalid_request` (param `voice`). On first
  use the 24 kHz clip is encoded on the connection task with the Higgs Audio V2
  tokenizer and cached in `voice.json`.

`language` defaults to `arz` (Egyptian Arabic) when omitted or `auto`; `input`
is passed verbatim with no implicit text normalizer. `speed` scales the
estimated target duration (`> 0`, default `1.0`).

```sh
# Auto-voice (Egyptian Arabic + English code-switching)
curl -s localhost:9000/v1/audio/speech -o cairo.wav -d '{
  "model": "voicetut-tts",
  "input": "عندي meeting مهم بكرة الصبح في المكتب."
}'

# Voice design
curl -s localhost:9000/v1/audio/speech -o design.wav -d '{
  "model": "voicetut-tts",
  "input": "أهلاً بيك في القاهرة.",
  "instructions": "male, young adult, moderate pitch"
}'
```

## Streaming and cancellation

The head is sent once the prompt is prefilled; the samples follow chunk
by chunk. An event stream silent for 10 s gets an SSE comment (`:
speaking`). An error after the head (the device fails mid-utterance) is
one `data: {"error": {…}}` event in an event stream; a raw stream ends
without its last chunk, which an HTTP client reports as a truncated
body.

A client that closes the connection stops its request between chunks:
the server notices within a second, frees the synthesizer, and the next
request starts. A request still waiting is taken out of the queue; one
that waits `serve.timeout` without starting is `529 timeout`. A
1.5 s client timeout on a long sentence stopped it after 2.96 s of
audio had been made, and `GET /v1/health` showed nothing speaking a
second later.

## Scheduling

Speech runs on the one GPU worker as short items: each run makes one
chunk, so a chunk runs between the steps of a chat generation (and of
its hosted tool waits) and speech never waits for a generation to end.
Between a generation's steps speech takes only what playback needs: a
run lasts at most as long as the step before it, at least one frame
(about 37 ms on the 1.7B), and stops once the audio sent is ahead of its
playback, counted from the request's first byte at real time, by a
second or by 1.5 steps, whichever is more. After a generated token
that is one frame; after a prefill chunk, hundreds of milliseconds of
frames in chunks of four, so the voice outlasts the next chunk. So
beside a generation a request is made at playback speed with a second
in hand, and the generation keeps the rest of the GPU (on the 1.7B about
half its rate, on the 0.6B 60 %: [record, the voice loop](../benchmarks/record/api.md#the-voice-loop-2026-10-27-commit-f96f750); a prefill
beside it the same, [record, speech beside a prefill](../benchmarks/record/api.md#speech-beside-a-prefill-2026-11-07)); alone, with nothing
waiting, it runs at full speed in chunks of four. Opening a voice is the exception: a request whose voice
is not open waits for the running generation to end, since opening may
close the language model it runs on.

A voice speaks one request at a time (its talker cache and the codec's
scratch are one synthesizer's), so concurrent speech requests queue in
arrival order, at most 64 waiting (`529 busy` past them); each one's head
is sent when its turn begins. Three sent together answered at 0.14,
1.57, and 2.83 s, each when the one before it ended. A client that does
not read keeps at most about 5 s of audio queued on the server; the
request then waits for it rather than hold the GPU.

The speech models are listed by `GET /v1/models` (`kind` `speech`, with
`voices`, `instructions`, `sample_rate` (24000; 44100 for `drama-3`),
and `formats`), the
transcription models too (`kind` `transcription`, `sample_rate` 16000), each
with `present`, `loaded`, `default`, `size_bytes` (weights and codec),
`repo`, and `revision`. `GET /v1/health` gives the open `speech` and
`transcription` models and `speech_requests`: `waiting`, whether one is
`speaking`, and those `answered` since the start.

