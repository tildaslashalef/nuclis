<!-- kind: reference -->
# Architecture: the paths beside the token loop

The decision model, the embedding model, and the HTTP API reach the engine
without going through `Engine`'s token loop, or wrap it. The layers they
sit in are [architecture.md § 2](architecture.md#2-the-repository-as-layers).

## The decision path

`nuclis decide` does not go through `Engine`. A decision model (Laya) is an
encoder: it reads one whole sequence in both directions and scores the
options at their `[MASK]` rows; nothing is decoded and nothing is carried
between sequences, so there is no session, no sampler, and no KV cache.
`inference.decide.Decider` composes three pieces from one checkpoint
directory: the Hugging Face tokenizer (`tokenizer/hf_json.zig`), the
profile (`profiles/laya.zig`: question validation, the sequence and its
budgets, calibration), and the model (`models/laya.zig` over
`models/modernbert*.zig`: F32 on the CPU through `backends/cpu/dense.zig`,
or `models/laya_metal.zig` on Metal, which packs the sequences of a call
into batches with per-row bounds instead of padding). Every question about
every state is one sequence. It pays as a filter, one
question over many states the language model never has to read
([laya.md](models/laya.md)).

The second family, clef-flash, shares the seam and not the shape: a
Qwen3.5 backbone (the qwen35 adapter at the file's shape, its
`hiddenRows`/`prefillHidden` giving every row after `output_norm` with no
output head), images through the Qwen3-VL projector, and a joint schema
head on the CPU (`models/clef.zig`) that reads one sequence per state
holding every question (`profiles/clef.zig`). `Decider` is a tagged union
of the two; the family decides how a question is validated
(`decide.parseQuestion`), so `src/decision/` parses a request only once
it knows the model, and a `Question` carries the order the model reads
its options in beside the order it is answered in
([clef-flash.md](models/clef-flash.md)).

## The embedding path

`nuclis embed` and `POST /v1/embeddings` do not go through `Engine`
either. `inference.embed.Embedder` opens an EmbeddingGemma 2 file and,
optionally, its mmproj. It turns an input's ordered parts into rows: text
tokens, and for each image or clip its begin marker, the projector's rows
(`vision/gemma4.zig` with Google's activation, `audio/gemma4a.zig` after
`audio/mel.zig`), and its end marker. Then one bidirectional pass of the
24-block encoder (`models/embeddinggemma*.zig`), a mean over every row,
and a unit vector. There is no cache and no session, and on Metal inputs
are packed back to back with per-row bounds, as Laya's are. The media
encoders are built on first use. Images and audio follow Google's
processor, not the chat path's reference pipeline, because the oracle is
Google's float32 vectors ([embeddinggemma.md](models/embeddinggemma.md),
[audio.md](engine/audio.md)). nuclis stops at the vector: indexing and
search belong to the application that calls it.

## The API layer

`nuclis serve` (`src/api/`) keeps decision models open behind an HTTP/1.1
server speaking TypeSafe's Jev protocol, and one language model behind
OpenAI's Chat Completions ([api.md](guide/api.md)). Its layers stay
apart, so each service is a directory and one registration:

- `http.zig` serves one connection over any reader and writer: each
  request read whole into the connection's arena (up to its route's body
  limit), handed to a handler, answered with `Content-Length` in one flush,
  or, when the response carries a `Stream`, with the head flushed at once
  and the body in chunks as it is produced; it knows no route.
  `router.zig` maps method and path to a service's handler; `errors.zig`
  is the one error body; `log.zig` writes the transport's line per
  response (the handler adds a note) under one lock.
- `gpu.zig` is the one executor that owns the GPU: a single worker runs
  submitted items one at a time in arrival order, so two models never run
  at once and only the worker touches a model. An item lives in its
  submitter's frame; a waiter that times out unlinks an item that never
  started and otherwise waits for it. A long item (a generation) calls
  `runShort` between its steps, so short ones (decision batches, speech
  chunks, transcription pieces) run there instead of after it; one that
  asks `betweenSteps` takes no more there than it needs (speech stays a
  second ahead of playback, a file transcription a quarter of the time).
  Each model has its own Metal queue and buffers. `memory.zig` is the one budget every open model counts
  against: before a model opens, the least recently used others close,
  of any kind; a model in use is pinned for the request, and one named by
  `--pin` for the server's life. `slot.zig` is the one-model holder of
  the kinds the server keeps one of (a voice, a recognizer). `pipe.zig` is the bounded byte
  queue a streamed response crosses from the worker to its connection, so
  the worker never writes to a socket.
- `decisions/` validates on the connection's thread, then waits in the
  batcher: one executor item takes the oldest waiting request's model and
  every request waiting for it that fits one pass, and answers them with
  `Decider.decideJobs` (each request prepared in its own arena, one
  `logitsBatch` for all). `pool.zig` keeps two deciders open, opened and
  closed only on the worker.
- `chat/` reads a request into the profile's shapes on the connection's
  task (`wire.zig`, pure), then waits for one executor item that opens
  the named model if another is open and completes the conversation
  through the agent's `Completer` (`model.zig`), so a served conversation
  continues or restores the model's state exactly as `nuclis agent` does.
  A whole response waits for the item; a streamed one drains its chunks
  from a `pipe.zig` queue. Either way the connection watches its client
  (`http.Peer`), and one that leaves sets the flag the item's observer
  turns into a cancellation. Clients resend the whole conversation; no
  response is stored.
- `speech/` queues requests in arrival order behind one short executor
  item that speaks the head one a chunk of frames per run, so speech
  advances between a generation's steps; the response's head waits for
  the prompt's prefill, then the audio drains from a `pipe.zig` queue
  (the item parks rather than block the worker when its client lags).
  `transcription/` keeps the recognizer's slot for the routes that use it:
  `file.zig` takes a whole recording (`multipart.zig` parses the form)
  and feeds it in 2.56 s pieces, one short item a piece; its segments,
  subtitles, and word times come from `src/audio/transcript.zig`, which
  `nuclis transcribe` shares.
- `src/decision/` holds the wire format and model names that `nuclis
  decide` and the service share, so the CLI's `--json` and the API's
  body are the same bytes.
- `websocket.zig` is RFC 6455's server side (the handshake, client
  frames read into bounded messages, server frames written); a response
  carrying an `http.Upgrade` makes the transport answer `101` and hand
  the connection to the route. `transcription/stream.zig` is the live
  route on it: per connection a reader task, the connection's task
  writing the events, and one short executor item that owns the
  connection's `Transcriber` and pushes one encoder chunk a run.
