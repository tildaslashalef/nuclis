# TODO — active plan

This file is the only progress tracker. It holds **unfinished work only**:
where we are, what is next, and the design of each remaining unit. When a
unit closes, its outcome moves to
[docs/engineering-log.md](docs/engineering-log.md) and
its section is deleted here; when the last unit closes, this file is emptied
back to this header. Requirements live in [docs/spec.md](docs/spec.md); the
engine map in [docs/architecture.md](docs/architecture.md); how to build,
test, and measure in [docs/development.md](docs/development.md).

Session protocol (also in [AGENTS.md](AGENTS.md)): read this file first. If
it lists work, summarize *Where we are* and ask the user how to continue. If
it is empty, ask what to work on and write the agreed plan here.


## Where we are

**Next: APPS-19 session 1** (its section follows this one): `nuclis
serve`, the nuclis API layer in `src/api/`, decisions first. Record its
`Base:` at the first change. The order of the units that follow is the
table at the end of this section; the history of the speed theme comes
first.

Theme agreed 2026-09-30: **decode speed on Metal**, Qwen3.8-27B first,
toward the 20 tokens/s of [ADR 0001](docs/adr/0001-qwen-decode-verifier.md)
(proposed). ENGN-18 closed 2026-09-30: saved prefixes, `bench
--verify-rows`, and `make speed` / `make speed-base` exist, and the
opening baseline is in
[bench.md § The decode-speed baseline](docs/reference/bench.md#the-decode-speed-baseline-engn-18-2026-09-30).
The base binary for `make speed` lives in `.zig-cache/speed/base/` (not
committed; `make speed-base` after each kept change). KERN-20 closed 2026-09-30: captures read in Xcode,
`bench --kernel-stats`, and
[apple-gpu.md](docs/reference/apple-gpu.md) with four kernel readings; KERN-05
and KERN-12 answered. KERN-21 closed 2026-09-30: verify batches run the flash-decoding split
pass (Qwen's 4-row C 387 → 288 ms at 4K, 1,157 → 376 at 32K; Gemma 12B
908 → 200 at 32K), checked at depth by `qwen38-verify-depth-*`. KERN-24 closed 2026-10-01
below its target: the register-fragment tile and two routings cut Qwen's
4-row verify C at 4K from 289 to 244 ms (512: 275 → 230) and Gemma 12B
QAT's by 23 %. ENGN-19 closed 2026-10-01: verify batches of up to 8 rows
step DeltaNet without writing the state and recovery replays a tape, so
Qwen's 4-row C fell 48–51 ms at every depth (4K 239 → 191 ms, 32K 329 →
279) and the 1.25 GB row-slot region is gone. ENGN-20 closed 2026-10-02:
speculation re-priced on every family; the catalogue turns it on for Qwen
(draft 7: prose 1.25–1.57×, short code 20.3–20.9 tok/s), Gemma 12B QAT
(5), E4B (6), and Muse (6), and off for Gemma 26B-A4B; `make
spec-matrix` measures it, and `agent --print` now loads the drafter. An
existing `~/.nuclis/nuclis.json` keeps its entries' old values until the
user edits them or re-runs `config init`. The base binary for `make
speed` is at ENGN-19's commit (no engine arithmetic changed since).
REPO-27 closed 2026-10-02: an external review's fixes (a failed
recording is discarded, so metal-check reports instead of hanging; the
kernel table is derived from `Kernel`; `--file` names a support file's
weights; parser property tests). Its log entry lists what waits for
KERN-23: pruning the 56 check-only pipelines (1.70 s of a cold start)
and the GGUF type-id enum. No engine arithmetic changed, so the `make
speed` base binary stands.

**Re-ordered 2026-10-02 (user).** Decision models move ahead of the last
speed unit: first APPS-19, `nuclis serve`, the nuclis API (`src/api/`)
with decisions as its first service, models kept loaded and concurrent
requests batched, an OpenAI-compatible service expected later; then MODL-34,
Cloudflare's clef-flash decision model; then KERN-23, which closes the
decode-speed theme. Dropped (engineering log): KERN-22, long-context
decode attention, a small win at 32K only; AGNT-18, the agent's `decide`
tool, since decision models are served by `nuclis serve` and the agent
stays a tool for language models.

| # | Unit | Sessions |
| --- | --- | ---: |
| 1 | APPS-19 — `nuclis serve`: the nuclis API, decisions first, batched across requests | 2 |
| 2 | MODL-34 — clef-flash: Cloudflare's 9B decision model, text then vision | 4 |
| 3 | KERN-23 — Weight streaming for one row and a few (closes the decode-speed theme) | 2 |
| — | AGNT-19 — Saved prefixes for the agent across processes | queued after KERN-23 |

## APPS-19 — `nuclis serve`: the nuclis API, decisions first, batched across requests (2 sessions) — next

Base: recorded when the unit starts.

Every `nuclis decide` call starts a process and opens its checkpoint:
about 75 ms for `laya` and 0.4 s for `laya-multilingual` (parsing its
34 MB `tokenizer.json`) before a 230–300 ms Metal encode, so a client
deciding every second spends half its time loading. A long-running
process that keeps decision models open removes that; batching requests
that arrive together into one Metal pass raises throughput the way a
multi-state call already does (one state 2,440 tokens/s, 50 states 3,500,
[laya.md § Time per call](docs/reference/laya.md#time-per-call)); an
endpoint listing the models lets clients follow the catalogue.

Where the time goes decides the design: the encode is hundreds of
milliseconds and HTTP on loopback is tens of microseconds, so the server
must add nothing measurable around the encode, never leave the GPU idle
while a request waits, and pack whatever is waiting into the next pass.

### Structure: an API layer, decisions its first service

`nuclis serve` is the nuclis API, not a decision server: decisions are its
first service, and an OpenAI-compatible service for the language models
(`/v1/chat/completions`, streamed) is expected later. This unit builds only
decisions, but the layers below keep that addition to a new service
directory and one registration line.

```
src/api/
  root.zig        serve(gpa, io, options): listener, router, services; the
                  `nuclis serve` command's entry, called from cli.zig
  http.zig        transport only: accept loop, connections, keep-alive,
                  per-connection arena, limits, Request/Response; knows no
                  route and no model
  router.zig      method + path → handler; the /v1 prefix; JSON error
                  bodies; each service registers its routes
  errors.zig      ApiError (code, HTTP status, message) and its JSON writer,
                  shared by every service
  models.zig      GET /v1/models, aggregated from every service's listing,
                  OpenAI's list shape with a `nuclis` object per entry
  gpu.zig         the one executor that owns the GPU: services submit work,
                  it runs one item at a time (two models never run at once,
                  laya.md § Limits), FIFO across services
  decisions/
    service.zig   registers the decision routes; the handlers
    pool.zig      open Deciders keyed by resolved directory, LRU, at most 2
    batcher.zig   session 2: merges waiting jobs into one GPU item
src/decision/     the decision wire format, shared by the CLI and the API
  request.zig     the Jev-shaped request JSON → Request (from today's
                  parseJson, stateFromJson, questionsFromJson, buildRequest);
                  `{"file": path}` states only when the caller allows files
                  (the CLI does, the API does not)
  response.zig    Answer/Result → the --json body (from writeAnswer and
                  run's JSON writer), so CLI and API write the same bytes
  catalog.zig     resolve and list decision entries with their budgets
                  (rl_agent_config.json max_len, head_max_len)
src/decide.zig    the `nuclis decide` command only: flags, terminal view
```

Rules that keep it decoupled: `http.zig` and `router.zig` import nothing
from `inference`; a service never touches sockets; `src/decision/` knows
no HTTP; the GPU is reached only through `gpu.zig`. A later chat service
adds `src/api/chat/`, registers its routes, contributes generation entries
to `/v1/models`, and submits its steps to the same executor; streaming
responses (server-sent events) are a `http.zig` addition then, not now.

### Surface

- **Command.** `nuclis serve` (APPS surface): `--host` (default
  `127.0.0.1`), `--port` (default 8735), `--model <name|path>`
  (repeatable, opened at start; otherwise each opens on first use),
  `--backend cpu|metal`. Binding anywhere but loopback prints a warning;
  there is no authentication.
- **Routes** (JSON in and out; errors `{"error": {"code", "message"}}`
  with the HTTP status that fits):
  - `POST /v1/decisions`: the body `nuclis decide --request` reads
    (`questions`, `state` or `states`, an optional `model` naming a
    decision model, default `decide.model`), `?explain=1` for the explain
    fields. The response is byte-for-byte what `nuclis decide --json`
    writes for the same request, timings aside. File states are refused.
  - `POST /v1/systemone`: Jev's single-state call for clients written
    against TypeSafe's API: one `state`, the response `{model, answers,
    usage}`.
  - `GET /v1/models`: OpenAI's shape, `{"object": "list", "data": [{"id",
    "object": "model", "owned_by", "nuclis": {"kind": "decision",
    "present", "loaded", "max_len", "head_max_len", "repo",
    "revision"}}]}`, so OpenAI clients read the ids and nuclis clients
    the rest. Decision entries only in this unit.
  - `GET /v1/health`: version, backend, loaded models, queue depth.

### Session 1: the layers, lean

- **The move first.** `src/decision/` takes the wire format out of
  `src/decide.zig` with no behaviour change: `nuclis decide --json` gives
  the same bytes for the 8 root fixture requests before and after, checked
  by hand and by the existing unit tests.
- **I/O.** `std.Io` (Zig 0.16) net listener and `std.http.Server`: one
  accept loop, each connection its own task (`io.async`), HTTP/1.1
  keep-alive, no pipelining. Each connection owns a fixed read buffer
  (16 KiB headers) and an arena reset after every response, so a warm
  connection allocates nothing that outlives a request. Bodies are read
  into the arena up to the limit, parsed once, rendered once; the
  response goes out with `Content-Length` in one write.
- **Pool and executor.** `decisions/pool.zig` opens each model once under
  a lock however many first requests race; `gpu.zig` runs one job at a
  time in this session.
- **Limits.** Host constants: body 4 MiB, headers 16 KiB, 64
  connections, 64 queued jobs (then 503 `busy`), 30 s per request
  (`timeout`); `inference.decide`'s request limits (`max_states`,
  `max_questions`, `max_options`, `max_state_bytes`) apply unchanged.
- **Targets** (Apple M4 Pro, ReleaseFast, `ab -k`, 1,000 requests):
  `GET /v1/health` p50 ≤ 0.2 ms and p99 ≤ 1 ms at concurrency 1, ≥
  20,000 requests/s at concurrency 16; a warm `laya-multilingual`
  `POST /v1/decisions` (one state, two questions) within 2 ms of the
  encode `timings_ms` reports, against 0.75 s through the subprocess.
- **Correctness.** Unit tests per layer: `http.zig` over in-memory
  streams (keep-alive, oversized headers and bodies, malformed requests)
  with no route; `router.zig` with stub handlers (404, 405, the error
  body); the decision service on the tiny synthetic checkpoint on the CPU
  (every error: malformed JSON, unknown model, a file state, busy,
  timeout; the pool's eviction; responses equal to what `nuclis decide
  --json` writes); all under `std.testing.allocator` with no leaks. Once,
  at close, the fresh binary serves the 8 root Laya fixture requests and
  the responses are compared with `nuclis decide --json` by hand; the log
  records it. No gate is added.

### Session 2: batching across requests

- **A jobs call.** `Decider.decide` asks every question of every state, so
  requests with different questions cannot share one call.
  `inference/src/decide.zig` gains `decideJobs`: a list of jobs, each its
  own states and questions, every sequence of every job built and sent to
  the model in one `Laya.logitsBatch`, each job's results returned apart.
  `decide` becomes the one-job case of it.
- **Batcher.** While the executor runs, arriving decision jobs queue per
  model in `decisions/batcher.zig`. When the GPU frees, the batcher takes
  every queued job for the oldest job's model whose sequences fit the
  Metal plan's 2,048 rows (in arrival order, a job never split) and
  submits them as one executor item. No waiting window: an idle GPU starts
  the first job at once, so a lone client pays nothing for batching. The
  CPU backend batches the same way.
- **Why it is safe.** A packed sequence's logits do not depend on what is
  packed beside it (bit-identical,
  [laya.md § On Metal](docs/reference/laya.md#on-metal)), so batching
  changes timing, never answers. The test asserts it: 16 concurrent
  requests with different questions return exactly the results each gets
  alone.
- **Fairness.** Jobs for another model wait at most one batch: the next
  batch serves the oldest waiting job's model.
- **Targets** (same machine, `ab -k -c 16`, a one-state two-question
  `laya` request, 512 requests): throughput ≥ 2× session 1's at
  concurrency 16, p99 latency ≤ 2 batches' time; concurrency 1 unchanged
  within 2 %.
- **Measured and recorded**: requests/s and p50/p99 at concurrency 1, 4,
  16 for both checkpoints, before and after batching, in
  `docs/reference/api.md`.

Docs: **`docs/reference/api.md` (new), the nuclis API reference clients
build against**: the server and its limits, then one section per service
(decisions now, chat later): every route with its request and response
JSON, field by field, the error codes with their HTTP statuses, the
batching behaviour, an example per route with `curl`, and the measured
rates. It is written for a client author who has not read the code, and it
is the document handed to client projects when the unit closes. Also
`docs/architecture.md` (the API layer and its rules, beside the decision
path), `docs/spec.md` (the command table), `docs/reference/laya.md` (a
link from its `nuclis decide` section), `docs/development.md` (the port,
measuring with `ab`), `src/help.zig`.

**Fast loop, no model gates.** `make verify-auto` selects only `fmt` and
`unit` for this unit: no gate lists `src/api/**`, `src/decision/**`,
`src/decide.zig`, `src/cli.zig`, `src/help.zig`, or
`inference/src/decide.zig`. Keep it so: do not edit the root `build.zig`
(it selects 51 gates; the new tests run in the existing `make test`), and
add no gate. If batching must touch `inference/src/models/laya*.zig` or
`profiles/laya.zig`, the six Laya gates (about 22 s together) run, as
they should. No numerical behaviour changes, so no Metal or CPU tier.

Gates: `make check`, `make verify-auto`.

## MODL-34 — clef-flash: Cloudflare's 9B decision model, text then vision (4 sessions) — after APPS-19

Base: recorded when the unit starts.

[Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash)
(Apache-2.0) answers the same typed questions as Laya (`noul`, `choice`,
`score`; one logit per option, softmax per question) about a text or JSON
state, optionally with images or video. It is Qwen3.5-9B with its vision
encoder plus a **joint schema head**, a small transformer over the
backbone's final hidden states that routes evidence to each question and
scores every option of every question in one pass (Laya runs one pass per
question). States up to 16,384 tokens. The card reports Decision Index
0.2.1 results and a 38.8 ms median latency on an H200 (torch 2.11,
transformers 5.10.2).

Known facts (2026-10-02):

- **Backbone** (`config.json`): `Qwen3_5ForConditionalGeneration`, text
  `qwen3_5_text`, 32 layers, hidden 4,096, `linear_attention` layers with
  `full_attention` every 4th, 16 heads, 4 KV heads, vocabulary 248,320:
  the hybrid DeltaNet family `models/qwen35.zig` already runs for
  Qwen3.8. BF16 in four shards (4.94 + 4.99 + 4.96 + 3.93 GB).
- **Head**: `joint_head.safetensors` (0.244 GB, BF16),
  `joint_head_config.json` `{hidden_size 4096, width 1024, routing_layers
  2, layers 4, heads 16, feedforward 4096}`; its forward and the input
  construction are in `joint_schema_model.py`.
- **Weights to run: the GGUF backbone plus the original head.**
  [bartowski/Cloudflare_clef-flash-GGUF](https://huggingface.co/bartowski/Cloudflare_clef-flash-GGUF)
  (llama.cpp b11279, imatrix) carries the backbone and the projector
  (`mmproj-…-bf16.gguf`, 0.92 GB), not the head. Q6_K (7.79 GB) first: the
  batched matmul has a specialized Q6_K path (`nu_matmul_q6_k`), and a
  decision is one prefill, compute-bound, so fewer bits buy little speed.
  Q8_0 (9.55 GB) if Q6_K misses the agreement bound; bf16 (17.92 GB) as
  the reference. The head stays BF16, decoded to F32.
- **Rejected: MLX 4-bit** (`mlx-community/clef-flash-4bit`, affine,
  group 64): its own card reports 96.4 % agreement with bf16 and a mean
  probability difference of 0.040, too many flipped decisions for a
  decision model, and nuclis has no kernels for the format.
- **Estimate, not a measurement**: about 20× Laya's compute; from Qwen
  27B's 90 tokens/s prefill, roughly 1–3 s per 500–1,000-token decision
  on the M4 Pro.

### Session 1: facts and the oracle

Read `joint_schema_model.py`, `chat_template.jinja`,
`processor_config.json`, and `tokenizer_config.json`: how a request
becomes a sequence (the template, where questions, options, and the state
sit, which positions the head reads), the head's forward exactly, the
calibration, the budgets (`max_state_tokens`), and how images enter.
Check that `nuclis inspect` reads the Q6_K and bf16 GGUFs as `qwen35`.
Write `scripts/clef-reference.py` (a venv beside `.reference/laya-venv`,
the package pinned) emitting fixtures under
`inference/src/models/fixtures/clef/`: for 8 requests (the Laya shapes
plus a 4,000-token state and a multi-question request) the ids, the head's
input positions, hidden states at a few backbone layers and at the head's
input, and the logits, all from the bf16 model. End the session by
rewriting this section at the level of files, functions, and numbers.

### Session 2: text on the CPU

A decision-family seam in `inference/src/decide.zig` (today it composes
Laya only): a family's profile builds sequences and calibrates, its model
returns per-option logits; Laya moves behind it unchanged (its gates
pass). `profiles/clef.zig` (the sequence and budgets), `models/clef.zig`
(the head on `backends/cpu/dense.zig`), and the backbone's final hidden
states from `qwen35_runtime.zig` without the output head. Checked
against the oracle with Laya's relative bounds (`zig build test-clef`,
gate `clef-cpu`).

### Session 3: Metal, the agreement, the catalogue

The backbone through `qwen35_metal.zig`'s prefill, the head as a Metal
plan beside Laya's. The catalogue entry `clef-flash` (kind `decision`;
the GGUF backbone, the head and support files from Cloudflare's repo,
pinned by revision); `nuclis decide --model clef-flash`; `nuclis serve`
lists and serves it. **Agreement**: Q6_K against the bf16 reference on
the fixtures plus 200 seeded requests: decision agreement and mean
|Δp|; Q6_K stays if agreement ≥ 99.5 %, else Q8_0. Timings for 1, 10,
and 50 states at 500 and 2,000 tokens, recorded in a new
`docs/reference/clef.md`. Gates `clef-metal`, `clef-agreement`.

### Session 4: vision

The projector through `inference/src/vision/` (the Qwen3-VL adapter;
check `clip.projector_type` and the projection width, 4,096 here against
Qwen3.8's 5,120), images in the decision request (the field the reference
defines, mirrored in `nuclis decide --request` and `nuclis serve`),
checked against the oracle on image fixtures; tokens and time per image
recorded.

Gates: `make check`, `make verify-auto`, `make verify` (it touches the
inference stack), and `make verify-cpu` once, since it brings up a family.
Docs: `docs/reference/clef.md` (new), `docs/spec.md` (the catalogue and
`decide`), `docs/reference/artifacts.md`, `docs/architecture.md` (the
decision path's family seam), `THIRD_PARTY_NOTICES.md` if a constant is
taken from the reference.

## The theme: decode speed on Metal

**Why this order.** Our own records already say where the verify batch
goes (ADR 0001, *Where the verify cost goes*): the chunk attention a
verify runs on 24 threadgroups costs about 103 ms per batch at 4K and
413 ms at 16K, and decides every context from 4K up. The single-row path
runs at 63 % of the 273 GB/s peak against about 78 % for MLX on the same
chip, and every verify budget also needs that gap closed. So: tools that
make an experiment take a minute (ENGN-18) and let us see the GPU's
limiters (KERN-20), then the levers ranked by measured cost.

**What the other families get.** Every lever below except ENGN-19 sits in
shared code:

| Lever | Unit | Qwen 27B | Bonsai 27B | Gemma 4 (E4B, 12B, 26B-A4B) | Muse 30B |
| --- | --- | :---: | :---: | :---: | :---: |
| Few-query split-KV verify attention | KERN-21 | ✓ | ✓ | ✓ (draft heads) | ✓ (speculation on) |
| Single-row and few-row matvec bandwidth | KERN-23 | ✓ | ✓ | ✓ (their encodings) | ✓ |
| DeltaNet recurrent verify, replay tape | ENGN-19 | ✓ | ✓ | — | — |

Bonsai runs on the Qwen adapter (`qwen35_metal.zig` with its Hadamard
fold); Gemma and Muse verify through the same `attentionChunk` and
`mmRows` routes. Every unit re-measures each family's decode it touches
through `make speed` before it closes.

### The loop (every experiment in every unit)

Fast, cheap to abandon, and nothing lands without a measured gain.

1. **Idea → micro-bench** (no model, seconds): `make bench-attention`,
   `make bench-matvec-rows`, `make bench-kernels`, or a new sweep in
   `inference/metal-check.zig`. Write the prediction down first, in the
   unit's ledger below. A kernel that misses its micro prediction by more
   than half is dropped here.
2. **Correctness in seconds**: `make test-metal` (the kernel fixtures,
   against the F64 CPU references, poison past the visible rows).
3. **End-to-end A/B** (about a minute per context, ENGN-18): `make speed
   ARGS='--contexts 512,4096,16384,32639'` interleaves the saved base
   binary and the candidate on restored prefixes.
4. **Keep rule.** Keep a change only when its median decode (or, for a
   verify lever, the batch cost C at the unit's row counts) improves by
   **≥ 2 %** at one context or more and no context regresses by more than
   1 %, over ≥ 5 interleaved pairs; the record's run-to-run spread is about
   1.5 %. Then `make verify-auto` (plus `make verify-long` for attention or
   cache changes) and commit `perf(inference): …` with the numbers in the
   body. Refresh the base binary (`make speed-base`) after each kept commit.
5. **Otherwise revert** (`git restore`, nothing committed), and add one
   ledger line: idea, prediction, measured, why it lost. The ledger is
   committed with the next kept change or at session end, and moves into
   the log when the unit closes, so a negative result stays findable and
   is not re-tried unchanged.

**Be brave, but bounded.** A unit lists more ideas than it needs; try the
cheapest falsifying measurement of each, and spend a session on none that
has not beaten its micro-bench. ENGN-18 and KERN-20 are tooling and land
without a speed gain; every other code change lands only through step 4.

**Apple GPU knowledge** found along the way (limiters, occupancy,
register limits, load widths, measured on this M4 Pro) goes into
[docs/reference/apple-gpu.md](docs/reference/apple-gpu.md), created by
KERN-20, with the source or the measurement for each fact.

### Order

| # | Unit | Sessions | Lands when |
| --- | --- | ---: | --- |
| 1 | KERN-23 — Weight streaming for one row and a few: decode and the verify body | 2 | 512 decode ≥ 11.5 tok/s, a 4-row verify C ≤ 150 ms at 512, or Qwen speculative prose 512 ≥ 18 tok/s; or closed at its ledger |

**Re-ranked after ENGN-20** (2026-10-02): speculation is on for Qwen
(draft 7), Gemma 12B QAT (5), E4B (6), and Muse (6), so a verify lever now
moves the default decode rate, and every decode lever moves the plain
path the drafters fall back to. Qwen's C / 50E at draft 7 is 1.21–1.42 at
512 and 4K (C 164–182 ms against 128–141 ms budgets) and 1.6–1.9 from
16K; both Gemma entries break even at 32K, where the verify attention is
linear in rows, so KERN-22 now pays in speculation as well as decode.
The drafter's proposal is 16–18 ms per Qwen batch at draft 7, its head
the output head KERN-23's idea (f) names.

**Re-ordered after ENGN-19** (user, 2026-10-01): ENGN-20 first, since
speculation now pays (an estimated 1.2–1.3× from the ENGN-17 acceptance
at the new C) and its measured E decides how much each decode lever is
worth; then KERN-23, the largest per-token lever (matvecs 85 of ~95 ms
per step) and the groundwork for a multi-row scalar verify body; then
KERN-22, which pays mainly at depth (about 14 ms per step at 32K, plus
the verify's attention there). Pull KERN-22 forward if long agent
sessions become the main use.

**Re-ranked after ENGN-19** (`--profile`, 4-row Qwen verify at 4K,
2026-10-01): matrices 195 ms, attention 14.5, DeltaNet 6.4 (was 41.3),
recover 2–3 ms, checkpoint 3. The verify's remaining cost is the matrices'
floor KERN-24 found; decode (KERN-22, KERN-23) is next, and ENGN-20
re-prices speculation at the new C.

**Re-ranked after KERN-24** (`--profile`, 4-row Qwen verify at 4K,
2026-10-01): matrices 195.3 ms (was 236.8; 2.3 decode steps), DeltaNet
41.3, attention 14.5. The matrices stay the largest term but KERN-24
found their floor (the padded 8×8 multiplies); ENGN-19 is next, then
decode. ENGN-20 must price the fragment tile's flat cost: a 7-draft
verify pays the same matrices as a 3-draft one.

**The order was ENGN-18's cost table** (bench.md § The decode-speed
baseline, `--profile` of a 4-row verify and a decode step, 2026-09-30).
A 4-row verify batch at 4K is 399 ms of kernel time: weight matmuls 237
(2.8 decode steps' worth), verify attention 106, DeltaNet 44; at 32K
attention is 824 of 1,114. KERN-21 first: the largest term at depth, and
the drafter's commit (48 ms per batch at 32K) runs the same
`attentionChunk`. KERN-24 next: the largest term at 512 and 4K. ENGN-19:
`delta_chunk` 38 ms + recover 14 + checkpoint 3 per batch. Then decode:
KERN-22 (decode attention 5.2 ms per step at 4K, 27.5 at 32K) and KERN-23
(weight matvecs 85 ms per step at every depth).

Identifiers are provisional in this order; they are fixed in the order
the units close. The units are independent of one another: re-rank them
when a kept change moves the cost table.

## KERN-23 — Weight streaming for one row and a few: decode and the verify body (2 sessions)

Base: recorded when the unit starts (APPS-19 and MODL-34 land first).

**Why, since ENGN-20.** Speculation is on for Qwen at draft 7, so a
default turn spends its time in verify batches: at 512, C 164 ms = propose
16.5 + checkpoint 3 + verify 138 + recover 2.5 + commit 4 (cold, greedy;
bench.md § The re-priced speculative verdicts). The verify's matrices sit
at KERN-24's floor: the fragment tile pads every 8×8 multiply and costs the
same from 1 to 8 rows, its rates at 4 rows 107 GB/s (Q4_K) to 152 (Q6_K),
against 179–212 GB/s for the single-row matvec in `make bench-kernels`.
Plain decode still runs wherever speculation is off (Gemma 26B-A4B, Bonsai,
files outside the catalogue, a turn after an image) and is 85 ms of
matvecs in a 95 ms step at 512.

**What the counters say** (apple-gpu.md, at full clocks): the Q4_K matvec
is issue-bound on the integer and complex pipe, not on memory, at half its
target occupancy with 192 registers. The multi-row matvec kernels for 2–8
tokens already exist (`nu_matvec_rows_{q4_k,q5_k,q6_k,iq4_xs}_t2..t8`,
`nu_matvec_rows_k_body` and siblings in `kernels.metal`), but only 2 rows
route to them (KERN-12): beyond that each token's accumulators push the
registers and the body loses to the tile. KERN-24's close names the gap:
a scalar body that shares activations across rows without that register
wall; none was tried. Cheaper decode arithmetic and fewer live registers
serve both halves of this unit.

**Session 1 — the single-row matvec.** Ideas in this order, each judged
first on `make bench-kernels` (Q4_K, Q5_K, Q6_K, IQ4_XS, IQ4_NL, Q8_0 at
Qwen's shapes), then by `make speed` decode at 512 and 4K: (b) the half
magic-number decode (a nibble OR'd into a half of exponent 1024, minus
1024) with float accumulation, per encoding; (c) scale and bias once per
group, `s·Σqx + b·Σx`; (d) 16-byte aligned loads per lane; then (a)
weights in Metal-allocated buffers instead of `newBufferWithBytesNoCopy`,
only if a capture shows an MMU limiter; (e) the command buffer split so
the GPU starts while the CPU encodes; (f) the Q6_K output head (about 1
GB) as its own kernel, which the drafter pays per proposed position too.
`--kernel-stats` reads each candidate's thread limit before it is timed.

**Session 2 — the multi-row body.** Carry session 1's decode into the
`_t3..t8` bodies with the accumulators bounded (rows × tokens per
SIMD group chosen so the thread limit stays at 1,024 in
`--kernel-stats`), and measure every encoding at 3–8 tokens against the
fragment tile with `make bench-matvec-rows ARGS="8 frag"`. Where the body
beats the tile at a token count, route verify batches of that count to it
(the routing that sends 2 tokens to `matvec_rows` today, in
`inference/src/backends/metal/root.zig`), per encoding. Kept only through
`make speed --verify-rows 4,8` and the speculative record below; Gemma's
Q4_0 verify gets the same body if Q4_0 joins.

**Predictions** (written before code; each idea's own goes into the
ledger):

- Q4_K single-row matvec 179 → ≥ 200 GB/s in `make bench-kernels`; 512
  plain decode 10.56 → ≥ 11.5 tok/s.
- A 4-token multi-row body ≥ 150 GB/s on Qwen's Q4_K and IQ4_XS shapes
  (the tile reads 107 and 122); Qwen's 4-row verify C at 512 177 → ≤ 150
  ms in `make speed`.
- Qwen prose 512 with speculation (`make spec-matrix ARGS='--model qwen38
  --contexts 512 --drafts 7 --cooldown 90 --rev <rev>'`, greedy and
  instruct): 16.5 / 15.9 → ≥ 18 tok/s.

**Lands when** any prediction is met with no context regressing, or the
unit closes at its ledger. **Correctness:** the per-encoding matvec and
multi-row fixtures in `metal-check` against the F64 CPU references
(poisoned rows past the visible ones), the trace gates of every family
whose encodings change, `qwen38-verify-depth-*` and
`qwen38-generation-metal` (verify rows against stepped decode), `make
verify`.

Gates: `make test-metal`, `make verify-auto`, `make verify`; the
speculative record above for the third prediction.

## AGNT-19 — Saved prefixes for the agent: the primed prefix and `/resume` across processes (2 sessions) — queued

`nuclis agent` prefills its system block and tool definitions at every
start (about 11 s for 934 tokens on Qwen, llm-guide.md § 22) and replays a whole
conversation on `/resume` (minutes at 16K). ENGN-18's `src/prefix_cache.zig`
already writes and restores a model snapshot keyed by model files, tokens,
and layout; this unit uses it for real work, where a stale state is a
correctness bug, not a timing one.

- **Key.** `prefix_cache.Key` gains the build: the binary's
  `nuclis --version` string and the git revision from `build_options`
  (a dev build's revision plus a dirty flag). A different build is a
  miss, never a restore: a saved state must equal what this build's
  prefill would compute. The file header records all four keys.
- **Session 1: the primed prefix.** `Completer.prime` (`src/agent/loop.zig`)
  first calls `prefix_cache.load` from `<NUCLIS_HOME>/cache/prefix/`
  (`src/paths.zig`); on a hit it restores and skips the prefill, on a
  miss it prefills as today, snapshots (already done), and saves.
  `restorePrimed` is unchanged. A `/ctx` change re-opens the engine and
  keys on the new capacity. Speculation on and off are different layouts
  (the draft block's cache), so each has its own file.
- **Session 2: `/resume`.** When the agent writes a session file
  (`src/agent/history.zig` / the save path), it also saves the model
  snapshot at the end of the last completed turn, keyed by the consumed
  tokens; `/resume` restores it when the rendered prefix's tokens match
  and prefills only the remainder, falling back to the replay otherwise.
- **Bounds.** A per-file bound (the session's used extent: about 150 MB
  of recurrent state plus 64 KiB per token for Qwen, 2.3 GB at 32K) and a
  directory budget (`cache.prefix_bytes` in `nuclis.json`, default 8 GB;
  0 disables), evicted oldest-accessed first after each save. `nuclis
  cache ls` and `nuclis cache clear` (APPS surface) show and empty it.
- **Limits, stated in the docs.** Exact token prefixes only (an edited
  system prompt or tool list misses); bound to capacity, KV precision, and
  the draft layout; disk cost as above.
- **Prediction.** The agent's first prompt appears with the primed prefix
  restored in ≤ 0.5 s instead of ~11 s; `/resume` of a 16K conversation
  in ≤ 2 s instead of minutes.
- **Correctness.** Unit tests: a build-key miss, a budget eviction, a
  corrupt file refused and re-prefilled (the ENGN-18 tests extended). A
  restored primed prefix gives the same first-turn tokens as a fresh
  prime (greedy, both backends' generation checks). The agent surface
  through `make shot` (a cold start, then a warm start showing the
  restore; a `/resume`), and `make agent-eval` before and after, since
  the loop's behaviour changes.

Gates: `make check`, `make lint-py`, `make verify-auto`, `make shot`,
`make agent-eval VARIANT=…`. Docs: `docs/reference/session.md`,
`docs/development.md` § User directories, the agent's help.
