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

The model-management unit is in progress on `model-management`
(2026-10-28): what Muon's engine manager and settings need from `nuclis
model` and `nuclis serve`. The speech unit shipped in v0.10.0.

| Unit | Branch | State |
| --- | --- | --- |
| Model management for Muon: `model rm`, `pull --force` that downloads again, resumable pulls, `serve --web-url` | `model-management` | in progress |

## Model management for Muon

Base: `babce65`

Asked by the user (2026-10-28) for Muon's Settings › Engine (Redownload,
Remove) and its first-run downloads. One branch, one pull request.

| # | Task | Depends on |
| --- | --- | --- |
| 1 | `nuclis model rm` | — |
| 2 | `pull --force` downloads every file again, replacing it once verified | — |
| 3 | Resumable pulls: SIGTERM keeps the partial file, the next pull continues | 2 |
| 4 | `serve --web-url <url>` and `--no-web` | — |

#### 1. `nuclis model rm <name> [--with mmproj,mtp | --all] [--json]`

`src/model.zig` `rm`, parsed in `src/cli.zig` `parseModelArgs` (action
`rm`), with help, completion, and `docs/guide/getting-started.md`.

- **The entry's files.** A catalogue name: the main file, its support
  files (a decision or speech entry's `support`, a speech entry's codec
  and `codec_support`; clef-flash's backbone), each with its
  `.nuclis.json` sidecar; companions (`mmproj`, `mtp`) only with `--with`
  or `--all`, as `pull` takes them. A registry name: its main file and
  the `mmproj`/`mtp` it names (only with `--with`/`--all`), resolved as
  `model ls` resolves registry entries.
- **Shared files stay.** A file another *present* entry uses (a
  catalogue entry, or a registry entry of another name pointing at it)
  is kept and reported with that entry's name. A registry entry of the
  same name as the catalogue entry, which `config init` writes for every
  catalogue model, is the same model, not a second user.
- **Tidy.** Each directory emptied by the removal, up to the models
  directory, is removed.
- **Report.** Text, or with `--json` one object on stdout: `removed`
  (path, bytes), `kept` (path, the entry that keeps it), `freed_bytes`.
  A name nothing pulled is an error naming it (non-zero exit); a partly
  present entry removes what is there. The registry entry stays: `model
  ls` then shows it missing.
- **A running `serve`.** macOS keeps an open mapping valid after the
  file is unlinked, so a model already open keeps working until the
  server closes it; the disk space is freed then, and reopening it fails
  as a missing file. Documented, not refused (the server cannot be asked
  from here). Tests over a temporary models directory: an entry, a
  shared file kept, the same-name registry twin, a partly present entry,
  a name never pulled, the directory tidying.

#### 2. `pull --force` downloads every file again

Today `--force` replaces only a file whose sidecar records other content;
a verified copy is kept (`verified_local`). Now `--force` downloads every
file of the entry again and replaces each one only once the new copy is
verified (the old file stays until then). The `huggingface` package's
`Request` gains `replace: bool`: skip `verifyExisting`, and publish the
verified temporary file over the destination (`Io.File.Atomic`'s
replace, where today it links and refuses to clobber). `src/model.zig`
stops deleting files before the download under `--force` and passes
`replace`; the sidecar is rewritten after. `--json` marks each file
`xet` or `http`, never `verified_local`. Help, `docs/guide/getting-started.md`
§ pulling, the release notes say `--force`'s meaning grew (0.x).

#### 3. Resumable pulls

Today an interrupted pull removes its partial file (`AtomicFile` cleanup),
and the next pull starts that file again. Now:

- **SIGINT** keeps its meaning: stop and remove the partial file
  (Muon's Cancel).
- **SIGTERM** stops and keeps the partial file and its state (Muon's
  quit): `<path>.partial` beside the destination and
  `<path>.partial.json` with the repository, the revision, the file's
  SHA-256 (or git blob) and size, and the bytes written.
- **The next pull** of the same file at the same revision and digest
  hashes the partial bytes already on disk into the verifier and asks
  `transfer.fetch` for the rest (its `start` already exists); the whole
  file is still verified before publish, so a damaged partial fails the
  digest and is removed. A partial for another revision or digest is
  removed and the file starts over. `--force` ignores a partial.
- `src/interrupt.zig` tells the two signals apart; the package gets a
  `keep_partial` cancellation outcome. Tests in the package with a fake
  transport: a resumed file equals one downloaded whole; a stale partial
  is discarded; SIGINT removes. A real check: SIGTERM a pull of a 1–2 GB
  file midway, pull again, see it continue (bytes transferred < size).

If it outgrows the unit, it goes to *Deferred* with what was learned.

#### 4. `serve --web-url <url>` and `--no-web`

Muon must not rewrite `~/.nuclis/nuclis.json`, so the hosted web tools
take their address from the command line: `--web-url <url>` turns the
tools on against that URL (the same `web.validBase` check as `web.url`),
`--no-web` turns them off whatever the file says; the two are exclusive.
`src/api/root.zig` `parseArgs` and the web service's construction;
`GET /v1/health` already reports `web`. Help, `docs/guide/api.md` §
Hosted web tools, `docs/guide/configuration.md`.

## Deferred

Agreed and not scheduled; when one is picked up, it becomes a unit above.

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
