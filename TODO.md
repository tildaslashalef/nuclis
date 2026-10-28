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

In progress: **Q5_0 and Q5_1** (below), on `q5-0-q5-1-encodings`. Muon's
voice loop (Gemma 4 calls in plain quotes, a started chunk for hosted web
calls, state reused after a search, phrases the recognizer should expect)
is in its pull request, whose description is its record; the next release
(0.11.0) carries it with the model-management unit
([nuclis#36](https://github.com/tildaslashalef/nuclis/pull/36)).

## Q5_0 and Q5_1

Base: `a52d799`. Branch `q5-0-q5-1-encodings`.

**Why.** K-quants need rows that are multiples of 256. Where a row is not
(Gemma 4 26B-A4B's `ffn_down`: 2,112 dense, 704 per expert), llama.cpp's
quantizer falls back to a 32-value block of similar size: Q4_K → Q5_0,
Q5_K → Q5_1, Q6_K → Q8_0, IQ4_XS → IQ4_NL. So most K-quant Gemma 4
files carry Q5_0 or Q5_1 `ffn_down` tensors, and `model inspect` refuses
them ("encoding id 6/7, which nuclis does not store"). Measured on the
Hub headers: Unsloth's non-QAT 26B-A4B `UD-Q4_K_S` has 29 Q5_0 expert
`ffn_down`, `UD-Q4_K_M` 29 Q5_1.

**The formats** (ggml-common.h, blocks of 32, little-endian):
- Q5_0, id 6, 22 B: `d` F16, `qh` u32, `qs[16]`. Value j (0..15):
  `q = (qs[j] & 15) | ((qh >> j) & 1) << 4`; value j+16: `q = (qs[j] >> 4)
  | ((qh >> (j+16)) & 1) << 4`; `w = d·(q − 16)`.
- Q5_1, id 7, 24 B: `d` F16, `m` F16, `qh`, `qs[16]`; same codes;
  `w = d·q + m`.

**Tasks.**
1. `inference/src/tensor/encoding.zig` `layout`: ids 6 and 7.
   `inference/src/quant/decode.zig`: `row` and `validateRow` decode both;
   hand-calculated tests pin both halves of the `qh` plane, every code,
   the bias of 16, Q5_1's offset, successive scales, and the size errors.
2. Fixture `inference/src/quant/fixtures/five-bit.json`: eight blocks of
   each (256 values) with seeded payloads, written by
   `scripts/five-bit-fixtures.py` from the equations above; a `decode.zig`
   test holds the CPU decoder to it, `metal-check` the GPU (add
   `"five-bit"` to every fixture list, `tameScales` positions).
3. Metal: `dequant.metal` `nu_dequant_q5_0`/`_q5_1` and
   `nu_block_geometry`; `matvec.metal` `nu_matvec_q5_0`/`_q5_1` (Q4_0's
   lane-per-block body, the fifth bit spread into the packed code bytes,
   the bias through the input sum), in `matvec_split.metal`'s
   `nu_segment_sums`; `matmul.metal` tile decoders and the routed sets
   (64×64, `_32`, `_8`, `_f2`) and the gathered expert tile;
   `kernels.zig` names; `ops/matmul.zig` `blockAligned` (Q5_0 2, Q5_1 4),
   the pickers, `matmulGeometry`; `ops/matvec.zig` `specializedMatvec`;
   `ops/experts.zig` the expert tile for 2, 6, 7. `metal-check`'s expert
   chains run the down projection in Q4_0, Q5_0, and Q5_1.
4. `inference/src/models/gemma4.zig` `executableEncoding`: 6 and 7 (Qwen's
   set unchanged: no Qwen file needs them).
5. Docs: `docs/engine/quantization.md` (equations, evidence, the fallback
   rule), `docs/models/catalogue.md` (implemented list), the
   `dequant.metal` header table.

**Gates.** `zig build test`; `make metal-check`; `make verify-auto`;
`make verify` once; the CPU tier is not needed (no existing CPU arithmetic
changes, the new decoders are pinned by unit tests). The cross-encoding
check is not run in this unit (the user's call: no 60 GB of reference
downloads); end to end, a local Gemma 4 26B-A4B Q4_K_M file with Q5_0
`ffn_down` must load, generate coherent text on Metal and the CPU, and
agree between them on a short greedy run.

## Deferred

Agreed and not scheduled; when one is picked up, it becomes a unit above.

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
