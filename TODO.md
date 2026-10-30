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

Fish Audio Drama 3 is in progress on `fish-audio-drama3-support`. By the
user's decision (2026-10-30), its five units below share that one branch
and one pull request, an exception to one unit per branch. One session
leads; it may run at most two subagents (the same model) at a time, on
units or pieces that touch disjoint files. The user tested the artifact
in the TTS lab (`~/Code/ttslab`, speech.cpp 0.9.2) and chose Q4_K.

Delivered so far: the facts, the pin, and the Fish traces
(`tests/fixtures/drama3/`, `scripts/speech-traces/drama3_trace.py`); the
backbone, tower, and codec on the CPU, each replayed against Fish's
traces with its gate (`drama3-trace-cpu`, `drama3-codec-cpu`); the
sampler, the text and cue checks, the chunker, and the streaming
synthesizer with the silence stop and the retry; the backbone and tower
on Metal (`drama3-metal`) and the codec decoder on Metal
(`drama3-codec-metal`), taken by `nuclis speak` and the speech service
(a take at about 0.65 × real time); the round trip through Nemotron on
Metal (`drama3-roundtrip-metal`, WER 1.05 %); the stall measurement
(§ Stalls); and `drama-3` in the catalogue, `nuclis speak`,
`/v1/audio/speech` (whose log records the seed, redraws, and silence
stops), and the saved voices (a 44.1 kHz copy of the clip, Fish codes
keyed by the GGUF). In flight: answers to the Muon app's questions (cue
placement against stalls; the silence stop made relative to the take's
peak, since a fixed threshold cut whispers). Left: the verify tiers and
the pull request.

## Fish Audio Drama 3

Base: `0f08226`. The model: [fishaudio/drama-3](https://huggingface.co/fishaudio/drama-3),
a 35B-A3B text-to-speech model (a Qwen3.5-MoE-shaped backbone predicting
the semantic codebook, a 4-layer shared plus 9-layer residual "audio
tower" predicting codebooks 1–9, and Fish's 44.1 kHz DAC codec, 2,048
samples a frame), 83 languages, `<|speaker:N|>` dialogue, free-form
`[bracketed]` cues, cloning from a clip and its transcript. Fish Audio
Community License 1.0: no hosted service for third parties without
Fish's agreement, no cloning of a person without consent, "Built with
Fish Audio" on redistribution.

**The artifact.** `sakasegawa/drama-3-GGUF` `Drama-35B-A3B-v3-Q4_K.gguf`
at `9a01354d04ab6eb521bc9cdfc56ab6824c474af2`, 20,787,264,352 bytes,
SHA-256 `2a57b14c9b208a2ff833ab7305b920ca8f154ad807004a52d96a86a90a55523e`
(pulled, sidecar beside it). speech.cpp's own GGUF layout, architecture
`drama3`, 1,531 tensors: 522 Q4_K, 216 F16 (the codec), 793 F32; it
embeds the tokenizer, the prompt strings (`drama3.prompt.*`), sampling
and repeat-window defaults, chunking limits, and the codec geometry.
Chosen over the MLX conversion because every encoding in it already
executes on the CPU and Metal (no MLX affine unit), and speech.cpp
reports its layout matching Fish's implementation at 96–133 dB. The
mlx-audio port only matches PyTorch at 86 % semantic top-1, so it is not
a trace source. Semantics come from Fish's code,
`fishaudio/fish-speech` at `0f8a12e8a651062fdee9c33a894bdccd3a8513f1`,
unpacked at `.reference/fish-speech-0f8a12e/` (`fish_speech/models/text2semantic/fish_omni/`,
`fish_speech/models/dac/`, `fish_speech/content_sequence.py`); speech.cpp
is read only as the definition of its GGUF layout (tensor names, the
metadata it writes).

**Measured in the lab (2026-10-30, M4 Pro, speech.cpp, Q4_K).** Load 9 s,
20.0 GB footprint; 7.9 s of English in 6.4 s, first audio 0.56 s; an
Egyptian clone 6.2 s in 6.0 s, first audio 1.3 s, its voice file made
once in 6.6 s on the CPU. Stalls on short Egyptian phrases (Nemotron
coverage < 60 % or > 1.5 s of trailing silence), 2 voices × 4 seeds:
24 of 128 takes over 8 phrases at temperature 1, top-p 0.9 (top-k 30 or
off alike); on the 4 worst phrases 13 of 32 whole, 13 of 32 at
temperature 0.7/top-p 0.7, **5 of 32 split at sentence ends**. Two
kinds: the model speaks the first sentence of "إزيك يا باشا؟ عامل إيه؟"
then 10–28 s of silence (splitting fixes it), and a three-word phrase
comes out empty or as silence (needs a stop on silence and a retry).
bf16 stalls too (fcmeyer's 500-clip study: 4 % of clips), so it is the
model's, not the quantization's.

| Unit | Delivers | State |
| --- | --- | --- |
| Facts and pin | `docs/models/drama3.md`, inventory fixture, tokenizer check, pinned traces from Fish's code | done |
| Backbone on the CPU | `drama3` adapter, binding, the Qwen3.5-MoE forward against the traces | done |
| Tower, codec, sampler on the CPU | codebooks 1–9, the codec decoder and encoder, Fish's sampler, the first WAV | done |
| Metal | the three networks on the GPU, gated against the CPU, speed recorded | done (the encoder stays on the CPU, once per voice) |
| Voices and the API | saved voices, speakers, cues, chunking, `speak`, `/v1/audio/speech`, catalogue, round-trip gates | done |

### Facts and pin

Read the GGUF and Fish's code; write what the next units need, then
rewrite their sections at the level of files and functions.

- `scripts/gguf-inventory.py <file> --out inference/src/models/fixtures/drama3.json`
  (the binder's fixture); the pin in `docs/models/catalogue.md` § Pinned
  commits and digests.
- `docs/models/drama3.md` (listed in `docs/models/README.md` and
  `docs/README.md`): artifact, every metadata key the adapter reads,
  tensors by role, and the forward pass in equations, each fact sourced
  (file, or Fish's code by path): the backbone (Gated DeltaNet and gated
  attention as Qwen3.8's, with partial RoPE on 64 of 256 dimensions; the
  softmax top-8 router with normalized weights and the sigmoid-gated
  shared expert; zero-centred norms, which the GGUF may store shifted);
  the prompt (`reference-bracket-v1`: system, `[Reference N]` text and
  speech blocks, the `<|voice|>` assistant turn, how VQ codes enter as
  embeddings); the tower (`codebook_mtp_shared`: the 4 shared enrichment
  layers, the 9 residual layers, `eh_proj` over the normed hidden and
  embedding, heads); the codec (encoder rates 2·4·8·8, transformer
  windows 512 and 128, the ×2×2 down/up-sampling, semantic codebook 4,096
  and 9 residual codebooks of 1,024 at dimension 8, decoder rates
  8·8·4·2, snake activations); sampling (Fish's top-k/top-p/temperature
  and the repeat window: 16, top-p 0.95); chunking (300 bytes, 3
  speakers, the first and the latest chunk as context); stop and length.
- The tokenizer: the GGUF's vocabulary through nuclis's BPE (the
  `drama3.tokenizer.pre_tokenizer` regex, the 4,172 added ids), checked
  on a dozen adversarial strings against Fish's tokenizer output.
- Traces: a script under `scripts/` (stdlib plus torch in a throwaway
  venv under `.reference/`, never a dependency) that builds Fish's
  modules one layer at a time with weights dequantized from the pinned
  GGUF, and writes per-layer hidden states, the semantic logits, each
  tower layer and head, and the codec's stages for one short prompt and
  a 2 s clip, under `tests/fixtures/drama3-*/` with a row in
  `tests/fixtures/provenance.md`.
- The lab's stall harness, in the record: a `scripts/` port that drives
  `nuclis speak` and Nemotron, so later units measure the same thing.

### Backbone on the CPU

`inference/src/models/drama3.zig` (the `family` namespace, registered in
`models/root.zig`) binds the `drama3.text.*` keys and `blk.*` tensors;
`drama3_runtime.zig` runs the forward over `cpu.*`, reusing Qwen3.8's
DeltaNet (`backends/cpu/recurrent.zig`) and attention, and the experts
path Gemma 4 26B-A4B brought (`backends/cpu/experts.zig`) with Qwen's
router. Gates `drama3-backbone-trace-cpu` (tier `verify-cpu`). To be
rewritten by the facts unit.

### Tower, codec, sampler on the CPU

`drama3_runtime.zig` gains the tower and `Synthesizer`;
`inference/src/audio/fish_codec.zig` (decoder) and
`fish_codec_encoder.zig`; the sampler, repeat window, and frame budget.
The stall mitigations go here: split at sentence ends (as the chunker
does), stop on a run of silent frames past the speech, and retry with
the next seed when a chunk ends with no speech. Gates
`drama3-tower-trace-cpu`, `drama3-codec-cpu`. To be rewritten by the
facts unit.

### Metal

`drama3_metal.zig` from Qwen3.8's plan (DeltaNet, gated attention) and
Gemma's expert kernels; the tower as dense 2,048-wide layers; the codec
convolutions and transformers in `fish_codec_metal.zig`. Gates
`drama3-trace-metal`, `drama3-codec-metal`, `drama3-roundtrip-metal`
(Nemotron WER). Speed against the lab's speech.cpp numbers on the same
machine. To be rewritten by the facts unit.

### Voices and the API

The speech path (`src/speak.zig`, `src/api/speech/service.zig`,
`src/voices.zig`, `src/audio/catalog.zig`) is written against
`qwen3_tts_runtime`; it becomes a union over the two families. Saved
voices as for Qwen3-TTS Base (`/v1/audio/voices`, `voice` on
`/v1/audio/speech`), the codes cached per model as the Base caches its
encoding; `<|speaker:N|>` with one saved voice per speaker; cue checks
(the six reserved strings refused, speakers 0–15, balanced brackets, no
nesting, paired tags closed, line breaks as spaces); the catalogue entry
`drama-3` with its license; `serve.memory_bytes` (at most 16 GiB today)
against its ~21 GB. To be rewritten by the facts unit.

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
