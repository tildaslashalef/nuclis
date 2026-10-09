# TODO — active plan

This file is the queue: the units agreed and not yet merged, each with its
design, and a *Where we are* note. A unit is one session, one branch, and
one pull request, whose description is the record of the work; when it
merges, its section is deleted here, and when the last one goes, this file
is emptied back to this header. Requirements live in [docs/spec.md](docs/spec.md); the
engine map in [docs/architecture.md](docs/architecture.md); how to build,
test, and measure in [docs/development.md](docs/development.md).

Session protocol (also in [AGENTS.md](AGENTS.md)): read this file first. If
it lists work, summarize *Where we are* and ask the user how to continue. If
it is empty, ask what to work on and write the agreed plan here.

## Where we are

The plan below was agreed on 2026-10-08. It is **one unit over several
sessions on one branch, `embedding-gemma-2`**, cut from `main` at
`5e69b80` and last merged with `main` at `b0297c2`.

Base: `b0297c2`

**The user's rule for this unit: commit on the branch, do not push and do
not open the pull request until the whole unit is implemented** (this
overrides the usual "push and draft a pull request at the end of each
session").

**Session 1 is delivered (2026-10-09).** The three files are pulled and
pinned. The facts are in `docs/models/embeddinggemma.md`, which is now the
reference for everything in *The theme* below; where the two disagree, the
document wins. The inventory fixtures, the llama.cpp oracle at build
`b11514` (`.reference/llama.cpp-embed`), the sentence-transformers oracle
(`.reference/venv-embed`), the 104-case input set, and the recorded
vectors, traces and mel features are committed
(`tests/fixtures/provenance.md`, `embeddinggemma-*` rows).

**Session 2 is delivered (2026-10-09).** The binding, the CPU forward, the
`Embedder` (`inference/src/embed.zig`), and `embeddinggemma-check` with
three `verify-cpu` gates are committed. On the BF16 file the CPU forward
matches Google's float32 to `1 − cos` = 4.0e-12. Side fix in this unit:
`Engine.open` returned its vocabulary by value while the encoder kept a
pointer to the local copy; it is now on the heap (commit `c2a02a4`).

**Session 3 is delivered (2026-10-09).** The text encoder runs on Metal
(`inference/src/models/embeddinggemma_metal.zig`, `Embedder` with
`Backend.metal` and `embedBatch`). Results:
- **Accuracy.** BF16 against Google's f32: `1 − cos` 4.2e-12. Q8_0
  against llama.cpp: 4.2e-7. Packed batches give each input's vector bit
  for bit.
- **Gates.** `embeddinggemma-vectors-metal` and
  `embeddinggemma-google-metal`, tier `verify`, 7 s each.
- **Attention.** `Backend.attentionSegmentsGrouped` materializes the
  scores. The planned online-softmax kernel was built, twice, and ran near
  the 0.7 TFLOP/s ceiling the prefill chunk kernels had shown, so the
  scores and P·V are F32 matmul tiles now.
- **Rates against llama.cpp** (which stages matmul activations as half).
  At 64 × 256 tokens: 38.3 inputs/s against 41.0 on Q8_0, level on BF16.
  Ahead 3–10 % at 8192 tokens, behind 9–14 % on one 512-token input. The
  record is in `docs/models/embeddinggemma.md` § The Metal plan.
- **The lever left is the user's call.** The batch is 80 % F32 matmul at
  the generic tile's ceiling; half-operand tiles would be about 35 %
  faster and are what the card's f16 warning rules out
  (`docs/engine/metal-backend.md` § Segments attention with materialized
  scores).

`make verify-auto` and `make verify` pass.

**Session 4 is delivered (2026-10-09).** The `embedding` kind, the
`embeddinggemma-2` catalogue entry (`embedding_entries`, pulled with
`--with mmproj`), `embed.model`, the wire types in `src/embedding/`
(`catalog.zig`, `request.zig`, `response.zig`), and `nuclis embed` with
help and completion are committed; spec §3, §4, §5.8, a new §5.10, §6 and
§10 say so. The CLI's vectors agree with the oracles as the gates do
(`docs/models/embeddinggemma.md` § `nuclis embed`). Also in this session:
- **Kind checks.** Text commands, the chat API (`api/chat/model.zig`) and
  `decision/catalog.zig` refuse an embedding name, each saying which kind
  it is. `model pull <owner/repo> --register` records a `gemma-embedding2`
  GGUF as `"kind": "embedding"` (key `embed.model`). The `model ls`
  listing is schema 5.
- **`serve.port` defaults to 9000 (user, 2026-10-09)**, not 8000, which
  `make site-serve` keeps. Code, help, completion, `api-check.py`, the
  README and the guides changed with it (its own commit).

What Session 5 inherits:
- `request.inputFromJson` already parses one API input (a string or chat
  content parts); `image_url` and `input_audio` parts are refused there,
  and `--image`/`--audio` in `embed.parseArgs`, with
  `UnsupportedModality`. Sessions 6 and 7 lift those refusals and add the
  two flags to the help page and the completion table, which list neither
  yet.
- `response.Body` is the data both writers need; the API adds OpenAI's
  writer beside `response.write`. `request.max_inputs` (2048) is the
  per-request bound for both surfaces.
- A file without a sidecar has its SHA-256 computed per command
  (`embedding/catalog.digest`, about 1 s for the Q8_0 file); the server
  should compute it once per open model.
- With a task, the prefix goes in front of an input's first text part.
  Session 6 checks that placement against sentence-transformers on
  `mix.logo` before images ship.

**Session 5 is delivered (2026-10-09).** `POST /v1/embeddings`
(`src/api/embeddings/`: service, batcher, pool), the `embedding` memory
kind, embedding rows in `GET /v1/models` and `/v1/health`, the
`docs/guide/api.md` § Embeddings section with § Measured rates, and spec
§5.10 and §6 are committed. `scripts/api-check.py --only embeddings`
passes: float and base64 vectors are bit-identical, and a batch,
concurrent requests and a request across passes equal single calls bit
for bit. Decisions taken in the session:
- **Token arrays are refused** (`400 unsupported_feature`), not accepted
  as the plan said: OpenAI clients send them tokenized with OpenAI's
  vocabulary (LangChain's `OpenAIEmbeddings` by default), which would
  give noise vectors silently. The message names LangChain's setting.
- **One pass per GPU item.** A request larger than a 2,048-row pass
  continues in the next, so passes run between a generation's steps.
  2,048 against 8,192 rows: 38.0 against 38.3 inputs/s at 64 × 256
  tokens, within the spread, while a pass holds the GPU 0.2 s instead of
  1.7 s.
- `/v1/models` lists `modalities: ["text"]` whatever the projector:
  what the server embeds, not what the model could.

What Session 6 inherits:
- `request.fromJson` and `inputFromJson` refuse `image_url` and
  `input_audio` parts, and `embed.parseArgs` `--image`/`--audio`, with
  `UnsupportedModality`; lift them, and add the flags to the help page
  and the completion table.
- The service's `list` adds `image` to `modalities` once the projector
  is pulled and bound. The pool counts only the main file; the
  projector's half is added with `Budget.resize` when it binds.

Next: **Session 6, Images**.

| Session | What |
| --- | --- |
| 1. Facts and oracles (done) | Pull and pin the files, `docs/models/embeddinggemma.md`, inventory fixtures, the second llama.cpp checkout and the sentence-transformers oracle, recorded traces and vectors |
| 2. Text encoder on the CPU (done) | `gemma-embedding2` adapter, the bidirectional KV-free forward, pooling, projection, normalization, Matryoshka; `Embedder` in `inference/src/embed.zig` |
| 3. Text encoder on Metal (done) | Grouped-query segments attention with materialized scores; packed batches, bit-exact; F32 activations throughout; measured rates |
| 4. `nuclis embed` and the catalogue (done) | `ModelKind.embedding`, the catalogue table, the shared wire types in `src/embedding/`, tasks and titles, `model ls` |
| 5. `POST /v1/embeddings` (done) | The service, its batcher and pool, `Kind.embedding` in the memory budget, `GET /v1/models` fields, the API guide and spec |
| 6. Images | The small Gemma 4 vision encoder from this mmproj, rows spliced unscaled, 280 soft tokens by default, `--image` and image parts |
| 7. Audio | AudioToolbox decode to 16 kHz mono, the log-mel front end, the `gemma4a` conformer on CPU then Metal, `--audio` and `input_audio` parts |
| 8. Acceptance and close | Q8_0 against BF16 against Google's f32 vectors, a retrieval check, the rates, the documents, the pull request |

## The theme: EmbeddingGemma 2, a third kind of model

[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) maps
text, images, audio (and video, not here) into one 768-dimensional,
L2-normalized space. It is neither a language model (it decodes nothing) nor a
decision model (it answers no question). It becomes a third registry and
memory kind, `embedding`, beside `generation`/`language` and `decision`.

**Decisions (user, 2026-10-08).**
- **Files: `embeddinggemma-2-Q8_0.gguf` with `mmproj-BF16.gguf`** from
  `unsloth/embeddinggemma-2-GGUF`. The reasons, written into the model
  document:
  - Embedding is a single prefill-only pass over a 130M transformer, so
    weight bandwidth barely matters. Half of the file is `token_embd`
    (262144 × 512), which is gathered, not multiplied.
  - Stored vectors outlive the run, so quantization noise is paid in every
    index row.
  - The model card says activations overflow f16, and our K-quant prefill
    tiles hold activations as half. Q8_0 runs on the generic F32-operand
    `nu_matmul`.
  - The mmproj in BF16 is the original weight, as every other catalogue
    entry's mmproj is. F16 is the format the card warns against.
  - Session 8 measures the choice against the BF16 file and Google's f32
    vectors.
- **Modalities: text, images, audio.** Video is a later unit: 1 fps
  frames through AVFoundation, 140 soft tokens per frame, at most 32
  frames, token `<|video|>` 258884. It reuses the vision encoder.
- **No implicit task.** The CLI and the API embed exactly what they are
  given. `task` (and `title` for documents) are opt-in fields that render
  Google's prefixes. Prefixes apply to text parts only.
- **The goal (user, 2026-10-09): a fast text, audio and image embedding
  API over the user's own files, for a RAG application built later.**
  Throughput over a corpus (packed batches on Metal, the API's batcher) is
  the measure that matters, beside the vectors' agreement with Google's.
- **The personal index is a later theme.** Searching the user's own docs,
  audio and images is planned after this unit merges. This unit rewords
  spec §10 ("an embedding index is ruled out") to name it as a separate
  theme, not ruled out.
- **F32 only on Metal (user, 2026-10-09).** Half-operand matmul tiles
  would make a batch about 20–25 % faster. They are not built: f16 may
  overflow (the feed-forward down input is not normalized), and bf16 gives
  up the agreement with Google's vectors. Session 8 measures the
  activation ranges. A bf16 opt-in is revisited only if indexing speed
  matters once the API exists, accepted only at cosine ≥ 0.99999 against
  Google's vectors and unchanged retrieval top-5.
- **One branch, no push until complete** (see *Where we are*).
- **A catalogue and registry entry, the README at the close (user,
  2026-10-09).** Session 1 pulled the files without `--register`: until
  `ModelKind.embedding` exists, a registered entry would be a generation
  model. Session 4 adds the catalogue row and the registry kind; Session 8
  updates the README.

**Facts read during planning (2026-10-08), now recorded in `docs/models/embeddinggemma.md`.**

Repos:
- `unsloth/embeddinggemma-2-GGUF` at `031f0d4b35536f69ab3509d4893c923264fcf253`:

  | File | Bytes | sha256 |
  | --- | --- | --- |
  | `embeddinggemma-2-Q8_0.gguf` | 309855520 | `6f1bd4ac6c5df7444f9cca7ca36cafe6cfa34cd6f49fefb1e0b4be8143aed8bc` |
  | `embeddinggemma-2-BF16.gguf` | 557950240 | `f315cbbb30dd487e44d501c8902abe88808755e43753a96beed1964f0a48aa4f` |
  | `mmproj-BF16.gguf` | 982074880 | `995aaa56e88b9b631f651861d659b728a625ccc02a238be37cff56cf11dc0032` |

- `google/embeddinggemma-2` (safetensors, sentence-transformers, not gated)
  at `914f7f89142e33e77833254d9c9b90c3cef7303b`.

Text model (`general.architecture = "gemma-embedding2"`, 413 tensors):
- 24 blocks, width 512, FFN 2048 (GELU-tanh, gated).
- 4 query heads. Sliding layers have 2 KV heads with head width 256; the
  global layers (5, 11, 17, 23) have 1 KV head with width 512.
- `attention.causal = false`.
- `sliding_window = 1024` is the **symmetric** window: Google's
  `sliding_window` 512, doubled by the converter. Key `j` is visible from
  query `i` when |i − j| ≤ 512, which is llama.cpp's
  `LLAMA_SWA_TYPE_SYMMETRIC`, and exactly the `window` semantics of
  `Backend.attentionSegments`.
- RoPE bases are 1e6 (global) and 1e4 (sliding), plain RoPE on every
  layer with no `rope_freqs` tensor. Rotary width is 512 global and 256
  sliding.
- The attention scale is 1.0: the Q/K norms do the scaling, and V is
  RMS-normalized without a weight.
- Per-layer inputs: `per_layer_model_proj` [512, 24·512] and
  `per_layer_proj_norm` [512]. There is **no `per_layer_token_embd`**, so
  `ple = rmsnorm_w(P · x0 / √512)` per 512-wide slice, with no table term
  and no `/√2`.
- After the FFN residual add:
  `x += rmsnorm(proj · (gelu(inp_gate · x) ⊙ ple[l])) ⊙ post_norm`, then
  `x *= layer_output_scale`.
- At the end: `output_norm`, then the separate `output.weight` [512 → 768],
  mean over **every** row (prompt, BOS and EOS included; sentence-transformers
  `include_prompt: true`), then L2 normalization. The projection has no
  bias, so projecting before or after the mean is the same.
- Encodings in the Q8_0 file: Q8_0 matrices, F32 norms and scales, and
  `per_layer_model_proj` in **BF16**.
- Tokenizer `gemma4`, 262144 tokens, `add_bos_token` and
  `add_eos_token` both true (BOS 2, EOS 1, pad 0).
- The chat template only concatenates text and places `<|image|>`,
  `<|audio|>` or `<|video|>` placeholders. The context is 8192 tokens
  shared by every modality.

mmproj (`clip`, 963 tensors):
- **Vision** (`gemma4v`):
  - 16 blocks, width 768, FFN 3072, 12 heads of 64, patch 16,
    `position_embd` [768, 10240, 2].
  - **No clipped-linear bounds.** `vision/gemma4.zig` already treats them
    as optional.
  - `mm.input_projection` [768 → 512].
  - Default 280 soft tokens per image (`max_soft_tokens`, pooling kernel
    3), configurable from 70 to 1120. Mean 0 and std 1, so the only
    normalization is the 1/255 rescale.
  - Tokens: BOI 255999, image 258880, EOI 258882.
- **Audio** (`gemma4a`), the same encoder as Gemma 4 E4B's:
  - Two conv2d subsampling layers with 128 and 32 channels.
  - 12 conformer blocks of width 1024 with 8 heads. Each has two
    half-step FFNs (`ffn_*`, `ffn_*_1`), chunked local attention (chunk
    12, left context 13, right context 0, logit cap 50, `attn_k_rel`,
    `per_dim_scale`), and a conv module (`conv_pw1` GLU, depthwise kernel
    5, `conv_pw2`). Residual weight 0.5, clipped linears everywhere.
  - Then `pre_encode.out` and `mm.a.input_projection` [1536 → 512].
  - Front end: 16 kHz mono; frame 320, hop 160, FFT 512; 128 mel bins over
    0–8000 Hz; `mel_floor` 1e-3; no preemphasis, no dither.
  - 40 ms per token (25 per second). Tokens: BOA 256000, audio 258881,
    EOA 258883.

Oracles: see `docs/models/embeddinggemma.md` § Reference oracle status.
llama.cpp is build `b11514` (`de7fa0a3c`, the newest build on 2026-10-09;
release `v0.6.0` predates the model), and the main pin `7620399` does not
move.

What our code assumes today (the reasons this is a new family):
- `models/gemma4.zig` accepts only `architecture = "gemma4"` and picks a
  config by `block_count` (48/30/42).
- It requires `rope_freqs` and `per_layer_token_embd`, treats a separate
  `output.weight` as `UnexpectedTensor`, and has no BF16 in
  `executableEncoding`.
- Its runtime and plan always write a KV cache, and their windows are
  one-sided.
- `Backend.attentionSegments` (the Laya path) is symmetric and segment-aware,
  but it has no GQA, its width is at most 256, and its rows are at most
  `attention_full_max_rows` = 4096: `nu_attention_segments` keeps every score
  in threadgroup memory.
- There is no audio code anywhere. `chat/wire.zig` refuses audio.

## Session 6. Images

**Why.** Photos and screenshots of documents are half of the personal
corpus.

1. **The encoder.** Make `vision/gemma4.zig` accept this mmproj:
   - It is `siglip.small` (16 blocks, 768, 12 heads) with no clamp
     tensors, which `ClampBounds` already treats as optional, and
     `projection_dim` 512.
   - Load only `v.*` and `mm.input_projection`, lazily on the first
     image (`isAudio` already leaves `a.*` unbound).
   - The projector's output width must equal the text width, 512.
2. **The budget.** Default to 280 soft tokens for this family (Google's
   `max_soft_tokens`), not chat's 1120. The soft-token count follows the
   aspect ratio inside the budget (the 1254 × 1254 logo gives 256, the
   1200 × 630 card 276). **The resize must be Google's, not llama.cpp's**:
   `clip.cpp`'s bicubic costs 0.5 % of cosine on `image.og` against
   PIL's antialiased bicubic. Before reusing the
   `preprocess.smartSize` letterbox, check its size and resampling against
   `Gemma4ImageProcessor`. `nuclis.png` is RGBA: drop alpha as PIL's
   `convert("RGB")` does. `--image-tokens 70..1120` on the CLI
   and `nuclis.image_tokens` in the API raise or lower it, and it is
   recorded in the response, since it changes the vector.
3. **Splicing.** BOI 255999, the projector rows, EOI 258882, at each
   `<|image|>` or in part order.
   - The rows enter as `x0` unscaled, and their `ple` comes from
     `P · x0 / √512`.
   - The whole input is one bidirectional pass.
   - Confirm the exact framing (BOI/EOI present, image-token count)
     against the oracle's token dump before coding it.
4. **Checks.**
   - Add gates `embeddinggemma-image-cpu` and `embeddinggemma-image-metal`
     against `embeddinggemma-image.logo` and `-mix.logo`: projector rows
     (`media-0.f32`) by trace bounds. These test the encoder from
     llama.cpp's pixels. The vectors are compared by cosine against
     `st-f32`, the image oracle (llama.cpp's own image vectors are 0.995
     to 0.99986 of it).
   - Run `nuclis embed --image` and send an `image_url` part.

**Gates.** `zig build test`, `zig build test-metal`, `make verify-auto`,
and `make verify` (shared vision code changed).

## Session 7. Audio

**Why.** The largest new piece: nuclis has no audio at all. The
conformer is the same one Gemma 4 E4B carries, so this also prepares audio
input for chat (not in this unit).

1. **Decoding.** Add `inference/src/audio/audio_bridge.m`, beside
   `image_bridge.m`: AudioToolbox `ExtAudioFile` decodes
   WAV/AIFF/MP3/M4A/FLAC and converts to 16 kHz mono f32, behind a C
   interface and an opaque handle.
   - Bound duration by the token budget: 8192 tokens is about 327 s, and
     the input size is bounded.
   - The failure is typed, not a panic.
2. **The front end.** Add `inference/src/audio/mel.zig`, pure Zig with no
   I/O: frame 320, hop 160, FFT 512, a 128-bin mel filterbank over
   0–8000 Hz, log with `mel_floor` 1e-3, and right padding.
   - Test it against the `Gemma4AudioFeatureExtractor` features in
     `tests/fixtures/embeddinggemma-audio-features/` (frames × 128, padding
     dropped). The HF processor is the oracle for the features.
   - Record the filterbank construction in `docs/engine/audio.md`. If any
     constant is format-defining, record it in `THIRD_PARTY_NOTICES.md`.
3. **The encoder on the CPU.** Add `inference/src/audio/gemma4a.zig`
   (binding of `a.*`, `mm.a.*`) and its CPU `Runtime`:
   - The two conv2d layers, then 12 conformer blocks.
   - Each block: the clipped linears, the chunked local attention with
     relative positions and logit cap, and the conv module.
   - Then `pre_encode.out` and the 1536 → 512 projection.
   - Write the derivation from the HF config and llama.cpp's
     `tools/mtmd/models/gemma4a.cpp` and `mtmd_audio_preprocessor_gemma4a`
     (read as references only) into `docs/engine/audio.md`.
   - Trace it against `tests/fixtures/embeddinggemma-audio.north/media-0.f32`.
   - Read the norm epsilon from `clip.audio.attention.layer_norm_epsilon`
     (1e-6 here, 1e-5 in E4B's mmproj).
4. **The encoder on Metal.** Add `inference/src/audio/gemma4a_metal.zig`
   with existing kernels where they fit. Any new kernel (the depthwise
   conv, the chunked relative attention) is tested against the CPU in
   `test-metal`.
5. **Splicing and surfaces.**
   - Splice BOA 256000, the rows, EOA 258883, and bind `a.*` lazily.
   - `nuclis embed --audio F`. The API's `input_audio` takes wav and mp3
     as OpenAI's format names, plus aiff, flac and m4a.
   - Chat keeps refusing audio (`chat/wire.zig`).
6. **Checks.**
   - Add gates `embeddinggemma-audio-cpu` (`verify-cpu`) and
     `embeddinggemma-audio-metal` (`verify`).
   - Cross-modal sanity: a spoken sentence's vector ranks its own text
     above the other fixture texts, recorded, not gated.
7. **Documents.** Add `docs/engine/audio.md` (new; listed in
   `docs/README.md`) and an architecture map line in
   `docs/architecture.md`.

**Gates.** `zig build test`, `zig build test-metal`, `make verify-auto`,
and `make verify`.

## Session 8. Acceptance and close

1. **The quantization record.** Over the fixture set, measure the cosine
   for Q8_0 against BF16 (ours, Metal), Q8_0 against sentence-transformers
   f32, and BF16 against sentence-transformers f32: the minimum and mean
   per modality.
   - Also run a retrieval check: the `docs/` paragraphs as the corpus,
     about 20 hand-written queries, top-5 agreement between Q8_0, BF16 and
     f32.
   - Write a dated, revision-cited table into the model document. If Q8_0
     loses measurably, raise it with the user before closing.
   - **Activation ranges** (the f16 question, decided 2026-10-09). Over
     the fixture set, record the largest |value| at each matmul's input
     (per stage, the worst case and layer), from the CPU runtime's
     observer or a temporary probe. Write a short table and its reading
     into `docs/models/embeddinggemma.md` § The f16 hazard: whether any
     input exceeds f16's 65504, or comes within 2⁸ of it. This is a fact
     for a later decision, not a change: the plan stays F32.
2. **The rates.** Text from Session 3, plus one image and 10 s of audio,
   end to end through `nuclis serve`. Record them in the model document
   and in `docs/guide/api.md` § Measured rates.
3. **Tiers.** Run `make verify`, and `make verify-cpu` (the unit brings up
   a family). `make verify-long` is needed only if a shared attention path
   changed; Sessions 2 and 3 are designed so none does.
4. **Documents.** Update the README's model list (and `make site-check` if
   a mirrored table changes) and `docs/architecture.md` (the embedding
   kind). Remove stale lines (`chat/wire.zig`'s audio note stays true).
5. **Close.**
   - Delete this unit's sections.
   - Write *Where we are* for the next theme: the personal index, with
     video as its own unit.
   - Push `embedding-gemma-2` and open the pull request
     `feat: EmbeddingGemma 2, text, image and audio embeddings` with the
     record (`.github/pull_request_template.md`).
