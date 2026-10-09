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
`5e69b80` and last merged with `main` at `b0297c2`. **The user's rule for this unit: commit on the branch, do not
push and do not open the pull request until the whole unit is
implemented** (this overrides the usual "push and draft a pull request at
the end of each session"). Nothing is implemented yet; the planning session
only read the facts recorded below. Next: **Session 1, the facts and the
oracles**.

| Session | What |
| --- | --- |
| 1. Facts and oracles | Pull and pin the files, `docs/models/embeddinggemma.md`, inventory fixtures, the second llama.cpp checkout and the sentence-transformers oracle, recorded traces and vectors |
| 2. Text encoder on the CPU | `gemma-embedding2` adapter, the bidirectional KV-free forward, pooling, projection, normalization, Matryoshka; `Embedder` in `inference/src/embed.zig` |
| 3. Text encoder on Metal | A grouped-query, 512-wide, online-softmax segments attention kernel; packed batches; F32 activations throughout; measured rates |
| 4. `nuclis embed` and the catalogue | `ModelKind.embedding`, the catalogue table, the shared wire types in `src/embedding/`, tasks and titles, `model ls` |
| 5. `POST /v1/embeddings` | The service, its batcher and pool, `Kind.embedding` in the memory budget, `GET /v1/models` fields, the API guide and spec |
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
  document in Session 1:
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
- **The personal index is a later theme.** Searching the user's own docs,
  audio and images is planned after this unit merges. This unit rewords
  spec §10 ("an embedding index is ruled out") to name it as a separate
  theme, not ruled out.
- **One branch, no push until complete** (see *Where we are*).

**Facts read during planning (2026-10-08), to be recorded in Session 1.**

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

Oracles:
- Our pin `7620399` does not know `gemma-embedding2`.
- Upstream llama.cpp added it, with vision and audio, in `4fbc76dec`
  (#30054, 2026-10-06). The upstream master read on 2026-10-08 is
  `06cad0b9e`, and it also carries typed multimodal input for
  `/v1/embeddings` (#29556).

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

## Session 1. Facts and oracles

Base: `b0297c2`.

**Why.** Every later session compares against recorded numbers. This one
pins the files, writes the model document, and records the oracle outputs
before any engine code.

1. **Pull and pin.** Download the three files above into
   `~/.nuclis/models/unsloth/embeddinggemma-2-GGUF/` with their provenance
   sidecars. Use `huggingface/`'s downloader, either with a temporary
   catalogue row or `make hf-downloader`. Check every sha256 against the
   table.
2. **`docs/models/embeddinggemma.md`.** Follow `docs/models/README.md`
   §§ 0–1:
   - Artifacts (repo, revision, sizes, digests, and the quantization
     decision with its reasons).
   - Metadata (`gemma-embedding2.*`, `clip.*`).
   - Tensors.
   - The forward pass as written in *The theme*.
   - Tokenizer, and prompt prefixes: the full table from Google's
     `config_sentence_transformers.json` and the card's best-practices
     section, `title: {title} | text: {content}`, `title: none`.
   - Matryoshka (128/256/512/768, renormalize).
   - The f16 hazard.
   - The input contract per modality, and the oracle status.

   Add its line to `docs/README.md` and `docs/models/README.md`.
3. **Inventory fixtures.** Run `scripts/gguf-inventory.py` on the Q8_0 file
   to produce `inference/src/models/fixtures/embeddinggemma-2-q8_0.json`,
   and on the mmproj to produce
   `inference/src/vision/fixtures/embeddinggemma-2-mmproj.json`.
4. **The second llama.cpp checkout.** Create `.reference/llama.cpp-embed`
   at upstream `06cad0b9e` (or the newest master that still holds
   `src/models/gemma-embedding2.cpp`), built with the recipe of
   `docs/development.md`. The main pin `7620399` does not move.
   - Record the checkout beside the Prism one in `docs/development.md`
     (the bullet naming the reference checkouts) and in
     `docs/benchmarks/llama-cpp.md` § The second oracle.
   - Check it loads the Q8_0 and BF16 files with `--embeddings --pooling
     mean` and returns unit vectors.
5. **The trace and vector driver.** Add `scripts/reference-embedding.cpp`,
   modeled on `scripts/reference-generation.cpp`'s eval callback, built
   against the new checkout.
   - For a token list it dumps `inp_scaled`, `inp_per_layer`, `l_out`
     per layer, `result_norm`, `result_embd`, and the pooled, normalized
     vector.
   - For an image or audio file it goes through `mtmd` and also dumps the
     projector rows.
   - Fixtures go in `tests/fixtures/embeddinggemma-<case>/`, compared by
     `scripts/compare-generation.py` (2e-3 max abs, 1e-4 relative RMS),
     or by a small `compare-embedding.py` if the shapes do not fit.
6. **The semantic oracle.** Add `scripts/embedding-reference.py`, run in a
   venv under `.reference/venv-embed` (sentence-transformers ≥ 6.1,
   transformers 5.18.dev as the card pins). It embeds the fixture set
   with `google/embeddinggemma-2` at the pinned revision in **float32**
   and writes `tests/fixtures/embeddinggemma-vectors/st-f32.json`.
   This is the end-to-end truth for the input processing too (prefixes,
   image resize, mel features).
   - Add it to `make lint-py`.
   - It needs a network and about 1.5 GB, so it is a recorded fixture,
     not a test.
7. **The fixture set.** It must be small and attributable, our own
   material only:
   - About 24 texts: sentences written for the purpose, three paragraphs
     of `docs/`, one Zig function from `inference/`, and some multilingual
     lines. Each is rendered raw, as `search result` query, as `title:
     none` document, and as `code retrieval` query.
   - One long text of about 3000 tokens, to cross the 512 window, and one
     near 8192 tokens.
   - Images: `docs/branding/nuclis.png` and one captured site screenshot.
   - Audio: `say -o` clips (AIFF, then WAV) of two sentences from the text
     set, plus a generated 1 kHz tone, for a cross-modal check.
   - One interleaved case: text, `<|image|>`, text.

**Gates.** `make docs-check`, `make lint-py`, the fixtures committed, and
each oracle's vectors unit length.

## Session 2. Text encoder on the CPU

**Why.** The numerical contract, before any kernel. It is the CPU tier's
reference for this family.

1. **The binding.** Add `inference/src/models/embeddinggemma.zig` with
   `Config` and `bind`:
   - Validate `gemma-embedding2.*` and refuse unknown keys, the way
     `gemma4.zig`'s `validateMetadata` does.
   - Layer kinds come from `attention.sliding_window_pattern`. Per-layer
     KV heads and widths come from `head_count_kv`, `key_length` and
     `key_length_swa`.
   - Window half-width = `sliding_window / 2`.
   - `embedding_length_out` gives the output width, and `pooling_type`
     must be 1 (mean).
   - Executable encodings are **F32, BF16, Q8_0 only**. Anything else is
     `UnsupportedEncoding`, which keeps the half-operand K-quant tiles out
     of this family by construction.
   - Refuse `causal = true`.
   - Share helpers with `gemma4.zig` where one already exists (norm and
     GELU), but do not widen `gemma4.zig`'s accepted metadata.
   - It is **not** an entry in the generation adapter `table`
     (`models/root.zig`). It decodes nothing, like Laya.
2. **The forward.** Add `inference/src/models/embeddinggemma_runtime.zig`
   `Runtime.encode(rows) -> [rows][512]`:
   - One full-sequence pass in F32, with no KV cache.
   - Attention is per query over the visible keys: symmetric |i − j| ≤ 512
     on sliding layers, everything on global layers, with GQA.
   - Inputs are row sources: a token id (gathered, then × √512) or a
     projector row (taken as is, not scaled).
   - `ple` comes from `P · x0 / √512` for every row, image rows included.
3. **Pooling and the vector.** Add `inference/src/embed.zig`:
   - `Embedder` has `open(dir or files)`, `prepare(input) -> Prepared`
     (tokens and modality spans, bounded by `max_tokens` 8192),
     `embed(prepared) -> Vector`, and later `embedJobs` for batches.
   - The vector is computed as `output_norm` → `output.weight` → mean over
     every row → L2 normalize.
   - `truncate(vector, dims)` keeps the first `dims` ∈ {768, 512, 256,
     128} and renormalizes. Other widths are `error.UnsupportedDimensions`.
   - Ownership: the vector is caller-owned, and `Prepared` borrows nothing
     from the request.
4. **Tokenizer.** Add EOS when `add_eos_token` is true for this family
   (today `add_eos` is read nowhere), giving BOS + text + EOS. Check
   against the oracle's token ids.
5. **The check tool.** Add `inference/embeddinggemma-check.zig`, modeled
   on `inference/laya-check.zig`, with a build step.
   - It compares the per-layer traces of Session 1 and the pooled vectors
     against llama.cpp: cosine ≥ 0.9999 on Q8_0, and the trace bounds of
     `compare-generation.py`.
   - Add gates `embeddinggemma-trace-cpu` and `embeddinggemma-vectors-cpu`
     to `gates.json` (tier `verify-cpu`), with the paths that select them.
6. **Tests.** Unit tests for:
   - The symmetric mask edges: 512 visible, 513 not.
   - Matryoshka truncation and renormalization.
   - Refusal of a K-quant matrix, of `causal = true`, and of input over
     8192 tokens.
   - The inventory fixture binding with no weights.

**Gates.** `zig build test`, `make verify-auto`, and the new CPU gates.

## Session 3. Text encoder on Metal

**Why.** Indexing a corpus is many passes. Metal is the path users run;
the CPU stays the oracle.

1. **The attention kernel.** Add `Backend.attentionSegmentsGrouped` and
   `nu_attention_segments_grouped` in `kernels.metal`:
   - F32 throughout, online softmax over key blocks, so there is no
     4096-row score buffer.
   - KV heads ≤ query heads (GQA), head width ≤ 512.
   - The per-row `[begin, end)` segment bounds and the symmetric `window`
     of `attentionSegments`.
   - It is tested against the CPU attention on random data at widths
     256 and 512, with 1 and 2 KV heads, 1 to 8192 rows, and packed
     segments, in `zig build test-metal`.
   - `attentionSegments` itself is not changed, so Laya is untouched.
2. **The plan.** Add `inference/src/models/embeddinggemma_metal.zig`
   `Plan`, modeled on `laya_metal.zig`'s packed batches: several inputs per
   pass with segment bounds, mean pooling per segment, and the projection
   and normalization.
   - Matrices go through the generic `nu_matmul`/matvec for Q8_0, BF16
     and F32.
   - Residual, activations and attention are F32. No half anywhere: the
     card's f16 warning.
   - `Embedder` uses the Metal plan when `--backend metal`, with
     `batchRows` as Laya's.
3. **Gates.** Add `embeddinggemma-vectors-metal` (tier `verify`, the fast
   Metal tier) with cosine ≥ 0.9999 against the CPU vectors and the
   llama.cpp vectors, covering the long-text and near-8192 cases.
4. **Rates.** Measure and record in the model document:
   - Inputs per second for 64 inputs of 256 tokens.
   - The latency of one 512-token input and one 8192-token input.
   - The same on the llama.cpp checkout.

   Record the hardware, build, file and methodology
   (`docs/benchmarks/README.md`).

**Gates.** `zig build test-metal`, `make verify-auto`, and `make verify`
once in this session.

## Session 4. `nuclis embed` and the catalogue

**Why.** The first surface, and the wire types the API will return
byte-for-byte, as `decide --json` is to `/v1/decisions`.

1. **Kind and catalogue.**
   - Add `embedding` to `src/catalog.zig` `ModelKind`.
   - Add an `embedding_entries` table with `EmbeddingEntry` (name, title,
     repo, file, revision, sha256, size, architecture, quantization,
     `mmproj: ?Artifact`, `dimensions`, `modalities`).
   - The entry is `embeddinggemma-2`, from the facts table. Its mmproj is
     pulled with `--with mmproj`; text works without it.
   - Registry `ModelEntry.kind = .embedding`.
   - `model ls` marks these rows `(nuclis embed)`, and the listing schema
     carries the kind.
   - Text commands (`generate`, `chat`, `decide`) refuse an embedding
     model by name, as they refuse decision models.
   - Add `embed.model` to the config (default `embeddinggemma-2`).
2. **Wire types.** Add `src/embedding/request.zig` and `response.zig`,
   one source for the human and JSON output.
   - Request: `inputs: []Input`. An input is text or an ordered list of
     parts (`text`, `image`, `audio`). Plus `task: ?Task`, `title`,
     `dimensions` (default 768), and `truncate` (default false).
   - `Task` is `search_query`, `document`, `question_answering`,
     `fact_checking`, `code_retrieval`, `classification`, `clustering`,
     `similarity`. Each renders Google's prefix onto **text parts only**:
     a query task gives `task: … | query: `, `document` gives
     `title: {title|none} | text: `.
   - Response: `vectors` (index, values), `dimensions`, `tokens` per
     input and in total, and `space`.
     - `space` is `<entry>@<sha256[0:12]>/<dims>`, so an index can
       detect vectors that are not comparable (another file or another
       width).
   - Over-long input is `input_too_long` with the count, unless
     `truncate`. Truncation is reported in the response, never silent.
3. **The command.** Add `src/embed.zig` and dispatch it from `cli.zig`:
   `nuclis embed [text…] [--task T] [--title S] [--dimensions N] [--image
   F] [--audio F] [--input-file inputs.jsonl] [--json] [--model M]
   [--backend B]`.
   - Human output: per input, the tokens, the dimension, the norm and the
     first values. With two or more inputs, a cosine matrix.
   - `--json` is the response type.
   - `--image` and `--audio` stay refused with a typed message until
     Sessions 6 and 7.
   - Add help and completion.
4. **Documents.**
   - `docs/spec.md`: §3, the kind, its catalogue table and registry
     refusal; a new §5 "The embedding path" contract, after "The decision
     path"; the §6 `embed` row; the §10 rewording (the personal index is
     a later theme, and video is deferred).
   - `docs/models/catalogue.md` § The catalogue.
   - `docs/guide/getting-started.md`, a short "Embeddings" paragraph.

**Gates.** `zig build test`, `make verify-auto`, `make docs-check`, and
`./zig-out/bin/nuclis model pull embeddinggemma-2 --with mmproj` followed
by `./zig-out/bin/nuclis embed --task search_query "…" ` against the
oracle vector.

## Session 5. `POST /v1/embeddings`

**Why.** The API the later index theme and outside clients build on, in
the shape of the decisions and chat services.

1. **The service.** Add `src/api/embeddings/service.zig`, registered in
   `src/api/root.zig` `Server.register`, for `POST /v1/embeddings` with a
   32 MiB body.
   - The request follows OpenAI's shape, extended:
     - `model` resolves to a registry or catalogue `embedding` name, and
       defaults to `embed.model`.
     - `input` is a string, an array of strings, token arrays, or an array
       of **content-part arrays**: chat's `text` / `image_url` (data URLs
       only) / `input_audio` (`{data, format}`). One input gives one
       vector.
     - `dimensions` ∈ {768, 512, 256, 128}, or 400 `unsupported_feature`
       with `param`.
     - `encoding_format` `float` | `base64` (little-endian f32; the
       OpenAI Python SDK sends `base64` by default).
     - `user` is accepted and ignored.
     - Top-level extensions `task`, `title` and `truncate`, named as in
       the CLI.
   - The response is OpenAI's `{object:"list", data:[{object:"embedding",
     index, embedding}], model, usage:{prompt_tokens, total_tokens}}`
     plus `nuclis:{space, dimensions, task, truncated}`.
   - Status codes follow chat (400, 404 `model_not_found`, 529, 503).
   - `not_an_embedding_model` names the kind. Generation routes answer
     `not_a_language_model` for an embedding name.
2. **Batching and the pool.**
   - `src/api/embeddings/batcher.zig` takes every waiting embedding job
     for the oldest job's model, in arrival order, while the rows fit
     one pass (2048 rows; a single longer input runs alone). It runs on
     `gpu.zig`'s one executor, between a generation's steps as decisions
     do.
   - `pool.zig`, with capacity 1.
   - `src/api/memory.zig` `Kind` gains `embedding`. A model counts its
     files: the mmproj halves are counted only once bound (Session 6/7
     bind them lazily).
3. **`GET /v1/models`.** Embedding rows carry `kind:"embedding"`, `name`,
   `architecture`, `quantization`, `size_bytes`, `present`, `loaded`,
   `default`, `dimensions:[768,512,256,128]`, `modalities` (text, plus
   image/audio when the mmproj is present), `max_tokens:8192`, `tasks`,
   `repo` and `revision`.
4. **Checks.**
   - Extend `scripts/api-check.py` with embeddings: float and base64 are
     identical, a batch of 3 equals 3 single calls bit for bit ("batching
     changes timing, never answers"), and the dimensions and errors
     behave as specified.
   - Check the OpenAI Python SDK's `client.embeddings.create` once by
     hand.
5. **Documents.** Add a `docs/guide/api.md` `## Embeddings` section
   (request, response, refusals, batching, memory), the `/v1/models`
   fields and § Measured rates, and update the spec §6 `serve` row.

**Gates.** `zig build test`, `make verify-auto`, `make docs-check`, and
`api-check.py` against `./zig-out/bin/nuclis serve`.

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
   `max_soft_tokens`), not chat's 1120. Keep the
   `preprocess.smartSize` letterbox. `--image-tokens 70..1120` on the CLI
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
     against the Session 1 image and interleaved fixtures: projector rows
     by trace bounds, vectors by cosine against llama.cpp, and against
     sentence-transformers f32 recorded as a measured number.
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
   - Test it against `Gemma4AudioFeatureExtractor` features recorded in
     Session 1 (the HF processor is the oracle for the features).
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
   - Trace it against `reference-embedding.cpp`'s audio dump.
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
