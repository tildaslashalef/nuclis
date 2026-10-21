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

Theme: speculation, on the branch `saluki-mtp-companion` (no pushes or
pull requests until the user says).

- Qwen's MTP companion binds on Saluki; its verdict is off, because
  Saluki's verify costs 2.3× a step at any row count
  (`docs/engine/speculative-decoding.md` § The Qwen MTP companion).
- The user chose DFlash2 on the Qwen3.8 Q4_K_M next (2026-10-21), done in
  this session.

| Unit | State |
| --- | --- |
| DFlash2 on the Qwen3.8 Q4_K_M | implemented on both executors and gated; the matrix against MTP is running |

## DFlash2 on the Qwen3.8 Q4_K_M

Base: `c7bd570`

**Delivered** (2026-10-21; the record is in
`docs/engine/speculative-decoding.md` § The Qwen DFlash2 drafter):

- `dflash2.zig` (the binding) and `qwen35.attachDraft`;
- the CPU reference (`dflash2_runtime.zig`) and the Metal `Flash`, with
  `nu_dflash_conv` and a Q8_0 fragment tile;
- the `--flash-trace` check, `tests/fixtures/qwen38-dflash2`, and four
  gates.

**Remaining:**

- the speculative matrix, with MTP as the same-day baseline
  (`.zig-cache/dflash2/`), then the verdict and the documents;
- `make verify` and `make verify-cpu`.

**Facts read 2026-10-21.**

- **Checkpoint.** `z-lab/Qwen3.8-27B-DFlash2` (mirror of
  `incoai/Qwen3.8-27B-DFlash2`, Apache-2.0) ships safetensors only (BF16,
  3,848,817,896 B).
- **GGUFs.** `z-lab/Qwen3.8-27B-DFlash2-GGUF` at `2d9571f8` ships BF16
  (3,860,293,216 B), Q8_0 (2,056,414,816 B), and Q4_K_M
  (1,143,006,816 B). They come from llama.cpp PR #27342's converter, the
  only definition of the GGUF keys. Nothing is downloaded yet.
- **Architecture** (`config.json`; the math from `z-lab/dflash`
  `dflash/model.py` at `07ebd93`, MIT, read as the format's only
  definition):
  - **Layers.** 5 Qwen3 layers: hidden 5120, FFN 17408, 32 query and 8 KV
    heads of 128, q/k RMS norms, RoPE θ 1e7 over the whole head,
    non-causal inside the block, a sliding window of 2048 both ways,
    eps 1e-6.
  - **Context.** The residuals after target layers 5, 19, 33, 47 and 61
    (`hidden_states[id + 1]`) are concatenated (5 × 5120), then
    `fc` → `hidden_norm`. The result enters each layer only as attention
    keys and values (`k_proj`/`v_proj` of the context rows), cached;
    queries come from the block rows.
  - **The block.** 8 rows `[anchor, mask × 7]` (mask id 248070), embedded
    with the target's `token_embd`. The final `norm` feeds the target's
    `output` head, with no logit scale or softcap for Qwen. Rows 1–7 are
    the drafts. The block's own cache rows are cropped after every draft.
  - **New over the Muse DFlash in `dflash.zig`: two dynamic convolutions
    per layer.** One wraps attention and one the MLP. Each has
    `base_kernel [2][2][5120]` and `kernel_projection` 5120 → 2·2·320
    (groups of 16). `prepare` runs on the normed sublayer input `x`:
    `y[t] = Σ_{o∈{0,1}} (base[0][o] + dyn₀[t][o][group]) ⊙ x[t−o]`,
    with `x[−1] = 0` at the block's first row. `finish` runs the same on
    the sublayer output with `base[1]`, and with `dyn₁` projected from
    the same normed `x`.
  - **New: the candidate selector.** Per position it takes the top 16
    logits (unary) and projects the hidden through `hidden_projection`
    5120 → 256. Each candidate k scores
    `unary_k + Σ_r pred[prev][r] · h[r] · succ[cand_k][r]`, where `pred`
    and `succ` are 248,320 × 256 codebooks. The greedy chain starts from
    the anchor and takes the argmax at each position.
- **The contract fits.** `propose(k)` is one block forward plus the
  selector chain, for k ≤ 7. `commit` injects the 5 captured residuals
  per committed token, as the Muse drafter does. nuclis's acceptance
  draws from the target alone, so the selector's sampled q is not needed.
- **Needs.** A capture of the 5 residuals in the Qwen executors (CPU
  `qwen35_runtime` and Metal `qwen35_metal`; today only Muse's plan
  captures). A `dflash2` binding, with the GGUF keys read from the
  downloaded header. The conv and selector on both executors.
- **Validation.** No external oracle. Unit tests pin the conv and
  selector against hand-computed values. Metal is checked against the
  CPU rows (a pinned fixture). Greedy speculative output must equal
  plain. Acceptance (`--draft-stats`) is compared with the MTP block's.
- **Measure.** `make spec-matrix ARGS='--model qwen38 --contexts
  code,512,4096,16384,32639 --drafts 3-7'` against the MTP record, with
  the same verdict rule.

## Deferred

Agreed and not scheduled; when one is picked up, it becomes a unit above.

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
