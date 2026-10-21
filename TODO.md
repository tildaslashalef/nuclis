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

Theme: **speculative decoding for Saluki 27B**, the deferred decode goal
(≥ 18 tok/s at 32,639) attacked from the drafter side instead of the
matvecs. Saluki decodes 16.96 tok/s at 512 and 14.97 at 32,639 with
speculation off: its file has no prediction block. Qwen3.8's own block
reaches 1.50× at 512 and 1.26× at 32,639 on the Q4_K_M; the same ratios
on Saluki would be about 25 and 18.9 tok/s. Two numbers are unknown and
this unit measures them: what an 8-row verify batch costs on the 2-bit
encodings (their matvecs are instruction-bound, not bandwidth-bound), and
how often Saluki accepts a block trained on the full-precision model's
hidden states.

DFlash2 (z-lab's target-trained block drafter; `dflash.zig` refuses its
keys today) is not part of this unit. It becomes the next unit only if
this one shows acceptance, not verify cost, is what limits Saluki.

| Unit | Branch | State |
| --- | --- | --- |
| Saluki speculates with Qwen's MTP companion | `saluki-mtp-companion` | design written, next: code |

## Saluki speculates with Qwen's MTP companion

Base: `f227492`

**Facts read 2026-10-21.**

- `~/.nuclis/models/unsloth/Qwen3.8-27B-GGUF/MTP/mtp-Qwen3.8-27B-Q4_0.gguf`
  (1,369,590,656 B, SHA-256 `50d9ce5a…`, catalogue role `mtp`, already
  pulled; gates.json model `qwen38_mtp`). Header: `qwen35`, `block_count`
  65, `nextn_predict_layers` 1, every other `qwen35.*` dimension equal to
  the pinned Q4_K_M's and Saluki's; the vocabulary is the same 248,320
  tokens. 18 tensors: the **whole `blk.64` block** (`nextn.eh_proj`,
  `ffn_*` Q4_K; `attn_q`, `attn_k`, `attn_v`, `attn_output` Q6_K; norms
  F32) plus `token_embd` and `output` (Q3_K) and `output_norm`. The block
  in the Q4_K_M is the same weights at higher precision (`attn_k`/`attn_v`
  Q8_0, `ffn_*` and `eh_proj` Q6_K).
  `docs/engine/speculative-decoding.md` § The Qwen3.8 draft head says
  the file "adds only lower-quantized copies" of the embedding and head.
  That is wrong; correct it in this unit.
- Saluki (`docs/models/saluki.md` § Metadata) is the Qwen3.8 shape with
  `block_count` 64 and no `nextn` key, unrotated. Its binding has
  `draft == null`.
- `engine.Engine.open`: `qwen35` declares `embedded_draft = true` and no
  `bindDraft`. `DraftRequest.preferred` on such a family ignores the
  companion path and asks the executor for a drafter. On Saluki that is
  `NoDraftBlock` from `qwen35_runtime.Runtime.init` /
  `qwen35_metal.Plan.init`, which is why the user's `saluki-27b` entry
  keeps `speculative: false`. `.file` on `qwen35` is
  `DraftSourceUnsupported`.
- Both Qwen executors read every weight through one `weights.View`. In
  the runtime: `Runtime.mm` and `view.vector` in `init`. In the plan:
  `Plan.weight`, `Plan.mm`, and `Plan.constant`. Metal wraps one buffer
  per tensor (`Plan.weight`), so a second mapping needs no buffer change.
  DFlash's binding carries its own `view` for the same reason
  (`muse_glimmer.bindDraft`).
- An entry's `mtp` is resolved by `src/engine.zig` `draftPath`: absolute,
  or relative to the model file's directory. `spec-matrix.py` and
  `speed.py` reach a companion only through a gates.json model's
  `entry`.

**Changes.**

1. `inference/src/models/qwen35.zig`
   - `DraftBlock` gains `source: ?DraftSource = null`. `DraftSource` is
     `{ view: weights.View, tensors: []const gguf.Tensor }`; null means
     the main file.
   - New `pub fn bindDraft(alloc, doc, view, target_view, target:
     *const Binding) DraftError!DraftBlock`, wired into `family` with
     `draft_architecture = architecture`:
     - Validation: `validateMetadata(doc)` must report `auxiliary` and a
       shape equal to `target.shape`. A rotated target
       (`target.rotation != null`) is `UnsupportedConfiguration`, because
       the companion's block is in the plain basis.
     - Binding: a `Binder` over the companion's tensors takes only
       `blk.<target.shape.layers>.*`, exactly the tensors `bind` takes
       for the embedded block. Every other tensor in the companion
       (`token_embd`, `output`, `output_norm`) is deliberately left
       unbound: the drafter uses the target's embedding and head, as the
       embedded block does. Each bound tensor passes `executableEncoding`.
   - Unit tests on a new inventory fixture,
     `fixtures/qwen35-mtp-companion.json` (`scripts/gguf-inventory.py`,
     no weights):
     - it binds onto a Saluki-shaped target (the 27B fixture with
       `block_count` 64 and no `nextn`);
     - it refuses a shape mismatch, a companion without `nextn`, and a
       rotated target;
     - the bound tensors point into the companion's document.
2. `inference/src/engine.zig` `Engine.open`
   - `.preferred` on an embedded family uses the companion only when
     `binding.draft == null`, so the Q4_K_M keeps its own block. `.file`
     replaces an embedded block.
   - `want_draft` is true when the binding ends with a draft block. A
     Bonsai file with speculation on is therefore `DraftSourceMissing`,
     the typed refusal the rule requires, not `NoDraftBlock` from init.
   - The companion mapping lives in `draft_mapped` as for Muse.
   - Tests: the decision table, on fixtures.
3. Draft-block weight reads in `qwen35_runtime.zig` and
   `qwen35_metal.zig`
   - The draft constants in `init` read `source.view` when a source is
     set.
   - Matrix lookups (`Runtime.mm` and its batched sibling; `Plan.weight`,
     which `Plan.mm` and the matmul paths use) choose the view with a
     `viewOf(tensor)` helper: `source.view` when the pointer lies inside
     `source.tensors`, else the main view.
   - Nothing else in the forward changes.
4. `inference/generation-check.zig`: `--draft-model` is honored on
   `qwen35` for `--draft-trace`, `--draft-stats`, and
   `--speculative-check`. It binds the companion with `qwen35.bindDraft`
   into the binding, replacing an embedded block.
5. `gates.json`
   - The `saluki` model gains `"mtp": "qwen38_mtp"` and `"entry":
     "saluki-27b"`.
   - New gates:

     | Gate | Tier | What it checks |
     | --- | --- | --- |
     | `qwen38-mtp-companion-trace-{metal,cpu}` | `verify`, `verify-cpu` | The Q4_K_M with `--draft-model {mtp}` against `fixtures/qwen35-mtp`: the cross-encoding check of the block. Bounds are set from the first measurement and recorded; expect around 1e-2 relative RMS from Q4_K against Q6_K. |
     | `saluki-draft-trace-cpu` | `verify-cpu` | Writes the rows that are pinned as `fixtures/saluki-mtp/` (the CPU reference as fixture, as `saluki-trace-*` do). |
     | `saluki-draft-trace-metal` | `verify` | Against `fixtures/saluki-mtp/` within Saluki's `saluki-trace-f16` bounds. |
     | `saluki-speculative-{metal,cpu}` | `verify`, `verify-cpu` | `--speculative-check --draft-model {mtp}`: greedy speculative output equals plain. |

6. The user's `saluki-27b` entry (their file; set only on their word):
   `nuclis config set models.saluki-27b.mtp
   ../../unsloth/Qwen3.8-27B-GGUF/MTP/mtp-Qwen3.8-27B-Q4_0.gguf`.

**Measurements** (Metal, F16 cache, cooled chip, written into the docs
as taken).

1. Acceptance, `test-generation -- <model> --metal --draft-stats [--draft-model {mtp}]`
   on three pairs: the Q4_K_M with its embedded block, the Q4_K_M with
   the companion, and Saluki with the companion. The first two separate
   what the companion's lower precision costs from what Saluki's 2-bit
   hidden costs.
2. The speculative matrix: `make spec-matrix ARGS='--model saluki
   --contexts 512,4096,16384,32639,code --drafts 3-7 --cooldown 60'`,
   greedy and instruct. It gives E, C, C / 50E, and the off/on rates.
3. The verdict follows the one rule of `docs/benchmarks/history.md`
   § The re-priced speculative verdicts: on at draft L when no prose cell
   at L is below 0.98× and the prose geometric mean is ≥ 1.10×.
   The goal check: 18 tok/s at 32,639 needs C ≤ 55.6·E ms per batch.
4. **Stop rule.** Stop if Saluki's 8-row verify at 512 costs more than
   2.5× its single-row step (about 59 ms). Then the 2-bit multi-row
   matvec is the gap and no drafter fixes it. Record that, leave
   speculation off, and the deferred decode goal stays with the
   matvecs.

**Documents.**

- `docs/engine/speculative-decoding.md`: a new section, "The Qwen MTP
  companion", plus the correction above.
- `docs/models/saluki.md`: a Speculation section with the numbers and
  the verdict.
- `docs/benchmarks/history.md`: the record.
- `docs/spec.md` § Speculative decoding and `docs/architecture.md`: the
  draft-sources row.
- `docs/guide/configuration.md`: an entry's `mtp` may name a companion
  in another repository's directory.

**Tiers.**

- `make verify-auto` while iterating.
- `make verify` once.
- `make verify-cpu`, because the unit brings up a draft source.
- Not `verify-long`: no attention, KV cache, or schedule changes.

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
