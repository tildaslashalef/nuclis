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

MODL-27 (Gemma 4 E4B QAT, end to end) was taken on 2026-09-26, the user's
call. The three files are pulled and registered as `gemma-4-e4b-qat` in
`~/.nuclis/nuclis.json` (commit `8c5a9e4fd5482e2be20fe0bf013b4c262a8f4265`
of `unsloth/gemma-4-E4B-it-qat-GGUF`). The user's working mode for this
unit: **one unit, implement end to end first, then test once; no commit
until every check below passes.**

**2026-09-26 session: implementation 1–7 done and committed without the
check pass, at the user's call; the unit stays open until checks 4–6 pass
(no log entry yet).** Measured so far
(ReleaseFast/ReleaseSafe, M4 Pro, write these into the docs on close):
- traces vs the reference on `<bos>Hello,` (fixtures in
  `tests/fixtures/gemma4-e4b-hello-comma/`): CPU max abs 3.1e-5 / rel RMS
  1.8e-6; Metal F32 6.1e-5 / 2.5e-6; Metal F16 1.3e-2 / 4.3e-4; top-5 equal.
- `test-generation --metal`: passed; chunk 64 half tiles 3.1e-1 / 1.86e-2
  (bound 6e-1 / 2e-2), F32 tiles 2.4e-3 / 1.4e-4.
- draft trace (`fixtures/gemma4-mtp-e4b/`): CPU and Metal ≤ 6.9e-5, greedy
  26352 236764 equal, propose 236764; fixed a pre-existing arena leak in
  `gemma4_runtime.zig` `Runtime.init` (the head allocated after the arena moved).
- vision on Metal (`vision/fixtures/gemma4v-e4b-synthetic/`): rows 3.6e-2 /
  7.5e-3, logits 6.1e-3, 8 greedy tokens equal. The E4B's image span is
  **causal** (reference `mtmd_decode_use_non_causal`), `Config.bidirectional_images`.
- vocabulary check (`gemma4_e` profile, fixture `profiles/fixtures/gemma4_e-text.json`
  captured from the reference server): passed, 34 sequences.
- perplexity vs llama-perplexity: 512×8 36.0558 vs 36.0595 (−0.010 %);
  4096×4 23.1732 vs 23.1780 (−0.021 %).
- first look: decode 52.4 tok/s; `--speculative on` 69.9 tok/s (1.34×,
  188/230 accepted, code prompt), greedy output identical; kept off (< 1.5×).
- user data: the user deleted `~/.nuclis/nuclis.json` to regenerate it with
  `config init --discover` (the pre-MODL-27 copy is
  `~/.nuclis/nuclis.json.bak-modl27`); the E4B's template selects `gemma4_e`.

**Where to pick up:** `make check` passes format and the 563 unit tests,
but `metal-check` failed once with `error: ProfileExceedsCommandBuffer`
(it passed earlier in the session with the new `clamp` fixture). Find out
whether it is flaky or caused by this unit's diff, then run checks 4–6
below (the user agreed: `make verify`, `make verify-long`, and the Gemma
CPU-tier gates only, `make gate NAME='gemma4*-cpu'`; the non-Gemma CPU gates
are skipped), then close (check 8).

## Order

| Unit | Title | Sessions |
| --- | --- | --- |
| MODL-27 | Gemma 4 E4B QAT: text, per-layer embeddings, shared KV, vision, draft head, profile, catalogue | several; closes once |
| APPS-17 | The catalogue at five entries; registry entries ordered by name | one |
| REPO-17 | README rewritten: grouped models, `--discover`, an agent recording | one |

Decided 2026-09-26 (user): APPS-17 and REPO-17 go ahead while MODL-27's
checks are pending; the README may list the E4B.

## APPS-17 — the catalogue at five entries; registry ordered by name

- `src/catalog.zig` `entries`: keep `qwen3.8-27b`, `gemma-4-12b-qat`,
  `gemma-4-26b-a4b`, `gemma-4-e4b-qat`, `muse-glimmer-30b`; drop
  `gemma-4-12b` (K-quant) and `bonsai-2-27b`. Their files stay runnable:
  `model pull <owner/repo> --file <f>` then `config init --discover`
  (Bonsai's template is not pinned, so discovery forces `qwen38`; check its
  `…-mmproj-Q8_0.gguf` is found as the companion). Fix the catalogue tests
  and every doc/help example naming a dropped entry
  (`grep -rn "gemma-4-12b\b\|bonsai-2-27b"`); gates keep their files by
  path (`gates.json` `models.*.entry` is informational: update the text).
- Ordering: `config init` (catalogue registration), `config
  registerDiscovered`, and `model pull --register` write `models` sorted by
  name, and `--discover`'s report lists candidates by name. Unit tests for
  both orders.
- Check: `make check`; `rm` a scratch `NUCLIS_HOME`'s file, run `config
  init --discover --dry-run` and `config init --discover` on the real models
  dir, and read the result.

## REPO-17 — README rewritten

- Shorter README: pitch, recording, status; quick start (`make metal`,
  `config init --discover`, `model pull`, `agent`); supported models grouped
  (Qwen3.8, Gemma 4 ×3 with one line each: dense QAT, mixture of experts,
  per-layer embeddings + shared KV; Muse Glimmer) and one paragraph on
  anything else (pull by repository, `--discover`); one results table
  linking bench.md; docs, design rules, disclosure, licence, name trimmed.
- The recording: `brew install vhs` (authorized 2026-09-26); a
  `scripts/agent-demo.tape` driving `nuclis agent` with Qwen3.8-27B in
  `~/Code/playground` on a short task, sped up and labelled as sped up;
  output `docs/media/agent.gif` (target < 5 MB). First confirm the TUI
  renders in vhs's terminal.

## MODL-27 — Gemma 4 E4B QAT, end to end

**Artifacts** (under `~/.nuclis/models/unsloth/gemma-4-E4B-it-qat-GGUF/`,
revision `8c5a9e4f…`):

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf` | 4,215,695,776 | `df0fd4ee07072c607c29a0a1cb4f98918426cca12f45a2776bdd6ee6d09a4de3` |
| `mmproj-BF16.gguf` | 991,552,320 | `7c9bafa27f82d658eda805c1d82ef62bb0368e1ff75f64f77de58ad318beaaf9` |
| `MTP/mtp-gemma-4-E4B-it-Q4_0.gguf` | 59,678,016 | `423074e537504b4f9ec5eafed5c639fac82c96631626efccacdd3c4039b20605` |

Every weight matrix of the main file and head is Q4_0 (already executed),
so no kernel encoding work. Inventories: `scripts/gguf-inventory.py <file>
--out …` (committed as fixtures, below).

### Facts read from the files and the pinned reference (`7620399`)

Main file (`gemma4`, 42 blocks, 666 tensors, width 2560, FFN 10240, ctx
131072, template SHA `241c50d8…`):

- **Attention geometry:** 8 query heads (not 16); sliding window **512**;
  sliding layers 2 KV heads × 256, global layers 2 KV heads × 512;
  `head_count_kv` is a **scalar** 2 (the 12B/26B files carry an array).
  Pattern period 6, global at `il % 6 == 5` (35 sliding + 7 global).
- **Global layers have `attn_v`** (`[2560, 1024]`): V is its own projection,
  not K (`layer.value` non-null on global layers; the runtimes already
  branch on `layer.value`).
- **Shared KV:** `gemma4.attention.shared_kv_layers = 18` → layers 0–23
  own K/V; layers 24–41 have **no `attn_k`, `attn_v`, `attn_k_norm`**, still
  have Q, Q norm, O. A shared layer reads the cache of layer
  `24 − 2 = 22` if sliding and `24 − 1 = 23` if global
  (`llama-model.cpp:2615`, reuse callback). The reference still computes
  and rotates Q on shared layers.
- **Per-layer embeddings (PLE),** `embedding_length_per_layer_input = 256`
  (`src/models/gemma4.cpp`, `build_inp_per_layer`,
  `project_per_layer_inputs`, and the tail of the layer loop):
  - tensors: `per_layer_token_embd [10752, 262144]` Q4_0 (row = 42 × 256),
    `per_layer_model_proj [2560, 10752]` Q4_0, `per_layer_proj_norm [256]`;
    per layer `inp_gate [2560, 256]`, `proj [256, 2560]`, `post_norm [2560]`.
  - per token, once, before layer 0, with `x0 = embed(t) · √2560`:
    `sel = per_layer_token_embd[t] · √256` (42 × 256);
    `pr = (per_layer_model_proj · x0) · (1/√2560)`, then RMS-norm each
    256-slice with `per_layer_proj_norm`; `ple = (pr + sel) · (1/√2)`.
  - per layer `il`, after the FFN residual and **before** the layer output
    scale: `g = gelu(inp_gate · x)`; `g ⊙= ple[il]`;
    `x += rmsnorm(proj · g) ⊙ post_norm`; then `x *= layer_output_scale`.
  - image rows: `x0` is the projector row unscaled, and `sel` is
    **token 0's** row (padding), scaled by √256 as usual.
- Same as the 12B: rope factors (`rope_freqs [256]`) on global layers,
  base 1e6 / 1e4, V RMS-normed without weight, attention scale 1, softcap 30,
  tied embeddings, raw norm weights.
- **Template** differs from the pinned `845f1ee4…` in one rendered way: with
  thinking off, the generation prompt does **not** append the empty
  `<|channel>thought\n<channel|>` (diff of the two templates: that block
  removed, plus one whitespace-only line). Everything else renders the same.

Draft head (`gemma4-assistant`, 49 tensors): width **256**, FFN **2048**,
**4** query heads, `head_count_kv` scalar 2, window 512, context 131072,
`embedding_length_out` 2560, `pre_projection [5120, 256]`,
`post_projection [256, 2560]`; no centroid tensors. Same block structure
as the 12B head (three sliding, one global).

Projector (`clip`, 1,411 tensors): vision is **`gemma4v`** (the SigLIP path
the 26B-A4B uses), 16 blocks, width 768, 12 heads (head 64), FFN 3072,
projection 2560, BF16 matrices. Differences from the 26B-A4B's `gemma4v`:
- **clipped linears:** every block's `attn_q/k/v/out` and
  `ffn_gate/up/down` carry F32 scalars `<name>.input_min/input_max/
  output_min/output_max`: `y = clamp(W · clamp(x, in_min, in_max),
  out_min, out_max)` (`tools/mtmd/models/gemma4v.cpp`, `build_mm`);
  `mm.input_projection` has none.
- **no `v.std_bias` / `v.std_scale`** (standardization absent: skip it).
- the audio encoder (`a.*`, `mm.a.*`, 745 tensors) is not ours: skip.

### Implementation (in this order, no commits in between)

1. **Adapter** `inference/src/models/gemma4.zig`: move `heads`, `window`,
   the sliding KV heads, and the context length into `Config`; add
   `per_layer_input: usize` (0 or 256) and `kv_layers: usize` (= layer
   count, or 24); `config_e4b` (42 layers, 2560, 10240, 2/2 KV heads, 8
   heads, window 512, ctx 131072, PLE 256, kv_layers 24); selection by
   `block_count` (42). `validateMetadata`: per-config values for the moved
   keys, accept a scalar `head_count_kv`. `Layer.key`/`key_norm` optional
   (null on shared layers), `Binding` gains the PLE tensors and per-layer
   `inp_gate`/`proj`/`post_norm`; `kvSource(il)` helper. Fixture
   `inference/src/models/fixtures/gemma4-e4b.json` and a bind test;
   `max_layers`/`max_kv_width` still hold.
2. **CPU runtime** `gemma4_runtime.zig`: session layouts for
   `kv_layers` only; `project` skips K/V on shared layers, `attend` reads
   `state.layers[kvSource(il)]`; heads/window from config; PLE (`ple`
   workspace 42 × 256) computed in `step` and `prefillSpan` (token 0's row
   for image rows), applied after the FFN residual. `sourceLayer` for the
   head maps through `kvSource`.
3. **Metal plan** `gemma4_metal.zig`: the same for `step`, `prefill`
   chunks, `verify`, `prefillVision`, and the head; KV buffers for
   `kv_layers` only. PLE with existing kernels (gather a Q4_0 row, the
   matmul, the per-head RMS norm over 42 × 256 with a weight, the GELU
   gate-pair on `inp_gate` output × the layer's slice, matmul, norm,
   residual add). A new kernel only if an existing one lacks a stride.
4. **Draft head** `gemma4_assistant.zig`: `config_e4b` (width 256, FFN
   2048, 4 heads, 2/2 KV heads, window 512, ctx 131072, out 2560); move
   the pinned constants that differ into `Config`; scalar `head_count_kv`.
   Fixture `fixtures/gemma4-head-e4b.json`.
5. **Vision** `inference/src/vision/gemma4.zig` + `gemma4_metal.zig`:
   accept the smaller SigLIP geometry, skip `a.*` tensors, bind the
   optional clamp scalars per matrix, apply them on CPU and Metal (a clamp
   epilogue or kernel), make standardization optional. Fixture
   `inference/src/vision/fixtures/gemma4-e4b-mmproj.json`.
6. **Profile** `inference/src/profiles/`: a `gemma4_e` tag (or a digest
   flag on the gemma4 profile) that omits the empty thought channel when
   effort is off, accepting template `241c50d8…`; `vocabulary-check`
   expectations and the `tokenizer-fixtures.py` capture for it.
7. **Catalogue** `src/catalog.zig`: entry `gemma-4-e4b-qat` (quantization
   `Q4_0 (QAT)`, the three digests above, speculative off until
   measured); `docs/reference/artifacts.md` rows; help text if it lists
   entries.

### Checks (run once, after 1–7; all must pass before the one commit)

1. `zig build` + `make check` (format, unit tests, GPU fixtures, manifests).
2. Reference traces: `scripts/reference-generation.cpp` on the E4B for the
   hello-comma prompt → `tests/fixtures/gemma4-e4b-hello-comma/`; gates
   `gemma4-e4b-trace-cpu`, `-trace-f32`, `-trace-f16` (bring-up thresholds:
   max abs 2e-3, rel RMS 1e-4, same greedy token and top-5).
3. `gemma4-e4b-generation-metal`, `-vocabulary`, `-vision-metal`,
   `-draft-trace-metal`, and a perplexity reference + gate
   (`scripts/reference-perplexity.py`, `c512x8`) in `gates.json`.
4. `make verify` (Metal tier: the existing Gemma gates must be unchanged).
5. `make verify-long` (KV cache and the window changed), with a
   `gemma4-e4b-perplexity-4k` gate.
6. The heavy CPU checks last: the E4B's `-trace-cpu`, `-generation-cpu`,
   `-vision-cpu`, `-draft-trace-cpu`, and the existing `gemma4*-cpu` gates
   (the CPU runtime is shared); the full `make verify-cpu` if time allows.
7. Exercise `./zig-out/bin/nuclis generate --model gemma-4-e4b-qat` (text,
   an image, `--speculative`), and first-look `nuclis bench` vs
   `llama-bench` into `docs/reference/bench.md`.
8. Close: engineering-log entry + table row, `docs/reference/gemma4.md`
   (E4B section: PLE, shared KV), `vision.md`, `speculative-decoding.md`,
   `artifacts.md`, `THIRD_PARTY_NOTICES.md` if needed; empty this file.
