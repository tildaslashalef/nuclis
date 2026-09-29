# Laya

Laya (`convaiinnovations/laya`, Apache-2.0) is a decision model: a
ModernBERT-large encoder with a typed-decision head. It answers a typed
question (`choice`, `score`, or `noul`, a yes/no probability) about a state
(text, or a JSON document) in one forward pass, with no decoding. nuclis
runs the root checkpoint (commit `55cf4c4e`, pulled under
`~/.nuclis/models/convaiinnovations/laya/`) in three parts:

| Part | Files | Status |
| --- | --- | --- |
| Tokenizer | `inference/src/tokenizer/hf_json.zig`, `gpt2.zig`, `nfc.zig` | landed ([tokenizer.md § Hugging Face `tokenizer.json`](tokenizer.md#hugging-face-tokenizerjson)) |
| Encoder and head, CPU | `inference/src/models/modernbert.zig`, `modernbert_runtime.zig`, `laya.zig`, `inference/src/backends/cpu/dense.zig` | landed (below) |
| Input contract, calibration, `nuclis decide` | `inference/src/profiles/laya.zig`, `inference/src/decide.zig`, `src/decide.zig` | planned ([TODO.md](../../TODO.md)) |

## The encoder

`encoder/config.json` (`model_type` `modernbert`) gives 28 layers, hidden
1024, 16 heads of 64, intermediate 2624, vocabulary 50368, no biases,
LayerNorm eps 1e-5. Layer `i` attends globally when `i % 3 == 0` (RoPE
theta 160000), else within a window (theta 10000): query `i` sees keys
`|i − j| ≤ 64`, half of `local_attention` 128. `modernbert.parseConfig`
also reads the Transformers 4.x spellings (`global_attn_every_n_layers`,
`global_rope_theta`, `local_rope_theta`) and rejects biases, an activation
other than exact GELU, head widths not a multiple of 8, and other layer
types.

Tensors, PyTorch `[out, in]`, under `encoder.`:
`embeddings.tok_embeddings.weight [50368,1024]`, `embeddings.norm.weight`,
per layer `attn_norm.weight` (absent on layer 0), `attn.Wqkv.weight
[3072,1024]`, `attn.Wo.weight`, `mlp_norm.weight`, `mlp.Wi.weight
[5248,1024]`, `mlp.Wo.weight [1024,2624]`, then `final_norm.weight`. The
root set stores them in F16.

The forward, as Transformers 5.17's `modeling_modernbert.py` (the oracle's)
computes it:

```
x = LayerNorm(embed[ids])
per layer:  x += Wo(attention(rope(Wqkv(attn_norm(x)))))     # attn_norm: identity on layer 0
            a, g = halves of Wi(mlp_norm(x));  x += mlp.Wo(gelu(a) * g)
x = LayerNorm_final(x)
```

`Wqkv` rows are q, k, v, each `[16][64]`. RoPE is split-half (pairs `d` and
`d + 32`) with inverse frequencies `theta^(−2i/64)`; the reference forms
each angle as an F32 product, so `modernbert_runtime.zig` rounds the angle
to F32 before its cosine and sine. Attention is bidirectional, scale 1/8.
GELU is the exact erf form. LayerNorm has a weight and no bias.

## The head

In `model.safetensors`, unprefixed: `type_emb.weight [3,1024]` is added to
every row (choice 0, score 1, noul 2); then two PyTorch
`TransformerEncoderLayer`s (`d=1024`, 16 heads, feed-forward 4096,
`norm_first`, ReLU, LayerNorm eps 1e-5 with bias, every linear with bias):
`x += out_proj(mha(norm1(x)))`, `x += linear2(relu(linear1(norm2(x))))`,
`in_proj_weight [3072,1024]` stacking q, k, v. At each option's `[MASK]`
row the scorer gives the option's logit: `LayerNorm(scorer.0) →
Linear(scorer.1) → GELU → Linear(scorer.3, → 1)`. `act_head.*` and the
`temperature` buffer are known and unused; `Laya.open` rejects any other
tensor, and any missing or misshapen one.

## On the CPU

`Laya.open(gpa, io, dir)` parses the config, maps the checkpoint, binds
every tensor, and decodes all but the embedding table to F32 (about 1.5 GB
for the root set; embedding rows decode as ids need them, so the mapping
stays open). `Laya.logits(io, gpa, ids, markers, kind, out, trace)` scores
one built sequence; `trace` sees the residual stream after each encoder
layer, after `final_norm`, and after each head layer.

The kernels are `backends/cpu/dense.zig`: `Y = X·Wᵀ + b` in 4×4 output
tiles of eight-lane F32 vectors, output columns split across `Io` tasks;
LayerNorm with F64 statistics; bidirectional attention, full or windowed,
heads split across tasks. Each output is summed by one task in a fixed
order, so results do not depend on the core count.

### Checked against the oracle

`scripts/laya-reference.py` runs the `laya` 0.3.20 package (PyTorch 2.14.0,
Transformers 5.17.0) in F32 on the CPU and writes
`inference/src/models/fixtures/laya/`: for 8 requests the ids, markers, raw
logits, and answer, and in `activations.f32` rows of the residual stream
(row 0, up to three markers, the last row) after encoder layers 0, 1, 3,
and 27, after `final_norm`, and after both head layers (marker rows only).

```sh
zig build test-laya -Doptimize=ReleaseFast -- ~/.nuclis/models/convaiinnovations/laya
```

(gate `laya-cpu`, tier verify). The bounds are relative, because F32
summation-order differences scale with the values summed and the
pre-norm residual carries outlier dimensions up to 1.65·10⁴: per tensor,
the largest difference over the largest reference value ≤ 1e-5 and the
difference's RMS over the reference's ≤ 1e-5; per logit, the difference
over max(1, |logit|) ≤ 1e-4. (The first plan's absolute 1e-4 held through
layer 3 and after `final_norm`, not at layer 27 or in the head, whose
values reach 10²–10⁴.)

Measured 2026-09-29, Apple M4 Pro (12 cores), Zig 0.16.0, ReleaseFast,
the MODL-30 session 2 commit; worst stage per request:

| Request | Tokens | Options | ms | scaled max \|Δ\| (stage) | rel RMS | scaled logit \|Δ\| |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `choice_described` | 59 | 3 | 279 | 3.1e-6 (encoder.27) | 2.3e-6 | 2.1e-6 |
| `choice_labels` | 36 | 3 | 169 | 2.6e-6 (final) | 1.7e-6 | 5.2e-6 |
| `score` | 49 | 3 | 230 | 2.4e-6 (final) | 2.5e-6 | 6.0e-6 |
| `noul` | 56 | 2 | 256 | 6.6e-6 (head.0) | 4.8e-6 | 4.0e-6 |
| `json_state` | 119 | 2 | 552 | 4.0e-6 (final) | 5.4e-6 | 4.8e-6 |
| `long_text` | 512 | 3 | 2410 | 1.7e-6 (head.0) | 2.8e-6 | 7.9e-6 |
| `long_list` | 512 | 2 | 2408 | 8.3e-6 (head.0) | 7.0e-6 | 9.0e-6 |
| `choice_20` | 193 | 20 | 906 | 2.9e-6 (encoder.27) | 2.1e-6 | 7.5e-6 |

The times are the forward alone (encoder and head, the trace's comparison
included); opening took 212 ms with the checkpoint in the page cache. A
512-token sequence is about 2·10¹¹ multiply-adds, so 2.4 s is roughly 170
GFLOP/s across the cores.
