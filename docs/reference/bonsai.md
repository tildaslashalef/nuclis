# Bonsai 2 27B: facts from the artifact and the fork

Prism ML's ternary re-encoding of Qwen3.8-27B (`general.architecture =
qwen35`): the pinned Qwen adapter, tokenizer, and runtime schedule apply
unchanged, and the work sits below the adapter — two weight encodings, a
BF16 pair per DeltaNet layer, and a Hadamard rotation of every projection
input that the converter folded into the stored weights. Everything here
was read on 2026-09-18 from the pulled PQ2_0 file with
`scripts/gguf-inventory.py` (the committed inventory is
[`fixtures/bonsai-2-27b.json`](../../inference/src/models/fixtures/bonsai-2-27b.json),
rotation arrays included), from `nuclis model inspect`, and from the
PrismML llama.cpp fork at its release `prism-b10687-5d80cff`
(`ggml/src/ggml-common.h`, `ggml/src/ggml-quants.c`, `src/llama-model.cpp`,
`src/llama-graph.cpp`, `src/llama-impl.h`), which is read as the format and
semantics reference and used as the numerical oracle, never copied. The
whitepaper is the mathematical source for the rotation; the fork's loader
is the source for what the file's keys mean. Where a fact comes from the
fork rather than the file, the sentence says so.

## Artifacts

| | PQ2_0 (the catalogue's file) | PTQ1_0 (the footprint win, follows) |
| --- | --- | --- |
| Repository | `prism-ml/Ternary-Bonsai-2-27B-gguf` | same |
| Commit | `6ed5e12bf84b7a63069882c91dd9e9218647d17b` (Apache-2.0, released 2026-09-17) | same |
| File | `Ternary-Bonsai-2-27B-PQ2_0.gguf` | `Ternary-Bonsai-2-27B-PTQ1_0.gguf` |
| Size | 7,206,168,928 B | 5,946,648,928 B |
| SHA-256 | `3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1` | `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3` |
| Pulled | 2026-09-18 (`nuclis model pull bonsai-2-27b`, with the Q8_0 projector) | not pulled |
| `general.file_type` | 141 | 143 |

Companions: `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` (629,246,976 B, SHA-256
`6807ede6…`, pinned under the `mmproj` role) and `…-mmproj-BF16.gguf`
(931,145,856 B, `e287342d…`, the reference projector, not pinned); the
repository also lists an F16 language file of 53.8 GB (not pulled). The
digests are in [artifacts.md](artifacts.md). Stock llama.cpp rejects both
language files (ids past its type count); the oracle is the fork
([§ Oracle](#oracle-the-prismml-fork)).

## Metadata

The PQ2_0 header has 49 keys and 851 tensors (directory 11,120,982 bytes).
Every `qwen35.*` key equals the pinned Qwen3.8-27B's except the block count:
**`qwen35.block_count` is 64 and there is no `qwen35.nextn_predict_layers`
key**, where the Qwen release declares 65 blocks with one `nextn` layer —
the auxiliary prediction block is absent from the file (851 tensors are
exactly the Qwen file's 851 text tensors; its 15 auxiliary tensors have no
counterpart). The adapter accepts both declarations and reports
`auxiliary_prediction_layers` 0 or 1 accordingly. The rest: embedding 5120,
FFN 17408, 24 / 4 heads, key and value length 256, rope sections
[11, 11, 10, 0], θ 1e7, epsilon 1e-6, `full_attention_interval` 4, ssm
conv 4 / state 128 / groups 16 / rank 48 / inner 6144, context 262,144.
`general.name` "Hf", `general.version` "v5", `general.basename` "folded",
`general.size_label` "27B", `general.quantization_version` 2. Sampling
hints `general.sampling.temp` 1.0, `top_p` 0.95, `top_k` 20 (the card adds
instruct-mode 0.7 / 0.80 / 20 with presence penalty 1.5; reasoning effort
`xhigh` by default, `medium` supported, `low` behaves like `xhigh`).

Tokenizer identical in kind to Qwen3.8's: `gpt2` / `pre = qwen35`, 248,320
tokens, 247,587 merges, BOS = PAD 248044, EOS 248046, `add_bos_token`
false. `Hello,` tokenizes to `[9419, 11]` on both files.

## Tensors (851)

| Encoding | Tensors | Bytes | What |
| --- | ---: | ---: | --- |
| PQ2_0 (id 142) | 402 | 7,137,280,000 | every matrix: `token_embd.weight` and `output.weight` [5120, 248320] (untied), per DeltaNet block `attn_qkv` [5120, 10240], `attn_gate` [5120, 6144], `ssm_out` [6144, 5120], per attention block `attn_q` [5120, 12288], `attn_k` / `attn_v` [5120, 1024], `attn_output` [6144, 5120], every block's `ffn_gate` / `ffn_up` [5120, 17408] and `ffn_down` [17408, 5120] |
| F32 (id 0) | 353 | 10,582,016 | norms, `ssm_a`, `ssm_conv1d`, `ssm_dt.bias`, `attn_q_norm` / `attn_k_norm` |
| BF16 (id 30) | 96 | 47,185,920 | `ssm_alpha.weight` and `ssm_beta.weight` [5120, 48] on the 48 DeltaNet blocks (Q8_0 in the Qwen file) |

Text weights 7,195,047,936 bytes. The PTQ1_0 file has the same 851 tensors
with the 402 matrices in id 143 (read from its header on 2026-09-18 by
range request; to be re-read from the pulled file when it is measured).

## Encodings (from the fork's `ggml-common.h` and `ggml-quants.c`)

Both are group-128 ternary: `w = d · t` with `t ∈ {−1, 0, +1}` and one F16
scale `d` per 128 weights. Implemented from these contracts in
[`quant/decode.zig`](../../inference/src/quant/decode.zig) and proven
bit-identical to the fork's `dequantize_row_pq2_0` / `dequantize_row_ptq1_0`
on the committed [`ternary.json`](../../inference/src/quant/fixtures/ternary.json)
fixture (arbitrary payload bytes, so non-canonical codes reproduce the
reference arithmetic too, plus all-zero, all-one, and canonical blocks;
`scripts/quant-fixtures.py --prism-checkout`).

- **PQ2_0** (id 142; the fork's file type `MOSTLY_PQ2_0` 128): 34 bytes —
  `ggml_half d` then `uint8_t qs[32]`. Element `j` is the two bits at
  `(j % 4) * 2` of byte `j / 4`, low bits first; the value is `d · (q − 1)`,
  so code 3 decodes to +2 as the reference does (a ternary file never
  stores it). 2.125 bits per weight. The unpack is the Q4_0 nibble path's
  shape, which is why this file is the bring-up target.
- **PTQ1_0** (id 143; `MOSTLY_PTQ1_0` 129): 28 bytes — `uint8_t qs[24]`,
  `uint8_t qh[2]`, then `ggml_half d`. Mainline's TQ1_0 trit packing at
  group 128: a byte holds five trits in base 3 scaled so the leading digit
  sits in the top bits (`byte = ceil(Σ tₙ 3^(4−n) · 256 / 243)`); digit `n`
  is `((byte · 3ⁿ mod 256) · 3) >> 8` and the value `d · (digit − 1)`. The
  24 bytes are two runs of 16 and 8 bytes (the fork's stage table
  `{32, 16, 8}` skips the 32 at this width), each emitted **digit-major**:
  all first digits of the run, then all second digits, … (80 + 40 = 120
  values); the two `qh` bytes hold four trits each at the same leading
  positions (the value times three), emitted digit-major too (8 values).
  1.75 bits per weight.
- **BF16** (id 30): the high half of an IEEE single, widened without
  rounding.

The fork also defines a legacy group-64 `Q2_0` (id 42, retired from its
files) and keeps mainline's TQ1_0 (id 34, group 256), which nuclis does
not store.

## Rotation (`prism.hadamard.*`, as the fork's loader reads it)

The converter rotated every projection's input basis by
`R = (1/√n) · Hₙ · S` with n = 1024 and S the diagonal of a ±1 sign vector,
folded `R` into the stored weights, and stores the embedding table in the
rotated basis. Inference therefore applies the transform to the
**activation** before each rotated projection and the inverse to the
embedding row after lookup. The keys, all retained whole by the parser
(`gguf.retained_key_prefixes`):

| Key | Value in the file | Meaning (fork `llama-model.cpp` / `llama-graph.cpp`) |
| --- | --- | --- |
| `version` | 1 | the only version the fork loads |
| `block_size` | 1024 | the transform acts per 1024 consecutive input elements; every rotated width is a multiple |
| `transform` | `normalized-sylvester-walsh-hadamard` | `H[row][col] = (1/√1024) · (−1)^popcount(row & col)`, symmetric and orthonormal, so `H⁻¹ = H` |
| `axis` | `input-last-dimension` | the transform acts along the projection's input (GGUF dimension 0) |
| `sign_mode` | `explicit` | a sign vector per width is stored (`identity` would mean none) |
| `sign_widths` | [5120, 6144, 17408] | the three rotated input widths |
| `sign_values` | 28,672 int32 of ±1 (14,168 +1, 14,504 −1) | the three vectors concatenated in `sign_widths` order |
| `weight_names` | 401 strings | `output.weight`; per DeltaNet block `attn_qkv`, `attn_gate`, `ssm_out`, `ffn_gate`, `ffn_up`, `ffn_down`; per attention block `attn_q`, `attn_k`, `attn_v`, `attn_output`, `ffn_gate`, `ffn_up`, `ffn_down` — every matrix except the embedding and `ssm_alpha` / `ssm_beta` |
| `inverse_weight_names` | [`token_embd.weight`] | the row lookup gets the inverse |
| `gdn_v_grouped` | true | see below |

**Forward transform** (the fork's `build_lora_mm`, applied once per
activation and memoized when several folded weights share it): reshape the
activation to rows of 1024, `x' = x ⊙ signs[width]` elementwise, then
`y = H · x'` per row (`llama_mul_mat_hadamard`: a matmul against the
1024 × 1024 matrix with the `GGML_HINT_SRC0_IS_HADAMARD` hint, which the
backends turn into a fast Walsh-Hadamard transform). The projection then
runs on the stored weights: `W · y`. The DeltaNet `ssm_alpha` and
`ssm_beta` projections and every norm see the untransformed activation.

**Inverse after lookup** (`token_embd.weight`): `h = signs[5120] ⊙ (H · z)`
where `z` is the looked-up row — the transform first, then the signs, the
reverse order of the forward path.

**`gdn_v_grouped`** (fork: applied only to `ssm_out`): the DeltaNet output
that enters `ssm_out` has width 6144 = 48 value heads × 128 in *tiled* head
order `[hd = 128, nk = 16, rep = 3]` (head dimension fastest, then the 16
key groups, then the 3 value heads sharing each group: value head
`rep · 16 + nk`). The fold was computed in the *grouped* order
`[hd, rep, nk]` (value head `nk · 3 + rep`), so the fork permutes the
activation `[hd, nk, rep] → [hd, rep, nk]` before the signs and the
transform. The fork's numbers: `perm_hd = 6144 / ssm_dt_rank(48) = 128`,
`perm_nk = ssm_n_group = 16`, `perm_rep = 48 / 16 = 3`. How nuclis's
DeltaNet mixer orders its output decides whether this permutation is a
real gather or a no-op; session 2 reads it from `qwen35_runtime.zig`
rather than assuming.

The adapter pins this contract ([`qwen35.zig`](../../inference/src/models/qwen35.zig)
`validateRotation`): version 1, block 1024, the transform and axis strings,
explicit signs of exactly these widths and values ±1, the weight list equal
to the 401-name set above (each name claims one slot; a name outside the
set or claimed twice is refused), the inverse list equal to the embedding,
`gdn_v_grouped` present; any other `prism.hadamard.*` key, or rotation keys
without the version, is `UnsupportedConfiguration`. The binding carries
`Rotation` (block, the three sign vectors borrowed from the Document,
`value_grouped`), and `validate` prints the basis. Until the runtimes apply
the transform, `Runtime.init` and `Plan.init` refuse a rotated binding with
`error.UnsupportedRotation` rather than run in the stored basis.

Prism's whitepaper calls the transform "one of the larger non-matmul costs
of a decode step" at batch 1 on Metal and fuses the sign flip into its
load path; whether nuclis fuses it into the matvec input load is measured
in KERN-10, not assumed.

## Chat template

SHA-256 `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`
(8,952 bytes): **the upstream Qwen3.8 template**, not the pinned `qwen38`
profile's (`12827f24…`, 9,993 bytes), which carries Unsloth's fixes
(`developer` role, merged system messages, `high` mapped to `xhigh`,
tool-argument type checks). The alias gate
(`scripts/profile-alias-check.py --profile qwen38 --reference-revision
5d80cff0…`, the fork's server holding the PQ2_0 file, 2026-09-18) replayed
the pinned fixtures: **24 of 28 text prompt cases, all 20 tool cases, and
all 20 token cases are byte-identical**; the 4 that differ are the
`merged_system_*` cases (a `system`, `developer`, `system`, `user`
history), which the upstream template refuses with `System message must be
at the beginning.` instead of merging. So the digest is not an alias (an
alias must render every case), and the difference is a stricter input
contract, not a different rendering: for every conversation the file's own
template accepts, the pinned profile renders the same bytes. MODL-17
decides the profile from this evidence; the recommendation is the
catalogue entry's `profile = .qwen38` (a registry `profile` renders the
pinned protocol onto the file), with the deviation recorded: nuclis merges
system messages the file's template would refuse. The fork's
`llama-completion --jinja` aborts at its start-up template self-test on
this template (the same raise, from its fixed example conversation); the
fork's server renders it.

## Oracle: the PrismML fork

`github.com/PrismML-Eng/llama.cpp` (MIT, tracks mainline), release
`prism-b10687-5d80cff` = commit `5d80cff0b8cb9f2bf823cfc4e71e3abb97f290d6`
(2026-09-17), pinned by tag since the fork has already retired one format.
Built as the mainline recipe with `.zig-cache/reference/prism-llama.cpp`
in place of the mainline checkout
([reference-baseline.md § The second oracle](reference-baseline.md#the-second-oracle-the-prismml-fork-modl-16-2026-09-18)).
Two pins, never one moving one: `scripts/quant-fixtures.py` carries
`PRISM_REVISION` beside `REVISION`, and `profile-alias-check.py` takes
`--reference-revision`.

Runs on 2026-09-18 (M4 Pro, Metal, the PQ2_0 file):

- `llama-completion -ngl 99 -p "Hello," -n 16 --temp 0 -no-cnv`:
  `Hello, I'm a student in the University of the West of England (UWE)`;
  17.14 tok/s decode over 15 tokens, 8.22 tok/s prompt over 2 (a smoke
  run, not a benchmark). The loader logs
  `loaded 402 Hadamard-folded weight(s) (1 inverse-lookup) using 1 rotation(s) and 3 sign vector(s)`,
  `file type = PQ2_0 - 2.13 bpw (group 128)`, `model params = 26.90 B`.
- `scripts/reference-generation.cpp` built against the fork
  (`.zig-cache/generation/prism-reference-generation`, the same source, the
  fork's headers and libraries) on `Hello,`: 129 files in
  [`tests/fixtures/bonsai-hello-comma/`](../../tests/fixtures/bonsai-hello-comma/)
  (two positions × 64 layers × 5,120 F32, `logits.f32` of 248,320),
  greedy token 353 (` I`) — the same token the Qwen3.8 oracle picks, with
  logit 11.51 against Qwen's 12.41; the layer-63 residual at position 1
  has RMS 7.03 (Qwen 5.46) and a peak of 327 (Qwen 91). The payload of
  `make compare-bonsai-cpu` (session 2).

## CPU reference against the fork (MODL-16, 2026-09-18)

[`cpu/hadamard.zig`](../../inference/src/backends/cpu/hadamard.zig) is the
transform: `forward` (signs, then the butterflies per block, F64 scratch,
scale `1/√block`) and `inverse` (butterflies, then signs), tested against
the parity-defined matrix and as a round trip. The Qwen runtime
([`qwen35_runtime.zig`](../../inference/src/models/qwen35_runtime.zig))
decodes the three sign vectors to F32 at init and, on a rotated binding,
applies `inverse` to the embedding row after lookup and `forward` to each
activation a rotated weight reads — the normed residual before the mixer
(shared by `attn_qkv` / `attn_gate` or `q` / `k` / `v`), the mixer output
before `ssm_out` / `attn_output`, the normed residual before the FFN
(shared by `gate` / `up`), the FFN hidden before `ffn_down`, and the
normed residual before the output head — into a separate scratch, so
`ssm_alpha` / `ssm_beta` keep the untransformed residual. `ssm_out`'s
input is regathered first: the mixer emits the 48 value heads tiled
(head `rep · 16 + nk`, key group `h % 16`, the same order as the fork's
mixer), and `value_grouped` says the fold used the grouped order
(`nk · 3 + rep`), so the permutation is a real gather. A plain Qwen file
takes none of these paths.

`make compare-bonsai-cpu` (`generate --backend cpu --raw --prompt 'Hello,'`
on the PQ2_0 file, then `compare-generation.py` against
`tests/fixtures/bonsai-hello-comma` at the bring-up thresholds 2e-3 /
1e-4) **passes on the first run**: 129 files, max abs 2.44e-4 (at
`token-1-layer-63`), max relative RMS 5.4e-6, greedy token 353 with the
same top three logits as the fork to three decimals (353: 11.505, 1204:
9.753, 198: 9.348). The two positions take about 56 s on the CPU.

## Status

MODL-16 closed on 2026-09-18: the file validates and binds, the three
encodings decode against the fork's fixtures, the CPU reference applies
the rotation and matches the fork's traces. KERN-10 session 1 (the same
day) added the Metal decoders, matvecs, and tiles for the three encodings
([metal-backend.md § Ternary](metal-backend.md#ternary-matvecs-and-tiles-kern-10-2026-09-18):
PQ2_0 119 GB/s, PTQ1_0 88 GB/s on the output head — at the kernel set's
multiply rate, half the Q4_0 byte rate by density); session 2 is the
transform kernel. The Metal plan, profile, catalogue, and acceptance are
MODL-17. Until then `generate --backend metal` on the file returns
`UnsupportedRotation`.
