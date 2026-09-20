# Muse Glimmer 30B: facts from the artifact and the reference

Meta's dense 30B chat model (`general.architecture = muse-glimmer`): a
period-4 sliding/global layer pattern with NoPE on the global layers, an
attention output gate, sandwich norms with two epsilons, a gpt-4o-family
tokenizer, and the ATEM tool protocol in its template. Everything here was
read on 2026-09-19 from the pulled UD-Q4_K_XL file with
`scripts/gguf-inventory.py` (the committed inventory is
[`fixtures/muse-glimmer-30b.json`](../../inference/src/models/fixtures/muse-glimmer-30b.json))
and from the pinned llama.cpp reference `7620399`
(`src/models/muse-glimmer.cpp`, `src/llama-hparams.cpp`,
`src/llama-vocab.cpp`, `src/unicode.cpp`), which is read as the semantics
reference and used as the numerical oracle, never copied. The file carries
no per-layer arrays and no epsilon, scale, or gate keys beyond the ones
listed under *Metadata*: every "folded at conversion" and "hard-coded"
fact below comes from the reference source, and the sentence says so.

## Artifacts

| | Language model |
| --- | --- |
| Repository | `unsloth/Muse-Glimmer-30B-GGUF` |
| Commit | `faa5b025c584459c13febfa5c59883516710ae39` |
| File | `Muse-Glimmer-30B-UD-Q4_K_XL.gguf` |
| Size | 15,878,222,368 B |
| SHA-256 | `82bece304887a313ece08400bc030f6066c7bff5b906b0cd40308ec8a409fd38` |
| Pulled | 2026-09-17 (`nuclis model pull`, sidecar verified) |
| `general.file_type` | 15 (Q4_K_M base; `quantize.imatrix.*` names an imatrix over 416 entries, 166 chunks) |

Companions, pulled and verified the same day, not executed:
`mmproj-kquant.gguf` (1,400,328,928 B, SHA-256 `f48b4523…`, the `mmproj`
role) and `dflash-kquant.gguf` (1,631,205,312 B, `27d9a805…`, a DFlash
drafter carried under the `mtp` role). Digests and the rest of the
repository's listing are in [artifacts.md](artifacts.md). The quantization
choice (Meta's "K-Quant-17GB" tier, 1.0 % measured degradation, wide
headroom on 48 GB) is recorded in the plan that chose it and needs no
restating here.

## Metadata

36 keys. The model's own, all scalars:

| Key | Value |
| --- | --- |
| `muse-glimmer.block_count` | 52 |
| `muse-glimmer.context_length` | 131,072 |
| `muse-glimmer.embedding_length` | 6656 |
| `muse-glimmer.feed_forward_length` | 19,968 |
| `muse-glimmer.attention.head_count` | 32 |
| `muse-glimmer.attention.head_count_kv` | 2 |
| `muse-glimmer.attention.key_length` / `value_length` | 128 / 128 |
| `muse-glimmer.attention.layer_norm_rms_epsilon` | 1e-5 (`9.999999747378752e-06`) |
| `muse-glimmer.attention.sliding_window` | 2048 |
| `muse-glimmer.attention.sliding_window_pattern` | 4 |
| `muse-glimmer.rope.freq_base` | 500,000 |
| `muse-glimmer.logit_scale` | 0.1961161345243454 |
| `muse-glimmer.final_logit_softcapping` | 20.0 |

`general.name` is `Muse-Glimmer-30B`, `general.size_label` `28B`. There is
no `rope.dimension_count`, no `rope.scaling.*`, no per-layer array, no
attention-scale or gate key, no `general.sampling.*`; the reference
derives everything else (below). Tokenizer keys are under *Tokenizer*.

## Tensors (731)

313 F32, 410 Q4_K, 8 Q5_K. Non-layer: `token_embd.weight` [6656 ×
202,048] Q4_K, `output.weight` [6656 × 202,048] Q5_K (untied),
`output_norm.weight` [6656] F32. Every layer `blk.N` has the same fourteen
tensors (shapes as stored, `[in, out]`):

| Tensor | Shape | Encoding |
| --- | --- | --- |
| `attn_norm`, `post_attention_norm`, `ffn_norm`, `post_ffw_norm` | [6656] | F32 |
| `attn_q` | [6656, 4096] | Q4_K |
| `attn_k`, `attn_v` | [6656, 256] | Q4_K |
| `attn_gate` | [6656, 4096] | Q4_K |
| `attn_output` | [4096, 6656] | Q4_K (layers 0–44), Q5_K (45–51) |
| `attn_q_norm`, `attn_k_norm` | [128] | F32 |
| `ffn_gate`, `ffn_up` | [6656, 19,968] | Q4_K |
| `ffn_down` | [19,968, 6656] | Q4_K |

Every encoding already has CPU and Metal kernels; nothing on the encoding
side is new for this family.

## Forward pass (from the reference graph, `muse-glimmer.cpp`)

Notation: `d` = 6656, `hd` = 128, `rms_ε(x) = x / sqrt(mean(x²) + ε)`,
`norm_w(x) = rms_1e-5(x) · w`. The norm weights are stored with the `+1`
already folded in at conversion (the reference says so in its tensor
loader); `attn_q_norm` absorbs the model's `qk_scale_factor` and
`attn_k_norm` is ones. The post norms use a hard-coded `ε = 1e-8`
(`post_norm_eps` in the graph), the pre norms the file's 1e-5.

1. **Embedding.** `x = E[token]`; `h = rms_1e-5(x)` **without a weight**
   (`build_norm` with null weight, the file's epsilon).
2. **Layer kinds.** `set_swa_pattern(4)`: layer `l` is *sliding* when
   `l mod 4 < 3` and *global* when `l mod 4 = 3`: 39 sliding layers, 13
   global (3, 7, …, 51).
3. **Per layer** (`h` is the residual stream):
   - `a = norm_attn(h)`.
   - `q = Wq a` as [32][hd]; `k = Wk a` as [2][hd]; `v = Wv a` as [2][hd];
     `g = Wgate a` (4096 values, one per query-head dimension).
   - `q = norm_q(q)`, `k = norm_k(k)` per head over `hd` (ε 1e-5).
   - **Sliding layers only:** `q = rope(q)`, `k = rope(k)`: NORM pairing
     (adjacent dimensions `(2i, 2i+1)`), `n_rot = 128` (the whole head; no
     `rope.dimension_count` key, so the head width), base 500,000, no
     scaling, no YaRN terms. **Global layers have no position encoding.**
   - Scores `q · k / sqrt(hd)` (`kq_scale = 1/√128`), causal; on sliding
     layers additionally masked to `p_key > p_query − 2048` (the reference's
     `LLAMA_SWA_TYPE_STANDARD`: masked when `p_query − p_key ≥ n_swa`, so
     2048 positions including the query's own are visible). GQA 32 over 2
     (groups of 16). No attention logit softcap.
   - `o = concat(heads) ⊙ sigmoid(g)` (the gate multiplies the attention
     output element-wise *before* the output projection), then `o = Wo o`.
   - `h = rms_1e-8(o) · w_post_attn + h`.
   - `f = norm_ffn(h)`; `f = Wdown (silu(Wgate f) ⊙ (Wup f))`.
   - `h = rms_1e-8(f) · w_post_ffn + h`.
4. **Head.** `y = norm_out(h)`; `logits = Wout y · 0.196116`; `logits =
   20 · tanh(logits / 20)`.

The reference keeps the sliding layers' cache in a separate window-sized
cache (`build_attn_inp_kv_iswa`); the plan for the Metal unit keeps the
full context on every layer and masks. It also exposes each layer's input
residual (`t_layer_inp`) for the DFlash drafter; that belongs to the
speculative-decoding unit.

## Tokenizer

`tokenizer.ggml.model = gpt2`, `tokenizer.ggml.pre = llama4`, 202,048
tokens and 439,802 merges; BOS `<|begin_of_text|>` 200000, EOS
`<|end_of_text|>` 200001, EOT `<|eot|>` 200008, padding
`<|finetune_right_pad|>` 200018, `add_bos_token` true, `add_sep_token`
false. Token types: 200,000 normal (ids 0–199,999) and 2,048 control
(200000–202047). Fifteen of the control tokens are named; the rest are
`<|reserved_special_token_N|>`: `<|begin_of_text|>` 200000,
`<|end_of_text|>` 200001, `<|eom|>` 200007, `<|eot|>` 200008,
`<|finetune_right_pad|>` 200018, `<|start|>` 200022, `<|message|>` 200023,
and the media markers `<|image_start|>` 200080, `<|image_end|>` 200081,
`<|vid_start|>` 200082, `<|vid_end|>` 200083, `<|vid_frame_separator|>`
200087, `<|image|>` 200090, `<|video|>` 200091, `<|patch|>` 200092. The
role words are ordinary tokens: `system` 15651, `user` 1556, `assistant`
140680, `tool` 21188.

The reference (`llama-vocab.cpp`) maps the `llama4` label, with `gpt-4o`,
`kanana2`, and `talkie`, to `LLAMA_VOCAB_PRE_TYPE_GPT4O` with
`clean_spaces = false`, and at load re-types `<|start|>` and `<|message|>`
(with `<|channel|>` and `<|constrain|>` where they exist) as
**user-defined**, so `/tokenize` matches them even with `parse_special`
off; `vocabulary.zig` applies the same override. It marks `<|eot|>`,
`<|eom|>`, and `<|end_of_text|>` as end-of-generation.

**The regex the reference runs is not the one the tokenizer declares.** The
gpt-4o pattern with `\p{Lu}`/`\p{Ll}` classes and `(?i:…)` contractions is
a comment in `llama-vocab.cpp`; the active expression is a rewritten form,

```
[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))*((?=[\p{L}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?
|[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))+((?=[\p{L}])([^A-Z]))*(?:'[sS]|…)?
|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

run through `unicode.cpp`'s generic path (no custom splitter for this
label): every code point ≥ 128 collapses to one byte per category (letter,
number, mark, punctuation, symbol, whitespace, other) and `std::regex` in
ECMAScript grammar runs on that. Consequences, all pinned by the harness
below: "uppercase" is *a letter that is not ASCII a–z* and "lowercase" *a
letter that is not ASCII A–Z*, so every non-ASCII letter is both;
combining marks are not letters; `\s` is the reference's whitespace set;
alternation is leftmost-first with backtracking. Resolved into rules,
which [gpt4o.zig](../../inference/src/tokenizer/gpt4o.zig) implements
without a regex engine:

1. **Word.** One optional prefix character that is not a letter, number,
   or CR/LF (whitespace and marks included), then a letter run cut as
   follows: take the greedy run of upper-ish letters; if an ASCII
   lowercase letter follows, add the greedy run of lower-ish letters;
   otherwise cut after the last non-ASCII letter of the upper-ish run;
   otherwise (all ASCII uppercase) take the whole run. Then an optional
   contraction `'s 't 're 've 'm 'll 'd` (ASCII case-insensitive).
   `HelloWORLD` → `Hello`,`WORLD`; `ÀBC` → `À`,`BC`; `ABCÀÉ` whole;
   `ÀÉbcDE` → `ÀÉbc`,`DE`; `don't` whole; ` I'M` whole; `é` → `e`
   then the mark.
2. **Number.** Up to three consecutive `\p{N}` characters (`1234567` →
   `123`,`456`,`7`; `①②③④` → `①②③`,`④`).
3. **Other.** An optional space, then a run of characters that are neither
   whitespace, letter, nor number, then any run of CR, LF, and `/`
   (`!!\r\n/x` → `!!\r\n/`,`x`; ` 's` → ` '`,`s`).
4. **Whitespace.** As qwen35: a run ending at its last CR/LF; else all but
   the last whitespace when text follows; else the run.

The boundaries were read with `scripts/reference-split.cpp`, a harness
linking the reference's `unicode.cpp` and `unicode-data.cpp` and printing
`unicode_regex_split` pieces, over some sixty adversarial strings (token
ids from `/tokenize` cannot show them: `WORLD` becomes `W`,`ORLD` through
merges either way); the splitter's unit test pins those pieces. The shared category table
(`unicode-ranges.bin`) needed no change: the seam reads ASCII case only.

The committed fixture
[`muse_glimmer-text.json`](../../inference/src/profiles/fixtures/muse_glimmer-text.json)
holds the reference server's ids for 20 standalone strings (the shared set
plus the channel markers, with `parse_special` on and off) and 28 rendered
prompts; `zig build test-vocabulary -- <file>` matches all of them
(2026-09-19: 202,048 tokens, 439,802 merges, 125 distinct normal token
pieces round-trip through BPE). The check selects the Muse expectations by
template digest until the profile exists.

## Chat template

`tokenizer.chat_template` is 7,167 bytes, SHA-256
`114f55ebdc1804c1af371197b9fdf2d6bb925966c9dfe46b73782a71bc07965e`, no
trailing newline. It differs from the Hub repository's
`chat_template.jinja`; the profile pins the file's digest as always.
Facts the fixture capture confirmed, which `profiles/muse_glimmer.zig`
implements (MODL-13; the clauses are in
[prompt-profile.md § Muse Glimmer](prompt-profile.md#muse-glimmer-muse_glimmer),
the tool protocol in AGNT-10):

- The prompt opens with `bos_token` as text, but the server's
  `/apply-template` output omits it (its tokenizer adds BOS): the fixture
  prompts start at `<|start|>`, as Gemma's do. nuclis's encoder never adds
  BOS, so the profile writes `<|begin_of_text|>` itself.
- Turns are `<|start|>ROLE<|message|>…<|eot|>`, content verbatim (the
  template trims nothing). The `<|eom|>` ending only applies to the last
  tool call of an assistant message that another assistant message
  follows; two content messages in a row both end with `<|eot|>` (the
  `continued_*` cases). The system turn is the caller's text,
  `\n\nReasoning strength: LEVEL.`, then
  `\n\n# Valid recipients: "self", "user".` (tool namespaces between). The
  level is the `reasoning_strength` template variable (default `high`, no
  off); the fixture captures `low`, `medium`, `high`, `xhigh`, and the
  profile renders the shared `off` as `low`.
- Without a system message the template synthesizes one: `You are a
  helpful AI assistant.`, `Knowledge cutoff: 2026-01-04.`, and, because the
  server defines `strftime_now`, `Current date: 2026-09-19.` on capture
  day (the fixture's `single_*`, `history_*`, `continued_*`, `unicode_*`,
  and `empty_*` cases carry that line). The profile decision of
  2026-09-16 renders the default without the date line; the MODL-13 test
  must account for the captured line.
- The template has no branch for the `developer` role, but the reference
  sends it as `system`: every leading system or developer message is its
  own system turn, each with the strength and recipients lines (the
  `merged_system_*` cases render three).
- Assistant history: `reasoning_content` renders as
  `<|start|>assistant to=self<|message|>…<|eom|>` wherever it appears,
  then the answer as `<|start|>assistant to=user<|message|>…<|eot|>`; the
  generation prompt is `<|start|>assistant`, so the model's first header
  (` to=self`) arrives as text.

## Reference oracle status

The pinned reference `7620399` loads and runs the file (2026-09-19,
Metal, `-ngl 99`): `llama-completion -p "Hello," -n 16 --temp 0 -no-cnv`
continues `Hello, I am trying to use the following code to create a new
table in a database`; `--jinja --single-turn` on `Explain slices in Zig in
one sentence.` renders the synthesized system turn and the model opens a
`to=self` reasoning message. Smoke rates from those runs, not a baseline:
prompt 87.9 tok/s over 64 tokens, generation 14.5 tok/s (68.9 ms per
token). The server (`llama-server --ctx-size 2048 --parallel 1 --device
MTL0 --n-gpu-layers 99`) is the fixture source.

## CPU reference against the oracle (MODL-11, 2026-09-19)

The adapter (`inference/src/models/muse_glimmer.zig`) pins every
`muse-glimmer.*` key and the vocabulary size, binds the 731 tensors by
name and shape (15,865,108,480 weight bytes), classifies layers by
`kindOf` (`index % 4 == 3` is global), and rejects any other
`muse-glimmer.*` key, missing or extra tensor, wrong shape, or
non-executable encoding with a typed error; its tests bind the committed
inventory and eighteen mutations of it. The CPU runtime
(`muse_glimmer_runtime.zig`) executes the forward pass above with the
shared primitives; the only addition to the CPU backend was an
adjacent-pair mode on `cpu.rope` (`Pairing.adjacent`, the GGUF "normal"
rope type; Qwen and Gemma rotate split-half). The Metal plan is
[§ Metal plan](#metal-plan-modl-12-2026-09-19).

The oracle traces are `tests/fixtures/muse-glimmer-hello-comma/`
([provenance](../../tests/fixtures/provenance.md)): the pinned reference
on `<|begin_of_text|>Hello,` (ids `[200000, 19873, 24]`; the harness
tokenizes with special tokens parsed, so BOS is the text marker), Metal,
F32 cache, one token per decode: 156 layer files (three positions × 52
layers × 6,656 floats) and the 202,048 logits, greedy token 372.

`make compare-muse-glimmer-cpu` (`nuclis generate --backend cpu
--prompt-tokens …`, then `scripts/compare-generation.py` at the bring-up
thresholds, max abs 2e-3 and relative RMS 1e-4 per file), 2026-09-19:

| Rows | Max abs | Max relative RMS | Greedy |
| --- | ---: | ---: | --- |
| 157 files (156 layers + logits) | 1.53e-4 | 6.35e-7 | 372 on both sides, top-5 identical, reference margin 0.31 |

The logits themselves differ by at most 6.0e-6 (relative RMS 4.2e-7).
The whole run, three CPU tokens through 52 layers with F64 accumulation
over the Q4_K/Q5_K weights, takes 1 min 41 s on the M4 Pro; the CPU path
is the reference, not a way to run the model.

## Metal plan (MODL-12, 2026-09-19)

`inference/src/models/muse_glimmer_metal.zig` records the forward pass
above as `Backend` encoder calls, one command buffer per token (`step`)
or per prompt chunk (`prefill`), the same shape as the Gemma plan. What
the schedule needed from the backend, and what it reused unchanged
([metal-backend.md](metal-backend.md)):

| Operation | Kernel | Status |
| --- | --- | --- |
| Projections, FFN, the untied head | `nu_matvec_q4_k`, `nu_matvec_q5_k`, `nu_matmul_*`, `nu_matvec_segments` | reused; the four attention projections (q, k, v, gate) merge into one segment dispatch, the FFN pair into the SiLU pair mode |
| Embedding norm without a weight | `nu_rmsnorm` with a row of ones | reused |
| Pre norms and q/k norms at 1e-5, post norms at 1e-8 | `nu_rmsnorm` (`Norm.eps`) | reused; the epsilon is per call |
| RoPE, adjacent pairing over the whole 128-wide head, base 5e5, sliding layers only | `nu_rope`, `nu_rope_rows` | **extended**: a `pairing` parameter (`Backend.Pairing`, the CPU's enum) selects `(2i, 2i+1)`; the table is the same for either pairing |
| Decode attention (32 over 2, width 128, scale 1/√128) | `nu_attention_decode` / `_h` | reused; the window is a cache-row slice (`firstVisible`), as in the CPU reference |
| Prefill attention | `nu_attention_chunk` / `_h` with `window = 2048` on sliding layers | reused (the MODL-06 window mask) |
| Attention output gate `o ⊙ sigmoid(g)` | `nu_sigmoid_gate` | reused (the Qwen3.5 gate epilogue; here the gate is its own projection, stride 128, offset 0) |
| Logit scale and soft-cap | `nu_scale`, `nu_softcap` | reused; the scale is one rounding before the tanh, where the reference folds it into the argument |

Every cache is allocated for the full session capacity, sliding layers
included (the CPU reference does the same): 52 × 2 × 256 halves per
position, 1.7 GB at 32,768 tokens with the F16 cache. A ring layout for
the 39 windowed layers is a session-layout unit of its own
(ENGN-19 in [TODO.md](../../TODO.md)).

**Against the pinned traces** (`make compare-muse-glimmer`, three
positions of `<|begin_of_text|>Hello,`, 157 files, 2026-09-19):

| Path | Max absolute | Max relative RMS | Threshold | Greedy / top-5 |
| --- | --- | --- | --- | --- |
| CPU reference | 1.53e-4 | 6.35e-7 | 2e-3 / 1e-4 | 372, identical |
| Metal, F32 cache | 1.53e-4 (layers), 6.0e-6 (logits) | 8.0e-7 | 2e-3 / 1e-4 | 372, identical |
| Metal, F16 cache | 7.3e-2 (layer 49 of token 0), 2.6e-3 (logits) | 1.97e-4 (layers), 1.0e-4 (logits) | 0.1 / 3e-4 (its own) | 372, identical, reference margin 0.31 |

The F16 row sets the family's tolerance: seventeen of the 156 layer files
sit above the Qwen bound of 3e-2 (all in the last eight layers, none
above 0.1) while every relative RMS stays under 2e-4 and the logits
within 2.6e-3. The kernels are the ones `test-metal` holds within 2e-4
of the CPU over the rounded operands; the residual stream of the late
layers is simply large. `--kv f32` is available for numerical work; the
default stays `f16` as for the other families.

**Generation check** (`make test-generation-muse-glimmer-metal`,
2026-09-19): sessions bit-identical, cancellation and reset through both
observer callbacks, snapshot/restore bit-exact (106,496 bytes at position
1); chunked prefill vs per-token steps on 70 tokens: chunks of 64 / 48 /
32 at 1.39e-2 / 1.26e-2 / 6.3e-3 max abs and 4.5e-3 / 4.0e-3 / 1.2e-3
relative RMS (recorded bounds 5e-2 / 1e-2), the generic F32 tiles at
3.7e-5 / 4.1e-6 (the schedule itself), the F16 cache stepped at 2.4e-3 /
2.6e-4 and chunked at 5.0e-3 / 6.0e-4 (bounds 2e-2 / 2e-3), argmax 75 of
75 on every path. The gap between the half tiles and the F32 tiles is the
specialized matmul's half-operand rounding (ENGN-05), as on Gemma.

**First-look rates** (`nuclis bench`, Metal, greedy, context 2,048, three
measured runs after one warmup; Apple M4 Pro 48 GB, Zig 0.16.0
ReleaseSafe, 2026-09-19; not the acceptance record, which MODL-13 takes
against the reference harness). The prompt is raw (`--raw`) because the
profile does not exist yet:

| Workload | Prefill tok/s | Decode tok/s | First token |
| --- | ---: | ---: | ---: |
| 10-token raw prompt (the `make bench` text through Muse's tokenizer), 64 out, `--kv f16` | 18.3 | 9.99 | 547 ms |
| same, `--kv f32` | 18.2 | 9.91 | 549 ms |
| 512-token array (the first 512 ids of `docs/spec.md`'s opening 6,000 bytes through Muse's tokenizer), 128 out, `--kv f16` | 93.2 | 9.59 | 5,492 ms |
| same, `--kv f32` | 93.0 | 9.52 | 5,504 ms |
| Reference `llama-bench` `7620399` on the same file (`-p 512 -n 128 -ngl 99 -fa 1 -ctk f16 -ctv f16 -r 3`) | 101.9 ± 0.1 | 14.08 ± 0.11 | — |

Decode is 71 % of the reference (15.87 GB of weights at 9.99 tok/s is
159 GB/s effective against the reference's 223) and prefill 91 %. The
Qwen `make bench` is unchanged by the shared-kernel change (39.75 /
10.44 tok/s the same day; its recorded spread is 10.36–10.47). The
decode gap is wider than Gemma's (81 %) or Qwen's; where the time goes
is the performance theme's question (KERN-14 in [TODO.md](../../TODO.md)), with
the per-kernel profile below as its starting point.

**Per-kernel profile** (`make bench-profile MODEL=<file> ARGS=--raw`,
the raw 10-token prompt, 64 output tokens, 192 measured decode steps;
profiling itself costs 10 %: 9.07 tok/s profiled against 9.99). 922
dispatches per token, 107.5 ms attributed of 111.6 ms of command-buffer
time:

| Kernel | Shape (rows × columns) | Dispatches/step | ms/step | Share | GB/s |
| --- | --- | ---: | ---: | ---: | ---: |
| `matvec_segments` (FFN gate + up, SiLU pair) | 39,936 × 6,656 Q4_K | 51.2 | 43.8 | 40.8 % | 174.6 |
| `matvec_q4_k` (FFN down) | 6,656 × 19,968 | 51.2 | 26.1 | 24.2 % | 146.9 |
| `matvec_segments` (q, k, v, gate) | 8,704 × 6,656 Q4_K | 51.2 | 11.5 | 10.7 % | 144.6 |
| `matvec_q4_k` / `matvec_q5_k` (attention output) | 6,656 × 4,096 | 51.2 | 5.2 | 4.9 % | 151.8 / 167.0 |
| `matvec_q5_k` (the untied head) | 202,048 × 6,656 | 1.0 | 4.4 | 4.1 % | 207.9 |
| `rmsnorm` (six per layer, the embedding and output norms) | — | 314 | 4.3 | 4.0 % | — |
| `attention_decode_h` + `attention_merge` | 32 over 2, ≤ 74 visible | 51.2 + 51.2 | 2.8 | 2.6 % | — |
| everything else (`add`, `rope`, `sigmoid_gate`, `pack_half`, the prompt's matmul tiles amortized) | — | — | 9.4 | 8.7 % | — |

The four attention projections merge into one segment dispatch only
because the plan packs the query with the gate and the scratch key with
the value: the segment kernel binds seven buffers, and four weights with
four separate outputs need nine (unmerged, the 256-row k and v matvecs
ran at 52 GB/s; merging them moved decode from 9.88 to 9.99 tok/s). The
matvecs read at 145–175 GB/s where the head's Q5_K rows reach 208 and
the reference averages 223 across the whole token, so the gap is spread
over the large Q4_K matrices rather than sitting in one kernel; that is
the performance theme's starting point, not this unit's.

## Status

MODL-11 closed on 2026-09-19: the artifact is pinned and verified, the
facts recorded, the `llama4` splitter native and matched to the
reference, the adapter binds the file (`nuclis model inspect` says
*supported*), and the CPU reference matches the oracle traces at the
bring-up thresholds with the same greedy token. MODL-12 closed the same
day: the Metal plan matches the traces in both cache precisions, passes
the generation check, and runs the file at 9.9 tok/s. MODL-13 closed the
same day with the profile (`profiles/muse_glimmer.zig`, the channel-grammar
decoder, the `high` effort), the catalogue pin, and the acceptance record
([bench.md](bench.md#muse-glimmer-30b-acceptance-record-modl-13-2026-09-19):
9.60 tok/s decode at 512 tokens, 6.62 at 32,639, against the reference's
13.69 and 9.98), and AGNT-10 with the ATEM tool protocol
([tool-calling.md](tool-calling.md#muse-glimmer-atem-calls-as-their-own-messages)).
The family is complete for text; the vision projector and the DFlash
drafter are planned (MODL-23 and MODL-20 in [TODO.md](../../TODO.md)).
