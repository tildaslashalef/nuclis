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
Facts the fixture capture confirmed (the profile is MODL-13, the tool
protocol AGNT-10; the plan in `TODO.md` holds their designs):

- The prompt opens with `bos_token` as text, but the server's
  `/apply-template` output omits it (its tokenizer adds BOS): the fixture
  prompts start at `<|start|>`, as Gemma's do. nuclis's encoder never adds
  BOS, so the profile writes `<|begin_of_text|>` itself.
- Turns are `<|start|>ROLE<|message|>…<|eot|>`; two consecutive messages
  of one role end the first with `<|eom|>`. The system turn is the
  caller's text, `\n\nReasoning strength: LEVEL.`, then
  `\n\n# Valid recipients: "self", "user".` (tool namespaces between). The
  level is the `reasoning_strength` template variable (default `high`, no
  off); the fixture captures `low`, `medium`, `high`, `xhigh`.
- Without a system message the template synthesizes one: `You are a
  helpful AI assistant.`, `Knowledge cutoff: 2026-01-04.`, and, because the
  server defines `strftime_now`, `Current date: 2026-09-19.` on capture
  day (the fixture's `single_*`, `history_*`, `continued_*`, `unicode_*`,
  and `empty_*` cases carry that line). The profile decision of
  2026-09-16 renders the default without the date line; the MODL-13 test
  must account for the captured line.
- A `developer` message renders nothing (the template has no branch for
  the role); a later `system` message renders as a second system turn.
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

## Status

Session 1 of MODL-11 (2026-09-19): artifact pinned and verified, facts
recorded, the `llama4` splitter native and matched to the reference on
the fixture, the inventory fixture committed. The adapter
(`models/muse_glimmer.zig`), the CPU reference schedule, the `Hello,`
traces, and `make compare-muse-glimmer-cpu` are session 2; `nuclis model
inspect` still says *not runnable: no adapter*.
