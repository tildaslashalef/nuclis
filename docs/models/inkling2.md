<!-- kind: card -->
# Inkling 2 Mini

Thinking Machines' looped mixture of experts (`general.architecture =
inkling2`): 24 stored blocks run as 48 layer applications, 128 experts
(top 8) per block, sliding and global attention without RoPE, a learned
relative-position bias, short convolutions, and attention over the depth
in place of a plain residual stream. The file is Unsloth's
`unsloth/Inkling-2-Mini-GGUF` `Inkling-2-Mini-UD-Q4_K_XL.gguf` (17.6 GB),
with a vision projector (`mmproj-F16.gguf`, projector `inkling2v`) and a
DSpark drafter (`dspark-Inkling-2-Mini-Q8_0.gguf`, architecture
`dflash`); commits and digests in
[the record](../benchmarks/record/catalogue.md#pinned-commits-and-digests-2026-09-11).
The header's facts are pinned in
`inference/src/models/fixtures/inkling2-q4_k_xl.json`. The CPU reference
([inkling2_runtime.zig](../../inference/src/models/inkling2_runtime.zig))
and the Metal plan ([§ Metal](#metal)) run it.

## Architecture

From the file's 35 `inkling2.*` keys:

| Fact | Value |
| --- | --- |
| Width | 2,304; every norm RMS with ε 1e-6 |
| Attention | 24 query over 8 KV heads of 128 on every application; no RoPE (`rope.dimension_count` 0) |
| Layout | 48 applications, sliding window 1,024; global at `il mod 6 = 5` (5, 11, …, 47) |
| Loop | `num_loops` 3, core stored blocks 6–17, controller rank 64 |
| Application → stored | `il < 6` → `il`; `6 ≤ il < 42` → `6 + (il − 6) mod 12`; `il ≥ 42` → `il − 24`; the global applications land on stored 5, 11, 17, 23 |
| Depth mix | blocks of `attn_res.block_size` 6 |
| Relative bias | 16 coefficients a head, extent 1,024 on both kinds; clip 8 |
| Clips, scales | `qk_clip` 22.627 (16√2); log scale floor 8,192, α 0.1 |
| Short convolution | kernel 4 (`shortconv.l_cache`), history 3 rows |
| Experts | 128, top 8, 1,152 wide; router F32, sigmoid gating with a selection bias; weights scale 8 |
| Activation | SITU, β 4, linear β 25 |
| Head | untied `output`, soft cap 30; 201,024 rows, of which the first 200,058 are the vocabulary |
| Context | 1,048,576 declared |

## Forward pass

Notation: `rms(x) = x / √(mean(x²) + 1e-6)`; `σ` the logistic function;
`c = 1/√128`. A stored block's tensors are `blk.<ip>.*`, `ip` the stored
index of application `il`.

1. **Embedding.** `x = rms(E[t]) · token_embd_norm` for a token row; an
   image row enters as is. No √width factor.
2. **Depth mix.** `mix(S, Q)` attends over the list of states `S` (the
   closed blocks, then the current partial `cur`): keys and values are
   `K = rms(S)` (no gain), and each query row `q` of `Q` gives
   `Σ_s softmax_s(q·K_s / √2304) · K_s`. `Q` is one row, or three
   (q, k, v inputs) on a global layer's `attn_res_query`.
3. **Attention inputs.** For `il > 0`, `in = mix(blocks ∪ {cur},
   attn_res_query)` (`blk.0` has no query: its inputs are `cur`). In
   passes 2 and 3 of the core, each input first goes through the loop
   controller (item 6). Norms: a sliding layer applies `attn_norm` to its
   one input, or to each of its three when the controller produced
   three; a global layer applies `attn_{q,k,v}_inp_norm`.
4. **Attention.** With `in_q, in_k, in_v` the normed inputs:
   - `q = Wq in_q`, `k = conv(Wk in_k)`, `v = conv(Wv in_v)`,
     `r = Wrel in_q` (24 × 16), `g = Wgate in_q`, `t = Wsink in_q` (24).
     A global layer adds dynamic taps `δk = Wk_conv_proj in_k` and
     `δv = Wv_conv_proj in_v` (8 heads × 4).
   - **Short convolution**, per channel over the token's own sequence:
     `conv(x)[τ] = (x[τ] + Σ_{i=0}^{3} w[i]·x[τ−3+i] + Σ_{j=0}^{3}
     δ[h][j]·x[τ−j]) / √(n + 1)`, `n` = 4 without dynamic taps, 8 with
     them. The static taps run oldest first (`w[3]` multiplies the current
     row), the dynamic taps newest first; `h` is the channel's KV head.
     Rows before the sequence's first are zero.
   - **Clips**, per head row: `clip(x, a) = x · min(1, a / (‖x‖ + 1e-6))`.
     `q ← clip(q ⊙ attn_q_clip[h], 22.627)`, `k ← clip(k, 22.627)`,
     `r ← clip(r ⊙ attn_rel_clip[h], 8)`. Then `v ← rms(v)` per head, no
     gain.
   - **Log scale**, global layers only: `λ = 1 + 0.1·ln(max(1, (pos +
     1)/8192))` at the query's position; `q ← λq`, `r ← λr`.
   - **Scores**: `s = c·q·k + b`, `b = r·B[:, d]` for `d = pos_q −
     pos_k` with `0 ≤ d < 1024`, else `b = 0`; `B` is `attn_rel_bias`
     (16 × 1,024). Causal; a sliding layer also masks `d ≥ 1024`.
   - **Softmax with a sink**: `p = exp(s) / (exp(t_h) + Σ exp(s))`; the
     sink takes probability mass and contributes no value.
   - **Output**: `o = (p·V) ⊙ σ(g)`; `a = convpost(Wo o)` with
     `attn_post_conv` (static taps only).
5. **Residuals.** At a block start (`ip mod 6 = 0`) the layer's input
   `cur` is appended to `blocks` and `cur ← a`; otherwise `cur ← cur + a`.
   The FFN input is `f = rms(mix(blocks ∪ {cur}, ffn_res_query)) ·
   ffn_norm`, with the mlp controller applied to the mix before the norm
   in passes 2 and 3. Then `cur ← cur + convffn(moe(f))` with
   `ffn_post_conv`.
6. **The loop.** Passes run the core (stored 6–17) three times. Before
   each pass after the first, the bank is the previous pass's `blocks`
   past the prelude and its first core block (one state: the block opened
   at stored 12), and `bank_last` its final `cur`; then `blocks` and
   `cur` return to the prelude's (as they were entering stored 6). For
   core block `j = ip − 6` the messages are `mix(bank ∪ {bank_last},
   loop_res_query rows 2j, 2j+1)` → (attn, mlp) and the same with
   `loop_kv_res_query` → (k, v). The controller of module `m` (attn, k,
   v, mlp): `h = Wdown [rms(x); rms(msg)]` (4,608 → 64), `x ← (1 −
   tanh(Wforget h)) ⊙ x + tanh(Wgain h) ⊙ msg`.
7. **Experts.** Router logits `z = W_gate_inp f` in F32; select the top 8
   of `σ(z) + exp_probs_b`; weights `σ(z_e)` of the chosen, divided by
   their sum plus 1e-20, times 8. Expert `e`: `[gate; up] =
   W_gate_up[e] f · s_gu[e]` (rows 0–1,151 gate, 1,152–2,303 up; one
   scale for both), `y_e = W_down[e] SITU(gate, up) · s_down[e]`, with
   `SITU(g, u) = 4·tanh(g/4)·σ(g) · 25·tanh(u/25)`. The scales
   `ffn_*_exps.scale` exist on stored 0–19 only and are 1 elsewhere.
8. **Head.** `y = rms(mix(blocks ∪ {cur}, output_res_query)) ·
   output_norm`; logits `30·tanh((Wout y)/30)`, the first 200,058 kept.

### Session state

- 48 KV caches, one per application: 8 full, 40 windowed at 1,024; F16 K and V take 4 KiB a token per application (1 GiB at 32,768 tokens).
- Per application, a convolution history of 3 rows × 6,656 F32 channels:
  K 0–1,023, V 1,024–2,047, attention output 2,048–4,351, FFN output
  4,352–6,655 (78 KiB; 3.7 MiB for the 48).
- That history is recurrent, so the session is hybrid ([spec § 5.4](../spec.md#54-session-contract)):
  a rewind restores a checkpoint, verify recovers by rewind and replay,
  `truncate` and `rewindTo` refuse. Attention layer `il`, recurrent
  `48 + il`, whose matrix region is one unread value (`Session.init`
  refuses an empty one); the depth residual is rebuilt from each token.

### Readings confirmed

On the real file each alternative reading (taps reversed, bias and sink
inside `c`, the sink's sign, up before gate) moved all 48 layers an order
of magnitude further from the format's graph. The sliding bias table has
the window's columns; `relative_extent` is the same 1,024.

## Files and encodings

734 tensors, none Q4_K despite the name. By encoding:

| Encoding | Tensors | What |
| --- | ---: | --- |
| F32 | 508 | norms, router, depth and loop queries, conv taps, relative tables and clips, loop controllers, expert scales |
| Q8_0 | 154 | every projection, `token_embd`, `output`, and the experts of stored 20–23 |
| NVFP4 (ggml id 40) | 40 | the experts of stored 0–19, each with a per-expert F32 `ffn_{gate_up,down}_exps.scale [128]` |
| BF16 | 32 | `attn_sink_proj`, `attn_{k,v}_conv_proj` |

- **NVFP4 block**: 64 values in 36 bytes, four UE4M3 scales then 32 bytes
  of E2M1 codes ([quantization.md](../engine/quantization.md)); no scale
  byte is `0x00`, `0x7F`, or has bit 7 set. Trained in NVFP4 (MXFP8 on the
  last four blocks), the rest is the converter's Q8_0 requantization, so
  nuclis validates the file against its own CPU reference only.
- **Binding** ([inkling2.zig](../../inference/src/models/inkling2.zig)):
  every tensor by name, shape, and encoding; block 0 has no
  `attn_res_query`, the core alone has loop controllers, and a block has
  both expert scales or neither.

## Tokenizer

The `gpt2` model with `pre = inkling2`: 200,058 tokens (60 control),
424,413 merges, EOS and EOT 200006, padding 199999, no BOS. The splitter,
[o200k.zig](../../inference/src/tokenizer/o200k.zig), is the checkpoint's
own expression with Unicode's letter classes
([tokenizer.md § Full text encoding](../engine/tokenizer.md#full-text-encoding)),
not `llama4`'s folded rewrite: `ÇA` is one token (118428)
([record](../benchmarks/record/prompt-profile.md#inkling-2s-tokenizer-and-template-2026-11-26)).

## Prompt profile

[inkling2.zig](../../inference/src/profiles/inkling2.zig), for the
template `53a9035f…` (6,868 bytes): messages `ROLE CONTENT-TYPE text
<|end_message|>` (`<|message_{user,model,system,tool}|>`, `<|content_{text,
thinking,invoke_tool_json,xml,image}|>`); a model turn ends with the stop
token `<|content_model_end_sampling|>` (200006).

- **Effort**: a system message `Thinking effort level: X` first (off `0`,
  low `0.2`, medium `0.7`, high `0.9`, xhigh `0.99`; default `0.9`), so
  part of `prefix`; a developer message renders as a system one.
- **No generation prompt**: the model writes `<|message_model|>` before
  each segment, and the channel grammar routes each body by its content
  type at every effort; a reasoning budget cuts with `<|end_message|>`.
- **Tools**: JSON in messages of their own
  ([tool-calling.md § Inkling 2](../engine/tool-calling.md#inkling-2-json-calls-in-their-own-messages)).
  **Images**: a `<|content_image|>` user message each, one
  `<|unused_200054|>` per merged row, before the text.
- **Sampling**: temperature 1, top-k 0, top-p 1, min-p 0. Fixtures:
  `inkling2-{text,tools,media}.json` (9 prompts with ids and 11 strings, 5, 2).

## Metal

[inkling2_metal.zig](../../inference/src/models/inkling2_metal.zig) steps
each token in one command buffer, stage for stage the CPU reference's. On
the real file its F32 trace is 7.6e-5 (relative RMS 8.7e-7) from the
reference's; rounding the reference's own cache to F16
(`Runtime.cache_rounding`) moves it 2.57 / 0.082, Metal's F16 cache 3.0e-3
/ 8.8e-5 from that. The F16 gate bounds the rounding; the packed path, the
window, the log scale, and Q8_0 are held by `make test-tiny` alone.

## What is verified

The CPU reference agrees with the format's definition to 5 % relative RMS
at every application of an 8-token prompt, the largest errors where the
router's eighth expert differs at a near tie
([record](../benchmarks/record/inkling2.md)); its own trace is pinned.

<!-- gen:gates-for:inkling2 -->
| Gate | Tier | Model | Evidence |
| --- | --- | --- | --- |
| `inkling2-vocabulary` | `verify` | inkling2 | [inkling2.md § prompt-profile](inkling2.md#prompt-profile) |
| `inkling2-trace-cpu` | `verify-cpu` | inkling2 | [inkling2.md](inkling2.md) |
| `inkling2-trace-f32` | `verify` | inkling2 | [inkling2.md § metal](inkling2.md#metal) |
| `inkling2-trace-f16` | `verify` | inkling2 | [inkling2.md § metal](inkling2.md#metal) |
<!-- /gen -->

## Known limits

- Only the pinned commit binds; `main` (2026-11-26) re-uploads it as `inkling2_loop`.
