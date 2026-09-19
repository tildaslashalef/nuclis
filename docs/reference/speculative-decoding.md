# Speculative decoding: recovery contract, draft sources, measurements

The reference document of the speculative-decoding theme. Its requirements
are in the [spec](../spec.md#speculative-decoding); the units that fill it
in are planned in [TODO.md](../../TODO.md) (ENGN-11, MODL-18, ENGN-12,
MODL-19, MODL-20); closed outcomes are cited from the
[engineering log](../engineering-log.md).

As of 2026-09-19 the recovery contract is implemented and measured
(ENGN-11). What exists: the Qwen adapter binds and validates the 15
embedded `nextn` tensors without executing them
([qwen-validation.md](qwen-validation.md)); the session has a host-side
snapshot and restore and, now, an in-block checkpoint
([session.md](session.md)); every family's draft companion is pinned and
pulled ([artifacts.md](artifacts.md)).

Sections to come, one per unit: the draft contract, each family's draft
source with its facts and provenance, and the measurements behind each
catalogue verdict.

## The recovery contract (ENGN-11)

A verify batch feeds the main model the last chosen token followed by `k`
drafts in one forward and keeps every row's logits. If `a` drafts are
accepted, the session must end at the state after them — the accepted
prefix. The two state kinds recover differently:

- **Attention caches rewind by position.** A row's content never depends on
  later rows, so `truncate(P + a + 1)` sets the position and nothing is
  copied; rows past it are ignored by contract. This is what Gemma 4 and
  Muse Glimmer (attention-only) use.
- **Recurrent state is replayed.** Qwen 27B and Bonsai 2 hold 48 DeltaNet
  layers whose matrices are a function of every token fed. The batch's
  checkpoint is taken before the forward; on partial acceptance the session
  rewinds to it and re-runs the accepted prefix as a second batched
  forward, which rewrites the same attention rows. Truncating the position
  alone would leave the matrices at `P + k + 1`.

The session owns the mechanism: a page-aligned region inside the byte
block, one recurrent copy, `checkpoint`/`rewind`/`truncate` with their
refusals, all described in [session.md § Checkpoint and
rewind](session.md#checkpoint-and-rewind-engn-11). `engine.Model.recover`
is the accepted-prefix operation, and `generation-check` exercises it per
accepted length on every family, on both executors: replay is bit-identical
to sequential decoding on the CPU, and within the family's
chunk-versus-step bound on Metal (`make test-generation` /
`test-generation-metal`, recorded in the log).

Measured costs on Qwen 27B (2026-09-19): region 156,893,184 bytes;
checkpoint and rewind 3 ms per batch on Metal and 2 ms on the CPU
reference (one 150 MB copy each way); the replay is a short-chunk prefill
(KERN-11's tile). Losing `k − a` drafts
therefore costs one 150 MB copy plus an `a + 1`-token prefill, not a
context-long replay. A bounded alternative — per-token recurrent
checkpoints written by the DeltaNet chunk kernel, `(k + 1) × 150 MB` of
device scratch on Qwen — was not needed by these numbers and is not built;
the decision and its measurement are here for the next revisiter.

## The Qwen3.8 draft head (MODL-18)

Facts confirmed on 2026-09-19 from the pinned reference
(`7620399f5`): the dense MTP graph `src/models/qwen35.cpp:485-639`, the
driver `common/speculative.cpp:1324-1760`
(`common_speculative_impl_draft_mtp`), and the example loop
`examples/speculative-simple/speculative-simple.cpp`.

**The block.** The main file's 65th block is one dense full-attention
decoder layer of the main model's shape plus four `nextn` tensors. There
is no separate draft model: the reference opens the *same* file as a
second context of type `LLAMA_CONTEXT_TYPE_MTP`
(`llama_context_params.ctx_type`), which executes only that block with its
own full-attention cache (`llama_set_embeddings_nextn(ctx, true, true)`).
The separate `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` adds only lower-quantized
copies of `token_embd`/`output`/`output_norm` under `nextn.embed_tokens` /
`nextn.shared_head_head` / `nextn.shared_head_norm`; the main file's block
carries `shared_head_norm` and falls back to the main `token_embd` and
`output` (`qwen35.cpp:518,621-635`). The embedded block is therefore the
source.

**The pair.** At MTP position `p` the block consumes the token `x_p` and
the target hidden `h_{p-1}`: `concat = [enorm(embed(x_p)); hnorm(h_{p-1})]`
along the feature axis (`e_norm` first, `h_norm` second), projected by
`nextn.eh_proj` `[2·5120 -> 5120]` (`qwen35.cpp:539-549`,
`ggml_concat(..., dim=0)`). `h` is the target's `t_h_nextn`, i.e. the
final hidden **after** `output_norm` and the input to the target's own
head (`qwen35.cpp:204-209`), not the pre-norm residual; the block applies
its own `hnorm` on top. In our runtime this is exactly the `self.normalized`
row produced by `norm(self.x, self.normalized, self.output_norm)` before
`self.mm(self.binding.output, ...)`.

**The layer.** `attn_norm`; `attn_q` is the merged query+gate `[5120 ->
12288]` with each head storing `[query(256); gate(256)]`; `attn_q_norm` /
`attn_k_norm` are applied per head before RoPE; `attn_k`/`attn_v` are
`[5120 -> 1024]` (4 KV heads, Q8_0 in the file, a supported encoding);
24 query heads, key/value width 256, RoPE base 1e7 with sections
`[11, 11, 10, 0]`; the attention output is multiplied elementwise by
`sigmoid(gate)` and projected by `attn_output` `[6144 -> 5120]`; residual
to the `eh_proj` output; `post_attention_norm`; dense FFN `ffn_gate` /
`ffn_up` `[5120 -> 17408]`, `ffn_down` `[17408 -> 5120]`, `silu(gate)·up`;
residual; `shared_head_norm`; the shared `output` head.

**Drafting.** `draft()` seeds the block at `n_past` with token `id_last`
and the carried `pending_h` (the target `h` of the last committed token),
decodes one row, and takes the greedy top-1 (sampler `top_k = 10`,
candidate 0) as the first draft; each later step feeds the drafted token
and the block's own output hidden `h_nextn` as the next `h`, at position
`n_past + i + 1`. Drafting stops at `n_max` or when the top-1 probability
falls below `p_min` (`speculative.cpp:1596-1745`).

**Advancing the cache.** After the target decodes `[id_last, drafts...]`,
`process()` runs the block over the whole batch (token + shifted target
`h`: row `k` gets `h` of the previous row; the first row gets `pending_h`),
writing its cache rows at the batch positions. It stores every target `h`
row for acceptance and carries the last one as `pending_h`
(`speculative.cpp:1478-1594`). On partial acceptance the target and the
draft cache are both rewound and `accept(n)` sets `pending_h` to the
target `h` at index `n` (`speculative.cpp:1747-1760`). The block's own
output hidden is used to chain drafts only; the cache is filled from the
target's `h`, never the block's.

**Which encoding.** The block's `attn_k`/`attn_v` are Q8_0; `qwen35`'s
`executableEncoding` already admits id 8 (generic matvec and generic F32
tile), and the binder validates every block tensor's shape and encoding
before it is bound.
