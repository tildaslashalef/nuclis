# Roadmap beyond v0.1

Work that is accepted in principle but not active. When a theme starts, its
design moves into [../TODO.md](../TODO.md) as an `AREA-NN` unit; closed
outcomes go to [engineering-log.md](engineering-log.md).
What a unit must satisfy lives in the [spec](spec.md), and the spec's
[deferred list](spec.md#local-serving-and-deferred-work) remains the scope
boundary.

Themes, in order (reordered 2026-09-17, once the third family was planned;
the Bonsai 2 27B units — Qwen3.8-27B in Prism ML's ternary encoding, a
new weight format on the existing adapter — closed on 2026-09-18, and the
Muse Glimmer 30B units, MODL-11 to MODL-13 and AGNT-10, on 2026-09-19).
Speculative decoding across the families, the first theme of that order,
moved into [../TODO.md](../TODO.md) on 2026-09-19 as ENGN-11, MODL-18,
ENGN-12, MODL-19, and MODL-20; its accepted configuration lives in the
[spec](spec.md#speculative-decoding). What remains here:

1. **Performance follow-ups** — measured experiments on text generation,
   not assumptions; text decoding and prefill are finished before vision
   starts.
2. **Vision through the companion projectors** — one contract, three
   projectors, its own milestone.
3. **Agent expansion** — HTTP serving on the closed agent loop.

## Performance follow-ups

Each is a measured experiment, not an assumption; the linked document holds
the profile that motivated it.

- **Long-context prefill attention, second attempt.** The 32K acceptance
  record measured prefill below the reference and the profile put the chunk
  attention kernel at 30% of the 16K prefill, running at 0.7 TFLOP/s.
  Sharing the cache tiles across the six heads of a KV group does not help
  (three variants, all slower; the re-reads were cache hits). The kernel is
  latency-bound with one `simdgroup_load` per multiply; the lever left is
  register-level reuse as in the matmul tiles: split the value columns across
  the four SIMD groups of a (head, 32-query) tile so each V block serves four
  row blocks, share the probability tiles through threadgroup memory, and
  study the register budget (48 live matrices per SIMD group) before writing
  it ([metal-backend.md](reference/metal-backend.md)). Measure at 16K and
  32,639.
- **A ring layout for windowed attention caches.** Gemma 4's 40 sliding
  layers see 1,024 positions but their caches are allocated for the full
  session capacity on both backends (11.3 GB of a 32K F16 session, of which a
  ring would keep 0.35 GB). A `session.Layout` variant with a modulo row index
  touches the CPU reference's slice, the decode slice, the chunk kernel's key
  loop (wraparound inside a chunk), and `snapshot`/`restore`; a session-layout
  unit with its own fixtures ([gemma4.md](reference/gemma4.md)).
- **Gemma 4 12B decode and prefill against the reference.** On the catalogue's
  QAT file decode is 78–87% of the reference and prefill 80% at 512 falling to
  51% at 32,639, with nothing tuned for Gemma
  ([bench.md](reference/bench.md)). The first per-kernel profile says where
  decode's 5.6 ms/step gap is not: the Q4_0 matvecs hold their isolated
  bandwidth and are 85% of the step. What is left, in order: the RMS-norm
  launches per step (launch-bound; a fused norm-and-scale or a batched launch
  is the experiment), the matvecs' own share of the bus, the Q4_0 prefill tile
  (two 16-value segments per block, so a staged block decode or a wider
  segment is the first experiment), the long-context chunk attention above,
  the tied Q4_0 head, the sliding decode, and the wide global-layer attention.
  A prefill tile that keeps activations in F32 belongs here too: the QAT
  checkpoint amplifies half-operand rounding about five times more than the
  K-quant file, which is precision, not correctness
  ([gemma4.md](reference/gemma4.md)).
- **The ternary matvec's arithmetic.** Bonsai 2 27B decodes at 1.3× the
  Qwen3.8-27B rate where its byte count promises 2–3×, and at 80–87 % of
  the PrismML fork: the PQ2_0 / PTQ1_0 matvecs move about 100 GB/s of
  weight bytes because they are at the kernel set's multiply-rate ceiling
  (one field mask per four values, one integer-to-float conversion, one FMA
  per value; a ternary byte carries twice the values of a 4-bit byte). The
  experiments are packed integer products (several trits against several
  inputs in one integer multiply-add) and sharing one decoded weight across
  several inputs; the transform and the gather are 2.3 % of the step and
  not the lever. Judge against the acceptance record
  ([bench.md](reference/bench.md#bonsai-2-27b-acceptance-record-modl-17-2026-09-18),
  [metal-backend.md](reference/metal-backend.md#ternary-matvecs-and-tiles-kern-10-2026-09-18)).
- **Muse Glimmer 30B decode against the reference.** The acceptance
  record measured decode at 66–71 % of the reference at every length,
  the widest gap of the four families, and the per-kernel profile says
  where it is: the matvecs' bandwidth tracks the matrix's row count, not
  its encoding — the 202,048-row head streams at 208 GB/s, the 39,936-row
  FFN pair at 175, and the two shapes with 6,656 to 8,704 rows (the FFN
  down projection over 19,968 columns, the merged q/k/v/gate) at 145 to
  147 — because a matrix with few rows launches too few threadgroups to
  keep the bus busy, and Muse's 6,656-wide residual puts more than a third
  of its bytes in those shapes. At the head's rate the token would take
  84 ms instead of 100, so that is two thirds of the gap; the rest is
  launch count, 922 dispatches per token with six RMS-norm launches per
  layer (about 9 ms). Three experiments, in order, each judged against the
  acceptance record: split-K on the decode matvec for shapes below about
  16K rows (several threadgroups per row summing column ranges, reduced at
  the end, the idea the short-prompt bullet above proposes for the matmul);
  fusing the post norms into the residual add and the q/k head norms into
  the RoPE launch; then, only if the first two move the 32K row, the
  flash-decoding split pass over 2 KV heads on the 13 global layers, whose
  parallelism per layer is the lowest of the families
  ([muse-glimmer.md](reference/muse-glimmer.md#metal-plan-modl-12-2026-09-19),
  [bench.md](reference/bench.md#muse-glimmer-30b-acceptance-record-modl-13-2026-09-19)).
  The 4K row's decode drift within one run (9.14 to 8.12 tok/s in four
  minutes) is to be reproduced first, with the GPU's clock watched, before
  any of them is measured.
- **A GPU penalty kernel.** Apply the token history to the logits on the
  device before `nu_topk_partial`, so the instruct profile (presence penalty
  1.5) returns to the GPU sampling path. The 32K record measured the cost of
  the full readback it pays today at +20.7 ms/token
  ([bench.md](reference/bench.md)); a measured follow-up, not part of v0.1.

## Vision through the companion projectors

Every catalogue entry carries its projector, pulled and pinned: Qwen3.8's
`mmproj-BF16.gguf` (`clip`, `qwen3vl_merger`), the Gemma 4 12B and 26B-A4B
`mmproj-BF16.gguf`, and Muse Glimmer's `mmproj-kquant.gguf`. Vision is its
own milestone after text generation is finished and sped up; the agent gets
no image tool in it.

**Design.** Load `mmproj-*.gguf` as a second GGUF (its own architecture,
validated separately), image preprocessing bounded by host constants, the
vision encoder and projector on Metal, and image tokens spliced into the
prompt by the model's profile: one loading and splicing contract, three
projector architectures brought up in turn (Gemma first, whose text path
is the most exercised). CLI: an image argument on `generate` and a chat
attachment.

**Acceptance.** Projector output vs a pinned reference trace per family; one
end-to-end captioning fixture each; memory recorded.

## Agent expansion

The agent loop, tools, and tool-call handling are specified in
[agent-spec.md](agent-spec.md) and closed for the current phase. The next
expansion is HTTP serving with an
[OpenAI-compatible chat-completions protocol](agent-spec.md#relationship-to-external-agent-products),
under which the tool-call parser lands; it reuses the closed loop. Two
decisions bound it: the profile owns tool syntax, and there is no permission
system — supervision is visibility plus the workspace boundary.

## Also deferred

- Teacher-forced `eval` command ([spec § CLI](spec.md#cli-proposal)).
