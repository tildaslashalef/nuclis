# Roadmap beyond v0.1

Work that is accepted in principle but not active. When a theme starts, its
design moves into [../TODO.md](../TODO.md) as an `AREA-NN` unit; closed
outcomes go to [engineering-log.md](engineering-log.md).
What a unit must satisfy lives in the [spec](spec.md), and the spec's
[deferred list](spec.md#local-serving-and-deferred-work) remains the scope
boundary.

Themes, in order (reordered 2026-09-17, once the third family was planned;
the Gemma 4 26B-A4B and Muse Glimmer 30B units themselves are in
[../TODO.md](../TODO.md)):

1. **Speculative decoding across the families** — draft tokens from each
   model's own draft source (Qwen's MTP head, Gemma's MTP companion, Muse's
   DFlash drafter) verified by the main model under one recovery contract.
2. **Performance follow-ups** — measured experiments on text generation,
   not assumptions; text decoding and prefill are finished before vision
   starts.
3. **Vision through the companion projectors** — one contract, three
   projectors, its own milestone.
4. **Agent expansion** — HTTP serving on the closed agent loop.

## Speculative decoding across the families

A draft source proposes cheap tokens; speculative decoding is the protocol
that verifies those drafts with the main model and commits only accepted
state. The two are separable: the draft source is model-specific (an MTP
head predicts the next few tokens from the main model's hidden state; a
DFlash drafter is a small block-diffusion model that proposes a block at
once), the verification and recovery protocol is not. Its gains complement
kernel specialization but depend on acceptance rate, verification cost, and
recovery cost; no speedup is assumed and each family is measured on its
own. The spec lists MTP and speculation as
[deferred from v0.1](spec.md#local-serving-and-deferred-work). Every family
in the catalogue now has its draft companion pulled and pinned
([reference/artifacts.md](reference/artifacts.md)); the `mtp` companion
role names the draft source the unit loads, whatever its mechanism.

The pinned main GGUF already contains 15 auxiliary prediction tensors (one
block, 351,008,768 stored bytes), validated but not executed. The upstream
[MTP directory](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/main/MTP)
also lists `mtp-Qwen3.8-27B-Q4_0.gguf` (1.37 GB as listed on 2026-09-08). The
separate head is available locally under `~/.nuclis/models/qwen/`. Inspect its
metadata and required shared tensors before deciding whether it is needed;
prefer the existing embedded weights if their contract is sufficient. A
separate file is not an implementation of MTP, and is not yet a dependency.

### Speculative state recovery

**Design.** Define checkpoint, verify, accept-prefix, and reject semantics for
both attention and DeltaNet state. A checkpoint at the start of a speculative
batch must cover recurrent matrices, convolution history, position, and any
MTP state. Preserve committed attention rows; discard unaccepted appended row.
Start with checkpoint/restore plus replay of the accepted prefix for
correctness, then measure whether per-token recurrent checkpoints or another
bounded recovery scheme are needed. GPU work must complete before restoring
host-visible state. This extends the KV-cache checkpoint/restore contract
across a draft batch.

**Acceptance.** Reject at every possible draft position; subsequent state and
logits match ordinary sequential execution within the documented backend
contract. Test complete acceptance, immediate rejection, cancellation, errors,
context boundaries, allocation failures, and independent sessions. Record peak
memory and recovery cost at short and long context. Never rewind DeltaNet by
truncating KV length alone.

### MTP prediction reference and Metal execution

**Design.** Inspect the embedded auxiliary block and establish its precise
weight mapping, hidden-state inputs, token inputs, positional handling, and
state layout from authoritative architecture sources and pinned reference
traces. Implement the CPU reference first, with fixtures for each operation,
then the Metal prediction path. Keep MTP architecture semantics in the Qwen
adapter; expose draft candidates through a model-independent runtime contract
that the Gemma companion head and the Muse drafter (below) implement in turn.
Any external artifact becomes a pinned, explicitly requested dependency only
if inspection proves the embedded weights insufficient (for Gemma 4 the
companion file is the only source).

**Acceptance.** CPU and GPU draft logits match the pinned reference with stated
tolerances at several positions. Tests cover draft state reset and recovery.
Record draft latency, extra memory, and acceptance statistics on fixed coding
prompts. This unit alone does not claim speculative speedup.

### Batched verification and speculative generation

**Design.** Add a verification interface returning main-model logits for every
draft position (chunked prefill currently needs only the final logits). Verify
a small configurable draft batch using the existing chunked kernels. Start
with greedy acceptance: accept the longest matching prefix, emit the main
model's correction at the first mismatch, and restore/advance state according
to the recovery design. Then implement sampled speculative acceptance with the
appropriate rejection correction so the target distribution is preserved;
identical seeded token streams are not a requirement for a sampler that
consumes random draws differently.

**Acceptance.** Greedy output matches ordinary decoding on pinned prompts;
synthetic sampled tests verify the acceptance/correction distribution,
including zero-probability and full-rejection cases. Test EOS, stop limits,
partial batch acceptance, cancellation, and context capacity. Run `make check`
and `make compare`, plus the Metal generation target. Benchmark ordinary versus
speculative decode on identical artifacts, sampling options, prompts, and
contexts, reporting draft length, acceptance rate, verification/recovery cost,
memory, and end-to-end tok/s. Ship enabled by default only if measured gains
justify it; record negative results.

### Draft sources of the other families

The draft contract becomes "draft candidates from a source the adapter
chooses": Qwen3.8's embedded tensors (or its companion head), Gemma 4's
companion head (`MTP/mtp-*.gguf`, the 12B and the 26B-A4B alike), and Muse
Glimmer's `dflash-kquant.gguf`, whose facts (architecture, inputs, block
size, how the reference drives it) are read from the file before the
contract is declared shared — a block-diffusion drafter proposes several
positions at once and may need a different acceptance loop than a
next-token head. Each companion is a pinned, explicitly configured
dependency (`models.<name>.mtp`), never an implicit download. Speculation
ships enabled per family only where its measured acceptance rate pays for
verification; negative results are recorded.

**Where the detail goes.** Update the runtime, Metal, and generation reference
documents, the benchmark records, the engineering log, and
[llm-guide.md](llm-guide.md) as each concept is implemented.

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
- **Short-prompt prefill near the weight-bandwidth floor.** The 22-token
  benchmark prompt measured well above one token through the matvec path; with
  one token tile the matmul tiles run at 20–25 GB/s (too few threadgroups, a
  latency-bound K loop). A split-K or bandwidth-bound small-M kernel for
  chunks under ~64 tokens would cut first-token latency of chat turns by
  several times ([bench.md](reference/bench.md)).
- **A GPU penalty kernel.** Apply the token history to the logits on the
  device before `nu_topk_partial`, so the instruct profile (presence penalty
  1.5) returns to the GPU sampling path. The 32K record measured the cost of
  the full readback it pays today at +20.7 ms/token
  ([bench.md](reference/bench.md)); a measured follow-up, not part of v0.1.

## Also deferred

- Teacher-forced `eval` command ([spec § CLI](spec.md#cli-proposal)).
