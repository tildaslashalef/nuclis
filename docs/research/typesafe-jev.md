# TypeSafe / Jev research notes

Initial Jev/training review: 2026-09-18. DiffusionGemma, vLLM and kev update:
2026-09-20. Primary sources only for conclusions. No model executed or
benchmark reproduced.

**Current recommendation:** investigate text-only DiffusionGemma support,
then a bounded, no-training decision-read experiment (§4–5). This is a
plausible route to a local typed-decision interface, not evidence of Jev's
architecture or calibration. kev (§6) offers a separate trained pointer-head
route. The native Zig training proposal (§3) remains optional. Implementation
and roadmap changes have not been accepted by this research update.

## What is established

- Jev consumes text or textual structured state plus independent, typed questions. Choice returns a categorical distribution and its highest-probability option; Score returns a probability-weighted rubric level; Noul returns probability of yes. JSON on the wire does not imply model-generated JSON. Question IDs are bookkeeping and not passed to inference. [Introduction](https://docs.typesafe.ai/introduction.md), [API contract](https://docs.typesafe.ai/api.md).
- Multiple questions share the input state and are evaluated independently in parallel. That prevents one question's text/answer from contaminating another, but does not remove distraction from irrelevant shared state. Dependent reasoning still requires code to combine decisions or make subsequent requests. [Introduction](https://docs.typesafe.ai/introduction.md), [known limitations](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md).
- TypeSafe describes RLCD (reinforcement learning for calibrated decisions) as post-training starting from pretrained language models. The training target is probabilities that match outcome frequencies, rather than preferred prose. Calibration is statistical across predictions, not a guarantee for one case. [AI primer](https://docs.typesafe.ai/introduction/machine-learning-primer.md).
- Choice/Score `confidence` is a computed summary of the distribution's concentration, not a separate verification oracle or necessarily the probability the selected answer is correct. Noul has no extra confidence field. Docs do not give the confidence formula. [Confidence](https://docs.typesafe.ai/confidence.md).
- Current documented model is `jev-1.13.0`, accessed via hosted API; latest/preview aliases both point to it. [Models](https://docs.typesafe.ai/models.md).

## Architecture boundary

The launch calls the stack a new model architecture plus parallel sampler, and contrasts parallel outputs with autoregressive token generation. The reviewed public sources do not specify base model, parameter count, layers, attention layout, exact forward-pass schedule, training objective equation, or public Jev weights. Do not infer “not a transformer,” “diffusion,” “one matrix multiplication,” or literally “one forward pass.” RLCD names a training approach, not the tensor graph. [Launch](https://typesafe.ai/blog/introducing-system-one-models-and-jev), [primer](https://docs.typesafe.ai/introduction/machine-learning-primer.md).

The official GitHub organization has SDKs and an LLM compatibility adapter, plus forks of vLLM and LLaDA. A fork is not proof Jev uses either architecture. I found no Jev weights or implementation among the organization's public repositories listed through its API. [Organization repository listing](https://api.github.com/orgs/typesafe-ai/repos?per_page=100), [official LLM adapter](https://github.com/typesafe-ai/system-one-adapter-python).

## Claims and limitations

The launch advertises 193.6x faster / 444.6x cheaper from four vendor workflows, and says these likely represent the high end of gains. Reference answers are average outputs of Astra and Fable, not labeled ground truth. Baseline LLMs use a wrapper producing probabilities, which vendor notes costs more than discrete decisions. Short-input demo favors Jev; reported latency measured near West Coast service. Therefore no direct comparison to nuclis's local M4 Pro inference can be made. “Zero hallucination” evidence is schema validity, not truth of judgments. [Launch](https://typesafe.ai/blog/introducing-system-one-models-and-jev).

Known weaknesses include arithmetic/counting/date comparisons, multi-hop indirection, adversarial inputs, and irrelevant long context. It cannot write code/prose/explanations. For extraction, code first finds candidates and Jev can select from those. Context limits: 64K total across state/questions and 32K state plus longest question. [Jev 1.13 limitations, reviewed by vendor September 16](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md).

## 1. Jev-like use cases in software development

You supply the **possible answers**, not the correct answer. Choices can
come from runtime data: a repository search can find 30 files, then the
model judges their relevance. If the right file is missing, the workflow
needs another search or an “insufficient evidence” result.

| Use case | Example decision | Advantage over generating an answer token by token |
| --- | --- | --- |
| Tool routing | Choose a registered search, read, or test handler | Return a known handler ID without generating a tool-call explanation |
| Context selection | Score each retrieved file against a bug report | Return relevance scores without generating a ranked-list narrative |
| Failure triage | Classify a failure as compilation, test assertion, or environment | Give code a bounded branch to follow |
| Patch screening | Estimate whether a diff changes a public interface | Supply a signal for review; tests and code analysis still verify behavior |

Our [engine](../../inference/src/engine.zig) prefills the prompt, predicts a
token, feeds it back, and repeats. A decision interface can stop after
scoring answers. Jev also evaluates independent questions in parallel;
this is particularly useful for many small judgments over shared input.
Input processing still costs computation, and “one query” does not reveal
the number of internal forward passes.
[System One](https://docs.typesafe.ai/concepts/system-one),
[Introduction](https://docs.typesafe.ai/introduction.md).

Generating a patch or explaining a bug remains an open-ended task for
Qwen/Gemma. An existing LLM can also score answer tokens directly, so the
fair speed comparison includes that baseline—not only verbose JSON
output. Typed output prevents invalid choices, not incorrect judgments.

## 2. Fine-tuning on the M4 Pro / 48 GB and a nuclis decision interface

**Start from pretrained weights; training from scratch is unnecessary.**
The following are research options, not implemented support or measured
memory guarantees:

| Option | What we train | Inference and tradeoff |
| --- | --- | --- |
| No-training baseline | Nothing; prompt a supported model with single-token labels | Read label logits after prefill; simplest way to test decision quality |
| LoRA / QLoRA on a small decoder | Small weight updates; QLoRA keeps base weights quantized | Train state + question + choices → answer label; preserves the decoder graph |
| Classification or scoring head | A small output layer, optionally with LoRA on the backbone | Predict fixed categories, or score each state/candidate pair; requires a new output contract and training code |

**External-training option:** a 0.5–3B decoder with 4-bit QLoRA, batch size
1, and 512–2048-token examples. These are conservative experiment settings
for this Mac, not measured capacity limits. A 7–8B model is a later candidate
if quality warrants it. Activations, sequence length, optimizer state, and
the OS share the 48 GB; measure peak memory and step time before scaling.
Apple's MLX supports training on Apple silicon; MLX-LM provides LoRA/QLoRA,
with batch size, sequence length, and gradient checkpointing as memory
controls. [Apple MLX overview](https://developer.apple.com/videos/play/wwdc2025/298/),
[MLX-LM training guide](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md).

A fixed classification head learns named categories. Supporting arbitrary
choices needs examples with varying candidate descriptions, or a scorer
trained on `(state, question, candidate)` pairs. Such a head needs a custom
training/export path; MLX-LM's ordinary text fine-tuning does not install
it automatically. [Classification training example](https://huggingface.co/docs/transformers/tasks/sequence_classification).

**Proposed nuclis interface:** `decide(state, questions) → typed results`,
with bounded choices, per-choice probabilities, and an explicit abstention
policy. Initially, evaluate questions separately using final-position label
logits; sharing state computation and batching questions are later execution
work. A classification head instead exposes decision scores directly. Keep
this execution path separate from the text-generation loop while reusing
compatible tokenizer, weight, and backend operations.

If using MLX, train externally and import a validated artifact. For LoRA, merge
updates into weights before conversion to avoid needing a live LoRA runtime.
MLX-LM's documented direct GGUF export is limited; choose and verify the
model family, conversion path, tensor encodings, and nuclis adapter support
**before training**. A small Qwen or Gemma name alone does not establish
compatibility. Our model adapters execute weights; they do not train them.
[MLX-LM export limitations](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md#fuse).

Use labeled examples split by repository/task to limit leakage. Compare
accuracy, latency, and calibration against the unfine-tuned baseline;
include missing-answer cases and shuffled choices. Fit any probability
calibration on validation data and report results on a separate test set.
Token probabilities and concentrated distributions alone do not establish
correctness confidence. This would be a local decision model, not a
reproduction of Jev's undisclosed RLCD recipe.
[TypeSafe primer](https://docs.typesafe.ai/introduction/machine-learning-primer.md),
[confidence semantics](https://docs.typesafe.ai/confidence.md).

## 3. Proposed next experiment: a Zig training pipeline

The earlier options assumed an external trainer. **A native Zig trainer is
an additional experiment**, with MLX/PyTorch serving as numerical references.
Recommendation: start with a frozen small decoder and a trainable scoring
head, then add limited LoRA if the task needs more capacity. This trains new
parameters on pretrained features; it does not yet fine-tune the backbone
or reproduce Jev's undisclosed architecture.

### Package boundary

Proposed ownership, not existing directories or commands:

```text
src/       CLI: train / evaluate / decide
training/  datasets, loss, backward passes, optimizer, checkpoints
    ↓
inference/ model forward passes, tokenizer, weights, decision execution
```

A sibling `training/` package may depend on reusable `inference` APIs;
`inference` must not depend on the trainer. Reuse compatible CPU/Metal math
and format readers. Keep trainable weights, gradients, optimizer state, and
saved activations under training ownership, separate from immutable mapped
inference weights. Extract a shared lower-level package only if actual reuse
makes the dependency awkward.

The current [registry](../../inference/src/models/root.zig) supports Qwen3.5,
Gemma4 and Muse Glimmer families; it is not a general Hugging Face executor.
The current
[Qwen binding](../../inference/src/models/qwen35.zig) pins the 27B dimensions.
All candidates below require model/tokenizer validation and an adapter
addition or extension. Existing inference scratch is overwritten during
forward execution; it is not a training activation tape. A Zig wrapper around
MLX would be orchestration, not our own backward implementation.

### A bounded first training experiment

1. **Forward correctness:** pin a small model revision, tokenizer and weight
   digests; validate final hidden features against a reference. Start with
   floating-point weights, short sequences (128–256 tokens), batch size 1.
   Use a verified floating-point GGUF conversion or add a bounded safetensors
   loader; do not start by training packed 4-bit weights.
2. **Train a head:** encode `(bug report, candidate file excerpt)` and use
   the final non-padding token's normalized hidden vector `h` to predict
   relevance: `sigmoid(w·h + b)`. Freeze the backbone, cache features for the
   fixed dataset, and implement binary cross-entropy, analytic gradients,
   and an optimizer in Zig. This needs no transformer backward pass or
   general-purpose automatic differentiation engine.
3. **Serve and evaluate:** export the head with its base-model digest,
   feature convention and input template; load it through a proposed
   `scoreCandidates` interface. Independent relevance scores are not a
   mutually exclusive choice distribution. Add categorical `choose` only
   with a defined training and normalization contract.
4. **Then fine-tune:** add low-rank updates to selected projections in the
   final block, with handwritten backward operations. Extending LoRA into
   earlier blocks requires backpropagating through downstream frozen layers:
   frozen weights remove their updates, not their input derivatives.
   Full-model training and QLoRA are later options, not prerequisites.
   [LoRA paper](https://arxiv.org/abs/2106.09685).

Acceptance should cover finite-difference gradient checks, a reference
optimizer step, loss reduction on a tiny fixture, checkpoint/resume
agreement, and held-out retrieval quality and calibration. Split by
repository/task and include difficult negative candidates. Compare with
lexical search and an existing reranker; failure to beat them is a valid
experimental result. Record peak memory and time separately for feature
extraction, training and serving.

For scale: 135M parameters take about **0.27 GB at 16 bits**. A hypothetical
full F32 Adam training state (weights + gradients + two moments) is about
**2.16 GB**, before activations, scratch, caches or copies. These are decimal
arithmetic estimates, not measurements. On a 48 GB Mac, this makes 135M a
reasonable starting hypothesis without quantization complexity; throughput
and total memory still need measurement.

### Hugging Face candidate shortlist


Checked official cards and configs on 2026-09-18. These are engineering
recommendations; no candidate has been downloaded or benchmarked here.
All five model cards list Apache-2.0.

| Candidate | Relevant facts | Recommended role |
| --- | --- | --- |
| [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) | 30 layers, hidden 576, vocabulary 49,152; conventional Llama-style attention, no QKV bias | **First Zig training target:** small enough for rapid correctness work; limited general reasoning quality |
| [SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) | 32 layers, hidden 960, same vocabulary and architecture family | Scale the same implementation after the 135M experiment |
| [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) | 24 layers, hidden 896, vocabulary 151,936; QKV bias | Alternative for multilingual/coding data; additional adapter/tokenizer work |
| [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) | 28 layers, hidden 1024, Q/K normalization; 128-wide heads | Later training candidate; query projection is 2048-wide, so hidden/head-count is not its head dimension |
| [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) | Already trained for instruction-conditioned query/document scoring; uses final-position yes/no logits | **Serving/quality baseline:** test direct decisions without training our own model first |

Dimension sources: [135M config](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/raw/main/config.json),
[360M config](https://huggingface.co/HuggingFaceTB/SmolLM2-360M/raw/main/config.json),
[Qwen2.5 config](https://huggingface.co/Qwen/Qwen2.5-0.5B/raw/main/config.json),
[Qwen3 base config](https://huggingface.co/Qwen/Qwen3-0.6B-Base/raw/main/config.json).
The reranker's official Transformers example reads logits without generating
text; its scores still require task-specific calibration checks.
[Official scoring example](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B#using-transformers).

Choose base weights for the first custom-head experiment to make the
training setup explicit, not because base is necessarily more accurate.
SmolLM2 has separate `-Instruct` checkpoints; `Qwen3-0.6B` without `-Base`
is already post-trained. A head trained for file relevance does not
thereby learn every arbitrary question or rubric.

[Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B/raw/main/config.json)
is closer to our existing hybrid family, but its text path mixes recurrent
linear attention and full attention. That adds recurrent backward work for
backbone tuning, and its smaller dimensions are not accepted by the current
27B binding. Prefer the simpler SmolLM2 graph for the first training engine;
family resemblance alone is not an implementation shortcut.

**Suggested progression:** SmolLM2-135M forward + head training → limited
last-block LoRA → compare 360M or Qwen on the same held-out task. This makes
native Zig training the experiment, with external frameworks used for
validation. It remains a research recommendation, not a change to the
active implementation plan.

## 4. DiffusionGemma first: a plausible inference experiment

**Recommendation (2026-09-20):** adding text-only DiffusionGemma, validating
ordinary diffusion generation, then experimenting with typed decisions is a
coherent progression. It exercises a publicly inspectable model before
changing its execution semantics. Native training in §3 is an independent
option, not a prerequisite. This is a research recommendation, not an
accepted reorder of the speculative-decoding plan in `TODO.md`.

### Artifact and architecture facts

The [Unsloth repository](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF/tree/f4183a2c7a354128d02545752303c4354d165bf0)
was inspected at `f4183a2c7a354128d02545752303c4354d165bf0`.
Its card lists Q4_K_M at approximately 16 GB and Q8_0 at 25 GB, and directs
users to a dedicated `llama-diffusion-cli`. The generic app launch snippets
above the card are not evidence those apps implement the diffusion loop.
The linked [llama.cpp PR #24423](https://github.com/ggml-org/llama.cpp/pull/24423)
was **open, unmerged**, head `12e0a9627d02c6395fd4bbf2aadff93d0d46a0e4`,
when checked. Pin that reference separately from nuclis's existing oracle.
No GGUF weights were downloaded or their tensor inventory verified here.

The [official configuration](https://huggingface.co/google/diffusiongemma-26B-A4B-it/blob/main/config.json)
declares `DiffusionGemmaForBlockDiffusion` / `diffusion_gemma`, a 256-token
canvas, 30 text layers, hidden width 2816, vocabulary 262144, 16 query
heads, 8 sliding KV heads, 2 global KV heads, and 128 experts with 8 selected.
The attention schedule is five sliding layers then one global layer;
sliding/global head widths are 256/512, and the sliding window is 1024.
The configured maximum position count is 262144; that is not a tested
nuclis capacity or a memory guarantee. Record the config revision and the
actual GGUF metadata/digest during any implementation inventory.

Google describes one fine-tuned backbone used in causal encoding and
bidirectional denoising modes. A canvas starts with random vocabulary
tokens, not a dedicated mask token. Denoising can revise positions; the
completed canvas is encoded into the persistent prefix before the next
canvas. Self-conditioning carries information from the previous
probability distribution through the embedding table into the next step.
[Google architecture explanation](https://ai.google.dev/gemma/docs/diffusiongemma/explained).

The recommended sampler uses up to 48 iterations, a 0.8→0.4 temperature
schedule, entropy-bound selection (0.1), and early stopping requiring both
low average entropy (0.005) and unchanged predictions on consecutive
steps. These are model inference settings, not decision calibration.
[Google serving configuration](https://ai.google.dev/gemma/docs/diffusiongemma).

### What existing Gemma support buys us, and what it does not

nuclis already has a Gemma 4 26B-A4B binding, CPU schedule, Metal plan,
expert routing and quantized expert kernels. The registry also includes
Muse Glimmer. Reuse candidates are the tokenizer infrastructure, verified
Gemma math, MoE kernels and GGUF decoding, subject to actual tensor and
profile compatibility checks.
[Registry](../../inference/src/models/root.zig),
[Gemma binding](../../inference/src/models/gemma4.zig),
[Metal plan](../../inference/src/models/gemma4_metal.zig).

The pinned [reference graph](https://github.com/ggml-org/llama.cpp/blob/12e0a9627d02c6395fd4bbf2aadff93d0d46a0e4/src/models/diffusion-gemma.cpp)
shows why changing the architecture string is insufficient: prompt and
canvas embeddings have different normalization, layers use separate
encoder/decoder scalars, and the attention mask depends on the region.
It also binds a self-conditioning gated MLP. The cached path keeps prompt
KV separate from canvas evaluation; a no-cache combined graph provides a
useful equivalence check. These are reference facts, not code to copy.

Proposed implementation boundaries:

| Component | Required work and invariant |
| --- | --- |
| Model adapter | Validate diffusion metadata, all tensors and encodings; own encoder/denoiser schedules and mode-specific transformations |
| Attention | Causal prompt processing plus bidirectional canvas attention to itself and permitted prefix positions; preserve exact sliding-window and position rules |
| Session | Persistent encoded prefix; separate transient canvas, previous-step predictions and scratch; an iteration must not advance committed history |
| Sampler | Seeded canvas initialization, entropy selection, renoising, self-conditioning and bounded stopping; test independently of the model |
| Generation driver | Prefill → repeated denoise → finalize → encode finalized block; enforce output/context limits and cancellation across these phases |
| Output/profile | Check the real template and special tokens; emit committed text only, with optional provisional visualization kept out of tool execution |

This should be a distinct generation strategy alongside the existing
[autoregressive loop](../../inference/src/engine.zig), with shared math below
it. It is neither an MTP head nor a DFlash draft source: DiffusionGemma is
the final generator, with no separate causal target verifying its proposals.
Do not route it through `runtime/draft.zig` merely because it predicts a
block. Reusable batched kernels do not make the cache semantics equivalent.

### Memory and performance questions for the M4 Pro

Q4_K_M is a sensible first *candidate* for 48 GiB, not a demonstrated fit.
All experts' stored weights still consume memory even though only eight
are selected per token. Account for weights, prefix KV, canvas attention,
expert activations, self-conditioning, backend copies and OS headroom.

A full `256 × 262144` F32 logit matrix is **256 MiB**; keeping probabilities
and previous logits separately multiplies that allocation. A dense F32
`262144 × 2816` embedding representation is **2.75 GiB**. These are shape
arithmetic, not observed allocations. The reference graph explicitly
constructs a transposed/dequantized embedding for self-conditioning; inspect
its storage precision before estimating its actual footprint. Restricting
returned decision logits later does not automatically eliminate the
full-vocabulary self-conditioning computation.

Diffusion trades sequential token steps for repeated wide forwards. Our
Metal batched kernels might benefit, but output projection, MoE routing,
self-conditioning and synchronization can dominate. Measure prompt prefill,
first **committed** output latency, per-canvas iteration count, total time,
useful tokens/s and peak memory separately. Include short answers where
canvas overhead may erase the throughput benefit. Cloud GPU claims do not
predict M4 Pro results, and faster generation alone says nothing about
classification accuracy.

### Suggested acceptance sequence (not scheduled units)

1. **Inventory and oracle:** pin model/converter/reference revisions and
   hashes; inspect the GGUF header and tensor encodings; capture tokenization,
   encoder features, zero/nonzero self-conditioned denoiser logits, and
   sampler fixtures. Start text-only with a short context.
2. **CPU semantics:** reproduce the reference with fixed noise and sampler
   inputs, including prefix-cache equivalence, one completed canvas and a
   second canvas. Test EOS, output budget, cancellation and session reset.
3. **Metal parity and generation:** match CPU/reference intermediate traces
   at declared tolerances; verify resources outlive submitted work. Measure
   end-to-end generation and memory before promising a catalogue default.
4. **Decision experiment:** add the constrained path described below only
   once ordinary generation is trustworthy. Compare it with existing-model
   answer-logit scoring on the same labeled tasks.

A reference PR and a model card are starting points for this sequence;
neither substitutes for measured Apple/Metal correctness or performance.

## 5. vLLM PR #57250: no-training structured reads

Inspected [PR #57250](https://github.com/vllm-project/vllm/pull/57250)
with `gh pr view` and `gh pr diff`: **open, unmerged**, head
`ceb8eebf3eedddb964a50180f33838a9a6b13ee2`. Findings below refer to its
[pinned model implementation](https://github.com/vllm-project/vllm/blob/ceb8eebf3eedddb964a50180f33838a9a6b13ee2/vllm/model_executor/models/diffusion_gemma.py)
and [example server](https://github.com/vllm-project/vllm/blob/ceb8eebf3eedddb964a50180f33838a9a6b13ee2/examples/features/diffusion_reads/structured_server.py).
An open proposal and a CI trigger are not a released support guarantee.

This is a no-training experiment using the pretrained DiffusionGemma vocabulary head,
not a Jev checkpoint or a reconstruction of Jev's training. A client builds a canvas
with answer-format token IDs and random/noised single-token decision slots. The model
reads that canvas conditioned on the prompt. The client retrieves exact vocabulary
logprobs for allowed label IDs at each selected position and renormalizes over those
labels. Multi-token semantic labels are mapped to single-token aliases (A/B/etc.);
single-token-ness must be checked with the actual tokenizer in context.

The server additions expose seed canvas, denoising-step cap, read-only mode (emit
converging argmax and temperature-1 logprobs; omit the commit forward), and per-request
canvas width no larger than the configured width. Existing `logprob_token_ids` supports
at most 128 IDs. The diff also changes scheduler and state lifecycle behavior, includes
tests, skips self-conditioning work for single-step requests, and isolates diffusion
scheduling in a subclass following maintainer feedback. This is more than a prompt
template or grammar change. “One read” still requires prompt prefill plus a decoder
canvas forward on a cold request; it is not one universal forward pass including prompt
ingestion.

Important limitation: the exposed seed canvas is initialization. No fixed-position mask
is among the documented request fields; do not promote the PR prose's “fixed tokens”
into a claim of hard-clamped positions throughout arbitrary iterative denoising.
One-step slot reads are the demonstrated use.

The PR author reports DGX Spark / NVIDIA NVFP4 performance: 32-position canvas, three
decisions, schema-prefix cache, cache-busted state; 8.7 requests/s at concurrency 1
(~0.12 s) and 54.0 requests/s at concurrency 32 (~0.58 s), ~162 decisions/s. These are
author measurements on another stack, not M4 Pro/GGUF projections. Small examples
include mistakes (9/10 language identification, 10/12 unit comparison); there is no
common calibrated benchmark proving equivalence to Jev.

### Probability caveats verified in the implementation

`slot_distribution` applies softmax to allowed-label logprobs and separately reports
`label_mass` and `argmax_is_label`. Conditional label probabilities can be high even
when total vocabulary mass assigned to the label set is poor. Preserve these diagnostics
in any experiment.

Its `entropy` is `-sum(p*log(p))` over the returned token set, without renormalizing
that set. It is a partial vocabulary entropy, not full-vocabulary entropy nor normalized
label-distribution entropy. The adaptive reread policy uses that value; threshold
portability is not established. The diff still contains this calculation at the pinned
head, so this is not merely an obsolete review criticism.

The multiple-read standard error is computed from the chosen label's probabilities
across random canvas seeds; agreement is the fraction of per-read argmaxes matching the
aggregate argmax. These quantify seed sensitivity, not empirical probability of being
correct, calibrated confidence intervals, or guaranteed epistemic uncertainty. Repeated
correlated confident mistakes remain possible.

PR discussion raises mixed-phase concurrency regression coverage, bounds on automatic
rereads and question fan-out, truncation when max_tokens is smaller than canvas width,
and async wasted steps. Some implementation issues evolved after the review: do not
assert all historical criticisms remain unresolved without checking the current diff.
They are useful acceptance-test categories for nuclis.

## 6. kev: trained pointer decisions without diffusion

Inspected [README](https://github.com/jaredpalmer/kev/blob/20fa6268c8ceb226530be2fb5266ab2c36b37724/README.md)
and [model implementation](https://github.com/jaredpalmer/kev/blob/20fa6268c8ceb226530be2fb5266ab2c36b37724/kev/model.py)
at `20fa6268c8ceb226530be2fb5266ab2c36b37724`. Performance and evaluation
numbers in this section are the author's reports, not nuclis measurements.

kev is a different architectural route: a Qwen2.5/Qwen3 base backbone plus trained
rank-16 LoRA and a pointer readout, with no vocabulary output head and no text decoding.
The implementation loads the bare transformer, packs shared state plus question
branches, assigns each branch positions restarting after the state, and masks attention
so branches cannot read siblings. Each question's decision hidden state scores its
option-end hidden states through learned query/key linear projections (default head
dimension 256) with a scaled dot product; softmax produces a variable-length
distribution. Option text can span multiple tokens. This does not require diffusion.

README supports noul, choice (2–255 options), score, and TypeSafe-shaped `/v1/systemone`
requests. Noul is yes probability; choice uses argmax, with confidence `(pmax - 1/K)/(1 - 1/K)`; score is expected zero-based level. Score-confidence formula is explicitly a
stand-in because TypeSafe's is unpublished. Thus similar names across kev/vLLM/TypeSafe
do not imply identical confidence or score semantics (the vLLM sample computes a
one-based expected level and calls pmax confidence).

Reported isolation/packed-vs-separate deltas are about 4e-6. Exact question isolation is
different from option permutation invariance: default options remain causal inside a
branch; optional option isolation exists but reportedly costs 4B accuracy. No equivalent
cross-question isolation is established by ordinary shared DiffusionGemma canvases.

README reports on frozen transfer-v4 dev items: kev-4b accuracy .790, Brier .328,
confident-error rate 8.2%; kev-8b .796/.337/9.9%; hosted Jev .857/.211/3.7%. These are
repository-author results, not independently reproduced; Jev's exposure to public data
is unknown. README explicitly states out-of-domain calibration does not transfer (ECE
around .1), despite its introductory “calibrated probabilities” wording. Cross-entropy
training alone does not guarantee calibration.

0.6B/4B/8B are preview checkpoints; the predeclared per-seed policy-pair gate is unmet.
Released 0.5B is weaker out of domain. Reported M5/bf16 serving figures (~1 s 4B, ~2
s 8B) are not directly comparable to the PR's warmed DGX measurements. Serving is one
request at a time, no cross-request KV cache, dense per-sample mask, max 8192 serving
tokens, trained with 384 state/1024 branch token bounds. README calls the
implementation a reconstruction based on an external architecture analysis; it is not
authoritative evidence of Jev internals.

## 7. Common evaluation and adoption criteria

1. Model support first is technically coherent if DiffusionGemma itself is wanted, but
   decision reading should be a later experiment with explicit engine primitives: seed
   canvas, noncommitting canvas evaluation, exact selected-position/selected-label
   logits, per-request width, bounded rereads, prefix-state lifetime. GGUF availability
   alone does not supply these capabilities.
2. Separate generic typed-result/application contracts from the numerical model adapter
   and diffusion lifecycle. A shared decision API could compare causal next-token label
   baselines, diffusion reads, and trained pointer heads without pretending they share
   mechanics or calibration.
3. Treat kev as a contrasting optional backend/research reference, not a mode that
   naturally appears after adding DiffusionGemma. Native kev support needs the correct
   Qwen base/checkpoint, LoRA handling or validated merging, pointer-head tensor
   serialization/loading, hidden-state output, branch attention masks and reset position
   IDs, and a faithful renderer. A generic chat GGUF is insufficient.
4. Freeze one common eval set before optimization: accuracy, Brier/log loss,
   reliability/ECE, abstention coverage, OOD and prompt-injection robustness;
   sibling-question contamination, option permutations, label aliases, canvas
   seed/width/step sweeps, total allowed-label mass; cold/warm latency and memory
   separately. Compare full precision/quantization where feasible. Standard-error across
   seeds is diagnostic, not the success criterion.
5. This research supports experimentation; it neither accepts a new roadmap item nor
   establishes local feasibility/performance. Keep the adoption decision separate from
   the documentation update.


For the decision experiment, use a common result contract while retaining
backend-specific diagnostics:

```text
Choice: option IDs + normalized probabilities + selected ID
Noul: probability of yes
Score: explicit rubric values + distribution + expected value
Diagnostics: label mass (when applicable), seed sensitivity,
             inference steps, latency, calibration version
```

For vocabulary-label reads, define
`p(i | allowed) = exp(logit_i) / sum_allowed(exp(logit_j))` and separately
retain `sum_allowed p_vocab(j)`. For a pointer head, all modeled outcomes
are options, so vocabulary label mass is not defined. An explicit
“none/insufficient evidence” option and a validation-fitted abstention rule
are preferable to presenting every concentrated distribution as certainty.
Do not expose one backend's `confidence` formula under a supposedly
universal meaning.

**Decision gate:** first establish useful quality versus single-token
answer scoring on an already supported model. Then require an advantage
in measured decision latency, memory, or accuracy on the same held-out
workload. Diffusion might win with several decisions over shared context;
a smaller trained pointer model might win on memory and short requests.
Those are hypotheses. Neither requires claiming access to Jev weights or
reproducing undisclosed RLCD.
