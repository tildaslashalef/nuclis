# TypeSafe / Jev research notes

Checked 2026-09-18. Primary sources only for conclusions. No model executed or benchmark reproduced.

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

The current [registry](../../inference/src/models/root.zig) supports Qwen3.5
and Gemma4 families; it is not a general Hugging Face executor. The current
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
