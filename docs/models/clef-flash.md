<!-- kind: card -->
# clef-flash

clef-flash (`Cloudflare/clef-flash`, Apache-2.0) is a decision model like
[Laya](laya.md): it answers typed questions (`noul`, `choice`, `score`)
about a state, with one logit per option and no decoding. It differs in
two ways. It is a 9B **Qwen3.5 backbone** (the hybrid DeltaNet family
nuclis runs for Qwen3.8, at a smaller shape) with its vision encoder, so
it reads images. And its **joint schema head**, a small transformer over
the backbone's final hidden states, scores every option of every question
in **one pass** over one sequence, where Laya runs one pass per question.
States run up to 16,384 tokens. nuclis runs it in these parts:

| Part | Files |
| --- | --- |
| Backbone (qwen35 at the file's shape, under the `clef` id), CPU and Metal | `inference/src/models/qwen35.zig`, `qwen35_runtime.zig`, `qwen35_metal.zig` |
| Input contract: questions, the sequence, images' tokens | `inference/src/profiles/clef.zig`, `profiles/decision.zig` |
| Head (CPU, F32, read from the same GGUF), the model | `inference/src/models/clef.zig` |
| Projector (the Qwen3-VL merger at width 4,096) | `inference/src/vision/qwen3vl.zig`, `qwen3vl_metal.zig`, `projector.zig` |
| The family seam, `nuclis decide`, `nuclis serve` | `inference/src/decide.zig`, `src/decide.zig`, `src/decision/`, `src/api/decisions/` |
| Checks | `inference/clef-check.zig`, the reference's fixtures in `inference/src/models/fixtures/clef/` |

## Architecture: the joint schema head

`JointSchemaHead` (`clef.decision.*`: hidden 4,096, width 1,024, 2 routing
layers, 4 decoder layers, 16 heads, feed-forward 4,096; 122 M parameters),
its tensors `decision.*` and `dec.blk.0`–`5` (routing layers first) in the
same file as the backbone, decoded to F32 at load (about 0.5 GB):

1. A LayerNorm over every hidden row, then the **memory**: every row
   projected to 1,024.
2. Per question, the mean of its instruction span; per option, the mean of
   its span (context) and the mean of the output head's rows over its
   tokens (lexical, `output.weight` decoded from Q8_0).
3. **Option queries** (context + lexical + question projections) pass two
   evidence-routing layers: cross-attention to the normalized memory, a
   GELU feed-forward.
4. **Fields**: the question projection, a softmax-weighted summary of its
   routed options, the last row's projection, a type embedding; four
   pre-norm decoder layers (self-attention among the fields,
   cross-attention to the memory, feed-forward).
5. **Scores**: a lexical prior (cosine of the option's lexical vector and
   the question + last-row anchor, scaled) plus a gated joint term (scaled
   cosine of field and normalized option, and a residual MLP of `[f, o,
   f·o, |f−o|]`). The file stores the two scales and the gate as the
   forward uses them (`decision.scales`: exp, exp, sigmoid).

It runs on the CPU in both backends (`cpu.dense` matmuls across cores,
attention per head as two matmuls): 30–130 ms per request up to 829
tokens, 450 ms at 3,931, about 4 % of the Metal backbone's time; the head
has no Metal path.

## Files and encodings

`nuclis model pull clef-flash` fetches one file from
`ggml-org/Clef-Flash-GGUF`, pinned at `c3d92545` and checked against the
catalogue's SHA-256; `--with mmproj` adds the projector:

| What | File | Size |
| --- | --- | ---: |
| Backbone, head, and tokenizer (architecture `clef`) | `Clef-Flash-Q4_K_M.gguf`: the backbone Q4_K and Q8_0, the head's matrices Q8_0, its norms, biases, and scales F32 (97 Q4_K, 212 Q8_0, 277 F32, 1 BF16 tensors) | 6.73 GB |
| Projector (`--with mmproj`) | `mmproj-Clef-Flash-BF16.gguf` | 0.92 GB |

Q4_K_M rather than the Q6_K nuclis first ran: the prefill is
compute-bound, so the rate is the same and the file 1 GB smaller. Against
Cloudflare's reference the answers' probabilities moved by at most 1.55e-2
and no fixture answer changed
([record, 2026-11-21](../benchmarks/record/clef-flash.md#ggml-orgs-single-file-2026-11-21)).
The BF16 projector, not the Q8_0 one: the Metal projector plan executes
F32, F16, and BF16 matrices.

## The input contract

From Cloudflare's `joint_schema_model.py` at `17f0b0ad`
(`encode_record`, `question_options`, `systemone`):

- **Questions.** `type` is `noul`, `choice`, or `score`; `instructions`
  is optional (the question id stands in); `choice` and `score` need
  non-empty `criteria` (an object of option → description, a list of level
  descriptions). A `noul` question's options are `true` and `false`,
  described by default and overridable through `criteria`.
- **Options** are rendered as JSON, `{"description": …, "option_id": …}`
  (`json.dumps(…, ensure_ascii=False, separators=(",", ":"),
  sort_keys=True)`, no description when it is null). The model reads
  `choice` options sorted by key and `noul` as `true` then `false`;
  answers come back in the request's order (`Question.model_order`).
- **The sequence**: `<|im_start|>system` and a fixed instruction,
  `<|im_start|>user\nSTATE:\n`, then the images' tokens, the state (a
  string as it is, anything else as the sorted compact JSON above), the
  schema (`SCHEMA FIELDS:`, per question `FIELD n`, `ID`, `TYPE`,
  `INSTRUCTION`, `ALLOWED OPTIONS`, `END FIELD`), and the assistant's
  empty thinking block ending `JOINT SCHEMA DECISIONS:`. Every piece is
  tokenized on its own and concatenated; the head reads the spans of each
  question's instruction and each option's JSON.
- **Budget**: 16,384 tokens in all; the state is cut at its end to fit
  (whatever its shape: the reference keeps a state's head), and a schema
  that alone overflows is refused.
- **Answers** are a softmax per question, never calibrated. `confidence`
  is the top probability (Laya reports a normalized entropy there), a
  `score`'s value the expected level.
- **Images** precede the state: per image `<|vision_start|>`, one
  `<|image_pad|>` per token, `<|vision_end|>`, then a newline. An image is
  stretched (bicubic) to a grid of 32-pixel cells with at least 65,536
  pixels (`processor_config.json`), as the Hugging Face processor resizes,
  not letterboxed as the Qwen3.8 chat path does.

Tokenizing goes through the file's own vocabulary and the `qwen35`
splitter; the sequence fixture proves it builds the reference's ids.

## `nuclis decide` and `nuclis serve`

`nuclis decide --model clef-flash` takes the same questions and states
as Laya (`--request`, `--questions`, inline questions) plus `--image
<path>`, repeatable; a request's `"images"` holds data URLs, base64, or
`{"file": path}` (the CLI only). Images go before every state of the call.
`nuclis serve` serves it under the same calls: `POST /v1/systemone`
answers with exactly the reference's `systemone` fields. Without the
projector pulled, a request with images is refused
(`images_unsupported`); Laya refuses images the same way.

## What is verified

<!-- gen:gates-for:clef -->
| Gate | Tier | Model | Evidence |
| --- | --- | --- | --- |
| `clef-sequences` | `verify` | clef | [clef-flash.md](clef-flash.md) |
| `clef-metal` | `verify` | clef | [clef-flash.md](clef-flash.md) |
| `clef-cpu` | `verify-cpu` | clef | [clef-flash.md](clef-flash.md) |
<!-- /gen -->

The backbone is the qwen35 family nuclis validates for Qwen3.8 and the
projector is Qwen3.8's at another width; the new parts (the sequence and
the head) were held to Cloudflare's reference at bring-up
([record, 2026-10-02](../benchmarks/record/clef-flash.md#checked-against-the-oracle-2026-10-02)),
and the GGUF's head to that bring-up's tensor by tensor
([record, 2026-11-21](../benchmarks/record/clef-flash.md#ggml-orgs-single-file-2026-11-21)).
`clef-sequences` rebuilds the reference's ids and spans for 10 requests;
`clef-metal` holds every request, images included, within 1e-3 of
nuclis's pinned Metal logits for this file, within 2e-2 of the reference's
probabilities, and to each answer of a sanity set; `clef-cpu` does the
same for the two requests under 160 tokens, the logits within 2e-3.

## Numbers

One `noul` question over a log state, Metal, `nuclis decide --json`,
ReleaseSafe, load (0.27 s) excluded (M4 Pro, 2026-11-21;
[record](../benchmarks/record/clef-flash.md#ggml-orgs-single-file-2026-11-21)):

| State | Tokens per sequence | 1 state | 10 states |
| --- | ---: | ---: | ---: |
| about 500 tokens of log | 621–628 | 2.36 s | 22.7 s |
| about 2,000 tokens of log | 2,111–2,131 | 7.56 s | 78.6 s |

The time is the backbone's prefill, about 260–295 tok/s, linear in the
states. Questions are almost free: they add their schema tokens to one
pass, where Laya pays a pass per question. An image adds its projector
pass (80 image tokens: 1.19 s against 0.61 s for a 143-token text
request).

## Known limits

- One sequence at a time: a batch of requests runs them in turn, with no
  packing (the backbone's session holds one sequence).
- An image is at most 1,024 tokens (about a megapixel), the projector
  plan's bound; the reference allows 16,384, so a larger image is scaled
  down further than it would be there. Videos are not read.
- The CPU backbone runs about a token a second (the reference forward,
  slow by design); the CPU path is for checks, not for use.
- Only the Q4_K_M file is pinned; ggml-org's Q8_0 and BF16 files have not
  been run.
