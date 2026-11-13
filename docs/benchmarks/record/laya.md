<!-- kind: record -->
# Laya: the record

The dated record behind [the Laya card](../../models/laya.md): the oracle checks on the CPU and Metal, the times per call, and the multilingual checkpoint's measurements, as written on their dates.

## Checked against the oracle (2026-09-29)

`scripts/laya-reference.py` (retired; at tag `v0.11.0`) ran the `laya` 0.3.20
package (PyTorch 2.14.0, Transformers 5.17.0) in F32 on the CPU and wrote
`inference/src/models/fixtures/laya/`: for 8 requests the ids, markers, raw
logits, and answer, and in `activations.f32` rows of the residual stream
(row 0, up to three markers, the last row) after encoder layers 0, 1, 3,
and 27, after `final_norm`, and after both head layers (marker rows only).

```sh
zig build test-laya -Doptimize=ReleaseFast -- ~/.nuclis/models/convaiinnovations/laya
```

(gate `laya-cpu`, tier verify). The bounds are relative, because F32
summation-order differences scale with the values summed and the
pre-norm residual carries outlier dimensions up to 1.65·10⁴: per tensor,
the largest difference over the largest reference value ≤ 1e-5 and the
difference's RMS over the reference's ≤ 1e-5; per logit, the difference
over max(1, |logit|) ≤ 1e-4. (The first plan's absolute 1e-4 held through
layer 3 and after `final_norm`, not at layer 27 or in the head, whose
values reach 10²–10⁴.)

Measured 2026-09-29, Apple M4 Pro (12 cores), Zig 0.16.0, ReleaseFast,
the commit that brought Laya up on the CPU; worst stage per request:

| Request | Tokens | Options | ms | scaled max \|Δ\| (stage) | rel RMS | scaled logit \|Δ\| |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `choice_described` | 59 | 3 | 279 | 3.1e-6 (encoder.27) | 2.3e-6 | 2.1e-6 |
| `choice_labels` | 36 | 3 | 169 | 2.6e-6 (final) | 1.7e-6 | 5.2e-6 |
| `score` | 49 | 3 | 230 | 2.4e-6 (final) | 2.5e-6 | 6.0e-6 |
| `noul` | 56 | 2 | 256 | 6.6e-6 (head.0) | 4.8e-6 | 4.0e-6 |
| `json_state` | 119 | 2 | 552 | 4.0e-6 (final) | 5.4e-6 | 4.8e-6 |
| `long_text` | 512 | 3 | 2410 | 1.7e-6 (head.0) | 2.8e-6 | 7.9e-6 |
| `long_list` | 512 | 2 | 2408 | 8.3e-6 (head.0) | 7.0e-6 | 9.0e-6 |
| `choice_20` | 193 | 20 | 906 | 2.9e-6 (encoder.27) | 2.1e-6 | 7.5e-6 |

The times are the forward alone (encoder and head, the trace's comparison
included); opening took 212 ms with the checkpoint in the page cache. A
512-token sequence is about 2·10¹¹ multiply-adds, so 2.4 s is roughly 170
GFLOP/s across the cores.

## Checked against the oracle and the CPU (2026-09-29)

```sh
zig build test-laya -Doptimize=ReleaseSafe -- ~/.nuclis/models/convaiinnovations/laya --backend metal
```

(gate `laya-metal`, tier verify). The same fixtures and the same bounds
as the CPU forward; with a trace the plan commits after every stage and
the host reads the residual rows. Measured 2026-09-29, Apple M4 Pro,
Zig 0.16.0, ReleaseFast, the commit that brought Laya to Metal; worst stage per request, the
times with a commit per traced stage (31 command buffers; the first
includes pipeline warm-up):

| Request | Tokens | Options | ms | scaled max \|Δ\| (stage) | rel RMS | scaled logit \|Δ\| |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `choice_described` | 59 | 3 | 383 | 1.5e-6 (encoder.27) | 1.4e-6 | 1.1e-6 |
| `choice_labels` | 36 | 3 | 38 | 4.0e-6 (encoder.27) | 1.9e-6 | 8.3e-7 |
| `score` | 49 | 3 | 25 | 2.9e-6 (final) | 2.5e-6 | 2.5e-6 |
| `noul` | 56 | 2 | 27 | 2.9e-6 (head.0) | 2.4e-6 | 1.7e-6 |
| `json_state` | 119 | 2 | 48 | 7.6e-6 (final) | 8.9e-6 | 2.3e-6 |
| `long_text` | 512 | 3 | 210 | 1.4e-6 (head.0) | 2.1e-6 | 6.5e-6 |
| `long_list` | 512 | 2 | 209 | 6.0e-7 (final) | 1.1e-6 | 1.2e-6 |
| `choice_20` | 193 | 20 | 76 | 4.7e-6 (head.0) | 3.7e-6 | 6.5e-6 |

The check then runs all 8 requests as one packed batch (1,536 rows, 538
ms): each logit equals the request's run alone to the bit, since no
row's arithmetic depends on what is packed beside it. On the tiny
synthetic checkpoint it opens, runs, and closes the plan three times, and
fails every allocation of `open` in turn with no leak. The two kernels
have their own model-free checks in `make test-metal`: attention over
packed sequences of 1, 37, and 90 rows, full and within windows 0, 3,
and 64, with NaN past the batch, against F64 (worst 7.2e-7); the GeGLU
over strided fused rows against `cpu.geluErf`.

## `nuclis decide` through the fresh binary (2026-09-29)

Checked through the fresh binary, 2026-09-29: the 8 fixture requests
(`--json`) give exactly the package's answers and usage; the card's
quickstart gives identical JSON through `--request` (file and standard
input), `--questions`, and inline flags; one question fanned out over
four log files ranks them, a long one flagged as cut; malformed requests
fail with typed errors naming the question.

## Time per call (2026-09-29)

Apple M4 Pro, Zig 0.16.0, commit `792c47f`, best of three, the checkpoint
in the page cache; `load` is opening the directory (tokenizer, config,
weights decoded to F32), separate from the call:

| Call | Tokens | Load (ReleaseSafe / Fast) | Tokenize | Encode (ReleaseSafe / Fast) |
| --- | ---: | --- | --- | --- |
| one noul, the quickstart ticket | 56 | 647 / 197 ms | 0.1 ms | 269 / 259 ms |
| the quickstart's three questions | 164 | 613 / 194 ms | 0.2 ms | 801 / 772 ms |
| one question, a 512-token state | 512 | 596 / 193 ms | 0.9 ms | 2582 / 2458 ms |

Encoding is the cost; every question is its own sequence, so a call's time
is about the sum of its sequences' (a short sequence runs at a lower rate:
the matrices are too narrow to fill the cores).

**Metal against the CPU**, as a filter runs: one noul question over 1,
10, and 50 synthetic log states of about 60 and about 500 tokens
(`nuclis decide --backend … --json --request`, `timings_ms`). Apple M4
Pro, Zig 0.16.0, ReleaseSafe, the commit that brought Laya to Metal, best of three, the
pipelines compiled and the checkpoint in the page cache:

| States × state tokens | Input tokens | Encode, Metal | Encode, CPU | Metal, tokens/s |
| --- | ---: | ---: | ---: | ---: |
| 1 × 57 | 95 | 39 ms | 519 ms | 2,440 |
| 10 × 54–60 | 947 | 278 ms | 5,047 ms | 3,410 |
| 50 × 54–60 | 4,704 | 1,344 ms | 26,485 ms | 3,500 |
| 1 × 492 | 512 | 207 ms | 2,866 ms | 2,470 |
| 10 × 487–500 | 5,120 | 1,970 ms | 30,020 ms | 2,600 |
| 50 × 484–504 | 25,600 | 9,801 ms | — | 2,610 |

Load: 69–75 ms on Metal (the tokenizer, the config, the pipelines, the
wrapped mapping), 650–690 ms on the CPU. Metal is 13–20× the CPU.

Where the GPU time goes (`laya-check … --backend metal --profile`, each
dispatch timestamped, ReleaseSafe): for the 512-token `long_text`
request, 197 ms in kernels, attention 45 % (88 ms), the matmuls 52 %
(about 3.5 TFLOP/s), everything else 3 %; for all 8 requests packed
(1,536 rows, 539 ms), attention 39 %. The attention kernel runs one
threadgroup per (row, head) that reads every visible key itself, so its
cost grows with the sequence, which is why short states run faster per
token.

## The multilingual checkpoint checked (2026-09-29)

Checked by the same tools on the oracle's 16 requests
(`scripts/laya-reference.py --subfolder multilingual`,
`fixtures/laya-multilingual/`): the root set's 8 shapes, where
`long_text` and `long_list` now fill 1,024 tokens, then states in
French, German, Spanish, Arabic, Chinese, and Hindi (questions in the
state's language or in English), a language question, and an 889-token
French log whose one error line sits past position 512. The tensors kept
are encoder layers 0, 1, 3, and 21, `final`, and both head layers; the
bounds are the root set's. Gates `laya-multilingual-vocabulary`,
`laya-multilingual-cpu`, `laya-multilingual-metal` (the first tier verify,
the other two verify-release). Measured
2026-09-29, Apple M4 Pro, Zig 0.16.0, ReleaseFast, the commit that added the multilingual checkpoint;
worst stage per request, CPU then Metal:

| Request | Tokens | Options | CPU ms | CPU scaled max \|Δ\| (stage) | CPU scaled logit \|Δ\| | Metal ms | Metal scaled max \|Δ\| (stage) | Metal scaled logit \|Δ\| |
| --- | ---: | ---: | ---: | --- | --- | ---: | --- | --- |
| `choice_described` | 56 | 3 | 99 | 1.5e-6 (head.0) | 4.7e-6 | 35 | 1.2e-6 (head.0) | 2.3e-6 |
| `choice_labels` | 36 | 3 | 65 | 1.9e-6 (head.1) | 1.1e-6 | 10 | 1.6e-6 (head.1) | 1.6e-6 |
| `score` | 52 | 3 | 90 | 1.2e-6 (head.0) | 2.5e-6 | 11 | 2.9e-7 (head.0) | 1.8e-7 |
| `noul` | 56 | 2 | 97 | 1.0e-6 (head.0) | 7.7e-7 | 11 | 6.8e-7 (final) | 7.7e-7 |
| `json_state` | 124 | 2 | 212 | 8.9e-7 (head.0) | 1.7e-6 | 21 | 5.0e-7 (final) | 4.3e-7 |
| `long_text` | 1024 | 3 | 1957 | 2.5e-6 (head.0) | 9.2e-6 | 287 | 8.7e-7 (head.0) | 2.1e-6 |
| `long_list` | 1024 | 2 | 1947 | 7.6e-6 (head.1) | 9.2e-6 | 285 | 1.2e-6 (head.0) | 1.5e-6 |
| `choice_20` | 273 | 20 | 473 | 1.4e-6 (head.0) | 5.0e-5 | 47 | 6.4e-7 (encoder.21) | 2.7e-6 |
| `french_choice` | 77 | 3 | 134 | 1.7e-6 (head.0) | 1.4e-6 | 16 | 5.8e-7 (head.0) | 4.4e-7 |
| `german_noul` | 66 | 2 | 116 | 1.8e-6 (encoder.21) | 4.1e-6 | 15 | 5.8e-7 (final) | 2.6e-6 |
| `spanish_score` | 53 | 3 | 92 | 1.5e-6 (head.0) | 4.3e-6 | 11 | 5.2e-7 (head.0) | 2.4e-6 |
| `arabic_choice` | 44 | 3 | 78 | 2.4e-6 (head.0) | 2.1e-5 | 11 | 1.3e-6 (head.0) | 1.1e-5 |
| `chinese_choice` | 56 | 3 | 97 | 1.1e-6 (head.0) | 9.9e-7 | 11 | 4.9e-7 (encoder.21) | 3.2e-7 |
| `hindi_noul` | 58 | 2 | 102 | 8.2e-7 (head.0) | 1.2e-7 | 11 | 4.6e-7 (encoder.21) | 1.2e-7 |
| `language_id` | 36 | 7 | 65 | 1.4e-6 (head.0) | 5.2e-6 | 10 | 9.7e-7 (head.0) | 1.7e-5 |
| `long_state` | 889 | 3 | 1686 | 1.4e-6 (head.0) | 6.6e-7 | 234 | 6.0e-7 (head.0) | 1.1e-7 |

Worst relative RMS: 6.5e-6 on the CPU, 1.8e-6 on Metal. All 16 as one
packed Metal batch (3,924 rows, over two batches, 908 ms) give each
logit bit-identical to its run alone. Opening the weights alone took 63
ms on the CPU and 28 ms on Metal.

**Time per call**, measured the way the root set's was (one noul over
synthetic log states, `nuclis decide --json`, `timings_ms`, ReleaseSafe,
best of three, the commit that added the multilingual checkpoint, Apple M4 Pro), with states sized to
this tokenizer. The root set on the same states, Metal, is on the right.
The CPU skipped the 50-state rows of long states:

| States × state tokens | Input tokens | Encode, Metal | Encode, CPU | Metal, tokens/s | Root set, Metal: input, encode, tokens/s |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 × 88 | 121 | 31 ms | 246 ms | 3,950 | 94, 39 ms, 2,420 |
| 10 × 55–88 | 1,018 | 124 ms | 2,080 ms | 8,200 | 809, 230 ms, 3,510 |
| 50 × 55–88 | 4,998 | 577 ms | 10,223 ms | 8,660 | 3,984, 1,084 ms, 3,680 |
| 1 × 483 | 516 | 108 ms | 1,112 ms | 4,790 | 367, 140 ms, 2,620 |
| 10 × 450–486 | 4,965 | 858 ms | 10,786 ms | 5,790 | 3,552, 1,213 ms, 2,930 |
| 50 × 448–486 | 24,722 | 4,143 ms | — | 5,970 | 17,709, 6,037 ms, 2,930 |
| 1 × 936 | 969 | 286 ms | 2,289 ms | 3,390 | 512 (cut), 201 ms, 2,550 |
| 10 × 900–939 | 9,481 | 2,547 ms | 22,286 ms | 3,720 | 5,120 (cut), 1,926 ms, 2,660 |
| 50 × 898–939 | 47,295 | 12,612 ms | — | 3,750 | 25,600 (cut), 9,562 ms, 2,680 |

Metal is 8–18× the CPU. Per token the multilingual set runs 1.3–2.4×
the root set (its encoder is 22 layers of 768 against 28 of 1,024), less
for long states, where attention grows with the sequence. Load is 370–425 ms on Metal
and 570–600 ms on the CPU, against 73–87 ms for the root set: parsing
the 34 MB `tokenizer.json`.

**What it is for.** The oracle's language requests through both
checkpoints (`nuclis decide --json`, the answer and its confidence):

| Request | `laya` | `laya-multilingual` |
| --- | --- | --- |
| `french_choice` (department) | billing, 0.83 | billing, 1.00 |
| `german_noul` (threatens to cancel) | 0.06 | 0.93 |
| `spanish_score` (urgency, 0–2) | 1.56 | 1.54 |
| `arabic_choice` (sentiment) | neutral, 0.51 | negative, 0.13 |
| `chinese_choice` (department, question in Chinese) | billing, 0.32 | billing, 1.00 |
| `hindi_noul` (wants to close the account) | 0.00 | 1.00 |
| `long_state` (French log, most severe level) | info, 0.04 | error, 0.99 |

It reads the decision in other languages where the root set cannot.
Language identification is another matter: asked which of English,
French, German, or Spanish a one-sentence state is written in, the
multilingual set names French (0.34), German (0.35), and English right
and calls Spanish English (0.47); the root set names Spanish (0.91) and English
right and calls French Spanish and German English. The manual test's
three-option French question gives the multilingual set french 0.42
(the root set german 0.37).
