# TODO — active plan

This file is the only progress tracker. It holds **unfinished work only**:
where we are, what is next, and the design of each remaining unit. When a
unit closes, its outcome moves to
[docs/engineering-log.md](docs/engineering-log.md) and
its section is deleted here; when the last unit closes, this file is emptied
back to this header. Requirements live in [docs/spec.md](docs/spec.md); the
engine map in [docs/architecture.md](docs/architecture.md); how to build,
test, and measure in [docs/development.md](docs/development.md).

Session protocol (also in [AGENTS.md](AGENTS.md)): read this file first. If
it lists work, summarize *Where we are* and ask the user how to continue. If
it is empty, ask what to work on and write the agreed plan here.


## Where we are

**Next: MODL-34 session 1** (its section follows this one): clef-flash,
first pulling its files with `nuclis model pull`. Record its `Base:` at
the first change. APPS-19 closed 2026-10-02: `nuclis serve` keeps
decision models open behind the nuclis API on 127.0.0.1:8000
([api.md](docs/reference/api.md)), TypeSafe's Jev call included, and
batches requests that wait together (1.24–1.30× at concurrency 16, the
Laya encode's packing ceiling, below the planned 2×); MODL-34's catalogue
entry is served by it once it lands. The order of the units that follow
is the table at the end of this section; the history of the speed theme
comes first.

Theme agreed 2026-09-30: **decode speed on Metal**, Qwen3.8-27B first,
toward the 20 tokens/s of [ADR 0001](docs/adr/0001-qwen-decode-verifier.md)
(proposed). ENGN-18 closed 2026-09-30: saved prefixes, `bench
--verify-rows`, and `make speed` / `make speed-base` exist, and the
opening baseline is in
[bench.md § The decode-speed baseline](docs/reference/bench.md#the-decode-speed-baseline-engn-18-2026-09-30).
The base binary for `make speed` lives in `.zig-cache/speed/base/` (not
committed; `make speed-base` after each kept change). KERN-20 closed 2026-09-30: captures read in Xcode,
`bench --kernel-stats`, and
[apple-gpu.md](docs/reference/apple-gpu.md) with four kernel readings; KERN-05
and KERN-12 answered. KERN-21 closed 2026-09-30: verify batches run the flash-decoding split
pass (Qwen's 4-row C 387 → 288 ms at 4K, 1,157 → 376 at 32K; Gemma 12B
908 → 200 at 32K), checked at depth by `qwen38-verify-depth-*`. KERN-24 closed 2026-10-01
below its target: the register-fragment tile and two routings cut Qwen's
4-row verify C at 4K from 289 to 244 ms (512: 275 → 230) and Gemma 12B
QAT's by 23 %. ENGN-19 closed 2026-10-01: verify batches of up to 8 rows
step DeltaNet without writing the state and recovery replays a tape, so
Qwen's 4-row C fell 48–51 ms at every depth (4K 239 → 191 ms, 32K 329 →
279) and the 1.25 GB row-slot region is gone. ENGN-20 closed 2026-10-02:
speculation re-priced on every family; the catalogue turns it on for Qwen
(draft 7: prose 1.25–1.57×, short code 20.3–20.9 tok/s), Gemma 12B QAT
(5), E4B (6), and Muse (6), and off for Gemma 26B-A4B; `make
spec-matrix` measures it, and `agent --print` now loads the drafter. An
existing `~/.nuclis/nuclis.json` keeps its entries' old values until the
user edits them or re-runs `config init`. The base binary for `make
speed` is at ENGN-19's commit (no engine arithmetic changed since).
REPO-27 closed 2026-10-02: an external review's fixes (a failed
recording is discarded, so metal-check reports instead of hanging; the
kernel table is derived from `Kernel`; `--file` names a support file's
weights; parser property tests). Its log entry lists what waits for
KERN-23: pruning the 56 check-only pipelines (1.70 s of a cold start)
and the GGUF type-id enum. No engine arithmetic changed, so the `make
speed` base binary stands.

**Re-ordered 2026-10-02 (user).** Decision models move ahead of the last
speed unit: first APPS-19 (closed), `nuclis serve`, the nuclis API
(`src/api/`) with decisions as its first service, an OpenAI-compatible
service expected later; then MODL-34,
Cloudflare's clef-flash decision model; then KERN-23, which closes the
decode-speed theme. Dropped (engineering log): KERN-22, long-context
decode attention, a small win at 32K only; AGNT-18, the agent's `decide`
tool, since decision models are served by `nuclis serve` and the agent
stays a tool for language models.

| # | Unit | Sessions |
| --- | --- | ---: |
| 1 | MODL-34 — clef-flash: Cloudflare's 9B decision model, text then vision | 4 |
| 2 | KERN-23 — Weight streaming for one row and a few (closes the decode-speed theme) | 2 |
| — | AGNT-19 — Saved prefixes for the agent across processes | queued after KERN-23 |

## MODL-34 — clef-flash: Cloudflare's 9B decision model, text then vision (4 sessions) — next

Base: recorded when the unit starts.

[Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash)
(Apache-2.0) answers the same typed questions as Laya (`noul`, `choice`,
`score`; one logit per option, softmax per question) about a text or JSON
state, optionally with images or video. It is Qwen3.5-9B with its vision
encoder plus a **joint schema head**, a small transformer over the
backbone's final hidden states that routes evidence to each question and
scores every option of every question in one pass (Laya runs one pass per
question). States up to 16,384 tokens. The card reports Decision Index
0.2.1 results and a 38.8 ms median latency on an H200 (torch 2.11,
transformers 5.10.2).

Known facts (2026-10-02):

- **Backbone** (`config.json`): `Qwen3_5ForConditionalGeneration`, text
  `qwen3_5_text`, 32 layers, hidden 4,096, `linear_attention` layers with
  `full_attention` every 4th, 16 heads, 4 KV heads, vocabulary 248,320:
  the hybrid DeltaNet family `models/qwen35.zig` runs for Qwen3.8.
- **The qwen35 adapter rejects it today** (`nuclis model inspect
  bartowski/Cloudflare_clef-flash-GGUF --file Cloudflare_clef-flash-Q6_K.gguf`:
  `UnsupportedConfiguration`): `integer_settings` in `models/qwen35.zig`
  pins Qwen3.8-27B's shape. The Q6_K header (read 2026-10-02) differs in
  six values: `embedding_length` 4,096 (pinned 5,120),
  `feed_forward_length` 12,288 (17,408), `attention.head_count` 16 (24),
  `block_count` 32 with no `nextn_predict_layers` (64 or 65),
  `ssm.time_step_rank` 32 (48), `ssm.inner_size` 4,096 (6,144). Every
  other pinned value matches: context 262,144, KV heads 4, key/value
  length 256, conv kernel 4, state size 128, group count 16, full
  attention every 4, rope 64 with sections [11, 11, 10, 0], base 1e7,
  epsilon 1e-6. So the adapter, its CPU runtime, and its Metal plan read
  a shape from the file instead of constants, and every Qwen3.8 gate
  passes unchanged. The Q6_K file is 125 Q6_K tensors, 77 Q8_0, 225 F32.
- **The projector is Qwen3.8's, one width apart.**
  `mmproj-Cloudflare_clef-flash-bf16.gguf` (0.92 GB, 334 tensors, BF16 +
  F32) is `clip.projector_type = qwen3vl_merger`, image size 768, patch
  16, hidden 1,152, feed-forward 4,304, 27 blocks, 16 heads, merge 2,
  GELU, epsilon 1e-6, no deepstack layers: exactly what
  `vision/qwen3vl.zig` pins, except `clip.vision.projection_dim` 4,096
  against its `output_width = 5120`. Supporting it is reading the width
  from the file, not a new projector.
- **Head**: `joint_head.safetensors` (0.244 GB, BF16),
  `joint_head_config.json` `{hidden_size 4096, width 1024, routing_layers
  2, layers 4, heads 16, feedforward 4096}`; its forward and the input
  construction are in `joint_schema_model.py`. The head stays BF16,
  decoded to F32.
- **Weights: the Q6_K GGUF backbone, the bf16 mmproj, the original head.**
  [bartowski/Cloudflare_clef-flash-GGUF](https://huggingface.co/bartowski/Cloudflare_clef-flash-GGUF)
  (llama.cpp b11279, imatrix) carries the backbone and the projector, not
  the head. Q6_K (7.79 GB): a decision is one prefill, compute-bound, so
  fewer bits buy little speed and cost flipped answers. No bf16 backbone
  is pulled and no quantization agreement is measured (decided
  2026-10-02); Q6_K against Q8_0 inside nuclis is the cheap measurement
  if a number is ever wanted.
- **Rejected: MLX 4-bit** (`mlx-community/clef-flash-4bit`, affine,
  group 64): its own card reports 96.4 % agreement with bf16 and a mean
  probability difference of 0.040, too many flipped decisions for a
  decision model, and nuclis has no kernels for the format.
- **Estimate, not a measurement**: about 20× Laya's compute; from Qwen
  27B's 90 tokens/s prefill, roughly 1–3 s per 500–1,000-token decision
  on the M4 Pro.

**How it is checked: the new parts only** (decided 2026-10-02). The
backbone is the qwen35 family, already validated for Qwen3.8 on the CPU
and Metal, and the projector is Qwen3.8's; neither gets a Python
reference. Their checks: Qwen3.8's gates unchanged after the shape is
read from the file, nuclis's CPU and Metal agreeing on clef-flash, and
`config.json` and the GGUF metadata showing no flag that changes the
arithmetic. What is new gets a reference that needs no backbone weights:
(1) **the sequence**: the template, where questions, options, and the
state sit, which positions the head reads, how many tokens an image
becomes, from the tokenizer, template, and processor files alone; (2)
**the head**: `joint_schema_model.py`'s head in torch on the CPU (0.24
GB), fed the final hidden states nuclis's CPU backbone dumps, its logits
the fixture. The backbone risk nothing else catches (a difference neither
file reveals) is covered by a **sanity set**: 10 requests, text and
image, whose answers are obvious, plus the card's examples if it prints
probabilities; each must pick the expected option.

### Session 1: pull, read, the shape on the CPU, the sequence reference

**First, before reading or writing anything, pull the files** with the
fresh `./zig-out/bin/nuclis model pull`, pinned by commit (bartowski
`d7f376ea88c05e7bb1014dd5351a93df9dd8029e`, Cloudflare
`17f0b0ad64efb65d273590632833508766b2aae6`, both read 2026-10-02):

```sh
R1=d7f376ea88c05e7bb1014dd5351a93df9dd8029e
R2=17f0b0ad64efb65d273590632833508766b2aae6
./zig-out/bin/nuclis model pull bartowski/Cloudflare_clef-flash-GGUF --revision $R1 \
  --file Cloudflare_clef-flash-Q6_K.gguf                       # 7.79 GB, the backbone
./zig-out/bin/nuclis model pull bartowski/Cloudflare_clef-flash-GGUF --revision $R1 \
  --file mmproj-Cloudflare_clef-flash-bf16.gguf --role mmproj  # 0.92 GB, the projector
./zig-out/bin/nuclis model pull Cloudflare/clef-flash --revision $R2 \
  --file joint_head.safetensors                                # 0.24 GB, the head
```

About 9 GB under `~/.nuclis/models/`, all kept: they are what the
catalogue entry runs. The head pull brings the `.json`/`.jinja` files
beside it (`config.json`, `joint_head_config.json`, `tokenizer.json`,
`tokenizer_config.json`, `chat_template.jinja`, `processor_config.json`,
`generation_config.json`), and not the backbone shards. `joint_schema_model.py`
is not a file nuclis pulls (`.py`); the reference script fetches it with
`hf_hub_download` at `R2`. If a pull fails, that is a bug in `nuclis
model pull` to fix first (an APPS side unit), not a reason to download
by hand.

Then:

- **Read** `joint_schema_model.py`, `chat_template.jinja`,
  `processor_config.json`, `tokenizer_config.json`: the sequence, the
  head's forward exactly, the calibration, the budgets
  (`max_state_tokens`), how images enter.
- **The qwen35 shape from the file, on the CPU.** `models/qwen35.zig`
  replaces the six pinned values with a shape read from the metadata and
  checked against the tensor dimensions (the unchanged constants stay
  pinned); `qwen35_runtime.zig` sizes buffers and loops from it.
  `nuclis inspect` accepts the Q6_K file; Qwen3.8's CPU gates pass
  unchanged.
- **`scripts/clef-reference.py`** (a venv beside `.reference/laya-venv`,
  torch, transformers, and tokenizers pinned): the sequence mode writes,
  for 8 requests (the Laya shapes, a 4,000-token state, a
  multi-question request), the ids and the head's input positions under
  `inference/src/models/fixtures/clef/`; the head mode is written now
  and run in session 2.

End the session by rewriting sessions 2–4 at the level of files,
functions, and numbers.

### Session 2: the decision seam and the head on the CPU

A decision-family seam in `inference/src/decide.zig` (today it composes
Laya only): a family's profile builds sequences and calibrates, its model
returns per-option logits; Laya moves behind it unchanged (its gates
pass). The seam keeps `prepare` and `decideJobs`, which `nuclis serve`
batches through (`src/api/decisions/batcher.zig`; a family's pass limit
replaces `decide.batch_rows`). `profiles/clef.zig` (the sequence and budgets, equal to the
sequence fixtures), `models/clef.zig` (the head on
`backends/cpu/dense.zig`), and the backbone's final hidden states from
`qwen35_runtime.zig` without the output head. The CPU backbone dumps the
head's input for the 8 requests (under `.zig-cache/clef/`, never
committed); the script's head mode turns each dump into logits, the
committed fixture; nuclis's head on the same input matches them within
Laya's relative bounds, and the text sanity requests pick their expected
options (`zig build test-clef`, gate `clef-cpu`).

### Session 3: Metal, the catalogue, timings

The backbone through `qwen35_metal.zig`'s prefill with the shape from
the file (any kernel or plan constant tied to 5,120 / 17,408 / 24 heads
/ 48 value heads found and parameterized; Qwen3.8's Metal gates and
`make speed` unchanged), the head as a Metal plan beside Laya's, CPU and
Metal agreeing on the 8 requests within Laya's bounds. The catalogue
entry `clef-flash` (kind `decision`; the GGUF backbone, the mmproj as its
`mmproj` companion so `nuclis model pull clef-flash --with mmproj`
fetches it, the head and support files from Cloudflare's repo, all
pinned by the revisions above); `nuclis decide --model clef-flash`;
`nuclis serve` lists and serves it. Timings for 1, 10, and 50 states at
500 and 2,000 tokens, recorded in a new `docs/reference/clef.md`. Gate
`clef-metal`. If parameterizing the Metal plan is larger than a session,
this session splits and the table says so.

### Session 4: vision

**mmproj support is part of the unit; it does not close text-only.**
The pulled `mmproj-Cloudflare_clef-flash-bf16.gguf` through
`inference/src/vision/qwen3vl.zig`: `output_width` becomes the file's
`clip.vision.projection_dim`, accepted when it equals the backbone's
embedding length (4,096 here, 5,120 for Qwen3.8), on the CPU projector
and the Metal plan (`qwen3vl_metal.zig`); Qwen3.8's vision gates pass
unchanged. Images in the decision request (the field the reference
defines, mirrored in `nuclis decide --request` and `nuclis serve`), the
image token count equal to the sequence reference's, CPU and Metal
agreeing, and the image sanity requests picking their expected options;
tokens and time per image recorded.

Gates: `make check`, `make verify-auto`, `make verify` (it touches the
inference stack), and `make verify-cpu` once, since it brings up a family
and changes the qwen35 CPU forward.
Docs: `docs/reference/clef.md` (new: the sequence, the head, the sanity
set, the timings), `docs/spec.md` (the catalogue and `decide`),
`docs/reference/artifacts.md`, `docs/architecture.md` (the decision
path's family seam), `THIRD_PARTY_NOTICES.md` if a constant is taken
from the reference.

## The theme: decode speed on Metal

**Why this order.** Our own records already say where the verify batch
goes (ADR 0001, *Where the verify cost goes*): the chunk attention a
verify runs on 24 threadgroups costs about 103 ms per batch at 4K and
413 ms at 16K, and decides every context from 4K up. The single-row path
runs at 63 % of the 273 GB/s peak against about 78 % for MLX on the same
chip, and every verify budget also needs that gap closed. So: tools that
make an experiment take a minute (ENGN-18) and let us see the GPU's
limiters (KERN-20), then the levers ranked by measured cost.

**What the other families get.** Every lever below except ENGN-19 sits in
shared code:

| Lever | Unit | Qwen 27B | Bonsai 27B | Gemma 4 (E4B, 12B, 26B-A4B) | Muse 30B |
| --- | --- | :---: | :---: | :---: | :---: |
| Few-query split-KV verify attention | KERN-21 | ✓ | ✓ | ✓ (draft heads) | ✓ (speculation on) |
| Single-row and few-row matvec bandwidth | KERN-23 | ✓ | ✓ | ✓ (their encodings) | ✓ |
| DeltaNet recurrent verify, replay tape | ENGN-19 | ✓ | ✓ | — | — |

Bonsai runs on the Qwen adapter (`qwen35_metal.zig` with its Hadamard
fold); Gemma and Muse verify through the same `attentionChunk` and
`mmRows` routes. Every unit re-measures each family's decode it touches
through `make speed` before it closes.

### The loop (every experiment in every unit)

Fast, cheap to abandon, and nothing lands without a measured gain.

1. **Idea → micro-bench** (no model, seconds): `make bench-attention`,
   `make bench-matvec-rows`, `make bench-kernels`, or a new sweep in
   `inference/metal-check.zig`. Write the prediction down first, in the
   unit's ledger below. A kernel that misses its micro prediction by more
   than half is dropped here.
2. **Correctness in seconds**: `make test-metal` (the kernel fixtures,
   against the F64 CPU references, poison past the visible rows).
3. **End-to-end A/B** (about a minute per context, ENGN-18): `make speed
   ARGS='--contexts 512,4096,16384,32639'` interleaves the saved base
   binary and the candidate on restored prefixes.
4. **Keep rule.** Keep a change only when its median decode (or, for a
   verify lever, the batch cost C at the unit's row counts) improves by
   **≥ 2 %** at one context or more and no context regresses by more than
   1 %, over ≥ 5 interleaved pairs; the record's run-to-run spread is about
   1.5 %. Then `make verify-auto` (plus `make verify-long` for attention or
   cache changes) and commit `perf(inference): …` with the numbers in the
   body. Refresh the base binary (`make speed-base`) after each kept commit.
5. **Otherwise revert** (`git restore`, nothing committed), and add one
   ledger line: idea, prediction, measured, why it lost. The ledger is
   committed with the next kept change or at session end, and moves into
   the log when the unit closes, so a negative result stays findable and
   is not re-tried unchanged.

**Be brave, but bounded.** A unit lists more ideas than it needs; try the
cheapest falsifying measurement of each, and spend a session on none that
has not beaten its micro-bench. ENGN-18 and KERN-20 are tooling and land
without a speed gain; every other code change lands only through step 4.

**Apple GPU knowledge** found along the way (limiters, occupancy,
register limits, load widths, measured on this M4 Pro) goes into
[docs/reference/apple-gpu.md](docs/reference/apple-gpu.md), created by
KERN-20, with the source or the measurement for each fact.

### Order

| # | Unit | Sessions | Lands when |
| --- | --- | ---: | --- |
| 1 | KERN-23 — Weight streaming for one row and a few: decode and the verify body | 2 | 512 decode ≥ 11.5 tok/s, a 4-row verify C ≤ 150 ms at 512, or Qwen speculative prose 512 ≥ 18 tok/s; or closed at its ledger |

**Re-ranked after ENGN-20** (2026-10-02): speculation is on for Qwen
(draft 7), Gemma 12B QAT (5), E4B (6), and Muse (6), so a verify lever now
moves the default decode rate, and every decode lever moves the plain
path the drafters fall back to. Qwen's C / 50E at draft 7 is 1.21–1.42 at
512 and 4K (C 164–182 ms against 128–141 ms budgets) and 1.6–1.9 from
16K; both Gemma entries break even at 32K, where the verify attention is
linear in rows, so KERN-22 now pays in speculation as well as decode.
The drafter's proposal is 16–18 ms per Qwen batch at draft 7, its head
the output head KERN-23's idea (f) names.

**Re-ordered after ENGN-19** (user, 2026-10-01): ENGN-20 first, since
speculation now pays (an estimated 1.2–1.3× from the ENGN-17 acceptance
at the new C) and its measured E decides how much each decode lever is
worth; then KERN-23, the largest per-token lever (matvecs 85 of ~95 ms
per step) and the groundwork for a multi-row scalar verify body; then
KERN-22, which pays mainly at depth (about 14 ms per step at 32K, plus
the verify's attention there). Pull KERN-22 forward if long agent
sessions become the main use.

**Re-ranked after ENGN-19** (`--profile`, 4-row Qwen verify at 4K,
2026-10-01): matrices 195 ms, attention 14.5, DeltaNet 6.4 (was 41.3),
recover 2–3 ms, checkpoint 3. The verify's remaining cost is the matrices'
floor KERN-24 found; decode (KERN-22, KERN-23) is next, and ENGN-20
re-prices speculation at the new C.

**Re-ranked after KERN-24** (`--profile`, 4-row Qwen verify at 4K,
2026-10-01): matrices 195.3 ms (was 236.8; 2.3 decode steps), DeltaNet
41.3, attention 14.5. The matrices stay the largest term but KERN-24
found their floor (the padded 8×8 multiplies); ENGN-19 is next, then
decode. ENGN-20 must price the fragment tile's flat cost: a 7-draft
verify pays the same matrices as a 3-draft one.

**The order was ENGN-18's cost table** (bench.md § The decode-speed
baseline, `--profile` of a 4-row verify and a decode step, 2026-09-30).
A 4-row verify batch at 4K is 399 ms of kernel time: weight matmuls 237
(2.8 decode steps' worth), verify attention 106, DeltaNet 44; at 32K
attention is 824 of 1,114. KERN-21 first: the largest term at depth, and
the drafter's commit (48 ms per batch at 32K) runs the same
`attentionChunk`. KERN-24 next: the largest term at 512 and 4K. ENGN-19:
`delta_chunk` 38 ms + recover 14 + checkpoint 3 per batch. Then decode:
KERN-22 (decode attention 5.2 ms per step at 4K, 27.5 at 32K) and KERN-23
(weight matvecs 85 ms per step at every depth).

Identifiers are provisional in this order; they are fixed in the order
the units close. The units are independent of one another: re-rank them
when a kept change moves the cost table.

## KERN-23 — Weight streaming for one row and a few: decode and the verify body (2 sessions)

Base: recorded when the unit starts (APPS-19 and MODL-34 land first).

**Why, since ENGN-20.** Speculation is on for Qwen at draft 7, so a
default turn spends its time in verify batches: at 512, C 164 ms = propose
16.5 + checkpoint 3 + verify 138 + recover 2.5 + commit 4 (cold, greedy;
bench.md § The re-priced speculative verdicts). The verify's matrices sit
at KERN-24's floor: the fragment tile pads every 8×8 multiply and costs the
same from 1 to 8 rows, its rates at 4 rows 107 GB/s (Q4_K) to 152 (Q6_K),
against 179–212 GB/s for the single-row matvec in `make bench-kernels`.
Plain decode still runs wherever speculation is off (Gemma 26B-A4B, Bonsai,
files outside the catalogue, a turn after an image) and is 85 ms of
matvecs in a 95 ms step at 512.

**What the counters say** (apple-gpu.md, at full clocks): the Q4_K matvec
is issue-bound on the integer and complex pipe, not on memory, at half its
target occupancy with 192 registers. The multi-row matvec kernels for 2–8
tokens already exist (`nu_matvec_rows_{q4_k,q5_k,q6_k,iq4_xs}_t2..t8`,
`nu_matvec_rows_k_body` and siblings in `kernels.metal`), but only 2 rows
route to them (KERN-12): beyond that each token's accumulators push the
registers and the body loses to the tile. KERN-24's close names the gap:
a scalar body that shares activations across rows without that register
wall; none was tried. Cheaper decode arithmetic and fewer live registers
serve both halves of this unit.

**Session 1 — the single-row matvec.** Ideas in this order, each judged
first on `make bench-kernels` (Q4_K, Q5_K, Q6_K, IQ4_XS, IQ4_NL, Q8_0 at
Qwen's shapes), then by `make speed` decode at 512 and 4K: (b) the half
magic-number decode (a nibble OR'd into a half of exponent 1024, minus
1024) with float accumulation, per encoding; (c) scale and bias once per
group, `s·Σqx + b·Σx`; (d) 16-byte aligned loads per lane; then (a)
weights in Metal-allocated buffers instead of `newBufferWithBytesNoCopy`,
only if a capture shows an MMU limiter; (e) the command buffer split so
the GPU starts while the CPU encodes; (f) the Q6_K output head (about 1
GB) as its own kernel, which the drafter pays per proposed position too.
`--kernel-stats` reads each candidate's thread limit before it is timed.

**Session 2 — the multi-row body.** Carry session 1's decode into the
`_t3..t8` bodies with the accumulators bounded (rows × tokens per
SIMD group chosen so the thread limit stays at 1,024 in
`--kernel-stats`), and measure every encoding at 3–8 tokens against the
fragment tile with `make bench-matvec-rows ARGS="8 frag"`. Where the body
beats the tile at a token count, route verify batches of that count to it
(the routing that sends 2 tokens to `matvec_rows` today, in
`inference/src/backends/metal/root.zig`), per encoding. Kept only through
`make speed --verify-rows 4,8` and the speculative record below; Gemma's
Q4_0 verify gets the same body if Q4_0 joins.

**Predictions** (written before code; each idea's own goes into the
ledger):

- Q4_K single-row matvec 179 → ≥ 200 GB/s in `make bench-kernels`; 512
  plain decode 10.56 → ≥ 11.5 tok/s.
- A 4-token multi-row body ≥ 150 GB/s on Qwen's Q4_K and IQ4_XS shapes
  (the tile reads 107 and 122); Qwen's 4-row verify C at 512 177 → ≤ 150
  ms in `make speed`.
- Qwen prose 512 with speculation (`make spec-matrix ARGS='--model qwen38
  --contexts 512 --drafts 7 --cooldown 90 --rev <rev>'`, greedy and
  instruct): 16.5 / 15.9 → ≥ 18 tok/s.

**Lands when** any prediction is met with no context regressing, or the
unit closes at its ledger. **Correctness:** the per-encoding matvec and
multi-row fixtures in `metal-check` against the F64 CPU references
(poisoned rows past the visible ones), the trace gates of every family
whose encodings change, `qwen38-verify-depth-*` and
`qwen38-generation-metal` (verify rows against stepped decode), `make
verify`.

Gates: `make test-metal`, `make verify-auto`, `make verify`; the
speculative record above for the third prediction.

## AGNT-19 — Saved prefixes for the agent: the primed prefix and `/resume` across processes (2 sessions) — queued

`nuclis agent` prefills its system block and tool definitions at every
start (about 11 s for 934 tokens on Qwen, llm-guide.md § 22) and replays a whole
conversation on `/resume` (minutes at 16K). ENGN-18's `src/prefix_cache.zig`
already writes and restores a model snapshot keyed by model files, tokens,
and layout; this unit uses it for real work, where a stale state is a
correctness bug, not a timing one.

- **Key.** `prefix_cache.Key` gains the build: the binary's
  `nuclis --version` string and the git revision from `build_options`
  (a dev build's revision plus a dirty flag). A different build is a
  miss, never a restore: a saved state must equal what this build's
  prefill would compute. The file header records all four keys.
- **Session 1: the primed prefix.** `Completer.prime` (`src/agent/loop.zig`)
  first calls `prefix_cache.load` from `<NUCLIS_HOME>/cache/prefix/`
  (`src/paths.zig`); on a hit it restores and skips the prefill, on a
  miss it prefills as today, snapshots (already done), and saves.
  `restorePrimed` is unchanged. A `/ctx` change re-opens the engine and
  keys on the new capacity. Speculation on and off are different layouts
  (the draft block's cache), so each has its own file.
- **Session 2: `/resume`.** When the agent writes a session file
  (`src/agent/history.zig` / the save path), it also saves the model
  snapshot at the end of the last completed turn, keyed by the consumed
  tokens; `/resume` restores it when the rendered prefix's tokens match
  and prefills only the remainder, falling back to the replay otherwise.
- **Bounds.** A per-file bound (the session's used extent: about 150 MB
  of recurrent state plus 64 KiB per token for Qwen, 2.3 GB at 32K) and a
  directory budget (`cache.prefix_bytes` in `nuclis.json`, default 8 GB;
  0 disables), evicted oldest-accessed first after each save. `nuclis
  cache ls` and `nuclis cache clear` (APPS surface) show and empty it.
- **Limits, stated in the docs.** Exact token prefixes only (an edited
  system prompt or tool list misses); bound to capacity, KV precision, and
  the draft layout; disk cost as above.
- **Prediction.** The agent's first prompt appears with the primed prefix
  restored in ≤ 0.5 s instead of ~11 s; `/resume` of a 16K conversation
  in ≤ 2 s instead of minutes.
- **Correctness.** Unit tests: a build-key miss, a budget eviction, a
  corrupt file refused and re-prefilled (the ENGN-18 tests extended). A
  restored primed prefix gives the same first-turn tokens as a fresh
  prime (greedy, both backends' generation checks). The agent surface
  through `make shot` (a cold start, then a warm start showing the
  restore; a `/resume`), and `make agent-eval` before and after, since
  the loop's behaviour changes.

Gates: `make check`, `make lint-py`, `make verify-auto`, `make shot`,
`make agent-eval VARIANT=…`. Docs: `docs/reference/session.md`,
`docs/development.md` § User directories, the agent's help.
