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

Theme agreed 2026-09-27 (user): Laya (`convaiinnovations/laya`, a
ModernBERT encoder with a typed-decision head, Apache-2.0) end to end in
nuclis, behind a decision API any client can call: a person writing the
state and questions by hand, a script, or later an LLM. Why it is worth
having beside an LLM: every question re-reads its state (the encoder is
bidirectional, so nothing is shared across questions), and an LLM must
spend decode tokens writing each question, so delegating one decision the
LLM could make itself saves nothing. It pays as a **filter**: one question
fanned out over many states the LLM never reads (30 search hits, every
log section, each diff hunk), where the LLM's prefill is the cost avoided
(docs/reference/laya.md gives the measured numbers). Landed before the
plan: safetensors pull (MODL-28) and the safetensors loader (MODL-29).
Kept to three units on purpose, to iterate fast.

MODL-30 closed 2026-09-29 (engineering log): `nuclis decide` runs Laya on
the CPU end to end, matching the `laya` 0.3.20 package exactly on its 8
oracle requests (gates `laya-vocabulary`, `laya-cpu`); the `laya`
catalogue entry and `decide.model`.

Added 2026-09-29 (user), ahead of the Laya units: cut debugging and
verification time. REPO-20 closed 2026-09-29 (engineering log): the
gates re-derived from what they protect (`make verify` 704 s → 246 s,
a `verify-release` tier) and `make verify-auto`, which runs the checks
and gates a diff selects and names the tiers it requires.

Reordered 2026-09-29 (user): Laya end to end first, then KERN-19 (the
threaded CPU reference), its "before" times not yet taken. MODL-31
closed 2026-09-29 (engineering log): Laya on Metal, `nuclis decide
--backend cpu|metal` (metal by default), packed batches, 13–20× the CPU
(a 512-token state in about 0.2 s), gate `laya-metal`.

Added 2026-09-29 (user), ahead of AGNT-18: the multilingual checkpoint.
Manual tests of the English one showed it cannot read other languages (a
French sentence scored German 0.37, French 0.30). Next is MODL-33; its
facts are read (below), so it starts at the oracle.

## Order

| Unit | Title | Sessions |
| --- | --- | --- |
| MODL-33 | Laya multilingual (mmBERT-base): the pull, the Metaspace tokenizer, the contract, checked on both backends | one or two |
| AGNT-18 | The agent's `decide` tool: LLM-written questions over tool-supplied states (the experiment) | one |
| KERN-19 | A threaded CPU reference, bit-identical: the CPU tier ≤ 30 min | one |

Decisions (user, 2026-09-27): the oracle is Laya's own Python package;
the root checkpoint first (multilingual later, same family code); CPU
first, then Metal; a new command, `nuclis decide`, with three input tiers,
fan-out over many states, a styled terminal view, and `--json`; each
`results[i]` a complete Jev response; LLM-written questions are the last
unit, an experiment.

## MODL-33 — Laya multilingual

Base: `01b65fe`

Facts read 2026-09-29 from the Hub (`convaiinnovations/laya` at
`55cf4c4e`, the same revision as the root set) and the `laya` 0.3.20
package (`.zig-cache/reference/laya-venv`, which loads it as
`Agent("convaiinnovations/laya", subfolder="multilingual")`):

- Files under `multilingual/`: `encoder/config.json`, `model.safetensors`
  (643,835,514 bytes, 169 F16 tensors and `temperature` F32),
  `rl_agent_config.json`, `tokenizer/tokenizer.json` (34,363,188 bytes),
  `tokenizer/tokenizer_config.json`.
- Encoder (mmBERT-base): 22 layers, hidden 768, 12 heads of 64,
  intermediate 1152, vocabulary 256,000, `layer_types` global every
  third from 0, `local_attention` 128 (window 64), both RoPE thetas
  160,000 (`rope_parameters`, which `modernbert.parseConfig` reads;
  the package's `_apply_rope_config` exists because Transformers 4.x
  would read 10,000 for the local one). Same tensor names as the root
  set. Head: 2 layers, `linear1` [3072, 768], 12 heads of 64;
  `act_head.0.weight` [256, 772]. Every matrix fits the Metal matmul
  (rows % 8, columns % 64: 768, 1152, 3072).
- `rl_agent_config.json`: `max_len` 1024, `head_max_len` 256,
  `temperature` [1, 1, 1], `temperature_by_options` {}, `max_prefixes`
  6 (check what reads it).
- Tokenizer: BPE, `byte_fallback` true, `fuse_unk` true, `unk_token`
  `<unk>`, 580,604 merges; normalizer `Replace " " → "▁"`;
  pre-tokenizer `Metaspace` (`▁`, `prepend_scheme` always, `split`
  true); decoder `Replace ▁ → " "`, `ByteFallback`, `Fuse`; 249 added
  tokens. `tokenizer_config.json`: `cls_token` `<bos>` (2), `sep_token`
  `<eos>` (1), `mask_token` `<mask>` (4), `pad` 0; the package's
  `build_sequence` uses `tok.cls_token_id`, `sep_token_id`,
  `mask_token_id`, and replaces `tok.mask_token` in texts.

Work:

- **Oracle.** `scripts/laya-reference.py --subfolder multilingual` runs
  the package on the staged `multilingual/` set and writes
  `inference/src/models/fixtures/laya-multilingual/` (`requests.json`,
  `activations.f32`, `tokens.json`): the root set's 8 request shapes
  plus French, German, Spanish, Arabic, Chinese, and Hindi states, a
  language-identification choice, and a state past 512 tokens (the
  1024 budget, positions past 512); tokenizer texts across scripts,
  emoji and unseen bytes (byte fallback), `<mask>` and `[MASK]` in text,
  whitespace runs, newlines, leading and trailing spaces.
- **Tokenizer.** `tokenizer/hf_json.zig` accepts this second shape
  exactly (other shapes still rejected by name) and encodes each
  Metaspace word through `bpe.encodeSpmBudget` (Gemma's SentencePiece
  BPE with byte fallback); `tokens.json` pins the semantics (where `▁`
  is prepended, how `split` cuts, added tokens before normalization).
- **Profile.** `profiles/laya.zig` `Specials` from
  `tokenizer_config.json` (`cls_token`, `sep_token`, `mask_token`;
  `[CLS]`/`[SEP]`/`[MASK]` when absent) and `unmask` with the
  checkpoint's mask text; the budget and temperatures from
  `rl_agent_config.json` (1024/256; check `head_max_len` is read).
- **Model.** Expected unchanged on both backends; the oracle's stages
  decide.
- **Catalogue.** A `laya-multilingual` decision entry: the five
  `multilingual/` files at the pin, into
  `~/.nuclis/models/convaiinnovations/laya/multilingual/`; `nuclis
  decide --model laya-multilingual`. No automatic routing by language
  (the package's `Router`).
- **Checks.** `vocabulary-check` and `laya-check` pick the fixture set
  from the checkpoint (or a `--fixtures` flag); gates
  `laya-multilingual-vocabulary`, `-cpu`, `-metal` (tier verify), a
  `laya_multilingual` model in `gates.json`; the same relative bounds
  as the root set. Measured: MODL-31's timing grid on Metal and the CPU;
  the manual tests' language question in
  `~/Downloads/laya-manual-tests.md` rerun on it. `make verify` once.

## AGNT-18 — The agent's `decide` tool (experiment)

- A tool in `src/agent/tools/` calling the in-process `Decider` (loaded
  on first use, `decide.model`): one question (choice, score, or noul
  with its options) over up to 64 candidates given as workspace file
  paths or as the items of the previous tool result; returns the
  candidates ranked with probabilities, truncation marked; limits are host
  constants; failures are results.
- Its description and the system-prompt line measured on the playground
  task list before and after (`make agent-eval VARIANT=…`), on Qwen3.8-27B
  and Gemma 4 E4B: task success, wall time, and prefill tokens saved.
  Kept only if it helps; the result is logged either way.

## KERN-19 — A threaded CPU reference, bit-identical

Split from REPO-20 at its close (user, 2026-09-29), which delivered the
fast gate tiers and `make verify-auto` (engineering log). Goal: the CPU
tier ≤ 30 min (hours today) and one family's CPU trace ≤ 1 min, with
not one bit of the reference's output changed.

Base: to be recorded when it starts (`d46e907` was recorded before the
reorder; take the commit before its first change).

- First: `python3 scripts/gates.py --tier verify-cpu --json >
  .zig-cache/gates/kern19-before.json` in the background, the per-gate
  "before" times (14 gates, hours; nothing else running meanwhile, since
  compiles and GPU runs distort it), then copy
  `.zig-cache/gates/trace/*-trace-cpu/` to `.zig-cache/gates/kern19-before/`
  for the byte comparison.

- Split independent work across `Io` tasks without changing any sum's
  order, the pattern of `backends/cpu/dense.zig` (`std.Io.Group`,
  `taskCount`): `backends/cpu/root.zig` `matvec` by rows (each row still
  one F64 sum in column order; the decode scratch becomes one `columns`
  slice per task, so the callers' workspaces grow to `taskCount ×
  columns`), `attention.apply` by head, `experts.ffn` by expert, and the
  vision encoders' matvecs. The kernels take `std.Io`; the family
  runtimes (`*_runtime.zig`, `vision/*.zig` CPU paths) pass theirs.
- Proof of no change: every `*-trace-cpu` directory and logits file
  `cmp`-identical before and after (`cmp -r` against
  `.zig-cache/gates/kern19-before/`), all `verify-cpu` gates passing;
  times before and after per gate.
- `make verify-cpu` once (this changes how the reference runs, not what
  it computes; the byte comparison is the evidence).
