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

MODL-12 closed on 2026-09-19: the Muse Glimmer Metal plan matches the
pinned traces in both cache precisions, passes the generation check,
and decodes at 9.99 tok/s (71 % of the reference; the per-kernel
profile is in [docs/reference/muse-glimmer.md](docs/reference/muse-glimmer.md)).
Next is MODL-13, the profile (design below): `nuclis bench` and `agent`
still refuse the file (`UnsupportedPromptTemplate`) until it lands.

Order: MODL-13 → AGNT-10.
After AGNT-10 the roadmap continues with speculative decoding across the
families, then performance (the ternary matvec arithmetic among it), then
vision ([docs/roadmap.md](docs/roadmap.md)).

| Unit | Title | Sessions |
| --- | --- | --- |
| MODL-13 | Muse Glimmer 30B: profile (text, reasoning channel), catalogue, acceptance | 1 |
| AGNT-10 | Muse Glimmer ATEM tool calling: rendering, decoding, fixtures | 1 |

## Muse Glimmer 30B — the artifact (decided 2026-09-16)

**`Muse-Glimmer-30B-UD-Q4_K_XL.gguf`** at commit
`faa5b025c584459c13febfa5c59883516710ae39`, 15,878,222,368 B, SHA-256
`82bece304887a313ece08400bc030f6066c7bff5b906b0cd40308ec8a409fd38` (from
the listing; verified by the pull in MODL-11). Header: architecture
`muse-glimmer`, 52 blocks, embedding 6656, context 131072, 731 tensors:
313 F32, 410 Q4_K, 8 Q5_K (`output.weight` among them). Verdict today:
*not runnable: no adapter* — every encoding already has CPU and Metal
kernels, so the encoding side needs nothing.

Why this quantization: it is Meta's "K-Quant-17GB" tier (1.0 % measured
degradation across 15 benchmarks) and Unsloth's recommended starting point;
at 15.9 GB it leaves the 48 GB machine wide headroom for a 32K F16 KV cache
(52 layers × 1 KB/token ≈ 1.7 GB), prefill scratch, and later the vision
projector (`mmproj-kquant.gguf`, 1.40 GB) and the DFlash drafter
(`dflash-kquant.gguf`, 1.63 GB). Decode is memory-bound, so the projection
is the Qwen3.8-27B rate at the same byte count (~10 tok/s). The larger
files are quality upgrades to measure afterwards, not bring-up targets:
`UD-Q5_K_M` 19.2 GB, `UD-Q6_K_XL` 26.3 GB (Unsloth's "Mac 48 GB" row;
check its encodings by `inspect` first), `Q8_0` 29.6 GB (too tight beside
the cache and companions).

Companions in the repository, sizes from the listing (the pulled ones'
digests are in artifacts.md): `mmproj-kquant.gguf` 1,400,328,928 B,
`mmproj-Muse-Glimmer-30B-BF16.gguf` 3,849,173,728 B,
`dflash-kquant.gguf` 1,631,205,312 B.

## MODL-13 — Muse Glimmer 30B: profile (text, reasoning channel), catalogue, acceptance

**Design.**
- `profiles/muse_glimmer.zig` pinned to the GGUF template digest. The
  prompt starts with `<|begin_of_text|>` as text (the encoder never adds
  BOS). Turns are `<|start|>ROLE<|message|>…<|eot|>`; two consecutive
  messages of one role end the first with `<|eom|>`. The system turn is
  the caller's text followed by `\n\nReasoning strength: LEVEL.` and
  `\n\n# Valid recipients: "self", "user".` (with tools, the tool block
  and namespaces come in between — AGNT-10). Without a system message the
  template synthesizes one ("You are a helpful AI assistant.", the
  knowledge cutoff 2026-01-04, and a current-date line when the engine
  defines `strftime_now`): **decided 2026-09-16** — the profile renders
  that default without the date line (it takes no clock), documented as a
  deviation and pinned with a fixture captured with `current_date` unset;
  the agent always sends a system message anyway.
  Assistant history: `reasoning_content` renders as
  `<|start|>assistant to=self<|message|>…<|eom|>` (kept everywhere the
  template keeps it — the template has no last-user gate), then the
  answer as `<|start|>assistant to=user<|message|>…<|eot|>`; the
  generation prompt is `<|start|>assistant`.
- **Effort.** Muse has low/medium/high/xhigh and no off. Add `high` to
  the shared `Effort` (Qwen's template also knows it: re-capture
  `qwen38-text.json` with a `high` case and pin its instruction line;
  Gemma treats it as on); Muse renders `off` as `low` (the model always
  opens its reasoning message) and the help text says so.
- **Decoder.** A new channel grammar in `profiles/stream.zig`: the
  generation prompt ends after `<|start|>assistant`, so the model's
  stream is a sequence of messages `HEADER<|message|>BODY(<|eom|>|<|eot|>)`
  where the first header arrives as ordinary text (` to=self`) and later
  ones follow a `<|start|>` control token. The header routes the body:
  `assistant to=self` → thinking, `assistant` / `assistant to=user` →
  answer, `assistant to=NAME` → a tool body handed to the profile's parser
  (AGNT-10). `<|eom|>` ends a message, `<|eot|>` ends the turn. This is
  a `StreamMarkers` variant selected by the profile, with the existing
  bracket grammar untouched; tests drive it with fake ids at every split
  point as the Gemma header test does.
- Stop tokens `<|eot|>`, `<|end_of_text|>`; sampling defaults from the
  card; `stream_markers` for the reasoning header.
- The catalogue entry `muse-glimmer-30b` exists since 2026-09-17 with
  `mmproj-kquant.gguf` and `dflash-kquant.gguf` under the `mtp` role
  (decided: the role names the draft source the speculative-decoding unit
  loads, whatever its mechanism) and a `null` profile: this unit fills the
  profile in (`profile = .muse_glimmer`, and `catalog.Entry.profile` may
  become non-optional again); `nuclis --help`; the acceptance
  record (`scripts/reference-baseline.py --family muse-glimmer`,
  `make baseline-muse-glimmer`, the table in bench.md).

**Acceptance.** Text fixtures match byte for byte across the four levels;
`make test-vocabulary` on the file; `nuclis agent --model muse-glimmer-30b`
answers with its thinking shown; the acceptance run recorded.

## AGNT-10 — Muse Glimmer ATEM tool calling: rendering, decoding, fixtures

**Facts.** Declarations go into the system turn as prose plus one JSON
line per tool (`{"name": …, "description": …, "parameters": …}` in the
reference's `tojson` style, preceded by a `// Tool metadata` line per
namespace — a name's part before the first `.`, empty description — and
followed by a fixed example); the recipients line lists `"self"`, each
namespace as `"NS.*"`, and `"user"`. A call is its own message:
`<|start|>assistant to=NAME<|message|><atem:function_calls>\n<atem:invoke name="NAME">\n<atem:parameter name="K">V</atem:parameter>\n…</atem:invoke>\n</atem:function_calls>`
ended by `<|eom|>` when another message follows, else `<|eot|>`;
scalars are written as is, `true`/`false`/`null`, lists and objects as
JSON, strings verbatim (spaces not stripped). A result is its own turn:
`<|start|>tool NAME<|message|><tool_output name="NAME">\nCONTENT\n</tool_output><|eot|>`,
the name resolved from the call id. The reference parses the body with a
schema-aware grammar (string-typed parameters verbatim, others as JSON)
and treats `<|eot|>` as the end of the call step.

**Design.** Rendering in the profile from the shared `ToolDefinition` and
history; `parseTool` for the ATEM body (the inverse of the renderer: a
value that parses as JSON keeps its type, anything else is a literal
string — Qwen's rule); the header decoder from MODL-13 routes
`assistant to=NAME` bodies to it, and `<|eot|>` after a call ends the
step (several calls arrive as `<|eom|>`-separated messages before it).
`scripts/profile-tools-fixtures.py --profile muse_glimmer` captures the
same case shapes as Gemma's (a schema-rich declaration, nested values,
loops ending on results, two steps), and the ATEM markup inside content
or arguments is rejected as structure smuggling.

**Acceptance.** Tool fixtures byte for byte; parser round trip and
malformed-body rejections; a decoder test with two calls in one turn;
`make check`; a live `--print --json` turn on the pulled file executes
`write_file` then `read_file` with the file on disk as asked.
