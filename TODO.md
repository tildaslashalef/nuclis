# TODO — active plan

This file is the queue: the units agreed and not yet merged, each with its
design, and a *Where we are* note. A unit is one session, one branch, and
one pull request, whose description is the record of the work; when it
merges, its section is deleted here, and when the last one goes, this file
is emptied back to this header. Requirements live in [docs/spec.md](docs/spec.md); the
engine map in [docs/architecture.md](docs/architecture.md); how to build,
test, and measure in [docs/development.md](docs/development.md).

Session protocol (also in [AGENTS.md](AGENTS.md)): read this file first. If
it lists work, summarize *Where we are* and ask the user how to continue. If
it is empty, ask what to work on and write the agreed plan here.

## Where we are

One unit, **prompt-end cache boundary**, on branch
`prompt-end-cache-boundary`; nothing implemented yet. It answers a
client's report (Seshat, `nuclis serve`, 0.6.0 at `5e69b80`): on Gemma 4
a second turn that sends back the first exactly as returned reports
`cached_tokens: 0` and is prefilled whole. `nuclis chat` on Gemma 4 has
the same miss.

| Unit | Branch | State |
| --- | --- | --- |
| Prompt-end cache boundary | `prompt-end-cache-boundary` | designed |

## Prompt-end cache boundary

Base: `5e69b80`

### The fault

`Completer.run` (`src/completer.zig`) continues the session when the
render starts with `seen` (prompt plus generated text minus the last
token), else restores `cache.longest(full)`. Both answer-end saves
(`src/agent/loop.zig` and `src/api/chat/model.zig` call
`Model.checkpoint` after an answer) are keyed on `seen`. Gemma 4's render
(`inference/src/profiles/gemma4.zig`, `renderWith`) never reproduces
`seen` on the next turn:

1. Reasoning off, 12B and 26B-A4B (`Variant.empty_thought_when_off`):
   the generation prompt ends `<|turn>model\n<|channel>thought\n<channel|>`,
   history renders `<|turn>model\n{content}<turn|>\n`.
2. Reasoning on, every Gemma 4: the reasoning gate (`gate = i > last_user`)
   drops an answer's thought once a user message follows it.

Measured by the reporter: E4B with reasoning off reuses 40 of 54 tokens; E4B
with reasoning on, and 12B with it off or on, reuse 0. Qwen3.8 and Muse
Glimmer keep reasoning on every past turn and are unaffected.

Separately, nothing in `serve` saves the system block, so an identical
request, or a new question under the same system prompt, prefills
everything (the agent primes through `Completer.prime`; the server never
does).

### Design

Two more save points inside `Completer.run`, both before a control token,
so the split encodes the same as the whole (`cache.zig` module doc):

- **System end**: `eng.prefix(messages[0..profiles.leadingSystemCount(messages)], definitions, effort)`
  (the profile pins it as a byte prefix of `render`). Saved when it lies
  inside the remainder (`seen.items.len < len < full.len`), the cache does
  not hold it (`cache.exact`), and it encodes to at least
  `min_boundary_tokens` = 64 tokens (a Qwen snapshot is 150 MB before its
  first row). In the agent, `prime` has already saved it, so this is a
  no-op there.
- **Prompt end**: the byte offset of the last occurrence of the profile's
  model-turn opener in `full` (`<|turn>model\n` for Gemma 4,
  `<|im_start|>assistant\n` for Qwen3.8, Muse's own), saved when it lies
  inside the remainder and the profile **rewrites its own turn** at this
  effort. A tool-result step never qualifies: its opener lies before `seen`.

Profile surface (`inference/src/profiles/root.zig`, `Profile`): two
additions dispatched like `reasoning`:
- `turnOpen(self) []const u8`: the model-turn opener text, a constant in
  each module (`turn_open`).
- `rewritesTurn(self, effort) bool`: whether the history render of a
  generated answer differs from what the generation consumed. `gemma4`
  true at every effort; `gemma4_e` `effort != .off`; `qwen38` and
  `muse_glimmer` false. Each module pins it with a test: render
  `[sys, user]`, append a plausible generated answer, render `[sys, user,
  assistant, user]`, and assert `startsWith` holds exactly when
  `rewritesTurn` is false.

`Completer.run` changes, after the remainder is chosen and before
`imagePrefill`:
- Compute the boundaries (system end, then prompt end) that qualify, as
  byte offsets into `full`. None when the remainder has image placeholders
  (`placeholderRuns`), a per-layer observer is set (as `prime`), or the
  memory tier is off (`cache.budget == 0`).
- For each boundary in order: encode `full[seen.len..boundary]`, feed it
  as `prime` does (`inference.engine.commitPrompt` with a drafter, else
  `eng.model.prefill`), `h.observe` each token, append the text to `seen`,
  then `snapshot` and `keep`. The rest of `full` goes to `complete` as
  today. `prompt_tokens` and `reused_tokens` stay as reported: reused is
  the position before the first segment, prompt is reused plus every
  segment's tokens. The output-budget check (`outputBudget`) runs on the
  whole remainder before the first segment is fed, so a refused request
  feeds nothing.
- Record `turn_start: ?cache.Boundary` for the step (the prompt-end
  boundary, when saved).

`Completer.checkpoint`, when `rewritesTurn` holds: no answer-end snapshot
(it can never be matched); the disk save (`save_turns`) writes the
memory entry at `turn_start` (`cache.exact(seen[0..turn_start.bytes])`)
and returns that boundary, so `/resume` restores the prompt end. When it
does not hold, unchanged.

`cache.zig` needs no change: `longest` already picks the furthest entry.

### Docs

- `docs/engine/session.md` § The agent's token cache: the two boundaries
  in *Boundaries*, the Gemma 4 evidence.
- `docs/guide/api.md` § Conversations and the cache: the system and
  prompt-end saves and Gemma 4 numbers; § `GET /v1/models`: `efforts`
  lists the profile's names, where `off` is the request's `none`.

### Gates

- `zig build test` (the profile tests above; `increment`/boundary
  selection as a pure function `promptBoundaries(seen_len, full, …)` with
  unit tests).
- `make verify-auto`; `make verify` once (prefill is split differently).
- Evidence with `./zig-out/bin/nuclis serve`: the report's two-turn
  script on `gemma-4-e4b-qat`, `gemma4-12b-qat-…-balanced`, and Qwen3.8,
  reasoning `none` and `low`: turn-2 `cached_tokens > 0`; an identical
  request twice and a second question under the same long system prompt
  report the system block cached; time to first token before/after on the
  12B persona chat. Temperature 0: turn 2's text equals a cold run's.
- `make shot` on Gemma 4 12B: two chat turns, the bar shows the restore.
- No `make agent-eval`: the loop's prompts and steps are unchanged.
