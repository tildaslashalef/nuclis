<!-- kind: record -->
# Agent concepts: the record

The dated record behind [the agent concepts](../../app/agent.md): the
measurements behind the web tools and the token cache.

### The context budget (2026-11-17)

**Method.** Qwen3.8-27B UD-Q4_K_M, Metal, kv f16, M4 Pro 48 GB; effort
`low`, `agent.thinking_budget` 2048 (the registry entry's), so a step
reserves 4,096 and the ceiling is 12,288 at 16K and 28,672 at 32K; no
drafter; `make agent-eval` with
seeds 1 and 2. *Before* is the build of `051faab` (the whole `max_tokens`
reserved, a result an eighth of the window, compaction until the render
fits); *after* is that revision with this change, dirty. The `replay` task
is three turns, then the five-page `data/requests.txt` prompt, each turn a
`--print` run continued through `--resume`; "replayed" sums `prompt_tokens`
over its steps whose session entry says `replayed` (the three resumes
prefill about 100 of them in every variant), at `--ctx-size 16384`.

| variant | seed | passed | prompt tokens | replayed | compactions | model s |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| before | 1 | yes | 22,531 | 13,971 | 3 turns dropped, 3 elisions | 339.8 |
| before | 2 | no: `ContextFull` (17,440 of 16,384) on the second page | 7,983 | 100 | 3 turns dropped | — |
| mark 0.75 | 1 | yes | 16,443 | 6,598 | 3 turns dropped, 1 elision | 290.4 |
| mark 0.75 | 2 | yes | 15,900 | 6,065 | 3 turns dropped, 1 elision | 422.7 |
| mark 0.5 | 1 | yes | 14,581 | 4,736 | 3 turns dropped, 1 elision (two results) | 272.5 |
| mark 0.5 | 2 | yes | 15,900 | 6,065 | 3 turns dropped, 1 elision | 424.6 |

Before, each later page elided one more result and replayed the turn
(4,523, 4,623, 4,725 tokens); after, one compaction serves the rest of the
turn. The mark 0.5 replayed no more than 0.75 on either seed and 1,862
fewer tokens on seed 1, so it is the default (`low_water_percent`).

**The task list at 16K** (fifteen tasks, two seeds, the after build at the
0.5 mark; its `replay` runs are the mark's above). The thirteen
single-turn tasks ran identically before and after: same steps, tokens,
and answers, since none of them comes near the ceiling.

| | before | after |
| --- | ---: | ---: |
| passed | 29/30 (14/15 tasks on both seeds) | 29/30 (14/15) |
| prompt tokens, mean per run | 2,673 | 2,305 |
| earlier turns dropped | 6 | 6 |
| elisions | 9 | 3 |
| results clipped to the budget | 19 | 22 |
| replayed prompt tokens, all runs | 41,012 | 17,261 |
| model seconds, mean per run | 70.3 | 80.5 |

The failures differ: before, `replay` seed 2 ended in `ContextFull`
(the whole `max_tokens` reserved); after, `pages` seed 1 read up to
`req=502` through clipped pages and answered 600, the page boundary it
assumed, not the line it saw. The pages are clipped either way: a
`requests.txt` line is about 93 tokens, so the budget keeps 22
of a page's 120 lines; at 16K the budget is 2,048 either way (a sixth of
12,288, an eighth of 16,384). The mean model seconds rise
on that one run's 3,203 generated tokens.

**The task list at 32K**, the same way. The thirteen single-turn tasks
again ran identically.

| | before | after |
| --- | ---: | ---: |
| passed | 30/30 | 30/30 |
| prompt tokens, mean per run | 3,499 | 3,804 |
| earlier turns dropped | 0 | 0 |
| elisions | 2 | 0 |
| results clipped to the budget | 20 | 20 |
| replayed prompt tokens, all runs | 17,077 | 11,460 |
| model seconds, mean per run | 98.3 | 111.1 |

`replay` no longer compacts at 32K (before: one elision per seed and
8,272 and 8,605 tokens replayed; after: none, 100 replayed by the
resumes). A page keeps 51 lines against 44 (a 4,778-token budget against
4,096), which is the rise in prompt tokens and model seconds. The 11,260
tokens after replayed in `pages` seed 1 are not compaction: the step
before reasoned 2,129 tokens and its render rewrote the turn.

**Memory at the 32K default** (`/usr/bin/time -l` over one `agent
--print` process, `--ctx-size 32768 --kv f16 --think off`; the weights
are mapped from their files into Metal buffers without a copy, so they
count in neither figure and add the files' sizes, 16.56 GB for the model
and 1.37 GB for its `mtp` drafter):

| | peak resident set | peak footprint |
| --- | ---: | ---: |
| alone, after the warm-up and a one-line prompt | 3.15 GB | 3.30 GB |
| alone, after a 26,281-token prompt on the primed prefix | 4.87 GB | 4.93 GB |
| with the drafter, after the warm-up | 4.36 GB | 4.54 GB |
| with the drafter, after the same prompt | 6.62 GB | 6.80 GB |

The largest case is 6.80 GB with 17.93 GB of mapped weights, under 25 GB
of the 48, so no catalogue entry needs its own `ctx_size`.

### Why a third tool (2026-10-16)

**Why a third tool.** A probe of ten questions (six coding, four chat;
2026-10-16) compared `/v1/context` with a search and a fetch of the top hit.
Context at 2,048 tokens held the answer every time in 4.3–9.2K characters
and 1.1–1.4 s; 4,096 doubled the sources without adding an answer. Snippets
answered simple facts but missed a before/after example, cut text mid-word,
and left a version question ambiguous; two of ten top-hit fetches met a bot
challenge whose page context still had. Search and fetch stay for finding
links and reading a page whole.

### The agent's token cache (2026-10-04)

**Evidence** (Qwen3.8-27B Q4_K_M, Metal, M4 Pro, 16K window, dirty build of
2026-10-04 on `9f45bf0`). Start-up warm-up of the 1,333-token prefix at
`low` (`make shot`, `.zig-cache/tui/c1-start`, `c8-warm-start`, `c6-low`):
16.3 s prefilled, 1.0 s from disk, 0.4 s from memory. `agent -p` wall time
with the weights in the page cache, three pairs: 17.4 s cold, 2.6 s with
the disk entry. A cancel then a prompt (`c4-after-cancel`): `replayed
cancel 41, 1369 restored`; low → xhigh → low then a prompt
(`c7-after-effort`): `replayed effort 17, 1427 restored`. A cold and a
warm `agent -p` at temperature 0 give the same text with the drafter and
without it. A primed entry on disk is 242–251 MB.

Switching back to Qwen3.8 after Gemma 4 12B restored the primed prefix from disk in 0.1 s (2026-10-04).

### The save inside a prompt and the rewind (2026-10-08)

**Evidence for the save inside a prompt and the rewind** (2026-10-08):
the generation gates' prompt-saves check, run then on Gemma 4 12B QAT,
the 26B-A4B, E4B, Qwen3.8, and Muse Glimmer (it runs today in
`qwen38-generation-metal` and `gemma4-{26b-a4b,e4b}-generation-metal`): a run that saves after 8 of 17 prompt tokens emits
the same 8 greedy tokens as one fed those 8 then the rest; the saved state
restored, and on the attention-only families the session moved back to it
after the run, emit them again; Qwen3.8 refuses the rewind. Through
`nuclis serve` and `nuclis agent` on Gemma 4 12B, E4B, and Qwen3.8 the
answers at temperature 0 equal the previous build's, and the reuse is the
table in [api.md § Conversations and the cache](api.md#conversations-and-the-cache-2026-10-08);
in `nuclis agent` on the 12B (speculation on), turns 2 and 3 prefilled 28
and 35 tokens where the previous build reported `replayed rewrite 44,
3183 restored` and `76, 3183 restored`.
