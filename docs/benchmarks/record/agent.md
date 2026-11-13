<!-- kind: record -->
# Agent concepts: the record

The dated record behind [the agent concepts](../../app/agent.md): the
measurements behind the web tools and the token cache.

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
