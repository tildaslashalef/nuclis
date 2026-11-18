<!-- kind: record -->
# Agent concepts: the record

The dated record behind [the agent concepts](../../app/agent.md): the
measurements behind the web tools and the token cache.

### How deep compaction cuts: low water 50, 30, 25 (2026-11-18)

**Question.** [§ Where an agent turn goes](#where-an-agent-turn-goes-2026-11-18)
derived that a lower `low_water_percent` admits more new tokens per
compaction (13,000 at 50 %, 18,200 at 30 %, 19,500 at 25 %) for the same
prefill from the system block up. Would it save wall time on the list,
and at what cost in passes?

**Method.** Three ReleaseSafe binaries of `4defc23` differing only in the
constant (50, 30, 25), the 26B-A4B (UD-Q4_K_XL) with its MTP drafter at
draft length 4, its registry effort and thinking budget, Metal, kv f16,
the 32K window, M4 Pro 48 GB. The eighteen tasks at seeds 1 to 3, then
the four that fill the window (`pages`, `replay`, `longread`, `longlog`)
at seeds 4 to 6. Wall and its split are `--breakdown`'s.

| variant | list: passed | wall s | prefill s | decode s | new tok | four tasks, seeds 1–6: passed | wall s | new tok | rewrites | dropped / elided |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 (control) | 45 / 54 | 2,928 | 1,798 | 1,087 | 300,735 | 21 / 24 | 3,831 | 481,964 | 7 | 9 / 7 |
| 30 | 45 / 54 | 2,924 | 1,795 | 1,086 | 300,735 | 21 / 24 | 3,774 | 477,583 | 7 | 9 / 7 |
| 25 | 45 / 54 | 2,922 | 1,794 | 1,085 | 300,735 | 21 / 24 | 3,773 | 477,583 | 7 | 9 / 7 |

The two-term fit does not move: 0.19 s + 2.49–2.52 ms/tok + 159–160 ns
on every variant.

**Reading.**
- The mark decides nothing on this list. 65 of 66 runs are token for
  token the same at all three marks; the control reproduces
  § Where an agent turn goes's drafter row (300,735 new tokens, 2,923 s
  there, 2,928 here). The list's wall differs by 0.1–0.2 %, inside the
  run-to-run spread, and every pass and failure is the same run.
- Compaction is rare and done once a run: 4 of 66 runs compact (`replay`
  seeds 1, 4, and 6, `pages` seed 2), each in one decision, and no run fills
  the window a second time, which is the only case a deeper cut pays
  for. The decision's own granularity sets the depth: whole earlier
  turns, then results down to the last two kept, so 50 % and 30 % land
  on the same cut in three of the four.
- The one exception, `replay` seed 6, cut one result deeper at 30 % and
  25 % and prefilled 4,381 fewer tokens, 57 s of its 340.
- Keep rule (no pass lost, at least 2 % less wall on the list): not met.
  50 stays. The `keep_results` 3 variant the design held in reserve
  applies only if 30 had won on time, so it did not run.

### Where an agent turn goes (2026-11-18)

**Method.** `make agent-eval`'s eighteen tasks (the thirteen of before,
`longread` and `manysteps` among the new), seeds 1 to 3, each variant's
own registry effort and thinking budget, Metal, kv f16, the 32K window,
M4 Pro 48 GB. The 26B-A4B (UD-Q4_K_XL) ran from a ReleaseSafe binary of
`d5b359f`, speculation off and on with its MTP drafter at draft length 4;
the E4B (QAT) from `29fb842`, with its drafter, as its entry has it;
Qwen3.8-27B (the re-made UD-Q4_K_M) from `d5b359f`, seed 1 only, off and
with each drafter at draft length 7. The
step fields are those [§ 12 of the concepts](../../app/agent.md#12-the-agents-token-cache)
lists; the table is `agent-eval.py --breakdown`, the fit a least-squares
line of each step's prefill seconds on its new prompt tokens.

| variant | runs passed | steps | wall s | prefill s (new / reused tok) | decode s (tok, tok/s) | accepted / drafted | prefill fit |
| --- | ---: | ---: | ---: | --- | --- | ---: | --- |
| E4B, drafter | 46 / 54 | 322 | 1,976 | 1,016 (242,830 / 1,458,176) | 942 (91,602, 97.3) | 63 % | −0.20 s + 4.45 ms/tok |
| 26B-A4B, off | 42 / 54 | 394 | 3,420 | 1,682 (306,802 / 1,925,291) | 1,705 (88,775, 52.1) | — | 0.10 s + 5.36 ms/tok |
| 26B-A4B, MTP 4 | 46 / 54 | 356 | 2,923 | 1,794 (300,735 / 1,691,832) | 1,086 (74,540, 68.6) | 74 % | 0.08 s + 5.87 ms/tok |
| Qwen3.8, off | 17 / 18 | 95 | 2,839 | 1,308 (71,709 / 402,555) | 1,525 (15,098, 9.9) | — | 0.28 s + 17.9 ms/tok |
| Qwen3.8, MTP 7 | 18 / 18 | 98 | 2,505 | 1,445 (80,237 / 442,271) | 1,052 (20,945, 19.9) | 76 % | 0.56 s + 17.3 ms/tok |
| Qwen3.8, DFlash2 7 | 18 / 18 | 91 | 2,113 | 1,296 (76,658 / 404,206) | 810 (17,988, 22.2) | 77 % | 0.49 s + 16.3 ms/tok |

Shares of wall time, prefill / decode / tools / idle: E4B 51 / 48 / 0 / 1;
26B-A4B off 49 / 50 / 0 / 1; with the drafter 61 / 37 / 0 / 1; Qwen3.8
off 46 / 54, MTP 58 / 42, DFlash2 61 / 38 (tools and idle under 1 %).
Prefill tokens per generated token: 2.65, 3.46, 4.03; Qwen 4.75, 3.83,
4.26. A first DFlash2 run, launched as MTP but taking the user's
configured drafter, gave the same tokens at 2,196 s against 2,113: about
4 % run to run on wall time.

**Prefill with depth.** The line above ignores what a step's new tokens
attend to. Fitted instead as `c + a·n + b·((d+n)² − d²)`, n the step's new
tokens and d the reused depth before them (`--breakdown`'s second fit),
every variant's prefill splits into a part per token, a part that grows
with depth, and a fixed part per step:

| variant | c, s | a, ms/tok | b, ns | R² | prefill s: per token + depth + fixed |
| --- | ---: | ---: | ---: | ---: | --- |
| E4B, drafter | 0.12 | 1.89 | 104 | 0.9997 | 459 + 518 + 39 |
| 26B-A4B, off | 0.21 | 2.20 | 149 | 0.9999 | 675 + 924 + 84 |
| 26B-A4B, MTP 4 | 0.19 | 2.50 | 160 | 0.9999 | 751 + 975 + 69 |
| Qwen3.8, off | 0.51 | 10.76 | 297 | 0.9991 | 772 + 488 + 49 |
| Qwen3.8, MTP 7 | 0.48 | 11.01 | 292 | 0.9996 | 883 + 516 + 47 |
| Qwen3.8, DFlash2 7 | 0.38 | 10.55 | 274 | 1.0000 | 809 + 453 + 34 |

The 26B-A4B's b matches the cold prompt's 164 ns
([gemma4 record § Context past 32K](gemma4.md#context-past-32k-2026-11-18)):
the global layers' attention, a third of the list's wall time with the
drafter on (975 of 2,923 s).

**Reuse.** Of the 26B-A4B's 394 steps without the drafter, 54 were a
turn's first, 231 reused the whole saved state, 95 rewound to the mark
(Gemma's template renders a step's own output differently on the next
one), 9 resumed a stored conversation, and 5 were rewrites: the context
budget elided or dropped a result and the completer restored only the
system prefix. The rewinds lost 16,382 tokens, about 170 a step; the five
rewrites lost 104,863 and prefilled 46,803 again (elision had shortened
the prompt), 204 of the 1,682 prefill seconds (12 %). The MTP variant: 92
rewinds lost 22,709 tokens, 5 rewrites lost 106,227 and took 228 s (13 %).
The E4B: 66 rewinds for 18,226, its rewrites 4,419.

**Reading.**
- On the 26B-A4B the drafter takes decode from 52.1 to 68.6 tokens/s on
  what the agent emits (1.32×, 74 % of drafts accepted; the prose and code
  benchmark's 1.18-1.65×), and the list's wall time from 3,420 s to 2,923
  (−15 %; model seconds per run 60.6 to 51.5). Prefill is then 61 % of
  the time: the next lever is prefill, and inside it the global layers'
  attention at depth (54 % of prefill seconds); the rewrites are a third
  of the lost tokens and 12–13 % of the seconds.
- The fixed cost per step is small on either Gemma: 0.12–0.21 s with the
  depth term, 0.1 s at most on the line; Qwen3.8's is 0.38–0.51 s.
  Prefill runs at 186 tokens/s (26B-A4B) and 225 (E4B) of new tokens over
  a mean reused depth of 4,900 and 4,500.
- Gemma's rewinds are frequent and cheap; the rewrites are rare and are
  most of the lost reuse, in `pages` and `replay`, the two tasks that fill
  the window. Little of a rewrite can be kept: elision stubs the turn's
  oldest result first, and dropping a turn cuts right after the system
  block, so the first changed token sits at about 2,000 (`pages` seed 2:
  the system block is 1,892 tokens, the first result begins near 2,039)
  and everything after it moves. Each time the window fills, the prompt
  is prefilled again from the system block up to the ceiling, whatever
  `low_water_percent` (50) is; what the cut decides is how many new tokens
  a cycle admits before the next one, about 13,000 at 50 % and 18,200 at
  30 % (derived, not measured), against what the model loses with the
  results it no longer sees.
- On Qwen3.8 the drafters double decode (9.9 to 19.9 and 22.2 tokens/s)
  but cut model seconds per run only 11 % (MTP, 150.6 to 133.8) and 28 %
  (DFlash2, to 107.8): prefill, at 17 ms a token, is then 58–61 % of the
  time. Per task it takes 108 s with DFlash2 against the 26B-A4B's 51 s.
- Passes differ by a few runs between variants of the same model; the
  drafter changes no greedy output, and these runs sample, so the
  difference is the seeds', not the drafter's.

### An edit's echo cut to its hunk headers (2026-11-17)

**Method.** `make agent-eval`'s six edit tasks (`rename`, `heading`,
`bugfix`, `validate`, `newfile`, `json`), seeds 1 to 5, Qwen3.8-27B
UD-Q4_K_M with its drafter, Metal, kv f16, M4 Pro 48 GB, the 32K default
window, effort `low`, thinking budget 2048. *Full* is the build of
`4d3cafe`, whose `edit_file` result carries the unified diff with one line
of context; *headers* is that revision with the result cut to its first
line and the `@@` hunk headers (the transcript still drew the whole diff),
and the guideline's "the result shows the diff" changed to "the result
names the lines changed". The two builds alternated seed by seed. Result
tokens are `scripts/app/agent-tokens.py`'s count over all 30 runs of each.

| variant | passes | failed edits | `reread` | `edit_file` calls | `edit_file` result tokens | all result tokens | model s, median / sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| full | 29 / 30 | 0 | 0 | 58 | 5,971 | 23,220 | 54.5 / 2,105 |
| headers | 29 / 30 | 0 | 2 | 63 | 1,572 | 23,636 | 59.0 / 1,961 |

The echo was a quarter of what it was, but the turn's result tokens did
not fall: the saving went into more reads, greps, and edits. Both rereads
are one run, `json` seed 2, and its reasoning gives the cause: after an
edit whose result was `@@ -12,2 +12,5 @@` and `@@ -19,7 +22,12 @@`, "I
need to check what I actually wrote", then a read of the file, which
showed two lines its edit had left behind, then two more edits to repair
it and a second read to confirm. With the full echo the first of those
lines is the context row under the change, and no run of the full build re-read a file
it had just edited. The one failure in each arm is `validate` (seed 1 for
headers, seed 4 for full), both the same miss: a test that asserts the
`ValueError` without its message. The sum of model seconds is 7 % lower
for headers, inside the spread of two outlier runs of the full build
(`json` seed 4 at 291 s, `validate` seed 5 at 238 s); the medians go the
other way.

**Decision.** The kill criterion was a rise in `reread`, failed edits, or
lost passes; `reread` rose, and the result tokens the change was for did
not come back as a smaller turn. The full echo stays: the diff's `-` and
`+` lines repeat the call, but its context lines are how the model sees
where its edit landed without reading the file again.

### The thinking budget (2026-11-17)

**Method.** `make agent-eval`'s fifteen tasks at the build of `3ddc0c5`
with the runner's reasoning counts added, M4 Pro 48 GB, Metal, kv f16,
the 32K default window, `max_tokens` 8192, each model with its registry
entry's generation settings (Qwen3.8-27B UD-Q4_K_M with its drafter,
Gemma 4 26B-A4B UD-Q4_K_XL without). Every variant passes
`-- --thinking-budget N --think E`, and the flag wins over the entry's
`agent.thinking_budget` (`config.resolve`: the flag, else the entry,
else the file). The recorded steps confirm it: every cut step reasons
exactly its variant's cap plus one token (1,025, 2,049, 4,097), so no
1,024 or 4,096 variant ran at the entries' 2,048. Reasoning tokens are
per step (`reasoning_tokens` in the session), as nearest-rank
percentiles over every step of a variant; "cut" counts steps whose
`reasoning_cut` is set.

**What ran.** The whole matrix (four variants, two models, three seeds)
was estimated beforehand at about 10.5 hours from the 32K record's ~30
minutes per Qwen sweep and a first Gemma sweep of ~20 minutes, so it ran
two seeds of every variant and the third only for 1024 and 2048 at
`low`, the two the default is chosen between: 284 runs, 6.8 hours of
wall time. Gemma's `medium` ran one seed: its template only switches
reasoning on, so `medium` at 2048 renders as `low` with a 4,096 cap, and
its seed-1 sweep reproduced 4096 at `low` token for token (the same
reasoning per step and the same answers on all fourteen tasks it ran
before its `replay`, which was then dropped). Each command was one
variant, seed, and task group (the thirteen single-turn tasks, `pages`,
`replay`); eight single runs, all `pages` or `replay`, took 10 to 12.5
minutes and could not be split further. One `pages` run was killed by
the runner's guard and re-run whole.

**Per variant**, every seed pooled; model seconds are a mean per run.

| model | budget | effort | runs | passed | cut steps | reasoning p50 / p90 / max | steps over 1,024 | model s | prompt tokens, mean | dropped / elided |
| --- | ---: | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | --- |
| Qwen3.8-27B | 1024 | low | 45 | 45 | 1 | 33 / 143 / 1,025 | 1 of 212 | 102 | 3,778 | 0 / 0 |
| Qwen3.8-27B | 2048 | low | 45 | 45 | 1 | 34 / 151 / 2,049 | 2 of 213 | 108 | 3,898 | 3 / 1 |
| Qwen3.8-27B | 4096 | low | 30 | 30 | 0 | 32 / 143 / 1,410 | 1 of 148 | 100 | 3,718 | 6 / 2 |
| Qwen3.8-27B | 2048 | medium | 30 | 30 | 0 | 32 / 161 / 1,004 | 0 of 138 | 76 | 2,685 | 6 / 2 |
| Gemma 4 26B-A4B | 1024 | low | 45 | 34 | 10 | 45 / 396 / 1,025 | 10 of 276 | 58 | 5,901 | 6 / 4 |
| Gemma 4 26B-A4B | 2048 | low | 45 | 38 | 3 | 43 / 421 / 2,049 | 12 of 266 | 54 | 5,103 | 3 / 3 |
| Gemma 4 26B-A4B | 4096 | low | 30 | 25 | 1 | 48 / 540 / 4,097 | 10 of 187 | 77 | 7,963 | 6 / 6 |

**Per seed.**

| model | budget | effort | seed | passed | steps | cut | reasoning p50 / p90 / max | model s | prompt tokens, mean | dropped / elided / replayed |
| --- | ---: | --- | ---: | --- | ---: | ---: | --- | ---: | ---: | --- |
| Qwen3.8-27B | 1024 | low | 1 | 15/15 | 70 | 1 | 36 / 143 / 1,025 | 1,624 | 3,881 | 0 / 0 / 3,066 |
| Qwen3.8-27B | 1024 | low | 2 | 15/15 | 77 | 0 | 33 / 134 / 376 | 1,525 | 3,672 | 0 / 0 / 100 |
| Qwen3.8-27B | 1024 | low | 3 | 15/15 | 65 | 0 | 29 / 146 / 969 | 1,451 | 3,782 | 0 / 0 / 100 |
| Qwen3.8-27B | 2048 | low | 1 | 15/15 | 71 | 1 | 41 / 151 / 2,049 | 1,838 | 4,048 | 0 / 0 / 11,360 |
| Qwen3.8-27B | 2048 | low | 2 | 15/15 | 77 | 0 | 30 / 134 / 482 | 1,516 | 3,561 | 0 / 0 / 100 |
| Qwen3.8-27B | 2048 | low | 3 | 15/15 | 65 | 0 | 33 / 237 / 969 | 1,503 | 4,087 | 3 / 1 / 10,611 |
| Qwen3.8-27B | 4096 | low | 1 | 15/15 | 71 | 0 | 34 / 143 / 1,410 | 1,534 | 3,718 | 3 / 1 / 8,825 |
| Qwen3.8-27B | 4096 | low | 2 | 15/15 | 77 | 0 | 31 / 134 / 376 | 1,480 | 3,718 | 3 / 1 / 9,189 |
| Qwen3.8-27B | 2048 | medium | 1 | 15/15 | 68 | 0 | 37 / 166 / 849 | 1,172 | 2,590 | 3 / 1 / 9,089 |
| Qwen3.8-27B | 2048 | medium | 2 | 15/15 | 70 | 0 | 28 / 118 / 1,004 | 1,101 | 2,780 | 3 / 1 / 8,777 |
| Gemma 4 26B-A4B | 1024 | low | 1 | 12/15 (not count, history, validate) | 98 | 4 | 45 / 510 / 1,025 | 1,159 | 8,602 | 3 / 3 / 32,062 |
| Gemma 4 26B-A4B | 1024 | low | 2 | 10/15 (not count, newfile, pages, replay, validate) | 86 | 3 | 43 / 421 / 1,025 | 826 | 5,251 | 3 / 1 / 11,911 |
| Gemma 4 26B-A4B | 1024 | low | 3 | 12/15 (not history, replay, validate) | 92 | 3 | 53 / 282 / 1,025 | 645 | 3,850 | 0 / 0 / 206 |
| Gemma 4 26B-A4B | 2048 | low | 1 | 12/15 (not count, history, validate) | 80 | 1 | 40 / 522 / 2,049 | 726 | 4,338 | 0 / 0 / 251 |
| Gemma 4 26B-A4B | 2048 | low | 2 | 12/15 (not count, newfile, pages) | 90 | 0 | 40 / 388 / 1,763 | 872 | 5,857 | 3 / 2 / 17,363 |
| Gemma 4 26B-A4B | 2048 | low | 3 | 14/15 (not history) | 96 | 2 | 53 / 375 / 2,049 | 842 | 5,114 | 0 / 1 / 10,432 |
| Gemma 4 26B-A4B | 4096 | low | 1 | 12/15 (not count, history, validate) | 89 | 1 | 45 / 766 / 4,097 | 1,038 | 6,821 | 3 / 2 / 22,709 |
| Gemma 4 26B-A4B | 4096 | low | 2 | 13/15 (not count, newfile) | 98 | 0 | 50 / 471 / 1,763 | 1,286 | 9,105 | 3 / 4 / 47,947 |

**The default: 2048.** Qwen3.8 passes every run at every budget and is
cut once in 212 steps at 1,024: its ordinary step reasons for a p90 of
about 150 tokens, and the budget barely binds. Gemma 4 is where it
binds: at 1,024 ten steps were cut and three failures follow a cut
(`validate` seeds 2 and 3, `pages` seed 2), and `validate` failed on all
three seeds. At 2,048 the three cuts are all on `json`, which passed
every time, and none of its seven failures has a cut: `count`,
`history`, `newfile` seed 2, and `validate` seed 1 fail at every
budget, and `pages` seed 2 stopped after three reads without a cut (it
passed at 4,096). 4,096 cut one step, a 4,097-token runaway on `json`
seed 1 that every budget cuts, and paid for its larger reservation in
compaction: three turns dropped in `replay` on every seed of both models
(and at Qwen's `medium`, the same 6,144 reserved), against one seed of
three per model at 2,048; Gemma's replayed tokens rise from 28,046 over
three seeds to 70,656 over two. The model seconds do not separate the
`low` budgets beyond the seeds' spread on Qwen; Gemma at 4,096 took 77 s
a run against 54 to 58. Qwen at `medium` ran faster (76 s) on fewer
prompt tokens, its steps no longer. So the default and every catalogue
entry `config init` writes are 2048, what the registry entries already
set by hand.

**The cap: `max_tokens` less `answer_reserve`.** Under the old rule (half
of `max_tokens`) a 2,048 base gave 2,048 at `low` and 4,096 at `medium`,
`high`, and `xhigh` alike, so the top three efforts stopped differing.
The runs say a step needs no more than 4,096 at `medium`: Qwen's longest
`medium` step reasoned 1,004 tokens, and Gemma's only step past 2,048 was
the runaway. So nothing argues for raising `max_tokens`: that would let
`high` and `xhigh` reserve more than 8,192 of the window, and the 4,096
rows show what a larger reservation costs. The cap is `max_tokens` less
the 2,048 the step reservation already holds for the answer (half of a
`max_tokens` below 4,096): 2,048, 4,096, 6,144, and 6,144 at the
defaults, so `high` again exceeds `medium` and the answer keeps the same
2,048 at every effort. `xhigh` exceeds `high` only with `max_tokens` of
10,240 or more, which a user may set; neither was measured here, since
no task reached the 4,096 cap except the runaway.

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

### Web pages sized from the result budget (2026-11-17)

**Method.** Qwen3.8-27B UD-Q4_K_M, Metal, kv f16, M4 Pro 48 GB, effort
`low`, thinking budget 2048, so the result budget is 2,048 tokens at 16K
and 4,778 at 32K; seeds 1 and 2; the live search service. Three web
turns through `make agent-eval`'s runner, not in its task list: `wpages`
(PEP 8 by `web_fetch` with no question, three pages one after another,
then the last heading read; 51,104 characters), `wfocus` (a question to
the `functools` reference), and `wcontext` (PEP 8's line length by
`web_context`). *Before* is the build of `6cf6ea1`, which asked for 16,000
characters; *after* is that revision with this change, dirty, asking for
`pageChars` of the budget (5,632 and 13,139). A cut is the loop's own
(`[truncated to fit the context:`); an offset is followed when it is the
one the previous result stated.

| window | variant | web results cut | offsets followed / guessed | `wpages` prompt tokens (s1, s2) | `wcontext` | `wfocus` | `wpages` model s |
| --- | --- | ---: | --- | --- | ---: | ---: | --- |
| 16K | before | 8 of 10 | 0 / 4 | 5,604, 5,827 | 1,860, 1,898 | 2,092 | 125.8, 171.7 |
| 16K | after | 0 of 10 | 4 / 0 | 4,414, 4,414 | 1,271, 1,188 | 1,570 | 100.2, 83.3 |
| 32K | before | 0 of 10 | 4 / 0 | 12,316, 12,316 | 2,322, 2,308 | 2,092 | 218.1, 231.3 |
| 32K | after | 0 of 10 | 4 / 0 | 10,139, 10,139 | 2,322, 2,308 | 2,092 | 188.7, 182.6 |

Before, at 16K, every page and every context was cut to the budget with
"narrow the request for the rest", and the model guessed the next offset
on both seeds (16,000 then 32,000 on seed 1, past text it never saw; 241
then 245 on seed 2). After, no result was cut and every continuation was
the stated one. At 32K the pages already fit; the smaller page reads less
of PEP 8 in three steps for 18 % fewer prompt tokens. `wfocus` failed
seed 2 before at both windows, its answer omitting the `TypeError`; it
passed every run after.

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
