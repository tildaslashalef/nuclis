<!-- kind: reference -->
# Agent concepts

The concepts behind `nuclis agent`'s loop: how a turn runs its steps, how a
tool runs a process, how a session resumes, the web tools, and the token
cache. The design is in [spec.md § The agent](../spec.md#7-the-agent); the
inference stack is taught in the companion [llm-guide.md](../llm-guide.md).
The concepts are numbered once across three documents: §1, §4, and §7, the
terminal's, are in [terminal.md](terminal.md); §2, §3, §8, and §9, how a tool
call crosses the token boundary, are in
[agent-tool-calls.md](agent-tool-calls.md).

## 5. A turn is a loop: steps, the two seams, and a parser that must hold back

Chat is a single completion: render the conversation, stream an answer, stop.
An *agent* turn is a loop. The model may answer with a request to run a tool
instead of (or before) an answer, and the host has to execute it and hand the
result back before the model can continue. `src/agent/loop.zig` is that loop:
render the history, complete one step, parse the response for calls, execute
every call in order, append the results, and repeat — until a response carries
no calls, a step budget is spent (16), or the step is cancelled. A step that
was cancelled is *not* appended to the history: the model never saw its end,
so it is display-only.

The loop is deliberately the one piece the terminal surface and `--print`
share, and neither should be able to make it behave differently. So it is
written against two small seams rather than against an engine and a terminal.
`Model.run(messages)` is the completion side: the real `Completer` (`src/completer.zig`) renders
through `inference.engine` and, crucially, keeps the bookkeeping that lets a
growing conversation prefill only its *new* suffix (`seen`), replaying from an
empty session when the render no longer starts with what the session consumed.
`Events` is the presentation side: `send` for terminal events and `record` for
session entries. A test implements both with a scripted answer and a recorder,
which is why the loop's budget, cancellation, and tool-execution paths are
pinned without a model or a GPU.

The parse lives in the prompt profile. Native tool decoding belongs to the
same place that owns the tokenizer's special-token boundaries ([§2](agent-tool-calls.md#2-why-a-streaming-parser-needs-token-boundaries)): the
shared `profiles/stream.zig` decoder holds the call's bracket token ids and a
body parser. Streaming shapes the problem: the decoder collects
the body between the two control tokens and never executes a call that is
still open when the stream ends: at EOS a body that parses is still one call,
and anything else, after a budget stop or a cancellation, is dropped with a
notice. Released as answer text, its markers would read as the model's prose
and come back as control tokens the next turn. It cannot emit `<tool_call>` as
ordinary text and then discover the call, because the opening token is
recognized by id, which is the only signal a byte-stream consumer cannot see.
That is the parser analog of [§2](agent-tool-calls.md#2-why-a-streaming-parser-needs-token-boundaries)'s rule: a byte's meaning depends on what came
after it, and the only safe move is to delay the decision. [§8](agent-tool-calls.md#8-decoding-a-tool-call-at-the-token-boundary) follows the
decoder in detail.

The loop carries **native** tool history: an assistant message with its
`tool_calls` and one `.tool` result message per call, which the profile renders
in the artifact's own syntax, so the wire format lives in the profile and
nowhere else.

## 6. Running a command is a lifetime problem

A read tool borrows bytes; a shell tool creates a *process*. That process has
a lifetime, an output stream, and an environment, and all three are the host's
responsibility the moment a model can ask for one.

The bounds are not decoration. Output must stop at a byte cap, because a
command can print forever and the transcript would grow without limit; wall
time must stop at a deadline, because a command can block forever. Both are
enforced by *not reading* once the cap is hit or the deadline passes, and then
killing the child. Cancellation is the third bound and the subtle one: the
engine notices Ctrl-C between layers of the model, but a running child is not
in that loop, so the tool has to poll the same `interrupt` flag between reads
and kill the child itself. The three paths share one cleanup. The command runs
in a process group of its own, so the stop reaches what it started in the
background too: the group gets `SIGTERM`, then `SIGKILL` if anything is still
alive after a two-second grace, and the child is reaped, so cancel, timeout,
and overflow all leave no orphan. A cancel that arrives between the calls of
one step answers the calls it has not run with a typed `not run` result: every
call keeps its result, so the history and the session stay valid.

Reading output is a second lifetime hazard. stdout and stderr are separate
pipes, and reading one to end before touching the other deadlocks as soon as
the child fills the pipe you are ignoring — the classic reason a
`MultiReader` waits on both at once. But waiting on both loses the *order*
between them, which a build log needs (an error printed between two progress
lines should stay between them). The fix is to fold stderr into stdout at the
shell level, `exec 2>&1`, so there is one ordered stream by construction,
while the empty second pipe is still drained so no writer can block.

The environment is the quiet security boundary. A tool that inherits the
parent's environment hands the model whatever secrets are in it; a command as
innocent as `env` would print them into the conversation. nuclis forwards a
short allowlist — `PATH`, `HOME`, `LANG`, `TERM`, `TMPDIR` — and drops the
rest, so the child is useful and the model still cannot read the host's
tokens. `expand_arg0` is off for the same reason: `/bin/sh` is a path, not a
name resolved through anything the child can influence.

Finally, the split between `grep` and `bash` is deliberate. `grep` is literal,
native, and bounded: one predictable contract that the tool layer can test
without a shell, and it reads each directory's `.gitignore` itself, so the
same tree gives the same result on every machine. `bash` is the escape hatch
for everything the fixed tools do not cover — regex, pipelines — and it pays
for that power with the process lifetime above.

What a tool returns is prefill the next step pays for: on the playground
task list four fifths of a turn's prefilled tokens were tool results. So
every result is shaped for the model's reading, not for completeness: a
page rather than a file, a file's outline before its body, matches grouped
under their path, a diff with one line of context, a long command output's
head and tail, and a pointer instead of a range already in the context.
Whatever is left out is stated with how to get it.

One more turn of the same screw. When the terminal is in raw mode, Ctrl-C is a
*key*, not a signal — ISIG is off — so the interrupt the engine's observer
checks is never set while a blocking tool runs. The tool cannot poll a flag
that nothing raises, and the driver cannot read the keyboard while the
tool holds the loop. The seam that closes the gap is a beat: an optional
`Tick` the agent installs on the workspace, invoked by `bash` once per poll
iteration, on which the driver reads the keyboard, repaints, and sets the same
`interrupt` flag the tool is already watching. A tool that never blocks pays
nothing, and print mode installs no tick at all.

The bounds are host constants, never the model's; the tools' rows
(`agent`, `read_file`, `grep`, `glob`) are read from the code:

<!-- gen:limits -->
| Limit | Value | Declared in |
| --- | --- | --- |
| context ceiling (tokens) | 32,768 | inference/src/runtime/session.zig `max_capacity` |
| output budget (tokens) | 16,384 | src/config.zig `max_output_tokens` |
| draft length | 15 | inference/src/engine.zig `max_draft_length` |
| prefill chunk (tokens) | 256 | inference/src/engine.zig `prefill_chunk` |
| config file (bytes) | 65,536 | src/config.zig `max_file_bytes` |
| agent steps per turn | 16 | src/agent/loop.zig `budget_default` |
| agent turns per session | 256 | src/agent/loop.zig `max_turns` |
| agent instructions file (bytes) | 8,192 | src/agent/system_prompt.zig `max_instructions_bytes` |
| read_file lines per call | 2,000 | src/agent/tools/read_file.zig `max_lines` |
| read_file default lines | 120 | src/agent/tools/read_file.zig `default_lines` |
| read_file bytes | 1,048,576 | src/agent/tools/read_file.zig `max_bytes` |
| grep matches shown | 50 | src/agent/tools/grep.zig `max_shown` |
| grep files scanned | 20,000 | src/agent/tools/grep.zig `max_files` |
| grep seconds | 10 | src/agent/tools/grep.zig `max_seconds` |
| glob results | 200 | src/agent/tools/glob.zig `max_results` |
| image tokens per image | 4,096 | inference/src/vision/root.zig `max_image_tokens` |
| server output tokens | 16,384 | src/api/chat/wire.zig `max_output_tokens` |
| server audio clip (bytes) | 16,777,216 | src/api/chat/wire.zig `max_audio_bytes` |
| server web rounds | 6 | src/api/chat/service.zig `max_web_rounds` |
| server speech input (bytes) | 16,384 | src/api/speech/wire.zig `max_input_bytes` |
| server live transcriptions | 16 | src/api/transcription/stream.zig `max_sessions` |
| decision questions per call | 32 | inference/src/decide.zig `max_questions` |
| decision states per call | 64 | inference/src/decide.zig `max_states` |
<!-- /gen -->

## 10. Resuming a session is a replay, not a restore

The intuition that saving a conversation means saving the model's state is the
one to resist. A `Session` is a page-aligned block of attention rows and, for
this hybrid, recurrent vectors; it cannot survive a process exit, and it cannot
be rewound by truncating a cache ([§22 of the inference companion](../llm-guide.md#22-checkpoints-not-rewinds)). So `~/.nuclis/agent/sessions/*.jsonl`
stores the other thing: the **conversation** — what the user typed, what the
model reasoned and answered, which call it made, what the tool returned. That
is exactly the input the render function needs, and nothing else.

`/resume` and `--resume <id>` therefore **replay**: the loader parses the
entries back into `profiles.Message`s (user, assistant-with-`tool_calls`, tool
result with its `tool_call_id`), `Agent.restore` appends them to a fresh loop
and continues the host correlation ids past the highest restored one, and the
next turn renders the whole thing through the profile and prefills it. The
transcript is replayed from the same entries, so what the terminal shows after
a resume is what it showed before. The fresh session file is seeded with the
entries so it can itself be resumed.

Two details make it work. The host ids stored on assistant calls are what let a
result match its call on load — the model never echoes them, so without them the
conversation would not validate. And a **cancelled step is not a message** the
model ever finished, so it is recorded for display but neither appended to
history nor given stored calls; a file whose assistant call has no following
result would not load. The live loop also shrinks its view to fit
the window, and each cut is a `compaction` entry the loader applies where it
was written: `context_full` drops the turns before `first_kept`,
`results_elided` turns the tool results of the turn in progress before
`first_kept` into the same one-line stubs, and `turn_aborted` drops a turn that
failed. So a resumed conversation is the one the model last saw, not the
longer one the file holds. The cost of the replay is that it is prefilled
again on resume — which is the honest trade: the alternative is persisting
opaque numerical state that the format cannot describe and a later change
cannot safely reinterpret.

This is also why the resumable unit is the *entry*, not the event. A turn emits
hundreds of `answer_delta` events and writes one `assistant` entry; the file
records the turn, so a resume re-renders rather than replays keystrokes.

## 11. The web tools: a host's search service, opt-in

`web_search`, `web_fetch`, and `web_context` are not the agent's own: they
call a search service the user runs on another host (for example one in
front of the Brave Search API with a self-hosted SearXNG that answers when
Brave cannot). nuclis knows only the service's base URL and the v1 contract
below, and offers the tools only when the configuration says so.

**Opt-in, per surface.** `web.url` names the service; `agent.web` offers the
tools to the agent, `serve.web` lets a chat request ask the server to run
them ([configuration](../guide/configuration.md#web)). All three default to
off, and with them off the agent's tool list and system prompt are the same
bytes as without the feature. The tools come in tiers, `tools.Web`: `off`
is the six tools, `search` adds `web_search` and `web_fetch`, `context`
adds `web_context` too. The guideline loop renders only offered tools, and
`Agent.init` derives the prompt and the definitions from
`Workspace.webTier()` so they cannot disagree. A web call the model makes
for a tool its tier lacks is "not a tool I have", which also covers a
resumed session recorded with more tools on.

**The probe.** `web_context` exists only on a service with Brave, so the
agent's setup asks `/v1/health` once (2 s at most) before the prompt is
built and offers it when `endpoints` lists `/v1/context`; a slow or absent
service starts at the `search` tier. The tier is fixed for the session, so
the primed prompt prefix stays valid. `nuclis serve` probes too, for its
start line only: a server runs for days and Brave may come and go, so it
accepts the hosted type either way and a call to a service without it
answers `not_available` as a result.

**Why a separate host.** The fetcher reads URLs the model chooses, and a page
it reads can tell the model which URL to choose next. On the Mac that fetcher
would sit beside the workspace, the credentials, and the LAN. On a disposable
guest it sits behind its own firewall and an address guard that refuses any
non-public address at every redirect, and the hostile HTML is parsed there;
nuclis receives bounded Markdown. The same separation lets the search backend
change (a paid API, an offline index) without touching nuclis.

**The contract nuclis relies on (v1).** `GET`, JSON, no
authentication; nuclis sends `User-Agent: nuclis/<version>` and ignores fields
it does not know, so additions never break it.

| Call | Answer |
| --- | --- |
| `/v1/search?q=&n=[&freshness=]` | `{query, altered_query, provider, results: [{title, url, snippet, extra_snippets, engine, published}], engines_failed}`; `q` 1–400 bytes, `n` 1–10; `altered_query` is the corrected query the provider searched, else `null`; up to 5 `extra_snippets` (`[]` from SearXNG); no results while engines failed is `no_engines` |
| `/v1/context?q=&tokens=&urls=[&freshness=]` | `{query, provider, sources: [{url, title, published, snippets}], engines_failed}`: passages Brave picked from several pages for one query, `tokens` 1,024–16,384, `urls` 1–20 (not always honoured: 8 asked returned 8–24); zero sources is a `200`; 20 s; `501 not_available` on a host without Brave |
| `/v1/fetch?url=&start=&max_chars=` | `{url, final_url, status, content_type, title, text, start, chars, truncated}`; `http`/`https`, ports 80 and 443, at most 5 redirects and 5 MiB; `text` is Markdown with absolute links; `start` and `chars` count code points, and a `start` at or past `chars` returns empty text |
| `/v1/health` | always `200`, `{version, providers: {<name>: ok \| down}, endpoints, billed, quota, cache}`; `endpoints` lists `/v1/context` when Brave is on |

`freshness` is `pd`, `pw`, `pm`, `py`, or `YYYY-MM-DDtoYYYY-MM-DD`; nuclis
offers only the first four (`day` … `year`), since a range needs today's
date. Search and context answers carry `X-Search-Cache: hit | miss |
stale`; nuclis never sends `Cache-Control: no-cache`.

Errors are non-2xx with `{"error": {"code", "message"}}`: `bad_request` (also
a query Brave refuses), `not_found`, `method_not_allowed`, `blocked_address`,
`unsupported_content`, `upstream_failed` (also a page behind a bot challenge,
its message saying so), `no_engines`, `busy`, `upstream_timeout`,
`not_available`, `internal`. A code nuclis does not know is still a failure
carrying the server's message.

**The allowance.** Brave is metered: about 1,000 free requests a month, then
US$5 per 1,000, capped at US$10 a month in its dashboard. Every uncached
search or context is one; fetches and cache hits are free. So a `Service`
answers at most `max_billed_calls` (6) uncached calls between `beginTurn`s,
which the agent's loop calls at each turn and the API's job at each request;
the seventh answers `budget` without a request, and the model reads that
the turn's budget is spent and to answer from what it has. The count reads
`X-Search-Cache` (an answer without it counts, so the stub and older
services do). One turn at a time uses a service, so it needs no lock.

**What the model reads.** `src/web/tools.zig` holds the three definitions and
their rendering, shared by the agent and the server. A search is a numbered
list (`N. title — url`, the snippet under it, and up to two extra snippets
under the first three results) at most 8 long, led by the corrected query
when the provider changed it and followed by a note when engines failed. A
context is the same numbering per source, its passages indented beneath,
asked at a 2,048-token budget from 8 pages and cut at a source boundary past
12,000 characters with the number not shown. A fetch is the page's URL and
title, its text 16,000 code points at a time, and a bracketed note in
`read_file`'s form: the range, the length, and the `offset` that continues
it. Every failure is one sentence naming its cause (`busy` reads "try
again", a bot challenge names the host and, when offered, points to
`web_context`, whose provider has the page's text from its index); none ends
the turn. Queries are checked (≤ 400 bytes, a known `freshness`) and URLs too
(`http`/`https`, ≤ 2,048 bytes) before anything leaves the machine.

**Why a third tool.** Context passages answer what snippets and a top-hit
fetch miss; search and fetch stay for finding links and reading a page
whole ([record, 2026-10-16](../benchmarks/record/agent.md#why-a-third-tool-2026-10-16)).

**Waiting without freezing.** `src/web/client.zig` runs each request in a
concurrent task and waits in 50 ms beats, a pattern `bash`'s polling loop
already uses: each beat the agent reads keys and repaints, and Ctrl-C cancels
the request (the task is joined before the call returns, so nothing it
borrowed outlives it). Its own deadline (15 s search, 25 s context, 30 s fetch) sits above
the service's budgets, so the service's `upstream_timeout` normally arrives
first.

**In the API.** A chat request lists `{"type": "web_search"}`,
`{"type": "web_fetch"}`, or `{"type": "web_context"}` among its tools. The job intercepts those calls in its
sink, so the client never sees them as `tool_calls`, runs them on the GPU worker
while short decision and embedding items run in the waits, appends the
assistant message and the results, and completes again, at most six rounds
([api-chat.md § Hosted web tools](../guide/api-chat.md#hosted-web-tools)). The model's
state carries over between rounds, as between an agent's steps. Since the
client's next request resends only the message it got back, the job keeps
what the model saw in a bounded stash (`src/api/chat/stash.zig`) keyed by a
digest of the client's conversation through that message, and puts it back
when the message returns.

**What it costs in trust.** Queries and URLs leave the machine: the search
engines see the queries, the sites see the fetches. The guideline tells the
model never to put workspace code, paths, or secrets in a query, and the
transcript (or the API's `nuclis_web_calls`) shows every query and URL. A
fetched page can steer the model; its text is data, as the guideline says, and
`bash` could already reach the web, so the tools add no capability, only a
better one.

## 12. The agent's token cache

Without a cache the agent re-prefills tokens it has already computed whenever
the session cannot continue where it stands: every start (the system block and tools),
a cancelled step, an effort change, compaction. `src/completer/cache.zig` keeps
snapshots at **boundaries** and `Completer` (`src/completer.zig`) restores
the longest one a render starts with, then prefills the rest. New tokens
(a tool result, the user's message) are never cached; only re-prefill is
removed.

- **Boundaries.** The primed prefix, one per effort (the effort is part
  of the system block), and the end of every turn that ends in an answer
  (`Model.checkpoint`, called by the loop and by the server). A step whose
  remainder holds the system block's end also keeps the state there, of at
  least 64 tokens, when the memory tier does not hold it: `complete` feeds
  the prompt in two segments and calls back between them
  (`engine.Saves`); this is the server's primed prefix, since it never
  primes. Every boundary sits before a control token, which BPE never
  merges across, so the remainder encodes the same alone.
- **Media.** The placeholder text of an image or a clip is the
  same for every image or clip of its size, so the text the completer
  matches on is the render with each placeholder run followed by its
  item's SHA-256 between two 0xFF bytes, never valid UTF-8 and so never in
  rendered text (`keyedRender`); the tokenizer reads it without them
  (`plainText`). A state with media is then kept, restored, and moved back
  to like any other, under the same media only. Speculation stays off
  while the consumed text holds media (the drafter's cache is stale over
  their rows), and a step that prefills media saves no state inside its
  prompt (`engine.Saves` refuses an image prefill).
- **Rewritten turns.** Gemma 4's template never re-renders an answer as
  the model generated it (`Profile.rewritesTurn`), so neither the session
  nor an answer-end state is ever a prefix of the next turn. On an
  attention-only model the step marks where its prompt ended
  (`Profile.promptEnd`: before `<|turn>model\n`, or before the thought
  channel a tool result leaves open), with the position and the penalty
  set there; `checkpoint` moves the session back to the mark
  (`Session.rewindTo`, no copy) before keeping it, and a step that never
  reached `checkpoint` (an answer cut at its budget) moves back when the
  next render starts with the mark. The next turn continues from there,
  prefilling the answer as rendered and the new message. A recurrent model
  would need a snapshot at the mark instead; no profile it runs rewrites
  its turns.
- **Matching** is on the rendered text the state consumed, not on tokens:
  a turn's generated tokens need not be the canonical encoding of their
  text, so re-encoding the render would miss them. A hit must be a proper
  prefix, leaving a token whose logits the next sample needs. A cached
  state further along than the session wins even when the session could
  continue (after a re-prime under a conversation, the session stands at
  the system block). The penalty history is a 31 KB bitset copied with the
  state.
- **Memory tier**: in the process, least recently used out first under
  `cache.memory_bytes` (4 GiB; 0 keeps none). A Qwen state is 150 MB plus
  64 KiB per token, about 250 MB primed and 1.2 GB at 16K. A context or
  model change (`/model`, Ctrl-W) drops it (the states are bound to the
  engine that made them).
- **Disk tier**: the primed prefix, and the last turn end of each
  conversation that has a session file (each save replaces that
  conversation's previous one), under `<root>/cache/prefix/` through
  `inference.prefix_cache` (header with the keys, content digest; a file
  that fails either is deleted and reads as a miss). The key's token half
  is the digest of the rendered text the state consumed, which is also the
  middle of the file name, so a session's states are found without loading
  a model (`cache.removeDigest`, used by `/resume`'s Ctrl-D and `agent rm`). The
  key's model half hashes the model files (and the draft companion when a
  drafter is loaded), the executor, and the build: version and commit,
  plus the executable's size and modification time for a build from a
  modified tree, since such builds share a commit (`cache.buildId`). A
  different build misses, never restores. `cache.disk_bytes` (8 GiB; 0
  disables) bounds the directory; a hit refreshes a file's modification
  time and each save evicts the oldest beyond the budget.
- **Resume.** The turn's last `assistant` entry records its `boundary`
  (the text's byte length and digest). A resume (`Completer.resumeFrom`)
  tries the file's last boundary before the memory tier: when the render
  starts with it, the state is loaded from disk and the penalty history
  is rebuilt from the text's encoding. The system block carries the date,
  so a conversation resumed on another day misses and replays.
- **Model switch.** `/model` unloads the engine, opens the chosen file,
  empties the memory tier, and re-opens the disk tier under the new
  model's key, so a model used before restores its primed prefix from disk. The
  conversation is rendered through the new profile and prefilled whole
  (cause `model`), and the session file records a `model` entry beside
  `context` and `effort`.
- **Accounting.** A step that could not continue the session reports a
  `Replay` with its cause (`resumed`, `cancel`, `rewrite` for compaction or
  elision, `effort`, `context`, `model`, `failure`) and the tokens restored; the
  bar shows `replayed <cause> <prefilled>, <restored> restored`, and the
  session file records `replay` and `restored_tokens` beside `replayed`.

What the cache saves, and the checks of the save inside a prompt and the
rewind: [record, 2026-10-04 and 2026-10-08](../benchmarks/record/agent.md#the-agents-token-cache-2026-10-04).
