# Changelog

Notable changes, newest first.

## [v0.4.0] - 2026-10-03

### Features

- **nuclis:** serve drains queued decisions on Ctrl-C and exits cleanly
- **nuclis:** /v1/models reports each decision model's family, whether it packs, and whether it reads images
- **nuclis:** serve.timeout and --timeout: a request waits up to 300 s for the GPU (clef-flash holds it for seconds)
- **inference:** clef-flash, a second decision family: the joint schema head, images, decide and serve
- **inference:** qwen35 reads its shape from the file; hidden rows for decision heads
- **nuclis:** nuclis serve reads a serve section (port 8000), opens decide.model at start, logs a coloured line per request
- **nuclis:** nuclis serve batches decision requests that wait together into one pass
- **nuclis:** nuclis serve, the nuclis API: decisions over HTTP, TypeSafe's systemone call
- **nuclis:** speculation on by default where it pays; close ENGN-20
- **nuclis:** bench --draft-p-min and the speculative matrix driver
- **bench:** --kernel-stats prints each Metal pipeline's limits
- **bench:** saved prefixes, verify cost at depth, and the speed loop's A/B driver
- **metal-check:** capture names without spaces; the counter export's naming
- **metal-check:** CAPTURE= records a micro-benchmark case into a .gputrace
- **bench:** --capture records one decode step into a Metal .gputrace; make capture
- **inference:** Laya multilingual, the Metaspace tokenizer; close MODL-33
- **inference:** Laya on Metal, packed batches over sequence bounds; close MODL-31
- **gates:** make verify-auto picks the checks and gates a change needs (REPO-20 session 2, part)
- **gates:** a fast tier from code paths and a release tier (REPO-20 session 2, part)
- **nuclis:** Jev's answer fields, the decision path documented; close MODL-30
- **nuclis:** the laya decision catalogue entry, decide.model, registry kind decision (MODL-30 session 3, part)
- **nuclis:** nuclis decide over Laya's profile and the Decider (MODL-30 session 3, part)
- **inference:** Laya's encoder and decision head on the CPU (MODL-30 session 2)
- **inference:** Hugging Face tokenizer.json with NFC, and the Laya oracle (MODL-30 session 1)
- **nuclis:** shell completion answered by the binary; make install (APPS-18)
- **inference:** a generic safetensors loader, inspect on it (MODL-29)
- **nuclis:** pull safetensors sets with their support files, pinned like GGUF (MODL-28)

### Bug Fixes

- **nuclis:** name a support file passed to --file; skip other formats' folders
- **inference:** discard a failed recording; derive the kernel table
- **bench:** saved-prefix I/O in 1 GiB pieces; a failed save keeps the run
- **huggingface:** keep the Hub listing's digests in ReleaseFast builds (MODL-32)
- **tui:** exit keeps the transcript; reset margins without homing the cursor (TERM-13)

### Performance

- **inference:** word-outer multi-row matvec for 2-3 row batches
- **inference:** IQ4_XS matvec reads its code table from threadgroup memory
- **inference:** Q5_K matvec takes the half magic-number decode
- **inference:** Q4_K matvec decodes nibbles through half magic numbers
- **inference:** DeltaNet verify by replay tape; close ENGN-19
- **inference:** IQ4_NL small batches on the fragment tile
- **inference:** the fragment tile for Q3_K, IQ3_S, Q4_0, PQ2_0, PTQ1_0
- **inference:** few-row generic matrices of a small batch on the multi-row matvec
- **inference:** register-fragment small-batch matmul tile
- **inference:** few-query verify attention through the split pass
- **inference:** close KERN-19, the CPU tier in 22 minutes, bit-identical
- **inference:** the CPU reference's matvec across every core, bit-identical

### Other

- docs: AGNT-19 widened to token caching, turn-boundary snapshots in memory and on disk
- docs: the llm guide's speculation verdict after the decode-speed theme; AGNT-19 runs no inference gate tiers
- docs: close KERN-23, weight streaming for one row and a few
- docs: KERN-23, the IQ4_XS matvec's counters, make verify on session 1's kernels
- docs: KERN-23 session 1 hand-off
- docs: KERN-23 ledger, the K-scale conversion fails the amended keep rule
- chore(speed): the keep rule takes a 1-2 % gain whose pairs agree
- docs: KERN-23 ledger, the encode gap, K-scales below the keep rule, the step's profile
- docs: close REPO-30, Zig 0.17 holds Qwen's 4K speed
- docs: close REPO-29, the Zig 0.17.0 upgrade
- refactor: adopt Zig 0.17's @divCeil and ArrayList.lastPtr; require 0.17 (REPO-29)
- build: migrate the tree to Zig 0.17.0 (REPO-29 session 1)
- docs: REPO-29 checks 0.17 speed against the records instead of a 0.16 A/B
- docs: the project's Zig is the stable link, not current
- docs: record the zigup-managed Zig install and its local std and langref
- docs: REPO-29 adopts Zig 0.17's features and measures 0.16 → 0.17 speed
- docs: plan REPO-29, the Zig 0.17.0 upgrade, ahead of KERN-23
- docs: the README presents clef-flash beside Laya
- docs: TODO hand-off for KERN-23 after the cache cleanup: make speed-base first
- docs: close MODL-34, clef-flash on both backends with images, in decide and serve
- docs: close APPS-19, the nuclis API reference for clients
- refactor(nuclis): the decision wire format moves to src/decision/, shared by decide and the API
- docs(todo): MODL-34 checks only the new parts: no bf16 backbone, a head-only reference, a sanity set
- docs(todo): MODL-34 pulls its files first, reads the qwen35 shape from the file, keeps the mmproj in scope
- docs(todo): name the next step first
- docs(todo): APPS-19 builds the nuclis API layer, decisions its first service
- docs(todo): APPS-19 runs without model gates
- docs(todo): APPS-19 writes a client-facing API reference, docs/reference/serve.md
- docs(todo): drop KERN-22
- docs(todo): serve and clef-flash before KERN-23; KERN-22 deferred, AGNT-18 dropped
- docs(todo): APPS-19 serves lean and batches across requests
- docs(todo): queue APPS-19, nuclis serve, a local decision API
- docs(todo): the decide tool settles the default Laya checkpoint by a stated rule
- docs(todo): the decide tool measures both Laya checkpoints; the default stays laya
- docs(todo): base the decide tool's design on laya-multilingual and a labeled set
- docs: close REPO-27, the external review fixes
- refactor: name Qwen's mixer constants; comments stand without unit ids
- test(inference): corrupted GGUF and safetensors files never panic or leak
- docs(todo): REPO-27, the verified review fixes, ahead of KERN-23
- docs(todo): KERN-23 re-scoped to the verify's multi-row body
- docs(readme): the speculative defaults and their rates; REPO-26
- docs(todo): ENGN-20's code map, matrix, and verdict rule
- docs(todo): re-order the decode-speed units: ENGN-20, KERN-23, KERN-22
- docs(todo): ENGN-19 session 1 hand-off, tape implemented, gates pending
- docs(todo): ENGN-19's code map and first session
- docs(log): the register-fragment verify matmul; close KERN-24
- docs(todo): where KERN-24 stands
- docs(reference): the fragment tile's counters; KERN-24 session 1 hand-off
- test(metal-check): capture labels on the fragment-tile sweep; KERN-24 ledger
- docs(log): Gemma and Muse verify costs at depth; close KERN-21
- test(generation): verify rows against stepped decode at depth
- docs(reference): the multi-row matvec spills at 8 rows; close KERN-20
- docs(todo): cite the primed-prefix cost
- docs(todo): queue AGNT-19, saved prefixes for the agent across processes
- docs(reference): the verify's attention latency-bound on an empty GPU, its matmul tile issue-bound
- docs(todo): the speed loop's base is 4dc7c70
- docs(bench): the decode-speed baseline; close ENGN-18
- docs(development): who captures, exports, renames, and reads
- docs(reference): the last-level miss rate at full clocks
- docs(reference): the Q4_K matvec's integer-pipe limit confirmed at full clocks
- docs(reference): apple-gpu.md, the Q4_K matvec issue-bound on the integer pipe
- docs(development): capture profiling needs Xcode's Metal Toolchain
- docs(todo): where we are after the capture tooling
- docs(development): Xcode 27's MCP tools read no GPU capture
- docs(todo): the decode-speed theme, eight units behind a measured keep rule
- docs(adr): propose the Qwen small-batch verifier; close REPO-24
- build(gates): skip a gate whose model is absent, strict for releases; close REPO-25
- docs(readme): a shorter Laya section led by its read speed
- docs(todo): KERN-19 base and design from the CPU backend's call paths
- docs(readme): the Laya experiment and its results; close REPO-23
- build: durable reference state in .reference/, a disposable .zig-cache; close REPO-22
- refactor(scripts): a generated agent playground, typed and formatted Python; close REPO-21
- docs(todo): REPO-21, a generated playground and typed scripts, ahead of AGNT-18
- docs(todo): MODL-33, Laya multilingual, ahead of AGNT-18
- docs(todo): MODL-31 design from the Metal backend's facts
- docs(todo): Laya first (MODL-31, AGNT-18), then KERN-19; MODL-31 base
- docs(todo): KERN-19 base
- chore(gates): verify-auto runs one model at a time; close REPO-20
- docs(todo): REPO-20 gate set approved, sessions swapped
- chore(gates): time each gate's phases, the coverage matrix (REPO-20 session 1, part)
- docs: plan fast verification: gates from code paths, a threaded reference (REPO-20)
- chore(gates): retire the Gemma 4 12B K-quant file's gates (REPO-19)
- docs: plan shell completion and make install (APPS-18)
- docs: remove docs/research, carry its conclusion into the plan (REPO-18)
- docs: plan Laya end to end: CPU, Metal, the agent tool (MODL-30, MODL-31, AGNT-18)
- docs: plan safetensors acquisition and loader (MODL-28, MODL-29)
- chore: begin 0.4.0-dev

## [v0.3.0] - 2026-09-27

### Features

- **nuclis:** the catalogue at five entries, the registry in name order (APPS-17)
- **inference:** Gemma 4 E4B QAT end to end, checks pending (MODL-27)
- **agent:** blank bash output said, the paged-file rule, Ctrl-O's output view, a reasoning budget at low (AGNT-17)
- **eval:** teacher-forced perplexity against the reference, the all-rows prefill, a gate per family (APPS-14)
- **vision:** Bonsai 2's Q8_0 projector through the Qwen3-VL adapter (MODL-25); TERM-13 dropped
- **vision:** close MODL-23: Muse Glimmer's vision record, the image token cap, the docs
- **vision:** Muse Glimmer's windowed projector on both executors and image spans in its language model (MODL-23)
- **config:** generation.image_max_tokens, auto by default, with a per-model override and --image-max-tokens (MODL-23)
- **vision:** close MODL-22: the Gemma 4 vision record, the chunk buffers grown all-or-nothing, the docs
- **vision:** Gemma 4's projectors on both executors and bidirectional image spans in its language model (MODL-22)
- **agent:** a steer during a reasoning-only step restarts it; close AGNT-16, draft TERM-13
- **repo:** the screenshot harness runs the agent in a Ghostty window of its own (--direct)
- **agent:** images and text files in the chat — drop, /image, the chips, the projector turn, the detail row, the preview, sessions (AGNT-15)
- **vision:** image spans through the Qwen3.8 language model, generate --image, the vision gates (MODL-21)
- **vision:** session rope triples and the CPU multi-axis RoPE for image spans (MODL-21)
- **vision:** the Qwen3-VL projector on both executors, the image decoders, exact preprocessing, and the oracle harness (MODL-21, first half)
- **agent:** the Qwen decoder keeps a value's trailing newline, read_file serves the first MiB, Enter steers a running turn (AGNT-14)
- **agent:** the system prompt as sections, measured on the playground task list (AGNT-13)
- **term:** `!` and `!!` run a shell command, Ctrl-O folds tool output, Ctrl-X copies the last answer, Ctrl-G edits the input externally (TERM-10)
- **term:** markdown hardening: prefix fuzz, pathological goldens, nested quotes, bold italic, soft breaks kept, rows never past the edge (TERM-10)
- **term:** the diff's gutter, marker cell, and bands; the header counts the change (TERM-10)
- **term:** operation dots that pulse while running, Name(argument) call rows, write summaries, and the turn's operations row (TERM-10)
- **term:** frame the input box; its edge carries the effort's colour and the running spinner (TERM-10)
- **term:** the status bar states its settings at the right edge, the speculative switch among them (TERM-10)
- **term:** frame the header in a box, and keep it on screen through the first frames (TERM-10)
- **term:** the warm-up shows its progress in the region and reports when it is done (TERM-10)
- **term:** repaint at 10 Hz through every GPU wait, and rewrite only the changed rows (TERM-10)
- **term:** the tmux screenshot harness, and the header keeps its first row (TERM-10)
- **apps:** raise the output budget default to 4096 and its cap to 16384 (APPS-16)

### Bug Fixes

- **gemma4:** Metal verify rows carry the final soft-cap (MODL-26)
- **vision:** decode after an image writes and reads the right cache rows; the vision gate compares decode with one prefill; a spinner while images encode (MODL-24, TERM-12)
- **tui:** re-anchor the live region on a resize from the cursor report; the preview fits above the region and is off under tmux (TERM-11)
- **agent:** the typed-path scan skips command lines; the harness gains a bracketed paste step and tmux passthrough (AGNT-15)
- **term:** a fold toggle keeps the editor at the bottom, the input box loses its background, the bar says speculative (TERM-10)

### Performance

- **tokenizer:** queued BPE merge for long pieces; Gemma lines no longer hit the work bound (MODL-26 follow-up)
- **vision:** Muse's windows on the chunk attention kernel (encode 67 to 47 s at 16,320 patches); generate reports image_milliseconds; the check tool profiles the projector (MODL-23)

### Other

- docs: README rewritten with a recorded agent session; close MODL-27 (REPO-17)
- chore(scripts): nuclis_mem_usage.py splits a running process's memory into GPU and CPU (REPO-16)
- test(eval): a long-context perplexity tier; Gemma 4 12B passes at 4K, the 26B-A4B's -1.79 % is open (MODL-26 follow-up)
- docs(repo): the CPU tier runs only when a unit changes what the CPU reference computes, and before a release (REPO-15)
- docs(todo): MODL-23 takes generation.image_max_tokens (auto by default, a per-model override and a flag)
- docs(todo): MODL-23 facts from the reference and the Muse projector file; the pinned synthetic fixture; the oracle takes --n-ctx and --vision-flash
- docs(todo): MODL-22 facts from the reference and the Gemma 4 projector files; the oracle takes image token bounds
- docs(repo): the screenshot harness is the validation step for surface changes (REPO-14)
- docs(plan): close AGNT-15 — images and text files in the chat, the log entry, the spec and reference updates
- docs(plan): AGNT-15 opened — text-file chips, feature ownership, the speculation rule, the preview's terminal test
- docs(plan): close MODL-21 — the Qwen3.8 vision projector, verified on both executors
- docs(plan): AGNT-16 ordered: the Muse value contract measured, steering that interrupts reasoning, the guessed-path guideline
- docs(plan): AGNT-12 dropped, its steering folded into AGNT-14 with the two measured tool fixes; the images unit renumbered AGNT-15
- docs(repo): the Metal tier is run for a unit that touched the inference stack, judged by the work
- docs(term): TERM-10 closed: the repaint tick, the frame, the operation rows, the banded diff, and the editor's shell and keys
- docs(repo): one specification: spec.md rewritten as a technical spec with the agent spec merged in (REPO-13)
- docs(repo): the architecture guide as a map: tables, bullets, and pointers, no identifiers in prose (REPO-12)
- docs(repo): the architecture guide follows the KV cache end to end; the inference guide rewritten as one narrative (REPO-12)
- docs(term): the transcript's operation rows, the bar's groups, the warm-up, and the session-1 hand-off (TERM-10)
- docs(plan): order the agent units before the vision units
- docs(plan): order AGNT-11 ahead of MODL-21
- docs(plan): draft AGNT-13, the system prompt as sections, after pi's
- docs(plan): widen TERM-10 to two sessions with the harness, the repaint tick, and pi's ideas
- docs(repo): point the plan at TERM-10 after the v0.2.0 tag
- chore: begin 0.3.0-dev

## [v0.2.0] - 2026-09-21

### Breaking Changes

- **engn:** speculative configuration, the bench off/on pair, and the loop edge tests (ENGN-12)

### Features

- **apps:** config init --discover and self-contained help pages (APPS-15)
- **repo:** benchmark workloads as data and generated record tables (REPO-10)
- **repo:** one gate registry with tiers and change triggers (REPO-09)
- **apps:** set the speculative verdicts and bench's true baseline (ENGN-17)
- **model:** run the Muse DFlash drafter on Metal (MODL-20)
- **model:** add the Muse DFlash drafter's CPU reference (MODL-20)
- **model:** add the Gemma 4 assistant draft heads (MODL-19)
- **engn:** stop drafting below the p_min probability; drop the adaptive length (ENGN-16)
- **engn:** decide sampled acceptance from the per-row top-k readback (ENGN-15)
- **kern:** apply the history penalties on the device before selection (KERN-13)
- **engn:** per-row recurrent checkpoints replace the recovery replay (ENGN-14)
- **engn:** measure recovery by accepted length and take the seed-only replay through step (ENGN-14)
- **kern:** register-tile the multi-row matvec and route 2-row batches (KERN-12)
- **kern:** multi-row matvec kernels and sweep, routing gated off (KERN-12)
- **engn:** time propose and commit, carried into the bench sample (ENGN-13)
- **engn:** prime the drafter and surface accepted drafts per step (ENGN-13)
- **engn:** batched drafter commit on the Metal plan (ENGN-13)
- **engn:** commit the prompt to the drafter in prefill chunks (ENGN-13)
- **engn:** prefill returns post-norm hidden rows (ENGN-13)
- **engn:** the speculative record closes ENGN-12; the plan revamped for the speed units, vision, and the chat polish
- **engn:** speculative configuration, the bench off/on pair, and the loop edge tests (ENGN-12)
- **engn:** batched verify, the speculative loop, and sampled acceptance (ENGN-12)
- **modl:** the Qwen prediction block on the Metal plan, the compare rows, and the acceptance statistic (MODL-18)
- **apps:** validate reports the embedded draft head
- **modl:** the Qwen prediction block on the CPU, the draft contract, and the load request (MODL-18)
- **modl:** bind the Qwen3.8 prediction head and keep the target hidden (MODL-18)
- **engn:** speculative state recovery — checkpoint, rewind, truncate, recover (ENGN-11)
- **metal:** an 8x8 split-K prefill tile for chunks up to eight tokens
- **profiles:** Muse Glimmer profile acceptance: the reference and nuclis records, the harness family, the log; the unit closes (MODL-13)
- **profiles:** Muse Glimmer ATEM tool calling: declarations, calls, results, and the body parser match the pinned fixtures (AGNT-10)
- **profiles:** the Muse Glimmer profile with the channel-grammar decoder, the high effort, and the catalogue pin (MODL-13)
- **models:** the Muse Glimmer Metal plan matches the pinned traces in both cache precisions; RoPE gains adjacent pairing (MODL-12)
- **models:** the Muse Glimmer adapter and CPU reference match the pinned traces; the unit closes (MODL-11)
- **tokenizer:** the llama4 splitter matches the reference on Muse Glimmer; the family's facts, fixtures, and inventory (MODL-11)
- **config:** the default context window is 16K (APPS-12)
- **metal:** the Qwen plan runs Bonsai 2 27B: the rotation on every activation, the catalogue on PTQ1_0, the acceptance record (MODL-17)
- **metal:** the signed Walsh-Hadamard transform kernel and its per-token bench (KERN-10)
- **metal:** ternary matvecs and prefill tiles for PQ2_0 and PTQ1_0, BF16 rows (KERN-10 session 1)
- **engine:** the Qwen CPU reference applies the Hadamard rotation; Bonsai 2 27B matches the fork's traces (MODL-16)
- **model:** the Qwen adapter binds Bonsai 2 27B: rotation contract, optional auxiliary block
- **quant:** decode PQ2_0, PTQ1_0, and BF16 rows against the PrismML fork's fixtures
- **model:** Bonsai 2 27B accepted ahead of Muse: catalogue entry pinned and pulled, facts read, three units planned; MODL-15 closed
- **model:** Gemma 4 26B-A4B supported: acceptance record against the reference, per-kernel profile, agent check; MODL-10 closed
- **model:** Gemma 4 26B-A4B Metal plan: expert block on decode and chunked prefill, two-KV-head global attention; MODL-09 closed
- **model:** Gemma 4 26B-A4B adapter and CPU expert layer against pinned reference traces (MODL-09 session 1)
- **kern:** expert prefill path: GPU row lists and gathered matmul tiles; KERN-09 closed
- **term:** welcome with the ASCII wordmark and session facts; model name on the status bar (TERM-08)
- **apps:** model ls sizes in decimal units (APPS-10)
- **apps:** model ls as one aligned grid (APPS-10)
- **apps:** config set, model pull --register, registry names in model ls (APPS-09)
- **apps:** --prompt-profile and the registry's profile force a prompt profile; template-alias gate script (APPS-07)
- **model:** catalogue entries for gemma-4-26b-a4b and muse-glimmer-30b; roadmap reordered
- **kern:** expert routing and gathered expert kernels, decode path (KERN-09 session 1)
- **profile:** Gemma 4 native tool calling: declarations, calls, results, and the tool_response handoff (AGNT-09)
- **agent:** nuclis agent ls, and --resume alone continues the newest session (APPS-06)
- **engine:** primed sessions: prefill the system prefix at startup, restore it on new (ENGN-09)
- **agent:** per-step thinking blocks with their own duration (TERM-06)
- **agent:** in-turn elision of older tool results on a full window (AGNT-08 closed)
- **agent:** context-relative result budget and an honest context-full message (AGNT-08, session 1)
- **agent:** two-row tool lines, call and detail (TERM-05)

### Bug Fixes

- **repo:** repair multi-row benchmark controls
- **engn:** sampled acceptance draws from the target; the batch catches the real cancellation (ENGN-12)
- **metal:** set small_chunk_tokens to 24 from the wall-clock crossover
- **metal:** order the split tile's step reuse with simdgroup barriers
- **metal:** the split-K tile stores each partial into its own tile
- **tokenizer:** the special-token scan searches each marker's first byte, so a vocabulary with thousands of markers tokenizes long text within budget
- **tui:** a tool call closes the step's text, and a closed thought keeps its fold label in the live region (TERM-09)
- **agent:** an empty tool result costs zero tokens instead of ending the turn (AGNT-12)
- **model:** pull keeps the sidecar of a verified file whose encoding this build does not store (APPS-11)
- **agent:** released calls keep their bracket, reopened thought channels are thinking, own markers are stripped from history (AGNT-11)
- **model:** free each job's local path in pull; sidecar file names lose their leading slash (APPS-08)
- **term:** reuse the rows a shrinking live region releases instead of leaving gaps (TERM-07)
- **agent:** the system prompt forbids reconstructing a file from memory
- **agent:** steer a whole-file question to one command instead of paging; the bar's word after a turn
- **agent:** a cut read's row states the kept range; a step's prefill resets the bar's word and decode clock
- **agent:** free the sidecar strings and the cwd sentinel on the interactive path
- **agent:** read_file counts a trailing newline as the end of the last line

### Performance

- **kern:** fuse the decode norms behind a control flag, measured below target (KERN-18)
- **kern:** measure the register-reuse chunk attention and keep its verify window (KERN-16)
- **kern:** measure the split-K decode matvec behind the single pass (KERN-15)
- **kern:** measure the wide 32x8 small-batch tile and keep the 16x8 control (KERN-14)
- **metal:** 16 rows per split-tile threadgroup and a 16-token threshold

### Other

- docs(repo): plan APPS-15 (config init --discover, self-contained help); drop REPO-11
- docs(repo): reshape REPO-09/REPO-10 around gate tiers and change triggers
- docs(repo): spell out the Makefile and bench.md division of REPO-09/REPO-10
- docs(repo): drop the Gemma decode, ring-layout, and ternary units from the plan
- docs(repo): order the gate registry and benchmark workloads after the verdict (REPO-09, REPO-10)
- docs(model): pin the Gemma 4 assistant head facts and traces (MODL-19)
- docs(repo): add KERN-18 (fused decode norms) and rescope ENGN-18 to the Gemma tiles
- docs(repo): pull the Gemma 4 and Muse drafters ahead of the verdict (MODL-19, MODL-20)
- docs(engn): close ENGN-14 with the row-checkpoint record and replan the speculative units
- build(repo): default compare aggregates to Metal; wire bench-matvec-rows
- docs(engn): close ENGN-13, the prompt and batched drafter commit (ENGN-13)
- docs(engn): record the ENGN-13 prompt-commit and batched-commit numbers (ENGN-13)
- test(engn): check the batched commit against the serial path (ENGN-13)
- docs(engn): record the batched drafter commit measurement (ENGN-13)
- docs(engn): record the prefill-chunk prompt commit measurement (ENGN-13)
- docs(repo): retire the roadmap; themes are agreed in session and, when architectural, recorded as ADRs on request (REPO-07)
- docs: research DiffusionGemma decisions and kev
- docs(plan): MODL-18 session 1 complete on the CPU; session 2 next
- docs(metal): the small-chunk tile's threshold and geometry stated consistently after KERN-11
- docs(kern): KERN-11 closes; the log entry, the roadmap bullet removed, the plan points at ENGN-11
- docs(bench): KERN-11's 22-token prompt: prefill +10.9 %, first token -9.8 %
- bench(metal): batch matmulBench dispatches and attribute bytes per token tile
- docs(plan): the speculative units rewritten at implementation level from the companions' headers and the reference's driver; the protocol asks for that level
- docs(plan): KERN-11 session 1 reviewed; the batched bench, the activation operand, and the barrier set session 2's order
- docs(metal): the small-chunk tile's first measurements and the KERN-11 session 1 hand-off
- style(metal): reflow the kernel_names array after the new entries
- bench(metal): report weight-byte GB/s and the selected tile in matmulBench
- test(metal): matmul exactness at 1, 5, and 8 tokens and the split-tile thresholds
- docs(plan): KERN-11, the small-chunk prefill matmul, inserted ahead of the speculative units
- docs(plan): speculative decoding across the families planned as five units; its requirements move to the spec
- docs(roadmap): the Muse Glimmer decode gap's experiments join the performance theme (ENGN-10)
- docs(agent): Muse Glimmer ATEM tool calling closes on the pinned fixtures and the live turn; the plan is empty (AGNT-10)
- docs: Bonsai 2 27B facts, the fork as second oracle, traces, alias evidence, guide § 49 (MODL-16 session 1)
- docs(readme): acknowledge ds4 as the Metal bridge's design reference
- docs(readme): every supported file in the results table
- docs: toolchain paths read as the author's machine, not as requirements
- docs(readme): project status, contributions policy, and AI and llama.cpp disclosure for going public (REPO-05)
- docs: research Jev and native Zig training options
- docs(log): TERM-07 confirmed on a real terminal
- docs(agents): the session protocol as two states over three files
- docs(log): AGNT-09 table row, MODL-14 for the catalogue and roadmap work; AGENTS.md closing rules
- docs(roadmap): speculative decoding configuration: file per entry, per-command switch, draft length
- docs(todo): Gemma 4 26B-A4B (KERN-09, MODL-09, MODL-10) ahead of Muse Glimmer; renumber
- docs(todo): Muse profile renders the default system turn without a date line
- docs(todo): plan Muse Glimmer 30B bring-up and ATEM tool calling (MODL-09..11, AGNT-10)
- docs(todo): plan AGNT-09, Gemma 4 native tool calling
- docs(todo): open TERM-06 per-step thinking and ENGN-09 primed sessions
- docs: llm-guide extends only on request; drop the TERM-05 section
- chore: begin 0.2.0-dev

## [v0.1.0] - 2026-09-15

### Features

- import nuclis v0.1
