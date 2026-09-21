# Changelog

Notable changes, newest first.

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
