# Changelog

Notable changes, newest first. From v0.6.0 on, release-please writes each
section from the merged pull requests' titles, led by the release's
highlights; each line links its pull request, whose description is the
record of the work. Releases up to v0.5.0 list the units they closed, linked
to the worklog at their tag.

## [v0.5.0] - 2026-10-04

The agent gets faster per step. A token cache keeps the primed prompt and each turn's end in memory and on disk, so a session resumes without prefilling it again (a 9.8K-token resume 126.5 → 2.05 s; task wall time −23 %). Tools read less (pages, outlines, grouped `grep`, short diffs; tool-result tokens −28 %), and a new command surface with `/model` switches models in place. Breaking: the slash commands changed.

### Agent

- **[AGNT-19](https://github.com/tildaslashalef/nuclis/blob/v0.5.0/docs/engineering-log.md#agnt-19--token-caching-for-the-agent-turn-boundary-snapshots-in-memory-and-on-disk-session-management-2026-10-04-two-sessions)** Token caching for the agent: turn-boundary snapshots in memory and on disk; session management
- **[AGNT-20](https://github.com/tildaslashalef/nuclis/blob/v0.5.0/docs/engineering-log.md#agnt-20--a-faster-agent-step-the-command-surface-and-model-reading-less-and-the-fixes-from-testing-2026-10-04-two-sessions)** A faster agent step: the command surface and `/model`, reading less, and the fixes from testing

### Repository

- **[REPO-31](https://github.com/tildaslashalef/nuclis/blob/v0.5.0/docs/engineering-log.md#repo-31--ci-installs-zig-0170-2026-10-03)** CI installs Zig 0.17.0

### Breaking changes

- **agent:** the command surface and /model

<details><summary>All 19 commits</summary>

- [`6cc9ec8`](https://github.com/tildaslashalef/nuclis/commit/6cc9ec884ca387f16aa8001853f703383c9098f7) chore(release): v0.5.0
- [`320bfeb`](https://github.com/tildaslashalef/nuclis/commit/320bfeb09511d3d337c897b92631f754f14e5487) docs: close AGNT-20, the command surface, reading less, and the fixes from testing
- [`ba75456`](https://github.com/tildaslashalef/nuclis/commit/ba754562983a410e37b9f6b4cd62fdd18c600695) feat(config): max_tokens 8192 and a reasoning cap at every effort (AGNT-20)
- [`6fe11f1`](https://github.com/tildaslashalef/nuclis/commit/6fe11f1f3d180d648d6ca28d71ff24a8d46346b9) feat(agent): read less: pages, outlines, grouped grep, short diffs, repeat pointers (AGNT-20)
- [`4c55dfc`](https://github.com/tildaslashalef/nuclis/commit/4c55dfc413396d92ca0d35e5b737a8a4ebbc4214) fix(inference): bound reopened reasoning, drop cut calls and stray closes, name bad tool arguments (AGNT-20)
- [`c141cd5`](https://github.com/tildaslashalef/nuclis/commit/c141cd5ddba707505d5e8c415691972e025a9bb2) fix(agent): Enter on the command list runs the highlighted command; a bare slash is not a prompt (AGNT-20)
- [`b86c5d1`](https://github.com/tildaslashalef/nuclis/commit/b86c5d1fdd761e757e811e2c196d0506913e407b) test(agent): measure tool-result tokens per tool, add the about task (AGNT-20)
- [`1007b08`](https://github.com/tildaslashalef/nuclis/commit/1007b08d3ee69274919555dd236f0a15047070a8) feat(agent)!: the command surface and /model (AGNT-20)
- [`6bfae8c`](https://github.com/tildaslashalef/nuclis/commit/6bfae8ce806052893a2984c8ad52ce8dfe16938e) docs: close AGNT-19, the agent's token cache and session management
- [`e8a26b3`](https://github.com/tildaslashalef/nuclis/commit/e8a26b3cd0a1cd1154096a8ca67248c532ce49a1) docs: plan AGNT-20, the command surface and /model, reading less, faster prefill
- [`c1bb45a`](https://github.com/tildaslashalef/nuclis/commit/c1bb45a7a25deb0b40b0fda5214cc0b39789392d) fix(agent): print mode's --resume restores the last turn from the token cache (AGNT-19)
- [`db0ae43`](https://github.com/tildaslashalef/nuclis/commit/db0ae43e2aa3e468c9934b9eb10c829d836608bd) fix(agent): a closed picker leaves the bar ready (AGNT-19)
- [`4e10876`](https://github.com/tildaslashalef/nuclis/commit/4e10876b46ef8449272f166ef4ba257643023697) feat(agent): resume from the token cache, /list, /delete, agent rm, nuclis cache, Esc cancels (AGNT-19)
- [`4df6f9f`](https://github.com/tildaslashalef/nuclis/commit/4df6f9f83217b9656ecb165a2eb87501d3904289) fix(agent): bash spawns from the process directory, fuzzy @ completion (AGNT-19)
- [`3b43873`](https://github.com/tildaslashalef/nuclis/commit/3b43873f7de609383cb6a1f4069a1c812937a4ed) feat(agent): token cache for the primed prefix and turn ends, in memory and on disk (AGNT-19 session 1)
- [`9f45bf0`](https://github.com/tildaslashalef/nuclis/commit/9f45bf080d2392c9f1ca38e41fa76fb0ef1746fb) docs: AGNT-19 adds /list, /delete, and agent rm to session 2
- [`7bc3a3a`](https://github.com/tildaslashalef/nuclis/commit/7bc3a3acde573e4506064f63dd3119aff1b9ffba) fix(ci): release installs Zig from the tag's version and the workflow's digests (REPO-31)
- [`900e5ea`](https://github.com/tildaslashalef/nuclis/commit/900e5eab69bca8e6b8caa77a225f3c33212295bd) fix(ci): pin Zig 0.17.0 tarball digests for install-zig (REPO-31)
- [`f3af8ee`](https://github.com/tildaslashalef/nuclis/commit/f3af8eee470cf2602b4ffcdc3618d19ea09d9063) chore: begin 0.5.0-dev

</details>

**Full diff:** [v0.4.0...v0.5.0](https://github.com/tildaslashalef/nuclis/compare/v0.4.0...v0.5.0)

## [v0.4.0] - 2026-10-03

nuclis moves to Zig 0.17.0 with Qwen's speed unchanged (10.0 tok/s at 4K under both compilers). Speculation is re-priced per family and on by default where it pays: Qwen at draft 7 writes short code at 20.3–20.9 tok/s, helped by the DeltaNet replay tape and cheaper verify attention, and plain decode rises from 10.37 to 11.78 tok/s. Two decision models, Laya (ModernBERT) and Cloudflare's clef-flash, run on both backends and are served over HTTP by the new `nuclis serve`. Also: safetensors downloads, shell completion, and `make verify` in a third of the time (704 → 246 s).

### Models

- **[MODL-28](https://github.com/tildaslashalef/nuclis/blob/v0.4.0/docs/engineering-log.md#modl-28--safetensors-sets-through-the-hub-client-and-model-pull-2026-09-27)** Safetensors sets through the Hub client and `model pull`
- **[MODL-29](https://github.com/tildaslashalef/nuclis/blob/v0.4.0/docs/engineering-log.md#modl-29--a-generic-safetensors-loader-and-nuclis-inspect-on-it-2026-09-27)** A generic safetensors loader and `nuclis inspect` on it
- **[MODL-30](https://github.com/tildaslashalef/nuclis/blob/v0.4.0/docs/engineering-log.md#modl-30--laya-on-the-cpu-end-to-end-the-oracle-tokenizerjson-modernbert-and-the-decision-head-nuclis-decide-2026-09-29)** Laya on the CPU end to end: the oracle, `tokenizer.json`, ModernBERT and the decision head, `nuclis decide`
- **[MODL-31](https://github.com/tildaslashalef/nuclis/blob/v0.4.0/docs/engineering-log.md#modl-31--laya-on-metal-packed-batches-bidirectional-windowed-attention-over-sequence-bounds-2026-09-29)** Laya on Metal: packed batches, bidirectional windowed attention over sequence bounds
- **[MODL-32](https://github.com/tildaslashalef/nuclis/blob/v0.4.0/docs/engineering-log.md#modl-32--the-hub-listing-keeps-its-digests-in-releasefast-builds-2026-09-29)** The Hub listing keeps its digests in ReleaseFast builds
- **[MODL-33](https://github.com/tildaslashalef/nuclis/blob/v0.4.0/docs/engineering-log.md#modl-33--laya-multilingual-the-metaspace-tokenizer-the-checkpoints-own-special-tokens-checked-on-both-backends-2026-09-29)** Laya multilingual: the Metaspace tokenizer, the checkpoint's own special tokens, checked on both backends
- **[MODL-34](https://github.com/tildaslashalef/nuclis/blob/v0.4.0/docs/engineering-log.md#modl-34--clef-flash-cloudflares-9b-decision-model-text-and-vision-on-both-backends-2026-10-02-four-planned-sessions-in-one)** clef-flash: Cloudflare's 9B decision model, text and vision, on both backends

### Engine

- **[ENGN-18](https://github.com/tildaslashalef/nuclis/blob/v0.4.0/docs/engineering-log.md#engn-18--the-speed-loop-saved-prefixes-verify-cost-at-depth-interleaved-ab-and-the-decode-speed-baseline-2026-09-30)** The speed loop: saved prefixes, verify cost at depth, interleaved A/B, and the decode-speed baseline
- **[ENGN-19](https://github.com/tildaslashalef/nuclis/blob/v0.4.0/docs/engineering-log.md#engn-19--the-deltanet-replay-tape-2026-10-01)** The DeltaNet replay tape
- **[ENGN-20](https://github.com/tildaslashalef/nuclis/blob/v0.4.0/docs/engineering-log.md#engn-20--speculation-re-priced-per-family-the-catalogues-defaults-2026-10-02)** Speculation re-priced per family; the catalogue's defaults

### Kernels

- **[KERN-19](https://github.com/tildaslashalef/nuclis/blob/v0.4.0/docs/engineering-log.md#kern-19--a-threaded-cpu-reference-bit-identical-the-cpu-tier-in-22-minutes-2026-09-30)** A threaded CPU reference, bit-identical: the CPU tier in 22 minutes
- **[KERN-20](https://github.com/tildaslashalef/nuclis/blob/v0.4.0/docs/engineering-log.md#kern-20--seeing-inside-the-gpu-metal-captures-pipeline-statistics-apple-gpumd-2026-09-30)** Seeing inside the GPU: Metal captures, pipeline statistics, `apple-gpu.md`
- **[KERN-21](https://github.com/tildaslashalef/nuclis/blob/v0.4.0/docs/engineering-log.md#kern-21--few-query-verify-attention-through-the-split-pass-2026-09-30)** Few-query verify attention through the split pass
- **[KERN-22](https://github.com/tildaslashalef/nuclis/blob/v0.4.0/docs/engineering-log.md#kern-22--long-context-decode-attention-dropped-before-it-started-2026-10-02)** Long-context decode attention: dropped before it started
- **[KERN-23](https://github.com/tildaslashalef/nuclis/blob/v0.4.0/docs/engineering-log.md#kern-23--weight-streaming-for-one-row-and-a-few-qwen-decode-14--23-row-verify-batches-612--cheaper-2026-10-03-two-sessions)** Weight streaming for one row and a few: Qwen decode +14 %, 2–3-row verify batches 6–12 % cheaper
- **[KERN-24](https://github.com/tildaslashalef/nuclis/blob/v0.4.0/docs/engineering-log.md#kern-24--the-register-fragment-verify-matmul-2026-10-01-closed-below-its-target)** The register-fragment verify matmul

### Agent

- **[AGNT-18](https://github.com/tildaslashalef/nuclis/blob/v0.4.0/docs/engineering-log.md#agnt-18--the-agents-decide-tool-dropped-before-it-started-2026-10-02)** The agent's `decide` tool: dropped before it started
- **AGNT-19** Token caching for the agent: snapshots at turn boundaries, in memory and on disk _(in progress)_

### Application

- **[APPS-18](https://github.com/tildaslashalef/nuclis/blob/v0.4.0/docs/engineering-log.md#apps-18--shell-completion-answered-by-the-binary-make-install-2026-09-27)** Shell completion answered by the binary; `make install`
- **[APPS-19](https://github.com/tildaslashalef/nuclis/blob/v0.4.0/docs/engineering-log.md#apps-19--nuclis-serve-the-nuclis-api-decisions-first-batched-across-requests-2026-10-02-two-sessions-in-one)** `nuclis serve`, the nuclis API: decisions first, batched across requests
- **[APPS-20](https://github.com/tildaslashalef/nuclis/blob/v0.4.0/docs/engineering-log.md#apps-20--nuclis-serve-configured-and-logged-the-serve-section-port-8000-the-default-model-at-start-a-line-per-request-2026-10-02)** `nuclis serve` configured and logged: the `serve` section, port 8000, the default model at start, a line per request

### Terminal

- **[TERM-13](https://github.com/tildaslashalef/nuclis/blob/v0.4.0/docs/engineering-log.md#term-13--exit-keeps-the-transcript-2026-09-27)** Exit keeps the transcript

### Repository

- **[REPO-18](https://github.com/tildaslashalef/nuclis/blob/v0.4.0/docs/engineering-log.md#repo-18--docsresearch-removed-2026-09-27)** `docs/research/` removed
- **[REPO-19](https://github.com/tildaslashalef/nuclis/blob/v0.4.0/docs/engineering-log.md#repo-19--the-gemma-4-12b-k-quant-files-gates-retired-with-the-file-2026-09-29)** The Gemma 4 12B K-quant file's gates retired with the file
- **[REPO-20](https://github.com/tildaslashalef/nuclis/blob/v0.4.0/docs/engineering-log.md#repo-20--fast-verification-gates-re-derived-from-code-paths-a-release-tier-make-verify-auto-2026-09-29)** Fast verification: gates re-derived from code paths, a release tier, `make verify-auto`
- **[REPO-21](https://github.com/tildaslashalef/nuclis/blob/v0.4.0/docs/engineering-log.md#repo-21--a-generated-playground-the-python-scripts-typed-and-formatted-2026-09-29)** A generated playground; the Python scripts typed and formatted
- **[REPO-22](https://github.com/tildaslashalef/nuclis/blob/v0.4.0/docs/engineering-log.md#repo-22--reference-for-durable-local-state-zig-cache-disposable-2026-09-29)** `.reference/` for durable local state; `.zig-cache` disposable
- **[REPO-23](https://github.com/tildaslashalef/nuclis/blob/v0.4.0/docs/engineering-log.md#repo-23--the-readmes-laya-section-the-experiment-and-its-results-2026-09-29)** The README's Laya section: the experiment and its results
- **[REPO-24](https://github.com/tildaslashalef/nuclis/blob/v0.4.0/docs/engineering-log.md#repo-24--qwen-decode-at-20-tokenss-evidence-and-experiment-proposal-2026-09-30)** Qwen decode at 20 tokens/s: evidence and experiment proposal
- **[REPO-25](https://github.com/tildaslashalef/nuclis/blob/v0.4.0/docs/engineering-log.md#repo-25--a-gate-whose-model-is-absent-is-skipped-not-failed-2026-09-30)** A gate whose model is absent is skipped, not failed
- **[REPO-26](https://github.com/tildaslashalef/nuclis/blob/v0.4.0/docs/engineering-log.md#repo-26--the-readmes-speculative-decoding-results-2026-10-02)** The README's speculative decoding results
- **[REPO-27](https://github.com/tildaslashalef/nuclis/blob/v0.4.0/docs/engineering-log.md#repo-27--external-review-fixes-the-metal-discard-path-the-kernel-table-the-downloader-parser-property-tests-2026-10-02)** External review fixes: the Metal discard path, the kernel table, the downloader, parser property tests
- **[REPO-28](https://github.com/tildaslashalef/nuclis/blob/v0.4.0/docs/engineering-log.md#repo-28--the-readme-presents-clef-flash-beside-laya-2026-10-03)** The README presents clef-flash beside Laya
- **[REPO-29](https://github.com/tildaslashalef/nuclis/blob/v0.4.0/docs/engineering-log.md#repo-29--zig-0170-the-tree-migrated-and-its-features-adopted-2026-10-03-two-sessions)** Zig 0.17.0: the tree migrated and its features adopted
- **[REPO-30](https://github.com/tildaslashalef/nuclis/blob/v0.4.0/docs/engineering-log.md#repo-30--zig-017-holds-qwens-4k-speed-the-regression-repo-29-left-open-is-not-one-2026-10-03)** Zig 0.17 holds Qwen's 4K speed: the regression REPO-29 left open is not one

### Outside units

- **inference:** the CPU reference's matvec across every core, bit-identical
- **bench:** --capture records one decode step into a Metal .gputrace; make capture
- **metal-check:** CAPTURE= records a micro-benchmark case into a .gputrace
- **metal-check:** capture names without spaces; the counter export's naming
- **bench:** saved prefixes, verify cost at depth, and the speed loop's A/B driver
- **bench:** saved-prefix I/O in 1 GiB pieces; a failed save keeps the run
- **bench:** --kernel-stats prints each Metal pipeline's limits
- **inference:** few-query verify attention through the split pass
- **inference:** register-fragment small-batch matmul tile
- **inference:** few-row generic matrices of a small batch on the multi-row matvec
- **inference:** the fragment tile for Q3_K, IQ3_S, Q4_0, PQ2_0, PTQ1_0
- **inference:** IQ4_NL small batches on the fragment tile
- **nuclis:** bench --draft-p-min and the speculative matrix driver
- **inference:** discard a failed recording; derive the kernel table
- **nuclis:** name a support file passed to --file; skip other formats' folders
- **nuclis:** nuclis serve, the nuclis API: decisions over HTTP, TypeSafe's systemone call
- **nuclis:** nuclis serve batches decision requests that wait together into one pass
- **nuclis:** nuclis serve reads a serve section (port 8000), opens decide.model at start, logs a coloured line per request
- **inference:** qwen35 reads its shape from the file; hidden rows for decision heads
- **inference:** clef-flash, a second decision family: the joint schema head, images, decide and serve
- **nuclis:** serve.timeout and --timeout: a request waits up to 300 s for the GPU (clef-flash holds it for seconds)
- **nuclis:** /v1/models reports each decision model's family, whether it packs, and whether it reads images
- **nuclis:** serve drains queued decisions on Ctrl-C and exits cleanly
- **inference:** Q4_K matvec decodes nibbles through half magic numbers
- **inference:** Q5_K matvec takes the half magic-number decode
- **inference:** IQ4_XS matvec reads its code table from threadgroup memory
- **inference:** word-outer multi-row matvec for 2-3 row batches

<details><summary>All 132 commits</summary>

- [`0729012`](https://github.com/tildaslashalef/nuclis/commit/0729012a227d1cb904ae8dd2fe0a308b322a88e0) chore(release): v0.4.0
- [`7391ca8`](https://github.com/tildaslashalef/nuclis/commit/7391ca8c0c28a290cd9bac1efc10240b9c0d62ff) docs: AGNT-19 widened to token caching, turn-boundary snapshots in memory and on disk
- [`e405974`](https://github.com/tildaslashalef/nuclis/commit/e405974e1070ab6f63b2728ad043a7443d9a70aa) docs: the llm guide's speculation verdict after the decode-speed theme; AGNT-19 runs no inference gate tiers
- [`90b1936`](https://github.com/tildaslashalef/nuclis/commit/90b19368b62999384055764b077df46b08d64b77) docs: close KERN-23, weight streaming for one row and a few
- [`2f1fe98`](https://github.com/tildaslashalef/nuclis/commit/2f1fe98e1a6f9f44bc3aa23a338eaaec48fba54f) perf(inference): word-outer multi-row matvec for 2-3 row batches
- [`a692566`](https://github.com/tildaslashalef/nuclis/commit/a692566cf1e9d2294bf878c725dd1e21ad8e87e6) docs: KERN-23, the IQ4_XS matvec's counters, make verify on session 1's kernels
- [`6a0d321`](https://github.com/tildaslashalef/nuclis/commit/6a0d3211c4701752480d7b5328c1a9c6d689e29c) docs: KERN-23 session 1 hand-off
- [`07b6d5c`](https://github.com/tildaslashalef/nuclis/commit/07b6d5cb7725ab05553180cf71ef10793ab899f0) docs: KERN-23 ledger, the K-scale conversion fails the amended keep rule
- [`3b49578`](https://github.com/tildaslashalef/nuclis/commit/3b4957872ed4d8982efe90e4ebb6361e122fb55e) chore(speed): the keep rule takes a 1-2 % gain whose pairs agree
- [`96fefa0`](https://github.com/tildaslashalef/nuclis/commit/96fefa040ec148019d99cc20f96632396a4baf80) docs: KERN-23 ledger, the encode gap, K-scales below the keep rule, the step's profile
- [`c151ee2`](https://github.com/tildaslashalef/nuclis/commit/c151ee271997d51f038c3e843575881dc4e65647) perf(inference): IQ4_XS matvec reads its code table from threadgroup memory
- [`1137eb2`](https://github.com/tildaslashalef/nuclis/commit/1137eb23af25b75c6139adc618129637de72ac58) perf(inference): Q5_K matvec takes the half magic-number decode
- [`e36b77d`](https://github.com/tildaslashalef/nuclis/commit/e36b77dc8b59dec78b55e97bfa6b7c3d3ee35b86) perf(inference): Q4_K matvec decodes nibbles through half magic numbers
- [`8c47b56`](https://github.com/tildaslashalef/nuclis/commit/8c47b56bfb8ce45e6ac904ab20218002ae18f3cd) docs: close REPO-30, Zig 0.17 holds Qwen's 4K speed
- [`43c4840`](https://github.com/tildaslashalef/nuclis/commit/43c484057ec3316fafb844fcca64519453f3c0cd) docs: close REPO-29, the Zig 0.17.0 upgrade
- [`be7354b`](https://github.com/tildaslashalef/nuclis/commit/be7354be7f46b67a8cd34abcdfb8fdae2caee001) feat(nuclis): serve drains queued decisions on Ctrl-C and exits cleanly
- [`2ace031`](https://github.com/tildaslashalef/nuclis/commit/2ace031f54b655217acfde37cccf289c7f5a2969) refactor: adopt Zig 0.17's @divCeil and ArrayList.lastPtr; require 0.17 (REPO-29)
- [`61fa2ed`](https://github.com/tildaslashalef/nuclis/commit/61fa2edfed77d541cfe2a2f07aa6d9ecbcdae2f6) build: migrate the tree to Zig 0.17.0 (REPO-29 session 1)
- [`8f3e6ed`](https://github.com/tildaslashalef/nuclis/commit/8f3e6ed58769f90b5f15d6e8cb8bef94385e97c4) docs: REPO-29 checks 0.17 speed against the records instead of a 0.16 A/B
- [`be1f9af`](https://github.com/tildaslashalef/nuclis/commit/be1f9af2f209218905f0444ed92f5e99ec80c17c) docs: the project's Zig is the stable link, not current
- [`be75713`](https://github.com/tildaslashalef/nuclis/commit/be75713a5ed876219446f0d9090db3d8226a7d3f) docs: record the zigup-managed Zig install and its local std and langref
- [`155cab7`](https://github.com/tildaslashalef/nuclis/commit/155cab738f7947c9efb8d32d60cc3f9ec43487a9) docs: REPO-29 adopts Zig 0.17's features and measures 0.16 → 0.17 speed
- [`0be3124`](https://github.com/tildaslashalef/nuclis/commit/0be31240d42bf8f7dc278c18aa08882ea4caa737) docs: plan REPO-29, the Zig 0.17.0 upgrade, ahead of KERN-23
- [`fd09aa4`](https://github.com/tildaslashalef/nuclis/commit/fd09aa4cd378e5ac514f73cd3fc289be3c7d0413) feat(nuclis): /v1/models reports each decision model's family, whether it packs, and whether it reads images
- [`5dc6783`](https://github.com/tildaslashalef/nuclis/commit/5dc6783bc9fdc2d510893fb3752ab29be414fc8f) feat(nuclis): serve.timeout and --timeout: a request waits up to 300 s for the GPU (clef-flash holds it for seconds)
- [`f3cfd33`](https://github.com/tildaslashalef/nuclis/commit/f3cfd33c20a96dd95e512eaf9fbbe503146bb942) docs: the README presents clef-flash beside Laya
- [`cdeb5ab`](https://github.com/tildaslashalef/nuclis/commit/cdeb5ab07637e8f5df7c020c7e9c273319ce10ed) docs: TODO hand-off for KERN-23 after the cache cleanup: make speed-base first
- [`d5eedb2`](https://github.com/tildaslashalef/nuclis/commit/d5eedb2b0a34a730f5b2540a66a2d926f7a042db) docs: close MODL-34, clef-flash on both backends with images, in decide and serve
- [`5a9292c`](https://github.com/tildaslashalef/nuclis/commit/5a9292c5c1d0b63b5f69a16237ce7a18d8936ffc) feat(inference): clef-flash, a second decision family: the joint schema head, images, decide and serve
- [`b2bd22a`](https://github.com/tildaslashalef/nuclis/commit/b2bd22ae213b6a4c38a723453209c07c9bb498af) feat(inference): qwen35 reads its shape from the file; hidden rows for decision heads
- [`04c3d90`](https://github.com/tildaslashalef/nuclis/commit/04c3d908d5d0ea93bc8f89c30608ba316625eed0) feat(nuclis): nuclis serve reads a serve section (port 8000), opens decide.model at start, logs a coloured line per request
- [`275d0c8`](https://github.com/tildaslashalef/nuclis/commit/275d0c8c7870f8ad560af9d63fb25b4df2253775) docs: close APPS-19, the nuclis API reference for clients
- [`cd35ed8`](https://github.com/tildaslashalef/nuclis/commit/cd35ed8ffe5fac3ddf49862d6f4e39bd764d88bd) feat(nuclis): nuclis serve batches decision requests that wait together into one pass
- [`cb79cde`](https://github.com/tildaslashalef/nuclis/commit/cb79cde11fb44ee881d0fb49afd8b996edef64e7) feat(nuclis): nuclis serve, the nuclis API: decisions over HTTP, TypeSafe's systemone call
- [`8c5d2e6`](https://github.com/tildaslashalef/nuclis/commit/8c5d2e60fb61e68ddcc5fa43c1ff7cb986b7dc8d) refactor(nuclis): the decision wire format moves to src/decision/, shared by decide and the API
- [`83f8238`](https://github.com/tildaslashalef/nuclis/commit/83f8238a29caa3abcfa617588cb01e1142c6f9cf) docs(todo): MODL-34 checks only the new parts: no bf16 backbone, a head-only reference, a sanity set
- [`1f11510`](https://github.com/tildaslashalef/nuclis/commit/1f11510a55921fa35b9d0046bde140f2dda68899) docs(todo): MODL-34 pulls its files first, reads the qwen35 shape from the file, keeps the mmproj in scope
- [`612df55`](https://github.com/tildaslashalef/nuclis/commit/612df557a2faab27080e516ca3823298f733c55b) docs(todo): name the next step first
- [`c0daafc`](https://github.com/tildaslashalef/nuclis/commit/c0daafc067d27bb60e2285c3a6b4dc4ca6433cc1) docs(todo): APPS-19 builds the nuclis API layer, decisions its first service
- [`0d9fc59`](https://github.com/tildaslashalef/nuclis/commit/0d9fc599b35f20afab99137cb27dd99d00171f3f) docs(todo): APPS-19 runs without model gates
- [`59b246b`](https://github.com/tildaslashalef/nuclis/commit/59b246b43464d79b765c1964b7cd665da1c6a61e) docs(todo): APPS-19 writes a client-facing API reference, docs/reference/serve.md
- [`96527f9`](https://github.com/tildaslashalef/nuclis/commit/96527f9d9df6d4ae219cec49e456cf681d70b4ee) docs(todo): drop KERN-22
- [`396cc39`](https://github.com/tildaslashalef/nuclis/commit/396cc39ad44830217feeab055536d838a2a7c374) docs(todo): serve and clef-flash before KERN-23; KERN-22 deferred, AGNT-18 dropped
- [`d1a33e3`](https://github.com/tildaslashalef/nuclis/commit/d1a33e30ffdabfa813d5db3040a236026ce7186a) docs(todo): APPS-19 serves lean and batches across requests
- [`cd08618`](https://github.com/tildaslashalef/nuclis/commit/cd086186fb8677ddffa0d0bae074a1ef2d1663f9) docs(todo): queue APPS-19, nuclis serve, a local decision API
- [`7739069`](https://github.com/tildaslashalef/nuclis/commit/77390694c69852574662bbae82cb220bf9c8f463) docs(todo): the decide tool settles the default Laya checkpoint by a stated rule
- [`06a9bf9`](https://github.com/tildaslashalef/nuclis/commit/06a9bf9c553e52996593c69a5c817d49fc6cc8d4) docs(todo): the decide tool measures both Laya checkpoints; the default stays laya
- [`bcef958`](https://github.com/tildaslashalef/nuclis/commit/bcef9589dbe60cf2352f098aa0dc1c9fdc424bcf) docs(todo): base the decide tool's design on laya-multilingual and a labeled set
- [`383d274`](https://github.com/tildaslashalef/nuclis/commit/383d274f44e6181ce1df857b28e5e64a1ba10b63) docs: close REPO-27, the external review fixes
- [`c2c03a4`](https://github.com/tildaslashalef/nuclis/commit/c2c03a40acefd3335f2d704f6a4aa1cf26653494) refactor: name Qwen's mixer constants; comments stand without unit ids
- [`87f5876`](https://github.com/tildaslashalef/nuclis/commit/87f58764df53c0465d5e6a67726745eeeb8aeadd) test(inference): corrupted GGUF and safetensors files never panic or leak
- [`4ec3e39`](https://github.com/tildaslashalef/nuclis/commit/4ec3e3922be1955ed4e7fa42f212671a5e9791de) fix(nuclis): name a support file passed to --file; skip other formats' folders
- [`878be0e`](https://github.com/tildaslashalef/nuclis/commit/878be0e22fe4e3186284309f4e3f67acb5f69563) fix(inference): discard a failed recording; derive the kernel table
- [`6099030`](https://github.com/tildaslashalef/nuclis/commit/6099030fd003c4cca442eb30deb838b4a81808df) docs(todo): REPO-27, the verified review fixes, ahead of KERN-23
- [`7120a5a`](https://github.com/tildaslashalef/nuclis/commit/7120a5a543a188d36e9674c7de01f89f95e2a47b) docs(todo): KERN-23 re-scoped to the verify's multi-row body
- [`2479d62`](https://github.com/tildaslashalef/nuclis/commit/2479d62a254da1e05cdc20f7841e2336565e72c3) docs(readme): the speculative defaults and their rates; REPO-26
- [`e06c971`](https://github.com/tildaslashalef/nuclis/commit/e06c971c26cff410575a2980ea4f6d9a4459beff) feat(nuclis): speculation on by default where it pays; close ENGN-20
- [`c7b6d2b`](https://github.com/tildaslashalef/nuclis/commit/c7b6d2b642b9009d013f27e51a3416fb5d233984) feat(nuclis): bench --draft-p-min and the speculative matrix driver
- [`c582da4`](https://github.com/tildaslashalef/nuclis/commit/c582da461e99c150f412c93dc34f5f363fcc42c9) docs(todo): ENGN-20's code map, matrix, and verdict rule
- [`fec1ec3`](https://github.com/tildaslashalef/nuclis/commit/fec1ec3696a805947a12b0e4a9c676b4aa969fdf) docs(todo): re-order the decode-speed units: ENGN-20, KERN-23, KERN-22
- [`4a2e513`](https://github.com/tildaslashalef/nuclis/commit/4a2e5135a381f3bfd323c0688ea8b69b0c613c6e) perf(inference): DeltaNet verify by replay tape; close ENGN-19
- [`bbc887b`](https://github.com/tildaslashalef/nuclis/commit/bbc887b3ad09eb528f44132e965a4e1a2a9e96ec) docs(todo): ENGN-19 session 1 hand-off, tape implemented, gates pending
- [`e35381b`](https://github.com/tildaslashalef/nuclis/commit/e35381b10e712fe5fa53858e0b86b8327cb8e5b3) docs(todo): ENGN-19's code map and first session
- [`7f1a287`](https://github.com/tildaslashalef/nuclis/commit/7f1a287dadc101c1aff92bde0714e9c54358bf57) docs(log): the register-fragment verify matmul; close KERN-24
- [`88ecb25`](https://github.com/tildaslashalef/nuclis/commit/88ecb2589ac39d2c953c247125176847bfefacd1) docs(todo): where KERN-24 stands
- [`280c111`](https://github.com/tildaslashalef/nuclis/commit/280c1116fb3815de6ee2500ec9d4768bb72de33c) docs(reference): the fragment tile's counters; KERN-24 session 1 hand-off
- [`5644036`](https://github.com/tildaslashalef/nuclis/commit/56440364a4ce89183a9ad957fcbd3b0fc8f51fcc) test(metal-check): capture labels on the fragment-tile sweep; KERN-24 ledger
- [`2a6b1ac`](https://github.com/tildaslashalef/nuclis/commit/2a6b1ac20c5a685c42a378d0a1a0eed796c0e76c) perf(inference): IQ4_NL small batches on the fragment tile
- [`3ee3998`](https://github.com/tildaslashalef/nuclis/commit/3ee3998bce5b77770a9674cb4aec9953098e7490) perf(inference): the fragment tile for Q3_K, IQ3_S, Q4_0, PQ2_0, PTQ1_0
- [`2c34960`](https://github.com/tildaslashalef/nuclis/commit/2c349602e7edef553f03586382e39c7660fcaaa5) perf(inference): few-row generic matrices of a small batch on the multi-row matvec
- [`e5d0129`](https://github.com/tildaslashalef/nuclis/commit/e5d0129724551fdc2b7af3ce33e8a792e1237829) perf(inference): register-fragment small-batch matmul tile
- [`d39e3d4`](https://github.com/tildaslashalef/nuclis/commit/d39e3d49f36f3a4d8b90660d1054998e350d2cc0) docs(log): Gemma and Muse verify costs at depth; close KERN-21
- [`d1f0a1b`](https://github.com/tildaslashalef/nuclis/commit/d1f0a1bae8b73f4766122090ce96853e3f855d6c) test(generation): verify rows against stepped decode at depth
- [`e460a75`](https://github.com/tildaslashalef/nuclis/commit/e460a753f5257a714fa18b5b6d5a1312839d24a5) docs(reference): the multi-row matvec spills at 8 rows; close KERN-20
- [`96d07ba`](https://github.com/tildaslashalef/nuclis/commit/96d07ba06bac6dc06c9200a73cb0e760a1e9cedd) perf(inference): few-query verify attention through the split pass
- [`9ecea66`](https://github.com/tildaslashalef/nuclis/commit/9ecea66e3f3f744e2694650ce644ac4fb4bcb8ec) feat(bench): --kernel-stats prints each Metal pipeline's limits
- [`765a195`](https://github.com/tildaslashalef/nuclis/commit/765a195445cebe1f24c2607caced845ec91436a7) docs(todo): cite the primed-prefix cost
- [`38eb676`](https://github.com/tildaslashalef/nuclis/commit/38eb676d355b4617d697cfd97e858f7d37377231) docs(todo): queue AGNT-19, saved prefixes for the agent across processes
- [`102cdf3`](https://github.com/tildaslashalef/nuclis/commit/102cdf3e9962ed076acc93e525114dff6e0ee627) docs(reference): the verify's attention latency-bound on an empty GPU, its matmul tile issue-bound
- [`65f8c19`](https://github.com/tildaslashalef/nuclis/commit/65f8c19d53f0e1580de83a14021a33ebb85bdafe) docs(todo): the speed loop's base is 4dc7c70
- [`4dc7c70`](https://github.com/tildaslashalef/nuclis/commit/4dc7c705d45b6c8b82d1717ecb6f300247fc29e7) docs(bench): the decode-speed baseline; close ENGN-18
- [`beccae1`](https://github.com/tildaslashalef/nuclis/commit/beccae1fa3a4bca9645c98f32547f99e5f6a6b14) fix(bench): saved-prefix I/O in 1 GiB pieces; a failed save keeps the run
- [`236e900`](https://github.com/tildaslashalef/nuclis/commit/236e900a7b773f016d6cf30812f69d6542e31978) feat(bench): saved prefixes, verify cost at depth, and the speed loop's A/B driver
- [`343ae3c`](https://github.com/tildaslashalef/nuclis/commit/343ae3cecf1bb9a096a2c3bddfe35dca6d1a7e50) docs(development): who captures, exports, renames, and reads
- [`3e2c13e`](https://github.com/tildaslashalef/nuclis/commit/3e2c13ece3ec003f7759c679dd03f22152206c18) feat(metal-check): capture names without spaces; the counter export's naming
- [`19e192b`](https://github.com/tildaslashalef/nuclis/commit/19e192b0304b2878b148e2ad4ae9264367b9c30f) docs(reference): the last-level miss rate at full clocks
- [`defeb9e`](https://github.com/tildaslashalef/nuclis/commit/defeb9e38935d8e476af25301df2aa3920c766ae) docs(reference): the Q4_K matvec's integer-pipe limit confirmed at full clocks
- [`237a325`](https://github.com/tildaslashalef/nuclis/commit/237a325e6dd17d3bbb23a285e4851e68cdf0c83a) docs(reference): apple-gpu.md, the Q4_K matvec issue-bound on the integer pipe
- [`c75f644`](https://github.com/tildaslashalef/nuclis/commit/c75f644aaba2ecb59d769dfcd31af10a463b99ba) docs(development): capture profiling needs Xcode's Metal Toolchain
- [`ecdc9dc`](https://github.com/tildaslashalef/nuclis/commit/ecdc9dc671eae6eec920af8143915e59f4b21140) docs(todo): where we are after the capture tooling
- [`5ea6dc3`](https://github.com/tildaslashalef/nuclis/commit/5ea6dc356a87eb2529715d684238b897260aaa27) docs(development): Xcode 27's MCP tools read no GPU capture
- [`60f1e8b`](https://github.com/tildaslashalef/nuclis/commit/60f1e8b3164d7b7d604985e76a35501f752c68dd) feat(metal-check): CAPTURE= records a micro-benchmark case into a .gputrace
- [`28a2850`](https://github.com/tildaslashalef/nuclis/commit/28a2850f6814ad8dcbe205dd80bac826da15647e) feat(bench): --capture records one decode step into a Metal .gputrace; make capture
- [`c3004ff`](https://github.com/tildaslashalef/nuclis/commit/c3004ff0a534d81af4b1a7ac9bc36929273ee057) docs(todo): the decode-speed theme, eight units behind a measured keep rule
- [`d66c371`](https://github.com/tildaslashalef/nuclis/commit/d66c37156ff4df9bcac982424aaf6664edffb97e) docs(adr): propose the Qwen small-batch verifier; close REPO-24
- [`e174a3e`](https://github.com/tildaslashalef/nuclis/commit/e174a3ebd81f146f5272eba8db70a415d9a42678) perf(inference): close KERN-19, the CPU tier in 22 minutes, bit-identical
- [`357ded5`](https://github.com/tildaslashalef/nuclis/commit/357ded562ad660bd75d6576b2750fb3612559819) build(gates): skip a gate whose model is absent, strict for releases; close REPO-25
- [`41f538a`](https://github.com/tildaslashalef/nuclis/commit/41f538aaa3ec05e28f7e6a5d7c5f7db161b2bd96) perf(inference): the CPU reference's matvec across every core, bit-identical
- [`48b55d8`](https://github.com/tildaslashalef/nuclis/commit/48b55d810856b01a62c2adeb03c24e5bfdb7fd82) docs(readme): a shorter Laya section led by its read speed
- [`b5c872c`](https://github.com/tildaslashalef/nuclis/commit/b5c872c40e109110dc83d7cccd93ca4013e04cf3) docs(todo): KERN-19 base and design from the CPU backend's call paths
- [`f264d67`](https://github.com/tildaslashalef/nuclis/commit/f264d6722d17cc43ccbc37d877e4e36e2d11cff2) docs(readme): the Laya experiment and its results; close REPO-23
- [`5b185ad`](https://github.com/tildaslashalef/nuclis/commit/5b185addc4bf1558abf3d5a183eec57d8dbfb9d7) build: durable reference state in .reference/, a disposable .zig-cache; close REPO-22
- [`b703a67`](https://github.com/tildaslashalef/nuclis/commit/b703a67fad1f293118b4e81fb511612b1f2836f6) refactor(scripts): a generated agent playground, typed and formatted Python; close REPO-21
- [`ede9134`](https://github.com/tildaslashalef/nuclis/commit/ede9134d7e5d1a78cc54633a10782402a664d924) docs(todo): REPO-21, a generated playground and typed scripts, ahead of AGNT-18
- [`0942463`](https://github.com/tildaslashalef/nuclis/commit/0942463d72f1905aa915060007683b26238bffec) feat(inference): Laya multilingual, the Metaspace tokenizer; close MODL-33
- [`b65680c`](https://github.com/tildaslashalef/nuclis/commit/b65680c57ddae4f46a763855f207f7b6a651dcbd) docs(todo): MODL-33, Laya multilingual, ahead of AGNT-18
- [`01b65fe`](https://github.com/tildaslashalef/nuclis/commit/01b65fefd2ed5730a95fcb5347cb60130978cfa0) feat(inference): Laya on Metal, packed batches over sequence bounds; close MODL-31
- [`9809db4`](https://github.com/tildaslashalef/nuclis/commit/9809db41d5d663046b9659bfc63da1bf588c5207) docs(todo): MODL-31 design from the Metal backend's facts
- [`9758c56`](https://github.com/tildaslashalef/nuclis/commit/9758c56b6fdead1d70d458ae692503c176d832a9) docs(todo): Laya first (MODL-31, AGNT-18), then KERN-19; MODL-31 base
- [`26eccf1`](https://github.com/tildaslashalef/nuclis/commit/26eccf194e9440023147140c3b3d5cae2f92c34b) docs(todo): KERN-19 base
- [`d46e907`](https://github.com/tildaslashalef/nuclis/commit/d46e9074df68f11c2a79807b797139634670445e) chore(gates): verify-auto runs one model at a time; close REPO-20
- [`b06aebc`](https://github.com/tildaslashalef/nuclis/commit/b06aebc039ead37431bb080285b6dfdbd321817a) feat(gates): make verify-auto picks the checks and gates a change needs (REPO-20 session 2, part)
- [`0ecbc4f`](https://github.com/tildaslashalef/nuclis/commit/0ecbc4fe4af62cc57451a27446b6d4549df1147e) feat(gates): a fast tier from code paths and a release tier (REPO-20 session 2, part)
- [`940a1e4`](https://github.com/tildaslashalef/nuclis/commit/940a1e44512ae780ef9aeafddd1a49a22092a94a) docs(todo): REPO-20 gate set approved, sessions swapped
- [`f5982c4`](https://github.com/tildaslashalef/nuclis/commit/f5982c4415a19ffb9c31be52cd2fe576fa10c1d7) chore(gates): time each gate's phases, the coverage matrix (REPO-20 session 1, part)
- [`acce06a`](https://github.com/tildaslashalef/nuclis/commit/acce06a9f7ded32d6572bb38e84073a80d56c676) docs: plan fast verification: gates from code paths, a threaded reference (REPO-20)
- [`14ace19`](https://github.com/tildaslashalef/nuclis/commit/14ace195524c787169628b03e141e6de27ea41a5) chore(gates): retire the Gemma 4 12B K-quant file's gates (REPO-19)
- [`678417a`](https://github.com/tildaslashalef/nuclis/commit/678417a6a8f0ec2ea94395ca716a0502eab1f02f) feat(nuclis): Jev's answer fields, the decision path documented; close MODL-30
- [`792c47f`](https://github.com/tildaslashalef/nuclis/commit/792c47f422b7dcdc3deb998dc17de5d2dc4d6c93) feat(nuclis): the laya decision catalogue entry, decide.model, registry kind decision (MODL-30 session 3, part)
- [`f8a3d5b`](https://github.com/tildaslashalef/nuclis/commit/f8a3d5b587f804426abeadb846feefd7a8541e5f) fix(huggingface): keep the Hub listing's digests in ReleaseFast builds (MODL-32)
- [`a5af02c`](https://github.com/tildaslashalef/nuclis/commit/a5af02c30bc2342d6688847db99cd933cd6825a1) feat(nuclis): nuclis decide over Laya's profile and the Decider (MODL-30 session 3, part)
- [`0852fb1`](https://github.com/tildaslashalef/nuclis/commit/0852fb126a5bc385492af53c54e8235821e24f4a) feat(inference): Laya's encoder and decision head on the CPU (MODL-30 session 2)
- [`7763099`](https://github.com/tildaslashalef/nuclis/commit/7763099a437f378807fa9463245276d8aebc7c2f) feat(inference): Hugging Face tokenizer.json with NFC, and the Laya oracle (MODL-30 session 1)
- [`1eac855`](https://github.com/tildaslashalef/nuclis/commit/1eac855edc9cf29f0af77a69cd5c402a9d2d9d6d) feat(nuclis): shell completion answered by the binary; make install (APPS-18)
- [`17b5b8f`](https://github.com/tildaslashalef/nuclis/commit/17b5b8feffc4338ed45ec7d0227ee6832c9b38a9) docs: plan shell completion and make install (APPS-18)
- [`6241b85`](https://github.com/tildaslashalef/nuclis/commit/6241b8596a1ffee9d5699026d294c1774fdb70f6) fix(tui): exit keeps the transcript; reset margins without homing the cursor (TERM-13)
- [`60f53fb`](https://github.com/tildaslashalef/nuclis/commit/60f53fbc4bd9802b67eee05ec302097ab804d55b) docs: remove docs/research, carry its conclusion into the plan (REPO-18)
- [`f16edc4`](https://github.com/tildaslashalef/nuclis/commit/f16edc41899c756b41e989e599c039678ca7f98c) docs: plan Laya end to end: CPU, Metal, the agent tool (MODL-30, MODL-31, AGNT-18)
- [`ee3dffd`](https://github.com/tildaslashalef/nuclis/commit/ee3dffd1bd970980a195f4a9bea311ba26ae7c61) feat(inference): a generic safetensors loader, inspect on it (MODL-29)
- [`bc703d9`](https://github.com/tildaslashalef/nuclis/commit/bc703d915cc88b260ac3c18634a6f8d54bd7d5f4) feat(nuclis): pull safetensors sets with their support files, pinned like GGUF (MODL-28)
- [`45dd8e5`](https://github.com/tildaslashalef/nuclis/commit/45dd8e55ee158895a67043b21ccf2d5ded4fa564) docs: plan safetensors acquisition and loader (MODL-28, MODL-29)
- [`c2f3cf4`](https://github.com/tildaslashalef/nuclis/commit/c2f3cf4b97c281201609ef139e4ce97af8e43a2b) chore: begin 0.4.0-dev

</details>

**Full diff:** [v0.3.0...v0.4.0](https://github.com/tildaslashalef/nuclis/compare/v0.3.0...v0.4.0)

## [v0.3.0] - 2026-09-27

Models see images: Qwen3.8, Gemma 4, Muse Glimmer, and Bonsai 2 read them through their projectors on both the CPU and Metal, and the agent takes them by drop or `/image`. Gemma 4 E4B QAT joins, `nuclis eval` measures perplexity against the reference per family, and the agent's system prompt is rebuilt as sections measured on a playground task list. The terminal UI gets a repaint tick, banded diffs, and resize handling.

### Models

- **[MODL-21](https://github.com/tildaslashalef/nuclis/blob/v0.3.0/docs/engineering-log.md#modl-21--the-vision-contract-image-input-and-the-qwen38-projector-on-both-executors-2026-09-22)** The vision contract, image input, and the Qwen3.8 projector on both executors
- **[MODL-22](https://github.com/tildaslashalef/nuclis/blob/v0.3.0/docs/engineering-log.md#modl-22--gemma-4-vision-the-unified-embedder-12b-and-the-siglip-encoder-26b-a4b-on-both-executors-bidirectional-image-spans-2026-09-23)** Gemma 4 vision: the unified embedder (12B) and the SigLIP encoder (26B-A4B) on both executors; bidirectional image spans
- **[MODL-23](https://github.com/tildaslashalef/nuclis/blob/v0.3.0/docs/engineering-log.md#modl-23--muse-glimmers-windowed-vision-encoder-on-both-executors-the-image-token-cap-2026-09-23)** Muse Glimmer's windowed vision encoder on both executors; the image token cap
- **[MODL-24](https://github.com/tildaslashalef/nuclis/blob/v0.3.0/docs/engineering-log.md#modl-24--decode-after-an-image-the-metal-steps-cache-row-and-rotary-position-separated-the-vision-gate-compares-decode-with-one-prefill-2026-09-23)** Decode after an image: the Metal step's cache row and rotary position separated; the vision gate compares decode with one prefill
- **[MODL-25](https://github.com/tildaslashalef/nuclis/blob/v0.3.0/docs/engineering-log.md#modl-25--bonsai-2s-projector-through-the-qwen3-vl-adapter-q8_0-and-f16-weights-2026-09-23)** Bonsai 2's projector through the Qwen3-VL adapter: Q8_0 and F16 weights
- **[MODL-26](https://github.com/tildaslashalef/nuclis/blob/v0.3.0/docs/engineering-log.md#modl-26--gemma-4s-metal-verify-rows-carry-the-final-soft-cap-2026-09-24)** Gemma 4's Metal `verify` rows carry the final soft-cap
- **[MODL-27](https://github.com/tildaslashalef/nuclis/blob/v0.3.0/docs/engineering-log.md#modl-27--gemma-4-e4b-qat-per-layer-embeddings-shared-kv-vision-draft-head-2026-09-27)** Gemma 4 E4B QAT: per-layer embeddings, shared KV, vision, draft head

### Agent

- **[AGNT-11](https://github.com/tildaslashalef/nuclis/blob/v0.3.0/docs/engineering-log.md#agnt-11--a-truncated-tool-call-no-longer-bricks-the-session-reopened-thought-channels-copied-bracket-pieces-2026-09-17)** A truncated tool call no longer bricks the session; reopened thought channels; copied bracket pieces _(follow-up)_
- **[AGNT-12](https://github.com/tildaslashalef/nuclis/blob/v0.3.0/docs/engineering-log.md#agnt-12--an-empty-tool-result-no-longer-aborts-the-turn-2026-09-18)** An empty tool result no longer aborts the turn _(follow-up)_
- **[AGNT-13](https://github.com/tildaslashalef/nuclis/blob/v0.3.0/docs/engineering-log.md#agnt-13--the-system-prompt-as-sections-measured-the-playground-task-list-the-guidelines-that-changed-behaviour-the-instructions-file-2026-09-22)** The system prompt as sections, measured: the playground task list, the guidelines that changed behaviour, the instructions file
- **[AGNT-14](https://github.com/tildaslashalef/nuclis/blob/v0.3.0/docs/engineering-log.md#agnt-14--three-measured-fixes-the-qwen-decoder-keeps-a-values-trailing-newline-read_file-serves-the-first-mib-enter-steers-a-running-turn-2026-09-22)** Three measured fixes: the Qwen decoder keeps a value's trailing newline, `read_file` serves the first MiB, Enter steers a running turn
- **[AGNT-15](https://github.com/tildaslashalef/nuclis/blob/v0.3.0/docs/engineering-log.md#agnt-15--images-and-text-files-in-the-chat-drop-image-the-chips-the-projector-turn-the-detail-row-and-preview-sessions-2026-09-23)** Images and text files in the chat: drop, `/image`, the chips, the projector turn, the detail row and preview, sessions
- **[AGNT-16](https://github.com/tildaslashalef/nuclis/blob/v0.3.0/docs/engineering-log.md#agnt-16--agnt-14s-follow-ups-the-muse-value-contract-pinned-and-measured-steering-that-restarts-a-reasoning-only-step-the-guessed-path-guideline-measured-and-dropped-2026-09-23)** AGNT-14's follow-ups: the Muse value contract pinned and measured, steering that restarts a reasoning-only step, the guessed-path guideline measured and dropped
- **[AGNT-17](https://github.com/tildaslashalef/nuclis/blob/v0.3.0/docs/engineering-log.md#agnt-17--the-playground-fixes-blank-output-said-the-paged-file-rule-tool-output-in-the-transcript-a-reasoning-budget-2026-09-26)** The playground fixes: blank output said, the paged-file rule, tool output in the transcript, a reasoning budget

### Application

- **[APPS-14](https://github.com/tildaslashalef/nuclis/blob/v0.3.0/docs/engineering-log.md#apps-14--teacher-forced-eval-perplexity-against-the-references-per-token-run-the-all-rows-prefill-a-gate-per-family-2026-09-24)** Teacher-forced `eval`: perplexity against the reference's per-token run, the all-rows prefill, a gate per family
- **[APPS-16](https://github.com/tildaslashalef/nuclis/blob/v0.3.0/docs/engineering-log.md#apps-16--output-budget-default-4096-cap-16384-2026-09-21)** Output budget: default 4096, cap 16384
- **[APPS-17](https://github.com/tildaslashalef/nuclis/blob/v0.3.0/docs/engineering-log.md#apps-17--the-catalogue-at-five-entries-the-registry-in-name-order-2026-09-26)** The catalogue at five entries; the registry in name order

### Terminal

- **[TERM-10](https://github.com/tildaslashalef/nuclis/blob/v0.3.0/docs/engineering-log.md#term-10--chat-polish-the-repaint-tick-the-frame-the-operation-rows-the-banded-diff-and-the-editors-shell-and-keys-2026-09-21--2026-09-22)** Chat polish: the repaint tick, the frame, the operation rows, the banded diff, and the editor's shell and keys
- **[TERM-11](https://github.com/tildaslashalef/nuclis/blob/v0.3.0/docs/engineering-log.md#term-11--the-region-re-anchors-on-a-resize-from-the-terminals-cursor-report-the-preview-fits-above-the-region-and-is-off-under-tmux-2026-09-23)** The region re-anchors on a resize from the terminal's cursor report; the preview fits above the region and is off under tmux
- **[TERM-12](https://github.com/tildaslashalef/nuclis/blob/v0.3.0/docs/engineering-log.md#term-12--a-spinner-row-while-the-projector-loads-and-images-encode-2026-09-23)** A spinner row while the projector loads and images encode
- **TERM-13** Exit keeps the transcript _(in progress)_

### Repository

- **[REPO-12](https://github.com/tildaslashalef/nuclis/blob/v0.3.0/docs/engineering-log.md#repo-12--the-architecture-guide-follows-the-kv-cache-end-to-end-the-inference-guide-rewritten-as-one-narrative-2026-09-22)** The architecture guide follows the KV cache end to end; the inference guide rewritten as one narrative
- **[REPO-13](https://github.com/tildaslashalef/nuclis/blob/v0.3.0/docs/engineering-log.md#repo-13--one-specification-specmd-rewritten-as-a-technical-specification-with-the-agent-spec-merged-in-2026-09-22)** One specification: `spec.md` rewritten as a technical specification with the agent spec merged in
- **[REPO-14](https://github.com/tildaslashalef/nuclis/blob/v0.3.0/docs/engineering-log.md#repo-14--the-screenshot-harness-is-the-validation-step-for-surface-changes-2026-09-23)** The screenshot harness is the validation step for surface changes
- **[REPO-15](https://github.com/tildaslashalef/nuclis/blob/v0.3.0/docs/engineering-log.md#repo-15--the-cpu-tier-leaves-the-unit-routine-it-runs-when-a-unit-changes-what-the-cpu-reference-computes-and-before-a-release-2026-09-23)** The CPU tier leaves the unit routine: it runs when a unit changes what the CPU reference computes, and before a release
- **[REPO-16](https://github.com/tildaslashalef/nuclis/blob/v0.3.0/docs/engineering-log.md#repo-16--scriptsnuclis_mem_usagepy-a-running-processs-memory-split-into-gpu-and-cpu-2026-09-26)** `scripts/nuclis_mem_usage.py`: a running process's memory split into GPU and CPU
- **[REPO-17](https://github.com/tildaslashalef/nuclis/blob/v0.3.0/docs/engineering-log.md#repo-17--readme-rewritten-with-a-recorded-agent-session-2026-09-27)** README rewritten, with a recorded agent session

### Outside units

- **repo:** the screenshot harness runs the agent in a Ghostty window of its own (--direct)

<details><summary>All 62 commits</summary>

- [`056122e`](https://github.com/tildaslashalef/nuclis/commit/056122e00ee2117ef2546a6c701ad2be2176bec3) chore(release): v0.3.0
- [`84f1530`](https://github.com/tildaslashalef/nuclis/commit/84f153039a92f910119d784ff5c0b20b97a451ae) docs: README rewritten with a recorded agent session; close MODL-27 (REPO-17)
- [`bd96cbb`](https://github.com/tildaslashalef/nuclis/commit/bd96cbb544e64a68039852e167271c2679b92ede) feat(nuclis): the catalogue at five entries, the registry in name order (APPS-17)
- [`4f2f977`](https://github.com/tildaslashalef/nuclis/commit/4f2f977243e4bcc18b5c6caa4b0fc4e092a8733b) feat(inference): Gemma 4 E4B QAT end to end, checks pending (MODL-27)
- [`d0ecb52`](https://github.com/tildaslashalef/nuclis/commit/d0ecb52498c2faaee7d7a1253ae9656c132d053b) feat(agent): blank bash output said, the paged-file rule, Ctrl-O's output view, a reasoning budget at low (AGNT-17)
- [`a3253cb`](https://github.com/tildaslashalef/nuclis/commit/a3253cb72bab460e034f445cfe2faaef95bb7b0c) chore(scripts): nuclis_mem_usage.py splits a running process's memory into GPU and CPU (REPO-16)
- [`e9981d8`](https://github.com/tildaslashalef/nuclis/commit/e9981d88b288209c4fc8b7a29e2e00978d8de7c0) test(eval): a long-context perplexity tier; Gemma 4 12B passes at 4K, the 26B-A4B's -1.79 % is open (MODL-26 follow-up)
- [`d61685f`](https://github.com/tildaslashalef/nuclis/commit/d61685f30da74221d090a7e111a43ade026725e6) perf(tokenizer): queued BPE merge for long pieces; Gemma lines no longer hit the work bound (MODL-26 follow-up)
- [`3f92395`](https://github.com/tildaslashalef/nuclis/commit/3f923951c667ea3b5dee4ff5d0da635b5d4ccf0f) feat(eval): teacher-forced perplexity against the reference, the all-rows prefill, a gate per family (APPS-14)
- [`eb5e8dc`](https://github.com/tildaslashalef/nuclis/commit/eb5e8dca41cafdcfd07cf673cd354a3ea526a728) fix(gemma4): Metal verify rows carry the final soft-cap (MODL-26)
- [`32ff1fe`](https://github.com/tildaslashalef/nuclis/commit/32ff1fe5cf8febce903ce6fe4725245c6eacc851) docs(repo): the CPU tier runs only when a unit changes what the CPU reference computes, and before a release (REPO-15)
- [`9b57353`](https://github.com/tildaslashalef/nuclis/commit/9b5735399373292975e9a66457d2f6262a48a10a) feat(vision): Bonsai 2's Q8_0 projector through the Qwen3-VL adapter (MODL-25); TERM-13 dropped
- [`e13b6e3`](https://github.com/tildaslashalef/nuclis/commit/e13b6e39e15fb4c3c40a36a322071e79ff4a2014) feat(vision): close MODL-23: Muse Glimmer's vision record, the image token cap, the docs
- [`81e5394`](https://github.com/tildaslashalef/nuclis/commit/81e53949d6fdf49001be11d99cc75e1c5a6b233d) perf(vision): Muse's windows on the chunk attention kernel (encode 67 to 47 s at 16,320 patches); generate reports image_milliseconds; the check tool profiles the projector (MODL-23)
- [`626ba27`](https://github.com/tildaslashalef/nuclis/commit/626ba271f520bed5b3f13397e5fc1bd69b5be29f) feat(vision): Muse Glimmer's windowed projector on both executors and image spans in its language model (MODL-23)
- [`3d23b49`](https://github.com/tildaslashalef/nuclis/commit/3d23b498e99c4094eb9e163c7f5a515f97c4cdfa) feat(config): generation.image_max_tokens, auto by default, with a per-model override and --image-max-tokens (MODL-23)
- [`a73029e`](https://github.com/tildaslashalef/nuclis/commit/a73029ebf40ca02e568b5b1862b7366ee33c5a64) docs(todo): MODL-23 takes generation.image_max_tokens (auto by default, a per-model override and a flag)
- [`e06b551`](https://github.com/tildaslashalef/nuclis/commit/e06b551b561c4e470eaeb55f71ad374b07e8b8a4) docs(todo): MODL-23 facts from the reference and the Muse projector file; the pinned synthetic fixture; the oracle takes --n-ctx and --vision-flash
- [`b9267e1`](https://github.com/tildaslashalef/nuclis/commit/b9267e13c235900aa29443f3f24dd59b30e4eb5c) feat(vision): close MODL-22: the Gemma 4 vision record, the chunk buffers grown all-or-nothing, the docs
- [`3266a3d`](https://github.com/tildaslashalef/nuclis/commit/3266a3d52b5b0a8e695a99411b6cb79163aff065) feat(vision): Gemma 4's projectors on both executors and bidirectional image spans in its language model (MODL-22)
- [`f253d13`](https://github.com/tildaslashalef/nuclis/commit/f253d1343dbdec82d75c03f8a958d8c9cff15f09) docs(todo): MODL-22 facts from the reference and the Gemma 4 projector files; the oracle takes image token bounds
- [`990b708`](https://github.com/tildaslashalef/nuclis/commit/990b708470b5af7cbba443363337476c78a82f25) feat(agent): a steer during a reasoning-only step restarts it; close AGNT-16, draft TERM-13
- [`dbdf644`](https://github.com/tildaslashalef/nuclis/commit/dbdf644edea49b784665dd12683d7f14f479ef59) feat(repo): the screenshot harness runs the agent in a Ghostty window of its own (--direct)
- [`d8ab61b`](https://github.com/tildaslashalef/nuclis/commit/d8ab61bfc442d7e75bb8b640d74d2d51a2b770e4) fix(vision): decode after an image writes and reads the right cache rows; the vision gate compares decode with one prefill; a spinner while images encode (MODL-24, TERM-12)
- [`d90d252`](https://github.com/tildaslashalef/nuclis/commit/d90d252aee5359e3ff9c89febdf09e5d1caa3f9a) fix(tui): re-anchor the live region on a resize from the cursor report; the preview fits above the region and is off under tmux (TERM-11)
- [`19e88c8`](https://github.com/tildaslashalef/nuclis/commit/19e88c8d0043ab7799fd9a34edb4743e06f6c9b9) docs(repo): the screenshot harness is the validation step for surface changes (REPO-14)
- [`4f6e1ac`](https://github.com/tildaslashalef/nuclis/commit/4f6e1ac2380ce39ae8f89c6569897e3c59c48223) docs(plan): close AGNT-15 — images and text files in the chat, the log entry, the spec and reference updates
- [`3f2e4dd`](https://github.com/tildaslashalef/nuclis/commit/3f2e4dd4a3e80869d105970178208051d98a26ee) fix(agent): the typed-path scan skips command lines; the harness gains a bracketed paste step and tmux passthrough (AGNT-15)
- [`b9c34c7`](https://github.com/tildaslashalef/nuclis/commit/b9c34c7af520492534561ce6261e90438de22173) feat(agent): images and text files in the chat — drop, /image, the chips, the projector turn, the detail row, the preview, sessions (AGNT-15)
- [`737cb11`](https://github.com/tildaslashalef/nuclis/commit/737cb11472600c38bd79ecd9fb795059b6829654) docs(plan): AGNT-15 opened — text-file chips, feature ownership, the speculation rule, the preview's terminal test
- [`cac35c3`](https://github.com/tildaslashalef/nuclis/commit/cac35c33190627e7568b80f805dbe6171de92a7f) docs(plan): close MODL-21 — the Qwen3.8 vision projector, verified on both executors
- [`cd57b72`](https://github.com/tildaslashalef/nuclis/commit/cd57b72f9ab7ab192c189a560e0c308ae8ff4882) feat(vision): image spans through the Qwen3.8 language model, generate --image, the vision gates (MODL-21)
- [`e25c82e`](https://github.com/tildaslashalef/nuclis/commit/e25c82e9c74eb0359ec0fa7bc3e4e09ac528c31a) feat(vision): session rope triples and the CPU multi-axis RoPE for image spans (MODL-21)
- [`147cf81`](https://github.com/tildaslashalef/nuclis/commit/147cf8186685d93721fe07852d5facf4bc2743b1) feat(vision): the Qwen3-VL projector on both executors, the image decoders, exact preprocessing, and the oracle harness (MODL-21, first half)
- [`d7acc31`](https://github.com/tildaslashalef/nuclis/commit/d7acc31f88c8e15f2cac379bcaae4c39960f03d4) docs(plan): AGNT-16 ordered: the Muse value contract measured, steering that interrupts reasoning, the guessed-path guideline
- [`816ca0c`](https://github.com/tildaslashalef/nuclis/commit/816ca0c4d0279ff0544f8c916e41a55700fe1fa3) feat(agent): the Qwen decoder keeps a value's trailing newline, read_file serves the first MiB, Enter steers a running turn (AGNT-14)
- [`2cfb5a3`](https://github.com/tildaslashalef/nuclis/commit/2cfb5a3c47b3973641ffb94af1d5ab8053ee9e72) docs(plan): AGNT-12 dropped, its steering folded into AGNT-14 with the two measured tool fixes; the images unit renumbered AGNT-15
- [`0be2323`](https://github.com/tildaslashalef/nuclis/commit/0be2323e2e9623fedd9afe7feb87148a0c6d7ca0) feat(agent): the system prompt as sections, measured on the playground task list (AGNT-13)
- [`ad89a71`](https://github.com/tildaslashalef/nuclis/commit/ad89a712112c0ad46d58a0fd94f0bf649a165528) docs(repo): the Metal tier is run for a unit that touched the inference stack, judged by the work
- [`a984d0a`](https://github.com/tildaslashalef/nuclis/commit/a984d0aeaf1e70f2cffa04b054cee9d5048964b3) docs(term): TERM-10 closed: the repaint tick, the frame, the operation rows, the banded diff, and the editor's shell and keys
- [`a429e60`](https://github.com/tildaslashalef/nuclis/commit/a429e604c4ef420a49c9b75cbd48ca75b479c8e1) feat(term): `!` and `!!` run a shell command, Ctrl-O folds tool output, Ctrl-X copies the last answer, Ctrl-G edits the input externally (TERM-10)
- [`f7a8bd7`](https://github.com/tildaslashalef/nuclis/commit/f7a8bd77a61372d4f0f651530a801fc4a2878524) feat(term): markdown hardening: prefix fuzz, pathological goldens, nested quotes, bold italic, soft breaks kept, rows never past the edge (TERM-10)
- [`77346b2`](https://github.com/tildaslashalef/nuclis/commit/77346b2bf5d7e987099c7a6d5abccfedf8f97f2b) feat(term): the diff's gutter, marker cell, and bands; the header counts the change (TERM-10)
- [`22005bf`](https://github.com/tildaslashalef/nuclis/commit/22005bf2d5ed84074c4e9ea7c4a21b26206bff2d) docs(repo): one specification: spec.md rewritten as a technical spec with the agent spec merged in (REPO-13)
- [`15ae076`](https://github.com/tildaslashalef/nuclis/commit/15ae0760f98285a6421c85b7e0a6fed2bda159ae) docs(repo): the architecture guide as a map: tables, bullets, and pointers, no identifiers in prose (REPO-12)
- [`ed0ba36`](https://github.com/tildaslashalef/nuclis/commit/ed0ba367acea895da0cb9591f4c1b909b6cdd491) docs(repo): the architecture guide follows the KV cache end to end; the inference guide rewritten as one narrative (REPO-12)
- [`21aa2ee`](https://github.com/tildaslashalef/nuclis/commit/21aa2ee2f2c63ed03dddd7c4a54c16c397850f5e) fix(term): a fold toggle keeps the editor at the bottom, the input box loses its background, the bar says speculative (TERM-10)
- [`81b3adc`](https://github.com/tildaslashalef/nuclis/commit/81b3adc129c7747fe70f473235f49fb6f3b6ed24) docs(term): the transcript's operation rows, the bar's groups, the warm-up, and the session-1 hand-off (TERM-10)
- [`9c1f0a5`](https://github.com/tildaslashalef/nuclis/commit/9c1f0a5a305bb0ffe16eeaf81587c2eeae8866e5) feat(term): operation dots that pulse while running, Name(argument) call rows, write summaries, and the turn's operations row (TERM-10)
- [`bbe58df`](https://github.com/tildaslashalef/nuclis/commit/bbe58df9b8fbf9177dfd747e8a63cd22a9e71505) feat(term): frame the input box; its edge carries the effort's colour and the running spinner (TERM-10)
- [`6361a32`](https://github.com/tildaslashalef/nuclis/commit/6361a321df9bf3d8f744088f27e6dbe2ae448b55) feat(term): the status bar states its settings at the right edge, the speculative switch among them (TERM-10)
- [`093843c`](https://github.com/tildaslashalef/nuclis/commit/093843cf7d246fb047e6996102afef8b410a97ca) feat(term): frame the header in a box, and keep it on screen through the first frames (TERM-10)
- [`56d6eb5`](https://github.com/tildaslashalef/nuclis/commit/56d6eb585d346485a3ad976a95a24dd819a63f49) feat(term): the warm-up shows its progress in the region and reports when it is done (TERM-10)
- [`870d33d`](https://github.com/tildaslashalef/nuclis/commit/870d33de87d77f3d113a733e91ec8117b01a7446) feat(term): repaint at 10 Hz through every GPU wait, and rewrite only the changed rows (TERM-10)
- [`50e326d`](https://github.com/tildaslashalef/nuclis/commit/50e326d62eed14ac1d26d6a1d40dc3e1e985a062) feat(term): the tmux screenshot harness, and the header keeps its first row (TERM-10)
- [`082a379`](https://github.com/tildaslashalef/nuclis/commit/082a379cf0cfd8f6cf2c513cc3bba4cc3e9c09ad) docs(plan): order the agent units before the vision units
- [`d41ff60`](https://github.com/tildaslashalef/nuclis/commit/d41ff603c107c55b07bbc05ac6262b968de4abde) docs(plan): order AGNT-11 ahead of MODL-21
- [`4635ced`](https://github.com/tildaslashalef/nuclis/commit/4635ced0041743cb35d1e810d53879b0ecb798cc) docs(plan): draft AGNT-13, the system prompt as sections, after pi's
- [`41477d9`](https://github.com/tildaslashalef/nuclis/commit/41477d93b5e3ced441eaec5f14a88ac06c5a3921) docs(plan): widen TERM-10 to two sessions with the harness, the repaint tick, and pi's ideas
- [`5d4eedb`](https://github.com/tildaslashalef/nuclis/commit/5d4eedb29d179a02b2a86b8e65d1d1f93bbcdc5d) feat(apps): raise the output budget default to 4096 and its cap to 16384 (APPS-16)
- [`a174cd4`](https://github.com/tildaslashalef/nuclis/commit/a174cd490301f3417c4cab4f163a0282e181bde4) docs(repo): point the plan at TERM-10 after the v0.2.0 tag
- [`c01cbc6`](https://github.com/tildaslashalef/nuclis/commit/c01cbc62d7c3c365b556da1462855a75e9b3b739) chore: begin 0.3.0-dev

</details>

**Full diff:** [v0.2.0...v0.3.0](https://github.com/tildaslashalef/nuclis/compare/v0.2.0...v0.3.0)

## [v0.2.0] - 2026-09-21

Speculative decoding arrives: batched verification with checkpoint and rewind of the recurrent state, Qwen3.8's draft head, Gemma 4's draft heads, and Muse Glimmer's DFlash drafter, each measured per family (on for Muse; off for Qwen and Gemma in this release). Three families join: Gemma 4 26B-A4B (mixture of experts), Bonsai 2 27B (ternary weights), and Muse Glimmer 30B. The agent learns Gemma's and Muse's native tool calls, survives truncated and empty tool results, and resumes sessions; the default context is 16K.

### Models

- **[MODL-09](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#modl-09--gemma-4-26b-a4b-artifact-pin-facts-adapter-cpu-reference-metal-plan-2026-09-18-two-sessions)** Gemma 4 26B-A4B: artifact pin, facts, adapter, CPU reference, Metal plan
- **[MODL-10](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#modl-10--gemma-4-26b-a4b-catalogue-verdict-acceptance-record-agent-check-2026-09-18)** Gemma 4 26B-A4B: catalogue verdict, acceptance record, agent check
- **[MODL-11](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#modl-11--muse-glimmer-30b-artifact-pin-facts-tokenizer-binding-cpu-reference-2026-09-19)** Muse Glimmer 30B: artifact pin, facts, tokenizer, binding, CPU reference
- **[MODL-12](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#modl-12--muse-glimmer-30b-metal-plan-2026-09-19)** Muse Glimmer 30B: Metal plan
- **[MODL-13](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#modl-13--muse-glimmer-30b-profile-text-reasoning-channel-catalogue-acceptance-2026-09-19)** Muse Glimmer 30B: profile (text, reasoning channel), catalogue, acceptance
- **[MODL-14](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#modl-14--catalogue-entries-ahead-of-their-adapters-roadmap-reordered-around-speculation-2026-09-17)** Catalogue entries ahead of their adapters; roadmap reordered around speculation
- **[MODL-15](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#modl-15--bonsai-2-27b-accepted-ahead-of-muse-artifact-pinned-facts-read-three-units-planned-2026-09-18)** Bonsai 2 27B accepted ahead of Muse: artifact pinned, facts read, three units planned
- **[MODL-16](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#modl-16--bonsai-2-27b-oracle-facts-ternary-encodings-hadamard-transform-cpu-reference-2026-09-18)** Bonsai 2 27B: oracle, facts, ternary encodings, Hadamard transform, CPU reference
- **[MODL-17](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#modl-17--bonsai-2-27b-the-qwen-plan-on-rotated-weights-catalogue-acceptance-2026-09-18)** Bonsai 2 27B: the Qwen plan on rotated weights, catalogue, acceptance
- **[MODL-18](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#modl-18--qwen38-draft-head-the-embedded-prediction-block-on-the-cpu-reference-and-the-metal-plan-2026-09-19--2026-09-20-two-sessions)** Qwen3.8 draft head: the embedded prediction block on the CPU reference and the Metal plan
- **[MODL-19](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#modl-19--gemma-4-draft-heads-the-gemma4-assistant-companion-as-a-second-gguf-correct-but-a-negative-default-at-the-plans-draft-length-2026-09-21)** Gemma 4 draft heads: the `gemma4-assistant` companion as a second GGUF, correct but a negative default at the plan's draft length
- **[MODL-20](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#modl-20--muse-glimmer-dflash-drafter-the-companion-the-cpu-reference-and-its-trace-the-metal-plan-a-positive-verdict-2026-09-21-two-sessions)** Muse Glimmer DFlash drafter: the companion, the CPU reference and its trace, the Metal plan, a positive verdict

### Engine

- **[ENGN-09](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#engn-09--primed-sessions-prefill-the-prefix-at-startup-restore-it-on-new-2026-09-15)** Primed sessions: prefill the prefix at startup, restore it on new
- **[ENGN-10](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#engn-10--muse-glimmer-decode-gap-the-experiments-accepted-into-the-performance-theme-2026-09-19)** Muse Glimmer decode gap: the experiments accepted into the performance theme
- **[ENGN-11](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#engn-11--speculative-state-recovery-checkpoint-rewind-truncate-recover-2026-09-19-one-session)** Speculative state recovery: checkpoint, rewind, truncate, recover
- **[ENGN-12](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#engn-12--batched-verification-speculative-generation-greedy-and-sampled-the-switch-and-the-draft-length-the-benchmark-record-2026-09-20-two-sessions)** Batched verification, speculative generation (greedy and sampled), the switch and the draft length, the benchmark record
- **[ENGN-13](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#engn-13--prompt-commit-at-the-plans-chunk-and-the-batched-drafter-commit-2026-09-20)** Prompt commit at the plan's chunk and the batched drafter commit
- **[ENGN-14](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#engn-14--recovery-without-the-whole-stack-replay-2026-09-20-two-sessions)** Recovery without the whole-stack replay
- **[ENGN-15](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#engn-15--sampled-acceptance-on-the-gpu-top-k-readback-2026-09-20)** Sampled acceptance on the GPU top-k readback
- **[ENGN-16](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#engn-16--draft-proposal-policy-the-p_min-early-stop-shipped-the-adaptive-length-dropped-2026-09-20)** Draft proposal policy: the `p_min` early stop shipped, the adaptive length dropped
- **[ENGN-17](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#engn-17--the-speculative-verdict-qwen-and-gemma-off-muse-on-the-full-record-and-benchs-true-baseline-2026-09-21)** The speculative verdict: Qwen and Gemma off, Muse on; the full record and `bench`'s true baseline
- **ENGN-18** The speed loop: saved prefixes, verify cost at depth, interleaved A/B, and the decode-speed baseline _(in progress)_

### Kernels

- **[KERN-09](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#kern-09--expert-routing-and-gathered-expert-kernels-decode-and-prefill-2026-09-17--2026-09-18)** Expert routing and gathered expert kernels (decode and prefill)
- **[KERN-10](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#kern-10--ternary-matvec-and-matmul-tiles-the-walsh-hadamard-kernel-2026-09-18)** Ternary matvec and matmul tiles, the Walsh-Hadamard kernel
- **[KERN-11](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#kern-11--small-chunk-prefill-matmul-near-the-weight-bandwidth-floor-2026-09-19-two-sessions)** Small-chunk prefill matmul near the weight-bandwidth floor
- **[KERN-12](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#kern-12--a-multi-row-matvec-for-28-rows-the-2-row-routing-2026-09-20-two-sessions-closed-below-its-target)** A multi-row matvec for 2–8 rows: the 2-row routing
- **[KERN-13](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#kern-13--a-gpu-penalty-kernel-the-token-history-applied-on-the-device-before-the-top-k-2026-09-20)** A GPU penalty kernel: the token history applied on the device before the top-k
- **[KERN-14](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#kern-14--the-wide-328-small-batch-tile-measured-closed-negative-2026-09-20)** The wide 32×8 small-batch tile: measured, closed negative
- **[KERN-15](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#kern-15--split-k-decode-matvec-measured-behind-the-single-pass-closed-negative-2026-09-21)** Split-K decode matvec: measured behind the single pass, closed negative
- **[KERN-16](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#kern-16--long-context-prefill-attention-register-level-reuse-measured-25--at-chunk-sizes-closed-negative-2026-09-21)** Long-context prefill attention: register-level reuse measured 2–5 % at chunk sizes, closed negative
- **[KERN-18](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#kern-18--fused-decode-norms-182192-dispatches-per-decode-step-shipped-the-speed-bars-missed-closed-below-its-target-2026-09-21)** Fused decode norms: −182…−192 dispatches per decode step shipped, the speed bars missed; closed below its target

### Agent

- **[AGNT-08](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#agnt-08--context-budget-bounded-results-in-turn-elision-honest-failure-2026-09-15)** Context budget: bounded results, in-turn elision, honest failure
- **[AGNT-09](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#agnt-09--gemma-4-native-tool-calling-2026-09-16)** Gemma 4 native tool calling
- **[AGNT-10](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#agnt-10--muse-glimmer-atem-tool-calling-rendering-decoding-fixtures-2026-09-19)** Muse Glimmer ATEM tool calling: rendering, decoding, fixtures
- **[AGNT-11](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#agnt-11--a-truncated-tool-call-no-longer-bricks-the-session-reopened-thought-channels-copied-bracket-pieces-2026-09-17)** A truncated tool call no longer bricks the session; reopened thought channels; copied bracket pieces
- **[AGNT-12](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#agnt-12--an-empty-tool-result-no-longer-aborts-the-turn-2026-09-18)** An empty tool result no longer aborts the turn

### Application

- **[APPS-06](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#apps-06--nuclis-agent-ls-and---resume-to-the-newest-session-2026-09-16)** `nuclis agent ls` and `--resume` to the newest session
- **[APPS-07](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#apps-07----prompt-profile-and-the-registrys-profile-the-template-alias-gate-2026-09-17)** `--prompt-profile` and the registry's `profile`; the template-alias gate
- **[APPS-08](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#apps-08--model-pull-a-leaked-path-per-file-and-a-leading-slash-in-the-sidecars-file-name-2026-09-17)** `model pull`: a leaked path per file and a leading slash in the sidecar's file name
- **[APPS-09](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#apps-09--config-set-model-pull---register-registry-names-in-model-ls-2026-09-17)** `config set`, `model pull --register`, registry names in `model ls`
- **[APPS-10](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#apps-10--model-ls-as-one-aligned-grid-2026-09-17)** `model ls` as one aligned grid
- **[APPS-11](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#apps-11--model-pull-a-verified-file-whose-encoding-this-build-does-not-store-keeps-its-sidecar-2026-09-18)** `model pull`: a verified file whose encoding this build does not store keeps its sidecar
- **[APPS-12](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#apps-12--the-default-context-window-is-16k-2026-09-18)** The default context window is 16K
- **[APPS-13](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#apps-13--the-configuration-section-is-generation-not-generate-2026-09-20)** The configuration section is `generation`, not `generate`
- **[APPS-15](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#apps-15--config-init---discover-and-self-contained-help-pages-2026-09-21)** `config init --discover` and self-contained help pages

### Terminal

- **[TERM-05](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#term-05--two-row-tool-lines-call-and-detail-2026-09-15)** Two-row tool lines: call and detail
- **[TERM-06](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#term-06--per-step-thinking-blocks-with-their-own-duration-2026-09-15)** Per-step thinking blocks with their own duration
- **[TERM-07](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#term-07--rows-released-by-a-shrinking-live-region-are-reused-not-left-as-gaps-2026-09-17)** Rows released by a shrinking live region are reused, not left as gaps
- **[TERM-08](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#term-08--the-welcome-ascii-wordmark-and-session-facts-the-models-name-on-the-status-bar-2026-09-17)** The welcome: ASCII wordmark and session facts; the model's name on the status bar
- **[TERM-09](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#term-09--a-step-that-answers-and-then-calls-a-tool-no-longer-holds-the-turn-in-the-live-region-2026-09-18)** A step that answers and then calls a tool no longer holds the turn in the live region

### Repository

- **[REPO-05](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#repo-05--readme-for-a-public-repository-project-status-contributions-disclosure-2026-09-18)** README for a public repository: project status, contributions, disclosure
- **[REPO-06](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#repo-06--diffusiongemma-structured-reads-and-kev-research-2026-09-20)** DiffusionGemma structured reads and kev research
- **[REPO-07](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#repo-07--the-roadmap-file-retired-themes-are-agreed-in-session-and-when-architectural-recorded-as-adrs-on-request-2026-09-20)** The roadmap file retired; themes are agreed in session and, when architectural, recorded as ADRs on request
- **[REPO-08](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#repo-08--repair-the-multi-row-benchmark-controls-and-hand-off-2026-09-20)** Repair the multi-row benchmark controls and hand-off
- **[REPO-09](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#repo-09--one-gate-registry-tiers-change-triggers-and-the-model-specific-checks-as-data-2026-09-21)** One gate registry: tiers, change triggers, and the model-specific checks as data
- **[REPO-10](https://github.com/tildaslashalef/nuclis/blob/v0.2.0/docs/engineering-log.md#repo-10--benchmark-workloads-as-data-and-generated-record-tables-2026-09-21)** Benchmark workloads as data and generated record tables

### Breaking changes

- **engn:** speculative configuration, the bench off/on pair, and the loop edge tests

### Outside units

- **agent:** read_file counts a trailing newline as the end of the last line
- **agent:** free the sidecar strings and the cwd sentinel on the interactive path
- **agent:** a cut read's row states the kept range; a step's prefill resets the bar's word and decode clock
- **agent:** steer a whole-file question to one command instead of paging; the bar's word after a turn
- **agent:** the system prompt forbids reconstructing a file from memory
- **model:** catalogue entries for gemma-4-26b-a4b and muse-glimmer-30b; roadmap reordered
- **quant:** decode PQ2_0, PTQ1_0, and BF16 rows against the PrismML fork's fixtures
- **model:** the Qwen adapter binds Bonsai 2 27B: rotation contract, optional auxiliary block
- **tokenizer:** the special-token scan searches each marker's first byte, so a vocabulary with thousands of markers tokenizes long text within budget
- **metal:** an 8x8 split-K prefill tile for chunks up to eight tokens
- **metal:** the split-K tile stores each partial into its own tile
- **metal:** order the split tile's step reuse with simdgroup barriers
- **metal:** 16 rows per split-tile threadgroup and a 16-token threshold
- **metal:** set small_chunk_tokens to 24 from the wall-clock crossover
- **apps:** validate reports the embedded draft head
- **repo:** repair multi-row benchmark controls

<details><summary>All 132 commits</summary>

- [`142fa81`](https://github.com/tildaslashalef/nuclis/commit/142fa81cd54b53426a68c45ebd073dd4ad29545d) chore(release): v0.2.0
- [`d058fe5`](https://github.com/tildaslashalef/nuclis/commit/d058fe50d6013dac6ba5332cedbeb5aafd5d845c) feat(apps): config init --discover and self-contained help pages (APPS-15)
- [`5c03fad`](https://github.com/tildaslashalef/nuclis/commit/5c03fade0ca110cacba3b4cd76ffe14b425c86f3) docs(repo): plan APPS-15 (config init --discover, self-contained help); drop REPO-11
- [`31fb283`](https://github.com/tildaslashalef/nuclis/commit/31fb2838ba384418ae913a57accb540cdea65afa) feat(repo): benchmark workloads as data and generated record tables (REPO-10)
- [`7eeeac7`](https://github.com/tildaslashalef/nuclis/commit/7eeeac70be722be97e9b6233541e26983ba6f0d4) feat(repo): one gate registry with tiers and change triggers (REPO-09)
- [`8d75efb`](https://github.com/tildaslashalef/nuclis/commit/8d75efb6cf9c57fb94f7f6bdfbab845c1e8ecedd) docs(repo): reshape REPO-09/REPO-10 around gate tiers and change triggers
- [`31c32e1`](https://github.com/tildaslashalef/nuclis/commit/31c32e18ffde7720c35e82c8bd17f60c55313418) feat(apps): set the speculative verdicts and bench's true baseline (ENGN-17)
- [`981f74d`](https://github.com/tildaslashalef/nuclis/commit/981f74d6a9ef0ddcca927faa6fba80b31be58fe6) feat(model): run the Muse DFlash drafter on Metal (MODL-20)
- [`316accc`](https://github.com/tildaslashalef/nuclis/commit/316accc5e480141a3af010278ce8b64f6d105622) feat(model): add the Muse DFlash drafter's CPU reference (MODL-20)
- [`af906fc`](https://github.com/tildaslashalef/nuclis/commit/af906fc13282618ac7389bbeadebe1b2e61e12e9) docs(repo): spell out the Makefile and bench.md division of REPO-09/REPO-10
- [`f1feb50`](https://github.com/tildaslashalef/nuclis/commit/f1feb5094c710687e77e4cdfdc9eb257a3a75335) docs(repo): drop the Gemma decode, ring-layout, and ternary units from the plan
- [`85994ef`](https://github.com/tildaslashalef/nuclis/commit/85994ef35a48c48a1e28ce93366a5e424a0d5c15) docs(repo): order the gate registry and benchmark workloads after the verdict (REPO-09, REPO-10)
- [`4ead869`](https://github.com/tildaslashalef/nuclis/commit/4ead869dcc1b10297c7757a8911cd5739fd1fdf9) feat(model): add the Gemma 4 assistant draft heads (MODL-19)
- [`4bc7b8d`](https://github.com/tildaslashalef/nuclis/commit/4bc7b8dae0f6b858c281a1d7381f68e01b5628e4) docs(model): pin the Gemma 4 assistant head facts and traces (MODL-19)
- [`1a3ee68`](https://github.com/tildaslashalef/nuclis/commit/1a3ee68d38d893cbf75cc4aaec1051323ad4f156) perf(kern): fuse the decode norms behind a control flag, measured below target (KERN-18)
- [`1d82c18`](https://github.com/tildaslashalef/nuclis/commit/1d82c1899ab23845d9cd5334e0d47e973c0b8766) perf(kern): measure the register-reuse chunk attention and keep its verify window (KERN-16)
- [`652a0cc`](https://github.com/tildaslashalef/nuclis/commit/652a0ccd66a91d3a42d021b8e29159adbb704e2a) docs(repo): add KERN-18 (fused decode norms) and rescope ENGN-18 to the Gemma tiles
- [`ef129e6`](https://github.com/tildaslashalef/nuclis/commit/ef129e6e8aa480e9ad2adb04d78d5631a7bb8cdc) docs(repo): pull the Gemma 4 and Muse drafters ahead of the verdict (MODL-19, MODL-20)
- [`efac408`](https://github.com/tildaslashalef/nuclis/commit/efac4085054f7a8082b53525547d2cc7cf1d9356) perf(kern): measure the split-K decode matvec behind the single pass (KERN-15)
- [`10cf6ba`](https://github.com/tildaslashalef/nuclis/commit/10cf6baa769917cfafbadd62d2d5e379561593a5) feat(engn): stop drafting below the p_min probability; drop the adaptive length (ENGN-16)
- [`bdaaead`](https://github.com/tildaslashalef/nuclis/commit/bdaaead188c9a5e011a034319d9889e96814dfea) perf(kern): measure the wide 32x8 small-batch tile and keep the 16x8 control (KERN-14)
- [`a1ef2d0`](https://github.com/tildaslashalef/nuclis/commit/a1ef2d06aa58140655e5be485927eaed74bfc514) feat(engn): decide sampled acceptance from the per-row top-k readback (ENGN-15)
- [`54798e1`](https://github.com/tildaslashalef/nuclis/commit/54798e1446c29c49238564766e2491eed6979c24) feat(kern): apply the history penalties on the device before selection (KERN-13)
- [`d31c5cd`](https://github.com/tildaslashalef/nuclis/commit/d31c5cd3afc815cc644bc7e7c048f35380d9ef49) docs(engn): close ENGN-14 with the row-checkpoint record and replan the speculative units
- [`417aea0`](https://github.com/tildaslashalef/nuclis/commit/417aea0e752cd094295e33ba62c432b80d180dc0) feat(engn): per-row recurrent checkpoints replace the recovery replay (ENGN-14)
- [`b2ad171`](https://github.com/tildaslashalef/nuclis/commit/b2ad171e338ef33b2dc46bc959a439841c176437) feat(engn): measure recovery by accepted length and take the seed-only replay through step (ENGN-14)
- [`6fb2dfe`](https://github.com/tildaslashalef/nuclis/commit/6fb2dfe050594309cad8e5ee442a0ebb16814584) fix(repo): repair multi-row benchmark controls
- [`27303ed`](https://github.com/tildaslashalef/nuclis/commit/27303edcc5c9444e40cdc48773397a4ca030c18d) feat(kern): register-tile the multi-row matvec and route 2-row batches (KERN-12)
- [`888e526`](https://github.com/tildaslashalef/nuclis/commit/888e5262aa5206f3699f7da5f2c35d3d7f3c14d2) feat(kern): multi-row matvec kernels and sweep, routing gated off (KERN-12)
- [`487bfbc`](https://github.com/tildaslashalef/nuclis/commit/487bfbc7d0e29c82fe319c9cc30094b3ec1f3150) build(repo): default compare aggregates to Metal; wire bench-matvec-rows
- [`5e24b69`](https://github.com/tildaslashalef/nuclis/commit/5e24b691f57306e13cfbcc3f54ec9aab2c91a381) docs(engn): close ENGN-13, the prompt and batched drafter commit (ENGN-13)
- [`ccb191b`](https://github.com/tildaslashalef/nuclis/commit/ccb191b8c01b690c6560320875ce145dd1fe2251) docs(engn): record the ENGN-13 prompt-commit and batched-commit numbers (ENGN-13)
- [`ded2a30`](https://github.com/tildaslashalef/nuclis/commit/ded2a30a60e1acdff8339556e6c77ef516548e8c) test(engn): check the batched commit against the serial path (ENGN-13)
- [`9a5d3cf`](https://github.com/tildaslashalef/nuclis/commit/9a5d3cf6c2f4ecfa3a6fe91395c0c62600a422b2) feat(engn): time propose and commit, carried into the bench sample (ENGN-13)
- [`390a9cf`](https://github.com/tildaslashalef/nuclis/commit/390a9cfa05879793786a7ca6e4d310f2dcc394cd) feat(engn): prime the drafter and surface accepted drafts per step (ENGN-13)
- [`327a297`](https://github.com/tildaslashalef/nuclis/commit/327a29714be957b0304174c703f7096f8bfab848) docs(engn): record the batched drafter commit measurement (ENGN-13)
- [`4525e81`](https://github.com/tildaslashalef/nuclis/commit/4525e81f65441db8a2ae211c284081ac10e88379) feat(engn): batched drafter commit on the Metal plan (ENGN-13)
- [`672b258`](https://github.com/tildaslashalef/nuclis/commit/672b258dc22f4f164649599dc97e529180850785) docs(engn): record the prefill-chunk prompt commit measurement (ENGN-13)
- [`db9cf80`](https://github.com/tildaslashalef/nuclis/commit/db9cf80acfff6d770b6683144bfa5e1a45d44bb3) feat(engn): commit the prompt to the drafter in prefill chunks (ENGN-13)
- [`ff90509`](https://github.com/tildaslashalef/nuclis/commit/ff90509358707af102501035b80ea139a123fb31) feat(engn): prefill returns post-norm hidden rows (ENGN-13)
- [`b40209c`](https://github.com/tildaslashalef/nuclis/commit/b40209ca3f90bcddaeae0c07b6133b6521e54d57) docs(repo): retire the roadmap; themes are agreed in session and, when architectural, recorded as ADRs on request (REPO-07)
- [`955a166`](https://github.com/tildaslashalef/nuclis/commit/955a1664518dcc99a25198a60766b52a5c313320) feat(engn): the speculative record closes ENGN-12; the plan revamped for the speed units, vision, and the chat polish
- [`87d4f19`](https://github.com/tildaslashalef/nuclis/commit/87d4f19c1723cb2827511aea719dedf6da7af1b9) docs: research DiffusionGemma decisions and kev
- [`3d5cb94`](https://github.com/tildaslashalef/nuclis/commit/3d5cb94dda8066108c03e0ec29f48a487d591116) fix(engn): sampled acceptance draws from the target; the batch catches the real cancellation (ENGN-12)
- [`891f1bf`](https://github.com/tildaslashalef/nuclis/commit/891f1bf0f8e0dde5dcf2ea075db7c2abb906d6dc) feat(engn)!: speculative configuration, the bench off/on pair, and the loop edge tests (ENGN-12)
- [`6794493`](https://github.com/tildaslashalef/nuclis/commit/6794493cc65cd90ed36ddbf30f77a86fe6f7231e) feat(engn): batched verify, the speculative loop, and sampled acceptance (ENGN-12)
- [`215c7a0`](https://github.com/tildaslashalef/nuclis/commit/215c7a05cba470464e0c90b2e9f931cc4bd37880) feat(modl): the Qwen prediction block on the Metal plan, the compare rows, and the acceptance statistic (MODL-18)
- [`2496046`](https://github.com/tildaslashalef/nuclis/commit/2496046dfb896b0d0ee1ff7edc437057b2e7c136) docs(plan): MODL-18 session 1 complete on the CPU; session 2 next
- [`f62b351`](https://github.com/tildaslashalef/nuclis/commit/f62b351c7c4fe294323d6918b1ecbd7c157e191f) feat(apps): validate reports the embedded draft head
- [`bbad8e3`](https://github.com/tildaslashalef/nuclis/commit/bbad8e3a7c234009855996e3c749d5d320ba3f84) feat(modl): the Qwen prediction block on the CPU, the draft contract, and the load request (MODL-18)
- [`31b4929`](https://github.com/tildaslashalef/nuclis/commit/31b49294eaea6b379af8389f03e5ca3cb2d1cffb) feat(modl): bind the Qwen3.8 prediction head and keep the target hidden (MODL-18)
- [`d73b83f`](https://github.com/tildaslashalef/nuclis/commit/d73b83f3bf27ed00fa9b709d0d9d05827eb3c522) feat(engn): speculative state recovery — checkpoint, rewind, truncate, recover (ENGN-11)
- [`a4c4d09`](https://github.com/tildaslashalef/nuclis/commit/a4c4d09c47344dc4ac8edf577741b902f1366f89) docs(metal): the small-chunk tile's threshold and geometry stated consistently after KERN-11
- [`af9865f`](https://github.com/tildaslashalef/nuclis/commit/af9865fcd3d852b4f904f29b15b687ce41e71aaf) docs(kern): KERN-11 closes; the log entry, the roadmap bullet removed, the plan points at ENGN-11
- [`502f934`](https://github.com/tildaslashalef/nuclis/commit/502f9349f11461178cfc93db07aa1816a36baab6) docs(bench): KERN-11's 22-token prompt: prefill +10.9 %, first token -9.8 %
- [`0eb50ec`](https://github.com/tildaslashalef/nuclis/commit/0eb50ec80f5b771ad016d1c7a8d172bbad8271b5) fix(metal): set small_chunk_tokens to 24 from the wall-clock crossover
- [`06b62b7`](https://github.com/tildaslashalef/nuclis/commit/06b62b774bf2aab195e660c8261345deed2cd577) perf(metal): 16 rows per split-tile threadgroup and a 16-token threshold
- [`315322d`](https://github.com/tildaslashalef/nuclis/commit/315322dce3bf30d9f64ac441777720a46b6f96e1) fix(metal): order the split tile's step reuse with simdgroup barriers
- [`2fee4c5`](https://github.com/tildaslashalef/nuclis/commit/2fee4c5c6562d09a6dac191355adc6ee5fcc61d2) bench(metal): batch matmulBench dispatches and attribute bytes per token tile
- [`5e8954e`](https://github.com/tildaslashalef/nuclis/commit/5e8954ea458b819a8ce2712fcb2336f5f611f228) docs(plan): the speculative units rewritten at implementation level from the companions' headers and the reference's driver; the protocol asks for that level
- [`1c29083`](https://github.com/tildaslashalef/nuclis/commit/1c29083ee36622704d05097a868e4f0597f4e9d4) docs(plan): KERN-11 session 1 reviewed; the batched bench, the activation operand, and the barrier set session 2's order
- [`5a22f15`](https://github.com/tildaslashalef/nuclis/commit/5a22f15c5feac7038e2b9eb078bc47194a1801aa) docs(metal): the small-chunk tile's first measurements and the KERN-11 session 1 hand-off
- [`c486bcb`](https://github.com/tildaslashalef/nuclis/commit/c486bcb736706ea226ee4a302f34eee13ee14deb) style(metal): reflow the kernel_names array after the new entries
- [`c066bdc`](https://github.com/tildaslashalef/nuclis/commit/c066bdc4612214a99a65d5966b5f8d12113e679e) fix(metal): the split-K tile stores each partial into its own tile
- [`5559226`](https://github.com/tildaslashalef/nuclis/commit/5559226c3efd9d83dec0a6989d9ffc62cd92db25) bench(metal): report weight-byte GB/s and the selected tile in matmulBench
- [`d219379`](https://github.com/tildaslashalef/nuclis/commit/d219379d4028c6bba118f57da2a25cfe56b000dd) test(metal): matmul exactness at 1, 5, and 8 tokens and the split-tile thresholds
- [`5d344a8`](https://github.com/tildaslashalef/nuclis/commit/5d344a8632eabf35f26e5ff1f5636cafc2674f33) feat(metal): an 8x8 split-K prefill tile for chunks up to eight tokens
- [`a947b52`](https://github.com/tildaslashalef/nuclis/commit/a947b52c2970c230fafce9a8b5a4dd8666745b5b) docs(plan): KERN-11, the small-chunk prefill matmul, inserted ahead of the speculative units
- [`4608bbc`](https://github.com/tildaslashalef/nuclis/commit/4608bbcfd55724892ba1f6e462ac7ac7c35f839d) docs(plan): speculative decoding across the families planned as five units; its requirements move to the spec
- [`8d23c1f`](https://github.com/tildaslashalef/nuclis/commit/8d23c1fc2693d155c932a523f42af64187155f82) docs(roadmap): the Muse Glimmer decode gap's experiments join the performance theme (ENGN-10)
- [`7194e1e`](https://github.com/tildaslashalef/nuclis/commit/7194e1e1009f2a893136a33e701c540c024d7d4e) docs(agent): Muse Glimmer ATEM tool calling closes on the pinned fixtures and the live turn; the plan is empty (AGNT-10)
- [`6758ac9`](https://github.com/tildaslashalef/nuclis/commit/6758ac9bf06fe306277af5aeb612e0d4c85390c9) feat(profiles): Muse Glimmer profile acceptance: the reference and nuclis records, the harness family, the log; the unit closes (MODL-13)
- [`bbac7ac`](https://github.com/tildaslashalef/nuclis/commit/bbac7acaa96a6ebf732dac371289493576879e92) fix(tokenizer): the special-token scan searches each marker's first byte, so a vocabulary with thousands of markers tokenizes long text within budget
- [`d3f1a5b`](https://github.com/tildaslashalef/nuclis/commit/d3f1a5b6240d721c88c5987370a149f5b61b4f7a) feat(profiles): Muse Glimmer ATEM tool calling: declarations, calls, results, and the body parser match the pinned fixtures (AGNT-10)
- [`20d686b`](https://github.com/tildaslashalef/nuclis/commit/20d686bb43585f5c6547f159322ba21c6b4a2969) feat(profiles): the Muse Glimmer profile with the channel-grammar decoder, the high effort, and the catalogue pin (MODL-13)
- [`35b6e51`](https://github.com/tildaslashalef/nuclis/commit/35b6e5186d4388cc0cf0a0ae6c2fc822702525ba) feat(models): the Muse Glimmer Metal plan matches the pinned traces in both cache precisions; RoPE gains adjacent pairing (MODL-12)
- [`0806c0e`](https://github.com/tildaslashalef/nuclis/commit/0806c0e4d2fe4f1cd4e47f946af0bed3e3ab5eda) feat(models): the Muse Glimmer adapter and CPU reference match the pinned traces; the unit closes (MODL-11)
- [`2df66a6`](https://github.com/tildaslashalef/nuclis/commit/2df66a6855ab00d81178a1e47abcf22be809b82d) feat(tokenizer): the llama4 splitter matches the reference on Muse Glimmer; the family's facts, fixtures, and inventory (MODL-11)
- [`8672fb2`](https://github.com/tildaslashalef/nuclis/commit/8672fb23ecf4813913fa0d205bb052f463cab99d) fix(tui): a tool call closes the step's text, and a closed thought keeps its fold label in the live region (TERM-09)
- [`aa5f52c`](https://github.com/tildaslashalef/nuclis/commit/aa5f52c8497f62e3d0fe05771c1ad131eb60d25d) feat(config): the default context window is 16K (APPS-12)
- [`6def0c6`](https://github.com/tildaslashalef/nuclis/commit/6def0c62d26c05d456154cfbb87eced40e3949b7) fix(agent): an empty tool result costs zero tokens instead of ending the turn (AGNT-12)
- [`8c6b7f7`](https://github.com/tildaslashalef/nuclis/commit/8c6b7f7fd6a97dde9b927590142f96cab5ed2f54) feat(metal): the Qwen plan runs Bonsai 2 27B: the rotation on every activation, the catalogue on PTQ1_0, the acceptance record (MODL-17)
- [`7177307`](https://github.com/tildaslashalef/nuclis/commit/7177307d96ef5eebb4fa75095b9c35ba8977c57b) feat(metal): the signed Walsh-Hadamard transform kernel and its per-token bench (KERN-10)
- [`cdfcaee`](https://github.com/tildaslashalef/nuclis/commit/cdfcaeeeaf8cfdd55b196ae135ccb95cc26b9feb) feat(metal): ternary matvecs and prefill tiles for PQ2_0 and PTQ1_0, BF16 rows (KERN-10 session 1)
- [`adcee8b`](https://github.com/tildaslashalef/nuclis/commit/adcee8b21c2ce7f3f5349c75b119fb02858024e1) feat(engine): the Qwen CPU reference applies the Hadamard rotation; Bonsai 2 27B matches the fork's traces (MODL-16)
- [`9adef98`](https://github.com/tildaslashalef/nuclis/commit/9adef9808e2e4f039db7ee1ac66ab065bb3f4ece) docs: Bonsai 2 27B facts, the fork as second oracle, traces, alias evidence, guide § 49 (MODL-16 session 1)
- [`24b63c9`](https://github.com/tildaslashalef/nuclis/commit/24b63c99b880896d7b5656e34c8d59838c90409b) feat(model): the Qwen adapter binds Bonsai 2 27B: rotation contract, optional auxiliary block
- [`0be05ae`](https://github.com/tildaslashalef/nuclis/commit/0be05aed575e0bb5ddc8991f66ecc4192199b53e) feat(quant): decode PQ2_0, PTQ1_0, and BF16 rows against the PrismML fork's fixtures
- [`28bc32c`](https://github.com/tildaslashalef/nuclis/commit/28bc32cf13fddbeae3c491fc3f6111951684e377) docs(readme): acknowledge ds4 as the Metal bridge's design reference
- [`cf9a3f7`](https://github.com/tildaslashalef/nuclis/commit/cf9a3f788dc6d7e86ea03176198523ced3a5979c) docs(readme): every supported file in the results table
- [`c384f07`](https://github.com/tildaslashalef/nuclis/commit/c384f07502fb1d556bc43bda6be94f904c0c6f86) docs: toolchain paths read as the author's machine, not as requirements
- [`383b3da`](https://github.com/tildaslashalef/nuclis/commit/383b3dac7530024da33eb3fa1ce5a695215e01d5) docs(readme): project status, contributions policy, and AI and llama.cpp disclosure for going public (REPO-05)
- [`9a95db5`](https://github.com/tildaslashalef/nuclis/commit/9a95db5929e7db7efd4c72c74d8b54c18138bbc2) fix(model): pull keeps the sidecar of a verified file whose encoding this build does not store (APPS-11)
- [`be11ee3`](https://github.com/tildaslashalef/nuclis/commit/be11ee39bd47098333d4e46ef256edfb258c640a) feat(model): Bonsai 2 27B accepted ahead of Muse: catalogue entry pinned and pulled, facts read, three units planned; MODL-15 closed
- [`cb16ba6`](https://github.com/tildaslashalef/nuclis/commit/cb16ba626bb2cca402a77b3edc8f9007e161a310) feat(model): Gemma 4 26B-A4B supported: acceptance record against the reference, per-kernel profile, agent check; MODL-10 closed
- [`56ef7d4`](https://github.com/tildaslashalef/nuclis/commit/56ef7d42a034c1cdedda662b48a8b9d982171a8a) feat(model): Gemma 4 26B-A4B Metal plan: expert block on decode and chunked prefill, two-KV-head global attention; MODL-09 closed
- [`7701e6b`](https://github.com/tildaslashalef/nuclis/commit/7701e6b37e2ae74f3f97b71ccba1dfa9042a0d75) feat(model): Gemma 4 26B-A4B adapter and CPU expert layer against pinned reference traces (MODL-09 session 1)
- [`205b916`](https://github.com/tildaslashalef/nuclis/commit/205b916c27d7c43494684cd4f78ee91935380ea6) docs: research Jev and native Zig training options
- [`58e194c`](https://github.com/tildaslashalef/nuclis/commit/58e194ca9a8991fdc1c2710e7fc7cde40ddb405b) feat(kern): expert prefill path: GPU row lists and gathered matmul tiles; KERN-09 closed
- [`c00eb19`](https://github.com/tildaslashalef/nuclis/commit/c00eb19df24beb974ec125039369e932c097a3b6) feat(term): welcome with the ASCII wordmark and session facts; model name on the status bar (TERM-08)
- [`0e915e6`](https://github.com/tildaslashalef/nuclis/commit/0e915e66591163334dbf2447bcff6ab30abd78a5) feat(apps): model ls sizes in decimal units (APPS-10)
- [`9ad6045`](https://github.com/tildaslashalef/nuclis/commit/9ad604589a3c8f10da3f4bf57b26b3869b5af2d9) feat(apps): model ls as one aligned grid (APPS-10)
- [`b4a408d`](https://github.com/tildaslashalef/nuclis/commit/b4a408d1a2d48aea936040916a0d9c936cd3c67c) fix(agent): released calls keep their bracket, reopened thought channels are thinking, own markers are stripped from history (AGNT-11)
- [`8f60399`](https://github.com/tildaslashalef/nuclis/commit/8f603996f49eb8901536aa626fde5feffbbaf333) feat(apps): config set, model pull --register, registry names in model ls (APPS-09)
- [`035d3fc`](https://github.com/tildaslashalef/nuclis/commit/035d3fc15262848b17eb7daee07c0bd3f44cb2e7) fix(model): free each job's local path in pull; sidecar file names lose their leading slash (APPS-08)
- [`c66f855`](https://github.com/tildaslashalef/nuclis/commit/c66f855d3d7d774de11e8dec4ce11b2604d2eb0f) feat(apps): --prompt-profile and the registry's profile force a prompt profile; template-alias gate script (APPS-07)
- [`3cbd852`](https://github.com/tildaslashalef/nuclis/commit/3cbd8528fe8ab456f5390c572537d1fc1e5ad8bf) docs(log): TERM-07 confirmed on a real terminal
- [`94b3228`](https://github.com/tildaslashalef/nuclis/commit/94b3228712d7c647f130bb1579c801832ac71156) fix(term): reuse the rows a shrinking live region releases instead of leaving gaps (TERM-07)
- [`013dee2`](https://github.com/tildaslashalef/nuclis/commit/013dee25b642ade43c6ff8c5d8a86d53f4ae876e) docs(agents): the session protocol as two states over three files
- [`6d6187e`](https://github.com/tildaslashalef/nuclis/commit/6d6187ec7b508842c75bc7a899b6c7a4c337ec38) docs(log): AGNT-09 table row, MODL-14 for the catalogue and roadmap work; AGENTS.md closing rules
- [`e8d5dbc`](https://github.com/tildaslashalef/nuclis/commit/e8d5dbc9c45a1f19ba5270c5b6f75c1cf60ed3a7) docs(roadmap): speculative decoding configuration: file per entry, per-command switch, draft length
- [`c3a28d3`](https://github.com/tildaslashalef/nuclis/commit/c3a28d36acb8c36f58491217c5fbbc01329a8c79) feat(model): catalogue entries for gemma-4-26b-a4b and muse-glimmer-30b; roadmap reordered
- [`8cfdb32`](https://github.com/tildaslashalef/nuclis/commit/8cfdb32da20afe14a568e1eef030ffa1aba0dfac) feat(kern): expert routing and gathered expert kernels, decode path (KERN-09 session 1)
- [`51aba66`](https://github.com/tildaslashalef/nuclis/commit/51aba66aaeb72cda7454fa74ce89b8f0c101b2f7) docs(todo): Gemma 4 26B-A4B (KERN-09, MODL-09, MODL-10) ahead of Muse Glimmer; renumber
- [`c0906c9`](https://github.com/tildaslashalef/nuclis/commit/c0906c953535d35f67b4ca080cf6c35735097415) docs(todo): Muse profile renders the default system turn without a date line
- [`0d6a70e`](https://github.com/tildaslashalef/nuclis/commit/0d6a70ea556cf068398d66c19f0b1bf195099447) docs(todo): plan Muse Glimmer 30B bring-up and ATEM tool calling (MODL-09..11, AGNT-10)
- [`493d507`](https://github.com/tildaslashalef/nuclis/commit/493d507da877f2ad114dbea14933616201825f6a) feat(profile): Gemma 4 native tool calling: declarations, calls, results, and the tool_response handoff (AGNT-09)
- [`1c88549`](https://github.com/tildaslashalef/nuclis/commit/1c88549b68a15590741053467c9acb10b713241e) docs(todo): plan AGNT-09, Gemma 4 native tool calling
- [`52bfd8e`](https://github.com/tildaslashalef/nuclis/commit/52bfd8e1a53de8ad1276e5b5c5e804c466d5a351) fix(agent): the system prompt forbids reconstructing a file from memory
- [`0cdf588`](https://github.com/tildaslashalef/nuclis/commit/0cdf5880a638df32873432098ac0fec1615a2c0b) fix(agent): steer a whole-file question to one command instead of paging; the bar's word after a turn
- [`b779d18`](https://github.com/tildaslashalef/nuclis/commit/b779d18cfb80b7f8ac27bb55cbe91774808e821f) feat(agent): nuclis agent ls, and --resume alone continues the newest session (APPS-06)
- [`55dfc55`](https://github.com/tildaslashalef/nuclis/commit/55dfc55d053168151f9ee67f78069fb00f50808e) fix(agent): a cut read's row states the kept range; a step's prefill resets the bar's word and decode clock
- [`9ef0022`](https://github.com/tildaslashalef/nuclis/commit/9ef00226794705a6a54ecb27b03fee203bf4e907) feat(engine): primed sessions: prefill the system prefix at startup, restore it on new (ENGN-09)
- [`8895b36`](https://github.com/tildaslashalef/nuclis/commit/8895b36cc9d05f625502773db4f3631ce53a6f76) feat(agent): per-step thinking blocks with their own duration (TERM-06)
- [`a2ad757`](https://github.com/tildaslashalef/nuclis/commit/a2ad7573e083e6966d14254e3a6573893e5da963) feat(agent): in-turn elision of older tool results on a full window (AGNT-08 closed)
- [`7689de3`](https://github.com/tildaslashalef/nuclis/commit/7689de37a3558ba10dcfb0d390e0cbb06c42c982) feat(agent): context-relative result budget and an honest context-full message (AGNT-08, session 1)
- [`5bcaf32`](https://github.com/tildaslashalef/nuclis/commit/5bcaf323762ef1d9bd822960cac89bdd146a9534) fix(agent): free the sidecar strings and the cwd sentinel on the interactive path
- [`1c15b1c`](https://github.com/tildaslashalef/nuclis/commit/1c15b1c0caead397074ffe053f428bd516f9f7e1) docs(todo): open TERM-06 per-step thinking and ENGN-09 primed sessions
- [`4f62c53`](https://github.com/tildaslashalef/nuclis/commit/4f62c53a378b68fec8fe847e8bf8b4ec3d133f25) fix(agent): read_file counts a trailing newline as the end of the last line
- [`6441796`](https://github.com/tildaslashalef/nuclis/commit/6441796b2c2ce8dd8242d130efba8a6467636d53) docs: llm-guide extends only on request; drop the TERM-05 section
- [`8184bde`](https://github.com/tildaslashalef/nuclis/commit/8184bde49d564cea7bbda71d47abc0d7d5d54f05) feat(agent): two-row tool lines, call and detail (TERM-05)
- [`62b945d`](https://github.com/tildaslashalef/nuclis/commit/62b945d55a0e9d1aaf3c32e9b335c95f5dfa5d4f) chore: begin 0.2.0-dev

</details>

**Full diff:** [v0.1.0...v0.2.0](https://github.com/tildaslashalef/nuclis/compare/v0.1.0...v0.2.0)

## [v0.1.0] - 2026-09-15

The first release: Qwen3.8-27B runs on Apple Silicon through an engine written in Zig with a Metal backend, from the GGUF bytes to the sampled token, every kernel checked against a pinned llama.cpp build. It passes the 32K-context acceptance run, with chunked prefill, tiled attention, chunkwise DeltaNet, and an F16 KV cache. Gemma 4 12B joins as a second family, models are pulled from the Hugging Face Hub with pinned digests, and `nuclis agent` puts a small coding agent (read, edit, and bash tools) on top in a terminal UI.

### Models

- **[MODL-01](https://github.com/tildaslashalef/nuclis/blob/v0.1.0/docs/engineering-log.md#modl-01--qwen38-sampling-profiles-min_p-penalties-2026-09-08)** Qwen3.8 sampling profiles, `min_p`, penalties
- **[MODL-02](https://github.com/tildaslashalef/nuclis/blob/v0.1.0/docs/engineering-log.md#modl-02--model-download-through-the-huggingface-package-2026-09-11)** Model download through the `huggingface` package
- **[MODL-03](https://github.com/tildaslashalef/nuclis/blob/v0.1.0/docs/engineering-log.md#modl-03--models-directory-catalogue-and-registry-2026-09-11)** Models directory, catalogue, and registry
- **[MODL-04](https://github.com/tildaslashalef/nuclis/blob/v0.1.0/docs/engineering-log.md#modl-04--adapter-registry-2026-09-11)** Adapter registry
- **[MODL-05](https://github.com/tildaslashalef/nuclis/blob/v0.1.0/docs/engineering-log.md#modl-05--gemma-4-12b-facts-binding-cpu-reference-tokenizer-2026-09-11)** Gemma 4 12B facts, binding, CPU reference, tokenizer
- **[MODL-06](https://github.com/tildaslashalef/nuclis/blob/v0.1.0/docs/engineering-log.md#modl-06--gemma-4-12b-metal-plan-2026-09-11)** Gemma 4 12B Metal plan
- **[MODL-07](https://github.com/tildaslashalef/nuclis/blob/v0.1.0/docs/engineering-log.md#modl-07--gemma-4-12b-profile-catalogue-acceptance-new-model-guide-2026-09-12)** Gemma 4 12B profile, catalogue, acceptance, new-model guide
- **[MODL-08](https://github.com/tildaslashalef/nuclis/blob/v0.1.0/docs/engineering-log.md#modl-08--q4_0-path-and-the-qat-catalogue-entry-2026-09-12)** Q4_0 path and the QAT catalogue entry

### Engine

- **[ENGN-01](https://github.com/tildaslashalef/nuclis/blob/v0.1.0/docs/engineering-log.md#engn-01--library-engine-api-the-engine-seam-moved-into-inference-2026-09-08)** Library engine API: the engine seam moved into `inference`
- **[ENGN-02](https://github.com/tildaslashalef/nuclis/blob/v0.1.0/docs/engineering-log.md#engn-02--chunked-prefill-with-the-batched-matmul-2026-09-08)** Chunked prefill with the batched matmul
- **[ENGN-03](https://github.com/tildaslashalef/nuclis/blob/v0.1.0/docs/engineering-log.md#engn-03--causal-tiled-attention-for-prefill-2026-09-09)** Causal tiled attention for prefill
- **[ENGN-04](https://github.com/tildaslashalef/nuclis/blob/v0.1.0/docs/engineering-log.md#engn-04--chunkwise-deltanet-wy-form-2026-09-09)** Chunkwise DeltaNet (WY form)
- **[ENGN-05](https://github.com/tildaslashalef/nuclis/blob/v0.1.0/docs/engineering-log.md#engn-05--matmul-tile-ceiling-specialized-half-operand-6464-tiles-2026-09-09)** Matmul tile ceiling: specialized half-operand 64×64 tiles
- **[ENGN-06](https://github.com/tildaslashalef/nuclis/blob/v0.1.0/docs/engineering-log.md#engn-06--session-snapshot-and-restore-2026-09-10)** Session snapshot and restore
- **[ENGN-07](https://github.com/tildaslashalef/nuclis/blob/v0.1.0/docs/engineering-log.md#engn-07--32k-acceptance-run-and-benchmark-record-2026-09-10)** 32K acceptance run and benchmark record
- **[ENGN-08](https://github.com/tildaslashalef/nuclis/blob/v0.1.0/docs/engineering-log.md#engn-08--long-context-prefill-attention-2026-09-10)** Long-context prefill attention

### Kernels

- **[KERN-01](https://github.com/tildaslashalef/nuclis/blob/v0.1.0/docs/engineering-log.md#kern-01--specialized-matvec-for-q4_k-q5_k-q6_k-iq4_xs-2026-09-07)** Specialized matvec for Q4_K, Q5_K, Q6_K, IQ4_XS
- **[KERN-02](https://github.com/tildaslashalef/nuclis/blob/v0.1.0/docs/engineering-log.md#kern-02--per-kernel-gpu-profile-observer-checklayer-split-2026-09-07)** Per-kernel GPU profile, `Observer` check/layer split
- **[KERN-03](https://github.com/tildaslashalef/nuclis/blob/v0.1.0/docs/engineering-log.md#kern-03--specialized-matvec-for-q3_k-iq3_s-iq4_nl-deferred-2026-09-08)** Specialized matvec for Q3_K, IQ3_S (IQ4_NL deferred)
- **[KERN-04](https://github.com/tildaslashalef/nuclis/blob/v0.1.0/docs/engineering-log.md#kern-04--merged-projections-via-a-segment-table-2026-09-08)** Merged projections via a segment table
- **[KERN-05](https://github.com/tildaslashalef/nuclis/blob/v0.1.0/docs/engineering-log.md#kern-05--per-block-cost-of-the-specialized-kernels-2026-09-08)** Per-block cost of the specialized kernels
- **[KERN-06](https://github.com/tildaslashalef/nuclis/blob/v0.1.0/docs/engineering-log.md#kern-06--gpu-partial-top-k-for-sampled-decoding-2026-09-08)** GPU partial top-k for sampled decoding
- **[KERN-07](https://github.com/tildaslashalef/nuclis/blob/v0.1.0/docs/engineering-log.md#kern-07--f16-kv-cache-as-a-session-layout-option-2026-09-10)** F16 KV cache as a session layout option
- **[KERN-08](https://github.com/tildaslashalef/nuclis/blob/v0.1.0/docs/engineering-log.md#kern-08--flash-decoding-attention-2026-09-10)** Flash-decoding attention

### Agent

- **[AGNT-01](https://github.com/tildaslashalef/nuclis/blob/v0.1.0/docs/engineering-log.md#agnt-01--completion-events-and-the-tool-seam-2026-09-13--2026-09-14)** Completion events and the tool seam
- **[AGNT-02](https://github.com/tildaslashalef/nuclis/blob/v0.1.0/docs/engineering-log.md#agnt-02--the-agent-loop-state-machine-provisional-parser-read-tools-2026-09-14)** The agent loop: state machine, provisional parser, read tools
- **[AGNT-03](https://github.com/tildaslashalef/nuclis/blob/v0.1.0/docs/engineering-log.md#agnt-03--read-tools-and-bash-2026-09-14)** Read tools and `bash`
- **[AGNT-04](https://github.com/tildaslashalef/nuclis/blob/v0.1.0/docs/engineering-log.md#agnt-04--mutations-and-diffs-2026-09-14)** Mutations and diffs
- **[AGNT-05](https://github.com/tildaslashalef/nuclis/blob/v0.1.0/docs/engineering-log.md#agnt-05--qwen-tool-rendering-and-pinned-fixtures-2026-09-14)** Qwen tool rendering and pinned fixtures
- **[AGNT-06](https://github.com/tildaslashalef/nuclis/blob/v0.1.0/docs/engineering-log.md#agnt-06--tool-call-decoding-and-the-profile-driven-loop-2026-09-14)** Tool-call decoding and the profile-driven loop
- **[AGNT-07](https://github.com/tildaslashalef/nuclis/blob/v0.1.0/docs/engineering-log.md#agnt-07--polish-resume-and-the-agent-plan-closed-2026-09-14)** Polish, resume, and the agent plan closed

### Application

- **[APPS-01](https://github.com/tildaslashalef/nuclis/blob/v0.1.0/docs/engineering-log.md#apps-01----think-reasoning-effort-2026-09-08)** `--think` reasoning effort
- **[APPS-02](https://github.com/tildaslashalef/nuclis/blob/v0.1.0/docs/engineering-log.md#apps-02--nuclis-chat-playground-2026-09-08)** `nuclis chat` playground
- **[APPS-03](https://github.com/tildaslashalef/nuclis/blob/v0.1.0/docs/engineering-log.md#apps-03--engine-configuration-nuclisnuclisjson-2026-09-09)** Engine configuration `~/.nuclis/nuclis.json`
- **[APPS-04](https://github.com/tildaslashalef/nuclis/blob/v0.1.0/docs/engineering-log.md#apps-04--styled-command-output-2026-09-11)** Styled command output
- **[APPS-05](https://github.com/tildaslashalef/nuclis/blob/v0.1.0/docs/engineering-log.md#apps-05--config-init-registers-the-catalogue-chat--agent-2026-09-11)** `config init` registers the catalogue; `chat` → `agent`

### Terminal

- **[TERM-01](https://github.com/tildaslashalef/nuclis/blob/v0.1.0/docs/engineering-log.md#term-01--agent-terminal-surface-2026-09-12)** Agent terminal surface
- **[TERM-02](https://github.com/tildaslashalef/nuclis/blob/v0.1.0/docs/engineering-log.md#term-02--grapheme-correct-width-wrapping-and-cursor-motion-2026-09-14)** Grapheme-correct width, wrapping, and cursor motion
- **[TERM-03](https://github.com/tildaslashalef/nuclis/blob/v0.1.0/docs/engineering-log.md#term-03--osc-8-hyperlinks-in-markdown-2026-09-14)** OSC 8 hyperlinks in markdown
- **[TERM-04](https://github.com/tildaslashalef/nuclis/blob/v0.1.0/docs/engineering-log.md#term-04--focus-tracking-and-turn-complete-notifications-2026-09-14)** Focus tracking and turn-complete notifications

### Repository

- **[REPO-01](https://github.com/tildaslashalef/nuclis/blob/v0.1.0/docs/engineering-log.md#repo-01--repository-restructure-standalone-src--inference-2026-09-08)** Repository restructure: standalone `src/` + `inference/`
- **[REPO-02](https://github.com/tildaslashalef/nuclis/blob/v0.1.0/docs/engineering-log.md#repo-02--reference-oracle-promoted-to-committed-testsfixtures-2026-09-08)** Reference oracle promoted to committed `tests/fixtures/`
- **[REPO-03](https://github.com/tildaslashalef/nuclis/blob/v0.1.0/docs/engineering-log.md#repo-03--version-wiring-nuclis---version-from-the-manifest-2026-09-08)** Version wiring: `nuclis --version` from the manifest
- **[REPO-04](https://github.com/tildaslashalef/nuclis/blob/v0.1.0/docs/engineering-log.md#repo-04--comment-and-identifier-hygiene-2026-09-14)** Comment and identifier hygiene

### Outside units

- import nuclis v0.1

<details><summary>All 2 commits</summary>

- [`1a31568`](https://github.com/tildaslashalef/nuclis/commit/1a31568d2114624d8e41503ac3fc004af01a3950) chore(release): v0.1.0
- [`cf76bc0`](https://github.com/tildaslashalef/nuclis/commit/cf76bc0643fb28b9a9ef8de11a09a3837dcee99e) feat: import nuclis v0.1

</details>
