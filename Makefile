# nuclis task runner: thin wrappers around `zig build`. Run `make help`.
#
# Variables (override on the command line, e.g. `make bench MODEL=/x.gguf`):
#   MODEL     GGUF path for full-model targets (default: the pinned artifact)
#   OPT       Zig optimize mode for release/metal targets (ReleaseSafe)
#   BACKEND   cpu | metal for run targets (metal)
#   PROMPT    prompt text for run/bench targets
#   ARGS      extra arguments appended to run/bench/generate

ZIG      ?= zig
CACHE    ?= --global-cache-dir .zig-cache/global
OPT      ?= ReleaseSafe
MODEL    ?= $(HOME)/.nuclis/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf
BACKEND  ?= metal
VARIANT  ?= baseline
PROMPT   ?= Write a Zig function that reverses a string.
ARGS     ?=
BIN      := ./zig-out/bin/nuclis
METAL    := -Dmetal=true -Doptimize=$(OPT) $(CACHE)

.DEFAULT_GOAL := build
.PHONY: help build debug build-cpu metal test test-metal check verify verify-long verify-cpu verify-changed gate gates-list gates-validate \
        fmt fmt-check inspect validate generate bench bench-profile bench-kernels bench-matvec-split bench-matmul bench-matvec-rows bench-hadamard bench-experts bench-attention \
        workload workloads-list workloads-validate \
        agent agent-eval model-ls eval-corpus trace clean distclean hf-downloader test-hf changelog release

help: ## Show this help
	@awk 'BEGIN{FS=":.*##"} /^[a-zA-Z_-]+:.*##/{printf "  \033[36m%-24s\033[0m %s\n",$$1,$$2}' $(MAKEFILE_LIST)
	@echo; echo "Variables: MODEL OPT BACKEND PROMPT ARGS  (see Makefile header)"

# ---- build -----------------------------------------------------------------

build: ## Release build (Metal is on by default on Apple Silicon)
	$(ZIG) build -Doptimize=$(OPT) $(CACHE)

debug: ## Debug build
	$(ZIG) build $(CACHE)

build-cpu: ## Release build without the Metal backend (-Dmetal=false)
	$(ZIG) build -Doptimize=$(OPT) -Dmetal=false $(CACHE)

metal: ## Release build with the Metal backend stated explicitly
	$(ZIG) build $(METAL)

# ---- tests -----------------------------------------------------------------

test: ## Default unit tests (no model, no GPU)
	$(ZIG) build test $(CACHE) --summary all

test-metal: ## GPU kernel fixture/lifecycle checks (needs a Metal device, no model)
	$(ZIG) build test-metal $(METAL)

check: fmt-check test test-metal gates-validate workloads-validate ## Format check + unit tests + GPU fixtures + the two manifests (seconds)

# ---- gates (gates.json; docs/development.md § Gates) ----------------------
# Every model-specific numerical check is a gate in the manifest: a tier
# (verify = Metal, minutes; verify-cpu = the CPU reference, hours) and the
# source globs that make it relevant. `make verify-changed BASE=<rev>` runs
# what a change needs.

BASE ?= HEAD
verify: eval-corpus ## The Metal tier: every trace, generation, speculative, vocabulary, and perplexity gate (once per unit)
	python3 scripts/gates.py --tier verify $(ARGS)

verify-long: eval-corpus ## The long-context tier: 4K-token perplexity (Gemma 4 12B so far), when a unit touches attention, the caches, or a windowed schedule, and before a release
	python3 scripts/gates.py --tier verify-long $(ARGS)

verify-cpu: ## The CPU-reference tier (hours): when a unit changes what the CPU reference computes, and before a release
	python3 scripts/gates.py --tier verify-cpu $(ARGS)

verify-changed: eval-corpus ## The Metal-tier gates whose paths match `git diff --name-only $(BASE)` plus untracked files (BASE=HEAD); ARGS='--tier verify-long' or '--tier verify-cpu' for those tiers'
	python3 scripts/gates.py --changed $(BASE) $(ARGS)

gate: eval-corpus ## Gates by name or glob: make gate NAME=gemma4-qat-trace-f16, NAME='muse-*' (ARGS=--dry-run prints the commands)
	python3 scripts/gates.py --gate $(NAME) $(ARGS)

gates-list: ## Every gate with its tier, family, model, and evidence
	python3 scripts/gates.py --list

gates-validate: ## Validate gates.json and run the runner's and the perplexity reference's self-tests (no model)
	python3 scripts/gates.py --validate
	python3 scripts/reference-perplexity.py --self-test

# ---- workloads (workloads.json; docs/development.md § The record) ---------
# Every benchmark workload is data: one `nuclis bench` invocation (or the
# acceptance script) with its model from gates.json; reports are saved under
# .zig-cache/bench/<workload>/ and scripts/bench-report.py makes the tables.

workload: ## Run workloads by name or glob: make workload NAME=gemma4-qat/prose512-draft, NAME='qwen38/spec/*' (ARGS=--dry-run)
	python3 scripts/workloads.py --run $(NAME) $(ARGS)

workloads-list: ## Every workload with its model's status, shape, and evidence
	python3 scripts/workloads.py --list

workloads-validate: ## Validate workloads.json and run the driver's and the report script's self-tests (no model)
	python3 scripts/workloads.py --validate
	python3 scripts/bench-report.py --self-test

# ---- formatting ------------------------------------------------------------

fmt: ## Format Zig sources and build files in place
	$(ZIG) fmt build.zig build.zig.zon src/ inference/ huggingface/

fmt-check: ## Fail if any Zig source is not formatted
	$(ZIG) fmt --check build.zig build.zig.zon src/ inference/ huggingface/

# ---- run (always the freshly built binary) --------------------------------

inspect: metal ## Inspect the model container
	$(BIN) inspect --model "$(MODEL)" $(ARGS)

validate: metal ## Validate the Qwen structural profile
	$(BIN) validate --model "$(MODEL)" $(ARGS)

generate: metal ## Generate text: make generate PROMPT='...' BACKEND=cpu|metal ARGS='--max-tokens 64'
	$(BIN) generate --backend $(BACKEND) --model "$(MODEL)" --prompt "$(PROMPT)" $(ARGS)

bench: metal ## Repeated prefill/decode measurement (see docs/reference/bench.md)
	$(BIN) bench --backend $(BACKEND) --model "$(MODEL)" --prompt "$(PROMPT)" --max-tokens 64 --ctx-size 2048 $(ARGS)

bench-profile: metal ## Per-kernel GPU time per token from GPU timestamps (diagnostic; see docs/reference/bench.md)
	$(BIN) bench --backend metal --model "$(MODEL)" --prompt "$(PROMPT)" --max-tokens 64 --ctx-size 2048 --profile $(ARGS)

bench-kernels: ## Achieved GB/s of each matvec kernel on model-shaped matrices, or one with ARGS=<ENCODING> (no model)
	$(ZIG) build bench-kernels $(METAL) $(if $(ARGS),-- $(ARGS))

bench-matvec-split: ## Split-K matvec GB/s on the row-poor shapes at 1/2/4/8 splits, or ARGS=<ENCODING> (no model)
	$(ZIG) build bench-matvec-split $(METAL) $(if $(ARGS),-- $(ARGS))

bench-matmul: ## Throughput of the batched prefill matmul on model shapes, 256 tokens or ARGS=<tokens> (no model)
	$(ZIG) build bench-matmul $(METAL) $(if $(ARGS),-- $(ARGS))

bench-matvec-rows: ## Multi-row matvec vs the 16x8 tile at 1-8 rows, or ARGS=<max rows> (no model)
	$(ZIG) build bench-matvec-rows $(METAL) $(if $(ARGS),-- $(ARGS))

bench-hadamard: ## GPU time of one token's 258 Hadamard transforms on the Bonsai schedule, and per block (no model; docs/reference/metal-backend.md)
	$(ZIG) build bench-hadamard $(METAL)

bench-experts: ## GB/s of the gathered expert kernels on the 26B-A4B shape, 8 of 128 experts; prefill tiles over ARGS tokens (no model)
	$(ZIG) build bench-experts $(METAL) $(if $(ARGS),-- $(ARGS))

bench-attention: ## Prefill chunk attention, row-split vs register-reuse, 4K-32K visible and the verify-shaped counts (no model)
	$(ZIG) build bench-attention $(METAL)

agent: metal ## Interactive agent surface on the engine (Metal by default; ARGS="--think low")
	$(BIN) agent --backend $(BACKEND) --model "$(MODEL)" $(ARGS)

agent-eval: build ## The agent's task list against the playground: make agent-eval VARIANT=name ARGS='--system-prompt p.txt --seeds 1,2' (scripts/agent-eval.py --help)
	python3 scripts/agent-eval.py --variant $(VARIANT) $(ARGS)

shot: metal ## Drive `nuclis agent` in tmux and capture its screen: make shot ARGS='until=ready,60 capture=idle --stop' (scripts/tui-shot.py --help)
	python3 scripts/tui-shot.py --command "$(BIN) agent --backend $(BACKEND) --model $(MODEL)" $(ARGS)

model-ls: build ## List the GGUF files under <root>/models with their provenance sidecars
	$(BIN) model ls $(ARGS)

# ---- the perplexity gates' text (docs/reference/eval.md) --------------------
# Fetched, never committed; the pinned references name its SHA-256 and
# `nuclis eval --reference` refuses any other file.

EVAL_DIR := .zig-cache/eval
EVAL_TEXT := $(EVAL_DIR)/wikitext-2-raw/wiki.test.raw
eval-corpus: ## Fetch wikitext-2-raw (the perplexity gates' text) into .zig-cache/eval and check its SHA-256
	@mkdir -p $(EVAL_DIR)
	@test -f $(EVAL_TEXT) || (curl -fsSL -o $(EVAL_DIR)/wikitext-2-raw-v1.zip \
	  https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip && \
	  unzip -o -q $(EVAL_DIR)/wikitext-2-raw-v1.zip -d $(EVAL_DIR))
	@echo "173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08  $(EVAL_TEXT)" | shasum -a 256 -c -

# ---- GPU trace (Xcode Instruments via xctrace; see AGENTS.md § Local toolchain) --

XCODE_DEV ?= /Applications/Xcode.app/Contents/Developer
TRACE_OUT ?= .zig-cache/trace/bench.trace
trace: metal ## Metal System Trace of one `bench` run under xctrace (needs Xcode.app; output under .zig-cache/trace)
	mkdir -p $(dir $(TRACE_OUT)) && rm -rf "$(TRACE_OUT)"
	DEVELOPER_DIR=$(XCODE_DEV) xcrun xctrace record --template 'Metal System Trace' --instrument 'Metal GPU Counters' \
	  --output "$(TRACE_OUT)" --no-prompt --launch -- $(BIN) bench --backend metal --model "$(MODEL)" \
	  --prompt "$(PROMPT)" --max-tokens 64 --ctx-size 2048 --repeat 1 --warmup 1 $(ARGS)
	DEVELOPER_DIR=$(XCODE_DEV) xcrun xctrace export --input "$(TRACE_OUT)" --toc | grep -o '<table schema="metal-gpu[^"]*"' | sort -u

# ---- huggingface package (a path dependency of the root build since MODL-02) -----

hf-downloader: ## Build the package's standalone downloader (zig-out/bin/hf-downloader)
	$(ZIG) build hf-downloader -Doptimize=$(OPT) $(CACHE)

test-hf: ## Offline tests of the huggingface package only (`make test` includes them)
	$(ZIG) build test-hf $(CACHE)

# ---- releases --------------------------------------------------------------

changelog: ## Insert the CHANGELOG section since the previous tag (ARGS='vX.Y.Z')
	python3 scripts/changelog.py $(ARGS)

release: ## Cut a release from the manifest's -dev version: check, changelog, commit, tag, next-dev bump (DRY_RUN=1 to preview)
	python3 scripts/release.py $(if $(DRY_RUN),--dry-run)

# ---- housekeeping ----------------------------------------------------------

clean: ## Remove build outputs (keeps the cache and the reference checkout)
	rm -rf zig-out

distclean: clean ## Also remove .zig-cache (deletes the llama.cpp reference build and traces)
	rm -rf .zig-cache
