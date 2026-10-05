# nuclis task runner: thin wrappers around `zig build`. Run `make help`.
#
# Variables (override on the command line, e.g. `make bench MODEL=/x.gguf`):
#   MODEL     GGUF path for full-model targets (default: the pinned artifact)
#   OPT       Zig optimize mode for release/metal targets (ReleaseSafe)
#   BACKEND   cpu | metal for run targets (metal)
#   PROMPT    prompt text for run/bench targets
#   ARGS      extra arguments appended to run/bench/generate
#   PREFIX    install root for `make install` ($(HOME)/.local: the binary goes to $(PREFIX)/bin)

ZIG      ?= zig
# `zig build` reads the global cache location from the environment only.
export ZIG_GLOBAL_CACHE_DIR ?= $(CURDIR)/.zig-cache/global
OPT      ?= ReleaseSafe
MODEL    ?= $(HOME)/.nuclis/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf
BACKEND  ?= metal
VARIANT  ?= baseline
PROMPT   ?= Write a Zig function that reverses a string.
ARGS     ?=
PREFIX   ?= $(HOME)/.local
BIN      := ./zig-out/bin/nuclis
METAL    := -Dmetal=true -Doptimize=$(OPT)

.DEFAULT_GOAL := build
.PHONY: help build debug build-cpu metal install uninstall test test-metal check verify-auto verify verify-release verify-long verify-cpu verify-changed gate gates-list gates-validate \
        fmt fmt-check fmt-py lint-py inspect validate generate bench bench-profile bench-kernels bench-matvec-split bench-matmul bench-matvec-rows bench-hadamard bench-experts bench-attention \
        workload workloads-list workloads-validate speed speed-base spec-matrix \
        agent agent-eval playground model-ls eval-corpus trace capture clean clean-cache distclean hf-downloader test-hf changelog release site-check site-serve docs-check

help: ## Show this help
	@awk 'BEGIN{FS=":.*##"} /^[a-zA-Z_-]+:.*##/{printf "  \033[36m%-24s\033[0m %s\n",$$1,$$2}' $(MAKEFILE_LIST)
	@echo; echo "Variables: MODEL OPT BACKEND PROMPT ARGS PREFIX  (see Makefile header)"

# ---- build -----------------------------------------------------------------

build: ## Release build (Metal is on by default on Apple Silicon)
	$(ZIG) build -Doptimize=$(OPT)

debug: ## Debug build
	$(ZIG) build

build-cpu: ## Release build without the Metal backend (-Dmetal=false)
	$(ZIG) build -Doptimize=$(OPT) -Dmetal=false

metal: ## Release build with the Metal backend stated explicitly
	$(ZIG) build $(METAL)

# The old binary is removed before the copy, never overwritten in place: on
# Apple Silicon the kernel keeps the old file's code signature cached, and a
# binary rewritten under it is killed at launch.
install: metal ## Install the Metal build as PREFIX/bin/nuclis (PREFIX=~/.local by default)
	@mkdir -p "$(PREFIX)/bin"
	@rm -f "$(PREFIX)/bin/nuclis"
	@cp $(BIN) "$(PREFIX)/bin/nuclis"
	@echo "installed $(PREFIX)/bin/nuclis ($$("$(PREFIX)/bin/nuclis" --version))"
	@case ":$$PATH:" in *":$(PREFIX)/bin:"*) ;; *) echo "note: $(PREFIX)/bin is not on PATH";; esac
	@echo "shell completion: nuclis completion --help"

uninstall: ## Remove PREFIX/bin/nuclis
	rm -f "$(PREFIX)/bin/nuclis"

# ---- tests -----------------------------------------------------------------

test: ## Default unit tests (no model, no GPU)
	$(ZIG) build test --summary all

test-metal: ## GPU kernel fixture/lifecycle checks (needs a Metal device, no model)
	$(ZIG) build test-metal $(METAL)

check: fmt-check test test-metal gates-validate workloads-validate ## Format check + unit tests + GPU fixtures + the two manifests (seconds)

# ---- gates (gates.json; docs/development.md § Gates) ----------------------
# Every model-specific numerical check is a gate in the manifest: a tier
# (verify = Metal, minutes; verify-release = whole files, before a release;
# verify-cpu = the CPU reference, hours) and the
# source globs that make it relevant. `make verify-auto` runs what a change
# needs: the model-free checks and gates its paths select, then names the
# tiers it requires.

BASE ?= HEAD
verify-auto: eval-corpus ## What a change needs: the model-free checks and fast-tier gates its paths select (cheapest first, stop at the first failure), then the tiers it requires; BASE= defaults to TODO.md's `Base:` line
	python3 scripts/gates.py --auto $(if $(filter command line,$(origin BASE)),$(BASE)) $(ARGS)

verify: eval-corpus ## The fast Metal tier: one representative file per family and the paths only a variant has (once per unit)
	python3 scripts/gates.py --tier verify $(ARGS)

verify-release: eval-corpus ## Whole-file acceptance before a release: 8-window perplexities, the Gemma 12B QAT file, draft statistics
	python3 scripts/gates.py --tier verify-release --strict $(ARGS)

verify-long: eval-corpus ## The long-context tier: 4K-token perplexity (Gemma 4 E4B so far), when a unit touches attention, the caches, or a windowed schedule, and before a release
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

workloads-validate: ## Validate workloads.json and run the driver's, the report script's, the speed loop's, and the speculative matrix's self-tests (no model)
	python3 scripts/workloads.py --validate
	python3 scripts/bench-report.py --self-test
	python3 scripts/speed.py --self-test
	python3 scripts/spec-matrix.py --self-test

# ---- the speed loop (scripts/speed.py; docs/development.md § The speed loop) --
# A saved base binary against the tree, interleaved on saved prefixes under
# .zig-cache/speed/; the keep rule decides.

speed-base: metal ## Save the built binary as the speed loop's base (.zig-cache/speed/base, with its revision)
	python3 scripts/speed.py --save-base

speed: metal ## Interleaved A/B of the tree against the base: make speed ARGS='--contexts 512,4096 --verify-rows 1,4,8 --model qwen38'
	python3 scripts/speed.py $(ARGS)

spec-matrix: metal ## Real speculation, off/on pairs per cell: make spec-matrix ARGS='--model qwen38 --contexts 512,4096,code --drafts 2-7'
	python3 scripts/spec-matrix.py $(ARGS)

# ---- formatting ------------------------------------------------------------

fmt: ## Format Zig sources and build files in place
	$(ZIG) fmt build.zig build.zig.zon src/ inference/ huggingface/

fmt-check: ## Fail if any Zig source is not formatted
	$(ZIG) fmt --check build.zig build.zig.zon src/ inference/ huggingface/

# The Python scripts' tools, pinned and fetched by uv (pyproject.toml).
RUFF    := uvx ruff@0.14.11
PYRIGHT := uvx basedpyright@1.40.1

fmt-py: ## Format the Python scripts in place (ruff)
	$(RUFF) format scripts

docs-check: ## Every documentation link and anchor resolves: Markdown, the docs/ paths in JSON, scripts, and Zig, the site's repository links (no network); ARGS='--move old=new' moves documents and rewrites them
	python3 scripts/docs-check.py --self-test
	python3 scripts/docs-check.py $(ARGS)

site-check: ## Check the nuclis.dev site in site/: references, head tags, CSP fit, versions, and the README's figures (no model, no network)
	python3 scripts/site-check.py --self-test
	python3 scripts/site-check.py

site-serve: ## Preview the nuclis.dev site at http://localhost:8000 (Cloudflare's _headers are not applied)
	python3 -m http.server 8000 --directory site

lint-py: ## Lint, format-check, and type-check the Python scripts (ruff, basedpyright)
	$(RUFF) check scripts
	$(RUFF) format --check scripts
	$(PYRIGHT) scripts

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

# CAPTURE=<substring> on bench-kernels, bench-matvec-rows, bench-attention:
# each matching case's last warm-up command buffer into
# .zig-cache/trace/kernels/<label>.gputrace (small; open in Xcode for counters).
CAPTURE_ENV = $(if $(CAPTURE),rm -rf .zig-cache/trace/kernels && mkdir -p .zig-cache/trace/kernels && MTL_CAPTURE_ENABLED=1 NUCLIS_CAPTURE="$(CAPTURE)")

bench-kernels: ## Achieved GB/s of each matvec kernel on model-shaped matrices, or one with ARGS=<ENCODING> (no model)
	$(CAPTURE_ENV) $(ZIG) build bench-kernels $(METAL) $(if $(ARGS),-- $(ARGS))

bench-matvec-split: ## Split-K matvec GB/s on the row-poor shapes at 1/2/4/8 splits, or ARGS=<ENCODING> (no model)
	$(ZIG) build bench-matvec-split $(METAL) $(if $(ARGS),-- $(ARGS))

bench-matmul: ## Throughput of the batched prefill matmul on model shapes, 256 tokens or ARGS=<tokens> (no model)
	$(ZIG) build bench-matmul $(METAL) $(if $(ARGS),-- $(ARGS))

bench-matvec-rows: ## Multi-row matvec vs the 16x8 tile at 1-8 rows, or ARGS=<max rows> (no model)
	$(CAPTURE_ENV) $(ZIG) build bench-matvec-rows $(METAL) $(if $(ARGS),-- $(ARGS))

bench-hadamard: ## GPU time of one token's 258 Hadamard transforms on the Bonsai schedule, and per block (no model; docs/engine/metal-backend.md)
	$(ZIG) build bench-hadamard $(METAL)

bench-experts: ## GB/s of the gathered expert kernels on the 26B-A4B shape, 8 of 128 experts; prefill tiles over ARGS tokens (no model)
	$(ZIG) build bench-experts $(METAL) $(if $(ARGS),-- $(ARGS))

bench-attention: ## Prefill chunk attention, row-split vs register-reuse, 4K-32K visible and the verify-shaped counts (no model)
	$(CAPTURE_ENV) $(ZIG) build bench-attention $(METAL)

agent: metal ## Interactive agent surface on the engine (Metal by default; ARGS="--think low")
	$(BIN) agent --backend $(BACKEND) --model "$(MODEL)" $(ARGS)

playground: ## Generate the agent's scratch project under .zig-cache/playground/ (scripts/playground.py) and print its path
	python3 scripts/playground.py

agent-eval: build ## The agent's task list against the playground: make agent-eval VARIANT=name ARGS='--system-prompt p.txt --seeds 1,2' (scripts/agent-eval.py --help)
	python3 scripts/agent-eval.py --variant $(VARIANT) $(ARGS)

shot: metal ## Drive `nuclis agent` in tmux and capture its screen: make shot ARGS='until=ready,60 capture=idle --stop' (scripts/tui-shot.py --help)
	python3 scripts/tui-shot.py --command "$(BIN) agent --backend $(BACKEND) --model $(MODEL)" $(ARGS)

model-ls: build ## List the model files (GGUF, safetensors) under <root>/models with their provenance sidecars
	$(BIN) model ls $(ARGS)

# ---- the perplexity gates' text (docs/guide/eval.md) --------------------
# Fetched, never committed; the pinned references name its SHA-256 and
# `nuclis eval --reference` refuses any other file.

EVAL_DIR := .reference/eval
EVAL_TEXT := $(EVAL_DIR)/wikitext-2-raw/wiki.test.raw
eval-corpus: ## Fetch wikitext-2-raw (the perplexity gates' text) into .reference/eval and check its SHA-256
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

CAPTURE_OUT ?= .zig-cache/trace/decode.gputrace
capture: metal ## One decode step captured into a .gputrace for Xcode's Metal debugger (counters, limiters, occupancy; output under .zig-cache/trace)
	mkdir -p $(dir $(CAPTURE_OUT)) && rm -rf "$(CAPTURE_OUT)"
	MTL_CAPTURE_ENABLED=1 $(BIN) bench --backend metal --model "$(MODEL)" --prompt "$(PROMPT)" --max-tokens 8 \
	  --ctx-size 2048 --repeat 1 --warmup 0 --capture "$(CAPTURE_OUT)" $(ARGS)
	@echo "open $(CAPTURE_OUT)   # Xcode: Metal debugger, Performance / Counters"

# ---- huggingface package (a path dependency of the root build) -----------------

hf-downloader: ## Build the package's standalone downloader (zig-out/bin/hf-downloader)
	$(ZIG) build hf-downloader -Doptimize=$(OPT)

test-hf: ## Offline tests of the huggingface package only (`make test` includes them)
	$(ZIG) build test-hf

# ---- releases --------------------------------------------------------------

changelog: ## Insert the CHANGELOG section since the previous tag (ARGS='vX.Y.Z [--dry-run] [--range A..B]')
	python3 scripts/changelog.py --self-test
	python3 scripts/changelog.py $(ARGS)

release: ## Cut a release from the manifest's -dev version: highlights, check, changelog, commit, tag, next-dev bump (DRY_RUN=1 to preview, HIGHLIGHTS=file)
	python3 scripts/release.py $(if $(DRY_RUN),--dry-run) $(if $(HIGHLIGHTS),--highlights-file $(HIGHLIGHTS))

# ---- housekeeping ----------------------------------------------------------

clean: ## Remove build outputs (keeps the caches and .reference/)
	rm -rf zig-out

clean-cache: ## Remove Zig's build cache, the part of .zig-cache that grows with every build
	rm -rf .zig-cache/o .zig-cache/h .zig-cache/z .zig-cache/tmp .zig-cache/global

distclean: clean ## Also remove .zig-cache (builds, traces, results; .reference/ is kept)
	rm -rf .zig-cache
