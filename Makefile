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
PROMPT   ?= Write a Zig function that reverses a string.
ARGS     ?=
BIN      := ./zig-out/bin/nuclis
METAL    := -Dmetal=true -Doptimize=$(OPT) $(CACHE)

.DEFAULT_GOAL := build
.PHONY: help build debug build-cpu metal test test-metal test-generation test-generation-metal \
        compare-draft compare-draft-metal compare-draft-cpu draft-stats \
        speculative-check speculative-check-metal speculative-record \
        test-vocabulary check fmt fmt-check inspect validate generate bench bench-profile bench-kernels bench-matmul bench-hadamard bench-experts \
        baseline baseline-gemma4-qat baseline-gemma4 baseline-gemma4-26b-a4b agent model-ls trace compare compare-f32 compare-f16 \
        compare-gemma4-qat compare-gemma4-qat-cpu compare-gemma4-qat-f32 compare-gemma4-qat-f16 \
        compare-gemma4 compare-gemma4-cpu compare-gemma4-f32 compare-gemma4-f16 \
        compare-gemma4-26b-a4b compare-gemma4-26b-a4b-cpu compare-gemma4-26b-a4b-f32 compare-gemma4-26b-a4b-f16 \
        compare-bonsai compare-bonsai-cpu compare-bonsai-f32 compare-bonsai-f16 test-generation-bonsai-metal baseline-bonsai \
        compare-muse-glimmer compare-muse-glimmer-cpu compare-muse-glimmer-f32 compare-muse-glimmer-f16 test-generation-muse-glimmer-metal baseline-muse-glimmer \
        test-generation-gemma4 test-generation-gemma4-metal test-generation-gemma4-qat-metal test-generation-gemma4-26b-a4b-metal clean distclean hf-downloader test-hf changelog release

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

test-generation: ## Full-model session isolation/reset check on the CPU reference (slow)
	$(ZIG) build test-generation -Doptimize=$(OPT) $(CACHE) -- "$(MODEL)"

test-generation-metal: ## Same protocol on the GPU plan
	$(ZIG) build test-generation $(METAL) -- "$(MODEL)" --metal

# The embedded prediction block's `Hello,` rows against the pinned reference
# trace (`inference/src/models/fixtures/qwen35-mtp/`), on each backend.
compare-draft: compare-draft-metal compare-draft-cpu ## The prediction block's trace rows vs the reference (MODL-18)

compare-draft-metal: metal ## Native GPU prediction-block rows vs the pinned reference trace
	rm -rf "$(TRACE)-draft-metal" && mkdir -p "$(TRACE)-draft-metal"
	$(ZIG) build test-generation $(METAL) -- "$(MODEL)" --metal --draft-trace "$(TRACE)-draft-metal"
	python3 scripts/compare-generation.py "$(TRACE)-draft-metal" inference/src/models/fixtures/qwen35-mtp --positions 2 --draft \
	  | python3 -c 'import json,sys; d=json.load(sys.stdin); c=d["comparisons"]; print("draft metal", "passed", d["passed"], "rows", len(c), "max abs", max(x["max_absolute"] for x in c), "max rel rms", max(x["relative_rms"] for x in c), "greedy", d["native_greedy"])'

compare-draft-cpu: ## Native CPU reference prediction-block rows vs the pinned reference trace
	rm -rf "$(TRACE)-draft-cpu" && mkdir -p "$(TRACE)-draft-cpu"
	$(ZIG) build test-generation -Doptimize=$(OPT) $(CACHE) -- "$(MODEL)" --draft-trace "$(TRACE)-draft-cpu"
	python3 scripts/compare-generation.py "$(TRACE)-draft-cpu" inference/src/models/fixtures/qwen35-mtp --positions 2 --draft \
	  | python3 -c 'import json,sys; d=json.load(sys.stdin); c=d["comparisons"]; print("draft cpu", "passed", d["passed"], "rows", len(c), "max abs", max(x["max_absolute"] for x in c), "max rel rms", max(x["relative_rms"] for x in c), "greedy", d["native_greedy"])'

draft-stats: metal ## Per-depth acceptance of the embedded prediction head on the fixed prompts (MODL-18)
	$(ZIG) build test-generation $(METAL) -- "$(MODEL)" --metal --draft-stats

speculative-check: ## Greedy speculation vs ordinary greedy on the CPU reference (very slow)
	$(ZIG) build test-generation -Doptimize=$(OPT) $(CACHE) -- "$(MODEL)" --speculative-check

speculative-check-metal: metal ## Greedy speculation vs ordinary greedy on the Metal plan (ENGN-12)
	$(ZIG) build test-generation $(METAL) -- "$(MODEL)" --metal --speculative-check

test-generation-gemma4: ## The generation check on gemma-4-12b, CPU reference (very slow: ~35 s per token)
	$(ZIG) build test-generation -Doptimize=$(OPT) $(CACHE) -- "$(GEMMA_MODEL)"

test-generation-gemma4-metal: ## The generation check on gemma-4-12b, Metal plan (MODL-06)
	$(ZIG) build test-generation $(METAL) -- "$(GEMMA_MODEL)" --metal

test-generation-gemma4-qat-metal: ## The same on gemma-4-12b-qat, whose Q4_0 path amplifies the half-operand rounding (MODL-08)
	$(ZIG) build test-generation $(METAL) -- "$(GEMMA_QAT_MODEL)" --metal

test-generation-gemma4-26b-a4b-metal: ## The generation check on gemma-4-26b-a4b (mixture of experts), Metal plan (MODL-09)
	$(ZIG) build test-generation $(METAL) -- "$(GEMMA_26B_A4B_MODEL)" --metal

test-vocabulary: ## Tokenizer check against the real artifact
	$(ZIG) build test-vocabulary -Doptimize=$(OPT) $(CACHE) -- "$(MODEL)"

check: fmt-check test test-metal ## Format check + unit tests + GPU fixtures

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

baseline: metal ## The reference workload (512/4K/16K/32,639 tokens, 128 out) on the committed token arrays; writes docs/benchmarks/nuclis-<date>.json (see docs/reference/bench.md § Acceptance runs)
	python3 scripts/nuclis-baseline.py --model "$(MODEL)" --nuclis $(BIN) $(ARGS)

baseline-gemma4-qat: metal ## The same workload on gemma-4-12b-qat (Q4_0) with its own reference arrays (tests/fixtures/run-2026-09-12-gemma4-qat); writes docs/benchmarks/nuclis-<date>-gemma4-qat.json
	python3 scripts/nuclis-baseline.py --model "$(GEMMA_QAT_MODEL)" --nuclis $(BIN) --run run-2026-09-12-gemma4-qat \
	  --reference-records reference-2026-09-12-gemma4-qat.json --output docs/benchmarks/nuclis-$$(date +%F)-gemma4-qat.json $(ARGS)

baseline-gemma4: metal ## The same on gemma-4-12b (the K-quant entry) (tests/fixtures/run-2026-09-12-gemma4); writes docs/benchmarks/nuclis-<date>-gemma4.json
	python3 scripts/nuclis-baseline.py --model "$(GEMMA_MODEL)" --nuclis $(BIN) --run run-2026-09-12-gemma4 \
	  --reference-records reference-2026-09-12-gemma4.json --output docs/benchmarks/nuclis-$$(date +%F)-gemma4.json $(ARGS)

baseline-gemma4-26b-a4b: metal ## The same on gemma-4-26b-a4b (the mixture of experts) with its own reference arrays (tests/fixtures/run-2026-09-18-gemma4-26b-a4b); writes docs/benchmarks/nuclis-<date>-gemma4-26b-a4b.json
	python3 scripts/nuclis-baseline.py --model "$(GEMMA_26B_A4B_MODEL)" --nuclis $(BIN) --run run-2026-09-18-gemma4-26b-a4b \
	  --reference-records reference-2026-09-18-gemma4-26b-a4b.json --output docs/benchmarks/nuclis-$$(date +%F)-gemma4-26b-a4b.json $(ARGS)

baseline-muse-glimmer: metal ## The same workload on muse-glimmer-30b with its own reference arrays (tests/fixtures/run-2026-09-19-muse-glimmer); writes docs/benchmarks/nuclis-<date>-muse-glimmer.json
	python3 scripts/nuclis-baseline.py --model "$(MUSE_MODEL)" --nuclis $(BIN) --run run-2026-09-19-muse-glimmer \
	  --reference-records reference-2026-09-19-muse-glimmer.json --output docs/benchmarks/nuclis-$$(date +%F)-muse-glimmer.json $(ARGS)

bench-kernels: ## Achieved GB/s of each matvec kernel on model-shaped matrices, or one with ARGS=<ENCODING> (no model)
	$(ZIG) build bench-kernels $(METAL) $(if $(ARGS),-- $(ARGS))

bench-matmul: ## Throughput of the batched prefill matmul on model shapes, 256 tokens or ARGS=<tokens> (no model)
	$(ZIG) build bench-matmul $(METAL) $(if $(ARGS),-- $(ARGS))

bench-hadamard: ## GPU time of one token's 258 Hadamard transforms on the Bonsai schedule, and per block (no model; docs/reference/metal-backend.md)
	$(ZIG) build bench-hadamard $(METAL)

bench-experts: ## GB/s of the gathered expert kernels on the 26B-A4B shape, 8 of 128 experts; prefill tiles over ARGS tokens (no model)
	$(ZIG) build bench-experts $(METAL) $(if $(ARGS),-- $(ARGS))

agent: metal ## Interactive agent surface on the engine (Metal by default; ARGS="--think low")
	$(BIN) agent --backend $(BACKEND) --model "$(MODEL)" $(ARGS)

model-ls: build ## List the GGUF files under <root>/models with their provenance sidecars
	$(BIN) model ls $(ARGS)

# ---- GPU trace (Xcode Instruments via xctrace; see AGENTS.md § Local toolchain) --

XCODE_DEV ?= /Applications/Xcode.app/Contents/Developer
TRACE_OUT ?= .zig-cache/trace/bench.trace
trace: metal ## Metal System Trace of one `bench` run under xctrace (needs Xcode.app; output under .zig-cache/trace)
	mkdir -p $(dir $(TRACE_OUT)) && rm -rf "$(TRACE_OUT)"
	DEVELOPER_DIR=$(XCODE_DEV) xcrun xctrace record --template 'Metal System Trace' --instrument 'Metal GPU Counters' \
	  --output "$(TRACE_OUT)" --no-prompt --launch -- $(BIN) bench --backend metal --model "$(MODEL)" \
	  --prompt "$(PROMPT)" --max-tokens 64 --ctx-size 2048 --repeat 1 --warmup 1 $(ARGS)
	DEVELOPER_DIR=$(XCODE_DEV) xcrun xctrace export --input "$(TRACE_OUT)" --toc | grep -o '<table schema="metal-gpu[^"]*"' | sort -u

# ---- numerical comparison against the pinned reference ---------------------

TRACE ?= .zig-cache/generation/make-check
# $(1) cache precision, $(2) max absolute, $(3) max relative RMS per trace file.
define compare_run
	rm -rf "$(TRACE)-$(1)" && mkdir -p "$(TRACE)-$(1)"
	$(BIN) generate --backend $(BACKEND) --model "$(MODEL)" --raw --prompt 'Hello,' --max-tokens 1 --ctx-size 8 --kv $(1) \
	  --logits "$(TRACE)-$(1)/logits.f32" --trace-dir "$(TRACE)-$(1)" $(ARGS) > /dev/null
	python3 scripts/compare-generation.py "$(TRACE)-$(1)" tests/fixtures/reference-hello-comma --positions 2 --max-absolute $(2) --max-relative-rms $(3) \
	  | python3 -c 'import json,sys; d=json.load(sys.stdin); c=d["comparisons"]; print("$(1)", "passed", d["passed"], "files", len(c), "max abs", max(x["max_absolute"] for x in c), "max rel rms", max(x["relative_rms"] for x in c))'
endef
compare: compare-f32 compare-f16 ## Layer/logit trace comparison of `Hello,` vs the reference traces, both cache precisions (needs docs/reference/generation.md setup)

compare-f32: metal ## The F32 cache at the bring-up thresholds (max abs 2e-3, relative RMS 1e-4)
	$(call compare_run,f32,0.002,0.0001)

compare-f16: metal ## The F16 cache at its own tolerance (max abs 3e-2, relative RMS 2e-4; see docs/reference/metal-backend.md)
	$(call compare_run,f16,0.03,0.0002)

# `bonsai-2-27b`: Qwen3.8-27B re-encoded ternary in a Hadamard-rotated basis
# (docs/reference/bonsai.md), with its own traces from the PrismML fork
# (captured from the PQ2_0 packing, which the catalogue's PTQ1_0 file of the
# same weights matches at the same thresholds).
BONSAI_MODEL ?= $(HOME)/.nuclis/models/prism-ml/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf
# $(1) label, $(2) backend flags, $(3) max absolute, $(4) max relative RMS per trace file.
define compare_bonsai_run
	rm -rf "$(TRACE)-bonsai-$(1)" && mkdir -p "$(TRACE)-bonsai-$(1)"
	$(BIN) generate $(2) --model "$(BONSAI_MODEL)" --raw --prompt 'Hello,' --max-tokens 1 --ctx-size 8 \
	  --logits "$(TRACE)-bonsai-$(1)/logits.f32" --trace-dir "$(TRACE)-bonsai-$(1)" $(ARGS) > /dev/null
	python3 scripts/compare-generation.py "$(TRACE)-bonsai-$(1)" tests/fixtures/bonsai-hello-comma --positions 2 --max-absolute $(3) --max-relative-rms $(4) \
	  | python3 -c 'import json,sys; d=json.load(sys.stdin); c=d["comparisons"]; print("bonsai $(1)", "passed", d["passed"], "files", len(c), "max abs", max(x["max_absolute"] for x in c), "max rel rms", max(x["relative_rms"] for x in c))'
endef
compare-bonsai: compare-bonsai-cpu compare-bonsai-f32 compare-bonsai-f16 ## bonsai-2-27b vs the PrismML fork's traces: CPU reference, Metal F32 and F16 caches (docs/reference/bonsai.md)

compare-bonsai-cpu: metal ## The Qwen CPU reference on the Bonsai file (ternary weights, the Hadamard transform) vs the fork's traces at the bring-up thresholds
	$(call compare_bonsai_run,cpu,--backend cpu,0.002,0.0001)

compare-bonsai-f32: metal ## The Qwen Metal plan on the Bonsai file with the F32 cache at the bring-up thresholds
	$(call compare_bonsai_run,f32,--backend metal --kv f32,0.002,0.0001)

compare-bonsai-f16: metal ## The Qwen Metal plan on the Bonsai file with the F16 cache at its own tolerance
	$(call compare_bonsai_run,f16,--backend metal --kv f16,0.03,0.0002)

test-generation-bonsai-metal: ## The generation check on bonsai-2-27b, Metal plan
	$(ZIG) build test-generation $(METAL) -- "$(BONSAI_MODEL)" --metal

speculative-record: metal ## The speculative-decoding record on Qwen3.8-27B: off/on pairs over the corpus arrays and the code prompt, greedy and instruct, draft 2/4/7; JSON under .zig-cache/bench/spec (docs/reference/bench.md § Speculative record)
	python3 scripts/nuclis-speculative.py --model "$(MODEL)" --nuclis $(BIN) $(ARGS)

baseline-bonsai: metal ## The reference workload on bonsai-2-27b against the PrismML fork's records (tests/fixtures/run-2026-09-18-bonsai, whose token arrays are the Qwen run's); writes docs/benchmarks/nuclis-<date>-bonsai.json
	python3 scripts/nuclis-baseline.py --model "$(BONSAI_MODEL)" --nuclis $(BIN) --run run-2026-09-18-bonsai \
	  --reference-records reference-2026-09-18-bonsai.json --output docs/benchmarks/nuclis-$$(date +%F)-bonsai.json $(ARGS)

# The catalogue's Gemma 4 12B file (QAT, every matrix Q4_0; MODL-08) and the
# K-quant file the adapter was brought up on (MODL-05–MODL-07); both stay pinned in
# docs/reference/artifacts.md and each has its own trace fixtures.
# One variable per catalogue entry, named after it: `gemma-4-12b` is the
# K-quant release the adapter was brought up on, `gemma-4-12b-qat` Google's
# quantization-aware-trained checkpoint. Each has its own pinned traces.
GEMMA_MODEL ?= $(HOME)/.nuclis/models/unsloth/gemma-4-12b-it-GGUF/gemma-4-12b-it-UD-Q4_K_XL.gguf
GEMMA_QAT_MODEL ?= $(HOME)/.nuclis/models/unsloth/gemma-4-12B-it-qat-GGUF/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf
# `gemma-4-26b-a4b` is the QAT mixture of experts (MODL-09), with its own traces.
GEMMA_26B_A4B_MODEL ?= $(HOME)/.nuclis/models/unsloth/gemma-4-26B-A4B-it-qat-GGUF/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf
# $(1) label, $(2) model file, $(3) fixture directory, $(4) backend flags, $(5) max absolute, $(6) max relative RMS per trace file,
# $(7) embedding width and $(8) layer count of the traces (3840 and 48 for the 12B files).
define compare_gemma4_run
	rm -rf "$(TRACE)-gemma4-$(1)" && mkdir -p "$(TRACE)-gemma4-$(1)"
	$(BIN) generate $(4) --model "$(2)" --prompt-tokens $(3)/prompt-tokens.json \
	  --max-tokens 1 --ctx-size 8 --temperature 0 --logits "$(TRACE)-gemma4-$(1)/logits.f32" --trace-dir "$(TRACE)-gemma4-$(1)" $(ARGS) > /dev/null
	python3 scripts/compare-generation.py "$(TRACE)-gemma4-$(1)" $(3) --positions 3 --embedding $(or $(7),3840) --layers $(or $(8),48) --vocab 262144 \
	  --max-absolute $(5) --max-relative-rms $(6) \
	  | python3 -c 'import json,sys; d=json.load(sys.stdin); c=d["comparisons"]; print("gemma4 $(1)", "passed", d["passed"], "files", len(c), "max abs", max(x["max_absolute"] for x in c), "max rel rms", max(x["relative_rms"] for x in c))'
endef
MUSE_MODEL ?= $(HOME)/.nuclis/models/unsloth/Muse-Glimmer-30B-GGUF/Muse-Glimmer-30B-UD-Q4_K_XL.gguf
# $(1) label, $(2) backend flags, $(3) max absolute, $(4) max relative RMS per trace file.
define compare_muse_glimmer_run
	rm -rf "$(TRACE)-muse-glimmer-$(1)" && mkdir -p "$(TRACE)-muse-glimmer-$(1)"
	$(BIN) generate $(2) --model "$(MUSE_MODEL)" --prompt-tokens tests/fixtures/muse-glimmer-hello-comma/prompt-tokens.json \
	  --max-tokens 1 --ctx-size 8 --temperature 0 --logits "$(TRACE)-muse-glimmer-$(1)/logits.f32" --trace-dir "$(TRACE)-muse-glimmer-$(1)" $(ARGS) > /dev/null
	python3 scripts/compare-generation.py "$(TRACE)-muse-glimmer-$(1)" tests/fixtures/muse-glimmer-hello-comma --positions 3 --embedding 6656 --layers 52 --vocab 202048 \
	  --max-absolute $(3) --max-relative-rms $(4) \
	  | python3 -c 'import json,sys; d=json.load(sys.stdin); c=d["comparisons"]; print("muse-glimmer $(1)", "passed", d["passed"], "files", len(c), "max abs", max(x["max_absolute"] for x in c), "max rel rms", max(x["relative_rms"] for x in c))'
endef
compare-muse-glimmer: compare-muse-glimmer-cpu compare-muse-glimmer-f32 compare-muse-glimmer-f16 ## muse-glimmer-30b vs its pinned llama.cpp traces (`<|begin_of_text|>Hello,`, three positions): CPU reference, Metal F32 and F16 caches (docs/reference/muse-glimmer.md)

compare-muse-glimmer-cpu: metal ## The Muse Glimmer CPU reference at the bring-up thresholds (max abs 2e-3, relative RMS 1e-4)
	$(call compare_muse_glimmer_run,cpu,--backend cpu,0.002,0.0001)

compare-muse-glimmer-f32: metal ## The Muse Glimmer Metal plan with the F32 cache at the bring-up thresholds
	$(call compare_muse_glimmer_run,f32,--backend metal --kv f32,0.002,0.0001)

compare-muse-glimmer-f16: metal ## The Muse Glimmer Metal plan with the F16 cache at the family's tolerance (max abs 0.1, relative RMS 3e-4; docs/reference/muse-glimmer.md)
	$(call compare_muse_glimmer_run,f16,--backend metal --kv f16,0.1,0.0003)

test-generation-muse-glimmer-metal: ## The generation check on muse-glimmer-30b, Metal plan
	$(ZIG) build test-generation $(METAL) -- "$(MUSE_MODEL)" --metal

compare-gemma4-26b-a4b: compare-gemma4-26b-a4b-cpu compare-gemma4-26b-a4b-f32 compare-gemma4-26b-a4b-f16 ## gemma-4-26b-a4b (mixture of experts) vs its pinned traces: CPU reference, Metal F32 and F16 caches (MODL-09, docs/reference/gemma4.md)

compare-gemma4-26b-a4b-cpu: metal ## The Gemma CPU reference on the 26B-A4B (expert) file vs its pinned traces at the bring-up thresholds
	$(call compare_gemma4_run,26b-a4b-cpu,$(GEMMA_26B_A4B_MODEL),tests/fixtures/gemma4-26b-a4b-hello-comma,--backend cpu,0.002,0.0001,2816,30)

compare-gemma4-26b-a4b-f32: metal ## The Gemma Metal plan on the 26B-A4B file with the F32 cache at the bring-up thresholds
	$(call compare_gemma4_run,26b-a4b-f32,$(GEMMA_26B_A4B_MODEL),tests/fixtures/gemma4-26b-a4b-hello-comma,--backend metal --kv f32,0.002,0.0001,2816,30)

compare-gemma4-26b-a4b-f16: metal ## The Gemma Metal plan on the 26B-A4B file with the F16 cache at the family's tolerance
	$(call compare_gemma4_run,26b-a4b-f16,$(GEMMA_26B_A4B_MODEL),tests/fixtures/gemma4-26b-a4b-hello-comma,--backend metal --kv f16,1.0,0.05,2816,30)

compare-gemma4-qat: compare-gemma4-qat-cpu compare-gemma4-qat-f32 compare-gemma4-qat-f16 ## gemma-4-12b-qat vs its pinned llama.cpp traces (`<bos>Hello,`, three positions): CPU reference, Metal F32 and F16 caches (MODL-08, docs/reference/gemma4.md)

compare-gemma4-qat-cpu: metal ## The Gemma CPU reference on the QAT file at the bring-up thresholds (max abs 2e-3, relative RMS 1e-4)
	$(call compare_gemma4_run,qat-cpu,$(GEMMA_QAT_MODEL),tests/fixtures/gemma4-qat-hello-comma,--backend cpu,0.002,0.0001)

compare-gemma4-qat-f32: metal ## The Gemma Metal plan on the QAT file with the F32 cache at the bring-up thresholds
	$(call compare_gemma4_run,qat-f32,$(GEMMA_QAT_MODEL),tests/fixtures/gemma4-qat-hello-comma,--backend metal --kv f32,0.002,0.0001)

compare-gemma4-qat-f16: metal ## The Gemma Metal plan on the QAT file with the F16 cache at its own tolerance (the model's key-rounding sensitivity; see docs/reference/gemma4.md)
	$(call compare_gemma4_run,qat-f16,$(GEMMA_QAT_MODEL),tests/fixtures/gemma4-qat-hello-comma,--backend metal --kv f16,1.0,0.05)

compare-gemma4: compare-gemma4-cpu compare-gemma4-f32 compare-gemma4-f16 ## gemma-4-12b (K-quant) vs its pinned traces (MODL-05/MODL-06)

compare-gemma4-cpu: metal ## The Gemma CPU reference on the K-quant file at the bring-up thresholds
	$(call compare_gemma4_run,cpu,$(GEMMA_MODEL),tests/fixtures/gemma4-hello-comma,--backend cpu,0.002,0.0001)

compare-gemma4-f32: metal ## The Gemma Metal plan on the K-quant file with the F32 cache at the bring-up thresholds
	$(call compare_gemma4_run,f32,$(GEMMA_MODEL),tests/fixtures/gemma4-hello-comma,--backend metal --kv f32,0.002,0.0001)

compare-gemma4-f16: metal ## The Gemma Metal plan on the K-quant file with the F16 cache at its own tolerance
	$(call compare_gemma4_run,f16,$(GEMMA_MODEL),tests/fixtures/gemma4-hello-comma,--backend metal --kv f16,1.0,0.05)

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
