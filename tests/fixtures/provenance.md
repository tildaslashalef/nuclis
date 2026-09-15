# Fixture provenance

Everything in this directory is **committed evidence**: outputs of running the
pinned reference at a known revision against the pinned model artifact. It is
the oracle that `make compare` and the benchmark records are checked against.
The regenerable parts of the reference pipeline (the llama.cpp checkout, live
run outputs under `.zig-cache/`) stay out of git and are rebuilt on demand by
[../../docs/reference/reference-baseline.md](../../docs/reference/reference-baseline.md).

| Path | What it is | Produced by |
| --- | --- | --- |
| `model.sha256` | SHA-256 `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482` of the pinned `Qwen3.8-27B-UD-Q4_K_M.gguf` under `~/.nuclis/models/qwen/` | captured at baseline time (2026-09-06) |
| `tokenizer-metadata.json` | vocabulary facts asserted by `test-vocabulary` | captured at baseline time from the pinned artifact |
| `reference-hello-comma/` | llama.cpp layer/logit traces for the `'Hello,'` single-token comparison (129 `.f32` files; the slices `compare-generation.py` reads at `--positions 2`) | `scripts/reference-generation.cpp` against llama.cpp `7620399f58aebfd2196b74021f9581bcf7218cb9`, 2026-09-06 |
| `reference-chat/` | multi-turn chat trace, wider oracle for session/prefix work | same revision and artifact, 2026-09-06 |
| `gemma4-hello-comma/` | llama.cpp layer/logit traces for Gemma 4 12B (`unsloth/gemma-4-12b-it-GGUF` `gemma-4-12b-it-UD-Q4_K_XL.gguf`, SHA-256 `90fd944d…`) on the prompt `<bos>Hello,` (`prompt-tokens.json` = `[2, 9259, 236764]`): 144 layer files (three positions × 48 layers, 3840 floats each) and `logits.f32` (262,144 floats); the payload of `make compare-gemma4` | the same `scripts/reference-generation.cpp` (generalized to any F32 width) against llama.cpp `7620399f58aebfd2196b74021f9581bcf7218cb9`, Metal, F32 cache, 2026-09-11 |
| `gemma4-qat-hello-comma/` | the same traces for the catalogue's QAT file (`unsloth/gemma-4-12B-it-qat-GGUF` `gemma-4-12B-it-qat-UD-Q4_K_XL.gguf`, SHA-256 `90fd44e2…`, every matrix Q4_0) on the same prompt and ids; the payload of `make compare-gemma4` (the K-quant traces above serve `make compare-gemma4-kquant`) | the same harness against the same llama.cpp revision, Metal, F32 cache, 2026-09-12 (MODL-08) |
| `run-2026-09-06/` | raw baseline run records: harness, prompt files (512/4,096/16,384/32,640 tokens), responses, checksums. The `prompt-*.json` token arrays are also the inputs of nuclis's own acceptance runs (`scripts/nuclis-baseline.py` → `bench --prompt-tokens`), so both sides measured the same tokens | `scripts/reference-baseline.py` run, 2026-09-06; summarized in [../../docs/benchmarks/reference-2026-09-06.json](../../docs/benchmarks/reference-2026-09-06.json) |
| `run-2026-09-12-gemma4/` | the same harness on Gemma 4 12B (`gemma-4-12b-it-UD-Q4_K_XL.gguf`, SHA-256 `90fd944d…`, `--family gemma4`): prompt arrays for 512/4,096/16,384/32,639 tokens (each starting with `<bos>` 2, then the Gemma template's `<|turn>user` prefix, the synthetic corpus, and the suffix ending in the pre-closed thought channel), responses, checksums, summary. The inputs of `make baseline-gemma4` | `scripts/reference-baseline.py --family gemma4 --prompt-lengths 512,4096,16384,32639`, 2026-09-12, server at 32,768 context with the reference recipe's flags; summarized in [../../docs/benchmarks/reference-2026-09-12-gemma4.json](../../docs/benchmarks/reference-2026-09-12-gemma4.json) |
| `run-2026-09-12-gemma4-qat/` | the same harness on the catalogue's QAT Gemma 4 12B file (`gemma-4-12B-it-qat-UD-Q4_K_XL.gguf`, SHA-256 `90fd44e2…`, `--family gemma4`): the same four prompt lengths and construction (identical token arrays, since both files share the vocabulary and template), its own responses, checksums, and summary. The inputs of `make baseline-gemma4` (`baseline-gemma4-kquant` uses the directory above) | `scripts/reference-baseline.py --family gemma4 --prompt-lengths 512,4096,16384,32639`, 2026-09-12 (MODL-08), server at 32,768 context with the reference recipe's flags; summarized in [../../docs/benchmarks/reference-2026-09-12-gemma4-qat.json](../../docs/benchmarks/reference-2026-09-12-gemma4-qat.json) |
| `boundary-2026-09-06/` | context-boundary run records (32,767/32,768 positions); `prompt-32639.json` is the 32K acceptance input | same run; summarized in [../../docs/benchmarks/reference-boundary-2026-09-06.json](../../docs/benchmarks/reference-boundary-2026-09-06.json) |

Rules for agents:

- **Never edit or regenerate these files in place.** They are the fixed point
  that proves kernel changes are equivalent. If a trace must be re-produced
  (e.g. a new reference revision), generate it under `.zig-cache/reference/`
  first, verify it byte-for-byte against these fixtures, and only then propose
  a fixture update as its own reviewed commit.
- The acceptance value of `make compare` is max abs `0.0001220703125` over
  129 files (see [../../docs/reference/bench.md](../../docs/reference/bench.md)).
  No threshold is loosened without a documented reason.
- `reference-chat/` is an opt-in oracle (session/prefix comparisons); the
  default `make compare` payload is only `reference-hello-comma/`.
