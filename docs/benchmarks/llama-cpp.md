# The pinned llama.cpp reference

While the project bootstrapped, a pinned llama.cpp build was nuclis's
numerical oracle and its speed reference. That role ended on 2026-10-14:
nuclis is now its own reference
([models § 4](../models/README.md#4-validate-against-nuclis-itself)). This
page is the record of what was pinned, how the pinned measurements were
taken, and where their evidence stays. The scripts that drove it (the
`reference-*` harnesses and their comparison and conversion helpers) left
the tree with that change; read them with `git show v0.11.0:scripts/<name>`.

## The pinned build

The mainline revision was `7620399f58aebfd2196b74021f9581bcf7218cb9`,
checked out and built under the ignored `.reference/` directory with:

```sh
cmake -S .reference/llama.cpp -B .reference/llama.cpp/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_OPENSSL=OFF \
  -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build .reference/llama.cpp/build \
  --target llama-server llama-bench llama-completion --parallel 8
```

The Metal shaders were embedded source compiled at runtime, which worked
with the Command Line Tools alone.

## The second oracle: the PrismML fork (2026-09-18)

Bonsai 2 27B ([bonsai.md](../models/bonsai.md)) is stored in two encodings
the mainline pin rejected, so its oracle was the PrismML fork of llama.cpp
(`github.com/PrismML-Eng/llama.cpp`, MIT) at release tag
`prism-b10687-5d80cff`, commit `5d80cff0b8cb9f2bf823cfc4e71e3abb97f290d6`
(2026-09-17), built with the same flags in its own checkout. The mainline
pin stayed the oracle for every other family. The fork's server took the
mainline flags below except `--lazy-mode`, which it did not know. The
Bonsai acceptance run of 2026-09-18 recorded the fork's revision as the
reference side of
[reference-2026-09-18-bonsai.json](reference-2026-09-18-bonsai.json)
([history § Bonsai 2 27B acceptance record](history.md#bonsai-2-27b-acceptance-record-2026-09-18)).

## The third oracle: EmbeddingGemma 2 (2026-10-09)

The mainline pin did not know `gemma-embedding2`
([embeddinggemma.md](../models/embeddinggemma.md)), so that family's
oracle was a second mainline checkout at build tag `b11514`, commit
`de7fa0a3c6a2e1b4cd9f22eb8d6bf5b12dbdb63b` (2026-10-08), built with the
same flags plus the `llama-embedding` and `llama-mtmd-cli` targets. Its
vectors and traces are the `embeddinggemma-*` rows of
[provenance.md](../../tests/fixtures/provenance.md). The two text traces
came from ggml's CPU backend over a copy of the Q8_0 file with every tensor
widened to F32, so no matmul rounded its activations; ggml's GELU was still
an f16 lookup table, which with its other F32 differences set the traces'
bound ([embeddinggemma.md § CPU reference](../models/embeddinggemma.md#cpu-reference-against-the-oracles-2026-10-09)).
Images were sized with `--image-max-tokens 280`, the count Google's
processor uses.

## Run the workload

The reference records were taken against one server per family, started
with these flags (the Qwen3.8 file shown):

```sh
.reference/llama.cpp/build/bin/llama-server \
  --model "$HOME/.nuclis/models/qwen/Qwen3.8-27B-UD-Q4_K_M.gguf" \
  --ctx-size 32768 --parallel 1 --device MTL0 --n-gpu-layers 99 \
  --flash-attn on --cache-type-k f16 --cache-type-v f16 \
  --batch-size 2048 --ubatch-size 512 --threads 8 --threads-batch 8 \
  --host 127.0.0.1 --port 18087 --reasoning off --no-context-shift \
  --cache-ram 0 --cache-reuse 0 --load-mode mmap --lazy-mode off --fit off \
  --perf --no-webui --cors-origins localhost --log-verbosity 4
```

The harness checked the server's revision and one-slot capacity, applied
the model's chat template with thinking off, and recorded the exact input
token arrays for 512, 4,096, and 16,384 tokens: a template prefix, a
truncated synthetic-code sequence, and a closing instruction. The near-capacity
check ran separately at 32,639 prompt tokens, one token of margin. Those
token arrays are the inputs nuclis's own acceptance runs replay
(`bench --prompt-tokens`), so both sides measured the same tokens.

Each size had one discarded warm-up and three recorded repetitions: greedy
(`temperature: 0`), seed 1, no prefix reuse, 128 generated tokens with EOS
ignored. Prompt and decode rates are the server's own timings (decode over
the intervals after the first generated token); streaming was off, so no
time to first token was measured. OS file caches and the Metal compiler
cache were not purged, and other applications kept running.

For other families the template check changed: `gemma4` checked the
thinking-off marker and wrote `<bos>` into the prompt text, `gemma4-e` a
bare generation prompt, `muse-glimmer` the template's `Reasoning strength:
high.` line (that template cannot switch reasoning off).

## Where the evidence is

- Compact records, one per run: `reference-*.json` in this directory.
- Token arrays and raw run records: `tests/fixtures/run-2026-09-06/`,
  `boundary-2026-09-06/`, `run-2026-09-18-gemma4-26b-a4b/`,
  `run-2026-10-04-gemma4-e4b/`; layer traces, perplexity results, and
  embedding vectors pinned from this build are listed in
  [provenance.md](../../tests/fixtures/provenance.md). They stay in the
  tree; a re-pinned file's fixtures are made by nuclis
  ([models § Which executor makes which fixture](../models/README.md#which-executor-makes-which-fixture)).
- Later runs against the pin are in [README.md](README.md) and
  [history.md](history.md).

## Recorded environment

| Item | Value on 2026-09-06 |
| --- | --- |
| Hardware | Apple M4 Pro, 12 CPU cores (8 performance + 4 efficiency), 16 GPU cores |
| Unified memory | 51,539,607,552 bytes (48 GiB) |
| OS | macOS 26.6.2, build 25G83 |
| Compiler / SDK | Apple clang 21.0.0 (`clang-2100.1.1.101`), macOS SDK 26.5 |
| Build | CMake 4.4.3, Release, native ARM CPU instructions, Metal and Accelerate |
| Power | AC; `pmset -g custom` reports AC `powermode 0` |
| Model | `Qwen3.8-27B-UD-Q4_K_M.gguf`, the first pin (`322e194f…`) |
| Runtime device | Explicit `MTL0`, Apple M4 Pro |
| Attention / recurrent state | F16 K/V, F32 recurrent state, flash attention enabled |

The server log showed a 2,048 MiB KV buffer, a 149.62 MiB recurrent-state
buffer, a 257.30 MiB Metal compute buffer, and a 52.02 MiB CPU compute
buffer.

## Results (2026-09-06)

The first, Qwen-only run. Warm request rates, mean ± sample standard
deviation over three repetitions:

| Prompt tokens | Output tokens | Prompt tokens/s | Decode tokens/s |
| ---: | ---: | ---: | ---: |
| 512 | 128 | 89.19 ± 0.08 | 9.66 ± 0.07 |
| 4,096 | 128 | 89.26 ± 1.17 | 9.21 ± 0.25 |
| 16,384 | 128 | 74.07 ± 0.66 | 7.32 ± 0.25 |

A first capacity request at the exact boundary (32,640 input tokens plus
128 outputs, 487.67 s, 69.54 prefill and 6.93 decode tokens/s) finished but
set the server's `truncated` flag: at this revision the server checks
`prompt.n_tokens() + 1 >= n_ctx` before the output-budget stop. The harness
rejected it. The [measurement record](reference-2026-09-06.json) keeps the
nine accepted timings, input-file hashes, and the flagged result.

## One-token-margin follow-up (2026-09-06)

At 32,639 input tokens plus 128 outputs, both the warm-up and the measured
request passed: zero cached prompt tokens, `truncated=false`,
`stop_type=limit`. The measured request took 504.09 s, with 67.28 prefill
and 6.71 decode tokens/s; one capacity sample, not a replacement for the
warm means. System swap was 247.81 MiB at the start and 239.81 MiB after
the warm-up; server RSS ranged from 18,086,272 to 18,328,896 KiB. These
are snapshots, not peak-memory measurements, and did not establish
zero-swap acceptance. The [compact record](reference-boundary-2026-09-06.json)
keeps both timings, the acceptance flags and counts, hashes, and memory
counters.

## Source references

Reference behaviour was checked against the pinned
[server API documentation](https://github.com/ggml-org/llama.cpp/blob/7620399f58aebfd2196b74021f9581bcf7218cb9/tools/server/README.md),
[Metal build configuration](https://github.com/ggml-org/llama.cpp/blob/7620399f58aebfd2196b74021f9581bcf7218cb9/ggml/src/ggml-metal/CMakeLists.txt),
and [Qwen35 adapter](https://github.com/ggml-org/llama.cpp/blob/7620399f58aebfd2196b74021f9581bcf7218cb9/src/models/qwen35.cpp).
