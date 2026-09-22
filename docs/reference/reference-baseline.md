# Pinned Metal reference baseline

This guide records the reference configuration for Qwen3.8-27B on the target
Mac. Measurements here belong to llama.cpp; nuclis does not execute models yet.
Use matching token inputs and settings for a later comparison.

## Build the reference

The source revision is `7620399f58aebfd2196b74021f9581bcf7218cb9`.
Keep its checkout and build in the ignored reference cache. Run these commands
from the nuclis repository root:

```sh
mkdir -p .zig-cache/reference
gh repo clone ggml-org/llama.cpp .zig-cache/reference/llama.cpp
git -C .zig-cache/reference/llama.cpp checkout --detach 7620399f58aebfd2196b74021f9581bcf7218cb9

cmake -S .zig-cache/reference/llama.cpp -B .zig-cache/reference/llama.cpp/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_OPENSSL=OFF \
  -DLLAMA_USE_PREBUILT_UI=OFF

cmake --build .zig-cache/reference/llama.cpp/build \
  --target llama-server llama-bench llama-completion --parallel 8
```

The cache survives clearing `/tmp`, but deleting `.zig-cache` removes both the
checkout and its build. Reconstruct them with the pinned commands above. CMake
records absolute paths: after moving a checkout, configure a fresh build
directory instead of reusing a build generated at the old location.

This uses embedded shader source compiled through Metal at runtime. It worked
with Command Line Tools even though `xcrun --find metal` found no standalone
compiler. Initial shader compilation and subsequent driver-cache hits can have
different startup costs. No system toolchain changes are required for this path.

## The second oracle: the PrismML fork (MODL-16, 2026-09-18)

Bonsai 2 27B ([bonsai.md](bonsai.md)) is stored in two encodings the
mainline reference rejects, so its oracle is the PrismML fork of llama.cpp
(`github.com/PrismML-Eng/llama.cpp`, MIT, tracks mainline) at its release
tag `prism-b10687-5d80cff`, commit
`5d80cff0b8cb9f2bf823cfc4e71e3abb97f290d6` (2026-09-17). Two pins, never one
moving one: the mainline revision above stays the oracle for every other
family, and every script that checks a served build's revision takes the
fork's as a parameter (`profile-alias-check.py --reference-revision`) or a
second constant (`quant-fixtures.py`'s `PRISM_REVISION`). The same recipe,
in its own checkout:

```sh
git clone https://github.com/PrismML-Eng/llama.cpp .zig-cache/reference/prism-llama.cpp
git -C .zig-cache/reference/prism-llama.cpp checkout --detach 5d80cff0b8cb9f2bf823cfc4e71e3abb97f290d6

cmake -S .zig-cache/reference/prism-llama.cpp -B .zig-cache/reference/prism-llama.cpp/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_OPENSSL=OFF \
  -DLLAMA_USE_PREBUILT_UI=OFF

cmake --build .zig-cache/reference/prism-llama.cpp/build \
  --target llama-server llama-bench llama-completion --parallel 8
```

The trace harness (`scripts/reference-generation.cpp`, public C API only)
builds against this checkout unchanged with its include and library paths
in place of the mainline ones, into
`.zig-cache/generation/prism-reference-generation`
([generation.md](generation.md#numerical-traces)). The fork's server takes
the mainline one's flags below except `--lazy-mode`, which its older base
does not know (drop it; the rest are accepted); `--jinja` renders the
Bonsai template, while the fork's `llama-completion --jinja` aborts at its
own start-up template self-test on that template (see bonsai.md). The
Bonsai acceptance run (MODL-17, 2026-09-18) ran the workload harness
against that server with `--reference-revision 5d80cff0…` (the harness
otherwise insists on the mainline pin) and `--family qwen38`, the fork's
revision recorded as the reference side of
[reference-2026-09-18-bonsai.json](../benchmarks/reference-2026-09-18-bonsai.json)
and read from there by `nuclis-baseline.py` into the nuclis record
([bench.md](bench.md#bonsai-2-27b-acceptance-record-modl-17-2026-09-18)).

## Run the workload

Start one reference server in a separate terminal:

```sh
mkdir -p .zig-cache/reference
.zig-cache/reference/llama.cpp/build/bin/llama-server \
  --model "$HOME/.nuclis/models/qwen/Qwen3.8-27B-UD-Q4_K_M.gguf" \
  --ctx-size 32768 --parallel 1 --device MTL0 --n-gpu-layers 99 \
  --flash-attn on --cache-type-k f16 --cache-type-v f16 \
  --batch-size 2048 --ubatch-size 512 --threads 8 --threads-batch 8 \
  --host 127.0.0.1 --port 18087 --reasoning off --no-context-shift \
  --cache-ram 0 --cache-reuse 0 --load-mode mmap --lazy-mode off --fit off \
  --perf --no-webui --cors-origins localhost --log-verbosity 4 \
  > .zig-cache/reference/server.log 2>&1
```

Then run the opt-in development harness from this repository:

```sh
python3 scripts/reference-baseline.py \
  --output-dir .zig-cache/reference/warm-new-run
```

Run the near-capacity check separately, leaving one token of boundary margin:

```sh
python3 scripts/reference-baseline.py \
  --output-dir .zig-cache/reference/boundary-new-run \
  --prompt-lengths 32639 --repetitions 1 --server-pid <pid>
```

For a second family, start the server on its file and pass `--family`
(`qwen38` by default; `gemma4` checks that family's thinking-off marker
and writes `<bos>` into the prompt text, since the server's
`/apply-template` strips the BOS its own tokenizer re-adds while nuclis's
encoder never adds one). The Gemma 4 12B run of 2026-09-12 used the same
server flags on `gemma-4-12b-it-UD-Q4_K_XL.gguf` with
`--prompt-lengths 512,4096,16384,32639` in one invocation
(`tests/fixtures/run-2026-09-12-gemma4/`). `--family muse-glimmer` (the
run of 2026-09-19, `tests/fixtures/run-2026-09-19-muse-glimmer/`) checks
the template's default `Reasoning strength: high.` line instead, since
that template cannot switch reasoning off, writes `<|begin_of_text|>`, and
gives the smoke request `low` strength and a 384-token budget so the
model's reasoning message leaves room for the code it checks for.

Use a new output directory for every invocation. The harness refuses to overwrite
an existing directory. It uses only the Python standard library, accepts only a
loopback endpoint, and sends generated synthetic Zig code rather than workspace
files. Stop the server with Ctrl-C afterward to release the model and GPU state.
In an agent sandbox, both GPU access and the loopback connection may require
execution outside the sandbox.
Optional `--server-pid <pid>` samples that process's RSS between requests.
Swap and VM counters are recorded around every timed request. These snapshots
are observations, not continuous peak-memory measurements.

The harness verifies the server's revision identifier and one-slot context
capacity, then checks a simple code-generation response. It applies the model's
chat template with thinking disabled and records exact input token arrays for
512-, 4,096-, and 16,384-token prompts. These combine a template prefix, a
truncated synthetic-code sequence, and a closing instruction/template suffix.
The cut can occur inside a function; this is a reproducible throughput workload,
not a coding-quality benchmark.

Each size gets one discarded warmup followed by three recorded repetitions.
Requests use greedy sampling (`temperature: 0`, only the temperature sampler),
seed 1, no prefix reuse, and a fixed budget of 128 generated tokens. EOS is
ignored for those timed requests to keep the work amount fixed. The smoke test
uses normal EOS behavior. The separate near-capacity command runs one warmup
and one measured request, each with 32,639 prompt tokens plus 128 outputs.
The legacy `--capacity-check` option instead uses 32,640 prompt tokens plus 128
outputs once. That exact-boundary diagnostic is expected to trip the pinned
server's truncation flag; use the margin command for a clean capacity check.

Raw responses, server properties, prompt-construction details, token arrays,
individual timings, and summaries are written to the selected untracked output
directory. Using `.zig-cache/reference` keeps them across restarts that clear
`/tmp`; deleting the Zig cache also deletes these raw artifacts. The runner
rejects truncated/reused prompts and completions that miss the output budget.
Device selection and allocation details must also be checked in the server log;
the runner's revision check alone does not establish that Metal was used.

## Measurement boundaries

Prompt and decode rates are the reference server's separate reported timings.
The client additionally records total request wall time. Tokenization and prompt
formatting happen before each timed request, and HTTP streaming is disabled, so
these runs do not measure streaming time to first token. At this revision,
decode rate uses the intervals after the first generated token; retain raw
`predicted_n` and `predicted_ms` rather than recomputing it as 128 divided by time.

Prefix reuse is disabled by both the request and server cache settings. The
reference still uses its default recurrent-state checkpoints within requests;
their overhead is part of the server baseline. Warmup results are retained but
excluded from reported means. Means and sample standard deviations summarize
per-request rates rather than pooling different contexts. The full-file checksum
was reverified before timing, which also reads the artifact through the OS cache.
OS file caches and the Metal compiler cache were
not purged, and applications on the Mac were not stopped. A cold-start or isolated
peak-throughput claim would require a separate experiment.

## Recorded environment

| Item | Value on 2026-09-06 |
| --- | --- |
| Hardware | Apple M4 Pro, 12 CPU cores (8 performance + 4 efficiency), 16 GPU cores |
| Unified memory | 51,539,607,552 bytes (48 GiB) |
| OS | macOS 26.6.2, build 25G83 |
| Compiler / SDK | Apple clang 21.0.0 (`clang-2100.1.1.101`), macOS SDK 26.5 |
| Build | CMake 4.4.3, Release, native ARM CPU instructions, Metal and Accelerate |
| Power | AC; `pmset -g custom` reports AC `powermode 0` |
| Model | Pinned `Qwen3.8-27B-UD-Q4_K_M.gguf` from the [spec](../spec.md#4-supported-models-and-artifacts) |
| Runtime device | Explicit `MTL0`, Apple M4 Pro |
| Attention / recurrent state | F16 K/V, F32 recurrent state, flash attention enabled |

The log confirms GPU offload and an auxiliary-block skip consistent with the
[Qwen adapter](qwen-validation.md). It reports a 2,048 MiB KV buffer, 149.62 MiB
recurrent-state buffer, 257.30 MiB Metal compute buffer, and 52.02 MiB CPU compute
buffer. Mapped model views may overlap; summing their reported sizes is not a
valid resident-memory calculation.

## Results

Warm request rates, mean ± sample standard deviation over three repetitions:

| Prompt tokens | Output tokens | Prompt tokens/s | Decode tokens/s |
| ---: | ---: | ---: | ---: |
| 512 | 128 | 89.19 ± 0.08 | 9.66 ± 0.07 |
| 4,096 | 128 | 89.26 ± 1.17 | 9.21 ± 0.25 |
| 16,384 | 128 | 74.07 ± 0.66 | 7.32 ± 0.25 |

The capacity request finished: 32,640 input tokens, 128 output tokens, zero
cached prompt tokens, 487.67 seconds wall time. Diagnostic rates were 69.54
prefill tokens/s and 6.93 decode tokens/s. The response set `truncated: true`,
so the harness rejected it and did **not** write `summary.json`. The nine valid
warm samples remain in `samples.json`; the raw capacity response was also saved.

At the pinned revision, `server-context.cpp` checks
`!ctx_shift && prompt.n_tokens() + 1 >= n_ctx` before the normal output-budget
stop. This sets the truncation flag at the exact context boundary even when all
requested input and output counts were met. Thus this result demonstrates a
completed boundary workload, but does not pass our strict no-truncation check.
The follow-up below leaves one token of margin (32,639 + 128) and verifies the
flag, counts, and memory without weakening the guard.

A compact [measurement record](../benchmarks/reference-2026-09-06.json) preserves
all nine accepted timings, input-file hashes, and the flagged capacity result.
Saved warm/warmup memory snapshots showed zero swap usage; they are not peak
memory measurements. The reference process was no longer running when checked
after completion. Raw local evidence is in
`tests/fixtures/run-2026-09-06/` (committed fixtures); `.zig-cache/reference/server.log`
is a regenerable cache artifact.

An earlier interrupted attempt experienced growing swap usage and lost its
temporary artifacts when `/tmp` was cleared. Its partial timings are excluded
from this report. The fresh run records memory conditions alongside every sample.

The raw evidence of the accepted run is committed as fixtures:
[tests/fixtures/run-2026-09-06/](../../tests/fixtures/run-2026-09-06/) — see
[tests/fixtures/provenance.md](../../tests/fixtures/provenance.md). The
working copies under `.zig-cache/reference/` are regenerable caches; the
fixtures are the retained oracle. The reference process was no longer running
when checked after completion.

## One-token-margin follow-up — 2026-09-06

The separate near-capacity command passed both its warmup and measured request:
32,639 input tokens, 128 output tokens, zero cached prompt tokens,
`truncated=false`, and `stop_type=limit`. The harness wrote `summary.json`.
The measured request took 504.09 seconds, with 67.28 prefill tokens/s and
6.71 decode tokens/s. The warmup took 455.17 seconds and is excluded from that
measurement. This is one capacity sample, not a replacement for the warm means.

System swap was already 247.81 MiB at harness start. It fell to 239.81 MiB after
warmup and stayed there through the measured request. The system swap-out counter
remained 180,352 pages across all saved snapshots; swap-ins increased. Server
RSS ranged from 18,086,272 to 18,328,896 KiB around the requests. These are system
snapshots and process RSS, not continuous peak GPU/unified-memory measurements.
They do not establish zero-swap memory acceptance. Normal development activity,
including Zig compilation during warmup, continued alongside this check.

The [compact follow-up record](../benchmarks/reference-boundary-2026-09-06.json)
retains both timings, explicit acceptance flags/counts, input and response hashes,
and memory counters. Raw evidence is in
`tests/fixtures/boundary-2026-09-06/`; device/allocation evidence is in
`.zig-cache/reference/server-boundary-2026-09-06.log`. The server was stopped
cleanly after completion. This closes the reference boundary check; nuclis's own
32K execution, numerical correctness, and memory acceptance remain pending.

After measurement, the pinned checkout was moved from temporary storage into
`.zig-cache/reference/llama.cpp` and its CMake build regenerated there. The
measurement used the pre-relocation build. `build-before-relocation` preserves
that old build as local evidence; it contains obsolete absolute paths and must
not be used for future runs. Use `build/bin/` from the reconstruction commands.
The relocated `llama-server`, `llama-bench`, and `llama-completion` targets build
successfully. A tiny Metal `llama-bench` smoke run (16 prompt / 4 generated
tokens, one repetition) passed, and CPU fixture regeneration produced identical
files. These relocation checks do not replace or extend the performance baseline.

## Source references

Reference behavior was checked against the pinned
[server API documentation](https://github.com/ggml-org/llama.cpp/blob/7620399f58aebfd2196b74021f9581bcf7218cb9/tools/server/README.md),
[Metal build configuration](https://github.com/ggml-org/llama.cpp/blob/7620399f58aebfd2196b74021f9581bcf7218cb9/ggml/src/ggml-metal/CMakeLists.txt),
and [Qwen35 adapter](https://github.com/ggml-org/llama.cpp/blob/7620399f58aebfd2196b74021f9581bcf7218cb9/src/models/qwen35.cpp).
