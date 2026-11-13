<!-- kind: record -->
# Qwen3.8-27B: the record

The dated record behind [the Qwen3.8 card](../../models/qwen3.8.md): the first validation of the binding and the re-made file, as written on their dates.

## Evidence and references (2026-09-06)

On 2026-09-06, the downloaded artifact passed the adapter and produced the counts
above. A 39 KB [descriptor fixture](../../../inference/src/models/fixtures/qwen35-27b.json)
captures actual metadata and every tensor's name, dimensions, and encoding.
It contains no weight values, tokens, or private prompts. Tests hydrate this
independent inventory rather than generating the expected layout from the binder.

Debug and ReleaseSafe validation both pass on the actual file. A process-level
negative check used a temporary sparse file with the same directory but an
altered attention interval: generic `inspect` succeeded, while `validate`
returned `UnsupportedConfiguration` with no success JSON. No weights were copied.

Tests cover the hybrid schedule, borrowed references, text/auxiliary accounting,
changed dimensions, wrong types, missing metadata, RoPE overrides, missing and
unexpected tensors, altered query/gate shapes, forbidden encodings, auxiliary
tensor errors, and allocation failure cleanup. Default tests require no model
download or GPU.

Layout interpretation was checked against the
[Qwen35 implementation at llama.cpp revision 7620399](https://github.com/ggml-org/llama.cpp/blob/7620399f58aebfd2196b74021f9581bcf7218cb9/src/models/qwen35.cpp)
and the [upstream model configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json).
The former distinguishes main and auxiliary blocks and accounts for gated query
dimensions. That revision was also the reference the first families were
brought up and measured against
([README.md § 4](../../models/README.md#4-validate-against-nuclis-itself)). No external
implementation code was copied into this adapter.

## The re-made file (2026-11-12)

On 2026-11-11 Unsloth re-made the 2- to 6-bit files of
`unsloth/Qwen3.8-27B-GGUF`. The index pins the new `UD-Q4_K_M` since
2026-11-12: commit `ce331efc48d5722b4b8041f72793e530eae18d9b`,
16,564,259,744 B, SHA-256 `1bce504d…c529`, in place of `322e194f…3482` at
`4ca72078` ([catalogue.md § Pinned commits and digests](catalogue.md#pinned-commits-and-digests-2026-09-11)).
It is the first file re-pinned under
[README.md § Re-pinning a file](../../models/README.md#re-pinning-a-file).

- **Same model, different mix.** The same architecture, 19 `qwen35.*`
  keys, 866 tensors, and nine encodings (now F32 360, Q8_0 107, Q3_K 2,
  Q4_K 74, Q5_K 142, Q6_K 28, IQ4_NL 9, IQ3_S 2, IQ4_XS 142); 99.8 MB
  more in the text decoder. The tokenizer gate passes, and the chat
  template is byte-identical (SHA-256 `12827f24…`), so the profile and
  its fixtures stand. The projector and the MTP file are byte-identical
  at the new commit.
- **Cross-encoding check: not run.** The first pin was not on the
  machine and no encoding is new.
- **Fixtures, from the CPU reference** (ReleaseFast): the `Hello,` traces
  (`tests/fixtures/reference-hello-comma`, 6 s), the MTP rows
  (`qwen35-mtp`, 40 s; drafts still 9419, 271), and the DFlash2 rows
  (`tests/fixtures/qwen38-dflash2`, 12 s; anchor 353, drafts 2688,
  14169, 11, 694, 353, 1144, 310). The vision check's greedy tokens are
  unchanged (224 s on the CPU).
- **Perplexity.** Metal over eight windows of 512: 6.6320 ± 0.3595,
  against 6.6696 for the first pin's llama-perplexity reference (nuclis
  matched it at −0.000 %): 0.56 % lower, so no worse. That run is the
  pinned reference (`tests/fixtures/perplexity/qwen38-wikitext2-c512x8.json`),
  cross-checked by one CPU window: 4.2975 against Metal's 4.2978,
  −0.006 %, in 1,701 s (eight would take about 3.8 h; a run of all eight
  was stopped after 1 h 50 min).
- **Gates.** All 25 `qwen38-*` gates pass on the new fixtures, every
  tier (2,289 s): F32 traces at 6.1e-5 max abs, the DFlash2 rows at
  1.5e-2, verify depth to 32,638.
- **Speed**, a new baseline: decode 11.79 / 11.41 tok/s and prefill
  94.22 / 86.46 tok/s at 512 / 4,096; MTP 1.61× / 1.42× and DFlash2
  1.63× / 1.53× at draft 7, code 1.65× and 1.69×
  ([history.md § The re-made Qwen3.8 file](../history.md#the-re-made-qwen38-file-2026-11-12)).
