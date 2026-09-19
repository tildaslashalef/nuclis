# Model artifacts: sources, conventions, and the models directory

What we download, from where, how it is quantized, and where it lives on
disk. Facts here are recorded from the upstream repositories and the Unsloth
guides on the dates given; numbers quoted from a guide are that guide's
claims, not our measurements. The pinned artifact of the active plan and its
SHA-256 are in [../development.md § Environment](../development.md#environment).

## Sources

We use [Unsloth](https://huggingface.co/unsloth)'s GGUF repositories. Their
guides worth reading before touching an artifact:

- [Dynamic 3.0 GGUFs](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs) —
  the quantization method and its naming.
- [MTP GGUFs](https://unsloth.ai/docs/models/mtp) — multi-token-prediction
  draft heads and how each family packages them.
- [Gemma 4](https://unsloth.ai/docs/models/gemma-4) — variants, official
  sampling, thinking control, memory at four bits.

### Quantization generations (recorded 2026-09-09)

- **Dynamic 3.0** files carry the `UD-` prefix and are calibrated with a
  published importance matrix (`imatrix_unsloth.gguf_file` in the repo);
  the guide recommends at least `UD-Q2_K_XL` for agentic and tool-calling use
  and warns against 1-bit variants for those cases. Qwen3.8 is the first
  family quantized with 3.0; the pinned `Qwen3.8-27B-UD-Q4_K_M.gguf` is one.
- **Dynamic 2.0 (older)** also uses the `UD-` prefix; the Gemma 4 repos below
  were produced with it. The prefix alone does not identify the generation;
  the repo card does.
- A filename encoding (`Q4_K_M`, `Q4_K_XL`) names the dominant type only.
  Every tensor's actual encoding is enumerated by `nuclis inspect`
  ([gguf-inspection.md](gguf-inspection.md)) and the loader rejects
  unimplemented encodings explicitly; it never requantizes. Implemented
  (checked 2026-09-09 against `quant/decode.zig` and the Qwen binder, Q4_0
  added 2026-09-12 by MODL-08): F32, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL,
  IQ3_S, IQ4_XS, and Q4_0 for matrices (F16 decodes on the CPU but the
  adapters do not accept it for a matrix; the Qwen adapter's executable
  set does not list Q4_0, the Gemma adapter's does); specialized Metal
  matvec and matmul kernels exist for Q3_K, Q4_K, Q5_K, Q6_K, IQ3_S,
  IQ4_XS, and Q4_0, the generic kernels serve the rest. Not implemented:
  Q4_1, Q2_K, BF16, Q8_K, IQ1_*, IQ2_*, IQ3_XXS, MXFP4. A repository file is *runnable* when every tensor is in the
  implemented set (only `inspect` can tell; the filename cannot) and
  *supported* only when the catalogue pins it (see [the catalogue](#the-catalogue)). Every
  correctness fixture and benchmark record cites the pinned file.

### Pinned commits and digests (MODL-02, 2026-09-11)

Resolved and verified by `nuclis model pull` into a scratch home; the
values MODL-03's catalogue and the Gemma 4 12B records cite:

| Repository | Commit | File | Size (B) | SHA-256 |
| --- | --- | --- | ---: | --- |
| `unsloth/Qwen3.8-27B-GGUF` | `4ca720788d1e01f1bff70c033e0d0028fd02e502` | `Qwen3.8-27B-UD-Q4_K_M.gguf` | 16,464,440,224 | `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482` |
| | | `mmproj-BF16.gguf` | 931,146,432 | `83ee4f4f205fa514161778c41df1ea14144faa0f713510893b63c2395f5c2d53` |
| | | `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` | 1,369,590,656 | `50d9ce5a6da381bbcfb31061cf73df94a90e6faf8efeddee379a9cb8f1501c6e` |
| | | `imatrix_unsloth.gguf` | 13,642,656 | `0ee5b10bd0c2fa2127c6f4b43dbfe1efd71e383b63217af9dade1de36599f1c1` |
| `unsloth/gemma-4-12b-it-GGUF` | `fc034cfff751157913579611efad8462ac1be606` | `gemma-4-12b-it-UD-Q4_K_XL.gguf` | 7,366,423,360 | `90fd944d227e9d9b68e7e2c7d5b57b79d4c66ed521b0919fbbd932cf834f6f8e` |
| `unsloth/gemma-4-12B-it-qat-GGUF` | `980b060c40a8539ac159e0501a3e0f66a6365af3` | `gemma-4-12B-it-qat-UD-Q4_K_XL.gguf` | 6,716,356,800 | `90fd44e29e0d7cffeb0fd00dc73cfdab9ed0b0e95306ecf7821ea634c940c370` |
| `unsloth/gemma-4-26B-A4B-it-qat-GGUF` | `7b92b5b28818151e8669af2e45e88d6086f490dd` | `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` | 14,249,047,104 | `a7c5bc715f5ff8e99a3e8901ce7d2b42b402c669bf24f7c5250747633d0f5891` |
| | | `mmproj-BF16.gguf` | 1,194,828,256 | `7b06953ccdbe8cf363f47841a7afaacd2b1c2ff9a8d6b426fdec7521a6878744` |
| | | `MTP/mtp-gemma-4-26B-A4B-it-Q4_0.gguf` | 251,939,328 | `7272d97595f0d4c74bd7b623492b7dbdaafd8b7c72f329a8270ba4eca68f768a` |
| `unsloth/Muse-Glimmer-30B-GGUF` | `faa5b025c584459c13febfa5c59883516710ae39` | `Muse-Glimmer-30B-UD-Q4_K_XL.gguf` | 15,878,222,368 | `82bece304887a313ece08400bc030f6066c7bff5b906b0cd40308ec8a409fd38` |
| | | `mmproj-kquant.gguf` | 1,400,328,928 | `f48b452316f9b213758e8659444029b961a24a07f99a1abb2a9f88b06f7c00c6` |
| | | `dflash-kquant.gguf` | 1,631,205,312 | `27d9a805fa29b943cfb6ad4843367cd4eaaaf06bd452d8cc3e00a2cd18a677bc` |
| `prism-ml/Ternary-Bonsai-2-27B-gguf` | `6ed5e12bf84b7a63069882c91dd9e9218647d17b` | `Ternary-Bonsai-2-27B-PQ2_0.gguf` | 7,206,168,928 | `3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1` |
| | | `Ternary-Bonsai-2-27B-PTQ1_0.gguf` | 5,946,648,928 | `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3` |
| | | `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` | 629,246,976 | `6807ede61d570bb86ba34b756a0fa109edc33668604de867c6ea6d8f1d631903` |
| | | `Ternary-Bonsai-2-27B-mmproj-BF16.gguf` | 931,145,856 | `e287342d92332fa3577ed1d42e921dac9370c08da58ba9337fa450f6cc76cfd7` |

The 26B-A4B and Muse rows were verified by pulls on 2026-09-17 and entered
the catalogue the same day as `gemma-4-26b-a4b` and `muse-glimmer-30b`,
ahead of their adapters. The 26B-A4B is supported since 2026-09-18: the
expert layers run on both backends (MODL-09) and it has its own reference
traces, acceptance record, and agent check (MODL-10;
[bench.md](bench.md#gemma-4-26b-a4b-acceptance-record-modl-10-2026-09-18)).
Since MODL-11 (2026-09-19) the Muse adapter binds the file and the CPU
reference matches the pinned reference's traces on it, so `inspect` says
*supported*; the Metal plan (MODL-12) and the profile with its acceptance
record (MODL-13) landed the same day. Its facts are in
[muse-glimmer.md](muse-glimmer.md). The Bonsai rows were read by `model inspect` on 2026-09-18 and the
PQ2_0 file and its Q8_0 projector verified by the pull the same day;
`bonsai-2-27b` entered the catalogue then: the file is Qwen3.8-27B's
architecture with every matrix in Prism ML's ternary packings (ids 142
and 143) in a Hadamard-rotated basis ([bonsai.md](bonsai.md)). Since
MODL-16 the Qwen adapter binds it and `inspect` says *supported*; since
MODL-17 both backends run it against the fork's traces and it has its
acceptance record and agent check
([bench.md](bench.md#bonsai-2-27b-acceptance-record-modl-17-2026-09-18)).
The PTQ1_0 file was pulled and verified on 2026-09-18 for the packing
measurement (`nuclis model pull prism-ml/Ternary-Bonsai-2-27B-gguf --file
Ternary-Bonsai-2-27B-PTQ1_0.gguf --revision 6ed5e12b…`) and became the
entry's file the same day, measured not slower than PQ2_0 on the whole
token at 1.26 GB less ([bonsai.md](bonsai.md#metal-plan-modl-17-2026-09-18));
the PQ2_0 file stays the traces' source. The repository also lists
`Ternary-Bonsai-2-27B-F16.gguf` (53,808,408,928 B). The 26B-A4B
repository also lists `mmproj-F16.gguf` (1,193,058,784 B), `mmproj-F32.gguf`
(2,291,200,480 B), `MTP/mtp-gemma-4-26B-A4B-it-{BF16,F16}.gguf`
(855,247,360 B each), `MTP/…-Q8_0.gguf` (461,785,600 B), and a root-level
`mtp-gemma-4-26B-A4B-it.gguf` of the Q4_0 head's size. The Muse repository
lists sixteen quantizations from `UD-IQ2_XXS` to `Q8_0` and a two-part
`BF16/`, `mmproj-Muse-Glimmer-30B-{BF16,Q8_0}.gguf` (3,849,173,728 and
2,051,685,088 B), and `dflash-kquant.gguf`: a DFlash drafter, not an MTP
head, carried under the `mtp` companion role because that role names the
draft source the speculative-decoding unit loads, whatever its mechanism.

The Gemma rows are two different checkpoints, read remotely with
`nuclis model inspect` on 2026-09-11; both digests were then verified by
pulls (the K-quant file that day, the QAT file on 2026-09-11 through a
registry entry and again on 2026-09-12 through the catalogue entry, MODL-08).
Both declare architecture
`gemma4`, 48 blocks, 667 tensors, embedding 3840, context 262,144.
`gemma-4-12b-it` is post-training quantized from the bf16 weights: 338
F32 tensors and 329 weight matrices as Q4_K (242), Q5_K (31), and Q6_K
(56), every one in the executable set today. `gemma-4-12B-it-qat` is
Google's quantization-aware-trained checkpoint (the header calls itself
"smart Q4_0, QAT-lossless"): the same 338 F32 tensors and all 329 weight
matrices as Q4_0, executed since MODL-08 (2026-09-12). Its repository (capital
`B` in the id) also carries `mmproj-{BF16,F16,F32}.gguf` (175–210 MB),
`MTP/mtp-gemma-4-12B-it-{Q4_0,Q8_0,BF16,F16}.gguf` (254 MB, 465 MB,
862 MB, 862 MB), and `mtp-gemma-4-12B-it.gguf` (254 MB, the Q4_0 head by
its size). Its pulled companions, verified 2026-09-11 and 2026-09-12:
`mmproj-BF16.gguf` 175,115,840 B, SHA-256
`dcb8103adad042b1bf99df767aaf34eb37c5a73a4a2f0417e4d7ba557e91664f`;
`mtp-gemma-4-12B-it.gguf` 253,708,800 B, SHA-256
`fcb35dea42c71333db904cee11baac525c9ef872818ee3753f6cb156f3c6f4f6`.

### Companion files

| Companion | Purpose | Qwen3.8-27B | Gemma 4 |
| --- | --- | --- | --- |
| Main GGUF | Text model | `Qwen3.8-27B-UD-Q4_K_M.gguf` (pinned) | `gemma-4-12b-it-UD-Q4_K_XL.gguf` (7.37 GB), `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf` (16.9 GB) |
| `mmproj-*.gguf` | Vision encoder and projector | `mmproj-BF16.gguf` (931,146,432 B; `clip`, `qwen3vl_merger`, see [gguf-inspection.md](gguf-inspection.md)), `mmproj-F16.gguf` (927,607,488 B) | 12B: `mmproj-BF16.gguf` and `mmproj-F16.gguf` (175,115,840 B each), `mmproj-F32.gguf` (209,522,240 B); 26B-A4B: `mmproj-BF16.gguf` (1.19 GB) |
| `MTP/mtp-*.gguf` | Multi-token-prediction draft head | `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` (1,369,590,656 B); the main file also embeds one prediction block | 12B: `MTP/mtp-gemma-4-12b-it-BF16.gguf` and `-F16.gguf` (861,520,128 B each), `MTP/mtp-gemma-4-12b-it-Q8_0.gguf` (465,109,248 B), plus a root-level `mtp-gemma-4-12b-it.gguf` of the Q8_0 size; `MTP/README.md`. 26B-A4B: `MTP/mtp-gemma-4-26B-A4B-it.gguf` (462 MB); not embedded |
| `imatrix_unsloth.gguf*` | Calibration statistics, not weights | `imatrix_unsloth.gguf` (13,642,656 B) | `imatrix_unsloth.gguf_file` (7,480,640 B) |

Sizes as listed on Hugging Face on 2026-09-09 (byte-exact where the tree
API reported them; corrected the same day: the Qwen repository does ship
projectors). The Qwen repository also lists a `BF16/` directory and
twenty-four quantizations from `UD-IQ1_S` to `UD-Q8_K_XL` plus `Q4_0`,
`Q4_1`, `Q8_0`; the Gemma 12B repository twenty-two from `UD-IQ2_M` to
`BF16`. Companions are only loaded when the configuration names them
(see [the catalogue](#the-catalogue)); nothing is discovered by scanning a directory. Text
inference loads neither projector nor draft head today (vision is the vision unit,
the draft head is the MTP unit), but the catalogue and the download sidecars
record them from the start so a pulled model directory is complete and
verified before those units exist.

## The models directory

Decided in MODL-02 (2026-09-11): the layout is the `huggingface` package's
`<root>/models/<owner>/<repo>/<file>`, the Hub path without the commit, so
it is predictable from the repository id alone (`Client.localPath`) and a
registry entry can name a file without a per-model naming rule. Every
repository's own subdirectories are kept (`MTP/`), so files that every
repository names identically (`mmproj-BF16.gguf`, `MTP/`) never collide:

```text
~/.nuclis/models/
  unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf
  unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf.nuclis.json
  unsloth/Qwen3.8-27B-GGUF/mmproj-BF16.gguf            (+ .nuclis.json)
  unsloth/Qwen3.8-27B-GGUF/MTP/mtp-Qwen3.8-27B-Q4_0.gguf (+ .nuclis.json)
  unsloth/gemma-4-12b-it-GGUF/gemma-4-12b-it-UD-Q4_K_XL.gguf (+ .nuclis.json)
```

`nuclis model pull <owner/repo> --file <name>` places files there and
`nuclis model ls` lists the layout
([development.md § Model download](../development.md#model-download));
nothing outside `<owner>/<repo>/` is listed, and nuclis never moves user
files itself.

### Provenance sidecars

Beside each file nuclis verified sits `<file>.nuclis.json` (the model's own
name as prefix so listings pair them, a suffix nuclis owns so nothing on the
Hub collides):

```json
{
  "schema_version": 1,
  "repo": "unsloth/Qwen3.8-27B-GGUF",
  "file": "Qwen3.8-27B-UD-Q4_K_M.gguf",
  "revision": "4ca720788d1e01f1bff70c033e0d0028fd02e502",
  "sha256": "322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482",
  "size": 16464442496,
  "role": "main",
  "downloaded_at": "2026-09-11T09:12:33Z",
  "nuclis_version": "0.1.0-dev"
}
```

- `revision` is always the resolved 40-character commit, whatever
  `--revision` said (`main`, a tag, a commit), which is what makes the
  SHA-256 beside it meaningful; the MODL-03 catalogue and every benchmark record
  cite commits the same way.
- `role` is `main`, `mmproj`, `mtp`, or `imatrix`: from the header when it
  says so (`general.type` `imatrix` or `mmproj`, or a `clip` architecture;
  the main model and the separate MTP head both say `model`), else from
  `--role` (the MODL-03 catalogue later), else `main`. A flag that contradicts
  the header is `RoleMismatch`.
- Written only after the package's atomic publication, or its full-hash
  verification of a file already in place, succeeded: a sidecar's presence
  means the file beside it was checked against the Hub's catalog. Nothing
  is written for a file the user copied in by hand until a `pull` verifies
  it. A sidecar recording other content makes `pull` refuse
  (`ExistingFileMismatch`) unless `--force` replaces file and sidecar.
- `size` lets `model ls` flag a file that changed underneath its sidecar.

### The catalogue

`src/catalog.zig` is the only source of "supported": a file under
`models/` is *runnable* when its architecture id has an adapter,
*supported* only when the catalogue pins it (name, repository, file,
commit, SHA-256, size, quantization, architecture, sampling profile, and
companions with the unit that will load them: `mmproj` by the vision unit, `mtp` by
the MTP unit). Entries are named after the model (`qwen3.8-27b`), not the
quantization. `nuclis model pull <name> [--with mmproj,mtp | --all]`
fetches through it, `--model` and `engine.model` resolve names through it,
and `model ls` reports every entry's status from sidecars (`present`,
`absent`, `mismatch`, `unverified`) with the other GGUF files in the
layout beneath. The Qwen entry was proven from a clean state on
2026-09-11: the old `models/qwen/` directory was deleted and
`nuclis model pull qwen3.8-27b --all` rebuilt the model directory (MODL-03).
Gemma 4 12B entered the catalogue as `gemma-4-12b` with its profile on
2026-09-12 (MODL-07) naming the K-quant file with `mmproj-BF16.gguf`
(175,115,840 B, SHA-256 `2e269f906eb15169ee9ce880ea649bd6d42d4964c21f8ede10d0d0efc738bcbb`)
and `mtp-gemma-4-12b-it.gguf` (465,109,248 B,
`145db9094bc0f85f1701e255a2ed216dcc9800fc8bc8631ad00905b456bd451b`);
later that day (MODL-08) that one entry moved, name unchanged, to the QAT
file. Since 2026-09-12 there are **two** entries instead, so each
quantization has a name of its own: `gemma-4-12b` is the K-quant release
again (the pins of MODL-07, above) and `gemma-4-12b-qat` is the QAT file
`gemma-4-12B-it-qat-UD-Q4_K_XL.gguf` with the companions listed above.
Both are pinned, both are verified by `nuclis model pull <name> --all`,
and each has its own reference traces and acceptance record — which is
what the single name obscured while it pointed at whichever file was
newest. `gemma-4-26b-a4b`, the mixture of experts, followed on 2026-09-17
with its adapter arriving the next day; its verdict is *supported* since
its acceptance record (MODL-10). `nuclis model inspect`
judges a remote file from its directory alone and compares the Hub's
digest against the catalogue (main file or companion) for its verdict
([development.md § Model download](../development.md#model-download));
the `models` registry in `nuclis.json` names models above the catalogue
with per-model overrides, pinning no digest
([development.md § Configuration file](../development.md#configuration-file)).
