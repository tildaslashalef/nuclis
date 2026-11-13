<!-- kind: reference -->
# Model artifacts: sources, conventions, and the models directory

What we download, from where, how it is checked, and where it lives on
disk. What is pinned is the index ([§ The catalogue](#the-catalogue));
each pin's history, and what the upstream guides said when it was taken,
is [the record](../benchmarks/record/catalogue.md).

## Sources

The chat models come from [Unsloth](https://huggingface.co/unsloth)'s GGUF
repositories; the other kinds from their authors' repositories (or a
conversion, as `drama-3`). Unsloth's guides worth reading before touching
an artifact:

- [Dynamic 3.0 GGUFs](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs) —
  the quantization method and its naming.
- [MTP GGUFs](https://unsloth.ai/docs/models/mtp) — multi-token-prediction
  draft heads and how each family packages them.
- [Gemma 4](https://unsloth.ai/docs/models/gemma-4) — variants, official
  sampling, thinking control, memory at four bits.

### Encodings

A filename encoding (`Q4_K_M`, `Q4_K_XL`) names the dominant type only,
and the `UD-` prefix does not say which Dynamic generation made the file;
the repository card does
([record, 2026-09-09](../benchmarks/record/catalogue.md#quantization-generations-recorded-2026-09-09)).
Every tensor's actual encoding is enumerated by `nuclis inspect`
([gguf.md](../engine/gguf.md)); the loader knows
<!-- gen:encodings-count -->23<!-- /gen --> encodings
([quantization.md](../engine/quantization.md)), each adapter accepts its
own executable set for matrices, and anything else is rejected
explicitly, never requantized. A repository file is *runnable* when every
tensor is in its adapter's set (only `inspect` can tell; the filename
cannot) and *supported* only when the catalogue pins it. Every
correctness fixture and benchmark record cites the pinned file.

### Pins and companions

The index pins every file by commit, size, and SHA-256 in
`src/catalog.json`; `nuclis model ls -l` shows them for what is installed,
and `nuclis model inspect <name>` compares them with the Hub's. The
pins that earlier records cite, files outside the index included, are in
[the record](../benchmarks/record/catalogue.md#pinned-commits-and-digests-2026-09-11).

A model's companions are separate files with roles: `mmproj` (the vision,
and for Gemma 4 E4B audio, encoder and projector), `draft` (a
multi-token-prediction head or a DFlash drafter for speculative decoding),
`imatrix` (calibration statistics, never loaded), `codec` and `backbone`.
Companions are loaded only when the configuration names them; nothing is
discovered by scanning a directory. The projector serves image input
([engine/vision.md](../engine/vision.md)) and the draft file speculative
decoding ([engine/speculative-decoding.md](../engine/speculative-decoding.md));
the catalogue and the download sidecars record both, so a pulled model
directory is complete and verified.

## The models directory

Decided 2026-09-11: the layout is the `huggingface` package's
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
([guide/getting-started.md § Model download](../guide/getting-started.md#model-download));
nothing outside `<owner>/<repo>/` is listed, and nuclis never moves user
files itself.

### Provenance sidecars

Beside each file nuclis verified sits `<file>.nuclis.json` (the model's own
name as prefix so listings pair them, a suffix nuclis owns so nothing on the
Hub collides):

```json
{
  "schema_version": 2,
  "repo": "unsloth/Qwen3.8-27B-GGUF",
  "file": "Qwen3.8-27B-UD-Q4_K_M.gguf",
  "revision": "ce331efc48d5722b4b8041f72793e530eae18d9b",
  "sha256": "1bce504dfecd84153666928a41d8a66706f0158d3a317bc5196b72819d7dc529",
  "size": 16564259744,
  "role": "main",
  "artifact": "qwen3.8-27b",
  "downloaded_at": "2026-11-12T13:56:02Z",
  "nuclis_version": "0.12.0-dev"
}
```

- Schema 2 (2026-11-12) adds `artifact` and `variant` (the index
  artifact and draft variant the file belongs to) and `adopted_at`;
  schema 1 sidecars read the same, their `mtp` role as `draft`.
- `revision` is always the resolved 40-character commit, whatever
  `--revision` said (`main`, a tag, a commit), which is what makes the
  SHA-256 beside it meaningful; the index and every benchmark record
  cite commits the same way. Only an adopted file that matched no index
  file has none (`null`).
- `role` is `main`, `mmproj`, `draft`, `imatrix`, `support`, `codec`, or
  `backbone`: from the
  header when it says so (`general.type` `imatrix` or `mmproj`, or a `clip`
  architecture; the main model and a separate draft file both say `model`),
  else from the index entry's role or `--role`, else `main`. A flag that
  contradicts the header is `RoleMismatch`. Safetensors weights are `main`;
  the configuration, tokenizer, and index files a safetensors artifact
  brings are `support` (see *Safetensors artifacts* below).
- `git_blob` (optional, 40 hex) is the Hub's git blob id of a plain git
  file (a config or small tokenizer), which has no SHA-256 on the Hub: the
  pull verified the bytes by it, and `sha256` is the digest computed from
  those bytes. LFS/Xet files omit it; older sidecars lack it and read the
  same.
- Written only after the package's atomic publication, or its full-hash
  verification of a file already in place, succeeded: a sidecar's presence
  means the file beside it was checked against the Hub's catalog. A file
  the user copied in by hand gets one from `pull`, which verifies it, or
  from `model adopt`, which hashes it and records `adopted_at` instead of
  `downloaded_at`. A sidecar recording other content makes `pull` refuse
  (`ExistingFileMismatch`) unless `--force`, which downloads the file
  again and replaces it, then its sidecar, once the new copy verifies.
- `size` lets `model ls` flag a file that changed underneath its sidecar.

### Safetensors artifacts

A safetensors repository is pulled as a set, since the weights alone do not
load: `nuclis model pull <owner/repo> [--file <x>.safetensors]` takes the
file (or its whole `-NNNNN-of-MMMMM.safetensors` shard set) and its
*support files*: `.json`, `.txt`, `.model`, and `.jinja` files in the
set's directory and below, except below a subdirectory holding other
weights of any format (another artifact: GPT-2's `onnx/` export, by its
`.onnx` files) and except an `*.safetensors.index.json` of another set.
Code, pickles, images, and READMEs are never fetched. `--file` names a
weight file: a support file is refused with `NotAWeightFile`, naming the
weights it comes with. Each file lands at its Hub path and gets its own
sidecar, so `convaiinnovations/laya --file model.safetensors` gives:

```text
~/.nuclis/models/convaiinnovations/laya/
  model.safetensors                  (+ .nuclis.json, role main)
  rl_agent_config.json               (+ .nuclis.json, role support)
  encoder/config.json                (+ .nuclis.json, role support)
  eval/results.json                  (+ .nuclis.json, role support)
  tokenizer/tokenizer.json           (+ .nuclis.json, role support)
  tokenizer/tokenizer_config.json    (+ .nuclis.json, role support)
```

`multilingual/model.safetensors` and `typed-decisions/model.safetensors`
are separate choices with their own support files. `model ls` lists the
weights (not the support files). `--register` takes a safetensors artifact
in two layouts. With `rl_agent_config.json` beside the weights (Laya's)
it writes the entry with `"kind": "decision"`, which `nuclis decide`
opens. With a `config.json` that `nuclis speak` opens (a Qwen3-TTS
CustomVoice checkpoint, judged by the same parse once the files are
verified) it writes `"kind": "speech"`; the codec is the separate
`--file speech_tokenizer/model.safetensors` pull. Text commands refuse
both kinds. Any other safetensors artifact is `NotRunnable`, before the
download when no `config.json` comes with it, after it (the files kept,
nothing registered) when its `config.json` is not one of these. `config init` and `config sync` register a safetensors artifact outside the index
only in these two layouts, with its kind; remote `model
inspect` reads GGUF directories only.

### The catalogue

`src/catalog.zig` is the *index*: one table, `artifacts`, of one type
(`catalog.Artifact`), and the only source of "supported". Its data copy,
`src/catalog.json`, is what the gate runner and the bench scripts read
(they import none of nuclis); `make index` rewrites it after an edit to
the table, and a unit test fails until it does
([development.md § Gates](../development.md#gates)). What is
*installed* is what the sidecars under `models/` say; what is *known* is
the index, consulted to pull, to verify, and to supply defaults. A file is
*runnable* when its architecture id has an adapter, *supported* only when
it is a `File` of the index with a pinned commit and SHA-256. Each artifact
has a name (after the model, `qwen3.8-27b`, not the quantization), a kind
(`generation`, shown as "chat", `decision`, `embedding`, `speech`,
`transcription`), a lifecycle, and its files: the main file first, then
its companions by role (`mmproj` for image input, and audio input on the
one artifact marked `hears`, Gemma 4 E4B; `draft` for speculative
decoding; `backbone` and `codec` for the decision and speech models), each
with its own repository and commit when it is not the artifact's.

- **Lifecycle.** `current` artifacts are looked after: they have gates and
  `config init` writes them. A `legacy` artifact (`gemma-4-12b-qat`,
  `muse-glimmer-30b`, decided 2026-11-10) still pulls, verifies, and runs by
  name, but `config init` never writes it, `model ls --available` hides it
  without `--all`, and nothing gates it.
- **Draft variants.** An artifact may carry several draft files, each with
  a variant name and its own speculative verdict (on or off, draft
  length). `qwen3.8-27b` has `mtp` (Unsloth's head, the default) and
  `dflash2` (`z-lab/Qwen3.8-27B-DFlash2-GGUF` at
  `2d9571f8ce46e151f61c6499c99dee6079e1d610`,
  `Qwen3.8-27B-DFlash2-Q8_0.gguf`, 2,056,414,816 B, SHA-256
  `c18e800daedc59ca68fd13b6a856d795746af6d399a9279ac6a277d1d422f87e`;
  on, draft 7, from
  [engine/speculative-decoding.md](../engine/speculative-decoding.md)). A
  file fetched from another repository at a Hub name of its own lands
  where the index places it, beside its parent: the DFlash2 file is
  `unsloth/Qwen3.8-27B-GGUF/MTP/Qwen3.8-27B-DFlash2-Q8_0.gguf`.
  `--draft <variant>` (or `generation.draft` in a registry entry) picks
  one at run time; `--with draft=<variant>` pulls one, `--all` takes the
  default variant of each role, `--all --variants` every variant.
  `--with mtp` and a registry entry's `mtp` key are the old spellings of
  `draft` and print a notice.
- **Installed first.** `nuclis model ls` lists what is installed, grouped
  by kind, one row per artifact (`-l` adds the files, sizes, commits, and
  status); `--available` lists the current index entries that are not
  installed, `--available --all` the legacy ones too. The JSON listing is
  schema 7. `nuclis model adopt <path>` hashes a file the user placed
  under `models/` by hand, matches it to an index file by size and
  digest, and writes the sidecar a pull would have; an unmatched GGUF that
  parses gets a sidecar with no commit, and anything else is refused
  (`NotAModelFile`).
- **The configuration follows.** `nuclis config init` writes a registry
  entry for every installed current chat artifact and judges every other
  installed GGUF by its adapter; `nuclis config sync [--prune] [--dry-run]`
  adds what was installed since and, with `--prune`, drops entries whose
  files are gone ([guide/configuration.md](../guide/configuration.md)).
  `engine.model` defaults to `gemma-4-26b-a4b`, the fastest current chat
  model with a draft that pays on this machine; `config init` names the
  first installed chat model instead when that one is absent.

`nuclis model pull <name> [--with mmproj,draft | --all]` fetches through
the index, and `--model` and `engine.model` resolve names through it.
Status (`present`, `absent`, `mismatch`, `unverified`) is read from
sidecars, never by hashing, so a listing is instant.

<!-- gen:catalogue -->
| Name | Kind | Encoding | Main file | Status |
| --- | --- | --- | --- | --- |
| `qwen3.8-27b` | generation | UD-Q4_K_M | 16.56 GB | current |
| `gemma-4-12b-qat` | generation | Q4_0 (QAT) | 6.72 GB | legacy |
| `gemma-4-26b-a4b` | generation | Q4_0 (QAT) | 14.25 GB | current |
| `gemma-4-e4b-qat` | generation | Q4_0 (QAT) | 4.22 GB | current |
| `muse-glimmer-30b` | generation | UD-Q4_K_XL | 15.88 GB | legacy |
| `laya` | decision | F16 | 842.6 MB | current |
| `laya-multilingual` | decision | F16 | 643.8 MB | current |
| `clef-flash` | decision | Q6_K | 243.5 MB | current |
| `embeddinggemma-2` | embedding | Q8_0 | 309.9 MB | current |
| `qwen3-tts-1.7b-customvoice` | speech | BF16 | 3.83 GB | current |
| `qwen3-tts-0.6b-customvoice` | speech | BF16 | 1.81 GB | current |
| `qwen3-tts-1.7b-base` | speech | BF16 | 3.86 GB | current |
| `qwen3-tts-0.6b-base` | speech | BF16 | 1.83 GB | current |
| `qwen3-tts-1.7b-voicedesign` | speech | BF16 | 3.83 GB | current |
| `drama-3` | speech | Q4_K | 20.79 GB | current |
| `nemotron-asr-0.6b` | transcription | F32 | 2.55 GB | current |
<!-- /gen -->

**Decision artifacts** (`laya`, `laya-multilingual`, `clef-flash`) are
checkpoints only `nuclis decide` opens; `decide.model` defaults to `laya`,
and text commands refuse the names. `nuclis model pull laya` fetches the
weights and their named support files at the pinned commit and refuses
the set if a digest differs. `clef-flash` spans two repositories: the
backbone GGUF (and with `--with mmproj` its projector) from
`bartowski/Cloudflare_clef-flash-GGUF`, pulled first, then the head and
its support files from `Cloudflare/clef-flash`; each lands under its own
repository's directory, and a digest that is not the catalogue's fails
the pull (`CatalogMismatch`).

**The embedding artifact** `embeddinggemma-2` only `nuclis embed` opens:
the text model alone, or with `--with mmproj` the vision and audio
encoders. The entry carries the trained widths (768, 512, 256, 128) and
the modalities its files read; why Q8_0 with a BF16 projector is in
[embeddinggemma.md § Artifacts](embeddinggemma.md#files-and-encodings).
`embed.model` defaults to it. `model pull <owner/repo> --register <name>`
records a GGUF whose header says `gemma-embedding2` as `"kind":
"embedding"`.

**The audio artifacts** are kind `speech` (`nuclis speak`: the five
Qwen3-TTS names and `drama-3`) and `transcription` (`nuclis transcribe`:
`nemotron-asr-0.6b`). A Qwen3-TTS name says CustomVoice (built-in
speakers), Base (a saved voice), or VoiceDesign (a described voice); the
names before 0.11.0, `qwen3-tts-1.7b` and `qwen3-tts-0.6b`, resolve to
nothing, and every command and the server refuse them with the name each
became (`catalog.renamedAudio`). `nuclis model pull <name>` lists the
repository once at the pinned commit and fetches the weights with their
support files and, for a Qwen3-TTS voice, the codec
`speech_tokenizer/model.safetensors` with its own (Drama 3's one GGUF
needs nothing else), each digest checked before anything downloads
(`CatalogMismatch`). The codec's sidecar records the role `codec`.
`--with`, `--all`, `--file`, `--revision`, and `--register` are refused
for these names. `model ls` lists them under `speech` or
`transcription`, `+codec` when a voice's codec is there. `speech.model`
defaults to `qwen3-tts-1.7b-customvoice`, `transcribe.model` to
`nemotron-asr-0.6b`; the other commands refuse these names, and `speak`
and `transcribe` refuse each other's.

How each entry arrived, and the entries that left (`gemma-4-12b`,
`bonsai-2-27b`): [record](../benchmarks/record/catalogue.md#the-entries-and-their-pins-2026-09-11-to-2026-10-30).
