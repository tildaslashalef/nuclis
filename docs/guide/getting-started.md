# Getting started

From nothing to a chat with a local model: install nuclis, fetch a model,
and know where it keeps things. What each command does is in `nuclis
--help`; every setting is in [configuration.md](configuration.md).

## Install

nuclis runs on macOS on Apple silicon. The default model, Qwen3.8-27B, is
a 16.5 GB file (about 18 GB with its drafter and vision companions) and
runs in 48 GB of memory; the smaller families need much less
([models/catalogue.md](../models/catalogue.md)).

**From a release.** Download the tarball for the newest release from
[Releases](https://github.com/tildaslashalef/nuclis/releases), then:

```sh
gh attestation verify nuclis-v<version>-aarch64-macos.tar.gz --repo tildaslashalef/nuclis   # optional
tar xzf nuclis-v<version>-aarch64-macos.tar.gz
cd nuclis-v<version>-aarch64-macos
xattr -d com.apple.quarantine nuclis   # the binary is not notarized
```

Each tarball carries a build provenance attestation (the first line shows
the workflow that built it) and is listed in the release's `SHA256SUMS`.

**From source.** Zig 0.17.0 and the Command Line Tools:

```sh
git clone https://github.com/tildaslashalef/nuclis
cd nuclis && make metal       # ./zig-out/bin/nuclis, release build, Metal backend
make install                  # optional: ~/.local/bin (PREFIX= elsewhere)
```

`nuclis completion fish|bash|zsh` prints Tab completion for that shell.

## A first run

```sh
nuclis model pull qwen3.8-27b --all    # pinned commit, SHA-256 verified, with its companions
nuclis config init --discover          # ~/.nuclis/nuclis.json, every model found
nuclis agent                           # the chat, with tools, in this directory
```

`nuclis generate`, `bench`, `eval`, `decide`, `embed`, and `serve` cover
the rest: [eval.md](eval.md) for perplexity, [api.md](api.md) for the HTTP
API.

### Embeddings

`nuclis embed` turns text into vectors for search and similarity, with
EmbeddingGemma 2 (310 MB; `--with mmproj` adds the image and audio
encoders, used once those inputs land):

```sh
nuclis model pull embeddinggemma-2
nuclis embed "a cat" "a kitten" "a tax form"                  # three vectors and their cosines
nuclis embed --task search_query "how do auroras form?"       # a query, with the model's prefix
nuclis embed --task document --input-file docs.jsonl --json   # one JSON string per line, every vector
```

Each argument is one input, embedded exactly as given unless `--task`
names a use: queries and documents get different prefixes, and only
vectors of one **space** compare. The report prints it,
`embeddinggemma-2@6f1bd4ac6c5d/768`: the model, its file's digest, and the
width (`--dimensions` 512, 256, or 128 keeps the leading values).

## User directories

Everything nuclis keeps lives under one root, `~/.nuclis`. Set
`NUCLIS_HOME` to an absolute path to move it (an empty or relative value is
an error).

```text
~/.nuclis/
  nuclis.json              your settings (`nuclis config init` writes it)
  models/<owner>/<repo>/   downloaded models, each file with a <file>.nuclis.json
                           sidecar recording where it came from and its digest
  agent/
    history.jsonl          prompts you submitted, across sessions
    sessions/<cwd-slug>/   one file per conversation, per working directory
    exports/               `nuclis agent export` output
  cache/prefix/            the agent's saved model states (`nuclis cache` manages them)
```

- **Nothing is created until it is needed.** `--help` and inspection
  commands write nothing, and existing files are never migrated or
  overwritten behind your back.
- **Credentials come from the environment only:** `HF_TOKEN`, for gated
  Hugging Face repositories. Public ones need none.

More on each: the models layout in [models/catalogue.md](../models/catalogue.md),
the settings in [configuration.md](configuration.md), sessions in
[spec.md § Sessions and storage](../spec.md#74-sessions-and-storage), the
saved states in [engine/session.md § The agent's token cache](../engine/session.md#the-agents-token-cache-2026-10-04).

## Model download

### Pull a model nuclis supports

The catalogue is the list of models nuclis ships support for, each pinned
to an exact commit and SHA-256 digest:

| Name | What it is |
| --- | --- |
| `qwen3.8-27b` | Qwen3.8-27B, the default model (16.5 GB) |
| `gemma-4-12b-qat` | Gemma 4 12B, quantization-aware (6.7 GB) |
| `gemma-4-26b-a4b` | Gemma 4 26B-A4B, a mixture of experts (14.3 GB) |
| `gemma-4-e4b-qat` | Gemma 4 E4B, the small one (4.2 GB) |
| `muse-glimmer-30b` | Muse Glimmer 30B (15.9 GB) |
| `laya`, `laya-multilingual` | decision encoders for `nuclis decide` |
| `clef-flash` | Cloudflare's decision model for `nuclis decide` |
| `embeddinggemma-2` | EmbeddingGemma 2, text into vectors for `nuclis embed` (310 MB) |

```sh
nuclis model pull qwen3.8-27b             # the model alone
nuclis model pull qwen3.8-27b --all       # with its drafter and vision companions
nuclis model pull qwen3.8-27b --with mtp  # or pick them: mmproj, mtp
nuclis model ls                           # what is here, and whether it verifies
```

Files land in `~/.nuclis/models/<owner>/<repo>/<file>`, each SHA-256
verified before it is published. If the Hub's digest at the pinned commit
is not the catalogue's, the pull stops before writing a sidecar
(`CatalogMismatch`).

### Pull any other model from Hugging Face

```sh
nuclis model pull <owner/repo> --file <name> [--revision <rev>]
nuclis model pull <owner/repo> --file <name> --register mymodel   # and name it
```

- The revision (`main` by default; a tag, branch, or commit) is resolved
  once to a commit, printed, and recorded in the sidecar.
- Without `--file`, a repository with several files lists them with their
  sizes and stops (`SelectionRequired`). Names match in full, folders
  included: `--file MTP/mtp-Qwen3.8-27B-Q4_0.gguf`.
- A safetensors set comes with its config and tokenizer files
  ([models/catalogue.md § Safetensors artifacts](../models/catalogue.md#safetensors-artifacts)).
- `--role mmproj|mtp|imatrix` marks a companion file.
- `--register <name> [--profile <p>]` writes the pull into `nuclis.json` as a
  named model once every file verifies, so `--model <name>` and `nuclis
  config set engine.model <name>` work from then on; a companion `--role`
  fills the same entry's `mmproj` or `mtp`. A name that already locates
  other content is refused, and so is a catalogue name unless the pull is
  that catalogue entry's own file.

Whether such a file actually runs depends on its architecture having an
adapter; `nuclis model inspect` tells you before you download (below).

### Pull a model you named in your settings

`nuclis model pull <entry>` pulls a model entry from `nuclis.json` (its
`repo`, `file`, and `revision`, `main` when unset; `--with` and `--all`
add its companions). An entry that points at a local `path` has nothing to
pull (`NotPullable`); a name that is neither an entry, a catalogue name,
nor `owner/repo` is `UnknownModel`. The entries are described in
[configuration.md § Configuration file](configuration.md#configuration-file).

### What happens on a second pull, and on failure

- **Pulling again** re-hashes the file and downloads nothing.
- **A file that does not match** its sidecar, or that nuclis never verified,
  is refused (`ExistingFileMismatch`); `--force` replaces it.
- **Progress** is one updating line (bytes, rate, time left) on a terminal,
  plain lines otherwise. **Ctrl-C** stops the download and removes the
  partial file; a second Ctrl-C kills the process and may leave a temporary
  file behind.
- Every form takes `--json` for scripts.

The download itself is the `huggingface` package's work (Xet
reconstruction, SHA-256 verification, atomic publication):
[its README](../../huggingface/README.md).

### Check a model before downloading it

```sh
nuclis model inspect gemma-4-e4b-qat
nuclis model inspect <owner/repo> --file <name> [--revision <rev>]
```

It reads only the head of the file from the Hub, in 8 MiB windows, until
the GGUF directory parses (a few seconds; nothing is written), prints what
`nuclis inspect` prints for a local file, and ends with a verdict:

- **supported**: in the catalogue with the Hub's digest, and an adapter
  binds it;
- **runnable**: an adapter for its architecture binds it, but it is not a
  catalogue file;
- **not runnable**: names the first tensor or encoding nuclis cannot run,
  the missing adapter, or the adapter's reason.

### What is on disk

`nuclis model ls [--json]` (or `make model-ls`) lists the catalogue with each
model's status (`present`, `absent`, `mismatch`, or `unverified` when the
file has no sidecar) and its companions, then every other model file in the
folder with its facts, and marks the files your settings name (`registered
as <name>`). `--model` and `engine.model` accept a settings entry, a
catalogue name, or a path; a missing file fails before anything loads,
naming the path it looked for.
