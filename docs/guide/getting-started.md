<!-- kind: guide -->
# Getting started

From nothing to a chat with a local model: install nuclis, fetch a model,
and know where it keeps things. What each command does is in `nuclis
--help`; every setting is in [configuration.md](configuration.md).

## Install

nuclis runs on macOS on Apple silicon, and as a CPU-only build on x86-64
Linux (AVX2, glibc 2.35 or newer), where it serves the small models such as
EmbeddingGemma 2; images, audio, and the GPU need the macOS build. The
default model, Qwen3.8-27B, is a 16.6 GB file (20.9 GB with its drafters
and vision companion) and runs in 48 GB of memory; the small chat model,
Gemma 4 E4B, is 4.2 GB, and the other kinds need less
([models/catalogue.md](../models/catalogue.md)).

**From a release.** Download the tarball for the newest release from
[Releases](https://github.com/tildaslashalef/nuclis/releases), then:

```sh
gh attestation verify nuclis-v<version>-aarch64-macos.tar.gz --repo tildaslashalef/nuclis   # optional
tar xzf nuclis-v<version>-aarch64-macos.tar.gz
cd nuclis-v<version>-aarch64-macos
xattr -d com.apple.quarantine nuclis   # the binary is not notarized
```

On Linux, the same with `x86_64-linux` in the name and no `xattr` line.

Each tarball carries a build provenance attestation (the first line shows
the workflow that built it) and is listed in the release's `SHA256SUMS`.

**From source.** Zig 0.17.1 and the Command Line Tools:

```sh
git clone https://github.com/tildaslashalef/nuclis
cd nuclis && make metal       # ./zig-out/bin/nuclis, release build, Metal backend
make install                  # optional: ~/.local/bin (PREFIX= elsewhere)
```

`nuclis completion fish|bash|zsh` prints Tab completion for that shell.

## A first run

```sh
nuclis model pull qwen3.8-27b --all  # pinned commit, SHA-256 verified, with its companions
nuclis config init                   # ~/.nuclis/nuclis.json, every model installed
nuclis agent                         # the chat, with tools, in this directory
```

`nuclis generate`, `bench`, `eval`, `decide`, `embed`, and `serve` cover
the rest: [eval.md](eval.md) for perplexity, [api.md](api.md) for the HTTP
API.

### Embeddings

`nuclis embed` turns text, images and audio into vectors for search and
similarity, with EmbeddingGemma 2 (310 MB; `--with mmproj` adds the image
and audio encoders, 982 MB):

```sh
nuclis model pull embeddinggemma-2 --with mmproj
nuclis embed "a cat" "a kitten" "a tax form"                  # three vectors and their cosines
nuclis embed --image cat.jpg "a photo of a cat"               # an image and a caption, one space
nuclis embed --audio memo.m4a "the shopping list"             # a voice memo against a text
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
saved states in [app/agent.md § 12](../app/agent.md#12-the-agents-token-cache).

## Model download

### Pull a model nuclis supports

The index is the list of models nuclis ships support for, each pinned
to an exact commit and SHA-256 digest (`nuclis model ls --available`
lists the ones not installed):

<!-- gen:catalogue -->
| Name | Kind | Role | Support | Encoding | Main file |
| --- | --- | --- | --- | --- | --- |
| `qwen3.8-27b` | generation | default | verified | UD-Q4_K_M | 16.56 GB |
| `gemma-4-12b-qat` | generation | extra | archived | Q4_0 (QAT) | 6.72 GB |
| `gemma-4-26b-a4b` | generation | extra | archived | Q4_0 (QAT) | 14.25 GB |
| `gemma-4-e4b-qat` | generation | extra | verified | Q4_0 (QAT) | 4.22 GB |
| `inkling-2-mini` | generation | extra | verified | UD-Q4_K_XL | 17.63 GB |
| `muse-glimmer-30b` | generation | extra | archived | UD-Q4_K_XL | 15.88 GB |
| `laya` | decision | default | verified | F16 | 842.6 MB |
| `laya-multilingual` | decision | extra | archived | F16 | 643.8 MB |
| `clef-flash` | decision | extra | verified | Q4_K_M | 6.73 GB |
| `embeddinggemma-2` | embedding | default | verified | Q8_0 | 309.9 MB |
| `qwen3-tts-1.7b-customvoice` | speech | default | verified | BF16 | 3.83 GB |
| `qwen3-tts-0.6b-customvoice` | speech | extra | archived | BF16 | 1.81 GB |
| `qwen3-tts-1.7b-base` | speech | extra | archived | BF16 | 3.86 GB |
| `qwen3-tts-0.6b-base` | speech | extra | archived | BF16 | 1.83 GB |
| `qwen3-tts-1.7b-voicedesign` | speech | extra | archived | BF16 | 3.83 GB |
| `drama-3` | speech | extra | archived | Q4_K | 20.79 GB |
| `voicetut-tts` | speech | extra | verified | Q8_0 | 1.35 GB |
| `nemotron-asr-0.6b` | transcription | default | verified | F32 | 2.55 GB |
<!-- /gen -->

Each kind has its command: `generation` models chat (`nuclis agent`,
`generate`), `decision` models answer `nuclis decide`, `embedding` models
`nuclis embed`, `speech` models `nuclis speak` (in voices saved with `nuclis
voice add`, on the Bases and Drama 3), and `transcription` models `nuclis
transcribe`. What each one is: [models/catalogue.md](../models/catalogue.md).
Each kind has one *default*, the model its configuration key names; every
other entry is an *extra*. A *verified* model is gated by every release; a
*preview* one is gated too but prints why it rests on something still
moving when it runs; an *archived* one pulls, verifies, and runs by name
with no gate required, but `--catalogue` never pulls it and `model ls
--available` shows it only with `--all`.

```sh
nuclis model pull qwen3.8-27b                     # the model alone
nuclis model pull qwen3.8-27b --all               # with its default draft and vision companions
nuclis model pull qwen3.8-27b --with draft=mtp   # or pick them: mmproj, draft[=variant]
nuclis model pull qwen3.8-27b --all --variants    # every draft variant too
nuclis model pull --catalogue                     # each kind's default with --all, about 29 GB
nuclis model pull --catalogue --extra             # and every extra not archived, about 33 GB more
nuclis model ls                                   # what is here
```

`--catalogue` sets a machine up in one command (`--extra` adds the
extras that are not archived): it prints the plan (each
entry's files, and what is already here) before the first download, skips
files already verified, and when an entry fails reports it and goes on to
the next. `nuclis config init` then registers the verified chat models
installed; every other entry resolves by its index name. On the Linux
CPU build, pull `gemma-4-e4b-qat` by name or add `--extra`: plain
`--catalogue` fetches only the 27B for chat, sized for the macOS GPU.

Files land in `~/.nuclis/models/<owner>/<repo>/<file>`, each SHA-256
verified before it is published. If the Hub's digest at the pinned commit
is not the catalogue's, the pull stops before writing a sidecar.

### Pull any other model from Hugging Face

```sh
nuclis model pull <owner/repo> --file <name> [--revision <rev>]
nuclis model pull <owner/repo> --file <name> --register mymodel   # and name it
```

- The revision (`main` by default; a tag, branch, or commit) is resolved
  once to a commit, printed, and recorded in the sidecar.
- Without `--file`, a repository with several files lists them with their
  sizes and stops. Names match in full, folders
  included: `--file MTP/mtp-Qwen3.8-27B-Q4_0.gguf`.
- A safetensors set comes with its config and tokenizer files
  ([models/catalogue.md § Safetensors artifacts](../models/catalogue.md#safetensors-artifacts)).
- `--role mmproj|draft|imatrix` marks a companion file (`mtp` is the old
  spelling of `draft`).
- `--register <name> [--profile <p>]` writes the pull into `nuclis.json` as a
  named model once every file verifies, so `--model <name>` and `nuclis
  config set engine.model <name>` work from then on; a companion `--role`
  fills the same entry's `mmproj` or `draft`. A name that already locates
  other content is refused, and so is a catalogue name unless the pull is
  that catalogue entry's own file.
- A Qwen3-TTS CustomVoice fine-tune registers as a speech model
  (`speech.model`, `nuclis speak --model <name>`); its codec is a second
  pull of the same repository:

  ```sh
  nuclis model pull itshamdi404/Egy_Arabic_Qwen3-TTS-12Hz-1.7B-Base --file model.safetensors --register qwen3-tts-1.7b-eg
  nuclis model pull itshamdi404/Egy_Arabic_Qwen3-TTS-12Hz-1.7B-Base --file speech_tokenizer/model.safetensors
  nuclis speak --model qwen3-tts-1.7b-eg --voice egyptian_speaker "ازيك؟" -o egy.wav
  ```

Whether such a file actually runs depends on its architecture having an
adapter; `nuclis model inspect` tells you before you download (below).

### Pull a model you named in your settings

`nuclis model pull <entry>` pulls a model entry from `nuclis.json` (its
`repo`, `file`, and `revision`, `main` when unset; `--with` and `--all`
add its companions). An entry that points at a local `path` has nothing to
pull, and a name that is neither an entry, a catalogue name, nor
`owner/repo` is refused. The entries are described in
[configuration.md § Configuration file](configuration.md#configuration-file).

### What happens on a second pull, and on failure

- **Pulling again** re-hashes the file and downloads nothing.
- **A file that does not match** its sidecar, or that nuclis never verified,
  is refused.
- **`--force`** downloads every file of the pull again, verified ones
  included, and replaces each only once the new copy verifies: the old
  file stays in place until then, and a failed or interrupted download
  leaves it as it was. The sidecar is rewritten after; the report marks
  each file `xet` or `http`, never `verified_local`.
- **Progress** is a line per file on stderr: on a terminal it updates in
  place (percentage, bytes, rate, time left) and ends as
  `✓ <file>  <size> in <time> · <rate>`; otherwise a timestamped line when
  a download starts and one when the file ends
  ([command-output.md](../app/command-output.md)). A file downloads into `<file>.partial` beside its
  destination and is renamed into place once its digest verifies.
- **Ctrl-C** (SIGINT) stops the pull and removes the partial file; a second
  Ctrl-C kills the process and may leave `<file>.partial` behind, which the
  next pull starts over.
- **SIGTERM** (an app quitting, `kill`) stops the pull and keeps
  `<file>.partial` with `<file>.partial.json` (repository, commit, digest,
  size, bytes written). The next pull of the same file at the same commit
  and digest hashes those bytes again and fetches only the rest; the whole
  file is still verified before it is published, so a damaged partial fails
  and is removed (pull again to download it whole). A partial of another
  commit or digest is discarded and the file starts over; `--force` ignores
  partials. Two pulls of the same file at once: the second is refused while
  the first holds the partial. The report's `resumed_bytes` says how much of
  a file came from a kept partial.
- Every form takes `--json` for scripts: the report on stdout, and the
  progress on stderr as one JSON object per line instead of the text, for
  a program that shows its own (Muon's first-run window):

  ```json
  {"event":"downloading","file":"model.safetensors","index":0,"count":11,"done":107578,"size":1811626576,"pull_done":107578,"pull_size":2498383610}
  ```

  `event` is `resolving`, `checking` (for a file already in place),
  `resuming` (the first line of a file continued from a kept partial, its
  bytes already in `done`), `downloading` (at most every 250 ms, and when
  the file is whole),
  `verifying`, `reused` (a file in place that verified), `complete` (the
  file is published), or `interrupted`; `file` is its Hub name, `index`
  of `count` the files of the pull (a safetensors set counts its
  support files), `done` of `size` this file's bytes, `pull_done` of
  `pull_size` the whole pull's.

The download itself is the `huggingface` package's work (Xet
reconstruction, SHA-256 verification, atomic publication):
[its README](../../huggingface/README.md).

### Check a model before downloading it

```sh
nuclis model inspect gemma-4-e4b-qat
nuclis model inspect <owner/repo> --file <name> [--revision <rev>]
nuclis model inspect https://huggingface.co/<owner>/<repo>/blob/<rev>/<file>
```

A pasted Hub URL (a `blob`, `resolve`, or repository link) is the same as
the repository, `--file`, and `--revision`, and `nuclis inspect --model
<url>` does the same. It reads only the head of the file from the Hub, in
8 MiB windows, until the GGUF directory parses (a few seconds; nothing is
written), prints what `nuclis inspect` prints for a local file with the
Hub's card (pipeline, library, `config.json` architectures), and ends with
a verdict. A safetensors file gets every shard's header and the scalars of
its `config.json` instead; nuclis generates text from GGUF, so its verdict
is not runnable and says what the Hub lists it as.

The verdict:

- **supported**: in the catalogue with the Hub's digest, and an adapter
  binds it (a decision, embedding, or speech model: its own command runs
  it);
- **runnable**: an adapter for its architecture binds it, but it is not a
  catalogue file;
- **not runnable**: names the first tensor or encoding nuclis cannot run,
  the missing adapter, or the adapter's reason.

`--against <ref>` compares the file with another (an index name, a Hub
URL, or a local file): the keys and tensors only one side has, the values,
shapes, and block sets that differ, and whether that makes it the same
model, a configuration of one nuclis runs, or a new family.

`--layout` prints the block structure from the tensors alone: each kind
of block (its tensors, shapes, and encodings, each dimension named by the
metadata or `config.json` value it equals), and their schedule, such as
`(A A A B) × 16 · C` for three delta-net blocks to one attention block,
then the draft head. Shapes are as the file stores them: a GGUF's
innermost dimension first, a safetensors file's outermost first.

`--context <n,…>` estimates the memory at those context lengths (`max` is
the declared one): the weights, plus each block's own K and V widths per
token at f16 and at q8_0, against this machine's memory. A block that
shares another's KV adds nothing; recurrent state is constant and not
counted; a declared sliding window is not applied, so the cache is then an
upper bound. Activations and scratch buffers are not included.

`--pull` then fetches the file through `nuclis model pull`, pinned to the
commit the inspection read, when the verdict is supported or runnable;
otherwise nothing is downloaded. It takes pull's flags (`--register`,
`--with`, `--force`).

`--metadata`, `--tensors`, and `--tensor <name>` add every metadata key,
every tensor's encoding and shape, or one tensor's first values with its
range, mean, RMS, and NaN, Inf, and zero counts (over a GGUF tensor's first
row, or a safetensors tensor's first 1,048,576 values; that much is read
from the Hub), as `nuclis inspect` shows them for a local file. The
statistics are numbers, not a reading: a norm stored as its scale can
average near 0 (Qwen 3's first input norm averages 0.011).

### What is on disk

`nuclis model ls [--json]` (or `make model-ls`) lists what is installed,
grouped by kind (chat, speech, transcription, decision, embedding), one
row per model with its quantization, size, and companions (`+mmproj
+draft`); a file the index does not pin is marked `not in the index`, and
the one `engine.model` names `engine.model` (JSON `configured`). `-l` adds every file with its commit, digest,
path, and status (`present`, `absent`, `mismatch`, or `unverified` when it
has no sidecar), and for an index model its one-line summary. `--available`
lists the index, installed or not, by kind with each kind's default first
(a ROLE column, and SUPPORT when a row is a preview or archived);
`--available --all` adds the archived entries. `--model` and `engine.model`
accept a settings entry, an index name, or a path; a missing file fails
before anything loads, naming the path it looked for.

A file copied into `~/.nuclis/models` by hand has no sidecar, so it lists
as `unverified`. `nuclis model adopt <path> [--force]` hashes it, matches
it to an index file by size and SHA-256, and writes the sidecar a pull
would have written; a GGUF the index does not know gets a sidecar with no
commit, and a file that is not a model is refused.

### Remove a model

```sh
nuclis model rm qwen3.8-27b               # the model file alone
nuclis model rm qwen3.8-27b --with mmproj # and the companions named: mmproj, draft
nuclis model rm qwen3.8-27b --all         # and every companion
nuclis model rm laya-multilingual --json  # the report as one JSON object
```

`nuclis model rm <name>` takes a catalogue name or a settings entry and
deletes its files with their `.nuclis.json` sidecars: a catalogue entry's
main file, its support files (configuration, tokenizer), and a speech
model's codec; companions only with `--with` or `--all`,
as `pull` takes them. A settings entry removes its file and, asked, the
`mmproj` and `draft` it names; a file outside `~/.nuclis/models` is never
touched. The directories the removal leaves empty go too.

A file another model still uses stays, and the report names that model: a
catalogue entry whose file is present, or a settings entry of another name
pointing at the file. The settings entry of the same name that `config
init` writes for an installed index model is the same model, not another user.
Your settings are not edited: `nuclis model ls` then lists the entry among
those without a file. A name with nothing on disk is an error
(`NotPulled`); an entry only partly present loses what is there.

With `--json` the report is one object on stdout, absolute paths, every
sidecar its own `removed` item (here the four support files' eight items
are left out):

```
{
  "schema_version": 1,
  "name": "laya-multilingual",
  "models_dir": "/Users/you/.nuclis/models",
  "removed": [
    { "path": "/Users/you/.nuclis/models/convaiinnovations/laya/multilingual/model.safetensors", "bytes": 643835514 },
    { "path": "/Users/you/.nuclis/models/convaiinnovations/laya/multilingual/model.safetensors.nuclis.json", "bytes": 376 },
    …
  ],
  "kept": [],
  "freed_bytes": 678203650
}
```

`kept` lists `{ "path", "used_by" }`, `used_by` naming the entry that
keeps the file. A model a running `nuclis serve` has
open keeps working after `rm`: macOS keeps an open file's mapping valid
once the file is unlinked, so the disk space is freed only when the server
closes the model (it is unloaded, or the server stops), and opening it
again fails as a missing file until it is pulled again. `rm` cannot ask the
server, so it does not refuse.
