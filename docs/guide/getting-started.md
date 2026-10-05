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

`nuclis generate`, `bench`, `eval`, `decide`, and `serve` cover the rest:
[eval.md](eval.md) for perplexity, [api.md](api.md) for the HTTP API.

## User directories

nuclis uses `~/.nuclis` as its user configuration/data root. `NUCLIS_HOME`,
when set, overrides it and must be an absolute path. An empty or relative
override is an error. Otherwise resolve the root from `HOME`.

```text
~/.nuclis/
  models/                  model artifacts: <owner>/<repo>/<file> plus a provenance
                           sidecar <file>.nuclis.json per verified file ([models/catalogue.md](models/catalogue.md))
  nuclis.json              engine configuration (APPS-03; `nuclis config init` writes it)
  agent/                   the agent's data root (`paths.agentPath`; TERM-01)
    history.jsonl          submitted prompts across sessions (TERM-01; append-only,
                           one JSON object per line, tail-read at startup)
    sessions/<cwd-slug>/   one append-only JSONL file per session
    exports/               `nuclis agent export` markdown
                           ([spec.md § Sessions and storage](spec.md#74-sessions-and-storage))
  cache/                   regenerable runtime data
    prefix/                the agent's saved model states, one per file: primed prefixes
                           and the last turn of each recorded session (`nuclis cache`)
                           (`cache.disk_bytes`; [engine/session.md § The agent's token cache](engine/session.md#the-agents-token-cache-agnt-19))
```

Create directories only when an operation needs to write them. Inspection/help
must not create settings or directories. Do not migrate or overwrite existing
user files implicitly. Credentials remain environment-only (`HF_TOKEN` for
gated Hub repositories; public ones need none).

## Model download

`nuclis model pull <name> [--with mmproj,mtp | --all]` fetches a catalogue
entry (`src/catalog.zig`, the only source of "supported": name, repository,
file, pinned commit, digests, companions with the unit that will load them;
`qwen3.8-27b` today) into `<root>/models/<owner>/<repo>/<file>` through the
`huggingface` package ([its README](../../huggingface/README.md): native Xet
reconstruction, SHA-256 verified, atomic publication, verified reuse of a
file already in place) and writes the sidecar beside each file; the Hub's
digest at the pinned commit must equal the catalogue's (`CatalogMismatch`
otherwise, and no sidecar). `nuclis model pull <owner/repo> [--file <name>]
[--revision <rev>] [--role main|mmproj|mtp|imatrix]` fetches any other
artifact by repository id (a GGUF, or a safetensors set with its config
and tokenizer files, [models/catalogue.md § Safetensors
artifacts](../models/catalogue.md#safetensors-artifacts)): the revision (`main` by default; a tag, branch, or
commit) is resolved once, printed as the 40-character commit, and that
commit pins the transfer and is what the sidecar records; a repository
with several artifacts and no `--file` prints the choices with sizes and fails
with `SelectionRequired`; exact names match in full, subdirectories
included (`--file MTP/mtp-Qwen3.8-27B-Q4_0.gguf` keeps the subdirectory).
Both forms take `--force` and `--json` and need the root and nothing from
`nuclis.json`. A registry entry name (`nuclis model pull gemma`, see
[§ Configuration file](configuration.md#configuration-file)) is the one form that reads the
file: it pulls the entry's `repo`/`file` at its `revision` (`main` when
unset) with the Hub's digest, `--with mmproj,mtp` or `--all` adding the
entry's companion names; an entry that names a `path` has nothing to pull
(`NotPullable`), and a name that is none of the three forms is
`UnknownModel`. A second pull of the same file hashes it, downloads nothing,
and rewrites the sidecar. A sidecar recording other content is `ExistingFileMismatch` unless
`--force` replaces file and sidecar; a differing file nuclis never verified
is the same error from the package, and `--force` replaces it too. Progress
is one updating line on stderr (bytes, rate, ETA) on a terminal, phase lines
otherwise; Ctrl-C cancels through the package's sink, which removes the
partial file (a second Ctrl-C kills the process the ordinary way and may
leave the temporary file). `nuclis model ls [--json]` prints the catalogue
with each entry's local status from its sidecar alone (`present`,
`absent`, `mismatch`, `unverified`: the file is there without a sidecar)
and its companions beneath with a "not loaded yet" note, then the other
model files (GGUF, safetensors weights) in the layout with their sidecar facts (runnable only if their
architecture has an adapter); files above `<owner>/<repo>/` are counted,
not listed; every listed file that a registry entry locates (`path`, or
`repo` and `file`) says `registered as <name>`, with the profile when the
entry forces one, and entries whose file is absent are listed last (a
`nuclis.json` that fails to load leaves the listing unannotated with one
warning line). `make model-ls` wraps it. `nuclis model pull <owner/repo>
--file <name> --register <name> [--profile <p>]` also writes the pull as a
registry entry (`repo`, `file`, the resolved commit, and the forced
profile) once every file is verified, so `--model <name>` and `config set
engine.model <name>` work from then on; a companion role fills the same
entry's `mmproj`/`mtp`, a name that locates other content is refused, a
catalogue name is refused before the transfer unless it names the
catalogue's own file (the registry resolves first, so such an entry would
shadow the catalogue; the loader rejects one however it got there), and a
registry-entry pull refuses `--register`. `--model` and `engine.model`
accept a registry entry, a catalogue name, or a path; a missing file
fails before anything opens, naming the resolved path.

`nuclis model inspect (<name> | <owner/repo> --file <name>) [--revision <rev>]
[--json]` answers "will this quantization load" before a download. It lists
the repository at the resolved commit, fetches the head of the file through
the package's `readRange` in 8 MiB windows until the GGUF directory parses
(never past the parser's 64 MiB directory bound; the Qwen directory is
11.0 MB and Gemma 4 12B's 15.8 MB, two requests and about 4 s each on
2026-09-11), prints what `inspect` prints for a local file, and ends with
a verdict: `supported` (the catalogue pins the Hub's digest for the file
and the adapter binds the directory), `runnable` (an adapter for
`general.architecture` binds it, but the file is not in the catalogue or
carries another digest), or `not runnable` naming the first offending
tensor and its encoding (a layout nuclis does not store, or one outside
the adapter's executable set), the missing adapter (`gemma4`, `clip`), or
the adapter's rejection; a catalogue companion (`mmproj-BF16.gguf`) is
reported as the companion it is. Nothing is written and no weights are
downloaded; a repository with several GGUFs and no `--file` lists them as
`pull` does.

