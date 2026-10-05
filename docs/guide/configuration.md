# Configuring nuclis

Every setting lives in one file, `~/.nuclis/nuclis.json`, and a
command-line flag overrides it for one run. Where the file and the rest of
`~/.nuclis` live: [getting-started.md § User directories](getting-started.md#user-directories).
The schema and its defaults are `src/config.zig`.

## Configuration file

```sh
nuclis config init               # write the file: defaults, every catalogue model registered
nuclis config init --discover    # also register the models you downloaded yourself
nuclis config show               # every setting in effect, and where it came from
nuclis config set <key> <value>  # change one setting by its dotted name
```

`init` keeps a file that already exists, and ends by saying which model to
pull next. The file has one section per scope, not per command:

| Section | Holds | Read by |
| --- | --- | --- |
| `engine` | the model and its session: backend, context, cache precision | every model command |
| `generation` | how tokens are produced: output budget, reasoning effort, speculative decoding, sampling | `generate`, `agent` |
| `agent` | the chat surface's own settings | `agent` |
| `decide` | the decision model | `decide`, `serve` |
| `serve` | where the API listens | `serve` |
| `cache` | the agent's saved model states | `agent` |
| `models` | your named models (the registry) | anything that takes `--model` |

`bench` reads only `engine`; its output budget, repetitions, and greedy
sampling stay on its command line so runs remain comparable.

A file as `init` writes it:

```json
{
  "schema_version": 1,
  "engine":   { "model": "qwen3.8-27b", "backend": "metal", "ctx_size": 16384,
                "kv_precision": "f16" },
  "generation": { "max_tokens": 8192, "think": "off", "speculative": false, "draft_length": 4,
                "image_max_tokens": "auto",
                "sampling": { "temperature": null, "top_k": null, "top_p": null, "min_p": null,
                              "presence_penalty": null, "repetition_penalty": null } },
  "agent":    { "think": "low", "fold_thinking": true, "theme": "gruvbox-dark", "instructions": "auto",
                "thinking_budget": 1024 },
  "decide":   { "model": "laya" },
  "serve":    { "host": "127.0.0.1", "port": 8000, "log": true },
  "cache":    { "memory_bytes": 4294967296, "disk_bytes": 8589934592 },
  "models":   { "qwen3.8-27b": { "kind": null, "path": null,
                                 "repo": "unsloth/Qwen3.8-27B-GGUF", "file": "Qwen3.8-27B-UD-Q4_K_M.gguf",
                                 "revision": "4ca720788d1e01f1bff70c033e0d0028fd02e502",
                                 "mmproj": "mmproj-BF16.gguf", "mtp": "MTP/mtp-Qwen3.8-27B-Q4_0.gguf",
                                 "profile": null, "ctx_size": null,
                                 "generation": { "max_tokens": null, "think": null, "speculative": false, "draft_length": 4, "image_max_tokens": null, "sampling": { "…": null } },
                                 "agent": { "think": null, "fold_thinking": null } } }
}
```

## What overrides what

From weakest to strongest:

1. the built-in defaults;
2. the model's sampling profile (the official settings for its reasoning mode);
3. the file's sections;
4. the registry entry of the model in use;
5. command-line flags.

There are no per-key environment variables and no per-project files;
`NUCLIS_HOME` only moves the root.

## The settings

### `engine`

| Key | Default | Meaning |
| --- | --- | --- |
| `model` | `qwen3.8-27b` | a registry entry, a catalogue name, or a path (under `~/.nuclis/models` unless absolute), tried in that order; flag `--model` |
| `backend` | `metal` (`cpu` in a build without Metal) | where the model runs |
| `ctx_size` | `16384` | the context window in tokens, 1..32,768 |
| `kv_precision` | `f16` | the attention cache on the GPU: `f16` halves its memory and the bytes read per token; `f32` otherwise. The CPU always keeps F32. Flag `--kv` |

### `generation`

| Key | Default | Meaning |
| --- | --- | --- |
| `max_tokens` | `8192` | the most tokens one completion (one agent step) produces, 1..16,384 |
| `think` | `off` | reasoning effort: `off`, `low`, `medium`, `high`, `xhigh` |
| `speculative` | `false` | speculative decoding with the model's drafter; flag `--speculative` |
| `draft_length` | `4` | drafted tokens per step |
| `image_max_tokens` | `auto` | the most tokens one image becomes, or a positive integer |
| `sampling.*` | `null` | overrides of the profile: `temperature`, `top_k`, `top_p`, `min_p`, `presence_penalty`, `repetition_penalty` |

`null` sampling keys take the official profile for the reasoning mode
([engine/sampling.md](../engine/sampling.md#sampling-profiles-and-the-selection-chain-2026-09-08)),
so the file never freezes a model's recommended settings.

Speculation is decided per model: each catalogue entry `init` writes
carries its family's measured verdict in its own `generation.speculative`
and `draft_length` (on where it pays, off where it does not;
[benchmarks/README.md](../benchmarks/README.md#speculation-pairs)). The
global `generation.speculative` applies to models without an entry.

### `agent`

| Key | Default | Meaning |
| --- | --- | --- |
| `think` | `low` | the agent's reasoning effort |
| `fold_thinking` | `true` | start with reasoning folded (Tab unfolds) |
| `theme` | `gruvbox-dark` | the terminal palette |
| `instructions` | `auto` | the project file the agent reads: `auto` (`AGENTS.md`, then `CLAUDE.md`), `off`, or a path in the workspace |
| `thinking_budget` | `1024` | the most reasoning tokens a step spends at `low` before the engine closes the reasoning; `0` for no cap |

### `decide` and `serve`

| Key | Default | Meaning |
| --- | --- | --- |
| `decide.model` | `laya` | the decision model `nuclis decide` opens, and `nuclis serve` opens at start: a registry entry of kind `decision`, a decision catalogue name, or a directory. A text model is refused |
| `serve.host` | `127.0.0.1` | an IP literal or `localhost`; beyond loopback the API is reachable from the network with no authentication. Flag `--host` |
| `serve.port` | `8000` | flag `--port` |
| `serve.log` | `true` | a line per request on stdout; `--quiet` turns it off |
| `serve.timeout` | `300` | seconds a request may wait for the GPU before `529 timeout` (a clef-flash request can hold it for minutes); flag `--timeout` |

### `cache`

| Key | Default | Meaning |
| --- | --- | --- |
| `memory_bytes` | 4 GiB | the agent's saved model states kept in memory; `0` keeps none |
| `disk_bytes` | 8 GiB | the same under `~/.nuclis/cache/prefix/`; `0` disables it |

## Naming your models: the registry

`models` maps a name to a model, so `--model <name>` and `engine.model`
can use it. Entries are optional for catalogue models (a catalogue name
works without one) and are written by `config init` (the catalogue's),
`config init --discover`, and `model pull --register`, never by `config set`.

| Field | Meaning |
| --- | --- |
| `repo` + `file` (+ `revision`) | a downloaded model in the layout `model pull` writes, `~/.nuclis/models/<repo>/<file>` |
| `path` | instead, a file anywhere (under `~/.nuclis/models` unless absolute) |
| `mmproj`, `mtp` | companion files in the same folder: the vision projector and the drafter |
| `profile` | force a prompt profile (`qwen38`, `gemma4`, …) whatever the file's template |
| `ctx_size`, `generation`, `agent` | overrides that apply only while this model is in use; `null` means the global value |
| `kind` | `"decision"` for a decision model (`nuclis decide` opens it; text commands refuse it) |

- **Names** are 1..64 printable characters, no `/`, not ending in `.gguf`.
- **A registry name wins over a catalogue name** for `--model` and
  `engine.model`; `model pull` tries the catalogue first.
- **Entries pin no digest.** Pulling by entry name takes the Hub's.
- **A forced `profile`** runs a finetune converted with another revision of
  its family's template, which nuclis would otherwise refuse
  (`UnsupportedPromptTemplate`). The agent says so at startup, and the
  prompt is rendered the pinned way, not necessarily the file's own
  ([engine/prompt-profile.md](../engine/prompt-profile.md#evidence-and-reproduction)).
  `--prompt-profile` does the same for one run.

## Registering what you already downloaded

`nuclis config init --discover [--dry-run] [--json]` walks
`~/.nuclis/models` and writes an entry for every runnable model file that
neither the catalogue nor an existing entry covers:

- **Skipped:** files an entry already locates, and companions (by sidecar
  role, or a name carrying `mmproj`, `mtp`, or `dflash`).
- **Judged** as `nuclis model inspect` judges: an adapter for the
  architecture, every tensor in the executable set, the binding. Every
  skipped file is reported with its reason.
- **Named** after the repository's last path segment, lower-cased, `-gguf`
  dropped; a taken name gains the quantization, then a counter. Never a
  catalogue name.
- **Filled:** `repo`, `file`, and `revision` from the sidecar (or `path`
  without one); companions beside the file as `mmproj` and `mtp`; the
  family's `profile` when the template matches none (a finetune); the
  catalogue's speculation verdict for the same architecture.
- `--dry-run` reports without writing; the file is created if absent and
  otherwise kept.

## Changing one setting

```sh
nuclis config set engine.model gemma-4-12b-qat
nuclis config set generation.sampling.temperature 0.7
nuclis config set models.gemma-4-12b-qat.ctx_size 32768
nuclis config set generation.sampling.top_k null     # clear an override
```

- The file's own text is edited, so your keys and their order survive.
- The result is validated before it is written; a refused value leaves the
  file as it was and names the key.
- `engine.model` must resolve to a file that exists.
- `models.<name>.<key>` needs an existing entry (create one with `model
  pull --register`).

## Seeing what is in effect

`nuclis config show [--json]` prints every key's effective value and its
source: `default`, `profile`, `file`, `model` (the registry entry), or
`flag`. Above the table it names the model, how it resolved (registry
entry, catalogue name, or path), and its profile; below, each registry
entry's own keys.

The profile named there is the catalogue's, chosen without opening the
file. A run samples with the profile of the file it opens (chosen by its
template digest), so a Gemma file reached by path still gets Gemma's
defaults. Each command's own report (`generate --json`, `bench`) shows the
flags it ran with.

## Validation

- Unknown keys are rejected with their dotted path
  (`models.<name>.imatrix` included).
- Wrong types and values name the key and the accepted form; ranges are the
  flags' (context 1..32,768, tokens 1..16,384, sampling options as the
  sampler checks them).
- An entry must locate its model one way: `path`, or `repo` + `file`.
- The file is limited to 64 KiB.
- A `schema_version` other than 1 is an error that states the migration:
  move the file aside, run `config init`, copy your settings back. A new
  key with a default does not change the version.
- A file with a `chat` section (the old name of `agent`) is refused with a
  message saying to rename it.
