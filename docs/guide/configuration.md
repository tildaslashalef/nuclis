# Configuring nuclis

Every setting lives in one file, `~/.nuclis/nuclis.json`; a flag overrides
it for one run. Where the file and the rest of `~/.nuclis` live:
[getting-started.md § User directories](getting-started.md#user-directories).

## Configuration file

`nuclis.json` is one sectioned document (`engine`, `generation`, `agent`,
`decide`, and the `models` registry) with a `schema_version`. The sections name a
*scope*, not a command: `engine` (the artifact and its session) and
`generation` (how tokens are produced: budget, effort, speculative decoding,
sampling) are shared by `generate` and `agent`; `agent` holds only the chat
surface's own settings (`think`, `fold_thinking`, `theme`, `instructions`,
`thinking_budget`, and was named
`chat` until 2026-09-11, see
[spec.md § Configuration](../spec.md#58-configuration)); `bench` reads `engine` plus
its own flags. The section was named `generate` until 2026-09-20, when the
rename made the scope explicit;
`src/config.zig` is its schema and the built-in defaults. `nuclis config
init` writes the defaults with every catalogue model as a registry entry
(one today), so the file shows the entry shape with the catalogue's facts
(the entries are optional: a catalogue name resolves without one), then
prints the effective engine keys, each catalogue model's local status,
and the `nuclis model pull <name>` to run next (the example below).

`nuclis config init --discover [--dry-run] [--json]` registers what the
catalogue does not name: it walks `<root>/models` as `model ls` does,
skips the files a registry entry already locates and the companions
(sidecar role, or a name carrying `mmproj`, `mtp`, or `dflash`), reads each
remaining file's GGUF directory and judges it as `model inspect` does (an
adapter for its architecture, every tensor in the executable set, the
binding), and writes one entry per runnable file: the name is the
repository's last path segment lower-cased (`-gguf` dropped; a taken name
gains the quantization suffix, then a counter; never a catalogue name),
the entry is `repo` + `file` + `revision` from the sidecar (or `path` when
there is none), companions beside the file fill `mmproj` and `mtp`,
`profile` is forced to the family's when the template digest matches no
profile (the finetune case, otherwise left to the digest), and
`generation.speculative` / `draft_length` take the catalogue's verdict for
the same architecture (off when the family drafts from a companion that
is absent). Every skipped file is reported with its reason, a header that
fails to parse included; the file is created first when absent and kept
when present, and `--dry-run` prints the report without writing
(APPS-15). Discovered on 2026-09-21: the HauhauCS Gemma 4 12B finetune
with its projector and a forced `gemma4` profile, and the Bonsai 2 PQ2_0
bring-up file. The example:

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

Each entry's `generation.speculative` / `generation.draft_length` is the
family's measured verdict (`src/catalog.zig`; [benchmarks § Definitions](../benchmarks/README.md#definitions)),
so a fresh file already turns speculation on for the family whose record
pays and off for the rest; a user's global `generation.speculative` still
applies to models with no entry, and `--speculative` overrides either.

- Precedence: built-in defaults < the model's sampling profile < the
  file's global sections < the registry entry the model names < command-line
  flags. `NUCLIS_HOME` only moves the root; there are no per-key environment
  overrides and no per-project files. `nuclis config show [--json]` prints
  the effective value of every key with its source (`default`, `profile`,
  `file`, `model`, `flag`), one line above the table naming the model, how
  it resolved (registry entry, catalogue name, path), its profile, and that
  a `null` sampling key takes the profile's value for the configured
  `generation.think`; then each registry entry's stated keys. The profile
  named there is the catalogue entry's (the first profile for a bare path
  or an unknown registry name), chosen without opening the file; a run
  samples with the opened file's own profile, selected by its template
  digest, so a Gemma file reached through a path still gets Gemma's
  defaults (MODL-07). An entry's `profile` (`qwen38`, `gemma4`) or the
  `--prompt-profile` flag forces that profile on the file whatever its
  template digest — the way to run a finetune converted with another
  revision of the template, which the engine would otherwise refuse as
  `UnsupportedPromptTemplate`; the agent prints a notice at startup, and
  the rendering is the pinned protocol's, not necessarily the file's own
  ([prompt-profile.md § Evidence](../engine/prompt-profile.md#evidence-and-reproduction)).
  `nuclis config set <key> <value>` changes one key by its dotted name
  (`engine.model hauhau`, `generation.sampling.temperature 0.7`,
  `models.<name>.profile gemma4`; `null` clears an override): the file's
  own text is edited so stated keys and their order survive, the result
  goes through the same loader before it is written (a refused value
  leaves the file untouched and names the key), `engine.model` must
  resolve to a file that exists, and a missing file is created as `init`
  writes it. Entries are created by `model pull --register`, never by
  `set` (`models.<name>.<key>` on an unknown name says so). The JSON form
  carries the file as loaded (`config`, the registry as a map), the
  `effective` view, and the `sources` map. The flag layer is visible in each
  command's own report (`generate --json`, the `bench` report's `config`
  and settings fields).
- `engine.model` is a registry entry name, a catalogue name (`qwen3.8-27b`,
  the default), or a path, tried in that order; a path resolves under
  `<root>/models` unless absolute. `--model` takes the same forms, a path
  being as given. `engine.backend` defaults to `metal` when the build
  has it, `cpu` otherwise. `engine.kv_precision` (`f16` default, `f32`;
  flag `--kv`) is the attention cache layout on the GPU: `f16` halves the
  cache's memory and the bytes attention reads per token (KERN-07); the CPU
  reference always keeps F32, and every report (`generate --json`, the
  `bench` header and JSON) states the precision the session actually used
  beside its `session_bytes`. `bench` takes model, backend, and context
  (the entry's `ctx_size` included) from the file and keeps its output
  budget (32), repetitions, and greedy sampling on the command line so runs
  stay comparable; its report records the file it ran with.
- The registry: `models` maps a name (1..64 printable characters, no `/`,
  not ending in `.gguf`) to an entry that locates a model one way, `path`
  (relative to `<root>/models` unless absolute) or `repo` + `file` (the
  layout `model pull` writes, `<root>/models/<repo>/<file>`) with an
  optional `revision`; optional `mmproj` and `mtp` companion file names
  in the same directory (recorded for the units that will load them, the vision unit
  and the MTP unit, and used by `model pull <name> --with`); and optional
  `ctx_size`, `generation` (`max_tokens`, `think`, `speculative`, `draft_length`, `image_max_tokens`, `sampling`), and `agent`
  (`think`, `fold_thinking`) overrides that apply only while that entry is
  the model, `null` meaning the global value. Entries pin no digest (a
  pull by entry name takes the Hub's). A registry name shadows a catalogue
  name for `--model`/`engine.model`; `model pull` tries the catalogue
  first, since it needs nothing from the file. An entry with `"kind":
  "decision"` (written by `model pull --register` for a Laya layout) is a
  decision checkpoint: `repo` + `file` or `path` name its weights, whose
  directory `nuclis decide` opens; text commands refuse it by name.
- `decide.model` (default `laya`) is the checkpoint `nuclis decide` opens,
  and the one `nuclis serve` opens at start and uses for a request naming
  no model or a `jev-…` id:
  a registry entry of kind `decision`, a decision catalogue name
  ([catalogue.md § The catalogue](../models/catalogue.md#the-catalogue)),
  or a directory (under `<root>/models` unless absolute); `--model` takes
  the same forms. A text model's name is refused.
- `serve.host` (default `127.0.0.1`, an IP literal or `localhost`) and
  `serve.port` (default 8000) are where `nuclis serve` listens; `--host`
  and `--port` override them for a run. `serve.log` (default `true`)
  writes a line per request to stdout; `--quiet` turns it off for a run.
  `serve.timeout` (default 300 seconds, `--timeout` for a run) is how
  long a decision request may wait for the GPU before `529 timeout`: a
  Laya request takes milliseconds, a clef-flash one seconds to minutes,
  and a request waits for every pass ahead of it.
- Sampling entries are overrides: `null` means the official profile of the
  reasoning mode ([sampling.md](../engine/sampling.md#sampling-profiles-and-the-selection-chain-modl-01)),
  so the file never freezes a model's recommended settings. The profile is
  the adapter's (`qwen38` for the one adapter; the catalogue records it per
  entry so `config show` names it without opening the file; the adapter registry dispatches
  per architecture).
- Validation: unknown keys are rejected with their dotted path (registry
  keys as `models.<name>.<key>`, so an unknown companion such as
  `imatrix` is named), wrong types and enum values name the key and the
  accepted form, ranges are the same as for flags (context 1..32768,
  tokens 1..16384, sampling options through the sampler's rules), an entry
  must locate its model one way, the file is bounded at 64 KiB, and a
  `schema_version` other than 1 is an error that states the migration
  (move the file aside, `config init`, copy settings back). Adding a key
  with a default does not bump the version (the registry was added to
  schema 1; a file without it loads with an empty one); renaming or
  re-typing one does, except the `chat` → `agent` section rename, whose
  keys and defaults are unchanged: a file with a `chat` section is
  rejected with a message that says to rename it. Keys arrive with the features that read them
  (`engine.kv_precision` arrived with KERN-07; an `agent` section comes with
  the embedded agent).

