<!-- kind: guide -->
# Development guide

Working on nuclis: the layout, the environment, the gates, the agent's
harnesses, the website, Linux, and versioning. The rest of the work has a
guide of its own: [writing code](guide/code.md) (the toolchain, the build,
tests, hostile input), [the gates in detail](guide/gates.md),
[benchmarking](benchmarks/README.md) (workloads, the record, the speed
loop), and [CI and releases](guide/ci.md). Installing and using nuclis is
in [guide/getting-started.md](guide/getting-started.md); product behavior
and acceptance criteria belong in [spec.md](spec.md); agent-specific
working instructions belong in [../AGENTS.md](../AGENTS.md).

- [Current state and layout](#current-state-and-layout)
- [Environment](#environment)
- [Gates](#gates)
- [Working on the agent](#working-on-the-agent)
- [The website](#the-website)
- [Linux](#linux)
- [Versioning](#versioning)
- [GPU counters by capture](#gpu-counters-by-capture)
- [Commits, progress, and code explanations](#commits-progress-and-code-explanations)
- [Reference implementations and third-party material](#reference-implementations-and-third-party-material)
- [Documentation conventions](#documentation-conventions)

## Current state and layout

Every command in `nuclis --help` works:

<!-- gen:cli-commands -->
| Command | What it does |
| --- | --- |
| `agent` | the interactive chat; -p <text> runs one turn |
| `generate` | one completion from a prompt |
| `bench` | repeated prefill and decode measurements |
| `tokenize` | the prompt's token ids and byte offsets, no model run |
| `eval` | perplexity of a text file, teacher-forced |
| `inspect` | an artifact's identity, dimensions, tensor encodings |
| `validate` | whether a file binds to its architecture's adapter |
| `model` | pull, list, judge, and remove Hugging Face Hub artifacts |
| `cache` | list or clear the agent's saved model states |
| `decide` | typed questions about text or JSON, answered by Laya |
| `embed` | text, images, audio into vectors, by EmbeddingGemma 2 |
| `speak, voice, transcribe` | speech by Qwen3-TTS, saved voices, Nemotron ASR |
| `serve` | the nuclis API over HTTP: decisions, models kept open |
| `config` | write, show, or set a key of ~/.nuclis/nuclis.json |
| `completion` | the shell completion script for fish, bash, or zsh |
<!-- /gen -->

The engine runs on the CPU reference
and, built with `-Dmetal=true`, on the Metal backend. The
[merged pull requests](https://github.com/tildaslashalef/nuclis/pulls?q=is%3Apr+is%3Amerged) what changed and why, and
[worklog.md](https://github.com/tildaslashalef/nuclis/blob/v0.6.0/docs/worklog.md) the work before v0.6.0.

```text
src/                 executable: CLI, model commands, config
  src/tui/           terminal surface, engine-free: screen, editor, theme,
                     markdown, keys
  src/completer.zig  the completion side of a conversation: render, continue
                     or restore the session, complete (src/completer/cache.zig)
  src/agent/         agent composition: engine, conversation, sessions, tools
  src/decision/      the decision request and response JSON, model names
  src/api/           nuclis serve: HTTP transport, router, GPU executor,
                     services (decisions/, chat/)
inference/           engine library: runtime, quantization, tokenizer,
                     sampling, backends, profiles
huggingface/         Hub download library (Xet) and its standalone binary;
                     imported by src/ only
tools/gates/         the gate runner, nuclis-gates; imports no package
scripts/             gate/, bench/, repo/, app/, fixtures/, lib/
docs/                the hub README.md: guide/, models/, engine/, app/, benchmarks/
site/                the engine.nuclis.dev website: static HTML, CSS, JS, no build
build.zig            build
build.zig.zon        manifest; single source of the version
```

The executable composes the `inference` and `huggingface` modules through
the root build. Keep kernels with the inference library that owns them.

## Environment

Facts every unit depends on.

- Zig 0.17.1 (on the author's machine under `~/.local/opt/zig/stable`, managed
  by `zigup`; any install of that version works); consult its installed std source for API
  details. Apple M4 Pro, 48 GiB, macOS 27. Metal compiles
  shaders at runtime from the Command Line Tools; Xcode-only tools are reached
  per process (see [../AGENTS.md § Local toolchain](../AGENTS.md#local-toolchain); `make trace` holds the full `xctrace` invocation).
- Model `~/.nuclis/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf`
  (catalogue name `qwen3.8-27b`; the default `engine.model` and the
  Makefile's `MODEL` are `gemma-4-26b-a4b` since 2026-11-12), SHA-256
  `1bce504dfecd84153666928a41d8a66706f0158d3a317bc5196b72819d7dc529`,
  from [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
  at commit `ce331efc48d5722b4b8041f72793e530eae18d9b` (Unsloth's re-made
  file of 2026-11-11), pulled by `nuclis model pull qwen3.8-27b --all
  --variants` on 2026-11-12. Records before that date name the first pin,
  `322e194f…` at `4ca72078`
  ([benchmarks/record/qwen3.8.md § The re-made file](benchmarks/record/qwen3.8.md#the-re-made-file-2026-11-12)).
  Its companions `mmproj-BF16.gguf` and `MTP/mtp-Qwen3.8-27B-Q4_0.gguf`
  sit beside it: the projector for images, the MTP head for speculation
  ([benchmarks/record/catalogue.md](benchmarks/record/catalogue.md#pinned-commits-and-digests-2026-09-11),
  [benchmarks/record/gguf.md](benchmarks/record/gguf.md#companion-files-in-modelsqwen-2026-09-08)).
- Two ignored directories hold local state. `.zig-cache/` is disposable:
  Zig's build cache (`make clean-cache` drops it; it grows with every
  build), gate traces and results, the generated playground; `make
  distclean` or `rm -rf` loses nothing that does not come back by itself.
  `.reference/` is durable: pinned downloads (`eval/`, `speech/`, `ucd/`) and, on a
  machine that ran them, the former oracles' checkouts and venvs, all slow
  to set up again; nothing in the Makefile deletes it.
- nuclis is its own reference
  ([models § 4](models/README.md#4-validate-against-nuclis-itself)). The
  external oracles of the bring-up, their pins, and their dated runs are
  recorded in [llama-cpp.md](benchmarks/llama-cpp.md) and
  [the record](benchmarks/record/development.md#external-oracles-2026-10-14);
  the fixtures they pinned stay under `tests/fixtures/`
  ([provenance](../tests/fixtures/provenance.md)).
- `make help` lists all tasks. While editing: `make watch` reports
  compile errors about a quarter of a second after each save
  ([guide/code.md § Toolchain](guide/code.md#toolchain)). Per-commit: `make check` (fmt-check, the
  input-safety check, unit tests with the fuzz corpora, `test-metal`,
  `test-tiny`, the gate manifest's validation). Before a
  pull request: `make verify-auto`, the Focus lane of the gate registry
  ([§ Gates](#gates)); `make verify-cpu` only when the unit changes what
  the CPU reference computes (AGENTS.md § Validation). `make bench` when performance is claimed; a
  workload ([benchmarks § The record](benchmarks/README.md#the-record),
  `make workload NAME=<workload>`) when a record is claimed. The kernel micro-benchmarks (`bench-kernels`,
  `bench-matmul` with `ARGS=<tokens>`, `bench-matvec-split`,
  `bench-matvec-rows`, `bench-hadamard`, `bench-experts`,
  `bench-attention`, `bench-profile`, `trace`) measure and never gate;
  [benchmarks](benchmarks/README.md) says what each records. `generate
  --prompt-tokens <json>` feeds a token array untokenized, as `bench` does.

## Gates

Every model-specific numerical check is a *gate* in [`gates.json`](../gates.json),
run by `nuclis-gates` (`tools/gates`); bounds, model paths, and evidence
anchors live there and nowhere else. A gate whose model or data is not on
this machine is skipped, not failed, and a skip is not a pass; `--strict`
(every release) turns it into a failure. The gates at this commit, by tier:

<!-- gen:gates-by-tier -->
| Tier | Gates | Names |
| --- | --- | --- |
| `verify` | 52 | `qwen38-trace-f32`, `qwen38-trace-f16`, `qwen38-trace-q8_0`, `qwen38-generation-metal`, `qwen38-vision-metal`, `qwen38-speculative-metal`, `qwen38-verify-depth-512`, `qwen38-verify-depth-4k`, `qwen38-dflash2-trace-metal`, `qwen38-dflash2-speculative-metal`, `qwen38-draft-trace-metal`, `qwen38-vocabulary`, `laya-vocabulary`, `laya-cpu`, `laya-metal`, `laya-multilingual-vocabulary`, `clef-sequences`, `clef-metal`, `embeddinggemma-vectors-metal`, `embeddinggemma-google-metal`, `embeddinggemma-image-metal`, `embeddinggemma-audio-metal`, `gemma4-26b-a4b-trace-f16`, `gemma4-26b-a4b-generation-metal`, `gemma4-26b-a4b-vision-metal`, `qwen38-mtp-companion-trace-metal`, `qwen38-perplexity`, `gemma4-26b-a4b-perplexity-full`, `gemma4-e4b-trace-f32`, `gemma4-e4b-trace-f16`, `gemma4-e4b-generation-metal`, `gemma4-e4b-vision-metal`, `gemma4-e4b-audio-metal`, `gemma4-e4b-audio-asr`, `gemma4-e4b-draft-trace-metal`, `gemma4-e4b-vocabulary`, `gemma4-e4b-perplexity`, `qwen3-tts-0.6b-trace-metal`, `qwen3-tts-0.6b-clone-cpu`, `qwen3-tts-0.6b-clone-metal`, `qwen3-tts-1.7b-trace-metal`, `qwen3-tts-1.7b-design-metal`, `qwen3-tts-codec-metal`, `qwen3-tts-0.6b-roundtrip-metal`, `drama3-metal`, `drama3-roundtrip-metal`, `drama3-codec-metal`, `nemotron-asr-trace-metal`, `nemotron-asr-wer-metal`, `nemotron-asr-file-route`, `nemotron-asr-long`, `nemotron-asr-stream-route` |
| `verify-cpu` | 29 | `qwen38-generation-cpu`, `qwen38-vision-cpu`, `qwen38-speculative-cpu`, `qwen38-dflash2-trace-cpu`, `qwen38-dflash2-speculative-cpu`, `qwen38-draft-trace-cpu`, `clef-cpu`, `embeddinggemma-trace-cpu`, `embeddinggemma-vectors-cpu`, `embeddinggemma-google-cpu`, `embeddinggemma-image-cpu`, `embeddinggemma-audio-cpu`, `gemma4-26b-a4b-trace-cpu`, `gemma4-26b-a4b-vision-cpu`, `qwen38-mtp-companion-trace-cpu`, `gemma4-e4b-trace-cpu`, `gemma4-e4b-generation-cpu`, `gemma4-e4b-vision-cpu`, `gemma4-e4b-draft-trace-cpu`, `qwen3-tts-codec-cpu`, `qwen3-tts-0.6b-trace-cpu`, `qwen3-tts-0.6b-clone-trace-cpu`, `qwen3-tts-1.7b-trace-cpu`, `qwen3-tts-1.7b-design-trace-cpu`, `drama3-trace-cpu`, `drama3-codec-cpu`, `nemotron-asr-encoder-cpu`, `nemotron-asr-trace-cpu`, `nemotron-asr-wer-cpu` |
| `verify-release` | 7 | `qwen38-draft-stats`, `laya-multilingual-cpu`, `laya-multilingual-metal`, `gemma4-26b-a4b-trace-f32`, `qwen38-perplexity-full`, `gemma4-e4b-perplexity-full`, `qwen3-tts-1.7b-roundtrip-metal` |
| `verify-long` | 3 | `gemma4-e4b-perplexity-4k`, `qwen38-verify-depth-16k`, `qwen38-verify-depth-32k` |

91 gates.
<!-- /gen -->

**Three lanes.** Always: `make check`, on every commit, and `zig build
test` in CI. Focus: `make verify-auto`, before a unit's commit, when the
engine changed: it runs the model-free checks and the `verify` gates the
changed paths select, cheapest first, one model at a time, and names any
further tier the change asks for (`ARGS=--dry-run` prints the plan and
why). Release: every tier, strict, before a tag. A tier runs alone as
`make verify`, `make verify-cpu` (when a unit changes what the CPU
reference computes or brings up a family or draft source), `make
verify-long` (when it changes attention, the KV cache, or a windowed
schedule), and `make verify-release`; one gate as `make gate` with
`NAME=` a name or a glob; `make sweep` runs every gate not already
remembered, whenever the Mac is free. A pass is remembered by the
fingerprints of what it depends on, in `$NUCLIS_HOME/gates/`.

How a gate is defined, selected, remembered, and costed, and what each
one protects: [guide/gates.md](guide/gates.md).

## Working on the agent

How the agent is looked at and measured while it changes; what it does
for a user is in [guide/](guide/), how its terminal works in
[app/terminal.md](app/terminal.md).

### Looking at the agent without a person at the keyboard

`make shot ARGS='<steps>'` (`scripts/app/tui-shot.py`) starts `nuclis agent`
in a detached tmux session (`tmux -L nuclis`, 160×45 by default, the status
bar off so the pane is the whole window, `COLORTERM=truecolor` as Ghostty
sets it so the theme resolves to the same palette) and runs the steps in
order: `keys=<text>`, `enter`, `key=<tmux key>` (`C-c`, `Tab`, `Escape`),
`paste=<text>` (a bracketed paste, as a dropped file's path arrives),
`wait=<s>`, `until=<text>,<s>` (wait for the text on screen; prints how
long it took, to 0.05 s, so `enter until=generating,60` times Enter to
the first token), `capture=<name>`,
`burst=<name>,<seconds>,<hz>`, and `buffer=<name>` (tmux's paste buffer,
where an OSC 52 copy lands, to `<name>.buffer.txt`); `--env KEY=VALUE`
sets a variable for the agent (`EDITOR`, to drive Ctrl-G). A step with a
space is one quoted argument. A capture writes `<name>.txt`,
`<name>.ansi`, and `<name>.tagged.txt` (every styled run as
`[fg=#hex bg=#hex bold]…[/]`, so a colour is readable as text) under
`.zig-cache/tui/`; a burst captures at the given rate, keeps the frames that
differ from their predecessor, and reports the mean interval between them
in `<name>.burst.json` — the way an animation's cadence is measured rather
than eyeballed. `--ghostty` also opens one Ghostty window attached to the
session and photographs it to `<name>.png` on each capture (`screencapture`
needs the screen-recording permission once). `--direct` runs the agent in a
Ghostty window of its own with no tmux in between, for what only the
terminal does (kitty image placements, its cursor reports and resizes): a
pty relay in the window forwards the steps (through a FIFO) and the
window's own keys, records every byte the agent writes to
`<session>.stream`, `until=` searches that stream, and a capture writes the
PNG and `<name>.stream`; the window stays open after the agent exits so its
last screen can be photographed. The session stays up unless
`--stop` is given; `tmux -L nuclis attach -t shot` joins it. <!-- ok: a tmux command, not nuclis --> Captures are
evidence for the log, never fixtures: the goldens in `src/tui/` stay the
contract.

This is the iteration loop for every change to the surface: `make build`,
one `make shot` run that drives the feature (a `paste=` for a drop, `keys=`
for a prompt, `until=ready,<s>` to wait out a turn, `capture=` at each
state worth looking at), the `.txt` for the layout and the `.tagged.txt`
for the colours, then the fix and the same run again. A crash lands in the
capture as the panic and its trace, since tmux keeps the dead pane
(`remain-on-exit`), so a failure is read the same way as a success. To look
at a run in a real terminal, pass `--ghostty` (the harness opens the window
attached) or leave `--stop` off and attach from Ghostty. One thing the
harness cannot show is the inline image preview: it is off under tmux,
which cannot scroll a picture it does not know about, so that one is looked
at by running `./zig-out/bin/nuclis agent` in Ghostty directly.

### The agent's task list

`scripts/app/agent-eval.py` (`make agent-eval VARIANT=<name> ARGS='…'`) runs
thirteen small tasks against the playground: a tiny Python geometry package
that `scripts/app/playground.py` generates on first use under
`.zig-cache/playground/` (`make playground` builds it and prints the path)
and commits with `git init` as the baseline its own Makefile's `reset`
target returns to. Nothing of it is committed to nuclis; it holds only what the tasks and
the README's GIF (`scripts/app/agent-demo.py`, which clones it) touch, from
text and fixed seeds, so every machine gets the same baseline commit, and
a version stamp in its `.git` rebuilds it when the generator changes
(`--workspace <dir>` runs another project whose Makefile has `reset` and
`test` targets). The tasks: whole-file questions (a line count
and a maximum, the release list), edits (a rename, the repeated heading,
a planted bug, a validation with its test), a new module with tests, a
flag added to the CLI, a silent `exit 7`, a tree-wide `grep`, a three-bullet
summary, the 1 MiB file, and "what is this project about". Each task is one
`nuclis agent --print` turn with a fixed seed from the committed baseline, checked by its own
predicate (file contents, a command's output, the answer's text) and
scored on steps, tool calls, tool errors, failed edits, prompt and
generated tokens, the model's seconds, the answer's length, and four
counted habits read from the session (a regex handed to the literal
`grep`, a guessed `pytest`, a file re-read right after its own edit, a
`cd` before a command); the records land under
`.zig-cache/agent-eval/<variant>/` with their session files, and
`--compare a b …` renders the variants side by side. `--system-prompt
<file>` runs a candidate prompt without a rebuild (the instructions
section still follows it), `--instructions <file>` plants the playground's
`AGENTS.md` (`tests/fixtures/agent-eval/AGENTS.md` by default, `none` for
no file), and `--seeds 1,2` is the default because a sampled turn varies:
two seeds cannot resolve a 10 % difference in seconds, so a change is
judged on the pass marks, the habit counts, and the medians rather than
the means. This is how a change to the system prompt, a tool description,
or the loop is judged: before and after, on the same list and seeds, and
the unit's record cites the table; the first such record is the 2026-09-22
change that split the system prompt into sections, in the
[worklog archived at `v0.6.0`](https://github.com/tildaslashalef/nuclis/blob/v0.6.0/docs/worklog.md).
`scripts/app/agent-tokens.py <variant> [<variant> …]` says where a variant's
prefill went: per tool, the calls, the tokens their results added (counted
by `nuclis tokenize --raw`, the engine's own tokenizer), and each tool's
share, with the variants side by side when given several.

## The website

`site/` is engine.nuclis.dev: one page (`index.html`, `style.css`, `site.js`), a
`404.html`, and the files crawlers and Cloudflare read. There is no build
step and no dependency; what is in the folder is what is served.

| File | Holds |
| --- | --- |
| `index.html` | the page; its head carries the canonical URL, Open Graph and Twitter tags, JSON-LD, and the `#bench` block of measured figures |
| `site.js` | the agent replay, the engine strip, the drafter, the charts, the Decision Dungeons rooms, and the latest-release lookup |
| `fonts/` | Archivo and JetBrains Mono, subset WOFF2 ([THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md#presentation)); a new subset gets a new file name, since `_headers` caches the folder as immutable |
| `_headers` | Cloudflare's response headers: the CSP (no inline script or style anywhere), security headers, caching |
| `wrangler.jsonc`, `.assetsignore` | the Worker that serves the folder: its name, the 404 page, the `engine.nuclis.dev` route |
| `og.png` | the 1200×630 share card, rendered from `scripts/repo/site-og.html` |
| `robots.txt`, `sitemap.xml`, `site.webmanifest`, `llms.txt` | crawlers, installs, and language-model readers |

**Figures are copied, never typed twice.** The `#bench` block holds the
README's *Results* and *Speculative decoding* tables, speedup ratios
included (the README computes them from unrounded rates, so the page shows
them as given rather than dividing its rounded ones). `make site-check`
fails when the two differ, when the JSON-LD `softwareVersion` or the
download button's fallback is not the root `build.zig.zon`'s version, when a
reference or `#fragment` does not resolve, when anything would load from
another origin, or when an inline style or script would break the CSP.
`gates.json` runs it when `site/`, `README.md`, or `build.zig.zon` changes,
so a release or a new benchmark table that the page does not follow fails
`make verify-auto`; `make version` moves the page with the manifests, so a
version pull request passes it. The download button asks GitHub's API for the latest
release when the page loads and keeps the checked fallback when it cannot.

**Preview and look.** `make site-serve` serves the folder at
<http://localhost:8000> (without `_headers`). Check a change at 1440 and
390 px wide; any Chromium takes screenshots without a window:

```sh
chromium --headless --hide-scrollbars --window-size=1440,7800 \
  --virtual-time-budget=9000 --screenshot=.zig-cache/site.png http://localhost:8000/
```

The share card is rendered the same way, from the template beside the
scripts:

```sh
chromium --headless --hide-scrollbars --window-size=1200,630 --allow-file-access-from-files \
  --screenshot=site/og.png "file://$PWD/scripts/repo/site-og.html"
```

To test the CSP locally, put `_headers`' policy (less `frame-ancestors`)
into a `<meta http-equiv="Content-Security-Policy">` of a copy and load it
with `--enable-logging=stderr`: a violation prints as a `CONSOLE` line.

**Deploy.** A Cloudflare Worker named `nuclis` with static assets and no
code (`site/wrangler.jsonc`; `.assetsignore` keeps the config itself from
being served), connected to this repository through Workers Builds: root
directory `site`, no build command, deploy command `npx wrangler deploy`,
production branch `main`, builds only when a push touches `site/*`. The
config routes the subdomain `engine.nuclis.dev` as a custom domain (the zone is on the
same account, so the deploy writes its DNS record and Cloudflare issues the
certificate) and turns `workers.dev` and preview URLs off, so the site has
one address. The apex `nuclis.dev` is left for a separate site; this
Worker never routes it. The app was created in the dashboard (Workers & Pages →
Create → Continue with GitHub), which also mints the build token; the
watch path was narrowed with `cf builds triggers update <trigger>
--path-includes 'site/*'`, and `cf workers list` and `cf builds triggers
list --external-script-id <worker id>` show the state. Pushing `main` is
the deploy; nothing in CI deploys. `npx
wrangler deploy --dry-run` in `site/` validates the config without
uploading.

## Linux

A CPU-only target: the executable links libc on every target, the memory
budget reads `sysinfo` (capped by a cgroup v2 `memory.max`), and image and
audio input are refused with the formats the build reads (none). Cross-
compile from the Mac for the host's glibc and CPU, for example an x86_64
host with AVX-512 and glibc 2.43:

```sh
zig build -Dmetal=false -Doptimize=ReleaseSafe -Dtarget=x86_64-linux-gnu.2.43 -Dcpu=x86_64_v4
```

A plain `-Dtarget=x86_64-linux-gnu` builds for baseline x86-64 (SSE2), which
leaves the CPU kernels' vectors narrow. Copy `zig-out/bin/nuclis` over,
then `nuclis model pull <name>` there and `nuclis serve` (the backend is
the CPU when Metal is not built).

The release's Linux archive is built for the widest reach at little cost:
`-Dtarget=x86_64-linux-gnu.2.35 -Dcpu=x86_64_v3` (the Makefile's
`LINUX_TARGET` and `LINUX_CPU`; `make build-linux` and `release.yml` both
build through it), so any x86-64 with AVX2
(Intel since Haswell, AMD since Excavator) and glibc 2.35 or newer (Ubuntu
22.04, Debian 12). On a 4-vCPU Ryzen 9 9955HX VM, a 17,788-token
EmbeddingGemma 2 request to `serve` takes 49.6 s, against 44.3 s for an
`x86_64_v4` build, with the same vectors
([benchmarks/record/embeddinggemma.md § Every core on
Linux](benchmarks/record/embeddinggemma.md#every-core-on-linux-2026-10-31)). No second
archive is shipped for that 11 %: nothing would pick it for the user, and
a host with AVX-512 can build its own with the command above. Its tarballs are
written with `tar --no-xattrs`: macOS's tar otherwise records
`com.apple.provenance`, which GNU tar warns about on unpacking.

The build commands, the format check, and the Python lint are
[guide/code.md § Build and format contract](guide/code.md#build-and-format-contract).

## Versioning

The version lives in the root `build.zig.zon`: the last released version.
The manifests under `inference/` and `huggingface/` mirror it (the Zig
package format requires a `.version`; the path dependencies ignore it), and
`make version V=X.Y.Z` sets all three, and the site's download with them
([§ The website](#the-website)). The executable reports it as `nuclis --version`, fed from the root
manifest through build options, but only when built from a clean checkout
of its tag `vX.Y.Z` (or outside git, as from a release's source archive).
Any other build reports the next minor as a prerelease with the commit:
`0.11.0-dev+48e8e1e` on `main` after 0.10.0, `.dirty` appended when the
work tree differed, so an installed binary built from `main` never claims
the last release. `release.yml` builds at the tag and refuses to publish a
binary that does not report the plain version.

- **SemVer with the 0.x convention.** Before 1.0 the minor number is the
  breaking axis: a release with breaking changes or features bumps the
  minor, one with only fixes the patch.
- **Releasing** (`scripts/repo/release.py`; an agent runs it on "release"):

  | Step | Command | What it does |
  | --- | --- | --- |
  | Gather | `make release-draft` | the commits on `main` since the last tag, each with the first paragraph of its record's *What and why*; the suggested version; where the highlights go |
  | Write | (the agent) | the highlights, in `.zig-cache/highlights-vX.Y.Z.txt` |
  | Verify | `make verify`, `verify-cpu`, `verify-long`, `verify-release` | the release tiers on `main` (the pinned models, about an hour; passes already remembered do not rerun); skipped when no engine code changed |
  | Propose | `make release V=X.Y.Z TIERS='…'` | from an up-to-date `main`: the branch, `make version`, the commit, the push, and the pull request `chore(release): vX.Y.Z` with the highlights and the tiers in its description |
  | Merge | the user | |
  | Tag | `make tag V=X.Y.Z` | on `main` at `origin/main` with the manifest at `X.Y.Z`: the annotated tag with the highlights as its message, signed when git signs tags (`tag.gpgSign`), pushed; `release.yml` publishes |

  Each step refuses before changing anything when its conditions fail (a
  dirty tree, `main` behind, a version not above the last, a tag that
  exists, no highlights). **Good highlights** are two to four sentences for
  someone who uses nuclis: what changed for them first (a model, a speed, a
  command), measured numbers only, each one in the benchmark record; no
  internal names, unit jargon, or file paths. They head the release notes,
  above the list of commit subjects.
- **A tag that exists on the remote never moves.** A ruleset refuses
  updating or deleting a `v*` tag; the fix is the next number.
- The [Releases page](https://github.com/tildaslashalef/nuclis/releases)
  is the changelog. The notes of v0.5.0 and earlier stay in
  [`CHANGELOG.md` at the `v0.5.0` tag](https://github.com/tildaslashalef/nuclis/blob/v0.5.0/CHANGELOG.md);
  the file is no longer on `main` or in the release archive.
- Benchmark records cite the git revision, and the release tag once one
  exists.
- **The Zig toolchain is a separate axis.** `minimum_zig_version` pins source
  compatibility, the exact compiler is recorded in benchmark records, and Zig
  upgrades are their own unit of work.

## GPU counters by capture

`xctrace`'s GPU counter profile is unsupported on the M4 Pro, but a Metal
capture replayed in Xcode reports the counters: ALU (FP32, FP16, integer)
utilization and limiters, occupancy, caches, and the MMU limiter per
dispatch. `make capture` runs `bench --capture` with
`MTL_CAPTURE_ENABLED=1`: after the runs, one more decode step is recorded
into `.zig-cache/trace/decode.gputrace` (`CAPTURE_OUT=` to move it,
`MODEL=`, `PROMPT=`, and `ARGS=` as for `bench`; the budget's context sets
the visible cache). The document holds every buffer the step reads, the
weights included: about 15 GB for Qwen3.8-27B, so delete it when done.
Xcode 27 ships no command-line reader for it, and its MCP tools (`xcrun
mcpbridge`) include none: open it in Xcode, tick *Profile after replay*
(the first time, Xcode asks to download its Metal Toolchain component,
which shader profiling needs), replay,
and read *Performance* (counters per dispatch, limiters). Never commit a
capture.

**Counters need a short command buffer.** Xcode 27 profiles a command
buffer in full only below a GPU run-time limit; above it the profiler runs
in *lite* mode ("the Metal workload exceeds the maximum size supported for
full profiling"): the timeline and the effective GPU time, with Shaders,
Heat Map, Cost Graph, and Counters disabled, and a Counters export of a
header alone. A 21 ms micro-benchmark buffer profiled in full; a 4-row
Qwen verify at 4K (332 ms at Maximum, 2026-09-30) did not. A whole-model
capture therefore confirms GPU time and dispatch order; counters come from
kernel captures.

**Pipeline limits without a capture.** `nuclis bench --kernel-stats`
(no model; `--json` for a machine-readable list) compiles the kernels and
prints each pipeline's `maxTotalThreadsPerThreadgroup`,
`threadExecutionWidth`, and static threadgroup memory. On this family 9
GPU the thread limit stays 1,024 whatever a kernel's registers
(dynamic caching), so register pressure is read only from a capture;
the threadgroup-memory column is the cheap check
([apple-gpu.md § Registers and occupancy](engine/apple-gpu.md#registers-and-occupancy-under-dynamic-caching)).

For one kernel on one shape, capture a micro-benchmark instead: `make
bench-kernels ARGS=Q4_K CAPTURE='matvec-Q4_K-5120x17408 (ffn_down)-block'`
(also `bench-matvec-rows` and `bench-attention`) records the last warm-up
command buffer of each case whose printed label contains `CAPTURE` into
`.zig-cache/trace/kernels/` (spaces become `-`, parentheses are
dropped), replacing the previous set.

**The division of work.** The agent makes the capture and asks the user
to profile it (gauge menu: Maximum, *Profile*) and export Counters as CSV
into `.zig-cache/trace/` under any name. When the user says it is done,
the agent renames the export to `<capture name>_<YYYY-MM-DDTHHMM>_<performance
state>.csv` (the time is the file's modification time; the state is
checked against the replay's GPU time, which at Maximum matches the
benchmark's rate), for example
`matvec-Q4_K-5120x17408-ffn_down-block_2026-09-30T0947_max.csv`, reads
it, copies the numbers that matter into
[apple-gpu.md](engine/apple-gpu.md) citing the file name, and deletes
the capture (the `.gputrace`), keeping the CSV. About 1 GB each:
the benchmark's buffers are sized for its largest shape, and the capture
holds whole buffers.

## Commits, progress, and code explanations

Follow the commit, branch, and pull request conventions in
[../AGENTS.md](../AGENTS.md). A unit's tests and documentation
travel in its pull request, whose
description is the record of the work.

The project is also intended to deepen the user's understanding of Zig. Add
module-level explanations of data flow and ownership, document public interfaces,
and explain subtle invariants beside the code that enforces them. Avoid comments
that merely translate obvious statements into English. Write implementation
walkthroughs as modules become stable, grounded in the working code.

## Reference implementations and third-party material

nuclis is its own reference. llama.cpp was the oracle and the speed
baseline while the first families were brought up; that is history
([models/README.md § 4](models/README.md#4-validate-against-nuclis-itself)),
and the fixtures it made stay until their file is re-pinned. Fixtures are
now made by nuclis's CPU reference
([§ Which executor makes which fixture](models/README.md#which-executor-makes-which-fixture)),
and speed is measured against nuclis's own previous record. Another engine
is read only as the definition of a format that has no other; its code is
never copied. Kernels and decoders are written from the format contract and
proven against the CPU reference by fixtures (bit-exact where the operation
is exact, stated tolerances otherwise). Constants that define a format (codebooks,
lookup tables) and any other third-party material that must be in the tree are
listed in [../THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md), which links
license texts on the web. Do not commit license text files.

## Documentation conventions

- Keep product requirements and acceptance criteria in [spec.md](spec.md);
  the active plan lives in the maintainer's tracker and each finished
  unit's record in its pull request.
- Add a document under `docs/` when there is concrete information to record,
  in the folder whose rule it fits, and give it a line in
  [README.md](README.md), the hub.
- Label proposed behavior, implemented behavior, and measured results accurately.
- Update existing authoritative guidance instead of copying it into a new file.
- Use relative repository links. Move a document with `make docs-check
  ARGS='--move old=new'`, which rewrites every reference; `make docs-check`
  fails on any link or anchor that does not resolve.
