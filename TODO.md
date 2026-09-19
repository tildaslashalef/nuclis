# TODO — active plan

This file is the only progress tracker. It holds **unfinished work only**:
where we are, what is next, and the design of each remaining unit. When a
unit closes, its outcome moves to
[docs/engineering-log.md](docs/engineering-log.md) and
its section is deleted here; when the last unit closes, this file is emptied
back to this header. Requirements live in [docs/spec.md](docs/spec.md); the
engine map in [docs/architecture.md](docs/architecture.md); how to build,
test, and measure in [docs/development.md](docs/development.md).

Session protocol (also in [AGENTS.md](AGENTS.md)): read this file first. If
it lists work, summarize *Where we are* and ask the user how to continue. If
it is empty, ask what to work on and write the agreed plan here.


## Where we are

MODL-13 closed on 2026-09-19: the Muse Glimmer profile, the channel-grammar
decoder, the `high` effort, the catalogue pin, and the acceptance record
([docs/reference/bench.md](docs/reference/bench.md)). AGNT-10's code and
fixtures are in the tree and pass; its live `--print --json` turn ran
(`write_file` then `read_file`, the file on disk as asked), so it closes
next with its log entry.

Order: AGNT-10.

| Unit | Title | Sessions |
| --- | --- | --- |
| AGNT-10 | Muse Glimmer ATEM tool calling: rendering, decoding, fixtures | 1 |
