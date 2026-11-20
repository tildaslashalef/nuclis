<!-- kind: reference -->
# Command output

How every `nuclis` command's text report reads. They all render through
one kit, `src/ui/`, whose design follows artoo's CLI, so each looks
the same. The agent's chat surface draws on its own
([terminal.md](terminal.md)), and `--json` output is never styled.

- **Header line**: nuclis's mark `◆` (`nuclis` in ASCII), the subject of
  a view about one thing (a model, a file), and its context, muted. The
  command is not echoed.
- **Detail views**: aligned key/value sections; a value longer than the
  terminal wraps under its column.
- **Tables**: upper-case column titles, numbers right-aligned; on a narrow
  terminal optional columns drop first, then text columns shrink with `…`;
  names and identifiers are never cut. Piped output has no width limit.
- **Status**: a glyph and a word in its colour: `✓ ok`, `▲ warning`,
  `✗ failed`, `– not checked`, `● running`; ASCII fallbacks.
- **Values**: one format each: counted nouns (`1 file`, `3 files`, never
  `file(s)`), decimal sizes (`16.56 GB`, the Hub's units), durations
  (`812 ms`, `42.1 s`, `5m 03s`), grouped counts (`65,536`).
- **Plans**: before work that downloads or writes in bulk, the plan
  (what, where, from → to).
- **Palette**: the agent's theme (gruvbox dark): titles and identifiers
  blue, muted gray, ok green, warning yellow, failure red, a changed value
  aqua, high impact orange. `NO_COLOR`, a dumb terminal, or a pipe turn it
  off; `NUCLIS_ASCII=1` forces the ASCII glyphs; `COLUMNS` pins the width.
- **Errors**: one line on stderr, `nuclis: <message>`, never an error's
  name: the diagnostic the failure set, or a sentence for the error. A
  usage error names the argument (`unknown option '--frob' for 'nuclis
  model pull'`) and adds `Run 'nuclis <command> --help' for usage.`
  Warnings and notes from deeper code read `nuclis: warning: …`.
- **Exit status**: 0 success, 1 failure, 2 a usage error, 130 an
  interrupt.

