#!/bin/sh
# Print one release's section of CHANGELOG.md, and the footer every release
# carries. The changelog's sections are `## [vX.Y.Z] - <date>`
# (docs/development.md § Versioning); this only slices it, so the notes and
# the changelog have one source and cannot drift.
#
# Usage: release-notes.sh <tag>
set -eu

tag="${1:?usage: release-notes.sh <tag>}"
here="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
root="$(dirname "$here")"

section="$(awk -v tag="$tag" '
  # The heading of the wanted release starts the section...
  $0 ~ "^## \\[" tag "\\]" { inside = 1; next }
  # ...and the next release heading ends it.
  inside && /^## \[/ { exit }
  inside { print }
' "$root/CHANGELOG.md")"

if [ -z "$section" ]; then
  echo "release-notes: $tag has no section in CHANGELOG.md" >&2
  exit 1
fi

printf '%s\n' "$section"
cat "$here/release-footer.md"
