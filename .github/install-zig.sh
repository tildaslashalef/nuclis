#!/bin/sh
# Install the pinned Zig into a directory, verifying the tarball's SHA-256.
#
# Usage: install-zig.sh <destination>
#
# The version comes from `minimum_zig_version` in build.zig.zon and the digest
# from `.github/zig-toolchain`, so there is no version to pass and nothing to
# keep in step by hand: a Zig upgrade edits the manifest and that file, and
# this script refuses anything else. Pinning the digest rather than reading it
# from the download index at run time is the rule the rest of this project
# follows for every artifact it fetches (docs/reference/artifacts.md).
#
# POSIX shell and awk only: no jq, no Python. A workflow step should depend on
# the two tools that are on every machine, and nothing else.
set -eu

destination="${1:?usage: install-zig.sh <destination>}"
here="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
root="$(dirname "$here")"

version="$(sed -n 's/.*\.minimum_zig_version = "\([^"]*\)".*/\1/p' "$root/build.zig.zon")"
[ -n "$version" ] || { echo "install-zig: no minimum_zig_version in build.zig.zon" >&2; exit 1; }

case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) target="aarch64-macos" ;;
  Darwin-x86_64) target="x86_64-macos" ;;
  Linux-aarch64) target="aarch64-linux" ;;
  Linux-x86_64) target="x86_64-linux" ;;
  *) echo "install-zig: unsupported host $(uname -s)-$(uname -m)" >&2; exit 1 ;;
esac

expected="$(awk -v v="$version" -v t="$target" '$1 == v && $2 == t { print $3 }' "$here/zig-toolchain")"
if [ -z "$expected" ]; then
  echo "install-zig: .github/zig-toolchain has no digest for $version $target" >&2
  echo "  add one from https://ziglang.org/download/index.json" >&2
  exit 1
fi

archive="zig-${target}-${version}.tar.xz"
url="https://ziglang.org/download/${version}/${archive}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

echo "install-zig: $url"
curl --fail --silent --show-error --location "$url" -o "$work/$archive"
actual="$(shasum -a 256 "$work/$archive" | cut -d' ' -f1)"
if [ "$actual" != "$expected" ]; then
  echo "install-zig: digest mismatch for $archive" >&2
  echo "  expected $expected" >&2
  echo "  actual   $actual" >&2
  exit 1
fi

mkdir -p "$destination"
tar -xJf "$work/$archive" -C "$work"
# The tarball unpacks into zig-<target>-<version>/; its contents move up so
# `zig` sits directly in the destination and the PATH entry is stable.
cp -R "$work/zig-${target}-${version}/." "$destination/"
"$destination/zig" version
