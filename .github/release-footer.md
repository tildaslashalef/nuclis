
---

### The attached build

macOS on Apple Silicon, release mode, with the Metal backend. It is **not
signed or notarized** — there is no Apple Developer ID for this project — so
macOS quarantines it on download. After unpacking:

```sh
shasum -a 256 -c SHA256SUMS               # optional, and worth it
tar -xzf nuclis-*-aarch64-macos.tar.gz
xattr -d com.apple.quarantine nuclis-*/nuclis
./nuclis-*/nuclis --help
```

A source archive (`nuclis-*-src.tar.gz`) is attached alongside the binary.
Building from source needs only Zig and the Command Line Tools, and takes
about a minute:

```sh
zig build -Doptimize=ReleaseSafe
```

Every number in this repository was measured on one machine — an M4 Pro with
48 GiB — and nothing has been run on other hardware. Full methodology is in
[docs/reference/bench.md](../docs/reference/bench.md).
