# Hugging Face model downloads

A Hugging Face download library with a standalone binary, `hf-downloader`,
that also renders the library's progress events.

Native **Xet is the default** for Xet-backed artifacts. The library resolves the
Hub file ID, acquires a read token, requests CAS reconstruction metadata, downloads
signed xorb ranges, and reconstructs the file itself. A Xet error does **not**
silently switch to the Hub's HTTP compatibility bridge. Ordinary HTTP range
download is used only when the Hub reports no Xet hash.

## Building

Build and test from the repository root so the only `.zig-cache` and `zig-out`
are the root's (`huggingface/` gets none of its own):

```sh
make hf-downloader        # zig-out/bin/hf-downloader
make test-hf              # offline library tests
./zig-out/bin/hf-downloader --help
```

The package is self-contained (`build.zig`, `build.zig.zon`, `src/`) and a
path dependency of the root build since MODL-02 (2026-09-11): `nuclis model pull`
and `model ls` (`src/model.zig`) are the host of the API below, and
`zig build test` at the root runs these tests too. Only the executable imports
it; the inference library never does.

## Internal API

```zig
const hf = @import("huggingface");

// init is the host application's std.process.Init.
var client = hf.Client.init(init.gpa, init.io, init.environ_map);
client.concurrency = 8; // default; supported range 1..16

var result = try client.download(.{
    .repo_id = "unsloth/Qwen3.8-27B-GGUF",
    .filename = "Qwen3.8-27B-UD-Q4_K_M.gguf", // optional
    // .revision = "main",                    // optional, default main
    // .local_dir = "/data/models",           // optional
});
defer result.deinit();

switch (result.outcome) {
    .downloaded => |files| {
        // Each file provides path, size, SHA-256, and transport/reuse status.
        for (files) |file| _ = file.path;
    },
    .selection_required => |choices| {
        // Show exact filenames/sizes in the TUI, then call download with a choice.
        for (choices) |choice| _ = choice.name;
    },
}
```

| Operation | Result and ownership |
| --- | --- |
| `client.list(request)` | Owned `Catalog`: pinned commit and GGUF filenames, sizes, SHA-256 values. Call `deinit()`. |
| `client.download(request)` | Owned `Result`: downloaded paths or selection choices. Call `deinit()`. |
| `client.localPath(request, filename)` | Where `download` would put the file; no network or filesystem access. Free with `client.allocator`. |
| `client.readRange(request, offset, length)` | Allocated bytes (1 byte to 8 MiB); free with `client.allocator`. Requires an exact filename. Does not publish a file. |

`repo_id` is required. Omitting `revision` resolves the current `main` on each
call. The returned 40-character commit pins every subsequent request in that
operation. Explicit tags, branches (including `/`), and commits are supported.

Omitting `filename` selects the sole GGUF or sole complete split-GGUF set in the
repository. Multiple choices return `selection_required` **before any model
bytes or directories are created**. Exact filenames are matched in full, including
subdirectories; there is no suffix matching or quantization guessing. Selecting
one standard `-00001-of-00002.gguf` filename downloads its entire ordered shard
set. Missing shards are errors. The catalog includes auxiliary GGUFs such as
projectors and importance matrices; choosing an artifact is not proof that the
inference engine supports it.

Directory precedence:

1. `request.local_dir`, absolute or relative to the process working directory.
2. `$NUCLIS_HOME/models` (NUCLIS_HOME must be absolute).
3. `$HOME/.nuclis/models`.

The path is `<directory>/<owner>/<repo>/<filename>`: the Hub path without the
commit, so a model registry can predict it (`client.localPath`). The pinned
commit is returned in `Result.revision`. A file already at that path is reused
only after its size and full SHA-256 match the catalog; a different file there
(another revision, a partial copy) is `error.ExistingFileMismatch`, never
overwritten. `~` is not expanded inside API strings; the CLI shell can expand
`~/models` before passing it. Repository/file traversal components are rejected.

The client borrows environment strings and the optional `HF_TOKEN` (a blank
value counts as unset); keep the environment map alive. Public repositories
work without a token, so none is required. When the Hub refuses a request the
error names the fix: `TokenRequired` (the repository needs authentication and
no token was set), `TokenRejected` (the token was sent and refused: wrong,
expired, or revoked), `AccessDenied` (the token lacks access, typically a
gated repository whose terms must be accepted on the Hub with that account).
`hf-downloader` prints those hints. Hosts may also
construct `Client` directly with an explicit thread-safe allocator, `std.Io`,
`token`, `nuclis_home`, and `home`. The library does no global environment reads
and prints nothing. The allocator must be thread safe, as required by
`std.http.Client` and concurrent downloads. `Client` owns no long-lived resource
and needs no `deinit()`.

## Progress and TUI integration

Set `client.progress` to a `Progress` sink. `ProgressEvent` includes:

- `phase`: resolving, selection_required, downloading, verifying, complete.
- `filename`, zero-based `file_index`, and `file_count`.
- `file_completed`/`file_total` and overall `completed_bytes`/`total_bytes`.
- `reused` for a verified existing local file.
- `fraction()`: optional 0–1 overall completion; null before totals are known.

```zig
const DownloadStatus = struct {
    fraction: ?f64 = null,
    bytes: u64 = 0,
    total: u64 = 0,

    fn update(context: ?*anyopaque, event: hf.ProgressEvent) error{Canceled}!void {
        const self: *@This() = @ptrCast(@alignCast(context.?));
        self.fraction = event.fraction();
        self.bytes = event.completed_bytes;
        self.total = event.total_bytes;
        // The TUI can timestamp events using its injected Io for rate/ETA.
        // Return error.Canceled here if the user requested cancellation.
    }
};

var status: DownloadStatus = .{};
client.progress = .{ .context = &status, .update = DownloadStatus.update };
```

Callbacks are synchronous and serialized on the task calling `download`, never
on the network worker tasks. They borrow the filename only during the callback;
copy it before storing/enqueuing it. A TUI on another task should use its own
synchronized event queue. `hf-downloader`'s `Display` (src/main.zig) is a
complete example: one line per phase, and during a download a line rewritten
in place at most every 200 ms with percentage, bytes, rate, and ETA.

Byte events are whole decoded chunks (at most 128 KiB) delivered in file order
as soon as they are reconstructed, so they arrive continuously; a sink that
renders should throttle itself. A fraction of 1 still requires the `complete`
phase: checksum verification and file publication can fail. Metadata and full
verification of an existing local file have phase events but no incremental
byte events.

Returning `error.Canceled` stops at the callback. Hosts can cancel the task through
their injected `Io` to interrupt pending I/O sooner. All workers are joined before
their borrowed buffers and temporary file are freed. Already published shards
remain valid if cancellation or failure occurs later in a multi-file operation.

## How a Xet file is transferred

One `Session` per file holds a pooled, keep-alive `std.http.Client` shared by
every worker (connections to the Hub, the CAS, and the xorb store are reused
instead of a TLS handshake per request) and the CAS read token, fetched once
and refreshed under a mutex near expiry or after a 401/403.

The file is fetched in spans of 256 MiB of output. One reconstruction call per
span returns *terms* (runs of chunks of one xorb in file order) and, per xorb,
signed *units* (a byte range of the xorb holding a range of chunks). The CAS
cuts units on its own boundaries (about 6 MB for the projector file, whole
64 MiB xorbs for the 16 GB model), so a term usually spans part of one or two
units and neighbouring terms share them. The plan dedups units and records
each one's last user; execution fetches every unit exactly once, as 8 MiB
pieces through a sliding window of `concurrency` concurrent futures in
first-use order, each piece landing in its place in the unit's buffer;
decodes each term's chunks straight into the output in file order (skipping
chunks outside the term by their headers, without decompressing); and frees
a unit after its last term. The previous design fetched per 8 MiB output
window; every window pulled the whole units it touched and the next window
pulled them again, 1.5× the file in CDN traffic.

Measured on a 931 MB GGUF (`mmproj-BF16.gguf`, Xet, 2026-09-09, one run each,
default concurrency 8, same machine and link):

| Client | Time | Rate |
| --- | ---: | ---: |
| `hf download` 1.30.0 | 10.1 s | 92 MB/s |
| this library, per-window design | failed at 117 MB (multipart preamble), 43 MB/s when it ran | |
| this library, keep-alive pool + token cache | 21.8 s | 43 MB/s |
| this library, planned units, 64 MiB spans | 18.3 s | 52 MB/s |
| this library, planned units, 256 MiB spans | 14.0 s | 67 MB/s |
| this library, units fetched as 8 MiB pieces (final) | 13.1 s | 71 MB/s |

The 16 GB `Qwen3.8-27B-UD-Q4_K_M.gguf` (whole 64 MiB xorbs as units) took
216 s with the final design, 76 MB/s, byte-identical to a copy fetched with
`hf`; the per-window design could not reconstruct it at all (a 64 MiB xorb
with its framing exceeded its unit bound).

The remaining gap is start-up latency (five sequential round trips before the
first byte) and the drain at each span boundary, where the window empties
before the next plan is requested; overlapping the next span's plan and units
with the current span is the next step. Concurrency 12 or 16 did not beat 8.

## Implementation contracts

- `hub.zig`: validated identities, percent encoding, commit pinning, file selection.
- `http.zig`: native Zig HTTPS on a shared pool, 120-second per-request
  deadlines via `Io.Select`, owned bounded responses read to the end of the
  body into capacity reserved from Content-Length, or into a caller's buffer
  (`into`) for pieces. HTTPS only; no credentials sent to signed xorb URLs.
- `transfer.zig`: session, v2 reconstruction with v1 fallback only on 404/501,
  span planning over deduplicated units, ordered execution over 8 MiB pieces
  with resume from the last delivered byte on transient failures. Every
  request is a single range, so multipart/byteranges responses never occur.
- `xorb.zig`: independently written chunk decoder for None, LZ4 frames, and
  ByteGrouping4LZ4, including non-multiple-of-four lengths and LZ4 checksums,
  plus a header-only skip.
- `root.zig`: the sink that checks the GGUF magic, hashes in file order,
  appends to an atomic file, and reports progress; publication through
  `Io.Dir.createFileAtomic`/`File.Atomic.link`.

Limits are explicit: 4 MiB metadata, 10,000 catalog entries, 512 GiB per artifact,
65,536 terms per span, 65 MiB per unit (a 64 MiB xorb plus its framing),
128 KiB raw chunks. Concurrency defaults to **8** and is configurable from 1
to 16; peak memory is the buffers of the units whose pieces are in flight or
being decoded, about two or three units (up to 64 MiB each) plus any unit held
for a later term, independent of file size.

Transient HTTP 429/500/502/503/504 and timeouts are retried per unit with
bounded backoff; an expired token or signed URL (401/403) re-plans the rest of
the span with fresh authorization; three attempts in a row before giving up.
Bytes are never delivered twice. There is no persistent chunk cache,
cross-span prefetch, adaptive concurrency, or persistent interrupted-download
resume yet.

Full SHA-256 and GGUF magic are checked before publication. Existing paths are
reused only after size and full SHA-256 verification; conflicts are errors.
Publication refuses to overwrite a concurrently created destination. Each shard
is published independently; the whole shard set is not a filesystem transaction.

Partial `readRange` data is structurally checked over authenticated HTTPS but
does not constitute a full-file SHA-256 or Xet Merkle proof. No upload/chunking
pipeline, vision/model support, or inference behavior is implied.

## Running the binary

```sh
# Latest main, list choices without fetching model data:
./zig-out/bin/hf-downloader unsloth/Qwen3.8-27B-GGUF --list

# Select exact quantization; defaults to 8 units in flight and the models directory:
./zig-out/bin/hf-downloader unsloth/Qwen3.8-27B-GGUF \
  --file Qwen3.8-27B-UD-Q4_K_M.gguf

# Override directory and concurrency, no progress:
./zig-out/bin/hf-downloader unsloth/Qwen3.8-27B-GGUF \
  --file Qwen3.8-27B-UD-Q4_K_M.gguf --local-dir /tmp/my-models --concurrency 4 --quiet

# Cheap native Xet range inspection; prints SHA-256 and GGUF magic:
./zig-out/bin/hf-downloader unsloth/Qwen3.8-27B-GGUF \
  --file Qwen3.8-27B-UD-Q4_K_M.gguf --range 0 1048576
```

Progress goes to stderr, results to stdout (revision, then one line per file:
path, size, transport `xet` / `http` / `verified_local`). Interrupting the
standalone binary with a signal leaves the temporary file of the current
download behind (the atomic file is cleaned up only on cancellation through
the API); `nuclis model pull` turns Ctrl-C into that cancellation from its
progress sink, so it leaves no partial file.

## Sources and provenance

- [Hugging Face Xet specification](https://huggingface.co/docs/xet/index),
  [download protocol](https://huggingface.co/docs/xet/download-protocol),
  [CAS API](https://huggingface.co/docs/xet/api),
  [authentication](https://huggingface.co/docs/xet/auth),
  [file IDs](https://huggingface.co/docs/xet/file-id), and
  [xorb format](https://huggingface.co/docs/xet/xorb).
- [LZ4 frame format](https://github.com/lz4/lz4/blob/dev/doc/lz4_Frame_format.md)
  and [block format](https://github.com/lz4/lz4/blob/dev/doc/lz4_Block_format.md).
  The frame magic, flag bits, and Xet header field values come from these wire
  contracts; there are no imported implementation tables or third-party assets.
