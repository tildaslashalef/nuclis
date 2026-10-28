//! Internal model acquisition API, independent of the inference engine and CLI.
//!
//! Client borrows its environment strings and optional token for its lifetime.
//! All I/O uses the injected std.Io; the allocator must be thread safe. Calls
//! return owned results with deinit(), never pointers into HTTP response buffers.
//! Xet is the default for Xet-backed files. A failed Xet transfer is an error,
//! not an implicit HTTP fallback. Full files are verified before publish: by
//! the Hub's SHA-256, or by the git blob id of a plain (non-LFS) file.
//!
//! `HF_TOKEN` is optional: public repositories download anonymously. When the
//! Hub refuses a request the error says what to do: `TokenRequired` (no token
//! was set), `TokenRejected` (the token was refused), `AccessDenied` (gated
//! repository; accept its terms on the Hub with the token's account).
//!
//! Layout: `<models dir>/<owner>/<repo>/<filename>`, mirroring the Hub path
//! without the commit hash, so a registry can predict where a file lives. The
//! pinned revision is returned in `Result`; a file that exists at that path
//! with a different size or SHA-256 (another revision, a partial copy) is a
//! conflict, never silently replaced (`Request.replace` asks for it).
//!
//! A file is written to `<path>.partial` and renamed into place once
//! verified. A progress sink that returns `error.Paused` keeps the partial
//! and `<path>.partial.json` (repository, commit, digest, size, bytes), and
//! the next download of the same file and digest continues from there.
const std = @import("std");
const Io = std.Io;
const hub = @import("hub.zig");
const transfer = @import("transfer.zig");
pub const Request = hub.Request;
pub const RemoteFile = hub.File;
pub const Format = hub.Format;
pub const format = hub.format;
pub const Catalog = hub.Catalog;
pub const max_range_bytes = transfer.window_size;
/// Which files a request would download from a catalog (null: several
/// choices), so a host can check destinations before calling `download`.
pub const select = hub.select;
pub const owners = hub.owners;

pub const LocalFile = struct {
    path: []const u8,
    size: u64,
    /// Computed from the published bytes, whichever digest verified them.
    sha256: [32]u8,
    /// The Hub file this is, digests included.
    remote: RemoteFile,
    transport: enum { xet, http, verified_local },
    /// Bytes continued from a kept partial download rather than transferred.
    resumed_bytes: u64 = 0,
};

pub const Result = struct {
    /// Every string/slice in outcome belongs to this result.
    arena: std.heap.ArenaAllocator,
    revision: [40]u8,
    outcome: union(enum) {
        downloaded: []const LocalFile,
        selection_required: []const RemoteFile,
    },
    pub fn deinit(r: *Result) void {
        r.arena.deinit();
        r.* = undefined;
    }
};

pub const Progress = @import("progress.zig").Sink;
pub const ProgressEvent = @import("progress.zig").Event;

pub const Client = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    token: ?[]const u8 = null,
    nuclis_home: ?[]const u8 = null,
    home: ?[]const u8 = null,
    progress: ?Progress = null,
    /// Signed xorb ranges in flight at once (1..16; typically ~8 MB each, at
    /// most 64 MiB). Lower this to bound peak memory.
    concurrency: u8 = 8,

    /// Environment is supplied by the host, never read globally inside the library.
    pub fn init(a: std.mem.Allocator, io: std.Io, env: *const std.process.Environ.Map) Client {
        return .{ .allocator = a, .io = io, .token = hub.tokenFromEnvironment(env.get("HF_TOKEN")), .nuclis_home = env.get("NUCLIS_HOME"), .home = env.get("HOME") };
    }

    pub fn list(c: Client, request: Request) !Catalog {
        return hub.catalog(c.allocator, c.io, c.token, request);
    }

    /// Where `download` would place a file of `repo_id`: `<models dir>/<repo_id>/<filename>`.
    /// Caller owns the result. Does not touch the network or the filesystem.
    pub fn localPath(c: Client, request: Request, filename: []const u8) ![]const u8 {
        try hub.validate(request);
        if (!hub.validPath(filename)) return error.InvalidFilename;
        const base = try hub.directory(c.allocator, request.local_dir, c.nuclis_home, c.home);
        defer c.allocator.free(base);
        return std.fs.path.join(c.allocator, &.{ base, request.repo_id, filename });
    }

    /// A repo-only request with several artifacts returns selection_required
    /// without creating directories or downloading model bytes. An exact split
    /// filename selects its whole shard set, a safetensors file its support
    /// files too; `request.exact` takes the one named file. Existing files are
    /// reused only after full verification; conflicting files are never
    /// overwritten unless `request.replace`, which downloads every file again.
    pub fn download(c: Client, request: Request) !Result {
        if (c.concurrency == 0 or c.concurrency > max_concurrency) return error.InvalidConcurrency;
        try c.report(.{ .phase = .resolving });
        var catalog = try c.list(request);
        defer catalog.deinit();
        var arena = std.heap.ArenaAllocator.init(c.allocator);
        errdefer arena.deinit();
        const a = arena.allocator();
        const selected = if (request.exact) try exactFile(a, &catalog, request.filename.?) else try hub.select(a, &catalog, request.filename);
        if (selected == null) {
            try c.report(.{ .phase = .selection_required });
            const choices = try a.dupe(RemoteFile, catalog.files);
            for (choices) |*f| f.name = try a.dupe(u8, f.name);
            return .{ .arena = arena, .revision = catalog.revision, .outcome = .{ .selection_required = choices } };
        }
        const base = try hub.directory(a, request.local_dir, c.nuclis_home, c.home);
        const local = try a.alloc(LocalFile, selected.?.len);
        var total: u64 = 0;
        for (selected.?) |file| total = std.math.add(u64, total, file.size) catch return error.ModelTooLarge;
        var completed: u64 = 0;
        for (selected.?, local, 0..) |file, *destination, file_index| {
            try c.io.checkCancel();
            const path = try std.fs.path.join(a, &.{ base, catalog.repo_id, file.name });
            var remote_file = file;
            remote_file.name = try a.dupe(u8, file.name);
            destination.* = .{ .path = path, .size = file.size, .sha256 = undefined, .remote = remote_file, .transport = .verified_local };
            var event: ProgressEvent = .{ .phase = .verifying, .filename = file.name, .file_index = file_index, .file_count = local.len, .file_total = file.size, .completed_bytes = completed, .total_bytes = total };
            if (!request.replace) {
                try c.report(event);
                if (try verifyExisting(c.io, path, file)) |digest| {
                    destination.sha256 = digest;
                    completed += file.size;
                    event.reused = true;
                    event.file_completed = file.size;
                    event.completed_bytes = completed;
                    try c.report(event);
                    continue;
                }
            }
            // One session per file: the connection pool and the CAS read token
            // are shared by every window of this file and torn down after the
            // last worker has been joined (stream's defer runs first).
            var session = transfer.Session.init(c.allocator, c.io, c.token, catalog);
            defer session.deinit();
            const remote = try transfer.resolve(&session, file);
            defer remote.deinit(c.allocator);
            destination.transport = if (remote.xet_hash != null) .xet else .http;
            const origin: Origin = .{ .repo_id = catalog.repo_id, .revision = &catalog.revision };
            const fetched = try c.fetchFile(origin, path, file, request.replace, &event, completed, Network{ .session = &session, .remote = remote, .concurrency = c.concurrency });
            destination.sha256 = fetched.sha256;
            destination.resumed_bytes = fetched.resumed;
            completed += file.size;
        }
        try c.report(.{ .phase = .complete, .file_count = local.len, .completed_bytes = completed, .total_bytes = total });
        return .{ .arena = arena, .revision = catalog.revision, .outcome = .{ .downloaded = local } };
    }

    pub const max_concurrency = transfer.max_concurrency;

    /// Where a file comes from, recorded in a kept partial's state so only
    /// the same content continues it.
    const Origin = struct { repo_id: []const u8, revision: []const u8 };

    /// The network source of one file for `fetchFile`; tests pass a fake.
    const Network = struct {
        session: *transfer.Session,
        remote: transfer.Remote,
        concurrency: u8,
        fn fetch(n: Network, start: u64, count: u64, sink: *FileSink) !void {
            return transfer.fetch(n.session, n.remote, start, count, n.concurrency, sink);
        }
    };

    const Fetched = struct { sha256: [32]u8, resumed: u64 };

    /// Writes `file` to `<path>.partial` (held under an exclusive lock, so a
    /// second download of it fails with `DownloadInProgress`), continuing a
    /// kept partial of the same origin and digest unless `replace`, then
    /// verifies the whole file and renames it to `path`: over an existing
    /// file when `replace`, else refusing to clobber one. On `error.Paused`
    /// the partial and its state are kept; on any other failure, removed.
    fn fetchFile(c: Client, origin: Origin, path: []const u8, file: RemoteFile, replace: bool, event: *ProgressEvent, completed: u64, source: anytype) !Fetched {
        const io = c.io;
        const cwd = Io.Dir.cwd();
        const partial_path = try std.mem.concat(c.allocator, u8, &.{ path, partial_suffix });
        defer c.allocator.free(partial_path);
        const state_path = try std.mem.concat(c.allocator, u8, &.{ path, state_suffix });
        defer c.allocator.free(state_path);
        if (std.fs.path.dirname(path)) |dir| try cwd.createDirPath(io, dir);
        const partial = cwd.createFile(io, partial_path, .{ .read = true, .truncate = false, .lock = .exclusive, .lock_nonblocking = true }) catch |err| switch (err) {
            error.WouldBlock => return error.DownloadInProgress,
            else => return err,
        };
        var keep = false;
        var published = false;
        defer {
            // Deleted before close, while the lock still excludes another download.
            if (!published and !keep) cwd.deleteFile(io, partial_path) catch {};
            partial.close(io);
        }
        const start: u64 = if (replace) 0 else c.resumeOffset(origin, state_path, partial, file);
        // The state describes a paused download only; a pause writes it again.
        try deleteIfPresent(io, state_path);
        var verifier: Verifier = .init(file);
        var sink: FileSink = .{ .client = c, .file = partial, .verifier = &verifier, .container = hub.format(file.name), .size = file.size, .event = event, .completed = completed };
        if (start == 0) try partial.setLength(io, 0) else try sink.prime(start);
        event.phase = .downloading;
        event.resumed_bytes = start;
        event.file_completed = start;
        event.completed_bytes = completed + start;
        fetchRest(c, event.*, source, &sink) catch |err| {
            if (err == error.Paused and sink.position > 0) keep = c.savePartial(origin, state_path, partial, file, sink.position);
            return err;
        };
        event.phase = .verifying;
        try c.report(event.*);
        const digest = verifier.finish(file) catch return if (start > 0) error.ResumedChecksumMismatch else error.ChecksumMismatch;
        try partial.sync(io);
        if (replace) try cwd.rename(partial_path, cwd, path, io) else try cwd.renamePreserve(partial_path, cwd, path, io);
        published = true;
        return .{ .sha256 = digest, .resumed = start };
    }

    fn fetchRest(c: Client, first: ProgressEvent, source: anytype, sink: *FileSink) !void {
        try c.report(first);
        try source.fetch(sink.position, sink.size - sink.position, sink);
    }

    /// The bytes of a kept partial download this download may continue from:
    /// its state must name the same origin, digest, and size, and the file
    /// must hold at least the bytes it records (a longer tail is cut). 0
    /// (start over) when there is none or anything disagrees.
    fn resumeOffset(c: Client, origin: Origin, state_path: []const u8, partial: Io.File, file: RemoteFile) u64 {
        const text = Io.Dir.cwd().readFileAlloc(c.io, state_path, c.allocator, .limited(4096)) catch return 0;
        defer c.allocator.free(text);
        const parsed = std.json.parseFromSlice(PartialState, c.allocator, text, .{}) catch return 0;
        defer parsed.deinit();
        const have = parsed.value;
        var digests: PartialState.Digests = undefined;
        const want = PartialState.of(origin, file, have.bytes, &digests);
        if (!std.mem.eql(u8, have.repo_id, want.repo_id) or !std.mem.eql(u8, have.revision, want.revision) or have.size != want.size) return 0;
        if (!optionalEql(have.sha256, want.sha256) or !optionalEql(have.git_blob, want.git_blob)) return 0;
        if (have.bytes == 0 or have.bytes > file.size) return 0;
        const length = partial.length(c.io) catch return 0;
        if (length < have.bytes) return 0;
        partial.setLength(c.io, have.bytes) catch return 0;
        return have.bytes;
    }

    /// Syncs the partial and records its state; false when either fails (the
    /// caller then removes the partial rather than keep one it cannot trust).
    fn savePartial(c: Client, origin: Origin, state_path: []const u8, partial: Io.File, file: RemoteFile, bytes: u64) bool {
        partial.sync(c.io) catch return false;
        var digests: PartialState.Digests = undefined;
        var buffer: [1024]u8 = undefined;
        var w: std.Io.Writer = .fixed(&buffer);
        std.json.Stringify.value(PartialState.of(origin, file, bytes, &digests), .{}, &w) catch return false;
        Io.Dir.cwd().writeFile(c.io, .{ .sub_path = state_path, .data = w.buffered() }) catch return false;
        return true;
    }

    /// Receives the file's bytes in order from `transfer.fetch` (whole chunks
    /// of at most 128 KiB): checks the container's first bytes, hashes,
    /// writes them at their offset in the partial file, and reports
    /// progress. Runs on the calling task only.
    const FileSink = struct {
        client: Client,
        file: Io.File,
        verifier: *Verifier,
        container: ?Format,
        size: u64,
        event: *ProgressEvent,
        completed: u64,
        position: u64 = 0,
        head: [9]u8 = undefined,
        pub fn write(sink: *FileSink, bytes: []const u8) !void {
            try sink.take(bytes);
            try sink.file.writePositionalAll(sink.client.io, bytes, sink.position);
            sink.position += bytes.len;
            sink.event.file_completed = sink.position;
            sink.event.completed_bytes = sink.completed + sink.position;
            try sink.client.report(sink.event.*);
        }
        /// Checks and hashes the file's next bytes without writing them.
        fn take(sink: *FileSink, bytes: []const u8) !void {
            if (sink.position < sink.head.len) {
                const n = @min(bytes.len, sink.head.len - @as(usize, @intCast(sink.position)));
                @memcpy(sink.head[@intCast(sink.position)..][0..n], bytes[0..n]);
                try checkContainer(sink.container, sink.head[0 .. @as(usize, @intCast(sink.position)) + n], sink.size);
            }
            sink.verifier.update(bytes);
        }
        /// Takes the first `bytes` already in the file (a kept partial).
        fn prime(sink: *FileSink, bytes: u64) !void {
            var chunk: [64 * 1024]u8 = undefined;
            while (sink.position < bytes) {
                try sink.client.io.checkCancel();
                const want: usize = @intCast(@min(chunk.len, bytes - sink.position));
                const n = try sink.file.readPositionalAll(sink.client.io, chunk[0..want], sink.position);
                if (n != want) return error.ResumedChecksumMismatch;
                // A partial whose head is not the container is damaged.
                sink.take(chunk[0..n]) catch return error.ResumedChecksumMismatch;
                sink.position += n;
            }
        }
    };

    fn report(c: Client, event: ProgressEvent) Progress.Error!void {
        if (c.progress) |sink| try sink.report(event);
    }

    /// Inspect up to 8 MiB at an arbitrary offset without publishing a model.
    /// Caller owns the returned bytes. Explicit filename is required here.
    pub fn readRange(c: Client, request: Request, start: u64, count: usize) ![]u8 {
        const filename = request.filename orelse return error.FilenameRequired;
        var catalog = try c.list(request);
        defer catalog.deinit();
        for (catalog.files) |file| if (std.mem.eql(u8, file.name, filename)) {
            var session = transfer.Session.init(c.allocator, c.io, c.token, catalog);
            defer session.deinit();
            const remote = try transfer.resolve(&session, file);
            defer remote.deinit(c.allocator);
            if (count == 0 or count > max_range_bytes or start >= file.size or count > file.size - start) return error.InvalidRange;
            var out: std.ArrayList(u8) = .empty;
            errdefer out.deinit(c.allocator);
            try out.ensureTotalCapacityPrecise(c.allocator, count);
            var sink: Collect = .{ .allocator = c.allocator, .list = &out };
            try transfer.fetch(&session, remote, start, count, c.concurrency, &sink);
            return out.toOwnedSlice(c.allocator);
        };
        return error.FileNotFound;
    }
};

const Collect = struct {
    allocator: std.mem.Allocator,
    list: *std.ArrayList(u8),
    pub fn write(c: *Collect, bytes: []const u8) !void {
        try c.list.appendSlice(c.allocator, bytes);
    }
};

/// The one file `exact` names, weights or support; null never (a missing
/// name is `FileNotFound`).
fn exactFile(a: std.mem.Allocator, catalog: *const Catalog, name: []const u8) !?[]const RemoteFile {
    for ([_][]const RemoteFile{ catalog.files, catalog.support }) |list| for (list) |f| {
        if (std.mem.eql(u8, f.name, name)) return try a.dupe(RemoteFile, &.{f});
    };
    return error.FileNotFound;
}

/// Rejects a download whose first bytes are not the container its name
/// says, before the rest arrives: the GGUF magic, or a safetensors header
/// length that fits the file followed by the JSON object's `{`. `head`
/// grows across calls; checks run once enough of it is present.
fn checkContainer(container: ?Format, head: []const u8, size: u64) !void {
    switch (container orelse return) {
        .gguf => {
            const n = @min(head.len, 4);
            if (!std.mem.eql(u8, head[0..n], "GGUF"[0..n])) return error.InvalidGguf;
        },
        .safetensors => {
            if (head.len < 8) return;
            const header = std.mem.readInt(u64, head[0..8], .little);
            if (header < 2 or header > size - 8) return error.InvalidSafetensors;
            if (head.len > 8 and head[8] != '{') return error.InvalidSafetensors;
        },
    }
}

/// SHA-256 always (it is what the sidecar records); the git blob SHA-1 too
/// when the Hub has no SHA-256 for the file.
const Verifier = struct {
    sha256: std.crypto.hash.sha2.Sha256 = .init(.{}),
    git: ?std.crypto.hash.Sha1 = null,

    fn init(file: RemoteFile) Verifier {
        var v: Verifier = .{};
        if (file.sha256 == null) {
            v.git = .init(.{});
            var buffer: [32]u8 = undefined;
            v.git.?.update(std.fmt.bufPrint(&buffer, "blob {d}\x00", .{file.size}) catch unreachable);
        }
        return v;
    }
    fn update(v: *Verifier, bytes: []const u8) void {
        v.sha256.update(bytes);
        if (v.git) |*g| g.update(bytes);
    }
    /// The computed SHA-256, once the known digest matches.
    fn finish(v: *Verifier, expected: RemoteFile) error{ChecksumMismatch}![32]u8 {
        var digest: [32]u8 = undefined;
        v.sha256.final(&digest);
        if (expected.sha256) |want| {
            if (!std.mem.eql(u8, &digest, &want)) return error.ChecksumMismatch;
        } else {
            var oid: [20]u8 = undefined;
            v.git.?.final(&oid);
            if (!std.mem.eql(u8, &oid, &(expected.git_oid orelse return error.ChecksumMismatch))) return error.ChecksumMismatch;
        }
        return digest;
    }
};

pub const partial_suffix = ".partial";
pub const state_suffix = ".partial.json";

/// `<path>.partial.json`: what a kept partial download is part of, and how
/// many of its bytes were written and synced. Digests are lowercase hex.
const PartialState = struct {
    repo_id: []const u8,
    revision: []const u8,
    sha256: ?[]const u8 = null,
    git_blob: ?[]const u8 = null,
    size: u64,
    bytes: u64,

    const Digests = struct { sha256: [64]u8, git_blob: [40]u8 };

    /// Borrows `origin` and `digests`, which hold the hex.
    fn of(origin: Client.Origin, file: RemoteFile, bytes: u64, digests: *Digests) PartialState {
        var state: PartialState = .{ .repo_id = origin.repo_id, .revision = origin.revision, .size = file.size, .bytes = bytes };
        if (file.sha256) |d| {
            digests.sha256 = std.fmt.bytesToHex(d, .lower);
            state.sha256 = &digests.sha256;
        }
        if (file.git_oid) |d| {
            digests.git_blob = std.fmt.bytesToHex(d, .lower);
            state.git_blob = &digests.git_blob;
        }
        return state;
    }
};

fn optionalEql(a: ?[]const u8, b: ?[]const u8) bool {
    if (a == null or b == null) return a == null and b == null;
    return std.mem.eql(u8, a.?, b.?);
}

fn deleteIfPresent(io: std.Io, path: []const u8) !void {
    Io.Dir.cwd().deleteFile(io, path) catch |err| switch (err) {
        error.FileNotFound => {},
        else => return err,
    };
}

/// The file's SHA-256 when it is already at `path` and verifies; null when
/// absent; `ExistingFileMismatch` when something else is there.
fn verifyExisting(io: std.Io, path: []const u8, expected: RemoteFile) !?[32]u8 {
    const file = std.Io.Dir.cwd().openFile(io, path, .{}) catch |err| switch (err) {
        error.FileNotFound => return null,
        else => return err,
    };
    defer file.close(io);
    const stat = try file.stat(io);
    if (stat.kind != .file or stat.size != expected.size) return error.ExistingFileMismatch;
    var buffer: [64 * 1024]u8 = undefined;
    var reader = file.reader(io, &buffer);
    var verifier: Verifier = .init(expected);
    var chunk: [64 * 1024]u8 = undefined;
    var size: u64 = 0;
    while (true) {
        const n = try reader.interface.readSliceShort(&chunk);
        if (n == 0) break;
        size += n;
        if (size > expected.size) return error.ExistingFileMismatch;
        verifier.update(chunk[0..n]);
    }
    if (size != expected.size) return error.ExistingFileMismatch;
    return verifier.finish(expected) catch error.ExistingFileMismatch;
}

test {
    _ = hub;
    _ = @import("http.zig");
    _ = @import("xorb.zig");
    _ = transfer;
    _ = @import("progress.zig");
}

/// A file source for `fetchFile` tests: serves `bytes` in 3-byte chunks and
/// records where each fetch started and how many bytes it sent.
const FakeSource = struct {
    bytes: []const u8,
    start: *?u64,
    sent: *u64,
    fn fetch(f: FakeSource, start: u64, count: u64, sink: *Client.FileSink) !void {
        f.start.* = start;
        var at: usize = @intCast(start);
        const end: usize = @intCast(start + count);
        while (at < end) {
            const n = @min(3, end - at);
            try sink.write(f.bytes[at..][0..n]);
            f.sent.* += n;
            at += n;
        }
    }
};

/// Stops a download with `stop` once `after` bytes of the file are written.
const Stopper = struct {
    after: u64 = std.math.maxInt(u64),
    stop: Progress.Error = error.Paused,
    fn update(context: ?*anyopaque, event: ProgressEvent) Progress.Error!void {
        const s: *Stopper = @ptrCast(@alignCast(context.?));
        if (event.phase == .downloading and event.file_completed >= s.after) return s.stop;
    }
};

const FetchTest = struct {
    tmp: std.testing.TmpDir,
    dir: [:0]const u8,
    path: []const u8,
    stopper: Stopper = .{},
    const content = "GGUF-0123456789abcdefghijklmnopqrstuvwxyz";
    const origin: Client.Origin = .{ .repo_id = "a/b", .revision = "0123456789012345678901234567890123456789" };

    fn init() !FetchTest {
        var tmp = std.testing.tmpDir(.{});
        errdefer tmp.cleanup();
        const dir = try tmp.dir.realPathFileAlloc(std.testing.io, ".", std.testing.allocator);
        errdefer std.testing.allocator.free(dir);
        const path = try std.fs.path.join(std.testing.allocator, &.{ dir, "repo", "model.gguf" });
        return .{ .tmp = tmp, .dir = dir, .path = path };
    }
    fn deinit(t: *FetchTest) void {
        std.testing.allocator.free(t.path);
        std.testing.allocator.free(t.dir);
        t.tmp.cleanup();
    }
    fn remote(bytes: []const u8) RemoteFile {
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
        return .{ .name = "model.gguf", .size = bytes.len, .sha256 = digest };
    }
    /// One download of `served` as `file`; `start` and `sent` say what moved.
    fn run(t: *FetchTest, origin_: Client.Origin, file: RemoteFile, served: []const u8, replace: bool, start: *?u64, sent: *u64) !Client.Fetched {
        start.* = null;
        sent.* = 0;
        const c: Client = .{ .allocator = std.testing.allocator, .io = std.testing.io, .progress = .{ .context = &t.stopper, .update = Stopper.update } };
        var event: ProgressEvent = .{ .phase = .verifying, .file_total = file.size };
        return c.fetchFile(origin_, t.path, file, replace, &event, 0, FakeSource{ .bytes = served, .start = start, .sent = sent });
    }
    fn read(t: *FetchTest, suffix: []const u8) !?[]u8 {
        const p = try std.mem.concat(std.testing.allocator, u8, &.{ t.path, suffix });
        defer std.testing.allocator.free(p);
        return Io.Dir.cwd().readFileAlloc(std.testing.io, p, std.testing.allocator, .limited(1 << 20)) catch |err| switch (err) {
            error.FileNotFound => null,
            else => err,
        };
    }
    fn expectFile(t: *FetchTest, suffix: []const u8, want: ?[]const u8) !void {
        const have = try t.read(suffix);
        defer if (have) |h| std.testing.allocator.free(h);
        if (want) |w| try std.testing.expectEqualStrings(w, have orelse return error.TestExpectedFile) else try std.testing.expect(have == null);
    }
};

test "a download publishes only verified bytes and replaces only when asked" {
    var t = try FetchTest.init();
    defer t.deinit();
    const content = FetchTest.content;
    const file = FetchTest.remote(content);
    var start: ?u64 = null;
    var sent: u64 = 0;
    // A digest that does not match publishes nothing and leaves no partial.
    var wrong = file;
    wrong.sha256 = @splat(0);
    try std.testing.expectError(error.ChecksumMismatch, t.run(FetchTest.origin, wrong, content, false, &start, &sent));
    try t.expectFile("", null);
    try t.expectFile(partial_suffix, null);
    const fetched = try t.run(FetchTest.origin, file, content, false, &start, &sent);
    try std.testing.expectEqual(file.sha256.?, fetched.sha256);
    try t.expectFile("", content);
    try t.expectFile(partial_suffix, null);
    try std.testing.expectEqual(file.sha256.?, (try verifyExisting(std.testing.io, t.path, file)).?);
    try std.testing.expectError(error.ExistingFileMismatch, verifyExisting(std.testing.io, t.path, wrong));
    // Without `replace` an existing destination is never clobbered.
    try std.testing.expectError(error.PathAlreadyExists, t.run(FetchTest.origin, file, content, false, &start, &sent));
    try t.expectFile("", content);
    try t.expectFile(partial_suffix, null);
    // A replacement that fails, by damage or by cancel, keeps the old file.
    const newer = "GGUF-the-next-revision-of-the-model-file";
    const newer_file = FetchTest.remote(newer);
    try std.testing.expectError(error.ChecksumMismatch, t.run(FetchTest.origin, newer_file, "GGUF-the-next-revision-of-the-model-fil!", true, &start, &sent));
    try t.expectFile("", content);
    t.stopper = .{ .after = 9, .stop = error.Canceled };
    try std.testing.expectError(error.Canceled, t.run(FetchTest.origin, newer_file, newer, true, &start, &sent));
    try t.expectFile("", content);
    try t.expectFile(partial_suffix, null);
    t.stopper = .{};
    _ = try t.run(FetchTest.origin, newer_file, newer, true, &start, &sent);
    try std.testing.expectEqual(@as(u64, newer.len), sent);
    try t.expectFile("", newer);
    try t.expectFile(partial_suffix, null);
}

test "a paused download keeps its partial and the next one continues it" {
    var t = try FetchTest.init();
    defer t.deinit();
    const content = FetchTest.content;
    const file = FetchTest.remote(content);
    var start: ?u64 = null;
    var sent: u64 = 0;
    t.stopper = .{ .after = 10 };
    try std.testing.expectError(error.Paused, t.run(FetchTest.origin, file, content, false, &start, &sent));
    try t.expectFile("", null);
    try t.expectFile(partial_suffix, content[0..12]);
    const state = (try t.read(state_suffix)).?;
    defer std.testing.allocator.free(state);
    try std.testing.expect(std.mem.indexOf(u8, state, "\"bytes\":12") != null);
    t.stopper = .{};
    const fetched = try t.run(FetchTest.origin, file, content, false, &start, &sent);
    try std.testing.expectEqual(@as(?u64, 12), start);
    try std.testing.expectEqual(@as(u64, 12), fetched.resumed);
    try std.testing.expectEqual(@as(u64, content.len - 12), sent);
    try std.testing.expectEqual(file.sha256.?, fetched.sha256);
    try t.expectFile("", content);
    try t.expectFile(partial_suffix, null);
    try t.expectFile(state_suffix, null);
}

test "a canceled download removes its partial" {
    var t = try FetchTest.init();
    defer t.deinit();
    const file = FetchTest.remote(FetchTest.content);
    var start: ?u64 = null;
    var sent: u64 = 0;
    t.stopper = .{ .after = 10, .stop = error.Canceled };
    try std.testing.expectError(error.Canceled, t.run(FetchTest.origin, file, FetchTest.content, false, &start, &sent));
    try t.expectFile(partial_suffix, null);
    try t.expectFile(state_suffix, null);
    try t.expectFile("", null);
}

test "a partial of other content, a damaged one, or a forced download starts over" {
    var t = try FetchTest.init();
    defer t.deinit();
    const content = FetchTest.content;
    const file = FetchTest.remote(content);
    var start: ?u64 = null;
    var sent: u64 = 0;
    // Another revision: the partial is discarded and the file starts over.
    t.stopper = .{ .after = 10 };
    try std.testing.expectError(error.Paused, t.run(FetchTest.origin, file, content, false, &start, &sent));
    t.stopper = .{};
    var other = FetchTest.origin;
    other.revision = "9999999999999999999999999999999999999999";
    var fetched = try t.run(other, file, content, false, &start, &sent);
    try std.testing.expectEqual(@as(?u64, 0), start);
    try std.testing.expectEqual(@as(u64, 0), fetched.resumed);
    try t.expectFile("", content);
    try Io.Dir.cwd().deleteFile(std.testing.io, t.path);
    // Another digest: the same.
    t.stopper = .{ .after = 10 };
    try std.testing.expectError(error.Paused, t.run(FetchTest.origin, file, content, false, &start, &sent));
    t.stopper = .{};
    const newer = "GGUF-the-next-revision-of-the-model-file";
    fetched = try t.run(FetchTest.origin, FetchTest.remote(newer), newer, false, &start, &sent);
    try std.testing.expectEqual(@as(?u64, 0), start);
    try t.expectFile("", newer);
    try Io.Dir.cwd().deleteFile(std.testing.io, t.path);
    // `replace` ignores a partial.
    t.stopper = .{ .after = 10 };
    try std.testing.expectError(error.Paused, t.run(FetchTest.origin, file, content, false, &start, &sent));
    t.stopper = .{};
    fetched = try t.run(FetchTest.origin, file, content, true, &start, &sent);
    try std.testing.expectEqual(@as(?u64, 0), start);
    try std.testing.expectEqual(@as(u64, content.len), sent);
    try t.expectFile(state_suffix, null);
    try Io.Dir.cwd().deleteFile(std.testing.io, t.path);
    // A damaged partial fails the whole file's digest and is removed.
    t.stopper = .{ .after = 10 };
    try std.testing.expectError(error.Paused, t.run(FetchTest.origin, file, content, false, &start, &sent));
    t.stopper = .{};
    const partial_path = try std.mem.concat(std.testing.allocator, u8, &.{ t.path, partial_suffix });
    defer std.testing.allocator.free(partial_path);
    const damaged = try Io.Dir.cwd().openFile(std.testing.io, partial_path, .{ .mode = .read_write });
    try damaged.writePositionalAll(std.testing.io, "#", 7);
    damaged.close(std.testing.io);
    try std.testing.expectError(error.ResumedChecksumMismatch, t.run(FetchTest.origin, file, content, false, &start, &sent));
    try std.testing.expectEqual(@as(?u64, 12), start);
    try t.expectFile(partial_suffix, null);
    try t.expectFile(state_suffix, null);
    try t.expectFile("", null);
}

test "a plain file verifies by its git blob id and records its SHA-256" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const content = "{\"vocab_size\": 8}\n";
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = content });
    // `printf '{"vocab_size": 8}\n' | git hash-object --stdin`
    var oid: [20]u8 = undefined;
    _ = try std.fmt.hexToBytes(&oid, "ef44a803696cdca06508fe711488cdc982eb64d3");
    const remote: RemoteFile = .{ .name = "config.json", .size = content.len, .git_oid = oid };
    const path = try tmp.dir.realPathFileAlloc(io, "config.json", std.testing.allocator);
    defer std.testing.allocator.free(path);
    var sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(content, &sha, .{});
    try std.testing.expectEqual(sha, (try verifyExisting(io, path, remote)).?);
    var wrong = remote;
    wrong.git_oid.?[0] ^= 1;
    try std.testing.expectError(error.ExistingFileMismatch, verifyExisting(io, path, wrong));
    try std.testing.expect(try verifyExisting(io, "/nonexistent/config.json", remote) == null);
}

test "the first bytes must be the container the name says" {
    try checkContainer(.gguf, "GG", 100);
    try std.testing.expectError(error.InvalidGguf, checkContainer(.gguf, "GGML", 100));
    try checkContainer(null, "anything", 100);
    var head: [9]u8 = undefined;
    std.mem.writeInt(u64, head[0..8], 2, .little);
    head[8] = '{';
    try checkContainer(.safetensors, &head, 10);
    try checkContainer(.safetensors, head[0..5], 10);
    try std.testing.expectError(error.InvalidSafetensors, checkContainer(.safetensors, &head, 9));
    head[8] = '[';
    try std.testing.expectError(error.InvalidSafetensors, checkContainer(.safetensors, &head, 10));
}

test "local path mirrors the Hub path without the revision" {
    const c: Client = .{ .allocator = std.testing.allocator, .io = std.testing.io, .nuclis_home = "/data/nuclis" };
    const path = try c.localPath(.{ .repo_id = "unsloth/Qwen3.8-27B-GGUF" }, "Qwen3.8-27B-UD-Q4_K_M.gguf");
    defer std.testing.allocator.free(path);
    try std.testing.expectEqualStrings("/data/nuclis/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf", path);
    try std.testing.expectError(error.InvalidFilename, c.localPath(.{ .repo_id = "a/b" }, "../x.gguf"));
}

test "progress can cancel before any network or file operation" {
    const Handler = struct {
        fn update(_: ?*anyopaque, event: ProgressEvent) error{Canceled}!void {
            std.debug.assert(event.phase == .resolving);
            return error.Canceled;
        }
    };
    const c: Client = .{ .allocator = std.testing.allocator, .io = std.testing.io, .progress = .{ .update = Handler.update } };
    try std.testing.expectError(error.Canceled, c.download(.{ .repo_id = "a/b" }));
    var invalid = c;
    invalid.concurrency = 0;
    try std.testing.expectError(error.InvalidConcurrency, invalid.download(.{ .repo_id = "a/b" }));
}
