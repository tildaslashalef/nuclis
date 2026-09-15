//! Internal model acquisition API, independent of the inference engine and CLI.
//!
//! Client borrows its environment strings and optional token for its lifetime.
//! All I/O uses the injected std.Io; the allocator must be thread safe. Calls
//! return owned results with deinit(), never pointers into HTTP response buffers.
//! Xet is the default for Xet-backed files. A failed Xet transfer is an error,
//! not an implicit HTTP fallback. Full files are SHA-256 verified before publish.
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
//! conflict, never silently replaced.
const std = @import("std");
const Io = std.Io;
const hub = @import("hub.zig");
const transfer = @import("transfer.zig");
pub const Request = hub.Request;
pub const RemoteFile = hub.File;
pub const Catalog = hub.Catalog;
pub const max_range_bytes = transfer.window_size;
/// Which files a request would download from a catalog (null: several
/// choices), so a host can check destinations before calling `download`.
pub const select = hub.select;

pub const LocalFile = struct {
    path: []const u8,
    size: u64,
    sha256: [32]u8,
    transport: enum { xet, http, verified_local },
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

    /// A repo-only request with multiple quantizations returns selection_required
    /// without creating directories or downloading model bytes. An exact split
    /// filename selects its whole shard set. Existing files are reused only after
    /// size and full SHA-256 verification; conflicting files are never overwritten.
    pub fn download(c: Client, request: Request) !Result {
        if (c.concurrency == 0 or c.concurrency > max_concurrency) return error.InvalidConcurrency;
        try c.report(.{ .phase = .resolving });
        var catalog = try c.list(request);
        defer catalog.deinit();
        var arena = std.heap.ArenaAllocator.init(c.allocator);
        errdefer arena.deinit();
        const a = arena.allocator();
        const selected = try hub.select(a, catalog.files, request.filename);
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
            destination.* = .{ .path = path, .size = file.size, .sha256 = file.sha256, .transport = .verified_local };
            var event: ProgressEvent = .{ .phase = .verifying, .filename = file.name, .file_index = file_index, .file_count = local.len, .file_total = file.size, .completed_bytes = completed, .total_bytes = total };
            try c.report(event);
            if (try verifyExisting(c.io, path, file)) {
                completed += file.size;
                event.reused = true;
                event.file_completed = file.size;
                event.completed_bytes = completed;
                try c.report(event);
                continue;
            }
            event.phase = .downloading;
            try c.report(event);
            // One session per file: the connection pool and the CAS read token
            // are shared by every window of this file and torn down after the
            // last worker has been joined (stream's defer runs first).
            var session = transfer.Session.init(c.allocator, c.io, c.token, catalog);
            defer session.deinit();
            const remote = try transfer.resolve(&session, file);
            defer remote.deinit(c.allocator);
            destination.transport = if (remote.xet_hash != null) .xet else .http;
            // AtomicFile owns cleanup on every failure, including cancellation.
            // Its final link refuses to clobber a concurrently created destination.
            var atomic = try std.Io.Dir.cwd().createFileAtomic(c.io, path, .{ .make_path = true });
            defer atomic.deinit(c.io);
            var hasher = std.crypto.hash.sha2.Sha256.init(.{});
            var sink: FileSink = .{ .client = c, .atomic = &atomic, .hasher = &hasher, .event = &event, .completed = completed };
            try transfer.fetch(&session, remote, 0, file.size, c.concurrency, &sink);
            event.phase = .verifying;
            try c.report(event);
            try publish(c.io, &atomic, &hasher, file.sha256);
            completed += file.size;
        }
        try c.report(.{ .phase = .complete, .file_count = local.len, .completed_bytes = completed, .total_bytes = total });
        return .{ .arena = arena, .revision = catalog.revision, .outcome = .{ .downloaded = local } };
    }

    pub const max_concurrency = transfer.max_concurrency;

    /// Receives the file's bytes in order from `transfer.fetch` (whole chunks
    /// of at most 128 KiB): checks the GGUF magic, hashes, appends to the
    /// atomic file, and reports progress. Runs on the calling task only.
    const FileSink = struct {
        client: Client,
        atomic: *Io.File.Atomic,
        hasher: *std.crypto.hash.sha2.Sha256,
        event: *ProgressEvent,
        completed: u64,
        position: u64 = 0,
        pub fn write(sink: *FileSink, bytes: []const u8) !void {
            if (sink.position < 4) {
                const magic = "GGUF";
                const n = @min(bytes.len, 4 - @as(usize, @intCast(sink.position)));
                if (!std.mem.eql(u8, bytes[0..n], magic[@intCast(sink.position)..][0..n])) return error.InvalidGguf;
            }
            sink.hasher.update(bytes);
            try sink.atomic.file.writeStreamingAll(sink.client.io, bytes);
            sink.position += bytes.len;
            sink.event.file_completed = sink.position;
            sink.event.completed_bytes = sink.completed + sink.position;
            try sink.client.report(sink.event.*);
        }
    };

    fn report(c: Client, event: ProgressEvent) error{Canceled}!void {
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

fn publish(io: std.Io, atomic: *std.Io.File.Atomic, hasher: *std.crypto.hash.sha2.Sha256, expected: [32]u8) !void {
    var hash: [32]u8 = undefined;
    hasher.final(&hash);
    if (!std.mem.eql(u8, &hash, &expected)) return error.ChecksumMismatch;
    try atomic.file.sync(io);
    try atomic.link(io);
}

fn verifyExisting(io: std.Io, path: []const u8, expected: RemoteFile) !bool {
    const file = std.Io.Dir.cwd().openFile(io, path, .{}) catch |err| switch (err) {
        error.FileNotFound => return false,
        else => return err,
    };
    defer file.close(io);
    const stat = try file.stat(io);
    if (stat.kind != .file or stat.size != expected.size) return error.ExistingFileMismatch;
    var buffer: [64 * 1024]u8 = undefined;
    var reader = file.reader(io, &buffer);
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    var chunk: [64 * 1024]u8 = undefined;
    var size: u64 = 0;
    while (true) {
        const n = try reader.interface.readSliceShort(&chunk);
        if (n == 0) break;
        size += n;
        if (size > expected.size) return error.ExistingFileMismatch;
        hasher.update(chunk[0..n]);
    }
    var digest: [32]u8 = undefined;
    hasher.final(&digest);
    if (size != expected.size or !std.mem.eql(u8, &digest, &expected.sha256)) return error.ExistingFileMismatch;
    return true;
}

test {
    _ = hub;
    _ = @import("http.zig");
    _ = @import("xorb.zig");
    _ = transfer;
    _ = @import("progress.zig");
}

test "publication verifies before link and preserves conflicting destinations" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update("GGUFtest");
    var expected: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash("GGUFtest", &expected, .{});
    {
        var file = try tmp.dir.createFileAtomic(io, "model.gguf", .{});
        defer file.deinit(io);
        try file.file.writeStreamingAll(io, "GGUFtest");
        try std.testing.expectError(error.ChecksumMismatch, publish(io, &file, &hash, @splat(0)));
        try std.testing.expectError(error.FileNotFound, tmp.dir.openFile(io, "model.gguf", .{}));
    }
    {
        var file = try tmp.dir.createFileAtomic(io, "model.gguf", .{});
        defer file.deinit(io);
        try file.file.writeStreamingAll(io, "GGUFtest");
        hash = .init(.{});
        hash.update("GGUFtest");
        try publish(io, &file, &hash, expected);
    }
    {
        var file = try tmp.dir.createFileAtomic(io, "model.gguf", .{});
        defer file.deinit(io);
        try file.file.writeStreamingAll(io, "GGUFtest");
        hash = .init(.{});
        hash.update("GGUFtest");
        try std.testing.expectError(error.PathAlreadyExists, publish(io, &file, &hash, expected));
    }
    const path = try tmp.dir.realPathFileAlloc(io, "model.gguf", std.testing.allocator);
    defer std.testing.allocator.free(path);
    try std.testing.expect(try verifyExisting(io, path, .{ .name = "model.gguf", .size = 8, .sha256 = expected }));
    try std.testing.expectError(error.ExistingFileMismatch, verifyExisting(io, path, .{ .name = "model.gguf", .size = 8, .sha256 = @splat(0) }));
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
