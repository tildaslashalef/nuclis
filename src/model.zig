//! `nuclis model`: acquiring artifacts through the `huggingface` package,
//! listing what the models directory holds, and judging a remote file
//! from its directory alone before any download (`inspect`).
//!
//! Layout: `<root>/models/<owner>/<repo>/<file>`, the package's own
//! (`Client.localPath`), predictable from the repository id alone and
//! mirroring the Hub so files every repository names identically
//! (`mmproj-BF16.gguf`, `MTP/`) never collide.
//!
//! Provenance: beside every verified file sits `<file>.nuclis.json`, a
//! sidecar recording the repository, resolved commit, SHA-256, size, role,
//! download time, and the nuclis version that wrote it. It is written only
//! after the package's atomic publication, or its full-hash verification of
//! a file already in place, succeeded; nothing is written for a hand-copied
//! file until a `pull` verifies it.
//!
//! Each command allocates into one arena freed on return. Progress renders on
//! stderr from the package's sink, where the Ctrl-C flag is polled; results
//! go to the caller's writer so `--json` and the text form derive from the
//! same values. Remote inspection reads the file's head through `readRange`
//! until the GGUF directory parses (bounded by the parser's 64 MiB limit):
//! the adapter binding makes a file runnable, the catalogue pinning the Hub
//! digest makes it supported, and otherwise the first offending tensor or
//! encoding is the verdict. `Sidecar` is both the JSON schema and the value,
//! so `std.json` walks its fields to parse and to print.
const std = @import("std");
const hf = @import("huggingface");
const inference = @import("inference");
const config = @import("config.zig");
const catalog = @import("catalog.zig");
const interrupt = @import("interrupt.zig");
const inspection = @import("inspect.zig");
const style = @import("tui/style.zig");
const Allocator = std.mem.Allocator;

pub const version = @import("build_options").version;

/// What a file is for. The header decides when it says so (`general.type`
/// `imatrix` or `mmproj`, or a `clip` architecture); otherwise the `--role`
/// flag (the catalogue resolves it later), and `main` when nothing says.
pub const Role = enum { main, mmproj, mtp, imatrix };

pub const sidecar_suffix = ".nuclis.json";
/// A sidecar is a few hundred bytes; anything larger is not ours.
pub const max_sidecar_bytes = 64 * 1024;
/// Upper bound on files a listing walks, and on nesting below a repository.
pub const max_listed_files = 10_000;
pub const max_depth_below_repo = 4;

pub const PullOptions = struct {
    /// A catalogue name (`qwen3.8-27b`) or a repository id (`owner/repo`).
    repo: []const u8 = "",
    /// Companions to add to a catalogue pull, by role; `all` takes every
    /// companion the entry lists. Meaningless with a repository id.
    with: std.EnumSet(Role) = .initEmpty(),
    all: bool = false,
    file: ?[]const u8 = null,
    /// `main`, a tag, a branch, or a commit; the sidecar always records the
    /// commit it resolved to.
    revision: ?[]const u8 = null,
    role: ?Role = null,
    /// Replace a file whose sidecar records different content, or one nuclis
    /// never verified; never set by default.
    force: bool = false,
    /// A registry entry's name, for the report (`fromRegistry` sets it).
    name: ?[]const u8 = null,
    /// Companion files to fetch beside `file` from a repository id, by
    /// exact Hub name; `fromRegistry` fills them from the entry for
    /// `--with`/`--all`.
    mmproj: ?[]const u8 = null,
    mtp: ?[]const u8 = null,
    /// `--register <name>`: write the pulled files as a registry entry of
    /// `config_path` once they are verified; `profile` forces the entry's
    /// prompt profile (`--profile`, meaningful only with `--register`).
    register: ?[]const u8 = null,
    profile: ?config.Profile = null,
    config_path: ?[]const u8 = null,
};

/// A name that is neither a catalogue entry nor an `owner/repo` id can
/// only be a registry entry: the caller then reads `nuclis.json`, which
/// the other forms never need (a broken file must not block a download).
pub fn isRegistryName(name: []const u8) bool {
    return name.len > 0 and catalog.find(name) == null and std.mem.indexOfScalar(u8, name, '/') == null;
}

/// `pull <entry>`: the entry's repository, file, and pinned revision as a
/// repository-id pull, with `--with`/`--all` mapped to its companion names.
/// The Hub's digests apply (a registry entry pins no digest).
pub fn fromRegistry(options: PullOptions, entry: *const config.ModelEntry, diag: *config.Diagnostic) !PullOptions {
    if (entry.path) |path| {
        diag.set("{s} names a local path ({s}); nothing to pull", .{ options.repo, path });
        return error.NotPullable;
    }
    if (options.file != null or options.revision != null) {
        diag.set("{s} is a registry entry and pins its file and revision; pull by repository id ({s}) to choose another", .{ options.repo, entry.repo.? });
        return error.ConflictingOptions;
    }
    var wanted = options.with.iterator();
    while (wanted.next()) |role| {
        const companion: ?[]const u8 = switch (role) {
            .mmproj => entry.mmproj,
            .mtp => entry.mtp,
            else => null,
        };
        if (companion == null) {
            diag.set("{s} has no {s} companion in the registry", .{ options.repo, @tagName(role) });
            return error.NoSuchCompanion;
        }
    }
    var translated = options;
    translated.name = options.repo;
    translated.repo = entry.repo.?;
    translated.file = entry.file.?;
    translated.revision = entry.revision;
    translated.with = .initEmpty();
    translated.all = false;
    if (options.all or options.with.contains(.mmproj)) translated.mmproj = entry.mmproj;
    if (options.all or options.with.contains(.mtp)) translated.mtp = entry.mtp;
    return translated;
}

/// The provenance record. `revision` is the 40-character commit and
/// `sha256` the 64-character lowercase digest, both as the Hub reported
/// them; `size` lets `ls` flag a file that changed underneath its sidecar.
pub const Sidecar = struct {
    schema_version: u32 = 1,
    repo: []const u8,
    file: []const u8,
    revision: []const u8,
    sha256: []const u8,
    size: u64,
    role: Role,
    /// RFC 3339 UTC, second resolution.
    downloaded_at: []const u8,
    nuclis_version: []const u8,
};

pub fn sidecarPath(alloc: Allocator, path: []const u8) ![]u8 {
    return std.mem.concat(alloc, u8, &.{ path, sidecar_suffix });
}

/// Null when there is no sidecar; `InvalidSidecar` when there is one nuclis
/// cannot read (a hand edit, a future schema). Strings live in `alloc`.
pub fn readSidecar(alloc: Allocator, io: std.Io, dir: std.Io.Dir, sub_path: []const u8) !?Sidecar {
    const bytes = dir.readFileAlloc(io, sub_path, alloc, .limited(max_sidecar_bytes)) catch |err| switch (err) {
        error.FileNotFound => return null,
        else => return err,
    };
    defer alloc.free(bytes);
    const parsed = std.json.parseFromSliceLeaky(Sidecar, alloc, bytes, .{ .allocate = .alloc_always }) catch return error.InvalidSidecar;
    if (parsed.schema_version != 1 or parsed.revision.len != 40 or parsed.sha256.len != 64) return error.InvalidSidecar;
    return parsed;
}

/// Written whole; a sidecar is small enough that a torn write is a
/// `InvalidSidecar` on the next read, which `pull` treats as absent.
pub fn writeSidecar(alloc: Allocator, io: std.Io, dir: std.Io.Dir, sub_path: []const u8, sidecar: Sidecar) !void {
    var buffer: std.Io.Writer.Allocating = .init(alloc);
    defer buffer.deinit();
    try std.json.Stringify.value(sidecar, .{ .whitespace = .indent_2 }, &buffer.writer);
    try buffer.writer.writeByte('\n');
    try dir.writeFile(io, .{ .sub_path = sub_path, .data = buffer.written() });
}

/// `YYYY-MM-DDThh:mm:ssZ` from Unix seconds; negative times are clamped to
/// the epoch (the clock is only ever read after 1970).
pub fn rfc3339(buffer: *[20]u8, unix_seconds: i64) []const u8 {
    const secs: u64 = if (unix_seconds < 0) 0 else @intCast(unix_seconds);
    const epoch: std.time.epoch.EpochSeconds = .{ .secs = secs };
    const day = epoch.getEpochDay().calculateYearDay();
    const month_day = day.calculateMonthDay();
    const time = epoch.getDaySeconds();
    return std.fmt.bufPrint(buffer, "{d:0>4}-{d:0>2}-{d:0>2}T{d:0>2}:{d:0>2}:{d:0>2}Z", .{
        day.year,               month_day.month.numeric(), @as(u32, month_day.day_index) + 1,
        time.getHoursIntoDay(), time.getMinutesIntoHour(), time.getSecondsIntoMinute(),
    }) catch unreachable;
}

/// The role rule: the header wins when it speaks; a flag that contradicts it
/// is an error rather than a silent override.
pub fn resolveRole(header: ?Role, flag: ?Role) error{RoleMismatch}!Role {
    if (header) |h| {
        if (flag) |f| if (f != h) return error.RoleMismatch;
        return h;
    }
    return flag orelse .main;
}

/// What the GGUF header says about itself, reading the directory only.
fn roleFromHeader(alloc: Allocator, io: std.Io, path: []const u8) !?Role {
    var doc = try inference.gguf.open(alloc, io, path, .{});
    defer doc.deinit();
    if (doc.string("general.type")) |kind| {
        if (std.mem.eql(u8, kind, "imatrix")) return .imatrix;
        if (std.mem.eql(u8, kind, "mmproj")) return .mmproj;
    }
    if (doc.string("general.architecture")) |arch| if (std.mem.eql(u8, arch, "clip")) return .mmproj;
    return null;
}

pub fn modelsDir(alloc: Allocator, root: []const u8) ![]u8 {
    return std.fs.path.join(alloc, &.{ root, "models" });
}

// ---- pull ------------------------------------------------------------------

const PulledFile = struct {
    path: []const u8,
    size: u64,
    sha256: []const u8,
    transport: []const u8,
    role: Role,
    sidecar: []const u8,
};

const PullReport = struct {
    schema_version: u32 = 2,
    /// The catalogue or registry name when one was given, else null.
    name: ?[]const u8,
    repo: []const u8,
    requested_revision: []const u8,
    revision: []const u8,
    files: []const PulledFile,
    /// The registry entry `--register` wrote, and the file it lives in.
    registered: ?struct { name: []const u8, config: []const u8 } = null,
};

const SelectionReport = struct {
    schema_version: u32 = 1,
    repo: []const u8,
    revision: []const u8,
    selection_required: []const Choice,
    const Choice = struct { name: []const u8, size: u64 };
};

/// One file to fetch: its exact Hub name, the role the caller knows (the
/// catalogue's or `--role`), and the digest it must have (the catalogue's
/// pinned one, or the Hub's own for a raw repository).
const Job = struct { name: []const u8, role: ?Role, sha256: []const u8, size: u64 };

/// Resolves the repository (a catalogue name or an `owner/repo` id) and
/// its revision, prints the pinned commit, downloads or verifies each file,
/// and writes the sidecars. `diag` carries the reason of every typed
/// failure.
pub fn pull(gpa: Allocator, io: std.Io, environ: *const std.process.Environ.Map, root: []const u8, options: PullOptions, json: bool, out: *std.Io.Writer, sty: style.Style, diag: *config.Diagnostic) !void {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    if (options.repo.len == 0) return error.MissingRepository;
    const models = try modelsDir(arena, root);

    interrupt.install();
    // The package's allocator must be thread safe: the concurrent workers
    // share it. `gpa` from `std.process.Init` is.
    var client = hf.Client.init(gpa, io, environ);
    var display: Display = .{ .io = io, .started = .now(io, .awake), .tty = std.Io.File.stderr().isTty(io) catch false };
    client.progress = .{ .context = &display, .update = Display.update };

    // 1. Resolve. A catalogue name pins repository, files, commit, and
    //    digests from the table; a repository id resolves them on the Hub.
    var jobs: std.ArrayList(Job) = .empty;
    var request: hf.Request = .{ .repo_id = options.repo, .revision = options.revision orelse "main", .local_dir = models };
    const entry = catalog.find(options.repo);
    var pinned: [40]u8 = undefined;
    if (entry) |e| {
        if (options.file != null or options.revision != null) {
            diag.set("{s} is a catalogue name and pins its file and commit; pull by repository id ({s}) to choose another", .{ e.name, e.repo });
            return error.ConflictingOptions;
        }
        request.repo_id = e.repo;
        request.revision = e.revision;
        @memcpy(&pinned, e.revision[0..40]);
        try jobs.append(arena, .{ .name = e.file, .role = .main, .sha256 = e.sha256, .size = e.size });
        for (e.companions) |c| if (options.all or options.with.contains(c.role)) try jobs.append(arena, .{ .name = c.file, .role = c.role, .sha256 = c.sha256, .size = c.size });
        var wanted = options.with.iterator();
        while (wanted.next()) |role| if (e.companion(role) == null) {
            diag.set("{s} has no {s} companion in the catalogue", .{ e.name, @tagName(role) });
            return error.NoSuchCompanion;
        };
    } else {
        if (options.all or options.with.count() != 0) {
            diag.set("--with and --all need a catalogue name; {s} is a repository id (pass --file per companion)", .{options.repo});
            return error.ConflictingOptions;
        }
        request.filename = options.file;
        var remote = client.list(request) catch |err| return hubFailure(err, options.repo, diag);
        defer remote.deinit();
        pinned = remote.revision;
        const selected = (hf.select(arena, remote.files, request.filename) catch |err| return hubFailure(err, options.repo, diag)) orelse {
            try renderSelection(out, .{ .repo = options.repo, .revision = &pinned, .selection_required = try choices(arena, remote.files) }, json, sty);
            try out.flush();
            diag.set("{s} has {d} GGUF files; pass --file <name>", .{ options.repo, remote.files.len });
            return error.SelectionRequired;
        };
        for (selected) |f| try jobs.append(arena, .{ .name = try arena.dupe(u8, f.name), .role = options.role, .sha256 = try arena.dupe(u8, &std.fmt.bytesToHex(f.sha256, .lower)), .size = f.size });
        // A registry entry's companions, by exact name at the same commit.
        for ([_]struct { name: ?[]const u8, role: Role }{ .{ .name = options.mmproj, .role = .mmproj }, .{ .name = options.mtp, .role = .mtp } }) |companion| {
            const name = companion.name orelse continue;
            const file = for (remote.files) |f| {
                if (std.mem.eql(u8, f.name, name)) break f;
            } else {
                diag.set("{s}: no GGUF named {s} in the repository (the registry entry's {s} companion)", .{ options.repo, name, @tagName(companion.role) });
                return error.FileNotFound;
            };
            try jobs.append(arena, .{ .name = try arena.dupe(u8, file.name), .role = companion.role, .sha256 = try arena.dupe(u8, &std.fmt.bytesToHex(file.sha256, .lower)), .size = file.size });
        }
    }
    const revision: []const u8 = &pinned;
    request.revision = revision;
    if (options.register) |name| {
        if (options.config_path == null) return error.MissingHome;
        const main_file = for (jobs.items) |job| {
            if (job.role == null or job.role == .main) break job.name;
        } else null;
        try config.registrable(name, request.repo_id, main_file, diag);
    }
    if (!json) try renderHeader(out, sty, if (entry) |e| e.name else options.name, request.repo_id, if (entry != null) null else options.revision orelse "main", revision);
    try out.flush();

    // 2. Destinations: a sidecar recording other content is a conflict
    //    (`--force` replaces file and sidecar); a file without one is left
    //    to the package's verification unless forced.
    for (jobs.items) |job| {
        const path = try client.localPath(request, job.name);
        defer gpa.free(path);
        const sidecar = try sidecarPath(arena, path);
        const recorded = readSidecar(arena, io, .cwd(), sidecar) catch |err| switch (err) {
            error.InvalidSidecar => null,
            else => return err,
        };
        if (recorded) |r| {
            if (std.mem.eql(u8, r.sha256, job.sha256) and r.size == job.size) continue;
            if (!options.force) {
                diag.set("{s}: its sidecar records commit {s} (sha256 {s}…), the request resolved to commit {s} (sha256 {s}…); pass --force to replace it", .{ path, r.revision[0..12], r.sha256[0..12], revision[0..12], job.sha256[0..12] });
                return error.ExistingFileMismatch;
            }
            try deleteIfPresent(io, path);
            try deleteIfPresent(io, sidecar);
        } else if (options.force) {
            try deleteIfPresent(io, path);
        }
    }

    // 3. Transfer, one package call per job (a split file expands to its
    //    shard set inside the package). The package verifies existing
    //    files, never overwrites, and removes its temporary file on every
    //    failure including cancel. 4. Provenance, only once a file is
    //    verified, so an interrupted multi-file pull keeps the sidecars of
    //    what it finished.
    var report: std.ArrayList(PulledFile) = .empty;
    // What `--register` records: filled per verified file by its role.
    var registration: config.Registration = .{ .repo = request.repo_id, .revision = revision, .file = null, .profile = options.profile };
    var stamp: [20]u8 = undefined;
    const downloaded_at = rfc3339(&stamp, std.Io.Timestamp.now(io, .real).toSeconds());
    for (jobs.items) |job| {
        request.filename = job.name;
        var result = client.download(request) catch |err| switch (err) {
            error.Canceled => {
                diag.set("interrupted; the partial download was removed", .{});
                return error.Cancelled;
            },
            error.ExistingFileMismatch => {
                diag.set("{s}: a file nuclis never verified is at the destination and differs from the Hub's; move it away or pass --force", .{job.name});
                return err;
            },
            else => return hubFailure(err, options.repo, diag),
        };
        defer result.deinit();
        const files = switch (result.outcome) {
            .downloaded => |files| files,
            .selection_required => return error.SelectionRequired,
        };
        for (files) |file| {
            const digest = try arena.dupe(u8, &std.fmt.bytesToHex(file.sha256, .lower));
            // The Hub at the pinned commit must agree with the catalogue;
            // otherwise the table is wrong, and the file stays without a
            // sidecar so it is never taken for verified.
            if (entry != null and !std.mem.eql(u8, digest, job.sha256)) {
                diag.set("{s}: the Hub's digest at commit {s} is {s}…, the catalogue pins {s}…; the catalogue entry needs updating", .{ file.path, revision[0..12], digest[0..12], job.sha256[0..12] });
                return error.CatalogMismatch;
            }
            const header = try roleFromHeader(gpa, io, file.path);
            const role = resolveRole(header, job.role) catch |err| {
                diag.set("{s}: the header says its role is {s}; --role {s} contradicts it (no sidecar written)", .{ file.path, @tagName(header.?), @tagName(job.role.?) });
                return err;
            };
            const sidecar = try sidecarPath(arena, file.path);
            // The Hub name of a published file (a shard set returns
            // several): the path below the repository directory, without
            // the separator `join` leaves in front of it.
            const repo_dir = try std.fs.path.join(arena, &.{ models, request.repo_id });
            try writeSidecar(arena, io, .cwd(), sidecar, .{
                .repo = request.repo_id,
                .file = if (std.mem.startsWith(u8, file.path, repo_dir)) std.mem.trimStart(u8, file.path[repo_dir.len..], "/") else job.name,
                .revision = revision,
                .sha256 = digest,
                .size = file.size,
                .role = role,
                .downloaded_at = downloaded_at,
                .nuclis_version = version,
            });
            try report.append(arena, .{ .path = try arena.dupe(u8, file.path), .size = file.size, .sha256 = digest, .transport = @tagName(file.transport), .role = role, .sidecar = sidecar });
            switch (role) {
                .main => registration.file = job.name,
                .mmproj => registration.mmproj = job.name,
                .mtp => registration.mtp = job.name,
                .imatrix => {},
            }
        }
    }
    // 5. The registry entry, only once every file above is verified, so a
    //    failed pull registers nothing.
    var registered: ?@FieldType(PullReport, "registered") = null;
    if (options.register) |name| {
        const config_path = options.config_path orelse return error.MissingHome;
        const current = try config.readText(gpa, io, .cwd(), config_path, diag);
        defer if (current) |c| gpa.free(c);
        const text = try config.register(gpa, current, config_path, name, registration, diag);
        defer gpa.free(text);
        try config.write(io, .cwd(), config_path, text);
        registered = .{ .name = name, .config = config_path };
    }
    try renderPull(out, .{ .name = if (entry) |e| e.name else options.name, .repo = request.repo_id, .requested_revision = options.revision orelse (if (entry) |e| e.revision else "main"), .revision = revision, .files = report.items, .registered = registered orelse null }, json, sty);
}

/// `name = repo @ requested: commit <40 hex>`; the name and the requested
/// revision are omitted when absent (a catalogue pull pins its commit).
fn renderHeader(out: *std.Io.Writer, sty: style.Style, name: ?[]const u8, repo: []const u8, requested: ?[]const u8, revision: []const u8) !void {
    if (name) |n| try out.print("{s}{s}{s} = ", .{ sty.on(.keyword), n, sty.off() });
    try out.print("{s}{s}{s} @ ", .{ sty.on(.code), repo, sty.off() });
    if (requested) |r| try out.print("{s}: ", .{r});
    try out.print("commit {s}{s}{s}\n", .{ sty.on(.comment), revision, sty.off() });
}

fn choices(arena: Allocator, files: []const hf.RemoteFile) ![]const SelectionReport.Choice {
    const out = try arena.alloc(SelectionReport.Choice, files.len);
    for (files, out) |f, *c| c.* = .{ .name = f.name, .size = f.size };
    return out;
}

fn renderSelection(out: *std.Io.Writer, report: SelectionReport, json: bool, sty: style.Style) !void {
    if (json) {
        try std.json.Stringify.value(report, .{ .whitespace = .indent_2 }, out);
        return out.writeByte('\n');
    }
    try out.print("{s}{s}{s} @ commit {s}{s}{s} has {s}{d}{s} GGUF files; choose one with --file <name>:\n", .{ sty.on(.code), report.repo, sty.off(), sty.on(.comment), report.revision, sty.off(), sty.on(.number), report.selection_required.len, sty.off() });
    for (report.selection_required) |c| try out.print("  {s}{s}{s}\t{s}{d}{s}\n", .{ sty.on(.code), c.name, sty.off(), sty.on(.number), c.size, sty.off() });
}

fn renderPull(out: *std.Io.Writer, report: PullReport, json: bool, sty: style.Style) !void {
    if (json) {
        try std.json.Stringify.value(report, .{ .whitespace = .indent_2 }, out);
        return out.writeByte('\n');
    }
    for (report.files) |f| {
        try out.print("{s}{s}{s} {s}{s}{s}\n  {s}{d}{s} bytes, sha256 {s}{s}{s}, role {s}{s}{s}; provenance in {s}{s}{s}{s}\n", .{
            sty.on(.success),
            if (std.mem.eql(u8, f.transport, "verified_local")) "verified" else "downloaded",
            sty.off(),
            sty.on(.code),
            f.path,
            sty.off(),
            sty.on(.number),
            f.size,
            sty.off(),
            sty.on(.comment),
            f.sha256,
            sty.off(),
            sty.on(.number),
            @tagName(f.role),
            sty.off(),
            sty.on(.dim),
            std.fs.path.basename(f.path),
            sidecar_suffix,
            sty.off(),
        });
    }
    if (report.registered) |r| try out.print("{s}registered{s} as {s}{s}{s} in {s}{s}{s} {s}(`nuclis config set engine.model {s}` makes it the default){s}\n", .{ sty.on(.success), sty.off(), sty.on(.keyword), r.name, sty.off(), sty.on(.code), r.config, sty.off(), sty.on(.dim), r.name, sty.off() });
}

fn deleteIfPresent(io: std.Io, path: []const u8) !void {
    std.Io.Dir.cwd().deleteFile(io, path) catch |err| switch (err) {
        error.FileNotFound => {},
        else => return err,
    };
}

/// Names the fix for the Hub's refusals (the package's typed errors); other
/// errors pass through unchanged.
fn hubFailure(err: anyerror, repo: []const u8, diag: *config.Diagnostic) anyerror {
    switch (err) {
        error.TokenRequired => diag.set("{s} needs authentication: set HF_TOKEN to a Hugging Face user access token", .{repo}),
        error.TokenRejected => diag.set("HF_TOKEN was refused by the Hub: check the token value and that it has not expired or been revoked", .{}),
        error.AccessDenied => diag.set("access to {s} denied: if it is gated, accept its terms on huggingface.co with the account HF_TOKEN belongs to", .{repo}),
        error.NotFound => diag.set("{s}: no such repository, revision, or file on the Hub", .{repo}),
        error.FileNotFound => diag.set("{s}: no GGUF of that name in the repository (names match in full, subdirectories included)", .{repo}),
        else => {},
    }
    return err;
}

/// Renders the package's progress on stderr: one line per phase and, on a
/// terminal, a line rewritten in place at most every 200 ms with the file's
/// percentage, bytes, throughput since the download started, and the
/// estimated remaining time. Off a terminal only the phase lines print.
/// Write failures are ignored: progress is advisory. The interrupt flag is
/// polled here because this is the one callback the package makes between
/// chunks.
const Display = struct {
    io: std.Io,
    started: std.Io.Timestamp,
    tty: bool,
    last_line: ?std.Io.Timestamp = null,
    last_phase: ?@FieldType(hf.ProgressEvent, "phase") = null,
    download_started: ?std.Io.Timestamp = null,
    download_base: u64 = 0,
    open_line: bool = false,

    fn update(context: ?*anyopaque, event: hf.ProgressEvent) error{Canceled}!void {
        const d: *Display = @ptrCast(@alignCast(context.?));
        if (interrupt.requested()) {
            d.render(.{ .phase = .selection_required }) catch {};
            return error.Canceled;
        }
        d.render(event) catch {};
    }

    fn render(d: *Display, event: hf.ProgressEvent) !void {
        var buffer: [512]u8 = undefined;
        // Streaming, not positional: a positional writer restarts at offset
        // zero on every call and overwrites a redirected log.
        var stderr = std.Io.File.stderr().writerStreaming(d.io, &buffer);
        const w = &stderr.interface;
        defer w.flush() catch {};
        const now: std.Io.Timestamp = .now(d.io, .awake);
        const name = event.filename orelse "";
        const phase_changed = d.last_phase == null or d.last_phase.? != event.phase;
        d.last_phase = event.phase;
        switch (event.phase) {
            // Each package call starts here: the overall clock is per file.
            .resolving => {
                d.started = now;
                try w.writeAll("resolving revision and files\n");
            },
            // Reused by `update` as the cancel notice: the phase never comes
            // from the package after selection succeeded.
            .selection_required => {
                try d.endLine(w);
                try w.writeAll("interrupted\n");
            },
            .verifying => if (event.reused) {
                try d.endLine(w);
                try w.print("{s}: existing file verified, reused\n", .{name});
            } else if (event.file_completed == 0) {
                try w.print("{s}: checking for an existing file\n", .{name});
            } else {
                try d.endLine(w);
                try w.print("{s}: verifying SHA-256 and publishing\n", .{name});
            },
            .downloading => {
                if (phase_changed or event.file_completed == 0) {
                    d.download_started = now;
                    d.download_base = event.file_completed;
                    if (event.file_count > 1) try w.print("file {d} of {d}\n", .{ event.file_index + 1, event.file_count });
                    if (!d.tty) try w.print("{s}: downloading {d:.1} MB\n", .{ name, mb(event.file_total) });
                }
                if (!d.tty) return;
                if (d.last_line) |last| if (event.file_completed < event.file_total and last.durationTo(now).toMilliseconds() < 200) return;
                d.last_line = now;
                const elapsed_ms = d.download_started.?.durationTo(now).toMilliseconds();
                const moved = event.file_completed - d.download_base;
                const rate_bps = if (elapsed_ms > 0) @as(f64, @floatFromInt(moved)) * 1000 / @as(f64, @floatFromInt(elapsed_ms)) else 0;
                const remaining = event.file_total - event.file_completed;
                const eta_s: u64 = if (rate_bps > 0) @intFromFloat(@as(f64, @floatFromInt(remaining)) / rate_bps) else 0;
                const percent = if (event.file_total == 0) 0 else 100 * event.file_completed / event.file_total;
                try w.print("\r{s}: {d:>3}% {d:.1}/{d:.1} MB {d:.1} MB/s eta {d}s   ", .{ name, percent, mb(event.file_completed), mb(event.file_total), rate_bps / 1e6, eta_s });
                d.open_line = true;
            },
            .complete => {
                try d.endLine(w);
                const total_ms = d.started.durationTo(now).toMilliseconds();
                try w.print("done: {d:.1} MB in {d:.1} s ({d:.1} MB/s overall, verified reuse included)\n", .{ mb(event.completed_bytes), @as(f64, @floatFromInt(total_ms)) / 1000, if (total_ms > 0) mb(event.completed_bytes) * 1000 / @as(f64, @floatFromInt(total_ms)) else 0 });
            },
        }
    }

    fn endLine(d: *Display, w: *std.Io.Writer) !void {
        if (d.open_line) try w.writeAll("\n");
        d.open_line = false;
    }

    fn mb(bytes: u64) f64 {
        return @as(f64, @floatFromInt(bytes)) / 1e6;
    }
};

// ---- ls --------------------------------------------------------------------

pub const Listed = struct {
    /// Relative to the models directory: `<owner>/<repo>/<file>`.
    path: []const u8,
    size: u64,
    sidecar: ?Sidecar,
    /// The registry entry that locates this file, when one does.
    registered: ?RegisteredAs = null,
};

/// How a file is named in `nuclis.json`: the entry and, when the entry
/// forces one, its prompt profile.
pub const RegisteredAs = struct { name: []const u8, profile: ?config.Profile };

/// A registry entry whose file is not on disk.
pub const MissingEntry = struct { name: []const u8, path: []const u8 };

pub const CompanionRow = struct {
    role: Role,
    status: catalog.Status,
    path: []const u8,
    size: u64,
    loaded_by: []const u8,
};

pub const CatalogRow = struct {
    name: []const u8,
    status: catalog.Status,
    path: []const u8,
    size: u64,
    architecture: []const u8,
    quantization: []const u8,
    revision: []const u8,
    sha256: []const u8,
    companions: []const CompanionRow,
    /// A registry entry under another name than the catalogue's, or one
    /// forcing a profile (the catalogue's own name is not repeated).
    registered: ?RegisteredAs = null,
};

pub const Listing = struct {
    schema_version: u32 = 3,
    models_dir: []const u8,
    /// Every catalogue entry with its local status, companions beneath.
    catalog: []const CatalogRow,
    /// GGUF files in the layout the catalogue does not name.
    other: []const Listed,
    /// GGUF files found above the `<owner>/<repo>/` level; they are not
    /// listed because nothing can say which repository they came from.
    outside_layout: usize,
    /// Registry entries whose file is absent.
    missing: []const MissingEntry = &.{},

    pub fn render(self: Listing, out: *std.Io.Writer, json: bool, sty: style.Style) !void {
        if (json) {
            try std.json.Stringify.value(self, .{ .whitespace = .indent_2 }, out);
            return out.writeByte('\n');
        }
        const off = sty.off();
        const path = sty.on(.code);
        const number = sty.on(.number);
        const hash = sty.on(.comment);
        try out.print("{s}models directory:{s} {s}{s}{s}\n\n{s}catalogue{s} {s}(supported artifacts; `nuclis model pull <name> [--all]`){s}\n", .{ sty.on(.label), off, path, self.models_dir, off, sty.on(.header), off, sty.on(.dim), off });
        // One grid for the catalogue: the name column fits the widest name
        // (a companion's role sits two cells in), the status column the
        // widest status word, the path column the widest path of a main
        // file or a companion; sizes are right-aligned after it.
        var name_w: usize = catalog.name_width;
        var path_w: usize = 0;
        for (self.catalog) |row| {
            name_w = @max(name_w, row.name.len);
            path_w = @max(path_w, row.path.len);
            for (row.companions) |c| path_w = @max(path_w, c.path.len);
        }
        const status_w: usize = 10;
        for (self.catalog) |row| {
            try out.writeAll("  ");
            try column(out, sty.on(.keyword), row.name, off, name_w);
            try column(out, sty.on(statusStyle(row.status)), @tagName(row.status), off, status_w);
            try column(out, path, row.path, off, 0);
            try out.writeByte('\n');
            try out.splatByteAll(' ', 2 + name_w + 1 + status_w + 1);
            try out.print("{s}{d:>13}{s} bytes  {s} {s}  commit {s}{s}{s}  sha256 {s}{s}{s}\n", .{ number, row.size, off, row.architecture, row.quantization, hash, row.revision[0..12], off, hash, row.sha256, off });
            if (row.registered) |r| {
                try out.splatByteAll(' ', 2 + name_w + 1 + status_w + 1);
                try renderRegistered(out, r, sty);
            }
            for (row.companions) |c| {
                try out.writeAll("    ");
                try column(out, number, @tagName(c.role), off, name_w - 2);
                try column(out, sty.on(statusStyle(c.status)), @tagName(c.status), off, status_w);
                try column(out, path, c.path, off, path_w);
                try out.print(" {s}{d:>13}{s} bytes  {s}not loaded yet: {s}{s}\n", .{ number, c.size, off, sty.on(.dim), c.loaded_by, off });
            }
        }
        if (self.other.len > 0) {
            try out.print("\n{s}other GGUF files in the layout{s} {s}(not in the catalogue; runnable only if their architecture has an adapter){s}\n", .{ sty.on(.header), off, sty.on(.dim), off });
            var width: usize = 0;
            for (self.other) |f| width = @max(width, f.path.len);
            for (self.other) |f| {
                try out.print("  {s}{s}{s}", .{ path, f.path, off });
                try out.splatByteAll(' ', width - f.path.len + 2);
                try out.print("{s}{d:>13}{s}  ", .{ number, f.size, off });
                if (f.sidecar) |s| {
                    try out.print("{s}{s:<7}{s}  {s}{s}  {s}{s}", .{ number, @tagName(s.role), off, hash, s.revision[0..12], s.sha256, off });
                    if (s.size != f.size) try out.print("  {s}(size differs from the sidecar){s}", .{ sty.on(.warning), off });
                } else try out.print("{s}(no sidecar: not verified by nuclis){s}", .{ sty.on(.warning), off });
                try out.writeByte('\n');
                if (f.registered) |r| {
                    try out.writeAll("    ");
                    try renderRegistered(out, r, sty);
                }
            }
        }
        if (self.missing.len > 0) {
            try out.print("\n{s}registry entries without a file{s} {s}(`nuclis model pull <name>` fetches an entry with repo and file){s}\n", .{ sty.on(.header), off, sty.on(.dim), off });
            var width: usize = 0;
            for (self.missing) |m| width = @max(width, m.name.len);
            for (self.missing) |m| {
                try out.writeAll("  ");
                try column(out, sty.on(.keyword), m.name, off, width);
                try out.print("{s}{s}{s}\n", .{ path, m.path, off });
            }
        }
        if (self.outside_layout > 0) try out.print("\n{s}{d} GGUF file(s) outside the <owner>/<repo>/ layout are not listed{s}\n", .{ sty.on(.warning), self.outside_layout, off });
    }
};

/// One grid cell: styled text padded to `width` (plus one separating
/// space); a width of 0 pads nothing, for the last cell of a row.
fn column(out: *std.Io.Writer, on: []const u8, text: []const u8, off: []const u8, width: usize) !void {
    try out.print("{s}{s}{s}", .{ on, text, off });
    if (width > 0) try out.splatByteAll(' ', (width -| text.len) + 1);
}

fn renderRegistered(out: *std.Io.Writer, r: RegisteredAs, sty: style.Style) !void {
    try out.print("{s}registered as{s} {s}{s}{s}", .{ sty.on(.dim), sty.off(), sty.on(.keyword), r.name, sty.off() });
    if (r.profile) |p| try out.print(" {s}(profile {s} forced){s}", .{ sty.on(.dim), @tagName(p), sty.off() });
    try out.writeByte('\n');
}

/// Status words by how good the news is.
pub fn statusStyle(status: catalog.Status) style.Kind {
    return switch (status) {
        .present => .success,
        .absent => .dim,
        .mismatch => .error_text,
        .unverified => .warning,
    };
}

/// The catalogue's status from sidecars, then a walk of `<root>/models` in
/// the layout only: owner directories, repository directories, then files
/// (with subdirectories such as `MTP/`). Nothing is created; a missing
/// directory lists the catalogue as absent.
pub fn list(arena: Allocator, io: std.Io, root: []const u8, registry: config.Models) !Listing {
    const models = try modelsDir(arena, root);
    const rows = try arena.alloc(CatalogRow, catalog.entries.len);
    for (&catalog.entries, rows) |*e, *row| {
        const path = try catalog.localPath(arena, models, e, e.file);
        const companions = try arena.alloc(CompanionRow, e.companions.len);
        for (e.companions, companions) |c, *cr| {
            const cpath = try catalog.localPath(arena, models, e, c.file);
            cr.* = .{ .role = c.role, .status = try catalog.status(arena, io, cpath, c.sha256, c.size), .path = cpath[models.len + 1 ..], .size = c.size, .loaded_by = c.loaded_by };
        }
        row.* = .{ .name = e.name, .status = try catalog.status(arena, io, path, e.sha256, e.size), .path = path[models.len + 1 ..], .size = e.size, .architecture = e.architecture, .quantization = e.quantization, .revision = e.revision, .sha256 = e.sha256, .companions = companions };
    }
    var files: std.ArrayList(Listed) = .empty;
    var outside: usize = 0;
    if (std.Io.Dir.cwd().openDir(io, models, .{ .iterate = true })) |dir| {
        var d = dir;
        defer d.close(io);
        try collect(arena, io, d, "", 0, &files, &outside);
    } else |err| switch (err) {
        error.FileNotFound => {},
        else => return err,
    }
    // Files the catalogue names are reported under their entry, not twice.
    var other: std.ArrayList(Listed) = .empty;
    scan: for (files.items) |f| {
        for (rows) |row| {
            if (std.mem.eql(u8, f.path, row.path)) continue :scan;
            for (row.companions) |c| if (std.mem.eql(u8, f.path, c.path)) continue :scan;
        }
        try other.append(arena, f);
    }
    std.mem.sort(Listed, other.items, {}, struct {
        fn less(_: void, a: Listed, b: Listed) bool {
            return std.mem.lessThan(u8, a.path, b.path);
        }
    }.less);
    // The registry's view: each entry's location relative to the models
    // directory names a row, or the entry is missing its file.
    var missing: std.ArrayList(MissingEntry) = .empty;
    for (registry.entries) |named| {
        const e = named.entry;
        const as: RegisteredAs = .{ .name = named.name, .profile = e.profile };
        const relative: ?[]const u8 = if (e.path) |p| blk: {
            if (!std.fs.path.isAbsolute(p)) break :blk p;
            if (std.mem.startsWith(u8, p, models) and p.len > models.len + 1 and p[models.len] == '/') break :blk p[models.len + 1 ..];
            // Outside the layout: only its existence can be checked.
            std.Io.Dir.cwd().access(io, p, .{}) catch try missing.append(arena, .{ .name = named.name, .path = p });
            continue;
        } else if (e.repo != null and e.file != null) try std.fs.path.join(arena, &.{ e.repo.?, e.file.? }) else null;
        const location = relative orelse continue;
        var found = false;
        for (rows) |*row| if (std.mem.eql(u8, row.path, location)) {
            found = row.status != .absent;
            if (found and (!std.mem.eql(u8, row.name, named.name) or e.profile != null)) row.registered = as;
        };
        for (other.items) |*f| if (std.mem.eql(u8, f.path, location)) {
            f.registered = as;
            found = true;
        };
        if (!found) try missing.append(arena, .{ .name = named.name, .path = location });
    }
    return .{ .models_dir = models, .catalog = rows, .other = other.items, .outside_layout = outside, .missing = missing.items };
}

fn collect(arena: Allocator, io: std.Io, dir: std.Io.Dir, prefix: []const u8, depth: u8, files: *std.ArrayList(Listed), outside: *usize) !void {
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        const is_gguf = std.mem.endsWith(u8, entry.name, ".gguf");
        switch (entry.kind) {
            .directory => {
                if (depth >= 2 + max_depth_below_repo) continue;
                var child = dir.openDir(io, entry.name, .{ .iterate = true }) catch continue;
                defer child.close(io);
                const child_prefix = try std.mem.concat(arena, u8, &.{ prefix, entry.name, "/" });
                try collect(arena, io, child, child_prefix, depth + 1, files, outside);
            },
            .file => {
                if (!is_gguf) continue;
                if (depth < 2) {
                    outside.* += 1;
                    continue;
                }
                if (files.items.len == max_listed_files) return error.TooManyFiles;
                const stat = try dir.statFile(io, entry.name, .{});
                const sidecar_name = try sidecarPath(arena, entry.name);
                const sidecar = readSidecar(arena, io, dir, sidecar_name) catch |err| switch (err) {
                    error.InvalidSidecar => null,
                    else => return err,
                };
                try files.append(arena, .{ .path = try std.mem.concat(arena, u8, &.{ prefix, entry.name }), .size = stat.size, .sidecar = sidecar });
            },
            else => {},
        }
    }
}

pub fn ls(gpa: Allocator, io: std.Io, root: []const u8, registry: config.Models, json: bool, out: *std.Io.Writer, sty: style.Style) !void {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const listing = try list(arena_state.allocator(), io, root, registry);
    try listing.render(out, json, sty);
}

// ---- tests -----------------------------------------------------------------

// ---- inspect ---------------------------------------------------------------

/// The most of a file's head `model inspect` fetches: the parser's own
/// directory bound, so a directory the parser would refuse is never
/// downloaded either.
pub const max_inspect_bytes: u64 = (inference.gguf.Limits{}).directory_bytes;

pub const Verdict = struct {
    status: enum { supported, runnable, not_runnable },
    /// The catalogue entry whose pinned digest the Hub's equals, and the
    /// file's role in it, when any (a companion is never runnable by the
    /// text engine; the verdict still names what it is).
    catalog: ?[]const u8 = null,
    role: ?Role = null,
    /// One sentence, the same in both forms.
    reason: []const u8,
    /// The first offending tensor and its encoding when they are the reason.
    tensor: ?[]const u8 = null,
    encoding: ?[]const u8 = null,
};

pub const InspectReport = struct {
    schema_version: u32 = 1,
    /// The catalogue name when one was given.
    name: ?[]const u8,
    repo: []const u8,
    file: []const u8,
    requested_revision: []const u8,
    revision: []const u8,
    size: u64,
    sha256: []const u8,
    /// How much of the file's head was fetched to parse the directory, in
    /// how many range requests; no weights are downloaded.
    bytes_read: u64,
    requests: usize,
    /// Null when the directory itself was rejected (the verdict says why).
    directory: ?inspection.Snapshot,
    verdict: Verdict,
};

/// What identifies the file being inspected, as the Hub listed it.
const Identity = struct {
    name: ?[]const u8,
    repo: []const u8,
    file: []const u8,
    requested_revision: []const u8,
    revision: []const u8,
    size: u64,
    sha256: []const u8,
};

/// `model inspect`: resolves the request like `pull` (a catalogue name
/// pins repository, file, and commit; a repository id resolves them on the
/// Hub and needs `--file` when it holds several GGUFs), then parses the
/// directory from the head of the file and prints what `inspect` prints
/// for a local file, ending with the verdict.
pub fn inspect(gpa: Allocator, io: std.Io, environ: *const std.process.Environ.Map, options: PullOptions, json: bool, out: *std.Io.Writer, sty: style.Style, diag: *config.Diagnostic) !void {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    if (options.repo.len == 0) return error.MissingRepository;
    const client = hf.Client.init(gpa, io, environ);
    var request: hf.Request = .{ .repo_id = options.repo, .revision = options.revision orelse "main", .filename = options.file };
    const entry = catalog.find(options.repo);
    if (entry) |e| {
        if (options.file != null or options.revision != null) {
            diag.set("{s} is a catalogue name and pins its file and commit; inspect by repository id ({s}) to choose another", .{ e.name, e.repo });
            return error.ConflictingOptions;
        }
        request = .{ .repo_id = e.repo, .revision = e.revision, .filename = e.file };
    }
    var remote = client.list(request) catch |err| return hubFailure(err, options.repo, diag);
    defer remote.deinit();
    const selected = (hf.select(arena, remote.files, request.filename) catch |err| return hubFailure(err, options.repo, diag)) orelse {
        try renderSelection(out, .{ .repo = options.repo, .revision = &remote.revision, .selection_required = try choices(arena, remote.files) }, json, sty);
        try out.flush();
        diag.set("{s} has {d} GGUF files; pass --file <name>", .{ options.repo, remote.files.len });
        return error.SelectionRequired;
    };
    // A split file's directory is in its first shard.
    const file = selected[0];
    const revision = try arena.dupe(u8, &remote.revision);
    request.revision = revision;
    request.filename = file.name;
    var reader: HubRange = .{ .client = client, .request = request };
    const report = try inspectDirectory(arena, gpa, &reader, .{
        .name = if (entry) |e| e.name else null,
        .repo = request.repo_id,
        .file = try arena.dupe(u8, file.name),
        .requested_revision = options.revision orelse (if (entry) |e| e.revision else "main"),
        .revision = revision,
        .size = file.size,
        .sha256 = try arena.dupe(u8, &std.fmt.bytesToHex(file.sha256, .lower)),
    }, diag);
    try renderInspect(out, report, json, sty);
}

/// The production range reader: one `readRange` per window, bytes owned by
/// the client's allocator (the `gpa` `inspectDirectory` frees with).
const HubRange = struct {
    client: hf.Client,
    request: hf.Request,
    fn read(self: *HubRange, start: u64, count: usize) ![]u8 {
        return self.client.readRange(self.request, start, count);
    }
};

/// Fetches the head of the file in `max_range_bytes` steps until the
/// directory parses or is rejected, then judges it. `reader` has
/// `fn read(self, start: u64, count: usize) ![]u8` returning bytes owned
/// by `gpa` (possibly fewer than `count`, never none); production wraps
/// `readRange`, the tests a byte slice. Every string of the report lives
/// in `arena`.
fn inspectDirectory(arena: Allocator, gpa: Allocator, reader: anytype, id: Identity, diag: *config.Diagnostic) !InspectReport {
    var head: std.ArrayList(u8) = .empty;
    defer head.deinit(gpa);
    var requests: usize = 0;
    const limit: u64 = @min(id.size, max_inspect_bytes);
    var doc = while (true) {
        var fixed: std.Io.Reader = .fixed(head.items);
        var rejection: inference.gguf.Rejection = .{};
        if (inference.gguf.parseDiagnosed(gpa, &fixed, id.size, .{}, &rejection)) |parsed| break parsed else |err| switch (err) {
            error.EndOfStream => {
                if (head.items.len >= limit) {
                    diag.set("{s}: the directory does not end within the first {d} bytes; not a GGUF nuclis reads", .{ id.file, limit });
                    return error.DirectoryTooLarge;
                }
                const count: usize = @intCast(@min(hf.max_range_bytes, limit - head.items.len));
                const chunk = try reader.read(head.items.len, count);
                defer gpa.free(chunk);
                if (chunk.len == 0) return error.UnexpectedEndOfFile;
                try head.appendSlice(gpa, chunk);
                requests += 1;
            },
            error.UnsupportedTensorType, error.InvalidShape, error.Overflow => {
                const tensor = rejection.tensor() orelse {
                    diag.set("{s}: not a GGUF nuclis reads ({s})", .{ id.file, @errorName(err) });
                    return error.InvalidGguf;
                };
                const encoding = try encodingLabel(arena, rejection.encoding_id.?);
                const reason = if (err == error.UnsupportedTensorType)
                    try std.fmt.allocPrint(arena, "tensor {s} uses encoding {s}, which nuclis does not store", .{ tensor, encoding })
                else
                    try std.fmt.allocPrint(arena, "tensor {s} has a shape encoding {s} cannot store", .{ tensor, encoding });
                return finish(id, head.items.len, requests, null, .{ .status = .not_runnable, .reason = reason, .tensor = try arena.dupe(u8, tensor), .encoding = encoding });
            },
            else => {
                diag.set("{s}: not a GGUF nuclis reads ({s})", .{ id.file, @errorName(err) });
                return error.InvalidGguf;
            },
        }
    };
    defer doc.deinit();
    // The snapshot borrows the document's strings; the report outlives it.
    var snapshot = try inspection.snapshot(doc, try std.fmt.allocPrint(arena, "{s}/{s}", .{ id.repo, id.file }), try arena.alloc(inspection.EncodingCount, 32));
    if (snapshot.architecture) |a| snapshot.architecture = try arena.dupe(u8, a);
    if (snapshot.name) |n| snapshot.name = try arena.dupe(u8, n);
    return finish(id, head.items.len, requests, snapshot, try judge(arena, gpa, &doc, id));
}

fn finish(id: Identity, bytes_read: u64, requests: usize, directory: ?inspection.Snapshot, verdict: Verdict) InspectReport {
    return .{ .name = id.name, .repo = id.repo, .file = id.file, .requested_revision = id.requested_revision, .revision = id.revision, .size = id.size, .sha256 = id.sha256, .bytes_read = bytes_read, .requests = requests, .directory = directory, .verdict = verdict };
}

/// "Q4_K (id 12)", or "id 999" for a layout nuclis does not know.
fn encodingLabel(arena: Allocator, encoding_id: u32) ![]const u8 {
    if (inference.encoding.layout(encoding_id)) |layout| return std.fmt.allocPrint(arena, "{s} (id {d})", .{ layout.name, encoding_id });
    return std.fmt.allocPrint(arena, "id {d}", .{encoding_id});
}

/// The verdict on a parsed directory. Runnable needs an adapter for the
/// architecture, every tensor in its executable set, and the adapter's
/// binding; supported adds the catalogue pinning the Hub's digest.
fn judge(arena: Allocator, gpa: Allocator, doc: *const inference.gguf.Document, id: Identity) !Verdict {
    const pinned = catalog.findFile(id.repo, id.file);
    const in_catalog = pinned != null and std.mem.eql(u8, pinned.?.sha256, id.sha256);
    const architecture = doc.string("general.architecture") orelse "";
    const adapter = inference.models.adapterFor(architecture) orelse {
        if (in_catalog and pinned.?.role != .main) return .{
            .status = .not_runnable,
            .catalog = pinned.?.entry.name,
            .role = pinned.?.role,
            .reason = try std.fmt.allocPrint(arena, "the {s} companion of catalogue entry {s} (loaded by {s}, not by the text engine); architecture \"{s}\" has no adapter", .{ @tagName(pinned.?.role), pinned.?.entry.name, pinned.?.loaded_by.?, architecture }),
        };
        return .{
            .status = .not_runnable,
            .reason = if (architecture.len == 0) "the file declares no general.architecture" else try std.fmt.allocPrint(arena, "no adapter for architecture \"{s}\"", .{architecture}),
        };
    };
    for (doc.tensors) |tensor| if (!inference.models.registry.executableEncoding(adapter, tensor.encoding_id)) {
        const encoding = try encodingLabel(arena, tensor.encoding_id);
        return .{
            .status = .not_runnable,
            .reason = try std.fmt.allocPrint(arena, "tensor {s} uses encoding {s}, outside the {s} adapter's executable set", .{ tensor.name, encoding, @tagName(adapter) }),
            .tensor = try arena.dupe(u8, tensor.name),
            .encoding = encoding,
        };
    };
    _ = inference.models.registry.validate(adapter, gpa, doc) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => return .{ .status = .not_runnable, .reason = try std.fmt.allocPrint(arena, "the {s} adapter rejects the file: {s}", .{ @tagName(adapter), @errorName(err) }) },
    };
    if (in_catalog and pinned.?.role == .main) return .{
        .status = .supported,
        .catalog = pinned.?.entry.name,
        .role = .main,
        .reason = try std.fmt.allocPrint(arena, "catalogue entry {s}: the Hub's digest at this commit equals the pinned one, and the {s} adapter binds it", .{ pinned.?.entry.name, @tagName(adapter) }),
    };
    return .{
        .status = .runnable,
        .reason = if (pinned != null)
            try std.fmt.allocPrint(arena, "the {s} adapter binds it; the catalogue pins another digest for this file (entry {s})", .{ @tagName(adapter), pinned.?.entry.name })
        else
            try std.fmt.allocPrint(arena, "the {s} adapter binds it; not in the catalogue", .{@tagName(adapter)}),
    };
}

fn renderInspect(out: *std.Io.Writer, report: InspectReport, json: bool, sty: style.Style) !void {
    if (json) {
        try std.json.Stringify.value(report, .{ .whitespace = .indent_2 }, out);
        return out.writeByte('\n');
    }
    try renderHeader(out, sty, report.name, report.repo, if (std.mem.eql(u8, report.requested_revision, report.revision)) null else report.requested_revision, report.revision);
    try out.print("{s}{s}{s}: {s}{d}{s} bytes, sha256 {s}{s}{s}\n", .{ sty.on(.code), report.file, sty.off(), sty.on(.number), report.size, sty.off(), sty.on(.comment), report.sha256, sty.off() });
    if (report.directory) |directory| {
        try directory.renderText(out, sty);
        try out.writeByte('\n');
    }
    const word: []const u8, const kind: style.Kind = switch (report.verdict.status) {
        .supported => .{ "supported", .success },
        .runnable => .{ "runnable", .header },
        .not_runnable => .{ "not runnable", .error_text },
    };
    try out.print("{s}Verdict:{s} {s}{s}{s}{s}: {s}\n", .{ sty.on(.bold), sty.off(), sty.on(.bold), sty.on(kind), word, sty.off(), report.verdict.reason });
    try out.print("{s}Directory read: {d} bytes in {d} request(s); no weights downloaded.{s}\n", .{ sty.on(.dim), report.bytes_read, report.requests, sty.off() });
}

test "rfc3339 renders UTC from Unix seconds" {
    var buffer: [20]u8 = undefined;
    try std.testing.expectEqualStrings("1970-01-01T00:00:00Z", rfc3339(&buffer, 0));
    try std.testing.expectEqualStrings("2026-09-11T08:04:05Z", rfc3339(&buffer, 1789113845));
    try std.testing.expectEqualStrings("2000-02-29T23:59:59Z", rfc3339(&buffer, 951868799));
    try std.testing.expectEqualStrings("1970-01-01T00:00:00Z", rfc3339(&buffer, -5));
}

test "the header wins over the flag, a contradiction is an error, main is the default" {
    try std.testing.expectEqual(Role.main, try resolveRole(null, null));
    try std.testing.expectEqual(Role.mtp, try resolveRole(null, .mtp));
    try std.testing.expectEqual(Role.imatrix, try resolveRole(.imatrix, null));
    try std.testing.expectEqual(Role.mmproj, try resolveRole(.mmproj, .mmproj));
    try std.testing.expectError(error.RoleMismatch, resolveRole(.imatrix, .main));
}

test "sidecar round trip, absence, and rejection of what nuclis did not write" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena_state = std.heap.ArenaAllocator.init(alloc);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const name = try sidecarPath(arena, "model.gguf");
    try std.testing.expectEqualStrings("model.gguf.nuclis.json", name);
    try std.testing.expect(try readSidecar(arena, io, tmp.dir, name) == null);
    const written: Sidecar = .{
        .repo = "unsloth/Qwen3.8-27B-GGUF",
        .file = "model.gguf",
        .revision = "0123456789abcdef0123456789abcdef01234567",
        .sha256 = "322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482",
        .size = 16,
        .role = .mtp,
        .downloaded_at = "2026-09-11T00:00:00Z",
        .nuclis_version = version,
    };
    try writeSidecar(arena, io, tmp.dir, name, written);
    const read = (try readSidecar(arena, io, tmp.dir, name)).?;
    try std.testing.expectEqualStrings(written.repo, read.repo);
    try std.testing.expectEqualStrings(written.revision, read.revision);
    try std.testing.expectEqualStrings(written.sha256, read.sha256);
    try std.testing.expectEqual(written.size, read.size);
    try std.testing.expectEqual(Role.mtp, read.role);
    try std.testing.expectEqualStrings(written.downloaded_at, read.downloaded_at);
    // Rewriting replaces the record (a second pull records its own commit).
    var again = written;
    again.revision = "89abcdef0123456789abcdef0123456789abcdef";
    try writeSidecar(arena, io, tmp.dir, name, again);
    try std.testing.expectEqualStrings(again.revision, (try readSidecar(arena, io, tmp.dir, name)).?.revision);
    // Not ours: a short commit, an unknown key, a truncated file.
    try tmp.dir.writeFile(io, .{ .sub_path = name, .data = "{\"schema_version\":1,\"repo\":\"a/b\",\"file\":\"m\",\"revision\":\"abc\",\"sha256\":\"\",\"size\":1,\"role\":\"main\",\"downloaded_at\":\"\",\"nuclis_version\":\"\"}" });
    try std.testing.expectError(error.InvalidSidecar, readSidecar(arena, io, tmp.dir, name));
    try tmp.dir.writeFile(io, .{ .sub_path = name, .data = "{\"schema_version\":1,\"extra\":true}" });
    try std.testing.expectError(error.InvalidSidecar, readSidecar(arena, io, tmp.dir, name));
    try tmp.dir.writeFile(io, .{ .sub_path = name, .data = "{\"schema_version\":1," });
    try std.testing.expectError(error.InvalidSidecar, readSidecar(arena, io, tmp.dir, name));
}

test "selection and pull reports render the same values in both forms" {
    const alloc = std.testing.allocator;
    var out: std.Io.Writer.Allocating = .init(alloc);
    defer out.deinit();
    const selection: SelectionReport = .{ .repo = "a/b", .revision = "0123456789abcdef0123456789abcdef01234567", .selection_required = &.{ .{ .name = "Q4.gguf", .size = 4 }, .{ .name = "MTP/mtp.gguf", .size = 2 } } };
    try renderSelection(&out.writer, selection, false, .none);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "choose one with --file") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "  MTP/mtp.gguf\t2\n") != null);
    out.clearRetainingCapacity();
    try renderSelection(&out.writer, selection, true, .none);
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, out.written(), .{});
    defer parsed.deinit();
    try std.testing.expectEqual(@as(usize, 2), parsed.value.object.get("selection_required").?.array.items.len);
    try std.testing.expectEqualStrings(selection.revision, parsed.value.object.get("revision").?.string);
    out.clearRetainingCapacity();
    const pulled: PullReport = .{ .name = null, .repo = "a/b", .requested_revision = "main", .revision = selection.revision, .files = &.{.{ .path = "/m/a/b/Q4.gguf", .size = 4, .sha256 = "ab", .transport = "verified_local", .role = .main, .sidecar = "/m/a/b/Q4.gguf.nuclis.json" }} };
    try renderPull(&out.writer, pulled, false, .none);
    try std.testing.expect(std.mem.startsWith(u8, out.written(), "verified /m/a/b/Q4.gguf\n  4 bytes, sha256 ab, role main; provenance in Q4.gguf.nuclis.json\n"));
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "registered") == null);
    out.clearRetainingCapacity();
    var named = pulled;
    named.registered = .{ .name = "q", .config = "/r/nuclis.json" };
    try renderPull(&out.writer, named, false, .none);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "registered as q in /r/nuclis.json (`nuclis config set engine.model q` makes it the default)\n") != null);
    out.clearRetainingCapacity();
    try renderPull(&out.writer, pulled, true, .none);
    const parsed_pull = try std.json.parseFromSlice(std.json.Value, alloc, out.written(), .{});
    defer parsed_pull.deinit();
    try std.testing.expectEqualStrings("main", parsed_pull.value.object.get("files").?.array.items[0].object.get("role").?.string);
}

test "ls reports the catalogue from sidecars, then the other files in the layout" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", alloc);
    defer alloc.free(root);
    var arena_state = std.heap.ArenaAllocator.init(alloc);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    // Absent directory: the catalogue is absent, nothing else, nothing created.
    const empty = try list(arena, io, root, .{});
    try std.testing.expectEqual(catalog.entries.len, empty.catalog.len);
    try std.testing.expectEqual(catalog.Status.absent, empty.catalog[0].status);
    try std.testing.expectEqual(@as(usize, 0), empty.other.len);
    try std.testing.expectError(error.FileNotFound, tmp.dir.access(io, "models", .{}));
    const qwen = catalog.find("qwen3.8-27b").?;
    try tmp.dir.createDirPath(io, "models/unsloth/Qwen3.8-27B-GGUF/MTP");
    try tmp.dir.createDirPath(io, "models/unsloth/Repo-GGUF/MTP");
    try tmp.dir.createDirPath(io, "models/qwen");
    // The pinned main file with a matching sidecar, the MTP head with a wrong one.
    try tmp.dir.writeFile(io, .{ .sub_path = "models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf", .data = "GGUF" });
    try writeSidecar(arena, io, tmp.dir, "models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf.nuclis.json", .{ .repo = qwen.repo, .file = qwen.file, .revision = qwen.revision, .sha256 = qwen.sha256, .size = qwen.size, .role = .main, .downloaded_at = "2026-09-11T00:00:00Z", .nuclis_version = version });
    try tmp.dir.writeFile(io, .{ .sub_path = "models/unsloth/Qwen3.8-27B-GGUF/MTP/mtp-Qwen3.8-27B-Q4_0.gguf", .data = "GGUF" });
    try writeSidecar(arena, io, tmp.dir, "models/unsloth/Qwen3.8-27B-GGUF/MTP/mtp-Qwen3.8-27B-Q4_0.gguf.nuclis.json", .{ .repo = qwen.repo, .file = "MTP/mtp-Qwen3.8-27B-Q4_0.gguf", .revision = qwen.revision, .sha256 = "0000000000000000000000000000000000000000000000000000000000000000", .size = 4, .role = .mtp, .downloaded_at = "2026-09-11T00:00:00Z", .nuclis_version = version });
    // Files outside the catalogue: one with a sidecar, one without, two above the layout.
    try tmp.dir.writeFile(io, .{ .sub_path = "models/unsloth/Repo-GGUF/Repo-Q4.gguf", .data = "GGUFxxxx" });
    try tmp.dir.writeFile(io, .{ .sub_path = "models/unsloth/Repo-GGUF/README.md", .data = "not listed" });
    try tmp.dir.writeFile(io, .{ .sub_path = "models/unsloth/Repo-GGUF/MTP/mtp-Repo.gguf", .data = "GGUF" });
    try tmp.dir.writeFile(io, .{ .sub_path = "models/qwen/old-layout.gguf", .data = "GGUF" });
    try tmp.dir.writeFile(io, .{ .sub_path = "models/stray.gguf", .data = "GGUF" });
    try writeSidecar(arena, io, tmp.dir, "models/unsloth/Repo-GGUF/Repo-Q4.gguf.nuclis.json", .{
        .repo = "unsloth/Repo-GGUF",
        .file = "Repo-Q4.gguf",
        .revision = "0123456789abcdef0123456789abcdef01234567",
        .sha256 = "322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482",
        .size = 8,
        .role = .main,
        .downloaded_at = "2026-09-11T00:00:00Z",
        .nuclis_version = version,
    });
    // The registry: an entry naming the other repository's file with a
    // forced profile, the catalogue's own name (not repeated), a second
    // name for the catalogue file, one whose file is absent, and a path.
    const entries = [_]config.NamedModel{
        .{ .name = "repo", .entry = .{ .repo = "unsloth/Repo-GGUF", .file = "Repo-Q4.gguf", .profile = .gemma4 } },
        .{ .name = "qwen3.8-27b", .entry = .{ .repo = qwen.repo, .file = qwen.file } },
        .{ .name = "big", .entry = .{ .repo = qwen.repo, .file = qwen.file } },
        .{ .name = "gone", .entry = .{ .repo = "unsloth/Gone-GGUF", .file = "gone.gguf" } },
        .{ .name = "local", .entry = .{ .path = "unsloth/Repo-GGUF/MTP/mtp-Repo.gguf" } },
    };
    const listing = try list(arena, io, root, .{ .entries = &entries });
    try std.testing.expectEqualStrings("big", listing.catalog[0].registered.?.name);
    try std.testing.expect(listing.catalog[0].registered.?.profile == null);
    try std.testing.expectEqualStrings("local", listing.other[0].registered.?.name);
    try std.testing.expectEqualStrings("repo", listing.other[1].registered.?.name);
    try std.testing.expectEqual(.gemma4, listing.other[1].registered.?.profile.?);
    try std.testing.expectEqual(@as(usize, 1), listing.missing.len);
    try std.testing.expectEqualStrings("gone", listing.missing[0].name);
    try std.testing.expectEqualStrings("unsloth/Gone-GGUF/gone.gguf", listing.missing[0].path);
    const row = listing.catalog[0];
    try std.testing.expectEqualStrings("qwen3.8-27b", row.name);
    try std.testing.expectEqual(catalog.Status.present, row.status);
    try std.testing.expectEqualStrings("unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf", row.path);
    try std.testing.expectEqual(catalog.Status.absent, row.companions[0].status);
    try std.testing.expectEqual(Role.mtp, row.companions[1].role);
    try std.testing.expectEqual(catalog.Status.mismatch, row.companions[1].status);
    try std.testing.expectEqual(@as(usize, 2), listing.other.len);
    try std.testing.expectEqual(@as(usize, 2), listing.outside_layout);
    try std.testing.expectEqualStrings("unsloth/Repo-GGUF/MTP/mtp-Repo.gguf", listing.other[0].path);
    try std.testing.expect(listing.other[0].sidecar == null);
    try std.testing.expectEqualStrings("unsloth/Repo-GGUF/Repo-Q4.gguf", listing.other[1].path);
    try std.testing.expectEqual(@as(u64, 8), listing.other[1].size);
    try std.testing.expectEqualStrings("0123456789abcdef0123456789abcdef01234567", listing.other[1].sidecar.?.revision);
    var out: std.Io.Writer.Allocating = .init(alloc);
    defer out.deinit();
    try listing.render(&out.writer, false, .none);
    // The column width comes from the catalogue's widest name, so the
    // assertion is on the two parts rather than the spacing between them.
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "qwen3.8-27b ") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), " present") != null);
    // The role, status, and path columns line up under the main row's:
    // the role cell is the name column less its two-cell indent, and the
    // detail rows start at the path column.
    const indent = try alloc.alloc(u8, 2 + catalog.name_width + 1 + 10 + 1);
    defer alloc.free(indent);
    @memset(indent, ' ');
    const role_row = try std.fmt.allocPrint(alloc, "    mtp{s}mismatch   unsloth/Qwen3.8-27B-GGUF/MTP/mtp-Qwen3.8-27B-Q4_0.gguf", .{indent[0 .. catalog.name_width - 2 - 3 + 1]});
    defer alloc.free(role_row);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), role_row) != null);
    const detail = try std.fmt.allocPrint(alloc, "\n{s}  16464440224 bytes  qwen35 UD-Q4_K_M  commit 4ca720788d1e", .{indent});
    defer alloc.free(detail);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), detail) != null);
    const big = try std.fmt.allocPrint(alloc, "\n{s}registered as big\n", .{indent});
    defer alloc.free(big);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), big) != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "not loaded yet: the vision unit") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "no sidecar") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "main     0123456789ab  322e194f") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "2 GGUF file(s) outside") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "\n    registered as repo (profile gemma4 forced)\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "registry entries without a file") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "  gone unsloth/Gone-GGUF/gone.gguf\n") != null);
    out.clearRetainingCapacity();
    try listing.render(&out.writer, true, .none);
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, out.written(), .{});
    defer parsed.deinit();
    const rows = parsed.value.object.get("catalog").?.array.items;
    try std.testing.expectEqualStrings("present", rows[0].object.get("status").?.string);
    try std.testing.expectEqualStrings("mismatch", rows[0].object.get("companions").?.array.items[1].object.get("status").?.string);
    const files = parsed.value.object.get("other").?.array.items;
    try std.testing.expectEqual(@as(usize, 2), files.len);
    try std.testing.expect(files[0].object.get("sidecar").? == .null);
    try std.testing.expectEqualStrings("main", files[1].object.get("sidecar").?.object.get("role").?.string);
    try std.testing.expectEqual(@as(i64, 2), parsed.value.object.get("outside_layout").?.integer);
    try std.testing.expectEqualStrings("repo", files[1].object.get("registered").?.object.get("name").?.string);
    try std.testing.expectEqualStrings("gone", parsed.value.object.get("missing").?.array.items[0].object.get("name").?.string);
}

/// Test double for `readRange`: serves a directory image in `chunk`-sized
/// pieces so the assembly loop is exercised, and nothing past it (the
/// weights a real file would have).
const StubRange = struct {
    gpa: Allocator,
    bytes: []const u8,
    chunk: usize,
    requests: usize = 0,
    fn read(self: *StubRange, start: u64, count: usize) ![]u8 {
        self.requests += 1;
        const from: usize = @intCast(start);
        if (from >= self.bytes.len) return error.InvalidRange;
        const n = @min(count, @min(self.chunk, self.bytes.len - from));
        return self.gpa.dupe(u8, self.bytes[from .. from + n]);
    }
};

const Mutation = struct {
    architecture: ?[]const u8 = null,
    encoding: ?struct { tensor: []const u8, id: u32 } = null,
};

fn writeGgufString(w: *std.Io.Writer, text: []const u8) !void {
    try w.writeInt(u64, text.len, .little);
    try w.writeAll(text);
}

/// Serializes a hydrated directory (the adapter's inventory fixture) back
/// into GGUF wire format, with an optional edit, so the inspection path is
/// tested on the real tensor set without a model download. The vocabulary
/// array keeps its count with empty strings (the binding checks the count).
fn serializeDirectory(alloc: Allocator, doc: inference.gguf.Document, mutation: Mutation) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(alloc);
    errdefer out.deinit();
    const w = &out.writer;
    try w.writeAll("GGUF");
    try w.writeInt(u32, 3, .little);
    try w.writeInt(u64, doc.tensors.len, .little);
    try w.writeInt(u64, doc.metadata.len, .little);
    for (doc.metadata) |m| {
        try writeGgufString(w, m.key);
        try w.writeInt(u32, @intFromEnum(m.kind), .little);
        switch (m.value) {
            .unsigned => |n| try w.writeInt(u32, @intCast(n), .little),
            .signed => |n| try w.writeInt(i32, @intCast(n), .little),
            .float => |f| try w.writeInt(u32, @bitCast(@as(f32, @floatCast(f))), .little),
            .boolean => |b| try w.writeInt(u8, @intFromBool(b), .little),
            .string => |text| try writeGgufString(w, if (mutation.architecture != null and std.mem.eql(u8, m.key, "general.architecture")) mutation.architecture.? else text),
            .array => |array| {
                try w.writeInt(u32, @intFromEnum(array.element_type), .little);
                try w.writeInt(u64, array.count, .little);
                if (array.values) |values| {
                    for (values) |value| switch (array.element_type) {
                        .int32 => try w.writeInt(i32, @intCast(value.signed), .little),
                        .uint32 => try w.writeInt(u32, @intCast(value.unsigned), .little),
                        else => return error.UnsupportedFixtureValue,
                    };
                } else if (array.element_type == .string) {
                    // The vocabulary: its count is validated, its strings are
                    // not read here, so empty ones keep the count honest.
                    for (0..@intCast(array.count)) |_| try w.writeInt(u64, 0, .little);
                } else return error.UnsupportedFixtureValue;
            },
        }
    }
    for (doc.tensors) |tensor| {
        try writeGgufString(w, tensor.name);
        try w.writeInt(u32, @intCast(tensor.dimensions.len), .little);
        for (tensor.dimensions) |dim| try w.writeInt(u64, dim, .little);
        const id = if (mutation.encoding != null and std.mem.eql(u8, tensor.name, mutation.encoding.?.tensor)) mutation.encoding.?.id else tensor.encoding_id;
        try w.writeInt(u32, id, .little);
        try w.writeInt(u64, tensor.offset, .little);
    }
    return out.toOwnedSlice();
}

fn inspectFixture(arena: Allocator, gpa: Allocator, mutation: Mutation, sha256: []const u8, chunk: usize) !InspectReport {
    var inventory = try inference.models.qwen35.inventoryDocument(gpa);
    defer inventory.deinit();
    const directory = try serializeDirectory(arena, inventory, mutation);
    var stub: StubRange = .{ .gpa = gpa, .bytes = directory, .chunk = chunk };
    var diag: config.Diagnostic = .{};
    const entry = catalog.find("qwen3.8-27b").?;
    return inspectDirectory(arena, gpa, &stub, .{
        .name = null,
        .repo = entry.repo,
        .file = entry.file,
        .requested_revision = "main",
        .revision = entry.revision,
        // The weights would follow the aligned directory.
        .size = std.mem.alignForward(u64, directory.len, 32) + inventory.file_bytes,
        .sha256 = sha256,
    }, &diag);
}

test "inspect judges the pinned directory supported, runnable, or not, from the head of the file" {
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const entry = catalog.find("qwen3.8-27b").?;
    // The real inventory with the pinned digest: supported, assembled from
    // several windows, nothing read past the directory.
    const supported = try inspectFixture(arena, gpa, .{}, entry.sha256, 256 * 1024);
    if (supported.verdict.status != .supported) std.debug.print("verdict: {s}\n", .{supported.verdict.reason});
    try std.testing.expectEqual(.supported, supported.verdict.status);
    try std.testing.expectEqualStrings("qwen3.8-27b", supported.verdict.catalog.?);
    try std.testing.expectEqual(Role.main, supported.verdict.role.?);
    try std.testing.expect(supported.requests > 1);
    try std.testing.expect(supported.bytes_read >= supported.directory.?.directory_bytes and supported.bytes_read < supported.directory.?.directory_bytes + 256 * 1024);
    try std.testing.expectEqualStrings("qwen35", supported.directory.?.architecture.?);
    try std.testing.expectEqual(@as(usize, 866), supported.directory.?.tensor_count);
    // The fixture's size is what its offsets imply, not the artifact's.
    try std.testing.expectEqual(supported.directory.?.data_offset + supported.directory.?.tensor_bytes, supported.size);
    // Another digest of the same file: runnable, not supported.
    const runnable = try inspectFixture(arena, gpa, .{}, "0000", 1 << 20);
    try std.testing.expectEqual(.runnable, runnable.verdict.status);
    try std.testing.expect(runnable.verdict.catalog == null);
    try std.testing.expect(std.mem.indexOf(u8, runnable.verdict.reason, "pins another digest") != null);
    try std.testing.expect(runnable.requests < supported.requests);
    // A storable encoding outside the adapter's executable set (Q4_0, the
    // same 144 bytes per 256 elements as Q4_K, so the offsets still hold).
    const stored = try inspectFixture(arena, gpa, .{ .encoding = .{ .tensor = "token_embd.weight", .id = 2 } }, entry.sha256, 1 << 20);
    try std.testing.expectEqual(.not_runnable, stored.verdict.status);
    try std.testing.expectEqualStrings("token_embd.weight", stored.verdict.tensor.?);
    try std.testing.expectEqualStrings("Q4_0 (id 2)", stored.verdict.encoding.?);
    try std.testing.expect(std.mem.indexOf(u8, stored.verdict.reason, "outside the qwen35 adapter's executable set") != null);
    try std.testing.expect(stored.directory != null);
    // An encoding nuclis does not store: rejected by the parser, named anyway.
    const unknown = try inspectFixture(arena, gpa, .{ .encoding = .{ .tensor = "blk.0.ffn_up.weight", .id = 16 } }, entry.sha256, 1 << 20);
    try std.testing.expectEqual(.not_runnable, unknown.verdict.status);
    try std.testing.expectEqualStrings("blk.0.ffn_up.weight", unknown.verdict.tensor.?);
    try std.testing.expectEqualStrings("id 16", unknown.verdict.encoding.?);
    try std.testing.expect(std.mem.indexOf(u8, unknown.verdict.reason, "does not store") != null);
    try std.testing.expect(unknown.directory == null);
    // An architecture without an adapter.
    const foreign = try inspectFixture(arena, gpa, .{ .architecture = "gemma3" }, entry.sha256, 1 << 20);
    try std.testing.expectEqual(.not_runnable, foreign.verdict.status);
    try std.testing.expectEqualStrings("no adapter for architecture \"gemma3\"", foreign.verdict.reason);
    try std.testing.expect(foreign.verdict.tensor == null);
    // An architecture with an adapter that rejects this directory (the
    // Gemma adapter over the Qwen tensor list).
    const rejected = try inspectFixture(arena, gpa, .{ .architecture = "gemma4" }, entry.sha256, 1 << 20);
    try std.testing.expectEqual(.not_runnable, rejected.verdict.status);
    try std.testing.expect(std.mem.startsWith(u8, rejected.verdict.reason, "the gemma4 adapter rejects the file: "));
    // Both renderings carry the verdict.
    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    try renderInspect(&out.writer, supported, false, .none);
    try std.testing.expect(std.mem.startsWith(u8, out.written(), "unsloth/Qwen3.8-27B-GGUF @ main: commit 4ca720788d1e01f1bff70c033e0d0028fd02e502\nQwen3.8-27B-UD-Q4_K_M.gguf: "));
    try std.testing.expect(std.mem.indexOf(u8, out.written(), " bytes, sha256 322e194f") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "\nArchitecture: \"qwen35\"\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "\nVerdict: supported: catalogue entry qwen3.8-27b") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "no weights downloaded") != null);
    out.clearRetainingCapacity();
    try renderInspect(&out.writer, stored, false, .none);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "\nVerdict: not runnable: tensor token_embd.weight uses encoding Q4_0 (id 2)") != null);
    out.clearRetainingCapacity();
    try renderInspect(&out.writer, unknown, true, .none);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, out.written(), .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings("not_runnable", parsed.value.object.get("verdict").?.object.get("status").?.string);
    try std.testing.expectEqualStrings("blk.0.ffn_up.weight", parsed.value.object.get("verdict").?.object.get("tensor").?.string);
    try std.testing.expect(parsed.value.object.get("directory").? == .null);
    try std.testing.expect(parsed.value.object.get("requests").?.integer >= 1);
}

test "registry entries translate into repository pulls with their companions" {
    var diag: config.Diagnostic = .{};
    const gemma: config.ModelEntry = .{ .repo = "unsloth/gemma-4-12b-it-GGUF", .file = "g.gguf", .revision = "fc034cfff751157913579611efad8462ac1be606", .mmproj = "mmproj-F16.gguf" };
    var with: std.EnumSet(Role) = .initEmpty();
    with.insert(.mmproj);
    const pull_options = try fromRegistry(.{ .repo = "gemma", .with = with, .force = true }, &gemma, &diag);
    try std.testing.expectEqualStrings("gemma", pull_options.name.?);
    try std.testing.expectEqualStrings("unsloth/gemma-4-12b-it-GGUF", pull_options.repo);
    try std.testing.expectEqualStrings("g.gguf", pull_options.file.?);
    try std.testing.expectEqualStrings("fc034cfff751157913579611efad8462ac1be606", pull_options.revision.?);
    try std.testing.expectEqualStrings("mmproj-F16.gguf", pull_options.mmproj.?);
    try std.testing.expect(pull_options.mtp == null and pull_options.force and !pull_options.all and pull_options.with.count() == 0);
    const all = try fromRegistry(.{ .repo = "gemma", .all = true }, &gemma, &diag);
    try std.testing.expect(all.mmproj != null and all.mtp == null and all.revision != null);
    const plain = try fromRegistry(.{ .repo = "gemma" }, &.{ .repo = "a/b", .file = "f.gguf" }, &diag);
    try std.testing.expect(plain.mmproj == null and plain.revision == null);
    with = .initEmpty();
    with.insert(.mtp);
    try std.testing.expectError(error.NoSuchCompanion, fromRegistry(.{ .repo = "gemma", .with = with }, &gemma, &diag));
    try std.testing.expect(std.mem.indexOf(u8, diag.message(), "no mtp companion") != null);
    try std.testing.expectError(error.NotPullable, fromRegistry(.{ .repo = "local" }, &.{ .path = "/scratch/x.gguf" }, &diag));
    try std.testing.expectError(error.ConflictingOptions, fromRegistry(.{ .repo = "gemma", .file = "other.gguf" }, &gemma, &diag));
    try std.testing.expect(isRegistryName("gemma") and !isRegistryName("qwen3.8-27b") and !isRegistryName("a/b") and !isRegistryName(""));
}
